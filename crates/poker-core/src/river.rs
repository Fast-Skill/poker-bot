//! River subgames: one betting round with the board complete.
//!
//! The river is where postflop theory is exact. With no cards to come, a hand
//! is simply better or worse than another, and the equilibrium betting and
//! calling frequencies follow from pot odds alone:
//!
//! - **Minimum defense frequency.** Facing a bet of `s` times the pot, the
//!   defender must continue with `1 / (1 + s)` of their range. Defend less and
//!   any two cards bluff profitably.
//! - **Optimal bluff ratio.** Of the hands that bet `s` times the pot,
//!   `s / (1 + 2s)` should be bluffs — a 2:1 value-to-bluff ratio at a pot-sized
//!   bet. Bluff more and the caller profits by calling; bluff less and they
//!   profit by folding.
//!
//! Both are closed-form, so this module is to postflop what
//! [`crate::kuhn`] is to the solver core: a game whose answer is known in
//! advance, used to prove the machinery before it is pointed at spots where
//! nobody can check the result by hand.
//!
//! # Simplifications
//!
//! Ranges are independent — card removal between the two players is not
//! modelled — and betting is capped at one bet and one call, with no raises.
//! Both are lifted later; neither affects the frequencies above.

use crate::cfr::{Game, InfoKey};
use crate::rng::Rng;
use std::fmt;

/// One holding in a player's range.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Holding {
    /// Showdown strength. Higher beats lower; equal chops. Any monotone scale
    /// works, so [`crate::eval::HandRank`] bits drop straight in.
    pub strength: u32,
    /// How much of the range this holding is. Weights are normalised, so they
    /// need not sum to one.
    pub weight: f64,
}

impl Holding {
    pub const fn new(strength: u32, weight: f64) -> Holding {
        Holding { strength, weight }
    }
}

/// Where a hand stands in the river tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// Hands not yet dealt.
    Deal,
    /// Out of position acts first.
    OopFirst,
    /// In position acts after a check.
    IpVsCheck,
    /// Out of position faces a bet after checking.
    OopVsBet,
    /// In position faces a bet.
    IpVsBet,
    /// Someone folded; the index is who.
    Folded(u8),
    /// Cards are turned over.
    Showdown,
}

const NUM_STAGES: usize = 4;

impl Stage {
    fn decision_index(self) -> Option<usize> {
        Some(match self {
            Stage::OopFirst => 0,
            Stage::IpVsCheck => 1,
            Stage::OopVsBet => 2,
            Stage::IpVsBet => 3,
            _ => return None,
        })
    }

    fn actor(self) -> Option<usize> {
        Some(match self {
            Stage::OopFirst | Stage::OopVsBet => 0,
            Stage::IpVsCheck | Stage::IpVsBet => 1,
            _ => return None,
        })
    }
}

/// A node in the river tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct State {
    hands: [u16; 2],
    stage: Stage,
    /// Chips each player has added this street, in hundredths of a blind.
    /// Equal at showdown, since an unmatched bet never reaches one.
    extra: u32,
}

impl State {
    /// Index into `player`'s range.
    pub fn holding(&self, player: usize) -> usize {
        self.hands[player] as usize
    }

    pub fn stage(&self) -> Stage {
        self.stage
    }
}

const SCALE: f64 = 100.0;

fn to_chips(blinds: f64) -> u32 {
    (blinds * SCALE).round() as u32
}

fn to_blinds(chips: u32) -> f64 {
    chips as f64 / SCALE
}

/// A river spot: two ranges, a pot, and a set of bet sizes.
#[derive(Debug, Clone)]
pub struct River {
    pot: u32,
    stack: u32,
    /// Bet amounts in chips, ascending and de-duplicated.
    bets: Vec<u32>,
    ranges: [Vec<Holding>; 2],
    /// Normalised weights, parallel to `ranges`.
    weights: [Vec<f64>; 2],
}

impl River {
    /// Builds a spot.
    ///
    /// `bet_fractions` are multiples of the pot. Sizes above the stack are
    /// clamped to it and then de-duplicated, so a short stack does not offer
    /// the same all-in amount several times over.
    ///
    /// # Panics
    /// Panics if either range is empty, if any weight is not positive, if the
    /// pot is not positive, or if no bet size survives clamping.
    pub fn new(
        pot: f64,
        stack: f64,
        bet_fractions: &[f64],
        oop: Vec<Holding>,
        ip: Vec<Holding>,
    ) -> River {
        assert!(pot > 0.0, "pot must be positive");
        assert!(stack > 0.0, "stack must be positive");
        assert!(!oop.is_empty() && !ip.is_empty(), "both ranges must be non-empty");

        let ranges = [oop, ip];
        let weights = std::array::from_fn(|player| {
            let range: &Vec<Holding> = &ranges[player];
            let total: f64 = range.iter().map(|h| h.weight).sum();
            assert!(total > 0.0, "range weights must be positive");
            for holding in range {
                assert!(holding.weight > 0.0, "every weight must be positive");
            }
            range.iter().map(|h| h.weight / total).collect()
        });

        let stack_chips = to_chips(stack);
        let mut bets: Vec<u32> = bet_fractions
            .iter()
            .map(|fraction| {
                assert!(*fraction > 0.0, "bet fractions must be positive");
                to_chips(pot * fraction).min(stack_chips).max(1)
            })
            .collect();
        bets.sort_unstable();
        bets.dedup();
        assert!(!bets.is_empty(), "at least one bet size is required");

        River {
            pot: to_chips(pot),
            stack: stack_chips,
            bets,
            ranges,
            weights,
        }
    }

    /// The pot at the start of the street.
    pub fn pot(&self) -> f64 {
        to_blinds(self.pot)
    }

    /// Available bet sizes, in big blinds.
    pub fn bet_sizes(&self) -> Vec<f64> {
        self.bets.iter().map(|b| to_blinds(*b)).collect()
    }

    /// A player's range.
    pub fn range(&self, player: usize) -> &[Holding] {
        &self.ranges[player]
    }

    /// The information set for `player`'s `holding` at `stage`.
    ///
    /// The stage occupies the low two bits, which is only sound while there are
    /// at most four decision stages — hence the assertion. Adding a fifth
    /// without widening the shift would silently alias information sets
    /// together, and the solver would average two unrelated decisions into one
    /// strategy.
    pub fn info_key(stage: Stage, holding: usize) -> InfoKey {
        let index = stage.decision_index().expect("not a decision stage");
        debug_assert!(
            NUM_STAGES <= 4 && index < NUM_STAGES,
            "stage index {index} does not fit in two bits"
        );
        (holding as InfoKey) << 2 | index as InfoKey
    }

    /// Index of the check or fold action, which is always first.
    pub const PASSIVE: usize = 0;

    /// Index of the call action at a stage facing a bet.
    pub const CALL: usize = 1;

    /// Index of the `n`th bet size at a stage where betting is allowed.
    pub const fn bet_action(n: usize) -> usize {
        n + 1
    }
}

impl Game for River {
    type State = State;

    fn initial(&self) -> State {
        State {
            hands: [0, 0],
            stage: Stage::Deal,
            extra: 0,
        }
    }

    fn is_terminal(&self, state: &State) -> bool {
        matches!(state.stage, Stage::Folded(_) | Stage::Showdown)
    }

    fn terminal_utility(&self, state: &State) -> f64 {
        // Each player owns half the starting pot, so folding surrenders that
        // half and winning claims the opponent's.
        let half_pot = to_blinds(self.pot) / 2.0;
        match state.stage {
            Stage::Folded(0) => -half_pot,
            Stage::Folded(1) => half_pot,
            Stage::Showdown => {
                let at_risk = half_pot + to_blinds(state.extra);
                let oop = self.ranges[0][state.holding(0)].strength;
                let ip = self.ranges[1][state.holding(1)].strength;
                match oop.cmp(&ip) {
                    std::cmp::Ordering::Greater => at_risk,
                    std::cmp::Ordering::Less => -at_risk,
                    std::cmp::Ordering::Equal => 0.0,
                }
            }
            other => unreachable!("{other:?} is not terminal"),
        }
    }

    fn is_chance(&self, state: &State) -> bool {
        state.stage == Stage::Deal
    }

    fn chance_outcomes(&self, state: &State) -> Vec<(State, f64)> {
        debug_assert!(self.is_chance(state));
        let mut outcomes = Vec::with_capacity(self.ranges[0].len() * self.ranges[1].len());
        for (oop, oop_weight) in self.weights[0].iter().enumerate() {
            for (ip, ip_weight) in self.weights[1].iter().enumerate() {
                outcomes.push((
                    State {
                        hands: [oop as u16, ip as u16],
                        stage: Stage::OopFirst,
                        extra: 0,
                    },
                    oop_weight * ip_weight,
                ));
            }
        }
        outcomes
    }

    fn sample_chance(&self, _state: &State, rng: &mut Rng) -> State {
        let draw = |player: usize, rng: &mut Rng| {
            let roll = rng.next_f64();
            let mut cumulative = 0.0;
            for (index, weight) in self.weights[player].iter().enumerate() {
                cumulative += weight;
                if roll < cumulative {
                    return index as u16;
                }
            }
            (self.weights[player].len() - 1) as u16
        };
        State {
            hands: [draw(0, rng), draw(1, rng)],
            stage: Stage::OopFirst,
            extra: 0,
        }
    }

    fn current_player(&self, state: &State) -> usize {
        state
            .stage
            .actor()
            .unwrap_or_else(|| unreachable!("{:?} is not a decision stage", state.stage))
    }

    fn info_key(&self, state: &State) -> InfoKey {
        let player = self.current_player(state);
        River::info_key(state.stage, state.holding(player))
    }

    fn num_actions(&self, state: &State) -> usize {
        match state.stage {
            // Check, or any bet size.
            Stage::OopFirst | Stage::IpVsCheck => 1 + self.bets.len(),
            // Fold or call. Raises are out of scope here.
            Stage::OopVsBet | Stage::IpVsBet => 2,
            other => unreachable!("{other:?} is not a decision stage"),
        }
    }

    fn apply(&self, state: &State, action: usize) -> State {
        match state.stage {
            Stage::OopFirst => {
                if action == River::PASSIVE {
                    State {
                        stage: Stage::IpVsCheck,
                        ..*state
                    }
                } else {
                    State {
                        stage: Stage::IpVsBet,
                        extra: self.bets[action - 1],
                        ..*state
                    }
                }
            }
            Stage::IpVsCheck => {
                if action == River::PASSIVE {
                    State {
                        stage: Stage::Showdown,
                        ..*state
                    }
                } else {
                    State {
                        stage: Stage::OopVsBet,
                        extra: self.bets[action - 1],
                        ..*state
                    }
                }
            }
            Stage::OopVsBet | Stage::IpVsBet => {
                let folder = self.current_player(state) as u8;
                if action == River::PASSIVE {
                    // The unmatched bet never entered the pot, so it is not at
                    // risk for either player.
                    State {
                        stage: Stage::Folded(folder),
                        extra: 0,
                        ..*state
                    }
                } else {
                    State {
                        stage: Stage::Showdown,
                        ..*state
                    }
                }
            }
            other => unreachable!("{other:?} is not a decision stage"),
        }
    }
}

impl fmt::Display for River {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "river: pot {:.2}, stack {:.2}, bets {:?}, ranges {}x{}",
            to_blinds(self.pot),
            to_blinds(self.stack),
            self.bet_sizes(),
            self.ranges[0].len(),
            self.ranges[1].len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfr::Solver;

    const ITERATIONS: usize = 400_000;

    /// The clairvoyance game: the bettor holds either the nuts or air in equal
    /// measure, the caller holds a pure bluff-catcher that beats air and loses
    /// to the nuts. Its equilibrium is known in closed form.
    fn clairvoyant(bet_fraction: f64) -> River {
        River::new(
            1.0,
            100.0,
            &[bet_fraction],
            vec![Holding::new(1_000, 0.5), Holding::new(0, 0.5)],
            vec![Holding::new(500, 1.0)],
        )
    }

    const NUTS: usize = 0;
    const AIR: usize = 1;
    const BLUFF_CATCHER: usize = 0;

    fn solve(spot: River) -> Solver<River> {
        let mut solver = Solver::new(spot);
        solver.train(ITERATIONS.min(20_000));
        solver
    }

    fn strategy(solver: &Solver<River>, stage: Stage, holding: usize) -> Vec<f64> {
        solver
            .average_strategy(River::info_key(stage, holding))
            .unwrap_or_else(|| panic!("holding {holding} at {stage:?} was never visited"))
    }

    #[test]
    fn the_tree_has_the_expected_shape() {
        let spot = clairvoyant(1.0);
        let root = spot.initial();
        assert!(spot.is_chance(&root));
        assert_eq!(spot.chance_outcomes(&root).len(), 2, "two hands by one");

        let mut rng = Rng::new(1);
        let dealt = spot.sample_chance(&root, &mut rng);
        assert_eq!(dealt.stage(), Stage::OopFirst);
        assert_eq!(spot.num_actions(&dealt), 2, "check or one bet size");

        let bet = spot.apply(&dealt, River::bet_action(0));
        assert_eq!(bet.stage(), Stage::IpVsBet);
        assert_eq!(spot.num_actions(&bet), 2, "fold or call");

        let checked = spot.apply(&dealt, River::PASSIVE);
        assert_eq!(checked.stage(), Stage::IpVsCheck);
        assert_eq!(
            spot.apply(&checked, River::PASSIVE).stage(),
            Stage::Showdown
        );
    }

    #[test]
    fn folding_surrenders_exactly_half_the_pot() {
        let spot = clairvoyant(1.0);
        let mut rng = Rng::new(2);
        let dealt = spot.sample_chance(&spot.initial(), &mut rng);

        let bet = spot.apply(&dealt, River::bet_action(0));
        let folded = spot.apply(&bet, River::PASSIVE);
        assert_eq!(folded.stage(), Stage::Folded(1));
        assert_eq!(spot.terminal_utility(&folded), 0.5, "wins the other half");
    }

    #[test]
    fn an_uncalled_bet_is_not_at_risk() {
        // A bet that gets folded to never enters the pot, so the winner gains
        // exactly half the pot regardless of how large the bet was.
        for fraction in [0.5, 1.0, 4.0] {
            let spot = clairvoyant(fraction);
            let mut rng = Rng::new(3);
            let dealt = spot.sample_chance(&spot.initial(), &mut rng);
            let folded = spot.apply(&spot.apply(&dealt, River::bet_action(0)), River::PASSIVE);
            assert_eq!(spot.terminal_utility(&folded), 0.5, "at {fraction}x pot");
        }
    }

    #[test]
    fn a_called_bet_puts_the_bet_at_risk() {
        let spot = clairvoyant(1.0);
        let state = State {
            hands: [NUTS as u16, BLUFF_CATCHER as u16],
            stage: Stage::Showdown,
            extra: to_chips(1.0),
        };
        // Half the pot plus the called bet.
        assert_eq!(spot.terminal_utility(&state), 1.5);

        let losing = State {
            hands: [AIR as u16, BLUFF_CATCHER as u16],
            ..state
        };
        assert_eq!(spot.terminal_utility(&losing), -1.5);
    }

    #[test]
    fn equal_strength_hands_chop() {
        let spot = River::new(
            2.0,
            100.0,
            &[1.0],
            vec![Holding::new(500, 1.0)],
            vec![Holding::new(500, 1.0)],
        );
        let state = State {
            hands: [0, 0],
            stage: Stage::Showdown,
            extra: 0,
        };
        assert_eq!(spot.terminal_utility(&state), 0.0);
    }

    #[test]
    fn the_nuts_always_bet_and_air_never_calls() {
        let solver = solve(clairvoyant(1.0));
        let nuts = strategy(&solver, Stage::OopFirst, NUTS);
        assert!(
            nuts[River::bet_action(0)] > 0.95,
            "the nuts must bet: {nuts:?}"
        );
    }

    /// The headline result: of the hands that bet `s` times the pot,
    /// `s / (1 + 2s)` must be bluffs.
    #[test]
    fn the_bluff_ratio_matches_theory() {
        for fraction in [0.5, 1.0, 2.0] {
            let solver = solve(clairvoyant(fraction));
            let bet = River::bet_action(0);

            let nuts_bet = strategy(&solver, Stage::OopFirst, NUTS)[bet];
            let air_bet = strategy(&solver, Stage::OopFirst, AIR)[bet];

            // Both halves of the range are equally likely, so the bluff share
            // of betting hands is air / (nuts + air).
            let bluff_share = air_bet / (nuts_bet + air_bet);
            let expected = fraction / (1.0 + 2.0 * fraction);
            assert!(
                (bluff_share - expected).abs() < 0.03,
                "at {fraction}x pot, bluffs were {bluff_share:.4}, theory says {expected:.4}"
            );
        }
    }

    /// The other side of the same coin: the caller continues with
    /// `1 / (1 + s)` of their range, making bluffing exactly break even.
    #[test]
    fn the_defence_frequency_matches_theory() {
        for fraction in [0.5, 1.0, 2.0] {
            let solver = solve(clairvoyant(fraction));
            let calling = strategy(&solver, Stage::IpVsBet, BLUFF_CATCHER)[River::CALL];
            let expected = 1.0 / (1.0 + fraction);
            assert!(
                (calling - expected).abs() < 0.03,
                "at {fraction}x pot, defence was {calling:.4}, theory says {expected:.4}"
            );
        }
    }

    #[test]
    fn bigger_bets_are_defended_less_and_bluffed_more() {
        // The comparative statics, independent of the exact numbers above.
        let small = solve(clairvoyant(0.5));
        let large = solve(clairvoyant(2.0));

        let small_defence = strategy(&small, Stage::IpVsBet, BLUFF_CATCHER)[River::CALL];
        let large_defence = strategy(&large, Stage::IpVsBet, BLUFF_CATCHER)[River::CALL];
        assert!(
            small_defence > large_defence,
            "defence should fall as bets grow: {small_defence:.3} vs {large_defence:.3}"
        );

        let bet = River::bet_action(0);
        let small_bluff = strategy(&small, Stage::OopFirst, AIR)[bet];
        let large_bluff = strategy(&large, Stage::OopFirst, AIR)[bet];
        assert!(
            large_bluff > small_bluff,
            "bluffing should rise with bet size: {small_bluff:.3} vs {large_bluff:.3}"
        );
    }

    #[test]
    fn information_sets_do_not_collide_across_stages() {
        // Two holdings for the bettor across two stages, one for the caller
        // across two. A packing bug would show up here as a smaller count.
        let solver = solve(clairvoyant(1.0));
        assert_eq!(solver.info_set_count(), 2 * 2 + 2);

        // Distinct (stage, holding) pairs must map to distinct keys.
        let stages = [
            Stage::OopFirst,
            Stage::IpVsCheck,
            Stage::OopVsBet,
            Stage::IpVsBet,
        ];
        let mut seen = std::collections::HashSet::new();
        for stage in stages {
            for holding in 0..50 {
                assert!(
                    seen.insert(River::info_key(stage, holding)),
                    "{stage:?} holding {holding} collided"
                );
            }
        }
    }

    #[test]
    fn the_equilibrium_is_close_to_unexploitable() {
        let solver = solve(clairvoyant(1.0));
        let exploitability = solver.exploitability(&solver.profile());
        assert!(
            exploitability < 0.01,
            "exploitable for {exploitability} big blinds"
        );
    }

    #[test]
    fn a_range_with_no_bluffs_is_never_called() {
        // With only value in the betting range, the bluff-catcher has to fold.
        let spot = River::new(
            1.0,
            100.0,
            &[1.0],
            vec![Holding::new(1_000, 1.0)],
            vec![Holding::new(500, 1.0)],
        );
        let solver = solve(spot);
        let calling = strategy(&solver, Stage::IpVsBet, BLUFF_CATCHER)[River::CALL];
        assert!(calling < 0.05, "calling a pure value range: {calling:.3}");
    }

    #[test]
    fn multiple_bet_sizes_are_offered_and_deduplicated() {
        let spot = River::new(10.0, 100.0, &[0.33, 0.5, 1.0], vec![Holding::new(1, 1.0)], vec![Holding::new(1, 1.0)]);
        assert_eq!(spot.bet_sizes(), vec![3.3, 5.0, 10.0]);

        // A short stack collapses every size onto the all-in amount.
        let shallow = River::new(10.0, 2.0, &[0.33, 0.5, 1.0], vec![Holding::new(1, 1.0)], vec![Holding::new(1, 1.0)]);
        assert_eq!(shallow.bet_sizes(), vec![2.0], "one distinct size remains");
    }

    #[test]
    #[should_panic(expected = "both ranges must be non-empty")]
    fn an_empty_range_is_rejected() {
        River::new(1.0, 100.0, &[1.0], vec![], vec![Holding::new(1, 1.0)]);
    }

    #[test]
    #[should_panic(expected = "pot must be positive")]
    fn a_zero_pot_is_rejected() {
        River::new(0.0, 100.0, &[1.0], vec![Holding::new(1, 1.0)], vec![Holding::new(1, 1.0)]);
    }

    #[test]
    #[should_panic(expected = "every weight must be positive")]
    fn a_zero_weight_holding_is_rejected() {
        River::new(
            1.0,
            100.0,
            &[1.0],
            vec![Holding::new(1, 1.0), Holding::new(2, 0.0)],
            vec![Holding::new(1, 1.0)],
        );
    }
}
