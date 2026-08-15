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
//! Both are closed-form, so this module is to postflop what [`crate::kuhn`] is
//! to the solver core: a game whose answer is known in advance, used to prove
//! the machinery before it is pointed at spots nobody can check by hand.
//!
//! # Betting tree
//!
//! Either player may bet, and the player facing a bet may fold, call, or raise.
//! A raise closes the action — the tree stops at one raise rather than allowing
//! an unbounded re-raise war, which is where the strategic content lives
//! without the tree exploding.
//!
//! # Information sets
//!
//! A player's information set is their holding *and the betting so far*,
//! because bet sizes are public. Facing a half-pot bet is a different decision
//! from facing a two-thirds-pot bet, and collapsing them would average two
//! unrelated spots into one strategy.
//!
//! # Simplification
//!
//! Ranges are independent — card removal between the two players is not
//! modelled. That affects absolute equities slightly but not the frequencies
//! above, which depend only on pot odds.

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
    /// Out of position faces a raise of their own bet.
    OopVsRaise,
    /// In position faces a check-raise.
    IpVsRaise,
    /// Someone folded; the index is who.
    Folded(u8),
    /// Cards are turned over.
    Showdown,
}

/// Decision stages. Must stay within the bit field in [`River::info_key`].
const NUM_STAGES: usize = 6;
/// Bits reserved for the stage, the facing bet, and the facing raise.
const STAGE_BITS: u32 = 3;
const SIZE_BITS: u32 = 3;
/// Most distinct bet or raise sizes an information set can encode.
pub const MAX_SIZES: usize = 1 << SIZE_BITS;

// Checked at compile time rather than in debug runs: adding a seventh stage
// without widening the field would silently alias two unrelated decisions into
// one information set, and the solver would average them together. That is a
// bug worth failing the build over.
const _: () = assert!(
    NUM_STAGES <= 1 << STAGE_BITS,
    "STAGE_BITS is too narrow for NUM_STAGES"
);

impl Stage {
    fn decision_index(self) -> Option<usize> {
        Some(match self {
            Stage::OopFirst => 0,
            Stage::IpVsCheck => 1,
            Stage::OopVsBet => 2,
            Stage::IpVsBet => 3,
            Stage::OopVsRaise => 4,
            Stage::IpVsRaise => 5,
            _ => return None,
        })
    }

    fn actor(self) -> Option<usize> {
        Some(match self {
            Stage::OopFirst | Stage::OopVsBet | Stage::OopVsRaise => 0,
            Stage::IpVsCheck | Stage::IpVsBet | Stage::IpVsRaise => 1,
            _ => return None,
        })
    }

    /// Whether the actor here is facing a wager and so may fold, call, or raise.
    fn faces_bet(self) -> bool {
        matches!(self, Stage::OopVsBet | Stage::IpVsBet)
    }

    /// Whether the actor here is facing a raise, which closes the action.
    fn faces_raise(self) -> bool {
        matches!(self, Stage::OopVsRaise | Stage::IpVsRaise)
    }
}

/// A node in the river tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct State {
    hands: [u16; 2],
    stage: Stage,
    /// Chips each player has put in this street, in hundredths of a blind.
    committed: [u32; 2],
    /// Which bet size is in play, and which raise size, as indices. Both are
    /// public knowledge and therefore part of an information set.
    bet: u8,
    raise: u8,
}

impl State {
    /// Index into `player`'s range.
    pub fn holding(&self, player: usize) -> usize {
        self.hands[player] as usize
    }

    pub fn stage(&self) -> Stage {
        self.stage
    }

    /// Chips `player` has committed this street, in big blinds.
    pub fn committed(&self, player: usize) -> f64 {
        to_blinds(self.committed[player])
    }
}

const SCALE: f64 = 100.0;

fn to_chips(blinds: f64) -> u32 {
    (blinds * SCALE).round() as u32
}

fn to_blinds(chips: u32) -> f64 {
    chips as f64 / SCALE
}

/// A river spot: two ranges, a pot, and the bet and raise sizes on offer.
#[derive(Debug, Clone)]
pub struct River {
    pot: u32,
    stack: u32,
    /// Bet amounts in chips, ascending and de-duplicated.
    bets: Vec<u32>,
    /// Raise targets per bet size: `raises[bet_index]` lists the total a raiser
    /// commits, given the bet they are facing.
    raises: Vec<Vec<u32>>,
    ranges: [Vec<Holding>; 2],
    /// Normalised weights, parallel to `ranges`.
    weights: [Vec<f64>; 2],
}

impl River {
    /// Index of the passive action — check when nothing is owed, fold when
    /// facing a wager. Always first.
    pub const PASSIVE: usize = 0;
    /// Index of the call action at a stage facing a wager.
    pub const CALL: usize = 1;

    /// Index of the `n`th bet size where betting is open.
    pub const fn bet_action(n: usize) -> usize {
        n + 1
    }

    /// Index of the `n`th raise size where raising is available.
    pub const fn raise_action(n: usize) -> usize {
        n + 2
    }

    /// Builds a spot.
    ///
    /// `bet_fractions` are multiples of the pot. `raise_fractions` are multiples
    /// of the pot *after* the raiser calls, matching how a "pot-sized raise" is
    /// normally described. Pass an empty slice to forbid raising.
    ///
    /// Sizes are clamped to the stack and de-duplicated, so a short stack does
    /// not offer the same all-in amount several times. A raise that cannot
    /// exceed the bet it faces is dropped rather than offered as a pseudo-call.
    ///
    /// # Panics
    /// Panics if either range is empty, if any weight is not positive, if the
    /// pot or stack is not positive, if no bet size survives clamping, or if
    /// more than [`MAX_SIZES`] distinct sizes are requested.
    pub fn new(
        pot: f64,
        stack: f64,
        bet_fractions: &[f64],
        raise_fractions: &[f64],
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

        let pot_chips = to_chips(pot);
        let stack_chips = to_chips(stack);

        let mut bets: Vec<u32> = bet_fractions
            .iter()
            .map(|fraction| {
                assert!(*fraction > 0.0, "bet fractions must be positive");
                to_chips(pot * fraction).clamp(1, stack_chips)
            })
            .collect();
        bets.sort_unstable();
        bets.dedup();
        assert!(!bets.is_empty(), "at least one bet size is required");
        assert!(
            bets.len() <= MAX_SIZES,
            "at most {MAX_SIZES} distinct bet sizes are supported"
        );

        // A raise is priced off the pot that calling would create.
        let raises: Vec<Vec<u32>> = bets
            .iter()
            .map(|&facing| {
                let after_call = pot_chips + 2 * facing;
                let mut targets: Vec<u32> = raise_fractions
                    .iter()
                    .map(|fraction| {
                        assert!(*fraction > 0.0, "raise fractions must be positive");
                        let extra = (after_call as f64 * fraction).round() as u32;
                        facing.saturating_add(extra).min(stack_chips)
                    })
                    // Anything that fails to exceed the bet is a call, not a raise.
                    .filter(|target| *target > facing)
                    .collect();
                targets.sort_unstable();
                targets.dedup();
                assert!(
                    targets.len() <= MAX_SIZES,
                    "at most {MAX_SIZES} distinct raise sizes are supported"
                );
                targets
            })
            .collect();

        River {
            pot: pot_chips,
            stack: stack_chips,
            bets,
            raises,
            ranges,
            weights,
        }
    }

    /// A spot with no raising, for comparison against closed-form theory.
    pub fn without_raises(
        pot: f64,
        stack: f64,
        bet_fractions: &[f64],
        oop: Vec<Holding>,
        ip: Vec<Holding>,
    ) -> River {
        River::new(pot, stack, bet_fractions, &[], oop, ip)
    }

    /// The pot at the start of the street.
    pub fn pot(&self) -> f64 {
        to_blinds(self.pot)
    }

    /// The effective stack behind.
    pub fn stack(&self) -> f64 {
        to_blinds(self.stack)
    }

    /// Available bet sizes, in big blinds.
    pub fn bet_sizes(&self) -> Vec<f64> {
        self.bets.iter().map(|b| to_blinds(*b)).collect()
    }

    /// Total commitments available to a player raising the `bet`th bet size.
    pub fn raise_sizes(&self, bet: usize) -> Vec<f64> {
        self.raises[bet].iter().map(|r| to_blinds(*r)).collect()
    }

    /// A player's range.
    pub fn range(&self, player: usize) -> &[Holding] {
        &self.ranges[player]
    }

    /// The information set for a `holding` at `stage`, facing the given bet and
    /// raise sizes.
    ///
    /// Bet sizes are public, so they belong in the key. Packing them into fixed
    /// bit fields keeps lookup free; the width is asserted so that adding a
    /// stage or a size cannot silently alias two spots into one strategy.
    pub fn info_key(stage: Stage, holding: usize, bet: usize, raise: usize) -> InfoKey {
        let index = stage.decision_index().expect("not a decision stage");
        debug_assert!(
            bet < MAX_SIZES && raise < MAX_SIZES,
            "size {bet}/{raise} does not fit in {SIZE_BITS} bits"
        );

        let mut key = index as InfoKey;
        key |= (bet as InfoKey) << STAGE_BITS;
        key |= (raise as InfoKey) << (STAGE_BITS + SIZE_BITS);
        key | (holding as InfoKey) << (STAGE_BITS + 2 * SIZE_BITS)
    }
}

impl Game for River {
    type State = State;

    fn initial(&self) -> State {
        State {
            hands: [0, 0],
            stage: Stage::Deal,
            committed: [0, 0],
            bet: 0,
            raise: 0,
        }
    }

    fn is_terminal(&self, state: &State) -> bool {
        matches!(state.stage, Stage::Folded(_) | Stage::Showdown)
    }

    fn terminal_utility(&self, state: &State) -> f64 {
        // Each player owns half the starting pot. Folding forfeits that half
        // *plus* anything already wagered this street — which is why betting
        // and then folding to a raise costs more than folding outright.
        let half_pot = to_blinds(self.pot) / 2.0;
        match state.stage {
            Stage::Folded(0) => -(half_pot + state.committed(0)),
            Stage::Folded(1) => half_pot + state.committed(1),
            Stage::Showdown => {
                debug_assert_eq!(
                    state.committed[0], state.committed[1],
                    "a showdown means the wager was matched"
                );
                let at_risk = half_pot + state.committed(0);
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
                        ..*state
                    },
                    oop_weight * ip_weight,
                ));
            }
        }
        outcomes
    }

    fn sample_chance(&self, state: &State, rng: &mut Rng) -> State {
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
            ..*state
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
        River::info_key(
            state.stage,
            state.holding(player),
            state.bet as usize,
            state.raise as usize,
        )
    }

    fn num_actions(&self, state: &State) -> usize {
        match state.stage {
            // Check, or any bet size.
            Stage::OopFirst | Stage::IpVsCheck => 1 + self.bets.len(),
            // Fold, call, or any raise available against this bet size.
            stage if stage.faces_bet() => 2 + self.raises[state.bet as usize].len(),
            // A raise closes the action: fold or call only.
            stage if stage.faces_raise() => 2,
            other => unreachable!("{other:?} is not a decision stage"),
        }
    }

    fn apply(&self, state: &State, action: usize) -> State {
        let actor = self.current_player(state);
        let opponent = 1 - actor;
        let mut next = *state;

        match state.stage {
            Stage::OopFirst | Stage::IpVsCheck => {
                if action == River::PASSIVE {
                    next.stage = if state.stage == Stage::OopFirst {
                        Stage::IpVsCheck
                    } else {
                        Stage::Showdown
                    };
                } else {
                    let bet = action - 1;
                    next.bet = bet as u8;
                    next.committed[actor] = self.bets[bet];
                    next.stage = if state.stage == Stage::OopFirst {
                        Stage::IpVsBet
                    } else {
                        Stage::OopVsBet
                    };
                }
            }
            stage if stage.faces_bet() => {
                if action == River::PASSIVE {
                    next.stage = Stage::Folded(actor as u8);
                } else if action == River::CALL {
                    next.committed[actor] = state.committed[opponent];
                    next.stage = Stage::Showdown;
                } else {
                    let raise = action - 2;
                    next.raise = raise as u8;
                    next.committed[actor] = self.raises[state.bet as usize][raise];
                    next.stage = if stage == Stage::IpVsBet {
                        Stage::OopVsRaise
                    } else {
                        Stage::IpVsRaise
                    };
                }
            }
            stage if stage.faces_raise() => {
                if action == River::PASSIVE {
                    next.stage = Stage::Folded(actor as u8);
                } else {
                    next.committed[actor] = state.committed[opponent];
                    next.stage = Stage::Showdown;
                }
            }
            other => unreachable!("{other:?} is not a decision stage"),
        }

        next
    }
}

impl fmt::Display for River {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "river: pot {:.2}, stack {:.2}, bets {:?}, ranges {}x{}",
            self.pot(),
            self.stack(),
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

    const NUTS: usize = 0;
    const AIR: usize = 1;
    const BLUFF_CATCHER: usize = 0;

    /// The clairvoyance game: the bettor holds either the nuts or air in equal
    /// measure, the caller holds a pure bluff-catcher that beats air and loses
    /// to the nuts. Its equilibrium is known in closed form — but only without
    /// raising, so raises are off here.
    fn clairvoyant(bet_fraction: f64) -> River {
        River::without_raises(
            1.0,
            100.0,
            &[bet_fraction],
            vec![Holding::new(1_000, 0.5), Holding::new(0, 0.5)],
            vec![Holding::new(500, 1.0)],
        )
    }

    fn solve(spot: River) -> Solver<River> {
        let mut solver = Solver::new(spot);
        solver.train(20_000);
        solver
    }

    fn strategy(solver: &Solver<River>, stage: Stage, holding: usize) -> Vec<f64> {
        solver
            .average_strategy(River::info_key(stage, holding, 0, 0))
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
        assert_eq!(spot.num_actions(&bet), 2, "fold or call, raising is off");

        let checked = spot.apply(&dealt, River::PASSIVE);
        assert_eq!(checked.stage(), Stage::IpVsCheck);
        assert_eq!(spot.apply(&checked, River::PASSIVE).stage(), Stage::Showdown);
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
    fn betting_and_then_folding_to_a_raise_costs_the_bet_as_well() {
        // The case a naive "folding loses half the pot" rule gets wrong. Bet 1
        // into a pot of 1, get raised, fold: the half pot *and* the bet are gone.
        let spot = River::new(
            1.0,
            100.0,
            &[1.0],
            &[1.0],
            vec![Holding::new(1_000, 0.5), Holding::new(0, 0.5)],
            vec![Holding::new(500, 1.0)],
        );
        let mut rng = Rng::new(4);
        let dealt = spot.sample_chance(&spot.initial(), &mut rng);

        let bet = spot.apply(&dealt, River::bet_action(0));
        assert_eq!(bet.committed(0), 1.0);

        let raised = spot.apply(&bet, River::raise_action(0));
        assert_eq!(raised.stage(), Stage::OopVsRaise);
        assert!(raised.committed(1) > 1.0, "a raise must exceed the bet");

        let folded = spot.apply(&raised, River::PASSIVE);
        assert_eq!(folded.stage(), Stage::Folded(0));
        assert_eq!(
            spot.terminal_utility(&folded),
            -1.5,
            "half the pot plus the surrendered bet"
        );
    }

    #[test]
    fn a_called_bet_puts_the_bet_at_risk() {
        let spot = clairvoyant(1.0);
        let state = State {
            hands: [NUTS as u16, BLUFF_CATCHER as u16],
            stage: Stage::Showdown,
            committed: [to_chips(1.0), to_chips(1.0)],
            bet: 0,
            raise: 0,
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
        let spot = River::without_raises(
            2.0,
            100.0,
            &[1.0],
            vec![Holding::new(500, 1.0)],
            vec![Holding::new(500, 1.0)],
        );
        let state = State {
            hands: [0, 0],
            stage: Stage::Showdown,
            committed: [0, 0],
            bet: 0,
            raise: 0,
        };
        assert_eq!(spot.terminal_utility(&state), 0.0);
    }

    #[test]
    fn the_nuts_always_bet() {
        let solver = solve(clairvoyant(1.0));
        let nuts = strategy(&solver, Stage::OopFirst, NUTS);
        assert!(nuts[River::bet_action(0)] > 0.95, "the nuts must bet: {nuts:?}");
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
    fn raises_are_priced_off_the_pot_after_calling() {
        // Pot 1, bet 1. Calling makes the pot 3, so a pot-sized raise commits
        // the 1 called plus 3 more.
        let spot = River::new(
            1.0,
            100.0,
            &[1.0],
            &[1.0],
            vec![Holding::new(1, 1.0)],
            vec![Holding::new(1, 1.0)],
        );
        assert_eq!(spot.raise_sizes(0), vec![4.0]);

        // A half-pot raise commits 1 plus 1.5.
        let half = River::new(
            1.0,
            100.0,
            &[1.0],
            &[0.5],
            vec![Holding::new(1, 1.0)],
            vec![Holding::new(1, 1.0)],
        );
        assert_eq!(half.raise_sizes(0), vec![2.5]);
    }

    #[test]
    fn raises_that_cannot_exceed_the_bet_are_dropped() {
        // With only 1 chip behind, a "raise" over a pot-sized bet is impossible.
        let spot = River::new(
            1.0,
            1.0,
            &[1.0],
            &[1.0],
            vec![Holding::new(1, 1.0)],
            vec![Holding::new(1, 1.0)],
        );
        assert!(spot.raise_sizes(0).is_empty(), "no legal raise exists");

        let mut rng = Rng::new(5);
        let dealt = spot.sample_chance(&spot.initial(), &mut rng);
        let bet = spot.apply(&dealt, River::bet_action(0));
        assert_eq!(spot.num_actions(&bet), 2, "fold or call only");
    }

    #[test]
    fn the_nuts_check_raise_when_raising_is_available() {
        // Given a raise, a polarised range should sometimes check the nuts and
        // raise instead of always betting — that is what a check-raise is for.
        let spot = River::new(
            1.0,
            100.0,
            &[0.75],
            &[1.0],
            vec![Holding::new(1_000, 0.5), Holding::new(0, 0.5)],
            // The in-position player bets a range, so there is something to raise.
            vec![Holding::new(900, 0.5), Holding::new(100, 0.5)],
        );
        let mut solver = Solver::new(spot);
        solver.train(40_000);

        let oop_vs_bet = solver
            .average_strategy(River::info_key(Stage::OopVsBet, NUTS, 0, 0))
            .expect("visited");
        assert_eq!(oop_vs_bet.len(), 3, "fold, call, raise");
        assert!(
            oop_vs_bet[River::raise_action(0)] > 0.05,
            "the nuts should check-raise at least sometimes: {oop_vs_bet:?}"
        );
        assert!(oop_vs_bet[River::PASSIVE] < 0.05, "the nuts never fold");
    }

    #[test]
    fn a_raise_closes_the_action() {
        let spot = River::new(
            1.0,
            100.0,
            &[1.0],
            &[1.0],
            vec![Holding::new(1, 1.0)],
            vec![Holding::new(1, 1.0)],
        );
        let mut rng = Rng::new(6);
        let dealt = spot.sample_chance(&spot.initial(), &mut rng);
        let raised = spot.apply(&spot.apply(&dealt, River::bet_action(0)), River::raise_action(0));
        assert_eq!(spot.num_actions(&raised), 2, "no re-raise is offered");
        assert_eq!(
            spot.apply(&raised, River::CALL).stage(),
            Stage::Showdown
        );
    }

    #[test]
    fn information_sets_encode_the_bet_size_they_face() {
        // A player facing a small bet is in a different spot from one facing a
        // large bet. Collapsing them would average two unrelated decisions.
        let small = River::info_key(Stage::IpVsBet, 7, 0, 0);
        let large = River::info_key(Stage::IpVsBet, 7, 1, 0);
        assert_ne!(small, large, "bet size must be part of the key");

        let mut seen = std::collections::HashSet::new();
        let stages = [
            Stage::OopFirst,
            Stage::IpVsCheck,
            Stage::OopVsBet,
            Stage::IpVsBet,
            Stage::OopVsRaise,
            Stage::IpVsRaise,
        ];
        for stage in stages {
            for holding in 0..20 {
                for bet in 0..MAX_SIZES {
                    for raise in 0..MAX_SIZES {
                        assert!(
                            seen.insert(River::info_key(stage, holding, bet, raise)),
                            "{stage:?} h{holding} b{bet} r{raise} collided"
                        );
                    }
                }
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
        let spot = River::without_raises(
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
        let spot = River::without_raises(
            10.0,
            100.0,
            &[0.33, 0.5, 1.0],
            vec![Holding::new(1, 1.0)],
            vec![Holding::new(1, 1.0)],
        );
        assert_eq!(spot.bet_sizes(), vec![3.3, 5.0, 10.0]);

        // A short stack collapses every size onto the all-in amount.
        let shallow = River::without_raises(
            10.0,
            2.0,
            &[0.33, 0.5, 1.0],
            vec![Holding::new(1, 1.0)],
            vec![Holding::new(1, 1.0)],
        );
        assert_eq!(shallow.bet_sizes(), vec![2.0], "one distinct size remains");
    }

    #[test]
    #[should_panic(expected = "both ranges must be non-empty")]
    fn an_empty_range_is_rejected() {
        River::without_raises(1.0, 100.0, &[1.0], vec![], vec![Holding::new(1, 1.0)]);
    }

    #[test]
    #[should_panic(expected = "pot must be positive")]
    fn a_zero_pot_is_rejected() {
        River::without_raises(
            0.0,
            100.0,
            &[1.0],
            vec![Holding::new(1, 1.0)],
            vec![Holding::new(1, 1.0)],
        );
    }

    #[test]
    #[should_panic(expected = "every weight must be positive")]
    fn a_zero_weight_holding_is_rejected() {
        River::without_raises(
            1.0,
            100.0,
            &[1.0],
            vec![Holding::new(1, 1.0), Holding::new(2, 0.0)],
            vec![Holding::new(1, 1.0)],
        );
    }

    #[test]
    #[should_panic(expected = "distinct bet sizes")]
    fn too_many_bet_sizes_are_rejected() {
        let fractions: Vec<f64> = (1..=MAX_SIZES + 1).map(|n| n as f64 * 0.1).collect();
        River::without_raises(
            10.0,
            100.0,
            &fractions,
            vec![Holding::new(1, 1.0)],
            vec![Holding::new(1, 1.0)],
        );
    }
}
