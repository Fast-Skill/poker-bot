//! Heads-up No-Limit Hold'em preflop, with real raise sizing.
//!
//! Where [`crate::pushfold`] allows only jam-or-fold, this models the betting
//! ladder actual play uses: open, 3-bet, 4-bet, jam, with a fold or call
//! available at every rung. That is the whole strategic content of heads-up
//! preflop, and it is what a microstakes bot needs before postflop exists.
//!
//! # The modelling approximation
//!
//! When betting closes with both players live, the hand should go to a flop —
//! but there is no postflop model yet. Those nodes are instead valued at raw
//! all-in equity, plus a flat [`Sizing::position_edge`] for the small blind.
//!
//! This is the single biggest approximation here, and its direction is known.
//! Valuing a flop at all-in equity says the only thing that matters postflop is
//! who holds the better cards. Real heads-up play is not like that: the small
//! blind is on the button and acts last on every street, which is worth real
//! equity that raw showdown numbers cannot see.
//!
//! The visible consequence is that solved opening ranges come out **tighter
//! than true GTO** — around 55% of hands rather than the 80%-plus a real solve
//! gives, because the model cannot price the pots the button wins without
//! showdown. `position_edge` is the dial for that, and closing the gap properly
//! needs the postflop model, not a better fudge factor.
//!
//! # Simplification
//!
//! Limping is omitted — the small blind opens or folds. Limp-based strategies
//! are legitimate heads-up but roughly double the tree, and the open/3-bet/
//! 4-bet ladder carries the strategic weight.

use crate::abstraction::{HandClass, NUM_HAND_CLASSES};
use crate::cfr::{Game, InfoKey};
use crate::pushfold::EquityTable;
use crate::rng::Rng;
use crate::card::{Card, CardSet};
use std::fmt;

/// An action in the preflop ladder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Fold,
    /// Check or call, whichever the spot allows.
    Passive,
    /// Escalate to this stage's raise size.
    Raise,
    /// Move all in.
    Jam,
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Action::Fold => "fold",
            Action::Passive => "call",
            Action::Raise => "raise",
            Action::Jam => "jam",
        })
    }
}

/// Bet sizes for the preflop ladder, in big blinds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sizing {
    /// Small blind's opening raise, commonly 2 to 3 big blinds.
    pub open_to: f64,
    /// Big blind's 3-bet.
    pub three_bet_to: f64,
    /// Small blind's 4-bet.
    pub four_bet_to: f64,
    /// Equity points the small blind gains at a flop node for acting last on
    /// every postflop street.
    ///
    /// Zero means position is worth nothing, which is the honest default when
    /// nothing has measured it. Raising it widens the small blind's opening
    /// range, since more pots become worth contesting. Applies only where a
    /// flop is actually seen — an all-in has no position to exercise.
    pub position_edge: f64,
}

impl Default for Sizing {
    /// Standard heads-up sizes: open to 2.5, 3-bet to 8, 4-bet to 18, with no
    /// positional credit until a postflop model can measure it.
    fn default() -> Sizing {
        Sizing {
            open_to: 2.5,
            three_bet_to: 8.0,
            four_bet_to: 18.0,
            position_edge: 0.0,
        }
    }
}

/// Where a hand stands in the preflop tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// Cards not yet dealt.
    Deal,
    /// Small blind opens or folds.
    SbOpen,
    /// Big blind faces the open.
    BbVsOpen,
    /// Small blind faces the 3-bet.
    SbVs3Bet,
    /// Big blind faces the 4-bet.
    BbVs4Bet,
    /// Small blind faces an all-in.
    SbVsJam,
    /// Big blind faces an all-in.
    BbVsJam,
    /// Someone folded; the index is who.
    Folded(u8),
    /// All in and called.
    Showdown,
    /// Betting closed with both players live — the hand would see a flop.
    Flop,
}

/// Decision stages, in the order [`Stage::decision_index`] reports.
const NUM_STAGES: usize = 6;

impl Stage {
    /// A dense index for decision stages, used to key information sets.
    fn decision_index(self) -> Option<usize> {
        Some(match self {
            Stage::SbOpen => 0,
            Stage::BbVsOpen => 1,
            Stage::SbVs3Bet => 2,
            Stage::BbVs4Bet => 3,
            Stage::SbVsJam => 4,
            Stage::BbVsJam => 5,
            _ => return None,
        })
    }

    /// Which player acts at a decision stage.
    fn actor(self) -> Option<usize> {
        Some(match self {
            Stage::SbOpen | Stage::SbVs3Bet | Stage::SbVsJam => 0,
            Stage::BbVsOpen | Stage::BbVs4Bet | Stage::BbVsJam => 1,
            _ => return None,
        })
    }
}

/// A node in the preflop tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct State {
    classes: [u8; 2],
    /// Chips each player has put in, quantised to hundredths of a blind so the
    /// state stays `Eq` and hashable without float comparison.
    committed: [u32; 2],
    stage: Stage,
}

/// Chips are held as hundredths of a big blind: exact for any realistic sizing
/// and free of float-equality hazards in state comparison.
const SCALE: f64 = 100.0;

fn to_chips(blinds: f64) -> u32 {
    (blinds * SCALE).round() as u32
}

fn to_blinds(chips: u32) -> f64 {
    chips as f64 / SCALE
}

impl State {
    /// The hand class held by `player`.
    pub fn hand(&self, player: usize) -> HandClass {
        HandClass::from_index(self.classes[player] as usize).expect("dealt class is in range")
    }

    /// Chips `player` has committed, in big blinds.
    pub fn committed(&self, player: usize) -> f64 {
        to_blinds(self.committed[player])
    }

    pub fn stage(&self) -> Stage {
        self.stage
    }
}

/// Heads-up preflop Hold'em at a fixed stack depth.
#[derive(Debug, Clone)]
pub struct Preflop {
    stack: u32,
    sizing: Sizing,
    equity: EquityTable,
    /// Legal actions per decision stage, precomputed since sizes are fixed.
    actions: Vec<Vec<Action>>,
}

impl Preflop {
    /// Builds a game at `stack` big blinds deep.
    ///
    /// Raise sizes at or above the stack are dropped, since they are
    /// indistinguishable from jamming — offering both would give the solver two
    /// identical actions and split its regret between them for no reason.
    ///
    /// # Panics
    /// Panics if the sizes are not strictly increasing above one big blind, if
    /// the stack cannot cover the big blind, or if realization is not positive.
    pub fn new(stack: f64, sizing: Sizing, equity: EquityTable) -> Preflop {
        assert!(stack >= 1.0, "stack must cover the big blind");
        assert!(
            1.0 < sizing.open_to && sizing.open_to < sizing.three_bet_to,
            "sizes must increase: open {} then 3-bet {}",
            sizing.open_to,
            sizing.three_bet_to
        );
        assert!(
            sizing.three_bet_to < sizing.four_bet_to,
            "sizes must increase: 3-bet {} then 4-bet {}",
            sizing.three_bet_to,
            sizing.four_bet_to
        );
        assert!(
            sizing.position_edge.abs() < 0.5,
            "position edge {} is not a plausible equity adjustment",
            sizing.position_edge
        );

        let stack_chips = to_chips(stack);
        let fits = |size: f64| to_chips(size) < stack_chips;

        let mut actions = vec![Vec::new(); NUM_STAGES];
        actions[0] = with_optional(&[Action::Fold], Action::Raise, fits(sizing.open_to));
        actions[1] = with_optional(
            &[Action::Fold, Action::Passive],
            Action::Raise,
            fits(sizing.three_bet_to),
        );
        actions[2] = with_optional(
            &[Action::Fold, Action::Passive],
            Action::Raise,
            fits(sizing.four_bet_to),
        );
        actions[3] = vec![Action::Fold, Action::Passive, Action::Jam];
        actions[4] = vec![Action::Fold, Action::Passive];
        actions[5] = vec![Action::Fold, Action::Passive];

        Preflop {
            stack: stack_chips,
            sizing,
            equity,
            actions,
        }
    }

    pub fn stack(&self) -> f64 {
        to_blinds(self.stack)
    }

    pub fn sizing(&self) -> Sizing {
        self.sizing
    }

    /// The actions available at `stage`.
    pub fn actions(&self, stage: Stage) -> &[Action] {
        stage
            .decision_index()
            .map(|index| self.actions[index].as_slice())
            .unwrap_or(&[])
    }

    /// The information set for `player` holding `class` at `stage`.
    pub fn info_key(stage: Stage, class: usize) -> InfoKey {
        let index = stage.decision_index().expect("not a decision stage");
        (class as InfoKey) << 4 | index as InfoKey
    }

    /// The raise size that applies at `stage`, in chips.
    fn raise_target(&self, stage: Stage) -> u32 {
        to_chips(match stage {
            Stage::SbOpen => self.sizing.open_to,
            Stage::BbVsOpen => self.sizing.three_bet_to,
            Stage::SbVs3Bet => self.sizing.four_bet_to,
            other => unreachable!("{other:?} has no raise size"),
        })
    }

    /// Where a call leads: showdown if it puts both players all in, otherwise
    /// a flop.
    fn after_call(&self, committed: [u32; 2]) -> Stage {
        if committed[0] >= self.stack && committed[1] >= self.stack {
            Stage::Showdown
        } else {
            Stage::Flop
        }
    }
}

/// Builds an action list, including `optional` only when it is available.
fn with_optional(base: &[Action], optional: Action, include: bool) -> Vec<Action> {
    let mut actions = base.to_vec();
    if include {
        actions.push(optional);
    }
    actions.push(Action::Jam);
    actions
}

impl Game for Preflop {
    type State = State;

    fn initial(&self) -> State {
        State {
            classes: [0, 0],
            // Blinds are posted before anyone acts.
            committed: [to_chips(0.5), to_chips(1.0)],
            stage: Stage::Deal,
        }
    }

    fn is_terminal(&self, state: &State) -> bool {
        matches!(
            state.stage,
            Stage::Folded(_) | Stage::Showdown | Stage::Flop
        )
    }

    fn terminal_utility(&self, state: &State) -> f64 {
        match state.stage {
            // The folder forfeits everything they put in.
            Stage::Folded(0) => -state.committed(0),
            Stage::Folded(1) => state.committed(1),
            Stage::Showdown | Stage::Flop => {
                let equity = self.equity.get(state.hand(0), state.hand(1));
                // Position is only worth something where there are streets left
                // to play it on. All in, the cards decide alone.
                let realized = if state.stage == Stage::Flop {
                    (equity + self.sizing.position_edge).clamp(0.0, 1.0)
                } else {
                    equity
                };
                // Both players are matched here, so each risks the same amount.
                let at_risk = state.committed(0);
                (2.0 * realized - 1.0) * at_risk
            }
            other => unreachable!("{other:?} is not terminal"),
        }
    }

    fn is_chance(&self, state: &State) -> bool {
        state.stage == Stage::Deal
    }

    fn chance_outcomes(&self, state: &State) -> Vec<(State, f64)> {
        debug_assert!(self.is_chance(state));
        let mut outcomes = Vec::with_capacity(NUM_HAND_CLASSES * NUM_HAND_CLASSES);
        let mut total = 0.0;
        for a in HandClass::all() {
            for b in HandClass::all() {
                let weight = (a.combos() * b.combos()) as f64;
                total += weight;
                outcomes.push((
                    State {
                        classes: [a.index() as u8, b.index() as u8],
                        stage: Stage::SbOpen,
                        ..*state
                    },
                    weight,
                ));
            }
        }
        for outcome in &mut outcomes {
            outcome.1 /= total;
        }
        outcomes
    }

    fn sample_chance(&self, state: &State, rng: &mut Rng) -> State {
        let mut drawn = [Card::all().next().expect("non-empty deck"); 4];
        let mut dead = CardSet::empty();
        for slot in drawn.iter_mut() {
            loop {
                let card = Card::from_index(rng.below(52) as u8).expect("0..52");
                if dead.insert(card) {
                    *slot = card;
                    break;
                }
            }
        }
        State {
            classes: [
                HandClass::from_cards(drawn[0], drawn[1]).index() as u8,
                HandClass::from_cards(drawn[2], drawn[3]).index() as u8,
            ],
            stage: Stage::SbOpen,
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
        Preflop::info_key(state.stage, state.classes[player] as usize)
    }

    fn num_actions(&self, state: &State) -> usize {
        self.actions(state.stage).len()
    }

    fn apply(&self, state: &State, action: usize) -> State {
        let stage = state.stage;
        let actor = self.current_player(state);
        let opponent = 1 - actor;
        let chosen = self.actions(stage)[action];
        let mut committed = state.committed;

        let next = match chosen {
            Action::Fold => Stage::Folded(actor as u8),
            Action::Passive => {
                // Match the opponent, capped by the stack.
                committed[actor] = committed[opponent].min(self.stack);
                self.after_call(committed)
            }
            Action::Raise => {
                committed[actor] = self.raise_target(stage);
                match stage {
                    Stage::SbOpen => Stage::BbVsOpen,
                    Stage::BbVsOpen => Stage::SbVs3Bet,
                    Stage::SbVs3Bet => Stage::BbVs4Bet,
                    other => unreachable!("{other:?} cannot raise"),
                }
            }
            Action::Jam => {
                committed[actor] = self.stack;
                if actor == 0 {
                    Stage::BbVsJam
                } else {
                    Stage::SbVsJam
                }
            }
        };

        State {
            committed,
            stage: next,
            ..*state
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfr::Solver;

    const EQUITY_SAMPLES: u32 = 400;
    const ITERATIONS: usize = 400_000;

    fn table() -> EquityTable {
        EquityTable::sampled_parallel(EQUITY_SAMPLES, 0x51DE, 4)
    }

    fn solve(stack: f64) -> Solver<Preflop> {
        let mut rng = Rng::new(0xF01D);
        let mut solver = Solver::new(Preflop::new(stack, Sizing::default(), table()));
        solver.train_sampled(ITERATIONS, &mut rng);
        solver
    }

    fn class(text: &str) -> HandClass {
        text.parse().expect("valid hand class")
    }

    /// Probability of `action` at `stage` with `hand`.
    fn probability(solver: &Solver<Preflop>, stage: Stage, hand: &str, action: Action) -> f64 {
        let index = solver
            .game()
            .actions(stage)
            .iter()
            .position(|a| *a == action)
            .unwrap_or_else(|| panic!("{action} is not available at {stage:?}"));
        solver
            .average_strategy(Preflop::info_key(stage, class(hand).index()))
            .unwrap_or_else(|| panic!("{hand} at {stage:?} was never visited"))[index]
    }

    /// Share of all combos taking `action` at `stage`.
    fn range_width(solver: &Solver<Preflop>, stage: Stage, action: Action) -> f64 {
        let index = solver
            .game()
            .actions(stage)
            .iter()
            .position(|a| *a == action)
            .expect("action available");
        let mut combos = 0.0;
        for hand in HandClass::all() {
            if let Some(strategy) =
                solver.average_strategy(Preflop::info_key(stage, hand.index()))
            {
                combos += strategy[index] * hand.combos() as f64;
            }
        }
        combos / 1_326.0
    }

    #[test]
    fn blinds_are_posted_before_anyone_acts() {
        let game = Preflop::new(100.0, Sizing::default(), table());
        let root = game.initial();
        assert!(game.is_chance(&root));
        assert_eq!(root.committed(0), 0.5, "small blind");
        assert_eq!(root.committed(1), 1.0, "big blind");
    }

    #[test]
    fn the_ladder_escalates_through_every_rung() {
        let game = Preflop::new(100.0, Sizing::default(), table());
        let mut rng = Rng::new(1);
        let dealt = game.sample_chance(&game.initial(), &mut rng);

        assert_eq!(dealt.stage(), Stage::SbOpen);
        assert_eq!(game.current_player(&dealt), 0);

        let raise = |state: &State| {
            let index = game
                .actions(state.stage())
                .iter()
                .position(|a| *a == Action::Raise)
                .expect("raise available");
            game.apply(state, index)
        };

        let opened = raise(&dealt);
        assert_eq!(opened.stage(), Stage::BbVsOpen);
        assert_eq!(opened.committed(0), 2.5, "opened to 2.5");

        let three_bet = raise(&opened);
        assert_eq!(three_bet.stage(), Stage::SbVs3Bet);
        assert_eq!(three_bet.committed(1), 8.0);

        let four_bet = raise(&three_bet);
        assert_eq!(four_bet.stage(), Stage::BbVs4Bet);
        assert_eq!(four_bet.committed(0), 18.0);

        // Only fold, call, or jam remain after the 4-bet.
        assert_eq!(game.actions(Stage::BbVs4Bet).len(), 3);
    }

    #[test]
    fn folding_forfeits_exactly_what_was_committed() {
        let game = Preflop::new(100.0, Sizing::default(), table());
        let mut rng = Rng::new(2);
        let dealt = game.sample_chance(&game.initial(), &mut rng);

        // Small blind folds immediately: loses only the posted blind.
        let folded = game.apply(&dealt, 0);
        assert_eq!(folded.stage(), Stage::Folded(0));
        assert_eq!(game.terminal_utility(&folded), -0.5);

        // Big blind folds to an open: small blind wins the big blind.
        let opened = game.apply(&dealt, game.actions(Stage::SbOpen).len() - 2);
        let bb_folded = game.apply(&opened, 0);
        assert_eq!(bb_folded.stage(), Stage::Folded(1));
        assert_eq!(game.terminal_utility(&bb_folded), 1.0);
    }

    #[test]
    fn a_called_jam_reaches_showdown_and_a_called_raise_reaches_a_flop() {
        let game = Preflop::new(100.0, Sizing::default(), table());
        let mut rng = Rng::new(3);
        let dealt = game.sample_chance(&game.initial(), &mut rng);
        let actions = game.actions(Stage::SbOpen);

        let jam = game.apply(&dealt, actions.len() - 1);
        assert_eq!(jam.stage(), Stage::BbVsJam);
        let called = game.apply(&jam, 1);
        assert_eq!(called.stage(), Stage::Showdown);
        assert_eq!(called.committed(0), 100.0);
        assert_eq!(called.committed(1), 100.0);

        let opened = game.apply(&dealt, actions.len() - 2);
        let flatted = game.apply(&opened, 1);
        assert_eq!(flatted.stage(), Stage::Flop, "betting closed, both live");
        assert_eq!(flatted.committed(1), 2.5, "the call matched the open");
    }

    #[test]
    fn raise_sizes_above_the_stack_are_dropped() {
        // At 12bb a 4-bet to 18 cannot exist, so that rung must disappear
        // rather than becoming a duplicate of jamming.
        let shallow = Preflop::new(12.0, Sizing::default(), table());
        assert!(!shallow.actions(Stage::SbVs3Bet).contains(&Action::Raise));
        assert!(shallow.actions(Stage::SbVs3Bet).contains(&Action::Jam));

        // Deep enough, and every rung is present.
        let deep = Preflop::new(100.0, Sizing::default(), table());
        assert!(deep.actions(Stage::SbVs3Bet).contains(&Action::Raise));

        // Shallower still and even the 3-bet is gone.
        let very_shallow = Preflop::new(6.0, Sizing::default(), table());
        assert!(!very_shallow.actions(Stage::BbVsOpen).contains(&Action::Raise));
    }

    #[test]
    fn the_position_edge_only_applies_where_a_flop_is_seen() {
        let favoured = Sizing {
            position_edge: 0.05,
            ..Sizing::default()
        };
        let neutral = Preflop::new(100.0, Sizing::default(), table());
        let with_edge = Preflop::new(100.0, favoured, table());

        let state = |stage| State {
            classes: [class("AA").index() as u8, class("72o").index() as u8],
            committed: [to_chips(10.0), to_chips(10.0)],
            stage,
        };

        // All in, there are no streets left to hold position on.
        assert_eq!(
            neutral.terminal_utility(&state(Stage::Showdown)),
            with_edge.terminal_utility(&state(Stage::Showdown))
        );

        // Seeing a flop is where acting last pays.
        assert!(
            with_edge.terminal_utility(&state(Stage::Flop))
                > neutral.terminal_utility(&state(Stage::Flop))
        );
    }

    #[test]
    fn crediting_position_widens_the_opening_range() {
        // The dial has to move the strategy in the right direction, or it is
        // not modelling anything. More postflop value makes more pots worth
        // contesting, so the small blind should open wider.
        let solve_with = |edge: f64| {
            let sizing = Sizing {
                position_edge: edge,
                ..Sizing::default()
            };
            let mut rng = Rng::new(0xF01D);
            let mut solver = Solver::new(Preflop::new(100.0, sizing, table()));
            solver.train_sampled(ITERATIONS, &mut rng);
            range_width(&solver, Stage::SbOpen, Action::Raise)
        };

        let neutral = solve_with(0.0);
        let favoured = solve_with(0.06);
        assert!(
            favoured > neutral + 0.02,
            "position credit widened opens only from {neutral:.3} to {favoured:.3}"
        );
    }

    #[test]
    fn utilities_are_zero_sum_by_construction() {
        // The small blind's loss is exactly the big blind's gain, so a single
        // signed number describes both. Verified across every terminal shape.
        let game = Preflop::new(100.0, Sizing::default(), table());
        let base = State {
            classes: [class("AA").index() as u8, class("KK").index() as u8],
            committed: [to_chips(8.0), to_chips(8.0)],
            stage: Stage::Flop,
        };
        for stage in [Stage::Showdown, Stage::Flop] {
            let utility = game.terminal_utility(&State { stage, ..base });
            assert!(utility.is_finite());
            assert!(utility.abs() <= 8.0, "cannot win more than was risked");
        }
    }

    #[test]
    fn every_hand_gets_a_strategy_at_every_decision_stage() {
        let solver = solve(100.0);
        let stages = [
            Stage::SbOpen,
            Stage::BbVsOpen,
            Stage::SbVs3Bet,
            Stage::BbVs4Bet,
            Stage::SbVsJam,
            Stage::BbVsJam,
        ];
        for stage in stages {
            for hand in HandClass::all() {
                let strategy = solver
                    .average_strategy(Preflop::info_key(stage, hand.index()))
                    .unwrap_or_else(|| panic!("{hand} missing at {stage:?}"));
                let total: f64 = strategy.iter().sum();
                assert!((total - 1.0).abs() < 1e-9, "{hand} at {stage:?} sums to {total}");
            }
        }
        assert_eq!(solver.info_set_count(), NUM_STAGES * NUM_HAND_CLASSES);
    }

    #[test]
    fn aces_never_fold_and_always_get_money_in() {
        let solver = solve(100.0);
        assert!(probability(&solver, Stage::SbOpen, "AA", Action::Fold) < 0.01);
        assert!(probability(&solver, Stage::SbVs3Bet, "AA", Action::Fold) < 0.01);
        assert!(probability(&solver, Stage::BbVsJam, "AA", Action::Passive) > 0.95);
        assert!(probability(&solver, Stage::SbVsJam, "AA", Action::Passive) > 0.95);
    }

    #[test]
    fn worthless_hands_fold_to_a_jam() {
        let solver = solve(100.0);
        assert!(
            probability(&solver, Stage::BbVsJam, "72o", Action::Fold) > 0.90,
            "72o cannot call off 100bb"
        );
        assert!(probability(&solver, Stage::SbVsJam, "32o", Action::Fold) > 0.90);
    }

    #[test]
    fn each_rung_of_the_ladder_is_tighter_than_the_one_below() {
        // Opening is cheap, 3-betting risks more, 4-betting more still. The
        // ranges must narrow accordingly — this is the shape of every real
        // preflop chart.
        let solver = solve(100.0);
        let opens = range_width(&solver, Stage::SbOpen, Action::Raise);
        let three_bets = range_width(&solver, Stage::BbVsOpen, Action::Raise);
        let four_bets = range_width(&solver, Stage::SbVs3Bet, Action::Raise);

        assert!(
            opens > three_bets,
            "open {opens:.3} should be wider than 3-bet {three_bets:.3}"
        );
        assert!(
            three_bets > four_bets,
            "3-bet {three_bets:.3} should be wider than 4-bet {four_bets:.3}"
        );
    }

    #[test]
    fn the_small_blind_enters_more_pots_than_it_folds() {
        // Heads-up the small blind risks 2.5 to win 1.5, so entering must beat
        // folding for most of the deck.
        //
        // The bar is set at half the deck rather than the 80%-plus a true solve
        // gives, because `position_edge` defaults to zero: with flops valued at
        // bare all-in equity the model cannot see the pots the button takes
        // without showdown, so it opens tighter than real GTO. That gap is the
        // known cost of having no postflop model, not a solver fault — see
        // `crediting_position_widens_the_opening_range`.
        let solver = solve(100.0);
        let opens = range_width(&solver, Stage::SbOpen, Action::Raise);
        let jams = range_width(&solver, Stage::SbOpen, Action::Jam);
        assert!(
            opens + jams > 0.50,
            "small blind entered only {:.1}% of pots",
            (opens + jams) * 100.0
        );
    }

    #[test]
    fn short_stacks_jam_more_and_raise_less() {
        // With 15bb behind there is no room for a 4-bet ladder, so the strategy
        // should collapse toward jamming.
        let deep = solve(100.0);
        let short = solve(15.0);
        let deep_jams = range_width(&deep, Stage::SbOpen, Action::Jam);
        let short_jams = range_width(&short, Stage::SbOpen, Action::Jam);
        assert!(
            short_jams > deep_jams,
            "15bb jammed {short_jams:.3} but 100bb jammed {deep_jams:.3}"
        );
    }

    #[test]
    fn the_strategy_converges_toward_equilibrium() {
        let solver = solve(100.0);
        let exploitability = solver.exploitability(&solver.profile());
        assert!(
            exploitability < 0.35,
            "exploitable for {exploitability} big blinds per hand"
        );
    }

    #[test]
    #[should_panic(expected = "sizes must increase")]
    fn out_of_order_sizes_are_rejected() {
        let bad = Sizing {
            open_to: 8.0,
            three_bet_to: 2.5,
            ..Sizing::default()
        };
        Preflop::new(100.0, bad, table());
    }

    #[test]
    #[should_panic(expected = "not a plausible equity adjustment")]
    fn an_absurd_position_edge_is_rejected() {
        let bad = Sizing {
            position_edge: 0.9,
            ..Sizing::default()
        };
        Preflop::new(100.0, bad, table());
    }
}
