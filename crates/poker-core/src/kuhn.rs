//! Kuhn poker: a three-card game used to verify the solver.
//!
//! Two players ante 1 and are dealt one card each from a three-card deck
//! (J, Q, K). There is a single betting round with one bet size. It is the
//! smallest game that still contains bluffing, value betting, and bluff-catching,
//! and — crucially — its Nash equilibrium is known analytically.
//!
//! # The known solution
//!
//! Player 0 has one free parameter `α ∈ [0, 1/3]`:
//!
//! - **J**: bet `α` (a pure bluff), fold when raised
//! - **Q**: always check; call a bet with probability `α + 1/3`
//! - **K**: bet `3α`, always call
//!
//! Player 1's strategy is *unique*, with no free parameter:
//!
//! - **J**: fold to a bet; bet 1/3 when checked to
//! - **Q**: call 1/3 of the time; check when checked to
//! - **K**: always bet and always call
//!
//! The game value to player 0 is exactly **-1/18**.
//!
//! Two properties make this a strong test. Player 1's uniqueness means the
//! solver must land on specific numbers, not merely something self-consistent.
//! And player 0's 3:1 value-to-bluff ratio holds for every valid `α`, so it can
//! be asserted without knowing which equilibrium CFR happens to select.

use crate::cfr::{Game, InfoKey};

/// Passive action: check when nothing is owed, fold when facing a bet.
pub const PASS: usize = 0;
/// Aggressive action: bet when nothing is owed, call when facing a bet.
pub const BET: usize = 1;

/// The jack, the weakest card.
pub const JACK: u8 = 0;
/// The queen.
pub const QUEEN: u8 = 1;
/// The king, the strongest card.
pub const KING: u8 = 2;

/// A node in the Kuhn game tree.
///
/// `Copy` and allocation-free: the solver clones this on every edge of every
/// traversal, and a heap allocation there dominates the actual arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct State {
    /// Each player's card. Meaningless until `dealt`.
    cards: [u8; 2],
    /// Actions so far, one per bit, earliest in the least significant bit.
    history: u8,
    /// How many actions have been taken.
    len: u8,
    dealt: bool,
}

impl State {
    /// The pre-deal root.
    pub const fn root() -> State {
        State {
            cards: [0, 0],
            history: 0,
            len: 0,
            dealt: false,
        }
    }

    /// The card held by `player`.
    pub const fn card(&self, player: usize) -> u8 {
        self.cards[player]
    }

    /// The action sequence so far, as a bitmask.
    pub const fn history(&self) -> u8 {
        self.history
    }

    pub const fn len(&self) -> u8 {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// The information set key for a player holding `card` after `len` actions
/// encoded as `history`.
///
/// Exposed so tests and analysis tools can address a specific decision without
/// reconstructing a state.
pub const fn info_key(card: u8, history: u8, len: u8) -> InfoKey {
    (card as InfoKey) << 8 | (len as InfoKey) << 4 | history as InfoKey
}

/// Kuhn poker as a [`Game`].
#[derive(Debug, Clone, Copy, Default)]
pub struct Kuhn;

impl Game for Kuhn {
    type State = State;

    fn initial(&self) -> State {
        State::root()
    }

    fn is_terminal(&self, state: &State) -> bool {
        match state.len {
            // check-bet leaves player 0 still to act; everything else is over.
            2 => state.history != 0b10,
            3 => true,
            _ => false,
        }
    }

    fn terminal_utility(&self, state: &State) -> f64 {
        debug_assert!(self.is_terminal(state));
        // Whoever holds the higher card wins at showdown.
        let showdown = |amount: f64| {
            if state.cards[0] > state.cards[1] {
                amount
            } else {
                -amount
            }
        };
        match (state.len, state.history) {
            (2, 0b00) => showdown(1.0),  // check, check
            (2, 0b01) => 1.0,            // bet, fold: player 1 gave up the ante
            (2, 0b11) => showdown(2.0),  // bet, call
            (3, 0b010) => -1.0,          // check, bet, fold: player 0 gave up
            (3, 0b110) => showdown(2.0), // check, bet, call
            other => unreachable!("not a terminal Kuhn history: {other:?}"),
        }
    }

    fn is_chance(&self, state: &State) -> bool {
        !state.dealt
    }

    fn chance_outcomes(&self, state: &State) -> Vec<(State, f64)> {
        debug_assert!(!state.dealt);
        let mut outcomes = Vec::with_capacity(6);
        // Every ordered pair of distinct cards, uniformly.
        for first in 0..3u8 {
            for second in 0..3u8 {
                if first == second {
                    continue;
                }
                outcomes.push((
                    State {
                        cards: [first, second],
                        dealt: true,
                        ..*state
                    },
                    1.0 / 6.0,
                ));
            }
        }
        outcomes
    }

    fn current_player(&self, state: &State) -> usize {
        (state.len % 2) as usize
    }

    fn info_key(&self, state: &State) -> InfoKey {
        // A player sees their own card and the action so far — never the
        // opponent's card. Leaving the opponent's card out of this key is what
        // makes the game imperfect-information.
        let player = self.current_player(state);
        info_key(state.cards[player], state.history, state.len)
    }

    fn num_actions(&self, _state: &State) -> usize {
        2
    }

    fn apply(&self, state: &State, action: usize) -> State {
        debug_assert!(action < 2);
        State {
            history: state.history | ((action as u8) << state.len),
            len: state.len + 1,
            ..*state
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfr::Solver;

    /// Vanilla CFR on a tree this small converges quickly; this is ample.
    const ITERATIONS: usize = 100_000;

    /// The analytically exact value of Kuhn poker to player 0.
    const GAME_VALUE: f64 = -1.0 / 18.0;

    fn solved() -> Solver<Kuhn> {
        let mut solver = Solver::new(Kuhn);
        solver.train(ITERATIONS);
        solver
    }

    /// Probability of the aggressive action at one information set.
    fn bet_probability(solver: &Solver<Kuhn>, card: u8, history: u8, len: u8) -> f64 {
        solver
            .average_strategy(info_key(card, history, len))
            .unwrap_or_else(|| panic!("info set card={card} history={history:b} len={len} unvisited"))
            [BET]
    }

    #[test]
    fn the_tree_has_the_expected_shape() {
        let game = Kuhn;
        let root = game.initial();
        assert!(game.is_chance(&root));
        assert_eq!(game.chance_outcomes(&root).len(), 6, "3 x 2 ordered deals");

        let dealt = game.chance_outcomes(&root)[0].0;
        assert!(!game.is_chance(&dealt));
        assert_eq!(game.current_player(&dealt), 0);

        // check, bet is the one two-action line that continues.
        let check_bet = game.apply(&game.apply(&dealt, PASS), BET);
        assert!(!game.is_terminal(&check_bet));
        assert_eq!(game.current_player(&check_bet), 0);

        for (a, b) in [(PASS, PASS), (BET, PASS), (BET, BET)] {
            let state = game.apply(&game.apply(&dealt, a), b);
            assert!(game.is_terminal(&state), "{a}{b} should end the hand");
        }
    }

    #[test]
    fn chance_outcomes_are_a_probability_distribution() {
        let game = Kuhn;
        let outcomes = game.chance_outcomes(&game.initial());
        let total: f64 = outcomes.iter().map(|(_, p)| p).sum();
        assert!((total - 1.0).abs() < 1e-12);
        // Never deal the same card twice.
        assert!(outcomes.iter().all(|(s, _)| s.cards[0] != s.cards[1]));
    }

    #[test]
    fn a_player_cannot_see_the_opponents_card() {
        let game = Kuhn;
        // Two deals that give player 0 a king but differ for player 1.
        let a = State { cards: [KING, JACK], dealt: true, ..State::root() };
        let b = State { cards: [KING, QUEEN], dealt: true, ..State::root() };
        assert_eq!(
            game.info_key(&a),
            game.info_key(&b),
            "player 0 must not distinguish these"
        );
    }

    #[test]
    fn terminal_utilities_are_zero_sum_and_correctly_signed() {
        let game = Kuhn;
        let deal = |c0, c1| State { cards: [c0, c1], dealt: true, ..State::root() };

        // Check-check: the better card wins one ante.
        let cc = game.apply(&game.apply(&deal(KING, JACK), PASS), PASS);
        assert_eq!(game.terminal_utility(&cc), 1.0);
        let cc_lose = game.apply(&game.apply(&deal(JACK, KING), PASS), PASS);
        assert_eq!(game.terminal_utility(&cc_lose), -1.0);

        // Bet-fold: player 0 wins regardless of cards.
        let bf = game.apply(&game.apply(&deal(JACK, KING), BET), PASS);
        assert_eq!(game.terminal_utility(&bf), 1.0);

        // Bet-call: two units at showdown.
        let bc = game.apply(&game.apply(&deal(KING, JACK), BET), BET);
        assert_eq!(game.terminal_utility(&bc), 2.0);

        // Check-bet-fold: player 0 folded and loses the ante.
        let cbf = game.apply(&game.apply(&game.apply(&deal(KING, JACK), PASS), BET), PASS);
        assert_eq!(game.terminal_utility(&cbf), -1.0, "folding the best hand still loses");
    }

    #[test]
    fn the_solver_finds_all_twelve_information_sets() {
        let solver = solved();
        // Each player: 3 cards x 2 reachable histories.
        assert_eq!(solver.info_set_count(), 12);
    }

    #[test]
    fn the_game_value_converges_to_minus_one_eighteenth() {
        let solver = solved();
        let value = solver.expected_value(&solver.profile());
        assert!(
            (value - GAME_VALUE).abs() < 5e-3,
            "expected {GAME_VALUE}, got {value}"
        );
    }

    #[test]
    fn the_average_strategy_becomes_almost_unexploitable() {
        let solver = solved();
        let exploitability = solver.exploitability(&solver.profile());
        assert!(
            exploitability < 5e-3,
            "average strategy is exploitable for {exploitability}"
        );
        assert!(exploitability >= 0.0, "exploitability cannot be negative");
    }

    #[test]
    fn player_one_matches_its_unique_equilibrium() {
        let solver = solved();
        let tolerance = 0.02;

        // Facing a bet (history = [BET], len 1).
        let facing_bet = 0b1;
        assert!(
            bet_probability(&solver, JACK, facing_bet, 1) < tolerance,
            "a jack must always fold to a bet"
        );
        assert!(
            (bet_probability(&solver, QUEEN, facing_bet, 1) - 1.0 / 3.0).abs() < tolerance,
            "a queen must bluff-catch exactly one third of the time"
        );
        assert!(
            bet_probability(&solver, KING, facing_bet, 1) > 1.0 - tolerance,
            "a king must always call"
        );

        // Checked to (history = [PASS], len 1).
        let checked_to = 0b0;
        assert!(
            (bet_probability(&solver, JACK, checked_to, 1) - 1.0 / 3.0).abs() < tolerance,
            "a jack must bluff exactly one third of the time"
        );
        assert!(
            bet_probability(&solver, QUEEN, checked_to, 1) < tolerance,
            "a queen has nothing to gain by betting"
        );
        assert!(
            bet_probability(&solver, KING, checked_to, 1) > 1.0 - tolerance,
            "a king must always bet"
        );
    }

    #[test]
    fn player_zero_bluffs_at_one_third_the_rate_it_value_bets() {
        let solver = solved();

        let jack = bet_probability(&solver, JACK, 0, 0);
        let king = bet_probability(&solver, KING, 0, 0);

        // Holds for every equilibrium, whichever alpha CFR settles on.
        assert!(
            (king - 3.0 * jack).abs() < 0.03,
            "expected king bets ({king}) to be three times jack bets ({jack})"
        );
        assert!((0.0..=1.0 / 3.0 + 0.02).contains(&jack), "alpha out of range: {jack}");
    }

    #[test]
    fn player_zero_never_opens_a_queen() {
        let solver = solved();
        assert!(
            bet_probability(&solver, QUEEN, 0, 0) < 0.02,
            "a queen should always check: it folds out worse and is called by better"
        );
    }

    #[test]
    fn player_zero_defends_a_queen_at_alpha_plus_one_third() {
        let solver = solved();
        let alpha = bet_probability(&solver, JACK, 0, 0);
        // History [PASS, BET]: player 0 checked, player 1 bet.
        let queen_call = bet_probability(&solver, QUEEN, 0b10, 2);
        assert!(
            (queen_call - (alpha + 1.0 / 3.0)).abs() < 0.03,
            "expected {} got {queen_call}",
            alpha + 1.0 / 3.0
        );
    }

    #[test]
    fn player_zero_folds_a_jack_and_calls_a_king_after_checking() {
        let solver = solved();
        let checked_then_bet = 0b10;
        assert!(
            bet_probability(&solver, JACK, checked_then_bet, 2) < 0.02,
            "a jack cannot beat anything that bets"
        );
        assert!(
            bet_probability(&solver, KING, checked_then_bet, 2) > 0.98,
            "a king is never folding"
        );
    }

    #[test]
    fn more_training_does_not_make_the_strategy_worse() {
        // Guards against divergence: CFR's bound is monotone in spirit, and a
        // sign error often shows up as exploitability climbing with training.
        let mut solver = Solver::new(Kuhn);
        solver.train(1_000);
        let early = solver.exploitability(&solver.profile());
        solver.train(50_000);
        let late = solver.exploitability(&solver.profile());
        assert!(
            late < early,
            "exploitability rose from {early} to {late} with more training"
        );
    }

    #[test]
    fn a_uniform_strategy_is_measurably_exploitable() {
        // Sanity check on the exploitability metric itself: if it reported ~0
        // for everything, the convergence tests above would prove nothing.
        let solver = Solver::new(Kuhn);
        let uniform = solver.profile();
        assert!(uniform.is_empty(), "untrained solver has no information sets");
        let exploitability = solver.exploitability(&uniform);
        assert!(
            exploitability > 0.05,
            "random play should be clearly exploitable, got {exploitability}"
        );
    }
}
