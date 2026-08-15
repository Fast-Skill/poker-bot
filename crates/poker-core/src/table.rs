//! A heads-up No-Limit Hold'em table that plays complete hands.
//!
//! This is the measuring instrument, not a solver. It deals real cards, runs
//! all four streets through [`crate::betting`], and settles at showdown through
//! [`crate::pot`] — so a strategy is judged in the game itself rather than
//! inside whatever abstraction produced it.
//!
//! That distinction is the whole point. A strategy can be near-unexploitable
//! within its own model and still lose money at a real table, because the model
//! left something out. Only a full-game match catches that, and only in chips.

use crate::betting::{Action, BettingRound, LegalActions, Seat, Street};
use crate::card::Card;
use crate::eval::evaluate;
use crate::pot::{award, build_pots, OddChip};
use crate::rng::Rng;
use std::fmt;

/// Where a player sits relative to the button.
///
/// Heads-up, the button posts the small blind and acts first before the flop,
/// then acts *last* on every later street. Position is worth real money and
/// agents are told about it explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Position {
    /// On the button: small blind, first to act preflop, last afterwards.
    Button,
    /// The big blind: last to act preflop, first afterwards.
    BigBlind,
}

/// What a player can see when it is their turn.
#[derive(Debug, Clone)]
pub struct View<'a> {
    /// The acting player's two cards.
    pub hole: [Card; 2],
    /// Board cards so far: none, three, four, or five.
    pub board: &'a [Card],
    pub street: Street,
    pub position: Position,
    /// Chips in the middle, including everything wagered this street.
    pub pot: u64,
    /// Chips the acting player must put in to call.
    pub to_call: u64,
    /// Chips the acting player has left.
    pub stack: u64,
    /// Chips the opponent has left.
    pub opponent_stack: u64,
    /// The big blind, so sizes can be reasoned about in blinds.
    pub big_blind: u64,
    /// What is legal here. An agent returning anything else is a bug.
    pub legal: &'a LegalActions,
}

impl View<'_> {
    /// The pot, measured in big blinds.
    pub fn pot_in_blinds(&self) -> f64 {
        self.pot as f64 / self.big_blind as f64
    }

    /// The stack that actually matters — the smaller of the two.
    pub fn effective_stack(&self) -> u64 {
        self.stack.min(self.opponent_stack)
    }

    /// A pot-relative raise target, clamped to what is legal.
    ///
    /// Returns `None` when raising is not available at all.
    pub fn raise_fraction(&self, fraction: f64) -> Option<u64> {
        let (min, max) = self.legal.raise_to?;
        let after_call = self.pot + self.to_call;
        let target = (after_call as f64 * fraction).round() as u64 + self.to_call;
        Some(target.clamp(min, max))
    }
}

/// Something that plays poker.
pub trait Agent {
    /// A short name, used in match reports.
    fn name(&self) -> &str;

    /// Chooses an action. The returned action must satisfy `view.legal`.
    fn act(&mut self, view: &View, rng: &mut Rng) -> Action;

    /// Called once at the start of each hand, for agents that track state.
    fn new_hand(&mut self) {}
}

/// The outcome of one hand.
#[derive(Debug, Clone, PartialEq)]
pub struct HandResult {
    /// Net chips won or lost, per seat. Always sums to zero.
    pub net: [i64; 2],
    /// The board as it ran out.
    pub board: Vec<Card>,
    /// How far the hand got.
    pub street: Street,
    /// Whether the hand ended at showdown rather than in a fold.
    pub showdown: bool,
}

/// A heads-up table.
#[derive(Debug, Clone)]
pub struct Table {
    big_blind: u64,
    small_blind: u64,
    starting_stack: u64,
}

impl Table {
    /// A table with the given blind and starting stack, in chips.
    ///
    /// # Panics
    /// Panics if the big blind is not positive or the stack cannot cover it.
    pub fn new(big_blind: u64, starting_stack: u64) -> Table {
        assert!(big_blind >= 2, "big blind must be at least 2 chips to halve");
        assert!(
            starting_stack >= big_blind,
            "a stack must cover the big blind"
        );
        Table {
            big_blind,
            small_blind: big_blind / 2,
            starting_stack,
        }
    }

    /// A table with 100 big-blind stacks, the usual cash-game depth.
    pub fn standard() -> Table {
        Table::new(100, 10_000)
    }

    pub fn big_blind(&self) -> u64 {
        self.big_blind
    }

    pub fn starting_stack(&self) -> u64 {
        self.starting_stack
    }

    /// Plays one hand. Seat 0 is the button and posts the small blind.
    ///
    /// `deck` must hold at least nine cards: two per player and five for the
    /// board. Supplying it rather than shuffling internally is what makes
    /// duplicate matches possible — the same deal can be replayed with the
    /// seats swapped.
    ///
    /// # Panics
    /// Panics if the deck is too short, or if an agent returns an illegal
    /// action. Both are caller bugs, and a table that quietly corrected them
    /// would hide exactly the defects a benchmark exists to find.
    pub fn play_hand(
        &self,
        agents: [&mut dyn Agent; 2],
        deck: &[Card],
        rng: &mut Rng,
    ) -> HandResult {
        assert!(deck.len() >= 9, "need nine cards to play a hand");
        let [button, big_blind] = agents;
        let mut agents: [&mut dyn Agent; 2] = [button, big_blind];
        for agent in agents.iter_mut() {
            agent.new_hand();
        }

        let hole = [[deck[0], deck[1]], [deck[2], deck[3]]];
        let full_board = &deck[4..9];

        let seats = vec![
            Seat::new(self.starting_stack),
            Seat::new(self.starting_stack),
        ];
        let mut round = BettingRound::preflop(
            seats,
            &[(0, self.small_blind), (1, self.big_blind)],
            0,
            self.big_blind,
        );

        let mut street = Street::Preflop;
        let mut showdown = false;

        loop {
            // Run this street to completion.
            while let Some(seat) = round.to_act() {
                let legal = round.legal_actions();
                let board = &full_board[..street.board_cards()];
                let view = View {
                    hole: hole[seat],
                    board,
                    street,
                    position: if seat == 0 {
                        Position::Button
                    } else {
                        Position::BigBlind
                    },
                    pot: round.contributions().iter().sum(),
                    to_call: legal.call_cost.unwrap_or(0),
                    stack: round.seats()[seat].stack,
                    opponent_stack: round.seats()[1 - seat].stack,
                    big_blind: self.big_blind,
                    legal: &legal,
                };

                let action = agents[seat].act(&view, rng);
                round.apply(action).unwrap_or_else(|error| {
                    panic!("{} played an illegal action: {error}", agents[seat].name())
                });
            }

            // One player left means the hand is over.
            if round.live_seat_count() <= 1 {
                break;
            }

            // Both all in: run the remaining board and settle.
            if round.seats().iter().all(|seat| seat.is_all_in()) {
                street = Street::River;
                showdown = true;
                break;
            }

            match street.next() {
                Some(next) => {
                    street = next;
                    // Out of position acts first once a board exists.
                    round.next_street(1);
                }
                None => {
                    showdown = true;
                    break;
                }
            }
        }

        let contributions = round.contributions();
        let folded = round.folded_flags();
        let pots = build_pots(&contributions, &folded);

        let board = full_board[..street.board_cards()].to_vec();
        let ranks: Vec<Option<crate::eval::HandRank>> = (0..2)
            .map(|seat| {
                if folded[seat] {
                    return None;
                }
                if board.len() < 5 {
                    // Everyone else folded before the board completed; the last
                    // player standing wins without showing.
                    return Some(crate::eval::HandRank::WORST);
                }
                let mut cards = hole[seat].to_vec();
                cards.extend_from_slice(&board);
                Some(evaluate(&cards))
            })
            .collect();

        let winnings = award(&pots, &ranks, OddChip::ToSeat(1));
        let net = [
            winnings[0] as i64 - contributions[0] as i64,
            winnings[1] as i64 - contributions[1] as i64,
        ];
        debug_assert_eq!(net[0] + net[1], 0, "chips must be conserved");

        HandResult {
            net,
            board,
            street,
            showdown: showdown && round.live_seat_count() > 1,
        }
    }
}

impl fmt::Display for Table {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "heads-up, {} bb stacks",
            self.starting_stack / self.big_blind
        )
    }
}

/// A shuffled deck, drawn from once per hand.
#[derive(Debug, Clone)]
pub struct Deck {
    cards: Vec<Card>,
}

impl Deck {
    /// A full deck in a fixed order.
    pub fn fresh() -> Deck {
        Deck {
            cards: Card::all().collect(),
        }
    }

    /// Shuffles in place with Fisher-Yates.
    pub fn shuffle(&mut self, rng: &mut Rng) {
        for index in (1..self.cards.len()).rev() {
            let pick = rng.below(index as u64 + 1) as usize;
            self.cards.swap(index, pick);
        }
    }

    /// The cards, in dealt order.
    pub fn cards(&self) -> &[Card] {
        &self.cards
    }

    /// The nine cards a heads-up hand consumes.
    pub fn hand_cards(&self) -> &[Card] {
        &self.cards[..9]
    }

    /// The same deal with the two players' holdings exchanged.
    ///
    /// This is what makes a duplicate match work: run every deal twice, once
    /// with each agent holding each hand, and most of the luck cancels out.
    pub fn swapped(&self) -> Deck {
        let mut cards = self.cards.clone();
        cards.swap(0, 2);
        cards.swap(1, 3);
        Deck { cards }
    }
}

impl Default for Deck {
    fn default() -> Deck {
        Deck::fresh()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::{parse_cards, CardSet};

    /// Folds whenever folding is legal, otherwise checks.
    struct Folder;
    impl Agent for Folder {
        fn name(&self) -> &str {
            "fold"
        }
        fn act(&mut self, view: &View, _rng: &mut Rng) -> Action {
            if view.legal.can_fold {
                Action::Fold
            } else {
                Action::Check
            }
        }
    }

    /// Never folds and never raises.
    struct Caller;
    impl Agent for Caller {
        fn name(&self) -> &str {
            "call"
        }
        fn act(&mut self, view: &View, _rng: &mut Rng) -> Action {
            if view.legal.can_check {
                Action::Check
            } else {
                Action::Call
            }
        }
    }

    /// Puts everything in at the first opportunity.
    struct Jammer;
    impl Agent for Jammer {
        fn name(&self) -> &str {
            "jam"
        }
        fn act(&mut self, view: &View, _rng: &mut Rng) -> Action {
            match view.legal.raise_to {
                Some((_, max)) => Action::RaiseTo(max),
                None if view.legal.call_cost.is_some() => Action::Call,
                None => Action::Check,
            }
        }
    }

    fn deck_from(text: &str) -> Vec<Card> {
        let mut cards = parse_cards(text).expect("valid cards");
        // Pad with whatever is left so the deck is always long enough.
        let used: CardSet = cards.iter().copied().collect();
        cards.extend(CardSet::full_deck().difference(used).iter());
        cards
    }

    #[test]
    fn chips_are_conserved_in_every_hand() {
        let table = Table::standard();
        let mut rng = Rng::new(1);
        let mut deck = Deck::fresh();

        for _ in 0..500 {
            deck.shuffle(&mut rng);
            let (mut a, mut b) = (Caller, Jammer);
            let result = table.play_hand([&mut a, &mut b], deck.hand_cards(), &mut rng);
            assert_eq!(result.net[0] + result.net[1], 0, "chips leaked: {result:?}");
        }
    }

    #[test]
    fn nobody_can_lose_more_than_their_stack() {
        let table = Table::standard();
        let mut rng = Rng::new(2);
        let mut deck = Deck::fresh();
        let limit = table.starting_stack() as i64;

        for _ in 0..500 {
            deck.shuffle(&mut rng);
            let (mut a, mut b) = (Jammer, Jammer);
            let result = table.play_hand([&mut a, &mut b], deck.hand_cards(), &mut rng);
            for seat in 0..2 {
                assert!(result.net[seat].abs() <= limit, "{result:?}");
            }
        }
    }

    #[test]
    fn folding_the_button_loses_exactly_the_small_blind() {
        let table = Table::standard();
        let mut rng = Rng::new(3);
        let deck = deck_from("As Ks Qd Jd 2c 3c 4c 5c 6c");

        let (mut folder, mut caller) = (Folder, Caller);
        let result = table.play_hand([&mut folder, &mut caller], &deck, &mut rng);

        assert_eq!(result.net[0], -(table.big_blind() as i64 / 2));
        assert_eq!(result.net[1], table.big_blind() as i64 / 2);
        assert!(!result.showdown, "a fold is not a showdown");
        assert_eq!(result.street, Street::Preflop);
    }

    #[test]
    fn two_callers_reach_showdown_and_the_better_hand_wins() {
        let table = Table::standard();
        let mut rng = Rng::new(4);
        // Button holds aces, the big blind holds deuces, board misses both.
        let deck = deck_from("As Ah 2c 2d 7s 8d 9h Jc Qs");

        let (mut a, mut b) = (Caller, Caller);
        let result = table.play_hand([&mut a, &mut b], &deck, &mut rng);

        assert!(result.showdown);
        assert_eq!(result.street, Street::River);
        assert_eq!(result.board.len(), 5);
        assert!(result.net[0] > 0, "aces should win: {result:?}");
        assert_eq!(result.net[0], table.big_blind() as i64, "one blind each way");
    }

    #[test]
    fn a_jammer_against_a_caller_settles_for_the_whole_stack() {
        let table = Table::standard();
        let mut rng = Rng::new(5);
        let deck = deck_from("As Ah 2c 2d 7s 8d 9h Jc Qs");

        let (mut jammer, mut caller) = (Jammer, Caller);
        let result = table.play_hand([&mut jammer, &mut caller], &deck, &mut rng);

        assert!(result.showdown);
        assert_eq!(
            result.net[0],
            table.starting_stack() as i64,
            "aces get it all in and hold"
        );
    }

    #[test]
    fn a_jammer_against_a_folder_wins_only_the_blind() {
        let table = Table::standard();
        let mut rng = Rng::new(6);
        let deck = deck_from("7c 2d As Ah 3c 4c 5c 8d 9d");

        let (mut jammer, mut folder) = (Jammer, Folder);
        let result = table.play_hand([&mut jammer, &mut folder], &deck, &mut rng);

        assert!(!result.showdown, "the big blind folded");
        assert_eq!(
            result.net[0],
            table.big_blind() as i64,
            "winning uncontested takes the blind, not the stack"
        );
    }

    #[test]
    fn the_board_only_runs_as_far_as_the_hand_does() {
        let table = Table::standard();
        let mut rng = Rng::new(7);
        let deck = deck_from("As Ks Qd Jd 2c 3c 4c 5c 6c");

        let (mut folder, mut caller) = (Folder, Caller);
        let folded = table.play_hand([&mut folder, &mut caller], &deck, &mut rng);
        assert!(folded.board.is_empty(), "no flop was dealt");

        let (mut a, mut b) = (Caller, Caller);
        let showdown = table.play_hand([&mut a, &mut b], &deck, &mut rng);
        assert_eq!(showdown.board.len(), 5);
    }

    #[test]
    fn a_shuffled_deck_is_a_permutation_of_the_real_one() {
        let mut rng = Rng::new(8);
        let mut deck = Deck::fresh();
        for _ in 0..100 {
            deck.shuffle(&mut rng);
            let seen: CardSet = deck.cards().iter().copied().collect();
            assert_eq!(seen.len(), 52, "the deck lost or repeated a card");
        }
    }

    #[test]
    fn swapping_a_deck_exchanges_the_two_holdings() {
        let mut rng = Rng::new(9);
        let mut deck = Deck::fresh();
        deck.shuffle(&mut rng);
        let swapped = deck.swapped();

        assert_eq!(swapped.cards()[0], deck.cards()[2]);
        assert_eq!(swapped.cards()[1], deck.cards()[3]);
        assert_eq!(swapped.cards()[2], deck.cards()[0]);
        assert_eq!(swapped.cards()[3], deck.cards()[1]);
        // The board is untouched, which is what makes the pairing fair.
        assert_eq!(&swapped.cards()[4..9], &deck.cards()[4..9]);
    }

    #[test]
    fn position_is_reported_correctly() {
        struct Watcher {
            seen: Vec<(Street, Position)>,
        }
        impl Agent for Watcher {
            fn name(&self) -> &str {
                "watch"
            }
            fn act(&mut self, view: &View, _rng: &mut Rng) -> Action {
                self.seen.push((view.street, view.position));
                if view.legal.can_check {
                    Action::Check
                } else {
                    Action::Call
                }
            }
        }

        let table = Table::standard();
        let mut rng = Rng::new(10);
        let deck = deck_from("As Ah 2c 2d 7s 8d 9h Jc Qs");
        let mut watcher = Watcher { seen: Vec::new() };
        let mut caller = Caller;
        table.play_hand([&mut watcher, &mut caller], &deck, &mut rng);

        assert!(watcher
            .seen
            .iter()
            .all(|(_, position)| *position == Position::Button));
        // The button acts on every street, preflop through river.
        assert!(watcher.seen.iter().any(|(street, _)| *street == Street::Preflop));
        assert!(watcher.seen.iter().any(|(street, _)| *street == Street::River));
    }

    #[test]
    fn the_pot_the_agent_sees_includes_the_blinds() {
        struct Peek {
            first_pot: Option<u64>,
        }
        impl Agent for Peek {
            fn name(&self) -> &str {
                "peek"
            }
            fn act(&mut self, view: &View, _rng: &mut Rng) -> Action {
                self.first_pot.get_or_insert(view.pot);
                if view.legal.can_check {
                    Action::Check
                } else {
                    Action::Fold
                }
            }
        }

        let table = Table::standard();
        let mut rng = Rng::new(11);
        let deck = deck_from("As Ah 2c 2d 7s 8d 9h Jc Qs");
        let mut peek = Peek { first_pot: None };
        let mut caller = Caller;
        table.play_hand([&mut peek, &mut caller], &deck, &mut rng);

        assert_eq!(
            peek.first_pot,
            Some(table.big_blind() + table.big_blind() / 2),
            "both blinds are already in"
        );
    }

    #[test]
    #[should_panic(expected = "illegal action")]
    fn an_illegal_action_is_a_loud_failure() {
        struct Cheat;
        impl Agent for Cheat {
            fn name(&self) -> &str {
                "cheat"
            }
            fn act(&mut self, _view: &View, _rng: &mut Rng) -> Action {
                // Far beyond any stack at this table.
                Action::RaiseTo(u64::MAX)
            }
        }

        let table = Table::standard();
        let mut rng = Rng::new(12);
        let deck = deck_from("As Ah 2c 2d 7s 8d 9h Jc Qs");
        let (mut cheat, mut caller) = (Cheat, Caller);
        table.play_hand([&mut cheat, &mut caller], &deck, &mut rng);
    }
}
