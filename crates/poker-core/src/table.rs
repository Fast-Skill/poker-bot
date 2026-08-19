//! A No-Limit Hold'em table that plays complete hands, two to seven handed.
//!
//! This is the measuring instrument, not a solver. It deals real cards, runs
//! all four streets through [`crate::betting`], and settles at showdown through
//! [`crate::pot`] — so a strategy is judged in the game itself rather than
//! inside whatever abstraction produced it.
//!
//! # Table size changes the rules
//!
//! Heads-up is not merely a small table, it is a different set of rules. The
//! button posts the *small* blind, acts first before the flop, and last on
//! every street after it. With three or more players the button, small blind
//! and big blind are separate seats, the first preflop action falls to the seat
//! left of the big blind, and the small blind leads every later street.
//!
//! Both are implemented here because a real table empties and fills between
//! hands, so a bot meets whichever it is dealt.

use crate::betting::{Action, BettingRound, LegalActions, Seat, Street};
use crate::card::Card;
use crate::eval::{evaluate, HandRank};
use crate::pot::{award, build_pots, OddChip};
use crate::rng::Rng;
use std::fmt;

/// The most players a hand can be dealt to.
pub const MAX_SEATS: usize = 7;

/// Where a seat sits relative to the button.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Position {
    /// On the button — last to act after the flop, and heads-up also the small
    /// blind.
    Button,
    SmallBlind,
    BigBlind,
    /// Anywhere else: neither on the button nor in the blinds.
    Middle,
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
    /// Which seat is acting. Seat 0 always holds the button.
    pub seat: usize,
    /// How many players were dealt into this hand.
    pub players: usize,
    /// How many have not folded, including the actor.
    pub active: usize,
    /// Chips in the middle, including everything wagered this street.
    pub pot: u64,
    /// Chips the acting player must put in to call.
    pub to_call: u64,
    /// Chips the acting player has left.
    pub stack: u64,
    /// Every seat's remaining stack, indexed by seat.
    pub stacks: &'a [u64],
    /// Every seat's total commitment this hand, indexed by seat.
    ///
    /// An agent that consults a solved strategy needs this: which decision it
    /// is facing depends on who has matched the current bet and who has not,
    /// and one seat's own `to_call` cannot say that about the others.
    pub committed: &'a [u64],
    /// Which seats still hold cards, indexed by seat.
    pub live: &'a [bool],
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

    /// Whether only one opponent remains — in which case the hand is heads-up
    /// whatever the table seats, and a two-player strategy applies exactly.
    pub fn is_heads_up(&self) -> bool {
        self.active == 2
    }

    /// The largest stack among opponents still in the hand, which bounds what
    /// can actually be won or lost.
    pub fn effective_stack(&self) -> u64 {
        let largest_opponent = self
            .stacks
            .iter()
            .enumerate()
            .filter(|(seat, _)| *seat != self.seat)
            .map(|(_, stack)| *stack)
            .max()
            .unwrap_or(0);
        self.stack.min(largest_opponent)
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

/// One action, as it happened.
///
/// Recording these is what makes a hand reviewable. When a live bot makes a
/// decision nobody understands, this is the only evidence of what it saw and
/// what it did.
#[derive(Debug, Clone, PartialEq)]
pub struct ActionRecord {
    pub seat: usize,
    pub street: Street,
    pub action: Action,
    /// The pot before this action went in.
    pub pot: u64,
    /// What it cost the actor to call at that moment.
    pub to_call: u64,
}

/// The outcome of one hand.
#[derive(Debug, Clone, PartialEq)]
pub struct HandResult {
    /// Net chips won or lost, per seat. Always sums to zero.
    pub net: Vec<i64>,
    /// Each seat's hole cards.
    pub hole: Vec<[Card; 2]>,
    /// The board as it ran out.
    pub board: Vec<Card>,
    /// How far the hand got.
    pub street: Street,
    /// Whether the hand ended at showdown rather than in a fold.
    pub showdown: bool,
    /// Every action taken, in order.
    pub actions: Vec<ActionRecord>,
}

/// A table.
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
    /// Panics if the big blind is not at least two chips, or the stack cannot
    /// cover it.
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

    /// Which seats post the blinds, as `(small, big)`.
    ///
    /// Heads-up the button posts the small blind; otherwise the blinds sit to
    /// its left. This one rule is the most commonly mis-implemented difference
    /// between heads-up and everything else.
    fn blind_seats(players: usize) -> (usize, usize) {
        if players == 2 {
            (0, 1)
        } else {
            (1, 2)
        }
    }

    /// The seat that opens the betting on each street.
    ///
    /// Preflop the action starts left of the big blind — which heads-up wraps
    /// back around to the button. Afterwards it starts left of the button.
    fn first_to_act(players: usize, street: Street) -> usize {
        match (players, street) {
            (2, Street::Preflop) => 0,
            (2, _) => 1,
            (_, Street::Preflop) => 3 % players,
            _ => 1,
        }
    }

    /// Where a seat sits relative to the button.
    pub fn position_of(players: usize, seat: usize) -> Position {
        let (small, big) = Table::blind_seats(players);
        if seat == 0 {
            Position::Button
        } else if seat == small {
            Position::SmallBlind
        } else if seat == big {
            Position::BigBlind
        } else {
            Position::Middle
        }
    }

    /// Cards a hand consumes: two per player plus a five-card board.
    pub const fn cards_needed(players: usize) -> usize {
        players * 2 + 5
    }

    /// Plays one hand. Seat 0 holds the button.
    ///
    /// `deck` supplies hole cards first, two per seat in order, then the board.
    /// Passing it rather than shuffling internally is what makes duplicate
    /// matches possible — the same deal can be replayed with seats exchanged.
    ///
    /// # Panics
    /// Panics if there are fewer than two or more than [`MAX_SEATS`] players,
    /// if the deck is too short, or if an agent returns an illegal action. All
    /// three are caller bugs, and a table that quietly corrected them would
    /// hide exactly the defects a benchmark exists to find.
    pub fn play_hand(
        &self,
        agents: &mut [&mut dyn Agent],
        deck: &[Card],
        rng: &mut Rng,
    ) -> HandResult {
        let players = agents.len();
        assert!(
            (2..=MAX_SEATS).contains(&players),
            "a hand needs 2 to {MAX_SEATS} players, got {players}"
        );
        assert!(
            deck.len() >= Table::cards_needed(players),
            "need {} cards for {players} players, got {}",
            Table::cards_needed(players),
            deck.len()
        );

        for agent in agents.iter_mut() {
            agent.new_hand();
        }

        let hole: Vec<[Card; 2]> = (0..players)
            .map(|seat| [deck[seat * 2], deck[seat * 2 + 1]])
            .collect();
        let board_start = players * 2;
        let full_board = &deck[board_start..board_start + 5];

        let (small, big) = Table::blind_seats(players);
        let seats: Vec<Seat> = (0..players).map(|_| Seat::new(self.starting_stack)).collect();
        let mut round = BettingRound::preflop(
            seats,
            &[(small, self.small_blind), (big, self.big_blind)],
            Table::first_to_act(players, Street::Preflop),
            self.big_blind,
        );

        let mut street = Street::Preflop;
        let mut showdown = false;
        let mut actions: Vec<ActionRecord> = Vec::new();

        loop {
            while let Some(seat) = round.to_act() {
                let legal = round.legal_actions();
                let board = &full_board[..street.board_cards()];
                let stacks: Vec<u64> = round.seats().iter().map(|s| s.stack).collect();
                let committed = round.contributions();
                let live: Vec<bool> = round.seats().iter().map(|s| !s.folded).collect();
                let view = View {
                    hole: hole[seat],
                    board,
                    street,
                    position: Table::position_of(players, seat),
                    seat,
                    players,
                    active: round.live_seat_count(),
                    pot: round.contributions().iter().sum(),
                    to_call: legal.call_cost.unwrap_or(0),
                    stack: stacks[seat],
                    stacks: &stacks,
                    committed: &committed,
                    live: &live,
                    big_blind: self.big_blind,
                    legal: &legal,
                };

                let action = agents[seat].act(&view, rng);
                actions.push(ActionRecord {
                    seat,
                    street,
                    action,
                    pot: view.pot,
                    to_call: view.to_call,
                });
                round.apply(action).unwrap_or_else(|error| {
                    panic!("{} played an illegal action: {error}", agents[seat].name())
                });
            }

            // One player left means the hand is over.
            if round.live_seat_count() <= 1 {
                break;
            }

            // Everyone still in is all in: run the rest of the board out.
            if round
                .seats()
                .iter()
                .filter(|seat| !seat.folded)
                .all(|seat| seat.is_all_in())
            {
                street = Street::River;
                showdown = true;
                break;
            }

            match street.next() {
                Some(next) => {
                    street = next;
                    round.next_street(Table::first_to_act(players, next));
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

        let ranks: Vec<Option<HandRank>> = (0..players)
            .map(|seat| {
                if folded[seat] {
                    return None;
                }
                if board.len() < 5 {
                    // Everyone else folded before the board completed; the last
                    // player standing wins without showing.
                    return Some(HandRank::WORST);
                }
                let mut cards = hole[seat].to_vec();
                cards.extend_from_slice(&board);
                Some(evaluate(&cards))
            })
            .collect();

        // Odd chips go to the first seat left of the button, as a card room
        // would award them.
        let winnings = award(&pots, &ranks, OddChip::ToSeat(1 % players));
        let net: Vec<i64> = (0..players)
            .map(|seat| winnings[seat] as i64 - contributions[seat] as i64)
            .collect();
        debug_assert_eq!(net.iter().sum::<i64>(), 0, "chips must be conserved");

        HandResult {
            net,
            hole,
            board,
            street,
            showdown: showdown && round.live_seat_count() > 1,
            actions,
        }
    }
}

impl fmt::Display for Table {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} bb stacks", self.starting_stack / self.big_blind)
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

    /// The cards a hand for `players` consumes.
    pub fn hand_cards(&self, players: usize) -> &[Card] {
        &self.cards[..Table::cards_needed(players)]
    }

    /// The same deal with two seats' holdings exchanged.
    ///
    /// This is what makes a duplicate match work: run every deal twice, with
    /// the agents swapping hands, and most of the luck cancels out.
    ///
    /// # Panics
    /// Panics if either seat is beyond the deck's capacity.
    pub fn swap_holdings(&self, a: usize, b: usize) -> Deck {
        let mut cards = self.cards.clone();
        cards.swap(a * 2, b * 2);
        cards.swap(a * 2 + 1, b * 2 + 1);
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
        let used: CardSet = cards.iter().copied().collect();
        cards.extend(CardSet::full_deck().difference(used).iter());
        cards
    }

    /// Plays one hand with `players` callers, for structural checks.
    fn play_with(table: &Table, players: usize, deck: &[Card], rng: &mut Rng) -> HandResult {
        let mut callers: Vec<Caller> = (0..players).map(|_| Caller).collect();
        let mut refs: Vec<&mut dyn Agent> =
            callers.iter_mut().map(|c| c as &mut dyn Agent).collect();
        table.play_hand(&mut refs, deck, rng)
    }

    #[test]
    fn heads_up_puts_the_small_blind_on_the_button() {
        // The rule that separates heads-up from every other table size.
        assert_eq!(Table::blind_seats(2), (0, 1));
        assert_eq!(Table::position_of(2, 0), Position::Button);
        assert_eq!(Table::position_of(2, 1), Position::BigBlind);

        // Three or more, and the blinds move off the button.
        assert_eq!(Table::blind_seats(3), (1, 2));
        assert_eq!(Table::position_of(6, 0), Position::Button);
        assert_eq!(Table::position_of(6, 1), Position::SmallBlind);
        assert_eq!(Table::position_of(6, 2), Position::BigBlind);
        assert_eq!(Table::position_of(6, 4), Position::Middle);
    }

    #[test]
    fn action_order_follows_the_table_size() {
        // Heads-up the button opens preflop and the big blind opens later.
        assert_eq!(Table::first_to_act(2, Street::Preflop), 0);
        assert_eq!(Table::first_to_act(2, Street::Flop), 1);

        // Otherwise the seat left of the big blind opens preflop, and the
        // small blind opens every street after.
        assert_eq!(Table::first_to_act(3, Street::Preflop), 0, "wraps to the button");
        assert_eq!(Table::first_to_act(6, Street::Preflop), 3);
        assert_eq!(Table::first_to_act(6, Street::Flop), 1);
    }

    #[test]
    fn chips_are_conserved_at_every_table_size() {
        let table = Table::standard();
        let mut rng = Rng::new(1);
        let mut deck = Deck::fresh();

        for players in 2..=MAX_SEATS {
            for _ in 0..200 {
                deck.shuffle(&mut rng);
                let mut jammers: Vec<Jammer> = (0..players).map(|_| Jammer).collect();
                let mut refs: Vec<&mut dyn Agent> =
                    jammers.iter_mut().map(|j| j as &mut dyn Agent).collect();
                let result = table.play_hand(&mut refs, deck.hand_cards(players), &mut rng);
                assert_eq!(
                    result.net.iter().sum::<i64>(),
                    0,
                    "{players}-handed leaked chips"
                );
                assert_eq!(result.net.len(), players);
                assert_eq!(result.hole.len(), players);
            }
        }
    }

    #[test]
    fn nobody_loses_more_than_their_stack() {
        let table = Table::standard();
        let mut rng = Rng::new(2);
        let mut deck = Deck::fresh();
        let limit = table.starting_stack() as i64;

        for players in [2usize, 3, 6] {
            for _ in 0..200 {
                deck.shuffle(&mut rng);
                let mut jammers: Vec<Jammer> = (0..players).map(|_| Jammer).collect();
                let mut refs: Vec<&mut dyn Agent> =
                    jammers.iter_mut().map(|j| j as &mut dyn Agent).collect();
                let result = table.play_hand(&mut refs, deck.hand_cards(players), &mut rng);
                for (seat, net) in result.net.iter().enumerate() {
                    // Losses are capped by a seat's own stack. Winnings are
                    // not — a multiway all-in collects every opponent's.
                    assert!(*net >= -limit, "seat {seat} lost {net}, more than its stack");
                    assert!(
                        *net <= limit * (players as i64 - 1),
                        "seat {seat} won {net}, more than the table held"
                    );
                }
            }
        }
    }

    #[test]
    fn folding_the_button_heads_up_loses_exactly_the_small_blind() {
        let table = Table::standard();
        let mut rng = Rng::new(3);
        let deck = deck_from("As Ks Qd Jd 2c 3c 4c 5c 6c");

        let (mut folder, mut caller) = (Folder, Caller);
        let mut refs: Vec<&mut dyn Agent> = vec![&mut folder, &mut caller];
        let result = table.play_hand(&mut refs, &deck, &mut rng);

        assert_eq!(result.net[0], -(table.big_blind() as i64 / 2));
        assert_eq!(result.net[1], table.big_blind() as i64 / 2);
        assert!(!result.showdown);
    }

    #[test]
    fn a_six_handed_pot_is_won_by_the_last_player_standing() {
        let table = Table::standard();
        let mut rng = Rng::new(4);
        let mut deck = Deck::fresh();
        deck.shuffle(&mut rng);

        // Seat 3 opens the action six-handed. It jams; everyone else folds.
        // A raise is what forces the big blind out — facing only a call it
        // would check its option and the hand would reach a flop.
        let mut folders: Vec<Folder> = (0..5).map(|_| Folder).collect();
        let mut jammer = Jammer;
        // split_at_mut hands out two disjoint borrows; chained iterators would
        // borrow the same vector twice.
        let (front, back) = folders.split_at_mut(3);
        let mut refs: Vec<&mut dyn Agent> = Vec::new();
        for folder in front.iter_mut() {
            refs.push(folder as &mut dyn Agent);
        }
        refs.push(&mut jammer);
        for folder in back.iter_mut() {
            refs.push(folder as &mut dyn Agent);
        }

        let result = table.play_hand(&mut refs, deck.hand_cards(6), &mut rng);
        assert_eq!(result.net.iter().sum::<i64>(), 0);
        assert!(!result.showdown, "everyone folded to the jam");
        assert_eq!(
            result.net[3],
            (table.big_blind() + table.big_blind() / 2) as i64,
            "the jam collected both blinds and nothing more"
        );
    }

    #[test]
    fn the_active_count_falls_as_players_fold() {
        /// Records how many players remained at each decision.
        struct Watcher {
            seen: Vec<(usize, usize)>,
        }
        impl Agent for Watcher {
            fn name(&self) -> &str {
                "watch"
            }
            fn act(&mut self, view: &View, _rng: &mut Rng) -> Action {
                self.seen.push((view.players, view.active));
                if view.legal.can_check {
                    Action::Check
                } else {
                    Action::Fold
                }
            }
        }

        let table = Table::standard();
        let mut rng = Rng::new(5);
        let mut deck = Deck::fresh();
        deck.shuffle(&mut rng);

        let mut watcher = Watcher { seen: Vec::new() };
        let mut others: Vec<Folder> = (0..3).map(|_| Folder).collect();
        let mut refs: Vec<&mut dyn Agent> = vec![&mut watcher];
        for other in others.iter_mut() {
            refs.push(other as &mut dyn Agent);
        }
        table.play_hand(&mut refs, deck.hand_cards(4), &mut rng);

        assert!(!watcher.seen.is_empty());
        for (players, active) in &watcher.seen {
            assert_eq!(*players, 4, "four were dealt in");
            assert!(*active >= 2 && *active <= 4, "active was {active}");
        }
    }

    #[test]
    fn a_hand_is_heads_up_when_only_two_remain() {
        // The property the bot routes on: a two-player pot at a six-handed
        // table is heads-up poker, and a two-player strategy applies exactly.
        struct Checker {
            heads_up_seen: bool,
        }
        impl Agent for Checker {
            fn name(&self) -> &str {
                "check"
            }
            fn act(&mut self, view: &View, _rng: &mut Rng) -> Action {
                if view.is_heads_up() {
                    self.heads_up_seen = true;
                    assert_eq!(view.active, 2);
                }
                if view.legal.can_check {
                    Action::Check
                } else {
                    Action::Call
                }
            }
        }

        let table = Table::standard();
        let mut rng = Rng::new(6);
        let mut deck = Deck::fresh();
        deck.shuffle(&mut rng);

        // Three-handed, the button acts first and folds — which leaves the
        // small blind and big blind heads-up, the exact shape the bot routes
        // its two-player strategy on.
        let mut folder = Folder;
        let mut checker = Checker { heads_up_seen: false };
        let mut caller = Caller;
        let mut refs: Vec<&mut dyn Agent> = vec![&mut folder, &mut checker, &mut caller];
        table.play_hand(&mut refs, deck.hand_cards(3), &mut rng);

        assert!(
            checker.heads_up_seen,
            "with four folded, the pot should have become heads-up"
        );
    }

    #[test]
    fn the_board_only_runs_as_far_as_the_hand_does() {
        let table = Table::standard();
        let mut rng = Rng::new(7);
        let deck = deck_from("As Ks Qd Jd 2c 3c 4c 5c 6c");

        let (mut folder, mut caller) = (Folder, Caller);
        let mut refs: Vec<&mut dyn Agent> = vec![&mut folder, &mut caller];
        assert!(table.play_hand(&mut refs, &deck, &mut rng).board.is_empty());

        let showdown = play_with(&table, 2, &deck, &mut rng);
        assert_eq!(showdown.board.len(), 5);
    }

    #[test]
    fn every_action_is_recorded_in_order() {
        let table = Table::standard();
        let mut rng = Rng::new(8);
        let deck = deck_from("As Ah 2c 2d 7s 8d 9h Jc Qs");
        let result = play_with(&table, 2, &deck, &mut rng);

        assert_eq!(result.actions.len(), 8, "two actions on each of four streets");
        assert_eq!(result.actions[0].seat, 0, "the button opens preflop");
        assert_eq!(result.actions[2].seat, 1, "the big blind opens the flop");
        assert!(result.actions.windows(2).all(|w| w[0].street <= w[1].street));
        assert!(result.actions.windows(2).all(|w| w[0].pot <= w[1].pot));
    }

    #[test]
    fn a_shuffled_deck_is_a_permutation_of_the_real_one() {
        let mut rng = Rng::new(9);
        let mut deck = Deck::fresh();
        for _ in 0..100 {
            deck.shuffle(&mut rng);
            let seen: CardSet = deck.cards().iter().copied().collect();
            assert_eq!(seen.len(), 52);
        }
    }

    #[test]
    fn swapping_holdings_exchanges_two_seats_and_leaves_the_board() {
        let mut rng = Rng::new(10);
        let mut deck = Deck::fresh();
        deck.shuffle(&mut rng);
        let swapped = deck.swap_holdings(0, 2);

        assert_eq!(swapped.cards()[0..2], deck.cards()[4..6]);
        assert_eq!(swapped.cards()[4..6], deck.cards()[0..2]);
        // Seat 1 and the board are untouched, which is what makes the pairing
        // fair rather than merely different.
        assert_eq!(swapped.cards()[2..4], deck.cards()[2..4]);
        assert_eq!(swapped.cards()[6..20], deck.cards()[6..20]);
    }

    #[test]
    fn the_deck_supplies_enough_cards_for_a_full_table() {
        assert_eq!(Table::cards_needed(2), 9);
        assert_eq!(Table::cards_needed(7), 19);
        assert!(Deck::fresh().hand_cards(MAX_SEATS).len() <= 52);
    }

    #[test]
    #[should_panic(expected = "2 to 7 players")]
    fn a_one_player_hand_is_rejected() {
        let table = Table::standard();
        let mut rng = Rng::new(11);
        let mut caller = Caller;
        let mut refs: Vec<&mut dyn Agent> = vec![&mut caller];
        table.play_hand(&mut refs, Deck::fresh().cards(), &mut rng);
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
                Action::RaiseTo(u64::MAX)
            }
        }

        let table = Table::standard();
        let mut rng = Rng::new(12);
        let deck = deck_from("As Ah 2c 2d 7s 8d 9h Jc Qs");
        let (mut cheat, mut caller) = (Cheat, Caller);
        let mut refs: Vec<&mut dyn Agent> = vec![&mut cheat, &mut caller];
        table.play_hand(&mut refs, &deck, &mut rng);
    }
}
