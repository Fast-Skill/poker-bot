//! Handing what the screen says to the engine that decides.
//!
//! The two halves of this bot were built against different pictures of a poker
//! table. The engine deals in chips, seats numbered from the button, and a
//! [`LegalActions`] saying exactly what may be done. The reader deals in big
//! blinds, seats numbered clockwise from the hero, and a row of buttons. This
//! is the join, and it is where a mistranslation would be most expensive:
//! everything upstream refuses when it is unsure, so anything arriving here is
//! trusted, and a wrong number now becomes a wrong action with no further gate
//! to catch it.
//!
//! # Blinds become chips
//!
//! The client shows every figure in big blinds to one decimal place. The engine
//! counts whole chips, so one big blind is defined here as 100 of them and a
//! reading of `56.7 BB` becomes 5670. That keeps the client's whole precision
//! in integers, where the engine's pot arithmetic cannot drift.
//!
//! # What cannot be seen is not invented
//!
//! Two things the engine wants are not on the screen. How many players were
//! *dealt in* is gone by the flop — a seat that folded looks exactly like one
//! that was never dealt — and the exact minimum raise depends on the size of
//! the last raise, which is history rather than state. Both are handled by
//! saying so rather than by guessing: the first is reported as the number still
//! holding cards, and the second by the widest bound the rules allow.

use poker_core::betting::{LegalActions, Street};
use poker_core::card::Card;
use poker_core::table::{Position, View};
use poker_vision::TableView;

/// Chips to a big blind.
///
/// The client displays one decimal place, so this keeps every figure it shows
/// as an exact integer.
pub const CHIPS_PER_BB: f64 = 100.0;

/// Why a reading could not be turned into a decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Untranslatable {
    NoHero,
    NoHoleCards,
    NoButton,
    /// A stack, the pot, or the amount to call did not read.
    MissingFigure(&'static str),
    /// More board cards than a hand of hold'em has.
    ImpossibleBoard(usize),
}

impl Untranslatable {
    pub fn explain(&self) -> String {
        match self {
            Untranslatable::NoHero => "the hero's own seat was not found".into(),
            Untranslatable::NoHoleCards => "the hero's two cards were not both read".into(),
            Untranslatable::NoButton => "the dealer button was not found".into(),
            Untranslatable::MissingFigure(what) => format!("the {what} did not read"),
            Untranslatable::ImpossibleBoard(n) => {
                format!("the board showed {n} cards, which is not a hold'em street")
            }
        }
    }
}

/// Everything the engine needs, owned, so a [`View`] can borrow from it.
///
/// [`View`] holds references, so the slices it points at have to outlive it.
/// Keeping them together in one value is what lets a caller build the view and
/// then pass it to an agent.
#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    pub hole: [Card; 2],
    pub board: Vec<Card>,
    pub street: Street,
    pub position: Position,
    pub seat: usize,
    pub players: usize,
    pub active: usize,
    pub pot: u64,
    pub to_call: u64,
    pub stack: u64,
    pub stacks: Vec<u64>,
    /// Every seat's commitment this hand, indexed from the button.
    pub committed: Vec<u64>,
    /// Which seats still hold cards, indexed from the button.
    pub live: Vec<bool>,
    /// Which seats have acted since the last full raise, indexed from the
    /// button, and how many raises this street has seen.
    ///
    /// Neither is on the screen. A frame shows what is in front of each player,
    /// not the order it arrived in, so these are what the live loop has been
    /// keeping track of across frames rather than anything read off the felt.
    /// Left at their defaults they describe a street nobody has acted on, which
    /// is true at the moment a street opens and false afterwards — so a caller
    /// that does not track history should not be consulting a postflop solve.
    pub acted: Vec<bool>,
    pub raises: u8,
    /// What was in the middle when this street opened, if the client says.
    ///
    /// The client's own `collected` figure — the chips already swept in from
    /// earlier streets — which is exactly this. `None` when it is not on
    /// screen, and then nobody knows it and nobody pretends to.
    pub settled: Option<u64>,
    pub legal: LegalActions,
}

impl Decision {
    /// The view an agent is handed.
    pub fn view(&self) -> View<'_> {
        View {
            hole: self.hole,
            board: &self.board,
            street: self.street,
            position: self.position,
            seat: self.seat,
            players: self.players,
            active: self.active,
            pot: self.pot,
            to_call: self.to_call,
            stack: self.stack,
            stacks: &self.stacks,
            committed: &self.committed,
            live: &self.live,
            // The two are the same reading here. Only this street's money sits
            // in front of players; whatever went in earlier has been swept into
            // the pot and is no longer attributable to a seat. Preflop that is
            // the whole hand, which is why the preflop routing is exact.
            street_committed: &self.committed,
            acted: &self.acted,
            raises: self.raises,
            settled: self.settled,
            big_blind: CHIPS_PER_BB as u64,
            legal: &self.legal,
        }
    }
}

/// What a frame cannot show: how a street has been bet.
///
/// # The gap this closes
///
/// A postflop solve keys on two facts a single frame does not carry: how many
/// raises this street has seen, and who has acted since the last one. The
/// screen shows what is in front of each player, not the order it arrived in.
/// Bet fifty into fifty and raise to fifty both leave fifty on the felt.
///
/// Almost all of it turns out not to need remembering.
///
/// # Who has acted, from position alone
///
/// After the flop the out-of-position player acts first, every street. So when
/// it is the hero's turn in a heads-up pot:
///
/// - in position, the opponent has necessarily already acted, whatever they did;
/// - out of position with nothing bet, the street has only just opened and
///   nobody has acted;
/// - out of position facing a bet, the opponent put it there.
///
/// That is the whole of it. Turn order is not a thing worth watching for,
/// because it cannot be otherwise.
///
/// # How many raises, from what the bot itself did
///
/// This one does need memory, but far less than watching every frame. Heads-up,
/// the opponent acts exactly once between the hero's turns, so comparing the
/// bet now against the bet at the hero's last turn says whether they raised —
/// no matter how slowly the loop polls or what it missed in between. The hero's
/// own raises need no observation at all: the bot made them.
///
/// Counting this way is exact where watching frames would be a race.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct History {
    /// Which street the counts below belong to, by how many board cards showed.
    board: usize,
    /// The hero's cards, to notice a new hand dealt on the same street count.
    hole: Option<[Card; 2]>,
    raises: u8,
    /// The largest wager on the felt at the hero's last turn this street.
    bet: u64,
}

impl History {
    pub fn new() -> History {
        History::default()
    }

    /// Fills in what the frame could not say, and remembers this turn.
    ///
    /// Call once per decision, before the view is handed to an agent.
    ///
    /// Preflop is left alone. The blinds are wagers nobody chose to make, so
    /// none of the reasoning above holds there — and nothing consults these
    /// fields preflop, where a solved ladder answers instead.
    pub fn observe(&mut self, decision: &mut Decision) {
        let bet = decision.committed.iter().copied().max().unwrap_or(0);

        // A new street, or a new hand that happens to show the same count.
        if decision.board.len() != self.board || self.hole != Some(decision.hole) {
            self.board = decision.board.len();
            self.hole = Some(decision.hole);
            self.raises = 0;
            self.bet = 0;
        }

        if decision.street == Street::Preflop {
            return;
        }

        // The opponent has had at most one turn since the hero's last, so any
        // increase is exactly one raise.
        if bet > self.bet {
            self.raises = self.raises.saturating_add(1);
        }
        self.bet = bet;

        decision.raises = self.raises;
        decision.acted = vec![false; decision.live.len()];
        if decision.active != 2 {
            // Three live players make position ambiguous — there is more than
            // one player before or after the hero — and no postflop solve
            // covers that pot anyway.
            return;
        }
        let Some(opponent) = (0..decision.live.len())
            .find(|seat| decision.live[*seat] && *seat != decision.seat)
        else {
            return;
        };
        // The hero's own bit is never set at the hero's own turn: acting sets
        // it, and the only thing that can bring the turn back round is a raise,
        // which clears it again.
        decision.acted[opponent] = self.hero_is_in_position(decision) || bet > 0;
    }

    /// Whether the hero acts last on this street.
    ///
    /// Postflop the order runs from the seat after the button round to the
    /// button itself, so the last live seat in that order has position.
    fn hero_is_in_position(&self, decision: &Decision) -> bool {
        let last = (1..decision.live.len())
            .chain(std::iter::once(0))
            .rfind(|seat| decision.live[*seat]);
        last == Some(decision.seat)
    }

    /// Tells the tracker the hero has raised to a total this street.
    ///
    /// The bot knows its own action without looking, and saying so here keeps
    /// the count exact through a raising war that a frame comparison alone
    /// would read as a single raise.
    pub fn hero_raised_to(&mut self, total: u64) {
        self.raises = self.raises.saturating_add(1);
        self.bet = self.bet.max(total);
    }
}

/// Rounds a reading in big blinds to whole chips.
fn chips(blinds: f64) -> u64 {
    (blinds * CHIPS_PER_BB).round().max(0.0) as u64
}

/// Turns a reading of the table into something the engine can decide from.
pub fn translate(table: &TableView) -> Result<Decision, Untranslatable> {
    let hero_index = table
        .seats
        .iter()
        .position(|s| s.hero)
        .ok_or(Untranslatable::NoHero)?;
    let hero = &table.seats[hero_index];

    let hole: [Card; 2] = table
        .hole
        .as_slice()
        .try_into()
        .map_err(|_| Untranslatable::NoHoleCards)?;
    let button = table.button.ok_or(Untranslatable::NoButton)?;
    let pot = table.pot.ok_or(Untranslatable::MissingFigure("pot"))?;
    let stack = hero.stack.ok_or(Untranslatable::MissingFigure("stack"))?;
    let to_call = table
        .to_call()
        .ok_or(Untranslatable::MissingFigure("amount to call"))?;

    let street = match table.board.len() {
        0 => Street::Preflop,
        3 => Street::Flop,
        4 => Street::Turn,
        5 => Street::River,
        other => return Err(Untranslatable::ImpossibleBoard(other)),
    };

    // The engine numbers seats from the button; the reader numbers them from
    // the hero. Both run the same way round the table, so this is a rotation.
    let seated = table.seats.len();
    let seat = (hero_index + seated - button) % seated;
    let position = match (seat, seated) {
        (0, _) => Position::Button,
        // Heads-up, the button posts the small blind and there is no seat that
        // is only the button.
        (1, 2) => Position::BigBlind,
        (1, _) => Position::SmallBlind,
        (2, _) => Position::BigBlind,
        _ => Position::Middle,
    };

    // Rotated the same way, so index 0 is always the button's.
    let mut stacks = vec![0u64; seated];
    let mut committed = vec![0u64; seated];
    let mut live = vec![false; seated];
    for (i, s) in table.seats.iter().enumerate() {
        let rotated = (i + seated - button) % seated;
        stacks[rotated] = chips(s.stack.unwrap_or(0.0));
        committed[rotated] = chips(s.bet.unwrap_or(0.0));
        live[rotated] = s.in_hand;
    }

    let stack = chips(stack);
    let to_call = chips(to_call).min(stack);
    let hero_committed = chips(hero.bet.unwrap_or(0.0));

    // What the client is offering, rather than what the rules would allow in
    // the abstract: the raise button being absent means there is nothing to
    // raise with, whatever the arithmetic says.
    let can_raise = table
        .action
        .as_ref()
        .is_some_and(|panel| panel.aggressive().is_some());
    let largest_bet = table
        .seats
        .iter()
        .filter_map(|s| s.bet)
        .fold(0.0f64, f64::max);
    let legal = LegalActions {
        can_fold: true,
        can_check: to_call == 0,
        call_cost: (to_call > 0).then_some(to_call),
        // The smallest legal raise depends on the size of the last raise, which
        // is history and not on the screen. A minimum of twice the largest bet
        // is the smallest that is always legal, and the maximum is everything
        // the hero has.
        raise_to: can_raise.then(|| {
            let all_in = hero_committed + stack;
            let smallest = (chips(largest_bet) * 2).max(CHIPS_PER_BB as u64).min(all_in);
            (smallest, all_in)
        }),
    };

    Ok(Decision {
        hole,
        board: table.board.clone(),
        street,
        position,
        seat,
        // Two different numbers, and using one for both was a real mistake.
        //
        // `players` is the size of the game — how many seats are in it — and it
        // chooses which solved tree applies. `active` is how many are still in
        // the pot, which the tree models internally as folds. A heads-up pot at
        // a seven-handed table is a node *inside* the seven-handed solve, with
        // the folded players' blinds correctly dead in the middle; treating it
        // as a two-handed game throws that solve away and reaches for a
        // heads-up one that prices a different pot.
        players: seated,
        active: table.active(),
        pot: chips(pot),
        to_call,
        stack,
        stacks,
        acted: vec![false; seated],
        raises: 0,
        settled: table.collected.map(chips),
        committed,
        live,
        legal,
    })
}

#[cfg(test)]
mod tests {
    use poker_core::card::parse_cards;
    use super::*;

    /// A heads-up street, bet and raised, is counted exactly.
    #[test]
    fn a_street_is_counted_through_a_raising_war() {
        // Seat 0 is the button and acts last; seat 1 is out of position.
        let flop = |seat: usize, committed: Vec<u64>| Decision {
            hole: parse_two("As Kd"),
            board: parse_cards("Ah 7c 2d").expect("three cards"),
            street: Street::Flop,
            position: Position::Button,
            seat,
            players: 2,
            active: 2,
            pot: 1_000 + committed.iter().sum::<u64>(),
            to_call: committed.iter().copied().max().unwrap_or(0) - committed[seat],
            stack: 9_000,
            stacks: vec![9_000, 9_000],
            committed,
            live: vec![true, true],
            acted: Vec::new(),
            raises: 0,
            settled: None,
            legal: LegalActions {
                can_fold: true,
                can_check: false,
                call_cost: Some(0),
                raise_to: Some((100, 9_000)),
            },
        };

        let mut history = History::new();

        // Out of position, nothing bet: the street has only just opened.
        let mut open = flop(1, vec![0, 0]);
        history.observe(&mut open);
        assert_eq!(open.raises, 0);
        assert_eq!(open.acted, vec![false, false]);

        // It checks to the button, who therefore knows the other has acted.
        let mut checked_to = flop(0, vec![0, 0]);
        history.observe(&mut checked_to);
        assert_eq!(checked_to.raises, 0);
        assert_eq!(checked_to.acted, vec![false, true]);

        // The button bets; the tracker is told, since the bot made the move.
        history.hero_raised_to(500);

        // Back to the other player, facing it.
        let mut facing = flop(1, vec![500, 0]);
        history.observe(&mut facing);
        assert_eq!(facing.raises, 1);
        assert_eq!(facing.acted, vec![true, false]);

        // They raise, which the hero sees only as a larger number on the felt.
        let mut raised_on = flop(0, vec![500, 1_500]);
        history.observe(&mut raised_on);
        assert_eq!(raised_on.raises, 2, "the opponent's raise was not counted");
        assert_eq!(raised_on.acted, vec![false, true]);
    }

    /// A new street starts the count again.
    #[test]
    fn the_turn_does_not_inherit_the_flop() {
        let mut history = History::new();
        let mut flop = sample_decision(Street::Flop, 3, vec![0, 800]);
        history.observe(&mut flop);
        assert_eq!(flop.raises, 1);

        let mut turn = sample_decision(Street::Turn, 4, vec![0, 0]);
        history.observe(&mut turn);
        assert_eq!(turn.raises, 0, "the flop's betting followed it to the turn");
    }

    /// A new hand starts the count again even on the same street.
    #[test]
    fn a_fresh_hand_is_not_the_last_one_continued() {
        let mut history = History::new();
        let mut first = sample_decision(Street::Flop, 3, vec![0, 800]);
        history.observe(&mut first);
        assert_eq!(first.raises, 1);

        let mut second = sample_decision(Street::Flop, 3, vec![0, 0]);
        second.hole = parse_two("7c 2h");
        history.observe(&mut second);
        assert_eq!(second.raises, 0, "the previous hand's betting carried over");
    }

    /// Preflop is left as it was found.
    #[test]
    fn the_blinds_are_not_read_as_betting() {
        let mut history = History::new();
        let mut preflop = sample_decision(Street::Preflop, 0, vec![100, 200]);
        history.observe(&mut preflop);
        assert_eq!(preflop.raises, 0);
        assert!(preflop.acted.is_empty());
    }

    fn parse_two(text: &str) -> [Card; 2] {
        let cards = parse_cards(text).expect("two cards");
        [cards[0], cards[1]]
    }

    fn sample_decision(street: Street, board: usize, committed: Vec<u64>) -> Decision {
        let full = parse_cards("Ah 7c 2d 9s 4h").expect("five cards");
        Decision {
            hole: parse_two("As Kd"),
            board: full[..board].to_vec(),
            street,
            position: Position::Button,
            seat: 0,
            players: 2,
            active: 2,
            pot: 1_000,
            to_call: 0,
            stack: 9_000,
            stacks: vec![9_000, 9_000],
            committed,
            live: vec![true, true],
            acted: Vec::new(),
            raises: 0,
            settled: None,
            legal: LegalActions {
                can_fold: true,
                can_check: true,
                call_cost: None,
                raise_to: Some((100, 9_000)),
            },
        }
    }

    #[test]
    fn blinds_become_chips_without_losing_the_decimal() {
        assert_eq!(chips(56.7), 5670);
        assert_eq!(chips(1.0), 100);
        assert_eq!(chips(0.5), 50);
        assert_eq!(chips(612.3), 61230);
    }

    #[test]
    fn a_negative_reading_cannot_become_a_stack() {
        assert_eq!(chips(-1.0), 0);
    }

    #[test]
    fn a_legal_action_set_always_permits_what_it_advertises() {
        let legal = LegalActions {
            can_fold: true,
            can_check: false,
            call_cost: Some(400),
            raise_to: Some((800, 12_000)),
        };
        assert!(legal.permits(poker_core::betting::Action::Fold));
        assert!(legal.permits(poker_core::betting::Action::Call));
        assert!(legal.permits(poker_core::betting::Action::RaiseTo(800)));
        assert!(legal.permits(poker_core::betting::Action::RaiseTo(12_000)));
        assert!(!legal.permits(poker_core::betting::Action::RaiseTo(799)));
        assert!(!legal.permits(poker_core::betting::Action::Check));
    }

    #[test]
    fn every_reason_a_reading_cannot_be_translated_can_be_explained() {
        for reason in [
            Untranslatable::NoHero,
            Untranslatable::NoHoleCards,
            Untranslatable::NoButton,
            Untranslatable::MissingFigure("pot"),
            Untranslatable::ImpossibleBoard(2),
        ] {
            assert!(!reason.explain().is_empty(), "{reason:?}");
        }
    }
}
