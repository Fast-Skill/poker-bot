//! Preflop hold'em at a ring table, solved rather than approximated.
//!
//! [`crate::preflop`] models the heads-up game as a named ladder of stages —
//! open, 3-bet, 4-bet — which is readable precisely because heads-up has so few
//! shapes. Three-handed has far more, and seven-handed more again, so this
//! models the betting as a *state machine* instead: who is live, who has acted
//! since the last raise, and how much each has in. The tree is then whatever
//! that machine generates, and adding a seat costs nothing in this file.
//!
//! # The seat count is limited by equity, not by the tree
//!
//! The state machine handles any number of seats. What limits it is the
//! showdown: a pot contested by `n` players needs `n`-way equity, and
//! [`crate::multiway::approximate_shares`] documents in detail why that cannot
//! be assembled from pairwise numbers. So a game is refused up front unless the
//! [`Showdown`] it is given can settle the widest pot it could ever produce —
//! which is every seat reaching a flop together.
//!
//! # Heads-up is not a special case here
//!
//! Run at two seats this plays the same game [`crate::preflop`] does, which is
//! the check that matters: the tree is either right against an answer already
//! known, or it is wrong somewhere a three-handed solve would never reveal.

use crate::abstraction::{HandClass, NUM_HAND_CLASSES};
use crate::card::{Card, CardSet};
use crate::cfr::{Game, InfoKey};
use crate::rng::Rng;
use crate::wide::Showdown;

/// The most seats the state machine carries. Showdowns are limited separately.
pub const MAX_SEATS: usize = 7;

/// How far apart two amounts are as a ratio, so that 5 against 9 counts as
/// closer than 5 against 2.5 — which by simple difference it would not.
fn ratio_gap(a: u32, b: u32) -> f64 {
    let (a, b) = (a.max(1) as f64, b.max(1) as f64);
    (a / b).max(b / a)
}

/// Chips are hundredths of a big blind: exact for any realistic sizing, and
/// free of the float-equality hazards that comparing states would otherwise
/// bring.
const SCALE: f64 = 100.0;

fn to_chips(blinds: f64) -> u32 {
    (blinds * SCALE).round() as u32
}

fn to_blinds(chips: u32) -> f64 {
    chips as f64 / SCALE
}

/// The raise ladder, in big blinds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ladder {
    /// The first raise into an unopened pot.
    pub open_to: f64,
    /// The raise over that.
    pub three_bet_to: f64,
    /// And over that. Beyond this the only raise left is all-in.
    pub four_bet_to: f64,
}

impl Default for Ladder {
    /// Sizes a little larger than the heads-up ladder, because opening into two
    /// players rather than one has to charge more to fold out the same share of
    /// hands.
    fn default() -> Ladder {
        Ladder {
            open_to: 2.5,
            three_bet_to: 9.0,
            four_bet_to: 22.0,
        }
    }
}

impl Ladder {
    /// What a raise commits the raiser to, given how many have gone in already.
    fn target(&self, raises: u8) -> f64 {
        match raises {
            0 => self.open_to,
            1 => self.three_bet_to,
            _ => self.four_bet_to,
        }
    }

    /// Rungs on the ladder. Past the last one the only raise left is all-in.
    const DEPTH: u8 = 3;

    /// The level recorded once somebody is all-in.
    ///
    /// All-in is not simply "one more raise". Facing a hundred blinds is a
    /// different decision from facing two and a half, and counting a jam as
    /// another rung made the two indistinguishable — which showed up as one
    /// information set claiming two different sets of actions.
    const ALL_IN: u8 = 4;
}

/// What a player may do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Move {
    Fold,
    /// Check when nothing is owed, call when something is.
    Passive,
    /// Raise to the next rung of the ladder.
    Raise,
    /// All-in.
    Jam,
}

/// A node in the tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct State {
    classes: [u8; MAX_SEATS],
    committed: [u32; MAX_SEATS],
    /// Bit per seat: still holding cards.
    live: u8,
    /// Bit per seat: has acted since the last raise.
    ///
    /// This is what gives the big blind its option. Everybody limping leaves
    /// the amounts level, but the big blind has not acted, so the betting is
    /// not finished — and no separate rule is needed to say so.
    acted: u8,
    to_act: u8,
    raises: u8,
    /// Who raised last, or `MAX_SEATS` if nobody has.
    aggressor: u8,
    dealt: bool,
}

impl State {
    /// The hand class held by `seat`.
    pub fn hand(&self, seat: usize) -> HandClass {
        HandClass::from_index(self.classes[seat] as usize).expect("a dealt class is in range")
    }

    /// Chips `seat` has put in, in big blinds.
    pub fn committed(&self, seat: usize) -> f64 {
        to_blinds(self.committed[seat])
    }

    pub fn is_live(&self, seat: usize) -> bool {
        self.live & (1 << seat) != 0
    }

    /// Seats still holding cards.
    pub fn live_seats(&self) -> Vec<usize> {
        (0..MAX_SEATS).filter(|&s| self.is_live(s)).collect()
    }
}

/// Preflop hold'em at a ring table.
#[derive(Debug, Clone)]
pub struct Ring {
    seats: usize,
    stack: u32,
    ladder: Ladder,
    showdown: Showdown,
}

impl Ring {
    /// Builds a game for `seats` players, each `stack` big blinds deep.
    ///
    /// # Panics
    /// Panics when `showdown` cannot settle a pot of `seats` players. The tree
    /// would be fine; the showdown would not, and a solve unable to price its
    /// own widest pot would be learning from a number nobody measured.
    pub fn new(seats: usize, stack: f64, ladder: Ladder, showdown: Showdown) -> Ring {
        assert!(
            (2..=crate::wide::MAX_PLAYERS).contains(&seats),
            "a table seats between two and {}, not {seats}",
            crate::wide::MAX_PLAYERS
        );
        assert!(
            seats <= showdown.reach(),
            "solving {seats}-handed needs {seats}-way equity, and the tables given reach only {}",
            showdown.reach()
        );
        Ring {
            seats,
            stack: to_chips(stack),
            ladder,
            showdown,
        }
    }


    pub fn seats(&self) -> usize {
        self.seats
    }

    /// Where the blinds sit. Heads-up the button posts the small blind.
    fn blind_seats(&self) -> (usize, usize) {
        if self.seats == 2 {
            (0, 1)
        } else {
            (1, 2)
        }
    }

    /// Who opens the betting. Heads-up that is the button; otherwise the seat
    /// after the big blind.
    fn first_to_act(&self) -> usize {
        if self.seats == 2 {
            0
        } else {
            3 % self.seats
        }
    }

    /// Whether a seat can still make a decision.
    fn can_act(&self, state: &State, seat: usize) -> bool {
        state.is_live(seat) && state.committed[seat] < self.stack
    }

    /// The largest amount any live seat has committed.
    fn to_match(&self, state: &State) -> u32 {
        (0..self.seats)
            .filter(|&s| state.is_live(s))
            .map(|s| state.committed[s])
            .max()
            .unwrap_or(0)
    }

    /// Whether the betting is finished.
    fn is_closed(&self, state: &State) -> bool {
        let live: Vec<usize> = (0..self.seats).filter(|&s| state.is_live(s)).collect();
        if live.len() <= 1 {
            return true;
        }
        let owed = self.to_match(state);
        live.iter().all(|&s| {
            // All-in players cannot act again and have matched all they can.
            state.committed[s] >= self.stack
                || (state.acted & (1 << s) != 0 && state.committed[s] == owed)
        })
    }

    /// The next seat with a decision to make, if any.
    fn next_actor(&self, state: &State, from: usize) -> Option<usize> {
        (1..=self.seats)
            .map(|step| (from + step) % self.seats)
            .find(|&seat| self.can_act(state, seat))
    }

    /// The moves available here, in a fixed order so action indices are stable.
    fn moves(&self, state: &State) -> Vec<Move> {
        let seat = state.to_act as usize;
        let owed = self.to_match(state);
        let mut moves = Vec::with_capacity(4);

        // Folding with nothing owed is legal at a real table and never right.
        // Leaving it out keeps the solver from splitting regret across an
        // action it would never take.
        if state.committed[seat] < owed {
            moves.push(Move::Fold);
        }
        moves.push(Move::Passive);

        if state.raises < Ladder::DEPTH {
            let target = to_chips(self.ladder.target(state.raises));
            // A raise at or above the stack is a jam wearing another name, and
            // offering both would split regret between identical actions.
            if target > owed && target < self.stack {
                moves.push(Move::Raise);
            }
        }
        // Only when jamming is actually a raise. Facing an all-in, shoving and
        // calling put in the same chips, and offering both would split the
        // solver's regret between two names for one action.
        if self.stack > owed {
            moves.push(Move::Jam);
        }
        moves
    }

    /// Shares of the pot at a showdown, indexed by position within `live`.
    fn shares(&self, state: &State, live: &[usize]) -> Vec<f64> {
        let hands: Vec<crate::abstraction::HandClass> =
            live.iter().map(|&seat| state.hand(seat)).collect();
        self.showdown.shares(&hands)
    }

    /// Builds an information key from what a table shows.
    ///
    /// This is the join between a screen and a solved strategy, and it exists
    /// because every field the key needs is visible in one frame: who still
    /// holds cards, what each has pushed out, whose turn it is, and the hand
    /// itself. No memory of how the betting got here is required — see
    /// [`Game::info_key`] for why the one field that would have needed it was
    /// left out.
    ///
    /// `live` and `committed` are indexed by seat *from the button*, matching
    /// the tree rather than the screen. Returns `None` when the chips do not
    /// correspond to any point on the ladder, which means the reading is of
    /// some other game — a different raise size, an ante, a straddle — and
    /// guessing at it would be worse than declining.
    pub fn key_from_table(
        &self,
        hero: usize,
        hand: HandClass,
        live: &[bool],
        committed: &[f64],
    ) -> Option<InfoKey> {
        if hero >= self.seats || live.len() < self.seats || committed.len() < self.seats {
            return None;
        }
        if !live[hero] {
            return None;
        }

        let chips: Vec<u32> = committed.iter().take(self.seats).map(|&b| to_chips(b)).collect();
        let owed = (0..self.seats)
            .filter(|&s| live[s])
            .map(|s| chips[s])
            .max()?;
        let raises = self.level_of(owed)?;

        let (_, big) = self.blind_seats();
        let mut live_mask = 0u8;
        let mut acted = 0u8;
        for seat in 0..self.seats {
            if !live[seat] {
                continue;
            }
            live_mask |= 1 << seat;
            // Posting the big blind is not acting: the option is still to come.
            let posted_only = seat == big && raises == 0;
            if chips[seat] == owed && !posted_only {
                acted |= 1 << seat;
            }
        }

        let mut key = hero as u64;
        key = (key << 3) | raises as u64;
        key = (key << 7) | live_mask as u64;
        key = (key << 7) | acted as u64;
        Some((key << 8) | hand.index() as u64)
    }

    /// The moves available to a seat, given what the table shows.
    ///
    /// The order matches the one the solver trained against, so a blueprint's
    /// probabilities line up with these by index. Returns `None` when the chips
    /// do not describe a spot in this game.
    pub fn moves_at(&self, hero: usize, live: &[bool], committed: &[f64]) -> Option<Vec<Move>> {
        let state = self.state_from_table(hero, live, committed)?;
        Some(self.moves(&state))
    }

    /// What a raising move commits the raiser to, in big blinds.
    pub fn raise_target(
        &self,
        chosen: Move,
        hero: usize,
        live: &[bool],
        committed: &[f64],
    ) -> Option<f64> {
        let state = self.state_from_table(hero, live, committed)?;
        match chosen {
            Move::Jam => Some(to_blinds(self.stack)),
            Move::Raise => Some(self.ladder.target(state.raises)),
            _ => None,
        }
    }

    /// Rebuilds enough of a state to ask what may be done in it.
    ///
    /// Only the fields the move list depends on are recovered; the hands are
    /// not, since which moves exist does not depend on what anybody holds.
    fn state_from_table(&self, hero: usize, live: &[bool], committed: &[f64]) -> Option<State> {
        if hero >= self.seats || live.len() < self.seats || committed.len() < self.seats {
            return None;
        }
        let mut state = self.initial();
        state.dealt = true;
        state.to_act = hero as u8;
        state.live = 0;
        for seat in 0..self.seats {
            if live[seat] {
                state.live |= 1 << seat;
            }
            state.committed[seat] = to_chips(committed[seat]);
        }
        state.raises = self.level_of(self.to_match(&state))?;
        Some(state)
    }

    /// Which rung of the ladder an amount corresponds to.
    ///
    /// # Real tables do not raise in the sizes a solve was built from
    ///
    /// This used to demand an exact match, and the consequence was severe and
    /// invisible: a live run showed the solve being consulted only in unopened
    /// pots. Every raise the opponents actually made — to 2, to 3, to 4, to 4.5
    /// — matched no rung, so the lookup declined and a heuristic played the
    /// hand. The strategy was correct, complete, and almost never used.
    ///
    /// So an amount is now mapped to the rung nearest it. That has a cost worth
    /// stating: facing 4 blinds while consulting the strategy for 2.5 means
    /// playing slightly too loosely, because the price of continuing is higher
    /// than the strategy believes. It is a far smaller error than abandoning
    /// the solve entirely, which is what exactness bought.
    ///
    /// The mapping is by ratio rather than by difference. Sizes grow
    /// multiplicatively — 1, 2.5, 9, 22 — so 5 sits nearer 9 than 2.5 by
    /// difference, while being much the same distance from either as a price.
    fn level_of(&self, owed: u32) -> Option<u8> {
        if owed == 0 {
            return None;
        }
        if owed >= self.stack {
            return Some(Ladder::ALL_IN);
        }

        // Every size this game can be in, with the level that describes it.
        let mut rungs = vec![(to_chips(1.0), 0u8)];
        for rung in 0..Ladder::DEPTH {
            let target = to_chips(self.ladder.target(rung));
            if target < self.stack {
                rungs.push((target, rung + 1));
            }
        }
        rungs.push((self.stack, Ladder::ALL_IN));

        rungs
            .iter()
            .min_by(|a, b| ratio_gap(owed, a.0).total_cmp(&ratio_gap(owed, b.0)))
            .map(|(_, level)| *level)
    }

    fn deal_from(&self, state: &State, classes: [u8; MAX_SEATS]) -> State {
        let mut next = State {
            classes,
            dealt: true,
            ..*state
        };
        next.to_act = self.first_to_act() as u8;
        next
    }
}

impl Game for Ring {
    type State = State;

    fn initial(&self) -> State {
        let (small, big) = self.blind_seats();
        let mut committed = [0u32; MAX_SEATS];
        committed[small] = to_chips(0.5);
        committed[big] = to_chips(1.0);
        State {
            classes: [0; MAX_SEATS],
            committed,
            live: (1u8 << self.seats) - 1,
            acted: 0,
            to_act: 0,
            raises: 0,
            aggressor: MAX_SEATS as u8,
            dealt: false,
        }
    }

    fn players(&self) -> usize {
        self.seats
    }

    fn is_terminal(&self, state: &State) -> bool {
        state.dealt && self.is_closed(state)
    }

    /// Only meaningful heads-up, where one signed number describes both.
    fn terminal_utility(&self, state: &State) -> f64 {
        debug_assert_eq!(self.seats, 2, "three-handed payoffs need utility_for");
        self.utility_for(state, 0)
    }

    fn utility_for(&self, state: &State, player: usize) -> f64 {
        debug_assert!(player < self.seats);
        let staked = to_blinds(state.committed[player]);
        if !state.is_live(player) {
            return -staked;
        }

        let live: Vec<usize> = (0..self.seats).filter(|&s| state.is_live(s)).collect();
        let pot: f64 = (0..self.seats).map(|s| to_blinds(state.committed[s])).sum();
        if live.len() == 1 {
            return pot - staked;
        }

        // Every stack is the same depth here, so a seat that is all-in is all-in
        // for the full amount and nobody is contesting a smaller pot than
        // anybody else. Side pots would be needed at mixed depths.
        debug_assert!(
            live.iter().all(|&s| state.committed[s] == state.committed[live[0]]),
            "a showdown should leave every live seat matched"
        );

        let shares = self.shares(state, &live);
        let mine = live
            .iter()
            .position(|&s| s == player)
            .expect("a live player is among the live seats");
        shares[mine] * pot - staked
    }

    fn is_chance(&self, state: &State) -> bool {
        !state.dealt
    }

    /// Every deal, which is only tractable heads-up.
    ///
    /// Three-handed this is 169³ outcomes per visit, so a three-handed solve
    /// must sample; the full distribution stays available because heads-up
    /// exhaustive training is what validates the tree.
    fn chance_outcomes(&self, state: &State) -> Vec<(State, f64)> {
        debug_assert!(self.is_chance(state));
        assert_eq!(
            self.seats, 2,
            "enumerating every deal past two players is not tractable; train by sampling"
        );

        let mut outcomes = Vec::with_capacity(NUM_HAND_CLASSES * NUM_HAND_CLASSES);
        let mut total = 0.0;
        for a in HandClass::all() {
            for b in HandClass::all() {
                let weight = (a.combos() * b.combos()) as f64;
                total += weight;
                let mut classes = [0u8; MAX_SEATS];
                classes[0] = a.index() as u8;
                classes[1] = b.index() as u8;
                outcomes.push((self.deal_from(state, classes), weight));
            }
        }
        for outcome in &mut outcomes {
            outcome.1 /= total;
        }
        outcomes
    }

    fn sample_chance(&self, state: &State, rng: &mut Rng) -> State {
        let first = Card::all().next().expect("a non-empty deck");
        let mut drawn = [first; MAX_SEATS * 2];
        let mut dead = CardSet::empty();
        for slot in drawn.iter_mut().take(self.seats * 2) {
            loop {
                let card = Card::from_index(rng.below(52) as u8).expect("0..52");
                if dead.insert(card) {
                    *slot = card;
                    break;
                }
            }
        }
        let mut classes = [0u8; MAX_SEATS];
        for seat in 0..self.seats {
            classes[seat] =
                HandClass::from_cards(drawn[seat * 2], drawn[seat * 2 + 1]).index() as u8;
        }
        self.deal_from(state, classes)
    }

    fn current_player(&self, state: &State) -> usize {
        state.to_act as usize
    }

    fn info_key(&self, state: &State) -> InfoKey {
        // Everything the acting player can see, packed into its own bits.
        //
        // Laid out by shifting rather than by multiplying out offsets by hand.
        // The first attempt did the latter and gave `raises` four values, when
        // it reaches five — three rungs of the ladder and then a jam — so a
        // deeply-raised pot collided with a different set of live seats. Two
        // states sharing a key is not a rounding error: the solver caches the
        // action count against the key, and the collision handed one state the
        // other's actions.
        //
        // The last raiser is deliberately absent. It is implied by the four
        // fields here — asserted, not assumed — and it is the one thing a
        // single frame cannot show, since a raise that has been called leaves
        // raiser and caller with the same chips in front of them. Leaving it
        // out is what lets a bot build this key from a screenshot instead of
        // from a remembered history of the hand.
        debug_assert!(state.raises < 8, "raises {} needs more bits", state.raises);

        let class = state.classes[state.to_act as usize] as u64;
        let mut key = state.to_act as u64;
        key = (key << 3) | state.raises as u64;
        key = (key << 7) | state.live as u64;
        key = (key << 7) | state.acted as u64;
        (key << 8) | class
    }

    fn num_actions(&self, state: &State) -> usize {
        self.moves(state).len()
    }

    fn apply(&self, state: &State, action: usize) -> State {
        let seat = state.to_act as usize;
        let chosen = self.moves(state)[action];
        let owed = self.to_match(state);
        let mut next = *state;

        match chosen {
            Move::Fold => {
                next.live &= !(1 << seat);
            }
            Move::Passive => {
                next.committed[seat] = owed.min(self.stack);
            }
            Move::Raise => {
                next.committed[seat] = to_chips(self.ladder.target(state.raises)).min(self.stack);
                next.raises += 1;
                next.aggressor = seat as u8;
                // A raise reopens the betting for everybody else.
                next.acted = 0;
            }
            Move::Jam => {
                next.committed[seat] = self.stack;
                next.raises = Ladder::ALL_IN;
                next.aggressor = seat as u8;
                next.acted = 0;
            }
        }
        next.acted |= 1 << seat;
        // Only live seats count as having acted. A folded seat's bit says
        // nothing about whether betting can close — `is_closed` looks only at
        // live seats — and keeping it would split one decision into two
        // information sets over history that no longer matters. It would also
        // put the key out of reach of a screen reader: whether a player who
        // folded did so before or after the last raise is not visible in a
        // frame, while "who has matched the current bet" is.
        next.acted &= next.live;

        if let Some(seat) = self.next_actor(&next, seat) {
            next.to_act = seat as u8;
        }
        next
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cheap tables: these tests exercise the tree, not the equity numbers.
    fn showdown() -> Showdown {
        Showdown::new(
            crate::pushfold::EquityTable::sampled_parallel(64, 0x51DE, 4),
            crate::threeway::ThreeWayEquity::sampled_parallel(1, 0x3EED, 4),
        )
    }

    fn ring(seats: usize) -> Ring {
        Ring::new(seats, 100.0, Ladder::default(), showdown())
    }

    /// Walks a state forward by naming moves, which reads like a hand does.
    fn play(game: &Ring, mut state: State, moves: &[Move]) -> State {
        for wanted in moves {
            let available = game.moves(&state);
            let index = available
                .iter()
                .position(|m| m == wanted)
                .unwrap_or_else(|| panic!("{wanted:?} not available among {available:?}"));
            state = game.apply(&state, index);
        }
        state
    }

    fn dealt(game: &Ring) -> State {
        let mut rng = Rng::new(1);
        game.sample_chance(&game.initial(), &mut rng)
    }

    #[test]
    fn the_blinds_are_posted_before_anyone_acts() {
        let three = ring(3);
        let state = three.initial();
        assert_eq!(state.committed(0), 0.0, "the button posts nothing");
        assert_eq!(state.committed(1), 0.5);
        assert_eq!(state.committed(2), 1.0);

        // Heads-up the button posts the small blind instead.
        let two = ring(2);
        let state = two.initial();
        assert_eq!(state.committed(0), 0.5);
        assert_eq!(state.committed(1), 1.0);
    }

    #[test]
    fn three_handed_the_button_opens_and_the_big_blind_closes() {
        let game = ring(3);
        let state = dealt(&game);
        assert_eq!(game.current_player(&state), 0, "the button acts first");

        let after_button = play(&game, state, &[Move::Passive]);
        assert_eq!(game.current_player(&after_button), 1, "then the small blind");

        let after_small = play(&game, after_button, &[Move::Passive]);
        assert_eq!(game.current_player(&after_small), 2, "then the big blind");
    }

    #[test]
    fn the_big_blind_still_has_a_say_when_everybody_limps() {
        // Everyone matching leaves the amounts level, which is what closes
        // betting everywhere else. The big blind has not acted, so it must not
        // close here.
        let game = ring(3);
        let state = play(&game, dealt(&game), &[Move::Passive, Move::Passive]);
        assert!(!game.is_terminal(&state), "the big blind has the option");
        assert_eq!(game.current_player(&state), 2);

        let checked = play(&game, state, &[Move::Passive]);
        assert!(game.is_terminal(&checked), "and once taken, betting is over");
    }

    #[test]
    fn a_raise_gives_everyone_who_already_acted_another_turn() {
        let game = ring(3);
        // Button calls, small blind raises: the button must be asked again.
        let state = play(&game, dealt(&game), &[Move::Passive, Move::Raise]);
        assert!(!game.is_terminal(&state));
        assert_eq!(game.current_player(&state), 2, "the big blind first");

        let state = play(&game, state, &[Move::Passive]);
        assert_eq!(game.current_player(&state), 0, "then back to the button");
        assert!(!game.is_terminal(&state));
    }

    #[test]
    fn folding_down_to_one_player_ends_the_hand() {
        let game = ring(3);
        let state = play(&game, dealt(&game), &[Move::Fold, Move::Fold]);
        assert!(game.is_terminal(&state));
        assert_eq!(state.live_seats(), vec![2], "the big blind is left");
    }

    #[test]
    fn folding_with_nothing_owed_is_not_offered() {
        // It is legal at a table and never right, and offering it would split
        // the solver's regret across an action it would never take.
        let game = ring(3);
        let state = play(&game, dealt(&game), &[Move::Passive, Move::Passive]);
        let moves = game.moves(&state);
        assert!(!moves.contains(&Move::Fold), "big blind facing no bet: {moves:?}");
        assert!(moves.contains(&Move::Passive));
    }

    #[test]
    fn the_hand_that_wins_uncontested_wins_exactly_what_the_others_put_in() {
        let game = ring(3);
        let state = play(&game, dealt(&game), &[Move::Fold, Move::Fold]);
        // Button folded having put in nothing; small blind loses its half blind.
        assert_eq!(game.utility_for(&state, 0), 0.0);
        assert_eq!(game.utility_for(&state, 1), -0.5);
        assert_eq!(game.utility_for(&state, 2), 0.5);
    }

    #[test]
    fn payoffs_always_sum_to_zero() {
        // Chips move between players and are never created. Any leak here would
        // teach the solver that some line prints money.
        let game = ring(3);
        let mut rng = Rng::new(7);
        for round in 0..200 {
            let mut state = game.sample_chance(&game.initial(), &mut rng);
            while !game.is_terminal(&state) {
                let count = game.num_actions(&state);
                state = game.apply(&state, rng.below(count as u64) as usize);
            }
            let total: f64 = (0..game.seats()).map(|p| game.utility_for(&state, p)).sum();
            assert!(total.abs() < 1e-9, "round {round}: payoffs sum to {total}");
        }
    }

    #[test]
    fn nobody_can_put_in_more_than_their_stack() {
        let game = ring(3);
        let mut rng = Rng::new(11);
        for _ in 0..200 {
            let mut state = game.sample_chance(&game.initial(), &mut rng);
            while !game.is_terminal(&state) {
                let count = game.num_actions(&state);
                state = game.apply(&state, rng.below(count as u64) as usize);
                for seat in 0..game.seats() {
                    assert!(
                        state.committed(seat) <= 100.0,
                        "seat {seat} committed {}",
                        state.committed(seat)
                    );
                }
            }
        }
    }

    #[test]
    fn every_decision_offers_at_least_two_choices() {
        // A node with one action is not a decision, and one with none is a bug
        // that would hang the walk.
        let game = ring(3);
        let mut rng = Rng::new(13);
        for _ in 0..200 {
            let mut state = game.sample_chance(&game.initial(), &mut rng);
            while !game.is_terminal(&state) {
                let count = game.num_actions(&state);
                assert!(count >= 2, "only {count} action(s) at {state:?}");
                state = game.apply(&state, rng.below(count as u64) as usize);
            }
        }
    }

    /// Heads-up, this must be the same game [`crate::preflop`] models.
    ///
    /// That is the check worth having: the state machine is either right
    /// against an answer already known, or it is wrong somewhere a
    /// three-handed solve would never show up.
    #[test]
    fn heads_up_the_small_blind_folding_loses_exactly_the_small_blind() {
        let game = ring(2);
        let state = play(&game, dealt(&game), &[Move::Fold]);
        assert!(game.is_terminal(&state));
        assert_eq!(game.utility_for(&state, 0), -0.5);
        assert_eq!(game.utility_for(&state, 1), 0.5);
        assert_eq!(game.terminal_utility(&state), -0.5, "the two-player reading agrees");
    }

    #[test]
    fn heads_up_a_limped_pot_still_gives_the_big_blind_its_option() {
        let game = ring(2);
        let state = play(&game, dealt(&game), &[Move::Passive]);
        assert!(!game.is_terminal(&state), "the big blind may still raise");
        assert_eq!(game.current_player(&state), 1);

        let checked = play(&game, state, &[Move::Passive]);
        assert!(game.is_terminal(&checked));
        assert_eq!(checked.committed(0), 1.0, "both in for one blind");
        assert_eq!(checked.committed(1), 1.0);
    }

    #[test]
    fn heads_up_the_button_acts_first_before_the_flop() {
        let game = ring(2);
        assert_eq!(game.current_player(&dealt(&game)), 0);
    }

    /// Solving at two seats should recover the shape of heads-up strategy.
    ///
    /// Not compared hand for hand against [`crate::preflop`]: the two abstract
    /// the tree differently, and sampled training would not agree to the third
    /// decimal even if they abstracted it identically. What must agree is the
    /// direction — the strongest hands raise, the worst fold, and the opening
    /// range comes out a plausible width rather than everything or nothing.
    #[test]
    #[ignore = "slow; run with --ignored"]
    fn solving_two_handed_produces_a_recognisable_opening_range() {
        use crate::cfr::Solver;

        let mut rng = Rng::new(0xF01D);
        let game = Ring::new(
            2,
            100.0,
            Ladder {
                open_to: 2.5,
                three_bet_to: 8.0,
                four_bet_to: 18.0,
            },
            Showdown::new(
                crate::pushfold::EquityTable::sampled_parallel(400, 0x51DE, 4),
                crate::threeway::ThreeWayEquity::sampled_parallel(1, 0x3EED, 4),
            ),
        );
        let mut solver = Solver::new(game);
        solver.train_sampled(400_000, &mut rng);

        let game = solver.game();
        let root = game.initial();
        let mut opened = 0.0;
        let mut counted = 0.0;
        let mut best = 0.0;
        let mut worst = 1.0;
        for class in HandClass::all() {
            let mut classes = [0u8; MAX_SEATS];
            classes[0] = class.index() as u8;
            let state = game.deal_from(&root, classes);
            let moves = game.moves(&state);
            let Some(strategy) = solver.average_strategy(game.info_key(&state)) else {
                continue;
            };
            // Anything other than folding is entering the pot.
            let enters: f64 = moves
                .iter()
                .zip(&strategy)
                .filter(|(m, _)| **m != Move::Fold)
                .map(|(_, p)| *p)
                .sum();
            let weight = class.combos() as f64;
            opened += enters * weight;
            counted += weight;
            if class == HandClass::new(crate::card::Rank::Ace, crate::card::Rank::Ace, false) {
                best = enters;
            }
            if class == HandClass::new(crate::card::Rank::Seven, crate::card::Rank::Two, false) {
                worst = enters;
            }
        }

        let width = opened / counted;
        println!("two-handed opening range: {:.1}% of hands", width * 100.0);
        assert!(best > 0.95, "aces should always play, got {best:.3}");
        assert!(worst < best, "72o should play less than aces");
        assert!(
            (0.35..0.95).contains(&width),
            "an opening range of {:.1}% is not a poker strategy",
            width * 100.0
        );
    }

    /// Two states sharing an information set must offer the same actions.
    ///
    /// The solver stores regrets against the key and trusts the action count to
    /// match, so a collision does not degrade the strategy — it indexes past the
    /// end of somebody else's action list. An earlier packing of this key did
    /// exactly that once the raise count reached five.
    #[test]
    fn no_two_situations_share_a_key_while_offering_different_actions() {
        for seats in [2, 3] {
            let game = ring(seats);
            let mut rng = Rng::new(0xC0111DE);
            let mut seen: std::collections::HashMap<InfoKey, (usize, State)> = Default::default();
            for _ in 0..4_000 {
                let mut state = game.sample_chance(&game.initial(), &mut rng);
                while !game.is_terminal(&state) {
                    let count = game.num_actions(&state);
                    let key = game.info_key(&state);
                    match seen.get(&key) {
                        Some((before, earlier)) => assert_eq!(
                            *before, count,
                            "{seats}-handed: key {key} describes two situations                              offering different actions
  {earlier:?}
  {state:?}"
                        ),
                        None => {
                            seen.insert(key, (count, state));
                        }
                    }
                    state = game.apply(&state, rng.below(count as u64) as usize);
                }
            }
        }
    }

    /// Reports what every solved blueprint actually plays.
    ///
    /// A solve can finish, save, and be worthless: folding everything and
    /// playing everything both produce a tidy file. What says otherwise is the
    /// shape — the first player to act should open tighter as the table grows,
    /// because there are more players left to run into.
    ///
    /// Nothing in the solver was told that. It is a property of the game that
    /// falls out of equity, so finding it in the result is evidence the result
    /// is poker rather than arithmetic that terminated.
    #[test]
    #[ignore = "needs the solved blueprints; run with --ignored --nocapture"]
    fn report_every_solved_opening_range() {
        let mut widths: Vec<(usize, f64)> = Vec::new();
        use crate::blueprint::Blueprint;
        use crate::card::Rank;

        println!();
        for seats in 3..=7usize {
            let path = format!(
                "{}/../../data/ring{seats}-100bb.bin",
                env!("CARGO_MANIFEST_DIR")
            );
            let Ok(blueprint) = Blueprint::load(&path) else {
                println!("{seats}-handed: no blueprint");
                continue;
            };

            // Only the tables needed to build the game; the numbers in them do
            // not matter here, since the blueprint is already solved.
            let mut showdown = Showdown::new(
                crate::pushfold::EquityTable::sampled_parallel(1, 0x51DE, 4),
                crate::threeway::ThreeWayEquity::sampled_parallel(1, 0x3EED, 4),
            );
            for players in 4..=seats {
                let table = format!(
                    "{}/../../data/equity{players}.bin",
                    env!("CARGO_MANIFEST_DIR")
                );
                match crate::wide::WideEquity::load(&table) {
                    Ok(loaded) => showdown = showdown.with(loaded),
                    Err(_) => break,
                }
            }
            if showdown.reach() < seats {
                println!("{seats}-handed: equity tables reach only {}", showdown.reach());
                continue;
            }
            let game = Ring::new(seats, 100.0, Ladder::default(), showdown);

            // The first seat to act, facing nothing but the blinds.
            let root = game.initial();
            let first = game.first_to_act();
            let (mut entered, mut weighed) = (0.0, 0.0);
            let (mut aces, mut worst) = (0.0, 0.0);
            for class in HandClass::all() {
                let mut classes = [0u8; MAX_SEATS];
                classes[first] = class.index() as u8;
                let state = game.deal_from(&root, classes);
                let moves = game.moves(&state);
                let Some(strategy) = blueprint.strategy(game.info_key(&state)) else {
                    continue;
                };
                let plays: f64 = moves
                    .iter()
                    .zip(strategy)
                    .filter(|(m, _)| **m != Move::Fold)
                    .map(|(_, p)| *p as f64)
                    .sum();
                let weight = class.combos() as f64;
                entered += plays * weight;
                weighed += weight;
                if class == HandClass::new(Rank::Ace, Rank::Ace, false) {
                    aces = plays;
                }
                if class == HandClass::new(Rank::Seven, Rank::Two, false) {
                    worst = plays;
                }
            }
            if weighed == 0.0 {
                println!("{seats}-handed: no information sets reached");
                continue;
            }
            let width = entered / weighed * 100.0;
            widths.push((seats, width));
            println!(
                "{seats}-handed  first to act plays {width:5.1}%   AA {:3.0}%   72o {:3.0}%",
                aces * 100.0,
                worst * 100.0
            );
            assert!(aces > 0.95, "{seats}-handed: aces should always play");
            assert!(worst < aces, "{seats}-handed: 72o should play less than aces");
            assert!(
                (2.0..60.0).contains(&width),
                "{seats}-handed: an opening range of {width:.1}% is not a strategy"
            );
        }

        // The property nobody encoded: more opponents, tighter opening.
        for pair in widths.windows(2) {
            let ((fewer, wider), (more, tighter)) = (pair[0], pair[1]);
            assert!(
                tighter < wider,
                "{more}-handed opens {tighter:.1}%, which is wider than {fewer}-handed                  at {wider:.1}% — a bigger table should open tighter"
            );
        }
    }

    #[test]
    fn a_table_too_wide_for_the_equity_on_hand_is_refused() {
        assert_eq!(showdown().reach(), 3, "no wide tables were supplied");
        let built = std::panic::catch_unwind(|| {
            Ring::new(4, 100.0, Ladder::default(), showdown())
        });
        assert!(built.is_err(), "four-handed needs four-way equity");
    }

    #[test]
    fn a_showdown_given_a_wide_table_reaches_further() {
        use crate::wide::{Buckets, WideEquity};
        let equity = crate::pushfold::EquityTable::sampled_parallel(64, 0x51DE, 4);
        let wide = WideEquity::sampled_parallel(4, Buckets::by_strength(&equity, 6), 4, 0x4EED, 4);
        let reaching = showdown().with(wide);
        assert_eq!(reaching.reach(), 4);
        // And the tree accepts the seat count it can now price.
        let game = Ring::new(4, 100.0, Ladder::default(), reaching);
        assert_eq!(game.seats(), 4);
    }
}
