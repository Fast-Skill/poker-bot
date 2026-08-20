//! Heads-up postflop hold'em, from the flop to showdown.
//!
//! This is the part the bot has been missing. Preflop it plays a solved
//! strategy; after the flop it has been guessing, and a live watch showed what
//! that guessing costs — calling twenty-eight big blinds with bottom pair,
//! because a heuristic with no model of the pot cannot tell a good price from a
//! terrible one.
//!
//! # Heads-up first, deliberately
//!
//! Most pots that see a flop are contested by two players, because preflop
//! betting is what thins the field. Multiway postflop is a much larger problem
//! and a rarer one, so this solves the common case properly rather than every
//! case badly.
//!
//! # What a hand is, here
//!
//! Not two cards — a strength group on a sampled board, from
//! [`crate::texture`]. Strength is measured street by street from the cards
//! visible at each, so a hand that will make a flush by the river is not rated
//! as though it already had. What the board is doing to a hand is therefore
//! part of the abstraction rather than absent from it.
//!
//! Showdowns are the exception: those are settled from the finished hands, not
//! from groups, because two hands in one group are not the same hand.

use crate::betting::Street;
use crate::cfr::{Game, InfoKey};
use crate::rng::Rng;
use crate::texture::{Textures, HOLDINGS};

/// Bet sizes, as fractions of the pot after calling.
///
/// Two of them. A larger menu makes a better strategy in principle and a worse
/// one in practice at this budget: every extra size multiplies the tree, and
/// regret spread thinly across many sizes converges more slowly than regret
/// concentrated on a few.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sizing {
    pub small: f64,
    pub large: f64,
}

impl Default for Sizing {
    /// A third of the pot and three-quarters of it, which is close to the
    /// presets the client itself offers.
    fn default() -> Sizing {
        Sizing {
            small: 0.33,
            large: 0.75,
        }
    }
}

/// What a player may do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Move {
    Fold,
    /// Check when nothing is owed, call when something is.
    Passive,
    Small,
    Large,
    Jam,
}

impl Move {
    pub fn name(&self) -> &'static str {
        match self {
            Move::Fold => "fold",
            Move::Passive => "check/call",
            Move::Small => "bet small",
            Move::Large => "bet large",
            Move::Jam => "all in",
        }
    }
}

/// A postflop situation, described the way a table describes one.
///
/// # Why this exists
///
/// The solve is a tree, and a tree node knows things a table never says: whose
/// holding is which index into which sampled board, how much of the stack the
/// hand has spent, which node it descended from. A real table offers a
/// different set of facts — a pot, a bet to call, what is behind, whose turn it
/// is. Both have to arrive at the same information key, or the bot looks up a
/// strategy solved for a spot it is not in — the failure that looks most like
/// working.
///
/// So the key is built from this and only this, and both sides fill it in: the
/// tree from its state, the table from what it can see. The test
/// `a_spot_taken_from_the_tree_keys_the_same_as_the_tree` holds them together.
///
/// Amounts may be in whatever unit the caller counts in — chips, blinds,
/// anything — because every use of them here is a ratio. What must not happen
/// is two different units inside one spot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Spot {
    pub street: Street,
    /// Which side is acting: 0 acts last on every street, 1 acts first.
    pub player: usize,
    /// The acting player's strength group, from the same abstraction the solve
    /// was trained on.
    pub strength: u8,
    /// Everything in the middle, this street's wagers included.
    pub pot: u64,
    /// The largest wager anyone has made this street.
    pub bet: u64,
    /// The acting player's own wager this street.
    pub mine: u64,
    /// What the acting player has left behind.
    pub behind: u64,
    /// What the other player has left behind.
    ///
    /// Only ever asked one question: whether there is anyone left to respond to
    /// a raise. Betting into a player who is already all in cannot win a chip —
    /// the excess comes straight back — and a real table will not accept the
    /// action at all.
    ///
    /// It takes unequal stacks to reach, which is why the solve itself never
    /// does: every seat there starts with the same amount, so a player who is
    /// all in has by then put in at least as much as the other, and no raise is
    /// offered anyway. A real table is not like that. An opponent sitting with
    /// thirty blinds against the hero's hundred can be all in for a third of
    /// what the hero still has behind, and without this the tree would offer a
    /// raise the client refuses.
    pub opponent_behind: u64,
    /// Raises made this street. A first bet counts as the first raise.
    pub raises: u8,
    /// Bit per player, set when they have acted since the last raise.
    pub acted: u8,
}

impl Spot {
    /// What calling costs.
    fn owed(&self) -> u64 {
        self.bet.saturating_sub(self.mine)
    }
}

/// How many raises a street allows before the only move left is all-in.
const RAISE_CAP: u8 = 3;

/// A node in the tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct State {
    /// Which sampled board this hand is being played on.
    board: u32,
    /// Each player's holding, as an index into that board's holdings.
    holdings: [u16; 2],
    street: Street,
    /// Chips each has put in on the current street.
    wagered: [u32; 2],
    /// Chips each has put in on earlier streets of this hand.
    ///
    /// Tracked per player as well as in the pot, because what a player has left
    /// behind depends on their own outlay rather than on the pot's size.
    spent: [u32; 2],
    /// Chips already in the pot from earlier streets, including preflop.
    settled: u32,
    to_act: u8,
    /// Bit per player: has acted since the last raise on this street.
    acted: u8,
    raises: u8,
    /// Who folded, if anyone.
    folded: Option<u8>,
    dealt: bool,
}

impl State {
    pub fn street(&self) -> Street {
        self.street
    }

    /// Chips in the middle, including this street's wagers.
    pub fn pot(&self) -> u32 {
        self.settled + self.wagered[0] + self.wagered[1]
    }

    pub fn holding(&self, player: usize) -> usize {
        self.holdings[player] as usize
    }

    pub fn board(&self) -> usize {
        self.board as usize
    }
}

/// Heads-up postflop, at one starting pot and stack depth.
///
/// Player 0 acts last on every street — the button. Player 1 acts first, which
/// is what being out of position means and most of why position is worth
/// anything.
#[derive(Debug, Clone)]
pub struct Postflop {
    /// The sampled boards strength was measured on.
    ///
    /// Only the solver reads this. Every question the bot asks — which moves
    /// exist, what they cost, which information set this is — is answered from
    /// a [`Spot`], whose strength has already been read off the real board. So
    /// a tree built for play carries none, and five rungs cost kilobytes rather
    /// than three-quarters of a gigabyte.
    textures: Option<Textures>,
    buckets: usize,
    /// Chips in the pot when the flop is dealt.
    pot: u32,
    /// Chips each player has behind at that moment.
    stack: u32,
    sizing: Sizing,
}

impl Postflop {
    pub fn new(textures: Textures, pot: u32, stack: u32, sizing: Sizing) -> Postflop {
        assert!(pot > 0, "a hand that reaches a flop has something in the pot");
        assert!(stack > 0, "a player with nothing behind has no decisions left");
        Postflop {
            buckets: textures.buckets(),
            textures: Some(textures),
            pot,
            stack,
            sizing,
        }
    }

    /// The same tree, for a bot rather than a solver.
    ///
    /// Carries no board sample. A bot reads strength off the board in front of
    /// it — see [`crate::texture::Reader`] — and hands it over in a [`Spot`],
    /// so the sample the solve was trained on is of no further use once the
    /// solve exists.
    ///
    /// `buckets` must be the number the solve was trained with, and `pot` and
    /// `stack` the ones it was solved at. None of the three changes an answer
    /// here, but a tree that misreports what it is would route a bot to the
    /// wrong rung.
    pub fn for_play(buckets: usize, pot: u32, stack: u32, sizing: Sizing) -> Postflop {
        Postflop {
            textures: None,
            buckets,
            pot,
            stack,
            sizing,
        }
    }

    /// How many groups hand strength is cut into.
    pub fn buckets(&self) -> usize {
        self.buckets
    }

    /// The board sample this tree solves from.
    ///
    /// # Panics
    /// Panics if this tree was built for play. Every caller is inside the
    /// solver, which cannot run without one, so the alternative is threading a
    /// `Result` through the training loop to describe a state no solve can be
    /// in.
    fn sample(&self) -> &Textures {
        self.textures
            .as_ref()
            .expect("this tree was built for play and has no board sample to solve from")
    }

    /// The board sample, if this tree carries one. Trees built for play do not.
    pub fn textures(&self) -> Option<&Textures> {
        self.textures.as_ref()
    }

    /// What a player still has behind.
    fn behind(&self, state: &State, player: usize) -> u32 {
        // Everything wagered on earlier streets is already inside `settled`,
        // so the stack is tracked by subtracting the whole hand's outlay.
        self.stack
            .saturating_sub(state.spent[player])
            .saturating_sub(state.wagered[player])
    }

    /// The largest wager on this street.
    fn owed(&self, state: &State) -> u32 {
        state.wagered[0].max(state.wagered[1])
    }

    /// Whether the betting on this street has finished.
    fn street_closed(&self, state: &State) -> bool {
        if state.folded.is_some() {
            return true;
        }
        let owed = self.owed(state);
        (0..2).all(|player| {
            self.behind(state, player) == 0
                || (state.acted & (1 << player) != 0 && state.wagered[player] == owed)
        })
    }

    /// What a raise to this fraction of the pot commits the raiser to.
    fn raise_to(&self, state: &State, fraction: f64) -> u32 {
        let owed = self.owed(state);
        let mine = state.wagered[state.to_act as usize];
        // The pot as it would stand after calling, which is what a bet sized
        // "in the pot" is conventionally measured against.
        let after_call = state.pot() + (owed - mine);
        owed + (after_call as f64 * fraction).round() as u32
    }

    /// What a move commits the acting player to this street.
    ///
    /// In the spot's own units, so a caller working in table chips gets chips
    /// back. Kept here rather than in the bot because the solve's idea of a
    /// size and the bot's have to be the same idea: a blueprint that said "bet
    /// small" meant this amount, and betting a different one plays a strategy
    /// nobody solved for.
    ///
    /// Folding and checking commit nothing beyond what is already in, so both
    /// answer with what is already in.
    pub fn target_of(&self, spot: &Spot, chosen: Move) -> u64 {
        let after_call = spot.pot + spot.owed();
        match chosen {
            Move::Fold => spot.mine,
            Move::Passive => spot.bet.min(spot.mine + spot.behind),
            Move::Small | Move::Large => {
                let fraction = if chosen == Move::Small {
                    self.sizing.small
                } else {
                    self.sizing.large
                };
                let target = spot.bet + (after_call as f64 * fraction).round() as u64;
                target.min(spot.mine + spot.behind)
            }
            Move::Jam => spot.mine + spot.behind,
        }
    }

    /// The sizes this tree bets in.
    pub fn sizing(&self) -> Sizing {
        self.sizing
    }

    /// The moves available here, in a fixed order so action indices are stable.
    fn moves(&self, state: &State) -> Vec<Move> {
        self.moves_at(&self.spot_of(state, 0))
    }

    /// The moves available in a situation, however it came to be described.
    ///
    /// Public because the bot needs it: knowing which action a blueprint
    /// probability belongs to means knowing this list, and the bot has a table
    /// in front of it rather than a tree node.
    pub fn moves_at(&self, spot: &Spot) -> Vec<Move> {
        let owed = spot.owed();
        let mut moves = Vec::with_capacity(5);

        // Folding with nothing owed is legal and never right; offering it would
        // split regret across an action no strategy would take.
        if owed > 0 {
            moves.push(Move::Fold);
        }
        moves.push(Move::Passive);

        // Raising is only a move if somebody can answer it.
        let contested = spot.opponent_behind > 0;

        if contested && spot.raises < RAISE_CAP {
            // The pot as it would stand after calling, which is what a bet
            // sized "in the pot" is conventionally measured against.
            let after_call = spot.pot + owed;
            for (fraction, sized) in [
                (self.sizing.small, Move::Small),
                (self.sizing.large, Move::Large),
            ] {
                let target = spot.bet + (after_call as f64 * fraction).round() as u64;
                // A size at or beyond the stack is a jam under another name.
                if target > spot.bet && target < spot.mine + spot.behind {
                    moves.push(sized);
                }
            }
        }
        // Jamming is only a distinct action if it puts in more than calling.
        if contested && spot.mine + spot.behind > spot.bet {
            moves.push(Move::Jam);
        }
        moves
    }

    /// How a tree node describes itself as a situation.
    pub fn spot_of(&self, state: &State, strength: u8) -> Spot {
        let player = state.to_act as usize;
        Spot {
            street: state.street,
            player,
            strength,
            pot: state.pot() as u64,
            bet: self.owed(state) as u64,
            mine: state.wagered[player] as u64,
            behind: self.behind(state, player) as u64,
            opponent_behind: self.behind(state, 1 - player) as u64,
            raises: state.raises,
            acted: state.acted,
        }
    }

    /// The information key for a situation, however it came to be described.
    ///
    /// This is the whole of the mapping between a solved strategy and a table.
    /// What a player can see: their own strength, the street, whose turn it is,
    /// how much betting has happened, what calling costs against the pot, how
    /// deep the pot is against what is behind, and which moves are on offer.
    pub fn key_at(&self, spot: &Spot) -> InfoKey {
        let behind = spot.behind.max(1);
        let pot = spot.pot.max(1);

        // The price of continuing, which is most of what decides a call.
        let price = ((spot.owed() as f64 / pot as f64) * 6.0).round().min(7.0) as u64;
        // How much is behind relative to the pot, which decides how much room
        // there is to play for. The same hand plays differently for a tenth of
        // a stack than for all of it.
        let depth = ((pot as f64 / behind as f64).log2().max(0.0) as u64).min(7);

        // Which kinds of move exist here, stated rather than inferred.
        //
        // Two situations that offer different actions must never share a key:
        // the solver stores its regrets against the key and trusts the count to
        // match, so a collision indexes past the end of another node's actions.
        // Facing an all-in and facing a small bet collided on everything else,
        // being alike in street, price band and depth while offering two moves
        // and five.
        //
        // One bit per optional move. Checking in the round — "are there any
        // sizes" — was not enough: having only the small size looked identical
        // to having both, and the two offer different numbers of actions.
        // Checking in the flat is exact, since checking is always available and
        // everything else is named here.
        let moves = self.moves_at(spot);
        let shape = [Move::Fold, Move::Small, Move::Large, Move::Jam]
            .iter()
            .enumerate()
            .map(|(bit, wanted)| (moves.contains(wanted) as u64) << bit)
            .fold(0, |mask, bit| mask | bit);

        let mut key = street_index(spot.street) as u64;
        key = (key << 1) | spot.player as u64;
        key = (key << 3) | spot.raises.min(7) as u64;
        key = (key << 2) | spot.acted as u64;
        key = (key << 3) | price;
        key = (key << 3) | depth;
        key = (key << 4) | shape;
        (key << 8) | spot.strength as u64
    }

    /// Moves the hand to the next street, or to showdown after the river.
    fn advance(&self, state: &State) -> State {
        let mut next = *state;
        for player in 0..2 {
            next.spent[player] += next.wagered[player];
        }
        next.settled = state.pot();
        next.wagered = [0, 0];
        next.acted = 0;
        next.raises = 0;
        next.street = match state.street {
            Street::Flop => Street::Turn,
            Street::Turn => Street::River,
            // The river has no next street; the caller checks for that.
            other => other,
        };
        // Out of position acts first on every postflop street.
        next.to_act = 1;
        next
    }

    /// Whether both players are all in, so the rest is dealt out.
    fn all_in(&self, state: &State) -> bool {
        (0..2).all(|player| self.behind(state, player) == 0)
    }
}

impl Game for Postflop {
    type State = State;

    fn initial(&self) -> State {
        State {
            board: 0,
            holdings: [0, 1],
            street: Street::Flop,
            wagered: [0, 0],
            spent: [0, 0],
            settled: self.pot,
            to_act: 1,
            acted: 0,
            raises: 0,
            folded: None,
            dealt: false,
        }
    }

    fn is_terminal(&self, state: &State) -> bool {
        if !state.dealt {
            return false;
        }
        if state.folded.is_some() {
            return true;
        }
        // All in with cards to come is terminal too: nobody has a decision left
        // and the rest is a formality the showdown already accounts for.
        self.street_closed(state) && (state.street == Street::River || self.all_in(state))
    }

    fn terminal_utility(&self, state: &State) -> f64 {
        let staked = |player: usize| (state.spent[player] + state.wagered[player]) as f64;
        match state.folded {
            Some(0) => -staked(0),
            Some(1) => staked(1),
            Some(_) => unreachable!("only two players can fold"),
            None => {
                // Matched by definition — betting closed with both live — so
                // the winner takes what the loser put in.
                let at_risk = staked(0).min(staked(1));
                match self
                    .sample()
                    .showdown(
                    state.board as usize,
                    state.holdings[0] as usize,
                    state.holdings[1] as usize,
                ) {
                    std::cmp::Ordering::Greater => at_risk,
                    std::cmp::Ordering::Less => -at_risk,
                    std::cmp::Ordering::Equal => 0.0,
                }
            }
        }
    }

    fn is_chance(&self, state: &State) -> bool {
        !state.dealt
    }

    fn enumerable(&self) -> bool {
        false
    }

    /// Every board and holding pair, which is far too many to enumerate.
    ///
    /// A thousand boards times a million holding pairs is not a distribution
    /// anyone can write down, so this game is trained by sampling only.
    fn chance_outcomes(&self, _state: &State) -> Vec<(State, f64)> {
        unimplemented!("postflop has too many deals to enumerate; train by sampling")
    }

    fn sample_chance(&self, state: &State, rng: &mut Rng) -> State {
        let sample = self.sample();
        let board = rng.below(sample.len() as u64) as usize;
        let holdings = sample.holdings(board);

        // Two holdings that do not share a card, since both cannot hold it.
        let first = rng.below(HOLDINGS as u64) as usize;
        let mut second = first;
        for _ in 0..64 {
            let candidate = rng.below(HOLDINGS as u64) as usize;
            let clash = holdings[candidate]
                .iter()
                .any(|card| holdings[first].contains(card));
            if !clash {
                second = candidate;
                break;
            }
        }

        State {
            board: board as u32,
            holdings: [first as u16, second as u16],
            dealt: true,
            ..*state
        }
    }

    fn current_player(&self, state: &State) -> usize {
        state.to_act as usize
    }

    fn info_key(&self, state: &State) -> InfoKey {
        let player = state.to_act as usize;
        let strength = self
            .sample()
            .strength(
                state.board as usize,
                state.street,
                state.holdings[player] as usize,
            )
            .expect("postflop streets are bucketed");
        self.key_at(&self.spot_of(state, strength))
    }

    fn num_actions(&self, state: &State) -> usize {
        self.moves(state).len()
    }

    fn apply(&self, state: &State, action: usize) -> State {
        let player = state.to_act as usize;
        let chosen = self.moves(state)[action];
        let owed = self.owed(state);
        let behind = self.behind(state, player);
        let mut next = *state;

        match chosen {
            Move::Fold => next.folded = Some(player as u8),
            Move::Passive => {
                next.wagered[player] = owed.min(state.wagered[player] + behind);
            }
            Move::Small | Move::Large => {
                let fraction = if chosen == Move::Small {
                    self.sizing.small
                } else {
                    self.sizing.large
                };
                next.wagered[player] = self.raise_to(state, fraction).min(state.wagered[player] + behind);
                next.raises += 1;
                next.acted = 0;
            }
            Move::Jam => {
                next.wagered[player] = state.wagered[player] + behind;
                if next.wagered[player] > owed {
                    next.raises += 1;
                    next.acted = 0;
                }
            }
        }
        next.acted |= 1 << player;

        if self.street_closed(&next) {
            let finished = next.folded.is_some()
                || next.street == Street::River
                || self.all_in(&next);
            if !finished {
                return self.advance(&next);
            }
        } else {
            next.to_act = 1 - player as u8;
        }
        next
    }
}

fn street_index(street: Street) -> usize {
    match street {
        Street::Preflop => 0,
        Street::Flop => 1,
        Street::Turn => 2,
        Street::River => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn game() -> Postflop {
        // Ten big blinds in, a hundred behind, on a small sample of boards.
        Postflop::new(
            Textures::sample(8, 10, 0x7E47, 4),
            1_000,
            10_000,
            Sizing::default(),
        )
    }

    fn dealt(game: &Postflop) -> State {
        let mut rng = Rng::new(5);
        game.sample_chance(&game.initial(), &mut rng)
    }

    fn play(game: &Postflop, mut state: State, moves: &[Move]) -> State {
        for wanted in moves {
            let available = game.moves(&state);
            let index = available
                .iter()
                .position(|m| m == wanted)
                .unwrap_or_else(|| panic!("{wanted:?} not among {available:?}"));
            state = game.apply(&state, index);
        }
        state
    }

    /// A raise is not offered when there is nobody left to answer it.
    ///
    /// Constructed rather than played, because a table where everyone starts
    /// with the same stack cannot produce it: by the time one player is all in
    /// they have put in at least as much as the other, and the sizes fall
    /// outside what is left anyway. Unequal stacks are ordinary at a real
    /// table, and that is where this matters.
    #[test]
    fn nobody_raises_into_a_player_who_is_already_all_in() {
        let game = game();
        // The hero has plenty behind; the opponent has shoved and has nothing.
        let facing_a_shove = Spot {
            street: Street::Flop,
            player: 0,
            strength: 5,
            pot: 260,
            bet: 200,
            mine: 20,
            behind: 400,
            opponent_behind: 0,
            raises: 1,
            acted: 0b10,
        };
        assert_eq!(
            game.moves_at(&facing_a_shove),
            vec![Move::Fold, Move::Passive],
            "there is nothing to do against a shove but pay it or not"
        );

        // The same spot with the opponent still holding chips is a real
        // decision, which is what makes the difference above meaningful.
        let contested = Spot {
            opponent_behind: 300,
            ..facing_a_shove
        };
        let moves = game.moves_at(&contested);
        assert!(
            moves.contains(&Move::Jam),
            "with chips behind on both sides a raise is a move; got {moves:?}"
        );
    }

    #[test]
    fn out_of_position_acts_first_on_every_street() {
        let game = game();
        let state = dealt(&game);
        assert_eq!(game.current_player(&state), 1, "on the flop");

        // Checked through to the turn.
        let turn = play(&game, state, &[Move::Passive, Move::Passive]);
        assert_eq!(turn.street(), Street::Turn);
        assert_eq!(game.current_player(&turn), 1, "and again on the turn");
    }

    #[test]
    fn checking_through_reaches_showdown_with_the_pot_untouched() {
        let game = game();
        let state = play(
            &game,
            dealt(&game),
            &[
                Move::Passive, Move::Passive, // flop
                Move::Passive, Move::Passive, // turn
                Move::Passive, Move::Passive, // river
            ],
        );
        assert!(game.is_terminal(&state));
        assert_eq!(state.pot(), 1_000, "nobody bet, so the pot is what it was");
    }

    #[test]
    fn folding_with_nothing_owed_is_not_offered() {
        let game = game();
        let moves = game.moves(&dealt(&game));
        assert!(!moves.contains(&Move::Fold), "facing no bet: {moves:?}");
        assert!(moves.contains(&Move::Passive));
    }

    #[test]
    fn a_bet_can_be_folded_to_and_the_bettor_takes_the_pot() {
        let game = game();
        let state = play(&game, dealt(&game), &[Move::Large, Move::Fold]);
        assert!(game.is_terminal(&state));
        // Player 1 bet and player 0 folded, so player 1 wins what 0 put in,
        // which on a fold before calling is nothing.
        assert_eq!(game.terminal_utility(&state), 0.0);
    }

    #[test]
    fn a_bet_reopens_the_action_for_the_other_player() {
        let game = game();
        let state = play(&game, dealt(&game), &[Move::Passive, Move::Large]);
        assert!(!game.is_terminal(&state));
        assert_eq!(game.current_player(&state), 1, "back to the bettor's opponent");
    }

    #[test]
    fn payoffs_are_zero_sum_and_bounded_by_the_stack() {
        // Chips move between players and are never created, and nobody can lose
        // more than they brought.
        let game = game();
        let mut rng = Rng::new(21);
        for round in 0..400 {
            let mut state = game.sample_chance(&game.initial(), &mut rng);
            while !game.is_terminal(&state) {
                let count = game.num_actions(&state);
                assert!(count >= 2, "only {count} action(s) at {state:?}");
                state = game.apply(&state, rng.below(count as u64) as usize);
            }
            let hero = game.utility_for(&state, 0);
            let villain = game.utility_for(&state, 1);
            assert!(
                (hero + villain).abs() < 1e-9,
                "round {round}: {hero} and {villain} do not sum to zero"
            );
            assert!(
                hero.abs() <= 11_000.0,
                "round {round}: {hero} exceeds what anyone brought"
            );
        }
    }

    #[test]
    fn nobody_wagers_more_than_they_have() {
        let game = game();
        let mut rng = Rng::new(33);
        for _ in 0..400 {
            let mut state = game.sample_chance(&game.initial(), &mut rng);
            while !game.is_terminal(&state) {
                let count = game.num_actions(&state);
                state = game.apply(&state, rng.below(count as u64) as usize);
                for player in 0..2 {
                    assert!(
                        state.spent[player] + state.wagered[player] <= 10_000,
                        "player {player} has put in more than their stack: {state:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_two_players_never_hold_the_same_card() {
        let game = game();
        let mut rng = Rng::new(41);
        for _ in 0..500 {
            let state = game.sample_chance(&game.initial(), &mut rng);
            let holdings = game
                .textures()
                .expect("a tree solved from a sample carries it")
                .holdings(state.board());
            let (mine, theirs) = (holdings[state.holding(0)], holdings[state.holding(1)]);
            assert!(
                !mine.iter().any(|card| theirs.contains(card)),
                "{mine:?} and {theirs:?} share a card"
            );
        }
    }

    #[test]
    fn no_two_situations_share_a_key_while_offering_different_actions() {
        // The solver caches the action count against the key, so a collision
        // does not blur a strategy — it indexes past the end of another one.
        let game = game();
        let mut rng = Rng::new(0xC0FFEE);
        let mut seen: std::collections::HashMap<InfoKey, usize> = Default::default();
        for _ in 0..3_000 {
            let mut state = game.sample_chance(&game.initial(), &mut rng);
            while !game.is_terminal(&state) {
                let count = game.num_actions(&state);
                let key = game.info_key(&state);
                match seen.get(&key) {
                    Some(&before) => assert_eq!(before, count, "key {key} at {state:?}"),
                    None => {
                        seen.insert(key, count);
                    }
                }
                state = game.apply(&state, rng.below(count as u64) as usize);
            }
        }
    }
}
