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
    /// Which seat is acting. Seat 0 acts last on every street; the rest act
    /// before it in ascending order, so seat 1 leads and the highest-numbered
    /// seat is the last of them.
    pub player: usize,
    /// How many players are still holding cards, the actor included.
    ///
    /// A three-way pot that folds to two is not the same as a pot that started
    /// with two: the money already in the middle is larger relative to the
    /// stacks, and the ranges that got there are different. Keying on this
    /// keeps those apart instead of averaging them together.
    pub live: u8,
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
    /// The most any live opponent has left behind.
    ///
    /// Only ever asked one question: whether there is anyone left to respond to
    /// a raise. With more than one opponent it is the largest of them, since a
    /// raise is answerable if *anybody* can answer it. Betting into a player who is already all in cannot win a chip —
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
    pub fn owed(&self) -> u64 {
        self.bet.saturating_sub(self.mine)
    }
}

/// How an information key is packed, least significant field first.
///
/// Named rather than written inline because two things read the key: the
/// packing below, and the search for a nearby one when a solve has never met
/// the exact spot. If those two ever disagreed about where a field sits, the
/// search would substitute a strategy from somewhere else entirely.
const STRENGTH_BITS: u32 = 8;
const SHAPE_BITS: u32 = 4;
const DEPTH_BITS: u32 = 3;
const PRICE_BITS: u32 = 3;

const DEPTH_SHIFT: u32 = STRENGTH_BITS + SHAPE_BITS;
const PRICE_SHIFT: u32 = DEPTH_SHIFT + DEPTH_BITS;

/// How many raises a street allows before the only move left is all-in.
const RAISE_CAP: u8 = 3;

/// The most players one postflop tree models.
pub const MAX_PLAYERS: usize = 3;

/// Set on every key belonging to a multiway tree.
///
/// # Why the two key spaces are kept apart
///
/// A multiway spot needs fields a heads-up one does not — which of three seats
/// is acting, how many are still in — and there is no room for them inside the
/// heads-up layout without shifting every field along. Shifting them would
/// renumber every key in the solved ladder already on disk, which is the one
/// change that breaks silently: the file still loads, lookups still succeed,
/// and each one returns the strategy for a different situation.
///
/// So heads-up keeps its layout untouched and multiway gets its own, marked by
/// a bit no heads-up key ever sets. The two cannot collide, and a blueprint
/// trained on one is visibly not the other.
const MULTIWAY: u64 = 1 << 40;

/// A node in the tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct State {
    /// Which sampled board this hand is being played on.
    board: u32,
    /// Each player's holding, as an index into that board's holdings.
    holdings: [u16; MAX_PLAYERS],
    street: Street,
    /// Chips each has put in on the current street.
    wagered: [u32; MAX_PLAYERS],
    /// Chips each has put in on earlier streets of this hand.
    ///
    /// Tracked per player as well as in the pot, because what a player has left
    /// behind depends on their own outlay rather than on the pot's size.
    spent: [u32; MAX_PLAYERS],
    /// Chips already in the pot from earlier streets, including preflop.
    settled: u32,
    to_act: u8,
    /// Bit per player: has acted since the last raise on this street.
    acted: u8,
    raises: u8,
    /// Bit per player, set when they have folded.
    ///
    /// A bitmask rather than "who folded", because with three players one fold
    /// does not end the hand and the second one has to be recorded beside the
    /// first.
    folded: u8,
    dealt: bool,
}

impl State {
    pub fn street(&self) -> Street {
        self.street
    }

    /// Chips in the middle, including this street's wagers.
    pub fn pot(&self) -> u32 {
        self.settled + self.wagered.iter().sum::<u32>()
    }

    /// Whether a player is still holding cards.
    pub fn is_live(&self, player: usize) -> bool {
        self.folded & (1 << player) == 0
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
    /// How many seats this tree deals to.
    players: usize,
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
            players: 2,
            pot,
            stack,
            sizing,
        }
    }

    /// The same tree dealt to more seats.
    ///
    /// # Why this is a separate constructor
    ///
    /// Three-way postflop is the largest hole in this bot: seven of every
    /// twelve spots it cannot answer are pots with more than two players in
    /// them, and they are the spots where a mistake costs a stack rather than a
    /// bet. Nothing published covers them — board texture makes charts
    /// impractical after the flop — so unlike preflop there is nothing to read
    /// and it has to be solved.
    ///
    /// The heads-up tree is left exactly as it was, down to the numbering of
    /// its information keys, because a solved ladder trained against those
    /// numbers is already on disk.
    ///
    /// # Panics
    /// Panics outside two to [`MAX_PLAYERS`] seats.
    pub fn multiway(
        players: usize,
        textures: Textures,
        pot: u32,
        stack: u32,
        sizing: Sizing,
    ) -> Postflop {
        assert!(
            (2..=MAX_PLAYERS).contains(&players),
            "a postflop tree holds two to {MAX_PLAYERS} seats, not {players}"
        );
        Postflop {
            players,
            ..Postflop::new(textures, pot, stack, sizing)
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
            players: 2,
            pot,
            stack,
            sizing,
        }
    }

    /// A tree for play with more than two seats.
    ///
    /// # Panics
    /// Panics outside two to [`MAX_PLAYERS`] seats.
    pub fn multiway_for_play(
        players: usize,
        buckets: usize,
        pot: u32,
        stack: u32,
        sizing: Sizing,
    ) -> Postflop {
        assert!(
            (2..=MAX_PLAYERS).contains(&players),
            "a postflop tree holds two to {MAX_PLAYERS} seats, not {players}"
        );
        Postflop {
            players,
            ..Postflop::for_play(buckets, pot, stack, sizing)
        }
    }

    /// How many seats this tree deals to.
    pub fn players(&self) -> usize {
        self.players
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
    ///
    /// Only live players count. A folded player's chips stay in the pot but
    /// stop being a price anyone has to match, and counting them would leave
    /// the survivors owing money to somebody who has given up the hand.
    fn owed(&self, state: &State) -> u32 {
        (0..self.players)
            .filter(|&player| state.is_live(player))
            .map(|player| state.wagered[player])
            .max()
            .unwrap_or(0)
    }

    /// How many players still hold cards.
    fn live_count(&self, state: &State) -> usize {
        (0..self.players)
            .filter(|&player| state.is_live(player))
            .count()
    }

    /// Whether the betting on this street has finished.
    fn street_closed(&self, state: &State) -> bool {
        if self.live_count(state) <= 1 {
            return true;
        }
        let owed = self.owed(state);
        (0..self.players).filter(|&p| state.is_live(p)).all(|player| {
            self.behind(state, player) == 0
                || (state.acted & (1 << player) != 0 && state.wagered[player] == owed)
        })
    }

    /// The next seat with a decision, going round from `from`.
    ///
    /// Seat 0 acts last, so the order runs 1, 2, ... and then back to 0.
    /// Players who have folded or have nothing behind are stepped over.
    fn next_actor(&self, state: &State, from: usize) -> Option<usize> {
        (1..=self.players)
            .map(|step| (from + step) % self.players)
            .find(|&seat| state.is_live(seat) && self.behind(state, seat) > 0)
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
            live: self.live_count(state) as u8,
            strength,
            pot: state.pot() as u64,
            bet: self.owed(state) as u64,
            mine: state.wagered[player] as u64,
            behind: self.behind(state, player) as u64,
            // The deepest live opponent, since a raise is answerable if
            // anybody can answer it.
            opponent_behind: (0..self.players)
                .filter(|&seat| seat != player && state.is_live(seat))
                .map(|seat| self.behind(state, seat) as u64)
                .max()
                .unwrap_or(0),
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
        if self.players > 2 {
            // A wider layout, kept apart from the heads-up one by `MULTIWAY`
            // so the ladder already solved for two players keeps its numbering.
            key = (key << 2) | spot.player.min(3) as u64;
            key = (key << 2) | u64::from(spot.live.min(3));
            key = (key << 3) | spot.raises.min(7) as u64;
            key = (key << 3) | u64::from(spot.acted & 0b111);
            key = (key << PRICE_BITS) | price;
            key = (key << DEPTH_BITS) | depth;
            key = (key << SHAPE_BITS) | shape;
            return MULTIWAY | (key << STRENGTH_BITS) | spot.strength as u64;
        }
        key = (key << 1) | spot.player as u64;
        key = (key << 3) | spot.raises.min(7) as u64;
        key = (key << 2) | spot.acted as u64;
        key = (key << PRICE_BITS) | price;
        key = (key << DEPTH_BITS) | depth;
        key = (key << SHAPE_BITS) | shape;
        (key << STRENGTH_BITS) | spot.strength as u64
    }

    /// Keys worth trying for a spot, nearest first.
    ///
    /// # Why the exact key is often missing
    ///
    /// A solve only ever faces its own two bet sizes, so the only prices it
    /// ever has to answer are the ones a third-pot and a three-quarter-pot bet
    /// produce. A real opponent bets whatever they like. Their half-pot bet
    /// lands in a price band the solve was never trained on, and an exact
    /// lookup finds nothing — not because the spot is unusual, but because
    /// nobody at the table was obliged to bet in the solver's sizes. Measured
    /// against a live opponent this accounted for every miss: thirty-one per
    /// cent of postflop decisions fell through to the heuristic, all of them
    /// for want of a price the solve had no reason to have met.
    ///
    /// This is the postflop half of action translation, and the same idea as
    /// the preflop ladder's: a price the solve does not know is played as the
    /// nearest one it does.
    ///
    /// # What is never substituted
    ///
    /// Only price and depth, which are bands over continuous quantities and so
    /// have a meaningful notion of nearby. Street, position, strength and the
    /// shape of the move list are exact facts, and a neighbouring value of any
    /// of them is a different spot rather than a nearby one. Holding the shape
    /// fixed also keeps the action count right, which is what makes the
    /// substituted strategy safe to index.
    pub fn keys_near(&self, spot: &Spot) -> Vec<InfoKey> {
        let exact = self.key_at(spot);
        let price = (exact >> PRICE_SHIFT) & ((1 << PRICE_BITS) - 1);
        let depth = (exact >> DEPTH_SHIFT) & ((1 << DEPTH_BITS) - 1);
        let stripped = exact
            & !(((1 << PRICE_BITS) - 1) << PRICE_SHIFT)
            & !(((1 << DEPTH_BITS) - 1) << DEPTH_SHIFT);

        // Nearest first; then depth in preference to price, since price is
        // what a call costs and decides more of a decision than how much is
        // left behind it; then, between two prices equally far away, the
        // dearer one.
        //
        // That last tie-break is not arbitrary. Being wrong about a price in
        // the cheap direction means calling bets that are pricier than the
        // strategy being played assumes, which is precisely the leak that makes
        // a bot a calling station. Being wrong in the dear direction means
        // folding a little too often, which costs a fraction of the same.
        let mut offsets: Vec<(i64, i64)> = Vec::new();
        for reach in 0..=2i64 {
            for dp in -reach..=reach {
                for dd in -reach..=reach {
                    if dp.abs().max(dd.abs()) == reach {
                        offsets.push((dp, dd));
                    }
                }
            }
        }
        offsets.sort_by_key(|(dp, dd)| {
            (
                dp.abs().max(dd.abs()),
                dp.abs(),
                dd.abs(),
                // Negated, so a positive offset — the dearer price — sorts first.
                -dp,
                -dd,
            )
        });

        let mut keys = Vec::with_capacity(offsets.len());
        for (dp, dd) in offsets {
            let p = price as i64 + dp;
            let d = depth as i64 + dd;
            if !(0..(1 << PRICE_BITS)).contains(&p) || !(0..(1 << DEPTH_BITS)).contains(&d) {
                continue;
            }
            keys.push(stripped | ((p as u64) << PRICE_SHIFT) | ((d as u64) << DEPTH_SHIFT));
        }
        keys
    }


    /// Moves the hand to the next street, or to showdown after the river.
    fn advance(&self, state: &State) -> State {
        let mut next = *state;
        for player in 0..self.players {
            next.spent[player] += next.wagered[player];
        }
        next.settled = state.pot();
        next.wagered = [0; MAX_PLAYERS];
        next.acted = 0;
        next.raises = 0;
        next.street = match state.street {
            Street::Flop => Street::Turn,
            Street::Turn => Street::River,
            // The river has no next street; the caller checks for that.
            other => other,
        };
        // Out of position acts first on every postflop street. Seat 1 leads;
        // if it has folded or is all in, the next seat that can act does.
        next.to_act = self.next_actor(&next, 0).unwrap_or(0) as u8;
        next
    }

    /// Whether nobody live has a decision left, so the rest is dealt out.
    ///
    /// One player with chips behind is enough to end the betting but not to
    /// make it a decision: with everyone else all in there is nothing left to
    /// bet at. So this asks whether at most one live player has anything
    /// behind, rather than whether every one of them is all in.
    fn all_in(&self, state: &State) -> bool {
        (0..self.players)
            .filter(|&player| state.is_live(player))
            .filter(|&player| self.behind(state, player) > 0)
            .count()
            <= 1
    }
}

impl Game for Postflop {
    type State = State;

    /// How many seats the solver should traverse.
    ///
    /// The trait's default is two, and inheriting it was a live bug for as long
    /// as this took to notice: a three-way tree would report two players, the
    /// solver would update regrets for seats 0 and 1 and never for seat 2, and
    /// the third seat would be left playing whatever an untrained node returns
    /// while everything else looked like it had converged.
    ///
    /// The inherent `Postflop::players` does not save it. A generic solver
    /// calls through the trait, so the trait is where the answer has to be.
    fn players(&self) -> usize {
        self.players
    }

    fn initial(&self) -> State {
        let mut holdings = [0u16; MAX_PLAYERS];
        for (seat, holding) in holdings.iter_mut().enumerate() {
            *holding = seat as u16;
        }
        State {
            board: 0,
            holdings,
            street: Street::Flop,
            wagered: [0; MAX_PLAYERS],
            spent: [0; MAX_PLAYERS],
            settled: self.pot,
            to_act: 1,
            acted: 0,
            raises: 0,
            folded: 0,
            dealt: false,
        }
    }

    fn is_terminal(&self, state: &State) -> bool {
        if !state.dealt {
            return false;
        }
        // Everyone but one has given up; there is nothing left to play for.
        if self.live_count(state) <= 1 {
            return true;
        }
        // All in with cards to come is terminal too: nobody has a decision left
        // and the rest is a formality the showdown already accounts for.
        self.street_closed(state) && (state.street == Street::River || self.all_in(state))
    }

    /// What the hand was worth to player 0.
    ///
    /// Only meaningful with two seats, where one signed number describes both.
    /// Three-handed there is no such number and [`Game::utility_for`] is the
    /// one to ask.
    fn terminal_utility(&self, state: &State) -> f64 {
        debug_assert_eq!(self.players, 2, "multiway payoffs need utility_for");
        self.utility_for(state, 0)
    }

    /// What the hand was worth to one player.
    ///
    /// # The pot that is already there
    ///
    /// A hand arrives at the flop with money in the middle that nobody puts in
    /// during this tree — it went in preflop — and whoever wins takes it. That
    /// dead money is most of what makes postflop poker a game.
    ///
    /// Leaving it out was a real bug, and an instructive one: every invariant
    /// held. Payoffs summed to zero, nobody wagered more than their stack, the
    /// tree was well formed, and the solve converged. What it converged to was
    /// nonsense. With nothing in the middle, betting can only win back what the
    /// opponent chooses to put in, folding immediately costs exactly nothing,
    /// and a bluff wins nothing at all. The equilibrium of that game is to
    /// check everything down, and that is what came out — checking 97% of the
    /// time holding the strongest hands on the board.
    ///
    /// No test caught it because no test asked whether the strategy was poker.
    /// `the_best_hands_bet` now does.
    ///
    /// # Why it is measured against a share rather than the whole pot
    ///
    /// The solver needs payoffs that sum to zero. Handing the winner the whole
    /// pot makes the game constant-sum instead, which is a different thing and
    /// breaks the arithmetic underneath.
    ///
    /// So every player is measured against the same baseline: getting their own
    /// money back, plus an equal share of the dead pot. Win, and you are up by
    /// everything the others staked and their shares besides; lose, and you are
    /// down your own stake and your share. Those cancel exactly. It is a fixed
    /// offset per player, and an offset changes no decision — what a strategy
    /// responds to is the difference between its actions.
    ///
    /// Written as one formula for any number of seats rather than as a case per
    /// outcome. The heads-up cases it replaced were four separate branches that
    /// each had to be got right, and the pinned keys and payoff tests confirm
    /// it reproduces them exactly.
    fn utility_for(&self, state: &State, player: usize) -> f64 {
        debug_assert!(player < self.players);
        let staked = |seat: usize| (state.spent[seat] + state.wagered[seat]) as f64;

        // Everything in the middle: what everyone put in, plus what arrived
        // from preflop.
        let whole: f64 = (0..self.players).map(staked).sum::<f64>() + self.pot as f64;
        // What this player would have if the hand had never been played.
        let baseline = staked(player) + self.pot as f64 / self.players as f64;

        let live: Vec<usize> = (0..self.players)
            .filter(|&seat| state.is_live(seat))
            .collect();
        if live.len() == 1 {
            let winner = live[0];
            return if winner == player { whole } else { 0.0 } - baseline;
        }

        // A showdown among everyone still holding cards. Settled from the
        // finished hands rather than from strength buckets, which are an
        // abstraction for deciding and not for paying out.
        let sample = self.sample();
        let rank = |seat: usize| sample.rank(state.board as usize, state.holdings[seat] as usize);
        let best = live.iter().map(|&seat| rank(seat)).max().expect("someone is live");
        let winners = live.iter().filter(|&&seat| rank(seat) == best).count();

        let won = if state.is_live(player) && rank(player) == best {
            whole / winners as f64
        } else {
            0.0
        };
        won - baseline
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

        // One holding per seat, none of them sharing a card: two players cannot
        // both be holding the ace of spades.
        //
        // Drawn by retrying rather than by dealing from a shrinking deck,
        // because the holdings are pre-enumerated per board and there is no
        // deck here to remove cards from. A clash is rare enough that a handful
        // of attempts finds a free holding; on the rare miss the seat keeps a
        // duplicate rather than looping forever, which costs one sampled hand
        // out of very many and cannot stall a solve.
        let mut dealt = [0usize; MAX_PLAYERS];
        for seat in 0..self.players {
            let mut choice = rng.below(HOLDINGS as u64) as usize;
            for _ in 0..64 {
                let clash = (0..seat).any(|taken| {
                    holdings[choice]
                        .iter()
                        .any(|card| holdings[dealt[taken]].contains(card))
                });
                if !clash {
                    break;
                }
                choice = rng.below(HOLDINGS as u64) as usize;
            }
            dealt[seat] = choice;
        }

        let mut next = State {
            board: board as u32,
            dealt: true,
            ..*state
        };
        for seat in 0..self.players {
            next.holdings[seat] = dealt[seat] as u16;
        }
        next
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
            Move::Fold => next.folded |= 1 << player,
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
            let finished = self.live_count(&next) <= 1
                || next.street == Street::River
                || self.all_in(&next);
            if !finished {
                return self.advance(&next);
            }
        } else if let Some(seat) = self.next_actor(&next, player) {
            next.to_act = seat as u8;
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

    /// The exact information keys a heads-up solve produces.
    ///
    /// # Why these are written down
    ///
    /// A blueprint is a map from these numbers to strategies. The numbers are
    /// not stored in it — only the keys they came out as — so any change to how
    /// a key is packed silently repoints every solved strategy at a different
    /// situation. Nothing fails: the file loads, lookups succeed, and the bot
    /// plays a river strategy on a flop.
    ///
    /// The solved ladder in `data/postflop` was trained against exactly these
    /// numbers. They are pinned here so that widening the tree to three players
    /// cannot move them without a test saying so.
    #[test]
    fn heads_up_keys_are_fixed() {
        let game = Postflop::for_play(48, 100, 1_000, Sizing::default());
        let spot = |street, player, strength, pot, bet, mine, behind, raises, acted| Spot {
            street,
            player,
            live: 2,
            strength,
            pot,
            bet,
            mine,
            behind,
            opponent_behind: behind,
            raises,
            acted,
        };

        for (expected, name, spot) in [
            (
                25_169_408,
                "flop, first to act, nothing bet",
                spot(Street::Flop, 1, 0, 100, 0, 0, 1_000, 0, 0),
            ),
            (
                18_386_712,
                "flop, in position, facing a bet",
                spot(Street::Flop, 0, 24, 133, 33, 0, 1_000, 1, 0b10),
            ),
            (
                44_371_759,
                "turn, facing a raise",
                spot(Street::Turn, 1, 47, 400, 200, 60, 800, 2, 0b01),
            ),
            (
                52_003_119,
                "river, all in",
                spot(Street::River, 0, 47, 900, 500, 0, 500, 1, 0b10),
            ),
        ] {
            assert_eq!(
                game.key_at(&spot),
                expected,
                "{name}: the solved ladder in data/postflop is keyed on the old number"
            );
        }
    }


    use super::*;
    use crate::cfr::Solver;

    fn game() -> Postflop {
        // Ten big blinds in, a hundred behind, on a small sample of boards.
        Postflop::new(
            Textures::sample(8, 10, 0x7E47, 4),
            1_000,
            10_000,
            Sizing::default(),
        )
    }

    /// A three-way tree on the same small board sample.
    fn three_way() -> Postflop {
        Postflop::multiway(
            3,
            Textures::sample(8, 10, 0x7E47, 4),
            1_000,
            10_000,
            Sizing::default(),
        )
    }

    /// Chips move between three players and are never created.
    ///
    /// The heads-up version of this passed throughout a period when the pot was
    /// not being awarded at all — summing to zero is necessary and nowhere near
    /// sufficient. What it does catch is the thing three players make easy to
    /// get wrong: paying a folded player, or paying the pot out twice when two
    /// hands tie.
    #[test]
    fn three_way_payoffs_sum_to_zero() {
        let game = three_way();
        let mut rng = Rng::new(21);
        for round in 0..400 {
            let mut state = game.sample_chance(&game.initial(), &mut rng);
            while !game.is_terminal(&state) {
                let count = game.num_actions(&state);
                assert!(count >= 2, "only {count} action(s) at {state:?}");
                state = game.apply(&state, rng.below(count as u64) as usize);
            }
            let payoffs: Vec<f64> = (0..3).map(|seat| game.utility_for(&state, seat)).collect();
            let total: f64 = payoffs.iter().sum();
            assert!(
                total.abs() < 1e-9,
                "round {round}: {payoffs:?} sum to {total}, not zero"
            );
            for (seat, payoff) in payoffs.iter().enumerate() {
                assert!(
                    payoff.abs() <= 21_000.0,
                    "round {round}: seat {seat} won or lost {payoff}, more than the table holds"
                );
            }
        }
    }

    #[test]
    fn nobody_in_a_three_way_pot_wagers_more_than_they_have() {
        let game = three_way();
        let mut rng = Rng::new(33);
        for _ in 0..400 {
            let mut state = game.sample_chance(&game.initial(), &mut rng);
            while !game.is_terminal(&state) {
                let count = game.num_actions(&state);
                state = game.apply(&state, rng.below(count as u64) as usize);
                for player in 0..3 {
                    assert!(
                        state.spent[player] + state.wagered[player] <= 10_000,
                        "player {player} has put in more than their stack: {state:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn three_players_never_hold_the_same_card() {
        let game = three_way();
        let mut rng = Rng::new(9);
        let sample = game.textures().expect("a solving tree carries boards");
        for _ in 0..300 {
            let state = game.sample_chance(&game.initial(), &mut rng);
            let holdings = sample.holdings(state.board as usize);
            let mut seen = Vec::new();
            for seat in 0..3 {
                for card in holdings[state.holdings[seat] as usize] {
                    assert!(!seen.contains(&card), "two seats hold {card}: {state:?}");
                    seen.push(card);
                }
            }
        }
    }

    /// One fold leaves a hand to play; the second ends it.
    ///
    /// This is the whole difference between a three-way tree and a heads-up
    /// one, and the place a heads-up assumption would survive unnoticed: the
    /// old tree treated any fold as the end of the hand.
    #[test]
    fn a_three_way_pot_survives_the_first_fold() {
        let game = three_way();
        let state = game.sample_chance(&game.initial(), &mut Rng::new(5));

        // Seat 1 leads on every postflop street.
        assert_eq!(game.current_player(&state), 1);

        let fold = |game: &Postflop, state: &State| {
            let at = game
                .moves(state)
                .iter()
                .position(|move_| *move_ == Move::Fold)
                .expect("a fold is on offer");
            game.apply(state, at)
        };
        let bet = |game: &Postflop, state: &State| {
            let at = game
                .moves(state)
                .iter()
                .position(|move_| *move_ == Move::Small)
                .expect("a bet is on offer");
            game.apply(state, at)
        };

        // Seat 1 bets, seat 2 folds. Two are left, so the hand goes on.
        let after_bet = bet(&game, &state);
        assert_eq!(game.current_player(&after_bet), 2, "seat 2 acts second");
        let one_gone = fold(&game, &after_bet);
        assert!(
            !game.is_terminal(&one_gone),
            "two players are still in: {one_gone:?}"
        );
        assert_eq!(game.current_player(&one_gone), 0, "seat 0 acts last");

        // Seat 0 folds too. Now there is one player and the hand is over.
        let both_gone = fold(&game, &one_gone);
        assert!(game.is_terminal(&both_gone), "only seat 1 is left");
        assert!(
            game.utility_for(&both_gone, 1) > 0.0,
            "the last player standing wins the pot"
        );
        assert!(game.utility_for(&both_gone, 0) < 0.0);
        assert!(game.utility_for(&both_gone, 2) < 0.0);
    }

    /// A folded player is not owed anything and cannot be paid.
    #[test]
    fn a_folded_player_is_never_paid() {
        let game = three_way();
        let mut rng = Rng::new(77);
        for _ in 0..400 {
            let mut state = game.sample_chance(&game.initial(), &mut rng);
            while !game.is_terminal(&state) {
                let count = game.num_actions(&state);
                state = game.apply(&state, rng.below(count as u64) as usize);
            }
            for seat in 0..3 {
                if !state.is_live(seat) {
                    assert!(
                        game.utility_for(&state, seat) < 0.0,
                        "seat {seat} folded and came out ahead: {state:?}"
                    );
                }
            }
        }
    }

    /// The two key spaces cannot collide, and neither can two spots that offer
    /// different numbers of actions.
    ///
    /// The second half is the one that bites. The solver stores its regrets
    /// against a key and trusts the action count to match, so a collision
    /// indexes past the end of another node's actions — which is not a wrong
    /// strategy but a wrong memory read.
    #[test]
    fn multiway_keys_stand_apart_from_heads_up_ones() {
        let heads_up = Postflop::for_play(48, 1_000, 10_000, Sizing::default());
        let three = Postflop::multiway_for_play(3, 48, 1_000, 10_000, Sizing::default());

        let spot = |live: u8, player: usize, bet: u64, mine: u64, raises: u8, acted: u8| Spot {
            street: Street::Flop,
            player,
            live,
            strength: 20,
            pot: 1_000,
            bet,
            mine,
            behind: 9_000,
            opponent_behind: 9_000,
            raises,
            acted,
        };

        // Nothing a three-way tree produces can be mistaken for a heads-up key.
        let mut by_key = std::collections::HashMap::new();
        for live in 2..=3u8 {
            for player in 0..3 {
                for (bet, mine) in [(0, 0), (300, 0), (300, 300), (900, 300)] {
                    for raises in 0..3u8 {
                        for acted in 0..8u8 {
                            let spot = spot(live, player, bet, mine, raises, acted);
                            let key = three.key_at(&spot);
                            assert!(
                                key & MULTIWAY != 0,
                                "a multiway key must carry its marker: {spot:?}"
                            );
                            let count = three.moves_at(&spot).len();
                            let held = by_key.entry(key).or_insert(count);
                            assert_eq!(
                                *held, count,
                                "two spots share key {key} while offering {held} and {count} actions"
                            );
                        }
                    }
                }
            }
        }

        for player in 0..2 {
            for (bet, mine) in [(0, 0), (300, 0), (300, 300), (900, 300)] {
                let spot = spot(2, player, bet, mine, 1, 0b10);
                let key = heads_up.key_at(&spot);
                assert!(key & MULTIWAY == 0, "a heads-up key carries no marker");
                assert!(
                    !by_key.contains_key(&key),
                    "heads-up key {key} collides with a multiway one"
                );
            }
        }
    }

    /// The solver must be told how many seats to traverse.
    ///
    /// It reaches this through the [`Game`] trait, not through the inherent
    /// method of the same name, and the trait's default is two. A three-way
    /// tree that inherited the default would train two of its three seats and
    /// report nothing wrong.
    #[test]
    fn the_tree_reports_its_seat_count_through_the_trait() {
        fn asked_as_a_game<G: Game>(game: &G) -> usize {
            game.players()
        }

        let heads_up = Postflop::for_play(48, 1_000, 10_000, Sizing::default());
        assert_eq!(asked_as_a_game(&heads_up), 2);

        let three = Postflop::multiway_for_play(3, 48, 1_000, 10_000, Sizing::default());
        assert_eq!(asked_as_a_game(&three), 3, "the solver would skip a seat");
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
    /// The solve plays a polarised strategy: it value bets and it bluffs.
    ///
    /// # Why a behavioural test
    ///
    /// Every other test here checks that the tree is well formed: payoffs sum
    /// to zero, stacks are respected, keys do not collide, players never share
    /// a card. All of them passed against a version of this game in which the
    /// pot the hand arrives with was never awarded to anybody — so there was
    /// nothing to win, and the solve learned to check its very best hands 97%
    /// of the time. The tree was impeccable and the poker was gone.
    ///
    /// So this asks the only question those cannot: having solved, does the
    /// strategy do what poker requires?
    ///
    /// The sharp end of that is bluffing. A bluff wins the pot or it wins
    /// nothing at all, so a solve that bluffs has necessarily learned there is
    /// a pot to be won — exactly the fact the bug removed. Asking only whether
    /// good hands bet does not work, and was tried first: they still did, 61%
    /// of the time, because a value bet can at least collect a call. It was the
    /// weakest hands that gave it away, betting 10% where a correct game bets
    /// 73%.
    ///
    /// The thresholds sit well below what a correct game produces — 76% and
    /// 73% against 40% and 25% — because the failure guarded against is total
    /// rather than marginal.
    #[test]
    fn the_best_hands_bet() {
        const BUCKETS: usize = 8;
        let textures = Textures::sample(40, BUCKETS, 0x51DE, 4);
        let game = Postflop::new(textures, 100, 400, Sizing::default());
        let mut rng = Rng::new(0xBE77);
        let mut solver = Solver::new(game);
        solver.train_sampled(400_000, &mut rng);

        let game = Postflop::for_play(BUCKETS, 100, 400, Sizing::default());
        let spot = |strength: u8| Spot {
            live: 2,
            street: Street::River,
            // Out of position, first to act, nothing bet yet.
            player: 1,
            strength,
            pot: 100,
            bet: 0,
            mine: 0,
            behind: 400,
            opponent_behind: 400,
            raises: 0,
            acted: 0,
        };

        let moves = game.moves_at(&spot(0));
        let betting = |strength: u8| -> Option<f64> {
            let strategy = solver.average_strategy(game.key_at(&spot(strength)))?;
            Some(
                moves
                    .iter()
                    .zip(strategy)
                    .filter(|(mv, _)| **mv != Move::Passive && **mv != Move::Fold)
                    .map(|(_, share)| share)
                    .sum(),
            )
        };

        let best = betting(BUCKETS as u8 - 1).expect("the strongest group was solved");
        let worst = betting(0).expect("the weakest group was solved");
        assert!(
            best > 0.4,
            "with the strongest hands on the board the solve bets only {:.0}% of the time",
            best * 100.0
        );
        assert!(
            worst > 0.25,
            "the solve bluffs its weakest hands only {:.0}% of the time; with nothing in the middle to win a bluff wins nothing, and this is what that looks like",
            worst * 100.0
        );
    }

    #[test]
    fn nobody_raises_into_a_player_who_is_already_all_in() {
        let game = game();
        // The hero has plenty behind; the opponent has shoved and has nothing.
        let facing_a_shove = Spot {
            live: 2,
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

    /// Folding gives up the pot, which is what makes it a decision.
    ///
    /// This test used to assert the opposite — that folding to a bet before
    /// putting anything in was worth exactly zero — and it passed, because the
    /// pot the hand arrived with was not being awarded to anybody. That is the
    /// bug written down as an assertion. With nothing to forfeit, folding is
    /// free, bluffing wins nothing, and the solve learned to check every hand
    /// down. See [`Postflop::terminal_utility`].
    #[test]
    fn folding_gives_up_the_pot() {
        let game = game();
        let state = play(&game, dealt(&game), &[Move::Large, Move::Fold]);
        assert!(game.is_terminal(&state));
        // Player 0 folds without having put a chip in, so loses none of their
        // own — and gives up their claim on the thousand already in the middle.
        // Measured against the two splitting it, that is half.
        assert_eq!(game.terminal_utility(&state), -500.0);
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
