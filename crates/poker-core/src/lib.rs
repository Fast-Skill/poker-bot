//! Core poker primitives: cards, hand evaluation, and game state.
//!
//! This crate is the foundation the solver, equity tools, and bot runtime all
//! build on. It is deliberately free of I/O and of any dependency on how a
//! table is observed, so the same types serve self-play, offline analysis, and
//! the live bot.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_debug_implementations)]

pub mod abstraction;
pub mod betting;
pub mod card;
pub mod cfr;
pub mod equity;
pub mod eval;
pub mod kuhn;
pub mod omaha;
pub mod pot;
pub mod rng;

pub use abstraction::{bucket_by_strength, BetSizing, HandClass, NUM_HAND_CLASSES};
pub use betting::{Action, ActionError, BettingRound, LegalActions, Seat, Street};
pub use cfr::{Game, InfoKey, Profile, Solver};
pub use card::{
    parse_cards, Card, CardSet, ParseCardError, ParseCardsError, ParseCardsErrorKind, Rank, Suit,
    NUM_CARDS, NUM_RANKS, NUM_SUITS,
};
pub use equity::{Equity, EquityError, Variant};
pub use eval::{evaluate, Category, HandRank};
pub use rng::Rng;
pub use omaha::{best_omaha_hand, evaluate_omaha};
pub use pot::{award, build_pots, OddChip, Pot};
