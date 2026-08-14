//! Abstraction: shrinking the game tree to a size CFR can actually solve.
//!
//! No-Limit Hold'em has on the order of 10^160 decision points. CFR cannot
//! touch that, so the tree is compressed along two independent axes:
//!
//! - **Card abstraction** groups hands that play alike. Preflop this is exact
//!   and lossless — suits carry no information before a board exists, so 1,326
//!   hole-card combinations collapse to 169 strategically distinct classes with
//!   nothing thrown away. Postflop it is lossy, bucketing hands by equity.
//! - **Action abstraction** restricts betting to a handful of pot-relative
//!   sizes instead of every legal integer amount.
//!
//! Abstraction is where a solver's real strength is decided. A perfect solve of
//! a crude abstraction loses to a rough solve of a good one, because the
//! abstraction determines which distinctions the strategy is even capable of
//! making.

use crate::betting::{Action, LegalActions};
use crate::card::{Card, Rank, NUM_RANKS};
use std::fmt;
use std::str::FromStr;

/// The number of strategically distinct starting hands in Hold'em.
pub const NUM_HAND_CLASSES: usize = NUM_RANKS * NUM_RANKS;

/// A starting-hand class such as `AA`, `AKs`, or `AKo`.
///
/// Preflop, two hands differing only in suit are strategically identical: with
/// no board out, a spade is indistinguishable from a heart. Collapsing the
/// 1,326 combinations into 169 classes is therefore **lossless**, and it shrinks
/// the preflop tree by a factor of nearly eight for free.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HandClass(u8);

impl HandClass {
    /// The class of a two-card holding. Order of the cards does not matter.
    ///
    /// # Panics
    /// Panics if both cards are the same.
    pub fn from_cards(a: Card, b: Card) -> HandClass {
        assert_ne!(a, b, "a hand cannot hold the same card twice");
        let (high, low) = {
            let (x, y) = (a.rank().index(), b.rank().index());
            if x >= y {
                (x, y)
            } else {
                (y, x)
            }
        };
        // A 13x13 grid: pairs on the diagonal, suited hands on one side,
        // offsuit on the other. Every cell is used exactly once, so the 169
        // classes map onto 0..169 with no gaps.
        let index = if a.suit() == b.suit() {
            high * NUM_RANKS as u8 + low
        } else {
            low * NUM_RANKS as u8 + high
        };
        HandClass(index)
    }

    /// Builds a class from its `0..169` index.
    pub const fn from_index(index: usize) -> Option<HandClass> {
        if index >= NUM_HAND_CLASSES {
            return None;
        }
        Some(HandClass(index as u8))
    }

    /// The index of this class, in `0..169`.
    #[inline]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    /// The grid coordinates this class occupies.
    #[inline]
    const fn coordinates(self) -> (u8, u8) {
        (self.0 / NUM_RANKS as u8, self.0 % NUM_RANKS as u8)
    }

    /// The higher of the two ranks.
    pub fn high(self) -> Rank {
        let (row, column) = self.coordinates();
        Rank::from_index(row.max(column)).expect("index is in range")
    }

    /// The lower of the two ranks, equal to [`HandClass::high`] for a pair.
    pub fn low(self) -> Rank {
        let (row, column) = self.coordinates();
        Rank::from_index(row.min(column)).expect("index is in range")
    }

    #[inline]
    pub const fn is_pair(self) -> bool {
        let (row, column) = self.coordinates();
        row == column
    }

    /// Whether both cards share a suit. Always false for a pair.
    #[inline]
    pub const fn is_suited(self) -> bool {
        let (row, column) = self.coordinates();
        row > column
    }

    /// How many card combinations make up this class: 6 for a pair, 4 for a
    /// suited hand, 12 for an offsuit one.
    ///
    /// These weights matter — treating all 169 classes as equally likely
    /// overstates pairs threefold against offsuit hands.
    pub const fn combos(self) -> u32 {
        if self.is_pair() {
            6
        } else if self.is_suited() {
            4
        } else {
            12
        }
    }

    /// Every hand class, in index order.
    pub fn all() -> impl Iterator<Item = HandClass> {
        (0..NUM_HAND_CLASSES).map(|i| HandClass(i as u8))
    }
}

impl fmt::Display for HandClass {
    /// Renders as `AA`, `AKs`, or `AKo`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (high, low) = (self.high().to_char(), self.low().to_char());
        if self.is_pair() {
            write!(f, "{high}{low}")
        } else if self.is_suited() {
            write!(f, "{high}{low}s")
        } else {
            write!(f, "{high}{low}o")
        }
    }
}

/// Why a hand-class string could not be parsed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseHandClassError {
    /// Not two or three characters.
    WrongLength(usize),
    /// A rank character was not recognised.
    BadRank(char),
    /// The suffix was neither `s` nor `o`.
    BadSuffix(char),
    /// A pair was written with a suffix, e.g. `AAs`.
    PairWithSuffix,
    /// A non-pair was written without a suffix, e.g. `AK`.
    MissingSuffix,
}

impl fmt::Display for ParseHandClassError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseHandClassError::WrongLength(n) => {
                write!(f, "expected 2 or 3 characters (e.g. \"AA\", \"AKs\"), got {n}")
            }
            ParseHandClassError::BadRank(c) => write!(f, "invalid rank character {c:?}"),
            ParseHandClassError::BadSuffix(c) => {
                write!(f, "expected 's' or 'o' suffix, got {c:?}")
            }
            ParseHandClassError::PairWithSuffix => {
                f.write_str("a pair cannot be suited or offsuit")
            }
            ParseHandClassError::MissingSuffix => {
                f.write_str("a non-pair needs an 's' or 'o' suffix")
            }
        }
    }
}

impl std::error::Error for ParseHandClassError {}

impl FromStr for HandClass {
    type Err = ParseHandClassError;

    /// Parses `"AA"`, `"AKs"`, or `"AKo"`.
    fn from_str(s: &str) -> Result<HandClass, ParseHandClassError> {
        let chars: Vec<char> = s.chars().collect();
        if !(2..=3).contains(&chars.len()) {
            return Err(ParseHandClassError::WrongLength(chars.len()));
        }

        let first = Rank::from_char(chars[0]).ok_or(ParseHandClassError::BadRank(chars[0]))?;
        let second = Rank::from_char(chars[1]).ok_or(ParseHandClassError::BadRank(chars[1]))?;
        let (high, low) = if first >= second {
            (first, second)
        } else {
            (second, first)
        };
        let is_pair = high == low;

        let suited = match chars.get(2) {
            None if is_pair => false,
            None => return Err(ParseHandClassError::MissingSuffix),
            Some(_) if is_pair => return Err(ParseHandClassError::PairWithSuffix),
            Some('s') | Some('S') => true,
            Some('o') | Some('O') => false,
            Some(&c) => return Err(ParseHandClassError::BadSuffix(c)),
        };

        let (row, column) = if suited || is_pair {
            (high.index(), low.index())
        } else {
            (low.index(), high.index())
        };
        Ok(HandClass(row * NUM_RANKS as u8 + column))
    }
}

/// Assigns items to equal-frequency buckets by their strength value.
///
/// `values[i]` is item `i`'s strength — normally equity against a random hand.
/// The result gives each item a bucket in `0..buckets`, with the weakest items
/// in bucket 0.
///
/// Equal *frequency* rather than equal width: hand strengths cluster heavily
/// around the middle, and fixed-width bins would leave the extremes nearly
/// empty while crowding everything interesting into one bucket. Ties may
/// straddle a boundary, which is acceptable — the point is resolution where
/// hands actually live.
///
/// # Panics
/// Panics if `buckets` is zero.
pub fn bucket_by_strength(values: &[f64], buckets: usize) -> Vec<u16> {
    assert!(buckets > 0, "need at least one bucket");
    if values.is_empty() {
        return Vec::new();
    }

    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap_or(std::cmp::Ordering::Equal));

    let mut assignment = vec![0u16; values.len()];
    for (rank, &item) in order.iter().enumerate() {
        // Spread ranks across buckets, keeping sizes within one of each other.
        let bucket = rank * buckets / values.len();
        assignment[item] = bucket.min(buckets - 1) as u16;
    }
    assignment
}

/// Which bet sizes the solver is allowed to consider.
///
/// Sizes are fractions of the pot *after* calling, matching how poker players
/// describe them: a "pot-sized raise" calls first, then bets the resulting pot.
#[derive(Debug, Clone, PartialEq)]
pub struct BetSizing {
    /// Pot fractions offered as bets or raises, ascending.
    pub fractions: Vec<f64>,
    /// Whether moving all in is always available.
    pub include_all_in: bool,
}

impl BetSizing {
    /// A compact tree suitable for microstakes: half pot, pot, and all in.
    ///
    /// Three sizes is deliberately small. A wider tree costs solve time
    /// super-linearly, and against weak opposition the gain from finer sizing
    /// is far smaller than the gain from solving the coarse tree properly.
    pub fn compact() -> BetSizing {
        BetSizing {
            fractions: vec![0.5, 1.0],
            include_all_in: true,
        }
    }

    /// A wider tree for tougher games: third pot through overbet, plus all in.
    pub fn wide() -> BetSizing {
        BetSizing {
            fractions: vec![0.33, 0.5, 0.75, 1.0, 1.5],
            include_all_in: true,
        }
    }

    /// The concrete actions this abstraction offers in a given spot.
    ///
    /// `pot` is everything already in the middle, excluding what the acting
    /// player still owes. Sizes that fall below the minimum raise or above the
    /// player's stack are clamped and de-duplicated, so the result is always
    /// legal and never offers the same amount twice.
    pub fn actions(&self, pot: u64, legal: &LegalActions) -> Vec<Action> {
        let mut actions = Vec::with_capacity(self.fractions.len() + 3);
        if legal.can_fold {
            actions.push(Action::Fold);
        }
        if legal.can_check {
            actions.push(Action::Check);
        }
        if legal.call_cost.is_some() {
            actions.push(Action::Call);
        }

        let Some((min_raise, max_raise)) = legal.raise_to else {
            return actions;
        };

        let call_cost = legal.call_cost.unwrap_or(0);
        let pot_after_call = pot + call_cost;
        let mut sizes: Vec<u64> = Vec::with_capacity(self.fractions.len() + 1);

        for &fraction in &self.fractions {
            // Total street commitment = the chips needed to call, plus a share
            // of the pot that calling would create.
            let target = (pot_after_call as f64 * fraction).round() as u64 + call_cost;
            sizes.push(target.clamp(min_raise, max_raise));
        }

        if self.include_all_in {
            sizes.push(max_raise);
        }

        sizes.sort_unstable();
        sizes.dedup();
        actions.extend(sizes.into_iter().map(Action::RaiseTo));
        actions
    }
}

impl Default for BetSizing {
    fn default() -> BetSizing {
        BetSizing::compact()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::betting::{BettingRound, Seat};
    use crate::card::{parse_cards, Suit};

    fn class(text: &str) -> HandClass {
        text.parse().unwrap_or_else(|e| panic!("{text}: {e}"))
    }

    fn class_of(text: &str) -> HandClass {
        let cards = parse_cards(text).expect("valid cards");
        assert_eq!(cards.len(), 2);
        HandClass::from_cards(cards[0], cards[1])
    }

    #[test]
    fn every_hole_pair_maps_into_one_of_169_classes() {
        // The definitive check: all 1,326 combinations, and the class sizes
        // must come out to the textbook 13 pairs, 78 suited, 78 offsuit.
        let deck: Vec<Card> = Card::all().collect();
        let mut seen = std::collections::HashMap::new();
        let mut combos = 0;

        for (i, &a) in deck.iter().enumerate() {
            for &b in &deck[i + 1..] {
                *seen.entry(HandClass::from_cards(a, b)).or_insert(0u32) += 1;
                combos += 1;
            }
        }

        assert_eq!(combos, 1_326, "C(52,2)");
        assert_eq!(seen.len(), 169, "strategically distinct starting hands");

        let pairs = seen.keys().filter(|c| c.is_pair()).count();
        let suited = seen.keys().filter(|c| c.is_suited()).count();
        let offsuit = seen.len() - pairs - suited;
        assert_eq!((pairs, suited, offsuit), (13, 78, 78));

        // Each class must contain exactly as many combos as it claims.
        for (class, &count) in &seen {
            assert_eq!(count, class.combos(), "{class} combo count");
        }
        assert_eq!(seen.values().sum::<u32>(), 1_326, "combos are conserved");
    }

    #[test]
    fn card_order_does_not_matter() {
        let cards = parse_cards("AsKh").expect("valid");
        assert_eq!(
            HandClass::from_cards(cards[0], cards[1]),
            HandClass::from_cards(cards[1], cards[0])
        );
    }

    #[test]
    fn suits_are_irrelevant_beyond_whether_they_match() {
        // Every suited ace-king is the same class...
        let mut suited = Vec::new();
        for suit in Suit::ALL {
            suited.push(HandClass::from_cards(
                Card::new(Rank::Ace, suit),
                Card::new(Rank::King, suit),
            ));
        }
        assert!(suited.windows(2).all(|w| w[0] == w[1]));

        // ...and it differs from the offsuit version.
        assert_ne!(class_of("AsKs"), class_of("AsKh"));
    }

    #[test]
    fn classes_render_in_standard_notation() {
        assert_eq!(class_of("AsAh").to_string(), "AA");
        assert_eq!(class_of("AsKs").to_string(), "AKs");
        assert_eq!(class_of("AsKh").to_string(), "AKo");
        assert_eq!(class_of("2c7d").to_string(), "72o");
        assert_eq!(class_of("Th9h").to_string(), "T9s");
    }

    #[test]
    fn notation_round_trips_through_parsing() {
        for class in HandClass::all() {
            let text = class.to_string();
            assert_eq!(text.parse::<HandClass>(), Ok(class), "{text}");
        }
    }

    #[test]
    fn parsing_accepts_either_rank_order_and_case() {
        assert_eq!(class("AKs"), class("KAs"));
        assert_eq!(class("AKo"), class("KAo"));
        assert_eq!(class("aks"), class("AKs"));
        assert_eq!(class("AKS"), class("AKs"));
    }

    #[test]
    fn malformed_hand_classes_are_rejected() {
        assert_eq!("A".parse::<HandClass>(), Err(ParseHandClassError::WrongLength(1)));
        assert_eq!("AKsx".parse::<HandClass>(), Err(ParseHandClassError::WrongLength(4)));
        assert_eq!("XKs".parse::<HandClass>(), Err(ParseHandClassError::BadRank('X')));
        assert_eq!("AKx".parse::<HandClass>(), Err(ParseHandClassError::BadSuffix('x')));
        assert_eq!("AAs".parse::<HandClass>(), Err(ParseHandClassError::PairWithSuffix));
        assert_eq!("AK".parse::<HandClass>(), Err(ParseHandClassError::MissingSuffix));
    }

    #[test]
    fn index_round_trips_and_covers_the_grid() {
        for class in HandClass::all() {
            assert_eq!(HandClass::from_index(class.index()), Some(class));
        }
        assert_eq!(HandClass::from_index(NUM_HAND_CLASSES), None);
        assert_eq!(HandClass::all().count(), 169);
    }

    #[test]
    fn high_and_low_ranks_are_ordered() {
        for class in HandClass::all() {
            assert!(class.high() >= class.low(), "{class}");
            if class.is_pair() {
                assert_eq!(class.high(), class.low());
                assert!(!class.is_suited(), "a pair is never suited");
            }
        }
        assert_eq!(class("AKs").high(), Rank::Ace);
        assert_eq!(class("AKs").low(), Rank::King);
    }

    #[test]
    fn buckets_are_ordered_weakest_first_and_evenly_filled() {
        let values: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let assignment = bucket_by_strength(&values, 10);

        assert_eq!(assignment[0], 0, "weakest hand in the bottom bucket");
        assert_eq!(assignment[99], 9, "strongest hand in the top bucket");
        // Strength must never decrease as the bucket rises.
        assert!(assignment.windows(2).all(|w| w[0] <= w[1]));

        let mut sizes = [0usize; 10];
        for &bucket in &assignment {
            sizes[bucket as usize] += 1;
        }
        assert!(sizes.iter().all(|&n| n == 10), "equal frequency: {sizes:?}");
    }

    #[test]
    fn bucketing_handles_degenerate_inputs() {
        assert!(bucket_by_strength(&[], 5).is_empty());
        assert_eq!(bucket_by_strength(&[0.5], 5), vec![0]);
        // Fewer items than buckets is fine; some buckets simply stay empty.
        let assignment = bucket_by_strength(&[0.1, 0.9], 10);
        assert_eq!(assignment.len(), 2);
        assert!(assignment[0] < assignment[1]);
        // All-equal values must not panic or produce an out-of-range bucket.
        let flat = bucket_by_strength(&[0.5; 20], 4);
        assert!(flat.iter().all(|&b| b < 4));
    }

    #[test]
    #[should_panic(expected = "at least one bucket")]
    fn zero_buckets_is_rejected() {
        bucket_by_strength(&[0.5], 0);
    }

    #[test]
    fn a_pot_sized_bet_is_measured_after_calling() {
        // Pot 100, nothing owed: a pot-sized bet is 100.
        let seats = vec![Seat::new(1000), Seat::new(1000)];
        let round = BettingRound::new(seats, 0, 2);
        let sizing = BetSizing {
            fractions: vec![0.5, 1.0],
            include_all_in: false,
        };

        let actions = sizing.actions(100, &round.legal_actions());
        assert!(actions.contains(&Action::Check));
        assert!(actions.contains(&Action::RaiseTo(50)), "half pot");
        assert!(actions.contains(&Action::RaiseTo(100)), "full pot");
    }

    #[test]
    fn raise_sizes_include_the_call_before_the_pot_share() {
        // Facing a bet of 100 into a pot of 100: calling makes it 300, so a
        // pot-sized raise is 100 (the call) + 300 = 400 total.
        let seats = vec![Seat::new(1000), Seat::new(1000)];
        let mut round = BettingRound::new(seats, 0, 2);
        round.apply(Action::RaiseTo(100)).expect("legal bet");

        let sizing = BetSizing {
            fractions: vec![1.0],
            include_all_in: false,
        };
        let actions = sizing.actions(200, &round.legal_actions());
        assert!(actions.contains(&Action::RaiseTo(400)), "got {actions:?}");
    }

    #[test]
    fn every_offered_size_is_actually_legal() {
        let seats = vec![Seat::new(1000), Seat::new(1000)];
        let mut round = BettingRound::new(seats, 0, 2);
        round.apply(Action::RaiseTo(100)).expect("legal bet");

        let legal = round.legal_actions();
        for action in BetSizing::wide().actions(200, &legal) {
            assert!(legal.permits(action), "{action:?} was offered but is illegal");
        }
    }

    #[test]
    fn sizes_are_clamped_and_deduplicated_for_a_short_stack() {
        // A 60-chip stack cannot make most of these raises, so they collapse
        // onto the all-in amount rather than being offered several times.
        let seats = vec![Seat::new(1000), Seat::new(60)];
        let mut round = BettingRound::new(seats, 0, 2);
        round.apply(Action::RaiseTo(20)).expect("legal bet");

        let legal = round.legal_actions();
        let actions = BetSizing::wide().actions(200, &legal);
        let raises: Vec<&Action> = actions
            .iter()
            .filter(|a| matches!(a, Action::RaiseTo(_)))
            .collect();

        assert_eq!(raises.len(), 1, "only one distinct raise fits: {raises:?}");
        assert_eq!(*raises[0], Action::RaiseTo(60), "the all-in amount");
        for action in &actions {
            assert!(legal.permits(*action), "{action:?} is illegal");
        }
    }

    #[test]
    fn no_raises_are_offered_when_the_action_is_closed() {
        // An all-in under-raise does not reopen betting, so the abstraction
        // must offer only fold and call.
        let seats = vec![Seat::new(1000), Seat::new(150)];
        let mut round = BettingRound::new(seats, 0, 2);
        round.apply(Action::RaiseTo(100)).expect("legal bet");
        round.apply(Action::RaiseTo(150)).expect("legal jam");

        let actions = BetSizing::wide().actions(250, &round.legal_actions());
        assert!(actions.contains(&Action::Fold));
        assert!(actions.contains(&Action::Call));
        assert!(
            !actions.iter().any(|a| matches!(a, Action::RaiseTo(_))),
            "the action was not reopened: {actions:?}"
        );
    }

    #[test]
    fn the_compact_tree_stays_small() {
        let seats = vec![Seat::new(1000), Seat::new(1000)];
        let round = BettingRound::new(seats, 0, 2);
        let actions = BetSizing::compact().actions(100, &round.legal_actions());
        // check, half pot, pot, all in — branching this narrow is what keeps
        // the solve tractable.
        assert_eq!(actions.len(), 4, "{actions:?}");
    }
}
