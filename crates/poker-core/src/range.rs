//! Hand ranges: the notation players actually write, turned into solver input.
//!
//! A [`Range`] assigns a weight in `0..=1` to each of the 169 starting-hand
//! classes. It parses the standard shorthand —
//!
//! ```text
//! TT+            pairs from tens up
//! A2s+           every suited ace
//! KTo+           KTo, KJo, KQo
//! 22-55          twos through fives
//! AA:0.5         aces, half the time
//! TT+,AKs,A5o    a comma-separated combination of the above
//! ```
//!
//! — and [`Range::holdings`] turns one into the weighted showdown strengths a
//! river solve consumes, evaluating every concrete combination against a board
//! and discarding the ones the board blocks.
//!
//! That last part matters more than it sounds. A range of aces is six
//! combinations in the abstract, but only three once an ace is on the board.
//! Ignoring that overstates the strong part of a range on exactly the boards
//! where it is most tempting to over-value.

use crate::abstraction::{HandClass, ParseHandClassError, NUM_HAND_CLASSES};
use crate::card::{Card, CardSet, Rank};
use crate::eval::evaluate;
use crate::river::Holding;
use std::fmt;
use std::str::FromStr;

/// Weights over the 169 starting-hand classes.
#[derive(Debug, Clone, PartialEq)]
pub struct Range {
    weights: Vec<f64>,
}

impl Default for Range {
    fn default() -> Range {
        Range::empty()
    }
}

impl Range {
    /// A range containing nothing.
    pub fn empty() -> Range {
        Range {
            weights: vec![0.0; NUM_HAND_CLASSES],
        }
    }

    /// Every hand, at full weight.
    pub fn any() -> Range {
        Range {
            weights: vec![1.0; NUM_HAND_CLASSES],
        }
    }

    /// The weight assigned to `class`.
    pub fn weight(&self, class: HandClass) -> f64 {
        self.weights[class.index()]
    }

    /// Sets the weight for `class`, clamped to `0..=1`.
    pub fn set(&mut self, class: HandClass, weight: f64) {
        self.weights[class.index()] = weight.clamp(0.0, 1.0);
    }

    /// Every class carrying weight, with that weight.
    pub fn entries(&self) -> impl Iterator<Item = (HandClass, f64)> + '_ {
        HandClass::all()
            .map(|class| (class, self.weights[class.index()]))
            .filter(|(_, weight)| *weight > 0.0)
    }

    /// Total weighted card combinations in the range.
    ///
    /// Combination-weighted, not class-weighted: a pair is six combinations
    /// and an offsuit hand is twelve, so counting classes would overstate pairs
    /// threefold.
    pub fn combos(&self) -> f64 {
        self.entries()
            .map(|(class, weight)| weight * class.combos() as f64)
            .sum()
    }

    /// Share of all 1,326 holdings this range covers, in `0..=1`.
    pub fn fraction(&self) -> f64 {
        self.combos() / 1_326.0
    }

    /// Whether the range holds anything at all.
    pub fn is_empty(&self) -> bool {
        self.entries().next().is_none()
    }

    /// The union of two ranges, taking the larger weight for each class.
    pub fn union(&self, other: &Range) -> Range {
        Range {
            weights: self
                .weights
                .iter()
                .zip(&other.weights)
                .map(|(a, b)| a.max(*b))
                .collect(),
        }
    }

    /// Evaluates every unblocked combination against `board`.
    ///
    /// The result feeds straight into a river solve: strengths are
    /// [`crate::eval::HandRank`] bits, which compare correctly by construction,
    /// and each combination carries its class's weight.
    ///
    /// # Panics
    /// Panics if `board` is not five cards, or holds a duplicate.
    pub fn holdings(&self, board: &[Card]) -> Vec<Holding> {
        assert_eq!(board.len(), 5, "a river board is five cards");
        let mut dead = CardSet::empty();
        for &card in board {
            assert!(dead.insert(card), "duplicate board card {card}");
        }

        let mut holdings = Vec::new();
        let mut hand = Vec::with_capacity(7);
        for (class, weight) in self.entries() {
            for combo in class.combinations() {
                // A holding the board already uses cannot be held.
                if dead.contains(combo[0]) || dead.contains(combo[1]) {
                    continue;
                }
                hand.clear();
                hand.extend_from_slice(&combo);
                hand.extend_from_slice(board);
                holdings.push(Holding::new(evaluate(&hand).to_bits(), weight));
            }
        }
        holdings
    }

    /// How many combinations survive on `board`, weighted.
    pub fn combos_on(&self, board: &[Card]) -> f64 {
        self.holdings(board).iter().map(|h| h.weight).sum()
    }
}

/// Why a range string could not be parsed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseRangeError {
    /// A term between commas was blank.
    EmptyTerm,
    /// A hand class in the term was not understood.
    BadClass {
        term: String,
        source: ParseHandClassError,
    },
    /// A dash range joined two hands of different shapes, e.g. `22-AKs`.
    MismatchedEndpoints { term: String },
    /// The weight suffix was not a number in `0..=1`.
    BadWeight { term: String },
}

impl fmt::Display for ParseRangeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseRangeError::EmptyTerm => f.write_str("empty term in range"),
            ParseRangeError::BadClass { term, source } => {
                write!(f, "in {term:?}: {source}")
            }
            ParseRangeError::MismatchedEndpoints { term } => write!(
                f,
                "{term:?} joins hands of different shapes; both ends must be pairs, \
                 or share a high card and suitedness"
            ),
            ParseRangeError::BadWeight { term } => {
                write!(f, "{term:?} has a weight outside 0 to 1")
            }
        }
    }
}

impl std::error::Error for ParseRangeError {}

impl FromStr for Range {
    type Err = ParseRangeError;

    /// Parses standard range notation. See the module docs for the grammar.
    fn from_str(text: &str) -> Result<Range, ParseRangeError> {
        let mut range = Range::empty();
        if text.trim().is_empty() {
            return Ok(range);
        }

        for raw in text.split(',') {
            let term = raw.trim();
            if term.is_empty() {
                return Err(ParseRangeError::EmptyTerm);
            }

            // An optional ":w" suffix scales the whole term.
            let (body, weight) = match term.split_once(':') {
                Some((body, suffix)) => {
                    let weight: f64 = suffix.trim().parse().map_err(|_| {
                        ParseRangeError::BadWeight {
                            term: term.to_string(),
                        }
                    })?;
                    if !(0.0..=1.0).contains(&weight) {
                        return Err(ParseRangeError::BadWeight {
                            term: term.to_string(),
                        });
                    }
                    (body.trim(), weight)
                }
                None => (term, 1.0),
            };

            for class in expand(body, term)? {
                range.set(class, weight);
            }
        }

        Ok(range)
    }
}

/// Expands one term into the classes it names.
fn expand(body: &str, term: &str) -> Result<Vec<HandClass>, ParseRangeError> {
    let bad = |source| ParseRangeError::BadClass {
        term: term.to_string(),
        source,
    };

    if let Some(base) = body.strip_suffix('+') {
        let class: HandClass = base.parse().map_err(bad)?;
        return Ok(walk_upward(class));
    }

    if let Some((low_end, high_end)) = split_dash(body) {
        let first: HandClass = low_end.parse().map_err(bad)?;
        let second: HandClass = high_end.parse().map_err(bad)?;
        return between(first, second).ok_or(ParseRangeError::MismatchedEndpoints {
            term: term.to_string(),
        });
    }

    Ok(vec![body.parse::<HandClass>().map_err(bad)?])
}

/// Splits `A2s-A5s` into its two endpoints, ignoring a leading sign.
fn split_dash(body: &str) -> Option<(&str, &str)> {
    let index = body.char_indices().skip(1).find(|(_, c)| *c == '-')?.0;
    Some((&body[..index], &body[index + 1..]))
}

/// Every class from `class` up to the top of its family.
///
/// For a pair that means up to aces. For anything else the high card is fixed
/// and the kicker climbs, so `KTo+` stops at `KQo` rather than running past the
/// king.
fn walk_upward(class: HandClass) -> Vec<HandClass> {
    let (high, low) = (class.high(), class.low());
    if class.is_pair() {
        return (low.index()..=Rank::Ace.index())
            .filter_map(Rank::from_index)
            .map(|rank| HandClass::new(rank, rank, false))
            .collect();
    }
    (low.index()..high.index())
        .filter_map(Rank::from_index)
        .map(|kicker| HandClass::new(high, kicker, class.is_suited()))
        .collect()
}

/// Every class between two endpoints of the same shape.
fn between(a: HandClass, b: HandClass) -> Option<Vec<HandClass>> {
    if a.is_pair() != b.is_pair() {
        return None;
    }
    if a.is_pair() {
        let (lo, hi) = order(a.low().index(), b.low().index());
        return Some(
            (lo..=hi)
                .filter_map(Rank::from_index)
                .map(|rank| HandClass::new(rank, rank, false))
                .collect(),
        );
    }
    // Both ends must name the same high card and the same suitedness; only the
    // kicker may move.
    if a.high() != b.high() || a.is_suited() != b.is_suited() {
        return None;
    }
    let (lo, hi) = order(a.low().index(), b.low().index());
    Some(
        (lo..=hi)
            .filter_map(Rank::from_index)
            .map(|kicker| HandClass::new(a.high(), kicker, a.is_suited()))
            .collect(),
    )
}

fn order(a: u8, b: u8) -> (u8, u8) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

impl fmt::Display for Range {
    /// Lists the classes carrying weight, strongest first.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut entries: Vec<(HandClass, f64)> = self.entries().collect();
        entries.sort_by_key(|(class, _)| std::cmp::Reverse(*class));
        for (index, (class, weight)) in entries.iter().enumerate() {
            if index > 0 {
                f.write_str(",")?;
            }
            write!(f, "{class}")?;
            if *weight < 1.0 {
                write!(f, ":{weight}")?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::parse_cards;

    fn range(text: &str) -> Range {
        text.parse().unwrap_or_else(|e| panic!("{text}: {e}"))
    }

    fn class(text: &str) -> HandClass {
        text.parse().expect("valid class")
    }

    fn classes(text: &str) -> Vec<HandClass> {
        let mut found: Vec<HandClass> = range(text).entries().map(|(c, _)| c).collect();
        found.sort();
        found
    }

    /// The notation a solver exports its preflop ranges in.
    ///
    /// Checked against a real cutoff opening range, because that is the form
    /// the preflop strategy will arrive in: published ranges rather than a
    /// solve of our own, which reached 845,000 information sets with a quarter
    /// of them untrained and disagreed with itself about king-queen.
    #[test]
    fn a_solver_export_reads_back_as_a_range() {
        let cutoff = "22+, A2s+, K7s+, Q9s+, J9s+, T8s+, 97s+, 87s, 76s,                       A9o+, KTo+, QTo+, JTo";
        let range: Range = cutoff.parse().expect("standard notation");

        let has = |hand: &str| {
            range.weight(hand.parse::<HandClass>().expect("a hand")) > 0.0
        };
        // Named directly, or reached by a `+`.
        assert!(has("AA") && has("22"), "22+ covers every pair");
        assert!(has("A2s") && has("AKs"), "A2s+ climbs to AKs");
        assert!(!has("A2o"), "A9o+ starts at nine, not two");
        assert!(has("A9o") && has("AKo"));
        assert!(has("JTo") && !has("J9o"), "JTo alone");
        assert!(has("76s") && !has("65s"));
        // A `+` on a kicker stops below the high card. There is no need to
        // check for `AAs`: it is not a hand and will not parse, so the notation
        // cannot express the mistake in the first place.

        // Weights survive, for hands a solver plays only part of the time.
        let mixed: Range = "AA, KQo:0.35".parse().expect("weights");
        assert_eq!(mixed.weight("AA".parse().expect("a hand")), 1.0);
        assert_eq!(mixed.weight("KQo".parse().expect("a hand")), 0.35);

        // And the whole thing round-trips through its own notation.
        let again: Range = range.to_string().parse().expect("our own output");
        for class in HandClass::all() {
            assert_eq!(
                again.weight(class),
                range.weight(class),
                "{class} did not survive a round trip"
            );
        }
    }

    #[test]
    fn a_single_class_parses() {
        assert_eq!(classes("AA"), vec![class("AA")]);
        assert_eq!(classes("AKs"), vec![class("AKs")]);
        assert_eq!(classes("72o"), vec![class("72o")]);
    }

    #[test]
    fn pairs_climb_to_aces() {
        let found = classes("TT+");
        assert_eq!(found.len(), 5, "TT, JJ, QQ, KK, AA");
        for text in ["TT", "JJ", "QQ", "KK", "AA"] {
            assert!(found.contains(&class(text)), "missing {text}");
        }
        assert!(!found.contains(&class("99")));
    }

    #[test]
    fn kickers_climb_but_never_past_the_high_card() {
        // KTo+ is KTo, KJo, KQo — it must not run on into KK or AKo.
        let found = classes("KTo+");
        assert_eq!(found.len(), 3);
        for text in ["KTo", "KJo", "KQo"] {
            assert!(found.contains(&class(text)), "missing {text}");
        }
        assert!(!found.contains(&class("KK")), "KTo+ must stop below the pair");
        assert!(!found.contains(&class("AKo")), "KTo+ must not reach aces");
    }

    #[test]
    fn every_suited_ace_is_twelve_classes() {
        let found = classes("A2s+");
        assert_eq!(found.len(), 12, "A2s through AKs");
        assert!(found.contains(&class("A2s")));
        assert!(found.contains(&class("AKs")));
        assert!(!found.contains(&class("AA")));
    }

    #[test]
    fn dashed_ranges_cover_both_ends() {
        assert_eq!(classes("22-55").len(), 4);
        assert_eq!(classes("A2s-A5s").len(), 4);
        // Order does not matter.
        assert_eq!(classes("55-22"), classes("22-55"));
    }

    #[test]
    fn terms_combine() {
        let found = classes("TT+,AKs,72o");
        assert_eq!(found.len(), 7, "five pairs plus two hands");
        assert!(found.contains(&class("AKs")));
        assert!(found.contains(&class("72o")));
    }

    #[test]
    fn weights_apply_to_a_whole_term() {
        let range = range("AA:0.5,KK");
        assert_eq!(range.weight(class("AA")), 0.5);
        assert_eq!(range.weight(class("KK")), 1.0);
        assert_eq!(range.weight(class("QQ")), 0.0);

        let partial = range.combos();
        // Three combinations of aces plus six of kings.
        assert!((partial - 9.0).abs() < 1e-9, "got {partial}");
    }

    #[test]
    fn combination_counts_weight_pairs_correctly() {
        // A single pair is six combinations, a suited hand four, offsuit twelve.
        assert_eq!(range("AA").combos(), 6.0);
        assert_eq!(range("AKs").combos(), 4.0);
        assert_eq!(range("AKo").combos(), 12.0);
        assert_eq!(Range::any().combos(), 1_326.0, "the whole deck");
        assert!((Range::any().fraction() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn an_empty_string_is_an_empty_range() {
        assert!(range("").is_empty());
        assert!(Range::empty().is_empty());
        assert!(!Range::any().is_empty());
    }

    #[test]
    fn ranges_round_trip_through_their_own_notation() {
        for text in ["AA", "TT+,AKs", "A2s+,72o"] {
            let original = range(text);
            let printed = original.to_string();
            assert_eq!(range(&printed), original, "{text} printed as {printed}");
        }
    }

    #[test]
    fn union_takes_the_larger_weight() {
        let merged = range("AA:0.3").union(&range("AA:0.8,KK"));
        assert_eq!(merged.weight(class("AA")), 0.8);
        assert_eq!(merged.weight(class("KK")), 1.0);
    }

    #[test]
    fn malformed_ranges_are_rejected() {
        assert_eq!("AA,,KK".parse::<Range>(), Err(ParseRangeError::EmptyTerm));
        assert!(matches!(
            "XX".parse::<Range>(),
            Err(ParseRangeError::BadClass { .. })
        ));
        assert!(matches!(
            "22-AKs".parse::<Range>(),
            Err(ParseRangeError::MismatchedEndpoints { .. })
        ));
        assert!(matches!(
            "A2s-K5s".parse::<Range>(),
            Err(ParseRangeError::MismatchedEndpoints { .. })
        ));
        assert!(matches!(
            "AA:2.0".parse::<Range>(),
            Err(ParseRangeError::BadWeight { .. })
        ));
        assert!(matches!(
            "AA:high".parse::<Range>(),
            Err(ParseRangeError::BadWeight { .. })
        ));
    }

    #[test]
    fn the_board_blocks_combinations_it_uses() {
        let board = parse_cards("As Kd 7h 2c 9s").expect("valid board");

        // Six combinations of aces in the abstract; with one ace on the board
        // the other three pair up C(3,2) = 3 ways.
        assert_eq!(range("AA").holdings(&board).len(), 3);

        // Offsuit ace-king is the case that catches people out. Twelve combos
        // to begin with; three use the As, three use the Kd — but As-Kd is in
        // both counts, so only five are blocked, not six.
        assert_eq!(range("AKo").holdings(&board).len(), 7);

        // Queens are untouched by this board.
        assert_eq!(range("QQ").holdings(&board).len(), 6);
    }

    #[test]
    fn a_hand_entirely_blocked_contributes_nothing() {
        // Both black aces and both red kings are gone.
        let board = parse_cards("As Ac Kh Kd 2c").expect("valid board");
        let holdings = range("AA").holdings(&board);
        assert_eq!(holdings.len(), 1, "only the red aces remain");
        assert_eq!(range("KK").holdings(&board).len(), 1);
    }

    #[test]
    fn holdings_rank_hands_by_real_strength() {
        // A board with three spades: a suited-ace flush must beat a set, which
        // must beat an overpair.
        let board = parse_cards("Qs 8s 3s 7d 2c").expect("valid board");

        let flush = range("AKs").holdings(&board);
        let set = range("88").holdings(&board);
        let overpair = range("KK").holdings(&board);

        let best = |holdings: &[Holding]| holdings.iter().map(|h| h.strength).max().expect("some");

        assert!(
            best(&flush) > best(&set),
            "the nut flush must beat a set of eights"
        );
        assert!(
            best(&set) > best(&overpair),
            "a set must beat an overpair"
        );
    }

    #[test]
    fn weights_survive_into_holdings() {
        let board = parse_cards("2c 7d 9h Jc 4s").expect("valid board");
        let holdings = range("AA:0.25").holdings(&board);
        assert_eq!(holdings.len(), 6);
        assert!(holdings.iter().all(|h| h.weight == 0.25));
        assert!((range("AA:0.25").combos_on(&board) - 1.5).abs() < 1e-9);
    }

    #[test]
    fn a_full_range_produces_every_unblocked_holding() {
        let board = parse_cards("As Kd 7h 2c 9s").expect("valid board");
        // Five cards are gone, leaving C(47,2) holdings.
        assert_eq!(Range::any().holdings(&board).len(), 47 * 46 / 2);
    }

    #[test]
    #[should_panic(expected = "five cards")]
    fn a_short_board_is_rejected() {
        range("AA").holdings(&parse_cards("As Kd 7h").expect("valid"));
    }

    #[test]
    #[should_panic(expected = "duplicate board card")]
    fn a_repeated_board_card_is_rejected() {
        let board = [
            "As".parse::<Card>().expect("valid"),
            "As".parse::<Card>().expect("valid"),
            "7h".parse::<Card>().expect("valid"),
            "2c".parse::<Card>().expect("valid"),
            "9s".parse::<Card>().expect("valid"),
        ];
        range("KK").holdings(&board);
    }
}
