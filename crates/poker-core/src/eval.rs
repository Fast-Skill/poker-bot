//! Five-to-seven card hand evaluation.
//!
//! [`evaluate`] maps any 5, 6, or 7 card hand to a [`HandRank`] that compares
//! correctly against any other: `a > b` exactly when hand `a` beats hand `b`,
//! and `a == b` exactly when they split the pot.
//!
//! # Encoding
//!
//! A [`HandRank`] packs a [`Category`] in bits 20..24 above five 4-bit rank
//! slots in bits 0..20, filled most-significant first and zero-padded. Because
//! the category dominates the integer value and each category always fills its
//! slots the same way, plain integer comparison is a correct hand comparison.
//!
//! # Why this implementation
//!
//! This is a straightforward counts-and-masks evaluator, not a table-driven
//! one. It is fast enough to build everything else on, and — more importantly —
//! it is simple enough to be obviously correct and is pinned down by an
//! exhaustive test over all 2,598,960 five-card hands. When the solver's inner
//! loop needs a faster evaluator, that version gets validated against this one
//! rather than against hand-written expectations.

use crate::card::{Card, Rank, NUM_RANKS, NUM_SUITS};
use std::fmt;

/// The nine hand categories, ordered worst to best.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Category {
    HighCard = 0,
    Pair,
    TwoPair,
    ThreeOfAKind,
    Straight,
    Flush,
    FullHouse,
    FourOfAKind,
    StraightFlush,
}

impl Category {
    pub const ALL: [Category; 9] = [
        Category::HighCard,
        Category::Pair,
        Category::TwoPair,
        Category::ThreeOfAKind,
        Category::Straight,
        Category::Flush,
        Category::FullHouse,
        Category::FourOfAKind,
        Category::StraightFlush,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Category::HighCard => "high card",
            Category::Pair => "pair",
            Category::TwoPair => "two pair",
            Category::ThreeOfAKind => "three of a kind",
            Category::Straight => "straight",
            Category::Flush => "flush",
            Category::FullHouse => "full house",
            Category::FourOfAKind => "four of a kind",
            Category::StraightFlush => "straight flush",
        }
    }
}

impl fmt::Display for Category {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The strength of an evaluated hand. Higher is better; equal values tie.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HandRank(u32);

const CATEGORY_SHIFT: u32 = 20;
const NUM_SLOTS: usize = 5;

impl HandRank {
    /// The lowest possible value, below every real hand. Useful as a fold value
    /// or as the identity when taking a maximum.
    pub const WORST: HandRank = HandRank(0);

    /// Packs a category and its tiebreak ranks, most significant first.
    #[inline]
    fn pack(category: Category, ranks: &[u8]) -> HandRank {
        debug_assert!(ranks.len() <= NUM_SLOTS);
        let mut bits = (category as u32) << CATEGORY_SHIFT;
        for (i, &rank) in ranks.iter().enumerate() {
            debug_assert!((rank as usize) < NUM_RANKS);
            bits |= (rank as u32) << (16 - 4 * i);
        }
        HandRank(bits)
    }

    #[inline]
    pub const fn category(self) -> Category {
        // SAFETY: `pack` is the only constructor and always writes a valid
        // discriminant into bits 20..24.
        unsafe { std::mem::transmute::<u8, Category>((self.0 >> CATEGORY_SHIFT) as u8) }
    }

    /// The raw packed value. Comparable, and compact enough to store in solver
    /// tables.
    #[inline]
    pub const fn to_bits(self) -> u32 {
        self.0
    }

    /// The tiebreak ranks, most significant first, including zero padding.
    fn slots(self) -> [u8; NUM_SLOTS] {
        let mut out = [0u8; NUM_SLOTS];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = ((self.0 >> (16 - 4 * i)) & 0xF) as u8;
        }
        out
    }
}

impl fmt::Debug for HandRank {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HandRank({self})")
    }
}

impl fmt::Display for HandRank {
    /// Renders the category with its tiebreak ranks, e.g. `full house (K over 7)`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if *self == HandRank::WORST {
            return f.write_str("(none)");
        }
        let category = self.category();
        write!(f, "{category}")?;

        let used = match category {
            Category::HighCard | Category::Flush => 5,
            Category::Pair => 4,
            Category::TwoPair | Category::ThreeOfAKind => 3,
            Category::FullHouse => 2,
            Category::FourOfAKind => 2,
            Category::Straight | Category::StraightFlush => 1,
        };
        let slots = self.slots();
        f.write_str(" (")?;
        for (i, &rank) in slots.iter().take(used).enumerate() {
            if i > 0 {
                f.write_str(" ")?;
            }
            match Rank::from_index(rank) {
                Some(r) => write!(f, "{r}")?,
                None => f.write_str("?")?,
            }
        }
        f.write_str(")")
    }
}

/// The 13-bit mask of an ace-low wheel: A, 5, 4, 3, 2.
const WHEEL: u16 = (1 << 12) | 0b1111;

/// The rank at the top of the best straight in `mask`, if any.
///
/// `mask` is a 13-bit set of present ranks. The ace-low wheel yields
/// [`Rank::Five`], since a five-high straight is what it makes.
#[inline]
fn straight_high(mask: u16) -> Option<Rank> {
    // Walk down from ace so the first hit is the best straight.
    for high in (4..NUM_RANKS as u8).rev() {
        let window = 0b11111u16 << (high - 4);
        if mask & window == window {
            return Rank::from_index(high);
        }
    }
    if mask & WHEEL == WHEEL {
        return Some(Rank::Five);
    }
    None
}

/// Writes the highest `n` ranks present in `mask` into `out`, descending.
///
/// Returns how many were written, which is fewer than `n` only when `mask`
/// holds fewer than `n` ranks.
#[inline]
fn top_ranks(mask: u16, n: usize, out: &mut [u8; NUM_SLOTS]) -> usize {
    let mut written = 0;
    let mut rank = NUM_RANKS as u8;
    while rank > 0 && written < n {
        rank -= 1;
        if mask >> rank & 1 != 0 {
            out[written] = rank;
            written += 1;
        }
    }
    written
}

/// Evaluates the best five-card hand available from `cards`.
///
/// # Panics
/// Panics if `cards` does not hold between 5 and 7 cards, or contains a
/// duplicate. Both indicate a bug in the caller rather than a recoverable
/// condition, and a silently wrong hand rank is far more expensive to debug.
pub fn evaluate(cards: &[Card]) -> HandRank {
    assert!(
        (5..=7).contains(&cards.len()),
        "evaluate expects 5 to 7 cards, got {}",
        cards.len()
    );

    let mut rank_counts = [0u8; NUM_RANKS];
    let mut suit_masks = [0u16; NUM_SUITS];
    let mut seen = 0u64;

    for &card in cards {
        debug_assert!(
            seen & card.to_bit() == 0,
            "duplicate card {card} passed to evaluate"
        );
        seen |= card.to_bit();
        rank_counts[card.rank().index() as usize] += 1;
        suit_masks[card.suit().index() as usize] |= 1 << card.rank().index();
    }

    // A flush, if present, settles the hand. With at most 7 cards a flush
    // cannot coexist with quads or a full house: either would need at least 8
    // cards once the maximum possible overlap with the flush suit is taken out.
    // So the answer is the straight flush if there is one, otherwise the flush.
    for mask in suit_masks {
        if mask.count_ones() >= 5 {
            if let Some(high) = straight_high(mask) {
                return HandRank::pack(Category::StraightFlush, &[high.index()]);
            }
            let mut top = [0u8; NUM_SLOTS];
            let n = top_ranks(mask, 5, &mut top);
            return HandRank::pack(Category::Flush, &top[..n]);
        }
    }

    // Gather rank multiplicities, descending, so the first of each is the best.
    let mut quads = [0u8; NUM_SLOTS];
    let mut trips = [0u8; NUM_SLOTS];
    let mut pairs = [0u8; NUM_SLOTS];
    let (mut n_quads, mut n_trips, mut n_pairs) = (0, 0, 0);
    let mut present = 0u16;

    for rank in (0..NUM_RANKS).rev() {
        match rank_counts[rank] {
            0 => continue,
            4 => {
                quads[n_quads] = rank as u8;
                n_quads += 1;
            }
            3 => {
                trips[n_trips] = rank as u8;
                n_trips += 1;
            }
            2 => {
                pairs[n_pairs] = rank as u8;
                n_pairs += 1;
            }
            _ => {}
        }
        present |= 1 << rank;
    }

    /// The highest `n` ranks in `present` excluding those in `exclude`.
    fn kickers(present: u16, exclude: &[u8], n: usize, out: &mut [u8; NUM_SLOTS]) -> usize {
        let mut mask = present;
        for &rank in exclude {
            mask &= !(1 << rank);
        }
        top_ranks(mask, n, out)
    }

    let mut buf = [0u8; NUM_SLOTS];

    if n_quads > 0 {
        let quad = quads[0];
        let n = kickers(present, &[quad], 1, &mut buf);
        let mut slots = [quad, 0];
        if n > 0 {
            slots[1] = buf[0];
        }
        return HandRank::pack(Category::FourOfAKind, &slots);
    }

    // A full house needs trips plus another trips or a pair. With two sets of
    // trips the lower one plays as the pair.
    if n_trips > 0 {
        let trip = trips[0];
        let best_pair = if n_pairs > 0 { Some(pairs[0]) } else { None };
        let second_trips = if n_trips > 1 { Some(trips[1]) } else { None };
        // Whichever is higher fills the pair slot.
        let paired = match (second_trips, best_pair) {
            (Some(t), Some(p)) => Some(t.max(p)),
            (Some(t), None) => Some(t),
            (None, p) => p,
        };
        if let Some(pair) = paired {
            return HandRank::pack(Category::FullHouse, &[trip, pair]);
        }
    }

    if let Some(high) = straight_high(present) {
        return HandRank::pack(Category::Straight, &[high.index()]);
    }

    if n_trips > 0 {
        let trip = trips[0];
        let n = kickers(present, &[trip], 2, &mut buf);
        let mut slots = [trip, 0, 0];
        slots[1..1 + n].copy_from_slice(&buf[..n]);
        return HandRank::pack(Category::ThreeOfAKind, &slots);
    }

    if n_pairs >= 2 {
        let (hi, lo) = (pairs[0], pairs[1]);
        let n = kickers(present, &[hi, lo], 1, &mut buf);
        let mut slots = [hi, lo, 0];
        if n > 0 {
            slots[2] = buf[0];
        }
        return HandRank::pack(Category::TwoPair, &slots);
    }

    if n_pairs == 1 {
        let pair = pairs[0];
        let n = kickers(present, &[pair], 3, &mut buf);
        let mut slots = [pair, 0, 0, 0];
        slots[1..1 + n].copy_from_slice(&buf[..n]);
        return HandRank::pack(Category::Pair, &slots);
    }

    let n = top_ranks(present, 5, &mut buf);
    HandRank::pack(Category::HighCard, &buf[..n])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::parse_cards;

    /// Evaluates a space-separated hand, e.g. `rank("As Ks Qs Js Ts")`.
    fn rank(s: &str) -> HandRank {
        evaluate(&parse_cards(s).expect("valid cards"))
    }

    fn category(s: &str) -> Category {
        rank(s).category()
    }

    #[test]
    fn identifies_every_category_from_five_cards() {
        assert_eq!(category("As Ks Qs Js Ts"), Category::StraightFlush);
        assert_eq!(category("5s 4s 3s 2s As"), Category::StraightFlush);
        assert_eq!(category("9c 9d 9h 9s 2c"), Category::FourOfAKind);
        assert_eq!(category("9c 9d 9h 2s 2c"), Category::FullHouse);
        assert_eq!(category("As Js 9s 5s 3s"), Category::Flush);
        assert_eq!(category("9c 8d 7h 6s 5c"), Category::Straight);
        assert_eq!(category("9c 9d 9h 5s 2c"), Category::ThreeOfAKind);
        assert_eq!(category("9c 9d 5h 5s 2c"), Category::TwoPair);
        assert_eq!(category("9c 9d 7h 5s 2c"), Category::Pair);
        assert_eq!(category("Ac Jd 9h 5s 2c"), Category::HighCard);
    }

    #[test]
    fn category_order_is_strictly_increasing() {
        let ladder = [
            "Ac Jd 9h 5s 2c",
            "9c 9d 7h 5s 2c",
            "9c 9d 5h 5s 2c",
            "9c 9d 9h 5s 2c",
            "9c 8d 7h 6s 5c",
            "As Js 9s 5s 3s",
            "9c 9d 9h 2s 2c",
            "9c 9d 9h 9s 2c",
            "As Ks Qs Js Ts",
        ];
        for pair in ladder.windows(2) {
            let (lo, hi) = (rank(pair[0]), rank(pair[1]));
            assert!(lo < hi, "expected {} < {}, from {:?}", lo, hi, pair);
        }
    }

    #[test]
    fn wheel_is_a_five_high_straight_not_ace_high() {
        let wheel = rank("5c 4d 3h 2s Ac");
        assert_eq!(wheel.category(), Category::Straight);
        // The classic off-by-one: the wheel must lose to a six-high straight.
        assert!(wheel < rank("6c 5d 4h 3s 2c"));
        assert!(wheel < rank("Ac Kd Qh Js Tc"));
    }

    #[test]
    fn steel_wheel_is_the_weakest_straight_flush() {
        let steel = rank("5s 4s 3s 2s As");
        assert_eq!(steel.category(), Category::StraightFlush);
        assert!(steel < rank("6s 5s 4s 3s 2s"));
        assert!(steel < rank("As Ks Qs Js Ts"));
        // But it still beats every quads.
        assert!(steel > rank("Ac Ad Ah As Kc"));
    }

    #[test]
    fn kickers_decide_within_a_category() {
        // Same pair, different kicker.
        assert!(rank("9c 9d Ah 5s 2c") > rank("9c 9d Kh 5s 2c"));
        // Same two pair, different kicker.
        assert!(rank("9c 9d 5h 5s Ac") > rank("9c 9d 5h 5s Kc"));
        // Same trips, different kickers.
        assert!(rank("9c 9d 9h As 2c") > rank("9c 9d 9h Ks 2c"));
        // Same quads, different kicker.
        assert!(rank("9c 9d 9h 9s Ac") > rank("9c 9d 9h 9s Kc"));
        // High card down to the last card.
        assert!(rank("Ac Jd 9h 5s 3c") > rank("Ac Jd 9h 5s 2c"));
    }

    #[test]
    fn higher_pair_beats_better_kicker() {
        assert!(rank("Tc Td 3h 4s 2c") > rank("9c 9d Ah Ks Qc"));
    }

    #[test]
    fn identical_hands_of_different_suits_tie() {
        assert_eq!(rank("Ac Kd 9h 5s 2c"), rank("Ad Kh 9s 5c 2d"));
        assert_eq!(rank("9c 9d 5h 5s 2c"), rank("9h 9s 5c 5d 2h"));
    }

    #[test]
    fn seven_cards_pick_the_best_five() {
        // Board pairs but the flush is the real hand.
        let hand = rank("As Ks 9s 5s 3s 2c 2d");
        assert_eq!(hand.category(), Category::Flush);
        assert_eq!(hand, rank("As Ks 9s 5s 3s"));

        // Straight hidden among seven cards.
        assert_eq!(category("9c 8d 7h 6s 5c Kd Qh"), Category::Straight);

        // Two pair on board plus a third pair: only the top two count.
        let two_pair = rank("Ac Ad Kc Kd 5h 5s 2c");
        assert_eq!(two_pair.category(), Category::TwoPair);
        assert_eq!(two_pair, rank("Ac Ad Kc Kd 5h"));
    }

    #[test]
    fn six_card_hands_are_supported() {
        assert_eq!(category("As Ks Qs Js Ts 2c"), Category::StraightFlush);
        assert_eq!(category("9c 9d 5h 5s 2c 3d"), Category::TwoPair);
    }

    #[test]
    fn two_sets_of_trips_make_a_full_house() {
        // Sevens full of threes: the lower trips plays as the pair.
        let hand = rank("7c 7d 7h 3c 3d 3h 2s");
        assert_eq!(hand.category(), Category::FullHouse);
        assert_eq!(hand, rank("7c 7d 7h 3c 3d"));
        assert!(hand > rank("7c 7d 7h 2c 2d"));
    }

    #[test]
    fn trips_plus_two_pairs_uses_the_higher_pair() {
        let hand = rank("7c 7d 7h Kc Kd 3c 3d");
        assert_eq!(hand.category(), Category::FullHouse);
        assert_eq!(hand, rank("7c 7d 7h Kc Kd"));
    }

    #[test]
    fn quads_on_seven_cards_take_the_best_kicker() {
        let hand = rank("9c 9d 9h 9s Ac Kd 2h");
        assert_eq!(hand.category(), Category::FourOfAKind);
        assert_eq!(hand, rank("9c 9d 9h 9s Ac"));
    }

    #[test]
    fn straight_uses_the_highest_of_overlapping_runs() {
        // 5..9 and 6..T both present; ten-high must win.
        let hand = rank("5c 6d 7h 8s 9c Td 2h");
        assert_eq!(hand.category(), Category::Straight);
        assert_eq!(hand, rank("Tc 9d 8h 7s 6c"));
    }

    #[test]
    fn flush_beats_a_lower_straight_on_the_same_seven_cards() {
        let hand = rank("9s 8s 7s 6s 2s Kd Qh");
        assert_eq!(hand.category(), Category::Flush);
    }

    #[test]
    fn display_is_human_readable() {
        assert_eq!(rank("As Ks Qs Js Ts").to_string(), "straight flush (A)");
        assert_eq!(rank("9c 9d 9h 2s 2c").to_string(), "full house (9 2)");
        assert_eq!(rank("9c 9d 7h 5s 2c").to_string(), "pair (9 7 5 2)");
        assert_eq!(HandRank::WORST.to_string(), "(none)");
    }

    #[test]
    fn worst_is_below_every_real_hand() {
        assert!(HandRank::WORST < rank("7c 5d 4h 3s 2c"));
    }

    #[test]
    #[should_panic(expected = "5 to 7 cards")]
    fn rejects_too_few_cards() {
        evaluate(&parse_cards("As Ks Qs Js").expect("valid"));
    }

    #[test]
    #[should_panic(expected = "5 to 7 cards")]
    fn rejects_too_many_cards() {
        evaluate(&parse_cards("As Ks Qs Js Ts 9s 8s 7s").expect("valid"));
    }

    /// The definitive correctness test: every distinct five-card hand, bucketed
    /// by category, compared against the textbook counts. If any hand were
    /// mis-categorised these totals could not all land.
    #[test]
    fn exhaustive_five_card_category_counts_match_textbook_values() {
        let deck: Vec<Card> = Card::all().collect();
        let mut counts = [0u64; 9];
        let mut hand = [deck[0]; 5];

        for a in 0..48 {
            hand[0] = deck[a];
            for b in a + 1..49 {
                hand[1] = deck[b];
                for c in b + 1..50 {
                    hand[2] = deck[c];
                    for d in c + 1..51 {
                        hand[3] = deck[d];
                        for &card in &deck[d + 1..] {
                            hand[4] = card;
                            counts[evaluate(&hand).category() as usize] += 1;
                        }
                    }
                }
            }
        }

        let expected = [
            (Category::HighCard, 1_302_540),
            (Category::Pair, 1_098_240),
            (Category::TwoPair, 123_552),
            (Category::ThreeOfAKind, 54_912),
            (Category::Straight, 10_200),
            (Category::Flush, 5_108),
            (Category::FullHouse, 3_744),
            (Category::FourOfAKind, 624),
            (Category::StraightFlush, 40),
        ];
        for (category, want) in expected {
            assert_eq!(
                counts[category as usize], want,
                "wrong count for {category}"
            );
        }
        assert_eq!(counts.iter().sum::<u64>(), 2_598_960);
    }

    /// The seven-card equivalent. Slower (133.8M hands), so it is opt-in:
    /// `cargo test --release -- --ignored`.
    #[test]
    #[ignore = "133M hands; run explicitly in release"]
    fn exhaustive_seven_card_category_counts_match_textbook_values() {
        let deck: Vec<Card> = Card::all().collect();
        let mut counts = [0u64; 9];
        let mut hand = [deck[0]; 7];

        for a in 0..46 {
            hand[0] = deck[a];
            for b in a + 1..47 {
                hand[1] = deck[b];
                for c in b + 1..48 {
                    hand[2] = deck[c];
                    for d in c + 1..49 {
                        hand[3] = deck[d];
                        for e in d + 1..50 {
                            hand[4] = deck[e];
                            for f in e + 1..51 {
                                hand[5] = deck[f];
                                for &card in &deck[f + 1..] {
                                    hand[6] = card;
                                    counts[evaluate(&hand).category() as usize] += 1;
                                }
                            }
                        }
                    }
                }
            }
        }

        let expected = [
            (Category::HighCard, 23_294_460),
            (Category::Pair, 58_627_800),
            (Category::TwoPair, 31_433_400),
            (Category::ThreeOfAKind, 6_461_620),
            (Category::Straight, 6_180_020),
            (Category::Flush, 4_047_644),
            (Category::FullHouse, 3_473_184),
            (Category::FourOfAKind, 224_848),
            (Category::StraightFlush, 41_584),
        ];
        for (category, want) in expected {
            assert_eq!(
                counts[category as usize], want,
                "wrong count for {category}"
            );
        }
        assert_eq!(counts.iter().sum::<u64>(), 133_784_560);
    }
}
