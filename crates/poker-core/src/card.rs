//! Card, rank, suit, and card-set primitives.
//!
//! A [`Card`] is a single byte holding an index in `0..52`, laid out as
//! `rank * 4 + suit`. Keeping rank in the high bits means `card.index() >> 2`
//! recovers the rank and cards sort naturally by rank, which the evaluator
//! relies on. Ranks run `0 => Two` through `12 => Ace`; suits are ordered
//! clubs, diamonds, hearts, spades.

use std::fmt;
use std::str::FromStr;

pub const NUM_RANKS: usize = 13;
pub const NUM_SUITS: usize = 4;
pub const NUM_CARDS: usize = 52;

const RANK_CHARS: [u8; NUM_RANKS] = *b"23456789TJQKA";
const SUIT_CHARS: [u8; NUM_SUITS] = *b"cdhs";

/// A card rank, `Two` through `Ace`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Rank {
    Two = 0,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Nine,
    Ten,
    Jack,
    Queen,
    King,
    Ace,
}

impl Rank {
    pub const ALL: [Rank; NUM_RANKS] = [
        Rank::Two,
        Rank::Three,
        Rank::Four,
        Rank::Five,
        Rank::Six,
        Rank::Seven,
        Rank::Eight,
        Rank::Nine,
        Rank::Ten,
        Rank::Jack,
        Rank::Queen,
        Rank::King,
        Rank::Ace,
    ];

    /// Builds a rank from its `0..13` index.
    #[inline]
    pub const fn from_index(index: u8) -> Option<Rank> {
        if index as usize >= NUM_RANKS {
            return None;
        }
        // SAFETY: bounds checked above, and `Rank` is a contiguous `repr(u8)`
        // enum covering exactly 0..13.
        Some(unsafe { std::mem::transmute::<u8, Rank>(index) })
    }

    #[inline]
    pub const fn index(self) -> u8 {
        self as u8
    }

    /// The ASCII character for this rank, e.g. `'T'` for [`Rank::Ten`].
    #[inline]
    pub const fn to_char(self) -> char {
        RANK_CHARS[self as usize] as char
    }

    /// Parses a rank character. Accepts either case.
    pub fn from_char(c: char) -> Option<Rank> {
        let upper = c.to_ascii_uppercase() as u8;
        let idx = RANK_CHARS.iter().position(|&r| r == upper)?;
        Rank::from_index(idx as u8)
    }
}

impl fmt::Display for Rank {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_char())
    }
}

/// A card suit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Suit {
    Clubs = 0,
    Diamonds,
    Hearts,
    Spades,
}

impl Suit {
    pub const ALL: [Suit; NUM_SUITS] = [Suit::Clubs, Suit::Diamonds, Suit::Hearts, Suit::Spades];

    /// Builds a suit from its `0..4` index.
    #[inline]
    pub const fn from_index(index: u8) -> Option<Suit> {
        if index as usize >= NUM_SUITS {
            return None;
        }
        // SAFETY: bounds checked above, and `Suit` is a contiguous `repr(u8)`
        // enum covering exactly 0..4.
        Some(unsafe { std::mem::transmute::<u8, Suit>(index) })
    }

    #[inline]
    pub const fn index(self) -> u8 {
        self as u8
    }

    #[inline]
    pub const fn to_char(self) -> char {
        SUIT_CHARS[self as usize] as char
    }

    /// Parses a suit character. Accepts either case.
    pub fn from_char(c: char) -> Option<Suit> {
        let lower = c.to_ascii_lowercase() as u8;
        let idx = SUIT_CHARS.iter().position(|&s| s == lower)?;
        Suit::from_index(idx as u8)
    }
}

impl fmt::Display for Suit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_char())
    }
}

/// A single playing card, stored as an index in `0..52`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Card(u8);

impl Card {
    /// Builds a card from a rank and suit.
    #[inline]
    pub const fn new(rank: Rank, suit: Suit) -> Card {
        Card((rank as u8) * 4 + suit as u8)
    }

    /// Builds a card from its `0..52` index.
    #[inline]
    pub const fn from_index(index: u8) -> Option<Card> {
        if index as usize >= NUM_CARDS {
            return None;
        }
        Some(Card(index))
    }

    /// Builds a card from an index without checking bounds.
    ///
    /// # Safety
    /// `index` must be less than 52. A larger value produces a card that will
    /// index out of bounds in the evaluator's lookup tables.
    #[inline]
    pub const unsafe fn from_index_unchecked(index: u8) -> Card {
        Card(index)
    }

    #[inline]
    pub const fn index(self) -> u8 {
        self.0
    }

    #[inline]
    pub const fn rank(self) -> Rank {
        // SAFETY: `self.0 < 52`, so `self.0 >> 2 < 13`.
        unsafe { std::mem::transmute::<u8, Rank>(self.0 >> 2) }
    }

    #[inline]
    pub const fn suit(self) -> Suit {
        // SAFETY: masking with 3 yields a value in 0..4.
        unsafe { std::mem::transmute::<u8, Suit>(self.0 & 3) }
    }

    /// This card as a one-hot bit in a 52-bit [`CardSet`] mask.
    #[inline]
    pub const fn to_bit(self) -> u64 {
        1u64 << self.0
    }

    /// Every card in the deck, in index order.
    pub fn all() -> impl Iterator<Item = Card> {
        (0..NUM_CARDS as u8).map(Card)
    }
}

impl fmt::Display for Card {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.rank().to_char(), self.suit().to_char())
    }
}

impl fmt::Debug for Card {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The compact form is far more readable in test failures than a
        // derived struct dump.
        write!(f, "{self}")
    }
}

/// Why a string could not be parsed as a card.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseCardError {
    /// The input was not exactly two characters.
    WrongLength(usize),
    /// The first character was not a valid rank.
    BadRank(char),
    /// The second character was not a valid suit.
    BadSuit(char),
}

impl fmt::Display for ParseCardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseCardError::WrongLength(n) => {
                write!(f, "expected 2 characters (e.g. \"As\"), got {n}")
            }
            ParseCardError::BadRank(c) => write!(f, "invalid rank character {c:?}"),
            ParseCardError::BadSuit(c) => write!(f, "invalid suit character {c:?}"),
        }
    }
}

impl std::error::Error for ParseCardError {}

impl FromStr for Card {
    type Err = ParseCardError;

    /// Parses a card such as `"As"`, `"Td"`, or `"2c"`.
    fn from_str(s: &str) -> Result<Card, ParseCardError> {
        let mut chars = s.chars();
        let (Some(r), Some(su), None) = (chars.next(), chars.next(), chars.next()) else {
            return Err(ParseCardError::WrongLength(s.chars().count()));
        };
        let rank = Rank::from_char(r).ok_or(ParseCardError::BadRank(r))?;
        let suit = Suit::from_char(su).ok_or(ParseCardError::BadSuit(su))?;
        Ok(Card::new(rank, suit))
    }
}

/// A set of distinct cards held as a 52-bit mask.
///
/// Used for dead-card tracking, deck state, and board/hole combinations, where
/// set operations need to be single instructions.
#[derive(Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct CardSet(u64);

impl CardSet {
    /// The empty set.
    #[inline]
    pub const fn empty() -> CardSet {
        CardSet(0)
    }

    /// A full 52-card deck.
    #[inline]
    pub const fn full_deck() -> CardSet {
        CardSet((1u64 << NUM_CARDS) - 1)
    }

    /// Wraps a raw 52-bit mask, ignoring any bits at or above bit 52.
    #[inline]
    pub const fn from_mask(mask: u64) -> CardSet {
        CardSet(mask & ((1u64 << NUM_CARDS) - 1))
    }

    #[inline]
    pub const fn mask(self) -> u64 {
        self.0
    }

    #[inline]
    pub const fn contains(self, card: Card) -> bool {
        self.0 & card.to_bit() != 0
    }

    /// Adds `card`, returning `true` if it was not already present.
    #[inline]
    pub fn insert(&mut self, card: Card) -> bool {
        let had = self.contains(card);
        self.0 |= card.to_bit();
        !had
    }

    /// Removes `card`, returning `true` if it was present.
    #[inline]
    pub fn remove(&mut self, card: Card) -> bool {
        let had = self.contains(card);
        self.0 &= !card.to_bit();
        had
    }

    #[inline]
    pub const fn len(self) -> u32 {
        self.0.count_ones()
    }

    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    #[inline]
    pub const fn union(self, other: CardSet) -> CardSet {
        CardSet(self.0 | other.0)
    }

    #[inline]
    pub const fn intersection(self, other: CardSet) -> CardSet {
        CardSet(self.0 & other.0)
    }

    /// The cards in `self` that are not in `other`.
    #[inline]
    pub const fn difference(self, other: CardSet) -> CardSet {
        CardSet(self.0 & !other.0)
    }

    #[inline]
    pub const fn intersects(self, other: CardSet) -> bool {
        self.0 & other.0 != 0
    }

    /// The 13-bit rank mask of the cards of `suit` in this set.
    ///
    /// Bit `r` is set when the card of rank `r` and suit `suit` is present.
    /// This is the flush-detection primitive the evaluator uses.
    #[inline]
    pub const fn rank_mask(self, suit: Suit) -> u16 {
        // Cards of one suit sit every 4th bit; compress them to bits 0..13.
        let mut mask = 0u16;
        let mut r = 0;
        while r < NUM_RANKS {
            if self.0 >> (r * 4 + suit as usize) & 1 != 0 {
                mask |= 1 << r;
            }
            r += 1;
        }
        mask
    }

    /// Iterates the cards in ascending index order.
    pub fn iter(self) -> CardSetIter {
        CardSetIter(self.0)
    }
}

impl fmt::Debug for CardSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[")?;
        for (i, card) in self.iter().enumerate() {
            if i > 0 {
                f.write_str(" ")?;
            }
            write!(f, "{card}")?;
        }
        f.write_str("]")
    }
}

impl FromIterator<Card> for CardSet {
    fn from_iter<I: IntoIterator<Item = Card>>(iter: I) -> CardSet {
        let mut set = CardSet::empty();
        for card in iter {
            set.insert(card);
        }
        set
    }
}

impl IntoIterator for CardSet {
    type Item = Card;
    type IntoIter = CardSetIter;

    fn into_iter(self) -> CardSetIter {
        self.iter()
    }
}

/// Iterator over the cards of a [`CardSet`], in ascending index order.
#[derive(Debug, Clone)]
pub struct CardSetIter(u64);

impl Iterator for CardSetIter {
    type Item = Card;

    #[inline]
    fn next(&mut self) -> Option<Card> {
        if self.0 == 0 {
            return None;
        }
        let idx = self.0.trailing_zeros() as u8;
        self.0 &= self.0 - 1;
        Some(Card(idx))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.0.count_ones() as usize;
        (n, Some(n))
    }
}

impl ExactSizeIterator for CardSetIter {}

/// Parses a list of cards, e.g. `"As Kd 7c"`.
///
/// Cards may be separated by whitespace or run together in the usual poker
/// shorthand, so `"AhKh"`, `"Ah Kh"`, and `"AsAdKsKd"` all parse. This is
/// unambiguous because a card is always exactly two characters.
///
/// Returns an error on the first unparseable card. Duplicates are rejected:
/// every caller here treats a repeated card as a bug rather than as input.
pub fn parse_cards(s: &str) -> Result<Vec<Card>, ParseCardsError> {
    let mut cards = Vec::new();
    let mut seen = CardSet::empty();

    for token in s.split_whitespace() {
        let chars: Vec<char> = token.chars().collect();
        if chars.is_empty() || chars.len() % 2 != 0 {
            return Err(ParseCardsError {
                token: token.to_string(),
                kind: ParseCardsErrorKind::Invalid(ParseCardError::WrongLength(chars.len())),
            });
        }

        for pair in chars.chunks(2) {
            let text: String = pair.iter().collect();
            let card = text.parse::<Card>().map_err(|source| ParseCardsError {
                token: text.clone(),
                kind: ParseCardsErrorKind::Invalid(source),
            })?;
            if !seen.insert(card) {
                return Err(ParseCardsError {
                    token: text,
                    kind: ParseCardsErrorKind::Duplicate,
                });
            }
            cards.push(card);
        }
    }

    Ok(cards)
}

/// Why a card list could not be parsed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseCardsError {
    pub token: String,
    pub kind: ParseCardsErrorKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseCardsErrorKind {
    Invalid(ParseCardError),
    Duplicate,
}

impl fmt::Display for ParseCardsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            ParseCardsErrorKind::Invalid(source) => {
                write!(f, "could not parse {:?}: {source}", self.token)
            }
            ParseCardsErrorKind::Duplicate => write!(f, "duplicate card {:?}", self.token),
        }
    }
}

impl std::error::Error for ParseCardsError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn card_index_layout_is_rank_major() {
        // The evaluator depends on this exact layout; pin it down.
        assert_eq!(Card::new(Rank::Two, Suit::Clubs).index(), 0);
        assert_eq!(Card::new(Rank::Two, Suit::Spades).index(), 3);
        assert_eq!(Card::new(Rank::Three, Suit::Clubs).index(), 4);
        assert_eq!(Card::new(Rank::Ace, Suit::Spades).index(), 51);
    }

    #[test]
    fn every_index_round_trips_through_rank_and_suit() {
        for i in 0..NUM_CARDS as u8 {
            let card = Card::from_index(i).expect("index in range");
            assert_eq!(card.index(), i);
            assert_eq!(Card::new(card.rank(), card.suit()), card);
        }
    }

    #[test]
    fn from_index_rejects_out_of_range() {
        assert!(Card::from_index(51).is_some());
        assert!(Card::from_index(52).is_none());
        assert!(Card::from_index(255).is_none());
    }

    #[test]
    fn all_yields_the_whole_deck_without_duplicates() {
        let set: CardSet = Card::all().collect();
        assert_eq!(set.len(), 52);
        assert_eq!(set, CardSet::full_deck());
    }

    #[test]
    fn display_and_parse_round_trip_for_every_card() {
        for card in Card::all() {
            let text = card.to_string();
            assert_eq!(text.len(), 2);
            assert_eq!(text.parse::<Card>(), Ok(card));
        }
    }

    #[test]
    fn parse_accepts_mixed_case() {
        let ace_spades = Card::new(Rank::Ace, Suit::Spades);
        assert_eq!("As".parse::<Card>(), Ok(ace_spades));
        assert_eq!("aS".parse::<Card>(), Ok(ace_spades));
        assert_eq!("AS".parse::<Card>(), Ok(ace_spades));
        assert_eq!("as".parse::<Card>(), Ok(ace_spades));
    }

    #[test]
    fn parse_rejects_malformed_input() {
        assert_eq!("".parse::<Card>(), Err(ParseCardError::WrongLength(0)));
        assert_eq!("A".parse::<Card>(), Err(ParseCardError::WrongLength(1)));
        assert_eq!("Ass".parse::<Card>(), Err(ParseCardError::WrongLength(3)));
        assert_eq!("1s".parse::<Card>(), Err(ParseCardError::BadRank('1')));
        assert_eq!("Ax".parse::<Card>(), Err(ParseCardError::BadSuit('x')));
    }

    #[test]
    fn rank_ordering_puts_ace_high() {
        assert!(Rank::Ace > Rank::King);
        assert!(Rank::Three > Rank::Two);
        assert_eq!(Rank::ALL.len(), NUM_RANKS);
        for (i, rank) in Rank::ALL.iter().enumerate() {
            assert_eq!(rank.index() as usize, i);
        }
    }

    #[test]
    fn card_set_insert_and_remove_report_membership_changes() {
        let mut set = CardSet::empty();
        let ace = Card::new(Rank::Ace, Suit::Spades);

        assert!(set.insert(ace), "first insert is a change");
        assert!(!set.insert(ace), "second insert is a no-op");
        assert!(set.contains(ace));
        assert_eq!(set.len(), 1);

        assert!(set.remove(ace), "removing a present card is a change");
        assert!(!set.remove(ace), "removing an absent card is a no-op");
        assert!(set.is_empty());
    }

    #[test]
    fn card_set_iterates_in_index_order() {
        let cards = parse_cards("As 2c Td").expect("valid");
        let set: CardSet = cards.iter().copied().collect();
        let seen: Vec<Card> = set.iter().collect();

        let mut expected = cards.clone();
        expected.sort();
        assert_eq!(seen, expected);
        assert_eq!(seen.len(), set.len() as usize);
    }

    #[test]
    fn card_set_algebra_matches_expectations() {
        let a: CardSet = parse_cards("As Ks Qs").expect("valid").into_iter().collect();
        let b: CardSet = parse_cards("Qs Js Ts").expect("valid").into_iter().collect();

        assert_eq!(a.union(b).len(), 5);
        assert_eq!(a.intersection(b).len(), 1);
        assert_eq!(a.difference(b).len(), 2);
        assert!(a.intersects(b));

        let disjoint: CardSet = parse_cards("2c 3c").expect("valid").into_iter().collect();
        assert!(!a.intersects(disjoint));
        assert_eq!(a.difference(a), CardSet::empty());
    }

    #[test]
    fn rank_mask_isolates_one_suit() {
        let set: CardSet = parse_cards("As Ks Qh 2s").expect("valid").into_iter().collect();

        let spades = set.rank_mask(Suit::Spades);
        assert_eq!(spades.count_ones(), 3);
        assert_ne!(spades & (1 << Rank::Ace.index()), 0);
        assert_ne!(spades & (1 << Rank::King.index()), 0);
        assert_ne!(spades & (1 << Rank::Two.index()), 0);
        assert_eq!(spades & (1 << Rank::Queen.index()), 0, "Qh is not a spade");

        assert_eq!(set.rank_mask(Suit::Hearts), 1 << Rank::Queen.index());
        assert_eq!(set.rank_mask(Suit::Clubs), 0);
    }

    #[test]
    fn rank_mask_of_full_deck_is_all_ranks() {
        for suit in Suit::ALL {
            assert_eq!(CardSet::full_deck().rank_mask(suit), 0x1FFF);
        }
    }

    #[test]
    fn parse_cards_reads_a_whitespace_separated_list() {
        let cards = parse_cards("As Kd 7c").expect("valid");
        assert_eq!(cards.len(), 3);
        assert_eq!(cards[0], Card::new(Rank::Ace, Suit::Spades));
        assert_eq!(cards[2], Card::new(Rank::Seven, Suit::Clubs));
        assert_eq!(parse_cards("").expect("empty is valid"), vec![]);
    }

    #[test]
    fn parse_cards_accepts_run_together_poker_shorthand() {
        // "AhKh" is how hands are written everywhere in poker.
        assert_eq!(parse_cards("AhKh").expect("valid"), parse_cards("Ah Kh").expect("valid"));
        assert_eq!(
            parse_cards("AsAdKsKd").expect("valid"),
            parse_cards("As Ad Ks Kd").expect("valid"),
            "an Omaha hand"
        );
        // Mixed styles in one string.
        assert_eq!(
            parse_cards("AhKh 7c2d").expect("valid").len(),
            4,
            "two two-card hands"
        );
    }

    #[test]
    fn parse_cards_rejects_odd_length_tokens() {
        let err = parse_cards("AhK").expect_err("dangling rank");
        assert_eq!(err.token, "AhK");
        assert!(matches!(
            err.kind,
            ParseCardsErrorKind::Invalid(ParseCardError::WrongLength(3))
        ));
    }

    #[test]
    fn parse_cards_catches_duplicates_inside_a_single_token() {
        let err = parse_cards("AhAh").expect_err("same card twice");
        assert_eq!(err.kind, ParseCardsErrorKind::Duplicate);
        assert_eq!(err.token, "Ah", "reports the card, not the whole token");
    }

    #[test]
    fn parse_cards_rejects_duplicates() {
        let err = parse_cards("As Kd As").expect_err("duplicate should fail");
        assert_eq!(err.kind, ParseCardsErrorKind::Duplicate);
        assert_eq!(err.token, "As");
    }

    #[test]
    fn parse_cards_reports_the_offending_token() {
        let err = parse_cards("As Xd 7c").expect_err("bad token should fail");
        assert_eq!(err.token, "Xd");
        assert!(matches!(err.kind, ParseCardsErrorKind::Invalid(_)));
    }
}
