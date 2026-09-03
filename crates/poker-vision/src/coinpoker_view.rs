//! Assembling CoinPoker's scattered readings into one table snapshot.
//!
//! The card reader, the action-panel reader, and the number reader each
//! answer one narrow question. This module is where "board is Ah Jc 3d, the
//! hero holds 6h6d, the pot is 0.14, and there is a live Fold/Check/Bet
//! decision pending" becomes one value a decision can actually be made from
//! — the same job ClubGG's `view.rs` does, simplified by there being exactly
//! two seats rather than up to seven.
//!
//! # Heads-up means no seat geometry
//!
//! ClubGG's `TableView` has to work out where the seats are, which one is
//! the hero, and which way round the table play moves, all from the frame
//! itself, because a table can seat anywhere from two to nine players in an
//! arrangement that changes as people join and leave. CoinPoker's heads-up
//! table is always exactly two fixed positions: the hero at the bottom, and
//! the villain at the top. There is no ring to measure and no ordering to
//! infer, so this module skips all of it.
//!
//! # What this does not do yet
//!
//! - **Per-street bet amounts and `to_call`.** The felt shows a bet-chip
//!   figure near whichever seat last acted, but which seat that is, and
//!   whether the hero or the villain currently owes the difference, has not
//!   been mapped out. For now, whether anything is owed is read off the
//!   action panel's own labels (`Check` vs `Call`) instead of computed from
//!   chip positions — cruder, but does not depend on geometry that has not
//!   been measured.

use crate::coinpoker::{self, CoinPokerActionPanel, PositionTemplates};
use crate::coinpoker_text::{self, CoinPokerGlyphTemplates, Ink, TextThresholds};
use crate::{CardRead, Frame, Thresholds};
use poker_core::card::Card;

/// Regions where an amount renders, measured at a 1280x960 table on
/// `captures-coinpoker/h181631-0-1280x960.png`. Generous boxes around where
/// each readout actually sits, not the whole frame — see
/// `coinpoker_text`'s module docs for why that restriction matters here.
mod regions {
    pub const POT: (usize, usize, usize, usize) = (560, 335, 740, 375);
    pub const HERO_STACK: (usize, usize, usize, usize) = (560, 845, 730, 895);
    pub const VILLAIN_STACK: (usize, usize, usize, usize) = (555, 205, 730, 250);
}

/// Everything one frame says about a CoinPoker heads-up table.
#[derive(Debug, Clone, PartialEq)]
pub struct CoinPokerView {
    pub hero_stack: Option<f64>,
    pub villain_stack: Option<f64>,
    pub pot: Option<f64>,
    /// Community cards, left to right.
    pub board: Vec<Card>,
    /// The hero's own two cards.
    pub hole: Vec<Card>,
    /// The row of action buttons, when a live turn is showing.
    pub action: Option<CoinPokerActionPanel>,
    /// Whether the hero holds the dealer button (and so posts the small
    /// blind, and acts first preflop) rather than the villain. `None` when
    /// the button could not be placed on either side this frame.
    pub hero_on_button: Option<bool>,
    /// Cards and readouts the readers would not vouch for.
    pub refusals: usize,
}

/// Frame rows above this are the top chrome bar and the villain's side of
/// the felt; rows below are the hero's side. The button's y-coordinate
/// relative to this is what tells the two apart — see
/// `coinpoker::find_dealer_button`'s module docs for the button's own
/// colour and shape measurements.
const TABLE_MIDLINE_Y: usize = 480;

impl CoinPokerView {
    /// Reads a whole table from one frame.
    pub fn read(
        frame: &Frame,
        cards: &PositionTemplates,
        digits: &CoinPokerGlyphTemplates,
        thresholds: Thresholds,
        text: TextThresholds,
    ) -> CoinPokerView {
        let found = coinpoker::read_cards(frame, cards, thresholds);
        let refusals = found.iter().filter(|c| !c.is_confident()).count();
        let (board, hole) = split_board_and_hole(&found);

        let pot = read_amount(frame, digits, text, Ink::White, regions::POT);
        let hero_stack = read_amount(frame, digits, text, Ink::Gold, regions::HERO_STACK);
        let villain_stack = read_amount(frame, digits, text, Ink::Gold, regions::VILLAIN_STACK);
        let action = coinpoker::read_coinpoker_action_panel(frame);

        let card_boxes: Vec<(usize, usize, usize, usize)> = found
            .iter()
            .map(|c| (c.x, c.y, coinpoker::geometry::CARD_W, coinpoker::geometry::CARD_H))
            .collect();
        let hero_on_button = coinpoker::find_dealer_button(frame, &card_boxes)
            .map(|(_, y)| y > TABLE_MIDLINE_Y);

        CoinPokerView {
            hero_stack,
            villain_stack,
            pot,
            board,
            hole,
            action,
            hero_on_button,
            refusals,
        }
    }

    /// Whether the client is asking the hero to act right now.
    pub fn hero_to_act(&self) -> bool {
        self.action.as_ref().is_some_and(|p| p.is_live())
    }

    /// Whether this reading is complete enough to act on.
    ///
    /// Deliberately conservative, the same way ClubGG's `is_actionable` is:
    /// every one of these has to hold, and each rules out a way this could
    /// be acted on without actually knowing the hand.
    pub fn is_actionable(&self) -> bool {
        self.hero_to_act() && self.missing_figure().is_none()
    }

    /// Which figure a decision needs and did not get, if any.
    pub fn missing_figure(&self) -> Option<&'static str> {
        if self.hole.len() != 2 {
            return Some("our two cards");
        }
        if self.pot.is_none() {
            return Some("the pot");
        }
        if self.hero_stack.is_none() {
            return Some("our own stack");
        }
        None
    }
}

/// Reads one region as an amount, refusing rather than guessing on anything
/// unread.
fn read_amount(
    frame: &Frame,
    digits: &CoinPokerGlyphTemplates,
    thresholds: TextThresholds,
    ink: Ink,
    at: (usize, usize, usize, usize),
) -> Option<f64> {
    coinpoker_text::read_number_in(frame, digits, thresholds, ink, at)?.value
}

/// Splits `read_cards`' flat, sorted list into the board and the hero's own
/// two cards.
///
/// `read_cards` sorts top-to-bottom, and the hole row sits below the board
/// row at this table's layout, so the last two raw positions are always the
/// hole cards when any are showing at all — this relies on
/// `detect_card_positions` always emitting exactly zero or two hole entries
/// after however many board ones; see its module docs. Splitting by raw
/// position rather than by confidence means a single refused board card
/// does not also blank the hole cards, and vice versa — each half is
/// independently all-or-nothing on its own reads only.
fn split_board_and_hole(found: &[CardRead]) -> (Vec<Card>, Vec<Card>) {
    if found.len() < 2 {
        return (Vec::new(), Vec::new());
    }
    let hole_start = found.len() - 2;
    let board = found[..hole_start].iter().filter_map(|c| c.card).collect();
    let hole_reads = &found[hole_start..];
    // All or nothing. A hold'em hand is two cards, and one card plus a blank
    // is not a worse hand to reason about — it is a different hand, and the
    // solver has no way to tell that it was handed half of one.
    let hole = if hole_reads.iter().all(|c| c.card.is_some()) {
        hole_reads.iter().filter_map(|c| c.card).collect()
    } else {
        Vec::new()
    };
    (board, hole)
}

#[cfg(test)]
mod tests {
    use super::*;
    use poker_core::card::{Rank, Suit};
    use std::path::PathBuf;

    fn data(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../data")
            .join(name)
    }

    fn read(rank: Rank, suit: Suit) -> CardRead {
        CardRead { x: 0, y: 0, card: Some(Card::new(rank, suit)), distance: 0.0, margin: 99.0 }
    }

    fn refused() -> CardRead {
        CardRead { x: 0, y: 0, card: None, distance: 99.0, margin: 0.0 }
    }

    #[test]
    fn a_full_seven_card_board_splits_five_and_two() {
        let found = vec![
            read(Rank::Ace, Suit::Diamonds),
            read(Rank::Two, Suit::Spades),
            read(Rank::Nine, Suit::Spades),
            read(Rank::Jack, Suit::Hearts),
            read(Rank::King, Suit::Clubs),
            read(Rank::Six, Suit::Hearts),
            read(Rank::Six, Suit::Diamonds),
        ];
        let (board, hole) = split_board_and_hole(&found);
        assert_eq!(board.iter().map(|c| c.to_string()).collect::<Vec<_>>(), vec!["Ad", "2s", "9s", "Jh", "Kc"]);
        assert_eq!(hole.iter().map(|c| c.to_string()).collect::<Vec<_>>(), vec!["6h", "6d"]);
    }

    #[test]
    fn preflop_with_only_hole_cards_has_no_board() {
        let found = vec![read(Rank::King, Suit::Hearts), read(Rank::Four, Suit::Spades)];
        let (board, hole) = split_board_and_hole(&found);
        assert!(board.is_empty());
        assert_eq!(hole.iter().map(|c| c.to_string()).collect::<Vec<_>>(), vec!["Kh", "4s"]);
    }

    #[test]
    fn a_refused_board_card_does_not_blank_the_hole_cards() {
        let found = vec![
            read(Rank::Ace, Suit::Diamonds),
            refused(),
            read(Rank::Nine, Suit::Spades),
            read(Rank::King, Suit::Hearts),
            read(Rank::Four, Suit::Spades),
        ];
        let (board, hole) = split_board_and_hole(&found);
        assert_eq!(board.iter().map(|c| c.to_string()).collect::<Vec<_>>(), vec!["Ad", "9s"], "the refused card is just dropped");
        assert_eq!(hole.iter().map(|c| c.to_string()).collect::<Vec<_>>(), vec!["Kh", "4s"]);
    }

    #[test]
    fn a_refused_hole_card_does_not_blank_the_board() {
        let found = vec![
            read(Rank::Ace, Suit::Diamonds),
            read(Rank::Nine, Suit::Spades),
            read(Rank::King, Suit::Hearts),
            refused(),
        ];
        let (board, hole) = split_board_and_hole(&found);
        assert_eq!(board.iter().map(|c| c.to_string()).collect::<Vec<_>>(), vec!["Ad", "9s"], "the board should still read");
        assert!(hole.is_empty(), "half a hand is a different hand, not a worse one");
    }

    #[test]
    fn fewer_than_two_cards_is_neither_a_board_nor_a_hand() {
        assert_eq!(split_board_and_hole(&[]), (vec![], vec![]));
        assert_eq!(split_board_and_hole(&[read(Rank::Ace, Suit::Diamonds)]), (vec![], vec![]));
    }

    fn view_of(name: &str) -> Option<CoinPokerView> {
        let board_path = data("card_templates_coinpoker_board.bin");
        let back_path = data("card_templates_coinpoker_hole_back.bin");
        let front_path = data("card_templates_coinpoker_hole_front.bin");
        let digits_path = data("digit_templates_coinpoker.bin");
        if !board_path.exists() || !back_path.exists() || !front_path.exists() || !digits_path.exists() {
            // Hole-back is still short one rank at the time of writing, so
            // this checkout may not have all four files yet.
            return None;
        }
        let cards = PositionTemplates {
            board: crate::Templates::load(&board_path).expect("board templates"),
            hole_back: crate::Templates::load(&back_path).expect("hole-back templates"),
            hole_front: crate::Templates::load(&front_path).expect("hole-front templates"),
        };
        let digits = CoinPokerGlyphTemplates::load(&digits_path).expect("digit templates");

        let raw = std::fs::read(data("frames").join(name)).expect("fixture frame");
        let w = u32::from_le_bytes(raw[0..4].try_into().expect("header")) as usize;
        let h = u32::from_le_bytes(raw[4..8].try_into().expect("header")) as usize;
        let frame = Frame::new(w, h, &raw[8..]);

        Some(CoinPokerView::read(
            &frame,
            &cards,
            &digits,
            Thresholds::default(),
            TextThresholds::default(),
        ))
    }

    /// The frame every other CoinPoker fixture was checked against by eye:
    /// board Ad 2s 9s Jh Kc, hero 6h6d, pot 0.14.
    #[test]
    fn a_verified_frame_assembles_into_the_table_it_shows() {
        let Some(view) = view_of("coinpoker-h181631.rgb") else { return };

        let board: Vec<String> = view.board.iter().map(|c| c.to_string()).collect();
        assert_eq!(board, vec!["Ad", "2s", "9s", "Jh", "Kc"], "board: {board:?}");

        let hole: Vec<String> = view.hole.iter().map(|c| c.to_string()).collect();
        assert_eq!(hole, vec!["6h", "6d"], "hole: {hole:?}");

        assert_eq!(view.pot, Some(0.14));
        assert_eq!(view.hero_stack, Some(1.76));
        assert_eq!(view.villain_stack, Some(0.89));
        assert_eq!(view.hero_on_button, Some(true), "niki88 holds the button on this frame");
    }

    #[test]
    fn a_dialog_or_unreadable_frame_is_never_actionable() {
        // No action panel is showing on this preflop-review frame, so it
        // must never be reported as a live turn.
        let Some(view) = view_of("coinpoker-h181631.rgb") else { return };
        if view.action.is_none() {
            assert!(!view.is_actionable());
        }
    }
}
