//! Reading a CoinPoker table (heads-up, no-real-money practice environment)
//! from raw screen pixels.
//!
//! This is a second, independent reader living alongside the original ClubGG
//! one in this crate — not a replacement. The two clients share nothing
//! visually: different card art, different table chrome, different window
//! behaviour. What they share is the underlying *method* (exact template
//! matching on bright, near-neutral card rectangles, corner rank+suit index
//! only, refuse rather than guess), so this module reuses the crate's generic
//! primitives ([`Frame`], [`Gray`], [`Templates`], [`components`],
//! [`best_match`]) and supplies its own geometry and detection.
//!
//! # Cards touch, they don't sit apart
//!
//! CoinPoker draws the hero's two hole cards fanned almost edge to edge (about
//! 2px of overlap), where ClubGG drew them with daylight between them. That
//! means the two hole cards merge into one connected bright region instead of
//! two, and [`detect_card_positions`] has to split that merged blob into two
//! fixed-width cards rather than simply rejecting anything outside a single
//! card's width range like the ClubGG reader does. Board cards are drawn
//! separated by a clear gap and behave exactly like ClubGG's.
//!
//! # The window forgets its size
//!
//! Unlike ClubGG, this client's table window resets to a smaller default size
//! at the start of every new hand, not just once at startup. Whatever drives
//! the table has to re-check and re-apply the window size before *every*
//! capture, not only when it first attaches.

use crate::{best_match, components, Frame, Templates, Thresholds};
use crate::CardRead;
use poker_core::card::{Card, Suit};

/// Card geometry, measured from captures at a 1280x960 table window.
///
/// Pixel-exact at that one size, the same way ClubGG's geometry module is
/// pixel-exact at 1430x1040. A different window size needs re-measuring, not
/// scaling — see the crate-level ClubGG geometry docs for why.
pub mod geometry {
    /// Width of a card face, in pixels.
    pub const CARD_W: usize = 86;
    /// Height of a card face.
    pub const CARD_H: usize = 124;
    /// Accepted width range for a single, non-overlapping card.
    pub const CARD_W_RANGE: (usize, usize) = (70, 100);
    /// Accepted height range for any card (board or hole).
    pub const CARD_H_RANGE: (usize, usize) = (105, 140);
    /// A connected region wider than this is two overlapping hole cards, not
    /// one card — see the module-level docs.
    pub const MAX_SINGLE_CARD_W: usize = CARD_W + 15;
    /// Horizontal inset from the detected edge. Unlike ClubGG's card art,
    /// this deck's corner index starts flush with the card's left edge.
    pub const INSET_X: usize = 0;
    /// Rank glyph, relative to the card's top-left. Wide enough to hold "10"
    /// as well as a single digit or letter.
    pub const RANK_TOP: usize = 8;
    pub const RANK_W: usize = 38;
    pub const RANK_H: usize = 31;
    /// Suit pip, below the rank.
    pub const SUIT_TOP: usize = 38;
    pub const SUIT_W: usize = 34;
    pub const SUIT_H: usize = 36;
}

/// The table window size every capture must be resized to first.
///
/// Chosen deliberately smaller than the 1383x1040 this client can comfortably
/// reach, to leave headroom on smaller displays. Measured clean and
/// undistorted at this size — see the geometry module's card measurements,
/// which were taken at exactly this window size.
pub const TABLE_W: usize = 1280;
pub const TABLE_H: usize = 960;

/// Finds card-sized bright rectangles, splitting any merged hole-card pair
/// into two fixed-width cards.
///
/// This is the one real difference from ClubGG's `detect_cards`: there, a
/// blob outside the single-card width range is simply not a card. Here, a
/// blob *wider* than one card is almost always two overlapping hole cards,
/// so it is split rather than discarded — discarding it would mean the
/// reader never sees hole cards at all, since they always touch.
fn detect_card_positions(frame: &Frame) -> Vec<(usize, usize)> {
    use geometry::*;
    let (w, h) = (frame.width, frame.height);
    let mut mask = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            let (r, g, b) = frame.pixel(x, y);
            let lo = r.min(g).min(b);
            let hi = r.max(g).max(b);
            mask[y * w + x] = lo > 110 && hi - lo < 60;
        }
    }

    let mut positions = Vec::new();
    for b in components(&mask, w, h) {
        if !(CARD_H_RANGE.0..=CARD_H_RANGE.1).contains(&b.height()) {
            continue;
        }
        if b.width() <= MAX_SINGLE_CARD_W {
            if (CARD_W_RANGE.0..=CARD_W_RANGE.1).contains(&b.width()) {
                positions.push((b.x0, b.y0));
            }
        } else {
            // The left card keeps the blob's own left edge; the right card
            // is flush with the blob's right edge. Both CARD_W wide, which
            // is right regardless of exactly how much the two overlap.
            positions.push((b.x0, b.y0));
            positions.push((b.x0 + b.width() - CARD_W, b.y0));
        }
    }
    positions
}

/// Reads the card at a known top-left position.
fn read_card_at(
    frame: &Frame,
    templates: &Templates,
    thresholds: Thresholds,
    x: usize,
    y: usize,
) -> Option<CardRead> {
    use geometry::*;
    let gx = x + INSET_X;

    let (rank, rank_distance, rank_margin) = best_match(
        frame,
        gx,
        y + RANK_TOP,
        RANK_W,
        RANK_H,
        templates.ranks().iter().map(|(r, g)| (*r, g)),
        thresholds.search,
    )?;

    let red = frame.is_red(gx, y + SUIT_TOP, SUIT_W, SUIT_H);
    let candidates = templates
        .suits()
        .iter()
        .filter(|(s, _)| matches!(s, Suit::Hearts | Suit::Diamonds) == red)
        .map(|(s, g)| (*s, g));
    let (suit, suit_distance, suit_margin) =
        best_match(frame, gx, y + SUIT_TOP, SUIT_W, SUIT_H, candidates, thresholds.search)?;

    let distance = rank_distance.max(suit_distance);
    let margin = rank_margin.min(suit_margin);
    let accepted = distance <= thresholds.max_distance && margin >= thresholds.min_margin;

    Some(CardRead {
        x,
        y,
        card: accepted.then(|| Card::new(rank, suit)),
        distance,
        margin,
    })
}

/// Finds every card face in a frame and reads it.
///
/// Results are ordered top to bottom, then left to right — board cards sit
/// higher up than hole cards at this table's layout, so this naturally comes
/// out in "board, then hero" order the same way ClubGG's does.
pub fn read_cards(frame: &Frame, templates: &Templates, thresholds: Thresholds) -> Vec<CardRead> {
    let mut reads: Vec<CardRead> = detect_card_positions(frame)
        .into_iter()
        .filter_map(|(x, y)| read_card_at(frame, templates, thresholds, x, y))
        .collect();
    reads.sort_by_key(|r| (r.y / 80, r.x));
    reads
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_wide_blob_splits_into_two_cards_not_one() {
        // Two 86-wide cards overlapping by 2px, the way this client actually
        // draws hole cards, must come back as two positions, not a single
        // one 170 wide and not discarded outright.
        let w = 260usize;
        let h = 200usize;
        let mut rgb = vec![20u8; w * h * 3]; // dark background (felt)
        // Paint a 170x124 bright merged blob at (40, 30), matching a real
        // measured hole-card pair.
        for y in 30..30 + 124 {
            for x in 40..40 + 170 {
                let i = (y * w + x) * 3;
                rgb[i] = 230;
                rgb[i + 1] = 230;
                rgb[i + 2] = 225;
            }
        }
        let frame = Frame::new(w, h, &rgb);
        let positions = detect_card_positions(&frame);
        assert_eq!(positions.len(), 2, "a merged pair must split into two cards");
        let mut xs: Vec<usize> = positions.iter().map(|(x, _)| *x).collect();
        xs.sort();
        assert_eq!(xs[0], 40, "left card keeps the blob's own left edge");
        assert_eq!(xs[1], 40 + 170 - geometry::CARD_W, "right card is flush with the blob's right edge");
    }

    #[test]
    fn a_lone_board_card_is_not_split() {
        let w = 200usize;
        let h = 200usize;
        let mut rgb = vec![20u8; w * h * 3];
        for y in 30..30 + 124 {
            for x in 40..40 + 86 {
                let i = (y * w + x) * 3;
                rgb[i] = 230;
                rgb[i + 1] = 230;
                rgb[i + 2] = 225;
            }
        }
        let frame = Frame::new(w, h, &rgb);
        let positions = detect_card_positions(&frame);
        assert_eq!(positions, vec![(40, 30)]);
    }
}
