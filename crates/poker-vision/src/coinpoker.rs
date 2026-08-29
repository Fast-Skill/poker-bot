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
    /// Suit pip dimensions, below the rank. One fixed size for every
    /// position — a stored template is one fixed size, so only *where* the
    /// window starts can differ between positions, never how big it is (see
    /// [`Position::suit_offset`]). This deck also draws a second, larger
    /// suit pip lower on the card, on both board and hole cards; a window
    /// generous enough to avoid clipping every position's actual pip reaches
    /// far enough down to bleed into that second one, which does not just
    /// look messier, it corrupts the pixel comparison. Hence a tight window
    /// with a per-position offset, rather than one generous shared one.
    pub const SUIT_W: usize = 34;
    pub const SUIT_H: usize = 27;
}

/// Which of the three positions a detected card came from.
///
/// A board card and the two hole cards are not just positioned differently
/// on screen, they are laid out differently *within their own corner index*:
/// measured offsets for the suit pip differ by several pixels between all
/// three, and the hole cards' rank and suit glyphs render distinctly enough
/// between the fanned-under (back) and fanned-over (front) card that a
/// template built from one does not reliably match the other — the front
/// card is drawn with what looks like a shadow or slight blend from
/// overlapping the back one. So each position gets its own template set,
/// not just its own offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Position {
    Board,
    /// The hole card fanned underneath — its own left edge is the blob's
    /// left edge.
    HoleBack,
    /// The hole card fanned on top — its own left edge sits `CARD_W` in
    /// from the blob's right edge.
    HoleFront,
}

impl Position {
    /// Where the suit pip starts, `(dx, top)` — `dx` relative to the card's
    /// own left edge (may be negative: the front card's pip sits slightly
    /// left of where its detected edge would suggest), `top` from the
    /// card's own top edge. Measured per position, not shared, because a
    /// single compromise offset either clips one position's pip or bleeds
    /// into the second, larger pip further down the card — see
    /// [`geometry::SUIT_W`].
    fn suit_offset(self) -> (i32, usize) {
        match self {
            Position::Board => (0, 43),
            Position::HoleBack => (7, 49),
            Position::HoleFront => (-5, 44),
        }
    }
}

/// The table window size every capture must be resized to first.
///
/// Chosen deliberately smaller than the 1383x1040 this client can comfortably
/// reach, to leave headroom on smaller displays. Measured clean and
/// undistorted at this size — see the geometry module's card measurements,
/// which were taken at exactly this window size.
pub const TABLE_W: usize = 1280;
pub const TABLE_H: usize = 960;

/// The three independent template sets a reading needs — one per
/// [`Position`]. See [`Position`]'s docs for why a shared set does not work.
pub struct PositionTemplates {
    pub board: Templates,
    pub hole_back: Templates,
    pub hole_front: Templates,
}

impl PositionTemplates {
    fn for_position(&self, position: Position) -> &Templates {
        match position {
            Position::Board => &self.board,
            Position::HoleBack => &self.hole_back,
            Position::HoleFront => &self.hole_front,
        }
    }
}

/// A detected card's top-left corner and which position it came from.
struct Detected {
    x: usize,
    y: usize,
    position: Position,
}

/// Finds card-sized bright rectangles, splitting any merged hole-card pair
/// into two fixed-width cards.
///
/// This is the one real difference from ClubGG's `detect_cards`: there, a
/// blob outside the single-card width range is simply not a card. Here, a
/// blob *wider* than one card is almost always two overlapping hole cards,
/// so it is split rather than discarded — discarding it would mean the
/// reader never sees hole cards at all, since they always touch.
fn detect_card_positions(frame: &Frame) -> Vec<Detected> {
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
                positions.push(Detected { x: b.x0, y: b.y0, position: Position::Board });
            }
        } else {
            // The left card keeps the blob's own left edge; the right card
            // is flush with the blob's right edge. Both CARD_W wide, which
            // is right regardless of exactly how much the two overlap.
            positions.push(Detected { x: b.x0, y: b.y0, position: Position::HoleBack });
            positions.push(Detected {
                x: b.x0 + b.width() - CARD_W,
                y: b.y0,
                position: Position::HoleFront,
            });
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
    position: Position,
) -> Option<CardRead> {
    use geometry::*;
    let gx = x + INSET_X;
    let (suit_dx, suit_top) = position.suit_offset();
    let sx = (x as i32 + suit_dx).max(0) as usize;

    let (rank, rank_distance, rank_margin) = best_match(
        frame,
        gx,
        y + RANK_TOP,
        RANK_W,
        RANK_H,
        templates.ranks().iter().map(|(r, g)| (*r, g)),
        thresholds.search,
    )?;

    let red = frame.is_red(sx, y + suit_top, SUIT_W, SUIT_H);
    let candidates = templates
        .suits()
        .iter()
        .filter(|(s, _)| matches!(s, Suit::Hearts | Suit::Diamonds) == red)
        .map(|(s, g)| (*s, g));
    let (suit, suit_distance, suit_margin) =
        best_match(frame, sx, y + suit_top, SUIT_W, SUIT_H, candidates, thresholds.search)?;

    let distance = rank_distance.max(suit_distance);
    let margin = rank_margin.min(suit_margin);
    let accepted = distance <= thresholds.max_distance && margin >= thresholds.min_margin;

    if std::env::var("COINPOKER_DEBUG").is_ok() {
        eprintln!(
            "x={x} y={y} rank={rank:?} rank_d={rank_distance:.1} rank_m={rank_margin:.1}  suit={suit:?} suit_d={suit_distance:.1} suit_m={suit_margin:.1}"
        );
    }

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
pub fn read_cards(frame: &Frame, templates: &PositionTemplates, thresholds: Thresholds) -> Vec<CardRead> {
    let mut reads: Vec<CardRead> = detect_card_positions(frame)
        .into_iter()
        .filter_map(|d| {
            read_card_at(
                frame,
                templates.for_position(d.position),
                thresholds,
                d.x,
                d.y,
                d.position,
            )
        })
        .collect();
    reads.sort_by_key(|r| (r.y / 80, r.x));
    reads
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn data(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../data")
            .join(name)
    }

    /// The template file is only as good as what it reads back correctly.
    /// This checks it against a frame whose contents were confirmed by eye
    /// against the live client at capture time, not against anything the
    /// reader itself produced - see the project's "verify the instrument
    /// first" rule.
    /// Loads the three position-specific template files, or `None` if this
    /// checkout hasn't built them yet — in which case there is nothing to
    /// verify against.
    fn load_templates() -> Option<PositionTemplates> {
        let paths = [
            "card_templates_coinpoker_board.bin",
            "card_templates_coinpoker_hole_back.bin",
            "card_templates_coinpoker_hole_front.bin",
        ]
        .map(data);
        if paths.iter().any(|p| !p.exists()) {
            return None;
        }
        Some(PositionTemplates {
            board: Templates::load(&paths[0]).expect("board templates should load"),
            hole_back: Templates::load(&paths[1]).expect("hole-back templates should load"),
            hole_front: Templates::load(&paths[2]).expect("hole-front templates should load"),
        })
    }

    #[test]
    fn reads_a_known_table_correctly() {
        let Some(templates) = load_templates() else {
            return;
        };

        let raw = std::fs::read(data("frames/coinpoker-h181631.rgb")).expect("fixture frame");
        let width = u32::from_le_bytes(raw[0..4].try_into().unwrap()) as usize;
        let height = u32::from_le_bytes(raw[4..8].try_into().unwrap()) as usize;
        let frame = Frame::new(width, height, &raw[8..]);

        let reads = read_cards(&frame, &templates, Thresholds::default());
        let cards: Vec<String> = reads
            .iter()
            .map(|r| match r.card {
                Some(c) => c.to_string(),
                None => format!("REFUSED({:.1},{:.1})", r.distance, r.margin),
            })
            .collect();

        // Board: Ad 2s 9s Jh Kc. Hero: 6h 6d. Confirmed by eye against the
        // live CoinPoker client at capture time (2026-08-29).
        assert_eq!(
            cards,
            vec!["Ad", "2s", "9s", "Jh", "Kc", "6h", "6d"],
            "misread one of the seven cards - got {cards:?}"
        );
    }

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
        assert_eq!(positions[0].position, Position::HoleBack, "left card is the back one");
        assert_eq!(positions[1].position, Position::HoleFront, "right card is the front one");
        let mut xs: Vec<usize> = positions.iter().map(|d| d.x).collect();
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
        assert_eq!(positions.len(), 1);
        assert_eq!((positions[0].x, positions[0].y), (40, 30));
        assert_eq!(positions[0].position, Position::Board, "a lone card is a board card, not a hole card");
    }
}
