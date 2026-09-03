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

use crate::{best_match, components, ActionButton, Bounds, Frame, Templates, Thresholds};
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

/// The row of action buttons, if the hero has a live decision pending.
///
/// # Colour, not label, tells the buttons apart
///
/// ClubGG draws all three buttons the same red and distinguishes them only by
/// label text (`Fold` vs `Check` vs `Raise to`), which is why its reader
/// measures label width. This client draws each button its own colour
/// instead — fold red, check-or-call green, bet-or-raise orange — measured at
/// a 1280x960 table as roughly (207,50,65), (33,159,132) and (238,136,57).
/// That makes the three buttons distinguishable by colour alone, with no need
/// to read text just to know which button is which.
///
/// # Why this refuses rather than tells "waiting" from "nothing showing"
///
/// When it is not the hero's turn, this client either shows nothing or shows
/// a greyed, disabled-looking row (armed for when the turn does arrive, the
/// same idea as ClubGG's `Check / Fold`). Neither matches these colour
/// thresholds, so both come back `None` here. That conflates two situations
/// a caller might eventually want to tell apart, but for the one thing that
/// actually matters — is this safe to press right now — they are the same
/// answer, and refusing on both is the safe default until there is a reason
/// to do more.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoinPokerActionPanel {
    pub fold: Option<ActionButton>,
    /// `Check` when there is nothing to call, `Call` when there is.
    pub passive: Option<ActionButton>,
    /// `Bet` or `Raise`. Absent when there is nothing left to raise with.
    pub aggressive: Option<ActionButton>,
}

impl CoinPokerActionPanel {
    /// Whether this is a live turn worth acting on at all.
    ///
    /// Fold and the passive option are the two buttons a genuine turn always
    /// carries — even a hero with nothing left to raise still sees `Fold`
    /// and `Check`/`Call`. `aggressive` alone is never enough by itself to
    /// call this a turn.
    pub fn is_live(&self) -> bool {
        self.fold.is_some() && self.passive.is_some()
    }
}

/// Action button geometry, measured at a 1280x960 table.
mod action_geometry {
    /// Buttons sit in the bottom fraction of the window; scanning only this
    /// band keeps the felt and the hole cards' art out of the colour search.
    pub const SCAN_TOP_FRACTION: f64 = 0.85;
    /// A button is roughly 161 wide. Generous on both sides since a
    /// single-line label (`Fold`, `Check`) and a two-line one (`Bet 0.02`)
    /// can split the coloured region into a taller or shorter connected
    /// blob depending on how the text divides it — see the module test
    /// `label_lines_do_not_change_which_button_is_found`.
    pub const WIDTH_RANGE: (usize, usize) = (100, 220);
    pub const HEIGHT_RANGE: (usize, usize) = (35, 100);
}

/// Finds the largest connected region matching a colour test, within the
/// button band at the bottom of the frame.
fn largest_colour_region(
    frame: &Frame,
    is_match: impl Fn(i16, i16, i16) -> bool,
) -> Option<ActionButton> {
    use action_geometry::*;
    let (w, h) = (frame.width, frame.height);
    let top = (h as f64 * SCAN_TOP_FRACTION) as usize;
    let mut mask = vec![false; w * h];
    for y in top..h {
        for x in 0..w {
            let (r, g, b) = frame.pixel(x, y);
            mask[y * w + x] = is_match(r as i16, g as i16, b as i16);
        }
    }
    components(&mask, w, h)
        .into_iter()
        .filter(|b| {
            (WIDTH_RANGE.0..=WIDTH_RANGE.1).contains(&b.width())
                && (HEIGHT_RANGE.0..=HEIGHT_RANGE.1).contains(&b.height())
        })
        .max_by_key(|b| b.width() * b.height())
        .map(|b| ActionButton {
            x: b.x0,
            y: b.y0,
            width: b.width(),
            height: b.height(),
            label_width: 0,
        })
}

/// Finds the action row, if a live (coloured) turn is showing.
pub fn read_coinpoker_action_panel(frame: &Frame) -> Option<CoinPokerActionPanel> {
    let fold = largest_colour_region(frame, |r, g, b| {
        r > 150 && r - g > 100 && r - b > 80 && g < 100
    });
    let passive = largest_colour_region(frame, |r, g, b| g > 120 && g > b && r < 100);
    let aggressive = largest_colour_region(frame, |r, g, b| {
        r > 180 && (80..180).contains(&g) && b < 100 && r - g > 60 && g - b > 40
    });

    let panel = CoinPokerActionPanel {
        fold,
        passive,
        aggressive,
    };
    panel.is_live().then_some(panel)
}

/// Finds the dealer button: a small red hexagonal badge with a white `D`.
///
/// # Why card regions have to be excluded
///
/// The button's badge combines the same two colours a card face does — dark
/// red and white — because it is a red hexagon with a white letter on it,
/// the same way a card is a white rectangle with red or black ink on it.
/// Colour and size alone are not enough to tell them apart: the "6" printed
/// in the corner of a hole card, cropped to its own connected red-and-white
/// blob, comes out close enough in both dimensions to pass the same filter
/// the button does. What actually separates them is *where* they are — the
/// button never sits on top of a card — so the caller passes in where the
/// cards were already found, and anything overlapping one of those boxes is
/// disqualified before it is ever scored.
///
/// # Why red-or-white, not red-with-a-hole
///
/// ClubGG's gold dealer button is found as a single connected region with
/// the dark `D` counted as an enclosed hole, because gold pixels still form
/// one unbroken ring around it. This button's white `D` touches its edge in
/// several places, which severs the surrounding red into a handful of
/// separate arcs rather than one ring — measured directly against
/// `captures-coinpoker/h181631-0-1280x960.png`, a red-only mask breaks the
/// button into more than a dozen fragments. Masking red *and* white
/// together and looking for one solid blob sidesteps the fragmentation
/// entirely, at the cost of needing the card-overlap exclusion above.
pub fn find_dealer_button(frame: &Frame, exclude: &[(usize, usize, usize, usize)]) -> Option<(usize, usize)> {
    /// Measured at a 1280x960 table: roughly 34 wide by 39 tall. Generous on
    /// both sides for the anti-aliased edge of the hexagon.
    const SIZE: (usize, usize) = (20, 50);
    const MIN_FILL: f32 = 0.35;
    /// The chrome bar along the top and the row of icons along the bottom
    /// carry their own red-and-white artwork (a home icon, a chat bubble,
    /// an emoji) that clears every other filter here just as easily as the
    /// button does. The button itself only ever sits on the felt beside a
    /// player's cards, comfortably inside this band at a 1280x960 table, so
    /// restricting the scan to it is simpler than trying to describe what
    /// makes a hexagon different from an icon.
    const FELT_Y_RANGE: (usize, usize) = (150, 820);

    let is_button_red = |r: i16, g: i16, b: i16| r > 100 && r < 190 && g < 50 && b < 50 && r - g > 60;
    let is_white = |r: i16, g: i16, b: i16| r > 190 && g > 190 && b > 190;

    let (w, h) = (frame.width, frame.height);
    let mut mask = vec![false; w * h];
    for y in FELT_Y_RANGE.0..FELT_Y_RANGE.1.min(h) {
        for x in 0..w {
            let (r, g, b) = frame.pixel(x, y);
            let (r, g, b) = (r as i16, g as i16, b as i16);
            mask[y * w + x] = is_button_red(r, g, b) || is_white(r, g, b);
        }
    }

    // Mostly inside a card, not merely touching its edge. The button sits
    // beside the hole cards closely enough that its own bounding box clips
    // a few pixels of the neighbouring card's - measured on
    // captures-coinpoker/h181631-0-1280x960.png, about 5% of the button's
    // area. A false positive found *inside* a card's corner index, by
    // contrast, is entirely contained by it. Excluding on any intersection
    // at all would throw out the real button along with the false one; this
    // only excludes a blob that is substantially a part of the card.
    const MOSTLY_INSIDE_A_CARD: f32 = 0.5;
    let overlaps_a_card = |b: &Bounds| {
        exclude.iter().any(|&(ex, ey, ew, eh)| {
            let ix0 = b.x0.max(ex);
            let iy0 = b.y0.max(ey);
            let ix1 = b.x1.min(ex + ew - 1);
            let iy1 = b.y1.min(ey + eh - 1);
            if ix0 > ix1 || iy0 > iy1 {
                return false;
            }
            let overlap = (ix1 - ix0 + 1) * (iy1 - iy0 + 1);
            overlap as f32 / (b.width() * b.height()) as f32 > MOSTLY_INSIDE_A_CARD
        })
    };

    let candidates: Vec<Bounds> = components(&mask, w, h)
        .into_iter()
        .filter(|b| {
            (SIZE.0..=SIZE.1).contains(&b.width())
                && (SIZE.0..=SIZE.1).contains(&b.height())
                && !overlaps_a_card(b)
        })
        .filter(|b| {
            let lit = (b.y0..=b.y1)
                .map(|y| (b.x0..=b.x1).filter(|&x| mask[y * w + x]).count())
                .sum::<usize>();
            lit as f32 / (b.width() * b.height()) as f32 >= MIN_FILL
        })
        .collect();

    // More than one candidate means an ambiguous frame - mid animation, or a
    // second card-shaped false positive that also happened to clear every
    // filter. Refusing is cheaper than guessing which one is real.
    match candidates.len() {
        1 => {
            let b = &candidates[0];
            Some((b.x0 + b.width() / 2, b.y0 + b.height() / 2))
        }
        _ => None,
    }
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

    /// Paints a synthetic three-button row using the exact colours measured
    /// at the real table, so the colour thresholds are checked independently
    /// of any one capture.
    fn paint_action_row(w: usize, h: usize, fold: bool, passive: bool, aggressive: bool) -> Vec<u8> {
        let mut rgb = vec![20u8; w * h * 3]; // dark background
        let mut paint = |x0: usize, colour: (u8, u8, u8)| {
            for y in 877..953 {
                for x in x0..x0 + 161 {
                    let i = (y * w + x) * 3;
                    rgb[i] = colour.0;
                    rgb[i + 1] = colour.1;
                    rgb[i + 2] = colour.2;
                }
            }
        };
        if fold { paint(760, (207, 50, 65)); }
        if passive { paint(935, (33, 159, 132)); }
        if aggressive { paint(1109, (238, 136, 57)); }
        rgb
    }

    #[test]
    fn a_full_turn_finds_all_three_buttons_by_colour() {
        let (w, h) = (1280, 960);
        let rgb = paint_action_row(w, h, true, true, true);
        let frame = Frame::new(w, h, &rgb);
        let panel = read_coinpoker_action_panel(&frame).expect("a live turn");
        assert!(panel.is_live());
        assert!(panel.fold.is_some());
        assert!(panel.passive.is_some());
        assert!(panel.aggressive.is_some());
        // Fold sits left of passive, which sits left of aggressive.
        assert!(panel.fold.unwrap().x < panel.passive.unwrap().x);
        assert!(panel.passive.unwrap().x < panel.aggressive.unwrap().x);
    }

    #[test]
    fn a_turn_with_nothing_left_to_raise_has_no_aggressive_button() {
        let (w, h) = (1280, 960);
        let rgb = paint_action_row(w, h, true, true, false);
        let frame = Frame::new(w, h, &rgb);
        let panel = read_coinpoker_action_panel(&frame).expect("still a live turn");
        assert!(panel.is_live());
        assert!(panel.aggressive.is_none());
    }

    #[test]
    fn no_coloured_buttons_is_not_a_live_turn() {
        let (w, h) = (1280, 960);
        let rgb = paint_action_row(w, h, false, false, false);
        let frame = Frame::new(w, h, &rgb);
        assert_eq!(read_coinpoker_action_panel(&frame), None);
    }

    #[test]
    fn only_a_fold_button_is_not_a_live_turn() {
        // Fold alone (no passive option) should never happen at a real
        // table, but it must not be mistaken for a real decision either.
        let (w, h) = (1280, 960);
        let rgb = paint_action_row(w, h, true, false, false);
        let frame = Frame::new(w, h, &rgb);
        assert_eq!(read_coinpoker_action_panel(&frame), None);
    }

    /// Checked against the real capture the geometry was measured from:
    /// the button sits beside the hero's hole cards.
    #[test]
    fn finds_the_dealer_button_on_a_real_capture() {
        let path = data("frames/coinpoker-h181631.rgb");
        if !path.exists() {
            return;
        }
        let raw = std::fs::read(path).expect("fixture frame");
        let width = u32::from_le_bytes(raw[0..4].try_into().unwrap()) as usize;
        let height = u32::from_le_bytes(raw[4..8].try_into().unwrap()) as usize;
        let frame = Frame::new(width, height, &raw[8..]);

        // Without excluding the cards, the hole card's own red-and-white
        // corner index is a second candidate, and the ambiguity refuses.
        assert_eq!(
            find_dealer_button(&frame, &[]),
            None,
            "the hole card's corner index should be mistaken for a second button"
        );

        let exclude: Vec<(usize, usize, usize, usize)> = detect_card_positions(&frame)
            .into_iter()
            .map(|d| (d.x, d.y, geometry::CARD_W, geometry::CARD_H))
            .collect();
        let button = find_dealer_button(&frame, &exclude).expect("the button should be found");
        // Measured directly against this capture: the button sits at about
        // (541, 691), just above and left of the hero's hole cards.
        assert!(
            button.0.abs_diff(541) <= 5 && button.1.abs_diff(691) <= 5,
            "button at {button:?}, expected near (541, 691)"
        );
    }

    /// Checked against the real capture the geometry was measured from.
    #[test]
    fn reads_the_action_row_from_a_real_capture() {
        let path = data("frames/coinpoker-h181631.rgb");
        if !path.exists() {
            return;
        }
        let raw = std::fs::read(path).expect("fixture frame");
        let width = u32::from_le_bytes(raw[0..4].try_into().unwrap()) as usize;
        let height = u32::from_le_bytes(raw[4..8].try_into().unwrap()) as usize;
        let frame = Frame::new(width, height, &raw[8..]);

        let panel = read_coinpoker_action_panel(&frame).expect("this frame is a live turn");
        assert!(panel.is_live());
        let fold = panel.fold.expect("fold button");
        let passive = panel.passive.expect("passive button");
        let aggressive = panel.aggressive.expect("this hand has room to raise");
        assert!(fold.x < passive.x && passive.x < aggressive.x);
        // Measured directly from this capture at the time the colour
        // thresholds were derived.
        assert_eq!((fold.x, fold.y), (760, 877));
        assert_eq!((passive.x, passive.y), (935, 877));
        assert_eq!((aggressive.x, aggressive.y), (1109, 877));
    }
}
