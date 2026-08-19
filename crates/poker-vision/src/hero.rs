//! Reading the two cards the client deals the hero.
//!
//! # Why these are not just more cards
//!
//! Board cards are found by sweeping the frame for card-shaped bright
//! rectangles. That does not work here for two reasons. The hero's pair is
//! drawn overlapping, so the two faces merge into a single blob nearly twice
//! the width of a card, and the top of that blob is the avatar disc behind
//! them rather than a card corner — there is no rectangle to measure glyph
//! offsets from. And the cards are drawn about a sixth larger than the board's,
//! so the board templates are the wrong shape: matching against them scored a
//! red pip at 37 to 43 where a sound match scores under 20, and on one frame an
//! eight scored nearer the six template than its own.
//!
//! So the hero's cards get their own anchor and their own templates. The anchor
//! is the corner suit pip: one solid shape, printed on a card face, with its
//! rank directly above it. The templates are harvested at the hero's card size
//! and, as everywhere else here, never resized.
//!
//! # A known gap
//!
//! The captures the templates were built from cover nine of the thirteen ranks
//! — every one except three, four, seven and king — because they span only
//! about a dozen hands. A card the templates do not cover is refused, and
//! [`crate::TableView`] withholds the whole hand rather than half of it, so the
//! gap costs readings and never causes wrong ones. Closing it needs captures
//! over more hands, not different code.

use crate::{components, invalid, read_label, read_u32, Bounds, CardRead, Frame};
use poker_core::card::{Card, Rank, Suit};
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

const MAGIC: &[u8; 4] = b"PKHC";
const VERSION: u32 = 1;

/// Ink is anything dark enough to be printing.
///
/// The threshold has to clear a red pip, which measures 71, while staying under
/// the grey outline drawn around a card, which is above 120. At 140 that
/// outline joins the pip to the dark plate behind the cards and the pip stops
/// being a shape of its own.
const INK: f32 = 110.0;

/// A shape at the size the client draws it, as a binary mask.
#[derive(Debug, Clone)]
struct Shape<T> {
    label: T,
    /// Only meaningful for suits: whether the pip is printed in red.
    red: bool,
    height: usize,
    width: usize,
    mask: Vec<u8>,
}

/// Rank and suit shapes at the hero's card size.
#[derive(Debug, Clone)]
pub struct HeroTemplates {
    ranks: Vec<Shape<Rank>>,
    suits: Vec<Shape<Suit>>,
}

impl HeroTemplates {
    /// Reads a template file produced by the extraction tooling.
    pub fn load(path: impl AsRef<Path>) -> io::Result<HeroTemplates> {
        let mut file = BufReader::new(File::open(path)?);

        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(invalid("not a hero card template file"));
        }
        if read_u32(&mut file)? != VERSION {
            return Err(invalid("unsupported hero card template version"));
        }

        let mut ranks = Vec::new();
        for _ in 0..read_u32(&mut file)? {
            let label = read_label(&mut file)?;
            let rank = crate::parse_rank(&label)
                .ok_or_else(|| invalid(format!("bad rank {label:?}")))?;
            let (height, width) = (read_u32(&mut file)? as usize, read_u32(&mut file)? as usize);
            let mut mask = vec![0u8; height * width];
            file.read_exact(&mut mask)?;
            ranks.push(Shape {
                label: rank,
                red: false,
                height,
                width,
                mask,
            });
        }

        let mut suits = Vec::new();
        for _ in 0..read_u32(&mut file)? {
            let label = read_label(&mut file)?;
            let suit = label
                .chars()
                .next()
                .and_then(Suit::from_char)
                .ok_or_else(|| invalid(format!("bad suit {label:?}")))?;
            let mut red = [0u8; 1];
            file.read_exact(&mut red)?;
            let (height, width) = (read_u32(&mut file)? as usize, read_u32(&mut file)? as usize);
            let mut mask = vec![0u8; height * width];
            file.read_exact(&mut mask)?;
            suits.push(Shape {
                label: suit,
                red: red[0] != 0,
                height,
                width,
                mask,
            });
        }

        if ranks.is_empty() || suits.is_empty() {
            return Err(invalid("hero card templates are incomplete"));
        }
        Ok(HeroTemplates { ranks, suits })
    }

    /// The ranks these templates can read, in ascending order.
    ///
    /// Deliberately public: the set is short of a full deck, and a caller that
    /// wants to know why a hand went unread should be able to ask.
    pub fn ranks(&self) -> Vec<Rank> {
        let mut seen: Vec<Rank> = self.ranks.iter().map(|r| r.label).collect();
        seen.sort_unstable();
        seen.dedup();
        seen
    }

    /// The suits these templates can read.
    pub fn suits(&self) -> Vec<Suit> {
        let mut seen: Vec<Suit> = self.suits.iter().map(|s| s.label).collect();
        seen.sort_unstable();
        seen.dedup();
        seen
    }
}

/// How confidently a hero card is accepted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeroThresholds {
    /// Reject beyond this mean absolute difference against the template, on a
    /// scale where a perfect match is 0 and a wholly different shape is 255.
    pub max_distance: f32,
    /// Require the best match to beat the nearest differently-labelled template
    /// by at least this much.
    pub min_margin: f32,
}

impl Default for HeroThresholds {
    fn default() -> HeroThresholds {
        HeroThresholds {
            max_distance: 45.0,
            min_margin: 15.0,
        }
    }
}

/// Reads the hero's two cards.
///
/// `near` is the hero's seat in frame coordinates; the search is confined to
/// the area just above it. Cards that do not clear the thresholds are simply
/// absent from the result — a countdown badge is pip-shaped and pip-sized, so
/// candidates that turn out not to be cards are expected rather than alarming.
pub fn read_hole_cards(
    frame: &Frame,
    templates: &HeroTemplates,
    thresholds: HeroThresholds,
    near: (usize, usize),
) -> Vec<CardRead> {
    let (w, h) = (frame.width, frame.height);
    let mask = ink_mask(frame, near);

    let mut reads: Vec<CardRead> = detect_pips(frame, &mask, w, h)
        .into_iter()
        .filter_map(|pip| {
            let rank_box = rank_above(frame, &mask, w, pip)?;
            let (rank, rank_distance, rank_margin) =
                best(&mask, w, h, rank_box, &templates.ranks, None)?;

            let red = ink_is_red(frame, &mask, w, pip);
            let (suit, suit_distance, suit_margin) =
                best(&mask, w, h, pip, &templates.suits, Some(red))?;

            let distance = rank_distance.max(suit_distance);
            let margin = rank_margin.min(suit_margin);
            (distance <= thresholds.max_distance && margin >= thresholds.min_margin).then_some(
                CardRead {
                    x: pip.x0,
                    y: pip.y0,
                    card: Some(Card::new(rank, suit)),
                    distance,
                    margin,
                },
            )
        })
        .collect();

    reads.sort_by_key(|r| r.x);
    reads.dedup_by_key(|r| r.card);
    reads.truncate(2);
    reads
}

/// Ink within reach of the hero's seat.
fn ink_mask(frame: &Frame, near: (usize, usize)) -> Vec<bool> {
    /// How far around the hero's seat the cards can be.
    const REACH_X: usize = 150;
    const ABOVE: usize = 240;
    const CLEAR: usize = 70;

    let (w, h) = (frame.width, frame.height);
    let x0 = near.0.saturating_sub(REACH_X);
    let x1 = (near.0 + REACH_X).min(w);
    let y0 = near.1.saturating_sub(ABOVE);
    let y1 = near.1.saturating_sub(CLEAR).min(h);

    let mut mask = vec![false; w * h];
    if x0 >= x1 || y0 >= y1 {
        return mask;
    }
    for y in y0..y1 {
        for x in x0..x1 {
            let (r, g, b) = frame.pixel(x, y);
            let luma = 0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32;
            mask[y * w + x] = luma < INK;
        }
    }
    mask
}

/// Corner suit pips: roughly square shapes of about this size, on a card face.
///
/// The large decorative pip in the middle of a card is wider than this and the
/// rank digits are narrower, so size alone separates all three.
fn detect_pips(frame: &Frame, mask: &[bool], w: usize, h: usize) -> Vec<Bounds> {
    const PIP: (usize, usize) = (22, 40);
    components(mask, w, h)
        .into_iter()
        .filter(|b| {
            (PIP.0..=PIP.1).contains(&b.width())
                && (PIP.0..=PIP.1).contains(&b.height())
                && on_card_face(frame, *b)
        })
        .collect()
}

/// The rank glyph directly above a pip, as a tight box around its ink.
///
/// A ten is drawn as two separate glyphs, so everything qualifying in the zone
/// is taken together as one rank.
fn rank_above(frame: &Frame, mask: &[bool], w: usize, pip: Bounds) -> Option<Bounds> {
    /// Rows above the pip that the rank occupies, and how far left or right of
    /// the pip it can sit — the two cards are drawn at different tilts, so this
    /// is wider than an untilted card would need.
    const ZONE_TOP: usize = 45;
    const ZONE_BOTTOM: usize = 3;
    const ZONE_LEFT: usize = 20;
    const ZONE_RIGHT: usize = 35;
    /// A rank glyph stands this tall and is no wider than a ten.
    const GLYPH_H: (usize, usize) = (25, 38);
    const GLYPH_W: (usize, usize) = (4, 30);

    let y0 = pip.y0.checked_sub(ZONE_TOP)?;
    let y1 = pip.y0.checked_sub(ZONE_BOTTOM)?;
    let x0 = pip.x0.saturating_sub(ZONE_LEFT);
    let x1 = (pip.x0 + ZONE_RIGHT).min(w - 1);

    let mut zone = vec![false; mask.len()];
    for y in y0..=y1 {
        for x in x0..=x1 {
            zone[y * w + x] = mask[y * w + x];
        }
    }

    let parts: Vec<Bounds> = components(&zone, w, frame.height)
        .into_iter()
        .filter(|b| {
            (GLYPH_H.0..=GLYPH_H.1).contains(&b.height())
                && (GLYPH_W.0..=GLYPH_W.1).contains(&b.width())
                && on_card_face(frame, *b)
        })
        .collect();
    if parts.is_empty() {
        return None;
    }

    let bounds = Bounds {
        x0: parts.iter().map(|b| b.x0).min()?,
        y0: parts.iter().map(|b| b.y0).min()?,
        x1: parts.iter().map(|b| b.x1).max()?,
        y1: parts.iter().map(|b| b.y1).max()?,
    };
    // Ink spanning the whole zone ran off the card and is not a glyph.
    (bounds.height() <= 38 && bounds.width() <= 34).then_some(bounds)
}

/// Best template match for a shape, among templates within a pixel of its size.
///
/// The comparison window is sampled afresh from the mask at each alignment
/// rather than the shape being stretched to fit, so nothing is ever resized.
fn best<T: Copy + PartialEq>(
    mask: &[bool],
    stride: usize,
    frame_height: usize,
    bounds: Bounds,
    templates: &[Shape<T>],
    red: Option<bool>,
) -> Option<(T, f32, f32)> {
    let (w, h) = (bounds.width(), bounds.height());
    let mut best: Option<(T, f32)> = None;
    let mut rival = f32::INFINITY;

    for shape in templates {
        if red.is_some_and(|red| shape.red != red)
            || shape.width.abs_diff(w) > 1
            || shape.height.abs_diff(h) > 1
        {
            continue;
        }
        let mut closest = f32::INFINITY;
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let (sx, sy) = (bounds.x0 as i32 + dx, bounds.y0 as i32 + dy);
                if sx < 0 || sy < 0 {
                    continue;
                }
                let (sx, sy) = (sx as usize, sy as usize);
                if sx + shape.width > stride || sy + shape.height > frame_height {
                    continue;
                }
                let mut total = 0u32;
                for y in 0..shape.height {
                    for x in 0..shape.width {
                        let here = if mask[(sy + y) * stride + sx + x] { 255u8 } else { 0 };
                        total += here.abs_diff(shape.mask[y * shape.width + x]) as u32;
                    }
                }
                closest = closest.min(total as f32 / (shape.width * shape.height) as f32);
            }
        }
        match best {
            Some((label, score)) if closest < score => {
                if label != shape.label {
                    rival = rival.min(score);
                }
                best = Some((shape.label, closest));
            }
            Some((label, _)) if label != shape.label => rival = rival.min(closest),
            Some(_) => {}
            None => best = Some((shape.label, closest)),
        }
    }
    best.map(|(label, score)| (label, score, rival - score))
}

/// Whether a shape's own ink is red, ignoring whatever is drawn behind it.
///
/// The client draws a decorative flame over the corner of the hero's cards, and
/// a box wide enough to hold a suit template catches enough of it to make a
/// black spade read as red. Only the shape's own pixels are counted.
fn ink_is_red(frame: &Frame, mask: &[bool], stride: usize, bounds: Bounds) -> bool {
    let mut sum = 0i64;
    let mut count = 0i64;
    for y in bounds.y0..=bounds.y1 {
        for x in bounds.x0..=bounds.x1 {
            if mask[y * stride + x] {
                let (r, _, b) = frame.pixel(x, y);
                sum += r as i64 - b as i64;
                count += 1;
            }
        }
    }
    count > 0 && sum as f32 / count as f32 > 20.0
}

/// Whether a shape is printed on a card rather than on the felt or a badge.
///
/// The countdown badge is pip-sized and pip-shaped, so shape alone does not
/// settle it. The background does: a card face is bright and very nearly
/// colourless, while the badge sits on orange.
fn on_card_face(frame: &Frame, bounds: Bounds) -> bool {
    const PAD: usize = 8;
    let x0 = bounds.x0.saturating_sub(PAD);
    let y0 = bounds.y0.saturating_sub(PAD);
    let x1 = (bounds.x1 + PAD).min(frame.width - 1);
    let y1 = (bounds.y1 + PAD).min(frame.height - 1);

    let mut face = 0usize;
    let mut total = 0usize;
    for y in y0..=y1 {
        for x in x0..=x1 {
            let (r, g, b) = frame.pixel(x, y);
            let lo = r.min(g).min(b);
            let hi = r.max(g).max(b);
            total += 1;
            if lo > 110 && hi - lo < 60 {
                face += 1;
            }
        }
    }
    total > 0 && face * 100 / total > 40
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

    fn templates() -> HeroTemplates {
        HeroTemplates::load(data("hero_cards.bin")).expect("hero templates should load")
    }

    #[test]
    fn all_four_suits_are_covered() {
        assert_eq!(templates().suits().len(), 4);
    }

    /// Records exactly which ranks the captures happened to contain.
    ///
    /// This is a gap in the data, not in the code: the frames the templates
    /// came from span about a dozen hands, so four ranks never reached the
    /// hero's seat. It is asserted so that closing it is visible as a change
    /// here rather than passing unnoticed.
    #[test]
    fn the_ranks_covered_are_the_ones_the_captures_contained() {
        let covered: Vec<String> = templates().ranks().iter().map(|r| r.to_string()).collect();
        assert_eq!(covered, vec!["2", "5", "6", "8", "9", "T", "J", "Q", "A"]);
    }

    #[test]
    fn a_file_that_is_not_a_hero_template_set_is_rejected() {
        let path = std::env::temp_dir().join("poker_vision_bad_hero.bin");
        std::fs::write(&path, b"nonsense").expect("write");
        assert!(HeroTemplates::load(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }
}
