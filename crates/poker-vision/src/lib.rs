//! Reading a poker table from raw screen pixels.
//!
//! The client renders deterministically — measured across 190 captured frames,
//! static chrome is 98.8% pixel-identical and a card face repeats to within a
//! few grey levels. So this is exact template matching, not a machine-learning
//! problem, and it runs in microseconds rather than milliseconds.
//!
//! # Refusing beats guessing
//!
//! The dangerous failure in a screen-reading bot is not a bad decision, it is a
//! **misread board**: the bot reasons perfectly about a hand that does not
//! exist, acts with total confidence, and looks healthy from the outside.
//!
//! So a read is accepted only when it matches a template closely *and* beats
//! the runner-up by a clear margin. Both thresholds come from measurement, not
//! taste: correct reads score 4–7, while a card partly covered by the pot
//! display scored 67 with a margin of 0.7. [`CardRead::card`] is `None` when
//! either test fails, and the caller must handle that rather than receive a
//! guess.
//!
//! # Frames are raw pixels
//!
//! Nothing here decodes PNG. Live capture hands over a raw buffer, and the
//! stored `.rgb` fixtures used in tests are that same buffer written to disk.

#![forbid(unsafe_code)]

mod action;
mod hero;
mod text;
mod view;

pub use action::{read_action_panel, ActionButton, ActionPanel};
pub use hero::{read_hole_cards, HeroTemplates, HeroThresholds};
pub use text::{read_numbers, GlyphTemplates, Ink, NumberRead, TextThresholds};
pub use view::{SeatView, TableView};

use poker_core::card::{Card, Rank, Suit};
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

/// Identifies a serialized template file.
const MAGIC: &[u8; 4] = b"PKVT";
const VERSION: u32 = 1;

/// Card geometry, measured from captures at a 1430x1040 table window.
///
/// The window is resizable and these are pixel-exact, so the bot must set a
/// known window size before reading. A different size needs different
/// templates, not scaled ones — the layout reflows rather than scaling.
pub mod geometry {
    /// Width of a card face, in pixels.
    pub const CARD_W: usize = 103;
    /// Height of a card face.
    pub const CARD_H: usize = 156;
    /// Accepted width range when detecting a card.
    pub const CARD_W_RANGE: (usize, usize) = (85, 115);
    /// Accepted height range. Wider than the card because a dimmed card at
    /// showdown is drawn slightly shorter.
    pub const CARD_H_RANGE: (usize, usize) = (130, 175);
    /// Horizontal inset from the detected edge, past the card's rounded border.
    pub const INSET_X: usize = 6;
    /// Rank glyph, relative to the card's top-left.
    ///
    /// Deliberately starts 16 rows down: the pot readout is drawn *over* the
    /// top of the middle board cards, and the rows above this are the ones it
    /// corrupts.
    pub const RANK_TOP: usize = 16;
    pub const RANK_W: usize = 50;
    pub const RANK_H: usize = 34;
    /// Suit pip, below the rank.
    pub const SUIT_TOP: usize = 52;
    pub const SUIT_W: usize = 44;
    pub const SUIT_H: usize = 34;
}

/// A single-channel image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gray {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u8>,
}

impl Gray {
    pub fn new(width: usize, height: usize, pixels: Vec<u8>) -> Gray {
        assert_eq!(pixels.len(), width * height, "pixel count must match size");
        Gray {
            width,
            height,
            pixels,
        }
    }

    #[inline]
    pub fn at(&self, x: usize, y: usize) -> u8 {
        self.pixels[y * self.width + x]
    }

    /// Stretches contrast so the darkest pixel becomes black.
    ///
    /// This is what lets a dimmed card — the client greys out holdings that
    /// lost at showdown — match the same template as a bright one.
    pub fn normalised(&self) -> Gray {
        let lo = *self.pixels.iter().min().unwrap_or(&0);
        let hi = *self.pixels.iter().max().unwrap_or(&255);
        if hi.saturating_sub(lo) < 30 {
            // Nearly flat: no glyph here, and stretching would amplify noise
            // into something that matches a template by accident.
            return Gray::new(self.width, self.height, vec![255; self.pixels.len()]);
        }
        let scale = 255.0 / (hi - lo) as f32;
        Gray::new(
            self.width,
            self.height,
            self.pixels
                .iter()
                .map(|p| (((p - lo) as f32) * scale).clamp(0.0, 255.0) as u8)
                .collect(),
        )
    }

    /// Mean absolute difference against another image of the same size.
    fn distance(&self, other: &Gray) -> f32 {
        debug_assert_eq!(self.pixels.len(), other.pixels.len());
        let total: u32 = self
            .pixels
            .iter()
            .zip(&other.pixels)
            .map(|(a, b)| a.abs_diff(*b) as u32)
            .sum();
        total as f32 / self.pixels.len() as f32
    }
}

/// A frame of raw pixels, as screen capture delivers them.
#[derive(Debug, Clone, Copy)]
pub struct Frame<'a> {
    pub width: usize,
    pub height: usize,
    /// Row-major RGB, three bytes per pixel.
    pub rgb: &'a [u8],
}

impl<'a> Frame<'a> {
    /// Wraps a buffer.
    ///
    /// # Panics
    /// Panics if the buffer is not exactly `width * height * 3` bytes.
    pub fn new(width: usize, height: usize, rgb: &'a [u8]) -> Frame<'a> {
        assert_eq!(rgb.len(), width * height * 3, "buffer size must match");
        Frame { width, height, rgb }
    }

    #[inline]
    pub(crate) fn pixel(&self, x: usize, y: usize) -> (u8, u8, u8) {
        let i = (y * self.width + x) * 3;
        (self.rgb[i], self.rgb[i + 1], self.rgb[i + 2])
    }

    /// Extracts a greyscale rectangle, using the same luma weights as the
    /// tooling that produced the templates.
    fn grey_rect(&self, x0: usize, y0: usize, w: usize, h: usize) -> Option<Gray> {
        if x0 + w > self.width || y0 + h > self.height {
            return None;
        }
        let mut pixels = Vec::with_capacity(w * h);
        for y in y0..y0 + h {
            for x in x0..x0 + w {
                let (r, g, b) = self.pixel(x, y);
                let luma = 0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32;
                pixels.push(luma.round().clamp(0.0, 255.0) as u8);
            }
        }
        Some(Gray::new(w, h, pixels))
    }

    /// Whether a rectangle reads as red rather than black ink.
    ///
    /// Measured on the captures, the red and blue channels are identical on
    /// black pips and around 53 apart on red ones — a clean split with nothing
    /// in between, so this halves the shape search with no risk.
    fn is_red(&self, x0: usize, y0: usize, w: usize, h: usize) -> bool {
        let mut sum = 0i64;
        for y in y0..(y0 + h).min(self.height) {
            for x in x0..(x0 + w).min(self.width) {
                let (r, _, b) = self.pixel(x, y);
                sum += r as i64 - b as i64;
            }
        }
        sum as f32 / (w * h) as f32 > 20.0
    }
}

/// The rank and suit glyphs the recogniser matches against.
#[derive(Debug, Clone)]
pub struct Templates {
    ranks: Vec<(Rank, Gray)>,
    suits: Vec<(Suit, Gray)>,
}

impl Templates {
    /// Reads a template file produced by the extraction tooling.
    pub fn load(path: impl AsRef<Path>) -> io::Result<Templates> {
        let mut file = BufReader::new(File::open(path)?);

        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(invalid("not a card template file"));
        }
        if read_u32(&mut file)? != VERSION {
            return Err(invalid("unsupported template version"));
        }

        let mut ranks = Vec::new();
        let (count, h, w) = (
            read_u32(&mut file)? as usize,
            read_u32(&mut file)? as usize,
            read_u32(&mut file)? as usize,
        );
        for _ in 0..count {
            let label = read_label(&mut file)?;
            let rank = parse_rank(&label).ok_or_else(|| invalid(format!("bad rank {label:?}")))?;
            ranks.push((rank, Gray::new(w, h, read_pixels(&mut file, w * h)?)));
        }

        let mut suits = Vec::new();
        let (count, h, w) = (
            read_u32(&mut file)? as usize,
            read_u32(&mut file)? as usize,
            read_u32(&mut file)? as usize,
        );
        for _ in 0..count {
            let label = read_label(&mut file)?;
            let suit = label
                .chars()
                .next()
                .and_then(Suit::from_char)
                .ok_or_else(|| invalid(format!("bad suit {label:?}")))?;
            suits.push((suit, Gray::new(w, h, read_pixels(&mut file, w * h)?)));
        }

        if ranks.len() != 13 || suits.len() != 4 {
            return Err(invalid(format!(
                "expected 13 ranks and 4 suits, found {} and {}",
                ranks.len(),
                suits.len()
            )));
        }
        Ok(Templates { ranks, suits })
    }

    pub fn rank_count(&self) -> usize {
        self.ranks.len()
    }

    pub fn suit_count(&self) -> usize {
        self.suits.len()
    }
}

/// How confidently a read is accepted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Thresholds {
    /// Reject beyond this mean absolute difference. A partly covered glyph
    /// scores far above it.
    pub max_distance: f32,
    /// Require the best match to beat the runner-up by at least this much.
    pub min_margin: f32,
    /// How far to search for the card being detected a pixel or two off.
    pub search: i32,
}

impl Default for Thresholds {
    fn default() -> Thresholds {
        Thresholds {
            max_distance: 25.0,
            min_margin: 4.0,
            search: 2,
        }
    }
}

/// One card position and what was read there.
#[derive(Debug, Clone, PartialEq)]
pub struct CardRead {
    /// Top-left of the detected card face, in frame coordinates.
    pub x: usize,
    pub y: usize,
    /// The card, or `None` when the read did not clear the thresholds.
    pub card: Option<Card>,
    /// Worst match distance across rank and suit.
    pub distance: f32,
    /// Smallest margin over the runner-up.
    pub margin: f32,
}

impl CardRead {
    /// Whether this position produced a usable card.
    pub fn is_confident(&self) -> bool {
        self.card.is_some()
    }
}

/// Finds every card face in a frame and reads it.
///
/// Results are ordered top to bottom, then left to right, so board cards come
/// out in board order.
pub fn read_cards(frame: &Frame, templates: &Templates, thresholds: Thresholds) -> Vec<CardRead> {
    let mut reads: Vec<CardRead> = detect_cards(frame)
        .into_iter()
        .filter_map(|(x, y)| read_at(frame, templates, thresholds, x, y))
        .collect();
    // Rows first, since a board sits at one height and hole cards at another.
    reads.sort_by_key(|r| (r.y / 80, r.x));
    reads
}

/// Reads the card at a known top-left position.
fn read_at(
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
        templates.ranks.iter().map(|(r, g)| (*r, g)),
        thresholds.search,
    )?;

    // Colour narrows the suit to two candidates before any shape work.
    let red = frame.is_red(gx, y + SUIT_TOP, SUIT_W, SUIT_H);
    let candidates = templates
        .suits
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

/// Best template match, searching a small window of offsets.
///
/// The window shifts where the *source* is sampled rather than shifting the
/// sampled pixels, so no wrapped-around edge pixels ever enter the comparison.
fn best_match<'a, T: Copy>(
    frame: &Frame,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    templates: impl Iterator<Item = (T, &'a Gray)>,
    search: i32,
) -> Option<(T, f32, f32)> {
    let mut best: Option<(T, f32)> = None;
    let mut runner_up = f32::INFINITY;

    for (label, template) in templates {
        let mut closest = f32::INFINITY;
        for dy in -search..=search {
            for dx in -search..=search {
                let sx = x as i32 + dx;
                let sy = y as i32 + dy;
                if sx < 0 || sy < 0 {
                    continue;
                }
                if let Some(patch) = frame.grey_rect(sx as usize, sy as usize, w, h) {
                    closest = closest.min(patch.normalised().distance(template));
                }
            }
        }
        match best {
            Some((_, score)) if closest >= score => runner_up = runner_up.min(closest),
            Some((_, score)) => {
                runner_up = score;
                best = Some((label, closest));
            }
            None => best = Some((label, closest)),
        }
    }

    best.map(|(label, score)| (label, score, runner_up - score))
}

/// A connected run of set pixels, as a bounding box.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Bounds {
    pub x0: usize,
    pub y0: usize,
    pub x1: usize,
    pub y1: usize,
}

impl Bounds {
    pub fn width(&self) -> usize {
        self.x1 - self.x0 + 1
    }
    pub fn height(&self) -> usize {
        self.y1 - self.y0 + 1
    }
}

/// Bounding boxes of every 4-connected group of set pixels.
///
/// Scanned top-to-bottom then left-to-right, so results come out in reading
/// order and callers can rely on it.
pub(crate) fn components(mask: &[bool], w: usize, h: usize) -> Vec<Bounds> {
    debug_assert_eq!(mask.len(), w * h);
    let mut seen = vec![false; w * h];
    let mut found = Vec::new();
    let mut stack: Vec<(usize, usize)> = Vec::new();

    for start_y in 0..h {
        for start_x in 0..w {
            if !mask[start_y * w + start_x] || seen[start_y * w + start_x] {
                continue;
            }
            let mut bounds = Bounds {
                x0: start_x,
                y0: start_y,
                x1: start_x,
                y1: start_y,
            };
            seen[start_y * w + start_x] = true;
            stack.push((start_x, start_y));

            while let Some((x, y)) = stack.pop() {
                bounds.x0 = bounds.x0.min(x);
                bounds.x1 = bounds.x1.max(x);
                bounds.y0 = bounds.y0.min(y);
                bounds.y1 = bounds.y1.max(y);
                let mut visit = |nx: usize, ny: usize, stack: &mut Vec<(usize, usize)>| {
                    let i = ny * w + nx;
                    if mask[i] && !seen[i] {
                        seen[i] = true;
                        stack.push((nx, ny));
                    }
                };
                if x + 1 < w {
                    visit(x + 1, y, &mut stack);
                }
                if x > 0 {
                    visit(x - 1, y, &mut stack);
                }
                if y + 1 < h {
                    visit(x, y + 1, &mut stack);
                }
                if y > 0 {
                    visit(x, y - 1, &mut stack);
                }
            }
            found.push(bounds);
        }
    }
    found
}

/// Finds card-sized bright rectangles.
///
/// Card faces are bright and near-neutral; felt is green and chrome is dark, so
/// one threshold separates them cleanly.
fn detect_cards(frame: &Frame) -> Vec<(usize, usize)> {
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

    components(&mask, w, h)
        .into_iter()
        .filter(|b| {
            (CARD_W_RANGE.0..=CARD_W_RANGE.1).contains(&b.width())
                && (CARD_H_RANGE.0..=CARD_H_RANGE.1).contains(&b.height())
        })
        .map(|b| (b.x0, b.y0))
        .collect()
}

/// Finds the dealer button.
///
/// It is a gold disc with a dark `D`, and the only thing on the table shaped
/// like one. Colour alone is not enough — chips, the straddle tag and the
/// countdown flames are all gold too — but those are open shapes while the
/// button is a solid disc, and how much of its own bounding box a shape fills
/// separates them completely: the button measures 0.50 to 0.80 across the
/// captures and nothing else reaches 0.36.
///
/// Returns `None` unless exactly one candidate is found. Two would mean the
/// frame was caught mid-animation as the button slides to the next seat, and
/// there is no safe way to pick between them.
pub fn detect_dealer_button(frame: &Frame) -> Option<(usize, usize)> {
    /// Measured at a 1430x1040 table.
    const SIZE: (usize, usize) = (26, 48);
    const MIN_FILL: f32 = 0.42;

    let (w, h) = (frame.width, frame.height);
    let mut gold = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            let (r, g, b) = frame.pixel(x, y);
            let (r, g, b) = (r as i16, g as i16, b as i16);
            gold[y * w + x] = r > 150 && g > 110 && b < 120 && r - b > 70 && g - b > 40;
        }
    }

    let mut found = None;
    for bounds in components(&gold, w, h) {
        if !(SIZE.0..=SIZE.1).contains(&bounds.width())
            || !(SIZE.0..=SIZE.1).contains(&bounds.height())
            || solidity(&gold, w, bounds) < MIN_FILL
        {
            continue;
        }
        if found.is_some() {
            return None;
        }
        found = Some((
            bounds.x0 + bounds.width() / 2,
            bounds.y0 + bounds.height() / 2,
        ));
    }
    found
}

/// What fraction of a shape's bounding box the shape encloses.
///
/// Holes count as enclosed: the `D` printed on the button is a hole, and so is
/// the gap left where the disc's shaded lower edge falls out of the colour mask.
/// Anything reachable from the edge of the box is outside the shape.
fn solidity(mask: &[bool], stride: usize, bounds: Bounds) -> f32 {
    let (w, h) = (bounds.width(), bounds.height());
    let mut outside = vec![false; w * h];
    let mut stack = Vec::new();

    for x in 0..w {
        for y in [0, h - 1] {
            if !mask[(bounds.y0 + y) * stride + bounds.x0 + x] && !outside[y * w + x] {
                outside[y * w + x] = true;
                stack.push((x, y));
            }
        }
    }
    for y in 0..h {
        for x in [0, w - 1] {
            if !mask[(bounds.y0 + y) * stride + bounds.x0 + x] && !outside[y * w + x] {
                outside[y * w + x] = true;
                stack.push((x, y));
            }
        }
    }

    while let Some((x, y)) = stack.pop() {
        let mut visit = |nx: usize, ny: usize, stack: &mut Vec<(usize, usize)>| {
            let i = ny * w + nx;
            if !outside[i] && !mask[(bounds.y0 + ny) * stride + bounds.x0 + nx] {
                outside[i] = true;
                stack.push((nx, ny));
            }
        };
        if x + 1 < w {
            visit(x + 1, y, &mut stack);
        }
        if x > 0 {
            visit(x - 1, y, &mut stack);
        }
        if y + 1 < h {
            visit(x, y + 1, &mut stack);
        }
        if y > 0 {
            visit(x, y - 1, &mut stack);
        }
    }

    let enclosed = outside.iter().filter(|o| !**o).count();
    enclosed as f32 / (w * h) as f32
}

pub(crate) fn parse_rank(label: &str) -> Option<Rank> {
    // The client draws a ten as "10", two characters, where card notation
    // elsewhere uses a single "T".
    if label == "10" {
        return Some(Rank::Ten);
    }
    label.chars().next().and_then(Rank::from_char)
}

pub(crate) fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

pub(crate) fn read_u32(file: &mut impl Read) -> io::Result<u32> {
    let mut word = [0u8; 4];
    file.read_exact(&mut word)?;
    Ok(u32::from_le_bytes(word))
}

pub(crate) fn read_label(file: &mut impl Read) -> io::Result<String> {
    let mut len = [0u8; 1];
    file.read_exact(&mut len)?;
    let mut bytes = vec![0u8; len[0] as usize];
    file.read_exact(&mut bytes)?;
    String::from_utf8(bytes).map_err(|_| invalid("label is not valid UTF-8"))
}

pub(crate) fn read_pixels(file: &mut impl Read, count: usize) -> io::Result<Vec<u8>> {
    let mut pixels = vec![0u8; count];
    file.read_exact(&mut pixels)?;
    Ok(pixels)
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

    fn templates() -> Templates {
        Templates::load(data("card_templates.bin")).expect("templates should load")
    }

    /// Loads a raw frame written by the extraction tooling.
    fn frame_bytes(name: &str) -> (usize, usize, Vec<u8>) {
        let raw = std::fs::read(data("frames").join(name)).expect("frame should exist");
        let w = u32::from_le_bytes(raw[0..4].try_into().expect("header")) as usize;
        let h = u32::from_le_bytes(raw[4..8].try_into().expect("header")) as usize;
        (w, h, raw[8..].to_vec())
    }

    #[test]
    fn templates_load_with_the_full_deck_of_glyphs() {
        let tpl = templates();
        assert_eq!(tpl.rank_count(), 13, "one glyph per rank");
        assert_eq!(tpl.suit_count(), 4);
    }

    #[test]
    fn a_file_that_is_not_a_template_set_is_rejected() {
        let path = std::env::temp_dir().join("poker_vision_bad.bin");
        std::fs::write(&path, b"nonsense").expect("write");
        assert!(Templates::load(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn normalising_makes_a_dimmed_glyph_match_a_bright_one() {
        // A showdown card is drawn greyed out. Contrast stretching is what lets
        // it match the same template as the bright version.
        let bright = Gray::new(2, 2, vec![0, 255, 0, 255]);
        let dimmed = Gray::new(2, 2, vec![80, 180, 80, 180]);
        assert_eq!(bright.normalised(), dimmed.normalised());
    }

    #[test]
    fn a_flat_patch_normalises_to_blank_rather_than_amplified_noise() {
        let flat = Gray::new(2, 2, vec![128, 130, 129, 131]);
        assert_eq!(flat.normalised().pixels, vec![255; 4]);
    }

    /// The port is only useful if it agrees with the validated implementation.
    #[test]
    fn rust_reads_match_the_reference_implementation() {
        let tpl = templates();
        let expected = std::fs::read_to_string(data("frames/expected.txt")).expect("expectations");

        let mut checked = 0;
        for line in expected.lines().filter(|l| !l.trim().is_empty()) {
            let mut parts = line.split_whitespace();
            let name = parts.next().expect("frame name");
            let (w, h, bytes) = frame_bytes(name);
            let frame = Frame::new(w, h, &bytes);
            let reads = read_cards(&frame, &tpl, Thresholds::default());

            let wanted: Vec<&str> = parts.collect();
            assert_eq!(
                reads.len(),
                wanted.len(),
                "{name}: found {} cards, reference found {}",
                reads.len(),
                wanted.len()
            );

            for (read, want) in reads.iter().zip(wanted) {
                let mut field = want.split(',');
                let x: usize = field.next().expect("x").parse().expect("x");
                let y: usize = field.next().expect("y").parse().expect("y");
                let card = field.next().expect("card");
                assert_eq!((read.x, read.y), (x, y), "{name}: card position");
                match card {
                    "-" => assert!(
                        read.card.is_none(),
                        "{name}: read {:?} at ({x},{y}) where the reference refused \
                         (distance {:.1}, margin {:.1})",
                        read.card,
                        read.distance,
                        read.margin
                    ),
                    text => {
                        let got = read.card.map(|c| c.to_string());
                        let normalised = text.replace("10", "T");
                        assert_eq!(
                            got.as_deref(),
                            Some(normalised.as_str()),
                            "{name}: at ({x},{y}) distance {:.1} margin {:.1}",
                            read.distance,
                            read.margin
                        );
                    }
                }
                checked += 1;
            }
        }
        assert!(checked >= 10, "only {checked} cards were compared");
    }

    #[test]
    fn occluded_cards_are_refused_rather_than_guessed() {
        // In this frame the pot readout is drawn over the top of two board
        // cards. Guessing at them is exactly the failure this module exists to
        // avoid.
        let tpl = templates();
        let (w, h, bytes) = frame_bytes("20260818-103911-022.rgb");
        let frame = Frame::new(w, h, &bytes);
        let reads = read_cards(&frame, &tpl, Thresholds::default());

        let refused: Vec<&CardRead> = reads.iter().filter(|r| !r.is_confident()).collect();
        assert_eq!(refused.len(), 2, "both covered cards should be refused");
        for read in refused {
            assert!(
                read.distance > Thresholds::default().max_distance
                    || read.margin < Thresholds::default().min_margin,
                "a refusal must have a reason: {read:?}"
            );
        }

        let confident: Vec<&CardRead> = reads.iter().filter(|r| r.is_confident()).collect();
        assert_eq!(confident.len(), 1);
        assert_eq!(confident[0].card.map(|c| c.to_string()).as_deref(), Some("7s"));
        assert!(confident[0].distance < 10.0, "a clean read scores low");
    }

    #[test]
    fn a_clean_frame_reads_every_card_confidently() {
        let tpl = templates();
        let (w, h, bytes) = frame_bytes("20260818-103636-001.rgb");
        let frame = Frame::new(w, h, &bytes);
        let reads = read_cards(&frame, &tpl, Thresholds::default());

        assert!(!reads.is_empty());
        assert!(
            reads.iter().all(|r| r.is_confident()),
            "nothing occludes this board: {reads:?}"
        );
        let cards: Vec<String> = reads
            .iter()
            .filter_map(|r| r.card.map(|c| c.to_string()))
            .collect();
        assert_eq!(cards, vec!["9c", "Jd", "Kd"]);
    }

    #[test]
    fn tightening_the_thresholds_refuses_more_and_never_reads_differently() {
        // A stricter gate may decline a card, but it must never change one
        // read into a different card.
        let tpl = templates();
        let (w, h, bytes) = frame_bytes("20260818-104236-010.rgb");
        let frame = Frame::new(w, h, &bytes);

        let lenient = read_cards(&frame, &tpl, Thresholds::default());
        let strict = read_cards(
            &frame,
            &tpl,
            Thresholds {
                max_distance: 6.0,
                min_margin: 10.0,
                ..Thresholds::default()
            },
        );

        assert_eq!(lenient.len(), strict.len());
        for (a, b) in lenient.iter().zip(&strict) {
            if let (Some(x), Some(y)) = (a.card, b.card) {
                assert_eq!(x, y, "a tighter threshold changed a card");
            }
        }
        assert!(strict.iter().filter(|r| r.is_confident()).count() <= lenient.len());
    }

    #[test]
    #[should_panic(expected = "buffer size must match")]
    fn a_mis_sized_buffer_is_rejected() {
        Frame::new(10, 10, &[0u8; 100]);
    }
}
