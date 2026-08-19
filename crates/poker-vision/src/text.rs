//! Reading the client's numeric readouts — pot, stacks and bet amounts.
//!
//! # Colour comes first
//!
//! The client colour-codes its readouts: stacks in cyan, pot and bets in gold.
//! Masking by colour separates a readout from the felt, the chrome and every
//! other readout before any shape work happens — the same trick that makes
//! red-versus-black suit detection exact rather than statistical.
//!
//! # Glyphs are never resized
//!
//! Text is drawn at a handful of fixed pixel heights. Scaling glyphs into a
//! common box to compare them destroys the thing being compared: resampling a
//! 9-pixel glyph up to 22 pixels invents interpolation artifacts that a
//! natively-22-pixel glyph does not share, and clustering then splits a single
//! character into dozens of apparent classes. Templates are therefore keyed by
//! exact `(height, width)` and a candidate only ever meets templates of its own
//! size. Several templates may share a label, since a digit has a few
//! anti-aliasing variants; that costs one comparison each and nothing else.
//!
//! # Refusing beats guessing
//!
//! As with cards, a misread number is worse than no number: a bot that thinks
//! the pot is 15 when it is 150 will size every bet wrong and look healthy
//! doing it. A glyph is accepted only when it matches closely *and* beats the
//! nearest differently-labelled template by a margin, and a run containing an
//! unreadable character yields no value at all.

use crate::{components, invalid, read_label, read_u32, Bounds, Frame};
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

const MAGIC: &[u8; 4] = b"PKGT";
const VERSION: u32 = 1;

/// Which readout a glyph belongs to, identified by its colour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ink {
    /// Seat stacks.
    Cyan,
    /// The total pot.
    Gold,
    /// Bet amounts on the felt — and player names, which the client
    /// unfortunately draws in the same white.
    White,
}

/// The least ink a blob may carry and still be considered a character.
///
/// This was three, chosen when the smallest known character — the decimal point
/// on the felt — carried four lit pixels. Then the raise amount box turned out
/// to draw its point with **two**, and a floor of three deleted it: `9.5` read
/// as `95`. A dropped digit is caught, because an unreadable character voids
/// its whole run; a dropped *point* is not, since what remains still parses,
/// and parses to ten times the truth.
///
/// Deriving the floor from the templates was the next attempt and was worse: a
/// glyph the client has drawn faded — mid-animation, or on a ghosted seat —
/// legitimately carries less ink than the template it matches, so that floor
/// dropped real characters too.
///
/// So it sits at two, which no single stray pixel can reach. Speckle that gets
/// through matches no template and therefore voids whatever run it lands in,
/// which costs a reading. That is the right way round: a refusal is a reading
/// not taken, while a dropped character is a number that is wrong.
const MIN_INK: usize = 2;

impl Ink {
    /// Whether a pixel belongs to this readout.
    ///
    /// Thresholds are measured from the captures, where the two inks sit far
    /// apart in every channel with nothing on the table falling between them.
    #[inline]
    fn matches(&self, (r, g, b): (u8, u8, u8)) -> bool {
        let (r, g, b) = (r as i16, g as i16, b as i16);
        match self {
            Ink::Cyan => b > 150 && g > 130 && r < 120 && b - r > 70,
            Ink::Gold => r > 170 && g > 140 && b < 120 && r - b > 80,
            Ink::White => r > 180 && g > 180 && b > 180 && (r - b).abs() < 25 && (g - b).abs() < 25,
        }
    }

    fn from_name(name: &str) -> Option<Ink> {
        match name {
            "cyan" => Some(Ink::Cyan),
            "gold" => Some(Ink::Gold),
            "white" => Some(Ink::White),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Ink::Cyan => "cyan",
            Ink::Gold => "gold",
            Ink::White => "white",
        }
    }
}

/// One character at one rendered size.
#[derive(Debug, Clone)]
struct Glyph {
    ink: Ink,
    label: char,
    width: usize,
    height: usize,
    /// Binary mask, one byte per pixel, 0 or 255.
    mask: Vec<u8>,
}

/// The glyph shapes the number reader matches against.
#[derive(Debug, Clone)]
pub struct GlyphTemplates {
    glyphs: Vec<Glyph>,
}

impl GlyphTemplates {
    /// Reads a template file produced by the extraction tooling.
    pub fn load(path: impl AsRef<Path>) -> io::Result<GlyphTemplates> {
        let mut file = BufReader::new(File::open(path)?);

        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(invalid("not a glyph template file"));
        }
        if read_u32(&mut file)? != VERSION {
            return Err(invalid("unsupported glyph template version"));
        }

        let count = read_u32(&mut file)? as usize;
        let mut glyphs = Vec::with_capacity(count);
        for _ in 0..count {
            let ink = read_label(&mut file)?;
            let ink = Ink::from_name(&ink).ok_or_else(|| invalid(format!("bad ink {ink:?}")))?;
            let label = read_label(&mut file)?;
            let mut chars = label.chars();
            let label = match (chars.next(), chars.next()) {
                (Some(c), None) => c,
                _ => return Err(invalid(format!("label {label:?} is not one character"))),
            };
            let height = read_u32(&mut file)? as usize;
            let width = read_u32(&mut file)? as usize;
            let mut mask = vec![0u8; width * height];
            file.read_exact(&mut mask)?;
            glyphs.push(Glyph {
                ink,
                label,
                width,
                height,
                mask,
            });
        }

        if glyphs.is_empty() {
            return Err(invalid("template file contains no glyphs"));
        }
        Ok(GlyphTemplates { glyphs })
    }

    pub fn len(&self) -> usize {
        self.glyphs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.glyphs.is_empty()
    }

    /// The distinct characters covered, in sorted order.
    pub fn alphabet(&self) -> Vec<char> {
        let mut seen: Vec<char> = self.glyphs.iter().map(|g| g.label).collect();
        seen.sort_unstable();
        seen.dedup();
        seen
    }

    /// The rendered heights covered for one ink, in sorted order.
    pub fn heights(&self, ink: Ink) -> Vec<usize> {
        let mut seen: Vec<usize> = self
            .glyphs
            .iter()
            .filter(|g| g.ink == ink)
            .map(|g| g.height)
            .collect();
        seen.sort_unstable();
        seen.dedup();
        seen
    }
}

/// How confidently a glyph is accepted, and how far apart characters may sit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextThresholds {
    /// Reject beyond this mean absolute difference against the template.
    pub max_distance: f32,
    /// Require the best match to beat the nearest *differently-labelled*
    /// template by at least this much. Variants of one character are close to
    /// each other by construction, so only a rival label counts as ambiguity.
    pub min_margin: f32,
    /// Largest horizontal gap that still joins two characters into one number.
    ///
    /// Measured across every capture: gaps *inside* a readout run from 0 to 18
    /// pixels, and the next glyph along a baseline is then 769 pixels away, at
    /// another seat. With a void that wide between the two populations the
    /// threshold is not a delicate choice, and 20 sits clear of both.
    pub max_gap: usize,
    /// Largest baseline difference that still joins two characters.
    pub max_baseline_drift: usize,
}

impl Default for TextThresholds {
    fn default() -> TextThresholds {
        TextThresholds {
            max_distance: 40.0,
            min_margin: 10.0,
            max_gap: 20,
            max_baseline_drift: 2,
        }
    }
}

/// One numeric readout and where it was found.
#[derive(Debug, Clone, PartialEq)]
pub struct NumberRead {
    /// Top-left of the whole run, in frame coordinates.
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
    pub ink: Ink,
    /// The characters as read, including any `BB` suffix.
    pub text: String,
    /// The parsed amount in big blinds, or `None` when the run did not parse.
    pub value: Option<f64>,
    /// Worst glyph distance in the run.
    pub distance: f32,
}

impl NumberRead {
    pub fn is_confident(&self) -> bool {
        self.value.is_some()
    }
}

/// A glyph-sized blob and what it was read as, if anything.
struct Placed {
    bounds: Bounds,
    /// `None` when no template cleared the thresholds. Kept rather than
    /// discarded: an unreadable character sitting inside a number is the whole
    /// reason that number cannot be trusted.
    label: Option<char>,
    distance: f32,
}

/// Finds every numeric readout in a frame.
///
/// Results are ordered top to bottom, then left to right.
pub fn read_numbers(
    frame: &Frame,
    templates: &GlyphTemplates,
    thresholds: TextThresholds,
) -> Vec<NumberRead> {
    let mut reads = Vec::new();
    for ink in [Ink::Cyan, Ink::Gold, Ink::White] {
        reads.extend(read_ink(frame, templates, thresholds, ink));
    }
    reads.sort_by_key(|r| (r.y, r.x));
    reads
}

/// Reads a number inside one named rectangle.
///
/// [`read_numbers`] sweeps the whole frame and requires white glyphs to sit on
/// green felt, which is what tells a bet from a player's name. That test is
/// wrong for a box the client draws on black — the bet amount field — so this
/// reads a place the caller already knows, and skips it.
pub fn read_number_in(
    frame: &Frame,
    templates: &GlyphTemplates,
    thresholds: TextThresholds,
    ink: Ink,
    at: (usize, usize, usize, usize),
) -> Option<NumberRead> {
    let (x0, y0, x1, y1) = at;
    if x0 >= x1 || y0 >= y1 || x1 > frame.width || y1 > frame.height {
        return None;
    }

    let (w, h) = (frame.width, frame.height);
    let mut mask = vec![false; w * h];
    for y in y0..y1 {
        for x in x0..x1 {
            mask[y * w + x] = ink.matches(frame.pixel(x, y));
        }
    }

    let sizes = templates.heights(ink);
    let mut placed: Vec<Placed> = components(&mask, w, h)
        .into_iter()
        .filter(|b| {
            sizes.iter().any(|size| size.abs_diff(b.height()) <= 1)
                && b.width() <= 32
                && ink_area(&mask, w, *b) >= MIN_INK
        })
        .map(|b| match best_glyph(&mask, w, h, b, ink, templates, thresholds) {
            Some((label, distance)) => Placed {
                bounds: b,
                label: Some(label),
                distance,
            },
            None => Placed {
                bounds: b,
                label: None,
                distance: f32::INFINITY,
            },
        })
        .collect();
    placed.sort_by_key(|p| p.bounds.x0);

    // One box holds one number, so the widest run is it.
    group_runs(&placed, ink, thresholds)
        .into_iter()
        .max_by_key(|run| run.width)
}

fn read_ink(
    frame: &Frame,
    templates: &GlyphTemplates,
    thresholds: TextThresholds,
    ink: Ink,
) -> Vec<NumberRead> {
    let (w, h) = (frame.width, frame.height);
    let mut mask = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            mask[y * w + x] = ink.matches(frame.pixel(x, y));
        }
    }

    // Only blobs the size of a rendered glyph are worth matching; the felt
    // texture and the chrome produce plenty of others.
    // Every blob of glyph height is kept, readable or not. Dropping the
    // unreadable ones is what would let an occluded digit silently disappear
    // and turn 213 into a confident 1.
    //
    // The height test carries the same one-pixel slack the matcher does, and
    // for the same reason: within one readout the client will render `1` at 13
    // pixels and the `B` beside it at 14. A stricter filter here would drop the
    // `B` before it ever reached a template.
    let sizes = templates.heights(ink);
    let mut placed: Vec<Placed> = components(&mask, w, h)
        .into_iter()
        .filter(|b| {
            sizes.iter().any(|size| size.abs_diff(b.height()) <= 1)
                && b.width() <= 32
                && ink_area(&mask, w, *b) >= MIN_INK
                && (ink != Ink::White || on_felt(frame, *b))
        })
        .map(|b| match best_glyph(&mask, w, h, b, ink, templates, thresholds) {
            Some((label, distance)) => Placed {
                bounds: b,
                label: Some(label),
                distance,
            },
            None => Placed {
                bounds: b,
                label: None,
                distance: f32::INFINITY,
            },
        })
        .collect();

    // Left to right, so runs build in reading order.
    placed.sort_by_key(|p| p.bounds.x0);
    group_runs(&placed, ink, thresholds)
}

/// Joins accepted glyphs into readouts.
///
/// Characters of one readout share a baseline and sit close together, while two
/// readouts are hundreds of pixels apart, so a gap test and a baseline test
/// separate them with a wide safety margin.
fn group_runs(placed: &[Placed], ink: Ink, thresholds: TextThresholds) -> Vec<NumberRead> {
    let mut runs: Vec<Vec<&Placed>> = Vec::new();
    for glyph in placed {
        let joined = runs.iter_mut().rev().find(|run| {
            let last = run.last().expect("runs are never empty");
            let drift = last.bounds.y1.abs_diff(glyph.bounds.y1);
            drift <= thresholds.max_baseline_drift
                && glyph.bounds.x0 >= last.bounds.x1
                && glyph.bounds.x0 - last.bounds.x1 <= thresholds.max_gap
        });
        match joined {
            Some(run) => run.push(glyph),
            None => runs.push(vec![glyph]),
        }
    }

    runs.into_iter()
        .filter(|run| {
            run.iter()
                .any(|g| g.label.is_some_and(|c| c.is_ascii_digit()))
        })
        .filter_map(|run| {
            // One unreadable character condemns the whole readout. A number
            // missing a digit still parses, and parses to something plausible
            // and wrong, which is worse than no reading at all.
            let complete = run.iter().all(|g| g.label.is_some());
            let text: String = run.iter().map(|g| g.label.unwrap_or('?')).collect();
            let value = complete.then(|| parse_amount(&text)).flatten();

            // A run that read perfectly well and still is not an amount was
            // never a readout — a timer badge, a hand count. Drop it. A run
            // with an unreadable character is kept and refused, because that
            // one might have been a readout, and losing it quietly is the
            // failure this whole module is built to avoid.
            if complete && value.is_none() {
                return None;
            }
            let x0 = run.iter().map(|g| g.bounds.x0).min().expect("non-empty");
            let y0 = run.iter().map(|g| g.bounds.y0).min().expect("non-empty");
            let x1 = run.iter().map(|g| g.bounds.x1).max().expect("non-empty");
            let y1 = run.iter().map(|g| g.bounds.y1).max().expect("non-empty");
            Some(NumberRead {
                x: x0,
                y: y0,
                width: x1 - x0 + 1,
                height: y1 - y0 + 1,
                ink,
                value,
                text,
                distance: run
                    .iter()
                    .filter_map(|g| g.label.map(|_| g.distance))
                    .fold(0.0, f32::max),
            })
        })
        .collect()
}

/// Parses a readout such as `348.5BB` into big blinds.
///
/// The suffix is required. Every amount this client displays carries it, and
/// plenty of other things on the table are numbers that are not amounts — the
/// countdown badge on each seat, the hand counter, the digits inside a name
/// like `Domg1025`. Demanding `BB` tells an amount from all of them with one
/// test.
///
/// Whatever remains after the suffix must be a plain number: a stray character
/// means the run was mis-segmented, and that has to surface as a refusal rather
/// than as the part that happened to parse.
fn parse_amount(text: &str) -> Option<f64> {
    let digits = text.strip_suffix("BB")?;
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit() || b == b'.') {
        return None;
    }
    digits.parse::<f64>().ok()
}

/// Whether a glyph sits on green table felt rather than on a name plate.
///
/// This is what separates a bet amount from a player name, the client drawing
/// both in the same white — and it is the same test used to harvest the white
/// templates, so reading and training agree on what counts as a bet.
fn on_felt(frame: &Frame, bounds: Bounds) -> bool {
    const PAD: usize = 4;
    let x0 = bounds.x0.saturating_sub(PAD);
    let y0 = bounds.y0.saturating_sub(PAD);
    let x1 = (bounds.x1 + PAD).min(frame.width - 1);
    let y1 = (bounds.y1 + PAD).min(frame.height - 1);

    let mut green = 0usize;
    let mut total = 0usize;
    for y in y0..=y1 {
        for x in x0..=x1 {
            let (r, g, b) = frame.pixel(x, y);
            let (r, g, b) = (r as i16, g as i16, b as i16);
            total += 1;
            if g > r + 15 && g > b + 15 {
                green += 1;
            }
        }
    }
    total > 0 && green * 100 / total > 35
}

/// How many pixels of a blob's box are lit.
fn ink_area(mask: &[bool], stride: usize, bounds: Bounds) -> usize {
    (bounds.y0..=bounds.y1)
        .map(|y| (bounds.x0..=bounds.x1).filter(|x| mask[y * stride + x]).count())
        .sum()
}

/// Best template match for one blob.
///
/// A candidate meets templates within one pixel of its own size, and the
/// comparison window is sampled afresh from the mask at each alignment rather
/// than the glyph being stretched to fit. That one-pixel slack matters: the
/// same digit renders 17 or 18 pixels tall depending on where the readout
/// falls against the pixel grid, and treating those as unrelated sizes would
/// mean hand-labelling a second full alphabet for every rendering.
fn best_glyph(
    mask: &[bool],
    stride: usize,
    frame_height: usize,
    bounds: Bounds,
    ink: Ink,
    templates: &GlyphTemplates,
    thresholds: TextThresholds,
) -> Option<(char, f32)> {
    let (w, h) = (bounds.width(), bounds.height());
    let mut best: Option<(char, f32)> = None;
    let mut rival = f32::INFINITY;

    for glyph in &templates.glyphs {
        if glyph.ink != ink || glyph.width.abs_diff(w) > 1 || glyph.height.abs_diff(h) > 1 {
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
                if sx + glyph.width > stride || sy + glyph.height > frame_height {
                    continue;
                }
                let mut total = 0u32;
                for y in 0..glyph.height {
                    for x in 0..glyph.width {
                        let ink_here = if mask[(sy + y) * stride + sx + x] { 255u8 } else { 0 };
                        total += ink_here.abs_diff(glyph.mask[y * glyph.width + x]) as u32;
                    }
                }
                closest = closest.min(total as f32 / (glyph.width * glyph.height) as f32);
            }
        }
        match best {
            Some((label, score)) if closest < score => {
                if label != glyph.label {
                    rival = rival.min(score);
                }
                best = Some((glyph.label, closest));
            }
            Some((label, _)) if label != glyph.label => rival = rival.min(closest),
            Some(_) => {}
            None => best = Some((glyph.label, closest)),
        }
    }

    match best {
        Some((label, score))
            if score <= thresholds.max_distance && rival - score >= thresholds.min_margin =>
        {
            Some((label, score))
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

    fn templates() -> GlyphTemplates {
        GlyphTemplates::load(data("digit_templates.bin")).expect("templates should load")
    }

    #[test]
    fn templates_cover_everything_a_readout_can_contain() {
        let tpl = templates();
        let alphabet = tpl.alphabet();
        for wanted in "0123456789.B".chars() {
            assert!(alphabet.contains(&wanted), "no template for {wanted:?}");
        }
        assert_eq!(alphabet.len(), 12, "nothing beyond digits, a point and B");
    }

    #[test]
    fn a_file_that_is_not_a_glyph_set_is_rejected() {
        let path = std::env::temp_dir().join("poker_vision_bad_glyphs.bin");
        std::fs::write(&path, b"nonsense").expect("write");
        assert!(GlyphTemplates::load(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn each_ink_is_rendered_at_its_measured_sizes() {
        let tpl = templates();
        assert_eq!(tpl.heights(Ink::Cyan), vec![3, 4, 14, 17, 18, 22]);
        assert_eq!(tpl.heights(Ink::Gold), vec![3, 11, 15]);
        // Two white sizes on the felt, and two a pixel smaller inside the
        // raise amount box, which the client draws in its own face.
        assert_eq!(tpl.heights(Ink::White), vec![1, 2, 12, 13]);
    }

    #[test]
    fn no_pixel_is_claimed_by_two_inks() {
        // If one pixel could read as two inks, a readout could leak glyphs into
        // another colour's runs.
        let inks = [Ink::Cyan, Ink::Gold, Ink::White];
        for r in (0u8..=250).step_by(5) {
            for g in (0u8..=250).step_by(5) {
                for b in (0u8..=250).step_by(5) {
                    let px = (r, g, b);
                    let claimed: Vec<&Ink> = inks.iter().filter(|i| i.matches(px)).collect();
                    assert!(claimed.len() <= 1, "{px:?} matches {claimed:?}");
                }
            }
        }
    }

    #[test]
    fn a_readout_parses_into_big_blinds() {
        assert_eq!(parse_amount("348.5BB"), Some(348.5));
        assert_eq!(parse_amount("1200BB"), Some(1200.0));
        assert_eq!(parse_amount("0.5BB"), Some(0.5));
    }

    #[test]
    fn a_mis_segmented_run_refuses_rather_than_parsing_what_it_can() {
        // Two readouts joined by too generous a gap test would look like this,
        // and it must not come back as a number.
        assert_eq!(parse_amount("12B34"), None);
        assert_eq!(parse_amount("BB"), None);
        assert_eq!(parse_amount(""), None);
        assert_eq!(parse_amount("1.2.3BB"), None);
    }

    #[test]
    fn a_number_without_the_bb_suffix_is_not_an_amount() {
        // The seat countdown badge, the hand counter and the digits inside a
        // name like Domg1025 are all numbers on this table that mean nothing to
        // a poker decision.
        assert_eq!(parse_amount("22"), None);
        assert_eq!(parse_amount("1025"), None);
    }

    /// Loads a raw frame written by the extraction tooling.
    fn frame_bytes(name: &str) -> (usize, usize, Vec<u8>) {
        let raw = std::fs::read(data("frames").join(name)).expect("frame should exist");
        let w = u32::from_le_bytes(raw[0..4].try_into().expect("header")) as usize;
        let h = u32::from_le_bytes(raw[4..8].try_into().expect("header")) as usize;
        (w, h, raw[8..].to_vec())
    }

    /// Ground truth read off the screenshot by eye: seven seat stacks, the pot
    /// and three posted bets, on a seven-handed table at 0.10/0.20.
    #[test]
    fn every_readout_on_a_verified_frame_matches_the_screen() {
        let tpl = templates();
        let (w, h, bytes) = frame_bytes("20260818-104053-025.rgb");
        let frame = Frame::new(w, h, &bytes);
        let reads = read_numbers(&frame, &tpl, TextThresholds::default());

        let got: Vec<(&str, Option<f64>)> = reads
            .iter()
            .map(|r| (r.ink.name(), r.value))
            .collect();
        assert_eq!(
            got,
            vec![
                ("cyan", Some(90.5)),   // Domg1025
                ("cyan", Some(139.8)),  // Ec4Hooo
                ("white", Some(0.5)),   // the small blind, posted on the felt
                ("gold", Some(2.5)),    // Total Pot
                ("cyan", Some(56.7)),   // Suuuun
                ("cyan", Some(612.3)),  // almightypeen
                ("white", Some(1.0)),   // the big blind
                ("white", Some(1.0)),   // the straddle
                ("cyan", Some(278.0)),  // Will Probcal
                ("cyan", Some(200.0)),  // kanyan9896
                ("cyan", Some(122.5)),  // lunakat, the hero seat
            ],
            "reads were {reads:#?}"
        );
    }

    #[test]
    fn a_readout_caught_mid_reflow_is_refused_rather_than_half_read() {
        // The client redraws the table when a player joins or leaves, and for a
        // few frames the old seat position is still on screen, faded. The faded
        // copy fails the colour mask in places, and half a stack figure is
        // exactly the kind of confident nonsense that must not escape.
        let tpl = templates();
        let (w, h, bytes) = frame_bytes("20260818-103911-022.rgb");
        let frame = Frame::new(w, h, &bytes);
        let reads = read_numbers(&frame, &tpl, TextThresholds::default());

        let refused: Vec<&NumberRead> = reads.iter().filter(|r| !r.is_confident()).collect();
        assert_eq!(refused.len(), 1, "reads were {reads:#?}");
        assert!(
            refused[0].text.contains('?'),
            "a refusal names the characters it could not read: {:?}",
            refused[0].text
        );

        // The authoritative copy at the seat's new position still reads.
        assert!(reads.iter().any(|r| r.value == Some(613.3)));
    }

    #[test]
    fn a_dialog_over_the_table_yields_nothing_rather_than_stale_numbers() {
        // This frame is the client's "Timeout Sit Out" modal, which dims the
        // whole table behind it. Reading dimmed figures would hand the bot a
        // snapshot of a table it is no longer playing.
        let tpl = templates();
        let (w, h, bytes) = frame_bytes("20260818-104742-014.rgb");
        let frame = Frame::new(w, h, &bytes);
        let reads = read_numbers(&frame, &tpl, TextThresholds::default());
        assert!(reads.is_empty(), "reads were {reads:#?}");
    }

    /// Not an assertion — a harness for eyeballing what the reader produces on
    /// a real frame while ground truth is being established.
    #[test]
    #[ignore = "diagnostic; run with --ignored --nocapture"]
    fn dump_numbers_from_every_fixture_frame() {
        let tpl = templates();
        let dir = data("frames");
        let mut names: Vec<_> = std::fs::read_dir(&dir)
            .expect("frames")
            .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
            .filter(|n| n.ends_with(".rgb"))
            .collect();
        names.sort();
        for name in names {
            let raw = std::fs::read(dir.join(&name)).expect("frame");
            let w = u32::from_le_bytes(raw[0..4].try_into().expect("header")) as usize;
            let h = u32::from_le_bytes(raw[4..8].try_into().expect("header")) as usize;
            let frame = Frame::new(w, h, &raw[8..]);
            let reads = read_numbers(&frame, &tpl, TextThresholds::default());
            println!("
{name}: {} readouts", reads.len());
            for r in &reads {
                println!(
                    "  {:4} x={:4} y={:4} {:3}x{:2}  {:>10}  value={:?} d={:.1}",
                    r.ink.name(), r.x, r.y, r.width, r.height, r.text, r.value, r.distance
                );
            }
        }
    }
}
