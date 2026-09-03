//! Reading CoinPoker's numeric readouts — pot, stacks and bet amounts.
//!
//! A sibling of ClubGG's `text.rs`, not a reuse of it: the two clients
//! colour-code their numbers differently, so the `Ink` here is its own small
//! type rather than an extra variant bolted onto ClubGG's. Keeping the two
//! separate means a threshold tuned for one client can never quietly start
//! matching pixels on the other's table.
//!
//! # Two inks, but no ClubGG-style safety net
//!
//! CoinPoker draws every bet-related number — the pot, the bet-chip amount,
//! the caption on the Bet/Raise button — in white, and stacks in gold.
//! Simpler than ClubGG's three inks, but ClubGG can get away with scanning
//! its *whole* frame because every real amount there carries a `BB` suffix,
//! which tells an amount from the seat countdown badge, the hand counter,
//! and the digits inside a player's name in one test. CoinPoker's amounts
//! have no suffix, and it draws its own seat/hand-count badges in the same
//! white as its amounts — confirmed by running an early whole-frame version
//! of this reader against a real capture, where it read a badge number as a
//! bogus pot value. So there is no [`read_numbers`]-style whole-frame
//! function here at all: [`read_number_in`] only ever looks at a region the
//! caller already knows holds an amount, which is the one thing that has
//! been shown safe.
//!
//! Measured at a 1280x960 table, on `captures-coinpoker/h181631-0-1280x960.png`:
//! the pot's "0.14" reads as (255,255,255) and a stack's "1.76" reads as
//! (255,197,51). Pot and bet-chip digits stand 19px tall there; stack digits
//! stand 22-23px tall.
//!
//! # A known gap: two digits are the wrong size
//!
//! White glyphs render at more than one size — 19px for the pot and
//! bet-chip, but closer to 20-21px on the Bet/Raise button's own caption —
//! and `5` and `9` currently only have a caption-sized (20px) template. A
//! pot or bet-chip reading containing a 5 or 9 will therefore refuse rather
//! than misread, which is the safe failure, but it is still a gap: find a
//! pot- or bet-chip-sized 5 or 9 and re-point its manifest entry in
//! `tools/build-coinpoker-text-templates.ps1`.
//!
//! The decimal point turned out to need the same per-location split: at
//! only 3x3-4px, it is small enough that the pot's point, a bet-chip's, and
//! (were one ever sourced) the stack's are each their own size bucket, not
//! one shared glyph — confirmed the hard way, by a point that matched fine
//! on the frame it was sourced from and refused on every other frame until
//! a second and third `.` template were added for the other two contexts.
//! `CoinPokerGlyphTemplates` allows more than one template per label for
//! exactly this reason: `best_glyph` keeps whichever scores lowest.

use crate::{components, invalid, read_label, read_u32, Bounds, Frame};
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

const MAGIC: &[u8; 4] = b"PKCT";
const VERSION: u32 = 1;

/// Which readout a glyph belongs to, identified by its colour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ink {
    /// Seat stacks, drawn gold.
    Gold,
    /// The pot, bet-chip amounts, and Bet/Raise button captions — all drawn
    /// in the same white, with nothing else on this table competing for it.
    White,
}

/// The least ink a blob may carry and still be considered a character.
///
/// See ClubGG's `text::MIN_INK` for why this exists at all: a stray lit
/// pixel must never pass as a decimal point. Not yet measured against a real
/// CoinPoker decimal point — carried over as a starting value, to be
/// corrected once real glyphs are harvested.
const MIN_INK: usize = 2;

impl Ink {
    /// Whether a pixel belongs to this readout.
    #[inline]
    fn matches(&self, (r, g, b): (u8, u8, u8)) -> bool {
        let (r, g, b) = (r as i16, g as i16, b as i16);
        match self {
            Ink::Gold => r > 200 && g > 150 && b < 100 && r - b > 100,
            Ink::White => r > 200 && g > 200 && b > 200 && (r - b).abs() < 20 && (g - b).abs() < 20,
        }
    }

    fn from_name(name: &str) -> Option<Ink> {
        match name {
            "gold" => Some(Ink::Gold),
            "white" => Some(Ink::White),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
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
///
/// Same file layout as ClubGG's `GlyphTemplates` (`PKCT` instead of `PKGT`
/// as the magic, so the two can never be loaded into each other by mistake),
/// kept as a separate type because its `Ink` is a different, smaller enum.
#[derive(Debug, Clone)]
pub struct CoinPokerGlyphTemplates {
    glyphs: Vec<Glyph>,
}

impl CoinPokerGlyphTemplates {
    pub fn load(path: impl AsRef<Path>) -> io::Result<CoinPokerGlyphTemplates> {
        let mut file = BufReader::new(File::open(path)?);

        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(invalid("not a CoinPoker glyph template file"));
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
            glyphs.push(Glyph { ink, label, width, height, mask });
        }

        if glyphs.is_empty() {
            return Err(invalid("template file contains no glyphs"));
        }
        Ok(CoinPokerGlyphTemplates { glyphs })
    }

    pub fn len(&self) -> usize {
        self.glyphs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.glyphs.is_empty()
    }

    pub fn alphabet(&self) -> Vec<char> {
        let mut seen: Vec<char> = self.glyphs.iter().map(|g| g.label).collect();
        seen.sort_unstable();
        seen.dedup();
        seen
    }

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
///
/// Values carried over from ClubGG's `TextThresholds::default()` as a
/// starting point — CoinPoker's own gaps and drift have not been measured
/// yet, since there is no template set to test them against.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextThresholds {
    pub max_distance: f32,
    pub min_margin: f32,
    pub max_gap: usize,
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
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
    pub ink: Ink,
    pub text: String,
    pub value: Option<f64>,
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
    label: Option<char>,
    distance: f32,
}

/// Reads the number inside one named rectangle, such as the pot badge or a
/// seat's stack figure.
///
/// # Why this takes a region, and `read_numbers` does not exist
///
/// ClubGG can scan its whole frame safely because every real amount there
/// carries a `BB` suffix, which tells an amount from the seat countdown
/// badge, the hand counter, and the digits inside a player's name with one
/// test. CoinPoker's amounts carry no suffix at all, and it draws its own
/// seat/hand-count badges in the same white as the pot and bet-chip figures
/// — confirmed by running an early, whole-frame version of this reader
/// against a real capture: it read a badge number as a bogus pot value.
/// Without a `BB`-style signal to separate an amount from every other white
/// or gold number on the table, a whole-frame scan cannot be trusted, so
/// this only ever looks where the caller already knows an amount belongs.
pub fn read_number_in(
    frame: &Frame,
    templates: &CoinPokerGlyphTemplates,
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
            Some((label, distance)) => Placed { bounds: b, label: Some(label), distance },
            None => Placed { bounds: b, label: None, distance: f32::INFINITY },
        })
        .collect();
    placed.sort_by_key(|p| p.bounds.x0);

    // One region holds one number, so the widest run is it.
    group_runs(&placed, ink, thresholds)
        .into_iter()
        .max_by_key(|run| run.width)
}

/// Joins accepted glyphs into readouts, the same way ClubGG's `group_runs`
/// does: a gap test and a baseline test separate one readout from the next.
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
        .filter(|run| run.iter().any(|g| g.label.is_some_and(|c| c.is_ascii_digit())))
        .filter_map(|run| {
            let complete = run.iter().all(|g| g.label.is_some());
            let text: String = run.iter().map(|g| g.label.unwrap_or('?')).collect();
            let value = complete.then(|| parse_amount(&text)).flatten();
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
                distance: run.iter().filter_map(|g| g.label.map(|_| g.distance)).fold(0.0, f32::max),
            })
        })
        .collect()
}

/// Parses a plain decimal readout, such as `0.14` or `1.76`.
///
/// Unlike ClubGG, no suffix to demand: CoinPoker's amounts carry no `BB`
/// (or currency symbol) baked into the same white/gold text, so the amount
/// is just whatever digits and one decimal point were read. This is looser
/// than ClubGG's `parse_amount` was able to be, and it is an open question
/// whether some other on-table number (a hand count, a seat number) also
/// happens to render in one of these two inks and would be wrongly accepted
/// here — something to check once real captures are being read, not
/// something this module can rule out from geometry alone yet.
fn parse_amount(text: &str) -> Option<f64> {
    if text.is_empty() || !text.bytes().all(|b| b.is_ascii_digit() || b == b'.') {
        return None;
    }
    text.parse::<f64>().ok()
}

fn ink_area(mask: &[bool], stride: usize, bounds: Bounds) -> usize {
    (bounds.y0..=bounds.y1)
        .map(|y| (bounds.x0..=bounds.x1).filter(|x| mask[y * stride + x]).count())
        .sum()
}

fn best_glyph(
    mask: &[bool],
    stride: usize,
    frame_height: usize,
    bounds: Bounds,
    ink: Ink,
    templates: &CoinPokerGlyphTemplates,
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

    /// The template file is only as good as what it reads back correctly.
    /// Checked against a frame whose pot and stacks were confirmed by eye
    /// against the live client at capture time - see the project's "verify
    /// the instrument first" rule.
    #[test]
    fn reads_the_pot_and_both_stacks_from_a_known_table() {
        let path = data("digit_templates_coinpoker.bin");
        if !path.exists() {
            // Not built yet in this checkout; nothing to verify against.
            return;
        }
        let templates = CoinPokerGlyphTemplates::load(&path).expect("templates should load");

        let raw = std::fs::read(data("frames/coinpoker-h181631.rgb")).expect("fixture frame");
        let width = u32::from_le_bytes(raw[0..4].try_into().unwrap()) as usize;
        let height = u32::from_le_bytes(raw[4..8].try_into().unwrap()) as usize;
        let frame = Frame::new(width, height, &raw[8..]);

        let thresholds = TextThresholds::default();
        // Same regions harvest-coinpoker-text.ps1 scans - measured generous
        // boxes around where each readout actually renders.
        let pot = read_number_in(&frame, &templates, thresholds, Ink::White, (560, 335, 740, 375));
        let hero_stack = read_number_in(&frame, &templates, thresholds, Ink::Gold, (560, 845, 730, 895));
        let villain_stack = read_number_in(&frame, &templates, thresholds, Ink::Gold, (555, 205, 730, 250));

        // Pot 0.14, hero (niki88) 1.76, villain (baloobond10) 0.89 -
        // confirmed by eye against the live CoinPoker client at capture time.
        assert_eq!(pot.and_then(|r| r.value), Some(0.14), "pot");
        assert_eq!(hero_stack.and_then(|r| r.value), Some(1.76), "hero stack");
        assert_eq!(villain_stack.and_then(|r| r.value), Some(0.89), "villain stack");
    }

    /// mentalrun (villain) had a live 0.03 bet showing here, on top of a
    /// 0.04 collected pot from an earlier street - confirmed by eye against
    /// the live client, and the source of `coinpoker_view::regions::VILLAIN_BET`.
    #[test]
    fn reads_a_live_bet_chip_amount_from_a_known_table() {
        let path = data("digit_templates_coinpoker.bin");
        if !path.exists() {
            return;
        }
        let templates = CoinPokerGlyphTemplates::load(&path).expect("templates should load");

        let raw = std::fs::read(data("frames/coinpoker-h205314.rgb")).expect("fixture frame");
        let width = u32::from_le_bytes(raw[0..4].try_into().unwrap()) as usize;
        let height = u32::from_le_bytes(raw[4..8].try_into().unwrap()) as usize;
        let frame = Frame::new(width, height, &raw[8..]);

        let thresholds = TextThresholds::default();
        let pot = read_number_in(&frame, &templates, thresholds, Ink::White, (560, 335, 740, 375));
        let villain_bet = read_number_in(&frame, &templates, thresholds, Ink::White, (470, 270, 540, 310));

        assert_eq!(pot.and_then(|r| r.value), Some(0.07), "pot: 0.04 collected + 0.03 live bet");
        assert_eq!(villain_bet.and_then(|r| r.value), Some(0.03), "villain's live bet-chip");
    }

    #[test]
    fn ink_thresholds_do_not_overlap() {
        // If one pixel could read as both inks, a readout could leak glyphs
        // into the wrong colour's run.
        for r in (0u8..=250).step_by(5) {
            for g in (0u8..=250).step_by(5) {
                for b in (0u8..=250).step_by(5) {
                    let px = (r, g, b);
                    let claimed = [Ink::Gold, Ink::White].iter().filter(|i| i.matches(px)).count();
                    assert!(claimed <= 1, "{px:?} matches both inks");
                }
            }
        }
    }

    #[test]
    fn the_measured_pot_and_stack_colours_are_recognised() {
        // Measured directly off captures-coinpoker/h181631-0-1280x960.png.
        assert!(Ink::White.matches((255, 255, 255)), "the pot's \"0.14\"");
        assert!(Ink::Gold.matches((255, 197, 51)), "a stack's \"1.76\"");
    }

    #[test]
    fn a_file_that_is_not_a_glyph_set_is_rejected() {
        let path = std::env::temp_dir().join("poker_vision_bad_coinpoker_glyphs.bin");
        std::fs::write(&path, b"nonsense").expect("write");
        assert!(CoinPokerGlyphTemplates::load(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_clubgg_glyph_file_is_rejected_by_its_different_magic() {
        // PKCT vs PKGT - the two template sets must never be interchangeable,
        // since one wrong load would match a whole client's numbers against
        // the other client's font.
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/digit_templates.bin");
        if path.exists() {
            assert!(CoinPokerGlyphTemplates::load(&path).is_err());
        }
    }

    #[test]
    fn a_readout_with_no_suffix_still_parses() {
        assert_eq!(parse_amount("0.14"), Some(0.14));
        assert_eq!(parse_amount("1.76"), Some(1.76));
        assert_eq!(parse_amount("100"), Some(100.0));
    }

    #[test]
    fn a_mis_segmented_run_refuses_rather_than_parsing_what_it_can() {
        assert_eq!(parse_amount("1.2.3"), None);
        assert_eq!(parse_amount(""), None);
        assert_eq!(parse_amount("?"), None);
    }
}
