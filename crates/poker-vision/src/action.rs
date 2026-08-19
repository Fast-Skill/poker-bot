//! Finding the buttons the bot has to press.
//!
//! This is the only part of the reader whose output becomes a click, so a
//! mistake here does not produce a bad reading — it produces a bad *action*,
//! at somebody else's turn, on a hand the bot never saw.
//!
//! # Two panels that look alike
//!
//! The client draws its action buttons in one row of identical red rectangles,
//! and two different panels use that row:
//!
//! ```text
//! Fold          | Call 4 BB | Raise to 7 BB
//! Check / Fold  | Check     | Bet 2 BB
//! ```
//!
//! The first is the hero's turn. The second appears while somebody *else* is
//! still acting, and arms an action for when the turn does come round —
//! confirmed against the client rather than inferred, because both states glow
//! the hero's name plate and run a countdown bar under it, so the table itself
//! gives no clue. Pressing its left button would commit the hero to folding a
//! hand nobody has looked at yet.
//!
//! Nothing about the rectangles distinguishes them — same size, same colour,
//! same place. The labels do: `Fold` spans 39 pixels of white and
//! `Check / Fold` spans 116, with nothing in between across every capture. So
//! the panel is identified by measuring its left label, and
//! [`ActionPanel::offers_plain_fold`] is what the caller must gate on.

use crate::{components, Bounds, Frame};

/// One button in the action row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActionButton {
    /// Top-left in frame coordinates.
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
    /// How many pixels across the white label runs, which is what tells
    /// `Fold` from `Check / Fold`.
    pub label_width: usize,
}

impl ActionButton {
    /// Where to click, in frame coordinates.
    pub fn centre(&self) -> (usize, usize) {
        (self.x + self.width / 2, self.y + self.height / 2)
    }
}

/// The row of buttons at the bottom of the table, if it is showing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionPanel {
    /// Left to right. Two buttons when there is nothing to raise.
    pub buttons: Vec<ActionButton>,
    /// The editable field holding the raise amount, when one is showing.
    ///
    /// The client offers a few preset sizes beside it, and a solved strategy
    /// means a specific amount rather than the nearest preset. This is where
    /// that amount is written, and — just as importantly — read back before
    /// anything is committed.
    pub amount_box: Option<ActionButton>,
}

/// Where the amount field sits, measured at a 1430x1040 table.
///
/// Found by shape rather than position would be nicer, but it is a plain dark
/// rectangle among other dark rectangles; what distinguishes it is being the
/// small one, on the right, in the row above the buttons.
const AMOUNT_BOX: (usize, usize, usize, usize) = (1184, 902, 1299, 935);

/// The widest a plain `Fold` label runs, and the narrowest `Check / Fold`.
///
/// Measured across the captures at 39 and 116 pixels, so this sits in a gap of
/// nearly threefold rather than on a boundary.
const PLAIN_FOLD_LABEL: usize = 70;

impl ActionPanel {
    /// Whether the leftmost button is a plain `Fold`.
    ///
    /// This is the test for "the hero is being asked to act now". A panel whose
    /// left button reads `Check / Fold` belongs to somebody else's turn and is
    /// offering to arm an action for later, so clicking into it would decide a
    /// hand in advance of seeing it. Ignoring it costs nothing: the hero's own
    /// turn comes round afterwards with the live panel.
    pub fn offers_plain_fold(&self) -> bool {
        self.buttons
            .first()
            .is_some_and(|b| b.label_width <= PLAIN_FOLD_LABEL)
    }

    /// The fold button, when the panel is a live one.
    pub fn fold(&self) -> Option<ActionButton> {
        self.offers_plain_fold().then(|| self.buttons[0])
    }

    /// The middle button — `Call` when facing a bet, `Check` when not.
    pub fn passive(&self) -> Option<ActionButton> {
        self.buttons.get(1).copied()
    }

    /// The rightmost button — `Raise to` or `Bet`. Absent when the hero is
    /// already all-in for the call and there is nothing left to raise with.
    pub fn aggressive(&self) -> Option<ActionButton> {
        (self.buttons.len() >= 3).then(|| self.buttons[2])
    }
}

/// Finds the action row.
///
/// Returns `None` when the row is not showing, which is most of the time — the
/// hero is only asked to act on a fraction of frames.
pub fn read_action_panel(frame: &Frame) -> Option<ActionPanel> {
    /// Buttons are this size at a 1430x1040 table, and the row sits along the
    /// bottom. Both bounds are generous: what matters is excluding the bet-size
    /// presets above the row, which are far smaller.
    const MIN: (usize, usize) = (120, 40);
    const MAX: (usize, usize) = (260, 130);

    let (w, h) = (frame.width, frame.height);
    let top = h.saturating_sub(h / 5);

    let mut red = vec![false; w * h];
    for y in top..h {
        for x in 0..w {
            let (r, g, b) = frame.pixel(x, y);
            let (r, g, b) = (r as i16, g as i16, b as i16);
            red[y * w + x] = r > 120 && r - g > 45 && r - b > 45;
        }
    }

    let mut buttons: Vec<ActionButton> = components(&red, w, h)
        .into_iter()
        .filter(|b| {
            (MIN.0..=MAX.0).contains(&b.width()) && (MIN.1..=MAX.1).contains(&b.height())
        })
        .map(|b| ActionButton {
            x: b.x0,
            y: b.y0,
            width: b.width(),
            height: b.height(),
            label_width: label_width(frame, b),
        })
        .collect();

    if buttons.is_empty() {
        return None;
    }
    buttons.sort_by_key(|b| b.x);

    // Only meaningful when there is something to raise; with no raise button
    // the client draws no amount field either.
    let amount_box = (buttons.len() >= 3).then_some(ActionButton {
        x: AMOUNT_BOX.0,
        y: AMOUNT_BOX.1,
        width: AMOUNT_BOX.2 - AMOUNT_BOX.0,
        height: AMOUNT_BOX.3 - AMOUNT_BOX.1,
        label_width: 0,
    });
    Some(ActionPanel {
        buttons,
        amount_box,
    })
}

/// Reads the amount currently in the raise field.
///
/// This is the check that makes typing an amount safe. Windows accepting the
/// keystrokes says nothing about what the field now holds, and committing a
/// raise without looking is how a bot bets 23 into a pot it meant to bet 2 —
/// so the caller writes, reads back, and only then presses the button.
pub fn read_amount(
    frame: &Frame,
    panel: &ActionPanel,
    glyphs: &crate::GlyphTemplates,
) -> Option<f64> {
    let box_ = panel.amount_box?;
    crate::read_number_in(
        frame,
        glyphs,
        crate::TextThresholds::default(),
        crate::Ink::White,
        (
            box_.x,
            box_.y,
            box_.x + box_.width,
            box_.y + box_.height,
        ),
    )?
    .value
}

/// How far the white label runs across a button.
///
/// Only the upper part of the button is measured. The lower line carries the
/// amount — `Call` sits above `4 BB` — and its width says nothing about which
/// action the button is.
fn label_width(frame: &Frame, bounds: Bounds) -> usize {
    let top = bounds.y0 + bounds.height() / 5;
    let bottom = bounds.y0 + bounds.height() / 2;
    let mut left = usize::MAX;
    let mut right = 0usize;
    for y in top..=bottom.min(bounds.y1) {
        for x in bounds.x0..=bounds.x1 {
            let (r, g, b) = frame.pixel(x, y);
            if r > 200 && g > 200 && b > 200 {
                left = left.min(x);
                right = right.max(x);
            }
        }
    }
    if left > right {
        0
    } else {
        right - left + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn frame_bytes(name: &str) -> (usize, usize, Vec<u8>) {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/frames")
            .join(name);
        let raw = std::fs::read(path).expect("frame should exist");
        let w = u32::from_le_bytes(raw[0..4].try_into().expect("header")) as usize;
        let h = u32::from_le_bytes(raw[4..8].try_into().expect("header")) as usize;
        (w, h, raw[8..].to_vec())
    }

    /// The amount field must read back what the client is showing.
    ///
    /// This is the check that makes writing an amount safe: the bot types,
    /// reads, and only presses the button when the two agree.
    #[test]
    fn the_raise_amount_can_be_read_back_from_its_box() {
        let (w, h, bytes) = frame_bytes("raise-ui.rgb");
        let frame = Frame::new(w, h, &bytes);
        let panel = read_action_panel(&frame).expect("the buttons are showing");
        assert_eq!(panel.buttons.len(), 3, "fold, call and raise");
        assert!(panel.amount_box.is_some());

        let glyphs = crate::GlyphTemplates::load(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../data/digit_templates.bin"),
        )
        .expect("glyph templates");
        // Checked against the frame: the field reads 9.5BB.
        assert_eq!(read_amount(&frame, &panel, &glyphs), Some(9.5));
    }

    /// A known gap, recorded rather than papered over.
    ///
    /// The amount box is drawn in its own twelve-pixel face, and the templates
    /// for it were harvested from amounts the client happened to display —
    /// which never included a nought, a three or an eight. An amount using one
    /// of those reads as nothing rather than as something plausible, so the bot
    /// declines to commit a raise it cannot verify.
    ///
    /// It closes the same way the hero's cards did: by collecting frames where
    /// the reading fails. Typing an amount and photographing the result would
    /// do it in one pass, since the bot chooses what to type.
    #[test]
    fn an_amount_using_a_digit_with_no_template_is_refused() {
        let (w, h, bytes) = frame_bytes("raise-ui-unreadable.rgb");
        let frame = Frame::new(w, h, &bytes);
        let panel = read_action_panel(&frame).expect("the buttons are showing");
        let glyphs = crate::GlyphTemplates::load(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../data/digit_templates.bin"),
        )
        .expect("glyph templates");
        // This frame's box reads 18.7BB, and there is no template for an eight.
        // Reading it as 1.7 or 17 would both be plausible and both be wrong.
        assert_eq!(read_amount(&frame, &panel, &glyphs), None);
    }

    #[test]
    fn a_panel_with_nothing_to_raise_has_no_amount_field() {
        let (w, h, bytes) = frame_bytes("20260818-103636-028.rgb");
        let frame = Frame::new(w, h, &bytes);
        let panel = read_action_panel(&frame).expect("the buttons are showing");
        assert_eq!(panel.buttons.len(), 2);
        assert_eq!(panel.amount_box, None, "nothing to size");
    }

    #[test]
    fn a_table_with_no_decision_pending_shows_no_buttons() {
        let (w, h, bytes) = frame_bytes("20260818-104053-025.rgb");
        let frame = Frame::new(w, h, &bytes);
        assert_eq!(read_action_panel(&frame), None);
    }

    #[test]
    fn a_live_turn_offers_fold_call_and_raise() {
        let (w, h, bytes) = frame_bytes("20260818-103636-015.rgb");
        let frame = Frame::new(w, h, &bytes);
        let panel = read_action_panel(&frame).expect("the buttons are showing");
        assert_eq!(panel.buttons.len(), 3);
        assert!(panel.offers_plain_fold(), "{panel:?}");
        assert_eq!(panel.fold().map(|b| b.centre()), Some((953, 988)));
        assert!(panel.passive().is_some());
        assert!(panel.aggressive().is_some());
    }

    #[test]
    fn a_turn_with_nothing_left_to_raise_offers_only_two() {
        // The hero can cover the call and no more, so the client drops the
        // raise button rather than greying it out.
        let (w, h, bytes) = frame_bytes("20260818-103636-028.rgb");
        let frame = Frame::new(w, h, &bytes);
        let panel = read_action_panel(&frame).expect("the buttons are showing");
        assert_eq!(panel.buttons.len(), 2);
        assert!(panel.offers_plain_fold());
        assert!(panel.passive().is_some());
        assert_eq!(panel.aggressive(), None, "there is nothing to raise with");
    }

    /// The panel this module exists to refuse.
    #[test]
    fn a_panel_offering_to_arm_an_action_in_advance_is_not_a_turn() {
        // This panel is drawn while another player is acting. `Check / Fold`
        // commits the hero to folding whenever the turn does arrive, and its
        // rectangle is identical to a live `Fold` — same size, colour and
        // position — so only the label separates them.
        let (w, h, bytes) = frame_bytes("20260818-104236-001.rgb");
        let frame = Frame::new(w, h, &bytes);
        let panel = read_action_panel(&frame).expect("the buttons are showing");
        assert_eq!(panel.buttons.len(), 3);
        assert!(
            !panel.offers_plain_fold(),
            "left label spans {} pixels",
            panel.buttons[0].label_width
        );
        assert_eq!(panel.fold(), None, "nothing here is safe to click");
    }

    #[test]
    fn the_sit_out_dialog_is_not_mistaken_for_an_action_row() {
        // The dialog has its own red button, "I'm Back", and clicking it
        // believing it were a fold would sit the bot back into a hand it has
        // not read.
        let (w, h, bytes) = frame_bytes("20260818-104742-014.rgb");
        let frame = Frame::new(w, h, &bytes);
        match read_action_panel(&frame) {
            None => {}
            Some(panel) => assert!(
                !panel.offers_plain_fold(),
                "the dialog must not present itself as a hero turn: {panel:?}"
            ),
        }
    }
}
