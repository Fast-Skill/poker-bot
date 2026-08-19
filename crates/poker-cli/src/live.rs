//! Acting on the live table.
//!
//! Everything else in this project reads. This is the part that writes, and the
//! asymmetry matters: a misread costs a reading, a miscue costs a stack. So the
//! rule here is that the bot does nothing unless it can show it should, and
//! every gate below exists because of a specific way that could go wrong.
//!
//! # Nothing happens on one frame
//!
//! The client animates — chips slide in, the button slides round, the table
//! re-lays itself out when a player joins. A single frame caught during any of
//! that can be internally inconsistent, and one of the stored fixtures shows a
//! seat's stack twice in two places because of it. So a decision needs two
//! captures a moment apart that agree, which is a cheaper test than recognising
//! each animation and does not need to know which ones exist.
//!
//! # Clicking is not acting
//!
//! `SendInput` returning success means Windows accepted the event, not that the
//! client did anything with it. The two are indistinguishable from the calling
//! side, so a click is only believed once the action row has gone away.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use poker_vision::{
    Frame, GlyphTemplates, HeroTemplates, HeroThresholds, TableView, Templates, TextThresholds,
    Thresholds,
};
use poker_win::Window;

/// What the bot decided to do with its turn.
///
/// Only `Fold` is reachable today: the loop that drives this is deliberately
/// limited to folding until a decision engine is wired to it, since folding is
/// the one choice that cannot lose more than what is already committed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum Choice {
    Fold,
    /// Check when there is nothing to call, call when there is.
    Passive,
    /// Bet when there is nothing to call, raise when there is.
    Aggressive,
}

impl Choice {
    pub fn name(&self) -> &'static str {
        match self {
            Choice::Fold => "fold",
            Choice::Passive => "check/call",
            Choice::Aggressive => "bet/raise",
        }
    }
}

/// Why the bot declined to act on a frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Held {
    /// The kill switch file exists.
    KillSwitch,
    /// The loss limit for this session has been reached.
    StopLoss,
    /// The window could not be captured, or came back blank.
    NoPicture,
    /// The two readings disagreed, so something was moving.
    NotSettled,
    /// The client is not asking the hero to act.
    NotOurTurn,
    /// The reading was incomplete — a refused figure, or unread cards.
    NotConfident,
    /// The button the choice needs is not on screen.
    NoSuchButton(Choice),
}

impl Held {
    pub fn explain(&self) -> String {
        match self {
            Held::KillSwitch => "the kill switch file is present".into(),
            Held::StopLoss => "the session loss limit has been reached".into(),
            Held::NoPicture => "the window could not be captured".into(),
            Held::NotSettled => "two readings disagreed, so the table was still moving".into(),
            Held::NotOurTurn => "the client is not asking us to act".into(),
            Held::NotConfident => {
                "the reading was incomplete - a figure was refused, or the hole cards \
                 did not both read"
                    .into()
            }
            Held::NoSuchButton(choice) => format!("there is no {} button on screen", choice.name()),
        }
    }
}

/// The limits a session runs under.
#[derive(Debug, Clone)]
pub struct Safety {
    /// While this file exists, nothing is clicked. Deleting and recreating it
    /// is the fastest way to stop a running bot from another window.
    pub kill_switch: PathBuf,
    /// Stop after losing this many big blinds against the starting stack.
    ///
    /// A limit that only ever stops the bot is the right shape for this: if the
    /// reader is wrong in some way the tests did not cover, the loss is what
    /// shows it, and it should show up as a halt rather than as a slow bleed.
    pub stop_loss_bb: f64,
    /// Stop after this many actions, whatever else happens.
    pub max_actions: usize,
}

impl Default for Safety {
    fn default() -> Safety {
        Safety {
            kill_switch: PathBuf::from("STOP"),
            stop_loss_bb: 200.0,
            max_actions: 500,
        }
    }
}

impl Safety {
    /// Whether the session must stop, given where the stack started and stands.
    pub fn breached(&self, started: Option<f64>, now: Option<f64>, actions: usize) -> Option<Held> {
        if self.kill_switch.exists() {
            return Some(Held::KillSwitch);
        }
        if actions >= self.max_actions {
            return Some(Held::StopLoss);
        }
        match (started, now) {
            (Some(started), Some(now)) if started - now >= self.stop_loss_bb => {
                Some(Held::StopLoss)
            }
            _ => None,
        }
    }
}

/// A live table the bot can read and act on.
pub struct Session {
    window: Window,
    cards: Templates,
    glyphs: GlyphTemplates,
    hero: HeroTemplates,
    pub safety: Safety,
    /// Where to keep frames the reader could not fully read.
    ///
    /// Only those are worth keeping. A frame the bot understood teaches it
    /// nothing, and at four and a half megabytes raw, keeping every frame of a
    /// half-hour session would cost gigabytes to say so.
    pub keep_unread: Option<PathBuf>,
    started_with: Option<f64>,
    actions: usize,
    kept: usize,
}

/// How long to leave between the two captures that must agree.
///
/// Long enough that an animation in flight moves visibly between them, short
/// enough to fit comfortably inside the client's action timer.
const SETTLE: Duration = Duration::from_millis(220);

/// How long to give the client to respond to a click before checking.
const RESPOND: Duration = Duration::from_millis(700);

impl Session {
    pub fn new(
        window: Window,
        cards: Templates,
        glyphs: GlyphTemplates,
        hero: HeroTemplates,
        safety: Safety,
    ) -> Session {
        Session {
            window,
            cards,
            glyphs,
            hero,
            safety,
            keep_unread: None,
            started_with: None,
            actions: 0,
            kept: 0,
        }
    }

    pub fn frames_kept(&self) -> usize {
        self.kept
    }

    pub fn actions_taken(&self) -> usize {
        self.actions
    }

    /// Reads the table once.
    pub fn look(&self) -> Option<TableView> {
        self.look_keeping(false).map(|(view, _)| view)
    }

    /// Reads the table, optionally keeping the frame if it could not be read.
    fn look_keeping(&self, may_keep: bool) -> Option<(TableView, bool)> {
        let capture = self.window.capture()?;
        if capture.is_blank() {
            return None;
        }
        let frame = Frame::new(capture.width, capture.height, &capture.rgb);
        let view = TableView::read(
            &frame,
            &self.cards,
            &self.glyphs,
            &self.hero,
            Thresholds::default(),
            TextThresholds::default(),
        );

        // The frames worth keeping are the ones where the hero plainly holds
        // cards and the reader could not name them — which is exactly what a
        // rank with no template looks like from the outside.
        let unread = view.hero().is_some_and(|h| h.in_hand) && view.hole.len() != 2;
        let keep = may_keep && unread && self.keep_unread.is_some();
        if keep {
            if let Some(dir) = &self.keep_unread {
                let _ = std::fs::create_dir_all(dir);
                let name = format!("unread-{:04}-{}.rgb", self.kept, self.actions);
                let mut bytes = Vec::with_capacity(8 + capture.rgb.len());
                bytes.extend_from_slice(&(capture.width as u32).to_le_bytes());
                bytes.extend_from_slice(&(capture.height as u32).to_le_bytes());
                bytes.extend_from_slice(&capture.rgb);
                let _ = std::fs::write(dir.join(name), bytes);
            }
        }
        Some((view, keep))
    }

    /// Reads the table twice and returns the reading only if both agree.
    pub fn look_settled(&self) -> Result<TableView, Held> {
        let first = self.look().ok_or(Held::NoPicture)?;
        std::thread::sleep(SETTLE);
        let second = self.look().ok_or(Held::NoPicture)?;
        if first.agrees_with(&second) {
            Ok(second)
        } else {
            Err(Held::NotSettled)
        }
    }

    /// Reads the table, and reports what may be done with the reading.
    ///
    /// Returns the settled view whether or not it can be acted on, because a
    /// caller watching the bot work wants to see the table it is looking at
    /// even on the frames it declines.
    pub fn assess(&mut self) -> (Option<TableView>, Option<Held>) {
        // One capture of the pair may be kept, so a frame the reader could not
        // name is not lost just because it was never the hero's turn.
        if let Some((_, kept)) = self.look_keeping(true) {
            if kept {
                self.kept += 1;
            }
        }

        let view = match self.look_settled() {
            Ok(view) => view,
            Err(held) => return (None, Some(held)),
        };

        let stack = view.hero().and_then(|h| h.stack);
        if self.started_with.is_none() {
            self.started_with = stack;
        }
        if let Some(held) = self.safety.breached(self.started_with, stack, self.actions) {
            return (Some(view), Some(held));
        }
        if !view.hero_to_act() {
            return (Some(view), Some(Held::NotOurTurn));
        }
        if !view.is_actionable() {
            return (Some(view), Some(Held::NotConfident));
        }
        (Some(view), None)
    }

    /// Carries out a choice and checks that the client took it.
    ///
    /// The view must be one this session already judged actionable; passing a
    /// stale or unchecked one is a programming error rather than a table
    /// condition, so it is asserted rather than reported.
    pub fn act(&mut self, view: &TableView, choice: Choice) -> Result<Duration, Held> {
        debug_assert!(
            view.is_actionable(),
            "act was called on a view that was never cleared to act on"
        );
        let panel = view.action.as_ref().ok_or(Held::NotOurTurn)?;
        let button = match choice {
            Choice::Fold => panel.fold(),
            Choice::Passive => panel.passive(),
            Choice::Aggressive => panel.aggressive(),
        }
        .ok_or(Held::NoSuchButton(choice))?;

        let (x, y) = button.centre();
        let began = Instant::now();
        self.window.focus();
        if !self.window.click_at(x, y) {
            return Err(Held::NoPicture);
        }
        self.actions += 1;

        // Windows accepting the event says nothing about the client acting on
        // it. The action row going away does.
        std::thread::sleep(RESPOND);
        match self.look() {
            Some(after) if after.hero_to_act() => Err(Held::NotSettled),
            _ => Ok(began.elapsed()),
        }
    }

    pub fn window_title(&self) -> String {
        self.window.title()
    }
}

/// Loads the templates a session needs.
pub fn templates(
    cards: &Path,
    glyphs: &Path,
    hero: &Path,
) -> Result<(Templates, GlyphTemplates, HeroTemplates), String> {
    let card = Templates::load(cards).map_err(|e| format!("{}: {e}", cards.display()))?;
    let glyph = GlyphTemplates::load(glyphs).map_err(|e| format!("{}: {e}", glyphs.display()))?;
    let hero = HeroTemplates::load(hero).map_err(|e| format!("{}: {e}", hero.display()))?;
    let _ = HeroThresholds::default();
    Ok((card, glyph, hero))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn safety() -> Safety {
        Safety {
            kill_switch: PathBuf::from("a-file-that-does-not-exist-here"),
            stop_loss_bb: 100.0,
            max_actions: 10,
        }
    }

    #[test]
    fn a_session_within_its_limits_is_not_stopped() {
        assert_eq!(safety().breached(Some(200.0), Some(150.0), 3), None);
    }

    #[test]
    fn losing_past_the_limit_stops_the_session() {
        // Down exactly the limit counts: the limit is a ceiling on losses, not
        // a level to keep playing at.
        assert_eq!(
            safety().breached(Some(200.0), Some(100.0), 3),
            Some(Held::StopLoss)
        );
        assert_eq!(
            safety().breached(Some(200.0), Some(99.0), 3),
            Some(Held::StopLoss)
        );
    }

    #[test]
    fn winning_never_stops_the_session() {
        assert_eq!(safety().breached(Some(100.0), Some(400.0), 3), None);
    }

    #[test]
    fn the_action_count_stops_the_session_on_its_own() {
        assert_eq!(safety().breached(None, None, 10), Some(Held::StopLoss));
    }

    #[test]
    fn an_unreadable_stack_does_not_look_like_a_loss() {
        // A refused stack reading must not be mistaken for the stack being
        // zero, which would trip the limit and halt a healthy session.
        assert_eq!(safety().breached(Some(200.0), None, 0), None);
        assert_eq!(safety().breached(None, Some(50.0), 0), None);
    }

    #[test]
    fn the_kill_switch_stops_everything_regardless() {
        let path = std::env::temp_dir().join("poker-live-kill-switch-test");
        std::fs::write(&path, b"stop").expect("write");
        let safety = Safety {
            kill_switch: path.clone(),
            ..safety()
        };
        assert_eq!(
            safety.breached(Some(100.0), Some(500.0), 0),
            Some(Held::KillSwitch),
            "even a winning session with no actions taken"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn every_reason_for_holding_off_can_be_explained() {
        for held in [
            Held::KillSwitch,
            Held::StopLoss,
            Held::NoPicture,
            Held::NotSettled,
            Held::NotOurTurn,
            Held::NotConfident,
            Held::NoSuchButton(Choice::Aggressive),
        ] {
            assert!(!held.explain().is_empty(), "{held:?}");
        }
    }
}
