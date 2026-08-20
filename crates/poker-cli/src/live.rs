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
    ActionPanel, Frame, GlyphTemplates, HeroTemplates, HeroThresholds, TableView, Templates,
    TextThresholds, Thresholds,
};
use poker_win::Window;

/// What the bot decided to do with its turn.
///
/// Only `Fold` is reachable today: the loop that drives this is deliberately
/// limited to folding until a decision engine is wired to it, since folding is
/// the one choice that cannot lose more than what is already committed.
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)]
pub enum Choice {
    Fold,
    /// Check when there is nothing to call, call when there is.
    Passive,
    /// Bet or raise to a specific total, in big blinds.
    ///
    /// The size is part of the decision, not a detail of carrying it out. A
    /// blueprint that chose to raise chose an amount, and putting in a
    /// different one plays a strategy nobody solved for.
    Aggressive { to_blinds: f64 },
}

impl Choice {
    pub fn name(&self) -> &'static str {
        match self {
            Choice::Fold => "fold",
            Choice::Passive => "check/call",
            Choice::Aggressive { .. } => "bet/raise",
        }
    }
}

/// Why the bot declined to act on a frame.
#[derive(Debug, Clone, PartialEq)]
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
    /// The client went on asking the hero to act long after the click.
    ///
    /// Distinct from [`Held::NotSettled`], which it used to be reported as, and
    /// which said the table was moving — the opposite of what had happened, and
    /// misleading in a log. This means the action row was still there when the
    /// waiting ran out.
    NotTaken,
    /// The client has sat the hero out for not acting in time.
    SatOut,
    /// The money on the table did not add up to the pot.
    NotAdding,
    /// The reading was incomplete — a refused figure, or unread cards.
    NotConfident,
    /// The button the choice needs is not on screen.
    NoSuchButton(Choice),
    /// The raise field would not take the amount, or did not read back as it.
    ///
    /// Carries what was wanted and what the field showed, because the two
    /// together say whether the write missed or the reading did.
    WrongAmount { wanted: f64, showing: Option<f64> },
}

impl Held {
    pub fn explain(&self) -> String {
        match self {
            Held::KillSwitch => "the kill switch file is present".into(),
            Held::StopLoss => "the session loss limit has been reached".into(),
            Held::NoPicture => "the window could not be captured".into(),
            Held::NotSettled => "two readings disagreed, so the table was still moving".into(),
            Held::NotAdding => "the chips on the table did not add up to the pot".into(),
            Held::NotTaken => "the action row was still up after the click, so nothing took".into(),
            Held::NotOurTurn => "the client is not asking us to act".into(),
            Held::SatOut => {
                "the client has sat us out; nothing will be dealt until we sit back in".into()
            }
            Held::NotConfident => {
                "a figure the decision needs did not read - our two cards, the pot, the amount to call, or our own stack"
                    .into()
            }
            Held::NoSuchButton(choice) => format!("there is no {} button on screen", choice.name()),
            Held::WrongAmount { wanted, showing } => match showing {
                Some(showing) => format!(
                    "the raise field reads {showing} where {wanted} was wanted, so nothing was committed"
                ),
                None => format!(
                    "the raise field could not be read after writing {wanted} to it, so nothing was committed"
                ),
            },
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
    /// Where to keep a picture of every frame where the hero is asked to act.
    ///
    /// Different from `keep_unread`, which keeps the frames the reader gave up
    /// on. These are the ones it read confidently — and confidence is exactly
    /// the problem when what it is confident about is wrong. Whether a seat
    /// still holds cards is judged from the picture and never refused, so a
    /// seat wrongly counted as live produces a clean-looking reading of a table
    /// that is not there, and no amount of logging catches it. Only the picture
    /// does.
    pub keep_turns: Option<PathBuf>,
    /// Where to keep frames the reader could not fully read.
    ///
    /// Only those are worth keeping. A frame the bot understood teaches it
    /// nothing, and at four and a half megabytes raw, keeping every frame of a
    /// half-hour session would cost gigabytes to say so.
    pub keep_unread: Option<PathBuf>,
    started_with: Option<f64>,
    actions: usize,
    kept: usize,
    /// Raises abandoned because the size would not go into the field.
    ///
    /// Counted because it is the difference between the bot playing the
    /// strategy that was measured and playing something meeker, and a session
    /// where it happens often is not the bot anybody benchmarked.
    retreats: usize,
    last_retreat: Option<Held>,
    turns: usize,
}

/// How long the hero may be to act before the bot stops holding out for a
/// reading it trusts.
///
/// # Why there has to be one
///
/// Every other refusal in this program is free: decline the frame, look again,
/// nothing is lost. Once the client is asking the hero to act that stops being
/// true. The clock is running, and running out of it is not a neutral outcome —
/// the client folds the hand and sits the hero out, which then makes every
/// subsequent reading fail correctly while the seat is lost.
///
/// So this is the point where declining becomes more expensive than acting on
/// an imperfect reading. It is set well inside the client's own timer, which
/// runs around fifteen seconds, leaving room for the fallback to be taken and
/// confirmed.
pub const DEADLINE: Duration = Duration::from_secs(6);

/// The safest thing that can be done with a reading not good enough to trust.
///
/// Checking when nothing is owed is free and cannot be a mistake whatever the
/// table holds. Otherwise folding gives up the hand, which is a bounded loss,
/// where letting the clock run out gives up the seat.
///
/// Deliberately not "call because it is cheap": the reason for being here is
/// that the reading cannot be trusted, and a price read off an untrusted
/// reading is no basis for putting money in.
pub fn last_resort(view: &TableView) -> Choice {
    match view.to_call() {
        Some(owed) if owed <= 0.0 => Choice::Passive,
        _ => Choice::Fold,
    }
}

/// How long to leave between the two captures that must agree.
///
/// Long enough that an animation in flight moves visibly between them, short
/// enough to fit comfortably inside the client's action timer.
const SETTLE: Duration = Duration::from_millis(220);

/// How long to keep watching for the action row to clear after a click.
///
/// The client takes a variable time to take an action down, and the whole of
/// that variability used to be read as failure.
const TOOK: Duration = Duration::from_millis(2_500);

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
            keep_turns: None,
            keep_unread: None,
            started_with: None,
            actions: 0,
            kept: 0,
            retreats: 0,
            last_retreat: None,
            turns: 0,
        }
    }

    pub fn frames_kept(&self) -> usize {
        self.kept
    }

    pub fn actions_taken(&self) -> usize {
        self.actions
    }

    /// How many raises were given up on because the size would not take, and
    /// what stopped the last one.
    pub fn retreats(&self) -> (usize, Option<Held>) {
        (self.retreats, self.last_retreat.clone())
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
    /// Sits back in if the client has sat the hero out.
    ///
    /// Worth handling rather than reporting, because being sat out is silent
    /// and self-sustaining: the table dims, every reading is then correctly
    /// refused, and the bot waits politely while the client counts down to
    /// removing it from the table. Returns whether the dialog was there.
    pub fn recover_from_sit_out(&self) -> bool {
        let Some(capture) = self.window.capture() else {
            return false;
        };
        let frame = Frame::new(capture.width, capture.height, &capture.rgb);
        let Some(dialog) = poker_vision::read_sit_out(&frame) else {
            return false;
        };
        let (x, y) = dialog.resume.centre();
        self.window.focus();
        std::thread::sleep(Duration::from_millis(150));
        self.window.click_at(x, y);
        std::thread::sleep(RESPOND);
        true
    }

    /// Whether the client is showing the sit-out dialog.
    pub fn is_sat_out(&self) -> bool {
        self.window
            .capture()
            .map(|capture| {
                let frame = Frame::new(capture.width, capture.height, &capture.rgb);
                poker_vision::read_sit_out(&frame).is_some()
            })
            .unwrap_or(false)
    }

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
            // A table that will not settle may simply be covered by the sit-out
            // dialog, which is worth naming rather than reporting as a failure
            // to read.
            Err(held) => {
                if self.is_sat_out() {
                    return (None, Some(Held::SatOut));
                }
                return (None, Some(held));
            }
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
        self.keep_turn(&view);
        // Named apart, because they want different work. A hole card that did
        // not read is the reader failing at its main job; a table whose money
        // does not add up is almost always a frame caught mid-animation and
        // will come good on its own.
        if !view.reads_what_a_decision_needs() {
            return (Some(view), Some(Held::NotConfident));
        }
        if !view.is_consistent() {
            return (Some(view), Some(Held::NotAdding));
        }
        (Some(view), None)
    }

    /// Saves a picture of the table on the hero's turn, if asked to.
    ///
    /// Named by what the reading claims, so the file itself says what to check:
    /// a frame called `turn-0003-3of7.png` showing four players with cards is
    /// the whole bug report.
    fn keep_turn(&mut self, view: &TableView) {
        let Some(dir) = self.keep_turns.clone() else {
            return;
        };
        let Some(capture) = self.window.capture() else {
            return;
        };
        if capture.is_blank() {
            return;
        }
        let _ = std::fs::create_dir_all(&dir);
        let name = format!(
            "turn-{:04}-{}of{}.png",
            self.turns,
            view.active(),
            view.occupied()
        );
        let bytes = crate::png::encode(capture.width, capture.height, &capture.rgb);
        if std::fs::write(dir.join(name), bytes).is_ok() {
            self.turns += 1;
        }
    }

    /// How many of the hero's turns have been pictured.
    pub fn turns_kept(&self) -> usize {
        self.turns
    }

    /// Carries out a choice and checks that the client took it.
    ///
    /// The view must be one this session already judged actionable; passing a
    /// stale or unchecked one is a programming error rather than a table
    /// condition, so it is asserted rather than reported.
    pub fn act(&mut self, view: &TableView, choice: Choice) -> Result<Duration, Held> {
        // Only that the client is asking. Whether the reading is good enough to
        // act on is the caller's judgement, because it is not always a free
        // choice: past [`DEADLINE`] the caller deliberately acts on a reading
        // that failed its checks, rather than lose the seat to the clock.
        debug_assert!(
            view.hero_to_act(),
            "act was called when the client was not asking the hero to act"
        );
        let panel = view.action.as_ref().ok_or(Held::NotOurTurn)?;

        // A raise has to be sized before it is made, and the size has to be
        // checked before it is committed. Windows accepting the keystrokes says
        // nothing about what the field holds.
        //
        // If the field will not take it, the raise is abandoned — but the turn
        // is not. Doing nothing here was what cost a seat: the size failed to
        // read back, `act` returned an error, and the bot went back to watching
        // while the client counted down. Calling is the right retreat, and
        // safely so: a strategy that chose to raise to some amount had already
        // accepted putting in at least a call, which is strictly less. Checking
        // costs nothing at all when nothing is owed.
        let mut choice = choice;
        if let Choice::Aggressive { to_blinds } = choice {
            if let Err(why) = self.set_amount(panel, to_blinds) {
                self.retreats += 1;
                self.last_retreat = Some(why);
                choice = Choice::Passive;
            }
        }

        let button = match choice {
            Choice::Fold => panel.fold(),
            Choice::Passive => panel.passive(),
            Choice::Aggressive { .. } => panel.aggressive(),
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
        //
        // Watched for, rather than glanced at once. A single look a fixed
        // moment after the click reported perfectly good folds as failures: the
        // client takes a variable time to clear the row, and a frame caught
        // before it does looks exactly like a click that never landed. The
        // difference showed up as a log full of "fold did not take" from a
        // session in which every fold did take.
        //
        // Reading nothing at all is not evidence either way — a frame arriving
        // mid-animation is unreadable whether or not the click landed — so it
        // waits rather than concluding.
        let deadline = Instant::now() + TOOK;
        loop {
            std::thread::sleep(Duration::from_millis(200));
            if let Some(after) = self.look() {
                if !after.hero_to_act() {
                    return Ok(began.elapsed());
                }
            }
            if Instant::now() >= deadline {
                return Err(Held::NotTaken);
            }
        }
    }

    /// Writes a raise size into the client's field and confirms it took.
    ///
    /// Nothing is pressed here. If the field will not hold the amount this
    /// returns an error and the caller commits nothing — better to miss a raise
    /// than to make one of the wrong size, which is a decision nobody took.
    fn set_amount(&self, panel: &ActionPanel, to_blinds: f64) -> Result<(), Held> {
        let field = panel
            .amount_box
            .ok_or(Held::NoSuchButton(Choice::Aggressive { to_blinds }))?;
        // The client shows one decimal place, so asking for more is asking for
        // a number it cannot display and will not read back.
        let wanted = (to_blinds * 10.0).round() / 10.0;

        let (x, y) = field.centre();
        self.window.focus();
        std::thread::sleep(Duration::from_millis(120));
        if !self.window.click_at(x, y) {
            return Err(Held::NoPicture);
        }
        std::thread::sleep(Duration::from_millis(180));
        self.window.type_text(&format!("{wanted}"));
        std::thread::sleep(Duration::from_millis(400));

        let capture = self.window.capture().ok_or(Held::NoPicture)?;
        let frame = Frame::new(capture.width, capture.height, &capture.rgb);
        let showing = poker_vision::read_action_panel(&frame)
            .and_then(|fresh| poker_vision::read_amount(&frame, &fresh, &self.glyphs));
        match showing {
            Some(showing) if (showing - wanted).abs() < 0.05 => Ok(()),
            showing => Err(Held::WrongAmount { wanted, showing }),
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
            Held::NoSuchButton(Choice::Aggressive { to_blinds: 9.0 }),
            Held::WrongAmount {
                wanted: 9.0,
                showing: Some(90.0),
            },
            Held::WrongAmount {
                wanted: 9.0,
                showing: None,
            },
        ] {
            assert!(!held.explain().is_empty(), "{held:?}");
        }
    }

    /// A size the client cannot display must not be asked for.
    ///
    /// The field shows one decimal place. Writing 9.25 into it leaves 9.2 or
    /// 9.3 showing, the check then disagrees with what was wanted, and a raise
    /// that was perfectly fine gets abandoned every time.
    #[test]
    fn a_raise_size_is_rounded_to_what_the_client_can_show() {
        let round = |v: f64| (v * 10.0).round() / 10.0;
        assert_eq!(round(9.25), 9.3);
        assert_eq!(round(18.7), 18.7);
        assert_eq!(round(2.0), 2.0);
        assert_eq!(round(0.5), 0.5);
    }
}
