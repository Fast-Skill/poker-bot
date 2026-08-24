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

use crate::bridge;
use poker_core::card::Card;
use poker_core::telemetry::{DecisionRecord, Observer, Source};
use poker_vision::{
    ActionPanel, Frame, GlyphTemplates, HeroTemplates, HeroThresholds, TableView, Templates,
    TextThresholds, Thresholds,
};
use poker_win::{Capture, Window};

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
    /// A figure the decision rests on did not read. Names which.
    NotConfident(&'static str),
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
            Held::NotConfident(missing) => {
                format!("{missing} did not read")
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
    /// Where to record the whole session, frame by frame.
    ///
    /// # Why a transcript and not just failures
    ///
    /// Everything else kept here is kept because something already went wrong:
    /// a frame that would not read, a turn the hero was asked to take. That
    /// finds the faults already suspected and no others. A table the reader
    /// misunderstands *confidently* leaves no trace at all — the seat count
    /// drifting, a folded player still counted in, a situation nobody thought
    /// to check — and those are the ones that quietly change what the bot does.
    ///
    /// So this writes a line for every reading, whether or not anything is
    /// wrong with it, and pictures alongside up to a bound. Read back in order
    /// the lines show what could not be seen in any single frame: a figure that
    /// jumps, a seat that vanishes and returns, a hand whose pot goes backwards.
    pub record: Option<PathBuf>,
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
    frames: usize,
    pictured: usize,
    /// The picture the last settled reading was made from.
    seen: Option<Capture>,
    /// The previous turn's reading, which this turn's is checked against.
    before: Option<TableView>,
    /// When a dialog was last looked for.
    looked_for_dialog: Option<Instant>,
}

/// Writes a hand history a person can read and argue with.
///
/// # Why this is not the console output
///
/// `--explain` already prints every decision, but it prints it between the
/// hundreds of lines the bot emits while waiting for its turn. Reviewing a
/// hundred hands from that means scrolling past several thousand lines of
/// "the client is not asking us to act", and the decisions do not group by
/// hand, so a flop and the river it led to sit a page apart.
///
/// What a reviewer needs is the opposite: one block per hand, every decision
/// in it, the strategy behind each, and the result at the end. Then the
/// question "was that fold right?" has everything beside it — the price, the
/// board, the frequencies the solve actually held — and can be answered
/// without the bot's help.
///
/// This is the deliverable of a decision-quality benchmark. A win rate over a
/// hundred hands says nothing; a hundred readable decisions say a hundred
/// things.
#[derive(Debug)]
pub struct Review {
    to: PathBuf,
    /// The hand being written, recognised by the hero's cards changing.
    hole: Vec<Card>,
    hands: u64,
    decisions: u64,
    /// Decisions that came from a solve rather than the fallback.
    solved: u64,
}

impl Review {
    pub fn new(to: PathBuf) -> Review {
        Review {
            to,
            hole: Vec::new(),
            hands: 0,
            decisions: 0,
            solved: 0,
        }
    }

    /// How much of the session a solve decided, which is one of the
    /// benchmark's own criteria.
    pub fn coverage(&self) -> (u64, u64) {
        (self.solved, self.decisions)
    }

    pub fn hands(&self) -> u64 {
        self.hands
    }

    /// Closes a hand with what it came to.
    pub fn note_result(&mut self, net: f64) {
        if self.hands > 0 {
            self.write(&format!("  result   {net:+.2} bb\n"));
        }
    }

    fn write(&self, text: &str) {
        use std::io::Write;
        if let Some(parent) = self.to.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.to)
        {
            let _ = file.write_all(text.as_bytes());
        }
    }
}

impl Observer for Review {
    fn on_decision(&mut self, record: &DecisionRecord) {
        let cards = |of: &[Card]| {
            of.iter()
                .map(|card| card.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        };

        if self.hole != record.perception.hole {
            self.hole = record.perception.hole.to_vec();
            self.hands += 1;
            self.write(&format!(
                "\nHAND {}  {}\n",
                self.hands,
                cards(&record.perception.hole)
            ));
        }

        self.decisions += 1;
        let (spot, solved) = match &record.source {
            Source::Blueprint { spot, .. } => (spot.clone(), true),
            Source::Fallback { reason } => (format!("HEURISTIC - {reason}"), false),
        };
        self.solved += u64::from(solved);

        let board = cards(&record.perception.board);
        let board = if board.is_empty() {
            "-".to_string()
        } else {
            board
        };
        // Big blinds, since that is how a poker player reads a pot.
        let bb = |chips: u64| chips as f64 / bridge::CHIPS_PER_BB;
        self.write(&format!(
            "  {:<8} board {:<14} pot {:>6.1}  to call {:>5.1}\n           {}\n           {}\n           => {:?}\n",
            format!("{:?}", record.perception.street).to_lowercase(),
            board,
            bb(record.perception.pot),
            bb(record.perception.to_call),
            spot,
            record
                .frequencies
                .iter()
                .map(|(name, share)| format!("{name} {:.0}%", share * 100.0))
                .collect::<Vec<_>>()
                .join("  "),
            record.action,
        ));
    }
}

/// A handle to a [`Review`] that can be given away and still read.
///
/// The agent takes ownership of whatever watches it, so a caller that wants to
/// print the session's coverage at the end needs a second way in.
#[derive(Debug, Clone)]
pub struct Shared(std::rc::Rc<std::cell::RefCell<Review>>);

impl Shared {
    pub fn new(review: Review) -> Shared {
        Shared(std::rc::Rc::new(std::cell::RefCell::new(review)))
    }

    pub fn note_result(&self, net: f64) {
        self.0.borrow_mut().note_result(net);
    }

    /// Hands written, decisions taken, and how many a solve decided.
    pub fn tally(&self) -> (u64, u64, u64) {
        let review = self.0.borrow();
        let (solved, decisions) = review.coverage();
        (review.hands(), decisions, solved)
    }
}

impl Observer for Shared {
    fn on_decision(&mut self, record: &DecisionRecord) {
        self.0.borrow_mut().on_decision(record);
    }
}

/// Two watchers where one is expected.
#[derive(Debug)]
pub struct Both(pub Box<dyn Observer>, pub Box<dyn Observer>);

impl Observer for Both {
    fn on_decision(&mut self, record: &DecisionRecord) {
        self.0.on_decision(record);
        self.1.on_decision(record);
    }
}

/// What a session has won or lost, counted a hand at a time.
///
/// # Why the stack and not the showdown
///
/// The obvious way to learn who won is to read the client saying so — the
/// `WIN` badge, the winner's cards turned face up, the amount floating over
/// their seat. All of it is drawn briefly, animated while it is drawn, and
/// absent whenever a hand ends without a showdown, which is most of them.
/// Reading it would mean a new set of templates for the least reliable moment
/// on the table.
///
/// The hero's own stack says the same thing and is already read at every
/// frame. Compared between one deal and the next it gives the net result of
/// everything in between — blinds posted, bets made, pots won — without
/// needing to understand any of it. A hand nobody showed down counts the same
/// as one that did.
///
/// # What it cannot see
///
/// Buying more chips looks exactly like winning them. So does a rebuy after
/// busting. Nothing here distinguishes those, and a session where the hero
/// tops up will read as a large win; the count of hands stays honest either
/// way. It is a scoreboard for an unattended session, not an accounting record.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Ledger {
    /// The hero's cards this hand, to notice a new deal.
    hole: Vec<Card>,
    /// The hero's stack when this hand was dealt.
    opened: Option<f64>,
    hands: u64,
    net: f64,
    best: f64,
    worst: f64,
}

impl Ledger {
    pub fn new() -> Ledger {
        Ledger::default()
    }

    /// Notes a reading, and reports the finished hand if this one ended it.
    ///
    /// Called with every settled view. Returns `Some(net)` on the frame where a
    /// new deal is first seen, that being the moment the previous hand's result
    /// is final and readable.
    pub fn observe(&mut self, view: &TableView) -> Option<f64> {
        // Only once the hero holds cards, since a stack read between hands may
        // be mid-animation while a pot is pushed.
        if view.hole.len() != 2 {
            return None;
        }
        let stack = view.hero().and_then(|hero| hero.stack)?;
        if view.hole == self.hole {
            return None;
        }

        let finished = self.opened.map(|opened| stack - opened);
        self.hole = view.hole.clone();
        self.opened = Some(stack);
        if let Some(net) = finished {
            self.hands += 1;
            self.net += net;
            self.best = self.best.max(net);
            self.worst = self.worst.min(net);
        }
        finished
    }

    /// Hands counted, and what they came to.
    pub fn tally(&self) -> (u64, f64) {
        (self.hands, self.net)
    }

    /// The best and worst single hand.
    pub fn extremes(&self) -> (f64, f64) {
        (self.best, self.worst)
    }

    /// Big blinds per hundred hands, once there are any hands.
    pub fn per_hundred(&self) -> Option<f64> {
        (self.hands > 0).then(|| self.net * 100.0 / self.hands as f64)
    }
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

/// How often to look for a dialog waiting to be dismissed.
///
/// Dialogs do not come and go on their own, so finding one a second or two late
/// costs nothing. Looking every frame costs a capture of the whole window on
/// every frame.
const DIALOG_EVERY: Duration = Duration::from_millis(2_000);

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
            record: None,
            keep_turns: None,
            keep_unread: None,
            started_with: None,
            actions: 0,
            kept: 0,
            retreats: 0,
            last_retreat: None,
            turns: 0,
            frames: 0,
            pictured: 0,
            seen: None,
            before: None,
            looked_for_dialog: None,
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
        self.look_keeping(false).map(|(view, _, _)| view)
    }


    /// Reads the table, optionally keeping the frame if it could not be read.
    ///
    /// Hands back the capture the reading came from, so a caller recording the
    /// session can keep the very pixels that produced it rather than taking a
    /// fresh picture of a table that has since moved on.
    fn look_keeping(&self, may_keep: bool) -> Option<(TableView, bool, Capture)> {
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
        Some((view, keep, capture))
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

    /// Declines a dialog that offers a choice, by closing it.
    ///
    /// The straddle offer arrives on the hero's turn and covers the action row.
    /// Neither of its buttons is safe to press — one straddles, one shoves —
    /// and it does not go away on its own, so the turn runs out and the hand is
    /// folded by the clock. Closing it is the only way to decline, and it hands
    /// the turn back.
    ///
    /// Returns whether a dialog was there to close.
    pub fn decline_dialog(&mut self) -> bool {
        // Not on every frame. This runs whenever anything at all is holding the
        // bot up, which is most of the time, and each attempt costs a capture
        // of the whole window on top of the two the reading already took.
        // Dialogs wait for an answer, so looking every couple of seconds finds
        // them just as surely for a third of the work.
        let now = Instant::now();
        if self.looked_for_dialog.is_some_and(|last| now - last < DIALOG_EVERY) {
            return false;
        }
        self.looked_for_dialog = Some(now);

        let Some(capture) = self.window.capture() else {
            return false;
        };
        if capture.is_blank() {
            return false;
        }
        let frame = Frame::new(capture.width, capture.height, &capture.rgb);
        let Some((x, y)) = poker_vision::read_dismiss(&frame) else {
            return false;
        };
        self.window.focus();
        std::thread::sleep(Duration::from_millis(150));
        self.window.click_at(x, y);
        std::thread::sleep(RESPOND);
        true
    }

    /// Whether a picture already taken shows the sit-out notice.
    fn showing_sit_out(&self, capture: &Capture) -> bool {
        let frame = Frame::new(capture.width, capture.height, &capture.rgb);
        poker_vision::read_sit_out(&frame).is_some()
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
        // One reading a turn, checked against the one before it.
        //
        // This used to take three: one that was read and thrown away so an
        // unreadable frame could be kept, then a pair a fifth of a second apart
        // to prove the table was still. Reading a table costs about a third of
        // a second, so that was more than a second of looking before a decision
        // could even begin, and it showed at the table — folds landing two and
        // three seconds after the turn arrived.
        //
        // The pair is still there; it is just spread across turns rather than
        // taken back to back. Last turn's reading is this turn's first half, so
        // the table still has to hold still across two readings to be acted on
        // — over a longer gap, if anything, which is a stricter test.
        let seen = self.look_keeping(self.keep_unread.is_some());
        let Some((view, kept, capture)) = seen else {
            self.before = None;
            self.seen = None;
            if self.is_sat_out() {
                return (None, Some(Held::SatOut));
            }
            return (None, Some(Held::NoPicture));
        };
        if kept {
            self.kept += 1;
        }

        let settled = self
            .before
            .as_ref()
            .is_some_and(|before| before.agrees_with(&view));
        self.before = Some(view.clone());
        if !settled {
            self.seen = None;
            if self.is_sat_out() {
                return (Some(view), Some(Held::SatOut));
            }
            return (Some(view), Some(Held::NotSettled));
        }

        self.seen = Some(capture);

        let stack = view.hero().and_then(|h| h.stack);
        if self.started_with.is_none() {
            self.started_with = stack;
        }
        if let Some(held) = self.safety.breached(self.started_with, stack, self.actions) {
            return (Some(view), Some(held));
        }
        if !view.hero_to_act() {
            // Being sat out reads as a perfectly ordinary quiet table.
            //
            // The check used to live only where a reading failed, on the
            // assumption that a dialog over the felt would stop two captures
            // agreeing. Once the settle test was narrowed to the figures a
            // decision needs, two readings of a dimmed table agreed easily —
            // both saw no seats and no turn — so the reading succeeded, this
            // returned "not our turn", and the seat ran down with the notice
            // sitting on screen the whole time.
            //
            // It costs nothing to ask here: the picture is already in hand.
            if self.seen.as_ref().is_some_and(|seen| self.showing_sit_out(seen)) {
                return (Some(view), Some(Held::SatOut));
            }
            return (Some(view), Some(Held::NotOurTurn));
        }
        self.keep_turn(&view);
        // Named apart, because they want different work. A hole card that did
        // not read is the reader failing at its main job; a table whose money
        // does not add up is almost always a frame caught mid-animation and
        // will come good on its own.
        if let Some(missing) = view.missing_figure() {
            return (Some(view), Some(Held::NotConfident(missing)));
        }
        if !view.is_consistent() {
            return (Some(view), Some(Held::NotAdding));
        }
        (Some(view), None)
    }

    /// Writes one line describing a reading, and a picture beside it.
    ///
    /// The line is written for every frame; pictures stop at [`Session::FRAMES`]
    /// because they are megabytes each and the lines are bytes. Between them the
    /// lines say *when* something went wrong and the pictures say *what* was on
    /// screen, which is the pair needed to fix a reader.
    pub fn record(&mut self, view: Option<&TableView>, held: Option<&Held>) {
        let Some(dir) = self.record.clone() else {
            return;
        };
        let _ = std::fs::create_dir_all(&dir);
        self.frames += 1;

        let line = match view {
            None => format!(
                "{:05} -- no reading -- {}\n",
                self.frames,
                held.map(|h| h.explain()).unwrap_or_default()
            ),
            Some(view) => {
                let seats: Vec<String> = view
                    .seats
                    .iter()
                    .map(|seat| {
                        format!(
                            "{}{}{}/{}",
                            if seat.hero { "*" } else { "" },
                            if seat.in_hand { "c" } else { "-" },
                            seat.stack.map(|s| format!("{s}")).unwrap_or_else(|| "?".into()),
                            seat.bet.map(|b| format!("{b}")).unwrap_or_else(|| "-".into()),
                        )
                    })
                    .collect();
                format!(
                    "{:05} pot {} coll {} board [{}] hole [{}] button {} act {} refused {} live {}/{} | {} | {}\n",
                    self.frames,
                    view.pot.map(|p| format!("{p}")).unwrap_or_else(|| "?".into()),
                    view.collected.map(|c| format!("{c}")).unwrap_or_else(|| "-".into()),
                    view.board.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(" "),
                    view.hole.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(" "),
                    view.button.map(|b| format!("{b}")).unwrap_or_else(|| "?".into()),
                    if view.action.is_some() { "yes" } else { "no" },
                    view.refusals,
                    view.active(),
                    view.occupied(),
                    seats.join(" "),
                    held.map(|h| h.explain()).unwrap_or_else(|| "ACTIONABLE".into()),
                )
            }
        };
        use std::io::Write;
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(dir.join("reading.log"))
        {
            let _ = file.write_all(line.as_bytes());
        }

        if self.pictured >= Session::FRAMES {
            return;
        }
        // The picture the line above was made from, not a new one.
        //
        // Taking a fresh capture here was wrong in a way that quietly undid the
        // point of recording: a few hundred milliseconds separated the reading
        // from the picture beside it, and at a live table that is a different
        // table. The frames were near enough to find a dialog being misread,
        // and not near enough to trust a seven-point difference in how often
        // the dealer button was found — which is exactly the kind of
        // measurement this exists to support.
        let Some(capture) = self.seen.as_ref() else {
            return;
        };
        let name = format!("frame-{:05}.png", self.frames);
        let bytes = crate::png::encode(capture.width, capture.height, &capture.rgb);
        if std::fs::write(dir.join(name), bytes).is_ok() {
            self.pictured += 1;
        }
    }

    /// How many pictures a recorded session keeps.
    ///
    /// A window is several megabytes uncompressed, so this is a few gigabytes
    /// at the top end — enough for a couple of minutes of continuous play,
    /// which is what it takes to see a hand through from deal to showdown
    /// several times over.
    const FRAMES: usize = 400;

    /// How much of the session was recorded.
    pub fn recorded(&self) -> (usize, usize) {
        (self.frames, self.pictured)
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
        // Looked at before being waited on. A client that took the action
        // immediately was still charged a fifth of a second for the privilege
        // of being asked politely.
        let deadline = Instant::now() + TOOK;
        loop {
            if let Some(after) = self.look() {
                if !after.hero_to_act() {
                    // The table has moved on, so what was remembered of it has
                    // too.
                    self.before = None;
                    return Ok(began.elapsed());
                }
            }
            if Instant::now() >= deadline {
                self.before = None;
                return Err(Held::NotTaken);
            }
            std::thread::sleep(Duration::from_millis(120));
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

    /// A hand's result is the stack between one deal and the next.
    #[test]
    fn the_ledger_counts_a_hand_from_deal_to_deal() {
        use poker_core::card::parse_cards;
        use poker_vision::SeatView;

        let table = |cards: &str, stack: f64| {
            let hole = parse_cards(cards).expect("two cards");
            TableView {
                seats: vec![SeatView {
                    x: 0.0,
                    y: 0.0,
                    stack: Some(stack),
                    bet: None,
                    hero: true,
                    in_hand: true,
                }],
                pot: Some(1.5),
                collected: None,
                board: Vec::new(),
                hole,
                button: Some(0),
                action: None,
                refusals: 0,
            }
        };

        let mut ledger = Ledger::new();
        // The first deal has nothing before it to settle.
        assert_eq!(ledger.observe(&table("As Kd", 100.0)), None);
        // Seeing the same hand again is not a new one.
        assert_eq!(ledger.observe(&table("As Kd", 97.0)), None);
        // The next deal settles the last: opened at 100, opens now at 96.
        assert_eq!(ledger.observe(&table("7c 2h", 96.0)), Some(-4.0));
        // And a winning one.
        assert_eq!(ledger.observe(&table("Qs Qh", 110.0)), Some(14.0));

        assert_eq!(ledger.tally(), (2, 10.0));
        assert_eq!(ledger.extremes(), (14.0, -4.0));
        assert_eq!(ledger.per_hundred(), Some(500.0));
    }

    /// A reading without the hero's cards or stack settles nothing.
    #[test]
    fn a_reading_that_cannot_be_trusted_is_not_counted() {
        let mut ledger = Ledger::new();
        let blank = TableView {
            seats: Vec::new(),
            pot: None,
            collected: None,
            board: Vec::new(),
            hole: Vec::new(),
            button: None,
            action: None,
            refusals: 3,
        };
        assert_eq!(ledger.observe(&blank), None);
        assert_eq!(ledger.tally(), (0, 0.0));
        assert_eq!(ledger.per_hundred(), None);
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
            Held::NotConfident("our two cards"),
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
