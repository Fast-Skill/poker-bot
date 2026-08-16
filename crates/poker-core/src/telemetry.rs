//! What the bot saw, what it concluded, and what it did.
//!
//! Every decision emits a [`DecisionRecord`]. Any display — a console monitor,
//! an overlay on the table, a dashboard on a second screen — consumes the same
//! stream, so how the bot is watched is decoupled from how it plays.
//!
//! # Why perception is the headline
//!
//! The dangerous failure in a screen-reading bot is not a bad decision. It is a
//! **misread board**. A bot that reads `Qh` as `Qd` reasons perfectly about a
//! hand that does not exist, acts with total confidence, and looks completely
//! healthy from the outside. Nothing downstream can catch that, because
//! everything downstream is working correctly.
//!
//! So a record carries what the bot *believes it sees*, element by element,
//! each with a confidence. Watching it means glancing at four card symbols and
//! comparing them to the table — which takes a second and catches the one
//! failure mode that matters.
//!
//! In self-play the cards are known exactly and confidence is
//! [`Confidence::certain`]. When a vision layer exists it fills the same fields
//! with real template-match scores, and every display already understands them.

use crate::betting::{Action, Street};
use crate::card::Card;
use crate::table::{HandResult, Position};
use std::fmt;

/// How sure the bot is about each thing it thinks it sees.
///
/// All values are in `0..=1`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Confidence {
    pub hole: f64,
    pub board: f64,
    pub pot: f64,
    pub stacks: f64,
}

impl Confidence {
    /// Everything known exactly, as in self-play.
    pub const fn certain() -> Confidence {
        Confidence {
            hole: 1.0,
            board: 1.0,
            pot: 1.0,
            stacks: 1.0,
        }
    }

    /// The least confident element, which is what should gate acting.
    ///
    /// A bot is only as trustworthy as the worst thing it read: perfect card
    /// recognition beside a misread pot still produces a wrong decision.
    pub fn weakest(&self) -> f64 {
        self.hole.min(self.board).min(self.pot).min(self.stacks)
    }

    /// Whether every element clears `threshold`.
    pub fn clears(&self, threshold: f64) -> bool {
        self.weakest() >= threshold
    }
}

impl Default for Confidence {
    fn default() -> Confidence {
        Confidence::certain()
    }
}

/// The table as the bot understands it.
#[derive(Debug, Clone, PartialEq)]
pub struct Perception {
    pub hole: [Card; 2],
    pub board: Vec<Card>,
    pub street: Street,
    pub position: Position,
    pub pot: u64,
    pub to_call: u64,
    /// The bot's stack, then the opponent's.
    pub stacks: [u64; 2],
    pub confidence: Confidence,
}

/// Where a decision came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    /// Looked up in the solved strategy.
    Blueprint {
        /// The information set consulted, for replay and debugging.
        key: u64,
        /// A readable name for the spot, e.g. `"sb-open"`.
        spot: String,
    },
    /// Decided by the heuristic, because the blueprint had nothing to say.
    Fallback {
        /// Why the blueprint was not used.
        reason: &'static str,
    },
}

impl Source {
    pub fn is_blueprint(&self) -> bool {
        matches!(self, Source::Blueprint { .. })
    }
}

/// One decision, start to finish.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionRecord {
    pub hand: u64,
    pub perception: Perception,
    pub source: Source,
    pub action: Action,
    /// The frequencies the action was drawn from, when there were any.
    ///
    /// Recorded because a bot that folds a hand it folds 5% of the time is
    /// behaving correctly, and without this it looks like a bug.
    pub frequencies: Vec<(String, f64)>,
}

/// Something that watches the bot play.
///
/// Requires `Debug` so an agent holding one stays inspectable — when a live bot
/// misbehaves, being able to print its whole state including what is watching
/// it is worth the small burden on implementors.
pub trait Observer: fmt::Debug {
    /// Called at every decision, before the action is taken.
    fn on_decision(&mut self, record: &DecisionRecord);

    /// Called once a hand is complete.
    fn on_hand_end(&mut self, _hand: u64, _result: &HandResult) {}
}

/// Running totals for a session.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionStats {
    pub big_blind: u64,
    pub hands: u64,
    pub decisions: u64,
    pub from_blueprint: u64,
    /// Decisions taken with at least one element below the confidence floor.
    pub low_confidence: u64,
    pub net_chips: i64,
}

impl SessionStats {
    pub fn new(big_blind: u64) -> SessionStats {
        SessionStats {
            big_blind: big_blind.max(1),
            hands: 0,
            decisions: 0,
            from_blueprint: 0,
            low_confidence: 0,
            net_chips: 0,
        }
    }

    /// Share of decisions the solved strategy actually made.
    ///
    /// The number that says whether the bot is playing the strategy it was
    /// benchmarked on, or improvising.
    pub fn blueprint_share(&self) -> f64 {
        if self.decisions == 0 {
            return 0.0;
        }
        self.from_blueprint as f64 / self.decisions as f64
    }

    pub fn net_bb(&self) -> f64 {
        self.net_chips as f64 / self.big_blind as f64
    }

    pub fn bb_per_100(&self) -> f64 {
        if self.hands == 0 {
            return 0.0;
        }
        self.net_bb() / self.hands as f64 * 100.0
    }

    /// Whether losses have reached `limit_bb`, a stop-loss in big blinds.
    ///
    /// Unattended, this is the difference between a bad session and an empty
    /// stack. Passing a non-positive limit disarms it.
    pub fn hit_stop_loss(&self, limit_bb: f64) -> bool {
        limit_bb > 0.0 && self.net_bb() <= -limit_bb
    }
}

impl fmt::Display for SessionStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} hands  {:+.2} bb ({:+.2} bb/100)  blueprint {:.0}%  low-confidence {}",
            self.hands,
            self.net_bb(),
            self.bb_per_100(),
            self.blueprint_share() * 100.0,
            self.low_confidence
        )
    }
}

/// Prints decisions as they happen and keeps a running summary.
#[derive(Debug)]
pub struct ConsoleMonitor {
    pub stats: SessionStats,
    /// Elements below this are flagged. Vision will produce real scores; in
    /// self-play nothing ever trips it.
    pub confidence_floor: f64,
    /// Whether to print each decision, or only accumulate.
    pub verbose: bool,
}

impl ConsoleMonitor {
    pub fn new(big_blind: u64) -> ConsoleMonitor {
        ConsoleMonitor {
            stats: SessionStats::new(big_blind),
            confidence_floor: 0.90,
            verbose: true,
        }
    }

    /// Renders one decision as the block a watcher reads at a glance.
    pub fn render(&self, record: &DecisionRecord) -> String {
        let perception = &record.perception;
        let blinds = |chips: u64| chips as f64 / self.stats.big_blind as f64;
        let cards = |cards: &[Card]| {
            if cards.is_empty() {
                "—".to_string()
            } else {
                cards
                    .iter()
                    .map(|card| card.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            }
        };

        let mut out = String::new();
        out.push_str(&format!("┌─ hand {} ─ {} ─\n", record.hand, perception.street));
        out.push_str(&format!(
            "│ SEE   hole {:<12} conf {:.2}\n",
            cards(&perception.hole),
            perception.confidence.hole
        ));
        out.push_str(&format!(
            "│       board {:<11} conf {:.2}\n",
            cards(&perception.board),
            perception.confidence.board
        ));
        out.push_str(&format!(
            "│       pot {:.2} bb, to call {:.2} bb   conf {:.2}\n",
            blinds(perception.pot),
            blinds(perception.to_call),
            perception.confidence.pot
        ));
        out.push_str(&format!(
            "│       stacks {:.1} / {:.1} bb   conf {:.2}\n",
            blinds(perception.stacks[0]),
            blinds(perception.stacks[1]),
            perception.confidence.stacks
        ));

        if !perception.confidence.clears(self.confidence_floor) {
            out.push_str(&format!(
                "│ WARN  weakest read {:.2} is below the {:.2} floor\n",
                perception.confidence.weakest(),
                self.confidence_floor
            ));
        }

        match &record.source {
            Source::Blueprint { spot, .. } => {
                out.push_str(&format!("│ THINK {spot} (solved)\n"));
            }
            Source::Fallback { reason } => {
                out.push_str(&format!("│ THINK heuristic — {reason}\n"));
            }
        }

        if !record.frequencies.is_empty() {
            let mixed: Vec<String> = record
                .frequencies
                .iter()
                .filter(|(_, probability)| *probability > 0.005)
                .map(|(name, probability)| format!("{name} {:.0}%", probability * 100.0))
                .collect();
            out.push_str(&format!("│       {}\n", mixed.join("  ")));
        }

        out.push_str(&format!("└ DO    {:?}\n", record.action));
        out
    }
}

impl Observer for ConsoleMonitor {
    fn on_decision(&mut self, record: &DecisionRecord) {
        self.stats.decisions += 1;
        if record.source.is_blueprint() {
            self.stats.from_blueprint += 1;
        }
        if !record
            .perception
            .confidence
            .clears(self.confidence_floor)
        {
            self.stats.low_confidence += 1;
        }
        if self.verbose {
            print!("{}", self.render(record));
        }
    }

    fn on_hand_end(&mut self, _hand: u64, result: &HandResult) {
        self.stats.hands += 1;
        self.stats.net_chips += result.net[0];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::parse_cards;

    fn record(confidence: Confidence, source: Source) -> DecisionRecord {
        let hole = parse_cards("AsKd").expect("valid");
        DecisionRecord {
            hand: 7,
            perception: Perception {
                hole: [hole[0], hole[1]],
                board: parse_cards("Qh 2c 4d").expect("valid"),
                street: Street::Flop,
                position: Position::Button,
                pot: 500,
                to_call: 0,
                stacks: [9_750, 9_400],
                confidence,
            },
            source,
            action: Action::RaiseTo(330),
            frequencies: vec![("check".into(), 0.28), ("bet".into(), 0.72)],
        }
    }

    fn solved() -> Source {
        Source::Blueprint {
            key: 42,
            spot: "flop, in position".into(),
        }
    }

    #[test]
    fn the_weakest_element_decides_confidence() {
        // Perfect cards beside a misread pot is still a wrong decision, so the
        // minimum is what counts, not the average.
        let mixed = Confidence {
            hole: 1.0,
            board: 1.0,
            pot: 0.42,
            stacks: 1.0,
        };
        assert_eq!(mixed.weakest(), 0.42);
        assert!(!mixed.clears(0.9));
        assert!(Confidence::certain().clears(1.0));
    }

    #[test]
    fn low_confidence_decisions_are_counted_and_flagged() {
        let mut monitor = ConsoleMonitor::new(100);
        monitor.verbose = false;

        monitor.on_decision(&record(Confidence::certain(), solved()));
        assert_eq!(monitor.stats.low_confidence, 0);

        let shaky = Confidence {
            board: 0.55,
            ..Confidence::certain()
        };
        monitor.on_decision(&record(shaky, solved()));
        assert_eq!(monitor.stats.low_confidence, 1);
        assert!(monitor.render(&record(shaky, solved())).contains("WARN"));
    }

    #[test]
    fn the_blueprint_share_reports_how_much_was_actually_solved() {
        let mut monitor = ConsoleMonitor::new(100);
        monitor.verbose = false;

        monitor.on_decision(&record(Confidence::certain(), solved()));
        monitor.on_decision(&record(
            Confidence::certain(),
            Source::Fallback { reason: "postflop" },
        ));
        monitor.on_decision(&record(
            Confidence::certain(),
            Source::Fallback { reason: "postflop" },
        ));

        assert_eq!(monitor.stats.decisions, 3);
        assert_eq!(monitor.stats.from_blueprint, 1);
        assert!((monitor.stats.blueprint_share() - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn a_rendered_decision_shows_what_was_seen_and_why() {
        let monitor = ConsoleMonitor::new(100);
        let text = monitor.render(&record(Confidence::certain(), solved()));

        assert!(text.contains("As Kd"), "the hole cards: {text}");
        assert!(text.contains("Qh 2c 4d"), "the board: {text}");
        assert!(text.contains("5.00 bb"), "the pot: {text}");
        assert!(text.contains("flop, in position"), "the spot: {text}");
        // Mixed frequencies, so a surprising action can be recognised as
        // correct rather than assumed to be a bug.
        assert!(text.contains("bet 72%"), "the frequencies: {text}");
        assert!(text.contains("check 28%"), "{text}");
        assert!(!text.contains("WARN"), "nothing was uncertain: {text}");
    }

    #[test]
    fn a_fallback_says_so_plainly() {
        let monitor = ConsoleMonitor::new(100);
        let text = monitor.render(&record(
            Confidence::certain(),
            Source::Fallback {
                reason: "no postflop solve",
            },
        ));
        assert!(text.contains("heuristic"), "{text}");
        assert!(text.contains("no postflop solve"), "{text}");
    }

    #[test]
    fn negligible_frequencies_are_not_printed() {
        let monitor = ConsoleMonitor::new(100);
        let mut decision = record(Confidence::certain(), solved());
        decision.frequencies = vec![
            ("fold".into(), 0.0),
            ("call".into(), 0.001),
            ("raise".into(), 0.999),
        ];
        let text = monitor.render(&decision);
        assert!(text.contains("raise 100%"), "{text}");
        assert!(!text.contains("fold"), "a zero-weight action is noise: {text}");
    }

    #[test]
    fn session_totals_track_the_running_result() {
        let mut stats = SessionStats::new(100);
        stats.hands = 200;
        stats.net_chips = 1_000;

        assert_eq!(stats.net_bb(), 10.0);
        assert_eq!(stats.bb_per_100(), 5.0);
        assert!(stats.to_string().contains("+10.00 bb"));
    }

    #[test]
    fn the_stop_loss_arms_only_when_it_is_set() {
        let mut stats = SessionStats::new(100);
        stats.net_chips = -4_000; // -40 bb

        assert!(!stats.hit_stop_loss(50.0), "not down 50 yet");
        assert!(stats.hit_stop_loss(40.0), "exactly at the limit");
        assert!(stats.hit_stop_loss(30.0));
        assert!(!stats.hit_stop_loss(0.0), "zero disarms it");
        assert!(!stats.hit_stop_loss(-10.0), "so does a negative limit");
    }

    #[test]
    fn an_empty_session_reports_zeroes_rather_than_dividing_by_zero() {
        let stats = SessionStats::new(100);
        assert_eq!(stats.blueprint_share(), 0.0);
        assert_eq!(stats.bb_per_100(), 0.0);
        assert_eq!(stats.net_bb(), 0.0);
    }
}
