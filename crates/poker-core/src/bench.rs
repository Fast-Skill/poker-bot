//! Measuring strategies against each other, in chips.
//!
//! [`crate::cfr::Solver::exploitability`] answers "how far from equilibrium is
//! this, inside its own model". This module answers a different and more
//! stubborn question: **does it win money in the real game.** The two can
//! disagree, and when they do it is the model that is wrong.
//!
//! # Duplicate matches
//!
//! Poker's standard deviation is roughly 100 bb/100. Detecting a 5 bb/100 edge
//! straightforwardly needs something like 150,000 hands, because most of what a
//! result measures is who got dealt aces.
//!
//! [`duplicate_match`] removes most of that. Every deal is played twice, with
//! the agents exchanging both seats and holdings, so each side plays the same
//! cards from the same position. Whatever the deck did, it did to both — and
//! the difference that survives is skill. In practice this cuts the hands
//! needed by roughly an order of magnitude.

use crate::abstraction::HandClass;
use crate::betting::{Action, Street};
use crate::eval::{evaluate, Category};
use crate::rng::Rng;
use crate::table::{Agent, Deck, Table, View};
use std::fmt;

/// The result of a match.
#[derive(Debug, Clone, PartialEq)]
pub struct MatchReport {
    pub names: [String; 2],
    /// Hands played in total, counting both halves of each duplicate pair.
    pub hands: u64,
    /// The first agent's win rate, in big blinds per 100 hands.
    pub bb_per_100: f64,
    /// Half-width of the 95% confidence interval on `bb_per_100`.
    pub confidence_95: f64,
    /// Hands that reached showdown.
    pub showdowns: u64,
}

impl MatchReport {
    /// Whether the measured edge is distinguishable from zero.
    ///
    /// A win rate without this check means very little: over a few thousand
    /// hands, a losing strategy shows a profit often enough to fool anyone who
    /// is not looking for it.
    pub fn is_significant(&self) -> bool {
        self.bb_per_100.abs() > self.confidence_95
    }

    /// The interval the true win rate probably lies in.
    pub fn interval(&self) -> (f64, f64) {
        (
            self.bb_per_100 - self.confidence_95,
            self.bb_per_100 + self.confidence_95,
        )
    }

    /// Whether the first agent beat the second at 95% confidence.
    pub fn first_agent_wins(&self) -> bool {
        self.interval().0 > 0.0
    }
}

impl fmt::Display for MatchReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (low, high) = self.interval();
        write!(
            f,
            "{} vs {}: {:+.2} bb/100  (95% CI {:+.2} to {:+.2}, {} hands, {:.0}% showdown){}",
            self.names[0],
            self.names[1],
            self.bb_per_100,
            low,
            high,
            self.hands,
            self.showdowns as f64 / self.hands as f64 * 100.0,
            if self.is_significant() { "" } else { "  [not significant]" }
        )
    }
}

/// Plays `pairs` duplicate deals and reports the first agent's win rate.
///
/// Each pair is two hands: the same cards and the same board, with the agents
/// exchanging seats. Statistics are computed over pairs rather than hands,
/// since the two halves of a pair are deliberately correlated and treating them
/// as independent would understate the error bars.
///
/// # Panics
/// Panics if `pairs` is zero, since there is nothing to measure.
pub fn duplicate_match(
    table: &Table,
    first: &mut dyn Agent,
    second: &mut dyn Agent,
    pairs: u64,
    rng: &mut Rng,
) -> MatchReport {
    assert!(pairs > 0, "a match needs at least one deal");

    let big_blind = table.big_blind() as f64;
    let mut deck = Deck::fresh();
    let mut totals = Vec::with_capacity(pairs as usize);
    let mut showdowns = 0u64;

    for _ in 0..pairs {
        deck.shuffle(rng);

        // First agent on the button.
        let forward = {
            let mut seats: Vec<&mut dyn Agent> = vec![&mut *first, &mut *second];
            table.play_hand(&mut seats, deck.hand_cards(2), rng)
        };
        // Same cards, seats exchanged: the first agent now holds what the
        // second just held, from the other side of the button.
        let swapped = deck.swap_holdings(0, 1);
        let reverse = {
            let mut seats: Vec<&mut dyn Agent> = vec![&mut *second, &mut *first];
            table.play_hand(&mut seats, swapped.hand_cards(2), rng)
        };

        if forward.showdown {
            showdowns += 1;
        }
        if reverse.showdown {
            showdowns += 1;
        }

        // The first agent was seat 0 going forward and seat 1 coming back.
        let net = forward.net[0] + reverse.net[1];
        totals.push(net as f64 / big_blind);
    }

    let hands = pairs * 2;
    let mean = totals.iter().sum::<f64>() / pairs as f64;

    // Sample variance over pairs; a single pair has none to speak of.
    let variance = if pairs > 1 {
        totals
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / (pairs - 1) as f64
    } else {
        0.0
    };
    let standard_error = (variance / pairs as f64).sqrt();

    // Each pair is two hands, so a per-pair mean scales to bb/100 by 50.
    MatchReport {
        names: [first.name().to_string(), second.name().to_string()],
        hands,
        bb_per_100: mean * 50.0,
        confidence_95: 1.96 * standard_error * 50.0,
        showdowns,
    }
}

/// Result of a ring-game match, per seat.
#[derive(Debug, Clone, PartialEq)]
pub struct RingReport {
    pub names: Vec<String>,
    pub hands: u64,
    /// Each agent's win rate in big blinds per 100 hands, in the order given.
    pub bb_per_100: Vec<f64>,
    /// Hands that reached showdown.
    pub showdowns: u64,
}

impl fmt::Display for RingReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}-handed, {} hands:", self.names.len(), self.hands)?;
        for (name, rate) in self.names.iter().zip(&self.bb_per_100) {
            write!(f, "  {name} {rate:+.1}")?;
        }
        Ok(())
    }
}

/// Plays a ring game, rotating the button so every agent sits in every seat.
///
/// Unlike [`duplicate_match`] the deal is not paired, so results are far
/// noisier — three-handed and up there is no clean way to replay a deal such
/// that every seat sees the same cards. Use this to measure *coverage* and to
/// check a bot plays legally at a full table; use duplicate matches to measure
/// an edge.
///
/// # Panics
/// Panics if there are fewer than two agents, or `hands` is zero.
pub fn ring_match(
    table: &Table,
    mut seats: Vec<&mut dyn Agent>,
    hands: u64,
    rng: &mut Rng,
) -> RingReport {
    assert!(seats.len() >= 2, "a ring game needs at least two agents");
    assert!(hands > 0, "a match needs at least one hand");

    let players = seats.len();
    let names: Vec<String> = seats.iter().map(|agent| agent.name().to_string()).collect();
    let big_blind = table.big_blind() as f64;
    let mut totals = vec![0i64; players];
    let mut showdowns = 0u64;
    let mut deck = Deck::fresh();
    // How far the seating has rotated: seat `i` currently holds agent
    // `(i + rotations) % players`.
    let mut rotations = 0usize;

    for _ in 0..hands {
        deck.shuffle(rng);
        let result = table.play_hand(&mut seats, deck.hand_cards(players), rng);
        if result.showdown {
            showdowns += 1;
        }
        for (seat, net) in result.net.iter().enumerate() {
            totals[(seat + rotations) % players] += net;
        }
        // Rotate so nobody keeps the button.
        seats.rotate_left(1);
        rotations = (rotations + 1) % players;
    }

    RingReport {
        names,
        hands,
        bb_per_100: totals
            .iter()
            .map(|chips| *chips as f64 / big_blind / hands as f64 * 100.0)
            .collect(),
        showdowns,
    }
}

// --- baseline opponents -----------------------------------------------------

/// Folds whenever folding is legal.
///
/// The floor: anything that cannot beat this is broken.
#[derive(Debug, Default, Clone, Copy)]
pub struct AlwaysFold;

impl Agent for AlwaysFold {
    fn name(&self) -> &str {
        "always-fold"
    }
    fn act(&mut self, view: &View, _rng: &mut Rng) -> Action {
        if view.legal.can_fold {
            Action::Fold
        } else {
            Action::Check
        }
    }
}

/// Calls everything and never raises — the classic calling station.
///
/// Cannot be bluffed, and loses to value betting.
#[derive(Debug, Default, Clone, Copy)]
pub struct AlwaysCall;

impl Agent for AlwaysCall {
    fn name(&self) -> &str {
        "always-call"
    }
    fn act(&mut self, view: &View, _rng: &mut Rng) -> Action {
        if view.legal.can_check {
            Action::Check
        } else {
            Action::Call
        }
    }
}

/// Moves all in at every opportunity.
///
/// Beating this requires only patience, but it punishes anything that folds too
/// much or calls too wide.
#[derive(Debug, Default, Clone, Copy)]
pub struct AlwaysJam;

impl Agent for AlwaysJam {
    fn name(&self) -> &str {
        "always-jam"
    }
    fn act(&mut self, view: &View, _rng: &mut Rng) -> Action {
        match view.legal.raise_to {
            Some((_, max)) => Action::RaiseTo(max),
            None if view.legal.call_cost.is_some() => Action::Call,
            None => Action::Check,
        }
    }
}

/// A rule-based opponent that plays recognisable poker.
///
/// Preflop it uses the Chen formula, a long-standing hand-strength heuristic;
/// postflop it bets made hands and gives up without one. It has clear leaks —
/// no bluffing, no balance, position barely considered — but it folds bad hands
/// and value-bets good ones, which is more than the degenerate baselines do.
///
/// This is the Stage 2 gate: beating it means beating a real strategy.
#[derive(Debug, Clone, Copy)]
pub struct ChartBot {
    /// Chen score required to enter a pot. Around 8 is roughly a 20% range.
    pub open_threshold: i32,
    /// Chen score required to call a raise.
    pub call_threshold: i32,
}

impl Default for ChartBot {
    fn default() -> ChartBot {
        ChartBot {
            open_threshold: 8,
            call_threshold: 10,
        }
    }
}

impl ChartBot {
    /// The Chen formula: a hand-strength heuristic that predates solvers and
    /// still sorts starting hands about right.
    ///
    /// High card scores first, pairs double it, suitedness adds, and gaps
    /// subtract — with a bonus for low connected cards that make straights.
    pub fn chen_score(class: HandClass) -> i32 {
        let value = |rank: crate::card::Rank| -> f64 {
            match rank {
                crate::card::Rank::Ace => 10.0,
                crate::card::Rank::King => 8.0,
                crate::card::Rank::Queen => 7.0,
                crate::card::Rank::Jack => 6.0,
                other => (other.index() as f64 + 2.0) / 2.0,
            }
        };

        let high = class.high();
        let low = class.low();
        let mut score = value(high);

        if class.is_pair() {
            score = (score * 2.0).max(5.0);
        } else {
            if class.is_suited() {
                score += 2.0;
            }
            let gap = high.index() as i32 - low.index() as i32 - 1;
            score -= match gap {
                0 => 0.0,
                1 => 1.0,
                2 => 2.0,
                3 => 4.0,
                _ => 5.0,
            };
            // Low connectors make straights and deserve a nudge back.
            if gap <= 1 && high < crate::card::Rank::Queen {
                score += 1.0;
            }
        }

        score.ceil() as i32
    }

    /// How strong the made hand is, once a board exists.
    fn made_hand(view: &View) -> Category {
        let mut cards = view.hole.to_vec();
        cards.extend_from_slice(view.board);
        evaluate(&cards).category()
    }
}

impl Agent for ChartBot {
    fn name(&self) -> &str {
        "chart-bot"
    }

    fn act(&mut self, view: &View, _rng: &mut Rng) -> Action {
        if view.street == Street::Preflop {
            let class = HandClass::from_cards(view.hole[0], view.hole[1]);
            let score = ChartBot::chen_score(class);

            // Facing a bet: continue only with the tighter range.
            if view.to_call > 0 {
                return if score >= self.call_threshold {
                    Action::Call
                } else {
                    Action::Fold
                };
            }
            // Unopened: raise the opening range, otherwise take the free card.
            if score >= self.open_threshold {
                if let Some(target) = view.raise_fraction(1.0) {
                    return Action::RaiseTo(target);
                }
            }
            return if view.legal.can_check {
                Action::Check
            } else {
                Action::Fold
            };
        }

        // Postflop: bet made hands, fold without one.
        let made = ChartBot::made_hand(view);
        let strong = made >= Category::Pair;

        if view.to_call > 0 {
            return if strong { Action::Call } else { Action::Fold };
        }
        if strong {
            if let Some(target) = view.raise_fraction(0.66) {
                return Action::RaiseTo(target);
            }
        }
        if view.legal.can_check {
            Action::Check
        } else {
            Action::Fold
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> Table {
        Table::standard()
    }

    #[test]
    fn folding_everything_costs_exactly_the_blinds_against_a_bettor() {
        // Against an opponent that always bets, a folder gives up the small
        // blind on the button and the big blind off it: 0.75 bb per hand, so
        // -75 bb/100 exactly. This is the arithmetic floor of poker.
        let mut rng = Rng::new(1);
        let (mut folder, mut jammer) = (AlwaysFold, AlwaysJam);
        let report = duplicate_match(&table(), &mut folder, &mut jammer, 500, &mut rng);

        assert!(
            (report.bb_per_100 + 75.0).abs() < 1.0,
            "expected -75 bb/100, got {}",
            report.bb_per_100
        );
        assert!(report.is_significant());
        assert!(!report.first_agent_wins());
    }

    #[test]
    fn a_folder_leaks_far_less_against_a_passive_opponent() {
        // The catch that makes "always fold" a poor name: as the big blind
        // facing a mere call, nothing is owed, so it *checks* — and checks all
        // the way to a free showdown it wins about half the time. Only an
        // opponent who bets can actually collect the blind.
        let mut rng = Rng::new(1);

        let (mut folder, mut caller) = (AlwaysFold, AlwaysCall);
        let passive = duplicate_match(&table(), &mut folder, &mut caller, 500, &mut rng);

        let (mut folder, mut jammer) = (AlwaysFold, AlwaysJam);
        let aggressive = duplicate_match(&table(), &mut folder, &mut jammer, 500, &mut rng);

        assert!(
            passive.bb_per_100 > aggressive.bb_per_100 + 30.0,
            "folding should cost far less against a limper: {passive} vs {aggressive}"
        );
        // Roughly half a blind per hand: given up on the button, kept in the
        // big blind.
        assert!(
            (passive.bb_per_100 + 25.0).abs() < 3.0,
            "expected about -25 bb/100, got {}",
            passive.bb_per_100
        );
    }

    #[test]
    fn duplicate_pairing_cancels_the_deal() {
        // Two identical agents cannot have an edge, and with the deal
        // duplicated the measured result should be almost exactly zero — not
        // merely zero on average.
        let mut rng = Rng::new(2);
        let (mut a, mut b) = (AlwaysCall, AlwaysCall);
        let report = duplicate_match(&table(), &mut a, &mut b, 300, &mut rng);

        assert!(
            report.bb_per_100.abs() < 1.0,
            "identical agents drifted to {} bb/100",
            report.bb_per_100
        );
        assert!(!report.is_significant(), "there is no edge to find");
    }

    #[test]
    fn a_calling_station_beats_a_folder() {
        // Only by the blinds it can actually collect — see the test above for
        // why this is 25 rather than 75.
        let mut rng = Rng::new(3);
        let (mut caller, mut folder) = (AlwaysCall, AlwaysFold);
        let report = duplicate_match(&table(), &mut caller, &mut folder, 500, &mut rng);

        assert!(report.first_agent_wins(), "{report}");
        assert!(report.bb_per_100 > 20.0, "{report}");
    }

    #[test]
    fn the_chart_bot_beats_every_degenerate_baseline() {
        // The Stage 1 gate. A strategy that folds bad hands and value-bets good
        // ones must beat opponents that do neither.
        let mut rng = Rng::new(4);

        let mut bot = ChartBot::default();
        let mut folder = AlwaysFold;
        let vs_fold = duplicate_match(&table(), &mut bot, &mut folder, 400, &mut rng);
        assert!(vs_fold.first_agent_wins(), "{vs_fold}");

        let mut bot = ChartBot::default();
        let mut caller = AlwaysCall;
        let vs_call = duplicate_match(&table(), &mut bot, &mut caller, 1_000, &mut rng);
        assert!(vs_call.first_agent_wins(), "{vs_call}");

        let mut bot = ChartBot::default();
        let mut jammer = AlwaysJam;
        let vs_jam = duplicate_match(&table(), &mut bot, &mut jammer, 1_000, &mut rng);
        assert!(vs_jam.first_agent_wins(), "{vs_jam}");
    }

    #[test]
    fn chen_scores_sort_hands_the_way_players_do() {
        let score = |text: &str| ChartBot::chen_score(text.parse().expect("valid class"));

        // The canonical values from the formula.
        assert_eq!(score("AA"), 20);
        assert_eq!(score("KK"), 16);
        assert_eq!(score("QQ"), 14);
        assert_eq!(score("AKs"), 12);
        assert_eq!(score("22"), 5, "the minimum for a pair");

        // Ordering, which is what actually gets used.
        assert!(score("AA") > score("KK"));
        assert!(score("AKs") > score("AKo"));
        assert!(score("AKo") > score("A2o"));
        assert!(score("JTs") > score("J2s"), "gaps cost");
        assert!(score("72o") < 0, "the worst hand in poker scores badly");
    }

    #[test]
    fn a_report_reads_clearly() {
        let report = MatchReport {
            names: ["hero".into(), "villain".into()],
            hands: 1_000,
            bb_per_100: 12.5,
            confidence_95: 4.0,
            showdowns: 300,
        };
        let text = report.to_string();
        assert!(text.contains("+12.50 bb/100"), "{text}");
        assert!(text.contains("+8.50"), "{text}");
        assert!(text.contains("+16.50"), "{text}");
        assert!(report.first_agent_wins());

        let noisy = MatchReport {
            confidence_95: 20.0,
            ..report
        };
        assert!(!noisy.is_significant());
        assert!(noisy.to_string().contains("not significant"));
    }

    #[test]
    fn confidence_narrows_as_hands_accumulate() {
        let mut rng = Rng::new(5);
        let short = {
            let (mut a, mut b) = (ChartBot::default(), AlwaysCall);
            duplicate_match(&table(), &mut a, &mut b, 100, &mut rng)
        };
        let long = {
            let (mut a, mut b) = (ChartBot::default(), AlwaysCall);
            duplicate_match(&table(), &mut a, &mut b, 2_000, &mut rng)
        };
        assert!(
            long.confidence_95 < short.confidence_95,
            "more hands should sharpen the estimate: {} vs {}",
            short.confidence_95,
            long.confidence_95
        );
    }

    #[test]
    #[should_panic(expected = "at least one deal")]
    fn an_empty_match_is_rejected() {
        let mut rng = Rng::new(6);
        let (mut a, mut b) = (AlwaysCall, AlwaysFold);
        duplicate_match(&table(), &mut a, &mut b, 0, &mut rng);
    }
}
