//! The bot: a solved blueprint playing at a real table.
//!
//! This is where a stored strategy becomes a player. The blueprint answers
//! preflop; a heuristic covers postflop until a postflop solve exists; and a
//! translation layer maps between the table's view of a hand and the solver's.
//!
//! # Action translation
//!
//! The solver thinks in abstract stages — "small blind facing a 3-bet" — while
//! the table reports pots, stacks, and amounts owed. Something has to map
//! between them, and that mapping is a known source of silent losses: a
//! technically correct strategy applied to a misidentified spot plays perfectly
//! by accident and badly on purpose.
//!
//! The mapping here is deliberately conservative. Spots it cannot identify are
//! handed to the fallback rather than guessed at, and every such handoff is
//! counted — [`BlueprintAgent::coverage`] reports how much of a session the
//! blueprint actually decided, which is the number that says whether the
//! abstraction fits the game being played.

use crate::bench::ChartBot;
use crate::betting::{Action, Street};
use crate::blueprint::Blueprint;
use crate::abstraction::HandClass;
use crate::preflop::{self, Preflop, Sizing};
use crate::rng::Rng;
use crate::table::{Agent, Position, View};

/// Which abstract action a blueprint index means, per stage.
///
/// Mirrors the order [`crate::preflop::Preflop`] builds its action lists in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Abstract {
    Fold,
    Passive,
    Raise,
    Jam,
}

/// A bot that plays a stored strategy.
#[derive(Debug)]
pub struct BlueprintAgent {
    name: String,
    blueprint: Blueprint,
    sizing: Sizing,
    /// Covers postflop, and any preflop spot the blueprint does not hold.
    fallback: ChartBot,
    /// Preflop decisions taken so far this hand, used to tell an open from a
    /// response to a re-raise.
    preflop_decisions: u32,
    decisions: u64,
    fallbacks: u64,
}

impl BlueprintAgent {
    /// Wraps a blueprint solved with `sizing`.
    ///
    /// The sizing must match the solve. A blueprint that decided to "raise"
    /// meant a specific amount, and betting a different one plays a strategy
    /// nobody solved for.
    pub fn new(name: impl Into<String>, blueprint: Blueprint, sizing: Sizing) -> BlueprintAgent {
        BlueprintAgent {
            name: name.into(),
            blueprint,
            sizing,
            fallback: ChartBot::default(),
            preflop_decisions: 0,
            decisions: 0,
            fallbacks: 0,
        }
    }

    /// Decisions taken, and how many came from the blueprint rather than the
    /// fallback.
    pub fn coverage(&self) -> (u64, u64) {
        (self.decisions - self.fallbacks, self.decisions)
    }

    /// Share of decisions the blueprint actually made, in `0..=1`.
    pub fn coverage_fraction(&self) -> f64 {
        if self.decisions == 0 {
            return 0.0;
        }
        (self.decisions - self.fallbacks) as f64 / self.decisions as f64
    }

    /// Identifies which solved spot this table state corresponds to.
    ///
    /// Returns `None` when the spot is outside the solved tree — postflop, or a
    /// preflop line deeper than the ladder covers.
    fn stage_for(&self, view: &View) -> Option<preflop::Stage> {
        if view.street != Street::Preflop {
            return None;
        }
        let button = view.position == Position::Button;

        // An opponent with nothing behind has moved all in, whatever the
        // betting looked like before that.
        if view.opponent_stack == 0 && view.to_call > 0 {
            return Some(if button {
                preflop::Stage::SbVsJam
            } else {
                preflop::Stage::BbVsJam
            });
        }

        match (button, self.preflop_decisions) {
            (true, 0) => Some(preflop::Stage::SbOpen),
            (false, 0) => Some(preflop::Stage::BbVsOpen),
            (true, 1) => Some(preflop::Stage::SbVs3Bet),
            (false, 1) => Some(preflop::Stage::BbVs4Bet),
            // Beyond a 4-bet the ladder stops; the fallback takes over.
            _ => None,
        }
    }

    /// The abstract actions available at a stage, in blueprint index order.
    fn actions_at(&self, stage: preflop::Stage) -> Vec<Abstract> {
        let raiseable = matches!(
            stage,
            preflop::Stage::SbOpen | preflop::Stage::BbVsOpen | preflop::Stage::SbVs3Bet
        );
        match stage {
            preflop::Stage::SbOpen => {
                let mut actions = vec![Abstract::Fold];
                if raiseable {
                    actions.push(Abstract::Raise);
                }
                actions.push(Abstract::Jam);
                actions
            }
            preflop::Stage::BbVsOpen | preflop::Stage::SbVs3Bet => {
                vec![Abstract::Fold, Abstract::Passive, Abstract::Raise, Abstract::Jam]
            }
            preflop::Stage::BbVs4Bet => {
                vec![Abstract::Fold, Abstract::Passive, Abstract::Jam]
            }
            preflop::Stage::SbVsJam | preflop::Stage::BbVsJam => {
                vec![Abstract::Fold, Abstract::Passive]
            }
            // Terminal and chance stages are never a decision, so `stage_for`
            // cannot return them.
            _ => Vec::new(),
        }
    }

    /// The chips a raise at `stage` should commit, in table units.
    fn raise_target(&self, stage: preflop::Stage, view: &View) -> Option<u64> {
        let blinds = match stage {
            preflop::Stage::SbOpen => self.sizing.open_to,
            preflop::Stage::BbVsOpen => self.sizing.three_bet_to,
            preflop::Stage::SbVs3Bet => self.sizing.four_bet_to,
            _ => return None,
        };
        let (min, max) = view.legal.raise_to?;
        let target = (blinds * view.big_blind as f64).round() as u64;
        Some(target.clamp(min, max))
    }

    /// Turns an abstract action into something the table will accept.
    ///
    /// Returns `None` when the abstract action has no legal counterpart, which
    /// sends the spot to the fallback rather than to a panic.
    fn concrete(&self, chosen: Abstract, stage: preflop::Stage, view: &View) -> Option<Action> {
        match chosen {
            Abstract::Fold => {
                // Folding for free is never offered; checking is strictly better.
                if view.legal.can_fold {
                    Some(Action::Fold)
                } else if view.legal.can_check {
                    Some(Action::Check)
                } else {
                    None
                }
            }
            Abstract::Passive => {
                if view.legal.can_check {
                    Some(Action::Check)
                } else if view.legal.call_cost.is_some() {
                    Some(Action::Call)
                } else {
                    None
                }
            }
            Abstract::Raise => self.raise_target(stage, view).map(Action::RaiseTo),
            Abstract::Jam => view.legal.raise_to.map(|(_, max)| Action::RaiseTo(max)),
        }
    }
}

impl Agent for BlueprintAgent {
    fn name(&self) -> &str {
        &self.name
    }

    fn new_hand(&mut self) {
        self.preflop_decisions = 0;
    }

    fn act(&mut self, view: &View, rng: &mut Rng) -> Action {
        self.decisions += 1;

        let action = self.stage_for(view).and_then(|stage| {
            let class = HandClass::from_cards(view.hole[0], view.hole[1]);
            let key = Preflop::info_key(stage, class.index());
            // Sampling, not the modal action: a mixed strategy played greedily
            // is a different and more exploitable strategy.
            let index = self.blueprint.sample(key, rng)?;
            let chosen = *self.actions_at(stage).get(index)?;
            self.concrete(chosen, stage, view)
        });

        if view.street == Street::Preflop {
            self.preflop_decisions += 1;
        }

        match action {
            Some(action) if view.legal.permits(action) => action,
            // Anything unrecognised or unplayable goes to the heuristic. It is
            // counted, because a bot silently falling back on most of its
            // decisions is not playing the strategy anybody solved.
            _ => {
                self.fallbacks += 1;
                self.fallback.act(view, rng)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::{duplicate_match, AlwaysCall, AlwaysFold, AlwaysJam};
    use crate::cfr::Solver;
    use crate::pushfold::EquityTable;
    use crate::table::Table;

    /// A small solve is enough to exercise the plumbing; the benchmarks use a
    /// properly trained blueprint.
    fn agent(stack_bb: f64) -> BlueprintAgent {
        let equity = EquityTable::sampled_parallel(300, 0x51DE, 4);
        let sizing = Sizing::default();
        let mut rng = Rng::new(0xF01D);
        let mut solver = Solver::new(Preflop::new(stack_bb, sizing, equity));
        solver.train_sampled(200_000, &mut rng);
        let blueprint = Blueprint::from_solver(&solver, format!("preflop/{stack_bb}bb"));
        BlueprintAgent::new("blueprint", blueprint, sizing)
    }

    fn table() -> Table {
        Table::standard()
    }

    #[test]
    fn the_bot_plays_legal_poker_for_a_long_session() {
        // The table panics on an illegal action, so simply surviving this is
        // the assertion.
        let mut bot = agent(100.0);
        let mut opponent = AlwaysCall;
        let mut rng = Rng::new(1);
        let report = duplicate_match(&table(), &mut bot, &mut opponent, 500, &mut rng);
        assert_eq!(report.hands, 1_000);
    }

    #[test]
    fn the_blueprint_decides_most_preflop_spots() {
        let mut bot = agent(100.0);
        let mut opponent = AlwaysCall;
        let mut rng = Rng::new(2);
        duplicate_match(&table(), &mut bot, &mut opponent, 300, &mut rng);

        let (from_blueprint, total) = bot.coverage();
        assert!(total > 0);
        assert!(
            bot.coverage_fraction() > 0.25,
            "the blueprint decided only {from_blueprint} of {total} spots"
        );
    }

    #[test]
    fn a_hand_resets_the_preflop_counter() {
        // Without this, the second hand of a session would be read as though
        // it were already three bets deep.
        let mut bot = agent(100.0);
        bot.preflop_decisions = 3;
        bot.new_hand();
        assert_eq!(bot.preflop_decisions, 0);
    }

    #[test]
    fn the_bot_beats_every_degenerate_baseline() {
        // Stage 1. Anything that cannot clear this is broken.
        let mut rng = Rng::new(3);

        let mut bot = agent(100.0);
        let mut folder = AlwaysFold;
        let vs_fold = duplicate_match(&table(), &mut bot, &mut folder, 400, &mut rng);
        assert!(vs_fold.first_agent_wins(), "{vs_fold}");

        let mut bot = agent(100.0);
        let mut caller = AlwaysCall;
        let vs_call = duplicate_match(&table(), &mut bot, &mut caller, 1_500, &mut rng);
        assert!(vs_call.first_agent_wins(), "{vs_call}");

        let mut bot = agent(100.0);
        let mut jammer = AlwaysJam;
        let vs_jam = duplicate_match(&table(), &mut bot, &mut jammer, 1_500, &mut rng);
        assert!(vs_jam.first_agent_wins(), "{vs_jam}");
    }

    #[test]
    fn an_all_in_opponent_is_recognised_regardless_of_the_betting_before_it() {
        // The jam stages are keyed off the opponent having nothing behind,
        // which holds however the pot got there.
        let bot = agent(20.0);
        let round = crate::betting::BettingRound::new(
            vec![crate::betting::Seat::new(1_000), crate::betting::Seat::new(0)],
            0,
            100,
        );
        let legal = round.legal_actions();
        let hole = crate::card::parse_cards("AsKd").expect("valid");

        let view = View {
            hole: [hole[0], hole[1]],
            board: &[],
            street: Street::Preflop,
            position: Position::Button,
            pot: 2_000,
            to_call: 500,
            stack: 1_000,
            opponent_stack: 0,
            big_blind: 100,
            legal: &legal,
        };
        assert_eq!(bot.stage_for(&view), Some(preflop::Stage::SbVsJam));
    }

    #[test]
    fn postflop_spots_fall_through_to_the_heuristic() {
        let bot = agent(100.0);
        let round = crate::betting::BettingRound::new(
            vec![crate::betting::Seat::new(1_000), crate::betting::Seat::new(1_000)],
            0,
            100,
        );
        let legal = round.legal_actions();
        let hole = crate::card::parse_cards("AsKd").expect("valid");
        let board = crate::card::parse_cards("2c 7d 9h").expect("valid");

        let view = View {
            hole: [hole[0], hole[1]],
            board: &board,
            street: Street::Flop,
            position: Position::Button,
            pot: 200,
            to_call: 0,
            stack: 1_000,
            opponent_stack: 1_000,
            big_blind: 100,
            legal: &legal,
        };
        assert_eq!(bot.stage_for(&view), None, "there is no solved flop yet");
    }

    #[test]
    fn a_blueprint_with_nothing_in_it_still_plays() {
        // An empty blueprint means every spot falls back — the bot must play on
        // rather than fail, and the coverage number must say so plainly.
        let empty = Blueprint::from_profile(&Default::default(), "empty");
        let mut bot = BlueprintAgent::new("empty", empty, Sizing::default());
        let mut opponent = AlwaysCall;
        let mut rng = Rng::new(4);

        duplicate_match(&table(), &mut bot, &mut opponent, 200, &mut rng);
        assert_eq!(bot.coverage_fraction(), 0.0, "nothing was ever looked up");
        assert!(bot.coverage().1 > 0, "but it kept playing");
    }
}
