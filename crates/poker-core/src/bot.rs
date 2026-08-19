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
use crate::ring::{Move, Ring};
use crate::rng::Rng;
use crate::table::{Agent, Position, View};
use crate::telemetry::{Confidence, DecisionRecord, Observer, Perception, Source};

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
    /// Solved multiway preflop strategies, each with the game whose information
    /// keys it is stored against, indexed by how many seats the game has.
    ///
    /// Indexed by seats rather than by how many are still in the pot, because
    /// folding is something the tree already models: five-handed with two live
    /// is a node inside the five-handed solve, and the difference between that
    /// and a genuine heads-up game is the dead money the folded players left
    /// behind. Choosing by the live count would discard the solve that prices
    /// it properly.
    rings: Vec<Option<(Blueprint, Ring)>>,
    /// Covers postflop, and any preflop spot the blueprint does not hold.
    fallback: ChartBot,
    /// Preflop decisions taken so far this hand, used to tell an open from a
    /// response to a re-raise.
    preflop_decisions: u32,
    decisions: u64,
    fallbacks: u64,
    /// Which hand of the session this is, for the record stream.
    hand: u64,
    /// Optional watcher. Costs nothing when absent.
    observer: Option<Box<dyn Observer>>,
}

/// What a solved lookup yields: the action, how often each action was played,
/// the information set consulted so a watcher can replay the decision, and how
/// many seats the game it came from has.
type Consulted = (Action, Vec<(String, f64)>, u64, usize);

impl BlueprintAgent {
    /// Wraps a blueprint solved with `sizing`.
    ///
    /// The sizing must match the solve. A blueprint that decided to "raise"
    /// meant a specific amount, and betting a different one plays a strategy
    /// nobody solved for.
    pub fn new(name: impl Into<String>, blueprint: Blueprint, sizing: Sizing) -> BlueprintAgent {
        BlueprintAgent {
            name: name.into(),
            rings: vec![None; crate::wide::MAX_PLAYERS + 1],
            blueprint,
            sizing,
            fallback: ChartBot::default(),
            preflop_decisions: 0,
            decisions: 0,
            fallbacks: 0,
            hand: 0,
            observer: None,
        }
    }

    /// Attaches a solved multiway preflop strategy.
    ///
    /// Without one for a given pot size, pots of that size fall through to the
    /// heuristic: a blueprint solved for a different number of players has
    /// nothing useful to say, and using it anyway is worse than not trying.
    ///
    /// The `ring` must be the game the blueprint was solved from — same seats,
    /// same stack, same ladder — because the blueprint is keyed against that
    /// game's information sets and a mismatch would look up the wrong ones.
    ///
    /// Call it once per size; each replaces only its own.
    pub fn with_ring(mut self, blueprint: Blueprint, ring: Ring) -> BlueprintAgent {
        let seats = ring.seats();
        self.rings[seats] = Some((blueprint, ring));
        self
    }

    /// The pot sizes this agent has a solved strategy for, ascending.
    pub fn solved_sizes(&self) -> Vec<usize> {
        (0..self.rings.len())
            .filter(|&seats| self.rings[seats].is_some())
            .collect()
    }

    /// Looks up a three-handed preflop decision, if one applies here.
    ///
    /// Every field the key needs is on the table — who is live, what each has
    /// pushed out, whose turn it is — so this needs no memory of how the
    /// betting arrived here. See [`Ring::key_from_table`].
    fn ring_action(&self, view: &View, rng: &mut Rng) -> Option<Consulted> {
        if view.street != Street::Preflop {
            return None;
        }
        // Chosen by the size of the game, not by how many are still in the pot.
        //
        // The tree models folding, so a pot that has come down to two players
        // at a seven-handed table is a node inside the seven-handed solve —
        // and one that prices the folded blinds correctly. Selecting by the
        // live count instead would look for a two-handed solve, and either find
        // nothing or find one that assumes no dead money.
        let (blueprint, ring) = self.rings.get(view.players)?.as_ref()?;

        let committed: Vec<f64> = view
            .committed
            .iter()
            .map(|&chips| chips as f64 / view.big_blind as f64)
            .collect();
        let class = HandClass::from_cards(view.hole[0], view.hole[1]);
        let key = ring.key_from_table(view.seat, class, view.live, &committed)?;
        let moves = ring.moves_at(view.seat, view.live, &committed)?;
        let strategy = blueprint.strategy(key)?;

        let frequencies: Vec<(String, f64)> = moves
            .iter()
            .zip(strategy.iter())
            .map(|(m, p)| (format!("{m:?}").to_lowercase(), *p as f64))
            .collect();

        // Sampled rather than taken at the mode, for the same reason as the
        // heads-up path: a mixed strategy played greedily is a different, and
        // more exploitable, strategy.
        let index = blueprint.sample(key, rng)?;
        let chosen = *moves.get(index)?;
        let action = match chosen {
            Move::Fold => Action::Fold,
            Move::Passive if view.to_call == 0 => Action::Check,
            Move::Passive => Action::Call,
            Move::Raise | Move::Jam => {
                let to = ring.raise_target(chosen, view.seat, view.live, &committed)?;
                Action::RaiseTo((to * view.big_blind as f64).round() as u64)
            }
        };
        // The solved size can fall outside what this table allows — a shorter
        // stack than the solve assumed, say. Falling through beats sending an
        // action the table will reject.
        view.legal
            .permits(action)
            .then_some((action, frequencies, key, ring.seats()))
    }

    /// Attaches a watcher that receives every decision.
    ///
    /// This is how the bot is made visible: the observer sees what was
    /// perceived, which spot was identified, the frequencies considered, and
    /// what was played — enough to check the bot's reading against the table
    /// without trusting it.
    pub fn watch(mut self, observer: Box<dyn Observer>) -> BlueprintAgent {
        self.observer = Some(observer);
        self
    }

    /// The attached observer, for reading session totals.
    pub fn observer(&self) -> Option<&dyn Observer> {
        self.observer.as_deref()
    }

    /// Names the spot for a stage, as a watcher would read it.
    fn spot_name(stage: preflop::Stage) -> &'static str {
        match stage {
            preflop::Stage::SbOpen => "sb-open",
            preflop::Stage::BbVsOpen => "bb-vs-open",
            preflop::Stage::SbVs3Bet => "sb-vs-3bet",
            preflop::Stage::BbVs4Bet => "bb-vs-4bet",
            preflop::Stage::SbVsJam => "sb-vs-jam",
            preflop::Stage::BbVsJam => "bb-vs-jam",
            _ => "unknown",
        }
    }

    /// A readable name for an abstract action.
    fn action_name(action: Abstract) -> &'static str {
        match action {
            Abstract::Fold => "fold",
            Abstract::Passive => "call",
            Abstract::Raise => "raise",
            Abstract::Jam => "jam",
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

        // The blueprint is a *two-player* solve. It applies exactly when the
        // pot is heads-up and the two players are the blinds — which is the
        // same game whether the table seats two or seven. Any other shape,
        // including a heads-up pot with folded players' blinds already dead in
        // the middle, prices differently and goes to the fallback rather than
        // being played by a strategy solved for a different game.
        if view.active != 2 {
            return None;
        }
        let button = match view.position {
            Position::Button | Position::SmallBlind => true,
            Position::BigBlind => false,
            Position::Middle => return None,
        };

        // An opponent with nothing behind has moved all in, whatever the
        // betting looked like before that.
        let opponent_all_in = view
            .stacks
            .iter()
            .enumerate()
            .any(|(seat, stack)| seat != view.seat && *stack == 0);
        if opponent_all_in && view.to_call > 0 {
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
        self.hand += 1;
    }

    fn act(&mut self, view: &View, rng: &mut Rng) -> Action {
        self.decisions += 1;

        // Three-handed preflop has its own solve, and it is consulted before
        // the heads-up one because the heads-up blueprint has nothing to say
        // about a three-way pot — it would fall through to the heuristic.
        if let Some((action, frequencies, key, seats)) = self.ring_action(view, rng) {
            self.preflop_decisions += 1;
            if let Some(observer) = self.observer.as_mut() {
                observer.on_decision(&DecisionRecord {
                    hand: self.hand,
                    perception: Perception {
                        hole: view.hole,
                        board: view.board.to_vec(),
                        street: view.street,
                        position: view.position,
                        pot: view.pot,
                        to_call: view.to_call,
                        stacks: view.stacks.to_vec(),
                        confidence: Confidence::certain(),
                    },
                    source: Source::Blueprint {
                        key,
                        // Named for the game it came from. This said
                        // "three-handed" whatever the table was, which made a
                        // live watch read as though one solve was answering
                        // every size of game.
                        spot: format!("{seats}-handed preflop"),
                    },
                    action,
                    frequencies,
                });
            }
            return action;
        }

        // Resolve the spot first, so a watcher can be told which one it was
        // even when the lookup then fails.
        let stage = self.stage_for(view);
        let mut source = Source::Fallback {
            reason: match view.street {
                Street::Preflop => "preflop line beyond the solved ladder",
                _ => "no postflop solve yet",
            },
        };
        let mut frequencies = Vec::new();

        let action = stage.and_then(|stage| {
            let class = HandClass::from_cards(view.hole[0], view.hole[1]);
            let key = Preflop::info_key(stage, class.index());
            let strategy = self.blueprint.strategy(key)?;
            let actions = self.actions_at(stage);

            frequencies = actions
                .iter()
                .zip(strategy.iter())
                .map(|(action, probability)| {
                    (
                        BlueprintAgent::action_name(*action).to_string(),
                        *probability as f64,
                    )
                })
                .collect();

            // Sampling, not the modal action: a mixed strategy played greedily
            // is a different and more exploitable strategy.
            let index = self.blueprint.sample(key, rng)?;
            let chosen = *actions.get(index)?;
            let concrete = self.concrete(chosen, stage, view)?;

            source = Source::Blueprint {
                key,
                spot: BlueprintAgent::spot_name(stage).to_string(),
            };
            Some(concrete)
        });

        if view.street == Street::Preflop {
            self.preflop_decisions += 1;
        }

        // Anything unrecognised or unplayable goes to the heuristic. It is
        // counted, because a bot silently falling back on most of its
        // decisions is not playing the strategy anybody benchmarked.
        let played = match action {
            Some(action) if view.legal.permits(action) => action,
            _ => {
                self.fallbacks += 1;
                source = Source::Fallback {
                    reason: match view.street {
                        Street::Preflop => "preflop line beyond the solved ladder",
                        _ => "no postflop solve yet",
                    },
                };
                frequencies.clear();
                self.fallback.act(view, rng)
            }
        };

        if let Some(observer) = self.observer.as_mut() {
            observer.on_decision(&DecisionRecord {
                hand: self.hand,
                perception: Perception {
                    hole: view.hole,
                    board: view.board.to_vec(),
                    street: view.street,
                    position: view.position,
                    pot: view.pot,
                    to_call: view.to_call,
                    stacks: view.stacks.to_vec(),
                    // Self-play deals the cards, so nothing was inferred. A
                    // vision layer replaces this with real match scores and
                    // every display already understands them.
                    confidence: Confidence::certain(),
                },
                source,
                action: played,
                frequencies,
            });
        }

        played
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
            seat: 0,
            players: 2,
            active: 2,
            pot: 2_000,
            to_call: 500,
            stack: 1_000,
            stacks: &[1_000, 0],
            committed: &[500, 1_000],
            live: &[true, true],
            big_blind: 100,
            legal: &legal,
        };
        assert_eq!(bot.stage_for(&view), Some(preflop::Stage::SbVsJam));
    }

    /// A pot that has folded down to two still uses the table's own solve.
    ///
    /// This was the bug the first live dry run exposed: the blueprint was
    /// chosen by how many were left in the pot, so a heads-up pot at a
    /// multiway table found no solve and fell through to the heuristic —
    /// throwing away a tree that models exactly that node, dead blinds and all.
    #[test]
    fn a_pot_folded_down_to_two_still_uses_the_tables_own_solve() {
        use crate::cfr::Solver;
        use crate::pushfold::EquityTable;
        use crate::ring::{Ladder, Ring};
        use crate::threeway::ThreeWayEquity;

        let ring = Ring::new(
            3,
            100.0,
            Ladder::default(),
            crate::wide::Showdown::new(
                EquityTable::sampled_parallel(8, 0x51DE, 4),
                ThreeWayEquity::sampled_parallel(1, 0x3EED, 4),
            ),
        );
        let mut rng = Rng::new(9);
        let mut solver = Solver::new(ring.clone());
        solver.train_sampled(40_000, &mut rng);
        let mut bot =
            agent(100.0).with_ring(Blueprint::from_solver(&solver, "ring3/100bb"), ring);

        // Three seats, the button folded: two live, and the small blind acts.
        let hole = crate::card::parse_cards("AsAd").expect("valid");
        let legal = crate::betting::LegalActions {
            can_fold: true,
            can_check: false,
            call_cost: Some(50),
            raise_to: Some((200, 10_000)),
        };
        let view = View {
            hole: [hole[0], hole[1]],
            board: &[],
            street: Street::Preflop,
            position: Position::SmallBlind,
            seat: 1,
            players: 3,
            active: 2,
            pot: 150,
            to_call: 50,
            stack: 9_950,
            stacks: &[10_000, 9_950, 9_900],
            committed: &[0, 50, 100],
            live: &[false, true, true],
            big_blind: 100,
            legal: &legal,
        };
        let (solved_before, _) = bot.coverage();
        bot.act(&view, &mut rng);
        let (solved_after, _) = bot.coverage();
        assert_eq!(
            solved_after,
            solved_before + 1,
            "a heads-up pot at a three-handed table is inside the three-handed solve"
        );
    }

    #[test]
    fn a_pot_size_with_no_solve_falls_through_rather_than_borrowing_one() {
        use crate::cfr::Solver;
        use crate::pushfold::EquityTable;
        use crate::ring::{Ladder, Ring};
        use crate::threeway::ThreeWayEquity;

        let ring = Ring::new(
            3,
            100.0,
            Ladder::default(),
            crate::wide::Showdown::new(
                EquityTable::sampled_parallel(8, 0x51DE, 4),
                ThreeWayEquity::sampled_parallel(1, 0x3EED, 4),
            ),
        );
        let mut rng = Rng::new(4);
        let mut solver = Solver::new(ring.clone());
        solver.train_sampled(5_000, &mut rng);
        let bot = agent(100.0).with_ring(Blueprint::from_solver(&solver, "ring3/100bb"), ring);

        assert_eq!(bot.solved_sizes(), vec![3], "only three-handed is solved");

        // A four-seat game has no strategy here. Reaching for the three-handed
        // one would be answering a different question with confidence.
        let mut bot = bot;
        let hole = crate::card::parse_cards("AsAd").expect("valid");
        let legal = crate::betting::LegalActions {
            can_fold: true,
            can_check: false,
            call_cost: Some(100),
            raise_to: Some((200, 10_000)),
        };
        let view = View {
            hole: [hole[0], hole[1]],
            board: &[],
            street: Street::Preflop,
            position: Position::Button,
            seat: 0,
            players: 4,
            active: 4,
            pot: 250,
            to_call: 100,
            stack: 10_000,
            stacks: &[10_000; 4],
            committed: &[0, 50, 100, 100],
            live: &[true, true, true, true],
            big_blind: 100,
            legal: &legal,
        };
        let (solved_before, _) = bot.coverage();
        bot.act(&view, &mut rng);
        let (solved_after, total) = bot.coverage();
        assert_eq!(total, 1);
        assert_eq!(
            solved_after, solved_before,
            "a four-seat game must not be answered from a three-handed solve"
        );
    }

    /// A three-handed pot must reach the three-handed solve, not the heuristic.
    #[test]
    fn a_three_handed_preflop_pot_is_decided_by_the_ring_blueprint() {
        use crate::cfr::Solver;
        use crate::pushfold::EquityTable;
        use crate::ring::{Ladder, Ring};
        use crate::threeway::ThreeWayEquity;

        // A small solve: this checks the routing, not the strategy.
        let ring = Ring::new(
            3,
            100.0,
            Ladder::default(),
            crate::wide::Showdown::new(
                EquityTable::sampled_parallel(8, 0x51DE, 4),
                ThreeWayEquity::sampled_parallel(1, 0x3EED, 4),
            ),
        );
        let mut rng = Rng::new(4);
        let mut solver = Solver::new(ring.clone());
        solver.train_sampled(20_000, &mut rng);
        let solved = Blueprint::from_solver(&solver, "ring3/100bb");

        let mut bot = agent(100.0).with_ring(solved, ring);
        let hole = crate::card::parse_cards("AsAd").expect("valid");
        // The button's own spot, stated directly rather than borrowed from an
        // unrelated round: facing the big blind, free to raise up to its stack.
        let legal = crate::betting::LegalActions {
            can_fold: true,
            can_check: false,
            call_cost: Some(100),
            raise_to: Some((200, 10_000)),
        };

        // The button, first to act three-handed, with the blinds posted.
        let view = View {
            hole: [hole[0], hole[1]],
            board: &[],
            street: Street::Preflop,
            position: Position::Button,
            seat: 0,
            players: 3,
            active: 3,
            pot: 150,
            to_call: 100,
            stack: 10_000,
            stacks: &[10_000, 9_950, 9_900],
            committed: &[0, 50, 100],
            live: &[true, true, true],
            big_blind: 100,
            legal: &legal,
        };

        // `coverage` reports (decided by a blueprint, decided at all).
        let (solved_before, _) = bot.coverage();
        let action = bot.act(&view, &mut rng);
        let (solved_after, total) = bot.coverage();
        assert_eq!(total, 1, "one decision was asked for");
        assert_eq!(
            solved_after,
            solved_before + 1,
            "aces on the button should be decided by the three-handed solve,              not fall through to the heuristic"
        );
        assert!(
            legal.permits(action),
            "the solved action {action:?} must be legal here"
        );
        assert_ne!(action, Action::Fold, "aces are not a fold");
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
            seat: 0,
            players: 2,
            active: 2,
            pot: 200,
            to_call: 0,
            stack: 1_000,
            stacks: &[1_000, 1_000],
            committed: &[100, 100],
            live: &[true, true],
            big_blind: 100,
            legal: &legal,
        };
        assert_eq!(bot.stage_for(&view), None, "there is no solved flop yet");
    }

    #[test]
    fn a_watcher_sees_what_the_bot_saw_and_why_it_acted() {
        use crate::telemetry::ConsoleMonitor;
        use std::cell::RefCell;
        use std::rc::Rc;

        /// Captures the stream instead of printing it, sharing the log with the
        /// test so it can be inspected after play.
        #[derive(Debug)]
        struct Capture(Rc<RefCell<Vec<DecisionRecord>>>);

        impl Observer for Capture {
            fn on_decision(&mut self, record: &DecisionRecord) {
                self.0.borrow_mut().push(record.clone());
            }
        }

        let log = Rc::new(RefCell::new(Vec::new()));
        let mut bot = agent(100.0).watch(Box::new(Capture(Rc::clone(&log))));
        let mut opponent = AlwaysCall;
        let mut rng = Rng::new(11);
        duplicate_match(&table(), &mut bot, &mut opponent, 30, &mut rng);

        let records = log.borrow();
        assert!(!records.is_empty(), "the watcher saw nothing");

        // Whatever the stream produces, the console renderer must handle it.
        let monitor = ConsoleMonitor::new(100);
        for record in records.iter() {
            assert_ne!(
                record.perception.hole[0], record.perception.hole[1],
                "a holding cannot repeat a card"
            );
            assert!(record.hand >= 1, "hands are numbered from one");
            let text = monitor.render(record);
            assert!(text.contains("SEE"), "{text}");
            assert!(text.contains("DO"), "{text}");
        }

        // Preflop decisions should be credited to the solved strategy...
        assert!(
            records.iter().any(|record| record.source.is_blueprint()),
            "nothing was credited to the blueprint"
        );
        // ...and postflop ones should say plainly that they were not.
        assert!(
            records.iter().any(|record| matches!(
                record.source,
                Source::Fallback { .. }
            )),
            "against a caller, postflop spots must show as fallbacks"
        );
    }

    #[test]
    fn a_blueprint_decision_reports_the_frequencies_it_chose_from() {
        use std::cell::RefCell;
        use std::rc::Rc;

        #[derive(Debug)]
        struct Capture(Rc<RefCell<Vec<DecisionRecord>>>);
        impl Observer for Capture {
            fn on_decision(&mut self, record: &DecisionRecord) {
                self.0.borrow_mut().push(record.clone());
            }
        }

        let log = Rc::new(RefCell::new(Vec::new()));
        let mut bot = agent(100.0).watch(Box::new(Capture(Rc::clone(&log))));
        let mut opponent = AlwaysFold;
        let mut rng = Rng::new(12);
        duplicate_match(&table(), &mut bot, &mut opponent, 30, &mut rng);

        let records = log.borrow();
        let solved: Vec<&DecisionRecord> = records
            .iter()
            .filter(|record| record.source.is_blueprint())
            .collect();
        assert!(!solved.is_empty());

        for record in solved {
            // Without these, a hand folded at its 5% frequency looks like a bug
            // rather than correct play.
            assert!(!record.frequencies.is_empty(), "no frequencies recorded");
            let total: f64 = record.frequencies.iter().map(|(_, p)| p).sum();
            assert!((total - 1.0).abs() < 1e-5, "frequencies summed to {total}");
        }
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

#[cfg(test)]
mod multiway_tests {
    use super::*;
    use crate::bench::{ring_match, ChartBot};
    use crate::blueprint::Blueprint;
    use crate::cfr::Solver;
    use crate::preflop::Preflop;
    use crate::pushfold::EquityTable;
    use crate::table::{Agent, Table};

    fn trained() -> Blueprint {
        let equity = EquityTable::sampled_parallel(300, 0x51DE, 4);
        let mut rng = Rng::new(0xF01D);
        let mut solver = Solver::new(Preflop::new(100.0, Sizing::default(), equity));
        solver.train_sampled(200_000, &mut rng);
        Blueprint::from_solver(&solver, "preflop/100bb")
    }

    /// Plays `seats`-handed and reports what share of decisions the two-player
    /// blueprint was able to make.
    fn coverage_at(seats: usize, hands: u64) -> f64 {
        let mut hero = BlueprintAgent::new("bot", trained(), Sizing::default());
        let mut others: Vec<ChartBot> = (0..seats - 1).map(|_| ChartBot::default()).collect();
        let mut refs: Vec<&mut dyn Agent> = vec![&mut hero];
        for other in others.iter_mut() {
            refs.push(other as &mut dyn Agent);
        }
        let mut rng = Rng::new(0xC0DE);
        ring_match(&Table::standard(), refs, hands, &mut rng);
        hero.coverage_fraction()
    }

    #[test]
    fn the_bot_plays_legally_at_every_table_size() {
        // The table panics on an illegal action, so surviving this is the
        // assertion. Two-player routing must not produce an illegal bet when
        // the fallback takes over multiway.
        for seats in 2..=6 {
            let share = coverage_at(seats, 200);
            assert!((0.0..=1.0).contains(&share), "{seats}-handed gave {share}");
        }
    }

    #[test]
    fn blueprint_coverage_falls_as_the_table_fills() {
        // The measurement that decides whether multiway preflop is urgent: a
        // two-player solve applies to fewer and fewer spots as more players
        // are dealt in.
        let heads_up = coverage_at(2, 400);
        let six_handed = coverage_at(6, 400);

        assert!(
            heads_up > 0.5,
            "heads-up should be almost fully covered, got {heads_up:.3}"
        );
        assert!(
            heads_up > six_handed,
            "coverage should fall with more players: {heads_up:.3} vs {six_handed:.3}"
        );
    }
}
