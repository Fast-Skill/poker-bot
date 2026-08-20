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
use crate::postflop::{self, Postflop, Spot};
use crate::ring::{Move, Ring};
use crate::rng::Rng;
use crate::table::{Agent, Position, View};
use crate::texture::Reader;
use crate::telemetry::{Confidence, DecisionRecord, Observer, Perception, Source};

/// Turns a solved move into something the table will accept.
///
/// The size comes from the tree rather than being recomputed here, so that a
/// blueprint which said "bet small" bets the amount it was solved for. What is
/// added is the table's own rules: a raise has a legal minimum and maximum, and
/// a size the tree named may fall outside them when the pot is small next to
/// the blind. Clamping keeps the action — a small bet stays the smallest bet on
/// offer — where refusing would hand an ordinary spot to the fallback.
///
/// `None` when even the clamped action is not legal, which should not happen
/// and is not worth guessing through if it does.
fn table_action(
    view: &View,
    game: &Postflop,
    spot: &Spot,
    chosen: postflop::Move,
) -> Option<Action> {
    let action = match chosen {
        postflop::Move::Fold => Action::Fold,
        postflop::Move::Passive => {
            if view.to_call == 0 {
                Action::Check
            } else {
                Action::Call
            }
        }
        postflop::Move::Small | postflop::Move::Large | postflop::Move::Jam => {
            let (least, most) = view.legal.raise_to?;
            Action::RaiseTo(game.target_of(spot, chosen).clamp(least, most))
        }
    };
    view.legal.permits(action).then_some(action)
}


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

/// A postflop solve and the depth it was solved at.
///
/// # Why there is a ladder rather than one solve
///
/// What can happen after the flop is set by how much is behind relative to what
/// is already in the middle. A pot of ten with ten behind offers exactly the
/// decisions a pot of a thousand with a thousand behind does, and neither
/// resembles a pot of ten with four hundred behind. So the solves are indexed
/// by that ratio and a real pot is routed to the nearest.
///
/// Preflop decides which rungs matter. A four-bet pot leaves roughly 1.7 behind
/// per unit of pot, a three-bet pot about 5, a single raise about 18, a limped
/// pot more still.
#[derive(Debug)]
struct Rung {
    /// Stack-to-pot ratio this was solved at.
    spr: f64,
    blueprint: Blueprint,
    game: Postflop,
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
    /// Solved postflop strategies, ascending by the depth they were solved at.
    ///
    /// Heads-up only. A pot with three players still in it after the flop is a
    /// game this tree does not model, and consulting it anyway would price a
    /// bet against one opponent's range when there are two — so those fall to
    /// the fallback and are counted.
    postflop: Vec<Rung>,
    /// Ranks the hero's hand on the board actually showing.
    ///
    /// The sampled boards a solve trained on are twenty thousand of two and a
    /// half million, so the board in play is almost never among them. What
    /// carries across is the procedure, not the table: this repeats it.
    reader: Option<Reader>,
    /// Covers postflop, and any preflop spot the blueprint does not hold.
    fallback: ChartBot,
    /// Preflop decisions taken so far this hand, used to tell an open from a
    /// response to a re-raise.
    preflop_decisions: u32,
    decisions: u64,
    fallbacks: u64,
    /// Why the fallback was reached, counted by reason.
    ///
    /// A single coverage percentage says how much of a session the solve
    /// decided but not what the rest was, and the two have very different
    /// remedies: a preflop line past the ladder wants a deeper solve, a
    /// multiway flop wants a tree that models one, and a spot the solve simply
    /// never visited wants more iterations. Without this they are one number.
    reasons: std::collections::BTreeMap<&'static str, u64>,
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
            postflop: Vec::new(),
            reader: None,
            blueprint,
            sizing,
            fallback: ChartBot::default(),
            preflop_decisions: 0,
            decisions: 0,
            fallbacks: 0,
            reasons: std::collections::BTreeMap::new(),
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

    /// Attaches a solved postflop strategy for one stack-to-pot ratio.
    ///
    /// The `game` must be the one the blueprint was solved from — same sizes,
    /// same strength abstraction — because the blueprint is keyed against that
    /// game's information sets and a mismatch looks up the wrong ones while
    /// appearing to work.
    ///
    /// Call it once per rung, in any order.
    ///
    /// # Panics
    /// Panics if the rungs disagree about how many strength groups there are.
    /// They index the same abstraction, and two solves that cut hand strength
    /// differently cannot be routed between: the same hand would be a different
    /// number depending on which rung answered.
    pub fn with_postflop(mut self, spr: f64, blueprint: Blueprint, game: Postflop) -> BlueprintAgent {
        let buckets = game.buckets();
        match &self.reader {
            Some(reader) => assert_eq!(
                reader.buckets(),
                buckets,
                "postflop rungs disagree on how many strength groups there are"
            ),
            None => self.reader = Some(Reader::new(buckets)),
        }
        self.postflop.push(Rung {
            spr,
            blueprint,
            game,
        });
        self.postflop
            .sort_by(|a, b| a.spr.total_cmp(&b.spr));
        self
    }

    /// The stack-to-pot ratios this agent has a postflop solve for, ascending.
    pub fn solved_depths(&self) -> Vec<f64> {
        self.postflop.iter().map(|rung| rung.spr).collect()
    }

    /// Which rung is nearest the depth a pot is actually being played at.
    ///
    /// Nearest by ratio rather than by difference, because depths grow
    /// multiplicatively. Between rungs at 3 and 12, a pot at 6 sits nearer 12
    /// by difference while being the same distance from either as a price.
    fn rung_for(&self, spr: f64) -> Option<&Rung> {
        self.postflop.iter().min_by(|a, b| {
            let gap = |rung: &Rung| (rung.spr / spr).max(spr / rung.spr);
            gap(a).total_cmp(&gap(b))
        })
    }

    /// The situation a view describes, as the postflop tree would describe it.
    ///
    /// `None` when the view is not one this tree models: not postflop, not
    /// heads-up, or a board that did not read.
    fn spot_for(&mut self, view: &View) -> Option<Spot> {
        if view.street == Street::Preflop || view.active != 2 {
            return None;
        }
        // The street and the board must be the same fact stated twice. They
        // are read separately — one from the cards on the felt, one from the
        // engine — and if they ever disagree, the strength would be measured on
        // one street and keyed against another. That is a misplay no log would
        // show, so it is refused rather than guessed through.
        if view.board.len() != view.street.board_cards() {
            return None;
        }
        if !(3..=5).contains(&view.board.len()) {
            return None;
        }
        let strength = self.reader.as_mut()?.strength(view.board, view.hole)?;

        // Who acts last. Postflop the order runs from the seat after the button
        // round to the button itself, so the last live seat in that order has
        // position — and position, not the blind that was posted, is what the
        // tree means by player 0.
        let last = (1..view.players)
            .chain(std::iter::once(0))
            .rfind(|seat| view.live[*seat])?;

        let bet = view
            .street_committed
            .iter()
            .enumerate()
            .filter(|(seat, _)| view.live[*seat])
            .map(|(_, amount)| *amount)
            .max()
            .unwrap_or(0);
        let mine = *view.street_committed.get(view.seat)?;

        // The tree numbers the two sides by position; the table numbers them by
        // seat. `acted` is a bit per tree player, so it has to be translated,
        // not copied.
        let hero = usize::from(view.seat != last);
        let mut acted = 0u8;
        for (seat, done) in view.acted.iter().enumerate() {
            if !view.live[seat] || !done {
                continue;
            }
            acted |= 1 << usize::from(seat != last);
        }

        Some(Spot {
            street: view.street,
            player: hero,
            strength,
            pot: view.pot,
            bet,
            mine,
            behind: view.stack,
            // The one opponent left, since this only runs on heads-up pots.
            opponent_behind: view
                .stacks
                .iter()
                .enumerate()
                .filter(|(seat, _)| view.live[*seat] && *seat != view.seat)
                .map(|(_, stack)| *stack)
                .max()
                .unwrap_or(0),
            raises: view.raises,
            acted,
        })
    }

    /// Consults a postflop solve, or says why it could not.
    ///
    /// The reason matters as much as the answer. A bot that falls back on a
    /// third of its decisions is not playing the strategy anybody measured, and
    /// "no solve covered it" is not one problem but five, with five different
    /// remedies — a deeper ladder, a multiway tree, more iterations, a better
    /// board reader, or a bug in the translation. Naming the step that declined
    /// is what separates them.
    fn postflop_action(
        &mut self,
        view: &View,
        rng: &mut Rng,
    ) -> Result<Consulted, &'static str> {
        if view.street == Street::Preflop {
            return Err("preflop line beyond the solved ladder");
        }
        if self.postflop.is_empty() {
            return Err("no postflop solve loaded");
        }
        if view.active != 2 {
            return Err("postflop pot with more than two players");
        }
        let spot = self
            .spot_for(view)
            .ok_or("the board and the street did not agree")?;

        // The depth this pot is being played at, measured as the street opened
        // rather than as it stands. Measuring it mid-street would move the bot
        // between solves as bets went in, and a hand that consulted one strategy
        // to check and another to face the raise is playing neither.
        let settled = view.pot.saturating_sub(
            view.street_committed
                .iter()
                .enumerate()
                .filter(|(seat, _)| view.live[*seat])
                .map(|(_, amount)| *amount)
                .sum::<u64>(),
        );
        let spr = (spot.behind + spot.mine) as f64 / settled.max(1) as f64;
        let rung = self.rung_for(spr).ok_or("no rung near this stack depth")?;

        let moves = rung.game.moves_at(&spot);
        // The exact spot first, then the nearest prices and depths the solve
        // does know. An opponent is under no obligation to bet in the sizes the
        // tree was solved with, and refusing every price it has not met leaves
        // most of a real session to the heuristic.
        //
        // A candidate offering a different number of actions than the tree
        // offers here is a key meaning something else. Skipping it rather than
        // playing the wrong index is the whole reason this is a search and not
        // a single fallback.
        let (key, strategy) = rung
            .game
            .keys_near(&spot)
            .into_iter()
            .find_map(|key| {
                let strategy = rung.blueprint.strategy(key)?;
                (strategy.len() == moves.len()).then_some((key, strategy))
            })
            .ok_or("no price the solve knows is near this one")?;

        let chosen = rung
            .blueprint
            .sample(key, rng)
            .ok_or("the solve never visited this spot")?;
        let action = table_action(view, &rung.game, &spot, moves[chosen])
            .ok_or("the solved move is not one the table allows")?;
        let frequencies = moves
            .iter()
            .zip(strategy.iter())
            .map(|(mv, share)| (mv.name().to_string(), *share as f64))
            .collect();
        Ok((action, frequencies, key, 2))
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

    /// Why the fallback was reached, most common first.
    pub fn fallback_reasons(&self) -> Vec<(&'static str, u64)> {
        let mut counted: Vec<_> = self.reasons.iter().map(|(&why, &n)| (why, n)).collect();
        counted.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
        counted
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

        // Postflop, if a solve covers the spot. Tried after the preflop ring so
        // that the two never contend: they answer disjoint streets.
        let postflop = self.postflop_action(view, rng);
        let declined = postflop.as_ref().err().copied();
        if let Ok((action, frequencies, key, _)) = postflop {
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
                        spot: format!("heads-up {:?}", view.street).to_lowercase(),
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
        let reason = declined.unwrap_or("preflop line beyond the solved ladder");
        let mut source = Source::Fallback { reason };
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
                *self.reasons.entry(reason).or_insert(0) += 1;
                source = Source::Fallback { reason };
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
    use crate::texture::Textures;

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

    /// A solved postflop ladder actually decides postflop, end to end.
    ///
    /// # Why this is worth a slow test
    ///
    /// Every piece of the postflop path is tested on its own: the abstraction
    /// against its own training, the tree against its invariants, the
    /// translation against what a table permits. None of that says the pieces
    /// are joined. A routing condition that never fires, a key built one way
    /// and looked up another, a blueprint whose action count does not match the
    /// tree's — each leaves a bot that quietly plays the heuristic while a
    /// solve sits loaded and unused, and every other test still passes.
    ///
    /// So this solves a small tree, attaches it, plays real hands, and requires
    /// that the blueprint is what answered after the flop. The solve is
    /// deliberately tiny: what is under test is that the strategy is reached at
    /// all, not that it is any good.
    #[test]
    fn a_loaded_postflop_ladder_is_what_decides_after_the_flop() {
        /// Counts who answered, by street.
        #[derive(Debug, Default)]
        struct Tally {
            solved: u64,
            fallback: u64,
        }

        impl Observer for Tally {
            fn on_decision(&mut self, record: &DecisionRecord) {
                if record.perception.street == Street::Preflop {
                    return;
                }
                match record.source {
                    Source::Blueprint { .. } => self.solved += 1,
                    _ => self.fallback += 1,
                }
            }
        }

        // Shared so the tally can be read after the agent has been consumed.
        #[derive(Debug, Clone, Default)]
        struct Shared(std::rc::Rc<std::cell::RefCell<Tally>>);

        impl Observer for Shared {
            fn on_decision(&mut self, record: &DecisionRecord) {
                self.0.borrow_mut().on_decision(record);
            }
        }

        let textures = Textures::sample(24, 12, 0x51DE, 4);
        let sizing = postflop::Sizing::default();
        let game = Postflop::new(textures, 100, 400, sizing);
        let mut rng = Rng::new(0xC0FFEE);
        let mut solver = Solver::new(game);
        solver.train_sampled(60_000, &mut rng);
        let solved = Blueprint::from_solver(&solver, "postflop/spr4/b12");

        let tally = Shared::default();
        let mut bot = agent(100.0)
            .with_postflop(4.0, solved, Postflop::for_play(12, 100, 400, sizing))
            .watch(Box::new(tally.clone()));

        assert_eq!(bot.solved_depths(), vec![4.0]);

        let mut opponent = AlwaysCall;
        let report = duplicate_match(&table(), &mut bot, &mut opponent, 120, &mut rng);
        assert!(report.hands > 0, "no hands were played");

        let counts = tally.0.borrow();
        assert!(
            counts.solved + counts.fallback > 50,
            "only {} postflop decisions were reached, too few to say anything",
            counts.solved + counts.fallback
        );
        // Every pot here is heads-up, so the ladder should answer nearly all of
        // them. What is left is spots the solve never visited, which a sixty
        // thousand iteration tree will have some of.
        let share = counts.solved as f64 / (counts.solved + counts.fallback) as f64;
        assert!(
            share > 0.9,
            "the ladder decided only {:.0}% of postflop spots ({} of {}); \
             the routing is not reaching it",
            share * 100.0,
            counts.solved,
            counts.solved + counts.fallback
        );
    }

    /// Every move the postflop tree offers must be one the table will accept.
    ///
    /// # The failure this exists to catch
    ///
    /// The tree and the table describe the same game in different languages,
    /// and a solved strategy is only worth anything if the translation between
    /// them holds. When it does not, the bot does not crash — it names an
    /// action the client refuses, falls through to the heuristic, and plays
    /// badly while appearing to consult a solve. Nothing in a session log looks
    /// wrong.
    ///
    /// So this plays real hands and, at every postflop spot the tree claims to
    /// cover, asks it for its whole action list and checks each one against
    /// what the table says is legal.
    ///
    /// The probe raises as well as calling, which is the point: a version that
    /// only checked and called reached a hundred spots and tested nothing, all
    /// of them being the same undisturbed check-through. Betting is what
    /// produces prices, re-raises, and short stacks.
    ///
    /// No blueprint is needed. What is under test is the mapping, not the
    /// strategy, and a solve would only slow the test down.
    #[test]
    fn every_postflop_move_the_tree_offers_is_one_the_table_allows() {
        struct Probe {
            bot: BlueprintAgent,
            game: Postflop,
            checked: usize,
        }

        impl Agent for Probe {
            fn name(&self) -> &str {
                "probe"
            }

            fn act(&mut self, view: &View, rng: &mut Rng) -> Action {
                if let Some(spot) = self.bot.spot_for(view) {
                    assert_eq!(
                        spot.pot,
                        view.pot,
                        "the spot and the table disagree about the pot"
                    );
                    assert!(
                        spot.bet >= spot.mine,
                        "the spot has the actor wagering {} into a bet of {}",
                        spot.mine,
                        spot.bet
                    );
                    // The price is most of what decides a call, and it is the
                    // one number the tree and the table both compute
                    // independently. Capped at the stack, since calling for
                    // more than everything is calling for everything.
                    assert_eq!(
                        spot.owed().min(spot.behind),
                        view.to_call,
                        "the tree prices this call at {} where the table says {}",
                        spot.owed().min(spot.behind),
                        view.to_call
                    );
                    for chosen in self.game.moves_at(&spot) {
                        let action = table_action(view, &self.game, &spot, chosen);
                        assert!(
                            action.is_some(),
                            "the tree offers {} on the {:?} at {spot:?}, which the table refuses: {:?}",
                            chosen.name(),
                            view.street,
                            view.legal
                        );
                    }
                    self.checked += 1;
                }
                // Play on, and play widely: the offer is what is under test,
                // and it only varies if the betting does.
                let raise = view.legal.raise_to.map(|(least, most)| {
                    let span = most - least;
                    Action::RaiseTo(least + if span == 0 { 0 } else { rng.below(span + 1) })
                });
                match rng.below(6) {
                    0 if view.legal.can_fold => Action::Fold,
                    1 | 2 => raise.unwrap_or(if view.to_call == 0 {
                        Action::Check
                    } else {
                        Action::Call
                    }),
                    _ if view.to_call == 0 => Action::Check,
                    _ => Action::Call,
                }
            }
        }

        // A tiny sample: the boards only have to be real, since the strength
        // read off them is not what is being checked.
        let game = Postflop::new(
            Textures::sample(4, 12, 0x51DE, 4),
            100,
            400,
            postflop::Sizing::default(),
        );
        let mut probe = Probe {
            bot: agent(100.0).with_postflop(
                4.0,
                Blueprint::from_profile(&Default::default(), "empty"),
                Postflop::new(
                    Textures::sample(4, 12, 0x51DE, 4),
                    100,
                    400,
                    postflop::Sizing::default(),
                ),
            ),
            game,
            checked: 0,
        };

        let table = table();
        let mut rng = Rng::new(0xB0A7);
        let mut deck = crate::table::Deck::fresh();
        for seats in 2..=6 {
            for _ in 0..120 {
                deck.shuffle(&mut rng);
                let mut others: Vec<AlwaysCall> = (0..seats - 1).map(|_| AlwaysCall).collect();
                let mut agents: Vec<&mut dyn Agent> = vec![&mut probe];
                agents.extend(others.iter_mut().map(|a| a as &mut dyn Agent));
                table.play_hand(&mut agents, deck.hand_cards(seats), &mut rng);
            }
        }
        assert!(
            probe.checked > 100,
            "only {} postflop spots were reached, too few to have tested anything",
            probe.checked
        );
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
            street_committed: &[500, 1_000],
            acted: &[false, false],
            raises: 0,
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
            street_committed: &[0, 50, 100],
            acted: &[false, false, false],
            raises: 0,
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
            street_committed: &[0, 50, 100, 100],
            acted: &[false, false, false, false],
            raises: 0,
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
            street_committed: &[0, 50, 100],
            acted: &[false, false, false],
            raises: 0,
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
            "aces on the button should be decided by the three-handed solve, not fall through to the heuristic"
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
            street_committed: &[100, 100],
            acted: &[false, false],
            raises: 0,
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
