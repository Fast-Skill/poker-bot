//! Preflop strategy read from published charts rather than solved.
//!
//! # Why these are read and not derived
//!
//! Preflop is the most thoroughly solved part of poker, and the answers are
//! published. Our own solve reached 845,000 information sets with about a
//! quarter of them still untrained after the compute budget ran out, and it
//! disagreed with itself in ways a chart never does — folding king-queen
//! offsuit 98% of the time while calling king-eight offsuit 66%. Getting the
//! last quarter trained would take hundreds of hours to arrive at something
//! already written down.
//!
//! So preflop is read. The solver's own effort is worth its cost after the
//! flop, where board texture makes charts impractical and there is nothing
//! published to read.
//!
//! # What a chart is here
//!
//! One spot — a position, and what has happened in front of it — and one file
//! per action available in it. A file lists the hands that take that action and
//! how often, in the combination-by-combination form solvers export (see
//! [`Range::from_combos`]).
//!
//! Whatever weight is left over after every named action is folding. That makes
//! a single-action chart complete and unambiguous: an opening range says raise
//! this often, and the rest is a fold, which is the whole of the strategy in an
//! unopened pot. It also makes an incomplete chart detectable rather than
//! silently wrong — see [`Charts::gaps`].

use std::collections::BTreeMap;
use std::path::Path;

use crate::abstraction::HandClass;
use crate::range::Range;
use crate::ring::Move;
use crate::betting::Street;
use crate::table::View;

/// Positions, named as six-handed play names them.
///
/// Charts are published per position, and six-handed is the size they are
/// published for. Larger tables map onto these names; see [`seat_position`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Seat {
    Utg,
    Hj,
    Co,
    Btn,
    Sb,
    Bb,
}

impl Seat {
    pub fn name(self) -> &'static str {
        match self {
            Seat::Utg => "utg",
            Seat::Hj => "hj",
            Seat::Co => "co",
            Seat::Btn => "btn",
            Seat::Sb => "sb",
            Seat::Bb => "bb",
        }
    }

    pub fn from_name(name: &str) -> Option<Seat> {
        Some(match name {
            "utg" => Seat::Utg,
            "hj" => Seat::Hj,
            "co" => Seat::Co,
            "btn" => Seat::Btn,
            "sb" => Seat::Sb,
            "bb" => Seat::Bb,
            _ => return None,
        })
    }
}

/// The names a chart file may carry for its action.
///
/// Deliberately not every move the tree knows: a chart covers the spots people
/// publish, and nobody publishes a preflop four-bet-jam chart per position. A
/// name outside this set is skipped rather than guessed at.
fn action_named(name: &str) -> Option<Move> {
    Some(match name {
        "raise" => Move::Raise,
        "call" => Move::Passive,
        "jam" => Move::Jam,
        _ => return None,
    })
}

/// Which chart a decision belongs to.
///
/// Either an unopened pot, where only the acting position matters, or one
/// facing a single raise, where the raiser's position matters as much. Charts
/// for deeper trees — facing a three-bet, facing a four-bet — are separate
/// spots and simply are not covered until their files exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Spot {
    pub seat: Seat,
    /// The position that raised, if this is a chart for facing one.
    pub versus: Option<Seat>,
    /// Coming in cold: two players have already raised and this seat has put
    /// in nothing but a blind.
    ///
    /// # Why one chart for all of them
    ///
    /// There are hundreds of these — every combination of who opened, who
    /// three-bet, and who is now looking at it — and no published set covers
    /// them. But the correct strategy barely varies: facing a raise and a
    /// re-raise with nothing invested, almost everything folds, and playing
    /// only the very top of the deck gives up little against playing each spot
    /// exactly. That is the client's own judgement of the cost, and it is the
    /// difference between one file and several hundred exports.
    ///
    /// So `seat` and `versus` are ignored when this is set, and every cold spot
    /// shares one range.
    pub cold: bool,
}

impl Spot {
    /// The file stem a spot's charts are named with: `utg`, `btn-vs-co`,
    /// `cold`.
    pub fn name(self) -> String {
        if self.cold {
            return "cold".to_string();
        }
        match self.versus {
            Some(versus) => format!("{}-vs-{}", self.seat.name(), versus.name()),
            None => self.seat.name().to_string(),
        }
    }

    /// A cold spot, which every position shares.
    ///
    /// The seat is recorded but never read; see [`Spot::cold`].
    pub fn cold_spot(seat: Seat) -> Spot {
        Spot {
            seat,
            versus: None,
            cold: true,
        }
    }

    pub fn from_name(name: &str) -> Option<Spot> {
        if name == "cold" {
            return Some(Spot::cold_spot(Seat::Utg));
        }
        match name.split_once("-vs-") {
            Some((seat, versus)) => Some(Spot {
                seat: Seat::from_name(seat)?,
                versus: Some(Seat::from_name(versus)?),
                cold: false,
            }),
            None => Some(Spot {
                seat: Seat::from_name(name)?,
                versus: None,
                cold: false,
            }),
        }
    }
}

/// The strategy for one spot: how often each action is taken, hand by hand.
#[derive(Debug, Default, Clone)]
pub struct Chart {
    by_action: Vec<(Move, Range)>,
}

impl Chart {
    /// Adds an action's range, replacing any already held for that action.
    pub fn with(mut self, action: Move, range: Range) -> Chart {
        self.by_action.retain(|(held, _)| *held != action);
        self.by_action.push((action, range));
        self.by_action.sort_by_key(|(action, _)| order_of(*action));
        self
    }

    pub fn actions(&self) -> impl Iterator<Item = Move> + '_ {
        self.by_action.iter().map(|(action, _)| *action)
    }

    /// How often each action is taken with this hand, folding included.
    ///
    /// Fold is the remainder, and comes first so the list reads in the order
    /// the tree offers moves in. A hand no chart names folds outright, which is
    /// a real answer and not a gap: charts list what they play.
    pub fn strategy(&self, class: HandClass) -> Vec<(Move, f64)> {
        let mut played = Vec::with_capacity(self.by_action.len() + 1);
        let mut total = 0.0;
        for (action, range) in &self.by_action {
            let weight = range.weight(class);
            if weight > 0.0 {
                played.push((*action, weight));
                total += weight;
            }
        }

        // Charts are exported per action and rounded separately, so the parts
        // can add to a hair over one. Scaling down is right where dropping the
        // excess from one action would not be — the overshoot belongs to all of
        // them.
        if total > 1.0 {
            for (_, weight) in &mut played {
                *weight /= total;
            }
            total = 1.0;
        }

        let mut strategy = vec![(Move::Fold, 1.0 - total)];
        strategy.extend(played);
        strategy.retain(|(_, weight)| *weight > 0.0);
        strategy
    }
}

/// Sorts actions the way the tree lists them, so frequencies read consistently.
fn order_of(action: Move) -> u8 {
    match action {
        Move::Fold => 0,
        Move::Passive => 1,
        Move::Raise => 2,
        Move::Jam => 3,
    }
}

/// Every chart the bot holds, looked up by spot.
#[derive(Debug, Default, Clone)]
pub struct Charts {
    spots: BTreeMap<String, Chart>,
}

/// Below this many players the published six-handed charts do not apply.
///
/// A three-handed button opens a far wider range than a six-handed one, and
/// naming both "btn" would quietly play the tight range in the loose spot. The
/// short-handed solves already cover those tables and are the right strategy
/// there, so charts stand aside.
pub const FEWEST_SEATS: usize = 6;

impl Charts {
    pub fn new() -> Charts {
        Charts::default()
    }

    pub fn insert(&mut self, spot: Spot, chart: Chart) {
        self.spots.insert(spot.name(), chart);
    }

    pub fn get(&self, spot: Spot) -> Option<&Chart> {
        self.spots.get(&spot.name())
    }

    pub fn is_empty(&self) -> bool {
        self.spots.is_empty()
    }

    pub fn spots(&self) -> impl Iterator<Item = (&str, &Chart)> {
        self.spots
            .iter()
            .map(|(name, chart)| (name.as_str(), chart))
    }

    /// Reads every `<spot>.<action>.txt` in a directory.
    ///
    /// Files are named for what they hold — `utg.raise.txt`,
    /// `btn-vs-co.call.txt` — because a range on its own does not say which
    /// action it belongs to, and guessing would be the kind of mistake that
    /// looks like a working bot. Anything not matching the pattern is skipped
    /// quietly, so notes and exports-in-progress can sit in the same folder.
    pub fn load(from: &Path) -> Result<Charts, String> {
        let mut charts = Charts::new();
        let listing =
            std::fs::read_dir(from).map_err(|why| format!("{}: {why}", from.display()))?;

        let mut files: Vec<_> = listing
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|kind| kind == "txt"))
            .collect();
        files.sort();

        for path in files {
            let stem = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or_default()
                .to_ascii_lowercase();
            let Some((spot, action)) = stem.rsplit_once('.') else {
                continue;
            };
            let (Some(spot), Some(action)) = (Spot::from_name(spot), action_named(action)) else {
                continue;
            };

            let text = std::fs::read_to_string(&path)
                .map_err(|why| format!("{}: {why}", path.display()))?;
            let range = read_range(&text).map_err(|why| format!("{}: {why}", path.display()))?;

            let held = charts.get(spot).cloned().unwrap_or_default();
            charts.insert(spot, held.with(action, range));
        }
        Ok(charts)
    }

    /// Spots that are held but genuinely incomplete.
    ///
    /// # Where a single raising range is the whole strategy
    ///
    /// Almost everywhere. Raise-or-fold is complete in an unopened pot, and —
    /// less obviously — three-bet-or-fold is complete facing a raise from every
    /// position except the big blind. That is not a simplification but the
    /// equilibrium: once the rake is priced in, flat-calling a raise out of
    /// position is dominated, so the solutions simply have no calling range to
    /// export.
    ///
    /// The exported ranges agree. Button versus cutoff three-bets 11.5% of
    /// hands, which is wide for that spot and is what a position three-bets
    /// when it has no flatting range to fall back on.
    ///
    /// This was first written the other way round, flagging all nine
    /// three-betting charts as half-finished and sending someone off to export
    /// nine calling ranges that do not exist.
    ///
    /// # The big blind, which is the real exception
    ///
    /// It has already paid a blind and it closes the action, so it defends by
    /// calling far more than it raises. A big-blind chart holding only a
    /// three-betting range folds most of a range it should be playing — the one
    /// spot where a missing calling range is a real and expensive hole.
    ///
    /// Reported rather than refused, since a partial chart still beats none.
    pub fn gaps(&self) -> Vec<String> {
        self.spots
            .iter()
            .filter(|(name, chart)| {
                // A cold chart is one tight range and is finished as it is.
                if name.as_str() == "cold" {
                    return false;
                }
                let facing = name.contains("-vs-");
                let defending = name.starts_with("bb-vs-");
                let calls = chart.actions().any(|action| action == Move::Passive);
                let raises = chart.actions().any(|action| action == Move::Raise);
                // Only the big blind needs both. Everywhere else a raising
                // range on its own is the finished strategy.
                (defending && !calls) || (facing && !defending && !raises)
            })
            .map(|(name, _)| {
                if name.starts_with("bb-vs-") {
                    format!(
                        "{name}: no calling range. The big blind defends mostly by \
                         calling, so this folds most of the range it should play."
                    )
                } else {
                    format!("{name}: no three-betting range, and outside the big blind three-bet-or-fold is the whole strategy")
                }
            })
            .collect()
    }
}

/// Reads a range written either way a range gets written.
///
/// # Why both
///
/// A solver exports one entry per combination — `AcKd: 1,Ah5s: 0.22` — and that
/// is what the files here normally hold. But a range a person writes by hand is
/// in the notation people read: `22+, A2s+, KTo+`. Both describe a range, and a
/// folder of charts is a place where both turn up: exports for the spots
/// somebody has got round to, and something hand-written standing in for the
/// spots they have not.
///
/// The two are told apart by looking rather than by asking. A combination entry
/// names two exact cards and always carries a weight after a colon, which no
/// hand-written term does unless it is a weighted one — and a weighted term
/// like `AA:0.75` still names a hand class rather than two cards. So a colon
/// following four characters that parse as two cards means an export, and
/// anything else is notation.
///
/// A hand-written stand-in is worse than an export and should be replaced by
/// one. It is allowed because the alternative is not "wait for the export", it
/// is "play no strategy at all in that spot", and a rough range beats the
/// heuristic that folds most of a big blind.
fn read_range(text: &str) -> Result<Range, String> {
    let looks_exported = text
        .split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .take(4)
        .any(|entry| {
            entry
                .split_once(':')
                .is_some_and(|(cards, _)| cards.trim().len() == 4)
        });

    if looks_exported {
        Range::from_combos(text).map_err(|why| why.to_string())
    } else {
        text.parse::<Range>().map_err(|why| why.to_string())
    }
}

/// The six-handed name for a seat at a table of this size.
///
/// Seat zero holds the button and the rest follow it round, so a position is
/// really a distance from the button and the names fall out of that. The blinds
/// and the three seats before the button are exact at any size. Everything
/// earlier collapses onto under-the-gun: at seven-handed that puts UTG and
/// UTG+1 on the same chart, which is the standard simplification and the one
/// the ranges' own author suggested — the two spots differ by one player's
/// worth of risk and the published ranges for them are nearly identical.
///
/// The order matters. At four-handed the cutoff and the hijack would both land
/// on seats already named for blinds, so blinds and button are claimed first
/// and the later names only take what is left. Sizes below six do not use
/// charts at all — see [`FEWEST_SEATS`] — but the naming stays honest anyway.
pub fn seat_position(seat: usize, players: usize) -> Option<Seat> {
    if seat >= players || players < 2 {
        return None;
    }
    Some(match seat {
        0 => Seat::Btn,
        1 => Seat::Sb,
        2 => Seat::Bb,
        seat if seat == players - 1 => Seat::Co,
        seat if seat == players - 2 => Seat::Hj,
        _ => Seat::Utg,
    })
}

/// Which chart, if any, covers this decision.
///
/// Returns nothing when the spot is off the edge of what charts cover: after
/// the flop, at a table too small for six-handed ranges, or once the betting
/// has gone past a single raise. Those fall through to the solves, which is the
/// point of returning `None` rather than reaching for the nearest chart.
pub fn spot_of(view: &View) -> Option<Spot> {
    if view.street != Street::Preflop || view.players < FEWEST_SEATS {
        return None;
    }
    let seat = seat_position(view.seat, view.players)?;

    match view.raises {
        // Nobody has raised. A limp in front is not a raise, but it does change
        // the spot enough that an opening chart is the wrong answer, so an
        // unopened pot means exactly that: nothing out there but blinds.
        0 => {
            // What a seat has out there without having chosen to: the blinds.
            // Written against the blind sizes and not against what the seat has
            // committed, which was the first attempt and is a tautology — it
            // compared the small blind's chips to themselves, so a small blind
            // that limped read as one that had only posted, and the big blind
            // behind it was handed an opening chart for a pot that was no
            // longer unopened.
            let posted = |at: usize| match at {
                1 => view.big_blind / 2,
                2 => view.big_blind,
                _ => 0,
            };
            let limped = (0..view.players).any(|at| view.committed[at] > posted(at));
            (!limped).then_some(Spot {
                seat,
                versus: None,
                cold: false,
            })
        }
        1 => {
            // The raiser is whoever has put in the most. Reading it off the
            // table rather than remembering it keeps this working from a single
            // glance at the screen, which is all the live path ever has.
            let raiser = (0..view.players)
                .filter(|&at| at != view.seat && view.live[at])
                .max_by_key(|&at| view.committed[at])
                .filter(|&at| view.committed[at] > view.big_blind)?;
            let versus = seat_position(raiser, view.players)?;
            (versus != seat).then_some(Spot {
                seat,
                versus: Some(versus),
                cold: false,
            })
        }
        // Two or more raises. If this seat has put in nothing but a blind it is
        // coming in cold, and one chart covers every such spot. If it is
        // already invested the spot is its own — the opener answering a
        // three-bet, say — and nothing here covers that yet.
        _ => {
            let posted = match view.seat {
                1 => view.big_blind / 2,
                2 => view.big_blind,
                _ => 0,
            };
            (view.committed[view.seat] <= posted).then_some(Spot::cold_spot(seat))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn range(notation: &str) -> Range {
        notation.parse().expect("standard notation")
    }

    /// The seat numbers run one way round the table and the position names run
    /// the other, so this is the join between them and worth pinning exactly.
    #[test]
    fn positions_are_named_by_distance_from_the_button() {
        let named = |seat, players| seat_position(seat, players).expect("a seat");

        // Six-handed, where the names came from: one seat each.
        assert_eq!(named(0, 6), Seat::Btn);
        assert_eq!(named(1, 6), Seat::Sb);
        assert_eq!(named(2, 6), Seat::Bb);
        assert_eq!(named(3, 6), Seat::Utg);
        assert_eq!(named(4, 6), Seat::Hj);
        assert_eq!(named(5, 6), Seat::Co);

        // Seven-handed, which is what the club deals. The blinds and the three
        // seats before the button keep their exact meaning; the extra seat
        // lands on under-the-gun alongside the real one, which is the agreed
        // simplification.
        assert_eq!(named(0, 7), Seat::Btn);
        assert_eq!(named(1, 7), Seat::Sb);
        assert_eq!(named(2, 7), Seat::Bb);
        assert_eq!(named(3, 7), Seat::Utg, "under the gun");
        assert_eq!(named(4, 7), Seat::Utg, "and the seat after it, folded in");
        assert_eq!(named(5, 7), Seat::Hj);
        assert_eq!(named(6, 7), Seat::Co);

        // Nine-handed, which the client prefers: four early seats collapse.
        assert_eq!(named(6, 9), Seat::Utg);
        assert_eq!(named(7, 9), Seat::Hj);
        assert_eq!(named(8, 9), Seat::Co);

        assert_eq!(seat_position(6, 6), None, "no such seat");
    }

    /// Four-handed the cutoff and hijack would land on seats already spoken
    /// for, since there are not enough seats to go round. The blinds and the
    /// button must win those collisions, or a chart written for the small blind
    /// would be played on the button.
    ///
    /// Under the gun is the deliberate exception: it is the bucket every early
    /// seat collapses into, so it may name several at once.
    #[test]
    fn only_under_the_gun_may_name_more_than_one_seat() {
        for players in 2..=9 {
            let mut seen = Vec::new();
            for seat in 0..players {
                let named = seat_position(seat, players).expect("a seat");
                assert!(
                    named == Seat::Utg || !seen.contains(&named),
                    "{named:?} named twice at {players} players"
                );
                seen.push(named);
            }
            // However few seats there are, the blinds and the button are the
            // ones that must survive: those are the spots whose charts differ
            // most from each other.
            for (seat, expected) in [(0, Seat::Btn), (1, Seat::Sb), (2, Seat::Bb)] {
                if seat < players {
                    assert_eq!(seat_position(seat, players), Some(expected));
                }
            }
        }
    }

    /// A limp is money nobody was forced to put in, and the blinds were.
    ///
    /// The first version of this compared each blind's chips against its own
    /// chips, which is always equal, so a small blind that limped looked like
    /// one that had merely posted. The big blind behind it would then be given
    /// an opening chart — a raise-or-fold strategy — for a pot it could check
    /// into for free.
    #[test]
    fn a_limp_means_the_pot_is_no_longer_unopened() {
        let hole = crate::card::parse_cards("AhJc").expect("two cards");
        let legal = crate::betting::LegalActions {
            can_fold: true,
            can_check: true,
            call_cost: None,
            raise_to: Some((200, 10_000)),
        };
        let stacks = [10_000u64; 6];
        let live = [true; 6];
        let acted = [false; 6];
        // Six-handed, hero in the big blind, everyone folded round to the
        // small blind. Nothing out there but the two blinds.
        let blinds_only = [0, 50, 100, 0, 0, 0];
        let base = View {
            hole: [hole[0], hole[1]],
            board: &[],
            street: Street::Preflop,
            position: crate::table::Position::BigBlind,
            seat: 2,
            players: 6,
            active: 2,
            pot: 150,
            to_call: 0,
            stack: 9_900,
            stacks: &stacks,
            committed: &blinds_only,
            live: &live,
            street_committed: &blinds_only,
            acted: &acted,
            raises: 0,
            settled: None,
            big_blind: 100,
            legal: &legal,
        };
        assert_eq!(
            spot_of(&base).map(|spot| spot.name()),
            Some("bb".to_string()),
            "blinds alone leave the pot unopened"
        );

        // Now the small blind completes. That is a limp, and no chart here
        // covers a limped pot.
        let limped = [0, 100, 100, 0, 0, 0];
        assert!(
            spot_of(&View {
                committed: &limped,
                street_committed: &limped,
                ..base
            })
            .is_none(),
            "a completed small blind is a limp"
        );

        // And so is a limp from a seat with no blind to post.
        let open_limp = [0, 50, 100, 0, 0, 100];
        assert!(spot_of(&View {
            committed: &open_limp,
            street_committed: &open_limp,
            ..base
        })
        .is_none());
    }

    /// Facing a raise, the chart is chosen by who raised, read off the table.
    #[test]
    fn the_raiser_is_found_by_who_has_put_in_the_most() {
        let hole = crate::card::parse_cards("AhJc").expect("two cards");
        let legal = crate::betting::LegalActions {
            can_fold: true,
            can_check: false,
            call_cost: Some(150),
            raise_to: Some((450, 10_000)),
        };
        let stacks = [10_000u64; 6];
        // Six-handed, hero on the button, the cutoff has opened to 250.
        let committed = [0, 50, 100, 0, 0, 250];
        let live = [true, true, true, false, false, true];
        let acted = [false, false, false, true, true, true];
        let view = View {
            hole: [hole[0], hole[1]],
            board: &[],
            street: Street::Preflop,
            position: crate::table::Position::Button,
            seat: 0,
            players: 6,
            active: 4,
            pot: 400,
            to_call: 250,
            stack: 10_000,
            stacks: &stacks,
            committed: &committed,
            live: &live,
            street_committed: &committed,
            acted: &acted,
            raises: 1,
            settled: None,
            big_blind: 100,
            legal: &legal,
        };
        assert_eq!(
            spot_of(&view).map(|spot| spot.name()),
            Some("btn-vs-co".to_string())
        );
    }

    /// Counts preflop spots by how often they happen *and* by how much money
    /// goes through them.
    ///
    /// # Why both numbers
    ///
    /// Counting decisions alone is misleading, and this test originally did
    /// only that: it reported three-bet pots as 7% of preflop decisions, which
    /// invited the conclusion that charting them was not worth the work. But a
    /// three-bet pot is not one seventh as important as a raised pot — it is
    /// several times larger. A fold from under the gun and a five-bet call are
    /// both one decision and they are not both worth one decision's attention.
    ///
    /// So the money is counted too: for each hand, how deep the preflop betting
    /// went, and how many chips actually changed hands. That is the number that
    /// says whether a spot is worth charting.
    ///
    /// # What this cannot tell you
    ///
    /// The bots playing are the ones being measured, and they have no charts
    /// past a single raise. Their three- and four-bet frequencies come from the
    /// solve and the heuristic, so the deep buckets here reflect how *this* bot
    /// reaches them rather than how a table of solid players would. Treat the
    /// deep percentages as the right order of magnitude, not a decimal place.
    ///
    /// Run with:
    /// `cargo test --release -p poker-core -- --ignored --nocapture census`
    #[test]
    #[ignore = "a simulation for reporting, not a pass/fail check; run with --ignored --nocapture"]
    fn census_of_preflop_spots() {
        use crate::betting::Action;
        use crate::blueprint::Blueprint;
        use crate::bot::BlueprintAgent;
        use crate::preflop::Sizing;
        use crate::rng::Rng;
        use crate::table::{Agent, Deck, Table};
        use std::cell::RefCell;
        use std::collections::BTreeMap;
        use std::rc::Rc;

        // Which folder of charts to measure. Set POKER_CHARTS to compare one
        // set against another — the point of the run is usually "does adding
        // these ranges move the numbers", and that needs two folders.
        let named = std::env::var("POKER_CHARTS").unwrap_or_else(|_| "data/charts".to_string());
        let charts = Charts::load(std::path::Path::new(&format!("../../{named}")))
            .or_else(|_| Charts::load(std::path::Path::new(&named)))
            .expect("the ranges named by POKER_CHARTS");
        println!("
measuring {named}");

        #[derive(Default)]
        struct Census {
            /// Preflop decisions, by what kind of spot they were.
            decisions: BTreeMap<&'static str, u64>,
            /// How deep the preflop betting has gone in the hand being played.
            deepest: u8,
            /// Seats that put money in preflop by choice this hand, and seats
            /// that raised. These are what a poker room's statistics show, and
            /// a table full of bots is recognisable by them long before anyone
            /// looks at a decision.
            voluntary: [bool; 7],
            raised: [bool; 7],
            /// Who raised this hand, in order, by position. The first is the
            /// opener and the second is the three-bettor, which together name
            /// the spot the opener then has to answer.
            raisers: Vec<Seat>,
            /// Three-bet pots by which pair of positions made them, and how
            /// much money went through each.
            pairs: BTreeMap<String, (u64, i64)>,
            hands_dealt: u64,
            vpip: u64,
            pfr: u64,
        }
        type Shared = Rc<RefCell<Census>>;

        /// Classifies each preflop decision, then plays it as normal.
        struct Counting {
            inner: BlueprintAgent,
            census: Shared,
            held: Charts,
        }

        impl Agent for Counting {
            fn name(&self) -> &str {
                self.inner.name()
            }

            fn new_hand(&mut self) {
                self.inner.new_hand();
            }

            fn act(&mut self, view: &View, rng: &mut Rng) -> Action {
                if view.street == Street::Preflop {
                    let seat = seat_position(view.seat, view.players);
                    let kind = match (spot_of(view), view.raises) {
                        (Some(spot), _) if self.held.get(spot).is_some() => "charted",
                        (Some(_), 0) => "unopened, no chart yet",
                        (Some(_), 1) => "facing a raise, no chart yet",
                        (Some(_), _) => "other, no chart yet",
                        (None, 0) => "limped pot",
                        (None, 1) if seat == Some(Seat::Utg) => {
                            "facing a raise from the same charted position"
                        }
                        (None, 1) => "facing a raise, uncharted shape",
                        (None, 2) => "three-bet pot",
                        (None, 3) => "four-bet pot",
                        (None, _) => "five-bet pot or deeper",
                    };
                    let mut census = self.census.borrow_mut();
                    *census.decisions.entry(kind).or_default() += 1;
                    census.deepest = census.deepest.max(view.raises);
                }
                let action = self.inner.act(view, rng);
                if view.street == Street::Preflop {
                    let mut census = self.census.borrow_mut();
                    match action {
                        // Calling a blind is voluntary; posting one is not, and
                        // the seat that checks its option has not chosen to
                        // play either.
                        Action::Call => census.voluntary[view.seat] = true,
                        Action::RaiseTo(_) => {
                            census.voluntary[view.seat] = true;
                            census.raised[view.seat] = true;
                            if let Some(seat) = seat_position(view.seat, view.players) {
                                census.raisers.push(seat);
                            }
                        }
                        _ => {}
                    }
                }
                action
            }
        }

        let census: Shared = Rc::new(RefCell::new(Census::default()));
        let blueprint = Blueprint::from_profile(&Default::default(), "empty");
        let mut agents: Vec<Counting> = (0..7)
            .map(|seat| Counting {
                inner: BlueprintAgent::new(
                    format!("seat {seat}"),
                    blueprint.clone(),
                    Sizing::default(),
                )
                .with_charts(charts.clone()),
                census: Rc::clone(&census),
                held: charts.clone(),
            })
            .collect();

        let table = Table::new(100, 10_000);
        let mut rng = Rng::new(99);
        let mut deck = Deck::fresh();
        let hands = 20_000u64;

        // Chips that actually changed hands, by how deep the preflop betting
        // went in that hand. Money lost by the losing seats is the cleanest
        // measure available from a hand result, and it is the money that a
        // wrong decision in that spot would have been wrong about.
        let mut money = BTreeMap::<&'static str, i64>::new();
        let mut hands_at = BTreeMap::<&'static str, u64>::new();
        let depth_name = |raises: u8| match raises {
            0 => "limped or walked",
            1 => "single raised pot",
            2 => "three-bet pot",
            3 => "four-bet pot",
            _ => "five-bet pot or deeper",
        };

        for _ in 0..hands {
            deck.shuffle(&mut rng);
            let result = {
                let mut seats: Vec<&mut dyn Agent> = agents
                    .iter_mut()
                    .map(|agent| agent as &mut dyn Agent)
                    .collect();
                table.play_hand(&mut seats, deck.hand_cards(7), &mut rng)
            };
            let mut census = census.borrow_mut();
            let name = depth_name(census.deepest);
            census.deepest = 0;
            // One hand per seat dealt, counted the way a tracker counts it.
            for seat in 0..7 {
                census.hands_dealt += 1;
                if census.voluntary[seat] {
                    census.vpip += 1;
                }
                if census.raised[seat] {
                    census.pfr += 1;
                }
            }
            // Which pair of positions built this three-bet pot, if it is one.
            if census.raisers.len() >= 2 {
                let (opener, three_better) = (census.raisers[0], census.raisers[1]);
                let name = format!("{}-vs-{}", opener.name(), three_better.name());
                let staked: i64 = result
                    .net
                    .iter()
                    .filter(|net| **net < 0)
                    .map(|net| -net)
                    .sum();
                let row = census.pairs.entry(name).or_insert((0, 0));
                row.0 += 1;
                row.1 += staked;
            }
            census.raisers.clear();
            census.voluntary = [false; 7];
            census.raised = [false; 7];
            drop(census);
            *money.entry(name).or_default() +=
                result.net.iter().filter(|net| **net < 0).map(|net| -net).sum::<i64>();
            *hands_at.entry(name).or_default() += 1;
        }

        let census = census.borrow();
        let total: u64 = census.decisions.values().sum();
        println!("\n{total} preflop decisions over {hands} seven-handed hands\n");
        let mut rows: Vec<_> = census.decisions.iter().collect();
        rows.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
        for (kind, count) in rows {
            println!(
                "  {:>6.2}%  {:>7}  {kind}",
                *count as f64 / total as f64 * 100.0,
                count
            );
        }
        let charted = census.decisions.get("charted").copied().unwrap_or(0);
        println!(
            "\n  charts already decide {:.1}% of preflop decisions",
            charted as f64 / total as f64 * 100.0
        );

        // And the same hands weighted by what was at stake, which is the
        // number that says where charting effort pays.
        let staked: i64 = money.values().sum();
        println!("\nmoney that changed hands, by how deep the preflop betting went:\n");
        let mut rows: Vec<_> = money.iter().collect();
        rows.sort_by_key(|(_, chips)| std::cmp::Reverse(*chips));
        for (depth, chips) in rows {
            let played = hands_at.get(*depth).copied().unwrap_or(0);
            println!(
                "  {:>6.2}% of money   {:>6.2}% of hands   {:>7.0} bb average pot   {depth}",
                *chips as f64 / staked as f64 * 100.0,
                played as f64 / hands as f64 * 100.0,
                if played > 0 {
                    *chips as f64 / played as f64 / 100.0
                } else {
                    0.0
                },
            );
        }
        // Which "you opened and got three-bet" spots actually happen, and
        // how much rides on each. Charting all fifteen pairs is thirty exports;
        // this says which of them are worth the clicking.
        let mut rows: Vec<_> = census.pairs.iter().collect();
        rows.sort_by_key(|(_, (_, money))| std::cmp::Reverse(*money));
        let total_money: i64 = census.pairs.values().map(|(_, money)| money).sum();
        let total_hands: u64 = census.pairs.values().map(|(hands, _)| hands).sum();
        println!("\nthree-bet pots by who opened and who three-bet:\n");
        let mut running = 0i64;
        for (name, (hands, money)) in rows {
            running += money;
            println!(
                "  {name:<12} {:>5.1}% of 3bet-pot money  ({:>5.1}% running)  {hands} hands",
                *money as f64 / total_money as f64 * 100.0,
                running as f64 / total_money as f64 * 100.0,
            );
        }
        println!("  {total_hands} three-bet pots in all");

        // What a poker room's own statistics would show. The club removes
        // players below 20% VPIP for stalling, so this is not a curiosity — it
        // is a condition of being allowed to keep playing.
        println!(
            "\nVPIP {:.1}%   PFR {:.1}%   over {} hands dealt",
            census.vpip as f64 / census.hands_dealt as f64 * 100.0,
            census.pfr as f64 / census.hands_dealt as f64 * 100.0,
            census.hands_dealt
        );
        assert!(total > 0 && staked > 0);
    }

    /// A folder of charts may hold exports and hand-written ranges together.
    #[test]
    fn both_ways_of_writing_a_range_are_read() {
        // What a solver exports: two exact cards and a weight.
        let exported = read_range("AcKd: 1,AcKh: 1,AcKs: 1,Ac5c: 0.5").expect("an export");
        assert!(exported.weight("AKo".parse().expect("a hand")) > 0.0);

        // What a person writes.
        let written = read_range("22+, A2s+, KTo+").expect("notation");
        assert!((written.weight("22".parse().expect("a hand")) - 1.0).abs() < 1e-9);
        assert!((written.weight("KTo".parse().expect("a hand")) - 1.0).abs() < 1e-9);
        assert_eq!(written.weight("K9o".parse().expect("a hand")), 0.0);

        // And a weighted hand-written term, which has a colon but names a hand
        // class rather than two cards, is still notation.
        let mixed = read_range("AA, KQo:0.35").expect("weighted notation");
        assert!((mixed.weight("KQo".parse().expect("a hand")) - 0.35).abs() < 1e-9);
    }

    #[test]
    fn spot_names_round_trip() {
        for name in ["utg", "btn", "sb-vs-utg", "bb-vs-btn", "co-vs-hj"] {
            let spot = Spot::from_name(name).expect("a spot");
            assert_eq!(spot.name(), name);
        }
        assert!(Spot::from_name("mp").is_none(), "not a position we chart");
        assert!(Spot::from_name("utg-vs-mp").is_none());
    }

    /// Whatever a chart does not name, it folds. That is what makes a
    /// single-range opening chart a complete strategy rather than half of one.
    #[test]
    fn the_leftover_weight_is_a_fold() {
        let chart = Chart::default().with(Move::Raise, range("AA, KK, A5o:0.25"));
        let strategy = |hand: &str| chart.strategy(hand.parse().expect("a hand"));

        assert_eq!(strategy("AA"), vec![(Move::Raise, 1.0)], "no fold to report");
        assert_eq!(strategy("72o"), vec![(Move::Fold, 1.0)], "never named");

        let mixed = strategy("A5o");
        assert_eq!(mixed, vec![(Move::Fold, 0.75), (Move::Raise, 0.25)]);
    }

    /// Two actions in one spot, which is what a chart facing a raise looks
    /// like once both its ranges are present.
    #[test]
    fn actions_share_a_hand_and_the_rest_folds() {
        let chart = Chart::default()
            .with(Move::Passive, range("AJs:0.6"))
            .with(Move::Raise, range("AJs:0.3"));
        let strategy = chart.strategy("AJs".parse().expect("a hand"));
        let expected = [(Move::Fold, 0.1), (Move::Passive, 0.6), (Move::Raise, 0.3)];
        assert_eq!(strategy.len(), expected.len(), "{strategy:?}");
        for ((action, share), (want_action, want_share)) in strategy.iter().zip(expected) {
            assert_eq!(*action, want_action);
            assert!((share - want_share).abs() < 1e-9, "{share} vs {want_share}");
        }
    }

    /// Ranges are exported and rounded one action at a time, so their parts can
    /// add to a shade over one. That must not come out as a negative fold.
    #[test]
    fn overlapping_ranges_are_scaled_rather_than_left_negative() {
        let chart = Chart::default()
            .with(Move::Passive, range("KK:0.7"))
            .with(Move::Raise, range("KK:0.4"));
        let strategy = chart.strategy("KK".parse().expect("a hand"));

        assert!(
            strategy.iter().all(|(_, share)| *share >= 0.0),
            "no negative frequencies: {strategy:?}"
        );
        let total: f64 = strategy.iter().map(|(_, share)| share).sum();
        assert!((total - 1.0).abs() < 1e-9, "still a strategy: {total}");
        assert!(
            !strategy.iter().any(|(action, _)| *action == Move::Fold),
            "nothing is left over to fold"
        );
    }

    /// Which single-range charts are finished and which are not.
    ///
    /// The distinction is not obvious and this had it backwards at first:
    /// three-bet-or-fold is the equilibrium from every position but the big
    /// blind, so a lone three-betting range is a complete strategy nearly
    /// everywhere. The big blind is where a missing calling range really costs
    /// something, because that is how the big blind mostly defends.
    #[test]
    fn only_the_big_blind_needs_a_calling_range() {
        let mut charts = Charts::new();
        charts.insert(
            Spot::from_name("utg").expect("a spot"),
            Chart::default().with(Move::Raise, range("22+")),
        );
        assert!(
            charts.gaps().is_empty(),
            "raise or fold is the whole strategy in an unopened pot"
        );

        // A lone three-betting range on the button. This is what the exports
        // actually hold, and it is finished: there is no calling range in the
        // solution to be missing.
        charts.insert(
            Spot::from_name("btn-vs-co").expect("a spot"),
            Chart::default().with(Move::Raise, range("QQ+, AKs")),
        );
        assert!(
            charts.gaps().is_empty(),
            "three-bet-or-fold is the strategy outside the big blind: {:?}",
            charts.gaps()
        );

        // A lone calling range is a gap anywhere, since the three-bets vanish.
        charts.insert(
            Spot::from_name("co-vs-utg").expect("a spot"),
            Chart::default().with(Move::Passive, range("22+")),
        );
        let gaps = charts.gaps();
        assert_eq!(gaps.len(), 1, "{gaps:?}");
        assert!(gaps[0].contains("co-vs-utg"), "{}", gaps[0]);

        // The big blind is the one seat that needs both, because calling is
        // most of how it defends.
        charts.insert(
            Spot::from_name("bb-vs-btn").expect("a spot"),
            Chart::default().with(Move::Raise, range("QQ+, AKs")),
        );
        let gaps = charts.gaps();
        assert_eq!(gaps.len(), 2, "{gaps:?}");
        assert!(
            gaps.iter().any(|gap| gap.starts_with("bb-vs-btn")),
            "{gaps:?}"
        );

        // With both, it is finished.
        charts.insert(
            Spot::from_name("bb-vs-btn").expect("a spot"),
            Chart::default()
                .with(Move::Raise, range("QQ+, AKs"))
                .with(Move::Passive, range("22+, A2s+, KTo+")),
        );
        assert!(
            !charts.gaps().iter().any(|gap| gap.starts_with("bb-vs-btn")),
            "{:?}",
            charts.gaps()
        );
    }
}
