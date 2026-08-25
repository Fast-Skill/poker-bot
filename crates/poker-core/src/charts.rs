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
}

impl Spot {
    /// The file stem a spot's charts are named with: `utg`, `btn-vs-co`.
    pub fn name(self) -> String {
        match self.versus {
            Some(versus) => format!("{}-vs-{}", self.seat.name(), versus.name()),
            None => self.seat.name().to_string(),
        }
    }

    pub fn from_name(name: &str) -> Option<Spot> {
        match name.split_once("-vs-") {
            Some((seat, versus)) => Some(Spot {
                seat: Seat::from_name(seat)?,
                versus: Some(Seat::from_name(versus)?),
            }),
            None => Some(Spot {
                seat: Seat::from_name(name)?,
                versus: None,
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
            let range =
                Range::from_combos(&text).map_err(|why| format!("{}: {why}", path.display()))?;

            let held = charts.get(spot).cloned().unwrap_or_default();
            charts.insert(spot, held.with(action, range));
        }
        Ok(charts)
    }

    /// Spots that are held but almost certainly incomplete.
    ///
    /// A spot facing a raise has two ways to continue — call, or three-bet —
    /// and solvers export one range per action. A chart holding only one of
    /// them is a playable strategy, just a narrower one than the solution it
    /// came from, and it fails silently: nothing crashes, the bot simply never
    /// takes the missing line. Only the range that is there says which loss it
    /// is, so the message names it.
    ///
    /// An unopened pot is the exception. Raise-or-fold really is the whole
    /// strategy there, so a single raising range is complete.
    ///
    /// Reported rather than refused, because a partial chart still beats no
    /// chart and the caller may already know what it is missing.
    pub fn gaps(&self) -> Vec<String> {
        self.spots
            .iter()
            .filter(|(name, chart)| name.contains("-vs-") && chart.by_action.len() < 2)
            .map(|(name, chart)| {
                let missing = match chart.by_action.first().map(|(action, _)| action) {
                    Some(Move::Raise) => "no calling range, so it three-bets or folds and never flats",
                    Some(Move::Passive) => "no three-betting range, so it calls or folds and never raises",
                    _ => "only one action, where the solution has two",
                };
                format!("{name}: {missing}")
            })
            .collect()
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
            (!limped).then_some(Spot { seat, versus: None })
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
            })
        }
        _ => None,
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

    /// A chart for a spot facing a raise that holds only one range would fold
    /// every three-bet — a leak that costs money quietly rather than crashing.
    #[test]
    fn a_half_exported_chart_is_reported_as_a_gap() {
        let mut charts = Charts::new();
        charts.insert(
            Spot::from_name("utg").expect("a spot"),
            Chart::default().with(Move::Raise, range("22+")),
        );
        assert!(
            charts.gaps().is_empty(),
            "raise or fold is the whole strategy in an unopened pot"
        );

        // Only a calling range: the three-bets go missing.
        charts.insert(
            Spot::from_name("btn-vs-co").expect("a spot"),
            Chart::default().with(Move::Passive, range("22+")),
        );
        let gaps = charts.gaps();
        assert_eq!(gaps.len(), 1, "{gaps:?}");
        assert!(gaps[0].contains("btn-vs-co"), "{}", gaps[0]);
        assert!(gaps[0].contains("never raises"), "{}", gaps[0]);

        // Only a raising range, which is what the exports on hand actually
        // hold: the strategy plays, it just never flat-calls a raise.
        charts.insert(
            Spot::from_name("btn-vs-co").expect("a spot"),
            Chart::default().with(Move::Raise, range("QQ+, AKs")),
        );
        let gaps = charts.gaps();
        assert!(gaps[0].contains("never flats"), "{}", gaps[0]);
    }
}
