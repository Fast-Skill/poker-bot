//! Solve strategies, store them as blueprints, and query them.
//!
//! This closes the loop the live bot needs: solving is slow and happens once,
//! offline; deciding is fast and happens at the table. The bot will call the
//! same lookup `query` does.
//!
//! ```text
//! poker solve pushfold --stack 10 --out data/pushfold-10bb.bin
//! poker solve preflop  --stack 100 --out data/preflop-100bb.bin
//! poker info  data/preflop-100bb.bin
//! poker query data/preflop-100bb.bin --hand AKs --stage sb-open
//! poker chart data/pushfold-10bb.bin --player sb
//! ```

mod bridge;
mod png;
#[cfg(windows)]
mod live;

use std::collections::HashMap;
use std::process::ExitCode;

use poker_core::abstraction::HandClass;
use poker_core::bench::{duplicate_match, AlwaysCall, AlwaysFold, AlwaysJam, ChartBot};
use poker_core::blueprint::Blueprint;
use poker_core::bot::BlueprintAgent;
use poker_core::card::{Rank, NUM_RANKS};
use poker_core::cfr::{InfoKey, Solver};
use poker_core::betting::Action;
use poker_core::equity::{exact, Variant};
use poker_core::river::{Holding, River, Stage as RiverStage};
use poker_core::table::{Agent, Deck, Table};
use poker_core::telemetry::ConsoleMonitor;
use poker_core::postflop::{Postflop, Sizing as PostflopSizing};
use poker_core::preflop::{self, Preflop, Sizing};
use poker_core::pushfold::{EquityTable, PushFold};
use poker_core::ring::{Ladder, Ring};
use poker_core::threeway::ThreeWayEquity;
use poker_core::wide::{Showdown, WideEquity};
use poker_core::rng::Rng;

/// Where the shared preflop equity table is cached.
const EQUITY_CACHE: &str = "data/preflop_equity.bin";
/// Samples per pairing when the equity table has to be built.
const EQUITY_SAMPLES: u32 = 60_000;
const EQUITY_SEED: u64 = 0x9E37_79B9;
const SOLVE_SEED: u64 = 0xF01D;
/// Card templates are pixel-exact and were measured at this window size.
const TABLE_W: usize = 1430;
const TABLE_H: usize = 1040;
const TEMPLATES: &str = "data/card_templates.bin";
const GLYPHS: &str = "data/digit_templates.bin";
const HERO_CARDS: &str = "data/hero_cards.bin";
const THREE_WAY_CACHE: &str = "data/threeway.bin";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("solve") => solve(&args[1..]),
        Some("info") => info(&args[1..]),
        Some("query") => query(&args[1..]),
        Some("chart") => chart(&args[1..]),
        Some("bench") => bench(&args[1..]),
        Some("play") => play(&args[1..]),
        Some("demo") => demo(&args[1..]),
        Some("equity3") => equity3(&args[1..]),
        Some("equity") => equity_wide(&args[1..]),
        Some("textures") => textures(&args[1..]),
        Some("compare") => compare(&args[1..]),
        Some("postflop") => postflop_chart(&args[1..]),
        Some("typetest") => typetest(&args[1..]),
        Some("see") => see(&args[1..]),
        Some("grab") => grab(&args[1..]),
        Some("live") => live_cmd(&args[1..]),
        Some("help") | Some("--help") | Some("-h") | None => {
            usage();
            Ok(())
        }
        Some(other) => Err(format!("unknown command {other:?}; try `poker help`")),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

fn usage() {
    println!(
        "\
poker - solve, store, and query poker strategies

USAGE
  poker solve <game> [options]     solve a game and write a blueprint
  poker info  <blueprint>          show what a blueprint contains
  poker query <blueprint> [options] look up one decision
  poker chart <blueprint> [options] print a 13x13 range grid
  poker bench <blueprint> [options] play it against baseline opponents
  poker play  <blueprint> [options] watch it play, hand by hand
  poker demo  [blueprint]          show the whole thing works, start to finish
  poker equity3 [options]          build the three-way equity table a 3-handed solve needs
  poker equity  [options]          build a wide equity table (--players 4..7)
  poker textures [options]         build the board sample a postflop solve needs
  poker compare <a> <b>            how far apart two blueprints play
  poker postflop <file>            what a postflop solve does, by hand strength
  poker see   [options]            look at the live table and report what it reads
  poker grab  [options]            save every window of the client as a PNG
  poker live  [options]            watch a live table and decide
                                   --act off|fold|play  (play risks real chips)
                                   --explain on  shows the reasoning behind each decision
  poker typetest [options]         check the bot can set a raise amount (commits nothing)
                                   --keep-unread <dir> saves hands it cannot read

GAMES
  pushfold    heads-up jam-or-fold
  ring3..7    multiway preflop (needs `poker equity3` and `poker equity` first)
  preflop     heads-up open / 3-bet / 4-bet / jam

SOLVE OPTIONS
  --stack <bb>        effective stack, in big blinds   [default 10 / 100]
  --iterations <n>    training iterations              [default 2000000]
  --out <path>        where to write the blueprint     [required]

QUERY OPTIONS
  --hand <class>      a starting hand, e.g. AKs, TT, 72o   [required]
  --stage <name>      which decision                        [default first]

CHART OPTIONS
  --player <sb|bb>    whose range to print              [default sb]
  --stage <name>      which decision                     [default first]

BENCH OPTIONS
  --vs <opponent>     fold | call | jam | chart | all   [default all]
  --hands <n>         hands per match                   [default 20000]
  --stack <bb>        stack depth for the table         [default 100]

PLAY OPTIONS
  --vs <opponent>     fold | call | jam | chart         [default chart]
  --hands <n>         hands to show                     [default 10]
  --stack <bb>        stack depth for the table         [default 100]
  --seed <n>          fix the shuffle, to replay a run  [default 1]
  --monitor <on|off>  show what the bot sees and why    [default off]

SEE OPTIONS
  --process <name>    which app to look at                [default ClubGG]
  --resize <on|off>   force the window to 1430x1040       [default off]

STAGES
  pushfold   sb, bb
  preflop    sb-open, bb-vs-open, sb-vs-3bet, bb-vs-4bet, sb-vs-jam, bb-vs-jam"
    );
}

/// A game a blueprint can describe, recorded in its label so `query` knows how
/// to build keys for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    PushFold,
    Preflop,
    /// Multiway preflop for a table of this many seats, solved from a state
    /// machine rather than a named ladder of stages. Its information sets are
    /// packed bit fields, so there are no stage names for `query` and `chart`
    /// to work from.
    Ring(usize),
}

impl Kind {
    fn parse(name: &str) -> Result<Kind, String> {
        match name {
            "pushfold" => Ok(Kind::PushFold),
            "preflop" => Ok(Kind::Preflop),
            other if other.starts_with("ring") => {
                let seats: usize = other[4..]
                    .parse()
                    .map_err(|_| format!("unknown game {other:?}; try ring3 to ring7"))?;
                if !(3..=poker_core::wide::MAX_PLAYERS).contains(&seats) {
                    return Err(format!(
                        "ring games run from three to {} seats; heads-up is `preflop`",
                        poker_core::wide::MAX_PLAYERS
                    ));
                }
                Ok(Kind::Ring(seats))
            }
            other => Err(format!(
                "unknown game {other:?}; try pushfold, preflop, or ring3 to ring7"
            )),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Kind::PushFold => "pushfold",
            Kind::Preflop => "preflop",
            Kind::Ring(3) => "ring3",
            Kind::Ring(4) => "ring4",
            Kind::Ring(5) => "ring5",
            Kind::Ring(6) => "ring6",
            Kind::Ring(_) => "ring7",
        }
    }

    /// Recovers the game from a blueprint label written by `solve`.
    fn from_label(label: &str) -> Result<Kind, String> {
        let name = label.split('/').next().unwrap_or_default();
        Kind::parse(name).map_err(|_| {
            format!("blueprint label {label:?} does not name a known game")
        })
    }

    /// The decision stages this game exposes, in tree order.
    fn stages(self) -> &'static [(&'static str, &'static [&'static str])] {
        match self {
            Kind::PushFold => &[("sb", &["fold", "jam"]), ("bb", &["fold", "call"])],
            // Three-handed has no fixed ladder of stages to name: whether the
            // big blind is facing a raise from the button or from the small
            // blind is part of the situation, and there are far too many of
            // those to give each a name.
            Kind::Ring(_) => &[],
            Kind::Preflop => &[
                ("sb-open", &["fold", "raise", "jam"]),
                ("bb-vs-open", &["fold", "call", "raise", "jam"]),
                ("sb-vs-3bet", &["fold", "call", "raise", "jam"]),
                ("bb-vs-4bet", &["fold", "call", "jam"]),
                ("sb-vs-jam", &["fold", "call"]),
                ("bb-vs-jam", &["fold", "call"]),
            ],
        }
    }

    /// The information-set key for a hand at a named stage.
    fn key(self, stage: &str, class: HandClass) -> Result<u64, String> {
        match self {
            // Its information sets are packed bit fields describing who is
            // live, who has acted and who raised last. There is no short name
            // for a situation like that, so there is nothing to look up by.
            Kind::Ring(_) => Err(
                "ring3 blueprints have no named stages to query; the bot reads them directly. Use `poker info` to see what one contains."
                    .to_string(),
            ),
            Kind::PushFold => match stage {
                "sb" => Ok(PushFold::info_key(0, class.index())),
                "bb" => Ok(PushFold::info_key(1, class.index())),
                other => Err(unknown_stage(self, other)),
            },
            Kind::Preflop => {
                let stage = match stage {
                    "sb-open" => preflop::Stage::SbOpen,
                    "bb-vs-open" => preflop::Stage::BbVsOpen,
                    "sb-vs-3bet" => preflop::Stage::SbVs3Bet,
                    "bb-vs-4bet" => preflop::Stage::BbVs4Bet,
                    "sb-vs-jam" => preflop::Stage::SbVsJam,
                    "bb-vs-jam" => preflop::Stage::BbVsJam,
                    other => return Err(unknown_stage(self, other)),
                };
                Ok(Preflop::info_key(stage, class.index()))
            }
        }
    }

    /// Action names at a stage, for readable output.
    fn actions(self, stage: &str) -> Result<&'static [&'static str], String> {
        self.stages()
            .iter()
            .find(|(name, _)| *name == stage)
            .map(|(_, actions)| *actions)
            .ok_or_else(|| unknown_stage(self, stage))
    }

    fn default_stage(self) -> &'static str {
        self.stages()[0].0
    }
}

fn unknown_stage(kind: Kind, stage: &str) -> String {
    let names: Vec<&str> = kind.stages().iter().map(|(name, _)| *name).collect();
    format!(
        "unknown stage {stage:?} for {}; try one of: {}",
        kind.name(),
        names.join(", ")
    )
}

// --- commands ---------------------------------------------------------------

/// Where solved postflop rungs are kept.
const POSTFLOP_DIR: &str = "data/postflop";

fn solve(args: &[String]) -> Result<(), String> {
    let game = args
        .first()
        .ok_or("solve needs a game; try `poker solve pushfold --out ...`")?;
    // Intercepted before the preflop games, which it shares no parameters
    // with: this one is sized by stack-to-pot ratio and reads board textures
    // where they read equity tables.
    if game == "postflop" {
        return solve_postflop(&args[1..]);
    }
    let kind = Kind::parse(game)?;
    let flags = Flags::parse(&args[1..])?;

    let stack: f64 = flags.number("stack", match kind {
        Kind::PushFold => 10.0,
        Kind::Preflop | Kind::Ring(_) => 100.0,
    })?;
    let iterations = flags.number("iterations", 2_000_000.0)? as usize;
    let out = flags.required("out")?;
    flags.reject_unknown(&["stack", "iterations", "out"])?;

    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    eprint!("equity table ({threads} threads)... ");
    let equity = EquityTable::load_or_build(EQUITY_CACHE, EQUITY_SAMPLES, EQUITY_SEED, threads)
        .map_err(|e| format!("could not prepare {EQUITY_CACHE}: {e}"))?;
    eprintln!("ready");

    // Display for f64 already prints 10.0 as "10", so the label stays tidy.
    let label = format!("{}/{stack}bb", kind.name());
    eprint!("solving {label} for {iterations} iterations... ");

    let mut rng = Rng::new(SOLVE_SEED);
    let blueprint = match kind {
        Kind::PushFold => {
            let mut solver = Solver::new(PushFold::new(stack, equity));
            solver.train_sampled(iterations, &mut rng);
            Blueprint::from_solver(&solver, label)
        }
        Kind::Preflop => {
            let mut solver = Solver::new(Preflop::new(stack, Sizing::default(), equity));
            solver.train_sampled(iterations, &mut rng);
            Blueprint::from_solver(&solver, label)
        }
        Kind::Ring(seats) => {
            // A pot of n players needs n-way equity, and none of it can be
            // assembled from the pairwise numbers. Each table costs minutes to
            // measure and is cached, so they are read here rather than rebuilt.
            eprint!("
  equity tables... ");
            let three_way = ThreeWayEquity::load(THREE_WAY_CACHE).map_err(|e| {
                format!("could not read {THREE_WAY_CACHE}: {e}
  Build it with `poker equity3`.")
            })?;
            let mut showdown = Showdown::new(equity, three_way);
            for players in 4..=seats {
                let path = format!("data/equity{players}.bin");
                let table = WideEquity::load(&path).map_err(|e| {
                    format!(
                        "could not read {path}: {e}
                           Build it with `poker equity --players {players}`."
                    )
                })?;
                showdown = showdown.with(table);
            }
            eprint!("reaching {}-way, solving... ", showdown.reach());
            let ring = Ring::new(seats, stack, Ladder::default(), showdown);
            let mut solver = Solver::new(ring);
            solver.train_sampled(iterations, &mut rng);
            Blueprint::from_solver(&solver, label)
        }
    };
    eprintln!("done");

    blueprint
        .save(&out)
        .map_err(|e| format!("could not write {out}: {e}"))?;

    println!("wrote {out}");
    println!("  information sets  {}", blueprint.len());
    println!("  size {} bytes", blueprint.size_on_disk());
    if let Some(exploitability) = blueprint.exploitability() {
        println!("  exploitability    {exploitability:.4} bb/hand");
    }
    Ok(())
}

/// Solves the heads-up postflop tree at one stack-to-pot ratio.
///
/// # Why the ratio and not the stakes
///
/// What a player can do after the flop is set by how much is behind relative
/// to what is already out there. Ten behind into a pot of ten offers exactly
/// the decisions that a thousand behind into a pot of a thousand does. So the
/// solve is indexed by that ratio, the pot is fixed at a round number, and one
/// solve serves every table where the ratio matches.
///
/// The ratios worth having are the ones preflop actually produces: a four-bet
/// pot leaves about 1.7, a three-bet pot about 5, a single raise about 18, and
/// a limped pot more still.
fn solve_postflop(args: &[String]) -> Result<(), String> {
    use poker_core::texture::Textures;

    let flags = Flags::parse(args)?;
    flags.reject_unknown(&["spr", "iterations", "textures", "out"])?;
    let spr = flags.number("spr", 4.0)?;
    if !(0.25..=64.0).contains(&spr) {
        return Err(format!(
            "--spr {spr} is outside 0.25..64; below that there is nothing to decide and above it the stack never comes into play"
        ));
    }
    let iterations = flags.number("iterations", 20_000_000.0)? as usize;
    let textures = flags.text("textures", "data/textures.bin");
    let out = flags.required("out")?;

    eprint!("board textures... ");
    let sample = Textures::load(&textures).map_err(|e| {
        format!(
            "could not read {textures}: {e}
  Build it with `poker textures --out {textures}`."
        )
    })?;
    eprintln!(
        "{} boards, {} strength groups",
        sample.len(),
        sample.buckets()
    );

    // A round pot keeps the printed sizes readable. Nothing depends on it: the
    // solve sees prices and depths as fractions of the pot either way.
    const POT: u32 = 100;
    let stack = (POT as f64 * spr).round() as u32;
    // The label carries everything a bot has to match, because a bot reads
    // hand strength for itself rather than taking it from the solve. A
    // blueprint cut into forty-eight strength groups means something different
    // by "group 30" than one cut into twenty, and nothing else on disk says
    // which this is.
    let label = format!("postflop/spr{spr}/b{}", sample.buckets());

    eprint!("solving {label} for {iterations} iterations... ");
    let began = std::time::Instant::now();
    let mut rng = Rng::new(SOLVE_SEED);
    let game = Postflop::new(sample, POT, stack, PostflopSizing::default());
    let mut solver = Solver::new(game);
    solver.train_sampled(iterations, &mut rng);
    let blueprint = Blueprint::from_solver(&solver, label);
    eprintln!("done in {:.0}s", began.elapsed().as_secs_f64());

    blueprint
        .save(&out)
        .map_err(|e| format!("could not write {out}: {e}"))?;

    println!("wrote {out}");
    println!("  pot / stack       {POT} / {stack}");
    println!("  information sets  {}", blueprint.len());
    println!("  size {} bytes", blueprint.size_on_disk());
    Ok(())
}

fn info(args: &[String]) -> Result<(), String> {
    let path = args.first().ok_or("info needs a blueprint path")?;
    let blueprint = open(path)?;
    let kind = Kind::from_label(blueprint.label());

    println!("{path}");
    println!("  label {}", blueprint.label());
    println!("  information sets  {}", blueprint.len());
    println!("  iterations {}", blueprint.iterations());
    match blueprint.exploitability() {
        Some(value) => println!("  exploitability    {value:.4} bb/hand"),
        None => println!("  exploitability    not measured"),
    }
    match kind {
        Ok(kind) => {
            let names: Vec<&str> = kind.stages().iter().map(|(name, _)| *name).collect();
            println!("  game {}", kind.name());
            println!("  stages {}", names.join(", "));
        }
        Err(message) => println!("  game unrecognised ({message})"),
    }
    Ok(())
}

fn query(args: &[String]) -> Result<(), String> {
    let path = args.first().ok_or("query needs a blueprint path")?;
    let flags = Flags::parse(&args[1..])?;
    flags.reject_unknown(&["hand", "stage"])?;

    let blueprint = open(path)?;
    let kind = Kind::from_label(blueprint.label())?;
    let stage = flags.text("stage", kind.default_stage());
    let hand = flags.required("hand")?;
    let class: HandClass = hand
        .parse()
        .map_err(|e| format!("could not read hand {hand:?}: {e}"))?;

    let key = kind.key(&stage, class)?;
    let actions = kind.actions(&stage)?;
    let strategy = blueprint.strategy(key).ok_or_else(|| {
        format!("{class} at {stage} is not in this blueprint - it was never solved")
    })?;

    println!("{class} at {stage}\n");
    for (index, probability) in strategy.iter().enumerate() {
        let name = actions.get(index).copied().unwrap_or("?");
        let bar = "#".repeat((probability * 40.0).round() as usize);
        println!("  {name:>6}  {:>6.2}%  {bar}", probability * 100.0);
    }
    Ok(())
}

fn chart(args: &[String]) -> Result<(), String> {
    let path = args.first().ok_or("chart needs a blueprint path")?;
    let flags = Flags::parse(&args[1..])?;
    flags.reject_unknown(&["player", "stage"])?;

    let blueprint = open(path)?;
    let kind = Kind::from_label(blueprint.label())?;
    let stage = match flags.get("stage") {
        Some(stage) => stage,
        None => match (kind, flags.text("player", "sb").as_str()) {
            (Kind::PushFold, "bb") => "bb".to_string(),
            (Kind::PushFold, _) => "sb".to_string(),
            (Kind::Preflop, "bb") => "bb-vs-open".to_string(),
            (Kind::Preflop, _) => "sb-open".to_string(),
            (Kind::Ring(_), _) => {
                return Err(
                    "ring3 blueprints have no named stages to chart; try `poker info`"
                        .to_string(),
                )
            }
        },
    };
    let actions = kind.actions(&stage)?;

    println!("{} - {stage}\n", blueprint.label());
    print!("     ");
    for column in 0..NUM_RANKS {
        let rank = Rank::from_index((NUM_RANKS - 1 - column) as u8).expect("in range");
        print!(" {} ", rank.to_char());
    }
    println!();

    for row in 0..NUM_RANKS {
        let rank = Rank::from_index((NUM_RANKS - 1 - row) as u8).expect("in range");
        print!("  {}  ", rank.to_char());
        for column in 0..NUM_RANKS {
            let high = Rank::from_index((NUM_RANKS - 1 - row.min(column)) as u8).expect("in range");
            let low = Rank::from_index((NUM_RANKS - 1 - row.max(column)) as u8).expect("in range");
            let class = HandClass::new(high, low, row < column);

            // Everything except folding counts as entering the pot.
            let entered = kind
                .key(&stage, class)
                .ok()
                .and_then(|key| blueprint.strategy(key))
                .map(|strategy| 1.0 - strategy.first().copied().unwrap_or(0.0))
                .unwrap_or(0.0);

            print!(
                " {} ",
                match entered {
                    f if f > 0.95 => '#',
                    f if f > 0.50 => '+',
                    f if f > 0.05 => '.',
                    _ => ' ',
                }
            );
        }
        println!();
    }

    println!("\nlegend:  # always   + mostly   . sometimes   (blank) never");
    println!("entering means anything but {:?}", actions[0]);
    Ok(())
}

fn bench(args: &[String]) -> Result<(), String> {
    let path = args.first().ok_or("bench needs a blueprint path")?;
    let flags = Flags::parse(&args[1..])?;
    flags.reject_unknown(&["vs", "hands", "stack"])?;

    let blueprint = open(path)?;
    let stack_bb = flags.number("stack", 100.0)?;
    let hands = flags.number("hands", 20_000.0)? as u64;
    if hands < 2 {
        return Err("--hands needs at least 2".to_string());
    }
    // Every deal is played twice, once from each side.
    let pairs = hands / 2;

    let opponents: Vec<&str> = match flags.text("vs", "all").as_str() {
        "all" => vec!["fold", "call", "jam", "chart", "heuristic"],
        one => vec![Box::leak(one.to_string().into_boxed_str()) as &str],
    };

    let big_blind = 100u64;
    let table = Table::new(big_blind, (stack_bb * big_blind as f64).round() as u64);

    println!("{} vs baselines", blueprint.label());
    println!("  table    {table}");
    println!("  hands    {hands} per match (duplicate dealt)\n");

    for name in opponents {
        let (hero, _) = load_postflop(
            BlueprintAgent::new("blueprint", blueprint.clone(), Sizing::default()),
            POSTFLOP_DIR,
        );
        let mut hero = hero;
        let mut rng = Rng::new(SOLVE_SEED);

        let report = match name {
            "fold" => duplicate_match(&table, &mut hero, &mut AlwaysFold, pairs, &mut rng),
            "call" => duplicate_match(&table, &mut hero, &mut AlwaysCall, pairs, &mut rng),
            "jam" => duplicate_match(&table, &mut hero, &mut AlwaysJam, pairs, &mut rng),
            "chart" => duplicate_match(
                &table,
                &mut hero,
                &mut ChartBot::default(),
                pairs,
                &mut rng,
            ),
            // The same bot with its postflop solve taken away. This is the one
            // measurement that says whether the postflop work was worth doing:
            // both sides play the identical preflop strategy, the deals are
            // duplicated so neither gets the better cards, and the only
            // difference left is what happens after the flop.
            "heuristic" => {
                let mut stripped =
                    BlueprintAgent::new("heuristic", blueprint.clone(), Sizing::default());
                duplicate_match(&table, &mut hero, &mut stripped, pairs, &mut rng)
            }
            other => {
                return Err(format!(
                    "unknown opponent {other:?}; try fold, call, jam, chart, heuristic, or all"
                ))
            }
        };

        let verdict = if report.first_agent_wins() {
            "WIN "
        } else if report.is_significant() {
            "LOSS"
        } else {
            "----"
        };
        let (from_blueprint, total) = hero.coverage();
        println!("  {verdict}  {report}");
        println!(
            " blueprint decided {from_blueprint} of {total} spots ({:.0}%)\n",
            hero.coverage_fraction() * 100.0
        );
        // What the rest was. One coverage number cannot tell a ladder that is
        // too shallow from a tree that models the wrong game, and the two want
        // different work done.
        for (why, count) in hero.fallback_reasons() {
            println!("   {count:>6}  {why}");
        }
        println!();
    }

    println!("WIN means the lower bound of the 95% interval is above zero.");
    Ok(())
}

fn play(args: &[String]) -> Result<(), String> {
    let path = args.first().ok_or("play needs a blueprint path")?;
    let flags = Flags::parse(&args[1..])?;
    flags.reject_unknown(&["vs", "hands", "stack", "seed", "monitor"])?;
    let monitor = flags.text("monitor", "off") == "on";

    let blueprint = open(path)?;
    let stack_bb = flags.number("stack", 100.0)?;
    let hands = flags.number("hands", 10.0)? as u64;
    let seed = flags.number("seed", 1.0)? as u64;
    let opponent_name = flags.text("vs", "chart");

    let big_blind = 100u64;
    let table = Table::new(big_blind, (stack_bb * big_blind as f64).round() as u64);
    let (hero, depths) = load_postflop(
        BlueprintAgent::new("bot", blueprint.clone(), Sizing::default()),
        POSTFLOP_DIR,
    );
    let mut hero = hero;
    println!("{}", postflop_summary(&depths));
    if monitor {
        // Shows what the bot believes it sees at every decision - the view an
        // overlay would render once a vision layer exists.
        hero = hero.watch(Box::new(ConsoleMonitor::new(big_blind)));
    }
    let mut opponent: Box<dyn Agent> = match opponent_name.as_str() {
        "fold" => Box::new(AlwaysFold),
        "call" => Box::new(AlwaysCall),
        "jam" => Box::new(AlwaysJam),
        "chart" => Box::new(ChartBot::default()),
        other => {
            return Err(format!(
                "unknown opponent {other:?}; try fold, call, jam, or chart"
            ))
        }
    };

    println!("{} vs {opponent_name}", blueprint.label());
    println!("{table}, seed {seed}\n");

    let mut rng = Rng::new(seed);
    let mut deck = Deck::fresh();
    let mut running = 0i64;

    for hand in 1..=hands {
        deck.shuffle(&mut rng);
        let result = {
            let mut seats: Vec<&mut dyn Agent> = vec![&mut hero, opponent.as_mut()];
            table.play_hand(&mut seats, deck.hand_cards(2), &mut rng)
        };

        println!("Hand {hand}  -  bot on the button");
        println!(
            "  bot {}   {} {}",
            show(&result.hole[0]),
            opponent_name,
            show(&result.hole[1])
        );

        let mut current = None;
        for record in &result.actions {
            if current != Some(record.street) {
                current = Some(record.street);
                let shown = &result.board[..record.street.board_cards().min(result.board.len())];
                if shown.is_empty() {
                    println!("  preflop");
                } else {
                    println!("  {:<8} {}", record.street.to_string(), show(shown));
                }
            }
            let who = if record.seat == 0 { "bot" } else { &opponent_name };
            println!(
                "    {:<10} {}",
                who,
                describe(&record.action, record.to_call, big_blind)
            );
        }

        if result.showdown && result.board.len() == 5 {
            println!("  showdown   {}", show(&result.board));
        }

        running += result.net[0];
        let net_bb = result.net[0] as f64 / big_blind as f64;
        println!(
            "  result     bot {net_bb:+.2} bb   (running {:+.2} bb)\n",
            running as f64 / big_blind as f64
        );
    }

    let (from_blueprint, total) = hero.coverage();
    println!(
        "over {hands} hands: {:+.2} bb total, blueprint decided {from_blueprint} of {total} spots ({:.0}%)",
        running as f64 / big_blind as f64,
        hero.coverage_fraction() * 100.0
    );
    Ok(())
}

/// Renders cards as "As Kd".
fn show(cards: &[poker_core::card::Card]) -> String {
    cards
        .iter()
        .map(|card| card.to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Renders an action the way a hand history would.
fn describe(action: &Action, to_call: u64, big_blind: u64) -> String {
    let blinds = |chips: u64| chips as f64 / big_blind as f64;
    match action {
        Action::Fold => "folds".to_string(),
        Action::Check => "checks".to_string(),
        Action::Call => format!("calls {:.2} bb", blinds(to_call)),
        Action::RaiseTo(amount) => {
            if to_call == 0 {
                format!("bets to {:.2} bb", blinds(*amount))
            } else {
                format!("raises to {:.2} bb", blinds(*amount))
            }
        }
    }
}

/// Walks a newcomer through the evidence that the prototype works.
///
/// Ordered by how convincing each step is to someone who has not been
/// following: first that the maths is right, then that it derived poker theory
/// nobody supplied, then that it plays, then that it wins.
fn demo(args: &[String]) -> Result<(), String> {
    let path = args
        .first()
        .cloned()
        .unwrap_or_else(|| "data/preflop-100bb.bin".to_string());

    rule("WHAT THIS IS");
    println!(
        "A poker bot built from scratch. Nobody gave it strategy, charts, or\n\
         rules of thumb — it works out how to play by playing itself millions\n\
         of times. What follows is the evidence that it works.\n"
    );

    // --- 1. the arithmetic ---------------------------------------------------
    rule("1. IT COUNTS CARDS CORRECTLY");
    println!("Equities anyone can check against a published poker table.\n");
    println!("{:<26} {:>10} {:>12}", "matchup", "computed", "known value");
    println!("{}", "-".repeat(50));
    for (label, one, two, expected) in [
        ("AA vs KK", "AsAd", "KsKd", "82.6%"),
        ("AA vs 72o", "AsAd", "7c2d", "~88%"),
        ("AKs vs 22 (coin flip)", "AsKs", "2c2d", "~50%"),
    ] {
        let hands = [cards_of(one)?, cards_of(two)?];
        let refs: Vec<&[poker_core::card::Card]> =
            hands.iter().map(|hand| hand.as_slice()).collect();
        let result = exact(&refs, &[], Variant::Holdem)
            .map_err(|e| format!("equity failed: {e}"))?;
        println!(
            "{label:<26} {:>9.2}% {:>12}",
            result[0].percent(),
            expected
        );
    }
    println!("\nThose match the textbook. The card engine is not guessing.\n");

    // --- 2. the theory it was never told ------------------------------------
    rule("2. IT REDISCOVERED POKER THEORY ON ITS OWN");
    println!(
        "Poker has known mathematical answers for how often to bluff, and how\n\
         often to call. Nothing in this program was told them. It played a\n\
         simplified game against itself until it worked them out.\n"
    );
    println!(
        "{:>10} {:>16} {:>10} {:>16} {:>10}",
        "bet size", "it bluffs", "theory", "it calls", "theory"
    );
    println!("{}", "-".repeat(68));
    for fraction in [0.5, 1.0, 2.0] {
        let spot = River::without_raises(
            1.0,
            100.0,
            &[fraction],
            vec![Holding::new(1_000, 0.5), Holding::new(0, 0.5)],
            vec![Holding::new(500, 1.0)],
        );
        let mut solver = Solver::new(spot);
        solver.train(20_000);

        let bet = River::bet_action(0);
        let strategy = |stage, holding| {
            solver
                .average_strategy(River::info_key(stage, holding, 0, 0))
                .expect("visited")
        };
        let value = strategy(RiverStage::OopFirst, 0)[bet];
        let bluff = strategy(RiverStage::OopFirst, 1)[bet];
        let calls = strategy(RiverStage::IpVsBet, 0)[River::CALL];

        println!(
            "{:>9.2}x {:>15.1}% {:>9.1}% {:>15.1}% {:>9.1}%",
            fraction,
            bluff / (value + bluff) * 100.0,
            fraction / (1.0 + 2.0 * fraction) * 100.0,
            calls * 100.0,
            1.0 / (1.0 + fraction) * 100.0,
        );
    }
    println!(
        "\nIt lands on the exact published formulas. This is the part that is\n\
         hard to fake: those numbers are not stored anywhere in the program.\n"
    );

    // --- 3. watching it play -------------------------------------------------
    // Everything below needs a trained strategy; say so plainly rather than
    // failing halfway through a demonstration.
    if Blueprint::load(&path).is_err() {
        println!("(no trained strategy at {path})");
        println!("run: poker solve preflop --stack 100 --out {path}");
        return Ok(());
    }

    rule("3. IT PLAYS POKER");
    println!(
        "Three hands against an opponent that calls everything. Watch what it\n\
         does with a weak ace on a board that misses it.\n"
    );
    play(&[
        path.clone(),
        "--vs".into(),
        "call".into(),
        "--hands".into(),
        "3".into(),
        "--seed".into(),
        "42".into(),
    ])?;

    // --- 4. proof that it wins ----------------------------------------------
    rule("4. IT WINS, AND THE NUMBERS SAY SO");
    println!(
        "Ten thousand hands against four opponents. \"bb/100\" is big blinds won\n\
         per hundred hands — poker's standard measure. The range in brackets is\n\
         the 95% confidence interval: if it stays above zero, the win is real\n\
         and not luck.\n"
    );
    bench(&[path.clone(), "--hands".into(), "10000".into()])?;

    // --- 5. what it learned --------------------------------------------------
    rule("5. WHAT IT WORKED OUT");
    println!(
        "Every possible starting hand, and how often it plays each one. Strong\n\
         hands top-left, weak ones bottom-right — the shape real poker charts\n\
         have.\n"
    );
    chart(std::slice::from_ref(&path))?;

    rule("WHERE IT STANDS");
    println!(
        "Working: the strategy engine, and a bot that plays and wins on a\n\
         built-in table.\n\n\
         Not yet connected: reading a real game's screen and clicking its\n\
         buttons. That needs screenshots of the target app.\n\n\
         Known gap: the solver currently decides preflop only. Later streets\n\
         fall back to simple rules — the monitor labels every one of those, so\n\
         nothing is hidden."
    );
    Ok(())
}

fn rule(title: &str) {
    println!("\n{}", "=".repeat(70));
    println!("  {title}");
    println!("{}\n", "=".repeat(70));
}

fn cards_of(text: &str) -> Result<Vec<poker_core::card::Card>, String> {
    poker_core::card::parse_cards(text).map_err(|e| format!("bad cards {text:?}: {e}"))
}


/// Picks the poker table out of the client's windows.
///
/// The client shows a lobby and a table, and "the biggest one" is not a safe
/// way to tell them apart: the lobby is tall and narrow, so once the table is
/// opened at a modest size the lobby has the larger area and wins. It cost a
/// run to learn that, with the resize landing on a 569x1040 lobby.
///
/// Shape is the reliable test. A poker table is always wider than it is tall
/// and a lobby never is.
#[cfg(windows)]
fn pick_table(windows: &[poker_win::Window]) -> Result<poker_win::Window, String> {
    let mut landscape: Vec<&poker_win::Window> = windows
        .iter()
        .filter(|w| {
            let (width, height) = w.size();
            width > height
        })
        .collect();
    landscape.sort_by_key(|w| {
        let (width, height) = w.size();
        std::cmp::Reverse(width * height)
    });

    match landscape.first() {
        Some(table) => Ok(**table),
        None => {
            let mut message = String::new();
            message.push_str("no table window found. Every window the client has open is ");
            message.push_str("taller than it is wide,
which is the shape of the lobby rather ");
            message.push_str("than a table. Open a table first.

Windows seen:");
            for window in windows {
                let (w, h) = window.size();
                message.push_str(&format!("
  {w:5} x {h:<5}  {}", window.title()));
            }
            Err(message)
        }
    }
}


/// Checks that a raise amount can be written into the client's field.
///
/// The bot has to be able to bet a specific size — the client's preset buttons
/// offer only a handful, and a solved strategy means a particular number rather
/// than the nearest one on offer. Whether the field accepts typed input is the
/// last assumption in that path which has not been tested against the real
/// application.
///
/// **This commits nothing.** It writes into the amount box, photographs the
/// result, and puts back whatever was there before. No action button is ever
/// pressed, so no chips can move: setting a raise size is not raising.
///
/// It doubles as a collection run. The amount box is drawn in its own small
/// face and the templates for it came from amounts the client happened to
/// display, which never included a nought, a three or an eight. Since the bot
/// chooses what to type, typing those fills the gap in one pass.
#[cfg(windows)]
fn typetest(args: &[String]) -> Result<(), String> {
    use poker_vision::{Frame, GlyphTemplates};
    use poker_win::Window;

    let flags = Flags::parse(args)?;
    flags.reject_unknown(&["process", "seconds", "out"])?;
    let process = flags.text("process", "ClubGG");
    let seconds: u64 = flags
        .text("seconds", "300")
        .parse()
        .map_err(|_| "--seconds wants a number")?;
    let out = std::path::PathBuf::from(flags.text("out", "captures/typetest"));

    let windows = Window::find_by_process(&process);
    if windows.is_empty() {
        return Err(format!("no visible window from a process matching {process:?}"));
    }
    let table = pick_table(&windows)?;
    let (w, h) = table.size();
    if (w, h) != (TABLE_W, TABLE_H) {
        let (w, h) = table.resize(TABLE_W, TABLE_H);
        if (w, h) != (TABLE_W, TABLE_H) {
            return Err(format!("the table must be {TABLE_W}x{TABLE_H}; it settled at {w}x{h}"));
        }
    }
    let glyphs =
        GlyphTemplates::load(GLYPHS).map_err(|e| format!("could not load {GLYPHS}: {e}"))?;
    std::fs::create_dir_all(&out).map_err(|e| format!("could not make {}: {e}", out.display()))?;

    println!("watching {} for a turn with a raise available.", table.title());
    println!("Nothing will be clicked. Play normally; press Ctrl+C to stop.
");

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(seconds);
    while std::time::Instant::now() < deadline {
        let Some(capture) = table.capture() else { continue };
        let frame = Frame::new(capture.width, capture.height, &capture.rgb);
        let Some(panel) = poker_vision::read_action_panel(&frame) else {
            std::thread::sleep(std::time::Duration::from_millis(400));
            continue;
        };
        let (Some(box_), true) = (panel.amount_box, panel.offers_plain_fold()) else {
            std::thread::sleep(std::time::Duration::from_millis(400));
            continue;
        };

        let before = poker_vision::read_amount(&frame, &panel, &glyphs);
        println!("A raise is available. The box currently reads {before:?}.");

        // Values chosen to contain every digit between them, so one pass
        // photographs all ten.
        let (cx, cy) = box_.centre();
        for (round, wanted) in ["30.8", "16.4", "27.9", "5.3"].iter().enumerate() {
            table.focus();
            std::thread::sleep(std::time::Duration::from_millis(150));
            if !table.click_at(cx, cy) {
                println!("  could not place the cursor on the box - stopping");
                return Ok(());
            }
            std::thread::sleep(std::time::Duration::from_millis(200));
            let sent = table.type_text(wanted);
            std::thread::sleep(std::time::Duration::from_millis(450));

            let Some(after) = table.capture() else { continue };
            let shot = Frame::new(after.width, after.height, &after.rgb);
            let read = poker_vision::read_amount(&shot, &panel, &glyphs);

            // Keep the box whatever it now shows, so the missing digits can be
            // harvested from it later.
            let name = out.join(format!("box-{round}-{}.rgb", wanted.replace('.', "_")));
            let mut bytes = Vec::with_capacity(8 + after.rgb.len());
            bytes.extend_from_slice(&(after.width as u32).to_le_bytes());
            bytes.extend_from_slice(&(after.height as u32).to_le_bytes());
            bytes.extend_from_slice(&after.rgb);
            let _ = std::fs::write(&name, bytes);

            let verdict = match read {
                Some(value) if (value - wanted.parse::<f64>().unwrap_or(-1.0)).abs() < 0.001 => {
                    "the field took it"
                }
                Some(_) => "the field changed, but not to what was typed",
                None => "the field changed to something not yet readable",
            };
            println!("  typed {wanted:>6}  keystrokes accepted: {sent:<5}  read back {read:?}  - {verdict}");
        }

        // Put back what was there, so nothing is left armed.
        if let Some(original) = before {
            table.focus();
            std::thread::sleep(std::time::Duration::from_millis(150));
            table.click_at(cx, cy);
            std::thread::sleep(std::time::Duration::from_millis(200));
            table.type_text(&format!("{original}"));
            println!("
Put the box back to {original}.");
        }
        println!("Frames kept in {}.", out.display());
        return Ok(());
    }

    println!("No turn with a raise came up in {seconds}s.");
    Ok(())
}

#[cfg(not(windows))]
fn typetest(_args: &[String]) -> Result<(), String> {
    Err("typing into a live window is only implemented on Windows".to_string())
}

/// Watches a live table, reporting what it sees and what it would do.
///
/// The only action it will actually take is folding, and only when asked with
/// `--act fold`. That is deliberate for a first outing: folding is the one
/// choice that cannot lose more than what is already committed, so the loop can
/// be proven end to end — see a turn, press a button, confirm the client took
/// it — before a decision engine is allowed near the controls.
#[cfg(windows)]
fn live_cmd(args: &[String]) -> Result<(), String> {
    use live::{Choice, Safety, Session};
    use poker_win::Window;
    use std::path::PathBuf;

    let flags = Flags::parse(args)?;
    flags.reject_unknown(&["process", "act", "seconds", "stop-loss", "kill-switch", "blueprint", "keep-unread", "ring", "explain", "keep-turns"])?;
    let process = flags.text("process", "ClubGG");
    let act = flags.text("act", "off");
    let seconds: u64 = flags.text("seconds", "60").parse().map_err(|_| "--seconds wants a number")?;
    let stop_loss: f64 = flags
        .text("stop-loss", "200")
        .parse()
        .map_err(|_| "--stop-loss wants a number of big blinds")?;
    let kill_switch = PathBuf::from(flags.text("kill-switch", "STOP"));
    let blueprint_path = flags.text("blueprint", "data/preflop-100bb.bin");
    let ring_dir = flags.text("ring", "data");
    let keep_unread = flags.text("keep-unread", "");
    let keep_turns = flags.text("keep-turns", "");
    let explain = flags.text("explain", "off") == "on";

    #[derive(PartialEq)]
    enum Acting {
        /// Watch only.
        Never,
        /// Fold when it is our turn, whatever the engine decided. The safest
        /// thing that still proves the loop end to end.
        FoldOnly,
        /// Carry out what the engine decided.
        Play,
    }
    let acting = match act.as_str() {
        "off" => Acting::Never,
        "fold" => Acting::FoldOnly,
        "play" => Acting::Play,
        other => {
            return Err(format!(
                "--act takes `off` to watch, `fold` to always fold, or `play` to act on the strategy; got {other:?}"
            ))
        }
    };

    let windows = Window::find_by_process(&process);
    if windows.is_empty() {
        return Err(format!("no visible window from a process matching {process:?}"));
    }
    let table = pick_table(&windows)?;
    let (w, h) = table.size();
    if (w, h) != (TABLE_W, TABLE_H) {
        let (w, h) = table.resize(TABLE_W, TABLE_H);
        if (w, h) != (TABLE_W, TABLE_H) {
            return Err(format!(
                "the table must be {TABLE_W}x{TABLE_H} for the templates to fit,                  but the client settled at {w}x{h}.
                 If that is the whole screen, the display is too small for this                  layout; if it is a lobby, open a table first."
            ));
        }
    }

    let (cards, glyphs, hero) = live::templates(
        std::path::Path::new(TEMPLATES),
        std::path::Path::new(GLYPHS),
        std::path::Path::new(HERO_CARDS),
    )?;
    let safety = Safety {
        kill_switch: kill_switch.clone(),
        stop_loss_bb: stop_loss,
        max_actions: 500,
    };
    let mut session = Session::new(table, cards, glyphs, hero, safety);
    if !keep_unread.is_empty() {
        session.keep_unread = Some(PathBuf::from(&keep_unread));
    }
    if !keep_turns.is_empty() {
        session.keep_turns = Some(PathBuf::from(&keep_turns));
    }
    let blueprint = open(&blueprint_path)?;
    let mut agent = BlueprintAgent::new("bot", blueprint, Sizing::default());
    if explain {
        // The full reasoning behind each decision: what was seen, how confident
        // the reading was, which spot was consulted and how often it plays each
        // action. Worth reading before letting it play, since a decision that
        // looks wrong is far easier to diagnose alongside what produced it.
        agent = agent.watch(Box::new(ConsoleMonitor::new(bridge::CHIPS_PER_BB as u64)));
    }

    // Every multiway solve that has been built gets loaded, and pots of a size
    // with no solve fall through to the heuristic. That is a real loss of
    // strength at that size, but not a reason to refuse to play the rest.
    let mut solved_sizes = Vec::new();
    if let Ok(three_way) = ThreeWayEquity::load(THREE_WAY_CACHE) {
        let pairwise = EquityTable::load_or_build(
            EQUITY_CACHE,
            EQUITY_SAMPLES,
            EQUITY_SEED,
            std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4),
        )
        .map_err(|e| format!("could not prepare {EQUITY_CACHE}: {e}"))?;

        // Built up one size at a time, because a game of n seats needs every
        // table from two-way up to n-way, not just the widest.
        let mut showdown = Showdown::new(pairwise, three_way);
        for seats in 3..=poker_core::wide::MAX_PLAYERS {
            if seats > 3 {
                match WideEquity::load(format!("data/equity{seats}.bin")) {
                    Ok(table) => showdown = showdown.with(table),
                    Err(_) => break,
                }
            }
            let path = format!("{ring_dir}/ring{seats}-100bb.bin");
            let Ok(solved) = open(&path) else { continue };
            let ring = Ring::new(seats, 100.0, Ladder::default(), showdown.clone());
            agent = agent.with_ring(solved, ring);
            solved_sizes.push(seats);
        }
    }
    let (with_postflop, depths) = load_postflop(agent, POSTFLOP_DIR);
    agent = with_postflop;
    println!("{}", postflop_summary(&depths));
    if solved_sizes.is_empty() {
        println!("solved   : heads-up only - multiway pots use the heuristic");
    } else {
        println!(
            "solved   : heads-up, and {} players",
            solved_sizes
                .iter()
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join("/")
        );
    }
    let mut rng = Rng::new(0x5EED_0BEE);

    println!("watching : {}", session.window_title());
    println!(
        "acting   : {}",
        match acting {
            Acting::Never => "no - watching only",
            Acting::FoldOnly => "folds every hand - checks when it is free - whatever the strategy says",
            Acting::Play => "PLAYS - folds, calls and raises for real money",
        }
    );
    // Watching a table is not a free thing to do while sitting at it. The bot
    // will see its turn, print what it would do, and touch nothing — and the
    // client reads that as a player who has stopped responding, times the seat
    // out, and eventually takes it away. Sitting back in is itself a click, so
    // watch mode cannot undo it either. Worth saying here rather than leaving
    // it to be discovered from the far side of a Timeout Sit Out dialog.
    if acting == Acting::Never {
        println!(
            "  note   : the client will time this seat out, since watching never presses anything."
        );
        println!(
            "           Use --act fold to keep the seat while still risking nothing but blinds."
        );
    }
    println!("stop     : after {seconds}s, on {stop_loss} BB lost, or when {} appears", kill_switch.display());
    if !keep_unread.is_empty() {
        println!("keeping  : frames whose hole cards would not read, into {keep_unread}");
    }
    println!();

    let until = std::time::Instant::now() + std::time::Duration::from_secs(seconds);
    let mut last = String::new();
    // What the engine decided on this frame, carried from the report to the act.
    let mut pending: Option<Choice> = None;
    let mut history = bridge::History::new();
    // When the client first asked the hero to act on the turn now in progress.
    //
    // Declining a frame is free right up until this is set. After that the
    // clock is running, and the cost of holding out for a reading that never
    // comes is the seat itself.
    let mut asking_since: Option<std::time::Instant> = None;
    let mut forced = 0usize;
    while std::time::Instant::now() < until {
        let (view, held) = session.assess();
        let line = match (&view, &held) {
            (Some(v), None) => {
                // The whole chain, on one line: what the screen says, what it
                // means to the engine, and what the engine would do about it.
                let (decided, choice) = match bridge::translate(v) {
                    Ok(mut decision) => {
                        // Fills in what a single frame cannot show: how many
                        // raises this street has seen and who has acted. A
                        // postflop solve keys on both, and without them it
                        // would be answering about a different spot.
                        history.observe(&mut decision);
                        let chosen = agent.act(&decision.view(), &mut rng);
                        // The bot knows its own raises without looking, which
                        // is what keeps the count exact through a raising war.
                        if let poker_core::betting::Action::RaiseTo(to) = chosen {
                            history.hero_raised_to(to);
                        }
                        // The engine speaks in chips; the client in big blinds.
                        let choice = match chosen {
                            poker_core::betting::Action::Fold => Some(Choice::Fold),
                            poker_core::betting::Action::Check
                            | poker_core::betting::Action::Call => Some(Choice::Passive),
                            poker_core::betting::Action::RaiseTo(to) => {
                                Some(Choice::Aggressive {
                                    to_blinds: to as f64 / bridge::CHIPS_PER_BB,
                                })
                            }
                        };
                        (format!("{chosen:?}"), choice)
                    }
                    Err(why) => (format!("no decision - {}", why.explain()), None),
                };
                pending = choice;
                format!(
                    "OUR TURN  {} {}  {} of {} live  to call {}  ->  {decided}",
                    v.hole.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(""),
                    if v.board.is_empty() {
                        "preflop".to_string()
                    } else {
                        format!(
                            "on {}",
                            v.board.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(" ")
                        )
                    },
                    v.active(),
                    v.occupied(),
                    v.to_call().map(|v| format!("{v}")).unwrap_or_else(|| "?".into())
                )
            }
            (Some(v), Some(reason)) => format!(
                "waiting   {} seats, pot {} - {}",
                v.occupied(),
                v.pot.map(|p| format!("{p}")).unwrap_or_else(|| "?".into()),
                reason.explain()
            ),
            (None, Some(reason)) => format!("waiting   {}", reason.explain()),
            (None, None) => "waiting".to_string(),
        };
        if line != last {
            println!("{line}");
            last = line;
        }

        // How long the client has been waiting on us, which is the only clock
        // that matters once it starts.
        match view.as_ref() {
            Some(v) if v.hero_to_act() => {
                asking_since.get_or_insert_with(std::time::Instant::now);
            }
            _ => asking_since = None,
        }

        match (&view, &held) {
            (Some(v), None) => {
                asking_since = None;
                let choice = match acting {
                    Acting::Never => None,
                    // Checking when nothing is owed, rather than folding.
                    // Folding a hand that could be seen for free is never right
                    // and is not always even offered — the client shows Check
                    // where the fold button would be — so the click went
                    // nowhere and the turn was lost. The point of this mode is
                    // to prove the click path while risking only blinds, and
                    // checking risks less than that.
                    Acting::FoldOnly => Some(live::last_resort(v)),
                    Acting::Play => pending,
                };
                if let Some(choice) = choice {
                    match session.act(v, choice) {
                        Ok(took) => println!(
                            "  {} - and the client took it ({} ms)",
                            choice.name(),
                            took.as_millis()
                        ),
                        Err(why) => println!("  {} did not take: {}", choice.name(), why.explain()),
                    }
                }
            }
            // The client is asking, the reading will not come good, and the
            // clock has run down far enough that waiting is the worse risk.
            //
            // This is the one place the bot acts on a reading it does not
            // trust, and it is not a lapse in the rule but the reason for it:
            // refusing a frame is free only while nothing is at stake. Timing
            // out folds the hand *and* sits the hero out, after which every
            // reading fails correctly while the seat is gone. So it takes the
            // safest action a bad reading still supports — checking when
            // nothing is owed, folding otherwise — and says plainly that it
            // did.
            (Some(v), Some(_))
                if acting != Acting::Never
                    && v.hero_to_act()
                    && asking_since.is_some_and(|since| since.elapsed() >= live::DEADLINE) =>
            {
                let choice = live::last_resort(v);
                forced += 1;
                asking_since = None;
                match session.act(v, choice) {
                    Ok(took) => println!(
                        "  CLOCK - reading never settled, {} instead ({} ms)",
                        choice.name(),
                        took.as_millis()
                    ),
                    Err(why) => {
                        println!("  CLOCK - {} did not take either: {}", choice.name(), why.explain())
                    }
                }
            }
            // Sitting back in is a recovery, not a poker decision, so it is
            // done whenever the bot is allowed to touch the table at all.
            (_, Some(live::Held::SatOut)) if acting != Acting::Never => {
                if session.recover_from_sit_out() {
                    println!("  sat out - clicked back in");
                }
            }
            (_, Some(reason))
                if matches!(reason, live::Held::KillSwitch | live::Held::StopLoss) =>
            {
                println!("
stopping: {}", reason.explain());
                break;
            }
            _ => {}
        }
        std::thread::sleep(std::time::Duration::from_millis(400));
    }

    println!("
{} action(s) taken.", session.actions_taken());
    // Both of these are the reading letting the bot down rather than the
    // strategy, and both are invisible in a long scroll. A session with many
    // of either is not the bot that was benchmarked.
    if forced > 0 {
        println!(
            "{forced} turn(s) taken on a reading that never settled - the clock forced it."
        );
    }
    let (retreats, why) = session.retreats();
    if retreats > 0 {
        println!(
            "{retreats} raise(s) given up as calls - the size would not go into the field{}.",
            why.map(|w| format!(" ({})", w.explain())).unwrap_or_default()
        );
    }
    if !keep_unread.is_empty() {
        println!("{} unreadable frame(s) kept in {keep_unread}.", session.frames_kept());
    }
    if !keep_turns.is_empty() {
        println!(
            "{} picture(s) of our own turns kept in {keep_turns}.",
            session.turns_kept()
        );
    }
    Ok(())
}

#[cfg(not(windows))]
fn live_cmd(_args: &[String]) -> Result<(), String> {
    Err("playing a live window is only implemented on Windows".to_string())
}




/// Saves every visible window of the client, exactly as the bot sees it.
///
/// # Why not just take a screenshot
///
/// Templates are matched against exact pixels, and a screenshot that has been
/// through a capture tool may have been scaled, colour-managed or re-encoded on
/// the way. A template cut from such an image misses against the real window in
/// a way that looks like the reader being flaky rather than the image being
/// wrong. This writes the window buffer itself.
///
/// # Why all of them
///
/// Because the interesting window is often the one a reader would refuse.
/// [`pick_table`] deliberately ignores portrait windows, the lobby being tall
/// and narrow, so the command that reads a table cannot be used to look at a
/// lobby. Saving everything means one run captures whatever is on screen and
/// the choice of what to look at comes afterwards.
#[cfg(windows)]
fn grab(args: &[String]) -> Result<(), String> {
    use poker_win::Window;

    let flags = Flags::parse(args)?;
    flags.reject_unknown(&["process", "out", "label"])?;
    let process = flags.text("process", "ClubGG");
    let out = flags.text("out", "captures");
    let label = flags.text("label", "window");

    let windows = Window::find_by_process(&process);
    if windows.is_empty() {
        return Err(format!(
            "no visible window from a process matching {process:?}"
        ));
    }
    std::fs::create_dir_all(&out).map_err(|e| format!("could not make {out}: {e}"))?;

    println!("windows from {process:?}:");
    let mut written = 0;
    for (index, window) in windows.iter().enumerate() {
        let (w, h) = window.size();
        let shape = if w >= h { "landscape" } else { "portrait" };
        print!("  {w:5} x {h:<5}  {shape:<9}  {}", window.title());

        window.focus();
        std::thread::sleep(std::time::Duration::from_millis(400));
        let Some(capture) = window.capture() else {
            println!("   -- could not be captured");
            continue;
        };
        if capture.is_blank() {
            println!("   -- came back blank");
            continue;
        }
        let name = format!("{label}-{index}-{}x{}.png", capture.width, capture.height);
        let bytes = png::encode(capture.width, capture.height, &capture.rgb);
        match std::fs::write(std::path::Path::new(&out).join(&name), &bytes) {
            Ok(()) => {
                println!("   -> {name}  ({} KB)", bytes.len() / 1024);
                written += 1;
            }
            Err(e) => println!("   -- could not be written: {e}"),
        }
    }

    if written == 0 {
        return Err("nothing could be captured".into());
    }
    println!("\n{written} written to {out}");
    Ok(())
}

#[cfg(not(windows))]
fn grab(_args: &[String]) -> Result<(), String> {
    Err("capturing a live window is only implemented on Windows".to_string())
}

/// Prints what a postflop solve actually does, by hand strength.
///
/// # What this is for
///
/// A blueprint is a few hundred thousand numbers and says nothing legible about
/// itself. The one question worth asking of a postflop solve is whether it bets
/// its good hands, and a coverage percentage cannot answer it — a solve that
/// checks everything has perfect coverage. This lays the strategy out against
/// the only axis it keys on, so that "it checks ninety-nine per cent of the
/// time with the second nuts" is visible rather than inferred from watching
/// hands go by.
fn postflop_chart(args: &[String]) -> Result<(), String> {
    use poker_core::betting::Street;
    use poker_core::postflop::Spot;

    let path = args.first().ok_or("postflop needs a blueprint path")?;
    let flags = Flags::parse(&args[1..])?;
    flags.reject_unknown(&["rows"])?;
    let rows: usize = flags
        .text("rows", "8")
        .parse()
        .map_err(|_| "--rows wants a number")?;

    let blueprint = open(path)?;
    let (spr, buckets) = postflop_label(blueprint.label()).ok_or_else(|| {
        format!(
            "{path} is labelled {:?}, which is not a postflop solve",
            blueprint.label()
        )
    })?;
    let stack = (100.0 * spr).round() as u32;
    let game = Postflop::for_play(buckets, 100, stack, PostflopSizing::default());

    println!("{path}");
    println!("  {}  ({} information sets)", blueprint.label(), blueprint.len());
    println!("  pot 100, {stack} behind\n");
    println!("  Strength runs from 0, the weakest hands on the board, to {}.",
             buckets - 1);
    println!("  Each row is a band of strengths; the numbers are how often the");
    println!("  solve plays each move there.\n");

    for (title, player, acted) in [
        ("first to act, nothing bet", 1usize, 0u8),
        ("checked to, nothing bet", 0usize, 0b10u8),
    ] {
        for street in [Street::Flop, Street::Turn, Street::River] {
            let spot = |strength: u8| Spot {
                street,
                player,
                strength,
                pot: 100,
                bet: 0,
                mine: 0,
                behind: stack as u64,
                opponent_behind: stack as u64,
                raises: 0,
                acted,
            };
            let moves = game.moves_at(&spot(0));
            println!("  {street:?}, {title}");
            print!("    {:<12}", "strength");
            for mv in &moves {
                print!("{:>12}", mv.name());
            }
            println!("   found");

            let width = buckets.div_ceil(rows);
            for band in 0..rows {
                let first = band * width;
                let last = ((band + 1) * width).min(buckets);
                if first >= last {
                    continue;
                }
                let mut totals = vec![0.0f64; moves.len()];
                let mut found = 0usize;
                for strength in first..last {
                    let here = spot(strength as u8);
                    let Some(strategy) = game
                        .keys_near(&here)
                        .into_iter()
                        .find_map(|key| blueprint.strategy(key))
                    else {
                        continue;
                    };
                    if strategy.len() != moves.len() {
                        continue;
                    }
                    found += 1;
                    for (total, share) in totals.iter_mut().zip(strategy) {
                        *total += *share as f64;
                    }
                }
                print!("    {:<12}", format!("{first}-{}", last - 1));
                for total in &totals {
                    let share = if found == 0 { 0.0 } else { total / found as f64 };
                    print!("{:>11.0}%", share * 100.0);
                }
                println!("   {found:>4}");
            }
            println!();
        }
    }
    Ok(())
}

/// Every postflop rung that has been solved, attached to an agent.
///
/// Missing rungs are not an error. A ladder with gaps still plays the depths it
/// has, and pots at the others fall through to the heuristic — worse, but far
/// better than refusing to play. What is returned alongside is the list of
/// depths actually loaded, so a caller can say plainly what is covered.
///
/// The board sample is not read. A solve needs it; a bot does not, because it
/// reads strength off the board in front of it. The rungs cost kilobytes.
fn load_postflop(mut agent: BlueprintAgent, dir: &str) -> (BlueprintAgent, Vec<f64>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return (agent, Vec::new());
    };
    let mut paths: Vec<_> = entries
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "bin"))
        .collect();
    paths.sort();

    let mut depths = Vec::new();
    for path in paths {
        let Ok(blueprint) = Blueprint::load(&path) else {
            continue;
        };
        let Some((spr, buckets)) = postflop_label(blueprint.label()) else {
            continue;
        };
        let stack = (100.0 * spr).round() as u32;
        let game = Postflop::for_play(buckets, 100, stack, PostflopSizing::default());
        agent = agent.with_postflop(spr, blueprint, game);
        depths.push(spr);
    }
    depths.sort_by(f64::total_cmp);
    (agent, depths)
}

/// Reads the depth and strength-group count out of a postflop label.
///
/// The label is written by `poker solve postflop` and looks like
/// `postflop/spr4/b48`. Anything else is not a postflop blueprint, and saying
/// so by returning nothing is what stops a preflop solve dropped into the same
/// directory from being loaded as one.
fn postflop_label(label: &str) -> Option<(f64, usize)> {
    let mut parts = label.split('/');
    if parts.next()? != "postflop" {
        return None;
    }
    let spr: f64 = parts.next()?.strip_prefix("spr")?.parse().ok()?;
    let buckets: usize = parts.next()?.strip_prefix('b')?.parse().ok()?;
    (spr > 0.0 && buckets >= 2).then_some((spr, buckets))
}

/// A line describing what postflop coverage an agent has.
fn postflop_summary(depths: &[f64]) -> String {
    if depths.is_empty() {
        return "postflop: none solved - the heuristic plays every flop".into();
    }
    let listed: Vec<String> = depths.iter().map(|spr| format!("{spr}")).collect();
    format!(
        "postflop: heads-up at {} times the pot; multiway uses the heuristic",
        listed.join("/")
    )
}

/// Reports how differently two blueprints play.
///
/// # What this is for
///
/// A solve converges when running it longer stops changing the answer. There
/// is no way to read that off a single blueprint, so it is measured by solving
/// twice at different lengths and comparing: if doubling the iterations barely
/// moves the strategy, the extra iterations were not buying anything.
///
/// The distance at one information set is half the sum of the absolute
/// differences between the two action distributions — the share of the time
/// the two strategies would pick differently there. Zero is identical, one is
/// no overlap at all.
///
/// The mean below counts every information set alike, including ones reached
/// once in a thousand hands. It therefore reads slightly pessimistic: the rare
/// nodes are the last to settle and the least costly to have wrong. The worst
/// case is reported beside it because a mean can hide a single node that
/// flipped outright.
fn compare(args: &[String]) -> Result<(), String> {
    let (first, second) = match args {
        [first, second] => (first, second),
        _ => return Err("compare needs two blueprint paths".into()),
    };
    let left = open(first)?;
    let right = open(second)?;

    let mut total = 0.0;
    let mut worst: (f64, InfoKey) = (0.0, 0);
    let mut shared = 0usize;
    let mut only_left = 0usize;
    let mut settled = 0usize;
    for (key, here) in left.entries() {
        let Some(there) = right.strategy(key) else {
            only_left += 1;
            continue;
        };
        if here.len() != there.len() {
            return Err(format!(
                "key {key} offers {} actions in {first} and {} in {second}; these are not the same game",
                here.len(),
                there.len()
            ));
        }
        let distance: f64 = here
            .iter()
            .zip(there)
            .map(|(a, b)| (a - b).abs() as f64)
            .sum::<f64>()
            / 2.0;
        shared += 1;
        total += distance;
        settled += (distance < 0.01) as usize;
        if distance > worst.0 {
            worst = (distance, key);
        }
    }
    let only_right = right.len() - shared;

    println!("{first}");
    println!("  vs {second}");
    println!();
    println!(
        "  information sets  {} and {}, {shared} shared",
        left.len(),
        right.len()
    );
    if only_left + only_right > 0 {
        println!("  reached by one only  {only_left} / {only_right}");
    }
    if shared == 0 {
        println!("
  nothing in common — different games, or different abstractions");
        return Ok(());
    }
    println!();
    println!("  mean distance     {:.4}", total / shared as f64);
    println!("  worst {:.4} at key {}", worst.0, worst.1);
    println!(
        "  within 1% {settled} of {shared} ({:.0}%)",
        100.0 * settled as f64 / shared as f64
    );
    Ok(())
}

/// Builds the board sample a postflop solve reads hand strength from.
///
/// Each board carries every holding's strength at the flop, turn and river,
/// measured from the cards visible at that street rather than from the finished
/// board — so a hand that goes on to make a flush is not rated as though it
/// already had. It also carries every holding's finished hand, because
/// showdowns are settled exactly rather than by comparing strength groups.
///
/// Strength is exact: every card that could still come is dealt. That costs
/// roughly a second per board and is what lets the bot read its own hand at the
/// table the same way the solve was trained to read it.
fn textures(args: &[String]) -> Result<(), String> {
    use poker_core::texture::Textures;

    let flags = Flags::parse(args)?;
    flags.reject_unknown(&["boards", "buckets", "threads", "seed", "out"])?;
    let boards: usize = flags
        .text("boards", "10000")
        .parse()
        .map_err(|_| "--boards wants a number")?;
    let buckets: usize = flags
        .text("buckets", "24")
        .parse()
        .map_err(|_| "--buckets wants a number")?;
    let threads: usize = flags
        .text("threads", "8")
        .parse()
        .map_err(|_| "--threads wants a number")?;
    let seed: u64 = flags
        .text("seed", "31415926")
        .parse()
        .map_err(|_| "--seed wants a number")?;
    let out = flags.text("out", "data/textures.bin");

    println!("{boards} boards, {buckets} strength groups, every runout dealt");
    println!("  seed   : {seed}  (identical whatever the core count)");

    let began = std::time::Instant::now();
    let sample = Textures::sample(boards, buckets, seed, threads);
    let took = began.elapsed();
    sample
        .save(&out)
        .map_err(|e| format!("could not write {out}: {e}"))?;
    println!("
built in {:.0}s, written to {out}", took.as_secs_f64());
    Ok(())
}

/// Builds a showdown equity table for pots wider than three players.
///
/// Four players is the last size that can be tabulated hand-class by hand-class
/// — 35 million entries, about an hour. Beyond that the entries grow as a
/// rising factorial: five players would be 23 GB and two days, six would be
/// 790 GB. So five and up group the classes by measured strength first, which
/// is where `--buckets` comes in.
fn equity_wide(args: &[String]) -> Result<(), String> {
    use poker_core::wide::{entries, Buckets, WideEquity};

    let flags = Flags::parse(args)?;
    flags.reject_unknown(&["players", "buckets", "samples", "threads", "seed", "out"])?;
    let players: usize = flags
        .text("players", "4")
        .parse()
        .map_err(|_| "--players wants a number")?;
    let buckets: usize = flags
        .text("buckets", "0")
        .parse()
        .map_err(|_| "--buckets wants a number")?;
    let samples: u32 = flags
        .text("samples", "400")
        .parse()
        .map_err(|_| "--samples wants a number")?;
    let threads: usize = flags
        .text("threads", "8")
        .parse()
        .map_err(|_| "--threads wants a number")?;
    let seed: u64 = flags
        .text("seed", "24682468")
        .parse()
        .map_err(|_| "--seed wants a number")?;
    let out = flags.text("out", &format!("data/equity{players}.bin"));

    // Sensible groupings by size, from the arithmetic above: full precision
    // while it is affordable, then as fine as each size allows.
    let buckets = if buckets > 0 {
        buckets
    } else {
        match players {
            0..=4 => 169,
            5 => 40,
            6 => 30,
            _ => 25,
        }
    };

    eprint!("pairwise equity... ");
    let pairwise = EquityTable::load_or_build(EQUITY_CACHE, EQUITY_SAMPLES, EQUITY_SEED, threads)
        .map_err(|e| format!("could not prepare {EQUITY_CACHE}: {e}"))?;
    eprintln!("ready");

    let grouping = if buckets >= 169 {
        Buckets::none()
    } else {
        Buckets::by_strength(&pairwise, buckets)
    };
    let count = entries(players, grouping.count());
    println!("{players}-way pots over {} groups", grouping.count());
    println!("  entries : {count}");
    println!("  memory  : {:.0} MB", (count * players * 4) as f64 / 1_048_576.0);
    println!("  samples : {samples} runouts each");
    println!("  seed    : {seed}  (identical whatever the core count)");

    let began = std::time::Instant::now();
    let table = WideEquity::sampled_parallel(players, grouping, samples, seed, threads);
    let took = began.elapsed();
    table
        .save(&out)
        .map_err(|e| format!("could not write {out}: {e}"))?;
    println!("
built in {:.0}s, written to {out}", took.as_secs_f64());
    Ok(())
}

/// Builds the three-way equity table.
///
/// A three-handed solve cannot reuse the pairwise table: multiplying pairwise
/// equities misorders real spots, and three-way equity is not a function of the
/// pairwise numbers. This measures all 818,805 triples of hand classes directly
/// and caches them, because building costs minutes and reading costs
/// milliseconds.
fn equity3(args: &[String]) -> Result<(), String> {
    use poker_core::threeway::{ThreeWayEquity, NUM_TRIPLES};

    let flags = Flags::parse(args)?;
    flags.reject_unknown(&["out", "samples", "threads", "seed"])?;
    let out = flags.text("out", "data/threeway.bin");
    let samples: u32 = flags
        .text("samples", "300")
        .parse()
        .map_err(|_| "--samples wants a number")?;
    let threads: usize = flags
        .text("threads", "8")
        .parse()
        .map_err(|_| "--threads wants a number")?;
    let seed: u64 = flags
        .text("seed", "13371337")
        .parse()
        .map_err(|_| "--seed wants a number")?;

    println!("measuring {NUM_TRIPLES} hand-class triples, {samples} runouts each");
    println!("threads  : {threads}");
    println!("seed     : {seed}  (the table is identical whatever the core count)");

    let began = std::time::Instant::now();
    let table = ThreeWayEquity::sampled_parallel(samples, seed, threads);
    let took = began.elapsed();

    table
        .save(&out)
        .map_err(|e| format!("could not write {out}: {e}"))?;
    println!("
built in {:.1}s, written to {out}", took.as_secs_f64());

    // A spot worth printing, because it is the one that justifies the table.
    let class = |text: &str| -> HandClass {
        let mut chars = text.chars();
        let high = Rank::from_char(chars.next().expect("rank")).expect("rank");
        let low = Rank::from_char(chars.next().expect("rank")).expect("rank");
        HandClass::new(high, low, chars.next() == Some('s'))
    };
    let shares = table.get(class("AA"), class("KK"), class("72o"));
    println!(
        "
AA vs KK vs 72o, three-handed: {:.1}% / {:.1}% / {:.1}%",
        shares[0] * 100.0,
        shares[1] * 100.0,
        shares[2] * 100.0
    );
    println!("Multiplying pairwise equities would give 72o under 2%, which is the");
    println!("mistake this table exists to avoid.");
    Ok(())
}

/// Looks at the live client and reports what the recogniser makes of it.
///
/// This is the vision layer end to end: find the window, size it, capture raw
/// pixels, read the cards. Everything it prints is what the bot itself would
/// see, so a disagreement between this and the screen is a bug worth chasing
/// before anything is allowed to act.
#[cfg(windows)]
fn see(args: &[String]) -> Result<(), String> {
    use poker_vision::{
        Frame, GlyphTemplates, HeroTemplates, Ink, TableView, Templates, TextThresholds, Thresholds,
    };
    use poker_win::Window;

    let flags = Flags::parse(args)?;
    flags.reject_unknown(&["process", "resize"])?;
    let process = flags.text("process", "ClubGG");
    let resize = flags.text("resize", "off") == "on";

    let windows = Window::find_by_process(&process);
    if windows.is_empty() {
        return Err(format!("no visible window from a process matching {process:?}"));
    }
    println!("windows from {process:?}, largest first:");
    for window in &windows {
        let (w, h) = window.size();
        println!("  {w:5} x {h:<5}  {}", window.title());
    }

    let table = pick_table(&windows)?;
    println!("
reading: {}", table.title());

    if resize {
        let (w, h) = table.resize(TABLE_W, TABLE_H);
        println!("resized to {w} x {h}");
        if (w, h) != (TABLE_W, TABLE_H) {
            println!("  note: the client did not accept that size exactly");
        }
    }

    let (w, h) = table.size();
    if (w, h) != (TABLE_W, TABLE_H) {
        println!(
            "  warning: templates were measured at {TABLE_W}x{TABLE_H}; at {w}x{h} the \
             layout reflows and reading will fail. Re-run with --resize on."
        );
    }

    table.focus();
    std::thread::sleep(std::time::Duration::from_millis(400));

    let capture = table.capture().ok_or("the window could not be captured")?;
    if capture.is_blank() {
        return Err("the capture came back a single flat colour - the window was covered, or \
             the client is blocking capture"
            .to_string());
    }
    println!("captured {} x {}", capture.width, capture.height);

    let templates = Templates::load(TEMPLATES)
        .map_err(|e| format!("could not load {TEMPLATES}: {e}"))?;
    let frame = Frame::new(capture.width, capture.height, &capture.rgb);
    let reads = poker_vision::read_cards(&frame, &templates, Thresholds::default());

    println!("
cards found: {}
", reads.len());
    println!("{:>10}  {:>8}  {:>7}  {:>7}", "position", "card", "dist", "margin");
    println!("{}", "-".repeat(38));
    for read in &reads {
        let shown = read.card.map(|c| c.to_string()).unwrap_or_else(|| "refused".into());
        println!(
            "{:>4},{:<5}  {shown:>8}  {:>7.1}  {:>7.1}",
            read.x, read.y, read.distance, read.margin
        );
    }

    let confident = reads.iter().filter(|r| r.is_confident()).count();
    println!("
{confident} of {} card(s) read confidently.", reads.len());
    if confident < reads.len() {
        println!("Refusals are cards something is drawn over - correct behaviour, not a failure.");
    }

    let glyphs =
        GlyphTemplates::load(GLYPHS).map_err(|e| format!("could not load {GLYPHS}: {e}"))?;
    let numbers = poker_vision::read_numbers(&frame, &glyphs, TextThresholds::default());

    println!("
readouts found: {}
", numbers.len());
    println!("{:>10}  {:>6}  {:>10}  {:>10}", "position", "kind", "text", "big blinds");
    println!("{}", "-".repeat(44));
    for number in &numbers {
        let kind = match number.ink {
            Ink::Gold => "pot",
            Ink::Cyan => "stack",
            Ink::White => "bet",
        };
        let shown = match number.value {
            Some(value) => format!("{value}"),
            None => "refused".to_string(),
        };
        println!(
            "{:>4},{:<5}  {kind:>6}  {:>10}  {shown:>10}",
            number.x, number.y, number.text
        );
    }

    let read = numbers.iter().filter(|n| n.is_confident()).count();
    println!("
{read} of {} readout(s) parsed.", numbers.len());
    if numbers.is_empty() {
        println!("No readouts at all usually means a dialog is covering the table.");
    }

    let hero_cards =
        HeroTemplates::load(HERO_CARDS).map_err(|e| format!("could not load {HERO_CARDS}: {e}"))?;
    let table = TableView::read(
        &frame,
        &templates,
        &glyphs,
        &hero_cards,
        Thresholds::default(),
        TextThresholds::default(),
    );

    println!("
--- the table ---
");
    if table.seats.is_empty() {
        println!("No seats could be read. A dialog covering the table does this,");
        println!("and so does a frame caught while the client is re-laying it out.");
        return Ok(());
    }

    let blinds = table.blinds();
    println!("{:>4}  {:>10}  {:>8}  {:<6}", "seat", "stack", "bet", "role");
    println!("{}", "-".repeat(48));
    for (i, seat) in table.seats.iter().enumerate() {
        let stack = seat.stack.map(|v| format!("{v}")).unwrap_or_else(|| "?".into());
        let bet = seat.bet.map(|v| format!("{v}")).unwrap_or_else(|| "-".into());
        let role = match (table.button, blinds) {
            (Some(b), _) if b == i => "button",
            (_, Some((small, _))) if small == i => "small",
            (_, Some((_, big))) if big == i => "big",
            _ => "",
        };
        let who = if seat.hero { "<- hero" } else { "" };
        println!("{i:>4}  {stack:>10}  {bet:>8}  {role:<6}  {who}");
    }

    let show = |cards: &[poker_core::card::Card]| {
        if cards.is_empty() {
            "-".to_string()
        } else {
            cards.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(" ")
        }
    };
    println!("
hole   : {}", show(&table.hole));
    println!("board  : {}", show(&table.board));
    println!("pot    : {}", table.pot.map(|v| format!("{v} BB")).unwrap_or_else(|| "?".into()));
    if let Some(collected) = table.collected {
        println!("middle : {collected} BB already gathered in");
    }
    println!("to call: {}", table.to_call().map(|v| format!("{v} BB")).unwrap_or_else(|| "?".into()));
    println!(
        "money  : {}",
        if table.is_consistent() {
            "balances"
        } else {
            "does NOT balance - do not act on this frame"
        }
    );
    if table.refusals > 0 {
        println!("{} reading(s) refused.", table.refusals);
    }

    match &table.action {
        None => println!("turn   : no buttons showing - nothing to do"),
        Some(panel) if panel.offers_plain_fold() => {
            println!("turn   : YOURS - {} button(s)", panel.buttons.len());
            for (name, button) in [
                ("fold", panel.fold()),
                ("check/call", panel.passive()),
                ("bet/raise", panel.aggressive()),
            ] {
                if let Some(b) = button {
                    let (x, y) = b.centre();
                    println!(" {name:<11} click ({x}, {y})");
                }
            }
        }
        Some(panel) => {
            println!("turn   : not yours - the buttons showing are the ones that arm");
            println!(" an action in advance ({} of them). Clicking them would", panel.buttons.len());
            println!(" decide the hand before it has been seen, so they are ignored.");
        }
    }
    Ok(())
}

#[cfg(not(windows))]
fn see(_args: &[String]) -> Result<(), String> {
    Err("looking at a live window is only implemented on Windows".to_string())
}

fn open(path: &str) -> Result<Blueprint, String> {
    Blueprint::load(path).map_err(|e| format!("could not read {path}: {e}"))
}

// --- flag parsing -----------------------------------------------------------

/// A minimal `--name value` parser.
///
/// Hand-rolled to keep the workspace dependency-free, which matters more than
/// it might seem: every dependency is something that has to be audited before
/// this ever runs unattended.
struct Flags {
    values: HashMap<String, String>,
}

/// Flags that are switches, and so mean `on` when given bare.
///
/// Listed rather than inferred, so that a flag needing a real value can never
/// silently become the string "on" because its value was left off.
const SWITCHES: &[&str] = &["explain", "monitor", "resize"];

impl Flags {
    /// Parses `--name value` pairs, and bare switches as `on`.
    ///
    /// # Why a bare flag means on
    ///
    /// The switches here read as switches — `--explain`, `--monitor`,
    /// `--resize` — and a bare one is unambiguous: nothing else could be meant
    /// by it. Demanding `--explain on` bought nothing and cost a run, twice,
    /// with the whole command rejected over a word that was never in doubt.
    ///
    /// Flags that take a real value are unaffected. A missing value is still an
    /// error for them, because guessing at a stake or a path is not the same
    /// kind of obvious.
    fn parse(args: &[String]) -> Result<Flags, String> {
        let mut values = HashMap::new();
        let mut index = 0;
        while index < args.len() {
            let arg = &args[index];
            let name = arg
                .strip_prefix("--")
                .ok_or_else(|| format!("expected a --flag, found {arg:?}"))?;
            let value = args.get(index + 1).filter(|next| !next.starts_with("--"));
            match value {
                Some(value) => {
                    values.insert(name.to_string(), value.clone());
                    index += 2;
                }
                None if SWITCHES.contains(&name) => {
                    values.insert(name.to_string(), "on".to_string());
                    index += 1;
                }
                None => return Err(format!("--{name} needs a value")),
            }
        }
        Ok(Flags { values })
    }

    fn get(&self, name: &str) -> Option<String> {
        self.values.get(name).cloned()
    }

    fn text(&self, name: &str, fallback: &str) -> String {
        self.get(name).unwrap_or_else(|| fallback.to_string())
    }

    fn required(&self, name: &str) -> Result<String, String> {
        self.get(name).ok_or_else(|| format!("--{name} is required"))
    }

    fn number(&self, name: &str, fallback: f64) -> Result<f64, String> {
        match self.get(name) {
            Some(text) => text
                .parse()
                .map_err(|_| format!("--{name} expects a number, got {text:?}")),
            None => Ok(fallback),
        }
    }

    /// Rejects flags this command does not understand, rather than ignoring a
    /// typo and silently solving something other than what was asked for.
    fn reject_unknown(&self, known: &[&str]) -> Result<(), String> {
        for name in self.values.keys() {
            if !known.contains(&name.as_str()) {
                return Err(format!(
                    "unknown flag --{name}; this command takes: {}",
                    known
                        .iter()
                        .map(|k| format!("--{k}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bare switch means on; a flag wanting a value still says so.
    #[test]
    fn a_bare_switch_is_on_and_a_bare_value_flag_is_an_error() {
        let args = |bits: &[&str]| -> Vec<String> {
            bits.iter().map(|b| b.to_string()).collect()
        };

        let flags = Flags::parse(&args(&["--explain"])).expect("a bare switch");
        assert_eq!(flags.text("explain", "off"), "on");

        let flags =
            Flags::parse(&args(&["--act", "off", "--explain"])).expect("a switch at the end");
        assert_eq!(flags.text("act", "off"), "off");
        assert_eq!(flags.text("explain", "off"), "on");

        let flags =
            Flags::parse(&args(&["--explain", "--act", "fold"])).expect("a switch before a flag");
        assert_eq!(flags.text("explain", "off"), "on");
        assert_eq!(flags.text("act", "off"), "fold");

        // Still explicit where it matters.
        let flags = Flags::parse(&args(&["--explain", "off"])).expect("explicit off");
        assert_eq!(flags.text("explain", "on"), "off");

        // A flag that wants a real value is not guessed at.
        assert!(Flags::parse(&args(&["--seconds"])).is_err());
        assert!(Flags::parse(&args(&["--out"])).is_err());
        assert!(Flags::parse(&args(&["--seconds", "--act", "off"])).is_err());
    }

    /// A postflop label says what a bot has to match, and nothing else parses.
    ///
    /// The directory is scanned rather than listed, so whatever is dropped in
    /// it gets opened. A preflop blueprint that parsed as a postflop one would
    /// be attached as a rung and consulted with keys built for another game,
    /// which is a silent misplay rather than an error.
    #[test]
    fn only_a_postflop_label_reads_as_a_postflop_rung() {
        assert_eq!(postflop_label("postflop/spr4/b48"), Some((4.0, 48)));
        assert_eq!(postflop_label("postflop/spr1.5/b20"), Some((1.5, 20)));

        for wrong in [
            "preflop/100bb",
            "ring3/100bb",
            "pushfold/10bb",
            "postflop/spr4",
            "postflop/b48/spr4",
            "postflop/spr0/b48",
            "postflop/sprx/b48",
            "postflop/spr4/b1",
            "postflop/spr4/48",
            "",
        ] {
            assert_eq!(postflop_label(wrong), None, "{wrong:?} should not parse");
        }
    }

    #[test]
    fn a_ladder_with_no_rungs_says_so() {
        assert!(postflop_summary(&[]).contains("none solved"));
        assert!(postflop_summary(&[1.5, 6.0]).contains("1.5/6"));
    }
}
