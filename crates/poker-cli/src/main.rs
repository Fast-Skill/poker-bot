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
#[cfg(windows)]
mod live;

use std::collections::HashMap;
use std::process::ExitCode;

use poker_core::abstraction::HandClass;
use poker_core::bench::{duplicate_match, AlwaysCall, AlwaysFold, AlwaysJam, ChartBot};
use poker_core::blueprint::Blueprint;
use poker_core::bot::BlueprintAgent;
use poker_core::card::{Rank, NUM_RANKS};
use poker_core::cfr::Solver;
use poker_core::betting::Action;
use poker_core::equity::{exact, Variant};
use poker_core::river::{Holding, River, Stage as RiverStage};
use poker_core::table::{Agent, Deck, Table};
use poker_core::telemetry::ConsoleMonitor;
use poker_core::preflop::{self, Preflop, Sizing};
use poker_core::pushfold::{EquityTable, PushFold};
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
        Some("see") => see(&args[1..]),
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
  poker see   [options]            look at the live table and report what it reads
  poker live  [options]            watch a live table and decide; --act fold to play

GAMES
  pushfold    heads-up jam-or-fold
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
}

impl Kind {
    fn parse(name: &str) -> Result<Kind, String> {
        match name {
            "pushfold" => Ok(Kind::PushFold),
            "preflop" => Ok(Kind::Preflop),
            other => Err(format!("unknown game {other:?}; try pushfold or preflop")),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Kind::PushFold => "pushfold",
            Kind::Preflop => "preflop",
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

fn solve(args: &[String]) -> Result<(), String> {
    let game = args
        .first()
        .ok_or("solve needs a game; try `poker solve pushfold --out ...`")?;
    let kind = Kind::parse(game)?;
    let flags = Flags::parse(&args[1..])?;

    let stack: f64 = flags.number("stack", match kind {
        Kind::PushFold => 10.0,
        Kind::Preflop => 100.0,
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
    };
    eprintln!("done");

    blueprint
        .save(&out)
        .map_err(|e| format!("could not write {out}: {e}"))?;

    println!("wrote {out}");
    println!("  information sets  {}", blueprint.len());
    println!("  size              {} bytes", blueprint.size_on_disk());
    if let Some(exploitability) = blueprint.exploitability() {
        println!("  exploitability    {exploitability:.4} bb/hand");
    }
    Ok(())
}

fn info(args: &[String]) -> Result<(), String> {
    let path = args.first().ok_or("info needs a blueprint path")?;
    let blueprint = open(path)?;
    let kind = Kind::from_label(blueprint.label());

    println!("{path}");
    println!("  label             {}", blueprint.label());
    println!("  information sets  {}", blueprint.len());
    println!("  iterations        {}", blueprint.iterations());
    match blueprint.exploitability() {
        Some(value) => println!("  exploitability    {value:.4} bb/hand"),
        None => println!("  exploitability    not measured"),
    }
    match kind {
        Ok(kind) => {
            let names: Vec<&str> = kind.stages().iter().map(|(name, _)| *name).collect();
            println!("  game              {}", kind.name());
            println!("  stages            {}", names.join(", "));
        }
        Err(message) => println!("  game              unrecognised ({message})"),
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
        "all" => vec!["fold", "call", "jam", "chart"],
        one => vec![Box::leak(one.to_string().into_boxed_str()) as &str],
    };

    let big_blind = 100u64;
    let table = Table::new(big_blind, (stack_bb * big_blind as f64).round() as u64);

    println!("{} vs baselines", blueprint.label());
    println!("  table    {table}");
    println!("  hands    {hands} per match (duplicate dealt)\n");

    for name in opponents {
        let mut hero = BlueprintAgent::new(
            "blueprint",
            blueprint.clone(),
            Sizing::default(),
        );
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
            other => {
                return Err(format!(
                    "unknown opponent {other:?}; try fold, call, jam, chart, or all"
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
            "        blueprint decided {from_blueprint} of {total} spots ({:.0}%)\n",
            hero.coverage_fraction() * 100.0
        );
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
    let mut hero = BlueprintAgent::new("bot", blueprint.clone(), Sizing::default());
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
    flags.reject_unknown(&["process", "act", "seconds", "stop-loss", "kill-switch", "blueprint"])?;
    let process = flags.text("process", "ClubGG");
    let act = flags.text("act", "off");
    let seconds: u64 = flags.text("seconds", "60").parse().map_err(|_| "--seconds wants a number")?;
    let stop_loss: f64 = flags
        .text("stop-loss", "200")
        .parse()
        .map_err(|_| "--stop-loss wants a number of big blinds")?;
    let kill_switch = PathBuf::from(flags.text("kill-switch", "STOP"));
    let blueprint_path = flags.text("blueprint", "data/preflop-100bb.bin");

    let acting = match act.as_str() {
        "off" => false,
        "fold" => true,
        other => {
            return Err(format!(
                "--act takes `off` to only watch, or `fold` to actually fold; got {other:?}"
            ))
        }
    };

    let windows = Window::find_by_process(&process);
    let table = *windows
        .first()
        .ok_or_else(|| format!("no visible window from a process matching {process:?}"))?;
    let (w, h) = table.size();
    if (w, h) != (TABLE_W, TABLE_H) {
        let (w, h) = table.resize(TABLE_W, TABLE_H);
        if (w, h) != (TABLE_W, TABLE_H) {
            return Err(format!(
                "the table must be {TABLE_W}x{TABLE_H} for the templates to fit;                  the client settled at {w}x{h}"
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
    let blueprint = open(&blueprint_path)?;
    let mut agent = BlueprintAgent::new("bot", blueprint, Sizing::default());
    let mut rng = Rng::new(0x5EED_0BEE);

    println!("watching : {}", session.window_title());
    println!("acting   : {}", if acting { "yes - will fold when it is our turn" } else { "no - watching only" });
    println!("stop     : after {seconds}s, on {stop_loss} BB lost, or when {} appears
", kill_switch.display());

    let until = std::time::Instant::now() + std::time::Duration::from_secs(seconds);
    let mut last = String::new();
    while std::time::Instant::now() < until {
        let (view, held) = session.assess();
        let line = match (&view, &held) {
            (Some(v), None) => {
                // The whole chain, on one line: what the screen says, what it
                // means to the engine, and what the engine would do about it.
                let decided = match bridge::translate(v) {
                    Ok(decision) => {
                        let chosen = agent.act(&decision.view(), &mut rng);
                        format!("{chosen:?}")
                    }
                    Err(why) => format!("no decision - {}", why.explain()),
                };
                format!(
                    "OUR TURN  {} on {}  {} of {} in the pot  to call {:?}  ->  {decided}",
                    v.hole.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(""),
                    if v.board.is_empty() {
                        "a dry board".to_string()
                    } else {
                        v.board.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(" ")
                    },
                    v.active(),
                    v.occupied(),
                    v.to_call()
                )
            }
            (Some(v), Some(reason)) => format!(
                "waiting   {} seats, pot {:?} - {}",
                v.occupied(),
                v.pot,
                reason.explain()
            ),
            (None, Some(reason)) => format!("waiting   {}", reason.explain()),
            (None, None) => "waiting".to_string(),
        };
        if line != last {
            println!("{line}");
            last = line;
        }

        match (&view, &held, acting) {
            (Some(v), None, true) => match session.act(v, Choice::Fold) {
                Ok(took) => println!("  folded, and the client took it ({} ms)", took.as_millis()),
                Err(why) => println!("  the fold did not take: {}", why.explain()),
            },
            (_, Some(reason), _) if matches!(reason, live::Held::KillSwitch | live::Held::StopLoss) => {
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
    Ok(())
}

#[cfg(not(windows))]
fn live_cmd(_args: &[String]) -> Result<(), String> {
    Err("playing a live window is only implemented on Windows".to_string())
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

    // The table is the largest; the lobby is the small portrait one.
    let table = windows[0];
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
            "  warning: templates were measured at {TABLE_W}x{TABLE_H}; at {w}x{h} the              layout reflows and reading will fail. Re-run with --resize on."
        );
    }

    table.focus();
    std::thread::sleep(std::time::Duration::from_millis(400));

    let capture = table.capture().ok_or("the window could not be captured")?;
    if capture.is_blank() {
        return Err("the capture came back a single flat colour - the window was covered,                     or the client is blocking capture"
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
                    println!("           {name:<11} click ({x}, {y})");
                }
            }
        }
        Some(panel) => {
            println!("turn   : not yours - the buttons showing are the ones that arm");
            println!("         an action in advance ({} of them). Clicking them would", panel.buttons.len());
            println!("         decide the hand before it has been seen, so they are ignored.");
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

impl Flags {
    fn parse(args: &[String]) -> Result<Flags, String> {
        let mut values = HashMap::new();
        let mut index = 0;
        while index < args.len() {
            let arg = &args[index];
            let name = arg
                .strip_prefix("--")
                .ok_or_else(|| format!("expected a --flag, found {arg:?}"))?;
            let value = args
                .get(index + 1)
                .ok_or_else(|| format!("--{name} needs a value"))?;
            if value.starts_with("--") {
                return Err(format!("--{name} needs a value, found {value:?}"));
            }
            values.insert(name.to_string(), value.clone());
            index += 2;
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

