//! Solves a real river spot: actual board, actual ranges, actual cards.
//!
//! Run with: `cargo run --release -p poker-core --example river_spot`

use poker_core::card::parse_cards;
use poker_core::cfr::Solver;
use poker_core::eval::{Category, HandRank};
use poker_core::range::Range;
use poker_core::river::{Holding, River, Stage};
use poker_core::rng::Rng;

const BOARD: &str = "Qs 8s 3s 7d 2c";
const OOP_RANGE: &str = "JJ+,AQs+,AKo";
const IP_RANGE: &str = "88+,ATs+,KQs,AQo+";
const POT: f64 = 20.0;
const STACK: f64 = 60.0;
const ITERATIONS: usize = 3_000_000;
/// Fixed so the printed solve is reproducible.
const SEED: u64 = 0x5217_3EF6_9A1C_0B44;

fn main() {
    let board = parse_cards(BOARD).expect("valid board");
    let oop: Range = OOP_RANGE.parse().expect("valid range");
    let ip: Range = IP_RANGE.parse().expect("valid range");

    let oop_holdings = oop.holdings(&board);
    let ip_holdings = ip.holdings(&board);

    println!("River spot\n");
    println!("  board             {BOARD}");
    println!("  pot               {POT:.0} bb, {STACK:.0} bb behind");
    println!(
        "  out of position   {OOP_RANGE}   ({} combos here)",
        oop_holdings.len()
    );
    println!(
        "  in position       {IP_RANGE}   ({} combos here)",
        ip_holdings.len()
    );

    let spot = River::new(
        POT,
        STACK,
        &[0.33, 0.75],
        &[1.0],
        oop_holdings.clone(),
        ip_holdings.clone(),
    );
    let bets = spot.bet_sizes();
    println!("  bet sizes         {bets:?} bb");

    let mut rng = Rng::new(SEED);
    let mut solver = Solver::new(spot);
    solver.train_sampled(ITERATIONS, &mut rng);

    println!("\nOut of position, first to act\n");
    println!("{:>18}  {:>7}  {:>9}  {:>9}", "made hand", "combos", "check", "bet");
    println!("{}", "-".repeat(50));
    for (category, members) in by_category(&oop_holdings) {
        let mut check = 0.0;
        let mut bet = 0.0;
        for &index in &members {
            let strategy = strategy_at(&solver, Stage::OopFirst, index, 0);
            check += strategy[River::PASSIVE];
            bet += strategy[1..].iter().sum::<f64>();
        }
        let n = members.len() as f64;
        println!(
            "{:>18}  {:>7}  {:>8.1}%  {:>8.1}%",
            category.name(),
            members.len(),
            check / n * 100.0,
            bet / n * 100.0
        );
    }

    println!("\nIn position, facing the {:.1} bb bet\n", bets[1]);
    println!(
        "{:>18}  {:>7}  {:>9}  {:>9}  {:>9}",
        "made hand", "combos", "fold", "call", "raise"
    );
    println!("{}", "-".repeat(62));
    for (category, members) in by_category(&ip_holdings) {
        let (mut fold, mut call, mut raise) = (0.0, 0.0, 0.0);
        for &index in &members {
            let strategy = strategy_at(&solver, Stage::IpVsBet, index, 1);
            fold += strategy[River::PASSIVE];
            call += strategy[River::CALL];
            raise += strategy.get(River::raise_action(0)).copied().unwrap_or(0.0);
        }
        let n = members.len() as f64;
        println!(
            "{:>18}  {:>7}  {:>8.1}%  {:>8.1}%  {:>8.1}%",
            category.name(),
            members.len(),
            fold / n * 100.0,
            call / n * 100.0,
            raise / n * 100.0
        );
    }

    println!(
        "\nexploitability: {:.4} bb per hand",
        solver.exploitability(&solver.profile())
    );
}

fn strategy_at(solver: &Solver<River>, stage: Stage, holding: usize, bet: usize) -> Vec<f64> {
    solver
        .average_strategy(River::info_key(stage, holding, bet, 0))
        .expect("information set was visited")
}

/// Groups holding indices by the made hand they hold, strongest category first.
fn by_category(holdings: &[Holding]) -> Vec<(Category, Vec<usize>)> {
    let mut groups: Vec<(Category, Vec<usize>)> = Vec::new();
    for category in Category::ALL.iter().rev() {
        let members: Vec<usize> = holdings
            .iter()
            .enumerate()
            .filter(|(_, holding)| {
                HandRank::from_bits(holding.strength)
                    .is_some_and(|rank| rank.category() == *category)
            })
            .map(|(index, _)| index)
            .collect();
        if !members.is_empty() {
            groups.push((*category, members));
        }
    }
    groups
}

