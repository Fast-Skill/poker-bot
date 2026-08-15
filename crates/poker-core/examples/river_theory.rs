//! Solves the clairvoyance river game and compares against closed-form theory.
//!
//! Run with: `cargo run --release -p poker-core --example river_theory`

use poker_core::cfr::Solver;
use poker_core::river::{Holding, River, Stage};

const NUTS: usize = 0;
const AIR: usize = 1;
const BLUFF_CATCHER: usize = 0;

fn main() {
    println!("Clairvoyance river game — solved vs closed-form theory\n");
    println!(
        "{:>10}  {:>18}  {:>10}  {:>18}  {:>10}",
        "bet", "bluff share", "theory", "defence freq", "theory"
    );
    println!("{}", "-".repeat(76));

    for fraction in [0.25, 0.33, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0] {
        let spot = River::new(
            1.0,
            100.0,
            &[fraction],
            // Half the range is the nuts, half is air.
            vec![Holding::new(1_000, 0.5), Holding::new(0, 0.5)],
            // A pure bluff-catcher: beats air, loses to the nuts.
            vec![Holding::new(500, 1.0)],
        );

        let mut solver = Solver::new(spot);
        solver.train(20_000);

        let bet = River::bet_action(0);
        let value = strategy(&solver, Stage::OopFirst, NUTS)[bet];
        let bluff = strategy(&solver, Stage::OopFirst, AIR)[bet];
        let bluff_share = bluff / (value + bluff);
        let defence = strategy(&solver, Stage::IpVsBet, BLUFF_CATCHER)[River::CALL];

        println!(
            "{:>9.2}x  {:>18.4}  {:>10.4}  {:>18.4}  {:>10.4}",
            fraction,
            bluff_share,
            fraction / (1.0 + 2.0 * fraction),
            defence,
            1.0 / (1.0 + fraction),
        );
    }

    println!("\nbluff share  = s / (1 + 2s)   — of the hands that bet, this many are bluffs");
    println!("defence freq = 1 / (1 + s)     — minimum defense frequency");
    println!("\nAt a pot-sized bet that is a 2:1 value-to-bluff ratio and a 50% call.");
}

fn strategy(solver: &Solver<River>, stage: Stage, holding: usize) -> Vec<f64> {
    solver
        .average_strategy(River::info_key(stage, holding))
        .expect("information set was visited")
}
