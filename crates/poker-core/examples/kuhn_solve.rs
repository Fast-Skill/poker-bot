//! Solves Kuhn poker and prints the strategy next to the analytical solution.
//!
//! Run with: `cargo run --release -p poker-core --example kuhn_solve`

use poker_core::cfr::Solver;
use poker_core::kuhn::{info_key, Kuhn, BET, JACK, KING, QUEEN};

fn main() {
    let checkpoints = [1_000usize, 10_000, 100_000, 1_000_000];
    let mut solver = Solver::new(Kuhn);
    let mut trained = 0;

    println!("Kuhn poker — vanilla CFR\n");
    println!("{:>12}  {:>14}  {:>16}", "iterations", "exploitability", "value to P0");
    println!("{}", "-".repeat(48));

    for target in checkpoints {
        solver.train(target - trained);
        trained = target;
        let profile = solver.profile();
        println!(
            "{:>12}  {:>14.6}  {:>16.6}",
            trained,
            solver.exploitability(&profile),
            solver.expected_value(&profile)
        );
    }

    println!("\nexact game value to player 0: {:.6}  (-1/18)", -1.0 / 18.0);

    let probability = |card: u8, history: u8, len: u8| {
        solver
            .average_strategy(info_key(card, history, len))
            .map(|s| s[BET])
            .unwrap_or(f64::NAN)
    };

    let alpha = probability(JACK, 0, 0);
    println!("\nPlayer 0 (alpha = {alpha:.4}, any value in [0, 1/3] is optimal)");
    println!("  {:<28} {:>8}  {:>10}", "decision", "solved", "expected");
    println!("  {}", "-".repeat(50));
    row("open J (bluff)", probability(JACK, 0, 0), alpha);
    row("open Q", probability(QUEEN, 0, 0), 0.0);
    row("open K (value)", probability(KING, 0, 0), 3.0 * alpha);
    row("call J after check-bet", probability(JACK, 0b10, 2), 0.0);
    row("call Q after check-bet", probability(QUEEN, 0b10, 2), alpha + 1.0 / 3.0);
    row("call K after check-bet", probability(KING, 0b10, 2), 1.0);

    println!("\nPlayer 1 (unique — no free parameter)");
    println!("  {:<28} {:>8}  {:>10}", "decision", "solved", "expected");
    println!("  {}", "-".repeat(50));
    row("call J facing bet", probability(JACK, 0b1, 1), 0.0);
    row("call Q facing bet", probability(QUEEN, 0b1, 1), 1.0 / 3.0);
    row("call K facing bet", probability(KING, 0b1, 1), 1.0);
    row("bet J when checked to", probability(JACK, 0b0, 1), 1.0 / 3.0);
    row("bet Q when checked to", probability(QUEEN, 0b0, 1), 0.0);
    row("bet K when checked to", probability(KING, 0b0, 1), 1.0);
}

fn row(label: &str, solved: f64, expected: f64) {
    let flag = if (solved - expected).abs() < 0.01 { "" } else { "  <-- off" };
    println!("  {label:<28} {solved:>8.4}  {expected:>10.4}{flag}");
}
