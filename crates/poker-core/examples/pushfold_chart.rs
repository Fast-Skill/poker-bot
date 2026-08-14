//! Solves heads-up push/fold Hold'em and prints the resulting Nash ranges.
//!
//! Run with: `cargo run --release -p poker-core --example pushfold_chart`

use poker_core::abstraction::HandClass;
use poker_core::card::{Rank, NUM_RANKS};
use poker_core::cfr::Solver;
use poker_core::pushfold::{EquityTable, PushFold, PUSH};
use poker_core::rng::Rng;
use std::time::Instant;

const EQUITY_SAMPLES: u32 = 4_000;
const ITERATIONS: usize = 3_000_000;

fn main() {
    print!("building the 169x169 equity table ({EQUITY_SAMPLES} samples per pairing)... ");
    let started = Instant::now();
    let mut rng = Rng::new(0x9E3779B9);
    let equity = EquityTable::sampled(EQUITY_SAMPLES, &mut rng);
    println!("{:.1?}", started.elapsed());

    println!("\nRange widths by stack depth (share of all 1,326 combos)\n");
    println!("{:>8}  {:>10}  {:>10}  {:>16}", "stack", "SB push", "BB call", "SB EV (bb/hand)");
    println!("{}", "-".repeat(52));

    let depths = [3.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0];
    let mut solved = Vec::new();

    for stack in depths {
        let mut rng = Rng::new(0xF01D);
        let mut solver = Solver::new(PushFold::new(stack, equity.clone()));
        solver.train_sampled(ITERATIONS, &mut rng);

        let profile = solver.profile();
        println!(
            "{:>7.0}bb  {:>9.1}%  {:>9.1}%  {:>16.4}",
            stack,
            range_width(&solver, 0) * 100.0,
            range_width(&solver, 1) * 100.0,
            solver.expected_value(&profile),
        );
        solved.push((stack, solver));
    }

    for (stack, solver) in &solved {
        if *stack == 10.0 {
            println!("\n\nSmall blind push range at {stack:.0}bb");
            print_grid(solver, 0);
            println!("\nBig blind call range at {stack:.0}bb");
            print_grid(solver, 1);
        }
    }

    println!("\nlegend:  # always   + mostly   . sometimes   (blank) never");
}

/// Share of all 1,326 combinations played aggressively.
fn range_width(solver: &Solver<PushFold>, player: usize) -> f64 {
    let mut combos = 0.0;
    for hand in HandClass::all() {
        let key = PushFold::info_key(player, hand.index());
        if let Some(strategy) = solver.average_strategy(key) {
            combos += strategy[PUSH] * hand.combos() as f64;
        }
    }
    combos / 1_326.0
}

/// Prints the standard 13x13 hand grid: pairs on the diagonal, suited above,
/// offsuit below.
fn print_grid(solver: &Solver<PushFold>, player: usize) {
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
            let high = (NUM_RANKS - 1 - row.min(column)) as u8;
            let low = (NUM_RANKS - 1 - row.max(column)) as u8;
            let suited = row < column;
            let index = if suited || high == low {
                high as usize * NUM_RANKS + low as usize
            } else {
                low as usize * NUM_RANKS + high as usize
            };

            let key = PushFold::info_key(player, index);
            let frequency = solver
                .average_strategy(key)
                .map(|s| s[PUSH])
                .unwrap_or(0.0);
            let mark = match frequency {
                f if f > 0.95 => '#',
                f if f > 0.50 => '+',
                f if f > 0.05 => '.',
                _ => ' ',
            };
            print!(" {mark} ");
        }
        println!();
    }
}
