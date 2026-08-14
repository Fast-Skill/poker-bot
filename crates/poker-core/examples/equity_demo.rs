//! Prints equities for a few classic matchups.
//!
//! Run with: `cargo run --release -p poker-core --example equity_demo`

use poker_core::card::{parse_cards, Card};
use poker_core::equity::{exact, monte_carlo, Variant};
use poker_core::rng::Rng;
use std::time::Instant;

fn main() {
    println!("Hold'em — exact enumeration of every runout\n");
    println!("{:<26} {:>10} {:>10} {:>12} {:>9}", "matchup", "hand 1", "hand 2", "runouts", "time");
    println!("{}", "-".repeat(72));

    let matchups = [
        ("AA vs KK", "AsAd", "KsKd", ""),
        ("AA vs 72o", "AsAd", "7c2d", ""),
        ("QQ vs AKs", "QsQd", "AhKh", ""),
        ("AK vs AQ (dominated)", "AsKd", "AhQd", ""),
        ("AKs vs 22", "AsKs", "2c2d", ""),
        ("flush draw vs pair", "AsKs", "7d7c", "2s 9s Td"),
        ("set vs open-ended", "7d7c", "JhTh", "7s 9c 2d"),
    ];

    for (label, one, two, board) in matchups {
        let hands = [cards(one), cards(two)];
        let refs: Vec<&[Card]> = hands.iter().map(|h| h.as_slice()).collect();
        let board = cards(board);

        let started = Instant::now();
        let result = exact(&refs, &board, Variant::Holdem).expect("valid matchup");
        let elapsed = started.elapsed();

        println!(
            "{:<26} {:>9.2}% {:>9.2}% {:>12} {:>8.0?}",
            label,
            result[0].percent(),
            result[1].percent(),
            result[0].trials,
            elapsed
        );
    }

    println!("\nThree-way pot, exact\n");
    let three = [cards("AsAd"), cards("KsKd"), cards("QhQc")];
    let refs: Vec<&[Card]> = three.iter().map(|h| h.as_slice()).collect();
    let result = exact(&refs, &[], Variant::Holdem).expect("valid");
    for (hand, equity) in ["AA", "KK", "QQ"].iter().zip(result.iter()) {
        println!("  {hand}: {:.2}%", equity.percent());
    }
    let total: f64 = result.iter().map(|e| e.share).sum();
    println!("  total: {:.6}", total);

    println!("\nOmaha — sampled, since a preflop enumeration costs 60 evaluations per runout\n");
    let omaha = [cards("AsAdKsKd"), cards("7c6d5h4s")];
    let refs: Vec<&[Card]> = omaha.iter().map(|h| h.as_slice()).collect();
    let mut rng = Rng::new(2024);
    let started = Instant::now();
    let result = monte_carlo(&refs, &[], Variant::Omaha, 200_000, &mut rng).expect("valid");
    println!(
        "  AAKK double-suited: {:.2}%   vs   7654: {:.2}%   ({} trials, {:.0?})",
        result[0].percent(),
        result[1].percent(),
        result[0].trials,
        started.elapsed()
    );
}

fn cards(text: &str) -> Vec<Card> {
    parse_cards(text).expect("valid cards")
}
