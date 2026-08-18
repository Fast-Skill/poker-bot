//! Measures how much of a real session the two-player blueprint can decide.
use poker_core::bench::{ring_match, ChartBot};
use poker_core::blueprint::Blueprint;
use poker_core::bot::BlueprintAgent;
use poker_core::preflop::Sizing;
use poker_core::rng::Rng;
use poker_core::table::{Agent, Table};

fn main() {
    let blueprint = Blueprint::load("data/preflop-100bb.bin").expect("solve preflop first");
    println!("Blueprint coverage by table size (20000 hands each)\n");
    println!("{:>8}  {:>10}  {:>12}", "seats", "coverage", "bot bb/100");
    println!("{}", "-".repeat(34));

    for seats in 2..=6usize {
        let mut hero = BlueprintAgent::new("bot", blueprint.clone(), Sizing::default());
        let mut others: Vec<ChartBot> = (0..seats - 1).map(|_| ChartBot::default()).collect();
        let mut refs: Vec<&mut dyn Agent> = vec![&mut hero];
        for other in others.iter_mut() {
            refs.push(other as &mut dyn Agent);
        }
        let mut rng = Rng::new(0xC0DE);
        let report = ring_match(&Table::standard(), refs, 20000, &mut rng);
        println!(
            "{seats:>8}  {:>9.1}%  {:>12.1}",
            hero.coverage_fraction() * 100.0,
            report.bb_per_100[0]
        );
    }
    println!("\ncoverage = share of decisions the two-player solve could make");
}
