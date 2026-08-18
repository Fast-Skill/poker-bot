//! Splitting a pot between three or more players.
//!
//! Heads-up, equity is a single number: your share, and the opponent takes the
//! rest. Three-handed it is not, and the gap is not a rounding detail. Knowing
//! that A beats B 60% of the time and A beats C 60% of the time does **not**
//! give A's share of a three-way pot, because the two events are correlated —
//! one board settles both matchups at once.
//!
//! Computing the real answer means enumerating boards for every triple of hand
//! classes: 169^3, roughly 4.8 million combinations. Too much to hold in memory,
//! and far too much to recompute inside a solver's inner loop.
//!
//! So the obvious shortcut is to estimate three-way shares from pairwise ones.
//! [`approximate_shares`] does that, [`exact_shares`] enumerates the truth, and
//! the tests below compare them.
//!
//! # The shortcut does not work
//!
//! Measured, not assumed. On `9h 4d 2s` holding AA, KK and 72o, the estimate
//! gives 72o a 4% share where the real figure is 18% — and ranks it *below* KK,
//! which is backwards.
//!
//! The reason is structural. 72o has paired the deuce, and when it improves to
//! trips or two pair it beats both opponents at once, off the same board. Wins
//! in a multiway pot are correlated, so multiplying two independent-looking
//! pairwise numbers systematically buries every hand that wins by improving —
//! which is most of what makes multiway play different from heads-up.
//!
//! `approximate_shares` is therefore kept as a documented negative result, not
//! as something to solve with. A real multiway solve needs three-way equities
//! sampled directly, over bucketed hand classes to keep the table small enough
//! to build.

use crate::card::{Card, CardSet};
use crate::eval::evaluate;

/// Estimates each player's share of a pot from pairwise equities.
///
/// `pairwise[i][j]` is the share player `i` takes against player `j` alone; the
/// diagonal is ignored.
///
/// **Not fit for solving.** Kept because it is the obvious approach and someone
/// will reach for it; the tests record precisely how it fails.
///
/// The estimate treats "beats `j`" and "beats `k`" as independent, so a
/// player's weight is the product of their pairwise equities, normalised across
/// the table. Independence is false — one board decides every matchup at once —
/// and the error is not merely large, it reverses the ordering of hands. See
/// the module documentation.
///
/// # Panics
/// Panics with fewer than two players, or if the matrix is not square.
pub fn approximate_shares(pairwise: &[Vec<f64>]) -> Vec<f64> {
    let players = pairwise.len();
    assert!(players >= 2, "need at least two players");
    assert!(
        pairwise.iter().all(|row| row.len() == players),
        "the pairwise matrix must be square"
    );

    let weights: Vec<f64> = (0..players)
        .map(|hero| {
            (0..players)
                .filter(|villain| *villain != hero)
                .map(|villain| pairwise[hero][villain])
                .product()
        })
        .collect();

    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        // Every player losing to every other is impossible; split the pot
        // rather than divide by zero.
        return vec![1.0 / players as f64; players];
    }
    weights.iter().map(|weight| weight / total).collect()
}

/// Each player's exact share, by enumerating every remaining board.
///
/// Chops are split among the players who tie, so shares always sum to one.
/// Cost grows as `C(remaining, missing)`, which makes this a measurement tool
/// rather than something to call from a solver.
///
/// # Panics
/// Panics with fewer than two hands, if any card repeats, or if the board holds
/// more than five cards.
pub fn exact_shares(hands: &[[Card; 2]], board: &[Card]) -> Vec<f64> {
    assert!(hands.len() >= 2, "need at least two hands");
    assert!(board.len() <= 5, "a board is at most five cards");

    let mut dead = CardSet::empty();
    for hand in hands {
        for card in hand {
            assert!(dead.insert(*card), "card {card} appears twice");
        }
    }
    for card in board {
        assert!(dead.insert(*card), "card {card} appears twice");
    }

    let deck: Vec<Card> = CardSet::full_deck().difference(dead).iter().collect();
    let missing = 5 - board.len();

    let mut shares = vec![0.0; hands.len()];
    let mut trials = 0u64;
    let mut full_board = Vec::with_capacity(5);
    let mut scratch = Vec::with_capacity(7);
    let mut chosen = Vec::with_capacity(missing);
    let mut winners: Vec<usize> = Vec::with_capacity(hands.len());

    enumerate(&deck, missing, 0, &mut chosen, &mut |runout| {
        full_board.clear();
        full_board.extend_from_slice(board);
        full_board.extend_from_slice(runout);

        let mut best = None;
        winners.clear();
        for (index, hand) in hands.iter().enumerate() {
            scratch.clear();
            scratch.extend_from_slice(hand);
            scratch.extend_from_slice(&full_board);
            let rank = evaluate(&scratch);
            match best {
                None => {
                    best = Some(rank);
                    winners.push(index);
                }
                Some(current) if rank > current => {
                    best = Some(rank);
                    winners.clear();
                    winners.push(index);
                }
                Some(current) if rank == current => winners.push(index),
                Some(_) => {}
            }
        }

        let split = 1.0 / winners.len() as f64;
        for winner in winners.iter() {
            shares[*winner] += split;
        }
        trials += 1;
    });

    if trials == 0 {
        return vec![1.0 / hands.len() as f64; hands.len()];
    }
    shares.iter().map(|share| share / trials as f64).collect()
}

/// Calls `f` with every `k`-card combination from `deck[start..]`.
fn enumerate(
    deck: &[Card],
    k: usize,
    start: usize,
    chosen: &mut Vec<Card>,
    f: &mut impl FnMut(&[Card]),
) {
    if chosen.len() == k {
        f(chosen);
        return;
    }
    let needed = k - chosen.len();
    if deck.len() - start < needed {
        return;
    }
    for index in start..=deck.len() - needed {
        chosen.push(deck[index]);
        enumerate(deck, k, index + 1, chosen, f);
        chosen.pop();
    }
}

/// The pairwise equity matrix for a set of hands on a board.
///
/// Entry `[i][j]` is `i`'s share against `j` alone, which feeds straight into
/// [`approximate_shares`].
pub fn pairwise_matrix(hands: &[[Card; 2]], board: &[Card]) -> Vec<Vec<f64>> {
    let players = hands.len();
    let mut matrix = vec![vec![0.5; players]; players];
    for hero in 0..players {
        for villain in hero + 1..players {
            let shares = exact_shares(&[hands[hero], hands[villain]], board);
            matrix[hero][villain] = shares[0];
            matrix[villain][hero] = shares[1];
        }
    }
    matrix
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::parse_cards;

    fn hands(text: &[&str]) -> Vec<[Card; 2]> {
        text.iter()
            .map(|hand| {
                let cards = parse_cards(hand).expect("valid hand");
                assert_eq!(cards.len(), 2);
                [cards[0], cards[1]]
            })
            .collect()
    }

    fn board(text: &str) -> Vec<Card> {
        parse_cards(text).expect("valid board")
    }

    #[test]
    fn shares_always_sum_to_one() {
        for (players, table) in [
            (vec!["AsAd", "KsKd"], "2c 7d 9h Jc 4s"),
            (vec!["AsAd", "KsKd", "QhQc"], "2c 7d 9h"),
            (vec!["AsAd", "KsKd", "QhQc", "JhJd"], "2c 7d 9h Jc 4s"),
        ] {
            let shares = exact_shares(&hands(&players), &board(table));
            let total: f64 = shares.iter().sum();
            assert!((total - 1.0).abs() < 1e-9, "summed to {total}");
        }
    }

    #[test]
    fn a_decided_board_gives_the_winner_everything() {
        let shares = exact_shares(&hands(&["AsAd", "KsKd", "QhQc"]), &board("2c 7d 9h Jc 4s"));
        assert_eq!(shares, vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn a_board_that_plays_itself_is_split_evenly() {
        // Everyone plays the same royal flush.
        let shares = exact_shares(&hands(&["2c3d", "4h5s", "6c7d"]), &board("As Ks Qs Js Ts"));
        for share in &shares {
            assert!((share - 1.0 / 3.0).abs() < 1e-9, "got {share}");
        }
    }

    #[test]
    fn the_approximation_is_exact_heads_up() {
        // With two players the "product of the others" is a single pairwise
        // number, so the estimate is not an estimate at all.
        let matrix = vec![vec![0.5, 0.82], vec![0.18, 0.5]];
        let shares = approximate_shares(&matrix);
        assert!((shares[0] - 0.82).abs() < 1e-9, "got {}", shares[0]);
        assert!((shares[1] - 0.18).abs() < 1e-9);
    }

    /// The finding that rules the shortcut out, pinned so nobody re-adopts it.
    #[test]
    fn the_product_approximation_misorders_a_multiway_pot() {
        // 72o has paired the board's deuce. When it improves it beats both
        // opponents off the same card, so its wins are correlated — exactly
        // what a product of pairwise equities cannot represent.
        let cards = hands(&["AsAd", "KsKd", "7c2h"]);
        let table = board("9h 4d 2s");
        let estimate = approximate_shares(&pairwise_matrix(&cards, &table));
        let truth = exact_shares(&cards, &table);

        // The truth: the pair of deuces is worth more than the overpair that
        // is drawing dead to it.
        assert!(truth[2] > truth[1], "72o should beat KK for share: {truth:?}");
        // The estimate gets that backwards.
        assert!(
            estimate[2] < estimate[1],
            "the approximation was expected to misorder these: {estimate:?}"
        );
        // And badly: four percent against a real eighteen.
        assert!(
            (truth[2] - estimate[2]) > 0.10,
            "understated by {:.3}, less than the documented error",
            truth[2] - estimate[2]
        );
    }

    /// The measurement that justifies using the approximation at all.
    #[test]
    fn the_approximation_error_is_bounded() {
        let spots = [
            (vec!["AsAd", "KsKd", "QhQc"], "9h 4d 2s"),
            (vec!["AsKs", "QdQh", "7c2h"], "Js 8s 3d"),
            (vec!["AhKh", "9s9d", "5c4c"], "Kc 9h 2s"),
            (vec!["AsAd", "7h6h", "Tc9c"], "8h 5s 2d"),
        ];

        let mut worst: f64 = 0.0;
        for (players, table) in spots {
            let cards = hands(&players);
            let felt = board(table);
            let estimate = approximate_shares(&pairwise_matrix(&cards, &felt));
            let truth = exact_shares(&cards, &felt);
            for (approx, exact) in estimate.iter().zip(truth.iter()) {
                worst = worst.max((approx - exact).abs());
            }
        }

        // Around twelve points of share, which is far too much to solve with —
        // a bet sized on a 4% share when the truth is 18% is simply a
        // different decision. Recorded so the number is on the record rather
        // than rediscovered later.
        assert!(
            worst > 0.08,
            "expected the approximation to be badly wrong; worst was only {worst:.4}"
        );
        assert!(worst < 0.20, "worse than previously measured: {worst:.4}");
        println!("worst three-way share error: {worst:.4}");
    }

    #[test]
    #[should_panic(expected = "appears twice")]
    fn a_repeated_card_is_rejected() {
        exact_shares(&hands(&["AsAd", "AsKd"]), &board("2c 7d 9h"));
    }

    #[test]
    #[should_panic(expected = "at least two")]
    fn a_single_hand_is_rejected() {
        exact_shares(&hands(&["AsAd"]), &board("2c 7d 9h"));
    }
}
