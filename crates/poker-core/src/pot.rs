//! Pot construction and showdown distribution.
//!
//! When players go all in for different amounts, the money splits into a main
//! pot and one or more side pots. Each pot can only be won by players who
//! contributed to it, so a short all-in player cannot win chips they never
//! covered.
//!
//! The rule that catches people out: a player who folds still leaves their
//! chips in the pot, and those chips are still won by someone — the folder is
//! simply not eligible. Contribution and eligibility are separate questions,
//! and this module keeps them separate.
//!
//! # Layer algorithm
//!
//! Sort the distinct contribution levels. Each adjacent pair of levels defines
//! a horizontal layer of the pot, funded by every player who reached that
//! level, and winnable by those of them who did not fold. Adjacent layers with
//! identical eligibility are merged, so the output matches how a dealer would
//! actually describe the pots.

use crate::eval::HandRank;

/// A pot or side pot: an amount, and the players who may win it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pot {
    /// Chips in this pot.
    pub amount: u64,
    /// Seat indices eligible to win it, ascending. Empty only if every
    /// contributor folded, which [`award`] treats as an error.
    pub eligible: Vec<usize>,
}

/// Splits total per-player contributions into a main pot and any side pots.
///
/// `contributed[i]` is everything player `i` put in across the whole hand, and
/// `folded[i]` is whether they folded. The returned pots are ordered main pot
/// first.
///
/// Chips are conserved exactly: the pot amounts always sum to the total
/// contributed.
///
/// # Invariant
/// The largest contributor should not be folded. That state cannot arise at a
/// real table — a player holding the largest contribution was never raised, so
/// had nothing to fold to — and it would produce a pot no one is eligible to
/// win. Chips are still conserved if it happens; [`award`] is what rejects it.
///
/// # Panics
/// Panics if the two slices have different lengths.
pub fn build_pots(contributed: &[u64], folded: &[bool]) -> Vec<Pot> {
    assert_eq!(
        contributed.len(),
        folded.len(),
        "contributed and folded must describe the same players"
    );

    let mut levels: Vec<u64> = contributed.iter().copied().filter(|&c| c > 0).collect();
    levels.sort_unstable();
    levels.dedup();

    let mut pots: Vec<Pot> = Vec::new();
    let mut previous = 0u64;

    for level in levels {
        let layer = level - previous;
        previous = level;
        if layer == 0 {
            continue;
        }

        // Everyone who reached this level funds the layer, folded or not.
        let funders = contributed.iter().filter(|&&c| c >= level).count() as u64;
        let amount = layer * funders;
        if amount == 0 {
            continue;
        }

        let eligible: Vec<usize> = (0..contributed.len())
            .filter(|&i| contributed[i] >= level && !folded[i])
            .collect();

        // Adjacent layers with the same eligible set are one pot in practice.
        match pots.last_mut() {
            Some(last) if last.eligible == eligible => last.amount += amount,
            _ => pots.push(Pot { amount, eligible }),
        }
    }

    pots
}

/// How an odd chip that cannot be split evenly is assigned.
///
/// Real card rooms award it to the first eligible seat left of the button.
/// Callers that care pass [`OddChip::ToSeat`] with that seat; the default is
/// deterministic rather than realistic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OddChip {
    /// Give the remainder to the lowest-numbered winning seat.
    #[default]
    ToLowestSeat,
    /// Give the remainder to the first winning seat at or after this one,
    /// wrapping around. Use the seat left of the button to match card-room
    /// rules.
    ToSeat(usize),
}

/// Distributes every pot to its best eligible hand, splitting ties.
///
/// `ranks[i]` is player `i`'s showdown hand, or `None` if they folded or are
/// otherwise not at showdown. Returns per-player winnings, which sum to the
/// total in `pots`.
///
/// # Panics
/// Panics if `ranks` is shorter than the highest seat index in `pots`, or if a
/// pot has no eligible player with a hand — both mean the caller built an
/// inconsistent showdown.
pub fn award(pots: &[Pot], ranks: &[Option<HandRank>], odd_chip: OddChip) -> Vec<u64> {
    let mut winnings = vec![0u64; ranks.len()];

    for pot in pots {
        let best = pot
            .eligible
            .iter()
            .filter_map(|&seat| {
                assert!(
                    seat < ranks.len(),
                    "pot references seat {seat} but only {} players were given hands",
                    ranks.len()
                );
                ranks[seat].map(|rank| (seat, rank))
            })
            .map(|(_, rank)| rank)
            .max()
            .unwrap_or_else(|| {
                panic!("pot of {} has no eligible player with a hand", pot.amount)
            });

        let winners: Vec<usize> = pot
            .eligible
            .iter()
            .copied()
            .filter(|&seat| ranks[seat] == Some(best))
            .collect();

        let share = pot.amount / winners.len() as u64;
        let mut remainder = pot.amount % winners.len() as u64;
        for &seat in &winners {
            winnings[seat] += share;
        }

        // Hand out the indivisible chips one at a time, starting from the
        // configured seat and wrapping.
        if remainder > 0 {
            let start = match odd_chip {
                OddChip::ToLowestSeat => 0,
                OddChip::ToSeat(seat) => seat,
            };
            let mut order: Vec<usize> = winners.clone();
            order.sort_by_key(|&seat| {
                // Distance from `start`, wrapping past the end of the table.
                (seat + ranks.len() - start % ranks.len()) % ranks.len()
            });
            for &seat in order.iter() {
                if remainder == 0 {
                    break;
                }
                winnings[seat] += 1;
                remainder -= 1;
            }
        }
    }

    winnings
}

/// Total chips held across a set of pots.
pub fn total(pots: &[Pot]) -> u64 {
    pots.iter().map(|p| p.amount).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::parse_cards;
    use crate::eval::evaluate;

    fn rank(s: &str) -> Option<HandRank> {
        Some(evaluate(&parse_cards(s).expect("valid cards")))
    }

    #[test]
    fn equal_contributions_make_a_single_pot() {
        let pots = build_pots(&[100, 100, 100], &[false, false, false]);
        assert_eq!(pots.len(), 1);
        assert_eq!(pots[0].amount, 300);
        assert_eq!(pots[0].eligible, vec![0, 1, 2]);
    }

    #[test]
    fn a_folded_player_funds_the_pot_but_cannot_win_it() {
        // Seat 1 folded after putting in a full 100.
        let pots = build_pots(&[100, 100, 100], &[false, true, false]);
        assert_eq!(total(&pots), 300, "folded chips stay in the pot");
        assert_eq!(pots.len(), 1);
        assert_eq!(pots[0].eligible, vec![0, 2], "seat 1 is not eligible");
    }

    #[test]
    fn a_short_all_in_creates_one_side_pot() {
        // Seat 0 is all in for 50; seats 1 and 2 contest 100 each.
        let pots = build_pots(&[50, 100, 100], &[false, false, false]);
        assert_eq!(pots.len(), 2);

        // Main pot: 50 from each of the three.
        assert_eq!(pots[0].amount, 150);
        assert_eq!(pots[0].eligible, vec![0, 1, 2]);

        // Side pot: the extra 50 from seats 1 and 2 only.
        assert_eq!(pots[1].amount, 100);
        assert_eq!(pots[1].eligible, vec![1, 2]);

        assert_eq!(total(&pots), 250);
    }

    #[test]
    fn three_all_in_levels_create_three_pots() {
        let pots = build_pots(&[25, 50, 100, 100], &[false, false, false, false]);
        assert_eq!(pots.len(), 3);

        assert_eq!(pots[0].amount, 100); // 25 x 4
        assert_eq!(pots[0].eligible, vec![0, 1, 2, 3]);

        assert_eq!(pots[1].amount, 75); // 25 x 3
        assert_eq!(pots[1].eligible, vec![1, 2, 3]);

        assert_eq!(pots[2].amount, 100); // 50 x 2
        assert_eq!(pots[2].eligible, vec![2, 3]);

        assert_eq!(total(&pots), 275);
    }

    #[test]
    fn an_uncalled_bet_becomes_a_pot_only_its_bettor_can_win() {
        // Seat 0 bet 100, everyone folded to it having put in 20.
        let pots = build_pots(&[100, 20, 20], &[false, true, true]);
        assert_eq!(total(&pots), 140);
        let solo = pots.last().expect("a pot exists");
        assert_eq!(solo.eligible, vec![0], "only the bettor can win the excess");
    }

    #[test]
    fn adjacent_layers_with_equal_eligibility_are_merged() {
        // Seats 1 and 2 folded at different amounts. That splits the money into
        // three layers (0-30, 30-60, 60-100), but seat 0 is the only eligible
        // player in all of them, so a dealer would call it one pot — and so
        // should we, rather than reporting three side pots with one winner each.
        let pots = build_pots(&[100, 30, 60], &[false, true, true]);
        assert_eq!(pots.len(), 1, "three layers, one eligibility set, one pot");
        assert_eq!(pots[0].amount, 190);
        assert_eq!(pots[0].eligible, vec![0]);
    }

    #[test]
    fn layers_with_different_eligibility_stay_separate() {
        // The mirror of the merge case: here eligibility genuinely changes at
        // each level, so the pots must not be collapsed.
        let pots = build_pots(&[100, 30, 60], &[false, false, false]);
        assert_eq!(pots.len(), 3);
        assert_eq!(pots[0].eligible, vec![0, 1, 2]);
        assert_eq!(pots[1].eligible, vec![0, 2]);
        assert_eq!(pots[2].eligible, vec![0]);
        assert_eq!(total(&pots), 190);
    }

    #[test]
    fn players_who_never_put_money_in_are_ignored() {
        let pots = build_pots(&[0, 100, 100], &[true, false, false]);
        assert_eq!(pots.len(), 1);
        assert_eq!(pots[0].amount, 200);
        assert_eq!(pots[0].eligible, vec![1, 2]);
    }

    #[test]
    fn an_empty_table_makes_no_pots() {
        assert!(build_pots(&[], &[]).is_empty());
        assert!(build_pots(&[0, 0], &[false, false]).is_empty());
    }

    #[test]
    #[should_panic(expected = "same players")]
    fn mismatched_input_lengths_are_rejected() {
        build_pots(&[100, 100], &[false]);
    }

    #[test]
    fn the_best_hand_takes_the_pot() {
        let pots = build_pots(&[100, 100], &[false, false]);
        let ranks = [rank("As Ks Qs Js Ts"), rank("2c 3d 4h 5s 7c")];
        let paid = award(&pots, &ranks, OddChip::default());
        assert_eq!(paid, vec![200, 0]);
    }

    #[test]
    fn tied_hands_split_the_pot() {
        let pots = build_pots(&[100, 100], &[false, false]);
        // Same hand in different suits.
        let ranks = [rank("Ac Kd 9h 5s 2c"), rank("Ad Kh 9s 5c 2d")];
        let paid = award(&pots, &ranks, OddChip::default());
        assert_eq!(paid, vec![100, 100]);
    }

    #[test]
    fn an_odd_chip_goes_to_the_configured_seat() {
        let pots = vec![Pot { amount: 101, eligible: vec![0, 1] }];
        let ranks = [rank("Ac Kd 9h 5s 2c"), rank("Ad Kh 9s 5c 2d")];

        let low = award(&pots, &ranks, OddChip::ToLowestSeat);
        assert_eq!(low, vec![51, 50]);

        let seat1 = award(&pots, &ranks, OddChip::ToSeat(1));
        assert_eq!(seat1, vec![50, 51], "the extra chip follows the button rule");

        assert_eq!(low.iter().sum::<u64>(), 101, "chips are conserved");
    }

    #[test]
    fn a_short_stack_can_only_win_the_main_pot() {
        // Seat 0 is all in for 50 with the nuts; seats 1 and 2 contest the side
        // pot, which seat 0 cannot touch.
        let pots = build_pots(&[50, 100, 100], &[false, false, false]);
        let ranks = [
            rank("As Ks Qs Js Ts"), // straight flush, best hand
            rank("9c 9d 9h 9s 2c"), // quads
            rank("2c 3d 4h 5s 7c"), // nothing
        ];
        let paid = award(&pots, &ranks, OddChip::default());

        assert_eq!(paid[0], 150, "wins the main pot only");
        assert_eq!(paid[1], 100, "best remaining hand wins the side pot");
        assert_eq!(paid[2], 0);
        assert_eq!(paid.iter().sum::<u64>(), total(&pots));
    }

    #[test]
    fn a_folded_player_is_paid_nothing_even_with_the_best_hand() {
        let pots = build_pots(&[100, 100], &[false, true]);
        // Seat 1 folded the winning hand; it must not be counted.
        let ranks = [rank("2c 3d 4h 5s 7c"), rank("As Ks Qs Js Ts")];
        let paid = award(&pots, &ranks, OddChip::default());
        assert_eq!(paid, vec![200, 0]);
    }

    #[test]
    #[should_panic(expected = "no eligible player")]
    fn a_pot_with_no_live_hand_is_a_caller_bug() {
        let pots = vec![Pot { amount: 100, eligible: vec![0] }];
        award(&pots, &[None], OddChip::default());
    }

    /// Chips must never be created or destroyed, whatever the contribution
    /// pattern. Driven by a small deterministic generator so failures reproduce.
    #[test]
    fn chips_are_conserved_across_many_random_tables() {
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            // xorshift64*, enough for spreading test inputs around.
            seed ^= seed >> 12;
            seed ^= seed << 25;
            seed ^= seed >> 27;
            seed.wrapping_mul(0x2545_F491_4F6C_DD1D)
        };

        for _ in 0..2000 {
            let players = 2 + (next() % 8) as usize;
            let contributed: Vec<u64> = (0..players).map(|_| next() % 500).collect();
            let mut folded: Vec<bool> = (0..players).map(|_| next() % 3 == 0).collect();

            // Keep the biggest contributor live. A player cannot fold while
            // holding the largest contribution — nobody raised them — so
            // generating that state would be testing an impossible table.
            let top = (0..players).max_by_key(|&i| contributed[i]).expect("non-empty");
            folded[top] = false;

            let pots = build_pots(&contributed, &folded);
            assert_eq!(
                total(&pots),
                contributed.iter().sum::<u64>(),
                "chip leak for {contributed:?} / {folded:?}"
            );

            // Every pot must be winnable by somebody, or the table state was
            // impossible to begin with (everyone folded).
            let all_folded = folded.iter().all(|&f| f);
            if !all_folded {
                for pot in &pots {
                    assert!(
                        !pot.eligible.is_empty() || pot.amount == 0,
                        "unwinnable pot for {contributed:?} / {folded:?}"
                    );
                }
            }
        }
    }
}
