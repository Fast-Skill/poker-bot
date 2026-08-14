//! Hand-versus-hand equity: the share of the pot a holding wins on average.
//!
//! Two engines, one interface. [`exact`] enumerates every remaining board
//! runout and is the ground truth. [`monte_carlo`] samples runouts and is used
//! when enumeration is too large — mainly Omaha preflop, where each runout
//! costs 60 five-card evaluations.
//!
//! Split pots are counted fractionally: a three-way tie awards each player a
//! third, not a win. Counting ties as wins inflates equity in exactly the spots
//! where accuracy matters most, since the marginal calling decisions a solver
//! cares about are the ones near a chop.

use crate::card::{Card, CardSet};
use crate::eval::{evaluate, HandRank};
use crate::omaha::evaluate_omaha;
use crate::rng::Rng;
use std::fmt;

/// Which game's hand-forming rules apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    /// Any five of the seven cards play.
    Holdem,
    /// Exactly two hole cards and exactly three board cards play.
    Omaha,
}

impl Variant {
    /// The hole-card counts this variant accepts.
    const fn hole_cards(self) -> (usize, usize) {
        match self {
            Variant::Holdem => (2, 2),
            // Four is standard; five and six card Omaha are played too.
            Variant::Omaha => (4, 6),
        }
    }
}

/// One player's share of the pot across all trials.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Equity {
    /// Trials this hand won outright.
    pub wins: u64,
    /// Trials this hand tied for best.
    pub ties: u64,
    /// Trials evaluated.
    pub trials: u64,
    /// Expected share of the pot in `0.0..=1.0`, counting chops fractionally.
    pub share: f64,
}

impl Equity {
    /// Share expressed as a percentage.
    pub fn percent(&self) -> f64 {
        self.share * 100.0
    }
}

impl fmt::Display for Equity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:.2}% ({} win, {} tie, {} trials)",
            self.percent(),
            self.wins,
            self.ties,
            self.trials
        )
    }
}

/// Why an equity calculation could not be run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EquityError {
    /// Fewer than two hands were given.
    TooFewHands(usize),
    /// A hand had the wrong number of hole cards for the variant.
    WrongHoleCount {
        hand: usize,
        found: usize,
        expected: (usize, usize),
    },
    /// More than five board cards.
    BoardTooLong(usize),
    /// The same card appeared twice across hands and board.
    DuplicateCard(Card),
    /// Not enough cards left in the deck to complete the board.
    DeckExhausted { needed: usize, available: usize },
}

impl fmt::Display for EquityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EquityError::TooFewHands(n) => write!(f, "need at least 2 hands, got {n}"),
            EquityError::WrongHoleCount {
                hand,
                found,
                expected: (lo, hi),
            } => {
                if lo == hi {
                    write!(f, "hand {hand} has {found} cards, expected {lo}")
                } else {
                    write!(f, "hand {hand} has {found} cards, expected {lo} to {hi}")
                }
            }
            EquityError::BoardTooLong(n) => write!(f, "board has {n} cards, at most 5 allowed"),
            EquityError::DuplicateCard(card) => write!(f, "card {card} appears more than once"),
            EquityError::DeckExhausted { needed, available } => {
                write!(f, "need {needed} more board cards but only {available} remain")
            }
        }
    }
}

impl std::error::Error for EquityError {}

/// Validates the inputs and returns the cards still in the deck.
fn prepare(
    hands: &[&[Card]],
    board: &[Card],
    variant: Variant,
) -> Result<Vec<Card>, EquityError> {
    if hands.len() < 2 {
        return Err(EquityError::TooFewHands(hands.len()));
    }
    if board.len() > 5 {
        return Err(EquityError::BoardTooLong(board.len()));
    }

    let (lo, hi) = variant.hole_cards();
    let mut used = CardSet::empty();
    for (index, hand) in hands.iter().enumerate() {
        if !(lo..=hi).contains(&hand.len()) {
            return Err(EquityError::WrongHoleCount {
                hand: index,
                found: hand.len(),
                expected: (lo, hi),
            });
        }
        for &card in *hand {
            if !used.insert(card) {
                return Err(EquityError::DuplicateCard(card));
            }
        }
    }
    for &card in board {
        if !used.insert(card) {
            return Err(EquityError::DuplicateCard(card));
        }
    }

    let deck: Vec<Card> = CardSet::full_deck().difference(used).iter().collect();
    let needed = 5 - board.len();
    if deck.len() < needed {
        return Err(EquityError::DeckExhausted {
            needed,
            available: deck.len(),
        });
    }
    Ok(deck)
}

/// Ranks one hand on a complete board.
#[inline]
fn rank_hand(variant: Variant, hole: &[Card], board: &[Card], scratch: &mut Vec<Card>) -> HandRank {
    match variant {
        Variant::Holdem => {
            scratch.clear();
            scratch.extend_from_slice(hole);
            scratch.extend_from_slice(board);
            evaluate(scratch)
        }
        Variant::Omaha => evaluate_omaha(hole, board),
    }
}

/// Running totals while trials accumulate.
#[derive(Clone)]
struct Tally {
    wins: u64,
    ties: u64,
    share: f64,
}

/// Scores one completed board and folds the result into `tallies`.
fn score(
    hands: &[&[Card]],
    board: &[Card],
    variant: Variant,
    ranks: &mut Vec<HandRank>,
    scratch: &mut Vec<Card>,
    tallies: &mut [Tally],
) {
    ranks.clear();
    for hand in hands {
        ranks.push(rank_hand(variant, hand, board, scratch));
    }

    let best = *ranks.iter().max().expect("at least two hands");
    let winners = ranks.iter().filter(|r| **r == best).count();
    let split = 1.0 / winners as f64;

    for (index, rank) in ranks.iter().enumerate() {
        if *rank != best {
            continue;
        }
        if winners == 1 {
            tallies[index].wins += 1;
        } else {
            tallies[index].ties += 1;
        }
        tallies[index].share += split;
    }
}

fn finish(tallies: Vec<Tally>, trials: u64) -> Vec<Equity> {
    tallies
        .into_iter()
        .map(|t| Equity {
            wins: t.wins,
            ties: t.ties,
            trials,
            share: if trials == 0 { 0.0 } else { t.share / trials as f64 },
        })
        .collect()
}

fn fresh_tallies(n: usize) -> Vec<Tally> {
    vec![
        Tally {
            wins: 0,
            ties: 0,
            share: 0.0
        };
        n
    ]
}

/// Exact equity by enumerating every remaining board runout.
///
/// This is ground truth. Cost grows as `C(remaining, missing)`: trivial from
/// the flop, about 1.7M runouts for a two-player Hold'em preflop, and heavier
/// in Omaha where every runout costs 60 evaluations.
pub fn exact(
    hands: &[&[Card]],
    board: &[Card],
    variant: Variant,
) -> Result<Vec<Equity>, EquityError> {
    let deck = prepare(hands, board, variant)?;
    let missing = 5 - board.len();

    let mut tallies = fresh_tallies(hands.len());
    let mut trials = 0u64;
    let mut full_board = Vec::with_capacity(5);
    let mut ranks = Vec::with_capacity(hands.len());
    let mut scratch = Vec::with_capacity(7);
    let mut chosen = Vec::with_capacity(missing);

    enumerate(&deck, missing, 0, &mut chosen, &mut |runout| {
        full_board.clear();
        full_board.extend_from_slice(board);
        full_board.extend_from_slice(runout);
        score(
            hands,
            &full_board,
            variant,
            &mut ranks,
            &mut scratch,
            &mut tallies,
        );
        trials += 1;
    });

    Ok(finish(tallies, trials))
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

/// Estimated equity from `trials` random runouts.
///
/// Standard error falls as `1/sqrt(trials)`, so roughly 100k trials buys about
/// a tenth of a percent. Seed the [`Rng`] to make a result reproducible.
pub fn monte_carlo(
    hands: &[&[Card]],
    board: &[Card],
    variant: Variant,
    trials: u64,
    rng: &mut Rng,
) -> Result<Vec<Equity>, EquityError> {
    let mut deck = prepare(hands, board, variant)?;
    let missing = 5 - board.len();

    let mut tallies = fresh_tallies(hands.len());
    let mut full_board = Vec::with_capacity(5);
    let mut ranks = Vec::with_capacity(hands.len());
    let mut scratch = Vec::with_capacity(7);

    for _ in 0..trials {
        // Partial Fisher-Yates: draw `missing` distinct cards by swapping them
        // to the front. Cheaper than reshuffling the whole deck each trial.
        for slot in 0..missing {
            let pick = slot + rng.below((deck.len() - slot) as u64) as usize;
            deck.swap(slot, pick);
        }
        full_board.clear();
        full_board.extend_from_slice(board);
        full_board.extend_from_slice(&deck[..missing]);
        score(
            hands,
            &full_board,
            variant,
            &mut ranks,
            &mut scratch,
            &mut tallies,
        );
    }

    Ok(finish(tallies, trials))
}

/// Runs [`exact`] when the enumeration is small enough, otherwise
/// [`monte_carlo`].
///
/// `max_exact` caps how many runouts are worth enumerating. Exact results are
/// always preferred when affordable, since sampling error is indistinguishable
/// from a strategy bug when a solver is being debugged.
pub fn estimate(
    hands: &[&[Card]],
    board: &[Card],
    variant: Variant,
    max_exact: u64,
    trials: u64,
    rng: &mut Rng,
) -> Result<Vec<Equity>, EquityError> {
    let deck = prepare(hands, board, variant)?;
    let combinations = binomial(deck.len() as u64, (5 - board.len()) as u64);
    if combinations <= max_exact {
        exact(hands, board, variant)
    } else {
        monte_carlo(hands, board, variant, trials, rng)
    }
}

/// `n choose k`, saturating rather than overflowing.
fn binomial(n: u64, k: u64) -> u64 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut result: u64 = 1;
    for step in 0..k {
        result = match result
            .checked_mul(n - step)
            .map(|value| value / (step + 1))
        {
            Some(value) => value,
            None => return u64::MAX,
        };
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::parse_cards;

    fn cards(s: &str) -> Vec<Card> {
        parse_cards(s).expect("valid cards")
    }

    fn holdem(hands: &[&str], board: &str) -> Vec<Equity> {
        let parsed: Vec<Vec<Card>> = hands.iter().map(|h| cards(h)).collect();
        let refs: Vec<&[Card]> = parsed.iter().map(|h| h.as_slice()).collect();
        exact(&refs, &cards(board), Variant::Holdem).expect("valid input")
    }

    fn total_share(equities: &[Equity]) -> f64 {
        equities.iter().map(|e| e.share).sum()
    }

    #[test]
    fn equities_always_sum_to_one() {
        // Whatever the spot, the pot is fully distributed.
        for (hands, board) in [
            (vec!["AsAd", "KsKd"], ""),
            (vec!["AsKs", "7d7c"], "2h 9s Td"),
            (vec!["AsAd", "KsKd", "QhQc"], "2h 9s Td Jc"),
            (vec!["7c2d", "AsKs", "JhTh", "5c5d"], "2h 9s Td"),
        ] {
            let equities = holdem(&hands, board);
            let total = total_share(&equities);
            assert!(
                (total - 1.0).abs() < 1e-9,
                "shares summed to {total} for {hands:?} on {board:?}"
            );
        }
    }

    #[test]
    fn mirror_image_hands_split_exactly_evenly() {
        // Swapping spades and hearts maps one hand onto the other and leaves
        // the remaining deck unchanged, so this must be exactly 50/50 — no
        // tolerance needed.
        let equities = holdem(&["AsKs", "AhKh"], "");
        assert!((equities[0].share - 0.5).abs() < 1e-12);
        assert!((equities[1].share - 0.5).abs() < 1e-12);
    }

    #[test]
    fn a_completed_board_is_decided_with_no_uncertainty() {
        let equities = holdem(&["AsAd", "KsKd"], "2c 7d 9h Jc 4s");
        assert_eq!(equities[0].share, 1.0);
        assert_eq!(equities[1].share, 0.0);
        assert_eq!(equities[0].trials, 1, "nothing left to enumerate");
        assert_eq!(equities[0].wins, 1);
    }

    #[test]
    fn a_board_that_plays_itself_is_a_chop() {
        // Both players play the royal flush on the board.
        let equities = holdem(&["2c3d", "4h5s"], "As Ks Qs Js Ts");
        assert_eq!(equities[0].share, 0.5);
        assert_eq!(equities[1].share, 0.5);
        assert_eq!(equities[0].ties, 1);
        assert_eq!(equities[0].wins, 0, "a chop is not a win");
    }

    #[test]
    fn aces_beat_kings_about_four_fifths_of_the_time() {
        let equities = holdem(&["AsAd", "KsKd"], "");
        assert!(
            (0.79..0.85).contains(&equities[0].share),
            "AA vs KK was {:.4}",
            equities[0].share
        );
    }

    #[test]
    fn a_pair_against_two_overcards_is_close_to_a_coin_flip() {
        let equities = holdem(&["QsQd", "AhKh"], "");
        assert!(
            (0.50..0.60).contains(&equities[0].share),
            "QQ vs AKs was {:.4}",
            equities[0].share
        );
    }

    #[test]
    fn a_dominated_hand_is_in_bad_shape() {
        // AK vs AQ: the queen is nearly drawing to a three-outer.
        let equities = holdem(&["AsKd", "AhQd"], "");
        assert!(
            equities[0].share > 0.70,
            "AK vs AQ was {:.4}",
            equities[0].share
        );
    }

    #[test]
    fn monte_carlo_agrees_with_exact_enumeration() {
        // The real test of the sampler: same spot, both engines, same answer.
        let hands = [cards("AsKs"), cards("7d7c")];
        let refs: Vec<&[Card]> = hands.iter().map(|h| h.as_slice()).collect();
        let board = cards("2h 9s Td");

        let exact_result = exact(&refs, &board, Variant::Holdem).expect("valid");
        let mut rng = Rng::new(0xC0FFEE);
        let sampled =
            monte_carlo(&refs, &board, Variant::Holdem, 200_000, &mut rng).expect("valid");

        for (index, (a, b)) in exact_result.iter().zip(sampled.iter()).enumerate() {
            assert!(
                (a.share - b.share).abs() < 0.005,
                "hand {index}: exact {:.4} vs sampled {:.4}",
                a.share,
                b.share
            );
        }
        assert!((total_share(&sampled) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn monte_carlo_is_reproducible_from_its_seed() {
        let hands = [cards("AsKs"), cards("7d7c")];
        let refs: Vec<&[Card]> = hands.iter().map(|h| h.as_slice()).collect();

        let run = || {
            let mut rng = Rng::new(42);
            monte_carlo(&refs, &[], Variant::Holdem, 5_000, &mut rng).expect("valid")
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn omaha_uses_omaha_rules() {
        let hands = [cards("AsAdKsKd"), cards("2c3d4h5s")];
        let refs: Vec<&[Card]> = hands.iter().map(|h| h.as_slice()).collect();
        // Four spades on the board, but Omaha needs two from hand.
        let board = cards("Qs Js 9s 2s 7h");
        let equities = exact(&refs, &board, Variant::Omaha).expect("valid");

        assert!((total_share(&equities) - 1.0).abs() < 1e-9);
        // AsKs plays two spades with three from the board for the nut flush.
        assert_eq!(equities[0].share, 1.0);
    }

    #[test]
    fn omaha_preflop_equity_is_computable_by_sampling() {
        let hands = [cards("AsAdKsKd"), cards("7c6d5h4s")];
        let refs: Vec<&[Card]> = hands.iter().map(|h| h.as_slice()).collect();
        let mut rng = Rng::new(11);
        let equities = monte_carlo(&refs, &[], Variant::Omaha, 20_000, &mut rng).expect("valid");

        assert!((total_share(&equities) - 1.0).abs() < 1e-9);
        assert!(
            equities[0].share > 0.5,
            "AAKK double-suited should be ahead, got {:.4}",
            equities[0].share
        );
    }

    #[test]
    fn estimate_enumerates_when_cheap_and_samples_when_not() {
        let hands = [cards("AsKs"), cards("7d7c")];
        let refs: Vec<&[Card]> = hands.iter().map(|h| h.as_slice()).collect();
        let mut rng = Rng::new(5);

        // On the flop there are 990 runouts, well under the cap.
        let flop = estimate(&refs, &cards("2h 9s Td"), Variant::Holdem, 10_000, 1_000, &mut rng)
            .expect("valid");
        assert_eq!(flop[0].trials, 990, "should have enumerated");

        // Preflop is 1.7M runouts, so it samples instead.
        let preflop =
            estimate(&refs, &[], Variant::Holdem, 10_000, 1_000, &mut rng).expect("valid");
        assert_eq!(preflop[0].trials, 1_000, "should have sampled");
    }

    #[test]
    fn binomial_matches_known_values() {
        assert_eq!(binomial(52, 5), 2_598_960);
        assert_eq!(binomial(48, 5), 1_712_304);
        assert_eq!(binomial(45, 2), 990);
        assert_eq!(binomial(5, 0), 1);
        assert_eq!(binomial(3, 5), 0);
    }

    #[test]
    fn duplicate_cards_are_rejected() {
        let hands = [cards("AsKs"), cards("AsQd")];
        let refs: Vec<&[Card]> = hands.iter().map(|h| h.as_slice()).collect();
        let error = exact(&refs, &[], Variant::Holdem).expect_err("duplicate ace");
        assert!(matches!(error, EquityError::DuplicateCard(_)));

        // Also across hand and board.
        let hands = [cards("AsKs"), cards("7d7c")];
        let refs: Vec<&[Card]> = hands.iter().map(|h| h.as_slice()).collect();
        let error = exact(&refs, &cards("As 2h 3d"), Variant::Holdem).expect_err("board clash");
        assert!(matches!(error, EquityError::DuplicateCard(_)));
    }

    #[test]
    fn malformed_inputs_are_rejected() {
        let three = cards("AsKsQs");
        let two = cards("7d7c");
        let refs: Vec<&[Card]> = vec![three.as_slice(), two.as_slice()];
        assert!(matches!(
            exact(&refs, &[], Variant::Holdem),
            Err(EquityError::WrongHoleCount { hand: 0, found: 3, .. })
        ));

        let one = [two.as_slice()];
        assert!(matches!(
            exact(&one, &[], Variant::Holdem),
            Err(EquityError::TooFewHands(1))
        ));

        let a = cards("AsKs");
        let b = cards("7d7c");
        let refs: Vec<&[Card]> = vec![a.as_slice(), b.as_slice()];
        assert!(matches!(
            exact(&refs, &cards("2h 3h 4h 5h 6h 7h"), Variant::Holdem),
            Err(EquityError::BoardTooLong(6))
        ));
    }

    #[test]
    fn equity_display_is_readable() {
        let equity = Equity {
            wins: 80,
            ties: 5,
            trials: 100,
            share: 0.825,
        };
        assert_eq!(equity.to_string(), "82.50% (80 win, 5 tie, 100 trials)");
        assert!((equity.percent() - 82.5).abs() < 1e-9);
    }
}
