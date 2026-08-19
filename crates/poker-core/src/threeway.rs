//! Three-way equity for every triple of preflop hand classes.
//!
//! A three-handed solve needs this and cannot borrow the pairwise table
//! instead. [`crate::multiway::approximate_shares`] records why: multiplying
//! pairwise equities misorders real spots badly, handing one hand a twentieth
//! of a pot it in truth wins a fifth of. Three-way equity is not a function of
//! the pairwise numbers, so it has to be measured in its own right.
//!
//! # Only sorted triples are stored
//!
//! Equity does not care which seat holds which hand, so `(A, B, C)` and
//! `(B, A, C)` are one measurement with the shares permuted. Keeping only
//! non-decreasing triples turns 169³ entries into 818,805 — a sixth of the work
//! and a sixth of the memory, for nothing given up.
//!
//! # Sampled, not enumerated
//!
//! Enumerating every runout for every triple is not affordable, and it is not
//! needed either: these numbers feed a solver whose action abstraction is far
//! coarser than a tenth of a percent of equity. What does matter is that the
//! table is the *same* table every time, so a solve can be reproduced, and that
//! is why every triple is seeded from its own classes rather than from whichever
//! worker happened to pick it up.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::abstraction::{HandClass, NUM_HAND_CLASSES};
use crate::card::{Card, CardSet};
use crate::eval::evaluate;
use crate::pushfold::sample_hand;
use crate::rng::Rng;

/// How many non-decreasing triples of hand classes there are.
pub const NUM_TRIPLES: usize =
    NUM_HAND_CLASSES * (NUM_HAND_CLASSES + 1) * (NUM_HAND_CLASSES + 2) / 6;

const MAGIC: &[u8; 8] = b"PKEQ3\0\0\0";
const VERSION: u32 = 1;

/// Every triple's three-way showdown shares.
#[derive(Debug, Clone, PartialEq)]
pub struct ThreeWayEquity {
    /// One entry per sorted triple, holding the shares in the order the sorted
    /// classes appear.
    shares: Vec<[f32; 3]>,
}

/// Where a sorted triple lives in the table.
fn triple_index(a: usize, b: usize, c: usize) -> usize {
    debug_assert!(a <= b && b <= c && c < NUM_HAND_CLASSES);
    let n = NUM_HAND_CLASSES;
    // Triples starting below `a`, then the pairs before `b` within `a`, then
    // how far along `c` sits.
    let before_a: usize = (0..a).map(|i| (n - i) * (n - i + 1) / 2).sum();
    let before_b: usize = (a..b).map(|j| n - j).sum();
    before_a + before_b + (c - b)
}

/// Every sorted triple, in table order.
fn all_triples() -> Vec<(usize, usize, usize)> {
    (0..NUM_HAND_CLASSES)
        .flat_map(|a| {
            (a..NUM_HAND_CLASSES).flat_map(move |b| (b..NUM_HAND_CLASSES).map(move |c| (a, b, c)))
        })
        .collect()
}

impl ThreeWayEquity {
    /// Builds the table by sampling `samples` runouts per triple.
    pub fn sampled_parallel(samples: u32, seed: u64, threads: usize) -> ThreeWayEquity {
        let triples = all_triples();
        debug_assert_eq!(triples.len(), NUM_TRIPLES);

        let threads = threads.max(1);
        let chunk = triples.len().div_ceil(threads);
        let computed: Vec<Vec<(usize, [f32; 3])>> = std::thread::scope(|scope| {
            let workers: Vec<_> = triples
                .chunks(chunk)
                .map(|slice| {
                    scope.spawn(move || {
                        let mut deck = Vec::with_capacity(46);
                        slice
                            .iter()
                            .map(|&(a, b, c)| {
                                let mut rng = Rng::new(triple_seed(seed, a, b, c));
                                let shares = sample_triple(a, b, c, samples, &mut rng, &mut deck);
                                (triple_index(a, b, c), shares)
                            })
                            .collect()
                    })
                })
                .collect();
            workers
                .into_iter()
                .map(|worker| worker.join().expect("equity worker panicked"))
                .collect()
        });

        let mut shares = vec![[1.0 / 3.0; 3]; NUM_TRIPLES];
        for batch in computed {
            for (index, value) in batch {
                shares[index] = value;
            }
        }
        ThreeWayEquity { shares }
    }

    /// The three players' shares of a pot they all reach showdown in.
    ///
    /// Shares come back in the order the classes were given, whatever order
    /// that is.
    pub fn get(&self, a: HandClass, b: HandClass, c: HandClass) -> [f64; 3] {
        // Sort the three, remembering where each came from, look up the sorted
        // triple, then put the shares back where they belong.
        let mut order = [(a.index(), 0usize), (b.index(), 1), (c.index(), 2)];
        order.sort_unstable();
        let stored = self.shares[triple_index(order[0].0, order[1].0, order[2].0)];
        let mut out = [0.0; 3];
        for (slot, (_, origin)) in order.iter().enumerate() {
            out[*origin] = stored[slot] as f64;
        }
        out
    }

    pub fn len(&self) -> usize {
        self.shares.len()
    }

    /// Measures one triple on its own, without building a table.
    ///
    /// Useful for checking a single spot, and for tests: building the whole
    /// table costs over a minute, which is not a price a unit test should pay
    /// to ask about three hands.
    pub fn measure(a: HandClass, b: HandClass, c: HandClass, samples: u32, seed: u64) -> [f64; 3] {
        let mut order = [(a.index(), 0usize), (b.index(), 1), (c.index(), 2)];
        order.sort_unstable();
        let (x, y, z) = (order[0].0, order[1].0, order[2].0);
        let mut rng = Rng::new(triple_seed(seed, x, y, z));
        let mut deck = Vec::with_capacity(46);
        let stored = sample_triple(x, y, z, samples, &mut rng, &mut deck);
        let mut out = [0.0; 3];
        for (slot, (_, origin)) in order.iter().enumerate() {
            out[*origin] = stored[slot] as f64;
        }
        out
    }

    pub fn is_empty(&self) -> bool {
        self.shares.is_empty()
    }

    /// Writes the table to `path`, creating parent directories as needed.
    ///
    /// Building it costs minutes; reading it costs milliseconds. A solve that
    /// rebuilt it every run would spend most of its time re-measuring numbers
    /// that never change.
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = BufWriter::new(File::create(path)?);
        file.write_all(MAGIC)?;
        file.write_all(&VERSION.to_le_bytes())?;
        file.write_all(&(self.shares.len() as u64).to_le_bytes())?;
        for triple in &self.shares {
            for share in triple {
                file.write_all(&share.to_le_bytes())?;
            }
        }
        file.flush()
    }

    /// Reads a table written by [`ThreeWayEquity::save`].
    pub fn load(path: impl AsRef<Path>) -> io::Result<ThreeWayEquity> {
        let mut file = BufReader::new(File::open(path)?);
        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(invalid("not a three-way equity table"));
        }
        let mut word = [0u8; 4];
        file.read_exact(&mut word)?;
        if u32::from_le_bytes(word) != VERSION {
            return Err(invalid("unsupported three-way equity version"));
        }
        let mut count = [0u8; 8];
        file.read_exact(&mut count)?;
        let count = u64::from_le_bytes(count) as usize;
        if count != NUM_TRIPLES {
            return Err(invalid(format!(
                "table holds {count} triples, but this build expects {NUM_TRIPLES}"
            )));
        }

        let mut shares = vec![[0.0f32; 3]; count];
        let mut bytes = vec![0u8; count * 12];
        file.read_exact(&mut bytes)?;
        for (index, triple) in shares.iter_mut().enumerate() {
            for (slot, share) in triple.iter_mut().enumerate() {
                let at = index * 12 + slot * 4;
                *share = f32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"));
            }
        }
        Ok(ThreeWayEquity { shares })
    }
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

/// A seed depending only on the triple, so the table is reproducible.
fn triple_seed(seed: u64, a: usize, b: usize, c: usize) -> u64 {
    seed ^ ((a as u64) << 40) ^ ((b as u64) << 20) ^ (c as u64)
}

/// Samples one triple's shares by dealing those classes and running boards out.
///
/// A draw that cannot be dealt — three classes wanting the same card between
/// them — is skipped rather than forced. If none can be dealt at all the triple
/// keeps an even split, which is the honest answer for a spot that cannot arise.
fn sample_triple(
    a: usize,
    b: usize,
    c: usize,
    samples: u32,
    rng: &mut Rng,
    deck: &mut Vec<Card>,
) -> [f32; 3] {
    let classes = [
        HandClass::from_index(a).expect("in range"),
        HandClass::from_index(b).expect("in range"),
        HandClass::from_index(c).expect("in range"),
    ];
    let first = Card::all().next().expect("non-empty deck");
    let mut totals = [0.0f64; 3];
    let mut trials = 0u32;
    let mut seven = [first; 7];

    'sample: for _ in 0..samples {
        let mut dead = CardSet::empty();
        let mut hands = [[first; 2]; 3];
        for (slot, class) in classes.iter().enumerate() {
            match sample_hand(*class, rng, dead) {
                Some(hand) => {
                    dead.insert(hand[0]);
                    dead.insert(hand[1]);
                    hands[slot] = hand;
                }
                None => continue 'sample,
            }
        }

        deck.clear();
        deck.extend(CardSet::full_deck().difference(dead).iter());
        for slot in 0..5 {
            let pick = slot + rng.below((deck.len() - slot) as u64) as usize;
            deck.swap(slot, pick);
        }
        seven[2..].copy_from_slice(&deck[..5]);

        let mut best = None;
        let mut winners = 0u32;
        let mut won = [false; 3];
        for (slot, hand) in hands.iter().enumerate() {
            seven[..2].copy_from_slice(hand);
            let rank = evaluate(&seven);
            match best {
                Some(current) if rank > current => {
                    best = Some(rank);
                    won = [false; 3];
                    won[slot] = true;
                    winners = 1;
                }
                Some(current) if rank == current => {
                    won[slot] = true;
                    winners += 1;
                }
                Some(_) => {}
                None => {
                    best = Some(rank);
                    won[slot] = true;
                    winners = 1;
                }
            }
        }
        // A chop is split among those who tie, so the three shares of any one
        // runout always add to exactly one.
        for slot in 0..3 {
            if won[slot] {
                totals[slot] += 1.0 / winners as f64;
            }
        }
        trials += 1;
    }

    if trials == 0 {
        return [1.0 / 3.0; 3];
    }
    [
        (totals[0] / trials as f64) as f32,
        (totals[1] / trials as f64) as f32,
        (totals[2] / trials as f64) as f32,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Enough runouts for the orderings below to be far outside noise.
    const SAMPLES: u32 = 4_000;

    /// Reads shorthand such as `AKs`, `72o` or `AA`.
    fn class(text: &str) -> HandClass {
        use crate::card::Rank;
        let mut chars = text.chars();
        let high = Rank::from_char(chars.next().expect("a rank")).expect("a rank");
        let low = Rank::from_char(chars.next().expect("a rank")).expect("a rank");
        HandClass::new(high, low, chars.next() == Some('s'))
    }

    fn measure(a: &str, b: &str, c: &str) -> [f64; 3] {
        ThreeWayEquity::measure(class(a), class(b), class(c), SAMPLES, 0x3EED)
    }

    #[test]
    fn every_sorted_triple_has_its_own_slot() {
        let mut seen = vec![false; NUM_TRIPLES];
        for (a, b, c) in all_triples() {
            let index = triple_index(a, b, c);
            assert!(!seen[index], "({a},{b},{c}) collided at {index}");
            seen[index] = true;
        }
        assert!(seen.into_iter().all(|s| s), "some slot was never filled");
    }

    #[test]
    fn the_table_is_the_size_the_arithmetic_says() {
        assert_eq!(all_triples().len(), NUM_TRIPLES);
        assert_eq!(NUM_TRIPLES, 818_805);
    }

    #[test]
    fn three_shares_of_one_pot_add_up_to_one_pot() {
        for triple in [("AA", "KK", "72o"), ("AKs", "QQ", "JTs"), ("22", "22", "22")] {
            let shares = measure(triple.0, triple.1, triple.2);
            let total: f64 = shares.iter().sum();
            assert!(
                (total - 1.0).abs() < 1e-6,
                "{triple:?} shares {shares:?} sum to {total}"
            );
        }
    }

    #[test]
    fn a_better_hand_takes_a_bigger_share() {
        let shares = measure("AA", "KK", "72o");
        assert!(shares[0] > shares[1], "AA should beat KK: {shares:?}");
        assert!(shares[1] > shares[2], "KK should beat 72o: {shares:?}");
        assert!(shares[0] > 0.5, "AA is a favourite three-handed: {shares:?}");
    }

    /// The reason this table exists rather than a product of pairwise numbers.
    ///
    /// Rather than compare against a remembered constant, this compares the two
    /// methods directly on the same spot. The pairwise product multiplies 72o's
    /// two small chances together and arrives at a share near zero; measured
    /// three-handed, the hand keeps several times that. Which is the whole
    /// point: three-way equity is not a function of the pairwise numbers.
    #[test]
    fn measuring_beats_multiplying_pairwise_equities() {
        let measured = measure("AA", "KK", "72o");

        let pairwise = vec![
            vec![0.5, 0.82, 0.88],
            vec![0.18, 0.5, 0.86],
            vec![0.12, 0.14, 0.5],
        ];
        let approximated = crate::multiway::approximate_shares(&pairwise);

        assert!(
            measured[2] > approximated[2] * 3.0,
            "measured {:.3} against approximated {:.3}",
            measured[2],
            approximated[2]
        );
        assert!(
            measured[2] > 0.05,
            "72o holds a real stake three-handed, got {:.3}",
            measured[2]
        );
    }

    #[test]
    fn shares_follow_the_hands_however_they_are_ordered() {
        // The same three hands in a different order must give the same three
        // numbers, moved to match. Getting this wrong would quietly hand one
        // seat another seat's equity.
        let straight = measure("AA", "72o", "KK");
        let shuffled = measure("KK", "AA", "72o");
        assert!((straight[0] - shuffled[1]).abs() < 1e-9);
        assert!((straight[1] - shuffled[2]).abs() < 1e-9);
        assert!((straight[2] - shuffled[0]).abs() < 1e-9);
    }

    /// Building the whole table is a minutes-long job, so it is not run by
    /// default; the pieces it is made of are covered above.
    #[test]
    #[ignore = "slow; run with --ignored"]
    fn a_full_table_covers_every_triple_and_stays_normalised() {
        let table = ThreeWayEquity::sampled_parallel(64, 0x3EED, 8);
        assert_eq!(table.len(), NUM_TRIPLES);
        for triple in [("AA", "KK", "72o"), ("72o", "72o", "AA")] {
            let shares = table.get(class(triple.0), class(triple.1), class(triple.2));
            let total: f64 = shares.iter().sum();
            assert!((total - 1.0).abs() < 1e-5, "{triple:?} sums to {total}");
        }
    }

    /// Checks the sampler against enumeration, which is the only way to know
    /// the sampling is measuring what it claims to.
    ///
    /// [`crate::multiway::exact_shares`] walks every runout, so it is the truth
    /// for one specific set of cards. The table works in hand *classes* and so
    /// averages over the suit combinations within each, which moves the answer
    /// slightly — hence a tolerance rather than equality. What it must not do
    /// is disagree by more than that.
    #[test]
    fn the_sampler_agrees_with_enumerating_every_runout() {
        use crate::card::parse_cards;

        let hands = [
            parse_cards("AsAh").expect("cards"),
            parse_cards("KsKh").expect("cards"),
            parse_cards("7c2d").expect("cards"),
        ];
        let hands: Vec<[Card; 2]> = hands
            .iter()
            .map(|cards| [cards[0], cards[1]])
            .collect();
        let truth = crate::multiway::exact_shares(&hands, &[]);
        let sampled = measure("AA", "KK", "72o");

        for (index, (exact, drawn)) in truth.iter().zip(sampled.iter()).enumerate() {
            assert!(
                (exact - drawn).abs() < 0.03,
                "player {index}: enumerated {exact:.4}, sampled {drawn:.4}"
            );
        }
    }

    #[test]
    fn a_table_survives_a_trip_through_a_file() {
        // A single sample per triple: this checks the file format, not the
        // numbers, and the numbers cost a minute.
        let table = ThreeWayEquity::sampled_parallel(1, 0x3EED, 8);
        let path = std::env::temp_dir().join("poker-threeway-roundtrip.bin");
        table.save(&path).expect("save");
        let read = ThreeWayEquity::load(&path).expect("load");
        assert_eq!(table, read);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_file_that_is_not_an_equity_table_is_rejected() {
        let path = std::env::temp_dir().join("poker-threeway-bad.bin");
        std::fs::write(&path, b"nonsense").expect("write");
        assert!(ThreeWayEquity::load(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }
}
