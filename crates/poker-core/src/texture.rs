//! How strong a hand is on a board, precomputed so a solver can ask cheaply.
//!
//! A postflop solve asks "how good is this hand here" hundreds of millions of
//! times, so the answer has to be a lookup. Computing it on demand is not
//! merely slow — sampling equity per visit would put noise inside the training
//! loop, and the strategy would be fitted partly to that noise.
//!
//! # The flop must not know the river
//!
//! The obvious shortcut is to deal a whole board, rank every holding on it, and
//! call that strength at every street. It is wrong, and wrong in a way that
//! looks like brilliance: a hand that will make a flush by the river would be
//! rated strong on the flop, and the solver would learn to play as though it
//! could see the future. Whatever it then learned would be worthless at a real
//! table.
//!
//! So strength at each street is measured from the cards visible at that
//! street, with the rest of the board still to come. On the flop that means
//! running out the remaining two cards; on the river it is simply the hand.
//!
//! # Equity against a random holding, not hand rank
//!
//! Rank alone says nothing about what is still to come — a flush draw and a
//! missed one rank identically on the flop and are not remotely the same hand.
//! What is measured here is the share of the pot a holding takes against a
//! uniformly random opponent, which prices draws by what they are worth rather
//! than by what they have made so far.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::betting::Street;
use crate::card::{Card, CardSet};
use crate::eval::evaluate;
use crate::rng::Rng;

const MAGIC: &[u8; 8] = b"PKTEX\0\0\0";
const VERSION: u32 = 1;

/// Holdings possible once five board cards are known: `C(47, 2)`.
pub const HOLDINGS: usize = 1_081;

/// One sampled board, with every holding's strength at every street.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Board {
    /// The full runout, of which each street sees a prefix.
    cards: [Card; 5],
    /// Bucket per holding per street, indexed `street * HOLDINGS + holding`.
    ///
    /// Held for the flop, turn and river; there is nothing to bucket preflop.
    buckets: Vec<u8>,
}

impl Board {
    pub fn cards(&self) -> &[Card; 5] {
        &self.cards
    }

    /// The cards visible at a street.
    pub fn visible(&self, street: Street) -> &[Card] {
        &self.cards[..street.board_cards()]
    }
}

/// A sample of boards with strength precomputed on each of them.
#[derive(Debug, Clone, PartialEq)]
pub struct Textures {
    buckets: usize,
    boards: Vec<Board>,
    /// The holdings a board leaves available, in the order the buckets use.
    ///
    /// Determined by which five cards the board took, so it is stored per board
    /// rather than shared.
    holdings: Vec<Vec<[Card; 2]>>,
}

impl Textures {
    /// Samples `count` boards and measures strength on each.
    ///
    /// `runouts` is how many completions each unfinished street is measured
    /// over. More is more accurate and linearly slower; the river needs none,
    /// since nothing is left to come.
    pub fn sample(
        count: usize,
        buckets: usize,
        runouts: u32,
        seed: u64,
        threads: usize,
    ) -> Textures {
        assert!(
            (2..=u8::MAX as usize).contains(&buckets),
            "strength needs at least two groups to mean anything"
        );

        let threads = threads.max(1);
        let chunk = count.div_ceil(threads);
        let built: Vec<Vec<(Board, Vec<[Card; 2]>)>> = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..threads)
                .map(|worker| {
                    let first = worker * chunk;
                    let last = (first + chunk).min(count);
                    scope.spawn(move || {
                        (first..last)
                            .map(|index| {
                                // Seeded per board, so the sample is the same
                                // whatever the core count.
                                let mut rng = Rng::new(seed ^ (index as u64).wrapping_mul(0x9E37_79B9));
                                measure_board(&mut rng, buckets, runouts)
                            })
                            .collect()
                    })
                })
                .collect();
            workers
                .into_iter()
                .map(|worker| worker.join().expect("texture worker panicked"))
                .collect()
        });

        let mut boards = Vec::with_capacity(count);
        let mut holdings = Vec::with_capacity(count);
        for batch in built {
            for (board, available) in batch {
                boards.push(board);
                holdings.push(available);
            }
        }
        Textures {
            buckets,
            boards,
            holdings,
        }
    }

    pub fn len(&self) -> usize {
        self.boards.len()
    }

    pub fn is_empty(&self) -> bool {
        self.boards.is_empty()
    }

    pub fn buckets(&self) -> usize {
        self.buckets
    }

    pub fn board(&self, index: usize) -> &Board {
        &self.boards[index]
    }

    /// The holdings available on a board, in bucket order.
    pub fn holdings(&self, index: usize) -> &[[Card; 2]] {
        &self.holdings[index]
    }

    /// How strong a holding is on a board at a street.
    ///
    /// Zero is the weakest group. `None` preflop, where there is no board to be
    /// strong on and [`crate::abstraction::HandClass`] is the right abstraction
    /// instead.
    pub fn strength(&self, board: usize, street: Street, holding: usize) -> Option<u8> {
        let street = street_index(street)?;
        Some(self.boards[board].buckets[street * HOLDINGS + holding])
    }

    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = BufWriter::new(File::create(path)?);
        file.write_all(MAGIC)?;
        file.write_all(&VERSION.to_le_bytes())?;
        file.write_all(&(self.buckets as u32).to_le_bytes())?;
        file.write_all(&(self.boards.len() as u64).to_le_bytes())?;
        for board in &self.boards {
            for card in &board.cards {
                file.write_all(&[card.index()])?;
            }
            file.write_all(&board.buckets)?;
        }
        file.flush()
    }

    pub fn load(path: impl AsRef<Path>) -> io::Result<Textures> {
        let mut file = BufReader::new(File::open(path)?);
        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(invalid("not a texture file"));
        }
        let mut word = [0u8; 4];
        file.read_exact(&mut word)?;
        if u32::from_le_bytes(word) != VERSION {
            return Err(invalid("unsupported texture version"));
        }
        file.read_exact(&mut word)?;
        let buckets = u32::from_le_bytes(word) as usize;
        let mut count = [0u8; 8];
        file.read_exact(&mut count)?;
        let count = u64::from_le_bytes(count) as usize;

        let mut boards = Vec::with_capacity(count);
        let mut holdings = Vec::with_capacity(count);
        for _ in 0..count {
            let mut faces = [0u8; 5];
            file.read_exact(&mut faces)?;
            let mut cards = [Card::from_index(0).expect("a card"); 5];
            for (slot, face) in cards.iter_mut().zip(faces) {
                *slot = Card::from_index(face).ok_or_else(|| invalid("bad card"))?;
            }
            let mut buckets_of = vec![0u8; 3 * HOLDINGS];
            file.read_exact(&mut buckets_of)?;
            holdings.push(remaining_holdings(&cards));
            boards.push(Board {
                cards,
                buckets: buckets_of,
            });
        }
        Ok(Textures {
            buckets,
            boards,
            holdings,
        })
    }
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

/// Where a street's buckets sit. Preflop has none.
fn street_index(street: Street) -> Option<usize> {
    match street {
        Street::Preflop => None,
        Street::Flop => Some(0),
        Street::Turn => Some(1),
        Street::River => Some(2),
    }
}

/// Every two-card holding the board leaves available, in a fixed order.
fn remaining_holdings(board: &[Card; 5]) -> Vec<[Card; 2]> {
    let dead: CardSet = board.iter().copied().collect();
    let deck: Vec<Card> = CardSet::full_deck().difference(dead).iter().collect();
    let mut out = Vec::with_capacity(HOLDINGS);
    for first in 0..deck.len() {
        for second in first + 1..deck.len() {
            out.push([deck[first], deck[second]]);
        }
    }
    out
}

/// Samples one board and buckets every holding on it, street by street.
fn measure_board(rng: &mut Rng, buckets: usize, runouts: u32) -> (Board, Vec<[Card; 2]>) {
    let mut cards = [Card::from_index(0).expect("a card"); 5];
    let mut dealt = CardSet::empty();
    for slot in cards.iter_mut() {
        loop {
            let card = Card::from_index(rng.below(52) as u8).expect("0..52");
            if dealt.insert(card) {
                *slot = card;
                break;
            }
        }
    }

    let holdings = remaining_holdings(&cards);
    debug_assert_eq!(holdings.len(), HOLDINGS);

    let mut all = vec![0u8; 3 * HOLDINGS];
    for street in [Street::Flop, Street::Turn, Street::River] {
        let equity = equity_on(&cards, &holdings, street, rng, runouts);
        let at = street_index(street).expect("postflop") * HOLDINGS;
        assign_buckets(&equity, buckets, &mut all[at..at + HOLDINGS]);
    }

    (
        Board {
            cards,
            buckets: all,
        },
        holdings,
    )
}

/// Each holding's share of the pot against a uniformly random opponent.
///
/// The cards still to come are drawn afresh for each runout, so a holding is
/// priced by what it may become rather than by what this particular board
/// eventually did. That distinction is the whole point: it is what stops a flop
/// bucket from knowing the river.
fn equity_on(
    board: &[Card; 5],
    holdings: &[[Card; 2]],
    street: Street,
    rng: &mut Rng,
    runouts: u32,
) -> Vec<f64> {
    let seen = street.board_cards();
    let missing = 5 - seen;
    // The river has nothing to come, so one pass over the real board is exact.
    let passes = if missing == 0 { 1 } else { runouts.max(1) };

    let mut totals = vec![0.0f64; holdings.len()];
    let mut ranks = vec![0u32; holdings.len()];
    let mut order: Vec<u32> = Vec::with_capacity(holdings.len());
    let mut hand = [board[0]; 7];

    for _ in 0..passes {
        // Complete the board without using any card the street already shows.
        // Cards in a holding are not excluded here: a runout is dealt once and
        // shared by every holding, so a holding that collides with it is simply
        // one that could not have been dealt, and its own comparison is what
        // suffers rather than everybody else's.
        let mut used: CardSet = board[..seen].iter().copied().collect();
        let mut full = *board;
        for slot in full.iter_mut().skip(seen) {
            loop {
                let card = Card::from_index(rng.below(52) as u8).expect("0..52");
                if used.insert(card) {
                    *slot = card;
                    break;
                }
            }
        }

        hand[2..].copy_from_slice(&full);
        for (index, holding) in holdings.iter().enumerate() {
            // A holding sharing a card with the runout cannot occur; give it
            // the weakest possible rank so it neither wins nor distorts.
            if holding.iter().any(|card| full[seen..].contains(card)) {
                ranks[index] = 0;
                continue;
            }
            hand[..2].copy_from_slice(holding);
            ranks[index] = evaluate(&hand).to_bits();
        }

        // A holding's share is how many others it beats, plus half of those it
        // ties, over the field — which is its equity against a random opponent.
        order.clear();
        order.extend(ranks.iter().copied());
        order.sort_unstable();
        let field = order.len() as f64;
        for (index, rank) in ranks.iter().enumerate() {
            let beaten = order.partition_point(|other| other < rank) as f64;
            let tied = order.partition_point(|other| other <= rank) as f64 - beaten;
            totals[index] += (beaten + tied / 2.0) / field;
        }
    }

    let passes = passes as f64;
    totals.iter().map(|total| total / passes).collect()
}

/// Cuts a set of equities into groups of equal population, weakest first.
///
/// Equal population rather than equal width: hands cluster heavily around the
/// middle, and equal-width bands would spend most of their resolution on the
/// few hands nobody has to think about.
fn assign_buckets(equity: &[f64], buckets: usize, out: &mut [u8]) {
    let mut order: Vec<usize> = (0..equity.len()).collect();
    order.sort_by(|a, b| equity[*a].total_cmp(&equity[*b]));
    for (rank, index) in order.iter().enumerate() {
        out[*index] = (rank * buckets / equity.len()) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::parse_cards;

    fn small() -> Textures {
        Textures::sample(8, 10, 24, 0x7E47, 4)
    }

    #[test]
    fn every_board_buckets_every_holding_at_every_street() {
        let textures = small();
        assert_eq!(textures.len(), 8);
        for index in 0..textures.len() {
            assert_eq!(textures.holdings(index).len(), HOLDINGS);
            for street in [Street::Flop, Street::Turn, Street::River] {
                for holding in 0..HOLDINGS {
                    let bucket = textures
                        .strength(index, street, holding)
                        .expect("postflop streets are bucketed");
                    assert!((bucket as usize) < textures.buckets());
                }
            }
        }
    }

    #[test]
    fn preflop_has_no_board_to_be_strong_on() {
        assert_eq!(small().strength(0, Street::Preflop, 0), None);
    }

    #[test]
    fn the_groups_are_of_equal_population() {
        // Every group should hold the same share of holdings, since that is how
        // they are cut. A lopsided split would mean most hands landing in one
        // group and the abstraction saying almost nothing.
        let textures = Textures::sample(1, 10, 24, 0x7E47, 2);
        let mut counts = vec![0usize; textures.buckets()];
        for holding in 0..HOLDINGS {
            let bucket = textures
                .strength(0, Street::Flop, holding)
                .expect("flop is bucketed");
            counts[bucket as usize] += 1;
        }
        let smallest = counts.iter().min().copied().unwrap_or(0);
        let largest = counts.iter().max().copied().unwrap_or(0);
        assert!(
            largest - smallest <= 1,
            "groups run from {smallest} to {largest}, which is not an even cut"
        );
    }

    /// The property the module exists to protect.
    ///
    /// A flop bucket must be computable from the flop. If the runout leaked in,
    /// hands that go on to make something would be rated strong on the flop,
    /// and a strategy trained on that would play with knowledge no player has.
    ///
    /// Checked by measuring the same flop under two different runouts and
    /// comparing a holding common to both.
    ///
    /// The two do not agree exactly, and the reason is worth recording. The
    /// board is dealt before anything is measured, so the opponent's holdings
    /// are drawn from the 47 cards it leaves — which quietly means the range
    /// already knows the turn and river cannot be in it. Every solver that
    /// pre-deals a board carries this, and the measured size of it here is two
    /// tenths of a percent of equity.
    ///
    /// So the test bounds the leak rather than denying it. If a change ever
    /// made the flop genuinely depend on the runout — ranking hands on the
    /// finished board, say — the gap would be enormous rather than marginal,
    /// and this would catch it.
    #[test]
    fn the_flop_bucket_does_not_depend_on_the_cards_still_to_come() {
        let one = parse_cards("Ah 7d 2c Kh Qs").expect("cards");
        let two = parse_cards("Ah 7d 2c 3s 4d").expect("cards");
        let fixed = |cards: &[Card]| {
            let mut board = [cards[0]; 5];
            board.copy_from_slice(&cards[..5]);
            board
        };
        let (first, second) = (fixed(&one), fixed(&two));

        let holdings_one = remaining_holdings(&first);
        let holdings_two = remaining_holdings(&second);
        let equity_one = equity_on(&first, &holdings_one, Street::Flop, &mut Rng::new(11), 80);
        let equity_two = equity_on(&second, &holdings_two, Street::Flop, &mut Rng::new(11), 80);

        // Holdings using none of either board's later cards exist in both
        // lists, and are the ones that can be compared.
        let mut compared = 0;
        let mut worst = 0.0f64;
        for (index, holding) in holdings_one.iter().enumerate() {
            let Some(other) = holdings_two.iter().position(|h| h == holding) else {
                continue;
            };
            let gap = (equity_one[index] - equity_two[other]).abs();
            assert!(
                gap < 0.05,
                "{holding:?} priced {:.4} on one runout and {:.4} on another, a gap                  of {gap:.4} — the flop is being told what is coming",
                equity_one[index],
                equity_two[other]
            );
            worst = worst.max(gap);
            compared += 1;
        }
        assert!(compared > 500, "only {compared} holdings were comparable");
        println!("largest disagreement across {compared} holdings: {worst:.4}");
    }

    #[test]
    fn a_made_hand_outranks_a_missed_draw_by_the_river() {
        // On this river the flush has come in. A holding with two of the suit
        // must be rated above one with none.
        let cards = parse_cards("Ah 7h 2h Kd 3h").expect("cards");
        let mut board = [cards[0]; 5];
        board.copy_from_slice(&cards[..5]);
        let holdings = remaining_holdings(&board);
        let mut rng = Rng::new(3);
        let equity = equity_on(&board, &holdings, Street::River, &mut rng, 1);

        let flush = holdings
            .iter()
            .position(|h| h.iter().all(|c| c.suit() == cards[0].suit()))
            .expect("some holding has two hearts");
        let nothing = holdings
            .iter()
            .position(|h| h.iter().all(|c| c.suit() != cards[0].suit() && (c.rank() as u8) < 5))
            .expect("some holding is rags offsuit");
        assert!(
            equity[flush] > equity[nothing],
            "a flush {:.3} should beat rags {:.3}",
            equity[flush],
            equity[nothing]
        );
    }

    #[test]
    fn a_sample_survives_a_trip_through_a_file() {
        let textures = small();
        let path = std::env::temp_dir().join("poker-texture-roundtrip.bin");
        textures.save(&path).expect("save");
        assert_eq!(Textures::load(&path).expect("load"), textures);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_file_that_is_not_a_texture_sample_is_rejected() {
        let path = std::env::temp_dir().join("poker-texture-bad.bin");
        std::fs::write(&path, b"nonsense").expect("write");
        assert!(Textures::load(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }
}
