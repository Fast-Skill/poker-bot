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

use std::collections::HashMap;
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
    /// Each holding's finished hand on this board.
    ///
    /// Showdowns are settled with these rather than by comparing buckets. A
    /// bucket is a group of roughly equal strength, so comparing them would
    /// call two different hands a tie whenever they landed together — and at
    /// twenty groups that is one showdown in twenty decided wrongly.
    made: Vec<u32>,
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
    /// Strength is exact — every runout is dealt, not a sample of them — so
    /// the only randomness here is which boards get drawn.
    pub fn sample(count: usize, buckets: usize, seed: u64, threads: usize) -> Textures {
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
                                measure_board(&mut rng, buckets)
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

    /// Who wins a showdown between two holdings on a board.
    ///
    /// Positive when the first wins, negative when the second does, zero for a
    /// split. Settled from the finished hands, not from buckets.
    pub fn showdown(&self, board: usize, first: usize, second: usize) -> std::cmp::Ordering {
        let made = &self.boards[board].made;
        made[first].cmp(&made[second])
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
            for made in &board.made {
                file.write_all(&made.to_le_bytes())?;
            }
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
            let mut raw = vec![0u8; HOLDINGS * 4];
            file.read_exact(&mut raw)?;
            let made = raw
                .chunks_exact(4)
                .map(|word| u32::from_le_bytes(word.try_into().expect("four bytes")))
                .collect();
            holdings.push(remaining_holdings(&cards));
            boards.push(Board {
                cards,
                buckets: buckets_of,
                made,
            });
        }
        Ok(Textures {
            buckets,
            boards,
            holdings,
        })
    }
}

/// Reads strength off real boards, remembering the ones it has seen.
///
/// # Why the cache is not an optimisation
///
/// Ranking a holding means ranking every holding, because a percentile is a
/// position within a field. On the flop that is 1176 runouts against 1176
/// holdings — a million and a half hand evaluations, about a tenth of a second.
/// Paid once per board that is fine; paid once per decision it is not, and a
/// benchmark of ten thousand hands would take hours instead of minutes.
///
/// So the whole field is bucketed the first time a board is seen and kept. Each
/// board costs about a kilobyte, and the hand that follows — turn, river, every
/// decision on each — is then a lookup. The cache is cleared wholesale when it
/// grows past its bound rather than evicted piecemeal: boards recur within a
/// hand and almost never across them, so there is no useful ordering to keep.
#[derive(Debug)]
pub struct Reader {
    buckets: usize,
    limit: usize,
    seen: HashMap<Vec<Card>, Vec<u8>>,
}

impl Reader {
    /// A reader cutting strength into `buckets` groups.
    ///
    /// This must be the number the solve was trained with. A strength read
    /// against a different number of groups is a different quantity, and the
    /// lookup would land on a key meaning something else.
    pub fn new(buckets: usize) -> Reader {
        Reader {
            buckets,
            limit: 4096,
            seen: HashMap::new(),
        }
    }

    pub fn buckets(&self) -> usize {
        self.buckets
    }

    /// How strong `hole` is on the board showing, or `None` if that is not a
    /// board and a holding off it.
    pub fn strength(&mut self, visible: &[Card], hole: [Card; 2]) -> Option<u8> {
        if !(3..=5).contains(&visible.len()) || hole[0] == hole[1] {
            return None;
        }
        let mut board = CardSet::empty();
        for card in visible {
            if !board.insert(*card) {
                return None;
            }
        }
        if hole.iter().any(|card| board.contains(*card)) {
            return None;
        }

        if !self.seen.contains_key(visible) {
            if self.seen.len() >= self.limit {
                self.seen.clear();
            }
            let field = field_of(board);
            let equity = equity_on(visible, &field);
            let mut buckets = vec![0u8; field.len()];
            assign_buckets(&equity, self.buckets, &mut buckets);
            self.seen.insert(visible.to_vec(), buckets);
        }

        let buckets = self.seen.get(visible)?;
        let field = field_of(board);
        let mut wanted = [hole[0], hole[1]];
        wanted.sort_by_key(|card| card.index());
        let at = field
            .iter()
            .position(|holding| {
                let mut pair = *holding;
                pair.sort_by_key(|card| card.index());
                pair == wanted
            })
            .expect("a holding off the board is in the field");
        Some(buckets[at])
    }

    /// How many boards are remembered.
    pub fn len(&self) -> usize {
        self.seen.len()
    }

    pub fn is_empty(&self) -> bool {
        self.seen.is_empty()
    }
}

/// Every two-card holding a set of visible cards leaves, in a fixed order.
fn field_of(shown: CardSet) -> Vec<[Card; 2]> {
    let live: Vec<Card> = CardSet::full_deck().difference(shown).iter().collect();
    let mut field = Vec::with_capacity(live.len() * (live.len() - 1) / 2);
    for first in 0..live.len() {
        for second in first + 1..live.len() {
            field.push([live[first], live[second]]);
        }
    }
    field
}

/// How strong a holding is on a board that is actually in front of you.
///
/// # Why this exists next to a table of precomputed strengths
///
/// The sample holds twenty thousand boards. A real table deals from two and a
/// half million, so the board in play is almost never one of them, and looking
/// up the nearest is not a thing that can be done — boards have no useful
/// notion of nearness. What can be repeated is the *procedure*: measure this
/// holding's equity on the cards showing, measure every other holding's, and
/// take the percentile. That is precisely what the sample stored, so the number
/// this returns means the same thing the solve was trained against.
///
/// # Where it differs, and by how much
///
/// One thing cannot be repeated. When the sample bucketed a flop it drew its
/// field from the holdings left by all five board cards, because the whole
/// runout had already been dealt. Here only three cards are showing, so the
/// field is the larger one those three leave — 1176 holdings rather than 1081.
/// A percentile taken over a slightly different field is a slightly different
/// percentile, and that is the whole of the remaining disagreement now that
/// neither side samples. The test below measures it rather than assuming it
/// away.
///
/// # Determinism
///
/// There is no randomness left in this: every runout is dealt. The same board
/// and holding always return the same bucket, which matters because the bot
/// re-reads the table repeatedly while deciding, and a strength that flickered
/// between neighbouring buckets would make it change its mind about a hand
/// nothing had happened to.
pub fn strength_of(visible: &[Card], hole: [Card; 2], buckets: usize) -> Option<u8> {
    if !(3..=5).contains(&visible.len()) || hole[0] == hole[1] {
        return None;
    }
    let mut board: CardSet = CardSet::empty();
    for card in visible {
        if !board.insert(*card) {
            return None;
        }
    }
    if hole.iter().any(|card| board.contains(*card)) {
        return None;
    }

    // Every holding the visible cards leave, hero's among them.
    let field = field_of(board);
    let mine = field
        .iter()
        .position(|holding| holding.contains(&hole[0]) && holding.contains(&hole[1]))?;

    let equity = equity_on(visible, &field);
    let ours = equity[mine];

    // The same cut the sample used: rank within the field, scaled to the number
    // of groups. Ties are ranked below, matching the sort there.
    let below = equity.iter().filter(|other| **other < ours).count();
    Some((below * buckets / field.len()) as u8)
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
fn measure_board(rng: &mut Rng, buckets: usize) -> (Board, Vec<[Card; 2]>) {
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
        let equity = equity_on(&cards[..street.board_cards()], &holdings);
        let at = street_index(street).expect("postflop") * HOLDINGS;
        assign_buckets(&equity, buckets, &mut all[at..at + HOLDINGS]);
    }

    let mut hand = [cards[0]; 7];
    hand[2..].copy_from_slice(&cards);
    let made = holdings
        .iter()
        .map(|holding| {
            hand[..2].copy_from_slice(holding);
            evaluate(&hand).to_bits()
        })
        .collect();

    (
        Board {
            cards,
            buckets: all,
            made,
        },
        holdings,
    )
}

/// Each holding's share of the pot against a uniformly random opponent.
///
/// Every card that could still come is dealt, not a sample of them: 1176 pairs
/// on the flop, 48 singles on the turn, and nothing on the river, which is
/// already finished. So this is the exact equity rather than an estimate of it.
///
/// # Why exactness is worth the cost here
///
/// Sampling was tried first and was much cheaper. It failed for a reason worth
/// recording. Holdings cluster tightly by equity — at forty-eight groups, two
/// dozen holdings share each group and the whole group spans about two parts in
/// a thousand of equity. Two hundred sampled runouts carry an error many times
/// that. The bucket a holding landed in was therefore mostly noise, and worse,
/// *independent* noise: the sample bucketed a board with one set of random
/// runouts and a live reading of the same board would draw others, so the same
/// hand read differently at the table than it had when the solve was trained on
/// it. Enumeration removes that entirely — both sides now compute the same
/// function of the same cards.
///
/// The cards still to come are dealt afresh rather than taken from the board
/// this sample happens to hold, which is what stops a flop bucket from knowing
/// the river.
fn equity_on(visible: &[Card], holdings: &[[Card; 2]]) -> Vec<f64> {
    let seen = visible.len();
    let missing = 5 - seen;
    let shown: CardSet = visible.iter().copied().collect();
    let deck: Vec<Card> = CardSet::full_deck().difference(shown).iter().collect();

    let mut totals = vec![0.0f64; holdings.len()];
    let mut ranks = vec![0u32; holdings.len()];
    let mut order: Vec<u32> = Vec::with_capacity(holdings.len());
    let mut hand = [visible[0]; 7];
    let mut full = [visible[0]; 5];
    full[..seen].copy_from_slice(visible);
    let mut passes = 0u32;

    // Cards in a holding are not excluded from the runout: one runout is dealt
    // and shared by every holding, so a holding colliding with it is one that
    // could not have been dealt, and its own comparison is what suffers rather
    // than everybody else's.
    let mut measure = |full: &[Card; 5],
                       ranks: &mut Vec<u32>,
                       order: &mut Vec<u32>,
                       totals: &mut Vec<f64>| {
        hand[2..].copy_from_slice(full);
        for (index, holding) in holdings.iter().enumerate() {
            if holding.iter().any(|card| full[seen..].contains(card)) {
                ranks[index] = 0;
                continue;
            }
            hand[..2].copy_from_slice(holding);
            ranks[index] = evaluate(&hand).to_bits();
        }
        // A holding's share is how many others it beats plus half of those it
        // ties, over the field — its equity against a random opponent.
        order.clear();
        order.extend(ranks.iter().copied());
        order.sort_unstable();
        let field = order.len() as f64;
        for (index, rank) in ranks.iter().enumerate() {
            let beaten = order.partition_point(|other| other < rank) as f64;
            let tied = order.partition_point(|other| other <= rank) as f64 - beaten;
            totals[index] += (beaten + tied / 2.0) / field;
        }
    };

    match missing {
        0 => {
            measure(&full, &mut ranks, &mut order, &mut totals);
            passes = 1;
        }
        1 => {
            for card in &deck {
                full[seen] = *card;
                measure(&full, &mut ranks, &mut order, &mut totals);
                passes += 1;
            }
        }
        2 => {
            for first in 0..deck.len() {
                for second in first + 1..deck.len() {
                    full[seen] = deck[first];
                    full[seen + 1] = deck[second];
                    measure(&full, &mut ranks, &mut order, &mut totals);
                    passes += 1;
                }
            }
        }
        _ => unreachable!("a board shows three, four or five cards"),
    }

    let passes = passes as f64;
    totals.iter().map(|total| total / passes).collect()
}

/// Cuts a set of equities into groups of equal population, weakest first.
///
/// Equal population rather than equal width: hands cluster heavily around the
/// middle, and equal-width bands would spend most of their resolution on the
/// few hands nobody has to think about.
///
/// # Holdings of equal equity share a bucket
///
/// The obvious way to write this is to sort and hand out ranks by position,
/// which is what it did at first. That splits ties across a boundary according
/// to where the sort happened to put them — and sort order is not a property of
/// a poker hand. It showed up as the river disagreeing with itself: ties are
/// exact and common there, so the stored bucket and a live reading of the very
/// same board sorted two different lists and came back with numbers up to
/// nineteen groups apart. Ranking a tie run by the count strictly below it
/// makes the answer a function of the cards alone, at the cost of groups being
/// exactly equal in population only where equities are distinct.
fn assign_buckets(equity: &[f64], buckets: usize, out: &mut [u8]) {
    let mut order: Vec<usize> = (0..equity.len()).collect();
    order.sort_by(|a, b| equity[*a].total_cmp(&equity[*b]));
    let mut at = 0;
    while at < order.len() {
        let mut end = at + 1;
        while end < order.len() && equity[order[end]] == equity[order[at]] {
            end += 1;
        }
        let bucket = (at * buckets / equity.len()) as u8;
        for index in &order[at..end] {
            out[*index] = bucket;
        }
        at = end;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::parse_cards;

    fn small() -> Textures {
        Textures::sample(8, 10, 0x7E47, 4)
    }

    /// The live reading and the stored one must agree, and this says how well.
    ///
    /// On the river they must agree exactly. Both rank the same 1081 holdings,
    /// by an equity that is now enumerated rather than sampled, under the same
    /// tie rule — so any difference at all means the two have drifted apart as
    /// computations, which is how the tie bug was found.
    ///
    /// Before the river they cannot agree exactly and should not be asked to.
    /// The stored flop bucket ranked its holding against the field left by all
    /// five board cards, because the whole runout had been dealt by then; a
    /// live reading only knows the three showing, so its field is 1176 rather
    /// than 1081. That is the entire remaining gap, and it is bounded here
    /// rather than assumed away: if the abstraction changes and it widens, this
    /// fails rather than quietly letting the bot read its own hand differently
    /// from how the solve was trained.
    #[test]
    fn reading_a_live_board_agrees_with_the_stored_bucket() {
        const BUCKETS: usize = 48;
        let textures = Textures::sample(4, BUCKETS, 0x51DE, 4);
        for street in [Street::Flop, Street::Turn, Street::River] {
            let (mut total, mut worst, mut counted) = (0i64, 0i64, 0i64);
            for index in 0..textures.len() {
                let board = textures.board(index);
                let holdings = textures.holdings(index);
                // A spread of holdings rather than all 1081, since each costs a
                // full field measurement.
                for holding in (0..HOLDINGS).step_by(37) {
                    let stored = textures
                        .strength(index, street, holding)
                        .expect("postflop is bucketed") as i64;
                    let live = strength_of(board.visible(street), holdings[holding], BUCKETS)
                        .expect("a real board and a holding off it")
                        as i64;
                    let gap = (stored - live).abs();
                    total += gap;
                    worst = worst.max(gap);
                    counted += 1;
                }
            }
            let mean = total as f64 / counted as f64;
            let (allowed_mean, allowed_worst) = match street {
                Street::River => (0.0, 0),
                _ => (1.0, 5),
            };
            assert!(
                mean <= allowed_mean && worst <= allowed_worst,
                "on the {street:?} a live reading drifts from the trained strength by {mean:.3} buckets on average and {worst} at worst, past the {allowed_mean}/{allowed_worst} the field difference accounts for; the solve would be answering a question other than the one asked"
            );
        }
    }

    /// A cached reading and an uncached one are the same reading.
    ///
    /// The cache exists because ranking a holding costs a tenth of a second and
    /// a hand asks several times. It would be worth nothing if it answered
    /// differently, so this checks the two against each other across streets
    /// and across the boundary where the cache fills.
    #[test]
    fn a_remembered_board_reads_the_same_as_a_fresh_one() {
        let textures = Textures::sample(3, 48, 0x51DE, 4);
        let mut reader = Reader::new(48);
        for index in 0..textures.len() {
            let board = textures.board(index);
            let holdings = textures.holdings(index);
            for street in [Street::Flop, Street::Turn, Street::River] {
                let visible = board.visible(street);
                for holding in (0..HOLDINGS).step_by(211) {
                    let hole = holdings[holding];
                    let direct = strength_of(visible, hole, 48);
                    assert_eq!(reader.strength(visible, hole), direct);
                    // Again, now that it is certainly cached.
                    assert_eq!(reader.strength(visible, hole), direct);
                }
            }
        }
        assert!(!reader.is_empty(), "nothing was remembered");
    }

    #[test]
    fn a_reader_refuses_what_is_not_a_board() {
        let mut reader = Reader::new(48);
        let board = parse_cards("Ah Kd 7c").expect("three cards");
        let hole = parse_cards("Qs Qh").expect("two cards");
        assert!(reader.strength(&board, [hole[0], hole[1]]).is_some());
        assert_eq!(reader.strength(&board[..2], [hole[0], hole[1]]), None);
        let clash = parse_cards("Ah Qh").expect("two cards");
        assert_eq!(reader.strength(&board, [clash[0], clash[1]]), None);
    }

    #[test]
    fn a_live_reading_is_the_same_every_time() {
        let board = parse_cards("Ah Kd 7c").expect("three cards");
        let hole = parse_cards("Qs Qh").expect("two cards");
        let hole = [hole[0], hole[1]];
        let first = strength_of(&board, hole, 48).expect("a bucket");
        for _ in 0..4 {
            assert_eq!(strength_of(&board, hole, 48), Some(first));
        }
    }

    #[test]
    fn a_holding_that_cannot_exist_has_no_strength() {
        let board = parse_cards("Ah Kd 7c").expect("three cards");
        let clash = parse_cards("Ah Qh").expect("two cards");
        assert_eq!(strength_of(&board, [clash[0], clash[1]], 48), None);
        let qs = parse_cards("Qs").expect("one card")[0];
        assert_eq!(strength_of(&board, [qs, qs], 48), None);
        let two = parse_cards("Ah Kd").expect("two cards");
        let hole = parse_cards("Qs Qh").expect("two cards");
        assert_eq!(strength_of(&two, [hole[0], hole[1]], 48), None);
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

    /// No group swallows the field.
    ///
    /// Groups are cut by population, and the failure this guards against is the
    /// cut collapsing — most hands landing in one group, so that the
    /// abstraction says almost nothing about a hand and the solve learns one
    /// strategy for everything.
    ///
    /// It is stated as the largest group rather than the spread between largest
    /// and smallest, because the spread measures the wrong thing on the river.
    /// Holdings of identical equity share a bucket rather than being split by
    /// sort order, and river ties are heavy — every holding that plays the
    /// board has the same equity, and on a board carrying a straight that can
    /// be most of them. A tie run can then span an entire boundary and leave
    /// the group beyond it empty, which looks alarming as a spread and is
    /// simply what those cards mean. Before the river, equity is an average
    /// over many runouts, exact ties are rare, and the cut comes out close to
    /// even; the bounds follow that.
    #[test]
    fn no_strength_group_swallows_the_field() {
        let textures = Textures::sample(6, 10, 0x7E47, 4);
        let even = 1.0 / textures.buckets() as f64;
        for street in [Street::Flop, Street::Turn, Street::River] {
            // Ties are as common as the street has runouts to average over:
            // 1176 on the flop, 48 on the turn, one exact number on the river.
            let allowed = match street {
                Street::Flop => 2.0 * even,
                Street::Turn => 2.5 * even,
                _ => 4.0 * even,
            };
            for index in 0..textures.len() {
                let mut counts = vec![0usize; textures.buckets()];
                for holding in 0..HOLDINGS {
                    let bucket = textures
                        .strength(index, street, holding)
                        .expect("postflop is bucketed");
                    counts[bucket as usize] += 1;
                }
                let largest = counts.iter().max().copied().unwrap_or(0);
                let share = largest as f64 / HOLDINGS as f64;
                assert!(
                    share < allowed,
                    "on the {street:?} the largest group holds {largest} of {HOLDINGS} holdings, {:.0}% of the field against an even share of {:.0}% — past the {:.0}% ties account for",
                    share * 100.0,
                    even * 100.0,
                    allowed * 100.0
                );
            }
        }
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
        let equity_one = equity_on(&first[..3], &holdings_one);
        let equity_two = equity_on(&second[..3], &holdings_two);

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
                "{holding:?} priced {:.4} on one runout and {:.4} on another, a gap of {gap:.4} — the flop is being told what is coming",
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
        let equity = equity_on(&board[..5], &holdings);

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

    /// Showdowns must not be settled by comparing buckets.
    ///
    /// A bucket holds many hands of roughly equal strength, so comparing them
    /// would split every pot between two holdings that landed in the same
    /// group — which at twenty groups is one showdown in twenty decided by a
    /// tie that is not one.
    #[test]
    fn a_showdown_separates_hands_that_share_a_bucket() {
        let textures = Textures::sample(4, 10, 0x7E47, 4);
        let mut sharing = 0;
        let mut separated = 0;
        for board in 0..textures.len() {
            for first in 0..HOLDINGS {
                for second in first + 1..HOLDINGS {
                    let a = textures.strength(board, Street::River, first).expect("river");
                    let b = textures.strength(board, Street::River, second).expect("river");
                    if a != b {
                        continue;
                    }
                    sharing += 1;
                    if textures.showdown(board, first, second) != std::cmp::Ordering::Equal {
                        separated += 1;
                    }
                }
            }
        }
        assert!(sharing > 1_000, "only {sharing} pairs shared a bucket");
        assert!(
            separated * 2 > sharing,
            "of {sharing} pairs sharing a bucket, only {separated} were actually separated at showdown — buckets are being treated as strength itself"
        );
    }

    #[test]
    fn a_showdown_agrees_with_the_hand_that_is_actually_better() {
        // A board where the nut flush is available; the hand holding it must
        // beat one holding nothing, whatever buckets they fall in.
        let textures = Textures::sample(1, 10, 0x7E47, 2);
        let board = textures.board(0).cards();
        let holdings = textures.holdings(0);
        let mut hand = [board[0]; 7];
        hand[2..].copy_from_slice(board);

        // Compare every pair against a fresh evaluation of the same two hands.
        for first in (0..HOLDINGS).step_by(97) {
            for second in (0..HOLDINGS).step_by(89) {
                hand[..2].copy_from_slice(&holdings[first]);
                let a = crate::eval::evaluate(&hand);
                hand[..2].copy_from_slice(&holdings[second]);
                let b = crate::eval::evaluate(&hand);
                assert_eq!(
                    textures.showdown(0, first, second),
                    a.cmp(&b),
                    "{:?} against {:?}",
                    holdings[first],
                    holdings[second]
                );
            }
        }
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
