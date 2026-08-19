//! Showdown equity for pots with more than three players.
//!
//! [`crate::threeway`] tabulates every triple of the 169 hand classes exactly.
//! That does not extend: the number of entries grows as a rising factorial, and
//! measured at 400 runouts each the cost runs
//!
//! ```text
//! 3 players        818,805 entries      9 MB      69 seconds
//! 4 players     35,208,615 entries    537 MB      66 minutes
//! 5 players  1,218,218,079 entries     23 GB      47 hours
//! 6 players 35,328,324,291 entries    790 GB      69 days
//! ```
//!
//! So four players is the last table that can be built hand-class by hand-class,
//! and five is where the approach has to change rather than merely take longer.
//!
//! # Buckets, and only where they are needed
//!
//! Beyond four players the classes are grouped by strength and the table is
//! built over groups. That loses precision, and it is worth being exact about
//! where: at 30 buckets roughly six hand classes share one equity number. But
//! it loses it where it costs least. In a six-way pot the difference between
//! ace-king suited and ace-queen suited is small next to how many ways there
//! are to be beaten, and full precision stays where most of the money is —
//! heads-up and three-handed pots, which this module does not touch.
//!
//! # Strength comes from measurement
//!
//! The buckets are not hand-picked. Each class is scored by its average equity
//! against a uniformly random hand, taken from the pairwise table that already
//! exists, and the ranking that falls out is what gets cut into groups. Nobody
//! decides that pocket twos belongs above ace-three offsuit; the measurement
//! does.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::abstraction::{HandClass, NUM_HAND_CLASSES};
use crate::card::{Card, CardSet};
use crate::eval::evaluate;
use crate::pushfold::{sample_hand, EquityTable};
use crate::rng::Rng;

const MAGIC: &[u8; 8] = b"PKEQW\0\0\0";
const VERSION: u32 = 1;

/// The most players a pot can be tabulated for.
pub const MAX_PLAYERS: usize = 7;

/// Groups the 169 hand classes by measured strength.
///
/// A grouping of 169 is no grouping at all, and every class keeps its own
/// entry — which is what four-handed uses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Buckets {
    /// Bucket index per hand class.
    of_class: Vec<u8>,
    count: usize,
}

impl Buckets {
    /// Cuts the classes into `count` groups of equal population, strongest
    /// first.
    ///
    /// Equal population rather than equal strength: the interesting decisions
    /// cluster among hands of middling strength, and equal-width bands would
    /// spend most of their resolution on the handful of hands nobody has to
    /// think about.
    pub fn by_strength(equity: &EquityTable, count: usize) -> Buckets {
        assert!(
            (1..=NUM_HAND_CLASSES).contains(&count),
            "a grouping into {count} is not a grouping of {NUM_HAND_CLASSES} classes"
        );

        let mut ranked: Vec<(usize, f64)> = HandClass::all()
            .map(|class| {
                // Average equity against a uniformly random hand, weighted by
                // how many ways each opponent holding can be dealt.
                let (mut total, mut weight) = (0.0, 0.0);
                for other in HandClass::all() {
                    let combos = other.combos() as f64;
                    total += equity.get(class, other) * combos;
                    weight += combos;
                }
                (class.index(), total / weight)
            })
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));

        let mut of_class = vec![0u8; NUM_HAND_CLASSES];
        for (rank, (index, _)) in ranked.iter().enumerate() {
            of_class[*index] = (rank * count / NUM_HAND_CLASSES) as u8;
        }
        Buckets { of_class, count }
    }

    /// One bucket per class, which is no grouping at all.
    pub fn none() -> Buckets {
        Buckets {
            of_class: (0..NUM_HAND_CLASSES).map(|i| i as u8).collect(),
            count: NUM_HAND_CLASSES,
        }
    }

    #[inline]
    pub fn of(&self, class: HandClass) -> usize {
        self.of_class[class.index()] as usize
    }

    pub fn count(&self) -> usize {
        self.count
    }
}

/// How many entries a table of `players` over `buckets` groups needs.
pub fn entries(players: usize, buckets: usize) -> usize {
    // Multisets of size `players` drawn from `buckets` symbols.
    let mut total = 1usize;
    for step in 0..players {
        total = total * (buckets + step) / (step + 1);
    }
    total
}

/// Where a sorted multiset lives in the table.
///
/// A non-decreasing multiset maps to a strictly increasing combination by
/// adding each element's position to it, and that combination has a standard
/// index. Written out rather than looked up because it is called once per
/// showdown during training.
fn index_of(sorted: &[usize]) -> usize {
    sorted
        .iter()
        .enumerate()
        .map(|(position, &value)| choose(value + position, position + 1))
        .sum()
}

/// The sorted multiset living at `index`, the inverse of [`index_of`].
///
/// Decoding on demand rather than listing every multiset up front. The
/// four-handed table has 35 million entries, and holding them as vectors would
/// cost about two gigabytes before a single showdown had been dealt.
fn multiset_at(mut index: usize, players: usize, symbols: usize) -> Vec<usize> {
    let mut out = vec![0usize; players];
    for position in (0..players).rev() {
        // The largest value whose block still fits inside what is left.
        let mut value = position;
        while choose(value + 1, position + 1) <= index && value + 1 < symbols + position {
            value += 1;
        }
        index -= choose(value, position + 1);
        out[position] = value - position;
    }
    out
}

/// `n` choose `k`, computed without overflowing for the sizes here.
fn choose(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let mut result = 1usize;
    for step in 0..k {
        result = result * (n - step) / (step + 1);
    }
    result
}

/// Showdown shares for pots of one size.
#[derive(Debug, Clone, PartialEq)]
pub struct WideEquity {
    players: usize,
    buckets: Buckets,
    /// `players` shares per entry, laid out flat.
    shares: Vec<f32>,
}

impl WideEquity {
    /// Builds the table by sampling `samples` runouts per entry.
    pub fn sampled_parallel(
        players: usize,
        buckets: Buckets,
        samples: u32,
        seed: u64,
        threads: usize,
    ) -> WideEquity {
        assert!(
            (2..=MAX_PLAYERS).contains(&players),
            "a pot has between two and {MAX_PLAYERS} players, not {players}"
        );

        let count = entries(players, buckets.count());
        let symbols = buckets.count();

        // A bucket is many classes; pick a representative for each so the
        // sampler has real hands to deal.
        let example: Vec<Vec<HandClass>> = (0..buckets.count())
            .map(|bucket| {
                HandClass::all()
                    .filter(|class| buckets.of(*class) == bucket)
                    .collect()
            })
            .collect();

        // Workers take a range of indices and decode each, so nothing larger
        // than the answer itself is ever held.
        let threads = threads.max(1);
        let span = count.div_ceil(threads);
        let even = 1.0 / players as f32;
        let mut shares = vec![even; count * players];

        std::thread::scope(|scope| {
            let mut rest = shares.as_mut_slice();
            let mut start = 0usize;
            let mut workers = Vec::new();
            while start < count {
                let end = (start + span).min(count);
                let (mine, tail) = rest.split_at_mut((end - start) * players);
                rest = tail;
                let example = &example;
                workers.push(scope.spawn(move || {
                    let mut deck = Vec::with_capacity(52);
                    for index in start..end {
                        let combo = multiset_at(index, players, symbols);
                        let mut rng = Rng::new(seed_for(seed, &combo));
                        let value = sample(&combo, example, samples, &mut rng, &mut deck);
                        let at = (index - start) * players;
                        mine[at..at + players].copy_from_slice(&value);
                    }
                }));
                start = end;
            }
            for worker in workers {
                worker.join().expect("equity worker panicked");
            }
        });
        WideEquity {
            players,
            buckets,
            shares,
        }
    }

    /// The players' shares of a pot they all reach showdown in.
    ///
    /// Shares come back in the order the classes were given.
    ///
    /// # Panics
    /// Panics unless exactly as many classes are given as the table was built
    /// for — a four-way table cannot settle a five-way pot, and quietly
    /// answering anyway is how a solver learns something that is not true.
    pub fn get(&self, classes: &[HandClass]) -> Vec<f64> {
        assert_eq!(
            classes.len(),
            self.players,
            "this table settles {}-way pots",
            self.players
        );

        let mut order: Vec<(usize, usize)> = classes
            .iter()
            .enumerate()
            .map(|(seat, class)| (self.buckets.of(*class), seat))
            .collect();
        order.sort_unstable();
        let sorted: Vec<usize> = order.iter().map(|(bucket, _)| *bucket).collect();
        let at = index_of(&sorted) * self.players;

        let mut out = vec![0.0; self.players];
        for (slot, (_, seat)) in order.iter().enumerate() {
            out[*seat] = self.shares[at + slot] as f64;
        }
        out
    }

    pub fn players(&self) -> usize {
        self.players
    }

    pub fn buckets(&self) -> usize {
        self.buckets.count()
    }

    pub fn len(&self) -> usize {
        self.shares.len() / self.players
    }

    pub fn is_empty(&self) -> bool {
        self.shares.is_empty()
    }

    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = BufWriter::new(File::create(path)?);
        file.write_all(MAGIC)?;
        file.write_all(&VERSION.to_le_bytes())?;
        file.write_all(&(self.players as u32).to_le_bytes())?;
        file.write_all(&(self.buckets.count as u32).to_le_bytes())?;
        for bucket in &self.buckets.of_class {
            file.write_all(&[*bucket])?;
        }
        file.write_all(&(self.shares.len() as u64).to_le_bytes())?;
        for share in &self.shares {
            file.write_all(&share.to_le_bytes())?;
        }
        file.flush()
    }

    pub fn load(path: impl AsRef<Path>) -> io::Result<WideEquity> {
        let mut file = BufReader::new(File::open(path)?);
        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(invalid("not a wide equity table"));
        }
        let mut word = [0u8; 4];
        file.read_exact(&mut word)?;
        if u32::from_le_bytes(word) != VERSION {
            return Err(invalid("unsupported wide equity version"));
        }
        file.read_exact(&mut word)?;
        let players = u32::from_le_bytes(word) as usize;
        file.read_exact(&mut word)?;
        let count = u32::from_le_bytes(word) as usize;

        let mut of_class = vec![0u8; NUM_HAND_CLASSES];
        file.read_exact(&mut of_class)?;
        let mut length = [0u8; 8];
        file.read_exact(&mut length)?;
        let length = u64::from_le_bytes(length) as usize;

        let mut bytes = vec![0u8; length * 4];
        file.read_exact(&mut bytes)?;
        let shares: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes(word.try_into().expect("four bytes")))
            .collect();

        if shares.len() != entries(players, count) * players {
            return Err(invalid("wide equity table is the wrong length for its shape"));
        }
        Ok(WideEquity {
            players,
            buckets: Buckets { of_class, count },
            shares,
        })
    }
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

/// Every non-decreasing multiset of `size` drawn from `symbols`.
///
/// Only used to check the index against an independent enumeration; building a
/// table walks indices instead, and never holds them all.
#[cfg(test)]
fn all_multisets(size: usize, symbols: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::with_capacity(entries(size, symbols));
    let mut current = vec![0usize; size];
    loop {
        out.push(current.clone());
        // Advance to the next non-decreasing tuple.
        let mut at = size;
        loop {
            if at == 0 {
                return out;
            }
            at -= 1;
            if current[at] + 1 < symbols {
                let next = current[at] + 1;
                for slot in current.iter_mut().skip(at) {
                    *slot = next;
                }
                break;
            }
        }
    }
}

/// A seed depending only on the entry, so the table is reproducible whatever
/// the core count.
fn seed_for(seed: u64, combo: &[usize]) -> u64 {
    let mut mixed = seed;
    for (position, value) in combo.iter().enumerate() {
        mixed ^= (*value as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) << (position % 8);
    }
    mixed
}

/// Samples one entry's shares.
fn sample(
    combo: &[usize],
    example: &[Vec<HandClass>],
    samples: u32,
    rng: &mut Rng,
    deck: &mut Vec<Card>,
) -> Vec<f32> {
    let players = combo.len();
    let first = Card::all().next().expect("a non-empty deck");
    let mut totals = vec![0.0f64; players];
    let mut trials = 0u32;
    let mut seven = [first; 7];
    let mut hands = vec![[first; 2]; players];
    let mut won = vec![false; players];

    'sample: for _ in 0..samples {
        let mut dead = CardSet::empty();
        for (slot, bucket) in combo.iter().enumerate() {
            // A bucket is a group of classes; draw one of them each time, so
            // the entry averages over the group rather than over one member.
            let members = &example[*bucket];
            if members.is_empty() {
                continue 'sample;
            }
            let class = members[rng.below(members.len() as u64) as usize];
            match sample_hand(class, rng, dead) {
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
        won.iter_mut().for_each(|w| *w = false);
        for (slot, hand) in hands.iter().enumerate() {
            seven[..2].copy_from_slice(hand);
            let rank = evaluate(&seven);
            match best {
                Some(current) if rank > current => {
                    best = Some(rank);
                    won.iter_mut().for_each(|w| *w = false);
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
        for slot in 0..players {
            if won[slot] {
                totals[slot] += 1.0 / winners as f64;
            }
        }
        trials += 1;
    }

    if trials == 0 {
        return vec![1.0 / players as f32; players];
    }

    // Positions holding the same bucket are the same hand as far as this table
    // can tell, so their shares must be equal. Independent sampling makes them
    // merely close, which would leave the lookup depending on which seat held
    // which of two indistinguishable hands. Pooling settles it and uses every
    // sample for every position it applies to.
    let mut shares: Vec<f32> = totals
        .iter()
        .map(|total| (total / trials as f64) as f32)
        .collect();
    let mut at = 0;
    while at < players {
        let mut end = at + 1;
        while end < players && combo[end] == combo[at] {
            end += 1;
        }
        let pooled = shares[at..end].iter().sum::<f32>() / (end - at) as f32;
        shares[at..end].fill(pooled);
        at = end;
    }
    shares
}

/// Every equity table a solve might need, chosen by how many players remain.
///
/// A pot contested by two is settled from the pairwise table, three from the
/// exact three-way one, and four or more from the wide tables. Which is used is
/// decided by the pot in front of it rather than by the size of the table, so a
/// seven-handed game that folds to two players settles that pot at full
/// precision — the coarse tables are consulted only for the pots that actually
/// need them.
#[derive(Debug, Clone)]
pub struct Showdown {
    pairwise: EquityTable,
    three: crate::threeway::ThreeWayEquity,
    /// Indexed by player count, so `wide[4]` settles four-way pots. Sizes with
    /// no table are absent rather than approximated.
    wide: Vec<Option<WideEquity>>,
}

impl Showdown {
    pub fn new(pairwise: EquityTable, three: crate::threeway::ThreeWayEquity) -> Showdown {
        Showdown {
            pairwise,
            three,
            wide: vec![None; MAX_PLAYERS + 1],
        }
    }

    /// Adds a table for pots of its own size.
    ///
    /// # Panics
    /// Panics if the table is for a size already covered exactly, since taking
    /// a bucketed answer where an exact one exists would be a silent downgrade.
    pub fn with(mut self, table: WideEquity) -> Showdown {
        let players = table.players();
        assert!(
            (4..=MAX_PLAYERS).contains(&players),
            "two- and three-way pots are settled exactly; a {players}-way table              would be consulted for neither"
        );
        self.wide[players] = Some(table);
        self
    }

    /// The largest pot this can settle. Beyond it, a solve has to refuse.
    pub fn reach(&self) -> usize {
        let mut most = 3;
        for players in 4..=MAX_PLAYERS {
            if self.wide[players].is_some() {
                most = players;
            } else {
                break;
            }
        }
        most
    }

    /// Shares of a pot, in the order the hands were given.
    ///
    /// # Panics
    /// Panics for a pot wider than [`Showdown::reach`]. That is a programming
    /// error rather than a table condition: a solve is refused up front unless
    /// its widest possible pot can be settled, so reaching one here means the
    /// check was skipped.
    pub fn shares(&self, hands: &[HandClass]) -> Vec<f64> {
        match hands.len() {
            0 => Vec::new(),
            1 => vec![1.0],
            2 => {
                let equity = self.pairwise.get(hands[0], hands[1]);
                vec![equity, 1.0 - equity]
            }
            3 => self.three.get(hands[0], hands[1], hands[2]).to_vec(),
            players => self.wide[players]
                .as_ref()
                .unwrap_or_else(|| {
                    panic!("no table settles a {players}-way pot; reach is {}", self.reach())
                })
                .get(hands),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn class(text: &str) -> HandClass {
        use crate::card::Rank;
        let mut chars = text.chars();
        let high = Rank::from_char(chars.next().expect("a rank")).expect("a rank");
        let low = Rank::from_char(chars.next().expect("a rank")).expect("a rank");
        HandClass::new(high, low, chars.next() == Some('s'))
    }

    fn equity() -> EquityTable {
        EquityTable::sampled_parallel(120, 0x51DE, 4)
    }

    #[test]
    fn the_entry_count_matches_the_published_arithmetic() {
        // The figures the module documents, so a change to the index would show
        // up here rather than as a table quietly the wrong size.
        assert_eq!(entries(3, NUM_HAND_CLASSES), 818_805);
        assert_eq!(entries(4, NUM_HAND_CLASSES), 35_208_615);
        assert_eq!(entries(5, 40), 1_086_008);
        assert_eq!(entries(6, 30), 1_623_160);
        assert_eq!(entries(7, 25), 2_629_575);
    }

    #[test]
    fn an_index_decodes_back_to_the_multiset_it_came_from() {
        // Building a table walks indices and decodes each one, so a decoder
        // that disagreed with the encoder would fill every slot with somebody
        // else's answer — silently, and at full speed.
        for (players, symbols) in [(2, 9), (3, 7), (4, 6), (5, 5), (4, 20)] {
            for index in 0..entries(players, symbols) {
                let combo = multiset_at(index, players, symbols);
                assert!(
                    combo.windows(2).all(|w| w[0] <= w[1]),
                    "{combo:?} is not sorted"
                );
                assert!(combo.iter().all(|&v| v < symbols), "{combo:?} out of range");
                assert_eq!(index_of(&combo), index, "{combo:?} round-tripped wrong");
            }
        }
    }

    #[test]
    fn every_multiset_gets_its_own_slot() {
        for (players, symbols) in [(2, 9), (3, 7), (4, 6), (5, 5)] {
            let combos = all_multisets(players, symbols);
            assert_eq!(combos.len(), entries(players, symbols));
            let mut seen = vec![false; combos.len()];
            for combo in &combos {
                assert!(combo.windows(2).all(|w| w[0] <= w[1]), "{combo:?} is unsorted");
                let at = index_of(combo);
                assert!(!seen[at], "{combo:?} collided at {at}");
                seen[at] = true;
            }
            assert!(seen.into_iter().all(|s| s), "a slot was never filled");
        }
    }

    #[test]
    fn strength_buckets_put_the_best_hands_at_the_top() {
        // Nobody assigns these; they fall out of measured equity against a
        // random hand. What the test asks is only that the measurement agrees
        // with what every poker player already knows.
        let buckets = Buckets::by_strength(&equity(), 10);
        assert_eq!(buckets.of(class("AA")), 0, "aces are in the strongest group");
        assert!(
            buckets.of(class("72o")) >= 8,
            "seven-deuce is near the bottom, got {}",
            buckets.of(class("72o"))
        );
        assert!(buckets.of(class("AKs")) < buckets.of(class("T9s")));
        assert!(buckets.of(class("QQ")) < buckets.of(class("55")));
    }

    #[test]
    fn no_grouping_leaves_every_class_its_own() {
        let plain = Buckets::none();
        assert_eq!(plain.count(), NUM_HAND_CLASSES);
        assert_eq!(plain.of(class("AA")), class("AA").index());
    }

    #[test]
    fn shares_of_one_pot_add_up_to_one_pot() {
        let table = WideEquity::sampled_parallel(4, Buckets::by_strength(&equity(), 8), 60, 0x4EED, 4);
        let hands = [class("AA"), class("KK"), class("72o"), class("JTs")];
        let shares = table.get(&hands);
        let total: f64 = shares.iter().sum();
        assert!((total - 1.0).abs() < 1e-6, "{shares:?} sums to {total}");
    }

    #[test]
    fn shares_follow_the_hands_however_they_are_ordered() {
        let table = WideEquity::sampled_parallel(4, Buckets::by_strength(&equity(), 8), 60, 0x4EED, 4);
        let straight = table.get(&[class("AA"), class("72o"), class("KK"), class("JTs")]);
        let shuffled = table.get(&[class("KK"), class("JTs"), class("AA"), class("72o")]);
        assert!((straight[0] - shuffled[2]).abs() < 1e-9);
        assert!((straight[1] - shuffled[3]).abs() < 1e-9);
        assert!((straight[2] - shuffled[0]).abs() < 1e-9);
        assert!((straight[3] - shuffled[1]).abs() < 1e-9);
    }

    #[test]
    fn a_stronger_hand_takes_a_bigger_share() {
        let table =
            WideEquity::sampled_parallel(4, Buckets::by_strength(&equity(), 12), 400, 0x4EED, 4);
        let shares = table.get(&[class("AA"), class("JTs"), class("72o"), class("KK")]);
        assert!(shares[0] > shares[1], "aces over jack-ten: {shares:?}");
        assert!(shares[1] > shares[2], "jack-ten over seven-deuce: {shares:?}");
    }

    /// What bucketing actually costs, written down rather than implied.
    ///
    /// At twelve groups roughly fourteen hand classes share one equity number,
    /// and aces and kings land in the same one — so this table cannot tell them
    /// apart, and says so by giving them the same share. That is the trade being
    /// made for tables of five players and up, where the exact version would run
    /// to tens of gigabytes and days of work.
    ///
    /// Four-handed does not pay it: that table is built at full precision, where
    /// every class keeps its own entry.
    #[test]
    fn coarse_buckets_cannot_tell_aces_from_kings() {
        let buckets = Buckets::by_strength(&equity(), 12);
        assert_eq!(
            buckets.of(class("AA")),
            buckets.of(class("KK")),
            "twelve groups is too few to separate them"
        );

        let fine = Buckets::by_strength(&equity(), NUM_HAND_CLASSES);
        assert_ne!(
            fine.of(class("AA")),
            fine.of(class("KK")),
            "at full precision they are distinct, which is what four-handed uses"
        );
    }

    #[test]
    fn a_table_refuses_a_pot_of_the_wrong_size() {
        let table = WideEquity::sampled_parallel(4, Buckets::by_strength(&equity(), 6), 20, 0x4EED, 4);
        let wrong = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            table.get(&[class("AA"), class("KK"), class("72o")])
        }));
        assert!(wrong.is_err(), "a four-way table cannot settle a three-way pot");
    }

    #[test]
    fn a_table_survives_a_trip_through_a_file() {
        let table = WideEquity::sampled_parallel(4, Buckets::by_strength(&equity(), 6), 10, 0x4EED, 4);
        let path = std::env::temp_dir().join("poker-wide-roundtrip.bin");
        table.save(&path).expect("save");
        assert_eq!(WideEquity::load(&path).expect("load"), table);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_file_that_is_not_a_wide_table_is_rejected() {
        let path = std::env::temp_dir().join("poker-wide-bad.bin");
        std::fs::write(&path, b"nonsense").expect("write");
        assert!(WideEquity::load(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }
}
