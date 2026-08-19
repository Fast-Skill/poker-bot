//! Heads-up push/fold No-Limit Hold'em.
//!
//! The small blind either moves all in or folds; the big blind either calls or
//! folds. It is the simplest complete NLHE game, and it is genuinely played —
//! short-stacked heads-up and late-stage tournament spots reduce to exactly
//! this.
//!
//! It is also the right first target for the solver. The tree is tiny (two
//! decisions), but the *cards* are real: 169 starting-hand classes, real
//! showdown equities, real blind structure. Published Nash push/fold charts
//! give an external reference, so a wrong answer is visible rather than merely
//! plausible.
//!
//! # Stakes and units
//!
//! Everything is in big blinds. The small blind posts 0.5, the big blind posts
//! 1, and `stack` is the effective stack both players sit behind.
//!
//! - small blind folds: `-0.5`
//! - small blind pushes, big blind folds: `+1.0`
//! - both all in: `±stack`, settled by equity

use crate::abstraction::{HandClass, NUM_HAND_CLASSES};
use crate::card::{Card, CardSet, Suit};
use crate::cfr::{Game, InfoKey};
use crate::eval::evaluate;
use crate::rng::Rng;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Fold, for either player.
pub const FOLD: usize = 0;
/// Push for the small blind, call for the big blind.
pub const PUSH: usize = 1;

/// Identifies a serialized equity table.
const TABLE_MAGIC: &[u8; 4] = b"PKEQ";
/// Bumped whenever the layout changes, so stale caches are rebuilt.
const TABLE_VERSION: u32 = 1;

/// Pairwise showdown equity between starting-hand classes.
///
/// `get(a, b)` is the share of the pot class `a` wins against class `b` on a
/// random runout, chops counted fractionally. Symmetric by construction:
/// `get(a, b) + get(b, a) == 1`.
#[derive(Debug, Clone)]
pub struct EquityTable {
    table: Vec<f32>,
}

impl EquityTable {
    /// Builds the table by sampling `samples` showdowns per class pairing.
    ///
    /// Only the upper triangle is sampled; the mirror follows from
    /// `equity(a, b) = 1 - equity(b, a)`, which halves the work. The diagonal
    /// is set to exactly 0.5, since a class against itself is symmetric and
    /// sampling would only add noise to a known answer.
    ///
    /// Roughly 1,000 samples gives about a percent of accuracy — enough for
    /// push/fold thresholds, which are not sensitive at that scale.
    pub fn sampled(samples: u32, rng: &mut Rng) -> EquityTable {
        let mut table = vec![0.5f32; NUM_HAND_CLASSES * NUM_HAND_CLASSES];
        let mut deck: Vec<Card> = Vec::with_capacity(48);

        for a in 0..NUM_HAND_CLASSES {
            let class_a = HandClass::from_index(a).expect("in range");
            for b in a + 1..NUM_HAND_CLASSES {
                let class_b = HandClass::from_index(b).expect("in range");
                let equity = sample_pair_equity(class_a, class_b, samples, rng, &mut deck);
                table[a * NUM_HAND_CLASSES + b] = equity as f32;
                table[b * NUM_HAND_CLASSES + a] = (1.0 - equity) as f32;
            }
        }
        EquityTable { table }
    }

    /// Builds the table across `threads` workers.
    ///
    /// Each class pairing is seeded from `seed` and its own indices, never from
    /// the worker it lands on, so the result is bit-identical no matter how many
    /// threads run it. A table that changed with the core count would make every
    /// downstream solve irreproducible.
    pub fn sampled_parallel(samples: u32, seed: u64, threads: usize) -> EquityTable {
        let pairs: Vec<(usize, usize)> = (0..NUM_HAND_CLASSES)
            .flat_map(|a| (a + 1..NUM_HAND_CLASSES).map(move |b| (a, b)))
            .collect();
        let threads = threads.max(1);
        let chunk = pairs.len().div_ceil(threads);

        let computed: Vec<Vec<(usize, usize, f64)>> = std::thread::scope(|scope| {
            let workers: Vec<_> = pairs
                .chunks(chunk)
                .map(|slice| {
                    scope.spawn(move || {
                        let mut deck = Vec::with_capacity(48);
                        slice
                            .iter()
                            .map(|&(a, b)| {
                                let mut rng = Rng::new(pair_seed(seed, a, b));
                                let class_a = HandClass::from_index(a).expect("in range");
                                let class_b = HandClass::from_index(b).expect("in range");
                                let equity = sample_pair_equity(
                                    class_a, class_b, samples, &mut rng, &mut deck,
                                );
                                (a, b, equity)
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

        let mut table = vec![0.5f32; NUM_HAND_CLASSES * NUM_HAND_CLASSES];
        for batch in computed {
            for (a, b, equity) in batch {
                table[a * NUM_HAND_CLASSES + b] = equity as f32;
                table[b * NUM_HAND_CLASSES + a] = (1.0 - equity) as f32;
            }
        }
        EquityTable { table }
    }

    /// Class `a`'s equity against class `b`.
    #[inline]
    pub fn get(&self, a: HandClass, b: HandClass) -> f64 {
        self.table[a.index() * NUM_HAND_CLASSES + b.index()] as f64
    }

    /// Writes the table to `path`, creating parent directories as needed.
    ///
    /// The format is a short header followed by row-major `f32` values —
    /// about 112 KB. Building a high-precision table takes minutes; reloading
    /// it takes milliseconds, so it is computed once and cached.
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = BufWriter::new(File::create(path)?);
        file.write_all(TABLE_MAGIC)?;
        file.write_all(&TABLE_VERSION.to_le_bytes())?;
        file.write_all(&(NUM_HAND_CLASSES as u32).to_le_bytes())?;
        for value in &self.table {
            file.write_all(&value.to_le_bytes())?;
        }
        file.flush()
    }

    /// Reads a table written by [`EquityTable::save`].
    pub fn load(path: impl AsRef<Path>) -> io::Result<EquityTable> {
        let mut file = BufReader::new(File::open(path)?);

        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if &magic != TABLE_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "not an equity table",
            ));
        }

        let mut word = [0u8; 4];
        file.read_exact(&mut word)?;
        let version = u32::from_le_bytes(word);
        if version != TABLE_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("equity table version {version}, expected {TABLE_VERSION}"),
            ));
        }

        file.read_exact(&mut word)?;
        let classes = u32::from_le_bytes(word) as usize;
        if classes != NUM_HAND_CLASSES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("table covers {classes} classes, expected {NUM_HAND_CLASSES}"),
            ));
        }

        let mut table = vec![0.0f32; NUM_HAND_CLASSES * NUM_HAND_CLASSES];
        for value in table.iter_mut() {
            file.read_exact(&mut word)?;
            *value = f32::from_le_bytes(word);
        }
        Ok(EquityTable { table })
    }

    /// Loads the cached table at `path`, building and saving it if absent.
    ///
    /// A cache that fails to load — corrupt, or written by an older version —
    /// is rebuilt rather than treated as an error, since it is derived data.
    pub fn load_or_build(
        path: impl AsRef<Path>,
        samples: u32,
        seed: u64,
        threads: usize,
    ) -> io::Result<EquityTable> {
        let path = path.as_ref();
        if let Ok(table) = EquityTable::load(path) {
            return Ok(table);
        }
        let table = EquityTable::sampled_parallel(samples, seed, threads);
        table.save(path)?;
        Ok(table)
    }
}

/// A per-pairing seed, independent of how work is divided across threads.
fn pair_seed(seed: u64, a: usize, b: usize) -> u64 {
    seed ^ ((a * NUM_HAND_CLASSES + b) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// Draws concrete cards for `class` that avoid `blocked`.
///
/// Returns `None` if no conflict-free draw turned up, which only happens for
/// heavily blocked classes such as a pair whose rank is already exhausted.
pub(crate) fn sample_hand(class: HandClass, rng: &mut Rng, blocked: CardSet) -> Option<[Card; 2]> {
    let suit = |rng: &mut Rng| Suit::from_index(rng.below(4) as u8).expect("0..4");
    for _ in 0..64 {
        let (high, low) = (class.high(), class.low());
        let cards = if class.is_suited() {
            let s = suit(rng);
            [Card::new(high, s), Card::new(low, s)]
        } else {
            let first = suit(rng);
            let mut second = suit(rng);
            while second == first {
                second = suit(rng);
            }
            [Card::new(high, first), Card::new(low, second)]
        };
        if !blocked.contains(cards[0]) && !blocked.contains(cards[1]) {
            return Some(cards);
        }
    }
    None
}

/// Estimates equity between two classes over random boards.
fn sample_pair_equity(
    a: HandClass,
    b: HandClass,
    samples: u32,
    rng: &mut Rng,
    deck: &mut Vec<Card>,
) -> f64 {
    let mut share = 0.0;
    let mut trials = 0u32;
    let mut hand = [Card::all().next().expect("non-empty deck"); 7];

    for _ in 0..samples {
        let Some(hero) = sample_hand(a, rng, CardSet::empty()) else {
            continue;
        };
        let used: CardSet = hero.iter().copied().collect();
        let Some(villain) = sample_hand(b, rng, used) else {
            continue;
        };

        let mut dead = used;
        dead.insert(villain[0]);
        dead.insert(villain[1]);

        deck.clear();
        deck.extend(CardSet::full_deck().difference(dead).iter());
        for slot in 0..5 {
            let pick = slot + rng.below((deck.len() - slot) as u64) as usize;
            deck.swap(slot, pick);
        }
        let board = &deck[..5];

        hand[..2].copy_from_slice(&hero);
        hand[2..].copy_from_slice(board);
        let hero_rank = evaluate(&hand);

        hand[..2].copy_from_slice(&villain);
        let villain_rank = evaluate(&hand);

        share += match hero_rank.cmp(&villain_rank) {
            std::cmp::Ordering::Greater => 1.0,
            std::cmp::Ordering::Equal => 0.5,
            std::cmp::Ordering::Less => 0.0,
        };
        trials += 1;
    }

    if trials == 0 {
        0.5
    } else {
        share / trials as f64
    }
}

/// Where a hand stands in the push/fold tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Node {
    /// Cards not yet dealt.
    Deal,
    /// Small blind to push or fold.
    SmallBlind,
    /// Big blind to call or fold, facing a push.
    BigBlind,
    /// Small blind folded, surrendering the small blind.
    SmallBlindFolded,
    /// Big blind folded to the push.
    BigBlindFolded,
    /// Both all in.
    Showdown,
}

/// A node in the push/fold tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct State {
    classes: [u8; 2],
    node: Node,
}

impl State {
    /// The hand class held by `player`, once dealt.
    pub fn hand(&self, player: usize) -> HandClass {
        HandClass::from_index(self.classes[player] as usize).expect("dealt class is in range")
    }
}

/// Heads-up push/fold Hold'em at a fixed stack depth.
#[derive(Debug, Clone)]
pub struct PushFold {
    stack: f64,
    equity: EquityTable,
}

impl PushFold {
    /// A game at `stack` big blinds deep, using `equity` for showdowns.
    ///
    /// # Panics
    /// Panics if `stack` is not at least the big blind — below that there is
    /// no meaningful decision, since the big blind is already all in.
    pub fn new(stack: f64, equity: EquityTable) -> PushFold {
        assert!(stack >= 1.0, "effective stack must cover the big blind");
        PushFold { stack, equity }
    }

    /// The effective stack, in big blinds.
    pub fn stack(&self) -> f64 {
        self.stack
    }

    /// The information set for `player` holding `class`.
    ///
    /// The two players' sets never collide: the small blind acts before any
    /// push exists, the big blind only after one.
    pub const fn info_key(player: usize, class: usize) -> InfoKey {
        (class as InfoKey) << 1 | player as InfoKey
    }
}

impl Game for PushFold {
    type State = State;

    fn initial(&self) -> State {
        State {
            classes: [0, 0],
            node: Node::Deal,
        }
    }

    fn is_terminal(&self, state: &State) -> bool {
        matches!(
            state.node,
            Node::SmallBlindFolded | Node::BigBlindFolded | Node::Showdown
        )
    }

    fn terminal_utility(&self, state: &State) -> f64 {
        match state.node {
            // The small blind surrenders the half blind already posted.
            Node::SmallBlindFolded => -0.5,
            // The push took the big blind's one blind uncontested.
            Node::BigBlindFolded => 1.0,
            Node::Showdown => {
                let equity = self.equity.get(state.hand(0), state.hand(1));
                // Both players risk `stack`; equity splits the doubled pot.
                (2.0 * equity - 1.0) * self.stack
            }
            other => unreachable!("{other:?} is not terminal"),
        }
    }

    fn is_chance(&self, state: &State) -> bool {
        state.node == Node::Deal
    }

    fn chance_outcomes(&self, state: &State) -> Vec<(State, f64)> {
        debug_assert!(self.is_chance(state));
        // Every ordered pair of classes, weighted by how many card
        // combinations each contains. Card removal *between* the two hands is
        // not modelled here; this path exists for exact best-response, while
        // `sample_chance` deals real cards and handles removal exactly.
        let mut outcomes = Vec::with_capacity(NUM_HAND_CLASSES * NUM_HAND_CLASSES);
        let mut total = 0.0;
        for a in HandClass::all() {
            for b in HandClass::all() {
                let weight = (a.combos() * b.combos()) as f64;
                total += weight;
                outcomes.push((
                    State {
                        classes: [a.index() as u8, b.index() as u8],
                        node: Node::SmallBlind,
                    },
                    weight,
                ));
            }
        }
        for outcome in &mut outcomes {
            outcome.1 /= total;
        }
        outcomes
    }

    fn sample_chance(&self, _state: &State, rng: &mut Rng) -> State {
        // Deal four distinct cards, which gets card removal exactly right and
        // costs nothing compared with building 28,561 weighted outcomes.
        let mut drawn = [Card::all().next().expect("non-empty deck"); 4];
        let mut dead = CardSet::empty();
        for slot in drawn.iter_mut() {
            loop {
                let card = Card::from_index(rng.below(52) as u8).expect("0..52");
                if dead.insert(card) {
                    *slot = card;
                    break;
                }
            }
        }
        State {
            classes: [
                HandClass::from_cards(drawn[0], drawn[1]).index() as u8,
                HandClass::from_cards(drawn[2], drawn[3]).index() as u8,
            ],
            node: Node::SmallBlind,
        }
    }

    fn current_player(&self, state: &State) -> usize {
        match state.node {
            Node::SmallBlind => 0,
            Node::BigBlind => 1,
            other => unreachable!("{other:?} is not a decision node"),
        }
    }

    fn info_key(&self, state: &State) -> InfoKey {
        let player = self.current_player(state);
        PushFold::info_key(player, state.classes[player] as usize)
    }

    fn num_actions(&self, _state: &State) -> usize {
        2
    }

    fn apply(&self, state: &State, action: usize) -> State {
        let node = match (state.node, action) {
            (Node::SmallBlind, FOLD) => Node::SmallBlindFolded,
            (Node::SmallBlind, PUSH) => Node::BigBlind,
            (Node::BigBlind, FOLD) => Node::BigBlindFolded,
            (Node::BigBlind, PUSH) => Node::Showdown,
            (other, action) => unreachable!("action {action} at {other:?}"),
        };
        State { node, ..*state }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfr::Solver;

    /// Low sample counts keep the suite quick; push/fold thresholds are not
    /// sensitive to a percent of equity error.
    const EQUITY_SAMPLES: u32 = 300;
    const TRAINING_ITERATIONS: usize = 300_000;

    fn table() -> EquityTable {
        let mut rng = Rng::new(0x51DE);
        EquityTable::sampled(EQUITY_SAMPLES, &mut rng)
    }

    fn solve(stack: f64, equity: EquityTable) -> Solver<PushFold> {
        let mut rng = Rng::new(0xF01D);
        let mut solver = Solver::new(PushFold::new(stack, equity));
        solver.train_sampled(TRAINING_ITERATIONS, &mut rng);
        solver
    }

    fn class(text: &str) -> HandClass {
        text.parse().expect("valid hand class")
    }

    /// How often `player` takes the aggressive action with `hand`.
    fn frequency(solver: &Solver<PushFold>, player: usize, hand: &str) -> f64 {
        let key = PushFold::info_key(player, class(hand).index());
        solver
            .average_strategy(key)
            .unwrap_or_else(|| panic!("{hand} for player {player} was never visited"))[PUSH]
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

    #[test]
    fn equity_table_is_symmetric_and_sane() {
        let table = table();

        for (a, b) in [("AA", "KK"), ("AKs", "22"), ("72o", "AA")] {
            let (x, y) = (class(a), class(b));
            let forward = table.get(x, y);
            let backward = table.get(y, x);
            assert!(
                (forward + backward - 1.0).abs() < 1e-6,
                "{a} vs {b}: {forward} + {backward}"
            );
        }

        // A class against itself is symmetric, so exactly half.
        assert_eq!(table.get(class("AA"), class("AA")), 0.5);

        // Sanity against known matchups, loose enough for 300 samples.
        assert!(table.get(class("AA"), class("KK")) > 0.75);
        assert!(table.get(class("AA"), class("72o")) > 0.80);
        assert!((table.get(class("AKs"), class("22")) - 0.5).abs() < 0.10);
    }

    #[test]
    fn parallel_building_does_not_depend_on_the_thread_count() {
        // Seeding per pairing rather than per worker is what makes this hold.
        // If it failed, every solve would be irreproducible across machines
        // with different core counts.
        let one = EquityTable::sampled_parallel(120, 0xABCD, 1);
        let many = EquityTable::sampled_parallel(120, 0xABCD, 4);
        for a in HandClass::all() {
            for b in HandClass::all() {
                assert_eq!(one.get(a, b), many.get(a, b), "{a} vs {b}");
            }
        }
    }

    #[test]
    fn a_table_round_trips_through_disk() {
        let original = EquityTable::sampled_parallel(120, 0x1234, 2);
        let path = std::env::temp_dir().join("poker_core_equity_roundtrip.bin");

        original.save(&path).expect("save should succeed");
        let loaded = EquityTable::load(&path).expect("load should succeed");

        for a in HandClass::all() {
            for b in HandClass::all() {
                assert_eq!(original.get(a, b), loaded.get(a, b), "{a} vs {b}");
            }
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn loading_a_corrupt_cache_is_an_error_not_a_wrong_table() {
        let path = std::env::temp_dir().join("poker_core_equity_corrupt.bin");
        std::fs::write(&path, b"not an equity table at all").expect("write");

        let result = EquityTable::load(&path);
        assert!(result.is_err(), "garbage must not load as a valid table");

        // load_or_build treats a bad cache as absent and rebuilds it.
        let rebuilt = EquityTable::load_or_build(&path, 60, 0x99, 2).expect("rebuild");
        assert!((rebuilt.get(class("AA"), class("72o")) - 0.85).abs() < 0.10);
        assert!(EquityTable::load(&path).is_ok(), "cache was repaired");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn equity_ranks_hands_in_the_expected_order() {
        // Against a fixed weak hand, stronger holdings must show more equity.
        // This is the property sampling noise was breaking in the solved charts.
        //
        // Choosing the rungs is harder than it looks, and the constraints are
        // all real poker, not test hygiene:
        //
        // 1. No rung shares a rank with the villain. Sharing one creates
        //    domination, which reorders equities independently of hand
        //    strength — see the test below.
        // 2. No two rungs sit in the same structural matchup. *Every* pair
        //    against two undercards runs about 87%, whether it is aces or
        //    tens, because what beats it is the undercards improving rather
        //    than the pair's rank. AA over TT here is under a point — far
        //    below the sampling error, so it is unorderable in a unit test.
        //
        // What remains are three genuinely different matchups roughly twenty
        // points apart: a pair over undercards, two overcards, and a hand
        // playing catch-up.
        let table = EquityTable::sampled_parallel(5_000, 0x5EED, 4);
        let villain = class("72o");

        let ladder = ["AA", "AKs", "54s"];
        for pair in ladder.windows(2) {
            let (stronger, weaker) = (class(pair[0]), class(pair[1]));
            assert!(
                table.get(stronger, villain) > table.get(weaker, villain),
                "{} ({:.4}) should beat {} ({:.4}) against 72o",
                pair[0],
                table.get(stronger, villain),
                pair[1],
                table.get(weaker, villain),
            );
        }
    }

    #[test]
    fn domination_outweighs_raw_hand_strength() {
        // 87s is a far weaker holding than AKs, yet it is the bigger favourite
        // against 72o: the seven ties the villain's seven while the eight
        // outkicks the deuce, so the villain is drawing thin. AKs merely holds
        // two overcards and has to improve.
        //
        // Equity is a property of the matchup, not of a hand in isolation. A
        // solver that ranked hands on a single strength axis would misprice
        // every dominated spot, so this is asserted rather than assumed.
        let table = EquityTable::sampled_parallel(5_000, 0x5EED, 4);
        let villain = class("72o");

        let dominating = table.get(class("87s"), villain);
        let overcards = table.get(class("AKs"), villain);
        assert!(
            dominating > overcards,
            "87s ({dominating:.4}) should beat AKs ({overcards:.4}) against 72o"
        );

        // The same holding loses its edge once the shared rank is gone.
        let no_overlap = table.get(class("87s"), class("A3o"));
        assert!(
            no_overlap < dominating,
            "87s is only this strong because it dominates: {no_overlap:.4} vs {dominating:.4}"
        );
    }

    #[test]
    fn the_tree_has_the_expected_shape() {
        let game = PushFold::new(10.0, table());
        let root = game.initial();
        assert!(game.is_chance(&root));

        let mut rng = Rng::new(1);
        let dealt = game.sample_chance(&root, &mut rng);
        assert!(!game.is_chance(&dealt));
        assert_eq!(game.current_player(&dealt), 0, "small blind acts first");

        let folded = game.apply(&dealt, FOLD);
        assert!(game.is_terminal(&folded));
        assert_eq!(game.terminal_utility(&folded), -0.5, "loses the posted blind");

        let pushed = game.apply(&dealt, PUSH);
        assert!(!game.is_terminal(&pushed));
        assert_eq!(game.current_player(&pushed), 1, "big blind responds");

        let taken = game.apply(&pushed, FOLD);
        assert!(game.is_terminal(&taken));
        assert_eq!(game.terminal_utility(&taken), 1.0, "wins the big blind");

        assert!(game.is_terminal(&game.apply(&pushed, PUSH)));
    }

    #[test]
    fn showdown_utility_scales_with_the_stack() {
        // A coin flip is worth nothing; a favourite's edge grows with what is
        // at risk.
        let equity = table();
        for stack in [5.0, 20.0] {
            let game = PushFold::new(stack, equity.clone());
            let state = State {
                classes: [class("AA").index() as u8, class("72o").index() as u8],
                node: Node::Showdown,
            };
            let utility = game.terminal_utility(&state);
            assert!(utility > 0.5 * stack, "AA at {stack}bb won only {utility}");
            assert!(utility < stack, "cannot exceed the stack");
        }
    }

    #[test]
    fn dealing_never_produces_an_impossible_hand() {
        let game = PushFold::new(10.0, table());
        let mut rng = Rng::new(99);
        let root = game.initial();
        for _ in 0..10_000 {
            let state = game.sample_chance(&root, &mut rng);
            for player in 0..2 {
                assert!((state.classes[player] as usize) < NUM_HAND_CLASSES);
            }
        }
    }

    #[test]
    fn chance_outcomes_form_a_distribution() {
        let game = PushFold::new(10.0, table());
        let outcomes = game.chance_outcomes(&game.initial());
        assert_eq!(outcomes.len(), NUM_HAND_CLASSES * NUM_HAND_CLASSES);
        let total: f64 = outcomes.iter().map(|(_, p)| p).sum();
        assert!((total - 1.0).abs() < 1e-9, "weights summed to {total}");
    }

    #[test]
    fn premium_hands_always_get_the_money_in() {
        let solver = solve(10.0, table());
        assert!(frequency(&solver, 0, "AA") > 0.97, "the small blind always pushes aces");
        assert!(frequency(&solver, 1, "AA") > 0.97, "the big blind always calls with aces");
        assert!(frequency(&solver, 0, "KK") > 0.97);
    }

    #[test]
    fn the_push_range_widens_as_stacks_get_shorter() {
        // The core dynamic of push/fold: with less behind, the blinds are
        // worth relatively more and risking the stack costs relatively less.
        let equity = table();
        let short = range_width(&solve(4.0, equity.clone()), 0);
        let deep = range_width(&solve(18.0, equity), 0);
        assert!(
            short > deep + 0.10,
            "expected a wider range at 4bb ({short:.3}) than at 18bb ({deep:.3})"
        );
    }

    #[test]
    fn deep_stack_push_ranges_are_contained_in_short_stack_ones() {
        // Nash push ranges nest: every hand worth jamming with more behind is
        // still worth jamming with less. Asserted across all 169 classes rather
        // than on one hand, because any single marginal holding sits near
        // indifference and its exact frequency is dominated by solver noise.
        let equity = table();
        let short = solve(5.0, equity.clone());
        let deep = solve(20.0, equity);

        for hand in HandClass::all() {
            let text = hand.to_string();
            let (near, far) = (
                frequency(&short, 0, &text),
                frequency(&deep, 0, &text),
            );
            assert!(
                near >= far - 0.15,
                "{text} pushes {far:.3} at 20bb but only {near:.3} at 5bb"
            );
        }
    }

    #[test]
    fn premium_hands_are_pushed_at_every_depth_and_trash_only_when_short() {
        let equity = table();
        for stack in [5.0, 20.0] {
            let solver = solve(stack, equity.clone());
            assert!(
                frequency(&solver, 0, "AA") > 0.95,
                "aces must jam at {stack}bb"
            );
            assert!(
                frequency(&solver, 0, "72o") < 0.30,
                "72o is never a standard jam, even at {stack}bb"
            );
        }
    }

    #[test]
    fn the_caller_is_tighter_than_the_pusher() {
        // The pusher wins the pot whenever the big blind folds, so it can
        // profitably push hands the big blind cannot profitably call.
        let solver = solve(10.0, table());
        let pushing = range_width(&solver, 0);
        let calling = range_width(&solver, 1);
        assert!(
            pushing > calling + 0.05,
            "push {pushing:.3} should exceed call {calling:.3}"
        );
    }

    #[test]
    fn the_strategy_is_close_to_unexploitable() {
        let solver = solve(10.0, table());
        let exploitability = solver.exploitability(&solver.profile());
        assert!(
            exploitability < 0.05,
            "exploitable for {exploitability} big blinds per hand"
        );
    }

    #[test]
    fn every_hand_class_gets_a_strategy_for_both_players() {
        let solver = solve(10.0, table());
        for hand in HandClass::all() {
            for player in 0..2 {
                let key = PushFold::info_key(player, hand.index());
                let strategy = solver
                    .average_strategy(key)
                    .unwrap_or_else(|| panic!("{hand} missing for player {player}"));
                let total: f64 = strategy.iter().sum();
                assert!((total - 1.0).abs() < 1e-9, "{hand} sums to {total}");
            }
        }
        assert_eq!(solver.info_set_count(), 2 * NUM_HAND_CLASSES);
    }
}
