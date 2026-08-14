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

/// Fold, for either player.
pub const FOLD: usize = 0;
/// Push for the small blind, call for the big blind.
pub const PUSH: usize = 1;

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

    /// Class `a`'s equity against class `b`.
    #[inline]
    pub fn get(&self, a: HandClass, b: HandClass) -> f64 {
        self.table[a.index() * NUM_HAND_CLASSES + b.index()] as f64
    }
}

/// Draws concrete cards for `class` that avoid `blocked`.
///
/// Returns `None` if no conflict-free draw turned up, which only happens for
/// heavily blocked classes such as a pair whose rank is already exhausted.
fn sample_hand(class: HandClass, rng: &mut Rng, blocked: CardSet) -> Option<[Card; 2]> {
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
    fn trash_is_playable_when_short_and_not_when_deep() {
        let equity = table();
        let short = frequency(&solve(3.0, equity.clone()), 0, "72o");
        let deep = frequency(&solve(20.0, equity), 0, "72o");
        assert!(short > deep, "72o: {short:.3} at 3bb vs {deep:.3} at 20bb");
        assert!(deep < 0.15, "72o should mostly fold at 20bb, got {deep:.3}");
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
