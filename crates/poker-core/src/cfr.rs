//! Counterfactual Regret Minimization for two-player zero-sum games.
//!
//! CFR converges to a Nash equilibrium by accumulating, at every information
//! set, the regret for not having played each action. Regret matching turns
//! those regrets into a strategy, and it is the *average* strategy over all
//! iterations — not the final one — that converges.
//!
//! # Why the game is abstract
//!
//! [`Game`] is a trait rather than hard-coded poker so the identical solver can
//! be validated on a game with a known analytical solution before being pointed
//! at No-Limit Hold'em. A sign error in the regret update or a missing
//! counterfactual weight produces a strategy that looks entirely plausible in
//! NLHE while quietly losing money. On Kuhn poker the same bug is a visible
//! numerical mismatch within seconds.
//!
//! # Conventions
//!
//! Utilities are always stated from **player 0's** point of view; player 1
//! receives the negation. That is what makes the game zero-sum, and it keeps
//! one sign convention in one place instead of scattered through the recursion.

use crate::rng::Rng;
use std::collections::HashMap;

/// Identifies an information set: everything the acting player knows.
///
/// Two histories sharing a key are indistinguishable to the player to act, so
/// they must be played identically.
pub type InfoKey = u64;

/// A two-player zero-sum extensive-form game with chance nodes.
pub trait Game {
    /// A node in the game tree. Cheap to clone — the solver clones constantly.
    type State: Clone;

    /// The root, normally a chance node that deals.
    fn initial(&self) -> Self::State;

    fn is_terminal(&self, state: &Self::State) -> bool;

    /// Payoff to player 0 at a terminal node. Player 1 receives the negation.
    fn terminal_utility(&self, state: &Self::State) -> f64;

    /// Whether chance acts here rather than a player.
    fn is_chance(&self, state: &Self::State) -> bool;

    /// Successor states and their probabilities. Probabilities must sum to 1.
    fn chance_outcomes(&self, state: &Self::State) -> Vec<(Self::State, f64)>;

    /// The player to act at a decision node.
    fn current_player(&self, state: &Self::State) -> usize;

    /// The acting player's information set.
    fn info_key(&self, state: &Self::State) -> InfoKey;

    /// How many actions are legal here.
    fn num_actions(&self, state: &Self::State) -> usize;

    /// The state after taking `action`, an index below [`Game::num_actions`].
    fn apply(&self, state: &Self::State, action: usize) -> Self::State;

    /// Draws a single chance outcome.
    ///
    /// The default walks [`Game::chance_outcomes`], which is correct but builds
    /// the whole distribution each visit. Games whose chance space is large —
    /// dealing from a 52-card deck, say — should override this to draw
    /// directly, since materialising every possible deal per iteration defeats
    /// the point of sampling.
    fn sample_chance(&self, state: &Self::State, rng: &mut Rng) -> Self::State {
        let outcomes = self.chance_outcomes(state);
        debug_assert!(!outcomes.is_empty(), "chance node with no outcomes");
        let roll = rng.next_f64();
        let mut cumulative = 0.0;
        for (next, probability) in &outcomes {
            cumulative += probability;
            if roll < cumulative {
                return next.clone();
            }
        }
        // Reached only through floating-point drift in the final comparison.
        outcomes
            .into_iter()
            .next_back()
            .expect("chance node with no outcomes")
            .0
    }
}

/// Converts regrets into a strategy: play each action in proportion to its
/// positive regret, or uniformly when no action has any.
fn regret_matching(regrets: &[f64]) -> Vec<f64> {
    let total: f64 = regrets.iter().filter(|r| **r > 0.0).sum();
    if total > 0.0 {
        regrets.iter().map(|r| r.max(0.0) / total).collect()
    } else {
        vec![1.0 / regrets.len() as f64; regrets.len()]
    }
}

/// Accumulated regrets and strategy weights for one information set.
#[derive(Debug, Clone)]
struct Node {
    regret_sum: Vec<f64>,
    strategy_sum: Vec<f64>,
}

impl Node {
    fn new(actions: usize) -> Node {
        Node {
            regret_sum: vec![0.0; actions],
            strategy_sum: vec![0.0; actions],
        }
    }

    fn current_strategy(&self) -> Vec<f64> {
        regret_matching(&self.regret_sum)
    }

    /// The average strategy, which is what converges to equilibrium.
    fn average_strategy(&self) -> Vec<f64> {
        let total: f64 = self.strategy_sum.iter().sum();
        if total > 0.0 {
            self.strategy_sum.iter().map(|s| s / total).collect()
        } else {
            vec![1.0 / self.strategy_sum.len() as f64; self.strategy_sum.len()]
        }
    }
}

/// A strategy profile: the probability of each action at each information set.
pub type Profile = HashMap<InfoKey, Vec<f64>>;

/// How regrets accumulate and strategies are averaged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Discount {
    /// Classic CFR. Regrets accumulate without a floor and every iteration
    /// contributes equally to the average strategy.
    Vanilla,
    /// CFR+. Two changes, both aimed at the same problem — early mistakes
    /// lingering far longer than they deserve to.
    ///
    /// Cumulative regret is floored at zero, so an action that was briefly
    /// terrible can be reconsidered as soon as it becomes good, rather than
    /// waiting to climb out of a deep negative hole. And the average strategy
    /// is weighted linearly by iteration, so the well-trained later iterations
    /// dominate the near-random early ones.
    ///
    /// Converges substantially faster in practice and is what every serious
    /// modern solver uses.
    #[default]
    Plus,
}

/// The update rule in force for one iteration.
#[derive(Debug, Clone, Copy)]
struct Update {
    floor_regrets: bool,
    /// Weight this iteration contributes to the average strategy.
    strategy_weight: f64,
}

impl Update {
    fn for_iteration(discount: Discount, iteration: u64) -> Update {
        match discount {
            Discount::Vanilla => Update {
                floor_regrets: false,
                strategy_weight: 1.0,
            },
            Discount::Plus => Update {
                floor_regrets: true,
                strategy_weight: iteration as f64,
            },
        }
    }

    /// Applies `delta` to a cumulative regret under this rule.
    #[inline]
    fn accumulate(&self, regret: &mut f64, delta: f64) {
        *regret += delta;
        if self.floor_regrets && *regret < 0.0 {
            *regret = 0.0;
        }
    }
}

/// Trains a strategy for a [`Game`] by running CFR.
#[derive(Debug)]
pub struct Solver<G: Game> {
    game: G,
    nodes: HashMap<InfoKey, Node>,
    discount: Discount,
    /// Iterations run so far, used for linear strategy weighting.
    iteration: u64,
}

impl<G: Game> Solver<G> {
    /// A solver using [`Discount::Plus`], the faster default.
    pub fn new(game: G) -> Solver<G> {
        Solver {
            game,
            nodes: HashMap::new(),
            discount: Discount::default(),
            iteration: 0,
        }
    }

    /// Chooses the update rule. Use [`Discount::Vanilla`] for textbook CFR.
    pub fn with_discount(mut self, discount: Discount) -> Solver<G> {
        self.discount = discount;
        self
    }

    /// Iterations run so far.
    pub fn iterations(&self) -> u64 {
        self.iteration
    }

    pub fn game(&self) -> &G {
        &self.game
    }

    /// Number of information sets discovered so far.
    pub fn info_set_count(&self) -> usize {
        self.nodes.len()
    }

    /// Runs `iterations` full CFR traversals of the game tree.
    ///
    /// This is vanilla CFR: it walks the entire tree every iteration, which is
    /// exact and deterministic. That is what makes it the right tool for
    /// validating correctness; sampling variants trade exactness for reach on
    /// large games.
    pub fn train(&mut self, iterations: usize) {
        let root = self.game.initial();
        for _ in 0..iterations {
            self.iteration += 1;
            let update = Update::for_iteration(self.discount, self.iteration);
            walk(&self.game, &mut self.nodes, &root, [1.0, 1.0], 1.0, update);
        }
    }

    /// Runs `iterations` of external-sampling MCCFR.
    ///
    /// Each iteration walks the tree twice, once per player. For the player
    /// being updated every action is explored; for the opponent and for chance,
    /// a single outcome is sampled. Cost per iteration is therefore
    /// proportional to the traverser's own decisions rather than to the whole
    /// tree, which is what makes games the size of No-Limit Hold'em reachable.
    ///
    /// The trade is variance for reach: individual iterations are noisy, but
    /// the estimator is unbiased, so the average strategy converges to the same
    /// equilibrium [`Solver::train`] finds — just via many more, much cheaper,
    /// iterations.
    pub fn train_sampled(&mut self, iterations: usize, rng: &mut Rng) {
        let root = self.game.initial();
        for _ in 0..iterations {
            self.iteration += 1;
            let update = Update::for_iteration(self.discount, self.iteration);
            // Alternate which player's regrets are being updated.
            for traverser in 0..2 {
                sample_walk(&self.game, &mut self.nodes, &root, traverser, rng, update);
            }
        }
    }

    /// The average strategy at one information set, if it has been visited.
    pub fn average_strategy(&self, key: InfoKey) -> Option<Vec<f64>> {
        self.nodes.get(&key).map(Node::average_strategy)
    }

    /// The full average strategy profile.
    pub fn profile(&self) -> Profile {
        self.nodes
            .iter()
            .map(|(&key, node)| (key, node.average_strategy()))
            .collect()
    }

    /// Expected payoff to player 0 when both sides play `profile`.
    pub fn expected_value(&self, profile: &Profile) -> f64 {
        expected_value(&self.game, profile, &self.game.initial())
    }

    /// How much a best-responding opponent could gain over the game value,
    /// averaged across the two players.
    ///
    /// Zero means `profile` is an exact Nash equilibrium; the value is in the
    /// same units as [`Game::terminal_utility`]. This is the honest measure of
    /// solution quality — a strategy can look sensible and still be wildly
    /// exploitable.
    pub fn exploitability(&self, profile: &Profile) -> f64 {
        let br0 = best_response_value(&self.game, profile, 0);
        let br1 = best_response_value(&self.game, profile, 1);
        (br0 + br1) / 2.0
    }
}

/// One CFR traversal, returning the expected utility to player 0.
///
/// `reach` holds each player's probability of playing to this node, and
/// `chance_reach` the probability chance did. A player's regret is weighted by
/// the *counterfactual* reach — everyone else's contribution but their own —
/// which is what makes the regret independent of how likely they were to get
/// here.
fn walk<G: Game>(
    game: &G,
    nodes: &mut HashMap<InfoKey, Node>,
    state: &G::State,
    reach: [f64; 2],
    chance_reach: f64,
    update: Update,
) -> f64 {
    if game.is_terminal(state) {
        return game.terminal_utility(state);
    }

    if game.is_chance(state) {
        return game
            .chance_outcomes(state)
            .into_iter()
            .map(|(next, probability)| {
                probability * walk(game, nodes, &next, reach, chance_reach * probability, update)
            })
            .sum();
    }

    let player = game.current_player(state);
    let actions = game.num_actions(state);
    let key = game.info_key(state);

    let strategy = nodes
        .entry(key)
        .or_insert_with(|| Node::new(actions))
        .current_strategy();

    // Utility of each action, and of the node under the current strategy.
    let mut action_utility = vec![0.0; actions];
    let mut node_utility = 0.0;
    for (action, utility) in action_utility.iter_mut().enumerate() {
        let next = game.apply(state, action);
        let mut next_reach = reach;
        next_reach[player] *= strategy[action];
        *utility = walk(game, nodes, &next, next_reach, chance_reach, update);
        node_utility += strategy[action] * *utility;
    }

    // Everyone's reach but this player's.
    let counterfactual = chance_reach * reach[1 - player];
    let node = nodes.get_mut(&key).expect("inserted above");
    for action in 0..actions {
        // Utilities are player 0's, so player 1's regret is the negation:
        // player 1 gains exactly what player 0 loses.
        let gain = action_utility[action] - node_utility;
        let regret = if player == 0 { gain } else { -gain };
        update.accumulate(&mut node.regret_sum[action], counterfactual * regret);

        // The average strategy is weighted by this player's own reach, and by
        // how much this iteration is worth under the discount rule.
        node.strategy_sum[action] +=
            update.strategy_weight * reach[player] * strategy[action];
    }

    node_utility
}

/// One external-sampling traversal, returning the value to `traverser`.
///
/// Utilities are converted to the traverser's perspective on entry, so the
/// regret arithmetic below needs no sign handling — unlike [`walk`], which
/// carries player 0's perspective throughout.
fn sample_walk<G: Game>(
    game: &G,
    nodes: &mut HashMap<InfoKey, Node>,
    state: &G::State,
    traverser: usize,
    rng: &mut Rng,
    update: Update,
) -> f64 {
    if game.is_terminal(state) {
        let utility = game.terminal_utility(state);
        return if traverser == 0 { utility } else { -utility };
    }

    if game.is_chance(state) {
        let next = game.sample_chance(state, rng);
        return sample_walk(game, nodes, &next, traverser, rng, update);
    }

    let player = game.current_player(state);
    let actions = game.num_actions(state);
    let key = game.info_key(state);
    let strategy = nodes
        .entry(key)
        .or_insert_with(|| Node::new(actions))
        .current_strategy();

    if player != traverser {
        // The opponent plays one sampled action. Their average strategy is
        // accumulated here, on the visits where they are not being updated.
        let node = nodes.get_mut(&key).expect("inserted above");
        for (action, probability) in strategy.iter().enumerate() {
            node.strategy_sum[action] += update.strategy_weight * probability;
        }
        let sampled = sample_action(&strategy, rng);
        return sample_walk(
            game,
            nodes,
            &game.apply(state, sampled),
            traverser,
            rng,
            update,
        );
    }

    // The traverser explores every action, which is what keeps the regret
    // estimate unbiased despite sampling everywhere else.
    let mut action_value = vec![0.0; actions];
    let mut node_value = 0.0;
    for (action, value) in action_value.iter_mut().enumerate() {
        *value = sample_walk(
            game,
            nodes,
            &game.apply(state, action),
            traverser,
            rng,
            update,
        );
        node_value += strategy[action] * *value;
    }

    let node = nodes.get_mut(&key).expect("inserted above");
    for (action, value) in action_value.iter().enumerate() {
        update.accumulate(&mut node.regret_sum[action], value - node_value);
    }

    node_value
}

/// Draws an action index from a strategy distribution.
fn sample_action(strategy: &[f64], rng: &mut Rng) -> usize {
    let roll = rng.next_f64();
    let mut cumulative = 0.0;
    for (action, probability) in strategy.iter().enumerate() {
        cumulative += probability;
        if roll < cumulative {
            return action;
        }
    }
    strategy.len() - 1
}

/// Expected payoff to player 0 when both players follow `profile`.
fn expected_value<G: Game>(game: &G, profile: &Profile, state: &G::State) -> f64 {
    if game.is_terminal(state) {
        return game.terminal_utility(state);
    }
    if game.is_chance(state) {
        return game
            .chance_outcomes(state)
            .into_iter()
            .map(|(next, p)| p * expected_value(game, profile, &next))
            .sum();
    }

    let actions = game.num_actions(state);
    let key = game.info_key(state);
    let strategy = strategy_for(profile, key, actions);

    (0..actions)
        .map(|a| strategy[a] * expected_value(game, profile, &game.apply(state, a)))
        .sum()
}

/// The profile's strategy at `key`, defaulting to uniform for unvisited sets.
fn strategy_for(profile: &Profile, key: InfoKey, actions: usize) -> Vec<f64> {
    profile
        .get(&key)
        .filter(|s| s.len() == actions)
        .cloned()
        .unwrap_or_else(|| vec![1.0 / actions as f64; actions])
}

/// The best payoff `br_player` can achieve against `profile`.
///
/// A best response must play identically across every history in an
/// information set — it cannot peek at the opponent's cards. So the choice at
/// an information set is made once, over the counterfactual-weighted sum of its
/// histories, rather than independently at each node.
///
/// Information sets are decided deepest-first. With perfect recall every set
/// reachable below a decision is strictly deeper than it, so by the time a set
/// is decided everything beneath it is already fixed.
fn best_response_value<G: Game>(game: &G, profile: &Profile, br_player: usize) -> f64 {
    // Gather each of the best responder's information sets: the histories in
    // it, how likely everyone else was to produce them, and how deep it sits.
    let mut histories: HashMap<InfoKey, Vec<(G::State, f64)>> = HashMap::new();
    let mut depth_of: HashMap<InfoKey, usize> = HashMap::new();
    collect(
        game,
        profile,
        br_player,
        &game.initial(),
        1.0,
        0,
        &mut histories,
        &mut depth_of,
    );

    let mut order: Vec<InfoKey> = depth_of.keys().copied().collect();
    order.sort_by_key(|key| std::cmp::Reverse(depth_of[key]));

    let mut choices: HashMap<InfoKey, usize> = HashMap::new();
    for key in order {
        let states = &histories[&key];
        let actions = states
            .first()
            .map(|(state, _)| game.num_actions(state))
            .unwrap_or(0);

        let mut best_action = 0;
        let mut best_value = f64::NEG_INFINITY;
        for action in 0..actions {
            let value: f64 = states
                .iter()
                .map(|(state, weight)| {
                    weight
                        * responder_value(
                            game,
                            profile,
                            br_player,
                            &game.apply(state, action),
                            &choices,
                        )
                })
                .sum();
            if value > best_value {
                best_value = value;
                best_action = action;
            }
        }
        choices.insert(key, best_action);
    }

    responder_value(game, profile, br_player, &game.initial(), &choices)
}

/// Walks the tree collecting the best responder's information sets, weighting
/// each history by the probability that chance and the opponent produced it.
#[allow(clippy::too_many_arguments)]
fn collect<G: Game>(
    game: &G,
    profile: &Profile,
    br_player: usize,
    state: &G::State,
    weight: f64,
    depth: usize,
    histories: &mut HashMap<InfoKey, Vec<(G::State, f64)>>,
    depth_of: &mut HashMap<InfoKey, usize>,
) {
    if game.is_terminal(state) || weight == 0.0 {
        return;
    }
    if game.is_chance(state) {
        for (next, p) in game.chance_outcomes(state) {
            collect(
                game,
                profile,
                br_player,
                &next,
                weight * p,
                depth + 1,
                histories,
                depth_of,
            );
        }
        return;
    }

    let actions = game.num_actions(state);
    if game.current_player(state) == br_player {
        let key = game.info_key(state);
        histories.entry(key).or_default().push((state.clone(), weight));
        // Keep the shallowest depth seen, so a set is only decided once
        // everything genuinely below it has been.
        depth_of
            .entry(key)
            .and_modify(|d| *d = (*d).min(depth))
            .or_insert(depth);
        // The responder's own probability is not part of the weight: it is
        // exactly what we are solving for.
        for action in 0..actions {
            collect(
                game,
                profile,
                br_player,
                &game.apply(state, action),
                weight,
                depth + 1,
                histories,
                depth_of,
            );
        }
    } else {
        let strategy = strategy_for(profile, game.info_key(state), actions);
        for (action, &probability) in strategy.iter().enumerate() {
            collect(
                game,
                profile,
                br_player,
                &game.apply(state, action),
                weight * probability,
                depth + 1,
                histories,
                depth_of,
            );
        }
    }
}

/// Value to `br_player` when they follow `choices` and the opponent follows
/// `profile`. Information sets without a choice yet fall back to the profile.
fn responder_value<G: Game>(
    game: &G,
    profile: &Profile,
    br_player: usize,
    state: &G::State,
    choices: &HashMap<InfoKey, usize>,
) -> f64 {
    if game.is_terminal(state) {
        let utility = game.terminal_utility(state);
        return if br_player == 0 { utility } else { -utility };
    }
    if game.is_chance(state) {
        return game
            .chance_outcomes(state)
            .into_iter()
            .map(|(next, p)| p * responder_value(game, profile, br_player, &next, choices))
            .sum();
    }

    let actions = game.num_actions(state);
    let key = game.info_key(state);

    if game.current_player(state) == br_player {
        if let Some(&action) = choices.get(&key) {
            return responder_value(game, profile, br_player, &game.apply(state, action), choices);
        }
    }

    let strategy = strategy_for(profile, key, actions);
    (0..actions)
        .map(|a| {
            strategy[a]
                * responder_value(game, profile, br_player, &game.apply(state, a), choices)
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regret_matching_is_proportional_to_positive_regret() {
        let strategy = regret_matching(&[3.0, 1.0, 0.0]);
        assert!((strategy[0] - 0.75).abs() < 1e-12);
        assert!((strategy[1] - 0.25).abs() < 1e-12);
        assert_eq!(strategy[2], 0.0);
    }

    #[test]
    fn regret_matching_ignores_negative_regret() {
        let strategy = regret_matching(&[5.0, -100.0]);
        assert_eq!(strategy, vec![1.0, 0.0]);
    }

    #[test]
    fn regret_matching_falls_back_to_uniform() {
        // No action has positive regret, so nothing is preferred yet.
        let strategy = regret_matching(&[0.0, 0.0, 0.0]);
        assert_eq!(strategy, vec![1.0 / 3.0; 3]);
        let all_negative = regret_matching(&[-1.0, -2.0]);
        assert_eq!(all_negative, vec![0.5, 0.5]);
    }
}
