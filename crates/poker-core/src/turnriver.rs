//! Two-street postflop: turn betting, a river card, then river betting.
//!
//! This is the first game here with a chance node *inside* it. Everything
//! before had chance only at the root — deal, then play. Here the river arrives
//! mid-hand and re-ranks every holding underneath the strategy: the best hand
//! on the turn is often not the best hand on the river, and a solver has to
//! plan for that rather than react to it.
//!
//! # What the river card does to information sets
//!
//! A player's river decision depends on their cards, the river card, *and* how
//! the turn was played — because the turn line determines the pot, and the pot
//! determines the price. Checking through to a 20 bb pot is a different game
//! from betting and calling into a 60 bb one, even holding the same cards on
//! the same river. All three go into the information-set key, and the tests
//! check they cannot collide.
//!
//! # Scope
//!
//! One bet size per street, fold-call-bet only. Raises are omitted here — the
//! pattern is already proven in [`crate::river`], and adding them multiplies
//! the tree without exercising anything new about multi-street play. Ranges are
//! concrete combinations rather than buckets, which is exact and affordable for
//! a single subgame; bucketing is what a full-game solve needs.

use crate::card::{Card, CardSet};
use crate::cfr::{Game, InfoKey};
use crate::eval::evaluate;
use crate::rng::Rng;
use std::fmt;

/// Check when nothing is owed, fold when facing a bet. Always action 0.
pub const PASSIVE: usize = 0;
/// Bet when nothing is owed, call when facing one. Always action 1.
pub const AGGRESSIVE: usize = 1;

/// How the turn was played, which fixes the pot going to the river.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnLine {
    /// Checked through.
    CheckedThrough = 0,
    /// Out of position bet, in position called.
    BetCalled = 1,
    /// Checked to, in position bet, out of position called.
    CheckRaisedCalled = 2,
}

impl TurnLine {
    const ALL: [TurnLine; 3] = [
        TurnLine::CheckedThrough,
        TurnLine::BetCalled,
        TurnLine::CheckRaisedCalled,
    ];
}

/// Where a hand stands in the two-street tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// Hands not yet dealt.
    Deal,
    TurnOopFirst,
    TurnIpVsCheck,
    TurnOopVsBet,
    TurnIpVsBet,
    /// The river card is about to arrive.
    RiverDeal,
    RiverOopFirst,
    RiverIpVsCheck,
    RiverOopVsBet,
    RiverIpVsBet,
    /// Someone folded; the index is who.
    Folded(u8),
    Showdown,
}

/// Decision stages. Must fit in [`STAGE_BITS`].
const NUM_STAGES: usize = 8;
const STAGE_BITS: u32 = 4;
const LINE_BITS: u32 = 2;
const CARD_BITS: u32 = 6;
/// Marks "no river card yet" in an information-set key.
const NO_RIVER: u64 = 63;

const _: () = assert!(
    NUM_STAGES <= 1 << STAGE_BITS,
    "STAGE_BITS is too narrow for NUM_STAGES"
);
const _: () = assert!(
    TurnLine::ALL.len() <= 1 << LINE_BITS,
    "LINE_BITS is too narrow for the turn lines"
);

impl Stage {
    fn decision_index(self) -> Option<usize> {
        Some(match self {
            Stage::TurnOopFirst => 0,
            Stage::TurnIpVsCheck => 1,
            Stage::TurnOopVsBet => 2,
            Stage::TurnIpVsBet => 3,
            Stage::RiverOopFirst => 4,
            Stage::RiverIpVsCheck => 5,
            Stage::RiverOopVsBet => 6,
            Stage::RiverIpVsBet => 7,
            _ => return None,
        })
    }

    fn actor(self) -> Option<usize> {
        Some(match self {
            Stage::TurnOopFirst
            | Stage::TurnOopVsBet
            | Stage::RiverOopFirst
            | Stage::RiverOopVsBet => 0,
            Stage::TurnIpVsCheck
            | Stage::TurnIpVsBet
            | Stage::RiverIpVsCheck
            | Stage::RiverIpVsBet => 1,
            _ => return None,
        })
    }

    /// Whether the actor faces a wager, and so folds or calls rather than
    /// checks or bets.
    fn faces_bet(self) -> bool {
        matches!(
            self,
            Stage::TurnOopVsBet
                | Stage::TurnIpVsBet
                | Stage::RiverOopVsBet
                | Stage::RiverIpVsBet
        )
    }

    fn is_river(self) -> bool {
        matches!(
            self,
            Stage::RiverOopFirst
                | Stage::RiverIpVsCheck
                | Stage::RiverOopVsBet
                | Stage::RiverIpVsBet
        )
    }
}

/// A node in the two-street tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct State {
    hands: [u16; 2],
    /// Card index of the river, or [`u8::MAX`] before it is dealt.
    river: u8,
    line: u8,
    stage: Stage,
    /// Chips each player has put in beyond the starting pot, in hundredths.
    committed: [u32; 2],
}

impl State {
    pub fn stage(&self) -> Stage {
        self.stage
    }

    /// The river card, once dealt.
    pub fn river(&self) -> Option<Card> {
        Card::from_index(self.river)
    }

    /// How the turn was played, once the river is reached.
    pub fn line(&self) -> TurnLine {
        TurnLine::ALL[self.line as usize]
    }

    /// Chips `player` has committed beyond the starting pot, in big blinds.
    pub fn committed(&self, player: usize) -> f64 {
        to_blinds(self.committed[player])
    }
}

const SCALE: f64 = 100.0;

fn to_chips(blinds: f64) -> u32 {
    (blinds * SCALE).round() as u32
}

fn to_blinds(chips: u32) -> f64 {
    chips as f64 / SCALE
}

/// A turn-and-river subgame.
#[derive(Debug, Clone)]
pub struct TurnRiver {
    board: Vec<Card>,
    pot: u32,
    stack: u32,
    turn_bet: u32,
    /// River bet size per turn line, since the pot differs by line.
    river_bet: [u32; 3],
    ranges: [Vec<[Card; 2]>; 2],
}

impl TurnRiver {
    /// Builds a subgame on a four-card turn board.
    ///
    /// Bet sizes are fractions of the pot at the time — so `river_fraction`
    /// prices off the pot the turn line actually produced, not the pot the
    /// hand started with.
    ///
    /// # Panics
    /// Panics if the board is not four distinct cards, if either range is
    /// empty, if a range holding collides with the board, or if the pot or
    /// stack is not positive.
    pub fn new(
        board: &[Card],
        pot: f64,
        stack: f64,
        turn_fraction: f64,
        river_fraction: f64,
        oop: Vec<[Card; 2]>,
        ip: Vec<[Card; 2]>,
    ) -> TurnRiver {
        assert_eq!(board.len(), 4, "a turn board is four cards");
        assert!(pot > 0.0 && stack > 0.0, "pot and stack must be positive");
        assert!(
            turn_fraction > 0.0 && river_fraction > 0.0,
            "bet fractions must be positive"
        );
        assert!(!oop.is_empty() && !ip.is_empty(), "both ranges must be non-empty");

        let mut dead = CardSet::empty();
        for &card in board {
            assert!(dead.insert(card), "duplicate board card {card}");
        }
        for range in [&oop, &ip] {
            for combo in range {
                assert_ne!(combo[0], combo[1], "a holding cannot repeat a card");
                for card in combo {
                    assert!(!dead.contains(*card), "holding uses board card {card}");
                }
            }
        }

        let pot_chips = to_chips(pot);
        let stack_chips = to_chips(stack);
        let turn_bet = to_chips(pot * turn_fraction).clamp(1, stack_chips);

        // Each turn line leaves a different pot behind, so each prices its own
        // river bet.
        let river_bet = std::array::from_fn(|line| {
            let extra = match TurnLine::ALL[line] {
                TurnLine::CheckedThrough => 0,
                TurnLine::BetCalled | TurnLine::CheckRaisedCalled => 2 * turn_bet,
            };
            let pot_here = pot_chips + extra;
            let remaining = stack_chips.saturating_sub(extra / 2).max(1);
            ((pot_here as f64 * river_fraction).round() as u32).clamp(1, remaining)
        });

        TurnRiver {
            board: board.to_vec(),
            pot: pot_chips,
            stack: stack_chips,
            turn_bet,
            river_bet,
            ranges: [oop, ip],
        }
    }

    pub fn pot(&self) -> f64 {
        to_blinds(self.pot)
    }

    pub fn turn_bet(&self) -> f64 {
        to_blinds(self.turn_bet)
    }

    pub fn river_bet(&self, line: TurnLine) -> f64 {
        to_blinds(self.river_bet[line as usize])
    }

    /// A player's concrete holdings.
    pub fn range(&self, player: usize) -> &[[Card; 2]] {
        &self.ranges[player]
    }

    /// The information set for a holding at a stage.
    ///
    /// River decisions carry the river card and the turn line as well, because
    /// both are public and both change the decision. Turn decisions use
    /// [`NO_RIVER`], which keeps the two streets from ever sharing a key.
    pub fn info_key(stage: Stage, holding: usize, river: Option<Card>, line: TurnLine) -> InfoKey {
        let index = stage.decision_index().expect("not a decision stage") as u64;
        let card = river.map(|c| c.index() as u64).unwrap_or(NO_RIVER);
        debug_assert!(card < (1 << CARD_BITS), "card field is too narrow");

        let mut key = index;
        key |= (line as u64) << STAGE_BITS;
        key |= card << (STAGE_BITS + LINE_BITS);
        key | (holding as u64) << (STAGE_BITS + LINE_BITS + CARD_BITS)
    }

    /// The bet facing a player at `stage`, in chips.
    fn wager(&self, stage: Stage, line: u8) -> u32 {
        if stage.is_river() {
            self.river_bet[line as usize]
        } else {
            self.turn_bet
        }
    }

    /// Cards that could still come on the river, given what is already out.
    fn river_candidates(&self, state: &State) -> Vec<Card> {
        let mut dead: CardSet = self.board.iter().copied().collect();
        for player in 0..2 {
            for card in self.ranges[player][state.hands[player] as usize] {
                dead.insert(card);
            }
        }
        CardSet::full_deck().difference(dead).iter().collect()
    }

    /// The five-card board including the river.
    fn full_board(&self, state: &State) -> Vec<Card> {
        let mut board = self.board.clone();
        board.push(state.river().expect("river has been dealt"));
        board
    }
}

impl Game for TurnRiver {
    type State = State;

    fn initial(&self) -> State {
        State {
            hands: [0, 0],
            river: u8::MAX,
            line: 0,
            stage: Stage::Deal,
            committed: [0, 0],
        }
    }

    fn is_terminal(&self, state: &State) -> bool {
        matches!(state.stage, Stage::Folded(_) | Stage::Showdown)
    }

    fn terminal_utility(&self, state: &State) -> f64 {
        // Each player owns half the starting pot. Folding gives up that half
        // plus everything already wagered; an unmatched bet was never at risk.
        let half_pot = to_blinds(self.pot) / 2.0;
        match state.stage {
            Stage::Folded(0) => -(half_pot + state.committed(0)),
            Stage::Folded(1) => half_pot + state.committed(1),
            Stage::Showdown => {
                debug_assert_eq!(
                    state.committed[0], state.committed[1],
                    "a showdown means the wager was matched"
                );
                let at_risk = half_pot + state.committed(0);
                let board = self.full_board(state);
                let mut hand = Vec::with_capacity(7);

                let rank_of = |player: usize, hand: &mut Vec<Card>| {
                    hand.clear();
                    hand.extend_from_slice(&self.ranges[player][state.hands[player] as usize]);
                    hand.extend_from_slice(&board);
                    evaluate(hand)
                };
                let oop = rank_of(0, &mut hand);
                let ip = rank_of(1, &mut hand);

                match oop.cmp(&ip) {
                    std::cmp::Ordering::Greater => at_risk,
                    std::cmp::Ordering::Less => -at_risk,
                    std::cmp::Ordering::Equal => 0.0,
                }
            }
            other => unreachable!("{other:?} is not terminal"),
        }
    }

    fn is_chance(&self, state: &State) -> bool {
        matches!(state.stage, Stage::Deal | Stage::RiverDeal)
    }

    fn chance_outcomes(&self, state: &State) -> Vec<(State, f64)> {
        match state.stage {
            Stage::Deal => {
                // Every pairing of holdings that does not share a card.
                let mut outcomes = Vec::new();
                for (oop, oop_cards) in self.ranges[0].iter().enumerate() {
                    for (ip, ip_cards) in self.ranges[1].iter().enumerate() {
                        if oop_cards.iter().any(|card| ip_cards.contains(card)) {
                            continue;
                        }
                        outcomes.push(State {
                            hands: [oop as u16, ip as u16],
                            stage: Stage::TurnOopFirst,
                            ..*state
                        });
                    }
                }
                let probability = 1.0 / outcomes.len() as f64;
                outcomes.into_iter().map(|s| (s, probability)).collect()
            }
            Stage::RiverDeal => {
                let candidates = self.river_candidates(state);
                let probability = 1.0 / candidates.len() as f64;
                candidates
                    .into_iter()
                    .map(|card| {
                        (
                            State {
                                river: card.index(),
                                stage: Stage::RiverOopFirst,
                                ..*state
                            },
                            probability,
                        )
                    })
                    .collect()
            }
            other => unreachable!("{other:?} is not a chance node"),
        }
    }

    fn sample_chance(&self, state: &State, rng: &mut Rng) -> State {
        match state.stage {
            Stage::Deal => loop {
                let oop = rng.below(self.ranges[0].len() as u64) as usize;
                let ip = rng.below(self.ranges[1].len() as u64) as usize;
                // Redraw when the two holdings would share a card.
                if self.ranges[0][oop]
                    .iter()
                    .any(|card| self.ranges[1][ip].contains(card))
                {
                    continue;
                }
                return State {
                    hands: [oop as u16, ip as u16],
                    stage: Stage::TurnOopFirst,
                    ..*state
                };
            },
            Stage::RiverDeal => {
                let candidates = self.river_candidates(state);
                let pick = rng.below(candidates.len() as u64) as usize;
                State {
                    river: candidates[pick].index(),
                    stage: Stage::RiverOopFirst,
                    ..*state
                }
            }
            other => unreachable!("{other:?} is not a chance node"),
        }
    }

    fn current_player(&self, state: &State) -> usize {
        state
            .stage
            .actor()
            .unwrap_or_else(|| unreachable!("{:?} is not a decision stage", state.stage))
    }

    fn info_key(&self, state: &State) -> InfoKey {
        let player = self.current_player(state);
        let river = if state.stage.is_river() {
            state.river()
        } else {
            None
        };
        TurnRiver::info_key(
            state.stage,
            state.hands[player] as usize,
            river,
            state.line(),
        )
    }

    fn num_actions(&self, _state: &State) -> usize {
        2
    }

    fn apply(&self, state: &State, action: usize) -> State {
        let actor = self.current_player(state);
        let opponent = 1 - actor;
        let mut next = *state;
        let aggressive = action == AGGRESSIVE;

        match state.stage {
            Stage::TurnOopFirst => {
                if aggressive {
                    next.committed[0] += self.turn_bet;
                    next.stage = Stage::TurnIpVsBet;
                } else {
                    next.stage = Stage::TurnIpVsCheck;
                }
            }
            Stage::TurnIpVsCheck => {
                if aggressive {
                    next.committed[1] += self.turn_bet;
                    next.stage = Stage::TurnOopVsBet;
                } else {
                    next.line = TurnLine::CheckedThrough as u8;
                    next.stage = Stage::RiverDeal;
                }
            }
            Stage::RiverOopFirst => {
                if aggressive {
                    next.committed[0] += self.wager(state.stage, state.line);
                    next.stage = Stage::RiverIpVsBet;
                } else {
                    next.stage = Stage::RiverIpVsCheck;
                }
            }
            Stage::RiverIpVsCheck => {
                if aggressive {
                    next.committed[1] += self.wager(state.stage, state.line);
                    next.stage = Stage::RiverOopVsBet;
                } else {
                    next.stage = Stage::Showdown;
                }
            }
            stage if stage.faces_bet() => {
                if !aggressive {
                    next.stage = Stage::Folded(actor as u8);
                } else {
                    next.committed[actor] = state.committed[opponent];
                    next.stage = if stage.is_river() {
                        Stage::Showdown
                    } else {
                        // Record which turn line led here; it sets the pot.
                        next.line = if stage == Stage::TurnIpVsBet {
                            TurnLine::BetCalled as u8
                        } else {
                            TurnLine::CheckRaisedCalled as u8
                        };
                        Stage::RiverDeal
                    };
                }
            }
            other => unreachable!("{other:?} is not a decision stage"),
        }

        next
    }
}

impl fmt::Display for TurnRiver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let board: Vec<String> = self.board.iter().map(|c| c.to_string()).collect();
        write!(
            f,
            "turn+river: {} pot {:.2} stack {:.2}, ranges {}x{}",
            board.join(" "),
            self.pot(),
            to_blinds(self.stack),
            self.ranges[0].len(),
            self.ranges[1].len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::parse_cards;
    use crate::cfr::Solver;
    use crate::range::Range;

    const BOARD: &str = "Qs 8h 3d 7c";

    fn combos(text: &str, board: &[Card]) -> Vec<[Card; 2]> {
        let range: Range = text.parse().expect("valid range");
        let dead: CardSet = board.iter().copied().collect();
        range
            .entries()
            .flat_map(|(class, _)| class.combinations())
            .filter(|combo| !dead.contains(combo[0]) && !dead.contains(combo[1]))
            .collect()
    }

    fn spot(oop: &str, ip: &str) -> TurnRiver {
        let board = parse_cards(BOARD).expect("valid board");
        let oop = combos(oop, &board);
        let ip = combos(ip, &board);
        TurnRiver::new(&board, 20.0, 60.0, 0.5, 0.75, oop, ip)
    }

    #[test]
    fn the_river_bet_is_priced_off_the_pot_the_turn_line_left() {
        let game = spot("AA", "KK");
        assert_eq!(game.turn_bet(), 10.0, "half of a 20 bb pot");

        // Checked through, the pot is still 20, so a 0.75 bet is 15.
        assert_eq!(game.river_bet(TurnLine::CheckedThrough), 15.0);
        // Bet and called, the pot is 40, so the same fraction is 30.
        assert_eq!(game.river_bet(TurnLine::BetCalled), 30.0);
        assert_eq!(game.river_bet(TurnLine::CheckRaisedCalled), 30.0);
    }

    #[test]
    fn the_tree_runs_turn_then_river() {
        let game = spot("AA", "KK");
        let root = game.initial();
        assert!(game.is_chance(&root));

        let mut rng = Rng::new(1);
        let dealt = game.sample_chance(&root, &mut rng);
        assert_eq!(dealt.stage(), Stage::TurnOopFirst);
        assert!(dealt.river().is_none(), "the river comes later");

        // Check, check reaches the river deal.
        let checked = game.apply(&game.apply(&dealt, PASSIVE), PASSIVE);
        assert_eq!(checked.stage(), Stage::RiverDeal);
        assert!(game.is_chance(&checked));
        assert_eq!(checked.line(), TurnLine::CheckedThrough);

        let river = game.sample_chance(&checked, &mut rng);
        assert_eq!(river.stage(), Stage::RiverOopFirst);
        assert!(river.river().is_some());

        // Check, check again ends it.
        let showdown = game.apply(&game.apply(&river, PASSIVE), PASSIVE);
        assert_eq!(showdown.stage(), Stage::Showdown);
    }

    #[test]
    fn the_turn_line_is_recorded_correctly() {
        let game = spot("AA", "KK");
        let mut rng = Rng::new(2);
        let dealt = game.sample_chance(&game.initial(), &mut rng);

        // Out of position bets, in position calls.
        let bet_called = game.apply(&game.apply(&dealt, AGGRESSIVE), AGGRESSIVE);
        assert_eq!(bet_called.stage(), Stage::RiverDeal);
        assert_eq!(bet_called.line(), TurnLine::BetCalled);
        assert_eq!(bet_called.committed(0), 10.0);
        assert_eq!(bet_called.committed(1), 10.0);

        // Checked to, in position bets, out of position calls.
        let checked = game.apply(&dealt, PASSIVE);
        let raised = game.apply(&checked, AGGRESSIVE);
        let called = game.apply(&raised, AGGRESSIVE);
        assert_eq!(called.line(), TurnLine::CheckRaisedCalled);
    }

    #[test]
    fn folding_on_the_turn_surrenders_only_what_was_committed() {
        let game = spot("AA", "KK");
        let mut rng = Rng::new(3);
        let dealt = game.sample_chance(&game.initial(), &mut rng);

        let bet = game.apply(&dealt, AGGRESSIVE);
        let folded = game.apply(&bet, PASSIVE);
        assert_eq!(folded.stage(), Stage::Folded(1));
        assert_eq!(
            game.terminal_utility(&folded),
            10.0,
            "wins half the pot; the uncalled bet was never at risk"
        );
    }

    #[test]
    fn folding_the_river_after_calling_the_turn_costs_both() {
        let game = spot("AA", "KK");
        let mut rng = Rng::new(4);
        let dealt = game.sample_chance(&game.initial(), &mut rng);

        // Turn: bet and called, so 10 each is already in.
        let turn = game.apply(&game.apply(&dealt, AGGRESSIVE), AGGRESSIVE);
        let river = game.sample_chance(&turn, &mut rng);
        // River: out of position bets, in position folds.
        let folded = game.apply(&game.apply(&river, AGGRESSIVE), PASSIVE);

        assert_eq!(folded.stage(), Stage::Folded(1));
        assert_eq!(
            game.terminal_utility(&folded),
            20.0,
            "half the pot plus the turn call the folder had already put in"
        );
    }

    #[test]
    fn the_river_never_duplicates_a_card_in_play() {
        let game = spot("AA,KK,QQ", "JJ,TT,99");
        let mut rng = Rng::new(5);
        let board: CardSet = parse_cards(BOARD).expect("valid").into_iter().collect();

        for _ in 0..2_000 {
            let dealt = game.sample_chance(&game.initial(), &mut rng);
            let turn = game.apply(&game.apply(&dealt, PASSIVE), PASSIVE);
            let river = game.sample_chance(&turn, &mut rng);
            let card = river.river().expect("dealt");

            assert!(!board.contains(card), "river repeated a board card");
            for player in 0..2 {
                let hand = game.ranges[player][river.hands[player] as usize];
                assert!(!hand.contains(&card), "river repeated a hole card");
            }
        }
    }

    #[test]
    fn the_two_players_never_hold_the_same_card() {
        // Both ranges contain aces, so the deal must skip the clashes.
        let game = spot("AA", "AA");
        let mut rng = Rng::new(6);
        for _ in 0..1_000 {
            let dealt = game.sample_chance(&game.initial(), &mut rng);
            let oop = game.ranges[0][dealt.hands[0] as usize];
            let ip = game.ranges[1][dealt.hands[1] as usize];
            assert!(
                !oop.iter().any(|card| ip.contains(card)),
                "{oop:?} and {ip:?} share a card"
            );
        }
    }

    #[test]
    fn chance_outcomes_form_distributions() {
        let game = spot("AA", "KK");
        let root = game.initial();
        let deal: f64 = game.chance_outcomes(&root).iter().map(|(_, p)| p).sum();
        assert!((deal - 1.0).abs() < 1e-9);

        let mut rng = Rng::new(7);
        let dealt = game.sample_chance(&root, &mut rng);
        let turn = game.apply(&game.apply(&dealt, PASSIVE), PASSIVE);
        let river = game.chance_outcomes(&turn);
        assert_eq!(river.len(), 44, "52 less four board and four hole cards");
        let total: f64 = river.iter().map(|(_, p)| p).sum();
        assert!((total - 1.0).abs() < 1e-9);
    }

    #[test]
    fn information_sets_separate_streets_lines_and_river_cards() {
        let ace = "As".parse::<Card>().expect("valid");
        let king = "Kd".parse::<Card>().expect("valid");

        // The turn and the river are never the same decision.
        assert_ne!(
            TurnRiver::info_key(Stage::TurnOopFirst, 3, None, TurnLine::CheckedThrough),
            TurnRiver::info_key(Stage::RiverOopFirst, 3, Some(ace), TurnLine::CheckedThrough),
        );
        // Different river cards are different decisions.
        assert_ne!(
            TurnRiver::info_key(Stage::RiverOopFirst, 3, Some(ace), TurnLine::CheckedThrough),
            TurnRiver::info_key(Stage::RiverOopFirst, 3, Some(king), TurnLine::CheckedThrough),
        );
        // Same cards, same river, different pot: still different decisions.
        assert_ne!(
            TurnRiver::info_key(Stage::RiverOopFirst, 3, Some(ace), TurnLine::CheckedThrough),
            TurnRiver::info_key(Stage::RiverOopFirst, 3, Some(ace), TurnLine::BetCalled),
        );

        let mut seen = std::collections::HashSet::new();
        let stages = [
            Stage::TurnOopFirst,
            Stage::TurnIpVsCheck,
            Stage::TurnOopVsBet,
            Stage::TurnIpVsBet,
            Stage::RiverOopFirst,
            Stage::RiverIpVsCheck,
            Stage::RiverOopVsBet,
            Stage::RiverIpVsBet,
        ];
        for stage in stages {
            for holding in 0..12 {
                for line in TurnLine::ALL {
                    for card in Card::all().take(20) {
                        let river = if stage.is_river() { Some(card) } else { None };
                        let key = TurnRiver::info_key(stage, holding, river, line);
                        // Turn keys repeat across river cards by design, since
                        // the river is not yet known there.
                        if !stage.is_river() && card.index() > 0 {
                            continue;
                        }
                        assert!(seen.insert(key), "{stage:?} h{holding} {line:?} collided");
                    }
                }
            }
        }
    }

    #[test]
    fn a_stronger_range_wins_money_against_a_weaker_one() {
        // Aces against kings on a board that helps neither: the favourite must
        // come out ahead once both play well.
        let game = spot("AA", "KK");
        let mut rng = Rng::new(8);
        let mut solver = Solver::new(game);
        solver.train_sampled(200_000, &mut rng);

        let value = solver.expected_value(&solver.profile());
        assert!(value > 1.0, "aces should profit, got {value} bb");
    }

    #[test]
    fn the_solve_converges() {
        let game = spot("AA,KK", "QQ,JJ");
        let mut rng = Rng::new(9);
        let mut solver = Solver::new(game);
        solver.train_sampled(400_000, &mut rng);

        let exploitability = solver.exploitability(&solver.profile());
        assert!(
            exploitability.is_finite() && exploitability >= 0.0,
            "exploitability was {exploitability}"
        );
        assert!(exploitability < 3.0, "exploitable for {exploitability} bb");
    }

    #[test]
    #[should_panic(expected = "a turn board is four cards")]
    fn a_five_card_board_is_rejected() {
        let board = parse_cards("Qs 8h 3d 7c 2s").expect("valid");
        TurnRiver::new(&board, 20.0, 60.0, 0.5, 0.75, vec![], vec![]);
    }

    #[test]
    #[should_panic(expected = "holding uses board card")]
    fn a_holding_that_clashes_with_the_board_is_rejected() {
        let board = parse_cards(BOARD).expect("valid");
        let clash = vec![[board[0], "2c".parse().expect("valid")]];
        let fine = combos("KK", &board);
        TurnRiver::new(&board, 20.0, 60.0, 0.5, 0.75, clash, fine);
    }
}
