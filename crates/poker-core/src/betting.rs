//! Betting round mechanics: legal actions, raise sizing, and round completion.
//!
//! # The rule that engines get wrong
//!
//! An all-in raise for *less* than a full raise does not reopen the betting.
//! If A bets 100 and B moves all in for 150 when a full raise would be 200,
//! then A must call or fold — A may not re-raise. But a player who had not yet
//! acted keeps full rights, including raising.
//!
//! Both cases fall out of a single piece of state: `has_acted`, meaning "acted
//! since the last *full* raise". A player may raise exactly when that is false,
//! and must act when they face unmatched chips or have not acted yet. The big
//! blind's preflop option is the same rule with no special case — the blind was
//! posted, not acted.
//!
//! # Amounts
//!
//! Raises are expressed as [`Action::RaiseTo`], a total street commitment
//! rather than an increment. "Raise by" semantics are ambiguous once a player
//! already has chips in front of them, and that ambiguity is a reliable source
//! of off-by-one-bet errors.

use std::fmt;

/// A betting street.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Street {
    Preflop,
    Flop,
    Turn,
    River,
}

impl Street {
    /// The street after this one, or `None` after the river.
    pub const fn next(self) -> Option<Street> {
        match self {
            Street::Preflop => Some(Street::Flop),
            Street::Flop => Some(Street::Turn),
            Street::Turn => Some(Street::River),
            Street::River => None,
        }
    }

    /// How many board cards are face up during this street.
    pub const fn board_cards(self) -> usize {
        match self {
            Street::Preflop => 0,
            Street::Flop => 3,
            Street::Turn => 4,
            Street::River => 5,
        }
    }
}

impl fmt::Display for Street {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Street::Preflop => "preflop",
            Street::Flop => "flop",
            Street::Turn => "turn",
            Street::River => "river",
        })
    }
}

/// An action a player can take.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Action {
    Fold,
    Check,
    Call,
    /// Raise this street's total commitment to the given amount.
    ///
    /// A first bet is a raise to the bet size, since the current bet is zero.
    RaiseTo(u64),
}

/// One player's state within a betting round.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Seat {
    /// Chips still behind.
    pub stack: u64,
    /// Chips committed on this street.
    pub committed: u64,
    /// Chips committed across the whole hand, including earlier streets.
    pub total_committed: u64,
    pub folded: bool,
    /// Acted since the last full raise. Drives both "must act" and "may raise".
    has_acted: bool,
}

impl Seat {
    /// A seat with a starting stack and nothing invested.
    pub fn new(stack: u64) -> Seat {
        Seat {
            stack,
            committed: 0,
            total_committed: 0,
            folded: false,
            has_acted: false,
        }
    }

    /// All chips are in the middle, so this seat can take no further action.
    #[inline]
    pub fn is_all_in(&self) -> bool {
        self.stack == 0
    }

    /// Still in the hand and able to act.
    #[inline]
    pub fn is_live(&self) -> bool {
        !self.folded && !self.is_all_in()
    }
}

/// What the player to act may legally do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LegalActions {
    pub can_fold: bool,
    pub can_check: bool,
    /// Chips needed to call, if calling is possible. Capped at the stack, so
    /// this is an all-in call when it equals the stack.
    pub call_cost: Option<u64>,
    /// Inclusive `(min, max)` total commitments for a legal raise.
    ///
    /// `min` equals `max` when the only raise available is all-in for less than
    /// a full raise.
    pub raise_to: Option<(u64, u64)>,
}

impl LegalActions {
    /// Whether `action` is permitted.
    pub fn permits(&self, action: Action) -> bool {
        match action {
            Action::Fold => self.can_fold,
            Action::Check => self.can_check,
            Action::Call => self.call_cost.is_some(),
            Action::RaiseTo(to) => self
                .raise_to
                .is_some_and(|(min, max)| (min..=max).contains(&to)),
        }
    }
}

/// Why an action was rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActionError {
    /// The round is over; nobody is to act.
    RoundComplete,
    /// The action is not legal in this spot.
    Illegal { action: Action, legal: Box<LegalActions> },
}

impl fmt::Display for ActionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ActionError::RoundComplete => f.write_str("betting round is already complete"),
            ActionError::Illegal { action, legal } => {
                write!(f, "{action:?} is not legal here; permitted: {legal:?}")
            }
        }
    }
}

impl std::error::Error for ActionError {}

/// A single betting round.
#[derive(Debug, Clone)]
pub struct BettingRound {
    seats: Vec<Seat>,
    to_act: usize,
    /// Highest commitment on this street.
    current_bet: u64,
    /// Increment a raise must add to be a full raise.
    min_raise_increment: u64,
    /// Fixed reference size, normally the big blind, used as the opening bet
    /// minimum after the flop.
    big_blind: u64,
}

impl BettingRound {
    /// Starts a round in which nobody has bet yet, `first_to_act` acts first.
    ///
    /// Use [`BettingRound::preflop`] for the blinds case.
    ///
    /// # Panics
    /// Panics if there are fewer than two seats or `first_to_act` is not a seat.
    pub fn new(seats: Vec<Seat>, first_to_act: usize, big_blind: u64) -> BettingRound {
        assert!(seats.len() >= 2, "a betting round needs at least two seats");
        assert!(first_to_act < seats.len(), "first_to_act is not a seat");

        let current_bet = seats.iter().map(|s| s.committed).max().unwrap_or(0);
        let mut round = BettingRound {
            seats,
            to_act: first_to_act,
            current_bet,
            min_raise_increment: big_blind.max(1),
            big_blind: big_blind.max(1),
        };
        round.seek_actor_from(first_to_act);
        round
    }

    /// Starts a preflop round with blinds already posted.
    ///
    /// `blinds` gives `(seat, amount)` pairs, posted in order. Posting is not
    /// acting, so the big blind retains the option to raise.
    ///
    /// # Panics
    /// Panics as [`BettingRound::new`], or if a blind seat does not exist.
    pub fn preflop(
        mut seats: Vec<Seat>,
        blinds: &[(usize, u64)],
        first_to_act: usize,
        big_blind: u64,
    ) -> BettingRound {
        for &(seat, amount) in blinds {
            assert!(seat < seats.len(), "blind posted by non-existent seat {seat}");
            let posted = amount.min(seats[seat].stack);
            seats[seat].stack -= posted;
            seats[seat].committed += posted;
            seats[seat].total_committed += posted;
        }
        BettingRound::new(seats, first_to_act, big_blind)
    }

    pub fn seats(&self) -> &[Seat] {
        &self.seats
    }

    /// The highest commitment on this street.
    pub fn current_bet(&self) -> u64 {
        self.current_bet
    }

    /// The seat to act, or `None` when the round is complete.
    pub fn to_act(&self) -> Option<usize> {
        (!self.is_complete()).then_some(self.to_act)
    }

    /// Chips committed this street across all seats.
    pub fn street_total(&self) -> u64 {
        self.seats.iter().map(|s| s.committed).sum()
    }

    /// Seats that have not folded.
    pub fn live_seat_count(&self) -> usize {
        self.seats.iter().filter(|s| !s.folded).count()
    }

    /// Whether the round has finished.
    ///
    /// It has when at most one player remains unfolded, or when no unfolded,
    /// non-all-in player still owes an action.
    pub fn is_complete(&self) -> bool {
        if self.live_seat_count() <= 1 {
            return true;
        }
        !self.seats.iter().any(|s| self.owes_action(s))
    }

    /// Whether this seat must still act: it can act at all, and either faces
    /// unmatched chips or has not acted since the last full raise.
    fn owes_action(&self, seat: &Seat) -> bool {
        seat.is_live() && (seat.committed < self.current_bet || !seat.has_acted)
    }

    /// What the seat to act may do.
    ///
    /// # Panics
    /// Panics if the round is already complete.
    pub fn legal_actions(&self) -> LegalActions {
        assert!(!self.is_complete(), "no legal actions once the round is over");
        let seat = &self.seats[self.to_act];
        let owed = self.current_bet.saturating_sub(seat.committed);

        // Folding when nothing is owed is legal but strictly dominated by
        // checking, so it is excluded to keep solver action sets minimal.
        let can_fold = owed > 0;
        let can_check = owed == 0;
        let call_cost = (owed > 0).then(|| owed.min(seat.stack));

        // A raise is only available to a player the action is still open to.
        let max_raise_to = seat.committed + seat.stack;
        let raise_to = if seat.has_acted || max_raise_to <= self.current_bet {
            None
        } else {
            // Postflop the first bet must be at least a big blind; a raise must
            // add at least the last raise increment.
            let full_raise_to = if self.current_bet == 0 {
                self.big_blind
            } else {
                self.current_bet + self.min_raise_increment
            };
            // Short stacks may always move all in, even for less.
            Some((full_raise_to.min(max_raise_to), max_raise_to))
        };

        LegalActions {
            can_fold,
            can_check,
            call_cost,
            raise_to,
        }
    }

    /// Applies `action` for the seat to act and moves the button along.
    pub fn apply(&mut self, action: Action) -> Result<(), ActionError> {
        if self.is_complete() {
            return Err(ActionError::RoundComplete);
        }
        let legal = self.legal_actions();
        if !legal.permits(action) {
            return Err(ActionError::Illegal {
                action,
                legal: Box::new(legal),
            });
        }

        let actor = self.to_act;
        match action {
            Action::Fold => {
                self.seats[actor].folded = true;
                self.seats[actor].has_acted = true;
            }
            Action::Check => {
                self.seats[actor].has_acted = true;
            }
            Action::Call => {
                let owed = self.current_bet - self.seats[actor].committed;
                self.commit(actor, owed.min(self.seats[actor].stack));
                self.seats[actor].has_acted = true;
            }
            Action::RaiseTo(to) => {
                let increment = to - self.current_bet;
                let extra = to - self.seats[actor].committed;
                self.commit(actor, extra);

                // A full raise reopens the action to everyone else. An all-in
                // for less does not: seats that already acted keep has_acted
                // set, so they may call or fold but not raise.
                if increment >= self.min_raise_increment {
                    self.min_raise_increment = increment;
                    for (i, seat) in self.seats.iter_mut().enumerate() {
                        seat.has_acted = i == actor;
                    }
                } else {
                    self.seats[actor].has_acted = true;
                }
                self.current_bet = to;
            }
        }

        self.seek_actor_from((actor + 1) % self.seats.len());
        Ok(())
    }

    /// Moves `amount` from a seat's stack into the pot.
    fn commit(&mut self, seat: usize, amount: u64) {
        debug_assert!(amount <= self.seats[seat].stack, "seat cannot cover {amount}");
        let seat = &mut self.seats[seat];
        seat.stack -= amount;
        seat.committed += amount;
        seat.total_committed += amount;
    }

    /// Points `to_act` at the first seat from `start` onward that owes an
    /// action, wrapping around the table.
    ///
    /// `start` is inclusive, so opening a round can land on the intended first
    /// actor. After an action, callers pass the seat *after* the one that just
    /// acted.
    fn seek_actor_from(&mut self, start: usize) {
        if self.is_complete() {
            return;
        }
        let n = self.seats.len();
        for step in 0..n {
            let candidate = (start + step) % n;
            if self.owes_action(&self.seats[candidate]) {
                self.to_act = candidate;
                return;
            }
        }
        debug_assert!(false, "round is not complete but no seat owes an action");
    }

    /// Clears street commitments and opens the next round with `first_to_act`.
    ///
    /// Per-hand totals are preserved, so pot construction still sees every chip.
    ///
    /// # Panics
    /// Panics if the current round is not complete.
    pub fn next_street(&mut self, first_to_act: usize) {
        assert!(self.is_complete(), "cannot advance while betting is open");
        assert!(first_to_act < self.seats.len(), "first_to_act is not a seat");

        for seat in &mut self.seats {
            seat.committed = 0;
            seat.has_acted = false;
        }
        self.current_bet = 0;
        self.min_raise_increment = self.big_blind;
        self.to_act = first_to_act;
        self.seek_actor_from(first_to_act);
    }

    /// Per-seat totals for the whole hand, for [`crate::pot::build_pots`].
    pub fn contributions(&self) -> Vec<u64> {
        self.seats.iter().map(|s| s.total_committed).collect()
    }

    /// Per-seat fold flags, for [`crate::pot::build_pots`].
    pub fn folded_flags(&self) -> Vec<bool> {
        self.seats.iter().map(|s| s.folded).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BB: u64 = 2;

    /// Heads-up preflop: seat 0 is the small blind and acts first.
    fn heads_up(stacks: [u64; 2]) -> BettingRound {
        let seats = stacks.into_iter().map(Seat::new).collect();
        BettingRound::preflop(seats, &[(0, 1), (1, BB)], 0, BB)
    }

    /// Three-handed preflop: seats 1 and 2 post, seat 0 acts first.
    fn three_handed(stacks: [u64; 3]) -> BettingRound {
        let seats = stacks.into_iter().map(Seat::new).collect();
        BettingRound::preflop(seats, &[(1, 1), (2, BB)], 0, BB)
    }

    fn apply(round: &mut BettingRound, action: Action) {
        round.apply(action).unwrap_or_else(|e| panic!("{action:?} rejected: {e}"));
    }

    #[test]
    fn blinds_are_posted_without_counting_as_actions() {
        let round = heads_up([200, 200]);
        assert_eq!(round.current_bet(), BB);
        assert_eq!(round.seats()[0].committed, 1);
        assert_eq!(round.seats()[1].committed, 2);
        assert_eq!(round.street_total(), 3);
        assert!(!round.is_complete());
    }

    #[test]
    fn the_big_blind_keeps_the_option_after_a_call() {
        let mut round = heads_up([200, 200]);
        apply(&mut round, Action::Call); // small blind completes

        // Commitments are level, but the blind was posted, not acted, so the
        // big blind still gets to act.
        assert!(!round.is_complete(), "big blind must still get the option");
        assert_eq!(round.to_act(), Some(1));

        let legal = round.legal_actions();
        assert!(legal.can_check);
        assert!(legal.raise_to.is_some(), "the option includes raising");

        apply(&mut round, Action::Check);
        assert!(round.is_complete());
    }

    #[test]
    fn checking_through_completes_the_round() {
        let seats = vec![Seat::new(200), Seat::new(200)];
        let mut round = BettingRound::new(seats, 0, BB);
        apply(&mut round, Action::Check);
        assert!(!round.is_complete());
        apply(&mut round, Action::Check);
        assert!(round.is_complete());
        assert_eq!(round.street_total(), 0);
    }

    #[test]
    fn a_bet_and_call_completes_the_round() {
        let seats = vec![Seat::new(200), Seat::new(200)];
        let mut round = BettingRound::new(seats, 0, BB);
        apply(&mut round, Action::RaiseTo(10));
        apply(&mut round, Action::Call);

        assert!(round.is_complete());
        assert_eq!(round.street_total(), 20);
        assert_eq!(round.seats()[0].stack, 190);
        assert_eq!(round.seats()[1].stack, 190);
    }

    #[test]
    fn folding_ends_the_hand_when_one_player_remains() {
        let mut round = heads_up([200, 200]);
        apply(&mut round, Action::Fold);
        assert!(round.is_complete());
        assert_eq!(round.live_seat_count(), 1);
        assert_eq!(round.to_act(), None);
    }

    #[test]
    fn a_raise_must_add_at_least_the_previous_increment() {
        let seats = vec![Seat::new(200), Seat::new(200)];
        let mut round = BettingRound::new(seats, 0, BB);
        apply(&mut round, Action::RaiseTo(10)); // bet 10

        // A re-raise must reach at least 20: the raise increment was 10.
        let legal = round.legal_actions();
        assert_eq!(legal.raise_to.expect("raise available").0, 20);
        assert!(round.apply(Action::RaiseTo(15)).is_err(), "15 is a short raise");
        apply(&mut round, Action::RaiseTo(20));
        assert_eq!(round.current_bet(), 20);
    }

    #[test]
    fn the_minimum_raise_grows_with_each_raise() {
        let seats = vec![Seat::new(1000), Seat::new(1000), Seat::new(1000)];
        let mut round = BettingRound::new(seats, 0, BB);
        apply(&mut round, Action::RaiseTo(10)); // increment 10
        apply(&mut round, Action::RaiseTo(40)); // increment 30
        // Next raise must add at least 30, so reach 70.
        assert_eq!(round.legal_actions().raise_to.expect("raise").0, 70);
    }

    #[test]
    fn a_postflop_opening_bet_must_be_at_least_a_big_blind() {
        let seats = vec![Seat::new(200), Seat::new(200)];
        let round = BettingRound::new(seats, 0, BB);
        assert_eq!(round.legal_actions().raise_to.expect("bet available").0, BB);
    }

    #[test]
    fn a_short_stack_may_always_move_all_in() {
        // Seat 1 cannot make a full raise to 20 but may still jam 15.
        let seats = vec![Seat::new(200), Seat::new(15)];
        let mut round = BettingRound::new(seats, 0, BB);
        apply(&mut round, Action::RaiseTo(10));

        let legal = round.legal_actions();
        assert_eq!(legal.raise_to, Some((15, 15)), "all-in is the only raise");
        apply(&mut round, Action::RaiseTo(15));
        assert!(round.seats()[1].is_all_in());
    }

    #[test]
    fn an_all_in_under_raise_does_not_reopen_betting() {
        // Seat 0 bets 100, seat 1 jams 150 — short of the 200 a full raise
        // needs. Seat 0 must call or fold, and may not re-raise.
        let seats = vec![Seat::new(1000), Seat::new(150)];
        let mut round = BettingRound::new(seats, 0, BB);
        apply(&mut round, Action::RaiseTo(100));
        apply(&mut round, Action::RaiseTo(150));

        assert_eq!(round.to_act(), Some(0), "seat 0 still owes the extra 50");
        let legal = round.legal_actions();
        assert_eq!(legal.call_cost, Some(50));
        assert!(legal.can_fold);
        assert_eq!(legal.raise_to, None, "the action was not reopened");
        assert!(round.apply(Action::RaiseTo(300)).is_err());
    }

    #[test]
    fn an_all_in_under_raise_still_leaves_full_rights_to_players_yet_to_act() {
        // Seat 0 bets 100 and seat 1 jams 150 short. Seat 2 has not acted, so
        // seat 2 keeps the right to raise even though seat 0 lost it.
        let seats = vec![Seat::new(1000), Seat::new(150), Seat::new(1000)];
        let mut round = BettingRound::new(seats, 0, BB);
        apply(&mut round, Action::RaiseTo(100));
        apply(&mut round, Action::RaiseTo(150));

        assert_eq!(round.to_act(), Some(2));
        let legal = round.legal_actions();
        assert!(
            legal.raise_to.is_some(),
            "a player who has not acted keeps full rights"
        );

        apply(&mut round, Action::Call);
        // Now back to seat 0, who may only call or fold.
        assert_eq!(round.to_act(), Some(0));
        assert_eq!(round.legal_actions().raise_to, None);
    }

    #[test]
    fn a_full_all_in_raise_does_reopen_betting() {
        // Seat 1 jams to 200, a full raise over the bet of 100.
        let seats = vec![Seat::new(1000), Seat::new(200), Seat::new(1000)];
        let mut round = BettingRound::new(seats, 0, BB);
        apply(&mut round, Action::RaiseTo(100));
        apply(&mut round, Action::RaiseTo(200));

        apply(&mut round, Action::Call); // seat 2 calls
        assert_eq!(round.to_act(), Some(0));
        assert!(
            round.legal_actions().raise_to.is_some(),
            "a full raise reopens the action"
        );
    }

    #[test]
    fn calling_off_a_short_stack_is_an_all_in_call() {
        let seats = vec![Seat::new(1000), Seat::new(40)];
        let mut round = BettingRound::new(seats, 0, BB);
        apply(&mut round, Action::RaiseTo(100));

        assert_eq!(round.legal_actions().call_cost, Some(40), "capped at the stack");
        apply(&mut round, Action::Call);
        assert!(round.seats()[1].is_all_in());
        assert_eq!(round.seats()[1].committed, 40);
        assert!(round.is_complete());
    }

    #[test]
    fn an_all_in_player_is_skipped_on_later_streets() {
        let seats = vec![Seat::new(1000), Seat::new(100), Seat::new(1000)];
        let mut round = BettingRound::new(seats, 0, BB);
        apply(&mut round, Action::RaiseTo(100));
        apply(&mut round, Action::Call); // seat 1 all in
        apply(&mut round, Action::Call);
        assert!(round.is_complete());

        round.next_street(0);
        apply(&mut round, Action::Check);
        assert_eq!(round.to_act(), Some(2), "the all-in seat is skipped");
    }

    #[test]
    fn folding_is_unavailable_when_checking_is_free() {
        let seats = vec![Seat::new(200), Seat::new(200)];
        let round = BettingRound::new(seats, 0, BB);
        let legal = round.legal_actions();
        assert!(legal.can_check);
        assert!(!legal.can_fold, "folding for free is dominated by checking");
    }

    #[test]
    fn next_street_clears_commitments_but_keeps_hand_totals() {
        let seats = vec![Seat::new(200), Seat::new(200)];
        let mut round = BettingRound::new(seats, 0, BB);
        apply(&mut round, Action::RaiseTo(50));
        apply(&mut round, Action::Call);

        round.next_street(0);
        assert_eq!(round.current_bet(), 0);
        assert_eq!(round.street_total(), 0, "street commitments reset");
        assert_eq!(round.contributions(), vec![50, 50], "hand totals persist");
        assert!(round.legal_actions().can_check);
    }

    #[test]
    fn contributions_feed_pot_construction() {
        use crate::pot::{build_pots, total};

        let seats = vec![Seat::new(1000), Seat::new(50), Seat::new(1000)];
        let mut round = BettingRound::new(seats, 0, BB);
        apply(&mut round, Action::RaiseTo(100));
        apply(&mut round, Action::Call); // seat 1 all in for 50
        apply(&mut round, Action::Fold); // seat 2 folds

        let pots = build_pots(&round.contributions(), &round.folded_flags());
        assert_eq!(total(&pots), round.street_total());
        assert_eq!(round.contributions(), vec![100, 50, 0]);
    }

    #[test]
    fn chips_are_never_created_or_destroyed() {
        let starting = [1000u64, 250, 60];
        let seats = starting.into_iter().map(Seat::new).collect();
        let mut round = BettingRound::new(seats, 0, BB);

        apply(&mut round, Action::RaiseTo(100));
        apply(&mut round, Action::RaiseTo(250)); // seat 1 all in
        apply(&mut round, Action::Call); // seat 2 all in for 60
        apply(&mut round, Action::Call); // seat 0 completes

        let held: u64 = round.seats().iter().map(|s| s.stack).sum();
        assert_eq!(held + round.street_total(), starting.iter().sum::<u64>());
    }

    #[test]
    fn acting_after_the_round_is_rejected() {
        let mut round = heads_up([200, 200]);
        apply(&mut round, Action::Fold);
        assert_eq!(round.apply(Action::Check), Err(ActionError::RoundComplete));
    }

    #[test]
    fn three_handed_action_runs_in_seat_order() {
        let mut round = three_handed([200, 200, 200]);
        assert_eq!(round.to_act(), Some(0));
        apply(&mut round, Action::Call);
        assert_eq!(round.to_act(), Some(1));
        apply(&mut round, Action::Call);
        assert_eq!(round.to_act(), Some(2), "big blind gets the option last");
        apply(&mut round, Action::Check);
        assert!(round.is_complete());
    }

    #[test]
    fn street_metadata_is_consistent() {
        assert_eq!(Street::Preflop.next(), Some(Street::Flop));
        assert_eq!(Street::River.next(), None);
        assert_eq!(Street::Flop.board_cards(), 3);
        assert_eq!(Street::River.board_cards(), 5);
        assert!(Street::Preflop < Street::River);
    }
}
