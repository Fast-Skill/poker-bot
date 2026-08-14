//! Omaha hand evaluation.
//!
//! Omaha's defining rule is that a hand plays **exactly two** hole cards and
//! **exactly three** board cards — never one, never three from hand. That single
//! constraint is responsible for most Omaha misreads: four to a flush on the
//! board is worthless holding one card of the suit, quads on the board play as
//! trips, and a board straight cannot simply be played.
//!
//! The functions here enumerate all valid splits and take the best. For a
//! standard four-card hand on a full board that is
//! `C(4,2) * C(5,3) = 6 * 10 = 60` five-card evaluations. Hole sizes other than
//! four are accepted, so five- and six-card Omaha variants work unchanged.

use crate::card::Card;
use crate::eval::{evaluate, HandRank};

/// The number of hole cards that must play in an Omaha hand.
pub const HOLE_CARDS_USED: usize = 2;
/// The number of board cards that must play in an Omaha hand.
pub const BOARD_CARDS_USED: usize = 3;

/// Evaluates the best legal Omaha hand from `hole` and `board`.
///
/// `board` may hold 3, 4, or 5 cards, so this works on the flop and turn as
/// well as at showdown.
///
/// # Panics
/// Panics if `hole` holds fewer than 2 cards or `board` fewer than 3, since no
/// legal hand exists and the caller has a bug.
pub fn evaluate_omaha(hole: &[Card], board: &[Card]) -> HandRank {
    best_omaha_hand(hole, board).0
}

/// Like [`evaluate_omaha`], but also returns the five cards that made the hand.
///
/// The two hole cards come first, then the three board cards. Useful for
/// logging and for sanity-checking what the table reader thinks it saw.
///
/// # Panics
/// Panics if `hole` holds fewer than 2 cards or `board` fewer than 3.
pub fn best_omaha_hand(hole: &[Card], board: &[Card]) -> (HandRank, [Card; 5]) {
    assert!(
        hole.len() >= HOLE_CARDS_USED,
        "Omaha needs at least {HOLE_CARDS_USED} hole cards, got {}",
        hole.len()
    );
    assert!(
        board.len() >= BOARD_CARDS_USED,
        "Omaha needs at least {BOARD_CARDS_USED} board cards, got {}",
        board.len()
    );

    let mut best = HandRank::WORST;
    let mut best_cards = [hole[0]; 5];
    let mut five = [hole[0]; 5];

    for (i, &hole_a) in hole.iter().enumerate() {
        five[0] = hole_a;
        for &hole_b in &hole[i + 1..] {
            five[1] = hole_b;
            for (a, &board_a) in board.iter().enumerate() {
                five[2] = board_a;
                for (b, &board_b) in board.iter().enumerate().skip(a + 1) {
                    five[3] = board_b;
                    for &board_c in &board[b + 1..] {
                        five[4] = board_c;
                        let rank = evaluate(&five);
                        if rank > best {
                            best = rank;
                            best_cards = five;
                        }
                    }
                }
            }
        }
    }

    (best, best_cards)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::parse_cards;
    use crate::eval::Category;

    fn omaha(hole: &str, board: &str) -> HandRank {
        let hole = parse_cards(hole).expect("valid hole cards");
        let board = parse_cards(board).expect("valid board");
        evaluate_omaha(&hole, &board)
    }

    /// Hold'em result for a *two*-card hand on the same board, for contrast.
    /// Takes two cards rather than a full Omaha hand because seven cards is
    /// what Hold'em actually evaluates.
    fn holdem(hole2: &str, board: &str) -> HandRank {
        let mut cards = parse_cards(hole2).expect("valid hole cards");
        assert_eq!(cards.len(), 2, "Hold'em comparison takes exactly two cards");
        cards.extend(parse_cards(board).expect("valid board"));
        evaluate(&cards)
    }

    #[test]
    fn one_hole_card_cannot_complete_a_flush() {
        // Four spades on the board, one in hand. A Hold'em player holding Qs
        // has a flush; the Omaha player cannot use that single card, so the
        // same holding is only ace-high.
        let (hole, board) = ("Qs Jd 9c 8c", "As Ks 7s 2s 3d");
        assert_eq!(holdem("Qs Jd", board).category(), Category::Flush);
        assert_eq!(omaha(hole, board).category(), Category::HighCard);
    }

    #[test]
    fn two_suited_hole_cards_do_complete_a_flush() {
        let hand = omaha("Qs Js 9c 8c", "As Ks 7s 2h 3d");
        assert_eq!(hand.category(), Category::Flush);
        // Ace-king-queen-jack-seven of spades.
        assert_eq!(hand, evaluate(&parse_cards("As Ks Qs Js 7s").expect("valid")));
    }

    #[test]
    fn a_board_straight_cannot_simply_be_played() {
        // Broadway on the board. Hold'em plays it; Omaha must use two hole
        // cards, and these four cannot extend or match it.
        let (hole, board) = ("2c 3d 4h 5s", "As Kd Qh Jc Ts");
        assert_eq!(holdem("2c 3d", board).category(), Category::Straight);
        assert_eq!(omaha(hole, board).category(), Category::HighCard);
    }

    #[test]
    fn quads_on_the_board_play_as_trips() {
        // Only three board cards may be used, so the fourth nine is unreachable.
        let (hole, board) = ("Ac Kd Qh Js", "9c 9d 9h 9s 2c");
        assert_eq!(holdem("Ac Kd", board).category(), Category::FourOfAKind);

        let hand = omaha(hole, board);
        assert_eq!(hand.category(), Category::ThreeOfAKind);
        // Trip nines with ace-king kickers.
        assert_eq!(hand, evaluate(&parse_cards("9c 9d 9h Ac Kd").expect("valid")));
    }

    #[test]
    fn a_pocket_pair_plus_board_trips_makes_a_full_house() {
        let hand = omaha("Ac Ad 7h 2s", "Kc Kd Kh 5s 3c");
        assert_eq!(hand.category(), Category::FullHouse);
        assert_eq!(hand, evaluate(&parse_cards("Kc Kd Kh Ac Ad").expect("valid")));
    }

    #[test]
    fn works_on_the_flop_and_turn() {
        // Exactly one board combination exists on the flop.
        let flop = omaha("As Ks 2c 3d", "Qs Js Ts");
        assert_eq!(flop.category(), Category::StraightFlush);

        let turn = omaha("As Ks 2c 3d", "Qs Js Ts 9h");
        assert_eq!(turn.category(), Category::StraightFlush);
    }

    #[test]
    fn returns_the_five_cards_that_played() {
        let hole = parse_cards("Ac Ad 7h 2s").expect("valid");
        let board = parse_cards("Kc Kd Kh 5s 3c").expect("valid");
        let (rank, cards) = best_omaha_hand(&hole, &board);

        assert_eq!(rank.category(), Category::FullHouse);
        assert_eq!(rank, evaluate(&cards), "returned cards must reproduce the rank");

        // Exactly two from hand, exactly three from board.
        assert_eq!(cards[..2].iter().filter(|c| hole.contains(c)).count(), 2);
        assert_eq!(cards[2..].iter().filter(|c| board.contains(c)).count(), 3);
    }

    #[test]
    fn five_card_omaha_hole_sizes_are_supported() {
        // PLO5: five hole cards, but still exactly two of them play.
        // Q-J from hand with T-9-8 from board is the only straight available;
        // the two spades in hand cannot flush, since the board holds only two.
        let hand = omaha("As Ks Qd Jd 2c", "Ts 9s 8h 4d 3c");
        assert_eq!(hand.category(), Category::Straight);
        assert_eq!(hand, evaluate(&parse_cards("Qd Jd Ts 9s 8h").expect("valid")));
    }

    #[test]
    fn evaluation_is_independent_of_card_order() {
        let a = omaha("Ac Ad 7h 2s", "Kc Kd Kh 5s 3c");
        let b = omaha("2s 7h Ad Ac", "3c 5s Kh Kd Kc");
        assert_eq!(a, b);
    }

    #[test]
    #[should_panic(expected = "at least 2 hole cards")]
    fn rejects_too_few_hole_cards() {
        evaluate_omaha(&parse_cards("As").expect("valid"), &parse_cards("Kc Kd Kh").expect("valid"));
    }

    #[test]
    #[should_panic(expected = "at least 3 board cards")]
    fn rejects_too_few_board_cards() {
        evaluate_omaha(
            &parse_cards("As Ks Qs Js").expect("valid"),
            &parse_cards("Kc Kd").expect("valid"),
        );
    }
}
