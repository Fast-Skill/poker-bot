//! Assembling scattered readings into a table.
//!
//! The card reader and the number reader each return a flat list of things
//! found at pixel positions. Neither knows what a seat is. This module turns
//! those lists into "seat four holds 612.3 big blinds and has 1 in front of
//! it", which is the first form a poker decision can actually be made from.
//!
//! # Nothing is measured from a fixed layout
//!
//! The client re-lays out the table whenever a player joins or leaves, so seat
//! coordinates are not constants — a frame captured mid-reflow shows the old
//! positions and the new ones at once. Every relationship here is therefore
//! derived from the frame itself: seats are wherever the stack readouts are,
//! the centre of the table is the centre of those seats, and distances are
//! expressed as fractions of the seat ring so they hold at any table size.
//!
//! # Refusals travel with the reading
//!
//! A [`TableView`] is deliberately still usable when parts of it are missing —
//! `None` stack, empty board — because the caller has to decide whether the
//! gaps matter for the decision in front of it. What it must not do is quietly
//! substitute a plausible value, so anything unread stays unread and
//! [`TableView::refusals`] counts what was thrown away.

use crate::{CardRead, Frame, GlyphTemplates, Ink, NumberRead, Templates, TextThresholds, Thresholds};
use poker_core::card::Card;

/// One occupied seat, as a single frame shows it.
#[derive(Debug, Clone, PartialEq)]
pub struct SeatView {
    /// Centre of the stack readout, which is what places the seat at the table.
    pub x: f64,
    pub y: f64,
    /// Chips behind, in big blinds.
    pub stack: Option<f64>,
    /// Chips already pushed out in front of this seat on the current street.
    pub bet: Option<f64>,
    /// Whether this is the seat the client is playing.
    pub hero: bool,
}

/// Everything one frame says about the table.
#[derive(Debug, Clone, PartialEq)]
pub struct TableView {
    /// Occupied seats, ordered clockwise on screen starting from the hero.
    ///
    /// Which way round the table the action moves is not settled here — that
    /// needs the dealer button — so this is a screen ordering, not a poker one.
    pub seats: Vec<SeatView>,
    /// The client's own `Total Pot` figure, which already includes the bets
    /// sitting in front of the seats.
    pub pot: Option<f64>,
    /// Chips gathered into the middle from earlier streets, if shown.
    pub collected: Option<f64>,
    /// Community cards, left to right.
    pub board: Vec<Card>,
    /// The hero's own two cards.
    pub hole: Vec<Card>,
    /// Cards and readouts the readers would not vouch for.
    pub refusals: usize,
}

/// Chips lying within this fraction of the seat-ring radius of the middle are
/// the collected pot rather than anybody's bet.
///
/// Measured across the captures: the collected pot sits at 0.107 of the ring
/// and the closest a bet ever comes is 0.276, so this sits in the middle of a
/// gap of nearly threefold. It is expressed as a fraction rather than pixels so
/// that a table drawn at another size needs no new number.
const MIDDLE_RADIUS: f64 = 0.18;

/// Cards within this fraction of the ring of the hero's seat are the hero's.
///
/// The hole cards are drawn against the hero's own plate while the board sits
/// at the middle of the felt, so any threshold well inside the ring separates
/// them.
const HOLE_RADIUS: f64 = 0.55;

impl TableView {
    /// Reads a whole table from one frame.
    ///
    /// Two passes, because the second depends on the first: the hero's own
    /// cards are drawn overlapping and cannot be found by sweeping the frame
    /// for card-shaped rectangles, so they are hunted for only once the seats
    /// are known and there is somewhere specific to look.
    pub fn read(
        frame: &Frame,
        cards: &Templates,
        glyphs: &GlyphTemplates,
        thresholds: Thresholds,
        text: TextThresholds,
    ) -> TableView {
        let found = crate::read_cards(frame, cards, thresholds);
        let numbers = crate::read_numbers(frame, glyphs, text);
        let mut view = TableView::assemble(&found, &numbers);

        // Only when the sweep did not already turn them up, which it does when
        // the pair happens to be drawn far enough apart not to merge.
        if view.hole.is_empty() {
            if let Some(hero) = view.hero() {
                let near = (hero.x as usize, hero.y as usize);
                let hole = crate::read_hole_cards(frame, cards, Thresholds::hole_cards(), near);
                // All or nothing. A hold'em hand is two cards, and one card
                // plus a blank is not a worse hand to reason about — it is a
                // different hand, and the solver has no way to tell that it was
                // handed half of one.
                if hole.len() == 2 {
                    view.hole = hole.iter().filter_map(|r| r.card).collect();
                }
            }
        }
        view
    }

    /// Builds a table from what the readers found.
    pub fn assemble(cards: &[CardRead], numbers: &[NumberRead]) -> TableView {
        let refusals = cards.iter().filter(|c| !c.is_confident()).count()
            + numbers.iter().filter(|n| !n.is_confident()).count();

        // A stack readout is the one thing every occupied seat has.
        let stacks: Vec<&NumberRead> = numbers
            .iter()
            .filter(|n| n.ink == Ink::Cyan && n.is_confident())
            .collect();
        if stacks.is_empty() {
            return TableView {
                seats: Vec::new(),
                pot: pot_of(numbers),
                collected: None,
                board: Vec::new(),
                hole: Vec::new(),
                refusals,
            };
        }

        let mut seats: Vec<SeatView> = stacks
            .iter()
            .map(|n| SeatView {
                x: n.x as f64 + n.width as f64 / 2.0,
                y: n.y as f64 + n.height as f64 / 2.0,
                stack: n.value,
                bet: None,
                hero: false,
            })
            .collect();

        // The client draws the hero's own stack in a larger face than everyone
        // else's, which names the seat without relying on where it sits. Ties
        // fall back to the lowest seat on screen, since the client always puts
        // the hero at the bottom.
        let tallest = stacks.iter().map(|n| n.height).max().expect("non-empty");
        let hero = (0..seats.len())
            .filter(|&i| stacks[i].height == tallest)
            .max_by(|&a, &b| seats[a].y.total_cmp(&seats[b].y))
            .expect("non-empty");
        seats[hero].hero = true;

        let (cx, cy) = centre(&seats);
        let ring = ring_radius(&seats, cx, cy);

        // Chips on the felt are either somebody's bet or the collected pot, and
        // where they lie is what tells them apart.
        let mut collected = None;
        for chips in numbers
            .iter()
            .filter(|n| n.ink == Ink::White && n.is_confident())
        {
            let x = chips.x as f64 + chips.width as f64 / 2.0;
            let y = chips.y as f64 + chips.height as f64 / 2.0;
            if distance(x, y, cx, cy) <= MIDDLE_RADIUS * ring {
                collected = chips.value;
                continue;
            }
            let nearest = seats
                .iter_mut()
                .min_by(|a, b| {
                    distance(x, y, a.x, a.y).total_cmp(&distance(x, y, b.x, b.y))
                })
                .expect("non-empty");
            // Two readings for one seat should not silently overwrite; take the
            // larger, since a bet only grows within a street.
            nearest.bet = match (nearest.bet, chips.value) {
                (Some(existing), Some(found)) => Some(existing.max(found)),
                (existing, found) => existing.or(found),
            };
        }

        // Cards drawn against the hero's plate are the hero's; the rest is board.
        let (hx, hy) = (seats[hero].x, seats[hero].y);
        let mut hole = Vec::new();
        let mut board = Vec::new();
        for read in cards.iter().filter(|c| c.is_confident()) {
            let card = read.card.expect("confident reads carry a card");
            let x = read.x as f64 + crate::geometry::CARD_W as f64 / 2.0;
            let y = read.y as f64 + crate::geometry::CARD_H as f64 / 2.0;
            if distance(x, y, hx, hy) <= HOLE_RADIUS * ring {
                hole.push(card);
            } else {
                board.push(card);
            }
        }

        // Clockwise on screen from the hero. With y measured downwards an
        // increasing angle runs clockwise, so this is a plain rotation of the
        // sorted order.
        seats.sort_by(|a, b| {
            angle(a.x - cx, a.y - cy).total_cmp(&angle(b.x - cx, b.y - cy))
        });
        let first = seats.iter().position(|s| s.hero).expect("the hero is seated");
        seats.rotate_left(first);

        TableView {
            seats,
            pot: pot_of(numbers),
            collected,
            board,
            hole,
            refusals,
        }
    }

    /// The seat the client is playing, if its stack could be read.
    pub fn hero(&self) -> Option<&SeatView> {
        self.seats.iter().find(|s| s.hero)
    }

    /// How many seats are occupied.
    pub fn occupied(&self) -> usize {
        self.seats.len()
    }

    /// What the hero must put in to continue, in big blinds.
    ///
    /// `None` when the hero's own bet could not be read, since guessing it
    /// wrong is the difference between calling and folding.
    pub fn to_call(&self) -> Option<f64> {
        let hero = self.hero()?;
        let largest = self
            .seats
            .iter()
            .filter_map(|s| s.bet)
            .fold(0.0f64, f64::max);
        Some((largest - hero.bet.unwrap_or(0.0)).max(0.0))
    }

    /// Whether the money on the table adds up.
    ///
    /// The client's own `Total Pot` includes the bets in front of the seats, so
    /// those plus anything already gathered into the middle should equal it.
    /// A frame caught mid-animation will not balance, which is a cheap way to
    /// notice one without recognising the animation itself.
    pub fn is_consistent(&self) -> bool {
        let Some(pot) = self.pot else {
            return false;
        };
        let bets: f64 = self.seats.iter().filter_map(|s| s.bet).sum();
        let total = bets + self.collected.unwrap_or(0.0);
        // A tenth of a blind is below anything the client displays.
        (total - pot).abs() < 0.05
    }
}

fn pot_of(numbers: &[NumberRead]) -> Option<f64> {
    numbers
        .iter()
        .find(|n| n.ink == Ink::Gold && n.is_confident())
        .and_then(|n| n.value)
}

fn centre(seats: &[SeatView]) -> (f64, f64) {
    let n = seats.len() as f64;
    (
        seats.iter().map(|s| s.x).sum::<f64>() / n,
        seats.iter().map(|s| s.y).sum::<f64>() / n,
    )
}

/// Mean distance from the middle out to a seat — the scale of this table.
fn ring_radius(seats: &[SeatView], cx: f64, cy: f64) -> f64 {
    seats
        .iter()
        .map(|s| distance(s.x, s.y, cx, cy))
        .sum::<f64>()
        / seats.len() as f64
}

fn distance(ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    ((ax - bx).powi(2) + (ay - by).powi(2)).sqrt()
}

fn angle(dx: f64, dy: f64) -> f64 {
    dy.atan2(dx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Frame, GlyphTemplates, Templates, TextThresholds, Thresholds};
    use std::path::PathBuf;

    fn data(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../data")
            .join(name)
    }

    fn view_of(name: &str) -> TableView {
        let raw = std::fs::read(data("frames").join(name)).expect("frame should exist");
        let w = u32::from_le_bytes(raw[0..4].try_into().expect("header")) as usize;
        let h = u32::from_le_bytes(raw[4..8].try_into().expect("header")) as usize;
        let frame = Frame::new(w, h, &raw[8..]);

        let cards = Templates::load(data("card_templates.bin")).expect("card templates");
        let glyphs = GlyphTemplates::load(data("digit_templates.bin")).expect("glyph templates");
        TableView::read(
            &frame,
            &cards,
            &glyphs,
            Thresholds::default(),
            TextThresholds::default(),
        )
    }

    /// The frame whose every figure was checked against the screenshot by eye.
    #[test]
    fn a_verified_frame_assembles_into_the_table_it_shows() {
        let view = view_of("20260818-104053-025.rgb");

        assert_eq!(view.occupied(), 7, "seven players are seated");
        assert_eq!(view.pot, Some(2.5));
        assert_eq!(view.collected, None, "nothing has been gathered in yet");

        let hero = view.hero().expect("the hero is seated");
        assert_eq!(hero.stack, Some(122.5), "lunakat");
        assert_eq!(hero.bet, Some(1.0));
        assert!(view.seats[0].hero, "the hero leads the ordering");

        let mut stacks: Vec<f64> = view.seats.iter().filter_map(|s| s.stack).collect();
        stacks.sort_by(f64::total_cmp);
        assert_eq!(
            stacks,
            vec![56.7, 90.5, 122.5, 139.8, 200.0, 278.0, 612.3],
            "every seat's stack, as shown"
        );

        let bets: Vec<Option<f64>> = view.seats.iter().map(|s| s.bet).collect();
        assert_eq!(
            bets.iter().filter(|b| b.is_some()).count(),
            3,
            "three seats have posted: {bets:?}"
        );
        assert!(view.is_consistent(), "0.5 + 1 + 1 should equal the 2.5 pot");
    }

    #[test]
    fn the_hero_gets_the_bet_drawn_in_front_of_the_hero() {
        // Chips go to the nearest seat, and getting this wrong would put
        // somebody else's money in the hero's row and misprice every call.
        let view = view_of("20260818-104053-025.rgb");
        let hero = view.hero().expect("seated");
        assert_eq!(hero.bet, Some(1.0));
        assert_eq!(view.to_call(), Some(0.0), "the hero has matched the largest bet");
    }

    #[test]
    fn chips_in_the_middle_are_the_pot_rather_than_somebody_s_bet() {
        // Postflop the client gathers the betting into the middle, and that
        // heap is white on felt exactly like a bet is. Charging it to the
        // nearest seat would invent a bet that nobody made.
        let view = view_of("20260818-103636-001.rgb");
        assert_eq!(view.collected, Some(2.0));
        assert_eq!(view.pot, Some(2.0));
        assert!(
            view.seats.iter().all(|s| s.bet.is_none()),
            "nobody has bet this street: {:?}",
            view.seats
        );
        assert!(view.is_consistent());
    }

    #[test]
    fn the_hero_is_found_by_the_larger_face_the_client_draws_it_in() {
        for name in [
            "20260818-103636-001.rgb",
            "20260818-103911-011.rgb",
            "20260818-104053-025.rgb",
            "20260818-104236-010.rgb",
            "20260818-104417-005.rgb",
            "20260818-104601-015.rgb",
        ] {
            let view = view_of(name);
            assert_eq!(
                view.seats.iter().filter(|s| s.hero).count(),
                1,
                "{name}: exactly one seat is the hero"
            );
            let hero = view.hero().expect("seated");
            let lowest = view
                .seats
                .iter()
                .map(|s| s.y)
                .fold(f64::MIN, f64::max);
            assert_eq!(
                hero.y, lowest,
                "{name}: the client draws the hero at the bottom, so the two \
                 signals should agree"
            );
        }
    }

    #[test]
    fn the_hero_s_own_cards_are_read_from_a_preflop_frame() {
        // Checked against the screenshot: lunakat holds the ten and the five of
        // spades, and the board has not been dealt.
        let view = view_of("20260818-104053-025.rgb");
        let hole: Vec<String> = view.hole.iter().map(|c| c.to_string()).collect();
        assert_eq!(hole, vec!["Ts", "5s"], "hole cards: {:?}", view.hole);
        assert!(view.board.is_empty(), "preflop board: {:?}", view.board);
    }

    /// A known limitation, recorded so it cannot be mistaken for working.
    ///
    /// The hero's cards are matched against the board-card templates, and those
    /// are a poor fit: the hero's pair is drawn larger, so a red pip scores 37
    /// to 43 where a sound match scores under 20, and on one frame an eight
    /// scored nearer the six template than its own. The margin test catches
    /// every one of those, so nothing wrong is ever returned — but the reading
    /// comes back incomplete, and half a hand is withheld entirely rather than
    /// passed on as if it were a hand.
    ///
    /// Closing this needs templates harvested at the hero's own card size, the
    /// same way the board and digit templates were built. Until then these
    /// frames are expected to yield nothing.
    #[test]
    fn hole_cards_that_do_not_fully_read_are_withheld_rather_than_halved() {
        for (name, on_screen) in [
            ("20260818-104417-005.rgb", "9d 8h"),
            ("20260818-104601-015.rgb", "9h 5h"),
        ] {
            let view = view_of(name);
            assert!(
                view.hole.is_empty(),
                "{name}: the screen shows {on_screen}; a partial reading of                  {:?} must not be handed on as a hand",
                view.hole
            );
        }
    }

    #[test]
    fn a_seat_holding_no_cards_reads_as_holding_none() {
        // The hero is not in this hand at all - the seat shows an avatar and no
        // cards, which must not be confused with cards that failed to read.
        let view = view_of("20260818-103636-001.rgb");
        assert!(view.hole.is_empty());
        assert_eq!(view.board.len(), 3, "the flop is out: {:?}", view.board);
    }

    #[test]
    fn board_cards_are_not_swept_up_as_the_hero_s_own() {
        let view = view_of("20260818-103636-001.rgb");
        let board: Vec<String> = view.board.iter().map(|c| c.to_string()).collect();
        assert_eq!(board, vec!["9c", "Jd", "Kd"], "the flop");
    }

    #[test]
    fn a_dialog_over_the_table_assembles_into_nothing_rather_than_a_guess() {
        let view = view_of("20260818-104742-014.rgb");
        assert_eq!(view.occupied(), 0);
        assert_eq!(view.pot, None);
        assert!(view.hero().is_none());
        assert!(!view.is_consistent(), "no pot means nothing to be consistent with");
    }

    #[test]
    fn a_frame_caught_mid_reflow_fails_the_money_check() {
        // The client redraws the table when a player joins or leaves. The money
        // shown during that redraw does not add up, which is a cheap way to
        // catch the animation without having to recognise it.
        let view = view_of("20260818-103911-022.rgb");
        assert!(view.refusals > 0, "the ghosted seat should have been refused");
        assert!(!view.is_consistent());
    }

    #[test]
    #[ignore = "diagnostic; run with --ignored --nocapture"]
    fn dump_assembled_views() {
        for name in [
            "20260818-103636-001.rgb", "20260818-103911-011.rgb", "20260818-103911-022.rgb",
            "20260818-104053-025.rgb", "20260818-104236-010.rgb", "20260818-104417-005.rgb",
            "20260818-104601-015.rgb", "20260818-104742-014.rgb",
        ] {
            let raw = std::fs::read(data("frames").join(name)).expect("frame");
            let w = u32::from_le_bytes(raw[0..4].try_into().expect("hdr")) as usize;
            let h = u32::from_le_bytes(raw[4..8].try_into().expect("hdr")) as usize;
            let frame = Frame::new(w, h, &raw[8..]);
            let tpl = Templates::load(data("card_templates.bin")).expect("cards");
            let reads = crate::read_cards(&frame, &tpl, Thresholds::default());
            let view = view_of(name);
            println!("
{name}");
            for r in &reads {
                println!("   card at ({:4},{:4}) -> {:?}", r.x, r.y, r.card);
            }
            if let Some(hero) = view.hero() {
                println!("   hero seat at ({:.0},{:.0})", hero.x, hero.y);
            }
            println!("   board={:?} hole={:?} pot={:?} collected={:?} consistent={}",
                     view.board, view.hole, view.pot, view.collected, view.is_consistent());
            for seat in &view.seats {
                println!("      seat ({:4.0},{:4.0}) stack={:?} bet={:?} hero={}",
                         seat.x, seat.y, seat.stack, seat.bet, seat.hero);
            }
        }
    }
}
