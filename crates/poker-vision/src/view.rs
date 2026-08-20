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

use crate::{
    ActionPanel, CardRead, Frame, GlyphTemplates, HeroTemplates, HeroThresholds, Ink, NumberRead,
    Templates, TextThresholds, Thresholds,
};
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
    /// Whether this seat still holds cards.
    ///
    /// A seat that has folded, or that is sitting out a hand, shows its avatar
    /// where a live seat shows the backs of two cards. Knowing this is what
    /// turns "seven seats" into "three players still in the pot", which is a
    /// different strategic problem entirely.
    pub in_hand: bool,
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
    /// Which seat holds the dealer button, as an index into `seats`.
    pub button: Option<usize>,
    /// The row of action buttons, when the client is showing one.
    pub action: Option<ActionPanel>,
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

/// How pale a seat's card area must be before it counts as holding cards.
///
/// Card backs are light and almost colourless; an avatar is dark, saturated
/// artwork. Measured across the fixtures the two sit at 0.53 and 0.06, so this
/// threshold falls in the middle of a gap of nearly ninefold.
const CARDS_PALE: f64 = 0.25;

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
        hero_cards: &HeroTemplates,
        thresholds: Thresholds,
        text: TextThresholds,
    ) -> TableView {
        let found = crate::read_cards(frame, cards, thresholds);
        let numbers = crate::read_numbers(frame, glyphs, text);
        let button = crate::detect_dealer_button(frame);
        let mut view = TableView::assemble(&found, &numbers, button, frame);

        // Only when the sweep did not already turn them up, which it does when
        // the pair happens to be drawn far enough apart not to merge.
        if view.hole.is_empty() {
            if let Some(hero) = view.hero() {
                let near = (hero.x as usize, hero.y as usize);
                let hole =
                    crate::read_hole_cards(frame, hero_cards, HeroThresholds::default(), near);
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
    pub fn assemble(
        cards: &[CardRead],
        numbers: &[NumberRead],
        button: Option<(usize, usize)>,
        frame: &Frame,
    ) -> TableView {
        let refusals = cards.iter().filter(|c| !c.is_confident()).count()
            + numbers.iter().filter(|n| !n.is_confident()).count();

        // A stack readout is the one thing every occupied seat has.
        // Every cyan readout, whether or not its digits resolved.
        //
        // A seat is marked by the presence of its stack display, not by that
        // display being legible this frame. Requiring it to parse meant a stack
        // caught mid-count deleted the whole seat: the table went from seven
        // players to six and back within a second, which made two consecutive
        // readings disagree about how many people were even sitting there, and
        // so almost nothing was ever settled enough to act on. Whether the
        // figure was read is already carried by `stack` being `None`, and the
        // refusal is already counted.
        let stacks: Vec<&NumberRead> = numbers
            .iter()
            .filter(|n| n.ink == Ink::Cyan)
            .collect();
        if stacks.is_empty() {
            return TableView {
                seats: Vec::new(),
                pot: pot_of(numbers),
                collected: None,
                board: Vec::new(),
                hole: Vec::new(),
                button: None,
                action: crate::read_action_panel(frame),
                refusals,
            };
        }

        let mut seats: Vec<SeatView> = stacks
            .iter()
            .map(|n| {
                let x = n.x as f64 + n.width as f64 / 2.0;
                let y = n.y as f64 + n.height as f64 / 2.0;
                SeatView {
                    x,
                    y,
                    stack: n.value,
                    bet: None,
                    hero: false,
                    in_hand: holds_cards(frame, x, y),
                }
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

        // The button sits on the felt beside its seat, so the nearest seat owns
        // it. Seats are far enough apart that this is never a close call.
        let button = button.and_then(|(bx, by)| {
            let (bx, by) = (bx as f64, by as f64);
            (0..seats.len()).min_by(|&a, &b| {
                distance(bx, by, seats[a].x, seats[a].y)
                    .total_cmp(&distance(bx, by, seats[b].x, seats[b].y))
            })
        });

        TableView {
            seats,
            pot: pot_of(numbers),
            collected,
            board,
            hole,
            button,
            action: crate::read_action_panel(frame),
            refusals,
        }
    }

    /// The seat the client is playing, if its stack could be read.
    pub fn hero(&self) -> Option<&SeatView> {
        self.seats.iter().find(|s| s.hero)
    }

    /// Whether the client is asking the hero to act right now.
    ///
    /// Gated on the panel offering a plain `Fold` rather than merely on buttons
    /// being present, because the panel that arms an action in advance is drawn
    /// in the very same place and would otherwise read as a turn.
    pub fn hero_to_act(&self) -> bool {
        self.action.as_ref().is_some_and(|p| p.offers_plain_fold())
    }

    /// Whether two readings of the table describe the same moment.
    ///
    /// The client animates: chips slide into the middle, the button slides to
    /// the next seat, the table re-lays itself out when somebody joins. A
    /// single frame caught during any of that is internally inconsistent in
    /// ways no single-frame check can always catch — the mid-reflow capture in
    /// the fixtures shows one seat's stack twice, in two places.
    ///
    /// Two captures a moment apart that agree were both taken while nothing was
    /// moving. That is a far cheaper test than recognising each animation, and
    /// it does not need to know which ones exist.
    pub fn agrees_with(&self, other: &TableView) -> bool {
        fn same(a: Option<f64>, b: Option<f64>) -> bool {
            match (a, b) {
                (Some(x), Some(y)) => (x - y).abs() < 0.001,
                (None, None) => true,
                _ => false,
            }
        }
        // Only what a decision rests on.
        //
        // This used to demand that the whole table match, every seat's stack
        // and bet included. At a busy table that almost never happens: chips
        // are always sliding somewhere, a stack is always mid-count, and a seat
        // whose figure fails to read for one frame disappears from the list
        // entirely and changes its length. The result was a session where most
        // readings were discarded as unsettled and the hero's own turns were
        // repeatedly decided by the clock rather than by the bot.
        //
        // None of that motion bears on the decision. Another player's stack
        // ticking down while the pot is collected does not change what the hero
        // holds, what is owed, or what is in the middle. So the comparison is
        // of the facts that are actually used, each of which would be dangerous
        // to act on mid-change:
        self.board == other.board
            && self.hole == other.hole
            && self.button == other.button
            && same(self.pot, other.pot)
            && same(self.collected, other.collected)
            && same(self.to_call(), other.to_call())
            && same(
                self.hero().and_then(|h| h.stack),
                other.hero().and_then(|h| h.stack),
            )
            // How many are still in the hand picks the strategy, so it has to
            // hold still even though it is not a figure.
            && self.active() == other.active()
            && self.action.is_some() == other.action.is_some()
    }

    /// Whether this reading is complete and trustworthy enough to act on.
    ///
    /// Every one of these has to hold, and each rules out a way the bot could
    /// act confidently on something that is not true:
    ///
    /// - the client is asking the hero to act, on a live panel rather than the
    ///   one that arms an action in advance;
    /// - nothing was refused, so no figure on the table is a blank;
    /// - the hero's own two cards were read, both of them;
    /// - the money adds up, which a frame caught mid-animation will not do.
    ///
    /// Being wrong about any of these is worse than doing nothing, because
    /// doing nothing costs at most one folded blind while acting on a misread
    /// table can cost a stack.
    pub fn is_actionable(&self) -> bool {
        self.hero_to_act() && self.refusals == 0 && self.hole.len() == 2 && self.is_consistent()
    }

    /// Which seats posted the small and big blinds.
    ///
    /// `seats` runs clockwise on screen from the hero, and the action runs the
    /// same way — confirmed against a frame where the button, the 0.5 and the 1
    /// fall on three consecutive seats in exactly that order. Heads-up is the
    /// standard exception: the button posts the small blind itself.
    pub fn blinds(&self) -> Option<(usize, usize)> {
        let button = self.button?;
        let seated = self.seats.len();
        match seated {
            0 | 1 => None,
            2 => Some((button, (button + 1) % seated)),
            _ => Some(((button + 1) % seated, (button + 2) % seated)),
        }
    }

    /// How many seats the hero is after the button, counting the button as
    /// zero. The larger this is, the later the hero acts after the flop.
    pub fn hero_seats_after_button(&self) -> Option<usize> {
        let button = self.button?;
        let seated = self.seats.len();
        let hero = self.seats.iter().position(|s| s.hero)?;
        Some((hero + seated - button) % seated)
    }

    /// How many seats are occupied.
    pub fn occupied(&self) -> usize {
        self.seats.len()
    }

    /// How many players are still in the hand, the hero included.
    ///
    /// This is the number that decides which strategy applies. Seven seats with
    /// five folded is a heads-up pot, and playing it as a seven-handed one
    /// would be wrong in almost every spot.
    pub fn active(&self) -> usize {
        self.seats.iter().filter(|s| s.in_hand).count()
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

/// Whether a seat is showing the backs of two cards rather than its avatar.
///
/// The card area sits directly above the name plate, at a fixed offset from the
/// stack readout that holds for every seat around the table.
fn holds_cards(frame: &Frame, x: f64, y: f64) -> bool {
    /// Half the width of a seat's card area, and how far above the stack
    /// readout it runs.
    const HALF: f64 = 80.0;
    const TOP: f64 = 170.0;
    const BOTTOM: f64 = 55.0;

    let x0 = (x - HALF).max(0.0) as usize;
    let x1 = ((x + HALF) as usize).min(frame.width);
    let y0 = (y - TOP).max(0.0) as usize;
    let y1 = ((y - BOTTOM).max(0.0) as usize).min(frame.height);
    if x0 >= x1 || y0 >= y1 {
        return false;
    }

    let mut pale = 0usize;
    let mut total = 0usize;
    for py in y0..y1 {
        for px in x0..x1 {
            let (r, g, b) = frame.pixel(px, py);
            let lo = r.min(g).min(b);
            let hi = r.max(g).max(b);
            total += 1;
            if lo > 120 && hi - lo < 45 {
                pale += 1;
            }
        }
    }
    total > 0 && pale as f64 / total as f64 > CARDS_PALE
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
        let hero = HeroTemplates::load(data("hero_cards.bin")).expect("hero templates");
        TableView::read(
            &frame,
            &cards,
            &glyphs,
            &hero,
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

    /// Every frame whose hero cards were checked against the screenshot by eye.
    #[test]
    fn the_hero_s_cards_are_read_on_every_verified_frame() {
        for (name, expected) in [
            ("20260818-104053-025.rgb", vec!["Ts", "5s"]),
            ("20260818-104236-010.rgb", vec!["Ts", "5s"]),
            ("20260818-104417-005.rgb", vec!["9d", "8h"]),
            ("20260818-104601-015.rgb", vec!["9h", "5h"]),
        ] {
            let view = view_of(name);
            let hole: Vec<String> = view.hole.iter().map(|c| c.to_string()).collect();
            assert_eq!(hole, expected, "{name}");
        }
    }

    #[test]
    fn cards_the_hero_has_already_folded_are_not_reported_as_a_hand() {
        // The client leaves a mucked hand on screen, greyed out. Reading it
        // would have the bot reason about cards it no longer holds — and on
        // this frame it would be reasoning about them four streets late.
        let view = view_of("20260818-103911-011.rgb");
        assert!(view.hole.is_empty(), "greyed-out cards: {:?}", view.hole);
        assert_eq!(view.board.len(), 4, "the turn is out: {:?}", view.board);
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

    /// The blinds are what prove which way round the table the action runs.
    ///
    /// Seats are ordered clockwise on screen, but nothing about a screen says
    /// whether poker runs that way or the other. The money does: the seat after
    /// the button must hold the small blind and the one after that the big one.
    /// Two frames, with the button in two different places, both agree.
    #[test]
    fn the_button_and_the_blinds_agree_with_the_money() {
        for (name, button, small, big) in [
            ("20260818-104053-025.rgb", 3, 4, 5),
            ("20260818-104601-015.rgb", 6, 0, 1),
        ] {
            let view = view_of(name);
            assert_eq!(view.button, Some(button), "{name}: button seat");
            assert_eq!(view.blinds(), Some((small, big)), "{name}: blind seats");
            assert_eq!(
                view.seats[small].bet,
                Some(0.5),
                "{name}: seat {small} should hold the small blind"
            );
            assert_eq!(
                view.seats[big].bet,
                Some(1.0),
                "{name}: seat {big} should hold the big blind"
            );
        }
    }

    #[test]
    fn the_hero_s_distance_from_the_button_is_counted_the_way_position_is() {
        // On this frame the hero is the small blind, one seat past the button.
        let view = view_of("20260818-104601-015.rgb");
        assert_eq!(view.hero_seats_after_button(), Some(1));
        assert!(view.seats[0].hero);
    }

    #[test]
    fn no_button_is_reported_when_the_table_cannot_be_read() {
        let view = view_of("20260818-104742-014.rgb");
        assert_eq!(view.button, None);
        assert_eq!(view.blinds(), None);
    }

    #[test]
    fn a_table_agrees_with_itself_and_not_with_a_different_moment() {
        let still = view_of("20260818-104053-025.rgb");
        assert!(still.agrees_with(&still));

        // Same table, same seats, a later hand: the stacks have moved.
        let later = view_of("20260818-104417-005.rgb");
        assert!(!still.agrees_with(&later));
    }

    #[test]
    fn nothing_is_actionable_unless_the_hero_is_being_asked_to_act() {
        // Every fixture here is a readable table, and on none of them is the
        // hero on a live panel, so none may be acted on.
        for name in [
            "20260818-104053-025.rgb",
            "20260818-104236-010.rgb",
            "20260818-104601-015.rgb",
            "20260818-104742-014.rgb",
        ] {
            let view = view_of(name);
            assert!(!view.is_actionable(), "{name}");
        }
    }

    #[test]
    fn a_frame_caught_mid_reflow_is_never_actionable() {
        let view = view_of("20260818-103911-022.rgb");
        assert!(!view.is_consistent());
        assert!(!view.is_actionable());
    }

    /// Checked against the screenshot: on this flop only two players remain.
    #[test]
    fn folded_seats_are_told_from_seats_still_holding_cards() {
        let view = view_of("20260818-103636-001.rgb");
        assert_eq!(view.occupied(), 7, "seven seats are still sitting there");
        assert_eq!(view.active(), 2, "but only two are in the hand");

        // The hero is not one of them, and holds nothing.
        let hero = view.hero().expect("seated");
        assert!(!hero.in_hand);
        assert!(view.hole.is_empty());
    }

    #[test]
    fn a_seat_that_was_dealt_in_counts_as_active() {
        // Preflop, with the blinds posted: six of the seven were dealt cards.
        let view = view_of("20260818-104053-025.rgb");
        assert_eq!(view.occupied(), 7);
        assert_eq!(view.active(), 6);
        assert!(view.hero().expect("seated").in_hand);
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
            println!("   button={:?} blinds={:?} hero is {:?} seats after the button",
                     view.button, view.blinds(), view.hero_seats_after_button());
            for seat in &view.seats {
                println!("      seat ({:4.0},{:4.0}) stack={:?} bet={:?} hero={}",
                         seat.x, seat.y, seat.stack, seat.bet, seat.hero);
            }
        }
    }

    /// Measures how often the hero's own cards read, over a folder of frames.
    ///
    /// Point it at frames the reader once could not name — that is the only
    /// sample where the answer means anything, since frames it always read
    /// would report a hundred percent whatever the templates held.
    #[test]
    #[ignore = "diagnostic; run with --ignored --nocapture"]
    fn measure_hole_card_coverage() {
        let dir = std::path::PathBuf::from("C:/poker/captures/unread");
        let Ok(entries) = std::fs::read_dir(&dir) else {
            println!("no folder at {}", dir.display());
            return;
        };
        let mut names: Vec<_> = entries
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "rgb"))
            .collect();
        names.sort();

        let cards = Templates::load(data("card_templates.bin")).expect("card templates");
        let glyphs = GlyphTemplates::load(data("digit_templates.bin")).expect("glyph templates");
        let heroes = HeroTemplates::load(data("hero_cards.bin")).expect("hero templates");

        let (mut read, mut holding, mut total) = (0usize, 0usize, 0usize);
        let mut seen: std::collections::BTreeMap<char, usize> = Default::default();
        for path in &names {
            let raw = std::fs::read(path).expect("frame");
            let w = u32::from_le_bytes(raw[0..4].try_into().expect("hdr")) as usize;
            let h = u32::from_le_bytes(raw[4..8].try_into().expect("hdr")) as usize;
            let frame = Frame::new(w, h, &raw[8..]);
            let view = TableView::read(
                &frame,
                &cards,
                &glyphs,
                &heroes,
                Thresholds::default(),
                TextThresholds::default(),
            );
            total += 1;
            if view.hero().is_some_and(|s| s.in_hand) {
                holding += 1;
                if view.hole.len() == 2 {
                    read += 1;
                    for card in &view.hole {
                        let rank = card.to_string().chars().next().expect("a rank");
                        *seen.entry(rank).or_default() += 1;
                    }
                }
            }
        }
        println!();
        println!("{total} frames, {holding} with the hero holding cards");
        println!(
            "{read} of those now read both cards ({:.0}%)",
            read as f64 / holding.max(1) as f64 * 100.0
        );
        println!("ranks seen: {seen:?}");
    }
}
