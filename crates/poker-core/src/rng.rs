//! A small deterministic random number generator.
//!
//! Rolled by hand rather than pulled from a crate for two reasons: the solver
//! and equity code need *reproducible* streams so a failing run can be replayed
//! exactly, and this stays dependency-free.
//!
//! The algorithm is SplitMix64 — fast, statistically sound for simulation work,
//! and trivially seedable. It is **not** cryptographically secure and must not
//! be used where that matters.

/// A seeded SplitMix64 generator.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Creates a generator from `seed`. The same seed always produces the same
    /// stream, which is what makes a failing simulation reproducible.
    pub const fn new(seed: u64) -> Rng {
        Rng { state: seed }
    }

    /// The next 64 random bits.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform value in `0..bound`.
    ///
    /// Uses Lemire's multiply-shift with rejection, so the result is unbiased
    /// rather than merely close. Modulo would skew low values, which matters
    /// when sampling card indices millions of times.
    ///
    /// # Panics
    /// Panics if `bound` is zero.
    #[inline]
    pub fn below(&mut self, bound: u64) -> u64 {
        assert!(bound > 0, "bound must be positive");
        let mut product = (self.next_u64() as u128).wrapping_mul(bound as u128);
        let mut low = product as u64;
        if low < bound {
            // Reject the tail that would otherwise be over-represented.
            let threshold = bound.wrapping_neg() % bound;
            while low < threshold {
                product = (self.next_u64() as u128).wrapping_mul(bound as u128);
                low = product as u64;
            }
        }
        (product >> 64) as u64
    }

    /// A uniform float in `[0, 1)`.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        // 53 bits is exactly the mantissa width, so every value is representable.
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_seed_replays_the_same_stream() {
        let mut a = Rng::new(12345);
        let mut b = Rng::new(12345);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(2);
        let differences = (0..100).filter(|_| a.next_u64() != b.next_u64()).count();
        assert!(differences > 95, "streams should not track each other");
    }

    #[test]
    fn below_stays_in_range() {
        let mut rng = Rng::new(7);
        for bound in [1u64, 2, 3, 52, 1000] {
            for _ in 0..1000 {
                assert!(rng.below(bound) < bound);
            }
        }
    }

    #[test]
    fn below_is_roughly_uniform() {
        let mut rng = Rng::new(99);
        let mut counts = [0u32; 52];
        const TRIALS: u32 = 520_000;
        for _ in 0..TRIALS {
            counts[rng.below(52) as usize] += 1;
        }
        let expected = (TRIALS / 52) as f64;
        for (value, &count) in counts.iter().enumerate() {
            let deviation = (count as f64 - expected).abs() / expected;
            assert!(deviation < 0.05, "value {value} appeared {count} times");
        }
    }

    #[test]
    fn floats_land_in_the_unit_interval() {
        let mut rng = Rng::new(3);
        let mut sum = 0.0;
        const TRIALS: usize = 100_000;
        for _ in 0..TRIALS {
            let x = rng.next_f64();
            assert!((0.0..1.0).contains(&x));
            sum += x;
        }
        let mean = sum / TRIALS as f64;
        assert!((mean - 0.5).abs() < 0.01, "mean was {mean}");
    }

    #[test]
    #[should_panic(expected = "bound must be positive")]
    fn below_zero_is_rejected() {
        Rng::new(1).below(0);
    }
}
