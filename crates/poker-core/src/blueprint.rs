//! Solved strategies, stored and looked up.
//!
//! A solve is expensive and a decision at the table is not. A [`Blueprint`] is
//! the frozen result of a solve: every information set the solver visited, with
//! the average strategy it settled on, in a form that loads in milliseconds and
//! answers a lookup in a few hundred nanoseconds.
//!
//! This is the line between a solver and a bot. A bot cannot solve while the
//! clock runs — it loads a blueprint and reads from it.
//!
//! # Layout
//!
//! Keys are sorted and searched by bisection rather than hashed. Probabilities
//! live in one flat array, indexed by offsets, so a lookup touches two cache
//! lines instead of chasing a hash table across memory. It also serialises
//! directly: the in-memory layout *is* the file layout.
//!
//! # Missing information sets
//!
//! [`Blueprint::strategy`] returns `None` for a key that was never solved,
//! rather than guessing. At the table that will happen — an abstraction never
//! covers every real situation — and the caller must decide what to do about
//! it. Silently returning a uniform strategy would turn a coverage gap into a
//! bot that quietly plays randomly and looks like it is working.

use crate::cfr::{Game, InfoKey, Profile, Solver};
use crate::rng::Rng;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Identifies a serialized blueprint.
const MAGIC: &[u8; 4] = b"PKBP";
/// Bumped whenever the layout changes, so stale files are rejected rather than
/// misread.
const VERSION: u32 = 1;

/// A solved strategy, ready for lookup.
#[derive(Debug, Clone, PartialEq)]
pub struct Blueprint {
    label: String,
    iterations: u64,
    exploitability: Option<f64>,
    /// Sorted, for bisection.
    keys: Vec<InfoKey>,
    /// `offsets[i]..offsets[i + 1]` bounds the probabilities for `keys[i]`.
    /// Always one longer than `keys`.
    offsets: Vec<u32>,
    probabilities: Vec<f32>,
}

impl Blueprint {
    /// Freezes a strategy profile.
    ///
    /// Probabilities are normalised on the way in, so a lookup is directly
    /// usable. An information set whose weights sum to zero — reachable but
    /// never meaningfully trained — becomes uniform rather than a division by
    /// zero.
    pub fn from_profile(profile: &Profile, label: impl Into<String>) -> Blueprint {
        let mut entries: Vec<(InfoKey, &Vec<f64>)> =
            profile.iter().map(|(key, value)| (*key, value)).collect();
        entries.sort_by_key(|(key, _)| *key);

        let mut keys = Vec::with_capacity(entries.len());
        let mut offsets = Vec::with_capacity(entries.len() + 1);
        let mut probabilities = Vec::new();

        for (key, strategy) in entries {
            keys.push(key);
            offsets.push(probabilities.len() as u32);

            let total: f64 = strategy.iter().sum();
            if total > 0.0 {
                probabilities.extend(strategy.iter().map(|p| (p / total) as f32));
            } else {
                let uniform = 1.0 / strategy.len() as f32;
                probabilities.extend(std::iter::repeat(uniform).take(strategy.len()));
            }
        }
        offsets.push(probabilities.len() as u32);

        Blueprint {
            label: label.into(),
            iterations: 0,
            exploitability: None,
            keys,
            offsets,
            probabilities,
        }
    }

    /// Freezes a solver's average strategy, recording how long it trained and —
    /// for two-player games — how exploitable the result is.
    ///
    /// The exploitability measurement walks the whole tree, so it is worth
    /// doing once at save time rather than repeatedly later. It is also
    /// recorded only where it means something. Exploitability is how much a
    /// best responder gains over *the value of the game*, and a three-player
    /// game has no single such value to be measured against: the best response
    /// to one opponent is not the best response to two, and the three payoffs
    /// sum to zero without any being another's negation. Rather than report a
    /// number that reads like a quality score and is not one, a blueprint from
    /// a larger game simply carries none.
    pub fn from_solver<G: Game>(solver: &Solver<G>, label: impl Into<String>) -> Blueprint {
        let profile = solver.profile();
        let exploitability = (solver.game().players() == 2)
            .then(|| solver.exploitability(&profile));
        Blueprint {
            iterations: solver.iterations(),
            exploitability,
            ..Blueprint::from_profile(&profile, label)
        }
    }

    /// What this blueprint solves — game, stakes, stack depth.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Iterations behind this solve, if recorded.
    pub fn iterations(&self) -> u64 {
        self.iterations
    }

    /// How exploitable the stored strategy is, if it was measured.
    pub fn exploitability(&self) -> Option<f64> {
        self.exploitability
    }

    /// Number of information sets stored.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Bytes this blueprint occupies once written.
    pub fn size_on_disk(&self) -> usize {
        let header = MAGIC.len() + 4 + 4 + self.label.len() + 8 + 1 + 8 + 8;
        header
            + self.keys.len() * 8
            + self.offsets.len() * 4
            + self.probabilities.len() * 4
    }

    /// The strategy at `key`, or `None` if it was never solved.
    ///
    /// Deliberately not defaulted: see the module docs.
    pub fn strategy(&self, key: InfoKey) -> Option<&[f32]> {
        let index = self.keys.binary_search(&key).ok()?;
        let start = self.offsets[index] as usize;
        let end = self.offsets[index + 1] as usize;
        Some(&self.probabilities[start..end])
    }

    /// Draws an action at `key` according to the stored frequencies.
    ///
    /// Sampling rather than always taking the most likely action is not
    /// optional: a mixed strategy played greedily is a *different*, and
    /// exploitable, strategy. Bluffing a third of the time and never bluffing
    /// are not the same thing.
    pub fn sample(&self, key: InfoKey, rng: &mut Rng) -> Option<usize> {
        let strategy = self.strategy(key)?;
        let roll = rng.next_f64();
        let mut cumulative = 0.0;
        for (action, probability) in strategy.iter().enumerate() {
            cumulative += *probability as f64;
            if roll < cumulative {
                return Some(action);
            }
        }
        // Only reachable through floating-point drift in the last comparison.
        Some(strategy.len() - 1)
    }

    /// The single most likely action at `key`.
    ///
    /// For inspection and debugging. Playing this at the table discards the
    /// mixing the solve produced — use [`Blueprint::sample`] instead.
    pub fn best_action(&self, key: InfoKey) -> Option<usize> {
        let strategy = self.strategy(key)?;
        strategy
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(action, _)| action)
    }

    /// Every stored information set, in key order.
    pub fn entries(&self) -> impl Iterator<Item = (InfoKey, &[f32])> + '_ {
        self.keys.iter().enumerate().map(move |(index, key)| {
            let start = self.offsets[index] as usize;
            let end = self.offsets[index + 1] as usize;
            (*key, &self.probabilities[start..end])
        })
    }

    /// Writes the blueprint to `path`, creating parent directories as needed.
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = BufWriter::new(File::create(path)?);

        file.write_all(MAGIC)?;
        file.write_all(&VERSION.to_le_bytes())?;

        let label = self.label.as_bytes();
        file.write_all(&(label.len() as u32).to_le_bytes())?;
        file.write_all(label)?;

        file.write_all(&self.iterations.to_le_bytes())?;
        match self.exploitability {
            Some(value) => {
                file.write_all(&[1])?;
                file.write_all(&value.to_le_bytes())?;
            }
            None => {
                file.write_all(&[0])?;
                file.write_all(&0f64.to_le_bytes())?;
            }
        }

        file.write_all(&(self.keys.len() as u64).to_le_bytes())?;
        for key in &self.keys {
            file.write_all(&key.to_le_bytes())?;
        }
        for offset in &self.offsets {
            file.write_all(&offset.to_le_bytes())?;
        }
        for probability in &self.probabilities {
            file.write_all(&probability.to_le_bytes())?;
        }
        file.flush()
    }

    /// Reads a blueprint written by [`Blueprint::save`].
    pub fn load(path: impl AsRef<Path>) -> io::Result<Blueprint> {
        let mut file = BufReader::new(File::open(path)?);

        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(invalid("not a blueprint file"));
        }

        let version = read_u32(&mut file)?;
        if version != VERSION {
            return Err(invalid(format!(
                "blueprint version {version}, expected {VERSION}"
            )));
        }

        let label_len = read_u32(&mut file)? as usize;
        let mut label_bytes = vec![0u8; label_len];
        file.read_exact(&mut label_bytes)?;
        let label = String::from_utf8(label_bytes)
            .map_err(|_| invalid("blueprint label is not valid UTF-8"))?;

        let iterations = read_u64(&mut file)?;
        let mut flag = [0u8; 1];
        file.read_exact(&mut flag)?;
        let measured = read_f64(&mut file)?;
        let exploitability = (flag[0] == 1).then_some(measured);

        let count = read_u64(&mut file)? as usize;
        let mut keys = Vec::with_capacity(count);
        for _ in 0..count {
            keys.push(read_u64(&mut file)?);
        }
        // Sorted order is what makes lookup a bisection; a tampered file that
        // broke it would silently return wrong strategies.
        if keys.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(invalid("blueprint keys are not strictly ascending"));
        }

        let mut offsets = Vec::with_capacity(count + 1);
        for _ in 0..=count {
            offsets.push(read_u32(&mut file)?);
        }
        if offsets.windows(2).any(|pair| pair[0] > pair[1]) {
            return Err(invalid("blueprint offsets are not ascending"));
        }

        let total = *offsets.last().unwrap_or(&0) as usize;
        let mut probabilities = Vec::with_capacity(total);
        for _ in 0..total {
            probabilities.push(read_f32(&mut file)?);
        }

        Ok(Blueprint {
            label,
            iterations,
            exploitability,
            keys,
            offsets,
            probabilities,
        })
    }
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn read_u32(file: &mut impl Read) -> io::Result<u32> {
    let mut word = [0u8; 4];
    file.read_exact(&mut word)?;
    Ok(u32::from_le_bytes(word))
}

fn read_u64(file: &mut impl Read) -> io::Result<u64> {
    let mut word = [0u8; 8];
    file.read_exact(&mut word)?;
    Ok(u64::from_le_bytes(word))
}

fn read_f32(file: &mut impl Read) -> io::Result<f32> {
    let mut word = [0u8; 4];
    file.read_exact(&mut word)?;
    Ok(f32::from_le_bytes(word))
}

fn read_f64(file: &mut impl Read) -> io::Result<f64> {
    let mut word = [0u8; 8];
    file.read_exact(&mut word)?;
    Ok(f64::from_le_bytes(word))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kuhn::{info_key, Kuhn, BET, JACK, KING, QUEEN};
    use std::collections::HashMap;

    fn temp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("poker_core_blueprint_{name}.bin"))
    }

    fn sample_profile() -> Profile {
        let mut profile = HashMap::new();
        profile.insert(10, vec![0.25, 0.75]);
        profile.insert(3, vec![1.0, 0.0, 0.0]);
        profile.insert(7, vec![0.5, 0.5]);
        profile
    }

    #[test]
    fn strategies_are_normalised_on_the_way_in() {
        let mut profile = HashMap::new();
        // Unnormalised weights, as a solver's strategy sums arrive.
        profile.insert(1, vec![30.0, 10.0]);
        let blueprint = Blueprint::from_profile(&profile, "test");

        let strategy = blueprint.strategy(1).expect("stored");
        assert!((strategy[0] - 0.75).abs() < 1e-6);
        assert!((strategy[1] - 0.25).abs() < 1e-6);
        assert!((strategy.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn an_untrained_information_set_becomes_uniform() {
        let mut profile = HashMap::new();
        profile.insert(1, vec![0.0, 0.0, 0.0]);
        let blueprint = Blueprint::from_profile(&profile, "test");

        let strategy = blueprint.strategy(1).expect("stored");
        assert_eq!(strategy, &[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
    }

    #[test]
    fn lookup_finds_stored_sets_and_refuses_unknown_ones() {
        let blueprint = Blueprint::from_profile(&sample_profile(), "test");
        assert_eq!(blueprint.len(), 3);

        assert!(blueprint.strategy(3).is_some());
        assert!(blueprint.strategy(7).is_some());
        assert!(blueprint.strategy(10).is_some());

        // Never solved: the caller has to decide, not this module.
        assert_eq!(blueprint.strategy(11), None);
        assert_eq!(blueprint.strategy(0), None);
        assert_eq!(blueprint.sample(999, &mut Rng::new(1)), None);
        assert_eq!(blueprint.best_action(999), None);
    }

    #[test]
    fn strategies_of_differing_lengths_stay_separate() {
        let blueprint = Blueprint::from_profile(&sample_profile(), "test");
        assert_eq!(blueprint.strategy(3).expect("stored").len(), 3);
        assert_eq!(blueprint.strategy(7).expect("stored").len(), 2);
        assert_eq!(blueprint.strategy(10).expect("stored").len(), 2);
    }

    #[test]
    fn best_action_picks_the_largest_probability() {
        let blueprint = Blueprint::from_profile(&sample_profile(), "test");
        assert_eq!(blueprint.best_action(10), Some(1), "0.75 beats 0.25");
        assert_eq!(blueprint.best_action(3), Some(0));
    }

    #[test]
    fn sampling_reproduces_the_stored_frequencies() {
        let mut profile = HashMap::new();
        profile.insert(1, vec![0.3, 0.7]);
        let blueprint = Blueprint::from_profile(&profile, "test");

        let mut rng = Rng::new(0xBEEF);
        let mut counts = [0u32; 2];
        const TRIALS: u32 = 200_000;
        for _ in 0..TRIALS {
            counts[blueprint.sample(1, &mut rng).expect("stored")] += 1;
        }

        let observed = counts[0] as f64 / TRIALS as f64;
        assert!(
            (observed - 0.3).abs() < 0.01,
            "sampled the first action {observed:.4} of the time, expected 0.30"
        );
    }

    #[test]
    fn a_deterministic_strategy_is_sampled_deterministically() {
        let mut profile = HashMap::new();
        profile.insert(1, vec![1.0, 0.0]);
        let blueprint = Blueprint::from_profile(&profile, "test");

        let mut rng = Rng::new(5);
        for _ in 0..1_000 {
            assert_eq!(blueprint.sample(1, &mut rng), Some(0));
        }
    }

    #[test]
    fn a_blueprint_round_trips_through_disk() {
        let original = Blueprint::from_profile(&sample_profile(), "heads-up 100bb");
        let path = temp("roundtrip");
        original.save(&path).expect("save");

        let loaded = Blueprint::load(&path).expect("load");
        assert_eq!(loaded, original);
        assert_eq!(loaded.label(), "heads-up 100bb");
        for key in [3u64, 7, 10] {
            assert_eq!(loaded.strategy(key), original.strategy(key), "key {key}");
        }

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn metadata_survives_a_round_trip() {
        let mut blueprint = Blueprint::from_profile(&sample_profile(), "labelled");
        blueprint.iterations = 1_234_567;
        blueprint.exploitability = Some(0.0042);

        let path = temp("metadata");
        blueprint.save(&path).expect("save");
        let loaded = Blueprint::load(&path).expect("load");

        assert_eq!(loaded.iterations(), 1_234_567);
        assert_eq!(loaded.exploitability(), Some(0.0042));
        assert_eq!(loaded.label(), "labelled");

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn an_unmeasured_exploitability_stays_absent() {
        let blueprint = Blueprint::from_profile(&sample_profile(), "unmeasured");
        assert_eq!(blueprint.exploitability(), None);

        let path = temp("unmeasured");
        blueprint.save(&path).expect("save");
        assert_eq!(Blueprint::load(&path).expect("load").exploitability(), None);

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn an_empty_blueprint_is_valid() {
        let blueprint = Blueprint::from_profile(&HashMap::new(), "empty");
        assert!(blueprint.is_empty());
        assert_eq!(blueprint.strategy(0), None);

        let path = temp("empty");
        blueprint.save(&path).expect("save");
        let loaded = Blueprint::load(&path).expect("load");
        assert!(loaded.is_empty());

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn a_file_that_is_not_a_blueprint_is_rejected() {
        let path = temp("garbage");
        fs::write(&path, b"this is not a blueprint at all").expect("write");
        let error = Blueprint::load(&path).expect_err("should refuse");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn a_truncated_file_is_rejected() {
        let blueprint = Blueprint::from_profile(&sample_profile(), "truncated");
        let path = temp("truncated");
        blueprint.save(&path).expect("save");

        let bytes = fs::read(&path).expect("read");
        fs::write(&path, &bytes[..bytes.len() / 2]).expect("truncate");
        assert!(Blueprint::load(&path).is_err(), "half a file is not a blueprint");

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn a_wrong_version_is_rejected_rather_than_misread() {
        let blueprint = Blueprint::from_profile(&sample_profile(), "versioned");
        let path = temp("version");
        blueprint.save(&path).expect("save");

        let mut bytes = fs::read(&path).expect("read");
        bytes[4..8].copy_from_slice(&(VERSION + 1).to_le_bytes());
        fs::write(&path, &bytes).expect("write");

        let error = Blueprint::load(&path).expect_err("should refuse");
        assert!(error.to_string().contains("version"), "{error}");

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn unsorted_keys_are_rejected() {
        // A blueprint whose keys are out of order would make bisection return
        // the wrong strategy, silently.
        let blueprint = Blueprint::from_profile(&sample_profile(), "sorted");
        let path = temp("unsorted");
        blueprint.save(&path).expect("save");

        let mut bytes = fs::read(&path).expect("read");
        // The key block starts after magic, version, label header, label,
        // iterations, the exploitability flag and value, and the count.
        let start = 4 + 4 + 4 + blueprint.label.len() + 8 + 1 + 8 + 8;
        bytes[start..start + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        fs::write(&path, &bytes).expect("write");

        let error = Blueprint::load(&path).expect_err("should refuse");
        assert!(error.to_string().contains("ascending"), "{error}");

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn many_keys_all_look_up_correctly() {
        // Exercises the bisection across a realistic key spread.
        let mut profile = HashMap::new();
        for index in 0..5_000u64 {
            let key = index.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            profile.insert(key, vec![index as f64 % 7.0 + 1.0, 2.0]);
        }
        let blueprint = Blueprint::from_profile(&profile, "wide");
        assert_eq!(blueprint.len(), profile.len());

        for (key, expected) in &profile {
            let stored = blueprint.strategy(*key).expect("stored");
            let total: f64 = expected.iter().sum();
            assert!((stored[0] as f64 - expected[0] / total).abs() < 1e-6, "key {key}");
        }
    }

    #[test]
    fn entries_walk_every_stored_set() {
        let blueprint = Blueprint::from_profile(&sample_profile(), "walk");
        let collected: Vec<u64> = blueprint.entries().map(|(key, _)| key).collect();
        assert_eq!(collected, vec![3, 7, 10], "in ascending key order");
        assert_eq!(blueprint.entries().count(), blueprint.len());
    }

    /// The integration case: solve, freeze, save, load, and confirm the loaded
    /// strategy plays the same way the solver did.
    #[test]
    fn a_solved_kuhn_strategy_survives_the_round_trip() {
        let mut solver = Solver::new(Kuhn);
        solver.train(50_000);
        let blueprint = Blueprint::from_solver(&solver, "kuhn poker");

        assert_eq!(blueprint.len(), 12, "every Kuhn information set");
        assert_eq!(blueprint.iterations(), 50_000);
        assert!(
            blueprint.exploitability().expect("measured") < 0.01,
            "the solve should be near equilibrium"
        );

        let path = temp("kuhn");
        blueprint.save(&path).expect("save");
        let loaded = Blueprint::load(&path).expect("load");

        // The known equilibrium must survive storage intact.
        let queen_call = loaded
            .strategy(info_key(QUEEN, 0b1, 1))
            .expect("stored")[BET];
        assert!(
            (queen_call as f64 - 1.0 / 3.0).abs() < 0.02,
            "bluff-catch frequency became {queen_call}"
        );
        let king_bet = loaded.strategy(info_key(KING, 0b0, 1)).expect("stored")[BET];
        assert!(king_bet > 0.95, "a king should still always bet");
        let jack_fold = loaded.strategy(info_key(JACK, 0b1, 1)).expect("stored")[BET];
        assert!(jack_fold < 0.05, "a jack should still always fold");

        // And every set matches the solver exactly.
        for (key, stored) in loaded.entries() {
            let original = solver.average_strategy(key).expect("solver has it");
            for (action, probability) in stored.iter().enumerate() {
                assert!(
                    (*probability as f64 - original[action]).abs() < 1e-6,
                    "key {key} action {action}"
                );
            }
        }

        let _ = fs::remove_file(&path);
    }
}
