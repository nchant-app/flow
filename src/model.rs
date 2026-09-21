//! Standalone data types for the mai-timing crate.
//!
//! This module contains all the types needed for timing model training and prediction,
//! designed to be independent of mai-shared for open-source distribution.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Phoneme type classification for the generic timing tree.
///
/// This enum categorizes phonemes into broad articulatory classes,
/// enabling fallback timing predictions when exact phoneme sequences
/// aren't found in the training data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum PhonemeType {
    /// Vowel sounds (monophthongs)
    Vowel,
    /// Diphthongs (gliding vowels)
    Diphthong,
    /// Affricates (combined stop + fricative)
    Affricate,
    /// Fricatives (continuous turbulent airflow)
    Fricative,
    /// Plosives/stops (complete closure then release)
    Plosive,
    /// Sonorants (nasals, approximants, laterals)
    Sonorant,
    /// Taps and flaps
    Tap,
    /// Special markers (silence, breath, etc.)
    Special,
    /// Unknown or unclassified
    #[default]
    None,
}

/// A phoneme map that defines available phonemes and their types.
///
/// Maps X-SAMPA phoneme strings to their articulatory type.
/// Can be built from a global.yaml file or constructed manually.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhonemeMap {
    /// Human-readable name for this phoneme map
    pub name: String,
    /// Mapping from X-SAMPA phoneme string to its type
    pub phonemes: HashMap<String, PhonemeType>,
}

impl PhonemeMap {
    /// Create a new empty phoneme map.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            phonemes: HashMap::new(),
        }
    }

    /// Add a phoneme to the map.
    pub fn add_phoneme(&mut self, name: impl Into<String>, phoneme_type: PhonemeType) {
        self.phonemes.insert(name.into(), phoneme_type);
    }

    /// Get the type of a phoneme by name.
    pub fn get_type(&self, name: &str) -> Option<PhonemeType> {
        self.phonemes.get(name).copied()
    }

    /// Check if a phoneme exists in the map.
    pub fn contains(&self, name: &str) -> bool {
        self.phonemes.contains_key(name)
    }
}

/// Language-specific information for timing generation.
///
/// Parsed from a language YAML file that uses X-SAMPA notation.
/// Defines which phonemes are vowels, diphthongs, and sonorants,
/// which is crucial for splitting utterances into consonant clusters during training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageInfo {
    /// Language name (e.g., "English", "Japanese")
    pub name: String,
    /// Plosive X-SAMPA strings.
    pub plosives: Vec<String>,
    /// Affricate X-SAMPA strings.
    pub affricates: Vec<String>,
    /// Fricative X-SAMPA strings.
    pub fricatives: Vec<String>,
    /// Sonorant X-SAMPA strings.
    pub sonorants: Vec<String>,
    /// Tap and flap X-SAMPA strings.
    pub taps: Vec<String>,
    /// Monophthong vowel X-SAMPA strings
    pub vowels: Vec<String>,
    /// Diphthong X-SAMPA strings
    pub diphthongs: Vec<String>,
    /// Mapping from X-SAMPA phonemes to their articulatory types.
    pub phonemes: HashMap<String, PhonemeType>,
}

impl LanguageInfo {
    /// Create new language info.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            plosives: Vec::new(),
            affricates: Vec::new(),
            fricatives: Vec::new(),
            sonorants: Vec::new(),
            taps: Vec::new(),
            vowels: Vec::new(),
            diphthongs: Vec::new(),
            phonemes: HashMap::new(),
        }
    }

    /// Add a vowel phoneme (X-SAMPA string).
    pub fn add_vowel(&mut self, phoneme: impl Into<String>) {
        let phoneme = phoneme.into();
        self.vowels.push(phoneme.clone());
        self.phonemes.insert(phoneme, PhonemeType::Vowel);
    }

    /// Add a diphthong phoneme (X-SAMPA string).
    pub fn add_diphthong(&mut self, phoneme: impl Into<String>) {
        let phoneme = phoneme.into();
        self.diphthongs.push(phoneme.clone());
        self.phonemes.insert(phoneme, PhonemeType::Diphthong);
    }

    /// Add a phoneme and its articulatory type.
    pub fn add_phoneme(&mut self, phoneme: impl Into<String>, phoneme_type: PhonemeType) {
        self.phonemes.insert(phoneme.into(), phoneme_type);
    }

    /// Get the articulatory type of a phoneme.
    pub fn get_type(&self, phoneme: &str) -> Option<PhonemeType> {
        self.phonemes.get(phoneme).copied()
    }

    /// Check whether a phoneme is present in this language inventory.
    pub fn contains(&self, phoneme: &str) -> bool {
        self.phonemes.contains_key(phoneme)
    }

    /// Check if a phoneme is a vowel or diphthong (i.e., a syllable nucleus).
    pub fn is_vowel(&self, phoneme: &str) -> bool {
        self.vowels.iter().any(|v| v == phoneme) || self.diphthongs.iter().any(|d| d == phoneme)
    }
}

/// Metadata for a timing model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingMetadata {
    /// Voice library name (e.g., "ANGEL", "DAMIEN")
    pub library: String,
    /// Language name (e.g., "English", "Japanese")
    pub language: String,
    /// Voice color/style name (e.g., "Default", "Soft")
    pub voice_color: String,
    /// ISO 8601 timestamp when the model was created
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
    /// List of source TextGrid files used for training
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_files: Option<Vec<String>>,
}

impl TimingMetadata {
    /// Create new metadata with required fields.
    pub fn new(
        library: impl Into<String>,
        language: impl Into<String>,
        voice_color: impl Into<String>,
    ) -> Self {
        Self {
            library: library.into(),
            language: language.into(),
            voice_color: voice_color.into(),
            created_at: None,
            source_files: None,
        }
    }
}

/// A single cluster timing entry.
///
/// Represents timing data for a specific sequence of phonemes,
/// collected from multiple observations in the training data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterTiming {
    /// The phoneme sequence (X-SAMPA strings)
    pub phonemes: Vec<String>,
    /// Duration samples in milliseconds.
    /// Each inner Vec represents one observation: [duration_phoneme_1, duration_phoneme_2, ...]
    pub samples: Vec<Vec<u16>>,
}

impl ClusterTiming {
    /// Create a new cluster timing entry.
    pub fn new(phonemes: Vec<String>) -> Self {
        Self {
            phonemes,
            samples: Vec::new(),
        }
    }

    /// Add a duration sample.
    pub fn add_sample(&mut self, durations: Vec<u16>) {
        self.samples.push(durations);
    }
}

/// A single generic timing entry.
///
/// Represents timing data for a sequence of phoneme types (not specific phonemes),
/// used as a fallback when exact phoneme sequences aren't found.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenericTiming {
    /// The phoneme type sequence
    pub types: Vec<PhonemeType>,
    /// Duration samples in milliseconds.
    /// Each inner Vec represents one observation.
    pub samples: Vec<Vec<u16>>,
}

impl GenericTiming {
    /// Create a new generic timing entry.
    pub fn new(types: Vec<PhonemeType>) -> Self {
        Self {
            types,
            samples: Vec::new(),
        }
    }

    /// Add a duration sample.
    pub fn add_sample(&mut self, durations: Vec<u16>) {
        self.samples.push(durations);
    }
}

/// A complete timing model.
///
/// Contains all the timing data learned from TextGrid files,
/// ready for serialization to YAML and use in prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingModel {
    /// Model metadata
    pub metadata: TimingMetadata,
    /// Cluster-specific timing data (exact phoneme matches)
    pub cluster_timings: Vec<ClusterTiming>,
    /// Generic timing data (phoneme type matches)
    pub generic_timings: Vec<GenericTiming>,
}

impl TimingModel {
    /// Create a new empty timing model.
    pub fn new(metadata: TimingMetadata) -> Self {
        Self {
            metadata,
            cluster_timings: Vec::new(),
            generic_timings: Vec::new(),
        }
    }
}

/// Phoneme timing result from prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhonemeTiming {
    /// Phoneme name (X-SAMPA)
    pub phoneme: String,
    /// Predicted duration in milliseconds
    pub duration_ms: u32,
}

/// Result of timing prediction for an utterance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingResult {
    /// Individual phoneme timings
    pub timings: Vec<PhonemeTiming>,
    /// Total duration in milliseconds
    pub total_duration_ms: u32,
}

impl TimingResult {
    /// Create from a list of phoneme-duration pairs.
    pub fn from_pairs(pairs: Vec<(String, u32)>) -> Self {
        let total_duration_ms = pairs.iter().map(|(_, d)| d).sum();
        let timings = pairs
            .into_iter()
            .map(|(phoneme, duration_ms)| PhonemeTiming {
                phoneme,
                duration_ms,
            })
            .collect();
        Self {
            timings,
            total_duration_ms,
        }
    }
}

impl std::fmt::Display for TimingResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for timing in &self.timings {
            writeln!(f, "{}: {}ms", timing.phoneme, timing.duration_ms)?;
        }
        writeln!(f, "---")?;
        write!(f, "Total: {}ms", self.total_duration_ms)
    }
}

/// Input format for utterance prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UtteranceInput {
    /// Phoneme names (X-SAMPA) to predict timings for
    pub phonemes: Vec<String>,
}

#[cfg(test)]
#[path = "model.test.rs"]
mod tests;
