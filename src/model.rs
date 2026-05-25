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

/// Derive the `PhonemeType` from an internal enum name string.
///
/// This uses the naming conventions from global.yaml's enum names
/// (e.g., "VoicelessAlveolarPlosive" -> Plosive) to determine the type.
pub fn derive_phoneme_type(enum_name: &str) -> PhonemeType {
    if enum_name == "SILENCE" || enum_name == "BREATH" {
        return PhonemeType::Special;
    }
    if enum_name == "SPEAKER_NOISE" || enum_name == "UNKNOWN_PHONEME" {
        return PhonemeType::None;
    }
    if enum_name.contains("Vowel") {
        return PhonemeType::Vowel;
    }
    if enum_name.contains("Diphthong") {
        return PhonemeType::Diphthong;
    }
    if enum_name.contains("Affricate") {
        return PhonemeType::Affricate;
    }
    if enum_name.contains("Plosive") {
        return PhonemeType::Plosive;
    }
    if enum_name.contains("Fricative") {
        return PhonemeType::Fricative;
    }
    if enum_name.contains("Tap") {
        return PhonemeType::Tap;
    }
    // Sonorants: nasals, approximants, trills, laterals, taps (other than AlveolarTap)
    if enum_name.contains("Nasal")
        || enum_name.contains("Approximant")
        || enum_name.contains("Trill")
        || enum_name.contains("Lateral")
    {
        return PhonemeType::Sonorant;
    }
    PhonemeType::None
}

/// A phoneme map that defines available phonemes and their types.
///
/// Maps X-SAMPA phoneme strings to their articulatory type.
/// Can be built from a global.yaml file or constructed manually.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhonemeMap {
    /// Version of the phoneme map format
    pub version: String,
    /// Human-readable name for this phoneme map
    pub name: String,
    /// Mapping from X-SAMPA phoneme string to its type
    pub phonemes: HashMap<String, PhonemeType>,
}

impl PhonemeMap {
    /// Create a new empty phoneme map.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            version: "1.0".to_string(),
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
/// Defines which phonemes are vowels, diphthongs, and syllabic consonants,
/// which is crucial for splitting utterances into consonant clusters during training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageInfo {
    /// Language name (e.g., "English", "Japanese")
    pub name: String,
    /// Monophthong vowel X-SAMPA strings
    pub vowels: Vec<String>,
    /// Diphthong X-SAMPA strings
    pub diphthongs: Vec<String>,
    /// Diphthong -> extension vowel mapping (e.g., "aI" -> "A")
    pub diphthong_extensions: HashMap<String, String>,
    /// Syllabic consonant X-SAMPA strings (for future use)
    pub syllabic_consonants: Vec<String>,
}

impl LanguageInfo {
    /// Create new language info.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            vowels: Vec::new(),
            diphthongs: Vec::new(),
            diphthong_extensions: HashMap::new(),
            syllabic_consonants: Vec::new(),
        }
    }

    /// Add a vowel phoneme (X-SAMPA string).
    pub fn add_vowel(&mut self, phoneme: impl Into<String>) {
        self.vowels.push(phoneme.into());
    }

    /// Add a diphthong phoneme (X-SAMPA string) with its extension vowel.
    pub fn add_diphthong(&mut self, phoneme: impl Into<String>, extension: impl Into<String>) {
        let p = phoneme.into();
        let e = extension.into();
        self.diphthong_extensions.insert(p.clone(), e);
        self.diphthongs.push(p);
    }

    /// Check if a phoneme is a vowel or diphthong (i.e., a syllable nucleus).
    pub fn is_vowel(&self, phoneme: &str) -> bool {
        self.vowels.iter().any(|v| v == phoneme) || self.diphthongs.iter().any(|d| d == phoneme)
    }
}

/// Metadata for a timing model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingMetadata {
    /// Voice library name (e.g., "Oliver", "Akane")
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
    /// Version of the timing model format
    pub version: String,
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
            version: "1.0".to_string(),
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

/// Input format for utterance prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UtteranceInput {
    /// Phoneme names (X-SAMPA) to predict timings for
    pub phonemes: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phoneme_map() {
        let mut map = PhonemeMap::new("Test Map");
        map.add_phoneme("t", PhonemeType::Plosive);
        map.add_phoneme("a", PhonemeType::Vowel);

        assert_eq!(map.get_type("t"), Some(PhonemeType::Plosive));
        assert_eq!(map.get_type("a"), Some(PhonemeType::Vowel));
        assert_eq!(map.get_type("zzz"), None);
    }

    #[test]
    fn test_language_info() {
        let mut lang = LanguageInfo::new("English");
        lang.add_vowel("{");
        lang.add_vowel("i");
        lang.add_diphthong("aI", "A");

        assert!(lang.is_vowel("{"));
        assert!(lang.is_vowel("aI"));
        assert!(!lang.is_vowel("t"));
    }

    #[test]
    fn test_timing_result() {
        let pairs = vec![("t".to_string(), 100), ("a".to_string(), 300)];
        let result = TimingResult::from_pairs(pairs);

        assert_eq!(result.timings.len(), 2);
        assert_eq!(result.total_duration_ms, 400);
    }

    #[test]
    fn test_derive_phoneme_type() {
        assert_eq!(
            derive_phoneme_type("VoicelessAlveolarPlosive"),
            PhonemeType::Plosive
        );
        assert_eq!(
            derive_phoneme_type("VoicelessPostalveolarAffricate"),
            PhonemeType::Affricate
        );
        assert_eq!(
            derive_phoneme_type("VoicelessLabiodentalFricative"),
            PhonemeType::Fricative
        );
        assert_eq!(
            derive_phoneme_type("CloseFrontUnroundedVowel"),
            PhonemeType::Vowel
        );
        assert_eq!(
            derive_phoneme_type("NasalizedMidFrontUnroundedVowel"),
            PhonemeType::Vowel
        );
        assert_eq!(
            derive_phoneme_type("OpenCentralUnroundedNearcloseNearfrontUnroundedDiphthong"),
            PhonemeType::Diphthong
        );
        assert_eq!(
            derive_phoneme_type("MidFrontUnroundedCloseFrontUnroundedDiphthong"),
            PhonemeType::Diphthong
        );
        assert_eq!(derive_phoneme_type("LabialNasal"), PhonemeType::Sonorant);
        assert_eq!(
            derive_phoneme_type("AlveolarApproximant"),
            PhonemeType::Sonorant
        );
        assert_eq!(derive_phoneme_type("AlveolarTap"), PhonemeType::Tap);
        assert_eq!(
            derive_phoneme_type("PalatalizedAlveolarTap"),
            PhonemeType::Tap
        );
        assert_eq!(derive_phoneme_type("AlveolarTrill"), PhonemeType::Sonorant);
        assert_eq!(derive_phoneme_type("MoraicNasal"), PhonemeType::Sonorant);
        assert_eq!(derive_phoneme_type("SILENCE"), PhonemeType::Special);
        assert_eq!(derive_phoneme_type("BREATH"), PhonemeType::Special);
        assert_eq!(derive_phoneme_type("SPEAKER_NOISE"), PhonemeType::None);
        assert_eq!(
            derive_phoneme_type("GlottalFricative"),
            PhonemeType::Fricative
        );
        assert_eq!(derive_phoneme_type("GlottalPlosive"), PhonemeType::Plosive);
    }

    #[test]
    fn test_derive_phoneme_type_unknown_returns_none() {
        assert_eq!(
            derive_phoneme_type("SomethingCompletelyUnknown"),
            PhonemeType::None
        );
        assert_eq!(derive_phoneme_type(""), PhonemeType::None);
        assert_eq!(derive_phoneme_type("UNKNOWN_PHONEME"), PhonemeType::None);
    }

    #[test]
    fn test_derive_phoneme_type_lateral() {
        assert_eq!(
            derive_phoneme_type("AlveolarLateralApproximant"),
            PhonemeType::Sonorant
        );
        assert_eq!(derive_phoneme_type("VelarLateral"), PhonemeType::Sonorant);
    }

    #[test]
    fn test_phoneme_type_default() {
        assert_eq!(PhonemeType::default(), PhonemeType::None);
    }

    #[test]
    fn test_phoneme_map_contains() {
        let mut map = PhonemeMap::new("Test Map");
        map.add_phoneme("t", PhonemeType::Plosive);

        assert!(map.contains("t"));
        assert!(!map.contains("z"));
    }

    #[test]
    fn test_phoneme_map_overwrite() {
        let mut map = PhonemeMap::new("Test Map");
        map.add_phoneme("t", PhonemeType::Plosive);
        map.add_phoneme("t", PhonemeType::Fricative);

        assert_eq!(map.get_type("t"), Some(PhonemeType::Fricative));
    }

    #[test]
    fn test_language_info_consonant_not_vowel() {
        let mut lang = LanguageInfo::new("English");
        lang.add_vowel("a");
        lang.add_vowel("i");

        assert!(!lang.is_vowel("k"));
        assert!(!lang.is_vowel("s"));
        assert!(!lang.is_vowel(""));
    }

    #[test]
    fn test_language_info_diphthong_extension() {
        let mut lang = LanguageInfo::new("English");
        lang.add_diphthong("aI", "A");
        lang.add_diphthong("eI", "E");

        assert_eq!(lang.diphthong_extensions.get("aI"), Some(&"A".to_string()));
        assert_eq!(lang.diphthong_extensions.get("eI"), Some(&"E".to_string()));
        assert_eq!(lang.diphthong_extensions.get("oU"), None);
    }

    #[test]
    fn test_cluster_timing_multiple_samples() {
        let mut ct = ClusterTiming::new(vec!["k".to_string(), "a".to_string()]);
        ct.add_sample(vec![100, 200]);
        ct.add_sample(vec![110, 210]);
        ct.add_sample(vec![90, 190]);

        assert_eq!(ct.samples.len(), 3);
        assert_eq!(ct.phonemes.len(), 2);
        assert_eq!(ct.samples[0], vec![100, 200]);
        assert_eq!(ct.samples[2], vec![90, 190]);
    }

    #[test]
    fn test_generic_timing_construction() {
        let mut gt = GenericTiming::new(vec![PhonemeType::Plosive, PhonemeType::Vowel]);
        gt.add_sample(vec![100, 300]);

        assert_eq!(gt.types.len(), 2);
        assert_eq!(gt.samples.len(), 1);
        assert_eq!(gt.types[0], PhonemeType::Plosive);
        assert_eq!(gt.types[1], PhonemeType::Vowel);
    }

    #[test]
    fn test_timing_model_new() {
        let metadata = TimingMetadata::new("TestLib", "English", "Default");
        let model = TimingModel::new(metadata);

        assert_eq!(model.version, "1.0");
        assert_eq!(model.metadata.library, "TestLib");
        assert_eq!(model.metadata.language, "English");
        assert_eq!(model.metadata.voice_color, "Default");
        assert!(model.cluster_timings.is_empty());
        assert!(model.generic_timings.is_empty());
    }

    #[test]
    fn test_timing_metadata_optional_fields() {
        let metadata = TimingMetadata::new("Lib", "Lang", "Color");

        assert!(metadata.created_at.is_none());
        assert!(metadata.source_files.is_none());
    }

    #[test]
    fn test_timing_result_empty() {
        let result = TimingResult::from_pairs(vec![]);

        assert_eq!(result.timings.len(), 0);
        assert_eq!(result.total_duration_ms, 0);
    }

    #[test]
    fn test_timing_result_single_phoneme() {
        let pairs = vec![("a".to_string(), 500)];
        let result = TimingResult::from_pairs(pairs);

        assert_eq!(result.timings.len(), 1);
        assert_eq!(result.timings[0].phoneme, "a");
        assert_eq!(result.timings[0].duration_ms, 500);
        assert_eq!(result.total_duration_ms, 500);
    }

    #[test]
    fn test_timing_result_total_is_sum() {
        let pairs = vec![
            ("t".to_string(), 100),
            ("a".to_string(), 300),
            ("k".to_string(), 80),
        ];
        let result = TimingResult::from_pairs(pairs);

        assert_eq!(result.total_duration_ms, 480);
    }

    #[test]
    fn test_timing_model_yaml_roundtrip() {
        let mut model = TimingModel::new(TimingMetadata::new("TestLib", "English", "Default"));

        let mut ct = ClusterTiming::new(vec!["k".to_string(), "a".to_string()]);
        ct.add_sample(vec![95, 280]);
        model.cluster_timings.push(ct);

        let mut gt = GenericTiming::new(vec![PhonemeType::Plosive, PhonemeType::Vowel]);
        gt.add_sample(vec![100, 300]);
        model.generic_timings.push(gt);

        let yaml = serde_yaml::to_string(&model).unwrap();
        let deserialized: TimingModel = serde_yaml::from_str(&yaml).unwrap();

        assert_eq!(deserialized.version, model.version);
        assert_eq!(deserialized.metadata.library, model.metadata.library);
        assert_eq!(deserialized.cluster_timings.len(), 1);
        assert_eq!(deserialized.cluster_timings[0].phonemes, vec!["k", "a"]);
        assert_eq!(deserialized.cluster_timings[0].samples[0], vec![95, 280]);
        assert_eq!(deserialized.generic_timings.len(), 1);
        assert_eq!(
            deserialized.generic_timings[0].types,
            vec![PhonemeType::Plosive, PhonemeType::Vowel]
        );
    }
}
