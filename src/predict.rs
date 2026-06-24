//! Timing prediction logic.
//!
//! This module provides the runtime lookup structure (`TimingLookup`) that enables
//! efficient phoneme duration prediction from a trained timing model.
//!
//! `TimingLookup` is generic over the phoneme key type `K`, allowing it to work
//! with both string-based keys (for the open-source API) and typed enum keys
//! (for integration with mai-shared's `id::Phoneme`).

use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

use crate::classifier::PhonemeClassifier;
use crate::error::TimingError;
use crate::model::{PhonemeMap, PhonemeType, TimingModel, TimingResult};

/// Default fallback duration (ms) for affricates when no data is available.
const DEFAULT_AFFRICATE_MS: u32 = 250;
/// Default fallback duration (ms) for diphthongs and vowels when no data is available.
const DEFAULT_VOWEL_MS: u32 = 400;
/// Default fallback duration (ms) for fricatives and sonorants when no data is available.
const DEFAULT_FRICATIVE_MS: u32 = 200;
/// Default fallback duration (ms) for plosives when no data is available.
const DEFAULT_PLOSIVE_MS: u32 = 100;
/// Default fallback duration (ms) for taps when no data is available.
const DEFAULT_TAP_MS: u32 = 50;
/// Default fallback duration (ms) for special phonemes when no data is available.
const DEFAULT_SPECIAL_MS: u32 = 200;

/// Maximum recursion depth for the Level 4 recursive split fallback.
/// Prevents stack overflow on very long unmatched clusters.
const MAX_RECURSION_DEPTH: usize = 16;

/// A node in the phoneme timing tree.
#[derive(Debug, Clone, Default)]
pub struct TreeNode<K> {
    pub children: HashMap<K, TreeNode<K>>,
    /// Duration samples: each inner Vec is one observation [phoneme1_ms, phoneme2_ms, ...]
    pub entries: Vec<Vec<u16>>,
}

/// Runtime lookup structure for efficient timing prediction.
///
/// Generic over the phoneme key type `K`. Use `TimingLookup<String>` (the default)
/// for the open-source string-based API, or `TimingLookup<YourPhonemeEnum>` for
/// typed phoneme keys.
///
/// Built from a `TimingModel` and `PhonemeMap`, this structure provides O(k) lookup
/// where k is the length of the phoneme cluster.
pub struct TimingLookup<K: Eq + Hash + Clone + Debug = String> {
    metadata_library: String,
    metadata_language: String,
    metadata_voice_color: String,
    metadata_version: String,
    metadata_created_at: Option<String>,
    metadata_source_files: Option<Vec<String>>,
    cluster_tree: TreeNode<K>,
    generic_tree: TreeNode<PhonemeType>,
    classifier: Box<dyn PhonemeClassifier<K> + Send + Sync>,
}

impl<K: Eq + Hash + Clone + Debug> Debug for TimingLookup<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TimingLookup")
            .field("library", &self.metadata_library)
            .field("language", &self.metadata_language)
            .field("voice_color", &self.metadata_voice_color)
            .finish()
    }
}

impl TimingLookup<String> {
    /// Build a TimingLookup from a TimingModel and PhonemeMap.
    ///
    /// The phoneme map provides the mapping from X-SAMPA phoneme names to types,
    /// which is needed for the generic tree fallback.
    pub fn from_model(model: &TimingModel, phoneme_map: &PhonemeMap) -> Self {
        let mut cluster_tree = TreeNode::default();
        let mut generic_tree = TreeNode::default();

        // Build cluster tree from cluster_timings
        for timing in &model.cluster_timings {
            let mut node = &mut cluster_tree;
            for phoneme in &timing.phonemes {
                node = node
                    .children
                    .entry(phoneme.clone())
                    .or_insert_with(TreeNode::default);
            }
            node.entries.extend(timing.samples.clone());
        }

        // Build generic tree from generic_timings
        for timing in &model.generic_timings {
            let mut node = &mut generic_tree;
            for ptype in &timing.types {
                node = node
                    .children
                    .entry(*ptype)
                    .or_insert_with(TreeNode::default);
            }
            node.entries.extend(timing.samples.clone());
        }

        let classifier = crate::classifier::MapClassifier::new(phoneme_map.phonemes.clone());

        Self {
            metadata_library: model.metadata.library.clone(),
            metadata_language: model.metadata.language.clone(),
            metadata_voice_color: model.metadata.voice_color.clone(),
            metadata_version: model.version.clone(),
            metadata_created_at: model.metadata.created_at.clone(),
            metadata_source_files: model.metadata.source_files.clone(),
            cluster_tree,
            generic_tree,
            classifier: Box::new(classifier),
        }
    }

    /// Get timing for a cluster and return as a TimingResult.
    pub fn predict(&self, phonemes: &[String]) -> TimingResult {
        let pairs = self.get_timing(phonemes);
        TimingResult::from_pairs(
            pairs
                .into_iter()
                .map(|(k, v)| (k.clone(), v))
                .collect(),
        )
    }
}

impl<K: Eq + Hash + Clone + Debug> TimingLookup<K> {
    /// Build a `TimingLookup` directly from pre-built trees and a classifier.
    ///
    /// This constructor allows using any phoneme key type, bypassing the
    /// string-based `TimingModel` intermediate format.
    pub fn from_trees(
        metadata_library: String,
        metadata_language: String,
        metadata_voice_color: String,
        cluster_tree: TreeNode<K>,
        generic_tree: TreeNode<PhonemeType>,
        classifier: Box<dyn PhonemeClassifier<K> + Send + Sync>,
    ) -> Self {
        Self {
            metadata_library,
            metadata_language,
            metadata_voice_color,
            metadata_version: "1.0".to_string(),
            metadata_created_at: None,
            metadata_source_files: None,
            cluster_tree,
            generic_tree,
            classifier,
        }
    }

    /// Get the library name from metadata.
    pub fn library(&self) -> &str {
        &self.metadata_library
    }

    /// Get the language name from metadata.
    pub fn language(&self) -> &str {
        &self.metadata_language
    }

    /// Get the voice color name from metadata.
    pub fn voice_color(&self) -> &str {
        &self.metadata_voice_color
    }

    /// Get the model version from metadata.
    pub fn model_version(&self) -> &str {
        &self.metadata_version
    }

    /// Get the creation timestamp, if available.
    pub fn created_at(&self) -> Option<&str> {
        self.metadata_created_at.as_deref()
    }

    /// Get the source file names, if available.
    pub fn source_files(&self) -> Option<&[String]> {
        self.metadata_source_files.as_deref()
    }

    /// Get the phoneme type for a given phoneme key.
    fn get_phoneme_type(&self, phoneme: &K) -> PhonemeType {
        self.classifier.classify(phoneme)
    }

    /// Calculate the average timing from multiple samples.
    fn get_average_timing(entries: &[Vec<u16>]) -> Vec<u32> {
        if entries.is_empty() {
            return vec![];
        }

        let num_phonemes = entries[0].len();
        let mut averages = Vec::with_capacity(num_phonemes);

        for i in 0..num_phonemes {
            let sum: u64 = entries.iter().map(|entry| u64::from(entry[i])).sum();
            let avg = sum / entries.len() as u64;
            averages.push(avg as u32);
        }

        averages
    }

    /// Get timing for a cluster of phonemes.
    ///
    /// This implements a 4-level fallback strategy:
    /// 1. Exact match in cluster tree
    /// 2. Type-based match in generic tree
    /// 3. Default timing for single phonemes
    /// 4. Recursive split for unmatched multi-phoneme clusters
    pub fn get_timing<'a>(&self, cluster: &'a [K]) -> Vec<(&'a K, u32)> {
        self.get_timing_inner(cluster, 0)
    }

    /// Inner implementation with depth tracking to prevent stack overflow.
    fn get_timing_inner<'a>(&self, cluster: &'a [K], depth: usize) -> Vec<(&'a K, u32)> {
        if cluster.is_empty() {
            return vec![];
        }

        // Level 1: Try exact match in cluster tree
        let mut node = &self.cluster_tree;
        let mut found = true;

        for phoneme in cluster {
            if let Some(child) = node.children.get(phoneme) {
                node = child;
            } else {
                found = false;
                break;
            }
        }

        if found && !node.entries.is_empty() {
            let times = Self::get_average_timing(&node.entries);
            return cluster
                .iter()
                .zip(times)
                .map(|(p, t)| (p, t))
                .collect();
        }

        // Level 2: Try generic tree (by phoneme type)
        let types: Vec<PhonemeType> = cluster.iter().map(|p| self.get_phoneme_type(p)).collect();

        let mut node = &self.generic_tree;
        let mut found = true;

        for ptype in &types {
            if let Some(child) = node.children.get(ptype) {
                node = child;
            } else {
                found = false;
                break;
            }
        }

        if found && !node.entries.is_empty() {
            let times = Self::get_average_timing(&node.entries);
            return cluster
                .iter()
                .zip(times)
                .map(|(p, t)| (p, t))
                .collect();
        }

        // Level 3: Default timing for single phonemes (also used as fallback when
        // recursion depth is exceeded)
        if cluster.len() == 1 || depth >= MAX_RECURSION_DEPTH {
            return cluster
                .iter()
                .map(|p| {
                    let ptype = self.get_phoneme_type(p);
                    let default_time = match ptype {
                        PhonemeType::Affricate => DEFAULT_AFFRICATE_MS,
                        PhonemeType::Diphthong | PhonemeType::Vowel => DEFAULT_VOWEL_MS,
                        PhonemeType::Fricative | PhonemeType::Sonorant => DEFAULT_FRICATIVE_MS,
                        PhonemeType::Plosive => DEFAULT_PLOSIVE_MS,
                        PhonemeType::Tap => DEFAULT_TAP_MS,
                        PhonemeType::Special => DEFAULT_SPECIAL_MS,
                        PhonemeType::None => 0,
                    };
                    (p, default_time)
                })
                .collect();
        }

        // Level 4: Recursive split for multi-phoneme clusters
        let mut first_part = self.get_timing_inner(&cluster[..cluster.len() - 1], depth + 1);
        let last_part = self.get_timing_inner(&cluster[1..], depth + 1);
        first_part.extend(last_part);
        first_part
    }
}

/// Validate that all phonemes in a sequence exist in the phoneme map.
///
/// Returns a vector of phoneme strings that are not found in the map.
/// An empty result means all phonemes are valid.
pub fn validate_phonemes(phonemes: &[String], phoneme_map: &PhonemeMap) -> Vec<String> {
    phonemes
        .iter()
        .filter(|p| !phoneme_map.contains(p))
        .cloned()
        .collect()
}

/// Load a timing model from YAML and create a lookup structure.
///
/// # Arguments
/// * `model_path` - Path to the timing model YAML file
/// * `global_path` - Optional path to a global phoneme file for type classification.
///   Pass `None` to use the default inventory bundled with maghni-timing.
///
/// # Returns
/// A `TimingLookup` ready for prediction, or a `TimingError`.
pub fn load_timing_lookup(
    model_path: &str,
    global_path: Option<&str>,
) -> Result<TimingLookup<String>, TimingError> {
    let model_content =
        std::fs::read_to_string(model_path).map_err(|e| TimingError::io(model_path, e))?;
    let model: TimingModel =
        serde_yaml::from_str(&model_content).map_err(|e| TimingError::yaml(model_path, e))?;

    let phoneme_map = crate::train::load_phoneme_map_from_global(global_path)?;

    Ok(TimingLookup::from_model(&model, &phoneme_map))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ClusterTiming, GenericTiming, TimingMetadata};

    fn create_test_model() -> TimingModel {
        let mut model = TimingModel::new(TimingMetadata::new("TestLib", "English", "Default"));

        // Cluster timings contain consonants only (vowels are never in the model).
        // E.g. [k, s] represents a consonant cluster between two vowels.
        let mut ct = ClusterTiming::new(vec!["k".to_string(), "s".to_string()]);
        ct.add_sample(vec![95, 110]);
        ct.add_sample(vec![100, 105]);
        ct.add_sample(vec![90, 115]);
        model.cluster_timings.push(ct);

        let mut ct2 = ClusterTiming::new(vec!["t".to_string()]);
        ct2.add_sample(vec![80]);
        ct2.add_sample(vec![90]);
        ct2.add_sample(vec![85]);
        model.cluster_timings.push(ct2);

        // Generic timings by phoneme type (also consonant-only)
        let mut gt = GenericTiming::new(vec![PhonemeType::Plosive, PhonemeType::Fricative]);
        gt.add_sample(vec![100, 120]);
        gt.add_sample(vec![90, 130]);
        model.generic_timings.push(gt);

        model
    }

    fn create_test_phoneme_map() -> PhonemeMap {
        let mut map = PhonemeMap::new("Test");
        map.add_phoneme("k", PhonemeType::Plosive);
        map.add_phoneme("t", PhonemeType::Plosive);
        map.add_phoneme("s", PhonemeType::Fricative);
        map.add_phoneme("f", PhonemeType::Fricative);
        map.add_phoneme("a", PhonemeType::Vowel);
        map
    }

    #[test]
    fn test_exact_match() {
        let model = create_test_model();
        let map = create_test_phoneme_map();
        let lookup = TimingLookup::from_model(&model, &map);

        let cluster = ["k".to_string(), "s".to_string()];
        let result = lookup.get_timing(&cluster);

        assert_eq!(result.len(), 2);
        // Average of [95, 100, 90] = 95
        assert_eq!(result[0].1, 95);
        // Average of [110, 105, 115] = 110
        assert_eq!(result[1].1, 110);
    }

    #[test]
    fn test_generic_fallback() {
        let model = create_test_model();
        let map = create_test_phoneme_map();
        let lookup = TimingLookup::from_model(&model, &map);

        // Use a plosive+fricative combination that's not in cluster_timings
        // but matches the generic pattern [Plosive, Fricative]
        let cluster = ["t".to_string(), "f".to_string()];
        let result = lookup.get_timing(&cluster);

        assert_eq!(result.len(), 2);
        // Average of [100, 90] = 95
        assert_eq!(result[0].1, 95);
        // Average of [120, 130] = 125
        assert_eq!(result[1].1, 125);
    }

    #[test]
    fn test_single_phoneme_exact() {
        let model = create_test_model();
        let map = create_test_phoneme_map();
        let lookup = TimingLookup::from_model(&model, &map);

        // Single plosive "t" has exact cluster data: average of [80, 90, 85] = 85
        let cluster = ["t".to_string()];
        let result = lookup.get_timing(&cluster);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].1, 85);
    }

    #[test]
    fn test_single_phoneme_default() {
        let model = create_test_model();
        let map = create_test_phoneme_map();
        let lookup = TimingLookup::from_model(&model, &map);

        // Single fricative "f" has no cluster data, falls to generic,
        // then to default: Fricative => 200ms
        let cluster = ["f".to_string()];
        let result = lookup.get_timing(&cluster);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].1, 200);
    }

    #[test]
    fn test_predict() {
        let model = create_test_model();
        let map = create_test_phoneme_map();
        let lookup = TimingLookup::from_model(&model, &map);

        let result = lookup.predict(&["k".to_string(), "s".to_string()]);

        assert_eq!(result.timings.len(), 2);
        assert_eq!(result.total_duration_ms, 95 + 110);
    }

    #[test]
    fn test_generic_from_trees() {
        // Test the from_trees constructor with i32 keys
        use crate::classifier::PhonemeClassifier;

        struct I32Classifier;
        impl PhonemeClassifier<i32> for I32Classifier {
            fn classify(&self, phoneme: &i32) -> PhonemeType {
                if *phoneme < 10 {
                    PhonemeType::Plosive
                } else {
                    PhonemeType::Fricative
                }
            }
        }

        let mut cluster_tree = TreeNode::default();
        let mut node_1 = TreeNode::default();
        node_1.entries.push(vec![100]);
        cluster_tree.children.insert(1, node_1);

        let generic_tree = TreeNode::default();

        let lookup = TimingLookup::from_trees(
            "TestLib".into(),
            "TestLang".into(),
            "Default".into(),
            cluster_tree,
            generic_tree,
            Box::new(I32Classifier),
        );

        let result = lookup.get_timing(&[1]);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].1, 100);

        // Test fallback to default for unknown key
        let result = lookup.get_timing(&[5]);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].1, DEFAULT_PLOSIVE_MS);
    }

    #[test]
    fn test_metadata_accessors() {
        let model = create_test_model();
        let map = create_test_phoneme_map();
        let lookup = TimingLookup::from_model(&model, &map);

        assert_eq!(lookup.library(), "TestLib");
        assert_eq!(lookup.language(), "English");
        assert_eq!(lookup.voice_color(), "Default");
        assert_eq!(lookup.model_version(), "1.0");
        assert!(lookup.created_at().is_none());
        assert!(lookup.source_files().is_none());
    }

    #[test]
    fn test_validate_phonemes() {
        let map = create_test_phoneme_map();
        let valid = vec!["k".to_string(), "a".to_string()];
        let invalid = vec!["k".to_string(), "zzz".to_string(), "a".to_string()];

        assert!(validate_phonemes(&valid, &map).is_empty());
        let issues = validate_phonemes(&invalid, &map);
        assert_eq!(issues, vec!["zzz"]);
    }
}
