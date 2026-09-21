//! Timing prediction logic.
//!
//! This module provides the runtime lookup structure (`TimingLookup`) that enables
//! efficient phoneme duration prediction from a trained timing model.
//!
//! `TimingLookup` is generic over the phoneme key type `K`, allowing it to work
//! with both string-based keys (for the open-source API) and typed enum keys
//! (for integration with mai-shared's `id::Phoneme`).

use std::fmt::Debug;
use std::hash::Hash;

use crate::classifier::PhonemeClassifier;
use crate::model::{LanguageInfo, PhonemeType, TimingModel, TimingResult};
use crate::tree::{PhonemeTree, TreeNode};

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

/// Runtime lookup structure for efficient timing prediction.
///
/// Generic over the phoneme key type `K`. Use `TimingLookup<String>` (the default)
/// for the open-source string-based API, or `TimingLookup<YourPhonemeEnum>` for
/// typed phoneme keys.
///
/// Built from a `TimingModel` and `LanguageInfo`, this structure provides O(k) lookup
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
    /// Build a TimingLookup from a TimingModel and LanguageInfo.
    ///
    /// The language info provides the mapping from X-SAMPA phoneme names to types,
    /// which is needed for the generic tree fallback.
    pub fn from_model(model: &TimingModel, language_info: &LanguageInfo) -> Self {
        let PhonemeTree {
            cluster_tree,
            generic_tree,
        } = PhonemeTree::from_model(model);

        let classifier = crate::classifier::MapClassifier::new(language_info.phonemes.clone());

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
        TimingResult::from_pairs(pairs.into_iter().map(|(k, v)| (k.clone(), v)).collect())
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
            return cluster.iter().zip(times).map(|(p, t)| (p, t)).collect();
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
            return cluster.iter().zip(times).map(|(p, t)| (p, t)).collect();
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

/// Validate that all phonemes in a sequence exist in the language inventory.
///
/// Returns a vector of phoneme strings that are not found in the map.
/// An empty result means all phonemes are valid.
pub fn validate_phonemes(phonemes: &[String], language_info: &LanguageInfo) -> Vec<String> {
    phonemes
        .iter()
        .filter(|p| !language_info.contains(p))
        .cloned()
        .collect()
}

#[cfg(test)]
#[path = "predict.test.rs"]
mod tests;
