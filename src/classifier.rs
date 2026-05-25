//! Phoneme classification trait for generic timing lookups.
//!
//! The `PhonemeClassifier` trait allows `TimingLookup` to work with any
//! phoneme key type by providing a way to map phonemes to their articulatory
//! `PhonemeType` for the generic fallback tree.

use crate::model::PhonemeType;

/// Classifies phonemes into articulatory types.
///
/// Implement this trait for your phoneme key type to use `TimingLookup<K>`
/// with the generic tree fallback.
pub trait PhonemeClassifier<K> {
    /// Returns the `PhonemeType` for the given phoneme.
    fn classify(&self, phoneme: &K) -> PhonemeType;
}

/// Built-in classifier for `String` keys using a `PhonemeMap`.
///
/// This is the default classifier used when building a `TimingLookup<String>`
/// from a `TimingModel`.
pub struct MapClassifier {
    types: std::collections::HashMap<String, PhonemeType>,
}

impl MapClassifier {
    /// Create a new classifier from a phoneme type map.
    pub fn new(types: std::collections::HashMap<String, PhonemeType>) -> Self {
        Self { types }
    }
}

impl PhonemeClassifier<String> for MapClassifier {
    fn classify(&self, phoneme: &String) -> PhonemeType {
        self.types.get(phoneme).copied().unwrap_or(PhonemeType::None)
    }
}
