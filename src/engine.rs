//! High-level engine for repeated timing predictions.
//!
//! `TimingEngine` bundles all resources (phoneme map, language info, timing model)
//! into a single object that can be loaded once and reused across multiple predictions.
//! This is more efficient than loading from files for each inference.
//!
//! The engine is `Clone`-cheap (uses `Arc` internally) and `Send + Sync`,
//! making it safe to share across threads.
//!
//! # Example
//!
//! ```no_run
//! use maghni_flow::TimingEngine;
//!
//! // The bundled global inventory supplies phoneme types and language info.
//! let engine = TimingEngine::from_paths("timing_model.yaml").unwrap();
//!
//! // Clone is cheap (Arc reference count bump)
//! let engine2 = engine.clone();
//!
//! // Use it for multiple predictions
//! let result1 = engine.predict(&["k".to_string(), "a".to_string()]);
//! let result2 = engine2.predict(&["s".to_string(), "t".to_string()]);
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::TimingError;
use crate::model::{LanguageInfo, PhonemeMap, PhonemeType, TimingModel, TimingResult};
use crate::predict::TimingLookup;
use crate::train::{load_language_info_from_global, load_phoneme_map_from_global, load_timing_model};

/// Shared inner state of the engine (immutable after construction).
struct TimingEngineInner {
    phoneme_map: PhonemeMap,
    language_info: LanguageInfo,
    lookup: TimingLookup<String>,
}

/// A high-level engine that bundles all resources for timing predictions.
///
/// This struct holds the phoneme map, language info, and timing lookup structure,
/// allowing efficient repeated predictions without reloading from disk.
///
/// `TimingEngine` is cheap to clone (internally uses `Arc`) and is `Send + Sync`,
/// making it safe to share across threads.
#[derive(Clone)]
pub struct TimingEngine {
    inner: Arc<TimingEngineInner>,
    /// Phoneme-type fallback durations (owned per-engine, not shared).
    fallbacks: HashMap<PhonemeType, u32>,
}

/// Default fallback durations (ms) by phoneme type.
fn default_fallbacks() -> HashMap<PhonemeType, u32> {
    let mut m = HashMap::new();
    m.insert(PhonemeType::Affricate, 250);
    m.insert(PhonemeType::Diphthong, 400);
    m.insert(PhonemeType::Vowel, 400);
    m.insert(PhonemeType::Fricative, 200);
    m.insert(PhonemeType::Sonorant, 200);
    m.insert(PhonemeType::Plosive, 100);
    m.insert(PhonemeType::Tap, 50);
    m.insert(PhonemeType::Special, 200);
    m.insert(PhonemeType::None, 0);
    m
}

impl TimingEngine {
    /// Create an engine from a timing model, using the default bundled global inventory.
    ///
    /// The global phoneme file supplies both the phoneme type classifier and the
    /// `LanguageInfo` (vowels, diphthongs, syllabic consonants). To supply a custom
    /// global file, use [`TimingEngine::from_paths_with_global`].
    pub fn from_paths(model_path: &str) -> Result<Self, TimingError> {
        Self::from_paths_with_global(model_path, None)
    }

    /// Create an engine from a timing model with an optional custom global phoneme file.
    ///
    /// Pass `Some(path)` for `global_path` to override the default inventory, or
    /// `None` to use the one bundled with maghni-flow.
    pub fn from_paths_with_global(
        model_path: &str,
        global_path: Option<&str>,
    ) -> Result<Self, TimingError> {
        let model = load_timing_model(model_path)?;
        let phoneme_map = load_phoneme_map_from_global(global_path)?;
        let language_info = load_language_info_from_global(global_path)?;

        Ok(Self::from_components(model, phoneme_map, language_info))
    }

    /// Create an engine from pre-loaded components.
    pub fn from_components(
        model: TimingModel,
        phoneme_map: PhonemeMap,
        language_info: LanguageInfo,
    ) -> Self {
        let lookup = TimingLookup::from_model(&model, &phoneme_map);

        Self {
            inner: Arc::new(TimingEngineInner {
                phoneme_map,
                language_info,
                lookup,
            }),
            fallbacks: default_fallbacks(),
        }
    }

    /// Get timing for a cluster of phonemes.
    pub fn get_timing<'a>(&self, phonemes: &'a [String]) -> Vec<(&'a String, u32)> {
        self.inner.lookup.get_timing(phonemes)
    }

    /// Predict timings for a phoneme sequence.
    pub fn predict(&self, phonemes: &[String]) -> TimingResult {
        self.inner.lookup.predict(phonemes)
    }

    /// Get a reference to the phoneme map.
    pub fn phoneme_map(&self) -> &PhonemeMap {
        &self.inner.phoneme_map
    }

    /// Get a reference to the language info.
    pub fn language_info(&self) -> &LanguageInfo {
        &self.inner.language_info
    }

    /// Get a reference to the underlying timing lookup.
    pub fn lookup(&self) -> &TimingLookup<String> {
        &self.inner.lookup
    }

    /// Get the library name from the model metadata.
    pub fn library(&self) -> &str {
        self.inner.lookup.library()
    }

    /// Get the language name from the model metadata.
    pub fn language(&self) -> &str {
        self.inner.lookup.language()
    }

    /// Get the voice color name from the model metadata.
    pub fn voice_color(&self) -> &str {
        self.inner.lookup.voice_color()
    }

    /// Get the model version.
    pub fn model_version(&self) -> &str {
        self.inner.lookup.model_version()
    }

    /// Get the creation timestamp, if available.
    pub fn created_at(&self) -> Option<&str> {
        self.inner.lookup.created_at()
    }

    /// Get the source file names, if available.
    pub fn source_files(&self) -> Option<&[String]> {
        self.inner.lookup.source_files()
    }

    /// Validate that all phonemes exist in the phoneme map.
    pub fn validate(&self, phonemes: &[String]) -> Vec<String> {
        crate::predict::validate_phonemes(phonemes, &self.inner.phoneme_map)
    }

    /// Predict timings for multiple utterances at once.
    pub fn predict_batch(&self, batches: &[Vec<String>]) -> Vec<TimingResult> {
        batches.iter().map(|p| self.predict(p)).collect()
    }

    /// Override the fallback duration for a phoneme type.
    ///
    /// This affects the cloned engine only (fallbacks are per-engine, not shared).
    pub fn with_fallback(&mut self, phoneme_type: PhonemeType, duration_ms: u32) -> &mut Self {
        self.fallbacks.insert(phoneme_type, duration_ms);
        self
    }

    /// Reload the engine from a timing model, replacing all inner state.
    ///
    /// Uses the default bundled global inventory; for a custom global file use
    /// [`TimingEngine::reload_from_paths_with_global`].
    pub fn reload_from_paths(&mut self, model_path: &str) -> Result<(), TimingError> {
        self.reload_from_paths_with_global(model_path, None)
    }

    /// Reload the engine with an optional custom global phoneme file.
    pub fn reload_from_paths_with_global(
        &mut self,
        model_path: &str,
        global_path: Option<&str>,
    ) -> Result<(), TimingError> {
        let new = Self::from_paths_with_global(model_path, global_path)?;
        self.inner = new.inner;
        self.fallbacks = new.fallbacks;
        Ok(())
    }
}

impl std::fmt::Debug for TimingEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TimingEngine")
            .field("library", &self.library())
            .field("language", &self.language())
            .field("voice_color", &self.voice_color())
            .field("phoneme_count", &self.inner.phoneme_map.phonemes.len())
            .field("language_name", &self.inner.language_info.name)
            .finish()
    }
}

/// Builder for constructing a `TimingEngine` step by step.
pub struct TimingEngineBuilder {
    model: Option<TimingModel>,
    phoneme_map: Option<PhonemeMap>,
    language_info: Option<LanguageInfo>,
    fallbacks: HashMap<PhonemeType, u32>,
}

impl TimingEngineBuilder {
    /// Create a new empty builder.
    pub fn new() -> Self {
        Self {
            model: None,
            phoneme_map: None,
            language_info: None,
            fallbacks: default_fallbacks(),
        }
    }

    /// Set the timing model.
    pub fn model(mut self, model: TimingModel) -> Self {
        self.model = Some(model);
        self
    }

    /// Load the timing model from a YAML file path.
    pub fn model_path(mut self, path: &str) -> Result<Self, TimingError> {
        self.model = Some(load_timing_model(path)?);
        Ok(self)
    }

    /// Set the phoneme map.
    pub fn phoneme_map(mut self, map: PhonemeMap) -> Self {
        self.phoneme_map = Some(map);
        self
    }

    /// Load both the phoneme map and language info from a global phoneme file.
    ///
    /// Pass `Some(path)` for a custom global file, or `None` to use the inventory
    /// bundled with maghni-flow. This populates the phoneme type classifier
    /// (for fallback) and the vowel / diphthong / syllabic-consonant data in one
    /// step — everything the engine needs apart from the timing model itself.
    pub fn global(mut self, path: Option<&str>) -> Result<Self, TimingError> {
        self.phoneme_map = Some(load_phoneme_map_from_global(path)?);
        self.language_info = Some(load_language_info_from_global(path)?);
        Ok(self)
    }

    /// Load just the phoneme map from a global phoneme file.
    ///
    /// Pass `Some(path)` for a custom global file, or `None` to use the inventory
    /// bundled with maghni-flow. The phoneme map drives type-based fallback.
    pub fn phoneme_map_from_global(mut self, path: Option<&str>) -> Result<Self, TimingError> {
        self.phoneme_map = Some(load_phoneme_map_from_global(path)?);
        Ok(self)
    }

    /// Set the language info.
    pub fn language_info(mut self, info: LanguageInfo) -> Self {
        self.language_info = Some(info);
        self
    }

    /// Load just the language info from a global phoneme file.
    ///
    /// Pass `Some(path)` for a custom global file, or `None` to use the inventory
    /// bundled with maghni-flow.
    pub fn language_info_from_global(mut self, path: Option<&str>) -> Result<Self, TimingError> {
        self.language_info = Some(load_language_info_from_global(path)?);
        Ok(self)
    }

    /// Override the fallback duration for a phoneme type.
    pub fn fallback(mut self, phoneme_type: PhonemeType, duration_ms: u32) -> Self {
        self.fallbacks.insert(phoneme_type, duration_ms);
        self
    }

    /// Build the engine. Returns an error if required components are missing.
    ///
    /// `phoneme_map` is optional: if not set, type-based fallback is disabled
    /// (all unknown clusters fall back to default durations by type, which defaults to 0ms
    /// for unclassified phonemes). Set it via `.phoneme_map()` or
    /// `.phoneme_map_from_global()` for accurate fallback.
    pub fn build(self) -> Result<TimingEngine, TimingError> {
        let model = self.model.ok_or_else(|| {
            TimingError::Other(
                "TimingModel is required. Use .model() or .model_path()".to_string(),
            )
        })?;
        let phoneme_map = self
            .phoneme_map
            .unwrap_or_else(|| PhonemeMap::new("empty"));
        let language_info = self.language_info.ok_or_else(|| {
            TimingError::Other(
                "LanguageInfo is required. Use .language_info() or .language_info_path()"
                    .to_string(),
            )
        })?;

        let lookup = TimingLookup::from_model(&model, &phoneme_map);
        let inner = TimingEngineInner {
            phoneme_map,
            language_info,
            lookup,
        };
        Ok(TimingEngine {
            inner: Arc::new(inner),
            fallbacks: self.fallbacks,
        })
    }
}

impl Default for TimingEngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ClusterTiming, TimingMetadata};

    fn create_test_engine() -> TimingEngine {
        let mut model = TimingModel::new(TimingMetadata::new("TestLib", "English", "Default"));

        let mut ct = ClusterTiming::new(vec!["k".to_string(), "s".to_string()]);
        ct.add_sample(vec![95, 110]);
        ct.add_sample(vec![100, 105]);
        model.cluster_timings.push(ct);

        let mut phoneme_map = PhonemeMap::new("Test");
        phoneme_map.add_phoneme("k", PhonemeType::Plosive);
        phoneme_map.add_phoneme("s", PhonemeType::Fricative);
        phoneme_map.add_phoneme("a", PhonemeType::Vowel);

        let mut language_info = LanguageInfo::new("English");
        language_info.add_vowel("a");

        TimingEngine::from_components(model, phoneme_map, language_info)
    }

    #[test]
    fn test_engine_predict() {
        let engine = create_test_engine();
        let result = engine.predict(&["k".to_string(), "s".to_string()]);

        assert_eq!(result.timings.len(), 2);
        assert_eq!(result.timings[0].duration_ms, 97);
        assert_eq!(result.timings[1].duration_ms, 107);
        assert_eq!(result.total_duration_ms, 97 + 107);
    }

    #[test]
    fn test_engine_get_timing() {
        let engine = create_test_engine();
        let phonemes = ["k".to_string(), "s".to_string()];
        let timings = engine.get_timing(&phonemes);

        assert_eq!(timings.len(), 2);
        assert_eq!(*timings[0].0, "k");
        assert_eq!(timings[0].1, 97);
    }

    #[test]
    fn test_engine_metadata() {
        let engine = create_test_engine();
        assert_eq!(engine.library(), "TestLib");
        assert_eq!(engine.language(), "English");
        assert_eq!(engine.voice_color(), "Default");
        assert_eq!(engine.model_version(), "1.0");
    }

    #[test]
    fn test_engine_accessors() {
        let engine = create_test_engine();
        assert_eq!(engine.phoneme_map().name, "Test");
        assert_eq!(engine.language_info().name, "English");
    }

    #[test]
    fn test_engine_clone() {
        let engine = create_test_engine();
        let engine2 = engine.clone();

        let r1 = engine.predict(&["k".to_string(), "s".to_string()]);
        let r2 = engine2.predict(&["k".to_string(), "s".to_string()]);
        assert_eq!(r1.total_duration_ms, r2.total_duration_ms);
    }

    #[test]
    fn test_engine_validate() {
        let engine = create_test_engine();
        let valid = vec!["k".to_string(), "a".to_string()];
        let invalid = vec!["k".to_string(), "zzz".to_string()];

        assert!(engine.validate(&valid).is_empty());
        assert_eq!(engine.validate(&invalid), vec!["zzz"]);
    }

    #[test]
    fn test_engine_predict_batch() {
        let engine = create_test_engine();
        let batches = vec![
            vec!["k".to_string(), "s".to_string()],
            vec!["k".to_string()],
        ];
        let results = engine.predict_batch(&batches);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].total_duration_ms, 97 + 107);
    }

    #[test]
    fn test_engine_with_fallback() {
        let mut engine = create_test_engine();
        engine.with_fallback(PhonemeType::Plosive, 80);
        assert_eq!(engine.fallbacks.get(&PhonemeType::Plosive), Some(&80));
    }

    #[test]
    fn test_engine_with_fallback_clone_isolation() {
        let mut engine = create_test_engine();
        let mut engine2 = engine.clone();
        engine.with_fallback(PhonemeType::Plosive, 80);
        engine2.with_fallback(PhonemeType::Plosive, 90);

        assert_eq!(engine.fallbacks.get(&PhonemeType::Plosive), Some(&80));
        assert_eq!(engine2.fallbacks.get(&PhonemeType::Plosive), Some(&90));
    }

    #[test]
    fn test_engine_builder() {
        let mut model = TimingModel::new(TimingMetadata::new("TestLib", "English", "Default"));
        let mut ct = ClusterTiming::new(vec!["k".to_string()]);
        ct.add_sample(vec![100]);
        model.cluster_timings.push(ct);

        let mut phoneme_map = PhonemeMap::new("Test");
        phoneme_map.add_phoneme("k", PhonemeType::Plosive);

        let language_info = LanguageInfo::new("English");

        let engine = TimingEngineBuilder::new()
            .model(model)
            .phoneme_map(phoneme_map)
            .language_info(language_info)
            .fallback(PhonemeType::Plosive, 90)
            .build()
            .unwrap();

        assert_eq!(engine.library(), "TestLib");
        assert_eq!(engine.fallbacks.get(&PhonemeType::Plosive), Some(&90));
    }

    #[test]
    fn test_engine_builder_missing_model() {
        let result = TimingEngineBuilder::new().build();
        assert!(result.is_err());
    }

    #[test]
    fn test_engine_debug() {
        let engine = create_test_engine();
        let debug = format!("{:?}", engine);
        assert!(debug.contains("TestLib"));
        assert!(debug.contains("English"));
    }
}
