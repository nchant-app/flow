//! High-level engine for repeated timing predictions.
//!
//! `TimingEngine` bundles all resources (language info and timing model)
//! into a single object that can be loaded once and reused across multiple predictions.
//! This is more efficient than loading from files for each inference.
//!
//! # Example
//!
//! ```no_run
//! use nchant_flow::TimingEngine;
//!
//! // The bundled global inventory supplies phoneme types and language info.
//! let engine = TimingEngine::from_paths("timing_model.yaml").unwrap();
//!
//!
//! // Use it for multiple predictions
//! let result1 = engine.predict(&["k".to_string(), "a".to_string()]);
//! let result2 = engine.predict(&["s".to_string(), "t".to_string()]);
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::TimingError;
use crate::model::{LanguageInfo, PhonemeType, TimingModel, TimingResult};
use crate::predict::TimingLookup;
use crate::resources::{load_language_info_from_path, load_timing_model};

/// Shared inner state of the engine (immutable after construction).
struct TimingEngineInner {
    language_info: LanguageInfo,
    lookup: TimingLookup<String>,
}

/// A high-level engine that bundles all resources for timing predictions.
///
/// This struct holds the language info, including phoneme classifications, and lookup structure,
/// allowing efficient repeated predictions without reloading from disk.
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
    /// `None` to use the one bundled with flow.
    pub fn from_paths_with_global(
        model_path: &str,
        global_path: Option<&str>,
    ) -> Result<Self, TimingError> {
        let model = load_timing_model(model_path)?;
        let language_info = load_language_info_from_path(global_path)?;

        Ok(Self::from_components(model, language_info))
    }

    /// Create an engine from pre-loaded components.
    pub fn from_components(model: TimingModel, language_info: LanguageInfo) -> Self {
        let lookup = TimingLookup::from_model(&model, &language_info);

        Self {
            inner: Arc::new(TimingEngineInner {
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

    /// Validate that all phonemes exist in the language inventory.
    pub fn validate(&self, phonemes: &[String]) -> Vec<String> {
        crate::predict::validate_phonemes(phonemes, &self.inner.language_info)
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
            .field("phoneme_count", &self.inner.language_info.phonemes.len())
            .field("language_name", &self.inner.language_info.name)
            .finish()
    }
}

/// Builder for constructing a `TimingEngine` step by step.
pub struct TimingEngineBuilder {
    model: Option<TimingModel>,
    language_info: Option<LanguageInfo>,
    fallbacks: HashMap<PhonemeType, u32>,
}

impl TimingEngineBuilder {
    /// Create a new empty builder.
    pub fn new() -> Self {
        Self {
            model: None,
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

    /// Load the complete language info from a global phoneme file.
    ///
    /// Pass `Some(path)` for a custom global file, or `None` to use the inventory
    /// bundled with flow. This populates the phoneme type classifier
    /// (for fallback) and the vowel / diphthong / syllabic-consonant data in one step.
    pub fn global(mut self, path: Option<&str>) -> Result<Self, TimingError> {
        self.language_info = Some(load_language_info_from_path(path)?);
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
    /// bundled with flow.
    pub fn language_info_from_global(mut self, path: Option<&str>) -> Result<Self, TimingError> {
        self.language_info = Some(load_language_info_from_path(path)?);
        Ok(self)
    }

    /// Override the fallback duration for a phoneme type.
    pub fn fallback(mut self, phoneme_type: PhonemeType, duration_ms: u32) -> Self {
        self.fallbacks.insert(phoneme_type, duration_ms);
        self
    }

    /// Build the engine. Returns an error if required components are missing.
    ///
    pub fn build(self) -> Result<TimingEngine, TimingError> {
        let model = self.model.ok_or_else(|| {
            TimingError::Other("TimingModel is required. Use .model() or .model_path()".to_string())
        })?;
        let language_info = self.language_info.ok_or_else(|| {
            TimingError::Other(
                "LanguageInfo is required. Use .language_info() or .language_info_path()"
                    .to_string(),
            )
        })?;

        let lookup = TimingLookup::from_model(&model, &language_info);
        let inner = TimingEngineInner {
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
#[path = "engine.test.rs"]
mod tests;
