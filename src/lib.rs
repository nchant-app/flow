//! # mai-timing
//!
//! Open-source phoneme timing model for voice synthesis.
//!
//! This crate provides functionality for:
//! - Training timing models from TextGrid files
//! - Predicting phoneme durations from trained models
//! - Working with YAML-based configuration and models
//!
//! ## Features
//!
//! - `cli` - Command-line interface
//! - `train` - TextGrid training functionality
//! - `full` - All features (cli + train)
//!
//! ## Quick Start
//!
//! ```no_run
//! use maghni_timing_model::TimingEngine;
//!
//! // Create an engine once, reusing it for multiple predictions
//! let engine = TimingEngine::from_paths(
//!     "path/to/timing_model.yaml",
//!     "path/to/global.yaml",
//!     "path/to/english.yaml",
//! ).unwrap();
//!
//! // Predict timings for an X-SAMPA phoneme sequence
//! let result = engine.predict(&["k".to_string(), "a".to_string(), "t".to_string()]);
//! println!("Total duration: {}ms", result.total_duration_ms);
//!
//! // Or get raw timing pairs
//! let timings = engine.get_timing(&["s".to_string(), "a".to_string()]);
//! ```

// ============================================================================
// Open-source modules (no mai-shared dependency)
// ============================================================================

/// Phoneme classification trait for generic timing lookups.
pub mod classifier;

/// Generic cluster-splitting for phoneme sequences.
pub mod cluster;

/// Error types for timing operations.
pub mod error;

/// High-level engine that bundles all resources for repeated predictions.
pub mod engine;

/// Data types for timing models, phoneme maps, and language info.
pub mod model;

/// Timing prediction logic and lookup structures.
pub mod predict;

/// Training from TextGrid files.
pub mod train;

/// Command-line interface.
#[cfg(feature = "cli")]
pub mod cli;

// Re-export commonly used types
pub use classifier::PhonemeClassifier;
pub use cluster::{split_into_clusters, Cluster};
pub use engine::TimingEngine;
pub use error::TimingError;
pub use model::{
    derive_phoneme_type, ClusterTiming, GenericTiming, LanguageInfo, PhonemeMap, PhonemeTiming,
    PhonemeType, TimingMetadata, TimingModel, TimingResult, UtteranceInput,
};
pub use predict::TimingLookup;
pub use train::{load_language_info, load_phoneme_map, load_timing_model, save_timing_model};

#[cfg(feature = "train")]
pub use train::{train_from_textgrids, TrainingConfig};
