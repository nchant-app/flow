//! YAML resource loading and model serialization helpers.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::error::TimingError;
use crate::model::{
    ClusterTiming, GenericTiming, LanguageInfo, PhonemeMap, PhonemeType, TimingModel,
};
use crate::predict::TimingLookup;

/// Save a timing model to a YAML file.
pub fn save_timing_model(model: &TimingModel, path: &str) -> Result<(), TimingError> {
    let yaml = serde_yaml::to_string(model).map_err(|e| TimingError::yaml(path, e))?;
    fs::write(path, yaml).map_err(|e| TimingError::io(path, e))
}

/// Load a timing model from a YAML file.
pub fn load_timing_model(path: &str) -> Result<TimingModel, TimingError> {
    let content = fs::read_to_string(path).map_err(|e| TimingError::io(path, e))?;
    serde_yaml::from_str(&content).map_err(|e| TimingError::yaml(path, e))
}

/// Load a timing model from YAML and create a lookup structure.
///
/// # Arguments
/// * `model_path` - Path to the timing model YAML file
/// * `global_path` - Optional path to a global phoneme file for type classification.
///   Pass `None` to use the default inventory bundled with flow.
///
/// # Returns
/// A `TimingLookup` ready for prediction, or a `TimingError`.
pub fn load_timing_lookup(
    model_path: &str,
    global_path: Option<&str>,
) -> Result<TimingLookup<String>, TimingError> {
    let model = load_timing_model(model_path)?;
    let language_info = load_language_info_from_path(global_path)?;

    Ok(TimingLookup::from_model(&model, &language_info))
}

/// Merge two timing models into one.
///
/// Combines cluster and generic timing data from both models.
/// The resulting model uses metadata from the primary model.
/// Duplicate phoneme sequences have their samples concatenated.
pub fn merge_models(primary: &TimingModel, secondary: &TimingModel) -> TimingModel {
    let mut cluster_map: HashMap<Vec<String>, Vec<Vec<u16>>> = HashMap::new();
    let mut generic_map: HashMap<Vec<PhonemeType>, Vec<Vec<u16>>> = HashMap::new();

    for ct in &primary.cluster_timings {
        cluster_map
            .entry(ct.phonemes.clone())
            .or_default()
            .extend(ct.samples.clone());
    }
    for gt in &primary.generic_timings {
        generic_map
            .entry(gt.types.clone())
            .or_default()
            .extend(gt.samples.clone());
    }

    for ct in &secondary.cluster_timings {
        cluster_map
            .entry(ct.phonemes.clone())
            .or_default()
            .extend(ct.samples.clone());
    }
    for gt in &secondary.generic_timings {
        generic_map
            .entry(gt.types.clone())
            .or_default()
            .extend(gt.samples.clone());
    }

    let cluster_timings: Vec<ClusterTiming> = cluster_map
        .into_iter()
        .map(|(phonemes, samples)| ClusterTiming { phonemes, samples })
        .collect();

    let generic_timings: Vec<GenericTiming> = generic_map
        .into_iter()
        .map(|(types, samples)| GenericTiming { types, samples })
        .collect();

    TimingModel {
        metadata: primary.metadata.clone(),
        cluster_timings,
        generic_timings,
    }
}

/// The default global phoneme inventory, embedded at compile time.
///
/// Users may override it by supplying their own global file with the same structure.
const DEFAULT_LANGUAGE_INFO: &str = include_str!("data/global.yaml");

/// Build the complete language info from a parsed global phoneme file.
fn build_language_info_from_raw(name: &str, raw: RawLanguageInfoFile) -> LanguageInfo {
    let mut phonemes = HashMap::new();

    for p in &raw.plosives {
        phonemes.insert(p.clone(), PhonemeType::Plosive);
    }
    for p in &raw.affricates {
        phonemes.insert(p.clone(), PhonemeType::Affricate);
    }
    for p in &raw.fricatives {
        phonemes.insert(p.clone(), PhonemeType::Fricative);
    }
    for p in &raw.sonorants {
        phonemes.insert(p.clone(), PhonemeType::Sonorant);
    }
    for p in &raw.taps {
        phonemes.insert(p.clone(), PhonemeType::Tap);
    }
    for p in &raw.vowels {
        phonemes.insert(p.clone(), PhonemeType::Vowel);
    }
    for diphthong in &raw.diphthongs {
        phonemes.insert(diphthong.clone(), PhonemeType::Diphthong);
    }

    LanguageInfo {
        name: name.to_string(),
        plosives: raw.plosives,
        affricates: raw.affricates,
        fricatives: raw.fricatives,
        sonorants: raw.sonorants,
        taps: raw.taps,
        vowels: raw.vowels,
        diphthongs: raw.diphthongs,
        phonemes,
    }
}

/// Load a `PhonemeMap` from a global phoneme file.
///
/// The global file is the single source of truth for phoneme type classification:
/// each top-level key is a `PhonemeType` (plosives, affricates, fricatives,
/// sonorants, taps, vowels, diphthongs) holding the phonemes of that type.
///
/// Pass `Some(path)` to use a custom global file, or `None` to use the default
/// inventory bundled with flow.
///
/// This compatibility helper is retained for callers migrating to
/// [`load_language_info_from_path`]. New code should load `LanguageInfo` once.
pub fn load_phoneme_map_from_path(path: Option<&str>) -> Result<PhonemeMap, TimingError> {
    let info = load_language_info_from_path(path)?;
    Ok(PhonemeMap {
        name: info.name,
        phonemes: info.phonemes,
    })
}

/// Raw structure for deserializing the global phoneme file.
///
/// Each top-level key is a `PhonemeType`; its list holds the phonemes of that
/// type. This single file is the source of truth for both type classification
/// and the sonorant, vowel, and diphthong data needed to split clusters.
///
/// ```yaml
/// plosives:
///   - "p"
///   - "b"
/// affricates:
///   - "ts"
/// fricatives:
///   - "f"
///   - "s"
/// sonorants:
///   - "m"
///   - "n"
/// taps:
///   - "4"
/// vowels:
///   - "a"
///   - "i"
///   - "aI"
///   - "eI"
/// ```
#[derive(Debug, serde::Deserialize)]
struct RawLanguageInfoFile {
    #[serde(default)]
    plosives: Vec<String>,
    #[serde(default)]
    affricates: Vec<String>,
    #[serde(default)]
    fricatives: Vec<String>,
    #[serde(default)]
    sonorants: Vec<String>,
    #[serde(default)]
    taps: Vec<String>,
    #[serde(default)]
    vowels: Vec<String>,
    #[serde(default)]
    diphthongs: Vec<String>,
}

/// Parse a global phoneme file (or the bundled default when `path` is `None`).
fn load_raw_language_info(path: Option<&str>) -> Result<(&str, RawLanguageInfoFile), TimingError> {
    match path {
        Some(path) => {
            let path_ref = Path::new(path);
            let name = path_ref
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("language");

            let content = fs::read_to_string(path_ref).map_err(|e| TimingError::io(path, e))?;
            let info = serde_yaml::from_str(&content).map_err(|e| TimingError::yaml(path, e))?;
            Ok((name, info))
        }
        None => {
            let info = serde_yaml::from_str(DEFAULT_LANGUAGE_INFO)
                .map_err(|e| TimingError::yaml("<embedded global>", e))?;
            Ok(("global", info))
        }
    }
}

/// Load language info from a global phoneme file.
///
/// The vowels and diphthongs declared in the global file become the vowel
/// boundaries used to split consonant clusters. Sonorants are taken from the
/// same file's `sonorants` list.
///
/// Pass `Some(path)` to use a custom global file, or `None` to use the default
/// inventory bundled with flow.
pub fn load_language_info_from_path(path: Option<&str>) -> Result<LanguageInfo, TimingError> {
    let (name, raw) = load_raw_language_info(path)?;
    Ok(build_language_info_from_raw(name, raw))
}

#[cfg(test)]
#[path = "resources.test.rs"]
mod tests;
