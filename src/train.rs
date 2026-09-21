//! Training module for building timing models from TextGrid files.
//!
//! This module provides functionality to extract phoneme timing data from
//! Praat TextGrid files and build a `TimingModel`.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::error::TimingError;
#[cfg(feature = "train")]
use crate::model::TimingMetadata;
use crate::model::{
    ClusterTiming, GenericTiming, LanguageInfo, PhonemeMap, PhonemeType, TimingModel,
};

#[cfg(feature = "train")]
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
#[cfg(feature = "train")]
use textgridde_rs::textgrid;

/// Special labels that indicate silence in TextGrid files.
const SILENCE_LABELS: &[&str] = &["Sil", "sil", "sp", "SP", ""];

/// Labels that indicate speaker noise (to be excluded).
const NOISE_LABELS: &[&str] = &["<noise>", "NOISE", "noise", "<spn>"];

/// Check if a label represents silence.
fn is_silence(label: &str) -> bool {
    SILENCE_LABELS.contains(&label)
}

/// Check if a label represents noise.
fn is_noise(label: &str) -> bool {
    NOISE_LABELS.contains(&label)
}

/// A parsed phoneme with its duration.
#[derive(Debug, Clone)]
struct ParsedPhoneme {
    label: String,
    duration_ms: u16,
}

/// Training configuration.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Minimum duration in ms to include (filters out very short phonemes)
    pub min_duration_ms: u16,
    /// Maximum duration in ms to include (filters out outliers)
    pub max_duration_ms: u16,
}

impl TrainingConfig {
    /// Default training configuration, usable in const contexts.
    pub const DEFAULT: Self = Self {
        min_duration_ms: 10,
        max_duration_ms: 2000,
    };
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Build a timing model from TextGrid files in a directory.
///
/// # Arguments
/// * `textgrid_dir` - Path to directory containing TextGrid files
/// * `language_info` - Language-specific info including vowel/diphthong lists
/// * `label_map` - Optional label translator mapping TextGrid labels to language file labels.
///   Pass `None` if your TextGrid files already use the same labels as the language file.
/// * `metadata` - Metadata for the output model
/// * `config` - Optional training configuration
///
/// # Returns
/// A `TimingModel` containing the extracted timing data, or an error.
#[cfg(feature = "train")]
pub fn train_from_textgrids(
    textgrid_dir: &str,
    language_info: &LanguageInfo,
    label_map: Option<&HashMap<String, String>>,
    metadata: TimingMetadata,
    config: Option<TrainingConfig>,
) -> Result<TimingModel, TimingError> {
    let config = config.unwrap_or_default();
    let dir_path = Path::new(textgrid_dir);

    if !dir_path.is_dir() {
        return Err(TimingError::NotADirectory {
            path: textgrid_dir.to_string(),
        });
    }

    // Collect all TextGrid files
    let textgrid_files: Vec<_> = fs::read_dir(dir_path)
        .map_err(|e| TimingError::io(textgrid_dir, e))?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry
                .path()
                .extension()
                .map(|ext| ext.eq_ignore_ascii_case("textgrid"))
                .unwrap_or(false)
        })
        .map(|entry| entry.path())
        .collect();

    if textgrid_files.is_empty() {
        return Err(TimingError::NoTextGridFiles {
            dir: textgrid_dir.to_string(),
        });
    }

    // Process all TextGrid files in parallel
    let all_clusters: Vec<Vec<ParsedPhoneme>> = textgrid_files
        .par_iter()
        .flat_map(
            |path| match parse_textgrid(path, language_info, &config, label_map) {
                Ok(clusters) => clusters,
                Err(e) => {
                    eprintln!("Warning: Failed to parse {:?}: {}", path, e);
                    vec![]
                }
            },
        )
        .collect();

    // Build cluster and generic timing maps
    let mut cluster_map: HashMap<Vec<String>, Vec<Vec<u16>>> = HashMap::new();
    let mut generic_map: HashMap<Vec<PhonemeType>, Vec<Vec<u16>>> = HashMap::new();

    for cluster in all_clusters {
        if cluster.is_empty() {
            continue;
        }

        let labels: Vec<String> = cluster.iter().map(|p| p.label.clone()).collect();
        let durations: Vec<u16> = cluster.iter().map(|p| p.duration_ms).collect();

        // Add to cluster map
        cluster_map
            .entry(labels.clone())
            .or_default()
            .push(durations.clone());

        // Add to generic map
        let types: Vec<PhonemeType> = labels
            .iter()
            .map(|label| language_info.get_type(label).unwrap_or(PhonemeType::None))
            .collect();
        generic_map.entry(types).or_default().push(durations);
    }

    // Convert maps to timing model format
    let cluster_timings: Vec<ClusterTiming> = cluster_map
        .into_iter()
        .map(|(phonemes, samples)| ClusterTiming { phonemes, samples })
        .collect();

    let generic_timings: Vec<GenericTiming> = generic_map
        .into_iter()
        .map(|(types, samples)| GenericTiming { types, samples })
        .collect();

    // Build metadata with source files
    let mut metadata = metadata;
    metadata.source_files = Some(
        textgrid_files
            .iter()
            .filter_map(|p| p.file_name())
            .filter_map(|n| n.to_str())
            .map(String::from)
            .collect(),
    );
    metadata.created_at = Some(chrono_timestamp());

    Ok(TimingModel {
        version: "1.0".to_string(),
        metadata,
        cluster_timings,
        generic_timings,
    })
}

/// Parse a single TextGrid file and extract phoneme clusters.
#[cfg(feature = "train")]
fn parse_textgrid(
    path: &Path,
    language_info: &LanguageInfo,
    config: &TrainingConfig,
    label_map: Option<&HashMap<String, String>>,
) -> Result<Vec<Vec<ParsedPhoneme>>, TimingError> {
    let path_str = path.display().to_string();
    let textgrid = textgridde_rs::parse_textgrid(path.to_path_buf(), true).map_err(|e| {
        TimingError::TextGrid {
            path: path_str.clone(),
            reason: format!("{:?}", e),
        }
    })?;

    // Get the last tier (assumed to be the phoneme tier)
    let tier = textgrid
        .tiers()
        .last()
        .ok_or_else(|| TimingError::TextGrid {
            path: path_str.clone(),
            reason: "TextGrid has no tiers".to_string(),
        })?;

    let intervals = match tier {
        textgrid::Tier::IntervalTier(t) => t.intervals(),
        textgrid::Tier::PointTier(_) => {
            return Err(TimingError::TextGrid {
                path: path_str,
                reason: "expected IntervalTier, got PointTier".to_string(),
            });
        }
    };

    // Parse all phonemes
    let mut phonemes: Vec<ParsedPhoneme> = Vec::new();
    for interval in intervals {
        let raw_label = interval.text().to_string();
        let duration_secs = interval.get_duration();
        let duration_ms = (duration_secs * 1000.0) as u16;

        // Skip if duration is out of bounds
        if duration_ms < config.min_duration_ms || duration_ms > config.max_duration_ms {
            continue;
        }

        // Translate label if a label map is provided
        let label = label_map
            .and_then(|m| m.get(&raw_label))
            .cloned()
            .unwrap_or(raw_label);

        // Warn if phoneme not in map (but still include it)
        if !is_silence(&label) && !is_noise(&label) && !language_info.contains(&label) {
            eprintln!(
                "Warning: Phoneme '{}' not found in phoneme map, treating as unknown",
                label
            );
        }

        phonemes.push(ParsedPhoneme { label, duration_ms });
    }

    // Split by silence
    let utterances: Vec<Vec<ParsedPhoneme>> = phonemes
        .split(|p| is_silence(&p.label))
        .map(|s| s.to_vec())
        .collect();

    // Split each utterance by vowels
    let mut all_clusters: Vec<Vec<ParsedPhoneme>> = Vec::new();

    for utterance in utterances {
        if utterance.is_empty() {
            continue;
        }

        // Split by vowels (vowels become cluster boundaries)
        let clusters = split_by_vowels(&utterance, language_info);
        all_clusters.extend(clusters);
    }

    // Filter out invalid clusters
    all_clusters.retain(|cluster| {
        !cluster.is_empty()
            && cluster
                .iter()
                .all(|p| !is_silence(&p.label) && !is_noise(&p.label))
    });

    Ok(all_clusters)
}

/// Split a phoneme sequence into consonant clusters by splitting at vowels.
///
/// Vowels and diphthongs act as boundaries but are **not** included in the
/// output clusters. Only consonants are kept — the timing model predicts
/// exclusively consonant durations.
///
/// Example: `[k, a, t, s, i]` -> `[k]`, `[t, s]`
fn split_by_vowels(
    phonemes: &[ParsedPhoneme],
    language_info: &LanguageInfo,
) -> Vec<Vec<ParsedPhoneme>> {
    let mut clusters: Vec<Vec<ParsedPhoneme>> = Vec::new();
    let mut current_cluster: Vec<ParsedPhoneme> = Vec::new();

    for phoneme in phonemes {
        // Vowels/diphthongs are boundaries — flush the current consonant cluster
        if language_info.is_vowel(&phoneme.label) {
            if !current_cluster.is_empty() {
                clusters.push(current_cluster);
                current_cluster = Vec::new();
            }
        } else {
            current_cluster.push(phoneme.clone());
        }
    }

    // Don't forget trailing consonants (after the last vowel)
    if !current_cluster.is_empty() {
        clusters.push(current_cluster);
    }

    clusters
}

/// Generate a simple ISO 8601 timestamp.
fn chrono_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();

    // Simple date calculation (not accounting for leap years perfectly)
    let days = secs / 86400;
    let years = 1970 + days / 365;
    let remaining_days = days % 365;
    let month = remaining_days / 30 + 1;
    let day = remaining_days % 30 + 1;
    let hours = (secs % 86400) / 3600;
    let minutes = (secs % 3600) / 60;
    let seconds = secs % 60;

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        years, month, day, hours, minutes, seconds
    )
}

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

/// Merge two timing models into one.
///
/// Combines cluster and generic timing data from both models.
/// The resulting model uses metadata from the primary model.
/// Duplicate phoneme sequences have their samples concatenated.
pub fn merge_models(primary: &TimingModel, secondary: &TimingModel) -> TimingModel {
    let mut cluster_map: HashMap<Vec<String>, Vec<Vec<u16>>> = HashMap::new();
    let mut generic_map: HashMap<Vec<PhonemeType>, Vec<Vec<u16>>> = HashMap::new();

    // Collect from primary
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

    // Merge from secondary
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
        version: primary.version.clone(),
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
        diphthong_extensions: HashMap::new(),
        syllabic_consonants: raw.syllabic_consonants,
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

/// Load a label translation map from a YAML file.
///
/// The file should be a simple flat mapping from TextGrid phoneme labels to the
/// labels used in your language file. For example:
///
/// ```yaml
/// ph: "p_h"
/// "p>": "p_}"
/// py: "p'"
/// ```
///
/// This is optional: if your TextGrid files already use the same labels as the
/// language file, you do not need a label map.
pub fn load_label_map(path: &str) -> Result<HashMap<String, String>, TimingError> {
    let content = fs::read_to_string(path).map_err(|e| TimingError::io(path, e))?;
    serde_yaml::from_str(&content).map_err(|e| TimingError::yaml(path, e))
}

/// Raw structure for deserializing the global phoneme file.
///
/// Each top-level key is a `PhonemeType`; its list holds the phonemes of that
/// type. This single file is the source of truth for both type classification
/// and the vowel / diphthong / syllabic-consonant data needed to split clusters.
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
    #[serde(default)]
    syllabic_consonants: Vec<String>,
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
/// boundaries used to split consonant clusters; diphthong extensions and
/// syllabic consonants are taken from the same file.
///
/// Pass `Some(path)` to use a custom global file, or `None` to use the default
/// inventory bundled with flow.
pub fn load_language_info_from_path(path: Option<&str>) -> Result<LanguageInfo, TimingError> {
    let (name, raw) = load_raw_language_info(path)?;
    Ok(build_language_info_from_raw(name, raw))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_silence() {
        assert!(is_silence("Sil"));
        assert!(is_silence("sil"));
        assert!(is_silence("sp"));
        assert!(is_silence(""));
        assert!(!is_silence("k"));
        assert!(!is_silence("a"));
    }

    #[test]
    fn test_split_by_vowels() {
        let mut lang = LanguageInfo::new("Test");
        lang.add_vowel("a");
        lang.add_vowel("i");

        let phonemes = vec![
            ParsedPhoneme {
                label: "k".to_string(),
                duration_ms: 100,
            },
            ParsedPhoneme {
                label: "a".to_string(),
                duration_ms: 200,
            },
            ParsedPhoneme {
                label: "t".to_string(),
                duration_ms: 80,
            },
            ParsedPhoneme {
                label: "s".to_string(),
                duration_ms: 90,
            },
            ParsedPhoneme {
                label: "i".to_string(),
                duration_ms: 180,
            },
        ];

        let clusters = split_by_vowels(&phonemes, &lang);

        // Vowels excluded: [k], [t, s]
        assert_eq!(clusters.len(), 2);
        assert_eq!(clusters[0].len(), 1); // k
        assert_eq!(clusters[0][0].label, "k");
        assert_eq!(clusters[1].len(), 2); // t, s
        assert_eq!(clusters[1][0].label, "t");
        assert_eq!(clusters[1][1].label, "s");
    }

    #[test]
    fn test_split_by_vowels_with_diphthong() {
        let mut lang = LanguageInfo::new("Test");
        lang.add_vowel("a");
        lang.add_diphthong("aI", "A");

        let phonemes = vec![
            ParsedPhoneme {
                label: "k".to_string(),
                duration_ms: 100,
            },
            ParsedPhoneme {
                label: "aI".to_string(),
                duration_ms: 250,
            },
            ParsedPhoneme {
                label: "t".to_string(),
                duration_ms: 80,
            },
        ];

        let clusters = split_by_vowels(&phonemes, &lang);

        // Vowels/diphthongs excluded: [k], [t]
        assert_eq!(clusters.len(), 2);
        assert_eq!(clusters[0].len(), 1); // k
        assert_eq!(clusters[0][0].label, "k");
        assert_eq!(clusters[1].len(), 1); // t
        assert_eq!(clusters[1][0].label, "t");
    }

    #[test]
    fn test_load_phoneme_map_from_global_default() {
        // The embedded default global file should parse and classify correctly.
        let map = load_phoneme_map_from_path(None).unwrap();

        assert_eq!(map.get_type("p"), Some(PhonemeType::Plosive));
        assert_eq!(map.get_type("s"), Some(PhonemeType::Fricative));
        assert_eq!(map.get_type("a"), Some(PhonemeType::Vowel));
        assert_eq!(map.get_type("aI"), Some(PhonemeType::Diphthong));
        assert_eq!(map.get_type("ts"), Some(PhonemeType::Affricate));
        assert_eq!(map.get_type("m"), Some(PhonemeType::Sonorant));
        assert_eq!(map.get_type("4"), Some(PhonemeType::Tap));
        assert_eq!(map.get_type("sil"), None);
    }

    #[test]
    fn test_load_phoneme_map_from_global_custom() {
        let yaml = r#"
plosives:
  - "p"
  - "t"
fricatives:
  - "s"
  - "f"
sonorants:
  - "m"
  - "n"
taps:
  - "4"
affricates:
  - "ts"
vowels:
  - "a"
  - "i"
diphthongs:
  - "aI"
"#;
        let tmp = std::env::temp_dir().join("test_global_map.yaml");
        fs::write(&tmp, yaml).unwrap();

        let map = load_phoneme_map_from_path(Some(tmp.to_str().unwrap())).unwrap();

        assert_eq!(map.get_type("p"), Some(PhonemeType::Plosive));
        assert_eq!(map.get_type("t"), Some(PhonemeType::Plosive));
        assert_eq!(map.get_type("s"), Some(PhonemeType::Fricative));
        assert_eq!(map.get_type("a"), Some(PhonemeType::Vowel));
        assert_eq!(map.get_type("aI"), Some(PhonemeType::Diphthong));
        assert_eq!(map.get_type("ts"), Some(PhonemeType::Affricate));
        assert_eq!(map.get_type("m"), Some(PhonemeType::Sonorant));
        assert_eq!(map.get_type("4"), Some(PhonemeType::Tap));
        assert_eq!(map.get_type("sil"), None);

        fs::remove_file(tmp).ok();
    }

    #[test]
    fn test_load_language_info_from_global_default() {
        // The bundled global file supplies vowels and diphthongs.
        let info = load_language_info_from_path(None).unwrap();

        assert!(info.is_vowel("{"));
        assert!(info.is_vowel("aI")); // diphthong counts as a vowel boundary
        assert!(!info.is_vowel("p"));
    }

    #[test]
    fn test_load_language_info_from_global_custom() {
        let yaml = r#"
plosives:
  - "p"
  - "t"
fricatives:
  - "f"
  - "s"
sonorants:
  - "m"
  - "n"
  - "l"
vowels:
  - "{"
  - "E"
  - "I"
diphthongs:
  - "aI"
  - "eI"
syllabic_consonants:
  - "m"
  - "n"
"#;
        let tmp = std::env::temp_dir().join("test_global_language.yaml");
        fs::write(&tmp, yaml).unwrap();

        let info = load_language_info_from_path(Some(tmp.to_str().unwrap())).unwrap();

        assert_eq!(info.vowels, vec!["{", "E", "I"]);
        assert_eq!(info.diphthongs, vec!["aI", "eI"]);
        assert_eq!(info.syllabic_consonants, vec!["m", "n"]);
        assert!(info.is_vowel("{"));
        assert!(info.is_vowel("aI"));
        assert!(!info.is_vowel("p"));

        fs::remove_file(tmp).ok();
    }

    #[test]
    fn test_merge_models() {
        let mut m1 = TimingModel::new(TimingMetadata::new("Lib1", "English", "Default"));
        let mut ct1 = ClusterTiming::new(vec!["k".to_string(), "s".to_string()]);
        ct1.add_sample(vec![95, 110]);
        m1.cluster_timings.push(ct1);

        let mut m2 = TimingModel::new(TimingMetadata::new("Lib2", "English", "Default"));
        let mut ct2 = ClusterTiming::new(vec!["k".to_string(), "s".to_string()]);
        ct2.add_sample(vec![100, 105]);
        m2.cluster_timings.push(ct2);
        let mut ct3 = ClusterTiming::new(vec!["t".to_string()]);
        ct3.add_sample(vec![80]);
        m2.cluster_timings.push(ct3);

        let merged = merge_models(&m1, &m2);

        assert_eq!(merged.metadata.library, "Lib1"); // uses primary metadata
        // k,s should have 2 samples (1 from each)
        let ks = merged
            .cluster_timings
            .iter()
            .find(|c| c.phonemes == vec!["k", "s"])
            .unwrap();
        assert_eq!(ks.samples.len(), 2);
        // t should have 1 sample from m2
        let t = merged
            .cluster_timings
            .iter()
            .find(|c| c.phonemes == vec!["t"])
            .unwrap();
        assert_eq!(t.samples.len(), 1);
    }
}
