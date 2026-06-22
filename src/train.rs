//! Training module for building timing models from TextGrid files.
//!
//! This module provides functionality to extract phoneme timing data from
//! Praat TextGrid files and build a `TimingModel`.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::error::TimingError;
use crate::model::{
    derive_phoneme_type, ClusterTiming, GenericTiming, LanguageInfo, PhonemeMap, PhonemeType,
    TimingMetadata, TimingModel,
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
/// * `phoneme_map` - Map from X-SAMPA phoneme labels to their types
/// * `language_info` - Language-specific info including vowel/diphthong lists
/// * `metadata` - Metadata for the output model
/// * `config` - Optional training configuration
///
/// # Returns
/// A `TimingModel` containing the extracted timing data, or an error.
#[cfg(feature = "train")]
pub fn train_from_textgrids(
    textgrid_dir: &str,
    phoneme_map: &PhonemeMap,
    language_info: &LanguageInfo,
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
            |path| match parse_textgrid(path, phoneme_map, language_info, &config) {
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
            .map(|label| phoneme_map.get_type(label).unwrap_or(PhonemeType::None))
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
    phoneme_map: &PhonemeMap,
    language_info: &LanguageInfo,
    config: &TrainingConfig,
) -> Result<Vec<Vec<ParsedPhoneme>>, TimingError> {
    let path_str = path.display().to_string();
    let textgrid = textgridde_rs::parse_textgrid(path.to_path_buf(), true).map_err(|e| {
        TimingError::TextGrid {
            path: path_str.clone(),
            reason: format!("{:?}", e),
        }
    })?;

    // Get the last tier (assumed to be the phoneme tier)
    let tier = textgrid.tiers().last().ok_or_else(|| TimingError::TextGrid {
        path: path_str.clone(),
        reason: "TextGrid has no tiers".to_string(),
    })?;

    let intervals = match tier {
        textgrid::Tier::IntervalTier(t) => t.intervals(),
        textgrid::Tier::PointTier(_) => {
            return Err(TimingError::TextGrid {
                path: path_str,
                reason: "expected IntervalTier, got PointTier".to_string(),
            })
        }
    };

    // Parse all phonemes
    let mut phonemes: Vec<ParsedPhoneme> = Vec::new();
    for interval in intervals {
        let label = interval.text().to_string();
        let duration_secs = interval.get_duration();
        let duration_ms = (duration_secs * 1000.0) as u16;

        // Skip if duration is out of bounds
        if duration_ms < config.min_duration_ms || duration_ms > config.max_duration_ms {
            continue;
        }

        // Warn if phoneme not in map (but still include it)
        if !is_silence(&label) && !is_noise(&label) && !phoneme_map.contains(&label) {
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

/// Raw structure for deserializing global.yaml format.
#[derive(Debug, serde::Deserialize)]
struct RawGlobalYaml {
    base_map: HashMap<String, String>,
}

/// Load a phoneme map from a global.yaml file.
///
/// Parses the `base_map` section (X-SAMPA -> enum name) and derives
/// `PhonemeType` from the enum name conventions.
pub fn load_phoneme_map(path: &str) -> Result<PhonemeMap, TimingError> {
    let content = fs::read_to_string(path).map_err(|e| TimingError::io(path, e))?;
    let raw: RawGlobalYaml =
        serde_yaml::from_str(&content).map_err(|e| TimingError::yaml(path, e))?;

    let mut phonemes = HashMap::new();
    for (xsampa, enum_name) in &raw.base_map {
        let ptype = derive_phoneme_type(enum_name);
        phonemes.insert(xsampa.clone(), ptype);
    }

    Ok(PhonemeMap {
        version: "1.0".to_string(),
        name: "Global".to_string(),
        phonemes,
    })
}

/// Raw structure for deserializing language YAML files.
///
/// Matches the language file format:
/// ```yaml
/// id: English
/// vowels:
///   - ["{", 65]
/// diphthongs:
///   - ["aI", 87, "A"]
/// syllabic_consonants:
///   - "m"
/// ```
#[derive(Debug, serde::Deserialize)]
struct RawLanguageFile {
    id: String,
    #[serde(default)]
    vowels: Vec<(String, serde_yaml::Value)>,
    #[serde(default)]
    diphthongs: Vec<RawDiphthong>,
    #[serde(default)]
    syllabic_consonants: Vec<String>,
}

/// Raw diphthong entry: [xsampa, id, extension_vowel]
#[derive(Debug)]
struct RawDiphthong {
    xsampa: String,
    extension: String,
}

impl<'de> serde::Deserialize<'de> for RawDiphthong {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let seq: Vec<serde_yaml::Value> = Vec::deserialize(deserializer)?;
        if seq.len() < 3 {
            return Err(serde::de::Error::custom(
                "Diphthong entry must have at least 3 elements: [xsampa, id, extension]",
            ));
        }
        let xsampa = seq[0]
            .as_str()
            .ok_or_else(|| serde::de::Error::custom("Diphthong xsampa must be a string"))?
            .to_string();
        let extension = seq[2]
            .as_str()
            .ok_or_else(|| serde::de::Error::custom("Diphthong extension must be a string"))?
            .to_string();
        Ok(RawDiphthong { xsampa, extension })
    }
}

/// Load language info from a language YAML file.
///
/// Parses the language file format with X-SAMPA phoneme notation.
/// Extracts vowels, diphthongs, and syllabic consonants.
pub fn load_language_info(path: &str) -> Result<LanguageInfo, TimingError> {
    let content = fs::read_to_string(path).map_err(|e| TimingError::io(path, e))?;
    let raw: RawLanguageFile =
        serde_yaml::from_str(&content).map_err(|e| TimingError::yaml(path, e))?;

    let vowels: Vec<String> = raw.vowels.into_iter().map(|(xsampa, _)| xsampa).collect();

    let mut diphthongs = Vec::new();
    let mut diphthong_extensions = HashMap::new();
    for d in raw.diphthongs {
        diphthong_extensions.insert(d.xsampa.clone(), d.extension);
        diphthongs.push(d.xsampa);
    }

    Ok(LanguageInfo {
        name: raw.id,
        vowels,
        diphthongs,
        diphthong_extensions,
        syllabic_consonants: raw.syllabic_consonants,
    })
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
    fn test_load_phoneme_map_from_global_yaml() {
        let yaml = r#"
base_map:
  "p": VoicelessLabialPlosive
  "t": VoicelessAlveolarPlosive
  "a": OpenCentralUnroundedVowel
  "i": CloseFrontUnroundedVowel
  "aI": OpenCentralUnroundedNearcloseNearfrontUnroundedDiphthong
  "sil": SILENCE
"#;
        let tmp = std::env::temp_dir().join("test_global.yaml");
        fs::write(&tmp, yaml).unwrap();

        let map = load_phoneme_map(tmp.to_str().unwrap()).unwrap();

        assert_eq!(map.get_type("p"), Some(PhonemeType::Plosive));
        assert_eq!(map.get_type("t"), Some(PhonemeType::Plosive));
        assert_eq!(map.get_type("a"), Some(PhonemeType::Vowel));
        assert_eq!(map.get_type("i"), Some(PhonemeType::Vowel));
        assert_eq!(map.get_type("aI"), Some(PhonemeType::Diphthong));
        assert_eq!(map.get_type("sil"), Some(PhonemeType::Special));

        fs::remove_file(tmp).ok();
    }

    #[test]
    fn test_load_language_info() {
        let yaml = r#"
id: English
family: IndoEuropean
branch: Germanic
writing_systems:
  - latin
consonants:
  - ["p", 0]
  - ["t", 6]
vowels:
  - ["{", 65]
  - ["E", 67]
  - ["I", 71]
diphthongs:
  - ["aI", 87, "A"]
  - ["eI", 88, "E"]
syllabic_consonants:
  - "m"
  - "n"
"#;
        let tmp = std::env::temp_dir().join("test_language.yaml");
        fs::write(&tmp, yaml).unwrap();

        let info = load_language_info(tmp.to_str().unwrap()).unwrap();

        assert_eq!(info.name, "English");
        assert_eq!(info.vowels, vec!["{", "E", "I"]);
        assert_eq!(info.diphthongs, vec!["aI", "eI"]);
        assert_eq!(info.diphthong_extensions.get("aI"), Some(&"A".to_string()));
        assert_eq!(info.diphthong_extensions.get("eI"), Some(&"E".to_string()));
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
