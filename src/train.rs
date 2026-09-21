//! Training module for building timing models from TextGrid files.
//!
//! This module provides functionality to extract phoneme timing data from
//! Praat TextGrid files and build a `TimingModel`.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::error::TimingError;
use crate::model::TimingMetadata;
use crate::model::{ClusterTiming, GenericTiming, LanguageInfo, PhonemeType, TimingModel};

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
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
}
