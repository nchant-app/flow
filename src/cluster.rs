//! Generic cluster-splitting for phoneme sequences.
//!
//! Splits a sequence of phonemes into consonant clusters separated by vowels.
//! This is the core algorithm used by both training and prediction.

/// A consonant cluster paired with its following vowel (if any).
///
/// In the timing model, consonant durations are predicted per-cluster,
/// and vowel duration is the remaining time after consonants are allocated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cluster<K> {
    /// The vowel following this consonant cluster, if any.
    pub vowel: Option<K>,
    /// The consonants in this cluster (may be empty for vowel-only clusters).
    pub consonants: Vec<K>,
}

/// Split a phoneme sequence into consonant clusters.
///
/// Vowels act as cluster boundaries. Consonants between vowels form a cluster,
/// and cross-word boundaries naturally create clusters from the trailing
/// consonants of one word and the leading consonants of the next.
///
/// # Arguments
///
/// * `phonemes` - The phoneme sequence to split
/// * `is_vowel` - A function that returns `true` if a phoneme is a vowel/diphthong
///
/// # Examples
///
/// ```
/// use nchant_flow::cluster::{split_into_clusters, Cluster};
///
/// // [k, a, t, s, i] → clusters: [k]+a, [t,s]+i
/// let phonemes = vec!["k", "a", "t", "s", "i"];
/// let clusters = split_into_clusters(&phonemes, |p| *p == "a" || *p == "i");
///
/// assert_eq!(clusters.len(), 2);
/// assert_eq!(clusters[0].vowel, Some(&"a"));
/// assert_eq!(clusters[0].consonants, vec![&"k"]);
/// assert_eq!(clusters[1].vowel, Some(&"i"));
/// assert_eq!(clusters[1].consonants, vec![&"t", &"s"]);
/// ```
pub fn split_into_clusters<'a, K, F>(phonemes: &'a [K], is_vowel: F) -> Vec<Cluster<&'a K>>
where
    F: Fn(&K) -> bool,
{
    let mut clusters: Vec<Cluster<&'a K>> = Vec::new();
    let mut current_consonants: Vec<&'a K> = Vec::new();

    for phoneme in phonemes {
        if is_vowel(phoneme) {
            clusters.push(Cluster {
                vowel: Some(phoneme),
                consonants: std::mem::take(&mut current_consonants),
            });
        } else {
            current_consonants.push(phoneme);
        }
    }

    // Trailing consonants after the last vowel
    if !current_consonants.is_empty() {
        clusters.push(Cluster {
            vowel: None,
            consonants: current_consonants,
        });
    }

    clusters
}

#[cfg(test)]
#[path = "cluster.test.rs"]
mod tests;
