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
/// use maghni_flow::cluster::{split_into_clusters, Cluster};
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
mod tests {
    use super::*;

    fn is_vowel(p: &&str) -> bool {
        matches!(*p, "a" | "i" | "u" | "e" | "o")
    }

    #[test]
    fn test_split_basic() {
        // [k, a, t, s, i] → [k]+a, [t,s]+i
        let phonemes = vec!["k", "a", "t", "s", "i"];
        let clusters = split_into_clusters(&phonemes, is_vowel);

        assert_eq!(clusters.len(), 2);
        assert_eq!(clusters[0].vowel, Some(&"a"));
        assert_eq!(clusters[0].consonants, vec![&"k"]);
        assert_eq!(clusters[1].vowel, Some(&"i"));
        assert_eq!(clusters[1].consonants, vec![&"t", &"s"]);
    }

    #[test]
    fn test_split_trailing_consonants() {
        // [k, a, t, s] → [k]+a, [t,s]+None
        let phonemes = vec!["k", "a", "t", "s"];
        let clusters = split_into_clusters(&phonemes, is_vowel);

        assert_eq!(clusters.len(), 2);
        assert_eq!(clusters[0].vowel, Some(&"a"));
        assert_eq!(clusters[0].consonants, vec![&"k"]);
        assert_eq!(clusters[1].vowel, None);
        assert_eq!(clusters[1].consonants, vec![&"t", &"s"]);
    }

    #[test]
    fn test_split_no_consonants() {
        // [a, i, u] → []+a, []+i, []+u
        let phonemes = vec!["a", "i", "u"];
        let clusters = split_into_clusters(&phonemes, is_vowel);

        assert_eq!(clusters.len(), 3);
        for cluster in &clusters {
            assert!(cluster.consonants.is_empty());
            assert!(cluster.vowel.is_some());
        }
    }

    #[test]
    fn test_split_no_vowels() {
        // [k, t, s] → [k,t,s]+None
        let phonemes = vec!["k", "t", "s"];
        let clusters = split_into_clusters(&phonemes, is_vowel);

        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].vowel, None);
        assert_eq!(clusters[0].consonants, vec![&"k", &"t", &"s"]);
    }

    #[test]
    fn test_split_empty() {
        let phonemes: Vec<&str> = vec![];
        let clusters = split_into_clusters(&phonemes, is_vowel);
        assert!(clusters.is_empty());
    }

    #[test]
    fn test_split_single_vowel() {
        let phonemes = vec!["a"];
        let clusters = split_into_clusters(&phonemes, is_vowel);

        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].vowel, Some(&"a"));
        assert!(clusters[0].consonants.is_empty());
    }

    #[test]
    fn test_split_leading_consonants() {
        // [t, k, a] → [t,k]+a
        let phonemes = vec!["t", "k", "a"];
        let clusters = split_into_clusters(&phonemes, is_vowel);

        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].vowel, Some(&"a"));
        assert_eq!(clusters[0].consonants, vec![&"t", &"k"]);
    }
}
