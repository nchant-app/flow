
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
