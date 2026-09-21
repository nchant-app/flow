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
    lang.add_diphthong("aI");

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
