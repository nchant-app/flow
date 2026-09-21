use super::*;
use crate::model::TimingMetadata;

#[test]
fn test_load_phoneme_map_from_global_default() {
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
    let info = load_language_info_from_path(None).unwrap();

    assert!(info.is_vowel("{"));
    assert!(info.is_vowel("aI"));
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
"#;
    let tmp = std::env::temp_dir().join("test_global_language.yaml");
    fs::write(&tmp, yaml).unwrap();

    let info = load_language_info_from_path(Some(tmp.to_str().unwrap())).unwrap();

    assert_eq!(info.vowels, vec!["{", "E", "I"]);
    assert_eq!(info.diphthongs, vec!["aI", "eI"]);
    assert_eq!(info.sonorants, vec!["m", "n", "l"]);
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

    assert_eq!(merged.metadata.library, "Lib1");
    let ks = merged
        .cluster_timings
        .iter()
        .find(|c| c.phonemes == vec!["k", "s"])
        .unwrap();
    assert_eq!(ks.samples.len(), 2);
    let t = merged
        .cluster_timings
        .iter()
        .find(|c| c.phonemes == vec!["t"])
        .unwrap();
    assert_eq!(t.samples.len(), 1);
}
