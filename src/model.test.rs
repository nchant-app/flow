
use super::*;

#[test]
fn test_phoneme_map() {
    let mut map = PhonemeMap::new("Test Map");
    map.add_phoneme("t", PhonemeType::Plosive);
    map.add_phoneme("a", PhonemeType::Vowel);

    assert_eq!(map.get_type("t"), Some(PhonemeType::Plosive));
    assert_eq!(map.get_type("a"), Some(PhonemeType::Vowel));
    assert_eq!(map.get_type("zzz"), None);
}

#[test]
fn test_language_info() {
    let mut lang = LanguageInfo::new("English");
    lang.add_vowel("{");
    lang.add_vowel("i");
    lang.add_diphthong("aI", "A");

    assert!(lang.is_vowel("{"));
    assert!(lang.is_vowel("aI"));
    assert!(!lang.is_vowel("t"));
}

#[test]
fn test_timing_result() {
    let pairs = vec![("t".to_string(), 100), ("a".to_string(), 300)];
    let result = TimingResult::from_pairs(pairs);

    assert_eq!(result.timings.len(), 2);
    assert_eq!(result.total_duration_ms, 400);
}

#[test]
fn test_phoneme_type_default() {
    assert_eq!(PhonemeType::default(), PhonemeType::None);
}

#[test]
fn test_phoneme_map_contains() {
    let mut map = PhonemeMap::new("Test Map");
    map.add_phoneme("t", PhonemeType::Plosive);

    assert!(map.contains("t"));
    assert!(!map.contains("z"));
}

#[test]
fn test_phoneme_map_overwrite() {
    let mut map = PhonemeMap::new("Test Map");
    map.add_phoneme("t", PhonemeType::Plosive);
    map.add_phoneme("t", PhonemeType::Fricative);

    assert_eq!(map.get_type("t"), Some(PhonemeType::Fricative));
}

#[test]
fn test_language_info_consonant_not_vowel() {
    let mut lang = LanguageInfo::new("English");
    lang.add_vowel("a");
    lang.add_vowel("i");

    assert!(!lang.is_vowel("k"));
    assert!(!lang.is_vowel("s"));
    assert!(!lang.is_vowel(""));
}

#[test]
fn test_language_info_diphthong_extension() {
    let mut lang = LanguageInfo::new("English");
    lang.add_diphthong("aI", "A");
    lang.add_diphthong("eI", "E");

    assert_eq!(lang.diphthong_extensions.get("aI"), Some(&"A".to_string()));
    assert_eq!(lang.diphthong_extensions.get("eI"), Some(&"E".to_string()));
    assert_eq!(lang.diphthong_extensions.get("oU"), None);
}

#[test]
fn test_cluster_timing_multiple_samples() {
    let mut ct = ClusterTiming::new(vec!["k".to_string(), "a".to_string()]);
    ct.add_sample(vec![100, 200]);
    ct.add_sample(vec![110, 210]);
    ct.add_sample(vec![90, 190]);

    assert_eq!(ct.samples.len(), 3);
    assert_eq!(ct.phonemes.len(), 2);
    assert_eq!(ct.samples[0], vec![100, 200]);
    assert_eq!(ct.samples[2], vec![90, 190]);
}

#[test]
fn test_generic_timing_construction() {
    let mut gt = GenericTiming::new(vec![PhonemeType::Plosive, PhonemeType::Vowel]);
    gt.add_sample(vec![100, 300]);

    assert_eq!(gt.types.len(), 2);
    assert_eq!(gt.samples.len(), 1);
    assert_eq!(gt.types[0], PhonemeType::Plosive);
    assert_eq!(gt.types[1], PhonemeType::Vowel);
}

#[test]
fn test_timing_model_new() {
    let metadata = TimingMetadata::new("TestLib", "English", "Default");
    let model = TimingModel::new(metadata);

    assert_eq!(model.version, "1.0");
    assert_eq!(model.metadata.library, "TestLib");
    assert_eq!(model.metadata.language, "English");
    assert_eq!(model.metadata.voice_color, "Default");
    assert!(model.cluster_timings.is_empty());
    assert!(model.generic_timings.is_empty());
}

#[test]
fn test_timing_metadata_optional_fields() {
    let metadata = TimingMetadata::new("Lib", "Lang", "Color");

    assert!(metadata.created_at.is_none());
    assert!(metadata.source_files.is_none());
}

#[test]
fn test_timing_result_empty() {
    let result = TimingResult::from_pairs(vec![]);

    assert_eq!(result.timings.len(), 0);
    assert_eq!(result.total_duration_ms, 0);
}

#[test]
fn test_timing_result_single_phoneme() {
    let pairs = vec![("a".to_string(), 500)];
    let result = TimingResult::from_pairs(pairs);

    assert_eq!(result.timings.len(), 1);
    assert_eq!(result.timings[0].phoneme, "a");
    assert_eq!(result.timings[0].duration_ms, 500);
    assert_eq!(result.total_duration_ms, 500);
}

#[test]
fn test_timing_result_total_is_sum() {
    let pairs = vec![
        ("t".to_string(), 100),
        ("a".to_string(), 300),
        ("k".to_string(), 80),
    ];
    let result = TimingResult::from_pairs(pairs);

    assert_eq!(result.total_duration_ms, 480);
}

#[test]
fn test_timing_result_display() {
    let pairs = vec![("t".to_string(), 100), ("a".to_string(), 300)];
    let result = TimingResult::from_pairs(pairs);
    let display = format!("{}", result);
    assert!(display.contains("t: 100ms"));
    assert!(display.contains("a: 300ms"));
    assert!(display.contains("Total: 400ms"));
}

#[test]
fn test_timing_model_yaml_roundtrip() {
    let mut model = TimingModel::new(TimingMetadata::new("TestLib", "English", "Default"));

    let mut ct = ClusterTiming::new(vec!["k".to_string(), "a".to_string()]);
    ct.add_sample(vec![95, 280]);
    model.cluster_timings.push(ct);

    let mut gt = GenericTiming::new(vec![PhonemeType::Plosive, PhonemeType::Vowel]);
    gt.add_sample(vec![100, 300]);
    model.generic_timings.push(gt);

    let yaml = serde_yaml::to_string(&model).unwrap();
    let deserialized: TimingModel = serde_yaml::from_str(&yaml).unwrap();

    assert_eq!(deserialized.version, model.version);
    assert_eq!(deserialized.metadata.library, model.metadata.library);
    assert_eq!(deserialized.cluster_timings.len(), 1);
    assert_eq!(deserialized.cluster_timings[0].phonemes, vec!["k", "a"]);
    assert_eq!(deserialized.cluster_timings[0].samples[0], vec![95, 280]);
    assert_eq!(deserialized.generic_timings.len(), 1);
    assert_eq!(
        deserialized.generic_timings[0].types,
        vec![PhonemeType::Plosive, PhonemeType::Vowel]
    );
}
