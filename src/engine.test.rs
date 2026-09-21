use super::*;
use crate::model::{ClusterTiming, TimingMetadata};

fn create_test_engine() -> TimingEngine {
    let mut model = TimingModel::new(TimingMetadata::new("TestLib", "English", "Default"));

    let mut ct = ClusterTiming::new(vec!["k".to_string(), "s".to_string()]);
    ct.add_sample(vec![95, 110]);
    ct.add_sample(vec![100, 105]);
    model.cluster_timings.push(ct);

    let mut language_info = LanguageInfo::new("English");
    language_info.add_phoneme("k", PhonemeType::Plosive);
    language_info.add_phoneme("s", PhonemeType::Fricative);
    language_info.add_vowel("a");

    TimingEngine::from_components(model, language_info)
}

#[test]
fn test_engine_predict() {
    let engine = create_test_engine();
    let result = engine.predict(&["k".to_string(), "s".to_string()]);

    assert_eq!(result.timings.len(), 2);
    assert_eq!(result.timings[0].duration_ms, 97);
    assert_eq!(result.timings[1].duration_ms, 107);
    assert_eq!(result.total_duration_ms, 97 + 107);
}

#[test]
fn test_engine_get_timing() {
    let engine = create_test_engine();
    let phonemes = ["k".to_string(), "s".to_string()];
    let timings = engine.get_timing(&phonemes);

    assert_eq!(timings.len(), 2);
    assert_eq!(*timings[0].0, "k");
    assert_eq!(timings[0].1, 97);
}

#[test]
fn test_engine_metadata() {
    let engine = create_test_engine();
    assert_eq!(engine.library(), "TestLib");
    assert_eq!(engine.language(), "English");
    assert_eq!(engine.voice_color(), "Default");
}

#[test]
fn test_engine_accessors() {
    let engine = create_test_engine();
    assert_eq!(engine.language_info().name, "English");
}

#[test]
fn test_engine_clone() {
    let engine = create_test_engine();
    let engine2 = engine.clone();

    let r1 = engine.predict(&["k".to_string(), "s".to_string()]);
    let r2 = engine2.predict(&["k".to_string(), "s".to_string()]);
    assert_eq!(r1.total_duration_ms, r2.total_duration_ms);
}

#[test]
fn test_engine_validate() {
    let engine = create_test_engine();
    let valid = vec!["k".to_string(), "a".to_string()];
    let invalid = vec!["k".to_string(), "zzz".to_string()];

    assert!(engine.validate(&valid).is_empty());
    assert_eq!(engine.validate(&invalid), vec!["zzz"]);
}

#[test]
fn test_engine_predict_batch() {
    let engine = create_test_engine();
    let batches = vec![
        vec!["k".to_string(), "s".to_string()],
        vec!["k".to_string()],
    ];
    let results = engine.predict_batch(&batches);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].total_duration_ms, 97 + 107);
}

#[test]
fn test_engine_with_fallback() {
    let mut engine = create_test_engine();
    engine.with_fallback(PhonemeType::Plosive, 80);
    assert_eq!(engine.fallbacks.get(&PhonemeType::Plosive), Some(&80));
}

#[test]
fn test_engine_with_fallback_clone_isolation() {
    let mut engine = create_test_engine();
    let mut engine2 = engine.clone();
    engine.with_fallback(PhonemeType::Plosive, 80);
    engine2.with_fallback(PhonemeType::Plosive, 90);

    assert_eq!(engine.fallbacks.get(&PhonemeType::Plosive), Some(&80));
    assert_eq!(engine2.fallbacks.get(&PhonemeType::Plosive), Some(&90));
}

#[test]
fn test_engine_builder() {
    let mut model = TimingModel::new(TimingMetadata::new("TestLib", "English", "Default"));
    let mut ct = ClusterTiming::new(vec!["k".to_string()]);
    ct.add_sample(vec![100]);
    model.cluster_timings.push(ct);

    let mut language_info = LanguageInfo::new("English");
    language_info.add_phoneme("k", PhonemeType::Plosive);

    let engine = TimingEngineBuilder::new()
        .model(model)
        .language_info(language_info)
        .fallback(PhonemeType::Plosive, 90)
        .build()
        .unwrap();

    assert_eq!(engine.library(), "TestLib");
    assert_eq!(engine.fallbacks.get(&PhonemeType::Plosive), Some(&90));
}

#[test]
fn test_engine_builder_missing_model() {
    let result = TimingEngineBuilder::new().build();
    assert!(result.is_err());
}

#[test]
fn test_engine_debug() {
    let engine = create_test_engine();
    let debug = format!("{:?}", engine);
    assert!(debug.contains("TestLib"));
    assert!(debug.contains("English"));
}
