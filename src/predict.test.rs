
use super::*;
use crate::model::{ClusterTiming, GenericTiming, TimingMetadata};

fn create_test_model() -> TimingModel {
    let mut model = TimingModel::new(TimingMetadata::new("TestLib", "English", "Default"));

    // Cluster timings contain consonants only (vowels are never in the model).
    // E.g. [k, s] represents a consonant cluster between two vowels.
    let mut ct = ClusterTiming::new(vec!["k".to_string(), "s".to_string()]);
    ct.add_sample(vec![95, 110]);
    ct.add_sample(vec![100, 105]);
    ct.add_sample(vec![90, 115]);
    model.cluster_timings.push(ct);

    let mut ct2 = ClusterTiming::new(vec!["t".to_string()]);
    ct2.add_sample(vec![80]);
    ct2.add_sample(vec![90]);
    ct2.add_sample(vec![85]);
    model.cluster_timings.push(ct2);

    // Generic timings by phoneme type (also consonant-only)
    let mut gt = GenericTiming::new(vec![PhonemeType::Plosive, PhonemeType::Fricative]);
    gt.add_sample(vec![100, 120]);
    gt.add_sample(vec![90, 130]);
    model.generic_timings.push(gt);

    model
}

fn create_test_phoneme_map() -> LanguageInfo {
    let mut map = LanguageInfo::new("Test");
    map.add_phoneme("k", PhonemeType::Plosive);
    map.add_phoneme("t", PhonemeType::Plosive);
    map.add_phoneme("s", PhonemeType::Fricative);
    map.add_phoneme("f", PhonemeType::Fricative);
    map.add_phoneme("a", PhonemeType::Vowel);
    map
}

#[test]
fn test_exact_match() {
    let model = create_test_model();
    let map = create_test_phoneme_map();
    let lookup = TimingLookup::from_model(&model, &map);

    let cluster = ["k".to_string(), "s".to_string()];
    let result = lookup.get_timing(&cluster);

    assert_eq!(result.len(), 2);
    // Average of [95, 100, 90] = 95
    assert_eq!(result[0].1, 95);
    // Average of [110, 105, 115] = 110
    assert_eq!(result[1].1, 110);
}

#[test]
fn test_generic_fallback() {
    let model = create_test_model();
    let map = create_test_phoneme_map();
    let lookup = TimingLookup::from_model(&model, &map);

    // Use a plosive+fricative combination that's not in cluster_timings
    // but matches the generic pattern [Plosive, Fricative]
    let cluster = ["t".to_string(), "f".to_string()];
    let result = lookup.get_timing(&cluster);

    assert_eq!(result.len(), 2);
    // Average of [100, 90] = 95
    assert_eq!(result[0].1, 95);
    // Average of [120, 130] = 125
    assert_eq!(result[1].1, 125);
}

#[test]
fn test_single_phoneme_exact() {
    let model = create_test_model();
    let map = create_test_phoneme_map();
    let lookup = TimingLookup::from_model(&model, &map);

    // Single plosive "t" has exact cluster data: average of [80, 90, 85] = 85
    let cluster = ["t".to_string()];
    let result = lookup.get_timing(&cluster);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].1, 85);
}

#[test]
fn test_single_phoneme_default() {
    let model = create_test_model();
    let map = create_test_phoneme_map();
    let lookup = TimingLookup::from_model(&model, &map);

    // Single fricative "f" has no cluster data, falls to generic,
    // then to default: Fricative => 200ms
    let cluster = ["f".to_string()];
    let result = lookup.get_timing(&cluster);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].1, 200);
}

#[test]
fn test_predict() {
    let model = create_test_model();
    let map = create_test_phoneme_map();
    let lookup = TimingLookup::from_model(&model, &map);

    let result = lookup.predict(&["k".to_string(), "s".to_string()]);

    assert_eq!(result.timings.len(), 2);
    assert_eq!(result.total_duration_ms, 95 + 110);
}

#[test]
fn test_generic_from_trees() {
    // Test the from_trees constructor with i32 keys
    use crate::classifier::PhonemeClassifier;

    struct I32Classifier;
    impl PhonemeClassifier<i32> for I32Classifier {
        fn classify(&self, phoneme: &i32) -> PhonemeType {
            if *phoneme < 10 {
                PhonemeType::Plosive
            } else {
                PhonemeType::Fricative
            }
        }
    }

    let mut cluster_tree = TreeNode::default();
    let mut node_1 = TreeNode::default();
    node_1.entries.push(vec![100]);
    cluster_tree.children.insert(1, node_1);

    let generic_tree = TreeNode::default();

    let lookup = TimingLookup::from_trees(
        "TestLib".into(),
        "TestLang".into(),
        "Default".into(),
        cluster_tree,
        generic_tree,
        Box::new(I32Classifier),
    );

    let result = lookup.get_timing(&[1]);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].1, 100);

    // Test fallback to default for unknown key
    let result = lookup.get_timing(&[5]);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].1, DEFAULT_PLOSIVE_MS);
}

/// Build a model and language inventory that exercise every level of the fallback
/// cascade with distinct, unambiguous timing values:
///
/// - Exact cluster `[k, s]`  -> avg `[95, 110]`
/// - Generic `[Plosive]`            -> `70`   (single-type fallback)
/// - Generic `[Plosive, Fricative]` -> `[95, 125]`
/// - Generic `[Sonorant, Sonorant]` -> `[150, 160]`
///
/// The exact cluster `[k, s]` means the prefix node `k` exists but carries no
/// entries of its own, which lets us test that a partial exact-path match
/// still falls through to the generic tree.
fn create_cascade_model() -> (TimingModel, LanguageInfo) {
    let mut model = TimingModel::new(TimingMetadata::new("Cascade", "English", "Default"));

    let mut ks = ClusterTiming::new(vec!["k".to_string(), "s".to_string()]);
    ks.add_sample(vec![95, 110]);
    model.cluster_timings.push(ks);

    let mut g_plosive = GenericTiming::new(vec![PhonemeType::Plosive]);
    g_plosive.add_sample(vec![70]);
    model.generic_timings.push(g_plosive);

    let mut g_plos_fric = GenericTiming::new(vec![PhonemeType::Plosive, PhonemeType::Fricative]);
    g_plos_fric.add_sample(vec![95, 125]);
    model.generic_timings.push(g_plos_fric);

    let mut g_son_son = GenericTiming::new(vec![PhonemeType::Sonorant, PhonemeType::Sonorant]);
    g_son_son.add_sample(vec![150, 160]);
    model.generic_timings.push(g_son_son);

    let mut map = LanguageInfo::new("Cascade");
    map.add_phoneme("k", PhonemeType::Plosive);
    map.add_phoneme("t", PhonemeType::Plosive);
    map.add_phoneme("s", PhonemeType::Fricative);
    map.add_phoneme("f", PhonemeType::Fricative);
    map.add_phoneme("m", PhonemeType::Sonorant);
    map.add_phoneme("n", PhonemeType::Sonorant);
    map.add_phoneme("a", PhonemeType::Vowel);

    (model, map)
}

#[test]
fn test_exact_takes_precedence_over_generic() {
    // `[k, s]` matches BOTH the exact cluster tree and the generic pattern
    // `[Plosive, Fricative]`. The exact tree must win.
    let (model, map) = create_cascade_model();
    let lookup = TimingLookup::from_model(&model, &map);

    let cluster = ["k".to_string(), "s".to_string()];
    let result = lookup.get_timing(&cluster);
    assert_eq!(result.len(), 2);
    // Exact values [95, 110], NOT the generic values [95, 125].
    assert_eq!(result[0].1, 95);
    assert_eq!(result[1].1, 110);
}

#[test]
fn test_unknown_cluster_falls_back_to_generic() {
    // `[t, f]` (Plosive, Fricative) is not in the exact tree but matches the
    // generic pattern `[Plosive, Fricative]`.
    let (model, map) = create_cascade_model();
    let lookup = TimingLookup::from_model(&model, &map);

    let cluster = ["t".to_string(), "f".to_string()];
    let result = lookup.get_timing(&cluster);
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].1, 95);
    assert_eq!(result[1].1, 125);
    // Keys must remain the original phonemes, not the types.
    assert_eq!(result[0].0, "t");
    assert_eq!(result[1].0, "f");
}

#[test]
fn test_partial_exact_path_falls_back_to_generic() {
    // `k` exists in the exact tree only as a prefix of `[k, s]`, so the `k`
    // node has no entries of its own. Lookup of the single phoneme `[k]`
    // must fall through to the generic single-type entry `[Plosive]` = 70,
    // rather than using the empty exact node or jumping to the default (100).
    let (model, map) = create_cascade_model();
    let lookup = TimingLookup::from_model(&model, &map);

    let cluster = ["k".to_string()];
    let result = lookup.get_timing(&cluster);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].1, 70);
}

#[test]
fn test_multi_phoneme_generic_match() {
    // `[m, n]` (Sonorant, Sonorant) is absent from the exact tree but matches
    // the generic pattern `[Sonorant, Sonorant]`.
    let (model, map) = create_cascade_model();
    let lookup = TimingLookup::from_model(&model, &map);

    let cluster = ["m".to_string(), "n".to_string()];
    let result = lookup.get_timing(&cluster);
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].1, 150);
    assert_eq!(result[1].1, 160);
}

#[test]
fn test_no_generic_single_falls_back_to_default() {
    // Single `a` (Vowel) has neither an exact nor a generic entry, so it must
    // fall back to the per-type default (Vowel => 400 ms).
    let (model, map) = create_cascade_model();
    let lookup = TimingLookup::from_model(&model, &map);

    let cluster = ["a".to_string()];
    let result = lookup.get_timing(&cluster);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].1, DEFAULT_VOWEL_MS);
}

#[test]
fn test_no_generic_multi_falls_back_to_recursive_split() {
    // `[s, t]` (Fricative, Plosive) is in neither the exact tree nor the
    // generic tree (which only has Plosive-rooted and Sonorant-rooted
    // patterns). It must recursively split into single phonemes, each of
    // which resolves independently:
    //   - `s` (Fricative): no exact, no generic => default Fricative (200)
    //   - `t` (Plosive):   no exact, but generic `[Plosive]` => 70
    let (model, map) = create_cascade_model();
    let lookup = TimingLookup::from_model(&model, &map);

    let cluster = ["s".to_string(), "t".to_string()];
    let result = lookup.get_timing(&cluster);
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].0, "s");
    assert_eq!(result[0].1, DEFAULT_FRICATIVE_MS);
    assert_eq!(result[1].0, "t");
    assert_eq!(result[1].1, 70);
}

#[test]
fn test_metadata_accessors() {
    let model = create_test_model();
    let map = create_test_phoneme_map();
    let lookup = TimingLookup::from_model(&model, &map);

    assert_eq!(lookup.library(), "TestLib");
    assert_eq!(lookup.language(), "English");
    assert_eq!(lookup.voice_color(), "Default");
    assert_eq!(lookup.model_version(), "1.0");
    assert!(lookup.created_at().is_none());
    assert!(lookup.source_files().is_none());
}

#[test]
fn test_validate_phonemes() {
    let map = create_test_phoneme_map();
    let valid = vec!["k".to_string(), "a".to_string()];
    let invalid = vec!["k".to_string(), "zzz".to_string(), "a".to_string()];

    assert!(validate_phonemes(&valid, &map).is_empty());
    let issues = validate_phonemes(&invalid, &map);
    assert_eq!(issues, vec!["zzz"]);
}
