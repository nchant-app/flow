
use super::*;

#[test]
fn test_insert_and_collect() {
    let mut node: TreeNode<String> = TreeNode::new();
    node.insert_path(["k".to_string(), "s".to_string()], &[vec![95, 110]]);
    node.insert_path(["k".to_string(), "s".to_string()], &[vec![100, 105]]);
    node.insert_path(["t".to_string()], &[vec![80]]);

    let mut paths = node.collect_paths();
    paths.sort_by_key(|(p, _)| p.join(","));

    assert_eq!(paths.len(), 2);
    assert_eq!(paths[0].0, vec!["k".to_string(), "s".to_string()]);
    assert_eq!(paths[0].1, vec![vec![95, 110], vec![100, 105]]);
    assert_eq!(paths[1].0, vec!["t".to_string()]);
    assert_eq!(paths[1].1, vec![vec![80]]);
}

#[test]
fn test_phoneme_tree_roundtrip() {
    use crate::model::TimingMetadata;

    let mut model = TimingModel::new(TimingMetadata::new("Lib", "English", "Default"));
    let mut ct = ClusterTiming::new(vec!["k".to_string(), "s".to_string()]);
    ct.add_sample(vec![95, 110]);
    model.cluster_timings.push(ct);
    let mut gt = GenericTiming::new(vec![PhonemeType::Plosive, PhonemeType::Fricative]);
    gt.add_sample(vec![100, 120]);
    model.generic_timings.push(gt);

    let tree = PhonemeTree::from_model(&model);
    let rebuilt = tree.to_model(model.version.clone(), model.metadata.clone());

    assert_eq!(rebuilt.cluster_timings.len(), 1);
    assert_eq!(rebuilt.cluster_timings[0].phonemes, vec!["k", "s"]);
    assert_eq!(rebuilt.cluster_timings[0].samples, vec![vec![95, 110]]);
    assert_eq!(rebuilt.generic_timings.len(), 1);
    assert_eq!(
        rebuilt.generic_timings[0].types,
        vec![PhonemeType::Plosive, PhonemeType::Fricative]
    );
}
