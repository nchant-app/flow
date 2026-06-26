//! Prefix-tree (trie) structures for phoneme timing lookups.
//!
//! This module provides the built-in support for the two timing trees used
//! throughout Maghni:
//!
//! - a **specific** cluster tree, keyed by phoneme (e.g. X-SAMPA `String`s),
//!   holding exact-match duration samples, and
//! - a **generic** fallback tree, keyed by [`PhonemeType`], holding type-based
//!   duration samples used when an exact cluster isn't found.
//!
//! [`TreeNode`] is the generic trie primitive (generic over the key type `K`),
//! while [`PhonemeTree`] bundles both trees and converts to and from a
//! [`TimingModel`]. These types are serde-(de)serializable so they can be used
//! in YAML pipelines and reused by downstream crates with their own key types.

use std::collections::HashMap;
use std::hash::Hash;

use serde::{Deserialize, Serialize};

use crate::model::{
    ClusterTiming, GenericTiming, PhonemeType, TimingMetadata, TimingModel,
};

/// A node in a phoneme timing trie.
///
/// Generic over the key type `K`, allowing the same structure to be keyed by
/// X-SAMPA `String`s (the open-source default), a [`PhonemeType`] (the generic
/// fallback tree), or any typed phoneme key supplied by a downstream crate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeNode<K: Eq + Hash> {
    /// Child nodes, keyed by the next phoneme in the cluster.
    pub children: HashMap<K, TreeNode<K>>,
    /// Duration samples in milliseconds. Each inner `Vec<u16>` is one
    /// observation: `[duration_phoneme_1, duration_phoneme_2, ...]`.
    pub entries: Vec<Vec<u16>>,
}

impl<K: Eq + Hash> Default for TreeNode<K> {
    fn default() -> Self {
        Self {
            children: HashMap::new(),
            entries: Vec::new(),
        }
    }
}

impl<K: Eq + Hash + Clone> TreeNode<K> {
    /// Create an empty node.
    #[must_use]
    pub fn new() -> Self {
        Self {
            children: HashMap::new(),
            entries: Vec::new(),
        }
    }

    /// Insert duration `samples` at the given key `path`, creating intermediate
    /// nodes as needed.
    pub fn insert_path<I>(&mut self, path: I, samples: &[Vec<u16>])
    where
        I: IntoIterator<Item = K>,
    {
        let mut node = self;
        for key in path {
            node = node.children.entry(key).or_default();
        }
        node.entries.extend_from_slice(samples);
    }

    /// Recursively collect every populated path in the tree.
    ///
    /// Returns one `(path, entries)` pair for each node that has duration
    /// samples, where `path` is the sequence of keys from the root to that node.
    #[must_use]
    pub fn collect_paths(&self) -> Vec<(Vec<K>, Vec<Vec<u16>>)> {
        let mut results = Vec::new();
        self.collect_into(Vec::new(), &mut results);
        results
    }

    /// Recursive helper for [`TreeNode::collect_paths`].
    fn collect_into(&self, prefix: Vec<K>, results: &mut Vec<(Vec<K>, Vec<Vec<u16>>)>) {
        if !self.entries.is_empty() {
            results.push((prefix.clone(), self.entries.clone()));
        }
        for (key, child) in &self.children {
            let mut path = prefix.clone();
            path.push(key.clone());
            child.collect_into(path, results);
        }
    }
}

/// The two timing trees bundled together: a specific cluster tree keyed by
/// X-SAMPA phoneme strings, and a generic fallback tree keyed by [`PhonemeType`].
///
/// This is the native, ready-to-use representation of a [`TimingModel`]'s
/// timing data as prefix trees, suitable for efficient O(k) cluster lookup.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PhonemeTree {
    /// Exact-match tree keyed by X-SAMPA phoneme strings.
    pub cluster_tree: TreeNode<String>,
    /// Type-based fallback tree keyed by [`PhonemeType`].
    pub generic_tree: TreeNode<PhonemeType>,
}

impl PhonemeTree {
    /// Build the cluster and generic trees from a [`TimingModel`].
    #[must_use]
    pub fn from_model(model: &TimingModel) -> Self {
        let mut cluster_tree = TreeNode::default();
        for timing in &model.cluster_timings {
            cluster_tree.insert_path(timing.phonemes.iter().cloned(), &timing.samples);
        }

        let mut generic_tree = TreeNode::default();
        for timing in &model.generic_timings {
            generic_tree.insert_path(timing.types.iter().copied(), &timing.samples);
        }

        Self {
            cluster_tree,
            generic_tree,
        }
    }

    /// Flatten the trees back into a [`TimingModel`] with the given metadata.
    #[must_use]
    pub fn to_model(&self, version: impl Into<String>, metadata: TimingMetadata) -> TimingModel {
        let cluster_timings = self
            .cluster_tree
            .collect_paths()
            .into_iter()
            .map(|(phonemes, samples)| ClusterTiming { phonemes, samples })
            .collect();

        let generic_timings = self
            .generic_tree
            .collect_paths()
            .into_iter()
            .map(|(types, samples)| GenericTiming { types, samples })
            .collect();

        TimingModel {
            version: version.into(),
            metadata,
            cluster_timings,
            generic_timings,
        }
    }
}

#[cfg(test)]
mod tests {
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
}
