//! Bulk exact-path matching and bounded top-k ranking.
//!
//! This keeps the `path::exact` ordering while preparing all rank inputs once
//! per candidate instead of recalculating them lazily during comparison.

use std::{cmp::Ordering, collections::BinaryHeap};

use crate::core::{
    InsertionVector, compare_key_asc, compare_score_desc, exact_distinct_coverage,
    exact_phrase_present, exact_subphrase_counts_vector,
};
use crate::path::{normalize_path, ordered_path_evidence, path_depth};
use crate::text::split_path_tokens;

pub type Score = f64;

#[derive(Clone, Debug)]
pub struct BulkPathMatch<'a> {
    pub path: &'a str,
    pub score: Score,
    pub segment: Vec<usize>,
    pub position: [usize; 3],
    pub insertions: InsertionVector,
    pub exact_phrase_present: bool,
    pub exact_subphrase_counts_vector: Vec<usize>,
    pub exact_coverage: usize,
    pub ordered_compound_coverage: usize,
    pub path_depth: usize,
}

impl BulkPathMatch<'_> {
    pub fn compare(&self, other: &Self) -> Ordering {
        other
            .exact_phrase_present
            .cmp(&self.exact_phrase_present)
            .then_with(|| {
                other
                    .exact_subphrase_counts_vector
                    .cmp(&self.exact_subphrase_counts_vector)
            })
            .then_with(|| other.exact_coverage.cmp(&self.exact_coverage))
            .then_with(|| {
                other
                    .ordered_compound_coverage
                    .cmp(&self.ordered_compound_coverage)
            })
            .then_with(|| other.segment.cmp(&self.segment))
            .then_with(|| other.position.cmp(&self.position))
            .then_with(|| compare_score_desc(self.score, other.score))
            .then_with(|| self.insertions.cmp(&other.insertions))
            .then_with(|| self.path_depth.cmp(&other.path_depth))
            .then_with(|| compare_key_asc(self.path, other.path))
    }
}

pub fn exact_match<'a>(
    path: &'a str,
    keywords: &[String],
    score: Score,
) -> Option<BulkPathMatch<'a>> {
    let normalized_path = normalize_path(path);
    let evidence = ordered_path_evidence(&normalized_path, keywords)?;
    let components = normalized_path
        .split(['/', '\\'])
        .filter(|component| !component.is_empty())
        .map(|component| split_path_tokens(component).into_iter().collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let exact_coverage = exact_distinct_coverage(keywords, components.iter().map(Vec::as_slice));
    let ordered_compound_coverage = evidence.ordered_compound_coverage;

    Some(BulkPathMatch {
        path,
        score,
        segment: evidence.segment,
        position: evidence.position,
        insertions: InsertionVector::new(evidence.insertions),
        exact_phrase_present: exact_phrase_present(keywords, components.iter().map(Vec::as_slice)),
        exact_subphrase_counts_vector: exact_subphrase_counts_vector(
            keywords,
            components.iter().map(Vec::as_slice),
        ),
        exact_coverage,
        ordered_compound_coverage,
        path_depth: path_depth(&normalized_path),
    })
}

pub fn sort_matches(matches: &mut [BulkPathMatch<'_>]) {
    matches.sort_unstable_by(BulkPathMatch::compare);
}

pub struct BulkTopK<'a> {
    limit: usize,
    heap: BinaryHeap<WorstFirst<BulkPathMatch<'a>>>,
}

impl<'a> BulkTopK<'a> {
    pub fn new(limit: usize) -> Self {
        Self {
            limit,
            heap: BinaryHeap::with_capacity(limit),
        }
    }

    pub fn push(&mut self, candidate: BulkPathMatch<'a>) {
        if self.limit == 0 {
            return;
        }
        if self.heap.len() < self.limit {
            self.heap.push(WorstFirst(candidate));
        } else if candidate.compare(&self.heap.peek().expect("non-empty top-k heap").0)
            == Ordering::Less
        {
            self.heap.pop();
            self.heap.push(WorstFirst(candidate));
        }
    }

    pub fn merge(mut self, other: Self) -> Self {
        debug_assert_eq!(self.limit, other.limit);
        for candidate in other.heap.into_iter().map(|entry| entry.0) {
            self.push(candidate);
        }
        self
    }

    pub fn into_sorted(self) -> Vec<BulkPathMatch<'a>> {
        let mut matches = self
            .heap
            .into_iter()
            .map(|entry| entry.0)
            .collect::<Vec<_>>();
        sort_matches(&mut matches);
        matches
    }
}

/// Retains and sorts the best `limit` candidates without fully sorting the
/// candidate corpus. The retained prefix has the same order as `sort_matches`.
pub fn select_top_matches(matches: &mut Vec<BulkPathMatch<'_>>, limit: usize) {
    if matches.len() <= limit {
        sort_matches(matches);
        return;
    }

    let mut top = BulkTopK::new(limit);
    for candidate in matches.drain(..) {
        top.push(candidate);
    }
    *matches = top.into_sorted();
}

struct WorstFirst<T>(T);

impl<T: HasBulkOrder> PartialEq for WorstFirst<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.compare_bulk(&other.0).is_eq()
    }
}

impl<T: HasBulkOrder> Eq for WorstFirst<T> {}

impl<T: HasBulkOrder> PartialOrd for WorstFirst<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: HasBulkOrder> Ord for WorstFirst<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.compare_bulk(&other.0)
    }
}

trait HasBulkOrder {
    fn compare_bulk(&self, other: &Self) -> Ordering;
}

impl HasBulkOrder for BulkPathMatch<'_> {
    fn compare_bulk(&self, other: &Self) -> Ordering {
        self.compare(other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::path::exact::{compare_exact_path_matches, exact_match as regular_exact_match};

    #[test]
    fn preserves_path_exact_ordering() {
        let keywords = ["app", "launch", "target"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let paths = [
            "/home/applaunchtarget",
            "/home/app-launch-target",
            "/home/target/app-launch",
        ];

        let mut exact = paths
            .iter()
            .filter_map(|path| regular_exact_match(path, &keywords, 1.0))
            .collect::<Vec<_>>();
        exact.sort_unstable_by(|left, right| compare_exact_path_matches(left, right, &keywords));

        let mut fast = paths
            .iter()
            .filter_map(|path| super::exact_match(path, &keywords, 1.0))
            .collect::<Vec<_>>();
        fast.sort_unstable_by(BulkPathMatch::compare);

        let exact_paths = exact
            .into_iter()
            .map(|candidate| candidate.path)
            .collect::<Vec<_>>();
        let fast_paths = fast
            .into_iter()
            .map(|candidate| candidate.path)
            .collect::<Vec<_>>();
        assert_eq!(fast_paths, exact_paths);
    }

    #[test]
    fn streaming_top_k_matches_the_fully_sorted_prefix() {
        let keywords = ["app", "launch"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let paths = [
            "/home/app-launch",
            "/home/applicationlauncher",
            "/home/projects/app-launch",
            "/home/app/launch",
        ];

        let mut fully_sorted = paths
            .iter()
            .filter_map(|path| super::exact_match(path, &keywords, 1.0))
            .collect::<Vec<_>>();
        sort_matches(&mut fully_sorted);

        let mut top = BulkTopK::new(2);
        for candidate in paths
            .iter()
            .filter_map(|path| super::exact_match(path, &keywords, 1.0))
        {
            top.push(candidate);
        }
        let top = top.into_sorted();

        assert_eq!(
            top.iter()
                .map(|candidate| candidate.path)
                .collect::<Vec<_>>(),
            fully_sorted
                .iter()
                .take(2)
                .map(|candidate| candidate.path)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn exact_terms_can_match_entirely_in_ancestor_components() {
        let keywords = ["gemini", "packages"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let path = "/home/lewis/repos/gemini-cli/packages/core/dist/docs/cli/using-agent-skills.md";
        let matched = super::exact_match(path, &keywords, 0.0).unwrap();

        assert_eq!(matched.segment, vec![0, 0, 0, 0, 0, 1, 1]);
        assert_eq!(matched.position, [2, 0, 0]);
        assert_eq!(matched.insertions.aligned(), &[0, 0]);
    }

    #[test]
    fn compact_query_token_matches_ordered_path_tokens() {
        let keywords = vec!["geminipackages".to_string()];
        let path = "/home/lewis/repos/gemini-cli/packages/core/dist/docs/cli/using-agent-skills.md";
        let matched = super::exact_match(path, &keywords, 0.0).unwrap();

        assert_eq!(matched.exact_coverage, 0);
        assert_eq!(matched.ordered_compound_coverage, 1);
        assert_eq!(matched.segment, vec![0, 0, 0, 0, 0, 1]);
        assert_eq!(matched.position, [1, 0, 0]);
        assert_eq!(matched.insertions.aligned(), &[5]);
    }

    #[test]
    fn trailing_separator_does_not_create_an_empty_basename_segment() {
        let keywords = vec!["codex".to_string()];
        let shallow = super::exact_match("/home/lewis/.codex/", &keywords, 0.0).unwrap();
        let nested =
            super::exact_match("/home/lewis/Dev/wordelites/.codex", &keywords, 0.0).unwrap();

        assert_eq!(shallow.segment, vec![1]);
        assert_eq!(shallow.path_depth, 3);
        assert!(shallow.compare(&nested).is_lt());
    }
}
