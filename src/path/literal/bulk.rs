//! Bulk exact-path matching and bounded top-k ranking.
//!
//! This keeps the `path::literal` ordering while preparing all rank inputs once
//! per candidate instead of recalculating them lazily during comparison.

use std::{cmp::Ordering, collections::BinaryHeap, sync::Arc};

use smallvec::SmallVec;

use crate::core::{
    InsertionVector, compare_key_asc, compare_score_desc, exact_distinct_coverage_by,
    exact_phrase_present, exact_subphrase_counts_vector,
};
use crate::path::{
    compound_term_may_match_with_tokens, normalize_path, ordered_path_evidence,
    ordered_path_evidence_with_tokens, path_depth, path_tokens, single_term_path_evidence,
};
use crate::text::iter_path_tokens;

pub type Score = f64;
pub type BulkSegmentVector = SmallVec<[usize; 8]>;

#[derive(Clone, Debug)]
pub struct BulkQuery {
    keywords: Arc<[String]>,
}

impl BulkQuery {
    pub fn new(keywords: &[String]) -> Self {
        Self {
            keywords: keywords.to_vec().into(),
        }
    }

    pub fn keywords(&self) -> &[String] {
        &self.keywords
    }
}

/// High-throughput exact path matcher that owns the normalized query.
///
/// Use `PreparedBulkPath` directly when a caller wants to manage query and
/// candidate preparation separately. This wrapper is the convenient canonical
/// execution-strategy API under `path::literal`.
#[derive(Clone, Debug)]
pub struct BulkMatcher {
    query: BulkQuery,
}

impl BulkMatcher {
    pub fn new(keywords: &[String]) -> Self {
        Self {
            query: BulkQuery::new(keywords),
        }
    }

    pub fn keywords(&self) -> &[String] {
        self.query.keywords()
    }

    pub fn match_path<'a>(&self, path: &'a str, score: Score) -> Option<BulkPathMatch<'a>> {
        exact_match(path, self.keywords(), score)
    }

    pub fn match_prepared<'a>(
        &self,
        path: &'a str,
        prepared: &PreparedBulkPath,
        score: Score,
    ) -> Option<BulkPathMatch<'a>> {
        prepared.exact_match(path, &self.query, score)
    }
}

/// Candidate preparation shared by repeated searches over the same source.
/// The original path remains owned by the caller; this stores only derived
/// data that is independent of the query.
#[derive(Clone, Debug)]
pub struct PreparedBulkPath {
    normalized: Option<String>,
    depth: usize,
}

impl PreparedBulkPath {
    pub fn new(path: &str) -> Self {
        let normalized = normalize_path(path);
        let depth = path_depth(&normalized);
        let normalized = (normalized != path).then_some(normalized);
        Self { normalized, depth }
    }

    fn normalized<'a>(&'a self, path: &'a str) -> &'a str {
        self.normalized.as_deref().unwrap_or(path)
    }

    pub fn exact_match<'a>(
        &self,
        path: &'a str,
        query: &BulkQuery,
        score: Score,
    ) -> Option<BulkPathMatch<'a>> {
        exact_match_prepared(path, self, query, score)
    }
}

#[derive(Clone, Debug)]
pub struct BulkPathMatch<'a> {
    pub path: &'a str,
    pub score: Score,
    pub segment: BulkSegmentVector,
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
    let query = BulkQuery::new(keywords);
    exact_match_with_query(path, &query, score)
}

pub fn exact_match_with_query<'a>(
    path: &'a str,
    query: &BulkQuery,
    score: Score,
) -> Option<BulkPathMatch<'a>> {
    let prepared = PreparedBulkPath::new(path);
    exact_match_prepared(path, &prepared, query, score)
}

pub fn exact_match_prepared<'a>(
    path: &'a str,
    prepared: &PreparedBulkPath,
    query: &BulkQuery,
    score: Score,
) -> Option<BulkPathMatch<'a>> {
    let keywords = query.keywords();
    if let [keyword] = keywords {
        return exact_match_single_term(path, prepared, keyword, score);
    }

    let normalized_path = prepared.normalized(path);
    let evidence = if keywords
        .iter()
        .all(|keyword| normalized_path.contains(keyword))
    {
        ordered_path_evidence(normalized_path, keywords)?
    } else {
        let path_tokens = path_tokens(normalized_path);
        if !keywords.iter().all(|keyword| {
            normalized_path.contains(keyword)
                || compound_term_may_match_with_tokens(&path_tokens, keyword)
        }) {
            return None;
        }
        ordered_path_evidence_with_tokens(normalized_path, &path_tokens, keywords)?
    };
    let components = normalized_path
        .split(['/', '\\'])
        .filter(|component| !component.is_empty())
        .map(|component| iter_path_tokens(component).collect::<SmallVec<[_; 8]>>())
        .collect::<SmallVec<[_; 8]>>();
    let exact_coverage = exact_distinct_coverage_by(
        keywords,
        components.iter().map(|tokens| tokens.as_slice()),
        |query, candidate| candidate.contains(query),
    );
    let ordered_compound_coverage = evidence.ordered_compound_coverage;

    Some(BulkPathMatch {
        path,
        score,
        segment: evidence.segment.into_iter().collect(),
        position: evidence.position,
        insertions: InsertionVector::new(evidence.insertions),
        exact_phrase_present: exact_phrase_present(
            keywords,
            components.iter().map(|tokens| tokens.as_slice()),
        ),
        exact_subphrase_counts_vector: exact_subphrase_counts_vector(
            keywords,
            components.iter().map(|tokens| tokens.as_slice()),
        ),
        exact_coverage,
        ordered_compound_coverage,
        path_depth: prepared.depth,
    })
}

fn exact_match_single_term<'a>(
    path: &'a str,
    prepared: &PreparedBulkPath,
    keyword: &str,
    score: Score,
) -> Option<BulkPathMatch<'a>> {
    let evidence = single_term_path_evidence(prepared.normalized(path), keyword)?;

    let mut segment = BulkSegmentVector::new();
    segment.resize(evidence.segment + 1, 0);
    segment[evidence.segment] = 1;

    let mut position = [0; 3];
    position[evidence.position.min(2)] = 1;

    Some(BulkPathMatch {
        path,
        score,
        segment,
        position,
        insertions: InsertionVector::from_slice(&[evidence.insertions]),
        exact_phrase_present: evidence.exact_token_prefix,
        exact_subphrase_counts_vector: Vec::new(),
        exact_coverage: usize::from(!evidence.compound),
        ordered_compound_coverage: usize::from(evidence.compound),
        path_depth: prepared.depth,
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
    use crate::path::literal::{compare_exact_path_matches, exact_match as regular_exact_match};

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
    fn single_term_fast_path_preserves_path_exact_ordering() {
        let keywords = vec!["a".to_string()];
        let paths = [
            "/home/lewis/a.out",
            "/home/lewis/.android",
            "/home/lewis/.nv/ComputeCache/a",
            "/home/lewis/.aspera/connect/bin/ascp",
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

        assert_eq!(
            fast.iter()
                .map(|candidate| candidate.path)
                .collect::<Vec<_>>(),
            exact
                .iter()
                .map(|candidate| candidate.path)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn direct_substring_coverage_precedes_ordered_compound_coverage() {
        let keywords = vec!["geminipackages".to_string()];
        let direct = super::exact_match("/home/xxgeminipackages", &keywords, 0.0).unwrap();
        let compound = super::exact_match("/home/gemini-cli/packages", &keywords, 100.0).unwrap();

        assert_eq!(direct.exact_coverage, 1);
        assert_eq!(direct.ordered_compound_coverage, 0);
        assert_eq!(compound.exact_coverage, 0);
        assert_eq!(compound.ordered_compound_coverage, 1);
        assert_eq!(direct.compare(&compound), Ordering::Less);
    }

    #[test]
    fn multi_token_substring_coverage_counts_each_distinct_query_token() {
        let keywords = ["licat", "launcher"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let matched = super::exact_match("/home/applicationlauncher", &keywords, 0.0).unwrap();

        assert_eq!(matched.exact_coverage, 2);
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

        assert_eq!(matched.segment.as_slice(), &[0, 0, 0, 0, 0, 1, 1]);
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
        assert_eq!(matched.segment.as_slice(), &[0, 0, 0, 0, 0, 1]);
        assert_eq!(matched.position, [1, 0, 0]);
        assert_eq!(matched.insertions.aligned(), &[5]);
    }

    #[test]
    fn trailing_separator_does_not_create_an_empty_basename_segment() {
        let keywords = vec!["codex".to_string()];
        let shallow = super::exact_match("/home/lewis/.codex/", &keywords, 0.0).unwrap();
        let nested =
            super::exact_match("/home/lewis/Dev/wordelites/.codex", &keywords, 0.0).unwrap();

        assert_eq!(shallow.segment.as_slice(), &[1]);
        assert_eq!(shallow.path_depth, 3);
        assert!(shallow.compare(&nested).is_lt());
    }

    #[test]
    fn bulk_matcher_wraps_query_and_prepared_path_matching() {
        let keywords = vec!["application".to_string()];
        let matcher = BulkMatcher::new(&keywords);
        let path = "/home/lewis/Dev/applicationlauncher";
        let prepared = PreparedBulkPath::new(path);

        assert_eq!(matcher.keywords(), keywords.as_slice());
        assert!(matcher.match_path(path, 0.0).is_some());
        assert!(matcher.match_prepared(path, &prepared, 0.0).is_some());
    }
}
