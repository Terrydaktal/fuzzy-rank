use std::sync::Arc;

use crate::core::{
    AlignmentCostVector, InsertionVector, OperationProfile, TypoCostVector, alignment_cost_milli,
    bounded_damerau_levenshtein, increment_simple_position_vector, max_typos,
    simple_position_vector,
};
use crate::path::literal::exact_match_prepared;
use crate::path::{PathExactRelevance, PreparedPath, Score, is_single_path_token};
use crate::ranking::{
    PathTypoSortKey, compare_path_lexical_sort_keys, compare_path_typo_sort_keys,
    compare_path_typo_tiebreak_sort_keys, path_typo_keys_are_ambiguous,
};
use crate::text::{split_path_tokens, to_lowercase};
use crate::token_match::{
    PositionedToken, SequenceMatchScratch, SingleTokenMatchScratch, TokenMatchScratch,
    aligned_token_distance_with_scratch, best_token_match, best_token_match_chars_with_scratch,
    choose_path_token_match_with_scratch, partitioned_token_distance_with_scratch,
};

/// Number of scalar features exposed by the path reranker.
pub const PATH_RANK_FEATURE_COUNT: usize = 22;

/// Stable names for [`PathRankFeatures`]. Positive values mean stronger
/// evidence for the candidate; negative values represent penalties.
pub const PATH_RANK_FEATURE_NAMES: [&str; PATH_RANK_FEATURE_COUNT] = [
    "match_exact",
    "exact_basename_component",
    "exact_component",
    "exact_phrase",
    "exact_subphrase_presence",
    "exact_coverage",
    "ordered_compound_coverage",
    "alignment_quality",
    "order_quality",
    "segment_proximity",
    "position_quality",
    "caller_score_log",
    "typo_quality",
    "insertion_quality",
    "substitution_quality",
    "deletion_quality",
    "transposition_quality",
    "keyboard_quality",
    "distance_quality",
    "ratio_quality",
    "path_shallowness",
    "query_signal",
];

/// A fixed-size, direction-normalized representation of a path match.
///
/// This is intentionally separate from the default lexicographic comparator.
/// Callers can use it for a learned or caller-specific reranker without
/// reimplementing path tokenization, exact evidence, or edit accounting.
#[derive(Clone, Debug, PartialEq)]
pub struct PathRankFeatures {
    values: [f64; PATH_RANK_FEATURE_COUNT],
}

impl PathRankFeatures {
    pub fn from_match(candidate: &PathMatch<'_>) -> Self {
        let alignment = candidate.alignment_costs.worst_first();
        let typo = candidate.typo_costs.worst_first();
        let insertions = candidate.insertions.worst_first();
        let segment_depth = candidate
            .segment
            .iter()
            .position(|count| *count != 0)
            .unwrap_or(candidate.segment.len()) as f64;
        let position_quality = candidate.position[0] as f64 + candidate.position[1] as f64 * 0.5
            - candidate.position[2] as f64 * 0.5;
        let exact_subphrase_presence = candidate
            .exact_subphrase_counts_vector()
            .iter()
            .filter(|count| **count != 0)
            .count() as f64;
        let exact_coverage = candidate.exact_distinct_coverage() as f64;
        let ordered_compound_coverage = candidate.ordered_compound_coverage() as f64;
        let worst_alignment = alignment.first().copied().unwrap_or(0) as f64;
        let worst_typo = typo.first().copied().unwrap_or(0) as f64;
        let worst_insertions = insertions.first().copied().unwrap_or(0) as f64;
        let path_depth = crate::path::path_depth(candidate.path) as f64;
        let query_signal =
            (exact_coverage + exact_subphrase_presence + ordered_compound_coverage).min(16.0);

        Self {
            values: [
                f64::from(candidate.exact),
                f64::from(candidate.exact_whole_basename_component()),
                f64::from(candidate.exact_whole_component()),
                f64::from(candidate.exact_whole_phrase()),
                exact_subphrase_presence,
                exact_coverage,
                ordered_compound_coverage,
                -worst_alignment / 1000.0,
                -(candidate.order_distance as f64),
                -segment_depth,
                position_quality,
                candidate.score.max(0.0).ln_1p(),
                -worst_typo,
                -worst_insertions,
                -(candidate.operations.substitutions as f64),
                -(candidate.operations.deletions as f64),
                -(candidate.operations.transpositions as f64),
                -(candidate.operations.keyboard_distance as f64),
                -(candidate.distance as f64),
                -candidate.ratio,
                -path_depth,
                query_signal,
            ],
        }
    }

    pub fn values(&self) -> &[f64; PATH_RANK_FEATURE_COUNT] {
        &self.values
    }

    pub fn get(&self, index: usize) -> Option<f64> {
        self.values.get(index).copied()
    }

    pub fn score(&self, model: &PathRankModel) -> f64 {
        let score = model.bias
            + self
                .values
                .iter()
                .zip(model.weights)
                .map(|(feature, weight)| {
                    let feature = if feature.is_finite() { *feature } else { 0.0 };
                    let weight = if weight.is_finite() { weight } else { 0.0 };
                    feature * weight
                })
                .sum::<f64>();
        if score.is_finite() { score } else { 0.0 }
    }
}

/// A compact native linear reranker for path candidates.
///
/// The model contains no runtime or allocator requirements. It is suitable
/// for a caller that wants to load learned weights and score only a small
/// candidate set after fuzzy-rank has performed matching.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PathRankModel {
    pub bias: f64,
    pub weights: [f64; PATH_RANK_FEATURE_COUNT],
}

impl PathRankModel {
    pub const fn new(bias: f64, weights: [f64; PATH_RANK_FEATURE_COUNT]) -> Self {
        Self { bias, weights }
    }

    /// A conservative built-in baseline for callers that do not have a
    /// trained model yet. It rewards strong lexical evidence and penalizes
    /// structural distance while leaving the normal comparator available as
    /// the final deterministic tie-breaker.
    pub const fn baseline() -> Self {
        Self::new(
            0.0,
            [
                10_000.0, 5_000.0, 1_000.0, 500.0, 100.0, 50.0, 25.0, 100.0, 10.0, 20.0, 10.0, 2.0,
                20.0, 5.0, 10.0, 10.0, 10.0, 2.0, 10.0, 1.0, 1.0, 5.0,
            ],
        )
    }

    pub fn score(&self, features: &PathRankFeatures) -> f64 {
        features.score(self)
    }

    pub fn is_active(&self) -> bool {
        // A constant bias cancels when two candidates are compared.
        self.weights
            .iter()
            .any(|weight| weight.is_finite() && *weight != 0.0)
    }

    pub fn learn_pair(&mut self, preferred: &PathRankFeatures, other: &PathRankFeatures) -> bool {
        self.learn_pair_with(
            preferred,
            other,
            crate::learning::DEFAULT_LEARNING_RATE,
            crate::learning::DEFAULT_MARGIN,
        )
    }

    pub fn learn_preference(
        &mut self,
        preferred: &PathRankFeatures,
        alternatives: &[PathRankFeatures],
    ) -> usize {
        alternatives
            .iter()
            .filter(|alternative| self.learn_pair(preferred, alternative))
            .count()
    }

    pub fn learn_pair_with(
        &mut self,
        preferred: &PathRankFeatures,
        other: &PathRankFeatures,
        learning_rate: f64,
        margin: f64,
    ) -> bool {
        crate::learning::learn_pair(
            &mut self.weights,
            preferred.values(),
            other.values(),
            learning_rate,
            margin,
        )
    }
}

impl Default for PathRankModel {
    fn default() -> Self {
        Self::new(0.0, [0.0; PATH_RANK_FEATURE_COUNT])
    }
}

#[derive(Clone, Debug)]
pub struct PathMatch<'a> {
    pub path: &'a str,
    pub exact: bool,
    pub distance: usize,
    pub order_distance: usize,
    pub ratio: f64,
    pub segment: Vec<usize>,
    pub position: [usize; 3],
    pub operations: OperationProfile,
    pub alignment_costs: AlignmentCostVector,
    pub typo_costs: TypoCostVector,
    pub insertions: InsertionVector,
    pub score: Score,
    exact_relevance: Option<PathExactRelevance<'a>>,
}

impl<'a> PathMatch<'a> {
    fn sort_key(&self) -> PathTypoSortKey<'_> {
        PathTypoSortKey {
            distance: self.distance,
            order_distance: self.order_distance,
            operations: self.operations,
            alignment_costs: &self.alignment_costs,
            typo_costs: &self.typo_costs,
            insertions: &self.insertions,
            segment: &self.segment,
            position: &self.position,
            score: self.score,
            key: self.path,
        }
    }

    pub fn compare(&self, other: &Self) -> std::cmp::Ordering {
        compare_exact_relevance(self, other)
            .then_with(|| compare_path_typo_sort_keys(&self.sort_key(), &other.sort_key()))
    }

    /// Compare only the structural and lexical path evidence that a learned
    /// reranker must not override.
    pub fn compare_lexical(&self, other: &Self) -> std::cmp::Ordering {
        compare_exact_relevance(self, other)
            .then_with(|| compare_path_lexical_sort_keys(&self.sort_key(), &other.sort_key()))
    }

    /// Compare the remaining keys only after `compare_lexical` has returned equal.
    pub fn compare_after_lexical(&self, other: &Self) -> std::cmp::Ordering {
        compare_path_typo_tiebreak_sort_keys(&self.sort_key(), &other.sort_key())
    }

    pub fn exact_whole_phrase(&self) -> bool {
        self.exact_relevance
            .as_ref()
            .is_some_and(PathExactRelevance::whole_phrase)
    }

    pub fn exact_whole_component(&self) -> bool {
        self.exact_relevance
            .as_ref()
            .is_some_and(PathExactRelevance::whole_component)
    }

    pub fn exact_whole_basename_component(&self) -> bool {
        self.exact_relevance
            .as_ref()
            .is_some_and(PathExactRelevance::whole_basename_component)
    }

    pub fn exact_subphrase_counts_vector(&self) -> &[usize] {
        self.exact_relevance
            .as_ref()
            .map_or(&[], PathExactRelevance::run_vector)
    }

    pub fn exact_distinct_coverage(&self) -> usize {
        self.exact_relevance
            .as_ref()
            .map_or(0, PathExactRelevance::distinct_coverage)
    }

    pub fn ordered_compound_coverage(&self) -> usize {
        self.exact_relevance
            .as_ref()
            .map_or(0, PathExactRelevance::ordered_compound_coverage)
    }

    pub fn rank_features(&self) -> PathRankFeatures {
        PathRankFeatures::from_match(self)
    }

    fn has_exact_token_signal(&self) -> bool {
        self.exact_relevance
            .as_ref()
            .is_some_and(PathExactRelevance::has_exact_token_signal)
    }
}

pub struct TypoQuery {
    text: String,
    keywords: Arc<[String]>,
    query_chars: Arc<[char]>,
    len: usize,
    limit: usize,
    token_count: usize,
    allow_typos: bool,
}

impl TypoQuery {
    pub fn new(query: &str) -> Option<Self> {
        Self::from_lowered_text(to_lowercase(query))
    }

    pub fn from_keywords(keywords: &[String]) -> Option<Self> {
        if keywords.is_empty() || keywords.iter().any(String::is_empty) {
            return None;
        }

        let mut query = String::new();
        for (index, keyword) in keywords.iter().enumerate() {
            if index > 0 {
                query.push(' ');
            }
            query.push_str(&to_lowercase(keyword));
        }
        Self::from_lowered_text(query)
    }

    fn from_lowered_text(query: String) -> Option<Self> {
        let len = query.chars().count();
        let keywords = split_path_tokens(&query)
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        if keywords.is_empty() {
            return None;
        }

        let token_count = keywords.len();
        Some(Self {
            query_chars: query.chars().collect::<Vec<_>>().into(),
            text: query,
            keywords: keywords.into(),
            len,
            limit: max_typos(len),
            token_count,
            allow_typos: true,
        })
    }

    pub fn with_typo_fallback(mut self, allow_typos: bool) -> Self {
        self.allow_typos = allow_typos;
        self
    }

    pub fn token_count(&self) -> usize {
        self.token_count
    }

    /// Cheap candidate gates for callers scanning a large path database.
    ///
    /// These only reject candidates. The full path matcher remains
    /// authoritative, so callers must still run the matching method after a
    /// gate succeeds.
    pub fn may_match_exact_path(&self, path: &str) -> bool {
        if is_ascii_lowercase_path(path) {
            return self.may_match_exact_normalized(path);
        }

        let normalized = crate::path::normalize_path(path);
        self.may_match_exact_normalized(&normalized)
    }

    pub fn may_match_typo_path(&self, path: &str) -> bool {
        if !self.allow_typos || self.len <= 1 {
            return false;
        }

        let normalized = if is_ascii_lowercase_path(path) {
            std::borrow::Cow::Borrowed(path)
        } else {
            std::borrow::Cow::Owned(crate::path::normalize_path(path))
        };

        // The full matcher cannot succeed when the entire candidate path is
        // shorter than the query by more than the allowed edit budget. This
        // lower-bound check is lossless; unlike a character-signal heuristic,
        // it cannot reject a valid typo or fragmented path match.
        normalized.chars().count() >= self.len.saturating_sub(self.limit)
    }

    pub fn may_match_exact_prepared(&self, path: &PreparedPath<'_>) -> bool {
        self.may_match_exact_normalized(path.normalized())
    }

    fn may_match_exact_normalized(&self, normalized: &str) -> bool {
        self.keywords.iter().all(|keyword| {
            normalized.contains(keyword)
                || crate::path::compound_term_may_match(normalized, keyword)
        })
    }

    pub fn best_match<'a>(&self, path: &'a str, score: Score) -> Option<PathMatch<'a>> {
        let prepared = PreparedPath::new(path);
        self.best_match_prepared(&prepared, score)
    }

    pub fn prepare_path<'a>(&self, path: &'a str) -> PreparedPath<'a> {
        PreparedPath::new(path)
    }

    pub fn best_match_prepared<'a>(
        &self,
        prepared: &PreparedPath<'a>,
        score: Score,
    ) -> Option<PathMatch<'a>> {
        let mut scratch = PathMatchScratch::default();
        self.best_match_prepared_with_scratch(prepared, score, &mut scratch)
    }

    pub fn best_match_prepared_with_scratch<'a>(
        &self,
        prepared: &PreparedPath<'a>,
        score: Score,
        scratch: &mut PathMatchScratch,
    ) -> Option<PathMatch<'a>> {
        best_match_with_prepared_path(prepared, score, self, scratch)
    }

    /// Matches only the exact path class without evaluating typo distance.
    pub fn best_exact_match<'a>(&self, path: &'a str, score: Score) -> Option<PathMatch<'a>> {
        let prepared = PreparedPath::new(path);
        self.best_exact_match_prepared(&prepared, score)
    }

    pub fn best_exact_match_prepared<'a>(
        &self,
        prepared: &PreparedPath<'a>,
        score: Score,
    ) -> Option<PathMatch<'a>> {
        exact_path_match(prepared, score, self)
    }

    /// Matches only the typo path class without repeating the exact check.
    pub fn best_typo_match<'a>(&self, path: &'a str, score: Score) -> Option<PathMatch<'a>> {
        let prepared = PreparedPath::new(path);
        self.best_typo_match_prepared(&prepared, score)
    }

    pub fn best_typo_match_prepared<'a>(
        &self,
        prepared: &PreparedPath<'a>,
        score: Score,
    ) -> Option<PathMatch<'a>> {
        let mut scratch = PathMatchScratch::default();
        self.best_typo_match_prepared_with_scratch(prepared, score, &mut scratch)
    }

    pub fn best_typo_match_prepared_with_scratch<'a>(
        &self,
        prepared: &PreparedPath<'a>,
        score: Score,
        scratch: &mut PathMatchScratch,
    ) -> Option<PathMatch<'a>> {
        if !self.allow_typos || self.len <= 1 {
            return None;
        }
        best_match_inner(
            prepared,
            score,
            &self.text,
            self.len,
            self.limit,
            self.token_count,
            &self.keywords,
            &self.query_chars,
            scratch,
        )
        .map(|candidate| with_exact_relevance(candidate, Arc::clone(&self.keywords), prepared))
    }

    /// Compatibility alias: skips exact admission, but retains lazy exact evidence
    /// so the resulting rank is identical to other typo-matching APIs.
    pub fn best_typo_match_prepared_without_exact<'a>(
        &self,
        prepared: &PreparedPath<'a>,
        score: Score,
        scratch: &mut PathMatchScratch,
    ) -> Option<PathMatch<'a>> {
        self.best_typo_match_prepared_with_scratch(prepared, score, scratch)
    }

    pub fn best_basename_match<'a>(&self, path: &'a str, score: Score) -> Option<PathMatch<'a>> {
        let prepared = PreparedPath::new(path);
        best_basename_match_with_query(&prepared, score, self)
    }
}

fn is_ascii_lowercase_path(path: &str) -> bool {
    path.is_ascii() && path.bytes().all(|byte| !byte.is_ascii_uppercase())
}

#[derive(Default)]
pub struct PathMatchScratch {
    token: TokenMatchScratch,
    sequence: SequenceMatchScratch,
    single: SingleTokenMatchScratch,
}

pub fn best_match<'a>(path: &'a str, score: Score, query: &str) -> Option<PathMatch<'a>> {
    let query = TypoQuery::new(query)?;
    query.best_match(path, score)
}

pub fn sort_matches(matches: &mut [PathMatch<'_>]) {
    matches.sort_unstable_by(PathMatch::compare);
}

pub fn is_ambiguous(first: &PathMatch<'_>, second: &PathMatch<'_>) -> bool {
    compare_exact_relevance(first, second).is_eq()
        && path_typo_keys_are_ambiguous(&first.sort_key(), &second.sort_key())
}

pub fn query_from_keywords(keywords: &[String]) -> Option<String> {
    if keywords.is_empty() || keywords.iter().any(String::is_empty) {
        return None;
    }

    Some(
        keywords
            .iter()
            .map(to_lowercase)
            .collect::<Vec<_>>()
            .join(" "),
    )
}

fn best_match_with_prepared_path<'a>(
    prepared: &PreparedPath<'a>,
    score: Score,
    query: &TypoQuery,
    scratch: &mut PathMatchScratch,
) -> Option<PathMatch<'a>> {
    let candidate = match exact_path_match(prepared, score, query) {
        Some(candidate) => candidate,
        None if query.allow_typos && query.len > 1 => best_match_inner(
            prepared,
            score,
            &query.text,
            query.len,
            query.limit,
            query.token_count,
            &query.keywords,
            &query.query_chars,
            scratch,
        )?,
        None => return None,
    };
    Some(if candidate.exact_relevance.is_some() {
        candidate
    } else {
        with_exact_relevance(candidate, Arc::clone(&query.keywords), prepared)
    })
}

fn best_basename_match_with_query<'a>(
    prepared: &PreparedPath<'a>,
    score: Score,
    query: &TypoQuery,
) -> Option<PathMatch<'a>> {
    let candidate = match exact_path_match(prepared, score, query) {
        Some(candidate) => candidate,
        None if query.allow_typos && query.len > 1 => best_basename_match_inner(
            prepared,
            score,
            &query.text,
            query.len,
            query.limit,
            query.token_count,
            query.limit,
        )?,
        None => return None,
    };
    Some(if candidate.exact_relevance.is_some() {
        candidate
    } else {
        with_exact_relevance(candidate, Arc::clone(&query.keywords), prepared)
    })
}

fn with_exact_relevance<'a>(
    mut candidate: PathMatch<'a>,
    query_tokens: Arc<[String]>,
    prepared: &PreparedPath<'a>,
) -> PathMatch<'a> {
    candidate.exact_relevance = Some(PathExactRelevance::with_prepared_path(
        prepared,
        query_tokens,
    ));
    candidate
}

fn exact_path_match<'a>(
    prepared: &PreparedPath<'a>,
    score: Score,
    query: &TypoQuery,
) -> Option<PathMatch<'a>> {
    let matched = exact_match_prepared(prepared, Arc::clone(&query.keywords), score)?;
    let insertion_costs = matched.insertions.aligned().to_vec();
    let insertions = insertion_costs.iter().sum();
    Some(PathMatch {
        path: prepared.path(),
        exact: true,
        distance: 0,
        order_distance: 0,
        ratio: 0.0,
        segment: matched.segment,
        position: matched.position,
        operations: OperationProfile {
            insertions,
            ..OperationProfile::default()
        },
        alignment_costs: AlignmentCostVector::new(vec![0; query.keywords.len()]),
        typo_costs: TypoCostVector::new(vec![0; query.keywords.len()]),
        insertions: InsertionVector::new(insertion_costs),
        score,
        exact_relevance: Some(matched.exact_relevance),
    })
}

fn segment_vector(segment: usize, count: usize) -> Vec<usize> {
    let mut vector = vec![0; segment + 1];
    vector[segment] = count;
    vector
}

#[allow(clippy::too_many_arguments)]
fn best_match_inner<'a>(
    prepared: &PreparedPath<'a>,
    score: Score,
    query: &str,
    query_len: usize,
    limit: usize,
    query_token_count: usize,
    query_tokens: &[String],
    query_chars: &[char],
    scratch: &mut PathMatchScratch,
) -> Option<PathMatch<'a>> {
    let component_count = prepared.component_count();
    if component_count == 0 {
        return None;
    }
    if query_token_count == 1 {
        return best_match_single_token(
            prepared,
            score,
            query,
            query_len,
            limit,
            component_count,
            query_tokens,
            query_chars,
            scratch,
        );
    }

    let basename_idx = component_count - 1;
    let basename = prepared.component(basename_idx);
    let mut ctx = MatchContext {
        path: prepared.path(),
        score,
        query,
        query_len,
        limit,
        token_count: query_token_count,
        query_tokens,
        query_chars,
        scratch,
    };

    let mut best = None;
    update_best(
        &mut best,
        candidate_for_token_sequence(&mut ctx, basename, 0),
    );
    update_best(
        &mut best,
        candidate_for_compound_component(&mut ctx, basename, 0),
    );
    update_best(
        &mut best,
        candidate_for_component_sequence(prepared, &mut ctx, component_count),
    );

    for idx in 0..basename_idx {
        let component = prepared.component(idx);
        let path_position = basename_idx - idx;
        update_best(
            &mut best,
            candidate_for_compound_component(&mut ctx, component, path_position),
        );
    }
    best
}

fn best_basename_match_inner<'a>(
    prepared: &PreparedPath<'a>,
    score: Score,
    query: &str,
    query_len: usize,
    limit: usize,
    token_count: usize,
    full_limit: usize,
) -> Option<PathMatch<'a>> {
    let component_count = prepared.component_count();
    if component_count == 0 {
        return None;
    }

    let basename_idx = component_count - 1;
    let basename = prepared.component(basename_idx);
    let query_chars = query.chars().collect::<Vec<_>>();
    let query_tokens = split_path_tokens(query)
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let mut token_scratch = PathMatchScratch::default();
    let mut ctx = MatchContext {
        path: prepared.path(),
        score,
        query,
        query_len,
        limit,
        token_count,
        query_tokens: &query_tokens,
        query_chars: &query_chars,
        scratch: &mut token_scratch,
    };
    let mut best = candidate_for_text(&ctx, basename, 0);
    update_best(
        &mut best,
        candidate_for_partitioned_single_query(&mut ctx, basename, 0),
    );
    for token in prepared.tokens(basename_idx) {
        update_best(&mut best, candidate_for_text(&ctx, token, 0));
        update_best(
            &mut best,
            candidate_for_partitioned_single_query(&mut ctx, token, 0),
        );
    }
    best.filter(|candidate| candidate.distance <= full_limit)
}

#[allow(clippy::too_many_arguments)]
fn best_match_single_token<'a>(
    prepared: &PreparedPath<'a>,
    score: Score,
    query: &str,
    query_len: usize,
    limit: usize,
    component_count: usize,
    query_tokens: &[String],
    query_chars: &[char],
    scratch: &mut PathMatchScratch,
) -> Option<PathMatch<'a>> {
    let basename_idx = component_count - 1;
    let basename = prepared.component(basename_idx);
    let mut ctx = MatchContext {
        path: prepared.path(),
        score,
        query,
        query_len,
        limit,
        token_count: 1,
        query_tokens,
        query_chars,
        scratch,
    };

    let mut best = candidate_for_single_token(&mut ctx, basename, 0);
    if best.as_ref().is_some_and(|candidate| {
        candidate
            .alignment_costs
            .worst_first()
            .first()
            .is_some_and(|cost| *cost <= 1_000)
    }) {
        return best;
    }
    for token in prepared.tokens(basename_idx) {
        if token != basename {
            update_best(&mut best, candidate_for_single_token(&mut ctx, token, 0));
        }
    }
    if best.as_ref().is_some_and(|candidate| {
        candidate
            .alignment_costs
            .worst_first()
            .first()
            .is_some_and(|cost| *cost <= 1_000)
    }) {
        return best;
    }

    for idx in 0..basename_idx {
        let component = prepared.component(idx);
        let path_position = basename_idx - idx;
        update_best(
            &mut best,
            candidate_for_single_token(&mut ctx, component, path_position),
        );
        for token in prepared.tokens(idx) {
            if token != component {
                update_best(
                    &mut best,
                    candidate_for_single_token(&mut ctx, token, path_position),
                );
            }
        }
    }

    update_best(
        &mut best,
        candidate_for_single_token(&mut ctx, prepared.normalized(), basename_idx + 1),
    );
    best
}

fn update_best<'a>(best: &mut Option<PathMatch<'a>>, candidate: Option<PathMatch<'a>>) {
    match (best.as_ref(), candidate) {
        (_, None) => {}
        (None, Some(candidate)) => *best = Some(candidate),
        (Some(current), Some(candidate)) if candidate.compare(current).is_lt() => {
            *best = Some(candidate)
        }
        _ => {}
    }
}

struct MatchContext<'a, 'q> {
    path: &'a str,
    score: Score,
    query: &'q str,
    query_len: usize,
    limit: usize,
    token_count: usize,
    query_tokens: &'q [String],
    query_chars: &'q [char],
    scratch: &'q mut PathMatchScratch,
}

fn candidate_for_single_token<'a>(
    ctx: &mut MatchContext<'a, '_>,
    candidate: &str,
    path_position: usize,
) -> Option<PathMatch<'a>> {
    if !is_single_path_token(candidate) {
        return None;
    }

    let candidate_len = candidate.chars().count();
    let matched = ctx
        .scratch
        .single
        .choice(ctx.query_chars, candidate, ctx.limit)?;
    let ratio = matched.distance as f64 / ctx.query_len.max(candidate_len) as f64;
    let minimum_breaks = usize::from(ctx.query_len.abs_diff(candidate_len) <= ctx.limit) + 1;
    if ratio > 0.5 || matched.fragmented && matched.internal_breaks < minimum_breaks {
        return None;
    }

    Some(PathMatch {
        path: ctx.path,
        exact: false,
        distance: matched.distance,
        order_distance: 0,
        ratio,
        segment: segment_vector(path_position, 1),
        position: simple_position_vector([matched.position_rank]),
        operations: matched.operations,
        alignment_costs: AlignmentCostVector::from_slice(&[matched.alignment_cost]),
        typo_costs: TypoCostVector::from_slice(&[matched.operations.core_typo_cost()]),
        insertions: InsertionVector::from_slice(&[matched.operations.insertions]),
        score: ctx.score,
        exact_relevance: None,
    })
}

fn candidate_for_text<'a>(
    ctx: &MatchContext<'a, '_>,
    candidate: &str,
    path_position: usize,
) -> Option<PathMatch<'a>> {
    if candidate.is_empty() {
        return None;
    }

    let candidate_len = candidate.chars().count();
    let query_token_count = ctx.token_count;

    if query_token_count == 1 && !is_single_path_token(candidate) {
        return None;
    }

    if query_token_count == 1
        && let Some(start) = candidate.find(ctx.query)
    {
        let start_chars = candidate[..start].chars().count();
        let end_chars = start_chars + ctx.query_len;
        let insertions = candidate_len.saturating_sub(ctx.query_len);
        let ratio = insertions as f64 / ctx.query_len.max(candidate_len) as f64;
        if ratio <= 0.5 {
            let position_rank = if start_chars == 0 {
                0
            } else if end_chars == candidate_len {
                1
            } else {
                2
            };
            return Some(PathMatch {
                path: ctx.path,
                exact: false,
                distance: insertions,
                order_distance: 0,
                ratio,
                segment: segment_vector(path_position, 1),
                position: simple_position_vector([position_rank]),
                operations: OperationProfile {
                    insertions,
                    ..OperationProfile::default()
                },
                alignment_costs: AlignmentCostVector::from_slice(&[0]),
                typo_costs: TypoCostVector::from_slice(&[0]),
                insertions: InsertionVector::from_slice(&[insertions]),
                score: ctx.score,
                exact_relevance: None,
            });
        }
    }

    if query_token_count != 1 {
        return None;
    }

    if ctx.limit == 0 || ctx.query_len.abs_diff(candidate_len) > ctx.limit {
        return None;
    }

    let (distance, operations) = bounded_damerau_levenshtein(candidate, ctx.query, ctx.limit)?;
    let max_len = ctx.query_len.max(candidate_len) as f64;
    let ratio = distance as f64 / max_len;
    if ratio > 0.5 {
        return None;
    }

    let position_rank = if query_token_count == 1 {
        token_relative_match_shape(ctx.query, candidate, ctx.limit).unwrap_or((0, 0))
    } else {
        (0, 0)
    }
    .0;

    Some(PathMatch {
        path: ctx.path,
        exact: false,
        distance,
        order_distance: 0,
        ratio,
        segment: segment_vector(path_position, query_token_count),
        position: simple_position_vector(std::iter::repeat_n(position_rank, query_token_count)),
        operations,
        alignment_costs: AlignmentCostVector::from_slice(&[alignment_cost_milli(
            operations, false, 0,
        )]),
        typo_costs: TypoCostVector::from_slice(&[operations.core_typo_cost()]),
        insertions: InsertionVector::from_slice(&[operations.insertions]),
        score: ctx.score,
        exact_relevance: None,
    })
}

fn candidate_for_token_sequence<'a>(
    ctx: &mut MatchContext<'a, '_>,
    candidate: &str,
    path_position: usize,
) -> Option<PathMatch<'a>> {
    let query_tokens = ctx.query_tokens;
    let candidate_tokens = split_path_tokens(candidate);
    if query_tokens.len() < 2 || query_tokens.len() != candidate_tokens.len() {
        return None;
    }
    let query_token_count = query_tokens.len();

    let mut distance = 0;
    let mut position = [0; 3];
    let mut operations = OperationProfile::default();
    let mut typo_costs = Vec::with_capacity(query_token_count);
    let mut insertion_costs = Vec::with_capacity(query_token_count);
    let mut alignment_costs = Vec::with_capacity(query_token_count);
    for (query_idx, candidate_token) in candidate_tokens.into_iter().enumerate() {
        let query_token = query_tokens[query_idx].as_str();
        let query_chars = query_token.chars().collect::<Vec<_>>();
        let remaining = ctx.limit.checked_sub(distance)?;
        let (cost, penalty, position_rank, _, token_operations) =
            best_token_match_chars_with_scratch(
                &query_chars,
                candidate_token,
                remaining,
                &mut ctx.scratch.token,
            )?;
        distance += cost;
        increment_simple_position_vector(&mut position, position_rank, 1);
        let token_operations = OperationProfile {
            insertions: token_operations.insertions + penalty,
            ..token_operations
        };
        operations = operations.merge(token_operations);
        typo_costs.push(token_operations.core_typo_cost());
        insertion_costs.push(token_operations.insertions);
        alignment_costs.push(alignment_cost_milli(token_operations, false, 0));
    }

    let ratio = distance as f64 / ctx.query_len as f64;
    if ratio > 0.5 {
        return None;
    }

    Some(PathMatch {
        path: ctx.path,
        exact: false,
        distance,
        order_distance: 0,
        ratio,
        segment: segment_vector(path_position, query_token_count),
        position,
        operations,
        alignment_costs: AlignmentCostVector::new(alignment_costs),
        typo_costs: TypoCostVector::new(typo_costs),
        insertions: InsertionVector::new(insertion_costs),
        score: ctx.score,
        exact_relevance: None,
    })
}

fn candidate_for_compound_component<'a>(
    ctx: &mut MatchContext<'a, '_>,
    candidate: &str,
    path_position: usize,
) -> Option<PathMatch<'a>> {
    let query_tokens = ctx.query_tokens;
    let candidate_tokens = split_path_tokens(candidate);
    if query_tokens.len() < 2 || candidate_tokens.len() != 1 {
        return None;
    }

    let query_token_refs = query_tokens.iter().map(String::as_str).collect::<Vec<_>>();
    let (distance, _structure, position, operations, typo_costs, insertion_costs, alignment_costs) =
        partitioned_token_distance_with_scratch(
            &query_token_refs,
            candidate,
            ctx.limit,
            &mut ctx.scratch.sequence,
        )?;
    let ratio = distance as f64 / ctx.query_len.max(candidate.chars().count()) as f64;
    if ratio > 0.5 {
        return None;
    }

    Some(PathMatch {
        path: ctx.path,
        exact: false,
        distance,
        order_distance: 0,
        ratio,
        segment: segment_vector(path_position, query_tokens.len()),
        position,
        operations,
        alignment_costs: AlignmentCostVector::new(alignment_costs),
        typo_costs: TypoCostVector::new(typo_costs),
        insertions: InsertionVector::new(insertion_costs),
        score: ctx.score,
        exact_relevance: None,
    })
}

fn candidate_for_partitioned_single_query<'a>(
    ctx: &mut MatchContext<'a, '_>,
    candidate: &str,
    path_position: usize,
) -> Option<PathMatch<'a>> {
    if split_path_tokens(candidate).len() != 1 {
        return None;
    }
    let candidate_len = candidate.chars().count();
    let matched = choose_path_token_match_with_scratch(
        ctx.query_chars,
        candidate,
        ctx.limit,
        &mut ctx.scratch.token,
    )?;
    let ratio = matched.distance as f64 / ctx.query_len.max(candidate_len) as f64;
    let minimum_breaks = usize::from(ctx.query_len.abs_diff(candidate_len) <= ctx.limit) + 1;
    if ratio > 0.5 || matched.fragmented && matched.internal_breaks < minimum_breaks {
        return None;
    }

    Some(PathMatch {
        path: ctx.path,
        exact: false,
        distance: matched.distance,
        order_distance: 0,
        ratio,
        segment: segment_vector(path_position, 1),
        position: simple_position_vector([matched.position_rank]),
        operations: matched.operations,
        alignment_costs: AlignmentCostVector::from_slice(&[matched.alignment_cost]),
        typo_costs: TypoCostVector::from_slice(&[matched.operations.core_typo_cost()]),
        insertions: InsertionVector::from_slice(&[matched.operations.insertions]),
        score: ctx.score,
        exact_relevance: None,
    })
}

fn candidate_for_component_sequence<'a>(
    prepared: &PreparedPath<'a>,
    ctx: &mut MatchContext<'a, '_>,
    component_count: usize,
) -> Option<PathMatch<'a>> {
    let query_tokens = ctx.query_tokens;
    if query_tokens.len() < 2 {
        return None;
    }
    let query_token_refs = query_tokens.iter().map(String::as_str).collect::<Vec<_>>();

    let basename_idx = component_count - 1;
    let candidate_tokens: Vec<_> = (0..component_count)
        .flat_map(|idx| {
            let path_position = basename_idx - idx;
            prepared.tokens(idx).map(move |token| PositionedToken {
                token,
                segment: path_position,
            })
        })
        .collect();
    if candidate_tokens.len() < query_tokens.len() {
        return None;
    }

    let (
        distance,
        order_distance,
        segment,
        position,
        _breaks,
        _structure,
        operations,
        typo_costs,
        insertion_costs,
        alignment_costs,
    ) = aligned_token_distance_with_scratch(
        &query_token_refs,
        &candidate_tokens,
        ctx.limit,
        &mut ctx.scratch.sequence,
    )?;
    let ratio = distance as f64 / ctx.query_len as f64;
    if ratio > 0.5 {
        return None;
    }

    Some(PathMatch {
        path: ctx.path,
        exact: false,
        distance,
        order_distance,
        ratio,
        segment,
        position,
        operations,
        alignment_costs: AlignmentCostVector::new(alignment_costs),
        typo_costs: TypoCostVector::new(typo_costs),
        insertions: InsertionVector::new(insertion_costs),
        score: ctx.score,
        exact_relevance: None,
    })
}

fn compare_exact_relevance(left: &PathMatch<'_>, right: &PathMatch<'_>) -> std::cmp::Ordering {
    let left_has_signal = left.has_exact_token_signal();
    let right_has_signal = right.has_exact_token_signal();

    if !left_has_signal && !right_has_signal {
        return std::cmp::Ordering::Equal;
    }

    right
        .exact_whole_basename_component()
        .cmp(&left.exact_whole_basename_component())
        .then_with(|| {
            right
                .exact_whole_component()
                .cmp(&left.exact_whole_component())
        })
        .then_with(|| right.exact_whole_phrase().cmp(&left.exact_whole_phrase()))
        .then_with(|| {
            right
                .exact_subphrase_counts_vector()
                .cmp(left.exact_subphrase_counts_vector())
        })
        .then_with(|| {
            right
                .exact_distinct_coverage()
                .cmp(&left.exact_distinct_coverage())
        })
        .then_with(|| {
            right
                .ordered_compound_coverage()
                .cmp(&left.ordered_compound_coverage())
        })
}

fn token_relative_match_shape(
    query: &str,
    candidate: &str,
    limit: usize,
) -> Option<(usize, usize)> {
    let mut best = None;
    for token in split_path_tokens(candidate) {
        let Some((_, penalty, position_rank, _, _)) = best_token_match(query, token, limit) else {
            continue;
        };
        let shape = (position_rank, penalty);
        best = Some(match best {
            None => shape,
            Some(current) if shape < current => shape,
            Some(current) => current,
        });
        if best == Some((0, 0)) {
            break;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literal_gate_keeps_alternative_compound_prefixes() {
        for (query, path) in [
            ("ababcd", "/ab/aba/bcd"),
            ("ababcd", "/ab/aba/bcdef"),
            ("abbc", "/a/ab/bc"),
            ("\u{e9}ab", "/\u{e9}/\u{e9}a/b"),
        ] {
            let query = TypoQuery::new(query).unwrap();
            let prepared = query.prepare_path(path);
            assert!(query.best_exact_match_prepared(&prepared, 0.0).is_some());
            assert!(query.may_match_exact_prepared(&prepared), "{path}");
        }
    }

    #[test]
    fn prepared_execution_strategies_preserve_exact_evidence_and_features() {
        let paths = [
            "/audit/alpha/gamma",
            "/audit/alpha/gamut",
            "/audit/alphi/gamma",
            "/home/lewis/Dev/applicationlauncher",
        ];
        let mut scratch = PathMatchScratch::default();
        for text in [
            "alpha gamam",
            "gamam alpha",
            "dev applaunch",
            "applaunch",
            "fuxxy",
        ] {
            let query = TypoQuery::new(text).unwrap();
            for path in paths {
                let prepared = query.prepare_path(path);
                let normal = query.best_typo_match_prepared(&prepared, 3.0);
                let fast =
                    query.best_typo_match_prepared_without_exact(&prepared, 3.0, &mut scratch);
                assert_eq!(normal.is_some(), fast.is_some(), "{text}: {path}");
                if let (Some(normal), Some(fast)) = (normal, fast) {
                    assert!(normal.compare(&fast).is_eq(), "{text}: {path}");
                    assert_eq!(
                        normal.rank_features(),
                        fast.rank_features(),
                        "{text}: {path}"
                    );
                    assert_eq!(
                        normal.exact_distinct_coverage(),
                        fast.exact_distinct_coverage()
                    );
                    assert_eq!(normal.operations, fast.operations);
                }
                let normal = query.best_match_prepared(&prepared, 3.0);
                let cached = query.best_match_prepared_with_scratch(&prepared, 3.0, &mut scratch);
                assert_eq!(
                    normal.as_ref().map(PathMatch::rank_features),
                    cached.as_ref().map(PathMatch::rank_features)
                );
            }
        }
    }

    #[test]
    fn model_alignment_feature_uses_worst_token_not_first_query_token() {
        let first = best_match("/audit/alpha/gamma", 0.0, "alpha gamam").unwrap();
        let second = best_match("/audit/alpha/gamma", 0.0, "gamam alpha").unwrap();
        assert_eq!(first.alignment_costs.aligned(), &[0, 1000]);
        assert_eq!(second.alignment_costs.aligned(), &[1000, 0]);
        assert_eq!(first.rank_features().get(7), Some(-1.0));
        assert_eq!(second.rank_features().get(7), Some(-1.0));
    }

    #[test]
    fn typo_matches_basename() {
        let candidate = best_match("/home/lewis/xfce4-terminal", 10.0, "xfce4-terinal").unwrap();
        assert_eq!(candidate.distance, 1);
    }

    #[test]
    fn single_token_component_matches_use_token_relative_position() {
        let candidate = best_match("/home/lewis/tasks/config", 10.0, "onfig").unwrap();
        assert_eq!(candidate.segment, [1]);
        assert_eq!(candidate.position, [0, 1, 0]);
    }

    #[test]
    fn basename_token_prefixes_do_not_count_as_typos() {
        let candidate = best_match("/home/lewis/xfce4-terminal", 10.0, "xzfce-ter").unwrap();
        assert_eq!(candidate.distance, 1);
    }

    #[test]
    fn spaced_query_tokens_can_match_single_compound_token() {
        let candidate = best_match("/home/lewis/applicationlauncher", 10.0, "app laucnh").unwrap();
        assert_eq!(candidate.segment, [2]);
        assert_eq!(candidate.distance, 1);
    }

    #[test]
    fn missing_space_query_matches_single_compound_token_via_internal_breaks_and_insertions() {
        let candidate = best_match("/home/lewis/applicationlauncher", 10.0, "applaunch").unwrap();
        assert_eq!(candidate.operations.core_typo_cost(), 0);
        assert_eq!(candidate.operations.insertions, 10);
    }

    #[test]
    fn compact_query_token_matches_ordered_path_tokens_without_typo_cost() {
        let candidate = best_match(
            "/home/lewis/repos/gemini-cli/packages/core/dist/docs/cli/using-agent-skills.md",
            10.0,
            "geminipackages",
        )
        .unwrap();

        assert!(candidate.exact);
        assert_eq!(candidate.exact_distinct_coverage(), 0);
        assert_eq!(candidate.ordered_compound_coverage(), 1);
        assert_eq!(candidate.typo_costs.aligned(), [0]);
        assert_eq!(candidate.insertions.aligned(), [5]);
    }

    #[test]
    fn whole_path_component_beats_exact_token_in_longer_component() {
        let mut matches = [
            best_match(
                "/home/lewis/Dev/wordelites/wordelites-server",
                1000.0,
                "wordelites",
            )
            .unwrap(),
            best_match("/home/lewis/Dev/wordelites", 0.0, "wordelites").unwrap(),
        ];

        sort_matches(&mut matches);

        assert_eq!(matches[0].path, "/home/lewis/Dev/wordelites");
        assert!(matches[0].exact_whole_basename_component());
        assert!(matches[0].exact_whole_component());
        assert!(!matches[1].exact_whole_basename_component());
        assert!(matches[1].exact_whole_component());
    }

    #[test]
    fn repeated_suffixes_do_not_become_exact_compound_matches() {
        for query in ["wordelitess", "wordelitesss"] {
            let mut matches = [
                best_match("/home/lewis/Dev/wordelites/wordelites-server", 1.2, query).unwrap(),
                best_match("/home/lewis/Dev/wordelites", 171.2, query).unwrap(),
            ];

            assert!(!matches[0].exact);
            assert_eq!(matches[0].ordered_compound_coverage(), 0);
            sort_matches(&mut matches);
            assert_eq!(matches[0].path, "/home/lewis/Dev/wordelites");
        }
    }

    #[test]
    fn trailing_separator_does_not_demote_an_exact_basename() {
        let mut matches = [
            best_match("/home/lewis/Dev/wordelites/.codex", 0.0, "codex").unwrap(),
            best_match("/home/lewis/.codex/", 0.0, "codex").unwrap(),
        ];
        sort_matches(&mut matches);

        assert_eq!(matches[0].path, "/home/lewis/.codex/");
        assert_eq!(matches[0].segment, [1]);
    }

    #[test]
    fn spaced_query_tokens_can_match_component_sequence() {
        let candidate = best_match("/home/lewis/tasks/config", 10.0, "tasks cinfig").unwrap();
        assert_eq!(candidate.segment, [1, 1]);
        assert_eq!(candidate.distance, 1);
        assert_eq!(candidate.typo_costs.aligned(), [0, 2]);
    }

    #[test]
    fn reordered_exact_tokens_have_zero_typo_cost() {
        let candidate = best_match("/home/lewis/repos/codex", 10.0, "codex rep").unwrap();
        assert_eq!(candidate.distance, 0);
        assert_eq!(candidate.order_distance, 1);
        assert_eq!(candidate.operations.core_typo_cost(), 0);
        assert_eq!(candidate.typo_costs.aligned(), [0, 0]);
    }

    #[test]
    fn token_match_penalty_tracks_unmatched_surrounding_chars() {
        assert_eq!(
            best_token_match("cinfig", "redragonmouseconfig", 3),
            Some((
                1,
                13,
                1,
                6,
                OperationProfile {
                    substitutions: 1,
                    insertions: 0,
                    deletions: 0,
                    transpositions: 0,
                    keyboard_distance: 2,
                    substitution_penalty: 2,
                    insertion_penalty: 0,
                },
            ))
        );
    }

    #[test]
    fn component_sequence_can_match_substring_inside_token() {
        let candidate = best_match(
            "/home/lewis/tasks/redragonmouseconfig",
            10.0,
            "tasks cinfig",
        )
        .unwrap();
        assert_eq!(candidate.segment, [1, 1]);
        assert_eq!(candidate.distance, 1);
    }

    #[test]
    fn multiple_tokens_can_match_the_same_non_basename_component() {
        let candidate = best_match(
            "/home/lewis/Dev/applicationlauncher/target/release",
            10.0,
            "ap laun",
        )
        .unwrap();
        assert_eq!(candidate.distance, 0);
        assert_eq!(candidate.segment, [0, 0, 2]);
    }

    #[test]
    fn extra_query_character_beats_substitution_for_paths() {
        let mut matches = [
            best_match("/home/lewis/abdef", 10.0, "abcdef").unwrap(),
            best_match("/home/lewis/abqdef", 10.0, "abcdef").unwrap(),
        ];
        sort_matches(&mut matches);
        assert_eq!(matches[0].path, "/home/lewis/abdef");
        assert_eq!(matches[0].distance, matches[1].distance);
        assert_eq!(matches[0].operations.deletions, 1);
        assert_eq!(matches[0].operations.insertions, 0);
        assert_eq!(matches[0].operations.core_typo_cost(), 6);
        assert!(matches[1].operations.core_typo_cost() > matches[0].operations.core_typo_cost());
    }

    #[test]
    fn long_queries_use_half_length_typo_limit() {
        let candidate = best_match("/home/lewis/xfce4-terminal", 10.0, "xgce4-tremriianl").unwrap();
        assert_eq!(candidate.distance, 5);
    }

    #[test]
    fn reversed_nine_token_paths_are_aligned_without_a_permutation_cutoff() {
        let candidate = best_match(
            "/ii/hh/gg/ff/ee/dd/cc/bb/aa",
            10.0,
            "aa bb cc dd ee ff gg hh ii",
        )
        .unwrap();
        assert_eq!(candidate.distance, 0);
        assert_eq!(candidate.order_distance, 36);
    }

    #[test]
    fn typo_matches_component_token() {
        let candidate = best_match("/home/lewis/xfce4-terminal", 10.0, "x4ce4").unwrap();
        assert_eq!(candidate.distance, 1);
    }

    #[test]
    fn five_character_queries_allow_half_length_typos() {
        let candidate = best_match("/home/lewis/xfce4-terminal", 10.0, "zgce4").unwrap();
        assert_eq!(candidate.distance, 2);
        assert_eq!(candidate.ratio, 0.4);
    }

    #[test]
    fn typo_matching_does_not_require_shared_character_bigrams() {
        let candidate = best_match("/home/lewis/fuzzy-rank", 10.0, "fuxxy").unwrap();
        assert_eq!(candidate.path, "/home/lewis/fuzzy-rank");
        assert_eq!(candidate.distance, 2);
    }

    #[test]
    fn typo_gate_is_lossless_for_short_signal_matches() {
        let query = TypoQuery::new("fuxxy").unwrap();
        assert!(query.may_match_typo_path("/home/lewis/fuzzy-rank"));
    }

    #[test]
    fn single_character_queries_only_use_the_exact_fast_path() {
        let exact = best_match("/home/lewis/foobar", 10.0, "f").unwrap();
        assert!(exact.exact);
        assert!(best_match("/home/lewis/foobar", 10.0, "x").is_none());
    }

    #[test]
    fn exact_path_class_beats_a_higher_score_typo() {
        let mut matches = [
            best_match("/home/lewis/cctv", 1.0, "cctv").unwrap(),
            best_match("/home/lewis/cctx", 100.0, "cctv").unwrap(),
        ];
        sort_matches(&mut matches);
        assert!(matches[0].exact);
        assert_eq!(matches[0].path, "/home/lewis/cctv");
    }

    #[test]
    fn rank_features_have_stable_shape_and_direction() {
        let match_ = best_match("/home/lewis/applicationlauncher", 10.0, "applaunch").unwrap();
        let features = match_.rank_features();
        assert_eq!(features.values().len(), PATH_RANK_FEATURE_COUNT);
        assert_eq!(PATH_RANK_FEATURE_NAMES.len(), PATH_RANK_FEATURE_COUNT);
        assert!(features.get(11).unwrap() > 0.0);
        assert!(features.get(13).unwrap() < 0.0);
    }

    #[test]
    fn lexical_comparison_protects_close_typos_from_unrelated_basename_matches() {
        let zoxide = best_match("/home/lewis/repos/zoxide", 0.0, "zoxie").unwrap();
        let unrelated =
            best_match("/home/lewis/tasks/windowsvirtualmachine", 0.0, "zoxie").unwrap();
        assert!(zoxide.compare_lexical(&unrelated).is_lt());

        let qwen = best_match("/home/lewis/tasks/qwen", 0.0, "wqwen").unwrap();
        let unrelated =
            best_match("/home/lewis/tasks/tplinkwirelesstransmitter", 0.0, "wqwen").unwrap();
        assert!(qwen.compare_lexical(&unrelated).is_lt());
    }

    #[test]
    fn typo_fallback_can_be_disabled_without_disabling_exact_matches() {
        let query = TypoQuery::new("cctx").unwrap().with_typo_fallback(false);
        assert!(query.best_match("/home/lewis/cctx", 1.0).unwrap().exact);
        assert!(query.best_match("/home/lewis/cctv", 1.0).is_none());
    }

    #[test]
    fn short_queries_allow_one_typo() {
        let candidate = best_match("/home/lewis/foo", 10.0, "foa").unwrap();
        assert_eq!(candidate.distance, 1);
    }

    #[test]
    fn core_typo_cost_breaks_equal_distance_ties_for_paths() {
        let mut matches = vec![
            best_match("/home/lewis/terminal", 10.0, "terminak").unwrap(),
            best_match("/home/lewis/terminap", 10.0, "terminak").unwrap(),
        ];
        sort_matches(&mut matches);
        assert_eq!(matches[0].path, "/home/lewis/terminal");
        assert!(!is_ambiguous(&matches[0], &matches[1]));
    }

    #[test]
    fn frecency_resolves_otherwise_ambiguous_matches() {
        let mut matches = vec![
            best_match("/home/lewis/repos/xfce4-terminal", 20.0, "xgce4-tremriianl").unwrap(),
            best_match(
                "/home/lewis/Dev/config/xfce4-terminal",
                10.0,
                "xgce4-tremriianl",
            )
            .unwrap(),
        ];
        sort_matches(&mut matches);
        assert_eq!(matches[0].path, "/home/lewis/repos/xfce4-terminal");
        assert!(!is_ambiguous(&matches[0], &matches[1]));
    }

    #[test]
    fn basename_position_beats_frecency_when_typo_quality_ties() {
        let mut matches = vec![
            best_match("/home/xfce4/project", 100.0, "x4ce4").unwrap(),
            best_match("/home/lewis/xfce4-terminal", 1.0, "x4ce4").unwrap(),
        ];
        sort_matches(&mut matches);
        assert_eq!(matches[0].path, "/home/lewis/xfce4-terminal");
    }

    #[test]
    fn frecency_can_beat_lower_raw_distance_for_paths() {
        let mut matches = vec![
            best_match("/home/lewis/xfce4-terminal", 1.0, "xfce4-terinal").unwrap(),
            best_match("/home/lewis/xfce4-terminals", 1000.0, "xfce4-terinal").unwrap(),
        ];
        sort_matches(&mut matches);
        assert_eq!(matches[0].path, "/home/lewis/xfce4-terminals");
    }

    #[test]
    fn path_position_beats_frecency_when_core_typo_quality_is_equal() {
        let mut matches = vec![
            best_match("/home/lewis/xfce4-terminal", 1.0, "x4ce4").unwrap(),
            best_match("/home/xfce4/project", 100.0, "x4ce4").unwrap(),
        ];
        sort_matches(&mut matches);
        assert_eq!(matches[0].path, "/home/lewis/xfce4-terminal");
        assert_eq!(matches[0].distance, matches[1].distance);
    }

    #[test]
    fn frecency_breaks_ties_after_path_position_for_direct_token_typos() {
        let mut matches = vec![
            best_match("/home/lewis/one/abdef", 30.5, "abcdef").unwrap(),
            best_match("/home/lewis/two/abdef", 122.4, "abcdef").unwrap(),
        ];
        sort_matches(&mut matches);
        assert_eq!(matches[0].path, "/home/lewis/two/abdef");
        assert_eq!(matches[0].distance, matches[1].distance);
        assert_eq!(matches[0].ratio, matches[1].ratio);
    }

    #[test]
    fn multi_token_component_sequence_can_use_fragmented_token_matching() {
        let candidate =
            best_match("/home/lewis/Dev/applicationlauncher", 93.2, "dev applaunch").unwrap();
        assert_eq!(candidate.operations.core_typo_cost(), 0);
        assert_eq!(candidate.operations.insertions, 10);
    }

    #[test]
    fn compact_whole_token_typos_beat_excessively_fragmented_matches() {
        let cases = [
            (
                "dwv",
                "/home/lewis/Dev",
                878.0,
                "/home/lewis/Dev/markdownviewer",
                64.0,
            ),
            (
                "trre",
                "/home/lewis/Dev/tree",
                239.2,
                "/home/lewis/tasks/tplinkwirelesstransmitter",
                21.0,
            ),
            (
                "nvtio",
                "/home/lewis/repos/nvtop",
                12.0,
                "/home/lewis/.gemini/antigravity-cli/conversations",
                14.5,
            ),
            (
                "dav",
                "/home/lewis/Dev",
                926.0,
                "/home/lewis/Dev/aimodeinteract",
                254.0,
            ),
            (
                "dav",
                "/home/lewis/Dev",
                926.0,
                "/home/lewis/Dev/daterepair",
                0.9,
            ),
            (
                "dav",
                "/home/lewis/Dev",
                926.0,
                "/home/lewis/tasks/agentdvr",
                0.4,
            ),
            (
                "dav",
                "/home/lewis/Dev",
                926.0,
                "/home/lewis/tasks/antigravitylogin",
                1.4,
            ),
            (
                "dav",
                "/home/lewis/Dev",
                926.0,
                "/home/lewis/.gemini/antigravity-cli/conversations",
                1.8,
            ),
            (
                "dav",
                "/home/lewis/Dev",
                926.0,
                "/home/lewis/.gemini/antigravity-cli/log",
                0.3,
            ),
            (
                "dav",
                "/home/lewis/Dev",
                926.0,
                "/home/lewis/.gemini/antigravity-cli/brain/session/scratch",
                0.9,
            ),
        ];
        let mut failures = Vec::new();

        for (query, expected, expected_score, decoy, decoy_score) in cases {
            let mut matches = vec![
                best_match(decoy, decoy_score, query).unwrap(),
                best_match(expected, expected_score, query).unwrap(),
            ];
            sort_matches(&mut matches);
            if matches[0].path != expected {
                failures.push(format!(
                    "{query:?}: expected {expected:?}, ranked {:?}",
                    matches[0].path
                ));
            }
        }

        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    #[test]
    fn exact_whole_phrase_precedes_path_position_and_frecency() {
        let mut matches = vec![
            best_match("/home/app-x-launch", 1000.0, "app launch").unwrap(),
            best_match("/home/app-launch", 1.0, "app launch").unwrap(),
        ];
        sort_matches(&mut matches);
        assert_eq!(matches[0].path, "/home/app-launch");
        assert!(matches[0].exact_whole_phrase());
    }

    #[test]
    fn longer_exact_query_run_precedes_shorter_runs() {
        let mut matches = vec![
            best_match("/home/one-two-x-three-four", 1000.0, "one two three four").unwrap(),
            best_match("/home/one-two-three-x-four", 1.0, "one two three four").unwrap(),
        ];
        sort_matches(&mut matches);
        assert_eq!(matches[0].path, "/home/one-two-three-x-four");
        assert_eq!(matches[0].exact_subphrase_counts_vector(), [1, 2]);
    }

    #[test]
    fn exact_subphrase_run_precedes_broader_scattered_coverage() {
        let mut matches = vec![
            best_match("/home/one-two-three-x", 1.0, "one two three four").unwrap(),
            best_match(
                "/home/four-x-three-x-two-x-one",
                1000.0,
                "one two three four",
            )
            .unwrap(),
        ];

        sort_matches(&mut matches);

        assert_eq!(matches[0].path, "/home/one-two-three-x");
        assert_eq!(matches[0].exact_subphrase_counts_vector(), [1, 2]);
        assert_eq!(matches[1].exact_distinct_coverage(), 4);
    }

    #[test]
    fn test_zoxide_lcunaher() {
        let applications = best_match(
            "/home/lewis/.local/share/applications",
            1.9,
            "application lcunaher",
        );
        let launcher = best_match(
            "/home/lewis/Dev/applicationlauncher",
            190.4,
            "application lcunaher",
        );
        println!("applications match: {:#?}", applications);
        println!("launcher match: {:#?}", launcher);

        let mut matches = vec![applications.unwrap(), launcher.unwrap()];
        sort_matches(&mut matches);

        println!("Sorted:");
        for m in &matches {
            println!(
                "  path: {}, distance: {}, core_typo_cost: {}, score: {}",
                m.path,
                m.distance,
                m.operations.core_typo_cost(),
                m.score
            );
        }

        assert_eq!(matches[0].path, "/home/lewis/Dev/applicationlauncher");
    }
}
