use std::sync::Arc;

use crate::core::{
    AlignmentCostVector, InsertionVector, OperationProfile, TypoCostVector, alignment_cost_milli,
    bounded_damerau_levenshtein, increment_simple_position_vector, max_typos,
    simple_position_vector,
};
use crate::path::exact::exact_match_prepared;
use crate::path::{PathExactRelevance, PreparedPath, Score};
use crate::ranking::{PathTypoSortKey, compare_path_typo_sort_keys, path_typo_keys_are_ambiguous};
use crate::text::{split_path_tokens, to_lowercase};
use crate::token_match::{
    PositionedToken, TokenMatchScratch, aligned_token_distance, best_token_match,
    best_token_match_chars_with_scratch, choose_path_token_match_with_scratch,
    partitioned_token_distance,
};

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

    pub fn exact_whole_phrase(&self) -> bool {
        self.exact_relevance
            .as_ref()
            .is_some_and(PathExactRelevance::whole_phrase)
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
        best_match_with_prepared_path(prepared, score, self, &mut scratch)
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
            &mut scratch.token,
        )
        .map(|candidate| with_exact_relevance(candidate, Arc::clone(&self.keywords), prepared))
    }

    pub fn best_typo_match_prepared_without_exact<'a>(
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
            &mut scratch.token,
        )
    }

    pub fn best_basename_match<'a>(&self, path: &'a str, score: Score) -> Option<PathMatch<'a>> {
        let prepared = PreparedPath::new(path);
        best_basename_match_with_query(&prepared, score, self)
    }
}

#[derive(Default)]
pub struct PathMatchScratch {
    token: TokenMatchScratch,
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
            &mut scratch.token,
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
    candidate.exact_relevance = Some(PathExactRelevance::with_normalized_path(
        candidate.path,
        prepared.normalized_arc(),
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
    scratch: &mut TokenMatchScratch,
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
        candidate_for_compound_component(&ctx, basename, 0),
    );
    update_best(
        &mut best,
        candidate_for_component_sequence(prepared, &ctx, component_count),
    );

    for idx in 0..basename_idx {
        let component = prepared.component(idx);
        let path_position = basename_idx - idx;
        update_best(
            &mut best,
            candidate_for_compound_component(&ctx, component, path_position),
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
    let mut token_scratch = TokenMatchScratch::default();
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
    scratch: &mut TokenMatchScratch,
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
    if best
        .as_ref()
        .is_some_and(|candidate| candidate.distance <= 1)
    {
        return best;
    }
    for token in prepared.tokens(basename_idx) {
        if token != basename {
            update_best(&mut best, candidate_for_single_token(&mut ctx, token, 0));
        }
    }
    if best
        .as_ref()
        .is_some_and(|candidate| candidate.distance <= 1)
    {
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
    scratch: &'q mut TokenMatchScratch,
}

fn candidate_for_single_token<'a>(
    ctx: &mut MatchContext<'a, '_>,
    candidate: &str,
    path_position: usize,
) -> Option<PathMatch<'a>> {
    if candidate.is_empty() {
        return None;
    }

    let candidate_len = candidate.chars().count();
    let matched =
        choose_path_token_match_with_scratch(ctx.query_chars, candidate, ctx.limit, ctx.scratch)?;
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
                ctx.scratch,
            )?;
        distance += cost;
        increment_simple_position_vector(&mut position, position_rank, 1);
        let _ = penalty;
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
    ctx: &MatchContext<'a, '_>,
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
        partitioned_token_distance(&query_token_refs, candidate, ctx.limit)?;
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
    let matched =
        choose_path_token_match_with_scratch(ctx.query_chars, candidate, ctx.limit, ctx.scratch)?;
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
    ctx: &MatchContext<'a, '_>,
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
    ) = aligned_token_distance(&query_token_refs, &candidate_tokens, ctx.limit)?;
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
        .exact_whole_phrase()
        .cmp(&left.exact_whole_phrase())
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
