use std::cell::OnceCell;
use std::sync::Arc;

use crate::core::{
    OperationProfile, bounded_damerau_levenshtein, exact_distinct_coverage, exact_phrase_count,
    exact_phrase_run_vector, max_typos,
};
use crate::path::exact::{exact_match, match_penalty};
use crate::path::Score;
use crate::ranking::{
    PathTypoSortKey, compare_path_typo_sort_keys, path_typo_keys_are_ambiguous,
};
use crate::text::{has_query_signal, split_path_tokens, to_lowercase};
use crate::token_match::{
    PositionedToken, aligned_token_distance, best_token_match, fragmented_token_match,
    partitioned_token_distance,
};

#[derive(Clone, Debug)]
struct PathExactRelevance<'a> {
    path: &'a str,
    query_tokens: Arc<[String]>,
    whole_phrase: OnceCell<bool>,
    run_vector: OnceCell<Vec<usize>>,
    distinct_coverage: OnceCell<usize>,
}

#[derive(Clone, Debug)]
pub struct PathMatch<'a> {
    pub path: &'a str,
    pub exact: bool,
    pub distance: usize,
    pub ratio: f64,
    pub segment: usize,
    pub position: usize,
    pub operations: OperationProfile,
    pub score: Score,
    exact_relevance: Option<PathExactRelevance<'a>>,
}

impl<'a> PathMatch<'a> {
    fn sort_key(&self) -> PathTypoSortKey<'_> {
        PathTypoSortKey {
            exact: self.exact,
            distance: self.distance,
            operations: self.operations,
            segment: self.segment,
            position: self.position,
            score: self.score,
            key: self.path,
        }
    }

    pub fn compare(&self, other: &Self) -> std::cmp::Ordering {
        other
            .exact
            .cmp(&self.exact)
            .then_with(|| {
                other
                    .exact_whole_phrase()
                    .cmp(&self.exact_whole_phrase())
            })
            .then_with(|| other.exact_run_vector().cmp(self.exact_run_vector()))
            .then_with(|| {
                other
                    .exact_distinct_coverage()
                    .cmp(&self.exact_distinct_coverage())
            })
            .then_with(|| compare_path_typo_sort_keys(&self.sort_key(), &other.sort_key()))
    }

    pub fn exact_whole_phrase(&self) -> bool {
        self.exact_relevance
            .as_ref()
            .is_some_and(PathExactRelevance::whole_phrase)
    }

    pub fn exact_run_vector(&self) -> &[usize] {
        self.exact_relevance
            .as_ref()
            .map_or(&[], PathExactRelevance::run_vector)
    }

    pub fn exact_distinct_coverage(&self) -> usize {
        self.exact_relevance
            .as_ref()
            .map_or(0, PathExactRelevance::distinct_coverage)
    }
}

pub struct TypoQuery {
    text: String,
    keywords: Arc<[String]>,
    len: usize,
    limit: usize,
    allow_typos: bool,
}

impl TypoQuery {
    pub fn new(query: &str) -> Option<Self> {
        let len = query.chars().count();
        let keywords = split_path_tokens(query)
            .into_iter()
            .map(to_lowercase)
            .collect::<Vec<_>>();
        if keywords.is_empty() {
            return None;
        }

        Some(Self {
            text: to_lowercase(query),
            keywords: keywords.into(),
            len,
            limit: max_typos(len),
            allow_typos: true,
        })
    }

    pub fn with_typo_fallback(mut self, allow_typos: bool) -> Self {
        self.allow_typos = allow_typos;
        self
    }

    pub fn best_match<'a>(&self, path: &'a str, score: Score) -> Option<PathMatch<'a>> {
        best_match_with_query(path, score, self)
    }

    /// Matches only the exact path class without evaluating typo distance.
    pub fn best_exact_match<'a>(&self, path: &'a str, score: Score) -> Option<PathMatch<'a>> {
        exact_path_match(path, score, self)
            .map(|candidate| with_exact_relevance(candidate, Arc::clone(&self.keywords)))
    }

    /// Matches only the typo path class without repeating the exact check.
    pub fn best_typo_match<'a>(&self, path: &'a str, score: Score) -> Option<PathMatch<'a>> {
        if !self.allow_typos || self.len <= 1 {
            return None;
        }
        best_match_inner(path, score, &self.text, self.len, self.limit)
            .map(|candidate| with_exact_relevance(candidate, Arc::clone(&self.keywords)))
    }

    pub fn best_basename_match<'a>(&self, path: &'a str, score: Score) -> Option<PathMatch<'a>> {
        best_basename_match_with_query(path, score, self)
    }
}

pub fn best_match<'a>(path: &'a str, score: Score, query: &str) -> Option<PathMatch<'a>> {
    let query = TypoQuery::new(query)?;
    best_match_with_query(path, score, &query)
}

pub fn sort_matches(matches: &mut [PathMatch<'_>]) {
    matches.sort_unstable_by(PathMatch::compare);
}

pub fn is_ambiguous(first: &PathMatch<'_>, second: &PathMatch<'_>) -> bool {
    first.exact == second.exact
        && first.exact_whole_phrase() == second.exact_whole_phrase()
        && first.exact_run_vector() == second.exact_run_vector()
        && first.exact_distinct_coverage() == second.exact_distinct_coverage()
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

fn best_match_with_query<'a>(
    path: &'a str,
    score: Score,
    query: &TypoQuery,
) -> Option<PathMatch<'a>> {
    let candidate = match exact_path_match(path, score, query) {
        Some(candidate) => candidate,
        None if query.allow_typos && query.len > 1 => {
            best_match_inner(path, score, &query.text, query.len, query.limit)?
        }
        None => return None,
    };
    Some(with_exact_relevance(candidate, Arc::clone(&query.keywords)))
}

fn best_basename_match_with_query<'a>(
    path: &'a str,
    score: Score,
    query: &TypoQuery,
) -> Option<PathMatch<'a>> {
    let candidate = match exact_path_match(path, score, query) {
        Some(candidate) => candidate,
        None if query.allow_typos && query.len > 1 => best_basename_match_inner(
            path,
            score,
            &query.text,
            query.len,
            query.limit,
            query.limit,
        )?,
        None => return None,
    };
    Some(with_exact_relevance(candidate, Arc::clone(&query.keywords)))
}

fn with_exact_relevance<'a>(
    mut candidate: PathMatch<'a>,
    query_tokens: Arc<[String]>,
) -> PathMatch<'a> {
    candidate.exact_relevance = Some(PathExactRelevance::new(candidate.path, query_tokens));
    candidate
}

fn exact_path_match<'a>(path: &'a str, score: Score, query: &TypoQuery) -> Option<PathMatch<'a>> {
    let matched = exact_match(path, &query.keywords, score)?;
    let insertions = match_penalty(path, &query.keywords)?;
    Some(PathMatch {
        path,
        exact: true,
        distance: 0,
        ratio: 0.0,
        segment: matched.segment,
        position: matched.position,
        operations: OperationProfile {
            insertions,
            ..OperationProfile::default()
        },
        score,
        exact_relevance: None,
    })
}

fn best_match_inner<'a>(
    path: &'a str,
    score: Score,
    query: &str,
    query_len: usize,
    limit: usize,
) -> Option<PathMatch<'a>> {
    let lower_path = to_lowercase(path);
    if !has_query_signal(split_path_tokens(query), &lower_path) {
        return None;
    }

    let components = path_components(&lower_path);
    if components.is_empty() {
        return None;
    }
    if split_path_tokens(query).len() == 1 {
        return best_match_single_token(
            path,
            score,
            query,
            query_len,
            limit,
            &lower_path,
            &components,
        );
    }

    let basename = *components.last().unwrap();
    let basename_idx = components.len() - 1;
    let ctx = MatchContext {
        path,
        score,
        query,
        query_len,
        limit,
    };

    let mut best = candidate_for_text(&ctx, basename, 0);
    update_best(&mut best, candidate_for_token_sequence(&ctx, basename, 0));
    update_best(
        &mut best,
        candidate_for_compound_component(&ctx, basename, 0),
    );
    update_best(
        &mut best,
        candidate_for_component_sequence(&ctx, &components),
    );

    for token in split_path_tokens(basename) {
        update_best(&mut best, candidate_for_text(&ctx, token, 0));
    }

    for (idx, component) in components[..components.len().saturating_sub(1)]
        .iter()
        .enumerate()
    {
        let path_position = basename_idx - idx;
        update_best(
            &mut best,
            candidate_for_text(&ctx, component, path_position),
        );
        update_best(
            &mut best,
            candidate_for_compound_component(&ctx, component, path_position),
        );
        for token in split_path_tokens(component) {
            update_best(&mut best, candidate_for_text(&ctx, token, path_position));
        }
    }

    update_best(
        &mut best,
        candidate_for_text(&ctx, &lower_path, basename_idx + 1),
    );
    best
}

fn best_basename_match_inner<'a>(
    path: &'a str,
    score: Score,
    query: &str,
    query_len: usize,
    limit: usize,
    full_limit: usize,
) -> Option<PathMatch<'a>> {
    let lower_path = to_lowercase(path);
    if !has_query_signal(split_path_tokens(query), &lower_path) {
        return None;
    }

    let components = path_components(&lower_path);
    if components.is_empty() {
        return None;
    }

    let basename = *components.last().unwrap();
    let ctx = MatchContext {
        path,
        score,
        query,
        query_len,
        limit,
    };
    let mut best = candidate_for_text(&ctx, basename, 0);
    update_best(
        &mut best,
        candidate_for_partitioned_single_query(&ctx, basename, 0),
    );
    for token in split_path_tokens(basename) {
        update_best(&mut best, candidate_for_text(&ctx, token, 0));
        update_best(
            &mut best,
            candidate_for_partitioned_single_query(&ctx, token, 0),
        );
    }
    best.filter(|candidate| candidate.distance <= full_limit)
}

#[allow(clippy::too_many_arguments)]
fn best_match_single_token<'a>(
    path: &'a str,
    score: Score,
    query: &str,
    query_len: usize,
    limit: usize,
    lower_path: &str,
    components: &[&str],
) -> Option<PathMatch<'a>> {
    let basename = *components.last().unwrap();
    let basename_idx = components.len() - 1;
    let ctx = MatchContext {
        path,
        score,
        query,
        query_len,
        limit,
    };

    let mut best = candidate_for_text(&ctx, basename, 0);
    update_best(
        &mut best,
        candidate_for_partitioned_single_query(&ctx, basename, 0),
    );
    if best
        .as_ref()
        .is_some_and(|candidate| candidate.distance <= 1)
    {
        return best;
    }
    for token in split_path_tokens(basename) {
        update_best(&mut best, candidate_for_text(&ctx, token, 0));
        update_best(
            &mut best,
            candidate_for_partitioned_single_query(&ctx, token, 0),
        );
    }
    if best
        .as_ref()
        .is_some_and(|candidate| candidate.distance <= 1)
    {
        return best;
    }

    for (idx, component) in components[..components.len().saturating_sub(1)]
        .iter()
        .enumerate()
    {
        let path_position = basename_idx - idx;
        update_best(
            &mut best,
            candidate_for_text(&ctx, component, path_position),
        );
        update_best(
            &mut best,
            candidate_for_partitioned_single_query(&ctx, component, path_position),
        );
        for token in split_path_tokens(component) {
            update_best(&mut best, candidate_for_text(&ctx, token, path_position));
            update_best(
                &mut best,
                candidate_for_partitioned_single_query(&ctx, token, path_position),
            );
        }
    }

    update_best(
        &mut best,
        candidate_for_text(&ctx, lower_path, basename_idx + 1),
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
    let query_token_count = split_path_tokens(ctx.query).len();

    if query_token_count == 1 {
        if let Some(start) = candidate.find(ctx.query) {
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
                    ratio,
                    segment: path_position,
                    position: position_rank,
                    operations: OperationProfile {
                        insertions,
                        ..OperationProfile::default()
                    },
                    score: ctx.score,
                    exact_relevance: None,
                });
            }
        }
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
        ratio,
        segment: path_position,
        position: position_rank,
        operations,
        score: ctx.score,
        exact_relevance: None,
    })
}

fn candidate_for_token_sequence<'a>(
    ctx: &MatchContext<'a, '_>,
    candidate: &str,
    path_position: usize,
) -> Option<PathMatch<'a>> {
    let query_tokens = split_path_tokens(ctx.query);
    let candidate_tokens = split_path_tokens(candidate);
    if query_tokens.len() < 2 || query_tokens.len() != candidate_tokens.len() {
        return None;
    }

    let mut distance = 0;
    let mut position = 0;
    let mut operations = OperationProfile::default();
    for (query_token, candidate_token) in query_tokens.into_iter().zip(candidate_tokens) {
        let remaining = ctx.limit.checked_sub(distance)?;
        let (cost, penalty, position_rank, _, token_operations) =
            best_token_match(query_token, candidate_token, remaining)?;
        distance += cost;
        position += position_rank;
        let _ = penalty;
        operations = operations.merge(token_operations);
    }

    let ratio = distance as f64 / ctx.query_len as f64;
    if ratio > 0.5 {
        return None;
    }

    Some(PathMatch {
        path: ctx.path,
        exact: false,
        distance,
        ratio,
        segment: path_position,
        position,
        operations,
        score: ctx.score,
        exact_relevance: None,
    })
}

fn candidate_for_compound_component<'a>(
    ctx: &MatchContext<'a, '_>,
    candidate: &str,
    path_position: usize,
) -> Option<PathMatch<'a>> {
    let query_tokens = split_path_tokens(ctx.query);
    let candidate_tokens = split_path_tokens(candidate);
    if query_tokens.len() < 2 || candidate_tokens.len() != 1 {
        return None;
    }

    let (distance, _structure, position, operations) =
        partitioned_token_distance(&query_tokens, candidate, ctx.limit)?;
    let ratio = distance as f64 / ctx.query_len.max(candidate.chars().count()) as f64;
    if ratio > 0.5 {
        return None;
    }

    Some(PathMatch {
        path: ctx.path,
        exact: false,
        distance,
        ratio,
        segment: path_position,
        position,
        operations,
        score: ctx.score,
        exact_relevance: None,
    })
}

fn candidate_for_partitioned_single_query<'a>(
    ctx: &MatchContext<'a, '_>,
    candidate: &str,
    path_position: usize,
) -> Option<PathMatch<'a>> {
    if split_path_tokens(candidate).len() != 1 {
        return None;
    }
    let candidate_len = candidate.chars().count();
    if ctx.query_len.abs_diff(candidate_len) <= ctx.limit {
        let matched = fragmented_token_match(ctx.query, candidate, ctx.limit)?;
        let ratio = matched.distance as f64 / ctx.query_len.max(candidate_len) as f64;
        if ratio > 0.5 || matched.internal_breaks < 2 {
            return None;
        }
        return Some(PathMatch {
            path: ctx.path,
            exact: false,
            distance: matched.distance,
            ratio,
            segment: path_position,
            position: matched.position_rank,
            operations: matched.operations,
            score: ctx.score,
            exact_relevance: None,
        });
    }
    let matched = fragmented_token_match(ctx.query, candidate, ctx.limit)?;
    let ratio = matched.distance as f64 / ctx.query_len.max(candidate_len) as f64;
    if ratio > 0.5 || matched.internal_breaks == 0 {
        return None;
    }

    Some(PathMatch {
        path: ctx.path,
        exact: false,
        distance: matched.distance,
        ratio,
        segment: path_position,
        position: matched.position_rank,
        operations: matched.operations,
        score: ctx.score,
        exact_relevance: None,
    })
}

fn candidate_for_component_sequence<'a>(
    ctx: &MatchContext<'a, '_>,
    components: &[&str],
) -> Option<PathMatch<'a>> {
    let query_tokens = split_path_tokens(ctx.query);
    if query_tokens.len() < 2 {
        return None;
    }

    let basename_idx = components.len() - 1;
    let candidate_tokens: Vec<_> = components
        .iter()
        .enumerate()
        .flat_map(|(idx, component)| {
            let path_position = basename_idx - idx;
            split_path_tokens(component)
                .into_iter()
                .map(move |token| PositionedToken {
                    token,
                    segment: path_position,
                })
        })
        .collect();
    if candidate_tokens.len() < query_tokens.len() {
        return None;
    }

    let (distance, segment, position, _breaks, _structure, operations) =
        aligned_token_distance(&query_tokens, &candidate_tokens, ctx.limit)?;
    let ratio = distance as f64 / ctx.query_len as f64;
    if ratio > 0.5 {
        return None;
    }

    Some(PathMatch {
        path: ctx.path,
        exact: false,
        distance,
        ratio,
        segment,
        position,
        operations,
        score: ctx.score,
        exact_relevance: None,
    })
}

fn path_components(path: &str) -> Vec<&str> {
    path.split(['/', '\\'])
        .filter(|component| !component.is_empty())
        .collect()
}

impl<'a> PathExactRelevance<'a> {
    fn new(path: &'a str, query_tokens: Arc<[String]>) -> Self {
        Self {
            path,
            query_tokens,
            whole_phrase: OnceCell::new(),
            run_vector: OnceCell::new(),
            distinct_coverage: OnceCell::new(),
        }
    }

    fn whole_phrase(&self) -> bool {
        *self.whole_phrase.get_or_init(|| {
            let lowered = to_lowercase(self.path);
            let components = path_components(&lowered)
                .into_iter()
                .map(split_path_tokens)
                .collect::<Vec<_>>();
            exact_phrase_count(
                &self.query_tokens,
                components.iter().map(Vec::as_slice),
            ) > 0
        })
    }

    fn run_vector(&self) -> &[usize] {
        self.run_vector.get_or_init(|| {
            let lowered = to_lowercase(self.path);
            let component_tokens = path_components(&lowered)
                .into_iter()
                .map(split_path_tokens)
                .collect::<Vec<_>>();
            exact_phrase_run_vector(
                &self.query_tokens,
                component_tokens.iter().map(Vec::as_slice),
            )
        })
    }

    fn distinct_coverage(&self) -> usize {
        *self.distinct_coverage.get_or_init(|| {
            let lowered = to_lowercase(self.path);
            let components = path_components(&lowered)
                .into_iter()
                .map(split_path_tokens)
                .collect::<Vec<_>>();
            exact_distinct_coverage(
                &self.query_tokens,
                components.iter().map(Vec::as_slice),
            )
        })
    }
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
        assert_eq!(candidate.segment, 0);
        assert_eq!(candidate.position, 1);
    }

    #[test]
    fn basename_token_prefixes_do_not_count_as_typos() {
        let candidate = best_match("/home/lewis/xfce4-terminal", 10.0, "xzfce-ter").unwrap();
        assert_eq!(candidate.distance, 1);
    }

    #[test]
    fn spaced_query_tokens_can_match_single_compound_token() {
        let candidate = best_match("/home/lewis/applicationlauncher", 10.0, "app laucnh").unwrap();
        assert_eq!(candidate.segment, 0);
        assert_eq!(candidate.position, 0);
        assert_eq!(candidate.distance, 1);
    }

    #[test]
    fn missing_space_query_matches_single_compound_token_via_internal_breaks_and_insertions() {
        let candidate = best_match("/home/lewis/applicationlauncher", 10.0, "applaunch").unwrap();
        assert_eq!(candidate.operations.core_typo_cost(), 0);
        assert_eq!(candidate.operations.insertions, 0);
    }

    #[test]
    fn spaced_query_tokens_can_match_component_sequence() {
        let candidate = best_match("/home/lewis/tasks/config", 10.0, "tasks cinfig").unwrap();
        assert_eq!(candidate.segment, 0);
        assert_eq!(candidate.position, 0);
        assert_eq!(candidate.distance, 1);
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
        assert_eq!(candidate.segment, 0);
        assert_eq!(candidate.position, 0);
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
        assert_eq!(candidate.segment, 2);
        assert_eq!(candidate.position, 0);
    }

    #[test]
    fn extra_typed_character_beats_substitution_for_paths() {
        let mut matches = [
            best_match("/home/lewis/abdef", 10.0, "abcdef").unwrap(),
            best_match("/home/lewis/abqdef", 10.0, "abcdef").unwrap(),
        ];
        sort_matches(&mut matches);
        assert_eq!(matches[0].path, "/home/lewis/abdef");
        assert_eq!(matches[0].distance, matches[1].distance);
        assert_eq!(matches[0].operations.insertions, 1);
        assert_eq!(matches[0].operations.core_typo_cost(), 6);
        assert!(matches[1].operations.core_typo_cost() > matches[0].operations.core_typo_cost());
    }

    #[test]
    fn long_queries_use_half_length_typo_limit() {
        let candidate = best_match("/home/lewis/xfce4-terminal", 10.0, "xgce4-tremriianl").unwrap();
        assert_eq!(candidate.distance, 6);
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
        let candidate = best_match(
            "/home/lewis/Dev/applicationlauncher",
            93.2,
            "dev applaunch",
        )
        .unwrap();
        assert_eq!(candidate.operations.core_typo_cost(), 0);
        assert_eq!(candidate.operations.insertions, 0);
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
        assert_eq!(matches[0].exact_run_vector(), [0, 1, 2]);
    }

    #[test]
    fn test_zoxide_lcunaher() {
        let applications = best_match("/home/lewis/.local/share/applications", 1.9, "application lcunaher");
        let launcher = best_match("/home/lewis/Dev/applicationlauncher", 190.4, "application lcunaher");
        println!("applications match: {:#?}", applications);
        println!("launcher match: {:#?}", launcher);
        
        let mut matches = vec![
            applications.unwrap(),
            launcher.unwrap(),
        ];
        sort_matches(&mut matches);
        
        println!("Sorted:");
        for m in &matches {
            println!("  path: {}, distance: {}, core_typo_cost: {}, score: {}", m.path, m.distance, m.operations.core_typo_cost(), m.score);
        }
        
        assert_eq!(matches[0].path, "/home/lewis/Dev/applicationlauncher");
    }
}
