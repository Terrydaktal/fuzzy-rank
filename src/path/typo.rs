use crate::core::{OperationProfile, bounded_damerau_levenshtein, max_typos};
use crate::path::Score;
use crate::ranking::{
    PathTypoSortKey, compare_path_typo_sort_keys, path_typo_keys_are_ambiguous,
};
use crate::text::{has_query_signal, split_path_tokens, to_lowercase};
use crate::token_match::{
    PositionedToken, aligned_token_distance, best_token_match, fragmented_token_match,
    partitioned_token_distance,
};

#[derive(Clone, Copy, Debug)]
pub struct PathMatch<'a> {
    pub path: &'a str,
    pub distance: usize,
    pub ratio: f64,
    pub path_position: usize,
    pub internal_breaks: usize,
    pub operations: OperationProfile,
    pub score: Score,
    pub path_depth: usize,
}

impl<'a> PathMatch<'a> {
    fn sort_key(&self) -> PathTypoSortKey<'_> {
        PathTypoSortKey {
            distance: self.distance,
            operations: self.operations,
            internal_breaks: self.internal_breaks,
            position: self.path_position,
            score: self.score,
            path_depth: self.path_depth,
            key: self.path,
        }
    }

    pub fn compare(&self, other: &Self) -> std::cmp::Ordering {
        compare_path_typo_sort_keys(&self.sort_key(), &other.sort_key())
    }
}

pub struct TypoQuery {
    text: String,
    len: usize,
    limit: usize,
}

impl TypoQuery {
    pub fn new(query: &str) -> Option<Self> {
        let len = query.chars().count();
        if len <= 1 {
            return None;
        }

        Some(Self {
            text: query.to_owned(),
            len,
            limit: max_typos(len),
        })
    }

    pub fn best_match<'a>(&self, path: &'a str, score: Score) -> Option<PathMatch<'a>> {
        best_match_with_query(path, score, self)
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
    path_typo_keys_are_ambiguous(&first.sort_key(), &second.sort_key())
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
    best_match_inner(path, score, &query.text, query.len, query.limit)
}

fn best_basename_match_with_query<'a>(
    path: &'a str,
    score: Score,
    query: &TypoQuery,
) -> Option<PathMatch<'a>> {
    best_basename_match_inner(
        path,
        score,
        &query.text,
        query.len,
        query.limit,
        query.limit,
    )
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
    let path_depth = components.len();

    if split_path_tokens(query).len() == 1 {
        return best_match_single_token(
            path,
            score,
            query,
            query_len,
            limit,
            &lower_path,
            &components,
            path_depth,
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
        path_depth,
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

    let path_depth = components.len();
    let basename = *components.last().unwrap();
    let ctx = MatchContext {
        path,
        score,
        query,
        query_len,
        limit,
        path_depth,
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
    path_depth: usize,
) -> Option<PathMatch<'a>> {
    let basename = *components.last().unwrap();
    let basename_idx = components.len() - 1;
    let ctx = MatchContext {
        path,
        score,
        query,
        query_len,
        limit,
        path_depth,
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
    path_depth: usize,
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
                    distance: insertions,
                    ratio,
                    path_position: path_position * 3 + position_rank,
                    internal_breaks: 0,
                    operations: OperationProfile {
                        insertions,
                        ..OperationProfile::default()
                    },
                    score: ctx.score,
                    path_depth: ctx.path_depth,
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
        distance,
        ratio,
        path_position: path_position * 3 + position_rank,
        internal_breaks: 0,
        operations,
        score: ctx.score,
        path_depth: ctx.path_depth,
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
    let mut path_metric = 0;
    let mut operations = OperationProfile::default();
    for (query_token, candidate_token) in query_tokens.into_iter().zip(candidate_tokens) {
        let remaining = ctx.limit.checked_sub(distance)?;
        let (cost, penalty, position_rank, _, token_operations) =
            best_token_match(query_token, candidate_token, remaining)?;
        distance += cost;
        path_metric += path_position * 3 + position_rank;
        let _ = penalty;
        operations = operations.merge(token_operations);
    }

    let ratio = distance as f64 / ctx.query_len as f64;
    if ratio > 0.5 {
        return None;
    }

    Some(PathMatch {
        path: ctx.path,
        distance,
        ratio,
        path_position: path_metric,
        internal_breaks: 0,
        operations,
        score: ctx.score,
        path_depth: ctx.path_depth,
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

    let (distance, _structure, position_metric, operations) =
        partitioned_token_distance(&query_tokens, candidate, ctx.limit)?;
    let ratio = distance as f64 / ctx.query_len.max(candidate.chars().count()) as f64;
    if ratio > 0.5 {
        return None;
    }

    Some(PathMatch {
        path: ctx.path,
        distance,
        ratio,
        path_position: path_position * 3 * query_tokens.len() + position_metric,
        internal_breaks: query_tokens.len().saturating_sub(1),
        operations,
        score: ctx.score,
        path_depth: ctx.path_depth,
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
        return None;
    }
    let matched = fragmented_token_match(ctx.query, candidate, ctx.limit)?;
    let ratio = matched.distance as f64 / ctx.query_len.max(candidate_len) as f64;
    if ratio > 0.5 || matched.internal_breaks == 0 {
        return None;
    }

    Some(PathMatch {
        path: ctx.path,
        distance: matched.distance,
        ratio,
        path_position: path_position * 3 + matched.position_rank,
        internal_breaks: matched.internal_breaks,
        operations: matched.operations,
        score: ctx.score,
        path_depth: ctx.path_depth,
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
                    position: path_position * 3,
                })
        })
        .collect();
    if candidate_tokens.len() < query_tokens.len() {
        return None;
    }

    let (distance, path_position, internal_breaks, _structure, operations) =
        aligned_token_distance(&query_tokens, &candidate_tokens, ctx.limit)?;
    let ratio = distance as f64 / ctx.query_len as f64;
    if ratio > 0.5 {
        return None;
    }

    Some(PathMatch {
        path: ctx.path,
        distance,
        ratio,
        path_position,
        internal_breaks,
        operations,
        score: ctx.score,
        path_depth: ctx.path_depth,
    })
}

fn path_components(path: &str) -> Vec<&str> {
    path.split(['/', '\\'])
        .filter(|component| !component.is_empty())
        .collect()
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
        assert_eq!(candidate.path_position, 1);
        assert_eq!(candidate.internal_breaks, 0);
    }

    #[test]
    fn basename_token_prefixes_do_not_count_as_typos() {
        let candidate = best_match("/home/lewis/xfce4-terminal", 10.0, "xzfce-ter").unwrap();
        assert_eq!(candidate.distance, 1);
    }

    #[test]
    fn spaced_query_tokens_can_match_single_compound_token() {
        let candidate = best_match("/home/lewis/applicationlauncher", 10.0, "app laucnh").unwrap();
        assert_eq!(candidate.path_position, 0);
        assert_eq!(candidate.distance, 1);
        assert_eq!(candidate.internal_breaks, 1);
    }

    #[test]
    fn missing_space_query_matches_single_compound_token_via_internal_breaks_and_insertions() {
        let candidate = best_match("/home/lewis/applicationlauncher", 10.0, "applaunch").unwrap();
        assert_eq!(candidate.operations.core_typo_cost(), 0);
        assert_eq!(candidate.internal_breaks, 2);
        assert_eq!(candidate.operations.insertions, 10);
    }

    #[test]
    fn spaced_query_tokens_can_match_component_sequence() {
        let candidate = best_match("/home/lewis/tasks/config", 10.0, "tasks cinfig").unwrap();
        assert_eq!(candidate.path_position, 3);
        assert_eq!(candidate.distance, 1);
        assert_eq!(candidate.internal_breaks, 0);
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
                    substitution_penalty: 47,
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
        assert_eq!(candidate.path_position, 3);
        assert_eq!(candidate.distance, 2);
        assert_eq!(candidate.internal_breaks, 1);
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
        assert_eq!(candidate.path_position, 12);
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
        assert_eq!(matches[0].operations.core_typo_cost(), 0);
        assert!(matches[1].operations.core_typo_cost() > matches[0].operations.core_typo_cost());
    }

    #[test]
    fn long_queries_use_half_length_typo_limit() {
        let candidate = best_match("/home/lewis/xfce4-terminal", 10.0, "xgce4-tremriianl").unwrap();
        assert_eq!(candidate.distance, 5);
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
    fn single_character_queries_are_not_corrected() {
        assert!(best_match("/home/lewis/foobar", 10.0, "f").is_none());
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
        assert_eq!(candidate.internal_breaks, 2);
        assert_eq!(candidate.operations.insertions, 10);
    }
}
