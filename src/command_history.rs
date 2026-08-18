use std::cmp::{Ordering, Reverse};

use crate::core::{
    InsertionVector, OperationProfile, TypoCostVector, bounded_damerau_levenshtein,
    compare_key_asc, compare_score_desc, exact_distinct_coverage, exact_phrase_present,
    exact_subphrase_counts_vector, max_typos, ordered_token_span,
};
use crate::text::{has_query_signal, split_search_tokens, split_whitespace_tokens, to_lowercase};

pub type Recency = f64;

#[derive(Clone, Copy, Debug)]
pub struct CommandHistoryCandidate<'a> {
    pub command: &'a str,
    pub tokens: &'a [&'a str],
    pub recency: Recency,
}

#[derive(Clone, Debug)]
pub struct CommandHistoryMatch<'a> {
    pub candidate: CommandHistoryCandidate<'a>,
    pub exact_phrase_present: bool,
    pub exact_coverage: usize,
    pub exact_subphrase_counts_vector: Vec<usize>,
    pub total_coverage: usize,
    pub fuzzy_subphrase_counts_vector: Vec<usize>,
    pub order_distance: usize,
    pub total_gap: usize,
    pub exact_count: usize,
    pub span_len: usize,
    pub distance: usize,
    pub ratio: f64,
    pub operations: OperationProfile,
    pub typo_costs: TypoCostVector,
    pub insertions: InsertionVector,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommandHistoryRank {
    pub exact_phrase_present: bool,
    pub exact_coverage: usize,
    pub exact_subphrase_counts_vector: Vec<usize>,
    pub total_coverage: usize,
    pub fuzzy_subphrase_counts_vector: Vec<usize>,
    pub typo_costs: TypoCostVector,
    pub insertions: InsertionVector,
    pub typo_operations: OperationProfile,
    pub order_distance: usize,
    pub total_gap: usize,
}

impl<'a> CommandHistoryMatch<'a> {
    pub fn rank(&self) -> CommandHistoryRank {
        CommandHistoryRank {
            exact_phrase_present: self.exact_phrase_present,
            exact_coverage: self.exact_coverage,
            exact_subphrase_counts_vector: self.exact_subphrase_counts_vector.clone(),
            total_coverage: self.total_coverage,
            fuzzy_subphrase_counts_vector: self.fuzzy_subphrase_counts_vector.clone(),
            typo_costs: self.typo_costs.clone(),
            insertions: self.insertions.clone(),
            typo_operations: self.operations,
            order_distance: self.order_distance,
            total_gap: self.total_gap,
        }
    }

    pub fn compare(&self, other: &Self) -> std::cmp::Ordering {
        Reverse(self.exact_phrase_present)
            .cmp(&Reverse(other.exact_phrase_present))
            .then_with(|| Reverse(self.exact_coverage).cmp(&Reverse(other.exact_coverage)))
            .then_with(|| {
                Reverse(&self.exact_subphrase_counts_vector)
                    .cmp(&Reverse(&other.exact_subphrase_counts_vector))
            })
            .then_with(|| Reverse(self.total_coverage).cmp(&Reverse(other.total_coverage)))
            .then_with(|| {
                Reverse(&self.fuzzy_subphrase_counts_vector)
                    .cmp(&Reverse(&other.fuzzy_subphrase_counts_vector))
            })
            .then_with(|| self.typo_costs.cmp(&other.typo_costs))
            .then_with(|| self.insertions.cmp(&other.insertions))
            .then_with(|| self.operations.cmp(&other.operations))
            .then_with(|| self.order_distance.cmp(&other.order_distance))
            .then_with(|| self.total_gap.cmp(&other.total_gap))
            .then_with(|| compare_score_desc(self.candidate.recency, other.candidate.recency))
            .then_with(|| compare_key_asc(self.candidate.command, other.candidate.command))
    }
}

pub fn compare_ranks(
    left: &CommandHistoryRank,
    left_recency: Recency,
    left_command: &str,
    right: &CommandHistoryRank,
    right_recency: Recency,
    right_command: &str,
) -> Ordering {
    Reverse(left.exact_phrase_present)
        .cmp(&Reverse(right.exact_phrase_present))
        .then_with(|| Reverse(left.exact_coverage).cmp(&Reverse(right.exact_coverage)))
        .then_with(|| {
            Reverse(&left.exact_subphrase_counts_vector)
                .cmp(&Reverse(&right.exact_subphrase_counts_vector))
        })
        .then_with(|| Reverse(left.total_coverage).cmp(&Reverse(right.total_coverage)))
        .then_with(|| {
            Reverse(&left.fuzzy_subphrase_counts_vector)
                .cmp(&Reverse(&right.fuzzy_subphrase_counts_vector))
        })
        .then_with(|| left.typo_costs.cmp(&right.typo_costs))
        .then_with(|| left.insertions.cmp(&right.insertions))
        .then_with(|| left.typo_operations.cmp(&right.typo_operations))
        .then_with(|| left.order_distance.cmp(&right.order_distance))
        .then_with(|| left.total_gap.cmp(&right.total_gap))
        .then_with(|| compare_score_desc(left_recency, right_recency))
        .then_with(|| compare_key_asc(left_command, right_command))
}

pub struct CommandHistoryQuery {
    tokens: Vec<String>,
    len: usize,
    limit: usize,
}

impl CommandHistoryQuery {
    pub fn new(query: &str) -> Option<Self> {
        let query = to_lowercase(query.trim());
        let len = query.chars().count();
        if len <= 1 {
            return None;
        }

        let tokens = split_whitespace_tokens(&query)
            .into_iter()
            .map(str::to_owned)
            .collect();

        Some(Self {
            tokens,
            len,
            limit: max_typos(len),
        })
    }

    pub fn best_match<'a>(
        &self,
        candidate: CommandHistoryCandidate<'a>,
    ) -> Option<CommandHistoryMatch<'a>> {
        let lower_command = to_lowercase(candidate.command);
        let lower_tokens = candidate
            .tokens
            .iter()
            .map(to_lowercase)
            .collect::<Vec<_>>();
        let lowered = LoweredCandidate {
            original: candidate,
            command: lower_command,
            tokens: lower_tokens,
        };
        best_match_inner(&lowered, &self.tokens, self.len, self.limit)
    }

    pub fn best_rank_prepared(
        &self,
        candidate: &PreparedCommandHistoryCandidate,
    ) -> Option<CommandHistoryRank> {
        self.best_match_prepared(candidate)
            .map(|matched| matched.rank)
    }

    pub fn best_match_prepared(
        &self,
        candidate: &PreparedCommandHistoryCandidate,
    ) -> Option<PreparedCommandHistoryMatch> {
        compute_match_fields(
            &candidate.normalized_command,
            &candidate.tokens,
            &self.tokens,
            self.len,
            self.limit,
        )
        .map(|computed| {
            let distance = computed.distance;
            PreparedCommandHistoryMatch {
                rank: computed.into_rank(),
                distance,
            }
        })
    }
}

pub fn best_match<'a>(
    candidate: CommandHistoryCandidate<'a>,
    query: &str,
) -> Option<CommandHistoryMatch<'a>> {
    CommandHistoryQuery::new(query)?.best_match(candidate)
}

pub fn parse_command_history_candidate(
    command: &str,
    recency: Recency,
) -> ParsedCommandHistoryCandidate<'_> {
    ParsedCommandHistoryCandidate {
        command,
        tokens: split_search_tokens(command),
        recency,
    }
}

pub struct ParsedCommandHistoryCandidate<'a> {
    command: &'a str,
    tokens: Vec<&'a str>,
    recency: Recency,
}

/// Owned command-history preparation for repeated queries over the same
/// history snapshot. The original command is retained for display, while the
/// normalized command and tokens are computed only once.
#[derive(Clone, Debug)]
pub struct PreparedCommandHistoryCandidate {
    command: String,
    normalized_command: String,
    tokens: Vec<String>,
    recency: Recency,
}

#[derive(Clone, Debug)]
pub struct PreparedCommandHistoryMatch {
    pub rank: CommandHistoryRank,
    pub distance: usize,
}

impl PreparedCommandHistoryCandidate {
    pub fn new(command: &str, recency: Recency) -> Self {
        let command = command.to_string();
        let normalized_command = to_lowercase(&command);
        let tokens = split_search_tokens(&normalized_command)
            .into_iter()
            .map(str::to_owned)
            .collect();
        Self {
            command,
            normalized_command,
            tokens,
            recency,
        }
    }

    pub fn command(&self) -> &str {
        &self.command
    }

    pub fn recency(&self) -> Recency {
        self.recency
    }
}

impl<'a> ParsedCommandHistoryCandidate<'a> {
    pub fn as_candidate(&self) -> CommandHistoryCandidate<'_> {
        CommandHistoryCandidate {
            command: self.command,
            tokens: &self.tokens,
            recency: self.recency,
        }
    }
}

pub fn sort_matches(matches: &mut [CommandHistoryMatch<'_>]) {
    matches.sort_unstable_by(CommandHistoryMatch::compare);
}

pub fn is_ambiguous(first: &CommandHistoryMatch<'_>, second: &CommandHistoryMatch<'_>) -> bool {
    first.exact_phrase_present == second.exact_phrase_present
        && first.exact_coverage == second.exact_coverage
        && first.exact_subphrase_counts_vector == second.exact_subphrase_counts_vector
        && first.total_coverage == second.total_coverage
        && first.fuzzy_subphrase_counts_vector == second.fuzzy_subphrase_counts_vector
        && first.typo_costs == second.typo_costs
        && first.insertions == second.insertions
        && first.operations == second.operations
        && first.order_distance == second.order_distance
        && first.total_gap == second.total_gap
}

struct LoweredCandidate<'a> {
    original: CommandHistoryCandidate<'a>,
    command: String,
    tokens: Vec<String>,
}

struct ComputedCommandHistory {
    exact_phrase_present: bool,
    exact_coverage: usize,
    exact_subphrase_counts_vector: Vec<usize>,
    total_coverage: usize,
    fuzzy_subphrase_counts_vector: Vec<usize>,
    order_distance: usize,
    total_gap: usize,
    exact_count: usize,
    span_len: usize,
    distance: usize,
    ratio: f64,
    operations: OperationProfile,
    typo_costs: TypoCostVector,
    insertions: InsertionVector,
}

impl ComputedCommandHistory {
    fn into_match<'a>(self, candidate: CommandHistoryCandidate<'a>) -> CommandHistoryMatch<'a> {
        CommandHistoryMatch {
            candidate,
            exact_phrase_present: self.exact_phrase_present,
            exact_coverage: self.exact_coverage,
            exact_subphrase_counts_vector: self.exact_subphrase_counts_vector,
            total_coverage: self.total_coverage,
            fuzzy_subphrase_counts_vector: self.fuzzy_subphrase_counts_vector,
            order_distance: self.order_distance,
            total_gap: self.total_gap,
            exact_count: self.exact_count,
            span_len: self.span_len,
            distance: self.distance,
            ratio: self.ratio,
            operations: self.operations,
            typo_costs: self.typo_costs,
            insertions: self.insertions,
        }
    }

    fn into_rank(self) -> CommandHistoryRank {
        CommandHistoryRank {
            exact_phrase_present: self.exact_phrase_present,
            exact_coverage: self.exact_coverage,
            exact_subphrase_counts_vector: self.exact_subphrase_counts_vector,
            total_coverage: self.total_coverage,
            fuzzy_subphrase_counts_vector: self.fuzzy_subphrase_counts_vector,
            typo_costs: self.typo_costs,
            insertions: self.insertions,
            typo_operations: self.operations,
            order_distance: self.order_distance,
            total_gap: self.total_gap,
        }
    }
}

fn best_match_inner<'a>(
    candidate: &LoweredCandidate<'a>,
    query_tokens: &[String],
    query_len: usize,
    limit: usize,
) -> Option<CommandHistoryMatch<'a>> {
    compute_match_fields(
        &candidate.command,
        &candidate.tokens,
        query_tokens,
        query_len,
        limit,
    )
    .map(|computed| computed.into_match(candidate.original))
}

fn compute_match_fields(
    command: &str,
    candidate_tokens: &[String],
    query_tokens: &[String],
    query_len: usize,
    limit: usize,
) -> Option<ComputedCommandHistory> {
    if !has_query_signal(query_tokens.iter().map(String::as_str), command) {
        return None;
    }

    if query_tokens.is_empty() || candidate_tokens.is_empty() {
        return None;
    }

    let fields = std::iter::once(candidate_tokens);
    let exact_phrase_present = exact_phrase_present(query_tokens, fields.clone());
    let exact_coverage = exact_distinct_coverage(query_tokens, fields.clone());
    let exact_subphrase_counts_vector = exact_subphrase_counts_vector(query_tokens, fields);

    let token_matches = query_tokens
        .iter()
        .map(|query_token| best_command_token_match(query_token, candidate_tokens, limit))
        .collect::<Vec<_>>();
    let total_coverage = token_matches
        .iter()
        .filter(|matched| matched.is_some())
        .count();
    if total_coverage < query_tokens.len() {
        return None;
    }

    let fuzzy_subphrase_counts_vector =
        command_fuzzy_subphrase_counts_vector(query_tokens, candidate_tokens, limit);
    let (order_distance, total_gap) = ordered_token_span(
        query_tokens,
        std::iter::once(candidate_tokens),
        |query, token| command_token_match(query, token, limit).is_some(),
    );
    let (distance, operations, compared_len) = token_matches.iter().flatten().fold(
        (0, OperationProfile::default(), 0),
        |(distance, operations, compared_len), matched| {
            (
                distance + matched.distance,
                operations.merge(matched.operations),
                compared_len + matched.compared_len,
            )
        },
    );
    let typo_costs = TypoCostVector::new(
        token_matches
            .iter()
            .map(|matched| {
                matched
                    .as_ref()
                    .map_or(0, |matched| matched.operations.core_typo_cost())
            })
            .collect(),
    );
    let insertions = InsertionVector::new(
        token_matches
            .iter()
            .map(|matched| {
                matched
                    .as_ref()
                    .map_or(0, |matched| matched.operations.insertions)
            })
            .collect(),
    );

    Some(ComputedCommandHistory {
        exact_phrase_present,
        exact_coverage,
        exact_subphrase_counts_vector,
        total_coverage,
        fuzzy_subphrase_counts_vector,
        order_distance,
        total_gap,
        exact_count: exact_coverage,
        span_len: total_gap.saturating_add(total_coverage),
        distance,
        ratio: distance as f64 / compared_len.max(query_len).max(1) as f64,
        operations,
        typo_costs,
        insertions,
    })
}

fn command_token_limit(query_len: usize, global_limit: usize) -> usize {
    global_limit.min((query_len / 3).max(1))
}

#[derive(Clone, Copy)]
struct CommandTokenMatch {
    distance: usize,
    operations: OperationProfile,
    compared_len: usize,
}

fn command_token_match(query: &str, token: &str, global_limit: usize) -> Option<CommandTokenMatch> {
    if token.starts_with(query) {
        return Some(CommandTokenMatch {
            distance: 0,
            operations: OperationProfile::default(),
            compared_len: query.chars().count().max(token.chars().count()),
        });
    }
    let query_len = query.chars().count();
    let token_len = token.chars().count();
    let limit = command_token_limit(query_len, global_limit);
    if query_len.abs_diff(token_len) > limit {
        return None;
    }
    let (distance, operations) = bounded_damerau_levenshtein(token, query, limit)?;
    (distance as f64 / query_len.max(token_len).max(1) as f64 <= 0.5).then_some(CommandTokenMatch {
        distance,
        operations,
        compared_len: query_len.max(token_len),
    })
}

fn best_command_token_match(
    query: &str,
    tokens: &[String],
    limit: usize,
) -> Option<CommandTokenMatch> {
    tokens
        .iter()
        .filter_map(|token| command_token_match(query, token, limit))
        .min_by(|left, right| {
            (left.distance, left.operations, left.compared_len).cmp(&(
                right.distance,
                right.operations,
                right.compared_len,
            ))
        })
}

fn command_fuzzy_subphrase_counts_vector(
    query_tokens: &[String],
    tokens: &[String],
    limit: usize,
) -> Vec<usize> {
    if query_tokens.len() < 3 {
        return Vec::new();
    }
    (2..query_tokens.len())
        .rev()
        .map(|window_len| {
            (0..=query_tokens.len() - window_len)
                .map(|query_start| {
                    let query_window = &query_tokens[query_start..query_start + window_len];
                    tokens
                        .windows(window_len)
                        .filter(|candidate_window| {
                            candidate_window
                                .iter()
                                .zip(query_window)
                                .all(|(token, query)| {
                                    command_token_match(query, token, limit).is_some()
                                })
                        })
                        .count()
                })
                .sum()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate<'a>(
        command: &'a str,
        tokens: &'a [&'a str],
        recency: Recency,
    ) -> CommandHistoryCandidate<'a> {
        CommandHistoryCandidate {
            command,
            tokens,
            recency,
        }
    }

    #[test]
    fn command_history_typo_matches_token() {
        let matched = best_match(candidate("git status", &["git", "status"], 10.0), "gti").unwrap();
        assert_eq!(matched.distance, 1);
    }

    #[test]
    fn prepared_command_history_matches_unprepared_candidate() {
        let query = CommandHistoryQuery::new("gti status").unwrap();
        let prepared = PreparedCommandHistoryCandidate::new("GIT STATUS", 10.0);
        let prepared_match = query.best_match_prepared(&prepared).unwrap();
        let regular_match = query
            .best_match(candidate("GIT STATUS", &["GIT", "STATUS"], 10.0))
            .unwrap();

        assert_eq!(prepared_match.distance, regular_match.distance);
        assert_eq!(prepared_match.rank, regular_match.rank());
    }

    #[test]
    fn multi_token_query_matches_executable_and_argument() {
        let matched = best_match(
            candidate("git status", &["git", "status"], 10.0),
            "git staus",
        )
        .unwrap();
        assert_eq!(matched.distance, 1);
    }

    #[test]
    fn multi_token_query_matches_non_contiguous_search_words() {
        let command = "npm run build && ELECTRON_OZONE_PLATFORM_HINT=wayland \
            NODE_ENV=production \
            CHATGPT_BRIDGE_VISIBLE=1 \
            npx electron .";
        let parsed = parse_command_history_candidate(command, 10.0);
        let matched = best_match(parsed.as_candidate(), "elextron brydge").unwrap();

        assert_eq!(matched.distance, 2);
    }

    #[test]
    fn exact_token_coverage_beats_lower_recency_and_typo_distance() {
        let mut matches = vec![
            best_match(
                candidate("electron bridge", &["electron", "bridge"], 1.0),
                "electron bridge",
            )
            .unwrap(),
            best_match(
                candidate("electron brydge", &["electron", "brydge"], 100.0),
                "electron bridge",
            )
            .unwrap(),
            best_match(
                candidate("elextron brydge", &["elextron", "brydge"], 200.0),
                "electron bridge",
            )
            .unwrap(),
        ];

        sort_matches(&mut matches);

        assert_eq!(matches[0].candidate.command, "electron bridge");
        assert_eq!(matches[0].exact_count, 2);
        assert_eq!(matches[1].candidate.command, "electron brydge");
        assert_eq!(matches[1].exact_count, 1);
        assert_eq!(matches[2].candidate.command, "elextron brydge");
        assert_eq!(matches[2].exact_count, 0);
    }

    #[test]
    fn exact_phrase_beats_scattered_exact_tokens() {
        let mut matches = vec![
            best_match(
                candidate(
                    "electron something bridge",
                    &["electron", "something", "bridge"],
                    100.0,
                ),
                "electron bridge",
            )
            .unwrap(),
            best_match(
                candidate("electron bridge", &["electron", "bridge"], 1.0),
                "electron bridge",
            )
            .unwrap(),
        ];

        sort_matches(&mut matches);

        assert_eq!(matches[0].candidate.command, "electron bridge");
        assert_eq!(matches[0].exact_count, 2);
        assert_eq!(matches[0].span_len, 2);
    }

    #[test]
    fn mixed_exact_typo_phrase_beats_scattered_mixed_match() {
        let mut matches = vec![
            best_match(
                candidate(
                    "electron something brydge",
                    &["electron", "something", "brydge"],
                    100.0,
                ),
                "electron bridge",
            )
            .unwrap(),
            best_match(
                candidate("electron brydge", &["electron", "brydge"], 1.0),
                "electron bridge",
            )
            .unwrap(),
        ];

        sort_matches(&mut matches);

        assert_eq!(matches[0].candidate.command, "electron brydge");
        assert_eq!(matches[0].exact_count, 1);
        assert_eq!(matches[0].span_len, 2);
    }

    #[test]
    fn summed_distance_orders_within_same_exact_coverage() {
        let mut matches = vec![
            best_match(
                candidate("electron brodge", &["electron", "brodge"], 100.0),
                "electron bridge",
            )
            .unwrap(),
            best_match(
                candidate("electron brxxge", &["electron", "brxxge"], 200.0),
                "electron bridge",
            )
            .unwrap(),
        ];

        sort_matches(&mut matches);

        assert_eq!(matches[0].candidate.command, "electron brodge");
        assert!(matches[0].distance < matches[1].distance);
    }

    #[test]
    fn missing_space_query_does_not_match_command_sequence() {
        assert!(
            best_match(
                candidate("git status", &["git", "status"], 10.0),
                "gitstaus",
            )
            .is_none()
        );
    }

    #[test]
    fn extra_space_query_does_not_match_compound_command_token() {
        assert!(best_match(candidate("gitstatus", &["gitstatus"], 10.0), "git staus",).is_none());
    }

    #[test]
    fn recency_breaks_equal_ties() {
        let mut matches = vec![
            best_match(candidate("cargo git", &["cargo", "git"], 100.0), "git").unwrap(),
            best_match(candidate("git status", &["git", "status"], 1.0), "git").unwrap(),
        ];
        sort_matches(&mut matches);
        assert_eq!(matches[0].candidate.command, "cargo git");
    }

    #[test]
    fn lower_distance_beats_recency() {
        let mut matches = vec![
            best_match(
                candidate("electron status", &["electron", "status"], 1.0),
                "elextron",
            )
            .unwrap(),
            best_match(
                candidate("elekron status", &["elekron", "status"], 100.0),
                "elextron",
            )
            .unwrap(),
        ];
        sort_matches(&mut matches);
        assert_eq!(matches[0].candidate.command, "electron status");
    }

    #[test]
    fn recency_breaks_remaining_ties() {
        let mut matches = vec![
            best_match(candidate("git status", &["git", "status"], 1.0), "gti").unwrap(),
            best_match(candidate("git stash", &["git", "stash"], 100.0), "gti").unwrap(),
        ];
        sort_matches(&mut matches);
        assert_eq!(matches[0].candidate.command, "git stash");
    }
}
