use crate::core::{OperationProfile, bounded_damerau_levenshtein, max_typos, ratio_milli};
use crate::ranking::{TypoSortKey, compare_typo_sort_keys, typo_keys_are_ambiguous};
use crate::text::{has_query_signal, split_search_tokens, split_whitespace_tokens, to_lowercase};

pub type Recency = f64;

#[derive(Clone, Copy, Debug)]
pub struct CommandHistoryCandidate<'a> {
    pub command: &'a str,
    pub tokens: &'a [&'a str],
    pub recency: Recency,
}

#[derive(Clone, Copy, Debug)]
pub struct CommandHistoryMatch<'a> {
    pub candidate: CommandHistoryCandidate<'a>,
    pub exact_count: usize,
    pub span_len: usize,
    pub distance: usize,
    pub ratio: f64,
    pub operations: OperationProfile,
}

impl<'a> CommandHistoryMatch<'a> {
    fn sort_key(&self) -> TypoSortKey<'_> {
        TypoSortKey {
            exact_count: self.exact_count,
            span_len: self.span_len,
            distance: self.distance,
            operations: self.operations,
            ratio_milli: ratio_milli(self.distance, self.compared_len()),
            recency: self.candidate.recency,
            alphabetical: self.candidate.command,
        }
    }

    fn compared_len(&self) -> usize {
        let query_estimate = if self.ratio == 0.0 {
            1
        } else {
            ((self.distance as f64 / self.ratio).round() as usize).max(1)
        };
        query_estimate.max(1)
    }

    pub fn compare(&self, other: &Self) -> std::cmp::Ordering {
        compare_typo_sort_keys(&self.sort_key(), &other.sort_key())
    }
}

pub struct CommandHistoryQuery {
    text: String,
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

        Some(Self {
            text: query.clone(),
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
        best_match_inner(&lowered, &self.text, self.len, self.limit)
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
    typo_keys_are_ambiguous(&first.sort_key(), &second.sort_key())
}

struct LoweredCandidate<'a> {
    original: CommandHistoryCandidate<'a>,
    command: String,
    tokens: Vec<String>,
}

fn best_match_inner<'a>(
    candidate: &LoweredCandidate<'a>,
    query: &str,
    query_len: usize,
    limit: usize,
) -> Option<CommandHistoryMatch<'a>> {
    if !has_query_signal(split_whitespace_tokens(query), &candidate.command) {
        return None;
    }

    let query_tokens = split_whitespace_tokens(query);
    if query_tokens.is_empty() || candidate.tokens.is_empty() {
        return None;
    }

    let mut best = None;

    if query_tokens.len() == 1 {
        for token in &candidate.tokens {
            update_best(
                &mut best,
                candidate_for_token(candidate.original, token, query, query_len, limit),
            );
        }
    } else if query_tokens.len() <= candidate.tokens.len() {
        update_best(
            &mut best,
            candidate_for_ordered_token_subsequence(
                candidate.original,
                &candidate.tokens,
                &query_tokens,
                query_len,
                limit,
            ),
        );
    }

    best
}

fn candidate_for_token<'a>(
    candidate: CommandHistoryCandidate<'a>,
    token: &str,
    query: &str,
    query_len: usize,
    limit: usize,
) -> Option<CommandHistoryMatch<'a>> {
    if token.is_empty() || limit == 0 {
        return None;
    }

    let token_len = token.chars().count();
    let token_limit = command_token_limit(query_len, limit);
    if query_len.abs_diff(token_len) > token_limit {
        return None;
    }

    let (distance, operations) = bounded_damerau_levenshtein(token, query, token_limit)?;
    let ratio = distance as f64 / query_len.max(token_len) as f64;
    if ratio > 0.5 {
        return None;
    }

    Some(CommandHistoryMatch {
        candidate,
        exact_count: usize::from(distance == 0),
        span_len: 1,
        distance,
        ratio,
        operations,
    })
}

fn candidate_for_ordered_token_subsequence<'a>(
    candidate: CommandHistoryCandidate<'a>,
    tokens: &[String],
    query_tokens: &[&str],
    query_len: usize,
    limit: usize,
) -> Option<CommandHistoryMatch<'a>> {
    let best = best_ordered_token_subsequence(tokens, query_tokens, 0, 0, limit)?;
    let (exact_count, span_len, _, distance, operations, compared_len) = best?;
    let ratio = distance as f64 / query_len.max(compared_len) as f64;
    if ratio > 0.5 {
        return None;
    }

    Some(CommandHistoryMatch {
        candidate,
        exact_count,
        span_len,
        distance,
        ratio,
        operations,
    })
}

fn best_ordered_token_subsequence(
    tokens: &[String],
    query_tokens: &[&str],
    query_idx: usize,
    token_start: usize,
    remaining: usize,
) -> Option<Option<(usize, usize, usize, usize, OperationProfile, usize)>> {
    if query_idx == query_tokens.len() {
        return Some(Some((0, 0, 0, 0, OperationProfile::default(), 0)));
    }

    let remaining_query = query_tokens.len() - query_idx;
    if tokens.len().saturating_sub(token_start) < remaining_query {
        return Some(None);
    }

    let query_token = query_tokens[query_idx];
    let query_token_len = query_token.chars().count();
    let last_start = tokens.len() - remaining_query;
    let mut best = None;

    for token_idx in token_start..=last_start {
        let token = &tokens[token_idx];
        let token_len = token.chars().count();
        let token_limit = command_token_limit(query_token_len, remaining);
        if query_token_len.abs_diff(token_len) > token_limit {
            continue;
        }

        let Some((token_distance, token_operations)) =
            bounded_damerau_levenshtein(token, query_token, token_limit)
        else {
            continue;
        };
        if token_distance > remaining {
            continue;
        }

        let Some(tail) = best_ordered_token_subsequence(
            tokens,
            query_tokens,
            query_idx + 1,
            token_idx + 1,
            remaining - token_distance,
        ) else {
            continue;
        };
        let Some((
            tail_exact_count,
            _tail_span_len,
            tail_last_idx,
            tail_distance,
            tail_operations,
            tail_compared_len,
        )) = tail
        else {
            continue;
        };

        let total = (
            usize::from(token_distance == 0) + tail_exact_count,
            if query_idx + 1 == query_tokens.len() {
                1
            } else {
                tail_last_idx - token_idx + 1
            },
            if query_idx + 1 == query_tokens.len() {
                token_idx
            } else {
                tail_last_idx
            },
            token_distance + tail_distance,
            token_operations.merge(tail_operations),
            query_token_len.max(token_len) + tail_compared_len,
        );
        best = Some(match best {
            None => total,
            Some(current) if command_history_match_tuple_compare(total, current).is_lt() => total,
            Some(current) => current,
        });
    }

    Some(best)
}

fn command_history_match_tuple_compare(
    left: (usize, usize, usize, usize, OperationProfile, usize),
    right: (usize, usize, usize, usize, OperationProfile, usize),
) -> std::cmp::Ordering {
    right
        .0
        .cmp(&left.0)
        .then_with(|| left.1.cmp(&right.1))
        .then_with(|| left.3.cmp(&right.3))
        .then_with(|| left.4.cmp(&right.4))
        .then_with(|| left.5.cmp(&right.5))
}

fn command_token_limit(query_len: usize, global_limit: usize) -> usize {
    global_limit.min((query_len / 3).max(1))
}

fn update_best<'a>(
    best: &mut Option<CommandHistoryMatch<'a>>,
    candidate: Option<CommandHistoryMatch<'a>>,
) {
    match (best.as_ref(), candidate) {
        (_, None) => {}
        (None, Some(candidate)) => *best = Some(candidate),
        (Some(current), Some(candidate)) if candidate.compare(current).is_lt() => {
            *best = Some(candidate)
        }
        _ => {}
    }
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
