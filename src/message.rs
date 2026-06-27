use std::cmp::{Ordering, Reverse};
use std::collections::HashMap;

use crate::text::{split_search_tokens, to_lowercase};

pub type Score = f64;

#[derive(Clone, Copy, Debug)]
pub struct MessageCandidate<'a> {
    pub key: &'a str,
    pub text: &'a str,
    pub score: Score,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MessageQuery {
    tokens: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct PreparedMessageCandidate<'a> {
    pub key: &'a str,
    pub text: &'a str,
    pub score: Score,
    lowered: String,
    tokens: Vec<String>,
    term_counts: HashMap<String, usize>,
}

#[derive(Clone, Copy, Debug)]
pub struct MessageMatch<'a> {
    pub key: &'a str,
    pub text: &'a str,
    pub score: Score,
    pub phrase_occurrences: usize,
    pub phrase_position: Option<usize>,
    pub adjacent_run_len: usize,
    pub adjacent_run_position: Option<usize>,
    pub covering_span_len: Option<usize>,
    pub covering_span_position: Option<usize>,
    pub matched_terms: usize,
    pub total_occurrences: usize,
}

impl<'a> MessageCandidate<'a> {
    pub fn prepare(self) -> PreparedMessageCandidate<'a> {
        let lowered = to_lowercase(self.text);
        let tokens = split_search_tokens(&lowered)
            .into_iter()
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>();
        let mut term_counts = HashMap::new();
        for token in &tokens {
            *term_counts.entry(token.clone()).or_insert(0) += 1;
        }

        PreparedMessageCandidate {
            key: self.key,
            text: self.text,
            score: self.score,
            lowered,
            tokens,
            term_counts,
        }
    }
}

impl MessageQuery {
    pub fn new(query: &str) -> Option<Self> {
        let lowered = to_lowercase(query);
        let mut tokens = Vec::new();
        for token in split_search_tokens(&lowered) {
            if !tokens.iter().any(|existing| existing == token) {
                tokens.push(token.to_owned());
            }
        }

        (!tokens.is_empty()).then_some(Self { tokens })
    }

    pub fn search_rank<'a>(&self, candidate: MessageCandidate<'a>) -> Option<MessageMatch<'a>> {
        let prepared = candidate.prepare();
        self.rank_from_parts(
            prepared.key,
            prepared.text,
            prepared.score,
            &prepared.tokens,
            &prepared.term_counts,
        )
    }

    pub fn search_rank_prepared<'a>(
        &self,
        candidate: &'a PreparedMessageCandidate<'a>,
    ) -> Option<MessageMatch<'a>> {
        self.rank_from_parts(
            candidate.key,
            candidate.text,
            candidate.score,
            &candidate.tokens,
            &candidate.term_counts,
        )
    }

    fn rank_from_parts<'a>(
        &self,
        key: &'a str,
        text: &'a str,
        score: Score,
        candidate_tokens: &[String],
        _term_counts: &HashMap<String, usize>,
    ) -> Option<MessageMatch<'a>> {
        if candidate_tokens.is_empty() {
            return None;
        }

        let counts = self
            .tokens
            .iter()
            .map(|query_token| count_token_prefix_occurrences(query_token, candidate_tokens))
            .collect::<Vec<_>>();

        let matched_terms = counts.iter().filter(|count| **count > 0).count();
        if matched_terms == 0 {
            return None;
        }

        let total_occurrences = counts.iter().sum();
        let (phrase_occurrences, phrase_position) = phrase_stats(&self.tokens, candidate_tokens);
        let (adjacent_run_len, adjacent_run_position) =
            adjacent_run_stats(&self.tokens, candidate_tokens);
        let (covering_span_len, covering_span_position) =
            covering_span_stats(&self.tokens, candidate_tokens);

        Some(MessageMatch {
            key,
            text,
            score,
            phrase_occurrences,
            phrase_position,
            adjacent_run_len,
            adjacent_run_position,
            covering_span_len,
            covering_span_position,
            matched_terms,
            total_occurrences,
        })
    }
}

pub fn sort_matches(matches: &mut [MessageMatch<'_>]) {
    matches.sort_by(|left, right| compare_matches(*left, *right));
}

fn compare_matches(left: MessageMatch<'_>, right: MessageMatch<'_>) -> Ordering {
    (
        Reverse(left.matched_terms),
        Reverse(left.phrase_occurrences > 0),
        Reverse(left.adjacent_run_len),
        left.covering_span_len.unwrap_or(usize::MAX),
        left.covering_span_position.unwrap_or(usize::MAX),
        Reverse(left.total_occurrences),
    )
        .cmp(&(
            Reverse(right.matched_terms),
            Reverse(right.phrase_occurrences > 0),
            Reverse(right.adjacent_run_len),
            right.covering_span_len.unwrap_or(usize::MAX),
            right.covering_span_position.unwrap_or(usize::MAX),
            Reverse(right.total_occurrences),
        ))
        .then_with(|| right.score.total_cmp(&left.score))
        .then_with(|| left.key.cmp(right.key))
}

pub fn contains_query_signal(
    query: &MessageQuery,
    candidate: &PreparedMessageCandidate<'_>,
) -> bool {
    query.tokens.iter().any(|token| {
        candidate.term_counts.contains_key(token)
            || candidate
                .tokens
                .iter()
                .any(|candidate_token| candidate_token.starts_with(token))
    }) || candidate.lowered.contains(query.tokens[0].as_str())
}

fn count_token_prefix_occurrences(query_token: &str, candidate_tokens: &[String]) -> usize {
    candidate_tokens
        .iter()
        .filter(|candidate_token| candidate_token.starts_with(query_token))
        .count()
}

fn phrase_stats(query_tokens: &[String], candidate_tokens: &[String]) -> (usize, Option<usize>) {
    if query_tokens.is_empty() || query_tokens.len() > candidate_tokens.len() {
        return (0, None);
    }

    let mut count = 0;
    let mut first_position = None;
    for (idx, window) in candidate_tokens.windows(query_tokens.len()).enumerate() {
        if window
            .iter()
            .zip(query_tokens)
            .all(|(candidate, query)| candidate.starts_with(query))
        {
            count += 1;
            first_position.get_or_insert(idx);
        }
    }

    (count, first_position)
}

fn adjacent_run_stats(
    query_tokens: &[String],
    candidate_tokens: &[String],
) -> (usize, Option<usize>) {
    let max_len = query_tokens.len().min(candidate_tokens.len());
    for len in (2..=max_len).rev() {
        let mut first_position: Option<usize> = None;
        for query_window in query_tokens.windows(len) {
            for (idx, candidate_window) in candidate_tokens.windows(len).enumerate() {
                if candidate_window
                    .iter()
                    .zip(query_window)
                    .all(|(candidate, query)| candidate.starts_with(query))
                {
                    first_position = Some(first_position.map_or(idx, |current| current.min(idx)));
                }
            }
        }

        if first_position.is_some() {
            return (len, first_position);
        }
    }

    (0, None)
}

fn covering_span_stats(
    query_tokens: &[String],
    candidate_tokens: &[String],
) -> (Option<usize>, Option<usize>) {
    if query_tokens.is_empty() || query_tokens.len() > candidate_tokens.len() {
        return (None, None);
    }

    let mut best_len = usize::MAX;
    let mut best_start = usize::MAX;

    for start in 0..candidate_tokens.len() {
        let mut matched = vec![false; query_tokens.len()];
        let mut matched_count = 0;

        for (end, candidate) in candidate_tokens.iter().enumerate().skip(start) {
            for (query_idx, query) in query_tokens.iter().enumerate() {
                if !matched[query_idx] && candidate.starts_with(query) {
                    matched[query_idx] = true;
                    matched_count += 1;
                }
            }

            if matched_count == query_tokens.len() {
                let span_len = end - start + 1;
                if span_len < best_len || (span_len == best_len && start < best_start) {
                    best_len = span_len;
                    best_start = start;
                }
                break;
            }
        }
    }

    if best_len == usize::MAX {
        (None, None)
    } else {
        (Some(best_len), Some(best_start))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MessageCandidate, MessageQuery, PreparedMessageCandidate, contains_query_signal,
        sort_matches,
    };

    fn matched_keys<'a>(
        query: &str,
        candidates: &'a [PreparedMessageCandidate<'a>],
    ) -> Vec<&'a str> {
        let query = MessageQuery::new(query).unwrap();
        let mut matches = candidates
            .iter()
            .filter_map(|candidate| query.search_rank_prepared(candidate))
            .collect::<Vec<_>>();
        sort_matches(&mut matches);
        matches.into_iter().map(|matched| matched.key).collect()
    }

    fn prepare<'a>(key: &'a str, text: &'a str, score: f64) -> PreparedMessageCandidate<'a> {
        MessageCandidate { key, text, score }.prepare()
    }

    #[test]
    fn phrase_match_beats_non_phrase_match() {
        let candidates = [
            prepare("phrase", "word1 word2 together", 0.0),
            prepare("split", "word1 between other word2", 0.0),
        ];

        assert_eq!(
            matched_keys("word1 word2", &candidates),
            vec!["phrase", "split"]
        );
    }

    #[test]
    fn full_coverage_beats_partial_with_more_occurrences() {
        let candidates = [
            prepare("both", "word1 word2", 0.0),
            prepare(
                "one-many",
                "word1 word1 word1 word1 word1 word1 word1 word1 word1 word1",
                0.0,
            ),
        ];

        assert_eq!(
            matched_keys("word1 word2", &candidates),
            vec!["both", "one-many"]
        );
    }

    #[test]
    fn full_coverage_beats_partial_phrase() {
        let candidates = [
            prepare("all", "word1 far away word2 and then word3", 0.0),
            prepare("partial-phrase", "word1 word2 word1 word2", 10.0),
        ];

        assert_eq!(
            matched_keys("word1 word2 word3", &candidates),
            vec!["all", "partial-phrase"]
        );
    }

    #[test]
    fn more_occurrences_break_ties_within_same_coverage() {
        let candidates = [
            prepare("many", "word1 word2 word1 word2", 0.0),
            prepare("few", "word1 word2", 0.0),
        ];

        assert_eq!(
            matched_keys("word1 word2", &candidates),
            vec!["many", "few"]
        );
    }

    #[test]
    fn smaller_covering_span_beats_more_occurrences() {
        let candidates = [
            prepare(
                "frequent-scattered",
                "word1 word1 word1 many unrelated words before word2 and then word3",
                10.0,
            ),
            prepare("close", "word1 word2 word3", 0.0),
        ];

        assert_eq!(
            matched_keys("word1 word2 word3", &candidates),
            vec!["close", "frequent-scattered"]
        );
    }

    #[test]
    fn earlier_covering_span_breaks_equal_span_ties() {
        let candidates = [
            prepare("later", "intro words word1 word2 word3", 10.0),
            prepare("earlier", "word1 word2 word3 later words", 0.0),
        ];

        assert_eq!(
            matched_keys("word1 word2 word3", &candidates),
            vec!["earlier", "later"]
        );
    }

    #[test]
    fn more_occurrences_of_single_matched_word_break_partial_ties() {
        let candidates = [
            prepare("many-word1", "word1 word1 word1", 0.0),
            prepare("one-word1", "word1", 0.0),
        ];

        assert_eq!(
            matched_keys("word1 word2", &candidates),
            vec!["many-word1", "one-word1"]
        );
    }

    #[test]
    fn score_breaks_remaining_ties() {
        let candidates = [
            prepare("lower", "word1 word2", 1.0),
            prepare("higher", "word1 word2", 2.0),
        ];

        assert_eq!(
            matched_keys("word1 word2", &candidates),
            vec!["higher", "lower"]
        );
    }

    #[test]
    fn token_prefixes_count_as_term_hits() {
        let candidates = [
            prepare("full", "assistant cachyos", 0.0),
            prepare("partial", "cachyos only", 10.0),
        ];

        assert_eq!(
            matched_keys("cachyos ass", &candidates),
            vec!["full", "partial"]
        );
    }

    #[test]
    fn prefix_phrase_match_beats_split_prefix_match() {
        let candidates = [
            prepare("phrase", "assistant cachyos", 0.0),
            prepare("split", "assistant between cachyos", 0.0),
        ];

        assert_eq!(
            matched_keys("ass cach", &candidates),
            vec!["phrase", "split"]
        );
    }

    #[test]
    fn earlier_phrase_beats_later_phrase() {
        let candidates = [
            prepare(
                "later",
                "cachyos mentioned early before many words and then cachyos assumptions",
                10.0,
            ),
            prepare("earlier", "cachyos assumptions near the start", 0.0),
        ];

        assert_eq!(
            matched_keys("cachyos as", &candidates),
            vec!["earlier", "later"]
        );
    }

    #[test]
    fn adjacent_query_subphrase_beats_scattered_full_coverage() {
        let candidates = [
            prepare(
                "scattered",
                "cachyos appears before a developer note about an assistant",
                10.0,
            ),
            prepare("adjacent", "cachyos ~/Dev/assistant", 0.0),
        ];

        assert_eq!(
            matched_keys("cachy /dev/assistant", &candidates),
            vec!["adjacent", "scattered"]
        );
    }

    #[test]
    fn signal_check_is_exact_token_or_fast_substring_hit() {
        let query = MessageQuery::new("word1 word2").unwrap();
        let matching = prepare("matching", "word2 appears here", 0.0);
        let partial = prepare("partial", "this contains word1x only", 0.0);

        assert!(contains_query_signal(&query, &matching));
        assert!(contains_query_signal(&query, &partial));
    }
}
