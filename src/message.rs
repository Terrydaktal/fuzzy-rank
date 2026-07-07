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

#[derive(Clone, Debug)]
pub struct MessageMatch<'a> {
    pub key: &'a str,
    pub text: &'a str,
    pub score: Score,
    pub exact_whole_phrase_occurrences: usize,
    pub exact_run_vector: Vec<usize>,
    pub exact_distinct_coverage: usize,
    pub all_query_tokens_matched: bool,
    pub same_order_scattered: bool,
    pub closeness_gap_distance: usize,
    pub repeated_exact_phrase_bonus: usize,
    pub repeated_token_bonus: usize,
    pub density_matched_tokens: usize,
    pub density_field_len: usize,
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

        let exact_distinct_coverage = counts.iter().filter(|count| **count > 0).count();
        if exact_distinct_coverage == 0 {
            return None;
        }

        let exact_whole_phrase_occurrences = exact_phrase_window_count(&self.tokens, candidate_tokens);
        let exact_run_vector = exact_run_vector(&self.tokens, candidate_tokens);
        let all_query_tokens_matched = exact_distinct_coverage == self.tokens.len();
        let (same_order_scattered, closeness_gap_distance) =
            same_order_scattered_stats(&self.tokens, candidate_tokens);
        let repeated_exact_phrase_bonus = exact_whole_phrase_occurrences.min(3);
        let repeated_token_bonus = counts.iter().map(|count| (*count).min(3)).sum::<usize>();
        let density_matched_tokens = exact_distinct_coverage;
        let density_field_len = candidate_tokens.len();

        Some(MessageMatch {
            key,
            text,
            score,
            exact_whole_phrase_occurrences,
            exact_run_vector,
            exact_distinct_coverage,
            all_query_tokens_matched,
            same_order_scattered,
            closeness_gap_distance,
            repeated_exact_phrase_bonus,
            repeated_token_bonus,
            density_matched_tokens,
            density_field_len,
        })
    }
}

pub fn sort_matches(matches: &mut [MessageMatch<'_>]) {
    matches.sort_by(compare_matches);
}

fn compare_matches(left: &MessageMatch<'_>, right: &MessageMatch<'_>) -> Ordering {
    Reverse(left.exact_whole_phrase_occurrences)
        .cmp(&Reverse(right.exact_whole_phrase_occurrences))
        .then_with(|| Reverse(&left.exact_run_vector).cmp(&Reverse(&right.exact_run_vector)))
        .then_with(|| Reverse(left.exact_distinct_coverage).cmp(&Reverse(right.exact_distinct_coverage)))
        .then_with(|| Reverse(left.all_query_tokens_matched).cmp(&Reverse(right.all_query_tokens_matched)))
        .then_with(|| Reverse(left.same_order_scattered).cmp(&Reverse(right.same_order_scattered)))
        .then_with(|| left.closeness_gap_distance.cmp(&right.closeness_gap_distance))
        .then_with(|| Reverse(left.repeated_exact_phrase_bonus).cmp(&Reverse(right.repeated_exact_phrase_bonus)))
        .then_with(|| Reverse(left.repeated_token_bonus).cmp(&Reverse(right.repeated_token_bonus)))
        .then_with(|| {
            (right.density_matched_tokens * left.density_field_len)
                .cmp(&(left.density_matched_tokens * right.density_field_len))
        })
        .then_with(|| left.density_field_len.cmp(&right.density_field_len))
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

fn exact_phrase_window_count(query_tokens: &[String], candidate_tokens: &[String]) -> usize {
    if query_tokens.is_empty() || candidate_tokens.len() < query_tokens.len() {
        return 0;
    }
    candidate_tokens
        .windows(query_tokens.len())
        .filter(|window| {
            window
                .iter()
                .zip(query_tokens)
                .all(|(candidate, query)| candidate.starts_with(query))
        })
        .count()
}

fn exact_run_vector(query_tokens: &[String], candidate_tokens: &[String]) -> Vec<usize> {
    if query_tokens.is_empty() {
        return Vec::new();
    }
    (2..=query_tokens.len())
        .rev()
        .map(|window_len| {
            let mut count = 0;
            for query_start in 0..=query_tokens.len() - window_len {
                let query_window = &query_tokens[query_start..query_start + window_len];
                count += candidate_tokens
                    .windows(window_len)
                    .filter(|candidate_window| {
                        candidate_window
                            .iter()
                            .zip(query_window.iter())
                            .all(|(candidate, query)| candidate.starts_with(query))
                    })
                    .count();
            }
            count
        })
        .collect()
}

fn same_order_scattered_stats(query_tokens: &[String], candidate_tokens: &[String]) -> (bool, usize) {
    let mut positions = Vec::new();
    let mut search_from = 0usize;
    for query_token in query_tokens {
        let Some((idx, _)) = candidate_tokens
            .iter()
            .enumerate()
            .skip(search_from)
            .find(|(_, candidate_token)| candidate_token.starts_with(query_token))
        else {
            return (false, usize::MAX);
        };
        positions.push(idx);
        search_from = idx + 1;
    }
    let span = positions.last().copied().unwrap_or(0) - positions.first().copied().unwrap_or(0);
    let gap = span.saturating_sub(query_tokens.len().saturating_sub(1));
    (true, gap)
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
    fn exact_phrase_count_can_beat_scattered_full_coverage() {
        let candidates = [
            prepare("all", "word1 far away word2 and then word3", 0.0),
            prepare("partial-phrase", "word1 word2 word1 word2", 10.0),
        ];

        assert_eq!(
            matched_keys("word1 word2 word3", &candidates),
            vec!["partial-phrase", "all"]
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
    fn score_breaks_equal_exact_message_ties() {
        let candidates = [
            prepare("later", "intro words word1 word2 word3", 10.0),
            prepare("earlier", "word1 word2 word3 later words", 0.0),
        ];

        assert_eq!(
            matched_keys("word1 word2 word3", &candidates),
            vec!["later", "earlier"]
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
