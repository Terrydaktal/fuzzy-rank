use std::cmp::{Ordering, Reverse};
use std::collections::HashMap;

use crate::core::{
    DensityRank, compare_key_asc, compare_score_desc, exact_distinct_coverage, exact_phrase_count,
    exact_subphrase_counts_vector, exact_token_occurrence_counts, ordered_token_span,
    saturated_exact_query_token_occurrence_total,
};
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
    pub exact_subphrase_counts_vector: Vec<usize>,
    pub exact_distinct_coverage: usize,
    pub order_distance: usize,
    pub total_gap: usize,
    pub saturated_exact_query_token_occurrence_total: usize,
    pub density: DensityRank,
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

        let fields = std::iter::once(candidate_tokens);
        let counts = exact_token_occurrence_counts(&self.tokens, fields.clone());

        let exact_distinct_coverage = exact_distinct_coverage(&self.tokens, fields.clone());
        if exact_distinct_coverage == 0 {
            return None;
        }

        let exact_whole_phrase_occurrences = exact_phrase_count(&self.tokens, fields.clone());
        let exact_subphrase_counts_vector = exact_subphrase_counts_vector(&self.tokens, fields);
        let (order_distance, total_gap) = ordered_token_span(
            &self.tokens,
            std::iter::once(candidate_tokens),
            |query, candidate| candidate.starts_with(query),
        );
        let saturated_exact_query_token_occurrence_total =
            saturated_exact_query_token_occurrence_total(&counts);
        let density = DensityRank::new(exact_distinct_coverage, candidate_tokens.len());

        Some(MessageMatch {
            key,
            text,
            score,
            exact_whole_phrase_occurrences,
            exact_subphrase_counts_vector,
            exact_distinct_coverage,
            order_distance,
            total_gap,
            saturated_exact_query_token_occurrence_total,
            density,
        })
    }
}

pub fn sort_matches(matches: &mut [MessageMatch<'_>]) {
    matches.sort_by(compare_matches);
}

fn compare_matches(left: &MessageMatch<'_>, right: &MessageMatch<'_>) -> Ordering {
    Reverse(left.exact_whole_phrase_occurrences)
        .cmp(&Reverse(right.exact_whole_phrase_occurrences))
        .then_with(|| {
            Reverse(left.exact_distinct_coverage).cmp(&Reverse(right.exact_distinct_coverage))
        })
        .then_with(|| {
            Reverse(&left.exact_subphrase_counts_vector)
                .cmp(&Reverse(&right.exact_subphrase_counts_vector))
        })
        .then_with(|| left.order_distance.cmp(&right.order_distance))
        .then_with(|| left.total_gap.cmp(&right.total_gap))
        .then_with(|| {
            Reverse(left.saturated_exact_query_token_occurrence_total)
                .cmp(&Reverse(right.saturated_exact_query_token_occurrence_total))
        })
        .then_with(|| left.density.cmp(&right.density))
        .then_with(|| compare_score_desc(left.score, right.score))
        .then_with(|| compare_key_asc(left.key, right.key))
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
    fn broader_coverage_beats_repeated_partial_subphrases() {
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
