use std::cmp::{Ordering, Reverse};

use crate::core::{
    CoverageRank, DensityRank, FieldOccurrenceRank, compare_key_asc, compare_score_desc,
    exact_phrase_occurrence_count_by, exact_phrase_present_by, exact_subphrase_counts_vector_by,
    exact_token_position, ordered_token_span, position_vector,
};
use crate::text::{split_search_tokens, to_lowercase};

pub type Score = f64;

#[derive(Clone, Copy, Debug)]
pub struct MessageField<'a> {
    pub priority: u8,
    pub value: &'a str,
}

#[derive(Clone, Copy, Debug)]
pub struct MessageCandidate<'a, 'b> {
    pub key: &'a str,
    pub fields: &'b [MessageField<'a>],
    pub score: Score,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MessageQuery {
    tokens: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct PreparedMessageCandidate<'a> {
    pub key: &'a str,
    pub score: Score,
    fields: Vec<PreparedMessageField>,
}

#[derive(Clone, Debug)]
struct PreparedMessageField {
    priority: u8,
    tokens: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct MessageMatch<'a> {
    pub key: &'a str,
    pub score: Score,
    pub exact_whole_phrase_present: bool,
    pub exact_subphrase_counts_vector: Vec<usize>,
    pub exact_weighted_positional_coverage_vector: CoverageRank,
    pub exact_field_quality: u8,
    pub order_distance: usize,
    pub total_gap: usize,
    pub exact_phrase_occurrence_rank: FieldOccurrenceRank,
    pub exact_token_occurrence_rank: FieldOccurrenceRank,
    pub density: DensityRank,
}

pub type Field<'a> = MessageField<'a>;
pub type Candidate<'a, 'b> = MessageCandidate<'a, 'b>;
pub type Query = MessageQuery;
pub type PreparedCandidate<'a> = PreparedMessageCandidate<'a>;
pub type Match<'a> = MessageMatch<'a>;

impl<'a, 'b> MessageCandidate<'a, 'b> {
    pub fn prepare(self) -> PreparedMessageCandidate<'a> {
        PreparedMessageCandidate {
            key: self.key,
            score: self.score,
            fields: self
                .fields
                .iter()
                .filter_map(|field| {
                    let lowered = to_lowercase(field.value.trim());
                    let tokens = split_search_tokens(&lowered)
                        .into_iter()
                        .map(ToOwned::to_owned)
                        .collect::<Vec<_>>();
                    (!tokens.is_empty()).then_some(PreparedMessageField {
                        priority: field.priority,
                        tokens,
                    })
                })
                .collect(),
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

    pub fn search_rank<'a, 'b>(
        &self,
        candidate: MessageCandidate<'a, 'b>,
    ) -> Option<MessageMatch<'a>> {
        let prepared = candidate.prepare();
        self.rank_from_parts(prepared.key, prepared.score, &prepared.fields)
    }

    pub fn search_rank_prepared<'a>(
        &self,
        candidate: &'a PreparedMessageCandidate<'a>,
    ) -> Option<MessageMatch<'a>> {
        self.rank_from_parts(candidate.key, candidate.score, &candidate.fields)
    }

    fn rank_from_parts<'a>(
        &self,
        key: &'a str,
        score: Score,
        fields: &[PreparedMessageField],
    ) -> Option<MessageMatch<'a>> {
        if fields.is_empty() {
            return None;
        }

        let field_tokens = fields.iter().map(|field| field.tokens.as_slice());
        let exact_weighted_positional_coverage_vector = CoverageRank::new(
            position_vector(
                self.tokens.iter().map(|query| {
                    fields
                        .iter()
                        .flat_map(|field| field.tokens.iter())
                        .filter_map(|candidate| exact_token_position(query, candidate))
                        .min()
                }),
                4,
            ),
            &[4, 2, 2, 1],
        );
        if exact_weighted_positional_coverage_vector.distinct_coverage == 0 {
            return None;
        }

        let exact_whole_phrase_present =
            exact_phrase_present_by(&self.tokens, field_tokens.clone(), literal_token_match);
        let exact_subphrase_counts_vector = exact_subphrase_counts_vector_by(
            &self.tokens,
            field_tokens.clone(),
            literal_token_match,
        );
        let exact_field_quality = fields
            .iter()
            .filter(|field| {
                field.tokens.iter().any(|candidate| {
                    self.tokens
                        .iter()
                        .any(|query| literal_token_match(query, candidate))
                })
            })
            .map(|field| field.priority)
            .min()
            .unwrap_or(u8::MAX);
        let (order_distance, total_gap) =
            ordered_token_span(&self.tokens, field_tokens.clone(), literal_token_match);
        let exact_phrase_occurrence_rank =
            FieldOccurrenceRank::from_entries(fields.iter().map(|field| {
                (
                    field.priority,
                    exact_phrase_occurrence_count_by(
                        &self.tokens,
                        std::iter::once(field.tokens.as_slice()),
                        literal_token_match,
                    ),
                )
            }));
        let exact_token_occurrence_rank =
            FieldOccurrenceRank::from_entries(fields.iter().map(|field| {
                (
                    field.priority,
                    field
                        .tokens
                        .iter()
                        .map(|candidate| {
                            self.tokens
                                .iter()
                                .filter(|query| literal_token_match(query, candidate))
                                .count()
                        })
                        .sum(),
                )
            }));
        let density = DensityRank::new(
            exact_weighted_positional_coverage_vector.distinct_coverage,
            fields
                .iter()
                .filter(|field| {
                    field.tokens.iter().any(|candidate| {
                        self.tokens
                            .iter()
                            .any(|query| literal_token_match(query, candidate))
                    })
                })
                .map(|field| field.tokens.len())
                .min()
                .unwrap_or(usize::MAX),
        );

        Some(MessageMatch {
            key,
            score,
            exact_whole_phrase_present,
            exact_subphrase_counts_vector,
            exact_weighted_positional_coverage_vector,
            exact_field_quality,
            order_distance,
            total_gap,
            exact_phrase_occurrence_rank,
            exact_token_occurrence_rank,
            density,
        })
    }
}

pub fn sort_matches(matches: &mut [MessageMatch<'_>]) {
    matches.sort_unstable_by(compare_matches);
}

/// Retain and order only the best `limit` matches.
pub fn select_top_k(matches: &mut Vec<MessageMatch<'_>>, limit: usize) {
    if matches.len() <= limit {
        sort_matches(matches);
        return;
    }
    if limit == 0 {
        matches.clear();
        return;
    }

    let _ = matches.select_nth_unstable_by(limit, compare_matches);
    matches.truncate(limit);
    matches.sort_unstable_by(compare_matches);
}

fn compare_matches(left: &MessageMatch<'_>, right: &MessageMatch<'_>) -> Ordering {
    Reverse(left.exact_whole_phrase_present)
        .cmp(&Reverse(right.exact_whole_phrase_present))
        .then_with(|| {
            Reverse(&left.exact_subphrase_counts_vector)
                .cmp(&Reverse(&right.exact_subphrase_counts_vector))
        })
        .then_with(|| {
            Reverse(&left.exact_weighted_positional_coverage_vector)
                .cmp(&Reverse(&right.exact_weighted_positional_coverage_vector))
        })
        .then_with(|| left.exact_field_quality.cmp(&right.exact_field_quality))
        .then_with(|| left.order_distance.cmp(&right.order_distance))
        .then_with(|| left.total_gap.cmp(&right.total_gap))
        .then_with(|| {
            Reverse(&left.exact_phrase_occurrence_rank)
                .cmp(&Reverse(&right.exact_phrase_occurrence_rank))
        })
        .then_with(|| {
            Reverse(&left.exact_token_occurrence_rank)
                .cmp(&Reverse(&right.exact_token_occurrence_rank))
        })
        .then_with(|| left.density.cmp(&right.density))
        .then_with(|| compare_score_desc(left.score, right.score))
        .then_with(|| compare_key_asc(left.key, right.key))
}

pub fn contains_query_signal(
    query: &MessageQuery,
    candidate: &PreparedMessageCandidate<'_>,
) -> bool {
    candidate.fields.iter().any(|field| {
        query.tokens.iter().any(|query_token| {
            field
                .tokens
                .iter()
                .any(|candidate_token| literal_token_match(query_token, candidate_token))
        })
    })
}

fn literal_token_match(query: &str, candidate: &str) -> bool {
    candidate.contains(query)
}

#[cfg(test)]
mod tests {
    use super::{
        MessageCandidate, MessageField, MessageQuery, PreparedMessageCandidate,
        contains_query_signal, sort_matches,
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
        let fields = [MessageField {
            priority: 0,
            value: text,
        }];
        MessageCandidate {
            key,
            fields: &fields,
            score,
        }
        .prepare()
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
    fn longer_exact_subphrases_beat_broader_scattered_coverage() {
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
    fn arbitrary_token_substrings_count_as_exact_hits() {
        let candidates = [
            prepare("substring", "assistant", 0.0),
            prepare("missing", "operator", 0.0),
        ];

        assert_eq!(matched_keys("ssis", &candidates), vec!["substring"]);
    }

    #[test]
    fn higher_priority_field_beats_lower_priority_field() {
        let query = MessageQuery::new("search term").unwrap();
        let low_priority = [MessageField {
            priority: 1,
            value: "search term",
        }];
        let high_priority = [MessageField {
            priority: 0,
            value: "search term",
        }];
        let mut matches = vec![
            query
                .search_rank(MessageCandidate {
                    key: "low",
                    fields: &low_priority,
                    score: 100.0,
                })
                .unwrap(),
            query
                .search_rank(MessageCandidate {
                    key: "high",
                    fields: &high_priority,
                    score: 0.0,
                })
                .unwrap(),
        ];
        sort_matches(&mut matches);
        assert_eq!(matches[0].key, "high");
    }

    #[test]
    fn exact_phrases_do_not_cross_message_fields() {
        let fields = [
            MessageField {
                priority: 0,
                value: "word1",
            },
            MessageField {
                priority: 1,
                value: "word2",
            },
        ];
        let query = MessageQuery::new("word1 word2").unwrap();
        let matched = query
            .search_rank(MessageCandidate {
                key: "split-fields",
                fields: &fields,
                score: 0.0,
            })
            .unwrap();

        assert!(!matched.exact_whole_phrase_present);
        assert_eq!(
            matched
                .exact_weighted_positional_coverage_vector
                .distinct_coverage,
            2
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
