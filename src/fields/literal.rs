use std::cell::OnceCell;
use std::cmp::{Ordering, Reverse};

use crate::core::{
    CoverageRank, DensityRank, FieldOccurrenceRank, TopK, compare_key_asc, compare_score_desc,
    exact_phrase_occurrence_count_by, exact_phrase_present_by, exact_subphrase_counts_vector_by,
    exact_token_position, ordered_token_span, position_vector,
};
use crate::fields::model::{
    FieldRankFeatures, FieldRankModel, density_quality, feature, longest_subphrase,
    occurrence_features, quality_from_priority, rerank_sorted_prefix, set_coverage_features,
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
    distinct_tokens: Vec<String>,
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
    pub query_token_count: usize,
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

impl MessageMatch<'_> {
    /// Convert this exact field match into the stable feature vector used by
    /// [`FieldRankModel`].  Fuzzy-only features remain zero for this exact
    /// adaptor.
    pub fn rank_features(&self) -> FieldRankFeatures {
        let mut features = FieldRankFeatures::zero();
        features.set(feature::SEARCH_RANK_QUALITY, 1.0);
        features.set(
            feature::EXACT_PHRASE_PRESENT,
            f64::from(self.exact_whole_phrase_present),
        );
        features.set(
            feature::EXACT_SUBPHRASE_PRESENCE,
            self.exact_subphrase_counts_vector
                .iter()
                .filter(|count| **count != 0)
                .count() as f64,
        );
        features.set(
            feature::EXACT_LONGEST_SUBPHRASE,
            longest_subphrase(&self.exact_subphrase_counts_vector, self.query_token_count),
        );
        set_coverage_features(
            &mut features,
            feature::EXACT_COVERAGE_WHOLE,
            &self.exact_weighted_positional_coverage_vector,
        );
        features.set(
            feature::EXACT_FIELD_QUALITY,
            quality_from_priority(self.exact_field_quality),
        );
        features.set(feature::ORDER_QUALITY, -(self.order_distance as f64));
        features.set(feature::GAP_QUALITY, -(self.total_gap as f64));
        let (phrase_priority, phrase_count, phrase_total) =
            occurrence_features(&self.exact_phrase_occurrence_rank);
        features.set(feature::EXACT_PHRASE_OCCURRENCE_PRIORITY, phrase_priority);
        features.set(feature::EXACT_PHRASE_OCCURRENCE_COUNT, phrase_count);
        features.set(feature::EXACT_PHRASE_OCCURRENCE_TOTAL, phrase_total);
        let (token_priority, token_count, token_total) =
            occurrence_features(&self.exact_token_occurrence_rank);
        features.set(feature::EXACT_TOKEN_OCCURRENCE_PRIORITY, token_priority);
        features.set(feature::EXACT_TOKEN_OCCURRENCE_COUNT, token_count);
        features.set(feature::EXACT_TOKEN_OCCURRENCE_TOTAL, token_total);
        features.set(feature::DENSITY_QUALITY, density_quality(self.density));
        features.set(feature::CALLER_SCORE, self.score.max(0.0).ln_1p());
        features
    }
}

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
        let tokens: Vec<_> = split_search_tokens(&lowered)
            .into_iter()
            .map(str::to_owned)
            .collect();
        let mut distinct_tokens = Vec::new();
        for token in &tokens {
            if !distinct_tokens.contains(token) {
                distinct_tokens.push(token.clone());
            }
        }

        (!tokens.is_empty()).then_some(Self {
            tokens,
            distinct_tokens,
        })
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
        candidate: &PreparedMessageCandidate<'a>,
    ) -> Option<MessageMatch<'a>> {
        self.rank_from_parts(candidate.key, candidate.score, &candidate.fields)
    }

    /// Retrieve the best literal field matches, evaluating subordinate relevance
    /// stages only on ties. Only the retained results are fully materialized.
    pub fn search_top_k_prepared<'a>(
        &self,
        candidates: &[PreparedMessageCandidate<'a>],
        limit: usize,
    ) -> Vec<MessageMatch<'a>> {
        if limit == 0 {
            return Vec::new();
        }
        let mut matches = TopK::new(limit.min(candidates.len()));
        for candidate in candidates
            .iter()
            .filter(|candidate| contains_query_signal(self, candidate))
        {
            matches.push(LazyLiteralMatch {
                key: candidate.key,
                score: candidate.score,
                relevance: LiteralRelevance::new(self, &candidate.fields),
            });
        }
        matches
            .into_sorted()
            .into_iter()
            .map(|matched| matched.relevance.materialize(matched.key, matched.score))
            .collect()
    }

    /// Lazy deterministic retrieval followed by the same bounded learned rerank
    /// used by [`select_top_k_with_model`].
    pub fn search_top_k_prepared_with_model<'a>(
        &self,
        candidates: &[PreparedMessageCandidate<'a>],
        limit: usize,
        model: &FieldRankModel,
        rerank_limit: usize,
    ) -> Vec<MessageMatch<'a>> {
        if limit == 0 {
            return Vec::new();
        }
        if !model.is_active() || rerank_limit == 0 {
            return self.search_top_k_prepared(candidates, limit);
        }
        let mut matches = self.search_top_k_prepared(candidates, limit.max(rerank_limit));
        rerank_sorted_matches(&mut matches, model, rerank_limit);
        matches.truncate(limit);
        matches
    }

    fn rank_from_parts<'a>(
        &self,
        key: &'a str,
        score: Score,
        fields: &[PreparedMessageField],
    ) -> Option<MessageMatch<'a>> {
        let relevance = LiteralRelevance::new(self, fields);
        (relevance.coverage().distinct_coverage != 0).then(|| relevance.materialize(key, score))
    }
}

struct LazyLiteralMatch<'a, 'q> {
    key: &'a str,
    score: Score,
    relevance: LiteralRelevance<'q>,
}

impl Ord for LazyLiteralMatch<'_, '_> {
    fn cmp(&self, other: &Self) -> Ordering {
        compare_relevance(&self.relevance, &other.relevance)
            .then_with(|| compare_score_desc(self.score, other.score))
            .then_with(|| compare_key_asc(self.key, other.key))
    }
}

impl PartialOrd for LazyLiteralMatch<'_, '_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for LazyLiteralMatch<'_, '_> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Eq for LazyLiteralMatch<'_, '_> {}

trait LiteralRank {
    fn whole_phrase(&self) -> bool;
    fn subphrases(&self) -> &[usize];
    fn coverage(&self) -> &CoverageRank;
    fn field_quality(&self) -> u8;
    fn span(&self) -> (usize, usize);
    fn phrase_occurrences(&self) -> &FieldOccurrenceRank;
    fn token_occurrences(&self) -> &FieldOccurrenceRank;
    fn density(&self) -> DensityRank;
}

struct LiteralRelevance<'a> {
    query: &'a MessageQuery,
    fields: &'a [PreparedMessageField],
    phrase: OnceCell<bool>,
    subphrases: OnceCell<Vec<usize>>,
    coverage: OnceCell<CoverageRank>,
    field_quality: OnceCell<u8>,
    span: OnceCell<(usize, usize)>,
    phrase_occurrences: OnceCell<FieldOccurrenceRank>,
    token_occurrences: OnceCell<FieldOccurrenceRank>,
    density: OnceCell<DensityRank>,
}

impl<'a> LiteralRelevance<'a> {
    fn new(query: &'a MessageQuery, fields: &'a [PreparedMessageField]) -> Self {
        Self {
            query,
            fields,
            phrase: OnceCell::new(),
            subphrases: OnceCell::new(),
            coverage: OnceCell::new(),
            field_quality: OnceCell::new(),
            span: OnceCell::new(),
            phrase_occurrences: OnceCell::new(),
            token_occurrences: OnceCell::new(),
            density: OnceCell::new(),
        }
    }

    fn field_matches(&self, field: &PreparedMessageField) -> bool {
        field.tokens.iter().any(|candidate| {
            self.query
                .distinct_tokens
                .iter()
                .any(|query| literal_token_match(query, candidate))
        })
    }

    fn materialize<'k>(&self, key: &'k str, score: Score) -> MessageMatch<'k> {
        let (order_distance, total_gap) = self.span();
        MessageMatch {
            key,
            score,
            query_token_count: self.query.tokens.len(),
            exact_whole_phrase_present: self.whole_phrase(),
            exact_subphrase_counts_vector: self.subphrases().to_vec(),
            exact_weighted_positional_coverage_vector: self.coverage().clone(),
            exact_field_quality: self.field_quality(),
            order_distance,
            total_gap,
            exact_phrase_occurrence_rank: self.phrase_occurrences().clone(),
            exact_token_occurrence_rank: self.token_occurrences().clone(),
            density: self.density(),
        }
    }
}

impl LiteralRank for LiteralRelevance<'_> {
    fn whole_phrase(&self) -> bool {
        *self.phrase.get_or_init(|| {
            exact_phrase_present_by(
                &self.query.tokens,
                self.fields.iter().map(|field| field.tokens.as_slice()),
                literal_token_match,
            )
        })
    }

    fn subphrases(&self) -> &[usize] {
        self.subphrases.get_or_init(|| {
            exact_subphrase_counts_vector_by(
                &self.query.tokens,
                self.fields.iter().map(|field| field.tokens.as_slice()),
                literal_token_match,
            )
        })
    }

    fn coverage(&self) -> &CoverageRank {
        self.coverage.get_or_init(|| {
            CoverageRank::new(
                position_vector(
                    self.query.distinct_tokens.iter().map(|query| {
                        self.fields
                            .iter()
                            .flat_map(|field| field.tokens.iter())
                            .filter_map(|candidate| exact_token_position(query, candidate))
                            .min()
                    }),
                    4,
                ),
                &[4, 2, 2, 1],
            )
        })
    }

    fn field_quality(&self) -> u8 {
        *self.field_quality.get_or_init(|| {
            self.fields
                .iter()
                .filter(|field| self.field_matches(field))
                .map(|field| field.priority)
                .min()
                .unwrap_or(u8::MAX)
        })
    }

    fn span(&self) -> (usize, usize) {
        *self.span.get_or_init(|| {
            ordered_token_span(
                &self.query.tokens,
                self.fields.iter().map(|field| field.tokens.as_slice()),
                literal_token_match,
            )
        })
    }

    fn phrase_occurrences(&self) -> &FieldOccurrenceRank {
        self.phrase_occurrences.get_or_init(|| {
            FieldOccurrenceRank::from_entries(self.fields.iter().map(|field| {
                (
                    field.priority,
                    exact_phrase_occurrence_count_by(
                        &self.query.tokens,
                        std::iter::once(field.tokens.as_slice()),
                        literal_token_match,
                    ),
                )
            }))
        })
    }

    fn token_occurrences(&self) -> &FieldOccurrenceRank {
        self.token_occurrences.get_or_init(|| {
            FieldOccurrenceRank::from_entries(self.fields.iter().map(|field| {
                (
                    field.priority,
                    field
                        .tokens
                        .iter()
                        .map(|candidate| {
                            self.query
                                .distinct_tokens
                                .iter()
                                .filter(|query| literal_token_match(query, candidate))
                                .count()
                        })
                        .sum(),
                )
            }))
        })
    }

    fn density(&self) -> DensityRank {
        *self.density.get_or_init(|| {
            DensityRank::new(
                self.coverage().distinct_coverage,
                self.fields
                    .iter()
                    .filter(|field| self.field_matches(field))
                    .map(|field| field.tokens.len())
                    .min()
                    .unwrap_or(usize::MAX),
            )
        })
    }
}

impl LiteralRank for MessageMatch<'_> {
    fn whole_phrase(&self) -> bool {
        self.exact_whole_phrase_present
    }
    fn subphrases(&self) -> &[usize] {
        &self.exact_subphrase_counts_vector
    }
    fn coverage(&self) -> &CoverageRank {
        &self.exact_weighted_positional_coverage_vector
    }
    fn field_quality(&self) -> u8 {
        self.exact_field_quality
    }
    fn span(&self) -> (usize, usize) {
        (self.order_distance, self.total_gap)
    }
    fn phrase_occurrences(&self) -> &FieldOccurrenceRank {
        &self.exact_phrase_occurrence_rank
    }
    fn token_occurrences(&self) -> &FieldOccurrenceRank {
        &self.exact_token_occurrence_rank
    }
    fn density(&self) -> DensityRank {
        self.density
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

/// Sort using the deterministic exact matcher, then rerank only the leading
/// `rerank_limit` candidates with a caller-trained field model.
pub fn sort_matches_with_model(
    matches: &mut [MessageMatch<'_>],
    model: &FieldRankModel,
    rerank_limit: usize,
) {
    sort_matches(matches);
    rerank_sorted_matches(matches, model, rerank_limit);
}

fn rerank_sorted_matches(
    matches: &mut [MessageMatch<'_>],
    model: &FieldRankModel,
    rerank_limit: usize,
) {
    rerank_sorted_prefix(matches, model, rerank_limit, MessageMatch::rank_features);
}

/// Retain the best candidates after deterministic retrieval and learned
/// reranking.  The model is deliberately limited to the rerank prefix so a
/// large database never requires full-corpus expensive feature work.
pub fn select_top_k_with_model(
    matches: &mut Vec<MessageMatch<'_>>,
    limit: usize,
    model: &FieldRankModel,
    rerank_limit: usize,
) {
    if limit == 0 {
        matches.clear();
        return;
    }
    if !model.is_active() || rerank_limit == 0 {
        select_top_k(matches, limit);
        return;
    }

    let pool_limit = matches.len().min(limit.max(rerank_limit));
    select_top_k(matches, pool_limit);
    rerank_sorted_matches(matches, model, rerank_limit.min(pool_limit));
    matches.truncate(limit);
}

fn compare_matches(left: &MessageMatch<'_>, right: &MessageMatch<'_>) -> Ordering {
    compare_relevance(left, right)
        .then_with(|| compare_score_desc(left.score, right.score))
        .then_with(|| compare_key_asc(left.key, right.key))
}

fn compare_relevance(left: &impl LiteralRank, right: &impl LiteralRank) -> Ordering {
    Reverse(left.whole_phrase())
        .cmp(&Reverse(right.whole_phrase()))
        .then_with(|| Reverse(left.subphrases()).cmp(&Reverse(right.subphrases())))
        .then_with(|| Reverse(left.coverage()).cmp(&Reverse(right.coverage())))
        .then_with(|| left.field_quality().cmp(&right.field_quality()))
        .then_with(|| left.span().cmp(&right.span()))
        .then_with(|| Reverse(left.phrase_occurrences()).cmp(&Reverse(right.phrase_occurrences())))
        .then_with(|| Reverse(left.token_occurrences()).cmp(&Reverse(right.token_occurrences())))
        .then_with(|| left.density().cmp(&right.density()))
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
    #[test]
    fn lazy_literal_selection_matches_eager_selection_with_and_without_model() {
        let texts = [
            "very very good",
            "very good",
            "good very",
            "very",
            "good",
            "server error log",
            "server log",
            "xserver",
            "unrelated",
        ];
        let prepared: Vec<_> = texts
            .iter()
            .enumerate()
            .map(|(index, text)| {
                super::MessageCandidate {
                    key: text,
                    score: (index % 3) as f64,
                    fields: &[super::MessageField {
                        priority: (index % 2) as u8,
                        value: text,
                    }],
                }
                .prepare()
            })
            .collect();
        let mut weights = [0.0; crate::fields::model::FIELD_RANK_FEATURE_COUNT];
        weights[super::feature::CALLER_SCORE] = 1.0;
        for text in [
            "very very good",
            "very good",
            "server error log",
            "good",
            "log server",
            "absent",
        ] {
            let query = super::MessageQuery::new(text).unwrap();
            let eager: Vec<_> = prepared
                .iter()
                .filter_map(|candidate| query.search_rank_prepared(candidate))
                .collect();
            for model in [
                super::FieldRankModel::default(),
                super::FieldRankModel::new(0.0, weights),
            ] {
                for rerank_limit in [0, 1, 3, 50] {
                    let mut full = eager.clone();
                    super::sort_matches_with_model(&mut full, &model, rerank_limit);
                    for limit in 0..=texts.len() + 1 {
                        let top = query.search_top_k_prepared_with_model(
                            &prepared,
                            limit,
                            &model,
                            rerank_limit,
                        );
                        assert_eq!(
                            top.iter()
                                .map(|m| (m.key, m.rank_features()))
                                .collect::<Vec<_>>(),
                            full.iter()
                                .take(limit)
                                .map(|m| (m.key, m.rank_features()))
                                .collect::<Vec<_>>()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn lazy_literal_comparison_does_not_evaluate_losing_subordinate_stages() {
        let query = super::MessageQuery::new("server error").unwrap();
        let full = super::MessageCandidate {
            key: "full",
            fields: &[super::MessageField {
                priority: 0,
                value: "server error",
            }],
            score: 0.0,
        }
        .prepare();
        let partial = super::MessageCandidate {
            key: "partial",
            fields: &[super::MessageField {
                priority: 0,
                value: "server",
            }],
            score: 100.0,
        }
        .prepare();
        let left = super::LiteralRelevance::new(&query, &full.fields);
        let right = super::LiteralRelevance::new(&query, &partial.fields);
        assert!(super::compare_relevance(&left, &right).is_lt());
        for relevance in [&left, &right] {
            assert!(relevance.phrase.get().is_some());
            assert!(relevance.subphrases.get().is_none());
            assert!(relevance.coverage.get().is_none());
            assert!(relevance.span.get().is_none());
            assert!(relevance.phrase_occurrences.get().is_none());
            assert!(relevance.token_occurrences.get().is_none());
            assert!(relevance.density.get().is_none());
        }
    }
    #[test]
    fn repeated_query_words_are_preserved_for_phrases_but_not_coverage() {
        let query = super::MessageQuery::new("very very good").unwrap();
        let full = [super::MessageField {
            priority: 0,
            value: "very very good",
        }];
        let shortened = [super::MessageField {
            priority: 0,
            value: "very good",
        }];
        let full = query
            .search_rank(super::MessageCandidate {
                key: "full",
                fields: &full,
                score: 0.0,
            })
            .unwrap();
        let shortened = query
            .search_rank(super::MessageCandidate {
                key: "shortened",
                fields: &shortened,
                score: 100.0,
            })
            .unwrap();
        assert_eq!(full.query_token_count, 3);
        assert!(full.exact_whole_phrase_present);
        assert!(!shortened.exact_whole_phrase_present);
        assert_eq!(
            full.exact_weighted_positional_coverage_vector
                .distinct_coverage,
            2
        );
        assert_eq!(
            shortened
                .exact_weighted_positional_coverage_vector
                .distinct_coverage,
            2
        );
        assert!(super::compare_matches(&full, &shortened).is_lt());
    }
    use super::{
        MessageCandidate, MessageField, MessageQuery, PreparedMessageCandidate,
        contains_query_signal, sort_matches, sort_matches_with_model,
    };
    use crate::fields::FieldRankModel;

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
    fn message_features_have_stable_shape_and_exact_direction() {
        let candidate = prepare("message", "server error", 3.0);
        let query = MessageQuery::new("server error").unwrap();
        let matched = query.search_rank_prepared(&candidate).unwrap();
        let features = matched.rank_features();

        assert_eq!(
            features.values().len(),
            crate::fields::FIELD_RANK_FEATURE_COUNT
        );
        assert_eq!(
            features.get(crate::fields::model::feature::EXACT_PHRASE_PRESENT),
            Some(1.0)
        );
        assert!(
            features
                .get(crate::fields::model::feature::CALLER_SCORE)
                .unwrap()
                > 0.0
        );
    }

    #[test]
    fn message_model_can_rerank_the_retrieval_prefix() {
        let candidates = [
            prepare("low-score", "server error", 1.0),
            prepare("high-score", "server error", 10.0),
        ];
        let query = MessageQuery::new("server error").unwrap();
        let mut matches = candidates
            .iter()
            .filter_map(|candidate| query.search_rank_prepared(candidate))
            .collect::<Vec<_>>();
        let mut weights = [0.0; crate::fields::FIELD_RANK_FEATURE_COUNT];
        weights[crate::fields::model::feature::CALLER_SCORE] = 10.0;
        let model = FieldRankModel::new(0.0, weights);

        sort_matches_with_model(&mut matches, &model, 2);
        assert_eq!(matches[0].key, "high-score");
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
