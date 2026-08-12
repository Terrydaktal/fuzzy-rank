use crate::core::{
    CoverageRank, DamerauScratch, DensityRank, FieldOccurrenceRank, InsertionVector,
    RankedTokenVector, TypoCostVector, bounded_damerau_levenshtein,
    bounded_damerau_levenshtein_chars_with_scratch, compare_key_asc, compare_score_desc,
    exact_phrase_occurrence_count_by, exact_phrase_present_by, exact_subphrase_counts_vector_by,
    exact_token_position, keyboard_substitution_cost, max_typos, ordered_token_span,
    position_vector, ratio_milli,
};
use crate::debug::{DebugReport, DebugStage};
use crate::ranking::{
    AbbreviationRank, DistanceRank, SearchRank, StructuralRank, compare_search_results,
};
use crate::text::{normalize_compact_alnum, split_search_tokens, to_lowercase, token_has_signal};
use crate::token_match::fragmented_token_match;
use std::borrow::Cow;
use std::cell::OnceCell;
use std::cmp::{Ordering, Reverse};

pub type Score = f64;

#[derive(Clone, Copy, Debug)]
pub struct SearchField<'a> {
    pub priority: u8,
    pub value: &'a str,
}

#[derive(Clone, Copy, Debug)]
pub struct MetadataCandidate<'a> {
    pub key: &'a str,
    pub fields: &'a [SearchField<'a>],
    pub score: Score,
}

/// A metadata field prepared once for repeated searches.  The value is
/// lower-cased and tokenised at construction time so ranking and sorting do
/// not redo that work for every candidate.
#[derive(Clone, Debug)]
pub struct PreparedMetadataField {
    pub priority: u8,
    pub value: String,
    pub tokens: Vec<String>,
}

impl PreparedMetadataField {
    pub fn new(priority: u8, value: &str) -> Option<Self> {
        let value = value.trim();
        let lowered = to_lowercase(value);
        let tokens = split_search_tokens(&lowered)
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        (!tokens.is_empty()).then_some(Self {
            priority,
            value: lowered,
            tokens,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PreparedMetadataCandidate<'a> {
    pub key: &'a str,
    pub fields: &'a [PreparedMetadataField],
    pub score: Score,
}

pub fn prepare_fields(fields: &[SearchField<'_>]) -> Vec<PreparedMetadataField> {
    fields
        .iter()
        .filter_map(|field| PreparedMetadataField::new(field.priority, field.value))
        .collect()
}

#[derive(Clone, Copy, Debug)]
pub struct MetadataMatch<'a> {
    pub candidate: MetadataCandidate<'a>,
    pub rank: &'a SearchRank,
}

#[derive(Clone, Debug)]
pub struct MetadataTermExplanation<'a> {
    pub query_term: String,
    pub rank: Option<SearchRank>,
    pub field: Option<SearchField<'a>>,
}

#[derive(Clone, Debug)]
pub struct MetadataExplanation<'a> {
    pub candidate: MetadataCandidate<'a>,
    pub overall_rank: SearchRank,
    pub overall_field: SearchField<'a>,
    pub term_explanations: Vec<MetadataTermExplanation<'a>>,
}

pub type Field<'a> = SearchField<'a>;
pub type Candidate<'a> = MetadataCandidate<'a>;
pub type Query = MetadataQuery;
pub type PreparedField = PreparedMetadataField;
pub type PreparedCandidate<'a> = PreparedMetadataCandidate<'a>;
pub type Match<'a> = MetadataMatch<'a>;
pub type TermExplanation<'a> = MetadataTermExplanation<'a>;
pub type Explanation<'a> = MetadataExplanation<'a>;

pub struct MetadataQuery {
    query: String,
    query_tokens: Vec<String>,
    allow_typo: bool,
}

trait MetadataFieldView {
    fn priority(&self) -> u8;
    fn lowered_value(&self) -> Cow<'_, str>;
}

impl MetadataFieldView for SearchField<'_> {
    fn priority(&self) -> u8 {
        self.priority
    }

    fn lowered_value(&self) -> Cow<'_, str> {
        Cow::Owned(to_lowercase(self.value.trim()))
    }
}

impl MetadataFieldView for PreparedMetadataField {
    fn priority(&self) -> u8 {
        self.priority
    }

    fn lowered_value(&self) -> Cow<'_, str> {
        Cow::Borrowed(&self.value)
    }
}

impl MetadataQuery {
    pub fn new(query: &str) -> Option<Self> {
        let query = normalize_metadata_query(query)?;
        let query_tokens = split_search_tokens(&query)
            .into_iter()
            .map(str::to_string)
            .collect();
        Some(Self {
            query,
            query_tokens,
            allow_typo: false,
        })
    }

    pub fn with_typo_fallback(mut self, allow_typo: bool) -> Self {
        self.allow_typo = allow_typo;
        self
    }

    pub fn search_rank(&self, candidate: MetadataCandidate<'_>) -> Option<SearchRank> {
        self.search_rank_fields(candidate.fields)
    }

    pub fn search_rank_prepared(
        &self,
        candidate: PreparedMetadataCandidate<'_>,
    ) -> Option<SearchRank> {
        self.search_rank_fields(candidate.fields)
    }

    fn search_rank_fields<F: MetadataFieldView>(&self, fields: &[F]) -> Option<SearchRank> {
        if self.query_tokens.len() <= 1 {
            return self.rank_single_query(self.query.as_str(), fields);
        }

        self.query_tokens
            .iter()
            .filter_map(|query| self.rank_single_query(query, fields))
            .min()
    }

    fn rank_single_query<F: MetadataFieldView>(
        &self,
        query: &str,
        fields: &[F],
    ) -> Option<SearchRank> {
        if self.allow_typo {
            best_structural_rank(query, fields)
                .map(SearchRank::Structural)
                .or_else(|| best_typo_rank(query, fields).map(SearchRank::Typo))
        } else {
            best_structural_rank(query, fields)
                .map(SearchRank::Structural)
                .or_else(|| best_abbreviation_rank(query, fields).map(SearchRank::Abbreviation))
                .or_else(|| best_fuzzy_rank(query, fields).map(SearchRank::Fuzzy))
        }
    }

    pub fn compare_candidates(
        &self,
        left_candidate: MetadataCandidate<'_>,
        left_rank: &SearchRank,
        right_candidate: MetadataCandidate<'_>,
        right_rank: &SearchRank,
    ) -> Ordering {
        let left_relevance =
            LazyRelevance::new(&self.query_tokens, self.allow_typo, left_candidate.fields);
        let right_relevance =
            LazyRelevance::new(&self.query_tokens, self.allow_typo, right_candidate.fields);

        left_relevance.compare(&right_relevance).then_with(|| {
            compare_score_desc(left_candidate.score, right_candidate.score)
                .then_with(|| left_rank.cmp(right_rank))
                .then_with(|| compare_key_asc(left_candidate.key, right_candidate.key))
        })
    }

    pub fn sort_matches(&self, matches: &mut [(MetadataCandidate<'_>, SearchRank)]) {
        let mut lazy_matches: Vec<_> = matches
            .iter()
            .map(|(cand, rank)| {
                (
                    *cand,
                    rank.clone(),
                    LazyRelevance::new(&self.query_tokens, self.allow_typo, cand.fields),
                )
            })
            .collect();

        lazy_matches.sort_unstable_by(
            |(left_cand, left_rank, left_rel), (right_cand, right_rank, right_rel)| {
                left_rel.compare(right_rel).then_with(|| {
                    compare_score_desc(left_cand.score, right_cand.score)
                        .then_with(|| left_rank.cmp(right_rank))
                        .then_with(|| compare_key_asc(left_cand.key, right_cand.key))
                })
            },
        );

        for (i, (cand, rank, _)) in lazy_matches.into_iter().enumerate() {
            matches[i] = (cand, rank);
        }
    }

    pub fn sort_matches_prepared(
        &self,
        matches: &mut [(PreparedMetadataCandidate<'_>, SearchRank)],
    ) {
        let mut lazy_matches: Vec<_> = matches
            .iter()
            .map(|(cand, rank)| {
                (
                    *cand,
                    rank.clone(),
                    LazyRelevance::from_prepared(&self.query_tokens, self.allow_typo, cand.fields),
                )
            })
            .collect();

        lazy_matches.sort_unstable_by(
            |(left_cand, left_rank, left_rel), (right_cand, right_rank, right_rel)| {
                left_rel.compare(right_rel).then_with(|| {
                    compare_score_desc(left_cand.score, right_cand.score)
                        .then_with(|| left_rank.cmp(right_rank))
                        .then_with(|| compare_key_asc(left_cand.key, right_cand.key))
                })
            },
        );

        for (i, (cand, rank, _)) in lazy_matches.into_iter().enumerate() {
            matches[i] = (cand, rank);
        }
    }

    /// Sort prepared candidates while retaining caller-owned payloads.
    ///
    /// The payload is cloned only once per candidate; relevance itself is
    /// still prepared once and reused for the complete sort.
    pub fn sort_matches_prepared_with<T: Clone>(
        &self,
        matches: &mut [(PreparedMetadataCandidate<'_>, SearchRank, T)],
    ) {
        let mut lazy_matches: Vec<_> = matches
            .iter()
            .map(|(candidate, rank, payload)| {
                (
                    *candidate,
                    rank.clone(),
                    payload.clone(),
                    LazyRelevance::from_prepared(
                        &self.query_tokens,
                        self.allow_typo,
                        candidate.fields,
                    ),
                )
            })
            .collect();

        lazy_matches.sort_unstable_by(
            |(left_candidate, left_rank, _, left_relevance),
             (right_candidate, right_rank, _, right_relevance)| {
                left_relevance.compare(right_relevance).then_with(|| {
                    compare_score_desc(left_candidate.score, right_candidate.score)
                        .then_with(|| left_rank.cmp(right_rank))
                        .then_with(|| compare_key_asc(left_candidate.key, right_candidate.key))
                })
            },
        );

        for (index, (candidate, rank, payload, _)) in lazy_matches.into_iter().enumerate() {
            matches[index] = (candidate, rank, payload);
        }
    }

    /// Explain every metadata comparator stage for one candidate.
    ///
    /// Unlike normal sorting, this deliberately evaluates every stage so the
    /// debug CLI can show values that a lazy comparison would not need for a
    /// particular candidate.
    pub fn debug_report<'a>(&self, candidate: MetadataCandidate<'a>) -> DebugReport {
        let relevance = LazyRelevance::new(&self.query_tokens, self.allow_typo, candidate.fields);
        let rank = self.search_rank(candidate);
        let stages = vec![
            DebugStage::new(
                "CORE.EXACT_PHRASE_PRESENT",
                format!("{}", relevance.get_exact_whole_phrase_present()),
            ),
            DebugStage::new(
                "CORE.EXACT_SUBPHRASE_COUNTS_VECTOR",
                format!("{:?}", relevance.get_exact_subphrase_counts_vector()),
            ),
            DebugStage::new(
                "CORE.EXACT_WEIGHTED_POSITIONAL_COVERAGE_VECTOR",
                Self::format_coverage_rank(
                    relevance.get_exact_weighted_positional_coverage_vector(),
                ),
            ),
            DebugStage::new(
                "CORE.EXACT_FIELD_QUALITY",
                format!("{}", relevance.get_exact_field_quality()),
            ),
            DebugStage::new(
                "METADATA.FUZZY_WHOLE_PHRASE",
                format!("{}", relevance.get_fuzzy_whole_phrase()),
            ),
            DebugStage::new(
                "METADATA.FUZZY_SUBPHRASE_COUNTS_VECTOR",
                format!("{:?}", relevance.get_fuzzy_subphrase_counts_vector()),
            ),
            DebugStage::new(
                "METADATA.FUZZY_WEIGHTED_POSITIONAL_COVERAGE_VECTOR",
                format!(
                    "{:?}",
                    relevance.get_fuzzy_weighted_positional_coverage_vector()
                ),
            ),
            DebugStage::new(
                "METADATA.FUZZY_FIELD_QUALITY",
                format!("{}", relevance.get_fuzzy_field_quality()),
            ),
            DebugStage::new(
                "CORE.TYPO_COST_VECTOR",
                format_ranked_vector(relevance.get_typo_cost_vector()),
            ),
            DebugStage::new(
                "CORE.INSERTION_VECTOR",
                format_ranked_vector(relevance.get_insertion_vector()),
            ),
            DebugStage::new(
                "CORE.ORDER_VIOLATION_DISTANCE",
                format!("{}", relevance.get_order_distance()),
            ),
            DebugStage::new(
                "CORE.ORDERED_SPAN.GAP",
                format!("{}", relevance.get_total_gap()),
            ),
            DebugStage::new(
                "CORE.EXACT_PHRASE_OCCURRENCE_RANK",
                format!("{:?}", relevance.get_exact_phrase_occurrence_rank()),
            ),
            DebugStage::new(
                "CORE.EXACT_TOKEN_OCCURRENCE_RANK",
                format!("{:?}", relevance.get_exact_token_occurrence_rank()),
            ),
            DebugStage::new(
                "CORE.DENSITY",
                format!("{:?}", relevance.get_density_rank()),
            ),
            DebugStage::new("CORE.SCORE_DESC", format!("{}", candidate.score)),
            DebugStage::new("RANKING.SEARCH_RANK", format!("{:?}", rank.as_ref())),
            DebugStage::new("CORE.KEY_ASC", candidate.key),
        ];

        match rank {
            Some(rank) => DebugReport::matched(format!("{:?}", rank), stages),
            None => DebugReport::no_match("no metadata match", stages),
        }
    }

    fn format_coverage_rank(rank: &CoverageRank) -> String {
        format!(
            "weighted_quality: {}\ndistinct_coverage: {}\nclass_counts: {:?}\nclasses: [whole, prefix, suffix, substring]\nweights: [4, 2, 2, 1]",
            rank.weighted_quality, rank.distinct_coverage, rank.class_counts
        )
    }

    pub fn search_rank_with_highlights(
        &self,
        candidate: MetadataCandidate<'_>,
    ) -> Option<(SearchRank, Vec<MatchedFieldHighlight>)> {
        let rank = self.search_rank(candidate)?;
        let highlights = self.highlights_for_rank(candidate, &rank);
        Some((rank, highlights))
    }

    /// Compute visual match ranges for a rank that has already been found.
    /// This avoids running metadata ranking a second time when callers first
    /// rank/sort a complete result set and only then render highlights.
    pub fn highlights_for_rank(
        &self,
        candidate: MetadataCandidate<'_>,
        _rank: &SearchRank,
    ) -> Vec<MatchedFieldHighlight> {
        let mut field_highlights = Vec::new();

        for (field_index, field) in candidate.fields.iter().enumerate() {
            let value = field.value;
            if value.trim().is_empty() {
                continue;
            }

            let char_count = value.chars().count();
            let mut char_status = vec![(None, usize::MAX); char_count];

            // 1. Exact phrase matches (if query has multiple words)
            if self.query_tokens.len() > 1 {
                let normalized_field = value.to_lowercase();
                let normalized_query = self.query.to_lowercase();
                let mut search_idx = 0;
                while let Some(start_byte) = normalized_field[search_idx..].find(&normalized_query)
                {
                    let start_byte_abs = search_idx + start_byte;
                    let end_byte_abs = start_byte_abs + normalized_query.len();
                    let start_char = value[..start_byte_abs].chars().count();
                    let end_char = value[..end_byte_abs].chars().count();
                    for status in char_status.iter_mut().take(end_char).skip(start_char) {
                        *status = (Some(true), 0);
                    }
                    search_idx = start_byte_abs + 1.max(normalized_query.len());
                }
            }

            // 2. Exact token matches and Typo matches
            // Always highlight the entire matched token (not just the matched slice)
            let tokens_with_ranges = alnum_tokens_with_char_ranges(value);
            for query_token in &self.query_tokens {
                let query_token_lower = query_token.to_lowercase();

                for token in &tokens_with_ranges {
                    let token_len = token.value.chars().count();
                    let token_end = token.start + token_len;
                    if literal_token_match(&query_token_lower, &token.value) {
                        // Exact literal match: highlight the containing token.
                        for status in char_status.iter_mut().take(token_end).skip(token.start) {
                            *status = (Some(true), 0);
                        }
                    } else if let Some((_s_char, _e_char, severity)) =
                        typo_match_token_slice(&token.value, &query_token_lower)
                    {
                        // Typo match — highlight whole token
                        for status in char_status.iter_mut().take(token_end).skip(token.start) {
                            if status.0.is_none()
                                || (status.0 == Some(false) && severity.core_cost < status.1)
                            {
                                *status = (Some(false), severity.core_cost);
                            }
                        }
                    }
                }
            }

            // Convert character statuses back to byte ranges
            let mut ranges = Vec::new();
            let mut current_segment: Option<(usize, bool, usize)> = None;

            for (char_idx, status) in char_status.iter().copied().enumerate() {
                if let Some(is_exact) = status.0 {
                    if let Some((start_char, seg_exact, min_cost)) = current_segment {
                        if seg_exact == is_exact {
                            current_segment = Some((start_char, seg_exact, min_cost.min(status.1)));
                        } else {
                            let (byte_start, byte_end) =
                                char_indices_to_byte_range(value, start_char, char_idx);
                            ranges.push((byte_start, byte_end, seg_exact, min_cost));
                            current_segment = Some((char_idx, is_exact, status.1));
                        }
                    } else {
                        current_segment = Some((char_idx, is_exact, status.1));
                    }
                } else if let Some((start_char, seg_exact, min_cost)) = current_segment {
                    let (byte_start, byte_end) =
                        char_indices_to_byte_range(value, start_char, char_idx);
                    ranges.push((byte_start, byte_end, seg_exact, min_cost));
                    current_segment = None;
                }
            }
            if let Some((start_char, seg_exact, min_cost)) = current_segment {
                let (byte_start, byte_end) =
                    char_indices_to_byte_range(value, start_char, char_count);
                ranges.push((byte_start, byte_end, seg_exact, min_cost));
            }

            if !ranges.is_empty() {
                let mut best_focus = None;
                for &(start, end, is_exact, cost) in &ranges {
                    let is_better = match (best_focus, is_exact) {
                        (None, _) => true,
                        (Some((_, _, best_exact, best_cost)), new_exact) => {
                            if best_exact != new_exact {
                                new_exact
                            } else {
                                cost < best_cost
                            }
                        }
                    };
                    if is_better {
                        best_focus = Some((start, end, is_exact, cost));
                    }
                }
                let focus_range = best_focus.map(|(s, e, _, _)| (s, e));
                let output_ranges = ranges.into_iter().map(|(s, e, ex, _)| (s, e, ex)).collect();

                field_highlights.push(MatchedFieldHighlight {
                    field_index,
                    ranges: output_ranges,
                    focus_range,
                });
            }
        }

        field_highlights
    }

    pub fn explain_candidate<'a>(
        &self,
        candidate: MetadataCandidate<'a>,
    ) -> Option<MetadataExplanation<'a>> {
        let overall_rank = self.search_rank(candidate)?;
        let overall_field = *candidate
            .fields
            .get(overall_rank.provenance().field_index)?;
        let query_terms = split_search_tokens(&self.query);
        let term_explanations = query_terms
            .into_iter()
            .map(|query_term| {
                let rank = Self {
                    query: query_term.to_string(),
                    query_tokens: vec![query_term.to_string()],
                    allow_typo: self.allow_typo,
                }
                .search_rank(candidate);
                let field = rank
                    .as_ref()
                    .and_then(|rank| candidate.fields.get(rank.provenance().field_index))
                    .copied();
                MetadataTermExplanation {
                    query_term: query_term.to_string(),
                    rank,
                    field,
                }
            })
            .collect();
        Some(MetadataExplanation {
            candidate,
            overall_rank,
            overall_field,
            term_explanations,
        })
    }
}

fn format_ranked_vector(vector: &RankedTokenVector) -> String {
    format!(
        "aligned={:?}, worst_first={:?}",
        vector.aligned(),
        vector.worst_first()
    )
}

struct LazyRelevance<'a> {
    query_tokens: &'a [String],
    allow_typo: bool,
    fields: Cow<'a, [PreparedMetadataField]>,

    exact_whole_phrase_present: OnceCell<bool>,
    exact_subphrase_counts_vector: OnceCell<Vec<usize>>,
    exact_weighted_positional_coverage_vector: OnceCell<CoverageRank>,
    fuzzy_weighted_positional_coverage_vector: OnceCell<CoverageRank>,
    fuzzy_whole_phrase: OnceCell<bool>,
    fuzzy_subphrase_counts_vector: OnceCell<Vec<usize>>,
    fuzzy_token_matches: OnceCell<TokenMatchMatrix>,
    fuzzy_best_matches: OnceCell<Vec<TokenMatch>>,
    exact_field_quality: OnceCell<u8>,
    fuzzy_field_quality: OnceCell<u8>,
    exact_phrase_occurrence_rank: OnceCell<FieldOccurrenceRank>,
    exact_token_occurrence_rank: OnceCell<FieldOccurrenceRank>,
    scattered_stats: OnceCell<(usize, usize)>,
    typo_cost_vector: OnceCell<TypoCostVector>,
    insertion_vector: OnceCell<InsertionVector>,
    density_rank: OnceCell<DensityRank>,
}

impl<'a> LazyRelevance<'a> {
    fn new(query_tokens: &'a [String], allow_typo: bool, fields: &[SearchField<'_>]) -> Self {
        Self::from_owned_fields(query_tokens, allow_typo, prepare_fields(fields))
    }

    fn from_prepared(
        query_tokens: &'a [String],
        allow_typo: bool,
        fields: &'a [PreparedMetadataField],
    ) -> Self {
        Self {
            query_tokens,
            allow_typo,
            fields: Cow::Borrowed(fields),
            exact_whole_phrase_present: OnceCell::new(),
            exact_subphrase_counts_vector: OnceCell::new(),
            exact_weighted_positional_coverage_vector: OnceCell::new(),
            fuzzy_weighted_positional_coverage_vector: OnceCell::new(),
            fuzzy_whole_phrase: OnceCell::new(),
            fuzzy_subphrase_counts_vector: OnceCell::new(),
            fuzzy_token_matches: OnceCell::new(),
            fuzzy_best_matches: OnceCell::new(),
            exact_field_quality: OnceCell::new(),
            fuzzy_field_quality: OnceCell::new(),
            exact_phrase_occurrence_rank: OnceCell::new(),
            exact_token_occurrence_rank: OnceCell::new(),
            scattered_stats: OnceCell::new(),
            typo_cost_vector: OnceCell::new(),
            insertion_vector: OnceCell::new(),
            density_rank: OnceCell::new(),
        }
    }

    fn from_owned_fields(
        query_tokens: &'a [String],
        allow_typo: bool,
        fields: Vec<PreparedMetadataField>,
    ) -> Self {
        Self {
            query_tokens,
            allow_typo,
            fields: Cow::Owned(fields),
            exact_whole_phrase_present: OnceCell::new(),
            exact_subphrase_counts_vector: OnceCell::new(),
            exact_weighted_positional_coverage_vector: OnceCell::new(),
            fuzzy_weighted_positional_coverage_vector: OnceCell::new(),
            fuzzy_whole_phrase: OnceCell::new(),
            fuzzy_subphrase_counts_vector: OnceCell::new(),
            fuzzy_token_matches: OnceCell::new(),
            fuzzy_best_matches: OnceCell::new(),
            exact_field_quality: OnceCell::new(),
            fuzzy_field_quality: OnceCell::new(),
            exact_phrase_occurrence_rank: OnceCell::new(),
            exact_token_occurrence_rank: OnceCell::new(),
            scattered_stats: OnceCell::new(),
            typo_cost_vector: OnceCell::new(),
            insertion_vector: OnceCell::new(),
            density_rank: OnceCell::new(),
        }
    }

    fn get_exact_whole_phrase_present(&self) -> bool {
        *self.exact_whole_phrase_present.get_or_init(|| {
            if self.query_tokens.is_empty() || self.fields.is_empty() {
                false
            } else {
                exact_phrase_present_by(
                    self.query_tokens,
                    self.fields.iter().map(|field| field.tokens.as_slice()),
                    literal_token_match,
                )
            }
        })
    }

    fn get_exact_subphrase_counts_vector(&self) -> &[usize] {
        self.exact_subphrase_counts_vector.get_or_init(|| {
            if self.query_tokens.is_empty() || self.fields.is_empty() {
                Vec::new()
            } else {
                exact_subphrase_counts_vector_by(
                    self.query_tokens,
                    self.fields.iter().map(|field| field.tokens.as_slice()),
                    literal_token_match,
                )
            }
        })
    }

    fn get_exact_weighted_positional_coverage_vector(&self) -> &CoverageRank {
        self.exact_weighted_positional_coverage_vector
            .get_or_init(|| {
                CoverageRank::new(
                    position_vector(
                        self.query_tokens.iter().map(|query_token| {
                            best_exact_token_scope(query_token, &self.fields)
                                .map(|scope| scope as usize)
                        }),
                        4,
                    ),
                    &[4, 2, 2, 1],
                )
            })
    }

    fn get_fuzzy_weighted_positional_coverage_vector(&self) -> &CoverageRank {
        self.fuzzy_weighted_positional_coverage_vector
            .get_or_init(|| {
                let matches = self.get_fuzzy_best_matches();
                CoverageRank::new(metadata_fuzzy_coverage_vector(matches), &[4, 2, 2, 1, 1])
            })
    }

    fn get_fuzzy_whole_phrase(&self) -> bool {
        *self.fuzzy_whole_phrase.get_or_init(|| {
            if !self.allow_typo {
                false
            } else {
                fuzzy_whole_phrase_present_cached(
                    self.query_tokens,
                    &self.fields,
                    true,
                    self.get_fuzzy_token_matches(),
                )
            }
        })
    }

    fn get_fuzzy_subphrase_counts_vector(&self) -> &[usize] {
        if self.query_tokens.is_empty() || self.fields.is_empty() {
            return &[];
        }
        self.fuzzy_subphrase_counts_vector.get_or_init(|| {
            fuzzy_subphrase_counts_vector_cached(
                self.query_tokens,
                &self.fields,
                self.get_fuzzy_token_matches(),
            )
        })
    }

    fn get_typo_cost_vector(&self) -> &TypoCostVector {
        self.typo_cost_vector.get_or_init(|| {
            if self.query_tokens.is_empty() || self.fields.is_empty() {
                TypoCostVector::default()
            } else {
                let matches = self.get_fuzzy_best_matches();
                typo_cost_vector(matches)
            }
        })
    }

    fn get_insertion_vector(&self) -> &InsertionVector {
        self.insertion_vector.get_or_init(|| {
            if self.query_tokens.is_empty() || self.fields.is_empty() {
                InsertionVector::default()
            } else {
                let matches = self.get_fuzzy_best_matches();
                insertion_vector(matches)
            }
        })
    }

    fn get_exact_field_quality(&self) -> u8 {
        *self.exact_field_quality.get_or_init(|| {
            if self.query_tokens.is_empty() || self.fields.is_empty() {
                u8::MAX
            } else {
                best_exact_field_priority(self.query_tokens, &self.fields)
            }
        })
    }

    fn get_fuzzy_field_quality(&self) -> u8 {
        *self.fuzzy_field_quality.get_or_init(|| {
            self.get_fuzzy_best_matches()
                .iter()
                .filter(|matched| matched.fuzzy && !matched.exact)
                .map(|matched| matched.field_priority)
                .min()
                .unwrap_or(u8::MAX)
        })
    }

    fn get_order_distance(&self) -> usize {
        if self.query_tokens.is_empty() || self.fields.is_empty() {
            return usize::MAX;
        }
        self.scattered_stats
            .get_or_init(|| ordered_span_stats(self.query_tokens, &self.fields, self.allow_typo))
            .0
    }

    fn get_total_gap(&self) -> usize {
        if self.query_tokens.is_empty() || self.fields.is_empty() {
            return usize::MAX;
        }
        self.scattered_stats
            .get_or_init(|| ordered_span_stats(self.query_tokens, &self.fields, self.allow_typo))
            .1
    }

    fn get_exact_phrase_occurrence_rank(&self) -> &FieldOccurrenceRank {
        self.exact_phrase_occurrence_rank.get_or_init(|| {
            field_occurrence_rank(&self.fields, |field| {
                exact_phrase_occurrence_count_by(
                    self.query_tokens,
                    std::iter::once(field.tokens.as_slice()),
                    literal_token_match,
                )
            })
        })
    }

    fn get_exact_token_occurrence_rank(&self) -> &FieldOccurrenceRank {
        self.exact_token_occurrence_rank.get_or_init(|| {
            field_occurrence_rank(&self.fields, |field| {
                field
                    .tokens
                    .iter()
                    .map(|candidate| {
                        self.query_tokens
                            .iter()
                            .filter(|query| literal_token_match(query, candidate))
                            .count()
                    })
                    .sum()
            })
        })
    }

    fn get_density_rank(&self) -> DensityRank {
        *self.density_rank.get_or_init(|| {
            if self.query_tokens.is_empty() || self.fields.is_empty() {
                DensityRank::new(0, usize::MAX)
            } else {
                let matches = self.get_fuzzy_best_matches();
                density_stats(self.query_tokens, &self.fields, matches)
            }
        })
    }

    fn get_fuzzy_token_matches(&self) -> &TokenMatchMatrix {
        self.fuzzy_token_matches.get_or_init(|| {
            self.query_tokens
                .iter()
                .map(|query_token| {
                    self.fields
                        .iter()
                        .map(|field| {
                            field
                                .tokens
                                .iter()
                                .map(|candidate_token| {
                                    if let Some(scope) =
                                        literal_token_scope(query_token, candidate_token)
                                    {
                                        TokenMatch {
                                            exact: true,
                                            fuzzy: true,
                                            severity: TokenTypoSeverity::default(),
                                            scope: Some(scope),
                                            field_priority: field.priority,
                                        }
                                    } else {
                                        fuzzy_or_exact_token_match(candidate_token, query_token)
                                            .map_or_else(TokenMatch::default, |detail| TokenMatch {
                                                exact: false,
                                                fuzzy: true,
                                                severity: detail.severity,
                                                scope: Some(detail.scope),
                                                field_priority: field.priority,
                                            })
                                    }
                                })
                                .collect()
                        })
                        .collect()
                })
                .collect()
        })
    }

    fn get_fuzzy_best_matches(&self) -> &Vec<TokenMatch> {
        self.fuzzy_best_matches.get_or_init(|| {
            self.get_fuzzy_token_matches()
                .iter()
                .map(|field_matches| {
                    best_fuzzy_token_match_from_matrix(field_matches, self.allow_typo)
                })
                .collect()
        })
    }

    fn compare(&self, other: &Self) -> Ordering {
        other
            .get_exact_whole_phrase_present()
            .cmp(&self.get_exact_whole_phrase_present())
            .then_with(|| {
                other
                    .get_exact_subphrase_counts_vector()
                    .cmp(self.get_exact_subphrase_counts_vector())
            })
            .then_with(|| {
                other
                    .get_exact_weighted_positional_coverage_vector()
                    .cmp(self.get_exact_weighted_positional_coverage_vector())
            })
            .then_with(|| {
                self.get_exact_field_quality()
                    .cmp(&other.get_exact_field_quality())
            })
            .then_with(|| {
                other
                    .get_exact_phrase_occurrence_rank()
                    .cmp(self.get_exact_phrase_occurrence_rank())
            })
            .then_with(|| {
                other
                    .get_exact_token_occurrence_rank()
                    .cmp(self.get_exact_token_occurrence_rank())
            })
            .then_with(|| {
                other
                    .get_fuzzy_whole_phrase()
                    .cmp(&self.get_fuzzy_whole_phrase())
            })
            .then_with(|| {
                other
                    .get_fuzzy_subphrase_counts_vector()
                    .cmp(self.get_fuzzy_subphrase_counts_vector())
            })
            .then_with(|| {
                other
                    .get_fuzzy_weighted_positional_coverage_vector()
                    .cmp(self.get_fuzzy_weighted_positional_coverage_vector())
            })
            .then_with(|| {
                self.get_fuzzy_field_quality()
                    .cmp(&other.get_fuzzy_field_quality())
            })
            .then_with(|| {
                self.get_typo_cost_vector()
                    .cmp(other.get_typo_cost_vector())
            })
            .then_with(|| {
                self.get_insertion_vector()
                    .cmp(other.get_insertion_vector())
            })
            .then_with(|| self.get_order_distance().cmp(&other.get_order_distance()))
            .then_with(|| self.get_total_gap().cmp(&other.get_total_gap()))
            .then_with(|| self.get_density_rank().cmp(&other.get_density_rank()))
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
struct TokenTypoSeverity {
    core_cost: usize,
    position: usize,
    insertions: usize,
}

impl TokenTypoSeverity {
    fn new(core_cost: usize, position: usize, insertions: usize) -> Self {
        Self {
            core_cost,
            position,
            insertions,
        }
    }
}

fn typo_cost_vector(matches: &[TokenMatch]) -> TypoCostVector {
    TypoCostVector::new(
        matches
            .iter()
            .map(|matched| {
                if matched.fuzzy && !matched.exact {
                    matched.severity.core_cost
                } else {
                    0
                }
            })
            .collect(),
    )
}

fn insertion_vector(matches: &[TokenMatch]) -> InsertionVector {
    InsertionVector::new(
        matches
            .iter()
            .map(|matched| matched.severity.insertions)
            .collect(),
    )
}

#[allow(dead_code)]
#[derive(Clone, Debug, Eq, PartialEq)]
struct MetadataRelevanceRank {
    exact_whole_phrase_present: bool,
    exact_subphrase_counts_vector: Vec<usize>,
    exact_weighted_positional_coverage_vector: CoverageRank,
    exact_field_quality: u8,
    fuzzy_weighted_positional_coverage_vector: CoverageRank,
    fuzzy_whole_phrase: bool,
    fuzzy_subphrase_counts_vector: Vec<usize>,
    fuzzy_field_quality: u8,
    order_distance: usize,
    total_gap: usize,
    typo_cost_vector: TypoCostVector,
    insertion_vector: InsertionVector,
    exact_phrase_occurrence_rank: FieldOccurrenceRank,
    exact_token_occurrence_rank: FieldOccurrenceRank,
    density: DensityRank,
}

impl Ord for MetadataRelevanceRank {
    fn cmp(&self, other: &Self) -> Ordering {
        Reverse(self.exact_whole_phrase_present)
            .cmp(&Reverse(other.exact_whole_phrase_present))
            .then_with(|| {
                Reverse(&self.exact_subphrase_counts_vector)
                    .cmp(&Reverse(&other.exact_subphrase_counts_vector))
            })
            .then_with(|| {
                Reverse(&self.exact_weighted_positional_coverage_vector)
                    .cmp(&Reverse(&other.exact_weighted_positional_coverage_vector))
            })
            .then_with(|| self.exact_field_quality.cmp(&other.exact_field_quality))
            .then_with(|| {
                Reverse(&self.exact_phrase_occurrence_rank)
                    .cmp(&Reverse(&other.exact_phrase_occurrence_rank))
            })
            .then_with(|| {
                Reverse(&self.exact_token_occurrence_rank)
                    .cmp(&Reverse(&other.exact_token_occurrence_rank))
            })
            .then_with(|| Reverse(self.fuzzy_whole_phrase).cmp(&Reverse(other.fuzzy_whole_phrase)))
            .then_with(|| {
                Reverse(&self.fuzzy_subphrase_counts_vector)
                    .cmp(&Reverse(&other.fuzzy_subphrase_counts_vector))
            })
            .then_with(|| {
                Reverse(&self.fuzzy_weighted_positional_coverage_vector)
                    .cmp(&Reverse(&other.fuzzy_weighted_positional_coverage_vector))
            })
            .then_with(|| self.fuzzy_field_quality.cmp(&other.fuzzy_field_quality))
            .then_with(|| self.typo_cost_vector.cmp(&other.typo_cost_vector))
            .then_with(|| self.insertion_vector.cmp(&other.insertion_vector))
            .then_with(|| self.order_distance.cmp(&other.order_distance))
            .then_with(|| self.total_gap.cmp(&other.total_gap))
            .then_with(|| self.density.cmp(&other.density))
    }
}

impl PartialOrd for MetadataRelevanceRank {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn normalize_metadata_query(query: &str) -> Option<String> {
    let lowered = to_lowercase(query.trim());
    if lowered.is_empty() {
        return None;
    }

    let tokens = split_search_tokens(&lowered);
    if tokens.is_empty() {
        return None;
    }

    Some(tokens.join(" "))
}

pub fn sort_matches(matches: &mut [(MetadataCandidate<'_>, SearchRank)]) {
    matches.sort_unstable_by(
        |(left_candidate, left_rank), (right_candidate, right_rank)| {
            compare_search_results(
                left_rank,
                left_candidate.score,
                left_candidate.key,
                right_rank,
                right_candidate.score,
                right_candidate.key,
            )
        },
    );
}

fn field_occurrence_rank(
    fields: &[PreparedMetadataField],
    count: impl Fn(&PreparedMetadataField) -> usize,
) -> FieldOccurrenceRank {
    FieldOccurrenceRank::from_entries(fields.iter().map(|field| (field.priority, count(field))))
}

#[derive(Clone, Copy, Debug, Default)]
struct TokenMatch {
    exact: bool,
    fuzzy: bool,
    severity: TokenTypoSeverity,
    scope: Option<TokenMatchScope>,
    field_priority: u8,
}

type TokenMatchMatrix = Vec<Vec<Vec<TokenMatch>>>;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum TokenMatchScope {
    WholeToken,
    Prefix,
    Suffix,
    Substring,
    Fragmented,
}

#[derive(Clone, Copy, Debug)]
struct TokenMatchDetail {
    severity: TokenTypoSeverity,
    scope: TokenMatchScope,
}

#[allow(dead_code)]
fn metadata_relevance_rank(
    query_tokens: &[String],
    allow_typo: bool,
    fields: &[SearchField<'_>],
) -> MetadataRelevanceRank {
    let normalized_fields = fields
        .iter()
        .filter_map(|field| {
            let lowered = to_lowercase(field.value.trim());
            let tokens = split_search_tokens(&lowered)
                .into_iter()
                .map(str::to_string)
                .collect::<Vec<_>>();
            (!tokens.is_empty()).then_some(PreparedMetadataField {
                priority: field.priority,
                value: lowered,
                tokens,
            })
        })
        .collect::<Vec<_>>();

    if query_tokens.is_empty() || normalized_fields.is_empty() {
        return MetadataRelevanceRank {
            exact_whole_phrase_present: false,
            exact_subphrase_counts_vector: Vec::new(),
            exact_weighted_positional_coverage_vector: CoverageRank::default(),
            exact_field_quality: u8::MAX,
            fuzzy_weighted_positional_coverage_vector: CoverageRank::default(),
            fuzzy_whole_phrase: false,
            fuzzy_subphrase_counts_vector: Vec::new(),
            fuzzy_field_quality: u8::MAX,
            order_distance: usize::MAX,
            total_gap: usize::MAX,
            typo_cost_vector: TypoCostVector::default(),
            insertion_vector: InsertionVector::default(),
            exact_phrase_occurrence_rank: FieldOccurrenceRank::default(),
            exact_token_occurrence_rank: FieldOccurrenceRank::default(),
            density: DensityRank::new(0, usize::MAX),
        };
    }

    let fuzzy_best_matches = query_tokens
        .iter()
        .map(|query_token| best_fuzzy_token_match(query_token, &normalized_fields, allow_typo))
        .collect::<Vec<_>>();

    let exact_whole_phrase_present = exact_phrase_present_by(
        query_tokens,
        normalized_fields
            .iter()
            .map(|field| field.tokens.as_slice()),
        literal_token_match,
    );

    let exact_subphrase_counts_vector = exact_subphrase_counts_vector_by(
        query_tokens,
        normalized_fields
            .iter()
            .map(|field| field.tokens.as_slice()),
        literal_token_match,
    );
    let fuzzy_subphrase_counts_vector =
        fuzzy_subphrase_counts_vector(query_tokens, &normalized_fields);
    let exact_weighted_positional_coverage_vector = CoverageRank::new(
        position_vector(
            query_tokens.iter().map(|query_token| {
                best_exact_token_scope(query_token, &normalized_fields).map(|scope| scope as usize)
            }),
            4,
        ),
        &[4, 2, 2, 1],
    );
    let fuzzy_weighted_positional_coverage_vector = CoverageRank::new(
        metadata_fuzzy_coverage_vector(&fuzzy_best_matches),
        &[4, 2, 2, 1, 1],
    );
    let fuzzy_whole_phrase =
        fuzzy_whole_phrase_present(query_tokens, &normalized_fields, allow_typo);

    let typo_cost_vector = typo_cost_vector(&fuzzy_best_matches);
    let insertion_vector = insertion_vector(&fuzzy_best_matches);

    let exact_field_quality = best_exact_field_priority(query_tokens, &normalized_fields);
    let fuzzy_field_quality = fuzzy_best_matches
        .iter()
        .filter(|matched| matched.fuzzy && !matched.exact)
        .map(|matched| matched.field_priority)
        .min()
        .unwrap_or(u8::MAX);
    let (order_distance, total_gap) =
        ordered_span_stats(query_tokens, &normalized_fields, allow_typo);

    let exact_phrase_occurrence_rank = field_occurrence_rank(&normalized_fields, |field| {
        exact_phrase_occurrence_count_by(
            query_tokens,
            std::iter::once(field.tokens.as_slice()),
            literal_token_match,
        )
    });
    let exact_token_occurrence_rank = field_occurrence_rank(&normalized_fields, |field| {
        field
            .tokens
            .iter()
            .map(|candidate| {
                query_tokens
                    .iter()
                    .filter(|query| literal_token_match(query, candidate))
                    .count()
            })
            .sum()
    });

    let density = density_stats(query_tokens, &normalized_fields, &fuzzy_best_matches);

    MetadataRelevanceRank {
        exact_whole_phrase_present,
        exact_subphrase_counts_vector,
        exact_weighted_positional_coverage_vector,
        exact_field_quality,
        fuzzy_weighted_positional_coverage_vector,
        fuzzy_whole_phrase,
        fuzzy_subphrase_counts_vector,
        fuzzy_field_quality,
        order_distance,
        total_gap,
        typo_cost_vector,
        insertion_vector,
        exact_phrase_occurrence_rank,
        exact_token_occurrence_rank,
        density,
    }
}

fn literal_token_match(query: &str, candidate: &str) -> bool {
    candidate.contains(query)
}

fn literal_token_scope(query: &str, candidate: &str) -> Option<TokenMatchScope> {
    exact_token_position(query, candidate).map(|position| match position {
        0 => TokenMatchScope::WholeToken,
        1 => TokenMatchScope::Prefix,
        2 => TokenMatchScope::Suffix,
        _ => TokenMatchScope::Substring,
    })
}

fn best_exact_token_scope(
    query: &str,
    fields: &[PreparedMetadataField],
) -> Option<TokenMatchScope> {
    fields
        .iter()
        .flat_map(|field| field.tokens.iter())
        .filter_map(|candidate| exact_token_position(query, candidate))
        .map(|position| match position {
            0 => TokenMatchScope::WholeToken,
            1 => TokenMatchScope::Prefix,
            2 => TokenMatchScope::Suffix,
            _ => TokenMatchScope::Substring,
        })
        .min()
}

fn metadata_fuzzy_coverage_vector(matches: &[TokenMatch]) -> Vec<usize> {
    position_vector(
        matches.iter().map(|matched| match matched.scope {
            Some(scope) if matched.fuzzy && !matched.exact => Some(scope as usize),
            _ => None,
        }),
        5,
    )
}

fn fuzzy_whole_phrase_present(
    query_tokens: &[String],
    fields: &[PreparedMetadataField],
    allow_typo: bool,
) -> bool {
    if !allow_typo || query_tokens.is_empty() {
        return false;
    }

    fields.iter().any(|field| {
        field.tokens.windows(query_tokens.len()).any(|window| {
            let mut contains_fuzzy = false;
            for (query, candidate) in query_tokens.iter().zip(window) {
                if literal_token_match(query, candidate) {
                    continue;
                }
                if fuzzy_or_exact_token_match(candidate, query).is_none() {
                    return false;
                }
                contains_fuzzy = true;
            }
            contains_fuzzy
        })
    })
}

fn fuzzy_subphrase_counts_vector(
    query_tokens: &[String],
    fields: &[PreparedMetadataField],
) -> Vec<usize> {
    if query_tokens.len() < 3 {
        return Vec::new();
    }
    (2..query_tokens.len())
        .rev()
        .map(|window_len| {
            let mut count = 0;
            for field in fields {
                if field.tokens.len() < window_len {
                    continue;
                }
                for query_start in 0..=query_tokens.len() - window_len {
                    let query_window = &query_tokens[query_start..query_start + window_len];
                    count += field
                        .tokens
                        .windows(window_len)
                        .filter(|candidate_window| {
                            candidate_window.iter().zip(query_window.iter()).all(
                                |(candidate, query)| {
                                    fuzzy_or_exact_token_match(candidate, query).is_some()
                                },
                            )
                        })
                        .count();
                }
            }
            count
        })
        .collect()
}

fn fuzzy_whole_phrase_present_cached(
    query_tokens: &[String],
    fields: &[PreparedMetadataField],
    allow_typo: bool,
    matches: &TokenMatchMatrix,
) -> bool {
    if !allow_typo || query_tokens.is_empty() {
        return false;
    }

    fields.iter().enumerate().any(|(field_idx, field)| {
        field
            .tokens
            .windows(query_tokens.len())
            .enumerate()
            .any(|(candidate_start, window)| {
                let mut contains_fuzzy = false;
                for (query_idx, _) in window.iter().enumerate() {
                    let matched = matches[query_idx][field_idx][candidate_start + query_idx];
                    if matched.exact {
                        continue;
                    }
                    if !matched.fuzzy {
                        return false;
                    }
                    contains_fuzzy = true;
                }
                contains_fuzzy
            })
    })
}

fn fuzzy_subphrase_counts_vector_cached(
    query_tokens: &[String],
    fields: &[PreparedMetadataField],
    matches: &TokenMatchMatrix,
) -> Vec<usize> {
    if query_tokens.len() < 3 {
        return Vec::new();
    }
    (2..query_tokens.len())
        .rev()
        .map(|window_len| {
            let mut count = 0;
            for (field_idx, field) in fields.iter().enumerate() {
                if field.tokens.len() < window_len {
                    continue;
                }
                for query_start in 0..=query_tokens.len() - window_len {
                    for candidate_start in 0..=field.tokens.len() - window_len {
                        if (0..window_len).all(|offset| {
                            matches[query_start + offset][field_idx][candidate_start + offset].fuzzy
                        }) {
                            count += 1;
                        }
                    }
                }
            }
            count
        })
        .collect()
}

fn best_fuzzy_token_match(
    query_token: &str,
    fields: &[PreparedMetadataField],
    allow_typo: bool,
) -> TokenMatch {
    let mut best = TokenMatch::default();
    for field in fields {
        for candidate_token in &field.tokens {
            if let Some(scope) = literal_token_scope(query_token, candidate_token) {
                let candidate = TokenMatch {
                    exact: true,
                    fuzzy: true,
                    severity: TokenTypoSeverity::default(),
                    scope: Some(scope),
                    field_priority: field.priority,
                };
                if !best.exact
                    || (candidate.scope, candidate.field_priority)
                        < (best.scope, best.field_priority)
                {
                    best = candidate;
                }
                continue;
            }
            if !allow_typo {
                continue;
            }
            let Some(detail) = fuzzy_or_exact_token_match(candidate_token, query_token) else {
                continue;
            };
            let candidate = TokenMatch {
                exact: false,
                fuzzy: true,
                severity: detail.severity,
                scope: Some(detail.scope),
                field_priority: field.priority,
            };
            if !best.fuzzy
                || (!best.exact
                    && (
                        candidate.scope,
                        candidate.severity,
                        candidate.field_priority,
                    ) < (best.scope, best.severity, best.field_priority))
            {
                best = candidate;
            }
        }
    }
    best
}

fn best_fuzzy_token_match_from_matrix(
    field_matches: &[Vec<TokenMatch>],
    allow_typo: bool,
) -> TokenMatch {
    let mut best = TokenMatch::default();
    for field_matches in field_matches {
        for candidate in field_matches {
            if candidate.exact {
                if !best.exact
                    || (candidate.scope, candidate.field_priority)
                        < (best.scope, best.field_priority)
                {
                    best = *candidate;
                }
            } else if allow_typo
                && candidate.fuzzy
                && (!best.fuzzy
                    || (!best.exact
                        && (
                            candidate.scope,
                            candidate.severity,
                            candidate.field_priority,
                        ) < (best.scope, best.severity, best.field_priority)))
            {
                best = *candidate;
            }
        }
    }
    best
}

fn fuzzy_or_exact_token_match(
    candidate_token: &str,
    query_token: &str,
) -> Option<TokenMatchDetail> {
    if let Some(scope) = literal_token_scope(query_token, candidate_token) {
        return Some(TokenMatchDetail {
            severity: TokenTypoSeverity::default(),
            scope,
        });
    }
    let query_chars: Vec<char> = query_token.chars().collect();
    let query_len = query_chars.len();
    if query_len < 3 || !token_has_signal(query_token, candidate_token) {
        return None;
    }

    let limit = max_typos(query_len).max(1);
    let contiguous = if let Some((_, _, start_idx, end_idx)) =
        fuzzy_match_details(query_token, candidate_token)
    {
        let candidate_chars: Vec<char> = candidate_token.chars().collect();
        let candidate_len = candidate_chars.len();
        let end = end_idx.min(candidate_len);
        let start_min = start_idx.saturating_sub(limit).min(end);
        let start_max = (start_idx + limit).min(end);
        let mut best_match: Option<TokenMatchDetail> = None;
        let mut scratch = DamerauScratch::default();
        for s in start_min..=start_max {
            if let Some((_distance, operations)) = bounded_damerau_levenshtein_chars_with_scratch(
                &candidate_chars[s..end],
                &query_chars,
                limit,
                &mut scratch,
            ) {
                let position = if s == 0 {
                    0
                } else if end == candidate_len {
                    1
                } else {
                    2
                };
                let candidate = TokenMatchDetail {
                    severity: TokenTypoSeverity::new(
                        operations.core_typo_cost(),
                        position,
                        operations.insertions,
                    ),
                    scope: if s == 0 && end == candidate_len {
                        TokenMatchScope::WholeToken
                    } else if s == 0 {
                        TokenMatchScope::Prefix
                    } else if end == candidate_len {
                        TokenMatchScope::Suffix
                    } else {
                        TokenMatchScope::Substring
                    },
                };
                if best_match.as_ref().is_none_or(|best| {
                    (candidate.scope, candidate.severity) < (best.scope, best.severity)
                }) {
                    best_match = Some(candidate);
                }
            }
        }
        best_match
    } else {
        None
    };

    let fragmented = if candidate_token.chars().count().saturating_sub(query_len) > limit {
        fragmented_token_match(query_token, candidate_token, limit).and_then(|matched| {
            let ratio =
                matched.distance as f64 / query_len.max(candidate_token.chars().count()) as f64;
            if ratio > 0.5 || matched.internal_breaks == 0 {
                return None;
            }
            Some(TokenMatchDetail {
                severity: TokenTypoSeverity::new(
                    matched.operations.core_typo_cost(),
                    matched.position_rank,
                    matched.operations.insertions,
                ),
                scope: TokenMatchScope::Fragmented,
            })
        })
    } else {
        None
    };

    match (contiguous, fragmented) {
        (Some(left), Some(right)) => Some(
            if (right.scope, right.severity) < (left.scope, left.severity) {
                right
            } else {
                left
            },
        ),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

fn best_exact_field_priority(query_tokens: &[String], fields: &[PreparedMetadataField]) -> u8 {
    fields
        .iter()
        .filter(|field| {
            field.tokens.iter().any(|candidate_token| {
                query_tokens
                    .iter()
                    .any(|query_token| literal_token_match(query_token, candidate_token))
            })
        })
        .map(|field| field.priority)
        .min()
        .unwrap_or(u8::MAX)
}

fn ordered_span_stats(
    query_tokens: &[String],
    fields: &[PreparedMetadataField],
    allow_typo: bool,
) -> (usize, usize) {
    ordered_token_span(
        query_tokens,
        fields.iter().map(|field| field.tokens.as_slice()),
        |query, candidate| {
            literal_token_match(query, candidate)
                || (allow_typo && fuzzy_or_exact_token_match(candidate, query).is_some())
        },
    )
}

fn density_stats(
    query_tokens: &[String],
    fields: &[PreparedMetadataField],
    fuzzy_matches: &[TokenMatch],
) -> DensityRank {
    let matched_count = fuzzy_matches
        .iter()
        .filter(|matched| matched.exact || matched.fuzzy)
        .count();
    let best_field_len = fields
        .iter()
        .filter(|field| {
            field.tokens.iter().any(|candidate_token| {
                query_tokens
                    .iter()
                    .any(|query_token| literal_token_match(query_token, candidate_token))
            })
        })
        .map(|field| field.tokens.len())
        .min()
        .unwrap_or_else(|| {
            fields
                .iter()
                .map(|field| field.tokens.len())
                .min()
                .unwrap_or(usize::MAX)
        });
    DensityRank::new(matched_count, best_field_len)
}

pub fn dedup_push_search_field<'a>(
    fields: &mut Vec<SearchField<'a>>,
    priority: u8,
    value: Option<&'a str>,
) {
    let Some(value) = value else {
        return;
    };
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return;
    }
    if fields
        .iter()
        .any(|field| field.priority == priority && field.value.eq_ignore_ascii_case(trimmed))
    {
        return;
    }
    fields.push(SearchField {
        priority,
        value: trimmed,
    });
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct MetadataEditCost {
    distance: usize,
    keyboard_distance: usize,
}

impl MetadataEditCost {
    fn infinite(limit: usize) -> Self {
        Self {
            distance: limit + 1,
            keyboard_distance: usize::MAX / 4,
        }
    }
}

impl Ord for MetadataEditCost {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .cmp(&other.distance)
            .then_with(|| self.keyboard_distance.cmp(&other.keyboard_distance))
    }
}

impl PartialOrd for MetadataEditCost {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Default)]
struct MetadataFuzzyScratch {
    query_chars: Vec<char>,
    target_chars: Vec<char>,
    prev_prev: Vec<MetadataEditCost>,
    prev: Vec<MetadataEditCost>,
    curr: Vec<MetadataEditCost>,
}

fn fuzzy_match_details(query: &str, target: &str) -> Option<(usize, usize, usize, usize)> {
    fuzzy_match_details_with_scratch(query, target, &mut MetadataFuzzyScratch::default())
}

fn fuzzy_match_details_with_scratch(
    query: &str,
    target: &str,
    scratch: &mut MetadataFuzzyScratch,
) -> Option<(usize, usize, usize, usize)> {
    let query_lower = lowercase_cow(query);
    let target_lower = lowercase_cow(target);

    let q_len = query_lower.chars().count();
    if q_len == 0 {
        return Some((0, 0, 0, 0));
    }

    let t_len = target_lower.chars().count();
    let limit = q_len / 2;

    if t_len < q_len.saturating_sub(limit) {
        return None;
    }

    scratch.query_chars.clear();
    scratch.query_chars.extend(query_lower.chars());
    scratch.target_chars.clear();
    scratch.target_chars.extend(target_lower.chars());
    let query_chars = &scratch.query_chars;
    let target_chars = &scratch.target_chars;

    let inf = MetadataEditCost::infinite(limit);
    scratch.prev_prev.resize(t_len + 1, inf);
    scratch.prev.resize(t_len + 1, MetadataEditCost::default());
    scratch.curr.resize(t_len + 1, inf);
    scratch.prev_prev.fill(inf);
    for cell in scratch.prev.iter_mut() {
        *cell = MetadataEditCost {
            distance: 0,
            keyboard_distance: 0,
        };
    }
    scratch.curr.fill(inf);

    for i in 1..=q_len {
        scratch.curr.fill(inf);
        scratch.curr[0] = MetadataEditCost {
            distance: i,
            keyboard_distance: i,
        };

        let mut row_min = inf;
        for j in 1..=t_len {
            let deletion = MetadataEditCost {
                distance: scratch.prev[j].distance + 1,
                keyboard_distance: scratch.prev[j].keyboard_distance + 1,
            };
            let insertion = MetadataEditCost {
                distance: scratch.curr[j - 1].distance + 1,
                keyboard_distance: scratch.curr[j - 1].keyboard_distance + 1,
            };
            let substitution = if query_chars[i - 1] == target_chars[j - 1] {
                scratch.prev[j - 1]
            } else {
                MetadataEditCost {
                    distance: scratch.prev[j - 1].distance + 1,
                    keyboard_distance: scratch.prev[j - 1].keyboard_distance
                        + keyboard_substitution_cost(query_chars[i - 1], target_chars[j - 1]),
                }
            };
            let mut cell = deletion.min(insertion).min(substitution);

            if i > 1
                && j > 1
                && query_chars[i - 1] == target_chars[j - 2]
                && query_chars[i - 2] == target_chars[j - 1]
            {
                cell = cell.min(MetadataEditCost {
                    distance: scratch.prev_prev[j - 2].distance + 1,
                    keyboard_distance: scratch.prev_prev[j - 2].keyboard_distance,
                });
            }

            scratch.curr[j] = cell;
            row_min = row_min.min(cell);
        }

        if row_min.distance > limit {
            return None;
        }

        std::mem::swap(&mut scratch.prev_prev, &mut scratch.prev);
        std::mem::swap(&mut scratch.prev, &mut scratch.curr);
    }

    let mut min_cost = inf;
    let mut best_end_idx = 0;
    let search_start = q_len.saturating_sub(limit).max(1);
    for (j, value) in scratch
        .prev
        .iter()
        .enumerate()
        .take(t_len + 1)
        .skip(search_start)
    {
        if *value < min_cost {
            min_cost = *value;
            best_end_idx = j;
        }
    }

    (min_cost.distance <= limit).then_some((
        min_cost.distance,
        min_cost.keyboard_distance,
        best_end_idx.saturating_sub(q_len),
        best_end_idx,
    ))
}

fn lowercase_cow(value: &str) -> Cow<'_, str> {
    if value.chars().all(|ch| !ch.is_uppercase()) {
        Cow::Borrowed(value)
    } else {
        Cow::Owned(to_lowercase(value))
    }
}

fn is_search_token_separator(c: char) -> bool {
    !c.is_alphanumeric()
}

fn structural_match_details(
    target: &str,
    start_idx: usize,
    matched_char_len: usize,
) -> (u8, usize, usize, usize, usize) {
    let chars: Vec<char> = target.chars().collect();
    let start_idx = start_idx.min(chars.len());
    let end_idx = (start_idx + matched_char_len).min(chars.len());
    let start_boundary = start_idx == 0
        || chars
            .get(start_idx.saturating_sub(1))
            .is_none_or(|c| is_search_token_separator(*c));
    let end_boundary = end_idx == chars.len()
        || chars
            .get(end_idx)
            .is_none_or(|c| is_search_token_separator(*c));

    let mut token_start = start_idx;
    while token_start > 0 && !is_search_token_separator(chars[token_start - 1]) {
        token_start -= 1;
    }

    let mut token_end = end_idx;
    while token_end < chars.len() && !is_search_token_separator(chars[token_end]) {
        token_end += 1;
    }

    let mut token_index = 0usize;
    let mut in_token = false;
    for (idx, ch) in chars.iter().enumerate() {
        if is_search_token_separator(*ch) {
            in_token = false;
            continue;
        }
        if !in_token {
            if idx >= token_start {
                break;
            }
            token_index += 1;
            in_token = true;
        }
    }

    let position_class = if start_idx == 0 && end_idx == chars.len() {
        0
    } else if start_boundary && end_boundary {
        1
    } else if start_idx == 0 {
        2
    } else if start_boundary {
        3
    } else if end_boundary {
        4
    } else {
        5
    };

    (
        position_class,
        token_index,
        (token_end - token_start).abs_diff(matched_char_len),
        start_idx,
        chars.len(),
    )
}

fn best_structural_rank<F: MetadataFieldView>(query: &str, fields: &[F]) -> Option<StructuralRank> {
    let query_lower = to_lowercase(query.trim());
    if query_lower.is_empty() {
        return None;
    }

    let query_len = query_lower.chars().count();
    let mut best_rank = None;

    for (field_index, field) in fields.iter().enumerate() {
        let target_lower = field.lowered_value();
        let target = target_lower.as_ref();
        if target.is_empty() {
            continue;
        }

        for (start_byte_idx, _) in target.match_indices(&query_lower) {
            let start_idx = target[..start_byte_idx].chars().count();
            let (position_class, token_index, token_span_delta, start_idx, field_len) =
                structural_match_details(target, start_idx, query_len);
            let rank = StructuralRank {
                field_index,
                field_priority: field.priority(),
                position_class,
                token_index,
                token_span_delta,
                start_idx,
                matched_char_len: query_len,
                field_len,
            };
            if best_rank.as_ref().is_none_or(|current| rank < *current) {
                best_rank = Some(rank);
            }
        }
    }

    best_rank
}

fn subsequence_match_details(query: &str, target: &str) -> Option<(usize, usize, usize, usize)> {
    let query_chars: Vec<char> = query.chars().collect();
    let target_chars: Vec<char> = target.chars().collect();
    if query_chars.is_empty() || target_chars.is_empty() {
        return None;
    }

    let mut q_idx = 0usize;
    let mut first_idx = None;
    let mut last_idx = 0usize;
    let mut gap_count = 0usize;

    for (t_idx, ch) in target_chars.iter().enumerate() {
        if q_idx < query_chars.len() && *ch == query_chars[q_idx] {
            if let Some(prev_idx) = first_idx.or(Some(last_idx))
                && q_idx > 0
                && t_idx > prev_idx + 1
            {
                gap_count += 1;
            }
            first_idx.get_or_insert(t_idx);
            last_idx = t_idx;
            q_idx += 1;
            if q_idx == query_chars.len() {
                let start_idx = first_idx?;
                let span = last_idx - start_idx + 1;
                return Some((
                    start_idx,
                    last_idx,
                    span.saturating_sub(query_chars.len()),
                    gap_count,
                ));
            }
        }
    }

    None
}

fn best_abbreviation_rank<F: MetadataFieldView>(
    query: &str,
    fields: &[F],
) -> Option<AbbreviationRank> {
    let query_lower = to_lowercase(query.trim());
    if query_lower.is_empty() {
        return None;
    }

    let query_compact = normalize_compact_alnum(&query_lower);
    if query_compact.chars().count() < 2 {
        return None;
    }

    let mut best_rank = None;

    for (field_index, field) in fields.iter().enumerate() {
        if field.priority() > 2 {
            continue;
        }
        let target_lower = field.lowered_value();
        let target = target_lower.as_ref();
        if target.is_empty() {
            continue;
        }

        // Fast path: check if target contains all characters of query in order
        let mut target_chars_iter = target_lower.chars();
        let mut found_all = true;
        for q_char in query_compact.chars() {
            if !target_chars_iter.any(|t_char| t_char == q_char) {
                found_all = false;
                break;
            }
        }
        if !found_all {
            continue;
        }

        let tokens = split_search_tokens(target);
        let mut candidates: Vec<(u8, String)> = Vec::new();

        for token in &tokens {
            candidates.push((0, (*token).to_string()));
        }
        candidates.push((1, target.to_string()));
        if tokens.len() > 1 {
            let collapsed = tokens.concat();
            if !collapsed.is_empty() {
                candidates.push((2, collapsed));
            }
        }

        for (variant_scope, candidate_target) in candidates {
            let Some((start_idx, last_idx, gap_span, gap_count)) =
                subsequence_match_details(&query_compact, &candidate_target)
            else {
                continue;
            };
            if gap_span == 0 {
                continue;
            }
            let matched_span = last_idx - start_idx + 1;
            let (position_class, token_index, _, _, field_len) =
                structural_match_details(&candidate_target, start_idx, matched_span);
            let rank = AbbreviationRank {
                field_index,
                field_priority: field.priority(),
                variant_scope,
                position_class,
                token_index,
                gap_span,
                gap_count,
                start_idx,
                matched_char_len: matched_span,
                field_len,
            };
            if best_rank.as_ref().is_none_or(|current| rank < *current) {
                best_rank = Some(rank);
            }
        }
    }

    best_rank
}

fn best_fuzzy_rank<F: MetadataFieldView>(query: &str, fields: &[F]) -> Option<DistanceRank> {
    let query_lower = to_lowercase(query.trim());
    if query_lower.is_empty() || query_lower.chars().count() < 3 {
        return None;
    }

    let query_collapsed = normalize_compact_alnum(&query_lower);
    let allow_collapsed =
        !query_collapsed.is_empty() && !query_lower.chars().any(is_search_token_separator);
    let mut best_rank = None;
    let mut scratch = MetadataFuzzyScratch::default();

    for (field_index, field) in fields.iter().enumerate() {
        let target_lower = field.lowered_value();
        let target = target_lower.as_ref();
        if target.is_empty() {
            continue;
        }
        if !token_has_signal(&query_lower, target) {
            continue;
        }
        let tokens = split_search_tokens(target);
        let mut candidates: Vec<(u8, String, String)> = Vec::new();

        for token in &tokens {
            candidates.push((1, query_lower.clone(), (*token).to_string()));
        }

        if allow_collapsed && tokens.len() > 1 {
            let collapsed = tokens.concat();
            if !collapsed.is_empty() {
                candidates.push((2, query_collapsed.clone(), collapsed));
            }
        }

        for (variant_scope, candidate_query, candidate_target) in candidates {
            if !token_has_signal(&candidate_query, &candidate_target) {
                continue;
            }
            let Some((distance, keyboard_distance, start_idx, _)) =
                fuzzy_match_details_with_scratch(&candidate_query, &candidate_target, &mut scratch)
            else {
                continue;
            };
            let compared_len = candidate_query
                .chars()
                .count()
                .max(candidate_target.chars().count())
                .max(1);
            let ratio_milli = ratio_milli(distance, compared_len);
            let matched_len = candidate_query.chars().count();
            let (position_class, token_index, token_span_delta, _, field_len) =
                structural_match_details(&candidate_target, start_idx, matched_len);
            let rank = DistanceRank {
                distance,
                ratio_milli,
                keyboard_distance,
                field_index,
                field_priority: field.priority(),
                variant_scope,
                position_class,
                token_index,
                token_span_delta,
                start_idx,
                matched_char_len: matched_len,
                field_len,
            };
            if best_rank.as_ref().is_none_or(|current| rank < *current) {
                best_rank = Some(rank);
            }
        }
    }

    best_rank
}

fn best_typo_rank<F: MetadataFieldView>(query: &str, fields: &[F]) -> Option<DistanceRank> {
    let query_lower = to_lowercase(query.trim());
    if query_lower.is_empty() || query_lower.chars().count() < 3 {
        return None;
    }

    let query_collapsed = normalize_compact_alnum(&query_lower);
    let allow_collapsed =
        !query_collapsed.is_empty() && !query_lower.chars().any(is_search_token_separator);
    let mut best_rank = None;
    let mut scratch = MetadataFuzzyScratch::default();

    for (field_index, field) in fields.iter().enumerate() {
        let target_lower = field.lowered_value();
        let target = target_lower.as_ref();
        if target.is_empty() {
            continue;
        }
        if !token_has_signal(&query_lower, target) {
            continue;
        }
        let tokens = split_search_tokens(target);
        let mut candidates: Vec<(u8, String, String, usize)> =
            vec![(0, query_lower.clone(), target.to_string(), 0)];

        for (idx, token) in tokens.iter().enumerate() {
            candidates.push((1, query_lower.clone(), (*token).to_string(), idx));
        }

        if allow_collapsed && tokens.len() > 1 {
            let collapsed = tokens.concat();
            if !collapsed.is_empty() {
                candidates.push((2, query_collapsed.clone(), collapsed, 0));
            }
        }

        for (variant_scope, candidate_query, candidate_target, token_idx) in candidates {
            let Some((distance, keyboard_distance, start_idx, _)) =
                fuzzy_match_details_with_scratch(&candidate_query, &candidate_target, &mut scratch)
            else {
                continue;
            };
            let compared_len = candidate_query
                .chars()
                .count()
                .max(candidate_target.chars().count())
                .max(1);
            let ratio_milli = ratio_milli(distance, compared_len);
            let matched_len = candidate_query.chars().count();
            let (position_class, mut token_index, token_span_delta, _, field_len) =
                structural_match_details(&candidate_target, start_idx, matched_len);
            if variant_scope == 1 {
                token_index = token_idx;
            }
            let rank = DistanceRank {
                distance,
                ratio_milli,
                keyboard_distance,
                field_index,
                field_priority: field.priority(),
                variant_scope,
                position_class,
                token_index,
                token_span_delta,
                start_idx,
                matched_char_len: matched_len,
                field_len,
            };
            if best_rank.as_ref().is_none_or(|current| rank < *current) {
                best_rank = Some(rank);
            }
        }
    }

    best_rank
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn app_name_field_beats_exec_field() {
        let fields = [
            SearchField {
                priority: 0,
                value: "Application Launcher",
            },
            SearchField {
                priority: 1,
                value: "applauncher",
            },
        ];
        let candidate = MetadataCandidate {
            key: "applicationlauncher.desktop",
            fields: &fields,
            score: 1.0,
        };
        let rank = MetadataQuery::new("app")
            .unwrap()
            .search_rank(candidate)
            .unwrap();
        match rank {
            SearchRank::Structural(rank) => assert_eq!(rank.field_priority, 0),
            _ => panic!("expected structural rank"),
        }
    }

    #[test]
    fn prepared_candidates_match_unprepared_candidates() {
        let fields = [
            SearchField {
                priority: 0,
                value: "Plasma Meta",
            },
            SearchField {
                priority: 1,
                value: "plasma-desktop",
            },
        ];
        let candidate = MetadataCandidate {
            key: "plasma-meta",
            fields: &fields,
            score: 0.0,
        };
        let prepared_fields = prepare_fields(&fields);
        let prepared = PreparedMetadataCandidate {
            key: candidate.key,
            fields: &prepared_fields,
            score: candidate.score,
        };
        let query = MetadataQuery::new("plasma")
            .unwrap()
            .with_typo_fallback(true);

        assert_eq!(
            query.search_rank(candidate),
            query.search_rank_prepared(prepared)
        );
    }

    #[test]
    fn typo_fallback_matches_collapsed_identifier() {
        let fields = [SearchField {
            priority: 0,
            value: "Application Launcher",
        }];
        let candidate = MetadataCandidate {
            key: "applicationlauncher.desktop",
            fields: &fields,
            score: 1.0,
        };
        let rank = MetadataQuery::new("applaunch")
            .unwrap()
            .with_typo_fallback(true)
            .search_rank(candidate)
            .unwrap();
        match rank {
            SearchRank::Typo(rank) => {
                assert!(rank.distance <= 4);
                assert!(rank.variant_scope <= 2);
            }
            _ => panic!("expected typo rank"),
        }
    }

    #[test]
    fn window_title_beats_class_when_both_match() {
        let fields = [
            SearchField {
                priority: 0,
                value: "CopyQ - Clipboard",
            },
            SearchField {
                priority: 1,
                value: "copyq",
            },
        ];
        let candidate = MetadataCandidate {
            key: "window:1",
            fields: &fields,
            score: 1.0,
        };
        let rank = MetadataQuery::new("copyq")
            .unwrap()
            .search_rank(candidate)
            .unwrap();
        match rank {
            SearchRank::Structural(rank) => assert_eq!(rank.field_priority, 0),
            _ => panic!("expected structural rank"),
        }
    }

    #[test]
    fn sort_uses_rank_then_score_then_key() {
        let fields = [SearchField {
            priority: 0,
            value: "git status",
        }];
        let a = MetadataCandidate {
            key: "a",
            fields: &fields,
            score: 1.0,
        };
        let b = MetadataCandidate {
            key: "b",
            fields: &fields,
            score: 2.0,
        };
        let mut matches = vec![
            (
                a,
                MetadataQuery::new("git").unwrap().search_rank(a).unwrap(),
            ),
            (
                b,
                MetadataQuery::new("git").unwrap().search_rank(b).unwrap(),
            ),
        ];
        MetadataQuery::new("git")
            .unwrap()
            .sort_matches(&mut matches);
        assert_eq!(matches[0].0.key, "b");
    }

    #[test]
    fn multi_term_metadata_queries_can_match_partial_coverage() {
        let fields = [
            SearchField {
                priority: 0,
                value: "term1 term2 term3",
            },
            SearchField {
                priority: 1,
                value: "term4",
            },
        ];
        let candidate = MetadataCandidate {
            key: "partial",
            fields: &fields,
            score: 0.0,
        };
        assert!(
            MetadataQuery::new("term1 term2 term3 term4")
                .unwrap()
                .search_rank(candidate)
                .is_some()
        );
    }

    #[test]
    fn total_coverage_breaks_exact_coverage_ties_with_one_typo() {
        let partial_fields = [SearchField {
            priority: 0,
            value: "one two three",
        }];
        let typo_fields = [SearchField {
            priority: 0,
            value: "one two three fuor",
        }];
        let partial = MetadataCandidate {
            key: "partial",
            fields: &partial_fields,
            score: 0.0,
        };
        let typo = MetadataCandidate {
            key: "typo",
            fields: &typo_fields,
            score: 0.0,
        };
        let query = MetadataQuery::new("one two three four")
            .unwrap()
            .with_typo_fallback(true);
        let mut matches = vec![
            (partial, query.search_rank(partial).unwrap()),
            (typo, query.search_rank(typo).unwrap()),
        ];

        query.sort_matches(&mut matches);

        assert_eq!(matches[0].0.key, "typo");
    }

    #[test]
    fn exact_token_classes_rank_whole_prefix_suffix_then_substring() {
        let whole_fields = [SearchField {
            priority: 0,
            value: "app",
        }];
        let prefix_fields = [SearchField {
            priority: 0,
            value: "application",
        }];
        let suffix_fields = [SearchField {
            priority: 0,
            value: "myapp",
        }];
        let substring_fields = [SearchField {
            priority: 0,
            value: "myapplication",
        }];
        let whole = MetadataCandidate {
            key: "whole",
            fields: &whole_fields,
            score: 0.0,
        };
        let prefix = MetadataCandidate {
            key: "prefix",
            fields: &prefix_fields,
            score: 100.0,
        };
        let suffix = MetadataCandidate {
            key: "suffix",
            fields: &suffix_fields,
            score: 200.0,
        };
        let substring = MetadataCandidate {
            key: "substring",
            fields: &substring_fields,
            score: 300.0,
        };
        let query = MetadataQuery::new("app").unwrap();
        let mut matches = vec![
            (substring, query.search_rank(substring).unwrap()),
            (suffix, query.search_rank(suffix).unwrap()),
            (prefix, query.search_rank(prefix).unwrap()),
            (whole, query.search_rank(whole).unwrap()),
        ];

        query.sort_matches(&mut matches);

        assert_eq!(
            matches
                .iter()
                .map(|(candidate, _)| candidate.key)
                .collect::<Vec<_>>(),
            ["whole", "prefix", "suffix", "substring"]
        );
    }

    #[test]
    fn weighted_exact_coverage_balances_quality_and_distinct_tokens() {
        let query = MetadataQuery::new("chrome profile error").unwrap();
        let one_whole_fields = [SearchField {
            priority: 0,
            value: "chrome settings",
        }];
        let three_prefix_fields = [
            SearchField {
                priority: 0,
                value: "chromedriver",
            },
            SearchField {
                priority: 0,
                value: "profilemanager",
            },
            SearchField {
                priority: 0,
                value: "errorlog",
            },
        ];
        let two_whole_fields = [
            SearchField {
                priority: 0,
                value: "chrome",
            },
            SearchField {
                priority: 0,
                value: "profile",
            },
        ];
        let candidate = |key, fields| MetadataCandidate {
            key,
            fields,
            score: 0.0,
        };
        let one_whole = candidate("one-whole", &one_whole_fields);
        let three_prefix = candidate("three-prefix", &three_prefix_fields);
        let two_whole = candidate("two-whole", &two_whole_fields);

        let mut first = vec![
            (one_whole, query.search_rank(one_whole).unwrap()),
            (three_prefix, query.search_rank(three_prefix).unwrap()),
        ];
        query.sort_matches(&mut first);
        assert_eq!(first[0].0.key, "three-prefix");

        let mut second = vec![
            (three_prefix, query.search_rank(three_prefix).unwrap()),
            (two_whole, query.search_rank(two_whole).unwrap()),
        ];
        query.sort_matches(&mut second);
        assert_eq!(second[0].0.key, "two-whole");
    }

    #[test]
    fn fuzzy_whole_token_beats_fuzzy_prefix() {
        let whole_fields = [SearchField {
            priority: 0,
            value: "chrome",
        }];
        let prefix_fields = [SearchField {
            priority: 0,
            value: "chromedriver",
        }];
        let whole = MetadataCandidate {
            key: "whole",
            fields: &whole_fields,
            score: 0.0,
        };
        let prefix = MetadataCandidate {
            key: "prefix",
            fields: &prefix_fields,
            score: 100.0,
        };
        let query = MetadataQuery::new("krome")
            .unwrap()
            .with_typo_fallback(true);
        let mut matches = vec![
            (prefix, query.search_rank(prefix).unwrap()),
            (whole, query.search_rank(whole).unwrap()),
        ];

        query.sort_matches(&mut matches);

        assert_eq!(matches[0].0.key, "whole");
    }

    #[test]
    fn fuzzy_field_quality_uses_the_field_that_supplied_the_fuzzy_match() {
        let preferred_fields = [SearchField {
            priority: 0,
            value: "chrome",
        }];
        let fallback_fields = [SearchField {
            priority: 5,
            value: "chrome",
        }];
        let preferred = MetadataCandidate {
            key: "preferred",
            fields: &preferred_fields,
            score: 0.0,
        };
        let fallback = MetadataCandidate {
            key: "fallback",
            fields: &fallback_fields,
            score: 100.0,
        };
        let query = MetadataQuery::new("krome")
            .unwrap()
            .with_typo_fallback(true);
        let mut matches = vec![
            (fallback, query.search_rank(fallback).unwrap()),
            (preferred, query.search_rank(preferred).unwrap()),
        ];

        query.sort_matches(&mut matches);

        assert_eq!(matches[0].0.key, "preferred");
    }

    #[test]
    fn metadata_subphrase_counts_precede_coverage() {
        let query = MetadataQuery::new("term1 term2 term3 term4").unwrap();

        let full_fields = [SearchField {
            priority: 0,
            value: "term1 term2 term3 term4",
        }];
        let two_three_fields = [
            SearchField {
                priority: 0,
                value: "term1 term2 term3",
            },
            SearchField {
                priority: 1,
                value: "term2 term3 term4",
            },
        ];
        let one_three_one_single_fields = [
            SearchField {
                priority: 0,
                value: "term1 term2 term3",
            },
            SearchField {
                priority: 1,
                value: "term4",
            },
        ];
        let one_three_fields = [SearchField {
            priority: 0,
            value: "term1 term2 term3",
        }];
        let three_two_fields = [
            SearchField {
                priority: 0,
                value: "term1 term2",
            },
            SearchField {
                priority: 1,
                value: "term2 term3",
            },
            SearchField {
                priority: 2,
                value: "term3 term4",
            },
        ];

        let full = MetadataCandidate {
            key: "full",
            fields: &full_fields,
            score: 0.0,
        };
        let two_three = MetadataCandidate {
            key: "two-three",
            fields: &two_three_fields,
            score: 0.0,
        };
        let one_three_one_single = MetadataCandidate {
            key: "one-three-one-single",
            fields: &one_three_one_single_fields,
            score: 0.0,
        };
        let one_three = MetadataCandidate {
            key: "one-three",
            fields: &one_three_fields,
            score: 0.0,
        };
        let three_two = MetadataCandidate {
            key: "three-two",
            fields: &three_two_fields,
            score: 0.0,
        };

        let mut matches = vec![
            (one_three, query.search_rank(one_three).unwrap()),
            (three_two, query.search_rank(three_two).unwrap()),
            (full, query.search_rank(full).unwrap()),
            (
                one_three_one_single,
                query.search_rank(one_three_one_single).unwrap(),
            ),
            (two_three, query.search_rank(two_three).unwrap()),
        ];

        query.sort_matches(&mut matches);

        assert_eq!(
            matches
                .iter()
                .map(|(candidate, _)| candidate.key)
                .collect::<Vec<_>>(),
            vec![
                "full",
                "two-three",
                "one-three-one-single",
                "one-three",
                "three-two",
            ]
        );
    }

    #[test]
    fn repeated_exact_phrase_does_not_beat_better_field_quality() {
        let preferred_fields = [SearchField {
            priority: 0,
            value: "app",
        }];
        let repeated_fields = [SearchField {
            priority: 5,
            value: "app app app app app",
        }];
        let preferred = MetadataCandidate {
            key: "preferred",
            fields: &preferred_fields,
            score: 0.0,
        };
        let repeated = MetadataCandidate {
            key: "repeated",
            fields: &repeated_fields,
            score: 100.0,
        };
        let query = MetadataQuery::new("app").unwrap();
        let mut matches = vec![
            (repeated, query.search_rank(repeated).unwrap()),
            (preferred, query.search_rank(preferred).unwrap()),
        ];

        query.sort_matches(&mut matches);

        assert_eq!(matches[0].0.key, "preferred");
    }

    #[test]
    fn occurrence_ranks_prioritize_title_hits_over_command_line_hits() {
        let title_heavy_fields = [
            SearchField {
                priority: 0,
                value: "codex codex",
            },
            SearchField {
                priority: 5,
                value: "command",
            },
        ];
        let command_line_heavy_fields = [
            SearchField {
                priority: 0,
                value: "codex",
            },
            SearchField {
                priority: 5,
                value: "codex codex codex codex codex codex codex codex",
            },
        ];
        let title_heavy = MetadataCandidate {
            key: "title-heavy",
            fields: &title_heavy_fields,
            score: 0.0,
        };
        let command_line_heavy = MetadataCandidate {
            key: "command-line-heavy",
            fields: &command_line_heavy_fields,
            score: 0.0,
        };
        let query = MetadataQuery::new("codex").unwrap();
        let mut matches = vec![
            (
                command_line_heavy,
                query.search_rank(command_line_heavy).unwrap(),
            ),
            (title_heavy, query.search_rank(title_heavy).unwrap()),
        ];

        query.sort_matches(&mut matches);

        assert_eq!(matches[0].0.key, "title-heavy");
    }

    #[test]
    fn fuzzy_whole_phrase_beats_scattered_fuzzy_tokens() {
        let phrase_fields = [SearchField {
            priority: 0,
            value: "visual stduio code",
        }];
        let scattered_fields = [SearchField {
            priority: 0,
            value: "visual unrelated stduio unrelated code",
        }];
        let phrase = MetadataCandidate {
            key: "phrase",
            fields: &phrase_fields,
            score: 0.0,
        };
        let scattered = MetadataCandidate {
            key: "scattered",
            fields: &scattered_fields,
            score: 100.0,
        };
        let query = MetadataQuery::new("visual studio code")
            .unwrap()
            .with_typo_fallback(true);
        let mut matches = vec![
            (scattered, query.search_rank(scattered).unwrap()),
            (phrase, query.search_rank(phrase).unwrap()),
        ];

        query.sort_matches(&mut matches);

        assert_eq!(matches[0].0.key, "phrase");
    }

    #[test]
    fn trailing_punctuation_is_ignored_in_query() {
        let fields = [SearchField {
            priority: 0,
            value: "film - codex - ~/Dev/film - Terminal",
        }];
        let candidate = MetadataCandidate {
            key: "window:film",
            fields: &fields,
            score: 1.0,
        };
        let plain = MetadataQuery::new("fil")
            .unwrap()
            .with_typo_fallback(true)
            .search_rank(candidate)
            .unwrap();
        let punctuated = MetadataQuery::new("fil,")
            .unwrap()
            .with_typo_fallback(true)
            .search_rank(candidate)
            .unwrap();
        assert_eq!(plain, punctuated);
    }

    #[test]
    fn normalized_punctuation_query_beats_unrelated_fish_window() {
        let film_fields = [SearchField {
            priority: 0,
            value: "film - codex - ~/Dev/film - Terminal",
        }];
        let fish_fields = [SearchField {
            priority: 0,
            value: "~ - fish - Terminal",
        }];
        let film = MetadataCandidate {
            key: "window:film",
            fields: &film_fields,
            score: 1.0,
        };
        let fish = MetadataCandidate {
            key: "window:fish",
            fields: &fish_fields,
            score: 1.0,
        };
        let query = MetadataQuery::new("fil,").unwrap().with_typo_fallback(true);
        let mut matches = vec![
            (film, query.search_rank(film).unwrap()),
            (fish, query.search_rank(fish).unwrap()),
        ];
        query.sort_matches(&mut matches);
        assert_eq!(matches[0].0.key, "window:film");
    }

    #[test]
    fn two_character_queries_use_exact_matching_only() {
        let fuzzy_only_fields = [SearchField {
            priority: 0,
            value: "videos — pcmanfm | pcmanfm | /usr/bin/pcmanfm | ~",
        }];
        let exact_fields = [SearchField {
            priority: 0,
            value: "mpv media player",
        }];
        let fuzzy_only = MetadataCandidate {
            key: "window:pcmanfm",
            fields: &fuzzy_only_fields,
            score: 1.0,
        };
        let exact = MetadataCandidate {
            key: "window:mpv",
            fields: &exact_fields,
            score: 1.0,
        };
        let typo_query = MetadataQuery::new("mp").unwrap().with_typo_fallback(true);
        let exact_query = MetadataQuery::new("mp").unwrap();

        assert!(typo_query.search_rank(fuzzy_only).is_none());
        assert!(exact_query.search_rank(exact).is_some());
    }

    #[test]
    fn nearby_key_typo_beats_distant_key_typo_for_metadata() {
        let near_fields = [SearchField {
            priority: 0,
            value: "film - codex - ~/Dev/film - Terminal",
        }];
        let far_fields = [SearchField {
            priority: 0,
            value: "pilm - codex - ~/Dev/pilm - Terminal",
        }];
        let near = MetadataCandidate {
            key: "window:film",
            fields: &near_fields,
            score: 1.0,
        };
        let far = MetadataCandidate {
            key: "window:pilm",
            fields: &far_fields,
            score: 1.0,
        };
        let query = MetadataQuery::new("gilm").unwrap().with_typo_fallback(true);
        let mut matches = vec![
            (near, query.search_rank(near).unwrap()),
            (far, query.search_rank(far).unwrap()),
        ];
        query.sort_matches(&mut matches);
        assert_eq!(matches[0].0.key, "window:film");
    }

    #[test]
    fn fragmented_metadata_typo_beats_heavier_contiguous_typo_match() {
        let unsplash_fields = [SearchField {
            priority: 0,
            value: "Unsplash",
        }];
        let splurge_fields = [SearchField {
            priority: 0,
            value: "splurge",
        }];
        let unsplash = MetadataCandidate {
            key: "window:unsplash",
            fields: &unsplash_fields,
            score: 1.0,
        };
        let splurge = MetadataCandidate {
            key: "window:splurge",
            fields: &splurge_fields,
            score: 1.0,
        };
        let query = MetadataQuery::new("splyush")
            .unwrap()
            .with_typo_fallback(true);
        let mut matches = vec![
            (splurge, query.search_rank(splurge).unwrap()),
            (unsplash, query.search_rank(unsplash).unwrap()),
        ];
        query.sort_matches(&mut matches);
        assert_eq!(matches[0].0.key, "window:splurge");
    }

    #[test]
    fn metadata_typo_rejects_unrelated_long_title_token_match() {
        let list_fields = [
            SearchField {
                priority: 0,
                value: "List the purpose of all of the top subdirectories of the main root directory on Linux according to the file system hierarchy standard IP. - Google Search - Google Chrome",
            },
            SearchField {
                priority: 1,
                value: "google-chrome",
            },
            SearchField {
                priority: 6,
                value: "chrome",
            },
        ];
        let unsplash_fields = [SearchField {
            priority: 0,
            value: "mike-uderevsky-JuJYPgv4Rag-unsplash exported",
        }];
        let query = MetadataQuery::new("splyush")
            .unwrap()
            .with_typo_fallback(true);
        let list = MetadataCandidate {
            key: "window:list-google",
            fields: &list_fields,
            score: 1.0,
        };
        let unsplash = MetadataCandidate {
            key: "window:unsplash",
            fields: &unsplash_fields,
            score: 1.0,
        };
        assert!(query.search_rank(list).is_none());
        assert!(query.search_rank(unsplash).is_some());
    }

    #[test]
    fn explanation_reports_overall_and_per_term_matches() {
        let fields = [
            SearchField {
                priority: 0,
                value: "Film Torrent Streaming Manager",
            },
            SearchField {
                priority: 1,
                value: "manager helper",
            },
        ];
        let candidate = MetadataCandidate {
            key: "film-manager.desktop",
            fields: &fields,
            score: 1.0,
        };
        let explanation = MetadataQuery::new("film manager")
            .unwrap()
            .explain_candidate(candidate)
            .unwrap();
        assert_eq!(
            explanation.overall_field.value,
            "Film Torrent Streaming Manager"
        );
        assert_eq!(explanation.term_explanations.len(), 2);
        assert_eq!(explanation.term_explanations[0].query_term, "film");
        assert_eq!(
            explanation.term_explanations[0].field.unwrap().value,
            "Film Torrent Streaming Manager"
        );
        assert_eq!(explanation.term_explanations[1].query_term, "manager");
        assert_eq!(
            explanation.term_explanations[1].field.unwrap().value,
            "Film Torrent Streaming Manager"
        );
    }

    #[test]
    fn explanation_tracks_typo_term_match_provenance() {
        let fields = [SearchField {
            priority: 0,
            value: "whisper service python3",
        }];
        let candidate = MetadataCandidate {
            key: "whisper",
            fields: &fields,
            score: 1.0,
        };
        let explanation = MetadataQuery::new("whusper")
            .unwrap()
            .with_typo_fallback(true)
            .explain_candidate(candidate)
            .unwrap();
        assert_eq!(explanation.term_explanations.len(), 1);
        let term = &explanation.term_explanations[0];
        assert_eq!(term.query_term, "whusper");
        assert!(matches!(term.rank, Some(SearchRank::Typo(_))));
        assert_eq!(term.field.unwrap().value, "whisper service python3");
        assert_eq!(term.rank.as_ref().unwrap().provenance().field_index, 0);
    }

    #[test]
    fn metadata_typo_reproduce_user_bug() {
        let list_fields = [
            SearchField {
                priority: 0,
                value: "List the purpose of all of the top subdirectories of the main root directory on Linux according to the file system hierarchy standard IP. - Google Search - Google Chrome",
            },
            SearchField {
                priority: 1,
                value: "google-chrome",
            },
            SearchField {
                priority: 4,
                value: "chromium-chatbot --remote-debugging-port=9233 --start-minimized --no-first-run --no-default-browser-check --disable-background-timer-throttling --disable-renderer-backgrounding --disable-backgrounding-occluded-windows --disable-features=CalculateNativeWinOcclusion",
            },
            SearchField {
                priority: 5,
                value: "/opt/google/chrome/chrome --ozone-platform=wayland --disable-features=Vulkan --enable-gpu-rasterization --enable-zero-copy --ignore-gpu-blocklist --user-data-dir=/home/lewis/.config/chromium-chatbot --remote-debugging-port=9233 --start-minimized --no-first-run --no-default-browser-check --disable-background-timer-throttling --disable-renderer-backgrounding --disable-backgrounding-occluded-windows --disable-features=CalculateNativeWinOcclusion",
            },
            SearchField {
                priority: 6,
                value: "chrome",
            },
            SearchField {
                priority: 7,
                value: "~/Dev/chatbot",
            },
        ];
        let unsplash_fields = [SearchField {
            priority: 0,
            value: "*[mike-uderevsky-JuJYPgv4Rag-unsplash] (exported)-1.0 (RGB colour 8-bit non-linear integer, sRGB IEC61966-2.1, 1 layer)",
        }];
        let query = MetadataQuery::new("splyush")
            .unwrap()
            .with_typo_fallback(true);
        let list = MetadataCandidate {
            key: "window:list-google",
            fields: &list_fields,
            score: 1.0,
        };
        let unsplash = MetadataCandidate {
            key: "window:unsplash",
            fields: &unsplash_fields,
            score: 1.0,
        };

        let list_rank = query.search_rank(list);
        let unsplash_rank = query.search_rank(unsplash);
        println!("list_rank: {:?}", list_rank);
        println!("unsplash_rank: {:?}", unsplash_rank);

        assert!(list_rank.is_none());
        assert!(unsplash_rank.is_some());
    }

    #[test]
    fn metadata_typo_reproduce_unsplash_bug() {
        let list_fields = [
            SearchField {
                priority: 0,
                value: "List the purpose of all of the top subdirectories of the main root directory on Linux according to the file system hierarchy standard IP. - Google Search - Google Chrome",
            },
            SearchField {
                priority: 1,
                value: "google-chrome",
            },
            SearchField {
                priority: 4,
                value: "chromium-chatbot --remote-debugging-port=9233 --start-minimized --no-first-run --no-default-browser-check --disable-background-timer-throttling --disable-renderer-backgrounding --disable-backgrounding-occluded-windows --disable-features=CalculateNativeWinOcclusion",
            },
            SearchField {
                priority: 5,
                value: "/opt/google/chrome/chrome --ozone-platform=wayland --disable-features=Vulkan --enable-gpu-rasterization --enable-zero-copy --ignore-gpu-blocklist --user-data-dir=/home/lewis/.config/chromium-chatbot --remote-debugging-port=9233 --start-minimized --no-first-run --no-default-browser-check --disable-background-timer-throttling --disable-renderer-backgrounding --disable-backgrounding-occluded-windows --disable-features=CalculateNativeWinOcclusion",
            },
            SearchField {
                priority: 6,
                value: "chrome",
            },
            SearchField {
                priority: 7,
                value: "~/Dev/chatbot",
            },
        ];
        let unsplash_fields = [SearchField {
            priority: 0,
            value: "*[mike-uderevsky-JuJYPgv4Rag-unsplash] (exported)-1.0 (RGB colour 8-bit non-linear integer, sRGB IEC61966-2.1, 1 layer)",
        }];
        let query = MetadataQuery::new("unsplash").unwrap();
        let list = MetadataCandidate {
            key: "window:list-google",
            fields: &list_fields,
            score: 1.0,
        };
        let unsplash = MetadataCandidate {
            key: "window:unsplash",
            fields: &unsplash_fields,
            score: 1.0,
        };

        let list_rank = query.search_rank(list);
        let unsplash_rank = query.search_rank(unsplash);
        println!("list_rank: {:?}", list_rank);
        println!("unsplash_rank: {:?}", unsplash_rank);

        let mut matches = vec![
            (list, list_rank.unwrap()),
            (unsplash, unsplash_rank.unwrap()),
        ];
        query.sort_matches(&mut matches);

        println!("Sorted results:");
        for (candidate, rank) in &matches {
            println!("  key: {}, rank: {:?}", candidate.key, rank);
        }

        assert_eq!(matches[0].0.key, "window:unsplash");
    }

    #[test]
    fn metadata_typo_diagnostic() {
        let list_fields = [
            SearchField {
                priority: 0,
                value: "List the purpose of all of the top subdirectories of the main root directory on Linux according to the file system hierarchy standard IP. - Google Search - Google Chrome",
            },
            SearchField {
                priority: 1,
                value: "google-chrome",
            },
            SearchField {
                priority: 4,
                value: "chromium-chatbot --remote-debugging-port=9233 --start-minimized --no-first-run --no-default-browser-check --disable-background-timer-throttling --disable-renderer-backgrounding --disable-backgrounding-occluded-windows --disable-features=CalculateNativeWinOcclusion",
            },
            SearchField {
                priority: 5,
                value: "/opt/google/chrome/chrome --ozone-platform=wayland --disable-features=Vulkan --enable-gpu-rasterization --enable-zero-copy --ignore-gpu-blocklist --user-data-dir=/home/lewis/.config/chromium-chatbot --remote-debugging-port=9233 --start-minimized --no-first-run --no-default-browser-check --disable-background-timer-throttling --disable-renderer-backgrounding --disable-backgrounding-occluded-windows --disable-features=CalculateNativeWinOcclusion",
            },
            SearchField {
                priority: 6,
                value: "chrome",
            },
            SearchField {
                priority: 7,
                value: "~/Dev/chatbot",
            },
        ];
        let unsplash_fields = [SearchField {
            priority: 0,
            value: "*[mike-uderevsky-JuJYPgv4Rag-unsplash] (exported)-1.0 (RGB colour 8-bit non-linear integer, sRGB IEC61966-2.1, 1 layer)",
        }];
        let query_str = "splyush";
        let query = MetadataQuery::new(query_str)
            .unwrap()
            .with_typo_fallback(true);
        let list = MetadataCandidate {
            key: "window:list-google",
            fields: &list_fields,
            score: 0.0,
        };
        let unsplash = MetadataCandidate {
            key: "window:unsplash",
            fields: &unsplash_fields,
            score: 0.0,
        };

        let list_rank = query.search_rank(list);
        let unsplash_rank = query.search_rank(unsplash);
        assert!(list_rank.is_none());
        assert!(unsplash_rank.is_some());
    }

    #[test]
    fn test_fiom_fish_highlights() {
        let query = MetadataQuery::new("fiom").unwrap().with_typo_fallback(true);
        let fields = [SearchField {
            priority: 0,
            value: "fish",
        }];
        let cand = MetadataCandidate {
            key: "fish",
            fields: &fields,
            score: 0.0,
        };
        let (rank, highlights) = query.search_rank_with_highlights(cand).unwrap();
        let prepared_highlights = query.highlights_for_rank(cand, &rank);
        println!("rank: {:?}", rank);
        println!("highlights: {:?}", highlights);
        assert_eq!(highlights.len(), 1);
        assert_eq!(highlights[0].ranges, vec![(0, 4, false)]);
        assert_eq!(prepared_highlights, highlights);
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MatchedFieldHighlight {
    pub field_index: usize,
    pub ranges: Vec<(usize, usize, bool)>, // (start, end, is_exact)
    pub focus_range: Option<(usize, usize)>,
}

struct CharToken {
    start: usize,
    _end: usize,
    value: String,
}

fn alnum_tokens_with_char_ranges(text: &str) -> Vec<CharToken> {
    let mut tokens = Vec::new();
    let mut token_start = None;
    for (char_idx, ch) in text.chars().enumerate() {
        if ch.is_alphanumeric() {
            token_start.get_or_insert(char_idx);
            continue;
        }
        if let Some(start) = token_start.take() {
            let end = char_idx;
            let val = text
                .chars()
                .skip(start)
                .take(end - start)
                .collect::<String>()
                .to_lowercase();
            if !val.is_empty() {
                tokens.push(CharToken {
                    start,
                    _end: end,
                    value: val,
                });
            }
        }
    }
    if let Some(start) = token_start {
        let end = text.chars().count();
        let val = text
            .chars()
            .skip(start)
            .take(end - start)
            .collect::<String>()
            .to_lowercase();
        if !val.is_empty() {
            tokens.push(CharToken {
                start,
                _end: end,
                value: val,
            });
        }
    }
    tokens
}

fn char_indices_to_byte_range(text: &str, start_char: usize, end_char: usize) -> (usize, usize) {
    let mut byte_start = 0;
    let mut byte_end = 0;
    for (char_idx, (byte_idx, _ch)) in text.char_indices().enumerate() {
        if char_idx == start_char {
            byte_start = byte_idx;
        }
        if char_idx == end_char {
            byte_end = byte_idx;
            break;
        }
    }
    if end_char >= text.chars().count() {
        byte_end = text.len();
    }
    (byte_start, byte_end)
}

fn typo_match_token_slice(
    candidate_token: &str,
    query_token: &str,
) -> Option<(usize, usize, TokenTypoSeverity)> {
    if let Some(start_byte) = candidate_token.find(query_token) {
        let start_char = candidate_token[..start_byte].chars().count();
        return Some((
            start_char,
            start_char + query_token.chars().count(),
            TokenTypoSeverity::default(),
        ));
    }
    let query_len = query_token.chars().count();
    if query_len < 3 || !token_has_signal(query_token, candidate_token) {
        return None;
    }

    let limit = max_typos(query_len).max(1);
    let contiguous = if let Some((_, _, start_idx, end_idx)) =
        fuzzy_match_details(query_token, candidate_token)
    {
        let candidate_chars: Vec<char> = candidate_token.chars().collect();
        let end = end_idx.min(candidate_chars.len());
        let start_min = start_idx.saturating_sub(limit).min(end);
        let start_max = (start_idx + limit).min(end);
        let mut best_severity: Option<(usize, usize, TokenTypoSeverity)> = None;
        for s in start_min..=start_max {
            let sliced: String = candidate_chars[s..end].iter().collect();
            if let Some((_distance, operations)) =
                bounded_damerau_levenshtein(&sliced, query_token, limit)
            {
                let position = if s == 0 {
                    0
                } else if end == candidate_chars.len() {
                    1
                } else {
                    2
                };
                let severity = TokenTypoSeverity::new(
                    operations.core_typo_cost(),
                    position,
                    operations.insertions,
                );
                if best_severity
                    .as_ref()
                    .is_none_or(|(_, _, best)| severity < *best)
                {
                    best_severity = Some((s, end, severity));
                }
            }
        }
        best_severity
    } else {
        None
    };

    let fragmented = if candidate_token
        .chars()
        .count()
        .saturating_sub(query_token.chars().count())
        > limit
    {
        fragmented_token_match(query_token, candidate_token, limit).and_then(|matched| {
            let ratio = matched.distance as f64
                / query_token
                    .chars()
                    .count()
                    .max(candidate_token.chars().count()) as f64;
            if ratio > 0.5 || matched.internal_breaks == 0 {
                return None;
            }
            let severity = TokenTypoSeverity::new(
                matched.operations.core_typo_cost(),
                matched.position_rank,
                matched.operations.insertions,
            );
            Some((0, candidate_token.chars().count(), severity))
        })
    } else {
        None
    };

    match (contiguous, fragmented) {
        (Some(left), Some(right)) => Some(if right.2 < left.2 { right } else { left }),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}
