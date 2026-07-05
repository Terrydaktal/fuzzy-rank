use crate::core::{bounded_damerau_levenshtein, keyboard_substitution_cost, max_typos};
use crate::token_match::fragmented_token_match;
use crate::ranking::{
    AbbreviationRank, DistanceRank, SearchRank, StructuralRank, compare_search_results, ratio_milli,
};
use crate::text::{normalize_compact_alnum, split_search_tokens, to_lowercase, token_has_signal};
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

pub struct MetadataQuery {
    query: String,
    query_tokens: Vec<String>,
    allow_typo: bool,
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
        let queries = if self.query_tokens.len() <= 1 {
            vec![self.query.as_str()]
        } else {
            self.query_tokens.iter().map(String::as_str).collect()
        };

        let mut best_rank = None;
        for query in queries {
            let rank = if self.allow_typo {
                best_typo_rank(query, candidate.fields).map(SearchRank::Typo)
            } else {
                best_structural_rank(query, candidate.fields)
                    .map(SearchRank::Structural)
                    .or_else(|| best_abbreviation_rank(query, candidate.fields).map(SearchRank::Abbreviation))
                    .or_else(|| best_fuzzy_rank(query, candidate.fields).map(SearchRank::Fuzzy))
            };
            if let Some(rank) = rank
                && best_rank.as_ref().is_none_or(|current| rank < *current)
            {
                best_rank = Some(rank);
            }
        }
        best_rank
    }

    pub fn compare_candidates(
        &self,
        left_candidate: MetadataCandidate<'_>,
        left_rank: &SearchRank,
        right_candidate: MetadataCandidate<'_>,
        right_rank: &SearchRank,
    ) -> Ordering {
        metadata_relevance_rank(&self.query_tokens, self.allow_typo, left_candidate.fields)
            .cmp(&metadata_relevance_rank(
                &self.query_tokens,
                self.allow_typo,
                right_candidate.fields,
            ))
            .then_with(|| {
                right_candidate
                    .score
                    .total_cmp(&left_candidate.score)
                    .then_with(|| left_rank.cmp(right_rank))
                    .then_with(|| left_candidate.key.cmp(right_candidate.key))
            })
    }

    pub fn sort_matches(&self, matches: &mut [(MetadataCandidate<'_>, SearchRank)]) {
        matches.sort_unstable_by(|(left_candidate, left_rank), (right_candidate, right_rank)| {
            self.compare_candidates(*left_candidate, left_rank, *right_candidate, right_rank)
        });
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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct TypoSeverity {
    core_cost: usize,
    internal_breaks: usize,
    insertions: usize,
    distance: usize,
}

impl Ord for TypoSeverity {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.core_cost, self.internal_breaks, self.insertions, self.distance).cmp(&(
            other.core_cost,
            other.internal_breaks,
            other.insertions,
            other.distance,
        ))
    }
}

impl PartialOrd for TypoSeverity {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct MetadataRelevanceRank {
    exact_whole_phrase_occurrences: usize,
    exact_run_vector: Vec<usize>,
    exact_distinct_coverage: usize,
    fuzzy_whole_phrase: bool,
    fuzzy_run_vector: Vec<usize>,
    total_distinct_coverage: usize,
    typo_severity: TypoSeverity,
    all_query_tokens_matched: bool,
    field_quality: u8,
    same_order_scattered: bool,
    closeness_gap_distance: usize,
    repeated_exact_phrase_bonus: usize,
    repeated_token_bonus: usize,
    density_matched_tokens: usize,
    density_field_len: usize,
}

impl Ord for MetadataRelevanceRank {
    fn cmp(&self, other: &Self) -> Ordering {
        Reverse(self.exact_whole_phrase_occurrences)
            .cmp(&Reverse(other.exact_whole_phrase_occurrences))
            .then_with(|| Reverse(&self.exact_run_vector).cmp(&Reverse(&other.exact_run_vector)))
            .then_with(|| Reverse(self.exact_distinct_coverage).cmp(&Reverse(other.exact_distinct_coverage)))
            .then_with(|| Reverse(self.fuzzy_whole_phrase).cmp(&Reverse(other.fuzzy_whole_phrase)))
            .then_with(|| Reverse(&self.fuzzy_run_vector).cmp(&Reverse(&other.fuzzy_run_vector)))
            .then_with(|| Reverse(self.total_distinct_coverage).cmp(&Reverse(other.total_distinct_coverage)))
            .then_with(|| self.typo_severity.cmp(&other.typo_severity))
            .then_with(|| Reverse(self.all_query_tokens_matched).cmp(&Reverse(other.all_query_tokens_matched)))
            .then_with(|| self.field_quality.cmp(&other.field_quality))
            .then_with(|| Reverse(self.same_order_scattered).cmp(&Reverse(other.same_order_scattered)))
            .then_with(|| self.closeness_gap_distance.cmp(&other.closeness_gap_distance))
            .then_with(|| Reverse(self.repeated_exact_phrase_bonus).cmp(&Reverse(other.repeated_exact_phrase_bonus)))
            .then_with(|| Reverse(self.repeated_token_bonus).cmp(&Reverse(other.repeated_token_bonus)))
            .then_with(|| {
                (other.density_matched_tokens * self.density_field_len)
                    .cmp(&(self.density_matched_tokens * other.density_field_len))
            })
            .then_with(|| self.density_field_len.cmp(&other.density_field_len))
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

#[derive(Clone, Debug)]
struct NormalizedMetadataField {
    priority: u8,
    tokens: Vec<String>,
}

#[derive(Clone, Copy, Debug, Default)]
struct TokenMatch {
    exact: bool,
    fuzzy: bool,
    severity: TypoSeverity,
}

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
            (!tokens.is_empty()).then_some(NormalizedMetadataField {
                priority: field.priority,
                tokens,
            })
        })
        .collect::<Vec<_>>();

    if query_tokens.is_empty() || normalized_fields.is_empty() {
        return MetadataRelevanceRank {
            exact_whole_phrase_occurrences: 0,
            exact_run_vector: Vec::new(),
            exact_distinct_coverage: 0,
            fuzzy_whole_phrase: false,
            fuzzy_run_vector: Vec::new(),
            total_distinct_coverage: 0,
            typo_severity: TypoSeverity::default(),
            all_query_tokens_matched: false,
            field_quality: u8::MAX,
            same_order_scattered: false,
            closeness_gap_distance: usize::MAX,
            repeated_exact_phrase_bonus: 0,
            repeated_token_bonus: 0,
            density_matched_tokens: 0,
            density_field_len: usize::MAX,
        };
    }

    let query_len = query_tokens.len();
    let exact_occurrence_counts = query_tokens
        .iter()
        .map(|query_token| {
            normalized_fields
                .iter()
                .map(|field| {
                    field
                        .tokens
                        .iter()
                        .filter(|candidate_token| candidate_token.starts_with(query_token))
                        .count()
                })
                .sum::<usize>()
        })
        .collect::<Vec<_>>();

    let exact_token_matches = query_tokens
        .iter()
        .map(|query_token| {
            normalized_fields
                .iter()
                .flat_map(|field| field.tokens.iter())
                .any(|candidate_token| candidate_token.starts_with(query_token))
        })
        .collect::<Vec<_>>();

    let fuzzy_best_matches = query_tokens
        .iter()
        .map(|query_token| best_fuzzy_token_match(query_token, &normalized_fields, allow_typo))
        .collect::<Vec<_>>();

    let exact_whole_phrase_occurrences: usize = normalized_fields
        .iter()
        .map(|field| exact_phrase_window_count(query_tokens, &field.tokens))
        .sum();

    let exact_run_vector = phrase_run_vector(query_tokens, &normalized_fields, MatchMode::Exact);
    let fuzzy_phrase_vector = phrase_run_vector(query_tokens, &normalized_fields, MatchMode::Fuzzy);
    let fuzzy_whole_phrase = fuzzy_phrase_vector
        .first()
        .copied()
        .unwrap_or_default()
        > 0;
    let fuzzy_run_vector = fuzzy_phrase_vector.into_iter().skip(1).collect::<Vec<_>>();

    let exact_distinct_coverage = exact_token_matches.iter().filter(|matched| **matched).count();
    let total_distinct_coverage = fuzzy_best_matches
        .iter()
        .filter(|matched| matched.exact || matched.fuzzy)
        .count();

    let typo_severity = fuzzy_best_matches
        .iter()
        .filter(|matched| matched.fuzzy && !matched.exact)
        .fold(TypoSeverity::default(), |acc, matched| TypoSeverity {
            core_cost: acc.core_cost + matched.severity.core_cost,
            internal_breaks: acc.internal_breaks + matched.severity.internal_breaks,
            insertions: acc.insertions + matched.severity.insertions,
            distance: acc.distance + matched.severity.distance,
        });

    let all_query_tokens_matched = total_distinct_coverage == query_len;

    let field_quality = best_field_priority(query_tokens, &normalized_fields, allow_typo);
    let (same_order_scattered, closeness_gap_distance) =
        same_order_scattered_stats(query_tokens, &normalized_fields, allow_typo);

    let repeated_exact_phrase_bonus = exact_whole_phrase_occurrences.min(3);
    let repeated_token_bonus = exact_occurrence_counts
        .iter()
        .map(|count| count.min(&3))
        .sum::<usize>();

    let (density_matched_tokens, density_field_len) =
        density_stats(query_tokens, &normalized_fields, &fuzzy_best_matches);

    MetadataRelevanceRank {
        exact_whole_phrase_occurrences,
        exact_run_vector,
        exact_distinct_coverage,
        fuzzy_whole_phrase,
        fuzzy_run_vector,
        total_distinct_coverage,
        typo_severity,
        all_query_tokens_matched,
        field_quality,
        same_order_scattered,
        closeness_gap_distance,
        repeated_exact_phrase_bonus,
        repeated_token_bonus,
        density_matched_tokens,
        density_field_len,
    }
}

#[derive(Clone, Copy)]
enum MatchMode {
    Exact,
    Fuzzy,
}

fn exact_phrase_window_count(query_tokens: &[String], field_tokens: &[String]) -> usize {
    if query_tokens.is_empty() || field_tokens.len() < query_tokens.len() {
        return 0;
    }
    field_tokens
        .windows(query_tokens.len())
        .filter(|window| {
            window
                .iter()
                .zip(query_tokens)
                .all(|(candidate, query)| candidate.starts_with(query))
        })
        .count()
}

fn phrase_run_vector(
    query_tokens: &[String],
    fields: &[NormalizedMetadataField],
    mode: MatchMode,
) -> Vec<usize> {
    if query_tokens.is_empty() {
        return Vec::new();
    }
    (2..=query_tokens.len())
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
                        .filter(|candidate_window| match mode {
                            MatchMode::Exact => candidate_window
                                .iter()
                                .zip(query_window.iter())
                                .all(|(candidate, query)| candidate.starts_with(query)),
                            MatchMode::Fuzzy => candidate_window
                                .iter()
                                .zip(query_window.iter())
                                .all(|(candidate, query)| fuzzy_or_exact_token_match(candidate, query).is_some()),
                        })
                        .count();
                }
            }
            count
        })
        .collect()
}

fn best_fuzzy_token_match(
    query_token: &str,
    fields: &[NormalizedMetadataField],
    allow_typo: bool,
) -> TokenMatch {
    let mut best = TokenMatch::default();
    for field in fields {
        for candidate_token in &field.tokens {
            if candidate_token.starts_with(query_token) {
                return TokenMatch {
                    exact: true,
                    fuzzy: true,
                    severity: TypoSeverity::default(),
                };
            }
            if !allow_typo {
                continue;
            }
            let Some(severity) = fuzzy_or_exact_token_match(candidate_token, query_token) else {
                continue;
            };
            let candidate = TokenMatch {
                exact: false,
                fuzzy: true,
                severity,
            };
            if !best.fuzzy || candidate.severity < best.severity {
                best = candidate;
            }
        }
    }
    best
}

fn fuzzy_or_exact_token_match(candidate_token: &str, query_token: &str) -> Option<TypoSeverity> {
    if candidate_token.starts_with(query_token) {
        return Some(TypoSeverity::default());
    }
    if !token_has_signal(query_token, candidate_token) {
        return None;
    }

    let limit = max_typos(query_token.chars().count()).max(1);
    let contiguous = bounded_damerau_levenshtein(candidate_token, query_token, limit).map(
        |(distance, operations)| TypoSeverity {
            core_cost: operations.core_typo_cost(),
            internal_breaks: 0,
            insertions: operations.insertions,
            distance,
        },
    );

    let fragmented = if candidate_token.chars().count().saturating_sub(query_token.chars().count()) > limit {
        fragmented_token_match(query_token, candidate_token, limit).and_then(|matched| {
            let ratio = matched.distance as f64
                / query_token.chars().count().max(candidate_token.chars().count()) as f64;
            if ratio > 0.5 || matched.internal_breaks == 0 {
                return None;
            }
            Some(TypoSeverity {
                core_cost: matched.operations.core_typo_cost(),
                internal_breaks: matched.internal_breaks,
                insertions: matched.operations.insertions,
                distance: matched.distance,
            })
        })
    } else {
        None
    };

    match (contiguous, fragmented) {
        (Some(left), Some(right)) => Some(if right < left { right } else { left }),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

fn best_field_priority(
    query_tokens: &[String],
    fields: &[NormalizedMetadataField],
    allow_typo: bool,
) -> u8 {
    fields
        .iter()
        .filter(|field| {
            field.tokens.iter().any(|candidate_token| {
                query_tokens.iter().any(|query_token| {
                    candidate_token.starts_with(query_token)
                        || (allow_typo && fuzzy_or_exact_token_match(candidate_token, query_token).is_some())
                })
            })
        })
        .map(|field| field.priority)
        .min()
        .unwrap_or(u8::MAX)
}

fn same_order_scattered_stats(
    query_tokens: &[String],
    fields: &[NormalizedMetadataField],
    allow_typo: bool,
) -> (bool, usize) {
    let mut found = false;
    let mut best_gap = usize::MAX;
    for field in fields {
        let mut positions = Vec::new();
        let mut search_from = 0usize;
        for query_token in query_tokens {
            let Some((idx, _)) = field
                .tokens
                .iter()
                .enumerate()
                .skip(search_from)
                .find(|(_, candidate_token)| {
                    candidate_token.starts_with(query_token)
                        || (allow_typo && fuzzy_or_exact_token_match(candidate_token, query_token).is_some())
                })
            else {
                positions.clear();
                break;
            };
            positions.push(idx);
            search_from = idx + 1;
        }
        if positions.len() == query_tokens.len() {
            found = true;
            let span = positions.last().copied().unwrap_or(0) - positions.first().copied().unwrap_or(0);
            let gap = span.saturating_sub(query_tokens.len().saturating_sub(1));
            best_gap = best_gap.min(gap);
        }
    }
    (found, best_gap)
}

fn density_stats(
    query_tokens: &[String],
    fields: &[NormalizedMetadataField],
    fuzzy_matches: &[TokenMatch],
) -> (usize, usize) {
    let matched_count = fuzzy_matches.iter().filter(|matched| matched.exact || matched.fuzzy).count();
    let best_field_len = fields
        .iter()
        .filter(|field| {
            field.tokens.iter().any(|candidate_token| {
                query_tokens
                    .iter()
                    .any(|query_token| candidate_token.starts_with(query_token))
            })
        })
        .map(|field| field.tokens.len())
        .min()
        .unwrap_or_else(|| fields.iter().map(|field| field.tokens.len()).min().unwrap_or(usize::MAX));
    (matched_count, best_field_len)
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

fn fuzzy_match_details(query: &str, target: &str) -> Option<(usize, usize, usize)> {
    let query_lower = to_lowercase(query);
    let target_lower = to_lowercase(target);

    let q_len = query_lower.chars().count();
    if q_len == 0 {
        return Some((0, 0, 0));
    }

    let t_len = target_lower.chars().count();
    let limit = q_len / 2;

    if t_len < q_len.saturating_sub(limit) {
        return None;
    }

    let query_chars: Vec<char> = query_lower.chars().collect();
    let target_chars: Vec<char> = target_lower.chars().collect();

    let inf = MetadataEditCost::infinite(limit);
    let mut prev_prev = vec![inf; t_len + 1];
    let mut prev = vec![MetadataEditCost::default(); t_len + 1];
    for cell in prev.iter_mut() {
        *cell = MetadataEditCost {
            distance: 0,
            keyboard_distance: 0,
        };
    }
    let mut curr = vec![inf; t_len + 1];

    for i in 1..=q_len {
        curr.fill(inf);
        curr[0] = MetadataEditCost {
            distance: i,
            keyboard_distance: i,
        };

        let mut row_min = inf;
        for j in 1..=t_len {
            let deletion = MetadataEditCost {
                distance: prev[j].distance + 1,
                keyboard_distance: prev[j].keyboard_distance + 1,
            };
            let insertion = MetadataEditCost {
                distance: curr[j - 1].distance + 1,
                keyboard_distance: curr[j - 1].keyboard_distance + 1,
            };
            let substitution = if query_chars[i - 1] == target_chars[j - 1] {
                prev[j - 1]
            } else {
                MetadataEditCost {
                    distance: prev[j - 1].distance + 1,
                    keyboard_distance: prev[j - 1].keyboard_distance
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
                    distance: prev_prev[j - 2].distance + 1,
                    keyboard_distance: prev_prev[j - 2].keyboard_distance,
                });
            }

            curr[j] = cell;
            row_min = row_min.min(cell);
        }

        if row_min.distance > limit {
            return None;
        }

        std::mem::swap(&mut prev_prev, &mut prev);
        std::mem::swap(&mut prev, &mut curr);
    }

    let mut min_cost = inf;
    let mut best_end_idx = 0;
    let search_start = q_len.saturating_sub(limit).max(1);
    for (j, value) in prev.iter().enumerate().take(t_len + 1).skip(search_start) {
        if *value < min_cost {
            min_cost = *value;
            best_end_idx = j;
        }
    }

    (min_cost.distance <= limit).then_some((
        min_cost.distance,
        min_cost.keyboard_distance,
        best_end_idx.saturating_sub(q_len),
    ))
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

fn best_structural_rank(query: &str, fields: &[SearchField<'_>]) -> Option<StructuralRank> {
    let query_lower = to_lowercase(query.trim());
    if query_lower.is_empty() {
        return None;
    }

    let query_len = query_lower.chars().count();
    let mut best_rank = None;

    for (field_index, field) in fields.iter().enumerate() {
        let target = field.value.trim();
        if target.is_empty() {
            continue;
        }

        let target_lower = to_lowercase(target);
        for (start_byte_idx, _) in target_lower.match_indices(&query_lower) {
            let start_idx = target_lower[..start_byte_idx].chars().count();
            let (position_class, token_index, token_span_delta, start_idx, field_len) =
                structural_match_details(&target_lower, start_idx, query_len);
            let rank = StructuralRank {
                field_index,
                field_priority: field.priority,
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

fn best_abbreviation_rank(query: &str, fields: &[SearchField<'_>]) -> Option<AbbreviationRank> {
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
        let target = field.value.trim();
        if target.is_empty() {
            continue;
        }

        let target_lower = to_lowercase(target);
        let tokens = split_search_tokens(&target_lower);
        let mut candidates: Vec<(u8, String)> = Vec::new();

        for token in &tokens {
            candidates.push((0, (*token).to_string()));
        }
        candidates.push((1, target_lower.clone()));
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
                field_priority: field.priority,
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

fn best_fuzzy_rank(query: &str, fields: &[SearchField<'_>]) -> Option<DistanceRank> {
    let query_lower = to_lowercase(query.trim());
    if query_lower.is_empty() {
        return None;
    }

    let query_collapsed = normalize_compact_alnum(&query_lower);
    let allow_collapsed =
        !query_collapsed.is_empty() && !query_lower.chars().any(is_search_token_separator);
    let mut best_rank = None;

    for (field_index, field) in fields.iter().enumerate() {
        let target = field.value.trim();
        if target.is_empty() {
            continue;
        }

        let target_lower = to_lowercase(target);
        let tokens = split_search_tokens(&target_lower);
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
            let Some((distance, keyboard_distance, start_idx)) =
                fuzzy_match_details(&candidate_query, &candidate_target)
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
                field_priority: field.priority,
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

fn best_typo_rank(query: &str, fields: &[SearchField<'_>]) -> Option<DistanceRank> {
    let query_lower = to_lowercase(query.trim());
    if query_lower.is_empty() {
        return None;
    }

    let query_collapsed = normalize_compact_alnum(&query_lower);
    let allow_collapsed =
        !query_collapsed.is_empty() && !query_lower.chars().any(is_search_token_separator);
    let mut best_rank = None;

    for (field_index, field) in fields.iter().enumerate() {
        let target = field.value.trim();
        if target.is_empty() {
            continue;
        }

        let target_lower = to_lowercase(target);
        let tokens = split_search_tokens(&target_lower);
        let mut candidates: Vec<(u8, String, String)> =
            vec![(0, query_lower.clone(), target_lower.clone())];

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
            let Some((distance, keyboard_distance, start_idx)) =
                fuzzy_match_details(&candidate_query, &candidate_target)
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
                field_priority: field.priority,
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
        MetadataQuery::new("git").unwrap().sort_matches(&mut matches);
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
        assert!(MetadataQuery::new("term1 term2 term3 term4")
            .unwrap()
            .search_rank(candidate)
            .is_some());
    }

    #[test]
    fn metadata_phrase_coverage_orders_longer_query_windows_first() {
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
            matches.iter().map(|(candidate, _)| candidate.key).collect::<Vec<_>>(),
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
    fn typo_search_handles_multibyte_separator_before_match() {
        let fields = [SearchField {
            priority: 0,
            value: "videos — pcmanfm | pcmanfm | /usr/bin/pcmanfm | ~",
        }];
        let candidate = MetadataCandidate {
            key: "window:pcmanfm",
            fields: &fields,
            score: 1.0,
        };
        let rank = MetadataQuery::new("mp")
            .unwrap()
            .with_typo_fallback(true)
            .search_rank(candidate);

        assert!(rank.is_some());
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
        let query = MetadataQuery::new("splyush").unwrap().with_typo_fallback(true);
        let mut matches = vec![
            (splurge, query.search_rank(splurge).unwrap()),
            (unsplash, query.search_rank(unsplash).unwrap()),
        ];
        query.sort_matches(&mut matches);
        assert_eq!(matches[0].0.key, "window:unsplash");
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
}
