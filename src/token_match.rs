use std::collections::HashMap;

use crate::core::{
    AlignmentCostVector, DamerauScratch, InsertionVector, OperationProfile, TypoCostVector,
    alignment_cost_milli, bounded_damerau_levenshtein_chars_with_scratch,
    increment_simple_position_vector, max_typos,
};
use crate::ranking::{PathTypoSortKey, compare_path_typo_sort_keys};

#[derive(Clone, Copy)]
pub struct PositionedToken<'a> {
    pub token: &'a str,
    pub segment: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FragmentedTokenMatch {
    pub distance: usize,
    pub internal_breaks: usize,
    pub operations: OperationProfile,
    pub position_rank: usize,
}

#[derive(Default)]
pub(crate) struct TokenMatchScratch {
    candidate_chars: Vec<char>,
    damerau: DamerauScratch,
    fragmented: Vec<Vec<[Option<FragmentCost>; 5]>>,
}

#[derive(Default)]
pub(crate) struct SingleTokenMatchScratch {
    query: Vec<char>,
    limit: usize,
    choices: HashMap<String, Option<TokenChoice>>,
    token: TokenMatchScratch,
}

impl SingleTokenMatchScratch {
    pub(crate) fn choice(
        &mut self,
        query: &[char],
        candidate: &str,
        limit: usize,
    ) -> Option<TokenChoice> {
        if self.limit != limit || self.query != query {
            self.query.clear();
            self.query.extend_from_slice(query);
            self.limit = limit;
            self.choices.clear();
        }
        if let Some(choice) = self.choices.get(candidate) {
            return *choice;
        }
        let choice = choose_path_token_match_with_scratch(query, candidate, limit, &mut self.token);
        if self.choices.len() < 8192 {
            self.choices.insert(candidate.to_owned(), choice);
        }
        choice
    }
}

#[derive(Default)]
pub(crate) struct SequenceMatchScratch {
    query: Vec<String>,
    query_chars: Vec<Vec<char>>,
    limit: usize,
    choices: HashMap<String, HashMap<(usize, usize), Option<TokenChoice>>>,
    compounds: HashMap<String, Option<PartitionedTokenDistance>>,
    partitions: HashMap<(usize, usize, usize), Option<PartitionedTokenDistance>>,
    boundaries: Vec<usize>,
    token: TokenMatchScratch,
}

impl SequenceMatchScratch {
    fn prepare(&mut self, query: &[&str], limit: usize) {
        if self.limit == limit
            && self
                .query
                .iter()
                .map(String::as_str)
                .eq(query.iter().copied())
        {
            return;
        }
        self.query = query.iter().map(|token| (*token).to_owned()).collect();
        self.query_chars = query.iter().map(|token| token.chars().collect()).collect();
        self.limit = limit;
        self.choices.clear();
        self.compounds.clear();
    }

    fn choice(&mut self, query_idx: usize, candidate: &str, limit: usize) -> Option<TokenChoice> {
        let key = (query_idx, limit);
        if let Some(choice) = self
            .choices
            .get(candidate)
            .and_then(|choices| choices.get(&key))
        {
            return *choice;
        }
        let choice = choose_path_token_match_with_scratch(
            &self.query_chars[query_idx],
            candidate,
            limit,
            &mut self.token,
        );
        if let Some(choices) = self.choices.get_mut(candidate) {
            choices.insert(key, choice);
        } else if self.choices.len() < 8192 {
            self.choices
                .entry(candidate.to_owned())
                .or_default()
                .insert(key, choice);
        }
        choice
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TokenChoice {
    pub(crate) distance: usize,
    pub(crate) penalty: usize,
    pub(crate) position_rank: usize,
    pub(crate) internal_breaks: usize,
    pub(crate) operations: OperationProfile,
    pub(crate) fragmented: bool,
    pub(crate) alignment_cost: usize,
}

pub type AlignedTokenDistance = (
    usize,
    usize,
    Vec<usize>,
    [usize; 3],
    usize,
    usize,
    OperationProfile,
    Vec<usize>,
    Vec<usize>,
    Vec<usize>,
);
pub type PartitionedTokenDistance = (
    usize,
    usize,
    [usize; 3],
    OperationProfile,
    Vec<usize>,
    Vec<usize>,
    Vec<usize>,
);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FragmentCost {
    distance: usize,
    internal_breaks: usize,
    operations: OperationProfile,
}

impl FragmentCost {
    fn sort_key(&self, position_rank: usize) -> (usize, usize, usize, usize, OperationProfile) {
        (
            self.operations.core_typo_cost(),
            self.internal_breaks,
            self.operations.insertions,
            position_rank,
            self.operations,
        )
    }
}

#[cfg(test)]
pub(crate) fn choose_path_token_match(
    query: &str,
    candidate: &str,
    limit: usize,
) -> Option<TokenChoice> {
    let mut scratch = TokenMatchScratch::default();
    let query_chars = query.chars().collect::<Vec<_>>();
    choose_path_token_match_with_scratch(&query_chars, candidate, limit, &mut scratch)
}

pub(crate) fn choose_path_token_match_with_scratch(
    query_chars: &[char],
    candidate: &str,
    limit: usize,
    scratch: &mut TokenMatchScratch,
) -> Option<TokenChoice> {
    let contiguous = best_token_match_chars_with_scratch(query_chars, candidate, limit, scratch)
        .map(|(distance, penalty, position_rank, _len, operations)| {
            let alignment_cost = alignment_cost_milli(operations, false, 0);
            let operations = OperationProfile {
                insertions: operations.insertions + penalty,
                ..operations
            };
            TokenChoice {
                distance,
                penalty,
                position_rank,
                internal_breaks: 0,
                operations,
                fragmented: false,
                alignment_cost,
            }
        });
    if contiguous
        .as_ref()
        .is_some_and(|choice| choice.alignment_cost <= 1_000)
    {
        return contiguous;
    }

    let fragmented = {
        let local_limit = limit.min(max_typos(query_chars.len()));
        fragmented_token_match_chars(query_chars, candidate, local_limit, scratch).map(|matched| {
            let penalty = candidate.chars().count().saturating_sub(query_chars.len());
            TokenChoice {
                distance: matched.distance,
                penalty,
                position_rank: matched.position_rank,
                internal_breaks: matched.internal_breaks,
                operations: matched.operations,
                fragmented: true,
                alignment_cost: alignment_cost_milli(
                    matched.operations,
                    true,
                    matched.internal_breaks,
                ),
            }
        })
    };

    match (contiguous, fragmented) {
        (Some(left), Some(right)) => {
            let left_key = (
                left.alignment_cost,
                left.operations.core_typo_cost(),
                left.position_rank,
                left.internal_breaks,
                left.operations.insertions,
                left.distance,
                left.operations,
            );
            let right_key = (
                right.alignment_cost,
                right.operations.core_typo_cost(),
                right.position_rank,
                right.internal_breaks,
                right.operations.insertions,
                right.distance,
                right.operations,
            );
            Some(if right_key < left_key { right } else { left })
        }
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

pub fn aligned_token_distance(
    query_tokens: &[&str],
    candidate_tokens: &[PositionedToken<'_>],
    limit: usize,
) -> Option<AlignedTokenDistance> {
    aligned_token_distance_with_scratch(
        query_tokens,
        candidate_tokens,
        limit,
        &mut SequenceMatchScratch::default(),
    )
}

pub(crate) fn aligned_token_distance_with_scratch(
    query_tokens: &[&str],
    candidate_tokens: &[PositionedToken<'_>],
    limit: usize,
    scratch: &mut SequenceMatchScratch,
) -> Option<AlignedTokenDistance> {
    if candidate_tokens.len() < query_tokens.len() {
        return None;
    }

    scratch.prepare(query_tokens, limit);
    let mut edit_bounds = Vec::with_capacity(candidate_tokens.len() * query_tokens.len());
    for candidate in candidate_tokens {
        scratch.token.candidate_chars.clear();
        scratch
            .token
            .candidate_chars
            .extend(candidate.token.chars());
        for query in &scratch.query_chars {
            edit_bounds.push(character_deficit_lower_bound(
                query,
                &scratch.token.candidate_chars,
            ));
        }
    }
    let mut search = AssignmentSearch {
        query_tokens,
        candidate_tokens,
        limit,
        best: None,
        assignment: Vec::with_capacity(query_tokens.len()),
        used: vec![false; query_tokens.len()],
        edit_bounds,
        scratch,
    };
    for start in 0..=candidate_tokens.len() - query_tokens.len() {
        search.search_window(start);
        if search.is_perfect(query_tokens.len()) {
            break;
        }
    }

    search.best
}

struct AssignmentSearch<'a, 's> {
    query_tokens: &'a [&'a str],
    candidate_tokens: &'a [PositionedToken<'a>],
    limit: usize,
    best: Option<AlignedTokenDistance>,
    assignment: Vec<usize>,
    used: Vec<bool>,
    edit_bounds: Vec<usize>,
    scratch: &'s mut SequenceMatchScratch,
}

impl AssignmentSearch<'_, '_> {
    fn search_window(&mut self, start: usize) {
        let len = self.query_tokens.len();
        self.assignment.clear();
        self.used.fill(false);

        let mut segment = Vec::new();
        let mut position = [0; 3];
        let mut typo_costs = Vec::with_capacity(len);
        let mut insertions = Vec::with_capacity(len);
        let mut alignment_costs = Vec::with_capacity(len);
        self.search(
            start,
            0,
            0,
            0,
            &mut segment,
            &mut position,
            0,
            0,
            OperationProfile::default(),
            &mut typo_costs,
            &mut insertions,
            &mut alignment_costs,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn search(
        &mut self,
        window_start: usize,
        query_idx: usize,
        distance: usize,
        inversion: usize,
        segment: &mut Vec<usize>,
        position: &mut [usize; 3],
        internal_breaks: usize,
        structure: usize,
        operations: OperationProfile,
        typo_costs: &mut Vec<usize>,
        insertions: &mut Vec<usize>,
        alignment_costs: &mut Vec<usize>,
    ) {
        let len = self.query_tokens.len();
        if query_idx == len {
            let total = (
                distance,
                inversion,
                segment.clone(),
                *position,
                internal_breaks,
                structure,
                operations,
                typo_costs.clone(),
                insertions.clone(),
                alignment_costs.clone(),
            );
            if self
                .best
                .as_ref()
                .is_none_or(|current| aligned_match_cmp(&total, current).is_lt())
            {
                self.best = Some(total);
            }
            return;
        }

        let Some((distance_bound, alignment_bound)) =
            self.lower_bound(window_start, query_idx, distance, alignment_costs)
        else {
            return;
        };
        if distance_bound > self.limit
            || self.best.as_ref().is_some_and(|current| {
                alignment_bound > AlignmentCostVector::from_slice(&current.9)
            })
        {
            return;
        }

        let mut choices = (0..len)
            .filter(|candidate_idx| !self.used[*candidate_idx])
            .collect::<Vec<_>>();
        choices.sort_unstable_by_key(|candidate_idx| {
            self.choice(window_start, query_idx, *candidate_idx, self.limit)
                .map_or((usize::MAX, usize::MAX, usize::MAX), |choice| {
                    (choice.alignment_cost, choice.distance, choice.penalty)
                })
        });

        for candidate_idx in choices {
            let Some(remaining) = self.limit.checked_sub(distance) else {
                break;
            };
            let Some(choice) = self.choice(window_start, query_idx, candidate_idx, remaining)
            else {
                continue;
            };
            let TokenChoice {
                distance: cost,
                penalty,
                position_rank,
                internal_breaks: breaks,
                operations: token_operations,
                fragmented: _,
                alignment_cost: token_alignment_cost,
            } = choice;

            self.used[candidate_idx] = true;
            self.assignment.push(candidate_idx);
            if segment.len() <= self.candidate_tokens[window_start + candidate_idx].segment {
                segment.resize(
                    self.candidate_tokens[window_start + candidate_idx].segment + 1,
                    0,
                );
            }
            let segment_idx = self.candidate_tokens[window_start + candidate_idx].segment;
            segment[segment_idx] += 1;
            increment_simple_position_vector(position, position_rank, 1);
            typo_costs.push(token_operations.core_typo_cost());
            insertions.push(token_operations.insertions);
            alignment_costs.push(token_alignment_cost);

            self.search(
                window_start,
                query_idx + 1,
                distance + cost,
                inversion
                    + self.assignment[..self.assignment.len() - 1]
                        .iter()
                        .filter(|previous| **previous > candidate_idx)
                        .count(),
                segment,
                position,
                internal_breaks + breaks,
                structure + penalty,
                operations.merge(token_operations),
                typo_costs,
                insertions,
                alignment_costs,
            );

            alignment_costs.pop();
            insertions.pop();
            typo_costs.pop();
            position[position_rank.min(2)] -= 1;
            segment[segment_idx] -= 1;
            while segment.last() == Some(&0) {
                segment.pop();
            }
            self.assignment.pop();
            self.used[candidate_idx] = false;
        }
    }

    fn choice(
        &mut self,
        window_start: usize,
        query_idx: usize,
        candidate_idx: usize,
        limit: usize,
    ) -> Option<TokenChoice> {
        self.scratch.choice(
            query_idx,
            self.candidate_tokens[window_start + candidate_idx].token,
            limit,
        )
    }

    fn lower_bound(
        &self,
        window_start: usize,
        query_idx: usize,
        distance: usize,
        alignment_costs: &[usize],
    ) -> Option<(usize, AlignmentCostVector)> {
        let mut lower_bound = distance;
        let mut alignment_bound = smallvec::SmallVec::<[usize; 8]>::from_slice(alignment_costs);
        for query_idx in query_idx..self.query_tokens.len() {
            // A preferred alignment can consume more edits than another valid
            // alignment. Only unavoidable missing characters are a safe bound.
            let min_cost = (0..self.query_tokens.len())
                .filter(|candidate_idx| !self.used[*candidate_idx])
                .map(|candidate_idx| {
                    self.edit_bounds
                        [(window_start + candidate_idx) * self.query_tokens.len() + query_idx]
                })
                .min()?;
            lower_bound = lower_bound.checked_add(min_cost)?;
            alignment_bound.push(alignment_cost_milli(
                OperationProfile {
                    deletions: min_cost,
                    ..OperationProfile::default()
                },
                false,
                0,
            ));
        }
        Some((
            lower_bound,
            AlignmentCostVector::from_slice(&alignment_bound),
        ))
    }

    fn is_perfect(&self, query_len: usize) -> bool {
        self.best.as_ref().is_some_and(|candidate| {
            candidate.0 == 0
                && candidate.1 == 0
                && candidate.2.first() == Some(&query_len)
                && candidate.3[0] == query_len
                && candidate.4 == 0
                && candidate.5 == 0
                && candidate.6 == OperationProfile::default()
                && candidate.7.iter().all(|cost| *cost == 0)
                && candidate.8.iter().all(|count| *count == 0)
                && candidate.9.iter().all(|cost| *cost == 0)
        })
    }
}

fn aligned_match_cmp(
    left: &AlignedTokenDistance,
    right: &AlignedTokenDistance,
) -> std::cmp::Ordering {
    compare_alignments(
        AlignmentView {
            distance: left.0,
            order: left.1,
            segment: &left.2,
            position: &left.3,
            operations: left.6,
            typos: &left.7,
            insertions: &left.8,
            alignment: &left.9,
        },
        AlignmentView {
            distance: right.0,
            order: right.1,
            segment: &right.2,
            position: &right.3,
            operations: right.6,
            typos: &right.7,
            insertions: &right.8,
            alignment: &right.9,
        },
    )
}

struct AlignmentView<'a> {
    distance: usize,
    order: usize,
    segment: &'a [usize],
    position: &'a [usize; 3],
    operations: OperationProfile,
    typos: &'a [usize],
    insertions: &'a [usize],
    alignment: &'a [usize],
}

fn compare_alignments(left: AlignmentView<'_>, right: AlignmentView<'_>) -> std::cmp::Ordering {
    let left_alignment = AlignmentCostVector::from_slice(left.alignment);
    let right_alignment = AlignmentCostVector::from_slice(right.alignment);
    let left_typos = TypoCostVector::from_slice(left.typos);
    let right_typos = TypoCostVector::from_slice(right.typos);
    let left_insertions = InsertionVector::from_slice(left.insertions);
    let right_insertions = InsertionVector::from_slice(right.insertions);
    compare_path_typo_sort_keys(
        &PathTypoSortKey {
            distance: left.distance,
            order_distance: left.order,
            operations: left.operations,
            alignment_costs: &left_alignment,
            typo_costs: &left_typos,
            insertions: &left_insertions,
            segment: left.segment,
            position: left.position,
            score: 0.0,
            key: "",
        },
        &PathTypoSortKey {
            distance: right.distance,
            order_distance: right.order,
            operations: right.operations,
            alignment_costs: &right_alignment,
            typo_costs: &right_typos,
            insertions: &right_insertions,
            segment: right.segment,
            position: right.position,
            score: 0.0,
            key: "",
        },
    )
    .then_with(|| left.distance.cmp(&right.distance))
}

pub fn partitioned_token_distance(
    query_tokens: &[&str],
    candidate: &str,
    limit: usize,
) -> Option<PartitionedTokenDistance> {
    partitioned_token_distance_with_scratch(
        query_tokens,
        candidate,
        limit,
        &mut SequenceMatchScratch::default(),
    )
}

pub(crate) fn partitioned_token_distance_with_scratch(
    query_tokens: &[&str],
    candidate: &str,
    limit: usize,
    scratch: &mut SequenceMatchScratch,
) -> Option<PartitionedTokenDistance> {
    scratch.prepare(query_tokens, limit);
    if let Some(cached) = scratch.compounds.get(candidate) {
        return cached.clone();
    }
    scratch.boundaries.clear();
    scratch
        .boundaries
        .extend(candidate.char_indices().map(|(idx, _)| idx));
    scratch.boundaries.push(candidate.len());
    if scratch.boundaries.len() <= query_tokens.len() {
        return None;
    }
    scratch.partitions.clear();
    let result = partitioned_token_distance_impl(
        query_tokens,
        &scratch.query_chars,
        candidate,
        &scratch.boundaries,
        0,
        0,
        limit,
        &mut scratch.partitions,
        &mut scratch.token,
    );
    if scratch.compounds.len() < 8192 {
        scratch
            .compounds
            .insert(candidate.to_owned(), result.clone());
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn partitioned_token_distance_impl(
    query_tokens: &[&str],
    query_chars: &[Vec<char>],
    candidate: &str,
    boundaries: &[usize],
    token_idx: usize,
    start_boundary: usize,
    remaining: usize,
    cache: &mut HashMap<(usize, usize, usize), Option<PartitionedTokenDistance>>,
    scratch: &mut TokenMatchScratch,
) -> Option<PartitionedTokenDistance> {
    let cache_key = (token_idx, start_boundary, remaining);
    if let Some(cached) = cache.get(&cache_key) {
        return cached.clone();
    }
    let last_token = token_idx + 1 == query_tokens.len();
    let min_end_boundary = start_boundary + 1;
    let max_end_boundary = boundaries.len() - (query_tokens.len() - token_idx);
    let min_end_boundary = if last_token {
        max_end_boundary
    } else {
        min_end_boundary
    };
    let mut best: Option<PartitionedTokenDistance> = None;

    for end_boundary in min_end_boundary..=max_end_boundary {
        if !last_token && end_boundary == boundaries.len() - 1 {
            break;
        }

        let end_boundary = if last_token {
            boundaries.len() - 1
        } else {
            end_boundary
        };
        let segment = &candidate[boundaries[start_boundary]..boundaries[end_boundary]];
        let (cost, structure, position_rank, _, operations) =
            match best_token_match_chars_with_scratch(
                &query_chars[token_idx],
                segment,
                remaining,
                scratch,
            ) {
                Some(values) => values,
                None => continue,
            };
        if cost > remaining {
            continue;
        }

        let mut position = [0; 3];
        increment_simple_position_vector(&mut position, position_rank, 1);
        let alignment_cost = alignment_cost_milli(operations, false, 0);
        let operations = OperationProfile {
            insertions: operations.insertions + structure,
            ..operations
        };
        let total = if last_token {
            (
                cost,
                structure,
                position,
                operations,
                vec![operations.core_typo_cost()],
                vec![operations.insertions],
                vec![alignment_cost],
            )
        } else {
            let Some((
                tail_cost,
                tail_structure,
                tail_position,
                tail_operations,
                tail_costs,
                tail_insertions,
                tail_alignment_costs,
            )) = partitioned_token_distance_impl(
                query_tokens,
                query_chars,
                candidate,
                boundaries,
                token_idx + 1,
                end_boundary,
                remaining - cost,
                cache,
                scratch,
            )
            else {
                continue;
            };
            let mut typo_costs = Vec::with_capacity(tail_costs.len() + 1);
            typo_costs.push(operations.core_typo_cost());
            typo_costs.extend(tail_costs);
            let mut insertions = Vec::with_capacity(tail_insertions.len() + 1);
            insertions.push(operations.insertions);
            insertions.extend(tail_insertions);
            let mut alignment_costs = Vec::with_capacity(tail_alignment_costs.len() + 1);
            alignment_costs.push(alignment_cost);
            alignment_costs.extend(tail_alignment_costs);
            (
                cost + tail_cost,
                structure + tail_structure,
                std::array::from_fn(|idx| position[idx] + tail_position[idx]),
                operations.merge(tail_operations),
                typo_costs,
                insertions,
                alignment_costs,
            )
        };

        best = Some(match best {
            None => total,
            Some(current)
                if compare_alignments(
                    AlignmentView {
                        distance: total.0,
                        order: 0,
                        segment: &[],
                        position: &total.2,
                        operations: total.3,
                        typos: &total.4,
                        insertions: &total.5,
                        alignment: &total.6,
                    },
                    AlignmentView {
                        distance: current.0,
                        order: 0,
                        segment: &[],
                        position: &current.2,
                        operations: current.3,
                        typos: &current.4,
                        insertions: &current.5,
                        alignment: &current.6,
                    },
                )
                .is_lt() =>
            {
                total
            }
            Some(current) => current,
        });
        if best.as_ref().is_some_and(|candidate| {
            candidate.0 == 0
                && candidate.1 == 0
                && candidate.2[0] == query_tokens.len()
                && candidate.3 == OperationProfile::default()
                && candidate.4.iter().all(|cost| *cost == 0)
                && candidate.5.iter().all(|count| *count == 0)
                && candidate.6.iter().all(|cost| *cost == 0)
        }) {
            break;
        }

        if last_token {
            break;
        }
    }

    cache.insert(cache_key, best.clone());
    best
}

pub fn best_token_match(
    query: &str,
    candidate: &str,
    limit: usize,
) -> Option<(usize, usize, usize, usize, OperationProfile)> {
    let query_chars: Vec<char> = query.chars().collect();
    let mut scratch = TokenMatchScratch::default();
    best_token_match_chars_with_scratch(&query_chars, candidate, limit, &mut scratch)
}

pub(crate) fn best_token_match_chars_with_scratch(
    query_chars: &[char],
    candidate: &str,
    limit: usize,
    scratch: &mut TokenMatchScratch,
) -> Option<(usize, usize, usize, usize, OperationProfile)> {
    if query_chars.is_empty() || candidate.is_empty() {
        return None;
    }

    scratch.candidate_chars.clear();
    scratch.candidate_chars.extend(candidate.chars());
    let candidate_chars = &scratch.candidate_chars;
    if character_deficit_lower_bound(query_chars, candidate_chars) > limit {
        return None;
    }

    let query_len = query_chars.len();
    let candidate_len = candidate_chars.len();

    // Exact windows are common in path and metadata searches. They have a
    // deterministic zero-distance result, so avoid running the DP once per
    // possible substring when one is present.
    let mut exact = None;
    if query_len <= candidate_len {
        for (start, window) in candidate_chars.windows(query_len).enumerate() {
            if window != query_chars {
                continue;
            }
            let end = start + query_len;
            let position_rank = if start == 0 {
                0
            } else if end == candidate_len {
                1
            } else {
                2
            };
            let total = (
                0,
                start + (candidate_len - end),
                position_rank,
                query_len,
                OperationProfile::default(),
            );
            exact = Some(match exact {
                None => total,
                Some(current) if total < current => total,
                Some(current) => current,
            });
        }
    }
    if exact.is_some() {
        return exact;
    }

    let mut best = None;
    for start in 0..candidate_len {
        let min_len = 1usize.max(query_len.saturating_sub(limit));
        let max_len = (query_len + limit).min(candidate_len - start);
        for len in min_len..=max_len {
            let end = start + len;
            let segment = &candidate_chars[start..end];
            let Some((distance, operations)) = bounded_damerau_levenshtein_chars_with_scratch(
                segment,
                query_chars,
                limit,
                &mut scratch.damerau,
            ) else {
                continue;
            };
            let penalty = start + (candidate_len - end);
            let position_rank = if start == 0 {
                0
            } else if end == candidate_len {
                1
            } else {
                2
            };
            let total = (distance, penalty, position_rank, len, operations);
            best = Some(match best {
                None => total,
                Some(current) if total < current => total,
                Some(current) => current,
            });
            if best == Some((0, 0, 0, candidate_len, OperationProfile::default())) {
                return best;
            }
        }
    }

    best
}

pub fn fragmented_token_match(
    query: &str,
    candidate: &str,
    limit: usize,
) -> Option<FragmentedTokenMatch> {
    let query_chars: Vec<_> = query.chars().collect();
    let mut scratch = TokenMatchScratch::default();
    fragmented_token_match_chars(&query_chars, candidate, limit, &mut scratch)
}

fn fragmented_token_match_chars(
    query_chars: &[char],
    candidate: &str,
    limit: usize,
    scratch: &mut TokenMatchScratch,
) -> Option<FragmentedTokenMatch> {
    if query_chars.is_empty() || candidate.is_empty() {
        return None;
    }

    scratch.candidate_chars.clear();
    scratch.candidate_chars.extend(candidate.chars());
    let candidate_chars = &scratch.candidate_chars;
    if character_deficit_lower_bound(query_chars, candidate_chars) > limit {
        return None;
    }

    let n = candidate_chars.len();
    let m = query_chars.len();

    #[derive(Clone, Copy)]
    enum State {
        NoRun = 0,
        InRunStartZero = 1,
        InRunStartNonZero = 2,
        GapStartZero = 3,
        GapStartNonZero = 4,
    }

    let dp = &mut scratch.fragmented;
    // Only the current row and the next two rows are reachable (including transpositions).
    dp.resize_with(3, || vec![[None; 5]; m + 1]);
    for row in dp.iter_mut() {
        row.resize(m + 1, [None; 5]);
        row.fill([None; 5]);
    }
    dp[0][0][State::NoRun as usize] = Some(FragmentCost {
        distance: 0,
        internal_breaks: 0,
        operations: OperationProfile::default(),
    });

    let state_position_rank = |state_idx: usize| match state_idx {
        x if x == State::InRunStartZero as usize => 0,
        x if x == State::InRunStartNonZero as usize => 1,
        x if x == State::GapStartZero as usize => 0,
        x if x == State::GapStartNonZero as usize => 2,
        _ => 2,
    };

    let update = |slot: &mut Option<FragmentCost>, candidate: FragmentCost, state_idx: usize| {
        if candidate.distance > limit {
            return;
        }
        match slot {
            Some(current)
                if current.sort_key(state_position_rank(state_idx))
                    <= candidate.sort_key(state_position_rank(state_idx)) => {}
            _ => *slot = Some(candidate),
        }
    };

    for i in 0..=n {
        for j in 0..=m {
            for state_idx in 0..5 {
                let Some(cost) = dp[i % 3][j][state_idx] else {
                    continue;
                };

                if i < n {
                    let mut next = cost;
                    next.operations = next.operations.with_insertion(
                        candidate_chars[i],
                        i.checked_sub(1).map(|idx| candidate_chars[idx]),
                        candidate_chars.get(i + 1).copied(),
                    );
                    let next_state = match state_idx {
                        x if x == State::NoRun as usize => {
                            next.internal_breaks += 1;
                            State::GapStartZero
                        }
                        x if x == State::InRunStartZero as usize => {
                            next.internal_breaks += 1;
                            State::GapStartZero
                        }
                        x if x == State::InRunStartNonZero as usize => {
                            next.internal_breaks += 1;
                            State::GapStartNonZero
                        }
                        x if x == State::GapStartZero as usize => State::GapStartZero,
                        _ => State::GapStartNonZero,
                    };
                    update(
                        &mut dp[(i + 1) % 3][j][next_state as usize],
                        next,
                        next_state as usize,
                    );
                }

                if j < m {
                    let mut next = cost;
                    next.distance += 1;
                    next.operations = next.operations.with_deletion(
                        query_chars[j],
                        i.checked_sub(1).map(|idx| candidate_chars[idx]),
                        candidate_chars.get(i).copied(),
                    );
                    let next_state = match state_idx {
                        x if x == State::NoRun as usize => State::NoRun,
                        x if x == State::InRunStartZero as usize => State::InRunStartZero,
                        x if x == State::InRunStartNonZero as usize => State::InRunStartNonZero,
                        x if x == State::GapStartZero as usize => State::GapStartZero,
                        _ => State::GapStartNonZero,
                    };
                    update(
                        &mut dp[i % 3][j + 1][next_state as usize],
                        next,
                        next_state as usize,
                    );
                }

                if i < n && j < m {
                    let mut next = cost;
                    if candidate_chars[i] != query_chars[j] {
                        next.distance += 1;
                        next.operations = next
                            .operations
                            .with_substitution(candidate_chars[i], query_chars[j]);
                    }
                    let next_state = match state_idx {
                        x if x == State::NoRun as usize && i == 0 => State::InRunStartZero,
                        x if x == State::NoRun as usize => State::InRunStartNonZero,
                        x if x == State::GapStartZero as usize => State::InRunStartZero,
                        x if x == State::GapStartNonZero as usize => State::InRunStartNonZero,
                        x if x == State::InRunStartZero as usize => State::InRunStartZero,
                        _ => State::InRunStartNonZero,
                    };
                    update(
                        &mut dp[(i + 1) % 3][j + 1][next_state as usize],
                        next,
                        next_state as usize,
                    );
                }

                if i + 1 < n
                    && j + 1 < m
                    && candidate_chars[i] == query_chars[j + 1]
                    && candidate_chars[i + 1] == query_chars[j]
                {
                    let mut next = cost;
                    next.distance += 1;
                    next.operations = next.operations.with_transposition();
                    let next_state = match state_idx {
                        x if x == State::NoRun as usize && i == 0 => State::InRunStartZero,
                        x if x == State::NoRun as usize => State::InRunStartNonZero,
                        x if x == State::GapStartZero as usize => State::InRunStartZero,
                        x if x == State::GapStartNonZero as usize => State::InRunStartNonZero,
                        x if x == State::InRunStartZero as usize => State::InRunStartZero,
                        _ => State::InRunStartNonZero,
                    };
                    update(
                        &mut dp[(i + 2) % 3][j + 2][next_state as usize],
                        next,
                        next_state as usize,
                    );
                }
            }
        }
        if i < n {
            dp[i % 3].fill([None; 5]);
        }
    }

    [
        State::InRunStartZero as usize,
        State::InRunStartNonZero as usize,
        State::GapStartZero as usize,
        State::GapStartNonZero as usize,
    ]
    .into_iter()
    .filter_map(|state| {
        dp[n % 3][m][state]
            .filter(|cost| cost.internal_breaks > 0)
            .map(|cost| (state, cost))
    })
    .min_by_key(|(state, cost)| cost.sort_key(state_position_rank(*state)))
    .map(|(state, cost)| FragmentedTokenMatch {
        distance: cost.distance,
        internal_breaks: cost.internal_breaks,
        operations: cost.operations,
        position_rank: state_position_rank(state),
    })
}

/// Returns a lossless lower bound for edit distance based on missing ASCII
/// characters. A candidate cannot contain a query substring if it lacks more
/// query-character occurrences than the edit budget can repair.
fn character_deficit_lower_bound(query: &[char], candidate: &[char]) -> usize {
    let mut query_counts = [0usize; 128];
    let mut candidate_counts = [0usize; 128];
    for &character in query {
        if let Some(count) = query_counts.get_mut(character as usize) {
            *count += 1;
        }
    }
    for &character in candidate {
        if let Some(count) = candidate_counts.get_mut(character as usize) {
            *count += 1;
        }
    }

    query_counts
        .into_iter()
        .zip(candidate_counts)
        .map(|(query_count, candidate_count)| query_count.saturating_sub(candidate_count))
        .sum()
}

#[cfg(test)]
#[path = "token_match/fragmented_reference.rs"]
mod fragmented_reference;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rolling_fragmented_alignment_agrees_with_full_matrix() {
        let words = [
            "", "a", "b", "ab", "ba", "aa", "aab", "abb", "\u{e9}a", "a\u{e9}", "abca", "aabc",
        ];
        let mut actual = TokenMatchScratch::default();
        let mut reference = TokenMatchScratch::default();
        for query in words {
            let chars: Vec<_> = query.chars().collect();
            for candidate in words {
                for limit in [0, 1, 2, 4] {
                    assert_eq!(
                        fragmented_token_match_chars(&chars, candidate, limit, &mut actual),
                        fragmented_reference::fragmented_token_match_reference(
                            &chars,
                            candidate,
                            limit,
                            &mut reference
                        ),
                        "query={query:?} candidate={candidate:?} limit={limit}"
                    );
                    assert!(actual.fragmented.len() <= 3);
                }
            }
        }
        for (query, candidate) in [
            ("applaunch", "applicationlauncher"),
            ("list", "unsplash"),
            ("splyush", "unsplash"),
            ("appliatoin", "applicationlauncher"),
            ("abcabc", "xabcabcy"),
        ] {
            let chars: Vec<_> = query.chars().collect();
            for limit in [0, 1, 3, 7] {
                assert_eq!(
                    fragmented_token_match_chars(&chars, candidate, limit, &mut actual),
                    fragmented_reference::fragmented_token_match_reference(
                        &chars,
                        candidate,
                        limit,
                        &mut reference
                    )
                );
            }
        }
    }

    #[test]
    fn single_token_cache_is_query_and_budget_scoped() {
        let mut cached = SingleTokenMatchScratch::default();
        for query in ["app", "fuxxy", "applaunch", "app", "\u{e9}a"] {
            let chars: Vec<_> = query.chars().collect();
            for limit in [0, 2, 6, 2] {
                for candidate in ["applicationlauncher", "fuzzy", "home", "\u{e9}a", "xxxx"] {
                    let expected = choose_path_token_match(query, candidate, limit);
                    for _ in 0..2 {
                        assert_eq!(cached.choice(&chars, candidate, limit), expected);
                    }
                }
            }
        }
        assert!(cached.choices.contains_key("xxxx"));
    }

    #[test]
    fn alignment_pruning_agrees_with_exhaustive_two_token_assignments() {
        for tokens in [
            [
                "xapp",
                "launch",
                "application",
                "launch",
                "dev",
                "applaunch",
            ],
            ["alpha", "gamam", "gamut", "alphi", "gamma", "applications"],
            ["very", "good", "verygood", "good", "goood", "vry"],
        ] {
            let candidates: Vec<_> = tokens
                .iter()
                .enumerate()
                .map(|(index, token)| PositionedToken {
                    token,
                    segment: (index / 2) % 2,
                })
                .collect();
            for query in [
                ["apq", "launch"],
                ["dev", "applaunch"],
                ["alpha", "gamam"],
                ["very", "good"],
            ] {
                for limit in [0, 1, 2, 4, 6] {
                    let mut expected: Option<AlignedTokenDistance> = None;
                    for start in 0..candidates.len() - 1 {
                        for order in [[0, 1], [1, 0]] {
                            let Some(first) = choose_path_token_match(
                                query[0],
                                candidates[start + order[0]].token,
                                limit,
                            ) else {
                                continue;
                            };
                            let Some(second) = choose_path_token_match(
                                query[1],
                                candidates[start + order[1]].token,
                                limit - first.distance,
                            ) else {
                                continue;
                            };
                            let mut segment = vec![0; 2];
                            for offset in order {
                                segment[candidates[start + offset].segment] += 1;
                            }
                            while segment.last() == Some(&0) {
                                segment.pop();
                            }
                            let mut position = [0; 3];
                            for matched in [first, second] {
                                increment_simple_position_vector(
                                    &mut position,
                                    matched.position_rank,
                                    1,
                                );
                            }
                            let total = (
                                first.distance + second.distance,
                                usize::from(order[0] > order[1]),
                                segment,
                                position,
                                first.internal_breaks + second.internal_breaks,
                                first.penalty + second.penalty,
                                first.operations.merge(second.operations),
                                vec![
                                    first.operations.core_typo_cost(),
                                    second.operations.core_typo_cost(),
                                ],
                                vec![first.operations.insertions, second.operations.insertions],
                                vec![first.alignment_cost, second.alignment_cost],
                            );
                            if expected
                                .as_ref()
                                .is_none_or(|best| aligned_match_cmp(&total, best).is_lt())
                            {
                                expected = Some(total);
                            }
                        }
                    }
                    let actual = aligned_token_distance(&query, &candidates, limit);
                    assert_eq!(
                        actual.is_some(),
                        expected.is_some(),
                        "{query:?} {tokens:?} limit={limit}"
                    );
                    if let (Some(actual), Some(expected)) = (actual, expected) {
                        assert!(
                            aligned_match_cmp(&actual, &expected).is_eq(),
                            "{query:?} {tokens:?} limit={limit}: {actual:?} != {expected:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn alignment_uses_final_path_position_order() {
        let tokens: Vec<_> = ["xapp", "launch", "application", "launch"]
            .into_iter()
            .map(|token| PositionedToken { token, segment: 0 })
            .collect();
        let matched = aligned_token_distance(&["apq", "launch"], &tokens, 4).unwrap();
        assert_eq!(matched.9, vec![1000, 0]);
        assert_eq!(matched.3, [2, 0, 0]);
    }

    #[test]
    fn sequence_caches_preserve_results_across_queries_and_budgets() {
        let mut scratch = SequenceMatchScratch::default();
        let candidates = [
            "applicationlauncher",
            "gamam",
            "launch",
            "verygood",
            "fuzzy-rank",
            "\u{e9}l\u{e9}phant",
        ];
        let tokens: Vec<_> = candidates
            .iter()
            .enumerate()
            .map(|(segment, token)| PositionedToken { token, segment })
            .collect();
        for query in [
            ["app", "launch"],
            ["alpha", "gamam"],
            ["dev", "applaunch"],
            ["apq", "launch"],
            ["app", "launch"],
        ] {
            for limit in [4, 1, 0, 6, 4] {
                for candidate in candidates {
                    let expected = partitioned_token_distance(&query, candidate, limit);
                    for _ in 0..2 {
                        assert_eq!(
                            partitioned_token_distance_with_scratch(
                                &query,
                                candidate,
                                limit,
                                &mut scratch
                            ),
                            expected
                        );
                    }
                }
                assert_eq!(
                    aligned_token_distance_with_scratch(&query, &tokens, limit, &mut scratch),
                    aligned_token_distance(&query, &tokens, limit)
                );
            }
        }
    }

    #[test]
    fn contiguous_transposition_beats_fragmented_application_alignment() {
        let query = "appliatoin";
        let candidate = "applicationlauncher";
        let contiguous = best_token_match(query, candidate, max_typos(query.len())).unwrap();
        let fragmented = fragmented_token_match(query, candidate, max_typos(query.len())).unwrap();
        let chosen = choose_path_token_match(query, candidate, max_typos(query.len())).unwrap();

        assert!(
            alignment_cost_milli(contiguous.4, false, 0)
                < alignment_cost_milli(fragmented.operations, true, fragmented.internal_breaks),
            "contiguous={contiguous:?}, fragmented={fragmented:?}"
        );
        assert!(!chosen.fragmented);
        assert_eq!(
            chosen.operations.core_typo_cost(),
            contiguous.4.core_typo_cost()
        );
    }
}
