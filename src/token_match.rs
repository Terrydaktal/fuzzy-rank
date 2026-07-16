use crate::core::{
    OperationProfile, bounded_damerau_levenshtein, increment_position_vector_simple,
    inversion_count, max_typos,
};

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

pub fn choose_path_token_match(
    query: &str,
    candidate: &str,
    limit: usize,
) -> Option<(usize, usize, usize, usize, OperationProfile)> {
    let contiguous = best_token_match(query, candidate, limit).map(
        |(distance, penalty, position_rank, _len, operations)| {
            (distance, penalty, position_rank, 0usize, operations)
        },
    );
    let fragmented = {
        let local_limit = limit.min(max_typos(query.chars().count()));
        fragmented_token_match(query, candidate, local_limit).map(|matched| {
            let penalty = candidate
                .chars()
                .count()
                .saturating_sub(query.chars().count());
            (
                matched.distance,
                penalty,
                matched.position_rank,
                matched.internal_breaks,
                matched.operations,
            )
        })
    };

    match (contiguous, fragmented) {
        (Some(left), Some(right)) => {
            let left_key = (
                left.4.core_typo_cost(),
                left.2,
                left.3,
                left.4.insertions,
                left.0,
                left.4,
            );
            let right_key = (
                right.4.core_typo_cost(),
                right.2,
                right.3,
                right.4.insertions,
                right.0,
                right.4,
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
) -> Option<(
    usize,
    usize,
    Vec<usize>,
    [usize; 3],
    usize,
    usize,
    OperationProfile,
    Vec<usize>,
    Vec<usize>,
)> {
    if candidate_tokens.len() < query_tokens.len() {
        return None;
    }

    let mut best: Option<(
        usize,
        usize,
        Vec<usize>,
        [usize; 3],
        usize,
        usize,
        OperationProfile,
        Vec<usize>,
        Vec<usize>,
    )> = None;
    let assignments = unordered_assignments(query_tokens.len());
    for start in 0..=candidate_tokens.len() - query_tokens.len() {
        let window = &candidate_tokens[start..start + query_tokens.len()];
        for assignment in &assignments {
            let mut distance = 0;
            let mut segment = Vec::new();
            let mut position = [0; 3];
            let mut internal_breaks = 0;
            let mut structure = 0;
            let mut operations = OperationProfile::default();
            let mut typo_costs = Vec::with_capacity(query_tokens.len());
            let mut insertions = Vec::with_capacity(query_tokens.len());
            let mut failed = false;

            for (query_idx, candidate_idx) in assignment.iter().enumerate() {
                let query_token = query_tokens[query_idx];
                let candidate = &window[*candidate_idx];
                let Some(remaining) = limit.checked_sub(distance) else {
                    failed = true;
                    break;
                };
                let Some((cost, penalty, position_rank, token_internal_breaks, token_operations)) =
                    choose_path_token_match(query_token, candidate.token, remaining)
                else {
                    failed = true;
                    break;
                };
                distance += cost;
                if segment.len() <= candidate.segment {
                    segment.resize(candidate.segment + 1, 0);
                }
                segment[candidate.segment] += 1;
                increment_position_vector_simple(&mut position, position_rank, 1);
                internal_breaks += token_internal_breaks;
                structure += penalty;
                operations = operations.merge(token_operations);
                typo_costs.push(token_operations.core_typo_cost());
                insertions.push(token_operations.insertions);
                if distance > limit {
                    failed = true;
                    break;
                }
            }

            if failed {
                continue;
            }

            let total = (
                distance,
                inversion_count(assignment),
                segment.clone(),
                position,
                internal_breaks,
                structure,
                operations,
                typo_costs.clone(),
                insertions.clone(),
            );
            best = Some(match best {
                None => total,
                Some(current)
                    if total
                        .0
                        .cmp(&current.0)
                        .then_with(|| total.1.cmp(&current.1))
                        .then_with(|| current.2.cmp(&total.2))
                        .then_with(|| current.3.cmp(&total.3))
                        .then_with(|| total.4.cmp(&current.4))
                        .then_with(|| total.5.cmp(&current.5))
                        .then_with(|| total.6.cmp(&current.6))
                        .then_with(|| total.7.cmp(&current.7))
                        .then_with(|| total.8.cmp(&current.8))
                        .is_lt() =>
                {
                    total
                }
                Some(current) => current,
            });

            if best.as_ref().is_some_and(|candidate| {
                candidate.0 == 0
                    && candidate.1 == 0
                    && candidate.2.first() == Some(&query_tokens.len())
                    && candidate.3[0] == query_tokens.len()
                    && candidate.4 == 0
                    && candidate.5 == 0
                    && candidate.6 == OperationProfile::default()
                    && candidate.7.iter().all(|cost| *cost == 0)
                    && candidate.8.iter().all(|count| *count == 0)
            }) {
                break;
            }
        }
        if best.as_ref().is_some_and(|candidate| {
            candidate.0 == 0
                && candidate.1 == 0
                && candidate.2.first() == Some(&query_tokens.len())
                && candidate.3[0] == query_tokens.len()
                && candidate.4 == 0
                && candidate.5 == 0
                && candidate.6 == OperationProfile::default()
                && candidate.7.iter().all(|cost| *cost == 0)
                && candidate.8.iter().all(|count| *count == 0)
        }) {
            break;
        }
    }

    best
}

fn unordered_assignments(len: usize) -> Vec<Vec<usize>> {
    if len > 8 {
        return vec![(0..len).collect()];
    }

    fn build(prefix: &mut Vec<usize>, used: &mut [bool], output: &mut Vec<Vec<usize>>) {
        if prefix.len() == used.len() {
            output.push(prefix.clone());
            return;
        }
        for index in 0..used.len() {
            if used[index] {
                continue;
            }
            used[index] = true;
            prefix.push(index);
            build(prefix, used, output);
            prefix.pop();
            used[index] = false;
        }
    }

    let mut output = Vec::new();
    build(&mut Vec::new(), &mut vec![false; len], &mut output);
    output
}

pub fn partitioned_token_distance(
    query_tokens: &[&str],
    candidate: &str,
    limit: usize,
) -> Option<(
    usize,
    usize,
    [usize; 3],
    OperationProfile,
    Vec<usize>,
    Vec<usize>,
)> {
    let mut boundaries: Vec<usize> = candidate.char_indices().map(|(idx, _)| idx).collect();
    boundaries.push(candidate.len());
    if boundaries.len() <= query_tokens.len() {
        return None;
    }

    partitioned_token_distance_impl(query_tokens, candidate, &boundaries, 0, 0, limit)
}

fn partitioned_token_distance_impl(
    query_tokens: &[&str],
    candidate: &str,
    boundaries: &[usize],
    token_idx: usize,
    start_boundary: usize,
    remaining: usize,
) -> Option<(
    usize,
    usize,
    [usize; 3],
    OperationProfile,
    Vec<usize>,
    Vec<usize>,
)> {
    let last_token = token_idx + 1 == query_tokens.len();
    let min_end_boundary = start_boundary + 1;
    let max_end_boundary = boundaries.len() - (query_tokens.len() - token_idx);
    let mut best: Option<(
        usize,
        usize,
        [usize; 3],
        OperationProfile,
        Vec<usize>,
        Vec<usize>,
    )> = None;

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
            match best_token_match(query_tokens[token_idx], segment, remaining) {
                Some(values) => values,
                None => continue,
            };
        if cost > remaining {
            continue;
        }

        let mut position = [0; 3];
        increment_position_vector_simple(&mut position, position_rank, 1);
        let total = if last_token {
            (
                cost,
                structure,
                position,
                operations,
                vec![operations.core_typo_cost()],
                vec![operations.insertions],
            )
        } else {
            let Some((
                tail_cost,
                tail_structure,
                tail_position,
                tail_operations,
                tail_costs,
                tail_insertions,
            )) = partitioned_token_distance_impl(
                query_tokens,
                candidate,
                boundaries,
                token_idx + 1,
                end_boundary,
                remaining - cost,
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
            (
                cost + tail_cost,
                structure + tail_structure,
                std::array::from_fn(|idx| position[idx] + tail_position[idx]),
                operations.merge(tail_operations),
                typo_costs,
                insertions,
            )
        };

        best = Some(match best {
            None => total,
            Some(current)
                if total
                    .0
                    .cmp(&current.0)
                    .then_with(|| total.1.cmp(&current.1))
                    .then_with(|| current.2.cmp(&total.2))
                    .then_with(|| total.3.cmp(&current.3))
                    .then_with(|| total.4.cmp(&current.4))
                    .then_with(|| total.5.cmp(&current.5))
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
        }) {
            break;
        }

        if last_token {
            break;
        }
    }

    best
}

pub fn best_token_match(
    query: &str,
    candidate: &str,
    limit: usize,
) -> Option<(usize, usize, usize, usize, OperationProfile)> {
    if query.is_empty() || candidate.is_empty() {
        return None;
    }

    let candidate_len = candidate.chars().count();
    let query_len = query.chars().count();
    let mut boundaries: Vec<usize> = candidate.char_indices().map(|(idx, _)| idx).collect();
    boundaries.push(candidate.len());

    let mut best = None;
    for start in 0..candidate_len {
        let min_len = 1usize.max(query_len.saturating_sub(limit));
        let max_len = (query_len + limit).min(candidate_len - start);
        for len in min_len..=max_len {
            let end = start + len;
            let segment = &candidate[boundaries[start]..boundaries[end]];
            let Some((distance, operations)) = bounded_damerau_levenshtein(segment, query, limit)
            else {
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
    if query.is_empty() || candidate.is_empty() {
        return None;
    }

    let query_chars: Vec<_> = query.chars().collect();
    let candidate_chars: Vec<_> = candidate.chars().collect();
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

    let mut dp = vec![vec![[None; 5]; m + 1]; n + 1];
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
                let Some(cost) = dp[i][j][state_idx] else {
                    continue;
                };

                if i < n {
                    let mut next = cost;
                    next.operations = next.operations.with_deletion(
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
                        &mut dp[i + 1][j][next_state as usize],
                        next,
                        next_state as usize,
                    );
                }

                if j < m {
                    let mut next = cost;
                    next.distance += 1;
                    next.operations = next.operations.with_insertion(
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
                        &mut dp[i][j + 1][next_state as usize],
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
                        &mut dp[i + 1][j + 1][next_state as usize],
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
                        &mut dp[i + 2][j + 2][next_state as usize],
                        next,
                        next_state as usize,
                    );
                }
            }
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
        dp[n][m][state]
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
