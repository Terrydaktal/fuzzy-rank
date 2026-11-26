use super::*;

pub(super) fn fragmented_token_match_reference(
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
    dp.resize_with(n + 1, || vec![[None; 5]; m + 1]);
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
                let Some(cost) = dp[i][j][state_idx] else {
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
                        &mut dp[i + 1][j][next_state as usize],
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
