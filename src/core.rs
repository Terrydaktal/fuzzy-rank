use std::cmp::Ordering;

#[derive(Clone, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub struct CoverageRank {
    pub weighted_quality: usize,
    pub distinct_coverage: usize,
    pub class_counts: Vec<usize>,
}

impl CoverageRank {
    pub fn new(class_counts: Vec<usize>, weights: &[usize]) -> Self {
        debug_assert_eq!(class_counts.len(), weights.len());
        let weighted_quality = class_counts
            .iter()
            .zip(weights)
            .map(|(count, weight)| count.saturating_mul(*weight))
            .sum();
        let distinct_coverage = class_counts.iter().sum();
        Self {
            weighted_quality,
            distinct_coverage,
            class_counts,
        }
    }
}

pub fn position_vector(
    classes: impl IntoIterator<Item = Option<usize>>,
    class_count: usize,
) -> Vec<usize> {
    let mut vector = vec![0; class_count];
    for class in classes.into_iter().flatten() {
        if let Some(count) = vector.get_mut(class) {
            *count += 1;
        }
    }
    vector
}

pub fn position_vector_simple(classes: impl IntoIterator<Item = usize>) -> [usize; 3] {
    let mut vector = [0; 3];
    for class in classes {
        increment_position_vector_simple(&mut vector, class, 1);
    }
    vector
}

pub fn increment_position_vector_simple(vector: &mut [usize; 3], class: usize, count: usize) {
    vector[class.min(2)] += count;
}

pub fn compare_score_desc(left: f64, right: f64) -> Ordering {
    right.total_cmp(&left)
}

pub fn compare_key_asc(left: &str, right: &str) -> Ordering {
    left.cmp(right)
}

pub fn ratio_milli(distance: usize, compared_len: usize) -> usize {
    distance * 1000 / compared_len.max(1)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DensityRank {
    pub matched_tokens: usize,
    pub field_len: usize,
}

impl DensityRank {
    pub fn new(matched_tokens: usize, field_len: usize) -> Self {
        Self {
            matched_tokens,
            field_len,
        }
    }
}

impl Ord for DensityRank {
    fn cmp(&self, other: &Self) -> Ordering {
        ((other.matched_tokens as u128) * (self.field_len as u128))
            .cmp(&((self.matched_tokens as u128) * (other.field_len as u128)))
            .then_with(|| self.field_len.cmp(&other.field_len))
    }
}

impl PartialOrd for DensityRank {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub fn exact_phrase_count<'a, Q, C, I>(query_tokens: &[Q], fields: I) -> usize
where
    Q: AsRef<str>,
    C: AsRef<str> + 'a,
    I: IntoIterator<Item = &'a [C]>,
{
    exact_phrase_count_by(query_tokens, fields, |query, candidate| {
        candidate.starts_with(query)
    })
}

pub fn exact_phrase_count_by<'a, Q, C, I, F>(
    query_tokens: &[Q],
    fields: I,
    token_matches: F,
) -> usize
where
    Q: AsRef<str>,
    C: AsRef<str> + 'a,
    I: IntoIterator<Item = &'a [C]>,
    F: Fn(&str, &str) -> bool + Copy,
{
    if query_tokens.is_empty() {
        return 0;
    }
    fields
        .into_iter()
        .map(|tokens| {
            tokens
                .windows(query_tokens.len())
                .filter(|window| token_window_matches(query_tokens, window, token_matches))
                .count()
        })
        .sum()
}

pub fn exact_subphrase_counts_vector<'a, Q, C, I>(query_tokens: &[Q], fields: I) -> Vec<usize>
where
    Q: AsRef<str>,
    C: AsRef<str> + 'a,
    I: IntoIterator<Item = &'a [C]>,
{
    exact_subphrase_counts_vector_by(query_tokens, fields, |query, candidate| {
        candidate.starts_with(query)
    })
}

pub fn exact_subphrase_counts_vector_by<'a, Q, C, I, F>(
    query_tokens: &[Q],
    fields: I,
    token_matches: F,
) -> Vec<usize>
where
    Q: AsRef<str>,
    C: AsRef<str> + 'a,
    I: IntoIterator<Item = &'a [C]>,
    F: Fn(&str, &str) -> bool + Copy,
{
    if query_tokens.len() < 3 {
        return Vec::new();
    }
    let fields = fields.into_iter().collect::<Vec<_>>();
    (2..query_tokens.len())
        .rev()
        .map(|window_len| {
            let mut count = 0;
            for query_start in 0..=query_tokens.len() - window_len {
                let query_window = &query_tokens[query_start..query_start + window_len];
                count += fields
                    .iter()
                    .map(|tokens| {
                        tokens
                            .windows(window_len)
                            .filter(|window| {
                                token_window_matches(query_window, window, token_matches)
                            })
                            .count()
                    })
                    .sum::<usize>();
            }
            count
        })
        .collect()
}

pub fn exact_distinct_coverage<'a, Q, C, I>(query_tokens: &[Q], fields: I) -> usize
where
    Q: AsRef<str>,
    C: AsRef<str> + 'a,
    I: IntoIterator<Item = &'a [C]>,
{
    exact_distinct_coverage_by(query_tokens, fields, |query, candidate| {
        candidate.starts_with(query)
    })
}

pub fn exact_distinct_coverage_by<'a, Q, C, I, F>(
    query_tokens: &[Q],
    fields: I,
    token_matches: F,
) -> usize
where
    Q: AsRef<str>,
    C: AsRef<str> + 'a,
    I: IntoIterator<Item = &'a [C]>,
    F: Fn(&str, &str) -> bool + Copy,
{
    let fields = fields.into_iter().collect::<Vec<_>>();
    query_tokens
        .iter()
        .filter(|query| {
            fields.iter().any(|tokens| {
                tokens
                    .iter()
                    .any(|candidate| token_matches(query.as_ref(), candidate.as_ref()))
            })
        })
        .count()
}

pub fn exact_token_occurrence_counts<'a, Q, C, I>(query_tokens: &[Q], fields: I) -> Vec<usize>
where
    Q: AsRef<str>,
    C: AsRef<str> + 'a,
    I: IntoIterator<Item = &'a [C]>,
{
    exact_token_occurrence_counts_by(query_tokens, fields, |query, candidate| {
        candidate.starts_with(query)
    })
}

pub fn exact_token_occurrence_counts_by<'a, Q, C, I, F>(
    query_tokens: &[Q],
    fields: I,
    token_matches: F,
) -> Vec<usize>
where
    Q: AsRef<str>,
    C: AsRef<str> + 'a,
    I: IntoIterator<Item = &'a [C]>,
    F: Fn(&str, &str) -> bool + Copy,
{
    let fields = fields.into_iter().collect::<Vec<_>>();
    query_tokens
        .iter()
        .map(|query| {
            fields
                .iter()
                .map(|tokens| {
                    tokens
                        .iter()
                        .filter(|candidate| token_matches(query.as_ref(), candidate.as_ref()))
                        .count()
                })
                .sum()
        })
        .collect()
}

pub fn saturated_exact_query_token_occurrence_total(counts: &[usize]) -> usize {
    counts.iter().map(|count| (*count).min(3)).sum()
}

pub fn saturated_exact_phrase_count(count: usize) -> usize {
    count.min(3)
}

pub fn inversion_count(values: &[usize]) -> usize {
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            values[index + 1..]
                .iter()
                .filter(|later| *later < value)
                .count()
        })
        .sum()
}

pub fn ordered_token_span<'a, Q, C, I, F>(
    query_tokens: &[Q],
    fields: I,
    mut token_matches: F,
) -> (usize, usize)
where
    Q: AsRef<str>,
    C: AsRef<str> + 'a,
    I: IntoIterator<Item = &'a [C]>,
    F: FnMut(&str, &str) -> bool,
{
    if query_tokens.is_empty() {
        return (usize::MAX, usize::MAX);
    }

    let mut best: Option<(usize, usize, usize)> = None;
    for field in fields {
        let mut used = vec![false; field.len()];
        let mut positions = Vec::new();
        for query in query_tokens {
            let Some((idx, _)) = field.iter().enumerate().find(|(idx, candidate)| {
                !used[*idx] && token_matches(query.as_ref(), candidate.as_ref())
            }) else {
                continue;
            };
            used[idx] = true;
            positions.push(idx);
        }

        if positions.is_empty() {
            continue;
        }

        let order_distance = inversion_count(&positions);
        let total_gap = positions
            .windows(2)
            .map(|pair| pair[0].abs_diff(pair[1]).saturating_sub(1))
            .sum::<usize>();
        let candidate = (positions.len(), order_distance, total_gap);
        if best.is_none_or(|current| {
            candidate.0 > current.0
                || (candidate.0 == current.0 && candidate.1 < current.1)
                || (candidate.0 == current.0 && candidate.1 == current.1 && candidate.2 < current.2)
        }) {
            best = Some(candidate);
        }
    }

    best.map_or((usize::MAX, usize::MAX), |(_, order, gap)| (order, gap))
}

fn token_window_matches<Q: AsRef<str>, C: AsRef<str>, F>(
    query: &[Q],
    candidate: &[C],
    token_matches: F,
) -> bool
where
    F: Fn(&str, &str) -> bool + Copy,
{
    candidate
        .iter()
        .zip(query)
        .all(|(candidate, query)| token_matches(query.as_ref(), candidate.as_ref()))
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OperationProfile {
    pub substitutions: usize,
    pub insertions: usize,
    pub deletions: usize,
    pub transpositions: usize,
    pub keyboard_distance: usize,
    pub substitution_penalty: usize,
    pub insertion_penalty: usize,
}

impl Ord for OperationProfile {
    fn cmp(&self, other: &Self) -> Ordering {
        (
            self.total_penalty(),
            self.substitution_penalty,
            self.insertion_penalty,
            self.substitutions,
            self.transpositions,
            self.deletions,
            self.insertions,
            self.keyboard_distance,
        )
            .cmp(&(
                other.total_penalty(),
                other.substitution_penalty,
                other.insertion_penalty,
                other.substitutions,
                other.transpositions,
                other.deletions,
                other.insertions,
                other.keyboard_distance,
            ))
    }
}

impl PartialOrd for OperationProfile {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl OperationProfile {
    pub fn with_substitution(mut self, left: char, right: char) -> Self {
        self.substitutions += 1;
        let keyboard_distance = keyboard_substitution_cost(left, right);
        self.keyboard_distance += keyboard_distance;
        self.substitution_penalty += substitution_likelihood_penalty(keyboard_distance);
        self
    }

    pub fn with_insertion(
        mut self,
        inserted: char,
        prev: Option<char>,
        next: Option<char>,
    ) -> Self {
        self.insertions += 1;
        self.insertion_penalty += insertion_likelihood_penalty(inserted, prev, next);
        self
    }

    pub fn with_deletion(mut self, deleted: char, prev: Option<char>, next: Option<char>) -> Self {
        self.deletions += 1;
        self.substitution_penalty += deletion_likelihood_penalty(deleted, prev, next);
        self
    }

    pub fn with_transposition(mut self) -> Self {
        self.transpositions += 1;
        self.substitution_penalty += 65;
        self
    }

    pub fn merge(mut self, other: Self) -> Self {
        self.substitutions += other.substitutions;
        self.insertions += other.insertions;
        self.deletions += other.deletions;
        self.transpositions += other.transpositions;
        self.keyboard_distance += other.keyboard_distance;
        self.substitution_penalty += other.substitution_penalty;
        self.insertion_penalty += other.insertion_penalty;
        self
    }

    fn total_penalty(&self) -> usize {
        self.substitution_penalty + self.insertion_penalty
    }

    pub fn core_typo_cost(&self) -> usize {
        self.keyboard_distance + self.insertions * 6
    }
}

pub fn keyboard_substitution_cost(left: char, right: char) -> usize {
    if left == right {
        return 0;
    }
    let Some((left_x, left_y)) = qwerty_position(left) else {
        return 100;
    };
    let Some((right_x, right_y)) = qwerty_position(right) else {
        return 100;
    };
    left_x.abs_diff(right_x) + left_y.abs_diff(right_y)
}

fn qwerty_position(ch: char) -> Option<(usize, usize)> {
    let ch = ch.to_ascii_lowercase();
    let rows = [
        ("1234567890", 0usize, 0usize),
        ("qwertyuiop", 1usize, 0usize),
        ("asdfghjkl", 2usize, 1usize),
        ("zxcvbnm", 3usize, 3usize),
    ];
    for (row_idx, (row, x_offset, y_scale)) in rows.iter().enumerate() {
        if let Some(col_idx) = row.chars().position(|candidate| candidate == ch) {
            return Some((x_offset + col_idx * 2, row_idx * 2 + y_scale));
        }
    }
    None
}

fn substitution_likelihood_penalty(keyboard_distance: usize) -> usize {
    keyboard_distance
}

fn insertion_likelihood_penalty(
    _inserted: char,
    _prev: Option<char>,
    _next: Option<char>,
) -> usize {
    6
}

fn deletion_likelihood_penalty(_deleted: char, _prev: Option<char>, _next: Option<char>) -> usize {
    6
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct EditCost {
    distance: usize,
    operations: OperationProfile,
}

impl EditCost {
    fn infinite(limit: usize) -> Self {
        Self {
            distance: limit + 1,
            operations: OperationProfile::default(),
        }
    }

    fn with_insertion(self, inserted: char, prev: Option<char>, next: Option<char>) -> Self {
        Self {
            distance: self.distance + 1,
            operations: self.operations.with_insertion(inserted, prev, next),
        }
    }

    fn with_deletion(self, deleted: char, prev: Option<char>, next: Option<char>) -> Self {
        Self {
            distance: self.distance + 1,
            operations: self.operations.with_deletion(deleted, prev, next),
        }
    }

    fn with_substitution(self, left: char, right: char) -> Self {
        Self {
            distance: self.distance + 1,
            operations: self.operations.with_substitution(left, right),
        }
    }

    fn with_transposition(self) -> Self {
        Self {
            distance: self.distance + 1,
            operations: self.operations.with_transposition(),
        }
    }
}

impl Ord for EditCost {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .cmp(&other.distance)
            .then_with(|| self.operations.cmp(&other.operations))
    }
}

impl PartialOrd for EditCost {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub fn max_typos(len: usize) -> usize {
    len / 2
}

pub fn bounded_damerau_levenshtein(
    intended: &str,
    typed: &str,
    limit: usize,
) -> Option<(usize, OperationProfile)> {
    let intended: Vec<_> = intended.chars().collect();
    let typed: Vec<_> = typed.chars().collect();
    if intended.len().abs_diff(typed.len()) > limit {
        return None;
    }

    let inf = EditCost::infinite(limit);
    let mut prev_prev = vec![inf; typed.len() + 1];
    let mut prev = vec![EditCost::default(); typed.len() + 1];
    for (idx, cell) in prev.iter_mut().enumerate() {
        *cell = EditCost {
            distance: idx,
            operations: OperationProfile {
                substitutions: 0,
                insertions: idx,
                deletions: 0,
                transpositions: 0,
                keyboard_distance: 0,
                substitution_penalty: 0,
                insertion_penalty: 0,
            },
        };
    }
    let mut curr = vec![inf; typed.len() + 1];

    for i in 1..=intended.len() {
        curr.fill(inf);
        curr[0] = EditCost {
            distance: i,
            operations: OperationProfile {
                substitutions: 0,
                insertions: 0,
                deletions: i,
                transpositions: 0,
                keyboard_distance: 0,
                substitution_penalty: 0,
                insertion_penalty: 0,
            },
        };

        let start = i.saturating_sub(limit).max(1);
        let end = (i + limit).min(typed.len());
        if start > end {
            return None;
        }

        let mut row_min = inf;
        for j in start..=end {
            let deletion = prev[j].with_deletion(
                intended[i - 1],
                i.checked_sub(2).map(|idx| intended[idx]),
                intended.get(i).copied(),
            );
            let insertion = curr[j - 1].with_insertion(
                typed[j - 1],
                i.checked_sub(1).map(|idx| intended[idx]),
                intended.get(i).copied(),
            );
            let substitution = if intended[i - 1] == typed[j - 1] {
                prev[j - 1]
            } else {
                prev[j - 1].with_substitution(intended[i - 1], typed[j - 1])
            };
            let mut cell = deletion.min(insertion).min(substitution);

            if i > 1 && j > 1 && intended[i - 1] == typed[j - 2] && intended[i - 2] == typed[j - 1]
            {
                cell = cell.min(prev_prev[j - 2].with_transposition());
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

    let cost = prev[typed.len()];
    (cost.distance <= limit).then_some((cost.distance, cost.operations))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_relevance_primitives_work_across_fields() {
        let query = vec!["one".to_string(), "two".to_string(), "three".to_string()];
        let first = vec!["one".to_string(), "twofold".to_string()];
        let second = vec!["three".to_string(), "one".to_string(), "two".to_string()];
        let fields = [first.as_slice(), second.as_slice()];

        assert_eq!(exact_phrase_count(&query, fields), 0);
        assert_eq!(exact_subphrase_counts_vector(&query, fields), vec![2]);
        assert_eq!(exact_distinct_coverage(&query, fields), 3);
        assert_eq!(exact_token_occurrence_counts(&query, fields), vec![2, 2, 1]);
    }

    #[test]
    fn query_token_occurrence_total_saturates_each_token_at_three() {
        assert_eq!(saturated_exact_query_token_occurrence_total(&[10, 2, 0]), 5);
        assert_eq!(saturated_exact_phrase_count(10), 3);
    }

    #[test]
    fn coverage_rank_balances_quality_and_distinct_coverage() {
        let weights = [4, 2, 2, 1, 1];
        let one_whole = CoverageRank::new(vec![1, 0, 0, 0, 0], &weights);
        let three_prefixes = CoverageRank::new(vec![0, 3, 0, 0, 0], &weights);
        let two_whole = CoverageRank::new(vec![2, 0, 0, 0, 0], &weights);

        assert!(three_prefixes > one_whole);
        assert!(two_whole > three_prefixes);
    }

    #[test]
    fn typo_cost_vector_retains_query_order_but_compares_worst_cost_first() {
        let left = TypoCostVector::new(vec![1, 3, 0]);
        let same_errors_different_order = TypoCostVector::new(vec![3, 0, 1]);
        let lower_worst_error = TypoCostVector::new(vec![2, 0, 0]);

        assert_eq!(left.aligned(), [1, 3, 0]);
        assert_eq!(left.worst_first(), [3, 1, 0]);
        assert_eq!(left.cmp(&same_errors_different_order), Ordering::Equal);
        assert!(lower_worst_error < left);
    }

    #[test]
    fn shared_position_vectors_support_simple_and_configurable_class_maps() {
        assert_eq!(position_vector_simple([0, 0, 1, 2]), [2, 1, 1]);
        assert_eq!(
            position_vector([Some(0), Some(1), Some(3), None], 4),
            [1, 1, 0, 1]
        );
    }

    #[test]
    fn density_prefers_higher_ratio_then_shorter_field() {
        assert!(DensityRank::new(2, 4) < DensityRank::new(1, 4));
        assert!(DensityRank::new(1, 2) < DensityRank::new(2, 6));
        assert!(DensityRank::new(1, 2) < DensityRank::new(2, 4));
    }

    #[test]
    fn ordered_span_uses_caller_match_policy_across_fields() {
        let query = ["one", "two", "three"];
        let scattered = ["one", "x", "two", "y", "three"];
        let closer = ["one", "two", "three"];
        assert_eq!(
            ordered_token_span(&query, [&scattered[..], &closer[..]], |query, candidate| {
                candidate.starts_with(query)
            }),
            (0, 0)
        );
    }

    #[test]
    fn ordered_span_scores_partial_matches_by_order_and_gap() {
        let query = ["one", "two", "three"];
        let partial = ["three", "x", "one"];
        assert_eq!(
            ordered_token_span(&query, [&partial[..]], |query, candidate| {
                candidate.starts_with(query)
            }),
            (1, 1)
        );
    }

    #[test]
    fn transpositions_are_cheaper_than_substitutions_as_a_tiebreak() {
        let swapped = bounded_damerau_levenshtein("abcd", "abdc", 2).unwrap();
        let substituted = bounded_damerau_levenshtein("abcd", "abxd", 2).unwrap();
        assert_eq!(swapped.0, substituted.0);
        assert!(substituted.1 < swapped.1);
    }

    #[test]
    fn nearby_key_substitutions_are_cheaper_than_distant_ones() {
        let nearby = bounded_damerau_levenshtein("film", "gilm", 1).unwrap();
        let distant = bounded_damerau_levenshtein("film", "pilm", 1).unwrap();
        assert_eq!(nearby.0, distant.0);
        assert!(nearby.1 < distant.1);
    }

    #[test]
    fn nearby_substitution_is_cheaper_than_transposition() {
        let nearby = bounded_damerau_levenshtein("film", "gilm", 1).unwrap();
        let swapped = bounded_damerau_levenshtein("film", "flim", 1).unwrap();
        assert_eq!(nearby.0, swapped.0);
        assert!(nearby.1 < swapped.1);
    }

    #[test]
    fn missing_character_is_cheaper_than_transposition() {
        let swapped = bounded_damerau_levenshtein("film", "flim", 1).unwrap();
        let missing = bounded_damerau_levenshtein("film", "flm", 1).unwrap();
        assert_eq!(swapped.0, missing.0);
        assert!(missing.1 < swapped.1);
    }

    #[test]
    fn missing_character_has_same_cost_as_extra_character() {
        let missing = bounded_damerau_levenshtein("film", "flm", 1).unwrap();
        let extra = bounded_damerau_levenshtein("film", "filmx", 1).unwrap();
        assert_eq!(missing.0, extra.0);
        assert_eq!(
            missing.1.substitution_penalty + missing.1.insertion_penalty,
            extra.1.substitution_penalty + extra.1.insertion_penalty
        );
    }

    #[test]
    fn arbitrary_insertion_is_cheaper_than_far_substitution() {
        let extra = bounded_damerau_levenshtein("film", "filmx", 1).unwrap();
        let far = bounded_damerau_levenshtein("film", "pilm", 1).unwrap();
        assert_eq!(extra.0, far.0);
        assert!(extra.1 < far.1);
    }
}
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RankedTokenVector {
    aligned: Vec<usize>,
    worst_first: Vec<usize>,
}

impl RankedTokenVector {
    pub fn new(aligned: Vec<usize>) -> Self {
        let mut worst_first = aligned.clone();
        worst_first.sort_unstable_by(|left, right| right.cmp(left));
        Self {
            aligned,
            worst_first,
        }
    }

    pub fn aligned(&self) -> &[usize] {
        &self.aligned
    }

    pub fn worst_first(&self) -> &[usize] {
        &self.worst_first
    }
}

impl Ord for RankedTokenVector {
    fn cmp(&self, other: &Self) -> Ordering {
        self.worst_first.cmp(&other.worst_first)
    }
}

impl PartialOrd for RankedTokenVector {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub type TypoCostVector = RankedTokenVector;
pub type InsertionVector = RankedTokenVector;
