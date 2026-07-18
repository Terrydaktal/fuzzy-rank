use std::cmp::Ordering;

#[derive(Clone, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub struct CoverageRank {
    pub weighted_quality: usize,
    pub distinct_coverage: usize,
    pub class_counts: Vec<usize>,
}

/// Compares occurrence counts by field priority, giving lower priorities
/// precedence and comparing counts within each priority.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct FieldOccurrenceRank(Vec<(u8, usize)>);

impl FieldOccurrenceRank {
    pub fn from_entries(entries: impl IntoIterator<Item = (u8, usize)>) -> Self {
        let mut entries = entries.into_iter().collect::<Vec<_>>();
        entries.sort_unstable_by_key(|(priority, _)| *priority);

        let mut merged = Vec::new();
        for (priority, count) in entries {
            if let Some((last_priority, last_count)) = merged.last_mut()
                && *last_priority == priority
            {
                *last_count += count;
            } else {
                merged.push((priority, count));
            }
        }
        Self(merged)
    }
}

impl Ord for FieldOccurrenceRank {
    fn cmp(&self, other: &Self) -> Ordering {
        let mut priorities = self
            .0
            .iter()
            .map(|(priority, _)| *priority)
            .chain(other.0.iter().map(|(priority, _)| *priority))
            .collect::<Vec<_>>();
        priorities.sort_unstable();
        priorities.dedup();

        priorities
            .into_iter()
            .map(|priority| {
                let left = self
                    .0
                    .iter()
                    .find(|(candidate_priority, _)| *candidate_priority == priority)
                    .map(|(_, count)| *count)
                    .unwrap_or(0);
                let right = other
                    .0
                    .iter()
                    .find(|(candidate_priority, _)| *candidate_priority == priority)
                    .map(|(_, count)| *count)
                    .unwrap_or(0);
                left.cmp(&right)
            })
            .find(|ordering| *ordering != Ordering::Equal)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for FieldOccurrenceRank {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
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

pub fn simple_position_vector(classes: impl IntoIterator<Item = usize>) -> [usize; 3] {
    let mut vector = [0; 3];
    for class in classes {
        increment_simple_position_vector(&mut vector, class, 1);
    }
    vector
}

/// Returns the best exact position class for a query within one candidate
/// token: whole, prefix, suffix, or arbitrary substring.
pub fn exact_token_position(query: &str, candidate: &str) -> Option<usize> {
    if candidate == query {
        Some(0)
    } else if candidate.starts_with(query) {
        Some(1)
    } else if candidate.ends_with(query) {
        Some(2)
    } else if candidate.contains(query) {
        Some(3)
    } else {
        None
    }
}

pub fn increment_simple_position_vector(vector: &mut [usize; 3], class: usize, count: usize) {
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

/// Returns whether the complete query phrase occurs at least once.
///
/// This is the structural phrase stage used by the adapters. It deliberately
/// stops at the first matching window; repeated occurrences are not stronger
/// evidence at this stage.
pub fn exact_phrase_present<'a, Q, C, I>(query_tokens: &[Q], fields: I) -> bool
where
    Q: AsRef<str>,
    C: AsRef<str> + 'a,
    I: IntoIterator<Item = &'a [C]>,
{
    exact_phrase_present_by(query_tokens, fields, |query, candidate| {
        candidate.starts_with(query)
    })
}

pub fn exact_phrase_present_by<'a, Q, C, I, F>(
    query_tokens: &[Q],
    fields: I,
    token_matches: F,
) -> bool
where
    Q: AsRef<str>,
    C: AsRef<str> + 'a,
    I: IntoIterator<Item = &'a [C]>,
    F: Fn(&str, &str) -> bool + Copy,
{
    if query_tokens.is_empty() {
        return false;
    }
    fields.into_iter().any(|tokens| {
        tokens
            .windows(query_tokens.len())
            .any(|window| token_window_matches(query_tokens, window, token_matches))
    })
}

/// Counts every matching complete-phrase window across the supplied fields.
///
/// Adapters should use this only for an explicit later occurrence tie-break,
/// never as the structural whole-phrase presence stage.
pub fn exact_phrase_occurrence_count<'a, Q, C, I>(query_tokens: &[Q], fields: I) -> usize
where
    Q: AsRef<str>,
    C: AsRef<str> + 'a,
    I: IntoIterator<Item = &'a [C]>,
{
    exact_phrase_occurrence_count_by(query_tokens, fields, |query, candidate| {
        candidate.starts_with(query)
    })
}

pub fn exact_phrase_occurrence_count_by<'a, Q, C, I, F>(
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

/// Counts distinct contiguous query subphrases present at least once.
///
/// Each vector entry corresponds to one subphrase length, starting with the
/// longest proper length. Repeated candidate occurrences of the same query
/// subphrase do not increase the entry.
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
            (0..=query_tokens.len() - window_len)
                .filter(|query_start| {
                    let query_window = &query_tokens[*query_start..*query_start + window_len];
                    fields.iter().any(|tokens| {
                        tokens
                            .windows(window_len)
                            .any(|window| token_window_matches(query_window, window, token_matches))
                    })
                })
                .count()
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

pub fn saturated_exact_phrase_occurrence_count(count: usize) -> usize {
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

/// Edit counts use candidate-to-query direction:
///
/// - `insertions` are candidate characters absent from the query.
/// - `deletions` are query characters absent from the candidate.
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
        candidate_char: char,
        prev: Option<char>,
        next: Option<char>,
    ) -> Self {
        self.insertions += 1;
        self.insertion_penalty += insertion_likelihood_penalty(candidate_char, prev, next);
        self
    }

    pub fn with_deletion(
        mut self,
        query_char: char,
        prev: Option<char>,
        next: Option<char>,
    ) -> Self {
        self.deletions += 1;
        self.substitution_penalty += deletion_likelihood_penalty(query_char, prev, next);
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
        self.keyboard_distance + self.deletions * 6
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
    candidate: &str,
    query: &str,
    limit: usize,
) -> Option<(usize, OperationProfile)> {
    let candidate: Vec<_> = candidate.chars().collect();
    let query: Vec<_> = query.chars().collect();
    if candidate.len().abs_diff(query.len()) > limit {
        return None;
    }

    let inf = EditCost::infinite(limit);
    let mut prev_prev = vec![inf; query.len() + 1];
    let mut prev = vec![EditCost::default(); query.len() + 1];
    for (idx, cell) in prev.iter_mut().enumerate() {
        *cell = EditCost {
            distance: idx,
            operations: OperationProfile {
                substitutions: 0,
                insertions: 0,
                deletions: idx,
                transpositions: 0,
                keyboard_distance: 0,
                substitution_penalty: 0,
                insertion_penalty: 0,
            },
        };
    }
    let mut curr = vec![inf; query.len() + 1];

    for i in 1..=candidate.len() {
        curr.fill(inf);
        curr[0] = EditCost {
            distance: i,
            operations: OperationProfile {
                substitutions: 0,
                insertions: i,
                deletions: 0,
                transpositions: 0,
                keyboard_distance: 0,
                substitution_penalty: 0,
                insertion_penalty: 0,
            },
        };

        let start = i.saturating_sub(limit).max(1);
        let end = (i + limit).min(query.len());
        if start > end {
            return None;
        }

        let mut row_min = inf;
        for j in start..=end {
            let insertion = prev[j].with_insertion(
                candidate[i - 1],
                i.checked_sub(2).map(|idx| candidate[idx]),
                candidate.get(i).copied(),
            );
            let deletion = curr[j - 1].with_deletion(
                query[j - 1],
                i.checked_sub(1).map(|idx| candidate[idx]),
                candidate.get(i).copied(),
            );
            let substitution = if candidate[i - 1] == query[j - 1] {
                prev[j - 1]
            } else {
                prev[j - 1].with_substitution(candidate[i - 1], query[j - 1])
            };
            let mut cell = deletion.min(insertion).min(substitution);

            if i > 1
                && j > 1
                && candidate[i - 1] == query[j - 2]
                && candidate[i - 2] == query[j - 1]
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

    let cost = prev[query.len()];
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

        assert!(!exact_phrase_present(&query, fields));
        assert_eq!(exact_subphrase_counts_vector(&query, fields), vec![1]);
        assert_eq!(exact_distinct_coverage(&query, fields), 3);
        assert_eq!(exact_token_occurrence_counts(&query, fields), vec![2, 2, 1]);
    }

    #[test]
    fn exact_phrase_presence_does_not_count_repeated_occurrences() {
        let query = vec!["one", "two"];
        let field = vec![
            "one".to_string(),
            "two".to_string(),
            "one".to_string(),
            "two".to_string(),
        ];

        assert!(exact_phrase_present(&query, [field.as_slice()]));
        assert_eq!(exact_phrase_occurrence_count(&query, [field.as_slice()]), 2);
    }

    #[test]
    fn exact_subphrase_vector_counts_presence_not_repeated_occurrences() {
        let query = vec!["one", "two", "three"];
        let field = vec![
            "one".to_string(),
            "two".to_string(),
            "one".to_string(),
            "two".to_string(),
        ];

        assert_eq!(
            exact_subphrase_counts_vector(&query, [field.as_slice()]),
            vec![1]
        );
    }

    #[test]
    fn query_token_occurrence_total_saturates_each_token_at_three() {
        assert_eq!(saturated_exact_query_token_occurrence_total(&[10, 2, 0]), 5);
        assert_eq!(saturated_exact_phrase_occurrence_count(10), 3);
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
        assert_eq!(simple_position_vector([0, 0, 1, 2]), [2, 1, 1]);
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

    #[test]
    fn query_and_candidate_only_characters_have_directional_costs() {
        let query_deletion = bounded_damerau_levenshtein("target", "tarxget", 1).unwrap();
        assert_eq!(query_deletion.1.deletions, 1);
        assert_eq!(query_deletion.1.insertions, 0);
        assert_eq!(query_deletion.1.core_typo_cost(), 6);

        let candidate_insertion = bounded_damerau_levenshtein("tarxget", "target", 1).unwrap();
        assert_eq!(candidate_insertion.1.deletions, 0);
        assert_eq!(candidate_insertion.1.insertions, 1);
        assert_eq!(candidate_insertion.1.core_typo_cost(), 0);
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
