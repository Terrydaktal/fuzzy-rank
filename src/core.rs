use std::cmp::Ordering;

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

    pub fn with_insertion(mut self, inserted: char, prev: Option<char>, next: Option<char>) -> Self {
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
        self.substitution_penalty
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
    if keyboard_distance == 0 {
        return 10;
    }
    if keyboard_distance <= 4 {
        45 + keyboard_distance
    } else {
        130 + keyboard_distance
    }
}

fn insertion_likelihood_penalty(inserted: char, prev: Option<char>, next: Option<char>) -> usize {
    let inserted = inserted.to_ascii_lowercase();
    let prev = prev.map(|ch| ch.to_ascii_lowercase());
    let next = next.map(|ch| ch.to_ascii_lowercase());
    if prev == Some(inserted) || next == Some(inserted) {
        return 80;
    }
    if prev.is_some_and(|ch| keyboard_substitution_cost(ch, inserted) <= 4)
        || next.is_some_and(|ch| keyboard_substitution_cost(ch, inserted) <= 4)
    {
        return 105;
    }
    145
}

fn deletion_likelihood_penalty(deleted: char, prev: Option<char>, next: Option<char>) -> usize {
    let deleted = deleted.to_ascii_lowercase();
    let prev = prev.map(|ch| ch.to_ascii_lowercase());
    let next = next.map(|ch| ch.to_ascii_lowercase());
    if prev == Some(deleted) || next == Some(deleted) {
        return 35;
    }
    40
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

            if i > 1
                && j > 1
                && intended[i - 1] == typed[j - 2]
                && intended[i - 2] == typed[j - 1]
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
    fn missing_character_is_cheaper_than_extra_character() {
        let missing = bounded_damerau_levenshtein("film", "flm", 1).unwrap();
        let extra = bounded_damerau_levenshtein("film", "filmx", 1).unwrap();
        assert_eq!(missing.0, extra.0);
        assert!(missing.1 < extra.1);
    }

    #[test]
    fn far_substitution_is_cheaper_than_arbitrary_insertion() {
        let extra = bounded_damerau_levenshtein("film", "filmx", 1).unwrap();
        let far = bounded_damerau_levenshtein("film", "pilm", 1).unwrap();
        assert_eq!(extra.0, far.0);
        assert!(far.1 < extra.1);
    }
}
