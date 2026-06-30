use std::cmp::Ordering;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OperationProfile {
    pub substitutions: usize,
    pub insert_delete: usize,
    pub transpositions: usize,
    pub keyboard_distance: usize,
}

impl Ord for OperationProfile {
    fn cmp(&self, other: &Self) -> Ordering {
        (
            self.substitutions,
            self.insert_delete,
            self.transpositions,
            self.keyboard_distance,
        )
            .cmp(&(
                other.substitutions,
                other.insert_delete,
                other.transpositions,
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
        self.keyboard_distance += keyboard_substitution_cost(left, right);
        self
    }

    pub fn with_insert_delete(mut self) -> Self {
        self.insert_delete += 1;
        self
    }

    pub fn with_transposition(mut self) -> Self {
        self.transpositions += 1;
        self
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

    fn with_insert_delete(self) -> Self {
        Self {
            distance: self.distance + 1,
            operations: self.operations.with_insert_delete(),
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
    left: &str,
    right: &str,
    limit: usize,
) -> Option<(usize, OperationProfile)> {
    let left: Vec<_> = left.chars().collect();
    let right: Vec<_> = right.chars().collect();
    if left.len().abs_diff(right.len()) > limit {
        return None;
    }

    let inf = EditCost::infinite(limit);
    let mut prev_prev = vec![inf; right.len() + 1];
    let mut prev = vec![EditCost::default(); right.len() + 1];
    for (idx, cell) in prev.iter_mut().enumerate() {
        *cell = EditCost {
            distance: idx,
            operations: OperationProfile {
                substitutions: 0,
                insert_delete: idx,
                transpositions: 0,
                keyboard_distance: 0,
            },
        };
    }
    let mut curr = vec![inf; right.len() + 1];

    for i in 1..=left.len() {
        curr.fill(inf);
        curr[0] = EditCost {
            distance: i,
            operations: OperationProfile {
                substitutions: 0,
                insert_delete: i,
                transpositions: 0,
                keyboard_distance: 0,
            },
        };

        let start = i.saturating_sub(limit).max(1);
        let end = (i + limit).min(right.len());
        if start > end {
            return None;
        }

        let mut row_min = inf;
        for j in start..=end {
            let deletion = prev[j].with_insert_delete();
            let insertion = curr[j - 1].with_insert_delete();
            let substitution = if left[i - 1] == right[j - 1] {
                prev[j - 1]
            } else {
                prev[j - 1].with_substitution(left[i - 1], right[j - 1])
            };
            let mut cell = deletion.min(insertion).min(substitution);

            if i > 1 && j > 1 && left[i - 1] == right[j - 2] && left[i - 2] == right[j - 1] {
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

    let cost = prev[right.len()];
    (cost.distance <= limit).then_some((cost.distance, cost.operations))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpositions_are_cheaper_than_substitutions_as_a_tiebreak() {
        let swapped = bounded_damerau_levenshtein("abdc", "abcd", 2).unwrap();
        let substituted = bounded_damerau_levenshtein("abxd", "abcd", 2).unwrap();
        assert_eq!(swapped.0, substituted.0);
        assert!(swapped.1 < substituted.1);
    }

    #[test]
    fn nearby_key_substitutions_are_cheaper_than_distant_ones() {
        let nearby = bounded_damerau_levenshtein("gilm", "film", 1).unwrap();
        let distant = bounded_damerau_levenshtein("pilm", "film", 1).unwrap();
        assert_eq!(nearby.0, distant.0);
        assert!(nearby.1 < distant.1);
    }
}
