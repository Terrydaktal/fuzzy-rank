use std::cmp::Ordering;

use crate::core::{
    InsertionVector, OperationProfile, TypoCostVector, compare_key_asc, compare_score_desc,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructuralRank {
    pub field_index: usize,
    pub field_priority: u8,
    pub position_class: u8,
    pub token_index: usize,
    pub token_span_delta: usize,
    pub start_idx: usize,
    pub matched_char_len: usize,
    pub field_len: usize,
}

impl Ord for StructuralRank {
    fn cmp(&self, other: &Self) -> Ordering {
        (
            self.field_priority,
            self.position_class,
            self.token_index,
            self.token_span_delta,
            self.start_idx,
            self.field_len,
        )
            .cmp(&(
                other.field_priority,
                other.position_class,
                other.token_index,
                other.token_span_delta,
                other.start_idx,
                other.field_len,
            ))
    }
}

impl PartialOrd for StructuralRank {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DistanceRank {
    pub distance: usize,
    pub ratio_milli: usize,
    pub keyboard_distance: usize,
    pub field_index: usize,
    pub field_priority: u8,
    pub variant_scope: u8,
    pub position_class: u8,
    pub token_index: usize,
    pub token_span_delta: usize,
    pub start_idx: usize,
    pub matched_char_len: usize,
    pub field_len: usize,
}

impl Ord for DistanceRank {
    fn cmp(&self, other: &Self) -> Ordering {
        (
            self.distance,
            self.ratio_milli,
            self.keyboard_distance,
            self.field_priority,
            self.variant_scope,
            self.position_class,
            self.token_index,
            self.token_span_delta,
            self.start_idx,
            self.field_len,
        )
            .cmp(&(
                other.distance,
                other.ratio_milli,
                other.keyboard_distance,
                other.field_priority,
                other.variant_scope,
                other.position_class,
                other.token_index,
                other.token_span_delta,
                other.start_idx,
                other.field_len,
            ))
    }
}

impl PartialOrd for DistanceRank {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AbbreviationRank {
    pub field_index: usize,
    pub field_priority: u8,
    pub variant_scope: u8,
    pub position_class: u8,
    pub token_index: usize,
    pub gap_span: usize,
    pub gap_count: usize,
    pub start_idx: usize,
    pub matched_char_len: usize,
    pub field_len: usize,
}

impl Ord for AbbreviationRank {
    fn cmp(&self, other: &Self) -> Ordering {
        (
            self.field_priority,
            self.variant_scope,
            self.position_class,
            self.token_index,
            self.gap_span,
            self.gap_count,
            self.start_idx,
            self.field_len,
        )
            .cmp(&(
                other.field_priority,
                other.variant_scope,
                other.position_class,
                other.token_index,
                other.gap_span,
                other.gap_count,
                other.start_idx,
                other.field_len,
            ))
    }
}

impl PartialOrd for AbbreviationRank {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SearchRank {
    Structural(StructuralRank),
    Abbreviation(AbbreviationRank),
    Fuzzy(DistanceRank),
    Typo(DistanceRank),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatchProvenance {
    pub field_index: usize,
    pub field_priority: u8,
    pub variant_scope: Option<u8>,
    pub token_index: usize,
    pub start_idx: usize,
    pub matched_char_len: usize,
    pub field_len: usize,
}

impl Ord for SearchRank {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (SearchRank::Structural(a), SearchRank::Structural(b)) => a.cmp(b),
            (SearchRank::Structural(_), _) => Ordering::Less,
            (_, SearchRank::Structural(_)) => Ordering::Greater,
            (SearchRank::Abbreviation(a), SearchRank::Abbreviation(b)) => a.cmp(b),
            (SearchRank::Abbreviation(_), _) => Ordering::Less,
            (_, SearchRank::Abbreviation(_)) => Ordering::Greater,
            (SearchRank::Fuzzy(a), SearchRank::Fuzzy(b)) => a.cmp(b),
            (SearchRank::Fuzzy(_), SearchRank::Typo(_)) => Ordering::Less,
            (SearchRank::Typo(_), SearchRank::Fuzzy(_)) => Ordering::Greater,
            (SearchRank::Typo(a), SearchRank::Typo(b)) => a.cmp(b),
        }
    }
}

impl PartialOrd for SearchRank {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl SearchRank {
    pub fn provenance(&self) -> MatchProvenance {
        match self {
            SearchRank::Structural(rank) => MatchProvenance {
                field_index: rank.field_index,
                field_priority: rank.field_priority,
                variant_scope: None,
                token_index: rank.token_index,
                start_idx: rank.start_idx,
                matched_char_len: rank.matched_char_len,
                field_len: rank.field_len,
            },
            SearchRank::Abbreviation(rank) => MatchProvenance {
                field_index: rank.field_index,
                field_priority: rank.field_priority,
                variant_scope: Some(rank.variant_scope),
                token_index: rank.token_index,
                start_idx: rank.start_idx,
                matched_char_len: rank.matched_char_len,
                field_len: rank.field_len,
            },
            SearchRank::Fuzzy(rank) | SearchRank::Typo(rank) => MatchProvenance {
                field_index: rank.field_index,
                field_priority: rank.field_priority,
                variant_scope: Some(rank.variant_scope),
                token_index: rank.token_index,
                start_idx: rank.start_idx,
                matched_char_len: rank.matched_char_len,
                field_len: rank.field_len,
            },
        }
    }
}

pub fn compare_search_results(
    left_rank: &SearchRank,
    left_score: f64,
    left_key: &str,
    right_rank: &SearchRank,
    right_score: f64,
    right_key: &str,
) -> Ordering {
    left_rank
        .cmp(right_rank)
        .then_with(|| right_score.total_cmp(&left_score))
        .then_with(|| left_key.cmp(right_key))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PathTypoSortKey<'a> {
    pub distance: usize,
    pub order_distance: usize,
    pub operations: OperationProfile,
    pub typo_costs: &'a TypoCostVector,
    pub insertions: &'a InsertionVector,
    pub segment: &'a [usize],
    pub position: &'a [usize; 3],
    pub score: f64,
    pub key: &'a str,
}

fn path_depth(path: &str) -> usize {
    path.chars().filter(|ch| matches!(ch, '/' | '\\')).count()
}

pub fn compare_path_typo_sort_keys(
    left: &PathTypoSortKey<'_>,
    right: &PathTypoSortKey<'_>,
) -> Ordering {
    left.typo_costs
        .cmp(right.typo_costs)
        .then_with(|| left.order_distance.cmp(&right.order_distance))
        .then_with(|| compare_path_tail(left, right))
}

fn compare_path_tail(left: &PathTypoSortKey<'_>, right: &PathTypoSortKey<'_>) -> Ordering {
    right
        .segment
        .cmp(left.segment)
        .then_with(|| right.position.cmp(left.position))
        .then_with(|| compare_score_desc(left.score, right.score))
        .then_with(|| left.insertions.cmp(right.insertions))
        .then_with(|| path_depth(left.key).cmp(&path_depth(right.key)))
        .then_with(|| compare_key_asc(left.key, right.key))
}

pub fn path_typo_keys_are_ambiguous(
    left: &PathTypoSortKey<'_>,
    right: &PathTypoSortKey<'_>,
) -> bool {
    left.typo_costs == right.typo_costs
        && left.order_distance == right.order_distance
        && left.segment == right.segment
        && left.position == right.position
        && left.score.total_cmp(&right.score).is_eq()
        && left.insertions == right.insertions
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path_key<'a>(
        segment: &'a [usize],
        position: &'a [usize; 3],
        typo_costs: &'a TypoCostVector,
        insertions: &'a InsertionVector,
    ) -> PathTypoSortKey<'a> {
        PathTypoSortKey {
            distance: 1,
            order_distance: 0,
            operations: OperationProfile::default(),
            typo_costs,
            insertions,
            segment,
            position,
            score: 0.0,
            key: "/same/depth",
        }
    }

    #[test]
    fn path_vectors_prefer_stronger_segment_then_position_counts() {
        let typo_costs = TypoCostVector::default();
        let insertions = InsertionVector::default();
        let basename_and_grandparent = path_key(&[1, 0, 1], &[0, 2, 0], &typo_costs, &insertions);
        let two_parent_matches = path_key(&[0, 2], &[2, 0, 0], &typo_costs, &insertions);
        assert!(
            compare_path_typo_sort_keys(&basename_and_grandparent, &two_parent_matches).is_lt()
        );

        let whole_or_prefix = path_key(&[2], &[2, 0, 0], &typo_costs, &insertions);
        let suffixes = path_key(&[2], &[0, 2, 0], &typo_costs, &insertions);
        assert!(compare_path_typo_sort_keys(&whole_or_prefix, &suffixes).is_lt());
    }
}
