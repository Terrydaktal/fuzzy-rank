//! Learned reranking for prioritized text fields.
//!
//! This module deliberately contains no persistence and no candidate
//! identity.  The caller owns those concerns; this crate only turns an
//! already-ranked field match into features and applies pairwise updates.

use crate::core::{CoverageRank, DensityRank, FieldOccurrenceRank};
use crate::learning::{
    DEFAULT_LEARNING_RATE as LEARNING_RATE, DEFAULT_MARGIN as MARGIN, MAX_WEIGHT as WEIGHT_LIMIT,
    learn_pair as learn_pair_weights,
};

/// Version of the persisted field-feature schema.
///
/// Callers that persist model weights should store and validate this value
/// before loading them.  Feature indices are part of the model contract.
pub const FIELD_RANK_FEATURE_SCHEMA_VERSION: u32 = 2;

/// Number of scalar features exposed by [`FieldRankFeatures`].
pub const FIELD_RANK_FEATURE_COUNT: usize = 34;

/// Stable names for [`FieldRankFeatures`].  Values are direction-normalized:
/// larger values always mean stronger evidence for the candidate.
pub const FIELD_RANK_FEATURE_NAMES: [&str; FIELD_RANK_FEATURE_COUNT] = [
    "search_rank_quality",
    "exact_phrase_present",
    "exact_subphrase_presence",
    "exact_longest_subphrase",
    "exact_coverage_whole",
    "exact_coverage_prefix",
    "exact_coverage_suffix",
    "exact_coverage_substring",
    "exact_coverage_quality",
    "exact_coverage_count",
    "exact_field_quality",
    "fuzzy_whole_phrase",
    "fuzzy_subphrase_presence",
    "fuzzy_longest_subphrase",
    "fuzzy_coverage_whole",
    "fuzzy_coverage_prefix",
    "fuzzy_coverage_suffix",
    "fuzzy_coverage_substring",
    "fuzzy_coverage_fragmented",
    "fuzzy_coverage_quality",
    "fuzzy_coverage_count",
    "fuzzy_field_quality",
    "typo_quality",
    "insertion_quality",
    "order_quality",
    "gap_quality",
    "exact_phrase_occurrence_priority",
    "exact_phrase_occurrence_count",
    "exact_phrase_occurrence_total",
    "exact_token_occurrence_priority",
    "exact_token_occurrence_count",
    "exact_token_occurrence_total",
    "density_quality",
    "caller_score",
];

/// Indices into [`FieldRankFeatures`].  The named indices keep adaptor code
/// readable and make persisted model weights self-describing alongside
/// [`FIELD_RANK_FEATURE_NAMES`].
pub mod feature {
    pub const SEARCH_RANK_QUALITY: usize = 0;
    pub const EXACT_PHRASE_PRESENT: usize = 1;
    pub const EXACT_SUBPHRASE_PRESENCE: usize = 2;
    pub const EXACT_LONGEST_SUBPHRASE: usize = 3;
    pub const EXACT_COVERAGE_WHOLE: usize = 4;
    pub const EXACT_COVERAGE_PREFIX: usize = 5;
    pub const EXACT_COVERAGE_SUFFIX: usize = 6;
    pub const EXACT_COVERAGE_SUBSTRING: usize = 7;
    pub const EXACT_COVERAGE_QUALITY: usize = 8;
    pub const EXACT_COVERAGE_COUNT: usize = 9;
    pub const EXACT_FIELD_QUALITY: usize = 10;
    pub const FUZZY_WHOLE_PHRASE: usize = 11;
    pub const FUZZY_SUBPHRASE_PRESENCE: usize = 12;
    pub const FUZZY_LONGEST_SUBPHRASE: usize = 13;
    pub const FUZZY_COVERAGE_WHOLE: usize = 14;
    pub const FUZZY_COVERAGE_PREFIX: usize = 15;
    pub const FUZZY_COVERAGE_SUFFIX: usize = 16;
    pub const FUZZY_COVERAGE_SUBSTRING: usize = 17;
    pub const FUZZY_COVERAGE_FRAGMENTED: usize = 18;
    pub const FUZZY_COVERAGE_QUALITY: usize = 19;
    pub const FUZZY_COVERAGE_COUNT: usize = 20;
    pub const FUZZY_FIELD_QUALITY: usize = 21;
    pub const TYPO_QUALITY: usize = 22;
    pub const INSERTION_QUALITY: usize = 23;
    pub const ORDER_QUALITY: usize = 24;
    pub const GAP_QUALITY: usize = 25;
    pub const EXACT_PHRASE_OCCURRENCE_PRIORITY: usize = 26;
    pub const EXACT_PHRASE_OCCURRENCE_COUNT: usize = 27;
    pub const EXACT_PHRASE_OCCURRENCE_TOTAL: usize = 28;
    pub const EXACT_TOKEN_OCCURRENCE_PRIORITY: usize = 29;
    pub const EXACT_TOKEN_OCCURRENCE_COUNT: usize = 30;
    pub const EXACT_TOKEN_OCCURRENCE_TOTAL: usize = 31;
    pub const DENSITY_QUALITY: usize = 32;
    pub const CALLER_SCORE: usize = 33;
}

/// A fixed-size representation of a message or metadata match.
///
/// This is intentionally independent from the deterministic comparator.  It
/// lets callers train or load a domain-specific model without making fuzzy
/// matching depend on a runtime, database, or serialization format.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FieldRankFeatures {
    values: [f64; FIELD_RANK_FEATURE_COUNT],
}

impl FieldRankFeatures {
    pub const fn new(values: [f64; FIELD_RANK_FEATURE_COUNT]) -> Self {
        Self { values }
    }

    pub const fn zero() -> Self {
        Self::new([0.0; FIELD_RANK_FEATURE_COUNT])
    }

    pub fn values(&self) -> &[f64; FIELD_RANK_FEATURE_COUNT] {
        &self.values
    }

    pub fn get(&self, index: usize) -> Option<f64> {
        self.values.get(index).copied()
    }

    pub(crate) fn set(&mut self, index: usize, value: f64) {
        self.values[index] = finite(value);
    }

    pub fn score(&self, model: &FieldRankModel) -> f64 {
        finite(
            finite(model.bias)
                + self
                    .values
                    .iter()
                    .zip(model.weights)
                    .map(|(feature, weight)| finite(*feature) * finite(weight))
                    .sum::<f64>(),
        )
    }
}

impl Default for FieldRankFeatures {
    fn default() -> Self {
        Self::zero()
    }
}

/// Small native linear model for field-result reranking.
///
/// The model is trained pairwise: a preferred result is compared with a
/// result the caller wants below it.  Persistence belongs to the application
/// because query/result identifiers and storage formats are domain-specific.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FieldRankModel {
    pub bias: f64,
    pub weights: [f64; FIELD_RANK_FEATURE_COUNT],
}

impl FieldRankModel {
    pub const SCHEMA_VERSION: u32 = FIELD_RANK_FEATURE_SCHEMA_VERSION;
    pub const DEFAULT_LEARNING_RATE: f64 = LEARNING_RATE;
    pub const DEFAULT_MARGIN: f64 = MARGIN;
    pub const MAX_WEIGHT: f64 = WEIGHT_LIMIT;

    pub const fn new(bias: f64, weights: [f64; FIELD_RANK_FEATURE_COUNT]) -> Self {
        Self { bias, weights }
    }

    pub const fn schema_version() -> u32 {
        Self::SCHEMA_VERSION
    }

    /// Construct a model from caller-owned persisted parts after validating
    /// the feature schema and numeric values.
    pub fn from_versioned_parts(schema_version: u32, bias: f64, weights: &[f64]) -> Option<Self> {
        if schema_version != Self::SCHEMA_VERSION
            || weights.len() != FIELD_RANK_FEATURE_COUNT
            || !bias.is_finite()
            || weights.iter().any(|weight| !weight.is_finite())
        {
            return None;
        }
        let mut model_weights = [0.0; FIELD_RANK_FEATURE_COUNT];
        model_weights.copy_from_slice(weights);
        Some(Self::new(bias, model_weights))
    }

    pub fn score(&self, features: &FieldRankFeatures) -> f64 {
        features.score(self)
    }

    pub fn is_active(&self) -> bool {
        // A constant bias cancels when two candidates are compared.
        self.weights.iter().any(|weight| finite(*weight) != 0.0)
    }

    /// Train one preferred-versus-other pair.
    ///
    /// Returns `true` when the model was updated.  Correctly ordered pairs
    /// are ignored once the configured margin is satisfied.
    pub fn learn_pair(&mut self, preferred: &FieldRankFeatures, other: &FieldRankFeatures) -> bool {
        self.learn_pair_with(
            preferred,
            other,
            Self::DEFAULT_LEARNING_RATE,
            Self::DEFAULT_MARGIN,
        )
    }

    /// Train a preferred result against a group of alternatives.  This is
    /// the operation callers use after a user selects one result from a
    /// displayed candidate set.
    pub fn learn_preference(
        &mut self,
        preferred: &FieldRankFeatures,
        alternatives: &[FieldRankFeatures],
    ) -> usize {
        alternatives
            .iter()
            .filter(|alternative| self.learn_pair(preferred, alternative))
            .count()
    }

    /// Configurable form of [`Self::learn_pair`], useful for offline replay
    /// or callers that want slower, more conservative updates.
    pub fn learn_pair_with(
        &mut self,
        preferred: &FieldRankFeatures,
        other: &FieldRankFeatures,
        learning_rate: f64,
        margin: f64,
    ) -> bool {
        let learning_rate = finite(learning_rate).max(0.0);
        let margin = finite(margin).max(0.0);
        learn_pair_weights(
            &mut self.weights,
            &preferred.values,
            &other.values,
            learning_rate,
            margin,
        )
    }
}

impl Default for FieldRankModel {
    fn default() -> Self {
        Self::new(0.0, [0.0; FIELD_RANK_FEATURE_COUNT])
    }
}

pub(crate) fn log_count(count: usize) -> f64 {
    (count as f64).ln_1p()
}

pub(crate) fn quality_from_priority(priority: u8) -> f64 {
    if priority == u8::MAX {
        -256.0
    } else {
        -(priority as f64)
    }
}

pub(crate) fn longest_subphrase(vector: &[usize], query_len: usize) -> f64 {
    vector
        .iter()
        .position(|count| *count != 0)
        .map_or(0.0, |index| query_len.saturating_sub(index + 1) as f64)
}

pub(crate) fn set_coverage_features(
    features: &mut FieldRankFeatures,
    start: usize,
    rank: &CoverageRank,
) {
    for (offset, count) in rank.class_counts.iter().copied().enumerate() {
        features.set(start + offset, count as f64);
    }
    features.set(
        start + rank.class_counts.len(),
        rank.weighted_quality as f64,
    );
    features.set(
        start + rank.class_counts.len() + 1,
        rank.distinct_coverage as f64,
    );
}

pub(crate) fn occurrence_features(rank: &FieldOccurrenceRank) -> (f64, f64, f64) {
    let best = rank
        .entries()
        .iter()
        .find(|(_, count)| *count != 0)
        .copied();
    let priority = best.map_or(quality_from_priority(u8::MAX), |(priority, _)| {
        quality_from_priority(priority)
    });
    let best_count = best.map_or(0, |(_, count)| count);
    let total = rank.entries().iter().map(|(_, count)| *count).sum();
    (priority, log_count(best_count), log_count(total))
}

pub(crate) fn density_quality(rank: DensityRank) -> f64 {
    if rank.field_len == 0 || rank.field_len == usize::MAX {
        0.0
    } else {
        rank.matched_tokens as f64 / rank.field_len as f64
    }
}

fn finite(value: f64) -> f64 {
    if value.is_finite() { value } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::{FIELD_RANK_FEATURE_COUNT, FieldRankFeatures, FieldRankModel};

    #[test]
    fn pairwise_learning_prefers_the_selected_features() {
        let mut preferred_values = [0.0; FIELD_RANK_FEATURE_COUNT];
        preferred_values[0] = 1.0;
        let mut other_values = [0.0; FIELD_RANK_FEATURE_COUNT];
        other_values[0] = 0.0;
        let preferred = FieldRankFeatures::new(preferred_values);
        let other = FieldRankFeatures::new(other_values);
        let mut model = FieldRankModel::default();

        assert!(model.learn_pair(&preferred, &other));
        assert!(model.score(&preferred) > model.score(&other));
    }

    #[test]
    fn satisfied_pairs_are_not_updated() {
        let mut preferred_values = [0.0; FIELD_RANK_FEATURE_COUNT];
        preferred_values[0] = 1.0;
        let preferred = FieldRankFeatures::new(preferred_values);
        let other = FieldRankFeatures::zero();
        let mut model = FieldRankModel::default();

        assert!(model.learn_pair_with(&preferred, &other, 1.0, 1.0));
        let weights = model.weights;
        assert!(!model.learn_pair_with(&preferred, &other, 1.0, 1.0));
        assert_eq!(model.weights, weights);
    }

    #[test]
    fn a_bias_only_model_is_inactive_for_pairwise_reranking() {
        let model = FieldRankModel::new(10.0, [0.0; FIELD_RANK_FEATURE_COUNT]);
        assert!(!model.is_active());
        assert_eq!(
            FieldRankModel::schema_version(),
            super::FIELD_RANK_FEATURE_SCHEMA_VERSION
        );
    }

    #[test]
    fn versioned_model_parts_reject_incompatible_storage() {
        assert!(
            FieldRankModel::from_versioned_parts(
                super::FIELD_RANK_FEATURE_SCHEMA_VERSION,
                0.0,
                &[0.0; FIELD_RANK_FEATURE_COUNT]
            )
            .is_some()
        );
        assert!(
            FieldRankModel::from_versioned_parts(
                super::FIELD_RANK_FEATURE_SCHEMA_VERSION + 1,
                0.0,
                &[0.0; FIELD_RANK_FEATURE_COUNT]
            )
            .is_none()
        );
    }
}
