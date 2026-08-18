//! Field-oriented ranking adaptors.

pub mod fuzzy;
pub mod literal;
pub mod model;

pub use model::{
    FIELD_RANK_FEATURE_COUNT, FIELD_RANK_FEATURE_NAMES, FIELD_RANK_FEATURE_SCHEMA_VERSION,
    FieldRankFeatures, FieldRankModel,
};
