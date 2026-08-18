pub mod command_history;
pub mod core;
pub mod debug;
pub mod fields;
pub mod learning;
pub mod path;

#[deprecated(note = "use fuzzy_rank::fields::literal instead")]
pub mod message;
#[deprecated(note = "use fuzzy_rank::fields::fuzzy instead")]
pub mod metadata;
pub mod ranking;
pub mod text;
pub mod token_match;
