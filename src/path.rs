use std::cell::OnceCell;
use std::sync::Arc;

use crate::core::{
    exact_distinct_coverage_by, exact_phrase_present, exact_subphrase_counts_vector,
};
use crate::text::{split_path_tokens, to_lowercase};

pub mod exact;
pub mod typo;

pub type Score = f64;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PathExactRelevance<'a> {
    path: &'a str,
    query_tokens: Arc<[String]>,
    component_tokens: OnceCell<Vec<Vec<String>>>,
    whole_phrase: OnceCell<bool>,
    run_vector: OnceCell<Vec<usize>>,
    distinct_coverage: OnceCell<usize>,
}

impl<'a> PathExactRelevance<'a> {
    pub(crate) fn new(path: &'a str, query_tokens: Arc<[String]>) -> Self {
        Self {
            path,
            query_tokens,
            component_tokens: OnceCell::new(),
            whole_phrase: OnceCell::new(),
            run_vector: OnceCell::new(),
            distinct_coverage: OnceCell::new(),
        }
    }

    fn component_tokens(&self) -> &[Vec<String>] {
        self.component_tokens.get_or_init(|| {
            self.path
                .split(['/', '\\'])
                .filter(|component| !component.is_empty())
                .map(|component| {
                    split_path_tokens(component)
                        .into_iter()
                        .map(to_lowercase)
                        .collect()
                })
                .collect()
        })
    }

    pub(crate) fn has_exact_token_signal(&self) -> bool {
        self.component_tokens().iter().any(|component| {
            component.iter().any(|candidate| {
                self.query_tokens
                    .iter()
                    .any(|query| candidate.contains(query))
            })
        })
    }

    pub(crate) fn whole_phrase(&self) -> bool {
        *self.whole_phrase.get_or_init(|| {
            exact_phrase_present(
                &self.query_tokens,
                self.component_tokens().iter().map(Vec::as_slice),
            )
        })
    }

    pub(crate) fn run_vector(&self) -> &[usize] {
        self.run_vector.get_or_init(|| {
            exact_subphrase_counts_vector(
                &self.query_tokens,
                self.component_tokens().iter().map(Vec::as_slice),
            )
        })
    }

    pub(crate) fn distinct_coverage(&self) -> usize {
        *self.distinct_coverage.get_or_init(|| {
            exact_distinct_coverage_by(
                &self.query_tokens,
                self.component_tokens().iter().map(Vec::as_slice),
                |query, candidate| candidate.contains(query),
            )
        })
    }
}
