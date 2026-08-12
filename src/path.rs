use std::{cell::OnceCell, ops::Range, path, sync::Arc};

use crate::core::{
    exact_distinct_coverage_by, exact_phrase_present, exact_subphrase_counts_vector,
    increment_simple_position_vector,
};
use crate::text::{split_path_tokens, to_lowercase};

pub mod fuzzy;
pub mod literal;

#[deprecated(note = "use fuzzy_rank::path::literal::bulk instead")]
pub mod bulk;
#[deprecated(note = "use fuzzy_rank::path::literal instead")]
pub mod exact;
#[deprecated(note = "use fuzzy_rank::path::fuzzy instead")]
pub mod typo;

pub type Score = f64;

pub(crate) fn normalize_path(path: &str) -> String {
    let mut normalized = to_lowercase(path);
    let trimmed_len = normalized.trim_end_matches(path::is_separator).len();
    if trimmed_len > 0 {
        normalized.truncate(trimmed_len);
    }
    normalized
}

pub(crate) fn path_depth(path: &str) -> usize {
    let trimmed = path.trim_end_matches(path::is_separator);
    let path = if trimmed.is_empty() { path } else { trimmed };
    path.chars().filter(|ch| path::is_separator(*ch)).count()
}

#[derive(Clone, Debug)]
pub struct PreparedPath<'a> {
    path: &'a str,
    normalized: Arc<str>,
    components: Vec<PreparedComponent>,
}

#[derive(Clone, Debug)]
struct PreparedComponent {
    range: Range<usize>,
    tokens: Vec<Range<usize>>,
}

impl<'a> PreparedPath<'a> {
    pub fn new(path: &'a str) -> Self {
        let normalized: Arc<str> = normalize_path(path).into();
        let components = prepared_components(&normalized);
        Self {
            path,
            normalized,
            components,
        }
    }

    pub fn path(&self) -> &'a str {
        self.path
    }

    pub(crate) fn normalized(&self) -> &str {
        &self.normalized
    }

    pub(crate) fn normalized_arc(&self) -> Arc<str> {
        Arc::clone(&self.normalized)
    }

    pub(crate) fn component_count(&self) -> usize {
        self.components.len()
    }

    pub(crate) fn component(&self, index: usize) -> &str {
        &self.normalized[self.components[index].range.clone()]
    }

    pub(crate) fn tokens(&self, component: usize) -> impl Iterator<Item = &str> + '_ {
        self.components[component]
            .tokens
            .iter()
            .map(|range| &self.normalized[range.clone()])
    }
}

fn prepared_components(normalized: &str) -> Vec<PreparedComponent> {
    let mut components = Vec::new();
    let mut component_start = None;

    for (index, character) in normalized
        .char_indices()
        .chain(std::iter::once((normalized.len(), '/')))
    {
        if path::is_separator(character) {
            if let Some(start) = component_start.take() {
                let range = start..index;
                let tokens = prepared_token_ranges(&normalized[range.clone()], start);
                components.push(PreparedComponent { range, tokens });
            }
        } else if component_start.is_none() {
            component_start = Some(index);
        }
    }

    components
}

fn prepared_token_ranges(component: &str, offset: usize) -> Vec<Range<usize>> {
    let mut ranges = Vec::new();
    let mut token_start = None;

    for (index, character) in component
        .char_indices()
        .chain(std::iter::once((component.len(), ' ')))
    {
        if is_path_token_separator(character) {
            if let Some(start) = token_start.take() {
                ranges.push(offset + start..offset + index);
            }
        } else if token_start.is_none() {
            token_start = Some(index);
        }
    }

    ranges
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PathExactRelevance<'a> {
    path: &'a str,
    query_tokens: Arc<[String]>,
    normalized_path: Arc<str>,
    component_tokens: OnceCell<Vec<Vec<String>>>,
    whole_phrase: OnceCell<bool>,
    run_vector: OnceCell<Vec<usize>>,
    distinct_coverage: OnceCell<usize>,
    ordered_compound_coverage: OnceCell<usize>,
}

impl<'a> PathExactRelevance<'a> {
    pub(crate) fn with_normalized_path(
        path: &'a str,
        normalized_path: Arc<str>,
        query_tokens: Arc<[String]>,
    ) -> Self {
        Self {
            path,
            query_tokens,
            normalized_path,
            component_tokens: OnceCell::new(),
            whole_phrase: OnceCell::new(),
            run_vector: OnceCell::new(),
            distinct_coverage: OnceCell::new(),
            ordered_compound_coverage: OnceCell::new(),
        }
    }

    pub(crate) fn with_ordered_compound_coverage(self, coverage: usize) -> Self {
        self.ordered_compound_coverage
            .set(coverage)
            .expect("new path relevance has no cached compound coverage");
        self
    }

    fn component_tokens(&self) -> &[Vec<String>] {
        self.component_tokens.get_or_init(|| {
            self.normalized_path
                .split(['/', '\\'])
                .filter(|component| !component.is_empty())
                .map(|component| {
                    split_path_tokens(component)
                        .into_iter()
                        .map(str::to_owned)
                        .collect()
                })
                .collect()
        })
    }

    pub(crate) fn has_exact_token_signal(&self) -> bool {
        let direct = self.component_tokens().iter().any(|component| {
            component.iter().any(|candidate| {
                self.query_tokens
                    .iter()
                    .any(|query| candidate.contains(query))
            })
        });
        direct || self.ordered_compound_coverage() > 0
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

    pub(crate) fn ordered_compound_coverage(&self) -> usize {
        *self
            .ordered_compound_coverage
            .get_or_init(|| ordered_compound_coverage(&self.normalized_path, &self.query_tokens))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct OrderedPathEvidence {
    pub(crate) segment: Vec<usize>,
    pub(crate) position: [usize; 3],
    pub(crate) insertions: Vec<usize>,
    pub(crate) ordered_compound_coverage: usize,
}

#[derive(Clone, Copy, Debug)]
struct TermMatch {
    start: usize,
    segment: usize,
    position: usize,
    insertions: usize,
    compound: bool,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SingleTermPathEvidence {
    pub(crate) segment: usize,
    pub(crate) position: usize,
    pub(crate) insertions: usize,
    pub(crate) compound: bool,
    pub(crate) exact_token_prefix: bool,
}

#[derive(Clone, Copy)]
pub(crate) struct PathToken<'a> {
    text: &'a str,
    start: usize,
    end: usize,
    segment: usize,
}

pub(crate) fn ordered_path_evidence(
    normalized_path: &str,
    query_tokens: &[String],
) -> Option<OrderedPathEvidence> {
    let matches = select_direct_matches(normalized_path, query_tokens).or_else(|| {
        let path_tokens = path_tokens(normalized_path);
        select_mixed_matches_with_tokens(normalized_path, &path_tokens, query_tokens)
    })?;
    Some(evidence_from_matches(
        normalized_path,
        query_tokens,
        &matches,
    ))
}

pub(crate) fn ordered_path_evidence_with_tokens(
    normalized_path: &str,
    path_tokens: &[PathToken<'_>],
    query_tokens: &[String],
) -> Option<OrderedPathEvidence> {
    let matches = select_direct_matches(normalized_path, query_tokens)
        .or_else(|| select_mixed_matches_with_tokens(normalized_path, path_tokens, query_tokens))?;
    Some(evidence_from_matches(
        normalized_path,
        query_tokens,
        &matches,
    ))
}

pub(crate) fn single_term_path_evidence(
    normalized_path: &str,
    query: &str,
) -> Option<SingleTermPathEvidence> {
    if query.is_empty() {
        return None;
    }

    let (matched, exact_token_prefix) = if let Some((start, exact_token_prefix)) =
        last_direct_term_match(normalized_path, query)
    {
        (
            direct_term_match(normalized_path, start, query.len()),
            exact_token_prefix,
        )
    } else {
        let tokens = path_tokens(normalized_path);
        let mut options = compound_options(normalized_path, &tokens, query, normalized_path.len())
            .collect::<Vec<_>>();
        options.sort_unstable_by(|left, right| {
            left.compound
                .cmp(&right.compound)
                .then_with(|| left.segment.cmp(&right.segment))
                .then_with(|| left.position.cmp(&right.position))
                .then_with(|| left.insertions.cmp(&right.insertions))
                .then_with(|| right.start.cmp(&left.start))
        });
        (options.into_iter().next()?, false)
    };

    Some(SingleTermPathEvidence {
        segment: matched.segment,
        position: matched.position,
        insertions: matched.insertions,
        compound: matched.compound,
        exact_token_prefix,
    })
}

fn last_direct_term_match(path: &str, query: &str) -> Option<(usize, bool)> {
    let last = path.rfind(query)?;
    if is_path_token_prefix(path, last) {
        return Some((last, true));
    }

    // The reverse lookup supplies the selected match. Only search earlier
    // occurrences when the selected match did not begin at a token boundary.
    let mut search_start = 0;
    while search_start < last {
        let Some(relative_start) = path[search_start..last].find(query) else {
            break;
        };
        let start = search_start + relative_start;
        if is_path_token_prefix(path, start) {
            return Some((last, true));
        }

        let advance = path[start..].chars().next()?.len_utf8();
        search_start = start + advance;
    }

    Some((last, false))
}

fn is_path_token_prefix(path: &str, start: usize) -> bool {
    start == 0
        || path[..start]
            .chars()
            .next_back()
            .is_some_and(is_path_token_separator)
}

pub(crate) fn ordered_compound_coverage(normalized_path: &str, query_tokens: &[String]) -> usize {
    let path_tokens = path_tokens(normalized_path);
    query_tokens
        .iter()
        .filter(|query| {
            !query.is_empty()
                && !path_tokens
                    .iter()
                    .any(|candidate| candidate.text.contains(query.as_str()))
                && compound_options(normalized_path, &path_tokens, query, normalized_path.len())
                    .next()
                    .is_some()
        })
        .count()
}

/// Returns whether a query can be formed from ordered path-token prefixes.
///
/// This is deliberately only a rejection gate. The full evidence calculation
/// remains authoritative, so this function may retain extra candidates but
/// must never reject a valid compound match.
pub(crate) fn compound_term_may_match(normalized_path: &str, query: &str) -> bool {
    if query.is_empty() {
        return false;
    }

    let path_tokens = path_tokens(normalized_path);
    compound_term_may_match_with_tokens(&path_tokens, query)
}

pub(crate) fn compound_term_may_match_with_tokens(
    path_tokens: &[PathToken<'_>],
    query: &str,
) -> bool {
    let mut remaining = query;
    let mut fragments = 0;
    for token in path_tokens {
        if remaining.is_empty() {
            break;
        }
        if let Some(next) = remaining.strip_prefix(token.text) {
            remaining = next;
            fragments += 1;
            continue;
        }
        if token.text.starts_with(remaining) {
            remaining = "";
            fragments += 1;
            break;
        }
    }

    remaining.is_empty() && fragments >= 2
}

fn select_direct_matches(path: &str, query_tokens: &[String]) -> Option<Vec<TermMatch>> {
    let (last, previous) = query_tokens.split_last()?;
    let mut search_end = path.len();
    let mut matches = Vec::with_capacity(query_tokens.len());

    for query in std::iter::once(last).chain(previous.iter().rev()) {
        let start = path[..search_end].rfind(query)?;
        matches.push(direct_term_match(path, start, query.len()));
        search_end = start;
    }

    matches.reverse();
    Some(matches)
}

fn select_mixed_matches_with_tokens(
    path: &str,
    path_tokens: &[PathToken<'_>],
    query_tokens: &[String],
) -> Option<Vec<TermMatch>> {
    if query_tokens.is_empty() {
        return None;
    }
    let mut reversed = Vec::with_capacity(query_tokens.len());
    if !select_mixed_matches_from(
        path,
        path_tokens,
        query_tokens,
        query_tokens.len(),
        path.len(),
        &mut reversed,
    ) {
        return None;
    }
    reversed.reverse();
    Some(reversed)
}

fn select_mixed_matches_from(
    path: &str,
    path_tokens: &[PathToken<'_>],
    query_tokens: &[String],
    query_end: usize,
    search_end: usize,
    selected: &mut Vec<TermMatch>,
) -> bool {
    if query_end == 0 {
        return true;
    }

    let query = &query_tokens[query_end - 1];
    let mut options = direct_options(path, query, search_end);
    options.extend(compound_options(path, path_tokens, query, search_end));
    options.sort_unstable_by(|left, right| {
        left.compound
            .cmp(&right.compound)
            .then_with(|| left.segment.cmp(&right.segment))
            .then_with(|| left.position.cmp(&right.position))
            .then_with(|| left.insertions.cmp(&right.insertions))
            .then_with(|| right.start.cmp(&left.start))
    });

    for option in options {
        selected.push(option);
        if select_mixed_matches_from(
            path,
            path_tokens,
            query_tokens,
            query_end - 1,
            option.start,
            selected,
        ) {
            return true;
        }
        selected.pop();
    }
    false
}

fn direct_options(path: &str, query: &str, search_end: usize) -> Vec<TermMatch> {
    if query.is_empty() {
        return Vec::new();
    }
    let mut options = Vec::new();
    let mut offset = 0;
    while offset < search_end {
        let Some(relative) = path[offset..search_end].find(query) else {
            break;
        };
        let start = offset + relative;
        options.push(direct_term_match(path, start, query.len()));
        offset = start + path[start..].chars().next().map_or(1, char::len_utf8);
    }
    options
}

fn compound_options<'a>(
    path: &'a str,
    tokens: &'a [PathToken<'a>],
    query: &'a str,
    search_end: usize,
) -> impl Iterator<Item = TermMatch> + 'a {
    let mut options = Vec::new();
    for (index, token) in tokens.iter().enumerate() {
        if token.end > search_end {
            break;
        }
        let Some(remaining) = query.strip_prefix(token.text) else {
            continue;
        };
        collect_compound_options(
            path,
            tokens,
            query,
            index + 1,
            remaining,
            token.start,
            token.end,
            token.segment,
            1,
            search_end,
            &mut options,
        );
    }
    options.into_iter()
}

#[allow(clippy::too_many_arguments)]
fn collect_compound_options(
    path: &str,
    tokens: &[PathToken<'_>],
    query: &str,
    next_index: usize,
    remaining: &str,
    start: usize,
    end: usize,
    segment: usize,
    fragments: usize,
    search_end: usize,
    options: &mut Vec<TermMatch>,
) {
    if remaining.is_empty() {
        if fragments >= 2 {
            options.push(TermMatch {
                start,
                segment,
                position: 0,
                insertions: path[start..end]
                    .chars()
                    .count()
                    .saturating_sub(query.chars().count()),
                compound: true,
            });
        }
        return;
    }

    for (index, token) in tokens.iter().enumerate().skip(next_index) {
        if token.end > search_end {
            break;
        }
        if let Some(next_remaining) = remaining.strip_prefix(token.text) {
            collect_compound_options(
                path,
                tokens,
                query,
                index + 1,
                next_remaining,
                start,
                token.end,
                segment.min(token.segment),
                fragments + 1,
                search_end,
                options,
            );
        } else if token.text.starts_with(remaining) {
            collect_compound_options(
                path,
                tokens,
                query,
                index + 1,
                "",
                start,
                token.start + remaining.len(),
                segment.min(token.segment),
                fragments + 1,
                search_end,
                options,
            );
        }
    }
}

fn evidence_from_matches(
    path: &str,
    query_tokens: &[String],
    matches: &[TermMatch],
) -> OrderedPathEvidence {
    let mut segment = Vec::new();
    let mut position = [0; 3];
    let mut insertions = Vec::with_capacity(matches.len());
    for matched in matches {
        increment(&mut segment, matched.segment);
        increment_simple_position_vector(&mut position, matched.position, 1);
        insertions.push(matched.insertions);
    }
    let ordered_compound_coverage = if matches.iter().any(|matched| matched.compound) {
        query_tokens
            .iter()
            .zip(matches)
            .filter(|(query, matched)| matched.compound && !path.contains(query.as_str()))
            .count()
    } else {
        0
    };
    OrderedPathEvidence {
        segment,
        position,
        insertions,
        ordered_compound_coverage,
    }
}

fn direct_term_match(path: &str, start: usize, len: usize) -> TermMatch {
    TermMatch {
        start,
        segment: path_segment_class(path, start),
        position: position_class(path, start, len),
        insertions: position_penalty(path, start, len),
        compound: false,
    }
}

pub(crate) fn path_tokens(path: &str) -> Vec<PathToken<'_>> {
    let mut tokens = Vec::new();
    let mut token_start = None;
    for (index, character) in path.char_indices() {
        if is_path_token_separator(character) {
            if let Some(start) = token_start.take() {
                tokens.push(PathToken {
                    text: &path[start..index],
                    start,
                    end: index,
                    segment: path_segment_class(path, start),
                });
            }
        } else if token_start.is_none() {
            token_start = Some(index);
        }
    }
    if let Some(start) = token_start {
        tokens.push(PathToken {
            text: &path[start..],
            start,
            end: path.len(),
            segment: path_segment_class(path, start),
        });
    }
    tokens
}

fn increment(vector: &mut Vec<usize>, index: usize) {
    if vector.len() <= index {
        vector.resize(index + 1, 0);
    }
    vector[index] += 1;
}

fn path_segment_class(path: &str, start: usize) -> usize {
    path[start..]
        .chars()
        .filter(|character| path::is_separator(*character))
        .count()
}

fn position_class(path: &str, start: usize, len: usize) -> usize {
    let end = start + len;
    let starts_token = start == 0
        || path[..start]
            .chars()
            .next_back()
            .is_some_and(is_path_token_separator);
    let ends_token = end == path.len()
        || path[end..]
            .chars()
            .next()
            .is_some_and(is_path_token_separator);
    if starts_token {
        0
    } else if ends_token {
        1
    } else {
        2
    }
}

fn position_penalty(path: &str, start: usize, len: usize) -> usize {
    let end = start + len;
    let token_start = path[..start]
        .rfind(is_path_token_separator)
        .map_or(0, |position| position + 1);
    let token_end = path[end..]
        .find(is_path_token_separator)
        .map_or(path.len(), |position| end + position);
    start - token_start + (token_end - end)
}

fn is_path_token_separator(character: char) -> bool {
    matches!(character, '/' | '\\' | '-' | '_' | '.') || character.is_whitespace()
}
