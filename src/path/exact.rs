use std::path;

use crate::core::{compare_key_asc, compare_score_desc};
use crate::text::to_lowercase;

pub type Score = f64;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum MatchQuality {
    Contains = 0,
    Suffix = 1,
    Prefix = 2,
    Exact = 3,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExactPathMatch<'a> {
    pub path: &'a str,
    pub score: Score,
    pub segment: usize,
    pub position: usize,
}

pub fn exact_match<'a>(
    path: &'a str,
    keywords: &[String],
    score: Score,
) -> Option<ExactPathMatch<'a>> {
    Some(ExactPathMatch {
        path,
        score,
        segment: match_path_segment(path, keywords)?,
        position: match_path_position(path, keywords)?,
    })
}

pub fn compare_exact_path_matches(
    left: &ExactPathMatch<'_>,
    right: &ExactPathMatch<'_>,
    keywords: &[String],
) -> std::cmp::Ordering {
    left.segment
        .cmp(&right.segment)
        .then_with(|| left.position.cmp(&right.position))
        .then_with(|| compare_score_desc(left.score, right.score))
        .then_with(|| match_penalty(left.path, keywords).cmp(&match_penalty(right.path, keywords)))
        .then_with(|| path_depth(left.path).cmp(&path_depth(right.path)))
        .then_with(|| compare_key_asc(left.path, right.path))
}

pub fn match_penalty(path: &str, keywords: &[String]) -> Option<usize> {
    let (keywords_last, keywords) = match keywords.split_last() {
        Some(split) => split,
        None => return Some(0),
    };

    let path = to_lowercase(path);
    let mut path = path.as_str();
    let mut penalty = 0;

    let idx = path.rfind(keywords_last)?;
    if path[idx + keywords_last.len()..].contains(path::is_separator) {
        return None;
    }
    penalty += match_position_penalty(path, idx, keywords_last.len());
    path = &path[..idx];

    for keyword in keywords.iter().rev() {
        let idx = path.rfind(keyword)?;
        penalty += match_position_penalty(path, idx, keyword.len());
        path = &path[..idx];
    }

    Some(penalty)
}

pub fn match_path_segment(path: &str, keywords: &[String]) -> Option<usize> {
    let (keywords_last, keywords) = match keywords.split_last() {
        Some(split) => split,
        None => return Some(0),
    };

    let full_path = to_lowercase(path);
    let mut search = full_path.as_str();
    let mut segment = 0;

    let idx = search.rfind(keywords_last)?;
    if search[idx + keywords_last.len()..].contains(path::is_separator) {
        return None;
    }
    segment += path_segment_class(&full_path, idx);
    search = &search[..idx];

    for keyword in keywords.iter().rev() {
        let idx = search.rfind(keyword)?;
        segment += path_segment_class(&full_path, idx);
        search = &search[..idx];
    }

    Some(segment / (keywords.len() + 1))
}

pub fn match_path_position(path: &str, keywords: &[String]) -> Option<usize> {
    let (keywords_last, keywords) = match keywords.split_last() {
        Some(split) => split,
        None => return Some(0),
    };

    let full_path = to_lowercase(path);
    let mut search = full_path.as_str();
    let mut position = 0;

    let idx = search.rfind(keywords_last)?;
    if search[idx + keywords_last.len()..].contains(path::is_separator) {
        return None;
    }
    position += position_rank(&full_path, idx, keywords_last.len());
    search = &search[..idx];

    for keyword in keywords.iter().rev() {
        let idx = search.rfind(keyword)?;
        position += position_rank(&full_path, idx, keyword.len());
        search = &search[..idx];
    }

    Some(position)
}

pub fn match_qualities(path: &str, keywords: &[String]) -> Option<Vec<MatchQuality>> {
    let (keywords_last, keywords) = match keywords.split_last() {
        Some(split) => split,
        None => return Some(Vec::new()),
    };

    let path = to_lowercase(path);
    let mut path = path.as_str();
    let mut qualities = Vec::with_capacity(keywords.len() + 1);

    let idx = path.rfind(keywords_last)?;
    if path[idx + keywords_last.len()..].contains(path::is_separator) {
        return None;
    }
    qualities.push(match_quality(path, idx, keywords_last.len()));
    path = &path[..idx];

    for keyword in keywords.iter().rev() {
        let idx = path.rfind(keyword)?;
        qualities.push(match_quality(path, idx, keyword.len()));
        path = &path[..idx];
    }

    Some(qualities)
}

fn match_quality(path: &str, idx: usize, keyword_len: usize) -> MatchQuality {
    let end = idx + keyword_len;
    let starts_token = idx == 0
        || path[..idx]
            .chars()
            .next_back()
            .is_some_and(is_path_token_separator);
    let ends_token = end == path.len()
        || path[end..]
            .chars()
            .next()
            .is_some_and(is_path_token_separator);

    if starts_token && ends_token {
        MatchQuality::Exact
    } else if starts_token {
        MatchQuality::Prefix
    } else if ends_token {
        MatchQuality::Suffix
    } else {
        MatchQuality::Contains
    }
}

fn path_depth(path: &str) -> usize {
    path.chars().filter(|ch| path::is_separator(*ch)).count()
}

fn match_position_penalty(path: &str, idx: usize, keyword_len: usize) -> usize {
    let end = idx + keyword_len;
    let token_start = path[..idx]
        .rfind(is_path_token_separator)
        .map_or(0, |position| position + 1);
    let token_end = path[end..]
        .find(is_path_token_separator)
        .map_or(path.len(), |position| end + position);
    idx - token_start + (token_end - end)
}

fn path_segment_class(path: &str, idx: usize) -> usize {
    path[idx..]
        .chars()
        .filter(|ch| path::is_separator(*ch))
        .count()
}

fn position_rank(path: &str, idx: usize, keyword_len: usize) -> usize {
    match match_quality(path, idx, keyword_len) {
        MatchQuality::Exact | MatchQuality::Prefix => 0,
        MatchQuality::Suffix => 1,
        MatchQuality::Contains => 2,
    }
}

fn is_path_token_separator(c: char) -> bool {
    matches!(c, '/' | '\\' | '-' | '_' | '.') || c.is_whitespace()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn match_penalty_sums_token_positions() {
        let keywords = ["asks", "onfig"];
        let keywords = keywords.into_iter().map(str::to_string).collect::<Vec<_>>();
        assert_eq!(
            match_penalty("/home/lewis/tasks/config", &keywords),
            Some(2)
        );
    }

    #[test]
    fn match_path_segment_prefers_basename_over_parent_components() {
        let keywords = ["ap", "laun"];
        let keywords = keywords.into_iter().map(str::to_string).collect::<Vec<_>>();
        assert_eq!(
            match_path_segment("/home/lewis/Dev/applicationlauncher", &keywords),
            Some(0)
        );
        assert_eq!(
            match_path_segment(
                "/home/lewis/Dev/applicationlauncher/target/release",
                &keywords
            ),
            None
        );
    }

    #[test]
    fn compare_exact_path_matches_prefers_segment_then_position_then_score_then_insertions_then_path_depth() {
        let keywords = ["tasks", "onfig"];
        let keywords = keywords.into_iter().map(str::to_string).collect::<Vec<_>>();
        let lower_insertions = exact_match("/home/lewis/tasks/config", &keywords, 10.0).unwrap();
        let higher_insertions =
            exact_match("/home/lewis/tasks/redragonmouseconfig", &keywords, 10.0).unwrap();
        let higher_score = exact_match("/home/lewis/tasks/config", &keywords, 20.0).unwrap();
        let worse_position = ExactPathMatch {
            path: "/home/lewis/tasks/config",
            score: 50.0,
            segment: lower_insertions.segment,
            position: lower_insertions.position + 1,
        };
        let worse_segment = ExactPathMatch {
            path: "/home/lewis/tasks/config",
            score: 100.0,
            segment: lower_insertions.segment + 1,
            position: 0,
        };

        assert_eq!(
            compare_exact_path_matches(&higher_score, &lower_insertions, &keywords),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            compare_exact_path_matches(&lower_insertions, &higher_insertions, &keywords),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            compare_exact_path_matches(&lower_insertions, &worse_position, &keywords),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            compare_exact_path_matches(&lower_insertions, &worse_segment, &keywords),
            std::cmp::Ordering::Less
        );
    }
}
