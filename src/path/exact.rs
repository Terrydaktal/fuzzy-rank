use std::sync::Arc;

use crate::core::{InsertionVector, compare_key_asc, compare_score_desc};
use crate::path::{PathExactRelevance, normalize_path, ordered_path_evidence, path_depth};

pub type Score = f64;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum MatchQuality {
    Contains = 0,
    Suffix = 1,
    Prefix = 2,
    Exact = 3,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactPathMatch<'a> {
    pub path: &'a str,
    pub score: Score,
    pub segment: Vec<usize>,
    pub position: [usize; 3],
    pub insertions: InsertionVector,
    pub(crate) exact_relevance: PathExactRelevance<'a>,
}

pub fn exact_match<'a>(
    path: &'a str,
    keywords: &[String],
    score: Score,
) -> Option<ExactPathMatch<'a>> {
    exact_match_with_query_tokens(path, keywords.to_vec().into(), score)
}

pub(crate) fn exact_match_with_query_tokens<'a>(
    path: &'a str,
    keywords: Arc<[String]>,
    score: Score,
) -> Option<ExactPathMatch<'a>> {
    let normalized_path: Arc<str> = normalize_path(path).into();
    let evidence = ordered_path_evidence(&normalized_path, &keywords)?;
    let exact_relevance = PathExactRelevance::with_normalized_path(path, normalized_path, keywords)
        .with_ordered_compound_coverage(evidence.ordered_compound_coverage);

    Some(ExactPathMatch {
        path,
        score,
        segment: evidence.segment,
        position: evidence.position,
        insertions: InsertionVector::new(evidence.insertions),
        exact_relevance,
    })
}

pub fn compare_exact_path_matches(
    left: &ExactPathMatch<'_>,
    right: &ExactPathMatch<'_>,
    _keywords: &[String],
) -> std::cmp::Ordering {
    right
        .exact_relevance
        .whole_phrase()
        .cmp(&left.exact_relevance.whole_phrase())
        .then_with(|| {
            right
                .exact_relevance
                .run_vector()
                .cmp(left.exact_relevance.run_vector())
        })
        .then_with(|| {
            right
                .exact_relevance
                .distinct_coverage()
                .cmp(&left.exact_relevance.distinct_coverage())
        })
        .then_with(|| {
            right
                .exact_relevance
                .ordered_compound_coverage()
                .cmp(&left.exact_relevance.ordered_compound_coverage())
        })
        .then_with(|| right.segment.cmp(&left.segment))
        .then_with(|| right.position.cmp(&left.position))
        .then_with(|| compare_score_desc(left.score, right.score))
        .then_with(|| left.insertions.cmp(&right.insertions))
        .then_with(|| path_depth(left.path).cmp(&path_depth(right.path)))
        .then_with(|| compare_key_asc(left.path, right.path))
}

pub fn match_penalty(path: &str, keywords: &[String]) -> Option<usize> {
    Some(match_penalties(path, keywords)?.into_iter().sum())
}

pub fn match_penalties(path: &str, keywords: &[String]) -> Option<Vec<usize>> {
    let path = normalize_path(path);
    Some(ordered_path_evidence(&path, keywords)?.insertions)
}

pub fn match_path_segment(path: &str, keywords: &[String]) -> Option<Vec<usize>> {
    let path = normalize_path(path);
    Some(ordered_path_evidence(&path, keywords)?.segment)
}

pub fn match_path_position(path: &str, keywords: &[String]) -> Option<[usize; 3]> {
    let path = normalize_path(path);
    Some(ordered_path_evidence(&path, keywords)?.position)
}

pub fn match_qualities(path: &str, keywords: &[String]) -> Option<Vec<MatchQuality>> {
    let (keywords_last, keywords) = match keywords.split_last() {
        Some(split) => split,
        None => return Some(Vec::new()),
    };

    let path = normalize_path(path);
    let mut path = path.as_str();
    let mut qualities = Vec::with_capacity(keywords.len() + 1);

    let idx = path.rfind(keywords_last)?;
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
            Some(vec![2])
        );
        assert_eq!(
            match_path_segment(
                "/home/lewis/Dev/applicationlauncher/target/release",
                &keywords
            ),
            Some(vec![0, 0, 2])
        );
    }

    #[test]
    fn exact_terms_can_match_entirely_in_ancestor_components() {
        let keywords = ["gemini", "packages"]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        let path = "/home/lewis/repos/gemini-cli/packages/core/dist/docs/cli/using-agent-skills.md";
        let matched = exact_match(path, &keywords, 0.0).unwrap();

        assert_eq!(matched.segment, vec![0, 0, 0, 0, 0, 1, 1]);
        assert_eq!(matched.position, [2, 0, 0]);
        assert_eq!(matched.insertions.aligned(), &[0, 0]);
    }

    #[test]
    fn compact_query_token_matches_ordered_path_tokens() {
        let keywords = vec!["geminipackages".to_string()];
        let path = "/home/lewis/repos/gemini-cli/packages/core/dist/docs/cli/using-agent-skills.md";
        let matched = exact_match(path, &keywords, 0.0).unwrap();

        assert_eq!(matched.exact_relevance.distinct_coverage(), 0);
        assert_eq!(matched.exact_relevance.ordered_compound_coverage(), 1);
        assert_eq!(matched.segment, vec![0, 0, 0, 0, 0, 1]);
        assert_eq!(matched.position, [1, 0, 0]);
        assert_eq!(matched.insertions.aligned(), &[5]);
    }

    #[test]
    fn compact_query_may_end_at_a_path_token_prefix() {
        let keywords = vec!["geminipack".to_string()];
        let matched = exact_match("/home/gemini-cli/packages", &keywords, 0.0).unwrap();

        assert_eq!(matched.exact_relevance.ordered_compound_coverage(), 1);
        assert_eq!(matched.insertions.aligned(), &[5]);
    }

    #[test]
    fn ordered_compound_matching_preserves_path_and_query_order() {
        let keywords = vec!["repos".to_string(), "geminipackages".to_string()];
        let matched = exact_match("/home/repos/gemini-cli/packages", &keywords, 0.0).unwrap();

        assert_eq!(matched.exact_relevance.distinct_coverage(), 1);
        assert_eq!(matched.exact_relevance.ordered_compound_coverage(), 1);
        assert!(
            exact_match("/home/repos/packages/gemini-cli", &keywords, 0.0).is_none(),
            "reversed path tokens must not satisfy ordered compound matching"
        );
    }

    #[test]
    fn direct_exact_coverage_precedes_ordered_compound_coverage() {
        let keywords = vec!["geminipackages".to_string()];
        let direct = exact_match("/home/geminipackages", &keywords, 0.0).unwrap();
        let compound = exact_match("/home/gemini-cli/packages", &keywords, 100.0).unwrap();

        assert_eq!(
            compare_exact_path_matches(&direct, &compound, &keywords),
            std::cmp::Ordering::Less
        );
    }

    #[test]
    fn exact_relevance_precedes_path_specific_ties() {
        let keywords = ["app", "launch", "target"]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        let phrase = exact_match("/home/app-launch-target", &keywords, 1.0).unwrap();
        let compact = exact_match("/home/applaunchtarget", &keywords, 100.0).unwrap();

        assert!(phrase.exact_relevance.whole_phrase());
        assert_eq!(phrase.exact_relevance.run_vector(), &[2]);
        assert_eq!(phrase.exact_relevance.distinct_coverage(), 3);
        assert_eq!(compact.exact_relevance.distinct_coverage(), 3);
        assert_eq!(
            compare_exact_path_matches(&phrase, &compact, &keywords),
            std::cmp::Ordering::Less
        );
    }

    #[test]
    fn compare_exact_path_matches_prefers_exact_relevance_then_path_specific_ties() {
        let keywords = ["tasks", "onfig"];
        let keywords = keywords.into_iter().map(str::to_string).collect::<Vec<_>>();
        let lower_insertions = exact_match("/home/lewis/tasks/config", &keywords, 10.0).unwrap();
        let higher_insertions =
            exact_match("/home/lewis/tasks/redragonmouseconfig", &keywords, 10.0).unwrap();
        let higher_score = exact_match("/home/lewis/tasks/config", &keywords, 20.0).unwrap();
        let worse_position = ExactPathMatch {
            path: "/home/lewis/tasks/config",
            score: 50.0,
            segment: lower_insertions.segment.clone(),
            position: [0, 0, 2],
            insertions: lower_insertions.insertions.clone(),
            exact_relevance: lower_insertions.exact_relevance.clone(),
        };
        let worse_segment = ExactPathMatch {
            path: "/home/lewis/tasks/config",
            score: 100.0,
            segment: vec![0, 2],
            position: lower_insertions.position,
            insertions: lower_insertions.insertions.clone(),
            exact_relevance: lower_insertions.exact_relevance.clone(),
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
