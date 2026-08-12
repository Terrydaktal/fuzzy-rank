pub fn to_lowercase(s: impl AsRef<str>) -> String {
    let s = s.as_ref();
    if s.is_ascii() {
        s.to_ascii_lowercase()
    } else {
        s.to_lowercase()
    }
}

pub fn token_has_signal(token: &str, candidate: &str) -> bool {
    if token.is_ascii() && candidate.is_ascii() {
        let token = token.as_bytes();
        let candidate = candidate.as_bytes();
        return match token {
            [] => false,
            [single] => candidate.contains(single),
            token if token.len() <= 4 => {
                token
                    .iter()
                    .filter(|ch| candidate.contains(ch))
                    .take(2)
                    .count()
                    >= 2
            }
            token => {
                token
                    .windows(2)
                    .filter(|pair| candidate.windows(2).any(|window| window == *pair))
                    .take(2)
                    .count()
                    >= 2
            }
        };
    }

    let chars = token.chars().collect::<Vec<_>>();
    match chars.as_slice() {
        [] => false,
        [single] => candidate.contains(*single),
        _ if chars.len() <= 4 => {
            let required = 2;
            let mut matched_chars = 0;
            for ch in &chars {
                if candidate.contains(*ch) {
                    matched_chars += 1;
                    if matched_chars >= required {
                        return true;
                    }
                }
            }
            false
        }
        _ => {
            let mut overlap = 0;
            let candidate_chars = candidate.chars().collect::<Vec<_>>();
            for pair in chars.windows(2) {
                if candidate_chars.windows(2).any(|w| w == pair) {
                    overlap += 1;
                    if overlap >= 2 {
                        return true;
                    }
                }
            }
            false
        }
    }
}

pub fn has_query_signal<'a>(
    query_tokens: impl IntoIterator<Item = &'a str>,
    candidate: &str,
) -> bool {
    query_tokens
        .into_iter()
        .any(|token| token_has_signal(token, candidate))
}

pub fn split_whitespace_tokens(value: &str) -> Vec<&str> {
    value
        .split_whitespace()
        .filter(|token| !token.is_empty())
        .collect()
}

pub fn split_search_tokens(value: &str) -> Vec<&str> {
    value
        .split(|c: char| !c.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .collect()
}

pub fn split_path_tokens(value: &str) -> Vec<&str> {
    iter_path_tokens(value).collect()
}

pub fn iter_path_tokens(value: &str) -> impl Iterator<Item = &str> {
    value
        .split(|c: char| matches!(c, '/' | '\\' | '-' | '_' | '.') || c.is_whitespace())
        .filter(|token| !token.is_empty())
}

pub fn normalize_compact_alnum(value: &str) -> String {
    value
        .trim()
        .to_lowercase()
        .chars()
        .filter(|ch| ch.is_alphanumeric())
        .collect()
}
