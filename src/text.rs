pub fn to_lowercase(s: impl AsRef<str>) -> String {
    let s = s.as_ref();
    if s.is_ascii() {
        s.to_ascii_lowercase()
    } else {
        s.to_lowercase()
    }
}

pub fn token_has_signal(token: &str, candidate: &str) -> bool {
    let chars = token.chars().collect::<Vec<_>>();
    match chars.as_slice() {
        [] => false,
        [single] => candidate.contains(*single),
        _ => {
            let required = if chars.len() <= 3 { 1 } else { 2 };
            let mut overlap = 0;
            for pair in chars.windows(2) {
                if candidate.contains([pair[0], pair[1]]) {
                    overlap += 1;
                    if overlap >= required {
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
    value
        .split(|c: char| matches!(c, '/' | '\\' | '-' | '_' | '.') || c.is_whitespace())
        .filter(|token| !token.is_empty())
        .collect()
}

pub fn normalize_compact_alnum(value: &str) -> String {
    value
        .trim()
        .to_lowercase()
        .chars()
        .filter(|ch| ch.is_alphanumeric())
        .collect()
}
