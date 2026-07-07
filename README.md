# Fuzzy Rank

`fuzzy-rank` is a Rust library for typo-tolerant matching and ranking across several kinds of searchable data.

It is structured as:

- one shared edit-distance engine
- one shared ranking/comparison layer
- one shared text/query-preparation layer
- one shared token-matching layer
- a small set of domain adapters on top

The goal is to keep the hard parts implemented once, while still allowing different match semantics for:

- filesystem paths
- shell command history
- application and window metadata
- chat and message text

## Project Structure

```text
fuzzy-rank/
├── Cargo.toml
├── README.md
└── src
    ├── lib.rs
    ├── core.rs
    ├── ranking.rs
    ├── text.rs
    ├── token_match.rs
    ├── path.rs
    ├── path/
    │   ├── exact.rs
    │   └── typo.rs
    ├── command_history.rs
    ├── message.rs
    └── metadata.rs
```

## Ownership Boundaries

The crate is intentionally split into shared layers and adapter layers.

### Shared layers

These modules are intended to be reused by all adapters:

- `core`
- `ranking`
- `text`
- `token_match`

They own the generic machinery. They do not own path-specific, command-history-specific, or metadata-specific policy.

### Adapter layers

These modules are domain-specific adapters:

- `path`
- `command_history`
- `message`
- `metadata`

They own the rules that differ by data type:

- which parts of a candidate are compared
- how position is interpreted
- which structural penalties matter
- how candidate fields are constructed
- what “stronger” means within that domain

They should stay thin. They are allowed to define policy, but they should consume shared utilities rather than reimplementing edit distance, token scanning, or generic ranking comparisons.

## What Is Shared

The following behavior is shared across adapters.

### `core.rs`

Owns the low-level edit-distance engine:

- bounded Damerau-Levenshtein
- `OperationProfile`
- typo-limit policy via `max_typos`

This module is responsible for:

- computing bounded intended-to-typed Damerau-Levenshtein distance
- tracking weighted typo-likelihood penalties for substitutions, insertions, deletions, and transpositions
- supporting bounded early-exit behavior
- providing a consistent raw typo distance to all adapters

This module does not know anything about:

- paths
- basename vs dirname
- shell-history command-token policy
- app/window field priority
- frecency calculation

### `ranking.rs`

Owns shared ranking and comparison primitives:

- `StructuralRank`
- `DistanceRank`
- `AbbreviationRank`
- `SearchRank`
- `TypoSortKey`
- `PathTypoSortKey`
- comparison helpers such as:
  - `compare_search_results`
  - `compare_typo_sort_keys`
  - `compare_path_typo_sort_keys`
  - ambiguity helpers
  - `ratio_milli`

This module is responsible for:

- centralizing tuple ordering
- keeping rank precedence consistent
- avoiding duplicated sort-key logic across adapters

This module does not decide:

- what a field is
- what a token is
- what counts as basename or parent directory
- which metadata field should be priority 0

### `text.rs`

Owns shared text preparation helpers:

- lowercasing
- token splitting
- compact alphanumeric normalization
- cheap signal/prefilter checks

Current helpers include:

- `to_lowercase`
- `token_has_signal`
- `has_query_signal`
- `split_whitespace_tokens`
- `split_search_tokens`
- `split_path_tokens`
- `normalize_compact_alnum`

This module is responsible for cheap, reusable preprocessing.

It does not decide final ranking policy.

### `token_match.rs`

Owns shared token-level typo matching helpers:

- `PositionedToken`
- `best_token_match`
- `aligned_token_distance`
- `partitioned_token_distance`

This module is responsible for:

- comparing query tokens against candidate tokens
- aligning token sequences
- matching multiple query tokens inside a single candidate token
- returning shared token-level distance, position, and structure outputs

It does not decide domain meaning for the resulting position numbers.

## What Is Not Shared

The following behavior remains adapter-specific on purpose.

### Frecency calculation

This crate accepts caller-owned priority inputs, but does not compute them.

Examples:

- zoxide computes directory frecency itself, then passes `score` into `path`
- a shell-history consumer should compute command history recency itself, then pass `recency` into `command_history`
- an application launcher should compute app/window recency or launch counts itself, then pass `score` into `metadata`

The crate uses those caller-owned priorities in ranking, but it does not define one universal frecency or recency model.

### Candidate extraction

This crate does not fetch or index application records, shell history, or path databases.

Each caller is responsible for:

- loading its own data
- building candidates
- deciding which fields to expose
- deciding the priority value for each candidate

### Domain semantics

The adapters intentionally do not share exact semantics, because the data is different.

Examples:

- path search cares about basename vs parent components
- command-history search cares about typo quality over command tokens, then recency
- message search cares about phrase continuity and token coverage
- metadata search cares about field priority such as title vs class vs description

These are not accidental differences. They are the reason the adapters exist.

## Adapters

## `path.rs`

Purpose:

- split exact-path and typo-path matching
- basename/dirname-aware ranking
- shared top-level namespace for path semantics

This adapter is intended for zoxide-style search.

It owns:

- `path::exact`
- `path::typo`

It uses shared core from:

- `path::exact` uses `text` for path normalization and token-boundary logic
- `path::typo` uses:
  - `core` for edit distance
  - `ranking` for typo sort-key comparison
  - `text` for path-token splitting and query variants
  - `token_match` for token alignment and compound-token matching

What is path-specific here:

- basename outranking parent components
- parent-depth-sensitive `path_position`
- path token separators such as `/`, `\\`, `-`, `_`, `.`, and whitespace

This is the adapter to use for filesystem path search.

### `path::exact`

`path::exact` owns the normal non-typo path helpers:

- `match_qualities`
- `match_path_segment`
- `match_path_position`
- `match_penalty`
- `exact_match`
- `compare_exact_path_matches`

The exact-path comparator now lives in the crate.
Its sort-key order is:

1. `segment`
2. `position`
3. descending `score`
4. `insertions`
5. `path_depth`
6. `path`

The exact-path helpers effectively expose these keys:

1. keyword matchability through `match_qualities`
2. averaged path segment through `match_path_segment`
3. summed within-segment position through `match_path_position`
4. caller-provided `score`
5. contiguous unmatched token slack through `match_penalty`
6. path depth through `exact_match`

`segment` uses the same component-distance meaning as `path::typo`: `0` for basename, `1` for the
parent component, `2` for the next parent, averaged across matched keywords. `position` uses `0`
for start, `1` for end, and `2` for middle, summed across matched keywords. `insertions` is the
contiguous unmatched character count around the matched keyword
span(s) inside the chosen token/component. That makes `application` rank closer to `applications`
than to `applicationlauncher` after position and score have already been compared.

Only the leading exact-path stages are materialized eagerly:

- `segment`
- `position`
- caller `score`

`insertions` and `path_depth` are deferred until an exact-path comparison reaches those tie-break
levels.

### `path::typo`

`path::typo` owns:

- `TypoQuery`
- `PathMatch`
- `MatchScope`
- `query_from_keywords`

Typo path matching uses the shared `PathTypoSortKey` comparator from `ranking.rs`.
The exact typo sort key order is:

1. weighted core typo cost
2. `segment`
3. `position`
4. descending `score`
5. `insertions`
6. `path_depth`
7. `key`

Here path typo ranking splits the operation profile into two stages between raw edit count and path-specific position. It is directional:

- candidate path token/component = intended text
- query text = what the user actually typed

So omitted separators/characters are treated as deletions from the intended candidate, while extra typed characters are treated as insertions into the typed query.

The path-specific comparator uses:

- weighted core typo cost for substitutions, deletions, and transpositions
- then `segment`, ascending, where `0` means basename, `1` the parent component, `2` the next parent, and multi-token matches average those component distances across the matched query tokens
- then `position`, ascending, where `0` means the match starts at the beginning of the chosen segment, `1` means it ends at the segment boundary, and `2` means it sits in the middle; multi-token matches sum those per-token position values
- then descending caller `score`
- then raw `insertions`, ascending, as a separate noise key

For single-token contiguous substring matches with no core typo, `path::typo` takes a direct fast path instead of running full Damerau-Levenshtein. Those matches come through with:

- core typo cost `0`
- raw `insertions` equal to the unmatched candidate-side slack

So among pure insertion-only matches, ranking falls through quickly to `segment`, then within-segment `position`, then to descending `score`, and later to `insertions` if those still tie.

For path typo ranking, `path_depth` is also deferred until a comparison actually reaches that final
structural tie-break.

Shared versus adapter-local:

- shared:
  - the shared weighted `OperationProfile`
  - the tuple ordering in `compare_path_typo_sort_keys`
- adapter-local:
  - splitting core typo cost from raw `insertions`
  - `segment`
  - within-segment `position`
  - `path_depth`
  - normal-path helper logic

## `command_history.rs`

Purpose:

- typo-tolerant shell-history command matching
- command-token typo matching
- recency-aware command-history ranking

This adapter is intended for fish-history-like search.

It owns:

- `CommandHistoryQuery`
- `CommandHistoryCandidate`
- `CommandHistoryMatch`
- `parse_command_history_candidate`

It uses shared core from:

- `core` for edit distance
- `ranking` for command-history typo sort-key comparison
- `text` for lowercase normalization, whitespace tokenization, and signal checks

What is command-history-specific here:

- command history is split into search words, so punctuation and underscores create token boundaries
- query tokens are compared against whole command-history search words
- multi-token queries can match non-contiguous command-history words in order
- missing-space and extra-space interpretations are not part of this adapter
- caller-provided `recency` breaks equal textual ties

This is the adapter to use for shell command history search.

### Command-history sort order

`command_history` uses the shared `TypoSortKey` comparator from `ranking.rs`.
The exact command-history sort key order is:

1. descending `exact_count`
2. `span_len`
3. `distance`
4. `operations`
5. `ratio_milli`
6. descending `recency`
7. `alphabetical`

Where:

- `exact_count` is the number of query tokens matched exactly
- `span_len` is the width of the matched command-token span, so phrase-like adjacent matches beat scattered matches
- `distance` is the summed bounded Damerau-Levenshtein typo distance
- `operations` is the shared intended-to-typed typo-likelihood profile
  It prefers, at equal edit distance:
  nearby-key substitutions, then transpositions, then omitted characters, then inserted characters, with far substitutions and arbitrary insertions penalized most.
- `ratio_milli` is the normalized typo ratio in thousandths
- `recency` is caller-provided and sorts descending
- `alphabetical` is the final deterministic command-string tie-break

Shared versus adapter-local:

- shared:
  - `distance`
  - `operations`
  - `ratio_milli`
  - the tuple ordering in `compare_typo_sort_keys`
- adapter-local:
  - `exact_count`
  - `span_len`
  - command-token candidate construction
  - whole-token typo policy
  - `recency`
  - `alphabetical`

## `message.rs`

Purpose:

- tokenized text matching for chat logs, notes, or message history
- phrase-aware ordering for exact multi-token sequences
- lightweight precomputation for repeated query evaluation

This adapter is intended for message search where metadata-style exact phrase and
coverage ranking are needed on a single text field, without typo handling.

It owns:

- `MessageQuery`
- `MessageCandidate`
- `PreparedMessageCandidate`
- `MessageMatch`
- `contains_query_signal`

It uses shared core from:

- `text` for lowercasing and token splitting

What is message-specific here:

- applying the metadata exact-only relevance stack to one text field
- omitting typo/fuzzy stages entirely
- omitting field-quality stages because message search is single-field

Message matching first prepares lowercased search tokens for the query and each candidate.
Candidates that do not contain any usable query signal are rejected before sorting.

Final message result ordering is:

1. exact whole-phrase occurrence count
2. exact longer phrase-run vector, lexicographic
3. exact distinct query-token coverage
4. whether all query tokens are matched
5. same-order scattered matching
6. closeness / gap distance
7. repeated exact phrase bonus, saturated
8. repeated token bonus, saturated
9. density / low clutter
10. descending `score`
11. `key`

Where:

- exact phrase stages use contiguous query-token windows matched by token-prefix exactness
- exact distinct coverage counts how many different query tokens are matched exactly
- same-order scattered matching prefers candidates where all query tokens can be recovered in query order
- closeness / gap distance prefers smaller gaps between those ordered matches
- repeated exact phrase bonus and repeated token bonus saturate at `3`
- density prefers candidates where matched concepts occupy a larger fraction of a shorter message

This is the adapter to use for:

- chat messages
- notes or snippets
- searchable conversation history
- other single-text records where phrase continuity matters

## `metadata.rs`

Purpose:

- multi-field matching for records like applications and windows
- field-priority-aware structural, abbreviation, fuzzy, and typo ranking

This adapter is intended for application launchers and window search.

It owns:

- `MetadataQuery`
- `MetadataCandidate`
- `SearchField`
- `MetadataMatch`
- `dedup_push_search_field`

It uses shared core from:

- `ranking` for structural, fuzzy, abbreviation, and typo rank comparison
- `text` for general search tokenization and normalized key cleanup

It currently keeps more logic locally than `path` and `command_history`, because metadata search has richer field-aware policy.

What is metadata-specific here:

- field priorities such as title/name/class/description
- choosing between structural, abbreviation, fuzzy, and typo modes
- ranking across multiple independent fields rather than one path or argv sequence

This is the adapter to use for:

- application names
- desktop-file metadata
- window titles
- WM class / instance values
- other structured multi-field records

### Metadata sort order

`metadata` now uses a query-aware relevance stack before caller score and final
deterministic tie-breaks.

Final metadata result ordering is:

1. exact whole-phrase occurrence count
2. exact longer phrase-run vector, lexicographic
3. exact distinct query-token coverage
4. fuzzy/typo whole-phrase match
5. fuzzy/typo phrase-run vector, lexicographic
6. total distinct query-token coverage, exact plus fuzzy
7. typo severity
8. whether all query tokens are matched
9. field quality
10. same-order scattered matching
11. closeness / gap distance
12. repeated exact phrase bonus, saturated
13. repeated token bonus, saturated
14. density / low clutter
15. descending caller `score`
16. `SearchRank`
17. `key`

Where:

- exact phrase stages use contiguous query-token windows matched by token-prefix exactness
- fuzzy phrase stages use the same contiguous window logic, but allow typo matches per token
- non-typo fuzzy matching and typo matching are token-level, not whole-field, and require shared token signal before edit-distance work runs
- one- and two-character metadata queries only use exact/prefix-style matching; fuzzy and typo token matching start at three characters to keep early interactive search cheap
- typo severity uses the shared intended-to-typed edit engine from `core.rs` and compares:
  - summed core typo cost
  - then summed per-token `position`
  - then summed `insertions`
  Here metadata `position` is still the token-local typo position used by the metadata matcher:
  `0` for start, `1` for end, and `2` for middle.
- field quality is `SearchField.priority`, ascending
- same-order scattered matching prefers candidates where all query tokens can be recovered in query order within one field
- closeness / gap distance prefers smaller gaps between those ordered matches
- density prefers candidates where matched concepts occupy a larger fraction of a smaller field

`SearchRank` is still computed and used for provenance, highlighting, and as a
late tie-break, but it is no longer the primary metadata result order.
The final tail of the comparator is:

1. descending caller `score`
2. `SearchRank`
3. `key`

Shared versus adapter-local:

- shared:
  - the bounded typo engine in `core.rs`
  - `SearchRank` provenance for highlighting and explanations
- adapter-local:
  - phrase-run accounting across multiple fields
  - exact and fuzzy coverage aggregation
  - field-priority policy
  - caller-controlled `score`

## Which Adapter To Use

- Use `path` for filesystem paths.
- Use `command_history` for shell command history lines.
- Use `message` for tokenized free-text records where phrase continuity matters.
- Use `metadata` for apps, windows, or generic multi-field searchable records.

If the data is not naturally a path, do not force it through `path`.
If the data is not naturally shell command history, do not force it through `command_history`.

## Current Shared/Core Story

The current design is:

1. `core` provides one edit-distance engine.
2. `ranking` provides one ranking/comparison framework.
3. `text` provides one shared text/query-prep layer.
4. `token_match` provides one shared token-matching layer.
5. `path`, `command_history`, `message`, and `metadata` adapt those shared layers to their own domains.

So the crate does not have separate ad hoc matcher implementations anymore.
It has:

- one shared engine
- shared ranking helpers
- shared text helpers
- shared token-level matching helpers
- several domain adapters that intentionally keep their own semantics

## Shared Keys vs Adapter Keys

The crate does not currently share every ranking key end-to-end.

What is fully shared:

- raw edit distance
- operation-profile tie-breaks
- ratio normalization
- the tuple ordering for:
  - `TypoSortKey`
  - `PathTypoSortKey`
  - `StructuralRank`
  - `AbbreviationRank`
  - `DistanceRank`
  - `SearchRank`

What is not fully shared:

- the meaning of `position`
- the meaning of `structure`
- field construction
- some normal non-typo path ordering helpers
- some metadata-specific candidate generation and matching logic

So the shared ranking layer standardizes tuple comparison, but the adapters still supply several of the values inside those tuples.

## Inputs And Outputs

### Inputs

Each adapter expects:

- a user query
- a domain-shaped candidate
- an externally computed priority such as `score` or `recency` if ranking should consider recency/frequency

### Outputs

Each adapter returns an adapter-specific match type or rank, for example:

- `PathMatch`
- `CommandHistoryMatch`
- `SearchRank`

Those outputs are intentionally not fully unified because the caller usually needs domain-specific details.

## Caller-Owned Priority

This library separates textual relevance from product-specific priority.

The matcher owns:

- textual ranking
- typo distance
- structural ranking
- field priority within a candidate
- deterministic comparison of match quality

The caller owns:

- pins
- favorites
- app type or category priority
- history/frecency policy
- browse order when no query is active

For example, an application launcher may want:

- pinned apps above normal apps
- normal apps above settings/system modules
- earlier pin positions above later pin positions

Those are not string-matching concepts, so they should not be hardcoded into `fuzzy-rank`.

Instead, callers should express those rules through the candidate `score` they pass in, or through separate non-search ordering when search is empty.

That means the intended split is:

- `fuzzy-rank` decides how good the text match is
- the caller decides how important the candidate is after textual rank ties or near-ties

This is why concepts like `pinned` or `system app` are not explicit library parameters today.
They are application policy, not shared matching policy.

## Design Rule

When adding new functionality:

- put generic edit logic in `core`
- put generic comparison logic in `ranking`
- put generic text preparation in `text`
- put generic token-level alignment/matching in `token_match`
- keep only domain semantics in adapters

If a helper would make sense for both `path` and `command_history`, it probably does not belong in either of them.
