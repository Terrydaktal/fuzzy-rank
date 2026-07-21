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
    ├── debug.rs
    ├── ranking.rs
    ├── text.rs
    ├── token_match.rs
    ├── path.rs
    ├── path/
    │   ├── exact.rs
    │   └── typo.rs
    ├── bin/
    │   └── fuzzy-rank-debug.rs
    ├── command_history.rs
    ├── message.rs
    └── metadata.rs
```

## Debug CLI

The crate includes `fuzzy-rank-debug`, a diagnostic command that runs one
adaptor against one query and candidate and prints every comparator stage with
the value produced at that stage. It intentionally evaluates all stages for
the report, even when the normal lazy comparator would stop after an earlier
stage. ANSI colors are enabled automatically for a terminal; use `--color` or
`--no-color` to override that behavior.

Run it from this repository with:

```sh
cargo run --bin fuzzy-rank-debug -- ADAPTOR "QUERY" "CANDIDATE" [OPTIONS]
```

Examples:

```sh
cargo run --bin fuzzy-rank-debug -- path::typo applaunch \
  /home/lewis/Dev/applicationlauncher
cargo run --bin fuzzy-rank-debug -- message "cachyos ssh" \
  "cachyos setup uses ssh and ssh-agent" --score 10
cargo run --bin fuzzy-rank-debug -- message "chat ssh" \
  "message body" --field "0:Chat title" --field "1:message body"
cargo run --bin fuzzy-rank-debug -- command-history elextron \
  "npx electron ." --recency 42
cargo run --bin fuzzy-rank-debug -- metadata "chrome profile" \
  chrome --field "0:Google Chrome" --field "2:chrome --profile-directory=Default" --typos
```

Supported adaptor names are `path`, `path::bulk`, `path::exact`, `path::typo`,
`message`, `command-history`, and `metadata`. `path` runs the unified exact-first
path adapter; `path::bulk` uses the high-throughput exact implementation, while
the other qualified names force one path class. Metadata treats the
candidate as one priority-0 field unless one or more repeatable `--field
PRIORITY:VALUE` options are supplied. Message uses the same rule: its positional
candidate is a priority-0 field, and repeatable `--field PRIORITY:VALUE`
options replace it with prioritized message fields. `--score` and `--recency`
are caller inputs, not values calculated by fuzzy-rank.

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

Owns domain-neutral matching and comparison primitives:

- bounded Damerau-Levenshtein
- `OperationProfile`
- typo-limit policy via `max_typos`
- `exact_phrase_present` and `exact_phrase_present_by`
- `exact_phrase_occurrence_count` and `exact_phrase_occurrence_count_by`
- `exact_subphrase_counts_vector`
- `exact_distinct_coverage`
- `exact_token_occurrence_counts`
- `saturated_exact_query_token_occurrence_total`
- `ordered_token_span`
- `inversion_count`
- `compare_score_desc`
- `compare_key_asc`
- `ratio_milli`

This module is responsible for:

- computing bounded candidate-to-query Damerau-Levenshtein distance
- tracking weighted typo-likelihood penalties for substitutions, insertions, deletions, and transpositions
- supporting bounded early-exit behavior
- providing a consistent raw typo distance to all adapters
- detecting exact phrase presence across one or more fields/components using an adapter-supplied token predicate
- counting raw exact phrase occurrences only for explicit later tie-breaks
- calculating exact phrase-run vectors and exact distinct coverage
- providing shared score and deterministic-key comparisons

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
- `PathTypoSortKey`
- comparison helpers such as:
  - `compare_search_results`
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
- cheap signal/prefilter checks for adapters that opt into them

Current helpers include:

- `to_lowercase`
- `token_has_signal`
- `has_query_signal` (used by command-history prefiltering)
- `split_whitespace_tokens`
- `split_search_tokens`
- `split_path_tokens`
- `normalize_compact_alnum`

This module is responsible for cheap, reusable preprocessing.

It does not decide final ranking policy.

### Cheap fuzzy gates

`token_has_signal` and `has_query_signal` are rejection gates, not ranking
keys. They avoid edit-distance work when the candidate has too little shared
character signal, so they can affect whether a candidate reaches a matcher but
never change the order of candidates that pass.

Their current adapter-specific uses are:

- `command_history`: `has_query_signal` runs once per command before command
  token matching. It requires at least one query-token signal in the command;
  the later full-query coverage check still requires every query token to
  match. This gate is retained because shell history can be large.
- `metadata`: `token_has_signal` gates the field-level fuzzy and typo
  `SearchRank` paths, each candidate variant in fuzzy ranking, per-token
  `TOKEN_MATCH` typo fallback, and typo highlighting. Literal substring
  matches bypass it. Queries shorter than three characters also bypass typo
  matching.
- `path`: path exact and typo matching do not use either signal gate. Path
  typo candidates are filtered by path/token length and the bounded matcher so
  valid cases such as `fuxxy` -> `fuzzy` cannot be rejected by a bigram
  heuristic.

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

### Sort-stage implementation names

Every sort stage is labelled with the named code chunk that implements it. The
same bracketed name in two pipelines means both call the same implementation.
Different names mean adaptor-specific code, even when the stages sound similar.
`CALLER.*` identifies input rather than code calculated by fuzzy-rank.

Shared implementation cross-reference:

| Implementation ID | Used by stages | Shared code |
| --- | --- | --- |
| `CORE.EXACT_PHRASE_PRESENT` | path, command history, message, metadata | `exact_phrase_present` / `exact_phrase_present_by` in `core.rs` |
| `CORE.EXACT_PHRASE_OCCURRENCE_COUNT` | message, metadata | `exact_phrase_occurrence_count_by` in `core.rs`, used by the field-priority occurrence rank |
| `CORE.EXACT_PHRASE_OCCURRENCE_RANK` | message, metadata | shared `FieldOccurrenceRank` over exact phrase counts grouped by field priority in `core.rs`; metadata uses it early so title counts outrank lower-priority field evidence |
| `CORE.EXACT_SUBPHRASE_COUNTS_VECTOR` | path, message, metadata | `exact_subphrase_counts_vector` in `core.rs`; counts distinct query subphrases present at least once |
| `CORE.EXACT_COVERAGE` | path, command history | `exact_distinct_coverage` in `core.rs`; path uses component substring evidence while command history uses its whole/prefix token predicate |
| `CORE.EXACT_WEIGHTED_POSITIONAL_COVERAGE_VECTOR` | message, metadata | shared `CoverageRank` over exact classes counted by `position_vector` in `core.rs` |
| `CORE.EXACT_FIELD_QUALITY` | message, metadata | best priority supplying literal evidence; adapter field normalization remains local |
| `METADATA.FUZZY_WEIGHTED_POSITIONAL_COVERAGE_VECTOR` | metadata | deferred fuzzy `CoverageRank` over classes counted by the shared `position_vector` counter in `core.rs` |
| `METADATA.FUZZY_WHOLE_PHRASE` | metadata | contiguous metadata token windows through shared token matching |
| `CORE.EXACT_TOKEN_OCCURRENCE_RANK` | message, metadata | shared `FieldOccurrenceRank` over exact token hits grouped by field priority in `core.rs`; metadata uses it immediately after exact phrase occurrence rank |
| `CORE.DENSITY` | message, metadata | `DensityRank` ratio and field-length ordering in `core.rs` |
| `CORE.ORDER_VIOLATION_DISTANCE` | path, command history, message, metadata | shared `inversion_count` in `core.rs`; `ordered_token_span` uses it for field token positions and path typo uses it for token assignments |
| `CORE.ORDERED_SPAN.GAP` | command history, message, metadata | total-gap half of `ordered_token_span` in `core.rs` |
| `CORE.DISTANCE` | command history; indirectly path and metadata typo stages | bounded Damerau-Levenshtein engine in `core.rs` |
| `CORE.OPERATIONS` | path, command history, metadata | `OperationProfile` in `core.rs` |
| `CORE.TYPO_COST_VECTOR` | path, command history, metadata | query-aligned and worst-first weighted typo-cost vectors in `core.rs` |
| `CORE.INSERTION_VECTOR` | path, command history, metadata | query-aligned and worst-first insertion-count vectors in `core.rs` |
| `CORE.SIMPLE_POSITION_VECTOR` | path | shared three-class counter from `simple_position_vector` and `increment_simple_position_vector` in `core.rs` |
| `CORE.POSITION_VECTOR` | metadata | shared configurable class counter from `position_vector` in `core.rs` |
| `CORE.RATIO` | command history | normalized distance helper in `ranking.rs` |
| `CORE.SCORE_DESC` | every adaptor | `compare_score_desc` in `core.rs` |
| `CORE.KEY_ASC` | every adaptor | `compare_key_asc` in `core.rs` |
| `RANKING.SEARCH_RANK` | metadata | shared provenance rank in `ranking.rs` |

`CORE.EXACT_SUBPHRASE_COUNTS_VECTOR` counts how many distinct proper contiguous
query-token subphrases are present at least once, from one less than the full
query length down to two tokens. The complete query is deliberately excluded
because `CORE.EXACT_PHRASE_PRESENT` already owns that evidence. For query
`term1 term2 term3 term4`, the vector is conceptually:

```text
[
  present("term1 term2 term3") + present("term2 term3 term4"),
  present("term1 term2") + present("term2 term3") + present("term3 term4"),
]
```

Each `present(...)` value is either zero or one. Repeating the same subphrase
ten times still contributes one; different query-position subphrases can each
contribute one. The vector is compared lexicographically in descending order,
so a longer phrase-run level always wins before shorter levels.

It does not count one-token subphrases, non-contiguous combinations, reordered
combinations, or typo/fuzzy matches. A phrase cannot cross a metadata field or
path-component boundary. The adapter chooses the literal token predicate:
metadata and message use arbitrary substring matching, while path and command
history use token-prefix matching.

Queries shorter than three tokens produce an empty vector because they have no
proper contiguous subphrase of at least two tokens.

### Adapter sort-order matrix

This table puts every final comparator side by side. Repeated bracketed names
refer to the same implementation. `—` means that the adaptor has no stage at
that position; it does not by itself mean the adaptor is missing required policy.

| Order | `path::exact` | `path::typo` | `command_history` | `message` | `metadata` |
| ---: | --- | --- | --- | --- | --- |
| 1 | `[CORE.EXACT_PHRASE_PRESENT]` | `[CORE.EXACT_PHRASE_PRESENT]` | `[CORE.EXACT_PHRASE_PRESENT]` | `[CORE.EXACT_PHRASE_PRESENT]` | `[CORE.EXACT_PHRASE_PRESENT]` |
| 2 | `[CORE.EXACT_SUBPHRASE_COUNTS_VECTOR]` | `[CORE.EXACT_SUBPHRASE_COUNTS_VECTOR]` | `[CORE.EXACT_COVERAGE]` | `[CORE.EXACT_SUBPHRASE_COUNTS_VECTOR]` | `[CORE.EXACT_SUBPHRASE_COUNTS_VECTOR]` |
| 3 | `[CORE.EXACT_COVERAGE]` | `[CORE.EXACT_COVERAGE]` | `[CORE.EXACT_SUBPHRASE_COUNTS_VECTOR]` | `[CORE.EXACT_WEIGHTED_POSITIONAL_COVERAGE_VECTOR]` | `[CORE.EXACT_WEIGHTED_POSITIONAL_COVERAGE_VECTOR]` |
| 4 | `[PATH.ORDERED_COMPOUND_COVERAGE]` | `[PATH.ORDERED_COMPOUND_COVERAGE]` | `[COMMAND.TOTAL_COVERAGE]` | `[CORE.EXACT_FIELD_QUALITY]` | `[CORE.EXACT_FIELD_QUALITY]` |
| 5 | `[PATH.SEGMENT_VECTOR]` | `[CORE.TYPO_COST_VECTOR]` | `[COMMAND.FUZZY_SUBPHRASE_COUNTS_VECTOR]` | `[CORE.ORDER_VIOLATION_DISTANCE]` | `[CORE.EXACT_PHRASE_OCCURRENCE_RANK]` |
| 6 | `[CORE.SIMPLE_POSITION_VECTOR]` | `[CORE.ORDER_VIOLATION_DISTANCE]` | `[CORE.TYPO_COST_VECTOR]` | `[CORE.ORDERED_SPAN.GAP]` | `[CORE.EXACT_TOKEN_OCCURRENCE_RANK]` |
| 7 | `[CORE.SCORE_DESC]` | `[PATH.SEGMENT_VECTOR]` | `[CORE.INSERTION_VECTOR]` | `[CORE.EXACT_PHRASE_OCCURRENCE_RANK]` | `[METADATA.FUZZY_WHOLE_PHRASE]` |
| 8 | `[CORE.INSERTION_VECTOR]` | `[CORE.SIMPLE_POSITION_VECTOR]` | `[CORE.OPERATIONS]` | `[CORE.EXACT_TOKEN_OCCURRENCE_RANK]` | `[METADATA.FUZZY_SUBPHRASE_COUNTS_VECTOR]` |
| 9 | `[PATH.DEPTH]` | `[CORE.SCORE_DESC]` | `[CORE.ORDER_VIOLATION_DISTANCE]` | `[CORE.DENSITY]` | `[METADATA.FUZZY_WEIGHTED_POSITIONAL_COVERAGE_VECTOR]` |
| 10 | `[CORE.KEY_ASC]` | `[CORE.INSERTION_VECTOR]` | `[CORE.ORDERED_SPAN.GAP]` | `[CORE.SCORE_DESC]` | `[METADATA.FUZZY_FIELD_QUALITY]` |
| 11 | — | `[PATH.DEPTH]` | `[CORE.SCORE_DESC]` | `[CORE.KEY_ASC]` | `[CORE.TYPO_COST_VECTOR]` |
| 12 | — | `[CORE.KEY_ASC]` | `[CORE.KEY_ASC]` | — | `[CORE.INSERTION_VECTOR]` |
| 13 | — | — | — | — | `[CORE.ORDER_VIOLATION_DISTANCE]` |
| 14 | — | — | — | — | `[CORE.ORDERED_SPAN.GAP]` |
| 15 | — | — | — | — | `[CORE.DENSITY]` |
| 16 | — | — | — | — | `[CORE.SCORE_DESC]` |
| 17 | — | — | — | — | `[RANKING.SEARCH_RANK]` |
| 18 | — | — | — | — | `[CORE.KEY_ASC]` |
| 19 | — | — | — | — | — |

The matrix exposes both forms of reuse:

- the same implementation at different orders or behind adapter policy, as shown by the shared implementation table
- the same implementation at different orders, such as `CORE.SCORE_DESC` and `CORE.KEY_ASC`

`CORE.TYPO_COST_VECTOR` stores two views of the same per-token evidence. For a
three-token query, aligned costs `[1, 2, 0]` mean that the first token has cost
`1`, the second cost `2`, and the third is exact. This aligned form is retained
for diagnostics and provenance. Ranking sorts a copy from worst to best, giving
`[2, 1, 0]`, and compares that vector lexicographically ascending. Query-token
order therefore does not make an otherwise identical typo pattern rank
differently. Exact coverage is compared before this vector.

For path and fuzzy metadata typo matching, each token cost is:

```text
keyboard distance + (query-only deletions * 6)
```

Here `deletion` means a character present in the typed query but absent from the
candidate, while `insertion` means a character present in the candidate but
absent from the query. Candidate-side insertions are deliberately excluded from
`CORE.TYPO_COST_VECTOR`; they are retained in `CORE.INSERTION_VECTOR` and used
later as a low-priority noise tie-break. The raw Damerau distance still controls
whether a candidate is admissible, but is not added as a separate sort key.

`CORE.INSERTION_VECTOR` uses the same representation and comparison. For
example, aligned insertions `[0, 4]` are retained for provenance and compared as
worst-first `[4, 0]`. This replaces summed insertion counts.

The position-vector implementations share counting policy but intentionally use
different class maps. `CORE.SIMPLE_POSITION_VECTOR` is path's
`[whole_or_prefix, suffix, middle]`; `CORE.POSITION_VECTOR` is metadata's
`[whole, prefix, suffix, substring]`.

Adaptor-only names identify policy that cannot be shared without changing its
meaning, such as path component depth or metadata field priority.

Shared implementation coverage, with each cell showing its sort position:

| Shared implementation | `path::exact` | `path::typo` | `command_history` | `message` | `metadata` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `CORE.EXACT_PHRASE_PRESENT` | 1 | 1 | 1 | 1 | 1 |
| `CORE.EXACT_PHRASE_OCCURRENCE_COUNT` | — | — | — | 7 | 5 |
| `CORE.EXACT_PHRASE_OCCURRENCE_RANK` | — | — | — | 7 | 5 |
| `CORE.EXACT_SUBPHRASE_COUNTS_VECTOR` | 2 | 2 | 3 | 2 | 2 |
| `CORE.EXACT_COVERAGE` | 3 | 3 | 2 | — | — |
| `PATH.ORDERED_COMPOUND_COVERAGE` | 4 | 4 | — | — | — |
| `CORE.EXACT_WEIGHTED_POSITIONAL_COVERAGE_VECTOR` | — | — | — | 3 | 3 |
| `CORE.EXACT_FIELD_QUALITY` | — | — | — | 4 | 4 |
| `CORE.EXACT_TOKEN_OCCURRENCE_RANK` | — | — | — | 8 | 6 |
| `CORE.ORDER_VIOLATION_DISTANCE` | — | 6 | 9 | 5 | 13 |
| `CORE.ORDERED_SPAN.GAP` | — | — | 8 | 6 | 14 |
| `CORE.DENSITY` | — | — | — | 9 | 15 |
| `CORE.DISTANCE` | — | through 5 | through 6 | — | through 9 |
| `CORE.OPERATIONS` | — | through 5 and 10 | 6 | — | through 9 and 10 |
| `CORE.INSERTION_VECTOR` | 8 | 10 | 7 | — | 10 |
| `CORE.RATIO` | — | — | — | — | internal `SearchRank` |
| `CORE.SCORE_DESC` | 7 | 9 | 11 | 10 | 16 |
| `CORE.KEY_ASC` | 10 | 12 | 12 | 11 | 18 |

`through N` means the shared primitive contributes to the named adaptor stage
rather than appearing as an independent key. A dash makes an omitted shared
primitive explicit, so potential future consolidation can be evaluated without
mistaking different ordinal positions for different code.

## Adapters

## `path.rs`

Purpose:

- split exact-path and typo-path matching
- basename/dirname-aware ranking
- shared top-level namespace for path semantics

This adapter is intended for zoxide-style search.

It owns:

- `path::bulk`
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

`path::exact` owns the reusable non-typo path helpers used internally by the
unified `path::typo` adapter:

- `match_qualities`
- `match_path_segment`
- `match_path_position`
- `match_penalty`
- `exact_match`
- `compare_exact_path_matches`

The standalone exact-path comparator remains available for callers that only
need exact matching. Its sort-key order is:

1. `[CORE.EXACT_PHRASE_PRESENT]` exact whole-query phrase presence within one path component
2. `[CORE.EXACT_SUBPHRASE_COUNTS_VECTOR]` exact proper contiguous query-token subphrase-presence vector
3. `[CORE.EXACT_COVERAGE]` distinct query-token coverage using whole/prefix/suffix/substring evidence as one point per token
4. `[PATH.ORDERED_COMPOUND_COVERAGE]` compact query tokens reconstructed from two or more ordered path tokens
5. `[PATH.SEGMENT_VECTOR]` segment-count vector
6. `[CORE.SIMPLE_POSITION_VECTOR]` position-class count vector
7. `[CORE.SCORE_DESC]` descending `score` from `CALLER.SCORE`
8. `[CORE.INSERTION_VECTOR]` per-query-token insertion slack, worst count first
9. `[PATH.DEPTH]` `path_depth`
10. `[CORE.KEY_ASC]` `path`

The exact-path match is built from the shared exact-relevance stages and these
path-specific helpers:

1. `[CORE.EXACT_PHRASE_PRESENT]`, `[CORE.EXACT_SUBPHRASE_COUNTS_VECTOR]`, and `[CORE.EXACT_COVERAGE]` through the shared `PathExactRelevance`
2. `[PATH.ORDERED_COMPOUND_COVERAGE]` shared ordered path-token reconstruction
3. `[PATH.SEGMENT_VECTOR]` path-segment count vector through `match_path_segment`
4. `[CORE.SIMPLE_POSITION_VECTOR]` within-segment class count vector through `match_path_position`
5. `[CORE.SCORE_DESC]` caller-provided `CALLER.SCORE`
6. `[CORE.INSERTION_VECTOR]` candidate-only slack through `match_penalties`
7. `[PATH.DEPTH]` path depth through `exact_match`

The segment vector is `[basename_count, parent_count, grandparent_count, ...]`. The position
vector is `[whole_or_prefix_count, suffix_count, middle_substring_count]`. Both are
compared lexicographically in descending order. Each matched query token increments exactly one
entry in each vector. Exact terms may match in any path component; basename
matches are preferred by the segment vector rather than required for eligibility.
Trailing `/` and `\\` separators are ignored when determining the basename,
segment vector, and depth, but remain present in the returned path and final
alphabetical key.
Ordered compound coverage is used when a query token that has no direct
whole/prefix/suffix/substring evidence can be reconstructed from at least two
path tokens in path order. Every non-final fragment consumes a complete path
token; the final fragment may consume a token prefix. Intervening path tokens
may be skipped. The reconstructed query token contributes one segment and one
position match, while candidate-only text between its first and last fragment
contributes to that token's insertion slack. Thus `geminipackages` matches
`gemini-cli/packages` with compound coverage `1` and insertion slack `5` for
`-cli/`. Direct exact coverage is an earlier stage and therefore always
outranks this reconstruction.
`insertions` is the candidate-only contiguous unmatched
character count around the matched keyword span(s) inside the chosen
token/component. That makes `application` rank closer to `applications`
than to `applicationlauncher` after position and score have already been compared.

Exact-path candidate construction materializes:

- segment vector
- position vector
- insertion vector
- caller `score`

Exact phrase, subphrase, direct coverage, and ordered compound coverage remain
cached relevance stages. `path_depth` is evaluated only when comparison reaches
that tie-break.

### `path::bulk`

`path::bulk` is the high-throughput execution strategy for scanning large path
collections with the `path::exact` ordering. It normalizes and scans each
candidate once, materializes an immutable `BulkPathMatch`, and supports both a
full sort and bounded top-k collection through `BulkTopK`.

It is not a separate relevance policy: `BulkPathMatch::compare` uses the same
ten exact-path sort stages documented above. The caller may use `BulkTopK`
inside parallel fold/reduce workers without requiring fuzzy-rank itself to own a
thread pool.

The direct literal scan remains the fast path. When it fails, bulk uses the
shared ordered compound matcher, so compact `geminipackages` can match
`gemini-cli/packages` without invoking typo distance.

### `path::typo`

`path::typo` is the unified exact-first path adapter. It owns:

- `TypoQuery`
- `PathMatch`
- `MatchScope`
- `query_from_keywords`

`TypoQuery::best_match` first runs the path exact check for one candidate and only
runs edit-distance matching when that candidate fails it and typo fallback is
enabled. `best_exact_match` and `best_typo_match` expose the two stages separately
so a collection-level caller can evaluate every exact candidate first and skip
the entire typo stage when the exact class is non-empty. This is the preferred
non-interactive fallback flow. A caller that must display both classes can use
`best_match`. `with_typo_fallback(false)` retains exact matching while disabling
typo recovery.

During multi-token typo fallback, candidate tokens are matched independently
within each candidate window rather than being forced into query order. For
example, query `codex rep` can match `repos codex` as `rep -> repos` and
`codex -> codex`, producing zero typo cost. The path remains an exact-class
match only when it passes the normal exact path check; this unordered fallback
prevents reversed but individually exact tokens from being charged as
substitutions.

`[CORE.ORDER_VIOLATION_DISTANCE]` is the inversion count of the chosen query-to-candidate
token assignment. It lets an in-order zero-cost match beat an otherwise
equivalent reversed match without charging token order as typo distance.

The effective `path::typo` comparator is a twelve-stage pipeline. `PathMatch::compare`
owns the first four stages, then delegates the remaining stages to the shared
`PathTypoSortKey` comparator in `ranking.rs`:

1. `[CORE.EXACT_PHRASE_PRESENT]` exact whole-query phrase presence, preferred, within one component
2. `[CORE.EXACT_SUBPHRASE_COUNTS_VECTOR]` exact proper contiguous query-token subphrase-presence vector, descending and lexicographic
3. `[CORE.EXACT_COVERAGE]` exact distinct query-token coverage, descending; each distinct query token contributes one point regardless of whether it matches as a whole token, prefix, suffix, or arbitrary substring
4. `[PATH.ORDERED_COMPOUND_COVERAGE]` compact query tokens reconstructed from two or more ordered path tokens, descending
5. `[CORE.TYPO_COST_VECTOR]` per-query-token weighted typo costs, worst cost first for comparison
6. `[CORE.ORDER_VIOLATION_DISTANCE]` query-to-candidate token assignment inversion count, ascending
7. `[PATH.SEGMENT_VECTOR]` segment-count vector, descending and lexicographic
8. `[CORE.SIMPLE_POSITION_VECTOR]` whole-or-prefix/suffix/middle count vector, descending and lexicographic
9. `[CORE.SCORE_DESC]` descending `CALLER.SCORE`
10. `[CORE.INSERTION_VECTOR]` per-query-token insertion counts, worst count first
11. `[PATH.DEPTH]` `path_depth`
12. `[CORE.KEY_ASC]` `key`

Phrase and subphrase stages use token-prefix exactness. Exact coverage uses
arbitrary substring exactness, so whole-token, prefix, suffix, and internal
substring matches each contribute one point. Phrase windows never cross a path
separator; distinct coverage may collect matches from multiple components.
Whole-phrase presence is boolean rather than an occurrence count because repeated
directory names are not stronger path relevance evidence.

Path typo matching does not use `has_query_signal` as a rejection gate. A cheap
character-bigram heuristic can reject valid short Damerau matches such as
`fuxxy` -> `fuzzy`, so path typo candidates are filtered by path/token length
and the bounded edit-distance matcher instead. Command-history retains its
separate signal prefilter because it searches a much larger history corpus.

Path exact relevance is evaluated lazily. Each candidate first gets one cached
exact-token signal probe using the same substring predicate as exact coverage.
When neither side of a comparison has that signal, exact phrase,
subphrase, and coverage stages are treated as zero and the comparator proceeds
to typo cost. Candidates with exact-token evidence still use those stages in
their documented order. This changes only when keys are calculated, not the
sort order.

Both branches are strict lexicographic pipelines. A subordinate key is compared
only when all preceding keys are equal; it can never outweigh an earlier key.
Collection-level callers should similarly partition exact candidates before
requesting typo candidates, using `best_exact_match` and `best_typo_match`.
Several typo values are outputs of the same Damerau-Levenshtein alignment. They
are produced together and then consumed lazily by the comparator; running a new
alignment for each key would add work without making the sort more selective.

`PathMatch.exact` is not a comparator stage. It is provenance indicating whether
the candidate came from the exact path check, and it remains available to callers
that need to partition exact candidates before requesting typo candidates. Exact
candidates bypass Damerau-Levenshtein, but the unified `path::typo` comparator
does not impose a separate exact-before-typo class; it ranks candidates through
the shared exact-evidence and typo-quality stages above.

The debug CLI includes the inputs used by each stage. For path typo matches,
`CORE.TYPO_COST_VECTOR` shows the worst-first cost vector, keyboard distance,
query-only deletions, and its formula. `CORE.INSERTION_VECTOR` separately shows
candidate-only insertions. Raw Damerau distance and normalized ratio are
admission/provenance data rather than additional path sort keys. The `PathMatch`
value in the result summary likewise includes internal/cache fields for
inspection; it is not a second ordering.

Path typo operation accounting is directional:

- candidate path token/component = target text
- query text = what the user actually typed

So extra typed characters are reported as deletions from the query, while
unmatched candidate characters are reported as insertions into the candidate.
For example, `applaunch` against `applicationlauncher` has zero keyboard/deletion
typo cost, one internal break, and ten candidate-side insertions. Likewise,
`applicationlauncher` against `application launcher` has zero core typo cost and
one candidate-side insertion for the space. The separate insertion vector
records both cases and is intentionally lower priority than core typo cost.

The path-specific comparator uses the exact stages above, then:

- weighted core typo cost: keyboard distance plus six per query-only deletion
- query-to-candidate order violation distance
- the segment vector, descending lexicographically: basename count, parent count, grandparent count, and so on
- the position vector, descending lexicographically: whole token or prefix, suffix, then middle substring
- descending caller `score`
- raw `insertions`, ascending, as a separate noise key

For single-token contiguous substring matches that satisfy the normal path exact
rules, `path::typo` takes the exact fast path instead of running
Damerau-Levenshtein. Those matches come through with:

- `exact = true`
- core typo cost `0`
- raw `insertions` equal to the unmatched candidate-side slack

So among pure insertion-only matches, ranking falls through quickly to `segment`, then within-segment `position`, then to descending `score`, and later to `insertions` if those still tie.

For path typo ranking, `path_depth` is also deferred until a comparison actually reaches that final
structural tie-break.

Shared versus adapter-local:

- shared:
  - the shared weighted `OperationProfile`
  - exact phrase count, exact run vectors, and exact coverage from `core.rs`
  - descending score and ascending key comparators from `core.rs`
  - the tuple ordering in `compare_path_typo_sort_keys`
- adapter-local:
  - splitting core typo cost from raw `insertions`
- segment-count vector
- within-segment position-class vector
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

It uses shared core for exact phrase count, exact coverage, exact subphrase
counts, typo operations, ordered-span distance, gap distance, descending recency,
and alphabetical comparison. `text` supplies lowercase normalization,
whitespace tokenization, and signal checks.

What is command-history-specific here:

- command history is split into search words, so punctuation and underscores create token boundaries
- query tokens are compared against whole command-history search words
- every query token must match one command-history word exactly or fuzzily
- multi-token queries can match non-contiguous or reordered command-history words
- missing-space and extra-space interpretations are not part of this adapter
- caller-provided `recency` breaks equal textual ties

This is the adapter to use for shell command history search.

### Command-history sort order

The exact command-history sort key order is:

1. `[CORE.EXACT_PHRASE_PRESENT]` whether the exact whole phrase occurs at least once
2. `[CORE.EXACT_COVERAGE]` exact distinct query-token coverage
3. `[CORE.EXACT_SUBPHRASE_COUNTS_VECTOR]` exact proper contiguous subphrase-presence vector
4. `[COMMAND.TOTAL_COVERAGE]` total exact-or-fuzzy query-token coverage
5. `[COMMAND.FUZZY_SUBPHRASE_COUNTS_VECTOR]` fuzzy proper contiguous subphrase-count vector
6. `[CORE.TYPO_COST_VECTOR]` per-query-token weighted typo costs, worst cost first for comparison
7. `[CORE.INSERTION_VECTOR]` per-query-token insertion counts, worst count first
8. `[CORE.OPERATIONS]` remaining summed operation-profile tie-break
9. `[CORE.ORDER_VIOLATION_DISTANCE]` out-of-order distance
10. `[CORE.ORDERED_SPAN.GAP]` total positional gap
11. `[CORE.SCORE_DESC]` descending `CALLER.RECENCY`
12. `[CORE.KEY_ASC]` command text

Where:

- exact stages use shared token-prefix semantics
- commands that do not recover every query token are rejected before sorting
- fuzzy subphrase counts include only proper subphrases of at least two tokens
- `operations` is the merged shared candidate-to-query typo-likelihood profile
- `typo_costs.aligned()` retains query order, while sorting uses `typo_costs.worst_first()`
- order distance and total gap operate on exact-or-fuzzy token matches
- `recency` is caller-provided and sorts descending
- command text is the final deterministic alphabetical tie-break

Command-local policy is limited to command tokenization, whole-token typo
eligibility, the full-query acceptance filter, fuzzy subphrase construction,
caller-provided recency, and preserving legacy diagnostic fields for fish.

## `message.rs`

Purpose:

- tokenized text matching for chat logs, notes, or message history
- phrase-aware ordering for exact multi-token sequences
- lightweight precomputation for repeated query evaluation

This adapter is intended for message search where metadata-style exact phrase,
coverage, and field-priority ranking are needed without typo handling. A message
candidate can expose fields such as chat title/message title at priority `0` and
message name/body at priority `1` (lower numbers are stronger).

It owns:

- `MessageQuery`
- `MessageField`
- `MessageCandidate`
- `PreparedMessageCandidate`
- `MessageMatch`
- `contains_query_signal`

It uses shared core from:

- `text` for lowercasing and token splitting
- the shared exact relevance primitives in `core`

What is message-specific here:

- applying the metadata exact-only relevance stack to prioritized message fields
- omitting typo/fuzzy stages entirely
- retaining field quality and field-priority occurrence stages

Message matching first prepares lowercased search tokens for the query and each
candidate field. Fields are tokenized independently, so phrases and spans cannot
cross a field boundary. Candidates that do not contain any usable exact query
signal are rejected before sorting. Exact matching uses arbitrary substring
matching within a candidate token, just like metadata; it does not calculate
fuzzy matches, edit distance, keyboard distance, or typo provenance.

Final message result ordering is:

1. `[CORE.EXACT_PHRASE_PRESENT]` whether the exact whole phrase occurs at least once
2. `[CORE.EXACT_SUBPHRASE_COUNTS_VECTOR]` exact proper contiguous subphrase-presence vector, lexicographic
3. `[CORE.EXACT_WEIGHTED_POSITIONAL_COVERAGE_VECTOR]` weighted whole/prefix/suffix/substring coverage, then distinct coverage, then class counts
4. `[CORE.EXACT_FIELD_QUALITY]` best field priority supplying literal match evidence
5. `[CORE.ORDER_VIOLATION_DISTANCE]` out-of-order distance across the query tokens that matched
6. `[CORE.ORDERED_SPAN.GAP]` total positional gap across the query tokens that matched
7. `[CORE.EXACT_PHRASE_OCCURRENCE_RANK]` exact whole-phrase occurrence counts, compared by field priority and then count
8. `[CORE.EXACT_TOKEN_OCCURRENCE_RANK]` exact query-token occurrence counts, compared by field priority and then count
9. `[CORE.DENSITY]` density / low clutter
10. `[CORE.SCORE_DESC]` descending `CALLER.SCORE`
11. `[CORE.KEY_ASC]` `key`

Where:

- exact phrase stages use contiguous query-token windows matched by literal substring exactness
- exact coverage classifies each distinct query token as whole, prefix, suffix, or substring
- field quality compares lower field priorities first, before field occurrence counts
- order distance counts inversions among matched query tokens, including partial matches
- total gap sums unmatched candidate-token positions between consecutive matched query tokens
- phrase and token occurrence ranks are uncapped, but a stronger-priority field always wins first
- density prefers candidates where matched concepts occupy a larger fraction of a shorter matching field

This is the adapter to use for:

- chat messages
- notes or snippets
- searchable conversation history
- other structured text records where phrase continuity and field priority matter

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

1. `[CORE.EXACT_PHRASE_PRESENT]` whether the whole exact query phrase occurs at least once
2. `[CORE.EXACT_SUBPHRASE_COUNTS_VECTOR]` exact proper contiguous subphrase-presence vector, descending and lexicographic
3. `[CORE.EXACT_WEIGHTED_POSITIONAL_COVERAGE_VECTOR]` weighted exact quality, then distinct coverage, then the whole/prefix/suffix/substring vector
4. `[CORE.EXACT_FIELD_QUALITY]` best field priority supplying literal match evidence
5. `[CORE.EXACT_PHRASE_OCCURRENCE_RANK]` exact whole-phrase occurrence counts, compared by field priority and then count
6. `[CORE.EXACT_TOKEN_OCCURRENCE_RANK]` exact query-token occurrence counts, compared by field priority and then count
7. `[METADATA.FUZZY_WHOLE_PHRASE]` whether a contiguous whole-query match containing at least one typo exists
8. `[METADATA.FUZZY_SUBPHRASE_COUNTS_VECTOR]` fuzzy/typo proper contiguous subphrase-count vector through `TOKEN_MATCH`
9. `[METADATA.FUZZY_WEIGHTED_POSITIONAL_COVERAGE_VECTOR]` weighted fuzzy quality, then distinct fuzzy coverage, then the whole/prefix/suffix/substring/fragmented vector
10. `[METADATA.FUZZY_FIELD_QUALITY]` best field priority attached to a winning fuzzy token match
11. `[CORE.TYPO_COST_VECTOR]` per-query-token weighted typo costs, worst cost first
12. `[CORE.INSERTION_VECTOR]` per-query-token insertion counts, worst count first
13. `[CORE.ORDER_VIOLATION_DISTANCE]` out-of-order distance across the exact-or-fuzzy query tokens that matched
14. `[CORE.ORDERED_SPAN.GAP]` total positional gap across the query tokens that matched
15. `[CORE.DENSITY]` density / low clutter
16. `[CORE.SCORE_DESC]` descending `CALLER.SCORE`
17. `[RANKING.SEARCH_RANK]` `SearchRank`
18. `[CORE.KEY_ASC]` `key`

Where:

- metadata exactness means arbitrary literal substring matching within a token
- exact phrase stages use contiguous query-token windows whose corresponding candidate tokens contain the query tokens
- the early exact whole-phrase stage is boolean, so raw repetition cannot dominate all later relevance evidence
- the exact subphrase vector counts every proper contiguous query subphrase of length two through `query_len - 1`; longer lengths are compared first
- exact coverage uses weights `[4, 2, 2, 1]` for whole, prefix, suffix, and substring; it compares weighted quality first, distinct covered query tokens second, and the class vector last
- all three exact coverage values come from one literal classification pass and require no edit-distance work
- exact field quality is evaluated immediately after exact coverage and never invokes fuzzy matching
- exact phrase and token occurrence ranks follow exact field quality; lower-priority fields cannot beat a higher-priority title/name field by repeating the query more often
- fuzzy coverage uses weights `[4, 2, 2, 1, 1]` for whole, prefix, suffix, substring, and fragmented matches; it compares weighted quality first, distinct fuzzy coverage second, and the class vector last
- fuzzy coverage is initialized only after the preceding keys tie; comparator short-circuiting therefore avoids token typo work whenever exact evidence decides the order
- fuzzy field quality comes from the cached per-token fuzzy match provenance; exact matches do not populate either the fuzzy coverage vector or fuzzy field-quality stage
- fuzzy subphrase counts use the same proper contiguous-window lengths as the exact vector, but allow typo matches per token
- fuzzy whole phrase requires a full contiguous query-length token window and at least one fuzzy token; exact whole phrases remain stage 1
- non-typo fuzzy matching and typo matching are token-level, not whole-field, and require shared token signal before edit-distance work runs
- one- and two-character metadata queries only use exact/prefix-style matching; fuzzy and typo token matching start at three characters to keep early interactive search cheap
- typo severity uses the shared candidate-to-query edit engine from `core.rs` and compares:
  - worst per-token core typo cost, then the next-worst, independent of query order
  - then summed per-token `position`
  - then the per-token insertion vector, worst count first
  Here metadata `position` is still the token-local typo position used by the metadata matcher:
  `0` for start, `1` for end, and `2` for middle.
- both field-quality stages compare `SearchField.priority` ascending and are tied to their corresponding exact or fuzzy evidence
- metadata occurrence ranks are uncapped, but field priority is compared before count: any title hit beats command-line-only hits, regardless of command-line count
- order distance counts inversions among the exact-or-fuzzy query tokens that matched within the best field, including partial matches
- total gap sums unmatched candidate-token positions between consecutive matched query tokens within that field
- density prefers candidates where matched concepts occupy a larger fraction of a smaller field

`SearchRank` is still computed and used for provenance, highlighting, and as a
late tie-break, but it is no longer the primary metadata result order.
The final tail of the comparator is:

1. `[CORE.SCORE_DESC]` descending `CALLER.SCORE`
2. `[RANKING.SEARCH_RANK]` `SearchRank`
3. `[CORE.KEY_ASC]` `key`

Shared versus adapter-local:

- shared:
  - the bounded typo engine in `core.rs`
  - exact phrase presence/count, exact subphrase vectors, position-vector counting, ordered spans, density, and exact token counts in `core.rs`
  - descending score and ascending key comparators in `core.rs`
  - `SearchRank` provenance for highlighting and explanations
- adapter-local:
  - fuzzy phrase-run accounting across multiple fields
  - assigning metadata's exact, fuzzy, and fragmented classes to the shared position-vector counter
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

1. `core` provides one edit-distance engine plus shared exact-relevance and primitive comparison code.
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

- predicate-driven exact phrase presence detection
- predicate-driven raw exact phrase occurrence counting for later tie-breaks
- predicate-driven exact phrase-run vectors
- exact distinct token coverage
- exact per-token occurrence counting
- ordered token-subsequence detection and gap calculation
- raw edit distance
- operation-profile tie-breaks
- ratio normalization
- descending floating-point score comparison
- ascending deterministic string-key comparison
- the tuple ordering for:
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
