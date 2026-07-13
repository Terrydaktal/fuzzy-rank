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

Owns domain-neutral matching and comparison primitives:

- bounded Damerau-Levenshtein
- `OperationProfile`
- typo-limit policy via `max_typos`
- `exact_phrase_count`
- `exact_phrase_run_vector`
- `exact_distinct_coverage`
- `exact_token_occurrence_counts`
- `ordered_token_span`
- `compare_score_desc`
- `compare_key_asc`
- `ratio_milli`

This module is responsible for:

- computing bounded intended-to-typed Damerau-Levenshtein distance
- tracking weighted typo-likelihood penalties for substitutions, insertions, deletions, and transpositions
- supporting bounded early-exit behavior
- providing a consistent raw typo distance to all adapters
- counting token-prefix exact phrases across one or more fields/components
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

### Sort-stage implementation names

Every sort stage is labelled with the named code chunk that implements it. The
same bracketed name in two pipelines means both call the same implementation.
Different names mean adaptor-specific code, even when the stages sound similar.
`CALLER.*` identifies input rather than code calculated by fuzzy-rank.

Shared implementation cross-reference:

| Implementation ID | Used by stages | Shared code |
| --- | --- | --- |
| `CORE.EXACT_PHRASE_COUNT` | path, message, metadata | `exact_phrase_count` in `core.rs` |
| `CORE.EXACT_RUN_VECTOR` | path, message, metadata | `exact_phrase_run_vector` in `core.rs` |
| `CORE.EXACT_COVERAGE` | path, message, metadata | `exact_distinct_coverage` in `core.rs` |
| `CORE.EXACT_TOKEN_COUNTS` | message, metadata | `exact_token_occurrence_counts` in `core.rs` |
| `CORE.ORDERED_SPAN` | message, metadata | `ordered_token_span` ordered-subsequence and gap traversal in `core.rs` |
| `CORE.DISTANCE` | command history; indirectly path and metadata typo stages | bounded Damerau-Levenshtein engine in `core.rs` |
| `CORE.OPERATIONS` | path, command history, metadata | `OperationProfile` in `core.rs` |
| `CORE.TYPO_COST` | path and metadata | weighted typo-cost primitives in `core.rs` |
| `CORE.RATIO` | command history | normalized distance helper in `ranking.rs` |
| `CORE.SCORE_DESC` | every adaptor | `compare_score_desc` in `core.rs` |
| `CORE.KEY_ASC` | every adaptor | `compare_key_asc` in `core.rs` |
| `RANKING.SEARCH_RANK` | metadata | shared provenance rank in `ranking.rs` |

### Adapter sort-order matrix

This table puts every final comparator side by side. Repeated bracketed names
refer to the same implementation. `—` means that the adaptor has no stage at
that position; it does not by itself mean the adaptor is missing required policy.

| Order | `path::exact` | `path::typo` | `command_history` | `message` | `metadata` |
| ---: | --- | --- | --- | --- | --- |
| 1 | `[PATH.SEGMENT]` | `[PATH.MATCH_CLASS]` | `[COMMAND.EXACT_COUNT]` | `[CORE.EXACT_PHRASE_COUNT]` | `[CORE.EXACT_PHRASE_COUNT]` |
| 2 | `[PATH.POSITION]` | `[CORE.EXACT_PHRASE_COUNT]` | `[COMMAND.SPAN]` | `[CORE.EXACT_RUN_VECTOR]` | `[CORE.EXACT_RUN_VECTOR]` |
| 3 | `[CORE.SCORE_DESC]` | `[CORE.EXACT_RUN_VECTOR]` | `[CORE.DISTANCE]` | `[CORE.EXACT_COVERAGE]` | `[CORE.EXACT_COVERAGE]` |
| 4 | `[PATH.MATCH_PENALTY]` | `[CORE.EXACT_COVERAGE]` | `[CORE.OPERATIONS]` | `[CORE.ORDERED_SPAN.MATCHED]` | `[METADATA.FUZZY_PHRASE]` |
| 5 | `[PATH.DEPTH]` | `[CORE.TYPO_COST]` | `[CORE.RATIO]` | `[CORE.ORDERED_SPAN.GAP]` | `[METADATA.FUZZY_RUN_VECTOR]` |
| 6 | `[CORE.KEY_ASC]` | `[PATH.SEGMENT]` | `[CORE.SCORE_DESC]` | `[MESSAGE.PHRASE_BONUS]` | `[METADATA.TOTAL_COVERAGE]` |
| 7 | — | `[PATH.POSITION]` | `[CORE.KEY_ASC]` | `[CORE.EXACT_TOKEN_COUNTS]` | `[METADATA.TYPO_SEVERITY]` |
| 8 | — | `[CORE.SCORE_DESC]` | — | `[MESSAGE.DENSITY]` | `[METADATA.ALL_TOKENS]` |
| 9 | — | `[CORE.OPERATIONS.INSERTIONS]` | — | `[CORE.SCORE_DESC]` | `[METADATA.FIELD_QUALITY]` |
| 10 | — | `[PATH.DEPTH]` | — | `[CORE.KEY_ASC]` | `[CORE.ORDERED_SPAN.MATCHED]` |
| 11 | — | `[CORE.KEY_ASC]` | — | — | `[CORE.ORDERED_SPAN.GAP]` |
| 12 | — | — | — | — | `[METADATA.PHRASE_BONUS]` |
| 13 | — | — | — | — | `[CORE.EXACT_TOKEN_COUNTS]` |
| 14 | — | — | — | — | `[METADATA.DENSITY]` |
| 15 | — | — | — | — | `[CORE.SCORE_DESC]` |
| 16 | — | — | — | — | `[RANKING.SEARCH_RANK]` |
| 17 | — | — | — | — | `[CORE.KEY_ASC]` |

The matrix exposes both forms of reuse:

- the same implementation at the same order, such as message and metadata stages 1-3
- the same implementation at different orders, such as `CORE.SCORE_DESC` and `CORE.KEY_ASC`

Adaptor-only names identify policy that cannot be shared without changing its
meaning, such as path component depth or metadata field priority.

Shared implementation coverage, with each cell showing its sort position:

| Shared implementation | `path::exact` | `path::typo` | `command_history` | `message` | `metadata` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `CORE.EXACT_PHRASE_COUNT` | — | 2 | — | 1 | 1 |
| `CORE.EXACT_RUN_VECTOR` | — | 3 | — | 2 | 2 |
| `CORE.EXACT_COVERAGE` | — | 4 | — | 3 | 3 |
| `CORE.EXACT_TOKEN_COUNTS` | — | — | — | 7 | 13 |
| `CORE.ORDERED_SPAN.MATCHED` | — | — | — | 4 | 10 |
| `CORE.ORDERED_SPAN.GAP` | — | — | — | 5 | 11 |
| `CORE.DISTANCE` | — | through 5 | 3 | — | through 7 |
| `CORE.OPERATIONS` | — | through 5 and 9 | 4 | — | through 7 |
| `CORE.RATIO` | — | — | 5 | — | internal `SearchRank` |
| `CORE.SCORE_DESC` | 3 | 8 | 6 | 9 | 15 |
| `CORE.KEY_ASC` | 6 | 11 | 7 | 10 | 17 |

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

1. `[PATH.SEGMENT]` `segment`
2. `[PATH.POSITION]` `position`
3. `[CORE.SCORE_DESC]` descending `score` from `CALLER.SCORE`
4. `[PATH.MATCH_PENALTY]` `insertions`
5. `[PATH.DEPTH]` `path_depth`
6. `[CORE.KEY_ASC]` `path`

The exact-path helpers map to the same stage IDs:

1. `[PATH.MATCH_QUALITY]` keyword matchability through `match_qualities`
2. `[PATH.SEGMENT]` averaged path segment through `match_path_segment`
3. `[PATH.POSITION]` summed within-segment position through `match_path_position`
4. `[CORE.SCORE_DESC]` caller-provided `CALLER.SCORE`
5. `[PATH.MATCH_PENALTY]` contiguous unmatched token slack through `match_penalty`
6. `[PATH.DEPTH]` path depth through `exact_match`

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

All results use one shared eleven-stage `PathTypoSortKey` comparator from
`ranking.rs`:

1. `[PATH.MATCH_CLASS]` match class: exact before typo
2. `[CORE.EXACT_PHRASE_COUNT]` exact whole-query phrase presence, preferred, within one component; path saturates the shared count to a boolean
3. `[CORE.EXACT_RUN_VECTOR]` exact contiguous query-token run vector, descending and lexicographic
4. `[CORE.EXACT_COVERAGE]` exact distinct query-token coverage, descending
5. `[CORE.TYPO_COST]` weighted core typo cost, with an effective value of `0` for exact matches
6. `[PATH.SEGMENT]` `segment`
7. `[PATH.POSITION]` `position`
8. `[CORE.SCORE_DESC]` descending `CALLER.SCORE`
9. `[CORE.OPERATIONS.INSERTIONS]` `insertions`
10. `[PATH.DEPTH]` `path_depth`
11. `[CORE.KEY_ASC]` `key`

Phrase and coverage stages use token-prefix exactness. Phrase windows never cross
a path separator; distinct coverage may collect matches from multiple components.
Whole-phrase presence is boolean rather than an occurrence count because repeated
directory names are not stronger path relevance evidence.

Both branches are strict lexicographic pipelines. A subordinate key is compared
only when all preceding keys are equal; it can never outweigh an earlier key.
Collection-level callers should similarly partition exact candidates before
requesting typo candidates, using `best_exact_match` and `best_typo_match`.
Several typo values are outputs of the same Damerau-Levenshtein alignment. They
are produced together and then consumed lazily by the comparator; running a new
alignment for each key would add work without making the sort more selective.

Therefore every exact path match ranks before every typo path match. Exact
candidates bypass Damerau-Levenshtein and carry `exact = true` in `PathMatch`.

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

1. `[COMMAND.EXACT_COUNT]` descending `exact_count`
2. `[COMMAND.SPAN]` `span_len`
3. `[CORE.DISTANCE]` `distance`
4. `[CORE.OPERATIONS]` `operations`
5. `[CORE.RATIO]` `ratio_milli`
6. `[CORE.SCORE_DESC]` descending `CALLER.RECENCY`
7. `[CORE.KEY_ASC]` `alphabetical`

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
  - descending recency through `CORE.SCORE_DESC`
  - alphabetical ordering through `CORE.KEY_ASC`
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
  /tmp/mousepad-build/mousepad/mousepad --disable-server

1. `[CORE.EXACT_PHRASE_COUNT]` exact whole-phrase occurrence count
2. `[CORE.EXACT_RUN_VECTOR]` exact longer phrase-run vector, lexicographic
3. `[CORE.EXACT_COVERAGE]` exact distinct query-token coverage
4. `[CORE.ORDERED_SPAN.MATCHED]` same-order scattered matching with an exact-prefix predicate
5. `[CORE.ORDERED_SPAN.GAP]` closeness / gap distance
6. `[MESSAGE.PHRASE_BONUS]` repeated exact phrase bonus, saturated
7. `[CORE.EXACT_TOKEN_COUNTS]` repeated token bonus, saturated by message policy
8. `[MESSAGE.DENSITY]` density / low clutter
9. `[CORE.SCORE_DESC]` descending `CALLER.SCORE`
10. `[CORE.KEY_ASC]` `key`

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

1. `[CORE.EXACT_PHRASE_COUNT]` exact whole-phrase occurrence count
2. `[CORE.EXACT_RUN_VECTOR]` exact longer phrase-run vector, lexicographic
3. `[CORE.EXACT_COVERAGE]` exact distinct query-token coverage
4. `[METADATA.FUZZY_PHRASE]` fuzzy/typo whole-phrase match through `TOKEN_MATCH`
5. `[METADATA.FUZZY_RUN_VECTOR]` fuzzy/typo phrase-run vector through `TOKEN_MATCH`
6. `[METADATA.TOTAL_COVERAGE]` total distinct query-token coverage through `TOKEN_MATCH`
7. `[METADATA.TYPO_SEVERITY]` typo severity built from `CORE.TYPO_COST`
8. `[METADATA.ALL_TOKENS]` whether all query tokens are matched
9. `[METADATA.FIELD_QUALITY]` field quality
10. `[CORE.ORDERED_SPAN.MATCHED]` same-order scattered matching with metadata's exact-or-fuzzy predicate
11. `[CORE.ORDERED_SPAN.GAP]` closeness / gap distance
12. `[METADATA.PHRASE_BONUS]` repeated exact phrase bonus, saturated
13. `[CORE.EXACT_TOKEN_COUNTS]` repeated token bonus, saturated by metadata policy
14. `[METADATA.DENSITY]` density / low clutter
15. `[CORE.SCORE_DESC]` descending `CALLER.SCORE`
16. `[RANKING.SEARCH_RANK]` `SearchRank`
17. `[CORE.KEY_ASC]` `key`

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

1. `[CORE.SCORE_DESC]` descending `CALLER.SCORE`
2. `[RANKING.SEARCH_RANK]` `SearchRank`
3. `[CORE.KEY_ASC]` `key`

Shared versus adapter-local:

- shared:
  - the bounded typo engine in `core.rs`
  - exact phrase count, exact run vectors, exact coverage, and exact token counts in `core.rs`
  - descending score and ascending key comparators in `core.rs`
  - `SearchRank` provenance for highlighting and explanations
- adapter-local:
  - fuzzy phrase-run accounting across multiple fields
  - fuzzy coverage aggregation
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

- exact token-prefix phrase occurrence counting
- exact token-prefix phrase-run vectors
- exact distinct token coverage
- exact per-token occurrence counting
- ordered token-subsequence detection and gap calculation
- raw edit distance
- operation-profile tie-breaks
- ratio normalization
- descending floating-point score comparison
- ascending deterministic string-key comparison
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
