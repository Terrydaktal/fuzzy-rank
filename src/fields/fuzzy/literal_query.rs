use super::*;

impl MetadataQuery {
    /// Preserve normal word matching, but match a nonempty symbol-only query
    /// literally instead of rejecting it. Literal fallback never uses typos.
    pub fn new_with_literal_fallback(query: &str) -> Option<Self> {
        Self::new(query).or_else(|| {
            let query = to_lowercase(query.trim());
            (!query.is_empty()).then(|| Self {
                query_tokens: vec![query.clone()],
                query,
                allow_typo: false,
                literal_only: true,
            })
        })
    }

    pub(super) fn literal_rank_fields<F: MetadataFieldView>(
        &self,
        fields: &[F],
    ) -> Option<SearchRank> {
        fields
            .iter()
            .enumerate()
            .filter_map(|(field_index, field)| {
                let value = field.lowered_value();
                let start = value.find(&self.query)?;
                Some(StructuralRank {
                    field_index,
                    field_priority: field.priority(),
                    position_class: if value.as_ref() == self.query {
                        0
                    } else if start == 0 {
                        1
                    } else if start + self.query.len() == value.len() {
                        2
                    } else {
                        3
                    },
                    token_index: 0,
                    token_span_delta: 0,
                    start_idx: value[..start].chars().count(),
                    matched_char_len: self.query.chars().count(),
                    field_len: value.chars().count(),
                })
            })
            .min()
            .map(SearchRank::Structural)
    }

    pub(super) fn literal_highlights(
        &self,
        fields: &[SearchField<'_>],
    ) -> Vec<MatchedFieldHighlight> {
        fields
            .iter()
            .enumerate()
            .filter_map(|(field_index, field)| {
                let mut ranges: Vec<(usize, usize, bool)> = Vec::new();
                let mut offset = 0;
                while let Some(found) = field.value[offset..].find(&self.query) {
                    let start = offset + found;
                    let end = start + self.query.len();
                    if let Some((_, previous_end, _)) = ranges
                        .last_mut()
                        .filter(|(_, previous_end, _)| start <= *previous_end)
                    {
                        *previous_end = (*previous_end).max(end);
                    } else {
                        ranges.push((start, end, true));
                    }
                    // Advance one original character so overlapping occurrences are retained.
                    offset = start + field.value[start..].chars().next().unwrap().len_utf8();
                }
                let &(start, end, _) = ranges.first()?;
                Some(MatchedFieldHighlight {
                    field_index,
                    ranges,
                    focus_range: Some((start, end)),
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbol_fallback_is_opt_in_and_never_matches_missing_or_fuzzy_symbols() {
        for symbol in ["~", "/", "::", "++", "[!]", ".", "_", "—"] {
            assert!(MetadataQuery::new(symbol).is_none());
            let query = MetadataQuery::new_with_literal_fallback(symbol)
                .unwrap()
                .with_typo_fallback(true);
            let value = format!("tool {symbol} project");
            let fields = [SearchField {
                priority: 0,
                value: &value,
            }];
            let candidate = MetadataCandidate {
                key: "match",
                fields: &fields,
                score: 0.0,
            };
            let (rank, highlights) = query.search_rank_with_highlights(candidate).unwrap();
            assert!(matches!(rank, SearchRank::Structural(_)));
            assert_eq!(highlights[0].ranges, vec![(5, 5 + symbol.len(), true)]);
            let explanation = query.explain_candidate(candidate).unwrap();
            assert_eq!(explanation.term_explanations.len(), 1);
            assert_eq!(explanation.term_explanations[0].rank, Some(rank.clone()));
            let report = query.debug_report(candidate);
            assert!(report.matched);
            assert_eq!(report.stages[0].name, "METADATA.LITERAL_FALLBACK");
            assert_eq!(report.stages[0].value, "true");
            let prepared = [PreparedMetadataField::new_with_literal_fallback(0, &value).unwrap()];
            let prepared_candidate = PreparedMetadataCandidate {
                key: "match",
                fields: &prepared,
                score: 0.0,
            };
            assert_eq!(query.search_rank_prepared(prepared_candidate), Some(rank));
            assert_eq!(
                query.rank_features(candidate),
                query.rank_features_prepared(prepared_candidate)
            );
            let fields = [SearchField {
                priority: 0,
                value: "no punctuation here",
            }];
            assert!(
                query
                    .search_rank(MetadataCandidate {
                        fields: &fields,
                        ..candidate
                    })
                    .is_none()
            );
        }
        for empty in ["", " ", "\t\n"] {
            assert!(MetadataQuery::new_with_literal_fallback(empty).is_none());
            assert!(PreparedMetadataField::new_with_literal_fallback(0, empty).is_none());
        }
    }

    #[test]
    fn symbol_only_prepared_fields_keep_literal_matches_and_original_unicode_highlight_offsets() {
        let query = MetadataQuery::new_with_literal_fallback("~~").unwrap();
        let fields = [
            SearchField {
                priority: 0,
                value: "~~~",
            },
            SearchField {
                priority: 1,
                value: "İ🙂 ~~~ and ~~",
            },
        ];
        assert!(PreparedMetadataField::new(0, fields[0].value).is_none());
        let prepared: Vec<_> = fields
            .iter()
            .filter_map(|field| {
                PreparedMetadataField::new_with_literal_fallback(field.priority, field.value)
            })
            .collect();
        assert_eq!(prepared.len(), fields.len());
        let candidate = MetadataCandidate {
            key: "symbols",
            fields: &fields,
            score: 0.0,
        };
        let (rank, highlights) = query.search_rank_with_highlights(candidate).unwrap();
        assert_eq!(
            query.search_rank_prepared(PreparedMetadataCandidate {
                key: "symbols",
                fields: &prepared,
                score: 0.0
            }),
            Some(rank)
        );
        assert_eq!(highlights[0].ranges, vec![(0, 3, true)]);
        let value = fields[1].value;
        assert_eq!(
            highlights[1]
                .ranges
                .iter()
                .map(|(start, end, exact)| (&value[*start..*end], *exact))
                .collect::<Vec<_>>(),
            vec![("~~~", true), ("~~", true)]
        );
    }

    #[test]
    fn literal_fallback_does_not_change_word_queries_rankings_highlights_or_features() {
        let fields = [SearchField {
            priority: 0,
            value: "fish - ~/Dev/film - Terminal",
        }];
        let candidate = MetadataCandidate {
            key: "film",
            fields: &fields,
            score: 4.0,
        };
        for text in ["film", "film!", "~/Dev", "fiom"] {
            for typos in [false, true] {
                let normal = MetadataQuery::new(text).unwrap().with_typo_fallback(typos);
                let fallback = MetadataQuery::new_with_literal_fallback(text)
                    .unwrap()
                    .with_typo_fallback(typos);
                assert_eq!(
                    normal.search_rank_with_highlights(candidate),
                    fallback.search_rank_with_highlights(candidate)
                );
                assert_eq!(
                    normal.rank_features(candidate),
                    fallback.rank_features(candidate)
                );
            }
        }
    }

    #[test]
    fn symbol_fallback_prepared_and_unprepared_sorting_keep_payloads_and_caller_scores_consistent()
    {
        let query = MetadataQuery::new_with_literal_fallback("~")
            .unwrap()
            .with_typo_fallback(true);
        let values = ["~", "~ project", "project ~", "unrelated"];
        let fields: Vec<_> = values
            .iter()
            .map(|value| [SearchField { priority: 0, value }])
            .collect();
        let prepared_fields: Vec<_> = values
            .iter()
            .map(|value| [PreparedMetadataField::new_with_literal_fallback(0, value).unwrap()])
            .collect();
        let mut raw = Vec::new();
        let mut prepared = Vec::new();
        for index in 0..values.len() {
            let candidate = MetadataCandidate {
                key: values[index],
                fields: &fields[index],
                score: index as f64,
            };
            let prepared_candidate = PreparedMetadataCandidate {
                key: values[index],
                fields: &prepared_fields[index],
                score: index as f64,
            };
            assert_eq!(
                query.search_rank(candidate),
                query.search_rank_prepared(prepared_candidate)
            );
            if let Some(rank) = query.search_rank(candidate) {
                raw.push((candidate, rank.clone()));
                prepared.push((prepared_candidate, rank, index));
            }
        }
        assert_eq!(raw.len(), 3);
        query.sort_matches(&mut raw);
        query.sort_matches_prepared_with(&mut prepared);
        assert_eq!(
            raw.iter()
                .map(|(candidate, _)| candidate.key)
                .collect::<Vec<_>>(),
            prepared
                .iter()
                .map(|(candidate, _, _)| candidate.key)
                .collect::<Vec<_>>()
        );
        assert!(
            prepared
                .iter()
                .all(|(candidate, _, payload)| candidate.key == values[*payload])
        );
        for pair in raw.windows(2) {
            assert!(
                !query
                    .compare_candidates(pair[0].0, &pair[0].1, pair[1].0, &pair[1].1)
                    .is_gt()
            );
        }
    }
}
