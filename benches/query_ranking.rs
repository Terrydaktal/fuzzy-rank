use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use fuzzy_rank::fields::{FieldRankModel, fuzzy, literal};
use fuzzy_rank::path::fuzzy::{PathMatchScratch, TypoQuery};

fn path_scratch(c: &mut Criterion) {
    let paths: Vec<_> = (0..128)
        .map(|index| {
            format!("/home/example/Dev/project-{index}/applicationlauncher/target/release")
        })
        .collect();
    let query = TypoQuery::new("dev applaunch").unwrap();
    let prepared: Vec<_> = paths.iter().map(|path| query.prepare_path(path)).collect();
    let mut group = c.benchmark_group("path_multi_token");
    for reuse in [false, true] {
        group.bench_function(
            if reuse {
                "shared_query_scratch"
            } else {
                "fresh_path_scratch"
            },
            |b| {
                b.iter(|| {
                    let mut scratch = PathMatchScratch::default();
                    for path in &prepared {
                        let matched = if reuse {
                            query.best_typo_match_prepared_with_scratch(path, 0.0, &mut scratch)
                        } else {
                            query.best_typo_match_prepared(path, 0.0)
                        };
                        black_box(matched);
                    }
                });
            },
        );
    }
    group.finish();
}

fn path_single_token_scratch(c: &mut Criterion) {
    let paths: Vec<_> = (0..128)
        .map(|index| {
            format!("/home/example/Dev/applicationlauncher/project-{index}/target/release")
        })
        .collect();
    let query = TypoQuery::new("applaunch").unwrap();
    let prepared: Vec<_> = paths.iter().map(|path| query.prepare_path(path)).collect();
    let mut group = c.benchmark_group("path_single_token");
    for reuse in [false, true] {
        group.bench_function(
            if reuse {
                "shared_query_scratch"
            } else {
                "fresh_path_scratch"
            },
            |b| {
                b.iter(|| {
                    let mut scratch = PathMatchScratch::default();
                    for path in &prepared {
                        black_box(if reuse {
                            query.best_typo_match_prepared_with_scratch(path, 0.0, &mut scratch)
                        } else {
                            query.best_typo_match_prepared(path, 0.0)
                        });
                    }
                });
            },
        );
    }
    group.finish();
}

fn fuzzy_top_k(c: &mut Criterion) {
    let texts: Vec<_> = (0..1000)
        .map(|i| {
            format!(
                "Chrome browser window {i} {}",
                "example settings command options title backup notes ".repeat(3)
            )
        })
        .collect();
    let fields: Vec<_> = texts
        .iter()
        .map(|text| {
            fuzzy::prepare_fields(&[fuzzy::Field {
                priority: 0,
                value: text,
            }])
        })
        .collect();
    let mut group = c.benchmark_group("fields_fuzzy");
    for term in ["chrome", "krome"] {
        let query = fuzzy::Query::new(term).unwrap().with_typo_fallback(true);
        let mut matches: Vec<_> = texts
            .iter()
            .zip(&fields)
            .filter_map(|(key, fields)| {
                let candidate = fuzzy::PreparedCandidate {
                    key,
                    fields,
                    score: 0.0,
                };
                query
                    .search_rank_prepared(candidate)
                    .map(|rank| (candidate, rank))
            })
            .collect();
        // Avoid measuring the already-sorted input shortcut.
        let mut seed = 293871_u64;
        for index in (1..matches.len()).rev() {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            matches.swap(index, (seed >> 32) as usize % (index + 1));
        }
        for top_k in [false, true] {
            group.bench_function(
                format!("{term}/{}", if top_k { "top20" } else { "full_sort" }),
                |b| {
                    b.iter_batched(
                        || matches.clone(),
                        |mut candidates| {
                            if top_k {
                                query.select_top_k_prepared_with_model(
                                    &mut candidates,
                                    20,
                                    &FieldRankModel::default(),
                                    0,
                                );
                            } else {
                                query.sort_matches_prepared(&mut candidates);
                            }
                            black_box(candidates);
                        },
                        criterion::BatchSize::SmallInput,
                    );
                },
            );
        }
    }
    group.finish();
}

fn literal_top_k(c: &mut Criterion) {
    let query = literal::Query::new("server error log").unwrap();
    let texts: Vec<_> = (0..1000)
        .map(|i| {
            format!(
                "{} {} {i}",
                if i < 20 {
                    "server error log"
                } else {
                    "server error"
                },
                "message body text ".repeat(100)
            )
        })
        .collect();
    let candidates: Vec<_> = texts
        .iter()
        .map(|text| {
            literal::Candidate {
                key: text,
                fields: &[literal::Field {
                    priority: 0,
                    value: text,
                }],
                score: 0.0,
            }
            .prepare()
        })
        .collect();
    let mut group = c.benchmark_group("fields_literal");
    for lazy in [false, true] {
        group.bench_function(if lazy { "lazy_top20" } else { "eager_top20" }, |b| {
            b.iter(|| {
                let matches = if lazy {
                    query.search_top_k_prepared(&candidates, 20)
                } else {
                    let mut matches: Vec<_> = candidates
                        .iter()
                        .filter_map(|candidate| query.search_rank_prepared(candidate))
                        .collect();
                    literal::select_top_k(&mut matches, 20);
                    matches
                };
                black_box(matches);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    path_scratch,
    path_single_token_scratch,
    fuzzy_top_k,
    literal_top_k
);
criterion_main!(benches);
