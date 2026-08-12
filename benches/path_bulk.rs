use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use fuzzy_rank::path::literal::bulk::{
    BulkQuery, BulkTopK, PreparedBulkPath, exact_match_prepared,
};

struct Corpus {
    paths: Vec<String>,
    prepared: Vec<PreparedBulkPath>,
}

impl Corpus {
    fn new(size: usize) -> Self {
        let paths = (0..size)
            .map(|index| {
                format!(
                    "/home/lewis/Dev/project-{index}/target/{}/applicationlauncher",
                    if index % 3 == 0 { "release" } else { "debug" }
                )
            })
            .collect::<Vec<_>>();
        let prepared = paths
            .iter()
            .map(|path| PreparedBulkPath::new(path))
            .collect();
        Self { paths, prepared }
    }
}

fn bench_bulk_matching(c: &mut Criterion) {
    let corpus = Corpus::new(100_000);
    let mut group = c.benchmark_group("path_bulk");

    for (name, terms) in [
        ("single_broad", vec!["a".to_string()]),
        ("single_specific", vec!["application".to_string()]),
        (
            "multi_token",
            vec!["application".to_string(), "release".to_string()],
        ),
        ("ordered_compound", vec!["projectrelease".to_string()]),
    ] {
        let query = BulkQuery::new(&terms);
        group.bench_function(name, |b| {
            b.iter(|| {
                let mut top = BulkTopK::new(256);
                let mut match_count = 0usize;
                for (path, prepared) in corpus.paths.iter().zip(&corpus.prepared) {
                    if let Some(candidate) = exact_match_prepared(path, prepared, &query, 0.0) {
                        match_count += 1;
                        top.push(candidate);
                    }
                }
                black_box((match_count, top.into_sorted().len()))
            })
        });
    }

    group.finish();
}

fn bench_preparation(c: &mut Criterion) {
    c.bench_function("prepare_100k_paths", |b| {
        b.iter(|| {
            let corpus = Corpus::new(100_000);
            black_box(corpus.prepared.len());
        })
    });
}

criterion_group!(benches, bench_bulk_matching, bench_preparation);
criterion_main!(benches);
