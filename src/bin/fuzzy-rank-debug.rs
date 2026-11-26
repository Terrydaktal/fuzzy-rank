use std::env;
use std::io::{self, IsTerminal};
use std::process;

use fuzzy_rank::command_history::{CommandHistoryQuery, parse_command_history_candidate};
use fuzzy_rank::core::{CoverageRank, RankedTokenVector};
use fuzzy_rank::debug::{DebugReport, DebugStage};
use fuzzy_rank::fields::fuzzy::{MetadataCandidate, MetadataQuery, SearchField};
use fuzzy_rank::fields::literal::{MessageCandidate, MessageField, MessageQuery};
use fuzzy_rank::path::fuzzy::{PathMatch, TypoQuery};
use fuzzy_rank::path::literal::bulk;
use fuzzy_rank::text::{split_path_tokens, to_lowercase};

#[derive(Clone, Copy)]
enum ColorMode {
    Auto,
    Always,
    Never,
}

struct Config {
    adaptor: String,
    query: String,
    candidate: String,
    score: f64,
    recency: f64,
    allow_typos: bool,
    color: ColorMode,
    fields: Vec<(u8, String)>,
}

fn main() {
    mod app_version_runtime {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/build-support/app_version_runtime.rs"
        ));
    }
    app_version_runtime::handle_cli();
    if let Err(error) = run() {
        eprintln!("error: {error}");
        eprintln!();
        print_help();
        process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let config = parse_args(env::args().skip(1).collect())?;
    let report = match config.adaptor.as_str() {
        "path" | "path::fuzzy" | "path::typo" | "path-typo" => run_path(&config, PathMode::Normal),
        "path::literal::bulk" | "path::bulk" | "path-bulk" => run_bulk_path(&config),
        "path::literal" | "path::exact" | "path-exact" => run_path(&config, PathMode::Exact),
        "fields::literal" | "message" => run_message(&config),
        "command-history" | "command_history" | "command" => run_command_history(&config),
        "fields::fuzzy" | "metadata" => run_metadata(&config),
        adaptor => {
            return Err(format!(
                "unknown adaptor `{adaptor}`; expected path::literal, path::fuzzy, path::literal::bulk, fields::literal, command-history, or fields::fuzzy"
            ));
        }
    }?;

    print_report(&config, report);
    Ok(())
}

fn parse_args(args: Vec<String>) -> Result<Config, String> {
    let mut positional = Vec::new();
    let mut score = 0.0;
    let mut recency = 0.0;
    let mut allow_typos = false;
    let mut color = ColorMode::Auto;
    let mut fields = Vec::new();
    let mut index = 0;

    while index < args.len() {
        match args[index].as_str() {
            "-h" | "--help" => {
                print_help();
                process::exit(0);
            }
            "--score" => {
                score = parse_f64_option(&args, &mut index, "--score")?;
            }
            "--recency" => {
                recency = parse_f64_option(&args, &mut index, "--recency")?;
            }
            "--field" => {
                let value = next_option_value(&args, &mut index, "--field")?;
                let (priority, field) = value
                    .split_once(':')
                    .ok_or_else(|| "--field expects PRIORITY:VALUE".to_string())?;
                let priority = priority
                    .parse::<u8>()
                    .map_err(|_| format!("invalid metadata field priority `{priority}`"))?;
                fields.push((priority, field.to_string()));
            }
            "--typos" => allow_typos = true,
            "--color" => color = ColorMode::Always,
            "--no-color" => color = ColorMode::Never,
            value if value.starts_with('-') => {
                return Err(format!("unknown option `{value}`"));
            }
            value => positional.push(value.to_string()),
        }
        index += 1;
    }

    if positional.len() != 3 {
        return Err("expected ADAPTOR QUERY CANDIDATE".to_string());
    }

    Ok(Config {
        adaptor: positional.remove(0),
        query: positional.remove(0),
        candidate: positional.remove(0),
        score,
        recency,
        allow_typos,
        color,
        fields,
    })
}

fn next_option_value(args: &[String], index: &mut usize, option: &str) -> Result<String, String> {
    *index += 1;
    args.get(*index)
        .cloned()
        .ok_or_else(|| format!("{option} requires a value"))
}

fn parse_f64_option(args: &[String], index: &mut usize, option: &str) -> Result<f64, String> {
    let value = next_option_value(args, index, option)?;
    value
        .parse::<f64>()
        .map_err(|_| format!("invalid number for {option}: `{value}`"))
}

#[derive(Clone, Copy)]
enum PathMode {
    Normal,
    Exact,
}

fn run_path(config: &Config, mode: PathMode) -> Result<DebugReport, String> {
    let query = TypoQuery::new(&config.query).ok_or("path query has no searchable tokens")?;
    let matched = match mode {
        PathMode::Normal => query.best_match(&config.candidate, config.score),
        PathMode::Exact => query.best_exact_match(&config.candidate, config.score),
    };

    let Some(matched) = matched else {
        return Ok(DebugReport::no_match(
            "no path match for this adaptor",
            Vec::new(),
        ));
    };

    let stages = vec![
        DebugStage::new(
            "PATH.EXACT_BASENAME_COMPONENT_PHRASE",
            matched.exact_whole_basename_component().to_string(),
        ),
        DebugStage::new(
            "PATH.EXACT_COMPONENT_PHRASE",
            matched.exact_whole_component().to_string(),
        ),
        DebugStage::new(
            "CORE.EXACT_PHRASE_PRESENT",
            matched.exact_whole_phrase().to_string(),
        ),
        DebugStage::new(
            "CORE.EXACT_SUBPHRASE_COUNTS_VECTOR",
            format!("{:?}", matched.exact_subphrase_counts_vector()),
        ),
        DebugStage::new(
            "CORE.EXACT_COVERAGE",
            matched.exact_distinct_coverage().to_string(),
        ),
        DebugStage::new(
            "PATH.ORDERED_COMPOUND_COVERAGE",
            matched.ordered_compound_coverage().to_string(),
        ),
        DebugStage::new(
            "CORE.ALIGNMENT_COST_VECTOR",
            format_alignment_cost(&matched),
        ),
        DebugStage::new(
            "CORE.ORDER_VIOLATION_DISTANCE",
            matched.order_distance.to_string(),
        ),
        DebugStage::new(
            "PATH.SEGMENT_VECTOR",
            format!(
                "{:?}\n[basename_matches, parent_matches, grandparent_matches, ...]",
                matched.segment
            ),
        ),
        DebugStage::new(
            "CORE.SIMPLE_POSITION_VECTOR",
            format!("{:?}\n[whole_or_prefix, suffix, middle]", matched.position),
        ),
        DebugStage::new("CORE.SCORE_DESC", matched.score.to_string()),
        DebugStage::new("CORE.TYPO_COST_VECTOR", format_path_typo_cost(&matched)),
        DebugStage::new(
            "CORE.INSERTION_VECTOR",
            format!("{:?}", matched.insertions.worst_first()),
        ),
        DebugStage::new("PATH.DEPTH", path_depth(matched.path).to_string()),
        DebugStage::new("CORE.KEY_ASC", matched.path),
    ];

    Ok(DebugReport::matched(
        format_path_match_summary(&matched),
        stages,
    ))
}

fn format_path_match_summary(matched: &PathMatch<'_>) -> String {
    let exact_coverage = matched.exact_distinct_coverage();
    let match_type = if matched.exact {
        "EXACT"
    } else if exact_coverage == 0 && matched.ordered_compound_coverage() == 0 {
        "TYPO"
    } else {
        "MIXED"
    };

    format!("{match_type} MATCH  {}", matched.path)
}

fn format_path_typo_cost(matched: &PathMatch<'_>) -> String {
    format!(
        "{:?}\nkeyboard distance: {}\ndeletions: {}\ntranspositions: {}\nformula: keyboard distance + (deletions * 6) + (transpositions * 5)",
        matched.typo_costs.worst_first(),
        matched.operations.keyboard_distance,
        matched.operations.deletions,
        matched.operations.transpositions,
    )
}

fn format_alignment_cost(matched: &PathMatch<'_>) -> String {
    format!(
        "{:?}\nformula: (substitutions + deletions + transpositions + fragmented + internal breaks) * 1000",
        matched.alignment_costs.worst_first(),
    )
}

fn run_bulk_path(config: &Config) -> Result<DebugReport, String> {
    let query_tokens = split_path_tokens(&config.query)
        .into_iter()
        .map(to_lowercase)
        .collect::<Vec<_>>();
    if query_tokens.is_empty() {
        return Err("path query has no searchable tokens".to_string());
    }
    let Some(matched) = bulk::exact_match(&config.candidate, &query_tokens, config.score) else {
        return Ok(DebugReport::no_match(
            "no bulk exact path match",
            Vec::new(),
        ));
    };

    let stages = vec![
        DebugStage::new(
            "PATH.EXACT_BASENAME_COMPONENT_PHRASE",
            matched.exact_basename_component_phrase_present.to_string(),
        ),
        DebugStage::new(
            "PATH.EXACT_COMPONENT_PHRASE",
            matched.exact_component_phrase_present.to_string(),
        ),
        DebugStage::new(
            "CORE.EXACT_PHRASE_PRESENT",
            matched.exact_phrase_present.to_string(),
        ),
        DebugStage::new(
            "CORE.EXACT_SUBPHRASE_COUNTS_VECTOR",
            format!("{:?}", matched.exact_subphrase_counts_vector),
        ),
        DebugStage::new("CORE.EXACT_COVERAGE", matched.exact_coverage.to_string()),
        DebugStage::new(
            "PATH.ORDERED_COMPOUND_COVERAGE",
            matched.ordered_compound_coverage.to_string(),
        ),
        DebugStage::new(
            "PATH.SEGMENT_VECTOR",
            format!(
                "{:?}\n[basename_matches, parent_matches, grandparent_matches, ...]",
                matched.segment
            ),
        ),
        DebugStage::new(
            "CORE.SIMPLE_POSITION_VECTOR",
            format!("{:?}\n[whole_or_prefix, suffix, middle]", matched.position),
        ),
        DebugStage::new("CORE.SCORE_DESC", matched.score.to_string()),
        DebugStage::new(
            "CORE.INSERTION_VECTOR",
            format!("{:?}", matched.insertions.worst_first()),
        ),
        DebugStage::new("PATH.DEPTH", matched.path_depth.to_string()),
        DebugStage::new("CORE.KEY_ASC", matched.path),
    ];
    Ok(DebugReport::matched(
        format!("EXACT MATCH  {}", matched.path),
        stages,
    ))
}

fn run_message(config: &Config) -> Result<DebugReport, String> {
    let query = MessageQuery::new(&config.query).ok_or("message query has no searchable tokens")?;
    let default_fields;
    let configured_fields;
    let fields = if config.fields.is_empty() {
        default_fields = [MessageField {
            priority: 0,
            value: &config.candidate,
        }];
        &default_fields[..]
    } else {
        configured_fields = config
            .fields
            .iter()
            .map(|(priority, value)| MessageField {
                priority: *priority,
                value: value.as_str(),
            })
            .collect::<Vec<_>>();
        configured_fields.as_slice()
    };
    let candidate = MessageCandidate {
        key: &config.candidate,
        fields,
        score: config.score,
    };
    let Some(matched) = query.search_rank(candidate) else {
        return Ok(DebugReport::no_match("no exact message match", Vec::new()));
    };

    let stages = vec![
        DebugStage::new(
            "CORE.EXACT_PHRASE_PRESENT",
            matched.exact_whole_phrase_present.to_string(),
        ),
        DebugStage::new(
            "CORE.EXACT_SUBPHRASE_COUNTS_VECTOR",
            format!("{:?}", matched.exact_subphrase_counts_vector),
        ),
        DebugStage::new(
            "CORE.EXACT_WEIGHTED_POSITIONAL_COVERAGE_VECTOR",
            format_coverage_rank(&matched.exact_weighted_positional_coverage_vector),
        ),
        DebugStage::new(
            "CORE.EXACT_FIELD_QUALITY",
            matched.exact_field_quality.to_string(),
        ),
        DebugStage::new(
            "CORE.ORDER_VIOLATION_DISTANCE",
            matched.order_distance.to_string(),
        ),
        DebugStage::new("CORE.ORDERED_SPAN.GAP", matched.total_gap.to_string()),
        DebugStage::new(
            "CORE.EXACT_PHRASE_OCCURRENCE_RANK",
            format!("{:?}", matched.exact_phrase_occurrence_rank),
        ),
        DebugStage::new(
            "CORE.EXACT_TOKEN_OCCURRENCE_RANK",
            format!("{:?}", matched.exact_token_occurrence_rank),
        ),
        DebugStage::new("CORE.DENSITY", format!("{:?}", matched.density)),
        DebugStage::new("CORE.SCORE_DESC", matched.score.to_string()),
        DebugStage::new("CORE.KEY_ASC", matched.key),
    ];

    Ok(DebugReport::matched("message match", stages))
}

fn run_command_history(config: &Config) -> Result<DebugReport, String> {
    let parsed = parse_command_history_candidate(&config.candidate, config.recency);
    let query = CommandHistoryQuery::new(&config.query)
        .ok_or("command-history query must have 2+ characters")?;
    let Some(matched) = query.best_match(parsed.as_candidate()) else {
        return Ok(DebugReport::no_match(
            "no command-history match",
            Vec::new(),
        ));
    };

    let stages = vec![
        DebugStage::new(
            "CORE.EXACT_PHRASE_PRESENT",
            matched.exact_phrase_present.to_string(),
        ),
        DebugStage::new("CORE.EXACT_COVERAGE", matched.exact_coverage.to_string()),
        DebugStage::new(
            "CORE.EXACT_SUBPHRASE_COUNTS_VECTOR",
            format!("{:?}", matched.exact_subphrase_counts_vector),
        ),
        DebugStage::new("COMMAND.TOTAL_COVERAGE", matched.total_coverage.to_string()),
        DebugStage::new(
            "COMMAND.FUZZY_SUBPHRASE_COUNTS_VECTOR",
            format!("{:?}", matched.fuzzy_subphrase_counts_vector),
        ),
        DebugStage::new(
            "CORE.TYPO_COST_VECTOR",
            format_ranked_vector(&matched.typo_costs),
        ),
        DebugStage::new(
            "CORE.INSERTION_VECTOR",
            format_ranked_vector(&matched.insertions),
        ),
        DebugStage::new("CORE.OPERATIONS", format!("{:?}", matched.operations)),
        DebugStage::new(
            "CORE.ORDER_VIOLATION_DISTANCE",
            matched.order_distance.to_string(),
        ),
        DebugStage::new("CORE.ORDERED_SPAN.GAP", matched.total_gap.to_string()),
        DebugStage::new("CALLER.RECENCY_DESC", matched.candidate.recency.to_string()),
        DebugStage::new("CORE.KEY_ASC", matched.candidate.command),
        DebugStage::new("DEBUG.DISTANCE", matched.distance.to_string()),
        DebugStage::new("DEBUG.RATIO", format!("{:.6}", matched.ratio)),
        DebugStage::new("DEBUG.EXACT_COUNT", matched.exact_count.to_string()),
        DebugStage::new("DEBUG.SPAN_LEN", matched.span_len.to_string()),
    ];

    Ok(DebugReport::matched("command-history match", stages))
}

fn run_metadata(config: &Config) -> Result<DebugReport, String> {
    let query = MetadataQuery::new_with_literal_fallback(&config.query)
        .ok_or("metadata query has no searchable tokens")?;
    let field_values = if config.fields.is_empty() {
        vec![(0, config.candidate.clone())]
    } else {
        config.fields.clone()
    };
    let fields = field_values
        .iter()
        .map(|(priority, value)| SearchField {
            priority: *priority,
            value: value.as_str(),
        })
        .collect::<Vec<_>>();
    let candidate = MetadataCandidate {
        key: &config.candidate,
        fields: &fields,
        score: config.score,
    };

    Ok(query
        .with_typo_fallback(config.allow_typos)
        .debug_report(candidate))
}

fn format_ranked_vector(vector: &RankedTokenVector) -> String {
    format!(
        "aligned: {:?}\nworst_first: {:?}",
        vector.aligned(),
        vector.worst_first()
    )
}

fn format_coverage_rank(rank: &CoverageRank) -> String {
    format!(
        "weighted_quality: {}\ndistinct_coverage: {}\nclass_counts: {:?}\nclasses: [whole, prefix, suffix, substring]\nweights: [4, 2, 2, 1]",
        rank.weighted_quality, rank.distinct_coverage, rank.class_counts
    )
}

fn path_depth(path: &str) -> usize {
    let trimmed = path.trim_end_matches(['/', '\\']);
    let path = if trimmed.is_empty() { path } else { trimmed };
    path.chars()
        .filter(|character| matches!(character, '/' | '\\'))
        .count()
}

fn print_report(config: &Config, report: DebugReport) {
    let use_color = match config.color {
        ColorMode::Always => true,
        ColorMode::Never => false,
        ColorMode::Auto => io::stdout().is_terminal(),
    };

    println!("{}", paint(use_color, "1;36", "fuzzy-rank debug report"));
    println!("adaptor:   {}", paint(use_color, "1;33", &config.adaptor));
    println!("query:     {}", paint(use_color, "1;33", &config.query));
    println!("candidate: {}", paint(use_color, "1;33", &config.candidate));
    println!();

    let is_path_report = matches!(
        config.adaptor.as_str(),
        "path"
            | "path::fuzzy"
            | "path::literal::bulk"
            | "path::bulk"
            | "path::literal"
            | "path::exact"
            | "path::typo"
            | "path-bulk"
            | "path-exact"
            | "path-typo"
    );
    let result = if report.matched {
        paint(use_color, "1;32", "MATCH")
    } else {
        paint(use_color, "1;31", "NO MATCH")
    };
    if is_path_report && report.matched {
        println!("result:    {}", paint(use_color, "1;32", &report.summary));
    } else if report.summary.contains('\n') {
        println!("result:    {result}");
        for line in report.summary.lines() {
            println!("  {}", paint(use_color, "37", line));
        }
    } else {
        println!(
            "result:    {result}  {}",
            paint(use_color, "37", &report.summary)
        );
    }
    println!();
    println!("{}", paint(use_color, "1;35", "stages:"));
    if report.stages.is_empty() {
        println!("  (no stages were reached)");
    } else {
        for (index, stage) in report.stages.iter().enumerate() {
            if stage.value.contains('\n') {
                let mut lines = stage.value.lines();
                let first = lines.next().unwrap_or_default();
                println!(
                    "  {:>2}. {:<48} {}",
                    index + 1,
                    paint(use_color, "36", stage.name),
                    paint(use_color, "37", first)
                );
                for line in lines {
                    println!("      {}", paint(use_color, "37", line));
                }
            } else {
                println!(
                    "  {:>2}. {:<48} {}",
                    index + 1,
                    paint(use_color, "36", stage.name),
                    paint(use_color, "37", &stage.value)
                );
            }
        }
    }
}

fn paint(enabled: bool, code: &str, value: &str) -> String {
    if enabled {
        format!("\x1b[{code}m{value}\x1b[0m")
    } else {
        value.to_string()
    }
}

fn print_help() {
    println!(
        "Usage: fuzzy-rank-debug ADAPTOR QUERY CANDIDATE [OPTIONS]\n\n\
Adaptors:\n  path::literal\n  path::fuzzy\n  path::literal::bulk\n  fields::literal\n  command-history\n  fields::fuzzy\n\n\
Options:\n  --score VALUE       Caller score for path, message, and metadata\n  --recency VALUE     Caller recency for command-history\n  --field P:VALUE     Metadata field with priority P; repeatable\n  --typos             Enable metadata typo fallback\n  --color             Force ANSI colors\n  --no-color          Disable ANSI colors\n  -h, --help          Show this help\n\n\
Queries and candidates containing spaces must be quoted. Metadata without\n\
--field treats CANDIDATE as one field with priority 0."
    );
}
