use calc_flow::arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use calc_flow::{
    AggregateFunction, Comparison, ComparisonCutoff, CompileMode, CompileOptions, HistoryScope,
    InputOptions, PortabilityRequest, Program, Result, TableExpr, UpdateSemantics, WindowSpec,
    table_input,
};
use std::{collections::BTreeMap, sync::Arc, time::Duration};

fn source() -> Result<TableExpr> {
    table_input(
        "events",
        Arc::new(Schema::new(vec![
            Field::new(
                "ts",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new("q", DataType::Int64, false),
        ])),
        InputOptions::default(),
    )
}

fn program(input: TableExpr, time: &str, sql: bool) -> Result<Program> {
    let bars = if sql {
        calc_flow::sql::sql(
            &format!(
                "SELECT window_start, window_end, SUM(q) AS s FROM CF_TUMBLE('a', '{time}', 10) GROUP BY window_start, window_end"
            ),
            BTreeMap::from([("a".into(), input)]),
        )?
    } else {
        input.window(
            WindowSpec::tumbling(time, Duration::from_micros(10))?.aggregate(
                AggregateFunction::Sum,
                "q",
                "s",
            )?,
        )?
    };
    Program::new(
        "proof",
        CompileMode::Batch,
        BTreeMap::from([("bars".into(), bars)]),
    )
}

fn checked() -> Result<CompileOptions> {
    Ok(CompileOptions {
        portability: Some(PortabilityRequest::new(
            Comparison::FinalRelation,
            HistoryScope::AcceptedHistory,
            ComparisonCutoff::AllSourcesEof,
        )?),
    })
}

async fn compile(program: &Program, mode: CompileMode, options: &CompileOptions) -> Result<()> {
    let udfs = calc_flow::UdfRegistrySnapshot::default();
    let program = program.for_mode(mode)?;
    match mode {
        CompileMode::Batch => program.compile_batch(&udfs, options).await.map(|_| ()),
        CompileMode::Stream => program
            .compile_stream(&udfs, &calc_flow::StreamRequirements::default(), options)
            .await
            .map(|_| ()),
    }
}

async fn assert_rejected(program: &Program) -> Result<()> {
    for mode in [CompileMode::Batch, CompileMode::Stream] {
        for options in [CompileOptions::default(), checked()?] {
            let error = compile(program, mode, &options)
                .await
                .expect_err("unsafe time admitted");
            let text = error.to_string();
            assert!(text.contains("capability_mismatch"), "{text}");
            assert!(text.contains("outputs.bars"), "{text}");
        }
    }
    Ok(())
}

#[tokio::test]
async fn wildcard_cannot_prove_a_computed_time_or_its_rename() -> Result<()> {
    for query in [
        "SELECT *, ts - INTERVAL '100 microseconds' AS shifted FROM input",
        "SELECT input.*, ts - INTERVAL '100 microseconds' AS shifted FROM input",
    ] {
        for rename in [false, true] {
            let (input, time) = rewritten(query, rename)?;
            assert_unsafe_time(input, time).await?;
        }
    }
    Ok(())
}

fn rewritten(query: &str, rename: bool) -> Result<(TableExpr, &str)> {
    let input = source()?.sql(query)?;
    if rename {
        Ok((input.sql("SELECT shifted AS time, q FROM input")?, "time"))
    } else {
        Ok((input, "shifted"))
    }
}

async fn assert_unsafe_time(input: TableExpr, time: &str) -> Result<()> {
    for sql in [false, true] {
        let program = program(input.clone(), time, sql)?;
        assert_rejected(&program).await?;
        assert_rejected(&program.with_update_semantics(UpdateSemantics::LatestRelation)).await?;
    }
    Ok(())
}

#[tokio::test]
async fn cte_cannot_shadow_the_registered_input_in_a_time_proof() -> Result<()> {
    let input = source()?.sql(
        "WITH input AS (SELECT ts - INTERVAL '100 microseconds' AS ts, q FROM input) SELECT ts, q FROM input",
    )?;
    for sql in [false, true] {
        let program = program(input.clone(), "ts", sql)?;
        assert_rejected(&program).await?;
        assert_rejected(&program.with_update_semantics(UpdateSemantics::LatestRelation)).await?;
    }
    Ok(())
}

async fn assert_admitted(program: &Program) -> Result<()> {
    for mode in [CompileMode::Batch, CompileMode::Stream] {
        for options in [CompileOptions::default(), checked()?] {
            compile(program, mode, &options).await?;
            compile(
                &program.with_update_semantics(UpdateSemantics::LatestRelation),
                mode,
                &options,
            )
            .await?;
        }
    }
    Ok(())
}

#[tokio::test]
async fn unquoted_sql_time_and_window_identifiers_follow_native_case_rules() -> Result<()> {
    let before = source()?.sql("SELECT TS AS TIME, Q FROM INPUT")?;
    assert_both_frontends(before, "time").await?;
    let bars = calc_flow::sql::sql(
        "SELECT WINDOW_START, WINDOW_END, COUNT(Q) AS N FROM CF_TUMBLE('a', 'ts', 10) GROUP BY WINDOW_START, WINDOW_END",
        BTreeMap::from([("a".into(), source()?)]),
    )?;
    assert_admitted(&Program::new(
        "case",
        CompileMode::Batch,
        BTreeMap::from([("bars".into(), bars)]),
    )?)
    .await
}

async fn assert_both_frontends(input: TableExpr, time: &str) -> Result<()> {
    for sql in [false, true] {
        assert_admitted(&program(input.clone(), time, sql)?).await?;
    }
    Ok(())
}

#[tokio::test]
async fn real_wildcard_time_survives_unrelated_computation() -> Result<()> {
    for query in [
        "SELECT * FROM input",
        "SELECT E.* FROM INPUT AS E",
        "SELECT *, q + 1 AS extra FROM input",
    ] {
        for sql in [false, true] {
            assert_admitted(&program(source()?.sql(query)?, "ts", sql)?).await?;
        }
    }
    Ok(())
}

#[tokio::test]
async fn quoted_identifiers_keep_their_exact_case() -> Result<()> {
    let events = table_input(
        "events",
        Arc::new(Schema::new(vec![
            Field::new(
                "Ts",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new("Q", DataType::Int64, false),
        ])),
        InputOptions::default(),
    )?;
    let before = events.sql(r#"SELECT "eX"."Ts" AS "TIME", "eX"."Q" FROM input AS "eX""#)?;
    let bars = calc_flow::sql::sql(
        "SELECT window_start, window_end, SUM(\"Q\") AS \"Total\" FROM CF_TUMBLE('a', 'TIME', 10) GROUP BY window_start, window_end",
        BTreeMap::from([("a".into(), before)]),
    )?;
    let candidate = Program::new(
        "quoted",
        CompileMode::Batch,
        BTreeMap::from([("bars".into(), bars)]),
    )?;
    let schema = candidate
        .collect_schema(&calc_flow::UdfRegistrySnapshot::default())
        .await?;
    assert_eq!(schema["bars"].field(2).name(), "Total");
    assert_admitted(&candidate).await
}
