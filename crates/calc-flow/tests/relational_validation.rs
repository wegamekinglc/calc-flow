use calc_flow::arrow::datatypes::{DataType, Field, Schema};
use calc_flow::{
    AnalysisOptions, CalcFlowError, CompileMode, CompileOptions, InputOptions, LiteralValue,
    Program, SupportStatus, UdfRegistry, lit, table_input,
};
use std::{collections::BTreeMap, sync::Arc};

fn input(name: &str) -> calc_flow::Result<calc_flow::TableExpr> {
    table_input(
        name,
        Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)])),
        InputOptions::default(),
    )
}

#[tokio::test]
async fn analysis_reports_schema_errors_at_logical_roles() -> calc_flow::Result<()> {
    let source = input("events")?;
    let invalid_sql = source.sql("SELECT missing FROM input")?;
    let program = Program::new(
        "invalid",
        CompileMode::Batch,
        BTreeMap::from([("out".into(), invalid_sql)]),
    )?;
    let udfs = UdfRegistry::new().snapshot();
    let report = program.analyze(&udfs, &AnalysisOptions::default()).await?;
    assert_eq!(
        report
            .nodes
            .iter()
            .find(|n| n.path == "outputs.out")
            .unwrap()
            .batch
            .support,
        SupportStatus::Unsupported
    );
    assert!(
        report
            .diagnostics
            .iter()
            .any(|d| d.path.starts_with("outputs.out") && d.code == "schema_mismatch")
    );
    let error = program.collect_schema(&udfs).await.unwrap_err();
    assert!(matches!(error, CalcFlowError::Compile { .. }));
    let error = error.to_string();
    for expected in ["outputs.out", "schema_mismatch", "missing"] {
        assert!(error.contains(expected), "{error}");
    }
    assert!(!error.contains("cf_rel_"), "{error}");
    Ok(())
}

#[tokio::test]
async fn derived_projection_cannot_replace_existing_field() -> calc_flow::Result<()> {
    let source = input("events")?;
    let derived = increment(&source, "x")?;
    let error = derived
        .collect_schema(&UdfRegistry::new().snapshot(), CompileMode::Batch)
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("duplicate_name"), "{error}");
    let valid = source.select(vec![source.column("x")?.alias("x")?])?;
    assert_eq!(
        valid
            .collect_schema(&UdfRegistry::new().snapshot(), CompileMode::Batch)
            .await?
            .field(0)
            .name(),
        "x"
    );
    Ok(())
}

#[tokio::test]
async fn explicit_inputs_are_not_silently_reinferred() -> calc_flow::Result<()> {
    let first = input("first")?;
    let second = input("second")?;
    let program = Program::new("explicit", CompileMode::Batch, BTreeMap::new())?
        .with_input(first)?
        .output("out", second)?;
    let error = program
        .collect_schema(&UdfRegistry::new().snapshot())
        .await
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("inputs.second") && error.contains("unresolved_type"),
        "{error}"
    );
    Ok(())
}

#[test]
fn sql_alias_normalization_is_explicit() -> calc_flow::Result<()> {
    let source = input("events")?;
    let error = calc_flow::sql::sql(
        "SELECT * FROM a",
        BTreeMap::from([("A".into(), source.clone()), ("a".into(), source)]),
    )
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("sql.tables.a") && error.contains("duplicate_name"),
        "{error}"
    );
    Ok(())
}

#[test]
fn host_validation_is_precise_and_inputs_are_immutable() -> calc_flow::Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Int64, true),
        Field::new("x", DataType::Int64, true),
    ]));
    let error = table_input("events", schema.clone(), InputOptions::default())
        .unwrap_err()
        .to_string();
    assert!(error.contains("inputs.events.schema[1]") && error.contains("duplicate_name"));
    assert_eq!(schema.fields().len(), 2);
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(
            lit(value)
                .unwrap_err()
                .to_string()
                .contains("invalid_literal")
        );
    }
    assert!(
        input("events")?
            .column("missing")
            .unwrap_err()
            .to_string()
            .contains("schema_mismatch")
    );
    assert!(
        lit(LiteralValue::Null)?
            .cast(DataType::Binary)
            .unwrap_err()
            .to_string()
            .contains("unsupported_type")
    );
    Ok(())
}

#[tokio::test]
async fn mode_shapes_are_checked_before_compilation() -> calc_flow::Result<()> {
    let source = input("events")?;
    let sql = calc_flow::sql::sql(
        "SELECT a.x FROM a JOIN b ON a.x = b.x",
        BTreeMap::from([("a".into(), source.clone()), ("b".into(), source)]),
    )?;
    let batch = Program::new(
        "aliases",
        CompileMode::Batch,
        BTreeMap::from([("out".into(), sql)]),
    )?;
    assert!(
        batch
            .for_mode(CompileMode::Stream)
            .unwrap_err()
            .to_string()
            .contains("unsupported_mode")
    );
    let udfs = UdfRegistry::new().snapshot();
    assert!(
        batch
            .compile_stream(
                &udfs,
                &calc_flow::StreamRequirements::default(),
                &CompileOptions::default()
            )
            .await
            .is_err()
    );
    assert_eq!(
        batch
            .analyze(&udfs, &AnalysisOptions::default())
            .await?
            .nodes
            .iter()
            .find(|n| n.path == "outputs.out")
            .unwrap()
            .stream
            .support,
        SupportStatus::Unsupported
    );
    Ok(())
}

#[test]
fn projected_names_and_special_input_paths_are_checked_without_planning() -> calc_flow::Result<()> {
    let source = input("event.s")?;
    let error = source.column("missing").unwrap_err().to_string();
    assert!(error.contains("inputs[\"event.s\"].column"), "{error}");
    let projected = source.select(vec![source.column("x")?.alias("renamed")?])?;
    assert!(
        projected
            .column("x")
            .unwrap_err()
            .to_string()
            .contains("schema_mismatch")
    );
    Ok(())
}

#[tokio::test]
async fn schema_planning_rejects_unresolved_or_nonportable_outputs() -> calc_flow::Result<()> {
    let source = input("events")?;
    let udfs = UdfRegistry::new().snapshot();
    for (table, code) in schema_failure_cases(&source)? {
        let error = table
            .collect_schema(&udfs, CompileMode::Batch)
            .await
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("outputs.result") && error.contains(code),
            "{error}"
        );
    }
    Ok(())
}

#[tokio::test]
async fn native_input_identity_survives_dataframe_but_is_cleared_by_sql() -> calc_flow::Result<()> {
    let filtered = ordered_dataframe()?;
    let sql = filtered.sql("SELECT * FROM input ORDER BY ts")?;
    let program = Program::new(
        "identity",
        CompileMode::Batch,
        BTreeMap::from([("df".into(), filtered), ("sql".into(), sql)]),
    )?;
    let report = program
        .analyze(&UdfRegistry::new().snapshot(), &AnalysisOptions::default())
        .await?;
    let df = report
        .nodes
        .iter()
        .find(|n| n.path == "outputs.df")
        .unwrap();
    assert!(df.batch.ordering.temporal_identity.is_some());
    let sql = report
        .nodes
        .iter()
        .find(|n| n.path == "outputs.sql")
        .unwrap();
    assert!(sql.batch.ordering.temporal_identity.is_none());
    assert!(
        report
            .nodes
            .iter()
            .any(|n| n.path == "outputs.df.filter.value")
    );
    Ok(())
}

#[test]
fn readonly_sql_constructor_keeps_public_query_path() -> calc_flow::Result<()> {
    let source = input("events")?;
    let error = source.sql("DELETE FROM input").unwrap_err().to_string();
    assert!(
        error.contains("sql.query") && error.contains("invalid_literal"),
        "{error}"
    );
    Ok(())
}

#[test]
fn readonly_sql_errors_keep_typed_public_diagnostics() -> calc_flow::Result<()> {
    let source = input("events")?;
    for query in [
        "DELETE FROM input",
        "UPDATE input SET x = 1",
        "INSERT INTO input VALUES (1)",
        "CREATE TABLE output (x BIGINT)",
        "SELECT x FROM input; SELECT x FROM input",
        "SELECT ( FROM input",
    ] {
        let error = source.sql(query).unwrap_err();
        let CalcFlowError::InvalidArgument { field, message } = error else {
            panic!("unexpected error for {query}: {error}");
        };
        assert_eq!(field, "sql.query", "{query}");
        assert!(
            message.starts_with("invalid_literal: "),
            "{query}: {message}"
        );
    }
    Ok(())
}

#[test]
fn window_sql_errors_keep_their_specific_field_and_code() -> calc_flow::Result<()> {
    let source = input("events")?;
    for (query, path, code) in [
        (
            "SELECT window_start, window_end, COUNT(x) AS n FROM CF_TUMBLE('a', 'ts', 10, 0) GROUP BY window_start, window_end",
            "sql.window.geometry",
            "invalid_literal",
        ),
        (
            "SELECT window_start, window_end, COUNT(*) AS n FROM CF_TUMBLE('a', 'ts', 10) GROUP BY window_start, window_end",
            "sql.window",
            "capability_mismatch",
        ),
    ] {
        let error =
            calc_flow::sql::sql(query, BTreeMap::from([("a".into(), source.clone())])).unwrap_err();
        let CalcFlowError::InvalidArgument { field, message } = error else {
            panic!("unexpected window error: {error}");
        };
        assert_eq!(field, path);
        assert!(message.starts_with(&format!("{code}: ")), "{message}");
    }
    Ok(())
}

#[tokio::test]
async fn known_empty_input_schema_is_not_an_unresolved_schema() -> calc_flow::Result<()> {
    let source = table_input("empty", Arc::new(Schema::empty()), InputOptions::default())?;
    let program = Program::new(
        "empty",
        CompileMode::Batch,
        BTreeMap::from([("out".into(), source)]),
    )?;
    let udfs = UdfRegistry::new().snapshot();
    assert!(
        program.collect_schema(&udfs).await?["out"]
            .fields()
            .is_empty()
    );
    let _ = program
        .compile_batch(&udfs, &CompileOptions::default())
        .await?;
    Ok(())
}

#[tokio::test]
async fn typed_literals_comparisons_and_casts_keep_native_schemas() -> calc_flow::Result<()> {
    let source = input("events")?;
    let x = source.column("x")?;
    let mut columns = comparison_columns(&x)?;
    columns.extend(literal_columns()?);
    let table = source.select(columns)?;
    let schema = table
        .collect_schema(&UdfRegistry::new().snapshot(), CompileMode::Batch)
        .await?;
    assert_eq!(schema.field(0).data_type(), &DataType::Boolean);
    assert_eq!(schema.field(5).data_type(), &DataType::UInt64);
    assert_eq!(schema.field(8).data_type(), &DataType::Date32);
    assert!(!schema.field(3).is_nullable());
    assert!(!schema.field(4).is_nullable());
    assert!(lit(1_i64)?.alias("").is_err());
    assert!(input("").is_err());
    Ok(())
}

fn comparison_columns(x: &calc_flow::ColumnExpr) -> calc_flow::Result<Vec<calc_flow::NamedExpr>> {
    Ok(vec![
        x.ne(lit(1_i64)?).alias("ne")?,
        x.lt(lit(2_i64)?).alias("lt")?,
        x.le(lit(3_i64)?).alias("le")?,
        x.is_not_null().alias("present")?,
    ])
}
fn literal_columns() -> calc_flow::Result<Vec<calc_flow::NamedExpr>> {
    let mut columns = vec![lit(true)?.alias("flag")?, lit(u64::MAX)?.alias("unsigned")?];
    columns.extend(string_and_date_literals()?);
    Ok(columns)
}
fn string_and_date_literals() -> calc_flow::Result<Vec<calc_flow::NamedExpr>> {
    Ok(vec![
        lit("a'b")?.alias("text")?,
        lit(String::from("owned"))?.alias("owned")?,
        date_literal()?,
    ])
}
fn date_literal() -> calc_flow::Result<calc_flow::NamedExpr> {
    lit(LiteralValue::Null)?
        .cast(DataType::Date32)?
        .alias("date")
}

#[tokio::test]
async fn unsigned_negation_uses_the_shared_row_local_type_constraint() -> calc_flow::Result<()> {
    let source = table_input(
        "events",
        Arc::new(Schema::new(vec![Field::new("u", DataType::UInt64, true)])),
        InputOptions::default(),
    )?;
    let result = source.select(vec![(-source.column("u")?).alias("negative.value")?])?;
    let error = result
        .collect_schema(&UdfRegistry::new().snapshot(), CompileMode::Batch)
        .await
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("outputs.result[\"negative.value\"]")
            && error.contains("unsupported_type")
            && error.contains("signed"),
        "{error}"
    );
    Ok(())
}

fn increment(source: &calc_flow::TableExpr, name: &str) -> calc_flow::Result<calc_flow::TableExpr> {
    source.select(vec![(source.column("x")? + lit(1_i64)?).alias(name)?])
}

fn schema_failure_cases(
    source: &calc_flow::TableExpr,
) -> calc_flow::Result<Vec<(calc_flow::TableExpr, &'static str)>> {
    Ok(vec![
        (
            source.select(vec![lit(LiteralValue::Null)?.alias("n")?])?,
            "unresolved_type",
        ),
        (
            source.sql("SELECT CAST(x AS DECIMAL(20, 2)) AS amount FROM input")?,
            "unsupported_type",
        ),
        (source.sql("SELECT x, x FROM input")?, "duplicate_name"),
    ])
}

fn ordered_input() -> calc_flow::Result<calc_flow::TableExpr> {
    use calc_flow::arrow::datatypes::TimeUnit;
    table_input(
        "events",
        Arc::new(Schema::new(vec![
            Field::new("symbol", DataType::Utf8, false),
            Field::new(
                "ts",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("seq", DataType::Int64, false),
            Field::new("x", DataType::Int64, true),
        ])),
        InputOptions {
            entity_by: vec!["symbol".into()],
            event_time: Some("ts".into()),
            sequence_by: vec!["seq".into()],
        },
    )
}
fn ordered_dataframe() -> calc_flow::Result<calc_flow::TableExpr> {
    let source = ordered_input()?;
    let added = source.with_columns(vec![(source.column("x")? + lit(1_i64)?).alias("y")?])?;
    added.filter(added.column("x")?.gt(lit(0_i64)?))
}
