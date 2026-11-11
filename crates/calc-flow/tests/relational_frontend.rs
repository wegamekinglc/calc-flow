use calc_flow::arrow::datatypes::{DataType, Field, Schema};
use calc_flow::{InputOptions, UdfRegistry, lit, table_input};
use std::sync::Arc;

#[tokio::test]
async fn typed_dataframe_plans_without_rows() -> calc_flow::Result<()> {
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
    let input = table_input("events", schema.clone(), InputOptions::default())?;
    let derived = input.with_columns(vec![(input.column("x")? + lit(1_i64)?).alias("y")?])?;
    let schema = derived
        .collect_schema(
            &UdfRegistry::new().snapshot(),
            calc_flow::CompileMode::Batch,
        )
        .await?;
    assert_eq!(schema.fields().len(), 2);
    assert_eq!(schema.field(1), &Field::new("y", DataType::Int64, true));
    Ok(())
}

#[tokio::test]
async fn dataframe_filter_cast_and_boolean_null_semantics() -> calc_flow::Result<()> {
    use calc_flow::{CompileOptions, Program};
    use std::collections::BTreeMap;
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
    let input = table_input("events", schema.clone(), InputOptions::default())?;
    let selected = filtered_projection(&input)?;
    let program = Program::new(
        "typed",
        calc_flow::CompileMode::Batch,
        BTreeMap::from([("out".into(), selected)]),
    )?;
    let plan = program
        .compile_batch(&UdfRegistry::new().snapshot(), &CompileOptions::default())
        .await?;
    for input in [vec![Some(1), None, Some(3), Some(1)], vec![], vec![None]] {
        assert_projection(&plan, schema.clone(), input).await?;
    }
    Ok(())
}

#[tokio::test]
async fn schema_errors_keep_logical_paths_and_codes() -> calc_flow::Result<()> {
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
    let input = table_input("events", schema.clone(), InputOptions::default())?;
    let other = table_input("other", schema, InputOptions::default())?;
    check_lineage_error(&input, &other).await?;
    check_predicate_error(&input).await?;
    check_division_error(&input).await?;
    Ok(())
}

#[tokio::test]
async fn explicit_inputs_and_immutable_output_builders_keep_inference() -> calc_flow::Result<()> {
    use calc_flow::{CompileMode, Program};
    use std::collections::BTreeMap;
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
    let input = table_input("events", schema, InputOptions::default())?;
    let empty = Program::new("values", CompileMode::Batch, BTreeMap::new())?;
    let inferred = empty
        .for_mode(CompileMode::Stream)?
        .output("out", input.clone())?;
    assert!(empty.outputs().is_empty());
    assert_eq!(inferred.inputs().len(), 1);
    let explicit = empty
        .with_input(input.clone())?
        .output("out", input.clone())?;
    assert_eq!(explicit.fingerprint(), inferred.fingerprint());
    assert!(
        explicit
            .with_input(input.clone())
            .unwrap_err()
            .to_string()
            .contains("duplicate_name")
    );
    assert!(
        explicit
            .output("out", input)
            .unwrap_err()
            .to_string()
            .contains("duplicate_name")
    );
    let udfs = UdfRegistry::new().snapshot();
    assert!(
        empty
            .collect_schema(&udfs)
            .await
            .unwrap_err()
            .to_string()
            .contains("unresolved_type")
    );
    assert!(
        explicit
            .compile_stream(
                &udfs,
                &calc_flow::StreamRequirements::default(),
                &calc_flow::CompileOptions::default()
            )
            .await
            .unwrap_err()
            .to_string()
            .contains("unsupported_mode")
    );
    Ok(())
}

fn numeric_features(x: &calc_flow::ColumnExpr) -> calc_flow::Result<Vec<calc_flow::NamedExpr>> {
    Ok(vec![
        (x.clone() - lit(1_i64)?).alias("difference")?,
        (-x.clone()).alias("negative")?,
        (x.cast(DataType::Float64)? / lit(2.0_f64)?).alias("half")?,
    ])
}
fn nullable_features(x: &calc_flow::ColumnExpr) -> calc_flow::Result<Vec<calc_flow::NamedExpr>> {
    Ok(vec![
        x.is_null().alias("missing")?,
        lit(calc_flow::LiteralValue::Null)?
            .cast(DataType::Int64)?
            .alias("nulls")?,
    ])
}
fn predicate(x: &calc_flow::ColumnExpr) -> calc_flow::Result<calc_flow::ColumnExpr> {
    Ok((!x.is_null() & x.ge(lit(1_i64)?)) | x.eq(lit(0_i64)?))
}
fn filtered_projection(input: &calc_flow::TableExpr) -> calc_flow::Result<calc_flow::TableExpr> {
    let x = input.column("x")?;
    let added = add_features(input, &x)?;
    let filtered = added.filter(predicate(&x)?)?;
    filtered.select(vec![filtered.column("difference")?.alias("difference")?])
}

async fn assert_projection(
    plan: &calc_flow::BatchExecutionPlan,
    schema: Arc<Schema>,
    input: Vec<Option<i64>>,
) -> calc_flow::Result<()> {
    use calc_flow::arrow::{array::Int64Array, record_batch::RecordBatch};
    use calc_flow::{Batch, BatchMetadata, ExecutionOptions};
    use std::collections::BTreeMap;
    let expected: Vec<_> = input.iter().filter_map(|v| v.map(|v| v - 1)).collect();
    let batch = Batch::table(
        vec![
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int64Array::from(input))]).unwrap(),
        ],
        BatchMetadata::default(),
    )?;
    let out = plan
        .execute(
            BTreeMap::from([("events".into(), batch)]),
            ExecutionOptions::default(),
        )
        .await?;
    let actual: Vec<_> = out.outputs["out"]
        .table_payload()?
        .batches()
        .iter()
        .flat_map(|r| {
            r.column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .iter()
                .map(|v| v.unwrap())
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(actual, expected);
    Ok(())
}

async fn check_lineage_error(
    input: &calc_flow::TableExpr,
    other: &calc_flow::TableExpr,
) -> calc_flow::Result<()> {
    let bad = input.select(vec![
        (input.column("x")? + other.column("x")?).alias("mixed")?,
    ])?;
    let udfs = UdfRegistry::new().snapshot();
    let error = bad
        .collect_schema(&udfs, calc_flow::CompileMode::Batch)
        .await
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("outputs.result.mixed.right.lineage: schema_mismatch"),
        "{error}"
    );
    Ok(())
}
async fn check_predicate_error(input: &calc_flow::TableExpr) -> calc_flow::Result<()> {
    let udfs = UdfRegistry::new().snapshot();
    let bad = input.filter(input.column("x")?)?;
    let error = bad
        .collect_schema(&udfs, calc_flow::CompileMode::Batch)
        .await
        .unwrap_err()
        .to_string();
    assert!(
        error.contains(
            "filter.predicate: unsupported_type: filter predicate requires bool, found int64"
        ),
        "{error}"
    );
    Ok(())
}
async fn check_division_error(input: &calc_flow::TableExpr) -> calc_flow::Result<()> {
    let udfs = UdfRegistry::new().snapshot();
    let integer_division = input.select(vec![(input.column("x")? / lit(2_i64)?).alias("half")?])?;
    let error = integer_division
        .collect_schema(&udfs, calc_flow::CompileMode::Batch)
        .await
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("unsupported_type") && error.contains("floating"),
        "{error}"
    );
    Ok(())
}

fn add_features(
    input: &calc_flow::TableExpr,
    x: &calc_flow::ColumnExpr,
) -> calc_flow::Result<calc_flow::TableExpr> {
    let mut columns = numeric_features(x)?;
    columns.extend(nullable_features(x)?);
    input.with_columns(columns)
}
