//! Standalone public `DataFrame` and SQL bounded backward ASOF joins.
use calc_flow::arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use calc_flow::{
    AsofJoinSide, AsofStateLimits, CompileMode, InputOptions, Program, Result, StreamAsofJoinSpec,
    table_input,
};
use std::{collections::BTreeMap, sync::Arc, time::Duration};

#[tokio::main]
async fn main() -> Result<()> {
    let (p, schema) = declaration()?;
    execute_fixture(&p, schema).await
}

fn declaration() -> Result<(Program, Arc<Schema>)> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("seq", DataType::UInt64, false),
    ]));
    let options = InputOptions {
        entity_by: vec!["k".into()],
        event_time: Some("ts".into()),
        sequence_by: vec!["seq".into()],
    };
    let a = table_input("a", schema.clone(), options.clone())?;
    let b = table_input("b", schema.clone(), options)?;
    let spec = join_spec()?;
    let config = serde_json::json!({"spec": &spec});
    let df = a.asof_join(&b, spec)?;
    let sq = calc_flow::sql::sql(
        &format!("SELECT * FROM CF_ASOF_JOIN('a', 'b', '{config}')"),
        BTreeMap::from([("a".into(), a), ("b".into(), b)]),
    )?;
    let p = Program::new(
        "joins",
        CompileMode::Batch,
        BTreeMap::from([("df".into(), df), ("sql".into(), sq)]),
    )?;
    Ok((p, schema))
}
fn join_spec() -> Result<StreamAsofJoinSpec> {
    let side = |prefix: &str| {
        AsofJoinSide::new(
            vec!["k".into()],
            "ts".into(),
            vec!["seq".into()],
            prefix.into(),
        )
    };
    StreamAsofJoinSpec::new(
        side("left")?,
        side("right")?,
        Duration::from_micros(2),
        AsofStateLimits::new(100, 1_048_576)?,
    )
}
async fn execute_fixture(p: &Program, schema: Arc<Schema>) -> Result<()> {
    let inputs = BTreeMap::from([
        ("a".into(), data(schema.clone(), &[10, 10])?),
        ("b".into(), data(schema, &[7, 8, 8, 13, 14])?),
    ]);
    let result = p
        .compile_batch(
            &calc_flow::UdfRegistry::new().snapshot(),
            &calc_flow::CompileOptions::default(),
        )
        .await?
        .execute(inputs, calc_flow::ExecutionOptions::default())
        .await?;
    for output in result.outputs.values() {
        let sequences: Vec<_> = output
            .table_payload()?
            .batches()
            .iter()
            .flat_map(|b| {
                b.column_by_name("right__seq")
                    .expect("right sequence")
                    .as_any()
                    .downcast_ref::<calc_flow::arrow::array::UInt64Array>()
                    .expect("uint64")
                    .iter()
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(sequences, vec![Some(2), Some(2)]);
    }
    Ok(())
}

fn data(schema: Arc<Schema>, times: &[i64]) -> Result<calc_flow::Batch> {
    use calc_flow::arrow::{
        array::{Int64Array, TimestampMicrosecondArray, UInt64Array},
        record_batch::RecordBatch,
    };
    calc_flow::Batch::table(
        vec![
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(Int64Array::from(vec![1; times.len()])),
                    Arc::new(TimestampMicrosecondArray::from(times.to_vec()).with_timezone("UTC")),
                    Arc::new(UInt64Array::from_iter_values(0..times.len() as u64)),
                ],
            )
            .unwrap(),
        ],
        calc_flow::BatchMetadata::default(),
    )
}
