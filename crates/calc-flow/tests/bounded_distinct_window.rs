use calc_flow::CompileOptions;
use calc_flow::arrow::{
    array::{Int64Array, TimestampMicrosecondArray},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use calc_flow::{
    Batch, BatchMetadata, CompileMode, DistinctSpec, EventTime, ExecutionOptions, InputOptions,
    Program, Result, StateBudget, UdfRegistry, table_input,
};
use std::{collections::BTreeMap, sync::Arc};

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("ts", DataType::Timestamp(TimeUnit::Microsecond, None), true),
        Field::new("q", DataType::Int64, true),
    ]))
}
fn data() -> Batch {
    Batch::table(
        vec![
            RecordBatch::try_new(
                schema(),
                vec![
                    Arc::new(TimestampMicrosecondArray::from(vec![
                        Some(-1),
                        Some(0),
                        Some(9),
                        Some(10),
                        None,
                    ])),
                    Arc::new(Int64Array::from(vec![
                        Some(9),
                        Some(1),
                        Some(1),
                        Some(2),
                        Some(3),
                    ])),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap()
}
fn spec() -> DistinctSpec {
    DistinctSpec::closed_window(
        "ts",
        EventTime::from_micros(0),
        EventTime::from_micros(10),
        StateBudget::new(2, 8 << 20).unwrap(),
    )
    .unwrap()
}

#[tokio::test]
async fn test_selected_window_filters_before_key_projection_and_row_budget() -> Result<()> {
    let input = table_input("events", schema(), InputOptions::default())?;
    let df = input.distinct(vec!["q".into()], spec())?;
    let sq = calc_flow::sql::sql(
        "SELECT DISTINCT q FROM CF_DISTINCT_WINDOW('a', 'ts', 0, 10, 2, 8388608)",
        BTreeMap::from([("a".into(), input)]),
    )?;
    let program = Program::new(
        "fixed_cut",
        CompileMode::Batch,
        BTreeMap::from([("df".into(), df), ("sql".into(), sq)]),
    )?;
    let plan = program
        .compile_batch(&UdfRegistry::new().snapshot(), &CompileOptions::default())
        .await?;
    let result = plan
        .execute(
            BTreeMap::from([("events".into(), data())]),
            ExecutionOptions::default(),
        )
        .await?;
    for output in result.outputs.values() {
        let t = output.table_payload()?;
        assert_eq!(
            t.schema().as_ref(),
            &Schema::new(vec![Field::new("q", DataType::Int64, true)])
        );
        let values = t
            .batches()
            .iter()
            .flat_map(|r| {
                r.column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .iter()
            })
            .collect::<Vec<_>>();
        assert_eq!(values, vec![Some(1)]);
    }
    Ok(())
}
