//! Finalize fixed UTC event windows using the standalone Rust core.
use calc_flow::arrow::{
    array::{Int8Array, Int64Array, TimestampMicrosecondArray, UInt64Array},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use calc_flow::{
    AggregateFunction, Batch, BatchMetadata, CompileMode, CompileOptions, ExecutionOptions,
    InputOptions, Program, Result, StreamRequirements, UdfRegistry, WindowSpec, table_input,
};
use std::{collections::BTreeMap, sync::Arc, time::Duration};

#[tokio::main]
async fn main() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("ts", DataType::Timestamp(TimeUnit::Microsecond, None), true),
        Field::new("q", DataType::Int8, true),
    ]));
    let input = table_input("trades", schema.clone(), InputOptions::default())?;
    let df = input.window(
        WindowSpec::tumbling("ts", Duration::from_micros(10))?
            .aggregate(AggregateFunction::Count, "q", "n")?
            .aggregate(AggregateFunction::Sum, "q", "volume")?,
    )?;
    let sql = calc_flow::sql::sql(
        "SELECT window_start, window_end, COUNT(q) AS n, SUM(q) AS volume FROM CF_TUMBLE('a', 'ts', 10) GROUP BY window_start, window_end",
        BTreeMap::from([("a".into(), input)]),
    )?;
    let program = Program::new(
        "bars",
        CompileMode::Batch,
        BTreeMap::from([("df".into(), df), ("sql".into(), sql)]),
    )?;
    let udfs = UdfRegistry::new().snapshot();
    let plan = program
        .compile_batch(&udfs, &CompileOptions::default())
        .await?;
    let record = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(TimestampMicrosecondArray::from(vec![-1, 0, 9, 10])),
            Arc::new(Int8Array::from(vec![Some(2), Some(3), None, Some(5)])),
        ],
    )
    .unwrap();
    let result = plan
        .execute(
            BTreeMap::from([(
                "trades".into(),
                Batch::table(vec![record], BatchMetadata::default())?,
            )]),
            ExecutionOptions::default(),
        )
        .await?;
    for batch in result.outputs.values() {
        let records = batch.table_payload()?.batches();
        let actual = records
            .iter()
            .flat_map(|record| {
                let starts = record
                    .column(0)
                    .as_any()
                    .downcast_ref::<TimestampMicrosecondArray>()
                    .unwrap();
                let counts = record
                    .column(2)
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .unwrap();
                let sums = record
                    .column(3)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap();
                (0..record.num_rows())
                    .map(|i| (starts.value(i), counts.value(i), sums.value(i)))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, [(-10, 1, 2), (0, 1, 3), (10, 1, 5)]);
    }
    program
        .for_mode(CompileMode::Stream)?
        .compile_stream(
            &udfs,
            &StreamRequirements::default(),
            &CompileOptions::default(),
        )
        .await?;
    println!("DataFrame and SQL finite window oracles passed; both lifecycles compile");
    Ok(())
}
