//! Event-time ordering must retain each entity's own history and identity once.

use std::{collections::BTreeMap, sync::Arc};

use calc_flow::{
    Batch, BatchMetadata, CancellationToken, EdgeCollector, EventTime, ExecutionOptions, JsonMap,
    OperatorMetadata, PipelineBuilder, RollingOperator, RollingSpec, RunResult, StateBudget,
    StreamJobContext, StreamOperator, StreamOperatorContext, UdfRegistry,
};
use datafusion::arrow::{
    array::{ArrayRef, Float64Array, StringArray, TimestampMicrosecondArray, UInt64Array},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use serde_json::json;

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("entity", DataType::Utf8, false),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("sequence", DataType::UInt64, false),
        Field::new("price", DataType::Float64, false),
    ]))
}

fn input(entities: &[&str], times: &[i64]) -> Batch {
    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(entities.to_vec())),
        Arc::new(TimestampMicrosecondArray::from(times.to_vec()).with_timezone("UTC")),
        Arc::new(UInt64Array::from_iter_values(
            0..u64::try_from(times.len()).unwrap(),
        )),
        Arc::new(Float64Array::from_iter_values(
            (0..times.len()).map(|index| f64::from(u32::try_from(index).unwrap()) + 1.0),
        )),
    ];
    Batch::table(
        vec![RecordBatch::try_new(schema(), columns).unwrap()],
        BatchMetadata::default(),
    )
    .unwrap()
}

fn operator(max_rows: u64, duration: Option<u64>) -> calc_flow::Result<RollingOperator> {
    let mut outputs = vec![json!({
        "kind":"rsi", "primitive_version":1, "input":"price", "output":"rsi", "period":1
    })];
    if let Some(micros) = duration {
        outputs.push(json!({
            "kind":"mean", "primitive_version":1,"input":"price", "output":"mean",
            "frame":{"kind":"duration", "micros":micros}, "min_periods":1
        }));
    }
    let spec: RollingSpec = serde_json::from_value(json!({
        "configuration_version":1,"state_layout_version":5,
        "partition_by":["entity"],"event_time":"time","sequence_by":["sequence"],
        "outputs":outputs,"allowed_lateness_micros":0,
        "late_policy":{"kind":"error","scope":"envelope"},
        "value_policy":"stateful_numeric_v1"
    }))
    .unwrap();
    let mut operator = RollingOperator::new("features", schema(), spec)?;
    operator.set_state_budget(StateBudget::new(max_rows, 4 << 20)?)?;
    Ok(operator)
}

async fn execute(
    entities: &[&str],
    times: &[i64],
    max_rows: u64,
    duration: Option<u64>,
) -> calc_flow::Result<RunResult> {
    PipelineBuilder::new("interleaved-prefix")?
        .add_node("features", operator(max_rows, duration)?)?
        .compile_batch(&UdfRegistry::new().snapshot())?
        .execute(
            BTreeMap::from([("input".into(), input(entities, times))]),
            ExecutionOptions::default(),
        )
        .await
}

#[tokio::test]
async fn test_interleaved_indicator_entities_consume_one_identity_each() {
    let result = execute(&["a", "b", "a", "b"], &[1, 2, 3, 4], 2, None)
        .await
        .unwrap();
    let rsi = result.outputs["output"].table_payload().unwrap().batches()[0]
        .column_by_name("rsi")
        .unwrap()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_eq!(
        rsi.iter().collect::<Vec<_>>(),
        [None, None, Some(100.0), Some(100.0)]
    );
    let error = execute(&["a", "b", "a", "b"], &[1, 2, 3, 4], 1, None)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("state_budget"), "{error}");
}

#[tokio::test]
async fn test_interleaved_history_tails_expire_only_at_their_entity_transition() {
    let entities = &["a", "b", "a", "b", "a", "b"];
    let times = &[1, 2, 3, 20, 21, 22];
    let result = execute(entities, times, 5, Some(10)).await.unwrap();
    let mean = result.outputs["output"].table_payload().unwrap().batches()[0]
        .column_by_name("mean")
        .unwrap()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_eq!(
        mean.iter().collect::<Vec<_>>(),
        [
            Some(1.0),
            Some(2.0),
            Some(2.0),
            Some(4.0),
            Some(5.0),
            Some(5.0)
        ]
    );
    let error = execute(entities, times, 4, Some(10)).await.unwrap_err();
    assert!(error.to_string().contains("state_budget"), "{error}");
}

#[tokio::test]
async fn test_stream_prefix_includes_pending_rows_before_final_eviction() {
    let job = StreamJobContext::new(
        23,
        "a".repeat(64),
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "features", None);
    for (limit, rejected) in [(6, false), (4, true)] {
        let mut operator = operator(limit, Some(10)).unwrap();
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data(
                "input",
                input(&["a", "b", "a", "b"], &[1, 2, 100, 101]),
                &context,
                &mut collector,
            )
            .await
            .unwrap();
        let result = operator
            .on_watermark(EventTime::from_micros(102), &context, &mut collector)
            .await;
        if rejected {
            let error = result.unwrap_err();
            assert!(error.to_string().contains("state_budget"), "{error}");
            assert!(collector.drain("output").is_empty());
        } else {
            result.unwrap();
            assert_eq!(collector.drain("output").len(), 1);
        }
    }
}
