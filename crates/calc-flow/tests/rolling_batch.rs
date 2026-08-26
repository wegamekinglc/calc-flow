use std::{collections::BTreeMap, sync::Arc};

use calc_flow::{
    Batch, BatchMetadata, ExecutionOptions, OperatorMetadata, PipelineBuilder, RollingOperator,
    RollingSpec, UdfRegistry,
};
use datafusion::arrow::{
    array::{
        Array, ArrayRef, Float64Array, Int64Array, StringArray, TimestampMicrosecondArray,
        UInt64Array,
    },
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};

fn input_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC"))),
            false,
        ),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("sequence", DataType::UInt64, false),
        Field::new("price", DataType::Float64, true),
        Field::new("volume", DataType::Int64, true),
    ]))
}

fn input_batch(
    event_times: Vec<i64>,
    symbols: Vec<&str>,
    sequences: Vec<u64>,
    prices: Vec<Option<f64>>,
    volumes: Vec<Option<i64>>,
) -> Batch {
    let record = RecordBatch::try_new(
        input_schema(),
        vec![
            Arc::new(TimestampMicrosecondArray::from(event_times).with_timezone("UTC")) as ArrayRef,
            Arc::new(StringArray::from(symbols)),
            Arc::new(UInt64Array::from(sequences)),
            Arc::new(Float64Array::from(prices)),
            Arc::new(Int64Array::from(volumes)),
        ],
    )
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

fn spec() -> RollingSpec {
    serde_json::from_value(serde_json::json!({
        "configuration_version": 1,
        "state_layout_version": 1,
        "partition_by": ["symbol"],
        "event_time": "ts",
        "sequence_by": ["sequence"],
        "outputs": [
            {
                "kind": "lag",
                "primitive_version": 1,
                "input": "price",
                "output": "price_lag_1",
                "periods": 1
            },
            {
                "kind": "delta",
                "primitive_version": 1,
                "input": "volume",
                "output": "volume_delta_2",
                "periods": 2
            }
        ],
        "allowed_lateness_micros": 0,
        "late_policy": {"kind": "error", "scope": "envelope"},
        "value_policy": "stateful_numeric_v1"
    }))
    .unwrap()
}

fn operator() -> RollingOperator {
    RollingOperator::new("rolling", input_schema(), spec()).unwrap()
}

fn output_column<T: Array + Clone + 'static>(batch: &Batch, column: &str) -> T {
    let table = batch.table_payload().unwrap();
    let record = table.batches()[0].clone();
    let array = record
        .column_by_name(column)
        .unwrap_or_else(|| panic!("missing output column {column}"));
    array
        .as_any()
        .downcast_ref::<T>()
        .unwrap_or_else(|| panic!("output column {column} has an unexpected type"))
        .clone()
}

async fn execute(input: Batch) -> calc_flow::Result<Batch> {
    let plan = PipelineBuilder::new("rolling batch")
        .unwrap()
        .add_node("rolling", operator())
        .unwrap()
        .compile_batch(&UdfRegistry::new().snapshot())
        .unwrap();
    let outputs = plan
        .execute(
            BTreeMap::from([("input".into(), input)]),
            ExecutionOptions::default(),
        )
        .await?;
    Ok(outputs.outputs["output"].clone())
}

#[tokio::test]
async fn batch_emits_canonically_ordered_lag_delta_rows_for_interleaved_entities() {
    let output = execute(input_batch(
        vec![30, 10, 20, 20, 10, 20],
        vec!["a", "a", "b", "a", "b", "b"],
        vec![3, 1, 1, 2, 1, 2],
        vec![
            Some(3.0),
            Some(1.0),
            Some(100.0),
            Some(2.0),
            Some(50.0),
            Some(200.0),
        ],
        vec![
            Some(30),
            Some(10),
            Some(1000),
            Some(20),
            Some(500),
            Some(2000),
        ],
    ))
    .await
    .unwrap();

    let event_times: TimestampMicrosecondArray = output_column(&output, "ts");
    assert_eq!(
        event_times.iter().flatten().collect::<Vec<_>>(),
        vec![10, 10, 20, 20, 20, 30]
    );
    let symbols: StringArray = output_column(&output, "symbol");
    assert_eq!(
        symbols.iter().flatten().collect::<Vec<_>>(),
        vec!["a", "b", "a", "b", "b", "a"]
    );
    let sequences: UInt64Array = output_column(&output, "sequence");
    assert_eq!(
        sequences.iter().flatten().collect::<Vec<_>>(),
        vec![1, 1, 2, 1, 2, 3]
    );
    let lags: Float64Array = output_column(&output, "price_lag_1");
    assert_eq!(
        lags.iter().collect::<Vec<_>>(),
        vec![None, None, Some(1.0), Some(50.0), Some(100.0), Some(2.0)]
    );
    let deltas: Int64Array = output_column(&output, "volume_delta_2");
    assert_eq!(
        deltas.iter().collect::<Vec<_>>(),
        vec![None, None, None, None, Some(1500), Some(20)]
    );
}

#[tokio::test]
async fn batch_duplicate_row_identity_is_a_data_error() {
    let error = execute(input_batch(
        vec![10, 10],
        vec!["a", "a"],
        vec![1, 1],
        vec![Some(1.0), Some(2.0)],
        vec![Some(10), Some(20)],
    ))
    .await
    .unwrap_err();
    assert!(
        matches!(
            error,
            calc_flow::CalcFlowError::Operator { ref node_id, .. } if node_id == "rolling"
        ),
        "unexpected error: {error}"
    );
}

#[tokio::test]
async fn batch_supports_duplicate_event_times_across_and_within_entities() {
    let output = execute(input_batch(
        vec![10, 10, 10, 10],
        vec!["b", "a", "a", "b"],
        vec![1, 1, 2, 2],
        vec![Some(10.0), Some(1.0), Some(2.0), Some(20.0)],
        vec![Some(100), Some(10), Some(20), Some(200)],
    ))
    .await
    .unwrap();
    let lags: Float64Array = output_column(&output, "price_lag_1");
    assert_eq!(
        lags.iter().collect::<Vec<_>>(),
        vec![None, Some(1.0), None, Some(10.0)]
    );
}

#[tokio::test]
async fn batch_empty_input_produces_an_empty_output_with_the_derived_schema() {
    let output = execute(input_batch(vec![], vec![], vec![], vec![], vec![]))
        .await
        .unwrap();
    let table = output.table_payload().unwrap();
    assert_eq!(
        table
            .batches()
            .iter()
            .map(RecordBatch::num_rows)
            .sum::<usize>(),
        0
    );
    assert_eq!(
        table.schema().as_ref(),
        operator().output_ports()[0].schema().unwrap().as_ref()
    );
}

#[tokio::test]
async fn batch_output_carries_the_operator_metadata_identity() {
    let output = execute(input_batch(
        vec![10],
        vec!["a"],
        vec![1],
        vec![Some(1.0)],
        vec![Some(10)],
    ))
    .await
    .unwrap();
    assert_eq!(output.metadata().source(), "rolling");
}
