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

// ---------------------------------------------------------------------------
// Key-type matrix: partition and sequence keys across the supported Arrow
// total-order types, exercising every KeyValue conversion and comparison arm.
// ---------------------------------------------------------------------------

use datafusion::arrow::array::{
    BooleanArray, Date32Array, Date64Array, Float32Array, Int16Array, Int32Array, UInt8Array,
};

fn matrix_spec(partition: &str, sequence: &str) -> RollingSpec {
    serde_json::from_value(serde_json::json!({
        "configuration_version": 1,
        "state_layout_version": 1,
        "partition_by": [partition],
        "event_time": "ts",
        "sequence_by": [sequence],
        "outputs": [
            {
                "kind": "lag",
                "primitive_version": 1,
                "input": "price",
                "output": "lagged",
                "periods": 1
            },
            {
                "kind": "delta",
                "primitive_version": 1,
                "input": "volume",
                "output": "changed",
                "periods": 1
            }
        ],
        "allowed_lateness_micros": 0,
        "late_policy": {"kind": "error", "scope": "envelope"},
        "value_policy": "stateful_numeric_v1"
    }))
    .unwrap()
}

fn matrix_schema(partition: Field, sequence: Field) -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC"))),
            false,
        ),
        partition,
        sequence,
        Field::new("price", DataType::Float64, true),
        Field::new("volume", DataType::Int64, true),
    ]))
}

fn utc_timestamps(values: Vec<i64>) -> ArrayRef {
    Arc::new(TimestampMicrosecondArray::from(values).with_timezone("UTC"))
}

fn prices(values: Vec<Option<f64>>) -> ArrayRef {
    Arc::new(Float64Array::from(values))
}

fn volumes(values: Vec<Option<i64>>) -> ArrayRef {
    Arc::new(Int64Array::from(values))
}

async fn execute_matrix(
    schema: Arc<Schema>,
    partition: &str,
    sequence: &str,
    columns: Vec<ArrayRef>,
) -> Batch {
    let spec = matrix_spec(partition, sequence);
    let operator = RollingOperator::new("rolling", schema.clone(), spec).unwrap();
    let record = RecordBatch::try_new(schema, columns).unwrap();
    let plan = PipelineBuilder::new("rolling key matrix")
        .unwrap()
        .add_node("rolling", operator)
        .unwrap()
        .compile_batch(&UdfRegistry::new().snapshot())
        .unwrap();
    let outputs = plan
        .execute(
            BTreeMap::from([(
                "input".into(),
                Batch::table(vec![record], BatchMetadata::default()).unwrap(),
            )]),
            ExecutionOptions::default(),
        )
        .await
        .unwrap();
    outputs.outputs["output"].clone()
}

fn lag_column(batch: &Batch) -> Vec<Option<f64>> {
    output_column::<Float64Array>(batch, "lagged")
        .iter()
        .collect()
}

fn delta_column(batch: &Batch) -> Vec<Option<i64>> {
    output_column::<Int64Array>(batch, "changed")
        .iter()
        .collect()
}

#[tokio::test]
async fn boolean_partition_int32_sequence_orders_and_lags() {
    let output = execute_matrix(
        matrix_schema(
            Field::new("part", DataType::Boolean, false),
            Field::new("seq", DataType::Int32, false),
        ),
        "part",
        "seq",
        vec![
            utc_timestamps(vec![10, 10, 11, 10]),
            Arc::new(BooleanArray::from(vec![true, false, true, false])),
            Arc::new(Int32Array::from(vec![1, 1, 2, 2])),
            prices(vec![Some(5.0), Some(1.0), Some(6.0), Some(2.0)]),
            volumes(vec![Some(50), Some(10), Some(60), Some(20)]),
        ],
    )
    .await;
    let parts: BooleanArray = output_column(&output, "part");
    assert_eq!(
        parts.iter().flatten().collect::<Vec<_>>(),
        vec![false, false, true, true]
    );
    assert_eq!(lag_column(&output), vec![None, Some(1.0), None, Some(5.0)]);
    assert_eq!(delta_column(&output), vec![None, Some(10), None, Some(10)]);
}

#[tokio::test]
async fn float32_partition_orders_ieee_total_with_nan_entity() {
    let output = execute_matrix(
        matrix_schema(
            Field::new("part", DataType::Float32, false),
            Field::new("seq", DataType::Boolean, false),
        ),
        "part",
        "seq",
        vec![
            utc_timestamps(vec![10, 10, 10, 11, 11]),
            Arc::new(Float32Array::from(vec![
                f32::NAN,
                -1.5,
                2.5,
                f32::NAN,
                -1.5,
            ])),
            Arc::new(BooleanArray::from(vec![false, false, false, true, true])),
            prices(vec![Some(9.0), Some(1.0), Some(5.0), Some(8.0), Some(2.0)]),
            volumes(vec![Some(90), Some(10), Some(50), Some(80), Some(20)]),
        ],
    )
    .await;
    let parts: Float32Array = output_column(&output, "part");
    let ordered: Vec<f32> = parts.iter().map(|value| value.unwrap()).collect();
    assert!(ordered[0] < ordered[1]);
    assert!(ordered[2].is_nan());
    assert!(ordered[3] < ordered[4] || ordered[4].is_nan());
    assert!(ordered[4].is_nan());
    assert_eq!(
        lag_column(&output),
        vec![None, None, None, Some(1.0), Some(9.0)]
    );
}

#[tokio::test]
async fn date32_partition_string_sequence_lags() {
    let output = execute_matrix(
        matrix_schema(
            Field::new("part", DataType::Date32, false),
            Field::new("seq", DataType::Utf8, false),
        ),
        "part",
        "seq",
        vec![
            utc_timestamps(vec![10, 10, 11, 10]),
            Arc::new(Date32Array::from(vec![20, 10, 20, 10])),
            Arc::new(StringArray::from(vec!["a", "a", "b", "b"])),
            prices(vec![Some(5.0), Some(1.0), Some(6.0), Some(2.0)]),
            volumes(vec![Some(50), Some(10), Some(60), Some(20)]),
        ],
    )
    .await;
    let parts: Date32Array = output_column(&output, "part");
    assert_eq!(
        parts.iter().flatten().collect::<Vec<_>>(),
        vec![10, 10, 20, 20]
    );
    assert_eq!(lag_column(&output), vec![None, Some(1.0), None, Some(5.0)]);
}

#[tokio::test]
async fn timestamp_partition_int64_sequence_lags() {
    let output = execute_matrix(
        matrix_schema(
            Field::new(
                "part",
                DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC"))),
                false,
            ),
            Field::new("seq", DataType::Int64, false),
        ),
        "part",
        "seq",
        vec![
            utc_timestamps(vec![10, 10, 11, 10]),
            utc_timestamps(vec![200, 100, 200, 100]),
            Arc::new(Int64Array::from(vec![1, 1, 2, 2])),
            prices(vec![Some(5.0), Some(1.0), Some(6.0), Some(2.0)]),
            volumes(vec![Some(50), Some(10), Some(60), Some(20)]),
        ],
    )
    .await;
    let parts: TimestampMicrosecondArray = output_column(&output, "part");
    assert_eq!(
        parts.iter().flatten().collect::<Vec<_>>(),
        vec![100, 100, 200, 200]
    );
    assert_eq!(lag_column(&output), vec![None, Some(1.0), None, Some(5.0)]);
}

#[tokio::test]
async fn date64_partition_timestamp_sequence_lags() {
    let output = execute_matrix(
        matrix_schema(
            Field::new("part", DataType::Date64, false),
            Field::new(
                "seq",
                DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC"))),
                false,
            ),
        ),
        "part",
        "seq",
        vec![
            utc_timestamps(vec![10, 10, 11, 10]),
            Arc::new(Date64Array::from(vec![
                2_000_000, 1_000_000, 2_000_000, 1_000_000,
            ])),
            utc_timestamps(vec![200, 100, 200, 150]),
            prices(vec![Some(5.0), Some(1.0), Some(6.0), Some(2.0)]),
            volumes(vec![Some(50), Some(10), Some(60), Some(20)]),
        ],
    )
    .await;
    let parts: Date64Array = output_column(&output, "part");
    assert_eq!(
        parts.iter().flatten().collect::<Vec<_>>(),
        vec![1_000_000, 1_000_000, 2_000_000, 2_000_000]
    );
    assert_eq!(lag_column(&output), vec![None, Some(1.0), None, Some(5.0)]);
}

#[tokio::test]
async fn narrow_integer_keys_lag_and_delta() {
    let output = execute_matrix(
        matrix_schema(
            Field::new("part", DataType::Int16, false),
            Field::new("seq", DataType::UInt8, false),
        ),
        "part",
        "seq",
        vec![
            utc_timestamps(vec![10, 10, 11, 10, 12]),
            Arc::new(Int16Array::from(vec![2, 1, 2, 1, 2])),
            Arc::new(UInt8Array::from(vec![1, 1, 2, 2, 3])),
            prices(vec![Some(5.0), Some(1.0), Some(6.0), Some(2.0), Some(7.0)]),
            volumes(vec![Some(50), Some(10), Some(60), Some(20), Some(70)]),
        ],
    )
    .await;
    let parts: Int16Array = output_column(&output, "part");
    assert_eq!(
        parts.iter().flatten().collect::<Vec<_>>(),
        vec![1, 1, 2, 2, 2]
    );
    assert_eq!(
        lag_column(&output),
        vec![None, Some(1.0), None, Some(5.0), Some(6.0)]
    );
    assert_eq!(
        delta_column(&output),
        vec![None, Some(10), None, Some(10), Some(10)]
    );
}

#[tokio::test]
async fn nullable_partition_with_null_entity_keys_groups_and_orders() {
    let output = execute_matrix(
        matrix_schema(
            Field::new("part", DataType::Float64, true),
            Field::new("seq", DataType::UInt64, false),
        ),
        "part",
        "seq",
        vec![
            utc_timestamps(vec![10, 10, 11, 10, 11]),
            Arc::new(Float64Array::from(vec![
                Some(1.5),
                None,
                Some(1.5),
                None,
                None,
            ])),
            Arc::new(UInt64Array::from(vec![1, 1, 2, 2, 3])),
            prices(vec![Some(5.0), Some(1.0), Some(6.0), Some(2.0), Some(3.0)]),
            volumes(vec![Some(50), Some(10), Some(60), Some(20), Some(30)]),
        ],
    )
    .await;
    let parts: Float64Array = output_column(&output, "part");
    let ordered: Vec<Option<f64>> = parts.iter().collect();
    assert_eq!(ordered, vec![None, None, Some(1.5), None, Some(1.5)]);
    assert_eq!(
        lag_column(&output),
        vec![None, Some(1.0), None, Some(2.0), Some(5.0)]
    );
    assert_eq!(
        delta_column(&output),
        vec![None, Some(10), None, Some(10), Some(10)]
    );
}

#[tokio::test]
async fn boxed_rolling_operator_converts_into_a_node_operator() {
    let plan = PipelineBuilder::new("boxed rolling")
        .unwrap()
        .add_node("rolling", Box::new(operator()))
        .unwrap()
        .compile_batch(&UdfRegistry::new().snapshot())
        .unwrap();
    assert_eq!(plan.topological_order(), vec!["rolling"]);
}

#[tokio::test]
async fn batch_plan_rejects_a_non_null_rolling_state_restore() {
    let plan = PipelineBuilder::new("rolling batch")
        .unwrap()
        .add_node("rolling", operator())
        .unwrap()
        .compile_batch(&UdfRegistry::new().snapshot())
        .unwrap();
    let error = plan
        .restore(&BTreeMap::from([("rolling".into(), serde_json::json!(1))]))
        .await
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("stateless operator state must be null"),
        "unexpected error: {error}"
    );
}
