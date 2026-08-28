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
    BooleanArray, Date32Array, Date64Array, Float32Array, Int8Array, Int16Array, Int32Array,
    UInt8Array, UInt16Array, UInt32Array,
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

// ---------------------------------------------------------------------
// SCE-07: rolling aggregates through the batch pipeline
// ---------------------------------------------------------------------

fn aggregate_spec() -> RollingSpec {
    serde_json::from_value(serde_json::json!({
        "configuration_version": 1,
        "state_layout_version": 1,
        "partition_by": ["symbol"],
        "event_time": "ts",
        "sequence_by": ["sequence"],
        "outputs": [
            {
                "kind": "count",
                "primitive_version": 1,
                "input": "price",
                "output": "price_count_3",
                "frame": {"kind": "rows", "size": 3},
                "min_periods": 2
            },
            {
                "kind": "sum",
                "primitive_version": 1,
                "input": "price",
                "output": "price_sum_3",
                "frame": {"kind": "rows", "size": 3},
                "min_periods": 1
            },
            {
                "kind": "mean",
                "primitive_version": 1,
                "input": "price",
                "output": "price_mean_3",
                "frame": {"kind": "rows", "size": 3},
                "min_periods": 1
            },
            {
                "kind": "variance",
                "primitive_version": 1,
                "input": "price",
                "output": "price_var_3",
                "frame": {"kind": "rows", "size": 3},
                "min_periods": 2,
                "ddof": 1
            },
            {
                "kind": "stddev",
                "primitive_version": 1,
                "input": "volume",
                "output": "volume_std_2",
                "frame": {"kind": "rows", "size": 2},
                "min_periods": 2,
                "ddof": 0
            }
        ],
        "allowed_lateness_micros": 0,
        "late_policy": {"kind": "error", "scope": "envelope"},
        "value_policy": "stateful_numeric_v1"
    }))
    .unwrap()
}

async fn execute_aggregates(input: Batch) -> calc_flow::Result<Batch> {
    let plan = PipelineBuilder::new("rolling aggregates batch")
        .unwrap()
        .add_node(
            "rolling",
            RollingOperator::new("rolling", input_schema(), aggregate_spec()).unwrap(),
        )
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
async fn batch_emits_aggregate_columns_with_frozen_null_nan_min_period_rules() {
    let output = execute_aggregates(input_batch(
        vec![10, 10, 11, 12, 13],
        vec!["a", "b", "a", "a", "a"],
        vec![1, 1, 2, 3, 4],
        vec![Some(1.0), Some(10.0), None, Some(3.0), Some(f64::NAN)],
        vec![Some(8), Some(80), Some(16), Some(24), Some(32)],
    ))
    .await
    .unwrap();
    let record = output.table_payload().unwrap().batches()[0].clone();
    assert_eq!(record.num_rows(), 5);
    let floats = |name: &str| -> Vec<Option<f64>> {
        record
            .column_by_name(name)
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .iter()
            .collect()
    };
    let counts = record
        .column_by_name("price_count_3")
        .unwrap()
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap()
        .iter()
        .collect::<Vec<_>>();
    // Entity a prices in canonical row order: 1.0, null, 3.0, NaN; b: 10.0.
    // Count/variance require two valid samples; sum/mean require one.
    assert_eq!(counts, vec![None, None, None, Some(2), None]);
    assert_eq!(
        floats("price_sum_3"),
        vec![Some(1.0), Some(10.0), Some(1.0), Some(4.0), Some(3.0)]
    );
    assert_eq!(
        floats("price_mean_3"),
        vec![Some(1.0), Some(10.0), Some(1.0), Some(2.0), Some(3.0)]
    );
    assert_eq!(
        floats("price_var_3"),
        vec![None, None, None, Some(2.0), None]
    );
    assert_eq!(
        floats("volume_std_2"),
        vec![None, None, Some(4.0), Some(4.0), Some(4.0)]
    );
}

// ---------------------------------------------------------------------------
// SCE-08: duration frames, extrema, covariance, and correlation
// ---------------------------------------------------------------------------

fn duration_spec() -> RollingSpec {
    serde_json::from_value(serde_json::json!({
        "configuration_version": 1,
        "state_layout_version": 1,
        "partition_by": ["symbol"],
        "event_time": "ts",
        "sequence_by": ["sequence"],
        "outputs": [
            {
                "kind": "mean",
                "primitive_version": 1,
                "input": "price",
                "output": "price_mean_10s",
                "frame": {"kind": "duration", "micros": 10_000_000},
                "min_periods": 1
            },
            {
                "kind": "max",
                "primitive_version": 1,
                "input": "price",
                "output": "price_max_10s",
                "frame": {"kind": "duration", "micros": 10_000_000},
                "min_periods": 1
            }
        ],
        "allowed_lateness_micros": 0,
        "late_policy": {"kind": "error", "scope": "envelope"},
        "value_policy": "stateful_numeric_v1"
    }))
    .unwrap()
}

async fn execute_duration(input: Batch, spec: RollingSpec) -> calc_flow::Result<Batch> {
    let plan = PipelineBuilder::new("rolling duration batch")
        .unwrap()
        .add_node(
            "rolling",
            RollingOperator::new("rolling", input_schema(), spec).unwrap(),
        )
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
async fn batch_duration_windows_evict_on_the_open_lower_boundary() {
    // One entity, event times 0s/5s/10s/15s/20s, frame (t - 10s, t].
    let output = execute_duration(
        input_batch(
            vec![0, 5_000_000, 10_000_000, 15_000_000, 20_000_000],
            vec!["a"; 5],
            vec![1, 2, 3, 4, 5],
            vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0)],
            vec![Some(8); 5],
        ),
        duration_spec(),
    )
    .await
    .unwrap();
    let record = output.table_payload().unwrap().batches()[0].clone();
    assert_eq!(record.num_rows(), 5);
    let floats = |name: &str| -> Vec<Option<f64>> {
        record
            .column_by_name(name)
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .iter()
            .collect()
    };
    // The row at exactly t - 10s is excluded (open lower, closed upper).
    assert_eq!(
        floats("price_mean_10s"),
        vec![Some(1.0), Some(1.5), Some(2.5), Some(3.5), Some(4.5)]
    );
    assert_eq!(
        floats("price_max_10s"),
        vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0)]
    );
}

#[tokio::test]
async fn batch_duration_windows_apply_the_equal_time_sequence_rule() {
    // Two rows share event time 5s; the canonical order places the lower
    // sequence first, so the earlier-sequence window holds only itself.
    let output = execute_duration(
        input_batch(
            vec![5_000_000, 5_000_000],
            vec!["a"; 2],
            vec![2, 1],
            vec![Some(7.0), Some(9.0)],
            vec![Some(8); 2],
        ),
        duration_spec(),
    )
    .await
    .unwrap();
    let record = output.table_payload().unwrap().batches()[0].clone();
    let floats = |name: &str| -> Vec<Option<f64>> {
        record
            .column_by_name(name)
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .iter()
            .collect()
    };
    assert_eq!(floats("price_max_10s"), vec![Some(9.0), Some(9.0)]);
    assert_eq!(floats("price_mean_10s"), vec![Some(9.0), Some(8.0)]);
}

fn extrema_spec() -> RollingSpec {
    serde_json::from_value(serde_json::json!({
        "configuration_version": 1,
        "state_layout_version": 1,
        "partition_by": ["symbol"],
        "event_time": "ts",
        "sequence_by": ["sequence"],
        "outputs": [
            {
                "kind": "min",
                "primitive_version": 1,
                "input": "price",
                "output": "price_min_3",
                "frame": {"kind": "rows", "size": 3},
                "min_periods": 1
            },
            {
                "kind": "max",
                "primitive_version": 1,
                "input": "volume",
                "output": "volume_max_3",
                "frame": {"kind": "rows", "size": 3},
                "min_periods": 2
            }
        ],
        "allowed_lateness_micros": 0,
        "late_policy": {"kind": "error", "scope": "envelope"},
        "value_policy": "stateful_numeric_v1"
    }))
    .unwrap()
}

#[tokio::test]
async fn batch_extrema_preserve_input_type_and_exclude_null_nan_samples() {
    let output = execute_duration(
        input_batch(
            vec![10, 10, 11, 12, 13],
            vec!["a", "b", "a", "a", "a"],
            vec![1, 1, 2, 3, 4],
            vec![Some(1.0), Some(10.0), None, Some(3.0), Some(f64::NAN)],
            vec![Some(8), Some(80), Some(16), None, Some(32)],
        ),
        extrema_spec(),
    )
    .await
    .unwrap();
    let record = output.table_payload().unwrap().batches()[0].clone();
    let price_min = record
        .column_by_name("price_min_3")
        .unwrap()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .iter()
        .collect::<Vec<_>>();
    let volume_max = record
        .column_by_name("volume_max_3")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .iter()
        .collect::<Vec<_>>();
    // Entity a prices in order: 1.0, null, 3.0, NaN; volumes: 8, 16, null, 32.
    assert_eq!(
        price_min,
        vec![Some(1.0), Some(10.0), Some(1.0), Some(1.0), Some(3.0)]
    );
    // volume_max needs two valid samples in the row frame; entity a
    // volumes in order are 8, 16, null, 32.
    assert_eq!(volume_max, vec![None, None, Some(16), Some(16), Some(32)]);
}

fn pair_spec() -> RollingSpec {
    serde_json::from_value(serde_json::json!({
        "configuration_version": 1,
        "state_layout_version": 1,
        "partition_by": ["symbol"],
        "event_time": "ts",
        "sequence_by": ["sequence"],
        "outputs": [
            {
                "kind": "covariance",
                "primitive_version": 1,
                "left": "price",
                "right": "volume",
                "output": "cov_4",
                "frame": {"kind": "rows", "size": 4},
                "min_periods": 2,
                "ddof": 1
            },
            {
                "kind": "correlation",
                "primitive_version": 1,
                "left": "price",
                "right": "volume",
                "output": "corr_4",
                "frame": {"kind": "rows", "size": 4},
                "min_periods": 2,
                "ddof": 1
            }
        ],
        "allowed_lateness_micros": 0,
        "late_policy": {"kind": "error", "scope": "envelope"},
        "value_policy": "stateful_numeric_v1"
    }))
    .unwrap()
}

#[tokio::test]
async fn batch_covariance_and_correlation_match_the_exact_reference() {
    let output = execute_duration(
        input_batch(
            vec![10, 11, 12, 13, 14],
            vec!["a"; 5],
            vec![1, 2, 3, 4, 5],
            vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0)],
            vec![Some(2), Some(4), Some(6), Some(8), Some(10)],
        ),
        pair_spec(),
    )
    .await
    .unwrap();
    let record = output.table_payload().unwrap().batches()[0].clone();
    let floats = |name: &str| -> Vec<Option<f64>> {
        record
            .column_by_name(name)
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .iter()
            .collect()
    };
    // Perfectly linear pair: covariance tracks the 4-row window, correlation 1.
    assert_eq!(
        floats("cov_4"),
        vec![
            None,
            Some(1.0),
            Some(2.0),
            Some(10.0 / 3.0),
            Some(10.0 / 3.0)
        ]
    );
    for value in floats("corr_4").into_iter().flatten() {
        assert!((value - 1.0).abs() < 1e-10, "corr {value}");
    }
}

#[tokio::test]
async fn batch_pair_outputs_count_only_pairwise_valid_positions() {
    let mut spec = pair_spec();
    for output in &mut spec.outputs {
        if let calc_flow::RollingOutputSpec::Covariance { frame, .. }
        | calc_flow::RollingOutputSpec::Correlation { frame, .. } = output
        {
            *frame = calc_flow::RollingFrameSpec::Rows { size: 5 };
        }
    }
    let output = execute_duration(
        input_batch(
            vec![10, 11, 12, 13, 14],
            vec!["a"; 5],
            vec![1, 2, 3, 4, 5],
            vec![Some(1.0), None, Some(3.0), Some(f64::NAN), Some(5.0)],
            vec![Some(10), Some(8), None, Some(4), Some(2)],
        ),
        spec,
    )
    .await
    .unwrap();
    let record = output.table_payload().unwrap().batches()[0].clone();
    let floats = |name: &str| -> Vec<Option<f64>> {
        record
            .column_by_name(name)
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .iter()
            .collect()
    };
    // Only rows 0 and 4 hold a pairwise-valid pair in the 5-row frame:
    // x = [1, 5], y = [10, 2] gives covariance -16 and correlation -1
    // (within the D13 pair tolerance for the West readout).
    assert_eq!(floats("cov_4"), vec![None, None, None, None, Some(-16.0)]);
    for correlation in floats("corr_4").into_iter().flatten() {
        assert!((correlation + 1.0).abs() < 1e-10, "corr {correlation}");
    }
}

#[tokio::test]
async fn batch_correlation_is_null_for_zero_variance_and_nan_for_infinity() {
    let spec = serde_json::from_value(serde_json::json!({
        "configuration_version": 1,
        "state_layout_version": 1,
        "partition_by": ["symbol"],
        "event_time": "ts",
        "sequence_by": ["sequence"],
        "outputs": [
            {
                "kind": "covariance",
                "primitive_version": 1,
                "left": "price",
                "right": "volume",
                "output": "cov",
                "frame": {"kind": "rows", "size": 3},
                "min_periods": 1,
                "ddof": 0
            },
            {
                "kind": "correlation",
                "primitive_version": 1,
                "left": "price",
                "right": "volume",
                "output": "corr",
                "frame": {"kind": "rows", "size": 3},
                "min_periods": 1,
                "ddof": 0
            }
        ],
        "allowed_lateness_micros": 0,
        "late_policy": {"kind": "error", "scope": "envelope"},
        "value_policy": "stateful_numeric_v1"
    }))
    .unwrap();
    // Row 0: one pair, ddof 0 — covariance 0, correlation null (zero
    // variance). Rows 1-2: constant volume keeps correlation null and
    // covariance 0. Rows 3-4 hold +inf on the left: any infinity in either
    // operand makes the pair statistic NaN, never null.
    let output = execute_duration(
        input_batch(
            vec![10, 11, 12, 13, 14],
            vec!["a"; 5],
            vec![1, 2, 3, 4, 5],
            vec![
                Some(1.0),
                Some(2.0),
                Some(3.0),
                Some(f64::INFINITY),
                Some(5.0),
            ],
            vec![Some(7); 5],
        ),
        spec,
    )
    .await
    .unwrap();
    let record = output.table_payload().unwrap().batches()[0].clone();
    let floats = |name: &str| -> Vec<Option<f64>> {
        record
            .column_by_name(name)
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .iter()
            .collect()
    };
    let cov = floats("cov");
    let corr = floats("corr");
    assert_eq!(cov[0], Some(0.0));
    assert_eq!(corr[0], None);
    assert_eq!(cov[1], Some(0.0));
    assert_eq!(corr[1], None);
    assert_eq!(corr[2], None);
    assert!(cov[3].unwrap().is_nan(), "infinity covariance {cov:?}");
    assert!(corr[3].unwrap().is_nan(), "infinity correlation {corr:?}");
    assert!(cov[4].unwrap().is_nan(), "infinity covariance {cov:?}");
    assert!(corr[4].unwrap().is_nan(), "infinity correlation {corr:?}");
}

#[tokio::test]
async fn batch_mixed_lag_and_duration_outputs_share_one_operator() {
    let spec = serde_json::from_value(serde_json::json!({
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
                "output": "price_lag_2",
                "periods": 2
            },
            {
                "kind": "mean",
                "primitive_version": 1,
                "input": "price",
                "output": "price_mean_10s",
                "frame": {"kind": "duration", "micros": 10_000_000},
                "min_periods": 1
            }
        ],
        "allowed_lateness_micros": 0,
        "late_policy": {"kind": "error", "scope": "envelope"},
        "value_policy": "stateful_numeric_v1"
    }))
    .unwrap();
    let output = execute_duration(
        input_batch(
            vec![0, 5_000_000, 10_000_000, 15_000_000, 20_000_000],
            vec!["a"; 5],
            vec![1, 2, 3, 4, 5],
            vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0)],
            vec![Some(8); 5],
        ),
        spec,
    )
    .await
    .unwrap();
    let record = output.table_payload().unwrap().batches()[0].clone();
    let lag = record
        .column_by_name("price_lag_2")
        .unwrap()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .iter()
        .collect::<Vec<_>>();
    let mean = record
        .column_by_name("price_mean_10s")
        .unwrap()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .iter()
        .collect::<Vec<_>>();
    // Duration eviction must not shorten the row-based lag history.
    assert_eq!(lag, vec![None, None, Some(1.0), Some(2.0), Some(3.0)]);
    assert_eq!(
        mean,
        vec![Some(1.0), Some(1.5), Some(2.5), Some(3.5), Some(4.5)]
    );
}

async fn execute_with(schema: Arc<Schema>, spec: RollingSpec, columns: Vec<ArrayRef>) -> Batch {
    let operator = RollingOperator::new("rolling", schema.clone(), spec).unwrap();
    let record = RecordBatch::try_new(schema, columns).unwrap();
    let plan = PipelineBuilder::new("rolling coverage")
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

#[tokio::test]
async fn batch_pair_statistics_classify_y_side_infinities_and_recover_after_slide_out() {
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC"))),
            false,
        ),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("sequence", DataType::UInt64, false),
        Field::new("price", DataType::Float64, true),
        Field::new("rate", DataType::Float64, true),
    ]));
    let spec = serde_json::from_value(serde_json::json!({
        "configuration_version": 1,
        "state_layout_version": 1,
        "partition_by": ["symbol"],
        "event_time": "ts",
        "sequence_by": ["sequence"],
        "outputs": [
            {
                "kind": "covariance",
                "primitive_version": 1,
                "left": "price",
                "right": "rate",
                "output": "cov",
                "frame": {"kind": "rows", "size": 3},
                "min_periods": 1,
                "ddof": 1
            },
            {
                "kind": "correlation",
                "primitive_version": 1,
                "left": "price",
                "right": "rate",
                "output": "corr",
                "frame": {"kind": "rows", "size": 3},
                "min_periods": 1,
                "ddof": 1
            }
        ],
        "allowed_lateness_micros": 0,
        "late_policy": {"kind": "error", "scope": "envelope"},
        "value_policy": "stateful_numeric_v1"
    }))
    .unwrap();
    // The right operand carries +inf and then -inf; both must classify the
    // pair statistics as NaN while inside the window and vanish cleanly once
    // the frame slides past them.
    let output = execute_with(
        schema,
        spec,
        vec![
            Arc::new(
                TimestampMicrosecondArray::from(vec![10, 11, 12, 13, 14, 15, 16, 17, 18])
                    .with_timezone("UTC"),
            ),
            Arc::new(StringArray::from(vec!["a"; 9])),
            Arc::new(UInt64Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9])),
            Arc::new(Float64Array::from(vec![
                Some(1.0),
                Some(2.0),
                Some(3.0),
                Some(4.0),
                Some(5.0),
                Some(6.0),
                Some(7.0),
                Some(8.0),
                Some(9.0),
            ])),
            Arc::new(Float64Array::from(vec![
                Some(1.0),
                Some(f64::INFINITY),
                Some(5.0),
                Some(6.0),
                Some(7.0),
                Some(f64::NEG_INFINITY),
                Some(9.0),
                Some(10.0),
                Some(11.0),
            ])),
        ],
    )
    .await;
    let cov: Vec<Option<f64>> = output_column::<Float64Array>(&output, "cov")
        .iter()
        .collect();
    let corr: Vec<Option<f64>> = output_column::<Float64Array>(&output, "corr")
        .iter()
        .collect();
    // Row 0 has one pairwise sample: the ddof=1 divisor is zero, so both
    // statistics stay null.
    assert_eq!(cov[0], None);
    assert_eq!(corr[0], None);
    // Rows 1-3 keep +inf inside the three-row frame; rows 5-7 keep -inf.
    for index in [1, 2, 3, 5, 6, 7] {
        assert!(cov[index].unwrap().is_nan(), "cov row {index}: {cov:?}");
        assert!(corr[index].unwrap().is_nan(), "corr row {index}: {corr:?}");
    }
    // Rows 4 and 8 slid past each infinity and read finite again — (3, 4, 5)
    // and (7, 8, 9) against equally spaced right operands are perfect lines,
    // so the window recovers with no NaN stickiness.
    for index in [4, 8] {
        assert_eq!(cov[index], Some(1.0), "cov row {index}: {cov:?}");
        assert!(
            (corr[index].unwrap() - 1.0).abs() < 1e-10,
            "corr row {index}: {corr:?}"
        );
    }
}

fn extrema_matrix() -> (Arc<Schema>, Vec<ArrayRef>) {
    (
        Arc::new(Schema::new(vec![
            Field::new(
                "ts",
                DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC"))),
                false,
            ),
            Field::new("symbol", DataType::Utf8, false),
            Field::new("sequence", DataType::UInt64, false),
            Field::new("flag", DataType::Boolean, true),
            Field::new("tiny", DataType::Int8, true),
            Field::new("small", DataType::Int16, true),
            Field::new("medium", DataType::Int32, true),
            Field::new("big", DataType::Int64, true),
            Field::new("u08", DataType::UInt8, true),
            Field::new("u16", DataType::UInt16, true),
            Field::new("u32", DataType::UInt32, true),
            Field::new("u64", DataType::UInt64, true),
            Field::new("f32x", DataType::Float32, true),
            Field::new("word", DataType::Utf8, true),
            Field::new("day", DataType::Date32, true),
            Field::new("instant", DataType::Date64, true),
        ])),
        vec![
            Arc::new(TimestampMicrosecondArray::from(vec![10, 11, 12]).with_timezone("UTC")),
            Arc::new(StringArray::from(vec!["a"; 3])),
            Arc::new(UInt64Array::from(vec![1, 2, 3])),
            Arc::new(BooleanArray::from(vec![
                Some(true),
                Some(false),
                Some(true),
            ])),
            Arc::new(Int8Array::from(vec![Some(3), Some(-7), Some(5)])),
            Arc::new(Int16Array::from(vec![Some(300), Some(-700), Some(500)])),
            Arc::new(Int32Array::from(vec![Some(70_000), Some(-70_000), Some(5)])),
            Arc::new(Int64Array::from(vec![
                Some(7_000_000_000),
                Some(-7_000_000_000),
                Some(5),
            ])),
            Arc::new(UInt8Array::from(vec![Some(3), Some(200), Some(5)])),
            Arc::new(UInt16Array::from(vec![Some(300), Some(60_000), Some(5)])),
            Arc::new(UInt32Array::from(vec![
                Some(3_000_000_000),
                Some(200),
                Some(5),
            ])),
            Arc::new(UInt64Array::from(vec![
                Some(7_000_000_000),
                Some(200),
                Some(5),
            ])),
            Arc::new(Float32Array::from(vec![Some(1.5), Some(-2.5), Some(0.5)])),
            Arc::new(StringArray::from(vec![
                Some("pear"),
                Some("apple"),
                Some("fig"),
            ])),
            Arc::new(Date32Array::from(vec![Some(10), Some(5), Some(7)])),
            Arc::new(Date64Array::from(vec![
                Some(60_000),
                Some(1_000),
                Some(7_000),
            ])),
        ],
    )
}

fn extrema_matrix_spec() -> RollingSpec {
    let inputs = [
        "flag", "tiny", "small", "medium", "big", "u08", "u16", "u32", "u64", "f32x", "word",
        "day", "instant",
    ];
    let declarations: Vec<_> = inputs
        .iter()
        .map(|input| {
            serde_json::json!({
                "kind": "min",
                "primitive_version": 1,
                "input": input,
                "output": format!("{input}_min"),
                "frame": {"kind": "rows", "size": 3},
                "min_periods": 1
            })
        })
        .chain([serde_json::json!({
            "kind": "max",
            "primitive_version": 1,
            "input": "flag",
            "output": "flag_max",
            "frame": {"kind": "rows", "size": 3},
            "min_periods": 1
        })])
        .collect();
    serde_json::from_value(serde_json::json!({
        "configuration_version": 1,
        "state_layout_version": 1,
        "partition_by": ["symbol"],
        "event_time": "ts",
        "sequence_by": ["sequence"],
        "outputs": declarations,
        "allowed_lateness_micros": 0,
        "late_policy": {"kind": "error", "scope": "envelope"},
        "value_policy": "stateful_numeric_v1"
    }))
    .unwrap()
}

async fn extrema_matrix_output() -> Batch {
    let (schema, columns) = extrema_matrix();
    execute_with(schema, extrema_matrix_spec(), columns).await
}

#[tokio::test]
async fn batch_extrema_order_integer_and_unsigned_families() {
    let output = extrema_matrix_output().await;
    // The growing rows(3) frame sees one, two, then all three samples, so
    // every family's minimum is the smallest value among the rows it has
    // observed under the frozen total order.
    let minima: Vec<Option<i8>> = output_column::<Int8Array>(&output, "tiny_min")
        .iter()
        .collect();
    assert_eq!(minima, vec![Some(3), Some(-7), Some(-7)]);
    let minima: Vec<Option<i16>> = output_column::<Int16Array>(&output, "small_min")
        .iter()
        .collect();
    assert_eq!(minima, vec![Some(300), Some(-700), Some(-700)]);
    let minima: Vec<Option<i32>> = output_column::<Int32Array>(&output, "medium_min")
        .iter()
        .collect();
    assert_eq!(minima, vec![Some(70_000), Some(-70_000), Some(-70_000)]);
    let minima: Vec<Option<i64>> = output_column::<Int64Array>(&output, "big_min")
        .iter()
        .collect();
    assert_eq!(
        minima,
        vec![
            Some(7_000_000_000),
            Some(-7_000_000_000),
            Some(-7_000_000_000)
        ]
    );
    let minima: Vec<Option<u8>> = output_column::<UInt8Array>(&output, "u08_min")
        .iter()
        .collect();
    assert_eq!(minima, vec![Some(3), Some(3), Some(3)]);
    let minima: Vec<Option<u16>> = output_column::<UInt16Array>(&output, "u16_min")
        .iter()
        .collect();
    assert_eq!(minima, vec![Some(300), Some(300), Some(5)]);
    let minima: Vec<Option<u32>> = output_column::<UInt32Array>(&output, "u32_min")
        .iter()
        .collect();
    assert_eq!(minima, vec![Some(3_000_000_000), Some(200), Some(5)]);
    let minima: Vec<Option<u64>> = output_column::<UInt64Array>(&output, "u64_min")
        .iter()
        .collect();
    assert_eq!(minima, vec![Some(7_000_000_000), Some(200), Some(5)]);
    let minima: Vec<Option<i32>> = output_column::<Date32Array>(&output, "day_min")
        .iter()
        .collect();
    assert_eq!(minima, vec![Some(10), Some(5), Some(5)]);
    let minima: Vec<Option<i64>> = output_column::<Date64Array>(&output, "instant_min")
        .iter()
        .collect();
    assert_eq!(minima, vec![Some(60_000), Some(1_000), Some(1_000)]);
}

#[tokio::test]
async fn batch_extrema_order_boolean_float_and_string_families() {
    let output = extrema_matrix_output().await;
    let minima: Vec<Option<bool>> = output_column::<BooleanArray>(&output, "flag_min")
        .iter()
        .collect();
    assert_eq!(minima, vec![Some(true), Some(false), Some(false)]);
    let maxima: Vec<Option<bool>> = output_column::<BooleanArray>(&output, "flag_max")
        .iter()
        .collect();
    assert_eq!(maxima, vec![Some(true), Some(true), Some(true)]);
    let minima: Vec<Option<f32>> = output_column::<Float32Array>(&output, "f32x_min")
        .iter()
        .collect();
    assert_eq!(minima, vec![Some(1.5), Some(-2.5), Some(-2.5)]);
    let words = output_column::<StringArray>(&output, "word_min");
    let minima: Vec<Option<&str>> = words.iter().collect();
    assert_eq!(minima, vec![Some("pear"), Some("apple"), Some("apple")]);
}
