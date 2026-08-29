//! Batch cross-section semantics: complete-group rank, percentile, z-score,
//! and demean over exact-time and fixed-bucket groups (SCE-00 D6, SCE-09).

use std::{collections::BTreeMap, sync::Arc};

use calc_flow::{
    Batch, BatchMetadata, CrossSectionOperator, CrossSectionOutputSpec, CrossSectionSpec,
    ExecutionOptions, OperatorMetadata, PipelineBuilder, UdfRegistry,
};
use datafusion::arrow::{
    array::{
        Array, ArrayRef, Float32Array, Float64Array, Int64Array, StringArray,
        TimestampMicrosecondArray, UInt64Array,
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
        Field::new("industry", DataType::Utf8, true),
        Field::new("sequence", DataType::UInt64, false),
        Field::new("momentum_20", DataType::Float64, true),
    ]))
}

type InputRow = (i64, &'static str, Option<&'static str>, u64, Option<f64>);

fn input_batch(rows: &[InputRow]) -> Batch {
    let record = RecordBatch::try_new(
        input_schema(),
        vec![
            Arc::new(
                TimestampMicrosecondArray::from(rows.iter().map(|row| row.0).collect::<Vec<_>>())
                    .with_timezone("UTC"),
            ) as ArrayRef,
            Arc::new(StringArray::from(
                rows.iter().map(|row| row.1).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                rows.iter().map(|row| row.2).collect::<Vec<_>>(),
            )),
            Arc::new(UInt64Array::from(
                rows.iter().map(|row| row.3).collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                rows.iter().map(|row| row.4).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

fn spec_value(
    grouping: &serde_json::Value,
    outputs: &[serde_json::Value],
    allowed_lateness_micros: u64,
) -> CrossSectionSpec {
    serde_json::from_value(serde_json::json!({
        "configuration_version": 1,
        "state_layout_version": 1,
        "event_time": "ts",
        "entity_by": ["symbol"],
        "partition_by": ["industry"],
        "sequence_by": ["sequence"],
        "grouping": grouping,
        "outputs": outputs,
        "allowed_lateness_micros": allowed_lateness_micros,
        "late_policy": {"kind": "error", "scope": "envelope"},
        "value_policy": "nan_exclude_preserve_v1"
    }))
    .unwrap()
}

fn rank_percentile_spec(tie: &str, placement: &str, direction: &str) -> CrossSectionSpec {
    spec_value(
        &serde_json::json!({"kind": "exact_time"}),
        &[
            serde_json::json!({
                "kind": "rank",
                "primitive_version": 1,
                "input": "momentum_20",
                "output": "momentum_rank",
                "direction": direction,
                "tie_method": tie,
                "null_placement": placement,
                "min_samples": 1
            }),
            serde_json::json!({
                "kind": "percentile",
                "primitive_version": 1,
                "input": "momentum_20",
                "output": "momentum_pct",
                "direction": direction,
                "tie_method": tie,
                "null_placement": placement,
                "min_samples": 1
            }),
        ],
        0,
    )
}

fn integer_rank_percentile_spec() -> CrossSectionSpec {
    let outputs = ["signed_measure", "unsigned_measure"]
        .into_iter()
        .flat_map(|input| {
            [
                serde_json::json!({
                    "kind": "rank",
                    "primitive_version": 1,
                    "input": input,
                    "output": format!("{input}_rank"),
                    "direction": "ascending",
                    "tie_method": "average",
                    "null_placement": "exclude",
                    "min_samples": 1
                }),
                serde_json::json!({
                    "kind": "percentile",
                    "primitive_version": 1,
                    "input": input,
                    "output": format!("{input}_pct"),
                    "direction": "ascending",
                    "tie_method": "average",
                    "null_placement": "exclude",
                    "min_samples": 1
                }),
                serde_json::json!({
                    "kind": "top",
                    "primitive_version": 1,
                    "input": input,
                    "output": format!("{input}_top"),
                    "count": 1,
                    "include_ties": false,
                    "min_samples": 1
                }),
                serde_json::json!({
                    "kind": "bottom",
                    "primitive_version": 1,
                    "input": input,
                    "output": format!("{input}_bottom"),
                    "count": 1,
                    "include_ties": false,
                    "min_samples": 1
                }),
            ]
        })
        .collect::<Vec<_>>();
    spec_value(&serde_json::json!({"kind": "exact_time"}), &outputs, 0)
}

fn statistics_spec(ddof: u8, min_samples: u64) -> CrossSectionSpec {
    spec_value(
        &serde_json::json!({"kind": "exact_time"}),
        &[
            serde_json::json!({
                "kind": "demean",
                "primitive_version": 1,
                "input": "momentum_20",
                "output": "momentum_demean",
                "min_samples": min_samples
            }),
            serde_json::json!({
                "kind": "zscore",
                "primitive_version": 1,
                "input": "momentum_20",
                "output": "momentum_z",
                "min_samples": min_samples,
                "ddof": ddof
            }),
        ],
        0,
    )
}

fn winsorize_spec(lower: f64, upper: f64, min_samples: u64) -> CrossSectionSpec {
    spec_value(
        &serde_json::json!({"kind": "exact_time"}),
        &[serde_json::json!({
            "kind": "winsorize",
            "primitive_version": 1,
            "input": "momentum_20",
            "output": "momentum_winsorized",
            "min_samples": min_samples,
            "lower": lower,
            "upper": upper
        })],
        0,
    )
}

fn grouped_features_spec() -> CrossSectionSpec {
    grouped_features_spec_for(&serde_json::json!({"kind": "exact_time"}))
}

fn grouped_features_spec_for(grouping: &serde_json::Value) -> CrossSectionSpec {
    spec_value(
        grouping,
        &[
            serde_json::json!({
                "kind": "top",
                "primitive_version": 1,
                "input": "momentum_20",
                "output": "is_top",
                "count": 2,
                "include_ties": true,
                "min_samples": 1
            }),
            serde_json::json!({
                "kind": "bottom",
                "primitive_version": 1,
                "input": "momentum_20",
                "output": "is_bottom",
                "count": 2,
                "include_ties": false,
                "min_samples": 1
            }),
            serde_json::json!({
                "kind": "mean_fill",
                "primitive_version": 1,
                "input": "momentum_20",
                "output": "momentum_filled",
                "min_samples": 1
            }),
            serde_json::json!({
                "kind": "winsorize",
                "primitive_version": 1,
                "input": "momentum_20",
                "output": "momentum_winsorized",
                "min_samples": 1,
                "lower": 0.25,
                "upper": 0.75
            }),
        ],
        0,
    )
}

fn operator(schema: Arc<Schema>, spec: CrossSectionSpec) -> CrossSectionOperator {
    CrossSectionOperator::new("cross_section", schema, spec).unwrap()
}

async fn execute_with_schema(
    schema: Arc<Schema>,
    spec: CrossSectionSpec,
    input: Batch,
) -> calc_flow::Result<Batch> {
    let plan = PipelineBuilder::new("cross section batch")
        .unwrap()
        .add_node("cross_section", operator(schema, spec))
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

async fn execute(spec: CrossSectionSpec, input: Batch) -> calc_flow::Result<Batch> {
    execute_with_schema(input_schema(), spec, input).await
}

fn string_column(batch: &Batch, name: &str) -> Vec<String> {
    let table = batch.table_payload().unwrap();
    let mut values = Vec::new();
    for record in table.batches() {
        let array = record
            .column_by_name(name)
            .unwrap_or_else(|| panic!("missing output column {name}"))
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap_or_else(|| panic!("output column {name} is not a string"));
        for index in 0..array.len() {
            values.push(array.value(index).to_owned());
        }
    }
    values
}

fn event_time_column(batch: &Batch, name: &str) -> Vec<i64> {
    let table = batch.table_payload().unwrap();
    let mut values = Vec::new();
    for record in table.batches() {
        let array = record
            .column_by_name(name)
            .unwrap_or_else(|| panic!("missing output column {name}"))
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap_or_else(|| panic!("output column {name} is not a timestamp"));
        for index in 0..array.len() {
            values.push(array.value(index));
        }
    }
    values
}

fn u64_column(batch: &Batch, name: &str) -> Vec<u64> {
    let table = batch.table_payload().unwrap();
    let mut values = Vec::new();
    for record in table.batches() {
        let array = record
            .column_by_name(name)
            .unwrap_or_else(|| panic!("missing output column {name}"))
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap_or_else(|| panic!("output column {name} is not uint64"));
        for index in 0..array.len() {
            values.push(array.value(index));
        }
    }
    values
}

fn float_column(batch: &Batch, name: &str) -> Vec<Option<f64>> {
    let table = batch.table_payload().unwrap();
    let mut values = Vec::new();
    for record in table.batches() {
        let array = record
            .column_by_name(name)
            .unwrap_or_else(|| panic!("missing output column {name}"))
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap_or_else(|| panic!("output column {name} is not float64"));
        for index in 0..array.len() {
            values.push(if array.is_null(index) {
                None
            } else {
                Some(array.value(index))
            });
        }
    }
    values
}

fn bool_column(batch: &Batch, name: &str) -> Vec<Option<bool>> {
    let table = batch.table_payload().unwrap();
    let mut values = Vec::new();
    for record in table.batches() {
        let array = record
            .column_by_name(name)
            .unwrap_or_else(|| panic!("missing output column {name}"))
            .as_any()
            .downcast_ref::<datafusion::arrow::array::BooleanArray>()
            .unwrap_or_else(|| panic!("output column {name} is not boolean"));
        for index in 0..array.len() {
            values.push(if array.is_null(index) {
                None
            } else {
                Some(array.value(index))
            });
        }
    }
    values
}

fn float32_column(batch: &Batch, name: &str) -> Vec<Option<f32>> {
    let table = batch.table_payload().unwrap();
    let mut values = Vec::new();
    for record in table.batches() {
        let array = record
            .column_by_name(name)
            .unwrap_or_else(|| panic!("missing output column {name}"))
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap_or_else(|| panic!("output column {name} is not float32"));
        values.extend(array.iter());
    }
    values
}

/// Interleaved arrival across two partitions and two event times.
fn interleaved_rows() -> Vec<InputRow> {
    vec![
        (100, "a", Some("tech"), 1, Some(2.0)),
        (200, "a", Some("tech"), 1, Some(5.0)),
        (100, "e", Some("fin"), 1, Some(10.0)),
        (100, "c", Some("tech"), 3, Some(1.0)),
        (100, "b", Some("tech"), 2, Some(2.0)),
        (100, "d", Some("tech"), 4, None),
    ]
}

#[tokio::test]
async fn batch_ranks_ties_and_nulls_with_canonical_row_order() {
    let output = execute(
        rank_percentile_spec("average", "exclude", "ascending"),
        input_batch(&interleaved_rows()),
    )
    .await
    .unwrap();

    // Canonical output order: groups by (finality coordinate, group key),
    // rows within one group by (event_time, entity, sequence).
    let symbols = string_column(&output, "symbol");
    assert_eq!(symbols, vec!["e", "a", "b", "c", "d", "a"]);
    let event_times = event_time_column(&output, "ts");
    assert_eq!(event_times, vec![100, 100, 100, 100, 100, 200]);
    let sequences = u64_column(&output, "sequence");
    assert_eq!(sequences, vec![1, 1, 2, 3, 4, 1]);

    // Group (100, tech): 1.0 -> rank 1, 2.0/2.0 tie -> 2.5, null excluded.
    // Group (100, fin) and (200, tech) hold one ordered value each -> 1.0.
    let ranks = float_column(&output, "momentum_rank");
    assert_eq!(
        ranks,
        vec![Some(1.0), Some(2.5), Some(2.5), Some(1.0), None, Some(1.0)]
    );

    // percentile = (rank - 1) / (ordered_count - 1); a single ordered value
    // is exactly 0.5 (SCE-00 D6).
    let percentiles = float_column(&output, "momentum_pct");
    assert_eq!(
        percentiles,
        vec![
            Some(0.5),
            Some(0.75),
            Some(0.75),
            Some(0.0),
            None,
            Some(0.5)
        ]
    );
}

#[tokio::test]
async fn batch_rank_tie_min_and_max_use_class_extremes() {
    for (tie, expected) in [("min", 2.0), ("max", 3.0)] {
        let output = execute(
            rank_percentile_spec(tie, "exclude", "ascending"),
            input_batch(&interleaved_rows()),
        )
        .await
        .unwrap();
        let ranks = float_column(&output, "momentum_rank");
        assert_eq!(
            ranks[1..3],
            [Some(expected), Some(expected)],
            "tie method {tie} produced {ranks:?}"
        );
    }
}

#[tokio::test]
async fn batch_integer_order_statistics_preserve_values_above_f64_precision() {
    const ADJACENT: i64 = 9_007_199_254_740_992;
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC"))),
            false,
        ),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("industry", DataType::Utf8, true),
        Field::new("sequence", DataType::UInt64, false),
        Field::new("signed_measure", DataType::Int64, false),
        Field::new("unsigned_measure", DataType::UInt64, false),
    ]));
    let record = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(
                TimestampMicrosecondArray::from(vec![100_i64, 100, 200, 200]).with_timezone("UTC"),
            ) as ArrayRef,
            Arc::new(StringArray::from(vec!["a", "b", "c", "d"])),
            Arc::new(StringArray::from(vec!["tech", "tech", "tech", "tech"])),
            Arc::new(UInt64Array::from(vec![1_u64, 2, 3, 4])),
            Arc::new(Int64Array::from(vec![
                ADJACENT,
                ADJACENT + 1,
                ADJACENT + 1,
                ADJACENT + 1,
            ])),
            Arc::new(UInt64Array::from(vec![
                ADJACENT as u64,
                ADJACENT as u64 + 1,
                ADJACENT as u64 + 1,
                ADJACENT as u64 + 1,
            ])),
        ],
    )
    .unwrap();
    let input = Batch::table(vec![record], BatchMetadata::default()).unwrap();

    let output = execute_with_schema(schema, integer_rank_percentile_spec(), input)
        .await
        .unwrap();

    for input in ["signed_measure", "unsigned_measure"] {
        assert_eq!(
            float_column(&output, &format!("{input}_rank")),
            vec![Some(1.0), Some(2.0), Some(1.5), Some(1.5)]
        );
        assert_eq!(
            float_column(&output, &format!("{input}_pct")),
            vec![Some(0.0), Some(1.0), Some(0.5), Some(0.5)]
        );
        assert_eq!(
            bool_column(&output, &format!("{input}_top")),
            vec![Some(false), Some(true), Some(true), Some(false)]
        );
        assert_eq!(
            bool_column(&output, &format!("{input}_bottom")),
            vec![Some(true), Some(false), Some(true), Some(false)]
        );
    }
}

#[tokio::test]
async fn batch_descending_direction_reverses_the_order() {
    let output = execute(
        rank_percentile_spec("average", "exclude", "descending"),
        input_batch(&interleaved_rows()),
    )
    .await
    .unwrap();
    // Descending: 2.0/2.0 tie leads with 1.5, 1.0 follows with 3.
    let ranks = float_column(&output, "momentum_rank");
    assert_eq!(
        ranks,
        vec![Some(1.0), Some(1.5), Some(1.5), Some(3.0), None, Some(1.0)]
    );
    let percentiles = float_column(&output, "momentum_pct");
    assert_eq!(
        percentiles,
        vec![
            Some(0.5),
            Some(0.25),
            Some(0.25),
            Some(1.0),
            None,
            Some(0.5)
        ]
    );
}

#[tokio::test]
async fn batch_null_placement_first_and_last_form_one_tied_class() {
    let first = execute(
        rank_percentile_spec("average", "first", "ascending"),
        input_batch(&interleaved_rows()),
    )
    .await
    .unwrap();
    // Null class first: nulls 1..=1, 1.0 -> 2, 2.0/2.0 -> 3.5; count 4.
    assert_eq!(
        float_column(&first, "momentum_rank"),
        vec![
            Some(1.0),
            Some(3.5),
            Some(3.5),
            Some(2.0),
            Some(1.0),
            Some(1.0)
        ]
    );
    assert_eq!(
        float_column(&first, "momentum_pct"),
        vec![
            Some(0.5),
            Some(2.5 / 3.0),
            Some(2.5 / 3.0),
            Some(1.0 / 3.0),
            Some(0.0),
            Some(0.5)
        ]
    );

    let last = execute(
        rank_percentile_spec("average", "last", "ascending"),
        input_batch(&interleaved_rows()),
    )
    .await
    .unwrap();
    // Null class last: 1.0 -> 1, 2.0/2.0 -> 2.5, nulls 4; count 4.
    assert_eq!(
        float_column(&last, "momentum_rank"),
        vec![
            Some(1.0),
            Some(2.5),
            Some(2.5),
            Some(1.0),
            Some(4.0),
            Some(1.0)
        ]
    );
    assert_eq!(
        float_column(&last, "momentum_pct"),
        vec![
            Some(0.5),
            Some(0.5),
            Some(0.5),
            Some(0.0),
            Some(1.0),
            Some(0.5)
        ]
    );
}

#[tokio::test]
async fn batch_demean_and_population_zscore_use_valid_samples_only() {
    let output = execute(statistics_spec(0, 1), input_batch(&interleaved_rows()))
        .await
        .unwrap();

    // Group (100, tech): valid [2.0, 2.0, 1.0], mean 5/3, population
    // variance 2/9, std sqrt(2)/3. Null row preserves null.
    // The West-style ordered fold differs from the closed form by a few
    // ULPs; the frozen tolerance table allows 1e-10 relative (SCE-00).
    let demean = float_column(&output, "momentum_demean");
    let expected_demean = [0.0, 1.0 / 3.0, 1.0 / 3.0, -2.0 / 3.0, 0.0, 0.0];
    for (index, value) in demean.iter().enumerate() {
        match (value, expected_demean[index]) {
            (None, expected) if expected == 0.0 && index == 4 => {}
            (Some(value), expected) => assert!(
                (value - expected).abs() < 1e-10,
                "demean[{index}] = {value}, expected {expected}"
            ),
            (value, expected) => panic!("demean[{index}] = {value:?}, expected {expected}"),
        }
    }
    let zscore = float_column(&output, "momentum_z");
    let expected = [
        0.0,
        1.0 / std::f64::consts::SQRT_2,
        1.0 / std::f64::consts::SQRT_2,
        -std::f64::consts::SQRT_2,
        0.0,
        0.0,
    ];
    for (index, value) in zscore.iter().enumerate() {
        let Some(value) = value else {
            continue;
        };
        assert!(
            (value - expected[index]).abs() < 1e-10,
            "zscore[{index}] = {value}, expected {}",
            expected[index]
        );
    }
    // Single-value groups: demean is 0 and zero standard deviation makes the
    // z-score null (SCE-00 D6).
    assert_eq!(zscore[0], None);
    assert_eq!(zscore[5], None);
}

#[tokio::test]
async fn batch_sample_ddof_and_min_samples_gate_results() {
    let output = execute(statistics_spec(1, 1), input_batch(&interleaved_rows()))
        .await
        .unwrap();
    // Sample variance of [2, 2, 1] with ddof 1: 1/3, std sqrt(1/3).
    let zscore = float_column(&output, "momentum_z");
    let expected = (1.0 / 3.0) / (1.0_f64 / 3.0).sqrt();
    for value in zscore.iter().skip(1).take(2) {
        let value = value.expect("ddof=1 keeps three valid samples eligible");
        assert!(
            (value - expected).abs() < 1e-12,
            "zscore = {value}, expected {expected}"
        );
    }

    let gated = execute(statistics_spec(0, 4), input_batch(&interleaved_rows()))
        .await
        .unwrap();
    // No group reaches four valid samples: every statistical result is null.
    assert_eq!(
        float_column(&gated, "momentum_demean"),
        vec![None, None, None, None, None, None]
    );
    assert_eq!(
        float_column(&gated, "momentum_z"),
        vec![None, None, None, None, None, None]
    );
}

#[tokio::test]
async fn batch_nan_rows_are_excluded_and_produce_nan() {
    let rows = vec![
        (100, "a", Some("tech"), 1, Some(1.0)),
        (100, "b", Some("tech"), 2, Some(f64::NAN)),
        (100, "c", Some("tech"), 3, Some(3.0)),
    ];
    let output = execute(
        rank_percentile_spec("average", "exclude", "ascending"),
        input_batch(&rows),
    )
    .await
    .unwrap();
    let ranks = float_column(&output, "momentum_rank");
    assert_eq!(ranks[0], Some(1.0));
    assert!(ranks[1].is_some_and(f64::is_nan));
    assert_eq!(ranks[2], Some(2.0));

    let statistics = execute(statistics_spec(0, 1), input_batch(&rows))
        .await
        .unwrap();
    let demean = float_column(&statistics, "momentum_demean");
    assert_eq!(demean[0], Some(-1.0));
    assert!(demean[1].is_some_and(f64::is_nan));
    assert_eq!(demean[2], Some(1.0));
}

#[tokio::test]
async fn batch_winsorize_uses_type_seven_quantiles_per_partition() {
    let rows = vec![
        (100, "a", Some("tech"), 1, Some(0.0)),
        (100, "b", Some("tech"), 2, Some(10.0)),
        (100, "c", Some("tech"), 3, Some(20.0)),
        (100, "d", Some("tech"), 4, Some(30.0)),
        (100, "e", Some("tech"), 5, Some(40.0)),
        (100, "f", Some("tech"), 6, None),
        (100, "g", Some("tech"), 7, Some(f64::NAN)),
        (100, "h", Some("fin"), 8, Some(100.0)),
    ];

    let output = execute(winsorize_spec(0.25, 0.75, 1), input_batch(&rows))
        .await
        .unwrap();
    let values = float_column(&output, "momentum_winsorized");

    assert_eq!(
        &values[..6],
        &[
            Some(100.0),
            Some(10.0),
            Some(10.0),
            Some(20.0),
            Some(30.0),
            Some(30.0),
        ]
    );
    assert_eq!(values[6], None);
    assert!(values[7].is_some_and(f64::is_nan));
}

#[tokio::test]
async fn batch_winsorize_min_samples_nulls_valid_rows_but_preserves_nan() {
    let rows = vec![
        (100, "a", Some("tech"), 1, Some(1.0)),
        (100, "b", Some("tech"), 2, Some(f64::NAN)),
        (100, "c", Some("tech"), 3, None),
    ];

    let output = execute(winsorize_spec(0.0, 1.0, 2), input_batch(&rows))
        .await
        .unwrap();
    let values = float_column(&output, "momentum_winsorized");

    assert_eq!(values[0], None);
    assert!(values[1].is_some_and(f64::is_nan));
    assert_eq!(values[2], None);
}

#[tokio::test]
async fn batch_winsorize_maps_undefined_infinite_quantiles_to_nan() {
    let rows = vec![
        (100, "a", Some("tech"), 1, Some(f64::NEG_INFINITY)),
        (100, "b", Some("tech"), 2, Some(f64::INFINITY)),
    ];

    let output = execute(winsorize_spec(0.5, 0.5, 1), input_batch(&rows))
        .await
        .unwrap();

    assert!(
        float_column(&output, "momentum_winsorized")
            .iter()
            .all(|value| value.is_some_and(f64::is_nan))
    );
}

#[tokio::test]
async fn batch_grouped_feature_min_samples_gates_selection_and_fill() {
    let rows = vec![
        (100, "a", Some("tech"), 1, Some(1.0)),
        (100, "b", Some("tech"), 2, None),
        (100, "c", Some("tech"), 3, Some(f64::NAN)),
    ];
    let spec = spec_value(
        &serde_json::json!({"kind": "exact_time"}),
        &[
            serde_json::json!({
                "kind": "top",
                "primitive_version": 1,
                "input": "momentum_20",
                "output": "is_top",
                "count": 1,
                "include_ties": true,
                "min_samples": 2
            }),
            serde_json::json!({
                "kind": "mean_fill",
                "primitive_version": 1,
                "input": "momentum_20",
                "output": "filled",
                "min_samples": 2
            }),
        ],
        0,
    );

    let output = execute(spec, input_batch(&rows)).await.unwrap();

    assert_eq!(bool_column(&output, "is_top"), vec![None, None, None]);
    let filled = float_column(&output, "filled");
    assert_eq!(filled[0], Some(1.0));
    assert_eq!(filled[1], None);
    assert!(filled[2].is_some_and(f64::is_nan));
}

#[tokio::test]
async fn batch_grouped_features_share_partition_order_and_missing_value_rules() {
    let rows = vec![
        (100, "a", Some("tech"), 1, Some(0.0)),
        (100, "b", Some("tech"), 2, Some(10.0)),
        (100, "c", Some("tech"), 3, Some(20.0)),
        (100, "d", Some("tech"), 4, Some(20.0)),
        (100, "e", Some("tech"), 5, None),
        (100, "f", Some("tech"), 6, Some(f64::NAN)),
        (100, "g", Some("fin"), 7, Some(100.0)),
    ];

    let output = execute(grouped_features_spec(), input_batch(&rows))
        .await
        .unwrap();

    assert_eq!(
        string_column(&output, "symbol"),
        vec!["g", "a", "b", "c", "d", "e", "f"]
    );
    assert_eq!(
        bool_column(&output, "is_top"),
        vec![
            Some(true),
            Some(false),
            Some(false),
            Some(true),
            Some(true),
            None,
            None
        ]
    );
    assert_eq!(
        bool_column(&output, "is_bottom"),
        vec![
            Some(true),
            Some(true),
            Some(true),
            Some(false),
            Some(false),
            None,
            None
        ]
    );
    let filled = float_column(&output, "momentum_filled");
    assert_eq!(
        &filled[..6],
        &[
            Some(100.0),
            Some(0.0),
            Some(10.0),
            Some(20.0),
            Some(20.0),
            Some(12.5)
        ]
    );
    assert!(filled[6].is_some_and(f64::is_nan));
    let schema = output.table_payload().unwrap().schema();
    assert_eq!(
        schema.field_with_name("is_top").unwrap().data_type(),
        &DataType::Boolean
    );
    assert_eq!(
        schema.field_with_name("is_bottom").unwrap().data_type(),
        &DataType::Boolean
    );
    assert_eq!(
        schema
            .field_with_name("momentum_filled")
            .unwrap()
            .data_type(),
        &DataType::Float64
    );
}

#[tokio::test]
async fn batch_winsorize_and_mean_fill_preserve_float32() {
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC"))),
            false,
        ),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("industry", DataType::Utf8, true),
        Field::new("sequence", DataType::UInt64, false),
        Field::new("momentum_20", DataType::Float32, true),
    ]));
    let record = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(TimestampMicrosecondArray::from(vec![100_i64; 4]).with_timezone("UTC"))
                as ArrayRef,
            Arc::new(StringArray::from(vec!["a", "b", "c", "d"])),
            Arc::new(StringArray::from(vec!["tech"; 4])),
            Arc::new(UInt64Array::from(vec![1_u64, 2, 3, 4])),
            Arc::new(Float32Array::from(vec![
                Some(0.0_f32),
                Some(10.0),
                None,
                Some(30.0),
            ])),
        ],
    )
    .unwrap();
    let input = Batch::table(vec![record], BatchMetadata::default()).unwrap();
    let spec = spec_value(
        &serde_json::json!({"kind": "exact_time"}),
        &[
            serde_json::json!({
                "kind": "winsorize",
                "primitive_version": 1,
                "input": "momentum_20",
                "output": "winsorized",
                "min_samples": 1,
                "lower": 0.25,
                "upper": 0.75
            }),
            serde_json::json!({
                "kind": "mean_fill",
                "primitive_version": 1,
                "input": "momentum_20",
                "output": "filled",
                "min_samples": 1
            }),
        ],
        0,
    );

    let output = execute_with_schema(schema, spec, input).await.unwrap();

    assert_eq!(
        float32_column(&output, "winsorized"),
        vec![Some(5.0), Some(10.0), None, Some(20.0)]
    );
    assert_eq!(
        float32_column(&output, "filled"),
        vec![Some(0.0), Some(10.0), Some(40.0 / 3.0), Some(30.0)]
    );
}

#[tokio::test]
async fn batch_fixed_buckets_floor_divide_and_close_on_bucket_end() {
    let rows = vec![
        (100, "a", Some("tech"), 1, Some(1.0)),
        (199, "b", Some("tech"), 1, Some(2.0)),
        (200, "c", Some("tech"), 1, Some(3.0)),
        (-50, "d", Some("tech"), 1, Some(4.0)),
    ];
    let output = execute(
        spec_value(
            &serde_json::json!({"kind": "fixed_bucket", "width_micros": 100}),
            &[
                serde_json::json!({
                    "kind": "rank",
                    "primitive_version": 1,
                    "input": "momentum_20",
                    "output": "momentum_rank",
                    "direction": "ascending",
                    "tie_method": "average",
                    "null_placement": "exclude",
                    "min_samples": 1
                }),
                serde_json::json!({
                    "kind": "percentile",
                    "primitive_version": 1,
                    "input": "momentum_20",
                    "output": "momentum_pct",
                    "direction": "ascending",
                    "tie_method": "average",
                    "null_placement": "exclude",
                    "min_samples": 1
                }),
            ],
            0,
        ),
        input_batch(&rows),
    )
    .await
    .unwrap();
    // Buckets: [-100, 0) holds -50; [100, 200) holds 100 and 199;
    // [200, 300) holds 200. Group order is by bucket start.
    let symbols = string_column(&output, "symbol");
    assert_eq!(symbols, vec!["d", "a", "b", "c"]);
    let ranks = float_column(&output, "momentum_rank");
    assert_eq!(ranks, vec![Some(1.0), Some(1.0), Some(2.0), Some(1.0)]);
    let percentiles = float_column(&output, "momentum_pct");
    assert_eq!(
        percentiles,
        vec![Some(0.5), Some(0.0), Some(1.0), Some(0.5)]
    );
}

#[tokio::test]
async fn batch_grouped_features_share_fixed_bucket_boundaries() {
    let rows = vec![
        (100, "a", Some("tech"), 1, Some(0.0)),
        (150, "b", Some("tech"), 2, Some(10.0)),
        (199, "c", Some("tech"), 3, Some(10.0)),
        (199, "e", Some("tech"), 4, None),
        (200, "d", Some("tech"), 1, Some(30.0)),
    ];
    let spec = grouped_features_spec_for(
        &serde_json::json!({"kind": "fixed_bucket", "width_micros": 100}),
    );

    let output = execute(spec, input_batch(&rows)).await.unwrap();

    assert_eq!(
        string_column(&output, "symbol"),
        vec!["a", "b", "c", "e", "d"]
    );
    assert_eq!(
        bool_column(&output, "is_top"),
        vec![Some(false), Some(true), Some(true), None, Some(true)]
    );
    assert_eq!(
        bool_column(&output, "is_bottom"),
        vec![Some(true), Some(true), Some(false), None, Some(true)]
    );
    assert_eq!(
        float_column(&output, "momentum_winsorized"),
        vec![Some(5.0), Some(10.0), Some(10.0), None, Some(30.0)]
    );
    let filled = float_column(&output, "momentum_filled");
    assert_eq!(filled[0], Some(0.0));
    assert_eq!(filled[1], Some(10.0));
    assert_eq!(filled[2], Some(10.0));
    assert!(filled[3].is_some_and(|value| (value - 20.0 / 3.0).abs() < 1e-12));
    assert_eq!(filled[4], Some(30.0));
}

#[tokio::test]
async fn batch_output_schema_appends_declared_outputs() {
    let output = execute(
        rank_percentile_spec("average", "exclude", "ascending"),
        input_batch(&interleaved_rows()),
    )
    .await
    .unwrap();
    let schema = output.table_payload().unwrap().schema();
    let names: Vec<&str> = schema
        .fields()
        .iter()
        .map(|field| field.name().as_str())
        .collect();
    assert_eq!(
        names,
        vec![
            "ts",
            "symbol",
            "industry",
            "sequence",
            "momentum_20",
            "momentum_rank",
            "momentum_pct"
        ]
    );
    for name in ["momentum_rank", "momentum_pct"] {
        let field = schema.field_with_name(name).unwrap();
        assert_eq!(field.data_type(), &DataType::Float64);
        assert!(field.is_nullable());
    }
}

#[tokio::test]
async fn batch_duplicate_row_identity_fails_before_output() {
    let rows = vec![
        (100, "a", Some("tech"), 1, Some(1.0)),
        (100, "a", Some("tech"), 1, Some(2.0)),
    ];
    let error = execute(
        rank_percentile_spec("average", "exclude", "ascending"),
        input_batch(&rows),
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("duplicate row identity"));
}

#[tokio::test]
async fn batch_duplicate_identity_across_partition_groups_fails() {
    // The identity is unique per logical input regardless of which partition
    // group each row maps to (SCE-00 D4): splitting the duplicate across
    // groups must not let both rows through.
    let rows = vec![
        (100, "a", Some("tech"), 1, Some(1.0)),
        (100, "a", Some("fin"), 1, Some(2.0)),
    ];
    let error = execute(
        rank_percentile_spec("average", "exclude", "ascending"),
        input_batch(&rows),
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("duplicate row identity"));
}

#[tokio::test]
async fn batch_empty_input_emits_an_empty_schema_correct_batch() {
    let output = execute(
        rank_percentile_spec("average", "exclude", "ascending"),
        input_batch(&[]),
    )
    .await
    .unwrap();
    let table = output.table_payload().unwrap();
    assert_eq!(table.batches()[0].num_rows(), 0);
    assert_eq!(table.schema().fields().len(), 7);
}

#[tokio::test]
async fn batch_infinities_stay_numeric_and_undefined_results_are_nan() {
    let rows = vec![
        (100, "a", Some("tech"), 1, Some(f64::INFINITY)),
        (100, "b", Some("tech"), 2, Some(1.0)),
        (100, "c", Some("tech"), 3, Some(f64::NEG_INFINITY)),
    ];
    let output = execute(statistics_spec(0, 1), input_batch(&rows))
        .await
        .unwrap();
    // Both infinity signs: the mean is the undefined inf - inf (NaN), so
    // demean is NaN everywhere on valid rows; the standard deviation is NaN,
    // not null (SCE-00 D3.2).
    let demean = float_column(&output, "momentum_demean");
    for value in demean {
        assert!(value.is_some_and(f64::is_nan));
    }
    let zscore = float_column(&output, "momentum_z");
    for value in zscore {
        assert!(value.is_some_and(f64::is_nan));
    }
}

#[tokio::test]
async fn batch_one_infinity_sign_classifies_the_mean() {
    let rows = vec![
        (100, "a", Some("tech"), 1, Some(f64::INFINITY)),
        (100, "b", Some("tech"), 2, Some(1.0)),
    ];
    let output = execute(statistics_spec(0, 1), input_batch(&rows))
        .await
        .unwrap();
    let demean = float_column(&output, "momentum_demean");
    // mean = +inf: inf - inf = NaN, 1 - inf = -inf.
    assert!(demean[0].is_some_and(f64::is_nan));
    assert_eq!(demean[1], Some(f64::NEG_INFINITY));
    // Variance involves inf - inf deviations: z-score is NaN on valid rows.
    let zscore = float_column(&output, "momentum_z");
    assert!(zscore.iter().all(|value| value.is_some_and(f64::is_nan)));
}

// ---------------------------------------------------------------------
// Declaration validation
// ---------------------------------------------------------------------

#[test]
fn non_numeric_input_column_is_rejected() {
    let spec = serde_json::from_value::<CrossSectionSpec>(serde_json::json!({
        "configuration_version": 1,
        "state_layout_version": 1,
        "event_time": "ts",
        "entity_by": ["symbol"],
        "partition_by": ["industry"],
        "sequence_by": ["sequence"],
        "grouping": {"kind": "exact_time"},
        "outputs": [
            {
                "kind": "rank",
                "primitive_version": 1,
                "input": "symbol",
                "output": "symbol_rank",
                "direction": "ascending",
                "tie_method": "average",
                "null_placement": "exclude",
                "min_samples": 1
            }
        ],
        "allowed_lateness_micros": 0,
        "late_policy": {"kind": "error", "scope": "envelope"},
        "value_policy": "nan_exclude_preserve_v1"
    }))
    .unwrap();
    let error = CrossSectionOperator::new("cross_section", input_schema(), spec)
        .unwrap_err()
        .to_string();
    assert!(error.contains("does not support"), "{error}");
}

#[test]
fn output_colliding_with_an_input_field_is_rejected() {
    let spec = rank_percentile_spec("average", "exclude", "ascending");
    let colliding = serde_json::from_value::<CrossSectionSpec>(serde_json::json!({
        "configuration_version": 1,
        "state_layout_version": 1,
        "event_time": "ts",
        "entity_by": ["symbol"],
        "partition_by": ["industry"],
        "sequence_by": ["sequence"],
        "grouping": {"kind": "exact_time"},
        "outputs": [
            {
                "kind": "demean",
                "primitive_version": 1,
                "input": "momentum_20",
                "output": "symbol",
                "min_samples": 1
            }
        ],
        "allowed_lateness_micros": 0,
        "late_policy": {"kind": "error", "scope": "envelope"},
        "value_policy": "nan_exclude_preserve_v1"
    }))
    .unwrap();
    let _ = spec;
    let error = CrossSectionOperator::new("cross_section", input_schema(), colliding)
        .unwrap_err()
        .to_string();
    assert!(error.contains("collides"), "{error}");
}

#[test]
fn ordering_fields_reject_invalid_declarations() {
    let base = serde_json::json!({
        "configuration_version": 1,
        "state_layout_version": 1,
        "event_time": "ts",
        "entity_by": ["symbol"],
        "partition_by": ["industry"],
        "sequence_by": ["sequence"],
        "grouping": {"kind": "exact_time"},
        "allowed_lateness_micros": 0,
        "late_policy": {"kind": "error", "scope": "envelope"},
        "value_policy": "nan_exclude_preserve_v1"
    });

    let empty_outputs = {
        let mut document = base.clone();
        document["outputs"] = serde_json::json!([]);
        document
    };
    assert!(
        CrossSectionOperator::new(
            "cross_section",
            input_schema(),
            serde_json::from_value(empty_outputs).unwrap()
        )
        .is_err(),
        "empty outputs were accepted"
    );

    let zero_min_samples = {
        let mut document = base.clone();
        document["outputs"] = serde_json::json!([{
            "kind": "demean",
            "primitive_version": 1,
            "input": "momentum_20",
            "output": "momentum_demean",
            "min_samples": 0
        }]);
        document
    };
    assert!(
        CrossSectionOperator::new(
            "cross_section",
            input_schema(),
            serde_json::from_value(zero_min_samples).unwrap()
        )
        .is_err(),
        "min_samples of zero was accepted"
    );

    let bad_ddof = {
        let mut document = base.clone();
        document["outputs"] = serde_json::json!([{
            "kind": "zscore",
            "primitive_version": 1,
            "input": "momentum_20",
            "output": "momentum_z",
            "min_samples": 1,
            "ddof": 2
        }]);
        document
    };
    assert!(
        CrossSectionOperator::new(
            "cross_section",
            input_schema(),
            serde_json::from_value(bad_ddof).unwrap()
        )
        .is_err(),
        "ddof of two was accepted"
    );

    let empty_entity = {
        let mut document = base.clone();
        document["entity_by"] = serde_json::json!([]);
        document["outputs"] = serde_json::json!([{
            "kind": "demean",
            "primitive_version": 1,
            "input": "momentum_20",
            "output": "momentum_demean",
            "min_samples": 1
        }]);
        document
    };
    assert!(
        CrossSectionOperator::new(
            "cross_section",
            input_schema(),
            serde_json::from_value(empty_entity).unwrap()
        )
        .is_err(),
        "an empty entity key was accepted"
    );
}

#[test]
fn grouped_feature_arguments_fail_closed() {
    let mut non_finite = winsorize_spec(0.1, 0.9, 1);
    non_finite.outputs[0] = CrossSectionOutputSpec::Winsorize {
        primitive_version: 1,
        input: "momentum_20".into(),
        output: "winsorized".into(),
        min_samples: 1,
        lower: f64::NAN,
        upper: 0.9,
    };
    let error = CrossSectionOperator::new("cross_section", input_schema(), non_finite)
        .unwrap_err()
        .to_string();
    assert!(error.contains("cross_section.outputs[0].lower"), "{error}");

    let zero_count = spec_value(
        &serde_json::json!({"kind": "exact_time"}),
        &[serde_json::json!({
            "kind": "top",
            "primitive_version": 1,
            "input": "momentum_20",
            "output": "is_top",
            "count": 0,
            "include_ties": true,
            "min_samples": 1
        })],
        0,
    );
    let error = CrossSectionOperator::new("cross_section", input_schema(), zero_count)
        .unwrap_err()
        .to_string();
    assert!(error.contains("cross_section.outputs[0].count"), "{error}");
}

#[test]
fn zero_bucket_width_is_rejected() {
    let document = serde_json::json!({
        "configuration_version": 1,
        "state_layout_version": 1,
        "event_time": "ts",
        "entity_by": ["symbol"],
        "partition_by": ["industry"],
        "sequence_by": ["sequence"],
        "grouping": {"kind": "fixed_bucket", "width_micros": 0},
        "outputs": [{
            "kind": "demean",
            "primitive_version": 1,
            "input": "momentum_20",
            "output": "momentum_demean",
            "min_samples": 1
        }],
        "allowed_lateness_micros": 0,
        "late_policy": {"kind": "error", "scope": "envelope"},
        "value_policy": "nan_exclude_preserve_v1"
    });
    assert!(
        CrossSectionOperator::new(
            "cross_section",
            input_schema(),
            serde_json::from_value(document).unwrap()
        )
        .is_err(),
        "a zero bucket width was accepted"
    );
}

#[test]
fn ordering_only_fields_on_statistics_are_rejected_by_the_strict_shape() {
    for field in ["direction", "tie_method", "null_placement"] {
        let mut document = serde_json::json!({
            "configuration_version": 1,
            "state_layout_version": 1,
            "event_time": "ts",
            "entity_by": ["symbol"],
            "partition_by": ["industry"],
            "sequence_by": ["sequence"],
            "grouping": {"kind": "exact_time"},
            "outputs": [{
                "kind": "demean",
                "primitive_version": 1,
                "input": "momentum_20",
                "output": "momentum_demean",
                "min_samples": 1
            }],
            "allowed_lateness_micros": 0,
            "late_policy": {"kind": "error", "scope": "envelope"},
            "value_policy": "nan_exclude_preserve_v1"
        });
        document["outputs"][0][field] = serde_json::json!("ascending");
        assert!(
            serde_json::from_value::<CrossSectionSpec>(document).is_err(),
            "demean accepted the ordering-only field {field}"
        );
    }
}

#[test]
fn unknown_and_missing_spec_fields_are_rejected() {
    let mut document = serde_json::json!({
        "configuration_version": 1,
        "state_layout_version": 1,
        "event_time": "ts",
        "entity_by": ["symbol"],
        "partition_by": ["industry"],
        "sequence_by": ["sequence"],
        "grouping": {"kind": "exact_time"},
        "outputs": [{
            "kind": "demean",
            "primitive_version": 1,
            "input": "momentum_20",
            "output": "momentum_demean",
            "min_samples": 1
        }],
        "allowed_lateness_micros": 0,
        "late_policy": {"kind": "error", "scope": "envelope"},
        "value_policy": "nan_exclude_preserve_v1"
    });
    let valid = document.clone();
    serde_json::from_value::<CrossSectionSpec>(valid).unwrap();

    document["unexpected"] = serde_json::json!(true);
    assert!(serde_json::from_value::<CrossSectionSpec>(document.clone()).is_err());

    let mut missing = serde_json::json!({
        "configuration_version": 1,
        "state_layout_version": 1,
        "event_time": "ts",
        "entity_by": ["symbol"],
        "partition_by": ["industry"],
        "sequence_by": ["sequence"],
        "grouping": {"kind": "exact_time"},
        "outputs": [{
            "kind": "demean",
            "primitive_version": 1,
            "input": "momentum_20",
            "output": "momentum_demean",
            "min_samples": 1
        }],
        "allowed_lateness_micros": 0,
        "late_policy": {"kind": "error", "scope": "envelope"}
    });
    missing.as_object_mut().unwrap().remove("value_policy");
    assert!(serde_json::from_value::<CrossSectionSpec>(missing).is_err());
}

#[test]
fn int64_measured_columns_are_supported() {
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC"))),
            false,
        ),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("industry", DataType::Utf8, true),
        Field::new("sequence", DataType::UInt64, false),
        Field::new("volume", DataType::Int64, true),
    ]));
    let spec = serde_json::from_value::<CrossSectionSpec>(serde_json::json!({
        "configuration_version": 1,
        "state_layout_version": 1,
        "event_time": "ts",
        "entity_by": ["symbol"],
        "partition_by": ["industry"],
        "sequence_by": ["sequence"],
        "grouping": {"kind": "exact_time"},
        "outputs": [
            {"kind": "rank", "primitive_version": 1, "input": "volume", "output": "volume_rank",
             "direction": "ascending", "tie_method": "average", "null_placement": "exclude",
             "min_samples": 1},
            {"kind": "demean", "primitive_version": 1, "input": "volume", "output": "volume_demean",
             "min_samples": 1}
        ],
        "allowed_lateness_micros": 0,
        "late_policy": {"kind": "error", "scope": "envelope"},
        "value_policy": "nan_exclude_preserve_v1"
    }))
    .unwrap();
    let operator = CrossSectionOperator::new("cross_section", schema, spec).unwrap();
    let output_ports = operator.output_ports();
    let schema = output_ports[0].schema().unwrap();
    assert_eq!(
        schema.field_with_name("volume_rank").unwrap().data_type(),
        &DataType::Float64
    );
    assert_eq!(
        schema.field_with_name("volume_demean").unwrap().data_type(),
        &DataType::Float64
    );
}

#[test]
fn validate_returns_the_derived_output_schema() {
    let spec = rank_percentile_spec("average", "exclude", "ascending");
    let schema = spec.validate(input_schema().as_ref()).unwrap();
    assert_eq!(schema.fields().len(), 7);
}
