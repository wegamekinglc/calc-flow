//! Correctness-only controls: actual message geometry and production callbacks.

use std::{collections::BTreeSet, sync::Arc, time::Duration};

use super::{STAMP, job};
use calc_flow::{
    AsofJoinSide, AsofStateLimits, Batch, BatchMetadata, EdgeCollector, JoinStateLimits,
    JoinTimeBounds, StreamAsofJoinOperator, StreamAsofJoinSpec, StreamJoinOperator, StreamJoinSpec,
    StreamOperator, StreamOperatorContext,
};
use datafusion::arrow::{
    array::{Int64Array, StringArray, TimestampMicrosecondArray},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use serde_json::{Value, json};

pub(crate) async fn layout_controls() -> Value {
    let mut controls = Vec::new();
    for layout in ["single_record", "multi_record", "unordered", "wide_key"] {
        controls.push(layout_sample(layout, false).await);
        controls.push(layout_sample(layout, true).await);
    }
    json!({"schema":"calc-flow.join-layout-controls.v1","check_only":true,"controls":controls})
}

fn layout_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("account_id", DataType::Utf8, false),
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("amount", DataType::Int64, false),
    ]))
}

fn join_operator() -> StreamJoinOperator {
    let spec = StreamJoinSpec::inner(
        ["account_id"],
        ["account_id"],
        "ts",
        "ts",
        JoinTimeBounds::new(Duration::from_secs(300), Duration::from_secs(60)).unwrap(),
        JoinStateLimits::new(1_000, 1 << 24, 10_000).unwrap(),
    )
    .unwrap();
    StreamJoinOperator::new("match", layout_schema(), layout_schema(), spec).unwrap()
}

fn input(layout: &str) -> Batch {
    let keys = (0..24)
        .map(|row| {
            let key_id = if layout == "unordered" { row % 4 } else { row };
            let key = format!("B{key_id:07}");
            if layout == "wide_key" {
                format!("{key}{}", "w".repeat(1016))
            } else {
                key
            }
        })
        .collect::<Vec<_>>();
    let times = (0..24)
        .map(|row| STAMP + if layout == "unordered" { 23 - row } else { row })
        .collect::<Vec<_>>();
    let record = RecordBatch::try_new(
        layout_schema(),
        vec![
            Arc::new(StringArray::from(keys)),
            Arc::new(TimestampMicrosecondArray::from(times).with_timezone("UTC")),
            Arc::new(Int64Array::from(vec![7; 24])),
        ],
    )
    .unwrap();
    let records = if layout == "multi_record" {
        vec![record.slice(0, 7), record.slice(7, 9), record.slice(16, 8)]
    } else {
        vec![record]
    };
    Batch::table(records, BatchMetadata::default()).unwrap()
}

fn asof_operator() -> StreamAsofJoinOperator {
    let side = |prefix: &str| {
        AsofJoinSide::new(
            vec!["account_id".into()],
            "ts".into(),
            vec!["amount".into()],
            prefix.into(),
        )
        .unwrap()
    };
    let spec = StreamAsofJoinSpec::new(
        side("left"),
        side("right"),
        Duration::ZERO,
        AsofStateLimits::new(1_000, 1 << 24).unwrap(),
    )
    .unwrap();
    StreamAsofJoinOperator::new("match", layout_schema(), layout_schema(), spec).unwrap()
}

async fn layout_sample(layout: &str, asof: bool) -> Value {
    let input = input(layout);
    let originals = input.table_payload().unwrap().batches().to_vec();
    let input_before = input_values(&input);
    let timestamps = originals
        .iter()
        .flat_map(|record| {
            record
                .column(1)
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .unwrap()
                .values()
                .to_vec()
        })
        .collect::<Vec<_>>();
    let descents = timestamps
        .windows(2)
        .filter(|pair| pair[0] > pair[1])
        .count();
    let key_bytes = input_before
        .iter()
        .map(|(key, _, _)| key.len())
        .sum::<usize>();
    let max_key_bytes = input_before
        .iter()
        .map(|(key, _, _)| key.len())
        .max()
        .unwrap();
    let mut last_times = std::collections::BTreeMap::new();
    let per_key_descents = input_before
        .iter()
        .filter(|(key, time, _)| {
            last_times
                .insert(key.clone(), *time)
                .is_some_and(|before| before > *time)
        })
        .count();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut operator: Box<dyn StreamOperator> = if asof {
        Box::new(asof_operator())
    } else {
        Box::new(join_operator())
    };
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    for side in ["right", "left"] {
        operator
            .process_data(side, input.clone(), &context, &mut output)
            .await
            .unwrap();
    }
    if asof {
        operator.on_end(&context, &mut output).await.unwrap();
    }
    let output_rows = validate_output(&mut output, &input_before, asof, layout);
    assert_eq!(input.table_payload().unwrap().batches(), originals);
    assert_eq!(input_values(&input), input_before);
    drop(operator);
    calc_flow::benchmark_internals::settle_job(&job)
        .await
        .unwrap();
    json!({"operator":if asof {"asof"} else {"join"},"layout":layout,
        "data_callbacks":2,"input_rows":48,"output_rows":output_rows,
        "event_record_batch_rows":vec![originals.iter().map(RecordBatch::num_rows).collect::<Vec<_>>();2],
        "event_record_batch_count":originals.len(),"timestamp_descents_per_event":descents,
        "per_key_timestamp_descents":per_key_descents,
        "unique_keys":input_before.iter().map(|(key,_,_)|key).collect::<BTreeSet<_>>().len(),
        "key_bytes_per_event":key_bytes,"max_key_bytes":max_key_bytes,
        "validated_all_rows":true,"input_unchanged":true,"owned_cleanup_complete":true})
}

fn validate_output(
    output: &mut EdgeCollector,
    input_rows: &[(String, i64, i64)],
    asof: bool,
    layout: &str,
) -> usize {
    let mut seen = BTreeSet::new();
    let mut actual_keys = Vec::new();
    for message in output.drain("output") {
        if let Some(batch) = message.as_data() {
            for record in batch.table_payload().unwrap().batches() {
                for row in 0..record.num_rows() {
                    let left = record
                        .column(0)
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .unwrap()
                        .value(row);
                    let right = record
                        .column(3)
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .unwrap()
                        .value(row);
                    assert_eq!(left, right);
                    let time = |column| {
                        record
                            .column(column)
                            .as_any()
                            .downcast_ref::<TimestampMicrosecondArray>()
                            .unwrap()
                            .value(row)
                    };
                    assert!(seen.insert((left.to_owned(), time(1), time(4))));
                    actual_keys.push(left.to_owned());
                    for column in [2, 5] {
                        assert_eq!(
                            record
                                .column(column)
                                .as_any()
                                .downcast_ref::<Int64Array>()
                                .unwrap()
                                .value(row),
                            7
                        );
                    }
                }
            }
        }
    }
    let expected_keys = input_rows
        .iter()
        .map(|(key, _, _)| key.clone())
        .collect::<Vec<_>>();
    let expected_pairs = expected_keys
        .iter()
        .zip(input_rows.iter().map(|(_, time, _)| time))
        .flat_map(|(left, left_time)| {
            expected_keys
                .iter()
                .zip(input_rows.iter().map(|(_, time, _)| time))
                .filter(move |(right, right_time)| {
                    left == *right && (!asof || left_time == *right_time)
                })
                .map(move |(_, right_time)| (left.clone(), *left_time, *right_time))
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(seen, expected_pairs);
    if !asof && layout != "unordered" {
        assert_eq!(actual_keys, expected_keys);
    }
    seen.len()
}

fn input_values(input: &Batch) -> Vec<(String, i64, i64)> {
    input
        .table_payload()
        .unwrap()
        .batches()
        .iter()
        .flat_map(|record| {
            let keys = record
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let times = record
                .column(1)
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .unwrap();
            let amounts = record
                .column(2)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            (0..record.num_rows())
                .map(|row| {
                    (
                        keys.value(row).to_owned(),
                        times.value(row),
                        amounts.value(row),
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect()
}
