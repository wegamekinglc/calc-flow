use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use calc_flow::benchmark_internals::PhaseClock;
use calc_flow::{
    Batch, BatchMetadata, CancellationToken, EdgeCollector, Epoch, JoinStateLimits, JoinTimeBounds,
    JsonMap, OperatorMetadata, OperatorStateSnapshot, StreamJobContext, StreamJoinOperator,
    StreamJoinSpec, StreamOperator, StreamOperatorContext,
};
use datafusion::arrow::array::{Int64Array, StringArray, TimestampMicrosecondArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use datafusion::arrow::record_batch::RecordBatch;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const STAMP: i64 = 100_000_000;
const BATCH_ROWS: usize = 64_000;

#[path = "native_join_layouts.rs"]
mod layouts;
pub(crate) use layouts::layout_controls;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Component {
    CaptureRestore,
    Compaction,
    DictionaryRefusalRetention,
    LayoutControls,
    #[cfg(feature = "benchmark-checkpoint-diagnostic")]
    CheckpointHistory,
}

pub(crate) fn parse_arguments(
    arguments: &[String],
) -> Result<Option<(Component, bool)>, &'static str> {
    if arguments.first().map(String::as_str) != Some("--native-component") {
        return Ok(None);
    }
    let (name, check) = component_arguments(arguments)?;
    Ok(Some((approved_component(name, check)?, check)))
}

fn component_arguments(arguments: &[String]) -> Result<(&str, bool), &'static str> {
    match arguments {
        [_, name] => Ok((name, false)),
        [_, name, check] if check == "--check" => Ok((name, true)),
        _ => Err("native component accepts one approved case and optional --check"),
    }
}

fn approved_component(name: &str, check: bool) -> Result<Component, &'static str> {
    match name {
        "join_capture_restore_1m" => Ok(Component::CaptureRestore),
        "join_compaction_1m" => Ok(Component::Compaction),
        "join_layout_controls" => check_only_component(
            Component::LayoutControls,
            check,
            "layout controls require --check",
        ),
        "join_dictionary_refusal_retention" => check_only_component(
            Component::DictionaryRefusalRetention,
            check,
            "dictionary refusal retention requires --check",
        ),
        #[cfg(feature = "benchmark-checkpoint-diagnostic")]
        "join_checkpoint_history_4m" => check_only_component(
            Component::CheckpointHistory,
            check,
            "checkpoint history requires --check",
        ),
        _ => Err("unknown native component"),
    }
}

fn check_only_component(
    component: Component,
    check: bool,
    error: &'static str,
) -> Result<Component, &'static str> {
    check.then_some(component).ok_or(error)
}

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("account_id", DataType::Utf8, false),
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new("amount", DataType::Int64, false),
    ]))
}

fn operator() -> StreamJoinOperator {
    let spec = StreamJoinSpec::inner(
        ["account_id"],
        ["account_id"],
        "ts",
        "ts",
        JoinTimeBounds::new(Duration::from_secs(300), Duration::from_secs(60)).unwrap(),
        JoinStateLimits::new(4_000_000, 4 * 1024 * 1024 * 1024, 100_000_000).unwrap(),
    )
    .unwrap();
    StreamJoinOperator::new("match", schema(), schema(), spec).unwrap()
}

fn job() -> StreamJobContext {
    StreamJobContext::new(
        1,
        "native-components",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
}

fn batch(keys: &[String]) -> Batch {
    let record = RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(StringArray::from(keys.to_vec())),
            Arc::new(TimestampMicrosecondArray::from(vec![STAMP; keys.len()])),
            Arc::new(Int64Array::from(vec![7; keys.len()])),
        ],
    )
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

fn keys(rows: usize) -> Vec<String> {
    (0..rows).map(|index| format!("B{index:07}")).collect()
}

fn retained(operator: &StreamJoinOperator) -> u64 {
    let status = operator.status();
    status.left.retained_rows + status.right.retained_rows
}

async fn feed(
    operator: &mut StreamJoinOperator,
    job: &StreamJobContext,
    side: &str,
    batch: &Batch,
) -> Vec<RecordBatch> {
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            side,
            batch.clone(),
            &StreamOperatorContext::new(job, "match", None),
            &mut output,
        )
        .await
        .unwrap();
    output
        .drain("output")
        .into_iter()
        .flat_map(|message| {
            message
                .as_data()
                .unwrap()
                .table_payload()
                .unwrap()
                .batches()
                .to_vec()
        })
        .collect()
}

fn segment_inventory(snapshot: &OperatorStateSnapshot) -> Vec<Value> {
    snapshot
        .segments
        .iter()
        .map(|(name, segment)| {
            json!({
                "name": name,
                "bytes": segment.bytes().len(),
                "sha256": hex::encode(Sha256::digest(segment.bytes())),
            })
        })
        .collect()
}

async fn validate_replay(
    operator: &mut StreamJoinOperator,
    job: &StreamJobContext,
    keys: &[String],
) -> Vec<RecordBatch> {
    let (left, right) = keys.split_at(keys.len() / 2);
    validate_contents(operator, job, left, right).await
}

async fn validate_contents(
    operator: &mut StreamJoinOperator,
    job: &StreamJobContext,
    left: &[String],
    right: &[String],
) -> Vec<RecordBatch> {
    let mut records = Vec::new();
    for (side, keys) in [("right", left), ("left", right)] {
        for chunk in keys.chunks(BATCH_ROWS) {
            records.extend(feed(operator, job, side, &batch(chunk)).await);
        }
    }
    let mut seen = BTreeSet::new();
    let mut count = 0;
    for record in &records {
        assert_eq!(record.num_columns(), 6);
        let left_keys = record
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let right_keys = record
            .column(3)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for row in 0..record.num_rows() {
            assert_eq!(left_keys.value(row), right_keys.value(row));
            assert!(seen.insert(left_keys.value(row).to_owned()));
            for column in [1, 4] {
                assert_eq!(
                    record
                        .column(column)
                        .as_any()
                        .downcast_ref::<TimestampMicrosecondArray>()
                        .unwrap()
                        .value(row),
                    STAMP
                );
            }
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
            count += 1;
        }
    }
    assert_eq!(count, left.len() + right.len());
    assert_eq!(seen, left.iter().chain(right).cloned().collect());
    records
}

pub(crate) async fn capture_restore(rows: usize) -> Value {
    capture_restore_batched(rows, BATCH_ROWS).await
}

pub(crate) async fn capture_restore_batched(rows: usize, batch_rows: usize) -> Value {
    let setup_start = Instant::now();
    assert!(rows >= 2 && rows % 2 == 0);
    let job = job();
    let keys = keys(rows);
    let (left, right) = keys.split_at(rows / 2);
    let mut original = operator();
    let mut admission_batch_rows = Vec::new();
    for (side, keys) in [("left", left), ("right", right)] {
        for chunk in keys.chunks(batch_rows) {
            assert!(
                feed(&mut original, &job, side, &batch(chunk))
                    .await
                    .is_empty()
            );
            admission_batch_rows.push(chunk.len());
        }
    }
    let resident_rows = retained(&original);
    assert_eq!(resident_rows, rows as u64);
    let context = StreamOperatorContext::new(&job, "match", None);
    let setup_seconds = setup_start.elapsed().as_secs_f64();
    let snapshot_resources = PhaseClock::start("snapshot");
    let start = Instant::now();
    original.prepare_checkpoint_async(&context).await.unwrap();
    let preparation_seconds = start.elapsed().as_secs_f64();
    let capture_start = Instant::now();
    let snapshot = original.checkpoint(Epoch::INITIAL).unwrap();
    let capture_seconds = capture_start.elapsed().as_secs_f64();
    let snapshot_seconds = start.elapsed().as_secs_f64();
    let snapshot_resources = snapshot_resources.finish();
    let mut restored = operator();
    let restore_resources = PhaseClock::start("restore");
    let restore_start = Instant::now();
    restored.restore(&snapshot).unwrap();
    let restore_seconds = restore_start.elapsed().as_secs_f64();
    let restore_resources = restore_resources.finish();
    let oracle_start = Instant::now();
    let segments = segment_inventory(&snapshot);
    let restored_live_rows = retained(&restored);
    assert_eq!(original.status(), restored.status());
    let original_output = validate_replay(&mut original, &job, &keys).await;
    let restored_output = validate_replay(&mut restored, &job, &keys).await;
    assert_eq!(original_output, restored_output);
    assert_eq!(segments, segment_inventory(&snapshot));
    let oracle_seconds = oracle_start.elapsed().as_secs_f64();
    let cleanup_start = Instant::now();
    drop(original);
    drop(restored);
    calc_flow::benchmark_internals::settle_job(&job)
        .await
        .unwrap();
    let cleanup_seconds = cleanup_start.elapsed().as_secs_f64();
    json!({
        "schema": "calc-flow.native-component.v1",
        "case": "join_capture_restore_1m",
        "checkpoint_enabled": true,
        "resident_rows": resident_rows,
        "captured_dirty_rows": resident_rows,
        "restored_live_rows": restored_live_rows,
        "snapshot_seconds": snapshot_seconds,
        "resource_phases": {"snapshot":snapshot_resources,"restore":restore_resources},
        "preparation_seconds": preparation_seconds,
        "capture_seconds": capture_seconds,
        "restore_seconds": restore_seconds,
        "snapshot_layout_version": snapshot.inline_metadata["layout_version"],
        "snapshot_bytes": snapshot.segments.values().map(|value| value.bytes().len()).sum::<usize>(),
        "snapshot_segments": segments,
        "validated_all_segments": true,
        "restore_output_equivalent": true,
        "replay_output_rows": rows,
        "admission_batch_rows": admission_batch_rows,
        "setup_seconds": setup_seconds,
        "oracle_seconds": oracle_seconds,
        "cleanup_seconds": cleanup_seconds,
        "owned_cleanup_complete": true,
    })
}

async fn armed(
    rows: usize,
    job: &StreamJobContext,
) -> (StreamJoinOperator, OperatorStateSnapshot, Vec<usize>) {
    assert!(rows >= 8 && rows % 2 == 0);
    let keys = keys(rows);
    let (left, right) = keys.split_at(rows / 2);
    let mut operator = operator();
    let bulk = left.len() - 3;
    let mut admission_batch_rows = Vec::new();
    for (side, keys) in [("left", left), ("right", right)] {
        for chunk in keys[..bulk].chunks(BATCH_ROWS) {
            assert!(
                feed(&mut operator, job, side, &batch(chunk))
                    .await
                    .is_empty()
            );
            admission_batch_rows.push(chunk.len());
        }
    }
    let mut before = operator.checkpoint(Epoch::INITIAL).unwrap();
    for index in 0..3 {
        assert!(
            feed(
                &mut operator,
                job,
                "left",
                &batch(&left[bulk + index..=bulk + index])
            )
            .await
            .is_empty()
        );
        assert!(
            feed(
                &mut operator,
                job,
                "right",
                &batch(&right[bulk + index..=bulk + index])
            )
            .await
            .is_empty()
        );
        before = operator
            .checkpoint(Epoch::new(index as u64 + 2).unwrap())
            .unwrap();
        admission_batch_rows.extend([1, 1]);
    }
    assert_eq!(retained(&operator), rows as u64);
    (operator, before, admission_batch_rows)
}

fn base_hashes(snapshot: &OperatorStateSnapshot) -> Value {
    snapshot
        .segments
        .iter()
        .filter(|(name, _)| name.ends_with("-base"))
        .map(|(name, segment)| {
            (
                name.clone(),
                Value::String(hex::encode(Sha256::digest(segment.bytes()))),
            )
        })
        .collect::<serde_json::Map<String, Value>>()
        .into()
}

fn delta_names(snapshot: &OperatorStateSnapshot) -> Vec<String> {
    snapshot
        .segments
        .keys()
        .filter(|name| name.contains("-delta-"))
        .cloned()
        .collect()
}

async fn compaction_sample(
    rows: usize,
    incoming: usize,
    steady: bool,
) -> (Value, Vec<RecordBatch>, calc_flow::StreamJoinStatus) {
    let setup_start = Instant::now();
    let job = job();
    let (mut operator, mut before, admission_batch_rows) = armed(rows, &job).await;
    let context = StreamOperatorContext::new(&job, "match", None);
    let epoch = if steady {
        operator.prepare_checkpoint_async(&context).await.unwrap();
        before = operator.checkpoint(Epoch::new(5).unwrap()).unwrap();
        6
    } else {
        5
    };
    let pre_handler_retained_rows = retained(&operator);
    let new_keys = (0..incoming)
        .map(|index| format!("X{index:07}"))
        .collect::<Vec<_>>();
    let incoming_batch = batch(&new_keys);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let setup_seconds = setup_start.elapsed().as_secs_f64();
    let clocks = measure_serial_handler(&mut operator, &context, incoming_batch, &mut output).await;
    let oracle_start = Instant::now();
    assert!(output.drain("output").is_empty());
    let status = operator.status();
    let post_handler_retained_rows = retained(&operator);
    assert_eq!(post_handler_retained_rows, (rows + incoming) as u64);
    let after = operator.checkpoint(Epoch::new(epoch).unwrap()).unwrap();
    let before_base = base_hashes(&before);
    let after_base = base_hashes(&after);
    let before_deltas = delta_names(&before);
    let after_deltas = delta_names(&after);
    if steady {
        assert_eq!(before_base, after_base);
        assert_eq!(
            before
                .inline_metadata
                .get("v2_inventory")
                .map(|v| &v["base_epoch"]),
            after
                .inline_metadata
                .get("v2_inventory")
                .map(|v| &v["base_epoch"])
        );
    } else {
        assert_ne!(before_base, after_base);
        assert!(before_deltas.len() > after_deltas.len());
    }
    let mut restored = self::operator();
    restored.restore(&after).unwrap();
    assert_eq!(restored.status(), status);
    let keys = keys(rows);
    let (left, right) = keys.split_at(rows / 2);
    let all_left = left.iter().chain(&new_keys).cloned().collect::<Vec<_>>();
    let original_output = validate_contents(&mut operator, &job, &all_left, right).await;
    let restored_output = validate_contents(&mut restored, &job, &all_left, right).await;
    assert_eq!(original_output, restored_output);
    let oracle_seconds = oracle_start.elapsed().as_secs_f64();
    let cleanup_start = Instant::now();
    drop(operator);
    drop(restored);
    calc_flow::benchmark_internals::settle_job(&job)
        .await
        .unwrap();
    let cleanup_seconds = cleanup_start.elapsed().as_secs_f64();
    let mut sample = clocks;
    sample.as_object_mut().unwrap().extend(
        json!({
            "pre_handler_retained_rows": pre_handler_retained_rows,
            "post_handler_retained_rows": post_handler_retained_rows,
            "replay_output_rows": rows + incoming,
            "admission_batch_rows": admission_batch_rows,
            "setup_seconds": setup_seconds,
            "oracle_seconds": oracle_seconds,
            "cleanup_seconds": cleanup_seconds,
            "owned_cleanup_complete": true,
            "compaction_witness": {
                "setup_dirty_epochs": 4,
                "layout_version": after.inline_metadata["layout_version"],
                "before_base_epoch": before.inline_metadata.get("v2_inventory").map(|v| &v["base_epoch"]),
                "after_base_epoch": after.inline_metadata.get("v2_inventory").map(|v| &v["base_epoch"]),
                "before_base_sha256": before_base,
                "after_base_sha256": after_base,
                "before_deltas": before_deltas,
                "after_deltas": after_deltas,
            }
        }).as_object().unwrap().clone()
    );
    (sample, original_output, status)
}

async fn measure_serial_handler(
    operator: &mut StreamJoinOperator,
    context: &StreamOperatorContext<'_>,
    incoming_batch: Batch,
    output: &mut EdgeCollector,
) -> Value {
    let combined_resources = PhaseClock::start("combined");
    let preparation_resources = PhaseClock::start("preparation");
    let combined_start = Instant::now();
    // Poll real preparation once before declaring the handler ready. The same
    // operator stays exclusively borrowed until preparation completes.
    let (first_poll_pending, handler_ready_ns) = {
        let preparation = operator.prepare_checkpoint_async(context);
        tokio::pin!(preparation);
        let first_poll_pending = match futures::poll!(&mut preparation) {
            std::task::Poll::Pending => true,
            std::task::Poll::Ready(result) => {
                result.unwrap();
                false
            }
        };
        let ready = combined_start.elapsed().as_nanos();
        if first_poll_pending {
            preparation.await.unwrap();
        }
        (first_poll_pending, ready)
    };
    let preparation_completed_ns = combined_start.elapsed().as_nanos();
    let preparation_seconds = combined_start.elapsed().as_secs_f64();
    let preparation_resources = preparation_resources.finish();
    let handler_resources = PhaseClock::start("handler");
    let handler_started_ns = combined_start.elapsed().as_nanos();
    let queue_seconds =
        Duration::from_nanos(u64::try_from(handler_started_ns - handler_ready_ns).unwrap())
            .as_secs_f64();
    let handler_start = Instant::now();
    operator
        .process_data("left", incoming_batch, context, output)
        .await
        .unwrap();
    let handler_seconds = handler_start.elapsed().as_secs_f64();
    let handler_completed_ns = combined_start.elapsed().as_nanos();
    let handler_resources = handler_resources.finish();
    let combined_seconds = combined_start.elapsed().as_secs_f64();
    let combined_resources = combined_resources.finish();
    json!({
        "preparation_seconds": preparation_seconds,
        "handler_seconds": handler_seconds,
        "combined_seconds": combined_seconds,
        "queue_seconds": queue_seconds,
        "resource_phases": {"preparation":preparation_resources,"handler":handler_resources,"combined":combined_resources},
        "schedule": {
            "clock":"monotonic-ns-from-preparation-start",
            "owner":"single-exclusive-operator",
            "preparation_started_ns":0,
            "preparation_first_poll_pending":first_poll_pending,
            "handler_ready_ns":handler_ready_ns,
            "preparation_completed_ns":preparation_completed_ns,
            "handler_started_ns":handler_started_ns,
            "handler_completed_ns":handler_completed_ns,
            "during_compaction_status":"unmeasured",
            "reason":"prepare_checkpoint_async exclusively borrows this operator through worker completion and installation; handler dispatch is serialized",
            "queue_scope":"benchmark handler readiness after first preparation poll to handler start; not a production edge-queue clock",
        },
    })
}

pub(crate) async fn compaction(rows: usize, incoming: usize) -> Value {
    let (armed, armed_output, armed_status) = compaction_sample(rows, incoming, false).await;
    let (steady, steady_output, steady_status) = compaction_sample(rows, incoming, true).await;
    assert_eq!(armed_output, steady_output);
    assert_eq!(armed_status, steady_status);
    json!({
        "schema": "calc-flow.native-component.v1",
        "case": "join_compaction_1m",
        "checkpoint_enabled": true,
        "resident_rows": rows,
        "incoming_rows": incoming,
        "armed": armed,
        "steady": steady,
        "outputs_equal": true,
        "state_equal": true,
        "restore_output_equivalent": true,
    })
}

#[cfg(test)]
mod dictionary_arguments {
    #[test]
    fn test_dictionary_refusal_requires_the_registered_check_only_case() {
        let arguments = [
            "--native-component".into(),
            "join_dictionary_refusal_retention".into(),
            "--check".into(),
        ];
        assert!(super::parse_arguments(&arguments).unwrap().unwrap().1);
        assert!(super::parse_arguments(&arguments[..2]).is_err());
        assert!(
            super::parse_arguments(&["--check".into()])
                .unwrap()
                .is_none()
        );
        let mut extra = arguments.to_vec();
        extra.extend(["--rows".into(), "3".into()]);
        assert!(super::parse_arguments(&extra).is_err());
    }
}
