use super::*;
use crate::{OperatorStateSnapshot, StateSegment};
use datafusion::execution::memory_pool::{
    GreedyMemoryPool, MemoryConsumer, MemoryPool, MemoryReservation,
};
use serde_json::json;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug)]
struct ObservedPool {
    pool: GreedyMemoryPool,
    peak: AtomicUsize,
}

impl ObservedPool {
    fn new(limit: usize) -> Self {
        Self {
            pool: GreedyMemoryPool::new(limit),
            peak: AtomicUsize::new(0),
        }
    }

    fn observe(&self) {
        self.peak.fetch_max(self.pool.reserved(), Ordering::SeqCst);
    }
}

impl std::fmt::Display for ObservedPool {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "ASOF projection test pool")
    }
}

impl MemoryPool for ObservedPool {
    fn name(&self) -> &str {
        self.pool.name()
    }

    fn memory_limit(&self) -> datafusion::execution::memory_pool::MemoryLimit {
        self.pool.memory_limit()
    }

    fn grow(&self, reservation: &MemoryReservation, additional: usize) {
        self.pool.grow(reservation, additional);
        self.observe();
    }

    fn shrink(&self, reservation: &MemoryReservation, shrink: usize) {
        self.pool.shrink(reservation, shrink);
    }

    fn try_grow(
        &self,
        reservation: &MemoryReservation,
        additional: usize,
    ) -> datafusion::common::Result<()> {
        self.pool.try_grow(reservation, additional)?;
        self.observe();
        Ok(())
    }

    fn reserved(&self) -> usize {
        self.pool.reserved()
    }
}

fn configured(spec: StreamAsofJoinSpec, projection: Option<&[usize]>) -> StreamAsofJoinOperator {
    let schema = input_schema();
    let mut operator = StreamAsofJoinOperator::new("asof", schema.clone(), schema, spec).unwrap();
    if let Some(projection) = projection {
        operator.set_output_projection(projection.to_vec()).unwrap();
    }
    operator
}

fn ports(ports: &[Port]) -> Vec<(String, BatchKind, bool, Option<SchemaRef>)> {
    ports
        .iter()
        .map(|port| {
            (
                port.name().into(),
                port.kind(),
                port.required(),
                port.schema().cloned(),
            )
        })
        .collect()
}

fn descriptor(operator: &StreamAsofJoinOperator, columns: [&[usize]; 2]) -> serde_json::Value {
    let digests = columns
        .iter()
        .enumerate()
        .map(|(side, columns)| {
            let schema = operator.schemas[side].project(columns).unwrap();
            hex::encode(codec::schema_digest(&schema).unwrap())
        })
        .collect::<Vec<_>>();
    json!({"columns": columns, "schema_digests": digests})
}

fn assert_columns(operator: &StreamAsofJoinOperator, left: &[&str], right: &[&str]) {
    let names = |fields: &[&str]| {
        fields
            .iter()
            .map(|field| (*field).to_owned())
            .collect::<Vec<_>>()
    };
    assert_eq!(
        retained_columns(operator),
        [(0, vec![names(left)]), (1, vec![names(right)])]
            .into_iter()
            .collect()
    );
}

fn assert_refused(
    operator: &mut StreamAsofJoinOperator,
    snapshot: &OperatorStateSnapshot,
    field: &str,
) {
    let before = operator.capture(Epoch::INITIAL).unwrap();
    let status = operator.status();
    let reserved = operator.runtime.pool.reserved();
    let binding = (
        ports(operator.output_ports()),
        operator.fingerprint.clone(),
        operator.next_output_sequence,
        operator.terminal,
    );
    let swept = operator.swept;
    let error = operator.restore(snapshot).unwrap_err();
    assert!(
        matches!(error, CalcFlowError::CheckpointMismatch { .. }),
        "{error}"
    );
    assert!(error.to_string().contains(field), "{error}");
    assert_eq!(operator.status(), status);
    assert_eq!(operator.runtime.pool.reserved(), reserved);
    assert_eq!(
        (
            ports(operator.output_ports()),
            operator.fingerprint.clone(),
            operator.next_output_sequence,
            operator.terminal
        ),
        binding
    );
    assert!(operator.swept == swept);
    let after = operator.capture(Epoch::INITIAL).unwrap();
    assert_eq!(after.inline_metadata, before.inline_metadata);
    assert_eq!(after.segments, before.segments);
}

fn backing_bytes(operator: &StreamAsofJoinOperator) -> usize {
    let mut allocations: BTreeMap<usize, usize> = BTreeMap::new();
    for (_, (payload, _)) in operator.state.batches.iter() {
        for column in payload.record.columns() {
            let data = column.to_data();
            for buffer in data.buffers().iter().chain(
                data.nulls()
                    .map(datafusion::arrow::buffer::NullBuffer::buffer),
            ) {
                allocations
                    .entry(buffer.data_ptr().as_ptr() as usize)
                    .and_modify(|capacity| *capacity = (*capacity).max(buffer.capacity()))
                    .or_insert(buffer.capacity());
            }
        }
    }
    allocations.values().sum()
}

async fn seed_one(operator: &mut StreamAsofJoinOperator, right: Batch) {
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let before = right.clone();
    operator
        .process_data("right", right, &context, &mut output)
        .await
        .unwrap();
    assert_eq!(before.table_payload().unwrap().batches()[0].num_rows(), 1);
    let (left, _, _) = input("left", &[("A", 100, 1, 1)]);
    operator
        .process_data("left", left, &context, &mut output)
        .await
        .unwrap();
    assert!(output.drain("output").is_empty());
    let snapshot = operator.capture(Epoch::INITIAL).unwrap();
    let columns = operator.output_ports()[0]
        .schema()
        .unwrap()
        .fields()
        .iter()
        .map(|field| operator.schemas[2].index_of(field.name()).unwrap())
        .collect::<Vec<_>>();
    let mut restored = configured(operator.spec.clone(), Some(&columns));
    restored.restore(&snapshot).unwrap();
    restored.on_end(&context, &mut output).await.unwrap();
    let messages = output.drain("output");
    let records = messages
        .iter()
        .flat_map(|message| {
            let batch = message.as_data().unwrap();
            assert_eq!(
                batch.metadata(),
                &BatchMetadata::new("asof", 0, JsonMap::new()).unwrap()
            );
            batch.table_payload().unwrap().batches()
        })
        .collect::<Vec<_>>();
    let independent = expected(Some(&columns)).slice(0, 1);
    assert_eq!(
        concat_batches(&independent.schema(), records).unwrap(),
        independent
    );
    assert_eq!(restored.status.matched_rows, 1);
    assert_eq!(restored.status.state_bytes, 0);
}

#[tokio::test]
async fn right_only_dependencies_keep_left_identity_without_left_value() {
    let (operator, _, _) = admitted(Some(&[11])).await;
    assert_columns(
        &operator,
        &["key", "time", "seq"],
        &["key", "time", "seq", "value"],
    );
}

#[tokio::test]
async fn overlapping_key_sequence_dependency_is_retained_once() {
    let template = operator(None);
    let side = |prefix: &str| {
        AsofJoinSide::new(
            vec!["key".into()],
            "time".into(),
            vec!["key".into(), "seq".into()],
            prefix.into(),
        )
        .unwrap()
    };
    let spec = StreamAsofJoinSpec::new(
        side("left"),
        side("right"),
        Duration::from_micros(1_000),
        template.spec.limits(),
    )
    .unwrap();
    let mut operator = configured(spec, None);
    let fingerprint = operator.fingerprint.clone();
    let inputs = ports(operator.input_ports());
    operator.set_output_projection(vec![11]).unwrap();
    assert_eq!(operator.fingerprint, fingerprint);
    assert_eq!(ports(operator.input_ports()), inputs);
    let (right, _, _) = input("right", &[("A", 90, u64::MAX, 500)]);
    seed_one(&mut operator, right).await;
    assert_columns(
        &operator,
        &["key", "time", "seq"],
        &["key", "time", "seq", "value"],
    );
}

#[tokio::test]
async fn shared_ipc_and_sliced_backing_are_detached_or_fully_paid() {
    let mut observed = Vec::new();
    for sliced in [false, true] {
        let mut operator = operator(Some(NARROW));
        let rows = if sliced {
            vec![("Z", 80, 19, 400), ("A", 90, u64::MAX, 500)]
        } else {
            vec![("A", 90, u64::MAX, 500)]
        };
        let (source, _, _) = input("right", &rows);
        let record = &source.table_payload().unwrap().batches()[0];
        let encoded = codec::encode_batch(record, 64 << 20, &mut Vec::new()).unwrap();
        let decoded = codec::decode_table_batch(
            &encoded,
            &codec::schema_digest(&input_schema()).unwrap(),
            &input_schema(),
            100,
        )
        .unwrap();
        let kept = decoded.column(3).to_data();
        let discarded = decoded.column(4).to_data();
        assert_eq!(
            kept.buffers()[0].data_ptr(),
            discarded.buffers()[1].data_ptr()
        );
        assert!(kept.buffers()[0].capacity() > TEXT_BYTES);
        let decoded = if sliced { decoded.slice(1, 1) } else { decoded };
        let before = decoded.clone();
        seed_one(
            &mut operator,
            Batch::table(vec![decoded], BatchMetadata::default()).unwrap(),
        )
        .await;
        assert_eq!(
            before
                .column(3)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(0),
            500
        );
        assert!(usize::try_from(operator.status.state_bytes).unwrap() >= backing_bytes(&operator));
        for (_, (payload, _)) in operator.state.batches.iter() {
            assert!(
                usize::try_from(state::capacity_batch_allocation(payload, "asof").unwrap())
                    .unwrap()
                    >= payload
                        .record
                        .columns()
                        .iter()
                        .map(|column| column.to_data().get_buffer_memory_size())
                        .max()
                        .unwrap()
            );
        }
        observed.push(operator);
    }
    for operator in observed {
        assert_columns(
            &operator,
            &["key", "time", "seq", "value"],
            &["key", "time", "seq", "value"],
        );
    }
}

#[test]
fn projection_plan_prepays_real_configuration_allocations() {
    let mut operator = operator(None);
    let observed = Arc::new(ObservedPool::new(64 << 20));
    operator.runtime.pool = observed.clone();
    let allocations =
        allocation_counter::measure(|| operator.set_output_projection(NARROW.to_vec()).unwrap());
    assert!(allocations.bytes_current > 0);
    assert!(
        i64::try_from(observed.reserved()).unwrap() >= allocations.bytes_current,
        "configuration owners have no retained credit: {allocations:?}"
    );
    assert!(
        observed.peak.load(Ordering::SeqCst) as u64 >= allocations.bytes_max,
        "configuration peak allocated before payment: {allocations:?}"
    );
    drop(operator);
    assert_eq!(observed.reserved(), 0);
}

#[test]
fn projection_plan_tiny_budget_refuses_atomically() {
    let template = operator(None);
    let spec = StreamAsofJoinSpec::new(
        template.spec.left().clone(),
        template.spec.right().clone(),
        Duration::from_micros(1_000),
        AsofStateLimits::new(100, 1).unwrap(),
    )
    .unwrap();
    let mut operator = configured(spec, None);
    let outputs = ports(operator.output_ports());
    let fingerprint = operator.fingerprint.clone();
    assert!(matches!(
        operator.set_output_projection(NARROW.to_vec()),
        Err(CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofWorkspaceLimitExceeded,
            ..
        })
    ));
    assert_eq!(ports(operator.output_ports()), outputs);
    assert_eq!(operator.fingerprint, fingerprint);
    assert_eq!(operator.runtime.pool.reserved(), 0);
}

#[tokio::test]
async fn capture_uses_layout_six_and_exact_physical_descriptor() {
    let (mut operator, _, _) = admitted(Some(NARROW)).await;
    let fingerprint = operator.fingerprint.clone();
    let snapshot = operator.capture(Epoch::INITIAL).unwrap();
    assert_eq!(snapshot.inline_metadata["fingerprint"], json!(fingerprint));
    assert_eq!(snapshot.inline_metadata["state_version"], json!(3));
    assert_eq!(snapshot.inline_metadata["layout_version"], json!(10));
    assert_eq!(snapshot.inline_metadata["accounting_version"], json!(10));
    assert_eq!(
        snapshot.inline_metadata.get("retained_payloads"),
        Some(&descriptor(&operator, [&[0, 1, 2, 3], &[0, 1, 2, 3]]))
    );
    assert!(
        snapshot.segments["asof-log-v10-1-0-1"]
            .bytes()
            .starts_with(b"CFASDL10")
    );
}

#[tokio::test]
async fn narrow_capture_refuses_full_or_different_retained_dependency_plan() {
    let (mut source, _, _) = admitted(Some(NARROW)).await;
    let snapshot = source.capture(Epoch::INITIAL).unwrap();
    for projection in [None, Some(&[12][..])] {
        let mut target = operator(projection);
        assert_refused(&mut target, &snapshot, "retained_payloads");
    }
}

#[tokio::test]
async fn same_retained_set_accepts_different_output_order_and_repeats() {
    let (mut source, _, _) = admitted(Some(NARROW)).await;
    let snapshot = source.capture(Epoch::INITIAL).unwrap();
    let projection = &[3, 11, 10];
    let mut target = operator(Some(projection));
    target.restore(&snapshot).unwrap();
    let job = StreamJobContext::new(2, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(target.output_ports().to_vec());
    target.on_end(&context, &mut output).await.unwrap();
    let messages = output.drain("output");
    let records = messages
        .iter()
        .flat_map(|message| {
            message
                .as_data()
                .unwrap()
                .table_payload()
                .unwrap()
                .batches()
        })
        .collect::<Vec<_>>();
    let independent = expected(Some(projection)).slice(0, 3);
    assert_eq!(
        concat_batches(&independent.schema(), records).unwrap(),
        independent
    );
    assert_eq!(target.status.matched_rows, 2);
    assert_eq!(target.status.unmatched_rows, 1);
}

#[tokio::test]
async fn wrong_missing_or_mixed_physical_descriptor_refuses_before_install() {
    let (mut source, _, _) = admitted(Some(NARROW)).await;
    let snapshot = source.capture(Epoch::INITIAL).unwrap();
    assert_eq!(snapshot.inline_metadata["layout_version"], json!(10));
    let good = descriptor(&source, [&[0, 1, 2, 3], &[0, 1, 2, 3]]);
    assert_eq!(snapshot.inline_metadata["retained_payloads"], good);
    let mut bad = Vec::new();
    let mut missing = snapshot.clone();
    missing.inline_metadata.remove("retained_payloads");
    bad.push((missing, "retained_payloads"));
    for replacement in [
        json!({"columns": [[0, 1, 3], [0, 1, 2, 3]], "schema_digests": good["schema_digests"]}),
        json!({"columns": [[0, 2, 1, 3], [0, 1, 2, 3]], "schema_digests": good["schema_digests"]}),
        json!({"columns": good["columns"], "schema_digests": ["0".repeat(64), good["schema_digests"][1]]}),
        json!({"columns": good["columns"], "schema_digests": good["schema_digests"], "extra": true}),
    ] {
        let mut changed = snapshot.clone();
        changed
            .inline_metadata
            .insert("retained_payloads".into(), replacement);
        bad.push((changed, "retained_payloads"));
    }
    let mut mixed = snapshot.clone();
    mixed
        .inline_metadata
        .insert("accounting_version".into(), json!(5));
    bad.push((mixed, "state version"));
    for (changed, field) in bad {
        let mut target = operator(Some(NARROW));
        assert_refused(&mut target, &changed, field);
    }
}

#[tokio::test]
async fn physical_payload_type_order_and_identity_columns_are_bound_to_descriptor() {
    let (mut source, _, _) = admitted(Some(NARROW)).await;
    let snapshot = source.capture(Epoch::INITIAL).unwrap();
    assert_eq!(snapshot.inline_metadata["layout_version"], json!(10));
    let (key, (payload, _)) = source.state.batches.iter().next().unwrap();
    let record = &payload.record;
    let mut fields = record
        .schema()
        .fields()
        .iter()
        .map(|field| field.as_ref().clone())
        .collect::<Vec<_>>();
    fields[2] = Field::new("seq", DataType::Int64, false);
    let mut columns = record.columns().to_vec();
    columns[2] = Arc::new(Int64Array::from(vec![7; record.num_rows()]));
    let wrong_type = RecordBatch::try_new(
        Arc::new(Schema::new_with_metadata(
            fields,
            record.schema().metadata().clone(),
        )),
        columns,
    )
    .unwrap();
    for changed in [
        record.project(&[1, 0, 2, 3]).unwrap(),
        record.project(&[0, 2, 3]).unwrap(),
        wrong_type,
    ] {
        let mut invalid = snapshot.clone();
        let bytes = codec::encode_batch(&changed, 64 << 20, &mut Vec::new()).unwrap();
        invalid.segments.insert(
            format!("asof-batch-{}-{}", key.0, key.1),
            StateSegment::new(bytes),
        );
        let segment = &invalid.segments[&format!("asof-batch-{}-{}", key.0, key.1)];
        for payload in invalid.inline_metadata.get_mut("checkpoint_log").unwrap()["payloads"]
            .as_array_mut()
            .unwrap()
        {
            if payload["key"] == json!([key.0, key.1]) {
                payload["sha256"] = json!(segment.sha256());
                payload["bytes"] = json!(segment.bytes().len());
            }
        }
        let mut target = operator(Some(NARROW));
        assert_refused(&mut target, &invalid, "Arrow");
    }
}

#[tokio::test]
async fn current_full_snapshot_preserves_gauges_metadata_and_complete_output() {
    let (mut source, _, _) = admitted(None).await;
    let snapshot = source.capture(Epoch::INITIAL).unwrap();
    assert_eq!(snapshot.inline_metadata["layout_version"], json!(10));
    let mut target = operator(None);
    target.restore(&snapshot).unwrap();
    assert_eq!(target.status.state_rows, 6);
    assert_eq!(
        target.status.state_bytes,
        snapshot.inline_metadata["metrics"]["state_bytes"]
            .as_u64()
            .unwrap()
    );
    let repeated = target.capture(Epoch::INITIAL).unwrap();
    assert_eq!(repeated.inline_metadata, snapshot.inline_metadata);
    assert_eq!(repeated.segments, snapshot.segments);
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(target.output_ports().to_vec());
    target.on_end(&context, &mut output).await.unwrap();
    let messages = output.drain("output");
    let records = messages
        .iter()
        .flat_map(|message| {
            let batch = message.as_data().unwrap();
            assert_eq!(
                batch.metadata(),
                &BatchMetadata::new("asof", 0, JsonMap::new()).unwrap()
            );
            batch.table_payload().unwrap().batches()
        })
        .collect::<Vec<_>>();
    let independent = expected(None).slice(0, 3);
    let actual = concat_batches(&independent.schema(), records).unwrap();
    assert_eq!(actual.schema(), independent.schema());
    assert_eq!(actual, independent);
    assert_eq!(target.status.matched_rows, 2);
    assert_eq!(target.status.unmatched_rows, 1);
    assert_eq!(target.status.state_bytes, 0);
    let pool = target.runtime.pool.clone();
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    drop(repeated);
    drop(target);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn current_gauge_and_progress_refusals_preserve_live_state() {
    let (mut target, _, _) = admitted(Some(NARROW)).await;
    let snapshot = target.capture(Epoch::INITIAL).unwrap();
    let mut forged = snapshot.clone();
    forged.inline_metadata.get_mut("metrics").unwrap()["state_bytes"] = json!(
        snapshot.inline_metadata["metrics"]["state_bytes"]
            .as_u64()
            .unwrap()
            + 1
    );
    assert_refused(
        &mut target,
        &forged,
        "recomputed state charge or gauges differ",
    );
    let status = target.status();
    let reserved = target.runtime.pool.reserved();
    let binding = (
        ports(target.output_ports()),
        target.fingerprint.clone(),
        target.next_output_sequence,
        target.terminal,
    );
    let swept = target.swept;
    let sequence = target.next_output_sequence;
    let invalid = IngressProgressSnapshot::new(BTreeMap::from([(
        "left".into(),
        IngressProgress::new(IngressState::Active, None),
    )]));
    let error = target
        .restore_with_progress(&snapshot, &invalid, None)
        .unwrap_err();
    assert!(matches!(error, CalcFlowError::CheckpointMismatch { .. }));
    assert!(
        error.to_string().contains("exact two-ingress progress"),
        "{error}"
    );
    assert_eq!(target.status(), status);
    assert_eq!(target.next_output_sequence, sequence);
    assert_eq!(target.runtime.pool.reserved(), reserved);
    assert_eq!(
        (
            ports(target.output_ports()),
            target.fingerprint.clone(),
            target.next_output_sequence,
            target.terminal
        ),
        binding
    );
    assert!(target.swept == swept);
    let repeated = target.capture(Epoch::INITIAL).unwrap();
    assert_eq!(repeated.inline_metadata, snapshot.inline_metadata);
    assert_eq!(repeated.segments, snapshot.segments);
}

#[tokio::test]
async fn current_invalid_progress_refunds_real_decode_allocations() {
    let invalid = IngressProgressSnapshot::new(BTreeMap::from([(
        "left".into(),
        IngressProgress::new(IngressState::Active, None),
    )]));
    for projection in [None, Some(NARROW)] {
        let (mut source, _, _) = admitted(projection).await;
        let snapshot = source.capture(Epoch::INITIAL).unwrap();
        let mut target = operator(None);
        let observed = Arc::new(ObservedPool::new(64 << 20));
        target.runtime.pool = observed.clone();
        if let Some(projection) = projection {
            target.set_output_projection(projection.to_vec()).unwrap();
        }
        let credit = observed.reserved();
        observed.peak.store(credit, Ordering::SeqCst);
        let mut failure = None;
        let allocations = allocation_counter::measure(|| {
            failure = Some(
                target
                    .restore_with_progress(&snapshot, &invalid, None)
                    .unwrap_err(),
            );
        });
        let error = failure.unwrap();
        assert!(matches!(error, CalcFlowError::CheckpointMismatch { .. }));
        assert!(
            error.to_string().contains("exact two-ingress progress"),
            "{error}"
        );
        assert_eq!(target.status.state_rows, 0);
        assert_eq!(observed.reserved(), credit);
        assert!(
            observed.peak.load(Ordering::SeqCst) as u64 >= allocations.bytes_max,
            "real restored owners exceed paid peak: {allocations:?}"
        );
        drop(target);
        assert_eq!(observed.reserved(), 0);
    }
}

#[tokio::test]
async fn current_restore_workspace_refusal_preserves_state_and_reservation() {
    let (mut target, _, _) = admitted(Some(NARROW)).await;
    let snapshot = target.capture(Epoch::INITIAL).unwrap();
    let status = target.status();
    let reserved = target.runtime.pool.reserved();
    let binding = (
        ports(target.output_ports()),
        target.fingerprint.clone(),
        target.next_output_sequence,
        target.terminal,
    );
    let swept = target.swept;
    let hold = MemoryConsumer::new("projection-restore-refusal").register(&target.runtime.pool);
    hold.try_grow((64 << 20) - reserved - 1).unwrap();
    assert!(matches!(
        target.restore(&snapshot),
        Err(CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofWorkspaceLimitExceeded,
            ..
        })
    ));
    drop(hold);
    assert_eq!(target.status(), status);
    assert_eq!(target.runtime.pool.reserved(), reserved);
    assert_eq!(
        (
            ports(target.output_ports()),
            target.fingerprint.clone(),
            target.next_output_sequence,
            target.terminal
        ),
        binding
    );
    assert!(target.swept == swept);
    let after = target.capture(Epoch::INITIAL).unwrap();
    assert_eq!(after.inline_metadata, snapshot.inline_metadata);
    assert_eq!(after.segments, snapshot.segments);
}

#[tokio::test]
async fn hard_row_budget_refusal_does_not_install_projected_payload() {
    let template = operator(None);
    let spec = StreamAsofJoinSpec::new(
        template.spec.left().clone(),
        template.spec.right().clone(),
        Duration::from_micros(1_000),
        AsofStateLimits::new(1, 64 << 20).unwrap(),
    )
    .unwrap();
    let mut target = configured(spec, Some(NARROW));
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(target.output_ports().to_vec());
    let (right, _, _) = input("right", &[("A", 90, u64::MAX, 500)]);
    target
        .process_data("right", right, &context, &mut output)
        .await
        .unwrap();
    let before = target.capture(Epoch::INITIAL).unwrap();
    let reserved = target.runtime.pool.reserved();
    let (left, owners, _) = input("left", &[("A", 100, 1, 1)]);
    assert!(matches!(
        target
            .process_data("left", left, &context, &mut output)
            .await,
        Err(CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofStateLimitExceeded,
            ..
        })
    ));
    assert_eq!(target.status.state_rows, 1);
    assert_eq!(target.status.left.accepted_rows, 0);
    assert_eq!(target.status.state_limit_failures, 1);
    assert!(owners.iter().all(|owner| owner.upgrade().is_none()));
    assert_eq!(target.runtime.pool.reserved(), reserved);
    assert_eq!(
        target.capture(Epoch::INITIAL).unwrap().segments,
        before.segments
    );
}

struct CancelOutput(CancellationToken);

#[async_trait]
impl StreamCollector for CancelOutput {
    async fn emit(&mut self, _port: &str, _batch: Batch) -> Result<()> {
        self.0.cancel();
        std::future::pending().await
    }
}

#[tokio::test]
async fn cancelled_delivery_preserves_projected_checkpoint_and_owners() {
    let (mut target, _, _) = admitted(Some(NARROW)).await;
    let before = target.capture(Epoch::INITIAL).unwrap();
    let status = target.status();
    let reserved = target.runtime.pool.reserved();
    let cancellation = CancellationToken::new();
    let job = StreamJobContext::new(2, "asof", JsonMap::new(), None, cancellation.clone());
    let context = StreamOperatorContext::new(&job, "asof", None);
    assert!(matches!(
        target
            .on_watermark(
                EventTime::from_micros(104),
                &context,
                &mut CancelOutput(cancellation)
            )
            .await,
        Err(CalcFlowError::Cancelled { .. })
    ));
    assert_eq!(target.status(), status);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    tokio::time::timeout(Duration::from_secs(2), async {
        while target.runtime.pool.reserved() != reserved {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let after = target.capture(Epoch::INITIAL).unwrap();
    assert_eq!(after.inline_metadata, before.inline_metadata);
    assert_eq!(after.segments, before.segments);
    assert_columns(
        &target,
        &["key", "time", "seq", "value"],
        &["key", "time", "seq", "value"],
    );
}

#[tokio::test]
async fn reset_releases_payload_owners_and_preserves_projection_configuration() {
    let (mut target, owners, _) = admitted(Some(NARROW)).await;
    let before = target.capture(Epoch::INITIAL).unwrap();
    let fingerprint = target.fingerprint.clone();
    let output = ports(target.output_ports());
    target.reset().unwrap();
    assert_eq!(target.status.state_bytes, 0);
    assert!(target.state.batches.is_empty());
    assert!(owners.iter().all(|owner| owner.upgrade().is_none()));
    assert_eq!(target.fingerprint, fingerprint);
    assert_eq!(ports(target.output_ports()), output);
    target.restore(&before).unwrap();
    assert_columns(
        &target,
        &["key", "time", "seq", "value"],
        &["key", "time", "seq", "value"],
    );
}

#[tokio::test]
async fn mixed_late_admission_uses_original_ordinals_then_retains_projected_payload() {
    let template = operator(None);
    let mut target = configured(
        template.spec.with_late_policy(AsofLatePolicy::Drop),
        Some(NARROW),
    );
    target.status.right.watermark_micros = Some(EventTime::from_micros(90));
    let (right, _, _) = input("right", &[("Z", 89, 7, 400), ("A", 90, u64::MAX, 500)]);
    let record = right.table_payload().unwrap().batches()[0].clone();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(target.output_ports().to_vec());
    target
        .process_data("right", right, &context, &mut output)
        .await
        .unwrap();
    assert_eq!(record.num_rows(), 2);
    assert_eq!(
        record
            .column(3)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values()
            .as_ref(),
        &[400, 500]
    );
    assert_eq!(target.status.right.late_rows, 1);
    assert_eq!(target.status.right.accepted_rows, 1);
    let (left, _, _) = input("left", &[("A", 100, 1, 1)]);
    target
        .process_data("left", left, &context, &mut output)
        .await
        .unwrap();
    let snapshot = target.capture(Epoch::INITIAL).unwrap();
    let mut restored = configured(target.spec.clone(), Some(NARROW));
    restored
        .restore_with_progress(&snapshot, &asymmetric_progress(90, 90), None)
        .unwrap();
    restored.on_end(&context, &mut output).await.unwrap();
    let messages = output.drain("output");
    let records = messages
        .iter()
        .flat_map(|message| {
            message
                .as_data()
                .unwrap()
                .table_payload()
                .unwrap()
                .batches()
        })
        .collect::<Vec<_>>();
    let independent = expected(Some(NARROW)).slice(0, 1);
    assert_eq!(
        concat_batches(&independent.schema(), records).unwrap(),
        independent
    );
    assert_columns(
        &target,
        &["key", "time", "seq", "value"],
        &["key", "time", "seq", "value"],
    );
}

#[test]
fn current_full_layout_rejects_historical_nonempty_inventory() {
    let value: serde_json::Value = serde_json::from_slice(include_bytes!(
        "fixtures/legacy-v5/retained-projection.json"
    ))
    .unwrap();
    let segments = value["segments"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, encoded)| {
            let hex = encoded["hex"].as_str().unwrap();
            let mut bytes = vec![0; hex.len() / 2];
            hex::decode_to_slice(hex, &mut bytes).unwrap();
            let segment = StateSegment::new(bytes);
            assert_eq!(segment.sha256(), encoded["sha256"]);
            (name.clone(), segment)
        })
        .collect();
    let snapshot = OperatorStateSnapshot {
        inline_metadata: serde_json::from_value(value["metadata"].clone()).unwrap(),
        segments,
    };
    assert_eq!(snapshot.inline_metadata["layout_version"], json!(5));
    assert_eq!(snapshot.inline_metadata["accounting_version"], json!(5));
    assert!(!snapshot.segments.is_empty());
    let mut target = operator(None);
    assert_eq!(
        snapshot.inline_metadata["fingerprint"],
        json!(target.fingerprint)
    );
    let before = target.capture(Epoch::INITIAL).unwrap();
    let pool = target.runtime.pool.clone();
    let result = target.restore(&snapshot);
    let unchanged = target.capture(Epoch::INITIAL).unwrap().inline_metadata
        == before.inline_metadata
        && pool.reserved() == 0;
    drop(target);
    assert_eq!(pool.reserved(), 0);
    assert!(
        matches!(result, Err(CalcFlowError::CheckpointMismatch { message })
        if message == "ASOF kind, schema, configuration or state version differs")
    );
    assert!(unchanged);
}
