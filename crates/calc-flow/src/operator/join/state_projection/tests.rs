use std::{sync::Arc, time::Duration};

use datafusion::arrow::{
    array::{
        Array, ArrayRef, Int8Array, Int64Array, ListArray, StringArray, StringViewArray,
        StructArray, TimestampMicrosecondArray,
    },
    buffer::Buffer,
    datatypes::{DataType, Field, Int64Type, Schema, TimeUnit},
    record_batch::RecordBatch,
};

use super::super::StreamJoinOperator;
use crate::{
    Batch, BatchMetadata, CancellationToken, EdgeCollector, JoinStateLimits, JoinTimeBounds,
    JsonMap, OperatorMetadata, StreamJobContext, StreamJoinSpec, StreamOperator,
    StreamOperatorContext,
};

mod dictionaries;
mod sparse;
mod validation;
mod worker;

pub(in crate::operator::join) async fn selected_payload_cases(
    service: &crate::runtime::streaming::gather_work::TestService,
) {
    Box::pin(worker::lifecycle_controls(service)).await;
    Box::pin(sparse::retained_sparse_repair()).await;
    super::super::checkpoint_v2::projected_copy::views::assert_dictionary_controls();
    Box::pin(dictionaries::noncontiguous_dictionary_payloads()).await;
    Box::pin(validation::projected_snapshot_refusals()).await;
    Box::pin(layout_changes_invalidate_fresh_history()).await;
    Box::pin(reject_narrow_legacy_checkpoint()).await;
    schema_digest_contracts();
    super::super::checkpoint_v2::projected_copy::assert_selected_copy_budget();
    let length = super::super::checkpoint_v2::projected_copy::views::validate_length;
    assert_eq!(
        length(u32::MAX as usize - 1).unwrap(),
        u32::MAX as usize - 1
    );
    assert!(matches!(
        length(u32::MAX as usize),
        Err(crate::CalcFlowError::DataFusion { .. })
    ));
    let nullable: ArrayRef = Arc::new(Int64Array::from(vec![Some(10), None, Some(30)]));
    selected_case(nullable, Arc::new(Int8Array::from(vec![7, 8, 9])), "left").await;
    let nested: ArrayRef = Arc::new(ListArray::from_iter_primitive::<Int64Type, _, _>([
        Some(vec![Some(10), None]),
        None,
        Some(vec![Some(30)]),
    ]));
    selected_case(nested, Arc::new(Int64Array::from(vec![7, 8, 9])), "right").await;
    selected_case(
        view_values(),
        Arc::new(Int64Array::from(vec![7, 8, 9])),
        "right",
    )
    .await;
    let views = view_values();
    let nested: ArrayRef = Arc::new(StructArray::from(vec![(
        Arc::new(Field::new("view", views.data_type().clone(), true)),
        views,
    )]));
    selected_case(nested, Arc::new(Int64Array::from(vec![7, 8, 9])), "left").await;
    projected_checkpoint_layout().await;
    Box::pin(legacy_projection_migration(2)).await;
    Box::pin(legacy_projection_migration(1)).await;
    Box::pin(empty_legacy_projection()).await;
    Box::pin(dead_legacy_projection()).await;
}

async fn layout_changes_invalidate_fresh_history() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let input = input(
        Arc::new(Int64Array::from(vec![10, 20, 30])),
        Arc::new(Int64Array::from(vec![7, 8, 9])),
    );
    let schema = input.table_payload().unwrap().batches()[0].schema();
    let mut join = operator(&schema, false);
    let count = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&count);
    join.set_checkpoint_writer_base_test_hook(Arc::new(move |_| {
        observed.fetch_add(1, Ordering::SeqCst);
    }));
    let job = StreamJobContext::new(
        1,
        "layout-history",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "selected-state", None);
    join.prepare_checkpoint_async(&context).await.unwrap();
    join.set_output_projection(vec![3, 7]).unwrap();
    join.prepare_checkpoint_async(&context).await.unwrap();
    let base_count = count.load(Ordering::SeqCst);
    let checkpoint = join.checkpoint(crate::Epoch::INITIAL).unwrap();
    cold_checkpoint(&join, checkpoint, &schema, &job)
        .await
        .unwrap();
    let mut output = EdgeCollector::new(join.output_ports().to_vec());
    join.process_data("right", input, &context, &mut output)
        .await
        .unwrap();
    let before = join.status();
    let reset = join.set_output_projection((0..8).collect());
    let refused = reset.is_err();
    assert_eq!(join.status(), before);
    let pool = join.runtime.runtime().unwrap().incremental_memory_pool();
    drop((join, output, context));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
    assert_eq!(
        (base_count, refused),
        (2, true),
        "empty cached history must bind layout, and live retained layout cannot be replaced"
    );
}

async fn reject_narrow_legacy_checkpoint() {
    let input = input(
        Arc::new(Int64Array::from(vec![10, 20, 30])),
        Arc::new(Int64Array::from(vec![7, 8, 9])),
    );
    let schema = input.table_payload().unwrap().batches()[0].schema();
    let mut projected = operator(&schema, true);
    projected.checkpoint_v1_test_producer = true;
    let job = StreamJobContext::new(
        1,
        "reject-legacy-producer",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "selected-state", None);
    assert!(matches!(
        projected.prepare_checkpoint_async(&context).await,
        Err(crate::CalcFlowError::CheckpointMismatch { .. })
    ));
    assert!(projected.compaction_release.is_none());
    assert!(projected.compaction_cleanup.is_none());
    assert!(
        matches!(
            projected.checkpoint(crate::Epoch::INITIAL),
            Err(crate::CalcFlowError::CheckpointMismatch { .. })
        ),
        "a retained layout must never be mislabeled as legacy V1"
    );
}

async fn legacy_projection_migration(layout: u64) {
    let input = input(
        Arc::new(Int64Array::from(vec![10, 20, 30])),
        Arc::new(Int64Array::from(vec![7, 8, 9])),
    );
    let schema = input.table_payload().unwrap().batches()[0].schema();
    let mut legacy = operator(&schema, false);
    legacy.checkpoint_v1_test_producer = layout == 1;
    let mut projected = operator(&schema, true);
    let job = StreamJobContext::new(
        1,
        "legacy-projection",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "selected-state", None);
    let mut collector = EdgeCollector::new(legacy.output_ports().to_vec());
    legacy
        .process_data("right", input, &context, &mut collector)
        .await
        .unwrap();
    legacy.prepare_checkpoint_async(&context).await.unwrap();
    let snapshot = legacy.checkpoint(crate::Epoch::INITIAL).unwrap();
    assert_eq!(snapshot.inline_metadata["layout_version"], layout);
    projected.restore(&snapshot).unwrap();
    assert_eq!(projected.status(), legacy.status());
    let mut managed = operator(&schema, true);
    let mut managed_snapshot = managed_input(&snapshot, &mut managed);
    managed
        .restore_managed_snapshot(&mut managed_snapshot, &job, None)
        .await
        .unwrap();
    assert_eq!(managed.status(), legacy.status());
    let escaping = managed.state.right[0].record.column(0).to_data().buffers()[0].clone();
    let fields = projected.state.right[0]
        .record
        .schema_ref()
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .collect::<Vec<_>>();
    if fields == ["key", "time", "value"] {
        migration_checkpoint(&mut projected, &snapshot, &schema, &context).await;
        legacy_tail_controls(&mut projected, &snapshot).await;
    }
    post_restore_gap(&mut managed, &mut legacy, &schema, &job).await;
    let source_pool = legacy.runtime.runtime().unwrap().incremental_memory_pool();
    let target_pool = projected
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let managed_pool = managed.runtime.runtime().unwrap().incremental_memory_pool();
    drop((
        legacy,
        projected,
        managed,
        snapshot,
        managed_snapshot,
        context,
        collector,
    ));
    let escaping_funded = managed_pool.reserved() > 0;
    let mut drain = Box::pin(job.gather_owner().close_and_drain());
    if escaping_funded {
        assert!(futures::poll!(drain.as_mut()).is_pending());
    }
    drop(escaping);
    assert!(drain.await.is_empty());
    drop(job);
    assert_eq!(source_pool.reserved(), 0);
    assert_eq!(target_pool.reserved(), 0);
    assert_eq!(managed_pool.reserved(), 0);
    assert_eq!(
        fields,
        ["key", "time", "value"],
        "legacy restore must install independent projected state"
    );
    assert!(
        escaping_funded,
        "escaping migrated buffers must retain credit"
    );
}

async fn post_restore_gap(
    restored: &mut StreamJoinOperator,
    full: &mut StreamJoinOperator,
    schema: &Arc<Schema>,
    job: &StreamJobContext,
) {
    let template = input(
        Arc::new(Int64Array::from(vec![10, 20, 30])),
        Arc::new(Int64Array::from(vec![7, 8, 9])),
    );
    let valid = template.table_payload().unwrap().batches()[0].slice(0, 1);
    let mut columns = valid.columns().to_vec();
    columns[2] = Arc::new(TimestampMicrosecondArray::from(vec![99]).with_timezone("UTC"));
    let late = RecordBatch::try_new(Arc::clone(schema), columns).unwrap();
    let source = Batch::table(vec![late, valid], BatchMetadata::default()).unwrap();
    let original = source.table_payload().unwrap().batches().to_vec();
    let context = worker::mixed_context(job, true);
    let mut actual = EdgeCollector::new(restored.output_ports().to_vec());
    let mut expected = EdgeCollector::new(full.output_ports().to_vec());
    restored
        .process_data("left", source.clone(), &context, &mut actual)
        .await
        .unwrap();
    full.process_data("left", source.clone(), &context, &mut expected)
        .await
        .unwrap();
    assert_eq!(restored.status(), full.status());
    assert_eq!(restored.state.left[0].row_id, 1);
    assert_eq!(restored.state.next_left_row_id, 2);
    assert_eq!(restored.state.left[0].charge, full.state.left[0].charge);
    assert_eq!(restored.state.metrics.left.late_rows, 1);
    assert_eq!(restored.state.next_output_sequence, 1);
    compare_output(&mut actual, &mut expected);
    assert_eq!(source.table_payload().unwrap().batches(), original);
}

async fn legacy_tail_controls(
    projected: &mut StreamJoinOperator,
    snapshot: &crate::OperatorStateSnapshot,
) {
    if snapshot.inline_metadata["layout_version"] == 1 {
        projected
            .assert_legacy_assembly_cancel_for_test(snapshot)
            .await;
    }
}

fn managed_input(
    snapshot: &crate::OperatorStateSnapshot,
    operator: &mut StreamJoinOperator,
) -> crate::OperatorStateSnapshot {
    let runtime = operator.runtime.runtime().unwrap();
    let mut result = snapshot.clone();
    for segment in result
        .segments
        .values_mut()
        .filter(|segment| !segment.has_owner())
    {
        let credit = runtime.incremental_reservation("legacy-managed-fixture");
        let bytes = segment.bytes_arc().capacity()
            + size_of::<crate::StateSegment>()
            + size_of::<datafusion::execution::memory_pool::MemoryReservation>()
            + 16 * size_of::<usize>()
            + 2 * segment.sha256().len();
        credit.try_grow(bytes).unwrap();
        *segment = segment.clone().with_owner(Arc::new(credit));
    }
    result
}

async fn empty_legacy_projection() {
    let input = input(
        Arc::new(Int64Array::from(vec![10, 20, 30])),
        Arc::new(Int64Array::from(vec![7, 8, 9])),
    );
    let schema = input.table_payload().unwrap().batches()[0].schema();
    let mut legacy = operator(&schema, false);
    let job = StreamJobContext::new(
        1,
        "empty-legacy-projection",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "selected-state", None);
    let mut collector = EdgeCollector::new(legacy.output_ports().to_vec());
    for side in ["left", "right"] {
        legacy
            .process_data(side, input.clone(), &context, &mut collector)
            .await
            .unwrap();
    }
    legacy.on_end(&context, &mut collector).await.unwrap();
    let snapshot = legacy.checkpoint_v1(crate::Epoch::INITIAL).unwrap();
    assert!(snapshot.segments.is_empty());
    assert!(legacy.state.ended);
    assert_eq!(legacy.state.next_left_row_id, 3);
    assert_eq!(legacy.status().emitted_match_rows, 3);
    let mut direct = operator(&schema, true);
    let mut managed = operator(&schema, true);
    let direct_result = direct.restore(&snapshot);
    let mut managed_snapshot = snapshot.clone();
    let managed_result = managed
        .restore_managed_snapshot(&mut managed_snapshot, &job, None)
        .await;
    if direct_result.is_ok() && managed_result.is_ok() {
        assert_eq!(direct.status(), legacy.status());
        assert_eq!(managed.status(), legacy.status());
        assert!(direct.state.ended && managed.state.ended);
        reject_empty_legacy_gauges(&mut direct, &snapshot);
        migration_checkpoint(&mut direct, &snapshot, &schema, &context).await;
    }
    let pools = [&mut legacy, &mut direct, &mut managed].map(|operator| {
        operator
            .runtime
            .runtime()
            .unwrap()
            .incremental_memory_pool()
    });
    drop((
        legacy,
        direct,
        managed,
        snapshot,
        managed_snapshot,
        collector,
        context,
    ));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert!(pools.iter().all(|pool| pool.reserved() == 0));
    direct_result.expect("projected restore must accept authentic empty V1 history");
    managed_result.expect("managed projected restore must accept authentic empty V1 history");
}

fn reject_empty_legacy_gauges(
    operator: &mut StreamJoinOperator,
    snapshot: &crate::OperatorStateSnapshot,
) {
    let before = operator.status();
    let epoch = operator.state.last_checkpoint_epoch;
    for (side, gauge) in [
        ("left", "retained_rows"),
        ("left", "retained_bytes"),
        ("right", "retained_rows"),
        ("right", "retained_bytes"),
    ] {
        let mut invalid = snapshot.clone();
        invalid.inline_metadata.get_mut("metrics").unwrap()[side][gauge] = 1.into();
        assert!(operator.restore(&invalid).is_err());
        assert_eq!(operator.status(), before);
        assert_eq!(operator.state.last_checkpoint_epoch, epoch);
    }
}

async fn dead_legacy_projection() {
    let input = input(
        Arc::new(Int64Array::from(vec![10, 20, 30])),
        Arc::new(Int64Array::from(vec![7, 8, 9])),
    );
    let schema = input.table_payload().unwrap().batches()[0].schema();
    let mut source = operator(&schema, false);
    source.checkpoint_v1_test_producer = true;
    let job = StreamJobContext::new(
        1,
        "dead-legacy-projection",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "selected-state", None);
    let mut collector = EdgeCollector::new(source.output_ports().to_vec());
    source
        .process_data("right", input, &context, &mut collector)
        .await
        .unwrap();
    source.prepare_checkpoint_async(&context).await.unwrap();
    let first = source.checkpoint(crate::Epoch::INITIAL).unwrap();
    let mut projected = operator(&schema, true);
    projected.restore(&first).unwrap();
    let before = projected.status();
    source.on_end(&context, &mut collector).await.unwrap();
    let dead = source.checkpoint(crate::Epoch::new(2).unwrap()).unwrap();
    assert!(dead.segments.keys().any(|name| name.ends_with("delta-2")));
    let damaged = damage_legacy_charge(&first, &dead);
    let pool = projected
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let baseline = pool.reserved();
    assert!(
        projected.restore(&damaged).is_err(),
        "dead historical charges must be validated"
    );
    assert_eq!(projected.status(), before);
    assert_eq!(
        projected.state.last_checkpoint_epoch,
        Some(crate::Epoch::INITIAL)
    );
    assert_eq!(pool.reserved(), baseline);
    projected.restore(&dead).unwrap();
    assert_eq!(projected.status(), source.status());
    assert!(projected.state.ended);
    let mut managed = operator(&schema, true);
    let mut managed_snapshot = managed_input(&dead, &mut managed);
    managed
        .restore_managed_snapshot(&mut managed_snapshot, &job, None)
        .await
        .unwrap();
    assert_eq!(managed.status(), source.status());
    migration_checkpoint(&mut projected, &dead, &schema, &context).await;
    let pools = [&mut source, &mut projected, &mut managed].map(|operator| {
        operator
            .runtime
            .runtime()
            .unwrap()
            .incremental_memory_pool()
    });
    drop((
        source,
        projected,
        managed,
        first,
        dead,
        damaged,
        managed_snapshot,
        collector,
        context,
    ));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert!(pools.iter().all(|pool| pool.reserved() == 0));
}

fn damage_legacy_charge(
    first: &crate::OperatorStateSnapshot,
    dead: &crate::OperatorStateSnapshot,
) -> crate::OperatorStateSnapshot {
    use super::super::{
        JOIN_DELTA_UPSERT_TAG, JOIN_STATE_MAGIC, read_segment_u64, take_segment_bytes,
    };
    let (name, segment) = first
        .segments
        .iter()
        .find(|(_, segment)| u64::from_le_bytes(segment.bytes()[8..16].try_into().unwrap()) > 0)
        .unwrap();
    let mut bytes = segment.bytes().to_vec();
    let mut offset = 16;
    if &bytes[..8] == JOIN_STATE_MAGIC {
        offset += 16;
    } else {
        assert_eq!(
            take_segment_bytes(&bytes, &mut offset, 1).unwrap(),
            [JOIN_DELTA_UPSERT_TAG]
        );
        offset += 16;
        let length = usize::try_from(read_segment_u64(&bytes, &mut offset).unwrap()).unwrap();
        offset += length;
    }
    let position = offset;
    let charge = read_segment_u64(&bytes, &mut offset).unwrap();
    bytes[position..offset].copy_from_slice(&(charge + 1).to_le_bytes());
    let mut damaged = dead.clone();
    assert!(
        damaged
            .segments
            .insert(name.clone(), crate::StateSegment::new(bytes))
            .is_some()
    );
    damaged
}

async fn migration_checkpoint(
    projected: &mut StreamJoinOperator,
    legacy: &crate::OperatorStateSnapshot,
    schema: &Arc<Schema>,
    context: &StreamOperatorContext<'_>,
) {
    let epoch = crate::Epoch::new(legacy.inline_metadata["epoch"].as_u64().unwrap() + 1).unwrap();
    assert!(
        projected.checkpoint(epoch).is_err(),
        "migration must prepare a new narrow base"
    );
    projected.prepare_checkpoint_async(context).await.unwrap();
    let narrow = projected.checkpoint(epoch).unwrap();
    assert_eq!(narrow.inline_metadata["layout_version"], 3);
    assert!(
        narrow.inline_metadata["v2_inventory"]["deltas"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    for name in legacy
        .segments
        .keys()
        .filter(|name| name.contains("-payload-"))
    {
        assert!(
            !narrow.segments.contains_key(name),
            "migration reused a full payload segment"
        );
    }
    cold_checkpoint(projected, narrow, schema, context.job())
        .await
        .unwrap();
}

fn schema_digest_contracts() {
    assert_eq!(
        logical_digest(DataType::Utf8),
        "7354f1bc5010a1b05ac49c96f3d6c94e3167b14d2b2a4998c58866ab44d3c1d8",
        "retained-schema/v1 wire identity must stay stable"
    );
    assert_ne!(
        logical_digest(DataType::Timestamp(TimeUnit::Microsecond, None)),
        logical_digest(DataType::Timestamp(TimeUnit::Microsecond, Some("".into())))
    );
    let plain = DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8));
    let nested = DataType::Dictionary(
        Box::new(DataType::Int8),
        Box::new(DataType::Dictionary(
            Box::new(DataType::Int16),
            Box::new(DataType::Utf8),
        )),
    );
    assert_ne!(
        logical_digest(plain.clone()),
        logical_digest(nested.clone())
    );
    assert_ne!(
        logical_projection(plain).checkpoint_identity,
        logical_projection(nested).checkpoint_identity
    );
    assert_ne!(
        logical_projection(DataType::Timestamp(TimeUnit::Microsecond, None)).checkpoint_identity,
        logical_projection(DataType::Timestamp(TimeUnit::Microsecond, Some("".into())))
            .checkpoint_identity
    );
}

fn logical_digest(omitted: DataType) -> String {
    logical_projection(omitted)
        .payload_projection
        .as_ref()
        .unwrap()
        .descriptor
        .logical_schema_digests[0]
        .clone()
}

fn logical_projection(omitted: DataType) -> StreamJoinOperator {
    let schema = Arc::new(Schema::new(vec![
        Field::new("unused", omitted, true),
        Field::new("key", DataType::Int64, false),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("value", DataType::Int64, true),
    ]));
    operator(&schema, true)
}

async fn projected_checkpoint_layout() {
    let input = input(
        Arc::new(Int64Array::from(vec![10, 20, 30])),
        Arc::new(Int64Array::from(vec![7, 8, 9])),
    );
    let schema = input.table_payload().unwrap().batches()[0].schema();
    let mut projected = operator(&schema, true);
    let job = StreamJobContext::new(
        1,
        "projected-checkpoint",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "selected-state", None);
    let mut collector = EdgeCollector::new(projected.output_ports().to_vec());
    projected
        .process_data("right", input.clone(), &context, &mut collector)
        .await
        .unwrap();
    let checkpoint =
        checkpoint_cycles(&mut projected, &input, &context, &mut collector, &schema).await;
    let pool = projected
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    drop((projected, collector, context));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
    assert_eq!(checkpoint.unwrap(), 3);
}

async fn checkpoint_cycles(
    projected: &mut StreamJoinOperator,
    input: &Batch,
    context: &StreamOperatorContext<'_>,
    collector: &mut EdgeCollector,
    schema: &Arc<Schema>,
) -> crate::Result<serde_json::Value> {
    projected.prepare_checkpoint_async(context).await?;
    let first = projected.checkpoint(crate::Epoch::INITIAL)?;
    let layout = cold_checkpoint(projected, first, schema, context.job()).await?;
    for step in 2..=6 {
        checkpoint_step(projected, input, context, collector, schema, step).await?;
    }
    Ok(layout)
}

async fn checkpoint_step(
    projected: &mut StreamJoinOperator,
    input: &Batch,
    context: &StreamOperatorContext<'_>,
    collector: &mut EdgeCollector,
    schema: &Arc<Schema>,
    step: u64,
) -> crate::Result<()> {
    projected
        .process_data("left", input.clone(), context, collector)
        .await?;
    projected.prepare_checkpoint_async(context).await?;
    let epoch = crate::Epoch::new(step).unwrap();
    let snapshot = projected.checkpoint(epoch)?;
    let expected_deltas = if step == 6 {
        0
    } else {
        usize::try_from(step - 1).unwrap()
    };
    assert_eq!(
        snapshot.inline_metadata["v2_inventory"]["deltas"]
            .as_array()
            .unwrap()
            .len(),
        expected_deltas
    );
    if matches!(step, 2 | 6) {
        cold_checkpoint(projected, snapshot, schema, context.job()).await?;
    }
    Ok(())
}

async fn cold_checkpoint(
    source: &StreamJoinOperator,
    mut snapshot: crate::OperatorStateSnapshot,
    schema: &Arc<Schema>,
    job: &StreamJobContext,
) -> crate::Result<serde_json::Value> {
    let layout = snapshot.inline_metadata["layout_version"].clone();
    let mut restored = operator(schema, true);
    restored.restore(&snapshot)?;
    assert_eq!(restored.status(), source.status());
    let mut managed = operator(schema, true);
    managed
        .restore_managed_snapshot(&mut snapshot, job, None)
        .await?;
    assert_eq!(managed.status(), source.status());
    let direct_pool = restored.runtime.runtime()?.incremental_memory_pool();
    let managed_pool = managed.runtime.runtime()?.incremental_memory_pool();
    drop((restored, managed, snapshot));
    assert_eq!(direct_pool.reserved(), 0);
    assert_eq!(managed_pool.reserved(), 0);
    Ok(layout)
}

fn view_values() -> ArrayRef {
    let huge = "discarded-backing".repeat(8192);
    let values = StringViewArray::from(vec![
        huge.as_str(),
        "retained-long-view-one",
        "retained-long-view-two",
        "retained-long-view-three",
        huge.as_str(),
    ]);
    Arc::new(values.slice(1, 3))
}

fn input(value: ArrayRef, keys: ArrayRef) -> Batch {
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from_iter_values([
            "unread".repeat(1024),
            "unused".repeat(1024),
            "padding".repeat(1024),
        ])),
        keys,
        Arc::new(TimestampMicrosecondArray::from(vec![100, 100, 100]).with_timezone("UTC")),
        value,
    ];
    let fields = ["unused", "key", "time", "value"]
        .into_iter()
        .zip(&arrays)
        .map(|(name, array)| Field::new(name, array.data_type().clone(), true))
        .collect::<Vec<_>>();
    let record = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap();
    Batch::table(
        vec![record.slice(0, 2), record.slice(2, 1)],
        BatchMetadata::default(),
    )
    .unwrap()
}

fn operator(schema: &Arc<Schema>, projected: bool) -> StreamJoinOperator {
    let spec = StreamJoinSpec::inner(
        ["key"],
        ["key"],
        "time",
        "time",
        JoinTimeBounds::new(Duration::ZERO, Duration::ZERO).unwrap(),
        JoinStateLimits::new(100, 1_000_000, 1000).unwrap(),
    )
    .unwrap();
    let mut operator = StreamJoinOperator::new(
        "selected-state",
        Arc::clone(schema),
        Arc::clone(schema),
        spec,
    )
    .unwrap();
    if projected {
        operator.set_output_projection(vec![3, 7]).unwrap();
    }
    operator
}

async fn selected_case(value: ArrayRef, keys: ArrayRef, first: &str) {
    let input = input(value, keys);
    selected_batch_case(input, first, |_| true).await;
}

async fn selected_batch_case(input: Batch, first: &str, check: fn(&StreamJoinOperator) -> bool) {
    let original = input.table_payload().unwrap().batches().to_vec();
    let schema = original[0].schema();
    let mut projected = operator(&schema, true);
    let mut full = operator(&schema, false);
    let job = StreamJobContext::new(
        1,
        "selected-state",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "selected-state", None);
    let mut actual = EdgeCollector::new(projected.output_ports().to_vec());
    let mut expected = EdgeCollector::new(full.output_ports().to_vec());
    let second = if first == "left" { "right" } else { "left" };
    for side in [first, second] {
        projected
            .process_data(side, input.clone(), &context, &mut actual)
            .await
            .unwrap();
        full.process_data(side, input.clone(), &context, &mut expected)
            .await
            .unwrap();
        assert_eq!(projected.status(), full.status());
    }
    assert_eq!(input.table_payload().unwrap().batches(), original);
    compare_output(&mut actual, &mut expected);
    let independent = selected_owners(&projected, &original);
    let selected_values = check(&projected);
    let pool = projected
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    drop((projected, full, actual, expected, context));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
    assert!(independent, "retained payload still shares caller backing");
    assert!(
        selected_values,
        "retained dictionary contains unused values"
    );
}

fn compare_output(actual: &mut EdgeCollector, expected: &mut EdgeCollector) {
    let actual = actual.drain("output");
    let expected = expected.drain("output");
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(&expected) {
        let actual = actual.as_data().unwrap().table_payload().unwrap();
        let expected = expected.as_data().unwrap().table_payload().unwrap();
        assert_eq!(actual.batches().len(), expected.batches().len());
        for (actual, expected) in actual.batches().iter().zip(expected.batches()) {
            assert_eq!(actual, &expected.project(&[3, 7]).unwrap());
        }
    }
}

fn selected_owners(operator: &StreamJoinOperator, originals: &[RecordBatch]) -> bool {
    let mut independent = true;
    let original = originals
        .iter()
        .flat_map(|record| {
            record
                .columns()
                .iter()
                .flat_map(|array| buffers(&array.to_data()))
        })
        .collect::<Vec<_>>();
    for row in operator
        .state
        .left
        .iter()
        .chain(operator.state.right.iter())
    {
        assert!(row.record.funded_owner().is_some());
        assert_eq!(
            row.record
                .schema_ref()
                .fields()
                .iter()
                .map(|field| field.name().as_str())
                .collect::<Vec<_>>(),
            ["key", "time", "value"]
        );
        for array in row.record.columns() {
            for buffer in buffers(&array.to_data()) {
                for source in &original {
                    independent &= !overlap(&buffer, source);
                }
            }
        }
    }
    independent
}

fn buffers(data: &arrow_data::ArrayData) -> Vec<Buffer> {
    data.buffers()
        .iter()
        .cloned()
        .chain(data.nulls().map(|nulls| nulls.buffer().clone()))
        .chain(data.child_data().iter().flat_map(buffers))
        .collect()
}

fn overlap(a: &Buffer, b: &Buffer) -> bool {
    let start = a.as_ptr() as usize;
    let other = b.as_ptr() as usize;
    !a.is_empty() && !b.is_empty() && start < other + b.len() && other < start + a.len()
}
