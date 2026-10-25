use super::*;
use futures::FutureExt;

#[derive(Clone, Copy, Debug)]
enum Route {
    Direct,
    Managed,
    Projected,
    Refused,
    Closed,
}

#[tokio::test]
async fn test_persistent_dictionary_restore_rejects_duplicate_side_row_ids() {
    let routes = [
        Route::Direct,
        Route::Managed,
        Route::Projected,
        Route::Refused,
        Route::Closed,
    ];
    check_duplicate_routes(&routes, false).await;
}

#[tokio::test]
async fn test_persistent_dictionary_duplicate_ids_preserve_prior_payload_error() {
    check_duplicate_routes(&[Route::Direct, Route::Managed], true).await;
}

async fn check_duplicate_routes(routes: &[Route], bad_charge: bool) {
    let owner = job();
    let context = StreamOperatorContext::new(&owner, "match", None);
    let (mut source, snapshot) = duplicate_snapshot(&context).await;
    let source_pool = source.runtime.runtime().unwrap().incremental_memory_pool();
    drop(source);
    let mut outcomes = Vec::new();
    for route in routes {
        outcomes.push(Box::pin(reject_duplicate_snapshot(&snapshot, *route, bad_charge)).await);
    }
    drop((snapshot, context));
    assert!(owner.gather_owner().close_and_drain().await.is_empty());
    drop(owner);
    assert_eq!(source_pool.reserved(), 0);
    for (route, typed, unchanged, healthy, credit, untouched, detail) in outcomes {
        assert!(
            typed,
            "duplicate-ID restore must reject, {route:?}: {detail}"
        );
        assert!(
            unchanged,
            "installed state/writer/history changed: {route:?}"
        );
        assert!(healthy, "old native state must remain usable: {route:?}");
        assert!(credit, "candidate credit must retire: {route:?}");
        assert!(
            untouched,
            "required validation must precede dictionary grant: {route:?}"
        );
    }
}

async fn duplicate_snapshot(
    context: &StreamOperatorContext<'_>,
) -> (StreamJoinOperator, OperatorStateSnapshot) {
    let mut source = operator(false);
    source.checkpoint_v1_test_producer = true;
    let mut output = EdgeCollector::new(source.output_ports().to_vec());
    for time in [100, 101] {
        source
            .process_data("left", timed_batch(time), context, &mut output)
            .await
            .unwrap();
    }
    source.state.deltas.needs_compaction = true;
    source.prepare_checkpoint_async(context).await.unwrap();
    let mut snapshot = source.checkpoint(Epoch::INITIAL).unwrap();
    assert_eq!(snapshot.inline_metadata["layout_version"], 1);
    let mut valid = operator(false);
    valid.restore(&snapshot).unwrap();
    assert_eq!(valid.state.left.len(), 2);
    assert_eq!(valid.state.left.at(0).row_id, 0);
    assert_eq!(valid.state.left.at(1).row_id, 1);
    assert_fees(&valid, 2, 0);
    let pool = valid.runtime.runtime().unwrap().incremental_memory_pool();
    drop(valid);
    assert_eq!(pool.reserved(), 0);
    let mut bytes = snapshot.segments["left-base"].bytes().to_vec();
    assert_eq!(&bytes[..8], JOIN_STATE_MAGIC);
    let mut offset = 8;
    assert_eq!(read_segment_u64(&bytes, &mut offset).unwrap(), 2);
    assert_eq!(read_segment_u64(&bytes, &mut offset).unwrap(), 0);
    assert_eq!(read_segment_i64(&bytes, &mut offset).unwrap(), 100);
    assert_eq!(read_segment_u64(&bytes, &mut offset).unwrap(), 133);
    let ipc = usize::try_from(read_segment_u64(&bytes, &mut offset).unwrap()).unwrap();
    offset += ipc;
    let duplicate = offset;
    assert_eq!(read_segment_u64(&bytes, &mut offset).unwrap(), 1);
    assert_eq!(read_segment_i64(&bytes, &mut offset).unwrap(), 101);
    assert_eq!(read_segment_u64(&bytes, &mut offset).unwrap(), 133);
    bytes[duplicate..duplicate + 8].copy_from_slice(&0_u64.to_le_bytes());
    snapshot
        .segments
        .insert("left-base".into(), StateSegment::new(bytes));
    drop(output);
    (source, snapshot)
}

fn timed_batch(time: i64) -> Batch {
    let original = batch(7, 20);
    let record = &original.table_payload().unwrap().batches()[0];
    let mut columns = record.columns().to_vec();
    columns[2] = Arc::new(TimestampMicrosecondArray::from(vec![time]).with_timezone("UTC"));
    Batch::table(
        vec![RecordBatch::try_new(record.schema(), columns).unwrap()],
        BatchMetadata::default(),
    )
    .unwrap()
}

async fn installed_target(
    context: &StreamOperatorContext<'_>,
    projected: bool,
) -> (StreamJoinOperator, Arc<DictionaryPool>, EdgeCollector) {
    let mut target = operator(projected);
    target.checkpoint_v1_test_producer = !projected;
    let pool = Arc::new(DictionaryPool::default());
    target
        .runtime
        .initialize_benchmark_pool(pool.clone())
        .unwrap();
    let mut output = EdgeCollector::new(target.output_ports().to_vec());
    target
        .process_data("right", batch(9, 99), context, &mut output)
        .await
        .unwrap();
    target.state.deltas.needs_compaction = true;
    target.prepare_checkpoint_async(context).await.unwrap();
    target.checkpoint(Epoch::INITIAL).unwrap();
    target
        .process_data("right", batch(9, 99), context, &mut output)
        .await
        .unwrap();
    assert_eq!(target.state.deltas.pending_count_for_test(), 1);
    (target, pool, output)
}

type Outcome = (Route, bool, bool, bool, bool, bool, String);

#[derive(Clone, Copy, Debug)]
struct FundingObservation {
    pool: usize,
    state: usize,
    writer: usize,
    controls: (usize, usize, usize),
}

impl FundingObservation {
    fn observe(
        target: &StreamJoinOperator,
        pool: &DictionaryPool,
        owner: &StreamJobContext,
    ) -> Self {
        Self {
            pool: pool.reserved(),
            state: native_lookup_tests::state_funding(target),
            writer: target.v2_writer.base_funding_for_test(),
            controls: owner.gather_owner().funding(),
        }
    }

    fn accounted(self) -> bool {
        let (home, generation, attempt) = self.controls;
        self.pool == self.state + self.writer + home + generation + attempt
    }
}

fn unchanged_funding(route: Route, before: FundingObservation, after: FundingObservation) -> bool {
    let controls = if matches!(route, Route::Closed) {
        before.controls.2 == 0 && after.controls == (before.controls.0, 0, 0)
    } else {
        before.controls == after.controls
    };
    before.accounted()
        && after.accounted()
        && (before.state, before.writer) == (after.state, after.writer)
        && controls
}

async fn reject_duplicate_snapshot(
    snapshot: &OperatorStateSnapshot,
    route: Route,
    bad_charge: bool,
) -> Outcome {
    let owner = job();
    let context = StreamOperatorContext::new(&owner, "match", None);
    let (mut target, pool, mut output) =
        installed_target(&context, matches!(route, Route::Projected)).await;
    let before = state_observation(&target);
    let history = history_observation(&target);
    let writer = target.v2_writer.observation_for_test();
    let old = canonical_right(&target);
    pool.refuse
        .store(matches!(route, Route::Refused), Ordering::SeqCst);
    if matches!(route, Route::Closed) {
        owner.gather_owner().close_admission();
        owner.gather_owner().wait_closed_generation_for_test().await;
    }
    // Global capacity pressure may retire the checkpoint's reusable executor.
    // Compare refunds after the same retirement boundary on both sides.
    owner.gather_owner().retire_idle_generation_for_test().await;
    let baseline = FundingObservation::observe(&target, &pool, &owner);
    let mut candidate = snapshot.clone();
    if bad_charge {
        let mut bytes = candidate.segments["left-base"].bytes().to_vec();
        bytes[32..40].copy_from_slice(&134_u64.to_le_bytes());
        candidate
            .segments
            .insert("left-base".into(), StateSegment::new(bytes));
    }
    let managed = matches!(route, Route::Managed | Route::Closed);
    let result =
        std::panic::AssertUnwindSafe(restore_snapshot(&mut target, &candidate, &context, managed))
            .catch_unwind()
            .await;
    let typed = expected_rejection(&result, bad_charge);
    let detail = match result {
        Ok(result) => format!("{result:?}"),
        Err(_) => "panic".into(),
    };
    eprintln!("duplicate side row ID {route:?}, bad_charge={bad_charge}: {detail}");
    let unchanged = state_observation(&target) == before
        && history_observation(&target) == history
        && target.v2_writer.observation_for_test() == writer
        && Arc::ptr_eq(&old, &canonical_right(&target));
    owner.gather_owner().retire_idle_generation_for_test().await;
    let current = FundingObservation::observe(&target, &pool, &owner);
    let credit = unchanged_funding(route, baseline, current);
    eprintln!("duplicate credit {route:?}: before={baseline:?}, after={current:?}");
    let untouched = pool.refusals.load(Ordering::SeqCst) == 0;
    pool.refuse.store(false, Ordering::SeqCst);
    let healthy = target
        .process_data("left", batch(9, 40), &context, &mut output)
        .await
        .is_ok()
        && output
            .drain("output")
            .iter()
            .map(|message| message.as_data().unwrap().num_rows())
            .sum::<usize>()
            == 2;
    drop((
        target, output, old, before, history, writer, candidate, context,
    ));
    assert!(owner.gather_owner().close_and_drain().await.is_empty());
    drop(owner);
    assert_eq!(pool.reserved(), 0);
    (route, typed, unchanged, healthy, credit, untouched, detail)
}

fn expected_rejection(result: &std::thread::Result<Result<()>>, bad_charge: bool) -> bool {
    match result {
        Ok(Err(CalcFlowError::CheckpointMismatch { message })) if bad_charge => {
            message.contains("row event time or charge is inconsistent")
        }
        Ok(Err(CalcFlowError::CheckpointMismatch { message })) => {
            message.contains("live row ID is duplicated")
        }
        _ => false,
    }
}

fn history_observation(target: &StreamJoinOperator) -> impl PartialEq + fmt::Debug + use<> {
    (
        target
            .state
            .deltas
            .base
            .iter()
            .map(|(side, segment)| {
                (
                    *side,
                    Arc::as_ptr(&segment.bytes_arc()) as usize,
                    segment.bytes().to_vec(),
                )
            })
            .collect::<Vec<_>>(),
        target
            .state
            .deltas
            .segments
            .iter()
            .map(|(identity, segment)| {
                (
                    *identity,
                    Arc::as_ptr(&segment.bytes_arc()) as usize,
                    segment.bytes().to_vec(),
                )
            })
            .collect::<Vec<_>>(),
    )
}
