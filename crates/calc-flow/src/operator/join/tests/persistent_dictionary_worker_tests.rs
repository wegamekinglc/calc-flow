use super::*;
use crate::operator::key_identity::{KeyAddress, WeakIdentity};
use crate::runtime::streaming::gather_work::TestService;
use datafusion::execution::memory_pool::MemoryPool;
use std::sync::atomic::Ordering;

type Pool = persistent_dictionary_refusal_tests::DictionaryPool;

struct Proof {
    history: Arc<columnar::FramedKey>,
    identity: WeakIdentity<Arc<columnar::FramedKey>>,
    address: KeyAddress,
}

#[test]
fn test_persistent_dictionary_actual_probe_result_cancel_and_abandon_preserve_state() {
    checkpoint_compaction_tests::isolated_checkpoint_test(|service, runtime| {
        runtime.block_on(result_boundary(service, false));
        runtime.block_on(result_boundary(service, true));
    });
}

#[test]
fn test_persistent_dictionary_actual_probe_inputs_keep_rebuilt_legacy_identity() {
    checkpoint_compaction_tests::isolated_checkpoint_test(|service, runtime| {
        runtime.block_on(input_boundary(service, data_work::ProbePhase::Count));
        runtime.block_on(input_boundary(service, data_work::ProbePhase::Fill));
    });
}

async fn fixture(
    service: &TestService,
) -> (
    StreamJoinOperator,
    Arc<Pool>,
    StreamJobContext,
    EdgeCollector,
) {
    let mut declaration = spec();
    declaration.limits = JoinStateLimits::new(20_000, 10_000_000, 8_192).unwrap();
    let (mut operator, pool) = persistent_dictionary_refusal_tests::fixture_spec(declaration);
    data_work::relax_probe_cost_gate_for_test();
    let owner = job().with_gather_owner(service.owner("dictionary-worker-lease".into()));
    let context = StreamOperatorContext::new(&owner, "match", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("right", right_batch(vec![0]), &context, &mut output)
        .await
        .unwrap();
    pool.mode.store(1, Ordering::SeqCst);
    operator
        .process_data("left", keyed_left_batch(&[8], &[0]), &context, &mut output)
        .await
        .unwrap();
    operator
        .process_data(
            "right",
            keyed_right_batch(&[8], &[0]),
            &context,
            &mut output,
        )
        .await
        .unwrap();
    pool.mode.store(0, Ordering::SeqCst);
    let progress = progress_context(
        &owner,
        (IngressState::Active, None),
        (IngressState::Active, None),
    );
    operator
        .on_ingress_progress("left", &progress)
        .await
        .unwrap();
    assert!(operator.key_bindings.left.get(0).is_some());
    assert!(operator.key_bindings.right.get(1).is_some());
    operator
        .process_data("left", keyed_left_batch(&[8], &[0]), &context, &mut output)
        .await
        .unwrap();
    assert_eq!(operator.status().emitted_match_rows, 2);
    output.drain("output");
    drop((context, progress));
    (operator, pool, owner, output)
}

fn proof(operator: &StreamJoinOperator) -> Proof {
    let retained = operator.state.right.at(1);
    let canonical = operator.key_bindings.right.canonical(&retained);
    let history = operator
        .state
        .deltas
        .pending
        .iter()
        .find_map(|op| match op {
            PendingOp::Upsert { encoded_key, .. } if Arc::ptr_eq(encoded_key, canonical) => {
                Some(Arc::clone(encoded_key))
            }
            _ => None,
        })
        .expect("actual checkpoint upsert shares the canonical byte-only owner");
    let lease = operator.key_dictionary.lease(&history).unwrap();
    Proof {
        history,
        identity: lease.downgrade(),
        address: lease.address(),
    }
}

async fn result_boundary(service: &TestService, cancel: bool) {
    let (mut operator, pool, owner, mut output) = fixture(service).await;
    let context = StreamOperatorContext::new(&owner, "match", None);
    let proof = proof(&operator);
    let before = operator.status();
    let inventory = operator.key_dictionary.inventory();
    let pending = operator.state.deltas.pending.iter().count();
    let gate = Arc::new(data_work::ResultTestGate::default());
    operator.probe_result_test_gate = Some(Arc::clone(&gate));
    let mut process = Box::pin(operator.process_data("left", incoming(), &context, &mut output));
    tokio::select! {
        () = gate.entered.notified() => {},
        result = &mut process => panic!("actual observed probe result was installed before its release gate: {result:?}"),
    }
    if cancel {
        owner.cancellation().cancel();
        assert!(matches!(
            process.as_mut().await,
            Err(CalcFlowError::Cancelled { .. })
        ));
    }
    drop(process);
    assert_eq!(operator.status(), before);
    assert_eq!(operator.key_dictionary.inventory(), inventory);
    assert_eq!(operator.state.deltas.pending.iter().count(), pending);
    assert!(output.drain("output").is_empty());
    operator.reset().unwrap();
    gate.release.notify_one();
    replay(&mut operator, &owner, &mut output, &proof).await;
    drop((operator, output, context, proof));
    assert!(owner.gather_owner().close_and_drain().await.is_empty());
    drop(owner);
    assert_eq!(pool.reserved(), 0);
}

async fn input_boundary(service: &TestService, phase: data_work::ProbePhase) {
    let (mut operator, pool, owner, mut output) = fixture(service).await;
    let context = StreamOperatorContext::new(&owner, "match", None);
    let proof = proof(&operator);
    let before = operator.status();
    let pending = operator.state.deltas.pending.iter().count();
    let (entered, started) = tokio::sync::oneshot::channel();
    let (release, wait) = std::sync::mpsc::channel();
    let gate = std::sync::Mutex::new(Some((entered, wait)));
    operator.probe_test_hook = Some(Arc::new(move |actual, _, _| {
        if actual == phase {
            let (entered, wait) = gate.lock().unwrap().take().unwrap();
            entered.send(()).unwrap();
            wait.recv_timeout(Duration::from_secs(10)).unwrap();
        }
    }));
    let mut process = Box::pin(operator.process_data("left", incoming(), &context, &mut output));
    tokio::select! {
        result = started => result.unwrap(),
        result = &mut process => panic!("probe input escaped its actual worker gate: {result:?}"),
    }
    if phase == data_work::ProbePhase::Fill {
        owner.cancellation().cancel();
        assert!(matches!(
            process.as_mut().await,
            Err(CalcFlowError::Cancelled { .. })
        ));
    }
    drop(process);
    assert_eq!(operator.status(), before);
    assert_eq!(operator.state.deltas.pending.iter().count(), pending);
    operator.reset().unwrap();
    assert!(
        proof.identity.strong_count() > 0,
        "only actual captured worker owners can keep the retired identity leased after reset"
    );
    assert!(operator.key_dictionary.lease(&proof.history).is_none());
    let resumed_job = job().with_gather_owner(owner.gather_owner().clone());
    let resumed = StreamOperatorContext::new(&resumed_job, "match", None);
    let mut mutation = Box::pin(operator.process_data(
        "right",
        keyed_right_batch(&[8], &[0]),
        &resumed,
        &mut output,
    ));
    assert!(futures::poll!(mutation.as_mut()).is_pending());
    drop(mutation);
    drop(resumed);
    drop(resumed_job);
    release.send(()).unwrap();
    replay(&mut operator, &owner, &mut output, &proof).await;
    drop((operator, output, context, proof));
    assert!(owner.gather_owner().close_and_drain().await.is_empty());
    drop(owner);
    assert_eq!(pool.reserved(), 0);
}

fn incoming() -> Batch {
    keyed_left_batch(&vec![8; 8192], &vec![0; 8192])
}

async fn replay(
    operator: &mut StreamJoinOperator,
    owner: &StreamJobContext,
    output: &mut EdgeCollector,
    proof: &Proof,
) {
    let retry_job = job().with_gather_owner(owner.gather_owner().clone());
    let retry = StreamOperatorContext::new(&retry_job, "match", None);
    operator
        .process_data("right", keyed_right_batch(&[8], &[0]), &retry, output)
        .await
        .unwrap();
    assert_eq!(
        proof.identity.strong_count(),
        0,
        "actual input/result release retires its identity lease despite byte-only checkpoint history"
    );
    let current = operator.state.right.key_at(0);
    assert_ne!(
        proof.address,
        operator.key_dictionary.lease(current).unwrap().address()
    );
    assert_eq!(proof.history.as_slice(), current.as_slice());
    assert!(operator.key_dictionary.lease(&proof.history).is_none());
    operator
        .process_data("left", incoming(), &retry, output)
        .await
        .unwrap();
    assert_eq!(operator.status().emitted_match_rows, 8192);
    assert_eq!(operator.key_dictionary.inventory().1, 8193);
    assert!(
        operator
            .state
            .left
            .iter()
            .chain(operator.state.right.iter())
            .all(|row| row.charge == 124)
    );
    assert_eq!(
        output.drain("output")[0].as_data().unwrap().num_rows(),
        8192
    );
}
