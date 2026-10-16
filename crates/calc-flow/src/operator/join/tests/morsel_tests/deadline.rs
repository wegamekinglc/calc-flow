use super::*;
use crate::runtime::streaming::gather_work::GatherOperatorId;
use crate::{CalcFlowError, StreamJobContext};

pub(in crate::operator::join) async fn stop_morsels_at_deadline(
    operator: &mut StreamJoinOperator,
    mut prepared: PreparedJoinBatch,
    job: &StreamJobContext,
    entering: tokio::sync::oneshot::Receiver<()>,
) {
    let DeferredProbe {
        mut inputs,
        counts,
        fallback: _fallback,
    } = *prepared.deferred.take().expect("owned morsel preparation");
    inputs.admitted = std::mem::take(&mut prepared.admitted);
    let (released, receiver) = tokio::sync::oneshot::channel();
    inputs.released = Some(released);
    operator.compaction_release = Some(receiver);
    operator.probe_control.clone_from(&inputs.control);
    let schema = Arc::clone(operator.output_ports[0].schema().unwrap());
    let scratch = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_reservation("stream-join-morsel");
    scratch
        .try_grow(
            funding::scratch_bytes(counts.total, inputs.admitted.len(), schema.fields().len())
                .unwrap(),
        )
        .unwrap();
    let work = Arc::new(GatherWork {
        inputs: Arc::new(inputs),
        counts,
        schema,
        projection: operator.output_columns.clone(),
        charges: prepared.admitted_charges.take().unwrap(),
        budget: EdgeBudget::default(),
        name: operator.name.clone(),
        scratch,
    });
    let scope = job
        .gather_owner()
        .client(GatherOperatorId::new(operator.name.clone().into()))
        .scope()
        .unwrap();
    let ticket = scope
        .submit_observed_parallel_work(
            work,
            operator.probe_control.as_ref().unwrap().new_empty(),
            GatherStop::from_job(job),
            job.gather_owner().retain_retirement().unwrap(),
            &mut operator.compaction_cleanup,
        )
        .await
        .unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(5), entering)
        .await
        .unwrap()
        .unwrap();

    // Start deadline observation only after a real morsel worker holds scratch.
    // The handler uses this same ticket waiter; no preparation or queue latency
    // can consume a wall-clock allowance before the worker reaches its gate.
    let deadline_job = StreamJobContext::new(
        job.job_id(),
        job.fingerprint(),
        job.settings().clone(),
        Some(chrono::Utc::now() + chrono::Duration::milliseconds(50)),
        job.cancellation().clone(),
    )
    .with_gather_owner(job.gather_owner().clone());
    assert!(!deadline_job.cancellation().is_cancelled());
    let context = StreamOperatorContext::new(&deadline_job, &operator.name, None);
    assert!(matches!(
        tokio::time::timeout(
            std::time::Duration::from_secs(5),
            super::super::process::finish_probe_ticket(
                ticket,
                &context,
                |_| -> Result<()> { panic!("expired morsels must not install output") },
                None,
            ),
        )
        .await
        .unwrap(),
        Err(CalcFlowError::Cancelled { run_id }) if run_id == job.job_id().to_string()
    ));
}
