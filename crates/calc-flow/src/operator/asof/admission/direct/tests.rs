use super::super::identity_tests::columnar::{direct::string_unsigned_input, fixture};
use super::*;
use crate::runtime::streaming::gather_work::TestService;
use crate::{CancellationToken, JsonMap, StreamJobContext, StreamOperator};

struct GateWork {
    work: ChunkWork,
    entered: tokio::sync::oneshot::Sender<()>,
    gate: std::sync::mpsc::Receiver<()>,
}

impl OwnedCpuWork for GateWork {
    type Output = PreparedInput;

    fn run(self, stop: &GatherStop) -> Result<PreparedInput> {
        stop.check()?;
        self.entered.send(()).unwrap();
        self.gate
            .recv_timeout(std::time::Duration::from_secs(10))
            .unwrap();
        self.work.run(stop)
    }
}

async fn abandoned(service: &TestService, cancel: bool) {
    let input = string_unsigned_input(4_097);
    let schema = input.table_payload().unwrap().batches()[0].schema();
    let mut operator = fixture(&schema);
    let pool = operator.runtime.pool.clone();
    let cancellation = CancellationToken::new();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, cancellation.clone())
        .with_gather_owner(service.owner("direct-left-abandon".into()));
    let context = StreamOperatorContext::new(&job, "asof", None);
    let validated = operator.validate_admission("left", &input).unwrap();
    let identity = operator.identity_workspace(&input, validated).unwrap();
    let identity_bytes = identity.reservation.size();
    let plan = plan(&operator, &input, validated, &context)
        .unwrap()
        .unwrap();
    let batches = encode_complete_payloads(&operator, &input).unwrap();
    let weak = Arc::downgrade(&batches[0]);
    let work = ChunkWork {
        keys: plan.keys,
        _descriptor: operator.reserve_left_work(batches.len()).unwrap(),
        batches,
        workspace: AdmissionWorkspace {
            _identity: identity.reservation,
            _payload: operator.input_workspace(&input, validated).unwrap(),
            _keys: None,
        },
        side: operator.spec.left().clone(),
        name: operator.name.clone(),
    };
    let (entered, started) = tokio::sync::oneshot::channel();
    let (release, gate) = std::sync::mpsc::channel();
    let mut pending = Box::pin(operator.run_cpu_work(
        GateWork {
            work,
            entered,
            gate,
        },
        &context,
    ));
    assert!(futures::poll!(pending.as_mut()).is_pending());
    started.await.unwrap();
    if cancel {
        cancellation.cancel();
    }
    drop(pending);
    assert_eq!(operator.status.left.accepted_rows, 0);
    operator.reset().unwrap();
    drop((operator, input));
    let retained = weak.upgrade().is_some() && pool.reserved() >= identity_bytes;
    let mut drain = Box::pin(job.gather_owner().close_and_drain());
    let waiting = futures::poll!(drain.as_mut()).is_pending();
    release.send(()).unwrap();
    assert!(drain.await.is_empty());
    assert!(
        retained,
        "running direct work lost its owner or identity credit"
    );
    assert!(waiting, "job drain passed a running direct preparation");
    drop(context);
    drop(job);
    assert!(weak.upgrade().is_none());
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn test_direct_left_cancelled_and_abandoned_work_retains_credit_until_drain() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    for cancel in [false, true] {
        runtime.block_on(abandoned(&service, cancel));
    }
    drop(runtime);
    service.shutdown();
}
