use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use calc_flow::StateSegment;

use async_trait::async_trait;
use calc_flow::{
    Batch, BatchKind, DeliveryGuarantee, EventTime, JsonMap, OperatorMetadata, PipelineBuilder,
    Port, Result, StreamCollector, StreamExecutionPlan, StreamOperator, StreamOperatorContext,
    StreamOperatorLifecycle, StreamRequirements, UdfRegistry,
};

#[test]
fn test_sequence_segment_clones_share_payload_and_digest_allocations() {
    let segment = StateSegment::new(vec![1, 2, 3, 4]);
    let cloned = segment.clone();
    assert!(Arc::ptr_eq(&segment.bytes_arc(), &cloned.bytes_arc()));
    assert!(std::ptr::eq(
        segment.sha256().as_ptr(),
        cloned.sha256().as_ptr()
    ));
    assert_eq!(segment, cloned);
}

struct ProofOperator {
    proof: StreamOperatorLifecycle,
    inputs: Vec<Port>,
    outputs: Vec<Port>,
    restore_calls: Arc<AtomicUsize>,
}

impl OperatorMetadata for ProofOperator {
    fn name(&self) -> &'static str {
        "sequence-proof"
    }
    fn input_ports(&self) -> &[Port] {
        &self.inputs
    }
    fn output_ports(&self) -> &[Port] {
        &self.outputs
    }
    fn configuration(&self) -> JsonMap {
        JsonMap::new()
    }
}

#[async_trait]
impl StreamOperator for ProofOperator {
    fn lifecycle(&self) -> StreamOperatorLifecycle {
        self.proof
    }
    fn restore(&mut self, snapshot: &calc_flow::OperatorStateSnapshot) -> Result<()> {
        self.restore_calls.fetch_add(1, Ordering::SeqCst);
        if !snapshot.inline_metadata.is_empty() {
            return Err(calc_flow::CalcFlowError::CheckpointMismatch {
                message: "restore-probe".into(),
            });
        }
        Ok(())
    }
    async fn process_data(
        &mut self,
        _ingress: &str,
        batch: Batch,
        _context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        output.emit("output", batch).await
    }
    async fn on_watermark(
        &mut self,
        _watermark: EventTime,
        _context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        Ok(())
    }
    async fn on_end(
        &mut self,
        _context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        Ok(())
    }
}

fn compile_proof(
    proof: StreamOperatorLifecycle,
    exactly_once: bool,
) -> Result<StreamExecutionPlan> {
    let operator: Box<dyn StreamOperator> = Box::new(ProofOperator {
        proof,
        inputs: vec![Port::new("input", BatchKind::Array, true, None)?],
        outputs: vec![Port::new("output", BatchKind::Array, true, None)?],
        restore_calls: Arc::new(AtomicUsize::new(0)),
    });
    let requirements = StreamRequirements {
        delivery: if exactly_once {
            std::collections::BTreeMap::from([("output".into(), DeliveryGuarantee::ExactlyOnce)])
        } else {
            std::collections::BTreeMap::new()
        },
    };
    PipelineBuilder::new("native-sequence-proof")?
        .add_node("sequence", operator)?
        .compile_stream(&UdfRegistry::new().snapshot(), &requirements)
}

#[test]
fn test_sequence_checkpointed_proof_requires_positive_version_and_microbatch_invariance() {
    let valid = StreamOperatorLifecycle::CheckpointedStateful {
        state_version: 1,
        microbatch_invariant: true,
        deterministic: true,
        replay_safe: true,
    };
    assert!(compile_proof(valid, false).is_ok());
    assert!(
        compile_proof(
            StreamOperatorLifecycle::CheckpointedStateful {
                state_version: 0,
                microbatch_invariant: true,
                deterministic: true,
                replay_safe: true
            },
            false
        )
        .is_err()
    );
    assert!(
        compile_proof(
            StreamOperatorLifecycle::CheckpointedStateful {
                state_version: 1,
                microbatch_invariant: false,
                deterministic: true,
                replay_safe: true
            },
            false
        )
        .is_err()
    );
}

#[test]
fn test_sequence_checkpointed_proof_keeps_exactly_once_determinism_and_replay_checks() {
    for (deterministic, replay_safe, accepted) in [
        (true, true, true),
        (false, true, false),
        (true, false, false),
    ] {
        let proof = StreamOperatorLifecycle::CheckpointedStateful {
            state_version: 1,
            microbatch_invariant: true,
            deterministic,
            replay_safe,
        };
        assert_eq!(compile_proof(proof, true).is_ok(), accepted);
    }
}

#[tokio::test]
async fn test_sequence_default_async_restore_preserves_sync_restore_and_checks_job_first() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut operator = ProofOperator {
        proof: StreamOperatorLifecycle::Unproven,
        inputs: vec![],
        outputs: vec![],
        restore_calls: calls.clone(),
    };
    let token = calc_flow::CancellationToken::new();
    let job =
        calc_flow::StreamJobContext::new(88, "restore-probe", JsonMap::new(), None, token.clone());
    let context = StreamOperatorContext::new(&job, "sequence", None);
    let snapshot = calc_flow::OperatorStateSnapshot {
        inline_metadata: std::collections::BTreeMap::from([("probe".into(), true.into())]),
        ..Default::default()
    };
    assert!(matches!(operator.restore_async(&snapshot, &context).await,
        Err(calc_flow::CalcFlowError::CheckpointMismatch { message }) if message == "restore-probe"));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    token.cancel();
    assert!(
        matches!(operator.restore_async(&calc_flow::OperatorStateSnapshot::default(), &context).await,
        Err(calc_flow::CalcFlowError::Cancelled { run_id }) if run_id == "88")
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}
