use std::sync::{
    Arc,
    atomic::{AtomicU32, Ordering},
};

use async_trait::async_trait;

use crate::{
    Batch, BatchKind, EventTime, JsonMap, OperatorMetadata, PipelineBuilder, Port, Result,
    StreamCollector, StreamOperator, StreamOperatorContext, StreamOperatorLifecycle,
    StreamRequirements,
};

use super::validate_external_provider_lifecycles;

#[test]
fn test_bounded_order_requires_explicit_sequence_for_strong_delivery() {
    use crate::{
        BoundedOrderOperator, DeliveryGuarantee, NullPlacement, SortDirection, SortKey, SortSpec,
        StateBudget,
    };
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
    let op = BoundedOrderOperator::new(
        "order",
        schema,
        vec![SortKey::new("x", SortDirection::Ascending, NullPlacement::Last).unwrap()],
        Some(1),
        SortSpec::bounded_source(StateBudget::new(10, 8 << 20).unwrap()).unwrap(),
    )
    .unwrap();
    let builder = PipelineBuilder::new("proof")
        .unwrap()
        .add_node("order", op)
        .unwrap();
    let requirements = StreamRequirements {
        delivery: std::collections::BTreeMap::from([(
            "output".into(),
            DeliveryGuarantee::ExactlyOnce,
        )]),
    };
    let result = validate_external_provider_lifecycles(&builder.nodes, &requirements);
    assert!(
        result.is_err(),
        "a built-in checkpoint codec must not bypass tie/replay proof"
    );
    assert!(result.unwrap_err().to_string().contains("ties"));
}

struct Proof {
    version: Arc<AtomicU32>,
    inputs: Vec<Port>,
    outputs: Vec<Port>,
}

impl OperatorMetadata for Proof {
    fn name(&self) -> &'static str {
        "native-proof-provenance"
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
impl StreamOperator for Proof {
    fn lifecycle(&self) -> StreamOperatorLifecycle {
        let state_version = self.version.load(Ordering::SeqCst);
        if state_version == 0 {
            return StreamOperatorLifecycle::Unproven;
        }
        StreamOperatorLifecycle::CheckpointedStateful {
            state_version,
            microbatch_invariant: true,
            deterministic: true,
            replay_safe: true,
        }
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

fn operator(version: Arc<AtomicU32>) -> Box<dyn StreamOperator> {
    Box::new(Proof {
        version,
        inputs: vec![Port::new("input", BatchKind::Array, true, None).unwrap()],
        outputs: vec![Port::new("output", BatchKind::Array, true, None).unwrap()],
    })
}

#[test]
fn test_only_explicit_test_injection_can_bypass_repeated_external_proof() {
    for changed in [0, 2] {
        let version = Arc::new(AtomicU32::new(1));
        let builder = PipelineBuilder::new("proof-provenance")
            .unwrap()
            .add_checkpoint_capable_node_with_version(
                "injected",
                operator(Arc::new(AtomicU32::new(0))),
                7,
            )
            .unwrap()
            .add_node("public", operator(version.clone()))
            .unwrap();
        assert!(builder.nodes["injected"].checkpoint_test_override);
        assert!(!builder.nodes["public"].checkpoint_test_override);
        assert!(
            validate_external_provider_lifecycles(&builder.nodes, &StreamRequirements::default())
                .is_ok()
        );
        version.store(changed, Ordering::SeqCst);
        assert!(
            matches!(validate_external_provider_lifecycles(&builder.nodes, &StreamRequirements::default()),
            Err(crate::CalcFlowError::Compile { message }) if message.contains("public") && message.contains("version 1"))
        );
    }
}
