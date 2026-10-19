use super::*;
use crate::{ExpressionOperator, PipelineBuilder, StreamRequirements, UdfRegistry};

struct PendingSource;

#[async_trait]
impl StreamSource for PendingSource {
    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replay_positioning: ReplayPositioning::ExactPauseReportAndSeek,
            delivery: SourceDeliveryCapability::Lossless,
            max_batch_rows: 1,
            max_batch_bytes: 1024,
            schema: SourceSchema::DynamicOrUnknown,
            native_watermarks: NativeWatermarkCapability::EmitsNative,
        }
    }

    async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        std::future::pending().await
    }

    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

struct NoopSink;

#[async_trait]
impl StreamSink for NoopSink {
    async fn open(&mut self) -> Result<()> {
        Ok(())
    }

    async fn write(&mut self, _batch: &Batch) -> Result<()> {
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

fn runner(checkpoint: Option<ManagedCheckpointRuntime>) -> StreamingRunner {
    let plan = PipelineBuilder::new("checkpoint-mode")
        .unwrap()
        .add_node(
            "forward",
            Box::new(
                ExpressionOperator::new("forward", "", vec!["value".into()], None, Vec::new())
                    .unwrap(),
            ),
        )
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap();
    let sources = BTreeMap::from([("input".into(), SourceBinding::new(PendingSource))]);
    let sinks = BTreeMap::from([(
        "output".into(),
        vec![SinkBinding::ordinary("sink", NoopSink).unwrap()],
    )]);
    match checkpoint {
        Some(checkpoint) => StreamingRunner::new(plan, sources, sinks, checkpoint),
        None => StreamingRunner::without_checkpoints(plan, sources, sinks),
    }
    .unwrap()
}

#[test]
fn test_checkpointing_mode_is_fixed_in_each_owned_job_context() {
    let context = StreamJobContext::new(
        1,
        "fingerprint",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    assert!(context.checkpointing());
    let disabled = context.clone().with_checkpointing(false);
    assert!(!disabled.checkpointing());
    assert!(!disabled.clone().checkpointing());
    assert!(context.checkpointing());
}

#[test]
fn test_checkpointing_hash_records_disabled_mode_without_changing_the_plan() {
    let runner = runner(None);
    let fingerprint = runner.plan.fingerprint().to_owned();
    let historical = StreamRuntimeConfig {
        edge_budget: crate::EdgeBudget::new(10_000, 64 << 20).unwrap(),
        ..StreamRuntimeConfig::default()
    };
    let enabled = runner.plan.runtime_config_hash(&historical).unwrap();
    let disabled = runner
        .plan
        .runtime_config_hash(&StreamRuntimeConfig {
            checkpointing: false,
            ..historical
        })
        .unwrap();
    assert_eq!(
        enabled,
        "fd02188bac4dfdca45ddd37267c08da51ea4a47c562edb06236b59b056f6e4bb"
    );
    assert_ne!(enabled, disabled);
    assert_ne!(
        enabled,
        runner
            .plan
            .runtime_config_hash(&StreamRuntimeConfig::default())
            .unwrap()
    );
    assert_eq!(runner.plan.fingerprint(), fingerprint);
}
