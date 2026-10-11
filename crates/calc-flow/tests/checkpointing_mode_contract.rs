use async_trait::async_trait;
use calc_flow::{
    Batch, CalcFlowError, Cursor, ExpressionOperator, ManagedCheckpointRuntime,
    NativeWatermarkCapability, PipelineBuilder, ReplayPositioning, Result, SinkBinding,
    SourceBinding, SourceCapabilities, SourceDeliveryCapability, SourceEvent, SourceSchema,
    StreamRequirements, StreamRuntimeConfig, StreamSink, StreamSource, StreamingErrorCategory,
    StreamingRunner, UdfRegistry,
};
use std::collections::BTreeMap;
struct PendingSource;
struct NoopSink;
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
fn test_checkpointing_defaults_on_and_rejects_backend_mode_conflicts() {
    assert!(StreamRuntimeConfig::default().checkpointing);
    let error = runner(None)
        .with_runtime_config(StreamRuntimeConfig::default())
        .err()
        .expect("enabling checkpoints requires managed storage");
    assert!(
        matches!(error, CalcFlowError::Streaming(error) if error.category() == StreamingErrorCategory::Validation)
    );

    let directory = tempfile::tempdir().unwrap();
    let error = runner(Some(
        ManagedCheckpointRuntime::new(directory.path().join("managed")).unwrap(),
    ))
    .with_runtime_config(StreamRuntimeConfig {
        checkpointing: false,
        ..StreamRuntimeConfig::default()
    })
    .err()
    .expect("disabling checkpoints cannot silently discard supplied storage");
    assert!(
        matches!(error, CalcFlowError::Streaming(error) if error.category() == StreamingErrorCategory::Validation)
    );
    assert!(!directory.path().join("managed").exists());
}
