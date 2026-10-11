use async_trait::async_trait;
use calc_flow::{
    Batch, CalcFlowError, Cursor, ExpressionOperator, JsonMap, ManagedCheckpointRuntime,
    NativeWatermarkCapability, PipelineBuilder, ReplayPositioning, Result, SinkBinding,
    SinkRecovery, SourceBinding, SourceCapabilities, SourceDeliveryCapability, SourceEvent,
    SourceSchema, StreamRequirements, StreamSource, StreamingErrorCategory, StreamingRunner,
    TransactionalStreamSink, UdfRegistry,
};
use std::{collections::BTreeMap, time::Duration};

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

struct FailingCommitSink;

#[async_trait]
impl TransactionalStreamSink for FailingCommitSink {
    async fn open(&mut self) -> Result<()> {
        Ok(())
    }

    async fn begin_epoch(&mut self, _epoch: calc_flow::Epoch) -> Result<()> {
        Ok(())
    }

    async fn write(&mut self, _batch: &Batch) -> Result<()> {
        Ok(())
    }

    async fn pre_commit(&mut self, _epoch: calc_flow::Epoch) -> Result<JsonMap> {
        Ok(JsonMap::new())
    }

    async fn commit(&mut self, _epoch: calc_flow::Epoch, _pre_commit: &JsonMap) -> Result<()> {
        Err(CalcFlowError::Internal {
            message: "credential-canary".into(),
        })
    }

    async fn abort(
        &mut self,
        _epoch: calc_flow::Epoch,
        _pre_commit: Option<&JsonMap>,
    ) -> Result<()> {
        Ok(())
    }

    async fn recover(&mut self, _recovery: &SinkRecovery) -> Result<()> {
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

struct FailingPreCommitSink;

#[async_trait]
impl TransactionalStreamSink for FailingPreCommitSink {
    async fn open(&mut self) -> Result<()> {
        Ok(())
    }

    async fn begin_epoch(&mut self, _epoch: calc_flow::Epoch) -> Result<()> {
        Ok(())
    }

    async fn write(&mut self, _batch: &Batch) -> Result<()> {
        Ok(())
    }

    async fn pre_commit(&mut self, _epoch: calc_flow::Epoch) -> Result<JsonMap> {
        Err(CalcFlowError::Internal {
            message: "credential-canary".into(),
        })
    }

    async fn commit(&mut self, _epoch: calc_flow::Epoch, _pre_commit: &JsonMap) -> Result<()> {
        Ok(())
    }

    async fn abort(
        &mut self,
        _epoch: calc_flow::Epoch,
        _pre_commit: Option<&JsonMap>,
    ) -> Result<()> {
        Ok(())
    }

    async fn recover(&mut self, _recovery: &SinkRecovery) -> Result<()> {
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

struct PanickingPreCommitSink;

#[async_trait]
impl TransactionalStreamSink for PanickingPreCommitSink {
    async fn open(&mut self) -> Result<()> {
        Ok(())
    }

    async fn begin_epoch(&mut self, _epoch: calc_flow::Epoch) -> Result<()> {
        Ok(())
    }

    async fn write(&mut self, _batch: &Batch) -> Result<()> {
        Ok(())
    }

    async fn pre_commit(&mut self, _epoch: calc_flow::Epoch) -> Result<JsonMap> {
        panic!("credential-canary")
    }

    async fn commit(&mut self, _epoch: calc_flow::Epoch, _pre_commit: &JsonMap) -> Result<()> {
        Ok(())
    }

    async fn abort(
        &mut self,
        _epoch: calc_flow::Epoch,
        _pre_commit: Option<&JsonMap>,
    ) -> Result<()> {
        Ok(())
    }

    async fn recover(&mut self, _recovery: &SinkRecovery) -> Result<()> {
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

#[tokio::test]
async fn manual_checkpoint_sink_commit_failure_keeps_connector_coordinates() {
    let plan = PipelineBuilder::new("checkpoint-sink-failure")
        .unwrap()
        .add_node(
            "operator",
            Box::new(
                ExpressionOperator::new("operator", "value = value", Vec::new(), None, Vec::new())
                    .unwrap(),
            ),
        )
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap();
    let source_id = plan.source_binding_ids()[0].to_owned();
    let output_id = plan.sink_binding_ids()[0].to_owned();
    let directory = tempfile::tempdir().unwrap();
    let job = StreamingRunner::new(
        plan,
        BTreeMap::from([(source_id, SourceBinding::new(PendingSource))]),
        BTreeMap::from([(
            output_id,
            vec![SinkBinding::transactional("archive", FailingCommitSink).unwrap()],
        )]),
        ManagedCheckpointRuntime::new(directory.path().join("managed")).unwrap(),
    )
    .unwrap()
    .start()
    .await
    .unwrap();

    let error = tokio::time::timeout(Duration::from_secs(5), job.trigger_checkpoint())
        .await
        .expect("the sink commit failure must settle")
        .unwrap_err();
    let CalcFlowError::Streaming(error) = error else {
        panic!("manual checkpoint failures must use the streaming boundary");
    };

    assert_eq!(error.category(), StreamingErrorCategory::Connector);
    assert_eq!(error.component_kind(), Some(calc_flow::ComponentKind::Sink));
    assert_eq!(error.component_id(), Some("archive"));
    assert_eq!(error.epoch(), Some(calc_flow::Epoch::INITIAL));
    assert_eq!(
        error.checkpoint_phase(),
        Some(calc_flow::CheckpointPhase::ManifestDurable)
    );
    assert_eq!(
        error.message(),
        "sink \"archive\" commit failed for checkpoint epoch 1"
    );
    assert!(!format!("{error:?}").contains("credential-canary"));
    let outcome = job.wait().await;
    assert_eq!(outcome.state, calc_flow::JobState::RecoveryRequired);
    let outcome_error = outcome
        .errors
        .iter()
        .find(|error| error.component_id() == Some("archive"))
        .expect("the terminal outcome must retain the sink commit failure");
    assert_eq!(outcome_error.category(), StreamingErrorCategory::Connector);
    assert_eq!(outcome_error.epoch(), Some(calc_flow::Epoch::INITIAL));
    assert_eq!(
        outcome_error.checkpoint_phase(),
        Some(calc_flow::CheckpointPhase::ManifestDurable)
    );
    assert_eq!(outcome_error.message(), error.message());
}

#[tokio::test]
async fn accepted_manual_checkpoint_reports_precommit_connector_failure() {
    let plan = PipelineBuilder::new("checkpoint-precommit-failure")
        .unwrap()
        .add_node(
            "operator",
            Box::new(
                ExpressionOperator::new("operator", "value = value", Vec::new(), None, Vec::new())
                    .unwrap(),
            ),
        )
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap();
    let source_id = plan.source_binding_ids()[0].to_owned();
    let output_id = plan.sink_binding_ids()[0].to_owned();
    let directory = tempfile::tempdir().unwrap();
    let job = StreamingRunner::new(
        plan,
        BTreeMap::from([(source_id, SourceBinding::new(PendingSource))]),
        BTreeMap::from([(
            output_id,
            vec![SinkBinding::transactional("archive", FailingPreCommitSink).unwrap()],
        )]),
        ManagedCheckpointRuntime::new(directory.path().join("managed")).unwrap(),
    )
    .unwrap()
    .start()
    .await
    .unwrap();

    let error = tokio::time::timeout(Duration::from_secs(5), job.trigger_checkpoint())
        .await
        .expect("the sink pre-commit failure must settle")
        .unwrap_err();
    let CalcFlowError::Streaming(error) = error else {
        panic!("accepted manual failures must use the streaming boundary");
    };

    assert_eq!(error.category(), StreamingErrorCategory::Connector);
    assert_eq!(error.component_kind(), Some(calc_flow::ComponentKind::Sink));
    assert_eq!(error.component_id(), Some("archive"));
    assert_eq!(error.epoch(), Some(calc_flow::Epoch::INITIAL));
    assert!(!format!("{error:?}").contains("credential-canary"));
    let _ = job.wait().await;
}

#[tokio::test]
async fn accepted_manual_checkpoint_reports_contained_precommit_panic() {
    let plan = PipelineBuilder::new("checkpoint-precommit-panic")
        .unwrap()
        .add_node(
            "operator",
            Box::new(
                ExpressionOperator::new("operator", "value = value", Vec::new(), None, Vec::new())
                    .unwrap(),
            ),
        )
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap();
    let source_id = plan.source_binding_ids()[0].to_owned();
    let output_id = plan.sink_binding_ids()[0].to_owned();
    let directory = tempfile::tempdir().unwrap();
    let job = StreamingRunner::new(
        plan,
        BTreeMap::from([(source_id, SourceBinding::new(PendingSource))]),
        BTreeMap::from([(
            output_id,
            vec![SinkBinding::transactional("archive", PanickingPreCommitSink).unwrap()],
        )]),
        ManagedCheckpointRuntime::new(directory.path().join("managed")).unwrap(),
    )
    .unwrap()
    .start()
    .await
    .unwrap();

    let error = tokio::time::timeout(Duration::from_secs(5), job.trigger_checkpoint())
        .await
        .expect("the contained sink panic must settle")
        .unwrap_err();
    let CalcFlowError::Streaming(error) = error else {
        panic!("accepted manual failures must use the streaming boundary");
    };

    assert_eq!(error.category(), StreamingErrorCategory::TaskPanicked);
    assert_eq!(error.epoch(), Some(calc_flow::Epoch::INITIAL));
    assert!(!format!("{error:?}").contains("credential-canary"));
    let _ = job.wait().await;
}
