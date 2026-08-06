use std::collections::BTreeMap;

use crate::{CalcFlowError, Result};

use super::{
    aggregate::IngressActivity,
    driver::{DriverPhase, DriverSnapshotPayload},
    prepare::{
        BindingIdentity, BindingOrdinal, NormalizedConfigFingerprint, PreparedJobFingerprint,
        RuntimeFenceConfigFingerprint,
    },
    trace::{
        AdmissionGateSnapshot, InboxFenceCoordinate, ProgressExecutionTrace, ProgressReplayRequest,
        RawUpstreamPosition,
    },
    types::{CurrentTimer, DrainEpoch, DriverClockCoordinate, IdleEpoch, TimerIdentity},
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CapturedBindingCoordinate {
    pub(crate) identity: BindingIdentity,
    pub(crate) ordinal: BindingOrdinal,
    pub(crate) normalized_config_fingerprint: NormalizedConfigFingerprint,
    pub(crate) upstream_position: RawUpstreamPosition,
    pub(crate) activity: IngressActivity,
    pub(crate) admission_gate: AdmissionGateSnapshot,
    pub(crate) last_source_watermark: Option<crate::EventTime>,
    pub(crate) generated_max_nanos: Option<i128>,
    pub(crate) last_generated_watermark: Option<crate::EventTime>,
    pub(crate) watermark_timer: Option<CurrentTimer>,
    pub(crate) idle_timer: Option<CurrentTimer>,
    pub(crate) next_local_sequence: u64,
    pub(crate) next_timer_generation: u64,
    pub(crate) next_timer_sequence: u64,
    pub(crate) next_inbox_sequence: u64,
    pub(crate) next_fence_sequence: u64,
    pub(crate) last_completed_fence: Option<InboxFenceCoordinate>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CapturedLogicalCoordinate {
    pub(crate) driver_clock: DriverClockCoordinate,
    pub(crate) runtime_fence_config_fingerprint: RuntimeFenceConfigFingerprint,
    pub(crate) bindings: Vec<CapturedBindingCoordinate>,
    pub(crate) next_drain_epoch: u64,
    pub(crate) next_admission_attempt: u64,
    pub(crate) next_receipt_sequence: u64,
    pub(crate) next_gate_close_ordinal: u64,
    pub(crate) next_global_sequence: u64,
    pub(crate) idle_epoch: IdleEpoch,
    pub(crate) next_idle_epoch: u64,
    pub(crate) aggregate_watermark: Option<crate::EventTime>,
    pub(crate) idle_latched: bool,
    pub(crate) terminal: bool,
    pub(crate) scheduled_timers: Vec<(TimerIdentity, CurrentTimer)>,
    pub(crate) unsettled_accepted_envelopes: usize,
    pub(crate) next_trace_record_ordinal: u64,
    pub(crate) consumed_trace_position: u64,
    pub(crate) progress_execution_trace: ProgressExecutionTrace,
}

#[derive(Clone, Debug)]
pub(crate) struct StreamProgressSnapshot {
    pub(crate) prepared_job_fingerprint: PreparedJobFingerprint,
    pub(crate) phase: DriverPhase,
    pub(crate) coordinate: CapturedLogicalCoordinate,
    pub(super) payload: DriverSnapshotPayload,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PausedExactUpstreams {
    pub(crate) positions: BTreeMap<BindingIdentity, RawUpstreamPosition>,
}

impl PausedExactUpstreams {
    pub(crate) fn new(positions: BTreeMap<BindingIdentity, RawUpstreamPosition>) -> Result<Self> {
        if positions
            .values()
            .any(|position| matches!(position, RawUpstreamPosition::Unavailable))
        {
            return Err(CalcFlowError::InvalidArgument {
                field: "runtime.progress.snapshot.paused_upstreams".into(),
                message:
                    "every paused upstream must report an exact replay cursor and control frontier"
                        .into(),
            });
        }
        Ok(Self { positions })
    }
}

pub(crate) struct RestoreRequest {
    pub(crate) snapshot: StreamProgressSnapshot,
    pub(crate) paused_upstreams: PausedExactUpstreams,
    pub(crate) replay: Option<ProgressReplayRequest>,
}

#[allow(
    dead_code,
    reason = "keeps the snapshot coordinate tied to its semantic counter type"
)]
const fn _drain_epoch_marker(epoch: DrainEpoch) -> u64 {
    epoch.0
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, num::NonZeroUsize, sync::Arc};

    use datafusion::arrow::{
        array::{ArrayRef, Int64Array},
        record_batch::RecordBatch,
    };

    use super::{PausedExactUpstreams, RestoreRequest};
    use crate::runtime::streaming::progress::{
        driver::{DriverEmission, ManualClock, RawIngressEvent, StreamProgressDriver},
        prepare::{
            BindingIdentity, DeclaredSchema, FenceSelectionPolicy, NativeWatermarkCapability,
            ReplayPositioningCapability, SourceBindingSpec, SourceDescriptor,
            StreamProgressRuntimeConfig, WatermarkPolicy, prepare_stream_job,
        },
        trace::{ProgressReplayRequest, RawUpstreamPosition},
        types::LogicalInstant,
    };
    use crate::{Batch, BatchMetadata};

    fn identity(value: &str) -> BindingIdentity {
        BindingIdentity::new(value).unwrap()
    }

    fn exact(value: u8) -> RawUpstreamPosition {
        RawUpstreamPosition::Exact {
            delivery_replay_cursor: vec![value],
            control_frontier: vec![value],
        }
    }

    fn source(binding: &str, replay: ReplayPositioningCapability) -> SourceBindingSpec {
        SourceBindingSpec {
            descriptor: SourceDescriptor::new(
                identity(binding),
                DeclaredSchema::DynamicOrUnknown,
                NativeWatermarkCapability::NeverEmits,
                replay,
                None,
            ),
            watermark_policy: WatermarkPolicy::Disabled { idle_timeout: None },
        }
    }

    fn prepared_with(
        fingerprint: &str,
        sources: &[SourceBindingSpec],
        capacity: usize,
    ) -> Arc<super::super::prepare::PreparedStreamJob> {
        Arc::new(
            prepare_stream_job(
                fingerprint,
                sources,
                StreamProgressRuntimeConfig {
                    per_binding_inbox_capacity: NonZeroUsize::new(capacity).unwrap(),
                    fence_selection: FenceSelectionPolicy::AllVisible,
                },
            )
            .unwrap(),
        )
    }

    fn prepared() -> Arc<super::super::prepare::PreparedStreamJob> {
        prepared_with(
            "compiled",
            &[source(
                "left",
                ReplayPositioningCapability::ExactPauseReportAndSeek,
            )],
            8,
        )
    }

    fn batch(value: i64) -> Batch {
        Batch::table(
            vec![
                RecordBatch::try_from_iter(vec![(
                    "value",
                    Arc::new(Int64Array::from(vec![value])) as ArrayRef,
                )])
                .unwrap(),
            ],
            BatchMetadata::default(),
        )
        .unwrap()
    }

    fn paused(position: RawUpstreamPosition) -> PausedExactUpstreams {
        PausedExactUpstreams::new(BTreeMap::from([(identity("left"), position)])).unwrap()
    }

    async fn commit_data(
        driver: &mut StreamProgressDriver<ManualClock>,
        sender: &super::super::driver::RawIngressSender,
        position: u8,
    ) -> Vec<&'static str> {
        let receipt = sender
            .submit(
                identity("left"),
                RawIngressEvent::Data(batch(i64::from(position))),
                exact(position),
            )
            .await
            .unwrap();
        let drain = driver.drain_ready().unwrap();
        receipt.wait_settled().await.unwrap();
        drain
            .emissions
            .iter()
            .map(|emission| match emission {
                DriverEmission::ForwardData { .. } => "data",
                DriverEmission::Progress(_) => "progress",
            })
            .collect()
    }

    async fn commit_end(
        driver: &mut StreamProgressDriver<ManualClock>,
        sender: &super::super::driver::RawIngressSender,
        position: u8,
    ) -> Vec<&'static str> {
        let receipt = sender
            .submit(
                identity("left"),
                RawIngressEvent::EndOfInput,
                exact(position),
            )
            .await
            .unwrap();
        let drain = driver.drain_ready().unwrap();
        receipt.wait_settled().await.unwrap();
        drain
            .emissions
            .iter()
            .map(|emission| match emission {
                DriverEmission::ForwardData { .. } => "data",
                DriverEmission::Progress(_) => "progress",
            })
            .collect()
    }

    #[tokio::test]
    async fn progress_snapshot_requires_quiescent_boundary() {
        let clock = ManualClock::new(LogicalInstant::ZERO);
        let (mut driver, sender) = StreamProgressDriver::new(prepared(), clock).unwrap();
        driver.start_running().unwrap();
        let receipt = sender
            .submit(identity("left"), RawIngressEvent::Data(batch(1)), exact(0))
            .await
            .unwrap();
        assert!(driver.capture_snapshot(&paused(exact(0))).is_err());
        driver.drain_ready().unwrap();
        receipt.wait_settled().await.unwrap();
        assert!(driver.capture_snapshot(&paused(exact(0))).is_ok());
    }

    #[tokio::test]
    async fn progress_snapshot_roundtrips_complete_logical_coordinate() {
        let prepared = prepared();
        let clock = ManualClock::new(LogicalInstant(7));
        let (mut driver, sender) =
            StreamProgressDriver::new(Arc::clone(&prepared), clock.clone()).unwrap();
        driver.start_running().unwrap();
        commit_data(&mut driver, &sender, 0).await;
        let snapshot = driver.capture_snapshot(&paused(exact(0))).unwrap();
        let expected_coordinate = snapshot.coordinate.clone();
        let expected_status = driver.status();
        let (restored, _) = StreamProgressDriver::restore(
            prepared,
            clock,
            RestoreRequest {
                snapshot,
                paused_upstreams: paused(exact(0)),
                replay: None,
            },
        )
        .unwrap();
        assert_eq!(restored.status(), expected_status);
        let restored_snapshot = restored.capture_snapshot(&paused(exact(0))).unwrap();
        assert_eq!(restored_snapshot.coordinate, expected_coordinate);
    }

    #[tokio::test]
    async fn progress_restore_rejects_prepared_job_mismatch() {
        let original = prepared();
        let clock = ManualClock::new(LogicalInstant::ZERO);
        let (mut driver, sender) = StreamProgressDriver::new(original, clock.clone()).unwrap();
        driver.start_running().unwrap();
        commit_data(&mut driver, &sender, 0).await;
        let snapshot = driver.capture_snapshot(&paused(exact(0))).unwrap();
        let different = prepared_with(
            "other-compiled",
            &[source(
                "left",
                ReplayPositioningCapability::ExactPauseReportAndSeek,
            )],
            8,
        );
        assert!(
            StreamProgressDriver::restore(
                different,
                clock,
                RestoreRequest {
                    snapshot,
                    paused_upstreams: paused(exact(0)),
                    replay: None,
                },
            )
            .is_err()
        );
    }

    #[tokio::test]
    async fn progress_restore_rejects_binding_identity_mismatch() {
        let original = prepared();
        let clock = ManualClock::new(LogicalInstant::ZERO);
        let (mut driver, sender) = StreamProgressDriver::new(original, clock.clone()).unwrap();
        driver.start_running().unwrap();
        commit_data(&mut driver, &sender, 0).await;
        let mut snapshot = driver.capture_snapshot(&paused(exact(0))).unwrap();
        snapshot.coordinate.bindings[0].identity = identity("wrong");
        assert!(
            StreamProgressDriver::restore(
                prepared(),
                clock,
                RestoreRequest {
                    snapshot,
                    paused_upstreams: paused(exact(0)),
                    replay: None,
                },
            )
            .is_err()
        );
    }

    #[tokio::test]
    async fn progress_restore_rejects_normalized_config_mismatch() {
        let original = prepared();
        let clock = ManualClock::new(LogicalInstant::ZERO);
        let (mut driver, sender) = StreamProgressDriver::new(original, clock.clone()).unwrap();
        driver.start_running().unwrap();
        commit_data(&mut driver, &sender, 0).await;
        let mut snapshot = driver.capture_snapshot(&paused(exact(0))).unwrap();
        let changed = prepared_with(
            "compiled",
            &[source(
                "left",
                ReplayPositioningCapability::ExactPauseReportAndSeek,
            )],
            9,
        );
        snapshot.prepared_job_fingerprint = changed.fingerprint;
        assert!(
            StreamProgressDriver::restore(
                changed,
                clock,
                RestoreRequest {
                    snapshot,
                    paused_upstreams: paused(exact(0)),
                    replay: None,
                },
            )
            .is_err()
        );
    }

    #[tokio::test]
    async fn progress_restore_requires_exact_captured_coordinate() {
        let prepared = prepared();
        let clock = ManualClock::new(LogicalInstant::ZERO);
        let (mut driver, sender) =
            StreamProgressDriver::new(Arc::clone(&prepared), clock.clone()).unwrap();
        driver.start_running().unwrap();
        commit_data(&mut driver, &sender, 0).await;
        let snapshot = driver.capture_snapshot(&paused(exact(0))).unwrap();
        clock.set(LogicalInstant(1));
        assert!(
            StreamProgressDriver::restore(
                prepared,
                clock,
                RestoreRequest {
                    snapshot,
                    paused_upstreams: paused(exact(0)),
                    replay: None,
                },
            )
            .is_err()
        );
    }

    #[tokio::test]
    async fn progress_snapshot_rejects_non_replayable_source() {
        let prepared = prepared_with(
            "compiled",
            &[source("left", ReplayPositioningCapability::Unsupported)],
            8,
        );
        let clock = ManualClock::new(LogicalInstant::ZERO);
        let (mut driver, sender) = StreamProgressDriver::new(prepared, clock).unwrap();
        driver.start_running().unwrap();
        commit_data(&mut driver, &sender, 0).await;
        assert!(driver.capture_snapshot(&paused(exact(0))).is_err());
    }

    #[tokio::test]
    async fn progress_restore_requires_paused_exact_upstream_position() {
        let prepared = prepared();
        let clock = ManualClock::new(LogicalInstant::ZERO);
        let (mut driver, sender) =
            StreamProgressDriver::new(Arc::clone(&prepared), clock.clone()).unwrap();
        driver.start_running().unwrap();
        commit_data(&mut driver, &sender, 0).await;
        let snapshot = driver.capture_snapshot(&paused(exact(0))).unwrap();
        assert!(
            StreamProgressDriver::restore(
                prepared,
                clock,
                RestoreRequest {
                    snapshot,
                    paused_upstreams: paused(exact(1)),
                    replay: None,
                },
            )
            .is_err()
        );
    }

    #[tokio::test]
    async fn progress_replay_is_deterministic_from_exact_coordinate() {
        let prepared = prepared();
        let clock = ManualClock::new(LogicalInstant::ZERO);
        let (mut recorded, sender) =
            StreamProgressDriver::new(Arc::clone(&prepared), clock.clone()).unwrap();
        recorded.start_running().unwrap();
        let expected_outputs = [
            commit_data(&mut recorded, &sender, 0).await,
            commit_end(&mut recorded, &sender, 1).await,
        ];
        let expected_trace = recorded.trace();
        for _ in 0..100 {
            let request = ProgressReplayRequest::prevalidate(expected_trace.clone()).unwrap();
            let (mut replay, sender) =
                StreamProgressDriver::replay(Arc::clone(&prepared), clock.clone(), request)
                    .unwrap();
            replay.start_running().unwrap();
            let outputs = [
                commit_data(&mut replay, &sender, 0).await,
                commit_end(&mut replay, &sender, 1).await,
            ];
            assert_eq!(outputs, expected_outputs);
            assert_eq!(replay.trace(), expected_trace);
            replay.finish_replay().unwrap();
        }
    }

    #[tokio::test]
    async fn progress_snapshot_roundtrips_execution_trace_coordinate() {
        let prepared = prepared();
        let clock = ManualClock::new(LogicalInstant::ZERO);
        let (mut recorded, sender) =
            StreamProgressDriver::new(Arc::clone(&prepared), clock.clone()).unwrap();
        recorded.start_running().unwrap();
        commit_data(&mut recorded, &sender, 0).await;
        let snapshot = recorded.capture_snapshot(&paused(exact(0))).unwrap();
        commit_end(&mut recorded, &sender, 1).await;
        let full_trace = recorded.trace();
        let request = ProgressReplayRequest::prevalidate(full_trace.clone()).unwrap();
        let (mut restored, sender) = StreamProgressDriver::restore(
            prepared,
            clock,
            RestoreRequest {
                snapshot,
                paused_upstreams: paused(exact(0)),
                replay: Some(request),
            },
        )
        .unwrap();
        commit_end(&mut restored, &sender, 1).await;
        assert_eq!(restored.trace(), full_trace);
        restored.finish_replay().unwrap();
    }
}
