#![allow(
    dead_code,
    reason = "durable restore primitives are owned internally by public continuous checkpoint recovery"
)]

use std::collections::{BTreeMap, BTreeSet};

use super::{
    prepare::{NormalizedWatermarkMode, PreparedStreamJob},
    types::LogicalInstant,
};
use crate::{
    CalcFlowError, CursorManifestEntry, EventTime, Result, SourceManifestEntry,
    SourceWatermarkManifestState,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DurableSourceCut {
    pub(crate) cursor: Option<CursorManifestEntry>,
    pub(crate) next_sequence: u64,
    pub(crate) ended: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RestoredSourceProgress {
    pub(crate) cursor: Option<CursorManifestEntry>,
    pub(crate) next_sequence: u64,
    pub(crate) ended: bool,
    pub(crate) idle: bool,
    pub(crate) observed_max: Option<EventTime>,
    pub(crate) last_watermark: Option<EventTime>,
    pub(crate) watermark_deadline: Option<LogicalInstant>,
    pub(crate) idle_deadline: Option<LogicalInstant>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DurableProgressRestore {
    pub(crate) origin: LogicalInstant,
    pub(crate) sources: BTreeMap<String, RestoredSourceProgress>,
    pub(crate) next_receipt_sequence: u64,
    pub(crate) trace_records: u64,
}

pub(crate) fn restore_durable_progress(
    prepared: &PreparedStreamJob,
    sources: &BTreeMap<String, SourceManifestEntry>,
    origin: LogicalInstant,
) -> Result<DurableProgressRestore> {
    let expected = prepared
        .bindings
        .iter()
        .map(|binding| binding.identity.as_str().to_owned())
        .collect::<BTreeSet<_>>();
    let found = sources.keys().cloned().collect::<BTreeSet<_>>();
    if found != expected {
        return Err(CalcFlowError::CheckpointMismatch {
            message: "durable source progress IDs do not match the prepared job".into(),
        });
    }
    let mut restored = BTreeMap::new();
    for binding in prepared.bindings.iter() {
        let id = binding.identity.as_str();
        let source = &sources[id];
        if source.identity_hash != binding.identity_hash() {
            return Err(CalcFlowError::CheckpointMismatch {
                message: format!("durable source identity for {id:?} does not match preflight"),
            });
        }
        let progress = restore_source(binding, source, origin)?;
        restored.insert(id.to_owned(), progress);
    }
    Ok(DurableProgressRestore {
        origin,
        sources: restored,
        next_receipt_sequence: 0,
        trace_records: 0,
    })
}

fn restore_source(
    binding: &super::prepare::PreparedSourceBinding,
    source: &SourceManifestEntry,
    origin: LogicalInstant,
) -> Result<RestoredSourceProgress> {
    let (idle, observed_max, last_watermark, watermark_delay, idle_delay) =
        match (&binding.normalized_watermark, &source.watermark_policy) {
            (
                NormalizedWatermarkMode::SourceProvided { .. },
                SourceWatermarkManifestState::SourceProvided {
                    last_emitted_micros,
                    idle,
                },
            ) => (*idle, None, *last_emitted_micros, None, None),
            (
                NormalizedWatermarkMode::Generated {
                    emit_interval,
                    idle_timeout,
                    ..
                },
                SourceWatermarkManifestState::BoundedOutOfOrderness {
                    observed_max_micros,
                    last_emitted_micros,
                    idle,
                },
            ) => (
                *idle,
                *observed_max_micros,
                *last_emitted_micros,
                Some(*emit_interval),
                *idle_timeout,
            ),
            (
                NormalizedWatermarkMode::Disabled { idle_timeout, .. },
                SourceWatermarkManifestState::Disabled { idle },
            ) => (*idle, None, None, None, *idle_timeout),
            _ => {
                return Err(CalcFlowError::CheckpointMismatch {
                    message: format!(
                        "durable watermark policy for {:?} does not match preflight",
                        binding.identity.as_str()
                    ),
                });
            }
        };
    if observed_max.is_some_and(|maximum| last_watermark.is_some_and(|last| last > maximum)) {
        return Err(CalcFlowError::CheckpointMismatch {
            message: format!(
                "durable watermark for {:?} exceeds its observed maximum",
                binding.identity.as_str()
            ),
        });
    }
    let watermark_deadline = rearm(origin, watermark_delay, source.ended)?;
    let idle_deadline = rearm(origin, idle_delay, source.ended)?;
    Ok(RestoredSourceProgress {
        cursor: source.cursor.clone(),
        next_sequence: source.sequence,
        ended: source.ended,
        idle,
        observed_max,
        last_watermark,
        watermark_deadline,
        idle_deadline,
    })
}

fn rearm(
    origin: LogicalInstant,
    delay: Option<std::time::Duration>,
    ended: bool,
) -> Result<Option<LogicalInstant>> {
    if ended {
        Ok(None)
    } else {
        delay.map(|delay| origin.checked_add(delay)).transpose()
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, sync::Arc, time::Duration};

    use datafusion::arrow::datatypes::{DataType, Field, Schema, TimeUnit};

    use super::restore_durable_progress;
    use crate::runtime::streaming::progress::{
        prepare::{
            BindingIdentity, DeclaredSchema, NativeWatermarkCapability,
            ReplayPositioningCapability, SourceBindingSpec, SourceDescriptor,
            StreamProgressRuntimeConfig, WatermarkPolicy, prepare_stream_job,
        },
        types::LogicalInstant,
    };
    use crate::{EventTime, SourceManifestEntry, SourceWatermarkManifestState};

    fn prepared() -> super::super::prepare::PreparedStreamJob {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "at",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        )]));
        prepare_stream_job(
            "compiled",
            &[SourceBindingSpec {
                descriptor: SourceDescriptor::new(
                    BindingIdentity::new("source").unwrap(),
                    DeclaredSchema::Known(schema),
                    NativeWatermarkCapability::NeverEmits,
                    ReplayPositioningCapability::ExactPauseReportAndSeek,
                    None,
                ),
                watermark_policy: WatermarkPolicy::BoundedOutOfOrderness {
                    event_time_column: Arc::from("at"),
                    max_out_of_orderness: Duration::from_secs(2),
                    emit_interval: Duration::from_secs(10),
                    idle_timeout: Some(Duration::from_secs(30)),
                },
            }],
            StreamProgressRuntimeConfig::default(),
        )
        .unwrap()
    }

    #[test]
    fn durable_restore_rearms_full_delays_from_a_fresh_origin() {
        let prepared = prepared();
        let identity_hash = prepared.bindings[0].identity_hash();
        let sources = BTreeMap::from([(
            "source".into(),
            SourceManifestEntry {
                cursor: None,
                identity_hash,
                sequence: 8,
                ended: false,
                watermark_policy: SourceWatermarkManifestState::BoundedOutOfOrderness {
                    observed_max_micros: Some(EventTime::from_micros(20_000_000)),
                    last_emitted_micros: Some(EventTime::from_micros(18_000_000)),
                    idle: false,
                },
            },
        )]);
        let origin = LogicalInstant(500);

        let restored = restore_durable_progress(&prepared, &sources, origin).unwrap();
        let source = &restored.sources["source"];

        assert_eq!(source.next_sequence, 8);
        assert_eq!(
            source.last_watermark,
            Some(EventTime::from_micros(18_000_000))
        );
        assert_eq!(
            source.watermark_deadline,
            Some(origin.checked_add(Duration::from_secs(10)).unwrap())
        );
        assert_eq!(
            source.idle_deadline,
            Some(origin.checked_add(Duration::from_secs(30)).unwrap())
        );
        assert_eq!(restored.next_receipt_sequence, 0);
        assert_eq!(restored.trace_records, 0);
    }
}
