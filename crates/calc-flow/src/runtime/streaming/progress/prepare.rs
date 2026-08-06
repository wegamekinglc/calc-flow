use std::{collections::BTreeSet, num::NonZeroUsize, sync::Arc, time::Duration};

use datafusion::arrow::datatypes::{DataType, SchemaRef, TimeUnit};
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::{CalcFlowError, Result, canonical_json};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct BindingIdentity(Arc<str>);

impl BindingIdentity {
    pub(crate) fn new(value: impl Into<Arc<str>>) -> Result<Self> {
        let value = value.into();
        if value.trim().is_empty() || value.contains('\0') {
            return Err(CalcFlowError::InvalidArgument {
                field: "source.binding".into(),
                message: "must be non-empty, non-whitespace, and contain no NUL".into(),
            });
        }
        Ok(Self(value))
    }

    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct BindingOrdinal(u64);

impl BindingOrdinal {
    pub(crate) const fn new(value: u64) -> Self {
        Self(value)
    }

    pub(crate) const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct PreparedJobFingerprint([u8; 32]);

impl PreparedJobFingerprint {
    pub(crate) const fn as_bytes(self) -> [u8; 32] {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct NormalizedConfigFingerprint([u8; 32]);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct RuntimeFenceConfigFingerprint([u8; 32]);

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum WatermarkPolicy {
    SourceProvided,
    BoundedOutOfOrderness {
        event_time_column: Arc<str>,
        max_out_of_orderness: Duration,
        emit_interval: Duration,
        idle_timeout: Option<Duration>,
    },
    Disabled {
        idle_timeout: Option<Duration>,
    },
}

impl Default for WatermarkPolicy {
    fn default() -> Self {
        Self::SourceProvided
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ArrowTimestampUnit {
    Second,
    Millisecond,
    Microsecond,
    Nanosecond,
}

impl From<&TimeUnit> for ArrowTimestampUnit {
    fn from(value: &TimeUnit) -> Self {
        match value {
            TimeUnit::Second => Self::Second,
            TimeUnit::Millisecond => Self::Millisecond,
            TimeUnit::Microsecond => Self::Microsecond,
            TimeUnit::Nanosecond => Self::Nanosecond,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ResolvedEventTimeColumn {
    pub(crate) name: Arc<str>,
    pub(crate) index: usize,
    pub(crate) unit: ArrowTimestampUnit,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NativeWatermarkDirective {
    LeaveEnabled,
    EnableThroughExistingPrivateRoute,
    DisableThroughExistingPrivateRoute,
    AlreadyDisabled,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum NormalizedWatermarkMode {
    SourceProvided {
        native_directive: NativeWatermarkDirective,
    },
    Generated {
        event_time: ResolvedEventTimeColumn,
        max_out_of_orderness: Duration,
        emit_interval: Duration,
        idle_timeout: Option<Duration>,
        native_directive: NativeWatermarkDirective,
    },
    Disabled {
        idle_timeout: Option<Duration>,
        native_directive: NativeWatermarkDirective,
    },
}

impl NormalizedWatermarkMode {
    pub(crate) const fn idle_timeout(&self) -> Option<Duration> {
        match self {
            Self::SourceProvided { .. } => None,
            Self::Generated { idle_timeout, .. } | Self::Disabled { idle_timeout, .. } => {
                *idle_timeout
            }
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum DeclaredSchema {
    Known(SchemaRef),
    DynamicOrUnknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NativeWatermarkCapability {
    NeverEmits,
    EmitsNative,
    RuntimeToggleable,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplayPositioningCapability {
    ExactPauseReportAndSeek,
    Unsupported,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ExistingPrivateToggleRoute {
    route_id: Arc<str>,
}

impl ExistingPrivateToggleRoute {
    pub(crate) fn new(route_id: impl Into<Arc<str>>) -> Result<Self> {
        let route_id = route_id.into();
        if route_id.trim().is_empty() || route_id.contains('\0') {
            return Err(CalcFlowError::InvalidArgument {
                field: "source.existing_toggle_route".into(),
                message: "must identify an existing private toggle route".into(),
            });
        }
        Ok(Self { route_id })
    }
}

#[derive(Clone, Debug)]
pub(crate) struct SourceDescriptor {
    pub(crate) binding: BindingIdentity,
    pub(crate) declared_schema: DeclaredSchema,
    pub(crate) native_watermarks: NativeWatermarkCapability,
    pub(crate) replay_positioning: ReplayPositioningCapability,
    pub(crate) existing_toggle_route: Option<ExistingPrivateToggleRoute>,
}

impl SourceDescriptor {
    pub(crate) fn new(
        binding: BindingIdentity,
        declared_schema: DeclaredSchema,
        native_watermarks: NativeWatermarkCapability,
        replay_positioning: ReplayPositioningCapability,
        existing_toggle_route: Option<ExistingPrivateToggleRoute>,
    ) -> Self {
        Self {
            binding,
            declared_schema,
            native_watermarks,
            replay_positioning,
            existing_toggle_route,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FenceSelectionPolicy {
    AllVisible,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct StreamProgressRuntimeConfig {
    pub(crate) per_binding_inbox_capacity: NonZeroUsize,
    pub(crate) fence_selection: FenceSelectionPolicy,
}

impl Default for StreamProgressRuntimeConfig {
    fn default() -> Self {
        Self {
            per_binding_inbox_capacity: NonZeroUsize::new(64).expect("64 is non-zero"),
            fence_selection: FenceSelectionPolicy::AllVisible,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct SourceBindingSpec {
    pub(crate) descriptor: SourceDescriptor,
    pub(crate) watermark_policy: WatermarkPolicy,
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedSourceBinding {
    pub(crate) identity: BindingIdentity,
    pub(crate) ordinal: BindingOrdinal,
    pub(crate) declared_schema: DeclaredSchema,
    pub(crate) normalized_watermark: NormalizedWatermarkMode,
    pub(crate) normalized_config_fingerprint: NormalizedConfigFingerprint,
    pub(crate) replay_positioning: ReplayPositioningCapability,
    pub(crate) existing_toggle_route: Option<ExistingPrivateToggleRoute>,
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedStreamJob {
    pub(crate) compiled_fingerprint: Arc<str>,
    pub(crate) bindings: Arc<[PreparedSourceBinding]>,
    pub(crate) runtime_progress_config: StreamProgressRuntimeConfig,
    pub(crate) runtime_fence_config_fingerprint: RuntimeFenceConfigFingerprint,
    pub(crate) fingerprint: PreparedJobFingerprint,
}

pub(crate) fn prepare_stream_job(
    compiled_fingerprint: &str,
    bindings: &[SourceBindingSpec],
    runtime_progress_config: StreamProgressRuntimeConfig,
) -> Result<PreparedStreamJob> {
    if compiled_fingerprint.is_empty() {
        return Err(CalcFlowError::InvalidArgument {
            field: "compiled.fingerprint".into(),
            message: "must not be empty".into(),
        });
    }
    let mut identities = BTreeSet::new();
    let mut prepared = Vec::with_capacity(bindings.len());
    for (index, binding) in bindings.iter().enumerate() {
        if !identities.insert(binding.descriptor.binding.clone()) {
            return Err(CalcFlowError::InvalidArgument {
                field: format!("sources.{}", binding.descriptor.binding.as_str()),
                message: "binding is configured more than once".into(),
            });
        }
        let ordinal = u64::try_from(index)
            .ok()
            .filter(|value| value.checked_add(1).is_some())
            .map(BindingOrdinal::new)
            .ok_or_else(|| CalcFlowError::InvalidArgument {
                field: "runtime.progress.binding_ordinal".into(),
                message: "counter exhausted before a successor could be reserved".into(),
            })?;
        let normalized = normalize_policy(&binding.descriptor, &binding.watermark_policy)?;
        let normalized_config_fingerprint = normalized_fingerprint(&normalized)?;
        prepared.push(PreparedSourceBinding {
            identity: binding.descriptor.binding.clone(),
            ordinal,
            declared_schema: binding.descriptor.declared_schema.clone(),
            normalized_watermark: normalized,
            normalized_config_fingerprint,
            replay_positioning: binding.descriptor.replay_positioning,
            existing_toggle_route: binding.descriptor.existing_toggle_route.clone(),
        });
    }
    let runtime_fence_config_fingerprint = runtime_fingerprint(&runtime_progress_config)?;
    let fingerprint = prepared_fingerprint(
        compiled_fingerprint,
        &prepared,
        runtime_fence_config_fingerprint,
    )?;
    Ok(PreparedStreamJob {
        compiled_fingerprint: Arc::from(compiled_fingerprint),
        bindings: prepared.into(),
        runtime_progress_config,
        runtime_fence_config_fingerprint,
        fingerprint,
    })
}

fn normalize_policy(
    descriptor: &SourceDescriptor,
    policy: &WatermarkPolicy,
) -> Result<NormalizedWatermarkMode> {
    let path = format!(
        "sources.{}.capabilities.native_watermarks",
        descriptor.binding.as_str()
    );
    let directive = normalize_capability(
        descriptor.native_watermarks,
        policy,
        descriptor.existing_toggle_route.is_some(),
    )
    .ok_or_else(|| CalcFlowError::InvalidArgument {
        field: path,
        message: "watermark policy conflicts with the source descriptor capability".into(),
    })?;
    match policy {
        WatermarkPolicy::SourceProvided => Ok(NormalizedWatermarkMode::SourceProvided {
            native_directive: directive,
        }),
        WatermarkPolicy::BoundedOutOfOrderness {
            event_time_column,
            max_out_of_orderness,
            emit_interval,
            idle_timeout,
        } => {
            validate_duration(descriptor, "max_out_of_orderness", *max_out_of_orderness)?;
            validate_duration(descriptor, "emit_interval", *emit_interval)?;
            if let Some(timeout) = idle_timeout {
                validate_duration(descriptor, "idle_timeout", *timeout)?;
            }
            let event_time = resolve_event_time(descriptor, event_time_column)?;
            Ok(NormalizedWatermarkMode::Generated {
                event_time,
                max_out_of_orderness: *max_out_of_orderness,
                emit_interval: *emit_interval,
                idle_timeout: *idle_timeout,
                native_directive: directive,
            })
        }
        WatermarkPolicy::Disabled { idle_timeout } => {
            if let Some(timeout) = idle_timeout {
                validate_duration(descriptor, "idle_timeout", *timeout)?;
            }
            Ok(NormalizedWatermarkMode::Disabled {
                idle_timeout: *idle_timeout,
                native_directive: directive,
            })
        }
    }
}

#[allow(
    clippy::match_same_arms,
    reason = "the explicit arms mirror the normative capability-policy matrix"
)]
fn normalize_capability(
    capability: NativeWatermarkCapability,
    policy: &WatermarkPolicy,
    has_toggle_route: bool,
) -> Option<NativeWatermarkDirective> {
    use NativeWatermarkCapability as Capability;
    use NativeWatermarkDirective as Directive;
    match (capability, policy) {
        (Capability::NeverEmits, WatermarkPolicy::SourceProvided) => None,
        (
            Capability::NeverEmits,
            WatermarkPolicy::BoundedOutOfOrderness { .. } | WatermarkPolicy::Disabled { .. },
        ) => Some(Directive::AlreadyDisabled),
        (Capability::EmitsNative, WatermarkPolicy::SourceProvided) => Some(Directive::LeaveEnabled),
        (
            Capability::EmitsNative,
            WatermarkPolicy::BoundedOutOfOrderness { .. } | WatermarkPolicy::Disabled { .. },
        ) => None,
        (Capability::RuntimeToggleable, WatermarkPolicy::SourceProvided) if has_toggle_route => {
            Some(Directive::EnableThroughExistingPrivateRoute)
        }
        (
            Capability::RuntimeToggleable,
            WatermarkPolicy::BoundedOutOfOrderness { .. } | WatermarkPolicy::Disabled { .. },
        ) if has_toggle_route => Some(Directive::DisableThroughExistingPrivateRoute),
        (Capability::RuntimeToggleable | Capability::Unknown, _) => None,
    }
}

fn validate_duration(descriptor: &SourceDescriptor, name: &str, duration: Duration) -> Result<()> {
    if !duration.is_zero() && u64::try_from(duration.as_nanos()).is_ok() {
        return Ok(());
    }
    Err(CalcFlowError::InvalidArgument {
        field: format!(
            "sources.{}.watermark_policy.{name}",
            descriptor.binding.as_str()
        ),
        message: "must be positive and representable as logical nanoseconds".into(),
    })
}

fn resolve_event_time(
    descriptor: &SourceDescriptor,
    column: &Arc<str>,
) -> Result<ResolvedEventTimeColumn> {
    let path = format!(
        "sources.{}.watermark_policy.event_time_column",
        descriptor.binding.as_str()
    );
    let DeclaredSchema::Known(schema) = &descriptor.declared_schema else {
        return Err(CalcFlowError::InvalidArgument {
            field: path,
            message: "generated watermarks require a statically known Arrow schema".into(),
        });
    };
    let (index, field) = schema
        .fields()
        .iter()
        .enumerate()
        .find(|(_, field)| field.name() == column.as_ref())
        .ok_or_else(|| CalcFlowError::InvalidArgument {
            field: path.clone(),
            message: format!("declared schema has no column {column:?}"),
        })?;
    let DataType::Timestamp(unit, timezone) = field.data_type() else {
        return Err(CalcFlowError::InvalidArgument {
            field: path,
            message: format!(
                "generated watermark column must be an Arrow timestamp, found {}",
                field.data_type()
            ),
        });
    };
    if timezone
        .as_ref()
        .is_some_and(|timezone| timezone.as_ref() != "UTC")
    {
        return Err(CalcFlowError::InvalidArgument {
            field: path,
            message: "event-time timestamp timezone must be UTC or absent".into(),
        });
    }
    Ok(ResolvedEventTimeColumn {
        name: Arc::clone(column),
        index,
        unit: unit.into(),
    })
}

fn digest(value: &serde_json::Value) -> Result<[u8; 32]> {
    let canonical = canonical_json(value)?;
    Ok(Sha256::digest(canonical.as_bytes()).into())
}

fn duration_nanos(duration: Duration) -> String {
    duration.as_nanos().to_string()
}

fn normalized_fingerprint(mode: &NormalizedWatermarkMode) -> Result<NormalizedConfigFingerprint> {
    let value = match mode {
        NormalizedWatermarkMode::SourceProvided { native_directive } => json!({
            "mode": "source_provided",
            "directive": format!("{native_directive:?}"),
        }),
        NormalizedWatermarkMode::Generated {
            event_time,
            max_out_of_orderness,
            emit_interval,
            idle_timeout,
            native_directive,
        } => json!({
            "mode": "generated",
            "event_time": {
                "name": event_time.name.as_ref(),
                "index": event_time.index,
                "unit": format!("{:?}", event_time.unit),
            },
            "max_out_of_orderness_ns": duration_nanos(*max_out_of_orderness),
            "emit_interval_ns": duration_nanos(*emit_interval),
            "idle_timeout_ns": idle_timeout.map(duration_nanos),
            "directive": format!("{native_directive:?}"),
        }),
        NormalizedWatermarkMode::Disabled {
            idle_timeout,
            native_directive,
        } => json!({
            "mode": "disabled",
            "idle_timeout_ns": idle_timeout.map(duration_nanos),
            "directive": format!("{native_directive:?}"),
        }),
    };
    digest(&value).map(NormalizedConfigFingerprint)
}

fn runtime_fingerprint(
    config: &StreamProgressRuntimeConfig,
) -> Result<RuntimeFenceConfigFingerprint> {
    digest(&json!({
        "capacity": config.per_binding_inbox_capacity.get(),
        "fence_selection": "all_visible",
    }))
    .map(RuntimeFenceConfigFingerprint)
}

fn prepared_fingerprint(
    compiled_fingerprint: &str,
    bindings: &[PreparedSourceBinding],
    runtime: RuntimeFenceConfigFingerprint,
) -> Result<PreparedJobFingerprint> {
    digest(&json!({
        "compiled": compiled_fingerprint,
        "runtime": hex::encode(runtime.0),
        "bindings": bindings.iter().map(|binding| json!({
            "identity": binding.identity.as_str(),
            "ordinal": binding.ordinal.get(),
            "normalized": hex::encode(binding.normalized_config_fingerprint.0),
            "native": format!("{:?}", binding.normalized_watermark),
            "replay": format!("{:?}", binding.replay_positioning),
            "toggle": binding.existing_toggle_route.as_ref().map(|route| route.route_id.as_ref()),
        })).collect::<Vec<_>>(),
    }))
    .map(PreparedJobFingerprint)
}

#[cfg(test)]
mod tests {
    use std::{num::NonZeroUsize, sync::Arc, time::Duration};

    use datafusion::arrow::datatypes::{DataType, Field, Schema, TimeUnit};

    use super::{
        BindingIdentity, DeclaredSchema, ExistingPrivateToggleRoute, FenceSelectionPolicy,
        NativeWatermarkCapability, NativeWatermarkDirective, NormalizedWatermarkMode,
        ReplayPositioningCapability, SourceBindingSpec, SourceDescriptor,
        StreamProgressRuntimeConfig, WatermarkPolicy, prepare_stream_job,
    };

    fn schema(data_type: DataType) -> Arc<Schema> {
        Arc::new(Schema::new(vec![Field::new("at", data_type, true)]))
    }

    fn descriptor(
        capability: NativeWatermarkCapability,
        declared_schema: DeclaredSchema,
        toggle: bool,
    ) -> SourceDescriptor {
        SourceDescriptor::new(
            BindingIdentity::new("left").unwrap(),
            declared_schema,
            capability,
            ReplayPositioningCapability::ExactPauseReportAndSeek,
            toggle.then(|| ExistingPrivateToggleRoute::new("private-route").unwrap()),
        )
    }

    fn generated() -> WatermarkPolicy {
        WatermarkPolicy::BoundedOutOfOrderness {
            event_time_column: Arc::from("at"),
            max_out_of_orderness: Duration::from_secs(1),
            emit_interval: Duration::from_secs(2),
            idle_timeout: Some(Duration::from_secs(3)),
        }
    }

    fn config() -> StreamProgressRuntimeConfig {
        StreamProgressRuntimeConfig {
            per_binding_inbox_capacity: NonZeroUsize::new(4).unwrap(),
            fence_selection: FenceSelectionPolicy::AllVisible,
        }
    }

    fn prepare(
        capability: NativeWatermarkCapability,
        declared_schema: DeclaredSchema,
        policy: WatermarkPolicy,
        toggle: bool,
    ) -> crate::Result<super::PreparedStreamJob> {
        prepare_stream_job(
            "compiled",
            &[SourceBindingSpec {
                descriptor: descriptor(capability, declared_schema, toggle),
                watermark_policy: policy,
            }],
            config(),
        )
    }

    #[test]
    fn generated_policy_rejects_unknown_schema_before_open() {
        let error = prepare(
            NativeWatermarkCapability::NeverEmits,
            DeclaredSchema::DynamicOrUnknown,
            generated(),
            false,
        )
        .unwrap_err();
        assert!(error.to_string().contains("statically known Arrow schema"));
    }

    #[test]
    fn generated_policy_reports_missing_column_path() {
        let policy = WatermarkPolicy::BoundedOutOfOrderness {
            event_time_column: Arc::from("missing"),
            max_out_of_orderness: Duration::from_secs(1),
            emit_interval: Duration::from_secs(2),
            idle_timeout: Some(Duration::from_secs(3)),
        };
        let error = prepare(
            NativeWatermarkCapability::NeverEmits,
            DeclaredSchema::Known(schema(DataType::Timestamp(TimeUnit::Microsecond, None))),
            policy,
            false,
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("sources.left.watermark_policy.event_time_column")
        );
    }

    #[test]
    fn generated_policy_rejects_unsupported_event_time_type() {
        let error = prepare(
            NativeWatermarkCapability::NeverEmits,
            DeclaredSchema::Known(schema(DataType::Int64)),
            generated(),
            false,
        )
        .unwrap_err();
        assert!(error.to_string().contains("Arrow timestamp"));
    }

    #[test]
    fn watermark_policy_rejects_invalid_durations() {
        for policy in [
            WatermarkPolicy::BoundedOutOfOrderness {
                event_time_column: Arc::from("at"),
                max_out_of_orderness: Duration::ZERO,
                emit_interval: Duration::from_secs(1),
                idle_timeout: None,
            },
            WatermarkPolicy::BoundedOutOfOrderness {
                event_time_column: Arc::from("at"),
                max_out_of_orderness: Duration::from_secs(1),
                emit_interval: Duration::ZERO,
                idle_timeout: None,
            },
            WatermarkPolicy::Disabled {
                idle_timeout: Some(Duration::ZERO),
            },
        ] {
            assert!(
                prepare(
                    NativeWatermarkCapability::NeverEmits,
                    DeclaredSchema::Known(schema(
                        DataType::Timestamp(TimeUnit::Microsecond, None,)
                    )),
                    policy,
                    false,
                )
                .is_err()
            );
        }
    }

    #[test]
    fn watermark_policy_capability_matrix_is_exhaustive() {
        use NativeWatermarkCapability::{EmitsNative, NeverEmits, RuntimeToggleable, Unknown};
        let known = DeclaredSchema::Known(schema(DataType::Timestamp(TimeUnit::Microsecond, None)));
        let policies = [
            WatermarkPolicy::SourceProvided,
            generated(),
            WatermarkPolicy::Disabled { idle_timeout: None },
        ];
        let expected = [
            [false, true, true],
            [true, false, false],
            [true, true, true],
            [false, false, false],
        ];
        for (capability_index, capability) in [NeverEmits, EmitsNative, RuntimeToggleable, Unknown]
            .into_iter()
            .enumerate()
        {
            for (policy_index, policy) in policies.iter().cloned().enumerate() {
                assert_eq!(
                    prepare(
                        capability,
                        known.clone(),
                        policy,
                        capability == RuntimeToggleable
                    )
                    .is_ok(),
                    expected[capability_index][policy_index],
                    "capability={capability:?}, policy_index={policy_index}"
                );
            }
        }
    }

    #[test]
    fn runtime_toggleable_requires_existing_private_hook() {
        assert!(
            prepare(
                NativeWatermarkCapability::RuntimeToggleable,
                DeclaredSchema::DynamicOrUnknown,
                WatermarkPolicy::SourceProvided,
                false,
            )
            .is_err()
        );
        let prepared = prepare(
            NativeWatermarkCapability::RuntimeToggleable,
            DeclaredSchema::DynamicOrUnknown,
            WatermarkPolicy::SourceProvided,
            true,
        )
        .unwrap();
        assert!(matches!(
            prepared.bindings[0].normalized_watermark,
            NormalizedWatermarkMode::SourceProvided {
                native_directive: NativeWatermarkDirective::EnableThroughExistingPrivateRoute
            }
        ));
    }

    #[test]
    fn watermark_modes_never_silently_merge() {
        let error = prepare(
            NativeWatermarkCapability::EmitsNative,
            DeclaredSchema::Known(schema(DataType::Timestamp(TimeUnit::Second, None))),
            generated(),
            false,
        )
        .unwrap_err();
        assert!(error.to_string().contains("conflicts"));
    }
}
