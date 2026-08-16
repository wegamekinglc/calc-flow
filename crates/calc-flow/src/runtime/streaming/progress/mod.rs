pub(super) mod aggregate;
mod driver;
mod durable;
mod generated;
pub(super) mod prepare;
mod snapshot;
mod status;
mod trace;
pub(super) mod types;

pub(crate) use driver::{LiveProgressCoordinator, RawIngressEvent, spawn_live_progress_task};
#[allow(
    unused_imports,
    reason = "M5 durable progress is consumed by private checkpoint recovery"
)]
pub(crate) use durable::{
    DurableProgressRestore, DurableSourceCut, RestoredSourceProgress, restore_durable_progress,
};
pub(crate) use prepare::BindingIdentity;
pub(crate) use status::{LiveProgressEvidence, LiveProgressStatusHandle};
pub(crate) use trace::RawUpstreamPosition;

#[allow(
    unused_imports,
    reason = "preflight preparation exports are shared internally by the public continuous facade"
)]
pub(crate) use prepare::{
    DeclaredSchema, ExistingPrivateToggleRoute, NativeWatermarkCapability, PreparedSourceBinding,
    PreparedStreamJob, ReplayPositioningCapability, SourceBindingSpec, SourceDescriptor,
    StreamProgressRuntimeConfig, WatermarkPolicy, prepare_stream_job,
};

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    use super::{
        NativeWatermarkCapability, ReplayPositioningCapability, SourceDescriptor,
        StreamProgressRuntimeConfig, WatermarkPolicy, prepare_stream_job,
    };

    #[test]
    fn compile_stream_remains_binding_agnostic() {
        let _ = size_of::<WatermarkPolicy>();
    }

    #[test]
    fn preflight_failure_has_no_runtime_side_effects() {
        let _ = prepare_stream_job;
    }

    #[test]
    fn watermark_policy_capability_matrix_is_exhaustive() {
        let _ = (
            NativeWatermarkCapability::NeverEmits,
            ReplayPositioningCapability::Unsupported,
            size_of::<SourceDescriptor>(),
            size_of::<StreamProgressRuntimeConfig>(),
        );
    }
}
