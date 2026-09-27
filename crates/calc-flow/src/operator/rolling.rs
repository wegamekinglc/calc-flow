//! Native row-window rolling operator: lag, delta, and the count/sum/mean/
//! variance/standard-deviation aggregates over entity-partitioned, event-time
//! ordered rows (SCE-00 D5, API note `symbolic-computation-engine` section
//! 3.2). The same calculation kernel serves batch and final-only stream
//! lifecycles; stream state is checkpointed at the aligned epoch cut, and
//! aggregate window state is rebuilt from the retained history rows on
//! restore.

use std::{
    cmp::Ordering,
    collections::{BTreeMap, HashMap, VecDeque},
    io::Cursor,
    sync::Arc,
};

use async_trait::async_trait;
use datafusion::arrow::{
    array::{Array, ArrayRef, Float64Array, UInt8Array, UInt64Array, new_null_array},
    compute::concat_batches,
    datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit},
    ipc::{
        convert::IpcSchemaEncoder,
        reader::FileReader,
        writer::{DictionaryTracker, FileWriter},
    },
    record_batch::RecordBatch,
};
use datafusion::common::ScalarValue;
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::{
    Batch, BatchKind, BatchMetadata, CalcFlowError, Epoch, EventTime, JsonMap, Port, Result,
    StateHandle, TableBatch, canonical_json,
    state::{SegmentDescriptor, SegmentKind, StateInventory},
};

use super::rolling_metrics::{RollingMetricsRecorder, RollingStage, RollingWork};
use super::{
    BatchOperator, BatchOperatorContext, LateMetricDelta, OperatorMetadata, StateBudget,
    StreamCollector, StreamOperator, StreamOperatorContext, accumulate_late_metrics,
    expression::required_input,
    late_output::{LateRowTally, record_late_row, reject_batch_mode},
    validate_operator_name,
};

mod budget;
mod generated_kernel_manifest;
mod kernel;
mod late;
mod ordered_stream;
mod state_v3;

use super::checkpoint::{checkpoint_mismatch, compile_error, internal_error, state_format};
use budget::StateCharge;
#[cfg(test)]
use kernel::KernelSelection;
#[cfg(test)]
pub(crate) use kernel::entity_parallel_tests::entity_parallel_test_pair;
use kernel::{RollingKernelPlan, RollingKernelState};
pub(crate) use kernel::{
    StreamKernelUpdate,
    entity_parallel::{
        ActualNumericWork, LaneProgress, LocatedPanic, NumericJoinSeed, NumericLaneExit,
        NumericLaneOutcome, NumericLaneRequest, NumericLaneStop, ScratchPlan,
        merge_numeric_results, run_numeric_lane,
    },
};

/// Semantic configuration version of the first rolling operator release.
pub const ROLLING_CONFIGURATION_VERSION: u32 = 1;
/// Durable state-layout version of the first rolling operator release.
pub const ROLLING_STATE_LAYOUT_VERSION: u32 = 1;
/// Durable state-layout version that persists exponential accumulators.
pub const ROLLING_EWMA_STATE_LAYOUT_VERSION: u32 = 2;
/// Durable columnar state-layout version written by current rolling operators.
pub const ROLLING_COLUMNAR_STATE_LAYOUT_VERSION: u32 = 3;

/// Versioned floating-point behavior for rolling numeric transitions.
///
/// `StableV1` remains the default and preserves the released operation order.
/// `StableV2Preview` is an explicit opt-in experiment whose serialized name is
/// `stable_v2`; it may not replace the default without a separate migration.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RollingNumericalProfile {
    /// Released West/Welford add/remove behavior.
    #[default]
    StableV1,
    /// Deterministically rebased, shifted-sum preview behavior.
    #[serde(rename = "stable_v2")]
    StableV2Preview,
}

impl RollingNumericalProfile {
    const fn name(self) -> &'static str {
        match self {
            Self::StableV1 => "stable_v1",
            Self::StableV2Preview => "stable_v2",
        }
    }

    #[allow(
        clippy::trivially_copy_pass_by_ref,
        reason = "serde skip_serializing_if requires a borrowed field predicate"
    )]
    const fn is_stable_v1(&self) -> bool {
        matches!(self, Self::StableV1)
    }
}

/// One SQL AVG/COUNT window accepted by the crate-private `DataFusion` rolling
/// physical planner.
#[derive(Clone, Debug)]
pub(crate) struct DataFusionRollingWindow {
    pub input_index: usize,
    pub output_name: String,
    pub rows: u64,
    pub is_count: bool,
}

/// Immutable typed rolling plan shared with `CalcFlowRollingExec`.
#[derive(Clone, Debug)]
pub(crate) struct DataFusionRollingKernel {
    plan: RollingKernelPlan,
}

/// Per-partition transition state owned by one `DataFusion` execution stream.
#[derive(Clone, Debug, Default)]
pub(crate) struct DataFusionRollingState {
    inner: kernel::SortedRollingState,
}

/// Deterministic execution facts forwarded to `DataFusion` physical metrics.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct DataFusionRollingMetrics {
    pub input_validation_ns: u64,
    pub order_proof_ns: u64,
    pub entity_encode_ns: u64,
    pub kernel_ns: u64,
    pub output_build_ns: u64,
    pub input_rows: usize,
    pub entities: usize,
    pub state_bytes: usize,
}

/// One typed `DataFusion` rolling transition and its next state.
#[derive(Debug)]
pub(crate) struct DataFusionRollingBatch {
    pub columns: Vec<ArrayRef>,
    pub state: DataFusionRollingState,
    pub metrics: DataFusionRollingMetrics,
}

impl DataFusionRollingKernel {
    pub(crate) fn compile(
        input_schema: &Schema,
        partition_indices: &[usize],
        order_indices: &[usize],
        windows: &[DataFusionRollingWindow],
    ) -> Option<Self> {
        if windows.is_empty()
            || windows.iter().any(|window| window.is_count)
            || !kernel::supports_datafusion_primitive("mean")
        {
            return None;
        }
        let (&event_time_index, sequence_indices) = order_indices.split_first()?;
        if !matches!(
            input_schema.field(event_time_index).data_type(),
            DataType::Timestamp(TimeUnit::Microsecond, timezone)
                if timezone.as_deref().is_none_or(|timezone| timezone == "UTC")
        ) || order_indices
            .iter()
            .any(|&index| input_schema.field(index).is_nullable())
            || partition_indices
                .iter()
                .any(|index| order_indices.contains(index))
        {
            return None;
        }
        let mut groups = Vec::new();
        let mut outputs = Vec::with_capacity(windows.len());
        for window in windows {
            if window.rows == 0
                || input_schema.field(window.input_index).data_type() != &DataType::Float64
            {
                return None;
            }
            let input_type = DataType::Float64;
            let evaluation = compile_aggregate(
                window.input_index,
                &input_type,
                RollingFrameSpec::Rows { size: window.rows },
                1,
                0,
                Statistic::Mean,
                &mut groups,
            );
            outputs.push(CompiledRollingOutput {
                input_index: window.input_index,
                name: window.output_name.clone(),
                input_type: input_type.clone(),
                output_type: DataType::Float64,
                evaluation,
            });
        }
        let sequence_columns = sequence_indices
            .iter()
            .copied()
            .map(|index| CompiledKeyColumn { index })
            .collect::<Vec<_>>();
        let physical_order = partition_indices
            .iter()
            .chain(order_indices)
            .copied()
            .collect::<Vec<_>>();
        let plan = RollingKernelPlan::compile_with_order(
            input_schema,
            ROLLING_STATE_LAYOUT_VERSION,
            RollingNumericalProfile::StableV1,
            event_time_index,
            physical_order,
            partition_indices.to_vec(),
            sequence_columns.iter().map(|column| column.index).collect(),
            &outputs,
            &groups,
        )
        .with_nan_as_value();
        plan.supports_typed_transition().then_some(Self { plan })
    }

    pub(crate) fn update_and_fill(
        &self,
        state: &DataFusionRollingState,
        input: &RecordBatch,
    ) -> Result<DataFusionRollingBatch> {
        let execution =
            self.plan
                .update_sorted_and_fill(&state.inner, input, "datafusion.rolling")?;
        let metrics = execution.metrics;
        Ok(DataFusionRollingBatch {
            columns: execution.columns,
            state: DataFusionRollingState {
                inner: execution.state,
            },
            metrics: DataFusionRollingMetrics {
                input_validation_ns: metrics.input_validation_ns,
                order_proof_ns: metrics.order_proof_ns,
                entity_encode_ns: metrics.entity_encode_ns,
                kernel_ns: metrics.kernel_ns,
                output_build_ns: metrics.output_build_ns,
                input_rows: metrics.input_rows,
                entities: metrics.entities,
                state_bytes: metrics.state_bytes,
            },
        })
    }

    pub(crate) fn fingerprint(&self) -> &str {
        self.plan.fingerprint()
    }

    pub(crate) const fn estimated_state_bytes_per_entity(&self) -> usize {
        self.plan.estimated_state_bytes_per_entity()
    }
}

/// Transaction scope of the `error` late-row policy (API note section 3.2).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum LateErrorScope {
    /// The complete input envelope is rejected atomically.
    Envelope,
}

/// Late-row handling for one rolling operator (SCE-00 D7).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum LatePolicySpec {
    /// Reject the complete input envelope without state, metric, or output
    /// changes.
    Error {
        /// The only supported transaction scope.
        scope: LateErrorScope,
    },
    /// Drop each late row and record the three SCE-00 D7 metrics.
    Drop {
        /// Metric transaction version; must equal `1`.
        metrics_version: u32,
    },
    /// Route late rows to a diagnostic table in stream mode.
    ///
    /// The diagnostic output has no event-time progress and participates in
    /// the same aligned checkpoint epochs as the normal output.
    SideOutput {
        /// Metric transaction version; must equal `1`.
        #[schemars(range(min = 1, max = 1))]
        metrics_version: u32,
        /// Diagnostic schema version; must equal `1`.
        #[schemars(range(min = 1, max = 1))]
        schema_version: u32,
    },
}

/// Frozen null/NaN policy for rolling values (SCE-00 D3, contract
/// section 5.2).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RollingValuePolicy {
    /// Lag/delta preserve a null or NaN current or referenced operand.
    StatefulNumericV1,
}

/// Rolling frame declaration (SCE-00 D5): a row-count frame or an
/// event-time duration frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RollingFrameSpec {
    /// Row-count frame `rows [i - size + 1, i]` including the current row.
    Rows {
        /// Positive retained row count.
        #[schemars(range(min = 1))]
        size: u64,
    },
    /// Event-time frame `(t - micros, t]` over the entity total order
    /// (SCE-00 D5: open lower bound, closed upper bound).
    Duration {
        /// Positive exact frame width in microseconds.
        #[schemars(range(min = 1))]
        micros: u64,
    },
}

impl RollingFrameSpec {
    const fn size(self) -> u64 {
        match self {
            Self::Rows { size } => size,
            Self::Duration { .. } => 0,
        }
    }

    const fn micros(self) -> u64 {
        match self {
            Self::Rows { .. } => 0,
            Self::Duration { micros } => micros,
        }
    }

    /// Retained-row bound contribution: row frames retain their size;
    /// duration frames retain by event time instead (SCE-08).
    const fn row_retention(self) -> u64 {
        self.size()
    }

    const fn is_duration(self) -> bool {
        matches!(self, Self::Duration { .. })
    }
}

/// One Float64 rolling readout used only inside a fused derived output.
///
/// These leaves declare state semantics but have no output name, so the
/// operator can share their accumulators without materializing intermediate
/// Arrow columns.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RollingFloatPrimitiveSpec {
    /// Float64 mean over a row or duration frame.
    Mean {
        /// Primitive version; must equal `1`.
        primitive_version: u32,
        /// Numeric input column name.
        input: String,
        /// Rolling frame.
        frame: RollingFrameSpec,
        /// Minimum valid samples for a non-null result.
        #[schemars(range(min = 1))]
        min_periods: u64,
    },
    /// Float64 variance over a row or duration frame.
    Variance {
        /// Primitive version; must equal `1`.
        primitive_version: u32,
        /// Numeric input column name.
        input: String,
        /// Rolling frame.
        frame: RollingFrameSpec,
        /// Minimum valid samples for a non-null result.
        #[schemars(range(min = 1))]
        min_periods: u64,
        /// Degrees-of-freedom adjustment; must be `0` or `1`.
        #[schemars(range(min = 0, max = 1))]
        ddof: u8,
    },
    /// Float64 standard deviation over a row or duration frame.
    Stddev {
        /// Primitive version; must equal `1`.
        primitive_version: u32,
        /// Numeric input column name.
        input: String,
        /// Rolling frame.
        frame: RollingFrameSpec,
        /// Minimum valid samples for a non-null result.
        #[schemars(range(min = 1))]
        min_periods: u64,
        /// Degrees-of-freedom adjustment; must be `0` or `1`.
        #[schemars(range(min = 0, max = 1))]
        ddof: u8,
    },
    /// Unadjusted exponentially weighted moving average.
    Ewma {
        /// Primitive version; must equal `1`.
        primitive_version: u32,
        /// Numeric input column name.
        input: String,
        /// Positive exponential span.
        #[schemars(range(min = 1))]
        span: u64,
        /// Minimum valid samples for a non-null result.
        #[schemars(range(min = 1))]
        min_periods: u64,
    },
}

impl RollingFloatPrimitiveSpec {
    const fn primitive_version(&self) -> u32 {
        match self {
            Self::Mean {
                primitive_version, ..
            }
            | Self::Variance {
                primitive_version, ..
            }
            | Self::Stddev {
                primitive_version, ..
            }
            | Self::Ewma {
                primitive_version, ..
            } => *primitive_version,
        }
    }

    fn input(&self) -> &str {
        match self {
            Self::Mean { input, .. }
            | Self::Variance { input, .. }
            | Self::Stddev { input, .. }
            | Self::Ewma { input, .. } => input,
        }
    }

    const fn retained_rows(&self) -> u64 {
        match self {
            Self::Mean { frame, .. }
            | Self::Variance { frame, .. }
            | Self::Stddev { frame, .. } => frame.row_retention(),
            Self::Ewma { .. } => 0,
        }
    }

    const fn retained_micros(&self) -> Option<u64> {
        match self {
            Self::Mean { frame, .. }
            | Self::Variance { frame, .. }
            | Self::Stddev { frame, .. }
                if frame.is_duration() =>
            {
                Some(frame.micros())
            }
            _ => None,
        }
    }

    const fn requires_ewma_layout(&self) -> bool {
        matches!(self, Self::Ewma { .. })
    }
}

/// One declared rolling output and its output column name.
///
/// # Examples
///
/// ```
/// use calc_flow::{RollingFrameSpec, RollingOutputSpec};
///
/// let output = RollingOutputSpec::Decay {
///     primitive_version: 1,
///     input: "price".into(),
///     output: "weighted_price".into(),
///     frame: RollingFrameSpec::Rows { size: 5 },
///     min_periods: 1,
/// };
/// assert!(matches!(output, RollingOutputSpec::Decay { .. }));
/// ```
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RollingOutputSpec {
    /// Value of the same column `periods` earlier in the entity total order.
    Lag {
        /// Primitive version; must equal `1`.
        primitive_version: u32,
        /// Input column name.
        input: String,
        /// Output column name.
        output: String,
        /// Positive lag distance in rows.
        #[schemars(range(min = 1))]
        periods: u64,
    },
    /// Checked difference between the current value and the value `periods`
    /// earlier in the entity total order.
    Delta {
        /// Primitive version; must equal `1`.
        primitive_version: u32,
        /// Input column name.
        input: String,
        /// Output column name.
        output: String,
        /// Positive lag distance in rows.
        #[schemars(range(min = 1))]
        periods: u64,
    },
    /// Unadjusted exponentially weighted moving average with
    /// `alpha = 2 / (span + 1)`.
    Ewma {
        /// Primitive version; must equal `1`.
        primitive_version: u32,
        /// Numeric input column name.
        input: String,
        /// Output column name.
        output: String,
        /// Positive exponential span.
        #[schemars(range(min = 1))]
        span: u64,
        /// Minimum valid samples for a non-null result.
        #[schemars(range(min = 1))]
        min_periods: u64,
    },
    /// Unbounded arithmetic mean of every valid sample seen for an entity.
    CumulativeMean {
        /// Primitive version; must equal `1`.
        primitive_version: u32,
        /// Numeric input column name.
        input: String,
        /// Output column name.
        output: String,
        /// Minimum valid samples for a non-null result.
        #[schemars(range(min = 1))]
        min_periods: u64,
    },
    /// Valid (non-null, non-NaN) sample count over the frame (SCE-00 D3,
    /// contract section 5.2).
    Count {
        /// Primitive version; must equal `1`.
        primitive_version: u32,
        /// Input column name.
        input: String,
        /// Output column name.
        output: String,
        /// Row-count frame.
        frame: RollingFrameSpec,
        /// Minimum valid samples for a non-null result.
        #[schemars(range(min = 1))]
        min_periods: u64,
    },
    /// Checked sum over the frame; integer results stay exact (SCE-00 D3,
    /// contract section 5.2).
    Sum {
        /// Primitive version; must equal `1`.
        primitive_version: u32,
        /// Input column name.
        input: String,
        /// Output column name.
        output: String,
        /// Row-count frame.
        frame: RollingFrameSpec,
        /// Minimum valid samples for a non-null result.
        #[schemars(range(min = 1))]
        min_periods: u64,
    },
    /// Float64 mean over the frame.
    Mean {
        /// Primitive version; must equal `1`.
        primitive_version: u32,
        /// Input column name.
        input: String,
        /// Output column name.
        output: String,
        /// Row-count frame.
        frame: RollingFrameSpec,
        /// Minimum valid samples for a non-null result.
        #[schemars(range(min = 1))]
        min_periods: u64,
    },
    /// Float64 variance over the frame (SCE-00 D5 divisor rules).
    Variance {
        /// Primitive version; must equal `1`.
        primitive_version: u32,
        /// Input column name.
        input: String,
        /// Output column name.
        output: String,
        /// Row-count frame.
        frame: RollingFrameSpec,
        /// Minimum valid samples for a non-null result.
        #[schemars(range(min = 1))]
        min_periods: u64,
        /// Degrees-of-freedom adjustment; must be `0` or `1`.
        #[schemars(range(min = 0, max = 1))]
        ddof: u8,
    },
    /// Float64 standard deviation over the frame (SCE-00 D5 divisor rules).
    Stddev {
        /// Primitive version; must equal `1`.
        primitive_version: u32,
        /// Input column name.
        input: String,
        /// Output column name.
        output: String,
        /// Row-count frame.
        frame: RollingFrameSpec,
        /// Minimum valid samples for a non-null result.
        #[schemars(range(min = 1))]
        min_periods: u64,
        /// Degrees-of-freedom adjustment; must be `0` or `1`.
        #[schemars(range(min = 0, max = 1))]
        ddof: u8,
    },
    /// Minimum valid sample over the frame; preserves the input type (SCE-00
    /// D3, contract section 5.2).
    Min {
        /// Primitive version; must equal `1`.
        primitive_version: u32,
        /// Input column name.
        input: String,
        /// Output column name.
        output: String,
        /// Row-count or duration frame.
        frame: RollingFrameSpec,
        /// Minimum valid samples for a non-null result.
        #[schemars(range(min = 1))]
        min_periods: u64,
    },
    /// Maximum valid sample over the frame; preserves the input type (SCE-00
    /// D3, contract section 5.2).
    Max {
        /// Primitive version; must equal `1`.
        primitive_version: u32,
        /// Input column name.
        input: String,
        /// Output column name.
        output: String,
        /// Row-count or duration frame.
        frame: RollingFrameSpec,
        /// Minimum valid samples for a non-null result.
        #[schemars(range(min = 1))]
        min_periods: u64,
    },
    /// Number of row positions since the oldest maximum in the frame.
    Argmax {
        /// Primitive version; must equal `1`.
        primitive_version: u32,
        /// Numeric input column name.
        input: String,
        /// Output column name.
        output: String,
        /// Row-count or duration frame.
        frame: RollingFrameSpec,
        /// Minimum valid samples for a non-null result.
        #[schemars(range(min = 1))]
        min_periods: u64,
    },
    /// Number of row positions since the oldest minimum in the frame.
    Argmin {
        /// Primitive version; must equal `1`.
        primitive_version: u32,
        /// Numeric input column name.
        input: String,
        /// Output column name.
        output: String,
        /// Row-count or duration frame.
        frame: RollingFrameSpec,
        /// Minimum valid samples for a non-null result.
        #[schemars(range(min = 1))]
        min_periods: u64,
    },
    /// Zero-based ascending rank of the current value; ties take the first rank.
    Rank {
        /// Primitive version; must equal `1`.
        primitive_version: u32,
        /// Numeric input column name.
        input: String,
        /// Output column name.
        output: String,
        /// Row-count or duration frame.
        frame: RollingFrameSpec,
        /// Minimum valid samples for a non-null result.
        #[schemars(range(min = 1))]
        min_periods: u64,
    },
    /// Current value's zero-based rank divided by `valid_count - 1`.
    Quantile {
        /// Primitive version; must equal `1`.
        primitive_version: u32,
        /// Numeric input column name.
        input: String,
        /// Output column name.
        output: String,
        /// Row-count or duration frame.
        frame: RollingFrameSpec,
        /// Minimum valid samples for a non-null result.
        #[schemars(range(min = 1))]
        min_periods: u64,
    },
    /// Number of distinct valid samples in the frame.
    UniqueCount {
        /// Primitive version; must equal `1`.
        primitive_version: u32,
        /// Totally ordered input column name.
        input: String,
        /// Output column name.
        output: String,
        /// Row-count or duration frame.
        frame: RollingFrameSpec,
        /// Minimum valid samples for a non-null result.
        #[schemars(range(min = 1))]
        min_periods: u64,
    },
    /// Linearly weighted mean of valid samples, newest carrying the largest weight.
    Decay {
        /// Primitive version; must equal `1`.
        primitive_version: u32,
        /// Numeric input column name.
        input: String,
        /// Output column name.
        output: String,
        /// Row-count or duration frame.
        frame: RollingFrameSpec,
        /// Minimum valid samples for a non-null result.
        #[schemars(range(min = 1))]
        min_periods: u64,
    },
    /// Float64 covariance of two columns over the frame, counting only
    /// pairwise-valid positions (SCE-00 D3, contract section 5.2; D5).
    Covariance {
        /// Primitive version; must equal `1`.
        primitive_version: u32,
        /// Left input column name.
        left: String,
        /// Right input column name.
        right: String,
        /// Output column name.
        output: String,
        /// Row-count or duration frame.
        frame: RollingFrameSpec,
        /// Minimum pairwise-valid samples for a non-null result.
        #[schemars(range(min = 1))]
        min_periods: u64,
        /// Degrees-of-freedom adjustment; must be `0` or `1`.
        #[schemars(range(min = 0, max = 1))]
        ddof: u8,
    },
    /// Float64 Pearson correlation of two columns over the frame; null when
    /// either side has zero variance (SCE-00 D3, contract section 5.2).
    Correlation {
        /// Primitive version; must equal `1`.
        primitive_version: u32,
        /// Left input column name.
        left: String,
        /// Right input column name.
        right: String,
        /// Output column name.
        output: String,
        /// Row-count or duration frame.
        frame: RollingFrameSpec,
        /// Minimum pairwise-valid samples for a non-null result.
        #[schemars(range(min = 1))]
        min_periods: u64,
        /// Degrees-of-freedom adjustment; must be `0` or `1`.
        #[schemars(range(min = 0, max = 1))]
        ddof: u8,
    },
    /// Difference of two Float64 rolling readouts, evaluated directly from
    /// their shared state without materializing either leaf.
    Difference {
        /// Primitive version; must equal `1`.
        primitive_version: u32,
        /// Left rolling readout.
        left: Box<RollingFloatPrimitiveSpec>,
        /// Right rolling readout.
        right: Box<RollingFloatPrimitiveSpec>,
        /// Output column name.
        output: String,
    },
}

impl RollingOutputSpec {
    fn primitive_version(&self) -> u32 {
        match self {
            Self::Lag {
                primitive_version, ..
            }
            | Self::Delta {
                primitive_version, ..
            }
            | Self::Ewma {
                primitive_version, ..
            }
            | Self::CumulativeMean {
                primitive_version, ..
            }
            | Self::Count {
                primitive_version, ..
            }
            | Self::Sum {
                primitive_version, ..
            }
            | Self::Mean {
                primitive_version, ..
            }
            | Self::Variance {
                primitive_version, ..
            }
            | Self::Stddev {
                primitive_version, ..
            }
            | Self::Min {
                primitive_version, ..
            }
            | Self::Max {
                primitive_version, ..
            }
            | Self::Argmax {
                primitive_version, ..
            }
            | Self::Argmin {
                primitive_version, ..
            }
            | Self::Rank {
                primitive_version, ..
            }
            | Self::Quantile {
                primitive_version, ..
            }
            | Self::UniqueCount {
                primitive_version, ..
            }
            | Self::Decay {
                primitive_version, ..
            }
            | Self::Covariance {
                primitive_version, ..
            }
            | Self::Correlation {
                primitive_version, ..
            }
            | Self::Difference {
                primitive_version, ..
            } => *primitive_version,
        }
    }

    /// Declared operand column; pair outputs report their left operand.
    fn input(&self) -> &str {
        match self {
            Self::Lag { input, .. }
            | Self::Delta { input, .. }
            | Self::Ewma { input, .. }
            | Self::CumulativeMean { input, .. }
            | Self::Count { input, .. }
            | Self::Sum { input, .. }
            | Self::Mean { input, .. }
            | Self::Variance { input, .. }
            | Self::Stddev { input, .. }
            | Self::Min { input, .. }
            | Self::Max { input, .. }
            | Self::Argmax { input, .. }
            | Self::Argmin { input, .. }
            | Self::Rank { input, .. }
            | Self::Quantile { input, .. }
            | Self::UniqueCount { input, .. }
            | Self::Decay { input, .. } => input,
            Self::Covariance { left, .. } | Self::Correlation { left, .. } => left,
            Self::Difference { left, .. } => left.input(),
        }
    }

    /// The second operand of a pair output, when declared.
    fn pair_right(&self) -> Option<&str> {
        match self {
            Self::Covariance { right, .. } | Self::Correlation { right, .. } => Some(right),
            _ => None,
        }
    }

    fn output(&self) -> &str {
        match self {
            Self::Lag { output, .. }
            | Self::Delta { output, .. }
            | Self::Ewma { output, .. }
            | Self::CumulativeMean { output, .. }
            | Self::Count { output, .. }
            | Self::Sum { output, .. }
            | Self::Mean { output, .. }
            | Self::Variance { output, .. }
            | Self::Stddev { output, .. }
            | Self::Min { output, .. }
            | Self::Max { output, .. }
            | Self::Argmax { output, .. }
            | Self::Argmin { output, .. }
            | Self::Rank { output, .. }
            | Self::Quantile { output, .. }
            | Self::UniqueCount { output, .. }
            | Self::Decay { output, .. }
            | Self::Covariance { output, .. }
            | Self::Correlation { output, .. }
            | Self::Difference { output, .. } => output,
        }
    }

    /// Rows one output needs retained per entity: the lag/delta distance or
    /// the row-frame size. Duration frames retain by event time instead.
    const fn retained_rows(&self) -> u64 {
        match self {
            Self::Lag { periods, .. } | Self::Delta { periods, .. } => *periods,
            Self::Ewma { .. } | Self::CumulativeMean { .. } => 0,
            Self::Count { frame, .. }
            | Self::Sum { frame, .. }
            | Self::Mean { frame, .. }
            | Self::Variance { frame, .. }
            | Self::Stddev { frame, .. }
            | Self::Min { frame, .. }
            | Self::Max { frame, .. }
            | Self::Argmax { frame, .. }
            | Self::Argmin { frame, .. }
            | Self::Rank { frame, .. }
            | Self::Quantile { frame, .. }
            | Self::UniqueCount { frame, .. }
            | Self::Decay { frame, .. }
            | Self::Covariance { frame, .. }
            | Self::Correlation { frame, .. } => frame.row_retention(),
            Self::Difference { left, right, .. } => {
                let left = left.retained_rows();
                let right = right.retained_rows();
                if left > right { left } else { right }
            }
        }
    }

    /// The widest duration frame one output declares, for time-based
    /// retention.
    const fn retained_micros(&self) -> Option<u64> {
        match self {
            Self::Lag { .. }
            | Self::Delta { .. }
            | Self::Ewma { .. }
            | Self::CumulativeMean { .. } => None,
            Self::Count { frame, .. }
            | Self::Sum { frame, .. }
            | Self::Mean { frame, .. }
            | Self::Variance { frame, .. }
            | Self::Stddev { frame, .. }
            | Self::Min { frame, .. }
            | Self::Max { frame, .. }
            | Self::Argmax { frame, .. }
            | Self::Argmin { frame, .. }
            | Self::Rank { frame, .. }
            | Self::Quantile { frame, .. }
            | Self::UniqueCount { frame, .. }
            | Self::Decay { frame, .. }
            | Self::Covariance { frame, .. }
            | Self::Correlation { frame, .. } => {
                if frame.is_duration() {
                    Some(frame.micros())
                } else {
                    None
                }
            }
            Self::Difference { left, right, .. } => {
                match (left.retained_micros(), right.retained_micros()) {
                    (Some(left), Some(right)) => Some(if left > right { left } else { right }),
                    (Some(value), None) | (None, Some(value)) => Some(value),
                    (None, None) => None,
                }
            }
        }
    }

    const fn frame(&self) -> Option<RollingFrameSpec> {
        match self {
            Self::Lag { .. }
            | Self::Delta { .. }
            | Self::Ewma { .. }
            | Self::CumulativeMean { .. }
            | Self::Difference { .. } => None,
            Self::Count { frame, .. }
            | Self::Sum { frame, .. }
            | Self::Mean { frame, .. }
            | Self::Variance { frame, .. }
            | Self::Stddev { frame, .. }
            | Self::Min { frame, .. }
            | Self::Max { frame, .. }
            | Self::Argmax { frame, .. }
            | Self::Argmin { frame, .. }
            | Self::Rank { frame, .. }
            | Self::Quantile { frame, .. }
            | Self::UniqueCount { frame, .. }
            | Self::Decay { frame, .. }
            | Self::Covariance { frame, .. }
            | Self::Correlation { frame, .. } => Some(*frame),
        }
    }

    const fn min_periods(&self) -> Option<u64> {
        match self {
            Self::Lag { .. } | Self::Delta { .. } | Self::Difference { .. } => None,
            Self::Ewma { min_periods, .. }
            | Self::CumulativeMean { min_periods, .. }
            | Self::Count { min_periods, .. }
            | Self::Sum { min_periods, .. }
            | Self::Mean { min_periods, .. }
            | Self::Variance { min_periods, .. }
            | Self::Stddev { min_periods, .. }
            | Self::Min { min_periods, .. }
            | Self::Max { min_periods, .. }
            | Self::Argmax { min_periods, .. }
            | Self::Argmin { min_periods, .. }
            | Self::Rank { min_periods, .. }
            | Self::Quantile { min_periods, .. }
            | Self::UniqueCount { min_periods, .. }
            | Self::Decay { min_periods, .. }
            | Self::Covariance { min_periods, .. }
            | Self::Correlation { min_periods, .. } => Some(*min_periods),
        }
    }

    const fn ddof(&self) -> Option<u8> {
        match self {
            Self::Variance { ddof, .. }
            | Self::Stddev { ddof, .. }
            | Self::Covariance { ddof, .. }
            | Self::Correlation { ddof, .. } => Some(*ddof),
            _ => None,
        }
    }

    const fn span(&self) -> Option<u64> {
        match self {
            Self::Ewma { span, .. } => Some(*span),
            _ => None,
        }
    }

    const fn requires_ewma_layout(&self) -> bool {
        match self {
            Self::Ewma { .. } | Self::CumulativeMean { .. } => true,
            Self::Difference { left, right, .. } => {
                left.requires_ewma_layout() || right.requires_ewma_layout()
            }
            _ => false,
        }
    }
}

/// Data-only declaration of one native row-window rolling operation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RollingSpec {
    /// Semantic configuration version; must equal
    /// [`ROLLING_CONFIGURATION_VERSION`].
    pub configuration_version: u32,
    /// Durable state-layout version. Existing primitives use
    /// [`ROLLING_STATE_LAYOUT_VERSION`]; EWMA requires
    /// [`ROLLING_EWMA_STATE_LAYOUT_VERSION`].
    pub state_layout_version: u32,
    /// Versioned floating-point behavior. `stable_v1` is omitted from the
    /// canonical configuration so existing project and checkpoint hashes stay
    /// compatible; `stable_v2` is an explicit preview opt-in.
    #[serde(default, skip_serializing_if = "RollingNumericalProfile::is_stable_v1")]
    pub numerical_profile: RollingNumericalProfile,
    /// Ordered non-empty entity partition key.
    pub partition_by: Vec<String>,
    /// Non-null UTC `timestamp[us]` event-time column.
    pub event_time: String,
    /// Ordered non-empty sequence key; floating columns are forbidden.
    pub sequence_by: Vec<String>,
    /// Rolling outputs in semantic declaration order.
    pub outputs: Vec<RollingOutputSpec>,
    /// Allowed lateness in exact microseconds (SCE-00 D7).
    pub allowed_lateness_micros: u64,
    /// Late-row policy.
    pub late_policy: LatePolicySpec,
    /// Frozen null/NaN value policy.
    pub value_policy: RollingValuePolicy,
}

impl RollingSpec {
    /// Validates the declaration against an exact Arrow input schema and
    /// returns the derived output schema: input fields followed by the
    /// declared outputs in order (SCE-00 D5).
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] for invalid declaration
    /// fields and [`CalcFlowError::Compile`] for missing, ambiguous, or
    /// unsupported input columns.
    pub fn validate(&self, input_schema: &Schema) -> Result<SchemaRef> {
        validate_arguments(self)?;
        let compiled = compile_spec(self, input_schema)?;
        Ok(Arc::new(output_schema(input_schema, &compiled.outputs)))
    }
}

/// Native row-window rolling operator over partitioned event-time rows.
pub struct RollingOperator {
    name: String,
    spec: RollingSpec,
    input_ports: [Port; 1],
    output_ports: Vec<Port>,
    compiled: Box<CompiledRollingSpec>,
    state: RollingStreamState,
    state_budget: StateBudget,
}

impl RollingOperator {
    /// Compiles one rolling declaration against an exact Arrow input schema.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] for invalid declaration
    /// fields and [`CalcFlowError::Compile`] for missing, ambiguous, or
    /// unsupported input columns.
    pub fn new(name: &str, input_schema: SchemaRef, spec: RollingSpec) -> Result<Self> {
        validate_operator_name(name)?;
        validate_arguments(&spec)?;
        let configuration = configuration(&spec)?;
        let compiled = Box::new(compile_spec_full(&spec, &input_schema, &configuration)?);
        let output_schema = Arc::new(output_schema(&input_schema, &compiled.outputs));
        let output_ports =
            super::late_output::output_ports(spec.late_policy, &input_schema, output_schema)?;
        Ok(Self {
            name: name.into(),
            spec,
            input_ports: [Port::with_schema_ref(
                "input",
                BatchKind::Table,
                true,
                Some(input_schema),
            )?],
            output_ports,
            compiled,
            state: RollingStreamState::default(),
            state_budget: StateBudget::default(),
        })
    }

    /// Sets the logical row and byte limit for buffered rows and retained
    /// per-entity rolling history. The default is one million rows and 256 MiB.
    /// Checkpoint copies and process resident memory are outside this limit.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when current state exceeds
    /// the requested budget.
    ///
    /// # Examples
    ///
    /// ```
    /// use calc_flow::{RollingOperator, StateBudget};
    /// fn configure(operator: &mut RollingOperator) -> calc_flow::Result<()> {
    ///     operator.set_state_budget(StateBudget::new(10_000, 64 << 20)?)
    /// }
    /// ```
    pub fn set_state_budget(&mut self, budget: StateBudget) -> Result<()> {
        if !budget.allows(self.state.charge.rows, self.state.charge.bytes) {
            return Err(CalcFlowError::InvalidArgument {
                field: "rolling.state_budget".into(),
                message: "existing rolling state exceeds the requested budget".into(),
            });
        }
        self.state_budget = budget;
        Ok(())
    }

    /// Returns the validated rolling declaration.
    pub const fn spec(&self) -> &RollingSpec {
        &self.spec
    }

    /// Returns the durable state layout written by this operator build.
    pub const fn state_layout_version(&self) -> u32 {
        self.compiled.state_layout_version
    }
}

impl std::fmt::Debug for RollingOperator {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RollingOperator")
            .field("name", &self.name)
            .field("spec", &self.spec)
            .field("state_budget", &self.state_budget)
            .field("input_ports", &self.input_ports)
            .field("output_ports", &self.output_ports)
            .field("kernel_version", &self.compiled.kernel_plan.version())
            .field(
                "kernel_state_layout_version",
                &self.compiled.kernel_plan.state_layout_version(),
            )
            .field("kernel", &self.compiled.kernel_plan.selection())
            .field("kernel_complexity", &self.compiled.kernel_plan.complexity())
            .field(
                "kernel_estimated_state_bytes_per_entity",
                &self.compiled.kernel_plan.estimated_state_bytes_per_entity(),
            )
            .field(
                "kernel_fingerprint",
                &self.compiled.kernel_plan.fingerprint(),
            )
            .field(
                "numerical_profile",
                &self.compiled.kernel_plan.numerical_profile(),
            )
            .field(
                "kernel_fallback_reason",
                &self.compiled.kernel_plan.fallback_reason(),
            )
            .finish_non_exhaustive()
    }
}

impl OperatorMetadata for RollingOperator {
    fn name(&self) -> &str {
        &self.name
    }

    fn input_ports(&self) -> &[Port] {
        &self.input_ports
    }

    fn output_ports(&self) -> &[Port] {
        &self.output_ports
    }

    fn configuration(&self) -> JsonMap {
        configuration(&self.spec).expect("validated rolling configuration remains serializable")
    }
}

#[async_trait]
impl BatchOperator for RollingOperator {
    /// Evaluates the complete input in canonical order without late-row
    /// classification (SCE-00 D7): every accepted row is final at
    /// end-of-input.
    // Batch evaluation intentionally owns the validate-read-sort-compute-build
    // pipeline in one pass with a stable error path per stage.
    // #lizard forgives
    async fn process(
        &mut self,
        inputs: &BTreeMap<String, Batch>,
        context: &BatchOperatorContext<'_>,
    ) -> Result<BTreeMap<String, Batch>> {
        reject_batch_mode(self.spec.late_policy, &self.name)?;
        let input = required_input(inputs, "input", &self.name, None)?;
        self.input_ports[0].validate(input, &format!("{}.input", self.name))?;
        context.run.check_cancelled()?;
        let table = input.table_payload()?;
        let output_schema = self.output_ports[0]
            .schema()
            .expect("rolling output always has an exact schema");
        let record = if let Some(record) =
            build_typed_batch_output(table, &self.compiled, output_schema, &self.name)?
        {
            record
        } else {
            let rows = read_buffered_rows(table, &self.compiled, &self.name)?;
            let ordered = sort_and_validate(rows, &self.name)?;
            let computed = compute_output_columns(
                &ordered,
                &RollingHistories::default(),
                &self.compiled,
                &self.name,
            )?;
            build_output_record(&ordered, computed.columns, output_schema, &self.name)?
        };
        let metadata = BatchMetadata::new(&self.name, 0, BTreeMap::new())?;
        let batch = Batch::table(vec![record], metadata)?;
        Ok(BTreeMap::from([("output".into(), batch)]))
    }
}

/// Live stream state owned by one rolling operator task. Mutation is
/// confined to this value; input batches stay read-only.
#[derive(Default)]
struct RollingStreamState {
    buffer: BTreeMap<RowIdentity, BufferedRow>,
    ordered: ordered_stream::OrderedStreamBuffer,
    histories: RollingHistories,
    last_input_watermark: Option<EventTime>,
    next_output_sequence: u64,
    next_late_output_sequence: u64,
    late_output_failed: bool,
    ended: bool,
    metrics: LateMetricDelta,
    pipeline_fingerprint: Option<String>,
    operator_id: Option<String>,
    last_checkpoint_epoch: Option<Epoch>,
    typed_kernel_state: Option<Box<RollingKernelState>>,
    charge: StateCharge,
}

/// Bounded inline manifest contribution of one rolling checkpoint (SCE-00
/// D11); retained rows never appear inline, only in segments.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RollingSnapshotMetadata {
    state_layout_version: u32,
    configuration_hash: String,
    state_schema_fingerprint: String,
    #[serde(default)]
    kernel_fingerprint: Option<String>,
    #[serde(default)]
    numerical_profile: Option<String>,
    epoch: Epoch,
    #[serde(deserialize_with = "deserialize_required_option")]
    pipeline_fingerprint: Option<String>,
    #[serde(deserialize_with = "deserialize_required_option")]
    operator_id: Option<String>,
    #[serde(deserialize_with = "deserialize_required_option")]
    last_input_watermark: Option<EventTime>,
    next_output_sequence: u64,
    ended: bool,
    metrics: LateMetricDelta,
    segment_inventory: Vec<SegmentDescriptor>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "super::late_output::deserialize_snapshot"
    )]
    late_output: Option<super::late_output::LateOutputSnapshot>,
}

#[async_trait]
impl StreamOperator for RollingOperator {
    /// Classifies and buffers one input envelope atomically (SCE-00 D7): the
    /// aggregate input watermark is sampled once, and no row changes state,
    /// metrics, or output before the complete envelope is validated.
    // Envelope classification keeps ingress, port, context, end-of-input, late,
    // and duplicate checks in one transactional pass with stable per-check errors.
    // #lizard forgives
    async fn process_data(
        &mut self,
        ingress: &str,
        batch: Batch,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        let observer = context.rolling_metrics();
        let validation = observer.map(|recorder| recorder.stage(RollingStage::InputValidation));
        if ingress != "input" {
            return Err(operator_error(
                context.operator_id(),
                &format!("unknown ingress {ingress:?}; expected \"input\""),
            ));
        }
        self.input_ports[0].validate(&batch, &format!("{}.input", self.name))?;
        self.observe_context(context)?;
        if self.state.ended {
            return Err(operator_error(
                context.operator_id(),
                "received data after end-of-input",
            ));
        }
        let watermark = context.input_watermark();
        drop(validation);
        if matches!(self.spec.late_policy, LatePolicySpec::SideOutput { .. }) {
            return self
                .process_late_data(&batch, watermark, context, output)
                .await;
        }
        if self.try_buffer_ordered(batch.table_payload()?, watermark, observer)? {
            self.install_context_identity(context);
            return Ok(());
        }
        self.materialize_ordered_buffer(observer)?;
        let rows = {
            let _stage = observer.map(|recorder| recorder.stage(RollingStage::InputValidation));
            let rows = read_buffered_rows(batch.table_payload()?, &self.compiled, &self.name)?;
            if let Some(recorder) = observer {
                for _ in self.input_ports[0].schema().unwrap().fields() {
                    recorder.add(RollingWork::ScalarValueConversions, rows.len());
                }
            }
            rows
        };
        let (accepted, metrics) =
            self.classify_envelope(rows, watermark, context.operator_id(), observer)?;
        let next_metrics = accumulate_late_metrics(self.state.metrics, metrics)?;
        let next_charge = self.state.charge.checked_add(
            budget::buffered_charge(accepted.values(), context.operator_id())?,
            context.operator_id(),
        )?;
        self.check_state_budget(next_charge, context.operator_id())?;
        for (identity, row) in accepted {
            self.state.buffer.insert(identity, row);
        }
        context.record_window_metrics(
            metrics.late_rows,
            metrics.max_lateness_micros,
            metrics.null_event_time_rows,
        )?;
        self.state.metrics = next_metrics;
        self.state.charge = next_charge;
        self.install_context_identity(context);
        Ok(())
    }

    /// Emits every newly final row in canonical order before the runtime
    /// forwards the watermark (SCE-00 D7 final-only output).
    async fn on_watermark(
        &mut self,
        watermark: EventTime,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        // Cancellation is checked before any state mutation so a cancelled
        // emission leaves the buffered rows available for a retry.
        context.check_cancelled()?;
        self.observe_context(context)?;
        if self
            .state
            .last_input_watermark
            .is_some_and(|previous| watermark <= previous)
        {
            return Err(operator_error(
                context.operator_id(),
                "input watermark did not advance strictly",
            ));
        }
        if self.state.ordered.is_empty() {
            let rows = {
                let _stage = context
                    .rolling_metrics()
                    .map(|recorder| recorder.stage(RollingStage::OrderingProof));
                let closing = self.closing_keys(watermark.as_micros(), context.operator_id())?;
                self.take_buffered(&closing)
            };
            self.emit_rows(rows, context, output).await?;
        } else {
            let last_identity = self.state.ordered.last_identity();
            let records = {
                let _stage = context
                    .rolling_metrics()
                    .map(|recorder| recorder.stage(RollingStage::OrderingProof));
                self.state.ordered.take_closed(
                    watermark,
                    &self.compiled,
                    self.spec.allowed_lateness_micros,
                    context.operator_id(),
                )?
            };
            self.emit_ordered(records, last_identity, context, output)
                .await?;
        }
        self.install_context_identity(context);
        self.state.last_input_watermark = Some(watermark);
        Ok(())
    }

    /// Flushes every buffered accepted row once in canonical order; no
    /// sentinel watermark is synthesized (SCE-00 D7).
    async fn on_end(
        &mut self,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        context.check_cancelled()?;
        self.observe_context(context)?;
        if self.state.ended {
            return Ok(());
        }
        if self.state.ordered.is_empty() {
            let rows = self.take_all_buffered();
            self.emit_rows(rows, context, output).await?;
        } else {
            let last_identity = self.state.ordered.last_identity();
            let records = self.state.ordered.take_all();
            self.emit_ordered(records, last_identity, context, output)
                .await?;
        }
        self.install_context_identity(context);
        self.state.ended = true;
        Ok(())
    }

    /// Captures the finality frontier, per-entity retained history, buffered
    /// unfinalized rows, and late metrics (SCE-00 D11) as one immutable base
    /// segment plus bounded inline metadata.
    fn checkpoint(&mut self, epoch: Epoch) -> Result<crate::OperatorStateSnapshot> {
        if self
            .state
            .last_checkpoint_epoch
            .is_some_and(|previous| epoch <= previous)
        {
            return Err(checkpoint_mismatch(
                "rolling checkpoint epoch did not advance strictly",
            ));
        }
        self.materialize_ordered_buffer(None)?;
        self.state
            .histories
            .materialize_columnar(&self.compiled, &self.name, None)?;
        let charge = budget::full_state_charge(
            &self.state.buffer,
            &self.state.ordered,
            &self.state.histories,
            &self.compiled,
            &self.name,
        )?;
        self.check_state_budget(charge, &self.name)?;
        self.state.charge = charge;
        let encoded = self.encode_state(epoch)?;
        let (descriptor, segments) = match encoded {
            Some(prepared) => {
                // One shared allocation and one digest serve both the snapshot
                // and the manifest descriptor; nothing re-encodes or re-hashes.
                let (segment_id, bytes) = prepared;
                let segment = crate::StateSegment::new(bytes);
                let descriptor = self.snapshot_segment_descriptor(epoch, &segment_id, &segment)?;
                let mut segments = BTreeMap::new();
                segments.insert(segment_id, segment);
                (Some(descriptor), segments)
            }
            None => (None, BTreeMap::new()),
        };
        let inventory = StateInventory::new(descriptor.into_iter().collect())
            .map_err(|error| checkpoint_mismatch(error.to_string()))?;
        let metadata = RollingSnapshotMetadata {
            state_layout_version: self.compiled.state_layout_version,
            configuration_hash: self.compiled.configuration_hash.clone(),
            state_schema_fingerprint: self.compiled.state_schema_fingerprint.clone(),
            kernel_fingerprint: Some(self.compiled.kernel_plan.fingerprint().to_owned()),
            numerical_profile: Some(self.compiled.kernel_plan.numerical_profile().to_owned()),
            epoch,
            pipeline_fingerprint: self.state.pipeline_fingerprint.clone(),
            operator_id: self.state.operator_id.clone(),
            last_input_watermark: self.state.last_input_watermark,
            next_output_sequence: self.state.next_output_sequence,
            ended: self.state.ended,
            metrics: self.state.metrics,
            segment_inventory: inventory.segments().to_vec(),
            late_output: super::late_output::LateOutputSnapshot::new(
                self.spec.late_policy,
                self.state.next_late_output_sequence,
            ),
        };
        let Value::Object(inline_metadata) =
            serde_json::to_value(metadata).map_err(|error| format_error(&error))?
        else {
            return Err(internal_error(
                "rolling snapshot metadata did not serialize as an object",
            ));
        };
        self.state.last_checkpoint_epoch = Some(epoch);
        Ok(crate::OperatorStateSnapshot {
            inline_metadata: inline_metadata.into_iter().collect(),
            segments,
        })
    }

    /// Replaces the complete live state from one validated snapshot; a failed
    /// restore leaves the current state untouched (SCE-00 D11).
    fn restore(&mut self, snapshot: &crate::OperatorStateSnapshot) -> Result<()> {
        if snapshot.inline_metadata.is_empty() && snapshot.segments.is_empty() {
            return StreamOperator::reset(self);
        }
        let metadata = parse_snapshot_metadata(snapshot)?;
        validate_snapshot_metadata(&metadata, &self.compiled, snapshot)?;
        let next_late_output_sequence = super::late_output::LateOutputSnapshot::restore_sequence(
            self.spec.late_policy,
            metadata.late_output.as_ref(),
        )?;
        let restored = self.decode_state(&metadata, snapshot)?;
        let charge = budget::buffered_charge(restored.buffer.values(), &self.name)?.checked_add(
            budget::histories_charge(&restored.histories, &self.name)?,
            &self.name,
        )?;
        if !self.state_budget.allows(charge.rows, charge.bytes) {
            return Err(checkpoint_mismatch(
                "restored rolling state exceeds the configured state budget",
            ));
        }
        self.state = RollingStreamState {
            buffer: restored.buffer,
            ordered: ordered_stream::OrderedStreamBuffer::default(),
            histories: restored.histories,
            last_input_watermark: metadata.last_input_watermark,
            next_output_sequence: metadata.next_output_sequence,
            next_late_output_sequence,
            late_output_failed: false,
            ended: metadata.ended,
            metrics: metadata.metrics,
            pipeline_fingerprint: metadata.pipeline_fingerprint,
            operator_id: metadata.operator_id,
            last_checkpoint_epoch: Some(metadata.epoch),
            typed_kernel_state: None,
            charge,
        };
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.state = RollingStreamState::default();
        Ok(())
    }
}

impl RollingOperator {
    fn check_state_budget(&self, charge: StateCharge, node: &str) -> Result<()> {
        if self.state_budget.allows(charge.rows, charge.bytes) {
            Ok(())
        } else {
            Err(operator_error(node, "rolling state budget exceeded"))
        }
    }

    fn projected_rows_charge(
        &self,
        rows: &[BufferedRow],
        touched: &HistoryUpdates,
        node: &str,
    ) -> Result<(StateCharge, StateCharge)> {
        let current = self
            .state
            .charge
            .checked_sub(budget::buffered_charge(rows.iter(), node)?)?;
        let (old_histories, new_histories) =
            budget::changed_histories_charge(&self.state.histories, touched, node)?;
        let next = current
            .checked_sub(old_histories)?
            .checked_add(new_histories, node)?;
        self.check_state_budget(next, node)?;
        Ok((current, next))
    }

    fn materialize_histories_with_charge(
        &mut self,
        node: &str,
        observer: Option<&RollingMetricsRecorder>,
    ) -> Result<()> {
        let has_columnar = self
            .state
            .histories
            .by_entity
            .values()
            .any(|state| !state.columnar.records.is_empty());
        let old = has_columnar
            .then(|| budget::histories_charge(&self.state.histories, node))
            .transpose()?;
        self.state
            .histories
            .materialize_columnar(&self.compiled, node, observer)?;
        if let Some(old) = old {
            let new = budget::histories_charge(&self.state.histories, node)?;
            self.state.charge = self.state.charge.checked_sub(old)?.checked_add(new, node)?;
        }
        Ok(())
    }

    fn observe_context(&self, context: &StreamOperatorContext<'_>) -> Result<()> {
        super::late_output::ensure_can_continue(
            self.state.late_output_failed,
            context.operator_id(),
        )?;
        if self
            .state
            .pipeline_fingerprint
            .as_deref()
            .is_some_and(|value| value != context.job().fingerprint())
        {
            return Err(operator_error(
                context.operator_id(),
                "rolling state was used with a different pipeline fingerprint",
            ));
        }
        if self
            .state
            .operator_id
            .as_deref()
            .is_some_and(|value| value != context.operator_id())
        {
            return Err(operator_error(
                context.operator_id(),
                "rolling state was used with a different operator ID",
            ));
        }
        Ok(())
    }

    fn install_context_identity(&mut self, context: &StreamOperatorContext<'_>) {
        self.state
            .pipeline_fingerprint
            .get_or_insert_with(|| context.job().fingerprint().to_owned());
        self.state
            .operator_id
            .get_or_insert_with(|| context.operator_id().to_owned());
    }

    /// Classifies one envelope into accepted rows and the late-metric delta
    /// without touching live state (SCE-00 D7 envelope transaction).
    fn classify_envelope(
        &self,
        rows: Vec<BufferedRow>,
        watermark: Option<EventTime>,
        node_id: &str,
        observer: Option<&RollingMetricsRecorder>,
    ) -> Result<(BTreeMap<RowIdentity, BufferedRow>, LateMetricDelta)> {
        let _validation = observer.map(|recorder| recorder.stage(RollingStage::InputValidation));
        let mut accepted = BTreeMap::new();
        let mut metrics = LateRowTally::default();
        for (row_index, row) in rows.into_iter().enumerate() {
            if self.is_late(row.identity.event_time, watermark, row_index, node_id)? {
                record_late_row(&mut metrics, watermark, row.identity.event_time, node_id)?;
                continue;
            }
            let _ordering = observer.map(|recorder| recorder.stage(RollingStage::OrderingProof));
            if let Some(recorder) = observer {
                recorder.add(RollingWork::OrderProofRows, 1);
            }
            if self.state.buffer.contains_key(&row.identity) || accepted.contains_key(&row.identity)
            {
                return Err(operator_error(
                    node_id,
                    &format!(
                        "duplicate row identity at event_time_micros={}",
                        row.identity.event_time
                    ),
                ));
            }
            accepted.insert(row.identity.clone(), row);
        }
        Ok((accepted, metrics.into_delta()))
    }

    fn is_late(
        &self,
        event_time: i64,
        watermark: Option<EventTime>,
        row_index: usize,
        node_id: &str,
    ) -> Result<bool> {
        let Some(watermark) = watermark else {
            return Ok(false);
        };
        let closing = closing_coordinate(event_time, self.spec.allowed_lateness_micros, node_id)?;
        if closing > watermark.as_micros() {
            return Ok(false);
        }
        match self.spec.late_policy {
            LatePolicySpec::Error { .. } => Err(operator_error(
                node_id,
                &format!(
                    "{node_id}: late_row: envelope rejected at row_index={row_index}; event_time_micros={event_time}, closed_at_watermark_micros={}",
                    watermark.as_micros()
                ),
            )),
            LatePolicySpec::Drop { .. } | LatePolicySpec::SideOutput { .. } => Ok(true),
        }
    }

    fn closing_keys(&self, watermark: i64, node_id: &str) -> Result<Vec<RowIdentity>> {
        let mut keys = Vec::new();
        for identity in self.state.buffer.keys() {
            let closing = closing_coordinate(
                identity.event_time,
                self.spec.allowed_lateness_micros,
                node_id,
            )?;
            if closing <= watermark {
                keys.push(identity.clone());
            }
        }
        Ok(keys)
    }

    fn take_buffered(&mut self, keys: &[RowIdentity]) -> Vec<BufferedRow> {
        keys.iter()
            .filter_map(|key| self.state.buffer.remove(key))
            .collect()
    }

    fn take_all_buffered(&mut self) -> Vec<BufferedRow> {
        std::mem::take(&mut self.state.buffer)
            .into_values()
            .collect()
    }

    // Final emission keeps compute, record building, chunking, sequence
    // accounting, and history application in one ordered pass so a partial
    // failure leaves consistent in-memory state.
    // #lizard forgives
    async fn emit_rows(
        &mut self,
        rows: Vec<BufferedRow>,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let observer = context.rolling_metrics();
        self.materialize_histories_with_charge(context.operator_id(), observer)?;
        let output_schema = self.output_ports[0]
            .schema()
            .expect("rolling output always has an exact schema");
        let typed = build_typed_stream_output(
            &rows,
            &self.state.histories,
            self.state.typed_kernel_state.as_deref(),
            &self.compiled,
            output_schema,
            context.operator_id(),
            observer,
        )?;
        let (record, next_kernel_state, touched) = if let Some(typed) = typed {
            typed
        } else {
            let computed = compute_output_columns_observed(
                &rows,
                &self.state.histories,
                &self.compiled,
                context.operator_id(),
                observer,
            )?;
            let _stage = observer.map(|recorder| recorder.stage(RollingStage::ArrowOutput));
            (
                build_output_record(
                    &rows,
                    computed.columns,
                    output_schema,
                    context.operator_id(),
                )?,
                None,
                computed.touched,
            )
        };
        let (current_charge, next_charge) =
            match self.projected_rows_charge(&rows, &touched, context.operator_id()) {
                Ok(charge) => charge,
                Err(error) => {
                    for row in rows {
                        self.state.buffer.insert(row.identity.clone(), row);
                    }
                    if let Ok(charge) = budget::full_state_charge(
                        &self.state.buffer,
                        &self.state.ordered,
                        &self.state.histories,
                        &self.compiled,
                        context.operator_id(),
                    ) {
                        self.state.charge = charge;
                    }
                    return Err(error);
                }
            };
        self.state.charge = current_charge;
        if let Some(recorder) = observer {
            recorder.add(RollingWork::OutputRowsPrepared, record.num_rows());
        }
        let budget_stage = observer.map(|recorder| recorder.stage(RollingStage::BudgetPreparation));
        let batches = chunk_output_record(
            &record,
            context.operator_id(),
            self.state.next_output_sequence,
            context.output_budget(),
        )?;
        let chunk_count = u64::try_from(batches.len()).map_err(|_| {
            operator_error(
                context.operator_id(),
                "output chunk count does not fit the sequence range",
            )
        })?;
        if let Some(recorder) = observer {
            recorder.add(RollingWork::OutputChunksPrepared, batches.len());
        }
        drop(budget_stage);
        for batch in batches {
            output.emit("output", batch).await?;
        }
        self.state.next_output_sequence = self
            .state
            .next_output_sequence
            .checked_add(chunk_count)
            .ok_or_else(|| operator_error(context.operator_id(), "output sequence overflowed"))?;
        let _stage = observer.map(|recorder| recorder.stage(RollingStage::HistoryMaintenance));
        self.state.histories.apply(touched);
        self.state.typed_kernel_state = next_kernel_state.map(Box::new);
        self.state.charge = next_charge;
        Ok(())
    }
}

impl RollingOperator {
    fn encode_state(&self, epoch: Epoch) -> Result<Option<(String, Vec<u8>)>> {
        let row_count = self
            .state
            .histories
            .by_entity
            .values()
            .map(|state| state.rows.len())
            .sum::<usize>()
            + self.state.buffer.len()
            + self
                .state
                .histories
                .by_entity
                .values()
                .map(|state| {
                    state
                        .windows
                        .iter()
                        .filter(|window| {
                            matches!(window, WindowState::Ewma(state) if state.valid_count > 0)
                        })
                        .count()
                })
                .sum::<usize>();
        if row_count == 0 {
            return Ok(None);
        }
        let pipeline_fingerprint =
            self.state.pipeline_fingerprint.clone().ok_or_else(|| {
                internal_error("rolling state is missing its pipeline fingerprint")
            })?;
        let operator_id = self
            .state
            .operator_id
            .clone()
            .ok_or_else(|| internal_error("rolling state is missing its operator identity"))?;
        let segment_id = format!("base-{:020}-00000000", epoch.as_u64());
        let bytes = encode_state_segment(
            &self.state.histories,
            &self.state.buffer,
            self.input_ports[0]
                .schema()
                .expect("rolling input always has an exact schema"),
            &self.compiled,
            &pipeline_fingerprint,
            &operator_id,
        )?;
        Ok(Some((segment_id, bytes)))
    }

    fn snapshot_segment_descriptor(
        &self,
        epoch: Epoch,
        segment_id: &str,
        segment: &crate::StateSegment,
    ) -> Result<SegmentDescriptor> {
        let operator_id = self.state.operator_id.as_deref().ok_or_else(|| {
            checkpoint_mismatch("rolling segment is missing its operator identity")
        })?;
        let relative_path = format!(
            "committed/{operator_id}/{:020}-{segment_id}.arrow",
            epoch.as_u64()
        );
        let byte_len = u64::try_from(segment.bytes().len())
            .map_err(|_| internal_error("rolling segment length does not fit u64"))?;
        Ok(SegmentDescriptor {
            kind: SegmentKind::Base,
            state_layout_version: self.compiled.state_layout_version,
            schema_fingerprint: self.compiled.state_schema_fingerprint.clone(),
            handle: StateHandle::new(
                operator_id,
                epoch,
                segment_id,
                &relative_path,
                byte_len,
                segment.sha256(),
            )?,
        })
    }

    fn decode_state(
        &self,
        metadata: &RollingSnapshotMetadata,
        snapshot: &crate::OperatorStateSnapshot,
    ) -> Result<DecodedRollingState> {
        let segments = snapshot_segments(snapshot, &metadata.segment_inventory)?;
        let Some(bytes) = segments.into_iter().next() else {
            return Ok(DecodedRollingState::default());
        };
        decode_state_segment(
            &bytes,
            self.input_ports[0]
                .schema()
                .expect("rolling input always has an exact schema"),
            &self.compiled,
            metadata,
        )
    }
}

#[derive(Default)]
struct DecodedRollingState {
    buffer: BTreeMap<RowIdentity, BufferedRow>,
    histories: RollingHistories,
}

/// Serialized state-row ordering key: row kind, entity, identity, and the
/// per-entity history position.
type StateRowOrderKey = (u8, Vec<Option<KeyValue>>, RowIdentity, Option<u64>);

fn state_fields(input_schema: &Schema, state_layout_version: u32) -> Vec<Field> {
    if state_layout_version == ROLLING_COLUMNAR_STATE_LAYOUT_VERSION {
        return state_v3::state_fields(input_schema);
    }
    let mut fields = vec![
        Field::new("_state_kind", DataType::UInt8, false),
        Field::new("_entity_position", DataType::UInt64, true),
    ];
    fields.extend(
        input_schema
            .fields()
            .iter()
            .map(|field| Field::new(field.name(), field.data_type().clone(), true)),
    );
    if state_layout_version == ROLLING_EWMA_STATE_LAYOUT_VERSION {
        fields.extend([
            Field::new("_ewma_group", DataType::UInt64, true),
            Field::new("_ewma_valid_count", DataType::UInt64, true),
            Field::new("_ewma_value", DataType::Float64, true),
        ]);
    }
    fields
}

fn state_schema_fingerprint(input_schema: &Schema, state_layout_version: u32) -> String {
    let schema = Schema::new(state_fields(input_schema, state_layout_version));
    let mut dictionary_tracker = DictionaryTracker::new(true);
    let encoded = IpcSchemaEncoder::new()
        .with_dictionary_tracker(&mut dictionary_tracker)
        .schema_to_fb(&schema);
    hex::encode(Sha256::digest(encoded.finished_data()))
}

fn state_schema(
    input_schema: &Schema,
    compiled: &CompiledRollingSpec,
    pipeline_fingerprint: &str,
    operator_id: &str,
) -> Schema {
    let mut metadata = HashMap::from([
        (
            "calc_flow.state_layout_version".into(),
            compiled.state_layout_version.to_string(),
        ),
        (
            "calc_flow.pipeline_fingerprint".into(),
            pipeline_fingerprint.into(),
        ),
        ("calc_flow.operator_id".into(), operator_id.into()),
        (
            "calc_flow.operator_configuration_hash".into(),
            compiled.configuration_hash.clone(),
        ),
        (
            "calc_flow.state_schema_fingerprint".into(),
            compiled.state_schema_fingerprint.clone(),
        ),
    ]);
    if compiled.state_layout_version == ROLLING_COLUMNAR_STATE_LAYOUT_VERSION {
        metadata.insert(
            "calc_flow.rolling_kernel_fingerprint".into(),
            compiled.kernel_plan.fingerprint().to_owned(),
        );
        metadata.insert(
            "calc_flow.numerical_profile".into(),
            compiled.kernel_plan.numerical_profile().to_owned(),
        );
    }
    Schema::new_with_metadata(
        state_fields(input_schema, compiled.state_layout_version),
        metadata,
    )
}

// State serialization writes deterministic history and buffer rows column
// by column with checked conversions for every value class.
// #lizard forgives
fn encode_state_segment(
    histories: &RollingHistories,
    buffer: &BTreeMap<RowIdentity, BufferedRow>,
    input_schema: &Schema,
    compiled: &CompiledRollingSpec,
    pipeline_fingerprint: &str,
    operator_id: &str,
) -> Result<Vec<u8>> {
    if compiled.state_layout_version == ROLLING_COLUMNAR_STATE_LAYOUT_VERSION {
        return state_v3::encode(
            histories,
            buffer,
            input_schema,
            compiled,
            pipeline_fingerprint,
            operator_id,
        );
    }
    encode_state_segment_legacy(
        histories,
        buffer,
        input_schema,
        compiled,
        pipeline_fingerprint,
        operator_id,
    )
}

fn encode_state_segment_legacy(
    histories: &RollingHistories,
    buffer: &BTreeMap<RowIdentity, BufferedRow>,
    input_schema: &Schema,
    compiled: &CompiledRollingSpec,
    pipeline_fingerprint: &str,
    operator_id: &str,
) -> Result<Vec<u8>> {
    let width = input_schema.fields().len();
    let mut kinds = Vec::new();
    let mut positions: Vec<Option<u64>> = Vec::new();
    let mut columns: Vec<Vec<Option<ScalarValue>>> = vec![Vec::new(); width];
    let mut ewma_groups = Vec::new();
    let mut ewma_counts = Vec::new();
    let mut ewma_values = Vec::new();
    let mut push_row =
        |kind: u8, position: Option<u64>, values: &[ScalarValue], ewma: Option<(u64, u64, f64)>| {
            kinds.push(kind);
            positions.push(position);
            for (index, column) in columns.iter_mut().enumerate() {
                column.push(values.get(index).cloned());
            }
            if compiled.state_layout_version == ROLLING_EWMA_STATE_LAYOUT_VERSION {
                ewma_groups.push(ewma.map(|state| state.0));
                ewma_counts.push(ewma.map(|state| state.1));
                ewma_values.push(ewma.map(|state| state.2));
            }
        };
    for state in histories.by_entity.values() {
        for (position, values) in state.rows.iter().enumerate() {
            let position = u64::try_from(position)
                .map_err(|_| internal_error("rolling history position does not fit u64"))?;
            push_row(0, Some(position), values, None);
        }
    }
    for row in buffer.values() {
        push_row(1, None, &row.values, None);
    }
    if compiled.state_layout_version == ROLLING_EWMA_STATE_LAYOUT_VERSION {
        for (entity, state) in &histories.by_entity {
            let values = ewma_entity_values(entity, input_schema, compiled)?;
            for (group, window) in state.windows.iter().enumerate() {
                let WindowState::Ewma(accumulator) = window else {
                    continue;
                };
                if accumulator.valid_count == 0 {
                    continue;
                }
                let group = u64::try_from(group)
                    .map_err(|_| internal_error("rolling EWMA group does not fit u64"))?;
                push_row(
                    2,
                    None,
                    &values,
                    Some((group, accumulator.valid_count, accumulator.value)),
                );
            }
        }
    }
    let schema = state_schema(input_schema, compiled, pipeline_fingerprint, operator_id);
    let mut arrays: Vec<ArrayRef> = vec![
        Arc::new(UInt8Array::from(kinds)),
        Arc::new(UInt64Array::from(positions)),
    ];
    for column in columns {
        arrays.push(
            ScalarValue::iter_to_array(
                column
                    .into_iter()
                    .map(|value| value.expect("rolling state rows carry full typed values")),
            )
            .map_err(|error| state_format(format!("rolling state array failed: {error}")))?,
        );
    }
    if compiled.state_layout_version == ROLLING_EWMA_STATE_LAYOUT_VERSION {
        arrays.extend([
            Arc::new(UInt64Array::from(ewma_groups)) as ArrayRef,
            Arc::new(UInt64Array::from(ewma_counts)) as ArrayRef,
            Arc::new(Float64Array::from(ewma_values)) as ArrayRef,
        ]);
    }
    let record = RecordBatch::try_new(Arc::new(schema.clone()), arrays)
        .map_err(|error| state_format(format!("rolling state batch is invalid: {error}")))?;
    let mut bytes = Vec::new();
    {
        let mut writer = FileWriter::try_new(&mut bytes, &schema)
            .map_err(|error| state_format(format!("rolling state IPC header failed: {error}")))?;
        writer
            .write(&record)
            .map_err(|error| state_format(format!("rolling state IPC write failed: {error}")))?;
        writer
            .finish()
            .map_err(|error| state_format(format!("rolling state IPC finish failed: {error}")))?;
    }
    Ok(bytes)
}

// State decode intentionally validates header metadata, shape, deterministic
// order, and per-row invariants before any state is installed.
// #lizard forgives
fn ewma_state_arrays(
    record: &RecordBatch,
    width: usize,
    enabled: bool,
) -> Result<(
    Option<&UInt64Array>,
    Option<&UInt64Array>,
    Option<&Float64Array>,
)> {
    if !enabled {
        return Ok((None, None, None));
    }
    let group = record
        .column(width + 2)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| state_format("rolling EWMA group column has the wrong type".to_owned()))?;
    let count = record
        .column(width + 3)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| state_format("rolling EWMA count column has the wrong type".to_owned()))?;
    let value = record
        .column(width + 4)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| state_format("rolling EWMA value column has the wrong type".to_owned()))?;
    Ok((Some(group), Some(count), Some(value)))
}

fn decode_state_segment(
    bytes: &[u8],
    input_schema: &Schema,
    compiled: &CompiledRollingSpec,
    metadata: &RollingSnapshotMetadata,
) -> Result<DecodedRollingState> {
    if metadata.state_layout_version == ROLLING_COLUMNAR_STATE_LAYOUT_VERSION {
        return state_v3::decode(bytes, input_schema, compiled, metadata);
    }
    let mut legacy = compiled.clone();
    legacy.state_layout_version = metadata.state_layout_version;
    legacy
        .state_schema_fingerprint
        .clone_from(&metadata.state_schema_fingerprint);
    decode_state_segment_legacy(bytes, input_schema, &legacy, metadata)
}

fn decode_state_segment_legacy(
    bytes: &[u8],
    input_schema: &Schema,
    compiled: &CompiledRollingSpec,
    metadata: &RollingSnapshotMetadata,
) -> Result<DecodedRollingState> {
    let reader = FileReader::try_new(Cursor::new(bytes), None)
        .map_err(|error| state_format(format!("rolling state IPC open failed: {error}")))?;
    validate_segment_schema_metadata(reader.schema().metadata(), metadata, compiled)?;
    let batches = reader
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|error| state_format(format!("rolling state IPC read failed: {error}")))?;
    let [record] = batches.try_into().map_err(|_| {
        state_format("rolling state segment must contain exactly one record batch".to_owned())
    })?;
    let width = input_schema.fields().len();
    let exponential_width =
        usize::from(compiled.state_layout_version == ROLLING_EWMA_STATE_LAYOUT_VERSION) * 3;
    if record.num_columns() != width + 2 + exponential_width {
        return Err(state_format(
            "rolling state segment column count does not match the state schema".to_owned(),
        ));
    }
    let kinds = record
        .column(0)
        .as_any()
        .downcast_ref::<UInt8Array>()
        .ok_or_else(|| state_format("rolling state kind column has the wrong type".to_owned()))?;
    let positions = record
        .column(1)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| {
            state_format("rolling state position column has the wrong type".to_owned())
        })?;
    let (ewma_groups, ewma_counts, ewma_values) =
        ewma_state_arrays(&record, width, exponential_width > 0)?;
    let mut decoded = DecodedRollingState::default();
    let mut previous: Option<StateRowOrderKey> = None;
    for row_index in 0..record.num_rows() {
        let values = (2..width + 2)
            .map(|index| {
                ScalarValue::try_from_array(record.column(index), row_index).map_err(|error| {
                    state_format(format!("rolling state row could not be read: {error}"))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let position = (!positions.is_null(row_index)).then(|| positions.value(row_index));
        let ewma = match (ewma_groups, ewma_counts, ewma_values) {
            (Some(groups), Some(counts), Some(values)) => match (
                (!groups.is_null(row_index)).then(|| groups.value(row_index)),
                (!counts.is_null(row_index)).then(|| counts.value(row_index)),
                (!values.is_null(row_index)).then(|| values.value(row_index)),
            ) {
                (Some(group), Some(count), Some(value)) => Some((group, count, value)),
                (None, None, None) => None,
                _ => {
                    return Err(state_format(
                        "rolling EWMA state columns are only partially populated".to_owned(),
                    ));
                }
            },
            (None, None, None) => None,
            _ => unreachable!("EWMA state arrays are discovered together"),
        };
        decode_state_row(
            kinds.value(row_index),
            position,
            values,
            ewma,
            &mut decoded,
            compiled,
            &mut previous,
        )?;
    }
    validate_decoded_state(&decoded, compiled)?;
    rebuild_windows(&mut decoded.histories, compiled, "rolling")?;
    Ok(decoded)
}

fn validate_segment_schema_metadata(
    metadata: &HashMap<String, String>,
    snapshot: &RollingSnapshotMetadata,
    _compiled: &CompiledRollingSpec,
) -> Result<()> {
    let mut expected = vec![
        (
            "calc_flow.state_layout_version",
            snapshot.state_layout_version.to_string(),
        ),
        (
            "calc_flow.pipeline_fingerprint",
            snapshot.pipeline_fingerprint.clone().unwrap_or_default(),
        ),
        (
            "calc_flow.operator_id",
            snapshot.operator_id.clone().unwrap_or_default(),
        ),
        (
            "calc_flow.operator_configuration_hash",
            snapshot.configuration_hash.clone(),
        ),
        (
            "calc_flow.state_schema_fingerprint",
            snapshot.state_schema_fingerprint.clone(),
        ),
    ];
    if snapshot.state_layout_version == ROLLING_COLUMNAR_STATE_LAYOUT_VERSION {
        expected.extend([
            (
                "calc_flow.rolling_kernel_fingerprint",
                snapshot.kernel_fingerprint.clone().unwrap_or_default(),
            ),
            (
                "calc_flow.numerical_profile",
                snapshot.numerical_profile.clone().unwrap_or_default(),
            ),
        ]);
    }
    for (key, value) in expected {
        if metadata.get(key).map(String::as_str) != Some(value.as_str()) {
            return Err(checkpoint_mismatch(format!(
                "rolling state segment metadata {key} does not match the snapshot"
            )));
        }
    }
    Ok(())
}

fn decode_state_row(
    kind: u8,
    position: Option<u64>,
    values: Vec<ScalarValue>,
    ewma: Option<(u64, u64, f64)>,
    decoded: &mut DecodedRollingState,
    compiled: &CompiledRollingSpec,
    previous: &mut Option<StateRowOrderKey>,
) -> Result<()> {
    if kind == 2 {
        return decode_ewma_state_row(position, &values, ewma, decoded, compiled, previous);
    }
    if ewma.is_some() {
        return Err(state_format(
            "rolling history or buffer row carries EWMA state".to_owned(),
        ));
    }
    let row = buffered_row_from_values(values, compiled)?;
    let ordering_key = (
        kind,
        row.identity.entity.clone(),
        row.identity.clone(),
        position,
    );
    if let Some(prior) = previous.as_ref()
        && !state_rows_in_order(prior, &ordering_key)
    {
        return Err(state_format(
            "rolling state segment rows are not in deterministic key order".to_owned(),
        ));
    }
    match kind {
        0 => {
            let state = decoded
                .histories
                .by_entity
                .entry(row.identity.entity.clone())
                .or_default();
            let expected = u64::try_from(state.rows.len()).unwrap_or(u64::MAX);
            if position != Some(expected) {
                return Err(state_format(
                    "rolling state segment history positions are not contiguous".to_owned(),
                ));
            }
            state.rows.push_back(row.values);
        }
        1 => {
            if decoded.buffer.insert(row.identity.clone(), row).is_some() {
                return Err(state_format(
                    "rolling state segment contains a duplicate buffered identity".to_owned(),
                ));
            }
        }
        other => {
            return Err(state_format(format!(
                "rolling state segment contains unknown row kind {other}"
            )));
        }
    }
    *previous = Some(ordering_key);
    Ok(())
}

fn decode_ewma_state_row(
    position: Option<u64>,
    values: &[ScalarValue],
    ewma: Option<(u64, u64, f64)>,
    decoded: &mut DecodedRollingState,
    compiled: &CompiledRollingSpec,
    previous: &mut Option<StateRowOrderKey>,
) -> Result<()> {
    if compiled.state_layout_version != ROLLING_EWMA_STATE_LAYOUT_VERSION {
        return Err(state_format(
            "rolling layout v1 contains an EWMA state row".to_owned(),
        ));
    }
    if position.is_some() {
        return Err(state_format(
            "rolling EWMA state row carries a history position".to_owned(),
        ));
    }
    let (group, valid_count, value) = ewma.ok_or_else(|| {
        state_format("rolling EWMA state row is missing its accumulator".to_owned())
    })?;
    if valid_count == 0 {
        return Err(state_format(
            "rolling EWMA state row has a zero valid count".to_owned(),
        ));
    }
    let group_index = usize::try_from(group)
        .map_err(|_| state_format("rolling EWMA group does not fit usize".to_owned()))?;
    if !matches!(
        compiled.window_groups.get(group_index),
        Some(CompiledWindowGroup::Ewma { .. })
    ) {
        return Err(state_format(
            "rolling EWMA state row references a non-EWMA group".to_owned(),
        ));
    }
    let entity = ewma_entity_from_values(values, compiled)?;
    let ordering_key = (
        2,
        entity.clone(),
        RowIdentity {
            event_time: 0,
            entity: entity.clone(),
            sequence: Vec::new(),
        },
        Some(group),
    );
    if let Some(prior) = previous.as_ref()
        && !state_rows_in_order(prior, &ordering_key)
    {
        return Err(state_format(
            "rolling state segment rows are not in deterministic key order".to_owned(),
        ));
    }
    let state = decoded
        .histories
        .by_entity
        .entry(entity)
        .or_insert_with(|| EntityRollingState::fresh(compiled));
    if state.windows.is_empty() {
        state.windows = fresh_windows(compiled);
    }
    let WindowState::Ewma(accumulator) = &mut state.windows[group_index] else {
        unreachable!("validated EWMA group has EWMA state")
    };
    if accumulator.valid_count != 0 {
        return Err(state_format(
            "rolling state segment contains a duplicate EWMA accumulator".to_owned(),
        ));
    }
    *accumulator = EwmaAccumulator { valid_count, value };
    *previous = Some(ordering_key);
    Ok(())
}

fn ewma_entity_from_values(
    values: &[ScalarValue],
    compiled: &CompiledRollingSpec,
) -> Result<Vec<Option<KeyValue>>> {
    for (index, value) in values.iter().enumerate() {
        if !compiled
            .partition_columns
            .iter()
            .any(|column| column.index == index)
            && !value.is_null()
        {
            return Err(state_format(
                "rolling EWMA state row populates a non-entity field".to_owned(),
            ));
        }
    }
    compiled
        .partition_columns
        .iter()
        .map(|column| KeyValue::from_nullable_scalar(&values[column.index], "rolling EWMA"))
        .collect()
}

fn state_rows_in_order(prior: &StateRowOrderKey, current: &StateRowOrderKey) -> bool {
    if prior.0 != current.0 {
        return prior.0 < current.0;
    }
    match prior.0 {
        0 => {
            if prior.1 != current.1 {
                return prior.1 < current.1;
            }
            match (prior.3, current.3) {
                (Some(left), Some(right)) => left < right,
                _ => false,
            }
        }
        1 => prior.2 < current.2,
        2 => {
            prior.1 < current.1
                || (prior.1 == current.1
                    && matches!((prior.3, current.3), (Some(left), Some(right)) if left < right))
        }
        _ => false,
    }
}

fn buffered_row_from_values(
    values: Vec<ScalarValue>,
    compiled: &CompiledRollingSpec,
) -> Result<BufferedRow> {
    let event_time = match &values[compiled.event_time_index] {
        ScalarValue::TimestampMicrosecond(Some(value), _) => *value,
        _ => {
            return Err(state_format(
                "rolling state row has a null or non-timestamp event time".to_owned(),
            ));
        }
    };
    let entity = compiled
        .partition_columns
        .iter()
        .map(|column| KeyValue::from_nullable_scalar(&values[column.index], "rolling"))
        .collect::<Result<Vec<_>>>()?;
    let sequence = compiled
        .sequence_columns
        .iter()
        .map(|column| KeyValue::from_required_scalar(&values[column.index], "rolling"))
        .collect::<Result<Vec<_>>>()?;
    Ok(BufferedRow::new(entity, sequence, event_time, values))
}

fn validate_decoded_state(
    decoded: &DecodedRollingState,
    compiled: &CompiledRollingSpec,
) -> Result<()> {
    let max_retained = usize::try_from(compiled.max_row_retention)
        .map_err(|_| internal_error("rolling max retained rows does not fit usize"))?;
    for state in decoded.histories.by_entity.values() {
        let bound = compiled
            .max_duration_micros
            .zip(
                state
                    .rows
                    .back()
                    .map(|values| history_event_time(values, compiled)),
            )
            .map(|(micros, last)| i128::from(last) - i128::from(micros));
        for (index, values) in state.rows.iter().enumerate() {
            let needed_by_count = state.rows.len() - index <= max_retained;
            let needed_by_time =
                bound.is_some_and(|bound| i128::from(history_event_time(values, compiled)) > bound);
            if !needed_by_count && !needed_by_time {
                return Err(state_format(
                    "rolling state segment retains more history than the declared frames"
                        .to_owned(),
                ));
            }
        }
    }
    Ok(())
}

/// Rebuilds every window accumulator as the ordered fold over the retained
/// history tail; the segment stores rows only, and the accumulator is the
/// deterministic function of those rows frozen in D5/D11. Extrema groups
/// fold pushes and expiries so the rebuilt queue front is the window
/// extremum, exactly as the live slide left it (SCE-08).
fn rebuild_windows(
    histories: &mut RollingHistories,
    compiled: &CompiledRollingSpec,
    node_id: &str,
) -> Result<()> {
    for state in histories.by_entity.values_mut() {
        let persisted = std::mem::take(&mut state.windows);
        let mut windows = fresh_windows(compiled);
        let last_time = state
            .rows
            .back()
            .map(|values| history_event_time(values, compiled));
        for (group_index, group) in compiled.window_groups.iter().enumerate() {
            match group {
                CompiledWindowGroup::Numeric {
                    input_index,
                    frame,
                    sum_class,
                } => {
                    let WindowState::Numeric(accumulator) = &mut windows[group_index] else {
                        return Err(internal_error("rolling numeric group state mismatch"));
                    };
                    let start = retained_window_start(*frame, state, compiled);
                    if spec_uses_stable_v2(compiled) && *sum_class == SumClass::Float {
                        *accumulator = kernel::stable_v2_float64_accumulator(
                            state.rows.iter().skip(start).filter_map(|values| {
                                let value = &values[*input_index];
                                is_valid_sample(value).then(|| float_sample(value))
                            }),
                            node_id,
                        )?;
                    } else {
                        for values in state.rows.iter().skip(start) {
                            let value = &values[*input_index];
                            if is_valid_sample(value) {
                                accumulator.add(value, node_id)?;
                            }
                        }
                    }
                    if let CompiledFrame::Duration(micros) = frame {
                        accumulator.expired_through = expired_through_bound(last_time, *micros);
                    }
                }
                CompiledWindowGroup::Extrema {
                    input_index, frame, ..
                } => {
                    let WindowState::Extrema(accumulator) = &mut windows[group_index] else {
                        return Err(internal_error("rolling extrema group state mismatch"));
                    };
                    rebuild_extrema_group(
                        accumulator,
                        state,
                        *input_index,
                        *frame,
                        compiled,
                        node_id,
                    )?;
                }
                CompiledWindowGroup::Pair {
                    left_index,
                    right_index,
                    frame,
                } => {
                    let WindowState::Pair(accumulator) = &mut windows[group_index] else {
                        return Err(internal_error("rolling pair group state mismatch"));
                    };
                    let start = retained_window_start(*frame, state, compiled);
                    if spec_uses_stable_v2(compiled) {
                        *accumulator = kernel::stable_v2_pair_accumulator(
                            state.rows.iter().skip(start).filter_map(|values| {
                                let x = &values[*left_index];
                                let y = &values[*right_index];
                                (is_valid_sample(x) && is_valid_sample(y))
                                    .then(|| (float_sample(x), float_sample(y)))
                            }),
                            node_id,
                        )?;
                    } else {
                        for values in state.rows.iter().skip(start) {
                            let x = &values[*left_index];
                            let y = &values[*right_index];
                            if is_valid_sample(x) && is_valid_sample(y) {
                                accumulator.add(x, y, node_id)?;
                            }
                        }
                    }
                    if let CompiledFrame::Duration(micros) = frame {
                        accumulator.expired_through = expired_through_bound(last_time, *micros);
                    }
                }
                CompiledWindowGroup::Ewma { .. } => {
                    if let Some(WindowState::Ewma(saved)) = persisted.get(group_index) {
                        windows[group_index] = WindowState::Ewma(*saved);
                    }
                }
            }
        }
        state.windows = windows;
    }
    Ok(())
}

/// Duration-frame expiry bound for a rebuild: the last retained row's
/// window lower edge, or "nothing expired" when no row is retained.
fn expired_through_bound(last_time: Option<i64>, micros: u64) -> i128 {
    last_time.map_or(i128::MIN, |last| i128::from(last) - i128::from(micros))
}

/// Canonical expiry key of one retained history row.
fn history_extrema_key(
    values: &[ScalarValue],
    compiled: &CompiledRollingSpec,
    node_id: &str,
) -> Result<ExtremaKey> {
    Ok(ExtremaKey {
        event_time: history_event_time(values, compiled),
        sequence: compiled
            .sequence_columns
            .iter()
            .map(|column| KeyValue::from_required_scalar(&values[column.index], node_id))
            .collect::<Result<Vec<_>>>()?,
    })
}

/// Rebuilds one extrema queue as the ordered push/expire fold over the
/// retained rows, mirroring the live slide so queue front, expiry keys, and
/// valid count match an uninterrupted run exactly.
fn rebuild_extrema_group(
    accumulator: &mut ExtremaAccumulator,
    state: &EntityRollingState,
    input_index: usize,
    frame: CompiledFrame,
    compiled: &CompiledRollingSpec,
    node_id: &str,
) -> Result<()> {
    let rows = usize::try_from(frame.rows())
        .map_err(|_| internal_error("rolling frame rows do not fit usize"))?;
    let mut cursor = 0_usize;
    for position in 0..state.rows.len() {
        let values = &state.rows[position];
        let key = history_extrema_key(values, compiled, node_id)?;
        let value = &values[input_index];
        if is_valid_sample(value) {
            accumulator.add(key.clone(), value.clone());
        }
        match frame {
            CompiledFrame::Rows(..) => {
                if position >= rows {
                    let leaving_row = &state.rows[position - rows];
                    if is_valid_sample(&leaving_row[input_index]) {
                        accumulator.remove();
                    }
                    accumulator.expire_through_key(&history_extrema_key(
                        leaving_row,
                        compiled,
                        node_id,
                    )?);
                }
            }
            CompiledFrame::Duration(micros) => {
                let bound = i128::from(key.event_time) - i128::from(micros);
                while cursor < position
                    && i128::from(history_event_time(&state.rows[cursor], compiled)) <= bound
                {
                    if is_valid_sample(&state.rows[cursor][input_index]) {
                        accumulator.remove();
                    }
                    cursor += 1;
                }
                accumulator.expire_through_time(bound);
                accumulator.expired_through = bound;
            }
        }
    }
    Ok(())
}

/// Start position of the retained window of the last retained row: the last
/// `rows` positions for row frames, the positions with event time in
/// `(t_last - d, t_last]` for duration frames. Restore-time only, so a
/// linear scan over the retained tail is acceptable.
fn retained_window_start(
    frame: CompiledFrame,
    state: &EntityRollingState,
    compiled: &CompiledRollingSpec,
) -> usize {
    let len = state.rows.len();
    let last_time = state
        .rows
        .back()
        .map(|values| history_event_time(values, compiled));
    match frame {
        CompiledFrame::Rows(rows) => {
            let rows = usize::try_from(rows).unwrap_or(usize::MAX);
            len.saturating_sub(rows)
        }
        CompiledFrame::Duration(micros) => match last_time {
            None => 0,
            Some(last) => {
                let bound = i128::from(last) - i128::from(micros);
                state
                    .rows
                    .iter()
                    .position(|values| i128::from(history_event_time(values, compiled)) > bound)
                    .unwrap_or(len)
            }
        },
    }
}

fn parse_snapshot_metadata(
    snapshot: &crate::OperatorStateSnapshot,
) -> Result<RollingSnapshotMetadata> {
    serde_json::from_value::<RollingSnapshotMetadata>(Value::Object(
        snapshot.inline_metadata.clone().into_iter().collect(),
    ))
    .map_err(|error| format_error(&error))
}

fn validate_snapshot_metadata(
    metadata: &RollingSnapshotMetadata,
    compiled: &CompiledRollingSpec,
    snapshot: &crate::OperatorStateSnapshot,
) -> Result<StateInventory> {
    let expected_schema_fingerprint =
        if metadata.state_layout_version == compiled.state_layout_version {
            &compiled.state_schema_fingerprint
        } else if metadata.state_layout_version == compiled.legacy_state_layout_version {
            &compiled.legacy_state_schema_fingerprint
        } else {
            return Err(checkpoint_mismatch(format!(
                "rolling state layout version {} does not match current {} or declared legacy {}",
                metadata.state_layout_version,
                compiled.state_layout_version,
                compiled.legacy_state_layout_version
            )));
        };
    if metadata.configuration_hash != compiled.configuration_hash {
        return Err(checkpoint_mismatch(
            "rolling operator configuration hash does not match the compiled operator",
        ));
    }
    if metadata.state_schema_fingerprint != *expected_schema_fingerprint {
        return Err(checkpoint_mismatch(
            "rolling state schema fingerprint does not match the compiled operator",
        ));
    }
    if metadata.state_layout_version == ROLLING_COLUMNAR_STATE_LAYOUT_VERSION
        && (metadata.kernel_fingerprint.as_deref() != Some(compiled.kernel_plan.fingerprint())
            || metadata.numerical_profile.as_deref()
                != Some(compiled.kernel_plan.numerical_profile()))
    {
        return Err(checkpoint_mismatch(
            "rolling kernel fingerprint or numerical profile does not match the compiled operator",
        ));
    }
    super::checkpoint::validate_inventory(
        &super::checkpoint::SnapshotContract {
            name: "rolling",
            state_layout_version: metadata.state_layout_version,
            schema_fingerprint: &metadata.state_schema_fingerprint,
            epoch: metadata.epoch,
            pipeline_fingerprint: metadata.pipeline_fingerprint.as_deref(),
            operator_id: metadata.operator_id.as_deref(),
            segment_inventory: metadata.segment_inventory.clone(),
        },
        snapshot,
    )
}

fn snapshot_segments(
    snapshot: &crate::OperatorStateSnapshot,
    inventory: &[SegmentDescriptor],
) -> Result<Vec<Arc<Vec<u8>>>> {
    inventory
        .iter()
        .map(|descriptor| {
            let segment_id = descriptor.handle.segment_id();
            let segment = snapshot.segments.get(segment_id).ok_or_else(|| {
                checkpoint_mismatch(format!(
                    "rolling snapshot is missing segment {segment_id:?}"
                ))
            })?;
            // A fresh session revalidates every referenced segment byte
            // against the manifest handle before any state is installed.
            let bytes = segment.bytes();
            if u64::try_from(bytes.len()).ok() != Some(descriptor.handle.byte_len()) {
                return Err(checkpoint_mismatch(
                    "rolling snapshot segment byte length does not match its handle",
                ));
            }
            if hex::encode(Sha256::digest(bytes)) != descriptor.handle.sha256() {
                return Err(checkpoint_mismatch(
                    "rolling snapshot segment checksum does not match its handle",
                ));
            }
            Ok(segment.bytes_arc())
        })
        .collect()
}

fn deserialize_required_option<'de, D, T>(
    deserializer: D,
) -> std::result::Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer)
}

fn closing_coordinate(event_time: i64, allowed_lateness_micros: u64, node_id: &str) -> Result<i64> {
    let lateness = i64::try_from(allowed_lateness_micros).map_err(|_| {
        operator_error(
            node_id,
            "allowed lateness exceeds the representable event-time range",
        )
    })?;
    event_time.checked_add(lateness).ok_or_else(|| {
        operator_error(
            node_id,
            "finality coordinate overflowed the event-time range",
        )
    })
}

/// Splits one rolling output record into edge-budget-sized messages via the
/// shared operator chunker.
fn chunk_output_record(
    record: &RecordBatch,
    operator_id: &str,
    first_sequence: u64,
    budget: crate::EdgeBudget,
) -> Result<Vec<Batch>> {
    super::output_chunk::chunk_output_record(
        record,
        operator_id,
        first_sequence,
        budget,
        super::output_chunk::OutputChunkErrors::ROLLING,
    )
}

/// Reads every input row with its canonical identity; null event-time or
/// sequence values are malformed runtime data (SCE-00 D4/D12).
fn read_buffered_rows(
    table: &TableBatch,
    compiled: &CompiledRollingSpec,
    node_id: &str,
) -> Result<Vec<BufferedRow>> {
    let mut rows = Vec::with_capacity(table.batches().iter().map(RecordBatch::num_rows).sum());
    for record in table.batches() {
        for row_index in 0..record.num_rows() {
            rows.push(read_buffered_row(record, row_index, compiled, node_id)?);
        }
    }
    Ok(rows)
}

fn read_buffered_row(
    record: &RecordBatch,
    row_index: usize,
    compiled: &CompiledRollingSpec,
    node_id: &str,
) -> Result<BufferedRow> {
    let mut values = Vec::with_capacity(record.num_columns());
    for column in record.columns() {
        values.push(
            ScalarValue::try_from_array(column, row_index).map_err(|error| {
                operator_error(
                    node_id,
                    &format!("rolling input row could not be read: {error}"),
                )
            })?,
        );
    }
    let event_time = match &values[compiled.event_time_index] {
        ScalarValue::TimestampMicrosecond(Some(value), _) => *value,
        _ => {
            return Err(operator_error(
                node_id,
                "rolling event-time value is null or not a microsecond timestamp",
            ));
        }
    };
    let entity = compiled
        .partition_columns
        .iter()
        .map(|column| KeyValue::from_nullable_scalar(&values[column.index], node_id))
        .collect::<Result<Vec<_>>>()?;
    let sequence = compiled
        .sequence_columns
        .iter()
        .map(|column| KeyValue::from_required_scalar(&values[column.index], node_id))
        .collect::<Result<Vec<_>>>()?;
    Ok(BufferedRow::new(entity, sequence, event_time, values))
}

/// Sorts accepted rows into the canonical observable order and rejects
/// duplicate identities before any output is produced (SCE-00 D4).
fn sort_and_validate(mut rows: Vec<BufferedRow>, node_id: &str) -> Result<Vec<BufferedRow>> {
    rows.sort_by(|left, right| left.identity.cmp(&right.identity));
    if let Some(duplicate) = rows
        .windows(2)
        .find(|pair| pair[0].identity == pair[1].identity)
    {
        return Err(operator_error(
            node_id,
            &format!(
                "duplicate row identity at event_time_micros={}",
                duplicate[0].identity.event_time
            ),
        ));
    }
    Ok(rows)
}

/// Builds one output record: canonical-order input columns followed by the
/// derived rolling outputs (SCE-00 D5).
fn build_output_record(
    rows: &[BufferedRow],
    derived: Vec<ArrayRef>,
    output_schema: &SchemaRef,
    node_id: &str,
) -> Result<RecordBatch> {
    let input_width = rows.first().map_or_else(
        || output_schema.fields().len() - derived.len(),
        |row| row.values.len(),
    );
    let mut columns = Vec::with_capacity(input_width + derived.len());
    for index in 0..input_width {
        if rows.is_empty() {
            columns.push(new_null_array(output_schema.field(index).data_type(), 0));
            continue;
        }
        columns.push(
            ScalarValue::iter_to_array(rows.iter().map(|row| row.values[index].clone())).map_err(
                |error| {
                    operator_error(
                        node_id,
                        &format!("rolling output row encoding failed: {error}"),
                    )
                },
            )?,
        );
    }
    columns.extend(derived);
    RecordBatch::try_new(Arc::clone(output_schema), columns).map_err(|error| {
        operator_error(
            node_id,
            &format!("rolling output record is invalid: {error}"),
        )
    })
}

/// Builds a batch result directly from Arrow buffers when the immutable
/// kernel plan supports the semantic shape and the input proves canonical
/// order. `None` preserves the general sort-capable fallback.
fn build_typed_batch_output(
    table: &TableBatch,
    compiled: &CompiledRollingSpec,
    output_schema: &SchemaRef,
    node_id: &str,
) -> Result<Option<RecordBatch>> {
    let input = if let [record] = table.batches() {
        record.clone()
    } else {
        concat_batches(table.schema(), table.batches()).map_err(|error| {
            operator_error(
                node_id,
                &format!("typed rolling input concatenation failed: {error}"),
            )
        })?
    };
    let Some(execution) = compiled.kernel_plan.open_and_fill(&input, node_id)? else {
        return Ok(None);
    };
    debug_assert_eq!(execution.metrics.input_rows, input.num_rows());
    debug_assert_eq!(execution.metrics.output_rows, input.num_rows());
    let mut columns = input.columns().to_vec();
    columns.extend(execution.columns);
    RecordBatch::try_new(Arc::clone(output_schema), columns)
        .map(Some)
        .map_err(|error| {
            operator_error(
                node_id,
                &format!("typed rolling output record is invalid: {error}"),
            )
        })
}

type TypedStreamOutput = (RecordBatch, Option<RollingKernelState>, HistoryUpdates);

// Bootstrap, transition, output slicing, and history replacement form one
// failure-atomic stream update; none of them may escape independently.
// #lizard forgives
fn build_typed_stream_output(
    rows: &[BufferedRow],
    histories: &RollingHistories,
    state: Option<&RollingKernelState>,
    compiled: &CompiledRollingSpec,
    output_schema: &SchemaRef,
    node_id: &str,
    observer: Option<&RollingMetricsRecorder>,
) -> Result<Option<TypedStreamOutput>> {
    if !compiled.kernel_plan.supports_typed_transition() {
        return Ok(None);
    }
    let input_schema = Arc::new(Schema::new(
        output_schema.fields()[..output_schema.fields().len() - compiled.outputs.len()].to_vec(),
    ));
    let restored_state;
    let prior = if let Some(state) = state {
        state
    } else {
        let _stage = observer.map(|recorder| recorder.stage(RollingStage::StatePreparation));
        restored_state = reconstruct_typed_state(histories, compiled, &input_schema, node_id)?;
        &restored_state
    };
    let input = {
        let _stage = observer.map(|recorder| recorder.stage(RollingStage::ArrowOutput));
        build_input_record(rows, input_schema, node_id)?
    };
    let execution = compiled
        .kernel_plan
        .update_stream_and_fill(prior, &input, node_id, observer)?
        .ok_or_else(|| {
            internal_error("typed rolling stream rows did not satisfy canonical ordering")
        })?;
    let columns = execution.columns;
    let record = {
        let _stage = observer.map(|recorder| recorder.stage(RollingStage::ArrowOutput));
        build_output_record(rows, columns, output_schema, node_id)?
    };
    let touched = {
        let _stage = observer.map(|recorder| recorder.stage(RollingStage::HistoryMaintenance));
        typed_history_updates(rows, histories, compiled, node_id)?
    };
    Ok(Some((record, Some(execution.state), touched)))
}

fn reconstruct_typed_state(
    histories: &RollingHistories,
    compiled: &CompiledRollingSpec,
    input_schema: &SchemaRef,
    node_id: &str,
) -> Result<RollingKernelState> {
    let bootstrap = typed_bootstrap_rows(histories, compiled)?;
    let reconstructed = if bootstrap.is_empty() {
        RollingKernelState::default()
    } else {
        let input = build_input_record(&bootstrap, Arc::clone(input_schema), node_id)?;
        compiled
            .kernel_plan
            .update_and_fill(&RollingKernelState::default(), &input, node_id)?
            .ok_or_else(|| internal_error("typed rolling restore history is not canonical"))?
            .state
    };
    seed_typed_restored_state(histories, compiled, input_schema, &reconstructed, node_id)
}

fn seed_typed_restored_state(
    histories: &RollingHistories,
    compiled: &CompiledRollingSpec,
    input_schema: &SchemaRef,
    state: &RollingKernelState,
    node_id: &str,
) -> Result<RollingKernelState> {
    if histories.by_entity.is_empty() {
        return Ok(state.clone());
    }
    let values = histories
        .by_entity
        .keys()
        .map(|entity| ewma_entity_values(entity, input_schema, compiled))
        .collect::<Result<Vec<_>>>()?;
    let seeds = histories
        .by_entity
        .values()
        .map(|entity| {
            let mut seeds = typed_ewma_seeds(entity, compiled)?;
            seeds.resize(compiled.kernel_plan.typed_group_count(), None);
            Ok(seeds)
        })
        .collect::<Result<Vec<_>>>()?;
    let transition_counts = histories
        .by_entity
        .values()
        .map(|entity| entity.transition_count)
        .collect::<Vec<_>>();
    let nullable_schema = Arc::new(Schema::new(
        input_schema
            .fields()
            .iter()
            .map(|field| Field::new(field.name(), field.data_type().clone(), true))
            .collect::<Vec<_>>(),
    ));
    let entities = build_value_record(&values, nullable_schema, node_id)?;
    compiled
        .kernel_plan
        .seed_restored_state(state, &entities, &transition_counts, &seeds, node_id)
}

fn typed_ewma_seeds(
    entity: &EntityRollingState,
    compiled: &CompiledRollingSpec,
) -> Result<Vec<Option<(u64, f64)>>> {
    compiled
        .window_groups
        .iter()
        .enumerate()
        .map(|(group_index, group)| {
            if !matches!(group, CompiledWindowGroup::Ewma { .. }) {
                return Ok(None);
            }
            match entity.windows.get(group_index) {
                Some(WindowState::Ewma(state)) if state.valid_count > 0 => {
                    Ok(Some((state.valid_count, state.value)))
                }
                Some(WindowState::Ewma(_)) | None => Ok(None),
                _ => Err(internal_error("rolling EWMA checkpoint state mismatch")),
            }
        })
        .collect()
}

fn typed_bootstrap_rows(
    histories: &RollingHistories,
    compiled: &CompiledRollingSpec,
) -> Result<Vec<BufferedRow>> {
    let mut rows = histories
        .by_entity
        .values()
        .flat_map(|state| state.rows.iter())
        .map(|values| buffered_row_from_values(values.clone(), compiled))
        .collect::<Result<Vec<_>>>()?;
    rows.sort_by(|left, right| left.identity.cmp(&right.identity));
    Ok(rows)
}

fn build_input_record(
    rows: &[BufferedRow],
    schema: SchemaRef,
    node_id: &str,
) -> Result<RecordBatch> {
    let arrays = (0..schema.fields().len())
        .map(|index| {
            ScalarValue::iter_to_array(rows.iter().map(|row| row.values[index].clone())).map_err(
                |error| {
                    operator_error(
                        node_id,
                        &format!("typed rolling stream input encoding failed: {error}"),
                    )
                },
            )
        })
        .collect::<Result<Vec<_>>>()?;
    RecordBatch::try_new(schema, arrays).map_err(|error| {
        operator_error(
            node_id,
            &format!("typed rolling stream input batch is invalid: {error}"),
        )
    })
}

fn build_value_record(
    rows: &[Vec<ScalarValue>],
    schema: SchemaRef,
    node_id: &str,
) -> Result<RecordBatch> {
    let arrays = (0..schema.fields().len())
        .map(|index| {
            ScalarValue::iter_to_array(rows.iter().map(|row| row[index].clone())).map_err(|error| {
                operator_error(
                    node_id,
                    &format!("typed rolling restore entity encoding failed: {error}"),
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;
    RecordBatch::try_new(schema, arrays).map_err(|error| {
        operator_error(
            node_id,
            &format!("typed rolling restore entity batch is invalid: {error}"),
        )
    })
}

fn typed_history_updates(
    rows: &[BufferedRow],
    histories: &RollingHistories,
    compiled: &CompiledRollingSpec,
    node_id: &str,
) -> Result<HistoryUpdates> {
    let has_ewma = compiled
        .window_groups
        .iter()
        .any(|group| matches!(group, CompiledWindowGroup::Ewma { .. }));
    group_rows_by_entity(rows)
        .into_iter()
        .map(|(entity, indices)| {
            let mut state = histories.by_entity.get(entity).cloned().unwrap_or_default();
            let transitions = u64::try_from(indices.len()).map_err(|_| {
                operator_error(node_id, "rolling micro-batch row count does not fit u64")
            })?;
            if has_ewma {
                advance_typed_ewma_windows(&mut state, rows, &indices, compiled, node_id)?;
            } else {
                state.windows.clear();
            }
            state
                .rows
                .extend(indices.into_iter().map(|index| rows[index].values.clone()));
            state.transition_count =
                state
                    .transition_count
                    .checked_add(transitions)
                    .ok_or_else(|| {
                        operator_error(node_id, "rolling entity transition count overflowed")
                    })?;
            evict_retained_history(&mut state, compiled);
            Ok((entity.clone(), state))
        })
        .collect()
}

fn advance_typed_ewma_windows(
    state: &mut EntityRollingState,
    rows: &[BufferedRow],
    indices: &[usize],
    compiled: &CompiledRollingSpec,
    node_id: &str,
) -> Result<()> {
    if state.windows.len() != compiled.window_groups.len() {
        state.windows = fresh_windows(compiled);
    }
    for &row_index in indices {
        for (group_index, group) in compiled.window_groups.iter().enumerate() {
            let CompiledWindowGroup::Ewma {
                input_index, alpha, ..
            } = group
            else {
                continue;
            };
            let WindowState::Ewma(accumulator) = &mut state.windows[group_index] else {
                return Err(internal_error("rolling typed EWMA history state mismatch"));
            };
            let sample = &rows[row_index].values[*input_index];
            if is_valid_sample(sample) {
                accumulator.add(sample, *alpha, node_id)?;
            }
        }
    }
    Ok(())
}

#[derive(Clone)]
struct CompiledRollingSpec {
    state_layout_version: u32,
    legacy_state_layout_version: u32,
    event_time_index: usize,
    partition_columns: Vec<CompiledKeyColumn>,
    sequence_columns: Vec<CompiledKeyColumn>,
    outputs: Vec<CompiledRollingOutput>,
    window_groups: Vec<CompiledWindowGroup>,
    kernel_plan: RollingKernelPlan,
    max_row_retention: u64,
    max_duration_micros: Option<u64>,
    configuration_hash: String,
    state_schema_fingerprint: String,
    legacy_state_schema_fingerprint: String,
}

#[derive(Clone)]
struct CompiledKeyColumn {
    index: usize,
}

#[derive(Clone)]
struct CompiledRollingOutput {
    input_index: usize,
    name: String,
    input_type: DataType,
    output_type: DataType,
    evaluation: CompiledEvaluation,
}

#[derive(Clone)]
enum CompiledEvaluation {
    Lag { periods: u64 },
    Delta { periods: u64 },
    Ewma(CompiledEwma),
    Aggregate(CompiledAggregate),
    Pair(CompiledPairAggregate),
    Difference(CompiledDifference),
    Scan(CompiledScan),
}

#[derive(Clone, Copy)]
struct CompiledScan {
    kind: ScanKind,
    frame: CompiledFrame,
    min_periods: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ScanKind {
    Argmax,
    Argmin,
    Rank,
    Quantile,
    UniqueCount,
    Decay,
}

#[derive(Clone)]
struct CompiledDifference {
    left: CompiledFloatReadout,
    right: CompiledFloatReadout,
}

#[derive(Clone, Copy, Debug)]
enum CompiledFloatReadout {
    Aggregate(CompiledAggregate),
    Ewma(CompiledEwma),
}

#[derive(Clone, Copy, Debug)]
struct CompiledEwma {
    group: usize,
    min_periods: u64,
}

#[derive(Clone, Copy, Debug)]
struct CompiledAggregate {
    group: usize,
    statistic: Statistic,
    min_periods: u64,
    ddof: u8,
}

#[derive(Clone)]
struct CompiledPairAggregate {
    group: usize,
    correlation: bool,
    min_periods: u64,
    ddof: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Statistic {
    Count,
    Sum,
    Mean,
    Variance,
    Stddev,
    Min,
    Max,
}

impl Statistic {
    const fn name(self) -> &'static str {
        match self {
            Self::Count => "count",
            Self::Sum => "sum",
            Self::Mean => "mean",
            Self::Variance => "variance",
            Self::Stddev => "stddev",
            Self::Min => "min",
            Self::Max => "max",
        }
    }
}

/// One shared per-entity sliding window (SCE-07 state sharing): every output
/// on the same accumulator key reads one group instead of maintaining
/// duplicate windows.
#[derive(Clone)]
enum CompiledWindowGroup {
    /// Reversible numeric accumulator shared by count/sum/mean/variance/
    /// stddev outputs on one `(input column, frame)`.
    Numeric {
        input_index: usize,
        frame: CompiledFrame,
        sum_class: SumClass,
    },
    /// Monotonic queue for one min or max output on `(input column, frame)`
    /// (SCE-08); min and max keep separate queues.
    Extrema {
        input_index: usize,
        frame: CompiledFrame,
        descending: bool,
    },
    /// Reversible co-moment accumulator shared by covariance and correlation
    /// outputs on one `(left column, right column, frame)`.
    Pair {
        left_index: usize,
        right_index: usize,
        frame: CompiledFrame,
    },
    /// Constant-state recursive average shared by outputs on one
    /// `(input column, span)` key.
    Ewma {
        input_index: usize,
        span: u64,
        alpha: f64,
    },
}

/// Compiled frame declaration: row count or event-time duration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CompiledFrame {
    Rows(u64),
    Duration(u64),
}

impl CompiledFrame {
    fn rows(self) -> u64 {
        match self {
            Self::Rows(rows) => rows,
            Self::Duration(..) => usize::MAX as u64,
        }
    }
}

/// Integer sums stay exact in their frozen 64-bit class; floating sums and
/// every mean/variance accumulate in `f64` (SCE-00 D3, contract section 5.2).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SumClass {
    Signed,
    Unsigned,
    Float,
    CountOnly,
}

impl SumClass {
    fn from_input(data_type: &DataType) -> Self {
        match data_type {
            DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => Self::Signed,
            DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
                Self::Unsigned
            }
            DataType::Float32 | DataType::Float64 => Self::Float,
            _ => Self::CountOnly,
        }
    }
}

/// One entity or sequence key component in the Arrow total order (null
/// before non-null); floats compare with the IEEE total order (SCE-00 D4).
#[derive(Clone, Debug)]
enum KeyValue {
    Boolean(bool),
    Signed(i64),
    Unsigned(u64),
    Float32(f32),
    Float64(f64),
    String(String),
    Date32(i32),
    Date64(i64),
    Timestamp(i64),
}

impl KeyValue {
    fn from_nullable_scalar(scalar: &ScalarValue, node_id: &str) -> Result<Option<Self>> {
        if scalar.is_null() {
            return Ok(None);
        }
        Self::from_required_scalar(scalar, node_id).map(Some)
    }

    fn from_required_scalar(scalar: &ScalarValue, node_id: &str) -> Result<Self> {
        let value = match scalar {
            ScalarValue::Boolean(value) => value.map(Self::Boolean),
            ScalarValue::Int8(value) => value.map(|value| Self::Signed(i64::from(value))),
            ScalarValue::Int16(value) => value.map(|value| Self::Signed(i64::from(value))),
            ScalarValue::Int32(value) => value.map(|value| Self::Signed(i64::from(value))),
            ScalarValue::Int64(value) => value.map(Self::Signed),
            ScalarValue::UInt8(value) => value.map(|value| Self::Unsigned(u64::from(value))),
            ScalarValue::UInt16(value) => value.map(|value| Self::Unsigned(u64::from(value))),
            ScalarValue::UInt32(value) => value.map(|value| Self::Unsigned(u64::from(value))),
            ScalarValue::UInt64(value) => value.map(Self::Unsigned),
            ScalarValue::Float32(value) => value.map(Self::Float32),
            ScalarValue::Float64(value) => value.map(Self::Float64),
            ScalarValue::Utf8(value) | ScalarValue::LargeUtf8(value) => {
                value.clone().map(Self::String)
            }
            ScalarValue::Date32(value) => value.map(Self::Date32),
            ScalarValue::Date64(value) => value.map(Self::Date64),
            ScalarValue::TimestampMicrosecond(value, _) => value.map(Self::Timestamp),
            other => {
                return Err(operator_error(
                    node_id,
                    &format!(
                        "rolling key column has unsupported value type {}",
                        other.data_type()
                    ),
                ));
            }
        };
        value.ok_or_else(|| operator_error(node_id, "rolling sequence key value is null"))
    }
}

impl PartialEq for KeyValue {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for KeyValue {}

impl PartialOrd for KeyValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for KeyValue {
    fn cmp(&self, other: &Self) -> Ordering {
        fn rank(value: &KeyValue) -> u8 {
            match value {
                KeyValue::Boolean(_) => 0,
                KeyValue::Signed(_) => 1,
                KeyValue::Unsigned(_) => 2,
                KeyValue::Float32(_) => 3,
                KeyValue::Float64(_) => 4,
                KeyValue::String(_) => 5,
                KeyValue::Date32(_) => 6,
                KeyValue::Date64(_) => 7,
                KeyValue::Timestamp(_) => 8,
            }
        }
        match (self, other) {
            (Self::Boolean(left), Self::Boolean(right)) => left.cmp(right),
            (Self::Unsigned(left), Self::Unsigned(right)) => left.cmp(right),
            (Self::Float32(left), Self::Float32(right)) => left.total_cmp(right),
            (Self::Float64(left), Self::Float64(right)) => left.total_cmp(right),
            (Self::String(left), Self::String(right)) => left.cmp(right),
            (Self::Date32(left), Self::Date32(right)) => i64::from(*left).cmp(&i64::from(*right)),
            (Self::Signed(left), Self::Signed(right))
            | (Self::Date64(left), Self::Date64(right))
            | (Self::Timestamp(left), Self::Timestamp(right)) => left.cmp(right),
            _ => rank(self).cmp(&rank(other)),
        }
    }
}

fn key_scalar(value: &KeyValue, data_type: &DataType) -> Result<ScalarValue> {
    let mismatch = || {
        state_format(format!(
            "rolling EWMA entity key is incompatible with state type {data_type}"
        ))
    };
    match (value, data_type) {
        (KeyValue::Boolean(value), DataType::Boolean) => Ok(ScalarValue::Boolean(Some(*value))),
        (KeyValue::Signed(value), DataType::Int8) => i8::try_from(*value)
            .map(|value| ScalarValue::Int8(Some(value)))
            .map_err(|_| mismatch()),
        (KeyValue::Signed(value), DataType::Int16) => i16::try_from(*value)
            .map(|value| ScalarValue::Int16(Some(value)))
            .map_err(|_| mismatch()),
        (KeyValue::Signed(value), DataType::Int32) => i32::try_from(*value)
            .map(|value| ScalarValue::Int32(Some(value)))
            .map_err(|_| mismatch()),
        (KeyValue::Signed(value), DataType::Int64) => Ok(ScalarValue::Int64(Some(*value))),
        (KeyValue::Unsigned(value), DataType::UInt8) => u8::try_from(*value)
            .map(|value| ScalarValue::UInt8(Some(value)))
            .map_err(|_| mismatch()),
        (KeyValue::Unsigned(value), DataType::UInt16) => u16::try_from(*value)
            .map(|value| ScalarValue::UInt16(Some(value)))
            .map_err(|_| mismatch()),
        (KeyValue::Unsigned(value), DataType::UInt32) => u32::try_from(*value)
            .map(|value| ScalarValue::UInt32(Some(value)))
            .map_err(|_| mismatch()),
        (KeyValue::Unsigned(value), DataType::UInt64) => Ok(ScalarValue::UInt64(Some(*value))),
        (KeyValue::Float32(value), DataType::Float32) => Ok(ScalarValue::Float32(Some(*value))),
        (KeyValue::Float64(value), DataType::Float64) => Ok(ScalarValue::Float64(Some(*value))),
        (KeyValue::String(value), DataType::Utf8) => Ok(ScalarValue::Utf8(Some(value.clone()))),
        (KeyValue::String(value), DataType::LargeUtf8) => {
            Ok(ScalarValue::LargeUtf8(Some(value.clone())))
        }
        (KeyValue::Date32(value), DataType::Date32) => Ok(ScalarValue::Date32(Some(*value))),
        (KeyValue::Date64(value), DataType::Date64) => Ok(ScalarValue::Date64(Some(*value))),
        (KeyValue::Timestamp(value), DataType::Timestamp(TimeUnit::Microsecond, timezone)) => Ok(
            ScalarValue::TimestampMicrosecond(Some(*value), timezone.clone()),
        ),
        _ => Err(mismatch()),
    }
}

fn ewma_entity_values(
    entity: &[Option<KeyValue>],
    input_schema: &Schema,
    compiled: &CompiledRollingSpec,
) -> Result<Vec<ScalarValue>> {
    if entity.len() != compiled.partition_columns.len() {
        return Err(internal_error("rolling EWMA entity key width mismatch"));
    }
    let mut values = input_schema
        .fields()
        .iter()
        .map(|field| typed_null(field.data_type()))
        .collect::<Vec<_>>();
    for (value, column) in entity.iter().zip(&compiled.partition_columns) {
        values[column.index] = match value {
            Some(value) => key_scalar(value, input_schema.field(column.index).data_type())?,
            None => typed_null(input_schema.field(column.index).data_type()),
        };
    }
    Ok(values)
}

/// Canonical row identity `(event_time, entity_key..., sequence_key...)`
/// (SCE-00 D4).
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct RowIdentity {
    event_time: i64,
    entity: Vec<Option<KeyValue>>,
    sequence: Vec<KeyValue>,
}

/// One accepted input row retained for final-order emission.
#[derive(Clone, Debug)]
struct BufferedRow {
    identity: RowIdentity,
    values: Vec<ScalarValue>,
}

impl BufferedRow {
    fn new(
        entity: Vec<Option<KeyValue>>,
        sequence: Vec<KeyValue>,
        event_time: i64,
        values: Vec<ScalarValue>,
    ) -> Self {
        Self {
            identity: RowIdentity {
                event_time,
                entity,
                sequence,
            },
            values,
        }
    }
}

/// Per-entity retained tail plus the shared sliding-window accumulators.
#[derive(Clone, Debug, Default)]
struct EntityRollingState {
    rows: VecDeque<Vec<ScalarValue>>,
    columnar: ordered_stream::ColumnarHistory,
    windows: Vec<WindowState>,
    transition_count: u64,
}

impl EntityRollingState {
    fn fresh(compiled: &CompiledRollingSpec) -> Self {
        Self {
            rows: VecDeque::new(),
            columnar: ordered_stream::ColumnarHistory::default(),
            windows: fresh_windows(compiled),
            transition_count: 0,
        }
    }
}

/// Live accumulator of one compiled window group.
#[derive(Clone, Debug)]
enum WindowState {
    Numeric(WindowAccumulator),
    Extrema(ExtremaAccumulator),
    Pair(PairAccumulator),
    Ewma(EwmaAccumulator),
}

fn fresh_windows(compiled: &CompiledRollingSpec) -> Vec<WindowState> {
    compiled
        .window_groups
        .iter()
        .map(|group| match group {
            CompiledWindowGroup::Numeric { sum_class, .. } => {
                WindowState::Numeric(WindowAccumulator::new(*sum_class))
            }
            CompiledWindowGroup::Extrema { descending, .. } => {
                WindowState::Extrema(ExtremaAccumulator::new(*descending))
            }
            CompiledWindowGroup::Pair { .. } => WindowState::Pair(PairAccumulator::default()),
            CompiledWindowGroup::Ewma { .. } => WindowState::Ewma(EwmaAccumulator::default()),
        })
        .collect()
}

/// Per-entity rolling state: retained tails of the last `max_retained_rows`
/// rows plus one accumulator set per compiled window group.
#[derive(Clone, Debug, Default)]
struct RollingHistories {
    by_entity: BTreeMap<Vec<Option<KeyValue>>, EntityRollingState>,
}

/// Kernel-produced per-entity state replacements (entity key, new state).
type HistoryUpdates = Vec<(Vec<Option<KeyValue>>, EntityRollingState)>;

impl RollingHistories {
    fn apply(&mut self, touched: HistoryUpdates) {
        for (entity, state) in touched {
            self.by_entity.insert(entity, state);
        }
    }
}

/// Reversible sliding-window accumulator (SCE-00 D5): exact checked integer
/// sums, `f64` sums, and West-style add/remove mean and M2 variance state.
/// The ordered add/remove sequence is the one frozen algorithm shared by the
/// batch and stream lifecycles. ±inf sample counts make the mean and
/// variance classifications pure multiset functions of the window (SCE-07
/// defect 1 ruling): a window's IEEE classification must not depend on where
/// the infinity sat in arrival order.
#[derive(Clone, Copy, Debug)]
struct WindowAccumulator {
    valid_count: u64,
    sum: Option<SumState>,
    mean: f64,
    m2: f64,
    /// Retained IEEE NaN samples. The normal rolling surface filters NaNs,
    /// while the `DataFusion` compatibility plan counts and propagates them.
    nan_count: u64,
    pos_inf: u64,
    neg_inf: u64,
    /// Duration frames only: event-time bound (exclusive) whose rows this
    /// accumulator has already expired, kept so each slide removes only the
    /// newly expired prefix (SCE-08). `i128::MIN` while nothing expired.
    expired_through: i128,
}

/// Constant-state unadjusted exponential average. A zero valid count is the
/// only unseeded representation; once seeded, the exact IEEE value is durable.
#[derive(Clone, Copy, Debug, Default)]
struct EwmaAccumulator {
    valid_count: u64,
    value: f64,
}

impl EwmaAccumulator {
    #[allow(
        clippy::cast_precision_loss,
        reason = "cumulative mean is a Float64 readout"
    )]
    fn add(&mut self, sample: &ScalarValue, alpha: f64, node_id: &str) -> Result<()> {
        let value = float_sample(sample);
        let next_count = self
            .valid_count
            .checked_add(1)
            .ok_or_else(|| operator_error(node_id, "rolling sample count overflowed"))?;
        self.value = if self.valid_count == 0 || alpha.to_bits() == 1.0_f64.to_bits() {
            value
        } else if alpha == 0.0 {
            if self.value.is_infinite() && value.is_finite() {
                self.value
            } else if value.is_infinite() {
                value + self.value
            } else {
                let difference = value - self.value;
                if difference.is_finite() {
                    self.value + difference / next_count as f64
                } else {
                    let weight = 1.0 / next_count as f64;
                    self.value * (1.0 - weight) + value * weight
                }
            }
        } else {
            self.value + alpha * (value - self.value)
        };
        self.valid_count = next_count;
        Ok(())
    }
}

/// Integer sums accumulate in the wide transient class so the
/// add-before-remove slide never reports a false overflow for a window whose
/// true sum is representable; the readout converts back with a checked
/// narrowing that keeps genuine overflow loud (SCE-07 defect 2 fix).
#[derive(Clone, Copy, Debug)]
enum SumState {
    Signed(i128),
    Unsigned(u128),
    Float(f64),
}

impl WindowAccumulator {
    fn new(sum_class: SumClass) -> Self {
        let sum = match sum_class {
            SumClass::Signed => Some(SumState::Signed(0)),
            SumClass::Unsigned => Some(SumState::Unsigned(0)),
            SumClass::Float => Some(SumState::Float(0.0)),
            SumClass::CountOnly => None,
        };
        Self {
            valid_count: 0,
            sum,
            mean: 0.0,
            m2: 0.0,
            nan_count: 0,
            pos_inf: 0,
            neg_inf: 0,
            expired_through: i128::MIN,
        }
    }

    /// Adds one valid sample. Null and NaN values never reach this method.
    #[allow(
        clippy::cast_precision_loss,
        reason = "the frozen mean/variance output type is Float64"
    )]
    fn add(&mut self, value: &ScalarValue, node_id: &str) -> Result<()> {
        self.valid_count = self
            .valid_count
            .checked_add(1)
            .ok_or_else(|| operator_error(node_id, "rolling valid sample count overflowed"))?;
        if let Some(sum) = &mut self.sum {
            match sum {
                SumState::Signed(total) => {
                    *total = total
                        .checked_add(i128::from(signed_sample(value)))
                        .ok_or_else(|| operator_error(node_id, "rolling integer sum overflowed"))?;
                }
                SumState::Unsigned(total) => {
                    *total = total
                        .checked_add(u128::from(unsigned_sample(value)))
                        .ok_or_else(|| operator_error(node_id, "rolling integer sum overflowed"))?;
                }
                SumState::Float(total) => *total += float_sample(value),
            }
            let sample = float_sample(value);
            if sample.is_infinite() {
                if sample > 0.0 {
                    self.pos_inf = self.pos_inf.saturating_add(1);
                } else {
                    self.neg_inf = self.neg_inf.saturating_add(1);
                }
            }
            let count = self.valid_count as f64;
            let delta = sample - self.mean;
            self.mean += delta / count;
            self.m2 += delta * (sample - self.mean);
        }
        Ok(())
    }

    /// Removes one previously added valid sample (West 1979 removal step).
    #[allow(
        clippy::cast_precision_loss,
        reason = "the frozen mean/variance output type is Float64"
    )]
    fn remove(&mut self, value: &ScalarValue) -> Result<()> {
        self.valid_count = self
            .valid_count
            .checked_sub(1)
            .ok_or_else(|| internal_error("rolling removal without a matching add"))?;
        if let Some(sum) = &mut self.sum {
            match sum {
                SumState::Signed(total) => {
                    *total = total
                        .checked_sub(i128::from(signed_sample(value)))
                        .ok_or_else(|| {
                            internal_error("rolling sum removal diverged from the adds")
                        })?;
                }
                SumState::Unsigned(total) => {
                    *total = total
                        .checked_sub(u128::from(unsigned_sample(value)))
                        .ok_or_else(|| {
                            internal_error("rolling sum removal diverged from the adds")
                        })?;
                }
                SumState::Float(total) => *total -= float_sample(value),
            }
            let sample = float_sample(value);
            if sample.is_infinite() {
                if sample > 0.0 {
                    self.pos_inf = self.pos_inf.saturating_sub(1);
                } else {
                    self.neg_inf = self.neg_inf.saturating_sub(1);
                }
            }
            if self.valid_count == 0 {
                self.mean = 0.0;
                self.m2 = 0.0;
            } else {
                let count = self.valid_count as f64;
                let delta = sample - self.mean;
                self.mean -= delta / count;
                self.m2 -= delta * (sample - self.mean);
            }
        }
        Ok(())
    }

    /// True when sliding arithmetic produced a non-finite component; the
    /// caller then re-folds the current window so the live state always
    /// matches the checkpoint rebuild for non-finite classifications.
    fn is_non_finite(&self) -> bool {
        let sum_non_finite = match &self.sum {
            Some(SumState::Float(total)) => !total.is_finite(),
            _ => false,
        };
        sum_non_finite || !self.mean.is_finite() || !self.m2.is_finite()
    }

    const fn has_float_sum(&self) -> bool {
        matches!(self.sum, Some(SumState::Float(_)))
    }

    fn reset(&mut self) {
        *self = Self::new(match &self.sum {
            Some(SumState::Signed(_)) => SumClass::Signed,
            Some(SumState::Unsigned(_)) => SumClass::Unsigned,
            Some(SumState::Float(_)) => SumClass::Float,
            None => SumClass::CountOnly,
        });
    }
}

fn spec_uses_stable_v2(compiled: &CompiledRollingSpec) -> bool {
    compiled.kernel_plan.numerical_profile_kind() == RollingNumericalProfile::StableV2Preview
}

/// Canonical expiry key of one queued extrema candidate: the entity-local
/// total order `(event_time, sequence...)` (SCE-00 D5 equal-time rule).
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ExtremaKey {
    event_time: i64,
    sequence: Vec<KeyValue>,
}

/// Monotonic-queue min/max accumulator (SCE-08): the queue keeps valid
/// candidates in canonical key order with values monotone from the front, so
/// the front is always the window extremum and each row is pushed and popped
/// at most once. Entries carry their canonical key so row-count frames expire
/// by identity comparison with the leaving row and duration frames expire by
/// event time; the queue is therefore coordinate-free across batches and
/// checkpoint restores.
#[derive(Clone, Debug)]
struct ExtremaAccumulator {
    valid_count: u64,
    descending: bool,
    queue: VecDeque<(ExtremaKey, ScalarValue)>,
    /// Duration frames only: exclusive event-time bound whose rows the count
    /// has already regressed; the queue expires by its own keys.
    expired_through: i128,
}

impl ExtremaAccumulator {
    fn new(descending: bool) -> Self {
        Self {
            valid_count: 0,
            descending,
            queue: VecDeque::new(),
            expired_through: i128::MIN,
        }
    }

    /// Adds one valid sample with its canonical key, dropping dominated
    /// candidates from the back.
    fn add(&mut self, key: ExtremaKey, value: ScalarValue) {
        self.valid_count = self.valid_count.saturating_add(1);
        while let Some((_, back)) = self.queue.back() {
            let dominated = if self.descending {
                compare_samples(back, &value) != Ordering::Greater
            } else {
                compare_samples(back, &value) != Ordering::Less
            };
            if !dominated {
                break;
            }
            self.queue.pop_back();
        }
        self.queue.push_back((key, value));
    }

    /// Removes one previously added valid sample from the count; dominated
    /// candidates were already dropped, so only the count regresses.
    fn remove(&mut self) {
        self.valid_count = self
            .valid_count
            .checked_sub(1)
            .unwrap_or_else(|| panic!("rolling extrema removal without a matching add"));
    }

    /// Row-count frame expiry: every candidate at or before the leaving
    /// row's canonical key has left the window.
    fn expire_through_key(&mut self, leaving: &ExtremaKey) {
        while self.queue.front().is_some_and(|(key, _)| key <= leaving) {
            self.queue.pop_front();
        }
    }

    /// Duration frame expiry: candidates at or before `bound` (exclusive)
    /// have left the window.
    fn expire_through_time(&mut self, bound: i128) {
        while self
            .queue
            .front()
            .is_some_and(|(key, _)| i128::from(key.event_time) <= bound)
        {
            self.queue.pop_front();
        }
    }

    fn extremum(&self) -> Option<&ScalarValue> {
        self.queue.front().map(|(_, value)| value)
    }
}

/// Reversible pairwise co-moment accumulator (SCE-08, SCE-00 D5): a
/// West-style joint add/remove maintains `mean_x`, `mean_y`, the co-moment
/// `Σ(x - x̄)(y - ȳ)`, and the per-column second moments `M2_x`/`M2_y` over
/// the pairwise-valid positions only. ±inf counts per column keep the
/// classification a pure multiset function, mirroring the SCE-07 variance
/// ruling: any infinity on either side makes covariance and correlation NaN
/// (undefined ∞ − ∞ territory), never null.
#[derive(Clone, Copy, Debug)]
struct PairAccumulator {
    valid_count: u64,
    mean_x: f64,
    mean_y: f64,
    co_moment: f64,
    m2_x: f64,
    m2_y: f64,
    pos_inf_x: u64,
    neg_inf_x: u64,
    pos_inf_y: u64,
    neg_inf_y: u64,
    /// Duration frames only: exclusive event-time bound already expired.
    expired_through: i128,
}

impl Default for PairAccumulator {
    fn default() -> Self {
        Self {
            valid_count: 0,
            mean_x: 0.0,
            mean_y: 0.0,
            co_moment: 0.0,
            m2_x: 0.0,
            m2_y: 0.0,
            pos_inf_x: 0,
            neg_inf_x: 0,
            pos_inf_y: 0,
            neg_inf_y: 0,
            expired_through: i128::MIN,
        }
    }
}

impl PairAccumulator {
    /// Adds one pairwise-valid sample. Null/NaN operands never reach this
    /// method.
    #[allow(
        clippy::cast_precision_loss,
        reason = "the frozen covariance/correlation output type is Float64"
    )]
    fn add(&mut self, x: &ScalarValue, y: &ScalarValue, node_id: &str) -> Result<()> {
        self.valid_count = self
            .valid_count
            .checked_add(1)
            .ok_or_else(|| operator_error(node_id, "rolling pair sample count overflowed"))?;
        let sample_x = float_sample(x);
        let sample_y = float_sample(y);
        if sample_x.is_infinite() {
            if sample_x > 0.0 {
                self.pos_inf_x = self.pos_inf_x.saturating_add(1);
            } else {
                self.neg_inf_x = self.neg_inf_x.saturating_add(1);
            }
        }
        if sample_y.is_infinite() {
            if sample_y > 0.0 {
                self.pos_inf_y = self.pos_inf_y.saturating_add(1);
            } else {
                self.neg_inf_y = self.neg_inf_y.saturating_add(1);
            }
        }
        let count = self.valid_count as f64;
        let delta_x = sample_x - self.mean_x;
        self.mean_x += delta_x / count;
        let delta_y = sample_y - self.mean_y;
        self.mean_y += delta_y / count;
        self.co_moment += delta_x * (sample_y - self.mean_y);
        self.m2_x += delta_x * (sample_x - self.mean_x);
        self.m2_y += delta_y * (sample_y - self.mean_y);
        Ok(())
    }

    /// Removes one previously added pairwise-valid sample (reverse step).
    #[allow(
        clippy::cast_precision_loss,
        reason = "the frozen covariance/correlation output type is Float64"
    )]
    fn remove(&mut self, x: &ScalarValue, y: &ScalarValue) -> Result<()> {
        self.valid_count = self
            .valid_count
            .checked_sub(1)
            .ok_or_else(|| internal_error("rolling pair removal without a matching add"))?;
        let sample_x = float_sample(x);
        let sample_y = float_sample(y);
        if sample_x.is_infinite() {
            if sample_x > 0.0 {
                self.pos_inf_x = self.pos_inf_x.saturating_sub(1);
            } else {
                self.neg_inf_x = self.neg_inf_x.saturating_sub(1);
            }
        }
        if sample_y.is_infinite() {
            if sample_y > 0.0 {
                self.pos_inf_y = self.pos_inf_y.saturating_sub(1);
            } else {
                self.neg_inf_y = self.neg_inf_y.saturating_sub(1);
            }
        }
        if self.valid_count == 0 {
            self.mean_x = 0.0;
            self.mean_y = 0.0;
            self.co_moment = 0.0;
            self.m2_x = 0.0;
            self.m2_y = 0.0;
        } else {
            let count = self.valid_count as f64;
            let delta_x = sample_x - self.mean_x;
            self.mean_x -= delta_x / count;
            let delta_y = sample_y - self.mean_y;
            self.mean_y -= delta_y / count;
            self.co_moment -= delta_x * (sample_y - self.mean_y);
            self.m2_x -= delta_x * (sample_x - self.mean_x);
            self.m2_y -= delta_y * (sample_y - self.mean_y);
        }
        Ok(())
    }

    fn is_non_finite(&self) -> bool {
        !self.mean_x.is_finite()
            || !self.mean_y.is_finite()
            || !self.co_moment.is_finite()
            || !self.m2_x.is_finite()
            || !self.m2_y.is_finite()
    }

    fn holds_infinity(&self) -> bool {
        self.pos_inf_x > 0 || self.neg_inf_x > 0 || self.pos_inf_y > 0 || self.neg_inf_y > 0
    }

    fn reset(&mut self) {
        *self = Self {
            expired_through: self.expired_through,
            ..Self::default()
        };
    }
}

/// A rolling sample is valid when it is neither null nor NaN (SCE-00 D3,
/// contract section 5.2); infinities stay numeric.
fn is_valid_sample(value: &ScalarValue) -> bool {
    if value.is_null() {
        return false;
    }
    !matches!(value, ScalarValue::Float32(Some(sample)) if sample.is_nan())
        && !matches!(value, ScalarValue::Float64(Some(sample)) if sample.is_nan())
}

fn signed_sample(value: &ScalarValue) -> i64 {
    match value {
        ScalarValue::Int8(Some(sample)) => i64::from(*sample),
        ScalarValue::Int16(Some(sample)) => i64::from(*sample),
        ScalarValue::Int32(Some(sample)) => i64::from(*sample),
        ScalarValue::Int64(Some(sample)) => *sample,
        other => unreachable!("signed rolling sample has type {}", other.data_type()),
    }
}

fn unsigned_sample(value: &ScalarValue) -> u64 {
    match value {
        ScalarValue::UInt8(Some(sample)) => u64::from(*sample),
        ScalarValue::UInt16(Some(sample)) => u64::from(*sample),
        ScalarValue::UInt32(Some(sample)) => u64::from(*sample),
        ScalarValue::UInt64(Some(sample)) => *sample,
        other => unreachable!("unsigned rolling sample has type {}", other.data_type()),
    }
}

#[allow(
    clippy::cast_precision_loss,
    reason = "the frozen mean/variance output type is Float64"
)]
fn float_sample(value: &ScalarValue) -> f64 {
    match value {
        ScalarValue::Float32(Some(sample)) => f64::from(*sample),
        ScalarValue::Float64(Some(sample)) => *sample,
        ScalarValue::Int8(_)
        | ScalarValue::Int16(_)
        | ScalarValue::Int32(_)
        | ScalarValue::Int64(_) => signed_sample(value) as f64,
        ScalarValue::UInt8(_)
        | ScalarValue::UInt16(_)
        | ScalarValue::UInt32(_)
        | ScalarValue::UInt64(_) => unsigned_sample(value) as f64,
        other => unreachable!("floating rolling sample has type {}", other.data_type()),
    }
}

/// Total order over comparable rolling samples of one column type (SCE-08
/// min/max): floats use the IEEE total order so −0.0/0.0 and ±inf compare
/// deterministically; NaN never reaches the queue.
fn compare_samples(left: &ScalarValue, right: &ScalarValue) -> Ordering {
    fn rank(value: &ScalarValue) -> u8 {
        match value {
            ScalarValue::Boolean(_) => 0,
            ScalarValue::Int8(_) | ScalarValue::Int16(_) | ScalarValue::Int32(_) => 1,
            ScalarValue::Int64(_) => 2,
            ScalarValue::UInt8(_) | ScalarValue::UInt16(_) | ScalarValue::UInt32(_) => 3,
            ScalarValue::UInt64(_) => 4,
            ScalarValue::Float32(_) => 5,
            ScalarValue::Float64(_) => 6,
            ScalarValue::Utf8(_) | ScalarValue::LargeUtf8(_) => 7,
            ScalarValue::Date32(_) => 8,
            ScalarValue::Date64(_) => 9,
            ScalarValue::TimestampMicrosecond(_, _) => 10,
            _ => 11,
        }
    }
    let (left, right) = match (left, right) {
        (ScalarValue::Float32(Some(left)), ScalarValue::Float32(Some(right))) => {
            // Samples in the extrema queue are always valid; NaN never
            // reaches this comparison and ±inf orders by sign.
            return left.total_cmp(right);
        }
        (ScalarValue::Float64(Some(left)), ScalarValue::Float64(Some(right))) => {
            return left.total_cmp(right);
        }
        _ => (left, right),
    };
    match (left, right) {
        (ScalarValue::Boolean(left), ScalarValue::Boolean(right)) => left.cmp(right),
        (
            ScalarValue::Utf8(left) | ScalarValue::LargeUtf8(left),
            ScalarValue::Utf8(right) | ScalarValue::LargeUtf8(right),
        ) => left.cmp(right),
        _ => match (signed_order_key(left), signed_order_key(right)) {
            (Some(left), Some(right)) => left.cmp(&right),
            _ => rank(left).cmp(&rank(right)),
        },
    }
}

/// Wide signed view of the integer-like sample classes for total-order
/// comparison; `None` for values compared by their own arm above.
fn signed_order_key(value: &ScalarValue) -> Option<i128> {
    match value {
        ScalarValue::Int8(Some(value)) => Some(i128::from(*value)),
        ScalarValue::Int16(Some(value)) => Some(i128::from(*value)),
        ScalarValue::Int32(Some(value)) | ScalarValue::Date32(Some(value)) => {
            Some(i128::from(*value))
        }
        ScalarValue::Int64(Some(value))
        | ScalarValue::Date64(Some(value))
        | ScalarValue::TimestampMicrosecond(Some(value), _) => Some(i128::from(*value)),
        ScalarValue::UInt8(Some(value)) => Some(i128::from(*value)),
        ScalarValue::UInt16(Some(value)) => Some(i128::from(*value)),
        ScalarValue::UInt32(Some(value)) => Some(i128::from(*value)),
        ScalarValue::UInt64(Some(value)) => Some(i128::from(*value)),
        _ => None,
    }
}

/// Kernel result: derived output columns plus the per-entity history updates
/// the caller installs only after complete success (transactional state).
#[derive(Debug)]
struct ComputedOutputs {
    columns: Vec<ArrayRef>,
    touched: HistoryUpdates,
}

/// Computes every declared rolling output over `rows` in canonical order,
/// reading entity histories without mutating them (SCE-00 D5: batch and
/// stream lifecycles share this kernel and row order).
fn compute_output_columns(
    rows: &[BufferedRow],
    histories: &RollingHistories,
    compiled: &CompiledRollingSpec,
    node_id: &str,
) -> Result<ComputedOutputs> {
    compute_output_columns_observed(rows, histories, compiled, node_id, None)
}

fn compute_output_columns_observed(
    rows: &[BufferedRow],
    histories: &RollingHistories,
    compiled: &CompiledRollingSpec,
    node_id: &str,
    observer: Option<&RollingMetricsRecorder>,
) -> Result<ComputedOutputs> {
    if rows.is_empty() {
        return Ok(ComputedOutputs {
            columns: compiled
                .outputs
                .iter()
                .map(|output| new_null_array(&output.output_type, 0))
                .collect(),
            touched: Vec::new(),
        });
    }
    let arrow_stage = observer.map(|recorder| recorder.stage(RollingStage::ArrowOutput));
    let mut derived: Vec<Vec<Option<ScalarValue>>> = compiled
        .outputs
        .iter()
        .map(|_| vec![None; rows.len()])
        .collect();
    drop(arrow_stage);
    let entities = {
        let _stage = observer.map(|recorder| recorder.stage(RollingStage::EntityResolution));
        let entities = group_rows_by_entity(rows);
        if let Some(recorder) = observer {
            recorder.add(RollingWork::ResolvedRows, rows.len());
            recorder.add(RollingWork::TouchedEntities, entities.len());
        }
        entities
    };
    let mut touched = Vec::with_capacity(entities.len());
    for (entity, indices) in entities {
        let mut entity_state = {
            let _stage = observer.map(|recorder| recorder.stage(RollingStage::StatePreparation));
            let prior = histories.by_entity.get(entity);
            if let Some(recorder) = observer {
                recorder.add(RollingWork::CopiedEntities, usize::from(prior.is_some()));
            }
            prior
                .cloned()
                .unwrap_or_else(|| EntityRollingState::fresh(compiled))
        };
        {
            let _stage = observer.map(|recorder| recorder.stage(RollingStage::NumericUpdate));
            let view = EntityRowView {
                rows,
                indices: &indices,
                history: &entity_state.rows,
                event_time_index: compiled.event_time_index,
            };
            let mut incremental_scans = compiled
                .outputs
                .iter()
                .map(|output| IncrementalScanState::from_history(output, &view))
                .collect::<Vec<_>>();
            for (position, &row_index) in indices.iter().enumerate() {
                entity_state.transition_count = entity_state
                    .transition_count
                    .checked_add(1)
                    .ok_or_else(|| {
                        operator_error(node_id, "rolling entity transition count overflowed")
                    })?;
                if let Some(recorder) = observer {
                    recorder.add(RollingWork::NumericRows, 1);
                }
                slide_windows(
                    &view,
                    position,
                    row_index,
                    entity_state.transition_count,
                    compiled,
                    &mut entity_state.windows,
                    node_id,
                )?;
                for (ordinal, output) in compiled.outputs.iter().enumerate() {
                    let value = if let Some(scan) = &mut incremental_scans[ordinal] {
                        scan.advance(
                            view.history.len() + position,
                            view.value(view.history.len() + position, output.input_index),
                        )
                    } else {
                        compute_output_value(
                            &view,
                            position,
                            row_index,
                            output,
                            &entity_state.windows,
                            node_id,
                        )?
                    };
                    derived[ordinal][row_index] = Some(value);
                }
            }
        }
        let _stage = observer.map(|recorder| recorder.stage(RollingStage::HistoryMaintenance));
        for &row_index in &indices {
            entity_state.rows.push_back(rows[row_index].values.clone());
        }
        evict_retained_history(&mut entity_state, compiled);
        touched.push((entity.clone(), entity_state));
    }
    let _stage = observer.map(|recorder| recorder.stage(RollingStage::ArrowOutput));
    let columns = encode_derived_columns(derived, compiled, node_id)?;
    Ok(ComputedOutputs { columns, touched })
}

/// Evicts rows that no declared output can ever observe again (SCE-08): a
/// row stays retained while it is within the last `max_row_retention` rows
/// of the entity's total order or inside the widest duration frame of the
/// last processed row. Both needs are suffixes of the canonical order, so
/// eviction pops from the front.
fn evict_retained_history(state: &mut EntityRollingState, compiled: &CompiledRollingSpec) {
    let max_rows = usize::try_from(compiled.max_row_retention).unwrap_or(usize::MAX);
    let bound = compiled
        .max_duration_micros
        .zip(
            state
                .rows
                .back()
                .map(|values| history_event_time(values, compiled)),
        )
        .map(|(micros, last)| i128::from(last) - i128::from(micros));
    while state.rows.len() > max_rows {
        let still_needed_by_time = bound.is_some_and(|bound| {
            state
                .rows
                .front()
                .is_some_and(|values| i128::from(history_event_time(values, compiled)) > bound)
        });
        if still_needed_by_time {
            break;
        }
        state.rows.pop_front();
    }
}

fn history_event_time(values: &[ScalarValue], compiled: &CompiledRollingSpec) -> i64 {
    match &values[compiled.event_time_index] {
        ScalarValue::TimestampMicrosecond(Some(value), _) => *value,
        _ => unreachable!("rolling history rows carry a timestamp event time"),
    }
}

fn group_rows_by_entity(rows: &[BufferedRow]) -> BTreeMap<&Vec<Option<KeyValue>>, Vec<usize>> {
    let mut entities: BTreeMap<&Vec<Option<KeyValue>>, Vec<usize>> = BTreeMap::new();
    for (index, row) in rows.iter().enumerate() {
        entities
            .entry(&row.identity.entity)
            .or_default()
            .push(index);
    }
    entities
}

fn encode_derived_columns(
    derived: Vec<Vec<Option<ScalarValue>>>,
    compiled: &CompiledRollingSpec,
    node_id: &str,
) -> Result<Vec<ArrayRef>> {
    derived
        .into_iter()
        .zip(&compiled.outputs)
        .map(|(column, output)| {
            ScalarValue::iter_to_array(
                column
                    .into_iter()
                    .map(|value| value.unwrap_or_else(|| typed_null(&output.input_type))),
            )
            .map_err(|error| {
                operator_error(
                    node_id,
                    &format!("rolling output column encoding failed: {error}"),
                )
            })
        })
        .collect()
}

/// Read-only view over one entity's retained history tail and its current
/// emission batch rows in canonical order; combined positions index the
/// history first and the batch rows second.
struct EntityRowView<'a> {
    rows: &'a [BufferedRow],
    indices: &'a [usize],
    history: &'a VecDeque<Vec<ScalarValue>>,
    event_time_index: usize,
}

impl EntityRowView<'_> {
    fn value(&self, combined: usize, input_index: usize) -> &ScalarValue {
        if combined < self.history.len() {
            &self.history[combined][input_index]
        } else {
            &self.rows[self.indices[combined - self.history.len()]].values[input_index]
        }
    }

    fn event_time(&self, combined: usize) -> i64 {
        if combined < self.history.len() {
            match &self.history[combined][self.event_time_index] {
                ScalarValue::TimestampMicrosecond(Some(value), _) => *value,
                _ => unreachable!("rolling history rows carry a timestamp event time"),
            }
        } else {
            self.rows[self.indices[combined - self.history.len()]]
                .identity
                .event_time
        }
    }

    /// Canonical expiry key of the row at one combined position.
    fn extrema_key(&self, combined: usize, compiled: &CompiledRollingSpec) -> ExtremaKey {
        if combined < self.history.len() {
            ExtremaKey {
                event_time: self.event_time(combined),
                sequence: compiled
                    .sequence_columns
                    .iter()
                    .map(|column| {
                        KeyValue::from_required_scalar(
                            &self.history[combined][column.index],
                            "rolling",
                        )
                        .expect("rolling history rows carry required sequence keys")
                    })
                    .collect(),
            }
        } else {
            let identity = &self.rows[self.indices[combined - self.history.len()]].identity;
            ExtremaKey {
                event_time: identity.event_time,
                sequence: identity.sequence.clone(),
            }
        }
    }

    /// First combined position whose event time exceeds `bound`; both the
    /// retained history and the batch rows are key-ordered, so the search is
    /// two partition points.
    fn first_after_bound(&self, bound: i128) -> usize {
        let (front, _back) = self.history.as_slices();
        let in_front = front.partition_point(|values| {
            matches!(
                &values[self.event_time_index],
                ScalarValue::TimestampMicrosecond(Some(time), _) if i128::from(*time) <= bound
            )
        });
        if in_front < front.len() {
            return in_front;
        }
        let in_back = self.history.as_slices().1.partition_point(|values| {
            matches!(
                &values[self.event_time_index],
                ScalarValue::TimestampMicrosecond(Some(time), _) if i128::from(*time) <= bound
            )
        });
        if in_front + in_back < self.history.len() {
            return in_front + in_back;
        }
        self.history.len()
            + self.indices.partition_point(|&row_index| {
                i128::from(self.rows[row_index].identity.event_time) <= bound
            })
    }
}

struct WindowSlideContext<'a> {
    view: &'a EntityRowView<'a>,
    row_index: usize,
    transition_count: u64,
    compiled: &'a CompiledRollingSpec,
    node_id: &'a str,
    combined: usize,
}

/// Slides every shared window group to the current row. Add precedes removal
/// so the frozen order matches the rebuild fold exactly.
fn slide_windows(
    view: &EntityRowView<'_>,
    position: usize,
    row_index: usize,
    transition_count: u64,
    compiled: &CompiledRollingSpec,
    windows: &mut [WindowState],
    node_id: &str,
) -> Result<()> {
    if compiled.window_groups.len() != windows.len() {
        return Err(internal_error(
            "rolling window group declarations and states differ",
        ));
    }
    let context = WindowSlideContext {
        view,
        row_index,
        transition_count,
        compiled,
        node_id,
        combined: view.history.len() + position,
    };
    for (group, state) in compiled.window_groups.iter().zip(windows) {
        match group {
            CompiledWindowGroup::Numeric {
                input_index, frame, ..
            } => slide_numeric_window(&context, *input_index, *frame, state)?,
            CompiledWindowGroup::Extrema {
                input_index, frame, ..
            } => slide_extrema_window(&context, *input_index, *frame, state)?,
            CompiledWindowGroup::Pair {
                left_index,
                right_index,
                frame,
            } => slide_pair_window(&context, *left_index, *right_index, *frame, state)?,
            CompiledWindowGroup::Ewma {
                input_index, alpha, ..
            } => slide_ewma_window(&context, *input_index, *alpha, state)?,
        }
    }
    Ok(())
}

fn slide_numeric_window(
    context: &WindowSlideContext<'_>,
    input_index: usize,
    frame: CompiledFrame,
    state: &mut WindowState,
) -> Result<()> {
    let WindowState::Numeric(accumulator) = state else {
        return Err(internal_error("rolling numeric group has the wrong state"));
    };
    let current = &context.view.rows[context.row_index].values[input_index];
    if is_valid_sample(current) {
        accumulator.add(current, context.node_id)?;
    }
    expire_numeric(
        accumulator,
        context.view,
        context.combined,
        input_index,
        frame,
        context.node_id,
    )?;
    let stable_v2_rebase = spec_uses_stable_v2(context.compiled)
        && accumulator.has_float_sum()
        && kernel::stable_v2_rebase_due(
            context.transition_count,
            window_positions(context.view, context.combined, frame, context.node_id)?.count(),
            accumulator.is_non_finite(),
        );
    if stable_v2_rebase {
        rebase_numeric_stable_v2(
            accumulator,
            context.view,
            context.combined,
            input_index,
            frame,
            context.node_id,
        )
    } else if accumulator.is_non_finite() {
        refold_numeric(
            accumulator,
            context.view,
            context.combined,
            input_index,
            frame,
            context.node_id,
        )
    } else {
        Ok(())
    }
}

fn slide_extrema_window(
    context: &WindowSlideContext<'_>,
    input_index: usize,
    frame: CompiledFrame,
    state: &mut WindowState,
) -> Result<()> {
    let WindowState::Extrema(accumulator) = state else {
        return Err(internal_error("rolling extrema group has the wrong state"));
    };
    let current = &context.view.rows[context.row_index].values[input_index];
    if is_valid_sample(current) {
        accumulator.add(
            context.view.extrema_key(context.combined, context.compiled),
            current.clone(),
        );
    }
    expire_extrema(
        accumulator,
        context.view,
        context.combined,
        input_index,
        frame,
        context.compiled,
        context.node_id,
    )
}

fn slide_pair_window(
    context: &WindowSlideContext<'_>,
    left_index: usize,
    right_index: usize,
    frame: CompiledFrame,
    state: &mut WindowState,
) -> Result<()> {
    let WindowState::Pair(accumulator) = state else {
        return Err(internal_error("rolling pair group has the wrong state"));
    };
    let left = &context.view.rows[context.row_index].values[left_index];
    let right = &context.view.rows[context.row_index].values[right_index];
    if is_valid_sample(left) && is_valid_sample(right) {
        accumulator.add(left, right, context.node_id)?;
    }
    expire_pair(
        accumulator,
        context.view,
        context.combined,
        left_index,
        right_index,
        frame,
        context.node_id,
    )?;
    let stable_v2_rebase = spec_uses_stable_v2(context.compiled)
        && kernel::stable_v2_rebase_due(
            context.transition_count,
            window_positions(context.view, context.combined, frame, context.node_id)?.count(),
            accumulator.is_non_finite(),
        );
    if stable_v2_rebase {
        rebase_pair_stable_v2(
            accumulator,
            context.view,
            context.combined,
            left_index,
            right_index,
            frame,
            context.node_id,
        )
    } else if accumulator.is_non_finite() {
        refold_pair(
            accumulator,
            context.view,
            context.combined,
            left_index,
            right_index,
            frame,
            context.node_id,
        )
    } else {
        Ok(())
    }
}

fn slide_ewma_window(
    context: &WindowSlideContext<'_>,
    input_index: usize,
    alpha: f64,
    state: &mut WindowState,
) -> Result<()> {
    let WindowState::Ewma(accumulator) = state else {
        return Err(internal_error("rolling EWMA group has the wrong state"));
    };
    let current = &context.view.rows[context.row_index].values[input_index];
    if is_valid_sample(current) {
        accumulator.add(current, alpha, context.node_id)?;
    }
    Ok(())
}

fn expire_extrema(
    accumulator: &mut ExtremaAccumulator,
    view: &EntityRowView<'_>,
    combined: usize,
    input_index: usize,
    frame: CompiledFrame,
    compiled: &CompiledRollingSpec,
    node_id: &str,
) -> Result<()> {
    match frame {
        CompiledFrame::Rows(rows) => {
            let rows = usize::try_from(rows)
                .map_err(|_| operator_error(node_id, "rolling frame rows do not fit usize"))?;
            if combined >= rows {
                let expiring = view.value(combined - rows, input_index);
                if is_valid_sample(expiring) {
                    accumulator.remove();
                }
                let leaving = view.extrema_key(combined - rows, compiled);
                accumulator.expire_through_key(&leaving);
            }
        }
        CompiledFrame::Duration(micros) => {
            let bound = duration_bound(view.event_time(combined), micros);
            let mut index = view.first_after_bound(accumulator.expired_through);
            while index < combined && i128::from(view.event_time(index)) <= bound {
                if is_valid_sample(view.value(index, input_index)) {
                    accumulator.remove();
                }
                index += 1;
            }
            accumulator.expired_through = accumulator.expired_through.max(bound);
            accumulator.expire_through_time(bound);
        }
    }
    Ok(())
}

/// Exclusive duration lower bound `t - d` in exact `i128` arithmetic; the
/// frame is `(t - d, t]` (SCE-00 D5), and the widened subtraction cannot
/// wrap.
fn duration_bound(event_time: i64, micros: u64) -> i128 {
    i128::from(event_time) - i128::from(micros)
}

/// Removes every numeric sample that left the frame at the current row: one
/// row for row-count frames, the newly expired event-time prefix for
/// duration frames.
fn expire_numeric(
    accumulator: &mut WindowAccumulator,
    view: &EntityRowView<'_>,
    combined: usize,
    input_index: usize,
    frame: CompiledFrame,
    node_id: &str,
) -> Result<()> {
    match frame {
        CompiledFrame::Rows(rows) => {
            let rows = usize::try_from(rows)
                .map_err(|_| operator_error(node_id, "rolling frame rows do not fit usize"))?;
            if combined >= rows {
                let expiring = view.value(combined - rows, input_index);
                if is_valid_sample(expiring) {
                    accumulator.remove(expiring)?;
                }
            }
            Ok(())
        }
        CompiledFrame::Duration(micros) => {
            let bound = duration_bound(view.event_time(combined), micros);
            let mut index = view.first_after_bound(accumulator.expired_through);
            while index < combined && i128::from(view.event_time(index)) <= bound {
                let expiring = view.value(index, input_index);
                if is_valid_sample(expiring) {
                    accumulator.remove(expiring)?;
                }
                index += 1;
            }
            accumulator.expired_through = accumulator.expired_through.max(bound);
            Ok(())
        }
    }
}

/// Removes every pairwise-valid sample that left the frame at the current
/// row (SCE-08).
fn expire_pair(
    accumulator: &mut PairAccumulator,
    view: &EntityRowView<'_>,
    combined: usize,
    left_index: usize,
    right_index: usize,
    frame: CompiledFrame,
    node_id: &str,
) -> Result<()> {
    match frame {
        CompiledFrame::Rows(rows) => {
            let rows = usize::try_from(rows)
                .map_err(|_| operator_error(node_id, "rolling frame rows do not fit usize"))?;
            if combined >= rows {
                let x = view.value(combined - rows, left_index);
                let y = view.value(combined - rows, right_index);
                if is_valid_sample(x) && is_valid_sample(y) {
                    accumulator.remove(x, y)?;
                }
            }
            Ok(())
        }
        CompiledFrame::Duration(micros) => {
            let bound = duration_bound(view.event_time(combined), micros);
            let mut index = view.first_after_bound(accumulator.expired_through);
            while index < combined && i128::from(view.event_time(index)) <= bound {
                let x = view.value(index, left_index);
                let y = view.value(index, right_index);
                if is_valid_sample(x) && is_valid_sample(y) {
                    accumulator.remove(x, y)?;
                }
                index += 1;
            }
            accumulator.expired_through = accumulator.expired_through.max(bound);
            Ok(())
        }
    }
}

/// Rebuilds one numeric accumulator as the ordered fold over the current
/// window; this is the same construction the checkpoint restore applies to
/// retained history (SCE-00 D11/D13).
fn refold_numeric(
    accumulator: &mut WindowAccumulator,
    view: &EntityRowView<'_>,
    combined: usize,
    input_index: usize,
    frame: CompiledFrame,
    node_id: &str,
) -> Result<()> {
    accumulator.reset();
    for index in window_positions(view, combined, frame, node_id)? {
        let value = view.value(index, input_index);
        if is_valid_sample(value) {
            accumulator.add(value, node_id)?;
        }
    }
    if let CompiledFrame::Duration(micros) = frame {
        accumulator.expired_through = duration_bound(view.event_time(combined), micros);
    }
    Ok(())
}

fn rebase_numeric_stable_v2(
    accumulator: &mut WindowAccumulator,
    view: &EntityRowView<'_>,
    combined: usize,
    input_index: usize,
    frame: CompiledFrame,
    node_id: &str,
) -> Result<()> {
    let expired_through = accumulator.expired_through;
    *accumulator = kernel::stable_v2_float64_accumulator(
        window_positions(view, combined, frame, node_id)?.filter_map(|index| {
            let value = view.value(index, input_index);
            is_valid_sample(value).then(|| float_sample(value))
        }),
        node_id,
    )?;
    accumulator.expired_through = expired_through;
    Ok(())
}

/// Rebuilds one pair accumulator as the ordered fold over the current
/// pairwise-valid window (SCE-00 D11/D13).
fn refold_pair(
    accumulator: &mut PairAccumulator,
    view: &EntityRowView<'_>,
    combined: usize,
    left_index: usize,
    right_index: usize,
    frame: CompiledFrame,
    node_id: &str,
) -> Result<()> {
    accumulator.reset();
    for index in window_positions(view, combined, frame, node_id)? {
        let x = view.value(index, left_index);
        let y = view.value(index, right_index);
        if is_valid_sample(x) && is_valid_sample(y) {
            accumulator.add(x, y, node_id)?;
        }
    }
    if let CompiledFrame::Duration(micros) = frame {
        accumulator.expired_through = duration_bound(view.event_time(combined), micros);
    }
    Ok(())
}

fn rebase_pair_stable_v2(
    accumulator: &mut PairAccumulator,
    view: &EntityRowView<'_>,
    combined: usize,
    left_index: usize,
    right_index: usize,
    frame: CompiledFrame,
    node_id: &str,
) -> Result<()> {
    let expired_through = accumulator.expired_through;
    *accumulator = kernel::stable_v2_pair_accumulator(
        window_positions(view, combined, frame, node_id)?.filter_map(|index| {
            let left = view.value(index, left_index);
            let right = view.value(index, right_index);
            (is_valid_sample(left) && is_valid_sample(right))
                .then(|| (float_sample(left), float_sample(right)))
        }),
        node_id,
    )?;
    accumulator.expired_through = expired_through;
    Ok(())
}

/// Combined positions of the current row's window members in canonical
/// order: the last `rows` positions for row frames, the positions with
/// event time in `(t - d, t]` for duration frames.
fn window_positions(
    view: &EntityRowView<'_>,
    combined: usize,
    frame: CompiledFrame,
    node_id: &str,
) -> Result<std::ops::RangeInclusive<usize>> {
    let start = match frame {
        CompiledFrame::Rows(rows) => {
            let rows = usize::try_from(rows)
                .map_err(|_| operator_error(node_id, "rolling frame rows do not fit usize"))?;
            (combined + 1).saturating_sub(rows)
        }
        CompiledFrame::Duration(micros) => {
            let bound = duration_bound(view.event_time(combined), micros);
            view.first_after_bound(bound)
        }
    };
    Ok(start..=combined)
}

fn compute_output_value(
    view: &EntityRowView<'_>,
    position: usize,
    row_index: usize,
    output: &CompiledRollingOutput,
    windows: &[WindowState],
    node_id: &str,
) -> Result<ScalarValue> {
    let periods = match &output.evaluation {
        CompiledEvaluation::Lag { periods } | CompiledEvaluation::Delta { periods } => {
            usize::try_from(*periods)
                .map_err(|_| operator_error(node_id, "rolling periods does not fit usize"))?
        }
        CompiledEvaluation::Aggregate(aggregate) => {
            return evaluate_aggregate(aggregate, windows, output, node_id);
        }
        CompiledEvaluation::Ewma(ewma) => {
            return evaluate_ewma(ewma, windows, output);
        }
        CompiledEvaluation::Pair(aggregate) => {
            return evaluate_pair_aggregate(aggregate, windows, output);
        }
        CompiledEvaluation::Difference(difference) => {
            return evaluate_difference(difference, windows);
        }
        CompiledEvaluation::Scan(scan) => {
            return evaluate_scan(view, position, output, *scan, node_id);
        }
    };
    let referenced = if position + view.history.len() < periods {
        None
    } else if position >= periods {
        Some(view.rows[view.indices[position - periods]].values[output.input_index].clone())
    } else {
        Some(view.history[view.history.len() + position - periods][output.input_index].clone())
    };
    if matches!(output.evaluation, CompiledEvaluation::Lag { .. }) {
        return Ok(referenced.unwrap_or_else(|| typed_null(&output.input_type)));
    }
    let current = &view.rows[row_index].values[output.input_index];
    if current.is_null() {
        return Ok(typed_null(&output.input_type));
    }
    let Some(reference) = referenced.filter(|value| !value.is_null()) else {
        return Ok(typed_null(&output.input_type));
    };
    current.sub_checked(&reference).map_err(|error| {
        operator_error(
            node_id,
            &format!("rolling delta failed with checked arithmetic: {error}"),
        )
    })
}

/// Batch-local scan state is rebuilt from the retained entity tail. The tail
/// is already part of the durable checkpoint, so no second scan-state format
/// or cross-callback mutable cache is required.
#[derive(Clone, Debug)]
enum IncrementalScanState {
    Extrema {
        window: usize,
        min_periods: u64,
        descending: bool,
        recent_valid: VecDeque<bool>,
        candidates: VecDeque<(usize, f64)>,
        valid_count: usize,
    },
    UniqueFloat64 {
        window: usize,
        min_periods: u64,
        recent: VecDeque<Option<u64>>,
        counts: HashMap<u64, usize>,
        valid_count: usize,
    },
}

impl IncrementalScanState {
    fn new(kind: ScanKind, window: usize, min_periods: u64) -> Option<Self> {
        match kind {
            ScanKind::Argmax | ScanKind::Argmin => Some(Self::Extrema {
                window,
                min_periods,
                descending: matches!(kind, ScanKind::Argmax),
                recent_valid: VecDeque::new(),
                candidates: VecDeque::new(),
                valid_count: 0,
            }),
            ScanKind::UniqueCount => Some(Self::UniqueFloat64 {
                window,
                min_periods,
                recent: VecDeque::new(),
                counts: HashMap::new(),
                valid_count: 0,
            }),
            ScanKind::Rank | ScanKind::Quantile | ScanKind::Decay => None,
        }
    }

    fn from_history(output: &CompiledRollingOutput, view: &EntityRowView<'_>) -> Option<Self> {
        let CompiledEvaluation::Scan(scan) = &output.evaluation else {
            return None;
        };
        if output.input_type != DataType::Float64 {
            return None;
        }
        let CompiledFrame::Rows(rows) = scan.frame else {
            return None;
        };
        let window = usize::try_from(rows).ok()?.max(1);
        let mut state = Self::new(scan.kind, window, scan.min_periods)?;
        for index in view.history.len().saturating_sub(window - 1)..view.history.len() {
            state.advance(index, view.value(index, output.input_index));
        }
        Some(state)
    }

    fn advance(&mut self, index: usize, value: &ScalarValue) -> ScalarValue {
        let sample = match value {
            ScalarValue::Float64(Some(value)) if !value.is_nan() => Some(*value),
            _ => None,
        };
        ScalarValue::UInt64(self.advance_float(index, sample))
    }

    fn valid_count(&self) -> usize {
        match self {
            Self::Extrema { valid_count, .. } | Self::UniqueFloat64 { valid_count, .. } => {
                *valid_count
            }
        }
    }

    fn estimated_bytes(&self) -> usize {
        match self {
            Self::Extrema {
                recent_valid,
                candidates,
                ..
            } => recent_valid
                .capacity()
                .saturating_mul(size_of::<bool>())
                .saturating_add(
                    candidates
                        .capacity()
                        .saturating_mul(size_of::<(usize, f64)>()),
                ),
            Self::UniqueFloat64 { recent, counts, .. } => {
                let buckets = counts.capacity().saturating_mul(8).div_ceil(7);
                let table_bytes = if buckets == 0 {
                    0
                } else {
                    buckets
                        .saturating_mul(size_of::<(u64, usize)>() + 1)
                        .saturating_add(16)
                };
                recent
                    .capacity()
                    .saturating_mul(size_of::<Option<u64>>())
                    .saturating_add(table_bytes)
            }
        }
    }

    fn advance_float(&mut self, index: usize, sample: Option<f64>) -> Option<u64> {
        let sample = sample.filter(|value| !value.is_nan());
        match self {
            Self::Extrema {
                window,
                min_periods,
                descending,
                recent_valid,
                candidates,
                valid_count,
            } => {
                recent_valid.push_back(sample.is_some());
                if let Some(sample) = sample {
                    *valid_count += 1;
                    while candidates.back().is_some_and(|(_, previous)| {
                        if *descending {
                            *previous < sample
                        } else {
                            *previous > sample
                        }
                    }) {
                        candidates.pop_back();
                    }
                    candidates.push_back((index, sample));
                }
                if recent_valid.len() > *window {
                    if recent_valid.pop_front() == Some(true) {
                        *valid_count -= 1;
                    }
                    let leaving = index - *window;
                    if candidates
                        .front()
                        .is_some_and(|(position, _)| *position == leaving)
                    {
                        candidates.pop_front();
                    }
                }
                if u64::try_from(*valid_count).unwrap_or(u64::MAX) < *min_periods {
                    return None;
                }
                candidates
                    .front()
                    .and_then(|(position, _)| u64::try_from(index - *position).ok())
            }
            Self::UniqueFloat64 {
                window,
                min_periods,
                recent,
                counts,
                valid_count,
            } => {
                // IEEE zero signs compare equal in the public unique-count
                // contract. NaNs were excluded with the other invalid samples.
                let key = sample.map(|value| if value == 0.0 { 0 } else { value.to_bits() });
                recent.push_back(key);
                if let Some(key) = key {
                    *counts.entry(key).or_default() += 1;
                    *valid_count += 1;
                }
                if recent.len() > *window
                    && let Some(leaving) = recent.pop_front().flatten()
                {
                    *valid_count -= 1;
                    let count = counts.get_mut(&leaving).expect("recent key is counted");
                    *count -= 1;
                    if *count == 0 {
                        counts.remove(&leaving);
                    }
                }
                if u64::try_from(*valid_count).unwrap_or(u64::MAX) < *min_periods {
                    None
                } else {
                    u64::try_from(counts.len()).ok()
                }
            }
        }
    }
}

#[allow(
    clippy::cast_precision_loss,
    reason = "rank and linear weights are defined as Float64 readouts"
)]
fn evaluate_scan(
    view: &EntityRowView<'_>,
    position: usize,
    output: &CompiledRollingOutput,
    scan: CompiledScan,
    node_id: &str,
) -> Result<ScalarValue> {
    let combined = view.history.len() + position;
    let samples: Vec<(usize, &ScalarValue)> =
        window_positions(view, combined, scan.frame, node_id)?
            .filter_map(|index| {
                let value = view.value(index, output.input_index);
                is_valid_sample(value).then_some((index, value))
            })
            .collect();
    if u64::try_from(samples.len()).unwrap_or(u64::MAX) < scan.min_periods {
        return Ok(typed_null(&output.output_type));
    }
    let current = view.value(combined, output.input_index);
    match scan.kind {
        ScanKind::Argmax | ScanKind::Argmin => {
            let mut extreme = samples[0];
            for sample in &samples[1..] {
                let ordering = compare_scan_samples(sample.1, extreme.1);
                if (matches!(scan.kind, ScanKind::Argmax) && ordering == Ordering::Greater)
                    || (matches!(scan.kind, ScanKind::Argmin) && ordering == Ordering::Less)
                {
                    extreme = *sample;
                }
            }
            Ok(ScalarValue::UInt64(Some((combined - extreme.0) as u64)))
        }
        ScanKind::Rank | ScanKind::Quantile => {
            if !is_valid_sample(current) {
                return Ok(typed_null(&output.output_type));
            }
            let rank = samples
                .iter()
                .filter(|(_, value)| compare_scan_samples(value, current) == Ordering::Less)
                .count();
            if matches!(scan.kind, ScanKind::Rank) {
                Ok(ScalarValue::UInt64(Some(rank as u64)))
            } else if samples.len() == 1 {
                Ok(ScalarValue::Float64(None))
            } else {
                Ok(ScalarValue::Float64(Some(
                    rank as f64 / (samples.len() - 1) as f64,
                )))
            }
        }
        ScanKind::UniqueCount => {
            let mut values: Vec<&ScalarValue> = samples.iter().map(|(_, value)| *value).collect();
            values.sort_by(|left, right| compare_scan_samples(left, right));
            values.dedup_by(|left, right| compare_scan_samples(left, right) == Ordering::Equal);
            Ok(ScalarValue::UInt64(Some(values.len() as u64)))
        }
        ScanKind::Decay => {
            let count = samples.len();
            let size = match scan.frame {
                CompiledFrame::Rows(size) => usize::try_from(size).map_err(|_| {
                    operator_error(node_id, "rolling decay frame does not fit usize")
                })?,
                CompiledFrame::Duration(_) => count,
            };
            let first_weight = size - count + 1;
            let (weighted_sum, total_weight) =
                samples
                    .iter()
                    .enumerate()
                    .fold((0.0, 0.0), |(sum, total), (index, (_, value))| {
                        let weight = (first_weight + index) as f64;
                        (sum + weight * float_sample(value), total + weight)
                    });
            Ok(ScalarValue::Float64(Some(weighted_sum / total_weight)))
        }
    }
}

/// Finance-style rank and set equality treats both IEEE zero signs as equal.
/// NaN is excluded before this comparison; existing extrema queues keep their
/// separate frozen total-order behavior.
fn compare_scan_samples(left: &ScalarValue, right: &ScalarValue) -> Ordering {
    #[cfg(test)]
    SCAN_COMPARE_COUNT.with(|count| count.set(count.get() + 1));
    match (left, right) {
        (ScalarValue::Float32(Some(left)), ScalarValue::Float32(Some(right))) => {
            left.partial_cmp(right).unwrap_or(Ordering::Equal)
        }
        (ScalarValue::Float64(Some(left)), ScalarValue::Float64(Some(right))) => {
            left.partial_cmp(right).unwrap_or(Ordering::Equal)
        }
        _ => compare_samples(left, right),
    }
}

#[cfg(test)]
thread_local! {
    static SCAN_COMPARE_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn evaluate_difference(
    difference: &CompiledDifference,
    windows: &[WindowState],
) -> Result<ScalarValue> {
    Ok(ScalarValue::Float64(
        evaluate_float_readout(difference.left, windows)?
            .zip(evaluate_float_readout(difference.right, windows)?)
            .map(|(left, right)| left - right),
    ))
}

#[allow(
    clippy::cast_precision_loss,
    reason = "the frozen fused rolling readout type is Float64"
)]
fn evaluate_float_readout(
    readout: CompiledFloatReadout,
    windows: &[WindowState],
) -> Result<Option<f64>> {
    match readout {
        CompiledFloatReadout::Ewma(readout) => {
            let WindowState::Ewma(state) = &windows[readout.group] else {
                return Err(internal_error("fused EWMA readout state mismatch"));
            };
            Ok((state.valid_count >= readout.min_periods).then_some(state.value))
        }
        CompiledFloatReadout::Aggregate(readout) => {
            let WindowState::Numeric(state) = &windows[readout.group] else {
                return Err(internal_error("fused aggregate readout state mismatch"));
            };
            if state.valid_count < readout.min_periods {
                return Ok(None);
            }
            match readout.statistic {
                Statistic::Mean => Ok(Some(match (state.pos_inf > 0, state.neg_inf > 0) {
                    (true, true) => f64::NAN,
                    (true, false) => f64::INFINITY,
                    (false, true) => f64::NEG_INFINITY,
                    (false, false) => state.mean,
                })),
                Statistic::Variance | Statistic::Stddev => {
                    let divisor = state.valid_count - u64::from(readout.ddof);
                    if divisor == 0 {
                        return Ok(None);
                    }
                    if state.pos_inf > 0 || state.neg_inf > 0 {
                        return Ok(Some(f64::NAN));
                    }
                    let variance = state.m2.max(0.0) / divisor as f64;
                    Ok(Some(if readout.statistic == Statistic::Stddev {
                        variance.sqrt()
                    } else {
                        variance
                    }))
                }
                _ => Err(internal_error(
                    "fused aggregate readout has a non-floating statistic",
                )),
            }
        }
    }
}

fn evaluate_ewma(
    ewma: &CompiledEwma,
    windows: &[WindowState],
    output: &CompiledRollingOutput,
) -> Result<ScalarValue> {
    let WindowState::Ewma(accumulator) = &windows[ewma.group] else {
        return Err(internal_error("rolling EWMA output reads a non-EWMA group"));
    };
    if accumulator.valid_count < ewma.min_periods {
        return Ok(typed_null(&output.output_type));
    }
    Ok(ScalarValue::Float64(Some(accumulator.value)))
}

/// Reads one aggregate output from its shared window accumulator: the
/// minimum-period gate uses the valid sample count (SCE-00 D3, contract
/// section 5.2), and the variance divisor is `valid_count - ddof` with a
/// non-positive divisor producing null (SCE-00 D5). Windows holding ±inf
/// samples classify from the reversible infinity counts — both signs is the
/// undefined ∞ − ∞ (NaN), one sign is that infinity, and no infinity keeps
/// the frozen finite-path West readout (SCE-07 defect 1 ruling);
/// variance/stddev over a window with any infinity is NaN because every
/// deviation involves ∞ − ∞.
#[allow(
    clippy::cast_precision_loss,
    reason = "the frozen aggregate output type is Float64"
)]
fn evaluate_aggregate(
    aggregate: &CompiledAggregate,
    windows: &[WindowState],
    output: &CompiledRollingOutput,
    node_id: &str,
) -> Result<ScalarValue> {
    let accumulator = match &windows[aggregate.group] {
        WindowState::Numeric(accumulator) => accumulator,
        WindowState::Extrema(accumulator) => {
            // Min/max read the monotonic queue front after the count gate.
            if accumulator.valid_count < aggregate.min_periods {
                return Ok(typed_null(&output.output_type));
            }
            return Ok(accumulator
                .extremum()
                .cloned()
                .unwrap_or_else(|| typed_null(&output.output_type)));
        }
        WindowState::Pair(_) => {
            return Err(internal_error(
                "rolling pair group serves a numeric aggregate output",
            ));
        }
        WindowState::Ewma(_) => {
            return Err(internal_error(
                "rolling EWMA group serves a numeric aggregate output",
            ));
        }
    };
    if accumulator.valid_count < aggregate.min_periods {
        return Ok(typed_null(&output.output_type));
    }
    match aggregate.statistic {
        Statistic::Count => Ok(ScalarValue::UInt64(Some(accumulator.valid_count))),
        Statistic::Sum => match accumulator.sum {
            Some(SumState::Signed(total)) => i64::try_from(total)
                .map(|narrowed| ScalarValue::Int64(Some(narrowed)))
                .map_err(|_| operator_error(node_id, "rolling integer sum overflowed")),
            Some(SumState::Unsigned(total)) => u64::try_from(total)
                .map(|narrowed| ScalarValue::UInt64(Some(narrowed)))
                .map_err(|_| operator_error(node_id, "rolling integer sum overflowed")),
            Some(SumState::Float(total)) => Ok(ScalarValue::Float64(Some(total))),
            None => Err(operator_error(
                node_id,
                "rolling sum requires a numeric window group",
            )),
        },
        Statistic::Mean => Ok(ScalarValue::Float64(Some(
            match (accumulator.pos_inf > 0, accumulator.neg_inf > 0) {
                (true, true) => f64::NAN,
                (true, false) => f64::INFINITY,
                (false, true) => f64::NEG_INFINITY,
                (false, false) => accumulator.mean,
            },
        ))),
        Statistic::Variance | Statistic::Stddev => {
            let divisor = accumulator.valid_count - u64::from(aggregate.ddof);
            if divisor == 0 {
                return Ok(ScalarValue::Float64(None));
            }
            if accumulator.pos_inf > 0 || accumulator.neg_inf > 0 {
                return Ok(ScalarValue::Float64(Some(f64::NAN)));
            }
            // Negative M2 is floating-point removal drift, never a real
            // negative variance; NaN propagates as the frozen undefined value.
            let m2 = if accumulator.m2 < 0.0 {
                0.0
            } else {
                accumulator.m2
            };
            let variance = m2 / divisor as f64;
            Ok(ScalarValue::Float64(Some(match aggregate.statistic {
                Statistic::Variance => variance,
                _ => variance.sqrt(),
            })))
        }
        Statistic::Min | Statistic::Max => Err(internal_error(
            "rolling extrema statistic reads an extrema group",
        )),
    }
}

/// Reads one covariance/correlation output from its shared pair
/// accumulator (SCE-00 D3, contract section 5.2; D5): null below the
/// pairwise minimum count or a non-positive divisor, null for correlation
/// with zero variance on either side, NaN when the window holds any
/// infinity, and the West-style co-moment readout otherwise. The ddof
/// divisor cancels in the correlation ratio; it only participates in the
/// divisor gate.
#[allow(
    clippy::cast_precision_loss,
    reason = "the frozen pair output type is Float64"
)]
fn evaluate_pair_aggregate(
    aggregate: &CompiledPairAggregate,
    windows: &[WindowState],
    output: &CompiledRollingOutput,
) -> Result<ScalarValue> {
    let WindowState::Pair(accumulator) = &windows[aggregate.group] else {
        return Err(internal_error("rolling pair output reads a pair group"));
    };
    if accumulator.valid_count < aggregate.min_periods {
        return Ok(typed_null(&output.output_type));
    }
    let divisor = accumulator.valid_count - u64::from(aggregate.ddof);
    if divisor == 0 {
        return Ok(typed_null(&output.output_type));
    }
    if accumulator.holds_infinity() {
        return Ok(ScalarValue::Float64(Some(f64::NAN)));
    }
    if aggregate.correlation {
        // Clamp negative drift to zero: a true zero-variance side yields the
        // frozen null; tiny negative M2 is removal drift.
        let m2_x = if accumulator.m2_x < 0.0 {
            0.0
        } else {
            accumulator.m2_x
        };
        let m2_y = if accumulator.m2_y < 0.0 {
            0.0
        } else {
            accumulator.m2_y
        };
        if m2_x == 0.0 || m2_y == 0.0 {
            return Ok(typed_null(&output.output_type));
        }
        let scale = m2_x.sqrt() * m2_y.sqrt();
        Ok(ScalarValue::Float64(Some(accumulator.co_moment / scale)))
    } else {
        Ok(ScalarValue::Float64(Some(
            accumulator.co_moment / divisor as f64,
        )))
    }
}

fn typed_null(data_type: &DataType) -> ScalarValue {
    ScalarValue::try_from(data_type).unwrap_or(ScalarValue::Null)
}

fn validate_arguments(spec: &RollingSpec) -> Result<()> {
    if spec.configuration_version != ROLLING_CONFIGURATION_VERSION {
        return Err(invalid_argument(
            "rolling.configuration_version",
            "unsupported rolling configuration version",
        ));
    }
    if !matches!(
        spec.state_layout_version,
        ROLLING_STATE_LAYOUT_VERSION | ROLLING_EWMA_STATE_LAYOUT_VERSION
    ) {
        return Err(invalid_argument(
            "rolling.state_layout_version",
            "unsupported rolling state layout version",
        ));
    }
    if spec
        .outputs
        .iter()
        .any(RollingOutputSpec::requires_ewma_layout)
        && spec.state_layout_version != ROLLING_EWMA_STATE_LAYOUT_VERSION
    {
        return Err(invalid_argument(
            "rolling.state_layout_version",
            "EWMA outputs require rolling state layout version 2",
        ));
    }
    validate_key_names("rolling.partition_by", &spec.partition_by)?;
    validate_key_names("rolling.sequence_by", &spec.sequence_by)?;
    validate_outputs(&spec.outputs)?;
    super::late_output::validate_policy(spec.late_policy, "rolling")
}

fn validate_key_names(field: &str, columns: &[String]) -> Result<()> {
    if columns.is_empty() {
        return Err(invalid_argument(field, "must not be empty"));
    }
    for (index, column) in columns.iter().enumerate() {
        let indexed = format!("{field}[{index}]");
        if column.is_empty() {
            return Err(invalid_argument(&indexed, "must not be empty"));
        }
        if columns[..index].contains(column) {
            return Err(invalid_argument(
                &indexed,
                "duplicates an earlier key column",
            ));
        }
    }
    Ok(())
}

fn validate_outputs(outputs: &[RollingOutputSpec]) -> Result<()> {
    if outputs.is_empty() {
        return Err(invalid_argument("rolling.outputs", "must not be empty"));
    }
    for (index, output) in outputs.iter().enumerate() {
        let base = format!("rolling.outputs[{index}]");
        if output.primitive_version() != 1 {
            return Err(invalid_argument(
                &format!("{base}.primitive_version"),
                "unsupported rolling primitive version",
            ));
        }
        if let RollingOutputSpec::Difference { left, right, .. } = output {
            validate_float_primitive(&format!("{base}.left"), left)?;
            validate_float_primitive(&format!("{base}.right"), right)?;
        }
        if output.span().is_some_and(|span| span == 0) {
            return Err(invalid_argument(
                &format!("{base}.span"),
                "must be greater than zero",
            ));
        }
        if let Some(frame) = output.frame() {
            let zero = match frame {
                RollingFrameSpec::Rows { size } => size == 0,
                RollingFrameSpec::Duration { micros } => micros == 0,
            };
            if zero {
                let field = match frame {
                    RollingFrameSpec::Rows { .. } => format!("{base}.frame.size"),
                    RollingFrameSpec::Duration { .. } => format!("{base}.frame.micros"),
                };
                return Err(invalid_argument(&field, "must be greater than zero"));
            }
        } else if output.span().is_none()
            && output.retained_rows() == 0
            && !matches!(
                output,
                RollingOutputSpec::Difference { .. } | RollingOutputSpec::CumulativeMean { .. }
            )
        {
            return Err(invalid_argument(
                &format!("{base}.periods"),
                "must be greater than zero",
            ));
        }
        if let Some(min_periods) = output.min_periods() {
            if min_periods == 0 {
                return Err(invalid_argument(
                    &format!("{base}.min_periods"),
                    "must be greater than zero",
                ));
            }
            // Only row-count frames cap min_periods at their size; a duration
            // frame has no row-count ceiling (SCE-00 D5).
            if matches!(output.frame(), Some(RollingFrameSpec::Rows { .. }))
                && min_periods > output.retained_rows()
            {
                return Err(invalid_argument(
                    &format!("{base}.min_periods"),
                    "must not exceed the row-frame size",
                ));
            }
        }
        if let Some(ddof) = output.ddof()
            && ddof > 1
        {
            return Err(invalid_argument(&format!("{base}.ddof"), "must be 0 or 1"));
        }
        if output.input().is_empty() {
            return Err(invalid_argument(
                &format!("{base}.input"),
                "must not be empty",
            ));
        }
        if let Some(right) = output.pair_right()
            && right.is_empty()
        {
            return Err(invalid_argument(
                &format!("{base}.right"),
                "must not be empty",
            ));
        }
        if output.output().is_empty() {
            return Err(invalid_argument(
                &format!("{base}.output"),
                "must not be empty",
            ));
        }
        if outputs[..index]
            .iter()
            .any(|earlier| earlier.output() == output.output())
        {
            return Err(invalid_argument(
                &format!("{base}.output"),
                "duplicates an earlier rolling output",
            ));
        }
    }
    Ok(())
}

fn validate_float_primitive(base: &str, primitive: &RollingFloatPrimitiveSpec) -> Result<()> {
    if primitive.primitive_version() != 1 {
        return Err(invalid_argument(
            &format!("{base}.primitive_version"),
            "unsupported rolling primitive version",
        ));
    }
    if primitive.input().is_empty() {
        return Err(invalid_argument(
            &format!("{base}.input"),
            "must not be empty",
        ));
    }
    match primitive {
        RollingFloatPrimitiveSpec::Ewma {
            span, min_periods, ..
        } => {
            if *span == 0 {
                return Err(invalid_argument(
                    &format!("{base}.span"),
                    "must be greater than zero",
                ));
            }
            validate_positive_min_periods(base, *min_periods, None)
        }
        RollingFloatPrimitiveSpec::Mean {
            frame, min_periods, ..
        } => validate_positive_min_periods(base, *min_periods, Some(*frame)),
        RollingFloatPrimitiveSpec::Variance {
            frame,
            min_periods,
            ddof,
            ..
        }
        | RollingFloatPrimitiveSpec::Stddev {
            frame,
            min_periods,
            ddof,
            ..
        } => {
            if *ddof > 1 {
                return Err(invalid_argument(&format!("{base}.ddof"), "must be 0 or 1"));
            }
            validate_positive_min_periods(base, *min_periods, Some(*frame))
        }
    }
}

fn validate_positive_min_periods(
    base: &str,
    min_periods: u64,
    frame: Option<RollingFrameSpec>,
) -> Result<()> {
    if min_periods == 0 {
        return Err(invalid_argument(
            &format!("{base}.min_periods"),
            "must be greater than zero",
        ));
    }
    if let Some(RollingFrameSpec::Rows { size }) = frame {
        if size == 0 {
            return Err(invalid_argument(
                &format!("{base}.frame.size"),
                "must be greater than zero",
            ));
        }
        if min_periods > size {
            return Err(invalid_argument(
                &format!("{base}.min_periods"),
                "must not exceed the row-frame size",
            ));
        }
    } else if matches!(frame, Some(RollingFrameSpec::Duration { micros: 0 })) {
        return Err(invalid_argument(
            &format!("{base}.frame.micros"),
            "must be greater than zero",
        ));
    }
    Ok(())
}

fn compile_spec(spec: &RollingSpec, input_schema: &Schema) -> Result<CompiledRollingSpec> {
    compile_spec_against_schema(spec, input_schema, String::new())
}

fn compile_spec_full(
    spec: &RollingSpec,
    input_schema: &Schema,
    configuration: &JsonMap,
) -> Result<CompiledRollingSpec> {
    let canonical = canonical_json(&Value::Object(configuration.clone().into_iter().collect()))?;
    let configuration_hash = hex::encode(Sha256::digest(canonical.as_bytes()));
    compile_spec_against_schema(spec, input_schema, configuration_hash)
}

fn compile_spec_against_schema(
    spec: &RollingSpec,
    input_schema: &Schema,
    configuration_hash: String,
) -> Result<CompiledRollingSpec> {
    super::late_output::validate_input(spec.late_policy, input_schema, "rolling")?;
    let event_time_index = exact_field_index(input_schema, &spec.event_time)?;
    validate_event_time(input_schema, event_time_index, &spec.event_time)?;
    let partition_columns = spec
        .partition_by
        .iter()
        .map(|column| compile_key_column(input_schema, column, KeyRole::Partition))
        .collect::<Result<Vec<_>>>()?;
    let sequence_columns = spec
        .sequence_by
        .iter()
        .map(|column| compile_key_column(input_schema, column, KeyRole::Sequence))
        .collect::<Result<Vec<_>>>()?;
    let mut window_groups = Vec::new();
    let outputs = spec
        .outputs
        .iter()
        .enumerate()
        .map(|(ordinal, output)| compile_output(input_schema, output, ordinal, &mut window_groups))
        .collect::<Result<Vec<_>>>()?;
    let max_row_retention = spec
        .outputs
        .iter()
        .map(RollingOutputSpec::retained_rows)
        .max()
        .unwrap_or(1);
    let max_duration_micros = spec
        .outputs
        .iter()
        .filter_map(RollingOutputSpec::retained_micros)
        .max();
    let kernel_plan = RollingKernelPlan::compile(
        input_schema,
        ROLLING_COLUMNAR_STATE_LAYOUT_VERSION,
        spec.numerical_profile,
        event_time_index,
        &partition_columns,
        &sequence_columns,
        &outputs,
        &window_groups,
    );
    Ok(CompiledRollingSpec {
        state_layout_version: ROLLING_COLUMNAR_STATE_LAYOUT_VERSION,
        legacy_state_layout_version: spec.state_layout_version,
        event_time_index,
        partition_columns,
        sequence_columns,
        outputs,
        window_groups,
        kernel_plan,
        max_row_retention,
        max_duration_micros,
        configuration_hash,
        state_schema_fingerprint: state_schema_fingerprint(
            input_schema,
            ROLLING_COLUMNAR_STATE_LAYOUT_VERSION,
        ),
        legacy_state_schema_fingerprint: state_schema_fingerprint(
            input_schema,
            spec.state_layout_version,
        ),
    })
}

#[derive(Clone, Copy)]
enum KeyRole {
    Partition,
    Sequence,
}

fn compile_key_column(
    input_schema: &Schema,
    column: &str,
    role: KeyRole,
) -> Result<CompiledKeyColumn> {
    let index = exact_field_index(input_schema, column)?;
    let field = input_schema.field(index);
    let data_type = field.data_type().clone();
    match role {
        KeyRole::Partition => {
            if !supports_total_order(&data_type) {
                return Err(compile_error(format!(
                    "rolling partition column {column:?} has unsupported type {data_type}"
                )));
            }
        }
        KeyRole::Sequence => {
            if field.is_nullable() {
                return Err(compile_error(format!(
                    "rolling sequence column {column:?} must be non-nullable"
                )));
            }
            if matches!(data_type, DataType::Float32 | DataType::Float64) {
                return Err(compile_error(format!(
                    "rolling sequence column {column:?} must not use a floating type"
                )));
            }
            if !supports_total_order(&data_type) {
                return Err(compile_error(format!(
                    "rolling sequence column {column:?} has unsupported type {data_type}"
                )));
            }
        }
    }
    Ok(CompiledKeyColumn { index })
}

fn compile_output(
    input_schema: &Schema,
    output: &RollingOutputSpec,
    ordinal: usize,
    window_groups: &mut Vec<CompiledWindowGroup>,
) -> Result<CompiledRollingOutput> {
    if input_schema
        .fields()
        .iter()
        .any(|field| field.name() == output.output())
    {
        return Err(invalid_argument(
            &format!("rolling.outputs[{ordinal}].output"),
            "collides with an input field name",
        ));
    }
    let input_index = exact_field_index(input_schema, output.input())?;
    let input_type = input_schema.field(input_index).data_type().clone();
    let evaluation = match output {
        RollingOutputSpec::Lag { periods, .. } => CompiledEvaluation::Lag { periods: *periods },
        RollingOutputSpec::Delta { periods, .. } => {
            require_numeric(output.input(), &input_type, "delta")?;
            CompiledEvaluation::Delta { periods: *periods }
        }
        RollingOutputSpec::Ewma {
            span, min_periods, ..
        } => {
            require_numeric(output.input(), &input_type, "ewma")?;
            let group = compile_ewma_group(input_index, *span, window_groups);
            CompiledEvaluation::Ewma(CompiledEwma {
                group,
                min_periods: *min_periods,
            })
        }
        RollingOutputSpec::CumulativeMean { min_periods, .. } => {
            require_numeric(output.input(), &input_type, "cumulative_mean")?;
            let group = compile_ewma_group(input_index, 0, window_groups);
            CompiledEvaluation::Ewma(CompiledEwma {
                group,
                min_periods: *min_periods,
            })
        }
        RollingOutputSpec::Covariance {
            left,
            right,
            frame,
            min_periods,
            ddof,
            ..
        }
        | RollingOutputSpec::Correlation {
            left,
            right,
            frame,
            min_periods,
            ddof,
            ..
        } => {
            let correlation = matches!(output, RollingOutputSpec::Correlation { .. });
            let left_index = exact_field_index(input_schema, left)?;
            let right_index = exact_field_index(input_schema, right)?;
            let left_type = input_schema.field(left_index).data_type().clone();
            let right_type = input_schema.field(right_index).data_type().clone();
            require_numeric(left, &left_type, "covariance")?;
            require_numeric(right, &right_type, "covariance")?;
            let group = compile_pair_group(left_index, right_index, *frame, window_groups);
            CompiledEvaluation::Pair(CompiledPairAggregate {
                group,
                correlation,
                min_periods: *min_periods,
                ddof: *ddof,
            })
        }
        RollingOutputSpec::Difference { left, right, .. } => {
            CompiledEvaluation::Difference(CompiledDifference {
                left: compile_float_readout(input_schema, left, window_groups)?,
                right: compile_float_readout(input_schema, right, window_groups)?,
            })
        }
        RollingOutputSpec::Argmax { .. }
        | RollingOutputSpec::Argmin { .. }
        | RollingOutputSpec::Rank { .. }
        | RollingOutputSpec::Quantile { .. }
        | RollingOutputSpec::UniqueCount { .. }
        | RollingOutputSpec::Decay { .. } => compile_scan_output(output, &input_type)?,
        aggregate => compile_aggregate_output(aggregate, input_index, &input_type, window_groups)?,
    };
    let output_type = compiled_output_type(&evaluation, &input_type);
    Ok(CompiledRollingOutput {
        input_index,
        name: output.output().to_owned(),
        output_type,
        input_type,
        evaluation,
    })
}

fn compiled_output_type(evaluation: &CompiledEvaluation, input_type: &DataType) -> DataType {
    match evaluation {
        CompiledEvaluation::Lag { .. } | CompiledEvaluation::Delta { .. } => input_type.clone(),
        CompiledEvaluation::Ewma(_)
        | CompiledEvaluation::Pair(_)
        | CompiledEvaluation::Difference(_) => DataType::Float64,
        CompiledEvaluation::Scan(scan) => match scan.kind {
            ScanKind::Argmax | ScanKind::Argmin | ScanKind::Rank | ScanKind::UniqueCount => {
                DataType::UInt64
            }
            ScanKind::Quantile | ScanKind::Decay => DataType::Float64,
        },
        CompiledEvaluation::Aggregate(aggregate) => match aggregate.statistic {
            Statistic::Count => DataType::UInt64,
            Statistic::Sum => match SumClass::from_input(input_type) {
                SumClass::Signed => DataType::Int64,
                SumClass::Unsigned => DataType::UInt64,
                _ => DataType::Float64,
            },
            Statistic::Mean | Statistic::Variance | Statistic::Stddev => DataType::Float64,
            // Min/max preserve the input type (SCE-00 D3, contract
            // section 5.2).
            Statistic::Min | Statistic::Max => input_type.clone(),
        },
    }
}

fn compile_scan_output(
    output: &RollingOutputSpec,
    input_type: &DataType,
) -> Result<CompiledEvaluation> {
    let (kind, frame, min_periods) = match output {
        RollingOutputSpec::Argmax {
            frame, min_periods, ..
        } => (ScanKind::Argmax, *frame, *min_periods),
        RollingOutputSpec::Argmin {
            frame, min_periods, ..
        } => (ScanKind::Argmin, *frame, *min_periods),
        RollingOutputSpec::Rank {
            frame, min_periods, ..
        } => (ScanKind::Rank, *frame, *min_periods),
        RollingOutputSpec::Quantile {
            frame, min_periods, ..
        } => (ScanKind::Quantile, *frame, *min_periods),
        RollingOutputSpec::UniqueCount {
            frame, min_periods, ..
        } => (ScanKind::UniqueCount, *frame, *min_periods),
        RollingOutputSpec::Decay {
            frame, min_periods, ..
        } => (ScanKind::Decay, *frame, *min_periods),
        _ => return Err(internal_error("non-scan output reached scan compiler")),
    };
    if matches!(kind, ScanKind::UniqueCount) {
        if !supports_total_order(input_type) {
            return Err(compile_error(format!(
                "rolling unique_count input {:?} has unsupported type {input_type}",
                output.input()
            )));
        }
    } else {
        require_numeric(output.input(), input_type, "rolling scan")?;
    }
    Ok(CompiledEvaluation::Scan(CompiledScan {
        kind,
        frame: compiled_frame(frame),
        min_periods,
    }))
}

fn compile_float_readout(
    input_schema: &Schema,
    primitive: &RollingFloatPrimitiveSpec,
    window_groups: &mut Vec<CompiledWindowGroup>,
) -> Result<CompiledFloatReadout> {
    let input_index = exact_field_index(input_schema, primitive.input())?;
    let input_type = input_schema.field(input_index).data_type();
    require_numeric(primitive.input(), input_type, "fused difference")?;
    match primitive {
        RollingFloatPrimitiveSpec::Ewma {
            span, min_periods, ..
        } => Ok(CompiledFloatReadout::Ewma(CompiledEwma {
            group: compile_ewma_group(input_index, *span, window_groups),
            min_periods: *min_periods,
        })),
        RollingFloatPrimitiveSpec::Mean {
            frame, min_periods, ..
        } => compile_float_aggregate_readout(
            input_index,
            input_type,
            *frame,
            *min_periods,
            0,
            Statistic::Mean,
            window_groups,
        ),
        RollingFloatPrimitiveSpec::Variance {
            frame,
            min_periods,
            ddof,
            ..
        } => compile_float_aggregate_readout(
            input_index,
            input_type,
            *frame,
            *min_periods,
            *ddof,
            Statistic::Variance,
            window_groups,
        ),
        RollingFloatPrimitiveSpec::Stddev {
            frame,
            min_periods,
            ddof,
            ..
        } => compile_float_aggregate_readout(
            input_index,
            input_type,
            *frame,
            *min_periods,
            *ddof,
            Statistic::Stddev,
            window_groups,
        ),
    }
}

fn compile_float_aggregate_readout(
    input_index: usize,
    input_type: &DataType,
    frame: RollingFrameSpec,
    min_periods: u64,
    ddof: u8,
    statistic: Statistic,
    window_groups: &mut Vec<CompiledWindowGroup>,
) -> Result<CompiledFloatReadout> {
    let CompiledEvaluation::Aggregate(aggregate) = compile_aggregate(
        input_index,
        input_type,
        frame,
        min_periods,
        ddof,
        statistic,
        window_groups,
    ) else {
        return Err(internal_error(
            "fused float aggregate did not compile as an aggregate",
        ));
    };
    Ok(CompiledFloatReadout::Aggregate(aggregate))
}

#[allow(
    clippy::cast_precision_loss,
    reason = "the frozen EWMA recurrence uses IEEE binary64"
)]
fn compile_ewma_group(
    input_index: usize,
    span: u64,
    window_groups: &mut Vec<CompiledWindowGroup>,
) -> usize {
    window_groups
        .iter()
        .position(|group| {
            matches!(
                group,
                CompiledWindowGroup::Ewma {
                    input_index: existing_input,
                    span: existing_span,
                    ..
                } if *existing_input == input_index && *existing_span == span
            )
        })
        .unwrap_or_else(|| {
            window_groups.push(CompiledWindowGroup::Ewma {
                input_index,
                span,
                alpha: if span == 0 {
                    0.0
                } else {
                    2.0 / (span as f64 + 1.0)
                },
            });
            window_groups.len() - 1
        })
}

fn compile_pair_group(
    left_index: usize,
    right_index: usize,
    frame: RollingFrameSpec,
    window_groups: &mut Vec<CompiledWindowGroup>,
) -> usize {
    let frame = compiled_frame(frame);
    window_groups
        .iter()
        .position(|group| match group {
            CompiledWindowGroup::Pair {
                left_index: existing_left,
                right_index: existing_right,
                frame: existing_frame,
            } => {
                *existing_left == left_index
                    && *existing_right == right_index
                    && *existing_frame == frame
            }
            _ => false,
        })
        .unwrap_or_else(|| {
            window_groups.push(CompiledWindowGroup::Pair {
                left_index,
                right_index,
                frame,
            });
            window_groups.len() - 1
        })
}

fn compiled_frame(frame: RollingFrameSpec) -> CompiledFrame {
    match frame {
        RollingFrameSpec::Rows { size } => CompiledFrame::Rows(size),
        RollingFrameSpec::Duration { micros } => CompiledFrame::Duration(micros),
    }
}

fn compile_aggregate_output(
    output: &RollingOutputSpec,
    input_index: usize,
    input_type: &DataType,
    window_groups: &mut Vec<CompiledWindowGroup>,
) -> Result<CompiledEvaluation> {
    let (frame, min_periods, ddof, statistic) = match output {
        RollingOutputSpec::Count {
            frame, min_periods, ..
        } => (*frame, *min_periods, 0, Statistic::Count),
        RollingOutputSpec::Sum {
            frame, min_periods, ..
        } => (*frame, *min_periods, 0, Statistic::Sum),
        RollingOutputSpec::Mean {
            frame, min_periods, ..
        } => (*frame, *min_periods, 0, Statistic::Mean),
        RollingOutputSpec::Variance {
            frame,
            min_periods,
            ddof,
            ..
        } => (*frame, *min_periods, *ddof, Statistic::Variance),
        RollingOutputSpec::Stddev {
            frame,
            min_periods,
            ddof,
            ..
        } => (*frame, *min_periods, *ddof, Statistic::Stddev),
        RollingOutputSpec::Min {
            frame, min_periods, ..
        } => (*frame, *min_periods, 0, Statistic::Min),
        RollingOutputSpec::Max {
            frame, min_periods, ..
        } => (*frame, *min_periods, 0, Statistic::Max),
        RollingOutputSpec::Lag { .. }
        | RollingOutputSpec::Delta { .. }
        | RollingOutputSpec::Ewma { .. }
        | RollingOutputSpec::CumulativeMean { .. }
        | RollingOutputSpec::Covariance { .. }
        | RollingOutputSpec::Correlation { .. }
        | RollingOutputSpec::Argmax { .. }
        | RollingOutputSpec::Argmin { .. }
        | RollingOutputSpec::Rank { .. }
        | RollingOutputSpec::Quantile { .. }
        | RollingOutputSpec::UniqueCount { .. }
        | RollingOutputSpec::Decay { .. }
        | RollingOutputSpec::Difference { .. } => {
            unreachable!("lag, delta, and pair outputs compile before aggregates")
        }
    };
    if !matches!(
        statistic,
        Statistic::Count | Statistic::Min | Statistic::Max
    ) {
        require_numeric(output.input(), input_type, statistic.name())?;
    } else if matches!(statistic, Statistic::Min | Statistic::Max)
        && !supports_total_order(input_type)
    {
        return Err(compile_error(format!(
            "rolling {} does not support column {:?} with type {input_type}",
            statistic.name(),
            output.input()
        )));
    }
    Ok(compile_aggregate(
        input_index,
        input_type,
        frame,
        min_periods,
        ddof,
        statistic,
        window_groups,
    ))
}

fn require_numeric(column: &str, input_type: &DataType, primitive: &str) -> Result<()> {
    if !is_numeric(input_type) {
        return Err(compile_error(format!(
            "rolling {primitive} does not support column {column:?} with type {input_type}"
        )));
    }
    Ok(())
}

fn compile_aggregate(
    input_index: usize,
    input_type: &DataType,
    frame: RollingFrameSpec,
    min_periods: u64,
    ddof: u8,
    statistic: Statistic,
    window_groups: &mut Vec<CompiledWindowGroup>,
) -> CompiledEvaluation {
    let frame = compiled_frame(frame);
    let group = if matches!(statistic, Statistic::Min | Statistic::Max) {
        let descending = matches!(statistic, Statistic::Max);
        window_groups
            .iter()
            .position(|group| match group {
                CompiledWindowGroup::Extrema {
                    input_index: existing_input,
                    frame: existing_frame,
                    descending: existing_descending,
                } => {
                    *existing_input == input_index
                        && *existing_frame == frame
                        && *existing_descending == descending
                }
                _ => false,
            })
            .unwrap_or_else(|| {
                window_groups.push(CompiledWindowGroup::Extrema {
                    input_index,
                    frame,
                    descending,
                });
                window_groups.len() - 1
            })
    } else {
        window_groups
            .iter()
            .position(|group| match group {
                CompiledWindowGroup::Numeric {
                    input_index: existing_input,
                    frame: existing_frame,
                    ..
                } => *existing_input == input_index && *existing_frame == frame,
                _ => false,
            })
            .unwrap_or_else(|| {
                window_groups.push(CompiledWindowGroup::Numeric {
                    input_index,
                    frame,
                    sum_class: SumClass::from_input(input_type),
                });
                window_groups.len() - 1
            })
    };
    CompiledEvaluation::Aggregate(CompiledAggregate {
        group,
        statistic,
        min_periods,
        ddof,
    })
}

fn exact_field_index(schema: &Schema, column: &str) -> Result<usize> {
    let matches = schema
        .fields()
        .iter()
        .enumerate()
        .filter(|(_, field)| field.name() == column)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [index] => Ok(*index),
        [] => Err(compile_error(format!(
            "rolling column {column:?} does not exist in the input schema"
        ))),
        _ => Err(compile_error(format!(
            "rolling column {column:?} is ambiguous in the input schema"
        ))),
    }
}

fn validate_event_time(schema: &Schema, index: usize, column: &str) -> Result<()> {
    let field = schema.field(index);
    if field.is_nullable() {
        return Err(compile_error(format!(
            "rolling event-time column {column:?} must be non-nullable"
        )));
    }
    if !matches!(
        field.data_type(),
        DataType::Timestamp(TimeUnit::Microsecond, Some(timezone)) if timezone.as_ref() == "UTC"
    ) {
        return Err(compile_error(format!(
            "rolling event-time column {column:?} must be a non-null UTC timestamp[us], found {}",
            field.data_type()
        )));
    }
    Ok(())
}

fn supports_total_order(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
            | DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Date32
            | DataType::Date64
    ) || matches!(
        data_type,
        DataType::Timestamp(TimeUnit::Microsecond, timezone)
            if timezone.as_deref().is_none_or(|timezone| timezone == "UTC")
    )
}

fn is_numeric(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
    )
}

fn output_schema(input_schema: &Schema, outputs: &[CompiledRollingOutput]) -> Schema {
    let mut fields = input_schema.fields().to_vec();
    fields.extend(
        outputs
            .iter()
            .map(|output| Field::new(&output.name, output.output_type.clone(), true).into()),
    );
    Schema::new(fields)
}

fn configuration(spec: &RollingSpec) -> Result<JsonMap> {
    let spec_json = serde_json::to_value(spec).map_err(|error| format_error(&error))?;
    Ok(JsonMap::from([
        ("kind".into(), json!("rolling")),
        ("spec".into(), spec_json),
    ]))
}

fn invalid_argument(field: &str, message: &str) -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: field.into(),
        message: message.into(),
    }
}

fn operator_error(node_id: &str, message: &str) -> CalcFlowError {
    CalcFlowError::Operator {
        node_id: node_id.into(),
        message: message.into(),
    }
}

fn format_error(error: &serde_json::Error) -> CalcFlowError {
    CalcFlowError::Format {
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests;
