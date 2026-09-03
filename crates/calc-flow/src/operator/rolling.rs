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

use super::{
    BatchOperator, BatchOperatorContext, LateMetricDelta, OperatorMetadata, StreamCollector,
    StreamOperator, StreamOperatorContext, accumulate_late_metrics, expression::required_input,
    validate_operator_name,
};

mod generated_kernel_manifest;
mod kernel;
mod state_v3;

#[cfg(test)]
use kernel::KernelSelection;
use kernel::{RollingKernelPlan, RollingKernelState};

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

    const fn is_stable_v1(&self) -> bool {
        matches!(self, Self::StableV1)
    }
}

/// One SQL AVG window accepted by the crate-private `DataFusion` rolling
/// physical planner.
#[derive(Clone, Debug)]
pub(crate) struct DataFusionRollingWindow {
    pub input_index: usize,
    pub output_name: String,
    pub rows: u64,
}

/// Immutable typed rolling plan shared with `CalcFlowRollingExec`.
#[derive(Clone, Debug)]
pub(crate) struct DataFusionRollingKernel {
    plan: RollingKernelPlan,
}

/// Per-partition transition state owned by one `DataFusion` execution stream.
#[derive(Clone, Debug, Default)]
pub(crate) struct DataFusionRollingState {
    inner: RollingKernelState,
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
        if windows.is_empty() || !kernel::supports_datafusion_primitive("mean") {
            return None;
        }
        let (&event_time_index, sequence_indices) = order_indices.split_first()?;
        if input_schema.field(event_time_index).data_type()
            != &DataType::Timestamp(TimeUnit::Microsecond, None)
            || order_indices
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
        );
        plan.supports_typed_transition().then_some(Self { plan })
    }

    pub(crate) fn update_and_fill(
        &self,
        state: &DataFusionRollingState,
        input: &RecordBatch,
    ) -> Result<DataFusionRollingBatch> {
        let execution = self
            .plan
            .update_and_fill(&state.inner, input, "datafusion.rolling")?
            .ok_or_else(|| internal_error("DataFusion input violated the planned rolling order"))?;
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
    /// Drop each late row and record the three D7 metrics.
    Drop {
        /// Metric transaction version; must equal `1`.
        metrics_version: u32,
    },
}

/// Frozen null/NaN policy for rolling values (SCE-00 D3.2).
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
    /// Valid (non-null, non-NaN) sample count over the frame (SCE-00 D3.2).
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
    /// Checked sum over the frame; integer results stay exact (SCE-00 D3.2).
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
    /// D3.2).
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
    /// D3.2).
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
    /// Float64 covariance of two columns over the frame, counting only
    /// pairwise-valid positions (SCE-00 D3.2/D5).
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
    /// either side has zero variance (SCE-00 D3.2).
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
            | Self::Count { input, .. }
            | Self::Sum { input, .. }
            | Self::Mean { input, .. }
            | Self::Variance { input, .. }
            | Self::Stddev { input, .. }
            | Self::Min { input, .. }
            | Self::Max { input, .. } => input,
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
            | Self::Count { output, .. }
            | Self::Sum { output, .. }
            | Self::Mean { output, .. }
            | Self::Variance { output, .. }
            | Self::Stddev { output, .. }
            | Self::Min { output, .. }
            | Self::Max { output, .. }
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
            Self::Ewma { .. } => 0,
            Self::Count { frame, .. }
            | Self::Sum { frame, .. }
            | Self::Mean { frame, .. }
            | Self::Variance { frame, .. }
            | Self::Stddev { frame, .. }
            | Self::Min { frame, .. }
            | Self::Max { frame, .. }
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
            Self::Lag { .. } | Self::Delta { .. } | Self::Ewma { .. } => None,
            Self::Count { frame, .. }
            | Self::Sum { frame, .. }
            | Self::Mean { frame, .. }
            | Self::Variance { frame, .. }
            | Self::Stddev { frame, .. }
            | Self::Min { frame, .. }
            | Self::Max { frame, .. }
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
            Self::Lag { .. } | Self::Delta { .. } | Self::Ewma { .. } | Self::Difference { .. } => {
                None
            }
            Self::Count { frame, .. }
            | Self::Sum { frame, .. }
            | Self::Mean { frame, .. }
            | Self::Variance { frame, .. }
            | Self::Stddev { frame, .. }
            | Self::Min { frame, .. }
            | Self::Max { frame, .. }
            | Self::Covariance { frame, .. }
            | Self::Correlation { frame, .. } => Some(*frame),
        }
    }

    const fn min_periods(&self) -> Option<u64> {
        match self {
            Self::Lag { .. } | Self::Delta { .. } | Self::Difference { .. } => None,
            Self::Ewma { min_periods, .. }
            | Self::Count { min_periods, .. }
            | Self::Sum { min_periods, .. }
            | Self::Mean { min_periods, .. }
            | Self::Variance { min_periods, .. }
            | Self::Stddev { min_periods, .. }
            | Self::Min { min_periods, .. }
            | Self::Max { min_periods, .. }
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
            Self::Ewma { .. } => true,
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
    output_ports: [Port; 1],
    compiled: Box<CompiledRollingSpec>,
    state: RollingStreamState,
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
        Ok(Self {
            name: name.into(),
            spec,
            input_ports: [Port::with_schema_ref(
                "input",
                BatchKind::Table,
                true,
                Some(input_schema),
            )?],
            output_ports: [Port::with_schema_ref(
                "output",
                BatchKind::Table,
                true,
                Some(output_schema),
            )?],
            compiled,
            state: RollingStreamState::default(),
        })
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
    histories: RollingHistories,
    last_input_watermark: Option<EventTime>,
    next_output_sequence: u64,
    ended: bool,
    metrics: LateMetricDelta,
    pipeline_fingerprint: Option<String>,
    operator_id: Option<String>,
    last_checkpoint_epoch: Option<Epoch>,
    typed_kernel_state: Option<Box<RollingKernelState>>,
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
}

#[derive(Default)]
struct PreparedLateMetrics {
    late_rows: u64,
    max_lateness_micros: Option<u64>,
}

impl PreparedLateMetrics {
    fn into_delta(self) -> LateMetricDelta {
        LateMetricDelta {
            late_rows: self.late_rows,
            affected_batches: u64::from(self.late_rows > 0),
            max_lateness_micros: self.max_lateness_micros,
            ..LateMetricDelta::default()
        }
    }
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
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
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
        let rows = read_buffered_rows(batch.table_payload()?, &self.compiled, &self.name)?;
        let (accepted, metrics) = self.classify_envelope(rows, watermark, context.operator_id())?;
        let next_metrics = accumulate_late_metrics(self.state.metrics, metrics)?;
        for (identity, row) in accepted {
            self.state.buffer.insert(identity, row);
        }
        context.record_window_metrics(
            metrics.late_rows,
            metrics.max_lateness_micros,
            metrics.null_event_time_rows,
        )?;
        self.state.metrics = next_metrics;
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
        let closing = self.closing_keys(watermark.as_micros(), context.operator_id())?;
        let rows = self.take_buffered(&closing);
        self.emit_rows(rows, context, output).await?;
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
        let rows = self.take_all_buffered();
        self.emit_rows(rows, context, output).await?;
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
                "rolling checkpoint epoch did not advance strictly".into(),
            ));
        }
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
        let restored = self.decode_state(&metadata, snapshot)?;
        self.state = RollingStreamState {
            buffer: restored.buffer,
            histories: restored.histories,
            last_input_watermark: metadata.last_input_watermark,
            next_output_sequence: metadata.next_output_sequence,
            ended: metadata.ended,
            metrics: metadata.metrics,
            pipeline_fingerprint: metadata.pipeline_fingerprint,
            operator_id: metadata.operator_id,
            last_checkpoint_epoch: Some(metadata.epoch),
            typed_kernel_state: None,
        };
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.state = RollingStreamState::default();
        Ok(())
    }
}

impl RollingOperator {
    fn observe_context(&self, context: &StreamOperatorContext<'_>) -> Result<()> {
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
    ) -> Result<(BTreeMap<RowIdentity, BufferedRow>, LateMetricDelta)> {
        let mut accepted = BTreeMap::new();
        let mut metrics = PreparedLateMetrics::default();
        for (row_index, row) in rows.into_iter().enumerate() {
            if self.is_late(row.identity.event_time, watermark, row_index, node_id)? {
                record_late_row(&mut metrics, watermark, row.identity.event_time, node_id)?;
                continue;
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
            LatePolicySpec::Drop { .. } => Ok(true),
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
        )?;
        let (record, next_kernel_state, touched) = if let Some(typed) = typed {
            typed
        } else {
            let computed = compute_output_columns(
                &rows,
                &self.state.histories,
                &self.compiled,
                context.operator_id(),
            )?;
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
        for batch in batches {
            output.emit("output", batch).await?;
        }
        self.state.next_output_sequence = self
            .state
            .next_output_sequence
            .checked_add(chunk_count)
            .ok_or_else(|| operator_error(context.operator_id(), "output sequence overflowed"))?;
        self.state.histories.apply(touched);
        self.state.typed_kernel_state = next_kernel_state.map(Box::new);
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
            checkpoint_mismatch("rolling segment is missing its operator identity".into())
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
            "rolling operator configuration hash does not match the compiled operator".into(),
        ));
    }
    if metadata.state_schema_fingerprint != *expected_schema_fingerprint {
        return Err(checkpoint_mismatch(
            "rolling state schema fingerprint does not match the compiled operator".into(),
        ));
    }
    if metadata.state_layout_version == ROLLING_COLUMNAR_STATE_LAYOUT_VERSION
        && (metadata.kernel_fingerprint.as_deref() != Some(compiled.kernel_plan.fingerprint())
            || metadata.numerical_profile.as_deref()
                != Some(compiled.kernel_plan.numerical_profile()))
    {
        return Err(checkpoint_mismatch(
            "rolling kernel fingerprint or numerical profile does not match the compiled operator"
                .into(),
        ));
    }
    let inventory = StateInventory::new(metadata.segment_inventory.clone())
        .map_err(|error| checkpoint_mismatch(error.to_string()))?;
    for descriptor in inventory.segments() {
        if descriptor.state_layout_version != metadata.state_layout_version
            || descriptor.schema_fingerprint != metadata.state_schema_fingerprint
        {
            return Err(checkpoint_mismatch(
                "rolling segment inventory layout or schema does not match the compiled operator"
                    .into(),
            ));
        }
        if descriptor.handle.epoch() > metadata.epoch {
            return Err(checkpoint_mismatch(
                "rolling segment inventory contains a future epoch".into(),
            ));
        }
        if metadata.operator_id.as_deref() != Some(descriptor.handle.operator_id()) {
            return Err(checkpoint_mismatch(
                "rolling segment inventory operator does not match snapshot metadata".into(),
            ));
        }
    }
    let expected_ids = inventory
        .segments()
        .iter()
        .map(|descriptor| descriptor.handle.segment_id().to_owned())
        .collect::<Vec<_>>();
    let actual_ids = snapshot.segments.keys().cloned().collect::<Vec<_>>();
    if expected_ids != actual_ids {
        return Err(checkpoint_mismatch(
            "rolling snapshot segment IDs are missing, extra, duplicated, or non-canonical".into(),
        ));
    }
    if !snapshot.segments.is_empty()
        && (metadata.pipeline_fingerprint.is_none() || metadata.operator_id.is_none())
    {
        return Err(checkpoint_mismatch(
            "rolling segments require pipeline and operator identity metadata".into(),
        ));
    }
    if let Some(fingerprint) = metadata.pipeline_fingerprint.as_deref()
        && (fingerprint.len() != 64
            || !fingerprint
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)))
    {
        return Err(checkpoint_mismatch(
            "rolling pipeline fingerprint is not lowercase SHA-256".into(),
        ));
    }
    if metadata
        .operator_id
        .as_deref()
        .is_some_and(|operator_id| operator_id.is_empty() || operator_id.contains('\0'))
    {
        return Err(checkpoint_mismatch(
            "rolling operator ID is empty or contains NUL".into(),
        ));
    }
    Ok(inventory)
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
                    "rolling snapshot segment byte length does not match its handle".into(),
                ));
            }
            if hex::encode(Sha256::digest(bytes)) != descriptor.handle.sha256() {
                return Err(checkpoint_mismatch(
                    "rolling snapshot segment checksum does not match its handle".into(),
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

fn record_late_row(
    metrics: &mut PreparedLateMetrics,
    watermark: Option<EventTime>,
    event_time: i64,
    node_id: &str,
) -> Result<()> {
    let Some(watermark) = watermark else {
        return Ok(());
    };
    metrics.late_rows = metrics
        .late_rows
        .checked_add(1)
        .ok_or_else(|| operator_error(node_id, "late row counter overflowed"))?;
    let lateness = u64::try_from(i128::from(watermark.as_micros()) - i128::from(event_time))
        .map_err(|_| operator_error(node_id, "late row distance overflowed"))?;
    metrics.max_lateness_micros = Some(
        metrics
            .max_lateness_micros
            .map_or(lateness, |maximum| maximum.max(lateness)),
    );
    Ok(())
}

// Budget chunking intentionally checks per-row cost, cumulative budget, and
// sequence range in one pass so oversize output fails before enqueue.
// #lizard forgives
fn chunk_output_record(
    record: &RecordBatch,
    operator_id: &str,
    first_sequence: u64,
    budget: crate::EdgeBudget,
) -> Result<Vec<Batch>> {
    let mut batches = Vec::new();
    let mut start = 0_usize;
    let mut sequence = first_sequence;
    while start < record.num_rows() {
        let mut end = start;
        let mut bytes = 0_usize;
        while end < record.num_rows() && end - start < budget.max_rows {
            let row = record.slice(end, 1);
            let row_batch = Batch::table(vec![row], BatchMetadata::default())
                .map_err(|error| operator_error(operator_id, &error.to_string()))?;
            let row_bytes = row_batch
                .estimated_bytes()
                .map_err(|error| operator_error(operator_id, &error.to_string()))?;
            if row_bytes > budget.max_bytes {
                return Err(CalcFlowError::InvalidArgument {
                    field: "message.bytes".into(),
                    message: format!(
                        "one rolling output row requires {row_bytes} bytes, exceeding the effective edge byte budget {}",
                        budget.max_bytes
                    ),
                });
            }
            let Some(candidate) = bytes.checked_add(row_bytes) else {
                break;
            };
            if candidate > budget.max_bytes {
                break;
            }
            bytes = candidate;
            end += 1;
        }
        if end == start {
            return Err(operator_error(
                operator_id,
                "validated rolling output row did not fit the effective edge budget",
            ));
        }
        let metadata = BatchMetadata::new(operator_id, sequence, BTreeMap::new())?;
        batches.push(Batch::table(
            vec![record.slice(start, end - start)],
            metadata,
        )?);
        sequence = sequence.checked_add(1).ok_or_else(|| {
            operator_error(operator_id, "output sequence overflowed before emission")
        })?;
        start = end;
    }
    Ok(batches)
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
        restored_state = reconstruct_typed_state(histories, compiled, &input_schema, node_id)?;
        &restored_state
    };
    let input = build_input_record(rows, input_schema, node_id)?;
    let execution = compiled
        .kernel_plan
        .update_and_fill(prior, &input, node_id)?
        .ok_or_else(|| {
            internal_error("typed rolling stream rows did not satisfy canonical ordering")
        })?;
    let columns = execution.columns;
    let record = build_output_record(rows, columns, output_schema, node_id)?;
    let touched = typed_history_updates(rows, histories, compiled, node_id)?;
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
        .map(|entity| typed_ewma_seeds(entity, compiled))
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
/// every mean/variance accumulate in `f64` (SCE-00 D3.2).
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
    windows: Vec<WindowState>,
    transition_count: u64,
}

impl EntityRollingState {
    fn fresh(compiled: &CompiledRollingSpec) -> Self {
        Self {
            rows: VecDeque::new(),
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
    fn add(&mut self, sample: &ScalarValue, alpha: f64, node_id: &str) -> Result<()> {
        let value = float_sample(sample);
        self.value = if self.valid_count == 0 {
            value
        } else {
            self.value + alpha * (value - self.value)
        };
        self.valid_count = self
            .valid_count
            .checked_add(1)
            .ok_or_else(|| operator_error(node_id, "rolling EWMA sample count overflowed"))?;
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

/// A rolling sample is valid when it is neither null nor NaN (SCE-00 D3.2);
/// infinities stay numeric.
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
    let mut derived: Vec<Vec<Option<ScalarValue>>> = compiled
        .outputs
        .iter()
        .map(|_| vec![None; rows.len()])
        .collect();
    let entities = group_rows_by_entity(rows);
    let mut touched = Vec::with_capacity(entities.len());
    for (entity, indices) in entities {
        let mut entity_state = histories
            .by_entity
            .get(entity)
            .cloned()
            .unwrap_or_else(|| EntityRollingState::fresh(compiled));
        {
            let view = EntityRowView {
                rows,
                indices: &indices,
                history: &entity_state.rows,
                event_time_index: compiled.event_time_index,
            };
            for (position, &row_index) in indices.iter().enumerate() {
                entity_state.transition_count = entity_state
                    .transition_count
                    .checked_add(1)
                    .ok_or_else(|| {
                        operator_error(node_id, "rolling entity transition count overflowed")
                    })?;
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
                    derived[ordinal][row_index] = Some(compute_output_value(
                        &view,
                        position,
                        row_index,
                        output,
                        &entity_state.windows,
                        node_id,
                    )?);
                }
            }
        }
        for &row_index in &indices {
            entity_state.rows.push_back(rows[row_index].values.clone());
        }
        evict_retained_history(&mut entity_state, compiled);
        touched.push((entity.clone(), entity_state));
    }
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

/// Slides every shared window group to the current row: add the current
/// valid sample, remove every sample that left the frame, then repair any
/// non-finite accumulator by re-folding the window so live state and the
/// checkpoint rebuild agree on non-finite classifications (SCE-00 D3.2).
// Add precedes removal so the frozen order matches the rebuild fold exactly.
// #lizard forgives
fn slide_windows(
    view: &EntityRowView<'_>,
    position: usize,
    row_index: usize,
    transition_count: u64,
    compiled: &CompiledRollingSpec,
    windows: &mut [WindowState],
    node_id: &str,
) -> Result<()> {
    let combined = view.history.len() + position;
    for (group_index, group) in compiled.window_groups.iter().enumerate() {
        match group {
            CompiledWindowGroup::Numeric {
                input_index, frame, ..
            } => {
                let current = &view.rows[row_index].values[*input_index];
                let WindowState::Numeric(accumulator) = &mut windows[group_index] else {
                    return Err(internal_error("rolling numeric group has the wrong state"));
                };
                if is_valid_sample(current) {
                    accumulator.add(current, node_id)?;
                }
                expire_numeric(accumulator, view, combined, *input_index, *frame, node_id)?;
                let stable_v2_rebase = spec_uses_stable_v2(compiled)
                    && accumulator.has_float_sum()
                    && kernel::stable_v2_rebase_due(
                        transition_count,
                        window_positions(view, combined, *frame, node_id)?.count(),
                        accumulator.is_non_finite(),
                    );
                if stable_v2_rebase {
                    rebase_numeric_stable_v2(
                        accumulator,
                        view,
                        combined,
                        *input_index,
                        *frame,
                        node_id,
                    )?;
                } else if accumulator.is_non_finite() {
                    refold_numeric(accumulator, view, combined, *input_index, *frame, node_id)?;
                }
            }
            CompiledWindowGroup::Extrema {
                input_index, frame, ..
            } => {
                let WindowState::Extrema(accumulator) = &mut windows[group_index] else {
                    return Err(internal_error("rolling extrema group has the wrong state"));
                };
                let current = &view.rows[row_index].values[*input_index];
                if is_valid_sample(current) {
                    accumulator.add(view.extrema_key(combined, compiled), current.clone());
                }
                expire_extrema(
                    accumulator,
                    view,
                    combined,
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
                let x = &view.rows[row_index].values[*left_index];
                let y = &view.rows[row_index].values[*right_index];
                let WindowState::Pair(accumulator) = &mut windows[group_index] else {
                    return Err(internal_error("rolling pair group has the wrong state"));
                };
                if is_valid_sample(x) && is_valid_sample(y) {
                    accumulator.add(x, y, node_id)?;
                }
                expire_pair(
                    accumulator,
                    view,
                    combined,
                    *left_index,
                    *right_index,
                    *frame,
                    node_id,
                )?;
                let stable_v2_rebase = spec_uses_stable_v2(compiled)
                    && kernel::stable_v2_rebase_due(
                        transition_count,
                        window_positions(view, combined, *frame, node_id)?.count(),
                        accumulator.is_non_finite(),
                    );
                if stable_v2_rebase {
                    rebase_pair_stable_v2(
                        accumulator,
                        view,
                        combined,
                        *left_index,
                        *right_index,
                        *frame,
                        node_id,
                    )?;
                } else if accumulator.is_non_finite() {
                    refold_pair(
                        accumulator,
                        view,
                        combined,
                        *left_index,
                        *right_index,
                        *frame,
                        node_id,
                    )?;
                }
            }
            CompiledWindowGroup::Ewma {
                input_index, alpha, ..
            } => {
                let current = &view.rows[row_index].values[*input_index];
                let WindowState::Ewma(accumulator) = &mut windows[group_index] else {
                    return Err(internal_error("rolling EWMA group has the wrong state"));
                };
                if is_valid_sample(current) {
                    accumulator.add(current, *alpha, node_id)?;
                }
            }
        }
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
/// minimum-period gate uses the valid sample count (SCE-00 D3.2), and the
/// variance divisor is `valid_count - ddof` with a non-positive divisor
/// producing null (SCE-00 D5). Windows holding ±inf samples classify from
/// the reversible infinity counts — both signs is the undefined ∞ − ∞ (NaN),
/// one sign is that infinity, and no infinity keeps the frozen finite-path
/// West readout (SCE-07 defect 1 ruling); variance/stddev over a window with
/// any infinity is NaN because every deviation involves ∞ − ∞.
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
/// accumulator (SCE-00 D3.2/D5): null below the pairwise minimum count or a
/// non-positive divisor, null for correlation with zero variance on either
/// side, NaN when the window holds any infinity, and the West-style
/// co-moment readout otherwise. The ddof divisor cancels in the correlation
/// ratio; it only participates in the divisor gate.
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
    if let LatePolicySpec::Drop { metrics_version } = spec.late_policy
        && metrics_version != 1
    {
        return Err(invalid_argument(
            "rolling.late_policy.metrics_version",
            "unsupported late-metrics version",
        ));
    }
    Ok(())
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
            && !matches!(output, RollingOutputSpec::Difference { .. })
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
        aggregate => compile_aggregate_output(aggregate, input_index, &input_type, window_groups)?,
    };
    let output_type = match &evaluation {
        CompiledEvaluation::Lag { .. } | CompiledEvaluation::Delta { .. } => input_type.clone(),
        CompiledEvaluation::Ewma(_)
        | CompiledEvaluation::Pair(_)
        | CompiledEvaluation::Difference(_) => DataType::Float64,
        CompiledEvaluation::Aggregate(aggregate) => match aggregate.statistic {
            Statistic::Count => DataType::UInt64,
            Statistic::Sum => match SumClass::from_input(&input_type) {
                SumClass::Signed => DataType::Int64,
                SumClass::Unsigned => DataType::UInt64,
                _ => DataType::Float64,
            },
            Statistic::Mean | Statistic::Variance | Statistic::Stddev => DataType::Float64,
            // Min/max preserve the input type (SCE-00 D3.2).
            Statistic::Min | Statistic::Max => input_type.clone(),
        },
    };
    Ok(CompiledRollingOutput {
        input_index,
        name: output.output().to_owned(),
        output_type,
        input_type,
        evaluation,
    })
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
                alpha: 2.0 / (span as f64 + 1.0),
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
        | RollingOutputSpec::Covariance { .. }
        | RollingOutputSpec::Correlation { .. }
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

fn checkpoint_mismatch(message: String) -> CalcFlowError {
    CalcFlowError::CheckpointMismatch { message }
}

fn internal_error(message: &str) -> CalcFlowError {
    CalcFlowError::Internal {
        message: message.into(),
    }
}

fn state_format(message: String) -> CalcFlowError {
    CalcFlowError::Format { message }
}

fn compile_error(message: String) -> CalcFlowError {
    CalcFlowError::Compile { message }
}

fn format_error(error: &serde_json::Error) -> CalcFlowError {
    CalcFlowError::Format {
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use datafusion::arrow::array::Array;
    use datafusion::arrow::datatypes::{DataType, Field, Schema, TimeUnit};
    use serde_json::{Value, json};

    use super::*;
    use crate::{CalcFlowError, OperatorMetadata};

    const TEST_FINGERPRINT: &str =
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn input_schema() -> Schema {
        Schema::new(vec![
            Field::new(
                "ts",
                DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC"))),
                false,
            ),
            Field::new("symbol", DataType::Utf8, false),
            Field::new("sequence", DataType::UInt64, false),
            Field::new("price", DataType::Float64, true),
            Field::new("volume", DataType::Int64, true),
            Field::new("label", DataType::Utf8, true),
        ])
    }

    fn valid_spec_json() -> Value {
        json!({
            "configuration_version": 1,
            "state_layout_version": 1,
            "partition_by": ["symbol"],
            "event_time": "ts",
            "sequence_by": ["sequence"],
            "outputs": [
                {
                    "kind": "lag",
                    "primitive_version": 1,
                    "input": "price",
                    "output": "price_lag_1",
                    "periods": 1
                },
                {
                    "kind": "delta",
                    "primitive_version": 1,
                    "input": "volume",
                    "output": "volume_delta_1",
                    "periods": 1
                }
            ],
            "allowed_lateness_micros": 0,
            "late_policy": {"kind": "error", "scope": "envelope"},
            "value_policy": "stateful_numeric_v1"
        })
    }

    fn valid_spec() -> RollingSpec {
        serde_json::from_value(valid_spec_json()).unwrap()
    }

    fn with_field(schema: &Schema, index: usize, replacement: &Field) -> Schema {
        let fields = schema
            .fields()
            .iter()
            .enumerate()
            .map(|(position, field)| {
                if position == index {
                    replacement.clone().into()
                } else {
                    field.clone()
                }
            })
            .collect::<Vec<_>>();
        Schema::new(fields)
    }

    // ------------------------------------------------------------------
    // Strict serialized model
    // ------------------------------------------------------------------

    #[test]
    fn canonical_lag_delta_spec_round_trips_the_frozen_json() {
        let spec: RollingSpec = serde_json::from_value(valid_spec_json()).unwrap();
        assert_eq!(spec.numerical_profile, RollingNumericalProfile::StableV1);
        assert_eq!(serde_json::to_value(&spec).unwrap(), valid_spec_json());
    }

    #[test]
    fn stable_v2_profile_is_explicit_and_fingerprinted() {
        let mut document = valid_spec_json();
        document["numerical_profile"] = json!("stable_v2");
        let preview: RollingSpec = serde_json::from_value(document.clone()).unwrap();
        let stable = valid_spec();

        assert_eq!(
            preview.numerical_profile,
            RollingNumericalProfile::StableV2Preview
        );
        assert_eq!(serde_json::to_value(&preview).unwrap(), document);
        assert_ne!(
            compile_spec(&stable, &input_schema())
                .unwrap()
                .kernel_plan
                .fingerprint(),
            compile_spec(&preview, &input_schema())
                .unwrap()
                .kernel_plan
                .fingerprint()
        );
    }

    #[test]
    fn drop_late_policy_uses_the_exact_frozen_shape() {
        let mut document = valid_spec_json();
        document["late_policy"] = json!({"kind": "drop", "metrics_version": 1});
        let spec: RollingSpec = serde_json::from_value(document.clone()).unwrap();
        assert_eq!(serde_json::to_value(&spec).unwrap(), document);
    }

    #[test]
    fn unknown_spec_field_is_rejected() {
        let mut document = valid_spec_json();
        document["unexpected"] = json!(true);
        assert!(serde_json::from_value::<RollingSpec>(document).is_err());
    }

    #[test]
    fn missing_semantic_field_is_rejected() {
        let mut document = valid_spec_json();
        document.as_object_mut().unwrap().remove("value_policy");
        assert!(serde_json::from_value::<RollingSpec>(document).is_err());
    }

    #[test]
    fn unsupported_output_kind_is_rejected() {
        for kind in ["std", "median", "skew"] {
            let mut document = valid_spec_json();
            document["outputs"][0] = json!({
                "kind": kind,
                "primitive_version": 1,
                "input": "price",
                "output": "price_unsupported",
                "frame": {"kind": "rows", "size": 20},
                "min_periods": 1
            });
            assert!(
                serde_json::from_value::<RollingSpec>(document).is_err(),
                "unsupported kind {kind} was accepted"
            );
        }
    }

    #[test]
    fn ewma_requires_layout_two_and_has_float64_output() {
        let mut document = valid_spec_json();
        document["state_layout_version"] = json!(2);
        document["outputs"] = json!([{
            "kind": "ewma",
            "primitive_version": 1,
            "input": "volume",
            "output": "volume_ema",
            "span": 3,
            "min_periods": 7
        }]);
        let spec: RollingSpec = serde_json::from_value(document).unwrap();
        let schema = spec.validate(&input_schema()).unwrap();
        assert_eq!(
            schema.field_with_name("volume_ema").unwrap().data_type(),
            &DataType::Float64
        );

        let mut layout_one = serde_json::to_value(spec).unwrap();
        layout_one["state_layout_version"] = json!(1);
        let error = serde_json::from_value::<RollingSpec>(layout_one)
            .unwrap()
            .validate(&input_schema())
            .unwrap_err();
        assert!(
            matches!(
                error,
                CalcFlowError::InvalidArgument { ref field, .. }
                    if field == "rolling.state_layout_version"
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn lag_output_rejects_aggregate_only_fields() {
        for field in ["frame", "min_periods", "ddof"] {
            let mut document = valid_spec_json();
            document["outputs"][0][field] = json!(1);
            assert!(
                serde_json::from_value::<RollingSpec>(document).is_err(),
                "lag accepted aggregate-only field {field}"
            );
        }
    }

    #[test]
    fn unknown_value_policy_is_rejected() {
        let mut document = valid_spec_json();
        document["value_policy"] = json!("lenient");
        assert!(serde_json::from_value::<RollingSpec>(document).is_err());
    }

    #[test]
    fn error_late_policy_rejects_metrics_version_and_drop_rejects_scope() {
        let mut document = valid_spec_json();
        document["late_policy"] =
            json!({"kind": "error", "scope": "envelope", "metrics_version": 1});
        assert!(serde_json::from_value::<RollingSpec>(document).is_err());
        let mut document = valid_spec_json();
        document["late_policy"] =
            json!({"kind": "drop", "metrics_version": 1, "scope": "envelope"});
        assert!(serde_json::from_value::<RollingSpec>(document).is_err());
    }

    // ------------------------------------------------------------------
    // Declaration and schema validation
    // ------------------------------------------------------------------

    #[test]
    fn valid_spec_derives_the_output_schema() {
        let output_schema = valid_spec().validate(&input_schema()).unwrap();
        let expected = Schema::new(vec![
            Field::new(
                "ts",
                DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC"))),
                false,
            ),
            Field::new("symbol", DataType::Utf8, false),
            Field::new("sequence", DataType::UInt64, false),
            Field::new("price", DataType::Float64, true),
            Field::new("volume", DataType::Int64, true),
            Field::new("label", DataType::Utf8, true),
            Field::new("price_lag_1", DataType::Float64, true),
            Field::new("volume_delta_1", DataType::Int64, true),
        ]);
        assert_eq!(output_schema.as_ref(), &expected);
    }

    #[test]
    fn unsupported_configuration_version_is_rejected() {
        let mut spec = valid_spec();
        spec.configuration_version = 2;
        let error = spec.validate(&input_schema()).unwrap_err();
        assert!(
            matches!(
                error,
                CalcFlowError::InvalidArgument { ref field, .. }
                    if field == "rolling.configuration_version"
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn unsupported_state_layout_version_is_rejected() {
        let mut spec = valid_spec();
        spec.state_layout_version = 0;
        let error = spec.validate(&input_schema()).unwrap_err();
        assert!(
            matches!(
                error,
                CalcFlowError::InvalidArgument { ref field, .. }
                    if field == "rolling.state_layout_version"
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn empty_partition_by_is_rejected() {
        let mut spec = valid_spec();
        spec.partition_by = Vec::new();
        let error = spec.validate(&input_schema()).unwrap_err();
        assert!(
            matches!(
                error,
                CalcFlowError::InvalidArgument { ref field, .. }
                    if field == "rolling.partition_by"
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn duplicate_partition_column_is_rejected() {
        let mut spec = valid_spec();
        spec.partition_by = vec!["symbol".into(), "symbol".into()];
        let error = spec.validate(&input_schema()).unwrap_err();
        assert!(
            matches!(
                error,
                CalcFlowError::InvalidArgument { ref field, .. }
                    if field == "rolling.partition_by[1]"
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn missing_partition_column_is_rejected() {
        let mut spec = valid_spec();
        spec.partition_by = vec!["industry".into()];
        let error = spec.validate(&input_schema()).unwrap_err();
        assert!(
            matches!(error, CalcFlowError::Compile { .. }),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn unsupported_partition_column_type_is_rejected() {
        let schema = with_field(
            &input_schema(),
            1,
            &Field::new("symbol", DataType::LargeBinary, false),
        );
        let error = valid_spec().validate(&schema).unwrap_err();
        assert!(
            matches!(error, CalcFlowError::Compile { .. }),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn missing_event_time_column_is_rejected() {
        let mut spec = valid_spec();
        spec.event_time = "event_ts".into();
        let error = spec.validate(&input_schema()).unwrap_err();
        assert!(
            matches!(error, CalcFlowError::Compile { .. }),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn nullable_event_time_is_rejected() {
        let schema = with_field(
            &input_schema(),
            0,
            &Field::new(
                "ts",
                DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC"))),
                true,
            ),
        );
        let error = valid_spec().validate(&schema).unwrap_err();
        assert!(
            matches!(error, CalcFlowError::Compile { .. }),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn non_utc_or_coarse_event_time_is_rejected() {
        for data_type in [
            DataType::Timestamp(TimeUnit::Microsecond, None),
            DataType::Timestamp(TimeUnit::Millisecond, Some(Arc::from("UTC"))),
            DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("Asia/Shanghai"))),
            DataType::Int64,
        ] {
            let schema = with_field(
                &input_schema(),
                0,
                &Field::new("ts", data_type.clone(), false),
            );
            let error = valid_spec().validate(&schema).unwrap_err();
            assert!(
                matches!(error, CalcFlowError::Compile { .. }),
                "event-time type {data_type} was accepted"
            );
        }
    }

    #[test]
    fn empty_sequence_by_is_rejected() {
        let mut spec = valid_spec();
        spec.sequence_by = Vec::new();
        let error = spec.validate(&input_schema()).unwrap_err();
        assert!(
            matches!(
                error,
                CalcFlowError::InvalidArgument { ref field, .. }
                    if field == "rolling.sequence_by"
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn nullable_sequence_column_is_rejected() {
        let schema = with_field(
            &input_schema(),
            2,
            &Field::new("sequence", DataType::UInt64, true),
        );
        let error = valid_spec().validate(&schema).unwrap_err();
        assert!(
            matches!(error, CalcFlowError::Compile { .. }),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn floating_sequence_column_is_rejected() {
        let schema = with_field(
            &input_schema(),
            2,
            &Field::new("sequence", DataType::Float64, false),
        );
        let error = valid_spec().validate(&schema).unwrap_err();
        assert!(
            matches!(error, CalcFlowError::Compile { .. }),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn empty_outputs_are_rejected() {
        let mut spec = valid_spec();
        spec.outputs = Vec::new();
        let error = spec.validate(&input_schema()).unwrap_err();
        assert!(
            matches!(
                error,
                CalcFlowError::InvalidArgument { ref field, .. }
                    if field == "rolling.outputs"
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn zero_periods_is_rejected() {
        for index in [0, 1] {
            let mut document = valid_spec_json();
            document["outputs"][index]["periods"] = json!(0);
            let spec: RollingSpec = serde_json::from_value(document).unwrap();
            let error = spec.validate(&input_schema()).unwrap_err();
            assert!(
                matches!(
                    error,
                    CalcFlowError::InvalidArgument { ref field, .. }
                        if field == &format!("rolling.outputs[{index}].periods")
                ),
                "unexpected error: {error}"
            );
        }
    }

    #[test]
    fn unsupported_primitive_version_is_rejected() {
        let mut document = valid_spec_json();
        document["outputs"][0]["primitive_version"] = json!(2);
        let spec: RollingSpec = serde_json::from_value(document).unwrap();
        let error = spec.validate(&input_schema()).unwrap_err();
        assert!(
            matches!(
                error,
                CalcFlowError::InvalidArgument { ref field, .. }
                    if field == "rolling.outputs[0].primitive_version"
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn missing_output_input_column_is_rejected() {
        let mut document = valid_spec_json();
        document["outputs"][0]["input"] = json!("close");
        let spec: RollingSpec = serde_json::from_value(document).unwrap();
        let error = spec.validate(&input_schema()).unwrap_err();
        assert!(
            matches!(error, CalcFlowError::Compile { .. }),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn delta_on_non_numeric_column_is_rejected() {
        let mut document = valid_spec_json();
        document["outputs"][1]["input"] = json!("label");
        let spec: RollingSpec = serde_json::from_value(document).unwrap();
        let error = spec.validate(&input_schema()).unwrap_err();
        assert!(
            matches!(error, CalcFlowError::Compile { .. }),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn lag_preserves_any_input_type() {
        let mut document = valid_spec_json();
        document["outputs"][0]["input"] = json!("label");
        document["outputs"][0]["output"] = json!("label_lag_1");
        let spec: RollingSpec = serde_json::from_value(document).unwrap();
        let output_schema = spec.validate(&input_schema()).unwrap();
        assert_eq!(
            output_schema.field_with_name("label_lag_1").unwrap(),
            &Field::new("label_lag_1", DataType::Utf8, true)
        );
    }

    #[test]
    fn duplicate_output_name_is_rejected() {
        let mut document = valid_spec_json();
        document["outputs"][1]["output"] = json!("price_lag_1");
        let spec: RollingSpec = serde_json::from_value(document).unwrap();
        let error = spec.validate(&input_schema()).unwrap_err();
        assert!(
            matches!(
                error,
                CalcFlowError::InvalidArgument { ref field, .. }
                    if field == "rolling.outputs[1].output"
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn output_name_colliding_with_an_input_field_is_rejected() {
        let mut document = valid_spec_json();
        document["outputs"][1]["output"] = json!("volume");
        let spec: RollingSpec = serde_json::from_value(document).unwrap();
        let error = spec.validate(&input_schema()).unwrap_err();
        assert!(
            matches!(
                error,
                CalcFlowError::InvalidArgument { ref field, .. }
                    if field == "rolling.outputs[1].output"
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn drop_metrics_version_must_be_one() {
        let mut document = valid_spec_json();
        document["late_policy"] = json!({"kind": "drop", "metrics_version": 2});
        let spec: RollingSpec = serde_json::from_value(document).unwrap();
        let error = spec.validate(&input_schema()).unwrap_err();
        assert!(
            matches!(
                error,
                CalcFlowError::InvalidArgument { ref field, .. }
                    if field == "rolling.late_policy.metrics_version"
            ),
            "unexpected error: {error}"
        );
    }

    // ------------------------------------------------------------------
    // Aggregate declarations (SCE-07, SCE-00 D3.2/D5)
    // ------------------------------------------------------------------

    fn aggregate_spec_json(outputs: Value) -> Value {
        let mut document = valid_spec_json();
        document["outputs"] = outputs;
        document
    }

    fn aggregate_output(kind: &str, input: &str, output: &str, size: u64) -> Value {
        json!({
            "kind": kind,
            "primitive_version": 1,
            "input": input,
            "output": output,
            "frame": {"kind": "rows", "size": size},
            "min_periods": 1
        })
    }

    fn ddof_output(kind: &str, input: &str, output: &str, size: u64, ddof: u64) -> Value {
        let mut declaration = aggregate_output(kind, input, output, size);
        declaration["ddof"] = json!(ddof);
        declaration
    }

    fn duration_output(kind: &str, input: &str, output: &str, micros: u64) -> Value {
        json!({
            "kind": kind,
            "primitive_version": 1,
            "input": input,
            "output": output,
            "frame": {"kind": "duration", "micros": micros},
            "min_periods": 1
        })
    }

    fn pair_output(
        kind: &str,
        left: &str,
        right: &str,
        output: &str,
        frame: Value,
        ddof: u64,
    ) -> Value {
        let mut declaration = json!({
            "kind": kind,
            "primitive_version": 1,
            "left": left,
            "right": right,
            "output": output,
            "min_periods": 1,
            "ddof": ddof
        });
        declaration["frame"] = frame;
        declaration
    }

    fn mean_leaf(input: &str, size: u64) -> Value {
        json!({
            "kind": "mean",
            "primitive_version": 1,
            "input": input,
            "frame": {"kind": "rows", "size": size},
            "min_periods": 1
        })
    }

    fn difference_output(left: &Value, right: &Value, output: &str) -> Value {
        json!({
            "kind": "difference",
            "primitive_version": 1,
            "left": left,
            "right": right,
            "output": output
        })
    }

    fn aggregate_spec(outputs: Value) -> RollingSpec {
        serde_json::from_value(aggregate_spec_json(outputs)).unwrap()
    }

    #[test]
    fn aggregate_outputs_round_trip_the_frozen_json() {
        let document = aggregate_spec_json(json!([
            aggregate_output("count", "price", "price_count_20", 20),
            aggregate_output("sum", "volume", "volume_sum_20", 20),
            aggregate_output("mean", "price", "price_mean_20", 20),
            ddof_output("variance", "price", "price_var_20", 20, 1),
            ddof_output("stddev", "price", "price_std_20", 20, 0),
        ]));
        let spec: RollingSpec = serde_json::from_value(document.clone()).unwrap();
        assert_eq!(serde_json::to_value(&spec).unwrap(), document);
    }

    #[test]
    fn duration_frames_round_trip_the_frozen_json() {
        let document = aggregate_spec_json(json!([
            duration_output("mean", "price", "price_mean_60s", 60_000_000),
            duration_output("count", "label", "label_count_60s", 60_000_000),
            pair_output(
                "correlation",
                "price",
                "volume",
                "price_volume_corr_60s",
                json!({"kind": "duration", "micros": 60_000_000}),
                1,
            ),
        ]));
        let spec: RollingSpec = serde_json::from_value(document.clone()).unwrap();
        assert_eq!(serde_json::to_value(&spec).unwrap(), document);
    }

    #[test]
    fn zero_duration_micros_is_rejected() {
        let spec = aggregate_spec(json!([duration_output("mean", "price", "m", 0)]));
        let error = spec.validate(&input_schema()).unwrap_err();
        assert!(
            matches!(
                error,
                CalcFlowError::InvalidArgument { ref field, .. }
                    if field == "rolling.outputs[0].frame.micros"
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn duration_frames_allow_min_periods_above_any_row_count() {
        // A duration frame has no row-count ceiling; only row frames cap
        // min_periods at their size (SCE-00 D5).
        let mut declaration = duration_output("mean", "price", "m", 60_000_000);
        declaration["min_periods"] = json!(10_000);
        let spec = aggregate_spec(json!([declaration]));
        assert!(spec.validate(&input_schema()).is_ok());
    }

    #[test]
    fn duration_frames_still_reject_zero_min_periods() {
        let mut declaration = duration_output("mean", "price", "m", 60_000_000);
        declaration["min_periods"] = json!(0);
        let spec = aggregate_spec(json!([declaration]));
        let error = spec.validate(&input_schema()).unwrap_err();
        assert!(
            matches!(
                error,
                CalcFlowError::InvalidArgument { ref field, .. }
                    if field == "rolling.outputs[0].min_periods"
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn duration_frames_reject_unknown_frame_fields() {
        let mut declaration = duration_output("mean", "price", "m", 60_000_000);
        declaration["frame"]["size"] = json!(5);
        let document = aggregate_spec_json(json!([declaration]));
        assert!(serde_json::from_value::<RollingSpec>(document).is_err());
    }

    #[test]
    fn extrema_outputs_round_trip_the_frozen_json() {
        let document = aggregate_spec_json(json!([
            aggregate_output("min", "price", "price_min_20", 20),
            aggregate_output("max", "price", "price_max_20", 20),
            duration_output("max", "volume", "volume_max_60s", 60_000_000),
        ]));
        let spec: RollingSpec = serde_json::from_value(document.clone()).unwrap();
        assert_eq!(serde_json::to_value(&spec).unwrap(), document);
    }

    #[test]
    fn extrema_outputs_reject_ddof() {
        for kind in ["min", "max"] {
            let declaration = ddof_output(kind, "price", "price_extrema", 20, 1);
            let document = aggregate_spec_json(json!([declaration]));
            assert!(
                serde_json::from_value::<RollingSpec>(document).is_err(),
                "{kind} with ddof was accepted"
            );
        }
    }

    #[test]
    fn pair_outputs_round_trip_the_frozen_json() {
        let document = aggregate_spec_json(json!([
            pair_output(
                "covariance",
                "price",
                "volume",
                "price_volume_cov_20",
                json!({"kind": "rows", "size": 20}),
                1,
            ),
            pair_output(
                "correlation",
                "price",
                "volume",
                "price_volume_corr_20",
                json!({"kind": "rows", "size": 20}),
                0,
            ),
        ]));
        let spec: RollingSpec = serde_json::from_value(document.clone()).unwrap();
        assert_eq!(serde_json::to_value(&spec).unwrap(), document);
    }

    #[test]
    fn pair_outputs_reject_missing_ddof_left_or_right() {
        let mut missing_ddof = pair_output(
            "covariance",
            "price",
            "volume",
            "price_volume_cov",
            json!({"kind": "rows", "size": 20}),
            1,
        );
        missing_ddof.as_object_mut().unwrap().remove("ddof");
        assert!(
            serde_json::from_value::<RollingSpec>(aggregate_spec_json(json!([missing_ddof])))
                .is_err()
        );
        for field in ["left", "right"] {
            let mut missing_operand = pair_output(
                "correlation",
                "price",
                "volume",
                "price_volume_corr",
                json!({"kind": "rows", "size": 20}),
                1,
            );
            missing_operand.as_object_mut().unwrap().remove(field);
            assert!(
                serde_json::from_value::<RollingSpec>(aggregate_spec_json(json!([
                    missing_operand
                ])))
                .is_err(),
                "pair output without {field} was accepted"
            );
        }
    }

    #[test]
    fn pair_outputs_reject_the_single_input_field() {
        let mut declaration = pair_output(
            "covariance",
            "price",
            "volume",
            "price_volume_cov",
            json!({"kind": "rows", "size": 20}),
            1,
        );
        declaration["input"] = json!("price");
        let document = aggregate_spec_json(json!([declaration]));
        assert!(serde_json::from_value::<RollingSpec>(document).is_err());
    }

    #[test]
    fn single_input_outputs_reject_left_and_right() {
        for kind in ["mean", "min", "max"] {
            let mut declaration = aggregate_output(kind, "price", "price_agg", 20);
            declaration["left"] = json!("price");
            let document = aggregate_spec_json(json!([declaration]));
            assert!(
                serde_json::from_value::<RollingSpec>(document).is_err(),
                "{kind} with left was accepted"
            );
        }
    }

    #[test]
    fn pair_outputs_reject_an_empty_right_column() {
        let spec = aggregate_spec(json!([pair_output(
            "covariance",
            "price",
            "",
            "price_volume_cov",
            json!({"kind": "rows", "size": 20}),
            1,
        )]));
        let error = spec.validate(&input_schema()).unwrap_err();
        assert!(
            matches!(
                error,
                CalcFlowError::InvalidArgument { ref field, .. }
                    if field == "rolling.outputs[0].right"
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn extrema_over_a_column_without_total_order_are_rejected() {
        let schema = with_field(
            &input_schema(),
            5,
            &Field::new(
                "label",
                DataType::Timestamp(TimeUnit::Nanosecond, None),
                true,
            ),
        );
        for kind in ["min", "max"] {
            let spec = aggregate_spec(json!([aggregate_output(
                kind,
                "label",
                "label_extrema_20",
                20
            )]));
            let error = spec.validate(&schema).unwrap_err();
            let expected = format!("rolling {kind} does not support column");
            assert!(
                matches!(
                    error,
                    CalcFlowError::Compile { ref message } if message.contains(&expected)
                ),
                "unexpected error for {kind}: {error}"
            );
        }
    }

    #[test]
    fn pair_output_ddof_above_one_is_rejected() {
        let spec = aggregate_spec(json!([pair_output(
            "correlation",
            "price",
            "volume",
            "price_volume_corr",
            json!({"kind": "rows", "size": 20}),
            2,
        )]));
        let error = spec.validate(&input_schema()).unwrap_err();
        assert!(
            matches!(
                error,
                CalcFlowError::InvalidArgument { ref field, .. }
                    if field == "rolling.outputs[0].ddof"
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn pair_outputs_reject_non_numeric_operands() {
        for left in ["label", "price"] {
            let right = if left == "label" { "price" } else { "label" };
            let spec = aggregate_spec(json!([pair_output(
                "covariance",
                left,
                right,
                "pair_stat",
                json!({"kind": "rows", "size": 20}),
                1,
            )]));
            let error = spec.validate(&input_schema()).unwrap_err();
            assert!(
                matches!(error, CalcFlowError::Compile { .. }),
                "unexpected error: {error}"
            );
        }
    }

    #[test]
    fn aggregate_outputs_reject_lag_only_fields() {
        let mut declaration = aggregate_output("mean", "price", "price_mean", 20);
        declaration["periods"] = json!(1);
        let document = aggregate_spec_json(json!([declaration]));
        assert!(serde_json::from_value::<RollingSpec>(document).is_err());
    }

    #[test]
    fn statistical_outputs_reject_missing_ddof() {
        for kind in ["variance", "stddev"] {
            let declaration = aggregate_output(kind, "price", "price_stat", 20);
            let document = aggregate_spec_json(json!([declaration]));
            assert!(
                serde_json::from_value::<RollingSpec>(document).is_err(),
                "{kind} without ddof was accepted"
            );
        }
    }

    #[test]
    fn non_statistical_aggregates_reject_ddof() {
        for kind in ["count", "sum", "mean"] {
            let declaration = ddof_output(kind, "price", "price_agg", 20, 1);
            let document = aggregate_spec_json(json!([declaration]));
            assert!(
                serde_json::from_value::<RollingSpec>(document).is_err(),
                "{kind} with ddof was accepted"
            );
        }
    }

    #[test]
    fn aggregate_output_schema_uses_the_frozen_type_table() {
        let spec = aggregate_spec(json!([
            aggregate_output("count", "price", "price_count", 20),
            aggregate_output("count", "label", "label_count", 20),
            aggregate_output("sum", "volume", "volume_sum", 20),
            aggregate_output("sum", "price", "price_sum", 20),
            aggregate_output("mean", "volume", "volume_mean", 20),
            ddof_output("variance", "price", "price_var", 20, 1),
            ddof_output("stddev", "volume", "volume_std", 20, 0),
        ]));
        let output_schema = spec.validate(&input_schema()).unwrap();
        let derived = &output_schema.fields()[input_schema().fields().len()..];
        let expected = [
            ("price_count", DataType::UInt64),
            ("label_count", DataType::UInt64),
            ("volume_sum", DataType::Int64),
            ("price_sum", DataType::Float64),
            ("volume_mean", DataType::Float64),
            ("price_var", DataType::Float64),
            ("volume_std", DataType::Float64),
        ];
        assert_eq!(derived.len(), expected.len());
        for (field, (name, data_type)) in derived.iter().zip(expected) {
            assert_eq!(field.name(), name);
            assert_eq!(field.data_type(), &data_type);
            assert!(field.is_nullable());
        }
    }

    #[test]
    fn extrema_and_pair_output_schema_uses_the_frozen_type_table() {
        let spec = aggregate_spec(json!([
            aggregate_output("min", "price", "price_min", 20),
            aggregate_output("max", "volume", "volume_max", 20),
            aggregate_output("max", "label", "label_max", 20),
            pair_output(
                "covariance",
                "price",
                "volume",
                "price_volume_cov",
                json!({"kind": "rows", "size": 20}),
                1,
            ),
            pair_output(
                "correlation",
                "price",
                "volume",
                "price_volume_corr",
                json!({"kind": "duration", "micros": 60_000_000}),
                1,
            ),
        ]));
        let output_schema = spec.validate(&input_schema()).unwrap();
        let derived = &output_schema.fields()[input_schema().fields().len()..];
        let expected = [
            ("price_min", DataType::Float64),
            ("volume_max", DataType::Int64),
            ("label_max", DataType::Utf8),
            ("price_volume_cov", DataType::Float64),
            ("price_volume_corr", DataType::Float64),
        ];
        assert_eq!(derived.len(), expected.len());
        for (field, (name, data_type)) in derived.iter().zip(expected) {
            assert_eq!(field.name(), name);
            assert_eq!(field.data_type(), &data_type);
            assert!(field.is_nullable());
        }
    }

    #[test]
    fn zero_frame_size_is_rejected() {
        let mut spec = aggregate_spec(json!([aggregate_output("mean", "price", "m", 20)]));
        let RollingOutputSpec::Mean { frame, .. } = &mut spec.outputs[0] else {
            panic!("expected a mean output");
        };
        let RollingFrameSpec::Rows { size } = frame else {
            panic!("expected a rows frame");
        };
        *size = 0;
        let error = spec.validate(&input_schema()).unwrap_err();
        assert!(
            matches!(
                error,
                CalcFlowError::InvalidArgument { ref field, .. }
                    if field == "rolling.outputs[0].frame.size"
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn zero_min_periods_is_rejected() {
        let mut spec = aggregate_spec(json!([aggregate_output("mean", "price", "m", 20)]));
        let RollingOutputSpec::Mean { min_periods, .. } = &mut spec.outputs[0] else {
            panic!("expected a mean output");
        };
        *min_periods = 0;
        let error = spec.validate(&input_schema()).unwrap_err();
        assert!(
            matches!(
                error,
                CalcFlowError::InvalidArgument { ref field, .. }
                    if field == "rolling.outputs[0].min_periods"
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn min_periods_above_the_frame_size_is_rejected() {
        let mut declaration = aggregate_output("mean", "price", "m", 3);
        declaration["min_periods"] = json!(4);
        let spec = aggregate_spec(json!([declaration]));
        let error = spec.validate(&input_schema()).unwrap_err();
        assert!(
            matches!(
                error,
                CalcFlowError::InvalidArgument { ref field, .. }
                    if field == "rolling.outputs[0].min_periods"
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn ddof_above_one_is_rejected() {
        let mut spec = aggregate_spec(json!([ddof_output("variance", "price", "v", 20, 1)]));
        let RollingOutputSpec::Variance { ddof, .. } = &mut spec.outputs[0] else {
            panic!("expected a variance output");
        };
        *ddof = 2;
        let error = spec.validate(&input_schema()).unwrap_err();
        assert!(
            matches!(
                error,
                CalcFlowError::InvalidArgument { ref field, .. }
                    if field == "rolling.outputs[0].ddof"
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn sum_mean_variance_and_stddev_reject_non_numeric_inputs() {
        for declaration in [
            aggregate_output("sum", "label", "label_sum", 20),
            aggregate_output("mean", "label", "label_mean", 20),
            ddof_output("variance", "label", "label_var", 20, 1),
            ddof_output("stddev", "label", "label_std", 20, 1),
        ] {
            let spec = aggregate_spec(json!([declaration]));
            let error = spec.validate(&input_schema()).unwrap_err();
            assert!(
                matches!(error, CalcFlowError::Compile { .. }),
                "unexpected error: {error}"
            );
        }
    }

    #[test]
    fn count_accepts_non_numeric_inputs() {
        let spec = aggregate_spec(json!([aggregate_output("count", "label", "n", 20)]));
        assert!(spec.validate(&input_schema()).is_ok());
    }

    // ------------------------------------------------------------------
    // Shared lag/delta kernel
    // ------------------------------------------------------------------

    fn ts_scalar(value: i64) -> ScalarValue {
        ScalarValue::TimestampMicrosecond(Some(value), Some(Arc::from("UTC")))
    }

    fn full_row(
        event_time: i64,
        symbol: &str,
        sequence: u64,
        rest: Vec<ScalarValue>,
    ) -> BufferedRow {
        let mut values = vec![
            ts_scalar(event_time),
            ScalarValue::Utf8(Some(symbol.into())),
            ScalarValue::UInt64(Some(sequence)),
        ];
        values.extend(rest);
        while values.len() < 6 {
            values.push(match values.len() {
                3 => ScalarValue::Float64(None),
                4 => ScalarValue::Int64(None),
                _ => ScalarValue::Utf8(None),
            });
        }
        BufferedRow::new(
            vec![Some(KeyValue::String(symbol.into()))],
            vec![KeyValue::Unsigned(sequence)],
            event_time,
            values,
        )
    }

    fn kernel_schema() -> Schema {
        Schema::new(vec![
            Field::new(
                "ts",
                DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC"))),
                false,
            ),
            Field::new("symbol", DataType::Utf8, false),
            Field::new("sequence", DataType::UInt64, false),
            Field::new("price", DataType::Float64, true),
            Field::new("volume", DataType::Int64, true),
            Field::new("label", DataType::Utf8, true),
        ])
    }

    fn kernel_spec(outputs: Value) -> RollingSpec {
        let mut document = valid_spec_json();
        document["partition_by"] = json!(["symbol"]);
        document["event_time"] = json!("ts");
        document["sequence_by"] = json!(["sequence"]);
        document["outputs"] = outputs;
        serde_json::from_value(document).unwrap()
    }

    fn exponential_kernel_spec(outputs: Value) -> RollingSpec {
        let mut document = valid_spec_json();
        document["state_layout_version"] = json!(2);
        document["partition_by"] = json!(["symbol"]);
        document["event_time"] = json!("ts");
        document["sequence_by"] = json!(["sequence"]);
        document["outputs"] = outputs;
        serde_json::from_value(document).unwrap()
    }

    fn ewma_price(span: u64, min_periods: u64, output: &str) -> Value {
        json!({
            "kind": "ewma",
            "primitive_version": 1,
            "input": "price",
            "output": output,
            "span": span,
            "min_periods": min_periods
        })
    }

    fn lag_price(periods: u64) -> Value {
        json!({
            "kind": "lag",
            "primitive_version": 1,
            "input": "price",
            "output": "price_lag",
            "periods": periods
        })
    }

    fn delta_price(periods: u64) -> Value {
        json!({
            "kind": "delta",
            "primitive_version": 1,
            "input": "price",
            "output": "price_delta",
            "periods": periods
        })
    }

    fn delta_volume(periods: u64) -> Value {
        json!({
            "kind": "delta",
            "primitive_version": 1,
            "input": "volume",
            "output": "volume_delta",
            "periods": periods
        })
    }

    fn compute(
        spec: &RollingSpec,
        histories: &RollingHistories,
        rows: &[BufferedRow],
    ) -> Result<ComputedOutputs> {
        let compiled = compile_spec(spec, &kernel_schema())?;
        compute_output_columns(rows, histories, &compiled, "rolling")
    }

    fn float_column(outputs: &ComputedOutputs, index: usize) -> Vec<Option<f64>> {
        outputs.columns[index]
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .iter()
            .collect()
    }

    #[test]
    fn stable_v2_rebases_large_offset_variance_with_shifted_sums() {
        let mut spec = kernel_spec(json!([ddof_output(
            "variance",
            "price",
            "price_variance",
            64,
            0
        )]));
        spec.numerical_profile = RollingNumericalProfile::StableV2Preview;
        let prices = (0..64)
            .map(|index| 1.0e12 + f64::from(index % 9) / 10.0)
            .collect::<Vec<_>>();
        let rows = prices
            .iter()
            .enumerate()
            .map(|(index, price)| {
                let sequence = u64::try_from(index + 1).unwrap();
                full_row(
                    i64::try_from(index + 1).unwrap(),
                    "a",
                    sequence,
                    vec![ScalarValue::Float64(Some(*price))],
                )
            })
            .collect::<Vec<_>>();

        let outputs = compute(&spec, &RollingHistories::default(), &rows).unwrap();
        let actual = float_column(&outputs, 0)[63].unwrap();
        let shift = prices[0];
        let shifted = prices.iter().map(|value| value - shift).collect::<Vec<_>>();
        let sum = shifted.iter().sum::<f64>();
        let expected =
            shifted.iter().map(|value| value * value).sum::<f64>() / 64.0 - (sum / 64.0).powi(2);

        assert_eq!(actual.to_bits(), expected.to_bits());
        assert_eq!(outputs.touched[0].1.transition_count, 64);
    }

    fn signed_column(outputs: &ComputedOutputs, index: usize) -> Vec<Option<i64>> {
        outputs.columns[index]
            .as_any()
            .downcast_ref::<datafusion::arrow::array::Int64Array>()
            .unwrap()
            .iter()
            .collect()
    }

    #[test]
    fn ewma_uses_first_sample_seeding_and_ignores_null_and_nan() {
        let spec = exponential_kernel_spec(json!([ewma_price(3, 2, "ema")]));
        let rows = vec![
            full_row(1, "a", 1, vec![ScalarValue::Float64(Some(10.0))]),
            full_row(2, "a", 2, vec![ScalarValue::Float64(None)]),
            full_row(3, "a", 3, vec![ScalarValue::Float64(Some(14.0))]),
            full_row(4, "a", 4, vec![ScalarValue::Float64(Some(f64::NAN))]),
            full_row(5, "a", 5, vec![ScalarValue::Float64(Some(18.0))]),
            full_row(6, "a", 6, vec![ScalarValue::Float64(Some(10.0))]),
        ];
        let outputs = compute(&spec, &RollingHistories::default(), &rows).unwrap();
        assert_eq!(
            float_column(&outputs, 0),
            vec![None, None, Some(12.0), Some(12.0), Some(15.0), Some(12.5)]
        );
    }

    #[test]
    fn ewma_shares_state_and_is_segmentation_invariant() {
        let spec = exponential_kernel_spec(json!([
            ewma_price(3, 1, "ema_ready"),
            ewma_price(3, 3, "ema_warm")
        ]));
        let compiled = compile_spec(&spec, &kernel_schema()).unwrap();
        assert_eq!(compiled.window_groups.len(), 1);
        let rows = vec![
            full_row(1, "a", 1, vec![ScalarValue::Float64(Some(10.0))]),
            full_row(2, "a", 2, vec![ScalarValue::Float64(Some(14.0))]),
            full_row(3, "a", 3, vec![ScalarValue::Float64(Some(18.0))]),
            full_row(4, "a", 4, vec![ScalarValue::Float64(Some(10.0))]),
        ];
        let all = compute_output_columns(&rows, &RollingHistories::default(), &compiled, "rolling")
            .unwrap();
        let first = compute_output_columns(
            &rows[..2],
            &RollingHistories::default(),
            &compiled,
            "rolling",
        )
        .unwrap();
        let mut histories = RollingHistories::default();
        histories.apply(first.touched.clone());
        let second = compute_output_columns(&rows[2..], &histories, &compiled, "rolling").unwrap();
        for column in 0..2 {
            let segmented = float_column(&first, column)
                .into_iter()
                .chain(float_column(&second, column))
                .collect::<Vec<_>>();
            assert_eq!(segmented, float_column(&all, column));
        }
    }

    #[test]
    fn ewma_layout_two_restores_without_retained_history() {
        let spec = exponential_kernel_spec(json!([ewma_price(3, 1, "ema")]));
        let compiled = compile_spec(&spec, &kernel_schema()).unwrap();
        let rows = vec![
            full_row(1, "a", 1, vec![ScalarValue::Float64(Some(10.0))]),
            full_row(2, "a", 2, vec![ScalarValue::Float64(Some(14.0))]),
        ];
        let first =
            compute_output_columns(&rows, &RollingHistories::default(), &compiled, "rolling")
                .unwrap();
        let mut histories = RollingHistories::default();
        histories.apply(first.touched);
        assert!(
            histories
                .by_entity
                .values()
                .all(|state| state.rows.is_empty())
        );
        let mut legacy = compiled.clone();
        legacy.state_layout_version = ROLLING_EWMA_STATE_LAYOUT_VERSION;
        legacy.state_schema_fingerprint = compiled.legacy_state_schema_fingerprint.clone();
        let bytes = encode_state_segment_legacy(
            &histories,
            &BTreeMap::new(),
            &kernel_schema(),
            &legacy,
            TEST_FINGERPRINT,
            "rolling",
        )
        .unwrap();
        let metadata = RollingSnapshotMetadata {
            state_layout_version: 2,
            configuration_hash: compiled.configuration_hash.clone(),
            state_schema_fingerprint: compiled.legacy_state_schema_fingerprint.clone(),
            kernel_fingerprint: None,
            numerical_profile: None,
            epoch: Epoch::new(1).unwrap(),
            pipeline_fingerprint: Some(TEST_FINGERPRINT.into()),
            operator_id: Some("rolling".into()),
            last_input_watermark: None,
            next_output_sequence: 0,
            ended: false,
            metrics: LateMetricDelta::default(),
            segment_inventory: Vec::new(),
        };
        let restored =
            decode_state_segment(&bytes, &kernel_schema(), &compiled, &metadata).unwrap();
        let continuation = vec![full_row(3, "a", 3, vec![ScalarValue::Float64(Some(18.0))])];
        let expected =
            compute_output_columns(&continuation, &histories, &compiled, "rolling").unwrap();
        let output_schema = Arc::new(output_schema(&kernel_schema(), &compiled.outputs));
        let (actual, state, _) = build_typed_stream_output(
            &continuation,
            &restored.histories,
            None,
            &compiled,
            &output_schema,
            "rolling",
        )
        .unwrap()
        .unwrap();
        let actual = actual
            .column(kernel_schema().fields().len())
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .iter()
            .collect::<Vec<_>>();
        assert_eq!(actual, float_column(&expected, 0));
        assert_eq!(actual, vec![Some(15.0)]);
        assert!(state.is_some());
    }

    #[test]
    fn layout_one_history_bootstraps_the_typed_transition() {
        let spec = kernel_spec(json!([aggregate_output("mean", "price", "price_mean", 3)]));
        let compiled = compile_spec(&spec, &kernel_schema()).unwrap();
        let first_rows = vec![
            full_row(1, "a", 1, vec![ScalarValue::Float64(Some(1.0))]),
            full_row(2, "a", 2, vec![ScalarValue::Float64(Some(3.0))]),
        ];
        let first = compute_output_columns(
            &first_rows,
            &RollingHistories::default(),
            &compiled,
            "rolling",
        )
        .unwrap();
        let mut histories = RollingHistories::default();
        histories.apply(first.touched);
        let mut legacy = compiled.clone();
        legacy.state_layout_version = ROLLING_STATE_LAYOUT_VERSION;
        legacy.state_schema_fingerprint = compiled.legacy_state_schema_fingerprint.clone();
        let bytes = encode_state_segment_legacy(
            &histories,
            &BTreeMap::new(),
            &kernel_schema(),
            &legacy,
            TEST_FINGERPRINT,
            "rolling",
        )
        .unwrap();
        let metadata = RollingSnapshotMetadata {
            state_layout_version: ROLLING_STATE_LAYOUT_VERSION,
            configuration_hash: compiled.configuration_hash.clone(),
            state_schema_fingerprint: compiled.legacy_state_schema_fingerprint.clone(),
            kernel_fingerprint: None,
            numerical_profile: None,
            epoch: Epoch::new(1).unwrap(),
            pipeline_fingerprint: Some(TEST_FINGERPRINT.into()),
            operator_id: Some("rolling".into()),
            last_input_watermark: None,
            next_output_sequence: 0,
            ended: false,
            metrics: LateMetricDelta::default(),
            segment_inventory: Vec::new(),
        };
        let restored =
            decode_state_segment(&bytes, &kernel_schema(), &compiled, &metadata).unwrap();
        let continuation = vec![full_row(3, "a", 3, vec![ScalarValue::Float64(Some(5.0))])];
        let expected =
            compute_output_columns(&continuation, &histories, &compiled, "rolling").unwrap();
        let output_schema = Arc::new(output_schema(&kernel_schema(), &compiled.outputs));
        let (actual, state, _) = build_typed_stream_output(
            &continuation,
            &restored.histories,
            None,
            &compiled,
            &output_schema,
            "rolling",
        )
        .unwrap()
        .unwrap();
        let actual = actual
            .column(kernel_schema().fields().len())
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .iter()
            .collect::<Vec<_>>();
        assert_eq!(actual, float_column(&expected, 0));
        assert_eq!(actual, vec![Some(3.0)]);
        assert!(state.is_some());
    }

    #[test]
    fn ewma_restore_rejects_noncanonical_accumulator_rows() {
        let spec = exponential_kernel_spec(json!([ewma_price(3, 1, "ema")]));
        let compiled = compile_spec(&spec, &kernel_schema()).unwrap();
        let entity = vec![Some(KeyValue::String("a".into()))];
        let values = ewma_entity_values(&entity, &kernel_schema(), &compiled).unwrap();

        let mut decoded = DecodedRollingState::default();
        let mut previous = None;
        let error = decode_ewma_state_row(
            None,
            &values,
            Some((0, 0, 10.0)),
            &mut decoded,
            &compiled,
            &mut previous,
        )
        .unwrap_err();
        assert!(matches!(error, CalcFlowError::Format { .. }));

        let mut populated = values.clone();
        populated[3] = ScalarValue::Float64(Some(10.0));
        let error = decode_ewma_state_row(
            None,
            &populated,
            Some((0, 1, 10.0)),
            &mut DecodedRollingState::default(),
            &compiled,
            &mut None,
        )
        .unwrap_err();
        assert!(matches!(error, CalcFlowError::Format { .. }));

        let error = decode_ewma_state_row(
            Some(0),
            &values,
            Some((0, 1, 10.0)),
            &mut DecodedRollingState::default(),
            &compiled,
            &mut None,
        )
        .unwrap_err();
        assert!(matches!(error, CalcFlowError::Format { .. }));
    }

    #[test]
    fn lag_references_the_previous_row_within_each_entity() {
        let spec = kernel_spec(json!([lag_price(1)]));
        let rows = vec![
            full_row(1, "a", 1, vec![ScalarValue::Float64(Some(1.0))]),
            full_row(1, "b", 1, vec![ScalarValue::Float64(Some(10.0))]),
            full_row(2, "a", 2, vec![ScalarValue::Float64(Some(2.0))]),
            full_row(2, "b", 2, vec![ScalarValue::Float64(Some(20.0))]),
            full_row(3, "a", 3, vec![ScalarValue::Float64(Some(3.0))]),
        ];
        let outputs = compute(&spec, &RollingHistories::default(), &rows).unwrap();
        assert_eq!(
            float_column(&outputs, 0),
            vec![None, None, Some(1.0), Some(10.0), Some(2.0)]
        );
    }

    #[test]
    fn lag_periods_span_the_shared_history_across_segmentation() {
        let spec = kernel_spec(json!([lag_price(2)]));
        let mut histories = RollingHistories::default();
        let first = vec![
            full_row(1, "a", 1, vec![ScalarValue::Float64(Some(1.0))]),
            full_row(2, "a", 2, vec![ScalarValue::Float64(Some(2.0))]),
        ];
        let outputs = compute(&spec, &histories, &first).unwrap();
        assert_eq!(float_column(&outputs, 0), vec![None, None]);
        histories.apply(outputs.touched);

        let second = vec![
            full_row(3, "a", 3, vec![ScalarValue::Float64(Some(3.0))]),
            full_row(4, "a", 4, vec![ScalarValue::Float64(Some(4.0))]),
            full_row(5, "a", 5, vec![ScalarValue::Float64(Some(5.0))]),
        ];
        let outputs = compute(&spec, &histories, &second).unwrap();
        assert_eq!(
            float_column(&outputs, 0),
            vec![Some(1.0), Some(2.0), Some(3.0)]
        );
    }

    #[test]
    fn lag_preserves_null_and_nan_at_the_referenced_position() {
        let spec = kernel_spec(json!([lag_price(1)]));
        let rows = vec![
            full_row(1, "a", 1, vec![ScalarValue::Float64(None)]),
            full_row(2, "a", 2, vec![ScalarValue::Float64(Some(f64::NAN))]),
            full_row(3, "a", 3, vec![ScalarValue::Float64(Some(3.0))]),
            full_row(4, "a", 4, vec![ScalarValue::Float64(Some(4.0))]),
        ];
        let outputs = compute(&spec, &RollingHistories::default(), &rows).unwrap();
        let values = outputs.columns[0]
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert!(values.is_null(0));
        assert!(values.is_null(1));
        assert!(values.value(2).is_nan());
        assert_eq!(values.value(3).to_bits(), 3.0_f64.to_bits());
    }

    #[test]
    fn lag_works_for_non_numeric_columns() {
        let spec = kernel_spec(json!([{
            "kind": "lag",
            "primitive_version": 1,
            "input": "label",
            "output": "label_lag",
            "periods": 1
        }]));
        let rows = vec![
            full_row(
                1,
                "a",
                1,
                vec![
                    ScalarValue::Float64(None),
                    ScalarValue::Int64(None),
                    ScalarValue::Utf8(Some("x".into())),
                ],
            ),
            full_row(
                2,
                "a",
                2,
                vec![
                    ScalarValue::Float64(None),
                    ScalarValue::Int64(None),
                    ScalarValue::Utf8(Some("y".into())),
                ],
            ),
        ];
        let outputs = compute(&spec, &RollingHistories::default(), &rows).unwrap();
        let values = outputs.columns[0]
            .as_any()
            .downcast_ref::<datafusion::arrow::array::StringArray>()
            .unwrap();
        assert!(values.is_null(0));
        assert_eq!(values.value(1), "x");
    }

    #[test]
    fn delta_subtracts_the_referenced_value_with_checked_integer_math() {
        let spec = kernel_spec(json!([delta_volume(1)]));
        let rows = vec![
            full_row(
                1,
                "a",
                1,
                vec![ScalarValue::Float64(None), ScalarValue::Int64(Some(7))],
            ),
            full_row(
                2,
                "a",
                2,
                vec![ScalarValue::Float64(None), ScalarValue::Int64(Some(10))],
            ),
            full_row(
                3,
                "a",
                3,
                vec![ScalarValue::Float64(None), ScalarValue::Int64(Some(4))],
            ),
        ];
        let outputs = compute(&spec, &RollingHistories::default(), &rows).unwrap();
        assert_eq!(signed_column(&outputs, 0), vec![None, Some(3), Some(-6)]);
    }

    #[test]
    fn delta_integer_overflow_is_a_data_error() {
        let spec = kernel_spec(json!([delta_volume(1)]));
        let rows = vec![
            full_row(
                1,
                "a",
                1,
                vec![ScalarValue::Float64(None), ScalarValue::Int64(Some(-1))],
            ),
            full_row(
                2,
                "a",
                2,
                vec![
                    ScalarValue::Float64(None),
                    ScalarValue::Int64(Some(i64::MAX)),
                ],
            ),
        ];
        let error = compute(&spec, &RollingHistories::default(), &rows).unwrap_err();
        assert!(
            matches!(error, CalcFlowError::Operator { ref node_id, .. } if node_id == "rolling"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn delta_preserves_null_and_propagates_nan() {
        let spec = kernel_spec(json!([delta_price(1)]));
        let rows = vec![
            full_row(1, "a", 1, vec![ScalarValue::Float64(None)]),
            full_row(2, "a", 2, vec![ScalarValue::Float64(Some(1.5))]),
            full_row(3, "a", 3, vec![ScalarValue::Float64(Some(f64::NAN))]),
            full_row(4, "a", 4, vec![ScalarValue::Float64(Some(2.5))]),
            full_row(5, "a", 5, vec![ScalarValue::Float64(Some(f64::INFINITY))]),
            full_row(6, "a", 6, vec![ScalarValue::Float64(Some(f64::INFINITY))]),
        ];
        let outputs = compute(&spec, &RollingHistories::default(), &rows).unwrap();
        let values = outputs.columns[0]
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert!(values.is_null(0));
        assert!(values.is_null(1));
        assert!(values.value(2).is_nan());
        assert!(values.value(3).is_nan());
        assert_eq!(values.value(4).to_bits(), f64::INFINITY.to_bits());
        assert!(values.value(5).is_nan());
    }

    #[test]
    fn delta_unsigned_underflow_is_a_data_error() {
        let spec = kernel_spec(json!([{
            "kind": "delta",
            "primitive_version": 1,
            "input": "sequence",
            "output": "sequence_delta",
            "periods": 1
        }]));
        let rows = vec![full_row(1, "a", 10, vec![]), full_row(2, "a", 3, vec![])];
        let error = compute(&spec, &RollingHistories::default(), &rows).unwrap_err();
        assert!(
            matches!(error, CalcFlowError::Operator { .. }),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn history_is_truncated_to_the_maximum_declared_periods() {
        let spec = kernel_spec(json!([lag_price(2), lag_price(1)]));
        let mut histories = RollingHistories::default();
        for batch in 0..3_u32 {
            let rows = (0..4_u32)
                .map(|index| {
                    let sequence = batch * 4 + index + 1;
                    full_row(
                        i64::from(sequence),
                        "a",
                        u64::from(sequence),
                        vec![ScalarValue::Float64(Some(f64::from(sequence)))],
                    )
                })
                .collect::<Vec<_>>();
            let outputs = compute(&spec, &histories, &rows).unwrap();
            histories.apply(outputs.touched);
        }
        for state in histories.by_entity.values() {
            assert!(state.rows.len() <= 2);
        }
    }

    #[test]
    fn failed_delta_leaves_histories_untouched() {
        let spec = kernel_spec(json!([delta_volume(1)]));
        let histories = RollingHistories::default();
        let rows = vec![
            full_row(
                1,
                "a",
                1,
                vec![ScalarValue::Float64(None), ScalarValue::Int64(Some(-1))],
            ),
            full_row(
                2,
                "a",
                2,
                vec![
                    ScalarValue::Float64(None),
                    ScalarValue::Int64(Some(i64::MAX)),
                ],
            ),
        ];
        assert!(compute(&spec, &histories, &rows).is_err());
        assert!(histories.by_entity.is_empty());
    }

    // ------------------------------------------------------------------
    // Shared aggregate kernel (SCE-07)
    // ------------------------------------------------------------------

    fn price_rows(prices: &[Option<f64>]) -> Vec<BufferedRow> {
        prices
            .iter()
            .enumerate()
            .map(|(index, price)| {
                let sequence = u64::try_from(index + 1).unwrap();
                full_row(
                    i64::try_from(index + 1).unwrap(),
                    "a",
                    sequence,
                    vec![ScalarValue::Float64(*price)],
                )
            })
            .collect()
    }

    fn unsigned_column(outputs: &ComputedOutputs, index: usize) -> Vec<Option<u64>> {
        outputs.columns[index]
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .iter()
            .collect()
    }

    fn float64_fast_record(rows: &[(i64, &str, u64, Option<f64>)]) -> RecordBatch {
        use datafusion::arrow::array::{Int64Array, StringArray, TimestampMicrosecondArray};

        let timestamps = rows
            .iter()
            .map(|(event_time, ..)| Some(*event_time))
            .collect::<TimestampMicrosecondArray>()
            .with_timezone("UTC");
        let symbols = StringArray::from_iter_values(rows.iter().map(|(_, symbol, ..)| *symbol));
        let sequences =
            UInt64Array::from_iter_values(rows.iter().map(|(_, _, sequence, _)| *sequence));
        let prices = rows
            .iter()
            .map(|(_, _, _, price)| *price)
            .collect::<Float64Array>();
        let volume = Int64Array::new_null(rows.len());
        let label = StringArray::new_null(rows.len());
        RecordBatch::try_new(
            Arc::new(kernel_schema()),
            vec![
                Arc::new(timestamps),
                Arc::new(symbols),
                Arc::new(sequences),
                Arc::new(prices),
                Arc::new(volume),
                Arc::new(label),
            ],
        )
        .unwrap()
    }

    fn float64_pair_schema() -> Schema {
        with_field(
            &kernel_schema(),
            4,
            &Field::new("volume", DataType::Float64, true),
        )
    }

    type Float64PairRow<'a> = (i64, &'a str, u64, Option<f64>, Option<f64>);

    fn float64_pair_record(rows: &[Float64PairRow<'_>]) -> RecordBatch {
        use datafusion::arrow::array::{StringArray, TimestampMicrosecondArray};

        let timestamps = rows
            .iter()
            .map(|(event_time, ..)| Some(*event_time))
            .collect::<TimestampMicrosecondArray>()
            .with_timezone("UTC");
        RecordBatch::try_new(
            Arc::new(float64_pair_schema()),
            vec![
                Arc::new(timestamps),
                Arc::new(StringArray::from_iter_values(
                    rows.iter().map(|(_, symbol, ..)| *symbol),
                )),
                Arc::new(UInt64Array::from_iter_values(
                    rows.iter().map(|(_, _, sequence, ..)| *sequence),
                )),
                Arc::new(
                    rows.iter()
                        .map(|(_, _, _, price, _)| *price)
                        .collect::<Float64Array>(),
                ),
                Arc::new(
                    rows.iter()
                        .map(|(_, _, _, _, volume)| *volume)
                        .collect::<Float64Array>(),
                ),
                Arc::new(StringArray::new_null(rows.len())),
            ],
        )
        .unwrap()
    }

    fn int64_fast_record(values: &[Option<i64>]) -> RecordBatch {
        use datafusion::arrow::array::{Int64Array, StringArray, TimestampMicrosecondArray};

        let len = values.len();
        RecordBatch::try_new(
            Arc::new(kernel_schema()),
            vec![
                Arc::new(
                    (1..=i64::try_from(len).unwrap())
                        .map(Some)
                        .collect::<TimestampMicrosecondArray>()
                        .with_timezone("UTC"),
                ),
                Arc::new(StringArray::from_iter_values(std::iter::repeat_n("a", len))),
                Arc::new(UInt64Array::from_iter_values(
                    1..=u64::try_from(len).unwrap(),
                )),
                Arc::new(Float64Array::new_null(len)),
                Arc::new(values.iter().copied().collect::<Int64Array>()),
                Arc::new(StringArray::new_null(len)),
            ],
        )
        .unwrap()
    }

    fn primitive_volume_record(values: ArrayRef) -> (Schema, RecordBatch) {
        use datafusion::arrow::array::{StringArray, TimestampMicrosecondArray};

        let len = values.len();
        let schema = with_field(
            &kernel_schema(),
            4,
            &Field::new("volume", values.data_type().clone(), true),
        );
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(
                    (1..=i64::try_from(len).unwrap())
                        .map(Some)
                        .collect::<TimestampMicrosecondArray>()
                        .with_timezone("UTC"),
                ),
                Arc::new(StringArray::from_iter_values(std::iter::repeat_n("a", len))),
                Arc::new(UInt64Array::from_iter_values(
                    1..=u64::try_from(len).unwrap(),
                )),
                Arc::new(Float64Array::new_null(len)),
                values,
                Arc::new(StringArray::new_null(len)),
            ],
        )
        .unwrap();
        (schema, batch)
    }

    fn assert_typed_matches_general(spec: &RollingSpec, schema: &Schema, input: &RecordBatch) {
        let compiled = compile_spec(spec, schema).unwrap();
        let fast = compiled
            .kernel_plan
            .open_and_fill(input, "rolling")
            .unwrap()
            .unwrap();
        let rows = (0..input.num_rows())
            .map(|row_index| read_buffered_row(input, row_index, &compiled, "rolling").unwrap())
            .collect::<Vec<_>>();
        let general =
            compute_output_columns(&rows, &RollingHistories::default(), &compiled, "rolling")
                .unwrap();

        assert_eq!(fast.columns.len(), general.columns.len());
        for (fast, general) in fast.columns.iter().zip(&general.columns) {
            assert_eq!(fast.to_data(), general.to_data());
        }
    }

    #[test]
    fn ordered_float64_plan_matches_general_aggregate_kernel_without_row_materialization() {
        let spec = kernel_spec(json!([
            aggregate_output("count", "price", "price_count", 2),
            aggregate_output("sum", "price", "price_sum", 2),
            aggregate_output("mean", "price", "price_mean", 2),
            ddof_output("variance", "price", "price_var", 2, 1),
            ddof_output("stddev", "price", "price_std", 2, 0),
        ]));
        let compiled = compile_spec(&spec, &kernel_schema()).unwrap();
        assert_eq!(
            compiled.kernel_plan.selection(),
            KernelSelection::OrderedPrimitive
        );
        let input = float64_fast_record(&[
            (1, "a", 1, Some(1.0)),
            (1, "b", 1, Some(10.0)),
            (2, "a", 2, None),
            (2, "b", 2, Some(20.0)),
            (3, "a", 3, Some(f64::NAN)),
            (3, "b", 3, Some(30.0)),
            (4, "a", 4, Some(4.0)),
        ]);
        let fast = compiled
            .kernel_plan
            .open_and_fill(&input, "rolling")
            .unwrap()
            .unwrap();
        let rows = (0..input.num_rows())
            .map(|row_index| read_buffered_row(&input, row_index, &compiled, "rolling").unwrap())
            .collect::<Vec<_>>();
        let general =
            compute_output_columns(&rows, &RollingHistories::default(), &compiled, "rolling")
                .unwrap();

        assert_eq!(fast.columns.len(), general.columns.len());
        for (fast, general) in fast.columns.iter().zip(&general.columns) {
            assert_eq!(fast.as_ref(), general.as_ref());
        }
        assert_eq!(fast.metrics.order_proof_rows, input.num_rows());
        assert_eq!(fast.metrics.entities, 2);
        assert_eq!(fast.metrics.scalar_value_conversions, 0);
        assert_eq!(fast.metrics.sort_count, 0);
    }

    #[test]
    fn typed_update_and_fill_matches_one_shot_fill_across_micro_batches() {
        use datafusion::arrow::compute::concat;

        let spec = kernel_spec(json!([
            aggregate_output("count", "price", "price_count", 3),
            aggregate_output("mean", "price", "price_mean", 3),
            ddof_output("variance", "price", "price_var", 3, 1),
        ]));
        let compiled = compile_spec(&spec, &kernel_schema()).unwrap();
        let input = float64_fast_record(&[
            (1, "a", 1, Some(1.0)),
            (1, "b", 1, Some(10.0)),
            (2, "a", 2, None),
            (2, "b", 2, Some(20.0)),
            (3, "a", 3, Some(3.0)),
            (3, "b", 3, Some(30.0)),
        ]);
        let one_shot = compiled
            .kernel_plan
            .open_and_fill(&input, "rolling")
            .unwrap()
            .unwrap();
        let first = compiled
            .kernel_plan
            .open_and_fill(&input.slice(0, 4), "rolling")
            .unwrap()
            .unwrap();
        let second = compiled
            .kernel_plan
            .update_and_fill(&first.state, &input.slice(4, 2), "rolling")
            .unwrap()
            .unwrap();

        for ((expected, prefix), suffix) in one_shot
            .columns
            .iter()
            .zip(&first.columns)
            .zip(&second.columns)
        {
            let combined = concat(&[prefix.as_ref(), suffix.as_ref()]).unwrap();
            assert_eq!(combined.as_ref(), expected.as_ref());
        }
        assert_eq!(second.metrics.entities, 2);
        assert_eq!(second.metrics.scalar_value_conversions, 0);
    }

    #[test]
    fn ordered_float64_plan_falls_back_for_unsorted_input() {
        let spec = kernel_spec(json!([aggregate_output("mean", "price", "price_mean", 2)]));
        let compiled = compile_spec(&spec, &kernel_schema()).unwrap();
        let input = float64_fast_record(&[(2, "a", 2, Some(2.0)), (1, "a", 1, Some(1.0))]);

        assert!(
            compiled
                .kernel_plan
                .open_and_fill(&input, "rolling")
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn ordered_float64_plan_preserves_infinity_and_overflow_classification() {
        let spec = kernel_spec(json!([
            aggregate_output("sum", "price", "price_sum", 2),
            aggregate_output("mean", "price", "price_mean", 2),
            ddof_output("variance", "price", "price_var", 2, 1),
            ddof_output("stddev", "price", "price_std", 2, 0),
        ]));
        let compiled = compile_spec(&spec, &kernel_schema()).unwrap();
        let input = float64_fast_record(&[
            (1, "a", 1, Some(f64::INFINITY)),
            (2, "a", 2, Some(f64::NEG_INFINITY)),
            (3, "a", 3, Some(1.0)),
            (4, "a", 4, Some(f64::MAX)),
            (5, "a", 5, Some(f64::MAX)),
        ]);
        let fast = compiled
            .kernel_plan
            .open_and_fill(&input, "rolling")
            .unwrap()
            .unwrap();
        let rows = (0..input.num_rows())
            .map(|row_index| read_buffered_row(&input, row_index, &compiled, "rolling").unwrap())
            .collect::<Vec<_>>();
        let general =
            compute_output_columns(&rows, &RollingHistories::default(), &compiled, "rolling")
                .unwrap();

        for (fast, general) in fast.columns.iter().zip(&general.columns) {
            assert_eq!(fast.to_data(), general.to_data());
        }
    }

    #[test]
    fn duration_float64_plan_matches_the_general_kernel() {
        let spec = aggregate_spec(json!([duration_output("mean", "price", "price_mean", 10)]));
        let compiled = compile_spec(&spec, &kernel_schema()).unwrap();
        let input = float64_fast_record(&[
            (1, "a", 1, Some(1.0)),
            (5, "a", 2, Some(5.0)),
            (12, "a", 3, Some(12.0)),
            (12, "b", 1, Some(20.0)),
            (14, "a", 4, Some(14.0)),
        ]);
        let fast = compiled
            .kernel_plan
            .open_and_fill(&input, "rolling")
            .unwrap()
            .unwrap();
        let rows = (0..input.num_rows())
            .map(|row_index| read_buffered_row(&input, row_index, &compiled, "rolling").unwrap())
            .collect::<Vec<_>>();
        let general =
            compute_output_columns(&rows, &RollingHistories::default(), &compiled, "rolling")
                .unwrap();

        assert_eq!(
            compiled.kernel_plan.selection(),
            KernelSelection::OrderedPrimitive
        );
        assert_eq!(fast.columns[0].as_ref(), general.columns[0].as_ref());
    }

    #[test]
    fn float64_extrema_and_pair_groups_match_the_general_kernel() {
        let spec = aggregate_spec(json!([
            duration_output("mean", "price", "price_mean", 10),
            duration_output("min", "price", "price_min", 10),
            duration_output("max", "price", "price_max", 10),
            pair_output(
                "covariance",
                "price",
                "volume",
                "price_volume_cov",
                json!({"kind": "duration", "micros": 10}),
                1
            ),
            pair_output(
                "correlation",
                "price",
                "volume",
                "price_volume_corr",
                json!({"kind": "rows", "size": 3}),
                1
            ),
        ]));
        let schema = float64_pair_schema();
        let compiled = compile_spec(&spec, &schema).unwrap();
        let input = float64_pair_record(&[
            (1, "a", 1, Some(-0.0), Some(2.0)),
            (5, "a", 2, Some(5.0), Some(4.0)),
            (12, "a", 3, Some(3.0), Some(8.0)),
            (12, "b", 1, Some(20.0), Some(1.0)),
            (14, "a", 4, Some(7.0), Some(16.0)),
            (15, "a", 5, Some(f64::NAN), Some(32.0)),
        ]);
        let fast = compiled
            .kernel_plan
            .open_and_fill(&input, "rolling")
            .unwrap()
            .unwrap();
        let rows = (0..input.num_rows())
            .map(|row_index| read_buffered_row(&input, row_index, &compiled, "rolling").unwrap())
            .collect::<Vec<_>>();
        let general =
            compute_output_columns(&rows, &RollingHistories::default(), &compiled, "rolling")
                .unwrap();

        assert_eq!(fast.columns.len(), general.columns.len());
        for (fast, general) in fast.columns.iter().zip(&general.columns) {
            assert_eq!(fast.to_data(), general.to_data());
        }
    }

    #[test]
    fn int64_numeric_groups_keep_exact_sums_and_match_the_general_kernel() {
        let spec = kernel_spec(json!([
            aggregate_output("count", "volume", "volume_count", 2),
            aggregate_output("sum", "volume", "volume_sum", 2),
            aggregate_output("mean", "volume", "volume_mean", 2),
            ddof_output("variance", "volume", "volume_var", 2, 1),
        ]));
        let compiled = compile_spec(&spec, &kernel_schema()).unwrap();
        let input = int64_fast_record(&[Some(i64::MAX), Some(-1), None, Some(2)]);
        let fast = compiled
            .kernel_plan
            .open_and_fill(&input, "rolling")
            .unwrap()
            .unwrap();
        let rows = (0..input.num_rows())
            .map(|row_index| read_buffered_row(&input, row_index, &compiled, "rolling").unwrap())
            .collect::<Vec<_>>();
        let general =
            compute_output_columns(&rows, &RollingHistories::default(), &compiled, "rolling")
                .unwrap();

        assert_eq!(fast.columns.len(), general.columns.len());
        for (fast, general) in fast.columns.iter().zip(&general.columns) {
            assert_eq!(fast.to_data(), general.to_data());
        }
    }

    #[test]
    fn uint64_numeric_groups_keep_exact_sums_and_match_the_general_kernel() {
        let spec = kernel_spec(json!([
            aggregate_output("count", "sequence", "sequence_count", 3),
            aggregate_output("sum", "sequence", "sequence_sum", 3),
            aggregate_output("mean", "sequence", "sequence_mean", 3),
        ]));
        let compiled = compile_spec(&spec, &kernel_schema()).unwrap();
        let input = float64_fast_record(&[
            (1, "a", 1, None),
            (2, "a", 2, None),
            (3, "a", 3, None),
            (4, "a", 4, None),
        ]);
        let fast = compiled
            .kernel_plan
            .open_and_fill(&input, "rolling")
            .unwrap()
            .unwrap();
        let rows = (0..input.num_rows())
            .map(|row_index| read_buffered_row(&input, row_index, &compiled, "rolling").unwrap())
            .collect::<Vec<_>>();
        let general =
            compute_output_columns(&rows, &RollingHistories::default(), &compiled, "rolling")
                .unwrap();

        for (fast, general) in fast.columns.iter().zip(&general.columns) {
            assert_eq!(fast.to_data(), general.to_data());
        }
    }

    #[test]
    fn primitive_numeric_and_extrema_types_preserve_the_frozen_output_schema() {
        use datafusion::arrow::array::{Float32Array, Int8Array, UInt16Array};

        let spec = kernel_spec(json!([
            aggregate_output("count", "volume", "volume_count", 3),
            aggregate_output("sum", "volume", "volume_sum", 3),
            aggregate_output("mean", "volume", "volume_mean", 3),
            aggregate_output("min", "volume", "volume_min", 3),
            aggregate_output("max", "volume", "volume_max", 3),
        ]));
        let cases = [
            primitive_volume_record(Arc::new(Int8Array::from(vec![
                Some(-128),
                Some(127),
                None,
                Some(-1),
            ]))),
            primitive_volume_record(Arc::new(UInt16Array::from(vec![
                Some(65_535),
                Some(1),
                None,
                Some(4),
            ]))),
            primitive_volume_record(Arc::new(Float32Array::from(vec![
                Some(-0.0),
                Some(5.5),
                Some(f32::NAN),
                Some(-2.25),
            ]))),
        ];

        for (schema, input) in cases {
            assert_typed_matches_general(&spec, &schema, &input);
            let compiled = compile_spec(&spec, &schema).unwrap();
            assert_eq!(
                &compiled.outputs[3].output_type,
                schema.field(4).data_type(),
            );
            assert_eq!(
                &compiled.outputs[4].output_type,
                schema.field(4).data_type(),
            );
        }
    }

    #[test]
    fn typed_ewma_shares_one_recurrence_and_matches_the_general_kernel() {
        let spec = exponential_kernel_spec(json!([
            ewma_price(3, 1, "ema_ready"),
            ewma_price(3, 3, "ema_warm"),
        ]));
        let input = float64_fast_record(&[
            (1, "a", 1, Some(1.0)),
            (1, "b", 1, Some(10.0)),
            (2, "a", 2, None),
            (3, "a", 3, Some(3.0)),
            (4, "a", 4, Some(f64::NAN)),
            (5, "a", 5, Some(5.0)),
        ]);

        assert_typed_matches_general(&spec, &kernel_schema(), &input);
        let compiled = compile_spec(&spec, &kernel_schema()).unwrap();
        assert_eq!(compiled.window_groups.len(), 1);
        assert_eq!(
            compiled.kernel_plan.selection(),
            KernelSelection::OrderedPrimitive
        );
    }

    #[test]
    fn typed_ewma_resumes_from_a_columnar_checkpoint_without_replaying_history() {
        let spec = exponential_kernel_spec(json!([ewma_price(3, 1, "ema")]));
        let compiled = compile_spec(&spec, &kernel_schema()).unwrap();
        let output_schema = Arc::new(output_schema(&kernel_schema(), &compiled.outputs));
        let rows = vec![
            full_row(1, "a", 1, vec![ScalarValue::Float64(Some(10.0))]),
            full_row(2, "a", 2, vec![ScalarValue::Float64(Some(14.0))]),
            full_row(3, "a", 3, vec![ScalarValue::Float64(Some(18.0))]),
            full_row(4, "a", 4, vec![ScalarValue::Float64(Some(10.0))]),
        ];
        let expected =
            compute_output_columns(&rows, &RollingHistories::default(), &compiled, "rolling")
                .unwrap();

        let (first, _, touched) = build_typed_stream_output(
            &rows[..2],
            &RollingHistories::default(),
            None,
            &compiled,
            &output_schema,
            "rolling",
        )
        .unwrap()
        .unwrap();
        let mut histories = RollingHistories::default();
        histories.apply(touched);
        let bytes = encode_state_segment(
            &histories,
            &BTreeMap::new(),
            &kernel_schema(),
            &compiled,
            TEST_FINGERPRINT,
            "rolling",
        )
        .unwrap();
        let metadata = RollingSnapshotMetadata {
            state_layout_version: ROLLING_COLUMNAR_STATE_LAYOUT_VERSION,
            configuration_hash: compiled.configuration_hash.clone(),
            state_schema_fingerprint: compiled.state_schema_fingerprint.clone(),
            kernel_fingerprint: Some(compiled.kernel_plan.fingerprint().to_owned()),
            numerical_profile: Some(compiled.kernel_plan.numerical_profile().to_owned()),
            epoch: Epoch::new(1).unwrap(),
            pipeline_fingerprint: Some(TEST_FINGERPRINT.into()),
            operator_id: Some("rolling".into()),
            last_input_watermark: None,
            next_output_sequence: 0,
            ended: false,
            metrics: LateMetricDelta::default(),
            segment_inventory: Vec::new(),
        };
        let restored =
            decode_state_segment(&bytes, &kernel_schema(), &compiled, &metadata).unwrap();
        assert!(
            restored
                .histories
                .by_entity
                .values()
                .all(|state| state.rows.is_empty())
        );
        let (second, _, _) = build_typed_stream_output(
            &rows[2..],
            &restored.histories,
            None,
            &compiled,
            &output_schema,
            "rolling",
        )
        .unwrap()
        .unwrap();

        let actual = first
            .column(kernel_schema().fields().len())
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .iter()
            .chain(
                second
                    .column(kernel_schema().fields().len())
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap()
                    .iter(),
            )
            .collect::<Vec<_>>();
        assert_eq!(actual, float_column(&expected, 0));
    }

    #[test]
    fn stable_v2_transition_count_round_trips_in_columnar_state() {
        let mut spec = kernel_spec(json!([aggregate_output("mean", "price", "price_mean", 64)]));
        spec.numerical_profile = RollingNumericalProfile::StableV2Preview;
        let compiled = compile_spec(&spec, &kernel_schema()).unwrap();
        let rows = (0..64)
            .map(|index| {
                let sequence = u64::try_from(index + 1).unwrap();
                full_row(
                    i64::try_from(index + 1).unwrap(),
                    "a",
                    sequence,
                    vec![ScalarValue::Float64(Some(1.0e12 + f64::from(index)))],
                )
            })
            .collect::<Vec<_>>();
        let outputs =
            compute_output_columns(&rows, &RollingHistories::default(), &compiled, "rolling")
                .unwrap();
        let mut histories = RollingHistories::default();
        histories.apply(outputs.touched);
        let bytes = encode_state_segment(
            &histories,
            &BTreeMap::new(),
            &kernel_schema(),
            &compiled,
            TEST_FINGERPRINT,
            "rolling",
        )
        .unwrap();
        let metadata = RollingSnapshotMetadata {
            state_layout_version: ROLLING_COLUMNAR_STATE_LAYOUT_VERSION,
            configuration_hash: compiled.configuration_hash.clone(),
            state_schema_fingerprint: compiled.state_schema_fingerprint.clone(),
            kernel_fingerprint: Some(compiled.kernel_plan.fingerprint().to_owned()),
            numerical_profile: Some("stable_v2".into()),
            epoch: Epoch::new(1).unwrap(),
            pipeline_fingerprint: Some(TEST_FINGERPRINT.into()),
            operator_id: Some("rolling".into()),
            last_input_watermark: None,
            next_output_sequence: 0,
            ended: false,
            metrics: LateMetricDelta::default(),
            segment_inventory: Vec::new(),
        };

        let restored =
            decode_state_segment(&bytes, &kernel_schema(), &compiled, &metadata).unwrap();
        assert_eq!(
            restored
                .histories
                .by_entity
                .values()
                .next()
                .unwrap()
                .transition_count,
            64
        );
    }

    #[test]
    fn fused_dual_mean_writes_only_the_final_difference_column() {
        let spec = kernel_spec(json!([difference_output(
            &mean_leaf("price", 2),
            &mean_leaf("price", 4),
            "mean_spread"
        )]));
        let compiled = compile_spec(&spec, &kernel_schema()).unwrap();
        let input = float64_fast_record(&[
            (1, "a", 1, Some(1.0)),
            (2, "a", 2, Some(3.0)),
            (3, "a", 3, Some(5.0)),
            (4, "a", 4, Some(9.0)),
        ]);
        let fast = compiled
            .kernel_plan
            .open_and_fill(&input, "rolling")
            .unwrap()
            .unwrap();
        let rows = (0..input.num_rows())
            .map(|row_index| read_buffered_row(&input, row_index, &compiled, "rolling").unwrap())
            .collect::<Vec<_>>();
        let general =
            compute_output_columns(&rows, &RollingHistories::default(), &compiled, "rolling")
                .unwrap();

        assert_eq!(compiled.outputs.len(), 1);
        assert_eq!(compiled.window_groups.len(), 2);
        assert_eq!(fast.columns.len(), 1);
        assert_eq!(fast.columns[0].to_data(), general.columns[0].to_data());
        assert_eq!(
            fast.columns[0]
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            vec![Some(0.0), Some(0.0), Some(1.0), Some(2.5)]
        );
    }

    #[test]
    fn count_sum_and_mean_slide_over_each_entity_window() {
        let spec = kernel_spec(json!([
            aggregate_output("count", "price", "price_count", 2),
            aggregate_output("sum", "price", "price_sum", 2),
            aggregate_output("mean", "price", "price_mean", 2),
        ]));
        let rows = vec![
            full_row(1, "a", 1, vec![ScalarValue::Float64(Some(1.0))]),
            full_row(1, "b", 1, vec![ScalarValue::Float64(Some(10.0))]),
            full_row(2, "a", 2, vec![ScalarValue::Float64(Some(2.0))]),
            full_row(2, "b", 2, vec![ScalarValue::Float64(Some(20.0))]),
            full_row(3, "a", 3, vec![ScalarValue::Float64(Some(3.0))]),
            full_row(4, "a", 4, vec![ScalarValue::Float64(Some(4.0))]),
        ];
        let outputs = compute(&spec, &RollingHistories::default(), &rows).unwrap();
        assert_eq!(
            unsigned_column(&outputs, 0),
            vec![Some(1), Some(1), Some(2), Some(2), Some(2), Some(2)]
        );
        assert_eq!(
            float_column(&outputs, 1),
            vec![
                Some(1.0),
                Some(10.0),
                Some(3.0),
                Some(30.0),
                Some(5.0),
                Some(7.0)
            ]
        );
        assert_eq!(
            float_column(&outputs, 2),
            vec![
                Some(1.0),
                Some(10.0),
                Some(1.5),
                Some(15.0),
                Some(2.5),
                Some(3.5)
            ]
        );
    }

    #[test]
    fn null_and_nan_samples_are_excluded_but_rows_still_emit() {
        let spec = kernel_spec(json!([
            aggregate_output("count", "price", "price_count", 2),
            aggregate_output("sum", "price", "price_sum", 2),
            aggregate_output("mean", "price", "price_mean", 2),
        ]));
        let rows = price_rows(&[Some(1.0), None, Some(f64::NAN), Some(4.0)]);
        let outputs = compute(&spec, &RollingHistories::default(), &rows).unwrap();
        assert_eq!(
            unsigned_column(&outputs, 0),
            vec![Some(1), Some(1), None, Some(1)]
        );
        assert_eq!(
            float_column(&outputs, 1),
            vec![Some(1.0), Some(1.0), None, Some(4.0)]
        );
        let means = float_column(&outputs, 2);
        assert_eq!(means[0], Some(1.0));
        assert_eq!(means[1], Some(1.0));
        assert_eq!(means[2], None);
        assert_eq!(means[3], Some(4.0));
    }

    #[test]
    fn min_periods_counts_valid_samples_not_rows() {
        let mut declaration = aggregate_output("mean", "price", "price_mean", 3);
        declaration["min_periods"] = json!(2);
        let spec = kernel_spec(json!([declaration]));
        let rows = price_rows(&[Some(1.0), None, Some(3.0), Some(4.0)]);
        let outputs = compute(&spec, &RollingHistories::default(), &rows).unwrap();
        assert_eq!(
            float_column(&outputs, 0),
            vec![None, None, Some(2.0), Some(3.5)]
        );
    }

    #[test]
    fn variance_and_stddev_follow_the_ddof_divisor() {
        let spec = kernel_spec(json!([
            ddof_output("variance", "price", "price_var_1", 2, 1),
            ddof_output("variance", "price", "price_var_0", 2, 0),
            ddof_output("stddev", "price", "price_std_1", 2, 1),
            ddof_output("stddev", "price", "price_std_0", 2, 0),
        ]));
        let rows = price_rows(&[Some(1.0), Some(2.0), Some(3.0), Some(4.0)]);
        let outputs = compute(&spec, &RollingHistories::default(), &rows).unwrap();
        let sample = float_column(&outputs, 0);
        assert_eq!(sample[0], None);
        assert_eq!(sample[1..], [Some(0.5), Some(0.5), Some(0.5)]);
        assert_eq!(
            float_column(&outputs, 1),
            vec![Some(0.0), Some(0.25), Some(0.25), Some(0.25)]
        );
        let std_sample = float_column(&outputs, 2);
        assert_eq!(std_sample[0], None);
        for value in &std_sample[1..] {
            assert!((value.unwrap() - 0.5_f64.sqrt()).abs() < 1e-15);
        }
        let std_population = float_column(&outputs, 3);
        assert_eq!(std_population[0], Some(0.0));
        for value in &std_population[1..] {
            assert!((value.unwrap() - 0.5).abs() < 1e-15);
        }
    }

    #[test]
    fn integer_sum_is_exact_and_checked() {
        let spec = kernel_spec(json!([aggregate_output("sum", "volume", "volume_sum", 2)]));
        let rows = [10_i64, 20, 30]
            .into_iter()
            .enumerate()
            .map(|(index, volume)| {
                let sequence = u64::try_from(index + 1).unwrap();
                full_row(
                    i64::try_from(index + 1).unwrap(),
                    "a",
                    sequence,
                    vec![ScalarValue::Float64(None), ScalarValue::Int64(Some(volume))],
                )
            })
            .collect::<Vec<_>>();
        let outputs = compute(&spec, &RollingHistories::default(), &rows).unwrap();
        assert_eq!(
            signed_column(&outputs, 0),
            vec![Some(10), Some(30), Some(50)]
        );
    }

    #[test]
    fn integer_sum_overflow_is_a_data_error() {
        let spec = kernel_spec(json!([aggregate_output("sum", "volume", "volume_sum", 2)]));
        let rows = vec![
            full_row(
                1,
                "a",
                1,
                vec![
                    ScalarValue::Float64(None),
                    ScalarValue::Int64(Some(i64::MAX - 1)),
                ],
            ),
            full_row(
                2,
                "a",
                2,
                vec![ScalarValue::Float64(None), ScalarValue::Int64(Some(2))],
            ),
        ];
        let error = compute(&spec, &RollingHistories::default(), &rows).unwrap_err();
        assert!(
            matches!(error, CalcFlowError::Operator { ref message, .. } if message.contains("sum")),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn integer_slide_transient_overflow_returns_representable_sums() {
        // Window sums [MAX], [MAX,-1], [-1,5] are all representable, but the
        // add-before-remove slide transient MAX-1+5 overflows narrow i64.
        let spec = kernel_spec(json!([aggregate_output("sum", "volume", "volume_sum", 2)]));
        let rows = [i64::MAX, -1, 5]
            .into_iter()
            .enumerate()
            .map(|(index, volume)| {
                full_row(
                    i64::try_from(index + 1).unwrap(),
                    "a",
                    u64::try_from(index + 1).unwrap(),
                    vec![ScalarValue::Float64(None), ScalarValue::Int64(Some(volume))],
                )
            })
            .collect::<Vec<_>>();
        let outputs = compute(&spec, &RollingHistories::default(), &rows).unwrap();
        assert_eq!(
            signed_column(&outputs, 0),
            vec![Some(i64::MAX), Some(i64::MAX - 1), Some(4)]
        );
    }

    #[test]
    fn unsigned_slide_transient_overflow_returns_representable_sums() {
        let schema = numeric_schema(DataType::UInt64);
        let spec = numeric_spec(json!([aggregate_output("sum", "value", "value_sum", 2)]));
        let compiled = compile_spec(&spec, &schema).unwrap();
        let rows = [u64::MAX, 0, 5]
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                numeric_row(
                    i64::try_from(index + 1).unwrap(),
                    u64::try_from(index + 1).unwrap(),
                    ScalarValue::UInt64(Some(value)),
                )
            })
            .collect::<Vec<_>>();
        let outputs =
            compute_output_columns(&rows, &RollingHistories::default(), &compiled, "rolling")
                .unwrap();
        assert_eq!(
            unsigned_column(&outputs, 0),
            vec![Some(u64::MAX), Some(u64::MAX), Some(5)]
        );
    }

    #[test]
    fn integer_transient_slide_matches_the_rebuild_fold() {
        let spec = kernel_spec(json!([aggregate_output("sum", "volume", "volume_sum", 2)]));
        let rows = [i64::MAX, -1, 5, 2]
            .into_iter()
            .enumerate()
            .map(|(index, volume)| {
                full_row(
                    i64::try_from(index + 1).unwrap(),
                    "a",
                    u64::try_from(index + 1).unwrap(),
                    vec![ScalarValue::Float64(None), ScalarValue::Int64(Some(volume))],
                )
            })
            .collect::<Vec<_>>();
        let one_shot = compute(&spec, &RollingHistories::default(), &rows).unwrap();
        let mut histories = RollingHistories::default();
        let mut segmented: Vec<Option<i64>> = Vec::new();
        for chunk in rows.chunks(3) {
            let outputs = compute(&spec, &histories, chunk).unwrap();
            let values = signed_column(&outputs, 0);
            histories.apply(outputs.touched);
            segmented.extend(values);
        }
        assert_eq!(segmented, signed_column(&one_shot, 0));
    }

    fn numeric_schema(data_type: DataType) -> Schema {
        Schema::new(vec![
            Field::new(
                "ts",
                DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC"))),
                false,
            ),
            Field::new("symbol", DataType::Utf8, false),
            Field::new("sequence", DataType::UInt64, false),
            Field::new("value", data_type, true),
        ])
    }

    fn numeric_row(event_time: i64, sequence: u64, value: ScalarValue) -> BufferedRow {
        BufferedRow::new(
            vec![Some(KeyValue::String("a".into()))],
            vec![KeyValue::Unsigned(sequence)],
            event_time,
            vec![
                ts_scalar(event_time),
                ScalarValue::Utf8(Some("a".into())),
                ScalarValue::UInt64(Some(sequence)),
                value,
            ],
        )
    }

    fn numeric_spec(outputs: Value) -> RollingSpec {
        let mut document = aggregate_spec_json(outputs);
        document["partition_by"] = json!(["symbol"]);
        document["event_time"] = json!("ts");
        document["sequence_by"] = json!(["sequence"]);
        serde_json::from_value(document).unwrap()
    }

    #[test]
    fn unsigned_sum_stays_exact_and_checked() {
        let schema = numeric_schema(DataType::UInt64);
        let spec = numeric_spec(json!([aggregate_output("sum", "value", "value_sum", 2)]));
        let compiled = compile_spec(&spec, &schema).unwrap();
        let rows = [5_u64, 7, 9]
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                let sequence = u64::try_from(index + 1).unwrap();
                numeric_row(
                    i64::try_from(index + 1).unwrap(),
                    sequence,
                    ScalarValue::UInt64(Some(value)),
                )
            })
            .collect::<Vec<_>>();
        let outputs =
            compute_output_columns(&rows, &RollingHistories::default(), &compiled, "rolling")
                .unwrap();
        assert_eq!(
            unsigned_column(&outputs, 0),
            vec![Some(5), Some(12), Some(16)]
        );
        let overflow = vec![
            numeric_row(1, 1, ScalarValue::UInt64(Some(u64::MAX))),
            numeric_row(2, 2, ScalarValue::UInt64(Some(1))),
        ];
        let error = compute_output_columns(
            &overflow,
            &RollingHistories::default(),
            &compiled,
            "rolling",
        )
        .unwrap_err();
        assert!(
            matches!(error, CalcFlowError::Operator { ref message, .. } if message.contains("sum")),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn float32_samples_widen_to_float64_outputs() {
        let schema = numeric_schema(DataType::Float32);
        let spec = numeric_spec(json!([
            aggregate_output("sum", "value", "value_sum", 2),
            aggregate_output("mean", "value", "value_mean", 2),
        ]));
        let compiled = compile_spec(&spec, &schema).unwrap();
        let rows = [1.5_f32, 2.5, 4.0]
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                let sequence = u64::try_from(index + 1).unwrap();
                numeric_row(
                    i64::try_from(index + 1).unwrap(),
                    sequence,
                    ScalarValue::Float32(Some(value)),
                )
            })
            .collect::<Vec<_>>();
        let outputs =
            compute_output_columns(&rows, &RollingHistories::default(), &compiled, "rolling")
                .unwrap();
        assert_eq!(
            float_column(&outputs, 0),
            vec![Some(1.5), Some(4.0), Some(6.5)]
        );
        assert_eq!(
            float_column(&outputs, 1),
            vec![Some(1.5), Some(2.0), Some(3.25)]
        );
    }

    #[test]
    fn infinities_follow_ieee_and_undefined_results_are_nan_not_null() {
        let spec = kernel_spec(json!([
            aggregate_output("sum", "price", "price_sum", 2),
            ddof_output("variance", "price", "price_var", 2, 1),
        ]));
        let rows = price_rows(&[Some(1.0), Some(f64::INFINITY), Some(3.0), Some(4.0)]);
        let outputs = compute(&spec, &RollingHistories::default(), &rows).unwrap();
        let sums = float_column(&outputs, 0);
        assert_eq!(sums[0], Some(1.0));
        assert_eq!(sums[1], Some(f64::INFINITY));
        assert_eq!(sums[2], Some(f64::INFINITY));
        assert_eq!(sums[3], Some(7.0));
        let variances = float_column(&outputs, 1);
        assert_eq!(variances[0], None);
        assert!(variances[1].unwrap().is_nan());
        assert!(variances[2].unwrap().is_nan());
        assert_eq!(variances[3], Some(0.5));
    }

    // ------------------------------------------------------------------
    // Frozen ±inf readout semantics (SCE-07 defect 1 ruling, A1-A6)
    // ------------------------------------------------------------------

    /// Rows for one entity with explicit per-row price values.
    fn entity_prices(symbol: &str, prices: &[Option<f64>]) -> Vec<BufferedRow> {
        prices
            .iter()
            .enumerate()
            .map(|(index, price)| {
                let sequence = u64::try_from(index + 1).unwrap();
                full_row(
                    i64::try_from(index + 1).unwrap(),
                    symbol,
                    sequence,
                    vec![ScalarValue::Float64(*price)],
                )
            })
            .collect()
    }

    #[test]
    fn a1_mean_classification_is_independent_of_infinity_position() {
        let spec = kernel_spec(json!([aggregate_output("mean", "price", "price_mean", 3)]));
        let mut rows = Vec::new();
        for (symbol, prices) in [
            ("a", [f64::INFINITY, 1.0, 2.0]),
            ("b", [1.0, f64::INFINITY, 2.0]),
            ("c", [1.0, 2.0, f64::INFINITY]),
            ("d", [f64::NEG_INFINITY, 1.0, 2.0]),
            ("e", [1.0, f64::NEG_INFINITY, 2.0]),
            ("f", [1.0, 2.0, f64::NEG_INFINITY]),
        ] {
            rows.extend(entity_prices(symbol, &prices.map(Some)));
        }
        let outputs = compute(&spec, &RollingHistories::default(), &rows).unwrap();
        let means = float_column(&outputs, 0);
        for index in [2_usize, 5, 8] {
            assert_eq!(
                means[index],
                Some(f64::INFINITY),
                "positive infinity multiset at row {index}"
            );
        }
        for index in [11_usize, 14, 17] {
            assert_eq!(
                means[index],
                Some(f64::NEG_INFINITY),
                "negative infinity multiset at row {index}"
            );
        }
    }

    #[test]
    fn a2_mixed_sign_infinities_yield_nan_in_any_order() {
        let spec = kernel_spec(json!([aggregate_output("mean", "price", "price_mean", 3)]));
        let mut rows = Vec::new();
        rows.extend(entity_prices(
            "a",
            &[Some(f64::INFINITY), Some(f64::NEG_INFINITY)],
        ));
        rows.extend(entity_prices(
            "b",
            &[Some(f64::NEG_INFINITY), Some(f64::INFINITY)],
        ));
        rows.extend(entity_prices(
            "c",
            &[
                Some(f64::INFINITY),
                Some(f64::INFINITY),
                Some(f64::NEG_INFINITY),
            ],
        ));
        let outputs = compute(&spec, &RollingHistories::default(), &rows).unwrap();
        let means = float_column(&outputs, 0);
        for index in [1_usize, 3, 6] {
            assert!(
                means[index].is_some_and(f64::is_nan),
                "mixed-sign window at row {index} must be NaN"
            );
        }
    }

    #[test]
    fn a3_departed_infinities_leave_no_residue_across_slide_and_refold() {
        let spec = kernel_spec(json!([aggregate_output("mean", "price", "price_mean", 2)]));
        let rows = entity_prices(
            "a",
            &[
                Some(1.0),
                Some(f64::INFINITY),
                Some(3.0),
                Some(4.0),
                Some(5.0),
            ],
        );
        let one_shot = compute(&spec, &RollingHistories::default(), &rows).unwrap();
        assert_eq!(
            float_column(&one_shot, 0),
            vec![
                Some(1.0),
                Some(f64::INFINITY),
                Some(f64::INFINITY),
                Some(3.5),
                Some(4.5)
            ]
        );

        let mut histories = RollingHistories::default();
        let mut segmented: Vec<Option<f64>> = Vec::new();
        let compiled = compile_spec(&spec, &kernel_schema()).unwrap();
        for chunk in rows.chunks(3) {
            let outputs = compute(&spec, &histories, chunk).unwrap();
            let values = float_column(&outputs, 0);
            histories.apply(outputs.touched);
            // Mirror a checkpoint restore: rebuild every window from the
            // retained rows instead of carrying the live accumulators.
            rebuild_windows(&mut histories, &compiled, "rolling").unwrap();
            segmented.extend(values);
        }
        assert_eq!(segmented, float_column(&one_shot, 0));
    }

    #[test]
    fn negative_infinity_departure_restores_finite_window_outputs() {
        let spec = kernel_spec(json!([aggregate_output("mean", "price", "price_mean", 2)]));
        let rows = entity_prices("a", &[Some(f64::NEG_INFINITY), Some(2.0), Some(5.0)]);
        let outputs = compute(&spec, &RollingHistories::default(), &rows).unwrap();

        assert_eq!(
            float_column(&outputs, 0),
            vec![Some(f64::NEG_INFINITY), Some(f64::NEG_INFINITY), Some(3.5)]
        );
    }

    #[test]
    fn a4_near_overflow_finite_window_keeps_the_west_mean() {
        let spec = kernel_spec(json!([aggregate_output("mean", "price", "price_mean", 2)]));
        let rows = entity_prices("a", &[Some(1e308), Some(1e308)]);
        let outputs = compute(&spec, &RollingHistories::default(), &rows).unwrap();
        let means = float_column(&outputs, 0);
        let value = means[1].expect("finite window mean must be non-null");
        assert!(
            value.is_finite(),
            "naive sum/count readout would overflow to infinity"
        );
        assert!((value - 1e308).abs() <= 1e308 * 1e-15);
    }

    #[test]
    fn a5_variance_and_stddev_with_inf_are_nan_after_the_null_gates() {
        let spec = kernel_spec(json!([
            ddof_output("variance", "price", "price_var_1", 2, 1),
            ddof_output("stddev", "price", "price_std_0", 2, 0),
        ]));
        let rows = entity_prices("a", &[Some(f64::INFINITY), Some(2.0), Some(5.0), Some(6.0)]);
        let outputs = compute(&spec, &RollingHistories::default(), &rows).unwrap();
        let variances = float_column(&outputs, 0);
        // ddof=1 with one valid sample: divisor zero wins over the NaN class.
        assert_eq!(variances[0], None);
        // Two valid samples with an inf present: NaN, not null and not inf.
        assert!(variances[1].unwrap().is_nan());
        // The inf has left the window: back to finite values (the removal
        // step carries West drift well inside the frozen D13 tolerance).
        assert!((variances[2].unwrap() - 4.5).abs() <= 4.5 * 1e-10);
        assert!((variances[3].unwrap() - 0.5).abs() <= 1e-12);
        let stddevs = float_column(&outputs, 1);
        // ddof=0 passes the divisor gate with one sample: NaN classification.
        assert!(stddevs[0].unwrap().is_nan());
        assert!(stddevs[1].unwrap().is_nan());
        assert!((stddevs[2].unwrap() - 1.5_f64).abs() <= 1e-12);
        assert!((stddevs[3].unwrap() - 0.5_f64).abs() <= 1e-12);
    }

    #[test]
    fn a6_infinities_count_toward_valid_samples_and_min_periods() {
        let spec = kernel_spec(json!([
            aggregate_output("count", "price", "price_count", 3),
            aggregate_output("mean", "price", "price_mean", 3),
        ]));
        let rows = entity_prices("a", &[Some(f64::INFINITY), Some(f64::NAN), None]);
        let outputs = compute(&spec, &RollingHistories::default(), &rows).unwrap();
        // NaN and null are excluded, so the valid count stays at one; the
        // infinity alone still satisfies min_periods=1.
        assert_eq!(
            unsigned_column(&outputs, 0),
            vec![Some(1), Some(1), Some(1)]
        );
        let means = float_column(&outputs, 1);
        assert_eq!(means[0], Some(f64::INFINITY));
        assert_eq!(means[1], Some(f64::INFINITY));
        assert_eq!(means[2], Some(f64::INFINITY));
    }

    #[test]
    fn compatible_outputs_share_one_window_state() {
        let spec = kernel_spec(json!([
            aggregate_output("count", "price", "price_count", 2),
            aggregate_output("sum", "price", "price_sum", 2),
            aggregate_output("mean", "price", "price_mean", 2),
            ddof_output("variance", "price", "price_var", 2, 1),
            ddof_output("stddev", "price", "price_std", 2, 1),
            aggregate_output("mean", "price", "price_mean_3", 3),
        ]));
        let compiled = compile_spec(&spec, &kernel_schema()).unwrap();
        assert_eq!(compiled.window_groups.len(), 2);
        let rows = price_rows(&[Some(1.0), Some(2.0), Some(3.0), Some(4.0)]);
        let outputs = compute(&spec, &RollingHistories::default(), &rows).unwrap();
        let variances = float_column(&outputs, 3);
        let stddevs = float_column(&outputs, 4);
        let means_3 = float_column(&outputs, 5);
        for (variance, stddev) in variances.iter().zip(&stddevs) {
            match (variance, stddev) {
                (Some(variance), Some(stddev)) => {
                    assert!((stddev * stddev - variance).abs() < 1e-12);
                }
                (None, None) => {}
                other => panic!("variance/stddev nullness diverged: {other:?}"),
            }
        }
        assert_eq!(means_3, vec![Some(1.0), Some(1.5), Some(2.0), Some(3.0)]);
    }

    #[test]
    fn frame_size_extends_history_retention_beyond_lag_periods() {
        let spec = kernel_spec(json!([
            lag_price(1),
            aggregate_output("mean", "price", "price_mean", 3),
        ]));
        let mut histories = RollingHistories::default();
        for batch in 0..3_u32 {
            let rows = (0..4_u32)
                .map(|index| {
                    let sequence = batch * 4 + index + 1;
                    full_row(
                        i64::from(sequence),
                        "a",
                        u64::from(sequence),
                        vec![ScalarValue::Float64(Some(f64::from(sequence)))],
                    )
                })
                .collect::<Vec<_>>();
            let outputs = compute(&spec, &histories, &rows).unwrap();
            histories.apply(outputs.touched);
        }
        for state in histories.by_entity.values() {
            assert!(state.rows.len() <= 3);
            assert_eq!(state.rows.len(), 3);
        }
    }

    #[test]
    fn aggregate_windows_survive_segmentation() {
        let spec = kernel_spec(json!([
            aggregate_output("sum", "price", "price_sum", 2),
            ddof_output("variance", "price", "price_var", 2, 1),
        ]));
        let all_rows = price_rows(&[Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0)]);
        let one_shot = compute(&spec, &RollingHistories::default(), &all_rows).unwrap();
        let mut histories = RollingHistories::default();
        let mut segmented: Vec<Vec<Option<f64>>> = vec![Vec::new(); 2];
        for chunk in all_rows.chunks(2) {
            let outputs = compute(&spec, &histories, chunk).unwrap();
            histories.apply(outputs.touched);
            for (index, column) in outputs.columns.iter().enumerate() {
                let values = column
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap()
                    .iter()
                    .collect::<Vec<_>>();
                segmented[index].extend(values);
            }
        }
        for (index, column) in one_shot.columns.iter().enumerate() {
            let expected = column
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .iter()
                .collect::<Vec<_>>();
            assert_eq!(segmented[index], expected);
        }
    }

    #[test]
    fn failed_aggregate_leaves_histories_untouched() {
        let spec = kernel_spec(json!([aggregate_output("sum", "volume", "volume_sum", 2)]));
        let histories = RollingHistories::default();
        let rows = vec![
            full_row(
                1,
                "a",
                1,
                vec![
                    ScalarValue::Float64(None),
                    ScalarValue::Int64(Some(i64::MAX)),
                ],
            ),
            full_row(
                2,
                "a",
                2,
                vec![ScalarValue::Float64(None), ScalarValue::Int64(Some(1))],
            ),
        ];
        assert!(compute(&spec, &histories, &rows).is_err());
        assert!(histories.by_entity.is_empty());
    }

    // ------------------------------------------------------------------
    // Operator construction and metadata
    // ------------------------------------------------------------------

    #[test]
    fn operator_exposes_exact_ports_and_frozen_configuration() {
        let operator =
            RollingOperator::new("rolling_features", Arc::new(input_schema()), valid_spec())
                .unwrap();
        assert_eq!(operator.name(), "rolling_features");
        let [input] = operator.input_ports() else {
            panic!("rolling exposes one input port");
        };
        assert_eq!(input.name(), "input");
        assert!(input.required());
        assert_eq!(input.schema().unwrap().as_ref(), &input_schema());
        let [output] = operator.output_ports() else {
            panic!("rolling exposes one output port");
        };
        assert_eq!(output.name(), "output");
        assert!(output.required());
        assert_eq!(
            output.schema().unwrap().as_ref(),
            valid_spec().validate(&input_schema()).unwrap().as_ref()
        );
        assert_eq!(
            serde_json::to_value(operator.configuration()).unwrap(),
            json!({
                "kind": "rolling",
                "spec": valid_spec_json(),
            })
        );
    }

    #[test]
    fn spec_getter_returns_the_validated_declaration_and_debug_stays_non_exhaustive() {
        let spec = valid_spec();
        let operator =
            RollingOperator::new("rolling_features", Arc::new(input_schema()), spec.clone())
                .unwrap();
        assert_eq!(operator.spec(), &spec);
        let rendered = format!("{operator:?}");
        assert!(rendered.contains("RollingOperator"));
        assert!(rendered.contains("rolling_features"));
    }

    #[tokio::test]
    async fn emission_chunks_by_edge_budget_and_oversize_rows_fail() {
        use crate::{CancellationToken, EdgeBudget, IngressProgressSnapshot, StreamJobContext};

        struct NoopLateMetrics;
        impl crate::operator::LateMetricSink for NoopLateMetrics {
            fn record(&self, _delta: LateMetricDelta) -> Result<()> {
                Ok(())
            }
        }

        fn matrix_record(times: Vec<i64>, prices: Vec<Option<f64>>) -> RecordBatch {
            let len = times.len();
            RecordBatch::try_new(
                Arc::new(input_schema()),
                vec![
                    Arc::new(
                        datafusion::arrow::array::TimestampMicrosecondArray::from(times)
                            .with_timezone("UTC"),
                    ) as ArrayRef,
                    Arc::new(datafusion::arrow::array::StringArray::from(vec!["a"; len]))
                        as ArrayRef,
                    Arc::new(UInt64Array::from((1..=len as u64).collect::<Vec<_>>())),
                    Arc::new(Float64Array::from(prices)),
                    Arc::new(datafusion::arrow::array::Int64Array::from(
                        (1..=len as u64)
                            .map(|v| Some(i64::try_from(v).unwrap() * 10))
                            .collect::<Vec<_>>(),
                    )),
                    Arc::new(datafusion::arrow::array::StringArray::from(vec!["x"; len])),
                ],
            )
            .unwrap()
        }

        let job = StreamJobContext::new(
            7,
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            JsonMap::new(),
            None,
            CancellationToken::new(),
        );
        let budget = EdgeBudget::new(2, usize::MAX).unwrap();
        let context = StreamOperatorContext::for_task(
            &job,
            "rolling",
            None,
            IngressProgressSnapshot::default(),
            budget,
            Arc::new(NoopLateMetrics),
        );
        let mut operator =
            RollingOperator::new("rolling", Arc::new(input_schema()), valid_spec()).unwrap();
        let mut collector = crate::EdgeCollector::new(operator.output_ports().to_vec());
        let record = matrix_record(vec![10, 11, 12], vec![Some(1.0), Some(2.0), Some(3.0)]);
        let batch = Batch::table(vec![record], BatchMetadata::default()).unwrap();
        operator
            .process_data("input", batch, &context, &mut collector)
            .await
            .unwrap();
        operator
            .on_watermark(EventTime::from_micros(20), &context, &mut collector)
            .await
            .unwrap();
        let emitted = collector.drain("output");
        assert_eq!(emitted.len(), 2);
        assert_eq!(emitted[0].as_data().unwrap().metadata().sequence(), 0);
        assert_eq!(emitted[1].as_data().unwrap().metadata().sequence(), 1);

        let tiny = EdgeBudget::new(10, 1).unwrap();
        let context = StreamOperatorContext::for_task(
            &job,
            "rolling",
            None,
            IngressProgressSnapshot::default(),
            tiny,
            Arc::new(NoopLateMetrics),
        );
        let mut operator =
            RollingOperator::new("rolling", Arc::new(input_schema()), valid_spec()).unwrap();
        let mut collector = crate::EdgeCollector::new(operator.output_ports().to_vec());
        let record = matrix_record(vec![10], vec![Some(1.0)]);
        let batch = Batch::table(vec![record], BatchMetadata::default()).unwrap();
        operator
            .process_data("input", batch, &context, &mut collector)
            .await
            .unwrap();
        let error = operator
            .on_watermark(EventTime::from_micros(20), &context, &mut collector)
            .await
            .unwrap_err();
        assert!(matches!(
                error,
                CalcFlowError::InvalidArgument { ref field, .. } if field == "message.bytes"
        ));
    }
}
