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
    compute::{cast as arrow_cast, concat_batches},
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

use super::rolling_metrics::{RollingMetricsRecorder, RollingStage, RollingWork, RollingWorkBatch};
use super::{
    BatchOperator, BatchOperatorContext, LateMetricDelta, OperatorMetadata, StateBudget,
    StreamCollector, StreamOperator, StreamOperatorContext, accumulate_late_metrics,
    expression::required_input,
    late_output::{LateRowTally, record_late_row, reject_batch_mode},
    validate_operator_name,
};

mod budget;
mod checkpoint;
mod compile;
mod cumulative;
use cumulative::CumulativeAccumulator;
mod generated_kernel_manifest;
mod history;
use history::PreparedHistoryUpdate;
mod indicators;
pub(crate) use indicators::{IndicatorRollingRecord, RollingIoReservation};
mod kernel;
mod late;
mod ordered_stream;
mod state_codec;
mod state_v3;
pub(crate) mod sync;

use compile::{
    compile_aggregate, compile_spec, compile_spec_full, configuration, format_error,
    operator_error, output_schema, validate_arguments,
};
use state_codec::{
    buffered_row_from_values, build_output_record, build_typed_batch_output,
    build_typed_stream_output, chunk_output_record, closing_coordinate, decode_state_segment,
    deserialize_required_option, encode_state_segment, ewma_entity_from_values, read_buffered_row,
    read_buffered_rows, reconstruct_typed_state, snapshot_segments, sort_and_validate,
    state_schema, state_schema_fingerprint, validate_segment_schema_metadata,
};
#[cfg(test)]
use state_codec::{decode_ewma_state_row, encode_state_segment_legacy, rebuild_windows};

use super::checkpoint::{checkpoint_mismatch, compile_error, internal_error, state_format};
use budget::StateCharge;
#[cfg(test)]
use kernel::KernelSelection;
pub(crate) use kernel::entity_parallel::{
    ActualNumericWork, LaneProgress, LocatedPanic, NumericLaneExit, NumericLaneOutcome,
    NumericLaneRequest, NumericLaneStop, merge_numeric_results, run_numeric_lane,
};
#[cfg(test)]
pub(crate) use kernel::entity_parallel_tests::{
    entity_parallel_test_pair, entity_parallel_test_work,
};
use kernel::{RollingKernelPlan, RollingKernelState, TypedStreamTransition};
#[cfg(test)]
pub(crate) use kernel::{
    StreamKernelUpdate,
    entity_parallel::{NumericJoinSeed, ScratchPlan},
};

/// Semantic configuration version of the first rolling operator release.
pub const ROLLING_CONFIGURATION_VERSION: u32 = 1;
/// Durable state-layout version of the first rolling operator release.
pub const ROLLING_STATE_LAYOUT_VERSION: u32 = 1;
/// Durable state-layout version that persists exponential accumulators.
pub const ROLLING_EWMA_STATE_LAYOUT_VERSION: u32 = 2;
/// Durable columnar layout for declarations without cumulative summaries.
pub const ROLLING_COLUMNAR_STATE_LAYOUT_VERSION: u32 = 3;
/// Durable columnar layout with exact cumulative summaries.
pub const ROLLING_CUMULATIVE_STATE_LAYOUT_VERSION: u32 = 4;
/// Columnar rolling state with complete RSI and ATR recurrence records.
pub const ROLLING_INDICATOR_STATE_LAYOUT_VERSION: u32 = 5;
/// Columnar indicator state additionally admitting complete MACD recurrences.
pub const ROLLING_MACD_STATE_LAYOUT_VERSION: u32 = 6;
/// State-table layout with bit-identical typed accumulator continuation.
pub(crate) const ROLLING_STATE_TABLE_LAYOUT_VERSION: u32 = 7;

/// One SQL rolling window accepted by the crate-private `DataFusion` physical planner.
#[derive(Clone, Debug)]
pub(crate) struct DataFusionRollingWindow {
    pub input_index: usize,
    pub output_name: String,
    pub rows: u64,
    pub is_count: bool,
    pub is_sum: bool,
    pub duration_micros: Option<u64>,
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
            || windows
                .iter()
                .any(|window| window.is_count || window.is_sum || window.duration_micros.is_some())
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
            kernel::KernelOrder::new(
                ROLLING_STATE_LAYOUT_VERSION,
                event_time_index,
                physical_order,
                partition_indices.to_vec(),
                sequence_columns.iter().map(|column| column.index).collect(),
            ),
            &outputs,
            &groups,
            &[],
        )
        .with_nan_as_value();
        plan.supports_typed_transition().then_some(Self { plan })
    }

    #[cfg(test)]
    pub(crate) fn update_and_fill(
        &self,
        state: &DataFusionRollingState,
        input: &RecordBatch,
    ) -> Result<DataFusionRollingBatch> {
        self.update_and_fill_owned(state.clone(), input)
    }

    pub(crate) fn update_and_fill_owned(
        &self,
        state: DataFusionRollingState,
        input: &RecordBatch,
    ) -> Result<DataFusionRollingBatch> {
        let execution =
            self.plan
                .update_sorted_and_fill_owned(state.inner, input, "datafusion.rolling")?;
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

/// One readout of a shared complete MACD recurrence.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MacdComponent {
    /// Fast average minus slow average.
    Line,
    /// Exponential average of the line, including carried valid lines.
    Signal,
    /// Line minus signal.
    Histogram,
}
impl MacdComponent {
    const fn ordinal(self) -> usize {
        match self {
            Self::Line => 0,
            Self::Signal => 1,
            Self::Histogram => 2,
        }
    }
}
const fn default_macd_fast_span() -> u64 {
    12
}
const fn default_macd_slow_span() -> u64 {
    26
}
const fn default_macd_signal_span() -> u64 {
    9
}
const fn default_macd_minimum() -> u64 {
    1
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
    /// One nullable Float64 component of a shared unadjusted MACD recurrence.
    MacdComponent {
        /// Primitive version; must equal `1`.
        primitive_version: u32,
        /// Signed/unsigned integer or floating input column.
        input: String,
        /// Selected nullable Float64 output column.
        output: String,
        /// Positive fast span, strictly smaller than the slow span.
        #[serde(default = "default_macd_fast_span")]
        #[schemars(range(min = 1))]
        fast_span: u64,
        /// Positive slow span.
        #[serde(default = "default_macd_slow_span")]
        #[schemars(range(min = 1))]
        slow_span: u64,
        /// Independent positive signal span.
        #[serde(default = "default_macd_signal_span")]
        #[schemars(range(min = 1))]
        signal_span: u64,
        /// Independent positive fast/slow warm-up count, without a span cap.
        #[serde(default = "default_macd_minimum")]
        #[schemars(range(min = 1))]
        min_periods: u64,
        /// Independent positive signal warm-up count, without a span cap.
        #[serde(default = "default_macd_minimum")]
        #[schemars(range(min = 1))]
        signal_min_periods: u64,
        /// Line, signal or histogram readout; all share one state owner.
        component: MacdComponent,
    },
    /// Relative strength index with a complete-period Wilder seed.
    Rsi {
        /// Primitive version; must equal `1`.
        primitive_version: u32,
        /// Float32 or Float64 price column.
        input: String,
        /// Nullable Float64 output column.
        output: String,
        /// Positive number of seed differences and Wilder period.
        #[schemars(range(min = 1))]
        period: u64,
    },
    /// Average true range over floating high, low and close columns.
    Atr {
        /// Primitive version; must equal `1`.
        primitive_version: u32,
        /// Float32 or Float64 high column.
        high: String,
        /// Float32 or Float64 low column.
        low: String,
        /// Float32 or Float64 close column.
        close: String,
        /// Nullable Float64 output column.
        output: String,
        /// Positive number of seed true ranges and Wilder period.
        #[schemars(range(min = 1))]
        period: u64,
    },
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
    /// Checked sum of every valid numeric sample seen for an entity.
    CumulativeSum {
        /// Primitive version; must equal `1`.
        primitive_version: u32,
        /// Numeric input column name.
        input: String,
        /// Output column name.
        output: String,
        /// Positive minimum valid sample count.
        #[schemars(range(min = 1))]
        min_periods: u64,
    },
    /// Minimum of every valid numeric sample seen for an entity.
    CumulativeMin {
        /// Primitive version; must equal `1`.
        primitive_version: u32,
        /// Numeric input column name.
        input: String,
        /// Output column name.
        output: String,
        /// Positive minimum valid sample count.
        #[schemars(range(min = 1))]
        min_periods: u64,
    },
    /// Maximum of every valid numeric sample seen for an entity.
    CumulativeMax {
        /// Primitive version; must equal `1`.
        primitive_version: u32,
        /// Numeric input column name.
        input: String,
        /// Output column name.
        output: String,
        /// Positive minimum valid sample count.
        #[schemars(range(min = 1))]
        min_periods: u64,
    },
    /// Online variance of every valid numeric sample seen for an entity.
    CumulativeVariance {
        /// Primitive version; must equal `1`.
        primitive_version: u32,
        /// Numeric input column name.
        input: String,
        /// Output column name.
        output: String,
        /// Positive minimum valid sample count.
        #[schemars(range(min = 1))]
        min_periods: u64,
        /// Degrees-of-freedom adjustment; must be `0` or `1`.
        #[schemars(range(min = 0, max = 1))]
        ddof: u8,
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
            Self::MacdComponent {
                primitive_version, ..
            }
            | Self::Rsi {
                primitive_version, ..
            }
            | Self::Atr {
                primitive_version, ..
            }
            | Self::Lag {
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
            | Self::CumulativeSum {
                primitive_version, ..
            }
            | Self::CumulativeMin {
                primitive_version, ..
            }
            | Self::CumulativeMax {
                primitive_version, ..
            }
            | Self::CumulativeVariance {
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
            Self::MacdComponent { input, .. }
            | Self::Rsi { input, .. }
            | Self::Lag { input, .. }
            | Self::Delta { input, .. }
            | Self::Ewma { input, .. }
            | Self::CumulativeMean { input, .. }
            | Self::CumulativeSum { input, .. }
            | Self::CumulativeMin { input, .. }
            | Self::CumulativeMax { input, .. }
            | Self::CumulativeVariance { input, .. }
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
            Self::Atr { high, .. } => high,
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
            Self::MacdComponent { output, .. }
            | Self::Rsi { output, .. }
            | Self::Atr { output, .. }
            | Self::Lag { output, .. }
            | Self::Delta { output, .. }
            | Self::Ewma { output, .. }
            | Self::CumulativeMean { output, .. }
            | Self::CumulativeSum { output, .. }
            | Self::CumulativeMin { output, .. }
            | Self::CumulativeMax { output, .. }
            | Self::CumulativeVariance { output, .. }
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
            Self::MacdComponent { .. }
            | Self::Rsi { .. }
            | Self::Atr { .. }
            | Self::Ewma { .. }
            | Self::CumulativeMean { .. }
            | Self::CumulativeSum { .. }
            | Self::CumulativeMin { .. }
            | Self::CumulativeMax { .. }
            | Self::CumulativeVariance { .. } => 0,
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
            Self::MacdComponent { .. }
            | Self::Rsi { .. }
            | Self::Atr { .. }
            | Self::Lag { .. }
            | Self::Delta { .. }
            | Self::Ewma { .. }
            | Self::CumulativeMean { .. }
            | Self::CumulativeSum { .. }
            | Self::CumulativeMin { .. }
            | Self::CumulativeMax { .. }
            | Self::CumulativeVariance { .. } => None,
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
            Self::MacdComponent { .. }
            | Self::Rsi { .. }
            | Self::Atr { .. }
            | Self::Lag { .. }
            | Self::Delta { .. }
            | Self::Ewma { .. }
            | Self::CumulativeMean { .. }
            | Self::CumulativeSum { .. }
            | Self::CumulativeMin { .. }
            | Self::CumulativeMax { .. }
            | Self::CumulativeVariance { .. }
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
            Self::MacdComponent { .. }
            | Self::Rsi { .. }
            | Self::Atr { .. }
            | Self::Lag { .. }
            | Self::Delta { .. }
            | Self::Difference { .. } => None,
            Self::Ewma { min_periods, .. }
            | Self::CumulativeMean { min_periods, .. }
            | Self::CumulativeSum { min_periods, .. }
            | Self::CumulativeMin { min_periods, .. }
            | Self::CumulativeMax { min_periods, .. }
            | Self::CumulativeVariance { min_periods, .. }
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
            Self::CumulativeVariance { ddof, .. }
            | Self::Variance { ddof, .. }
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

    const fn requires_cumulative_layout(&self) -> bool {
        matches!(
            self,
            Self::CumulativeSum { .. }
                | Self::CumulativeMin { .. }
                | Self::CumulativeMax { .. }
                | Self::CumulativeVariance { .. }
        )
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
    /// Project-document declaration level for the rolling feature set. This is
    /// distinct from the durable checkpoint encoding. Existing primitives use
    /// [`ROLLING_STATE_LAYOUT_VERSION`]; EWMA requires
    /// [`ROLLING_EWMA_STATE_LAYOUT_VERSION`]. Cumulative sum/min/max/variance
    /// require [`ROLLING_CUMULATIVE_STATE_LAYOUT_VERSION`]. RSI and ATR require
    /// [`ROLLING_INDICATOR_STATE_LAYOUT_VERSION`]; MACD components require
    /// [`ROLLING_MACD_STATE_LAYOUT_VERSION`].
    pub state_layout_version: u32,
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
    compiled: Arc<CompiledRollingSpec>,
    state_layout_version: u32,
    state: indicators::StateOwner,
    state_budget: StateBudget,
    indicator_resources: Option<indicators::Resources>,
    indicator_capture: Option<indicators::Capture>,
    indicator_checkpoint_epoch: Option<Epoch>,
    indicator_late_failed: bool,
    checkpoint_workers: Option<Arc<crate::array::RollingWorkers>>,
    checkpoint_pending: Option<Arc<crate::array::RollingPending>>,
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
        let configuration = validated_operator_configuration(name, &spec)?;
        let compiled = Arc::new(compile_spec_full(&spec, &input_schema, &configuration)?);
        let output_schema = Arc::new(output_schema(&input_schema, &compiled.outputs));
        let output_ports =
            super::late_output::output_ports(spec.late_policy, &input_schema, output_schema)?;
        let (state, indicator_resources) = indicators::initial_state(&compiled, name)?;
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
            checkpoint_workers: indicators::workers_for_layout(
                compiled.state_layout_version,
                indicator_resources.as_ref(),
            ),
            checkpoint_pending: indicators::pending_for_layout(compiled.state_layout_version, name),
            state_layout_version: if compiled.kernel_plan.supports_slab() {
                ROLLING_STATE_TABLE_LAYOUT_VERSION
            } else {
                compiled.state_layout_version
            },
            compiled,
            state,
            state_budget: StateBudget::default(),
            indicator_resources,
            indicator_capture: None,
            indicator_checkpoint_epoch: None,
            indicator_late_failed: false,
        })
    }

    /// Sets retained-row and logical-byte limits for this operator.
    ///
    /// For operators containing RSI, ATR or MACD, the byte limit covers the
    /// participating peak: committed state, isolated candidates, prepared
    /// output, scratch and live checkpoint segments. Mixed old/new outputs
    /// share this whole-operator limit in finite and streaming evaluation.
    /// Reset retains credits held by unfinished workers and captures.
    ///
    /// Old-only operators retain their retained-state byte accounting and
    /// exclude checkpoint copies, even with explicit indicator layout 5 or 6.
    /// The default is 9,007,199,254,740,991 rows and logical bytes. Process
    /// resident memory, other operators and transferred results are separate.
    /// This runtime setting is not serialized into project configuration.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when retained rows or the
    /// applicable live byte charges exceed the requested budget. A rejected
    /// change preserves the previous limits.
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
        if let Some(resources) = &self.indicator_resources {
            resources.install_budget(self.state.charge.bytes, budget, &self.name)?;
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
        self.state_layout_version
    }

    pub(crate) fn native_checkpoint_workers(&self) -> Arc<crate::array::RollingWorkers> {
        Arc::clone(
            self.checkpoint_workers
                .as_ref()
                .expect("layout5 checkpoint workers"),
        )
    }

    fn indicator_state_packet(&self) -> Option<Arc<RollingStreamState>> {
        self.state.packet()
    }
}

fn validated_operator_configuration(name: &str, spec: &RollingSpec) -> Result<JsonMap> {
    validate_operator_name(name)?;
    validate_arguments(spec)?;
    configuration(spec)
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
                "kernel_fallback_reason",
                &self.compiled.kernel_plan.fallback_reason(),
            )
            .finish_non_exhaustive()
    }
}

impl Drop for RollingOperator {
    fn drop(&mut self) {
        self.indicator_capture = None;
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
        if self.indicator_resources.is_some() {
            return Ok(BTreeMap::from([(
                "output".into(),
                self.process_indicator_finite(input.clone(), context)
                    .await?,
            )]));
        }
        let output_schema = self.output_ports[0]
            .schema()
            .expect("rolling output always has an exact schema");
        let record = if let Some(record) = build_typed_batch_output(
            table,
            &self.compiled,
            output_schema,
            &self.name,
            self.state_budget,
        )? {
            record
        } else {
            let rows = read_buffered_rows(table, &self.compiled, &self.name)?;
            if !self.compiled.indicator_groups.is_empty() {
                for (index, row) in rows.iter().enumerate() {
                    indicators::validate_row(
                        &self.compiled.indicator_groups,
                        &row.values,
                        &self.name,
                        indicators::Coordinate::InputRow(index),
                    )
                    .map_err(|error| match error {
                        CalcFlowError::InvalidArgument { field, message } => {
                            operator_error(&self.name, &format!("{field}: {message}"))
                        }
                        other => other,
                    })?;
                }
            }
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

fn input_row_is_late(
    policy: LatePolicySpec,
    allowed_lateness_micros: u64,
    event_time: i64,
    watermark: Option<EventTime>,
    row_index: usize,
    node_id: &str,
) -> Result<bool> {
    let Some(watermark) = watermark else {
        return Ok(false);
    };
    let closing = closing_coordinate(event_time, allowed_lateness_micros, node_id)?;
    if closing > watermark.as_micros() {
        return Ok(false);
    }
    match policy {
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

/// Live stream state owned by one rolling operator task. Mutation is
/// confined to this value; input batches stay read-only.
#[derive(Clone, Default)]
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
    deferred_history: bool,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    typed_last_identity: Option<Vec<u8>>,
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

impl RollingOperator {
    fn validate_indicator_batch(&self, input: &TableBatch) -> Result<()> {
        if self.indicator_resources.is_some() {
            return Ok(());
        }
        for record in input.batches() {
            for group in &self.compiled.indicator_groups {
                match group {
                    indicators::Group::Rsi { input, field, .. } => {
                        let values = arrow_cast(record.column(*input), &DataType::Float64)
                            .map_err(|error| operator_error(&self.name, &error.to_string()))?;
                        let values = values
                            .as_any()
                            .downcast_ref::<Float64Array>()
                            .expect("indicator cast produces Float64");
                        for row in 0..values.len() {
                            crate::array::indicators::validate_rsi_sample(
                                values.is_valid(row).then_some(values.value(row)),
                                field,
                            )
                            .map_err(|error| {
                                indicators::numeric_error(
                                    error,
                                    &self.name,
                                    indicators::Coordinate::InputRow(row),
                                )
                            })?;
                        }
                    }
                    indicators::Group::Atr { inputs, fields, .. } => {
                        let columns = inputs.map(|index| {
                            arrow_cast(record.column(index), &DataType::Float64)
                                .map_err(|error| operator_error(&self.name, &error.to_string()))
                        });
                        let columns = columns
                            .into_iter()
                            .collect::<Result<Vec<_>>>()?
                            .into_iter()
                            .map(|array| {
                                array
                                    .as_any()
                                    .downcast_ref::<Float64Array>()
                                    .expect("indicator cast produces Float64")
                                    .clone()
                            })
                            .collect::<Vec<_>>();
                        let [high, low, close] = columns.as_slice() else {
                            unreachable!("ATR has three inputs")
                        };
                        for row in 0..high.len() {
                            crate::array::indicators::validate_atr_samples(
                                [
                                    high.is_valid(row).then_some(high.value(row)),
                                    low.is_valid(row).then_some(low.value(row)),
                                    close.is_valid(row).then_some(close.value(row)),
                                ],
                                fields.each_ref().map(String::as_str),
                            )
                            .map_err(|error| {
                                indicators::numeric_error(
                                    error,
                                    &self.name,
                                    indicators::Coordinate::InputRow(row),
                                )
                            })?;
                        }
                    }
                    indicators::Group::Macd(_) => {}
                }
            }
        }
        Ok(())
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
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        if self.indicator_resources.is_some() {
            context.check_cancelled()?;
        }
        let checkpoint_guard = self.try_native_checkpoint_guard(context.operator_id())?;
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
        self.validate_indicator_batch(batch.table_payload()?)?;
        let watermark = context.input_watermark();
        drop(validation);
        if self.indicator_resources.is_some() {
            return self
                .process_indicator_input_worker(
                    batch,
                    watermark,
                    context,
                    output,
                    checkpoint_guard.expect("indicator pending gate"),
                )
                .await;
        }
        if matches!(self.spec.late_policy, LatePolicySpec::SideOutput { .. }) {
            return self
                .process_late_data(&batch, watermark, context, output)
                .await;
        }
        if self.try_buffer_ordered(batch.table_payload()?, watermark, observer)? {
            self.install_context_identity(context)?;
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
            self.state.owned_mut()?.buffer.insert(identity, row);
        }
        context.record_window_metrics(
            metrics.late_rows,
            metrics.max_lateness_micros,
            metrics.null_event_time_rows,
        )?;
        self.state.owned_mut()?.metrics = next_metrics;
        self.state.owned_mut()?.charge = next_charge;
        self.install_context_identity(context)?;
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
        if self.indicator_resources.is_some() {
            return self
                .finalize_indicator_worker(Some(watermark), context, output)
                .await;
        }
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
                self.take_buffered(&closing)?
            };
            self.emit_rows(rows, context, output).await?;
        } else {
            let last_identity = self.state.ordered.last_identity();
            let records = {
                let _stage = context
                    .rolling_metrics()
                    .map(|recorder| recorder.stage(RollingStage::OrderingProof));
                self.state.owned_mut()?.ordered.take_closed(
                    watermark,
                    &self.compiled,
                    self.spec.allowed_lateness_micros,
                    context.operator_id(),
                )?
            };
            self.emit_ordered(records, last_identity, context, output)
                .await?;
        }
        self.install_context_identity(context)?;
        self.state.owned_mut()?.last_input_watermark = Some(watermark);
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
        if self.indicator_resources.is_some() {
            return self.finalize_indicator_worker(None, context, output).await;
        }
        if self.state.ordered.is_empty() {
            let rows = self.take_all_buffered()?;
            self.emit_rows(rows, context, output).await?;
        } else {
            let last_identity = self.state.ordered.last_identity();
            let records = self.state.owned_mut()?.ordered.take_all();
            self.emit_ordered(records, last_identity, context, output)
                .await?;
        }
        self.install_context_identity(context)?;
        self.state.owned_mut()?.ended = true;
        Ok(())
    }

    async fn prepare_checkpoint_async(
        &mut self,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        if self.indicator_resources.is_some() {
            return self.prepare_indicator_capture(context).await;
        }
        context.check_cancelled()
    }

    /// Captures the finality frontier, per-entity retained history, buffered
    /// unfinalized rows, and late metrics (SCE-00 D11) as one immutable base
    /// segment plus bounded inline metadata.
    fn checkpoint(&mut self, epoch: Epoch) -> Result<crate::OperatorStateSnapshot> {
        if self.indicator_resources.is_some() {
            return self.capture_prepared(epoch);
        }
        self.checkpoint_legacy(epoch)
    }

    /// Replaces the complete live state from one validated snapshot; a failed
    /// restore leaves the current state untouched (SCE-00 D11).
    fn restore(&mut self, snapshot: &crate::OperatorStateSnapshot) -> Result<()> {
        if self.indicator_resources.is_some() {
            return self.restore_indicators(snapshot);
        }
        self.restore_legacy(snapshot)
    }

    async fn restore_async(
        &mut self,
        snapshot: &crate::OperatorStateSnapshot,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        if self.indicator_resources.is_some() {
            return self.restore_indicators_async(snapshot, context).await;
        }
        context.check_cancelled()?;
        StreamOperator::restore(self, snapshot)
    }

    fn reset(&mut self) -> Result<()> {
        if let Some(resources) = &self.indicator_resources {
            let state = indicators::StateOwner::Shared(resources.empty_state());
            resources.cancel_pending();
            self.indicator_late_failed = false;
            self.indicator_capture = None;
            self.indicator_checkpoint_epoch = None;
            self.state = state;
            resources.reset_committed();
            return Ok(());
        }
        let _guard = self.try_native_checkpoint_guard(self.checkpoint_identity())?;
        self.state = indicators::StateOwner::new(RollingStreamState::default(), false);
        Ok(())
    }
}

struct EmissionPreparation {
    record: RecordBatch,
    next_kernel_state: Option<TypedStreamTransition>,
    touched: PreparedHistoryUpdate,
    current_charge: StateCharge,
    next_charge: StateCharge,
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
        touched: &mut PreparedHistoryUpdate,
        typed_change: Option<(StateCharge, StateCharge)>,
        node: &str,
        observer: Option<&RollingMetricsRecorder>,
    ) -> Result<(StateCharge, StateCharge)> {
        let current = self
            .state
            .charge
            .checked_sub(budget::buffered_charge(rows.iter(), node)?)?;
        let (old_histories, new_histories) =
            touched.changed_charge_observed(&self.state.histories, node, observer)?;
        let mut next = current
            .checked_sub(old_histories)?
            .checked_add(new_histories, node)?;
        if let Some((old_typed, new_typed)) = typed_change {
            next = next.checked_sub(old_typed)?.checked_add(new_typed, node)?;
        }
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
            .any(|state| state.columnar.has_columnar());
        let old = has_columnar
            .then(|| budget::histories_charge_observed(&self.state.histories, node, observer))
            .transpose()?;
        self.state
            .owned_mut()?
            .histories
            .materialize_columnar(&self.compiled, node, observer)?;
        if let Some(old) = old {
            let new = budget::histories_charge_observed(&self.state.histories, node, observer)?;
            self.state.owned_mut()?.charge =
                self.state.charge.checked_sub(old)?.checked_add(new, node)?;
        }
        Ok(())
    }

    fn observe_context(&self, context: &StreamOperatorContext<'_>) -> Result<()> {
        super::late_output::ensure_can_continue(
            self.state.late_output_failed || self.indicator_late_failed,
            context.operator_id(),
        )?;
        if self.state.deferred_history
            && context.job().checkpointing()
            && !self.compiled.kernel_plan.supports_slab()
        {
            return Err(operator_error(
                context.operator_id(),
                "checkpointing cannot be enabled after deferred rolling history was installed",
            ));
        }
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

    fn install_context_identity(&mut self, context: &StreamOperatorContext<'_>) -> Result<()> {
        self.state
            .owned_mut()?
            .pipeline_fingerprint
            .get_or_insert_with(|| context.job().fingerprint().to_owned());
        self.state
            .owned_mut()?
            .operator_id
            .get_or_insert_with(|| context.operator_id().to_owned());
        Ok(())
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
        let mut work = RollingWorkBatch::new(observer);
        let mut accepted = BTreeMap::new();
        let mut metrics = LateRowTally::default();
        for (row_index, row) in rows.into_iter().enumerate() {
            if self.is_late(row.identity.event_time, watermark, row_index, node_id)? {
                record_late_row(&mut metrics, watermark, row.identity.event_time, node_id)?;
                continue;
            }
            indicators::validate_row(
                &self.compiled.indicator_groups,
                &row.values,
                node_id,
                indicators::Coordinate::InputRow(row_index),
            )?;
            let _ordering = observer.map(|recorder| recorder.stage(RollingStage::OrderingProof));
            work.add(RollingWork::OrderProofRows, 1);
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
        input_row_is_late(
            self.spec.late_policy,
            self.spec.allowed_lateness_micros,
            event_time,
            watermark,
            row_index,
            node_id,
        )
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

    fn take_buffered(&mut self, keys: &[RowIdentity]) -> Result<Vec<BufferedRow>> {
        let state = self.state.owned_mut()?;
        Ok(keys
            .iter()
            .filter_map(|key| state.buffer.remove(key))
            .collect())
    }

    fn take_all_buffered(&mut self) -> Result<Vec<BufferedRow>> {
        Ok(std::mem::take(&mut self.state.owned_mut()?.buffer)
            .into_values()
            .collect())
    }

    fn restore_rows_after_charge_failure(
        &mut self,
        rows: Vec<BufferedRow>,
        node: &str,
        observer: Option<&RollingMetricsRecorder>,
    ) -> Result<()> {
        for row in rows {
            self.state
                .owned_mut()?
                .buffer
                .insert(row.identity.clone(), row);
        }
        if self.state.deferred_history {
            return Ok(());
        }
        if let Ok(charge) = budget::full_state_charge_observed(
            &self.state.buffer,
            &self.state.ordered,
            &self.state.histories,
            &self.compiled,
            node,
            observer,
        ) {
            self.state.owned_mut()?.charge = charge;
        }
        Ok(())
    }

    async fn emit_rows(
        &mut self,
        rows: Vec<BufferedRow>,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        if self.compiled.kernel_plan.supports_slab()
            && self
                .state
                .typed_kernel_state
                .as_deref()
                .is_none_or(|state| self.compiled.kernel_plan.can_use_slab(state))
        {
            return self.emit_slab_rows(rows, context, output).await;
        }
        let EmissionPreparation {
            record,
            next_kernel_state,
            touched,
            current_charge,
            next_charge,
        } = self.prepare_emission(rows, context)?;
        let observer = context.rolling_metrics();
        self.state.owned_mut()?.charge = current_charge;
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
        let state = self.state.owned_mut()?;
        for batch in batches {
            output.emit("output", batch).await?;
        }
        state.next_output_sequence = state
            .next_output_sequence
            .checked_add(chunk_count)
            .ok_or_else(|| operator_error(context.operator_id(), "output sequence overflowed"))?;
        let _stage = observer.map(|recorder| recorder.stage(RollingStage::HistoryMaintenance));
        touched.commit(&mut state.histories);
        TypedStreamTransition::replace(next_kernel_state, &mut state.typed_kernel_state);
        state.charge = next_charge;
        Ok(())
    }

    fn prepare_emission(
        &mut self,
        rows: Vec<BufferedRow>,
        context: &StreamOperatorContext<'_>,
    ) -> Result<EmissionPreparation> {
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
        let (record, next_kernel_state, mut touched) = if let Some(typed) = typed {
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
                PreparedHistoryUpdate::Replace(computed.touched),
            )
        };
        let typed_change = if self.state.deferred_history {
            let next = next_kernel_state.as_ref().ok_or_else(|| {
                internal_error("deferred general rolling update is missing its typed transition")
            })?;
            let current = self.state.typed_kernel_state.as_deref().ok_or_else(|| {
                internal_error("deferred general rolling update is missing its resident state")
            })?;
            Some(next.charge_change(current, context.operator_id())?)
        } else {
            None
        };
        let (current_charge, next_charge) = match self.projected_rows_charge(
            &rows,
            &mut touched,
            typed_change,
            context.operator_id(),
            observer,
        ) {
            Ok(charge) => charge,
            Err(error) => {
                self.restore_rows_after_charge_failure(rows, context.operator_id(), observer)?;
                return Err(error);
            }
        };
        Ok(EmissionPreparation {
            record,
            next_kernel_state,
            touched,
            current_charge,
            next_charge,
        })
    }
}

impl RollingOperator {
    fn encode_state(&self, epoch: Epoch) -> Result<Option<(String, Vec<u8>)>> {
        if self.state.deferred_history {
            let _pipeline_fingerprint =
                self.state.pipeline_fingerprint.clone().ok_or_else(|| {
                    internal_error("rolling state is missing its pipeline fingerprint")
                })?;
            let _operator_id =
                self.state.operator_id.clone().ok_or_else(|| {
                    internal_error("rolling state is missing its operator identity")
                })?;
            let typed = self
                .state
                .typed_kernel_state
                .as_deref()
                .ok_or_else(|| internal_error("rolling layout-7 typed state is absent"))?;
            let segment_id = format!("base-{:020}-00000000", epoch.as_u64());
            let bytes = RollingKernelPlan::encode_slab_state(typed)?;
            return Ok(Some((segment_id, bytes)));
        }
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
                                || matches!(window, WindowState::Cumulative(_))
                        })
                        .count()
                })
                .sum::<usize>();
        if row_count == 0 && self.compiled.indicator_groups.is_empty() {
            return Ok(None);
        }
        if self.state.histories.by_entity.is_empty() && self.state.buffer.is_empty() {
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

    fn encode_layout7_pending(&self, epoch: Epoch) -> Result<Option<(String, Vec<u8>)>> {
        if !self.state.deferred_history {
            return Ok(None);
        }
        let mut records = self.state.ordered.records().cloned().collect::<Vec<_>>();
        if !self.state.buffer.is_empty() {
            let rows = self.state.buffer.values().cloned().collect::<Vec<_>>();
            records.push(self.slab_input_record(&rows, &self.name)?);
        }
        if records.is_empty() {
            return Ok(None);
        }
        let record = if records.len() == 1 {
            records[0].clone()
        } else {
            concat_batches(&records[0].schema(), &records)
                .map_err(|error| operator_error(&self.name, &error.to_string()))?
        };
        let mut bytes = Vec::new();
        let mut writer = FileWriter::try_new(&mut bytes, record.schema().as_ref())
            .map_err(|error| operator_error(&self.name, &error.to_string()))?;
        writer
            .write(&record)
            .map_err(|error| operator_error(&self.name, &error.to_string()))?;
        writer
            .finish()
            .map_err(|error| operator_error(&self.name, &error.to_string()))?;
        drop(writer);
        Ok(Some((
            format!("pending-{:020}-00000001", epoch.as_u64()),
            bytes,
        )))
    }

    fn decode_layout7_pending(&self, bytes: &[u8]) -> Result<Vec<RecordBatch>> {
        let mut reader = FileReader::try_new(Cursor::new(bytes), None)
            .map_err(|error| operator_error(&self.name, &error.to_string()))?;
        let mut records = Vec::new();
        while let Some(record) = reader
            .next()
            .transpose()
            .map_err(|error| operator_error(&self.name, &error.to_string()))?
        {
            if record.schema().as_ref()
                != self.input_ports[0]
                    .schema()
                    .expect("rolling input schema")
                    .as_ref()
            {
                return Err(checkpoint_mismatch(
                    "rolling layout-7 pending schema does not match the input",
                ));
            }
            records.push(record);
        }
        Ok(records)
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
            kind: if segment_id.starts_with("pending-") {
                SegmentKind::Delta
            } else {
                SegmentKind::Base
            },
            state_layout_version: if self.state.deferred_history {
                ROLLING_STATE_TABLE_LAYOUT_VERSION
            } else {
                self.compiled.state_layout_version
            },
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
        let mut segments = segments.into_iter();
        if metadata.state_layout_version == ROLLING_STATE_TABLE_LAYOUT_VERSION {
            let bytes = segments.next().ok_or_else(|| {
                checkpoint_mismatch("rolling layout-7 snapshot is missing its state-table segment")
            })?;
            let typed = self
                .compiled
                .kernel_plan
                .decode_slab_state(&bytes, metadata.typed_last_identity.clone())?;
            let mut ordered = ordered_stream::OrderedStreamBuffer::default();
            if let Some(pending) = segments.next() {
                let records = self.decode_layout7_pending(&pending)?;
                let last_identity = records
                    .last()
                    .map(|record| {
                        self.compiled
                            .kernel_plan
                            .ordered_stream_bounds(record, &self.name, None)
                    })
                    .transpose()?
                    .flatten()
                    .map(|(_, end)| end);
                ordered.restore_pending(records, last_identity);
            }
            return Ok(DecodedRollingState {
                ordered,
                typed_kernel_state: Some(Box::new(typed)),
                deferred_history: true,
                ..DecodedRollingState::default()
            });
        }
        let Some(bytes) = segments.next() else {
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
    ordered: ordered_stream::OrderedStreamBuffer,
    histories: RollingHistories,
    typed_kernel_state: Option<Box<RollingKernelState>>,
    deferred_history: bool,
}

/// Serialized state-row ordering key: row kind, entity, identity, and the
/// per-entity history position.
type StateRowOrderKey = (u8, Vec<Option<KeyValue>>, RowIdentity, Option<u64>);

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
    indicator_groups: Vec<indicators::Group>,
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
    Indicator(usize),
    Lag { periods: u64 },
    Delta { periods: u64 },
    Ewma(CompiledEwma),
    Cumulative(CompiledAggregate),
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
    Cumulative {
        input_index: usize,
        statistic: Statistic,
    },
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
    indicators: Vec<indicators::State>,
    charge_cache: Option<budget::HistoryCharge>,
}

impl EntityRollingState {
    fn fresh(compiled: &CompiledRollingSpec) -> Self {
        Self {
            rows: VecDeque::new(),
            columnar: ordered_stream::ColumnarHistory::default(),
            windows: fresh_windows(compiled),
            transition_count: 0,
            indicators: compiled
                .indicator_groups
                .iter()
                .map(indicators::Group::fresh)
                .collect(),
            charge_cache: None,
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
    Cumulative(CumulativeAccumulator),
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
            CompiledWindowGroup::Cumulative { .. } => {
                WindowState::Cumulative(CumulativeAccumulator::default())
            }
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
        for (entity, mut state) in touched {
            state.charge_cache = None;
            self.by_entity.insert(entity, state);
        }
    }
}

/// Reversible sliding-window state. Integer sums remain exact; all floating
/// mean and dispersion arithmetic is owned by the shared compensated families.
#[derive(Clone, Copy, Debug)]
struct WindowAccumulator {
    sum: Option<SumState>,
    numeric: crate::accumulate::NumericAccumulator,
    /// Duration frames only: event-time bound (exclusive) whose rows this
    /// accumulator has already expired, kept so each slide removes only the
    /// newly expired prefix (SCE-08). `i128::MIN` while nothing expired.
    expired_through: i128,
}

impl std::ops::Deref for WindowAccumulator {
    type Target = crate::accumulate::NumericAccumulator;

    fn deref(&self) -> &Self::Target {
        &self.numeric
    }
}

impl std::ops::DerefMut for WindowAccumulator {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.numeric
    }
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
        Self::with_dispersion(sum_class, true)
    }

    fn with_dispersion(sum_class: SumClass, dispersion: bool) -> Self {
        let sum = match sum_class {
            SumClass::Signed => Some(SumState::Signed(0)),
            SumClass::Unsigned => Some(SumState::Unsigned(0)),
            SumClass::Float => Some(SumState::Float(0.0)),
            SumClass::CountOnly => None,
        };
        Self {
            sum,
            numeric: if dispersion {
                crate::accumulate::NumericAccumulator::default()
            } else {
                crate::accumulate::NumericAccumulator::mean_only()
            },
            expired_through: i128::MIN,
        }
    }

    /// Adds one valid sample. Null values never reach this method; the typed
    /// sticky-AVG plan deliberately passes NaN through.
    fn add(&mut self, value: &ScalarValue, node_id: &str) -> Result<()> {
        match &mut self.sum {
            Some(SumState::Signed(total)) => {
                *total = total
                    .checked_add(i128::from(signed_sample(value)))
                    .ok_or_else(|| operator_error(node_id, "rolling integer sum overflowed"))?;
                self.numeric.insert(float_sample(value))
            }
            Some(SumState::Unsigned(total)) => {
                *total = total
                    .checked_add(u128::from(unsigned_sample(value)))
                    .ok_or_else(|| operator_error(node_id, "rolling integer sum overflowed"))?;
                self.numeric.insert(float_sample(value))
            }
            Some(SumState::Float(total)) => {
                if let Err(error) = self.numeric.insert(float_sample(value)) {
                    return Err(kernel::float_arithmetic_error(error, node_id));
                }
                *total = self.numeric.sum();
                Ok(())
            }
            None => self.numeric.insert_count_only(),
        }
        .map_err(|error| kernel::float_arithmetic_error(error, node_id))
    }

    /// Removes one previously added valid sample from its compensated families.
    fn remove(&mut self, value: &ScalarValue) -> Result<()> {
        let result = match &mut self.sum {
            Some(SumState::Signed(total)) => {
                *total = total
                    .checked_sub(i128::from(signed_sample(value)))
                    .ok_or_else(|| internal_error("rolling sum removal diverged from the adds"))?;
                self.numeric.evict(float_sample(value))
            }
            Some(SumState::Unsigned(total)) => {
                *total = total
                    .checked_sub(u128::from(unsigned_sample(value)))
                    .ok_or_else(|| internal_error("rolling sum removal diverged from the adds"))?;
                self.numeric.evict(float_sample(value))
            }
            Some(SumState::Float(total)) => {
                if let Err(error) = self.numeric.evict(float_sample(value)) {
                    return Err(kernel::float_arithmetic_error(error, ""));
                }
                *total = self.numeric.sum();
                Ok(())
            }
            None => self.numeric.evict_count_only(),
        };
        result.map_err(|error| kernel::float_arithmetic_error(error, ""))
    }

    fn rows(
        &mut self,
        leaving: Option<&ScalarValue>,
        entering: Option<&ScalarValue>,
        node_id: &str,
    ) -> Result<()> {
        if self.sum.is_none() {
            if leaving.is_some() {
                self.numeric.evict_count_only()
            } else {
                Ok(())
            }
            .map_err(|error| kernel::float_arithmetic_error(error, node_id))?;
            if entering.is_some() {
                self.numeric.insert_count_only()
            } else {
                Ok(())
            }
            .map_err(|error| kernel::float_arithmetic_error(error, node_id))?;
            return Ok(());
        }
        if let Some(value) = leaving {
            self.remove_exact_scalar(value)?;
        }
        if let Some(value) = entering {
            self.add_exact_scalar(value, node_id)?;
        }
        self.numeric
            .rows(leaving.map(float_sample), entering.map(float_sample))
            .map_err(|error| kernel::float_arithmetic_error(error, node_id))?;
        self.sync_float_sum();
        Ok(())
    }

    fn add_exact_scalar(&mut self, value: &ScalarValue, node_id: &str) -> Result<()> {
        match &mut self.sum {
            Some(SumState::Signed(total)) => {
                *total = total
                    .checked_add(i128::from(signed_sample(value)))
                    .ok_or_else(|| operator_error(node_id, "rolling integer sum overflowed"))?;
            }
            Some(SumState::Unsigned(total)) => {
                *total = total
                    .checked_add(u128::from(unsigned_sample(value)))
                    .ok_or_else(|| operator_error(node_id, "rolling integer sum overflowed"))?;
            }
            _ => {}
        }
        Ok(())
    }

    fn remove_exact_scalar(&mut self, value: &ScalarValue) -> Result<()> {
        match &mut self.sum {
            Some(SumState::Signed(total)) => {
                *total = total
                    .checked_sub(i128::from(signed_sample(value)))
                    .ok_or_else(|| internal_error("rolling sum removal diverged from the adds"))?;
            }
            Some(SumState::Unsigned(total)) => {
                *total = total
                    .checked_sub(u128::from(unsigned_sample(value)))
                    .ok_or_else(|| internal_error("rolling sum removal diverged from the adds"))?;
            }
            _ => {}
        }
        Ok(())
    }

    fn sync_float_sum(&mut self) {
        if let Some(SumState::Float(total)) = &mut self.sum {
            *total = self.numeric.sum();
        }
    }

    #[allow(
        clippy::cast_precision_loss,
        reason = "integer mean divides its exact numerator only at readout"
    )]
    fn mean_value(&self) -> f64 {
        match self.sum {
            Some(SumState::Signed(total)) => total as f64 / self.valid_count as f64,
            Some(SumState::Unsigned(total)) => total as f64 / self.valid_count as f64,
            Some(SumState::Float(_)) | None => self.numeric.mean(),
        }
    }
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
        match self.add_checked(key, value, &mut || Ok::<_, std::convert::Infallible>(())) {
            Ok(()) => {}
            Err(never) => match never {},
        }
    }

    fn add_checked<E>(
        &mut self,
        key: ExtremaKey,
        value: ScalarValue,
        check: &mut dyn FnMut() -> std::result::Result<(), E>,
    ) -> std::result::Result<(), E> {
        check()?;
        self.valid_count = self.valid_count.saturating_add(1);
        while let Some((_, back)) = self.queue.back() {
            check()?;
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
        Ok(())
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
        match self
            .expire_through_key_checked(leaving, &mut || Ok::<_, std::convert::Infallible>(()))
        {
            Ok(()) => {}
            Err(never) => match never {},
        }
    }

    fn expire_through_key_checked<E>(
        &mut self,
        leaving: &ExtremaKey,
        check: &mut dyn FnMut() -> std::result::Result<(), E>,
    ) -> std::result::Result<(), E> {
        check()?;
        while self.queue.front().is_some_and(|(key, _)| key <= leaving) {
            check()?;
            self.queue.pop_front();
        }
        Ok(())
    }

    /// Duration frame expiry: candidates at or before `bound` (exclusive)
    /// have left the window.
    fn expire_through_time(&mut self, bound: i128) {
        match self.expire_through_time_checked(bound, &mut || Ok::<_, std::convert::Infallible>(()))
        {
            Ok(()) => {}
            Err(never) => match never {},
        }
    }

    fn expire_through_time_checked<E>(
        &mut self,
        bound: i128,
        check: &mut dyn FnMut() -> std::result::Result<(), E>,
    ) -> std::result::Result<(), E> {
        check()?;
        while self
            .queue
            .front()
            .is_some_and(|(key, _)| i128::from(key.event_time) <= bound)
        {
            check()?;
            self.queue.pop_front();
        }
        Ok(())
    }

    fn extremum(&self) -> Option<&ScalarValue> {
        self.queue.front().map(|(_, value)| value)
    }
}

/// Pairwise-valid shifted compensated co-moments. Infinity counters remain
/// outside the finite family and make every pair readout NaN.
#[derive(Clone, Copy, Debug)]
struct PairAccumulator {
    valid_count: u64,
    finite_count: u64,
    family: crate::accumulate::PairFamily,
    since_rebuild: u32,
    n_prev: u64,
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
            finite_count: 0,
            family: crate::accumulate::PairFamily::default(),
            since_rebuild: 0,
            n_prev: 64,
            pos_inf_x: 0,
            neg_inf_x: 0,
            pos_inf_y: 0,
            neg_inf_y: 0,
            expired_through: i128::MIN,
        }
    }
}

impl PairAccumulator {
    /// Adds one pairwise-valid sample to the shifted compensated family.
    fn add(&mut self, x: &ScalarValue, y: &ScalarValue, node_id: &str) -> Result<()> {
        self.add_values(float_sample(x), float_sample(y), node_id)
    }

    fn add_values(&mut self, sample_x: f64, sample_y: f64, node_id: &str) -> Result<()> {
        self.valid_count = self
            .valid_count
            .checked_add(1)
            .ok_or_else(|| operator_error(node_id, "rolling pair sample count overflowed"))?;
        self.update_infinities(sample_x, sample_y, true);
        if sample_x.is_finite() && sample_y.is_finite() {
            self.finite_count = self
                .finite_count
                .checked_add(1)
                .ok_or_else(|| operator_error(node_id, "rolling finite pair count overflowed"))?;
            self.family.add(sample_x, sample_y);
        }
        Ok(())
    }

    /// Removes one previously added pairwise-valid sample.
    fn remove(&mut self, x: &ScalarValue, y: &ScalarValue) -> Result<()> {
        self.remove_values(float_sample(x), float_sample(y))
    }

    fn remove_values(&mut self, sample_x: f64, sample_y: f64) -> Result<()> {
        self.valid_count = self
            .valid_count
            .checked_sub(1)
            .ok_or_else(|| internal_error("rolling pair removal without a matching add"))?;
        self.update_infinities(sample_x, sample_y, false);
        if sample_x.is_finite() && sample_y.is_finite() {
            self.finite_count = self
                .finite_count
                .checked_sub(1)
                .ok_or_else(|| internal_error("rolling finite pair removal without an add"))?;
            self.family.remove(sample_x, sample_y);
        }
        Ok(())
    }

    fn rows_values(
        &mut self,
        leaving: Option<(f64, f64)>,
        entering: Option<(f64, f64)>,
        node_id: &str,
    ) -> Result<()> {
        let paired = leaving.is_some_and(|(x, y)| x.is_finite() && y.is_finite())
            && entering.is_some_and(|(x, y)| x.is_finite() && y.is_finite())
            && self.finite_count > 1;
        if paired {
            self.family.pair(entering.unwrap(), leaving.unwrap());
            return Ok(());
        }
        if let Some((x, y)) = leaving {
            self.remove_values(x, y)?;
        }
        self.reset_if_empty();
        if let Some((x, y)) = entering {
            self.add_values(x, y, node_id)?;
        }
        Ok(())
    }

    fn finish_rows(
        &mut self,
        window: usize,
        history: impl Iterator<Item = Option<(f64, f64)>> + Clone,
    ) -> Result<()> {
        let cadence = u64::try_from(window.max(64)).unwrap_or(u64::MAX);
        self.finish(cadence, history)
    }

    fn finish_duration(
        &mut self,
        history: impl Iterator<Item = Option<(f64, f64)>> + Clone,
    ) -> Result<()> {
        let cadence = self.n_prev.max(64);
        self.finish(cadence, history)
    }

    fn finish(
        &mut self,
        cadence: u64,
        history: impl Iterator<Item = Option<(f64, f64)>> + Clone,
    ) -> Result<()> {
        self.since_rebuild = self
            .since_rebuild
            .checked_add(1)
            .ok_or_else(|| internal_error("rolling pair rebuild counter overflowed"))?;
        let retained = history.clone().count();
        let finite = history.filter_map(|sample| {
            sample.filter(|(left, right)| left.is_finite() && right.is_finite())
        });
        if u64::from(self.since_rebuild) >= cadence {
            self.family.rebuild(finite.clone());
            self.since_rebuild = 0;
            self.n_prev = u64::try_from(retained).unwrap_or(u64::MAX);
        } else if self.family.is_non_finite() {
            self.family.rebuild(finite);
        }
        Ok(())
    }

    fn update_infinities(&mut self, left: f64, right: f64, add: bool) {
        update_pair_infinity(self, left, true, add);
        update_pair_infinity(self, right, false, add);
    }

    fn reset_if_empty(&mut self) {
        if self.finite_count == 0 {
            self.family = crate::accumulate::PairFamily::default();
            self.since_rebuild = 0;
            self.n_prev = 64;
        }
    }

    fn holds_infinity(&self) -> bool {
        self.pos_inf_x > 0 || self.neg_inf_x > 0 || self.pos_inf_y > 0 || self.neg_inf_y > 0
    }
}

fn update_pair_infinity(accumulator: &mut PairAccumulator, value: f64, left: bool, add: bool) {
    if !value.is_infinite() {
        return;
    }
    let positive = match (left, value > 0.0) {
        (true, true) => &mut accumulator.pos_inf_x,
        (true, false) => &mut accumulator.neg_inf_x,
        (false, true) => &mut accumulator.pos_inf_y,
        (false, false) => &mut accumulator.neg_inf_y,
    };
    *positive = if add {
        positive.saturating_add(1)
    } else {
        positive.saturating_sub(1)
    };
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
    let mut work = RollingWorkBatch::new(observer);
    work.add(RollingWork::NumericGeneralCalls, 1);
    let cells = compute_output_cells_checked(
        rows,
        histories,
        compiled,
        node_id,
        observer,
        Some(&mut work),
        &mut || Ok(()),
    )?;
    let _stage = observer.map(|recorder| recorder.stage(RollingStage::ArrowOutput));
    let columns = encode_derived_columns(cells.columns, compiled, node_id)?;
    Ok(ComputedOutputs {
        columns,
        touched: cells.touched,
    })
}

fn compute_output_cells_checked(
    rows: &[BufferedRow],
    histories: &RollingHistories,
    compiled: &CompiledRollingSpec,
    node: &str,
    observer: Option<&RollingMetricsRecorder>,
    preparations: Option<&mut RollingWorkBatch<'_>>,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<indicators::ComputedCells> {
    indicators::compute_cells(
        rows,
        histories,
        compiled,
        node,
        observer,
        preparations,
        check,
    )
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
    compiled: &'a CompiledRollingSpec,
    node_id: &'a str,
    combined: usize,
}

/// Slides every shared window group using the compensated per-row order.
fn slide_windows(
    view: &EntityRowView<'_>,
    position: usize,
    row_index: usize,
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
            CompiledWindowGroup::Cumulative {
                input_index,
                statistic,
            } => {
                let WindowState::Cumulative(accumulator) = state else {
                    return Err(internal_error("cumulative group has the wrong state"));
                };
                let current = &context.view.rows[context.row_index].values[*input_index];
                if is_valid_sample(current) {
                    accumulator.add(current, *statistic, node_id)?;
                }
            }
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
    match frame {
        CompiledFrame::Rows(rows) => {
            let rows = usize::try_from(rows).map_err(|_| {
                operator_error(context.node_id, "rolling frame rows do not fit usize")
            })?;
            let leaving = (context.combined >= rows)
                .then(|| context.view.value(context.combined - rows, input_index))
                .filter(|value| is_valid_sample(value));
            accumulator.rows(
                leaving,
                is_valid_sample(current).then_some(current),
                context.node_id,
            )?;
        }
        CompiledFrame::Duration(_) => {
            expire_numeric(
                accumulator,
                context.view,
                context.combined,
                input_index,
                frame,
                context.node_id,
            )?;
            accumulator.numeric.reset_if_empty();
            if is_valid_sample(current) {
                accumulator.add(current, context.node_id)?;
            }
        }
    }
    finish_numeric_window(accumulator, context, input_index, frame)?;
    accumulator.sync_float_sum();
    Ok(())
}

fn finish_numeric_window(
    accumulator: &mut WindowAccumulator,
    context: &WindowSlideContext<'_>,
    input_index: usize,
    frame: CompiledFrame,
) -> Result<()> {
    if accumulator.sum.is_none() {
        return Ok(());
    }
    let history =
        window_positions(context.view, context.combined, frame, context.node_id)?.map(|index| {
            let value = context.view.value(index, input_index);
            is_valid_sample(value).then(|| float_sample(value))
        });
    match frame {
        CompiledFrame::Rows(rows) => usize::try_from(rows)
            .map_err(|_| operator_error(context.node_id, "rolling frame rows do not fit usize"))
            .and_then(|rows| {
                accumulator
                    .numeric
                    .finish_rows(rows, history)
                    .map_err(|error| kernel::float_arithmetic_error(error, context.node_id))
            }),
        CompiledFrame::Duration(_) => accumulator
            .numeric
            .finish_duration(history)
            .map_err(|error| kernel::float_arithmetic_error(error, context.node_id)),
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
    match frame {
        CompiledFrame::Rows(rows) => {
            let rows = usize::try_from(rows).map_err(|_| {
                operator_error(context.node_id, "rolling frame rows do not fit usize")
            })?;
            let leaving = (context.combined >= rows)
                .then(|| {
                    let left = context.view.value(context.combined - rows, left_index);
                    let right = context.view.value(context.combined - rows, right_index);
                    (is_valid_sample(left) && is_valid_sample(right))
                        .then(|| (float_sample(left), float_sample(right)))
                })
                .flatten();
            let entering = (is_valid_sample(left) && is_valid_sample(right))
                .then(|| (float_sample(left), float_sample(right)));
            accumulator.rows_values(leaving, entering, context.node_id)?;
        }
        CompiledFrame::Duration(_) => {
            expire_pair(
                accumulator,
                context.view,
                context.combined,
                left_index,
                right_index,
                frame,
                context.node_id,
            )?;
            accumulator.reset_if_empty();
            if is_valid_sample(left) && is_valid_sample(right) {
                accumulator.add(left, right, context.node_id)?;
            }
        }
    }
    finish_pair_window(accumulator, context, left_index, right_index, frame)
}

fn finish_pair_window(
    accumulator: &mut PairAccumulator,
    context: &WindowSlideContext<'_>,
    left_index: usize,
    right_index: usize,
    frame: CompiledFrame,
) -> Result<()> {
    let history =
        window_positions(context.view, context.combined, frame, context.node_id)?.map(|index| {
            let left = context.view.value(index, left_index);
            let right = context.view.value(index, right_index);
            (is_valid_sample(left) && is_valid_sample(right))
                .then(|| (float_sample(left), float_sample(right)))
        });
    match frame {
        CompiledFrame::Rows(rows) => {
            let rows = usize::try_from(rows).map_err(|_| {
                operator_error(context.node_id, "rolling frame rows do not fit usize")
            })?;
            accumulator.finish_rows(rows, history)
        }
        CompiledFrame::Duration(_) => accumulator.finish_duration(history),
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
        CompiledEvaluation::Indicator(_) => {
            return Err(internal_error(
                "indicator readout requires its row transition",
            ));
        }
        CompiledEvaluation::Lag { periods } | CompiledEvaluation::Delta { periods } => {
            usize::try_from(*periods)
                .map_err(|_| operator_error(node_id, "rolling periods does not fit usize"))?
        }
        CompiledEvaluation::Aggregate(aggregate) => {
            return evaluate_aggregate(aggregate, windows, output, node_id);
        }
        CompiledEvaluation::Cumulative(aggregate) => {
            let WindowState::Cumulative(state) = &windows[aggregate.group] else {
                return Err(internal_error("cumulative output has the wrong state"));
            };
            return state.read(aggregate, &output.output_type, node_id);
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

    #[cfg(test)]
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
                Statistic::Mean => Ok(Some(state.mean_value())),
                Statistic::Variance | Statistic::Stddev => {
                    let divisor = state.valid_count - u64::from(readout.ddof);
                    if divisor == 0 {
                        return Ok(None);
                    }
                    Ok(Some(if readout.statistic == Statistic::Stddev {
                        state.numeric.stddev(readout.ddof)
                    } else {
                        state.numeric.variance(readout.ddof)
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
/// undefined ∞ − ∞ (NaN), one sign is that infinity, and no infinity reads
/// the compensated finite families (SCE-07 defect 1 ruling);
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
        WindowState::Cumulative(_) | WindowState::Ewma(_) => {
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
        Statistic::Mean => Ok(ScalarValue::Float64(Some(accumulator.mean_value()))),
        Statistic::Variance | Statistic::Stddev => {
            let divisor = accumulator.valid_count - u64::from(aggregate.ddof);
            if divisor == 0 {
                return Ok(ScalarValue::Float64(None));
            }
            let variance = accumulator.numeric.variance(aggregate.ddof);
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
/// infinity, and the shifted compensated co-moment readout otherwise. The ddof
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
        Ok(accumulator
            .family
            .correlation(accumulator.valid_count)
            .map_or_else(
                || typed_null(&output.output_type),
                |value| ScalarValue::Float64(Some(value)),
            ))
    } else {
        Ok(ScalarValue::Float64(Some(
            accumulator
                .family
                .covariance(accumulator.valid_count, aggregate.ddof),
        )))
    }
}

fn typed_null(data_type: &DataType) -> ScalarValue {
    ScalarValue::try_from(data_type).unwrap_or(ScalarValue::Null)
}

#[cfg(test)]
mod tests;
