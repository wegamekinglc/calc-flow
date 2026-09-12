//! SQL COUNT/AVG transitions retain `DataFusion`'s sum/count operation order.

use std::{collections::VecDeque, mem::size_of, ops::Range, time::Instant};

use datafusion::arrow::{
    array::{Array, ArrayRef, Float64Array, Float64Builder, Int64Builder},
    compute::kernels::partition::partition,
    datatypes::{DataType, Schema, TimeUnit},
    record_batch::RecordBatch,
    row::{RowConverter, SortField},
};
use sha2::{Digest, Sha256};

use super::{
    DataFusionRollingMetrics, DataFusionRollingWindow, RollingExecutionBatch, RollingExecutionState,
};

#[derive(Clone, Debug)]
struct SqlGroup {
    input_index: usize,
    rows: usize,
    needs_count: bool,
}

#[derive(Clone, Debug)]
pub(super) struct SqlRollingKernel {
    partition_indices: Vec<usize>,
    groups: Vec<SqlGroup>,
    outputs: Vec<(usize, bool)>,
    fingerprint: String,
}

#[derive(Clone, Debug, Default)]
pub(super) struct SqlRollingState {
    last_key: Option<Vec<u8>>,
    groups: Vec<SqlWindowState>,
}

#[derive(Clone, Debug)]
struct SqlWindowState {
    rows: usize,
    values: VecDeque<Option<f64>>,
    sum: Option<f64>,
    count: i64,
}

struct SqlGroupOutput {
    means: Float64Builder,
    counts: Option<Int64Builder>,
}

impl SqlRollingState {
    fn continuing_groups(&self, first_key: Option<&[u8]>, empty: bool) -> Vec<SqlWindowState> {
        if first_key == self.last_key.as_deref() || empty {
            self.groups.clone()
        } else {
            Vec::new()
        }
    }
}

impl SqlGroupOutput {
    fn fill_range(
        &mut self,
        state: &mut SqlWindowState,
        values: &Float64Array,
        range: Range<usize>,
    ) -> crate::Result<()> {
        for row in range {
            let value = (!values.is_null(row)).then(|| values.value(row));
            state.update(value)?;
            self.means.append_option(state.mean());
            if let Some(counts) = self.counts.as_mut() {
                counts.append_value(state.count);
            }
        }
        Ok(())
    }
}

struct SqlPartitions {
    ranges: Vec<Range<usize>>,
    first_key: Option<Vec<u8>>,
    last_key: Option<Vec<u8>>,
}

fn validate_order(
    schema: &Schema,
    partition_indices: &[usize],
    order_indices: &[usize],
) -> Option<()> {
    let &event_time = order_indices.first()?;
    if partition_indices.is_empty()
        || !is_event_time(schema.field(event_time).data_type())
        || order_indices
            .iter()
            .any(|&index| schema.field(index).is_nullable())
        || partition_indices
            .iter()
            .any(|index| order_indices.contains(index))
    {
        return None;
    }
    RowConverter::new(
        partition_indices
            .iter()
            .map(|&index| SortField::new(schema.field(index).data_type().clone()))
            .collect(),
    )
    .ok()?;
    Some(())
}

fn is_event_time(data_type: &DataType) -> bool {
    matches!(data_type, DataType::Timestamp(TimeUnit::Microsecond, timezone)
        if timezone.as_deref().is_none_or(|timezone| timezone == "UTC"))
}

fn window_rows(
    schema: &Schema,
    window: &DataFusionRollingWindow,
    windows: &[DataFusionRollingWindow],
) -> Option<usize> {
    let rows = usize::try_from(window.rows).ok()?;
    if rows == 0
        || i64::try_from(rows).is_err()
        || schema.field(window.input_index).data_type() != &DataType::Float64
        || (window.is_count && !has_mean_window(window, windows))
    {
        return None;
    }
    Some(rows)
}

fn has_mean_window(window: &DataFusionRollingWindow, windows: &[DataFusionRollingWindow]) -> bool {
    windows.iter().any(|other| {
        !other.is_count && other.input_index == window.input_index && other.rows == window.rows
    })
}

struct CompiledSqlGroups {
    groups: Vec<SqlGroup>,
    outputs: Vec<(usize, bool)>,
}

fn compile_groups(
    schema: &Schema,
    windows: &[DataFusionRollingWindow],
) -> Option<CompiledSqlGroups> {
    let mut groups: Vec<SqlGroup> = Vec::new();
    let mut outputs = Vec::with_capacity(windows.len());
    for window in windows {
        let rows = window_rows(schema, window, windows)?;
        let group = groups
            .iter()
            .position(|group| group.input_index == window.input_index && group.rows == rows)
            .unwrap_or_else(|| {
                groups.push(SqlGroup {
                    input_index: window.input_index,
                    rows,
                    needs_count: false,
                });
                groups.len() - 1
            });
        groups[group].needs_count |= window.is_count;
        outputs.push((group, window.is_count));
    }
    Some(CompiledSqlGroups { groups, outputs })
}

impl SqlRollingKernel {
    pub(super) fn compile(
        schema: &Schema,
        partition_indices: &[usize],
        order_indices: &[usize],
        windows: &[DataFusionRollingWindow],
    ) -> Option<Self> {
        validate_order(schema, partition_indices, order_indices)?;
        let CompiledSqlGroups { groups, outputs } = compile_groups(schema, windows)?;
        let fingerprint = format!(
            "sql-sum-count-v1:{:x}",
            Sha256::digest(format!(
                "{partition_indices:?}:{order_indices:?}:{windows:?}"
            ))
        );
        Some(Self {
            partition_indices: partition_indices.to_vec(),
            groups,
            outputs,
            fingerprint,
        })
    }

    pub(super) fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    pub(super) fn estimated_state_bytes_per_entity(&self) -> usize {
        self.groups
            .iter()
            .map(|group| {
                size_of::<SqlWindowState>()
                    .saturating_add(group.rows.saturating_mul(size_of::<Option<f64>>()))
            })
            .fold(0, usize::saturating_add)
    }

    pub(super) fn update_and_fill(
        &self,
        state: &SqlRollingState,
        input: &RecordBatch,
    ) -> crate::Result<RollingExecutionBatch> {
        let started = Instant::now();
        let arrays = self
            .groups
            .iter()
            .map(|group| {
                input
                    .column(group.input_index)
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| internal("SQL rolling input type changed"))
            })
            .collect::<crate::Result<Vec<_>>>()?;
        let input_validation_ns = elapsed_ns(started);
        let started = Instant::now();
        let SqlPartitions {
            ranges,
            first_key,
            last_key,
        } = self.partitions(input)?;
        let entity_encode_ns = elapsed_ns(started);
        let mut groups = state.continuing_groups(first_key.as_deref(), ranges.is_empty());
        let mut output = self
            .groups
            .iter()
            .map(|group| SqlGroupOutput {
                means: Float64Builder::with_capacity(input.num_rows()),
                counts: group
                    .needs_count
                    .then(|| Int64Builder::with_capacity(input.num_rows())),
            })
            .collect::<Vec<_>>();
        let started = Instant::now();
        self.fill_ranges(&ranges, &arrays, &mut groups, &mut output)?;
        let kernel_ns = elapsed_ns(started);
        let started = Instant::now();
        let columns = self.finish_columns(output)?;
        let output_build_ns = elapsed_ns(started);
        let last_key = last_key.or_else(|| state.last_key.clone());
        let state_bytes = groups
            .iter()
            .map(SqlWindowState::estimated_bytes)
            .sum::<usize>()
            + last_key.as_ref().map_or(0, Vec::len);
        Ok(RollingExecutionBatch {
            columns,
            metrics: DataFusionRollingMetrics {
                input_validation_ns,
                entity_encode_ns,
                kernel_ns,
                output_build_ns,
                input_rows: input.num_rows(),
                entities: usize::from(!groups.is_empty()),
                state_bytes,
                ..DataFusionRollingMetrics::default()
            },
            state: RollingExecutionState::Sql(SqlRollingState { last_key, groups }),
        })
    }

    fn partitions(&self, input: &RecordBatch) -> crate::Result<SqlPartitions> {
        let keys = self
            .partition_indices
            .iter()
            .map(|&index| input.column(index).clone())
            .collect::<Vec<_>>();
        let ranges = partition(&keys)
            .map_err(|error| internal(&format!("SQL rolling partitioning failed: {error}")))?
            .ranges();
        let first_key = ranges
            .first()
            .map(|range| self.key(input, range.start))
            .transpose()?;
        let last_key = if ranges.len() > 1 {
            Some(self.key(input, input.num_rows() - 1)?)
        } else {
            first_key.clone()
        };
        Ok(SqlPartitions {
            ranges,
            first_key,
            last_key,
        })
    }

    fn fill_ranges(
        &self,
        ranges: &[Range<usize>],
        arrays: &[&Float64Array],
        groups: &mut Vec<SqlWindowState>,
        output: &mut [SqlGroupOutput],
    ) -> crate::Result<()> {
        for (index, range) in ranges.iter().enumerate() {
            if index > 0 || groups.is_empty() {
                *groups = self
                    .groups
                    .iter()
                    .map(|group| SqlWindowState::new(group.rows, range.len()))
                    .collect();
            }
            for ((state, values), output) in groups.iter_mut().zip(arrays).zip(output.iter_mut()) {
                output.fill_range(state, values, range.clone())?;
            }
        }
        Ok(())
    }

    fn finish_columns(&self, output: Vec<SqlGroupOutput>) -> crate::Result<Vec<ArrayRef>> {
        let output = output
            .into_iter()
            .map(|mut output| {
                (
                    std::sync::Arc::new(output.means.finish()) as ArrayRef,
                    output
                        .counts
                        .map(|mut counts| std::sync::Arc::new(counts.finish()) as ArrayRef),
                )
            })
            .collect::<Vec<_>>();
        self.outputs
            .iter()
            .map(|&(group, is_count)| {
                if is_count {
                    output[group]
                        .1
                        .clone()
                        .ok_or_else(|| internal("SQL count output missing"))
                } else {
                    Ok(output[group].0.clone())
                }
            })
            .collect()
    }

    fn key(&self, input: &RecordBatch, row: usize) -> crate::Result<Vec<u8>> {
        let columns = self
            .partition_indices
            .iter()
            .map(|&index| input.column(index).slice(row, 1))
            .collect::<Vec<_>>();
        let fields = columns
            .iter()
            .map(|column| SortField::new(column.data_type().clone()))
            .collect();
        Ok(RowConverter::new(fields)
            .and_then(|converter| converter.convert_columns(&columns))
            .map_err(|error| internal(&format!("SQL rolling partition key failed: {error}")))?
            .row(0)
            .data()
            .to_vec())
    }
}

impl SqlWindowState {
    fn new(rows: usize, input_rows: usize) -> Self {
        Self {
            rows,
            values: VecDeque::with_capacity(rows.min(input_rows).saturating_add(1)),
            sum: None,
            count: 0,
        }
    }

    fn update(&mut self, value: Option<f64>) -> crate::Result<()> {
        if let Some(value) = value {
            self.count = self
                .count
                .checked_add(1)
                .ok_or_else(|| internal("SQL rolling count overflowed"))?;
            *self.sum.get_or_insert(0.0) += value;
        }
        self.values.push_back(value);
        if self.values.len() > self.rows
            && let Some(Some(value)) = self.values.pop_front()
        {
            self.count -= 1;
            self.sum = Some(
                self.sum
                    .ok_or_else(|| internal("SQL rolling sum missing"))?
                    - value,
            );
        }
        Ok(())
    }

    #[allow(
        clippy::cast_precision_loss,
        reason = "DataFusion AVG converts its non-null count to Float64"
    )]
    fn mean(&self) -> Option<f64> {
        (self.count != 0)
            .then(|| self.sum.map(|sum| sum / self.count as f64))
            .flatten()
    }

    fn estimated_bytes(&self) -> usize {
        size_of::<Self>() + self.values.capacity() * size_of::<Option<f64>>()
    }
}

fn elapsed_ns(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

fn internal(message: &str) -> crate::CalcFlowError {
    crate::CalcFlowError::Internal {
        message: message.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use datafusion::arrow::datatypes::Field;

    use super::*;

    fn kernel_with_metadata(note: &str) -> SqlRollingKernel {
        let metadata = HashMap::from([("caller_note".to_owned(), note.to_owned())]);
        let schema = Schema::new(vec![
            Field::new(
                "event_time",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new("sequence", DataType::UInt64, false),
            Field::new("symbol", DataType::Utf8, false),
            Field::new("price", DataType::Float64, false).with_metadata(metadata.clone()),
        ])
        .with_metadata(metadata);
        SqlRollingKernel::compile(
            &schema,
            &[2],
            &[0, 1],
            &[
                DataFusionRollingWindow {
                    input_index: 3,
                    output_name: "mean".to_owned(),
                    rows: 20,
                    is_count: false,
                },
                DataFusionRollingWindow {
                    input_index: 3,
                    output_name: "count".to_owned(),
                    rows: 20,
                    is_count: true,
                },
            ],
        )
        .unwrap()
    }

    #[test]
    fn sql_kernel_fingerprint_ignores_caller_schema_and_field_metadata() {
        let left = kernel_with_metadata("left");
        let right = kernel_with_metadata("right");
        assert_eq!(left.fingerprint(), right.fingerprint());
    }
}
