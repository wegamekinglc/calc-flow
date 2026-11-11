//! Failure-atomic retained history appends shared by rolling stream paths.

use datafusion::arrow::{array::UInt64Array, compute::take_record_batch};

use super::{
    BufferedRow, CompiledRollingSpec, CompiledWindowGroup, EntityRollingState, HistoryUpdates,
    KeyValue, RecordBatch, Result, RollingHistories, RollingMetricsRecorder, RollingStage,
    RollingWork, ScalarValue, VecDeque, WindowState, budget, fresh_windows, group_rows_by_entity,
    history_event_time, internal_error, kernel::StreamKernelUpdate, operator_error,
    read_buffered_row, state_codec::recurrence_windows,
};
use budget::StateCharge;

pub(super) enum PreparedHistoryUpdate {
    Replace(HistoryUpdates),
    Append(Vec<RetainedHistoryAppend>),
}

impl PreparedHistoryUpdate {
    pub(super) fn changed_charge(
        &self,
        histories: &RollingHistories,
        node: &str,
    ) -> Result<(StateCharge, StateCharge)> {
        let tails = match self {
            Self::Replace(updates) => {
                return budget::changed_histories_charge(histories, updates, node);
            }
            Self::Append(tails) => tails,
        };
        let mut old = StateCharge::default();
        let mut new = StateCharge::default();
        for tail in tails {
            let previous = histories.by_entity.get(&tail.entity);
            if let Some(previous) = previous {
                old = old.checked_add(
                    budget::history_entity_charge(&tail.entity, previous, node)?,
                    node,
                )?;
            }
            new = new.checked_add(tail.projected_charge(previous, node)?, node)?;
        }
        Ok((old, new))
    }

    pub(super) fn commit(self, histories: &mut RollingHistories) {
        match self {
            Self::Replace(updates) => histories.apply(updates),
            Self::Append(tails) => {
                for tail in tails {
                    tail.commit(histories);
                }
            }
        }
    }
}

pub(super) fn prepare_scalar_appends(
    rows: &[BufferedRow],
    histories: &RollingHistories,
    compiled: &CompiledRollingSpec,
    update: &StreamKernelUpdate,
    node: &str,
) -> Result<PreparedHistoryUpdate> {
    let seeds = update.recurrence_seeds();
    let tails = group_rows_by_entity(rows)
        .into_iter()
        .map(|(entity, indices)| {
            let previous = histories.by_entity.get(entity);
            let (discarded_rows, skip) = retained_prefix(previous, rows, &indices, compiled, node)?;
            let transitions = u64::try_from(indices.len()).map_err(|_| {
                operator_error(node, "rolling micro-batch row count does not fit u64")
            })?;
            let transition_count = previous
                .map_or(0, |state| state.transition_count)
                .checked_add(transitions)
                .ok_or_else(|| {
                    operator_error(node, "rolling entity transition count overflowed")
                })?;
            let mut windows = recurrence_windows(compiled, &seeds[update.entity_ids()[indices[0]]]);
            if windows.is_empty()
                && compiled
                    .window_groups
                    .iter()
                    .any(|group| matches!(group, CompiledWindowGroup::Ewma { .. }))
            {
                windows = fresh_windows(compiled);
            }
            Ok(RetainedHistoryAppend {
                entity: entity.clone(),
                rows: RetainedRows::Scalar(
                    indices
                        .into_iter()
                        .skip(skip)
                        .map(|index| rows[index].values.clone())
                        .collect(),
                ),
                prepared_front: None,
                transition_count,
                discarded_rows,
                windows,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(PreparedHistoryUpdate::Append(tails))
}

/// Rows and duration consumers retain the union of their required suffixes.
fn retained_prefix(
    previous: Option<&EntityRollingState>,
    rows: &[BufferedRow],
    indices: &[usize],
    compiled: &CompiledRollingSpec,
    node: &str,
) -> Result<(usize, usize)> {
    let previous_rows = previous.map_or(0, |state| state.rows.len());
    debug_assert!(previous.is_none_or(|state| state.columnar.records.is_empty()));
    let total = previous_rows
        .checked_add(indices.len())
        .ok_or_else(|| operator_error(node, "rolling retained history count overflowed"))?;
    let retention = usize::try_from(compiled.max_row_retention).unwrap_or(usize::MAX);
    let row_prefix = total.saturating_sub(retention);
    let time_prefix = compiled.max_duration_micros.map_or(total, |micros| {
        let last = rows[*indices.last().expect("an entity has at least one row")]
            .identity
            .event_time;
        let bound = i128::from(last) - i128::from(micros);
        let expired =
            |values: &Vec<ScalarValue>| i128::from(history_event_time(values, compiled)) <= bound;
        let old_prefix = previous.map_or(0, |state| {
            let (front, back) = state.rows.as_slices();
            let first = front.partition_point(expired);
            if first == front.len() {
                first + back.partition_point(expired)
            } else {
                first
            }
        });
        if old_prefix < previous_rows {
            old_prefix
        } else {
            previous_rows
                + indices
                    .partition_point(|&index| i128::from(rows[index].identity.event_time) <= bound)
        }
    });
    let discard = row_prefix.min(time_prefix);
    Ok((
        discard.min(previous_rows),
        discard.saturating_sub(previous_rows),
    ))
}

/// Stage only newly retained rows. Existing tails remain owned and untouched
/// until all output chunks have been emitted successfully.
pub(super) struct RetainedHistoryAppend {
    pub(super) entity: Vec<Option<KeyValue>>,
    pub(super) rows: RetainedRows,
    pub(super) prepared_front: Option<RecordBatch>,
    pub(super) transition_count: u64,
    pub(super) discarded_rows: usize,
    pub(super) windows: Vec<WindowState>,
}

pub(super) enum RetainedRows {
    Columnar(RecordBatch),
    Scalar(VecDeque<Vec<ScalarValue>>),
}

impl RetainedRows {
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Columnar(record) => record.num_rows(),
            Self::Scalar(rows) => rows.len(),
        }
    }
}

/// Every chunk owns a gathered entity tail, never a slice of the input batch.
#[derive(Clone, Debug, Default)]
pub(super) struct ColumnarHistory {
    pub(super) records: VecDeque<RecordBatch>,
    pub(super) rows: usize,
}

impl ColumnarHistory {
    pub(super) fn prepare_discard_front(
        &self,
        mut count: usize,
        node_id: &str,
    ) -> Result<Option<RecordBatch>> {
        if count == 0 {
            return Ok(None);
        }
        for record in &self.records {
            if count >= record.num_rows() {
                count -= record.num_rows();
                if count == 0 {
                    return Ok(None);
                }
                continue;
            }
            let retained = record.slice(count, record.num_rows() - count);
            let Some(logical_bytes) = crate::operator::row_cost::RowCosts::try_total(&retained)?
            else {
                return Err(internal_error(
                    "columnar history has unsupported row charges",
                ));
            };
            let backing_bytes = retained.columns().iter().fold(0_usize, |bytes, column| {
                bytes.saturating_add(column.get_buffer_memory_size())
            });
            // Compact geometrically by bytes, not rows: one evicted string can
            // own almost the whole buffer. The small-buffer floor avoids a
            // fresh allocation for every single-row append to a short window.
            if backing_bytes <= logical_bytes.saturating_mul(2).max(4_096) {
                return Ok(Some(retained));
            }
            let indices = UInt64Array::from_iter_values(
                (0..retained.num_rows())
                    .map(|index| u64::try_from(index).expect("Arrow row index fits u64")),
            );
            return take_record_batch(&retained, &indices)
                .map(Some)
                .map_err(|error| operator_error(node_id, &error.to_string()));
        }
        Ok(None)
    }

    fn discard_front(&mut self, mut count: usize, prepared_front: Option<RecordBatch>) {
        self.rows -= count;
        while count > 0 {
            let record = self.records.pop_front().expect("retained rows are present");
            if count < record.num_rows() {
                let retained = prepared_front.expect("partial history discard was prepared");
                self.records.push_front(retained);
                break;
            }
            count -= record.num_rows();
        }
    }

    fn push(&mut self, record: RecordBatch) {
        self.rows += record.num_rows();
        self.records.push_back(record);
    }

    fn materialize(
        &self,
        compiled: &CompiledRollingSpec,
        node_id: &str,
        observer: Option<&RollingMetricsRecorder>,
    ) -> Result<VecDeque<Vec<ScalarValue>>> {
        let mut rows = VecDeque::with_capacity(self.rows);
        for record in &self.records {
            for index in 0..record.num_rows() {
                rows.push_back(read_buffered_row(record, index, compiled, node_id)?.values);
                if let Some(recorder) = observer {
                    recorder.add(RollingWork::HistoryRowsMaterialized, 1);
                    recorder.add(RollingWork::ScalarValueConversions, record.num_columns());
                }
            }
        }
        Ok(rows)
    }
}

impl RollingHistories {
    pub(super) fn materialize_columnar(
        &mut self,
        compiled: &CompiledRollingSpec,
        node_id: &str,
        observer: Option<&RollingMetricsRecorder>,
    ) -> Result<()> {
        let _stage = observer.map(|recorder| recorder.stage(RollingStage::HistoryMaintenance));
        let prepared = self
            .by_entity
            .values()
            .map(|state| state.columnar.materialize(compiled, node_id, observer))
            .collect::<Result<Vec<_>>>()?;
        for (state, rows) in self.by_entity.values_mut().zip(prepared) {
            state.rows.extend(rows);
            state.columnar = ColumnarHistory::default();
        }
        Ok(())
    }
}

impl RetainedHistoryAppend {
    pub(super) fn projected_charge(
        &self,
        previous: Option<&EntityRollingState>,
        node: &str,
    ) -> Result<StateCharge> {
        let mut charge =
            budget::history_entity_base_charge(&self.entity, self.windows.len(), node)?;
        if let Some(previous) = previous {
            let scalar_discard = self.discarded_rows.min(previous.rows.len());
            for row in previous.rows.iter().skip(scalar_discard) {
                charge = charge.checked_add(budget::scalar_history_row_charge(row, node)?, node)?;
            }
            charge = charge.checked_add(
                previous.columnar.retained_charge(
                    self.discarded_rows - scalar_discard,
                    self.prepared_front.as_ref(),
                    node,
                )?,
                node,
            )?;
        }
        charge.checked_add(self.appended_charge(node)?, node)
    }

    fn appended_charge(&self, node: &str) -> Result<StateCharge> {
        match &self.rows {
            RetainedRows::Columnar(record) if record.num_rows() != 0 => {
                budget::ordered_record_charge(record, node)?
                    .ok_or_else(|| internal_error("columnar tail has unsupported charge"))
            }
            RetainedRows::Columnar(_) => Ok(StateCharge::default()),
            RetainedRows::Scalar(rows) => {
                rows.iter().try_fold(StateCharge::default(), |charge, row| {
                    charge.checked_add(budget::scalar_history_row_charge(row, node)?, node)
                })
            }
        }
    }

    pub(super) fn commit(self, histories: &mut RollingHistories) {
        let state = histories.by_entity.entry(self.entity).or_default();
        let scalar_discard = self.discarded_rows.min(state.rows.len());
        for _ in 0..scalar_discard {
            state.rows.pop_front();
        }
        state
            .columnar
            .discard_front(self.discarded_rows - scalar_discard, self.prepared_front);
        match self.rows {
            RetainedRows::Columnar(record) if record.num_rows() != 0 => state.columnar.push(record),
            RetainedRows::Columnar(_) => {}
            RetainedRows::Scalar(rows) => state.rows.extend(rows),
        }
        if self.discarded_rows != 0 {
            compact_scalar_capacity(&mut state.rows);
        }
        state.windows = self.windows;
        state.transition_count = self.transition_count;
    }
}

impl ColumnarHistory {
    fn retained_charge(
        &self,
        mut discard: usize,
        prepared_front: Option<&RecordBatch>,
        node: &str,
    ) -> Result<StateCharge> {
        let mut charge = StateCharge::default();
        for record in &self.records {
            if discard >= record.num_rows() {
                discard -= record.num_rows();
                continue;
            }
            let retained = if discard == 0 {
                record
            } else {
                prepared_front.expect("partial discard has a prepared front")
            };
            let retained_charge = budget::ordered_record_charge(retained, node)?
                .ok_or_else(|| internal_error("columnar history has unsupported charge"))?;
            charge = charge.checked_add(retained_charge, node)?;
            discard = 0;
        }
        Ok(charge)
    }
}

fn compact_scalar_capacity(rows: &mut VecDeque<Vec<ScalarValue>>) {
    if rows.is_empty() {
        *rows = VecDeque::new();
    } else if rows.capacity() > rows.len().saturating_mul(4).max(64) {
        rows.shrink_to(rows.len().saturating_mul(2));
    }
}
