//! Failure-atomic retained history appends shared by rolling stream paths.

use datafusion::arrow::{
    array::{TimestampMicrosecondArray, UInt64Array},
    compute::take_record_batch,
};

use super::{
    BufferedRow, CompiledRollingSpec, CompiledWindowGroup, EntityRollingState, HistoryUpdates,
    KeyValue, RecordBatch, Result, RollingHistories, RollingMetricsRecorder, RollingStage,
    RollingWork, ScalarValue, VecDeque, WindowState, budget, fresh_windows, group_rows_by_entity,
    history_event_time, internal_error, kernel::StreamKernelUpdate, operator_error,
    read_buffered_row, state_codec::recurrence_windows,
};
use budget::{HistoryCharge, StateCharge};

pub(super) enum PreparedHistoryUpdate {
    Replace(HistoryUpdates),
    Append(Vec<RetainedHistoryAppend>),
}

impl PreparedHistoryUpdate {
    pub(super) fn changed_charge(
        &mut self,
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
            let (previous, projected) = tail.changed_charge(previous, node)?;
            old = old.checked_add(previous, node)?;
            new = new.checked_add(projected, node)?;
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
                next_charge: None,
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
    debug_assert!(previous.is_none_or(|state| state.columnar.records.is_empty()));
    let policy = RetentionPolicy::new(compiled);
    let last = rows[*indices.last().expect("an entity has at least one row")]
        .identity
        .event_time;
    let discarded = policy.discarded_history(previous, indices.len(), last, compiled, node)?;
    let row_prefix = indices.len().saturating_sub(policy.rows);
    let time_prefix = policy.bound(last).map_or(indices.len(), |bound| {
        indices.partition_point(|&index| i128::from(rows[index].identity.event_time) <= bound)
    });
    Ok((discarded, row_prefix.min(time_prefix)))
}

#[derive(Clone, Copy)]
pub(super) struct RetentionPolicy {
    pub(super) rows: usize,
    pub(super) duration: Option<u64>,
}

impl RetentionPolicy {
    pub(super) fn new(compiled: &CompiledRollingSpec) -> Self {
        Self {
            rows: usize::try_from(compiled.max_row_retention).unwrap_or(usize::MAX),
            duration: compiled.max_duration_micros,
        }
    }

    pub(super) fn bound(self, last: i64) -> Option<i128> {
        self.duration
            .map(|micros| i128::from(last) - i128::from(micros))
    }

    pub(super) fn keeps_new(self, newer: usize, time: i64, last: i64) -> bool {
        newer < self.rows
            || self
                .bound(last)
                .is_some_and(|bound| i128::from(time) > bound)
    }

    pub(super) fn discarded_history(
        self,
        previous: Option<&EntityRollingState>,
        incoming: usize,
        last: i64,
        compiled: &CompiledRollingSpec,
        node: &str,
    ) -> Result<usize> {
        let Some(previous) = previous else {
            return Ok(0);
        };
        let old = previous
            .rows
            .len()
            .checked_add(previous.columnar.rows)
            .ok_or_else(|| operator_error(node, "rolling retained history count overflowed"))?;
        let total = old
            .checked_add(incoming)
            .ok_or_else(|| operator_error(node, "rolling retained history count overflowed"))?;
        let by_rows = total.saturating_sub(self.rows).min(old);
        let Some(bound) = self.bound(last) else {
            return Ok(by_rows);
        };
        let mut by_time = 0_usize;
        for chunk in previous.retained_chunks() {
            let expired = chunk.expired_prefix(bound, compiled)?;
            by_time += expired;
            if expired < chunk.len() {
                break;
            }
        }
        Ok(by_rows.min(by_time))
    }
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
    pub(super) next_charge: Option<HistoryCharge>,
}

type ScalarRows = VecDeque<Vec<ScalarValue>>;

#[derive(Clone, Debug)]
pub(super) enum RetainedRows {
    Columnar(RecordBatch),
    Scalar(ScalarRows),
}

impl RetainedRows {
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Columnar(record) => record.num_rows(),
            Self::Scalar(rows) => rows.len(),
        }
    }

    pub(super) fn as_ref(&self) -> HistoryChunkRef<'_> {
        match self {
            Self::Columnar(record) => HistoryChunkRef::Columnar(record),
            Self::Scalar(rows) => HistoryChunkRef::Scalar(rows),
        }
    }

    #[cfg(test)]
    pub(super) fn columnar_record(&self) -> &RecordBatch {
        let Self::Columnar(record) = self else {
            panic!("expected a columnar retained block")
        };
        record
    }
}

#[derive(Clone, Copy)]
pub(super) enum HistoryChunkRef<'a> {
    Scalar(&'a ScalarRows),
    Columnar(&'a RecordBatch),
}

impl HistoryChunkRef<'_> {
    pub(super) fn len(self) -> usize {
        match self {
            Self::Scalar(rows) => rows.len(),
            Self::Columnar(record) => record.num_rows(),
        }
    }

    fn expired_prefix(self, bound: i128, compiled: &CompiledRollingSpec) -> Result<usize> {
        match self {
            Self::Scalar(rows) => {
                let expired = |values: &Vec<ScalarValue>| {
                    i128::from(history_event_time(values, compiled)) <= bound
                };
                let (front, back) = rows.as_slices();
                let first = front.partition_point(expired);
                Ok(if first == front.len() {
                    first + back.partition_point(expired)
                } else {
                    first
                })
            }
            Self::Columnar(record) => {
                let times = record
                    .column(compiled.event_time_index)
                    .as_any()
                    .downcast_ref::<TimestampMicrosecondArray>()
                    .ok_or_else(|| internal_error("retained rolling timestamp type changed"))?;
                Ok(times
                    .values()
                    .partition_point(|&time| i128::from(time) <= bound))
            }
        }
    }
}

impl EntityRollingState {
    pub(super) fn retained_chunks(&self) -> impl Iterator<Item = HistoryChunkRef<'_>> {
        std::iter::once(HistoryChunkRef::Scalar(&self.rows))
            .chain(self.columnar.records.iter().map(RetainedRows::as_ref))
    }
}

/// Columnar chunks own gathered values; scalar chunks own their retained payloads.
#[derive(Clone, Debug, Default)]
pub(super) struct ColumnarHistory {
    pub(super) records: VecDeque<RetainedRows>,
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
        for chunk in &self.records {
            if count >= chunk.len() {
                count -= chunk.len();
                if count == 0 {
                    return Ok(None);
                }
                continue;
            }
            let RetainedRows::Columnar(record) = chunk else {
                return Ok(None);
            };
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
            let chunk = self.records.pop_front().expect("retained rows are present");
            let rows = chunk.len();
            if count < rows {
                let retained = match chunk {
                    RetainedRows::Columnar(_) => RetainedRows::Columnar(
                        prepared_front.expect("partial history discard was prepared"),
                    ),
                    RetainedRows::Scalar(mut values) => {
                        for _ in 0..count {
                            values.pop_front();
                        }
                        compact_scalar_capacity(&mut values);
                        RetainedRows::Scalar(values)
                    }
                };
                self.records.push_front(retained);
                break;
            }
            count -= rows;
        }
    }

    fn push(&mut self, chunk: RetainedRows) {
        self.rows += chunk.len();
        if let RetainedRows::Scalar(rows) = chunk {
            if let Some(RetainedRows::Scalar(previous)) = self.records.back_mut() {
                previous.extend(rows);
            } else if !rows.is_empty() {
                self.records.push_back(RetainedRows::Scalar(rows));
            }
        } else if chunk.len() != 0 {
            self.records.push_back(chunk);
        }
    }

    pub(super) fn has_columnar(&self) -> bool {
        self.records
            .iter()
            .any(|chunk| matches!(chunk, RetainedRows::Columnar(_)))
    }

    fn prepare_materialize(
        &self,
        compiled: &CompiledRollingSpec,
        node_id: &str,
        observer: Option<&RollingMetricsRecorder>,
    ) -> Result<Vec<Option<ScalarRows>>> {
        self.records
            .iter()
            .map(|chunk| {
                let RetainedRows::Columnar(record) = chunk else {
                    return Ok(None);
                };
                let mut rows = VecDeque::with_capacity(record.num_rows());
                for index in 0..record.num_rows() {
                    rows.push_back(read_buffered_row(record, index, compiled, node_id)?.values);
                    if let Some(recorder) = observer {
                        recorder.add(RollingWork::HistoryRowsMaterialized, 1);
                        recorder.add(RollingWork::ScalarValueConversions, record.num_columns());
                    }
                }
                Ok(Some(rows))
            })
            .collect()
    }

    fn materialize_into(&mut self, target: &mut ScalarRows, prepared: Vec<Option<ScalarRows>>) {
        debug_assert_eq!(self.records.len(), prepared.len());
        for (chunk, prepared) in std::mem::take(&mut self.records).into_iter().zip(prepared) {
            match chunk {
                RetainedRows::Scalar(rows) => target.extend(rows),
                RetainedRows::Columnar(_) => {
                    target.extend(prepared.expect("columnar conversion was prepared"));
                }
            }
        }
        self.rows = 0;
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
            .map(|state| {
                state
                    .columnar
                    .prepare_materialize(compiled, node_id, observer)
            })
            .collect::<Result<Vec<_>>>()?;
        for (state, rows) in self.by_entity.values_mut().zip(prepared) {
            if state.columnar.has_columnar() {
                state.charge_cache = None;
            }
            state.columnar.materialize_into(&mut state.rows, rows);
        }
        Ok(())
    }

    pub(super) fn prepare_charge_cache(
        &self,
        node: &str,
    ) -> Result<(StateCharge, Vec<HistoryCharge>)> {
        let charges = self
            .by_entity
            .iter()
            .map(|(key, state)| budget::history_entity_charge_snapshot(key, state, node))
            .collect::<Result<Vec<_>>>()?;
        let total = charges
            .iter()
            .try_fold(StateCharge::default(), |total, charge| {
                total.checked_add(charge.total(node)?, node)
            })?;
        Ok((total, charges))
    }

    pub(super) fn install_charge_cache(&mut self, charges: Vec<HistoryCharge>) {
        debug_assert_eq!(charges.len(), self.by_entity.len());
        for (state, charge) in self.by_entity.values_mut().zip(charges) {
            state.charge_cache = Some(charge);
        }
    }
}

impl RetainedHistoryAppend {
    pub(super) fn changed_charge(
        &mut self,
        previous: Option<&EntityRollingState>,
        node: &str,
    ) -> Result<(StateCharge, StateCharge)> {
        let prior = previous
            .map(|state| budget::cached_history_entity_charge(&self.entity, state, node))
            .transpose()?
            .unwrap_or_default();
        let old = prior.total(node)?;
        let discarded = previous
            .map(|state| self.discarded_charge(state, node))
            .transpose()?
            .unwrap_or_default();
        let next = HistoryCharge {
            retained: prior
                .retained
                .checked_sub(discarded)?
                .checked_add(self.appended_charge(node)?, node)?,
            metadata: budget::history_metadata_charge(
                &self.entity,
                &self.windows,
                previous.map_or(&[], |state| state.indicators.as_slice()),
                node,
            )?,
        };
        let new = next.total(node)?;
        self.next_charge = Some(next);
        Ok((old, new))
    }

    fn discarded_charge(&self, previous: &EntityRollingState, node: &str) -> Result<StateCharge> {
        let scalar_discard = self.discarded_rows.min(previous.rows.len());
        let mut charge = StateCharge::default();
        for row in previous.rows.iter().take(scalar_discard) {
            charge = charge.checked_add(budget::scalar_history_row_charge(row, node)?, node)?;
        }
        charge.checked_add(
            previous.columnar.discarded_charge(
                self.discarded_rows - scalar_discard,
                self.prepared_front.as_ref(),
                node,
            )?,
            node,
        )
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
            RetainedRows::Scalar(rows) if state.columnar.records.is_empty() => {
                state.rows.extend(rows);
            }
            rows => state.columnar.push(rows),
        }
        if self.discarded_rows != 0 {
            compact_scalar_capacity(&mut state.rows);
        }
        state.windows = self.windows;
        state.transition_count = self.transition_count;
        state.charge_cache = self.next_charge;
    }
}

impl ColumnarHistory {
    fn discarded_charge(
        &self,
        mut discard: usize,
        prepared_front: Option<&RecordBatch>,
        node: &str,
    ) -> Result<StateCharge> {
        let mut charge = StateCharge::default();
        for chunk in &self.records {
            if discard == 0 {
                break;
            }
            let rows = chunk.len();
            if let RetainedRows::Scalar(values) = chunk {
                for row in values.iter().take(discard) {
                    charge =
                        charge.checked_add(budget::scalar_history_row_charge(row, node)?, node)?;
                }
                discard = discard.saturating_sub(rows);
                continue;
            }
            let RetainedRows::Columnar(record) = chunk else {
                unreachable!()
            };
            let original = budget::ordered_record_charge(record, node)?
                .ok_or_else(|| internal_error("columnar history has unsupported charge"))?;
            if discard >= rows {
                charge = charge.checked_add(original, node)?;
                discard -= rows;
                continue;
            }
            let retained = prepared_front.expect("partial discard has a prepared front");
            let retained_charge = budget::ordered_record_charge(retained, node)?
                .ok_or_else(|| internal_error("columnar history has unsupported charge"))?;
            charge = charge.checked_add(original.checked_sub(retained_charge)?, node)?;
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
