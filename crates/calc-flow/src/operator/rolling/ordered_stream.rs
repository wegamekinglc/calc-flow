//! Columnar buffering for proven ordered, bounded row-window streams.
//!
//! Finality remains watermark driven. Unsupported shapes, late envelopes, and
//! overlapping or unordered arrivals enter the existing general row path.

use datafusion::arrow::{
    array::{TimestampMicrosecondArray, UInt64Array},
    compute::take_record_batch,
    datatypes::DataType,
};

use super::kernel::StreamKernelUpdate;
use crate::operator::rolling_metrics::{RollingMetricsRecorder, RollingStage, RollingWork};
use crate::runtime::streaming::entity_work::ReservePair;

use super::{
    Arc, Array, BTreeMap, Batch, BufferedRow, CompiledRollingSpec, EntityRollingState, EventTime,
    KeyValue, RecordBatch, Result, RollingHistories, RollingOperator, RowIdentity, ScalarValue,
    StreamCollector, StreamOperatorContext, TableBatch, VecDeque, WindowState, chunk_output_record,
    closing_coordinate, concat_batches, fresh_windows, internal_error, operator_error,
    read_buffered_row, reconstruct_typed_state,
};

#[derive(Default)]
pub(super) struct OrderedStreamBuffer {
    records: VecDeque<RecordBatch>,
    last_identity: Option<Vec<u8>>,
}

impl OrderedStreamBuffer {
    pub(super) fn records(&self) -> impl Iterator<Item = &RecordBatch> {
        self.records.iter()
    }

    pub(super) fn last_identity(&self) -> Option<Vec<u8>> {
        self.last_identity.clone()
    }

    pub(super) fn restore_front(
        &mut self,
        records: Vec<RecordBatch>,
        last_identity: Option<Vec<u8>>,
    ) {
        for record in records.into_iter().rev() {
            self.records.push_front(record);
        }
        self.last_identity = last_identity;
    }

    pub(super) fn clear(&mut self) {
        self.records.clear();
        self.last_identity = None;
    }

    pub(super) fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub(super) fn take_all(&mut self) -> Vec<RecordBatch> {
        self.last_identity = None;
        self.records.drain(..).collect()
    }

    pub(super) fn take_closed(
        &mut self,
        watermark: EventTime,
        compiled: &CompiledRollingSpec,
        allowed_lateness_micros: u64,
        node_id: &str,
    ) -> Result<Vec<RecordBatch>> {
        self.validate_closing_coordinates(compiled, allowed_lateness_micros, node_id)?;
        let mut closed = Vec::new();
        while let Some(record) = self.records.front() {
            let times = timestamps(record, compiled)?;
            let count = times.values().partition_point(|&time| {
                i128::from(time) + i128::from(allowed_lateness_micros)
                    <= i128::from(watermark.as_micros())
            });
            if count == 0 {
                break;
            }
            let record = self.records.pop_front().expect("front was present");
            if count == record.num_rows() {
                closed.push(record);
            } else {
                closed.push(record.slice(0, count));
                self.records
                    .push_front(record.slice(count, record.num_rows() - count));
                break;
            }
        }
        if self.records.is_empty() {
            self.last_identity = None;
        }
        Ok(closed)
    }
    fn validate_closing_coordinates(
        &self,
        compiled: &CompiledRollingSpec,
        allowed_lateness_micros: u64,
        node_id: &str,
    ) -> Result<()> {
        // Check every coordinate before changing the buffer, matching the
        // general path's atomic closing-key validation.
        for record in &self.records {
            let times = timestamps(record, compiled)?;
            closing_coordinate(
                times.value(record.num_rows() - 1),
                allowed_lateness_micros,
                node_id,
            )?;
        }
        Ok(())
    }
}

fn timestamps<'a>(
    record: &'a RecordBatch,
    compiled: &CompiledRollingSpec,
) -> Result<&'a TimestampMicrosecondArray> {
    record
        .column(compiled.event_time_index)
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()
        .ok_or_else(|| internal_error("validated rolling timestamp column changed type"))
}

impl RollingOperator {
    fn restore_ordered_failure(
        &mut self,
        records: Vec<RecordBatch>,
        last_identity: Option<Vec<u8>>,
        error: super::CalcFlowError,
        node: &str,
    ) -> Result<()> {
        self.state.ordered.restore_front(records, last_identity);
        if let Ok(charge) = super::budget::full_state_charge(
            &self.state.buffer,
            &self.state.ordered,
            &self.state.histories,
            &self.compiled,
            node,
        ) {
            self.state.charge = charge;
        }
        Err(error)
    }

    fn projected_ordered_charge(
        &mut self,
        prepared: &PreparedOrderedOutput,
        removed: super::budget::StateCharge,
        was_warm: bool,
        node: &str,
    ) -> Result<(super::budget::StateCharge, super::budget::StateCharge)> {
        if !was_warm {
            self.state.charge = super::budget::full_state_charge(
                &self.state.buffer,
                &self.state.ordered,
                &self.state.histories,
                &self.compiled,
                node,
            )?
            .checked_add(removed, node)?;
        }
        let current = self.state.charge.checked_sub(removed)?;
        let next = prepared.projected_charge(self, current, node)?;
        self.check_state_budget(next, node)?;
        Ok((current, next))
    }

    fn supports_ordered_buffer(&self) -> bool {
        self.compiled.kernel_plan.supports_typed_transition()
            && self.compiled.max_duration_micros.is_none()
            && self.state.buffer.is_empty()
    }

    fn wholly_on_time(&self, record: &RecordBatch, watermark: Option<EventTime>) -> Result<bool> {
        let Some(watermark) = watermark else {
            return Ok(true);
        };
        let times = timestamps(record, &self.compiled)?;
        // Lateness precedes duplicate rejection, including duplicate late rows under Drop.
        Ok(times.values().iter().all(|&time| {
            let closing = i128::from(time) + i128::from(self.spec.allowed_lateness_micros);
            closing <= i128::from(i64::MAX) && closing > i128::from(watermark.as_micros())
        }))
    }

    fn buffered_order_bounds(
        &self,
        record: &RecordBatch,
        watermark: Option<EventTime>,
        observer: Option<&RollingMetricsRecorder>,
    ) -> Result<Option<(Vec<u8>, Vec<u8>)>> {
        let validation = observer.map(|recorder| recorder.stage(RollingStage::InputValidation));
        let on_time = self.wholly_on_time(record, watermark)?;
        drop(validation);
        if !on_time {
            return Ok(None);
        }
        self.compiled
            .kernel_plan
            .ordered_stream_bounds(record, &self.name, observer)
    }

    pub(super) fn try_buffer_ordered(
        &mut self,
        table: &TableBatch,
        watermark: Option<EventTime>,
        observer: Option<&RollingMetricsRecorder>,
    ) -> Result<bool> {
        let _stage = observer.map(|recorder| recorder.stage(RollingStage::OrderingProof));
        if !self.supports_ordered_buffer() {
            return Ok(false);
        }
        let mut last = self.state.ordered.last_identity.clone();
        let mut prepared = Vec::new();
        for record in table
            .batches()
            .iter()
            .filter(|record| record.num_rows() != 0)
        {
            let Some((first, end)) = self.buffered_order_bounds(record, watermark, observer)?
            else {
                return Ok(false);
            };
            if last.as_ref().is_some_and(|previous| previous >= &first) {
                return Ok(false);
            }
            prepared.push(record.clone());
            last = Some(end);
        }
        let Some(incoming) =
            super::budget::ordered_buffer_charge(prepared.iter(), &self.compiled, &self.name)?
        else {
            return Ok(false);
        };
        let next_charge = self.state.charge.checked_add(incoming, &self.name)?;
        self.check_state_budget(next_charge, &self.name)?;
        self.state.ordered.records.extend(prepared);
        self.state.ordered.last_identity = last;
        self.state.charge = next_charge;
        Ok(true)
    }

    pub(super) fn materialize_ordered_buffer(
        &mut self,
        observer: Option<&RollingMetricsRecorder>,
    ) -> Result<()> {
        let materialized = self.prepare_ordered_buffer(observer)?;
        let old_charge = super::budget::ordered_buffer_charge(
            self.state.ordered.records.iter(),
            &self.compiled,
            &self.name,
        )?
        .expect("ordered buffer contains chargeable records");
        let next_charge = self.state.charge.checked_sub(old_charge)?.checked_add(
            super::budget::buffered_charge(materialized.values(), &self.name)?,
            &self.name,
        )?;
        self.check_state_budget(next_charge, &self.name)?;
        self.state.buffer.extend(materialized);
        self.state.ordered.clear();
        self.state.charge = next_charge;
        Ok(())
    }

    pub(super) fn prepare_ordered_buffer(
        &self,
        observer: Option<&RollingMetricsRecorder>,
    ) -> Result<BTreeMap<RowIdentity, BufferedRow>> {
        let _stage = observer.map(|recorder| recorder.stage(RollingStage::InputValidation));
        let mut materialized = BTreeMap::new();
        for record in &self.state.ordered.records {
            for index in 0..record.num_rows() {
                let row = read_buffered_row(record, index, &self.compiled, &self.name)?;
                if let Some(recorder) = observer {
                    recorder.add(RollingWork::ScalarValueConversions, record.num_columns());
                }
                materialized.insert(row.identity.clone(), row);
            }
        }
        Ok(materialized)
    }

    fn ensure_ordered_kernel_state(
        &mut self,
        input: &RecordBatch,
        node_id: &str,
        observer: Option<&RollingMetricsRecorder>,
    ) -> Result<()> {
        if self.state.typed_kernel_state.is_none() {
            let _stage = observer.map(|recorder| recorder.stage(RollingStage::StatePreparation));
            self.state
                .histories
                .materialize_columnar(&self.compiled, node_id, observer)?;
            let restored = reconstruct_typed_state(
                &self.state.histories,
                &self.compiled,
                &input.schema(),
                node_id,
            )?;
            self.state.typed_kernel_state = Some(Box::new(restored));
        }
        Ok(())
    }

    fn ordered_output_chunks(
        &self,
        input: &RecordBatch,
        update: &mut StreamKernelUpdate,
        context: &StreamOperatorContext<'_>,
    ) -> Result<(Vec<Batch>, u64)> {
        let observer = context.rolling_metrics();
        let arrow_stage = observer.map(|recorder| recorder.stage(RollingStage::ArrowOutput));
        let schema = self.output_ports[0]
            .schema()
            .expect("rolling output has an exact schema");
        let columns = input
            .columns()
            .iter()
            .cloned()
            .chain(update.take_columns())
            .collect();
        let record = RecordBatch::try_new(Arc::clone(schema), columns)
            .map_err(|error| operator_error(context.operator_id(), &error.to_string()))?;
        if let Some(recorder) = observer {
            recorder.add(RollingWork::OutputRowsPrepared, record.num_rows());
        }
        drop(arrow_stage);
        let _stage = observer.map(|recorder| recorder.stage(RollingStage::BudgetPreparation));
        let batches = chunk_output_record(
            &record,
            context.operator_id(),
            self.state.next_output_sequence,
            context.output_budget(),
        )?;
        let count = u64::try_from(batches.len())
            .map_err(|_| operator_error(context.operator_id(), "output chunk count overflowed"))?;
        if let Some(recorder) = observer {
            recorder.add(RollingWork::OutputChunksPrepared, batches.len());
        }
        let next_sequence = self
            .state
            .next_output_sequence
            .checked_add(count)
            .ok_or_else(|| operator_error(context.operator_id(), "output sequence overflowed"))?;
        Ok((batches, next_sequence))
    }

    async fn prepare_ordered_output(
        &mut self,
        input: &RecordBatch,
        context: &StreamOperatorContext<'_>,
    ) -> Result<PreparedOrderedOutput> {
        let observer = context.rolling_metrics();
        // Building/reconstructing typed state keeps this callback on the original
        // serial route, even when the resulting state happens to be eligible.
        let was_warm = self.state.typed_kernel_state.is_some();
        self.ensure_ordered_kernel_state(input, context.operator_id(), observer)?;
        let mut update = self
            .prepare_ordered_update(input, context, was_warm)
            .await?;
        let touched = {
            let _stage = observer.map(|recorder| recorder.stage(RollingStage::HistoryMaintenance));
            retained_histories(
                input,
                update.entity_ids(),
                &self.state.histories,
                &self.compiled,
                context.operator_id(),
                observer,
            )?
        };
        let ewma_seeds = update.ewma_seeds();
        let (batches, next_sequence) = self.ordered_output_chunks(input, &mut update, context)?;
        Ok(PreparedOrderedOutput {
            update,
            touched,
            ewma_seeds,
            batches,
            next_sequence,
        })
    }

    async fn prepare_ordered_update(
        &self,
        input: &RecordBatch,
        context: &StreamOperatorContext<'_>,
        was_warm: bool,
    ) -> Result<StreamKernelUpdate> {
        let observer = context.rolling_metrics();
        let plan = &self.compiled.kernel_plan;
        let prior = self
            .state
            .typed_kernel_state
            .as_deref()
            .expect("state initialized above");
        let Some(client) = context
            .entity_work()
            .filter(|_| was_warm && plan.supports_entity_parallel(input))
        else {
            return plan.prepare_ordered_stream(prior, input, context.operator_id(), observer);
        };
        let prepared =
            plan.prepare_ordered_stream_inputs(prior, input, context.operator_id(), observer)?;
        let Some(scratch) = prepared.scratch_plan() else {
            return prepared.finish_serial(observer);
        };
        match client.try_reserve(scratch) {
            ReservePair::Reserved(launch) => match prepared.split_two() {
                Ok((seed, requests)) => {
                    let stage =
                        observer.map(|recorder| recorder.stage(RollingStage::NumericUpdate));
                    let mut ticket = launch.start(requests, observer.cloned());
                    let joined = ticket.join().await;
                    drop(stage);
                    let _stage = observer.map(|recorder| recorder.stage(RollingStage::ArrowOutput));
                    joined.merge(seed)
                }
                Err(prepared) => {
                    drop(launch);
                    prepared.finish_serial(observer)
                }
            },
            ReservePair::Serial(_reason) => prepared.finish_serial(observer),
            ReservePair::Stopped => {
                context.check_cancelled()?;
                Err(crate::CalcFlowError::Cancelled {
                    run_id: context.job().job_id().to_string(),
                })
            }
        }
    }

    pub(super) async fn emit_ordered(
        &mut self,
        records: Vec<RecordBatch>,
        last_identity: Option<Vec<u8>>,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        let observer = context.rolling_metrics();
        let removed = match super::budget::ordered_buffer_charge(
            records.iter(),
            &self.compiled,
            context.operator_id(),
        ) {
            Ok(Some(charge)) => charge,
            Ok(None) => {
                return self.restore_ordered_failure(
                    records,
                    last_identity,
                    internal_error("ordered rolling state has unsupported state charge"),
                    context.operator_id(),
                );
            }
            Err(error) => {
                return self.restore_ordered_failure(
                    records,
                    last_identity,
                    error,
                    context.operator_id(),
                );
            }
        };
        let combined = {
            let _stage = observer.map(|recorder| recorder.stage(RollingStage::ArrowOutput));
            combine_ordered_records(&records, context.operator_id())
        };
        let input = match combined {
            Ok(Some(input)) => input,
            Ok(None) => return Ok(()),
            Err(error) => {
                return self.restore_ordered_failure(
                    records,
                    last_identity,
                    error,
                    context.operator_id(),
                );
            }
        };
        let was_warm = self.state.typed_kernel_state.is_some();
        let prepared = self.prepare_ordered_output(&input, context).await;
        let mut prepared = match prepared {
            Ok(prepared) => prepared,
            Err(error) => {
                return self.restore_ordered_failure(
                    records,
                    last_identity,
                    error,
                    context.operator_id(),
                );
            }
        };
        let (current_charge, next_charge) = match self.projected_ordered_charge(
            &prepared,
            removed,
            was_warm,
            context.operator_id(),
        ) {
            Ok(charge) => charge,
            Err(error) => {
                return self.restore_ordered_failure(
                    records,
                    last_identity,
                    error,
                    context.operator_id(),
                );
            }
        };
        self.state.charge = current_charge;
        for batch in std::mem::take(&mut prepared.batches) {
            output.emit("output", batch).await?;
        }
        let _stage = observer.map(|recorder| recorder.stage(RollingStage::HistoryMaintenance));
        prepared.commit(self);
        self.state.charge = next_charge;
        Ok(())
    }
}

fn combine_ordered_records(records: &[RecordBatch], node_id: &str) -> Result<Option<RecordBatch>> {
    let Some(first) = records.first() else {
        return Ok(None);
    };
    if records.len() == 1 {
        return Ok(Some(first.clone()));
    }
    concat_batches(&first.schema(), records)
        .map(Some)
        .map_err(|error| operator_error(node_id, &error.to_string()))
}

struct PreparedOrderedOutput {
    update: StreamKernelUpdate,
    touched: Vec<RetainedHistoryAppend>,
    ewma_seeds: Vec<Vec<Option<(u64, f64)>>>,
    batches: Vec<Batch>,
    next_sequence: u64,
}

impl PreparedOrderedOutput {
    fn projected_charge(
        &self,
        operator: &RollingOperator,
        current: super::budget::StateCharge,
        node: &str,
    ) -> Result<super::budget::StateCharge> {
        let retention = usize::try_from(operator.compiled.max_row_retention).unwrap_or(usize::MAX);
        let mut old = super::budget::StateCharge::default();
        let mut new = super::budget::StateCharge::default();
        for tail in &self.touched {
            let previous = operator.state.histories.by_entity.get(&tail.entity);
            if let Some(previous) = previous {
                old = old.checked_add(
                    super::budget::history_entity_charge(&tail.entity, previous, node)?,
                    node,
                )?;
            }
            let seeds = self
                .ewma_seeds
                .get(tail.entity_id)
                .expect("every touched entity has a prepared EWMA seed");
            new = new.checked_add(
                tail.projected_charge(previous, retention, &operator.compiled, seeds, node)?,
                node,
            )?;
        }
        current.checked_sub(old)?.checked_add(new, node)
    }

    fn commit(self, operator: &mut RollingOperator) {
        let retention = usize::try_from(operator.compiled.max_row_retention).unwrap_or(usize::MAX);
        for tail in self.touched {
            let seeds = self
                .ewma_seeds
                .get(tail.entity_id)
                .expect("every touched entity has a prepared EWMA seed");
            tail.commit(
                &mut operator.state.histories,
                retention,
                &operator.compiled,
                seeds,
            );
        }
        self.update.commit(
            operator
                .state
                .typed_kernel_state
                .as_deref_mut()
                .expect("state initialized above"),
        );
        operator.state.next_output_sequence = self.next_sequence;
    }
}

struct EntityTail {
    entity_id: usize,
    first: usize,
    transitions: u64,
    rows: VecDeque<usize>,
}

/// Stage only newly retained rows. Existing tails remain owned and untouched
/// until all output chunks have been emitted successfully.
struct RetainedHistoryAppend {
    entity_id: usize,
    entity: Vec<Option<KeyValue>>,
    rows: RetainedRows,
    prepared_front: Option<RecordBatch>,
    transition_count: u64,
}

enum RetainedRows {
    Columnar(RecordBatch),
    Scalar(VecDeque<Vec<ScalarValue>>),
}

impl RetainedRows {
    fn len(&self) -> usize {
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
    rows: usize,
}

impl ColumnarHistory {
    fn prepare_discard_front(
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
    fn projected_charge(
        &self,
        previous: Option<&EntityRollingState>,
        retention: usize,
        compiled: &CompiledRollingSpec,
        seeds: &[Option<(u64, f64)>],
        node: &str,
    ) -> Result<super::budget::StateCharge> {
        let windows = if seeds.iter().any(Option::is_some) {
            compiled.window_groups.len()
        } else {
            0
        };
        let mut charge = super::budget::history_entity_base_charge(&self.entity, windows, node)?;
        if let Some(previous) = previous {
            let keep = retention.saturating_sub(self.rows.len());
            let discard = (previous.rows.len() + previous.columnar.rows).saturating_sub(keep);
            let scalar_discard = discard.min(previous.rows.len());
            for row in previous.rows.iter().skip(scalar_discard) {
                charge = charge
                    .checked_add(super::budget::scalar_history_row_charge(row, node)?, node)?;
            }
            let mut columnar_discard = discard - scalar_discard;
            for record in &previous.columnar.records {
                if columnar_discard >= record.num_rows() {
                    columnar_discard -= record.num_rows();
                    continue;
                }
                let retained = if columnar_discard == 0 {
                    record
                } else {
                    self.prepared_front
                        .as_ref()
                        .expect("partial discard has a prepared front")
                };
                let retained_charge = super::budget::ordered_record_charge(retained, node)?
                    .ok_or_else(|| internal_error("columnar history has unsupported charge"))?;
                charge = charge.checked_add(retained_charge, node)?;
                columnar_discard = 0;
            }
        }
        match &self.rows {
            RetainedRows::Columnar(record) if record.num_rows() != 0 => {
                let appended = super::budget::ordered_record_charge(record, node)?
                    .ok_or_else(|| internal_error("columnar tail has unsupported charge"))?;
                charge = charge.checked_add(appended, node)?;
            }
            RetainedRows::Columnar(_) => {}
            RetainedRows::Scalar(rows) => {
                for row in rows {
                    charge = charge
                        .checked_add(super::budget::scalar_history_row_charge(row, node)?, node)?;
                }
            }
        }
        Ok(charge)
    }

    fn commit(
        self,
        histories: &mut RollingHistories,
        retention: usize,
        compiled: &CompiledRollingSpec,
        seeds: &[Option<(u64, f64)>],
    ) {
        let state = histories.by_entity.entry(self.entity).or_default();
        let keep = retention.saturating_sub(self.rows.len());
        let discard = (state.rows.len() + state.columnar.rows).saturating_sub(keep);
        let scalar_discard = discard.min(state.rows.len());
        for _ in 0..scalar_discard {
            state.rows.pop_front();
        }
        state
            .columnar
            .discard_front(discard - scalar_discard, self.prepared_front);
        match self.rows {
            RetainedRows::Columnar(record) if record.num_rows() != 0 => state.columnar.push(record),
            RetainedRows::Columnar(_) => {}
            RetainedRows::Scalar(rows) => state.rows.extend(rows),
        }
        state.windows.clear();
        if seeds.iter().any(Option::is_some) {
            state.windows = fresh_windows(compiled);
            for (window, seed) in state.windows.iter_mut().zip(seeds) {
                if let (WindowState::Ewma(window), Some((valid_count, value))) = (window, seed) {
                    window.valid_count = *valid_count;
                    window.value = *value;
                }
            }
        }
        state.transition_count = self.transition_count;
    }
}

fn entity_tails(
    entity_ids: &[usize],
    retention: usize,
    node_id: &str,
) -> Result<impl Iterator<Item = EntityTail>> {
    let mut tails = Vec::<EntityTail>::new();
    for (row, &entity_id) in entity_ids.iter().enumerate() {
        if entity_id == tails.len() {
            tails.push(EntityTail {
                entity_id,
                first: row,
                transitions: 0,
                rows: VecDeque::new(),
            });
        }
        let tail = tails
            .get_mut(entity_id)
            .ok_or_else(|| internal_error("prepared rolling entity IDs are not dense"))?;
        tail.transitions = tail
            .transitions
            .checked_add(1)
            .ok_or_else(|| operator_error(node_id, "rolling entity transition count overflowed"))?;
        tail.rows.push_back(row);
        if tail.rows.len() > retention {
            tail.rows.pop_front();
        }
    }
    Ok(tails.into_iter())
}

impl EntityTail {
    fn prepare(
        self,
        input: &RecordBatch,
        histories: &RollingHistories,
        compiled: &CompiledRollingSpec,
        columnar: bool,
        node_id: &str,
        observer: Option<&RollingMetricsRecorder>,
    ) -> Result<RetainedHistoryAppend> {
        let entity = compiled
            .partition_columns
            .iter()
            .map(|column| {
                let value = ScalarValue::try_from_array(input.column(column.index), self.first)
                    .map_err(|error| operator_error(node_id, &error.to_string()))?;
                if let Some(recorder) = observer {
                    recorder.add(RollingWork::ScalarValueConversions, 1);
                }
                KeyValue::from_nullable_scalar(&value, node_id)
            })
            .collect::<Result<Vec<_>>>()?;
        let prior = histories.by_entity.get(&entity);
        let transition_count = prior
            .map_or(0, |state| state.transition_count)
            .checked_add(self.transitions)
            .ok_or_else(|| operator_error(node_id, "rolling entity transition count overflowed"))?;
        let rows = if self.rows.is_empty() {
            RetainedRows::Scalar(VecDeque::new())
        } else if columnar {
            let indices = UInt64Array::from_iter_values(
                self.rows
                    .into_iter()
                    .map(|index| u64::try_from(index).expect("Arrow row index fits u64")),
            );
            let record = take_record_batch(input, &indices)
                .map_err(|error| operator_error(node_id, &error.to_string()))?;
            RetainedRows::Columnar(record)
        } else {
            let rows = self
                .rows
                .into_iter()
                .map(|index| {
                    let row = read_buffered_row(input, index, compiled, node_id)?;
                    if let Some(recorder) = observer {
                        recorder.add(RollingWork::HistoryRowsMaterialized, 1);
                        recorder.add(RollingWork::ScalarValueConversions, input.num_columns());
                    }
                    Ok(row.values)
                })
                .collect::<Result<VecDeque<_>>>()?;
            RetainedRows::Scalar(rows)
        };
        let retention = usize::try_from(compiled.max_row_retention).unwrap_or(usize::MAX);
        let keep = retention.saturating_sub(rows.len());
        let prepared_front = prior
            .map(|state| {
                let discard = (state.rows.len() + state.columnar.rows).saturating_sub(keep);
                state
                    .columnar
                    .prepare_discard_front(discard.saturating_sub(state.rows.len()), node_id)
            })
            .transpose()?
            .flatten();
        Ok(RetainedHistoryAppend {
            entity_id: self.entity_id,
            entity,
            rows,
            prepared_front,
            transition_count,
        })
    }
}

fn retained_histories(
    input: &RecordBatch,
    entity_ids: &[usize],
    histories: &RollingHistories,
    compiled: &CompiledRollingSpec,
    node_id: &str,
    observer: Option<&RollingMetricsRecorder>,
) -> Result<Vec<RetainedHistoryAppend>> {
    let retention = usize::try_from(compiled.max_row_retention).unwrap_or(usize::MAX);
    let columnar = input.columns().iter().all(|column| {
        column.data_type().primitive_width().is_some()
            || matches!(
                column.data_type(),
                DataType::Null
                    | DataType::Boolean
                    | DataType::Utf8
                    | DataType::LargeUtf8
                    | DataType::Binary
                    | DataType::LargeBinary
            )
    });
    entity_tails(entity_ids, retention, node_id)?
        .map(|tail| tail.prepare(input, histories, compiled, columnar, node_id, observer))
        .collect()
}
