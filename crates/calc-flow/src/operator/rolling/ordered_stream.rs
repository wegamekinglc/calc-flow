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
    Arc, Array, BTreeMap, Batch, CompiledRollingSpec, CompiledWindowGroup, EventTime, KeyValue,
    RecordBatch, Result, RollingHistories, RollingOperator, ScalarValue, StreamCollector,
    StreamOperatorContext, TableBatch, VecDeque, chunk_output_record, closing_coordinate,
    concat_batches, internal_error, operator_error, read_buffered_row, reconstruct_typed_state,
};

#[derive(Default)]
pub(super) struct OrderedStreamBuffer {
    records: VecDeque<RecordBatch>,
    last_identity: Option<Vec<u8>>,
}

impl OrderedStreamBuffer {
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
    fn supports_ordered_buffer(&self) -> bool {
        self.compiled.kernel_plan.supports_typed_transition()
            && self.compiled.max_duration_micros.is_none()
            && !self
                .compiled
                .window_groups
                .iter()
                .any(|group| matches!(group, CompiledWindowGroup::Ewma { .. }))
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
        self.state.ordered.records.extend(prepared);
        self.state.ordered.last_identity = last;
        Ok(true)
    }

    pub(super) fn materialize_ordered_buffer(
        &mut self,
        observer: Option<&RollingMetricsRecorder>,
    ) -> Result<()> {
        let _stage = observer.map(|recorder| recorder.stage(RollingStage::InputValidation));
        if self.state.ordered.is_empty() {
            return Ok(());
        }
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
        self.state.buffer.extend(materialized);
        self.state.ordered.take_all();
        Ok(())
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
        let (batches, next_sequence) = self.ordered_output_chunks(input, &mut update, context)?;
        Ok(PreparedOrderedOutput {
            update,
            touched,
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
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        let observer = context.rolling_metrics();
        let Some(input) = ({
            let _stage = observer.map(|recorder| recorder.stage(RollingStage::ArrowOutput));
            combine_ordered_records(&records, context.operator_id())?
        }) else {
            return Ok(());
        };
        let mut prepared = self.prepare_ordered_output(&input, context).await?;
        for batch in std::mem::take(&mut prepared.batches) {
            output.emit("output", batch).await?;
        }
        let _stage = observer.map(|recorder| recorder.stage(RollingStage::HistoryMaintenance));
        prepared.commit(self);
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
    batches: Vec<Batch>,
    next_sequence: u64,
}

impl PreparedOrderedOutput {
    fn commit(self, operator: &mut RollingOperator) {
        let retention = usize::try_from(operator.compiled.max_row_retention).unwrap_or(usize::MAX);
        for tail in self.touched {
            tail.commit(&mut operator.state.histories, retention);
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
    first: usize,
    transitions: u64,
    rows: VecDeque<usize>,
}

/// Stage only newly retained rows. Existing tails remain owned and untouched
/// until all output chunks have been emitted successfully.
struct RetainedHistoryAppend {
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
            let Some(logical_bytes) = super::row_cost::RowCosts::try_total(&retained)? else {
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
    fn commit(self, histories: &mut RollingHistories, retention: usize) {
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
            RetainedRows::Columnar(record) => state.columnar.push(record),
            RetainedRows::Scalar(rows) => state.rows.extend(rows),
        }
        state.windows.clear();
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
        let rows = if columnar {
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
