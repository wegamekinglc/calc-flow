//! Arrow buffering and bounded retained payloads for proven ordered streams.
//!
//! Finality remains watermark driven. Unsupported shapes, late envelopes, and
//! overlapping or unordered arrivals enter the existing general row path.

use datafusion::arrow::{
    array::{TimestampMicrosecondArray, UInt64Array},
    compute::take_record_batch,
    datatypes::DataType,
};

pub(super) use super::history::ColumnarHistory;
use super::history::{RetainedHistoryAppend, RetainedRows, RetentionPolicy};
use super::kernel::{EntityRows, RecurrenceSeed, StreamKernelUpdate};
use crate::operator::rolling_metrics::{
    RollingMetricsRecorder, RollingStage, RollingWork, RollingWorkBatch,
};
use crate::runtime::streaming::entity_work::{ReservePair, SerialReason};

use super::{
    Arc, Array, BTreeMap, Batch, BufferedRow, CompiledRollingSpec, EventTime, KeyValue,
    RecordBatch, Result, RollingHistories, RollingOperator, RollingStreamState, RowIdentity,
    ScalarValue, StreamCollector, StreamOperatorContext, TableBatch, VecDeque, chunk_output_record,
    closing_coordinate, concat_batches, internal_error, operator_error, read_buffered_row,
    reconstruct_typed_state,
};

#[derive(Clone, Default)]
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
        observer: Option<&RollingMetricsRecorder>,
    ) -> Result<()> {
        self.state
            .owned_mut()?
            .ordered
            .restore_front(records, last_identity);
        if let Ok(charge) = super::budget::full_state_charge_observed(
            &self.state.buffer,
            &self.state.ordered,
            &self.state.histories,
            &self.compiled,
            node,
            observer,
        ) {
            self.state.owned_mut()?.charge = charge;
        }
        Err(error)
    }

    fn projected_ordered_charge(
        &self,
        prepared: &mut PreparedOrderedOutput,
        removed: super::budget::StateCharge,
        node: &str,
        observer: Option<&RollingMetricsRecorder>,
    ) -> Result<(super::budget::StateCharge, super::budget::StateCharge)> {
        let current = self.state.charge.checked_sub(removed)?;
        let next = prepared.projected_charge(self, current, node, observer)?;
        self.check_state_budget(next, node)?;
        Ok((current, next))
    }

    fn supports_ordered_buffer(&self) -> bool {
        self.compiled.kernel_plan.supports_typed_transition() && self.state.buffer.is_empty()
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
        self.state.owned_mut()?.ordered.records.extend(prepared);
        self.state.owned_mut()?.ordered.last_identity = last;
        self.state.owned_mut()?.charge = next_charge;
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
        self.state.owned_mut()?.buffer.extend(materialized);
        self.state.owned_mut()?.ordered.clear();
        self.state.owned_mut()?.charge = next_charge;
        Ok(())
    }

    pub(super) fn prepare_ordered_buffer(
        &self,
        observer: Option<&RollingMetricsRecorder>,
    ) -> Result<BTreeMap<RowIdentity, BufferedRow>> {
        let _stage = observer.map(|recorder| recorder.stage(RollingStage::InputValidation));
        let mut work = RollingWorkBatch::new(observer);
        let mut materialized = BTreeMap::new();
        for record in &self.state.ordered.records {
            for index in 0..record.num_rows() {
                let row = read_buffered_row(record, index, &self.compiled, &self.name)?;
                work.add(RollingWork::ScalarValueConversions, record.num_columns());
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
            self.materialize_histories_with_charge(node_id, observer)?;
            let restored = reconstruct_typed_state(
                &self.state.histories,
                &self.compiled,
                &input.schema(),
                node_id,
            )?;
            self.state.owned_mut()?.typed_kernel_state = Some(Box::new(restored));
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
        self.ensure_ordered_kernel_state(input, context.operator_id(), observer)?;
        let mut update = self.prepare_ordered_update(input, context).await?;
        let touched = {
            let _stage = observer.map(|recorder| recorder.stage(RollingStage::HistoryMaintenance));
            retained_histories(
                input,
                &update,
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
    ) -> Result<StreamKernelUpdate> {
        let observer = context.rolling_metrics();
        let plan = &self.compiled.kernel_plan;
        let prior = self
            .state
            .typed_kernel_state
            .as_deref()
            .expect("state initialized above");
        let Some(client) = context.entity_work() else {
            return self.serial_ordered_update(
                input,
                context,
                RollingWork::ParallelFallbackNoWorkClient,
            );
        };
        if let Some(reason) = plan.entity_parallel_fallback(input) {
            return self.serial_ordered_update(input, context, reason);
        }
        let cpus = context.job().gather_owner().parallelism();
        #[cfg(test)]
        let cpus = client.parallelism(cpus);
        if cpus < 2 {
            return self.serial_ordered_update(
                input,
                context,
                RollingWork::ParallelFallbackCpuLimit,
            );
        }
        let prepared =
            plan.prepare_ordered_stream_inputs(prior, input, context.operator_id(), observer)?;
        let prepared = match prepared.qualify_parallel(cpus) {
            Ok(prepared) => prepared,
            Err((prepared, reason)) => {
                if let Some(recorder) = observer {
                    recorder.add(reason, 1);
                }
                return prepared.finish_serial(observer);
            }
        };
        match client.try_reserve(prepared.scratch()) {
            ReservePair::Reserved(launch) => match prepared.split() {
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
                    if let Some(recorder) = observer {
                        recorder.add(RollingWork::ParallelFallbackLanePartitionRejected, 1);
                    }
                    prepared.finish_serial(observer)
                }
            },
            ReservePair::Serial(reason) => {
                let work = match reason {
                    SerialReason::Busy => RollingWork::ParallelFallbackOwnerBusy,
                    SerialReason::AlreadyUsed => RollingWork::ParallelFallbackOwnerAlreadyUsed,
                    SerialReason::GenerationExhausted => {
                        RollingWork::ParallelFallbackGenerationExhausted
                    }
                    SerialReason::Unrepresentable => {
                        RollingWork::ParallelFallbackScratchUnrepresentable
                    }
                };
                if let Some(recorder) = observer {
                    recorder.add(work, 1);
                }
                prepared.finish_serial(observer)
            }
            ReservePair::Stopped => {
                context.check_cancelled()?;
                Err(crate::CalcFlowError::Cancelled {
                    run_id: context.job().job_id().to_string(),
                })
            }
        }
    }

    fn serial_ordered_update(
        &self,
        input: &RecordBatch,
        context: &StreamOperatorContext<'_>,
        reason: RollingWork,
    ) -> Result<StreamKernelUpdate> {
        let observer = context.rolling_metrics();
        let mut work = RollingWorkBatch::new(observer);
        work.add(reason, 1);
        work.add(RollingWork::NumericSerialCalls, 1);
        self.compiled
            .kernel_plan
            .prepare_ordered_stream_with_history(
                self.state
                    .typed_kernel_state
                    .as_deref()
                    .expect("state initialized above"),
                input,
                Some(&self.state.histories),
                context.operator_id(),
                observer,
            )
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
                    observer,
                );
            }
            Err(error) => {
                return self.restore_ordered_failure(
                    records,
                    last_identity,
                    error,
                    context.operator_id(),
                    observer,
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
                    observer,
                );
            }
        };
        let prepared = self.prepare_ordered_output(&input, context).await;
        let mut prepared = match prepared {
            Ok(prepared) => prepared,
            Err(error) => {
                return self.restore_ordered_failure(
                    records,
                    last_identity,
                    error,
                    context.operator_id(),
                    observer,
                );
            }
        };
        let (current_charge, next_charge) = match self.projected_ordered_charge(
            &mut prepared,
            removed,
            context.operator_id(),
            observer,
        ) {
            Ok(charge) => charge,
            Err(error) => {
                return self.restore_ordered_failure(
                    records,
                    last_identity,
                    error,
                    context.operator_id(),
                    observer,
                );
            }
        };
        let state = self.state.owned_mut()?;
        state.charge = current_charge;
        for batch in std::mem::take(&mut prepared.batches) {
            output.emit("output", batch).await?;
        }
        let _stage = observer.map(|recorder| recorder.stage(RollingStage::HistoryMaintenance));
        prepared.commit(state);
        state.charge = next_charge;
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
    fn projected_charge(
        &mut self,
        operator: &RollingOperator,
        current: super::budget::StateCharge,
        node: &str,
        observer: Option<&RollingMetricsRecorder>,
    ) -> Result<super::budget::StateCharge> {
        let mut work = RollingWorkBatch::new(observer);
        let mut old = super::budget::StateCharge::default();
        let mut new = super::budget::StateCharge::default();
        for tail in &mut self.touched {
            let previous = operator.state.histories.by_entity.get(&tail.entity);
            let (previous, projected) = tail.changed_charge(previous, node, &mut work)?;
            old = old.checked_add(previous, node)?;
            new = new.checked_add(projected, node)?;
        }
        current.checked_sub(old)?.checked_add(new, node)
    }

    fn commit(self, state: &mut RollingStreamState) {
        for tail in self.touched {
            tail.commit(&mut state.histories);
        }
        self.update.commit(
            state
                .typed_kernel_state
                .as_deref_mut()
                .expect("state initialized above"),
        );
        state.next_output_sequence = self.next_sequence;
    }
}

struct RetainedInput<'a> {
    record: &'a RecordBatch,
    costs: PayloadCosts,
    retention: RetentionPolicy,
    times: &'a [i64],
}

struct PayloadCosts {
    columnar: Option<usize>,
    scalar_per_row: usize,
}

impl PayloadCosts {
    fn new(input: &RecordBatch) -> Self {
        let supported = input.columns().iter().all(|column| {
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
        let columnar = supported.then(|| {
            input.columns().iter().fold(
                size_of::<RecordBatch>().saturating_add(
                    input
                        .num_columns()
                        .saturating_mul(size_of::<super::ArrayRef>()),
                ),
                |cost, column| {
                    cost.saturating_add(
                        column
                            .get_array_memory_size()
                            .saturating_sub(column.get_buffer_memory_size()),
                    )
                },
            )
        });
        Self {
            columnar,
            scalar_per_row: size_of::<Vec<ScalarValue>>()
                .saturating_add(input.num_columns().saturating_mul(size_of::<ScalarValue>())),
        }
    }

    fn columnar(&self, rows: usize) -> bool {
        self.columnar
            .is_some_and(|cost| cost <= self.scalar_per_row.saturating_mul(rows))
    }
}

struct EntityTail {
    entity_id: usize,
    first: usize,
    transitions: u64,
    /// Rows this batch contributes to the retained history.
    retained: usize,
    rows: VecDeque<usize>,
    last: usize,
    input_count: usize,
    seen: usize,
    finished: bool,
}

/// Builds each touched entity's retained tail from its routed row count,
/// walking back from the batch end only until every tail is complete.
fn entity_tails(
    entity_ids: &[usize],
    rows: &EntityRows,
    retention: RetentionPolicy,
    times: &[i64],
    node_id: &str,
) -> Result<std::vec::IntoIter<EntityTail>> {
    if rows.counts.iter().sum::<usize>() != entity_ids.len() {
        return Err(inconsistent_entity_rows());
    }
    if retention.duration.is_some() && times.len() != entity_ids.len() {
        return Err(inconsistent_entity_rows());
    }
    let mut tails = rows
        .counts
        .iter()
        .zip(&rows.first_rows)
        .enumerate()
        .map(|(entity_id, (&count, &first))| {
            EntityTail::new(entity_id, first, count, retention, node_id)
        })
        .collect::<Result<Vec<_>>>()?;
    collect_tail_rows(&mut tails, entity_ids, times, retention)?;
    Ok(tails.into_iter())
}

/// Fills every tail's newest retained rows in ascending row order.
fn collect_tail_rows(
    tails: &mut [EntityTail],
    entity_ids: &[usize],
    times: &[i64],
    retention: RetentionPolicy,
) -> Result<()> {
    let mut pending = tails.iter().filter(|tail| !tail.finished).count();
    for (row, &entity_id) in entity_ids.iter().enumerate().rev() {
        if pending == 0 {
            return Ok(());
        }
        let tail = tails
            .get_mut(entity_id)
            .ok_or_else(|| internal_error("prepared rolling entity IDs are not dense"))?;
        if tail.finished {
            continue;
        }
        if tail.seen == 0 {
            tail.last = row;
        }
        let time = times.get(row).copied().unwrap_or_default();
        let last = times.get(tail.last).copied().unwrap_or_default();
        let keep = retention.keeps_new(tail.rows.len(), time, last);
        tail.seen += 1;
        if keep {
            tail.rows.push_front(row);
        }
        if !keep
            || tail.seen == tail.input_count
            || (retention.duration.is_none() && tail.rows.len() == tail.retained)
        {
            tail.finished = true;
            pending -= 1;
        }
    }
    if pending == 0 {
        Ok(())
    } else {
        Err(inconsistent_entity_rows())
    }
}

fn inconsistent_entity_rows() -> super::CalcFlowError {
    internal_error("prepared rolling entity rows are inconsistent")
}

impl EntityTail {
    fn new(
        entity_id: usize,
        first: usize,
        count: usize,
        retention: RetentionPolicy,
        node_id: &str,
    ) -> Result<Self> {
        let transitions = u64::try_from(count)
            .map_err(|_| operator_error(node_id, "rolling entity transition count overflowed"))?;
        let retained = count.min(retention.rows);
        Ok(Self {
            entity_id,
            first,
            transitions,
            retained,
            rows: VecDeque::with_capacity(if retention.duration.is_some() {
                count.min(retained.max(1))
            } else {
                retained
            }),
            last: first,
            input_count: count,
            seen: 0,
            finished: count == 0 || (retention.duration.is_none() && retained == 0),
        })
    }

    fn prepare(
        self,
        input: &RetainedInput<'_>,
        histories: &RollingHistories,
        compiled: &CompiledRollingSpec,
        seeds: &[Option<RecurrenceSeed>],
        node_id: &str,
        work: &mut RollingWorkBatch<'_>,
    ) -> Result<RetainedHistoryAppend> {
        let source = input;
        let input = source.record;
        let entity = compiled
            .partition_columns
            .iter()
            .map(|column| {
                let value = ScalarValue::try_from_array(input.column(column.index), self.first)
                    .map_err(|error| operator_error(node_id, &error.to_string()))?;
                work.add(RollingWork::ScalarValueConversions, 1);
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
        } else if source.costs.columnar(self.rows.len()) {
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
                    let values = input
                        .columns()
                        .iter()
                        .map(|column| {
                            ScalarValue::try_from_array(column, index)
                                .map_err(|error| operator_error(node_id, &error.to_string()))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    work.add(RollingWork::HistoryRowsMaterialized, 1);
                    work.add(RollingWork::ScalarValueConversions, input.num_columns());
                    Ok(values)
                })
                .collect::<Result<VecDeque<_>>>()?;
            RetainedRows::Scalar(rows)
        };
        let discarded_rows = source.retention.discarded_history(
            prior,
            self.input_count,
            source.times[self.last],
            compiled,
            node_id,
        )?;
        let prepared_front = prior
            .map(|state| {
                state
                    .columnar
                    .prepare_discard_front(discarded_rows.saturating_sub(state.rows.len()), node_id)
            })
            .transpose()?
            .flatten();
        work.add(RollingWork::HistoryAppendRowsPrepared, rows.len());
        work.add(RollingWork::HistoryEvictRowsPrepared, discarded_rows);
        Ok(RetainedHistoryAppend {
            entity,
            rows,
            prepared_front,
            discarded_rows,
            windows: super::state_codec::recurrence_windows(compiled, seeds),
            transition_count,
            next_charge: None,
        })
    }
}

fn retained_histories(
    input: &RecordBatch,
    update: &StreamKernelUpdate,
    histories: &RollingHistories,
    compiled: &CompiledRollingSpec,
    node_id: &str,
    observer: Option<&RollingMetricsRecorder>,
) -> Result<Vec<RetainedHistoryAppend>> {
    let mut work = RollingWorkBatch::new(observer);
    let retention = RetentionPolicy::new(compiled);
    let times = timestamps(input, compiled)?;
    let input = RetainedInput {
        record: input,
        costs: PayloadCosts::new(input),
        retention,
        times: times.values(),
    };
    let seeds = update.recurrence_seeds();
    entity_tails(
        update.entity_ids(),
        update.entity_rows(),
        retention,
        input.times,
        node_id,
    )?
    .map(|tail| {
        let seeds = &seeds[tail.entity_id];
        tail.prepare(&input, histories, compiled, seeds, node_id, &mut work)
    })
    .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use super::{EntityRows, RetentionPolicy, entity_tails};

    /// Entity ID, first row, transitions, and retained rows of one tail.
    type Tail = (usize, usize, u64, Vec<usize>);

    /// The historical forward walk over every row.
    fn forward_tails(entity_ids: &[usize], retention: usize) -> Vec<Tail> {
        let mut tails = Vec::<(usize, usize, u64, VecDeque<usize>)>::new();
        for (row, &entity_id) in entity_ids.iter().enumerate() {
            if entity_id == tails.len() {
                tails.push((entity_id, row, 0, VecDeque::new()));
            }
            let tail = &mut tails[entity_id];
            tail.2 += 1;
            tail.3.push_back(row);
            if tail.3.len() > retention {
                tail.3.pop_front();
            }
        }
        tails
            .into_iter()
            .map(|(entity, first, transitions, rows)| (entity, first, transitions, rows.into()))
            .collect()
    }

    fn dense(keys: &[u8]) -> (Vec<usize>, EntityRows) {
        let mut distinct = Vec::new();
        let mut rows = EntityRows::default();
        let entity_ids = keys
            .iter()
            .enumerate()
            .map(|(row, key)| {
                let id = distinct
                    .iter()
                    .position(|seen| seen == key)
                    .unwrap_or_else(|| {
                        distinct.push(*key);
                        rows.counts.push(0);
                        rows.first_rows.push(row);
                        distinct.len() - 1
                    });
                rows.counts[id] += 1;
                id
            })
            .collect();
        (entity_ids, rows)
    }

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig {
            cases: 512,
            failure_persistence: None,
            ..proptest::prelude::ProptestConfig::default()
        })]

        #[test]
        fn routed_entity_tails_match_the_forward_walk(
            keys in proptest::collection::vec(0_u8..6, 0..48),
            retention in 0_usize..8,
        ) {
            let (entity_ids, rows) = dense(&keys);
            let tails = entity_tails(&entity_ids, &rows, RetentionPolicy { rows: retention, duration: None }, &[], "r")
                .unwrap()
                .map(|tail| (tail.entity_id, tail.first, tail.transitions, tail.rows.into()))
                .collect::<Vec<Tail>>();
            proptest::prop_assert_eq!(tails, forward_tails(&entity_ids, retention));
        }
    }

    #[test]
    fn test_duration_tail_counts_every_transition_before_the_retained_suffix() {
        let times = (1..=136).collect::<Vec<i64>>();
        let routing = EntityRows {
            counts: vec![136],
            first_rows: vec![0],
        };
        let policy = RetentionPolicy {
            rows: 2,
            duration: Some(100),
        };
        let tail = entity_tails(&vec![0; 136], &routing, policy, &times, "r")
            .unwrap()
            .next()
            .unwrap();
        assert_eq!(tail.transitions, 136);
        assert_eq!(tail.first, 0);
        assert_eq!(
            tail.rows.into_iter().collect::<Vec<_>>(),
            (36..136).collect::<Vec<_>>()
        );
    }

    #[test]
    fn routed_entity_tails_reject_inconsistent_routing() {
        let rows = EntityRows {
            counts: vec![1],
            first_rows: vec![0],
        };
        assert!(
            entity_tails(
                &[0, 1],
                &rows,
                RetentionPolicy {
                    rows: 2,
                    duration: None
                },
                &[],
                "r"
            )
            .is_err()
        );
        assert!(
            entity_tails(
                &[0, 0],
                &rows,
                RetentionPolicy {
                    rows: 2,
                    duration: None
                },
                &[],
                "r"
            )
            .is_err()
        );
        let split = EntityRows {
            counts: vec![1, 1],
            first_rows: vec![0, 1],
        };
        assert!(
            entity_tails(
                &[0, 0],
                &split,
                RetentionPolicy {
                    rows: 2,
                    duration: None
                },
                &[],
                "r"
            )
            .is_err()
        );
    }
}
