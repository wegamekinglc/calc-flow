//! Columnar buffering for proven ordered, bounded row-window streams.
//!
//! Finality remains watermark driven. Unsupported shapes, late envelopes, and
//! overlapping or unordered arrivals enter the existing general row path.

use datafusion::arrow::array::TimestampMicrosecondArray;

use super::{
    Arc, Array, BTreeMap, CompiledRollingSpec, CompiledWindowGroup, EventTime, KeyValue,
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
    pub(super) fn try_buffer_ordered(
        &mut self,
        table: &TableBatch,
        watermark: Option<EventTime>,
    ) -> Result<bool> {
        if !self.compiled.kernel_plan.supports_typed_transition()
            || self.compiled.max_duration_micros.is_some()
            || self
                .compiled
                .window_groups
                .iter()
                .any(|group| matches!(group, CompiledWindowGroup::Ewma { .. }))
            || !self.state.buffer.is_empty()
        {
            return Ok(false);
        }
        let mut last = self.state.ordered.last_identity.clone();
        let mut prepared = Vec::new();
        for record in table.batches() {
            if record.num_rows() == 0 {
                continue;
            }
            if let Some(watermark) = watermark {
                let times = timestamps(record, &self.compiled)?;
                // Lateness classification precedes duplicate rejection. In
                // particular, duplicate late rows under Drop must still drop.
                if times.values().iter().any(|&time| {
                    let closing = i128::from(time) + i128::from(self.spec.allowed_lateness_micros);
                    closing > i128::from(i64::MAX) || closing <= i128::from(watermark.as_micros())
                }) {
                    return Ok(false);
                }
            }
            let Some((first, end)) = self
                .compiled
                .kernel_plan
                .ordered_stream_bounds(record, &self.name)?
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

    pub(super) fn materialize_ordered_buffer(&mut self) -> Result<()> {
        if self.state.ordered.is_empty() {
            return Ok(());
        }
        let mut materialized = BTreeMap::new();
        for record in &self.state.ordered.records {
            for index in 0..record.num_rows() {
                let row = read_buffered_row(record, index, &self.compiled, &self.name)?;
                materialized.insert(row.identity.clone(), row);
            }
        }
        self.state.buffer.extend(materialized);
        self.state.ordered.take_all();
        Ok(())
    }

    pub(super) async fn emit_ordered(
        &mut self,
        records: Vec<RecordBatch>,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        let Some(first) = records.first() else {
            return Ok(());
        };
        let input = if records.len() == 1 {
            first.clone()
        } else {
            concat_batches(&first.schema(), &records)
                .map_err(|error| operator_error(context.operator_id(), &error.to_string()))?
        };
        if self.state.typed_kernel_state.is_none() {
            let restored = reconstruct_typed_state(
                &self.state.histories,
                &self.compiled,
                &input.schema(),
                context.operator_id(),
            )?;
            self.state.typed_kernel_state = Some(Box::new(restored));
        }
        let prior = self
            .state
            .typed_kernel_state
            .as_deref()
            .expect("state initialized above");
        let mut update = self.compiled.kernel_plan.prepare_ordered_stream(
            prior,
            &input,
            context.operator_id(),
        )?;
        let touched = retained_histories(
            &input,
            update.entity_ids(),
            &self.state.histories,
            &self.compiled,
            context.operator_id(),
        )?;
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
        let batches = chunk_output_record(
            &record,
            context.operator_id(),
            self.state.next_output_sequence,
            context.output_budget(),
        )?;
        let count = u64::try_from(batches.len())
            .map_err(|_| operator_error(context.operator_id(), "output chunk count overflowed"))?;
        let next_sequence = self
            .state
            .next_output_sequence
            .checked_add(count)
            .ok_or_else(|| operator_error(context.operator_id(), "output sequence overflowed"))?;
        for batch in batches {
            output.emit("output", batch).await?;
        }
        let retention = usize::try_from(self.compiled.max_row_retention).unwrap_or(usize::MAX);
        for tail in touched {
            tail.commit(&mut self.state.histories, retention);
        }
        update.commit(
            self.state
                .typed_kernel_state
                .as_deref_mut()
                .expect("state initialized above"),
        );
        self.state.next_output_sequence = next_sequence;
        Ok(())
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
    rows: VecDeque<Vec<ScalarValue>>,
    transition_count: u64,
}

impl RetainedHistoryAppend {
    fn commit(self, histories: &mut RollingHistories, retention: usize) {
        let state = histories.by_entity.entry(self.entity).or_default();
        let keep = retention.saturating_sub(self.rows.len());
        while state.rows.len() > keep {
            state.rows.pop_front();
        }
        state.rows.extend(self.rows);
        state.windows.clear();
        state.transition_count = self.transition_count;
    }
}

fn retained_histories(
    input: &RecordBatch,
    entity_ids: &[usize],
    histories: &RollingHistories,
    compiled: &CompiledRollingSpec,
    node_id: &str,
) -> Result<Vec<RetainedHistoryAppend>> {
    let retention = usize::try_from(compiled.max_row_retention).unwrap_or(usize::MAX);
    let mut tails = BTreeMap::<usize, EntityTail>::new();
    for (row, &entity_id) in entity_ids.iter().enumerate() {
        let tail = tails.entry(entity_id).or_insert_with(|| EntityTail {
            first: row,
            transitions: 0,
            rows: VecDeque::new(),
        });
        tail.transitions = tail
            .transitions
            .checked_add(1)
            .ok_or_else(|| operator_error(node_id, "rolling entity transition count overflowed"))?;
        tail.rows.push_back(row);
        if tail.rows.len() > retention {
            tail.rows.pop_front();
        }
    }
    tails
        .into_values()
        .map(|tail| {
            let sample = read_buffered_row(input, tail.first, compiled, node_id)?;
            let entity = sample.identity.entity;
            let prior = histories.by_entity.get(&entity);
            let mut first_values = Some(sample.values);
            let mut rows = VecDeque::with_capacity(tail.rows.len());
            for index in tail.rows {
                let values = if index == tail.first {
                    first_values.take().expect("first row occurs once")
                } else {
                    read_buffered_row(input, index, compiled, node_id)?.values
                };
                rows.push_back(values);
            }
            let transition_count = prior
                .map_or(0, |state| state.transition_count)
                .checked_add(tail.transitions)
                .ok_or_else(|| {
                    operator_error(node_id, "rolling entity transition count overflowed")
                })?;
            Ok(RetainedHistoryAppend {
                entity,
                rows,
                transition_count,
            })
        })
        .collect()
}
