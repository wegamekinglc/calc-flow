use super::{JoinSide, StreamJoinOperator, counter_overflow, operator_error};
use crate::{Batch, BatchMetadata, Result, StreamCollector, StreamOperatorContext};
use datafusion::arrow::{
    array::{Array, StringArray, TimestampMicrosecondArray, UInt64Array},
    compute::{concat_batches, take},
    datatypes::{DataType, SchemaRef, TimeUnit},
    record_batch::{RecordBatch, RecordBatchOptions},
};
use std::{collections::BTreeMap, sync::Arc};

#[derive(Clone, Copy, Eq, PartialEq)]
pub(super) enum HotKind {
    Equi,
    Band,
}

pub(super) struct HotJoinCore {
    kind: HotKind,
    query: String,
    left: Option<Batch>,
    right: Option<Batch>,
    completed: bool,
}

impl HotJoinCore {
    pub(super) fn new(operator: &StreamJoinOperator) -> Option<Self> {
        if output_schema(operator).fields().is_empty() {
            return None;
        }
        let kind = if operator.spec.bounds.after_micros == 0 {
            HotKind::Equi
        } else if operator.spec.bounds.before_micros == operator.spec.bounds.after_micros
            && operator.spec.bounds.before_micros.is_multiple_of(1_000_000)
            && operator.spec.bounds.before_micros <= 60_000_000
        {
            HotKind::Band
        } else {
            return None;
        };
        if operator.spec.left_keys() != ["symbol"]
            || operator.spec.right_keys() != ["symbol"]
            || operator.spec.left_event_time() != "event_time"
            || operator.spec.right_event_time() != "event_time"
        {
            return None;
        }
        for side in [0, 1] {
            let schema = operator.input_schema(side);
            for (name, data_type) in [("symbol", DataType::Utf8), ("sequence", DataType::UInt64)] {
                let index = schema.index_of(name).ok()?;
                if schema.field(index).data_type() != &data_type {
                    return None;
                }
            }
            let index = schema.index_of("event_time").ok()?;
            if !matches!(
                schema.field(index).data_type(),
                DataType::Timestamp(TimeUnit::Microsecond, _)
            ) {
                return None;
            }
            let price = schema.index_of("price").ok();
            let factor = schema.index_of("factor").ok();
            if price.is_none() && (kind == HotKind::Band || factor.is_none()) {
                return None;
            }
            for index in [price, factor].into_iter().flatten() {
                if schema.field(index).data_type() != &DataType::Float64 {
                    return None;
                }
            }
        }
        let query = if kind == HotKind::Band {
            band_query(operator.spec.bounds.before_micros / 1_000_000)
        } else {
            String::new()
        };
        Some(Self {
            kind,
            query,
            left: None,
            right: None,
            completed: false,
        })
    }

    pub(super) async fn process(
        &mut self,
        operator: &mut StreamJoinOperator,
        ingress: &str,
        batch: &Batch,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        if self.completed {
            return Err(operator_error(
                &operator.name,
                "whole-batch hot Join accepts one input batch per side",
            ));
        }
        let plan = operator.begin_batch(ingress, batch)?;
        let side = if plan.incoming_is_left {
            JoinSide::Left
        } else {
            JoinSide::Right
        };
        match side {
            JoinSide::Left => self.left = Some(batch.clone()),
            JoinSide::Right => self.right = Some(batch.clone()),
        }
        let (Some(left), Some(right)) = (self.left.as_ref(), self.right.as_ref()) else {
            return Ok(());
        };
        #[cfg(feature = "benchmark-internals")]
        let execute_span =
            context
                .job()
                .join_cpu_trace()
                .stage(&operator.name, "hot_execute", batch.num_rows());
        let joined = match self.kind {
            HotKind::Equi => Self::execute_equi(operator, left, right)?,
            HotKind::Band => self.execute_band(operator, left, right).await?,
        };
        #[cfg(feature = "benchmark-internals")]
        drop(execute_span);
        let mut sequence = operator.state.next_output_sequence;
        let schema = output_schema(operator);
        #[cfg(feature = "benchmark-internals")]
        let emit_span =
            context
                .job()
                .join_cpu_trace()
                .stage(&operator.name, "hot_emit", joined.num_rows());
        for record in joined.table_payload()?.batches() {
            let record = RecordBatch::try_new_with_options(
                Arc::clone(&schema),
                record.columns().to_vec(),
                &RecordBatchOptions::new().with_row_count(Some(record.num_rows())),
            )
            .map_err(|error| operator_error(&operator.name, &error.to_string()))?;
            let metadata = BatchMetadata::new(&operator.name, sequence, BTreeMap::new())?;
            sequence = sequence
                .checked_add(1)
                .ok_or_else(|| counter_overflow(&operator.name, "output_sequence"))?;
            output
                .emit("output", Batch::table(vec![record], metadata)?)
                .await?;
        }
        #[cfg(feature = "benchmark-internals")]
        drop(emit_span);
        operator.state.next_output_sequence = sequence;
        operator.state.next_left_row_id = u64::try_from(left.num_rows())
            .map_err(|_| counter_overflow(&operator.name, "left_row_id"))?;
        operator.state.next_right_row_id = u64::try_from(right.num_rows())
            .map_err(|_| counter_overflow(&operator.name, "right_row_id"))?;
        operator.state.metrics.emitted_match_rows = operator
            .state
            .metrics
            .emitted_match_rows
            .checked_add(
                u64::try_from(joined.num_rows())
                    .map_err(|_| counter_overflow(&operator.name, "emitted_match_rows"))?,
            )
            .ok_or_else(|| counter_overflow(&operator.name, "emitted_match_rows"))?;
        operator.ingress_progress = context.ingress_progress().clone();
        self.completed = true;
        self.left = None;
        self.right = None;
        Ok(())
    }

    fn execute_equi(operator: &StreamJoinOperator, left: &Batch, right: &Batch) -> Result<Batch> {
        let left = concat_input(operator, left)?;
        let right = concat_input(operator, right)?;
        let (left_indices, right_indices) =
            temporal_indices(operator, &left, &right, operator.spec.bounds.before_micros);
        let schema = output_schema(operator);
        let left_width = operator.input_schema(0).fields().len();
        let sources = (0..schema.fields().len())
            .map(|ordinal| output_source(operator, ordinal))
            .collect::<Vec<_>>();
        let records = (0..left_indices.len())
            .step_by(65_536)
            .map(|start| {
                let length = (left_indices.len() - start).min(65_536);
                let left_indices = left_indices.slice(start, length);
                let right_indices = right_indices.slice(start, length);
                let columns = sources
                    .iter()
                    .map(|&source| {
                        let (input, indices, column) = if source < left_width {
                            (&left, &left_indices, source)
                        } else {
                            (&right, &right_indices, source - left_width)
                        };
                        take(input.column(column), &indices, None)
                            .map_err(|error| operator_error(&operator.name, &error.to_string()))
                    })
                    .collect::<Result<Vec<_>>>()?;
                RecordBatch::try_new(Arc::clone(&schema), columns)
                    .map_err(|error| operator_error(&operator.name, &error.to_string()))
            })
            .collect::<Result<Vec<_>>>()?;
        Batch::table(records, BatchMetadata::default())
    }

    async fn execute_band(
        &self,
        operator: &StreamJoinOperator,
        left: &Batch,
        right: &Batch,
    ) -> Result<Batch> {
        let tables = BTreeMap::from([
            ("input".into(), left.clone()),
            ("reference".into(), right.clone()),
        ]);
        let query = crate::expression::parse_select_query(&self.query)
            .map_err(|error| operator_error(&operator.name, &error.to_string()))?;
        let left_width = operator.input_schema(0).fields().len();
        let schema = output_schema(operator);
        let columns = (0..schema.fields().len())
            .map(|ordinal| {
                let source = output_source(operator, ordinal);
                if source < left_width {
                    crate::datafusion::datafusion_join::BandColumn::Left(source)
                } else {
                    crate::datafusion::datafusion_join::BandColumn::Right(source - left_width)
                }
            })
            .collect::<Vec<_>>();
        let plan = crate::datafusion::datafusion_join::plan_with_output(
            &query, &tables, schema, columns, 1,
        )
        .ok_or_else(|| operator_error(&operator.name, "native band planning is unsupported"))?;
        let session = datafusion::execution::context::SessionContext::new_with_config(
            datafusion::execution::context::SessionConfig::new().with_target_partitions(1),
        );
        let records = datafusion::physical_plan::collect(plan, session.task_ctx())
            .await
            .map_err(|error| operator_error(&operator.name, &error.to_string()))?;
        Batch::table(records, BatchMetadata::default())
    }
}

#[derive(Clone, Copy)]
struct TemporalEntry {
    time: i64,
    row: u64,
}

fn concat_input(operator: &StreamJoinOperator, input: &Batch) -> Result<RecordBatch> {
    let table = input.table_payload()?;
    concat_batches(table.schema(), table.batches())
        .map_err(|error| operator_error(&operator.name, &error.to_string()))
}

fn temporal_indices(
    operator: &StreamJoinOperator,
    left: &RecordBatch,
    right: &RecordBatch,
    before: u64,
) -> (UInt64Array, UInt64Array) {
    let right_keys = right
        .column(operator.compiled.right_key_indices[0])
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("hot Join key is Utf8");
    let right_times = right
        .column(operator.compiled.right_event_time_index)
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()
        .expect("hot Join time is microseconds");
    let mut entries =
        hashbrown::HashMap::<String, Vec<TemporalEntry>, ahash::RandomState>::default();
    for row in 0..right.num_rows() {
        if right_keys.is_null(row) || right_times.is_null(row) {
            continue;
        }
        entries
            .entry_ref(right_keys.value(row))
            .or_default()
            .push(TemporalEntry {
                time: right_times.value(row),
                row: u64::try_from(row).expect("Arrow row fits u64"),
            });
    }
    for values in entries.values_mut() {
        values.sort_unstable_by_key(|entry| (entry.time, entry.row));
    }

    let left_keys = left
        .column(operator.compiled.left_key_indices[0])
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("hot Join key is Utf8");
    let left_times = left
        .column(operator.compiled.left_event_time_index)
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()
        .expect("hot Join time is microseconds");
    let before = i128::from(before);
    let mut left_indices = Vec::new();
    let mut right_indices = Vec::new();
    for row in 0..left.num_rows() {
        if left_keys.is_null(row) || left_times.is_null(row) {
            continue;
        }
        let Some(candidates) = entries.get(left_keys.value(row)) else {
            continue;
        };
        let time = i128::from(left_times.value(row));
        let lower = time - before;
        let start = candidates.partition_point(|entry| i128::from(entry.time) < lower);
        let end = candidates.partition_point(|entry| i128::from(entry.time) <= time);
        let row = u64::try_from(row).expect("Arrow row fits u64");
        left_indices.extend(std::iter::repeat_n(row, end - start));
        right_indices.extend(candidates[start..end].iter().map(|entry| entry.row));
    }
    (
        UInt64Array::from_iter_values(left_indices),
        UInt64Array::from_iter_values(right_indices),
    )
}

fn output_schema(operator: &StreamJoinOperator) -> SchemaRef {
    Arc::clone(
        operator.output_ports[0]
            .schema()
            .expect("compiled Join output schema"),
    )
}

fn output_source(operator: &StreamJoinOperator, ordinal: usize) -> usize {
    let field = &output_schema(operator).fields[ordinal];
    operator
        .logical_output_schema
        .index_of(field.name())
        .expect("projected Join field has a logical source")
}

fn band_query(seconds: u64) -> String {
    let seconds = i64::try_from(seconds).expect("hot band is limited to 60 seconds");
    let offsets = (-seconds..=seconds)
        .map(|offset| format!("(INTERVAL '{offset} seconds')"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "WITH offsets(event_offset) AS (VALUES {offsets}) \
         SELECT probes.left_sequence AS sequence, \
         reference.sequence AS right_sequence, \
         probes.left_price * reference.price AS value \
         FROM reference JOIN (SELECT input.sequence AS left_sequence, \
         input.symbol AS left_symbol, input.price AS left_price, \
         input.event_time + offsets.event_offset AS match_time \
         FROM input CROSS JOIN offsets) AS probes \
         ON probes.left_symbol = reference.symbol \
         AND probes.match_time = reference.event_time"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CancellationToken, EdgeCollector, JoinStateLimits, JoinTimeBounds, JsonMap,
        OperatorMetadata, StreamJobContext, StreamJoinSpec, StreamOperator,
    };
    use datafusion::arrow::{
        array::{Float64Array, StringArray, TimestampMicrosecondArray, UInt64Array},
        datatypes::{Field, Schema},
    };
    use std::time::Duration;

    fn quote_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new(
                "event_time",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("sequence", DataType::UInt64, false),
            Field::new("symbol", DataType::Utf8, false),
            Field::new("price", DataType::Float64, false),
        ]))
    }

    fn dimension_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("symbol", DataType::Utf8, false),
            Field::new("factor", DataType::Float64, false),
            Field::new("sequence", DataType::UInt64, false),
            Field::new(
                "event_time",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
        ]))
    }

    fn spec() -> StreamJoinSpec {
        StreamJoinSpec::inner(
            ["symbol"],
            ["symbol"],
            "event_time",
            "event_time",
            JoinTimeBounds::new(Duration::from_secs(300), Duration::ZERO).unwrap(),
            JoinStateLimits::new(100, 1_000_000, 100).unwrap(),
        )
        .unwrap()
        .with_prefixes("quote", "dimension")
        .unwrap()
    }

    #[tokio::test]
    async fn hot_join_emits_the_complete_downstream_schema() {
        for projection in [None, Some(vec![1, 3, 5])] {
            let mut operator =
                StreamJoinOperator::new("hot", quote_schema(), dimension_schema(), spec()).unwrap();
            if let Some(columns) = &projection {
                operator.set_output_projection(columns.clone()).unwrap();
            }
            operator.enable_run_core();
            let left = RecordBatch::try_new(
                quote_schema(),
                vec![
                    Arc::new(
                        TimestampMicrosecondArray::from(vec![0, 1_000_000]).with_timezone("UTC"),
                    ),
                    Arc::new(UInt64Array::from(vec![0, 1])),
                    Arc::new(StringArray::from(vec!["S000", "S000"])),
                    Arc::new(Float64Array::from(vec![2.0, 3.0])),
                ],
            )
            .unwrap();
            let right = RecordBatch::try_new(
                dimension_schema(),
                vec![
                    Arc::new(StringArray::from(vec!["S000"])),
                    Arc::new(Float64Array::from(vec![2.0])),
                    Arc::new(UInt64Array::from(vec![0])),
                    Arc::new(TimestampMicrosecondArray::from(vec![1_000_000]).with_timezone("UTC")),
                ],
            )
            .unwrap();
            let job = StreamJobContext::new(
                1,
                "fingerprint",
                JsonMap::new(),
                None,
                CancellationToken::new(),
            )
            .with_checkpointing(false);
            let context = StreamOperatorContext::new(&job, "hot", None);
            let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
            operator
                .process_data(
                    "left",
                    Batch::table(vec![left], BatchMetadata::default()).unwrap(),
                    &context,
                    &mut collector,
                )
                .await
                .unwrap();
            operator
                .process_data(
                    "right",
                    Batch::table(vec![right], BatchMetadata::default()).unwrap(),
                    &context,
                    &mut collector,
                )
                .await
                .unwrap();

            let messages = collector.drain("output");
            assert_eq!(messages.len(), 1);
            let output = messages[0].as_data().unwrap();
            assert_eq!(output.num_rows(), 1);
            let record = &output.table_payload().unwrap().batches()[0];
            assert_eq!(
                record.schema().fields().len(),
                projection.as_ref().map_or(8, Vec::len)
            );
            let price = record
                .column(record.schema().index_of("quote__price").unwrap())
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            assert_eq!(price.values(), &[3.0]);
            let factor = record
                .column(record.schema().index_of("dimension__factor").unwrap())
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            assert_eq!(factor.values(), &[2.0]);
            assert_eq!(operator.state.metrics.emitted_match_rows, 1);
            let core = operator.run_core.as_ref().unwrap();
            assert!(core.hot_enabled());
            assert_eq!(core.rows(JoinSide::Left), 0);
            assert_eq!(core.rows(JoinSide::Right), 0);
        }
    }

    fn band_spec() -> StreamJoinSpec {
        StreamJoinSpec::inner(
            ["symbol"],
            ["symbol"],
            "event_time",
            "event_time",
            JoinTimeBounds::new(Duration::from_secs(5), Duration::from_secs(5)).unwrap(),
            JoinStateLimits::new(100, 1_000_000, 100).unwrap(),
        )
        .unwrap()
        .with_prefixes("left", "right")
        .unwrap()
    }

    #[tokio::test]
    async fn hot_band_emits_requested_input_columns() {
        for projection in [None, Some(vec![1, 3, 5, 7])] {
            let mut operator =
                StreamJoinOperator::new("hot-band", quote_schema(), quote_schema(), band_spec())
                    .unwrap();
            if let Some(columns) = &projection {
                operator.set_output_projection(columns.clone()).unwrap();
            }
            operator.enable_run_core();
            let record = RecordBatch::try_new(
                quote_schema(),
                vec![
                    Arc::new(
                        TimestampMicrosecondArray::from(vec![0, 1_000_000]).with_timezone("UTC"),
                    ),
                    Arc::new(UInt64Array::from(vec![0, 1])),
                    Arc::new(StringArray::from(vec!["S000", "S000"])),
                    Arc::new(Float64Array::from(vec![2.0, 3.0])),
                ],
            )
            .unwrap();
            let job = StreamJobContext::new(
                1,
                "fingerprint",
                JsonMap::new(),
                None,
                CancellationToken::new(),
            )
            .with_checkpointing(false);
            let context = StreamOperatorContext::new(&job, "hot-band", None);
            let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
            for side in ["left", "right"] {
                operator
                    .process_data(
                        side,
                        Batch::table(vec![record.clone()], BatchMetadata::default()).unwrap(),
                        &context,
                        &mut collector,
                    )
                    .await
                    .unwrap();
            }

            let messages = collector.drain("output");
            assert_eq!(messages.len(), 1);
            let output = messages[0].as_data().unwrap();
            assert_eq!(output.num_rows(), 4);
            let record = &output.table_payload().unwrap().batches()[0];
            assert_eq!(
                record.schema().fields().len(),
                projection.as_ref().map_or(8, Vec::len)
            );
            let left = record
                .column(record.schema().index_of("left__price").unwrap())
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            assert_eq!(left.values(), &[2.0, 2.0, 3.0, 3.0]);
            let right = record
                .column(record.schema().index_of("right__price").unwrap())
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            assert_eq!(right.values(), &[2.0, 3.0, 2.0, 3.0]);
            let right_sequence = record
                .column(record.schema().index_of("right__sequence").unwrap())
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            assert_eq!(right_sequence.values(), &[0, 1, 0, 1]);
            assert_eq!(operator.state.metrics.emitted_match_rows, 4);
            assert!(operator.run_core.as_ref().unwrap().hot_enabled());
        }
    }
}
