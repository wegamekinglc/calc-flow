use super::{JoinSide, StreamJoinOperator, counter_overflow, operator_error};
use crate::{Batch, BatchMetadata, Result, StreamCollector, StreamOperatorContext};
use datafusion::arrow::{
    datatypes::{DataType, TimeUnit},
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
        if operator.output_columns.is_some() {
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
            let schema = operator.retained_schema(side);
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
        Some(Self {
            kind,
            query: query(operator, kind),
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
            HotKind::Equi => {
                let tables = BTreeMap::from([
                    ("input".into(), left.clone()),
                    ("reference".into(), right.clone()),
                ]);
                operator
                    .runtime
                    .runtime()?
                    .sql(&self.query, &tables, Some(&operator.name))
                    .await?
            }
            HotKind::Band => self.execute_band(operator, left, right).await?,
        };
        #[cfg(feature = "benchmark-internals")]
        drop(execute_span);
        let mut sequence = operator.state.next_output_sequence;
        let schema = Arc::clone(&operator.logical_output_schema);
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
        let left_width = operator.retained_schema(0).fields().len();
        let columns = (0..operator.logical_output_schema.fields().len())
            .map(|source| {
                if source < left_width {
                    crate::datafusion::datafusion_join::BandColumn::Left(source)
                } else {
                    crate::datafusion::datafusion_join::BandColumn::Right(source - left_width)
                }
            })
            .collect::<Vec<_>>();
        let plan = crate::datafusion::datafusion_join::plan_with_output(
            &query,
            &tables,
            Arc::clone(&operator.logical_output_schema),
            columns,
            1,
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

fn query(operator: &StreamJoinOperator, kind: HotKind) -> String {
    if kind == HotKind::Band {
        return band_query(operator.spec.bounds.before_micros / 1_000_000);
    }
    let left_width = operator.retained_schema(0).fields().len();
    let projection = operator
        .logical_output_schema
        .fields()
        .iter()
        .enumerate()
        .map(|(ordinal, field)| {
            let source = ordinal;
            let (alias, schema) = if source < left_width {
                ("input", operator.retained_schema(0))
            } else {
                ("reference", operator.retained_schema(1))
            };
            let source_field = &schema.fields[source % left_width];
            format!(
                "{}.{} AS {}",
                alias,
                quote(source_field.name()),
                quote(field.name())
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    let before = operator.spec.bounds.before_micros;
    format!(
        "SELECT {projection} FROM input JOIN reference ON input.\"symbol\" = reference.\"symbol\" \
         AND input.\"event_time\" >= reference.\"event_time\" - INTERVAL '{before} microseconds' \
         AND input.\"event_time\" <= reference.\"event_time\""
    )
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

fn quote(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
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
        let mut operator =
            StreamJoinOperator::new("hot", quote_schema(), dimension_schema(), spec()).unwrap();
        operator.enable_run_core();
        let left = RecordBatch::try_new(
            quote_schema(),
            vec![
                Arc::new(TimestampMicrosecondArray::from(vec![0, 1_000_000]).with_timezone("UTC")),
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
        assert_eq!(output.num_rows(), 2);
        let record = &output.table_payload().unwrap().batches()[0];
        let price = record
            .column(record.schema().index_of("quote__price").unwrap())
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(price.values(), &[2.0, 3.0]);
        let factor = record
            .column(record.schema().index_of("dimension__factor").unwrap())
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(factor.values(), &[2.0, 2.0]);
        assert_eq!(operator.state.metrics.emitted_match_rows, 2);
        let core = operator.run_core.as_ref().unwrap();
        assert!(core.hot_enabled());
        assert_eq!(core.rows(JoinSide::Left), 0);
        assert_eq!(core.rows(JoinSide::Right), 0);
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
    async fn hot_band_emits_all_input_columns() {
        let mut operator =
            StreamJoinOperator::new("hot-band", quote_schema(), quote_schema(), band_spec())
                .unwrap();
        operator.enable_run_core();
        let record = RecordBatch::try_new(
            quote_schema(),
            vec![
                Arc::new(TimestampMicrosecondArray::from(vec![0, 1_000_000]).with_timezone("UTC")),
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
