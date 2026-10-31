use std::{fmt, sync::Arc};

use datafusion::{
    arrow::{datatypes::SchemaRef, record_batch::RecordBatch},
    common::{DataFusionError, Result as DataFusionResult, tree_node::TreeNodeRecursion},
    execution::TaskContext,
    physical_expr::{EquivalenceProperties, PhysicalExpr},
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
        SendableRecordBatchStream, Statistics,
        execution_plan::{Boundedness, EmissionType},
        metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet},
        stream::RecordBatchStreamAdapter,
    },
};
use futures::stream;

use crate::{Batch, expression::ValidatedQuery};

#[path = "datafusion_join/execution.rs"]
mod execution;
#[path = "datafusion_join/planning.rs"]
mod planning;

#[derive(Clone, Debug)]
pub(super) struct NativeInput {
    schema: SchemaRef,
    batches: Arc<[RecordBatch]>,
}

impl NativeInput {
    fn new(batch: &Batch) -> Option<Self> {
        let table = batch.table_payload().ok()?;
        Some(Self {
            schema: Arc::clone(table.schema()),
            batches: table.batches().to_vec().into(),
        })
    }
}

#[derive(Clone, Debug)]
pub(super) struct AsofSpec {
    left: NativeInput,
    right: NativeInput,
    left_key: usize,
    right_key: usize,
    left_time: usize,
    right_time: usize,
    right_sequence: usize,
    left_output: usize,
    right_output: usize,
}

#[derive(Clone, Debug)]
pub(super) struct BandSpec {
    left: NativeInput,
    right: NativeInput,
    left_key: usize,
    right_key: usize,
    left_time: usize,
    right_time: usize,
    left_sequence: usize,
    right_sequence: usize,
    left_price: usize,
    right_price: usize,
    offsets: Arc<[i64]>,
}

#[derive(Clone, Debug)]
enum JoinSpec {
    Asof(AsofSpec),
    Band(BandSpec),
}

impl JoinSpec {
    fn name(&self) -> &'static str {
        match self {
            Self::Asof(_) => "CalcFlowAsofJoinExec",
            Self::Band(_) => "CalcFlowBandJoinExec",
        }
    }
}

#[derive(Debug)]
struct CalcFlowJoinExec {
    spec: JoinSpec,
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl CalcFlowJoinExec {
    fn new(spec: JoinSpec, schema: SchemaRef) -> Self {
        let properties = PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&schema)),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        );
        Self {
            spec,
            schema,
            properties: Arc::new(properties),
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }
}

impl DisplayAs for CalcFlowJoinExec {
    fn fmt_as(&self, format: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match format {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "{}: route=native_arrow", self.spec.name())
            }
            DisplayFormatType::TreeRender => write!(f, "route=native_arrow"),
        }
    }
}

impl ExecutionPlan for CalcFlowJoinExec {
    fn name(&self) -> &str {
        self.spec.name()
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        Vec::new()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if children.is_empty() {
            return Ok(self);
        }
        Err(DataFusionError::Internal(format!(
            "{} requires no children",
            self.spec.name()
        )))
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "{} has exactly one output partition",
                self.spec.name()
            )));
        }
        let spec = self.spec.clone();
        let schema = Arc::clone(&self.schema);
        let execution_schema = Arc::clone(&schema);
        let metrics = BaselineMetrics::new(&self.metrics, partition);
        let output = stream::once(async move {
            let timer = metrics.elapsed_compute().timer();
            let batch = execution::execute(&spec, execution_schema)?;
            metrics.record_output(batch.num_rows());
            timer.done();
            Ok(batch)
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, output)))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn statistics_from_inputs(
        &self,
        _input_stats: &[Arc<Statistics>],
        _args: &datafusion::physical_plan::statistics::StatisticsArgs,
    ) -> DataFusionResult<Arc<Statistics>> {
        Ok(Arc::new(Statistics::new_unknown(&self.schema)))
    }
}

pub(super) fn plan(
    query: &ValidatedQuery,
    tables: &std::collections::BTreeMap<String, Batch>,
    schema: SchemaRef,
) -> Option<Arc<dyn ExecutionPlan>> {
    planning::plan(query, tables)
        .map(|spec| Arc::new(CalcFlowJoinExec::new(spec, schema)) as Arc<dyn ExecutionPlan>)
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, sync::Arc};

    use datafusion::{
        arrow::{
            array::{Array as _, ArrayRef, Float64Array, PrimitiveArray, StringArray, UInt64Array},
            compute::concat_batches,
            datatypes::{DataType, Field, Schema, TimeUnit},
            record_batch::RecordBatch,
            util::display::{ArrayFormatter, FormatOptions},
        },
        datasource::MemTable,
        execution::context::{SessionConfig, SessionContext},
    };

    use crate::{Batch, BatchMetadata, DataFusionConfig, DataFusionRuntime};

    const ASOF_SQL: &str = "SELECT input.sequence, reference.price AS value FROM input LEFT JOIN \
        (SELECT symbol, event_time, price, ROW_NUMBER() OVER \
        (PARTITION BY symbol, event_time ORDER BY sequence DESC) AS position \
        FROM reference) AS reference ON input.symbol = reference.symbol \
        AND input.event_time = reference.event_time AND reference.position = 1";
    const INTERVAL_SQL: &str = "WITH offsets(event_offset) AS (VALUES \
        (INTERVAL '-5 seconds'), (INTERVAL '-4 seconds'), \
        (INTERVAL '-3 seconds'), (INTERVAL '-2 seconds'), \
        (INTERVAL '-1 seconds'), (INTERVAL '0 seconds'), \
        (INTERVAL '1 seconds'), (INTERVAL '2 seconds'), \
        (INTERVAL '3 seconds'), (INTERVAL '4 seconds'), \
        (INTERVAL '5 seconds')) \
        SELECT probes.left_sequence AS sequence, \
        reference.sequence AS right_sequence, \
        probes.left_price * reference.price AS value \
        FROM reference JOIN (SELECT input.sequence AS left_sequence, \
        input.symbol AS left_symbol, input.price AS left_price, \
        input.event_time + offsets.event_offset AS match_time \
        FROM input CROSS JOIN offsets) AS probes \
        ON probes.left_symbol = reference.symbol \
        AND probes.match_time = reference.event_time";

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new(
                "event_time",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("sequence", DataType::UInt64, false),
            Field::new("symbol", DataType::Utf8, false),
            Field::new("price", DataType::Float64, true),
        ]))
    }

    fn batch(rows: &[(i64, u64, &str, Option<f64>)]) -> Batch {
        let event_time = Arc::new(
            rows.iter()
                .map(|row| Some(row.0))
                .collect::<PrimitiveArray<datafusion::arrow::array::types::TimestampMicrosecondType>>()
                .with_timezone("UTC"),
        ) as ArrayRef;
        let sequence =
            Arc::new(UInt64Array::from_iter_values(rows.iter().map(|row| row.1))) as ArrayRef;
        let symbol =
            Arc::new(StringArray::from_iter_values(rows.iter().map(|row| row.2))) as ArrayRef;
        let price = Arc::new(rows.iter().map(|row| row.3).collect::<Float64Array>()) as ArrayRef;
        let record =
            RecordBatch::try_new(schema(), vec![event_time, sequence, symbol, price]).unwrap();
        let split = record.num_rows() / 2;
        let records = if split == 0 {
            vec![record]
        } else {
            vec![
                record.slice(0, split),
                record.slice(split, record.num_rows() - split),
            ]
        };
        Batch::table(records, BatchMetadata::default()).unwrap()
    }

    fn combined(batch: &Batch) -> RecordBatch {
        let table = batch.table_payload().unwrap();
        concat_batches(table.schema(), table.batches()).unwrap()
    }

    async fn reference_batches(
        query: &str,
        left: &Batch,
        right: &Batch,
    ) -> (Arc<Schema>, Vec<RecordBatch>) {
        let context =
            SessionContext::new_with_config(SessionConfig::new().with_target_partitions(1));
        for (name, batch) in [("input", left), ("reference", right)] {
            let table = batch.table_payload().unwrap();
            context
                .register_table(
                    name,
                    Arc::new(
                        MemTable::try_new(
                            Arc::clone(table.schema()),
                            vec![table.batches().to_vec()],
                        )
                        .unwrap(),
                    ),
                )
                .unwrap();
        }
        let dataframe = context.sql(query).await.unwrap();
        let schema = Arc::new(dataframe.schema().as_arrow().clone());
        let batches = dataframe.collect().await.unwrap();
        (schema, batches)
    }

    async fn reference(query: &str, left: &Batch, right: &Batch) -> RecordBatch {
        let (schema, batches) = reference_batches(query, left, right).await;
        if batches.iter().any(|batch| batch.num_rows() > 0) {
            concat_batches(&schema, batches.iter().filter(|batch| batch.num_rows() > 0)).unwrap()
        } else {
            RecordBatch::new_empty(schema)
        }
    }

    async fn execute(query: &str, left: &Batch, right: &Batch) -> (DataFusionRuntime, Batch) {
        let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
        let output = runtime
            .sql(
                query,
                &BTreeMap::from([
                    ("input".into(), left.clone()),
                    ("reference".into(), right.clone()),
                ]),
                None,
            )
            .await
            .unwrap();
        (runtime, output)
    }

    fn asof_rows(batch: &RecordBatch) -> Vec<(u64, Option<f64>)> {
        let sequences = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let values = batch
            .column(1)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        let mut rows = (0..batch.num_rows())
            .map(|index| {
                (
                    sequences.value(index),
                    (!values.is_null(index)).then(|| values.value(index)),
                )
            })
            .collect::<Vec<_>>();
        rows.sort_unstable_by(|left, right| left.0.cmp(&right.0));
        rows
    }

    fn interval_rows(batch: &RecordBatch) -> Vec<(u64, u64, f64)> {
        let mut rows = (0..batch.num_rows())
            .map(|index| {
                (
                    batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<UInt64Array>()
                        .unwrap()
                        .value(index),
                    batch
                        .column(1)
                        .as_any()
                        .downcast_ref::<UInt64Array>()
                        .unwrap()
                        .value(index),
                    batch
                        .column(2)
                        .as_any()
                        .downcast_ref::<Float64Array>()
                        .unwrap()
                        .value(index),
                )
            })
            .collect::<Vec<_>>();
        rows.sort_unstable_by(|left, right| {
            (left.0, left.1, left.2.to_bits()).cmp(&(right.0, right.1, right.2.to_bits()))
        });
        rows
    }

    #[tokio::test]
    async fn native_asof_selects_latest_sequence_and_preserves_unmatched_left_rows() {
        let right = batch(&[
            (0, 0, "S0", Some(10.0)),
            (0, 2, "S0", Some(20.0)),
            (0, 1, "S0", Some(11.0)),
            (1_000_000, 3, "S0", Some(30.0)),
            (2_000_000, 4, "S1", Some(40.0)),
        ]);
        let left = batch(&[
            (0, 100, "S0", Some(999.0)),
            (1_000_000, 101, "S0", Some(999.0)),
            (2_000_000, 102, "S0", Some(999.0)),
            (3_000_000, 103, "S1", Some(999.0)),
            (2_000_000, 104, "S1", Some(999.0)),
            (0, 105, "S2", Some(999.0)),
        ]);
        let (runtime, output) = execute(ASOF_SQL, &left, &right).await;
        let actual = combined(&output);
        assert_eq!(
            asof_rows(&actual),
            vec![
                (100, Some(20.0)),
                (101, Some(30.0)),
                (102, None),
                (103, None),
                (104, Some(40.0)),
                (105, None),
            ]
        );
        assert_eq!(
            asof_rows(&actual),
            asof_rows(&reference(ASOF_SQL, &left, &right).await)
        );
        let plan = &runtime.metrics()[0].physical_plan;
        assert!(
            plan.contains("CalcFlowAsofJoinExec"),
            "ASOF did not use the native top-1 path: {plan}"
        );
    }

    #[tokio::test]
    async fn native_interval_band_join_matches_plain_datafusion() {
        let right = batch(&[
            (0, 0, "S0", Some(2.0)),
            (1_000_000, 1, "S0", Some(3.0)),
            (10_000_000, 2, "S0", Some(5.0)),
            (0, 3, "S1", Some(7.0)),
        ]);
        let left = batch(&[
            (0, 100, "S0", Some(11.0)),
            (4_000_000, 101, "S0", Some(13.0)),
            (10_000_000, 102, "S0", Some(17.0)),
            (1_000_000, 103, "S1", Some(19.0)),
        ]);
        let (runtime, output) = execute(INTERVAL_SQL, &left, &right).await;
        let actual = combined(&output);
        assert!(actual.num_rows() > 0);
        assert_eq!(
            interval_rows(&actual),
            interval_rows(&reference(INTERVAL_SQL, &left, &right).await)
        );
        let plan = &runtime.metrics()[0].physical_plan;
        assert!(
            plan.contains("CalcFlowBandJoinExec"),
            "interval did not use the native band path: {plan}"
        );
    }

    fn nullable_batch(rows: &[(i64, Option<u64>, &str, Option<f64>)]) -> Batch {
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "event_time",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("sequence", DataType::UInt64, true),
            Field::new("symbol", DataType::Utf8, false),
            Field::new("price", DataType::Float64, true),
        ]));
        let event_time = Arc::new(
            rows.iter()
                .map(|row| Some(row.0))
                .collect::<PrimitiveArray<datafusion::arrow::array::types::TimestampMicrosecondType>>()
                .with_timezone("UTC"),
        ) as ArrayRef;
        let sequence = Arc::new(rows.iter().map(|row| row.1).collect::<UInt64Array>()) as ArrayRef;
        let symbol =
            Arc::new(StringArray::from_iter_values(rows.iter().map(|row| row.2))) as ArrayRef;
        let price = Arc::new(rows.iter().map(|row| row.3).collect::<Float64Array>()) as ArrayRef;
        let record =
            RecordBatch::try_new(schema, vec![event_time, sequence, symbol, price]).unwrap();
        Batch::table(vec![record], BatchMetadata::default()).unwrap()
    }

    fn nullable_interval_rows(batch: &RecordBatch) -> Vec<(Option<u64>, Option<u64>, Option<f64>)> {
        let left = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let right = batch
            .column(1)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let values = batch
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        let mut rows = (0..batch.num_rows())
            .map(|index| {
                (
                    (!left.is_null(index)).then(|| left.value(index)),
                    (!right.is_null(index)).then(|| right.value(index)),
                    (!values.is_null(index)).then(|| values.value(index)),
                )
            })
            .collect::<Vec<_>>();
        rows.sort_unstable_by(|left, right| {
            (left.0, left.1, left.2.map(f64::to_bits)).cmp(&(
                right.0,
                right.1,
                right.2.map(f64::to_bits),
            ))
        });
        rows
    }

    #[tokio::test]
    async fn native_asof_null_sequence_uses_nulls_first() {
        let right = nullable_batch(&[(0, None, "S0", Some(99.0)), (0, Some(1), "S0", Some(10.0))]);
        let left = nullable_batch(&[(0, Some(100), "S0", Some(999.0))]);
        let (runtime, output) = execute(ASOF_SQL, &left, &right).await;
        let actual = combined(&output);
        assert_eq!(asof_rows(&actual), vec![(100, Some(99.0))]);
        assert_eq!(
            asof_rows(&actual),
            asof_rows(&reference(ASOF_SQL, &left, &right).await)
        );
        assert!(
            runtime.metrics()[0]
                .physical_plan
                .contains("CalcFlowAsofJoinExec")
        );
    }

    #[tokio::test]
    async fn native_interval_null_sequences_do_not_filter_matches() {
        let right = nullable_batch(&[(0, None, "S0", Some(2.0))]);
        let left = nullable_batch(&[(0, None, "S0", Some(11.0))]);
        let (runtime, output) = execute(INTERVAL_SQL, &left, &right).await;
        let actual = combined(&output);
        assert_eq!(
            nullable_interval_rows(&actual),
            vec![(None, None, Some(22.0))]
        );
        assert_eq!(
            nullable_interval_rows(&actual),
            nullable_interval_rows(&reference(INTERVAL_SQL, &left, &right).await)
        );
        assert!(
            runtime.metrics()[0]
                .physical_plan
                .contains("CalcFlowBandJoinExec")
        );
    }

    #[tokio::test]
    async fn native_asof_computed_right_payload_uses_datafusion_fallback() {
        let right = batch(&[(0, 1, "S0", Some(10.0))]);
        let left = batch(&[(0, 100, "S0", Some(999.0))]);
        let query = ASOF_SQL.replace(
            "event_time, price, ROW_NUMBER()",
            "event_time, price * 2 AS price, ROW_NUMBER()",
        );
        let (_runtime, output) = execute(&query, &left, &right).await;
        assert_eq!(
            asof_rows(&combined(&output)),
            asof_rows(&reference(&query, &left, &right).await)
        );
    }

    #[tokio::test]
    async fn native_interval_outer_projection_uses_datafusion_fallback() {
        let right = batch(&[(0, 1, "S0", Some(2.0))]);
        let left = batch(&[(0, 100, "S0", Some(11.0))]);
        let query = INTERVAL_SQL.replace(
            "SELECT probes.left_sequence AS sequence",
            "SELECT probes.left_sequence + 1 AS sequence",
        );
        let (_runtime, output) = execute(&query, &left, &right).await;
        let formatted = |batches: &[RecordBatch]| {
            let options = FormatOptions::default();
            batches
                .iter()
                .flat_map(|batch| {
                    let formatters = batch
                        .columns()
                        .iter()
                        .map(|column| ArrayFormatter::try_new(column.as_ref(), &options).unwrap())
                        .collect::<Vec<_>>();
                    (0..batch.num_rows())
                        .map(|row| {
                            formatters
                                .iter()
                                .map(|formatter| formatter.value(row).to_string())
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };
        let table = output.table_payload().unwrap();
        assert_eq!(
            formatted(table.batches()),
            formatted(&reference_batches(&query, &left, &right).await.1)
        );
    }

    #[tokio::test]
    async fn native_join_empty_relations_match_plain_datafusion() {
        for (left, right) in [
            (batch(&[]), batch(&[(0, 1, "S0", Some(2.0))])),
            (batch(&[(0, 100, "S0", Some(11.0))]), batch(&[])),
        ] {
            for query in [ASOF_SQL, INTERVAL_SQL] {
                let (runtime, output) = execute(query, &left, &right).await;
                let actual = combined(&output);
                let expected = reference(query, &left, &right).await;
                assert_eq!(actual.num_rows(), expected.num_rows());
                if query == ASOF_SQL {
                    assert_eq!(asof_rows(&actual), asof_rows(&expected));
                    assert!(
                        runtime.metrics()[0]
                            .physical_plan
                            .contains("CalcFlowAsofJoinExec")
                    );
                } else {
                    assert_eq!(interval_rows(&actual), interval_rows(&expected));
                    assert!(
                        runtime.metrics()[0]
                            .physical_plan
                            .contains("CalcFlowBandJoinExec")
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn native_interval_supports_second_timestamps_and_duplicate_matches() {
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "event_time",
                DataType::Timestamp(TimeUnit::Second, Some("UTC".into())),
                false,
            ),
            Field::new("sequence", DataType::UInt64, false),
            Field::new("symbol", DataType::Utf8, false),
            Field::new("price", DataType::Float64, true),
        ]));
        let make = |rows: &[(i64, u64, &str, Option<f64>)]| {
            let event_time = Arc::new(
                rows.iter()
                    .map(|row| Some(row.0))
                    .collect::<PrimitiveArray<datafusion::arrow::array::types::TimestampSecondType>>()
                    .with_timezone("UTC"),
            ) as ArrayRef;
            let sequence =
                Arc::new(UInt64Array::from_iter_values(rows.iter().map(|row| row.1))) as ArrayRef;
            let symbol =
                Arc::new(StringArray::from_iter_values(rows.iter().map(|row| row.2))) as ArrayRef;
            let price =
                Arc::new(rows.iter().map(|row| row.3).collect::<Float64Array>()) as ArrayRef;
            let record = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![event_time, sequence, symbol, price],
            )
            .unwrap();
            Batch::table(vec![record], BatchMetadata::default()).unwrap()
        };
        let right = make(&[(0, 1, "S0", Some(2.0)), (0, 2, "S0", Some(3.0))]);
        let left = make(&[(0, 100, "S0", Some(11.0))]);
        let (runtime, output) = execute(INTERVAL_SQL, &left, &right).await;
        let actual = combined(&output);
        assert_eq!(actual.num_rows(), 2);
        assert_eq!(
            interval_rows(&actual),
            interval_rows(&reference(INTERVAL_SQL, &left, &right).await)
        );
        assert!(
            runtime.metrics()[0]
                .physical_plan
                .contains("CalcFlowBandJoinExec")
        );
    }
}
