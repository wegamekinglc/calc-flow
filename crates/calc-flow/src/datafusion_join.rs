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
    },
};

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
        context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "{} has exactly one output partition",
                self.spec.name()
            )));
        }
        let metrics = BaselineMetrics::new(&self.metrics, partition);
        Ok(execution::execute(
            &self.spec,
            Arc::clone(&self.schema),
            context.session_config().batch_size(),
            metrics,
        ))
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

    #[tokio::test]
    async fn native_interval_out_of_order_probes_reset_exhausted_cursors() {
        let query = "WITH offsets(event_offset) AS (VALUES (INTERVAL '0 seconds')) \
            SELECT probes.left_sequence AS sequence, reference.sequence AS right_sequence, \
            probes.left_price * reference.price AS value FROM reference JOIN \
            (SELECT input.sequence AS left_sequence, input.symbol AS left_symbol, \
            input.price AS left_price, input.event_time + offsets.event_offset AS match_time \
            FROM input CROSS JOIN offsets) AS probes \
            ON probes.left_symbol = reference.symbol \
            AND probes.match_time = reference.event_time";
        let right = batch(&[(0, 1, "S0", Some(2.0)), (10_000_000, 2, "S0", Some(3.0))]);
        let left = batch(&[
            (20_000_000, 100, "S0", Some(11.0)),
            (10_000_000, 101, "S0", Some(13.0)),
        ]);
        let (runtime, output) = execute(query, &left, &right).await;
        let actual = combined(&output);
        assert_eq!(
            interval_rows(&actual),
            interval_rows(&reference(query, &left, &right).await)
        );
        assert!(
            runtime.metrics()[0]
                .physical_plan
                .contains("CalcFlowBandJoinExec")
        );
    }

    #[tokio::test]
    async fn native_asof_explicit_nulls_last_uses_datafusion_fallback() {
        let right = nullable_batch(&[(0, None, "S0", Some(99.0)), (0, Some(1), "S0", Some(10.0))]);
        let left = nullable_batch(&[(0, Some(100), "S0", Some(999.0))]);
        let query = ASOF_SQL.replace("sequence DESC)", "sequence DESC NULLS LAST)");
        let (runtime, output) = execute(&query, &left, &right).await;
        assert_eq!(
            asof_rows(&combined(&output)),
            asof_rows(&reference(&query, &left, &right).await)
        );
        assert!(
            !runtime.metrics()[0]
                .physical_plan
                .contains("CalcFlowAsofJoinExec")
        );
    }

    #[tokio::test]
    async fn native_asof_additional_right_relation_uses_datafusion_fallback() {
        let right = batch(&[(0, 1, "S0", Some(10.0))]);
        let left = batch(&[(0, 100, "S0", Some(999.0))]);
        let query = ASOF_SQL.replace(
            "FROM reference) AS reference",
            "FROM reference, (SELECT 1 AS extra WHERE FALSE) AS other) AS reference",
        );
        let (runtime, output) = execute(&query, &left, &right).await;
        assert_eq!(
            asof_rows(&combined(&output)),
            asof_rows(&reference(&query, &left, &right).await)
        );
        assert!(
            !runtime.metrics()[0]
                .physical_plan
                .contains("CalcFlowAsofJoinExec")
        );
    }

    #[tokio::test]
    async fn native_interval_limited_offsets_use_datafusion_fallback() {
        let right = batch(&[(0, 1, "S0", Some(2.0))]);
        let left = batch(&[(0, 100, "S0", Some(11.0))]);
        let query =
            INTERVAL_SQL.replace("(INTERVAL '5 seconds'))", "(INTERVAL '5 seconds') LIMIT 1)");
        let (runtime, output) = execute(&query, &left, &right).await;
        assert_eq!(
            interval_rows(&combined(&output)),
            interval_rows(&reference(&query, &left, &right).await)
        );
        assert!(
            !runtime.metrics()[0]
                .physical_plan
                .contains("CalcFlowBandJoinExec")
        );
    }

    fn shifted_interval_sql() -> &'static str {
        "WITH offsets(event_offset) AS (VALUES (INTERVAL '-5 seconds')) \
        SELECT probes.left_sequence AS sequence, reference.sequence AS right_sequence, \
        probes.left_price * reference.price AS value FROM reference JOIN \
        (SELECT input.sequence AS left_sequence, input.symbol AS left_symbol, \
        input.price AS left_price, input.event_time + offsets.event_offset AS match_time \
        FROM input CROSS JOIN offsets) AS probes \
        ON probes.left_symbol = reference.symbol \
        AND probes.match_time = reference.event_time"
    }

    fn second_timestamp_batch(rows: &[(i64, Option<&str>)], timezone: &str) -> Batch {
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "event_time",
                DataType::Timestamp(TimeUnit::Second, Some(timezone.into())),
                false,
            ),
            Field::new("sequence", DataType::UInt64, false),
            Field::new("symbol", DataType::Utf8, true),
            Field::new("price", DataType::Float64, true),
        ]));
        let event_time = Arc::new(
            rows.iter()
                .map(|row| Some(row.0))
                .collect::<PrimitiveArray<datafusion::arrow::array::types::TimestampSecondType>>()
                .with_timezone(timezone),
        ) as ArrayRef;
        let sequence = Arc::new(UInt64Array::from_iter_values(
            rows.iter().enumerate().map(|(index, _)| index as u64),
        )) as ArrayRef;
        let symbol = Arc::new(rows.iter().map(|row| row.1).collect::<StringArray>()) as ArrayRef;
        let price = Arc::new(Float64Array::from(vec![Some(1.0); rows.len()])) as ArrayRef;
        Batch::table(
            vec![RecordBatch::try_new(schema, vec![event_time, sequence, symbol, price]).unwrap()],
            BatchMetadata::default(),
        )
        .unwrap()
    }

    async fn interval_error(query: &str, left: &Batch, right: &Batch) -> String {
        let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
        runtime
            .sql(
                query,
                &BTreeMap::from([
                    ("input".into(), left.clone()),
                    ("reference".into(), right.clone()),
                ]),
                None,
            )
            .await
            .unwrap_err()
            .to_string()
    }

    #[tokio::test]
    async fn native_interval_rejects_unrepresentable_timestamp_arithmetic() {
        let error = interval_error(
            INTERVAL_SQL,
            &second_timestamp_batch(&[(i64::MAX - 1_000_000, Some("S0"))], "UTC"),
            &second_timestamp_batch(&[(0, Some("S0"))], "UTC"),
        )
        .await;
        assert!(error.contains("Timestamp out of range"), "{error}");
    }

    #[tokio::test]
    async fn native_interval_rejects_out_of_range_source_before_shift() {
        let source = crate::datafusion_interval::representable_bounds(TimeUnit::Second).1 + 5;
        let error = interval_error(
            shifted_interval_sql(),
            &second_timestamp_batch(&[(source, Some("S0"))], "UTC"),
            &second_timestamp_batch(&[(0, Some("S0"))], "UTC"),
        )
        .await;
        assert!(error.contains("Timestamp out of range"), "{error}");
    }

    #[tokio::test]
    async fn native_interval_checks_timestamp_before_null_key_join() {
        let source = crate::datafusion_interval::representable_bounds(TimeUnit::Second).1 + 5;
        let error = interval_error(
            shifted_interval_sql(),
            &second_timestamp_batch(&[(source, None)], "UTC"),
            &second_timestamp_batch(&[(0, Some("S0"))], "UTC"),
        )
        .await;
        assert!(error.contains("Timestamp out of range"), "{error}");
    }

    #[tokio::test]
    async fn native_interval_rejects_invalid_timestamp_timezone() {
        let error = interval_error(
            INTERVAL_SQL,
            &second_timestamp_batch(&[(0, Some("S0"))], "Invalid/Zone"),
            &second_timestamp_batch(&[(0, Some("S0"))], "Invalid/Zone"),
        )
        .await;
        assert!(
            error.contains("Cannot coerce arithmetic expression")
                || error.to_lowercase().contains("timezone"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn native_asof_stops_materializing_at_the_output_limit() {
        let left = batch(
            &(0..8)
                .map(|index| (0, 100 + index, "S0", Some(11.0)))
                .collect::<Vec<_>>(),
        );
        let right = batch(&[(0, 1, "S0", Some(2.0))]);
        let runtime = DataFusionRuntime::new(DataFusionConfig {
            batch_size: 2,
            ..DataFusionConfig::default()
        })
        .unwrap();
        super::execution::take_asof_output_rows();
        let query = super::super::parse_select_query(ASOF_SQL).unwrap();
        runtime
            .snapshot_sql(
                &query,
                &BTreeMap::from([("input".into(), left), ("reference".into(), right)]),
                "limited",
                crate::EdgeBudget::new(3, 1_000_000).unwrap(),
            )
            .await
            .unwrap_err();
        assert_eq!(super::execution::take_asof_output_rows(), 4);
    }

    #[tokio::test]
    async fn native_asof_elapsed_compute_includes_initialization() {
        let rows = 100_000_i64;
        let schema = schema();
        let record =
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(
                        PrimitiveArray::<datafusion::arrow::array::types::TimestampMicrosecondType>::from_iter_values(0..rows)
                            .with_timezone("UTC"),
                    ),
                    Arc::new(UInt64Array::from_iter_values(
                        (0..rows).map(|row| u64::try_from(row).unwrap()),
                    )),
                    Arc::new(StringArray::from_iter_values(std::iter::repeat_n(
                        "S0",
                        usize::try_from(rows).unwrap(),
                    ))),
                    Arc::new(Float64Array::from_iter_values(
                        (0..rows).map(|row| f64::from(u32::try_from(row).unwrap())),
                    )),
                ],
            )
            .unwrap();
        let data = Batch::table(vec![record], BatchMetadata::default()).unwrap();
        let (runtime, _) = execute(ASOF_SQL, &data, &data).await;
        let metrics = runtime.metrics();
        let metric = &metrics[0];
        assert!(metric.physical_plan.contains("CalcFlowAsofJoinExec"));
        assert!(
            u64::try_from(metric.elapsed_compute_ns).unwrap() * 4
                >= metric.execution_to_first_batch_ns,
            "initialization was excluded from elapsed_compute: {} vs {}",
            metric.elapsed_compute_ns,
            metric.execution_to_first_batch_ns
        );
    }

    #[tokio::test]
    async fn native_join_chunks_output_at_configured_batch_size() {
        let right = batch(&[
            (0, 1, "S0", Some(2.0)),
            (0, 2, "S0", Some(3.0)),
            (0, 3, "S0", Some(5.0)),
        ]);
        let left = batch(&[
            (0, 100, "S0", Some(11.0)),
            (0, 101, "S0", Some(13.0)),
            (0, 102, "S0", Some(17.0)),
            (0, 103, "S0", Some(19.0)),
        ]);
        for query in [ASOF_SQL, INTERVAL_SQL] {
            let runtime = DataFusionRuntime::new(DataFusionConfig {
                batch_size: 2,
                ..DataFusionConfig::default()
            })
            .unwrap();
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
            let batches = output.table_payload().unwrap().batches();
            assert!(batches.len() > 1, "{query} was not chunked");
            assert!(batches.iter().all(|batch| batch.num_rows() <= 2));
            let actual = combined(&output);
            let expected = reference(query, &left, &right).await;
            if query == ASOF_SQL {
                assert_eq!(asof_rows(&actual), asof_rows(&expected));
            } else {
                assert_eq!(interval_rows(&actual), interval_rows(&expected));
            }
        }
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
