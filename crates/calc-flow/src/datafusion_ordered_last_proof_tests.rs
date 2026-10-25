//! T1 conformance proof; production compact execution is connected in T2.
//! Contract and source rationale: `.codex/artifacts/analysis/dal-431-ordered-last.md`.
use super::*;
use datafusion::{
    arrow::{
        array::{
            ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray,
            TimestampMicrosecondArray, new_null_array,
        },
        compute::cast,
        datatypes::{DataType, Field, Schema, TimeUnit},
    },
    common::{DFSchema, ScalarValue},
    execution::context::SessionConfig,
    logical_expr::{Aggregate, EmitTo, Expr, ExprSchemable, LogicalPlan, TableScan},
    physical_expr::{PhysicalExpr, aggregate::AggregateFunctionExpr, expressions::Column},
    physical_plan::{
        ExecutionPlan, InputOrderMode,
        aggregates::{AggregateExec, AggregateMode},
        projection::ProjectionExec,
        sorts::sort::SortExec,
    },
};

const QUERY: &str = "SELECT stock, LAST_VALUE(latest_price ORDER BY \"timestamp\") AS latest_price FROM input GROUP BY stock";

fn input() -> Batch {
    batch(vec![
        quotes(&[
            (Some("A"), None, Some(10.0), Some(-1), None),
            (Some("A"), None, Some(20.0), Some(-1), None),
        ])
        .project(&[0, 2, 3])
        .unwrap(),
    ])
}

#[tokio::test]
async fn test_ordered_last_direct_oracle_and_current_planning_paths() {
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let batch = input();
    let result = runtime
        .sql(
            QUERY,
            &BTreeMap::from([("input".into(), batch.clone())]),
            Some("oracle"),
        )
        .await
        .unwrap();
    assert_eq!(result.num_rows(), 1);
    assert_eq!(
        ScalarValue::try_from_array(result.table_payload().unwrap().batches()[0].column(1), 0)
            .unwrap(),
        ScalarValue::Float64(Some(10.0))
    );
    let query = parse_select_query(QUERY).unwrap();
    let schema = batch.table_payload().unwrap().schema().clone();
    let sync = runtime.incremental_sql_plan_sync(
        &query,
        "input",
        schema.clone(),
        "native",
        runtime.incremental_reservation("native"),
    );
    let sync = sync.unwrap();
    assert!(logical_shape(&sync.raw, "input"));
    let retained = runtime.retained_sql_plan_sync(
        &query,
        "input",
        schema,
        "retained",
        runtime.incremental_reservation("retained"),
    );
    let retained = retained.unwrap();
    assert!(logical_shape(&retained.raw, "input"));
    assert_eq!(sync.analyzed, retained.analyzed);
}

#[derive(Debug, Eq, PartialEq)]
enum Admission {
    Deferred,
    Retained(&'static str),
    Proven,
}

fn unalias(mut expr: &Expr) -> &Expr {
    while let Expr::Alias(alias) = expr {
        expr = &alias.expr;
    }
    expr
}

fn column_type(expr: &Expr, schema: &DFSchema, allowed: impl FnOnce(&DataType) -> bool) -> bool {
    matches!(unalias(expr), Expr::Column(_)) && expr.get_type(schema).is_ok_and(|t| allowed(&t))
}

fn key_type(dtype: &DataType) -> bool {
    dtype.is_integer()
        || matches!(
            dtype,
            DataType::Boolean | DataType::Utf8 | DataType::LargeUtf8
        )
}

fn value_type(dtype: &DataType) -> bool {
    dtype.is_integer() || matches!(dtype, DataType::Float32 | DataType::Float64)
}

fn ordering_type(dtype: &DataType) -> bool {
    dtype.is_integer() || matches!(dtype, DataType::Timestamp(..))
}

// Inspect the raw shape, before optimizers can erase a forbidden filter/limit.
fn logical_shape(plan: &LogicalPlan, alias: &str) -> bool {
    let plan = match plan {
        LogicalPlan::Sort(sort) if sort.fetch.is_none() => sort.input.as_ref(),
        _ => plan,
    };
    let LogicalPlan::Projection(projection) = plan else {
        return false;
    };
    let LogicalPlan::Aggregate(aggregate) = projection.input.as_ref() else {
        return false;
    };
    projection
        .expr
        .iter()
        .all(|e| matches!(unalias(e), Expr::Column(_)))
        && logical_aggregate_shape(aggregate, alias)
}

fn logical_aggregate_shape(aggregate: &Aggregate, alias: &str) -> bool {
    let LogicalPlan::TableScan(scan) = aggregate.input.as_ref() else {
        return false;
    };
    let schema = aggregate.input.schema();
    !aggregate.group_expr.is_empty()
        && aggregate
            .group_expr
            .iter()
            .all(|e| column_type(e, schema, key_type))
        && logical_scan_shape(scan, alias)
        && logical_last_value(&aggregate.aggr_expr, schema)
}

fn logical_scan_shape(scan: &TableScan, alias: &str) -> bool {
    scan.table_name.table() == alias && scan.filters.is_empty() && scan.fetch.is_none()
}

fn logical_last_value(expressions: &[Expr], schema: &DFSchema) -> bool {
    let [expr] = expressions else {
        return false;
    };
    let Expr::AggregateFunction(function) = unalias(expr) else {
        return false;
    };
    let params = &function.params;
    function.func == datafusion::functions_aggregate::first_last::last_value_udaf()
        && !params.distinct
        && params.filter.is_none()
        && matches!(params.args.as_slice(), [value] if column_type(value, schema, value_type))
        && logical_ordering(&params.order_by, schema)
}

fn logical_ordering(order_by: &[datafusion::logical_expr::SortExpr], schema: &DFSchema) -> bool {
    !order_by.is_empty()
        && order_by
            .iter()
            .all(|sort| column_type(&sort.expr, schema, ordering_type))
}

fn physical_shape(
    plan: &dyn ExecutionPlan,
    input: &Batch,
    below_aggregate: bool,
    census: &mut [usize; 2],
) -> bool {
    if plan.output_partitioning().partition_count() != 1 {
        return false;
    }
    physical_node_shape(plan, input, below_aggregate, census)
        && plan.children().iter().all(|child| {
            physical_shape(
                child.as_ref(),
                input,
                below_aggregate || plan.is::<AggregateExec>(),
                census,
            )
        })
}

fn physical_node_shape(
    plan: &dyn ExecutionPlan,
    input: &Batch,
    below_aggregate: bool,
    census: &mut [usize; 2],
) -> bool {
    if let Some(aggregate) = plan.downcast_ref::<AggregateExec>() {
        census[0] += 1;
        physical_aggregate_shape(aggregate)
    } else if let Some(source) =
        plan.downcast_ref::<datafusion::datasource::memory::DataSourceExec>()
    {
        census[1] += 1;
        grouped_float::fifo_source(source, input).unwrap()
    } else if let Some(projection) = plan.downcast_ref::<ProjectionExec>() {
        projection.expr().iter().all(|e| e.expr.is::<Column>())
    } else {
        !below_aggregate
            && plan
                .downcast_ref::<SortExec>()
                .is_some_and(|sort| sort.fetch().is_none())
    }
}

fn physical_aggregate_shape(aggregate: &AggregateExec) -> bool {
    *aggregate.mode() == AggregateMode::Single
        && *aggregate.input_order_mode() == InputOrderMode::Linear
        && aggregate.limit_options().is_none()
        && physical_group_shape(aggregate)
        && physical_last_value(aggregate.aggr_expr())
}

fn physical_group_shape(aggregate: &AggregateExec) -> bool {
    aggregate.group_expr().groups().len() == 1
        && !aggregate.group_expr().expr().is_empty()
        && aggregate
            .group_expr()
            .expr()
            .iter()
            .all(|(e, _)| e.is::<Column>())
        && aggregate.filter_expr().iter().all(Option::is_none)
}

fn physical_last_value(expressions: &[Arc<AggregateFunctionExpr>]) -> bool {
    let [expr] = expressions else {
        return false;
    };
    *expr.fun() == *datafusion::functions_aggregate::first_last::last_value_udaf()
        && expr.groups_accumulator_supported()
        && !expr.order_bys().is_empty()
        && expr.expressions().iter().all(|arg| arg.is::<Column>())
        && expr.order_bys().iter().all(|sort| sort.expr.is::<Column>())
}

#[tokio::test]
async fn test_ordered_last_compact_example_and_retained_fallback_paths() {
    use crate::{
        CancellationToken, EdgeCollector, OperatorMetadata, SqlOperator, StreamJobContext,
        StreamOperator, StreamOperatorContext,
    };
    let job = StreamJobContext::new(
        431,
        "ordered-proof",
        crate::JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "ordered-proof", None);
    for (query, config) in [
        (QUERY, DataFusionConfig::default()),
        (
            "SELECT stock, LAST_VALUE(latest_price ORDER BY CAST(timestamp AS BIGINT)) AS price FROM input GROUP BY stock",
            DataFusionConfig::default(),
        ),
        (
            QUERY,
            DataFusionConfig {
                target_partitions: 2,
                ..DataFusionConfig::default()
            },
        ),
    ] {
        let mut operator =
            SqlOperator::new("ordered-proof", query, vec!["input".into()], vec![]).unwrap();
        let compact = query == QUERY && config == DataFusionConfig::default();
        operator.set_stream_resources(config, UdfRegistrySnapshot::default(), vec![]);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        for rows in [2, 4] {
            operator
                .process_data("input", input(), &context, &mut collector)
                .await
                .unwrap();
            let outputs = collector.drain("output");
            assert_eq!(outputs.len(), 1);
            let actual = outputs[0].as_data().unwrap();
            assert_eq!(
                ScalarValue::try_from_array(
                    actual.table_payload().unwrap().batches()[0].column(1),
                    0
                )
                .unwrap(),
                ScalarValue::Float64(Some(10.0))
            );
            operator.prepare_checkpoint_async(&context).await.unwrap();
            let captured = operator.checkpoint(crate::Epoch::INITIAL).unwrap();
            assert_eq!(
                captured.inline_metadata["state_layout"],
                serde_json::json!(if compact { 5 } else { 4 })
            );
            assert_eq!(captured.inline_metadata["rows"], serde_json::json!(rows));
            assert_eq!(captured.segments.contains_key("input-retained"), !compact);
        }
    }
}

async fn admission(runtime: &DataFusionRuntime, sql: &str, input: &Batch) -> Admission {
    if input.num_rows() == 0 {
        return Admission::Deferred;
    }
    let expected = SessionConfig::new().with_target_partitions(1);
    let context = runtime.context_for_rows(input.num_rows(), None, "not_evaluated");
    let state = context.state();
    if !runtime
        .grouped_float_model_supported("ordered-proof")
        .unwrap()
        || std::any::Any::type_id(runtime.runtime_env.memory_pool.as_ref())
            != std::any::TypeId::of::<UnboundedMemoryPool>()
        || !state
            .config_options()
            .entries()
            .iter()
            .map(|e| (&e.key, &e.value))
            .eq(expected
                .options()
                .entries()
                .iter()
                .map(|e| (&e.key, &e.value)))
    {
        return Admission::Retained("execution model");
    }
    let query = parse_select_query(sql).unwrap();
    let (raw, _) = runtime
        .incremental_sql_plan(
            &query,
            "input",
            input.table_payload().unwrap().schema().clone(),
            "ordered-proof",
        )
        .await
        .unwrap();
    if !logical_shape(&raw, "input") {
        return Admission::Retained("query shape or types");
    }
    let _guard = runtime.query_lock.lock().await;
    let plan = runtime
        .prepare_query(
            context,
            &query,
            &BTreeMap::from([("input".into(), input.clone())]),
            Some("ordered-proof"),
        )
        .await
        .unwrap();
    let mut census = [0, 0];
    if physical_shape(plan.physical_plan.as_ref(), input, false, &mut census) && census == [1, 1] {
        Admission::Proven
    } else {
        Admission::Retained("physical plan")
    }
}

fn aggregate(plan: &dyn ExecutionPlan) -> &AggregateExec {
    if let Some(aggregate) = plan.downcast_ref::<AggregateExec>() {
        return aggregate;
    }
    let children = plan.children();
    assert_eq!(children.len(), 1);
    aggregate(children[0].as_ref())
}

type Row = (
    Option<&'static str>,
    Option<bool>,
    Option<f64>,
    Option<i64>,
    Option<i64>,
);

fn quotes(rows: &[Row]) -> RecordBatch {
    let columns = vec![
        (
            "stock",
            Arc::new(StringArray::from(
                rows.iter().map(|r| r.0).collect::<Vec<_>>(),
            )) as ArrayRef,
        ),
        (
            "desk",
            Arc::new(BooleanArray::from(
                rows.iter().map(|r| r.1).collect::<Vec<_>>(),
            )) as ArrayRef,
        ),
        (
            "latest_price",
            Arc::new(Float64Array::from(
                rows.iter().map(|r| r.2).collect::<Vec<_>>(),
            )) as ArrayRef,
        ),
        (
            "timestamp",
            Arc::new(
                TimestampMicrosecondArray::from(rows.iter().map(|r| r.3).collect::<Vec<_>>())
                    .with_timezone("UTC"),
            ) as ArrayRef,
        ),
        (
            "seq",
            Arc::new(Int64Array::from(
                rows.iter().map(|r| r.4).collect::<Vec<_>>(),
            )) as ArrayRef,
        ),
    ];
    let schema = Schema::new(
        columns
            .iter()
            .map(|(name, array)| Field::new(*name, array.data_type().clone(), true))
            .collect::<Vec<_>>(),
    );
    RecordBatch::try_new(
        Arc::new(schema),
        columns.into_iter().map(|(_, array)| array).collect(),
    )
    .unwrap()
}

fn batch(records: Vec<RecordBatch>) -> Batch {
    Batch::table(records, BatchMetadata::default()).unwrap()
}

fn cells(columns: &[ArrayRef], row: usize) -> Vec<ScalarValue> {
    columns
        .iter()
        .map(|a| ScalarValue::try_from_array(a, row).unwrap())
        .collect()
}

fn snapshot(records: &[RecordBatch]) -> BTreeMap<String, ScalarValue> {
    records
        .iter()
        .flat_map(|record| {
            (0..record.num_rows()).map(|row| {
                let cells = cells(record.columns(), row);
                (
                    format!("{:?}", &cells[..cells.len() - 1]),
                    cells.last().unwrap().clone(),
                )
            })
        })
        .collect()
}

// Use the real grouped factory, never scalar update_batch. Re-seeding each
// prefix also proves old-state-before-new-rows tie behavior needed by T2.
#[derive(Default)]
struct Model {
    keys: Vec<String>,
    state: Vec<ArrayRef>,
}

impl Model {
    async fn update(
        &mut self,
        runtime: &DataFusionRuntime,
        sql: &str,
        input: &Batch,
    ) -> BTreeMap<String, ScalarValue> {
        if input.num_rows() == 0 {
            return self.snapshot();
        }
        let _guard = runtime.query_lock.lock().await;
        let query = parse_select_query(sql).unwrap();
        let plan = runtime
            .prepare_query(
                runtime.context_for_rows(input.num_rows(), None, "not_evaluated"),
                &query,
                &BTreeMap::from([("input".into(), input.clone())]),
                Some("native-model"),
            )
            .await
            .unwrap();
        let aggregate = aggregate(plan.physical_plan.as_ref());
        let expression = &aggregate.aggr_expr()[0];
        let mut accumulator = expression.create_groups_accumulator().unwrap();
        if !self.state.is_empty() {
            accumulator
                .merge_batch(
                    &self.state,
                    &(0..self.keys.len()).collect::<Vec<_>>(),
                    self.keys.len(),
                )
                .unwrap();
        }
        let mut stream = aggregate.input().execute(0, plan.task_ctx.clone()).unwrap();
        while let Some(record) = stream.next().await {
            let record = record.unwrap();
            let evaluate = |e: &Arc<dyn PhysicalExpr>| {
                e.evaluate(&record)
                    .unwrap()
                    .into_array(record.num_rows())
                    .unwrap()
            };
            let keys = aggregate
                .group_expr()
                .expr()
                .iter()
                .map(|(e, _)| evaluate(e))
                .collect::<Vec<_>>();
            let indices = (0..record.num_rows())
                .map(|row| {
                    let key = format!("{:?}", cells(&keys, row));
                    self.keys.iter().position(|k| k == &key).unwrap_or_else(|| {
                        self.keys.push(key);
                        self.keys.len() - 1
                    })
                })
                .collect::<Vec<_>>();
            let values = expression
                .expressions()
                .iter()
                .chain(expression.order_bys().iter().map(|sort| &sort.expr))
                .map(evaluate)
                .collect::<Vec<_>>();
            accumulator
                .update_batch(&values, &indices, None, self.keys.len())
                .unwrap();
        }
        self.state = accumulator.state(EmitTo::All).unwrap();
        self.snapshot()
    }

    fn snapshot(&self) -> BTreeMap<String, ScalarValue> {
        self.keys
            .iter()
            .enumerate()
            .map(|(row, key)| {
                (
                    key.clone(),
                    ScalarValue::try_from_array(&self.state[0], row).unwrap(),
                )
            })
            .collect()
    }
}

async fn assert_prefixes(sql: &str, records: &[RecordBatch]) -> Model {
    use crate::{OperatorMetadata, StreamOperator};
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let mut model = Model::default();
    let job = crate::StreamJobContext::new(
        439,
        "ordered-production",
        crate::JsonMap::new(),
        None,
        crate::CancellationToken::new(),
    );
    let context = crate::StreamOperatorContext::new(&job, "ordered-production", None);
    let mut operator =
        crate::SqlOperator::new("ordered-production", sql, vec!["input".into()], vec![]).unwrap();
    for end in 1..=records.len() {
        let part = batch(vec![records[end - 1].clone()]);
        let decision = admission(&runtime, sql, &part).await;
        assert_eq!(
            decision,
            if part.num_rows() == 0 {
                Admission::Deferred
            } else {
                Admission::Proven
            },
            "{sql}"
        );
        let actual = model.update(&runtime, sql, &part).await;
        let mut output = crate::EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data("input", part.clone(), &context, &mut output)
            .await
            .unwrap();
        let produced = output.drain("output");
        assert_eq!(
            snapshot(
                produced[0]
                    .as_data()
                    .unwrap()
                    .table_payload()
                    .unwrap()
                    .batches()
            ),
            actual,
            "production {sql}, prefix {end}"
        );
        if records[..end].iter().any(|r| r.num_rows() != 0) {
            assert_eq!(
                operator
                    .checkpoint(crate::Epoch::INITIAL)
                    .unwrap()
                    .inline_metadata["state_layout"],
                serde_json::json!(5)
            );
        }
        let expected = runtime
            .sql(
                sql,
                &BTreeMap::from([("input".into(), batch(records[..end].to_vec()))]),
                Some("prefix-oracle"),
            )
            .await
            .unwrap();
        assert_eq!(
            actual,
            snapshot(expected.table_payload().unwrap().batches()),
            "{sql}, prefix {end}"
        );
    }
    model
}

#[tokio::test]
async fn test_ordered_last_every_prefix_and_legal_batch_splits_match_df55() {
    let rows = [
        (Some("A"), Some(true), Some(1.0), Some(-10), Some(0)),
        (Some("B"), Some(false), None, Some(5), Some(1)),
        (Some("A"), Some(true), Some(2.0), Some(3), Some(1)),
        (Some("A"), Some(true), Some(9.0), Some(3), Some(1)),
        (None, Some(true), None, None, None),
        (Some("A"), Some(true), None, None, Some(2)),
        (Some("B"), Some(false), Some(4.0), Some(0), Some(0)),
        (None, Some(true), Some(7.0), Some(-1), Some(0)),
        (Some("A"), Some(false), Some(8.0), Some(3), Some(3)),
        (Some("Z"), None, None, Some(6), None),
        (Some("Z"), None, None, Some(7), None),
    ];
    for (order, treatment) in [
        ("\"timestamp\"", ""),
        ("\"timestamp\" DESC", "RESPECT NULLS"),
        ("\"timestamp\" ASC NULLS FIRST", "RESPECT NULLS"),
        ("\"timestamp\" ASC NULLS LAST", "IGNORE NULLS"),
        ("\"timestamp\" DESC NULLS FIRST", "IGNORE NULLS"),
        ("\"timestamp\" DESC NULLS LAST", "RESPECT NULLS"),
        (
            "\"timestamp\" ASC NULLS FIRST, seq DESC NULLS LAST",
            "IGNORE NULLS",
        ),
        ("\"timestamp\" DESC NULLS LAST, seq ASC NULLS FIRST", ""),
    ] {
        let sql = format!(
            "SELECT stock, desk, LAST_VALUE(latest_price ORDER BY {order}) {treatment} AS price FROM input GROUP BY stock, desk ORDER BY stock, desk"
        );
        let mut final_snapshots = Vec::new();
        for width in [1, 3, 20] {
            let empty = quotes(&[]);
            let records = std::iter::once(empty.clone())
                .chain(
                    rows.chunks(width)
                        .flat_map(|part| [quotes(part), empty.clone()]),
                )
                .collect::<Vec<_>>();
            let model = assert_prefixes(&sql, &records).await;
            final_snapshots.push(model.state);
        }
        assert_eq!(final_snapshots[0], final_snapshots[1]);
        assert_eq!(final_snapshots[1], final_snapshots[2]);
    }
}

#[tokio::test]
async fn test_ordered_last_full_ties_keep_first_winner_in_and_across_batches() {
    let rows = [
        (Some("A"), Some(true), Some(10.0), Some(-1), Some(0)),
        (Some("A"), Some(true), Some(20.0), Some(-1), Some(0)),
        (Some("A"), Some(true), Some(30.0), Some(-1), Some(0)),
    ];
    for width in [1, 2, 3] {
        let records = rows.chunks(width).map(quotes).collect::<Vec<_>>();
        let model = assert_prefixes(QUERY, &records).await;
        assert_eq!(
            ScalarValue::try_from_array(&model.state[0], 0).unwrap(),
            ScalarValue::Float64(Some(10.0))
        );
    }
}

#[tokio::test]
async fn test_ordered_last_null_treatment_and_default_sort_options_are_exact() {
    let rows = [
        (Some("A"), None, Some(10.0), Some(1), Some(0)),
        (Some("A"), None, None, None, Some(0)),
        (Some("Z"), None, None, Some(-1), Some(0)),
    ];
    for (order, treatment, expected) in [
        ("\"timestamp\"", "", None),
        ("\"timestamp\"", "RESPECT NULLS", None),
        ("\"timestamp\"", "IGNORE NULLS", Some(10.0)),
        ("\"timestamp\" DESC", "", Some(10.0)),
        ("\"timestamp\" ASC NULLS FIRST", "", Some(10.0)),
        ("\"timestamp\" DESC NULLS LAST", "", None),
    ] {
        let sql = format!(
            "SELECT stock, LAST_VALUE(latest_price ORDER BY {order}) {treatment} AS price FROM input GROUP BY stock"
        );
        let model = assert_prefixes(&sql, &[quotes(&rows[..1]), quotes(&rows[1..])]).await;
        assert_eq!(
            ScalarValue::try_from_array(&model.state[0], 0).unwrap(),
            ScalarValue::Float64(expected)
        );
        assert!(
            ScalarValue::try_from_array(&model.state[0], 1)
                .unwrap()
                .is_null()
        );
        assert_eq!(
            ScalarValue::try_from_array(model.state.last().unwrap(), 1).unwrap(),
            ScalarValue::Boolean(Some(treatment != "IGNORE NULLS"))
        );
    }
}

#[tokio::test]
async fn test_ordered_last_supported_value_key_and_ordering_types() {
    let numeric = [
        DataType::Int8,
        DataType::Int16,
        DataType::Int32,
        DataType::Int64,
        DataType::UInt8,
        DataType::UInt16,
        DataType::UInt32,
        DataType::UInt64,
        DataType::Float32,
        DataType::Float64,
    ];
    let ordering = numeric[..8]
        .iter()
        .cloned()
        .chain(
            [
                TimeUnit::Second,
                TimeUnit::Millisecond,
                TimeUnit::Microsecond,
                TimeUnit::Nanosecond,
            ]
            .into_iter()
            .flat_map(|unit| {
                [
                    DataType::Timestamp(unit, None),
                    DataType::Timestamp(unit, Some("UTC".into())),
                ]
            }),
        )
        .collect::<Vec<_>>();
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let base = quotes(&[
        (Some("A"), Some(true), Some(2.0), Some(1), Some(1)),
        (None, None, None, Some(2), None),
    ]);
    // Vary one axis at a time; no Cartesian test matrix is needed for admission.
    for (column, types) in [
        (2, numeric.to_vec()),
        (3, ordering),
        (0, vec![DataType::Utf8, DataType::LargeUtf8]),
    ] {
        for dtype in types {
            let mut columns = base.columns().to_vec();
            let source = if column == 3 {
                base.column(4)
            } else {
                base.column(column)
            };
            columns[column] = cast(source, &dtype).unwrap();
            let fields = base
                .schema()
                .fields()
                .iter()
                .enumerate()
                .map(|(i, f)| {
                    if i == column {
                        Field::new(f.name(), dtype.clone(), true)
                    } else {
                        f.as_ref().clone()
                    }
                })
                .collect::<Vec<_>>();
            let record = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap();
            assert_eq!(
                admission(&runtime, QUERY, &batch(vec![record.clone()])).await,
                Admission::Proven,
                "{dtype:?}"
            );
            assert_prefixes(QUERY, &[record.slice(0, 1), record.slice(1, 1)]).await;
        }
    }
    assert_prefixes("SELECT stock, desk, seq, LAST_VALUE(latest_price ORDER BY \"timestamp\") FROM input GROUP BY stock, desk, seq", &[base]).await;
}

#[tokio::test]
async fn test_ordered_last_unsupported_shapes_remain_retained_candidates() {
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let input = input();
    for sql in [
        "SELECT LAST_VALUE(latest_price ORDER BY timestamp) FROM input",
        "SELECT stock, LAST_VALUE(latest_price) FROM input GROUP BY stock",
        "SELECT stock, LAST_VALUE(latest_price + 1 ORDER BY timestamp) FROM input GROUP BY stock",
        "SELECT stock, LAST_VALUE(latest_price ORDER BY latest_price) FROM input GROUP BY stock",
        "SELECT stock, LAST_VALUE(stock ORDER BY timestamp) FROM input GROUP BY stock",
        "SELECT stock, LAST_VALUE(latest_price ORDER BY timestamp), COUNT(*) FROM input GROUP BY stock",
        "SELECT stock, LAST_VALUE(latest_price ORDER BY timestamp) FROM input WHERE latest_price > 0 GROUP BY stock",
        "SELECT stock, LAST_VALUE(latest_price ORDER BY timestamp) FILTER (WHERE latest_price > 0) FROM input GROUP BY stock",
        "SELECT stock, LAST_VALUE(DISTINCT latest_price ORDER BY latest_price) FROM input GROUP BY stock",
        "SELECT stock, LAST_VALUE(latest_price ORDER BY timestamp) FROM input GROUP BY stock LIMIT 1",
        "SELECT stock, LAST_VALUE(latest_price ORDER BY timestamp) AS p FROM input GROUP BY stock HAVING p > 0",
        "SELECT stock, LAST_VALUE(latest_price ORDER BY timestamp) + 1 FROM input GROUP BY stock",
        "SELECT latest_price, LAST_VALUE(latest_price ORDER BY timestamp) FROM input GROUP BY latest_price",
        "SELECT stock, LAST_VALUE(latest_price) OVER (PARTITION BY stock ORDER BY timestamp) FROM input",
        "SELECT a.stock, LAST_VALUE(a.latest_price ORDER BY a.timestamp) FROM input a JOIN input b ON a.stock = b.stock GROUP BY a.stock",
        "SELECT stock, LAST_VALUE(latest_price ORDER BY CAST(timestamp AS BIGINT) + 1) FROM input GROUP BY stock",
    ] {
        assert_eq!(
            admission(&runtime, sql, &input).await,
            Admission::Retained("query shape or types"),
            "{sql}"
        );
    }
}

#[tokio::test]
async fn test_ordered_last_empty_first_batch_does_not_certify_or_lock_a_mode() {
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let empty = batch(vec![
        input().table_payload().unwrap().batches()[0].slice(0, 0),
    ]);
    assert_eq!(
        admission(&runtime, QUERY, &empty).await,
        Admission::Deferred
    );
    assert_eq!(
        admission(&runtime, QUERY, &input()).await,
        Admission::Proven
    );
}

#[tokio::test]
async fn test_ordered_last_nondefault_execution_models_do_not_certify() {
    for config in [
        DataFusionConfig {
            target_partitions: 2,
            ..DataFusionConfig::default()
        },
        DataFusionConfig {
            batch_size: 17,
            ..DataFusionConfig::default()
        },
        DataFusionConfig {
            parallelism_mode: DataFusionParallelismMode::Auto,
            ..DataFusionConfig::default()
        },
    ] {
        let runtime = DataFusionRuntime::new(config).unwrap();
        assert_eq!(
            admission(&runtime, QUERY, &input()).await,
            Admission::Retained("execution model")
        );
    }
    let runtime = DataFusionRuntime::with_benchmark_pool(
        DataFusionConfig::default(),
        Arc::new(datafusion::execution::memory_pool::GreedyMemoryPool::new(
            1 << 30,
        )),
    )
    .unwrap();
    assert_eq!(
        admission(&runtime, QUERY, &input()).await,
        Admission::Retained("execution model")
    );
}

#[tokio::test]
async fn test_ordered_last_non_numeric_values_and_non_integral_ordering_do_not_certify() {
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let base = input().table_payload().unwrap().batches()[0].clone();
    let values = vec![
        DataType::Utf8,
        DataType::Boolean,
        DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
        DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
    ];
    let orderings = vec![
        DataType::Float64,
        DataType::Utf8,
        DataType::Boolean,
        DataType::Date64,
    ];
    for (column, types) in [(1, values), (2, orderings)] {
        for dtype in types {
            let mut columns = base.columns().to_vec();
            columns[column] = new_null_array(&dtype, base.num_rows());
            let fields = base
                .schema()
                .fields()
                .iter()
                .enumerate()
                .map(|(i, f)| {
                    if i == column {
                        Field::new(f.name(), dtype.clone(), true)
                    } else {
                        f.as_ref().clone()
                    }
                })
                .collect::<Vec<_>>();
            let changed = batch(vec![
                RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap(),
            ]);
            assert_eq!(
                admission(&runtime, QUERY, &changed).await,
                Admission::Retained("query shape or types"),
                "{dtype:?}"
            );
        }
    }
}

#[tokio::test]
async fn test_ordered_last_physical_certificate_rejects_partial_reverse_and_input_sort() {
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let input = input();
    let query = parse_select_query(QUERY).unwrap();
    let tables = BTreeMap::from([("input".into(), input.clone())]);
    let plan = runtime
        .prepare_query(
            runtime.context_for_rows(input.num_rows(), None, "not_evaluated"),
            &query,
            &tables,
            Some("physical-proof"),
        )
        .await
        .unwrap();
    let aggregate = aggregate(plan.physical_plan.as_ref());
    let accept = |plan: &dyn ExecutionPlan| {
        let mut census = [0, 0];
        physical_shape(plan, &input, false, &mut census) && census == [1, 1]
    };
    assert!(accept(plan.physical_plan.as_ref()));
    let partial = AggregateExec::try_new(
        AggregateMode::Partial,
        aggregate.group_expr().clone(),
        aggregate.aggr_expr().to_vec(),
        aggregate.filter_expr().to_vec(),
        aggregate.input().clone(),
        aggregate.input_schema(),
    )
    .unwrap();
    assert!(!accept(&partial));
    let reversed = AggregateExec::try_new(
        AggregateMode::Single,
        aggregate.group_expr().clone(),
        vec![Arc::new(aggregate.aggr_expr()[0].reverse_expr().unwrap())],
        aggregate.filter_expr().to_vec(),
        aggregate.input().clone(),
        aggregate.input_schema(),
    )
    .unwrap();
    assert!(!accept(&reversed));
    let order =
        datafusion::physical_expr::LexOrdering::new(aggregate.aggr_expr()[0].order_bys().to_vec())
            .unwrap();
    let sorted = AggregateExec::try_new(
        AggregateMode::Single,
        aggregate.group_expr().clone(),
        aggregate.aggr_expr().to_vec(),
        aggregate.filter_expr().to_vec(),
        Arc::new(SortExec::new(order, aggregate.input().clone())),
        aggregate.input_schema(),
    )
    .unwrap();
    assert!(!accept(&sorted));
}

#[tokio::test]
async fn test_ordered_last_model_survives_batch_and_cardinality_thresholds() {
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    for (rows, groups) in [
        (1, 1),
        (8191, 8),
        (8193, 8),
        (100_003, 8),
        (100_003, 100_003),
    ] {
        let record = RecordBatch::try_from_iter(vec![
            (
                "stock",
                Arc::new(Int64Array::from(
                    (0..rows).map(|i| i % groups).collect::<Vec<_>>(),
                )) as ArrayRef,
            ),
            (
                "latest_price",
                Arc::new(Float64Array::from(vec![
                    1.0;
                    usize::try_from(rows).unwrap()
                ])) as ArrayRef,
            ),
            (
                "timestamp",
                Arc::new(Int64Array::from((0..rows).rev().collect::<Vec<_>>())) as ArrayRef,
            ),
        ])
        .unwrap();
        assert_eq!(
            admission(&runtime, QUERY, &batch(vec![record.clone()])).await,
            Admission::Proven,
            "rows={rows}, groups={groups}"
        );
        if groups == 8 {
            assert_prefixes(
                QUERY,
                &[
                    record.slice(0, 1),
                    record.slice(1, usize::try_from(rows - 1).unwrap()),
                ],
            )
            .await;
        }
    }
}
