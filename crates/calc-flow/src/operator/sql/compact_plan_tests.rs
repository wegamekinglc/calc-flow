use super::*;
use crate::{BatchMetadata, CancellationToken, DataFusionConfig, JsonMap, StreamJobContext};
use datafusion::{
    arrow::array::{Int32Array, StringArray, new_empty_array},
    catalog::{CatalogProvider, MemoryCatalogProvider, MemorySchemaProvider},
    common::{ScalarValue, TableReference},
    execution::{
        SessionStateBuilder,
        memory_pool::{MemoryConsumer, MemoryPool},
    },
};
use std::collections::{BTreeMap, HashMap as Metadata};

#[tokio::test]
async fn test_compact_paid_sync_native_fields_match_async_full_and_projected() {
    for rolling in [false, true] {
        let runtime = configured_runtime(rolling);
        for kind in [
            DataType::Int8,
            DataType::Int16,
            DataType::Int32,
            DataType::Int64,
            DataType::UInt8,
            DataType::UInt16,
            DataType::UInt32,
            DataType::UInt64,
        ] {
            for grouped in [false, true] {
                native_parity(&runtime, kind.clone(), grouped, false).await;
            }
        }
    }
    let runtime = configured_runtime(true);
    for (width, precision) in [(32, 9), (64, 18), (128, 38), (256, 76)] {
        for scale in [0, 2, precision] {
            let scale = i8::try_from(scale).unwrap();
            let kind = match width {
                32 => DataType::Decimal32(precision, scale),
                64 => DataType::Decimal64(precision, scale),
                128 => DataType::Decimal128(precision, scale),
                256 => DataType::Decimal256(precision, scale),
                _ => unreachable!(),
            };
            for grouped in [false, true] {
                native_parity(&runtime, kind.clone(), grouped, true).await;
            }
        }
    }
}

#[tokio::test]
async fn test_compact_paid_sync_native_ineligible_and_untrusted_bindings_release_credit() {
    let runtime = configured_runtime(false);
    let pool = pool(&runtime);
    for (kind, text) in [
        (DataType::Float64, "SELECT SUM(value) FROM events"),
        (DataType::Int64, "SELECT AVG(value) FROM events"),
        (DataType::Int64, "SELECT COUNT(DISTINCT value) FROM events"),
        (DataType::Int64, "SELECT SUM(value + 1) FROM events"),
        (
            DataType::Int64,
            "SELECT SUM(value) FILTER (WHERE value > 0) FROM events",
        ),
        (
            DataType::Int64,
            "SELECT SUM(value) FROM events WHERE value > 0",
        ),
        (
            DataType::Decimal128(10, -1),
            "SELECT AVG(value) FROM events",
        ),
    ] {
        let query = crate::expression::parse_select_query(text).unwrap();
        let logical = schema(&kind);
        let (raw, analyzed) = runtime
            .incremental_sql_plan(&query, "events", logical.clone(), "native")
            .await
            .unwrap();
        assert!(
            IncrementalSql::from_plan(&runtime, &query, logical.clone(), &raw, &analyzed, "native")
                .unwrap()
                .is_none()
        );
        assert!(
            IncrementalSql::plan_sync(
                &runtime,
                &query,
                "events",
                logical.clone(),
                logical,
                "native"
            )
            .unwrap()
            .is_none()
        );
        assert_eq!(pool.reserved(), 0);
    }
    untrusted_bindings_release_credit(&runtime, &pool);
}

fn untrusted_bindings_release_credit(runtime: &DataFusionRuntime, pool: &Arc<dyn MemoryPool>) {
    let logical = schema(&DataType::Int64);
    let query =
        crate::expression::parse_select_query("SELECT key, SUM(value) FROM events GROUP BY key")
            .unwrap();
    let original = logical.clone();
    for value in [
        logical.field(2).clone().with_metadata(Metadata::new()),
        Field::new("value", DataType::Int32, true)
            .with_metadata(logical.field(2).metadata().clone()),
        Field::new("value", DataType::Int64, false)
            .with_metadata(logical.field(2).metadata().clone()),
    ] {
        let physical = Arc::new(Schema::new(vec![value, logical.field(0).clone()]));
        assert!(
            IncrementalSql::plan_sync(
                runtime,
                &query,
                "events",
                logical.clone(),
                physical,
                "native"
            )
            .unwrap()
            .is_none()
        );
        assert_eq!(pool.reserved(), 0);
        assert_eq!(logical, original);
    }
    let unused = Field::new("unused", DataType::Binary, true)
        .with_metadata(Metadata::from([("audit".into(), "independent".into())]));
    let independent = Arc::new(Schema::new_with_metadata(
        vec![logical.field(0).clone(), unused, logical.field(2).clone()],
        logical.metadata().clone(),
    ));
    assert!(
        IncrementalSql::plan_sync(
            runtime,
            &query,
            "events",
            logical.clone(),
            independent.clone(),
            "native"
        )
        .unwrap()
        .is_none()
    );
    assert_eq!(pool.reserved(), 0);
    let candidate = IncrementalSql::plan_sync(
        runtime,
        &query,
        "events",
        independent.clone(),
        Arc::new(independent.project(&[2, 0]).unwrap()),
        "native",
    )
    .unwrap()
    .unwrap();
    assert_eq!(candidate.keys, vec![1]);
    assert_eq!(candidate.variable_columns, Vec::<usize>::new());
    drop(candidate);
    assert_eq!(pool.reserved(), 0);
    assert_eq!(logical, original);
}

#[tokio::test]
async fn test_compact_paid_sync_empty_null_continuation_matches_async_and_sql() {
    for kind in [
        DataType::Int64,
        DataType::Decimal32(9, 2),
        DataType::Decimal64(18, 2),
        DataType::Decimal128(38, 2),
        DataType::Decimal256(76, 2),
    ] {
        for grouped in [false, true] {
            empty_null_continuation(&kind, grouped).await;
        }
    }
}

async fn empty_null_continuation(kind: &DataType, grouped: bool) {
    let runtime = configured_runtime(true);
    let logical = schema(kind);
    let physical = Arc::new(logical.project(&[2, 0]).unwrap());
    let decimal = !matches!(kind, DataType::Int64);
    let text = query_text(grouped, decimal);
    let query = crate::expression::parse_select_query(&text).unwrap();
    let (raw, analyzed) = runtime
        .incremental_sql_plan(&query, "events", logical.clone(), "continuation")
        .await
        .unwrap();
    let mut oracle = IncrementalSql::from_plan(
        &runtime,
        &query,
        physical.clone(),
        &raw,
        &analyzed,
        "continuation",
    )
    .unwrap()
    .unwrap();
    let mut actual = IncrementalSql::plan_sync(
        &runtime,
        &query,
        "events",
        logical.clone(),
        physical,
        "continuation",
    )
    .unwrap()
    .unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "continuation", None);
    let mut prefix = Vec::new();
    for values in [
        vec![],
        vec![None, None, None],
        vec![Some(3), None, Some(7)],
        vec![],
    ] {
        let record = record(kind, &values);
        prefix.push(record.clone());
        let incoming = Batch::table(
            vec![record.project(&[2, 0]).unwrap()],
            BatchMetadata::default(),
        )
        .unwrap();
        let before = incoming.clone();
        let next = actual
            .update(&incoming, &context, "continuation")
            .await
            .unwrap();
        let reference = oracle
            .update(&incoming, &context, "continuation")
            .await
            .unwrap();
        assert_eq!(
            next.records
                .iter()
                .map(RecordBatch::schema)
                .collect::<Vec<_>>(),
            reference
                .records
                .iter()
                .map(RecordBatch::schema)
                .collect::<Vec<_>>()
        );
        assert_eq!(rows(&next.records), rows(&reference.records));
        let expected = runtime
            .sql(
                &text,
                &BTreeMap::from([(
                    "events".into(),
                    Batch::table(prefix.clone(), BatchMetadata::default()).unwrap(),
                )]),
                None,
            )
            .await
            .unwrap();
        assert_eq!(
            rows(&next.records),
            rows(expected.table_payload().unwrap().batches())
        );
        assert_eq!(
            actual.output_schema,
            expected.table_payload().unwrap().schema().clone()
        );
        actual.commit(next);
        oracle.commit(reference);
        assert_eq!(snapshot(&actual), snapshot(&oracle));
        assert_eq!(
            rows(incoming.table_payload().unwrap().batches()),
            rows(before.table_payload().unwrap().batches())
        );
    }
}

#[tokio::test]
async fn test_compact_paid_sync_catalog_shadow_and_function_identity_boundaries() {
    let runtime = configured_runtime(true);
    let context = runtime.incremental_planner_context(0);
    let catalog = Arc::new(MemoryCatalogProvider::new());
    catalog
        .register_schema("ticks", Arc::new(MemorySchemaProvider::new()))
        .unwrap();
    context.register_catalog("analysis", catalog);
    let state = context.state_ref();
    {
        let mut state = state.write();
        let options = state.config_mut().options_mut();
        options.catalog.default_catalog = "analysis".into();
        options.catalog.default_schema = "ticks".into();
        options.sql_parser.parse_float_as_decimal = true;
    }
    let logical = schema(&DataType::Int64);
    let query =
        crate::expression::parse_select_query("SELECT SUM(value) FROM analysis.ticks.events")
            .unwrap();
    let shadow = Arc::new(
        datafusion::datasource::MemTable::try_new(schema(&DataType::Float64), vec![vec![]])
            .unwrap(),
    );
    context.register_table("events", shadow.clone()).unwrap();
    let expected: Arc<dyn datafusion::catalog::TableProvider> = shadow;
    let plan = IncrementalSql::plan_sync(
        &runtime,
        &query,
        "events",
        logical.clone(),
        logical.clone(),
        "shadow",
    )
    .unwrap()
    .unwrap();
    assert_eq!(plan.aggregates[0].field().data_type(), &DataType::Int64);
    assert!(Arc::ptr_eq(
        &context.table_provider("events").await.unwrap(),
        &expected
    ));
    let error = runtime
        .incremental_sql_plan(&query, "events", logical.clone(), "shadow")
        .await
        .unwrap_err();
    let CalcFlowError::DataFusion { node_id, message } = error else {
        panic!("expected registration collision")
    };
    assert_eq!(node_id.as_deref(), Some("shadow"));
    assert!(message.contains("table events already exists"));
    assert!(Arc::ptr_eq(
        &context.table_provider("events").await.unwrap(),
        &expected
    ));
    assert!(Arc::ptr_eq(
        &context.deregister_table("events").unwrap().unwrap(),
        &expected
    ));
    drop(plan);
    let oracle = IncrementalSql::plan(&runtime, &query, "events", logical.clone(), "shadow")
        .await
        .unwrap()
        .unwrap();
    let plan = IncrementalSql::plan_sync(
        &runtime,
        &query,
        "events",
        logical.clone(),
        logical.clone(),
        "shadow",
    )
    .unwrap()
    .unwrap();
    same_native(&plan, &oracle);
    drop((plan, oracle));
    context.register_udaf(
        datafusion::functions_aggregate::min_max::min_udaf()
            .as_ref()
            .clone()
            .with_aliases(["sum"]),
    );
    assert!(
        IncrementalSql::plan(&runtime, &query, "events", logical.clone(), "shadow")
            .await
            .unwrap()
            .is_none()
    );
    assert!(
        IncrementalSql::plan_sync(
            &runtime,
            &query,
            "events",
            logical.clone(),
            logical,
            "shadow"
        )
        .unwrap()
        .is_none()
    );
    assert_eq!(pool(&runtime).reserved(), 0);
}

#[tokio::test]
async fn test_compact_paid_sync_old_new_coexistence_and_refusals_preserve_state() {
    let runtime = configured_runtime(false);
    let pool = pool(&runtime);
    let logical = schema(&DataType::Int64);
    let physical = Arc::new(logical.project(&[2, 0]).unwrap());
    let query = crate::expression::parse_select_query(&query_text(true, false)).unwrap();
    let mut old = IncrementalSql::plan(&runtime, &query, "events", logical.clone(), "credit")
        .await
        .unwrap()
        .unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "credit", None);
    let input = Batch::table(
        vec![record(&DataType::Int64, &[Some(3), None, Some(7)])],
        BatchMetadata::default(),
    )
    .unwrap();
    let transaction = old.update(&input, &context, "credit").await.unwrap();
    old.commit(transaction);
    let installed = snapshot(&old);
    let before = pool.reserved();
    let plans = prepare_sync_plan(&runtime, &query, "events", logical.clone(), "credit").unwrap();
    let scratch = plans.reserved_bytes();
    assert!(scratch > 0);
    assert_eq!(pool.reserved(), before + scratch);
    let candidate = IncrementalSql::from_plan(
        &runtime,
        &query,
        physical.clone(),
        &plans.raw,
        &plans.analyzed,
        "credit",
    )
    .unwrap()
    .unwrap();
    let native = candidate.reservation.size();
    assert_eq!(pool.reserved(), before + scratch + native);
    drop(plans);
    assert_eq!(pool.reserved(), before + native);
    drop(candidate);
    assert_eq!(pool.reserved(), before);
    for available in [0, scratch + native - 1] {
        let pressure = MemoryConsumer::new("planner-pressure").register(&pool);
        pressure.try_grow((1 << 30) - before - available).unwrap();
        let paid = pool.reserved();
        let result = IncrementalSql::plan_sync(
            &runtime,
            &query,
            "events",
            logical.clone(),
            physical.clone(),
            "credit",
        );
        let Err(CalcFlowError::DataFusion { node_id, .. }) = result else {
            panic!("expected paid planner refusal")
        };
        assert_eq!(node_id.as_deref(), Some("credit"));
        assert_eq!(snapshot(&old), installed);
        assert_eq!(pool.reserved(), paid);
        drop(pressure);
        assert_eq!(pool.reserved(), before);
    }
    let candidate =
        IncrementalSql::plan_sync(&runtime, &query, "events", logical, physical, "credit")
            .unwrap()
            .unwrap();
    assert_eq!(pool.reserved(), before + candidate.reservation.size());
    assert_eq!(snapshot(&old), installed);
    drop((candidate, old));
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn test_compact_paid_sync_lookup_refusals_release_credit_and_preserve_node_context() {
    let runtime = configured_runtime(false);
    let logical = schema(&DataType::Int64);
    for text in [
        "SELECT SUM(value) FROM other",
        "SELECT SUM(a.value) FROM events a JOIN events b ON a.key = b.key",
        "SELECT @amount FROM events",
        "SELECT events.value FROM events CROSS JOIN custom_table(1)",
        "SELECT nonexistent_function(value) FROM events",
        "SELECT SUM(value) FROM other_catalog.public.events",
    ] {
        let query = crate::expression::parse_select_query(text).unwrap();
        let result = IncrementalSql::plan_sync(
            &runtime,
            &query,
            "events",
            logical.clone(),
            logical.clone(),
            "sync-node",
        );
        let Err(CalcFlowError::DataFusion { node_id, .. }) = result else {
            panic!("expected unsupported planning error")
        };
        assert_eq!(node_id.as_deref(), Some("sync-node"));
        assert_eq!(pool(&runtime).reserved(), 0);
    }
    let context = runtime.incremental_planner_context(0);
    assert!(
        !context
            .state()
            .schema_for_ref(TableReference::from("events"))
            .unwrap()
            .table_exist("events")
    );
    assert_eq!(
        runtime
            .incremental_sql_plan_calls
            .load(std::sync::atomic::Ordering::Relaxed),
        0
    );
}

#[tokio::test]
async fn test_compact_paid_sync_external_type_and_variable_providers_are_rejected() {
    let runtime = configured_runtime(false);
    let context = runtime.incremental_planner_context(0);
    let replacement = SessionStateBuilder::new_from_existing(context.state())
        .with_type_planner(Arc::new(TypeExtension))
        .build();
    *context.state_ref().write() = replacement;
    extension_refusal(&runtime, "SELECT CAST(value AS BIGINT) FROM events").await;
    let runtime = configured_runtime(false);
    runtime.incremental_planner_context(0).register_variable(
        datafusion::logical_expr::var_provider::VarType::UserDefined,
        Arc::new(VariableExtension),
    );
    extension_refusal(&runtime, "SELECT @amount FROM events").await;
}

fn configured_runtime(rolling: bool) -> DataFusionRuntime {
    let runtime = DataFusionRuntime::new(DataFusionConfig {
        batch_size: if rolling { 257 } else { 19 },
        target_partitions: if rolling { 3 } else { 2 },
        min_rows_per_partition: 1,
        enable_rolling_rewrite: rolling,
        ..DataFusionConfig::default()
    })
    .unwrap();
    runtime.incremental_planner_context(1 << 20);
    runtime
}

fn pool(runtime: &DataFusionRuntime) -> Arc<dyn MemoryPool> {
    runtime
        .incremental_planner_context(0)
        .runtime_env()
        .memory_pool
        .clone()
}

fn schema(kind: &DataType) -> SchemaRef {
    Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("key", DataType::Int32, true)
                .with_metadata(Metadata::from([("role".into(), "group".into())])),
            Field::new("unused", DataType::Utf8, true),
            Field::new("value", kind.clone(), true)
                .with_metadata(Metadata::from([("unit".into(), "native".into())])),
        ],
        Metadata::from([("origin".into(), "full-input".into())]),
    ))
}

fn query_text(grouped: bool, decimal: bool) -> String {
    let prefix = if grouped { "key, " } else { "" };
    let suffix = if grouped { " GROUP BY key" } else { "" };
    let avg = if decimal { ", AVG(value) AS mean" } else { "" };
    format!(
        "SELECT {prefix}SUM(value) AS total, COUNT(value) AS valid, MIN(value) AS lo, MAX(value) AS hi, COUNT(*) AS rows, COUNT(1) AS repeated{avg} FROM events{suffix}"
    )
}

async fn native_parity(runtime: &DataFusionRuntime, kind: DataType, grouped: bool, decimal: bool) {
    let query = crate::expression::parse_select_query(&query_text(grouped, decimal)).unwrap();
    let logical = schema(&kind);
    let (raw, analyzed) = runtime
        .incremental_sql_plan(&query, "events", logical.clone(), "parity")
        .await
        .unwrap();
    let plans = prepare_sync_plan(runtime, &query, "events", logical.clone(), "parity").unwrap();
    assert_eq!(plans.raw, raw);
    assert_eq!(plans.analyzed, analyzed);
    assert_eq!(plans.analyzed.schema(), analyzed.schema());
    for physical in [logical.clone(), Arc::new(logical.project(&[2, 0]).unwrap())] {
        let oracle =
            IncrementalSql::from_plan(runtime, &query, physical.clone(), &raw, &analyzed, "parity")
                .unwrap()
                .unwrap();
        let actual = IncrementalSql::plan_sync(
            runtime,
            &query,
            "events",
            logical.clone(),
            physical.clone(),
            "parity",
        )
        .unwrap()
        .unwrap();
        same_native(&actual, &oracle);
        assert_eq!(
            actual.keys,
            if grouped {
                vec![physical.index_of("key").unwrap()]
            } else {
                vec![]
            }
        );
        assert_eq!(actual.aggregates.len(), if decimal { 6 } else { 5 });
        assert_eq!(
            actual.projection.len(),
            6 + usize::from(grouped) + usize::from(decimal)
        );
        assert!(
            actual
                .schema
                .field_with_name("value")
                .unwrap()
                .metadata()
                .contains_key("unit")
        );
    }
}

fn same_native(actual: &IncrementalSql, oracle: &IncrementalSql) {
    assert_eq!(actual.schema, oracle.schema);
    assert_eq!(actual.aggregate_schema, oracle.aggregate_schema);
    assert_eq!(actual.output_schema, oracle.output_schema);
    assert_eq!(actual.keys, oracle.keys);
    assert_eq!(actual.variable_columns, oracle.variable_columns);
    assert_eq!(
        (
            actual.plan_bytes,
            actual.aggregate_bytes,
            actual.finalizer_bytes
        ),
        (
            oracle.plan_bytes,
            oracle.aggregate_bytes,
            oracle.finalizer_bytes
        )
    );
    assert_eq!(
        actual
            .aggregates
            .iter()
            .map(|expr| expr.state_fields().unwrap())
            .collect::<Vec<_>>(),
        oracle
            .aggregates
            .iter()
            .map(|expr| expr.state_fields().unwrap())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        actual
            .aggregates
            .iter()
            .map(|expr| expr.field())
            .collect::<Vec<_>>(),
        oracle
            .aggregates
            .iter()
            .map(|expr| expr.field())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        actual
            .projection
            .iter()
            .map(|expr| (
                expr.data_type(&actual.aggregate_schema).unwrap(),
                expr.nullable(&actual.aggregate_schema).unwrap()
            ))
            .collect::<Vec<_>>(),
        oracle
            .projection
            .iter()
            .map(|expr| (
                expr.data_type(&oracle.aggregate_schema).unwrap(),
                expr.nullable(&oracle.aggregate_schema).unwrap()
            ))
            .collect::<Vec<_>>()
    );
}

fn record(kind: &DataType, values: &[Option<i64>]) -> RecordBatch {
    let value = if values.is_empty() {
        new_empty_array(kind)
    } else {
        ScalarValue::iter_to_array(values.iter().map(|value| match kind {
            DataType::Decimal32(p, s) => {
                ScalarValue::Decimal32(value.map(|x| i32::try_from(x).unwrap()), *p, *s)
            }
            DataType::Decimal64(p, s) => ScalarValue::Decimal64(*value, *p, *s),
            DataType::Decimal128(p, s) => ScalarValue::Decimal128(value.map(i128::from), *p, *s),
            DataType::Decimal256(p, s) => ScalarValue::Decimal256(
                value.map(|x| datafusion::arrow::datatypes::i256::from_i128(i128::from(x))),
                *p,
                *s,
            ),
            _ => ScalarValue::Int64(*value).cast_to(kind).unwrap(),
        }))
        .unwrap()
    };
    RecordBatch::try_new(
        schema(kind),
        vec![
            Arc::new(Int32Array::from(
                values
                    .iter()
                    .enumerate()
                    .map(|(i, _)| [Some(0), Some(1), None][i % 3])
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(vec![Some("unused"); values.len()])),
            value,
        ],
    )
    .unwrap()
}

fn rows(records: &[RecordBatch]) -> Vec<Vec<ScalarValue>> {
    let mut rows = records
        .iter()
        .flat_map(|record| {
            (0..record.num_rows()).map(|row| {
                record
                    .columns()
                    .iter()
                    .map(|array| ScalarValue::try_from_array(array, row).unwrap())
                    .collect::<Vec<_>>()
            })
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| left.partial_cmp(right).unwrap());
    rows
}

type GroupState = (Vec<ScalarValue>, Vec<Vec<ScalarValue>>, Vec<ScalarValue>);

fn snapshot(plan: &IncrementalSql) -> Vec<GroupState> {
    plan.groups
        .iter()
        .map(|group| {
            (
                group.values.to_vec(),
                group.states.clone(),
                group.results.clone(),
            )
        })
        .collect()
}

fn job() -> StreamJobContext {
    StreamJobContext::new(
        1,
        "compact-plan",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
}

async fn extension_refusal(runtime: &DataFusionRuntime, text: &str) {
    let logical = schema(&DataType::Int64);
    let query = crate::expression::parse_select_query(text).unwrap();
    let _oracle = runtime
        .incremental_sql_plan(&query, "events", logical.clone(), "extension")
        .await
        .unwrap();
    let result = IncrementalSql::plan_sync(
        runtime,
        &query,
        "events",
        logical.clone(),
        logical,
        "extension",
    );
    let Err(CalcFlowError::DataFusion { node_id, .. }) = result else {
        panic!("expected extension rejection")
    };
    assert_eq!(node_id.as_deref(), Some("extension"));
    assert_eq!(pool(runtime).reserved(), 0);
}

#[derive(Debug)]
struct TypeExtension;
impl datafusion::logical_expr::planner::TypePlanner for TypeExtension {
    fn plan_type_field(
        &self,
        _kind: &datafusion::sql::sqlparser::ast::DataType,
    ) -> datafusion::error::Result<Option<datafusion::arrow::datatypes::FieldRef>> {
        Ok(Some(Arc::new(Field::new("", DataType::Int64, true))))
    }
}

#[derive(Debug)]
struct VariableExtension;
impl datafusion::logical_expr::var_provider::VarProvider for VariableExtension {
    fn get_type(&self, _names: &[String]) -> Option<DataType> {
        Some(DataType::Int64)
    }
    fn get_value(&self, _names: Vec<String>) -> datafusion::error::Result<ScalarValue> {
        Ok(ScalarValue::Int64(Some(5)))
    }
}
