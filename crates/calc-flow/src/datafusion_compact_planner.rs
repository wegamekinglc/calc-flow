use super::*;
use datafusion::{
    arrow::datatypes::{DataType, Field, FieldRef, Schema},
    catalog::{CatalogProvider, MemoryCatalogProvider, MemorySchemaProvider},
    common::{
        DFSchema, ResolvedTableReference, ScalarValue, TableReference, config::ConfigOptions,
    },
    datasource::provider_as_source,
    error::{DataFusionError, Result as DFResult},
    execution::SessionState,
    logical_expr::{
        AggregateUDF, Expr, HigherOrderUDF, LogicalPlan, TableSource, WindowUDF,
        execution_props::ExecutionProps,
        planner::{ContextProvider, ExprPlanner, RelationPlanner, TypePlanner},
    },
    physical_expr::{aggregate::LoweredAggregateBuilder, create_physical_expr},
    sql::planner::{ParserOptions, SqlToRel},
};
use std::{cell::Cell, collections::HashMap};

#[tokio::test]
async fn test_compact_sync_integer_native_fields_match_async_planner() {
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
                let (prefix, suffix) = if grouped {
                    ("key, ", " GROUP BY key")
                } else {
                    ("", "")
                };
                let sql = format!(
                    "SELECT {prefix}SUM(value), COUNT(value), MIN(value), MAX(value), COUNT(*), COUNT(1) FROM events{suffix}"
                );
                let schema = input_schema(kind.clone());
                let census = assert_parity(&runtime, &sql, schema.clone(), "events")
                    .await
                    .unwrap();
                assert_eq!(census.states.len(), 5);
                assert_eq!(census.results.len(), 5);
                let output_columns = 6 + usize::from(grouped);
                assert_eq!(census.output.fields().len(), output_columns);
                assert_eq!(census.projection_types.len(), output_columns);
                let query = parse_select_query(&sql).unwrap();
                let (raw, _) = sync_plan(&runtime, &query, "events", schema).unwrap();
                let (projection, aggregate) = aggregate_shape(&raw).unwrap();
                let slots = projection
                    .expr
                    .iter()
                    .map(|expr| {
                        let Expr::Column(column) = unalias(expr) else {
                            panic!("expected aggregate output column");
                        };
                        aggregate.schema.index_of_column(column).unwrap()
                    })
                    .collect::<Vec<_>>();
                assert_eq!(
                    slots,
                    if grouped {
                        vec![0, 1, 2, 3, 4, 5, 5]
                    } else {
                        vec![0, 1, 2, 3, 4, 4]
                    }
                );
                assert_eq!(census.keys, if grouped { vec![0] } else { vec![] });
                assert!(census.states.iter().all(|fields| !fields.is_empty()));
            }
        }
    }
}

#[tokio::test]
async fn test_compact_sync_string_extrema_native_fields_match_async_planner() {
    for rolling in [false, true] {
        let runtime = configured_runtime(rolling);
        for kind in [DataType::Utf8, DataType::LargeUtf8] {
            for grouped in [false, true] {
                let (prefix, suffix) = if grouped {
                    ("key, ", " GROUP BY key")
                } else {
                    ("", "")
                };
                let sql = format!(
                    "SELECT {prefix}MIN(value), MAX(value), COUNT(value), COUNT(*), MIN(value) AS again FROM events{suffix}"
                );
                let census = assert_parity(&runtime, &sql, input_schema(kind.clone()), "events")
                    .await
                    .unwrap();
                assert_eq!(census.states.len(), 4);
                assert_eq!(census.results.len(), 4);
                for index in 0..2 {
                    assert_eq!(census.states[index].len(), 1);
                    assert_eq!(census.states[index][0].data_type(), &kind);
                    assert_eq!(census.results[index].data_type(), &kind);
                }
                assert_eq!(census.output.fields().len(), 5 + usize::from(grouped));
                assert_eq!(census.keys, if grouped { vec![0] } else { vec![] });
            }
        }
    }
}

#[tokio::test]
async fn test_compact_sync_decimal_native_promotions_match_async_planner() {
    let runtime = configured_runtime(true);
    for (width, precision) in [(32, 9), (64, 18), (128, 38), (256, 76)] {
        for scale in [0, 2, precision] {
            let kind = match width {
                32 => DataType::Decimal32(precision, i8::try_from(scale).unwrap()),
                64 => DataType::Decimal64(precision, i8::try_from(scale).unwrap()),
                128 => DataType::Decimal128(precision, i8::try_from(scale).unwrap()),
                256 => DataType::Decimal256(precision, i8::try_from(scale).unwrap()),
                _ => unreachable!(),
            };
            for grouped in [false, true] {
                let (prefix, suffix) = if grouped {
                    ("key, ", " GROUP BY key")
                } else {
                    ("", "")
                };
                let sql = format!(
                    "SELECT {prefix}SUM(value), COUNT(value), MIN(value), MAX(value), AVG(value) FROM events{suffix}"
                );
                let census = assert_parity(&runtime, &sql, input_schema(kind.clone()), "events")
                    .await
                    .unwrap();
                assert_eq!(census.states.len(), 5);
                assert_eq!(census.results.len(), 5);
                assert!(matches!(
                    census.results[4].data_type(),
                    DataType::Decimal32(..)
                        | DataType::Decimal64(..)
                        | DataType::Decimal128(..)
                        | DataType::Decimal256(..)
                ));
                assert_eq!(census.states[4].len(), 2);
            }
        }
    }
}

#[tokio::test]
async fn test_compact_sync_quoted_qualified_metadata_and_parser_options_match() {
    let runtime = configured_runtime(false);
    let schema = input_schema(DataType::Int16);
    for sql in [
        "SELECT events.key AS \"Key Result\", SUM(events.value) AS \"Sum Result\" FROM events GROUP BY events.key",
        "SELECT key, COUNT(1) FROM public.events GROUP BY key",
        "SELECT key, SUM(value) FROM datafusion.public.events GROUP BY key",
    ] {
        assert_parity(&runtime, sql, schema.clone(), "events")
            .await
            .unwrap();
    }
    assert!(
        assert_parity(
            &runtime,
            "SELECT \"E\".key, SUM(\"E\".value) FROM events AS \"E\" GROUP BY \"E\".key",
            schema.clone(),
            "events",
        )
        .await
        .is_none()
    );
    let state = runtime.context().state_ref();
    {
        let mut state = state.write();
        let parser = &mut state.config_mut().options_mut().sql_parser;
        parser.enable_ident_normalization = false;
        parser.parse_float_as_decimal = true;
        parser.map_string_types_to_utf8view = false;
        parser.enable_options_value_normalization = true;
        parser.collect_spans = true;
        parser.default_null_ordering = "nulls_first".into();
    }
    assert!(
        assert_parity(
            &runtime,
            "SELECT \"events\".value AS \"Value\", 1.25 AS literal FROM \"events\"",
            schema,
            "events",
        )
        .await
        .is_none()
    );
}

#[tokio::test]
async fn test_compact_sync_builtin_scalar_higher_order_window_catalogs_match() {
    let runtime = configured_runtime(true);
    for sql in [
        "SELECT lower(label) AS folded FROM events",
        "SELECT row_number() OVER (ORDER BY value) AS rank_value FROM events",
    ] {
        assert!(
            assert_parity(&runtime, sql, input_schema(DataType::Int64), "events")
                .await
                .is_none()
        );
    }
    let query =
        parse_select_query("SELECT array_transform([1, 2], x -> x + 1) AS transformed FROM events")
            .unwrap();
    assert!(
        runtime
            .context()
            .state()
            .higher_order_functions()
            .is_empty()
    );
    assert!(sync_plan(&runtime, &query, "events", input_schema(DataType::Int64)).is_err());
    assert!(
        runtime
            .incremental_sql_plan(
                &query,
                "events",
                input_schema(DataType::Int64),
                "higher-order-oracle"
            )
            .await
            .is_err()
    );
}

#[tokio::test]
async fn test_compact_sync_catalog_and_builtin_shadowing_are_explicit() {
    let runtime = configured_runtime(true);
    let context = runtime.context();
    let catalog = Arc::new(MemoryCatalogProvider::new());
    catalog
        .register_schema("ticks", Arc::new(MemorySchemaProvider::new()))
        .unwrap();
    context.register_catalog("analysis", catalog);
    {
        let state = context.state_ref();
        let mut state = state.write();
        state.config_mut().options_mut().catalog.default_catalog = "analysis".into();
        state.config_mut().options_mut().catalog.default_schema = "ticks".into();
    }
    let shadow =
        Arc::new(MemTable::try_new(input_schema(DataType::Float64), vec![vec![]]).unwrap());
    context.register_table("events", shadow.clone()).unwrap();
    let query = parse_select_query("SELECT SUM(value) FROM analysis.ticks.events").unwrap();
    let plans = sync_plan(&runtime, &query, "events", input_schema(DataType::Int64)).unwrap();
    assert!(
        native_census(&plans.0, &plans.1, &input_schema(DataType::Int64))
            .unwrap()
            .is_some()
    );
    let retained = context.table_provider("events").await.unwrap();
    let expected: Arc<dyn datafusion::catalog::TableProvider> = shadow;
    assert!(Arc::ptr_eq(&retained, &expected));
    let error = runtime
        .incremental_sql_plan(
            &query,
            "events",
            input_schema(DataType::Int64),
            "shadow-collision",
        )
        .await
        .unwrap_err();
    let CalcFlowError::DataFusion { node_id, message } = error else {
        panic!("expected table registration failure");
    };
    assert_eq!(node_id.as_deref(), Some("shadow-collision"));
    assert!(message.contains("failed to register table alias \"events\""));
    assert!(message.contains("table events already exists"));
    let retained = context.table_provider("events").await.unwrap();
    assert!(Arc::ptr_eq(&retained, &expected));
    let removed = context.deregister_table("events").unwrap().unwrap();
    assert!(Arc::ptr_eq(&removed, &expected));
    assert_parity(
        &runtime,
        query.text(),
        input_schema(DataType::Int64),
        "events",
    )
    .await
    .unwrap();
    context.register_udaf(
        datafusion::functions_aggregate::min_max::min_udaf()
            .as_ref()
            .clone()
            .with_aliases(["sum"]),
    );
    assert!(
        assert_parity(
            &runtime,
            "SELECT SUM(value) FROM events",
            input_schema(DataType::Int64),
            "events"
        )
        .await
        .is_none()
    );
}

#[tokio::test]
async fn test_compact_sync_native_ineligible_plans_do_not_claim_state_fields() {
    let runtime = configured_runtime(true);
    for (kind, sql) in [
        (DataType::Int64, "SELECT AVG(value) FROM events"),
        (DataType::Float64, "SELECT SUM(value) FROM events"),
        (DataType::Int64, "SELECT COUNT(DISTINCT value) FROM events"),
        (DataType::Int64, "SELECT SUM(value + 1) FROM events"),
        (
            DataType::Int64,
            "SELECT SUM(value) FROM events WHERE value > 0",
        ),
        (
            DataType::Decimal128(10, -1),
            "SELECT AVG(value) FROM events",
        ),
    ] {
        assert!(
            assert_parity(&runtime, sql, input_schema(kind), "events")
                .await
                .is_none()
        );
    }
    let schema = input_schema(DataType::Int64);
    let query = parse_select_query("SELECT SUM(value) FROM events").unwrap();
    let (raw, analyzed) = sync_plan(&runtime, &query, "events", schema.clone()).unwrap();
    for value in [
        schema.field(2).clone().with_metadata(HashMap::new()),
        Field::new("value", DataType::Int32, true)
            .with_metadata(schema.field(2).metadata().clone()),
    ] {
        let untrusted = Arc::new(Schema::new(vec![schema.field(0).clone(), value]));
        assert!(
            native_census(&raw, &analyzed, &untrusted)
                .unwrap()
                .is_none()
        );
    }
}

#[test]
fn test_compact_sync_lookup_extensions_and_variables_are_rejected() {
    let runtime = configured_runtime(false);
    for sql in [
        "SELECT SUM(value) FROM other",
        "SELECT SUM(a.value) FROM events a JOIN events b ON a.key = b.key",
        "SELECT events.value FROM events CROSS JOIN custom_table(1)",
        "SELECT @amount FROM events",
        "SELECT nonexistent_function(value) FROM events",
        "SELECT SUM(value) FROM other_catalog.public.events",
    ] {
        let query = parse_select_query(sql).unwrap();
        assert!(
            sync_plan(&runtime, &query, "events", input_schema(DataType::Int64)).is_err(),
            "accepted {sql}"
        );
    }
    assert!(parse_select_query("SELECT * FROM range(0, 2)").is_err());
    let query = parse_select_query("SELECT COUNT(1) FROM events").unwrap();
    let plans = sync_plan(&runtime, &query, "events", input_schema(DataType::Int64)).unwrap();
    let census = native_census(&plans.0, &plans.1, &input_schema(DataType::Int64))
        .unwrap()
        .unwrap();
    assert_eq!(census.states.len(), 1);
    assert_eq!(
        runtime.incremental_sql_plan_calls.load(Ordering::Relaxed),
        0
    );
    assert!(
        !runtime
            .context()
            .state()
            .schema_for_ref("events")
            .unwrap()
            .table_exist("events")
    );
}

#[tokio::test]
async fn test_compact_sync_external_type_and_variable_providers_are_rejected() {
    let runtime = configured_runtime(false);
    let context = runtime.context();
    let state = context.state_ref();
    let replacement = SessionStateBuilder::new_from_existing(context.state())
        .with_type_planner(Arc::new(TypeExtension))
        .build();
    *state.write() = replacement;
    let query = parse_select_query("SELECT CAST(value AS BIGINT) FROM events").unwrap();
    let _plans = runtime
        .incremental_sql_plan(
            &query,
            "events",
            input_schema(DataType::Int64),
            "extension-oracle",
        )
        .await
        .unwrap();
    assert!(sync_plan(&runtime, &query, "events", input_schema(DataType::Int64)).is_err());
    let runtime = configured_runtime(false);
    runtime.context().register_variable(
        datafusion::logical_expr::var_provider::VarType::UserDefined,
        Arc::new(VariableExtension),
    );
    let query = parse_select_query("SELECT @amount FROM events").unwrap();
    let _plans = runtime
        .incremental_sql_plan(
            &query,
            "events",
            input_schema(DataType::Int64),
            "extension-oracle",
        )
        .await
        .unwrap();
    assert!(sync_plan(&runtime, &query, "events", input_schema(DataType::Int64)).is_err());
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
    runtime.context_for_rows(1 << 20, None, "not_evaluated");
    runtime
}

fn input_schema(kind: DataType) -> SchemaRef {
    Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("key", DataType::Int32, true)
                .with_metadata(HashMap::from([("role".into(), "group".into())])),
            Field::new("unused", DataType::Binary, true),
            Field::new("value", kind, true)
                .with_metadata(HashMap::from([("unit".into(), "price".into())])),
            Field::new("label", DataType::Utf8, true),
        ],
        HashMap::from([("logical".into(), "full-input".into())]),
    ))
}

async fn assert_parity(
    runtime: &DataFusionRuntime,
    sql: &str,
    schema: SchemaRef,
    alias: &str,
) -> Option<NativeCensus> {
    let query = parse_select_query(sql).unwrap();
    let (raw, analyzed) = sync_plan(runtime, &query, alias, schema.clone()).unwrap();
    let state = runtime.context().state();
    assert_eq!(state.config().batch_size(), runtime.config.batch_size);
    assert_eq!(
        state.config().target_partitions(),
        runtime.effective_target_partitions.load(Ordering::Acquire)
    );
    let (oracle_raw, oracle_analyzed) = runtime
        .incremental_sql_plan(&query, alias, schema.clone(), "parity")
        .await
        .unwrap();
    assert_eq!(raw, oracle_raw, "raw plan differs for {sql}");
    assert_eq!(analyzed, oracle_analyzed, "analyzed plan differs for {sql}");
    assert_eq!(analyzed.schema(), oracle_analyzed.schema());
    if let Some((_, aggregate)) = aggregate_shape(&raw) {
        let LogicalPlan::TableScan(scan) = aggregate.input.as_ref() else {
            unreachable!();
        };
        assert_eq!(
            scan.source.schema(),
            schema,
            "input metadata differs for {sql}"
        );
    }
    let physical = Arc::new(Schema::new_with_metadata(
        vec![schema.field(0).clone(), schema.field(2).clone()],
        schema.metadata().clone(),
    ));
    let native = native_census(&raw, &analyzed, &physical).unwrap();
    let oracle = native_census(&oracle_raw, &oracle_analyzed, &physical).unwrap();
    assert_eq!(native, oracle, "native fields differ for {sql}");
    let full = native_census(&raw, &analyzed, &schema).unwrap();
    let oracle_full = native_census(&oracle_raw, &oracle_analyzed, &schema).unwrap();
    assert_eq!(full, oracle_full, "full native fields differ for {sql}");
    assert_eq!(
        native, full,
        "trusted projection changed native fields for {sql}"
    );
    native
}

fn sync_plan(
    runtime: &DataFusionRuntime,
    query: &ValidatedQuery,
    alias: &str,
    schema: SchemaRef,
) -> Result<(LogicalPlan, LogicalPlan)> {
    runtime.ensure_open()?;
    if !is_identifier(alias) {
        return Err(registration_error(
            alias,
            Some("compact-parity"),
            "alias must be a SQL identifier",
        ));
    }
    let state = runtime.context_for_rows(0, None, "not_evaluated").state();
    validate_planning_extensions(&state)?;
    let options = state.config_options();
    let declared = declared_lookup(&state, query, alias)?;
    let source = MemTable::try_new(schema.clone(), vec![vec![RecordBatch::new_empty(schema)]])
        .map_err(|error| datafusion_error(Some("compact-parity"), error))?;
    let provider = SingleInput {
        state: &state,
        declared,
        source: provider_as_source(Arc::new(source)),
        lookups: Cell::new(0),
    };
    let raw = SqlToRel::new_with_options(&provider, ParserOptions::from(&options.sql_parser))
        .statement_to_plan(query.statement())
        .map_err(|error| datafusion_error(Some("compact-parity"), error))?;
    let analyzed = state
        .analyzer()
        .execute_and_check(raw.clone(), options, |_, _| {})
        .map_err(|error| datafusion_error(Some("compact-parity"), error))?;
    Ok((raw, analyzed))
}

fn validate_planning_extensions(state: &SessionState) -> Result<()> {
    let mut inspection = SessionStateBuilder::new_from_existing(state.clone());
    if inspection.type_planner().is_some()
        || state
            .execution_props()
            .var_providers
            .as_ref()
            .is_some_and(|providers| !providers.is_empty())
    {
        return Err(datafusion_error(
            Some("compact-parity"),
            DataFusionError::Plan(
                "compact planner does not support external type or variable providers".into(),
            ),
        ));
    }
    Ok(())
}

fn declared_lookup(
    state: &SessionState,
    query: &ValidatedQuery,
    alias: &str,
) -> Result<ResolvedTableReference> {
    let options = state.config_options();
    let declared = TableReference::from(alias).resolve(
        &options.catalog.default_catalog,
        &options.catalog.default_schema,
    );
    let references = state
        .resolve_table_references(&query.statement())
        .map_err(|error| datafusion_error(Some("compact-parity"), error))?;
    if references.len() != 1
        || references[0].clone().resolve(
            &options.catalog.default_catalog,
            &options.catalog.default_schema,
        ) != declared
    {
        return Err(datafusion_error(
            Some("compact-parity"),
            DataFusionError::Plan("compact planner requires one declared input lookup".into()),
        ));
    }
    Ok(declared)
}

struct SingleInput<'a> {
    state: &'a SessionState,
    declared: ResolvedTableReference,
    source: Arc<dyn TableSource>,
    lookups: Cell<usize>,
}

impl ContextProvider for SingleInput<'_> {
    fn get_table_source(&self, name: TableReference) -> DFResult<Arc<dyn TableSource>> {
        let options = self.state.config_options();
        let resolved = name.resolve(
            &options.catalog.default_catalog,
            &options.catalog.default_schema,
        );
        self.lookups.set(self.lookups.get() + 1);
        if self.lookups.get() != 1 || resolved != self.declared {
            return Err(DataFusionError::Plan(
                "compact planner rejected repeated or undeclared lookup".into(),
            ));
        }
        Ok(self.source.clone())
    }

    fn get_table_function_source(
        &self,
        _name: &str,
        _args: Vec<Expr>,
    ) -> DFResult<Arc<dyn TableSource>> {
        Err(DataFusionError::Plan(
            "compact planner does not support table functions".into(),
        ))
    }

    fn get_expr_planners(&self) -> &[Arc<dyn ExprPlanner>] {
        self.state.expr_planners()
    }
    fn get_relation_planners(&self) -> &[Arc<dyn RelationPlanner>] {
        self.state.relation_planners()
    }
    fn get_function_meta(&self, name: &str) -> Option<Arc<ScalarUDF>> {
        self.state.scalar_functions().get(name).cloned()
    }
    fn get_higher_order_meta(&self, name: &str) -> Option<Arc<HigherOrderUDF>> {
        self.state.higher_order_functions().get(name).cloned()
    }
    fn get_aggregate_meta(&self, name: &str) -> Option<Arc<AggregateUDF>> {
        self.state.aggregate_functions().get(name).cloned()
    }
    fn get_window_meta(&self, name: &str) -> Option<Arc<WindowUDF>> {
        self.state.window_functions().get(name).cloned()
    }
    fn get_variable_type(&self, _names: &[String]) -> Option<DataType> {
        None
    }
    fn options(&self) -> &ConfigOptions {
        self.state.config_options()
    }
    fn udf_names(&self) -> Vec<String> {
        self.state.scalar_functions().keys().cloned().collect()
    }
    fn higher_order_function_names(&self) -> Vec<String> {
        self.state
            .higher_order_functions()
            .keys()
            .cloned()
            .collect()
    }
    fn udaf_names(&self) -> Vec<String> {
        self.state.aggregate_functions().keys().cloned().collect()
    }
    fn udwf_names(&self) -> Vec<String> {
        self.state.window_functions().keys().cloned().collect()
    }
}

#[derive(Debug, PartialEq, Eq)]
struct NativeCensus {
    keys: Vec<usize>,
    states: Vec<Vec<FieldRef>>,
    results: Vec<FieldRef>,
    output: SchemaRef,
    projection_types: Vec<(DataType, bool)>,
}

fn aggregate_shape(
    plan: &LogicalPlan,
) -> Option<(
    &datafusion::logical_expr::Projection,
    &datafusion::logical_expr::Aggregate,
)> {
    let LogicalPlan::Projection(projection) = plan else {
        return None;
    };
    let LogicalPlan::Aggregate(aggregate) = projection.input.as_ref() else {
        return None;
    };
    if !matches!(aggregate.input.as_ref(), LogicalPlan::TableScan(_))
        || !projection
            .expr
            .iter()
            .all(|expr| matches!(unalias(expr), Expr::Column(_)))
    {
        return None;
    }
    Some((projection, aggregate))
}

fn unalias(mut expr: &Expr) -> &Expr {
    while let Expr::Alias(alias) = expr {
        expr = &alias.expr;
    }
    expr
}

fn native_argument(expr: &Expr, schema: &SchemaRef) -> bool {
    let Expr::AggregateFunction(function) = unalias(expr) else {
        return false;
    };
    let params = &function.params;
    if !native_parameters_supported(function) {
        return false;
    }
    if function.func.name() == "count"
        && matches!(
            &params.args[0],
            Expr::Literal(ScalarValue::Int64(Some(1)), _)
        )
    {
        return true;
    }
    let Expr::Column(column) = &params.args[0] else {
        return false;
    };
    let Ok(field) = schema.field_with_name(&column.name) else {
        return false;
    };
    match function.func.name() {
        "sum" => exact_numeric(field.data_type()),
        "min" | "max" => {
            exact_numeric(field.data_type())
                || matches!(field.data_type(), DataType::Utf8 | DataType::LargeUtf8)
        }
        "count" => {
            exact_numeric(field.data_type())
                || matches!(
                    field.data_type(),
                    DataType::Boolean | DataType::Utf8 | DataType::LargeUtf8
                )
        }
        "avg" => matches!(
            field.data_type(),
            DataType::Decimal32(_, 0..)
                | DataType::Decimal64(_, 0..)
                | DataType::Decimal128(_, 0..)
                | DataType::Decimal256(_, 0..)
        ),
        _ => false,
    }
}

fn native_parameters_supported(
    function: &datafusion::logical_expr::expr::AggregateFunction,
) -> bool {
    let params = &function.params;
    !(params.distinct
        || params.filter.is_some()
        || !params.order_by.is_empty()
        || params.null_treatment.is_some()
        || params.args.len() != 1
        || !datafusion::functions_aggregate::all_default_aggregate_functions()
            .iter()
            .any(|builtin| builtin.as_ref() == function.func.as_ref()))
}

fn exact_numeric(kind: &DataType) -> bool {
    matches!(
        kind,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Decimal32(..)
            | DataType::Decimal64(..)
            | DataType::Decimal128(..)
            | DataType::Decimal256(..)
    )
}

fn native_keys(
    aggregate: &datafusion::logical_expr::Aggregate,
    schema: &SchemaRef,
) -> Option<Vec<usize>> {
    aggregate
        .group_expr
        .iter()
        .map(|expr| {
            let Expr::Column(column) = expr else {
                return None;
            };
            let ordinal = schema.index_of(&column.name).ok()?;
            matches!(
                schema.field(ordinal).data_type(),
                DataType::Int8
                    | DataType::Int16
                    | DataType::Int32
                    | DataType::Int64
                    | DataType::UInt8
                    | DataType::UInt16
                    | DataType::UInt32
                    | DataType::UInt64
                    | DataType::Boolean
                    | DataType::Utf8
                    | DataType::LargeUtf8
            )
            .then_some(ordinal)
        })
        .collect::<Option<Vec<_>>>()
}

fn native_census(
    raw: &LogicalPlan,
    analyzed: &LogicalPlan,
    schema: &SchemaRef,
) -> DFResult<Option<NativeCensus>> {
    let Some((_, raw_aggregate)) = aggregate_shape(raw) else {
        return Ok(None);
    };
    if !native_aggregates_supported(raw_aggregate, schema) {
        return Ok(None);
    }
    let Some(keys) = native_keys(raw_aggregate, schema) else {
        return Ok(None);
    };
    let Some((projection, aggregate)) = aggregate_shape(analyzed) else {
        return Ok(None);
    };
    let logical = aggregate.input.schema();
    let qualifiers = native_qualifiers(logical, schema);
    let Some(qualifiers) = qualifiers else {
        return Ok(None);
    };
    let rebound = DFSchema::from_field_specific_qualified_schema(qualifiers, schema)?;
    let props = ExecutionProps::new();
    let aggregates = aggregate
        .aggr_expr
        .iter()
        .map(|expr| {
            LoweredAggregateBuilder::new(expr, &rebound, schema, &props)
                .build()
                .map(|lowered| lowered.aggregate)
        })
        .collect::<DFResult<Vec<_>>>()?;
    if !native_accumulators_supported(&aggregates, !keys.is_empty()) {
        return Ok(None);
    }
    let projection_types = native_projection_types(projection, aggregate, &props)?;
    Ok(Some(NativeCensus {
        keys,
        states: aggregates
            .iter()
            .map(|expr| expr.state_fields())
            .collect::<DFResult<Vec<_>>>()?,
        results: aggregates.iter().map(|expr| expr.field()).collect(),
        output: Arc::new(analyzed.schema().as_arrow().clone()),
        projection_types,
    }))
}

fn native_aggregates_supported(
    aggregate: &datafusion::logical_expr::Aggregate,
    schema: &SchemaRef,
) -> bool {
    !aggregate.aggr_expr.is_empty()
        && aggregate
            .aggr_expr
            .iter()
            .all(|expr| native_argument(expr, schema))
}

fn native_qualifiers(
    logical: &DFSchema,
    schema: &SchemaRef,
) -> Option<Vec<Option<TableReference>>> {
    schema
        .fields()
        .iter()
        .map(|field| {
            let index = logical.as_arrow().index_of(field.name()).ok()?;
            let (qualifier, original) = logical.qualified_field(index);
            (original == field).then(|| qualifier.cloned())
        })
        .collect()
}

fn native_accumulators_supported(
    aggregates: &[Arc<datafusion::physical_expr::aggregate::AggregateFunctionExpr>],
    grouped: bool,
) -> bool {
    !(aggregates
        .iter()
        .any(|expr| expr.create_accumulator().is_err())
        || (grouped
            && aggregates.iter().any(|expr| {
                !expr.groups_accumulator_supported() || expr.create_groups_accumulator().is_err()
            })))
}

fn native_projection_types(
    projection: &datafusion::logical_expr::Projection,
    aggregate: &datafusion::logical_expr::Aggregate,
    props: &ExecutionProps,
) -> DFResult<Vec<(DataType, bool)>> {
    projection
        .expr
        .iter()
        .map(|expr| {
            let physical = create_physical_expr(expr, &aggregate.schema, props)?;
            Ok((
                physical.data_type(aggregate.schema.as_arrow())?,
                physical.nullable(aggregate.schema.as_arrow())?,
            ))
        })
        .collect::<DFResult<Vec<_>>>()
}

#[derive(Debug)]
struct TypeExtension;
impl TypePlanner for TypeExtension {
    fn plan_type_field(
        &self,
        _kind: &datafusion::sql::sqlparser::ast::DataType,
    ) -> DFResult<Option<FieldRef>> {
        Ok(Some(Arc::new(
            Field::new("", DataType::Int64, true)
                .with_metadata(HashMap::from([("type-extension".into(), "custom".into())])),
        )))
    }
}

#[derive(Debug)]
struct VariableExtension;
impl datafusion::logical_expr::var_provider::VarProvider for VariableExtension {
    fn get_type(&self, _names: &[String]) -> Option<DataType> {
        Some(DataType::Int64)
    }
    fn get_value(&self, _names: Vec<String>) -> DFResult<ScalarValue> {
        Ok(ScalarValue::Int64(Some(5)))
    }
}
