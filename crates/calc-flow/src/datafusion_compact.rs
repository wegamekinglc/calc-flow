use std::{cell::Cell, collections::HashMap, mem::size_of, sync::Arc};

use datafusion::{
    arrow::{
        datatypes::{DataType, SchemaRef},
        record_batch::RecordBatch,
    },
    common::{DFSchema, ResolvedTableReference, TableReference, config::ConfigOptions},
    datasource::{MemTable, provider_as_source},
    error::{DataFusionError, Result as DFResult},
    execution::{SessionState, memory_pool::MemoryReservation, session_state::SessionStateBuilder},
    logical_expr::{
        AggregateUDF, Expr, HigherOrderUDF, LogicalPlan, ScalarUDF, TableSource, WindowUDF,
        planner::{ContextProvider, ExprPlanner, RelationPlanner, TypePlanner},
        simplify::SimplifyContext,
    },
    optimizer::simplify_expressions::ExprSimplifier,
    sql::planner::{ParserOptions, SqlToRel},
};

use super::{DataFusionRuntime, datafusion_error, is_identifier, registration_error};
use crate::{Result, expression::ValidatedQuery};

pub(crate) struct PaidSqlPlan {
    pub(crate) raw: LogicalPlan,
    pub(crate) analyzed: LogicalPlan,
    _reservation: MemoryReservation,
}

impl PaidSqlPlan {
    #[cfg(test)]
    pub(crate) fn reserved_bytes(&self) -> usize {
        let Self {
            _reservation: reservation,
            ..
        } = self;
        reservation.size()
    }
}

pub(crate) fn plan(
    runtime: &DataFusionRuntime,
    query: &ValidatedQuery,
    alias: &str,
    schema: SchemaRef,
    node_id: &str,
    reservation: MemoryReservation,
) -> Result<PaidSqlPlan> {
    plan_with_policy(runtime, query, alias, schema, node_id, reservation, true)
}

pub(crate) fn plan_retained(
    runtime: &DataFusionRuntime,
    query: &ValidatedQuery,
    alias: &str,
    schema: SchemaRef,
    node_id: &str,
    reservation: MemoryReservation,
) -> Result<PaidSqlPlan> {
    plan_with_policy(runtime, query, alias, schema, node_id, reservation, false)
}

fn plan_with_policy(
    runtime: &DataFusionRuntime,
    query: &ValidatedQuery,
    alias: &str,
    schema: SchemaRef,
    node_id: &str,
    reservation: MemoryReservation,
    native: bool,
) -> Result<PaidSqlPlan> {
    runtime.ensure_open()?;
    if !is_identifier(alias) {
        return Err(registration_error(
            alias,
            Some(node_id),
            "alias must be a SQL identifier",
        ));
    }
    let context = runtime.context_for_rows(0, None, "not_evaluated");
    let state_ref = context.state_ref();
    let state = state_ref.read();
    reserve_plan_registry(&state, &reservation, node_id, native)?;
    let options = state.config_options();
    let declared = TableReference::from(alias).resolve(
        &options.catalog.default_catalog,
        &options.catalog.default_schema,
    );
    let references = state
        .resolve_table_references(&query.statement())
        .map_err(|error| datafusion_error(Some(node_id), error))?;
    validate_declared_input(&references, &declared, options, node_id, native)?;
    let source = MemTable::try_new(schema.clone(), vec![vec![RecordBatch::new_empty(schema)]])
        .map_err(|error| datafusion_error(Some(node_id), error))?;
    let type_planner = planning_type_planner(&state, native);
    let provider = SingleInput {
        state: &state,
        declared,
        source: provider_as_source(Arc::new(source)),
        lookups: Cell::new(0),
        native,
        type_planner,
    };
    let (raw, analyzed) = analyze_query(&state, &provider, options, query, node_id)?;
    Ok(PaidSqlPlan {
        raw,
        analyzed,
        _reservation: reservation,
    })
}

fn reserve_plan_registry(
    state: &SessionState,
    reservation: &MemoryReservation,
    node_id: &str,
    native: bool,
) -> Result<()> {
    let bytes = registry_charge(state, node_id)?;
    reservation
        .try_grow(bytes)
        .map_err(|error| datafusion_error(Some(node_id), error))?;
    if native {
        reject_extensions(state, node_id)?;
    }
    Ok(())
}

fn validate_declared_input(
    references: &[TableReference],
    declared: &ResolvedTableReference,
    options: &ConfigOptions,
    node_id: &str,
    native: bool,
) -> Result<()> {
    if (native && references.len() != 1)
        || references.iter().any(|reference| {
            reference.clone().resolve(
                &options.catalog.default_catalog,
                &options.catalog.default_schema,
            ) != *declared
        })
    {
        return Err(datafusion_error(
            Some(node_id),
            DataFusionError::Plan("compact planner requires one declared input lookup".into()),
        ));
    }
    Ok(())
}

fn planning_type_planner(state: &SessionState, native: bool) -> Option<Arc<dyn TypePlanner>> {
    if native {
        None
    } else {
        let mut view = SessionStateBuilder::new_from_existing(state.clone());
        view.type_planner().clone()
    }
}

fn analyze_query(
    state: &SessionState,
    provider: &SingleInput<'_>,
    options: &ConfigOptions,
    query: &ValidatedQuery,
    node_id: &str,
) -> Result<(LogicalPlan, LogicalPlan)> {
    let raw = SqlToRel::new_with_options(provider, ParserOptions::from(&options.sql_parser))
        .statement_to_plan(query.statement())
        .map_err(|error| datafusion_error(Some(node_id), error))?;
    let analyzed = state
        .analyzer()
        .execute_and_check(raw.clone(), options, |_, _| {})
        .map_err(|error| datafusion_error(Some(node_id), error))?;
    Ok((raw, analyzed))
}

fn reject_extensions(state: &SessionState, node_id: &str) -> Result<()> {
    let mut inspection = SessionStateBuilder::new_from_existing(state.clone());
    if inspection.type_planner().is_some()
        || state
            .execution_props()
            .var_providers
            .as_ref()
            .is_some_and(|providers| !providers.is_empty())
    {
        return Err(datafusion_error(
            Some(node_id),
            DataFusionError::Plan(
                "compact planner does not support external type or variable providers".into(),
            ),
        ));
    }
    Ok(())
}

fn registry_charge(state: &SessionState, node_id: &str) -> Result<usize> {
    let mut bytes = 0;
    for charge in [
        catalog_charge(state.scalar_functions(), node_id)?,
        catalog_charge(state.aggregate_functions(), node_id)?,
        catalog_charge(state.higher_order_functions(), node_id)?,
        catalog_charge(state.window_functions(), node_id)?,
        catalog_charge(state.table_functions(), node_id)?,
    ] {
        bytes = checked_charge(bytes, charge, 1, node_id)?;
    }
    [state.expr_planners().len(), state.relation_planners().len()]
        .into_iter()
        .try_fold(bytes, |bytes, count| {
            checked_charge(bytes, count, 2 * size_of::<Arc<dyn ExprPlanner>>(), node_id)
        })
}

fn catalog_charge<T>(catalog: &HashMap<String, Arc<T>>, node_id: &str) -> Result<usize> {
    let backing = checked_charge(
        0,
        catalog.capacity(),
        2 * (size_of::<(String, Arc<T>)>() + 1),
        node_id,
    )?;
    catalog.keys().try_fold(backing, |bytes, name| {
        checked_charge(bytes, name.capacity(), 1, node_id)
    })
}

fn checked_charge(bytes: usize, count: usize, width: usize, node_id: &str) -> Result<usize> {
    count
        .checked_mul(width)
        .and_then(|charge| bytes.checked_add(charge))
        .ok_or_else(|| {
            datafusion_error(
                Some(node_id),
                DataFusionError::Plan("compact planner temporary charge overflowed".into()),
            )
        })
}

struct SingleInput<'a> {
    state: &'a SessionState,
    declared: ResolvedTableReference,
    source: Arc<dyn TableSource>,
    lookups: Cell<usize>,
    native: bool,
    type_planner: Option<Arc<dyn TypePlanner>>,
}

impl ContextProvider for SingleInput<'_> {
    fn get_table_source(&self, name: TableReference) -> DFResult<Arc<dyn TableSource>> {
        let options = self.state.config_options();
        let resolved = name.resolve(
            &options.catalog.default_catalog,
            &options.catalog.default_schema,
        );
        self.lookups.set(self.lookups.get() + 1);
        if (self.native && self.lookups.get() != 1) || resolved != self.declared {
            return Err(DataFusionError::Plan(
                "compact planner rejected repeated or undeclared lookup".into(),
            ));
        }
        Ok(self.source.clone())
    }

    fn get_table_function_source(
        &self,
        name: &str,
        args: Vec<Expr>,
    ) -> DFResult<Arc<dyn TableSource>> {
        if self.native {
            return Err(DataFusionError::Plan(
                "compact planner does not support table functions".into(),
            ));
        }
        let function =
            self.state.table_functions().get(name).ok_or_else(|| {
                DataFusionError::Plan(format!("table function '{name}' not found"))
            })?;
        let context = SimplifyContext::builder()
            .with_config_options(Arc::clone(self.state.config_options()))
            .with_query_execution_start_time(
                self.state.execution_props().query_execution_start_time,
            )
            .build();
        let simplifier = ExprSimplifier::new(context);
        let schema = DFSchema::empty();
        let args = args
            .into_iter()
            .map(|arg| {
                simplifier
                    .coerce(arg, &schema)
                    .and_then(|arg| simplifier.simplify(arg))
            })
            .collect::<DFResult<Vec<_>>>()?;
        function
            .create_table_provider_with_args(datafusion::catalog::TableFunctionArgs::new(
                &args, self.state,
            ))
            .map(provider_as_source)
    }

    fn get_type_planner(&self) -> Option<Arc<dyn TypePlanner>> {
        self.type_planner.clone()
    }

    fn create_cte_work_table(
        &self,
        name: &str,
        schema: SchemaRef,
    ) -> DFResult<Arc<dyn TableSource>> {
        if self.native {
            return Err(DataFusionError::Plan(
                "compact planner does not support recursive CTEs".into(),
            ));
        }
        Ok(provider_as_source(Arc::new(
            datafusion::datasource::cte_worktable::CteWorkTable::new(name, schema),
        )))
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
    fn get_variable_type(&self, names: &[String]) -> Option<DataType> {
        use datafusion::logical_expr::var_provider::{VarType, is_system_variables};
        if self.native || names.is_empty() {
            return None;
        }
        let kind = if is_system_variables(names) {
            VarType::System
        } else {
            VarType::UserDefined
        };
        self.state
            .execution_props()
            .var_providers
            .as_ref()
            .and_then(|providers| providers.get(&kind)?.get_type(names))
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
