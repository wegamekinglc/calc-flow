//! Reusable physical plans for repeated read-only queries.
//!
//! Stream operators and repeated batch runs execute the same read-only query
//! over every batch. When the planned physical tree contains only
//! data-independent operators over in-memory scans of the registered tables,
//! and every function is immutable, the plan is kept as a template. Later
//! runs with the same query text, aliases, exact schemas, and partition count
//! bind their rows into the cached tree instead of registering tables and
//! planning again. Finite SQL queries bind rows into a fresh plan tree with
//! fresh execution metrics. Eligible single-table stream expressions and
//! row-local stream SQL compile the template into immutable steps with
//! filtering state owned by each evaluation.

use std::{
    collections::BTreeMap,
    sync::{Arc, OnceLock},
};

use datafusion::{
    arrow::{datatypes::SchemaRef, record_batch::RecordBatch},
    common::tree_node::{Transformed, TreeNode, TreeNodeRecursion},
    datasource::{
        memory::{DataSourceExec, MemorySourceConfig},
        source::DataSource,
    },
    error::{DataFusionError, Result as DataFusionResult},
    logical_expr::{Expr, LogicalPlan, Volatility},
    physical_plan::{
        ExecutionPlan,
        aggregates::AggregateExec,
        coalesce_partitions::CoalescePartitionsExec,
        filter::FilterExec,
        limit::{GlobalLimitExec, LocalLimitExec},
        projection::ProjectionExec,
        repartition::RepartitionExec,
        sorts::{sort::SortExec, sort_preserving_merge::SortPreservingMergeExec},
        union::{InterleaveExec, UnionExec},
    },
};
use parking_lot::Mutex;

use crate::{Batch, datafusion::split_record_batches};

mod compiled;
pub(crate) use compiled::CompiledExpression;
#[cfg(test)]
pub(crate) use compiled::check_arrow_error_envelope;

/// Bounded so a runtime planning many distinct queries keeps constant memory.
const MAX_TEMPLATES: usize = 8;

/// One cached physical plan, its input identity, and diagnostic plan text.
pub(crate) struct QueryTemplate {
    query: String,
    /// Registered aliases with their exact schemas, in alias order.
    inputs: Vec<(String, SchemaRef)>,
    partitions: usize,
    /// The alias and partition count of each in-memory scan, in plan
    /// traversal order.
    scans: Vec<(String, usize)>,
    plan: Arc<dyn ExecutionPlan>,
    pub(crate) logical_plan: String,
    pub(crate) physical_plan_text: String,
    compiled: OnceLock<Option<Arc<CompiledExpression>>>,
}

impl QueryTemplate {
    /// Compiled row-local steps for a single-table template, when eligible.
    pub(crate) fn compiled_expression(&self) -> Option<Arc<CompiledExpression>> {
        if self.inputs.len() != 1 {
            return None;
        }
        self.compiled
            .get_or_init(|| CompiledExpression::try_new(&self.plan).map(Arc::new))
            .clone()
    }

    /// Returns a fresh plan tree whose scans read each alias's current rows.
    pub(crate) fn bind(
        &self,
        tables: &BTreeMap<String, Batch>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let mut scans = self.scans.iter();
        Arc::clone(&self.plan)
            .transform_up(|node| {
                if memory_scan(node.as_ref()).is_none() {
                    return Ok(Transformed::no(node));
                }
                let scan = scans.next().ok_or_else(|| {
                    DataFusionError::Internal("reusable query template scan count changed".into())
                })?;
                let table = tables
                    .get(scan.0.as_str())
                    .and_then(|batch| batch.table_payload().ok())
                    .ok_or_else(|| {
                        DataFusionError::Internal(format!(
                            "reusable query template input {} is not a registered table",
                            scan.0
                        ))
                    })?;
                rebind_scan(node, table.batches(), scan.1)
            })
            .map(|transformed| transformed.data)
    }

    fn matches(&self, query: &str, tables: &BTreeMap<String, Batch>, partitions: usize) -> bool {
        self.query == query
            && self.partitions == partitions
            && self.inputs.len() == tables.len()
            && self
                .inputs
                .iter()
                .zip(tables)
                .all(|((alias, schema), (name, batch))| {
                    alias == name
                        && batch
                            .table_payload()
                            .is_ok_and(|table| table.schema() == schema)
                })
    }
}

/// Planned query parts from which a template may be captured.
pub(crate) struct TemplateCandidate<'a> {
    pub(crate) query: &'a str,
    pub(crate) tables: &'a BTreeMap<String, Batch>,
    pub(crate) partitions: usize,
    pub(crate) logical: &'a LogicalPlan,
    pub(crate) physical: &'a Arc<dyn ExecutionPlan>,
    pub(crate) logical_plan: &'a str,
    pub(crate) physical_plan_text: &'a str,
}

/// Runtime-owned, first-in first-out bounded template cache.
#[derive(Default)]
pub(crate) struct QueryTemplates {
    entries: Mutex<Vec<Arc<QueryTemplate>>>,
}

impl QueryTemplates {
    /// Finds the template planned for these aliases, schemas, and partitions.
    pub(crate) fn find(
        &self,
        query: &str,
        tables: &BTreeMap<String, Batch>,
        partitions: usize,
    ) -> Option<Arc<QueryTemplate>> {
        self.entries
            .lock()
            .iter()
            .find(|template| template.matches(query, tables, partitions))
            .cloned()
    }

    /// Captures an eligible plan; ineligible plans are ignored.
    pub(crate) fn capture(&self, candidate: &TemplateCandidate<'_>) {
        let Some(template) = template_for(candidate) else {
            return;
        };
        let mut entries = self.entries.lock();
        if entries.len() >= MAX_TEMPLATES {
            entries.remove(0);
        }
        entries.push(Arc::new(template));
    }
}

#[cfg(test)]
impl QueryTemplates {
    /// Rows held by cached scans; templates never retain input rows.
    pub(crate) fn retained_rows(&self) -> usize {
        self.entries
            .lock()
            .iter()
            .filter_map(|template| {
                let mut rows = 0;
                Arc::clone(&template.plan)
                    .apply(|node| {
                        if let Some(scan) = memory_scan(node.as_ref()) {
                            rows += scan
                                .partitions()
                                .iter()
                                .flatten()
                                .map(RecordBatch::num_rows)
                                .sum::<usize>();
                        }
                        Ok(TreeNodeRecursion::Continue)
                    })
                    .ok()
                    .map(|_| rows)
            })
            .sum()
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.lock().len()
    }
}

fn template_for(candidate: &TemplateCandidate<'_>) -> Option<QueryTemplate> {
    let inputs = candidate
        .tables
        .iter()
        .map(|(alias, batch)| {
            let schema = Arc::clone(batch.table_payload().ok()?.schema());
            Some((alias.clone(), schema))
        })
        .collect::<Option<Vec<_>>>()?;
    let eligible =
        only_immutable_functions(candidate.logical) && reusable_tree(candidate.physical.as_ref());
    if !eligible {
        return None;
    }
    let scans = scan_bindings(candidate.physical, candidate.tables)?;
    // Empty scans keep the cached tree from retaining the first input rows.
    let plan = Arc::clone(candidate.physical)
        .transform_up(|node| {
            let partitions =
                memory_scan(node.as_ref()).map_or(1, |scan| scan.partitions().len().max(1));
            rebind_scan(node, &[], partitions)
        })
        .ok()?
        .data;
    Some(QueryTemplate {
        query: candidate.query.to_owned(),
        inputs,
        partitions: candidate.partitions,
        scans,
        plan,
        logical_plan: candidate.logical_plan.to_owned(),
        physical_plan_text: candidate.physical_plan_text.to_owned(),
        compiled: OnceLock::new(),
    })
}

/// Stable and volatile functions may be folded against planning-time state,
/// so a plan containing one is never reused.
fn only_immutable_functions(plan: &LogicalPlan) -> bool {
    let mut immutable = true;
    let _ = plan.apply_with_subqueries(|node| {
        node.apply_expressions(|expression| {
            expression.apply(|expr| {
                if mutable_function(expr) {
                    immutable = false;
                    return Ok(TreeNodeRecursion::Stop);
                }
                Ok(TreeNodeRecursion::Continue)
            })
        })
    });
    immutable
}

/// Registered scalar UDFs are `Expr::ScalarFunction` and aggregates are
/// `Expr::AggregateFunction` in `DataFusion` 55. Any expression kind outside
/// the known data-independent set is treated as mutable.
fn mutable_function(expr: &Expr) -> bool {
    match expr {
        Expr::ScalarFunction(function) => {
            function.func.signature().volatility != Volatility::Immutable
        }
        Expr::AggregateFunction(function) => {
            function.func.signature().volatility != Volatility::Immutable
        }
        _ => !row_local_expression(expr),
    }
}

fn row_local_expression(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::Alias(_)
            | Expr::Column(_)
            | Expr::Literal(..)
            | Expr::BinaryExpr(_)
            | Expr::Not(_)
            | Expr::IsNotNull(_)
            | Expr::IsNull(_)
            | Expr::IsTrue(_)
            | Expr::IsFalse(_)
            | Expr::IsUnknown(_)
            | Expr::IsNotTrue(_)
            | Expr::IsNotFalse(_)
            | Expr::IsNotUnknown(_)
            | Expr::Negative(_)
            | Expr::Between(_)
            | Expr::Case(_)
            | Expr::Cast(_)
            | Expr::TryCast(_)
            | Expr::InList(_)
            | Expr::Like(_)
            | Expr::SimilarTo(_)
    )
}

/// Accepts only data-independent operators whose leaves are in-memory scans
/// without scan orderings.
fn reusable_tree(plan: &dyn ExecutionPlan) -> bool {
    if let Some(scan) = memory_scan(plan) {
        return scan.sort_information().is_empty();
    }
    let children = plan.children();
    data_independent_operator(plan)
        && !children.is_empty()
        && children.iter().all(|child| reusable_tree(child.as_ref()))
}

/// Operators whose planned state depends only on the query and session
/// configuration, never on the rows registered when the plan was captured.
/// Joins stay excluded because their build side is chosen from statistics.
fn data_independent_operator(plan: &dyn ExecutionPlan) -> bool {
    plan.downcast_ref::<ProjectionExec>().is_some()
        || plan.downcast_ref::<FilterExec>().is_some()
        || plan.downcast_ref::<AggregateExec>().is_some()
        || plan.downcast_ref::<UnionExec>().is_some()
        || plan.downcast_ref::<InterleaveExec>().is_some()
        || plan.downcast_ref::<CoalescePartitionsExec>().is_some()
        || plan.downcast_ref::<RepartitionExec>().is_some()
        || plan.downcast_ref::<SortExec>().is_some()
        || plan.downcast_ref::<SortPreservingMergeExec>().is_some()
        || plan.downcast_ref::<GlobalLimitExec>().is_some()
        || plan.downcast_ref::<LocalLimitExec>().is_some()
}

/// Attributes each in-memory scan, in plan traversal order, to the registered
/// alias whose rows it reads, with the scan's partition count. A scan that
/// does not read exactly one alias's rows makes the plan ineligible.
fn scan_bindings(
    plan: &Arc<dyn ExecutionPlan>,
    tables: &BTreeMap<String, Batch>,
) -> Option<Vec<(String, usize)>> {
    let mut bindings = Vec::new();
    let mut attributable = true;
    let _ = plan.apply(|node| {
        let Some(scan) = memory_scan(node.as_ref()) else {
            return Ok(TreeNodeRecursion::Continue);
        };
        let Some(alias) = scan_alias(scan, tables) else {
            attributable = false;
            return Ok(TreeNodeRecursion::Stop);
        };
        bindings.push((alias, scan.partitions().len().max(1)));
        Ok(TreeNodeRecursion::Continue)
    });
    attributable.then_some(bindings)
}

/// Registered rows reach the scan as shared Arrow arrays, so the first column
/// of every batch identifies the alias without comparing values. All of a
/// scan's batches must be the registered batches of exactly one alias, in the
/// same order.
fn scan_alias(scan: &MemorySourceConfig, tables: &BTreeMap<String, Batch>) -> Option<String> {
    let scan_columns = scan
        .partitions()
        .iter()
        .flatten()
        .map(|record| record.columns().first())
        .collect::<Vec<_>>();
    if scan_columns.is_empty() || scan_columns.iter().any(Option::is_none) {
        return None;
    }
    let mut matches = tables.iter().filter(|(_, batch)| {
        batch.table_payload().is_ok_and(|table| {
            *table.schema() == scan.original_schema()
                && table.batches().len() == scan_columns.len()
                && table
                    .batches()
                    .iter()
                    .zip(scan_columns.iter())
                    .all(|(record, column)| {
                        record.columns().first().is_some_and(|candidate| {
                            Arc::ptr_eq(candidate, column.expect("checked"))
                        })
                    })
        })
    });
    let (alias, _) = matches.next()?;
    matches.next().is_none().then(|| alias.clone())
}

fn memory_scan(plan: &dyn ExecutionPlan) -> Option<&MemorySourceConfig> {
    plan.downcast_ref::<DataSourceExec>()?
        .data_source()
        .downcast_ref::<MemorySourceConfig>()
}

fn rebind_scan(
    node: Arc<dyn ExecutionPlan>,
    batches: &[RecordBatch],
    partitions: usize,
) -> DataFusionResult<Transformed<Arc<dyn ExecutionPlan>>> {
    let Some(scan) = memory_scan(node.as_ref()) else {
        return Ok(Transformed::no(node));
    };
    if !scan.sort_information().is_empty() {
        return Err(DataFusionError::Internal(
            "reusable query templates never carry scan orderings".into(),
        ));
    }
    let source = MemorySourceConfig::try_new(
        &split_record_batches(batches, partitions.max(1)),
        scan.original_schema(),
        scan.projection().clone(),
    )?
    .with_show_sizes(scan.show_sizes())
    .with_limit(scan.fetch());
    Ok(Transformed::yes(DataSourceExec::from_data_source(source)))
}
