//! Reusable physical plans for repeated single-table row-local queries.
//!
//! Stream operators run the same read-only query over every batch. When the
//! planned physical tree contains only data-independent row-local nodes over
//! one in-memory scan, and every function is immutable, the plan is kept as a
//! template. Later batches with the same query text, alias, and exact schema
//! bind their rows into a fresh copy of the tree instead of parsing,
//! registering, and planning again. Rebuilding every ancestor of the scan also
//! gives each run fresh execution metrics.

use std::{collections::BTreeMap, sync::Arc};

use datafusion::{
    arrow::{datatypes::SchemaRef, record_batch::RecordBatch},
    common::tree_node::{Transformed, TreeNode, TreeNodeRecursion},
    datasource::{
        memory::{DataSourceExec, MemorySourceConfig},
        source::DataSource,
    },
    error::{DataFusionError, Result as DataFusionResult},
    logical_expr::{Expr, LogicalPlan, Volatility},
    physical_plan::{ExecutionPlan, filter::FilterExec, projection::ProjectionExec},
};
use parking_lot::Mutex;

use crate::Batch;

/// Bounded so a runtime planning many distinct queries keeps constant memory.
const MAX_TEMPLATES: usize = 8;

/// One cached single-table physical plan and its diagnostic plan text.
pub(crate) struct QueryTemplate {
    query: String,
    alias: String,
    schema: SchemaRef,
    plan: Arc<dyn ExecutionPlan>,
    pub(crate) logical_plan: String,
    pub(crate) physical_plan_text: String,
}

impl QueryTemplate {
    /// Returns a fresh plan tree reading `batches` in the template's scan.
    pub(crate) fn bind(&self, batches: &[RecordBatch]) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        Arc::clone(&self.plan)
            .transform_up(|node| rebind_scan(node, batches))
            .map(|transformed| transformed.data)
    }

    fn matches(&self, query: &str, alias: &str, schema: &SchemaRef) -> bool {
        self.query == query && self.alias == alias && self.schema == *schema
    }
}

/// Planned query parts from which a template may be captured.
pub(crate) struct TemplateCandidate<'a> {
    pub(crate) query: &'a str,
    pub(crate) tables: &'a BTreeMap<String, Batch>,
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
    /// Finds a template for a single-table query and returns its table rows.
    pub(crate) fn find<'a>(
        &self,
        query: &str,
        tables: &'a BTreeMap<String, Batch>,
    ) -> Option<(Arc<QueryTemplate>, &'a [RecordBatch])> {
        let (alias, batch) = single_table(tables)?;
        let table = batch.table_payload().ok()?;
        self.entries
            .lock()
            .iter()
            .find(|template| template.matches(query, alias, table.schema()))
            .map(|template| (Arc::clone(template), table.batches()))
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

fn single_table(tables: &BTreeMap<String, Batch>) -> Option<(&str, &Batch)> {
    let mut entries = tables.iter();
    let (alias, batch) = entries.next()?;
    entries.next().is_none().then_some((alias.as_str(), batch))
}

fn template_for(candidate: &TemplateCandidate<'_>) -> Option<QueryTemplate> {
    let (alias, batch) = single_table(candidate.tables)?;
    let schema = Arc::clone(batch.table_payload().ok()?.schema());
    let eligible = only_immutable_functions(candidate.logical)
        && reusable_tree(candidate.physical.as_ref(), &schema);
    eligible.then(|| QueryTemplate {
        query: candidate.query.to_owned(),
        alias: alias.to_owned(),
        schema,
        plan: Arc::clone(candidate.physical),
        logical_plan: candidate.logical_plan.to_owned(),
        physical_plan_text: candidate.physical_plan_text.to_owned(),
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

fn mutable_function(expr: &Expr) -> bool {
    match expr {
        Expr::ScalarFunction(function) => {
            function.func.signature().volatility != Volatility::Immutable
        }
        Expr::AggregateFunction(_) | Expr::WindowFunction(_) => true,
        _ => false,
    }
}

/// Accepts only data-independent row-local operators over exactly one
/// in-memory scan of the registered table schema.
fn reusable_tree(plan: &dyn ExecutionPlan, schema: &SchemaRef) -> bool {
    if let Some(scan) = memory_scan(plan) {
        return scan.original_schema() == *schema && scan.partitions().len() == 1;
    }
    let row_local = plan.downcast_ref::<ProjectionExec>().is_some()
        || plan.downcast_ref::<FilterExec>().is_some();
    let children = plan.children();
    row_local && children.len() == 1 && reusable_tree(children[0].as_ref(), schema)
}

fn memory_scan(plan: &dyn ExecutionPlan) -> Option<&MemorySourceConfig> {
    plan.downcast_ref::<DataSourceExec>()?
        .data_source()
        .downcast_ref::<MemorySourceConfig>()
}

fn rebind_scan(
    node: Arc<dyn ExecutionPlan>,
    batches: &[RecordBatch],
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
        &[batches.to_vec()],
        scan.original_schema(),
        scan.projection().clone(),
    )?
    .with_show_sizes(scan.show_sizes())
    .with_limit(scan.fetch());
    Ok(Transformed::yes(DataSourceExec::from_data_source(source)))
}
