use std::{collections::BTreeSet, ops::ControlFlow, sync::Arc};

use datafusion::{
    arrow::{
        datatypes::SchemaRef,
        ipc::{reader::FileReader, writer::FileWriter},
    },
    common::tree_node::{TreeNode, TreeNodeRecursion},
    execution::memory_pool::MemoryReservation,
    logical_expr::{Expr as LogicalExpr, LogicalPlan},
    sql::{
        parser::Statement as DFStatement,
        sqlparser::ast::{
            Expr, Function, FunctionArg, FunctionArgExpr, FunctionArguments, GroupByExpr, Ident,
            Query, Select, SelectItem, SetExpr, Statement, TableFactor, Visit, Visitor,
        },
    },
};
use sha2::{Digest, Sha256};

use super::{StateSegment, sql_state_error};
use crate::operator::retained_columns::RetainedColumns;
use crate::{DataFusionRuntime, Result, expression::ValidatedQuery};

pub(super) struct SqlProjection {
    pub columns: RetainedColumns,
    pub logical_segment: StateSegment,
    _reservation: Arc<MemoryReservation>,
}

impl SqlProjection {
    pub(super) fn resolve(
        runtime: &DataFusionRuntime,
        query: &ValidatedQuery,
        alias: &str,
        schema: SchemaRef,
        name: &str,
    ) -> Result<Option<Arc<Self>>> {
        let reservation = reserve_descriptor(runtime, query, &schema, name)?;
        let Some(ordinals) = required_columns(query, alias, &schema) else {
            return Ok(None);
        };
        if ordinals.len() == schema.fields().len() {
            return Ok(None);
        }
        let logical_segment = encode_schema(&schema)?;
        if logical_segment.bytes().len() > reservation.size() / 2 {
            return Err(sql_state_error(
                "logical schema exceeded descriptor reservation",
            ));
        }
        let digest = dependency_digest(query, alias, logical_segment.bytes(), &ordinals)?;
        let columns = RetainedColumns::try_new(schema, &ordinals, digest)?;
        let reservation = Arc::new(reservation);
        Ok(Some(Arc::new(Self {
            columns,
            logical_segment: logical_segment.with_owner(reservation.clone()),
            _reservation: reservation,
        })))
    }

    pub(super) async fn prepare_plan(
        &self,
        runtime: &DataFusionRuntime,
        query: &ValidatedQuery,
        alias: &str,
        name: &str,
    ) -> Result<Option<(LogicalPlan, LogicalPlan)>> {
        let (raw, analyzed) = runtime
            .incremental_sql_plan(
                query,
                alias,
                Arc::clone(self.columns.logical_schema()),
                name,
            )
            .await?;
        Ok((plan_dependencies_fit(&raw, &self.columns)?
            && plan_dependencies_fit(&analyzed, &self.columns)?)
        .then_some((raw, analyzed)))
    }
}

fn reserve_descriptor(
    runtime: &DataFusionRuntime,
    query: &ValidatedQuery,
    schema: &SchemaRef,
    name: &str,
) -> Result<MemoryReservation> {
    let fields = super::ipc::schema_bytes(schema)?;
    let charge = super::incremental::checked_bytes(
        8192,
        [
            (query.text().len(), 16),
            (fields, 16),
            (schema.fields().len(), 512),
        ],
        name,
    )?;
    let reservation = runtime.incremental_reservation(name);
    super::incremental::ensure_reservation(&reservation, charge, name)?;
    Ok(reservation)
}

pub(super) fn encode_schema(schema: &SchemaRef) -> Result<StateSegment> {
    let mut bytes = Vec::new();
    let mut writer = FileWriter::try_new(&mut bytes, schema)
        .map_err(|error| sql_state_error(&format!("logical schema IPC failed: {error}")))?;
    writer
        .finish()
        .map_err(|error| sql_state_error(&error.to_string()))?;
    drop(writer);
    Ok(StateSegment::new(bytes))
}

pub(super) fn decode_schema(segment: &StateSegment) -> Result<SchemaRef> {
    let mut reader = FileReader::try_new(std::io::Cursor::new(segment.bytes()), None)
        .map_err(|error| sql_state_error(&format!("logical schema IPC is invalid: {error}")))?;
    let schema = reader.schema();
    if reader.next().is_some() {
        return Err(sql_state_error(
            "logical schema segment must contain no records",
        ));
    }
    Ok(schema)
}

pub(super) fn schema_digest(schema: &SchemaRef) -> Result<String> {
    Ok(hex::encode(Sha256::digest(encode_schema(schema)?.bytes())))
}

fn dependency_digest(
    query: &ValidatedQuery,
    alias: &str,
    logical: &[u8],
    ordinals: &[usize],
) -> Result<[u8; 32]> {
    let identity = serde_json::json!({
        "query": query.text(), "alias": alias,
        "logical_schema_sha256": hex::encode(Sha256::digest(logical)),
        "retained_ordinals": ordinals, "udfs": [],
    });
    Ok(Sha256::digest(crate::canonical_json(&identity)?.as_bytes()).into())
}

fn plan_dependencies_fit(plan: &LogicalPlan, columns: &RetainedColumns) -> Result<bool> {
    let mut fits = true;
    plan.apply_with_subqueries(|node| {
        if let LogicalPlan::TableScan(scan) = node {
            if scan.source.schema() != *columns.logical_schema() {
                fits = false;
            }
        }
        node.apply_expressions(|expression| {
            expression.apply(|expr| {
                if let LogicalExpr::Column(column) = expr {
                    if let Ok(index) = columns.logical_schema().index_of(&column.name) {
                        fits &= columns.retained_index(index).is_ok();
                    }
                }
                Ok(TreeNodeRecursion::Continue)
            })
        })?;
        Ok(TreeNodeRecursion::Continue)
    })
    .map_err(|error| sql_state_error(&error.to_string()))?;
    Ok(fits)
}

fn required_columns(query: &ValidatedQuery, alias: &str, schema: &SchemaRef) -> Option<Vec<usize>> {
    let DFStatement::Statement(statement) = query.statement() else {
        return None;
    };
    let Statement::Query(_) = statement.as_ref() else {
        return None;
    };
    let mut visitor = Dependencies {
        schema,
        alias,
        table_alias: None,
        outputs: BTreeSet::new(),
        indices: BTreeSet::new(),
        queries: 0,
        selects: 0,
    };
    if matches!(statement.visit(&mut visitor), ControlFlow::Break(())) {
        return None;
    }
    (visitor.queries == 1 && visitor.selects == 1).then(|| visitor.indices.into_iter().collect())
}

struct Dependencies<'a> {
    schema: &'a SchemaRef,
    alias: &'a str,
    table_alias: Option<String>,
    outputs: BTreeSet<String>,
    indices: BTreeSet<usize>,
    queries: usize,
    selects: usize,
}

impl Dependencies<'_> {
    fn column(&mut self, identifier: &Ident) -> ControlFlow<()> {
        let name = normalized(identifier);
        if let Ok(index) = self.schema.index_of(&name) {
            self.indices.insert(index);
            ControlFlow::Continue(())
        } else if self.outputs.contains(&name) {
            ControlFlow::Continue(())
        } else {
            ControlFlow::Break(())
        }
    }

    fn collect_output_aliases(&mut self, projection: &[SelectItem]) -> ControlFlow<()> {
        for item in projection {
            match item {
                SelectItem::ExprWithAlias { alias, .. } => {
                    self.outputs.insert(normalized(alias));
                }
                SelectItem::UnnamedExpr(_) => {}
                _ => return ControlFlow::Break(()),
            }
        }
        ControlFlow::Continue(())
    }
}

impl Visitor for Dependencies<'_> {
    type Break = ();

    fn pre_visit_query(&mut self, query: &Query) -> ControlFlow<()> {
        self.queries += 1;
        if self.queries != 1
            || query.with.is_some()
            || !matches!(query.body.as_ref(), SetExpr::Select(_))
        {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }

    fn pre_visit_select(&mut self, select: &Select) -> ControlFlow<()> {
        self.selects += 1;
        if self.selects != 1
            || select.from.len() != 1
            || !select.from[0].joins.is_empty()
            || matches!(select.group_by, GroupByExpr::All(_))
        {
            return ControlFlow::Break(());
        }
        let TableFactor::Table {
            name,
            alias,
            args: None,
            ..
        } = &select.from[0].relation
        else {
            return ControlFlow::Break(());
        };
        if name.to_string() != self.alias {
            return ControlFlow::Break(());
        }
        if alias
            .as_ref()
            .is_some_and(|alias| !alias.columns.is_empty())
        {
            return ControlFlow::Break(());
        }
        self.table_alias = alias.as_ref().map(|alias| normalized(&alias.name));
        self.collect_output_aliases(&select.projection)
    }

    fn pre_visit_expr(&mut self, expr: &Expr) -> ControlFlow<()> {
        match expr {
            Expr::Identifier(identifier) => self.column(identifier),
            Expr::CompoundIdentifier(parts) if parts.len() == 2 => {
                let qualifier = normalized(&parts[0]);
                if qualifier != self.alias && self.table_alias.as_ref() != Some(&qualifier) {
                    ControlFlow::Break(())
                } else {
                    self.column(&parts[1])
                }
            }
            Expr::Function(function) => function_dependencies(function),
            Expr::Value(_)
            | Expr::BinaryOp { .. }
            | Expr::UnaryOp { .. }
            | Expr::Nested(_)
            | Expr::Cast { .. }
            | Expr::Between { .. }
            | Expr::InList { .. }
            | Expr::Case { .. }
            | Expr::IsNull(_)
            | Expr::IsNotNull(_)
            | Expr::IsTrue(_)
            | Expr::IsFalse(_)
            | Expr::IsNotTrue(_)
            | Expr::IsNotFalse(_)
            | Expr::IsUnknown(_)
            | Expr::IsNotUnknown(_)
            | Expr::Like { .. }
            | Expr::ILike { .. }
            | Expr::SimilarTo { .. }
            | Expr::Extract { .. }
            | Expr::Substring { .. }
            | Expr::Trim { .. }
            | Expr::Collate { .. }
            | Expr::Ceil { .. }
            | Expr::Floor { .. }
            | Expr::Position { .. } => ControlFlow::Continue(()),
            _ => ControlFlow::Break(()),
        }
    }
}

fn function_dependencies(function: &Function) -> ControlFlow<()> {
    if function.over.is_some() {
        return ControlFlow::Break(());
    }
    if let FunctionArguments::List(arguments) = &function.args {
        let wildcard = arguments.args.iter().any(|arg| {
            !matches!(
                arg,
                FunctionArg::Unnamed(FunctionArgExpr::Expr(_))
                    | FunctionArg::Named {
                        arg: FunctionArgExpr::Expr(_),
                        ..
                    }
                    | FunctionArg::ExprNamed {
                        arg: FunctionArgExpr::Expr(_),
                        ..
                    }
            )
        });
        if wildcard
            && !(function.name.to_string().eq_ignore_ascii_case("count")
                && matches!(
                    arguments.args.as_slice(),
                    [FunctionArg::Unnamed(FunctionArgExpr::Wildcard)]
                ))
        {
            return ControlFlow::Break(());
        }
    } else if matches!(function.args, FunctionArguments::Subquery(_)) {
        return ControlFlow::Break(());
    }
    ControlFlow::Continue(())
}

fn normalized(identifier: &Ident) -> String {
    if identifier.quote_style.is_some() {
        identifier.value.clone()
    } else {
        identifier.value.to_ascii_lowercase()
    }
}
