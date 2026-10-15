//! Input dependencies for row-local expressions over the declared input table.

use std::{collections::BTreeSet, ops::ControlFlow};

use datafusion::{
    arrow::datatypes::SchemaRef,
    execution::FunctionRegistry,
    logical_expr::Volatility,
    sql::{
        parser::Statement as DFStatement,
        sqlparser::ast::{
            Expr, Function, FunctionArg, FunctionArgExpr, FunctionArguments, Ident, Query, Select,
            SelectItem, SetExpr, Statement, Visit, Visitor,
        },
    },
};

use crate::expression::ValidatedQuery;

pub(in super::super) fn required_columns(
    query: &ValidatedQuery,
    input: &SchemaRef,
) -> Option<Vec<usize>> {
    if query.has_stream_aggregate() {
        return None;
    }
    let DFStatement::Statement(statement) = query.statement() else {
        return None;
    };
    let select = row_local_select(&statement)?;
    let mut visitor = Dependencies {
        input,
        columns: BTreeSet::new(),
    };
    for item in &select.projection {
        let (SelectItem::UnnamedExpr(expression)
        | SelectItem::ExprWithAlias {
            expr: expression, ..
        }) = item
        else {
            return None;
        };
        if expression.visit(&mut visitor).is_break() {
            return None;
        }
    }
    if select
        .selection
        .as_ref()
        .is_some_and(|expression| expression.visit(&mut visitor).is_break())
    {
        return None;
    }
    Some(visitor.columns.into_iter().collect())
}

fn row_local_select(statement: &Statement) -> Option<&Select> {
    let Statement::Query(query) = statement else {
        return None;
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        return None;
    };
    let projection = select
        .projection
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let predicate = select
        .selection
        .as_ref()
        .map_or_else(String::new, |expression| format!(" WHERE {expression}"));
    // Every other clause or table scope keeps the full input. This comparison
    // uses the stored AST's canonical formatter, rather than reparsing SQL.
    (query.to_string() == format!("SELECT {projection} FROM input{predicate}")).then_some(select)
}

struct Dependencies<'a> {
    input: &'a SchemaRef,
    columns: BTreeSet<usize>,
}

impl Dependencies<'_> {
    fn qualified_column(&mut self, parts: &[Ident]) -> ControlFlow<()> {
        let [qualifier, name] = parts else {
            return ControlFlow::Break(());
        };
        if super::identifier(qualifier) != "input" {
            return ControlFlow::Break(());
        }
        self.column(name)
    }

    fn column(&mut self, name: &Ident) -> ControlFlow<()> {
        let name = super::identifier(name);
        let mut fields = self
            .input
            .fields()
            .iter()
            .enumerate()
            .filter(|(_, field)| field.name() == &name);
        let Some((index, _)) = fields.next() else {
            return ControlFlow::Break(());
        };
        if fields.next().is_some() {
            return ControlFlow::Break(());
        }
        self.columns.insert(index);
        ControlFlow::Continue(())
    }
}

impl Visitor for Dependencies<'_> {
    type Break = ();

    fn pre_visit_query(&mut self, _query: &Query) -> ControlFlow<()> {
        // Only projection/predicate expressions are visited, so any query
        // reached here is a nested scope with batch-dependent semantics.
        ControlFlow::Break(())
    }

    fn pre_visit_expr(&mut self, expression: &Expr) -> ControlFlow<()> {
        match expression {
            Expr::Identifier(name) => self.column(name),
            Expr::CompoundIdentifier(parts) => self.qualified_column(parts),
            Expr::Function(function) if !row_local_function(function) => ControlFlow::Break(()),
            Expr::Wildcard(_) | Expr::QualifiedWildcard(..) => ControlFlow::Break(()),
            _ if row_local_expression(expression) => ControlFlow::Continue(()),
            _ => ControlFlow::Break(()),
        }
    }
}

fn row_local_function(function: &Function) -> bool {
    immutable_builtin(function)
        && function.over.is_none()
        && function.filter.is_none()
        && function.within_group.is_empty()
        && explicit_arguments(&function.parameters)
        && explicit_arguments(&function.args)
}

fn immutable_builtin(function: &Function) -> bool {
    let [name] = function.name.0.as_slice() else {
        return false;
    };
    let Some(name) = name.as_ident() else {
        return false;
    };
    crate::expression::default_function_registry()
        .udf(&super::identifier(name))
        .is_ok_and(|function| function.signature().volatility == Volatility::Immutable)
}

fn row_local_expression(expression: &Expr) -> bool {
    matches!(
        expression,
        Expr::Function(_)
            | Expr::Value(_)
            | Expr::BinaryOp { .. }
            | Expr::UnaryOp { .. }
            | Expr::IsFalse(_)
            | Expr::IsNotFalse(_)
            | Expr::IsTrue(_)
            | Expr::IsNotTrue(_)
            | Expr::IsNull(_)
            | Expr::IsNotNull(_)
            | Expr::IsUnknown(_)
            | Expr::IsNotUnknown(_)
            | Expr::IsDistinctFrom(..)
            | Expr::IsNotDistinctFrom(..)
            | Expr::InList { .. }
            | Expr::Between { .. }
            | Expr::Like { .. }
            | Expr::ILike { .. }
            | Expr::SimilarTo { .. }
            | Expr::RLike { .. }
            | Expr::Cast { .. }
            | Expr::Convert { .. }
            | Expr::Extract { .. }
            | Expr::Ceil { .. }
            | Expr::Floor { .. }
            | Expr::Position { .. }
            | Expr::Substring { .. }
            | Expr::Trim { .. }
            | Expr::Overlay { .. }
            | Expr::Collate { .. }
            | Expr::Nested(_)
            | Expr::Case { .. }
            | Expr::Tuple(_)
            | Expr::Array(_)
    )
}

fn explicit_arguments(arguments: &FunctionArguments) -> bool {
    match arguments {
        FunctionArguments::None => true,
        FunctionArguments::Subquery(_) => false,
        FunctionArguments::List(arguments) => {
            arguments.duplicate_treatment.is_none()
                && arguments.clauses.is_empty()
                && arguments.args.iter().all(|argument| {
                    let argument = match argument {
                        FunctionArg::Named { arg, .. }
                        | FunctionArg::ExprNamed { arg, .. }
                        | FunctionArg::Unnamed(arg) => arg,
                    };
                    matches!(argument, FunctionArgExpr::Expr(_))
                })
        }
    }
}
