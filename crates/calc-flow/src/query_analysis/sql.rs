//! Conservative SQL proofs over the runtime's already validated AST.

use datafusion::sql::{
    parser::Statement as DFStatement,
    sqlparser::ast::{
        Expr, Function, GroupByExpr, Ident, LimitClause, OrderByExpr, OrderByKind, Query, Select,
        SelectItem, SetExpr, Statement, Value,
    },
};

use crate::{
    NullPlacement, QueryOrderingFacts, QueryRowOrder, QuerySortKey, QueryTieOrder, SortDirection,
    expression::{parse_select_query, row_local_statement, simple_query},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Proof {
    RowLocal,
    Aggregate,
    PerBatch,
    Selection,
    PerBatchUnproven,
    Unknown,
}

impl Proof {
    pub(super) const fn per_batch(self) -> bool {
        matches!(
            self,
            Self::PerBatch | Self::Selection | Self::PerBatchUnproven
        )
    }
    pub(super) const fn counterexample(self) -> bool {
        matches!(self, Self::PerBatch | Self::Selection)
    }
}

pub(super) struct SqlFacts {
    pub aggregate: bool,
    pub proof: Proof,
    pub ordering: QueryOrderingFacts,
}

pub(super) fn inspect(
    text: &str,
    fields: &[crate::ArrowFieldSpec],
    latest: bool,
) -> crate::Result<SqlFacts> {
    let validated = parse_select_query(text)?;
    let aggregate = validated.has_stream_aggregate();
    let mut result = SqlFacts {
        aggregate,
        proof: Proof::Unknown,
        ordering: QueryOrderingFacts::default(),
    };
    let DFStatement::Statement(statement) = validated.statement() else {
        return Ok(result);
    };
    let Statement::Query(query) = statement.as_ref() else {
        return Ok(result);
    };
    if proved_cte(&statement, query, aggregate, latest) {
        result.proof = Proof::RowLocal;
        return Ok(result);
    }
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Ok(result);
    };
    if !has_simple_shape(query) {
        return Ok(result);
    }
    result.proof = query_proof(&statement, query, select, aggregate, latest);
    if let Some(order) = &query.order_by {
        result.ordering = order_keys(order, fields);
    }
    Ok(result)
}

fn proved_cte(statement: &Statement, query: &Query, aggregate: bool, latest: bool) -> bool {
    latest
        && !aggregate
        && query.with.is_some()
        && row_local_cte(query)
        && row_local_statement(statement)
}

fn has_simple_shape(query: &Query) -> bool {
    let mut normalized = query.clone();
    normalized.order_by = None;
    normalized.limit_clause = None;
    let SetExpr::Select(normal_select) = normalized.body.as_mut() else {
        return false;
    };
    normal_select.selection = None;
    normal_select.distinct = None;
    normal_select.having = None;
    normal_select.group_by = GroupByExpr::Expressions(vec![], vec![]);
    for item in &mut normal_select.projection {
        if matches!(
            item,
            SelectItem::Wildcard(_) | SelectItem::QualifiedWildcard(..)
        ) {
            *item = SelectItem::UnnamedExpr(Expr::Identifier(Ident::new("__query_field")));
        }
    }
    simple_query(&normalized)
}

fn query_proof(
    statement: &Statement,
    query: &Query,
    select: &Select,
    aggregate: bool,
    latest: bool,
) -> Proof {
    if aggregate {
        aggregate_proof(query, select)
    } else if row_local_statement(statement) {
        row_local_proof(query, select, latest)
    } else if select.projection.iter().all(row_number_item) {
        selection_proof(select.selection.is_none())
    } else {
        Proof::Unknown
    }
}

fn aggregate_proof(query: &Query, select: &Select) -> Proof {
    if simple_aggregate(select)
        && query.order_by.is_none()
        && query.limit_clause.is_none()
        && select.having.is_none()
    {
        Proof::Aggregate
    } else {
        Proof::Unknown
    }
}

fn row_local_proof(query: &Query, select: &Select, latest: bool) -> Proof {
    if let Some(limit) = &query.limit_clause {
        return selection_proof(positive_limit(limit) && select.selection.is_none());
    }
    if select.distinct.is_some() {
        return if select.selection.is_none() {
            Proof::PerBatch
        } else {
            Proof::PerBatchUnproven
        };
    }
    if latest && query.order_by.is_some() {
        Proof::PerBatchUnproven
    } else {
        Proof::RowLocal
    }
}

fn row_local_cte(query: &Query) -> bool {
    let mut local = query.clone();
    local.with = None;
    let SetExpr::Select(select) = local.body.as_ref() else {
        return false;
    };
    has_simple_shape(&local)
        && select.distinct.is_none()
        && select.having.is_none()
        && query.order_by.is_none()
        && query.limit_clause.is_none()
        && query.with.as_ref().is_none_or(|with| {
            !with.recursive && with.cte_tables.iter().all(|cte| row_local_cte(&cte.query))
        })
}

fn selection_proof(proved: bool) -> Proof {
    if proved {
        Proof::Selection
    } else {
        Proof::PerBatchUnproven
    }
}

fn row_number_item(item: &SelectItem) -> bool {
    match expression(item) {
        Some(Expr::Identifier(_) | Expr::CompoundIdentifier(_)) => true,
        Some(Expr::Function(function)) => {
            function.over.is_some() && function.name.to_string().eq_ignore_ascii_case("row_number")
        }
        _ => false,
    }
}

fn positive_limit(limit: &LimitClause) -> bool {
    let LimitClause::LimitOffset {
        limit: Some(Expr::Value(value)),
        offset: None,
        limit_by,
    } = limit
    else {
        return false;
    };
    limit_by.is_empty()
        && matches!(&value.value, Value::Number(number, _) if number.parse::<u64>().is_ok_and(|value| value > 0))
}

fn expression(item: &SelectItem) -> Option<&Expr> {
    match item {
        SelectItem::UnnamedExpr(expr) | SelectItem::ExprWithAlias { expr, .. } => Some(expr),
        _ => None,
    }
}

fn simple_aggregate(select: &Select) -> bool {
    let GroupByExpr::Expressions(groups, modifiers) = &select.group_by else {
        return false;
    };
    modifiers.is_empty()
        && groups
            .iter()
            .all(|expr| matches!(expr, Expr::Identifier(_) | Expr::CompoundIdentifier(_)))
        && select.projection.iter().all(simple_aggregate_item)
}

fn simple_aggregate_item(item: &SelectItem) -> bool {
    match expression(item) {
        Some(Expr::Identifier(_) | Expr::CompoundIdentifier(_)) => true,
        Some(Expr::Function(function)) => simple_aggregate_function(function),
        _ => false,
    }
}

fn simple_aggregate_function(function: &Function) -> bool {
    function.over.is_none()
        && function.filter.is_none()
        && function.within_group.is_empty()
        && matches!(
            function.name.to_string().to_ascii_lowercase().as_str(),
            "sum" | "count" | "min" | "max" | "avg"
        )
}

fn order_keys(
    order: &datafusion::sql::sqlparser::ast::OrderBy,
    fields: &[crate::ArrowFieldSpec],
) -> QueryOrderingFacts {
    let unknown = || QueryOrderingFacts {
        kind: QueryRowOrder::Unknown,
        keys: vec![],
        ties: QueryTieOrder::Unknown,
        temporal_identity: None,
    };
    let OrderByKind::Expressions(keys) = &order.kind else {
        return unknown();
    };
    if order.interpolate.is_some() || keys.is_empty() {
        return unknown();
    }
    let keys = keys
        .iter()
        .map(|key| order_key(key, fields))
        .collect::<Option<Vec<_>>>();
    keys.map_or_else(unknown, |keys| QueryOrderingFacts {
        kind: QueryRowOrder::Ordered,
        keys,
        ties: QueryTieOrder::Unknown,
        temporal_identity: None,
    })
}

fn order_key(key: &OrderByExpr, fields: &[crate::ArrowFieldSpec]) -> Option<QuerySortKey> {
    let Expr::Identifier(name) = &key.expr else {
        return None;
    };
    if key.with_fill.is_some() || !fields.iter().any(|field| field.name == name.value) {
        return None;
    }
    let ascending = key.options.asc.unwrap_or(true);
    Some(QuerySortKey {
        field: name.value.clone(),
        direction: if ascending {
            SortDirection::Ascending
        } else {
            SortDirection::Descending
        },
        nulls: if key.options.nulls_first.unwrap_or(!ascending) {
            NullPlacement::First
        } else {
            NullPlacement::Last
        },
    })
}
