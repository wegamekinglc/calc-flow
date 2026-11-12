//! Shared AST recognition and independent arm planning for append union.

use std::collections::{BTreeMap, BTreeSet};

use datafusion::{
    arrow::datatypes::SchemaRef,
    execution::FunctionRegistry,
    sql::{
        parser::Statement as DFStatement,
        sqlparser::ast::{
            Expr, Query, SetExpr, SetOperator, SetQuantifier, Statement, TableFactor,
            visit_expressions,
        },
    },
};
use serde::Serialize;

use crate::{ArrowFieldSpec, CalcFlowError, Result, SqlOperator};

#[doc(hidden)]
#[derive(Clone, Debug, Serialize)]
pub struct UnionArmQuery {
    pub alias: String,
    pub query: String,
}

#[doc(hidden)]
#[derive(Debug, Serialize)]
pub struct UnionArm {
    pub alias: String,
    pub query: String,
    pub schema: Vec<ArrowFieldSpec>,
    #[serde(skip)]
    pub(crate) planned_schema: SchemaRef,
}

#[doc(hidden)]
#[derive(Debug, Serialize)]
pub struct UnionPlan {
    pub arms: Vec<UnionArm>,
    pub mismatch: Option<(usize, String, String)>,
    pub planning_error: Option<(usize, String)>,
}

#[doc(hidden)]
pub fn union_all_queries(text: &str, aliases: &[String]) -> Result<Option<Vec<UnionArmQuery>>> {
    if aliases.len() < 2 {
        return Ok(None);
    }
    let parsed = crate::expression::parse_select_query(text)?;
    let DFStatement::Statement(statement) = parsed.statement() else {
        return Ok(None);
    };
    let Statement::Query(query) = statement.as_ref() else {
        return Ok(None);
    };
    let Some(parts) = flatten_query(query) else {
        return Ok(None);
    };
    Ok(named_arms(parts, aliases))
}

fn named_arms(parts: Vec<Query>, aliases: &[String]) -> Option<Vec<UnionArmQuery>> {
    let mut used = BTreeSet::new();
    let mut arms = Vec::new();
    for part in parts {
        let alias = arm_alias(&part, aliases)?;
        if !used.insert(alias.clone()) || !proved_arm(&part) {
            return None;
        }
        arms.push(UnionArmQuery {
            alias,
            query: part.to_string(),
        });
    }
    (arms.len() == aliases.len()).then_some(arms)
}

fn unmodified(query: &Query) -> bool {
    [
        query.with.is_none(),
        query.order_by.is_none(),
        query.limit_clause.is_none(),
        query.fetch.is_none(),
        query.locks.is_empty(),
        query.for_clause.is_none(),
        query.settings.is_none(),
        query.format_clause.is_none(),
        query.pipe_operators.is_empty(),
    ]
    .into_iter()
    .all(std::convert::identity)
}

fn flatten_query(query: &Query) -> Option<Vec<Query>> {
    if !unmodified(query) {
        return None;
    }
    flatten_set(&query.body, query)
}

fn flatten_set(body: &SetExpr, template: &Query) -> Option<Vec<Query>> {
    match body {
        SetExpr::SetOperation {
            op: SetOperator::Union,
            set_quantifier: SetQuantifier::All,
            left,
            right,
        } => {
            let mut parts = flatten_set(left, template)?;
            parts.extend(flatten_set(right, template)?);
            Some(parts)
        }
        SetExpr::Query(query) => flatten_query(query),
        SetExpr::Select(_) => {
            let mut arm = template.clone();
            arm.body = Box::new(body.clone());
            Some(vec![arm])
        }
        _ => None,
    }
}

fn arm_alias(query: &Query, aliases: &[String]) -> Option<String> {
    let SetExpr::Select(select) = query.body.as_ref() else {
        return None;
    };
    let [from] = select.from.as_slice() else {
        return None;
    };
    let TableFactor::Table { name, .. } = &from.relation else {
        return None;
    };
    let [part] = name.0.as_slice() else {
        return None;
    };
    let name = part.as_ident()?;
    aliases
        .iter()
        .find(|alias| {
            if name.quote_style.is_some() {
                *alias == &name.value
            } else {
                alias.eq_ignore_ascii_case(&name.value)
            }
        })
        .cloned()
}

fn proved_arm(query: &Query) -> bool {
    let text = query.to_string();
    if !crate::query_analysis::union_arm_proved(&text) {
        return false;
    }
    let registry = crate::expression::default_function_registry();
    if crate::expression::validate_no_volatile_functions("union", &text, registry).is_err() {
        return false;
    }
    let functions = visit_expressions(query, |expr| {
        if let Expr::Function(function) = expr {
            let known = function.name.0.len() == 1
                && function.name.0[0]
                    .as_ident()
                    .is_some_and(|name| registry.udf(&name.value.to_ascii_lowercase()).is_ok());
            if !known {
                return std::ops::ControlFlow::Break(());
            }
        }
        std::ops::ControlFlow::Continue(())
    });
    !matches!(functions, std::ops::ControlFlow::Break(()))
}

#[doc(hidden)]
pub async fn plan_union_all(
    text: &str,
    schemas: &BTreeMap<String, SchemaRef>,
) -> Result<Option<UnionPlan>> {
    let Some(queries) = union_all_queries(text, &schemas.keys().cloned().collect::<Vec<_>>())?
    else {
        return Ok(None);
    };
    let mut arms = Vec::new();
    for (index, arm) in queries.into_iter().enumerate() {
        match plan_arm(arm, schemas).await {
            Ok(arm) => arms.push(arm),
            Err(error) => {
                return Ok(Some(UnionPlan {
                    arms,
                    mismatch: None,
                    planning_error: Some((index, error.to_string())),
                }));
            }
        }
    }
    let mismatch =
        arms.iter().enumerate().skip(1).find_map(
            |(index, arm)| match crate::dataframe::union_schema(
                &arms[0].planned_schema,
                &arm.planned_schema,
                "",
            ) {
                Err(CalcFlowError::InvalidArgument { field, message }) => Some((
                    index,
                    field,
                    message
                        .strip_prefix("schema_mismatch: ")
                        .unwrap_or(&message)
                        .to_owned(),
                )),
                _ => None,
            },
        );
    Ok(Some(UnionPlan {
        arms,
        mismatch,
        planning_error: None,
    }))
}

async fn plan_arm(arm: UnionArmQuery, schemas: &BTreeMap<String, SchemaRef>) -> Result<UnionArm> {
    let operator = SqlOperator::new("union_arm", &arm.query, vec![arm.alias.clone()], vec![])?;
    let schema = operator
        .infer_schema(&BTreeMap::from([(
            arm.alias.clone(),
            schemas[&arm.alias].clone(),
        )]))
        .await?;
    Ok(UnionArm {
        alias: arm.alias,
        query: arm.query,
        schema: crate::query_analysis::schema_fields(&schema),
        planned_schema: schema,
    })
}
