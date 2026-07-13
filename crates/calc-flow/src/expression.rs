use std::sync::OnceLock;

use datafusion::sql::{parser::DFParser, sqlparser::dialect::GenericDialect};
use regex::Regex;

use crate::{CalcFlowError, Result};

/// Split a named assignment into its output name and expression.
///
/// # Panics
///
/// Panics if the constant assignment regular expression is invalid.
pub fn split_assignment(expression: &str) -> Option<(&str, &str)> {
    static ASSIGNMENT: OnceLock<Regex> = OnceLock::new();
    let regex = ASSIGNMENT.get_or_init(|| {
        Regex::new(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^=].*)$")
            .expect("constant regex is valid")
    });
    let captures = regex.captures(expression)?;
    Some((captures.get(1)?.as_str(), captures.get(2)?.as_str().trim()))
}

/// Build a table projection for an expression or named assignment.
///
/// # Errors
///
/// Returns [`CalcFlowError::InvalidArgument`] when `table_name` is not a SQL
/// identifier.
pub fn sql_projection(expression: &str, table_name: &str) -> Result<String> {
    if !is_identifier(table_name) {
        return Err(CalcFlowError::InvalidArgument {
            field: "table_name".into(),
            message: "must be a SQL identifier".into(),
        });
    }
    Ok(match split_assignment(expression) {
        Some((name, value)) => format!("SELECT *, ({value}) AS {name} FROM {table_name}"),
        None => format!("SELECT ({}) AS result FROM {table_name}", expression.trim()),
    })
}

/// Validate and normalize one SELECT or CTE query.
///
/// # Errors
///
/// Returns [`CalcFlowError::InvalidArgument`] when parsing fails or the input
/// is not exactly one SELECT or CTE query.
pub fn validate_select_query(query: &str) -> Result<String> {
    let statements =
        DFParser::parse_sql_with_dialect(query, &GenericDialect {}).map_err(|error| {
            CalcFlowError::InvalidArgument {
                field: "query".into(),
                message: error.to_string(),
            }
        })?;
    if statements.len() != 1
        || !matches!(
            statements.front(),
            Some(datafusion::sql::parser::Statement::Statement(statement))
                if matches!(
                    statement.as_ref(),
                    datafusion::sql::sqlparser::ast::Statement::Query(_)
                )
        )
    {
        return Err(CalcFlowError::InvalidArgument {
            field: "query".into(),
            message: "exactly one SELECT or CTE query is required".into(),
        });
    }
    Ok(query.trim().trim_end_matches(';').trim().to_owned())
}

fn is_identifier(value: &str) -> bool {
    let mut chars = value.chars();
    chars
        .next()
        .is_some_and(|ch| ch == '_' || ch.is_ascii_alphabetic())
        && chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}
