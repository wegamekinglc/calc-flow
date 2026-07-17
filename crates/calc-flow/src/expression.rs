use std::sync::OnceLock;

use datafusion::sql::{parser::DFParser, sqlparser::dialect::GenericDialect};
use regex::Regex;

use crate::{CalcFlowError, Result};

/// Split a named assignment into its output name and expression.
///
/// # Panics
///
/// Panics if the constant assignment regular expression is invalid.
pub(crate) fn split_assignment(expression: &str) -> Option<(&str, &str)> {
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
pub(crate) fn sql_projection(expression: &str, table_name: &str) -> Result<String> {
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
pub(crate) fn validate_select_query(query: &str) -> Result<String> {
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

#[cfg(test)]
mod tests {
    use super::{split_assignment, sql_projection, validate_select_query};
    use crate::CalcFlowError;

    #[test]
    fn assignment_accepts_comparisons_in_the_right_hand_side() {
        assert_eq!(split_assignment("total = a + b"), Some(("total", "a + b")));
        assert_eq!(
            split_assignment("eligible = amount >= threshold"),
            Some(("eligible", "amount >= threshold"))
        );
        assert_eq!(
            split_assignment("same = left == right"),
            Some(("same", "left == right"))
        );
        assert_eq!(
            split_assignment("label = 'a != b'"),
            Some(("label", "'a != b'"))
        );
    }

    #[test]
    fn comparison_delimiters_are_not_assignments() {
        assert_eq!(split_assignment("left == right"), None);
        assert_eq!(split_assignment("left != right"), None);
        assert_eq!(split_assignment("left <= right"), None);
        assert_eq!(split_assignment("left >= right"), None);
    }

    #[test]
    fn projection_builds_assignment_and_expression_queries() {
        assert_eq!(
            sql_projection("total = a + b", "input").unwrap(),
            "SELECT *, (a + b) AS total FROM input"
        );
        assert_eq!(
            sql_projection("a + b", "input").unwrap(),
            "SELECT (a + b) AS result FROM input"
        );
    }

    #[test]
    fn projection_rejects_invalid_table_identifiers() {
        for table_name in ["", "1input", "input.table", "input; DROP TABLE input"] {
            assert!(matches!(
                sql_projection("a + b", table_name),
                Err(CalcFlowError::InvalidArgument { field, .. }) if field == "table_name"
            ));
        }
    }

    #[test]
    fn select_query_validation_normalizes_one_trailing_semicolon() {
        assert_eq!(
            validate_select_query("  WITH x AS (SELECT 1) SELECT * FROM x;  ").unwrap(),
            "WITH x AS (SELECT 1) SELECT * FROM x"
        );
    }

    #[test]
    fn malformed_sql_reports_the_query_field() {
        assert!(matches!(
            validate_select_query("SELECT FROM"),
            Err(CalcFlowError::InvalidArgument { field, .. }) if field == "query"
        ));
    }

    #[test]
    fn select_query_validation_rejects_multiple_statements_and_dml() {
        assert!(validate_select_query("SELECT 1; SELECT 2").is_err());
        assert!(validate_select_query("INSERT INTO input VALUES (1)").is_err());
    }

    #[test]
    fn select_query_validation_rejects_datafusion_extension_ddl() {
        assert!(
            validate_select_query(
                "CREATE EXTERNAL TABLE input(c1 int) STORED AS CSV LOCATION 'input.csv'"
            )
            .is_err()
        );
    }
}
