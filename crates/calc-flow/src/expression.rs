use std::sync::OnceLock;

use datafusion::{
    execution::{FunctionRegistry, context::SessionContext},
    logical_expr::Volatility,
    sql::{
        parser::DFParser,
        sqlparser::{
            ast::{Expr, visit_expressions},
            dialect::GenericDialect,
        },
    },
};
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

/// `DataFusion` datetime built-ins that read the wall clock while declaring
/// [`Volatility::Stable`], so the volatility check alone lets them through
/// and checkpoint replays produce different output.
const WALL_CLOCK_BUILTINS: [&str; 3] = ["now", "current_date", "current_time"];

/// Reject a read-only query that calls a volatile or wall-clock built-in
/// function.
///
/// Stream plans replay deterministic work, so every function a stream query
/// can resolve must be non-volatile. The check resolves every function call
/// in the query against the supplied registry and rejects any function whose
/// signature is [`Volatility::Volatile`], plus the wall-clock built-ins in
/// [`WALL_CLOCK_BUILTINS`] that `DataFusion` marks stable even though they read
/// the wall clock and break deterministic replay. Matching happens on the
/// resolved function name, so aliases such as `current_timestamp` (of `now`)
/// are covered. Names the registry does not know are left to query planning,
/// which already rejects unknown functions.
///
/// # Errors
///
/// Returns [`CalcFlowError::Compile`] naming the node and the rejected
/// function, or [`CalcFlowError::InvalidArgument`] when the query does not
/// parse.
pub(crate) fn validate_no_volatile_functions(
    node_id: &str,
    query: &str,
    registry: &impl FunctionRegistry,
) -> Result<()> {
    let statements =
        DFParser::parse_sql_with_dialect(query, &GenericDialect {}).map_err(|error| {
            CalcFlowError::InvalidArgument {
                field: "query".into(),
                message: error.to_string(),
            }
        })?;
    let mut rejected = None;
    for statement in &statements {
        let datafusion::sql::parser::Statement::Statement(statement) = statement else {
            continue;
        };
        let _ = visit_expressions(statement.as_ref(), |expr| {
            if let Expr::Function(function) = expr {
                let name = function
                    .name
                    .0
                    .last()
                    .and_then(|part| part.as_ident())
                    .map(|ident| ident.value.to_lowercase())
                    .unwrap_or_default();
                if let Ok(udf) = registry.udf(&name) {
                    let kind = if matches!(udf.signature().volatility, Volatility::Volatile) {
                        "volatile"
                    } else if WALL_CLOCK_BUILTINS.contains(&udf.name()) {
                        "wall-clock"
                    } else {
                        return std::ops::ControlFlow::Continue(());
                    };
                    rejected = Some((name.clone(), kind));
                    return std::ops::ControlFlow::Break(());
                }
            }
            std::ops::ControlFlow::Continue(())
        });
        if rejected.is_some() {
            break;
        }
    }
    if let Some((name, kind)) = rejected {
        return Err(CalcFlowError::Compile {
            message: format!(
                "stream node {node_id:?} selects {kind} built-in function {name:?}; deterministic replay requires deterministic SQL"
            ),
        });
    }
    Ok(())
}

/// The shared default function registry used for stream SQL determinism
/// checks; built once because it mirrors the engine's execution sessions.
pub(crate) fn default_function_registry() -> &'static SessionContext {
    static REGISTRY: OnceLock<SessionContext> = OnceLock::new();
    REGISTRY.get_or_init(SessionContext::new)
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
    use std::sync::Arc;

    use super::{
        split_assignment, sql_projection, validate_no_volatile_functions, validate_select_query,
    };
    use crate::CalcFlowError;
    use datafusion::{
        arrow::datatypes::DataType,
        common::ScalarValue,
        execution::{FunctionRegistry, context::SessionContext},
        logical_expr::{ColumnarValue, ScalarUDF, Volatility, create_udf},
    };

    fn volatile_roll() -> ScalarUDF {
        create_udf(
            "roll",
            vec![],
            DataType::Float64,
            Volatility::Volatile,
            Arc::new(|_| Ok(ColumnarValue::Scalar(ScalarValue::Float64(Some(0.5))))),
        )
    }

    fn registry_with_roll() -> SessionContext {
        let context = SessionContext::new();
        context.register_udf(volatile_roll());
        context
    }

    #[test]
    fn volatile_builtin_function_is_rejected() {
        let registry = registry_with_roll();
        let error =
            validate_no_volatile_functions("features", "SELECT roll() AS x FROM input", &registry)
                .unwrap_err();
        let CalcFlowError::Compile { message } = error else {
            panic!("expected a compile error, got {error:?}");
        };
        assert!(message.contains("features"), "{message}");
        assert!(message.contains("roll"), "{message}");
    }

    #[test]
    fn volatile_function_is_matched_case_insensitively() {
        let registry = registry_with_roll();
        assert!(
            validate_no_volatile_functions("n", "SELECT ROLL() AS x FROM input", &registry)
                .is_err()
        );
    }

    #[test]
    fn volatile_function_is_found_in_where_case_and_subquery() {
        let registry = registry_with_roll();
        for query in [
            "SELECT x FROM input WHERE roll() > 0.5",
            "SELECT CASE WHEN roll() > 0.0 THEN 1.0 ELSE 0.0 END AS c FROM input",
            "SELECT r FROM (SELECT roll() AS r FROM input) AS t",
            "WITH rolled AS (SELECT roll() AS r FROM input) SELECT r FROM rolled",
        ] {
            assert!(
                validate_no_volatile_functions("n", query, &registry).is_err(),
                "{query}"
            );
        }
    }

    #[test]
    fn wall_clock_builtins_are_rejected() {
        let registry = SessionContext::new();
        for query in [
            "SELECT now() AS ts FROM input",
            "SELECT NOW() AS ts FROM input",
            "SELECT current_date() AS d FROM input",
            "SELECT current_date AS d FROM input",
            "SELECT current_time() AS t FROM input",
            "SELECT current_timestamp() AS ts FROM input",
            "SELECT current_timestamp AS ts FROM input",
            "SELECT today() AS d FROM input",
        ] {
            let error = validate_no_volatile_functions("features", query, &registry).unwrap_err();
            let CalcFlowError::Compile { message } = error else {
                panic!("expected a compile error for {query}, got {error:?}");
            };
            assert!(message.contains("features"), "{message} for {query}");
            assert!(message.contains("wall-clock"), "{message} for {query}");
        }
    }

    #[test]
    fn wall_clock_builtins_are_rejected_in_nested_positions() {
        let registry = SessionContext::new();
        for query in [
            "SELECT x FROM input WHERE now() > to_timestamp(0)",
            "SELECT CASE WHEN current_date() > to_timestamp(0) THEN 1.0 ELSE 0.0 END AS c FROM input",
            "WITH stamped AS (SELECT now() AS ts FROM input) SELECT ts FROM stamped",
        ] {
            assert!(
                validate_no_volatile_functions("n", query, &registry).is_err(),
                "{query}"
            );
        }
    }

    #[test]
    fn deterministic_datetime_functions_pass() {
        let registry = SessionContext::new();
        assert!(
            validate_no_volatile_functions(
                "n",
                "SELECT date_part('year', ts) AS y, to_timestamp(x) AS t FROM input WHERE date_bin(INTERVAL '1 minute', ts, to_timestamp(0)) IS NOT NULL",
                &registry,
            )
            .is_ok()
        );
    }

    #[test]
    fn deterministic_functions_pass() {
        let registry = registry_with_roll();
        assert!(
            validate_no_volatile_functions(
                "n",
                "SELECT abs(x) AS a, CASE WHEN x > 0.0 THEN ln(x) ELSE 0.0 END AS b FROM input WHERE sqrt(x) >= 0.0",
                &registry,
            )
            .is_ok()
        );
    }

    #[test]
    fn unknown_functions_are_left_to_query_planning() {
        let registry = registry_with_roll();
        assert!(
            validate_no_volatile_functions(
                "n",
                "SELECT definitely_not_a_builtin(x) AS a FROM input",
                &registry,
            )
            .is_ok()
        );
        let context = SessionContext::new();
        assert!(context.udf("roll").is_err());
        assert!(
            validate_no_volatile_functions("n", "SELECT roll() AS x FROM input", &context).is_ok()
        );
    }

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
