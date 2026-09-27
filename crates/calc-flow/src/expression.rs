use std::{ops::ControlFlow, sync::OnceLock};

use datafusion::{
    execution::{FunctionRegistry, context::SessionContext},
    logical_expr::Volatility,
    sql::{
        parser::{DFParser, Statement as DFStatement},
        sqlparser::{
            ast::{Expr, ObjectName, Query, TableFactor, Visit, Visitor, visit_expressions},
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

/// One read-only SELECT or CTE query, validated and parsed once.
///
/// Operators keep this value so repeated executions plan the stored statement
/// instead of reparsing the normalized text.
#[derive(Clone, Debug)]
pub(crate) struct ValidatedQuery {
    text: String,
    statement: DFStatement,
}

impl ValidatedQuery {
    /// The normalized query text.
    pub(crate) fn text(&self) -> &str {
        &self.text
    }

    /// An owned copy of the parsed statement for one logical planning pass.
    pub(crate) fn statement(&self) -> DFStatement {
        self.statement.clone()
    }
}

/// Validate, normalize, and parse one SELECT or CTE query.
///
/// # Errors
///
/// Returns [`CalcFlowError::InvalidArgument`] when parsing fails or the input
/// is not exactly one SELECT or CTE query.
pub(crate) fn parse_select_query(query: &str) -> Result<ValidatedQuery> {
    let statement = validated_statement(query)?;
    let text = query.trim().trim_end_matches(';').trim();
    // Planning consumes the normalized text's statement so source spans match it.
    let statement = if text == query {
        statement
    } else {
        validated_statement(text)?
    };
    Ok(ValidatedQuery {
        text: text.to_owned(),
        statement,
    })
}

/// Validate and normalize one SELECT or CTE query.
///
/// # Errors
///
/// Returns [`CalcFlowError::InvalidArgument`] when parsing fails or the input
/// is not exactly one SELECT or CTE query.
pub(crate) fn validate_select_query(query: &str) -> Result<String> {
    parse_select_query(query).map(|query| query.text)
}

fn validated_statement(query: &str) -> Result<DFStatement> {
    let mut statements =
        DFParser::parse_sql_with_dialect(query, &GenericDialect {}).map_err(|error| {
            CalcFlowError::InvalidArgument {
                field: "query".into(),
                message: error.to_string(),
            }
        })?;
    let Some(datafusion::sql::parser::Statement::Statement(statement)) = statements.front() else {
        return Err(CalcFlowError::InvalidArgument {
            field: "query".into(),
            message: "exactly one SELECT or CTE query is required".into(),
        });
    };
    let datafusion::sql::sqlparser::ast::Statement::Query(parsed) = statement.as_ref() else {
        return Err(CalcFlowError::InvalidArgument {
            field: "query".into(),
            message: "exactly one SELECT or CTE query is required".into(),
        });
    };
    if statements.len() != 1 {
        return Err(CalcFlowError::InvalidArgument {
            field: "query".into(),
            message: "exactly one SELECT or CTE query is required".into(),
        });
    }
    if let ControlFlow::Break(message) = parsed.visit(&mut ResourceLimitVisitor) {
        return Err(CalcFlowError::InvalidArgument {
            field: "query".into(),
            message: message.into(),
        });
    }
    statements
        .pop_front()
        .ok_or_else(|| CalcFlowError::Internal {
            message: "validated query statement disappeared".into(),
        })
}

struct ResourceLimitVisitor;

impl Visitor for ResourceLimitVisitor {
    type Break = &'static str;

    fn pre_visit_query(&mut self, query: &Query) -> ControlFlow<Self::Break> {
        if query.with.as_ref().is_some_and(|with| with.recursive) {
            ControlFlow::Break("recursive CTEs are unavailable")
        } else {
            ControlFlow::Continue(())
        }
    }

    fn pre_visit_table_factor(&mut self, factor: &TableFactor) -> ControlFlow<Self::Break> {
        let name = match factor {
            TableFactor::Table {
                name,
                args: Some(_),
                ..
            }
            | TableFactor::Function { name, .. } => Some(name),
            _ => None,
        };
        if name.is_some_and(is_unbounded_generator) {
            ControlFlow::Break("generate_series and range table functions are unavailable")
        } else {
            ControlFlow::Continue(())
        }
    }

    fn pre_visit_expr(&mut self, expr: &Expr) -> ControlFlow<Self::Break> {
        if matches!(expr, Expr::Function(function) if is_unbounded_generator(&function.name)) {
            ControlFlow::Break("generate_series and range table functions are unavailable")
        } else {
            ControlFlow::Continue(())
        }
    }
}

fn is_unbounded_generator(name: &ObjectName) -> bool {
    name.0
        .last()
        .and_then(|part| part.as_ident())
        .is_some_and(|ident| {
            ident.value.eq_ignore_ascii_case("generate_series")
                || ident.value.eq_ignore_ascii_case("range")
        })
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
                        return ControlFlow::Continue(());
                    };
                    rejected = Some((name.clone(), kind));
                    return ControlFlow::Break(());
                }
            }
            ControlFlow::Continue(())
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
        parse_select_query, split_assignment, sql_projection, validate_no_volatile_functions,
        validate_select_query,
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
    fn parsed_query_keeps_the_normalized_statement_for_planning() {
        let query = parse_select_query("  SELECT 1 AS one;  ").unwrap();

        assert_eq!(query.text(), "SELECT 1 AS one");
        assert_eq!(query.statement().to_string(), "SELECT 1 AS one");
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
    fn select_query_validation_rejects_unbounded_generators_and_recursion() {
        for query in [
            "SELECT * FROM generate_series(1, 1000000000)",
            "SELECT * FROM range(1000000000)",
            "SELECT * FROM (SELECT * FROM generate_series(1, 1000000000)) AS generated",
            "WITH RECURSIVE t(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM t) SELECT * FROM t",
        ] {
            assert!(
                matches!(
                    validate_select_query(query),
                    Err(CalcFlowError::InvalidArgument { field, .. }) if field == "query"
                ),
                "query was accepted: {query}"
            );
        }
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
