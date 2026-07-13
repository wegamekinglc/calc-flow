use calc_flow::{CalcFlowError, split_assignment, sql_projection, validate_select_query};

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
