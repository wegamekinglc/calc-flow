use calc_flow::{split_assignment, sql_projection, validate_select_query};

#[test]
fn assignment_ignores_comparisons() {
    assert_eq!(split_assignment("total = a + b"), Some(("total", "a + b")));
    assert_eq!(split_assignment("a == b"), None);
    assert_eq!(split_assignment("a >= b"), None);
}

#[test]
fn projection_and_query_validation_are_restricted() {
    assert_eq!(
        sql_projection("total = a + b", "input").unwrap(),
        "SELECT *, (a + b) AS total FROM input"
    );
    assert!(validate_select_query("WITH x AS (SELECT 1) SELECT * FROM x").is_ok());
    assert!(validate_select_query("DROP TABLE input").is_err());
    assert!(validate_select_query("SELECT 1; SELECT 2").is_err());
}
