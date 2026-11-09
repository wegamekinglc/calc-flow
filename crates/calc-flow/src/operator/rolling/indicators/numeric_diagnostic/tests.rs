use super::{Coordinate, numeric_error};
use crate::CalcFlowError;

fn domain() -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: "high".into(),
        message: "must be greater than or equal to low".into(),
    }
}

#[test]
fn test_unordered_coordinates_do_not_claim_finalized_positions() {
    for (coordinate, label) in [
        (Coordinate::InputRow(0), "input row 0"),
        (Coordinate::PendingRow(2), "pending row 2"),
        (Coordinate::AcceptedRow(1), "accepted row 1"),
    ] {
        let error = numeric_error(domain(), "features", coordinate);
        let CalcFlowError::InvalidArgument { field, message } = error else {
            panic!("{error}");
        };
        assert_eq!(field, "high");
        assert_eq!(
            message,
            format!(
                "rolling node \"features\", {label}; ordered position not assigned: must be greater than or equal to low"
            )
        );
    }
}

#[test]
fn test_finalized_coordinate_keeps_the_existing_ordered_grammar() {
    let error = numeric_error(domain(), "features", Coordinate::OrderedPosition(7));
    let CalcFlowError::InvalidArgument { message, .. } = error else {
        panic!("{error}");
    };
    assert_eq!(
        message,
        "rolling node \"features\", ordered position 7: must be greater than or equal to low"
    );
}

#[test]
fn test_non_numeric_error_preserves_its_cancellation_identity() {
    let error = numeric_error(
        CalcFlowError::Cancelled {
            run_id: "actual-job".into(),
        },
        "features",
        Coordinate::InputRow(0),
    );
    assert!(matches!(error, CalcFlowError::Cancelled { run_id } if run_id == "actual-job"));
}
