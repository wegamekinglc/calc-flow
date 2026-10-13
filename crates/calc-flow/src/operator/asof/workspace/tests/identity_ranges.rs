use super::*;
use datafusion::arrow::buffer::NullBuffer;

fn strings() -> [ArrayRef; 2] {
    let values = [
        0, 1, 7, 31, 32, 33, 55, 56, 57, 63, 64, 65, 127, 128, 129, 4_096, 48_000,
    ]
    .map(|length| "x".repeat(length));
    let valid = NullBuffer::from(
        (0..values.len())
            .map(|row| row % 4 != 2)
            .collect::<Vec<_>>(),
    );
    let small = StringArray::from_iter_values(&values);
    let large = LargeStringArray::from_iter_values(&values);
    [
        Arc::new(StringArray::new(
            small.offsets().clone(),
            small.values().clone(),
            Some(valid.clone()),
        )),
        Arc::new(LargeStringArray::new(
            large.offsets().clone(),
            large.values().clone(),
            Some(valid),
        )),
    ]
}

fn check_ranges(array: ArrayRef) {
    let rows = array.len();
    let columns = vec![
        (ColumnWorkspace::new(array.clone()).unwrap(), true, false),
        (ColumnWorkspace::new(array).unwrap(), true, true),
        (
            ColumnWorkspace::new(Arc::new(UInt64Array::from_iter_values(0..rows as u64))).unwrap(),
            false,
            true,
        ),
    ];
    for start in 0..=rows {
        for end in start..=rows {
            let expected = (start..end)
                .map(|row| identity_row_workspace(&columns, row, "asof").unwrap())
                .sum::<u64>();
            VARIABLE_LENGTH_CALLS.with(|calls| calls.set(0));
            let actual = identity_range_workspace(&columns, start..end, "asof").unwrap();
            let calls = VARIABLE_LENGTH_CALLS.with(|calls| calls.replace(0));
            assert_eq!(actual, expected, "range {start}..{end}");
            assert!(calls <= columns.len(), "scalar dispatches: {calls}");
        }
    }
}

fn check_overflow() {
    let column = ColumnWorkspace {
        column: Arc::new(StringArray::from(vec!["x"])),
        fixed: u64::MAX,
    };
    let columns = vec![(column, true, false)];
    let result = identity_range_workspace(&columns, 0..1, "range-node");
    assert!(matches!(
        result,
        Err(CalcFlowError::OperatorReason {
            node_id,
            reason_code: StreamingFailureReason::AsofCounterOverflow,
            message,
        }) if node_id == "range-node" && message == "ASOF column workspace arithmetic overflowed"
    ));
}

pub(super) fn check_string_ranges() {
    for array in strings() {
        check_ranges(array.clone());
        check_ranges(array.slice(1, array.len() - 2));
        check_ranges(array.slice(2, 0));
    }
    check_overflow();
}
