use super::*;

fn input(rows: usize, columns: usize) -> MaterializationInput {
    let length = i64::try_from(rows).unwrap();
    let left: ArrayRef = Arc::new(Int64Array::from_iter_values(0..length));
    let right: ArrayRef = Arc::new(Int64Array::from_iter_values(0..length));
    MaterializationInput {
        rows: OutputPlan {
            left: OutputSide {
                batches: vec![vec![left]],
                positions: Vec::new().into(),
                spans: vec![Span {
                    source: 0,
                    start: 0,
                    end: rows,
                }],
                has_nulls: false,
            },
            right: OutputSide {
                batches: vec![vec![right]],
                positions: (0..rows).map(|row| (1, row / 2)).collect(),
                spans: Vec::new(),
                has_nulls: false,
            },
            len: rows,
            matched: rows as u64,
            raw_bytes: rows as u64 * 16,
        },
        requests: (2 - columns..2)
            .map(|index| ColumnRequest {
                index,
                data_type: DataType::Int64,
            })
            .collect(),
        worker_gate: None,
        worker_probe: None,
    }
}

#[test]
fn cpu_pipeline_row_capacity_is_independent_of_output_column_count() {
    for columns in [1, 2] {
        let input = input(262_144, columns);
        assert!(
            input.parallelism() > columns,
            "row work is capped by output columns"
        );
        let shape = input
            .row_gather()
            .unwrap()
            .expect("single copied primitive column");
        assert_eq!(
            (shape.ordinal, shape.rows, shape.width),
            (columns - 1, 262_144, 8)
        );
    }
}

#[test]
fn cpu_pipeline_small_output_avoids_row_tasks() {
    assert!(input(128, 2).row_gather().unwrap().is_none());
}
