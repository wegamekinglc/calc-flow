use super::*;
use std::sync::Arc;

use datafusion::arrow::array::{ArrayRef, Float64Array};

#[test]
fn legacy_offset_charges_keep_join_caller_behavior() {
    let offsets = super::super::row_cost::Offsets::Narrow(&[0, 1, 5, 5]);
    let mut charges = vec![0; 3];
    offsets.add_charges(&mut charges).unwrap();
    assert_eq!(charges, [1, 4, 0]);
}

#[test]
fn variable_row_cost_materialization_checks_cancellation_during_scan() {
    use datafusion::arrow::array::StringArray;
    let array = StringArray::from(vec!["payload"; 8193]);
    let mut charges = ColumnCharges::default();
    assert!(charges.add_flat_column(&array).unwrap());
    let mut calls = 0;
    let result = charges.materialize_checked(array.len(), &mut || {
        calls += 1;
        if calls == 3 {
            Err(CalcFlowError::Cancelled {
                run_id: "variable-cost".into(),
            })
        } else {
            Ok(())
        }
    });
    assert!(
        matches!(result, Err(CalcFlowError::Cancelled { run_id }) if run_id == "variable-cost")
    );
}

#[test]
fn checked_chunking_preserves_nullable_row_ranges_and_sequences() {
    let record = RecordBatch::try_from_iter([(
        "price",
        Arc::new(Float64Array::from(vec![
            Some(0.0),
            None,
            Some(2.0),
            Some(3.0),
            Some(4.0),
            None,
        ])) as ArrayRef,
    )])
    .unwrap();
    let budget = EdgeBudget::new(2, 17).unwrap();
    let chunks: Vec<Batch> = chunk_output_record_checked(
        &record,
        "features",
        41,
        budget,
        OutputChunkErrors::ROLLING,
        &mut || Ok(()),
    )
    .unwrap();
    assert_eq!(
        chunks.iter().map(Batch::num_rows).collect::<Vec<_>>(),
        [2, 2, 2]
    );
    assert_eq!(
        chunks
            .iter()
            .map(|batch| batch.metadata().sequence())
            .collect::<Vec<_>>(),
        [41, 42, 43]
    );
    for chunk in chunks {
        assert!(chunk.estimated_bytes().unwrap() <= budget.max_bytes);
    }
}

#[test]
fn checked_chunking_preserves_cancellation_before_returning_partial_chunks() {
    let record = RecordBatch::try_from_iter([(
        "price",
        Arc::new(Float64Array::from_iter_values((0..8193_u32).map(f64::from))) as ArrayRef,
    )])
    .unwrap();
    let mut calls = 0;
    let result: Result<Vec<Batch>> = chunk_output_record_checked(
        &record,
        "features",
        0,
        EdgeBudget::new(1, 16).unwrap(),
        OutputChunkErrors::ROLLING,
        &mut || {
            calls += 1;
            if calls == 4 {
                Err(CalcFlowError::Cancelled {
                    run_id: "chunk-cancel".into(),
                })
            } else {
                Ok(())
            }
        },
    );
    assert!(matches!(result, Err(CalcFlowError::Cancelled { run_id }) if run_id == "chunk-cancel"));
}
