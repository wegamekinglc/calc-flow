use super::super::ir::DType;
use super::pack;
use crate::{Batch, BatchMetadata, CalcFlowError, array::ArrayExecutionOptions};
use datafusion::arrow::{
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use std::sync::Arc;

#[test]
fn test_table_nullability_scan_observes_cancellation_across_empty_chunks() {
    let record = RecordBatch::new_empty(Arc::new(Schema::new(vec![Field::new(
        "x",
        DataType::Float64,
        true,
    )])));
    let input = Batch::table(vec![record; 4097], BatchMetadata::default()).unwrap();
    let mut checks = 0;
    let mut check = || {
        checks += 1;
        if checks == 2 {
            Err(CalcFlowError::Cancelled {
                run_id: "scan".into(),
            })
        } else {
            Ok(())
        }
    };
    let result = pack::prepare(
        &input,
        &[0],
        DType::Float64,
        &ArrayExecutionOptions::default(),
        &mut check,
    );
    assert!(matches!(result, Err(CalcFlowError::Cancelled { .. })));
    assert_eq!(checks, 2);
}

#[test]
fn test_table_packing_observes_cancellation_across_empty_chunks() {
    let record = RecordBatch::new_empty(Arc::new(Schema::new(vec![Field::new(
        "x",
        DataType::Float64,
        true,
    )])));
    let input = Batch::table(vec![record; 8193], BatchMetadata::default()).unwrap();
    let prepared = pack::prepare(
        &input,
        &[0],
        DType::Float64,
        &ArrayExecutionOptions::default(),
        &mut || Ok(()),
    )
    .unwrap();
    let mut checks = 0;
    let mut check = || {
        checks += 1;
        if checks == 4 {
            Err(CalcFlowError::Cancelled {
                run_id: "copy".into(),
            })
        } else {
            Ok(())
        }
    };
    let result = pack::pack(&input, &[0], prepared, &mut check);
    assert!(matches!(result, Err(CalcFlowError::Cancelled { .. })));
    assert_eq!(checks, 4);
}
