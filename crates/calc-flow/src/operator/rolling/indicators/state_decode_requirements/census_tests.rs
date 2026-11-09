use super::*;
use crate::CalcFlowError;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::arrow::{
    array::{ArrayRef, UInt8Array, UInt32Array, UInt64Array},
    datatypes::{DataType, Field, Schema},
};
use std::sync::Arc;

fn record(
    kinds: Vec<Option<u8>>,
    entities: Vec<Option<u64>>,
    positions: Vec<Option<u64>>,
) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("_state_kind", DataType::UInt8, true),
            Field::new("_entity_id", DataType::UInt64, true),
            Field::new("_entity_position", DataType::UInt64, true),
        ])),
        vec![
            Arc::new(UInt8Array::from(kinds)),
            Arc::new(UInt64Array::from(entities)),
            Arc::new(UInt64Array::from(positions)),
        ],
    )
    .unwrap()
}
#[test]
fn test_restore_census_reads_actual_control_rows_without_allocating() {
    let input = record(
        vec![
            Some(0),
            Some(0),
            Some(1),
            Some(1),
            Some(2),
            Some(3),
            Some(4),
            Some(5),
        ],
        vec![
            Some(0),
            Some(1),
            Some(0),
            Some(0),
            Some(1),
            Some(0),
            Some(0),
            Some(0),
        ],
        vec![
            Some(3),
            Some(0),
            Some(0),
            Some(1),
            None,
            None,
            Some(0),
            None,
        ],
    );
    let mut result = None;
    let allocations = allocation_counter::measure(|| {
        result = Some(scan(&input, &mut || Ok(())).unwrap());
    });
    assert_eq!(allocations.count_total, 0);
    let census = result.unwrap();
    assert_eq!((census.entities, census.initialized), (2, 1));
    assert_eq!(
        (
            census.histories,
            census.pending,
            census.ewma,
            census.cumulative,
            census.indicators
        ),
        (2, 1, 1, 1, 1)
    );
    assert_eq!(census.max_history, 2);
    assert_eq!(census.retained_rows().unwrap(), 4);
}
#[test]
fn test_restore_census_rejects_bad_kind_ids_positions_and_zero_count_history() {
    for input in [
        record(
            vec![Some(0), Some(1)],
            vec![Some(0), Some(0)],
            vec![Some(1), Some(1)],
        ),
        record(
            vec![Some(0), Some(1)],
            vec![Some(0), Some(1)],
            vec![Some(1), Some(0)],
        ),
        record(
            vec![Some(0), Some(1)],
            vec![Some(0), Some(0)],
            vec![Some(0), Some(0)],
        ),
        record(
            vec![Some(0), Some(5), Some(3)],
            vec![Some(0); 3],
            vec![Some(1), None, None],
        ),
        record(
            vec![Some(0), Some(6)],
            vec![Some(0); 2],
            vec![Some(1), None],
        ),
        record(vec![None], vec![Some(u64::MAX)], vec![Some(0)]),
    ] {
        assert!(scan(&input, &mut || Ok(())).is_err());
    }
    let empty = record(vec![], vec![], vec![]);
    let error = scan(&empty, &mut || {
        Err(CalcFlowError::Cancelled {
            run_id: "decode".into(),
        })
    })
    .err()
    .unwrap();
    assert!(matches!(error, CalcFlowError::Cancelled { .. }));
}
#[test]
fn test_restore_census_rejects_wrong_physical_control_type() {
    let column: ArrayRef = Arc::new(UInt32Array::from(vec![0]));
    let input = RecordBatch::try_from_iter([("kind", column)]).unwrap();
    assert!(scan(&input, &mut || Ok(())).is_err());
}
