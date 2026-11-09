use std::sync::Arc;

use datafusion::arrow::{
    array::{ArrayRef, BinaryArray, UInt8Array, UInt64Array},
    datatypes::{DataType, Field, Schema},
    ipc::writer::FileWriter,
    record_batch::RecordBatch,
};

use super::read;
use crate::CalcFlowError;

fn fixture(payload: &[u8]) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("_state_kind", DataType::UInt8, false),
        Field::new("_entity_id", DataType::UInt64, false),
        Field::new("_entity_position", DataType::UInt64, true),
        Field::new("payload", DataType::Binary, true),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(UInt8Array::from(vec![0, 2])),
        Arc::new(UInt64Array::from(vec![0, 0])),
        Arc::new(UInt64Array::from(vec![Some(0), None])),
        Arc::new(BinaryArray::from(vec![None, Some(payload)])),
    ];
    RecordBatch::try_new(schema, columns).unwrap()
}
fn encode(record: &RecordBatch) -> Vec<u8> {
    let mut encoded = Vec::new();
    let mut writer = FileWriter::try_new(&mut encoded, &record.schema()).unwrap();
    writer.write(record).unwrap();
    writer.finish().unwrap();
    drop(writer);
    encoded
}

#[test]
fn test_reader_budget_rejects_before_copying_a_large_block() {
    let record = fixture(&vec![7; 4 << 20]);
    let encoded = encode(&record);
    let schema = record.schema();
    let mut result = None;
    let measured = allocation_counter::measure(|| {
        result = Some(read(
            &encoded,
            &schema,
            2,
            &mut |bytes| {
                if bytes > 64 << 10 {
                    Err(CalcFlowError::CheckpointMismatch {
                        message: "state.peak_bytes".into(),
                    })
                } else {
                    Ok(())
                }
            },
            &mut || Ok(()),
        ));
    });
    let error = result.unwrap().err().unwrap();
    assert!(
        matches!(error, CalcFlowError::CheckpointMismatch { message } if message == "state.peak_bytes")
    );
    assert!(measured.bytes_max < 64 << 10, "{measured:?}");
    assert_eq!(record.num_rows(), 2);
}

#[test]
fn test_reader_keeps_retained_quote_and_preserves_cancelled_identity() {
    let record = fixture(&[3; 32]);
    let encoded = encode(&record);
    let schema = record.schema();
    let mut quotes = Vec::new();
    let decoded = read(
        &encoded,
        &schema,
        2,
        &mut |bytes| {
            quotes.push(bytes);
            Ok(())
        },
        &mut || Ok(()),
    )
    .unwrap();
    assert_eq!(decoded.record, record);
    assert_eq!(quotes.last(), Some(&decoded.reader_bytes));
    assert!(quotes[0] > decoded.reader_bytes);
    let admitted = std::cell::Cell::new(false);
    let error = read(
        &encoded,
        &schema,
        2,
        &mut |_| {
            admitted.set(true);
            Ok(())
        },
        &mut || {
            if admitted.get() {
                Err(CalcFlowError::Cancelled {
                    run_id: "reader".into(),
                })
            } else {
                Ok(())
            }
        },
    )
    .err()
    .unwrap();
    assert!(matches!(error, CalcFlowError::Cancelled { run_id } if run_id == "reader"));
}
