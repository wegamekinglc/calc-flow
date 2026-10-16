use super::{FundedMorsel, JoinOutput, funding};
use crate::operator::join::{AdmittedRow, MatchedPair, StoredRow, columnar::RowPayload};
use crate::{EdgeBudget, time::EventTime};
use datafusion::arrow::{
    array::{
        ArrayRef, BinaryArray, LargeBinaryArray, LargeStringArray, StringArray, new_null_array,
    },
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool};
use std::sync::Arc;

fn source(text: &str) -> Arc<RecordBatch> {
    let types = [
        DataType::Float64,
        DataType::Float32,
        DataType::Decimal128(38, 2),
        DataType::Decimal256(76, 2),
        DataType::FixedSizeBinary(1024),
        DataType::Boolean,
        DataType::Null,
    ];
    let mut columns = types
        .iter()
        .map(|data_type| new_null_array(data_type, 5))
        .collect::<Vec<_>>();
    let padding = "unselected".repeat(16_384);
    let texts = [
        Some(padding.as_str()),
        Some(text),
        None,
        Some("中"),
        Some(padding.as_str()),
    ];
    let bytes = texts.map(|text| text.map(str::as_bytes));
    columns.extend([
        Arc::new(StringArray::from(texts.to_vec())) as ArrayRef,
        Arc::new(LargeStringArray::from(texts.to_vec())),
        Arc::new(BinaryArray::from(bytes.to_vec())),
        Arc::new(LargeBinaryArray::from(bytes.to_vec())),
    ]);
    Arc::new(
        RecordBatch::try_from_iter(
            columns
                .into_iter()
                .enumerate()
                .map(|(index, array)| (format!("c{index}"), array)),
        )
        .unwrap()
        .slice(1, 3),
    )
}

fn admitted(sources: &[Arc<RecordBatch>]) -> Vec<AdmittedRow> {
    (0..3)
        .map(|index| AdmittedRow {
            record: RowPayload::Rowed {
                parent: Arc::clone(&sources[index % sources.len()]),
                row: index,
            },
            event_time: EventTime::from_micros(0),
            row_id: index as u64,
            retain: true,
        })
        .collect()
}

fn opposite(sources: &[Arc<RecordBatch>]) -> Vec<StoredRow> {
    admitted(sources)
        .into_iter()
        .map(|row| StoredRow {
            record: row.record,
            event_time: row.event_time,
            row_id: row.row_id,
            charge: 1,
            encoded_key: Arc::new(vec![].into()),
        })
        .collect()
}

#[test]
fn test_morsel_physical_funding_covers_flat_take_interleave_and_projection() {
    for multiple in [false, true] {
        for projected in [false, true] {
            check_physical_funding(multiple, projected);
        }
    }
}

fn check_physical_funding(multiple: bool, projected: bool) {
    let mut sources = vec![source(&"selected".repeat(64))];
    if multiple {
        sources.push(source(&"other".repeat(32)));
    }
    let incoming = admitted(&sources);
    let opposite = opposite(&sources);
    let pairs = (0..512)
        .map(|index| MatchedPair {
            pos: index % 3,
            opposite_index: (index + 1) % 3,
        })
        .collect::<Vec<_>>();
    let fields = (0..22)
        .map(|index| {
            Field::new(
                format!("out_{index}"),
                sources[0].column(index % 11).data_type().clone(),
                true,
            )
        })
        .collect::<Vec<_>>();
    let schema = Schema::new(fields);
    let projection = projected.then(|| Arc::from([21, 0, 21, 9]));
    let schema = Arc::new(if projected {
        schema.project(&[21, 0, 21, 9]).unwrap()
    } else {
        schema
    });
    let output = JoinOutput {
        schema: &schema,
        projection: projection.as_ref(),
        admitted: &incoming,
        opposite: &opposite,
        matched: &pairs,
        incoming_is_left: true,
        operator_id: "physical-funding",
        admitted_charges: None,
    };
    let ranges = output
        .ranges(EdgeBudget::new(137, 1 << 20).unwrap())
        .unwrap();
    let lease = funding::output_bytes(&output, ranges.len()).expect("selected buffers fit cap");
    let records = ranges
        .into_iter()
        .map(|range| output.materialize(range).unwrap())
        .collect::<Vec<_>>();
    let backing = records
        .iter()
        .map(RecordBatch::get_array_memory_size)
        .sum::<usize>();
    assert!(
        backing <= lease,
        "take/interleave {multiple}, projection {projected}: {backing} > {lease}"
    );
    assert_eq!(sources[0], source(&"selected".repeat(64)));
}

#[test]
fn test_morsel_actual_capacity_refusal_releases_backing_and_credit() {
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(16 << 20));
    let credit = MemoryConsumer::new("physical-capacity").register(&pool);
    credit.try_grow(1).unwrap();
    let record = source("result");
    let weak = Arc::downgrade(record.column(0));
    let record = Arc::unwrap_or_clone(record);
    assert!(FundedMorsel::new(vec![record], credit).is_none());
    assert!(weak.upgrade().is_none());
    assert_eq!(pool.reserved(), 0);
}
