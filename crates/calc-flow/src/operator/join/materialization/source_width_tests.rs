use super::*;
use crate::EventTime;
use datafusion::arrow::{
    array::{ArrayRef, Int64Array, StringArray},
    datatypes::{DataType, Field, Schema},
};
use std::cell::Cell;

struct RecordPositions<'a>(&'a RecordBatch);

impl AdmittedRows for RecordPositions<'_> {
    fn len(&self) -> usize {
        self.0.num_rows()
    }

    fn row(&self, index: usize) -> super::super::admitted::AdmittedRef<'_> {
        super::super::admitted::AdmittedRef {
            record: self.payload(index),
            event_time: EventTime::from_micros(100),
            row_id: index as u64,
            retain: true,
        }
    }

    fn payload(&self, index: usize) -> PayloadRef<'_> {
        PayloadRef::Record(self.0, index)
    }

    fn group_end(&self, _start: usize) -> usize {
        self.len()
    }

    fn source_index(&self, index: usize) -> usize {
        index
    }
}

fn integer_record(rows: usize) -> RecordBatch {
    RecordBatch::try_from_iter(vec![(
        "value",
        Arc::new(Int64Array::from(vec![7; rows])) as ArrayRef,
    )])
    .unwrap()
}

fn stored(record: RecordBatch) -> StoredRow {
    StoredRow {
        record: record.into(),
        event_time: EventTime::from_micros(100),
        row_id: 0,
        charge: 0,
        encoded_key: Arc::new(vec![].into()),
    }
}

fn pair_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("left_value", DataType::Int64, false),
        Field::new("right_value", DataType::Int64, false),
    ]))
}

fn output<'a>(
    schema: &'a SchemaRef,
    projection: Option<&'a Arc<[usize]>>,
    admitted: &'a dyn AdmittedRows,
    opposite: &'a dyn Rows,
    matched: &'a dyn super::super::matches::Matches,
) -> JoinOutput<'a> {
    JoinOutput {
        schema,
        projection,
        admitted,
        opposite,
        matched,
        incoming_is_left: true,
        operator_id: "source-width",
        admitted_charges: None,
    }
}

fn matching_rows(rows: usize) -> Vec<MatchedPair> {
    (0..rows)
        .map(|pos| MatchedPair {
            pos,
            opposite_index: 0,
        })
        .collect()
}

fn reference_ranges(output: &JoinOutput<'_>, budget: EdgeBudget) -> Vec<Range<usize>> {
    let fast = output.ranges(budget).unwrap();
    let slow = {
        let _scan = tests::PerPairScan::force();
        output.ranges(budget).unwrap()
    };
    assert_eq!(fast, slow);
    fast
}

#[test]
fn test_constant_width_preflight_checks_one_source_for_a_million_distinct_positions() {
    let rows = 1_000_000;
    let record = integer_record(rows);
    let schema = pair_schema();
    let admitted = RecordPositions(&record);
    let opposite = vec![stored(integer_record(1))];
    let matched = (0..rows)
        .map(|pos| MatchedPair {
            pos,
            opposite_index: 0,
        })
        .collect::<Vec<_>>();
    let output = JoinOutput {
        schema: &schema,
        projection: None,
        admitted: &admitted,
        opposite: &opposite,
        matched: &matched,
        incoming_is_left: true,
        operator_id: "source-width",
        admitted_charges: None,
    };
    tests::SIDE_VISITS.with(|count| count.set(0));
    tests::FLAT_VISITS.with(|count| count.set(0));
    let measured = allocation_counter::measure(|| {
        let ranges = output
            .ranges(EdgeBudget::new(100_000, 1 << 20).unwrap())
            .unwrap();
        assert_eq!(ranges.first(), Some(&(0..65_536)));
        assert_eq!(ranges.last(), Some(&(983_040..rows)));
        assert_eq!(ranges.iter().map(Range::len).sum::<usize>(), rows);
    });
    assert_eq!(tests::FLAT_VISITS.with(Cell::get), 0);
    assert_eq!(
        tests::SIDE_VISITS.with(Cell::get),
        0,
        "fixed non-null output schemas need no source-row width checks"
    );
    assert!(
        measured.count_total < 16,
        "source-width caching must not allocate per matched position: {measured:?}"
    );
}

#[test]
fn test_constant_width_preflight_checks_each_source_transition() {
    use super::super::columnar::RowPayload;

    let parents = [Arc::new(integer_record(2)), Arc::new(integer_record(2))];
    let admitted = [0, 0, 1, 1, 0, 0]
        .into_iter()
        .enumerate()
        .map(|(index, source)| AdmittedRow {
            record: RowPayload::Rowed {
                parent: Arc::clone(&parents[source]),
                row: index % 2,
            },
            event_time: EventTime::from_micros(100),
            row_id: index as u64,
            retain: true,
        })
        .collect::<Vec<_>>();
    let schema = pair_schema();
    let opposite = vec![stored(integer_record(1))];
    let matched = matching_rows(admitted.len());
    let output = output(&schema, None, &admitted, &opposite, &matched);
    tests::SIDE_VISITS.with(|count| count.set(0));
    let ranges = reference_ranges(&output, EdgeBudget::new(10, 32).unwrap());
    assert_eq!(ranges, vec![0..2, 2..4, 4..6]);
    assert_eq!(
        tests::SIDE_VISITS.with(Cell::get),
        0,
        "fixed non-null output schemas need no source-transition checks"
    );
}

#[test]
fn test_constant_width_preflight_distinguishes_nullable_slices_of_one_parent() {
    let parent = RecordBatch::try_from_iter(vec![(
        "value",
        Arc::new(Int64Array::from(vec![Some(7), None])) as ArrayRef,
    )])
    .unwrap();
    let admitted = (0..2)
        .map(|index| AdmittedRow {
            record: parent.slice(index, 1).into(),
            event_time: EventTime::from_micros(100),
            row_id: index as u64,
            retain: true,
        })
        .collect::<Vec<_>>();
    let schema = Arc::new(Schema::new(vec![
        Field::new("left_value", DataType::Int64, true),
        Field::new("right_value", DataType::Int64, false),
    ]));
    let opposite = vec![stored(integer_record(1))];
    let matched = matching_rows(2);
    let output = output(&schema, None, &admitted, &opposite, &matched);
    assert_eq!(output.constant_row_bytes(), None);
    let ranges = reference_ranges(&output, EdgeBudget::new(2, 32).unwrap());
    assert_eq!(ranges, vec![0..1, 1..2]);
    let materialized = output.materialize(0..2).unwrap();
    let values = materialized
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(values.iter().collect::<Vec<_>>(), vec![Some(7), None]);
}

fn projected_input() -> RecordBatch {
    RecordBatch::try_from_iter(vec![
        (
            "key",
            Arc::new(StringArray::from(vec!["a", "longer-key"])) as ArrayRef,
        ),
        ("value", Arc::new(Int64Array::from(vec![7, 9])) as ArrayRef),
    ])
    .unwrap()
}

#[test]
fn test_constant_width_preflight_prices_duplicate_selected_columns_only() {
    let record = projected_input();
    let admitted = RecordPositions(&record);
    let opposite = vec![stored(integer_record(1))];
    let matched = matching_rows(2);
    let schema = pair_schema();
    let projection: Arc<[usize]> = Arc::from([1, 1]);
    let output = output(&schema, Some(&projection), &admitted, &opposite, &matched);
    assert_eq!(output.constant_row_bytes(), Some(16));
    let ranges = reference_ranges(&output, EdgeBudget::new(2, 16).unwrap());
    assert_eq!(ranges, vec![0..1, 1..2]);
    let materialized = output.materialize(0..2).unwrap();
    for column in materialized.columns() {
        let values = column.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(values.iter().collect::<Vec<_>>(), vec![Some(7), Some(9)]);
    }
}

#[test]
fn test_constant_width_preflight_preserves_empty_projection_and_selection() {
    let record = projected_input();
    let admitted = RecordPositions(&record);
    let opposite = vec![stored(integer_record(1))];
    let matched = matching_rows(2);
    let schema = Arc::new(Schema::empty());
    let projection: Arc<[usize]> = Arc::from([]);
    let output = output(&schema, Some(&projection), &admitted, &opposite, &matched);
    assert_eq!(output.constant_row_bytes(), Some(0));
    let budget = EdgeBudget::new(1, 1).unwrap();
    assert_eq!(reference_ranges(&output, budget), vec![0..1, 1..2]);
    let materialized = output.materialize(0..2).unwrap();
    assert_eq!(materialized.num_columns(), 0);
    assert_eq!(materialized.num_rows(), 2);
    let empty_matched: [MatchedPair; 0] = [];
    let empty = JoinOutput {
        matched: &empty_matched,
        ..output
    };
    assert!(reference_ranges(&empty, budget).is_empty());
    assert_eq!(empty.materialize(0..0).unwrap().num_rows(), 0);
}
