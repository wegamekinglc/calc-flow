use super::*;
use datafusion::arrow::{
    array::{BooleanArray, Int64Array, NullArray, StringArray},
    datatypes::{Field, Schema},
    record_batch::{RecordBatch, RecordBatchOptions},
};
use std::sync::Arc;

fn record(array: ArrayRef) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "value",
            array.data_type().clone(),
            true,
        )])),
        vec![array],
    )
    .unwrap()
}

fn assert_index_allocation(source: &RecordBatch, offset: usize, bytes_per_index: usize) {
    let count = 32_768;
    let mut gather = None;
    let allocations = allocation_counter::measure(|| {
        gather = SideGather::new((0..count).map(|_| PayloadRef::Record(source, offset)));
    });
    assert!(gather.is_some());
    assert!(
        allocations.bytes_total <= (count * bytes_per_index + 256) as u64,
        "index buffers must be built directly at the selected width: {allocations:?}"
    );
    assert_eq!(gather.unwrap().selected_column(0).unwrap().len(), count);
}

#[test]
fn test_single_source_builds_narrow_indices_without_a_wide_temporary() {
    let source = record(Arc::new(Int64Array::from(vec![7, 9])));
    assert_index_allocation(&source, 1, size_of::<u32>());
}

#[test]
#[cfg(target_pointer_width = "64")]
fn test_source_coordinate_boundary_selects_narrow_or_wide_without_row_buffers() {
    let limit = usize::try_from(u32::MAX).unwrap();
    // NullArray stores only its length, so these boundaries need no row buffers.
    let narrow = record(Arc::new(NullArray::new(limit + 1)));
    assert_index_allocation(&narrow, limit, size_of::<u32>());
    let gathered = SideGather::new(std::iter::once(PayloadRef::Record(&narrow, limit))).unwrap();
    let SideGather::Single {
        rows: Indices::Narrow(rows),
        ..
    } = gathered
    else {
        panic!("u32::MAX is a valid narrow source coordinate");
    };
    assert_eq!(rows.values().as_ref(), &[u32::MAX]);
    let wide = record(Arc::new(NullArray::new(limit + 2)));
    assert_index_allocation(&wide, limit + 1, size_of::<u64>());
    // A short output with a small first index does not bound the source domain.
    let gathered = SideGather::new(
        [0, limit + 1, limit]
            .into_iter()
            .map(|row| PayloadRef::Record(&wide, row)),
    )
    .unwrap();
    let SideGather::Single {
        rows: Indices::Wide(rows),
        ..
    } = gathered
    else {
        panic!("the full source bound cannot be proven narrow");
    };
    assert_eq!(
        rows.values().as_ref(),
        &[0, u64::from(u32::MAX) + 1, u64::from(u32::MAX)]
    );
}

#[test]
fn test_unknown_source_bound_keeps_wide_indices() {
    let source = RecordBatch::try_new_with_options(
        Arc::new(Schema::empty()),
        vec![],
        &RecordBatchOptions::new().with_row_count(Some(5)),
    )
    .unwrap();
    let gathered = SideGather::new(
        [4, 1, 4]
            .into_iter()
            .map(|row| PayloadRef::Record(&source, row)),
    )
    .unwrap();
    let SideGather::Single {
        rows: Indices::Wide(rows),
        ..
    } = gathered
    else {
        panic!("no column can prove a narrow source domain");
    };
    assert_eq!(rows.values().as_ref(), &[4, 1, 4]);
}

#[test]
fn test_narrow_indices_use_logical_slice_offsets_and_preserve_nulls() {
    let flags: ArrayRef = Arc::new(BooleanArray::from(vec![
        false, true, false, true, false, true, false, true,
    ]));
    let text: ArrayRef = Arc::new(StringArray::from(vec![
        Some("a"),
        None,
        Some("c"),
        None,
        Some("e"),
        Some("f"),
        None,
        Some("h"),
    ]));
    let source = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("flag", DataType::Boolean, false),
            Field::new("text", DataType::Utf8, true),
        ])),
        vec![flags, text],
    )
    .unwrap()
    .slice(2, 5);
    let original = source.clone();
    assert_eq!(source.column(0).offset(), 2);
    let gather = SideGather::new(
        [3, 0, 3, 1, 4]
            .into_iter()
            .map(|row| PayloadRef::Record(&source, row)),
    )
    .unwrap();
    let flags = gather.selected_column(0).unwrap();
    assert_eq!(
        flags.as_any().downcast_ref::<BooleanArray>().unwrap(),
        &BooleanArray::from(vec![true, false, true, true, false]),
    );
    let text = gather.selected_column(1).unwrap();
    assert_eq!(
        text.as_any().downcast_ref::<StringArray>().unwrap(),
        &StringArray::from(vec![Some("f"), Some("c"), Some("f"), None, None]),
    );
    assert_eq!(source, original);
}

#[test]
fn test_narrow_prefix_promotes_to_multiple_sources_in_order_with_duplicates() {
    let first = record(Arc::new(Int64Array::from(vec![11, 13, 17])));
    let second = record(Arc::new(Int64Array::from(vec![19, 23])));
    let payloads = [
        (&first, 2),
        (&first, 0),
        (&first, 2),
        (&second, 1),
        (&first, 0),
        (&second, 0),
        (&second, 1),
    ];
    let gather = SideGather::new(
        payloads
            .into_iter()
            .map(|(source, row)| PayloadRef::Record(source, row)),
    )
    .unwrap();
    let SideGather::Multiple { columns, rows } = &gather else {
        panic!("a source transition must preserve the collected narrow prefix");
    };
    assert_eq!(columns.len(), 2);
    assert_eq!(
        rows,
        &[(0, 2), (0, 0), (0, 2), (1, 1), (0, 0), (1, 0), (1, 1)]
    );
    assert_eq!(
        gather
            .selected_column(0)
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap(),
        &Int64Array::from(vec![17, 11, 17, 23, 11, 19, 23]),
    );
}

#[test]
#[cfg(target_pointer_width = "64")]
fn test_wide_prefix_promotes_to_multiple_sources_without_truncation() {
    let large = usize::try_from(u32::MAX).unwrap() + 1;
    let first = record(Arc::new(NullArray::new(large + 1)));
    let second = record(Arc::new(NullArray::new(2)));
    let gather = SideGather::new(
        [(&first, large), (&first, 0), (&second, 1), (&first, large)]
            .into_iter()
            .map(|(source, row)| PayloadRef::Record(source, row)),
    )
    .unwrap();
    let SideGather::Multiple { columns, rows } = &gather else {
        panic!("a source transition must preserve the collected wide prefix");
    };
    assert_eq!(columns.len(), 2);
    assert_eq!(rows, &[(0, large), (0, 0), (1, 1), (0, large)]);
    assert_eq!(gather.selected_column(0).unwrap().len(), 4);
}
