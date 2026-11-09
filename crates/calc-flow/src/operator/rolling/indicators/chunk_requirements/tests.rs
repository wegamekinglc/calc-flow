use super::*;
use datafusion::arrow::{
    array::{ArrayRef, Float64Array, ListArray, StringArray},
    buffer::OffsetBuffer,
    datatypes::{DataType, Field},
};
use std::sync::Arc;

fn compare(record: &RecordBatch, budget: EdgeBudget) -> (u64, u64) {
    let mut cost = None;
    let planning = allocation_counter::measure(|| {
        cost = Some(plan(record, budget, "features", &mut || Ok(())).unwrap());
    });
    assert_eq!(planning.bytes_max, 0);
    let cost = cost.unwrap();
    let mut output = None;
    let observed = allocation_counter::measure(|| {
        output = Some(
            crate::operator::output_chunk::chunk_output_record_checked(
                record,
                "features",
                0,
                budget,
                crate::operator::output_chunk::OutputChunkErrors::ROLLING,
                &mut || Ok(()),
            )
            .unwrap(),
        );
    });
    assert!(
        u64::try_from(observed.bytes_current).unwrap() <= cost.output_bytes,
        "{observed:?}, {cost:?}"
    );
    assert!(
        observed.bytes_max <= cost.output_bytes + cost.scratch_bytes,
        "{observed:?}, {cost:?}"
    );
    assert_eq!(
        output
            .unwrap()
            .iter()
            .map(crate::Batch::num_rows)
            .sum::<usize>(),
        record.num_rows()
    );
    (cost.output_bytes, cost.scratch_bytes)
}

#[test]
fn chunk_requirements_cover_whole_record_and_nullable_text_splits() {
    let flat = RecordBatch::try_from_iter([(
        "price",
        Arc::new(Float64Array::from(vec![1.0; 128])) as ArrayRef,
    )])
    .unwrap();
    compare(&flat, EdgeBudget::new(128, 4096).unwrap());
    let text = Arc::new(StringArray::from(
        (0..128_u32)
            .map(|value| {
                if value % 3 == 0 {
                    None
                } else {
                    Some("payload")
                }
            })
            .collect::<Vec<_>>(),
    )) as ArrayRef;
    let record =
        RecordBatch::try_from_iter([("price", flat.column(0).clone()), ("text", text)]).unwrap();
    compare(&record, EdgeBudget::new(4, 80).unwrap());
}

#[test]
fn chunk_requirements_exclude_shared_large_list_backing() {
    let values = Arc::new(Float64Array::from(vec![1.0; 200_000])) as ArrayRef;
    let array = ListArray::new(
        Arc::new(Field::new("item", DataType::Float64, false)),
        OffsetBuffer::new(vec![0_i32, 2, 2, 4].into()),
        values,
        None,
    );
    let record = RecordBatch::try_from_iter([("payload", Arc::new(array) as ArrayRef)]).unwrap();
    let (output, _) = compare(&record, EdgeBudget::new(1, 2 << 20).unwrap());
    assert!(
        output < 16 << 10,
        "shared1.6MB payload must not be new chunk storage: {output}"
    );
}

#[test]
fn chunk_requirements_cover_sparse_union_and_struct_slice_owners() {
    use datafusion::arrow::{
        array::{StructArray, UnionArray},
        datatypes::UnionFields,
    };
    let fields = (0..100)
        .map(|index| Field::new(format!("f{index}"), DataType::Float64, false))
        .collect::<Vec<_>>();
    let children = (0..100)
        .map(|_| Arc::new(Float64Array::from(vec![1.0; 3])) as ArrayRef)
        .collect::<Vec<_>>();
    let structure = StructArray::new(fields.clone().into(), children.clone(), None);
    let union = UnionArray::try_new(
        UnionFields::try_new(0..100_i8, fields).unwrap(),
        vec![0_i8, 1, 2].into(),
        None,
        children,
    )
    .unwrap();
    for array in [Arc::new(structure) as ArrayRef, Arc::new(union) as ArrayRef] {
        let record = RecordBatch::try_from_iter([("payload", array)]).unwrap();
        compare(&record, EdgeBudget::new(1, 64 << 10).unwrap());
    }
}

#[test]
fn chunk_requirements_keep_conservative_byte_view_descriptor_charges() {
    use datafusion::arrow::{array::StringViewArray, buffer::Buffer};
    let inline = StringViewArray::from(vec!["hi"; 3]);
    let buffers: Vec<Buffer> = (0..64).map(|_| Buffer::from(vec![0_u8; 16])).collect();
    let array = StringViewArray::try_new(inline.views().clone(), buffers, None).unwrap();
    let record = RecordBatch::try_from_iter([("view", Arc::new(array) as ArrayRef)]).unwrap();
    let (output, _) = compare(&record, EdgeBudget::new(1, 1024).unwrap());
    // The existing logical per-Array descriptor profile stays conservative
    // even though slicing shares the descriptor ArcSlice and data payload.
    assert!(output >= 3 * 64 * u64::try_from(size_of::<Buffer>()).unwrap());
}

#[test]
fn chunk_requirements_preserve_cancellation_and_empty_sequence_behavior() {
    let record = RecordBatch::try_from_iter([(
        "price",
        Arc::new(Float64Array::from(Vec::<f64>::new())) as ArrayRef,
    )])
    .unwrap();
    let cost = plan(&record, EdgeBudget::default(), "features", &mut || Ok(())).unwrap();
    assert_eq!((cost.output_bytes, cost.scratch_bytes), (0, 0));
    let result = plan(&record, EdgeBudget::default(), "features", &mut || {
        Err(crate::CalcFlowError::Cancelled {
            run_id: "chunk-plan".into(),
        })
    });
    assert!(
        matches!(result, Err(crate::CalcFlowError::Cancelled { run_id }) if run_id == "chunk-plan")
    );
}
