use super::*;
use datafusion::arrow::{
    array::{DictionaryArray, Int32Array},
    buffer::OffsetBuffer,
    compute::concat_batches,
    datatypes::Int32Type,
};

pub(super) async fn noncontiguous_dictionary_payloads() {
    let dictionary: ArrayRef = Arc::new(DictionaryArray::<Int32Type>::new(
        Int32Array::from(vec![1, 0, 2]),
        Arc::new(StringArray::from(vec![
            "unreferenced-dictionary-value".repeat(8192),
            "retained-one".into(),
            "retained-two".into(),
            "unused-tail".repeat(8192),
        ])),
    ));
    let field = Arc::new(Field::new(
        "dictionary",
        dictionary.data_type().clone(),
        true,
    ));
    let nested: ArrayRef = Arc::new(StructArray::from(vec![(
        Arc::clone(&field),
        Arc::clone(&dictionary),
    )]));
    Box::pin(selected_batch_case(
        sparse_input(nested),
        "right",
        two_values,
    ))
    .await;
    let nested: ArrayRef = Arc::new(ListArray::new(
        field,
        OffsetBuffer::new(vec![0, 1, 2, 3].into()),
        dictionary,
        None,
    ));
    Box::pin(selected_batch_case(
        sparse_input(nested),
        "left",
        two_values,
    ))
    .await;
    Box::pin(nested_dictionary()).await;
    let empty: ArrayRef = Arc::new(DictionaryArray::<Int32Type>::new(
        Int32Array::from(vec![None; 3]),
        Arc::new(StringArray::from(vec!["unused".repeat(8192)])),
    ));
    Box::pin(selected_batch_case(sparse_input(empty), "right", |join| {
        retained_dictionary_width(join, 0)
    }))
    .await;
}

async fn nested_dictionary() {
    let values: ArrayRef = Arc::new(DictionaryArray::<Int32Type>::new(
        Int32Array::from(vec![1, 0, 2, 1]),
        Arc::new(StringArray::from(vec![
            "unreferenced".repeat(8192),
            "kept-one".into(),
            "kept-two".into(),
            "unused".repeat(8192),
        ])),
    ));
    let value: ArrayRef = Arc::new(DictionaryArray::<Int32Type>::new(
        Int32Array::from(vec![Some(0), None, Some(2)]),
        values,
    ));
    Box::pin(selected_batch_case(sparse_input(value), "left", two_values)).await;
}

fn two_values(join: &StreamJoinOperator) -> bool {
    retained_dictionary_width(join, 2)
}

fn retained_dictionary_width(join: &StreamJoinOperator, expected: usize) -> bool {
    join.state
        .left
        .iter()
        .chain(join.state.right.iter())
        .all(|row| dictionary_width(&row.record.column(2).to_data(), expected))
}

fn dictionary_width(data: &arrow_data::ArrayData, expected: usize) -> bool {
    let matched = !matches!(data.data_type(), DataType::Dictionary(..))
        || data.child_data()[0].len() == expected;
    matched
        && data
            .child_data()
            .iter()
            .all(|child| dictionary_width(child, expected))
}

fn sparse_input(value: ArrayRef) -> Batch {
    let input = input(
        value,
        Arc::new(Int64Array::from(vec![Some(7), None, Some(9)])),
    );
    let records = input.table_payload().unwrap().batches();
    let record = concat_batches(&records[0].schema(), records).unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}
