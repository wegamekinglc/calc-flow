use super::*;
use datafusion::arrow::{
    array::{Array, ArrayRef},
    buffer::{Buffer, OffsetBuffer, ScalarBuffer},
    compute::concat_batches,
    record_batch::RecordBatchOptions,
};
use std::sync::Weak;

const NARROW: &[usize] = &[11, 3, 11, 10];
const TEXT_BYTES: usize = 8_192;
type InputRow<'a> = (&'a str, i64, u64, i64);

fn input_schema() -> SchemaRef {
    let metadata = |name: &str| [("identity".into(), name.into())].into_iter().collect();
    let mut fields = vec![
        Field::new("key", DataType::Utf8, false).with_metadata(metadata("key")),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("seq", DataType::UInt64, false),
        Field::new("value", DataType::Int64, false).with_metadata(metadata("value")),
    ];
    fields
        .extend((0..4).map(|column| Field::new(format!("unused{column}"), DataType::Utf8, false)));
    Arc::new(Schema::new_with_metadata(fields, metadata("input")))
}

fn operator(projection: Option<&[usize]>) -> StreamAsofJoinOperator {
    let side = |prefix: &str| {
        AsofJoinSide::new(
            vec!["key".into()],
            "time".into(),
            vec!["seq".into()],
            prefix.into(),
        )
        .unwrap()
    };
    let spec = StreamAsofJoinSpec::new(
        side("left"),
        side("right"),
        Duration::from_micros(1_000),
        AsofStateLimits::new(100, 64 << 20).unwrap(),
    )
    .unwrap();
    let schema = input_schema();
    let mut operator = StreamAsofJoinOperator::new("asof", schema.clone(), schema, spec).unwrap();
    if let Some(projection) = projection {
        operator.set_output_projection(projection.to_vec()).unwrap();
    }
    operator
}

fn exact_strings(values: &[String]) -> StringArray {
    let length = values.iter().map(String::len).sum();
    let mut bytes = Vec::with_capacity(length);
    let mut offsets = Vec::with_capacity(values.len() + 1);
    offsets.push(0);
    for value in values {
        bytes.extend_from_slice(value.as_bytes());
        offsets.push(i32::try_from(bytes.len()).unwrap());
    }
    StringArray::new(
        OffsetBuffer::new(ScalarBuffer::from(offsets)),
        Buffer::from_vec(bytes),
        None,
    )
}

fn unused_text(side: &str, column: usize, sequence: u64) -> String {
    let prefix = format!("{side}:{column}:{sequence}:");
    format!("{prefix}{}", "x".repeat(TEXT_BYTES - prefix.len()))
}

fn input(side: &str, rows: &[InputRow<'_>]) -> (Batch, Vec<Weak<dyn Array>>, u64) {
    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(exact_strings(
            &rows.iter().map(|row| row.0.into()).collect::<Vec<_>>(),
        )),
        Arc::new(
            TimestampMicrosecondArray::from_iter_values(rows.iter().map(|row| row.1))
                .with_timezone("UTC"),
        ),
        Arc::new(UInt64Array::from_iter_values(rows.iter().map(|row| row.2))),
        Arc::new(Int64Array::from_iter_values(rows.iter().map(|row| row.3))),
    ];
    let mut owners = Vec::new();
    let mut unused_bytes = 0;
    for column in 0..4 {
        let array: ArrayRef = Arc::new(exact_strings(
            &rows
                .iter()
                .map(|row| unused_text(side, column, row.2))
                .collect::<Vec<_>>(),
        ));
        unused_bytes += array.to_data().get_buffer_memory_size() as u64;
        owners.push(Arc::downgrade(&array));
        columns.push(array);
    }
    for column in &columns {
        let data = column.to_data();
        let sentinel = usize::from(column.data_type() == &DataType::Utf8) * size_of::<i32>();
        assert_eq!(
            data.get_buffer_memory_size(),
            data.get_slice_memory_size().unwrap() + sentinel
        );
    }
    let record = RecordBatch::try_new(input_schema(), columns).unwrap();
    let metadata = BatchMetadata::new(
        side,
        77,
        [("caller".into(), serde_json::json!({"preserve": true}))]
            .into_iter()
            .collect(),
    )
    .unwrap();
    (
        Batch::table(vec![record], metadata).unwrap(),
        owners,
        unused_bytes,
    )
}

fn expected(projection: Option<&[usize]>) -> RecordBatch {
    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(vec!["A", "B", "C", "A"])),
        Arc::new(TimestampMicrosecondArray::from(vec![100, 101, 102, 103]).with_timezone("UTC")),
        Arc::new(UInt64Array::from(vec![1, 2, 3, 4])),
        Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
    ];
    for column in 0..4 {
        columns.push(Arc::new(StringArray::from(
            (1..=4)
                .map(|sequence| unused_text("left", column, sequence))
                .collect::<Vec<_>>(),
        )));
    }
    columns.extend([
        Arc::new(StringArray::from(vec![
            Some("A"),
            None,
            Some("C"),
            Some("A"),
        ])) as ArrayRef,
        Arc::new(
            TimestampMicrosecondArray::from(vec![Some(90), None, Some(102), Some(103)])
                .with_timezone("UTC"),
        ),
        Arc::new(UInt64Array::from(vec![
            Some(u64::MAX),
            None,
            Some(12),
            Some(13),
        ])),
        Arc::new(Int64Array::from(vec![
            Some(500),
            None,
            Some(600),
            Some(700),
        ])),
    ]);
    for column in 0..4 {
        columns.push(Arc::new(StringArray::from(vec![
            Some(unused_text("right", column, u64::MAX)),
            None,
            Some(unused_text("right", column, 12)),
            Some(unused_text("right", column, 13)),
        ])));
    }
    let logical = input_schema();
    let fields = [("left", false), ("right", true)]
        .into_iter()
        .flat_map(|(side, nullable)| {
            logical.fields().iter().map(move |field| {
                field
                    .as_ref()
                    .clone()
                    .with_name(format!("{side}__{}", field.name()))
                    .with_nullable(nullable)
            })
        })
        .collect::<Vec<_>>();
    let indices = projection.map_or_else(|| (0..16).collect(), <[usize]>::to_vec);
    let schema = Arc::new(Schema::new(
        indices
            .iter()
            .map(|&index| fields[index].clone())
            .collect::<Vec<_>>(),
    ));
    RecordBatch::try_new_with_options(
        schema,
        indices
            .iter()
            .map(|&index| columns[index].clone())
            .collect(),
        &RecordBatchOptions::new().with_row_count(Some(4)),
    )
    .unwrap()
}

async fn admitted(
    projection: Option<&[usize]>,
) -> (StreamAsofJoinOperator, Vec<Weak<dyn Array>>, u64) {
    let mut original = operator(projection);
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(original.output_ports().to_vec());
    let mut owners = Vec::new();
    let mut unused_bytes = 0;
    for (side, rows) in [
        (
            "right",
            vec![
                ("A", 90, 10, 400),
                ("A", 90, u64::MAX, 500),
                ("C", 102, 12, 600),
            ],
        ),
        (
            "left",
            vec![("A", 100, 1, 1), ("B", 101, 2, 2), ("C", 102, 3, 3)],
        ),
    ] {
        let (batch, tracked, bytes) = input(side, &rows);
        let before = batch.clone();
        original
            .process_data(side, batch, &context, &mut output)
            .await
            .unwrap();
        let (independent, _, _) = input(side, &rows);
        assert_eq!(before.metadata(), independent.metadata());
        assert_eq!(
            before.table_payload().unwrap().batches(),
            independent.table_payload().unwrap().batches()
        );
        let side_index = u8::from(side == "right");
        // Observe admission copies as well as shared caller arrays.
        owners.extend(tracked.into_iter().enumerate().map(|(column, caller)| {
            original
                .state
                .batches
                .iter()
                .filter(|(key, _)| key.0 == side_index)
                .find_map(|(_, (payload, _))| {
                    payload
                        .record
                        .column_by_name(&format!("unused{column}"))
                        .map(Arc::downgrade)
                })
                .unwrap_or(caller)
        }));
        unused_bytes += bytes;
    }
    assert!(output.drain("output").is_empty());
    assert_eq!(original.status.pending_left_rows, 3);
    assert_eq!(original.status.retained_right_rows, 3);
    assert_eq!(original.input_ports()[0].schema(), Some(&input_schema()));
    let snapshot = original.capture(Epoch::INITIAL).unwrap();
    let mut resumed = operator(projection);
    resumed.restore(&snapshot).unwrap();
    assert_eq!(resumed.status(), original.status());
    for (side, rows) in [
        ("right", vec![("A", 103, 13, 700)]),
        ("left", vec![("A", 103, 4, 4)]),
    ] {
        let (batch, _, _) = input(side, &rows);
        resumed
            .process_data(side, batch, &context, &mut output)
            .await
            .unwrap();
    }
    resumed.on_end(&context, &mut output).await.unwrap();
    let messages = output.drain("output");
    let mut records = Vec::new();
    let mut sequence = 0;
    for message in &messages {
        let batch = message.as_data().unwrap();
        assert_eq!(
            batch.metadata(),
            &BatchMetadata::new("asof", sequence, JsonMap::new()).unwrap()
        );
        sequence += batch
            .table_payload()
            .unwrap()
            .batches()
            .iter()
            .map(|record| record.num_rows() as u64)
            .sum::<u64>();
        records.extend(batch.table_payload().unwrap().batches().iter().cloned());
    }
    let independent = expected(projection);
    let actual = concat_batches(&independent.schema(), &records).unwrap();
    assert_eq!(actual, independent);
    assert_eq!(resumed.status.matched_rows, 3);
    assert_eq!(resumed.status.unmatched_rows, 1);
    assert_eq!(resumed.status.state_bytes, 0);
    (original, owners, unused_bytes)
}

fn retained_columns(operator: &StreamAsofJoinOperator) -> BTreeMap<u8, Vec<Vec<String>>> {
    let mut columns: BTreeMap<u8, Vec<Vec<String>>> = BTreeMap::new();
    for (key, (batch, _)) in operator.state.batches.iter() {
        columns.entry(key.0).or_default().push(
            batch
                .record
                .schema()
                .fields()
                .iter()
                .map(|field| field.name().clone())
                .collect(),
        );
    }
    columns
}

#[tokio::test]
async fn retained_projection_prunes_both_physical_payloads_after_restore_controls() {
    let (operator, _, _) = admitted(Some(NARROW)).await;
    let expected: Vec<String> = ["key", "time", "seq", "value"]
        .into_iter()
        .map(str::to_owned)
        .collect();
    assert_eq!(
        retained_columns(&operator),
        [(0, vec![expected.clone()]), (1, vec![expected])]
            .into_iter()
            .collect()
    );
}

#[tokio::test]
async fn retained_projection_releases_unique_unused_array_owners() {
    let (operator, owners, _) = admitted(Some(NARROW)).await;
    assert_eq!(owners.len(), 8);
    assert_eq!(operator.status.state_rows, 6);
    assert!(
        owners.iter().all(|owner| owner.upgrade().is_none()),
        "retained state pins discarded payload arrays"
    );
}

#[tokio::test]
async fn retained_projection_charges_physical_payload_instead_of_unused_buffers() {
    let (narrow, _, unused_bytes) = admitted(Some(NARROW)).await;
    let (full, _, full_unused_bytes) = admitted(None).await;
    assert_eq!(unused_bytes, full_unused_bytes);
    assert_eq!(narrow.status.state_rows, full.status.state_rows);
    assert!(
        narrow.status.state_bytes + unused_bytes < full.status.state_bytes,
        "narrow retained gauge still includes unique unused backing"
    );
}

#[tokio::test]
async fn retained_projection_empty_output_preserves_identities_and_row_count() {
    let (operator, _, _) = admitted(Some(&[])).await;
    let expected: Vec<String> = ["key", "time", "seq"]
        .into_iter()
        .map(str::to_owned)
        .collect();
    assert_eq!(
        retained_columns(&operator),
        [(0, vec![expected.clone()]), (1, vec![expected])]
            .into_iter()
            .collect()
    );
}

#[tokio::test]
async fn retained_projection_full_output_preserves_all_payload_columns_and_owners() {
    let (operator, owners, _) = admitted(None).await;
    let expected = input_schema()
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .collect::<Vec<_>>();
    assert_eq!(
        retained_columns(&operator),
        [(0, vec![expected.clone()]), (1, vec![expected])]
            .into_iter()
            .collect()
    );
    assert!(owners.iter().all(|owner| owner.upgrade().is_some()));
}

#[tokio::test]
async fn retained_projection_keeps_union_dependencies_on_both_sides() {
    let (operator, _, _) = admitted(Some(&[3, 4, 11, 12])).await;
    let expected = ["key", "time", "seq", "value", "unused0"]
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    assert_eq!(
        retained_columns(&operator),
        [(0, vec![expected.clone()]), (1, vec![expected])]
            .into_iter()
            .collect()
    );
}

#[path = "retained_projection_contracts.rs"]
mod contracts;
