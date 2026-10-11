use calc_flow::{
    Batch, BatchMetadata, CancellationToken, EdgeCollector, Epoch, JsonMap, OperatorMetadata,
    OperatorStateSnapshot, SqlOperator, StateBudget, StateSegment, StreamJobContext,
    StreamOperator, StreamOperatorContext,
};
use datafusion::arrow::{
    array::{Array, ArrayRef, Int64Array, StringArray},
    buffer::Buffer,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use serde_json::{Value, json};
use std::sync::{Arc, Weak};

fn batch(source: &str, values: &[i64]) -> Batch {
    Batch::table(
        vec![
            RecordBatch::try_from_iter(vec![(
                "value",
                Arc::new(Int64Array::from(values.to_vec())) as Arc<dyn Array>,
            )])
            .unwrap(),
        ],
        BatchMetadata::new(source, 0, JsonMap::new()).unwrap(),
    )
    .unwrap()
}

fn job() -> StreamJobContext {
    StreamJobContext::new(
        1,
        "sql-test",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
}

fn total(collector: &mut EdgeCollector) -> i64 {
    let output = collector.drain("output");
    assert_eq!(output.len(), 1);
    let table = output[0].as_data().unwrap().table_payload().unwrap();
    table.batches()[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0)
}

fn same_segment(left: &OperatorStateSnapshot, right: &OperatorStateSnapshot) -> bool {
    let name = if left.inline_metadata["state_layout"] == json!(3) {
        "group-state"
    } else {
        "input-retained"
    };
    Arc::ptr_eq(
        &left.segments[name].bytes_arc(),
        &right.segments[name].bytes_arc(),
    )
}

fn setup(query: &str) -> (SqlOperator, StreamJobContext, EdgeCollector) {
    let operator = SqlOperator::new("totals", query, vec!["events".into()], vec![]).unwrap();
    let collector = EdgeCollector::new(operator.output_ports().to_vec());
    let job = StreamJobContext::new(
        1,
        "retained",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    (operator, job, collector)
}

fn wide_input(sequence: u64) -> (Batch, Weak<dyn Array>, Buffer) {
    let unused: ArrayRef = Arc::new(StringArray::from(vec![
        "unused payload".repeat(1024),
        "unused payload".repeat(2048),
        "unused payload".repeat(4096),
    ]));
    let weak = Arc::downgrade(&unused);
    let allocation = unused.to_data().buffers()[1].clone();
    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(vec![Some("a"), Some("b"), Some("a")])),
        Arc::new(Int64Array::from(vec![Some(1), Some(9), Some(2)])),
        Arc::new(Int64Array::from(vec![Some(1), Some(0), Some(1)])),
        unused,
    ];
    columns.extend((4..8).map(|_| Arc::new(StringArray::from(vec![Some("other"); 3])) as ArrayRef));
    let fields = vec![
        Field::new("key", DataType::Utf8, true),
        Field::new("value", DataType::Int64, true),
        Field::new("keep", DataType::Int64, true),
        Field::new("unused", DataType::Utf8, false),
        Field::new("extra4", DataType::Utf8, true),
        Field::new("extra5", DataType::Utf8, true),
        Field::new("extra6", DataType::Utf8, true),
        Field::new("extra7", DataType::Utf8, true),
    ];
    let record = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap();
    let metadata = BatchMetadata::new("retained", sequence, JsonMap::new()).unwrap();
    (
        Batch::table(vec![record], metadata).unwrap(),
        weak,
        allocation,
    )
}

fn assert_same_snapshot(left: &OperatorStateSnapshot, right: &OperatorStateSnapshot) {
    assert_eq!(left.inline_metadata, right.inline_metadata);
    assert_eq!(
        left.segments.keys().collect::<Vec<_>>(),
        right.segments.keys().collect::<Vec<_>>()
    );
    for (name, segment) in &left.segments {
        assert_eq!(segment.bytes(), right.segments[name].bytes());
    }
}

#[tokio::test]
async fn test_sql_query_and_byte_budget_errors_preserve_checkpoint() {
    let mut operator = SqlOperator::new(
        "totals",
        "SELECT SUM(CAST(value AS INT)) AS total FROM events",
        vec!["events".into()],
        Vec::new(),
    )
    .unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "totals", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    let input = batch("budget-initial", &[1]);
    let bytes = u64::try_from(input.estimated_bytes().unwrap()).unwrap();
    operator
        .process_data("events", input, &context, &mut collector)
        .await
        .unwrap();
    assert_eq!(total(&mut collector), 1);
    let first = operator.checkpoint(Epoch::INITIAL).unwrap();
    assert!(
        operator
            .process_data(
                "events",
                batch("query-failure", &[i64::MAX]),
                &context,
                &mut collector
            )
            .await
            .is_err()
    );
    assert!(collector.drain("output").is_empty());
    assert!(same_segment(
        &first,
        &operator.checkpoint(Epoch::INITIAL).unwrap()
    ));
    operator
        .set_state_budget(StateBudget::new(100, bytes).unwrap())
        .unwrap();
    let error = operator
        .process_data(
            "events",
            batch("budget-failure", &[2]),
            &context,
            &mut collector,
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("configured state budget"));
    assert!(collector.drain("output").is_empty());
    assert!(same_segment(
        &first,
        &operator.checkpoint(Epoch::INITIAL).unwrap()
    ));
    operator
        .set_state_budget(StateBudget::new(100, bytes * 2).unwrap())
        .unwrap();
    operator
        .process_data(
            "events",
            batch("budget-next", &[2]),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    assert_eq!(total(&mut collector), 3);
}

#[tokio::test]
async fn test_sql_retained_descriptor_rejects_forgery_atomically() {
    let (mut operator, job, mut collector) = setup("SELECT SUM(value) AS total FROM events");
    let context = StreamOperatorContext::new(&job, "totals", None);
    operator
        .process_data("events", wide_input(0).0, &context, &mut collector)
        .await
        .unwrap();
    let before = operator.checkpoint(Epoch::INITIAL).unwrap();
    assert_eq!(before.inline_metadata["state_layout"], json!(3));
    let mutations = [
        ("state_layout", json!(4)),
        ("state_accounting", json!(1)),
        ("rows", json!(4)),
        ("bytes", json!(0)),
        ("unknown", json!(0)),
    ];
    for (field, value) in mutations {
        let mut forged = before.clone();
        forged.inline_metadata.insert(field.into(), value);
        assert!(
            StreamOperator::restore(&mut operator, &forged).is_err(),
            "accepted forged {field}"
        );
        assert_same_snapshot(&before, &operator.checkpoint(Epoch::INITIAL).unwrap());
    }
    for (field, value) in [
        ("retained_ordinals", json!([0, 1])),
        ("logical_schema_sha256", json!("00".repeat(32))),
        ("physical_schema_sha256", json!("00".repeat(32))),
    ] {
        let mut forged = before.clone();
        let mut control: Value =
            serde_json::from_slice(forged.segments["control"].bytes()).unwrap();
        control["identity"][field] = value;
        let segment = StateSegment::new(serde_json::to_vec(&control).unwrap());
        forged
            .inline_metadata
            .insert("control_sha256".into(), json!(segment.sha256()));
        forged.segments.insert("control".into(), segment);
        assert!(
            StreamOperator::restore(&mut operator, &forged).is_err(),
            "accepted forged control {field}"
        );
        assert_same_snapshot(&before, &operator.checkpoint(Epoch::INITIAL).unwrap());
    }
    for name in ["input", "unknown"] {
        let mut forged = before.clone();
        forged
            .segments
            .insert(name.into(), StateSegment::new(vec![]));
        assert!(StreamOperator::restore(&mut operator, &forged).is_err());
        assert_same_snapshot(&before, &operator.checkpoint(Epoch::INITIAL).unwrap());
    }
}
