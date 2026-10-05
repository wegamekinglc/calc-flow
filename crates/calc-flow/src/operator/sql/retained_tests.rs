use std::sync::Weak;

use datafusion::arrow::{
    array::{ArrayRef, StringArray},
    buffer::Buffer,
    datatypes::{Field, Schema},
};

use super::*;

const RAW_SUM: &str = "SELECT SUM(abs(value)) AS total FROM events WHERE value IS NOT NULL";

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

fn retained_names(operator: &SqlOperator) -> Vec<String> {
    let schema = if let Some(state) = &operator.compact {
        assert!(operator.retained.is_none());
        state
            .projection()
            .unwrap()
            .columns
            .physical_schema()
            .clone()
    } else {
        operator.retained.as_ref().unwrap().records[0].schema()
    };
    schema
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .collect()
}

async fn assert_raw_oracle(operator: &SqlOperator, input: &Batch, observed: &Batch) {
    assert!(operator.incremental.is_none());
    assert!(operator.compact.is_none());
    let expected = DataFusionRuntime::new(DataFusionConfig::default())
        .unwrap()
        .sql(
            &operator.query,
            &BTreeMap::from([("events".into(), input.clone())]),
            Some("independent-current-raw"),
        )
        .await
        .unwrap();
    assert_eq!(observed.metadata(), expected.metadata());
    assert_eq!(
        observed.table_payload().unwrap().schema(),
        expected.table_payload().unwrap().schema()
    );
    assert_eq!(rows(observed), rows(&expected));
}

#[tokio::test]
async fn test_sql_retained_projection_releases_unused_array_ownership() {
    let (mut operator, job, mut collector) =
        setup("SELECT key, SUM(value) AS total FROM events GROUP BY key");
    let context = StreamOperatorContext::new(&job, "totals", None);
    let (input, unused, allocation) = wide_input(0);
    let metadata = input.metadata().clone();
    operator
        .process_data("events", input, &context, &mut collector)
        .await
        .unwrap();
    let output = collector.drain("output");
    assert_eq!(output[0].as_data().unwrap().metadata(), &metadata);
    assert_eq!(
        rows(output[0].as_data().unwrap()),
        vec![
            vec![
                datafusion::common::ScalarValue::Utf8(Some("a".into())),
                datafusion::common::ScalarValue::Int64(Some(3))
            ],
            vec![
                datafusion::common::ScalarValue::Utf8(Some("b".into())),
                datafusion::common::ScalarValue::Int64(Some(9))
            ],
        ]
    );
    assert!(
        unused.upgrade().is_none(),
        "unused source array remains owned by SQL state"
    );
    assert_eq!(
        allocation.strong_count(),
        1,
        "unused allocation remains shared"
    );
    assert_eq!(retained_names(&operator), vec!["key", "value"]);
}

#[tokio::test]
async fn test_sql_retained_projection_crops_checkpoint_ipc_fields() {
    let (mut operator, job, mut collector) = setup(RAW_SUM);
    let context = StreamOperatorContext::new(&job, "totals", None);
    let (input, _, _) = wide_input(0);
    let full_ipc_bytes = encode_sql_state(&input).unwrap().len();
    operator
        .process_data("events", input.clone(), &context, &mut collector)
        .await
        .unwrap();
    let emitted = collector.drain("output");
    assert_eq!(emitted.len(), 1);
    assert_raw_oracle(&operator, &input, emitted[0].as_data().unwrap()).await;
    operator.prepare_checkpoint_async(&context).await.unwrap();
    let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
    let fields = snapshot
        .segments
        .values()
        .filter_map(|segment| {
            decode_sql_state(segment.bytes())
                .ok()
                .filter(|batch| batch.num_rows() == 3)
        })
        .collect::<Vec<_>>();
    assert_eq!(fields.len(), 1);
    let table = fields[0].table_payload().unwrap();
    assert_eq!(
        table
            .schema()
            .fields()
            .iter()
            .map(|field| field.name().as_str())
            .collect::<Vec<_>>(),
        vec!["value"]
    );
    assert!(
        snapshot
            .segments
            .values()
            .map(|segment| segment.bytes().len())
            .sum::<usize>()
            < full_ipc_bytes / 4
    );
}

#[tokio::test]
async fn test_sql_retained_count_star_preserves_rows_without_columns() {
    let query = "SELECT COUNT(*) AS rows FROM events";
    let (mut operator, job, mut collector) = setup(query);
    let context = StreamOperatorContext::new(&job, "totals", None);
    for sequence in 0..2 {
        operator
            .process_data("events", wide_input(sequence).0, &context, &mut collector)
            .await
            .unwrap();
    }
    let output = collector.drain("output");
    assert_eq!(
        rows(output[1].as_data().unwrap()),
        vec![vec![datafusion::common::ScalarValue::Int64(Some(6))]]
    );
    assert!(operator.retained.is_none());
    let compact = operator.compact.as_ref().unwrap();
    assert_eq!(
        compact
            .projection()
            .unwrap()
            .columns
            .physical_schema()
            .fields()
            .len(),
        0
    );
    assert_eq!(compact.ledger.rows, 6);
    assert_eq!(compact.ledger.bytes, 0);
    operator.prepare_checkpoint_async(&context).await.unwrap();
    let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
    let (mut restored, _, mut recovered_output) = setup(query);
    StreamOperator::restore(&mut restored, &snapshot).unwrap();
    restored
        .process_data("events", wide_input(2).0, &context, &mut recovered_output)
        .await
        .unwrap();
    let output = recovered_output.drain("output");
    assert_eq!(
        rows(output[0].as_data().unwrap()),
        vec![vec![datafusion::common::ScalarValue::Int64(Some(9))]]
    );
}

#[tokio::test]
async fn test_sql_retained_projection_keeps_hidden_fallback_dependencies() {
    let query = "SELECT key, SUM(value) AS total FROM events WHERE keep > 0 GROUP BY key HAVING SUM(value) > 1 ORDER BY total DESC LIMIT 1";
    let (mut operator, job, mut collector) = setup(query);
    let context = StreamOperatorContext::new(&job, "totals", None);
    for sequence in 0..2 {
        operator
            .process_data("events", wide_input(sequence).0, &context, &mut collector)
            .await
            .unwrap();
    }
    let output = collector.drain("output");
    assert_eq!(
        rows(output[1].as_data().unwrap()),
        vec![vec![
            datafusion::common::ScalarValue::Utf8(Some("a".into())),
            datafusion::common::ScalarValue::Int64(Some(6)),
        ]]
    );
    assert_eq!(retained_names(&operator), vec!["key", "value", "keep"]);
}

#[tokio::test]
async fn test_sql_current_native_snapshot_continues_cumulative_output() {
    let query = "SELECT SUM(value) AS total FROM events";
    let (mut operator, job, mut collector) = setup(query);
    let context = StreamOperatorContext::new(&job, "totals", None);
    operator
        .process_data("events", wide_input(0).0, &context, &mut collector)
        .await
        .unwrap();
    operator.prepare_checkpoint_async(&context).await.unwrap();
    let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
    let (mut restored, _, mut recovered_output) = setup(query);
    StreamOperator::restore(&mut restored, &snapshot).unwrap();
    restored
        .process_data("events", wide_input(1).0, &context, &mut recovered_output)
        .await
        .unwrap();
    let output = recovered_output.drain("output");
    assert_eq!(
        rows(output[0].as_data().unwrap()),
        vec![vec![datafusion::common::ScalarValue::Int64(Some(24))]],
    );
}

#[path = "retained_contract_tests.rs"]
mod contract_tests;
