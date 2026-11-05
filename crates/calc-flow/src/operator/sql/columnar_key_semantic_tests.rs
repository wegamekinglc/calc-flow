use super::*;
use crate::operator::{OperatorMetadata, StreamCollector, StreamOperator};
use crate::{EdgeCollector, Epoch, SqlOperator};
use datafusion::arrow::array::BooleanArray;
use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryPool};

#[path = "columnar_key_boundary_tests.rs"]
mod boundary;

const FILTERED: &str = "SELECT left_key, right_key, SUM(value) FILTER (WHERE selected) AS total FROM events GROUP BY left_key, right_key";
const TUPLES: [(Option<&str>, Option<&str>); 8] = [
    (Some(""), Some("")),
    (None, Some("")),
    (Some(""), None),
    (None, None),
    (Some("a|b"), Some("c")),
    (Some("a"), Some("b|c")),
    (Some("é"), Some("e\u{301}")),
    (Some("x\0y"), Some("😀")),
];

fn integer(unsigned: bool, value: Option<u64>) -> ScalarValue {
    if unsigned {
        ScalarValue::UInt64(value)
    } else {
        ScalarValue::Int64(value.map(|bits| i64::from_ne_bytes(bits.to_ne_bytes())))
    }
}

fn maximum(unsigned: bool) -> u64 {
    if unsigned { u64::MAX } else { u64::MAX >> 1 }
}

fn tuple_group(phase: usize, row: usize) -> usize {
    (row + if phase == 2 { 6 } else { 0 }) % 8
}

fn phase_value(unsigned: bool, phase: usize, row: usize) -> ScalarValue {
    let value = match (phase, row) {
        (1, 0) => Some(maximum(unsigned)),
        (2, 2) => Some(1),
        (2, 0) => Some(2),
        (2, 1) => Some(99),
        _ => None,
    };
    integer(unsigned, value)
}

fn input(unsigned: bool, phase: usize, count: usize) -> Batch {
    let dtype = integer(unsigned, None).data_type();
    let values = if count == 0 {
        datafusion::arrow::array::new_empty_array(&dtype)
    } else {
        ScalarValue::iter_to_array((0..count).map(|row| phase_value(unsigned, phase, row))).unwrap()
    };
    let columns = vec![
        Arc::new(StringArray::from(
            (0..count)
                .map(|row| TUPLES[tuple_group(phase, row)].0)
                .collect::<Vec<_>>(),
        )) as ArrayRef,
        Arc::new(StringArray::from(
            (0..count)
                .map(|row| TUPLES[tuple_group(phase, row)].1)
                .collect::<Vec<_>>(),
        )) as ArrayRef,
        values,
        Arc::new(BooleanArray::from(
            (0..count)
                .map(|row| tuple_group(phase, row) != 7)
                .collect::<Vec<_>>(),
        )) as ArrayRef,
    ];
    let schema = Arc::new(Schema::new(vec![
        Field::new("left_key", DataType::Utf8, true),
        Field::new("right_key", DataType::Utf8, true),
        Field::new("value", dtype, true),
        Field::new("selected", DataType::Boolean, false),
    ]));
    let record = RecordBatch::try_new(schema, columns).unwrap();
    let split = count.min(17);
    Batch::table(
        vec![
            record.slice(0, 0),
            record.slice(0, split),
            record.slice(split, count - split),
        ],
        BatchMetadata::default(),
    )
    .unwrap()
}

async fn configured(
    runtime: &DataFusionRuntime,
    input: &Batch,
    sql: &str,
    groups: u64,
    bytes: u64,
) -> IncrementalSql {
    let query = crate::expression::parse_select_query(sql).unwrap();
    let mut plan = IncrementalSql::plan(
        runtime,
        &query,
        "events",
        input.table_payload().unwrap().schema().clone(),
        "columnar-semantic",
    )
    .await
    .unwrap()
    .unwrap();
    plan.set_compact_budget(
        Some(StateBudget::new(groups, bytes).unwrap()),
        "columnar-semantic",
    )
    .unwrap();
    plan
}

fn rows(records: &[RecordBatch]) -> Vec<Vec<ScalarValue>> {
    records
        .iter()
        .flat_map(|record| {
            (0..record.num_rows()).map(|row| {
                record
                    .columns()
                    .iter()
                    .map(|array| ScalarValue::try_from_array(array, row).unwrap())
                    .collect()
            })
        })
        .collect()
}

fn expected(unsigned: bool, phase: usize) -> Vec<Vec<ScalarValue>> {
    TUPLES
        .iter()
        .enumerate()
        .map(|(group, (left, right))| {
            let value = match (group, phase) {
                (0, 1) => Some(maximum(unsigned)),
                (0, 2) => Some(maximum(unsigned).wrapping_add(1)),
                (6, 2) => Some(2),
                _ => None,
            };
            vec![
                ScalarValue::Utf8(left.map(str::to_owned)),
                ScalarValue::Utf8(right.map(str::to_owned)),
                integer(unsigned, value),
            ]
        })
        .collect()
}

fn job(token: CancellationToken) -> StreamJobContext {
    StreamJobContext::new(979, "columnar-semantic", JsonMap::new(), None, token)
}

async fn apply(
    plan: &mut IncrementalSql,
    input: &Batch,
    expected: &[Vec<ScalarValue>],
    context: &StreamOperatorContext<'_>,
) -> Vec<RecordBatch> {
    let transaction = plan
        .update(input, context, "columnar-semantic")
        .await
        .unwrap();
    assert_eq!(rows(&transaction.records), expected);
    let records = transaction.records.clone();
    plan.commit(transaction);
    records
}

#[tokio::test]
async fn test_columnar_tuple_identity_filter_wrapping_restore_discard_and_held_outputs() {
    for unsigned in [false, true] {
        let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
        let first = input(unsigned, 0, 64);
        let saved_input = rows(first.table_payload().unwrap().batches());
        let mut state = configured(
            &runtime,
            &first,
            FILTERED,
            8,
            StateBudget::default().max_bytes(),
        )
        .await;
        let job = job(CancellationToken::new());
        let context = StreamOperatorContext::new(&job, "columnar-semantic", None);
        let held = apply(&mut state, &first, &expected(unsigned, 0), &context).await;
        apply(
            &mut state,
            &input(unsigned, 1, 16),
            &expected(unsigned, 1),
            &context,
        )
        .await;
        let saved = state
            .export_native_state("columnar-semantic", || Ok(()))
            .unwrap();
        let restored = configured(
            &runtime,
            &first,
            FILTERED,
            8,
            StateBudget::default().max_bytes(),
        )
        .await
        .import_native_state(saved.records(), 80, true, || Ok(()), "columnar-semantic")
        .unwrap();
        drop(state);
        state = restored;
        let incoming = input(unsigned, 2, 16);
        let discarded = state
            .update(&incoming, &context, "columnar-semantic")
            .await
            .unwrap();
        drop(discarded);
        let after_discard = state
            .export_native_state("columnar-semantic", || Ok(()))
            .unwrap();
        assert_eq!(saved.records(), after_discard.records());
        apply(&mut state, &incoming, &expected(unsigned, 2), &context).await;
        apply(
            &mut state,
            &input(unsigned, 0, 0),
            &expected(unsigned, 2),
            &context,
        )
        .await;
        assert!(state.key_work.columnar_chunks.load(Ordering::SeqCst) > 0);
        assert_eq!(rows(&held), expected(unsigned, 0));
        assert_eq!(rows(first.table_payload().unwrap().batches()), saved_input);
        drop((state, saved, after_discard));
        assert_eq!(runtime.incremental_memory_pool().reserved(), 0);
    }
}

async fn fallback_case(
    runtime: DataFusionRuntime,
    sql: &str,
    count: usize,
    bytes: u64,
    encoded: usize,
) {
    let incoming = input(false, 0, count);
    let mut state = configured(&runtime, &incoming, sql, 8, bytes).await;
    let job = job(CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "columnar-semantic", None);
    let transaction = state
        .update(&incoming, &context, "columnar-semantic")
        .await
        .unwrap();
    assert_eq!(state.key_work.columnar_chunks.load(Ordering::SeqCst), 0);
    assert_eq!(state.encoded_rows.load(Ordering::SeqCst), encoded);
    let groups = if sql.contains("WHERE selected GROUP") {
        7
    } else {
        8
    };
    assert_eq!(
        transaction
            .records
            .iter()
            .map(RecordBatch::num_rows)
            .sum::<usize>(),
        groups
    );
    state.commit(transaction);
    drop(state);
    assert_eq!(runtime.incremental_memory_pool().reserved(), 0);
}

#[tokio::test]
async fn test_columnar_fallback_preserves_below_edge_tight_bytes_custom_pool_and_where() {
    for count in [8, 15] {
        fallback_case(
            DataFusionRuntime::new(DataFusionConfig::default()).unwrap(),
            FILTERED,
            count,
            StateBudget::default().max_bytes(),
            count,
        )
        .await;
    }
    fallback_case(
        DataFusionRuntime::new(DataFusionConfig::default()).unwrap(),
        FILTERED,
        64,
        64 << 20,
        64,
    )
    .await;
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 30));
    fallback_case(
        DataFusionRuntime::with_benchmark_pool(DataFusionConfig::default(), pool).unwrap(),
        FILTERED,
        64,
        StateBudget::default().max_bytes(),
        64,
    )
    .await;
    let sql = "SELECT left_key,right_key,SUM(value) AS total FROM events WHERE selected GROUP BY left_key,right_key";
    fallback_case(
        DataFusionRuntime::new(DataFusionConfig::default()).unwrap(),
        sql,
        64,
        StateBudget::default().max_bytes(),
        64,
    )
    .await;
    let mixed = "SELECT left_key,right_key,SUM(value) AS total,COUNT(*) AS rows FROM events GROUP BY left_key,right_key";
    fallback_case(
        DataFusionRuntime::new(DataFusionConfig::default()).unwrap(),
        mixed,
        64,
        StateBudget::default().max_bytes(),
        64,
    )
    .await;
    wide_key_fallback().await;
}

async fn wide_key_fallback() {
    let original = input(false, 0, 64);
    let records = original
        .table_payload()
        .unwrap()
        .batches()
        .iter()
        .map(|record| {
            let mut columns = record.columns().to_vec();
            columns[0] =
                datafusion::arrow::compute::cast(&columns[0], &DataType::LargeUtf8).unwrap();
            let mut fields = record.schema().fields().to_vec();
            fields[0] = Arc::new(Field::new("left_key", DataType::LargeUtf8, true));
            RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
        })
        .collect();
    let incoming = Batch::table(records, BatchMetadata::default()).unwrap();
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let mut state = configured(
        &runtime,
        &incoming,
        FILTERED,
        8,
        StateBudget::default().max_bytes(),
    )
    .await;
    let job = job(CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "columnar-semantic", None);
    let transaction = state
        .update(&incoming, &context, "columnar-semantic")
        .await
        .unwrap();
    assert_eq!(state.key_work.columnar_chunks.load(Ordering::SeqCst), 0);
    assert_eq!(state.encoded_rows.load(Ordering::SeqCst), 64);
    assert_eq!(rows(&transaction.records).len(), 8);
    state.commit(transaction);
    drop(state);
    assert_eq!(runtime.incremental_memory_pool().reserved(), 0);
}

#[tokio::test]
async fn test_columnar_actual_half_input_edge_and_group_failure_keep_legacy_chunk_precedence() {
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let incoming = input(false, 0, 16);
    let mut state = configured(
        &runtime,
        &incoming,
        FILTERED,
        8,
        StateBudget::default().max_bytes(),
    )
    .await;
    let job = job(CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "columnar-semantic", None);
    apply(&mut state, &incoming, &expected(false, 0), &context).await;
    assert!(state.key_work.columnar_chunks.load(Ordering::SeqCst) > 0);
    let overflowing = input(false, 0, 32);
    let mut constrained = configured(
        &runtime,
        &overflowing,
        FILTERED,
        7,
        StateBudget::default().max_bytes(),
    )
    .await;
    let before = runtime.compact_state_bytes();
    let failed = constrained
        .update(&overflowing, &context, "columnar-semantic")
        .await;
    assert!(
        matches!(failed,Err(CalcFlowError::DataFusion{message,..}) if message.contains("group budget"))
    );
    assert!(constrained.key_work.columnar_chunks.load(Ordering::SeqCst) > 0);
    assert!(constrained.groups.is_empty());
    assert_eq!(runtime.compact_state_bytes(), before);
    assert_group_before_filter_error(&runtime, &context).await;
}

fn with_value(record: &RecordBatch, value: i64) -> RecordBatch {
    let mut columns = record.columns().to_vec();
    columns[2] = Arc::new(Int64Array::from(vec![Some(value); record.num_rows()]));
    RecordBatch::try_new(record.schema(), columns).unwrap()
}

async fn assert_group_before_filter_error(
    runtime: &DataFusionRuntime,
    context: &StreamOperatorContext<'_>,
) {
    let fixture = input(false, 0, 64);
    let record = &fixture.table_payload().unwrap().batches()[1];
    let incoming = Batch::table(
        vec![
            with_value(&record.slice(0, 8), 1),
            with_value(&record.slice(8, 8), 0),
        ],
        BatchMetadata::default(),
    )
    .unwrap();
    let sql = "SELECT left_key,right_key,SUM(value) FILTER (WHERE 1 / value > 0) AS total FROM events GROUP BY left_key,right_key";
    let mut state = configured(
        runtime,
        &incoming,
        sql,
        7,
        StateBudget::default().max_bytes(),
    )
    .await;
    let Err(error) = state.update(&incoming, context, "columnar-semantic").await else {
        panic!("group budget must reject before later division");
    };
    assert!(
        matches!(error,CalcFlowError::DataFusion{message,..} if message.contains("group budget"))
    );
    assert_eq!(state.key_work.columnar_chunks.load(Ordering::SeqCst), 1);
}

struct Reject;

#[async_trait::async_trait]
impl StreamCollector for Reject {
    async fn emit(&mut self, _port: &str, _batch: Batch) -> Result<()> {
        Err(df_error("columnar-reject", "injected output refusal"))
    }
}

#[tokio::test]
async fn test_columnar_selected_output_refusal_keeps_capture_and_once_retry() {
    let mut state =
        SqlOperator::new("columnar-semantic", FILTERED, vec!["events".into()], vec![]).unwrap();
    state
        .set_compact_state_budget(Some(
            StateBudget::new(8, StateBudget::default().max_bytes()).unwrap(),
        ))
        .unwrap();
    let job = job(CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "columnar-semantic", None);
    let mut collector = EdgeCollector::new(state.output_ports().to_vec());
    state
        .process_data("events", input(false, 0, 64), &context, &mut collector)
        .await
        .unwrap();
    collector.drain("output");
    let before = state.checkpoint(Epoch::INITIAL).unwrap();
    let paid = state.stream_state.runtime().unwrap().compact_state_bytes();
    let error = state
        .process_data("events", input(false, 1, 16), &context, &mut Reject)
        .await
        .unwrap_err();
    assert!(
        matches!(error,CalcFlowError::DataFusion{node_id:Some(name),..} if name=="columnar-reject")
    );
    assert_eq!(
        state.stream_state.runtime().unwrap().compact_state_bytes(),
        paid
    );
    let after = state.checkpoint(Epoch::INITIAL).unwrap();
    assert_eq!(before.inline_metadata, after.inline_metadata);
    assert_eq!(
        before.segments["group-state"].bytes(),
        after.segments["group-state"].bytes()
    );
    state
        .process_data("events", input(false, 1, 16), &context, &mut collector)
        .await
        .unwrap();
    assert_eq!(
        rows(
            collector.drain("output")[0]
                .as_data()
                .unwrap()
                .table_payload()
                .unwrap()
                .batches()
        ),
        expected(false, 1)
    );
    assert!(
        state
            .incremental
            .as_ref()
            .unwrap()
            .key_work
            .columnar_chunks
            .load(Ordering::SeqCst)
            > 0
    );
}
