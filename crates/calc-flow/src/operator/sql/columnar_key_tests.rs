use super::*;
use crate::{
    BatchMetadata, CancellationToken, DataFusionConfig, JsonMap, StateBudget, StreamJobContext,
};
use datafusion::arrow::array::{Int64Array, StringArray};
use std::sync::atomic::Ordering;

#[path = "columnar_key_semantic_tests.rs"]
mod semantics;

const GROUPS: usize = 256;
const ROWS: usize = 32768;
const SQL: &str =
    "SELECT left_key, right_key, SUM(value) AS total FROM events GROUP BY left_key, right_key";

fn assert_layout() {
    type LegacyInput = (
        KeyIndex,
        CandidateMap,
        usize,
        Option<NativeKeys>,
        Option<PartialGroups>,
        Option<grouped_float::Proof>,
    );
    eprintln!(
        "Columnar structure: plan={} input={}/legacy{} key_plan={} optional_key_plan={} Group={} Candidate={}",
        size_of::<IncrementalSql>(),
        size_of::<InputCandidates>(),
        size_of::<LegacyInput>(),
        size_of::<columnar_keys::Plan>(),
        size_of::<Option<columnar_keys::Plan>>(),
        size_of::<Group>(),
        size_of::<Candidate>()
    );
    assert!(size_of::<IncrementalSql>() <= 4096);
    assert!(size_of::<InputCandidates>() <= 4096);
    assert_eq!(size_of::<columnar_keys::Plan>(), size_of::<[usize; 2]>());
    assert!(size_of::<Option<columnar_keys::Plan>>() <= size_of::<[usize; 3]>());
    assert!(
        size_of::<InputCandidates>()
            <= size_of::<LegacyInput>()
                + size_of::<Option<Box<columnar_keys::Dictionary>>>()
                + align_of::<InputCandidates>()
    );
}

fn key(group: usize, right: bool) -> Option<String> {
    if !right && group == GROUPS - 1 {
        return None;
    }
    Some(format!(
        "key-{group}-{}-\0é😀",
        if right {
            "R".repeat(80)
        } else {
            "L".repeat(80)
        }
    ))
}

fn cost_input() -> Batch {
    let keys = [false, true].map(|right| {
        Arc::new(
            (0..ROWS)
                .map(|row| key(row % GROUPS, right))
                .collect::<StringArray>(),
        ) as ArrayRef
    });
    let schema = Arc::new(Schema::new(vec![
        Field::new("left_key", DataType::Utf8, true),
        Field::new("right_key", DataType::Utf8, true),
        Field::new("value", DataType::Int64, true),
    ]));
    let record = RecordBatch::try_new(
        schema,
        vec![
            keys[0].clone(),
            keys[1].clone(),
            Arc::new(Int64Array::from(vec![Some(1); ROWS])),
        ],
    )
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

fn input_bytes(batch: &Batch) -> usize {
    batch
        .table_payload()
        .unwrap()
        .batches()
        .iter()
        .map(|record| {
            record.columns()[..2]
                .iter()
                .map(|array| {
                    let array = array.as_any().downcast_ref::<StringArray>().unwrap();
                    let offsets = array.value_offsets();
                    usize::try_from(offsets[offsets.len() - 1] - offsets[0]).unwrap()
                })
                .sum::<usize>()
        })
        .sum()
}

fn unique_encoding_bytes(batch: &Batch) -> usize {
    let converter = RowConverter::new(vec![
        SortField::new(DataType::Utf8),
        SortField::new(DataType::Utf8),
    ])
    .unwrap();
    let record = &batch.table_payload().unwrap().batches()[0];
    let arrays = record.columns()[..2]
        .iter()
        .map(|array| array.slice(0, GROUPS))
        .collect::<Vec<_>>();
    converter
        .convert_columns(&arrays)
        .unwrap()
        .iter()
        .map(|row| row.as_ref().len())
        .sum()
}

async fn plan(runtime: &DataFusionRuntime, input: &Batch) -> IncrementalSql {
    let query = crate::expression::parse_select_query(SQL).unwrap();
    let mut plan = IncrementalSql::plan(
        runtime,
        &query,
        "events",
        input.table_payload().unwrap().schema().clone(),
        "columnar-cost",
    )
    .await
    .unwrap()
    .unwrap();
    plan.set_compact_budget(
        Some(
            StateBudget::new(
                u64::try_from(GROUPS).unwrap(),
                StateBudget::default().max_bytes(),
            )
            .unwrap(),
        ),
        "columnar-cost",
    )
    .unwrap();
    plan
}

fn assert_output(records: &[RecordBatch]) {
    let actual = records
        .iter()
        .flat_map(|record| {
            (0..record.num_rows()).map(|row| {
                record
                    .columns()
                    .iter()
                    .map(|array| ScalarValue::try_from_array(array, row).unwrap())
                    .collect::<Vec<_>>()
            })
        })
        .collect::<Vec<_>>();
    let expected = (0..GROUPS)
        .map(|group| {
            vec![
                ScalarValue::Utf8(key(group, false)),
                ScalarValue::Utf8(key(group, true)),
                ScalarValue::Int64(Some(i64::try_from(ROWS / GROUPS).unwrap())),
            ]
        })
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
}

#[test]
fn test_columnar_keys_reduce_actual_encoding_and_whole_update_allocated_bytes() {
    assert_layout();
    let executor = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let input = cost_input();
    let mut plan = executor.block_on(plan(&runtime, &input));
    let key_bytes = input_bytes(&input);
    let unique_bytes = unique_encoding_bytes(&input);
    let job = StreamJobContext::new(
        976,
        "columnar-cost",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "columnar-cost", None);
    let compact_before = runtime.compact_state_bytes();
    runtime.take_compact_state_peak();
    let mut records = None;
    let allocations = allocation_counter::measure(|| {
        let transaction = executor
            .block_on(plan.update(&input, &context, "columnar-cost"))
            .unwrap();
        records = Some(transaction.records.clone());
        plan.commit(transaction);
    });
    let rows = plan.encoded_rows.load(Ordering::SeqCst);
    let bytes = plan.key_work.encoded_bytes.load(Ordering::SeqCst);
    let paid = plan.key_work.workspace_peak.load(Ordering::SeqCst)
        + runtime.take_compact_state_peak()
        - compact_before;
    eprintln!(
        "Whole columnar update: rows={rows}, encoded_bytes={bytes}, unique_bytes={unique_bytes}, input_key_bytes={key_bytes}, paid_peak_bound={paid}, allocations={allocations:?}"
    );
    assert_output(records.as_ref().unwrap());
    assert!(allocations.bytes_max <= u64::try_from(paid).unwrap());
    assert!(
        allocations.bytes_total < u64::try_from(key_bytes).unwrap(),
        "whole update must avoid repeated composite-key byte allocation: {allocations:?}"
    );
    assert_eq!(
        rows, GROUPS,
        "canonical encoding must visit only transaction-distinct keys"
    );
    assert_eq!(bytes, unique_bytes);
    drop((records, plan));
    assert_eq!(runtime.incremental_memory_pool().reserved(), 0);
}
