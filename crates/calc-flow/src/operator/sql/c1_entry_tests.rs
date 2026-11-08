use super::*;
use crate::{CancellationToken, DataFusionConfig, JsonMap, StreamJobContext};
use datafusion::arrow::array::{Array, Int64Array, StringArray};
use std::collections::{BTreeMap, BTreeSet};

const NAME: &str = "c1-entry";
const GROUPS: usize = 10_000;
const APPEND_ROWS: usize = 1_000;
const APPENDS: usize = 20;
const SQL: &str = "SELECT account, stock, SUM(value) AS total FROM events GROUP BY account, stock";

#[path = "c1_cache_tests.rs"]
mod cache_tests;

type Key = (Option<String>, String);
type Totals = BTreeMap<Key, i64>;

struct Fixture {
    plan: IncrementalSql,
    keys: Vec<Key>,
    original: Vec<RecordBatch>,
    initial: Totals,
    expected: Totals,
}

fn warm_keys() -> Vec<Key> {
    (0..GROUPS)
        .map(|slot| {
            (
                (slot != 0).then(|| format!("account-{:03}", slot / 100)),
                format!("stock-{:03}", slot % 100),
            )
        })
        .collect()
}

fn input(keys: &[Key], value: i64) -> Batch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("account", DataType::Utf8, true),
        Field::new("stock", DataType::Utf8, false),
        Field::new("value", DataType::Int64, false),
    ]));
    let record = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(
                keys.iter()
                    .map(|(account, _)| account.as_deref())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from_iter_values(
                keys.iter().map(|(_, stock)| stock.as_str()),
            )) as ArrayRef,
            Arc::new(Int64Array::from(vec![value; keys.len()])) as ArrayRef,
        ],
    )
    .unwrap();
    Batch::table(vec![record], crate::BatchMetadata::default()).unwrap()
}

fn string_column(record: &RecordBatch, column: usize) -> &StringArray {
    record
        .column(column)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
}

fn record_keys(record: &RecordBatch) -> Vec<Key> {
    let accounts = string_column(record, 0);
    let stocks = string_column(record, 1);
    assert_eq!(stocks.null_count(), 0);
    (0..record.num_rows())
        .map(|row| {
            (
                (!accounts.is_null(row)).then(|| accounts.value(row).to_owned()),
                stocks.value(row).to_owned(),
            )
        })
        .collect()
}

fn totals(records: &[RecordBatch]) -> Totals {
    let mut observed = Totals::new();
    for record in records {
        let values = record
            .column(2)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(values.null_count(), 0);
        for (row, key) in record_keys(record).into_iter().enumerate() {
            assert!(observed.insert(key, values.value(row)).is_none());
        }
    }
    observed
}

fn group_key(group: &Group) -> Key {
    let [ScalarValue::Utf8(account), ScalarValue::Utf8(Some(stock))] = group.values.as_ref() else {
        panic!("fixture requires two Utf8 group keys");
    };
    (account.clone(), stock.clone())
}

fn state_totals(plan: &IncrementalSql) -> Totals {
    plan.groups
        .iter()
        .map(|group| {
            let ScalarValue::Int64(Some(value)) = *group.aggregates.result(0) else {
                panic!("fixture requires a non-null Int64 SUM");
            };
            (group_key(group), value)
        })
        .collect()
}

fn add_values(expected: &mut Totals, keys: &[Key], value: i64) {
    for key in keys {
        *expected.entry(key.clone()).or_default() += value;
    }
}

fn assert_snapshot(records: &[RecordBatch], keys: &[Key], expected: &Totals) {
    assert_eq!(
        records
            .iter()
            .map(RecordBatch::num_rows)
            .collect::<Vec<_>>(),
        vec![CHUNK_ROWS, GROUPS - CHUNK_ROWS]
    );
    for record in records {
        let fields = record
            .schema()
            .fields()
            .iter()
            .map(|field| (field.name().clone(), field.data_type().clone()))
            .collect::<Vec<_>>();
        assert_eq!(
            fields,
            vec![
                ("account".into(), DataType::Utf8),
                ("stock".into(), DataType::Utf8),
                ("total".into(), DataType::Int64),
            ]
        );
    }
    let observed_keys = records.iter().flat_map(record_keys).collect::<Vec<_>>();
    assert_eq!(observed_keys, keys);
    assert_eq!(&totals(records), expected);
}

async fn fixture(context: &StreamOperatorContext<'_>) -> Fixture {
    let keys = warm_keys();
    let batch = input(&keys, 1);
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let query = crate::expression::parse_select_query(SQL).unwrap();
    let mut plan = IncrementalSql::plan(
        &runtime,
        &query,
        "events",
        batch.table_payload().unwrap().schema().clone(),
        NAME,
    )
    .await
    .unwrap()
    .expect("ordinary grouped SUM must use compact native state");
    let warm = plan.update(&batch, context, NAME).await.unwrap();
    assert_eq!(warm.rows, GROUPS);
    let original = warm.records.clone();
    let mut initial = Totals::new();
    add_values(&mut initial, &keys, 1);
    assert_snapshot(&original, &keys, &initial);
    plan.commit(warm);
    let native_keys = plan.groups.iter().map(group_key).collect::<Vec<_>>();
    assert_eq!(native_keys, keys);
    assert_eq!(state_totals(&plan), initial);
    assert_eq!(totals(batch.table_payload().unwrap().batches()), initial);
    Fixture {
        plan,
        keys: native_keys,
        original,
        expected: initial.clone(),
        initial,
    }
}

fn assert_candidates(
    transaction: &Transaction,
    target_slots: &BTreeSet<usize>,
    keys: &[Key],
    expected_counts: &[usize; 2],
) -> [usize; 2] {
    assert_eq!(transaction.rows, APPEND_ROWS);
    assert!(transaction.new_groups.is_empty());
    assert!(transaction.container.is_none());
    let observed_slots = transaction
        .groups
        .iter()
        .map(|(slot, _)| *slot)
        .collect::<BTreeSet<_>>();
    assert_eq!(&observed_slots, target_slots);
    assert_eq!(transaction.groups.len(), APPEND_ROWS);
    let mut counts = [0; 2];
    for (slot, group) in &transaction.groups {
        assert_eq!(group_key(group), keys[*slot]);
        counts[slot / CHUNK_ROWS] += 1;
    }
    assert_eq!(&counts, expected_counts);
    counts
}

fn assert_key_columns_shared(original: &[RecordBatch], current: &[RecordBatch]) {
    for (before, after) in original.iter().zip(current) {
        for column in 0..2 {
            assert!(Arc::ptr_eq(before.column(column), after.column(column)));
        }
        assert_eq!(before.schema(), after.schema());
    }
}

async fn append(
    fixture: &mut Fixture,
    context: &StreamOperatorContext<'_>,
    target_slots: &BTreeSet<usize>,
    expected_counts: &[usize; 2],
    value: i64,
) -> [usize; 2] {
    let keys = target_slots
        .iter()
        .map(|slot| fixture.keys[*slot].clone())
        .collect::<Vec<_>>();
    let batch = input(&keys, value);
    let committed = fixture.expected.clone();
    let transaction = fixture.plan.update(&batch, context, NAME).await.unwrap();
    let counts = assert_candidates(&transaction, target_slots, &fixture.keys, expected_counts);
    assert_eq!(state_totals(&fixture.plan), committed);
    add_values(&mut fixture.expected, &keys, value);
    assert_snapshot(&transaction.records, &fixture.keys, &fixture.expected);
    assert_key_columns_shared(&fixture.original, &transaction.records);
    fixture.plan.commit(transaction);
    assert_eq!(fixture.plan.groups.len(), GROUPS);
    assert_eq!(state_totals(&fixture.plan), fixture.expected);
    assert_snapshot(&fixture.original, &fixture.keys, &fixture.initial);
    let mut expected_input = Totals::new();
    add_values(&mut expected_input, &keys, value);
    assert_eq!(
        totals(batch.table_payload().unwrap().batches()),
        expected_input
    );
    counts
}

async fn witness(target_slots: BTreeSet<usize>, expected_counts: [usize; 2]) {
    assert_eq!(CHUNK_ROWS, 8192);
    assert_eq!(target_slots.len(), APPEND_ROWS);
    let job = StreamJobContext::new(973, NAME, JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, NAME, None);
    let mut fixture = fixture(&context).await;
    let mut observed = Vec::new();
    for index in 0..APPENDS {
        observed.push(
            append(
                &mut fixture,
                &context,
                &target_slots,
                &expected_counts,
                i64::try_from(index + 1).unwrap(),
            )
            .await,
        );
    }
    eprintln!(
        "C1 native entry: groups={GROUPS}, appends={APPENDS}, rows_per_append={APPEND_ROWS}, observed_candidate_slots_per_chunk={observed:?}"
    );
}

#[tokio::test]
async fn test_c1_sparse_existing_group_updates_leave_first_snapshot_chunk_clean() {
    witness((9000..GROUPS).collect(), [0, APPEND_ROWS]).await;
}

#[tokio::test]
async fn test_c1_dense_existing_group_updates_touch_every_snapshot_chunk() {
    witness(
        (0..APPEND_ROWS)
            .map(|index| (index * 37) % GROUPS)
            .collect(),
        [853, 147],
    )
    .await;
}

#[tokio::test]
async fn test_c1_snapshot_reuses_only_clean_aggregate_chunks() {
    let job = StreamJobContext::new(974, NAME, JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, NAME, None);
    let mut fixture = fixture(&context).await;
    let changed_keys = fixture.keys[9000..].to_vec();
    let input = input(&changed_keys, 7);
    let transaction = fixture.plan.update(&input, &context, NAME).await.unwrap();

    assert!(Arc::ptr_eq(
        fixture.original[0].column(2),
        transaction.records[0].column(2),
    ));
    assert!(!Arc::ptr_eq(
        fixture.original[1].column(2),
        transaction.records[1].column(2),
    ));
    add_values(&mut fixture.expected, &changed_keys, 7);
    assert_snapshot(&transaction.records, &fixture.keys, &fixture.expected);
    assert_snapshot(&fixture.original, &fixture.keys, &fixture.initial);
    assert_eq!(state_totals(&fixture.plan), fixture.initial);
    fixture.plan.commit(transaction);
    assert_eq!(state_totals(&fixture.plan), fixture.expected);
}
