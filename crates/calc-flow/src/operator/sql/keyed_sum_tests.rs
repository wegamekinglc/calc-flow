use super::*;
use crate::{
    CancellationToken, DataFusionConfig, DataFusionParallelismMode, JsonMap, StreamJobContext,
};
use datafusion::arrow::array::{Int64Array, StringArray};

const QUERY: &str = "SELECT key, label, SUM(value) FILTER (WHERE selected) AS total \
                     FROM events WHERE keep GROUP BY key, label";

#[derive(Clone)]
struct Row {
    key: Option<i64>,
    label: Option<&'static str>,
    value: Option<u64>,
    selected: bool,
    keep: bool,
}

fn fixture(unsigned: bool) -> Vec<Row> {
    let mut rows = vec![
        Row {
            key: Some(2),
            label: Some("b"),
            value: Some(4),
            selected: true,
            keep: true,
        },
        Row {
            key: None,
            label: None,
            value: None,
            selected: true,
            keep: true,
        },
        Row {
            key: Some(1),
            label: Some("a"),
            value: Some(7),
            selected: false,
            keep: true,
        },
        Row {
            key: Some(2),
            label: Some("b"),
            value: Some(3),
            selected: true,
            keep: true,
        },
        Row {
            key: Some(9),
            label: Some("dropped"),
            value: Some(100),
            selected: true,
            keep: false,
        },
        Row {
            key: Some(1),
            label: Some("a"),
            value: Some(5),
            selected: true,
            keep: true,
        },
    ];
    rows.extend((0..37i64).map(|index| Row {
        key: Some(index % 5),
        label: (index % 3 != 0).then_some("same"),
        value: (index % 4 != 0).then_some(u64::try_from(index).unwrap()),
        selected: index % 6 != 0,
        keep: true,
    }));
    if unsigned {
        rows.push(Row {
            key: Some(2),
            label: Some("b"),
            value: Some(u64::MAX),
            selected: true,
            keep: true,
        });
    } else {
        rows.push(Row {
            key: Some(2),
            label: Some("b"),
            value: Some(u64::try_from(i64::MAX).unwrap()),
            selected: true,
            keep: true,
        });
    }
    rows
}

fn record(unsigned: bool, rows: &[Row]) -> RecordBatch {
    let values = if unsigned {
        Arc::new(UInt64Array::from(
            rows.iter().map(|row| row.value).collect::<Vec<_>>(),
        )) as ArrayRef
    } else {
        Arc::new(Int64Array::from(
            rows.iter()
                .map(|row| {
                    row.value
                        .map(|value| i64::from_ne_bytes(value.to_ne_bytes()))
                })
                .collect::<Vec<_>>(),
        )) as ArrayRef
    };
    RecordBatch::try_from_iter(vec![
        (
            "key",
            Arc::new(Int64Array::from(
                rows.iter().map(|row| row.key).collect::<Vec<_>>(),
            )) as ArrayRef,
        ),
        (
            "label",
            Arc::new(StringArray::from(
                rows.iter().map(|row| row.label).collect::<Vec<_>>(),
            )) as ArrayRef,
        ),
        ("value", values),
        (
            "selected",
            Arc::new(BooleanArray::from(
                rows.iter().map(|row| row.selected).collect::<Vec<_>>(),
            )) as ArrayRef,
        ),
        (
            "keep",
            Arc::new(BooleanArray::from(
                rows.iter().map(|row| row.keep).collect::<Vec<_>>(),
            )) as ArrayRef,
        ),
    ])
    .unwrap()
}

async fn plan(runtime: &DataFusionRuntime, unsigned: bool) -> IncrementalSql {
    let input = record(unsigned, &fixture(unsigned));
    let query = crate::expression::parse_select_query(QUERY).unwrap();
    IncrementalSql::plan(runtime, &query, "events", input.schema(), "keyed-sum")
        .await
        .unwrap()
        .unwrap()
}

fn output_rows(records: &[RecordBatch]) -> Vec<Vec<ScalarValue>> {
    records
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
        .collect()
}

fn expected(unsigned: bool, rows: &[Row]) -> Vec<Vec<ScalarValue>> {
    let mut keys: Vec<(Option<i64>, Option<&str>)> = Vec::new();
    let mut signed = Vec::<Option<i64>>::new();
    let mut unsigned_sums = Vec::<Option<u64>>::new();
    for row in rows.iter().filter(|row| row.keep) {
        let key = (row.key, row.label);
        let group = if let Some(group) = keys.iter().position(|candidate| *candidate == key) {
            group
        } else {
            keys.push(key);
            signed.push(None);
            unsigned_sums.push(None);
            keys.len() - 1
        };
        if row.selected
            && let Some(value) = row.value
        {
            if unsigned {
                unsigned_sums[group] = Some(unsigned_sums[group].unwrap_or(0).wrapping_add(value));
            } else {
                signed[group] = Some(
                    signed[group]
                        .unwrap_or(0)
                        .wrapping_add(i64::from_ne_bytes(value.to_ne_bytes())),
                );
            }
        }
    }
    keys.into_iter()
        .enumerate()
        .map(|(group, (key, label))| {
            let value = if unsigned {
                ScalarValue::UInt64(unsigned_sums[group])
            } else {
                ScalarValue::Int64(signed[group])
            };
            vec![
                ScalarValue::Int64(key),
                ScalarValue::Utf8(label.map(str::to_owned)),
                value,
            ]
        })
        .collect()
}

#[tokio::test]
async fn test_keyed_integer_sum_matches_naive_oracle_across_batch_splits() {
    for unsigned in [false, true] {
        let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
        let job = StreamJobContext::new(
            982,
            "keyed-sum",
            JsonMap::new(),
            None,
            CancellationToken::new(),
        );
        let context = StreamOperatorContext::new(&job, "keyed-sum", None);
        let rows = fixture(unsigned);
        let first = record(unsigned, &rows);
        for split in [1, 7, 4096] {
            let mut plan = plan(&runtime, unsigned).await;
            let mut history = Vec::new();
            for start in (0..rows.len()).step_by(split) {
                let end = (start + split).min(rows.len());
                let chunk = first.slice(start, end - start);
                history.extend_from_slice(&rows[start..end]);
                let batch = Batch::table(vec![chunk], crate::BatchMetadata::default()).unwrap();
                let transaction = plan.update(&batch, &context, "keyed-sum").await.unwrap();
                assert_eq!(
                    output_rows(&transaction.records),
                    expected(unsigned, &history)
                );
                plan.commit(transaction);
                assert!(plan.groups.is_empty());
                assert!(plan.index.is_empty());
            }
        }
        assert_eq!(runtime.incremental_memory_pool().reserved(), 0);
    }
}

#[tokio::test]
async fn test_keyed_integer_sum_discard_restores_values_births_ids_and_charges() {
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let mut plan = plan(&runtime, false).await;
    let job = StreamJobContext::new(
        983,
        "keyed-sum",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "keyed-sum", None);
    let rows = fixture(false);
    let full = record(false, &rows);
    let initial = Batch::table(vec![full.slice(0, 4)], crate::BatchMetadata::default()).unwrap();
    let transaction = plan.update(&initial, &context, "keyed-sum").await.unwrap();
    plan.commit(transaction);
    let committed_births = plan.keyed_sum.as_ref().unwrap().canonical_births();
    let before = plan.export_native_state("keyed-sum", || Ok(())).unwrap();
    assert_eq!(
        plan.keyed_sum.as_ref().unwrap().canonical_births(),
        committed_births,
        "an unchanged checkpoint must not re-frame canonical keys"
    );
    let keyed = plan.keyed_sum.as_ref().unwrap().clone();
    let baseline = (
        keyed.canonical_births(),
        keyed.next_id(),
        keyed.reserved_bytes(),
        runtime.incremental_memory_pool().reserved(),
    );
    let next = Batch::table(
        vec![full.slice(4, rows.len() - 4)],
        crate::BatchMetadata::default(),
    )
    .unwrap();
    let discarded = plan.update(&next, &context, "keyed-sum").await.unwrap();
    assert!(discarded.changes_state());
    drop(discarded);
    let after = plan.export_native_state("keyed-sum", || Ok(())).unwrap();
    assert_eq!(before.records(), after.records());
    assert_eq!(keyed.canonical_births(), baseline.0);
    assert_eq!(keyed.next_id(), baseline.1);
    assert_eq!(keyed.reserved_bytes(), baseline.2);
    drop(after);
    assert_eq!(runtime.incremental_memory_pool().reserved(), baseline.3);

    let transaction = plan.update(&next, &context, "keyed-sum").await.unwrap();
    assert_eq!(output_rows(&transaction.records), expected(false, &rows));
    plan.commit(transaction);
    drop((before, plan, keyed));
    assert_eq!(runtime.incremental_memory_pool().reserved(), 0);
    assert_eq!(runtime.compact_state_bytes(), 0);
}

#[tokio::test]
async fn test_keyed_integer_sum_restore_reexports_identically_and_continues() {
    for unsigned in [false, true] {
        let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
        let mut original = plan(&runtime, unsigned).await;
        let job = StreamJobContext::new(
            984,
            "keyed-sum",
            JsonMap::new(),
            None,
            CancellationToken::new(),
        );
        let context = StreamOperatorContext::new(&job, "keyed-sum", None);
        let rows = fixture(unsigned);
        let full = record(unsigned, &rows);
        let initial =
            Batch::table(vec![full.slice(0, 8)], crate::BatchMetadata::default()).unwrap();
        let transaction = original
            .update(&initial, &context, "keyed-sum")
            .await
            .unwrap();
        original.commit(transaction);
        let exported = original
            .export_native_state("keyed-sum", || Ok(()))
            .unwrap();
        let canonical = original.keyed_sum.as_ref().unwrap().canonical_bytes();
        let mut restored = plan(&runtime, unsigned)
            .await
            .import_native_state(
                exported.records(),
                u64::try_from(8).unwrap(),
                true,
                || Ok(()),
                "keyed-sum",
            )
            .unwrap();
        let reexported = restored
            .export_native_state("keyed-sum", || Ok(()))
            .unwrap();
        assert_eq!(exported.records(), reexported.records());
        assert_eq!(
            restored.keyed_sum.as_ref().unwrap().canonical_bytes(),
            canonical
        );
        let next = Batch::table(
            vec![full.slice(8, rows.len() - 8)],
            crate::BatchMetadata::default(),
        )
        .unwrap();
        let left = original.update(&next, &context, "keyed-sum").await.unwrap();
        let right = restored.update(&next, &context, "keyed-sum").await.unwrap();
        assert_eq!(left.records, right.records);
        assert_eq!(output_rows(&right.records), expected(unsigned, &rows));
        original.commit(left);
        restored.commit(right);
        assert_eq!(
            original.keyed_sum.as_ref().unwrap().layout_fingerprint(),
            restored.keyed_sum.as_ref().unwrap().layout_fingerprint()
        );
        drop((exported, reexported, original, restored));
        assert_eq!(runtime.incremental_memory_pool().reserved(), 0);
    }
}

fn partitioned(plan: &mut IncrementalSql, partitions: usize) {
    let keyed = plan
        .keyed_sum
        .take()
        .unwrap()
        .with_partition_count_for_test(partitions);
    plan.keyed_sum = Some(keyed);
}

#[tokio::test]
async fn test_keyed_sum_fixed_partitions_preserve_output_and_owner() {
    for unsigned in [false, true] {
        let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
        let job = StreamJobContext::new(
            985,
            "keyed-sum",
            JsonMap::new(),
            None,
            CancellationToken::new(),
        );
        let context = StreamOperatorContext::new(&job, "keyed-sum", None);
        let rows = fixture(unsigned);
        let full = record(unsigned, &rows);
        let full_batch = Batch::table(vec![full], crate::BatchMetadata::default()).unwrap();
        for partitions in [1, 2, 4] {
            let mut plan = plan(&runtime, unsigned).await;
            partitioned(&mut plan, partitions);
            let first = plan
                .update(&full_batch, &context, "keyed-sum")
                .await
                .unwrap();
            assert_eq!(output_rows(&first.records), expected(unsigned, &rows));
            plan.commit(first);
            let keyed = plan.keyed_sum.as_ref().unwrap().clone();
            assert_eq!(keyed.partition_count_for_test(), partitions);
            let owners = (0..keyed.len())
                .map(|group| {
                    let group = crate::operator::keyed::GroupId::new(group).unwrap();
                    keyed.owner_for_test(group)
                })
                .collect::<Vec<_>>();
            let expected_owners = (0..keyed.len())
                .map(|group| group % partitions)
                .collect::<Vec<_>>();
            assert_eq!(owners, expected_owners);

            let second = plan
                .update(&full_batch, &context, "keyed-sum")
                .await
                .unwrap();
            let history = [rows.clone(), rows.clone()].concat();
            assert_eq!(output_rows(&second.records), expected(unsigned, &history));
            plan.commit(second);
            let owners_after = (0..keyed.len())
                .map(|group| {
                    let group = crate::operator::keyed::GroupId::new(group).unwrap();
                    keyed.owner_for_test(group)
                })
                .collect::<Vec<_>>();
            assert_eq!(owners_after, owners);
        }
        assert_eq!(runtime.incremental_memory_pool().reserved(), 0);
    }
}

#[tokio::test]
async fn test_keyed_sum_parallel_workers_preserve_deterministic_output() {
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let job = StreamJobContext::new(
        987,
        "keyed-sum",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "keyed-sum", None);
    let rows = fixture(false);
    let full = record(false, &rows);
    let full = Batch::table(vec![full], crate::BatchMetadata::default()).unwrap();
    let mut plan = plan(&runtime, false).await;
    partitioned(&mut plan, 2);
    plan.keyed_sum.as_ref().unwrap().force_parallel_for_test();

    let transaction = plan.update(&full, &context, "keyed-sum").await.unwrap();
    assert_eq!(output_rows(&transaction.records), expected(false, &rows));
    assert!(plan.keyed_sum.as_ref().unwrap().worker_count_for_test() >= 2);
    plan.commit(transaction);
}

#[tokio::test]
async fn test_keyed_sum_worker_panic_rolls_back_every_partition() {
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let job = StreamJobContext::new(
        988,
        "keyed-sum",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "keyed-sum", None);
    let rows = fixture(false);
    let full = record(false, &rows);
    let mut plan = plan(&runtime, false).await;
    partitioned(&mut plan, 2);
    plan.keyed_sum.as_ref().unwrap().force_parallel_for_test();
    let initial = Batch::table(vec![full.slice(0, 8)], crate::BatchMetadata::default()).unwrap();
    let transaction = plan.update(&initial, &context, "keyed-sum").await.unwrap();
    plan.commit(transaction);
    let before = plan.export_native_state("keyed-sum", || Ok(())).unwrap();
    plan.keyed_sum
        .as_ref()
        .unwrap()
        .panic_group_for_test(crate::operator::keyed::GroupId::new(0).unwrap());
    let remaining = Batch::table(
        vec![full.slice(8, rows.len() - 8)],
        crate::BatchMetadata::default(),
    )
    .unwrap();

    assert!(
        plan.update(&remaining, &context, "keyed-sum")
            .await
            .is_err()
    );
    let after = plan.export_native_state("keyed-sum", || Ok(())).unwrap();
    assert_eq!(before.records(), after.records());

    plan.keyed_sum.as_ref().unwrap().clear_panic_for_test();
    let transaction = plan
        .update(&remaining, &context, "keyed-sum")
        .await
        .unwrap();
    assert_eq!(output_rows(&transaction.records), expected(false, &rows));
    plan.commit(transaction);
}

#[tokio::test]
async fn test_keyed_sum_restore_rejects_partition_count_mismatch() {
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let job = StreamJobContext::new(
        986,
        "keyed-sum",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "keyed-sum", None);
    let rows = fixture(false);
    let full = record(false, &rows);
    let mut original = plan(&runtime, false).await;
    partitioned(&mut original, 1);
    let initial = Batch::table(vec![full.slice(0, 8)], crate::BatchMetadata::default()).unwrap();
    let transaction = original
        .update(&initial, &context, "keyed-sum")
        .await
        .unwrap();
    original.commit(transaction);
    let exported = original
        .export_native_state("keyed-sum", || Ok(()))
        .unwrap();
    let mut restored = plan(&runtime, false).await;
    partitioned(&mut restored, 2);

    assert!(
        restored
            .import_native_state(exported.records(), 8, true, || Ok(()), "keyed-sum")
            .is_err()
    );
}

#[tokio::test]
async fn test_keyed_sum_ten_thousand_stock_groups_continue() {
    let runtime = DataFusionRuntime::new(DataFusionConfig {
        target_partitions: 1,
        min_rows_per_partition: 1,
        parallelism_mode: DataFusionParallelismMode::Fixed,
        ..DataFusionConfig::default()
    })
    .unwrap();
    let rows = 10_000;
    let record = RecordBatch::try_from_iter(vec![
        (
            "account",
            Arc::new(StringArray::from(
                (0..rows)
                    .map(|row| format!("account-{:03}", row / 100))
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
        ),
        (
            "stock",
            Arc::new(StringArray::from(
                (0..rows)
                    .map(|row| format!("stock-{:03}", row % 100))
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
        ),
        (
            "delta",
            Arc::new(Int64Array::from(vec![1; rows])) as ArrayRef,
        ),
    ])
    .unwrap();
    let query = crate::expression::parse_select_query(
        "SELECT account, stock, SUM(delta) AS position FROM input GROUP BY account, stock",
    )
    .unwrap();
    let mut plan = IncrementalSql::plan(&runtime, &query, "input", record.schema(), "keyed-sum")
        .await
        .unwrap()
        .unwrap();
    let job = StreamJobContext::new(
        987,
        "keyed-sum",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "keyed-sum", None);
    let batch = Batch::table(vec![record.clone()], crate::BatchMetadata::default()).unwrap();
    let transaction = plan.update(&batch, &context, "keyed-sum").await.unwrap();
    plan.commit(transaction);

    let wide = RecordBatch::try_from_iter(vec![
        (
            "account",
            Arc::new(StringArray::from(
                (0..100_000)
                    .map(|row| format!("account-{:03}", row % rows / 100))
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
        ),
        (
            "stock",
            Arc::new(StringArray::from(
                (0..100_000)
                    .map(|row| format!("stock-{:03}", row % 100))
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
        ),
        (
            "delta",
            Arc::new(Int64Array::from(vec![1; 100_000])) as ArrayRef,
        ),
    ])
    .unwrap();
    let wide = Batch::table(vec![wide], crate::BatchMetadata::default()).unwrap();
    let transaction = plan.update(&wide, &context, "keyed-sum").await.unwrap();
    assert_eq!(
        transaction
            .records
            .iter()
            .map(RecordBatch::num_rows)
            .sum::<usize>(),
        rows
    );
    plan.commit(transaction);
}
