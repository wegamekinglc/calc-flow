use std::{collections::BTreeMap, sync::Arc};

use datafusion::{
    arrow::{
        array::{ArrayRef, Int64Array, StringArray},
        datatypes::{DataType, Field, Schema, SchemaRef},
        record_batch::RecordBatch,
    },
    common::ScalarValue,
};

use super::super::*;
use crate::{CancellationToken, EdgeCollector, StreamJobContext};

const QUERY: &str = "SELECT key, SUM(value) AS total, COUNT(*) AS rows FROM events GROUP BY key";

pub(super) fn schema(wide: bool) -> SchemaRef {
    let mut fields = vec![
        Field::new("key", DataType::Utf8, true),
        Field::new("value", DataType::Int64, true),
    ];
    if wide {
        fields.push(Field::new("unused", DataType::Utf8, true));
    }
    Arc::new(Schema::new_with_metadata(
        fields,
        [("origin".into(), "compact-storage".into())].into(),
    ))
}

pub(super) fn input(
    wide: bool,
    keys: &[Option<&str>],
    values: &[Option<i64>],
    sequence: u64,
) -> Batch {
    let mut columns = vec![
        Arc::new(StringArray::from(keys.to_vec())) as ArrayRef,
        Arc::new(Int64Array::from(values.to_vec())),
    ];
    if wide {
        columns.push(Arc::new(StringArray::from(vec![
            "unused payload";
            keys.len()
        ])));
    }
    Batch::table(
        vec![RecordBatch::try_new(schema(wide), columns).unwrap()],
        BatchMetadata::new(
            "compact-storage",
            sequence,
            JsonMap::from([("prefix".into(), json!(sequence))]),
        )
        .unwrap(),
    )
    .unwrap()
}

pub(super) fn operator(wide: bool, declared: bool) -> SqlOperator {
    SqlOperator::new("compact", QUERY, vec!["events".into()], vec![])
        .unwrap()
        .with_ports(
            vec![
                Port::with_schema_ref(
                    "events",
                    BatchKind::Table,
                    true,
                    declared.then(|| schema(wide)),
                )
                .unwrap(),
            ],
            Port::new("output", BatchKind::Table, true, None).unwrap(),
        )
        .unwrap()
}

pub(super) fn rows(batch: &Batch) -> Vec<Vec<ScalarValue>> {
    let mut rows = batch
        .table_payload()
        .unwrap()
        .batches()
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
    rows.sort_by(|a, b| a.partial_cmp(b).unwrap());
    rows
}

pub(super) async fn process(
    operator: &mut SqlOperator,
    batch: Batch,
    context: &StreamOperatorContext<'_>,
) -> Batch {
    let weak = batch.table_payload().unwrap().batches()[0]
        .columns()
        .iter()
        .map(Arc::downgrade)
        .collect::<Vec<_>>();
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("events", batch, context, &mut output)
        .await
        .unwrap();
    let result = output.drain("output");
    assert_eq!(result.len(), 1);
    assert!(operator.retained.is_none());
    assert!(weak.iter().all(|weak| weak.upgrade().is_none()));
    result[0].as_data().unwrap().clone()
}

pub(super) fn same_snapshot(left: &OperatorStateSnapshot, right: &OperatorStateSnapshot) {
    assert_eq!(left.inline_metadata, right.inline_metadata);
    assert_eq!(
        left.segments.keys().collect::<Vec<_>>(),
        right.segments.keys().collect::<Vec<_>>()
    );
    for (name, segment) in &left.segments {
        assert_eq!(segment.bytes(), right.segments[name].bytes());
        assert!(Arc::ptr_eq(
            &segment.bytes_arc(),
            &right.segments[name].bytes_arc()
        ));
    }
}

#[tokio::test]
async fn test_compact_storage_two_roundtrips_continue_and_recapture_every_prefix() {
    for wide in [false, true] {
        for declared in [false, true] {
            let job = StreamJobContext::new(
                41,
                "compact",
                JsonMap::new(),
                None,
                CancellationToken::new(),
            );
            let context = StreamOperatorContext::new(&job, "compact", None);
            let oracle = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
            let mut state = operator(wide, declared);
            let mut keys = Vec::new();
            let mut values = Vec::new();
            for (sequence, incoming_keys, incoming_values) in [
                (
                    0,
                    vec![Some("a"), Some("b"), None],
                    vec![Some(2), None, Some(7)],
                ),
                (1, vec![Some("a")], vec![Some(5)]),
                (
                    2,
                    vec![Some("b"), None, Some("c")],
                    vec![Some(9), None, Some(1)],
                ),
            ] {
                keys.extend_from_slice(&incoming_keys);
                values.extend_from_slice(&incoming_values);
                let batch = input(wide, &incoming_keys, &incoming_values, sequence);
                let metadata = batch.metadata().clone();
                let expected = oracle
                    .sql(
                        QUERY,
                        &BTreeMap::from([("events".into(), input(wide, &keys, &values, sequence))]),
                        Some("oracle"),
                    )
                    .await
                    .unwrap();
                let actual = process(&mut state, batch, &context).await;
                assert_eq!(
                    actual.table_payload().unwrap().schema(),
                    expected.table_payload().unwrap().schema()
                );
                assert_eq!(rows(&actual), rows(&expected));
                assert_eq!(actual.metadata(), &metadata);
                state.prepare_checkpoint_async(&context).await.unwrap();
                let snapshot = state.checkpoint(Epoch::INITIAL).unwrap();
                assert_eq!(snapshot.inline_metadata["state_layout"], json!(3));
                assert_eq!(snapshot.inline_metadata["state_accounting"], json!(3));
                assert_eq!(snapshot.inline_metadata["rows"], json!(keys.len()));
                let control: super::control::CompactControl =
                    serde_json::from_slice(snapshot.segments["control"].bytes()).unwrap();
                assert_eq!(snapshot.segments.len(), 4 + control.group_log.frames.len());
                assert_eq!(snapshot.inline_metadata.len(), 6);
                let decoded = decode_sql_state(snapshot.segments["group-state"].bytes()).unwrap();
                assert_eq!(decoded.num_rows() as u64, control.group_log.base_groups);
                assert_eq!(control.groups, if sequence < 2 { 3 } else { 4 });
                let cached = state.checkpoint(Epoch::INITIAL).unwrap();
                same_snapshot(&snapshot, &cached);
                if sequence < 2 {
                    let mut restored = operator(wide, declared);
                    StreamOperator::restore(&mut restored, &snapshot).unwrap();
                    assert!(restored.retained.is_none());
                    assert!(restored.compact.is_some());
                    same_snapshot(&snapshot, &restored.checkpoint(Epoch::INITIAL).unwrap());
                    state = restored;
                }
            }
            metadata_reset_control(&mut state, wide, &context).await;
        }
    }
}

async fn metadata_reset_control(
    state: &mut SqlOperator,
    wide: bool,
    context: &StreamOperatorContext<'_>,
) {
    let previous = state.checkpoint(Epoch::INITIAL).unwrap();
    let previous_ledger = state.compact.as_ref().unwrap().ledger;
    let metadata_only = input(wide, &[], &[], 3);
    let latest = metadata_only.metadata().clone();
    process(state, metadata_only, context).await;
    let next = state.checkpoint(Epoch::INITIAL).unwrap();
    assert_eq!(state.compact.as_ref().unwrap().ledger, previous_ledger);
    assert!(Arc::ptr_eq(
        &previous.segments["group-state"].bytes_arc(),
        &next.segments["group-state"].bytes_arc()
    ));
    assert_ne!(
        previous.segments["batch-metadata"].bytes(),
        next.segments["batch-metadata"].bytes()
    );
    let (decoded, _owner) = metadata::decode(
        state.retention_runtime().unwrap(),
        &next.segments["batch-metadata"],
        "compact",
    )
    .unwrap();
    assert_eq!(decoded, latest);
    let mut fresh = state.clone();
    assert!(fresh.compact.is_none() && fresh.retained.is_none() && fresh.incremental.is_none());
    assert!(
        fresh
            .checkpoint(Epoch::INITIAL)
            .unwrap()
            .inline_metadata
            .is_empty()
    );
    StreamOperator::reset(state).unwrap();
    assert!(state.compact.is_none() && state.retained.is_none() && state.incremental.is_none());
}

struct Reject;

#[async_trait]
impl StreamCollector for Reject {
    async fn emit(&mut self, _port: &str, _batch: Batch) -> Result<()> {
        Err(CalcFlowError::Operator {
            node_id: "collector".into(),
            message: "reject output".into(),
        })
    }
}

#[tokio::test]
async fn test_compact_rejected_emit_keeps_state_quota_capture_and_input_release() {
    let job = StreamJobContext::new(
        42,
        "compact",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "compact", None);
    let mut state = operator(true, true);
    drop(
        process(
            &mut state,
            input(true, &[Some("a")], &[Some(2)], 0),
            &context,
        )
        .await,
    );
    let before = state.checkpoint(Epoch::INITIAL).unwrap();
    let ledger = state.compact.as_ref().unwrap().ledger;
    let incoming = input(true, &[Some("b")], &[Some(9)], 1);
    let weak = incoming.table_payload().unwrap().batches()[0]
        .columns()
        .iter()
        .map(Arc::downgrade)
        .collect::<Vec<_>>();
    assert!(
        state
            .process_data("events", incoming, &context, &mut Reject)
            .await
            .is_err()
    );
    assert!(weak.iter().all(|weak| weak.upgrade().is_none()));
    assert_eq!(state.compact.as_ref().unwrap().ledger, ledger);
    same_snapshot(&before, &state.checkpoint(Epoch::INITIAL).unwrap());
    state
        .set_state_budget(StateBudget::new(ledger.rows, ledger.bytes).unwrap())
        .unwrap();
    assert!(
        state
            .process_data(
                "events",
                input(true, &[Some("a")], &[Some(1)], 2),
                &context,
                &mut Reject
            )
            .await
            .is_err()
    );
    same_snapshot(&before, &state.checkpoint(Epoch::INITIAL).unwrap());
}

#[tokio::test]
async fn test_compact_cancelled_capture_preserves_prior_state_and_metadata_capture() {
    let job = StreamJobContext::new(
        43,
        "compact",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "compact", None);
    let mut state = operator(true, false);
    drop(
        process(
            &mut state,
            input(true, &[Some("a")], &[Some(2)], 0),
            &context,
        )
        .await,
    );
    let before = state.checkpoint(Epoch::INITIAL).unwrap();
    drop(process(&mut state, input(true, &[], &[], 1), &context).await);
    let calls = std::cell::Cell::new(0);
    let result = state.prepare_compact_capture(&|| {
        calls.set(calls.get() + 1);
        if calls.get() == 4 {
            Err(CalcFlowError::Cancelled {
                run_id: "capture".into(),
            })
        } else {
            Ok(())
        }
    });
    assert!(result.is_err());
    same_snapshot(
        &before,
        &state
            .compact
            .as_ref()
            .unwrap()
            .capture
            .as_ref()
            .unwrap()
            .snapshot,
    );
    let after = state.checkpoint(Epoch::INITIAL).unwrap();
    assert!(Arc::ptr_eq(
        &before.segments["group-state"].bytes_arc(),
        &after.segments["group-state"].bytes_arc()
    ));
    assert_eq!(
        after.inline_metadata["rows"],
        before.inline_metadata["rows"]
    );
    assert_ne!(
        after.segments["batch-metadata"].bytes(),
        before.segments["batch-metadata"].bytes()
    );
}
