use std::{collections::BTreeMap, io::Cursor, sync::Weak};

use datafusion::arrow::{
    array::{Array, ArrayRef, Int64Array, StringArray, new_empty_array},
    buffer::Buffer,
    ipc::reader::FileReader,
};
use serde_json::json;

use super::*;
use crate::{
    BatchMetadata, CancellationToken, DataFusionConfig, EdgeCollector, Epoch, JsonMap,
    StreamJobContext,
    operator::{OperatorMetadata, OperatorStateSnapshot, StateBudget, StreamOperator},
};

const GROUPED: &str = "SELECT MAX(value) AS hi, key, SUM(value) AS total, COUNT(value) AS valid, MIN(value) AS lo, COUNT(*) AS rows, SUM(value) AS again FROM events GROUP BY key";

#[derive(Clone)]
struct Input {
    keys: Vec<Option<i64>>,
    values: Vec<Option<i64>>,
}

struct Owners {
    arrays: Vec<Weak<dyn Array>>,
    buffers: Vec<Buffer>,
    unused: Weak<dyn Array>,
    unused_buffer: Buffer,
}

fn operator(query: &str) -> super::super::SqlOperator {
    super::super::SqlOperator::new("totals", query, vec!["events".into()], vec![]).unwrap()
}

fn known_operator(query: &str, data_type: &DataType) -> super::super::SqlOperator {
    let empty = Input {
        keys: vec![],
        values: vec![],
    };
    let schema = fixture(&empty, data_type, 0)
        .0
        .table_payload()
        .unwrap()
        .schema()
        .clone();
    operator(query)
        .with_ports(
            vec![
                crate::operator::Port::with_schema_ref(
                    "events",
                    crate::BatchKind::Table,
                    true,
                    Some(schema),
                )
                .unwrap(),
            ],
            crate::operator::Port::new("output", crate::BatchKind::Table, true, None).unwrap(),
        )
        .unwrap()
}

fn job() -> StreamJobContext {
    StreamJobContext::new(1, "compact", JsonMap::new(), None, CancellationToken::new())
}

fn fixed_input(value: i64) -> Input {
    Input {
        keys: [Some(0), Some(1), None].repeat(8),
        values: [Some(value), None, Some(-value)].repeat(8),
    }
}

fn fixture(input: &Input, data_type: &DataType, sequence: usize) -> (Batch, Owners) {
    assert_eq!(input.keys.len(), input.values.len());
    let key: ArrayRef = Arc::new(Int64Array::from(input.keys.clone()));
    let value = if input.values.is_empty() {
        new_empty_array(data_type)
    } else {
        ScalarValue::iter_to_array(input.values.iter().map(|value| match data_type {
            DataType::Int64 => ScalarValue::Int64(*value),
            DataType::Decimal128(precision, scale) => {
                ScalarValue::Decimal128(value.map(i128::from), *precision, *scale)
            }
            _ => panic!("unsupported test fixture type"),
        }))
        .unwrap()
    };
    assert!(key.get_buffer_memory_size() + value.get_buffer_memory_size() <= 64 << 10);
    let unused: ArrayRef = Arc::new(StringArray::from_iter_values(
        (0..input.keys.len()).map(|_| "unused fixed-group input"),
    ));
    let owners = Owners {
        arrays: vec![Arc::downgrade(&key), Arc::downgrade(&value)],
        buffers: vec![
            key.to_data().buffers()[0].clone(),
            value.to_data().buffers()[0].clone(),
        ],
        unused: Arc::downgrade(&unused),
        unused_buffer: unused.to_data().buffers()[1].clone(),
    };
    let fields = vec![
        Field::new("key", DataType::Int64, true),
        Field::new("value", data_type.clone(), true).with_metadata(
            std::collections::HashMap::from([("unit".into(), "native".into())]),
        ),
        Field::new("unused", DataType::Utf8, false),
    ];
    let schema = Arc::new(Schema::new_with_metadata(
        fields,
        std::collections::HashMap::from([("origin".into(), "compact-fixture".into())]),
    ));
    let record = RecordBatch::try_new(schema, vec![key, value, unused]).unwrap();
    let metadata = BatchMetadata::new(
        "compact-source",
        u64::try_from(sequence).unwrap(),
        JsonMap::from([("prefix".into(), json!(sequence))]),
    )
    .unwrap();
    (Batch::table(vec![record], metadata).unwrap(), owners)
}

fn table_rows(records: &[RecordBatch]) -> Vec<Vec<ScalarValue>> {
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

fn sorted_rows(batch: &Batch) -> Vec<Vec<ScalarValue>> {
    let mut rows = table_rows(batch.table_payload().unwrap().batches());
    rows.sort_by(|left, right| left.partial_cmp(right).unwrap());
    rows
}

async fn push_oracle(
    operator: &mut super::super::SqlOperator,
    runtime: &DataFusionRuntime,
    query: &str,
    data_type: &DataType,
    prefix: &[Input],
    context: &StreamOperatorContext<'_>,
) -> Owners {
    let last = prefix.len() - 1;
    let (incoming, owners) = fixture(&prefix[last], data_type, last);
    let caller = incoming.clone();
    let metadata = incoming.metadata().clone();
    let mut records = Vec::new();
    for (sequence, input) in prefix.iter().enumerate() {
        if !input.keys.is_empty() || records.is_empty() {
            records.extend_from_slice(
                fixture(input, data_type, sequence)
                    .0
                    .table_payload()
                    .unwrap()
                    .batches(),
            );
        }
    }
    let expected = runtime
        .sql(
            query,
            &BTreeMap::from([(
                "events".into(),
                Batch::table(records, metadata.clone()).unwrap(),
            )]),
            Some("independent-prefix"),
        )
        .await
        .unwrap();
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("events", incoming, context, &mut collector)
        .await
        .unwrap();
    let output = collector.drain("output");
    assert_eq!(output.len(), 1, "{query}: prefix {last}");
    let actual = output[0].as_data().unwrap();
    assert_eq!(
        actual.table_payload().unwrap().schema(),
        expected.table_payload().unwrap().schema(),
        "{query}: prefix {last}"
    );
    assert_eq!(
        sorted_rows(actual),
        sorted_rows(&expected),
        "{query}: prefix {last}"
    );
    assert_eq!(actual.metadata(), &metadata, "{query}: prefix {last}");
    let unchanged = fixture(&prefix[last], data_type, last).0;
    assert_eq!(caller.metadata(), unchanged.metadata());
    assert_eq!(
        caller.table_payload().unwrap().batches(),
        unchanged.table_payload().unwrap().batches()
    );
    drop(caller);
    assert!(owners.unused.upgrade().is_none());
    if !prefix[last].keys.is_empty() {
        assert_eq!(owners.unused_buffer.strong_count(), 1);
    }
    owners
}

async fn round_trip_twice(
    original: &mut super::super::SqlOperator,
    runtime: &DataFusionRuntime,
    query: &str,
    data_type: &DataType,
    prefix: &[Input],
    context: &StreamOperatorContext<'_>,
) {
    original.prepare_checkpoint_async(context).await.unwrap();
    let mut snapshot = original.checkpoint(Epoch::INITIAL).unwrap();
    let mut continued = prefix.to_vec();
    for value in [7, -2] {
        let mut restored = operator(query)
            .with_ports(
                original.input_ports().to_vec(),
                original.output_ports()[0].clone(),
            )
            .unwrap();
        StreamOperator::restore(&mut restored, &snapshot).unwrap();
        continued.push(Input {
            keys: vec![Some(0)],
            values: vec![Some(value)],
        });
        push_oracle(
            &mut restored,
            runtime,
            query,
            data_type,
            &continued,
            context,
        )
        .await;
        restored.prepare_checkpoint_async(context).await.unwrap();
        snapshot = restored.checkpoint(Epoch::INITIAL).unwrap();
    }
}

fn same_snapshot(before: &OperatorStateSnapshot, after: &OperatorStateSnapshot) {
    assert_eq!(before.inline_metadata, after.inline_metadata);
    assert_eq!(
        before.segments.keys().collect::<Vec<_>>(),
        after.segments.keys().collect::<Vec<_>>()
    );
    for (name, segment) in &before.segments {
        assert_eq!(segment.bytes(), after.segments[name].bytes());
    }
}

fn native_state_rows(operator: &super::super::SqlOperator) -> Vec<Vec<ScalarValue>> {
    operator
        .incremental
        .as_ref()
        .expect("real callback must select the existing native plan")
        .groups
        .iter()
        .map(|group| {
            group
                .values
                .iter()
                .cloned()
                .chain(group.states.iter().flatten().cloned())
                .collect()
        })
        .collect()
}

fn capture_rows(snapshot: &OperatorStateSnapshot) -> (usize, usize) {
    let state = ["group-state", "input-retained"]
        .into_iter()
        .find_map(|name| snapshot.segments.get(name))
        .unwrap();
    let reader = FileReader::try_new(Cursor::new(state.bytes()), None).unwrap();
    let records = reader.collect::<std::result::Result<Vec<_>, _>>().unwrap();
    (
        records.iter().map(RecordBatch::num_rows).sum(),
        snapshot
            .segments
            .values()
            .map(|segment| segment.bytes().len())
            .sum(),
    )
}

fn assert_native_capture(snapshot: &OperatorStateSnapshot, expected: &[Vec<ScalarValue>]) {
    let (rows, bytes) = capture_rows(snapshot);
    eprintln!(
        "compact capture census: layout={:?} rows={rows} bytes={bytes}",
        snapshot.inline_metadata.get("state_layout")
    );
    assert_eq!(
        snapshot.inline_metadata.get("state_layout"),
        Some(&json!(3))
    );
    assert_eq!(
        snapshot.inline_metadata.get("state_accounting"),
        Some(&json!(3))
    );
    assert_eq!(
        snapshot
            .segments
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["batch-metadata", "control", "group-state", "logical-schema"]
    );
    let reader =
        FileReader::try_new(Cursor::new(snapshot.segments["group-state"].bytes()), None).unwrap();
    let records = reader.collect::<std::result::Result<Vec<_>, _>>().unwrap();
    assert_eq!(rows, expected.len());
    assert_eq!(
        table_rows(&records),
        expected,
        "wire must contain keys plus complete native states, without cached results"
    );
}

#[tokio::test]
async fn test_fixed_groups_release_all_admitted_input_arrays_and_buffers() {
    let mut operator = known_operator(GROUPED, &DataType::Int64);
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "totals", None);
    let mut prefix = Vec::new();
    let mut owners = Vec::new();
    for value in 1..=16 {
        prefix.push(fixed_input(value));
        owners.push(
            push_oracle(
                &mut operator,
                &runtime,
                GROUPED,
                &DataType::Int64,
                &prefix,
                &context,
            )
            .await,
        );
    }
    assert_eq!(native_state_rows(&operator).len(), 3);
    round_trip_twice(
        &mut operator,
        &runtime,
        GROUPED,
        &DataType::Int64,
        &prefix,
        &context,
    )
    .await;
    let raw_records = operator
        .retained
        .as_ref()
        .map_or(0, |state| state.records.len());
    let pinned_arrays = owners
        .iter()
        .flat_map(|owner| &owner.arrays)
        .filter(|array| array.upgrade().is_some())
        .count();
    let pinned_buffers = owners
        .iter()
        .flat_map(|owner| &owner.buffers)
        .filter(|buffer| buffer.strong_count() != 1)
        .count();
    eprintln!(
        "compact ownership census: admitted=384 groups=3 raw_records={raw_records} pinned_arrays={pinned_arrays} pinned_buffers={pinned_buffers}"
    );
    assert_eq!(
        (raw_records, pinned_arrays, pinned_buffers),
        (0, 0, 0),
        "successful eligible commits must release raw history and actual input backing owners"
    );
}

#[tokio::test]
async fn test_checkpoint_contains_native_group_states_instead_of_admitted_history() {
    let mut operator = known_operator(GROUPED, &DataType::Int64);
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "totals", None);
    let mut prefix = vec![fixed_input(1)];
    push_oracle(
        &mut operator,
        &runtime,
        GROUPED,
        &DataType::Int64,
        &prefix,
        &context,
    )
    .await;
    operator.prepare_checkpoint_async(&context).await.unwrap();
    let small = operator.checkpoint(Epoch::INITIAL).unwrap();
    let small_native = native_state_rows(&operator);
    for value in 2..=16 {
        prefix.push(fixed_input(value));
        push_oracle(
            &mut operator,
            &runtime,
            GROUPED,
            &DataType::Int64,
            &prefix,
            &context,
        )
        .await;
    }
    round_trip_twice(
        &mut operator,
        &runtime,
        GROUPED,
        &DataType::Int64,
        &prefix,
        &context,
    )
    .await;
    let large = operator.checkpoint(Epoch::INITIAL).unwrap();
    let large_native = native_state_rows(&operator);
    eprintln!(
        "fixed-G captures: small={:?}, large={:?}",
        capture_rows(&small),
        capture_rows(&large)
    );
    assert_native_capture(&small, &small_native);
    assert_native_capture(&large, &large_native);
    assert_eq!(large.inline_metadata["rows"], json!(384));
    assert!(
        capture_rows(&large).1 <= capture_rows(&small).1 + 1024,
        "fixed-width G=3 capture must not scale with 24 to384 admitted rows"
    );
}

#[tokio::test]
async fn test_decimal_avg_scalar_and_grouped_discard_history_after_empty_null_prefixes() {
    let data_type = DataType::Decimal128(28, 2);
    let prefix = vec![
        Input {
            keys: vec![],
            values: vec![],
        },
        Input {
            keys: vec![Some(0), Some(1), None],
            values: vec![None; 3],
        },
        Input {
            keys: vec![Some(0)],
            values: vec![Some(1200)],
        },
        Input {
            keys: vec![Some(1), None],
            values: vec![None, Some(-400)],
        },
        Input {
            keys: vec![],
            values: vec![],
        },
    ];
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "totals", None);
    let mut completed = Vec::new();
    for query in [
        "SELECT COUNT(*) AS rows, COUNT(value) AS valid, SUM(value) AS total, AVG(value) AS mean FROM events",
        "SELECT key, COUNT(*) AS rows, COUNT(value) AS valid, SUM(value) AS total, AVG(value) AS mean FROM events GROUP BY key",
    ] {
        let mut operator = known_operator(query, &data_type);
        for last in 1..=prefix.len() {
            push_oracle(
                &mut operator,
                &runtime,
                query,
                &data_type,
                &prefix[..last],
                &context,
            )
            .await;
        }
        round_trip_twice(
            &mut operator,
            &runtime,
            query,
            &data_type,
            &prefix,
            &context,
        )
        .await;
        let state = operator.checkpoint(Epoch::INITIAL).unwrap();
        let native = native_state_rows(&operator);
        let raw = operator
            .retained
            .as_ref()
            .map_or(0, |state| state.records.len());
        completed.push((state, native, raw));
    }
    let raw = completed.iter().map(|(_, _, raw)| *raw).collect::<Vec<_>>();
    eprintln!("Decimal128AVG raw histories after scalar/grouped complete-prefix controls: {raw:?}");
    assert_eq!(raw, [0, 0]);
    for (snapshot, native, _) in completed {
        assert_native_capture(&snapshot, &native);
    }
}

#[tokio::test]
async fn test_current_quota_empty_metadata_and_live_raw_four_controls() {
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "totals", None);
    let first = fixed_input(1);
    let empty = Input {
        keys: vec![],
        values: vec![],
    };
    let (original, _) = fixture(&first, &DataType::Int64, 0);
    let physical = Batch::table(
        vec![
            original.table_payload().unwrap().batches()[0]
                .project(&[0, 1])
                .unwrap(),
        ],
        original.metadata().clone(),
    )
    .unwrap();
    let bytes = u64::try_from(physical.estimated_bytes().unwrap()).unwrap();
    for row_limit in [24, 1000] {
        let mut operator = operator(GROUPED);
        let byte_limit = if row_limit == 24 { 1 << 20 } else { bytes };
        operator
            .set_state_budget(StateBudget::new(row_limit, byte_limit).unwrap())
            .unwrap();
        push_oracle(
            &mut operator,
            &runtime,
            GROUPED,
            &DataType::Int64,
            std::slice::from_ref(&first),
            &context,
        )
        .await;
        push_oracle(
            &mut operator,
            &runtime,
            GROUPED,
            &DataType::Int64,
            &[first.clone(), empty.clone()],
            &context,
        )
        .await;
        let before = operator.checkpoint(Epoch::INITIAL).unwrap();
        assert_eq!(before.inline_metadata["rows"], json!(24));
        assert_eq!(before.inline_metadata["bytes"], json!(bytes));
        let setter = operator
            .set_state_budget(StateBudget::new(23, 1 << 20).unwrap())
            .unwrap_err();
        assert!(
            matches!(setter, CalcFlowError::InvalidArgument { field, .. } if field=="sql.state_budget")
        );
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        let error = operator
            .process_data(
                "events",
                fixture(&first, &DataType::Int64, 2).0,
                &context,
                &mut collector,
            )
            .await
            .unwrap_err();
        assert!(
            matches!(error, CalcFlowError::Operator { node_id, message } if node_id=="totals" && message=="SQL aggregate retained input exceeds the configured state budget")
        );
        assert!(collector.drain("output").is_empty());
        same_snapshot(&before, &operator.checkpoint(Epoch::INITIAL).unwrap());
        operator
            .set_state_budget(StateBudget::new(48, bytes.checked_mul(2).unwrap()).unwrap())
            .unwrap();
        push_oracle(
            &mut operator,
            &runtime,
            GROUPED,
            &DataType::Int64,
            &[first.clone(), empty.clone(), first.clone()],
            &context,
        )
        .await;
        let retried = operator.checkpoint(Epoch::INITIAL).unwrap();
        assert_eq!(retried.inline_metadata["rows"], json!(48));
        assert_eq!(retried.inline_metadata["bytes"], json!(bytes * 2));
    }
    assert_scalar_and_fallback_controls(&runtime, &context, first, empty).await;
    eprintln!(
        "quota row/byte/setter, global empty/null metadata, live ineligible raw4 restore controls completed"
    );
}

async fn assert_scalar_and_fallback_controls(
    runtime: &DataFusionRuntime,
    context: &StreamOperatorContext<'_>,
    first: Input,
    empty: Input,
) {
    let scalar = "SELECT COUNT(*) AS rows, COUNT(value) AS valid, SUM(value) AS total FROM events";
    let prefix = [
        empty.clone(),
        Input {
            keys: vec![None],
            values: vec![None],
        },
        first.clone(),
        empty,
    ];
    let mut scalar_operator = operator(scalar);
    for last in 1..=prefix.len() {
        push_oracle(
            &mut scalar_operator,
            runtime,
            scalar,
            &DataType::Int64,
            &prefix[..last],
            context,
        )
        .await;
    }
    round_trip_twice(
        &mut scalar_operator,
        runtime,
        scalar,
        &DataType::Int64,
        &prefix,
        context,
    )
    .await;
    let fallback = "SELECT key, COUNT(*) AS rows, SUM(abs(value)) AS total FROM events WHERE value IS NOT NULL GROUP BY key";
    let mut retained = operator(fallback);
    push_oracle(
        &mut retained,
        runtime,
        fallback,
        &DataType::Int64,
        std::slice::from_ref(&first),
        context,
    )
    .await;
    assert!(retained.incremental.is_none());
    assert!(retained.compact.is_none());
    let snapshot = retained.checkpoint(Epoch::INITIAL).unwrap();
    assert_eq!(snapshot.inline_metadata["state_layout"], json!(4));
    assert_eq!(snapshot.inline_metadata["state_accounting"], json!(4));
    round_trip_twice(
        &mut retained,
        runtime,
        fallback,
        &DataType::Int64,
        &[first],
        context,
    )
    .await;
}
