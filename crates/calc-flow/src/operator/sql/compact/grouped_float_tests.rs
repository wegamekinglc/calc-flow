use super::super::*;
use crate::{CancellationToken, EdgeCollector, StreamJobContext};
use async_trait::async_trait;
use datafusion::{
    arrow::{
        array::{Array, ArrayRef, Float32Array, Float64Array, Int64Array, StringArray},
        datatypes::{DataType, Field, Schema, SchemaRef},
        record_batch::RecordBatch,
    },
    common::ScalarValue,
};
use serde_json::json;
use std::{
    collections::BTreeMap,
    sync::{Arc, Weak},
};

type Bits = (u32, u64);
type Row = (Option<i64>, Option<Bits>);
type Part = Vec<Row>;
const ZERO: Bits = (0, 0);
const NEG_ZERO: Bits = (0x8000_0000, 0x8000_0000_0000_0000);
const ONE: Bits = (0x3f80_0000, 0x3ff0_0000_0000_0000);
const TWO: Bits = (0x4000_0000, 0x4000_0000_0000_0000);
const POS_INF: Bits = (0x7f80_0000, 0x7ff0_0000_0000_0000);
const NEG_INF: Bits = (0xff80_0000, 0xfff0_0000_0000_0000);
const NAN: Bits = (0x7fc0_0001, 0x7ff8_0000_0000_0001);
const SNAN: Bits = (0xff80_0005, 0xfff0_0000_0000_0005);
const QUERY: &str = "SELECT key, MIN(value) AS lo, MAX(value) AS hi, COUNT(value) AS valid, COUNT(*) AS rows FROM events GROUP BY key";

fn schema(dtype: &DataType) -> SchemaRef {
    let fields = (0..8)
        .map(|index| match index {
            0 => Field::new("key", DataType::Int64, true),
            6 => Field::new("value", dtype.clone(), true)
                .with_metadata([("unit".into(), "bit-winner".into())].into()),
            _ => Field::new(format!("unused_{index}"), DataType::Utf8, false),
        })
        .collect::<Vec<_>>();
    Arc::new(Schema::new_with_metadata(
        fields,
        [("origin".into(), "grouped-float".into())].into(),
    ))
}

fn input(dtype: &DataType, parts: &[Part], sequence: u64) -> Batch {
    let records = parts
        .iter()
        .map(|part| {
            let values: ArrayRef = match dtype {
                DataType::Float32 => Arc::new(Float32Array::from(
                    part.iter()
                        .map(|row| row.1.map(|bits| f32::from_bits(bits.0)))
                        .collect::<Vec<_>>(),
                )),
                DataType::Float64 => Arc::new(Float64Array::from(
                    part.iter()
                        .map(|row| row.1.map(|bits| f64::from_bits(bits.1)))
                        .collect::<Vec<_>>(),
                )),
                _ => unreachable!(),
            };
            let arrays = (0..8)
                .map(|index| match index {
                    0 => Arc::new(Int64Array::from(
                        part.iter().map(|row| row.0).collect::<Vec<_>>(),
                    )) as ArrayRef,
                    6 => values.clone(),
                    _ => {
                        Arc::new(StringArray::from(vec!["unused payload"; part.len()])) as ArrayRef
                    }
                })
                .collect();
            RecordBatch::try_new(schema(dtype), arrays).unwrap()
        })
        .collect();
    Batch::table(
        records,
        BatchMetadata::new(
            "grouped-float",
            sequence,
            JsonMap::from([
                ("prefix".into(), json!(sequence)),
                ("case".into(), json!(format!("{dtype:?}"))),
            ]),
        )
        .unwrap(),
    )
    .unwrap()
}

fn operator(dtype: &DataType) -> SqlOperator {
    SqlOperator::new("grouped_float", QUERY, vec!["events".into()], vec![])
        .unwrap()
        .with_ports(
            vec![
                Port::with_schema_ref("events", BatchKind::Table, true, Some(schema(dtype)))
                    .unwrap(),
            ],
            Port::new("output", BatchKind::Table, true, None).unwrap(),
        )
        .unwrap()
}

fn job() -> StreamJobContext {
    StreamJobContext::new(
        920,
        "grouped-float",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
}

#[derive(Debug, PartialEq)]
enum Cell {
    Float32(Option<u32>),
    Float64(Option<u64>),
    Other(ScalarValue),
}

fn cell(array: &ArrayRef, row: usize) -> Cell {
    match array.data_type() {
        DataType::Float32 => Cell::Float32((!array.is_null(row)).then(|| {
            array
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap()
                .value(row)
                .to_bits()
        })),
        DataType::Float64 => Cell::Float64((!array.is_null(row)).then(|| {
            array
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .value(row)
                .to_bits()
        })),
        _ => Cell::Other(ScalarValue::try_from_array(array, row).unwrap()),
    }
}

fn rows(batch: &Batch) -> BTreeMap<Option<i64>, Vec<Cell>> {
    let mut rows = BTreeMap::new();
    for record in batch.table_payload().unwrap().batches() {
        let keys = record
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        for row in 0..record.num_rows() {
            let key = keys.is_valid(row).then(|| keys.value(row));
            let values = record
                .columns()
                .iter()
                .map(|array| cell(array, row))
                .collect();
            assert!(rows.insert(key, values).is_none());
        }
    }
    rows
}

async fn process(
    state: &mut SqlOperator,
    batch: Batch,
    context: &StreamOperatorContext<'_>,
) -> Batch {
    let mut collector = EdgeCollector::new(state.output_ports().to_vec());
    state
        .process_data("events", batch, context, &mut collector)
        .await
        .unwrap();
    let output = collector.drain("output");
    assert_eq!(output.len(), 1);
    output[0].as_data().unwrap().clone()
}

async fn assert_oracle(actual: &Batch, dtype: &DataType, parts: &[Part], sequence: u64) {
    let batch = input(dtype, parts, sequence);
    let expected = DataFusionRuntime::new(DataFusionConfig::default())
        .unwrap()
        .sql(
            QUERY,
            &BTreeMap::from([("events".into(), batch.clone())]),
            Some("oracle"),
        )
        .await
        .unwrap();
    assert_eq!(
        actual.table_payload().unwrap().schema(),
        expected.table_payload().unwrap().schema()
    );
    assert_eq!(rows(actual), rows(&expected));
    assert_eq!(actual.metadata(), expected.metadata());
    assert_eq!(actual.metadata(), batch.metadata());
}

fn weak_arrays(batch: &Batch) -> Vec<Weak<dyn Array>> {
    batch
        .table_payload()
        .unwrap()
        .batches()
        .iter()
        .flat_map(|record| record.columns().iter().map(Arc::downgrade))
        .collect()
}

fn native_capture(state: &mut SqlOperator, dtype: &DataType) -> OperatorStateSnapshot {
    assert!(
        state.incremental.is_some() && state.compact.is_some() && state.retained.is_none(),
        "grouped Float extrema must own native state, not retained input"
    );
    let snapshot = state.checkpoint(Epoch::INITIAL).unwrap();
    assert_eq!(snapshot.inline_metadata["state_layout"], json!(3));
    assert_eq!(snapshot.inline_metadata["state_accounting"], json!(3));
    assert_eq!(
        snapshot
            .segments
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["batch-metadata", "control", "group-state", "logical-schema"]
    );
    let projection = state.compact.as_ref().unwrap().projection().unwrap();
    assert_eq!(projection.columns.logical_schema(), &schema(dtype));
    assert_eq!(projection.columns.ordinals(), &[0, 6]);
    let wire = decode_sql_state(snapshot.segments["group-state"].bytes()).unwrap();
    let fields = wire.table_payload().unwrap().schema().fields();
    assert_eq!(fields.len(), 5);
    assert_eq!(fields[1].data_type(), dtype);
    assert_eq!(fields[2].data_type(), dtype);
    let control: Value = serde_json::from_slice(snapshot.segments["control"].bytes()).unwrap();
    let policy = control.get("state_policy").unwrap().as_object().unwrap();
    assert_eq!(policy.len(), 1);
    let payload = policy
        .get("sequential_grouped_float_v1")
        .unwrap()
        .as_object()
        .unwrap();
    assert_eq!(payload.len(), 4);
    assert_eq!(
        payload.get("config").unwrap(),
        &serde_json::to_value(DataFusionConfig::default()).unwrap()
    );
    assert_eq!(payload.get("factory").unwrap(), &json!("primitive_v1"));
    assert_eq!(
        payload.get("model").unwrap(),
        &json!("df54_single_linear_memtable_v1")
    );
    let max_record_rows = payload.get("max_record_rows").unwrap().as_u64().unwrap();
    assert!(max_record_rows > 0);
    assert!(max_record_rows <= control["ledger"]["rows"].as_u64().unwrap());
    snapshot
}

fn same_snapshot(before: &OperatorStateSnapshot, after: &OperatorStateSnapshot) {
    assert_eq!(before.inline_metadata, after.inline_metadata);
    assert_eq!(
        before.segments.keys().collect::<Vec<_>>(),
        after.segments.keys().collect::<Vec<_>>()
    );
    for (id, segment) in &before.segments {
        assert_eq!(segment.bytes(), after.segments[id].bytes());
    }
}

fn arrivals(reverse: bool) -> Vec<Vec<Part>> {
    let (first, next) = if reverse {
        (NEG_ZERO, ZERO)
    } else {
        (ZERO, NEG_ZERO)
    };
    let mut boundary = vec![(Some(2), Some(TWO)); 10_003];
    boundary[8191] = (Some(2), Some(NAN));
    boundary[8192] = (Some(2), Some(ONE));
    vec![
        vec![vec![
            (Some(1), Some(first)),
            (Some(2), Some(ONE)),
            (None, None),
            (Some(7), Some(NAN)),
            (Some(7), Some(POS_INF)),
            (Some(8), Some(NAN)),
            (Some(8), Some(NEG_INF)),
            (Some(9), Some(POS_INF)),
            (Some(10), Some(NEG_INF)),
        ]],
        vec![
            vec![
                (Some(1), Some(next)),
                (Some(2), Some(NAN)),
                (Some(2), Some(TWO)),
                (Some(6), None),
            ],
            vec![(None, Some(SNAN))],
        ],
        vec![boundary],
    ]
}

async fn roundtrip(
    state: SqlOperator,
    dtype: &DataType,
    prefix: Vec<Part>,
    context: &StreamOperatorContext<'_>,
) {
    let mut state = state;
    let before = native_capture(&mut state, dtype);
    let control: Value = serde_json::from_slice(before.segments["control"].bytes()).unwrap();
    assert_eq!(
        control["state_policy"]["sequential_grouped_float_v1"]["max_record_rows"],
        json!(10_003)
    );
    let pool = state
        .stream_state
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let mut restored = operator(dtype);
    StreamOperator::restore(&mut restored, &before).unwrap();
    same_snapshot(&before, &restored.checkpoint(Epoch::INITIAL).unwrap());
    let empty = process(&mut restored, input(dtype, &[vec![]], 3), context).await;
    assert_oracle(&empty, dtype, &prefix, 3).await;
    let next = vec![vec![
        (Some(7), Some(TWO)),
        (Some(8), Some(ONE)),
        (Some(6), Some(SNAN)),
        (None, Some(NEG_ZERO)),
    ]];
    let batch = input(dtype, &next, 4);
    let weak = weak_arrays(&batch);
    let actual = process(&mut restored, batch, context).await;
    let all = prefix.into_iter().chain(next).collect::<Vec<_>>();
    assert_oracle(&actual, dtype, &all, 4).await;
    let after = native_capture(&mut restored, dtype);
    let after_control: Value = serde_json::from_slice(after.segments["control"].bytes()).unwrap();
    assert_eq!(
        after_control["state_policy"]["sequential_grouped_float_v1"]["max_record_rows"],
        json!(10_003)
    );
    assert_eq!(
        after_control.get("state_policy").unwrap(),
        control.get("state_policy").unwrap()
    );
    assert!(weak.iter().all(|array| array.upgrade().is_none()));
    let target = restored
        .stream_state
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    drop((state, restored, before, after, empty, actual));
    assert_eq!(pool.reserved(), 0);
    assert_eq!(target.reserved(), 0);
}

#[tokio::test]
async fn test_grouped_float_fixed_key_native_bits_capture_restore_continue() {
    for dtype in [DataType::Float32, DataType::Float64] {
        for reverse in [false, true] {
            let job = job();
            let context = StreamOperatorContext::new(&job, "grouped_float", None);
            let mut state = operator(&dtype);
            let mut prefix = Vec::new();
            let mut weak = Vec::new();
            for (sequence, parts) in arrivals(reverse).into_iter().enumerate() {
                let batch = input(&dtype, &parts, sequence as u64);
                weak.extend(weak_arrays(&batch));
                prefix.extend(parts);
                let actual = process(&mut state, batch, &context).await;
                assert_oracle(&actual, &dtype, &prefix, sequence as u64).await;
            }
            drop(native_capture(&mut state, &dtype));
            assert!(weak.iter().all(|array| array.upgrade().is_none()));
            roundtrip(state, &dtype, prefix, &context).await;
        }
    }
}

struct Reject;
#[async_trait]
impl StreamCollector for Reject {
    async fn emit(&mut self, _: &str, _: Batch) -> Result<()> {
        Err(CalcFlowError::Operator {
            node_id: "reject-grouped-float".into(),
            message: "injected failure".into(),
        })
    }
}

#[tokio::test]
async fn test_grouped_float_emit_failure_preserves_capture_then_single_retry() {
    for dtype in [DataType::Float32, DataType::Float64] {
        let job = job();
        let context = StreamOperatorContext::new(&job, "grouped_float", None);
        let mut state = operator(&dtype);
        let first = vec![vec![(Some(1), Some(ONE)), (None, None)]];
        drop(process(&mut state, input(&dtype, &first, 0), &context).await);
        let before = state.checkpoint(Epoch::INITIAL).unwrap();
        let pool = state
            .stream_state
            .runtime()
            .unwrap()
            .incremental_memory_pool();
        let reserved = pool.reserved();
        let next = vec![vec![
            (Some(1), Some(NAN)),
            (Some(1), Some(TWO)),
            (None, Some(SNAN)),
            (Some(4), Some(POS_INF)),
        ]];
        let rejected = input(&dtype, &next, 1);
        let weak = weak_arrays(&rejected);
        let failure = state
            .process_data("events", rejected, &context, &mut Reject)
            .await;
        assert!(
            matches!(failure, Err(CalcFlowError::Operator { node_id, .. }) if node_id == "reject-grouped-float")
        );
        assert!(weak.iter().all(|array| array.upgrade().is_none()));
        assert_eq!(pool.reserved(), reserved);
        same_snapshot(&before, &state.checkpoint(Epoch::INITIAL).unwrap());
        let actual = process(&mut state, input(&dtype, &next, 1), &context).await;
        let all = first.into_iter().chain(next).collect::<Vec<_>>();
        assert_oracle(&actual, &dtype, &all, 1).await;
        let after = native_capture(&mut state, &dtype);
        assert_eq!(after.inline_metadata["rows"], json!(6));
        drop((state, before, after, actual));
        assert_eq!(pool.reserved(), 0);
    }
}

#[path = "grouped_float_safety_tests.rs"]
mod safety;
