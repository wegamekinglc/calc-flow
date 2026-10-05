use std::{
    collections::BTreeMap,
    sync::{Arc, Weak},
};

use async_trait::async_trait;
use datafusion::{
    arrow::{
        array::{Array, ArrayRef, Float32Array, Float64Array, Int64Array},
        datatypes::{DataType, Field, Schema, SchemaRef},
        record_batch::RecordBatch,
    },
    common::ScalarValue,
};
use serde_json::json;

use super::super::*;
use crate::{CancellationToken, EdgeCollector, StreamJobContext};

#[path = "global_float_sum_avg_tests.rs"]
mod global_float_sum_avg_tests;

#[path = "post_projection_tests.rs"]
mod post_projection_tests;

type Bits = (u32, u64);
type Part = Vec<Option<Bits>>;
const ZERO: Bits = (0, 0);
const NEG_ZERO: Bits = (0x8000_0000, 0x8000_0000_0000_0000);
const ONE: Bits = (0x3f80_0000, 0x3ff0_0000_0000_0000);
const TWO: Bits = (0x4000_0000, 0x4000_0000_0000_0000);
const POS_INF: Bits = (0x7f80_0000, 0x7ff0_0000_0000_0000);
const NEG_INF: Bits = (0xff80_0000, 0xfff0_0000_0000_0000);
const NAN_A: Bits = (0x7fc0_0001, 0x7ff8_0000_0000_0001);
const NAN_B: Bits = (0x7fc0_0002, 0x7ff8_0000_0000_0002);
const SNAN: Bits = (0x7f80_0003, 0x7ff0_0000_0000_0003);
const NEG_NAN: Bits = (0xffc0_0004, 0xfff8_0000_0000_0004);
const NEG_SNAN: Bits = (0xff80_0005, 0xfff0_0000_0000_0005);
const GLOBAL: &str = "SELECT MIN(value) AS lo, MAX(value) AS hi, COUNT(value) AS valid, COUNT(*) AS rows FROM events";
const GROUPED: &str = "SELECT key, MIN(value) AS lo, MAX(value) AS hi FROM events GROUP BY key";

fn schema(dtype: &DataType) -> SchemaRef {
    Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("key", DataType::Int64, false),
            Field::new("value", dtype.clone(), true)
                .with_metadata([("unit".into(), "raw-float-bits".into())].into()),
        ],
        [("origin".into(), "current-float-extrema".into())].into(),
    ))
}

fn metadata(sequence: u64) -> BatchMetadata {
    BatchMetadata::new(
        "current-float-extrema",
        sequence,
        JsonMap::from([
            ("prefix".into(), json!(sequence)),
            ("case".into(), json!("bit-exact")),
        ]),
    )
    .unwrap()
}

fn input(dtype: &DataType, parts: &[Part], sequence: u64) -> Batch {
    let records = parts
        .iter()
        .map(|part| {
            let values: ArrayRef = match dtype {
                DataType::Float32 => Arc::new(Float32Array::from(
                    part.iter()
                        .map(|value| value.map(|bits| f32::from_bits(bits.0)))
                        .collect::<Vec<_>>(),
                )),
                DataType::Float64 => Arc::new(Float64Array::from(
                    part.iter()
                        .map(|value| value.map(|bits| f64::from_bits(bits.1)))
                        .collect::<Vec<_>>(),
                )),
                _ => unreachable!(),
            };
            RecordBatch::try_new(
                schema(dtype),
                vec![Arc::new(Int64Array::from(vec![1; part.len()])), values],
            )
            .unwrap()
        })
        .collect();
    Batch::table(records, metadata(sequence)).unwrap()
}

fn operator(dtype: &DataType, query: &str) -> SqlOperator {
    SqlOperator::new("float_extrema", query, vec!["events".into()], vec![])
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
        917,
        "float-extrema",
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

fn rows(batch: &Batch) -> Vec<Vec<Cell>> {
    batch
        .table_payload()
        .unwrap()
        .batches()
        .iter()
        .flat_map(|record| {
            (0..record.num_rows()).map(|row| {
                record
                    .columns()
                    .iter()
                    .map(|array| cell(array, row))
                    .collect()
            })
        })
        .collect()
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

async fn assert_oracle(
    actual: &Batch,
    query: &str,
    dtype: &DataType,
    parts: &[Part],
    sequence: u64,
) {
    let batch = input(dtype, parts, sequence);
    let expected = DataFusionRuntime::new(DataFusionConfig::default())
        .unwrap()
        .sql(
            query,
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

fn native_capture(state: &mut SqlOperator, dtype: &DataType) -> OperatorStateSnapshot {
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
    assert_eq!(projection.columns.ordinals(), &[1]);
    assert_eq!(projection.columns.physical_schema().fields().len(), 1);
    assert!(state.retained.is_none());
    let wire = decode_sql_state(snapshot.segments["group-state"].bytes()).unwrap();
    let fields = wire.table_payload().unwrap().schema().fields();
    assert_eq!(fields.len(), 4);
    assert_eq!(fields[0].data_type(), dtype);
    assert_eq!(fields[1].data_type(), dtype);
    assert_eq!(fields[2].data_type(), &DataType::Int64);
    assert_eq!(fields[3].data_type(), &DataType::Int64);
    snapshot
}

fn arrivals(reverse_zero: bool) -> Vec<Vec<Part>> {
    let mut boundary = vec![Some(ONE); 8194];
    boundary[8190] = None;
    boundary[8191] = Some(SNAN);
    boundary[8192] = Some(NAN_B);
    let zeros = if reverse_zero {
        [NEG_ZERO, ZERO]
    } else {
        [ZERO, NEG_ZERO]
    };
    vec![
        vec![vec![None, None]],
        vec![vec![]],
        vec![vec![Some(zeros[0])]],
        vec![vec![Some(zeros[1])]],
        vec![vec![Some(ONE), Some(POS_INF)], vec![Some(NEG_INF)]],
        vec![vec![
            Some(NAN_A),
            Some((0x7f80_0006, 0x7ff0_0000_0000_0006)),
        ]],
        vec![boundary],
        vec![
            vec![Some(NEG_SNAN), Some((0xff80_0006, 0xfff0_0000_0000_0006))],
            vec![
                Some(NEG_NAN),
                Some((0xffc0_0007, 0xfff8_0000_0000_0007)),
                None,
            ],
        ],
    ]
}

async fn recover_and_continue(
    state: SqlOperator,
    dtype: &DataType,
    prefix: Vec<Part>,
    context: &StreamOperatorContext<'_>,
) {
    let mut state = state;
    state.prepare_checkpoint_async(context).await.unwrap();
    let snapshot = native_capture(&mut state, dtype);
    let source_pool = state
        .stream_state
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let mut restored = operator(dtype, GLOBAL);
    StreamOperator::restore(&mut restored, &snapshot).unwrap();
    same_snapshot(&snapshot, &restored.checkpoint(Epoch::INITIAL).unwrap());
    let empty_input = input(dtype, &[vec![]], 8);
    let mut weak = weak_arrays(&empty_input);
    let empty = process(&mut restored, empty_input, context).await;
    assert_oracle(&empty, GLOBAL, dtype, &prefix, 8).await;
    let next = vec![vec![
        Some((0xffff_ffff, 0xffff_ffff_ffff_ffff)),
        Some((0x7fff_ffff, 0x7fff_ffff_ffff_ffff)),
        None,
    ]];
    let next_input = input(dtype, &next, 9);
    weak.extend(weak_arrays(&next_input));
    let actual = process(&mut restored, next_input, context).await;
    let all = prefix.into_iter().chain(next).collect::<Vec<_>>();
    assert_oracle(&actual, GLOBAL, dtype, &all, 9).await;
    let after = native_capture(&mut restored, dtype);
    assert!(weak.iter().all(|array| array.upgrade().is_none()));
    let target_pool = restored
        .stream_state
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    assert!(source_pool.reserved() > 0 && target_pool.reserved() > 0);
    drop((state, restored, snapshot, after, empty, actual));
    assert_eq!(source_pool.reserved(), 0);
    assert_eq!(target_pool.reserved(), 0);
}

#[tokio::test]
async fn test_current_float_extrema_global_bits_native3_roundtrip() {
    for dtype in [DataType::Float32, DataType::Float64] {
        for reverse_zero in [false, true] {
            let job = job();
            let context = StreamOperatorContext::new(&job, "float_extrema", None);
            let mut state = operator(&dtype, GLOBAL);
            let mut prefix = Vec::new();
            let mut weak = Vec::new();
            for (sequence, parts) in arrivals(reverse_zero).into_iter().enumerate() {
                let batch = input(&dtype, &parts, sequence as u64);
                weak.extend(weak_arrays(&batch));
                prefix.extend(parts);
                let actual = process(&mut state, batch, &context).await;
                assert_oracle(&actual, GLOBAL, &dtype, &prefix, sequence as u64).await;
            }
            let snapshot = native_capture(&mut state, &dtype);
            assert!(weak.iter().all(|array| array.upgrade().is_none()));
            drop(snapshot);
            recover_and_continue(state, &dtype, prefix, &context).await;
        }
    }
}

struct Reject;

#[async_trait]
impl StreamCollector for Reject {
    async fn emit(&mut self, _: &str, _: Batch) -> Result<()> {
        Err(CalcFlowError::Operator {
            node_id: "reject-extrema".into(),
            message: "injected emit failure".into(),
        })
    }
}

#[tokio::test]
async fn test_current_float_extrema_rejected_emit_preserves_state_and_retry() {
    for dtype in [DataType::Float32, DataType::Float64] {
        let job = job();
        let context = StreamOperatorContext::new(&job, "float_extrema", None);
        let mut state = operator(&dtype, GLOBAL);
        let first = vec![vec![Some(ZERO), None, Some(ONE)]];
        drop(process(&mut state, input(&dtype, &first, 0), &context).await);
        let before = state.checkpoint(Epoch::INITIAL).unwrap();
        let next = vec![vec![Some(NAN_A), Some(NEG_NAN), Some(NEG_ZERO)]];
        let rejected = input(&dtype, &next, 1);
        let weak = weak_arrays(&rejected);
        let failure = state
            .process_data("events", rejected, &context, &mut Reject)
            .await;
        assert!(
            matches!(failure, Err(CalcFlowError::Operator { node_id, .. }) if node_id == "reject-extrema")
        );
        assert!(weak.iter().all(|array| array.upgrade().is_none()));
        same_snapshot(&before, &state.checkpoint(Epoch::INITIAL).unwrap());
        let actual = process(&mut state, input(&dtype, &next, 1), &context).await;
        let all = first.into_iter().chain(next).collect::<Vec<_>>();
        assert_oracle(&actual, GLOBAL, &dtype, &all, 1).await;
        let after = native_capture(&mut state, &dtype);
        assert_eq!(after.inline_metadata["rows"], json!(6));
        let pool = state
            .stream_state
            .runtime()
            .unwrap()
            .incremental_memory_pool();
        drop((state, before, after, actual));
        assert_eq!(pool.reserved(), 0);
    }
}

async fn checked_prefixes(dtype: &DataType, query: &str, parts: Vec<Part>, native: bool) {
    let job = job();
    let context = StreamOperatorContext::new(&job, "float_extrema", None);
    let mut state = operator(dtype, query);
    let mut prefix = Vec::new();
    for (sequence, part) in parts.into_iter().enumerate() {
        let actual = process(
            &mut state,
            input(dtype, &[part.clone()], sequence as u64),
            &context,
        )
        .await;
        prefix.push(part);
        assert_oracle(&actual, query, dtype, &prefix, sequence as u64).await;
    }
    let snapshot = state.checkpoint(Epoch::INITIAL).unwrap();
    let layout = if native { 3 } else { 4 };
    assert_eq!(snapshot.inline_metadata["state_layout"], json!(layout));
    assert_eq!(snapshot.inline_metadata["state_accounting"], json!(layout));
    assert!(snapshot.segments.contains_key(if native {
        "group-state"
    } else {
        "input-retained"
    }));
    assert_eq!(
        state.incremental.is_some() && state.compact.is_some() && state.retained.is_none(),
        native
    );
    let pool = state
        .stream_state
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    drop((state, snapshot));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_current_float_extrema_grouped_and_mixed_sum_native3() {
    for dtype in [DataType::Float32, DataType::Float64] {
        checked_prefixes(
            &dtype,
            GROUPED,
            vec![vec![Some(ZERO)], vec![Some(NEG_ZERO)]],
            true,
        )
        .await;
        checked_prefixes(
            &dtype,
            GROUPED,
            vec![vec![Some(ONE)], vec![Some(NAN_A), Some(TWO)]],
            true,
        )
        .await;
        checked_prefixes(
            &dtype,
            "SELECT MIN(value) AS lo, MAX(value) AS hi, SUM(value) AS total FROM events",
            vec![vec![Some(ONE)], vec![Some(TWO), None]],
            true,
        )
        .await;
    }
}
