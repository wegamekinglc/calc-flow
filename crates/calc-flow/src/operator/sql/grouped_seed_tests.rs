use super::*;
use crate::{
    Batch, BatchMetadata, CancellationToken, DataFusionConfig, DataFusionRuntime, JsonMap,
    StreamJobContext, StreamOperatorContext,
};
use datafusion::arrow::{
    array::{BooleanArray, Float32Array, Float64Array, Int64Array},
    datatypes::{Field, Schema},
    record_batch::RecordBatch,
};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

const QUERY: &str = "SELECT key, SUM(value), AVG(value), MIN(value), MAX(value), COUNT(value), COUNT(*) FROM events GROUP BY key";
type MergeRows = Arc<Mutex<Vec<usize>>>;

struct ObservedAccumulator {
    native: Box<dyn GroupsAccumulator>,
    merge_rows: MergeRows,
}

impl GroupsAccumulator for ObservedAccumulator {
    fn update_batch(
        &mut self,
        values: &[ArrayRef],
        indices: &[usize],
        filter: Option<&BooleanArray>,
        count: usize,
    ) -> DataFusionResult<()> {
        self.native.update_batch(values, indices, filter, count)
    }

    fn evaluate(&mut self, emit_to: EmitTo) -> DataFusionResult<ArrayRef> {
        self.native.evaluate(emit_to)
    }

    fn state(&mut self, emit_to: EmitTo) -> DataFusionResult<Vec<ArrayRef>> {
        self.native.state(emit_to)
    }

    fn merge_batch(
        &mut self,
        values: &[ArrayRef],
        indices: &[usize],
        filter: Option<&BooleanArray>,
        count: usize,
    ) -> DataFusionResult<()> {
        self.merge_rows.lock().unwrap().push(indices.len());
        self.native.merge_batch(values, indices, filter, count)
    }

    fn size(&self) -> usize {
        self.native.size()
    }
}

fn observe(accumulator: PartialAccumulator) -> (PartialAccumulator, MergeRows) {
    let merge_rows = Arc::new(Mutex::new(Vec::new()));
    (
        PartialAccumulator {
            native: Box::new(ObservedAccumulator {
                native: accumulator.native,
                merge_rows: merge_rows.clone(),
            }),
            numeric: accumulator.numeric,
        },
        merge_rows,
    )
}

fn record(dtype: &DataType, start: i64, end: i64, phase: usize) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new("value", dtype.clone(), true),
    ]));
    let patterns = [
        Some((0, 0)),
        Some((0x8000_0000, 0x8000_0000_0000_0000)),
        Some((0x7fc0_0005, 0x7ff8_0000_0000_0005)),
        Some((0xff80_0007, 0xfff0_0000_0000_0007)),
        Some((f32::INFINITY.to_bits(), f64::INFINITY.to_bits())),
        Some((f32::NEG_INFINITY.to_bits(), f64::NEG_INFINITY.to_bits())),
        Some((1, 1)),
        Some((0x5a80_0000, 0x4350_0000_0000_0000)),
        Some((0xda80_0000, 0xc350_0000_0000_0000)),
        None,
    ];
    let bits = (start..end)
        .map(|key| patterns[(usize::try_from(key).unwrap() + phase) % patterns.len()])
        .collect::<Vec<_>>();
    let values: ArrayRef = match dtype {
        DataType::Float32 => Arc::new(Float32Array::from(
            bits.into_iter()
                .map(|bits| bits.map(|(single, _)| f32::from_bits(single)))
                .collect::<Vec<_>>(),
        )),
        DataType::Float64 => Arc::new(Float64Array::from(
            bits.into_iter()
                .map(|bits| bits.map(|(_, double)| f64::from_bits(double)))
                .collect::<Vec<_>>(),
        )),
        _ => unreachable!(),
    };
    RecordBatch::try_new(
        schema,
        vec![Arc::new(Int64Array::from_iter_values(start..end)), values],
    )
    .unwrap()
}

fn rows(records: &[RecordBatch]) -> BTreeMap<i64, Vec<String>> {
    records
        .iter()
        .flat_map(|record| {
            let keys = record
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            (0..record.num_rows()).map(move |row| {
                let values = record
                    .columns()
                    .iter()
                    .map(
                        |array| match ScalarValue::try_from_array(array, row).unwrap() {
                            ScalarValue::Float32(Some(value)) => {
                                format!("f32:{:08x}", value.to_bits())
                            }
                            ScalarValue::Float64(Some(value)) => {
                                format!("f64:{:016x}", value.to_bits())
                            }
                            value => format!("{value:?}"),
                        },
                    )
                    .collect();
                (keys.value(row), values)
            })
        })
        .collect()
}

#[tokio::test]
async fn test_grouped_seed_merges_columns_and_preserves_exact_prefix() {
    for dtype in [DataType::Float32, DataType::Float64] {
        case(&dtype).await;
    }
}

async fn case(dtype: &DataType) {
    use super::super::IncrementalSql;

    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let first = record(dtype, 0, 128, 0);
    let query = crate::expression::parse_select_query(QUERY).unwrap();
    let mut plan = IncrementalSql::plan(&runtime, &query, "events", first.schema(), "seed")
        .await
        .unwrap()
        .unwrap();
    let job = StreamJobContext::new(1933, "seed", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "seed", None);
    let initial = Batch::table(vec![first.clone()], BatchMetadata::default()).unwrap();
    let transaction = plan.update(&initial, &context, "seed").await.unwrap();
    plan.commit(transaction);
    assert!(plan.sequential.is_some());

    let records = vec![record(dtype, 0, 64, 1), record(dtype, 64, 160, 2)];
    let (reservation, workspace, mut candidates) = plan.input_candidates(160, "seed").unwrap();
    let partial = candidates.partial.as_mut().unwrap();
    let (accumulators, observations): (Vec<_>, Vec<MergeRows>) =
        partial.accumulators.drain(..).map(observe).unzip();
    partial.accumulators = accumulators;
    plan.update_records(
        &records,
        &mut candidates,
        (&reservation, workspace),
        &context,
        "seed",
    )
    .await
    .unwrap();
    plan.finish_candidates(&mut candidates, &context, "seed")
        .await
        .unwrap();
    let count = plan.output_count(&candidates, "seed").unwrap();
    let actual = plan
        .output_records(count, &candidates.groups, &reservation, &context, "seed")
        .await
        .unwrap();
    let mut prefix = vec![first];
    prefix.extend(records);
    let expected = runtime
        .sql(
            QUERY,
            &BTreeMap::from([(
                "events".into(),
                Batch::table(prefix, BatchMetadata::default()).unwrap(),
            )]),
            Some("seed"),
        )
        .await
        .unwrap();
    assert_eq!(
        &actual[0].schema(),
        expected.table_payload().unwrap().schema()
    );
    assert_eq!(
        rows(&actual),
        rows(expected.table_payload().unwrap().batches())
    );
    for (expression, observation) in plan.aggregates.iter().zip(observations) {
        if !super::super::grouped_float::selected(expression) {
            continue;
        }
        let calls = observation.lock().unwrap();
        let limit = if selected(expression) { 2 } else { 4 };
        assert!(
            calls.len() <= limit,
            "{} seeded {} singleton batches instead of columns",
            expression.fun().name(),
            calls.len()
        );
        assert!(calls.iter().all(|rows| *rows > 1));
    }
    drop((candidates, reservation, plan, initial, actual, expected));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert_eq!(runtime.incremental_memory_pool().reserved(), 0);
}
