use super::super::super::native_lookup::{reset_window_run_searches, window_run_searches};
use super::super::partition;
use super::*;
use crate::{
    Batch, BatchMetadata, CancellationToken, EdgeCollector, JoinStateLimits, JoinTimeBounds,
    JsonMap, OperatorMetadata, StreamJoinSpec, StreamOperator,
};
use datafusion::arrow::{
    array::{Int64Array, TimestampMicrosecondArray},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use std::time::Duration;

fn window_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
    ]))
}

fn window_batch(key: i64, times: &[i64]) -> Batch {
    Batch::table(
        vec![
            RecordBatch::try_new(
                window_schema(),
                vec![
                    Arc::new(Int64Array::from(vec![key; times.len()])),
                    Arc::new(TimestampMicrosecondArray::from(times.to_vec()).with_timezone("UTC")),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap()
}

// A gather probe over one key: event-time-ordered rows across two units, with
// one out-of-order row and one recovering row at the tail of the second unit.
async fn gather_fixture() -> (StreamJoinOperator, Arc<ProbeInputs>, StreamJobContext) {
    let spec = StreamJoinSpec::inner(
        ["key"],
        ["key"],
        "time",
        "time",
        JoinTimeBounds::new(Duration::from_micros(10), Duration::from_micros(20)).unwrap(),
        JoinStateLimits::new(100_000, 100_000_000, 1_000_000).unwrap(),
    )
    .unwrap();
    let mut operator =
        StreamJoinOperator::new("probe", window_schema(), window_schema(), spec).unwrap();
    let job = StreamJobContext::new(91, "probe", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "probe", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let mut right_times = Vec::new();
    for index in 0..40_i64 {
        let time = index * 3;
        right_times.push(time);
        if index % 4 == 0 {
            right_times.push(time);
        }
    }
    operator
        .process_data(
            "right",
            window_batch(7, &right_times),
            &context,
            &mut output,
        )
        .await
        .unwrap();
    let rows = 16_400;
    assert_eq!(partition::units(rows), 2, "fixture must span two units");
    let mut probe_times: Vec<i64> = (0..rows - 2)
        .map(|index| i64::try_from(index).unwrap() * 10)
        .collect();
    probe_times.extend([5, 15]);
    let input = window_batch(7, &probe_times);
    let plan = operator.begin_batch("left", &input).unwrap();
    let mut bundle = operator.admission_bundle(&plan, &input).unwrap();
    operator
        .admit_batch_records("left", &input, &plan, &context, &mut bundle)
        .await
        .unwrap();
    bundle.finish("probe").unwrap();
    let keys = operator
        .native_probe_keys(&plan, &bundle.admitted)
        .unwrap()
        .unwrap();
    assert!(operator.ensure_native_index(false).unwrap());
    let inputs = operator
        .capture_keyed_probe(&plan, bundle.admitted, keys, job.gather_owner())
        .unwrap()
        .ready
        .unwrap();
    (operator, inputs, job)
}

// Plain uncached window per probe row: the oracle for gathered pairs.
fn expected_pairs(inputs: &ProbeInputs) -> Vec<(usize, usize)> {
    let index = inputs.index.as_ref().expect("owned native index");
    (0..inputs.admitted.len())
        .flat_map(|position| {
            let id = inputs.slot(position).expect("retained opposite key");
            index
                .range_by_id(id, inputs.range(position))
                .map(|opposite_index| (position, opposite_index))
                .collect::<Vec<_>>()
        })
        .collect()
}

#[tokio::test]
async fn test_gather_probe_windows_use_cursors_instead_of_per_row_searches() {
    let (mut operator, inputs, job) = gather_fixture().await;
    let expected = expected_pairs(&inputs);
    assert!(!expected.is_empty());
    let stop = GatherStop::from_job(&job);
    let count = CountWork {
        inputs: Arc::clone(&inputs),
        limit: 1_000_000,
    };
    let units = count.unit_count();
    reset_window_run_searches();
    let values = (0..units)
        .map(|unit| count.run(unit, &stop).unwrap())
        .collect::<Vec<_>>();
    let counted = window_run_searches();
    let counts = Counts::ordered(values, 1_000_000).unwrap();
    let (funding, credit) = operator.admit_fill(counts).unwrap().unwrap();
    let fill = FillWork {
        inputs: Arc::clone(&inputs),
        counts,
        funding,
    };
    reset_window_run_searches();
    let fragments = (0..units)
        .map(|unit| fill.run(unit, &stop).unwrap())
        .collect::<Vec<_>>();
    let filled = window_run_searches();
    let output = pairs::ordered_output(fragments, credit, counts.total).unwrap();
    let identities = output
        .pairs
        .iter()
        .map(|pair| (pair.pos, pair.opposite_index))
        .collect::<Vec<_>>();
    assert_eq!(identities, expected);
    // Two units, one key: per phase each unit pays one first touch, and the
    // second unit also pays one out-of-order fallback — never four searches
    // per probed row.
    assert_eq!(counted, 2 * (2 + 1), "count-phase full-run searches");
    assert_eq!(filled, 2 * (2 + 1), "fill-phase full-run searches");
}
