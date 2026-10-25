use super::*;
use crate::StateSegment;
use datafusion::arrow::{
    compute::cast,
    datatypes::{DataType, TimeUnit},
};

pub(super) const RECOVERY_QUERY: &str = "SELECT stock, LAST_VALUE(latest_price ORDER BY timestamp ASC NULLS FIRST, seq DESC NULLS LAST) IGNORE NULLS AS price FROM input GROUP BY stock";
pub(super) const SEED: [Row; 8] = [
    (Some("A"), Some(10.0), Some(-1), Some(0)),
    (None, None, None, None),
    (Some("B"), None, Some(5), Some(0)),
    (Some("C"), Some(3.0), Some(0), Some(0)),
    (Some("D"), Some(4.0), Some(0), Some(0)),
    (Some("E"), Some(5.0), Some(0), Some(0)),
    (Some("F"), Some(6.0), Some(0), Some(0)),
    (Some("G"), Some(7.0), Some(0), Some(0)),
];

pub(super) fn detached(snapshot: &OperatorStateSnapshot) -> OperatorStateSnapshot {
    OperatorStateSnapshot {
        inline_metadata: snapshot.inline_metadata.clone(),
        segments: snapshot
            .segments
            .iter()
            .map(|(id, s)| (id.clone(), StateSegment::new(s.bytes().to_vec())))
            .collect(),
    }
}

pub(super) fn control(snapshot: &OperatorStateSnapshot) -> Value {
    serde_json::from_slice(snapshot.segments["control"].bytes()).unwrap()
}

pub(super) fn delta_ids(snapshot: &OperatorStateSnapshot) -> Vec<String> {
    snapshot
        .segments
        .keys()
        .filter(|id| id.starts_with("group-delta-"))
        .cloned()
        .collect()
}

pub(super) async fn seeded(ctx: &StreamOperatorContext<'_>) -> SqlOperator {
    let mut op = operator(RECOVERY_QUERY);
    drop(process(&mut op, input(&SEED), ctx).await);
    op
}

fn update(sequence: i32) -> Vec<Row> {
    let price = f64::from(sequence);
    match sequence % 4 {
        0 => vec![(Some("A"), Some(price), Some(-100), Some(0))],
        1 => vec![(Some("A"), Some(price), Some(i64::from(sequence)), Some(0))],
        2 => vec![(
            Some("A"),
            Some(price),
            Some(i64::from(sequence - 1)),
            Some(0),
        )],
        _ => vec![
            (Some("B"), None, None, None),
            (Some("new"), Some(price), Some(0), Some(0)),
        ],
    }
}

async fn assert_oracle(actual: &Batch, prefix: &[Row]) {
    let expected = DataFusionRuntime::new(DataFusionConfig::default())
        .unwrap()
        .sql(
            RECOVERY_QUERY,
            &BTreeMap::from([("input".into(), input(prefix))]),
            None,
        )
        .await
        .unwrap();
    assert_eq!(values(actual), values(&expected));
    assert_eq!(
        actual.table_payload().unwrap().schema(),
        expected.table_payload().unwrap().schema()
    );
}

fn assert_log(previous: &OperatorStateSnapshot, next: &OperatorStateSnapshot, sequence: i32) {
    let frames = if sequence <= 32 {
        sequence
    } else {
        sequence - 33
    };
    assert_eq!(delta_ids(next).len(), usize::try_from(frames).unwrap());
    let log = control(next)["group_log"].clone();
    assert_eq!(log["generation"], json!(sequence));
    assert_eq!(
        log["base_generation"],
        json!(if sequence <= 32 { 0 } else { 33 })
    );
    assert_eq!(
        Arc::ptr_eq(
            &previous.segments["group-state"].bytes_arc(),
            &next.segments["group-state"].bytes_arc()
        ),
        sequence != 33
    );
}

#[tokio::test]
async fn test_ordered_last_delta_rewrite_and_every_cold_prefix_match_uninterrupted_and_df55() {
    let job = job();
    let ctx = StreamOperatorContext::new(&job, "ordered_last", None);
    let mut live = seeded(&ctx).await;
    let mut cold = seeded(&ctx).await;
    let mut previous = cold.checkpoint(Epoch::INITIAL).unwrap();
    let mut prefix = SEED.to_vec();
    for sequence in 1..=34 {
        let rows = update(sequence);
        prefix.extend_from_slice(&rows);
        let expected = process(&mut live, input(&rows), &ctx).await;
        let metadata = BatchMetadata::new(
            "quotes",
            u64::try_from(sequence).unwrap(),
            JsonMap::from([("cut".into(), json!(sequence))]),
        )
        .unwrap();
        let actual = process(
            &mut cold,
            input(&rows).with_metadata(metadata.clone()),
            &ctx,
        )
        .await;
        assert_eq!(values(&actual), values(&expected));
        assert_oracle(&actual, &prefix).await;
        assert_eq!(actual.metadata(), &metadata);
        if sequence % 2 == 0 {
            cold.prepare_checkpoint_async(&ctx).await.unwrap();
        }
        let snapshot = cold.checkpoint(Epoch::INITIAL).unwrap();
        assert_log(&previous, &snapshot, sequence);
        let mut restored = operator(RECOVERY_QUERY);
        StreamOperator::restore(&mut restored, &detached(&snapshot)).unwrap();
        same_snapshot(&snapshot, &restored.checkpoint(Epoch::INITIAL).unwrap());
        let after = process(&mut restored, input(&[]).with_metadata(metadata), &ctx).await;
        // Output row order exposes first-seen native group slots, including unset groups.
        assert_eq!(
            after.table_payload().unwrap().batches(),
            actual.table_payload().unwrap().batches()
        );
        cold = restored;
        previous = cold.checkpoint(Epoch::INITIAL).unwrap();
    }
}

fn timestamp_input(rows: &[Row], dtype: &DataType) -> Batch {
    let batch = input(rows);
    let record = &batch.table_payload().unwrap().batches()[0];
    let mut columns = record.columns().to_vec();
    // Cast the integer values directly: unit tests must not truncate negative instants.
    let ticks: ArrayRef = Arc::new(Int64Array::from(
        rows.iter().map(|r| r.2).collect::<Vec<_>>(),
    ));
    columns[2] = cast(&ticks, dtype).unwrap();
    let fields = record
        .schema()
        .fields()
        .iter()
        .enumerate()
        .map(|(i, f)| {
            if i == 2 {
                f.as_ref().clone().with_data_type(dtype.clone())
            } else {
                f.as_ref().clone()
            }
        })
        .collect::<Vec<_>>();
    Batch::table(
        vec![
            RecordBatch::try_new(
                Arc::new(datafusion::arrow::datatypes::Schema::new(fields)),
                columns,
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap()
}

async fn timestamp_case(dtype: DataType) {
    let job = job();
    let ctx = StreamOperatorContext::new(&job, "ordered_last", None);
    let mut op = operator(RECOVERY_QUERY);
    drop(process(&mut op, timestamp_input(&SEED, &dtype), &ctx).await);
    let full = op.checkpoint(Epoch::INITIAL).unwrap();
    let mut cold = operator(RECOVERY_QUERY);
    StreamOperator::restore(&mut cold, &detached(&full)).unwrap();
    let next = [(Some("A"), Some(99.0), Some(2), Some(0))];
    let actual = process(&mut cold, timestamp_input(&next, &dtype), &ctx).await;
    let expected = process(&mut op, timestamp_input(&next, &dtype), &ctx).await;
    assert_eq!(values(&actual), values(&expected));
    let delta = cold.checkpoint(Epoch::INITIAL).unwrap();
    assert_eq!(delta_ids(&delta).len(), 1);
    StreamOperator::restore(&mut op, &detached(&delta)).unwrap();
    let tie = [(Some("A"), Some(100.0), Some(2), Some(0))];
    let actual = process(&mut op, timestamp_input(&tie, &dtype), &ctx).await;
    assert_eq!(values(&actual), values(&expected));
}

#[tokio::test]
async fn test_ordered_last_full_delta_preserve_all_timestamp_units_and_timezones() {
    for unit in [
        TimeUnit::Second,
        TimeUnit::Millisecond,
        TimeUnit::Microsecond,
        TimeUnit::Nanosecond,
    ] {
        timestamp_case(DataType::Timestamp(unit, None)).await;
        timestamp_case(DataType::Timestamp(unit, Some("+05:30".into()))).await;
    }
}
