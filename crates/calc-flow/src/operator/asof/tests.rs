use super::*;
#[path = "tests/columnar_production.rs"]
mod columnar_production;
mod source_replay;
use crate::{
    BatchMetadata, CancellationToken, EdgeBudget, EdgeCollector, Epoch, IngressProgress,
    IngressState, StreamJobContext, runtime::streaming::morsel_pool::JobMorselOwner,
};
use datafusion::arrow::{
    array::{Int64Array, StringArray, TimestampMicrosecondArray},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use std::{
    collections::BTreeMap,
    sync::{Arc, atomic::AtomicU64},
    time::Duration,
};

struct LateMetrics;

impl crate::operator::stream::LateMetricSink for LateMetrics {
    fn record(&self, _delta: crate::operator::stream::LateMetricDelta) -> Result<()> {
        Ok(())
    }
}

fn indexed_input(schema: &SchemaRef, rows: &[(&str, i64, i64)]) -> RecordBatch {
    RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from_iter_values(rows.iter().map(|row| row.0))),
            Arc::new(
                TimestampMicrosecondArray::from_iter_values(rows.iter().map(|row| row.1))
                    .with_timezone("UTC"),
            ),
            Arc::new(Int64Array::from_iter_values(rows.iter().map(|row| row.2))),
        ],
    )
    .unwrap()
}

pub(super) fn fixture() -> (StreamAsofJoinOperator, Batch) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Utf8, false),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("seq", DataType::Int64, false),
    ]));
    let side = |prefix: &str| {
        AsofJoinSide::new(
            vec!["key".into()],
            "time".into(),
            vec!["seq".into()],
            prefix.into(),
        )
        .unwrap()
    };
    let spec = StreamAsofJoinSpec::new(
        side("left"),
        side("right"),
        Duration::ZERO,
        AsofStateLimits::new(100, 1_048_576).unwrap(),
    )
    .unwrap();
    let operator =
        StreamAsofJoinOperator::new("asof", schema.clone(), schema.clone(), spec).unwrap();
    let batch = Batch::table(
        vec![indexed_input(&schema, &[("A", 100, 1)])],
        BatchMetadata::default(),
    )
    .unwrap();
    (operator, batch)
}

#[tokio::test]
async fn managed_checkpoint_uses_morsels_after_gather_admission_closes() {
    let (operator, _) = fixture();
    let morsels = JobMorselOwner::new(91, 2, Arc::new(AtomicU64::new(0)));
    let job = StreamJobContext::new(91, "asof", JsonMap::new(), None, CancellationToken::new())
        .with_morsel_owner(morsels.clone());
    job.gather_owner().close_admission();

    let (operator, snapshot) = Box::new(operator)
        .capture_managed(Epoch::INITIAL, &job, None)
        .await
        .unwrap();
    let progress = IngressProgressSnapshot::new(BTreeMap::from([
        (
            "left".into(),
            IngressProgress::new(IngressState::Active, None),
        ),
        (
            "right".into(),
            IngressProgress::new(IngressState::Active, None),
        ),
    ]));
    let restored = operator
        .restore_managed(snapshot, progress, None, &job, None)
        .await
        .unwrap();
    assert_eq!(restored.status().state_rows, 0);

    morsels.close_and_drain().await.unwrap();
    assert!(job.gather_owner().close_and_drain().await.is_empty());
}
