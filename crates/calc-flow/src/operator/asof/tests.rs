use super::*;
#[path = "tests/columnar_production.rs"]
mod columnar_production;
mod source_replay;
use crate::{
    BatchMetadata, CancellationToken, EdgeBudget, EdgeCollector, Epoch, IngressProgress,
    IngressState, StreamJobContext,
};
use datafusion::arrow::{
    array::{Int64Array, StringArray, TimestampMicrosecondArray},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use std::{collections::BTreeMap, sync::Arc, time::Duration};

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
