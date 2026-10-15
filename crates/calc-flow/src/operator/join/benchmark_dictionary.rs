//! Fixed three-row diagnostic for genuine optional dictionary refusal.

use std::{fmt, sync::Arc, time::Duration};

use datafusion::{
    arrow::{
        array::{Array, Int64Array, StringArray, TimestampMicrosecondArray},
        datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit},
        record_batch::RecordBatch,
    },
    execution::memory_pool::{MemoryLimit, MemoryPool, MemoryReservation, UnboundedMemoryPool},
};
use parking_lot::Mutex;
use serde::Serialize;
use serde_json::{Value, json};

use super::{
    JoinStateLimits, JoinTimeBounds, StreamJoinOperator, StreamJoinSpec, state_row_charge,
};
use crate::{
    Batch, BatchMetadata, CancellationToken, EdgeCollector, JsonMap, OperatorMetadata, Result,
    StreamJobContext, StreamOperator, StreamOperatorContext,
};

const DICTIONARY_CONSUMER: &str = "sql-incremental:stream-join-dictionary";

#[derive(Debug, Default, Serialize)]
struct PoolEvidence {
    failed_dictionary_grows: u64,
    successful_sql_grows: u64,
    last_rejected_consumer: Option<String>,
    last_rejected_bytes: usize,
}

#[derive(Debug, Default)]
struct DictionaryRefusalPool {
    inner: UnboundedMemoryPool,
    evidence: Mutex<PoolEvidence>,
}

impl DictionaryRefusalPool {
    fn record_sql_growth(&self, reservation: &MemoryReservation, bytes: usize) {
        if bytes > 0 && reservation.consumer().name().starts_with("HashJoinInput") {
            self.evidence.lock().successful_sql_grows += 1;
        }
    }

    fn snapshot(&self) -> Result<Value> {
        let mut evidence = serde_json::to_value(&*self.evidence.lock()).map_err(|error| {
            crate::benchmark_internals::invalid(format!("refusal pool evidence: {error}"))
        })?;
        evidence["reserved_bytes"] = self.reserved().into();
        Ok(evidence)
    }
}

impl fmt::Display for DictionaryRefusalPool {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("Join dictionary refusal observer")
    }
}

impl MemoryPool for DictionaryRefusalPool {
    fn name(&self) -> &'static str {
        "join-dictionary-refusal"
    }

    fn memory_limit(&self) -> MemoryLimit {
        self.inner.memory_limit()
    }

    fn grow(&self, reservation: &MemoryReservation, bytes: usize) {
        self.inner.grow(reservation, bytes);
        self.record_sql_growth(reservation, bytes);
    }

    fn shrink(&self, reservation: &MemoryReservation, bytes: usize) {
        self.inner.shrink(reservation, bytes);
    }

    fn reserved(&self) -> usize {
        self.inner.reserved()
    }

    fn try_grow(
        &self,
        reservation: &MemoryReservation,
        bytes: usize,
    ) -> datafusion::common::Result<()> {
        if bytes > 0 && reservation.consumer().name() == DICTIONARY_CONSUMER {
            let mut evidence = self.evidence.lock();
            evidence.failed_dictionary_grows += 1;
            evidence.last_rejected_consumer = Some(reservation.consumer().name().to_owned());
            evidence.last_rejected_bytes = bytes;
            return Err(datafusion::common::DataFusionError::ResourcesExhausted(
                "optional Join dictionary grant refused by diagnostic pool".into(),
            ));
        }
        self.inner.try_grow(reservation, bytes)?;
        self.record_sql_growth(reservation, bytes);
        Ok(())
    }
}

impl StreamJoinOperator {
    pub(crate) async fn benchmark_dictionary_refusal_retention() -> Result<Value> {
        let mut operator = operator()?;
        let pool = Arc::new(DictionaryRefusalPool::default());
        operator.runtime.initialize_benchmark_pool(pool.clone())?;
        let job = StreamJobContext::new(
            1,
            "join-dictionary-refusal",
            JsonMap::new(),
            None,
            CancellationToken::new(),
        );
        let observed = observe(&mut operator, &job).await;
        drop(operator);
        let cleanup = crate::benchmark_internals::settle_job(&job).await;
        drop(job);
        let mut observed = observed?;
        cleanup?;
        if pool.reserved() != 0 {
            return Err(crate::benchmark_internals::invalid(
                "dictionary refusal control retained memory after owned cleanup",
            ));
        }
        observed["pool"] = pool.snapshot()?;
        observed["owned_cleanup_complete"] = true.into();
        Ok(observed)
    }
}

fn operator() -> Result<StreamJoinOperator> {
    let schema = schema();
    let spec = StreamJoinSpec::inner(
        ["key"],
        ["key"],
        "time",
        "time",
        JoinTimeBounds::new(Duration::ZERO, Duration::ZERO)?,
        JoinStateLimits::new(100, 1_000_000, 1_000)?,
    )?;
    StreamJoinOperator::new("match", Arc::clone(&schema), schema, spec)
}

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("key", DataType::Utf8, false),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new("amount", DataType::Int64, false),
    ]))
}

fn batch(keys: Vec<&str>) -> Result<Batch> {
    let rows = keys.len();
    let record = RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(StringArray::from(keys)),
            Arc::new(TimestampMicrosecondArray::from(vec![100; rows])),
            Arc::new(Int64Array::from(vec![7; rows])),
        ],
    )
    .map_err(|error| crate::benchmark_internals::invalid(format!("refusal fixture: {error}")))?;
    Batch::table(vec![record], BatchMetadata::default())
}

async fn observe(operator: &mut StreamJoinOperator, job: &StreamJobContext) -> Result<Value> {
    let left = batch(vec!["a", "b"])?;
    let right = batch(vec!["a"])?;
    let expected_fees = row_fees(&left, &right)?;
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let context = StreamOperatorContext::new(job, "match", None);
    operator
        .process_data("left", left, &context, &mut output)
        .await?;
    operator
        .process_data("right", right, &context, &mut output)
        .await?;
    let actual_fees = operator
        .state
        .left
        .iter()
        .chain(operator.state.right.iter())
        .map(|row| row.charge)
        .collect::<Vec<_>>();
    if actual_fees != expected_fees {
        return Err(crate::benchmark_internals::invalid(
            "dictionary refusal changed full V1 retained row charges",
        ));
    }
    let output_keys = output_keys(&mut output)?;
    let status = operator.status();
    Ok(json!({
        "schema": "calc-flow.join-dictionary-refusal.v1",
        "case": "join_dictionary_refusal_retention",
        "check_only": true,
        "admission_batch_rows": [2, 1],
        "left_retained_rows": status.left.retained_rows,
        "right_retained_rows": status.right.retained_rows,
        "left_retained_bytes": status.left.retained_bytes,
        "right_retained_bytes": status.right.retained_bytes,
        "full_v1_row_charges": actual_fees,
        "output_keys": output_keys,
        "state_limit_failures": status.state_limit_failures,
        "match_limit_failures": status.match_limit_failures,
        "dictionary_grant_refusals": operator.key_work.dictionary_grant_refusals,
        "needs_rebuild_routes": operator.key_work.needs_rebuild_routes,
        "legacy_key_encodings": operator.key_work.legacy_key_encodings,
        "sql_probe_table_builds": operator.key_work.sql_probe_table_builds,
        "rebuild_attempts": operator.key_work.rebuild_attempts,
        "rebuild_completions": operator.key_work.rebuild_completions,
    }))
}

fn row_fees(left: &Batch, right: &Batch) -> Result<[u64; 3]> {
    let left_record = &left.table_payload()?.batches()[0];
    let right_record = &right.table_payload()?.batches()[0];
    Ok([
        state_row_charge(left_record, 0, &[0], "match")?,
        state_row_charge(left_record, 1, &[0], "match")?,
        state_row_charge(right_record, 0, &[0], "match")?,
    ])
}

fn output_keys(output: &mut EdgeCollector) -> Result<Vec<[String; 2]>> {
    let mut keys = Vec::new();
    for message in output.drain("output") {
        let batch = message.as_data().ok_or_else(|| {
            crate::benchmark_internals::invalid("unexpected refusal control output")
        })?;
        for record in batch.table_payload()?.batches() {
            keys.extend(record_keys(record)?);
        }
    }
    Ok(keys)
}

fn record_keys(record: &RecordBatch) -> Result<Vec<[String; 2]>> {
    let column = |index| {
        record
            .column(index)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| crate::benchmark_internals::invalid("invalid refusal output key"))
    };
    let left = column(0)?;
    let right = column(3)?;
    if left.null_count() != 0 || right.null_count() != 0 {
        return Err(crate::benchmark_internals::invalid(
            "null refusal output key",
        ));
    }
    Ok((0..record.num_rows())
        .map(|row| [left.value(row).to_owned(), right.value(row).to_owned()])
        .collect())
}

#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn test_persistent_dictionary_refusal_diagnostic_preserves_retention_and_pool_evidence() {
        let evidence = crate::benchmark_internals::join_dictionary_refusal_retention_evidence()
            .await
            .unwrap();
        assert_eq!(evidence["admission_batch_rows"], serde_json::json!([2, 1]));
        assert_eq!(evidence["left_retained_rows"], 2);
        assert_eq!(evidence["right_retained_rows"], 1);
        assert_eq!(evidence["left_retained_bytes"], 228);
        assert_eq!(evidence["right_retained_bytes"], 114);
        assert_eq!(
            evidence["full_v1_row_charges"],
            serde_json::json!([114, 114, 114])
        );
        assert_eq!(evidence["output_keys"], serde_json::json!([["a", "a"]]));
        assert_eq!(evidence["state_limit_failures"], 0);
        assert_eq!(evidence["match_limit_failures"], 0);
        assert_eq!(evidence["dictionary_grant_refusals"], 1);
        assert_eq!(evidence["needs_rebuild_routes"], 1);
        assert_eq!(evidence["legacy_key_encodings"], 3);
        assert_eq!(evidence["sql_probe_table_builds"], 1);
        assert_eq!(evidence["rebuild_attempts"], 0);
        assert_eq!(evidence["rebuild_completions"], 0);
        assert_eq!(evidence["pool"]["failed_dictionary_grows"], 1);
        assert_eq!(
            evidence["pool"]["last_rejected_consumer"],
            "sql-incremental:stream-join-dictionary"
        );
        assert!(evidence["pool"]["last_rejected_bytes"].as_u64().unwrap() > 0);
        assert!(evidence["pool"]["successful_sql_grows"].as_u64().unwrap() > 0);
        assert_eq!(evidence["pool"]["reserved_bytes"], 0);
        assert_eq!(evidence["owned_cleanup_complete"], true);
    }
}
