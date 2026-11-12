//! Run-local adaptation of the native admission, equality, bounds and gather kernels.
use super::{
    AdmittedRow, BatchEventTimes, RowAdmission, StoredRow, StreamJoinOperator, admission_masks,
    columnar, counter_overflow, decode_key_pairs, encode_join_key_v1, enforce_match_limit,
    equality_tables, filter_and_order_pairs, materialization, operator_error, probe_key_batch,
    state_key_batch, state_row_charge_with_key,
};
use crate::{Batch, DataFusionConfig, Result, expression::ValidatedQuery};
use crate::{BatchMetadata, BatchOperator, BatchOperatorContext, RunContext};
use async_trait::async_trait;
use datafusion::arrow::record_batch::RecordBatch;
use std::{collections::BTreeMap, sync::Arc, time::Duration};

#[async_trait]
impl BatchOperator for StreamJoinOperator {
    async fn process(
        &mut self,
        inputs: &BTreeMap<String, Batch>,
        context: &BatchOperatorContext<'_>,
    ) -> Result<BTreeMap<String, Batch>> {
        let runtime = crate::DataFusionRuntime::new(DataFusionConfig::default())?;
        self.process_finite(inputs, context.run, &runtime).await
    }
}

impl StreamJoinOperator {
    pub(crate) async fn process_finite(
        &self,
        inputs: &BTreeMap<String, Batch>,
        run: &RunContext,
        runtime: &crate::DataFusionRuntime,
    ) -> Result<BTreeMap<String, Batch>> {
        let mut scratch = Self::new(
            &self.name,
            self.input_schema(0).clone(),
            self.input_schema(1).clone(),
            self.spec.clone(),
        )?;
        let left = required(inputs, "left", &self.name)?;
        let right = required(inputs, "right", &self.name)?;
        scratch.begin_batch("left", left)?;
        scratch.begin_batch("right", right)?;
        let left = scratch.finite_rows("left", left, run).await?;
        let right = scratch.finite_rows("right", right, run).await?;
        let plan = scratch.side_plan("right")?;
        let admitted: Vec<AdmittedRow> = right
            .iter()
            .map(|row| AdmittedRow {
                record: row.record.clone(),
                event_time: row.event_time,
                row_id: row.row_id,
                retain: true,
            })
            .collect();
        let pairs = if left.is_empty() || right.is_empty() {
            Vec::new()
        } else {
            let keys = state_key_batch(
                &left,
                &scratch.compiled,
                &plan,
                Some(scratch.input_schema(0)),
            )?;
            let probe =
                probe_key_batch(&admitted, &plan.key_indices, Some(scratch.input_schema(1)))?;
            let tables = equality_tables(probe, keys)?;
            let result = finite_query(
                runtime,
                &scratch.compiled.equality_query,
                &tables,
                &scratch.name,
                run,
            )
            .await?;
            let equal = decode_key_pairs(&result)?;
            filter_and_order_pairs(&scratch.spec.bounds, &plan, &admitted, &left, equal)
        };
        enforce_match_limit(
            pairs.len(),
            &mut 0,
            scratch.spec.limits.max_matches_per_input_batch,
            &scratch.name,
        )?;
        run.check_cancelled()?;
        let schema = scratch.output_ports[0]
            .schema()
            .expect("compiled join schema");
        let records = if pairs.is_empty() {
            vec![RecordBatch::new_empty(schema.clone())]
        } else {
            let output = materialization::JoinOutput {
                schema,
                projection: None,
                admitted: &admitted,
                opposite: &left,
                matched: &pairs,
                incoming_is_left: false,
                operator_id: &scratch.name,
                admitted_charges: None,
            };
            let mut records = Vec::new();
            for start in (0..pairs.len()).step_by(8192) {
                run.check_cancelled()?;
                records.push(output.materialize(start..(start + 8192).min(pairs.len()))?);
                tokio::task::yield_now().await;
            }
            records
        };
        run.check_cancelled()?;
        Ok(BTreeMap::from([(
            "output".into(),
            Batch::table(records, BatchMetadata::new(&self.name, 0, BTreeMap::new())?)?,
        )]))
    }
}

fn required<'a>(inputs: &'a BTreeMap<String, Batch>, side: &str, name: &str) -> Result<&'a Batch> {
    inputs
        .get(side)
        .ok_or_else(|| operator_error(name, &format!("missing required {side} input")))
}

impl StreamJoinOperator {
    async fn finite_rows(
        &mut self,
        side: &str,
        batch: &Batch,
        run: &RunContext,
    ) -> Result<Vec<StoredRow>> {
        let plan = self.side_plan(side)?;
        let mut rows = Vec::new();
        let mut row_id = 0_u64;
        let mut bytes = 0_u64;
        for record in batch.table_payload()?.batches() {
            let times = BatchEventTimes::new(
                record.column(plan.event_time_index).as_ref(),
                &self.name,
                side,
            )?;
            for start in (0..record.num_rows()).step_by(64) {
                run.check_cancelled()?;
                let length = (record.num_rows() - start).min(64);
                let mask =
                    admission_masks::AdmissionMasks::finite(record, &plan, &times, start, length);
                for offset in 0..length {
                    let identity = row_id;
                    row_id = row_id
                        .checked_add(1)
                        .ok_or_else(|| counter_overflow(&self.name, "row id"))?;
                    if let RowAdmission::Admitted(event_time) = mask.at(offset, &self.name, side)? {
                        let physical = start + offset;
                        let key = encode_join_key_v1(record, physical, &plan.key_indices)?;
                        let charge =
                            state_row_charge_with_key(record, physical, key.len(), &self.name)?;
                        bytes = bytes
                            .checked_add(charge)
                            .ok_or_else(|| counter_overflow(&self.name, "state bytes"))?;
                        self.validate_state_totals(plan.incoming_is_left, rows.len() + 1, bytes)?;
                        rows.push(StoredRow {
                            record: columnar::RowPayload::from(record.clone())
                                .with_offset(physical),
                            event_time,
                            row_id: identity,
                            charge,
                            encoded_key: Arc::new(key.into()),
                        });
                    }
                }
                tokio::task::yield_now().await;
            }
        }
        Ok(rows)
    }
}

async fn finite_query(
    runtime: &crate::DataFusionRuntime,
    query: &ValidatedQuery,
    tables: &BTreeMap<String, Batch>,
    name: &str,
    run: &RunContext,
) -> Result<Batch> {
    let future = runtime.sql_validated(query, tables, Some(name));
    tokio::pin!(future);
    loop {
        run.check_cancelled()?;
        tokio::select! {
            result = &mut future => return result,
            () = tokio::time::sleep(Duration::from_millis(10)) => {}
        }
    }
}
