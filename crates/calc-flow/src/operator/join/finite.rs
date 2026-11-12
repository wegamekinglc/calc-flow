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
        let (left, right) = scratch.finite_inputs(inputs, run).await?;
        let pairs = scratch.finite_matches(&left, &right, run, runtime).await?;
        scratch.finite_output(&left, &right, &pairs, run).await
    }

    async fn finite_inputs(
        &mut self,
        inputs: &BTreeMap<String, Batch>,
        run: &RunContext,
    ) -> Result<(Vec<StoredRow>, Vec<AdmittedRow>)> {
        let left = required(inputs, "left", &self.name)?;
        let right = required(inputs, "right", &self.name)?;
        self.begin_batch("left", left)?;
        self.begin_batch("right", right)?;
        let left = self.finite_rows("left", left, run).await?;
        let right = self.finite_rows("right", right, run).await?;
        let admitted = right
            .into_iter()
            .map(|row| AdmittedRow {
                record: row.record,
                event_time: row.event_time,
                row_id: row.row_id,
                retain: true,
            })
            .collect();
        Ok((left, admitted))
    }

    async fn finite_matches(
        &self,
        left: &Vec<StoredRow>,
        right: &Vec<AdmittedRow>,
        run: &RunContext,
        runtime: &crate::DataFusionRuntime,
    ) -> Result<Vec<super::MatchedPair>> {
        if left.is_empty() || right.is_empty() {
            return Ok(Vec::new());
        }
        let plan = self.side_plan("right")?;
        let equal = self
            .finite_equal_pairs(left, right, &plan, run, runtime)
            .await?;
        Ok(filter_and_order_pairs(
            &self.spec.bounds,
            &plan,
            right,
            left,
            equal,
        ))
    }

    async fn finite_equal_pairs(
        &self,
        left: &Vec<StoredRow>,
        right: &Vec<AdmittedRow>,
        plan: &super::SidePlan,
        run: &RunContext,
        runtime: &crate::DataFusionRuntime,
    ) -> Result<Vec<(u64, u64)>> {
        let keys = state_key_batch(left, &self.compiled, plan, Some(self.input_schema(0)))?;
        let probe = probe_key_batch(right, &plan.key_indices, Some(self.input_schema(1)))?;
        let tables = equality_tables(probe, keys)?;
        let result = finite_query(
            runtime,
            &self.compiled.equality_query,
            &tables,
            &self.name,
            run,
        )
        .await?;
        decode_key_pairs(&result)
    }

    async fn finite_output(
        &self,
        left: &Vec<StoredRow>,
        right: &Vec<AdmittedRow>,
        pairs: &Vec<super::MatchedPair>,
        run: &RunContext,
    ) -> Result<BTreeMap<String, Batch>> {
        enforce_match_limit(
            pairs.len(),
            &mut 0,
            self.spec.limits.max_matches_per_input_batch,
            &self.name,
        )?;
        run.check_cancelled()?;
        let schema = self.output_ports[0].schema().expect("compiled join schema");
        let output = materialization::JoinOutput {
            schema,
            projection: None,
            admitted: right,
            opposite: left,
            matched: pairs,
            incoming_is_left: false,
            operator_id: &self.name,
            admitted_charges: None,
        };
        let records = finite_records(&output, pairs.len(), schema, run).await?;
        run.check_cancelled()?;
        let metadata = BatchMetadata::new(&self.name, 0, BTreeMap::new())?;
        Ok(BTreeMap::from([(
            "output".into(),
            Batch::table(records, metadata)?,
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
        let mut admission = FiniteAdmission::default();
        for record in batch.table_payload()?.batches() {
            self.finite_record(side, record, &plan, run, &mut admission)
                .await?;
        }
        Ok(admission.rows)
    }

    async fn finite_record(
        &mut self,
        side: &str,
        record: &RecordBatch,
        plan: &super::SidePlan,
        run: &RunContext,
        admission: &mut FiniteAdmission,
    ) -> Result<()> {
        let times = BatchEventTimes::new(
            record.column(plan.event_time_index).as_ref(),
            &self.name,
            side,
        )?;
        for start in (0..record.num_rows()).step_by(64) {
            run.check_cancelled()?;
            let length = (record.num_rows() - start).min(64);
            let mask = admission_masks::AdmissionMasks::finite(record, plan, &times, start, length);
            self.finite_block(side, record, plan, start..start + length, &mask, admission)?;
            tokio::task::yield_now().await;
        }
        Ok(())
    }

    fn finite_block(
        &mut self,
        side: &str,
        record: &RecordBatch,
        plan: &super::SidePlan,
        range: std::ops::Range<usize>,
        mask: &admission_masks::AdmissionMasks,
        admission: &mut FiniteAdmission,
    ) -> Result<()> {
        for offset in 0..range.len() {
            let identity = admission.row_id;
            admission.row_id = admission
                .row_id
                .checked_add(1)
                .ok_or_else(|| counter_overflow(&self.name, "row id"))?;
            if let RowAdmission::Admitted(event_time) = mask.at(offset, &self.name, side)? {
                self.finite_retain_row(
                    record,
                    plan,
                    range.start + offset,
                    identity,
                    event_time,
                    admission,
                )?;
            }
        }
        Ok(())
    }

    fn finite_retain_row(
        &mut self,
        record: &RecordBatch,
        plan: &super::SidePlan,
        physical: usize,
        identity: u64,
        event_time: crate::EventTime,
        admission: &mut FiniteAdmission,
    ) -> Result<()> {
        let row = self.finite_stored_row(record, plan, physical, identity, event_time)?;
        admission.bytes = admission
            .bytes
            .checked_add(row.charge)
            .ok_or_else(|| counter_overflow(&self.name, "state bytes"))?;
        self.validate_state_totals(
            plan.incoming_is_left,
            admission.rows.len() + 1,
            admission.bytes,
        )?;
        admission.rows.push(row);
        Ok(())
    }

    fn finite_stored_row(
        &self,
        record: &RecordBatch,
        plan: &super::SidePlan,
        physical: usize,
        row_id: u64,
        event_time: crate::EventTime,
    ) -> Result<StoredRow> {
        let key = encode_join_key_v1(record, physical, &plan.key_indices)?;
        let charge = state_row_charge_with_key(record, physical, key.len(), &self.name)?;
        Ok(StoredRow {
            record: columnar::RowPayload::from(record.clone()).with_offset(physical),
            event_time,
            row_id,
            charge,
            encoded_key: Arc::new(key.into()),
        })
    }
}

#[derive(Default)]
struct FiniteAdmission {
    row_id: u64,
    bytes: u64,
    rows: Vec<StoredRow>,
}

async fn finite_records(
    output: &materialization::JoinOutput<'_>,
    count: usize,
    schema: &datafusion::arrow::datatypes::SchemaRef,
    run: &RunContext,
) -> Result<Vec<RecordBatch>> {
    if count == 0 {
        return Ok(vec![RecordBatch::new_empty(schema.clone())]);
    }
    let mut records = Vec::new();
    for start in (0..count).step_by(8192) {
        run.check_cancelled()?;
        records.push(output.materialize(start..(start + 8192).min(count))?);
        tokio::task::yield_now().await;
    }
    Ok(records)
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
