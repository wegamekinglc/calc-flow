use datafusion::arrow::record_batch::RecordBatch;
use sha2::{Digest, Sha256};

use super::super::{
    decoded_record,
    footprint::Footprint,
    footprint_state,
    resources::Commit,
    state_decode_requirements,
    state_schema_requirements::{self, Metadata},
};
use super::{
    Check, Prepared, Result, Work, add, budget_error, generation_check, mismatch, outer, reserve,
};
use crate::operator::rolling::{DecodedRollingState, state_codec, state_v3};

struct Decoded {
    state: DecodedRollingState,
    reader: Option<decoded_record::DecodedRecord>,
    schema: Option<state_codec::FreshStateSchema>,
    candidate_limit: u64,
}
impl Work {
    pub(super) fn run(mut self, check: Check<'_>) -> Result<Prepared> {
        let generation = self.generation.clone();
        let job = self.job_id;
        let mut checked = || {
            check()?;
            generation_check(&generation, job)
        };
        checked()?;
        let next_late = crate::operator::late_output::LateOutputSnapshot::restore_sequence(
            self.late_policy,
            self.metadata.late_output.as_ref(),
        )?;
        let decoded = self.decode(&mut checked)?;
        let prepared = self.finish(decoded, next_late, &mut checked)?;
        checked()?;
        Ok(prepared)
    }
    fn finish(mut self, decoded: Decoded, next_late: u64, check: Check<'_>) -> Result<Prepared> {
        let baseline = reader_backing(decoded.reader.as_ref(), &self.node, check)?;
        let epoch = self.metadata.epoch;
        let state = outer::candidate(decoded.state, self.metadata, next_late);
        let (candidate, footprint) = footprint_state::prepare_checked(state, &self.node, check)?;
        verify_candidate(
            &candidate,
            &footprint,
            &baseline,
            decoded.candidate_limit,
            self.budget,
            &self.node,
        )?;
        drop(baseline);
        drop(decoded.reader);
        drop(decoded.schema);
        self.credit
            .adjust(footprint.clone(), 0, self.budget, &self.node)
            .map_err(|error| budget_error(error, &self.node))?;
        Ok(Prepared {
            source: self.source,
            candidate,
            commit: Commit::new(footprint),
            epoch,
            generation: self.generation,
            job_id: self.job_id,
            node: self.node,
            credit: self.credit,
            _guard: self.guard,
        })
    }
    fn decode(&mut self, check: Check<'_>) -> Result<Decoded> {
        let Some(segment) = self.segment.take() else {
            return self.empty_decoded();
        };
        checksum(&segment, &self.metadata, check)?;
        let result = self
            .decode_segment(&segment, check)
            .map_err(|error| state_codec::layout_five_restore_error(error, &self.node));
        drop(segment);
        result
    }
    fn empty_decoded(&mut self) -> Result<Decoded> {
        self.reserve_total(outer::arc_bytes(&self.node)?)?;
        Ok(Decoded {
            state: DecodedRollingState::default(),
            reader: None,
            schema: None,
            candidate_limit: add(outer::arc_bytes(&self.node)?, outer::ids(&self.metadata)?)?,
        })
    }
    fn decode_segment(
        &mut self,
        segment: &crate::StateSegment,
        check: Check<'_>,
    ) -> Result<Decoded> {
        let quote = state_schema_requirements::plan(&self.schema, self.schema_metadata()?, check)?;
        self.reserve_total(quote.construction_peak)?;
        let schema = self.construct_schema()?;
        let reader = self.read_record(segment, &schema, quote.retained, check)?;
        let (state, candidate_limit) =
            self.reconstruct(&reader.record, reader.reader_bytes, quote.retained, check)?;
        Ok(Decoded {
            state,
            reader: Some(reader),
            schema: Some(schema),
            candidate_limit,
        })
    }
    fn construct_schema(&self) -> Result<state_codec::FreshStateSchema> {
        let metadata = self.schema_metadata()?;
        Ok(state_codec::state_schema_prepared(
            &self.schema,
            &self.compiled,
            metadata.pipeline,
            metadata.node,
        ))
    }
    fn read_record(
        &mut self,
        segment: &crate::StateSegment,
        schema: &state_codec::FreshStateSchema,
        schema_bytes: u64,
        check: Check<'_>,
    ) -> Result<decoded_record::DecodedRecord> {
        let max_rows = physical_rows(
            self.budget,
            self.compiled.window_groups.len(),
            self.compiled.indicator_groups.len(),
        )?;
        let node = self.node.as_str();
        let base = self.metadata_bytes;
        let credit = &mut self.credit;
        let budget = self.budget;
        decoded_record::read(
            segment.bytes(),
            schema.as_schema(),
            max_rows,
            &mut |bytes| {
                reserve(
                    credit,
                    reader_total(base, schema_bytes, bytes)?,
                    budget,
                    node,
                )
            },
            check,
        )
    }
    fn reconstruct(
        &mut self,
        record: &RecordBatch,
        reader: u64,
        schema: u64,
        check: Check<'_>,
    ) -> Result<(DecodedRollingState, u64)> {
        let quote = self.reconstruction_quote(record, reader, schema, check)?;
        self.reserve_total(quote.additional)?;
        let state = state_v3::decode_record_checked(
            record,
            &self.schema,
            &self.compiled,
            &self.node,
            check,
        )
        .map_err(|error| state_codec::layout_five_restore_error(error, &self.node))?;
        Ok((state, quote.limit))
    }
    fn reconstruction_quote(
        &self,
        record: &RecordBatch,
        reader: u64,
        schema: u64,
        check: Check<'_>,
    ) -> Result<Reconstruction> {
        let plan = state_decode_requirements::plan(record, &self.schema, &self.compiled, check)?;
        check_rows(record, self.budget, check)?;
        let candidate = candidate_with_outer(plan.candidate_bytes, &self.node)?;
        let additional = reader_total(reader, schema, add(candidate, plan.scratch_bytes)?)?;
        let limit = candidate_limit(candidate, &self.metadata)?;
        Ok(Reconstruction { additional, limit })
    }
    fn reserve_total(&mut self, bytes: u64) -> Result<()> {
        reserve(
            &mut self.credit,
            add(self.metadata_bytes, bytes)?,
            self.budget,
            &self.node,
        )
    }
    fn schema_metadata(&self) -> Result<Metadata<'_>> {
        Ok(Metadata {
            pipeline: self
                .metadata
                .pipeline_fingerprint
                .as_deref()
                .ok_or_else(|| mismatch("identity", "missing graph"))?,
            node: self
                .metadata
                .operator_id
                .as_deref()
                .ok_or_else(|| mismatch("identity", "missing operator"))?,
            configuration: &self.compiled.configuration_hash,
            state_schema: &self.compiled.state_schema_fingerprint,
            kernel: self.compiled.kernel_plan.fingerprint(),
            numerical: self.compiled.kernel_plan.numerical_profile(),
        })
    }
}
fn checksum(
    segment: &crate::StateSegment,
    metadata: &crate::operator::rolling::RollingSnapshotMetadata,
    check: Check<'_>,
) -> Result<()> {
    let descriptor = metadata
        .segment_inventory
        .first()
        .ok_or_else(|| mismatch("segments", "missing base descriptor"))?;
    if u64::try_from(segment.bytes().len()).ok() != Some(descriptor.handle.byte_len()) {
        return Err(mismatch("encoded_bytes", "length differs"));
    }
    let mut hash = Sha256::new();
    for chunk in segment.bytes().chunks(4096) {
        check()?;
        hash.update(chunk);
    }
    let digest = hash.finalize();
    let mut hex = [0_u8; 64];
    hex::encode_to_slice(digest, &mut hex).map_err(|_| mismatch("sha256", "invalid digest"))?;
    if hex != descriptor.handle.sha256().as_bytes() {
        return Err(mismatch("sha256", "checksum differs"));
    }
    check()
}
fn physical_rows(budget: crate::StateBudget, windows: usize, indicators: usize) -> Result<u64> {
    let groups = (windows as u128)
        .checked_add(indicators as u128)
        .and_then(|value| value.checked_add(2))
        .ok_or_else(|| mismatch("rows", "count overflowed"))?;
    let rows = u128::from(budget.max_rows())
        .checked_mul(groups)
        .ok_or_else(|| mismatch("rows", "count overflowed"))?;
    u64::try_from(rows.min(u128::from(u64::MAX))).map_err(|_| mismatch("rows", "count overflowed"))
}
fn reader_backing(
    reader: Option<&decoded_record::DecodedRecord>,
    node: &str,
    check: Check<'_>,
) -> Result<Footprint> {
    let mut baseline = Footprint::default();
    if let Some(reader) = reader {
        for column in reader.record.columns() {
            check()?;
            baseline.backing(column, node, check)?;
        }
    }
    Ok(baseline)
}
fn verify_incremental(
    candidate: &Footprint,
    reader: &Footprint,
    limit: u64,
    node: &str,
) -> Result<()> {
    if candidate.incremental_bytes(reader, node)? > limit {
        return Err(mismatch(
            "candidate_bytes",
            "observed candidate exceeds its preflight requirement",
        ));
    }
    Ok(())
}

struct Reconstruction {
    additional: u64,
    limit: u64,
}
fn reader_total(base: u64, schema: u64, reader: u64) -> Result<u64> {
    add(add(base, schema)?, reader)
}
fn candidate_limit(
    candidate: u64,
    metadata: &crate::operator::rolling::RollingSnapshotMetadata,
) -> Result<u64> {
    add(candidate, outer::ids(metadata)?)
}
fn check_rows(record: &RecordBatch, budget: crate::StateBudget, check: Check<'_>) -> Result<()> {
    if state_decode_requirements::retained_rows(record, check)? > budget.max_rows() {
        return Err(mismatch(
            "max_rows",
            "restored rows exceed the configured budget",
        ));
    }
    Ok(())
}
fn verify_candidate(
    state: &crate::operator::rolling::RollingStreamState,
    candidate: &Footprint,
    reader: &Footprint,
    limit: u64,
    budget: crate::StateBudget,
    node: &str,
) -> Result<()> {
    if !budget.allows(state.charge.rows, state.charge.bytes) {
        return Err(mismatch(
            "max_state",
            "restored state exceeds the configured budget",
        ));
    }
    verify_incremental(candidate, reader, limit, node)
}

fn candidate_with_outer(payload: u64, node: &str) -> Result<u64> {
    add(payload, outer::arc_bytes(node)?)
}
