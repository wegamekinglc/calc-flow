use super::super::ValidatedMetadata;
use super::{OwnedExpectedSchemas, SchemaConstruction, SchemaSuccess, SchemaWork};
use crate::operator::join::{RestoreSchema, StoredRow, columnar};
use crate::runtime::streaming::gather_work::{
    AdmissionFailure, AttemptCleanup, GatherOperatorId, GatherStop, ObservedTicket, OwnedCpuWork,
    TaskId, cleanup_control_bytes,
};
use crate::{Result, StateSegment};
use datafusion::execution::memory_pool::MemoryReservation;
use std::sync::Arc;

mod input;
mod inspector;
mod inventory;

#[cfg(test)]
mod tests;

struct RestoreBasesWork {
    schema: Option<SchemaWork>,
    input: input::OwnedInput,
    workspace: Option<MemoryReservation>,
    #[cfg(test)]
    hook: Option<crate::operator::join::DecodedRowTestHook>,
    #[cfg(test)]
    schema_hook: Option<crate::operator::join::SchemaTestHook>,
}

enum RestoreBasesDecision {
    UseLegacy,
    Original(SchemaSuccess),
    Prepared(PreparedSides),
}

// Both row vectors and descriptors are released before the complete installation workspace.
struct PreparedSides {
    left: Vec<StoredRow>,
    right: Vec<StoredRow>,
    _schemas: OwnedExpectedSchemas,
    metadata: ValidatedMetadata,
    _workspace: MemoryReservation,
}

impl OwnedCpuWork for RestoreBasesWork {
    type Output = RestoreBasesDecision;

    fn control_bytes(&self) -> Result<usize> {
        super::super::inventory::caller_controls(&self.input.name)
            .and_then(|bytes| bytes.checked_add(cleanup_control_bytes::<RestoreBasesDecision>()))
            .ok_or_else(|| crate::CalcFlowError::Internal {
                message: "restore bases control overflow".into(),
            })
    }

    fn run(mut self, stop: &GatherStop) -> Result<RestoreBasesDecision> {
        let Some(success) = self.schema.take().expect("owned schema input").run(stop)? else {
            return Ok(RestoreBasesDecision::UseLegacy);
        };
        if !self.inspect_bases(&success.schemas, stop)? {
            return Ok(RestoreBasesDecision::Original(success));
        }
        self.populate_snapshot();
        let rows = self.restore_sides(&success.schemas);
        let (left, right) = match rows {
            Ok(rows) => rows,
            Err(error) => {
                drop(error);
                return Ok(RestoreBasesDecision::Original(success));
            }
        };
        let left = self.copy_rows(left, &success.schemas, 0, stop)?;
        let right = self.copy_rows(right, &success.schemas, 1, stop)?;
        Ok(RestoreBasesDecision::Prepared(PreparedSides {
            left,
            right,
            _schemas: success.schemas,
            metadata: success.metadata,
            _workspace: self
                .workspace
                .take()
                .expect("complete paid installation workspace"),
        }))
    }
}

impl RestoreBasesWork {
    fn inspect_bases(&self, schemas: &OwnedExpectedSchemas, stop: &GatherStop) -> Result<bool> {
        for (side, bytes) in self.input.bases.iter().enumerate() {
            let Some(count) = input::geometry_side(bytes) else {
                return Ok(false);
            };
            let mut offset = 16;
            for _ in 0..count {
                stop.check()?;
                let Some(range) = input::next_row(bytes, &mut offset) else {
                    return Ok(false);
                };
                if inspector::inspect(&bytes[range], schemas.schema(side)).is_none() {
                    return Ok(false);
                }
            }
            if offset != bytes.len() {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn populate_snapshot(&mut self) {
        for (bytes, side) in self.input.bases.iter_mut().zip(["left-base", "right-base"]) {
            self.input
                .snapshot
                .segments
                .insert(String::from(side), StateSegment::new(std::mem::take(bytes)));
        }
    }

    fn restore_sides(
        &self,
        schemas: &OwnedExpectedSchemas,
    ) -> Result<(Vec<StoredRow>, Vec<StoredRow>)> {
        let schema = |side| RestoreSchema {
            schema: schemas.schema(side),
            #[cfg(test)]
            hook: self.schema_hook.as_ref(),
            #[cfg(test)]
            credit: Some(schemas.credit()),
            #[cfg(test)]
            decoded_row_hook: self.hook.as_ref(),
            #[cfg(test)]
            decoded_row_credit: self.workspace.as_ref(),
            #[cfg(test)]
            decoded_owned_work: true,
        };
        crate::operator::join::restore_sides_from_segments(
            &self.input.snapshot,
            schema(0),
            schema(1),
            &self.input.key_indices[0],
            &self.input.key_indices[1],
            &self.input.name,
        )
    }

    fn copy_rows(
        &mut self,
        rows: Vec<StoredRow>,
        schemas: &OwnedExpectedSchemas,
        side: usize,
        stop: &GatherStop,
    ) -> Result<Vec<StoredRow>> {
        let leases = std::mem::take(&mut self.input.leases[side]);
        let mut leases = leases.into_iter();
        let mut output = Vec::with_capacity(rows.len());
        for mut row in rows {
            stop.check()?;
            let record = row.record.view();
            let payload = columnar::restored::copy_row(
                &record,
                schemas.schema(side),
                row.row_id,
                row.event_time,
                leases
                    .next()
                    .expect("one independently prepaid lease per base row"),
            );
            drop(record);
            row.record = payload;
            #[cfg(test)]
            if let Some(hook) = &self.hook {
                hook(
                    self.workspace.as_ref(),
                    row.record.funded_owner(),
                    true,
                    true,
                );
            }
            output.push(row);
        }
        Ok(output)
    }
}

struct RestoreBasesConstruction {
    schema: SchemaConstruction,
    input: input::OwnedInput,
    resident: Option<MemoryReservation>,
    workspace: Option<MemoryReservation>,
}

impl RestoreBasesConstruction {
    fn schema_work(
        &mut self,
        #[cfg(test)] metadata_hook: Option<crate::operator::join::MetadataTestHook>,
        #[cfg(test)] schema_hook: Option<crate::operator::join::SchemaTestHook>,
    ) -> SchemaWork {
        SchemaWork {
            metadata: super::super::MetadataWork {
                snapshot: std::mem::take(&mut self.schema.metadata.snapshot),
                expected: self
                    .schema
                    .metadata
                    .expected
                    .take()
                    .expect("completed metadata construction"),
                name: std::mem::take(&mut self.schema.metadata.name),
                #[cfg(test)]
                hook: metadata_hook,
            },
            plans: std::mem::take(&mut self.schema.plans),
            _input_credit: self.schema.input_credit.take().expect("paid schema input"),
            funding: self.schema.funding.take().expect("paid descriptor output"),
            #[cfg(test)]
            hook: schema_hook,
        }
    }

    fn submit<'a>(
        &'a mut self,
        observer: &'a mut Option<AttemptCleanup>,
        #[cfg(test)] metadata_hook: Option<crate::operator::join::MetadataTestHook>,
        #[cfg(test)] schema_hook: Option<crate::operator::join::SchemaTestHook>,
        #[cfg(test)] decoded_hook: Option<crate::operator::join::DecodedRowTestHook>,
    ) -> impl Future<
        Output = std::result::Result<ObservedTicket<RestoreBasesDecision>, AdmissionFailure>,
    > + 'a {
        let work = RestoreBasesWork {
            schema: Some(self.schema_work(
                #[cfg(test)]
                metadata_hook,
                #[cfg(test)]
                schema_hook.clone(),
            )),
            input: self.input.take(),
            workspace: self.workspace.take(),
            #[cfg(test)]
            hook: decoded_hook,
            #[cfg(test)]
            schema_hook,
        };
        self.schema
            .metadata
            .control
            .scope
            .as_ref()
            .expect("prepared metadata scope")
            .submit_observed_work(
                work,
                self.schema
                    .metadata
                    .credit
                    .take()
                    .expect("paid metadata input"),
                self.schema
                    .metadata
                    .control
                    .stop
                    .as_ref()
                    .expect("paid metadata stop")
                    .clone(),
                self.schema
                    .metadata
                    .retirement
                    .take()
                    .expect("registered metadata work"),
                observer,
            )
    }

    fn new(
        operator: &crate::operator::join::StreamJoinOperator,
        snapshot: &crate::OperatorStateSnapshot,
        metadata: super::super::Construction,
        job: &crate::StreamJobContext,
    ) -> Result<Option<Self>> {
        let Some(geometry) = input::geometry(snapshot) else {
            return Ok(None);
        };
        let Some(schema) = SchemaConstruction::new(operator, metadata, job)? else {
            return Ok(None);
        };
        let Some(bounds) = inventory::required(
            &geometry,
            [operator.input_schema(0), operator.input_schema(1)],
            [
                &operator.compiled.left_key_indices,
                &operator.compiled.right_key_indices,
            ],
            &operator.name,
        ) else {
            return Ok(None);
        };
        Self::fund(schema, &geometry, bounds, job)
    }

    fn fund(
        schema: SchemaConstruction,
        geometry: &input::Geometry,
        bounds: inventory::Bounds,
        job: &crate::StreamJobContext,
    ) -> Result<Option<Self>> {
        let original = schema
            .metadata
            .credit
            .as_ref()
            .expect("paid metadata input");
        let input = original.new_empty();
        let workspace = original.new_empty();
        let resident = original.new_empty();
        let Some(total) = geometry.rows[0]
            .checked_mul(bounds.resident[0])
            .and_then(|left| {
                geometry.rows[1]
                    .checked_mul(bounds.resident[1])
                    .and_then(|right| left.checked_add(right))
            })
        else {
            return Ok(None);
        };
        if input.try_grow(bounds.input).is_err()
            || workspace.try_grow(bounds.workspace).is_err()
            || resident.try_grow(total).is_err()
        {
            job.check_cancelled()?;
            return Ok(None);
        }
        Ok(Some(Self {
            schema,
            input: input::OwnedInput::new(input),
            resident: Some(resident),
            workspace: Some(workspace),
        }))
    }

    async fn copy(
        &mut self,
        operator: &crate::operator::join::StreamJoinOperator,
        snapshot: &crate::OperatorStateSnapshot,
        job: &crate::StreamJobContext,
    ) -> Result<bool> {
        self.schema.copy(operator, snapshot, job).await?;
        super::super::copy_boundary(job).await?;
        self.input.name = String::from(operator.name.as_str());
        self.copy_indices(operator, job).await?;
        self.input.copy_bases(snapshot, job).await?;
        self.prepare_leases(operator, snapshot, job).await
    }

    async fn copy_indices(
        &mut self,
        operator: &crate::operator::join::StreamJoinOperator,
        job: &crate::StreamJobContext,
    ) -> Result<()> {
        let indices = [
            &operator.compiled.left_key_indices,
            &operator.compiled.right_key_indices,
        ];
        for (output, source) in self.input.key_indices.iter_mut().zip(indices) {
            *output = Vec::with_capacity(source.len());
            for part in source.chunks(64) {
                super::super::copy_boundary(job).await?;
                output.extend_from_slice(part);
            }
        }
        Ok(())
    }

    async fn prepare_leases(
        &mut self,
        operator: &crate::operator::join::StreamJoinOperator,
        snapshot: &crate::OperatorStateSnapshot,
        job: &crate::StreamJobContext,
    ) -> Result<bool> {
        let geometry = input::geometry(snapshot).expect("unchanged borrowed snapshot");
        let registration = super::super::inventory::registration_controls()
            .expect("checked registration controls");
        for side in 0..2 {
            let bytes = columnar::restored::required(operator.input_schema(side), registration)
                .expect("checked resident constructor");
            self.input.leases[side] = Vec::with_capacity(geometry.rows[side]);
            for _ in 0..geometry.rows[side] {
                super::super::copy_boundary(job).await?;
                let guard = match job.gather_owner().retain_retirement() {
                    Ok(guard) => guard,
                    Err(crate::CalcFlowError::Cancelled { .. }) => {
                        job.check_cancelled()?;
                        return Ok(false);
                    }
                    Err(error) => return Err(error),
                };
                let credit = self
                    .resident
                    .as_ref()
                    .expect("prepaid resident group")
                    .split(bytes);
                self.input.leases[side].push(columnar::restored::ResidentLease::new(credit, guard));
            }
        }
        Ok(true)
    }
}

impl crate::operator::join::StreamJoinOperator {
    pub(in crate::operator::join) async fn try_restore_owned_bases(
        &mut self,
        snapshot: &crate::OperatorStateSnapshot,
        job: &crate::StreamJobContext,
        task: Option<TaskId>,
    ) -> Result<bool> {
        let Some(mut construction) = self.bases_construction(snapshot, job)? else {
            return Ok(false);
        };
        #[cfg(test)]
        if let Some(hook) = &self.metadata_test_hook {
            hook(construction.schema.metadata.credit.as_ref(), false);
        }
        if !construction.copy(self, snapshot, job).await? {
            return Ok(false);
        }
        construction.schema.metadata.control.scope = match job
            .gather_owner()
            .client(GatherOperatorId::new(Arc::from(self.name.as_str())).with_task(task))
            .scope()
        {
            Ok(scope) => Some(scope),
            Err(crate::CalcFlowError::Cancelled { .. }) => {
                job.check_cancelled()?;
                return Ok(false);
            }
            Err(error) => return Err(error),
        };
        let admission = construction
            .submit(
                &mut self.compaction_cleanup,
                #[cfg(test)]
                self.metadata_test_hook.clone(),
                #[cfg(test)]
                self.schema_test_hook.clone(),
                #[cfg(test)]
                self.decoded_row_test_hook.clone(),
            )
            .await;
        self.finish_bases_admission(
            admission,
            snapshot,
            job,
            construction
                .schema
                .metadata
                .control
                .stop
                .as_ref()
                .expect("paid metadata stop"),
        )
        .await
    }

    fn bases_construction(
        &self,
        snapshot: &crate::OperatorStateSnapshot,
        job: &crate::StreamJobContext,
    ) -> Result<Option<RestoreBasesConstruction>> {
        if input::geometry(snapshot).is_none() {
            return Ok(None);
        }
        let Some(metadata) = self.metadata_construction(snapshot, job)? else {
            return Ok(None);
        };
        RestoreBasesConstruction::new(self, snapshot, metadata, job)
    }

    async fn finish_bases_admission(
        &mut self,
        admission: std::result::Result<ObservedTicket<RestoreBasesDecision>, AdmissionFailure>,
        snapshot: &crate::OperatorStateSnapshot,
        job: &crate::StreamJobContext,
        stop: &GatherStop,
    ) -> Result<bool> {
        match admission {
            Ok(ticket) => self.finish_bases_ticket(ticket, snapshot, job, stop).await,
            Err(AdmissionFailure::Budget {
                stage: "attempt", ..
            }) => {
                job.check_cancelled()?;
                Ok(false)
            }
            Err(
                AdmissionFailure::Budget { .. }
                | AdmissionFailure::Runtime(crate::CalcFlowError::Cancelled { .. }),
            ) => self.finish_schema_refusal(snapshot, job, stop).await,
            Err(AdmissionFailure::Runtime(error)) => Err(error),
        }
    }

    async fn finish_bases_ticket(
        &mut self,
        ticket: ObservedTicket<RestoreBasesDecision>,
        snapshot: &crate::OperatorStateSnapshot,
        job: &crate::StreamJobContext,
        stop: &GatherStop,
    ) -> Result<bool> {
        let output = ticket.finish().await?;
        output.install(|decision| self.install_bases_decision(decision, snapshot, job))?;
        self.wait_metadata_cleanup(stop, job).await?;
        Ok(true)
    }

    fn install_bases_decision(
        &mut self,
        decision: RestoreBasesDecision,
        snapshot: &crate::OperatorStateSnapshot,
        job: &crate::StreamJobContext,
    ) -> Result<()> {
        match decision {
            RestoreBasesDecision::UseLegacy => self.restore_metadata_legacy(snapshot, job),
            RestoreBasesDecision::Original(success) => self.install_restored_metadata_with_schemas(
                snapshot,
                success.metadata,
                Some(&success.schemas),
                &|| job.check_cancelled(),
            ),
            RestoreBasesDecision::Prepared(mut prepared) => self.install_restored_rows(
                snapshot,
                prepared.metadata,
                std::mem::take(&mut prepared.left),
                std::mem::take(&mut prepared.right),
                &|| job.check_cancelled(),
            ),
        }
    }
}
