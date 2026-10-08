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

mod frame;
mod input;
use super::restore_bases::inspector;
mod inventory;

#[cfg(test)]
mod tests;

struct RestoreSegmentsWork {
    schema: Option<SchemaWork>,
    input: input::OwnedInput,
    workspace: Option<MemoryReservation>,
    #[cfg(test)]
    hook: Option<crate::operator::join::DecodedRowTestHook>,
    #[cfg(test)]
    schema_hook: Option<crate::operator::join::SchemaTestHook>,
}

enum RestoreSegmentsDecision {
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

impl OwnedCpuWork for RestoreSegmentsWork {
    type Output = RestoreSegmentsDecision;

    fn control_bytes(&self) -> Result<usize> {
        super::super::inventory::caller_controls(&self.input.name)
            .and_then(|bytes| bytes.checked_add(cleanup_control_bytes::<RestoreSegmentsDecision>()))
            .ok_or_else(|| crate::CalcFlowError::Internal {
                message: "restore segments control overflow".into(),
            })
    }

    fn run(mut self, stop: &GatherStop) -> Result<RestoreSegmentsDecision> {
        let Some(success) = self.schema.take().expect("owned schema input").run(stop)? else {
            return Ok(RestoreSegmentsDecision::UseLegacy);
        };
        if !self.inspect_segments(&success.schemas, stop)? {
            return Ok(RestoreSegmentsDecision::Original(success));
        }
        self.populate_snapshot(stop)?;
        self.prepare_rows(success, stop)
    }
}

impl RestoreSegmentsWork {
    fn prepare_rows(
        &mut self,
        success: SchemaSuccess,
        stop: &GatherStop,
    ) -> Result<RestoreSegmentsDecision> {
        let rows = self.restore_sides(&success.schemas, stop);
        let (left, right) = match rows {
            Ok(rows) => rows,
            Err(error) => {
                drop(error);
                stop.check()?;
                return Ok(RestoreSegmentsDecision::Original(success));
            }
        };
        let left = self.copy_rows(left, &success.schemas, 0, stop)?;
        let right = self.copy_rows(right, &success.schemas, 1, stop)?;
        Ok(RestoreSegmentsDecision::Prepared(PreparedSides {
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
    fn inspect_segments(&self, schemas: &OwnedExpectedSchemas, stop: &GatherStop) -> Result<bool> {
        for (id, bytes) in &self.input.segments {
            stop.check()?;
            let kind = frame::kind(id).expect("checked segment identity");
            let Some(mut cursor) = frame::Cursor::new(bytes, kind.delta) else {
                return Ok(false);
            };
            if !Self::inspect_frames(&mut cursor, bytes, schemas.schema(kind.side), stop)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn inspect_frames(
        cursor: &mut frame::Cursor<'_>,
        bytes: &[u8],
        expected: &datafusion::arrow::datatypes::Schema,
        stop: &GatherStop,
    ) -> Result<bool> {
        for _ in 0..cursor.count {
            stop.check()?;
            let Some(frame) = cursor.next() else {
                return Ok(false);
            };
            if let Some(range) = frame.ipc {
                if inspector::inspect(&bytes[range], expected).is_none() {
                    return Ok(false);
                }
            }
        }
        Ok(cursor.finished())
    }

    fn populate_snapshot(&mut self, stop: &GatherStop) -> Result<()> {
        for (id, bytes) in self.input.segments.drain(..) {
            stop.check()?;
            self.input
                .snapshot
                .segments
                .insert(id, StateSegment::new(bytes));
        }
        stop.check()
    }

    fn restore_sides(
        &self,
        schemas: &OwnedExpectedSchemas,
        stop: &GatherStop,
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
        crate::operator::join::restore_sides_from_segments_checked(
            &self.input.snapshot,
            schema(0),
            schema(1),
            &self.input.key_indices[0],
            &self.input.key_indices[1],
            &self.input.name,
            &|| stop.check(),
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
                    .expect("one independently prepaid lease per cumulative decoded row"),
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

// The original schema guards remain live until every new partial owner and credit is dropped.
struct RestoreSegmentsConstruction {
    input: input::OwnedInput,
    geometry: frame::Geometry,
    resident: Option<MemoryReservation>,
    workspace: Option<MemoryReservation>,
    schema: SchemaConstruction,
}

impl RestoreSegmentsConstruction {
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
        Output = std::result::Result<ObservedTicket<RestoreSegmentsDecision>, AdmissionFailure>,
    > + 'a {
        let work = RestoreSegmentsWork {
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
        metadata: super::super::Construction,
        geometry: &frame::Geometry,
        job: &crate::StreamJobContext,
    ) -> Result<Option<Self>> {
        let Some(schema) = SchemaConstruction::new(operator, metadata, job)? else {
            return Ok(None);
        };
        let Some(bounds) = inventory::required(
            geometry,
            [operator.input_schema(0), operator.input_schema(1)],
            [
                &operator.compiled.left_key_indices,
                &operator.compiled.right_key_indices,
            ],
            &operator.name,
        ) else {
            return Ok(None);
        };
        Self::fund(schema, geometry, bounds, job)
    }

    fn fund(
        schema: SchemaConstruction,
        geometry: &frame::Geometry,
        bounds: inventory::Bounds,
        job: &crate::StreamJobContext,
    ) -> Result<Option<Self>> {
        let Some(total) = resident_total(geometry, bounds.resident) else {
            return Ok(None);
        };
        let original = schema
            .metadata
            .credit
            .as_ref()
            .expect("paid metadata input");
        let construction = Self {
            input: input::OwnedInput::new(original.new_empty()),
            geometry: *geometry,
            resident: Some(original.new_empty()),
            workspace: Some(original.new_empty()),
            schema,
        };
        if !construction.fund_components(&bounds, total) {
            job.check_cancelled()?;
            return Ok(None);
        }
        Ok(Some(construction))
    }

    fn fund_components(&self, bounds: &inventory::Bounds, resident: usize) -> bool {
        self.input
            .credit
            .as_ref()
            .expect("input credit")
            .try_grow(bounds.input)
            .is_ok()
            && self
                .workspace
                .as_ref()
                .expect("workspace credit")
                .try_grow(bounds.workspace)
                .is_ok()
            && self
                .resident
                .as_ref()
                .expect("resident credit")
                .try_grow(resident)
                .is_ok()
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
        self.input.copy_segments(snapshot, job).await?;
        self.prepare_leases(operator, job).await
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
        job: &crate::StreamJobContext,
    ) -> Result<bool> {
        let geometry = &self.geometry;
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

fn resident_total(geometry: &frame::Geometry, resident: [usize; 2]) -> Option<usize> {
    geometry.rows[0]
        .checked_mul(resident[0])?
        .checked_add(geometry.rows[1].checked_mul(resident[1])?)
}

impl crate::operator::join::StreamJoinOperator {
    pub(in crate::operator::join) async fn try_restore_owned_segments(
        &mut self,
        snapshot: &crate::OperatorStateSnapshot,
        job: &crate::StreamJobContext,
        task: Option<TaskId>,
    ) -> Result<bool> {
        let Some(mut construction) = self.segments_construction(snapshot, job).await? else {
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
        self.finish_segments_admission(
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

    async fn segments_construction(
        &self,
        snapshot: &crate::OperatorStateSnapshot,
        job: &crate::StreamJobContext,
    ) -> Result<Option<RestoreSegmentsConstruction>> {
        let Some(geometry) = frame::scan(snapshot, job).await? else {
            return Ok(None);
        };
        let Some(metadata) = self.metadata_construction(snapshot, job)? else {
            return Ok(None);
        };
        RestoreSegmentsConstruction::new(self, metadata, &geometry, job)
    }

    async fn finish_segments_admission(
        &mut self,
        admission: std::result::Result<ObservedTicket<RestoreSegmentsDecision>, AdmissionFailure>,
        snapshot: &crate::OperatorStateSnapshot,
        job: &crate::StreamJobContext,
        stop: &GatherStop,
    ) -> Result<bool> {
        match admission {
            Ok(ticket) => {
                self.finish_segments_ticket(ticket, snapshot, job, stop)
                    .await
            }
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

    async fn finish_segments_ticket(
        &mut self,
        ticket: ObservedTicket<RestoreSegmentsDecision>,
        snapshot: &crate::OperatorStateSnapshot,
        job: &crate::StreamJobContext,
        stop: &GatherStop,
    ) -> Result<bool> {
        let output = ticket.finish().await?;
        output.install(|decision| self.install_segments_decision(decision, snapshot, job))?;
        self.wait_metadata_cleanup(stop, job).await?;
        Ok(true)
    }

    fn install_segments_decision(
        &mut self,
        decision: RestoreSegmentsDecision,
        snapshot: &crate::OperatorStateSnapshot,
        job: &crate::StreamJobContext,
    ) -> Result<()> {
        match decision {
            RestoreSegmentsDecision::UseLegacy => self.restore_metadata_legacy(snapshot, job),
            RestoreSegmentsDecision::Original(success) => self
                .install_restored_metadata_with_schemas(
                    snapshot,
                    success.metadata,
                    Some(&success.schemas),
                    &|| job.check_cancelled(),
                ),
            RestoreSegmentsDecision::Prepared(mut prepared) => self.install_restored_rows(
                snapshot,
                prepared.metadata,
                std::mem::take(&mut prepared.left),
                std::mem::take(&mut prepared.right),
                &|| job.check_cancelled(),
            ),
        }
    }
}
