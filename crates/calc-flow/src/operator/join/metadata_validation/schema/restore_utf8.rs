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

use super::restore_segments::frame;
mod input;
mod inspector;
mod inventory;

#[cfg(test)]
mod tests;

struct RestoreUtf8Work {
    schema: Option<SchemaWork>,
    input: input::OwnedInput,
    workspace: Option<MemoryReservation>,
    geometry: frame::Geometry,
    #[cfg(test)]
    hook: Option<crate::operator::join::DecodedRowTestHook>,
    #[cfg(test)]
    schema_hook: Option<crate::operator::join::SchemaTestHook>,
}

enum RestoreUtf8Decision {
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

impl OwnedCpuWork for RestoreUtf8Work {
    type Output = RestoreUtf8Decision;

    fn control_bytes(&self) -> Result<usize> {
        super::super::inventory::caller_controls(&self.input.name)
            .and_then(|bytes| bytes.checked_add(cleanup_control_bytes::<RestoreUtf8Decision>()))
            .ok_or_else(|| crate::CalcFlowError::Internal {
                message: "restore Utf8 control overflow".into(),
            })
    }

    fn run(mut self, stop: &GatherStop) -> Result<RestoreUtf8Decision> {
        let Some(success) = self.schema.take().expect("owned schema input").run(stop)? else {
            return Ok(RestoreUtf8Decision::UseLegacy);
        };
        let Some(facts) = self.inspect_segments(&success.schemas, stop)? else {
            return Ok(RestoreUtf8Decision::Original(success));
        };
        if !self.grow_workspace(&facts, stop)? {
            return Ok(RestoreUtf8Decision::Original(success));
        }
        self.populate_snapshot(stop)?;
        self.prepare_rows(success, stop)
    }
}

impl RestoreUtf8Work {
    fn prepare_rows(
        &mut self,
        success: SchemaSuccess,
        stop: &GatherStop,
    ) -> Result<RestoreUtf8Decision> {
        let rows = self.restore_sides(&success.schemas, stop);
        let (left, right) = match rows {
            Ok(rows) => rows,
            Err(error) => {
                drop(error);
                stop.check()?;
                return Ok(RestoreUtf8Decision::Original(success));
            }
        };
        let Some(left) = self.copy_rows(left, &success.schemas, 0, stop)? else {
            drop(right);
            stop.check()?;
            return Ok(RestoreUtf8Decision::Original(success));
        };
        let Some(right) = self.copy_rows(right, &success.schemas, 1, stop)? else {
            drop(left);
            stop.check()?;
            return Ok(RestoreUtf8Decision::Original(success));
        };
        Ok(RestoreUtf8Decision::Prepared(PreparedSides {
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
    fn grow_workspace(&self, facts: &[inspector::Facts; 2], stop: &GatherStop) -> Result<bool> {
        let Some(bytes) = inventory::dynamic_workspace(&self.geometry, facts) else {
            stop.check()?;
            return Ok(false);
        };
        let paid = self
            .workspace
            .as_ref()
            .expect("initial paid workspace")
            .try_grow(bytes)
            .is_ok();
        stop.check()?;
        Ok(paid)
    }

    fn inspect_segments(
        &self,
        schemas: &OwnedExpectedSchemas,
        stop: &GatherStop,
    ) -> Result<Option<[inspector::Facts; 2]>> {
        let mut facts = [inspector::Facts::default(); 2];
        for (id, bytes) in &self.input.segments {
            stop.check()?;
            let kind = frame::kind(id).expect("checked segment identity");
            let Some(mut cursor) = frame::Cursor::new(bytes, kind.delta) else {
                return Ok(None);
            };
            if !Self::inspect_frames(
                &mut cursor,
                bytes,
                schemas.schema(kind.side),
                &self.input.key_indices[kind.side],
                &mut facts[kind.side],
                stop,
            )? {
                return Ok(None);
            }
        }
        Ok(Some(facts))
    }

    fn inspect_frames(
        cursor: &mut frame::Cursor<'_>,
        bytes: &[u8],
        expected: &datafusion::arrow::datatypes::Schema,
        keys: &[usize],
        facts: &mut inspector::Facts,
        stop: &GatherStop,
    ) -> Result<bool> {
        for _ in 0..cursor.count {
            stop.check()?;
            let Some(frame) = cursor.next() else {
                return Ok(false);
            };
            if let Some(range) = frame.ipc {
                let Some(row) = inspector::inspect(&bytes[range], expected, keys, stop)? else {
                    return Ok(false);
                };
                if facts.add(row).is_none() {
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
    ) -> Result<Option<Vec<StoredRow>>> {
        let leases = std::mem::take(&mut self.input.leases[side]);
        let mut leases = leases.into_iter();
        let mut output = Vec::with_capacity(rows.len());
        for mut row in rows {
            stop.check()?;
            let record = row.record.view();
            let lease = leases
                .next()
                .expect("one prepared guard per cumulative row");
            let registration = super::super::inventory::registration_controls()
                .expect("checked registration controls");
            let Some(bytes) =
                columnar::restored::utf8::required(&record, schemas.schema(side), registration)
            else {
                drop(record);
                stop.check()?;
                return Ok(None);
            };
            if !lease.try_fund(bytes) {
                drop(record);
                stop.check()?;
                return Ok(None);
            }
            let payload = columnar::restored::utf8::copy_row(
                &record,
                schemas.schema(side),
                row.row_id,
                row.event_time,
                lease,
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
        Ok(Some(output))
    }
}

// The original schema guards remain live until every new partial owner and credit is dropped.
struct RestoreUtf8Construction {
    input: input::OwnedInput,
    geometry: frame::Geometry,
    workspace: Option<MemoryReservation>,
    schema: SchemaConstruction,
}

impl RestoreUtf8Construction {
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
        Output = std::result::Result<ObservedTicket<RestoreUtf8Decision>, AdmissionFailure>,
    > + 'a {
        let work = RestoreUtf8Work {
            schema: Some(self.schema_work(
                #[cfg(test)]
                metadata_hook,
                #[cfg(test)]
                schema_hook.clone(),
            )),
            input: self.input.take(),
            workspace: self.workspace.take(),
            geometry: self.geometry,
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
        deltas: usize,
        job: &crate::StreamJobContext,
    ) -> Result<Option<Self>> {
        let Some(schema) = SchemaConstruction::new(operator, metadata, job)? else {
            return Ok(None);
        };
        let Some(bounds) = inventory::required(
            geometry,
            deltas,
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
        let original = schema
            .metadata
            .credit
            .as_ref()
            .expect("paid metadata input");
        let construction = Self {
            input: input::OwnedInput::new(original.new_empty()),
            geometry: *geometry,
            workspace: Some(original.new_empty()),
            schema,
        };
        if !construction.fund_components(&bounds) {
            job.check_cancelled()?;
            return Ok(None);
        }
        Ok(Some(construction))
    }

    fn fund_components(&self, bounds: &inventory::Bounds) -> bool {
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
        _operator: &crate::operator::join::StreamJoinOperator,
        job: &crate::StreamJobContext,
    ) -> Result<bool> {
        let geometry = &self.geometry;
        for side in 0..2 {
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
                    .input
                    .credit
                    .as_ref()
                    .expect("paid input controls")
                    .new_empty();
                self.input.leases[side].push(columnar::restored::ResidentLease::new(credit, guard));
            }
        }
        Ok(true)
    }
}

impl crate::operator::join::StreamJoinOperator {
    pub(in crate::operator::join) async fn try_restore_owned_utf8(
        &mut self,
        snapshot: &crate::OperatorStateSnapshot,
        job: &crate::StreamJobContext,
        task: Option<TaskId>,
    ) -> Result<bool> {
        let Some(mut construction) = self.utf8_construction(snapshot, job).await? else {
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
        self.finish_utf8_admission(
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

    async fn utf8_construction(
        &self,
        snapshot: &crate::OperatorStateSnapshot,
        job: &crate::StreamJobContext,
    ) -> Result<Option<RestoreUtf8Construction>> {
        if !inventory::eligible(self) {
            return Ok(None);
        }
        let Some(geometry) = frame::scan_all(snapshot, job).await? else {
            return Ok(None);
        };
        let Some(deltas) = frame::delta_count(snapshot, &geometry) else {
            return Ok(None);
        };
        let Some(metadata) = self.metadata_construction(snapshot, job)? else {
            return Ok(None);
        };
        RestoreUtf8Construction::new(self, metadata, &geometry, deltas, job)
    }

    async fn finish_utf8_admission(
        &mut self,
        admission: std::result::Result<ObservedTicket<RestoreUtf8Decision>, AdmissionFailure>,
        snapshot: &crate::OperatorStateSnapshot,
        job: &crate::StreamJobContext,
        stop: &GatherStop,
    ) -> Result<bool> {
        match admission {
            Ok(ticket) => self.finish_utf8_ticket(ticket, snapshot, job, stop).await,
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

    async fn finish_utf8_ticket(
        &mut self,
        ticket: ObservedTicket<RestoreUtf8Decision>,
        snapshot: &crate::OperatorStateSnapshot,
        job: &crate::StreamJobContext,
        stop: &GatherStop,
    ) -> Result<bool> {
        let output = ticket.finish().await?;
        let mut fallback = None;
        output.install(|decision| {
            fallback = self.install_utf8_decision(decision, snapshot, job)?;
            Ok(())
        })?;
        self.wait_metadata_cleanup(stop, job).await?;
        if let Some(decision) = fallback {
            self.install_utf8_fallback(decision, snapshot, job)?;
        }
        Ok(true)
    }

    fn install_utf8_decision(
        &mut self,
        decision: RestoreUtf8Decision,
        snapshot: &crate::OperatorStateSnapshot,
        job: &crate::StreamJobContext,
    ) -> Result<Option<RestoreUtf8Decision>> {
        match decision {
            RestoreUtf8Decision::UseLegacy | RestoreUtf8Decision::Original(_) => Ok(Some(decision)),
            RestoreUtf8Decision::Prepared(mut prepared) => self
                .install_restored_rows(
                    snapshot,
                    prepared.metadata,
                    std::mem::take(&mut prepared.left),
                    std::mem::take(&mut prepared.right),
                    &|| job.check_cancelled(),
                )
                .map(|()| None),
        }
    }

    fn install_utf8_fallback(
        &mut self,
        decision: RestoreUtf8Decision,
        snapshot: &crate::OperatorStateSnapshot,
        job: &crate::StreamJobContext,
    ) -> Result<()> {
        job.check_cancelled()?;
        match decision {
            RestoreUtf8Decision::UseLegacy => self.restore_metadata_legacy(snapshot, job),
            RestoreUtf8Decision::Original(success) => self.install_restored_metadata_with_schemas(
                snapshot,
                success.metadata,
                Some(&success.schemas),
                &|| job.check_cancelled(),
            ),
            RestoreUtf8Decision::Prepared(_) => {
                unreachable!("prepared output installs with its workspace")
            }
        }
    }
}
