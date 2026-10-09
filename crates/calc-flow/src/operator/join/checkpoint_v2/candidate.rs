use std::{collections::BTreeMap, sync::Arc};

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::execution::memory_pool::MemoryReservation;

use super::super::{
    DeltaTracking, ExpirationIndex, JoinSide, OperatorStateSnapshot, StoredRow, StreamJoinSpec,
    StreamJoinState, columnar, metadata_validation::ValidatedMetadata,
    restored_retained_metrics_match,
};
use super::budget::{self, ContainerFunding};
use super::frame::Payload;
use super::geometry::{Geometry, checked, invalid};
use super::history::{self, SideHistory};
use super::inventory::{self, Inventory};
use super::ipc::accounting::{arc, reserve, sum};
use super::payload::{Funding, OwnedPayload};
use super::{ipc, metadata, row};
use crate::runtime::streaming::gather_work::RetirementGuard;
use crate::{Epoch, Result};

pub(super) struct Config<'a> {
    pub(super) name: &'a str,
    pub(super) spec: &'a StreamJoinSpec,
    pub(super) schemas: [SchemaRef; 2],
    pub(super) keys: [&'a [usize]; 2],
    pub(super) time_indices: [usize; 2],
    #[cfg(test)]
    pub(super) metadata_hook: Option<&'a super::super::MetadataTestHook>,
    #[cfg(test)]
    pub(super) decoded_hook: Option<&'a super::super::DecodedRowTestHook>,
    #[cfg(test)]
    pub(super) owned_work: bool,
}

// The candidate data is destroyed before either resident or temporary credit.
pub(super) struct Prepared {
    pub(super) state: StreamJoinState,
    pub(super) containers: Arc<ContainerFunding>,
    pub(super) workspace: MemoryReservation,
}

struct Decoded<'a> {
    frame: Payload<'a>,
    owner: Arc<OwnedPayload>,
}

pub(super) struct Decoder<'a, 'c> {
    config: &'c Config<'c>,
    snapshot: &'a OperatorStateSnapshot,
    workspace: MemoryReservation,
    containers: Arc<ContainerFunding>,
    verifier: ipc::VerifierCredit,
    retirements: Vec<RetirementGuard>,
    check: &'c dyn Fn() -> Result<()>,
}

impl<'a, 'c> Decoder<'a, 'c> {
    pub(super) fn new(
        snapshot: &'a OperatorStateSnapshot,
        config: &'c Config<'c>,
        credit: &MemoryReservation,
        mut retirements: Vec<RetirementGuard>,
        check: &'c dyn Fn() -> Result<()>,
    ) -> Result<Self> {
        let workspace = credit.new_empty();
        reserve(
            &workspace,
            budget::metadata_bytes(&snapshot.inline_metadata, config.spec)?,
        )?;
        let containers = ContainerFunding::new(credit.new_empty(), retirements.pop())?;
        let verifier = ipc::VerifierCredit::new(credit.new_empty())?;
        check()?;
        Ok(Self {
            config,
            snapshot,
            workspace,
            containers,
            verifier,
            retirements,
            check,
        })
    }

    pub(super) fn decode(mut self) -> Result<Prepared> {
        let inventory = Inventory::decode(self.snapshot, self.check)?;
        (self.check)()?;
        let geometry = Geometry::inspect(self.snapshot, &inventory, self.check)?;
        let metadata = self.decode_metadata()?;
        let histories = history::fold(
            self.snapshot,
            &inventory,
            &metadata,
            &geometry,
            &self.workspace,
            self.check,
        )?;
        let payloads = self.decode_payloads(geometry.payloads)?;
        let rows = self.validate_histories(&histories, &payloads)?;
        drop(histories);
        drop(payloads);
        self.prepare(metadata, rows)
    }

    fn decode_metadata(&self) -> Result<ValidatedMetadata> {
        #[cfg(test)]
        if let Some(hook) = self.config.metadata_hook {
            hook(Some(&self.workspace), false);
            hook(Some(&self.workspace), true);
        }
        let metadata = checked(metadata::decode(
            self.snapshot.inline_metadata.clone(),
            self.config.spec,
        ))?;
        (self.check)()?;
        Ok(metadata)
    }

    fn decode_payloads(
        &mut self,
        count: usize,
    ) -> Result<BTreeMap<(JoinSide, [u8; 32]), Decoded<'a>>> {
        reserve(
            &self.workspace,
            budget::tree::<(JoinSide, [u8; 32]), Decoded<'_>>(count)?,
        )?;
        let mut payloads = BTreeMap::new();
        for (name, segment) in &self.snapshot.segments {
            (self.check)()?;
            let (side, suffix) = checked(inventory::split_side(name))?;
            let Some(digest) = suffix.strip_prefix("payload-") else {
                continue;
            };
            let digest = checked(inventory::digest(digest))?;
            let payload = self.decode_framed_payload(side, segment.bytes())?;
            payloads.insert((side, digest), payload);
        }
        Ok(payloads)
    }

    fn decode_framed_payload(&mut self, side: JoinSide, bytes: &'a [u8]) -> Result<Decoded<'a>> {
        let frame = checked(Payload::decode(bytes, side))?;
        let owner = self.decode_payload(side, &frame)?;
        Ok(Decoded { frame, owner })
    }

    fn decode_payload(&mut self, side: JoinSide, frame: &Payload<'_>) -> Result<Arc<OwnedPayload>> {
        let credit = self.containers.credit.new_empty();
        reserve(&credit, budget::payload_seed()?)?;
        let funding = Funding::new(self.config.schemas.clone(), credit, self.retirements.pop());
        let index = side_index(side);
        #[cfg(test)]
        if let Some(hook) = self.config.decoded_hook {
            hook(Some(&self.workspace), None, false, self.config.owned_work);
        }
        let owner = ipc::decode(
            frame.ipc,
            Arc::clone(&self.config.schemas[index]),
            frame.rows,
            &ipc::Admission {
                workspace: &self.workspace,
                resident: &funding,
                verifier: &self.verifier,
                check: self.check,
            },
        )?;
        #[cfg(test)]
        if let Some(hook) = self.config.decoded_hook {
            hook(
                Some(&self.workspace),
                Some(owner.funded_owner()),
                true,
                self.config.owned_work,
            );
        }
        Ok(owner)
    }

    fn validate_histories(
        &self,
        histories: &[SideHistory<'_>; 2],
        payloads: &BTreeMap<(JoinSide, [u8; 32]), Decoded<'_>>,
    ) -> Result<[Vec<StoredRow>; 2]> {
        let count = ipc::accounting::add(histories[0].live.len(), histories[1].live.len())?;
        self.containers.grow(budget::containers(count)?)?;
        Ok([
            self.validate_side(JoinSide::Left, &histories[0], payloads)?,
            self.validate_side(JoinSide::Right, &histories[1], payloads)?,
        ])
    }

    fn validate_side(
        &self,
        side: JoinSide,
        history: &SideHistory<'_>,
        payloads: &BTreeMap<(JoinSide, [u8; 32]), Decoded<'_>>,
    ) -> Result<Vec<StoredRow>> {
        let context = self.row_context(side);
        let mut rows = Vec::with_capacity(history.live.len());
        for locator in history.historical.values() {
            (self.check)()?;
            let (payload, index) = located_payload(payloads, side, locator)?;
            let validated = row::validate(payload.owner.record(), index, locator, &context)?;
            if history.live.contains_key(&locator.row_id) {
                rows.push(self.stored_row(payload, index, locator, validated)?);
            }
        }
        Ok(rows)
    }

    fn row_context(&self, side: JoinSide) -> row::ValidationContext<'_> {
        let index = side_index(side);
        row::ValidationContext {
            key_indices: self.config.keys[index],
            event_time_index: self.config.time_indices[index],
            name: self.config.name,
            side,
            workspace: &self.workspace,
            check: self.check,
        }
    }

    fn stored_row(
        &self,
        payload: &Decoded<'_>,
        row: usize,
        locator: &history::Locator<'_>,
        validated: row::ValidatedRow,
    ) -> Result<StoredRow> {
        self.containers.grow(sum(&[
            validated.encoded_key.capacity(),
            arc::<columnar::FramedKey>()?,
        ])?)?;
        let encoded_key = columnar::funded_encoded_key(
            validated.encoded_key,
            Arc::clone(&self.containers.credit),
        );
        Ok(StoredRow {
            record: columnar::RowPayload::RestoredV2 {
                chunk: Arc::clone(&payload.owner),
                row,
            },
            event_time: validated.event_time,
            row_id: locator.row_id,
            charge: locator.charge,
            encoded_key,
        })
    }

    fn prepare(
        self,
        metadata: ValidatedMetadata,
        [left, right]: [Vec<StoredRow>; 2],
    ) -> Result<Prepared> {
        self.validate_retained_metrics(&metadata, [&left, &right])?;
        let state = StreamJoinState {
            left_expirations: expirations(&left, self.check)?,
            right_expirations: expirations(&right, self.check)?,
            left: left.into(),
            right: right.into(),
            next_left_row_id: metadata.next_left_row_id,
            next_right_row_id: metadata.next_right_row_id,
            next_output_sequence: metadata.next_output_sequence,
            metrics: metadata.metrics,
            ended: metadata.ended,
            last_checkpoint_epoch: Epoch::new(metadata.epoch),
            deltas: DeltaTracking {
                needs_compaction: true,
                ..DeltaTracking::default()
            },
        };
        for row in state.left.iter().chain(state.right.iter()) {
            (self.check)()?;
            row.record.mark_live();
        }
        (self.check)()?;
        Ok(Prepared {
            state,
            containers: self.containers,
            workspace: self.workspace,
        })
    }

    fn validate_retained_metrics(
        &self,
        metadata: &ValidatedMetadata,
        [left, right]: [&[StoredRow]; 2],
    ) -> Result<()> {
        let diagnostics =
            super::super::metadata_validation::schema::restore_bases::inventory::diagnostic_bytes(
                self.config.name,
            )
            .ok_or_else(|| invalid("V2 retained metrics diagnostic charge overflow"))?;
        reserve(&self.workspace, diagnostics)?;
        restored_retained_metrics_match(&metadata.metrics, left, right, self.config.name)?;
        validate_limits(self.config.spec, [left, right])
    }
}

fn located_payload<'p, 'a>(
    payloads: &'p BTreeMap<(JoinSide, [u8; 32]), Decoded<'a>>,
    side: JoinSide,
    locator: &history::Locator<'_>,
) -> Result<(&'p Decoded<'a>, usize)> {
    let payload = payloads
        .get(&(side, locator.digest))
        .ok_or_else(|| invalid("V2 upsert references an unknown payload"))?;
    let index = usize::try_from(locator.payload_row)
        .map_err(|_| invalid("V2 payload row does not fit usize"))?;
    if checked(payload.frame.row_id(index))? != locator.row_id {
        return Err(invalid("V2 payload row ID differs from its upsert"));
    }
    Ok((payload, index))
}

fn side_index(side: JoinSide) -> usize {
    match side {
        JoinSide::Left => 0,
        JoinSide::Right => 1,
    }
}

fn validate_limits(spec: &StreamJoinSpec, sides: [&[StoredRow]; 2]) -> Result<()> {
    let limits = spec.limits();
    for rows in sides {
        let count =
            u64::try_from(rows.len()).map_err(|_| invalid("V2 retained row count exceeds u64"))?;
        let bytes = rows
            .iter()
            .try_fold(0_u64, |sum, row| sum.checked_add(row.charge))
            .ok_or_else(|| invalid("V2 retained byte count overflow"))?;
        if count > limits.max_state_rows_per_side || bytes > limits.max_state_bytes_per_side {
            return Err(invalid("V2 restored state exceeds configured limits"));
        }
    }
    Ok(())
}

fn expirations(rows: &[StoredRow], check: &dyn Fn() -> Result<()>) -> Result<ExpirationIndex> {
    let mut entries = BTreeMap::new();
    for (index, row) in rows.iter().enumerate() {
        check()?;
        entries.insert((row.event_time, row.row_id), (index, index as u128));
    }
    Ok(ExpirationIndex {
        entries,
        next_ordinal: rows.len() as u128,
    })
}
