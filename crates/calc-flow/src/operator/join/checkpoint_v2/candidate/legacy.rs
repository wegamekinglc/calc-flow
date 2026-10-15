//! Fully validate legacy history before constructing independent narrow state.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use super::super::{
    budget,
    geometry::{checked, invalid},
    ipc::{
        self,
        accounting::{self, reserve, sum},
    },
    payload::{Funding, OwnedPayload},
    row,
};
use super::{Decoder, Prepared, StoredRow, ValidatedMetadata, migration, side_index};
use crate::runtime::streaming::gather_work::RetirementGuard;
use crate::{
    Result,
    operator::join::{
        JoinSide, SegmentKind, checkpoint_metadata_compatible, columnar, decode_join_metadata,
        parse_segment_kind,
    },
};

mod frame;

type Identity = (Vec<u8>, i64, u64);
type Fold = BTreeMap<Identity, StoredRow>;

struct Segment<'a> {
    side: JoinSide,
    kind: SegmentKind,
    bytes: &'a [u8],
}

type Inventory<'a> = BTreeMap<(usize, u64, bool), Segment<'a>>;

impl Decoder<'_, '_> {
    pub(super) fn decode_legacy(mut self) -> Result<Prepared> {
        let metadata = self.legacy_metadata()?;
        let retirement = self.shared_retirement()?;
        let inventory = self.legacy_inventory(metadata.epoch)?;
        Self::validate_legacy_inventory(&inventory, &metadata)?;
        let (mut rows, repeated_ids) =
            self.legacy_rows(&inventory, &metadata, retirement.as_ref())?;
        drop(inventory);
        self.validate_retained_metrics(&metadata, [&rows[0], &rows[1]])?;
        self.narrow_legacy_rows(&mut rows, retirement.as_ref())?;
        drop(retirement);
        self.assemble(
            metadata,
            rows,
            super::super::WriterState::migration(),
            repeated_ids,
        )
    }

    fn narrow_legacy_rows(
        &self,
        rows: &mut [Vec<StoredRow>; 2],
        retirement: Option<&Arc<RetirementGuard>>,
    ) -> Result<()> {
        let layout = self
            .config
            .projection
            .ok_or_else(|| invalid("legacy migration requires a retained layout"))?;
        migration::migrate_rows(rows, layout, &self.workspace, retirement, self.check)
    }

    fn legacy_metadata(&self) -> Result<ValidatedMetadata> {
        (self.check)()?;
        checked(super::super::metadata::validate(
            &self.snapshot.inline_metadata,
        ))?;
        let metadata = decode_join_metadata(self.snapshot, self.config.name)?;
        if !checkpoint_metadata_compatible(&metadata, self.config.spec) {
            return Err(invalid("legacy checkpoint specification is incompatible"));
        }
        (self.check)()?;
        Ok(metadata.into())
    }

    fn legacy_inventory(&self, epoch: u64) -> Result<Inventory<'_>> {
        reserve(
            &self.workspace,
            budget::tree::<(usize, u64, bool), Segment<'_>>(self.snapshot.segments.len())?,
        )?;
        let mut inventory = BTreeMap::new();
        for (name, segment) in &self.snapshot.segments {
            (self.check)()?;
            let (key, segment) = self.legacy_segment(name, segment.bytes(), epoch)?;
            if inventory.insert(key, segment).is_some() {
                return Err(invalid("legacy inventory repeats one side and epoch"));
            }
        }
        Ok(inventory)
    }

    fn validate_legacy_inventory(
        inventory: &Inventory<'_>,
        metadata: &ValidatedMetadata,
    ) -> Result<()> {
        if inventory.is_empty() && !crate::operator::join::empty_retained_metrics(&metadata.metrics)
        {
            return Err(invalid("legacy segment inventory is empty"));
        }
        Ok(())
    }

    fn legacy_segment<'a>(
        &self,
        name: &str,
        bytes: &'a [u8],
        epoch: u64,
    ) -> Result<((usize, u64, bool), Segment<'a>)> {
        let kind = parse_segment_kind(name, self.config.name)?;
        let side = if name.starts_with("left-") {
            JoinSide::Left
        } else {
            JoinSide::Right
        };
        let (segment_epoch, delta) = match kind {
            SegmentKind::Base => (0, false),
            SegmentKind::Delta(value) => (value, true),
        };
        if segment_epoch > epoch {
            return Err(invalid("legacy delta epoch exceeds the checkpoint"));
        }
        Ok((
            (side_index(side), segment_epoch, delta),
            Segment { side, kind, bytes },
        ))
    }

    fn legacy_rows(
        &self,
        inventory: &Inventory<'_>,
        metadata: &ValidatedMetadata,
        retirement: Option<&Arc<RetirementGuard>>,
    ) -> Result<([Vec<StoredRow>; 2], [bool; 2])> {
        let mut folded = [Fold::new(), Fold::new()];
        for segment in inventory.values() {
            let next_id = if side_index(segment.side) == 0 {
                metadata.next_left_row_id
            } else {
                metadata.next_right_row_id
            };
            self.fold_legacy_segment(
                segment,
                next_id,
                &mut folded[side_index(segment.side)],
                retirement,
            )?;
        }
        let (left, left_repeats) = self.finish_legacy_side(std::mem::take(&mut folded[0]))?;
        let (right, right_repeats) = self.finish_legacy_side(std::mem::take(&mut folded[1]))?;
        Ok(([left, right], [left_repeats, right_repeats]))
    }

    fn fold_legacy_segment(
        &self,
        segment: &Segment<'_>,
        next_id: u64,
        folded: &mut Fold,
        retirement: Option<&Arc<RetirementGuard>>,
    ) -> Result<()> {
        let mut cursor = frame::Cursor::new(
            segment.bytes,
            &segment.kind,
            self.config.name,
            segment.side.as_str(),
        )?;
        let mut seen = BTreeSet::new();
        while let Some(frame) = cursor.next()? {
            (self.check)()?;
            if frame.id >= next_id {
                return Err(invalid("legacy historical row ID exceeds its counter"));
            }
            self.apply_legacy_frame(&frame, segment.side, &mut seen, folded, retirement)?;
        }
        (self.check)()
    }

    fn apply_legacy_frame(
        &self,
        frame: &frame::Row<'_>,
        side: JoinSide,
        seen: &mut BTreeSet<Identity>,
        folded: &mut Fold,
        retirement: Option<&Arc<RetirementGuard>>,
    ) -> Result<()> {
        let row = self.legacy_row(frame, side, retirement)?;
        let identity = self.legacy_identity(frame, row.as_ref())?;
        if !seen.insert(identity.clone()) {
            return Err(invalid("legacy segment repeats one row identity"));
        }
        match row {
            Some(row) => {
                folded.insert(identity, row);
            }
            None => {
                folded.remove(&identity);
            }
        }
        Ok(())
    }

    fn legacy_identity(&self, frame: &frame::Row<'_>, row: Option<&StoredRow>) -> Result<Identity> {
        let key = match row {
            Some(row) => row.encoded_key.as_slice(),
            None => frame
                .key
                .ok_or_else(|| invalid("legacy tombstone key is missing"))?,
        };
        self.reserve_legacy_identity(key.len())?;
        (self.check)()?;
        Ok((key.to_vec(), frame.time, frame.id))
    }

    fn reserve_legacy_identity(&self, length: usize) -> Result<()> {
        reserve(
            &self.workspace,
            sum(&[
                budget::tree::<Identity, StoredRow>(1)?,
                budget::tree::<Identity, ()>(1)?,
                accounting::product(length, 2)?,
            ])?,
        )?;
        Ok(())
    }

    fn legacy_row(
        &self,
        frame: &frame::Row<'_>,
        side: JoinSide,
        retirement: Option<&Arc<RetirementGuard>>,
    ) -> Result<Option<StoredRow>> {
        let Some((charge, bytes)) = frame.payload else {
            return Ok(None);
        };
        let owner = self.legacy_payload(bytes, side, retirement)?;
        let workspace = self.workspace.new_empty();
        let validated = row::validate_legacy(
            owner.record(),
            frame.time,
            charge,
            frame.key,
            &self.row_context(side, &workspace),
        )?;
        self.containers.grow(sum(&[
            validated.encoded_key.capacity(),
            accounting::arc::<columnar::FramedKey>()?,
        ])?)?;
        let encoded_key = columnar::funded_encoded_key(
            validated.encoded_key,
            Arc::clone(&self.containers.credit),
        );
        Ok(Some(StoredRow {
            record: columnar::RowPayload::RestoredV2 {
                chunk: owner,
                row: 0,
            },
            event_time: validated.event_time,
            row_id: frame.id,
            charge,
            encoded_key,
        }))
    }

    fn legacy_payload(
        &self,
        bytes: &[u8],
        side: JoinSide,
        retirement: Option<&Arc<RetirementGuard>>,
    ) -> Result<Arc<OwnedPayload>> {
        let workspace = self.workspace.new_empty();
        let credit = self.containers.credit.new_empty();
        reserve(
            &credit,
            sum(&[
                budget::payload_seed()?,
                accounting::arc::<RetirementGuard>()?,
            ])?,
        )?;
        let funding = Funding::shared(self.config.schemas.clone(), credit, retirement.cloned());
        ipc::decode(
            bytes,
            Arc::clone(&self.config.schemas[side_index(side)]),
            1,
            &ipc::Admission {
                workspace: &workspace,
                resident: &funding,
                verifier: &self.verifier,
                check: self.check,
            },
        )
    }

    fn finish_legacy_side(&self, folded: Fold) -> Result<(Vec<StoredRow>, bool)> {
        let count = folded.len();
        reserve(&self.workspace, budget::tree::<(u64, i64), ()>(count)?)?;
        let mut identities = BTreeSet::new();
        let mut rows = self.empty_legacy_rows(count)?;
        let mut repeated = false;
        for row in folded.into_values() {
            (self.check)()?;
            if !identities.insert((row.row_id, row.event_time.as_micros())) {
                return Err(invalid("legacy live rows repeat an event-time identity"));
            }
            repeated |= super::super::super::restored_row_id_repeats(&identities, row.row_id);
            rows.push(row);
        }
        Ok((rows, repeated))
    }
    fn empty_legacy_rows(&self, count: usize) -> Result<Vec<StoredRow>> {
        self.containers.grow(budget::containers(count)?)?;
        let rows = Vec::with_capacity(count);
        if rows.capacity() != count {
            return Err(invalid(
                "legacy restored row capacity differs from its funding",
            ));
        }
        Ok(rows)
    }
}
