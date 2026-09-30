//! Stable buffer IDs follow their first canonical reference, never addresses.

use super::{Cursor, mismatch, put, require_capacity};
use crate::{
    Result,
    operator::asof::state::{EncodedBatch, Encoding},
};
use datafusion::arrow::{
    array::{BinaryArray, LargeBinaryArray},
    buffer::{Buffer, OffsetBuffer, ScalarBuffer},
};
use std::{collections::BTreeMap, sync::Arc};

#[derive(Default)]
pub(super) struct OwnerWriter {
    ids: BTreeMap<usize, u32>,
    owners: Vec<Encoding>,
    bytes: u64,
}

impl OwnerWriter {
    pub fn register(&mut self, encoding: &Encoding) {
        let Some((address, _)) = encoding.allocation() else {
            return;
        };
        if self.ids.contains_key(&address) {
            return;
        }
        let id = u32::try_from(self.owners.len()).expect("preflighted owner domain");
        self.ids.insert(address, id);
        self.bytes += encoding.owner_wire_length();
        self.owners.push(encoding.clone());
    }

    pub fn encoded_length(&self) -> u64 {
        self.bytes
    }

    pub fn write(&self, bytes: &mut Vec<u8>) {
        put(bytes, self.owners.len() as u64);
        for owner in &self.owners {
            write_owner(bytes, owner);
        }
    }

    pub fn reference(&self, bytes: &mut Vec<u8>, encoding: &Encoding) {
        let mut reference = [0_u8; 16];
        if let Encoding::Inline { len, bytes } = encoding {
            reference[1] = *len;
            reference[2..12].copy_from_slice(bytes);
        } else {
            reference[0] = 1;
            let (address, _) = encoding.allocation().expect("shared owner");
            reference[4..8].copy_from_slice(&self.ids[&address].to_le_bytes());
            let row = match encoding {
                Encoding::Batch { row, .. } => *row,
                _ => 0,
            };
            reference[8..12].copy_from_slice(&row.to_le_bytes());
        }
        bytes.extend_from_slice(&reference);
    }
}

fn write_owner(bytes: &mut Vec<u8>, encoding: &Encoding) {
    match encoding {
        Encoding::Shared(values) => {
            bytes.extend_from_slice(&[0, 0]);
            for value in [1, values.capacity(), values.len(), 0] {
                put(bytes, value as u64);
            }
            bytes.extend_from_slice(values);
        }
        Encoding::Batch { rows, .. } => {
            bytes.extend_from_slice(&[if rows.offset_width() == 4 { 1 } else { 2 }, 0]);
            for value in [
                rows.len(),
                rows.values().capacity(),
                rows.values().len(),
                rows.offset_capacity_bytes() / rows.offset_width(),
            ] {
                put(bytes, value as u64);
            }
            bytes.extend_from_slice(rows.values().as_slice());
            write_offsets(bytes, rows);
        }
        Encoding::Inline { .. } => unreachable!("inline encodings have no owner"),
    }
}

fn write_offsets(bytes: &mut Vec<u8>, rows: &EncodedBatch) {
    #[cfg(target_endian = "little")]
    bytes.extend_from_slice(rows.offset_bytes());
    #[cfg(target_endian = "big")]
    for offset in rows.offset_bytes().chunks_exact(rows.offset_width()) {
        bytes.extend(offset.iter().rev());
    }
}

enum Owner {
    Shared(Arc<Vec<u8>>),
    Batch(Arc<EncodedBatch>),
}

pub(super) fn restore_charge(cursor: &mut Cursor<'_>) -> Result<u64> {
    let count = cursor.address()?;
    if count > cursor.bytes.len() / 34 || u32::try_from(count).is_err() {
        return Err(mismatch("ASOF v3 owner count exceeds index size"));
    }
    let mut charge = count as u64 * 256;
    let mut largest = 0;
    for _ in 0..count {
        let (length, bytes) = scan_owner(cursor)?;
        largest = largest.max(length);
        charge = super::restore_add(charge, bytes)?;
    }

    super::restore_add(charge, largest)
}

pub(super) struct OwnerReader {
    owners: Vec<Owner>,
    used: Vec<bool>,
    next: usize,
}

impl OwnerReader {
    pub fn read(cursor: &mut Cursor<'_>) -> Result<Self> {
        let count = cursor.address()?;
        if u32::try_from(count).is_err() || count > cursor.bytes.len() / 34 {
            return Err(mismatch("ASOF v3 owner count exceeds index size"));
        }
        let mut owners = Vec::with_capacity(count);
        for _ in 0..count {
            owners.push(read_owner(cursor)?);
        }
        Ok(Self {
            owners,
            used: vec![false; count],
            next: 0,
        })
    }

    pub fn reference(&mut self, cursor: &mut Cursor<'_>) -> Result<Encoding> {
        let bytes = cursor.take(16)?;
        if bytes[12..] != [0; 4] {
            return Err(mismatch("ASOF v3 reference padding differs"));
        }
        match bytes[0] {
            0 => inline_reference(bytes),
            1 => self.shared_reference(bytes),
            _ => Err(mismatch("ASOF v3 reference tag differs")),
        }
    }

    fn shared_reference(&mut self, bytes: &[u8]) -> Result<Encoding> {
        if bytes[1..4] != [0; 3] {
            return Err(mismatch("ASOF v3 shared reference padding differs"));
        }
        let id = u32::from_le_bytes(bytes[4..8].try_into().expect("four bytes")) as usize;
        let row = u32::from_le_bytes(bytes[8..12].try_into().expect("four bytes"));
        self.owners
            .get(id)
            .ok_or_else(|| mismatch("ASOF v3 encoding owner is missing"))?;
        self.mark_used(id)?;
        let owner = &self.owners[id];
        match owner {
            Owner::Shared(values) if row == 0 => Ok(Encoding::Shared(values.clone())),
            Owner::Batch(rows) if (row as usize) < rows.len() => Ok(Encoding::Batch {
                rows: rows.clone(),
                row,
            }),
            _ => Err(mismatch("ASOF v3 encoding row reference differs")),
        }
    }

    fn mark_used(&mut self, id: usize) -> Result<()> {
        if !self.used[id] {
            if id != self.next {
                return Err(mismatch("ASOF v3 owner references are noncanonical"));
            }
            self.next += 1;
            self.used[id] = true;
        }
        Ok(())
    }

    pub fn finish(&self) -> Result<()> {
        if self.next == self.owners.len() {
            Ok(())
        } else {
            Err(mismatch("ASOF v3 contains an unreferenced encoding owner"))
        }
    }
}

fn inline_reference(bytes: &[u8]) -> Result<Encoding> {
    let length = usize::from(bytes[1]);
    if length > 10 || bytes[2 + length..12].iter().any(|byte| *byte != 0) {
        return Err(mismatch("ASOF v3 inline reference is noncanonical"));
    }
    Ok(Encoding::from_slice(&bytes[2..2 + length]))
}

#[derive(Clone, Copy)]
struct OwnerShape {
    kind: u8,
    rows: usize,
    capacity: usize,
    length: usize,
}

fn read_owner_shape(cursor: &mut Cursor<'_>) -> Result<OwnerShape> {
    let kind = cursor.byte()?;
    if cursor.byte()? != 0 {
        return Err(mismatch("ASOF v3 owner padding differs"));
    }
    let rows = cursor.address()?;
    let capacity = cursor.capacity(1)?;
    let length = cursor.address()?;
    require_capacity(capacity, length)?;
    Ok(OwnerShape {
        kind,
        rows,
        capacity,
        length,
    })
}

fn owner_offset_width(kind: u8) -> Result<usize> {
    match kind {
        0 => Ok(0),
        1 => Ok(4),
        2 => Ok(8),
        _ => Err(mismatch("ASOF v3 owner kind differs")),
    }
}

fn scan_owner(cursor: &mut Cursor<'_>) -> Result<(u64, u64)> {
    let shape = read_owner_shape(cursor)?;
    let width = owner_offset_width(shape.kind)?;
    let offsets = cursor.capacity(width)?;
    cursor.take(shape.length)?;
    scan_owner_offsets(cursor, &shape, width, offsets)?;
    Ok((
        shape.length as u64,
        owner_restore_allocation(&shape, offsets, width, cursor.limit)?,
    ))
}

fn owner_restore_allocation(
    shape: &OwnerShape,
    offsets: usize,
    width: usize,
    limit: u64,
) -> Result<u64> {
    let charge = super::restore_add(
        shape.capacity as u64,
        super::allocation(offsets, width, limit)?,
    )?;
    super::restore_add(charge, 512)
}

fn scan_owner_offsets(
    cursor: &mut Cursor<'_>,
    shape: &OwnerShape,
    width: usize,
    offsets: usize,
) -> Result<()> {
    if width != 0 {
        let count = offset_count(shape.rows, offsets)?;
        super::skip_rows(cursor, count, width)?;
    } else if shape.rows != 1 || offsets != 0 || shape.length <= 10 {
        return Err(mismatch("ASOF v3 shared owner shape differs"));
    }
    Ok(())
}

fn read_owner(cursor: &mut Cursor<'_>) -> Result<Owner> {
    let shape = read_owner_shape(cursor)?;
    let OwnerShape {
        kind,
        rows: _,
        capacity,
        length,
    } = shape;
    let offsets = cursor.capacity(if kind == 1 { 4 } else { 8 })?;
    let values = cursor.take(length)?;
    let mut buffer = Vec::with_capacity(capacity);
    buffer.extend_from_slice(values);
    build_owner(cursor, shape, offsets, buffer)
}

fn build_owner(
    cursor: &mut Cursor<'_>,
    shape: OwnerShape,
    offsets: usize,
    buffer: Vec<u8>,
) -> Result<Owner> {
    let OwnerShape {
        kind, rows, length, ..
    } = shape;
    match kind {
        0 if rows == 1 && offsets == 0 && length > 10 => Ok(Owner::Shared(Arc::new(buffer))),
        1 => read_binary(cursor, rows, offsets, buffer),
        2 => read_large_binary(cursor, rows, offsets, buffer),
        _ => Err(mismatch("ASOF v3 encoding owner shape differs")),
    }
}

fn offset_count(rows: usize, capacity: usize) -> Result<usize> {
    let count = rows
        .checked_add(1)
        .ok_or_else(|| mismatch("ASOF v3 offset count overflowed"))?;
    require_capacity(capacity, count)?;
    if rows == 0 || rows > u32::MAX as usize {
        return Err(mismatch("ASOF v3 owner rows exceed handle domain"));
    }
    Ok(count)
}

fn validate_offsets<T: Ord + Copy + TryInto<usize>>(offsets: &[T], length: usize) -> Result<()> {
    let first = offsets[0].try_into().ok();
    let last = offsets
        .last()
        .copied()
        .expect("one or more offsets")
        .try_into()
        .ok();
    if first != Some(0) || last != Some(length) || offsets.windows(2).any(|pair| pair[0] > pair[1])
    {
        return Err(mismatch("ASOF v3 binary offsets differ"));
    }
    Ok(())
}

fn read_binary(
    cursor: &mut Cursor<'_>,
    rows: usize,
    capacity: usize,
    values: Vec<u8>,
) -> Result<Owner> {
    let count = offset_count(rows, capacity)?;
    let raw = cursor.take(
        count
            .checked_mul(4)
            .ok_or_else(|| mismatch("ASOF v3 offset bytes overflowed"))?,
    )?;
    let mut offsets = Vec::with_capacity(capacity);
    for bytes in raw.chunks_exact(4) {
        offsets.push(i32::from_le_bytes(bytes.try_into().expect("four bytes")));
    }
    validate_offsets(&offsets, values.len())?;
    let array = BinaryArray::try_new(
        OffsetBuffer::new(ScalarBuffer::from(offsets)),
        Buffer::from_vec(values),
        None,
    )
    .map_err(|_| mismatch("ASOF v3 binary owner is invalid"))?;
    Ok(Owner::Batch(Arc::new(EncodedBatch::with_binary(array))))
}

fn read_large_binary(
    cursor: &mut Cursor<'_>,
    rows: usize,
    capacity: usize,
    values: Vec<u8>,
) -> Result<Owner> {
    let count = offset_count(rows, capacity)?;
    let raw = cursor.take(
        count
            .checked_mul(8)
            .ok_or_else(|| mismatch("ASOF v3 offset bytes overflowed"))?,
    )?;
    let mut offsets = Vec::with_capacity(capacity);
    for bytes in raw.chunks_exact(8) {
        offsets.push(i64::from_le_bytes(bytes.try_into().expect("eight bytes")));
    }
    validate_offsets(&offsets, values.len())?;
    let array = LargeBinaryArray::try_new(
        OffsetBuffer::new(ScalarBuffer::from(offsets)),
        Buffer::from_vec(values),
        None,
    )
    .map_err(|_| mismatch("ASOF v3 large binary owner is invalid"))?;
    Ok(Owner::Batch(Arc::new(EncodedBatch::with_large_binary(
        array,
    ))))
}
