//! Borrowed framing for the historical row-at-a-time representation.

use super::super::super::{geometry::invalid, ipc::accounting};
use crate::{
    Result,
    operator::join::{
        JOIN_DELTA_TOMBSTONE_TAG, JOIN_DELTA_UPSERT_TAG, SegmentKind, decode_side_header,
        delta_op_count, read_segment_i64, read_segment_u64, take_segment_bytes,
    },
};

pub(super) struct Row<'a> {
    pub(super) id: u64,
    pub(super) time: i64,
    pub(super) key: Option<&'a [u8]>,
    pub(super) payload: Option<(u64, &'a [u8])>,
}

pub(super) struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
    remaining: usize,
    delta: bool,
}

impl<'a> Cursor<'a> {
    pub(super) fn new(bytes: &'a [u8], kind: &SegmentKind, name: &str, side: &str) -> Result<Self> {
        let mut offset = 0;
        let (remaining, delta, minimum) = match kind {
            SegmentKind::Base => (
                usize::try_from(decode_side_header(bytes, &mut offset, name, side)?)
                    .map_err(|_| invalid("legacy row count exceeds usize"))?,
                false,
                32,
            ),
            SegmentKind::Delta(_) => (delta_op_count(bytes, &mut offset, name, side)?, true, 25),
        };
        if remaining > bytes.len().saturating_sub(offset) / minimum {
            return Err(invalid("legacy row count exceeds its segment framing"));
        }
        Ok(Self {
            bytes,
            offset,
            remaining,
            delta,
        })
    }

    pub(super) fn next(&mut self) -> Result<Option<Row<'a>>> {
        if self.remaining == 0 {
            if self.offset != self.bytes.len() {
                return Err(invalid("legacy segment has trailing bytes"));
            }
            return Ok(None);
        }
        self.remaining -= 1;
        let (tag, id, time, key) = self.header()?;
        let payload = self.payload(tag)?;
        Ok(Some(Row {
            id,
            time,
            key,
            payload,
        }))
    }

    fn header(&mut self) -> Result<(u8, u64, i64, Option<&'a [u8]>)> {
        let tag = if self.delta {
            take_segment_bytes(self.bytes, &mut self.offset, 1)?[0]
        } else {
            JOIN_DELTA_UPSERT_TAG
        };
        let id = read_segment_u64(self.bytes, &mut self.offset)?;
        let time = read_segment_i64(self.bytes, &mut self.offset)?;
        let key = if self.delta {
            Some(self.block()?)
        } else {
            None
        };
        Ok((tag, id, time, key))
    }

    fn payload(&mut self, tag: u8) -> Result<Option<(u64, &'a [u8])>> {
        match tag {
            JOIN_DELTA_UPSERT_TAG => {
                let charge = read_segment_u64(self.bytes, &mut self.offset)?;
                Ok(Some((charge, self.block()?)))
            }
            JOIN_DELTA_TOMBSTONE_TAG => Ok(None),
            _ => Err(invalid("legacy delta operation tag is invalid")),
        }
    }

    fn block(&mut self) -> Result<&'a [u8]> {
        let length = usize::try_from(read_segment_u64(self.bytes, &mut self.offset)?)
            .map_err(|_| invalid("legacy framed length exceeds usize"))?;
        let end = accounting::add(self.offset, length)?;
        let block = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| invalid("legacy framed block is truncated"))?;
        self.offset = end;
        Ok(block)
    }
}
