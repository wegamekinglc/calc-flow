use crate::{OperatorStateSnapshot, Result, StreamJobContext};
use std::ops::Range;

#[derive(Default, Clone, Copy)]
pub(in crate::operator::join::metadata_validation::schema) struct Geometry {
    pub rows: [usize; 2],
    pub wire: [usize; 2],
    pub segments: usize,
    pub id_bytes: usize,
    pub longest_id: usize,
    pub header_keys: usize,
    pub longest_key: usize,
    pub seen_keys: usize,
    pub seen_count: usize,
}

#[derive(Clone, Copy)]
pub(in crate::operator::join::metadata_validation::schema) struct SegmentKind {
    pub side: usize,
    pub delta: bool,
}

pub(in crate::operator::join::metadata_validation::schema) fn kind(
    id: &str,
) -> Option<SegmentKind> {
    for (side, name) in ["left", "right"].into_iter().enumerate() {
        if id.strip_suffix("-base") == Some(name) {
            return Some(SegmentKind { side, delta: false });
        }
        let prefix = match side {
            0 => "left-delta-",
            _ => "right-delta-",
        };
        if id.starts_with(prefix) {
            return Some(SegmentKind { side, delta: true });
        }
    }
    None
}

pub(in crate::operator::join::metadata_validation::schema) struct Frame {
    pub key_bytes: usize,
    pub ipc: Option<Range<usize>>,
}

pub(in crate::operator::join::metadata_validation::schema) struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
    pub count: usize,
    delta: bool,
}

impl<'a> Cursor<'a> {
    pub(in crate::operator::join::metadata_validation::schema) fn new(
        bytes: &'a [u8],
        delta: bool,
    ) -> Option<Self> {
        Some(Self {
            bytes,
            offset: 16,
            count: header_count(bytes, delta)?,
            delta,
        })
    }

    pub(in crate::operator::join::metadata_validation::schema) fn next(&mut self) -> Option<Frame> {
        if self.delta {
            self.delta_frame()
        } else {
            take(self.bytes, &mut self.offset, 24)?;
            Some(Frame {
                key_bytes: 0,
                ipc: Some(self.ipc()?),
            })
        }
    }

    fn delta_frame(&mut self) -> Option<Frame> {
        let tag = take(self.bytes, &mut self.offset, 1)?[0];
        take(self.bytes, &mut self.offset, 16)?;
        let key_bytes = read_usize(self.bytes, &mut self.offset)?;
        take(self.bytes, &mut self.offset, key_bytes)?;
        let ipc = match tag {
            crate::operator::join::JOIN_DELTA_UPSERT_TAG => {
                take(self.bytes, &mut self.offset, 8)?;
                Some(self.ipc()?)
            }
            crate::operator::join::JOIN_DELTA_TOMBSTONE_TAG => None,
            _ => return None,
        };
        Some(Frame { key_bytes, ipc })
    }

    fn ipc(&mut self) -> Option<Range<usize>> {
        let length = read_usize(self.bytes, &mut self.offset)?;
        let start = self.offset;
        take(self.bytes, &mut self.offset, length)?;
        Some(start..self.offset)
    }

    pub(in crate::operator::join::metadata_validation::schema) fn finished(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

pub(in crate::operator::join::metadata_validation::schema) async fn scan(
    snapshot: &OperatorStateSnapshot,
    job: &StreamJobContext,
) -> Result<Option<Geometry>> {
    if !eligible_inventory(snapshot) {
        return Ok(None);
    }
    scan_entries(snapshot, job).await
}

pub(in crate::operator::join::metadata_validation::schema) async fn scan_all(
    snapshot: &OperatorStateSnapshot,
    job: &StreamJobContext,
) -> Result<Option<Geometry>> {
    let complete_bases =
        snapshot.segments.contains_key("left-base") && snapshot.segments.contains_key("right-base");
    if !complete_bases && !eligible_inventory(snapshot) {
        return Ok(None);
    }
    scan_entries(snapshot, job).await
}

async fn scan_entries(
    snapshot: &OperatorStateSnapshot,
    job: &StreamJobContext,
) -> Result<Option<Geometry>> {
    let mut geometry = Geometry::default();
    for (id, segment) in &snapshot.segments {
        let Some((kind, count, rows, keys, longest)) =
            scan_segment(id, segment.bytes(), job).await?
        else {
            return Ok(None);
        };
        if !add_segment(
            &mut geometry,
            id,
            segment.bytes().len(),
            kind.side,
            rows,
            keys,
            longest,
        ) {
            return Ok(None);
        }
        if kind.delta {
            geometry.seen_count = geometry.seen_count.max(count);
            geometry.seen_keys = geometry.seen_keys.max(keys);
        }
    }
    Ok(Some(geometry))
}

fn eligible_inventory(snapshot: &OperatorStateSnapshot) -> bool {
    match (
        snapshot.segments.contains_key("left-base"),
        snapshot.segments.contains_key("right-base"),
    ) {
        (false, false) => !snapshot.segments.is_empty(),
        (true, true) => snapshot.segments.len() > 2,
        _ => false,
    }
}

pub(in crate::operator::join::metadata_validation::schema) fn delta_count(
    snapshot: &OperatorStateSnapshot,
    geometry: &Geometry,
) -> Option<usize> {
    let bases = usize::from(snapshot.segments.contains_key("left-base"))
        + usize::from(snapshot.segments.contains_key("right-base"));
    geometry.segments.checked_sub(bases)
}

async fn scan_segment(
    id: &str,
    bytes: &[u8],
    job: &StreamJobContext,
) -> Result<Option<(SegmentKind, usize, usize, usize, usize)>> {
    super::super::super::copy_boundary(job).await?;
    let Some(kind) = kind(id) else {
        return Ok(None);
    };
    if kind.delta && !valid_epoch(id, job).await? {
        return Ok(None);
    }
    let Some(mut cursor) = Cursor::new(bytes, kind.delta) else {
        return Ok(None);
    };
    let Some((rows, keys, longest)) = scan_frames(&mut cursor, job).await? else {
        return Ok(None);
    };
    Ok(Some((kind, cursor.count, rows, keys, longest)))
}

async fn valid_epoch(id: &str, job: &StreamJobContext) -> Result<bool> {
    let epoch = id.split_once("-delta-").expect("recognized delta prefix").1;
    let digits = epoch.strip_prefix('+').unwrap_or(epoch).as_bytes();
    if digits.is_empty() {
        return Ok(false);
    }
    let mut value = 0_u64;
    for part in digits.chunks(4096) {
        super::super::super::copy_boundary(job).await?;
        let Some(next) = parse_digits(value, part) else {
            return Ok(false);
        };
        value = next;
    }
    Ok(true)
}

fn parse_digits(mut value: u64, bytes: &[u8]) -> Option<u64> {
    for byte in bytes {
        if !byte.is_ascii_digit() {
            return None;
        }
        value = value.checked_mul(10)?.checked_add(u64::from(byte - b'0'))?;
    }
    Some(value)
}

fn header_count(bytes: &[u8], delta: bool) -> Option<usize> {
    let mut offset = 0;
    let magic = if delta {
        crate::operator::join::JOIN_DELTA_MAGIC
    } else {
        crate::operator::join::JOIN_STATE_MAGIC
    };
    if take(bytes, &mut offset, 8)? != magic {
        return None;
    }
    let count = read_usize(bytes, &mut offset)?;
    let minimum = if delta { 25 } else { 32 };
    (count.checked_mul(minimum)?.checked_add(offset)? <= bytes.len()).then_some(count)
}

async fn scan_frames(
    cursor: &mut Cursor<'_>,
    job: &StreamJobContext,
) -> Result<Option<(usize, usize, usize)>> {
    let mut rows = 0_usize;
    let mut keys = 0_usize;
    let mut longest = 0_usize;
    for index in 0..cursor.count {
        if index.is_multiple_of(64) {
            super::super::super::copy_boundary(job).await?;
        }
        let Some(frame) = cursor.next() else {
            return Ok(None);
        };
        let Some(total) = keys.checked_add(frame.key_bytes) else {
            return Ok(None);
        };
        keys = total;
        longest = longest.max(frame.key_bytes);
        rows += usize::from(frame.ipc.is_some());
    }
    Ok(cursor.finished().then_some((rows, keys, longest)))
}

fn add_segment(
    geometry: &mut Geometry,
    id: &str,
    bytes: usize,
    side: usize,
    rows: usize,
    keys: usize,
    longest: usize,
) -> bool {
    let Some(total) = geometry.rows[side].checked_add(rows) else {
        return false;
    };
    geometry.rows[side] = total;
    let Some(total) = geometry.wire[side].checked_add(bytes) else {
        return false;
    };
    geometry.wire[side] = total;
    let Some(total) = geometry.id_bytes.checked_add(id.len()) else {
        return false;
    };
    geometry.id_bytes = total;
    let Some(total) = geometry.header_keys.checked_add(keys) else {
        return false;
    };
    geometry.header_keys = total;
    geometry.longest_key = geometry.longest_key.max(longest);
    geometry.longest_id = geometry.longest_id.max(id.len());
    geometry.segments += 1;
    true
}

fn read_usize(bytes: &[u8], offset: &mut usize) -> Option<usize> {
    usize::try_from(u64::from_le_bytes(take(bytes, offset, 8)?.try_into().ok()?)).ok()
}

fn take<'a>(bytes: &'a [u8], offset: &mut usize, length: usize) -> Option<&'a [u8]> {
    let end = offset.checked_add(length)?;
    let part = bytes.get(*offset..end)?;
    *offset = end;
    Some(part)
}
