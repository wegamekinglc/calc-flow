use super::super::JoinSide;

const PAYLOAD_MAGIC: &[u8; 8] = b"CFJPAY2\0";
const BASE_MAGIC: &[u8; 8] = b"CFJIDX2\0";
const DELTA_MAGIC: &[u8; 8] = b"CFJDIX2\0";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct InvalidFrame(pub(super) &'static str);

type Result<T> = std::result::Result<T, InvalidFrame>;

pub(super) struct Payload<'a> {
    pub(super) rows: usize,
    ids: &'a [u8],
    pub(super) ipc: &'a [u8],
}

impl<'a> Payload<'a> {
    pub(super) fn decode(bytes: &'a [u8], side: JoinSide) -> Result<Self> {
        let mut reader = Reader::new(bytes);
        reader.header(*PAYLOAD_MAGIC, side)?;
        let rows = reader.usize()?;
        let ipc_bytes = reader.usize()?;
        if rows == 0 {
            return Err(InvalidFrame("payload must contain rows"));
        }
        Self::body(reader, rows, ipc_bytes)
    }

    fn body(mut reader: Reader<'a>, rows: usize, ipc_bytes: usize) -> Result<Self> {
        let ids = reader.take(checked_bytes(rows, 8)?)?;
        let ipc = reader.take(ipc_bytes)?;
        reader.finish()?;
        Ok(Self { rows, ids, ipc })
    }

    pub(super) fn row_id(&self, row: usize) -> Result<u64> {
        let start = checked_bytes(row, 8)?;
        let end = start
            .checked_add(8)
            .ok_or(InvalidFrame("row ID overflow"))?;
        let bytes = self
            .ids
            .get(start..end)
            .ok_or(InvalidFrame("payload row is out of range"))?;
        Ok(u64::from_le_bytes(bytes.try_into().expect("eight bytes")))
    }
}

pub(super) struct Index<'a> {
    pub(super) epoch: Option<u64>,
    pub(super) upserts: usize,
    pub(super) tombstones: usize,
    body: &'a [u8],
}

impl<'a> Index<'a> {
    pub(super) fn base(bytes: &'a [u8], side: JoinSide) -> Result<Self> {
        let mut reader = Reader::new(bytes);
        reader.header(*BASE_MAGIC, side)?;
        let index = Self::body(reader, None)?;
        if index.tombstones != 0 {
            return Err(InvalidFrame("base cannot contain tombstones"));
        }
        Ok(index)
    }

    pub(super) fn delta(bytes: &'a [u8], side: JoinSide, epoch: u64) -> Result<Self> {
        let mut reader = Reader::new(bytes);
        reader.header(*DELTA_MAGIC, side)?;
        if epoch == 0 || reader.u64()? != epoch {
            return Err(InvalidFrame("delta epoch does not match inventory"));
        }
        let index = Self::body(reader, Some(epoch))?;
        if index.upserts == 0 && index.tombstones == 0 {
            return Err(InvalidFrame("delta must contain operations"));
        }
        Ok(index)
    }

    fn body(mut reader: Reader<'a>, epoch: Option<u64>) -> Result<Self> {
        let upserts = reader.usize()?;
        let tombstones = reader.usize()?;
        let minimum = checked_bytes(upserts, 72)?
            .checked_add(checked_bytes(tombstones, 24)?)
            .ok_or(InvalidFrame("index count overflow"))?;
        let body = reader.remaining();
        if minimum > body.len() {
            return Err(InvalidFrame("index counts exceed its body"));
        }
        Ok(Self {
            epoch,
            upserts,
            tombstones,
            body,
        })
    }

    pub(super) fn records(&self) -> Records<'a> {
        Records {
            reader: Reader::new(self.body),
            upserts: self.upserts,
            tombstones: self.tombstones,
            previous_upsert: None,
            previous_tombstone: None,
        }
    }
}

pub(super) struct Upsert<'a> {
    pub(super) row_id: u64,
    pub(super) time: i64,
    pub(super) charge: u64,
    pub(super) digest: [u8; 32],
    pub(super) payload_row: u64,
    pub(super) key: &'a [u8],
}

pub(super) struct Tombstone<'a> {
    pub(super) row_id: u64,
    pub(super) time: i64,
    pub(super) key: &'a [u8],
}

pub(super) enum Record<'a> {
    Upsert(Upsert<'a>),
    Tombstone(Tombstone<'a>),
}

pub(super) struct Records<'a> {
    reader: Reader<'a>,
    upserts: usize,
    tombstones: usize,
    previous_upsert: Option<u64>,
    previous_tombstone: Option<u64>,
}

impl<'a> Records<'a> {
    pub(super) fn next(&mut self) -> Result<Option<Record<'a>>> {
        if self.upserts > 0 {
            let row = self.upsert()?;
            ordered_id(&mut self.previous_upsert, row.row_id)?;
            self.upserts -= 1;
            return Ok(Some(Record::Upsert(row)));
        }
        if self.tombstones > 0 {
            let row = self.tombstone()?;
            ordered_id(&mut self.previous_tombstone, row.row_id)?;
            self.tombstones -= 1;
            return Ok(Some(Record::Tombstone(row)));
        }
        self.reader.finish()?;
        Ok(None)
    }

    fn upsert(&mut self) -> Result<Upsert<'a>> {
        let row_id = self.reader.u64()?;
        let time = self.reader.i64()?;
        let charge = self.reader.u64()?;
        let digest = self.reader.take(32)?.try_into().expect("32 bytes");
        let payload_row = self.reader.u64()?;
        let key = self.reader.key()?;
        Ok(Upsert {
            row_id,
            time,
            charge,
            digest,
            payload_row,
            key,
        })
    }

    fn tombstone(&mut self) -> Result<Tombstone<'a>> {
        Ok(Tombstone {
            row_id: self.reader.u64()?,
            time: self.reader.i64()?,
            key: self.reader.key()?,
        })
    }
}

fn ordered_id(previous: &mut Option<u64>, id: u64) -> Result<()> {
    if previous.is_some_and(|previous| id <= previous) {
        return Err(InvalidFrame("index group IDs must be strictly increasing"));
    }
    *previous = Some(id);
    Ok(())
}

fn checked_bytes(count: usize, width: usize) -> Result<usize> {
    count
        .checked_mul(width)
        .filter(|bytes| *bytes <= usize::MAX >> 1)
        .ok_or(InvalidFrame("frame length overflow"))
}

struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes }
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        let (value, rest) = self
            .bytes
            .split_at_checked(count)
            .ok_or(InvalidFrame("truncated frame"))?;
        self.bytes = rest;
        Ok(value)
    }

    fn header(&mut self, magic: [u8; 8], side: JoinSide) -> Result<()> {
        if self.take(8)? != magic || self.take(4)? != 2_u32.to_le_bytes() {
            return Err(InvalidFrame("magic or codec version mismatch"));
        }
        let expected = match side {
            JoinSide::Left => 0,
            JoinSide::Right => 1,
        };
        if self.take(4)? != [expected, 0, 0, 0] {
            return Err(InvalidFrame("side or reserved bytes mismatch"));
        }
        Ok(())
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("eight bytes"),
        ))
    }

    fn i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(
            self.take(8)?.try_into().expect("eight bytes"),
        ))
    }

    fn usize(&mut self) -> Result<usize> {
        usize::try_from(self.u64()?).map_err(|_| InvalidFrame("length exceeds usize"))
    }

    fn key(&mut self) -> Result<&'a [u8]> {
        let count = self.usize()?;
        self.take(count)
    }

    const fn remaining(self) -> &'a [u8] {
        self.bytes
    }

    fn finish(&self) -> Result<()> {
        if self.bytes.is_empty() {
            Ok(())
        } else {
            Err(InvalidFrame("trailing frame bytes"))
        }
    }
}
