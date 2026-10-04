use super::{Callback, Record, mismatch};
use crate::{EventTime, IngressProgress, IngressState, Result};

const MAGIC: &[u8; 8] = b"CFASRPL1";
pub(super) const WIDTH: usize = 55;
pub(super) const HEADER: usize = 16;

pub(super) fn encode(records: &[Record]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(HEADER + records.len() * WIDTH);
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&(records.len() as u64).to_le_bytes());
    for record in records {
        let (kind, side, sequence) = match record.callback {
            Callback::Data { side, sequence } => (0, side, sequence),
            Callback::Progress => (1, 0, 0),
            Callback::End => (2, 0, 0),
        };
        bytes.extend_from_slice(&[kind, side]);
        bytes.extend_from_slice(&sequence.to_le_bytes());
        time(&mut bytes, record.input_watermark);
        for progress in record.progress {
            bytes.push(match progress.state() {
                IngressState::Active => 0,
                IngressState::Idle => 1,
                IngressState::Ended => 2,
            });
            time(&mut bytes, progress.watermark());
        }
        bytes.extend_from_slice(&(record.max_rows as u64).to_le_bytes());
        bytes.extend_from_slice(&(record.max_bytes as u64).to_le_bytes());
    }
    bytes
}

fn time(bytes: &mut Vec<u8>, time: Option<EventTime>) {
    bytes.push(u8::from(time.is_some()));
    bytes.extend_from_slice(&time.map_or(0, EventTime::as_micros).to_le_bytes());
}

pub(super) fn decode_into(bytes: &[u8], expected: u64, records: &mut Vec<Record>) -> Result<()> {
    let count = usize::try_from(expected).map_err(|_| mismatch("record count overflowed"))?;
    let length = count
        .checked_mul(WIDTH)
        .and_then(|length| length.checked_add(HEADER))
        .ok_or_else(|| mismatch("frame length overflowed"))?;
    if bytes.len() != length || !bytes.starts_with(MAGIC) {
        return Err(mismatch("frame header or length differs"));
    }
    let mut reader = Reader { bytes: &bytes[8..] };
    if reader.integer()? != expected {
        return Err(mismatch("frame record count differs"));
    }
    if records.capacity() - records.len() < count {
        return Err(mismatch("records exceed prepaid capacity"));
    }
    for _ in 0..count {
        records.push(reader.record()?);
    }
    Ok(())
}

struct Reader<'a> {
    bytes: &'a [u8],
}

impl Reader<'_> {
    fn byte(&mut self) -> Result<u8> {
        let (value, remaining) = self
            .bytes
            .split_first()
            .ok_or_else(|| mismatch("truncated frame"))?;
        self.bytes = remaining;
        Ok(*value)
    }

    fn integer(&mut self) -> Result<u64> {
        let bytes = self
            .bytes
            .get(..8)
            .ok_or_else(|| mismatch("truncated integer"))?;
        let value = u64::from_le_bytes(bytes.try_into().map_err(|_| mismatch("invalid integer"))?);
        self.bytes = &self.bytes[8..];
        Ok(value)
    }

    fn time(&mut self) -> Result<Option<EventTime>> {
        let tag = self.byte()?;
        let value = i64::from_le_bytes(self.integer()?.to_le_bytes());
        match (tag, value) {
            (0, 0) => Ok(None),
            (1, value) => Ok(Some(EventTime::from_micros(value))),
            _ => Err(mismatch("invalid timestamp tag")),
        }
    }

    fn progress(&mut self) -> Result<IngressProgress> {
        let state = match self.byte()? {
            0 => IngressState::Active,
            1 => IngressState::Idle,
            2 => IngressState::Ended,
            _ => return Err(mismatch("invalid ingress state")),
        };
        Ok(IngressProgress::new(state, self.time()?))
    }

    fn record(&mut self) -> Result<Record> {
        let (kind, side, sequence) = (self.byte()?, self.byte()?, self.integer()?);
        let callback = match (kind, side, sequence) {
            (0, side @ 0..=1, sequence) => Callback::Data { side, sequence },
            (1, 0, 0) => Callback::Progress,
            (2, 0, 0) => Callback::End,
            _ => return Err(mismatch("invalid callback identity")),
        };
        let input_watermark = self.time()?;
        let progress = [self.progress()?, self.progress()?];
        let max_rows =
            usize::try_from(self.integer()?).map_err(|_| mismatch("row budget overflowed"))?;
        let max_bytes =
            usize::try_from(self.integer()?).map_err(|_| mismatch("byte budget overflowed"))?;
        if max_rows == 0 || max_bytes == 0 {
            return Err(mismatch("empty output budget"));
        }
        Ok(Record {
            callback,
            input_watermark,
            progress,
            max_rows,
            max_bytes,
        })
    }
}
