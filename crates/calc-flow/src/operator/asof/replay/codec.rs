use super::{Callback, Record, mismatch};
use crate::{Cursor, EventTime, IngressProgress, IngressState, Result};
use datafusion::execution::memory_pool::MemoryReservation;
use std::sync::Arc;

const MAGIC: &[u8; 8] = b"CFASRPL2";
const WIDTH: usize = 56;
const HEADER: usize = 16;

pub(super) fn encoded_len(records: &[Record]) -> Result<usize> {
    records.iter().try_fold(HEADER, |total, record| {
        let cursor = match (&record.callback, &record.cursor) {
            (Callback::Data { .. }, Some(cursor)) if record.cursor_bytes > 0 => cursor,
            (Callback::Progress | Callback::End, None) if record.cursor_bytes == 0 => {
                return total
                    .checked_add(WIDTH)
                    .ok_or_else(|| mismatch("frame length overflowed"));
            }
            _ => return Err(mismatch("callback cursor differs")),
        };
        let owner = cursor
            .source_id()
            .ok_or_else(|| mismatch("cursor has no source owner"))?;
        [
            WIDTH,
            32,
            owner.len(),
            cursor.order().len(),
            cursor.payload_bytes(),
        ]
        .into_iter()
        .try_fold(total, |bytes, extra| {
            bytes
                .checked_add(extra)
                .ok_or_else(|| mismatch("frame length overflowed"))
        })
    })
}

pub(super) fn encode(records: &[Record]) -> Result<Vec<u8>> {
    let length = encoded_len(records)?;
    let mut bytes = Vec::with_capacity(length);
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
        bytes.push(u8::from(record.cursor.is_some()));
        if let Some(cursor) = &record.cursor {
            let owner = cursor
                .source_id()
                .expect("encoded length validated cursor owner");
            for value in [
                record.cursor_bytes,
                owner.len(),
                cursor.order().len(),
                cursor.payload_bytes(),
            ] {
                bytes.extend_from_slice(&(value as u64).to_le_bytes());
            }
            bytes.extend_from_slice(owner.as_bytes());
            bytes.extend_from_slice(cursor.order());
            serde_json::to_writer(&mut bytes, cursor.payload())
                .map_err(|error| mismatch(&error.to_string()))?;
        }
    }
    if bytes.len() != length {
        return Err(mismatch("encoded cursor length differs"));
    }
    Ok(bytes)
}

fn time(bytes: &mut Vec<u8>, time: Option<EventTime>) {
    bytes.push(u8::from(time.is_some()));
    bytes.extend_from_slice(&time.map_or(0, EventTime::as_micros).to_le_bytes());
}

pub(super) fn decode_into(
    bytes: &[u8],
    expected: u64,
    records: &mut Vec<Record>,
    credit: &mut MemoryReservation,
    reserve: &dyn Fn(u64) -> Result<MemoryReservation>,
) -> Result<()> {
    let count = usize::try_from(expected).map_err(|_| mismatch("record count overflowed"))?;
    let minimum = count
        .checked_mul(WIDTH)
        .and_then(|length| length.checked_add(HEADER))
        .ok_or_else(|| mismatch("frame length overflowed"))?;
    if bytes.len() < minimum || !bytes.starts_with(MAGIC) {
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
        records.push(reader.record(credit, reserve)?);
    }
    if !reader.bytes.is_empty() {
        return Err(mismatch("frame has trailing bytes"));
    }
    Ok(())
}

struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    fn take(&mut self, length: usize) -> Result<&'a [u8]> {
        let bytes = self
            .bytes
            .get(..length)
            .ok_or_else(|| mismatch("truncated frame"))?;
        self.bytes = &self.bytes[length..];
        Ok(bytes)
    }

    fn byte(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn integer(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(
            self.take(8)?
                .try_into()
                .map_err(|_| mismatch("invalid integer"))?,
        ))
    }

    fn length(&mut self) -> Result<usize> {
        usize::try_from(self.integer()?).map_err(|_| mismatch("cursor length overflowed"))
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

    fn record(
        &mut self,
        credit: &mut MemoryReservation,
        reserve: &dyn Fn(u64) -> Result<MemoryReservation>,
    ) -> Result<Record> {
        let (kind, side, sequence) = (self.byte()?, self.byte()?, self.integer()?);
        let callback = match (kind, side, sequence) {
            (0, side @ 0..=1, sequence) => Callback::Data { side, sequence },
            (1, 0, 0) => Callback::Progress,
            (2, 0, 0) => Callback::End,
            _ => return Err(mismatch("invalid callback identity")),
        };
        let input_watermark = self.time()?;
        let progress = [self.progress()?, self.progress()?];
        let max_rows = self.length()?;
        let max_bytes = self.length()?;
        if max_rows == 0 || max_bytes == 0 {
            return Err(mismatch("empty output budget"));
        }
        let (cursor, cursor_bytes) = self.cursor(callback, credit, reserve)?;
        Ok(Record {
            callback,
            input_watermark,
            progress,
            max_rows,
            max_bytes,
            cursor,
            cursor_bytes,
        })
    }

    fn cursor(
        &mut self,
        callback: Callback,
        credit: &mut MemoryReservation,
        reserve: &dyn Fn(u64) -> Result<MemoryReservation>,
    ) -> Result<(Option<Arc<Cursor>>, usize)> {
        match (callback, self.byte()?) {
            (Callback::Data { .. }, 1) => (),
            (Callback::Progress | Callback::End, 0) => return Ok((None, 0)),
            _ => return Err(mismatch("callback cursor tag differs")),
        }
        let (paid, owner_len, order_len, payload_len) = (
            self.length()?,
            self.length()?,
            self.length()?,
            self.length()?,
        );
        if paid == 0 || order_len == 0 || order_len > 16 * 1024 || owner_len == 0 {
            return Err(mismatch("cursor lengths or credit differ"));
        }
        let owner = std::str::from_utf8(self.take(owner_len)?)
            .map_err(|_| mismatch("cursor owner is not UTF-8"))?;
        let order = self.take(order_len)?;
        let payload = self.take(payload_len)?;
        let workspace = payload_len
            .checked_mul(512)
            .and_then(|bytes| bytes.checked_add(owner_len))
            .and_then(|bytes| bytes.checked_add(order_len))
            .and_then(|bytes| bytes.checked_add(4096))
            .ok_or_else(|| mismatch("cursor workspace overflowed"))?;
        let _workspace = reserve(workspace as u64)?;
        credit
            .size()
            .checked_add(paid)
            .ok_or_else(|| mismatch("cursor credit overflowed"))?;
        credit
            .try_grow(paid)
            .map_err(|_| mismatch("cursor exceeds prepaid state limits"))?;
        let result = (|| {
            let value = crate::json::parse_json_value(payload, "ASOF replay cursor")
                .map_err(|error| mismatch(&error.to_string()))?;
            let serde_json::Value::Object(payload) = value else {
                return Err(mismatch("cursor payload is not an object"));
            };
            let cursor = Cursor::new(owner, order.to_vec(), payload.into_iter().collect())
                .map_err(|error| mismatch(&error.to_string()))?;
            if cursor.retained_bytes()? > paid {
                return Err(mismatch("cursor exceeds its retained credit"));
            }
            Ok((Some(Arc::new(cursor)), paid))
        })();
        if result.is_err() {
            credit.shrink(paid);
        }
        result
    }
}
