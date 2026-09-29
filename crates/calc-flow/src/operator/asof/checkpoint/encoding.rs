use super::{MAGIC, mismatch};
use crate::Result;

pub(super) struct Decoder<'a> {
    bytes: &'a [u8],
    max_rows: u64,
    rows: u64,
}

impl<'a> Decoder<'a> {
    pub(super) fn new(bytes: &'a [u8], max_rows: u64) -> Self {
        Self {
            bytes,
            max_rows,
            rows: 0,
        }
    }

    pub(super) fn header(&mut self) -> Result<(u64, u64)> {
        if self.take(8)? != MAGIC {
            return Err(mismatch("ASOF segment magic differs"));
        }
        Ok((self.count()?, self.count()?))
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        if count > self.bytes.len() {
            return Err(mismatch("ASOF truncated segment"));
        }
        let (value, rest) = self.bytes.split_at(count);
        self.bytes = rest;
        Ok(value)
    }

    fn integer(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("eight bytes"),
        ))
    }

    pub(super) fn time(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(
            self.take(8)?.try_into().expect("eight bytes"),
        ))
    }

    pub(super) fn count(&mut self) -> Result<u64> {
        let count = self.integer()?;
        if count > self.max_rows {
            return Err(mismatch("ASOF declared row count exceeds limits"));
        }
        Ok(count)
    }

    pub(super) fn row(&mut self) -> Result<()> {
        self.rows = self
            .rows
            .checked_add(1)
            .filter(|rows| *rows <= self.max_rows)
            .ok_or_else(|| mismatch("ASOF decoded row count exceeds limits"))?;
        Ok(())
    }

    pub(super) fn blob(&mut self) -> Result<&'a [u8]> {
        let size = usize::try_from(self.integer()?)
            .map_err(|_| mismatch("ASOF field length exceeds address domain"))?;
        self.take(size)
    }

    pub(super) fn finish(&self, message: &str) -> Result<()> {
        if !self.bytes.is_empty() {
            return Err(mismatch(message));
        }
        Ok(())
    }
}

#[derive(Default)]
struct DecodeCharge {
    largest_payload: u64,
    largest_identity: u64,
}

impl DecodeCharge {
    fn left_row(&mut self, decoder: &mut Decoder<'_>) -> Result<()> {
        decoder.row()?;
        decoder.time()?;
        self.largest_identity = self.largest_identity.max(decoder.blob()?.len() as u64);
        self.largest_identity = self.largest_identity.max(decoder.blob()?.len() as u64);
        self.largest_payload = self.largest_payload.max(decoder.blob()?.len() as u64);
        Ok(())
    }

    fn bucket(&mut self, decoder: &mut Decoder<'_>) -> Result<()> {
        self.largest_identity = self.largest_identity.max(decoder.blob()?.len() as u64);
        let count = decoder.count()?;
        for _ in 0..count {
            self.right_row(decoder)?;
        }
        Ok(())
    }

    fn right_row(&mut self, decoder: &mut Decoder<'_>) -> Result<()> {
        decoder.row()?;
        decoder.time()?;
        self.largest_identity = self.largest_identity.max(decoder.blob()?.len() as u64);
        self.largest_payload = self.largest_payload.max(decoder.blob()?.len() as u64);
        Ok(())
    }

    fn workspace(&self, encoded_bytes: u64, rows: u64) -> Result<u64> {
        encoded_bytes
            .checked_add(
                self.largest_payload
                    .checked_mul(4)
                    .ok_or_else(|| mismatch("ASOF decode workspace overflowed"))?
                    .max(self.largest_identity),
            )
            .and_then(|value| value.checked_add(rows.checked_mul(384)?))
            .ok_or_else(|| mismatch("ASOF decode workspace overflowed"))
    }
}

pub(super) fn restore_charge(bytes: &[u8], max_rows: u64) -> Result<u64> {
    let mut decoder = Decoder::new(bytes, max_rows);
    let (left, buckets) = decoder.header()?;
    let mut charge = DecodeCharge::default();
    for _ in 0..left {
        charge.left_row(&mut decoder)?;
    }
    for _ in 0..buckets {
        charge.bucket(&mut decoder)?;
    }
    decoder.finish("ASOF segment contains trailing bytes")?;
    charge.workspace(bytes.len() as u64, decoder.rows)
}
