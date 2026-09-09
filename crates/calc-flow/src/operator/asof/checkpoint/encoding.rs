use super::super::{
    checked,
    codec::BoundedWriter,
    state::{Encoding, LeftOrder, RightOrder, State},
};
use super::{MAGIC, mismatch};
use crate::{Result, StateSegment, StreamOperatorContext};
use std::{collections::BTreeMap, io::Write as _};

pub(super) async fn encode_state(
    state: &State,
    length: u64,
    limit: usize,
    context: &StreamOperatorContext<'_>,
) -> Result<StateSegment> {
    let mut writer = BoundedWriter::with_capacity(
        usize::try_from(length).expect("reserved address domain"),
        limit,
    );
    write_header(&mut writer, state)?;
    write_left(&mut writer, state, context).await?;
    write_right(&mut writer, state, context).await?;
    Ok(StateSegment::new(writer.bytes))
}

fn write_header(writer: &mut BoundedWriter, state: &State) -> Result<()> {
    write_bytes(writer, MAGIC)?;
    write_u64(writer, state.left.len() as u64)?;
    write_u64(writer, state.right.len() as u64)
}

async fn write_left(
    writer: &mut BoundedWriter,
    state: &State,
    context: &StreamOperatorContext<'_>,
) -> Result<()> {
    for (identity, payload) in &state.left {
        context.check_cancelled()?;
        write_left_row(writer, identity, payload)?;
        tokio::task::yield_now().await;
    }
    Ok(())
}

fn write_left_row(
    writer: &mut BoundedWriter,
    (time, key, sequence): &LeftOrder,
    payload: &StateSegment,
) -> Result<()> {
    write_bytes(writer, &time.to_le_bytes())?;
    write_blob(writer, key)?;
    write_blob(writer, sequence)?;
    write_blob(writer, payload.bytes())
}

async fn write_right(
    writer: &mut BoundedWriter,
    state: &State,
    context: &StreamOperatorContext<'_>,
) -> Result<()> {
    for (key, bucket) in &state.right {
        write_blob(writer, key)?;
        write_u64(writer, bucket.len() as u64)?;
        write_bucket(writer, bucket, context).await?;
    }
    Ok(())
}

async fn write_bucket(
    writer: &mut BoundedWriter,
    bucket: &BTreeMap<RightOrder, Option<StateSegment>>,
    context: &StreamOperatorContext<'_>,
) -> Result<()> {
    for (identity, payload) in bucket {
        context.check_cancelled()?;
        write_right_row(writer, identity, payload.as_ref())?;
        tokio::task::yield_now().await;
    }
    Ok(())
}

fn write_right_row(
    writer: &mut BoundedWriter,
    (time, sequence): &RightOrder,
    payload: Option<&StateSegment>,
) -> Result<()> {
    write_bytes(writer, &time.to_le_bytes())?;
    write_blob(writer, sequence)?;
    write_blob(writer, payload.map_or(&[], StateSegment::bytes))
}

fn write_bytes(writer: &mut BoundedWriter, bytes: &[u8]) -> Result<()> {
    writer
        .write_all(bytes)
        .map_err(|error| mismatch(&error.to_string()))
}

fn write_u64(writer: &mut BoundedWriter, value: u64) -> Result<()> {
    write_bytes(writer, &value.to_le_bytes())
}

fn write_blob(writer: &mut BoundedWriter, value: &[u8]) -> Result<()> {
    write_u64(writer, value.len() as u64)?;
    write_bytes(writer, value)
}

pub(in super::super) fn encoded_length(state: &State, name: &str) -> Result<u64> {
    if state.left.is_empty() && state.right.is_empty() {
        return Ok(0);
    }
    let mut size = 24;
    for ((_, key, sequence), payload) in &state.left {
        size = left_length(size, name, key, sequence, payload)?;
    }
    for (key, bucket) in &state.right {
        size = right_length(size, name, key, bucket)?;
    }
    Ok(size)
}

fn left_length(
    mut size: u64,
    name: &str,
    key: &Encoding,
    sequence: &Encoding,
    payload: &StateSegment,
) -> Result<u64> {
    size = checked(name, size, 32)?;
    for bytes in [key.as_slice(), sequence.as_slice(), payload.bytes()] {
        size = checked(name, size, bytes.len() as u64)?;
    }
    Ok(size)
}

fn right_length(
    mut size: u64,
    name: &str,
    key: &Encoding,
    bucket: &BTreeMap<RightOrder, Option<StateSegment>>,
) -> Result<u64> {
    size = checked(name, size, 16 + key.len() as u64)?;
    for ((_, sequence), payload) in bucket {
        size = checked(name, size, 24 + sequence.len() as u64)?;
        size = checked(
            name,
            size,
            payload.as_ref().map_or(0, |row| row.bytes().len() as u64),
        )?;
    }
    Ok(size)
}

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
