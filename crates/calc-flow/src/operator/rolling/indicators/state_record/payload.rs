use crate::{CalcFlowError, Result};
use serde::Serialize;
use std::io::{self, Write};

type Check<'a> = &'a mut dyn FnMut() -> Result<()>;

pub(super) struct JsonPayload {
    pub(super) length: usize,
    #[cfg(test)]
    pub(super) capacity: usize,
    pub(super) construction_peak: u64,
}
pub(super) fn json(value: &impl Serialize, check: Check<'_>) -> Result<JsonPayload> {
    check()?;
    let mut writer = Counter {
        length: 0,
        capacity: 128,
        peak: 128,
    };
    serde_json::to_writer(&mut writer, value).map_err(|_| invalid())?;
    check()?;
    Ok(JsonPayload {
        length: writer.length,
        #[cfg(test)]
        capacity: writer.capacity,
        construction_peak: u64::try_from(writer.peak).map_err(|_| invalid())?,
    })
}
struct Counter {
    length: usize,
    capacity: usize,
    peak: usize,
}
impl Write for Counter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let required = self
            .length
            .checked_add(bytes.len())
            .ok_or(io::ErrorKind::InvalidInput)?;
        self.grow(required)?;
        self.length = required;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
impl Counter {
    fn grow(&mut self, required: usize) -> io::Result<()> {
        if required <= self.capacity {
            return Ok(());
        }
        let next = self
            .capacity
            .checked_mul(2)
            .ok_or(io::ErrorKind::InvalidInput)?
            .max(required);
        self.peak = self.peak.max(
            self.capacity
                .checked_add(next)
                .ok_or(io::ErrorKind::InvalidInput)?,
        );
        self.capacity = next;
        Ok(())
    }
}
fn invalid() -> CalcFlowError {
    CalcFlowError::Format {
        message: "rolling checkpoint JSON payload size or capacity overflowed".into(),
    }
}

#[cfg(test)]
#[path = "payload_tests.rs"]
mod tests;
