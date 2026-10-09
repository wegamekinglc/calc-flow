use std::io::Write;

use datafusion::execution::memory_pool::MemoryReservation;

use super::super::ipc::accounting::{add, sum};
use crate::{CalcFlowError, Result};

pub(super) struct PaidBuffer<'a> {
    bytes: Vec<u8>,
    credit: &'a MemoryReservation,
    scratch: usize,
}

impl<'a> PaidBuffer<'a> {
    pub(super) fn new(credit: &'a MemoryReservation, scratch: usize) -> Self {
        Self {
            bytes: Vec::new(),
            credit,
            scratch,
        }
    }

    pub(super) fn append(&mut self, input: &[u8]) -> Result<()> {
        self.write_all(input)
            .map_err(|_| error("V2 checkpoint output growth failed"))
    }

    pub(super) fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    fn grow(&mut self, required: usize) -> std::io::Result<()> {
        let capacity = required
            .max(256)
            .checked_next_power_of_two()
            .ok_or_else(|| std::io::Error::other("V2 checkpoint output overflow"))?;
        let charge =
            sum(&[self.scratch, self.bytes.capacity(), capacity]).map_err(std::io::Error::other)?;
        ensure(self.credit, charge).map_err(std::io::Error::other)?;
        self.bytes
            .try_reserve_exact(capacity - self.bytes.len())
            .map_err(std::io::Error::other)?;
        if self.bytes.capacity() > capacity {
            return Err(std::io::Error::other(
                "V2 checkpoint output exceeded paid capacity",
            ));
        }
        Ok(())
    }
}

impl Write for PaidBuffer<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let required = add(self.bytes.len(), bytes.len()).map_err(std::io::Error::other)?;
        if required > self.bytes.capacity() {
            self.grow(required)?;
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub(super) fn ensure(credit: &MemoryReservation, bytes: usize) -> Result<()> {
    if bytes > credit.size() {
        credit
            .try_grow(bytes - credit.size())
            .map_err(|_| error("V2 checkpoint writer admission failed"))?;
    }
    Ok(())
}

pub(super) fn error(message: &str) -> CalcFlowError {
    CalcFlowError::Internal {
        message: message.into(),
    }
}
