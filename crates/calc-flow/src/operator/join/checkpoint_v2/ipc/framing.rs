use datafusion::arrow::ipc;
use datafusion::execution::memory_pool::MemoryReservation;
use flatbuffers::{ErrorTraceDetail, VerifierOptions};

use crate::{CalcFlowError, Result};

pub(super) struct Message<'a> {
    pub(super) metadata: ipc::Message<'a>,
    pub(super) body: &'a [u8],
}

pub(in crate::operator::join::checkpoint_v2) struct VerifierCredit {
    _credit: MemoryReservation,
}

impl VerifierCredit {
    pub(in crate::operator::join::checkpoint_v2) fn new(credit: MemoryReservation) -> Result<Self> {
        let options = VerifierOptions::default();
        let controls = size_of::<Self>()
            + size_of::<flatbuffers::InvalidFlatbuffer>()
            + size_of::<flatbuffers::Verifier<'static, 'static>>()
            + size_of::<VerifierOptions>()
            + "V2 IPC metadata is not aligned to eight bytes".len();
        let required = trace_peak(options.max_depth)
            .and_then(|bytes| bytes.checked_add(controls))
            .ok_or_else(|| invalid("V2 IPC verifier credit overflow"))?;
        credit
            .try_grow(required)
            .map_err(|_| CalcFlowError::Internal {
                message: "V2 IPC verifier credit admission failed".into(),
            })?;
        Ok(Self { _credit: credit })
    }
}

pub(super) struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
    ended: bool,
}

impl<'a> Cursor<'a> {
    pub(super) fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            offset: 0,
            ended: false,
        }
    }

    pub(super) fn next(
        &mut self,
        verifier: &VerifierCredit,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<Option<Message<'a>>> {
        check()?;
        if self.ended {
            return Ok(None);
        }
        let length = self.metadata_length()?;
        if length == 0 {
            self.ended = true;
            if self.offset != self.bytes.len() {
                return Err(invalid("V2 IPC contains bytes after its end marker"));
            }
            return Ok(None);
        }
        self.read_message(length, verifier, check).map(Some)
    }

    fn metadata_length(&mut self) -> Result<usize> {
        if self.take(4)? != [0xff; 4] {
            return Err(invalid("V2 IPC requires modern continuation markers"));
        }
        let encoded = i32::from_le_bytes(self.take(4)?.try_into().expect("four bytes"));
        let length =
            usize::try_from(encoded).map_err(|_| invalid("V2 IPC metadata length is negative"))?;
        if length % 8 != 0 {
            return Err(invalid("V2 IPC metadata is not aligned to eight bytes"));
        }
        Ok(length)
    }

    fn read_message(
        &mut self,
        length: usize,
        _verifier: &VerifierCredit,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<Message<'a>> {
        let options = VerifierOptions::default();
        let metadata = self.take(length)?;
        let parsed = ipc::root_as_message_with_opts(&options, metadata);
        check()?;
        let metadata = parsed.map_err(|_| invalid("V2 IPC message verification failed"))?;
        if metadata.version() != ipc::MetadataVersion::V5 {
            return Err(invalid("V2 IPC metadata must use version V5"));
        }
        let body_length = usize::try_from(metadata.bodyLength())
            .map_err(|_| invalid("V2 IPC body length is negative"))?;
        if body_length % 8 != 0 {
            return Err(invalid("V2 IPC body is not aligned to eight bytes"));
        }
        let body = self.take(body_length)?;
        Ok(Message { metadata, body })
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or_else(|| invalid("V2 IPC frame length overflow"))?;
        let bytes = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| invalid("V2 IPC frame is truncated"))?;
        self.offset = end;
        Ok(bytes)
    }
}

fn trace_peak(depth: usize) -> Option<usize> {
    let length = depth.checked_mul(2)?;
    if length == 0 {
        return Some(0);
    }
    let element = size_of::<ErrorTraceDetail>();
    let minimum = trace_minimum_capacity(element);
    let capacity = length.max(minimum).checked_next_power_of_two()?;
    let overlap = if length <= minimum { 0 } else { capacity / 2 };
    let bytes = capacity.checked_add(overlap)?.checked_mul(element)?;
    std::alloc::Layout::from_size_align(bytes, align_of::<ErrorTraceDetail>()).ok()?;
    Some(bytes)
}

fn trace_minimum_capacity(element: usize) -> usize {
    if element <= 1 {
        8
    } else if element <= 1024 {
        4
    } else {
        1
    }
}

fn invalid(message: &str) -> CalcFlowError {
    CalcFlowError::CheckpointMismatch {
        message: message.into(),
    }
}
