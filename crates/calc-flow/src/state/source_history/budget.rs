use crate::{CalcFlowError, Result};
use parking_lot::Mutex;
use std::sync::Arc;

/// Per-source limits for immutable raw input history.
///
/// ```
/// use calc_flow::SourceHistoryLimits;
/// let limits = SourceHistoryLimits::new(128, 64 * 1024 * 1024, 8 * 1024 * 1024)?;
/// # Ok::<(), calc_flow::CalcFlowError>(())
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SourceHistoryLimits {
    /// Maximum retained raw segments.
    pub max_segments: usize,
    /// Maximum total raw bytes in managed storage.
    pub max_bytes: u64,
    /// Maximum concurrent raw buffers, including backend verification copies.
    pub max_buffer_bytes: usize,
}

impl SourceHistoryLimits {
    /// Constructs positive limits.
    ///
    /// # Errors
    /// Returns an invalid-argument error for a zero limit.
    pub fn new(max_segments: usize, max_bytes: u64, max_buffer_bytes: usize) -> Result<Self> {
        let value = Self {
            max_segments,
            max_bytes,
            max_buffer_bytes,
        };
        value.validate()?;
        Ok(value)
    }

    pub(crate) fn validate(self) -> Result<()> {
        if self.max_segments == 0 || self.max_bytes == 0 || self.max_buffer_bytes == 0 {
            return Err(limit_error("limits", "history limits must be positive"));
        }
        Ok(())
    }
}

impl Default for SourceHistoryLimits {
    fn default() -> Self {
        Self {
            max_segments: 4096,
            max_bytes: 1024 * 1024 * 1024,
            max_buffer_bytes: 64 * 1024 * 1024,
        }
    }
}

#[derive(Default)]
struct Usage {
    segments: usize,
    bytes: u64,
    buffers: usize,
}

pub(super) struct Ledger {
    pub(super) limits: SourceHistoryLimits,
    usage: Mutex<Usage>,
}

impl Ledger {
    pub(super) fn new(limits: SourceHistoryLimits) -> Arc<Self> {
        Arc::new(Self {
            limits,
            usage: Mutex::new(Usage::default()),
        })
    }

    pub(super) fn storage(self: &Arc<Self>, bytes: u64) -> Result<StorageCredit> {
        let mut usage = self.usage.lock();
        let total = usage
            .bytes
            .checked_add(bytes)
            .filter(|total| *total <= self.limits.max_bytes)
            .ok_or_else(|| limit_error("max_bytes", "raw history exceeds its byte limit"))?;
        if usage.segments >= self.limits.max_segments {
            return Err(limit_error(
                "max_segments",
                "raw history exceeds its segment limit",
            ));
        }
        usage.segments += 1;
        usage.bytes = total;
        Ok(StorageCredit {
            ledger: self.clone(),
            bytes,
        })
    }

    pub(super) fn buffer(self: &Arc<Self>, bytes: u64) -> Result<Arc<BufferCredit>> {
        let bytes = usize::try_from(bytes)
            .ok()
            .and_then(|bytes| bytes.checked_mul(2))
            .ok_or_else(|| limit_error("max_buffer_bytes", "raw buffer size overflowed"))?;
        let mut usage = self.usage.lock();
        usage.buffers = usage
            .buffers
            .checked_add(bytes)
            .filter(|total| *total <= self.limits.max_buffer_bytes)
            .ok_or_else(|| {
                limit_error("max_buffer_bytes", "raw buffers exceed their byte limit")
            })?;
        Ok(Arc::new(BufferCredit {
            ledger: self.clone(),
            bytes,
        }))
    }
}

pub(super) struct StorageCredit {
    ledger: Arc<Ledger>,
    bytes: u64,
}

impl Drop for StorageCredit {
    fn drop(&mut self) {
        let mut usage = self.ledger.usage.lock();
        usage.bytes -= self.bytes;
        usage.segments -= 1;
    }
}

pub(super) struct BufferCredit {
    ledger: Arc<Ledger>,
    bytes: usize,
}

impl Drop for BufferCredit {
    fn drop(&mut self) {
        self.ledger.usage.lock().buffers -= self.bytes;
    }
}

pub(super) fn limit_error(field: &str, message: &str) -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: format!("source_history.{field}"),
        message: message.into(),
    }
}
