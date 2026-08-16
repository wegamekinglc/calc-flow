//! Transport-orthogonal format codec contracts.
//!
//! Formats decode bytes into immutable [`Batch`] values and encode batches
//! back to bytes; transports select formats by identity instead of bundling
//! codecs. Decoding is bounded: expansion beyond row and byte limits fails
//! before any batch can reach an edge.

use crate::batch::Batch;
use crate::config::ArrowFieldSpec;
use crate::connector::capability::FormatIdentity;
use crate::{CalcFlowError, Result};

/// Hard row and byte limits for one decode expansion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeBounds {
    /// Maximum rows one decoded batch may carry.
    pub max_rows: u64,
    /// Maximum estimated bytes one decoded batch may carry.
    pub max_bytes: u64,
}

impl DecodeBounds {
    /// Builds non-zero bounds.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when either limit is
    /// zero.
    pub fn new(max_rows: u64, max_bytes: u64) -> Result<Self> {
        if max_rows == 0 {
            return Err(CalcFlowError::InvalidArgument {
                field: "max_rows".into(),
                message: "decode row limit must be greater than zero".into(),
            });
        }
        if max_bytes == 0 {
            return Err(CalcFlowError::InvalidArgument {
                field: "max_bytes".into(),
                message: "decode byte limit must be greater than zero".into(),
            });
        }
        Ok(Self {
            max_rows,
            max_bytes,
        })
    }

    /// Checks one expansion against the bounds.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] naming the codec
    /// identity and the violated bound when the expansion is oversized.
    pub fn check(&self, identity: &FormatIdentity, rows: u64, bytes: u64) -> Result<()> {
        if rows > self.max_rows {
            return Err(CalcFlowError::InvalidArgument {
                field: format!("format {}/{} rows", identity.name, identity.version),
                message: format!("decoded {rows} rows exceed the {} row limit", self.max_rows),
            });
        }
        if bytes > self.max_bytes {
            return Err(CalcFlowError::InvalidArgument {
                field: format!("format {}/{} bytes", identity.name, identity.version),
                message: format!(
                    "decoded {bytes} bytes exceed the {} byte limit",
                    self.max_bytes
                ),
            });
        }
        Ok(())
    }
}

/// Decodes bytes from a transport into immutable batches.
pub trait FormatDecoder: Send + Sync {
    /// Identity of the codec.
    fn identity(&self) -> FormatIdentity;

    /// Decodes one payload within `bounds`.
    ///
    /// Decoding is synchronous CPU-local work; transports read bytes
    /// asynchronously and hand the buffer to the codec. The optional
    /// schema is the full ordered field list the payload must match;
    /// codecs without an explicit schema infer one from the payload.
    ///
    /// # Errors
    ///
    /// Returns a safe error, failing oversized expansion before returning
    /// a batch, with the codec identity and violated bound preserved.
    fn decode(
        &self,
        bytes: &[u8],
        bounds: &DecodeBounds,
        schema: &[ArrowFieldSpec],
    ) -> Result<Batch>;
}

/// Encodes immutable batches for a transport.
pub trait FormatEncoder: Send + Sync {
    /// Identity of the codec.
    fn identity(&self) -> FormatIdentity;

    /// Encodes one batch.
    ///
    /// # Errors
    ///
    /// Returns a safe error when the batch cannot be represented in the
    /// format.
    fn encode(&self, batch: &Batch) -> Result<Vec<u8>>;
}
