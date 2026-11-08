use super::invalid;
use crate::Result;

/// Per-call limits for native result elements and execution buffers.
///
/// Borrowed input storage is excluded. Allocation accounting includes owned
/// output buffers and conservatively bounded kernel scratch, not process RSS
/// or fixed object/allocator overhead.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use]
pub struct ArrayExecutionOptions {
    max_elements: usize,
    max_allocation_bytes: usize,
}

impl Default for ArrayExecutionOptions {
    fn default() -> Self {
        Self {
            max_elements: 10_000_000,
            max_allocation_bytes: 256 * 1024 * 1024,
        }
    }
}

impl ArrayExecutionOptions {
    /// Returns limits with a new maximum number of output elements.
    ///
    /// Zero rejects every nonempty result, including a reduction scalar.
    pub const fn with_max_elements(self, max_elements: usize) -> Self {
        Self {
            max_elements,
            ..self
        }
    }

    /// Returns limits with a new maximum output-and-scratch byte charge.
    ///
    /// Zero rejects every execution requiring an allocated output buffer.
    pub const fn with_max_allocation_bytes(self, max_allocation_bytes: usize) -> Self {
        Self {
            max_allocation_bytes,
            ..self
        }
    }

    /// Returns the output element limit.
    pub const fn max_elements(&self) -> usize {
        self.max_elements
    }

    /// Returns the combined output-and-scratch byte limit.
    pub const fn max_allocation_bytes(&self) -> usize {
        self.max_allocation_bytes
    }

    pub(super) fn validate_allocation(
        &self,
        elements: usize,
        output_bytes: usize,
        scratch_bytes: usize,
    ) -> Result<()> {
        if elements > self.max_elements {
            return Err(invalid(
                "outputs.output.elements",
                format!(
                    "requires {elements} elements, limit is {}",
                    self.max_elements
                ),
            ));
        }
        if output_bytes > self.max_allocation_bytes {
            return Err(invalid(
                "outputs.output.bytes",
                format!(
                    "requires {output_bytes} output bytes, limit is {}",
                    self.max_allocation_bytes
                ),
            ));
        }
        let peak = output_bytes
            .checked_add(scratch_bytes)
            .ok_or_else(|| invalid("scratch.bytes", "allocation byte count overflowed usize"))?;
        if peak > self.max_allocation_bytes {
            return Err(invalid(
                "scratch.bytes",
                format!(
                    "requires {peak} output and scratch bytes, limit is {}",
                    self.max_allocation_bytes
                ),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::ArrayExecutionOptions;
    use crate::CalcFlowError;

    #[test]
    fn checked_peak_overflow_returns_a_typed_error() {
        let options = ArrayExecutionOptions::default().with_max_allocation_bytes(usize::MAX);
        let error = options.validate_allocation(1, usize::MAX, 1).unwrap_err();
        let CalcFlowError::InvalidArgument { field, .. } = error else {
            panic!("expected allocation overflow error");
        };
        assert_eq!(field, "scratch.bytes");
    }

    #[test]
    fn equal_allocation_boundary_is_admitted() {
        let options = ArrayExecutionOptions::default().with_max_allocation_bytes(16);
        options.validate_allocation(1, 8, 8).unwrap();
        assert!(options.validate_allocation(1, 8, 9).is_err());
        let zero = options.with_max_elements(0).with_max_allocation_bytes(0);
        zero.validate_allocation(0, 0, 0).unwrap();
        assert!(zero.validate_allocation(1, 0, 0).is_err());
    }
}
