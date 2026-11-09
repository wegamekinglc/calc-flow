use std::io::{self, Write};

use crate::{CalcFlowError, Result};

type Check<'a> = &'a mut dyn FnMut() -> Result<()>;

pub(super) struct EncodedOutput<'a> {
    bytes: Vec<u8>,
    quota: usize,
    node: &'a str,
    check: Check<'a>,
    failure: Option<CalcFlowError>,
}
impl<'a> EncodedOutput<'a> {
    pub(super) fn new(quota: u64, node: &'a str, check: Check<'a>) -> Result<Self> {
        check()?;
        let quota = usize::try_from(quota).map_err(|_| invalid(node))?;
        if quota > isize::MAX as usize {
            return Err(invalid(node));
        }
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(quota).map_err(|_| invalid(node))?;
        Ok(Self {
            bytes,
            quota,
            node,
            check,
            failure: None,
        })
    }
    pub(super) fn capacity(&self) -> usize {
        self.bytes.capacity()
    }
    pub(super) fn finish(self, completed: bool) -> Result<Vec<u8>> {
        if let Some(error) = self.failure {
            return Err(error);
        }
        if !completed {
            return Err(invalid(self.node));
        }
        (self.check)()?;
        Ok(self.bytes)
    }
    fn poll(&mut self) -> io::Result<()> {
        if self.failure.is_some() {
            return Err(stopped());
        }
        if let Err(error) = (self.check)() {
            return self.fail(error);
        }
        Ok(())
    }
    fn fail<T>(&mut self, error: CalcFlowError) -> io::Result<T> {
        self.failure = Some(error);
        Err(stopped())
    }
    fn admit(&mut self, length: usize) -> io::Result<()> {
        if self
            .bytes
            .len()
            .checked_add(length)
            .is_none_or(|total| total > self.quota)
        {
            return self.fail(invalid(self.node));
        }
        Ok(())
    }
}
impl Write for EncodedOutput<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.poll()?;
        self.admit(bytes.len())?;
        for chunk in bytes.chunks(4096) {
            self.poll()?;
            self.bytes.extend_from_slice(chunk);
        }
        self.poll()?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        self.poll()
    }
}
fn stopped() -> io::Error {
    io::Error::other("rolling encoded output stopped")
}
fn invalid(node: &str) -> CalcFlowError {
    CalcFlowError::Operator {
        node_id: node.into(),
        message: "state.encoded_bytes exceeds the prepared output quota or is invalid".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CalcFlowError;
    use std::io::Write;

    #[test]
    fn test_encoded_quota_rejects_before_copy_even_with_spare_capacity() {
        let input = vec![7_u8; 9];
        let mut check = || Ok(());
        let mut output = EncodedOutput::new(8, "features", &mut check).unwrap();
        output.bytes.reserve_exact(16);
        let capacity = output.capacity();
        assert!(output.write_all(&input).is_err());
        assert_eq!(output.bytes.len(), 0);
        assert_eq!(output.capacity(), capacity);
        assert_eq!(input, vec![7_u8; 9]);
        let error = output.finish(false).unwrap_err();
        assert!(
            matches!(error, CalcFlowError::Operator { node_id, message } if node_id == "features" && message.contains("state.encoded_bytes"))
        );
        let mut control_check = || Ok(());
        let mut control = EncodedOutput::new(8, "features", &mut control_check).unwrap();
        control.write_all(&[1, 2]).unwrap();
        control.write_all(&[3; 6]).unwrap();
        assert_eq!(control.finish(true).unwrap(), [1, 2, 3, 3, 3, 3, 3, 3]);
    }

    #[test]
    fn test_cancellation_interrupts_one_long_encoded_write_and_preserves_identity() {
        let input = vec![7_u8; 8193];
        let mut calls = 0;
        let mut check = || {
            calls += 1;
            if calls >= 4 {
                Err(CalcFlowError::Cancelled {
                    run_id: "encode".into(),
                })
            } else {
                Ok(())
            }
        };
        let mut output = EncodedOutput::new(8193, "features", &mut check).unwrap();
        assert!(output.write_all(&input).is_err());
        assert!(output.bytes.len() < input.len());
        let error = output.finish(false).unwrap_err();
        assert!(matches!(error, CalcFlowError::Cancelled { run_id } if run_id == "encode"));
        assert!(calls >= 4);
        assert_eq!(input, vec![7_u8; 8193]);
    }
}
