use std::{fs::File, io::Read as _, path::PathBuf, sync::Arc};

use super::{
    LocalStateLineageBackend, StateHandle, io_error, segment_metadata, segment_mismatch,
    validate_segment_metadata,
};
use crate::{CalcFlowError, Result, StateSegment};

pub(crate) struct NativeSegmentReader {
    _lineage: Arc<LocalStateLineageBackend>,
    path: PathBuf,
    handle: StateHandle,
    length: usize,
    #[cfg(test)]
    hook: Option<super::native_io::Hook>,
}

impl LocalStateLineageBackend {
    pub(crate) fn native_reader(
        self: &Arc<Self>,
        handle: &StateHandle,
    ) -> Result<NativeSegmentReader> {
        if !self.supports_prepaid_load() {
            return Err(reader_error("requires a certified local segment reader"));
        }
        let length = usize::try_from(handle.byte_len())
            .map_err(|_| reader_error("segment byte length exceeds usize"))?;
        let paths = self.managed_paths(handle)?;
        Ok(NativeSegmentReader {
            _lineage: Arc::clone(self),
            path: paths.committed,
            handle: handle.clone(),
            length,
            #[cfg(test)]
            hook: self.native_io_hook.lock().clone(),
        })
    }
}

impl NativeSegmentReader {
    pub(crate) const fn byte_len(&self) -> usize {
        self.length
    }

    pub(crate) fn read(
        self,
        bytes: Vec<u8>,
        scratch: Vec<u8>,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<StateSegment> {
        #[cfg(test)]
        if let Some(hook) = &self.hook {
            hook(super::native_io::Point::Read)?;
        }
        let bytes = self.read_bytes(bytes, scratch, check)?;
        let segment = StateSegment::new_controlled(bytes, check)?;
        if segment.sha256() != self.handle.sha256() {
            return Err(segment_mismatch(&self.handle, "checksum"));
        }
        check()?;
        Ok(segment)
    }

    fn read_bytes(
        &self,
        mut bytes: Vec<u8>,
        mut scratch: Vec<u8>,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Vec<u8>> {
        check()?;
        self.destination(&bytes, &scratch)?;
        let mut file = self.open()?;
        self.fill(&mut file, &mut bytes, &mut scratch, check)?;
        self.end(&mut file, check)?;
        Ok(bytes)
    }
    fn destination(&self, bytes: &Vec<u8>, scratch: &[u8]) -> Result<()> {
        if !bytes.is_empty() || bytes.capacity() < self.length {
            return Err(reader_error(
                "destination must be an empty precharged buffer",
            ));
        }
        if self.length != 0 && scratch.is_empty() {
            return Err(reader_error("requires a precharged read buffer"));
        }
        Ok(())
    }
    fn open(&self) -> Result<File> {
        let metadata = segment_metadata(&self.path, &self.handle)?;
        validate_segment_metadata(&self.path, &self.handle, &metadata)?;
        File::open(&self.path).map_err(|source| io_error(&self.path, source))
    }
    fn fill(
        &self,
        file: &mut File,
        bytes: &mut Vec<u8>,
        scratch: &mut [u8],
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        while bytes.len() < self.length {
            check()?;
            let count = (self.length - bytes.len()).min(scratch.len()).min(4096);
            file.read_exact(&mut scratch[..count])
                .map_err(|source| io_error(&self.path, source))?;
            bytes.extend_from_slice(&scratch[..count]);
        }
        Ok(())
    }
    fn end(&self, file: &mut File, check: &mut dyn FnMut() -> Result<()>) -> Result<()> {
        check()?;
        let mut extra = [0_u8; 1];
        if file
            .read(&mut extra)
            .map_err(|source| io_error(&self.path, source))?
            != 0
        {
            return Err(segment_mismatch(&self.handle, "byte length"));
        }
        check()
    }
}

fn reader_error(message: &str) -> CalcFlowError {
    CalcFlowError::CheckpointMismatch {
        message: format!("native:array_sequence@1 state.reader: {message}"),
    }
}
