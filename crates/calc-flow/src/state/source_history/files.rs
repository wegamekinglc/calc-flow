use super::budget::limit_error;
use crate::{CalcFlowError, Result};
use std::path::{Path, PathBuf};

fn metadata(path: &Path) -> Result<std::fs::Metadata> {
    std::fs::symlink_metadata(path).map_err(|source| CalcFlowError::Io {
        path: path.display().to_string(),
        source,
    })
}

fn regular_file(path: &Path, extension: &str) -> Result<()> {
    let metadata = metadata(path)?;
    if !metadata.is_file()
        || metadata.file_type().is_symlink()
        || path.extension().and_then(|value| value.to_str()) != Some(extension)
    {
        return Err(limit_error(
            "path",
            "history requires regular files of the declared format",
        ));
    }
    Ok(())
}

pub(super) fn discover(path: &Path, extension: &str, max_files: usize) -> Result<Vec<PathBuf>> {
    let metadata = metadata(path)?;
    if metadata.file_type().is_symlink() {
        return Err(limit_error(
            "path",
            "history requires a regular file or directory",
        ));
    }
    if !metadata.is_dir() {
        regular_file(path, extension)?;
        return Ok(vec![path.to_path_buf()]);
    }
    directory_files(path, extension, max_files)
}

fn directory_files(path: &Path, extension: &str, max_files: usize) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(path).map_err(|source| CalcFlowError::Io {
        path: path.display().to_string(),
        source,
    })? {
        let entry = entry.map_err(|source| CalcFlowError::Io {
            path: path.display().to_string(),
            source,
        })?;
        let entry_path = entry.path();
        regular_file(&entry_path, extension)?;
        if files.len() >= max_files {
            return Err(limit_error(
                "max_segments",
                "history exceeds its file limit",
            ));
        }
        files.push(entry_path);
    }
    files.sort();
    Ok(files)
}
