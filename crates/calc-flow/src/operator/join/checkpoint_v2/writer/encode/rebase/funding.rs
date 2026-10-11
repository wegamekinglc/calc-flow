use super::super::{budget, bulk_bytes};
use super::{HistoryInput, IndexInput, PayloadEntry, PayloadInput, StateSegment, error};
use crate::Result;

pub(super) fn inputs(indexes: usize, payloads: usize) -> Result<usize> {
    super::accounting::sum(&[
        bulk_bytes::<IndexInput>(indexes)?,
        bulk_bytes::<PayloadInput>(payloads)?,
        super::accounting::product(super::accounting::add(indexes, payloads)?, 64)?,
    ])
}

pub(super) fn output(history: &HistoryInput, check: &dyn Fn() -> Result<()>) -> Result<usize> {
    let mut bytes = super::accounting::sum(&[
        budget::tree::<String, StateSegment>(super::accounting::add(history.payloads.len(), 2)?)?,
        bulk_bytes::<PayloadEntry>(history.payloads.len())?,
        "left-base".len() + "right-base".len(),
    ])?;
    for payload in &history.payloads {
        check()?;
        bytes = super::accounting::sum(&[
            bytes,
            payload.side.as_str().len(),
            "-payload-".len(),
            3 * 64,
        ])?;
    }
    Ok(bytes)
}

pub(super) fn vector<T>(count: usize) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| error("V2 rebase control allocation failed"))?;
    if values.capacity() > count {
        return Err(error("V2 rebase vector exceeded paid capacity"));
    }
    Ok(values)
}

pub(super) fn text(parts: &[&str]) -> Result<String> {
    let length = parts
        .iter()
        .try_fold(0, |length, part| super::accounting::add(length, part.len()))?;
    let mut text = String::new();
    text.try_reserve_exact(length)
        .map_err(|_| error("V2 rebase name allocation failed"))?;
    if text.capacity() > length {
        return Err(error("V2 rebase name exceeded paid capacity"));
    }
    for part in parts {
        text.push_str(part);
    }
    Ok(text)
}
