use super::{BatchKey, mismatch};
use crate::Result;

pub(super) fn batch_segment(key: BatchKey) -> String {
    format!("asof-batch-{}-{}", key.0, key.1)
}

pub(super) fn parse_batch_segment(name: &str) -> Result<BatchKey> {
    let Some(rest) = name.strip_prefix("asof-batch-") else {
        return Err(mismatch("unexpected ASOF segment name"));
    };
    let Some((side, id)) = rest.split_once('-') else {
        return Err(mismatch("malformed ASOF batch segment name"));
    };
    let side = side
        .parse::<u8>()
        .map_err(|_| mismatch("invalid ASOF batch side"))?;
    let id = id
        .parse::<u64>()
        .map_err(|_| mismatch("invalid ASOF batch id"))?;
    if side > 1 || batch_segment((side, id)) != name {
        return Err(mismatch("noncanonical ASOF batch segment name"));
    }
    Ok((side, id))
}
