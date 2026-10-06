use super::BatchKey;

pub(super) fn batch_segment(key: BatchKey) -> String {
    format!("asof-batch-{}-{}", key.0, key.1)
}
