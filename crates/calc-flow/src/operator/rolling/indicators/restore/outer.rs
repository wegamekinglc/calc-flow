use super::{Result, add, length};
use crate::operator::rolling::{DecodedRollingState, RollingSnapshotMetadata, RollingStreamState};

pub(super) fn arc_bytes(node: &str) -> Result<u64> {
    super::super::footprint_state::packet_allocation_bytes(node)
}
pub(super) fn candidate(
    decoded: DecodedRollingState,
    metadata: RollingSnapshotMetadata,
    next_late: u64,
) -> RollingStreamState {
    RollingStreamState {
        buffer: decoded.buffer,
        ordered: crate::operator::rolling::ordered_stream::OrderedStreamBuffer::default(),
        histories: decoded.histories,
        last_input_watermark: metadata.last_input_watermark,
        next_output_sequence: metadata.next_output_sequence,
        next_late_output_sequence: next_late,
        late_output_failed: false,
        ended: metadata.ended,
        metrics: metadata.metrics,
        pipeline_fingerprint: metadata.pipeline_fingerprint,
        operator_id: metadata.operator_id,
        last_checkpoint_epoch: Some(metadata.epoch),
        typed_kernel_state: None,
        charge: crate::operator::rolling::budget::StateCharge::default(),
    }
}
pub(super) fn ids(metadata: &RollingSnapshotMetadata) -> Result<u64> {
    let graph = metadata
        .pipeline_fingerprint
        .as_ref()
        .map_or(0, String::len);
    let node = metadata.operator_id.as_ref().map_or(0, String::len);
    add(length(graph)?, length(node)?)
}
