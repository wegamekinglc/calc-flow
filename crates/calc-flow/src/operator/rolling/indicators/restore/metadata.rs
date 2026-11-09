use serde::Deserialize;
use serde::de::value::MapDeserializer;

use super::{Check, Result, add, mismatch};
use crate::operator::rolling::{CompiledRollingSpec, RollingSnapshotMetadata, state_codec};
use crate::state::SegmentDescriptor;
use crate::{JsonMap, OperatorStateSnapshot};
use serde_json::Value;

mod inline;
mod shape;
pub(in super::super) use inline::{
    copy as copy_inline, requirement as inline_requirement,
    segment_requirement as segment_container_requirement,
};
#[derive(Clone, Copy)]
pub(in super::super) struct Quote {
    pub(in super::super) retained: u64,
    pub(in super::super) peak: u64,
}
pub(super) fn quote(snapshot: &OperatorStateSnapshot, check: Check<'_>) -> Result<Quote> {
    if snapshot.segments.len() > 1 {
        return Err(mismatch("segments", "requires zero or one base segment"));
    }
    let facts = shape::plan(&snapshot.inline_metadata, check)?;
    build_quote(&facts, snapshot.segments.keys())
}
pub(in super::super) fn quote_entries<'a>(
    fields: impl Iterator<Item = (&'a String, &'a Value)>,
    check: Check<'_>,
) -> Result<Quote> {
    let facts = shape::plan_entries(fields, check)?;
    build_quote(&facts, std::iter::empty())
}
fn build_quote<'a>(
    facts: &shape::Facts,
    keys: impl ExactSizeIterator<Item = &'a String>,
) -> Result<Quote> {
    let descriptors = super::layout::<SegmentDescriptor>(facts.descriptors)?;
    let retained = metadata_bytes(facts.strings, descriptors)?;
    let validation = validation(keys, facts, descriptors)?;
    let scratch = construction_scratch(facts, validation)?;
    Ok(Quote {
        retained,
        peak: add(retained, scratch)?,
    })
}
fn construction_scratch(facts: &shape::Facts, validation: u64) -> Result<u64> {
    let path = super::super::state_decode_requirements::vector_construction::<&str>(
        facts.path_components,
    )?;
    Ok(validation.max(facts.handle_strings).max(path))
}

fn metadata_bytes(strings: u64, descriptors: u64) -> Result<u64> {
    add(
        add(strings, descriptors)?,
        super::layout::<RollingSnapshotMetadata>(1)?,
    )
}

fn validation<'a>(
    keys: impl ExactSizeIterator<Item = &'a String>,
    facts: &shape::Facts,
    descriptors: u64,
) -> Result<u64> {
    let copied = add(facts.descriptor_strings, descriptors)?;
    let inventories = add(copied, copied)?;
    let copied = add(inventories, key_copies(keys, facts)?)?;
    add(copied, inventory_workspace(facts)?)
}
fn inventory_workspace(facts: &shape::Facts) -> Result<u64> {
    let structures = inventory_sets(facts)?;
    let path = super::super::state_decode_requirements::vector_construction::<&str>(
        facts.path_components,
    )?;
    add(
        structures,
        add(path, expected_id_workspace(facts.descriptors)?)?,
    )
}
fn inventory_sets(facts: &shape::Facts) -> Result<u64> {
    use super::super::math_requirements::bytes;
    let identity = bytes::tree_nodes::<(String, crate::Epoch, String), ()>(facts.descriptors)?;
    let paths = bytes::tree_nodes::<String, ()>(facts.descriptors)?;
    add(add(identity, paths)?, facts.handle_strings)
}
fn expected_id_workspace(descriptors: usize) -> Result<u64> {
    if descriptors == 0 {
        return Ok(0);
    }
    super::super::state_decode_requirements::vector_construction::<u8>(34)
}
fn key_copies<'a>(
    keys: impl ExactSizeIterator<Item = &'a String>,
    facts: &shape::Facts,
) -> Result<u64> {
    let vectors = super::layout::<String>(keys.len())?;
    let keys = keys
        .into_iter()
        .try_fold(0, |sum, key| add(sum, super::length(key.len())?))?;
    let expected_vectors = super::layout::<String>(facts.descriptors)?;
    id_copies(keys, facts.segment_ids, vectors, expected_vectors)
}

fn id_copies(
    actual: u64,
    expected: u64,
    actual_vectors: u64,
    expected_vectors: u64,
) -> Result<u64> {
    let strings = add(actual, expected)?;
    let vectors = add(actual_vectors, expected_vectors)?;
    add(strings, vectors)
}

pub(super) fn parse(fields: &JsonMap) -> Result<RollingSnapshotMetadata> {
    parse_entries(fields.iter())
}
pub(in super::super) fn parse_entries<'a>(
    fields: impl Iterator<Item = (&'a String, &'a Value)>,
) -> Result<RollingSnapshotMetadata> {
    let entries = fields.map(|(key, value)| (key.as_str(), value));
    RollingSnapshotMetadata::deserialize(MapDeserializer::<_, serde_json::Error>::new(entries))
        .map_err(|_| mismatch("metadata", "invalid inline metadata"))
}
pub(super) fn validate(
    metadata: &RollingSnapshotMetadata,
    compiled: &CompiledRollingSpec,
    snapshot: &OperatorStateSnapshot,
) -> Result<()> {
    if metadata.state_layout_version != 5 {
        return Err(mismatch("metadata", "indicator state requires layout five"));
    }
    state_codec::validate_snapshot_metadata(metadata, compiled, snapshot)?;
    if metadata.segment_inventory.is_empty() && metadata.next_output_sequence > 0 {
        return Err(mismatch(
            "segments",
            "emitted indicator rows require entity state",
        ));
    }
    Ok(())
}
