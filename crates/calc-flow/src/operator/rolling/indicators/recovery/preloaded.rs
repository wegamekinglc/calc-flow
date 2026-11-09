//! Fund native metadata and its containing snapshot before constructing either.

use std::{any::Any, collections::BTreeMap, sync::Arc};

use super::super::{math_requirements::bytes, restore::metadata};
use super::{IndicatorRollingRecord, Resources, RollingSnapshotMetadata};
use crate::{JsonMap, OperatorStateSnapshot, Result, StateSegment, state::PreloadedSnapshot};

type Segment<'a> = Option<(&'a str, StateSegment)>;
type Receipt = Arc<dyn Any + Send + Sync>;

impl IndicatorRollingRecord {
    pub(crate) fn preloaded_snapshot(
        &self,
        fields: &JsonMap,
        segment: Segment<'_>,
    ) -> Result<PreloadedSnapshot> {
        let Some(resources) = &self.resources else {
            return Ok(PreloadedSnapshot::uncredited(OperatorStateSnapshot {
                inline_metadata: fields.clone(),
                segments: segments(segment),
            }));
        };
        self.preloaded_credited(fields, segment, resources)
    }

    fn preloaded_credited(
        &self,
        fields: &JsonMap,
        segment: Segment<'_>,
        resources: &Resources,
    ) -> Result<PreloadedSnapshot> {
        self.validate_preload(fields, segment.as_ref())?;
        let bytes = self.preload_requirement(fields, segment.as_ref().map(|(id, _)| *id))?;
        let receipt: Receipt = Arc::new(self.metadata_credit(resources, bytes)?);
        self.build_preloaded(fields, segment, receipt)
    }

    fn validate_preload(
        &self,
        fields: &JsonMap,
        segment: Option<&(&str, StateSegment)>,
    ) -> Result<()> {
        let metadata = self.metadata(fields)?;
        self.validate_preload_segment(&metadata, segment)
    }

    fn validate_preload_segment(
        &self,
        metadata: &RollingSnapshotMetadata,
        segment: Option<&(&str, StateSegment)>,
    ) -> Result<()> {
        match (metadata.segment_inventory.first(), segment) {
            (None, None) => Ok(()),
            (Some(descriptor), Some((id, segment))) => {
                self.validate_preload_payload(&descriptor.handle, id, segment)
            }
            _ => Err(self.mismatch("segments", "preloaded base presence differs")),
        }
    }

    fn validate_preload_payload(
        &self,
        handle: &crate::StateHandle,
        id: &str,
        segment: &StateSegment,
    ) -> Result<()> {
        if handle.segment_id() != id
            || Some(handle.byte_len()) != u64::try_from(segment.bytes().len()).ok()
            || handle.sha256() != segment.sha256()
        {
            return Err(self.mismatch("segments", "preloaded base identity or bytes differ"));
        }
        Ok(())
    }

    fn preload_requirement(&self, fields: &JsonMap, id: Option<&str>) -> Result<u64> {
        bytes::total([
            metadata::inline_requirement(fields, &mut || self.control.check_cancelled()),
            metadata::segment_container_requirement(id),
            Ok(PreloadedSnapshot::allocation_bytes()),
        ])
        .map_err(|error| self.metadata_error(error))
    }

    fn build_preloaded(
        &self,
        fields: &JsonMap,
        segment: Segment<'_>,
        receipt: Receipt,
    ) -> Result<PreloadedSnapshot> {
        let snapshot = OperatorStateSnapshot {
            inline_metadata: metadata::copy_inline(fields, &mut || self.control.check_cancelled())?,
            segments: segments(segment),
        };
        let prepared = PreloadedSnapshot::new(snapshot, Some(receipt));
        self.control.check_cancelled()?;
        Ok(prepared)
    }
}

fn segments(segment: Segment<'_>) -> BTreeMap<String, StateSegment> {
    segment.map_or_else(BTreeMap::new, |(id, data)| {
        BTreeMap::from([(id.to_owned(), data)])
    })
}
