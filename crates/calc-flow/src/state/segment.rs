use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{StateHandle, backend::validate_sha256};
use crate::{CalcFlowError, Result, canonical_json};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SegmentKind {
    Base,
    Delta,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SegmentDescriptor {
    pub(crate) kind: SegmentKind,
    pub(crate) state_layout_version: u32,
    pub(crate) schema_fingerprint: String,
    pub(crate) handle: StateHandle,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct StateInventory {
    segments: Vec<SegmentDescriptor>,
}

impl StateInventory {
    pub(crate) fn new(segments: Vec<SegmentDescriptor>) -> Result<Self> {
        let inventory = Self { segments };
        inventory.validate()?;
        Ok(inventory)
    }

    pub(crate) fn segments(&self) -> &[SegmentDescriptor] {
        &self.segments
    }

    pub(crate) fn predicted_manifest_bytes(&self) -> Result<usize> {
        let value =
            serde_json::to_value(&self.segments).map_err(|error| CalcFlowError::Format {
                message: error.to_string(),
            })?;
        canonical_json(&Value::Object(
            [("segments".into(), value)].into_iter().collect(),
        ))
        .map(|document| document.len())
    }

    fn validate(&self) -> Result<()> {
        let Some(first) = self.segments.first() else {
            return Ok(());
        };
        if first.state_layout_version == 0 {
            return Err(inventory_error("state layout version must be non-zero"));
        }
        validate_sha256("schema_fingerprint", &first.schema_fingerprint)
            .map_err(|error| inventory_error(error.to_string()))?;

        let operator_id = &first.handle.operator_id;
        let layout = first.state_layout_version;
        let schema = &first.schema_fingerprint;
        let mut previous_coordinate = None;
        let mut saw_base = false;
        let mut saw_delta = false;
        let mut identities = BTreeSet::new();
        let mut paths = BTreeSet::new();

        for descriptor in &self.segments {
            if descriptor.handle.operator_id != *operator_id {
                return Err(inventory_error(
                    "state inventory contains more than one operator",
                ));
            }
            if descriptor.state_layout_version != layout {
                return Err(inventory_error(
                    "state inventory contains more than one layout version",
                ));
            }
            if descriptor.schema_fingerprint != *schema {
                return Err(inventory_error(
                    "state inventory contains more than one schema fingerprint",
                ));
            }
            descriptor
                .handle
                .validate_for(operator_id, descriptor.handle.epoch)
                .map_err(|error| inventory_error(error.to_string()))?;

            let coordinate = (
                descriptor.handle.epoch,
                descriptor.handle.segment_id.as_str(),
            );
            if previous_coordinate.is_some_and(|previous| previous >= coordinate) {
                return Err(inventory_error(
                    "state inventory segments are not in canonical epoch/segment order",
                ));
            }
            previous_coordinate = Some(coordinate);

            match descriptor.kind {
                SegmentKind::Base => {
                    if saw_base {
                        return Err(inventory_error(
                            "state inventory contains more than one base segment",
                        ));
                    }
                    if saw_delta {
                        return Err(inventory_error(
                            "state inventory contains a delta older than its base",
                        ));
                    }
                    saw_base = true;
                }
                SegmentKind::Delta => saw_delta = true,
            }

            let identity = (
                descriptor.handle.operator_id.clone(),
                descriptor.handle.epoch,
                descriptor.handle.segment_id.clone(),
            );
            if !identities.insert(identity) {
                return Err(inventory_error(
                    "state inventory contains a duplicate handle identity",
                ));
            }
            if !paths.insert(descriptor.handle.relative_path.clone()) {
                return Err(inventory_error(
                    "state inventory contains a duplicate committed path",
                ));
            }
        }
        Ok(())
    }
}

fn inventory_error(message: impl Into<String>) -> CalcFlowError {
    CalcFlowError::Internal {
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::{SegmentDescriptor, SegmentKind, StateInventory};
    use crate::{CalcFlowError, Epoch, StateHandle};

    const SHA256: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn descriptor(kind: SegmentKind, epoch: u64, segment_id: &str) -> SegmentDescriptor {
        SegmentDescriptor {
            kind,
            state_layout_version: 1,
            schema_fingerprint: SHA256.into(),
            handle: StateHandle::new(
                "window",
                Epoch::new(epoch).unwrap(),
                segment_id,
                &format!("committed/pipeline/window/{epoch}-{segment_id}.arrow"),
                42,
                SHA256,
            )
            .unwrap(),
        }
    }

    #[test]
    fn inventory_accepts_one_base_followed_by_canonical_deltas() {
        let inventory = StateInventory::new(vec![
            descriptor(SegmentKind::Base, 1, "base"),
            descriptor(SegmentKind::Delta, 2, "delta-0001"),
            descriptor(SegmentKind::Delta, 3, "delta-0002"),
        ])
        .unwrap();

        assert_eq!(inventory.segments().len(), 3);
        assert!(inventory.predicted_manifest_bytes().unwrap() > 0);
    }

    #[test]
    fn inventory_rejects_order_base_operator_layout_schema_and_path_conflicts() {
        let invalid_order = vec![
            descriptor(SegmentKind::Delta, 2, "delta-0002"),
            descriptor(SegmentKind::Delta, 1, "delta-0001"),
        ];
        assert!(matches!(
            StateInventory::new(invalid_order),
            Err(CalcFlowError::Internal { .. })
        ));

        let multiple_bases = vec![
            descriptor(SegmentKind::Base, 1, "base-0001"),
            descriptor(SegmentKind::Base, 2, "base-0002"),
        ];
        assert!(matches!(
            StateInventory::new(multiple_bases),
            Err(CalcFlowError::Internal { .. })
        ));

        let delta_before_base = vec![
            descriptor(SegmentKind::Delta, 1, "delta"),
            descriptor(SegmentKind::Base, 2, "base"),
        ];
        assert!(matches!(
            StateInventory::new(delta_before_base),
            Err(CalcFlowError::Internal { .. })
        ));

        let mut wrong_operator = descriptor(SegmentKind::Delta, 2, "delta");
        wrong_operator.handle = StateHandle::new(
            "other",
            Epoch::new(2).unwrap(),
            "delta",
            "committed/pipeline/other/2-delta.arrow",
            42,
            SHA256,
        )
        .unwrap();
        assert!(matches!(
            StateInventory::new(vec![
                descriptor(SegmentKind::Base, 1, "base"),
                wrong_operator,
            ]),
            Err(CalcFlowError::Internal { .. })
        ));

        let mut wrong_layout = descriptor(SegmentKind::Delta, 2, "delta");
        wrong_layout.state_layout_version = 2;
        assert!(matches!(
            StateInventory::new(vec![descriptor(SegmentKind::Base, 1, "base"), wrong_layout,]),
            Err(CalcFlowError::Internal { .. })
        ));

        let mut wrong_schema = descriptor(SegmentKind::Delta, 2, "delta");
        wrong_schema.schema_fingerprint =
            "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff".into();
        assert!(matches!(
            StateInventory::new(vec![descriptor(SegmentKind::Base, 1, "base"), wrong_schema,]),
            Err(CalcFlowError::Internal { .. })
        ));

        let first = descriptor(SegmentKind::Delta, 1, "first");
        let mut duplicate_path = descriptor(SegmentKind::Delta, 2, "second");
        duplicate_path.handle.relative_path = first.handle.relative_path.clone();
        assert!(matches!(
            StateInventory::new(vec![first, duplicate_path]),
            Err(CalcFlowError::Internal { .. })
        ));
    }
}
