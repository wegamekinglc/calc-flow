use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{StateHandle, backend::validate_sha256};
use crate::{CalcFlowError, Epoch, Result, canonical_json};

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

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum StateOperation<T> {
    Upsert(T),
    Tombstone,
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

    pub(crate) fn needs_compaction(
        &self,
        max_delta_segments: usize,
        manifest_contribution_budget: usize,
    ) -> Result<bool> {
        if max_delta_segments == 0 {
            return Err(inventory_error(
                "maximum delta segment count must be non-zero",
            ));
        }
        if manifest_contribution_budget == 0 {
            return Err(inventory_error(
                "manifest contribution budget must be non-zero",
            ));
        }
        let delta_count = self
            .segments
            .iter()
            .filter(|descriptor| descriptor.kind == SegmentKind::Delta)
            .count();
        Ok(delta_count > max_delta_segments
            || self.predicted_manifest_bytes()? > manifest_contribution_budget)
    }

    pub(crate) fn replacement_after_full_compaction(
        &self,
        base: SegmentDescriptor,
    ) -> Result<Self> {
        if base.kind != SegmentKind::Base {
            return Err(inventory_error(
                "full compaction replacement must be a base segment",
            ));
        }
        if let Some(first) = self.segments.first()
            && (base.handle.operator_id() != first.handle.operator_id()
                || base.state_layout_version != first.state_layout_version
                || base.schema_fingerprint != first.schema_fingerprint)
        {
            return Err(inventory_error(
                "compaction replacement does not match the retained inventory",
            ));
        }
        if self
            .segments
            .last()
            .is_some_and(|last| base.handle.epoch() < last.handle.epoch())
        {
            return Err(inventory_error(
                "compaction base is older than the retained inventory",
            ));
        }
        Self::new(vec![base])
    }

    fn validate(&self) -> Result<()> {
        let Some(first) = self.segments.first() else {
            return Ok(());
        };
        validate_inventory_header(first)?;
        let operator_id = first.handle.operator_id();
        let layout = first.state_layout_version;
        let schema = &first.schema_fingerprint;
        let mut previous_coordinate = None;
        let mut saw_base = false;
        let mut saw_delta = false;
        let mut identities = BTreeSet::new();
        let mut paths = BTreeSet::new();

        for descriptor in &self.segments {
            validate_descriptor_identity(descriptor, operator_id, layout, schema)?;
            previous_coordinate = validate_descriptor_order(descriptor, previous_coordinate)?;
            record_segment_kind(descriptor.kind, &mut saw_base, &mut saw_delta)?;
            record_descriptor_identity(descriptor, &mut identities, &mut paths)?;
        }
        Ok(())
    }
}

fn validate_inventory_header(first: &SegmentDescriptor) -> Result<()> {
    if first.state_layout_version == 0 {
        return Err(inventory_error("state layout version must be non-zero"));
    }
    validate_sha256("schema_fingerprint", &first.schema_fingerprint)
        .map_err(|error| inventory_error(error.to_string()))
}

fn validate_descriptor_identity(
    descriptor: &SegmentDescriptor,
    operator_id: &str,
    layout: u32,
    schema: &str,
) -> Result<()> {
    if descriptor.handle.operator_id() != operator_id {
        return Err(inventory_error(
            "state inventory contains more than one operator",
        ));
    }
    if descriptor.state_layout_version != layout {
        return Err(inventory_error(
            "state inventory contains more than one layout version",
        ));
    }
    if descriptor.schema_fingerprint != schema {
        return Err(inventory_error(
            "state inventory contains more than one schema fingerprint",
        ));
    }
    descriptor
        .handle
        .validate_for(operator_id, descriptor.handle.epoch())
        .map_err(|error| inventory_error(error.to_string()))
}

fn validate_descriptor_order<'a>(
    descriptor: &'a SegmentDescriptor,
    previous: Option<(Epoch, &'a str)>,
) -> Result<Option<(Epoch, &'a str)>> {
    let coordinate = (descriptor.handle.epoch(), descriptor.handle.segment_id());
    if previous.is_some_and(|value| value >= coordinate) {
        return Err(inventory_error(
            "state inventory segments are not in canonical epoch/segment order",
        ));
    }
    Ok(Some(coordinate))
}

fn record_segment_kind(kind: SegmentKind, saw_base: &mut bool, saw_delta: &mut bool) -> Result<()> {
    match kind {
        SegmentKind::Base if *saw_base => Err(inventory_error(
            "state inventory contains more than one base segment",
        )),
        SegmentKind::Base if *saw_delta => Err(inventory_error(
            "state inventory contains a delta older than its base",
        )),
        SegmentKind::Base => {
            *saw_base = true;
            Ok(())
        }
        SegmentKind::Delta => {
            *saw_delta = true;
            Ok(())
        }
    }
}

fn record_descriptor_identity(
    descriptor: &SegmentDescriptor,
    identities: &mut BTreeSet<(String, Epoch, String)>,
    paths: &mut BTreeSet<String>,
) -> Result<()> {
    let identity = (
        descriptor.handle.operator_id().into(),
        descriptor.handle.epoch(),
        descriptor.handle.segment_id().into(),
    );
    if !identities.insert(identity) {
        return Err(inventory_error(
            "state inventory contains a duplicate handle identity",
        ));
    }
    if !paths.insert(descriptor.handle.relative_path().into()) {
        return Err(inventory_error(
            "state inventory contains a duplicate committed path",
        ));
    }
    Ok(())
}

pub(crate) fn fold_state_segments<K, V>(
    segments: impl IntoIterator<Item = Vec<(K, StateOperation<V>)>>,
) -> Result<BTreeMap<K, V>>
where
    K: Clone + Ord,
{
    let mut state = BTreeMap::new();
    for segment in segments {
        let mut seen = BTreeSet::new();
        for (key, operation) in segment {
            if !seen.insert(key.clone()) {
                return Err(inventory_error(
                    "state segment contains duplicate operations for one key",
                ));
            }
            match operation {
                StateOperation::Upsert(value) => {
                    state.insert(key, value);
                }
                StateOperation::Tombstone => {
                    state.remove(&key);
                }
            }
        }
    }
    Ok(state)
}

fn inventory_error(message: impl Into<String>) -> CalcFlowError {
    CalcFlowError::Internal {
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use proptest::{collection, prelude::*};

    use super::{
        SegmentDescriptor, SegmentKind, StateInventory, StateOperation, fold_state_segments,
    };
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
        duplicate_path.handle = StateHandle::new(
            duplicate_path.handle.operator_id(),
            duplicate_path.handle.epoch(),
            duplicate_path.handle.segment_id(),
            first.handle.relative_path(),
            duplicate_path.handle.byte_len(),
            duplicate_path.handle.sha256(),
        )
        .unwrap();
        assert!(matches!(
            StateInventory::new(vec![first, duplicate_path]),
            Err(CalcFlowError::Internal { .. })
        ));
    }

    #[test]
    fn compaction_triggers_before_configured_bounds_and_replaces_immutably() {
        let original = StateInventory::new(vec![
            descriptor(SegmentKind::Base, 1, "base"),
            descriptor(SegmentKind::Delta, 2, "delta-0001"),
            descriptor(SegmentKind::Delta, 3, "delta-0002"),
        ])
        .unwrap();
        let predicted = original.predicted_manifest_bytes().unwrap();

        assert!(!original.needs_compaction(2, predicted).unwrap());
        assert!(original.needs_compaction(1, predicted).unwrap());
        assert!(original.needs_compaction(2, predicted - 1).unwrap());
        assert!(original.needs_compaction(0, predicted).is_err());
        assert!(original.needs_compaction(2, 0).is_err());

        let replacement = original
            .replacement_after_full_compaction(descriptor(SegmentKind::Base, 3, "base-compact"))
            .unwrap();
        assert_eq!(replacement.segments().len(), 1);
        assert_eq!(original.segments().len(), 3);
        assert_eq!(original.segments()[0].handle.segment_id(), "base");
    }

    #[test]
    fn last_operation_wins_and_duplicate_operations_fail_closed() {
        let state = fold_state_segments(vec![
            vec![
                (b"alpha".to_vec(), StateOperation::Upsert(1_u64)),
                (b"beta".to_vec(), StateOperation::Upsert(2)),
            ],
            vec![
                (b"alpha".to_vec(), StateOperation::Upsert(3)),
                (b"beta".to_vec(), StateOperation::Tombstone),
            ],
        ])
        .unwrap();
        assert_eq!(
            state.into_iter().collect::<Vec<_>>(),
            vec![(b"alpha".to_vec(), 3)]
        );

        assert!(matches!(
            fold_state_segments(vec![vec![
                (b"duplicate".to_vec(), StateOperation::Upsert(1_u64)),
                (b"duplicate".to_vec(), StateOperation::Tombstone),
            ]]),
            Err(CalcFlowError::Internal { .. })
        ));
    }

    proptest! {
        #[test]
        fn compaction_fold_matches_an_independent_ordered_model(
            segments in collection::vec(
                collection::btree_map(0_u8..24, prop::option::of(any::<i16>()), 0..20),
                0..20,
            )
        ) {
            let operations = segments
                .iter()
                .map(|segment| {
                    segment
                        .iter()
                        .map(|(key, value)| {
                            let operation = value.map_or(
                                StateOperation::Tombstone,
                                StateOperation::Upsert,
                            );
                            (*key, operation)
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();

            let mut expected = BTreeMap::new();
            for segment in segments {
                for (key, value) in segment {
                    if let Some(value) = value {
                        expected.insert(key, value);
                    } else {
                        expected.remove(&key);
                    }
                }
            }

            prop_assert_eq!(fold_state_segments(operations).unwrap(), expected);
        }
    }
}
