use std::collections::{BTreeMap, BTreeSet};

use serde::{Serialize, Serializer, ser::SerializeMap};
use serde_json::Value;

use super::{CursorManifestEntry, SourceManifestEntry, SourceWatermarkManifestState};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct SourceEntries {
    entries: BTreeMap<String, SourceManifestEntry>,
    omitted_history: BTreeSet<String>,
}

impl SourceEntries {
    pub(super) const fn new(
        entries: BTreeMap<String, SourceManifestEntry>,
        omitted_history: BTreeSet<String>,
    ) -> Self {
        Self {
            entries,
            omitted_history,
        }
    }

    pub(super) const fn entries(&self) -> &BTreeMap<String, SourceManifestEntry> {
        &self.entries
    }
}

impl Serialize for SourceEntries {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.entries.len()))?;
        for (id, source) in &self.entries {
            map.serialize_entry(
                id,
                &SourceEntry {
                    source,
                    omit_history: self.omitted_history.contains(id),
                },
            )?;
        }
        map.end()
    }
}

struct SourceEntry<'a> {
    source: &'a SourceManifestEntry,
    omit_history: bool,
}

impl Serialize for SourceEntry<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if !self.omit_history {
            return self.source.serialize(serializer);
        }
        LegacySourceEntry {
            cursor: &self.source.cursor,
            identity_hash: &self.source.identity_hash,
            sequence: self.source.sequence,
            ended: self.source.ended,
            watermark_policy: &self.source.watermark_policy,
        }
        .serialize(serializer)
    }
}

#[derive(Serialize)]
struct LegacySourceEntry<'a> {
    cursor: &'a Option<CursorManifestEntry>,
    identity_hash: &'a str,
    sequence: u64,
    ended: bool,
    watermark_policy: &'a SourceWatermarkManifestState,
}

pub(super) fn normalize_legacy_sources(document: &mut Value) -> BTreeSet<String> {
    let mut omitted = BTreeSet::new();
    let Some(sources) = document.get_mut("sources").and_then(Value::as_object_mut) else {
        return omitted;
    };
    for (id, source) in sources {
        if let Some(fields) = source.as_object_mut()
            && !fields.contains_key("history")
        {
            fields.insert("history".into(), Value::Null);
            omitted.insert(id.clone());
        }
    }
    omitted
}
