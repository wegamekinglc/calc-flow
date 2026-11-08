use super::source;
use crate::{Result, STATIC_INPUT_DIGEST_VERSION};

#[derive(Clone, Copy)]
pub(crate) enum Mode {
    Finite,
    Managed {
        checkpointing: bool,
        context_copies: usize,
    },
}
impl Mode {
    pub(super) fn alias_bytes(self, name: &str) -> Result<usize> {
        match self {
            Self::Finite => checked_alias(name.len(), 1, 1, 0),
            Self::Managed {
                checkpointing,
                context_copies,
            } => managed_bytes(name.len(), checkpointing, context_copies),
        }
    }
}
fn managed_bytes(name: usize, checkpointing: bool, contexts: usize) -> Result<usize> {
    let (names, digests, fields) = managed_copies(checkpointing, contexts)?;
    let active = checked_alias(name, names, digests, fields)?;
    let checkpoint = usize::from(checkpointing);
    let before_install = checked_alias(
        name,
        5 + 3 * checkpoint,
        3 + 3 * checkpoint,
        (1 + checkpoint) * ("digest_version".len() + "sha256".len()),
    )?;
    Ok(active.max(before_install))
}
fn managed_copies(checkpointing: bool, contexts: usize) -> Result<(usize, usize, usize)> {
    if contexts == 0 {
        return Err(source::invalid(
            "bytes",
            "managed native preparation requires an actual job context",
        ));
    }
    let checkpoint = usize::from(checkpointing);
    let names = contexts.checked_add(2 + 4 * checkpoint);
    let digests = contexts.checked_add(1 + 4 * checkpoint);
    let fields = contexts
        .checked_add(checkpoint)
        .and_then(|copies| copies.checked_mul("digest_version".len() + "sha256".len()));
    names
        .zip(digests)
        .zip(fields)
        .map(|((names, digests), fields)| (names, digests, fields))
        .ok_or_else(|| source::invalid("bytes", "managed context descriptor bytes overflowed"))
}
fn checked_alias(name: usize, names: usize, digests: usize, fields: usize) -> Result<usize> {
    let digests = digests.checked_mul(STATIC_INPUT_DIGEST_VERSION.len() + 64);
    names
        .checked_mul(name)
        .and_then(|names| names.checked_add(digests?))
        .and_then(|bytes| bytes.checked_add(fields))
        .ok_or_else(|| source::invalid("bytes", "digest retention byte count overflowed"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{STATIC_INPUT_DIGEST_VERSION, StaticInputDigest};
    use serde_json::Value;
    use std::collections::{BTreeMap, BTreeSet};

    fn digest_bytes(value: &StaticInputDigest) -> usize {
        value.digest_version.capacity() + value.sha256.capacity()
    }
    fn json_bytes(value: &Value) -> usize {
        match value {
            Value::String(value) => value.capacity(),
            Value::Object(values) => values
                .iter()
                .map(|(key, value)| key.capacity() + json_bytes(value))
                .sum(),
            _ => 0,
        }
    }
    #[test]
    fn test_managed_digest_retention_matches_the_actual_existing_context_and_identity_copies() {
        let name = "node_a.beta";
        let digest = StaticInputDigest {
            digest_version: STATIC_INPUT_DIGEST_VERSION.into(),
            sha256: "0".repeat(64),
        };
        let scope_name: std::sync::Arc<str> = name.into();
        let latched_keys = BTreeSet::from([name.to_owned()]);
        let prepared = BTreeMap::from([(name.to_owned(), digest.clone())]);
        let context_latched_keys = latched_keys.clone();
        let settings = serde_json::json!({name: {"digest_version":digest.digest_version,"sha256":digest.sha256}});
        let identity = prepared.clone();
        let selected = identity.clone();
        let native_json = serde_json::to_value(&selected).unwrap();
        let observed = scope_name.len()
            + digest_bytes(&digest)
            + latched_keys.iter().map(String::capacity).sum::<usize>()
            + prepared
                .iter()
                .map(|(key, value)| key.capacity() + digest_bytes(value))
                .sum::<usize>()
            + context_latched_keys
                .iter()
                .map(String::capacity)
                .sum::<usize>()
            + json_bytes(&settings)
            + identity
                .iter()
                .map(|(key, value)| key.capacity() + digest_bytes(value))
                .sum::<usize>()
            + selected
                .iter()
                .map(|(key, value)| key.capacity() + digest_bytes(value))
                .sum::<usize>()
            + json_bytes(&native_json);
        assert_eq!(
            Mode::Managed {
                checkpointing: true,
                context_copies: 1
            }
            .alias_bytes(name)
            .unwrap(),
            observed
        );
        assert_eq!(
            Mode::Finite.alias_bytes(name).unwrap(),
            name.len() + digest_bytes(&digest)
        );
    }
    #[test]
    fn test_context_and_digest_capacity_math_are_checked() {
        assert!(
            Mode::Managed {
                checkpointing: false,
                context_copies: usize::MAX
            }
            .alias_bytes("calc.beta")
            .is_err()
        );
        assert!(
            Mode::Managed {
                checkpointing: false,
                context_copies: 0
            }
            .alias_bytes("calc.beta")
            .is_err()
        );
        assert!(checked_alias(usize::MAX, 6, 4, 20).is_err());
    }
    #[test]
    fn test_managed_digest_retention_counts_each_real_task_context_json_copy_once() {
        let name = "node_a.beta";
        let digest = StaticInputDigest {
            digest_version: STATIC_INPUT_DIGEST_VERSION.into(),
            sha256: "0".repeat(64),
        };
        let one = serde_json::json!({name:{"digest_version":digest.digest_version,"sha256":digest.sha256}});
        let latched_keys = BTreeSet::from([name.to_owned()]);
        let prepared = BTreeMap::from([(name.to_owned(), digest.clone())]);
        let context_static_keys = latched_keys.clone();
        let identity = prepared.clone();
        let selected = identity.clone();
        let native_json = serde_json::to_value(&selected).unwrap();
        let selected_bytes = selected
            .iter()
            .map(|(key, value)| key.capacity() + digest_bytes(value))
            .sum::<usize>();
        let scope = name.len() + digest_bytes(&digest);
        let context_static_bytes = context_static_keys
            .iter()
            .map(String::capacity)
            .sum::<usize>();
        let identity_bytes = identity
            .iter()
            .map(|(name, digest)| name.capacity() + digest_bytes(digest))
            .sum::<usize>();
        let before_install = scope
            + latched_keys.iter().map(String::capacity).sum::<usize>()
            + prepared
                .iter()
                .map(|(name, digest)| name.capacity() + digest_bytes(digest))
                .sum::<usize>()
            + context_static_bytes
            + identity_bytes
            + selected_bytes
            + json_bytes(&native_json)
            + json_bytes(&one);
        drop(latched_keys);
        drop(prepared);
        let current = identity.clone();
        let current_bytes = current
            .iter()
            .map(|(key, value)| key.capacity() + digest_bytes(value))
            .sum::<usize>();
        let mut copies = vec![one];
        for _ in 1..3 {
            copies.push(copies[0].clone());
        }
        let active = scope
            + context_static_bytes
            + identity_bytes
            + selected_bytes
            + current_bytes
            + json_bytes(&native_json)
            + copies.iter().map(json_bytes).sum::<usize>();
        let observed = before_install.max(active);
        assert_eq!(
            Mode::Managed {
                checkpointing: true,
                context_copies: 3
            }
            .alias_bytes(name)
            .unwrap(),
            observed
        );
    }
    #[test]
    fn test_managed_retention_includes_selected_current_manifest_and_native_json_copy() {
        let name = "node.beta";
        let digest = StaticInputDigest {
            digest_version: STATIC_INPUT_DIGEST_VERSION.into(),
            sha256: "0".repeat(64),
        };
        let scope = name.len() + digest_bytes(&digest);
        let latched_keys = BTreeSet::from([name.to_owned()]);
        let identity = BTreeMap::from([(name.to_owned(), digest)]);
        let selected = identity.clone();
        let current = identity.clone();
        let json = serde_json::to_value(&current).unwrap();
        let contexts = vec![json.clone(); 4];
        let owned_map = |map: &BTreeMap<String, StaticInputDigest>| {
            map.iter()
                .map(|(name, digest)| name.capacity() + digest_bytes(digest))
                .sum::<usize>()
        };
        let actual = scope
            + latched_keys.iter().map(String::capacity).sum::<usize>()
            + owned_map(&identity)
            + owned_map(&selected)
            + owned_map(&current)
            + json_bytes(&json)
            + contexts.iter().map(json_bytes).sum::<usize>();
        let reserved = Mode::Managed {
            checkpointing: true,
            context_copies: 4,
        }
        .alias_bytes(name)
        .unwrap();
        assert_eq!(
            reserved, actual,
            "the selected and current native descriptor copies remain owned during serialization"
        );
    }
}
