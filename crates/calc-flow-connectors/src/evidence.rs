//! Shared pre-commit evidence protocol checks.
//!
//! The `PostgreSQL`, `ClickHouse`, and `Kafka` transactional sinks all validate
//! the same data-only evidence contract before committing a prepared
//! segment: identity fields, epoch, segment id, schema hash, byte count,
//! SHA-256 checksum, and row count. The helpers here implement each check
//! once and return the standard failure message; the transport call sites
//! wrap that message with their own operation-specific error constructor.
//! Every check fails closed.

use calc_flow::{Epoch, JsonMap};
use serde_json::Value;
use sha2::{Digest as _, Sha256};

/// Extract a required string field, or fail with the standard message.
pub(crate) fn string_field(evidence: &JsonMap, field: &str) -> Result<String, String> {
    evidence
        .get(field)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| format!("pre-commit field {field:?} is missing"))
}

/// Require the evidence to name this pipeline, output, and target table.
pub(crate) fn check_identity(
    evidence: &JsonMap,
    pipeline: &str,
    output: &str,
    target: &str,
) -> Result<(), String> {
    if string_field(evidence, "pipeline")? == pipeline
        && string_field(evidence, "output")? == output
        && string_field(evidence, "target")? == target
    {
        Ok(())
    } else {
        Err("pre-commit evidence names a different sink identity".into())
    }
}

/// Require the evidence to carry exactly this epoch.
pub(crate) fn check_epoch(evidence: &JsonMap, epoch: Epoch) -> Result<(), String> {
    if evidence.get("epoch").and_then(Value::as_u64) == Some(epoch.as_u64()) {
        Ok(())
    } else {
        Err("pre-commit evidence names a different epoch".into())
    }
}

/// Require the evidence to carry the expected segment identifier.
pub(crate) fn check_segment_id(evidence: &JsonMap, expected: &str) -> Result<(), String> {
    if string_field(evidence, "segment_id")? == expected {
        Ok(())
    } else {
        Err("pre-commit segment identity is invalid".into())
    }
}

/// Require the declared `schema_hash` to be a 64-character hex string.
pub(crate) fn check_schema_hash(evidence: &JsonMap) -> Result<(), String> {
    let hash = string_field(evidence, "schema_hash")?;
    if hash.len() == 64 && hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Ok(())
    } else {
        Err("pre-commit evidence has an invalid schema hash".into())
    }
}

/// Require `segment_bytes` and `segment_sha256` to match the prepared bytes.
pub(crate) fn check_segment(evidence: &JsonMap, segment: &[u8]) -> Result<(), String> {
    let declared_bytes = evidence.get("segment_bytes").and_then(Value::as_u64);
    let actual_bytes = u64::try_from(segment.len()).unwrap_or(u64::MAX);
    if declared_bytes != Some(actual_bytes) {
        return Err("pre-commit segment byte count does not match its prepared payload".into());
    }
    if string_field(evidence, "segment_sha256")? != sha256_hex(segment) {
        return Err("pre-commit segment checksum does not match its prepared payload".into());
    }
    Ok(())
}

/// Require the declared row count to match the prepared row count.
pub(crate) fn check_rows(evidence: &JsonMap, actual_rows: u64) -> Result<(), String> {
    let declared = evidence
        .get("rows")
        .and_then(Value::as_u64)
        .ok_or_else(|| "pre-commit row count is missing".to_string())?;
    if declared == actual_rows {
        Ok(())
    } else {
        Err("pre-commit row count does not match its prepared payload".into())
    }
}

/// Hex-encode the SHA-256 digest of the prepared bytes.
pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence(entries: &[(&str, Value)]) -> JsonMap {
        entries
            .iter()
            .map(|(key, value)| ((*key).to_string(), value.clone()))
            .collect()
    }

    #[test]
    fn string_field_requires_present_strings() {
        let map = evidence(&[("pipeline", Value::String("p".into()))]);
        assert_eq!(string_field(&map, "pipeline").expect("present"), "p");
        assert_eq!(
            string_field(&map, "absent").expect_err("missing field"),
            r#"pre-commit field "absent" is missing"#
        );
    }

    #[test]
    fn identity_epoch_and_segment_checks_fail_closed() {
        let map = evidence(&[
            ("pipeline", Value::String("p".into())),
            ("output", Value::String("o".into())),
            ("target", Value::String("t".into())),
            ("epoch", Value::from(7)),
            ("segment_id", Value::String("seg".into())),
        ]);
        assert!(check_identity(&map, "p", "o", "t").is_ok());
        assert!(check_identity(&map, "p", "o", "other").is_err());
        assert!(check_epoch(&map, Epoch::new(7).expect("non-zero epoch")).is_ok());
        assert!(check_epoch(&map, Epoch::new(8).expect("non-zero epoch")).is_err());
        assert!(check_segment_id(&map, "seg").is_ok());
        assert!(check_segment_id(&map, "other").is_err());
    }

    #[test]
    fn schema_hash_requires_sixty_four_hex_characters() {
        let good = evidence(&[("schema_hash", Value::String("a".repeat(64)))]);
        let short = evidence(&[("schema_hash", Value::String("a".repeat(63)))]);
        let non_hex = evidence(&[("schema_hash", Value::String("z".repeat(64)))]);
        assert!(check_schema_hash(&good).is_ok());
        assert!(check_schema_hash(&short).is_err());
        assert!(check_schema_hash(&non_hex).is_err());
    }

    #[test]
    fn segment_checks_bind_bytes_and_checksum() {
        let payload = b"rows";
        let map = evidence(&[
            ("segment_bytes", Value::from(payload.len() as u64)),
            ("segment_sha256", Value::String(sha256_hex(payload))),
        ]);
        assert!(check_segment(&map, payload).is_ok());
        assert!(check_segment(&map, b"other").is_err());
        let no_bytes: JsonMap = evidence(&[]);
        assert!(check_segment(&no_bytes, payload).is_err());
    }

    #[test]
    fn row_count_must_be_declared_and_match() {
        let map = evidence(&[("rows", Value::from(3))]);
        assert!(check_rows(&map, 3).is_ok());
        assert!(check_rows(&map, 4).is_err());
        assert!(check_rows(&JsonMap::new(), 3).is_err());
    }
}
