use serde_json::{Map, Value};

use super::super::{JoinSide, OperatorStateSnapshot};
use super::frame::InvalidFrame;

type Result<T> = std::result::Result<T, InvalidFrame>;

pub(super) struct Inventory<'a> {
    pub(super) epoch: u64,
    pub(super) base_epoch: u64,
    deltas: &'a [Value],
    payloads: &'a [Value],
}

#[derive(Clone, Copy)]
pub(super) struct Payload {
    pub(super) side: JoinSide,
    pub(super) digest: [u8; 32],
    pub(super) rows: u64,
    pub(super) bytes: u64,
}

impl<'a> Inventory<'a> {
    pub(super) fn decode(
        snapshot: &'a OperatorStateSnapshot,
        check: &dyn Fn() -> crate::Result<()>,
    ) -> crate::Result<Self> {
        let metadata = &snapshot.inline_metadata;
        check()?;
        super::geometry::checked(validate_metadata(metadata))?;
        let inventory = super::geometry::checked(Self::decode_fields(metadata))?;
        inventory.validate_deltas(check)?;
        inventory.validate_payloads(check)?;
        inventory.validate_segments(snapshot, check)?;
        Ok(inventory)
    }

    fn decode_fields(metadata: &'a crate::json::JsonMap) -> Result<Self> {
        let epoch = integer(metadata.get("epoch"))?;
        let object = object(metadata.get("v2_inventory"), INVENTORY_KEYS)?;
        Ok(Self {
            epoch,
            base_epoch: validated_base_epoch(epoch, object)?,
            deltas: array(object.get("deltas"))?,
            payloads: array(object.get("payloads"))?,
        })
    }

    pub(super) fn payloads(&self) -> impl Iterator<Item = Result<Payload>> + '_ {
        self.payloads.iter().map(payload)
    }

    pub(super) fn delta_sides(&self) -> impl Iterator<Item = Result<(u64, &'a [Value])>> + '_ {
        self.deltas.iter().map(|value| {
            let object = object(Some(value), DELTA_KEYS)?;
            Ok((integer(object.get("epoch"))?, array(object.get("sides"))?))
        })
    }

    fn validate_deltas(&self, check: &dyn Fn() -> crate::Result<()>) -> crate::Result<()> {
        let mut previous = self.base_epoch;
        for delta in self.delta_sides() {
            check()?;
            let (epoch, sides) = super::geometry::checked(delta)?;
            if epoch <= previous || epoch > self.epoch {
                return Err(super::geometry::invalid(
                    "delta epochs must strictly increase",
                ));
            }
            validate_sides(sides, check)?;
            previous = epoch;
        }
        Ok(())
    }

    fn validate_payloads(&self, check: &dyn Fn() -> crate::Result<()>) -> crate::Result<()> {
        let mut previous = None;
        for entry in self.payloads() {
            check()?;
            let entry = super::geometry::checked(entry)?;
            let identity = (entry.side, entry.digest);
            if previous.is_some_and(|previous| identity <= previous) {
                return Err(super::geometry::invalid(
                    "payload inventory must be sorted and unique",
                ));
            }
            previous = Some(identity);
        }
        Ok(())
    }

    fn validate_segments(
        &self,
        snapshot: &OperatorStateSnapshot,
        check: &dyn Fn() -> crate::Result<()>,
    ) -> crate::Result<()> {
        if !snapshot.segments.contains_key("left-base")
            || !snapshot.segments.contains_key("right-base")
        {
            return Err(super::geometry::invalid("both V2 bases are required"));
        }
        if snapshot.segments.len() != self.expected_segments(check)? {
            return Err(super::geometry::invalid("segment inventory count mismatch"));
        }
        for (name, segment) in &snapshot.segments {
            check()?;
            self.validate_segment(name, segment.bytes().len(), check)?;
        }
        Ok(())
    }

    fn expected_segments(&self, check: &dyn Fn() -> crate::Result<()>) -> crate::Result<usize> {
        let mut expected = 2_usize
            .checked_add(self.payloads.len())
            .ok_or_else(|| super::geometry::invalid("segment count overflow"))?;
        for delta in self.delta_sides() {
            check()?;
            expected = expected
                .checked_add(super::geometry::checked(delta)?.1.len())
                .ok_or_else(|| super::geometry::invalid("segment count overflow"))?;
        }
        Ok(expected)
    }

    fn validate_segment(
        &self,
        name: &str,
        length: usize,
        check: &dyn Fn() -> crate::Result<()>,
    ) -> crate::Result<()> {
        if name == "left-base" || name == "right-base" {
            return Ok(());
        }
        let (side, suffix) = super::geometry::checked(split_side(name))?;
        if let Some(text) = suffix.strip_prefix("payload-") {
            return self.validate_payload_segment(
                side,
                super::geometry::checked(digest(text))?,
                length,
                check,
            );
        }
        if let Some(text) = suffix.strip_prefix("delta-") {
            return self.validate_delta_segment(
                side,
                super::geometry::checked(decimal_epoch(text))?,
                check,
            );
        }
        Err(super::geometry::invalid("unknown V2 segment name"))
    }

    fn validate_payload_segment(
        &self,
        side: JoinSide,
        digest: [u8; 32],
        length: usize,
        check: &dyn Fn() -> crate::Result<()>,
    ) -> crate::Result<()> {
        for entry in self.payloads() {
            check()?;
            let entry = super::geometry::checked(entry)?;
            if entry.side == side && entry.digest == digest {
                let length = u64::try_from(length)
                    .map_err(|_| super::geometry::invalid("segment length exceeds u64"))?;
                return if entry.bytes == length {
                    Ok(())
                } else {
                    Err(super::geometry::invalid("payload byte count mismatch"))
                };
            }
        }
        Err(super::geometry::invalid("unlisted payload segment"))
    }

    fn validate_delta_segment(
        &self,
        expected_side: JoinSide,
        expected_epoch: u64,
        check: &dyn Fn() -> crate::Result<()>,
    ) -> crate::Result<()> {
        for entry in self.delta_sides() {
            check()?;
            let (epoch, sides) = super::geometry::checked(entry)?;
            if epoch == expected_epoch {
                return validate_delta_side(sides, expected_side, check);
            }
        }
        Err(super::geometry::invalid("unlisted delta segment"))
    }
}

fn validate_delta_side(
    sides: &[Value],
    expected_side: JoinSide,
    check: &dyn Fn() -> crate::Result<()>,
) -> crate::Result<()> {
    for value in sides {
        check()?;
        if super::geometry::checked(side(value))? == expected_side {
            return Ok(());
        }
    }
    Err(super::geometry::invalid("unlisted delta segment"))
}

fn validate_sides(values: &[Value], check: &dyn Fn() -> crate::Result<()>) -> crate::Result<()> {
    if values.is_empty() {
        return Err(super::geometry::invalid("delta sides cannot be empty"));
    }
    let mut previous = None;
    for value in values {
        check()?;
        let side = super::geometry::checked(side(value))?;
        if previous.is_some_and(|previous| side <= previous) {
            return Err(super::geometry::invalid(
                "delta sides must be sorted and unique",
            ));
        }
        previous = Some(side);
    }
    Ok(())
}

fn validate_metadata(metadata: &crate::json::JsonMap) -> Result<()> {
    exact_keys(metadata.keys().map(String::as_str), METADATA_KEYS)?;
    super::metadata::validate(metadata)?;
    if integer(metadata.get("layout_version"))? != 2 {
        return Err(InvalidFrame("layout version must be 2"));
    }
    Ok(())
}

fn validated_base_epoch(epoch: u64, object: &Map<String, Value>) -> Result<u64> {
    if epoch == 0 || integer(object.get("codec_version"))? != 2 {
        return Err(InvalidFrame("inventory version or epoch is invalid"));
    }
    let base_epoch = integer(object.get("base_epoch"))?;
    if base_epoch > epoch {
        return Err(InvalidFrame("base epoch exceeds checkpoint epoch"));
    }
    Ok(base_epoch)
}

fn payload(value: &Value) -> Result<Payload> {
    let object = object(Some(value), PAYLOAD_KEYS)?;
    let rows = integer(object.get("rows"))?;
    if rows == 0 {
        return Err(InvalidFrame("inventory payload must contain rows"));
    }
    Ok(Payload {
        side: side(required(object.get("side"))?)?,
        digest: payload_digest(object)?,
        rows,
        bytes: integer(object.get("bytes"))?,
    })
}

fn payload_digest(object: &Map<String, Value>) -> Result<[u8; 32]> {
    digest(
        required(object.get("sha256"))?
            .as_str()
            .ok_or(InvalidFrame("payload digest must be a string"))?,
    )
}

pub(super) fn side(value: &Value) -> Result<JoinSide> {
    match value.as_str() {
        Some("left") => Ok(JoinSide::Left),
        Some("right") => Ok(JoinSide::Right),
        _ => Err(InvalidFrame("side must be left or right")),
    }
}

pub(super) fn split_side(name: &str) -> Result<(JoinSide, &str)> {
    if let Some(suffix) = name.strip_prefix("left-") {
        return Ok((JoinSide::Left, suffix));
    }
    if let Some(suffix) = name.strip_prefix("right-") {
        return Ok((JoinSide::Right, suffix));
    }
    Err(InvalidFrame("segment side prefix is invalid"))
}

pub(super) fn digest(text: &str) -> Result<[u8; 32]> {
    if text.len() != 64 {
        return Err(InvalidFrame("payload digest must have 64 characters"));
    }
    let mut digest = [0; 32];
    for (pair, output) in text.as_bytes().chunks_exact(2).zip(&mut digest) {
        *output = (hex_digit(pair[0])? << 4) | hex_digit(pair[1])?;
    }
    Ok(digest)
}

fn hex_digit(byte: u8) -> Result<u8> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        _ => Err(InvalidFrame("payload digest must be lowercase hexadecimal")),
    }
}

fn decimal_epoch(text: &str) -> Result<u64> {
    if text.is_empty() || text.starts_with('0') || !text.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(InvalidFrame("delta epoch name must be canonical decimal"));
    }
    text.parse()
        .map_err(|_| InvalidFrame("delta epoch name exceeds u64"))
}

pub(super) fn integer(value: Option<&Value>) -> Result<u64> {
    required(value)?
        .as_u64()
        .ok_or(InvalidFrame("metadata counter must be an exact u64"))
}

fn array(value: Option<&Value>) -> Result<&[Value]> {
    required(value)?
        .as_array()
        .map(Vec::as_slice)
        .ok_or(InvalidFrame("inventory field must be an array"))
}

pub(super) fn object<'a>(
    value: Option<&'a Value>,
    keys: &[&str],
) -> Result<&'a Map<String, Value>> {
    let object = required(value)?
        .as_object()
        .ok_or(InvalidFrame("inventory field must be an object"))?;
    exact_keys(object.keys().map(String::as_str), keys)?;
    Ok(object)
}

fn required(value: Option<&Value>) -> Result<&Value> {
    value.ok_or(InvalidFrame("required metadata field is missing"))
}

pub(super) fn exact_keys<'a>(keys: impl Iterator<Item = &'a str>, expected: &[&str]) -> Result<()> {
    let mut count = 0;
    for key in keys {
        if !expected.contains(&key) {
            return Err(InvalidFrame("unknown metadata field"));
        }
        count += 1;
    }
    if count != expected.len() {
        return Err(InvalidFrame("required metadata field is missing"));
    }
    Ok(())
}

const METADATA_KEYS: &[&str] = &[
    "layout_version",
    "spec",
    "next_left_row_id",
    "next_right_row_id",
    "next_output_sequence",
    "metrics",
    "ended",
    "epoch",
    "v2_inventory",
];
const INVENTORY_KEYS: &[&str] = &["codec_version", "base_epoch", "deltas", "payloads"];
const DELTA_KEYS: &[&str] = &["epoch", "sides"];
const PAYLOAD_KEYS: &[&str] = &["side", "sha256", "rows", "bytes"];
