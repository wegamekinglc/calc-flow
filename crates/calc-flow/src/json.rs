use std::collections::BTreeMap;

use serde::{
    Deserialize, Deserializer,
    de::{Error as _, MapAccess, SeqAccess, Visitor},
};
use serde_json::{Map, Number, Value};

use crate::{CalcFlowError, Result};

pub type JsonMap = BTreeMap<String, Value>;

/// Maximum JSON child depth accepted by canonical and persistence documents.
///
/// The top-level value is depth zero, and this inclusive bound is checked
/// iteratively before any recursive canonicalization.
pub const MAX_JSON_DEPTH: usize = 32;

pub(crate) fn validate_portable_identifier(field: &str, value: &str) -> Result<()> {
    if value.is_empty()
        || !value.chars().all(|character| {
            character == '-'
                || character == '_'
                || character == '.'
                || character.is_ascii_alphanumeric()
        })
    {
        Err(CalcFlowError::InvalidArgument {
            field: field.into(),
            message: "must be a non-empty portable identifier".into(),
        })
    } else {
        Ok(())
    }
}

/// Serializes a JSON value with recursively sorted mapping keys.
///
/// # Errors
///
/// Returns [`CalcFlowError::Format`] when the value cannot be serialized.
pub fn canonical_json(value: &Value) -> Result<String> {
    validate_json_depth(value, "JSON value")?;
    serde_json::to_string(&sort_value(value)).map_err(|error| CalcFlowError::Format {
        message: error.to_string(),
    })
}

pub(crate) fn parse_json_value(document: &[u8], label: &str) -> Result<Value> {
    let mut deserializer = serde_json::Deserializer::from_slice(document);
    let value = UniqueValue::deserialize(&mut deserializer)
        .map_err(|error| format_error(format!("{label} is invalid: {error}")))?
        .0;
    deserializer
        .end()
        .map_err(|error| format_error(format!("{label} is invalid: {error}")))?;
    validate_json_depth(&value, label)?;
    Ok(value)
}

pub(crate) fn validate_json_depth(value: &Value, label: &str) -> Result<()> {
    validate_json_depth_at(value, label, 0)
}

pub(crate) fn validate_json_depth_at(
    value: &Value,
    label: &str,
    initial_depth: usize,
) -> Result<()> {
    let mut pending = vec![(value, initial_depth)];
    while let Some((current, depth)) = pending.pop() {
        if depth > MAX_JSON_DEPTH {
            return Err(format_error(format!(
                "{label} exceeds the maximum JSON depth of {MAX_JSON_DEPTH}"
            )));
        }
        match current {
            Value::Array(values) => {
                pending.extend(values.iter().map(|value| (value, depth + 1)));
            }
            Value::Object(values) => {
                pending.extend(values.values().map(|value| (value, depth + 1)));
            }
            Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {}
        }
    }
    Ok(())
}

struct UniqueValue(Value);

impl<'de> Deserialize<'de> for UniqueValue {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(UniqueValueVisitor)
    }
}

struct UniqueValueVisitor;

impl<'de> Visitor<'de> for UniqueValueVisitor {
    type Value = UniqueValue;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a JSON value without duplicate object keys")
    }

    fn visit_bool<E>(self, value: bool) -> std::result::Result<Self::Value, E> {
        Ok(UniqueValue(Value::Bool(value)))
    }

    fn visit_i64<E>(self, value: i64) -> std::result::Result<Self::Value, E> {
        Ok(UniqueValue(Value::Number(Number::from(value))))
    }

    fn visit_u64<E>(self, value: u64) -> std::result::Result<Self::Value, E> {
        Ok(UniqueValue(Value::Number(Number::from(value))))
    }

    fn visit_f64<E>(self, value: f64) -> std::result::Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Number::from_f64(value)
            .map(Value::Number)
            .map(UniqueValue)
            .ok_or_else(|| E::custom("JSON numbers must be finite"))
    }

    fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        self.visit_string(value.to_owned())
    }

    fn visit_string<E>(self, value: String) -> std::result::Result<Self::Value, E> {
        Ok(UniqueValue(Value::String(value)))
    }

    fn visit_none<E>(self) -> std::result::Result<Self::Value, E> {
        Ok(UniqueValue(Value::Null))
    }

    fn visit_unit<E>(self) -> std::result::Result<Self::Value, E> {
        Ok(UniqueValue(Value::Null))
    }

    fn visit_some<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        UniqueValue::deserialize(deserializer)
    }

    fn visit_seq<A>(self, mut sequence: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::with_capacity(sequence.size_hint().unwrap_or(0));
        while let Some(value) = sequence.next_element::<UniqueValue>()? {
            values.push(value.0);
        }
        Ok(UniqueValue(Value::Array(values)))
    }

    fn visit_map<A>(self, mut mapping: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = Map::new();
        while let Some(key) = mapping.next_key::<String>()? {
            if values.contains_key(&key) {
                return Err(A::Error::custom(format!(
                    "duplicate JSON object key {key:?}"
                )));
            }
            let value = mapping.next_value::<UniqueValue>()?;
            values.insert(key, value.0);
        }
        Ok(UniqueValue(Value::Object(values)))
    }
}

fn sort_value(value: &Value) -> Value {
    match value {
        Value::Object(values) => Value::Object(
            values
                .iter()
                .map(|(key, value)| (key.clone(), sort_value(value)))
                .collect(),
        ),
        Value::Array(values) => Value::Array(values.iter().map(sort_value).collect()),
        scalar => scalar.clone(),
    }
}

fn format_error(message: String) -> CalcFlowError {
    CalcFlowError::Format { message }
}
