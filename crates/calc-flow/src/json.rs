use std::collections::BTreeMap;

use serde_json::Value;

use crate::{CalcFlowError, Result};

pub type JsonMap = BTreeMap<String, Value>;

/// Serializes a JSON value with recursively sorted mapping keys.
///
/// # Errors
///
/// Returns [`CalcFlowError::Format`] when the value cannot be serialized.
pub fn canonical_json(value: &Value) -> Result<String> {
    serde_json::to_string(&sort_value(value)).map_err(|error| CalcFlowError::Format {
        message: error.to_string(),
    })
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
