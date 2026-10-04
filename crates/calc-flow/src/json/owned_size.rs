use crate::JsonMap;
use serde_json::Value;

pub(crate) fn map_bytes(values: &JsonMap) -> Option<usize> {
    values.iter().try_fold(1024_usize, |total, (key, value)| {
        total
            .checked_add(256)?
            .checked_add(string_bytes(key)?)?
            .checked_add(value_bytes(value)?)
    })
}

fn string_bytes(value: &String) -> Option<usize> {
    Some(
        value
            .capacity()
            .max(value.len().checked_next_power_of_two()?.max(8)),
    )
}

fn value_bytes(value: &Value) -> Option<usize> {
    match value {
        Value::String(value) => string_bytes(value),
        Value::Array(values) => {
            let slots = values
                .capacity()
                .max(values.len().checked_next_power_of_two()?.max(4));
            values
                .iter()
                .try_fold(slots.checked_mul(size_of::<Value>())?, |total, value| {
                    total.checked_add(value_bytes(value)?)
                })
        }
        Value::Object(values) => values.iter().try_fold(1024_usize, |total, (key, value)| {
            total
                .checked_add(256)?
                .checked_add(string_bytes(key)?)?
                .checked_add(value_bytes(value)?)
        }),
        _ => Some(0),
    }
}
