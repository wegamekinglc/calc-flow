//! Shared connector option parsing.
//!
//! Every connector reads its configuration from the same data-only JSON
//! map, so the primitive extraction helpers live here once instead of one
//! near-identical copy per transport. All failures use the standard
//! [`calc_flow::CalcFlowError::InvalidArgument`] shape with the offending
//! option key as the field.

use calc_flow::{CalcFlowError, JsonMap, Result};
use serde_json::Value;

/// Require a string option and return an owned copy of it.
pub(crate) fn required_string(options: &JsonMap, key: &str) -> Result<String> {
    match options.get(key) {
        Some(Value::String(value)) => Ok(value.clone()),
        Some(_) => Err(CalcFlowError::InvalidArgument {
            field: key.into(),
            message: "option must be a string".into(),
        }),
        None => Err(CalcFlowError::InvalidArgument {
            field: key.into(),
            message: "option is required".into(),
        }),
    }
}

/// Require a string option and borrow it without copying.
pub(crate) fn required_str<'a>(options: &'a JsonMap, key: &str) -> Result<&'a str> {
    match options.get(key) {
        Some(Value::String(value)) => Ok(value.as_str()),
        Some(_) => Err(CalcFlowError::InvalidArgument {
            field: key.into(),
            message: "option must be a string".into(),
        }),
        None => Err(CalcFlowError::InvalidArgument {
            field: key.into(),
            message: "option is required".into(),
        }),
    }
}

/// Read an optional non-negative integer option; `null` counts as absent.
pub(crate) fn u64_option(options: &JsonMap, key: &str) -> Result<Option<u64>> {
    match options.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(number)) => {
            number
                .as_u64()
                .map(Some)
                .ok_or(CalcFlowError::InvalidArgument {
                    field: key.into(),
                    message: "option must be a non-negative integer".into(),
                })
        }
        Some(_) => Err(CalcFlowError::InvalidArgument {
            field: key.into(),
            message: "option must be a non-negative integer".into(),
        }),
    }
}

/// Read a positive integer option, substituting `default` when absent.
pub(crate) fn positive_option(options: &JsonMap, key: &str, default: u64) -> Result<u64> {
    let value = u64_option(options, key)?.unwrap_or(default);
    if value == 0 {
        Err(CalcFlowError::InvalidArgument {
            field: key.into(),
            message: "option must be greater than zero".into(),
        })
    } else {
        Ok(value)
    }
}

/// Read an optional boolean option; `null` counts as absent.
pub(crate) fn bool_option(options: &JsonMap, key: &str) -> Result<Option<bool>> {
    match options.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Bool(value)) => Ok(Some(*value)),
        Some(_) => Err(CalcFlowError::InvalidArgument {
            field: key.into(),
            message: "option must be a boolean".into(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(entries: &[(&str, Value)]) -> JsonMap {
        entries
            .iter()
            .map(|(key, value)| ((*key).to_string(), value.clone()))
            .collect()
    }

    #[test]
    fn required_string_rejects_missing_and_mistyped_values() {
        let options = map(&[
            ("name", Value::String("source".into())),
            ("count", Value::from(3)),
        ]);
        assert_eq!(
            required_string(&options, "name").expect("string present"),
            "source"
        );
        let missing = required_string(&options, "absent").expect_err("missing option fails");
        let mistyped = required_string(&options, "count").expect_err("number is not a string");
        for error in [missing, mistyped] {
            assert!(matches!(error, CalcFlowError::InvalidArgument { .. }));
        }
    }

    #[test]
    fn required_str_borrows_without_copying() {
        let options = map(&[("path", Value::String("events.csv".into()))]);
        assert_eq!(
            required_str(&options, "path").expect("string present"),
            "events.csv"
        );
        assert!(required_str(&options, "absent").is_err());
    }

    #[test]
    fn u64_option_treats_null_as_absent_and_rejects_negatives() {
        let options = map(&[
            ("rows", Value::from(12)),
            ("blank", Value::Null),
            ("negative", Value::from(-1)),
        ]);
        assert_eq!(
            u64_option(&options, "rows").expect("number present"),
            Some(12)
        );
        assert_eq!(u64_option(&options, "blank").expect("null is absent"), None);
        assert_eq!(
            u64_option(&options, "absent").expect("missing is absent"),
            None
        );
        assert!(u64_option(&options, "negative").is_err());
    }

    #[test]
    fn positive_option_applies_the_default_and_rejects_zero() {
        let options = map(&[("zero", Value::from(0)), ("size", Value::from(9))]);
        assert_eq!(
            positive_option(&options, "absent", 7).expect("default applies"),
            7
        );
        assert_eq!(
            positive_option(&options, "size", 7).expect("value applies"),
            9
        );
        assert!(positive_option(&options, "zero", 7).is_err());
    }

    #[test]
    fn bool_option_treats_null_as_absent_and_rejects_strings() {
        let options = map(&[
            ("tls", Value::Bool(true)),
            ("blank", Value::Null),
            ("text", Value::String("yes".into())),
        ]);
        assert_eq!(
            bool_option(&options, "tls").expect("bool present"),
            Some(true)
        );
        assert_eq!(
            bool_option(&options, "blank").expect("null is absent"),
            None
        );
        assert!(bool_option(&options, "text").is_err());
    }
}
