use serde_json::{Map, Value};

use crate::{
    JsonMap, Result, RollingNumericalProfile,
    array::{
        ArrayDType, ArrayExecutionOptions, ArrayStateOptions, EwmaOptions, RollingMeanOptions,
        invalid,
        sequence::codec::{KernelOptions, SequenceSpec},
    },
};

pub(super) struct Options {
    pub(super) spec: SequenceSpec,
    pub(super) execution: ArrayExecutionOptions,
    pub(super) state: ArrayStateOptions,
}

pub(super) fn parse(options: &JsonMap) -> Result<Options> {
    validate_document(options)?;
    Ok(Options {
        spec: specification(options)?,
        execution: execution(options.get("execution"))?,
        state: state(options.get("state"))?,
    })
}
fn validate_document(options: &JsonMap) -> Result<()> {
    super::super::validate::bounded(options)?;
    fields(
        options.keys().map(String::as_str),
        &["ir_version", "input_dtype", "kernel", "execution", "state"],
        "options",
    )?;
    if options.get("ir_version").and_then(Value::as_u64) != Some(1) {
        return Err(invalid("options.ir_version", "requires integer version 1"));
    }
    Ok(())
}
fn specification(options: &JsonMap) -> Result<SequenceSpec> {
    let values = object(options.get("kernel"), "options.kernel")?;
    SequenceSpec::new(dtype(options.get("input_dtype"))?, kernel(values)?)
}
fn dtype(value: Option<&Value>) -> Result<ArrayDType> {
    match value.and_then(Value::as_str) {
        Some("float32") => Ok(ArrayDType::Float32),
        Some("float64") => Ok(ArrayDType::Float64),
        _ => Err(invalid(
            "options.input_dtype",
            "requires float32 or float64",
        )),
    }
}

fn object<'a>(value: Option<&'a Value>, field: &str) -> Result<&'a Map<String, Value>> {
    value
        .and_then(Value::as_object)
        .ok_or_else(|| invalid(field, "expected an object"))
}
fn fields<'a>(keys: impl Iterator<Item = &'a str>, allowed: &[&str], prefix: &str) -> Result<()> {
    for key in keys {
        if !allowed.contains(&key) {
            return Err(invalid(&format!("{prefix}.{key}"), "unknown field"));
        }
    }
    Ok(())
}
fn count(value: Option<&Value>, field: &str) -> Result<u64> {
    value
        .and_then(Value::as_u64)
        .filter(|value| *value > 0)
        .ok_or_else(|| invalid(field, "expected a positive u64 count"))
}
fn minimum(values: &Map<String, Value>) -> Result<u64> {
    match values.get("min_periods") {
        Some(value) => count(Some(value), "options.kernel.min_periods"),
        None => Ok(1),
    }
}
fn kernel(values: &Map<String, Value>) -> Result<KernelOptions> {
    match values.get("function").and_then(Value::as_str) {
        Some("rolling_mean") => mean(values),
        Some("ewma") => ewma(values),
        _ => Err(invalid(
            "options.kernel.function",
            "requires rolling_mean or ewma",
        )),
    }
}
fn mean(values: &Map<String, Value>) -> Result<KernelOptions> {
    let (window, minimum) = mean_counts(values)?;
    let options = RollingMeanOptions::new(window)?.with_min_periods(minimum)?;
    Ok(KernelOptions::Mean(
        options.with_numerical_profile(profile(values)?),
    ))
}
fn mean_counts(values: &Map<String, Value>) -> Result<(u64, u64)> {
    fields(
        values.keys().map(String::as_str),
        &["function", "window", "min_periods", "numerical_profile"],
        "options.kernel",
    )?;
    let window = count(values.get("window"), "options.kernel.window")?;
    let minimum = minimum(values)?;
    if minimum > window {
        return Err(invalid("options.kernel.min_periods", "expected 1..=window"));
    }
    Ok((window, minimum))
}
fn profile(values: &Map<String, Value>) -> Result<RollingNumericalProfile> {
    match values.get("numerical_profile") {
        None => Ok(RollingNumericalProfile::StableV1),
        Some(Value::String(name)) if name == "stable_v1" => Ok(RollingNumericalProfile::StableV1),
        Some(Value::String(name)) if name == "stable_v2" => {
            Ok(RollingNumericalProfile::StableV2Preview)
        }
        _ => Err(invalid(
            "options.kernel.numerical_profile",
            "requires stable_v1 or stable_v2",
        )),
    }
}

fn ewma(values: &Map<String, Value>) -> Result<KernelOptions> {
    fields(
        values.keys().map(String::as_str),
        &["function", "span", "min_periods"],
        "options.kernel",
    )?;
    Ok(KernelOptions::Ewma(
        EwmaOptions::new(count(values.get("span"), "options.kernel.span")?)?
            .with_min_periods(minimum(values)?)?,
    ))
}
fn usize_count(value: &Value, field: &str) -> Result<usize> {
    value
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| invalid(field, "expected a nonnegative usize count"))
}
fn execution(value: Option<&Value>) -> Result<ArrayExecutionOptions> {
    match value {
        None => Ok(ArrayExecutionOptions::default()),
        Some(value) => execution_object(value),
    }
}
fn execution_object(value: &Value) -> Result<ArrayExecutionOptions> {
    let mut options = ArrayExecutionOptions::default();
    let values = object(Some(value), "options.execution")?;
    fields(
        values.keys().map(String::as_str),
        &["max_elements", "max_allocation_bytes"],
        "options.execution",
    )?;
    if let Some(value) = values.get("max_elements") {
        options = options.with_max_elements(usize_count(value, "options.execution.max_elements")?);
    }
    if let Some(value) = values.get("max_allocation_bytes") {
        options = options.with_max_allocation_bytes(usize_count(
            value,
            "options.execution.max_allocation_bytes",
        )?);
    }
    Ok(options)
}

fn state(value: Option<&Value>) -> Result<ArrayStateOptions> {
    let mut options = ArrayStateOptions::default();
    if let Some(value) = value {
        let values = object(Some(value), "options.state")?;
        fields(
            values.keys().map(String::as_str),
            &["max_state_bytes"],
            "options.state",
        )?;
        if let Some(value) = values.get("max_state_bytes") {
            options =
                options.with_max_state_bytes(usize_count(value, "options.state.max_state_bytes")?);
        }
    }
    Ok(options)
}
