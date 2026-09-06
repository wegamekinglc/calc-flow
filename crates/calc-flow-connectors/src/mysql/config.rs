use std::{collections::BTreeSet, time::Duration};

use crate::options::{bool_option, positive_option, required_str, required_string};
use calc_flow::{ArrowFieldSpec, CalcFlowError, JsonMap, Result};

pub(super) fn invalid(field: &str, message: &str) -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: field.into(),
        message: message.into(),
    }
}

pub(super) fn identifier(name: &str) -> Result<String> {
    if name.is_empty()
        || name.len() > 64
        || !name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
        || name.as_bytes()[0].is_ascii_digit()
    {
        return Err(invalid(
            "identifier",
            "expected an ASCII SQL identifier of 1 to 64 bytes",
        ));
    }
    Ok(format!("`{name}`"))
}

fn known_options(options: &JsonMap, allowed: &[&str]) -> Result<()> {
    for key in options.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(invalid(
                key,
                "unknown MySQL option; credentials belong in the url secret slot",
            ));
        }
    }
    Ok(())
}

fn names(options: &JsonMap, key: &str) -> Result<Vec<String>> {
    let Some(value) = options.get(key) else {
        return Ok(Vec::new());
    };
    let values = value
        .as_array()
        .ok_or_else(|| invalid(key, "expected an array of column names"))?;
    let mut seen = BTreeSet::new();
    values
        .iter()
        .map(|value| {
            let name = value
                .as_str()
                .ok_or_else(|| invalid(key, "expected a string column name"))?;
            identifier(name)?;
            if !seen.insert(name.to_ascii_lowercase()) {
                return Err(invalid(key, "column names must be unique"));
            }
            Ok(name.to_string())
        })
        .collect()
}

#[derive(Clone)]
pub(super) struct ConnectionConfig {
    pub table: String,
    pub tls: bool,
    pub timeout: Duration,
}

impl ConnectionConfig {
    fn parse(options: &JsonMap) -> Result<Self> {
        let table = required_string(options, "table")?;
        identifier(&table)?;
        let seconds = positive_option(options, "timeout_seconds", 30)?;
        if seconds > 3600 {
            return Err(invalid("timeout_seconds", "must be at most 3600"));
        }
        Ok(Self {
            table,
            tls: bool_option(options, "tls")?.unwrap_or(true),
            timeout: Duration::from_secs(seconds),
        })
    }
}

#[derive(Clone)]
pub(super) struct SourceConfig {
    pub connection: ConnectionConfig,
    pub incremental: bool,
    pub cursors: Vec<String>,
    pub columns: Vec<ArrowFieldSpec>,
    pub rows: u64,
    pub bytes: u64,
    pub poll: Duration,
}

impl SourceConfig {
    pub fn parse(options: &JsonMap) -> Result<Self> {
        known_options(
            options,
            &[
                "table",
                "tls",
                "timeout_seconds",
                "mode",
                "cursor_columns",
                "columns",
                "max_batch_rows",
                "max_batch_bytes",
                "poll_interval_ms",
                "assume_monotonic_cursor",
            ],
        )?;
        let incremental = source_mode(options)?;
        let cursors = source_cursors(options, incremental)?;
        let columns = source_columns(options, &cursors)?;
        let (rows, bytes, poll) = source_limits(options)?;
        Ok(Self {
            connection: ConnectionConfig::parse(options)?,
            incremental,
            cursors,
            columns,
            rows,
            bytes,
            poll,
        })
    }
}

fn source_mode(options: &JsonMap) -> Result<bool> {
    match options.get("mode").map(serde_json::Value::as_str) {
        None | Some(Some("snapshot")) => Ok(false),
        Some(Some("incremental_query")) => Ok(true),
        _ => Err(invalid("mode", "expected snapshot or incremental_query")),
    }
}

fn source_cursors(options: &JsonMap, incremental: bool) -> Result<Vec<String>> {
    let cursors = names(options, "cursor_columns")?;
    let monotonic = bool_option(options, "assume_monotonic_cursor")?.unwrap_or(false);
    validate_cursor_mode(incremental, &cursors, monotonic)?;
    Ok(cursors)
}

fn validate_cursor_mode(incremental: bool, cursors: &[String], monotonic: bool) -> Result<()> {
    if incremental && (cursors.is_empty() || !monotonic) {
        return Err(invalid(
            "cursor_columns",
            "incremental_query requires cursor_columns and assume_monotonic_cursor=true",
        ));
    }
    if !incremental && (!cursors.is_empty() || monotonic) {
        return Err(invalid(
            "cursor_columns",
            "cursor options require incremental_query",
        ));
    }
    Ok(())
}

fn source_columns(options: &JsonMap, cursors: &[String]) -> Result<Vec<ArrowFieldSpec>> {
    let columns: Vec<ArrowFieldSpec> = options
        .get("columns")
        .map(|value| serde_json::from_value(value.clone()))
        .transpose()
        .map_err(|_| invalid("columns", "expected Arrow field specifications"))?
        .unwrap_or_default();
    validate_column_names(&columns)?;
    crate::arrow_schema::schema_from_spec(&columns)?;
    validate_cursor_projection(&columns, cursors)?;
    Ok(columns)
}

fn validate_column_names(columns: &[ArrowFieldSpec]) -> Result<()> {
    let mut seen = BTreeSet::new();
    for field in columns {
        identifier(&field.name)?;
        if !seen.insert(field.name.to_ascii_lowercase()) {
            return Err(invalid("columns", "duplicate column"));
        }
    }
    Ok(())
}

fn validate_cursor_projection(columns: &[ArrowFieldSpec], cursors: &[String]) -> Result<()> {
    if !columns.is_empty()
        && cursors
            .iter()
            .any(|name| !columns.iter().any(|field| &field.name == name))
    {
        return Err(invalid(
            "columns",
            "projection must include every cursor column",
        ));
    }
    Ok(())
}

fn source_limits(options: &JsonMap) -> Result<(u64, u64, Duration)> {
    let rows = positive_option(options, "max_batch_rows", 8192)?;
    let bytes = positive_option(options, "max_batch_bytes", 8 * 1024 * 1024)?;
    let poll_ms = positive_option(options, "poll_interval_ms", 1000)?;
    if rows > 1_000_000 || bytes > 1024 * 1024 * 1024 || poll_ms > 3_600_000 {
        return Err(invalid(
            "bounds",
            "maximums are 1000000 rows, 1 GiB, and 3600000 poll milliseconds",
        ));
    }
    Ok((rows, bytes, Duration::from_millis(poll_ms)))
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum SinkMode {
    Append,
    Upsert,
    Transactional,
}

#[derive(Clone)]
pub(super) struct SinkConfig {
    pub connection: ConnectionConfig,
    pub mode: SinkMode,
    pub pipeline: String,
    pub output: String,
    pub rows: u64,
    pub bytes: u64,
}

impl SinkConfig {
    pub fn parse(options: &JsonMap) -> Result<Self> {
        known_options(
            options,
            &[
                "table",
                "tls",
                "timeout_seconds",
                "mode",
                "pipeline",
                "output",
                "max_epoch_rows",
                "max_epoch_bytes",
            ],
        )?;
        let mode = sink_mode(options)?;
        let (rows, bytes) = sink_limits(options)?;
        let connection = sink_connection(options)?;
        Ok(Self {
            connection,
            mode,
            pipeline: sink_identity(options, "pipeline", mode)?,
            output: sink_identity(options, "output", mode)?,
            rows,
            bytes,
        })
    }
}

fn sink_mode(options: &JsonMap) -> Result<SinkMode> {
    match options.get("mode").map(serde_json::Value::as_str) {
        None | Some(Some("append")) => Ok(SinkMode::Append),
        Some(Some("upsert")) => Ok(SinkMode::Upsert),
        Some(Some("transactional")) => Ok(SinkMode::Transactional),
        _ => Err(invalid("mode", "expected append, upsert, or transactional")),
    }
}

fn sink_identity(options: &JsonMap, key: &str, mode: SinkMode) -> Result<String> {
    if mode != SinkMode::Transactional {
        return Ok(String::new());
    }
    let value = required_str(options, key)?;
    if value.is_empty() || value.len() > 128 {
        return Err(invalid(key, "expected 1 to 128 UTF-8 bytes"));
    }
    Ok(value.into())
}

fn sink_limits(options: &JsonMap) -> Result<(u64, u64)> {
    let rows = positive_option(options, "max_epoch_rows", 100_000)?;
    let bytes = positive_option(options, "max_epoch_bytes", 64 * 1024 * 1024)?;
    if rows > 1_000_000 || !(1024..=1024 * 1024 * 1024).contains(&bytes) {
        return Err(invalid(
            "bounds",
            "maximum 1000000 rows; bytes must be between 1024 and 1 GiB",
        ));
    }
    Ok((rows, bytes))
}

fn sink_connection(options: &JsonMap) -> Result<ConnectionConfig> {
    let connection = ConnectionConfig::parse(options)?;
    if connection.table.eq_ignore_ascii_case(super::sink::LEDGER) {
        return Err(invalid("table", "the epoch ledger is reserved"));
    }
    Ok(connection)
}
