//! The `ClickHouse` sink (feature `clickhouse`).
//!
//! Writes at-least-once batches with a per-epoch stable
//! `insert_deduplication_token` derived from the pipeline/output/epoch
//! identity; the token deduplicates retries, never unconditional
//! exactly-once.

use std::collections::BTreeMap;
use std::time::Duration;

use async_trait::async_trait;
use calc_flow::{
    Batch, CalcFlowError, ConnectorError, ConnectorIdentity, ConnectorOperation, JsonMap, Result,
    SecretHandle, SecretReference, SecretResolver, SinkRecovery, TransactionalStreamSink,
};
use serde_json::Value;
use sha2::{Digest as _, Sha256};

use super::clickhouse::{
    ch_identifier, connector_identity, fail, redact_url_error, required_string,
    resolve_clickhouse_url,
};

/// Data-only configuration for one `ClickHouse` sink.
#[derive(Clone, Debug)]
pub struct ClickHouseSinkConfig {
    /// Secret key holding the endpoint URL.
    pub url_key: String,
    /// Target table.
    pub table: String,
    /// Stable pipeline identity for the dedup token.
    pub pipeline: String,
    /// Stable output identity for the dedup token.
    pub output: String,
}

impl ClickHouseSinkConfig {
    /// Parses the sink configuration.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] naming the offending
    /// option.
    pub fn from_options(options: &JsonMap) -> Result<Self> {
        Ok(Self {
            url_key: required_string(options, "url_key")?,
            table: ch_identifier(&required_string(options, "table")?)?,
            pipeline: required_string(options, "pipeline")?,
            output: required_string(options, "output")?,
        })
    }
}

/// Derives the per-epoch stable, secret-free dedup token.
pub fn dedup_token(pipeline: &str, output: &str, epoch: u64) -> String {
    let digest = Sha256::digest(format!("{pipeline}/{output}/{epoch}").as_bytes());
    format!("calc-flow-{}", hex::encode(&digest[..12]))
}

/// The at-least-once `ClickHouse` sink with per-epoch dedup tokens.
pub struct ClickHouseSink {
    config: ClickHouseSinkConfig,
    client: reqwest::Client,
    active_epoch: Option<calc_flow::Epoch>,
    pending_rows: Vec<String>,
    rows: u64,
}

impl ClickHouseSink {
    /// Builds the sink.
    ///
    /// # Errors
    ///
    /// Returns the configuration error.
    pub fn new(config: ClickHouseSinkConfig) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|error| fail("open", &error.to_string()))?;
        Ok(Self {
            config,
            client,
            active_epoch: None,
            pending_rows: Vec::new(),
            rows: 0,
        })
    }

    /// Writes the staged rows with the epoch's dedup token.
    ///
    /// # Errors
    ///
    /// Returns the connector error on insert failure.
    pub fn stage_batch(&mut self, batch: &Batch) -> Result<()> {
        if self.active_epoch.is_none() {
            return Err(fail("write", "write before begin_epoch"));
        }
        let payload = batch
            .table_payload()
            .map_err(|_| fail("write", "the clickhouse sink writes table batches only"))?;
        for record in payload.batches() {
            for row in 0..record.num_rows() {
                let mut object = serde_json::Map::new();
                for (col, field) in record.schema().fields().iter().enumerate() {
                    let column = record.column(col);
                    let value = arrow_cell_to_json(column, row);
                    object.insert(field.name().clone(), value);
                }
                let line = serde_json::to_string(&Value::Object(object))
                    .map_err(|error| fail("write", &error.to_string()))?;
                self.pending_rows.push(line);
                self.rows += 1;
            }
        }
        Ok(())
    }

    /// Flushes staged rows with the dedup token to `ClickHouse`.
    ///
    /// # Errors
    ///
    /// Returns the connector error when the insert fails.
    pub async fn flush_with_secrets(
        &mut self,
        epoch: calc_flow::Epoch,
        secrets: &dyn SecretResolver,
    ) -> Result<()> {
        let url = resolve_clickhouse_url(secrets, &self.config.url_key)?;
        let token = dedup_token(&self.config.pipeline, &self.config.output, epoch.as_u64());
        let body = self.pending_rows.join("\n");
        let sql = format!("INSERT INTO {} FORMAT JSONEachRow", self.config.table);
        let response = self
            .client
            .post(&url)
            .header("insert_deduplication_token", &token)
            .body(format!("{sql} {body}"))
            .send()
            .await
            .map_err(|error| fail("write", &redact_url_error(&error.to_string())))?;
        if !response.status().is_success() {
            return Err(fail(
                "write",
                &format!("ClickHouse returned status {}", response.status()),
            ));
        }
        self.pending_rows.clear();
        Ok(())
    }
}

#[async_trait]
impl TransactionalStreamSink for ClickHouseSink {
    async fn open(&mut self) -> Result<()> {
        Ok(())
    }

    async fn begin_epoch(&mut self, epoch: calc_flow::Epoch) -> Result<()> {
        self.active_epoch = Some(epoch);
        self.pending_rows.clear();
        self.rows = 0;
        Ok(())
    }

    async fn write(&mut self, _batch: &Batch) -> Result<()> {
        Err(fail("write", "use write_with_secrets for the endpoint URL"))
    }

    async fn pre_commit(&mut self, epoch: calc_flow::Epoch) -> Result<JsonMap> {
        if self.active_epoch != Some(epoch) {
            return Err(fail("pre_commit", "pre_commit names an inactive epoch"));
        }
        Ok(BTreeMap::from([
            (
                "pipeline".to_string(),
                Value::String(self.config.pipeline.clone()),
            ),
            (
                "output".to_string(),
                Value::String(self.config.output.clone()),
            ),
            ("epoch".to_string(), Value::from(epoch.as_u64())),
            ("rows".to_string(), Value::from(self.rows)),
        ]))
    }

    async fn commit(&mut self, epoch: calc_flow::Epoch, _pre_commit: &JsonMap) -> Result<()> {
        let _ = (epoch, _pre_commit);
        self.active_epoch = None;
        Ok(())
    }

    async fn abort(
        &mut self,
        _epoch: calc_flow::Epoch,
        _pre_commit: Option<&JsonMap>,
    ) -> Result<()> {
        self.pending_rows.clear();
        self.active_epoch = None;
        Ok(())
    }

    async fn recover(&mut self, recovery: &SinkRecovery) -> Result<()> {
        let evidence = recovery.pre_commit();
        if evidence.get("pipeline").and_then(Value::as_str) != Some(&self.config.pipeline)
            || evidence.get("output").and_then(Value::as_str) != Some(&self.config.output)
        {
            return Err(fail(
                "recover",
                "recovery evidence names a different pipeline/output identity",
            ));
        }
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Converts one Arrow cell into its JSON representation for
/// the `JSONEachRow` insert format insert format.
fn arrow_cell_to_json(column: &dyn arrow::array::Array, row: usize) -> Value {
    use arrow::array::{
        BooleanArray, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array, StringArray,
    };
    if column.is_null(row) {
        return Value::Null;
    }
    match column.data_type() {
        arrow::datatypes::DataType::Boolean => column
            .as_any()
            .downcast_ref::<BooleanArray>()
            .map_or(Value::Null, |a| Value::Bool(a.value(row))),
        arrow::datatypes::DataType::Int16 => column
            .as_any()
            .downcast_ref::<Int16Array>()
            .map_or(Value::Null, |a| Value::from(a.value(row))),
        arrow::datatypes::DataType::Int32 => column
            .as_any()
            .downcast_ref::<Int32Array>()
            .map_or(Value::Null, |a| Value::from(a.value(row))),
        arrow::datatypes::DataType::Int64 => column
            .as_any()
            .downcast_ref::<Int64Array>()
            .map_or(Value::Null, |a| Value::from(a.value(row))),
        arrow::datatypes::DataType::Float32 => column
            .as_any()
            .downcast_ref::<Float32Array>()
            .and_then(|a| serde_json::Number::from_f64(f64::from(a.value(row))).map(Value::Number))
            .unwrap_or(Value::Null),
        arrow::datatypes::DataType::Float64 => column
            .as_any()
            .downcast_ref::<Float64Array>()
            .and_then(|a| serde_json::Number::from_f64(a.value(row)).map(Value::Number))
            .unwrap_or(Value::Null),
        arrow::datatypes::DataType::Utf8 => column
            .as_any()
            .downcast_ref::<StringArray>()
            .map_or(Value::Null, |a| Value::String(a.value(row).to_string())),
        _ => Value::Null,
    }
}
