//! The `ClickHouse` sink (feature `clickhouse`).
//!
//! Writes at-least-once batches with a per-epoch stable
//! `insert_deduplication_token` derived from the pipeline/output/epoch
//! identity; the token deduplicates retries, never unconditional
//! exactly-once.

use std::collections::BTreeMap;
use std::time::Duration;

use arrow_json::LineDelimitedWriter;
use async_trait::async_trait;
use calc_flow::{
    Batch, Epoch, JsonMap, Result, SecretResolver, SinkRecovery, StreamSink,
    TransactionalStreamSink,
};
use serde_json::Value;
use sha2::{Digest as _, Sha256};

use super::clickhouse::{
    ch_identifier, fail, redact_url_error, required_string, resolve_clickhouse_url, u64_option,
};

const PREPARED_SEGMENT_ID: &str = "insert-block";
const PREFLIGHT_RESPONSE_LIMIT: usize = 64 * 1024;

/// Data-only configuration for one `ClickHouse` sink.
#[derive(Clone, Debug)]
pub struct ClickHouseSinkConfig {
    /// Target table.
    pub table: String,
    /// Stable pipeline identity for the dedup token.
    pub pipeline: String,
    /// Stable output identity for the dedup token.
    pub output: String,
    /// Request retry deduplication and require a replicated `MergeTree` target.
    pub retry_deduplicated: bool,
    /// Maximum rows staged in one epoch insert block.
    pub max_block_rows: u64,
    /// Maximum encoded JSON bytes staged in one epoch insert block.
    pub max_block_bytes: u64,
}

impl ClickHouseSinkConfig {
    /// Parses the sink configuration.
    ///
    /// # Errors
    ///
    /// Returns [`calc_flow::CalcFlowError::InvalidArgument`] naming the offending
    /// option.
    pub fn from_options(options: &JsonMap) -> Result<Self> {
        if options.contains_key("url_key") {
            return Err(calc_flow::CalcFlowError::InvalidArgument {
                field: "options".into(),
                message: "the endpoint URL must use a secret reference".into(),
            });
        }
        let retry_deduplicated = match options.get("retry_deduplicated") {
            None => false,
            Some(Value::Bool(value)) => *value,
            Some(_) => {
                return Err(calc_flow::CalcFlowError::InvalidArgument {
                    field: "retry_deduplicated".into(),
                    message: "option must be a boolean".into(),
                });
            }
        };
        let max_block_rows = positive_bound(options, "max_block_rows", 8192)?;
        let max_block_bytes = positive_bound(options, "max_block_bytes", 8 * 1024 * 1024)?;
        Ok(Self {
            table: ch_identifier(&required_string(options, "table")?)?,
            pipeline: required_string(options, "pipeline")?,
            output: required_string(options, "output")?,
            retry_deduplicated,
            max_block_rows,
            max_block_bytes,
        })
    }
}

fn positive_bound(options: &JsonMap, key: &str, default: u64) -> Result<u64> {
    let value = u64_option(options, key)?.unwrap_or(default);
    if value == 0 {
        Err(calc_flow::CalcFlowError::InvalidArgument {
            field: key.into(),
            message: "option must be greater than zero".into(),
        })
    } else {
        Ok(value)
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
    active_epoch: Option<Epoch>,
    pending_rows: Vec<String>,
    schema_hash: Option<String>,
    rows: u64,
    pending_bytes: u64,
    endpoint_url: Option<String>,
}

/// Ordinary at-least-once adapter used when retry deduplication is disabled.
pub(crate) struct OrdinaryClickHouseSink {
    inner: ClickHouseSink,
    next_epoch: Epoch,
}

impl OrdinaryClickHouseSink {
    pub(crate) fn new(config: ClickHouseSinkConfig, endpoint_url: String) -> Result<Self> {
        if config.retry_deduplicated {
            return Err(fail(
                "open",
                "retry_deduplicated requires the checkpoint-aware sink",
            ));
        }
        Ok(Self {
            inner: ClickHouseSink::new(config)?.with_endpoint(endpoint_url),
            next_epoch: Epoch::INITIAL,
        })
    }
}

#[async_trait]
impl StreamSink for OrdinaryClickHouseSink {
    async fn open(&mut self) -> Result<()> {
        TransactionalStreamSink::open(&mut self.inner).await
    }

    async fn write(&mut self, batch: &Batch) -> Result<()> {
        let epoch = self.next_epoch;
        let next_epoch = epoch.next()?;
        TransactionalStreamSink::begin_epoch(&mut self.inner, epoch).await?;
        if let Err(error) = TransactionalStreamSink::write(&mut self.inner, batch).await {
            let _ = TransactionalStreamSink::abort(&mut self.inner, epoch, None).await;
            return Err(error);
        }
        let evidence = match TransactionalStreamSink::pre_commit(&mut self.inner, epoch).await {
            Ok(evidence) => evidence,
            Err(error) => {
                let _ = TransactionalStreamSink::abort(&mut self.inner, epoch, None).await;
                return Err(error);
            }
        };
        TransactionalStreamSink::commit(&mut self.inner, epoch, &evidence).await?;
        self.next_epoch = next_epoch;
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        TransactionalStreamSink::close(&mut self.inner).await
    }
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
            schema_hash: None,
            rows: 0,
            pending_bytes: 0,
            endpoint_url: None,
        })
    }

    pub(crate) fn with_endpoint(mut self, endpoint_url: String) -> Self {
        self.endpoint_url = Some(endpoint_url);
        self
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
            let schema_hash = schema_hash(record.schema().as_ref());
            if self
                .schema_hash
                .as_ref()
                .is_some_and(|expected| expected != &schema_hash)
            {
                return Err(fail(
                    "write",
                    "all batches in one epoch must use the same Arrow schema",
                ));
            }
            self.schema_hash = Some(schema_hash);
            let mut writer = LineDelimitedWriter::new(Vec::<u8>::new());
            writer
                .write(record)
                .map_err(|error| fail("write", &error.to_string()))?;
            let encoded = String::from_utf8(writer.into_inner())
                .map_err(|error| fail("write", &error.to_string()))?;
            let lines = encoded.lines().collect::<Vec<_>>();
            if lines.len() != record.num_rows() {
                return Err(fail(
                    "write",
                    "Arrow JSON encoding did not preserve the record row count",
                ));
            }
            for line in lines {
                let line = line.to_string();
                let line_bytes = u64::try_from(line.len()).unwrap_or(u64::MAX);
                let separator = u64::from(!self.pending_rows.is_empty());
                let next_rows = self
                    .rows
                    .checked_add(1)
                    .ok_or_else(|| fail("write", "insert block row count exhausted"))?;
                let next_bytes = self
                    .pending_bytes
                    .checked_add(line_bytes)
                    .and_then(|bytes| bytes.checked_add(separator))
                    .ok_or_else(|| fail("write", "insert block byte count exhausted"))?;
                if next_rows > self.config.max_block_rows
                    || next_bytes > self.config.max_block_bytes
                {
                    return Err(fail("write", "insert block exceeds configured bounds"));
                }
                self.pending_rows.push(line);
                self.rows = next_rows;
                self.pending_bytes = next_bytes;
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
        epoch: Epoch,
        secrets: &dyn SecretResolver,
    ) -> Result<()> {
        let url = resolve_clickhouse_url(secrets, "url")?;
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
        self.schema_hash = None;
        self.rows = 0;
        self.pending_bytes = 0;
        Ok(())
    }

    fn prepared_evidence(&self, epoch: Epoch) -> JsonMap {
        let token = dedup_token(&self.config.pipeline, &self.config.output, epoch.as_u64());
        let insert_block = self.pending_rows.join("\n");
        BTreeMap::from([
            (
                "pipeline".to_string(),
                Value::String(self.config.pipeline.clone()),
            ),
            (
                "output".to_string(),
                Value::String(self.config.output.clone()),
            ),
            (
                "target".to_string(),
                Value::String(self.config.table.clone()),
            ),
            ("epoch".to_string(), Value::from(epoch.as_u64())),
            ("rows".to_string(), Value::from(self.rows)),
            ("token".to_string(), Value::String(token)),
            (
                "segment_id".to_string(),
                Value::String(PREPARED_SEGMENT_ID.into()),
            ),
            (
                "segment_bytes".to_string(),
                Value::from(u64::try_from(insert_block.len()).unwrap_or(u64::MAX)),
            ),
            (
                "segment_sha256".to_string(),
                Value::String(hex::encode(Sha256::digest(insert_block.as_bytes()))),
            ),
            (
                "schema_hash".to_string(),
                Value::String(
                    self.schema_hash
                        .clone()
                        .unwrap_or_else(|| hex::encode(Sha256::digest([]))),
                ),
            ),
        ])
    }

    fn validate_evidence(
        &self,
        epoch: Epoch,
        evidence: &JsonMap,
        insert_block: String,
    ) -> Result<PreparedInsert> {
        let string = |field: &str| {
            evidence
                .get(field)
                .and_then(Value::as_str)
                .map(str::to_string)
                .ok_or_else(|| fail("commit", &format!("pre-commit field {field:?} is missing")))
        };
        if string("pipeline")? != self.config.pipeline
            || string("output")? != self.config.output
            || string("target")? != self.config.table
        {
            return Err(fail(
                "commit",
                "pre-commit evidence names a different sink identity",
            ));
        }
        if evidence.get("epoch").and_then(Value::as_u64) != Some(epoch.as_u64()) {
            return Err(fail(
                "commit",
                "pre-commit evidence names a different epoch",
            ));
        }
        let expected_token =
            dedup_token(&self.config.pipeline, &self.config.output, epoch.as_u64());
        if string("token")? != expected_token {
            return Err(fail("commit", "pre-commit evidence has an invalid token"));
        }
        if string("segment_id")? != PREPARED_SEGMENT_ID {
            return Err(fail("commit", "pre-commit segment identity is invalid"));
        }
        let expected_bytes = evidence
            .get("segment_bytes")
            .and_then(Value::as_u64)
            .ok_or_else(|| fail("commit", "pre-commit segment byte count is missing"))?;
        if expected_bytes != u64::try_from(insert_block.len()).unwrap_or(u64::MAX) {
            return Err(fail(
                "commit",
                "pre-commit segment byte count does not match its insert block",
            ));
        }
        if string("segment_sha256")? != hex::encode(Sha256::digest(insert_block.as_bytes())) {
            return Err(fail(
                "commit",
                "pre-commit segment checksum does not match its insert block",
            ));
        }
        let schema_hash = string("schema_hash")?;
        if schema_hash.len() != 64 || !schema_hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(fail(
                "commit",
                "pre-commit evidence has an invalid schema hash",
            ));
        }
        let rows = evidence
            .get("rows")
            .and_then(Value::as_u64)
            .ok_or_else(|| fail("commit", "pre-commit row count is missing"))?;
        let actual_rows = if insert_block.is_empty() {
            0
        } else {
            u64::try_from(insert_block.lines().count()).unwrap_or(u64::MAX)
        };
        if rows != actual_rows {
            return Err(fail(
                "commit",
                "pre-commit row count does not match its insert block",
            ));
        }
        Ok(PreparedInsert {
            token: expected_token,
            insert_block,
            rows,
        })
    }

    async fn send_prepared(&self, operation: &str, prepared: &PreparedInsert) -> Result<()> {
        if prepared.rows == 0 {
            return Ok(());
        }
        let endpoint_url = self.endpoint_url.as_ref().ok_or_else(|| {
            fail(
                operation,
                "the clickhouse sink was not opened through its trusted factory",
            )
        })?;
        let sql = format!("INSERT INTO {} FORMAT JSONEachRow", self.config.table);
        let response = self
            .client
            .post(endpoint_url)
            .header("insert_deduplication_token", &prepared.token)
            .body(format!("{sql} {}", prepared.insert_block))
            .send()
            .await
            .map_err(|error| fail(operation, &redact_url_error(&error.to_string())))?;
        if !response.status().is_success() {
            return Err(fail(
                operation,
                &format!("ClickHouse returned status {}", response.status()),
            ));
        }
        Ok(())
    }

    async fn preflight_target(&self) -> Result<()> {
        let endpoint_url = self.endpoint_url.as_ref().ok_or_else(|| {
            fail(
                "open",
                "the clickhouse sink was not opened through its trusted factory",
            )
        })?;
        let mut response = self
            .client
            .post(endpoint_url)
            .query(&[("param_table", self.config.table.as_str())])
            .body(
                "SELECT engine, getSetting('insert_deduplicate') AS insert_deduplicate \
                 FROM system.tables WHERE database = currentDatabase() \
                   AND name = {table:String} FORMAT JSONEachRow",
            )
            .send()
            .await
            .map_err(|error| fail("open", &redact_url_error(&error.to_string())))?;
        let status = response.status();
        if !status.is_success() {
            return Err(fail(
                "open",
                &format!("ClickHouse returned status {status}"),
            ));
        }
        let mut body = Vec::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|error| fail("open", &redact_url_error(&error.to_string())))?
        {
            let next_len = body
                .len()
                .checked_add(chunk.len())
                .ok_or_else(|| fail("open", "ClickHouse preflight response exhausted usize"))?;
            if next_len > PREFLIGHT_RESPONSE_LIMIT {
                return Err(fail("open", "ClickHouse preflight response exceeds 64 KiB"));
            }
            body.extend_from_slice(&chunk);
        }
        let body = std::str::from_utf8(&body)
            .map_err(|_| fail("open", "ClickHouse preflight response is not UTF-8"))?;
        let row: Value = serde_json::from_str(body.trim()).map_err(|_| {
            fail(
                "open",
                "target table does not exist or returned invalid metadata",
            )
        })?;
        let engine = row
            .get("engine")
            .and_then(Value::as_str)
            .ok_or_else(|| fail("open", "target table engine is missing"))?;
        let deduplicate = row
            .get("insert_deduplicate")
            .and_then(|value| value.as_u64().or_else(|| value.as_str()?.parse().ok()))
            .unwrap_or(0);
        if self.config.retry_deduplicated && (!engine.starts_with("Replicated") || deduplicate != 1)
        {
            return Err(fail(
                "open",
                "retry_deduplicated requires a replicated MergeTree target and insert_deduplicate=1",
            ));
        }
        Ok(())
    }
}

struct PreparedInsert {
    token: String,
    insert_block: String,
    rows: u64,
}

#[async_trait]
impl TransactionalStreamSink for ClickHouseSink {
    async fn open(&mut self) -> Result<()> {
        self.preflight_target().await
    }

    async fn begin_epoch(&mut self, epoch: Epoch) -> Result<()> {
        self.active_epoch = Some(epoch);
        self.pending_rows.clear();
        self.schema_hash = None;
        self.rows = 0;
        self.pending_bytes = 0;
        Ok(())
    }

    async fn write(&mut self, batch: &Batch) -> Result<()> {
        self.stage_batch(batch)
    }

    async fn pre_commit(&mut self, epoch: Epoch) -> Result<JsonMap> {
        if self.active_epoch != Some(epoch) {
            return Err(fail("pre_commit", "pre_commit names an inactive epoch"));
        }
        Ok(self.prepared_evidence(epoch))
    }

    async fn pre_commit_segments(&mut self, epoch: Epoch) -> Result<BTreeMap<String, Vec<u8>>> {
        if self.active_epoch != Some(epoch) {
            return Err(fail(
                "pre_commit",
                "pre_commit segments name an inactive epoch",
            ));
        }
        Ok(BTreeMap::from([(
            PREPARED_SEGMENT_ID.into(),
            self.pending_rows.join("\n").into_bytes(),
        )]))
    }

    async fn commit(&mut self, epoch: Epoch, pre_commit: &JsonMap) -> Result<()> {
        let prepared = self.validate_evidence(epoch, pre_commit, self.pending_rows.join("\n"))?;
        self.send_prepared("commit", &prepared).await?;
        self.active_epoch = None;
        self.pending_rows.clear();
        self.schema_hash = None;
        self.rows = 0;
        self.pending_bytes = 0;
        Ok(())
    }

    async fn abort(&mut self, _epoch: Epoch, _pre_commit: Option<&JsonMap>) -> Result<()> {
        self.pending_rows.clear();
        self.schema_hash = None;
        self.active_epoch = None;
        self.rows = 0;
        self.pending_bytes = 0;
        Ok(())
    }

    async fn recover(&mut self, recovery: &SinkRecovery) -> Result<()> {
        let bytes = recovery
            .segments()
            .get(PREPARED_SEGMENT_ID)
            .ok_or_else(|| fail("recover", "prepared insert-block segment is missing"))?;
        let insert_block = String::from_utf8(bytes.clone())
            .map_err(|_| fail("recover", "prepared insert-block segment is not UTF-8"))?;
        let prepared =
            self.validate_evidence(recovery.epoch(), recovery.pre_commit(), insert_block)?;
        self.send_prepared("recover", &prepared).await?;
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

fn schema_hash(schema: &arrow::datatypes::Schema) -> String {
    let mut hasher = Sha256::new();
    for field in schema.fields() {
        hasher.update(field.name().as_bytes());
        hasher.update([0]);
        hasher.update(format!("{:?}", field.data_type()).as_bytes());
        hasher.update([u8::from(field.is_nullable())]);
    }
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use calc_flow::BatchMetadata;

    use super::*;

    fn options() -> JsonMap {
        BTreeMap::from([
            ("table".into(), Value::String("events".into())),
            ("pipeline".into(), Value::String("pipeline".into())),
            ("output".into(), Value::String("output".into())),
        ])
    }

    fn table_batch(values: &[i64]) -> Batch {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            false,
        )]));
        let record =
            RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(values.to_vec()))])
                .unwrap();
        Batch::table(
            vec![record],
            BatchMetadata::new("test", 1, BTreeMap::new()).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn configuration_rejects_legacy_secrets_and_invalid_bounds() {
        let mut candidate = options();
        candidate.insert("url_key".into(), Value::String("legacy".into()));
        assert!(ClickHouseSinkConfig::from_options(&candidate).is_err());

        for field in ["max_block_rows", "max_block_bytes"] {
            let mut candidate = options();
            candidate.insert(field.into(), Value::from(0));
            assert!(
                ClickHouseSinkConfig::from_options(&candidate).is_err(),
                "{field}"
            );
        }

        let mut candidate = options();
        candidate.insert("retry_deduplicated".into(), Value::String("yes".into()));
        assert!(ClickHouseSinkConfig::from_options(&candidate).is_err());
    }

    #[tokio::test]
    async fn empty_epoch_commits_without_an_endpoint() {
        let mut sink =
            ClickHouseSink::new(ClickHouseSinkConfig::from_options(&options()).unwrap()).unwrap();
        let epoch = Epoch::INITIAL;
        sink.begin_epoch(epoch).await.unwrap();
        let evidence = sink.pre_commit(epoch).await.unwrap();
        assert_eq!(evidence["rows"], Value::from(0));
        sink.commit(epoch, &evidence).await.unwrap();
        assert!(sink.active_epoch.is_none());
    }

    #[tokio::test]
    async fn prepared_evidence_round_trips_and_detects_tampering_offline() {
        let mut sink =
            ClickHouseSink::new(ClickHouseSinkConfig::from_options(&options()).unwrap()).unwrap();
        let epoch = Epoch::INITIAL;
        sink.begin_epoch(epoch).await.unwrap();
        sink.write(&table_batch(&[1, 2])).await.unwrap();
        let evidence = sink.pre_commit(epoch).await.unwrap();
        let segments = sink.pre_commit_segments(epoch).await.unwrap();
        let insert_block = String::from_utf8(segments[PREPARED_SEGMENT_ID].clone()).unwrap();
        let prepared = sink
            .validate_evidence(epoch, &evidence, insert_block.clone())
            .unwrap();
        assert_eq!(prepared.rows, 2);

        let mut tampered = evidence.clone();
        tampered.insert("segment_sha256".into(), Value::String("0".repeat(64)));
        assert!(
            sink.validate_evidence(epoch, &tampered, insert_block)
                .is_err()
        );
    }

    #[tokio::test]
    async fn staging_rejects_schema_changes_and_preserves_atomic_bounds() {
        let mut candidate = options();
        candidate.insert("max_block_rows".into(), Value::from(1));
        let mut sink =
            ClickHouseSink::new(ClickHouseSinkConfig::from_options(&candidate).unwrap()).unwrap();
        sink.begin_epoch(Epoch::INITIAL).await.unwrap();
        assert!(sink.write(&table_batch(&[1, 2])).await.is_err());
        assert_eq!(sink.rows, 1);

        sink.abort(Epoch::INITIAL, None).await.unwrap();
        sink.begin_epoch(Epoch::INITIAL).await.unwrap();
        sink.write(&table_batch(&[1])).await.unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "label",
            DataType::Utf8,
            false,
        )]));
        let record =
            RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(vec!["changed"]))])
                .unwrap();
        let changed = Batch::table(
            vec![record],
            BatchMetadata::new("test", 2, BTreeMap::new()).unwrap(),
        )
        .unwrap();
        assert!(sink.write(&changed).await.is_err());
        assert_eq!(sink.rows, 1);
    }
}
