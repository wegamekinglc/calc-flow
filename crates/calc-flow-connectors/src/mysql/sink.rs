use super::{
    config::{SinkConfig, SinkMode, identifier},
    connect, database, fail, require_innodb, types,
};
use arrow::{
    datatypes::{Schema, SchemaRef},
    ipc::{reader::StreamReader, writer::StreamWriter},
    record_batch::RecordBatch,
};
use async_trait::async_trait;
use calc_flow::{Batch, Epoch, JsonMap, Result, SinkRecovery, StreamSink, TransactionalStreamSink};
use mysql_async::{Conn, Transaction, TxOpts, Value, prelude::Queryable};
use serde_json::json;
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Cursor as IoCursor,
    sync::Arc,
};

pub(super) const LEDGER: &str = "calc_flow_mysql_epoch_ledger";
const SEGMENT: &str = "prepared-arrow";

pub(super) struct MySqlSink {
    config: SinkConfig,
    url: Option<String>,
    conn: Option<Conn>,
    active: Option<Epoch>,
    records: Vec<RecordBatch>,
    schema: Option<SchemaRef>,
    rows: u64,
    memory: u64,
    prepared: Option<Vec<u8>>,
}

impl MySqlSink {
    pub fn new(config: SinkConfig, url: String) -> Self {
        Self {
            config,
            url: Some(url),
            conn: None,
            active: None,
            records: Vec::new(),
            schema: None,
            rows: 0,
            memory: 0,
            prepared: None,
        }
    }

    async fn open_connection(&mut self) -> Result<()> {
        let url = self
            .url
            .take()
            .ok_or_else(|| fail("open", "sink already opened"))?;
        let mut conn = connect(&url, &self.config.connection, self.config.bytes).await?;
        require_innodb(
            &mut conn,
            &self.config.connection.table,
            self.config.connection.timeout,
        )
        .await?;
        if self.config.mode == SinkMode::Transactional {
            self.prepare_ledger(&mut conn).await?;
        }
        self.conn = Some(conn);
        Ok(())
    }

    async fn prepare_ledger(&self, conn: &mut Conn) -> Result<()> {
        database("open", self.config.connection.timeout, conn.query_drop(format!(
            "CREATE TABLE IF NOT EXISTS {LEDGER} (identity_hash BINARY(32) NOT NULL, epoch BIGINT UNSIGNED NOT NULL, payload_hash BINARY(32) NOT NULL, rows_written BIGINT UNSIGNED NOT NULL, PRIMARY KEY (identity_hash, epoch)) ENGINE=InnoDB"
        ))).await?;
        require_innodb(conn, LEDGER, self.config.connection.timeout).await?;
        let columns: Vec<(String, String, String)> = database("open", self.config.connection.timeout, conn.exec(
            "SELECT COLUMN_NAME, COLUMN_TYPE, IS_NULLABLE FROM information_schema.COLUMNS WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = ? ORDER BY ORDINAL_POSITION", (LEDGER,)
        )).await?;
        validate_ledger_columns(&columns)?;
        let keys: Vec<String> = database("open", self.config.connection.timeout, conn.exec(
            "SELECT COLUMN_NAME FROM information_schema.STATISTICS WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = ? AND INDEX_NAME = 'PRIMARY' ORDER BY SEQ_IN_INDEX", (LEDGER,)
        )).await?;
        if keys != ["identity_hash", "epoch"] {
            return Err(fail("open", "epoch ledger primary key is incompatible"));
        }
        Ok(())
    }

    fn stage(&mut self, batch: &Batch) -> Result<()> {
        if self.prepared.is_some() {
            return Err(fail("write", "epoch is already prepared"));
        }
        let payload = batch
            .table_payload()
            .map_err(|_| fail("write", "MySQL sink requires table batches"))?;
        let schema = payload.schema();
        self.check_staging_schema(schema)?;
        let (rows, memory) = self.staged_totals(batch.num_rows() as u64, payload.batches())?;
        payload.batches().iter().try_for_each(validate_record)?;
        // Validate encoded size before publishing any change to owned state.
        let records = self
            .records
            .iter()
            .chain(payload.batches())
            .cloned()
            .collect::<Vec<_>>();
        encode(&records, schema.as_ref(), self.config.bytes)?;
        self.records = records;
        self.schema = Some(Arc::clone(schema));
        self.rows = rows;
        self.memory = memory;
        Ok(())
    }

    fn check_staging_schema(&self, schema: &SchemaRef) -> Result<()> {
        if self
            .schema
            .as_ref()
            .is_some_and(|current| current != schema)
        {
            return Err(fail("write", "Arrow schema changed within epoch"));
        }
        insert_sql(&self.config, schema.as_ref(), 1).map(drop)
    }

    fn staged_totals(&self, added_rows: u64, records: &[RecordBatch]) -> Result<(u64, u64)> {
        let rows = self
            .rows
            .checked_add(added_rows)
            .ok_or_else(|| fail("write", "epoch row overflow"))?;
        let memory = self
            .memory
            .checked_add(
                records
                    .iter()
                    .map(|record| record.get_array_memory_size() as u64)
                    .sum::<u64>(),
            )
            .ok_or_else(|| fail("write", "epoch byte overflow"))?;
        if rows > self.config.rows || memory > self.config.bytes {
            return Err(fail("write", "epoch exceeds configured bounds"));
        }
        Ok((rows, memory))
    }

    fn reset(&mut self) {
        self.active = None;
        self.records.clear();
        self.schema = None;
        self.rows = 0;
        self.memory = 0;
        self.prepared = None;
    }

    fn evidence(&self, epoch: Epoch, bytes: &[u8]) -> JsonMap {
        let schema = self
            .schema
            .clone()
            .unwrap_or_else(|| Arc::new(Schema::empty()));
        JsonMap::from([
            ("pipeline".into(), json!(self.config.pipeline)),
            ("output".into(), json!(self.config.output)),
            ("target".into(), json!(self.config.connection.table)),
            ("epoch".into(), json!(epoch.as_u64())),
            ("rows".into(), json!(self.rows)),
            ("segment_id".into(), json!(SEGMENT)),
            ("segment_bytes".into(), json!(bytes.len())),
            (
                "segment_sha256".into(),
                json!(crate::evidence::sha256_hex(bytes)),
            ),
            ("schema_hash".into(), json!(schema_hash(&schema))),
        ])
    }

    fn validate_prepared(
        &self,
        operation: &str,
        epoch: Epoch,
        evidence: &JsonMap,
        bytes: &[u8],
    ) -> Result<Vec<RecordBatch>> {
        self.decode_prepared(epoch, evidence, bytes)
            .map_err(|error| in_operation(operation, error))
    }

    fn decode_prepared(
        &self,
        epoch: Epoch,
        evidence: &JsonMap,
        bytes: &[u8],
    ) -> Result<Vec<RecordBatch>> {
        self.check_evidence(epoch, evidence, bytes)?;
        let (schema, records, rows) = self.read_prepared(evidence, bytes)?;
        crate::evidence::check_rows(evidence, rows).map_err(|error| fail("commit", &error))?;
        if !schema.fields().is_empty() {
            insert_sql(&self.config, &schema, 1)?;
        }
        Ok(records)
    }

    fn check_evidence(&self, epoch: Epoch, evidence: &JsonMap, bytes: &[u8]) -> Result<()> {
        let protocol = |error: String| fail("commit", &error);
        crate::evidence::check_identity(
            evidence,
            &self.config.pipeline,
            &self.config.output,
            &self.config.connection.table,
        )
        .map_err(protocol)?;
        crate::evidence::check_epoch(evidence, epoch).map_err(protocol)?;
        crate::evidence::check_segment_id(evidence, SEGMENT).map_err(protocol)?;
        crate::evidence::check_schema_hash(evidence).map_err(protocol)?;
        crate::evidence::check_segment(evidence, bytes).map_err(protocol)?;
        if bytes.len() as u64 > self.config.bytes {
            return Err(fail("recover", "prepared bytes exceed max_epoch_bytes"));
        }
        Ok(())
    }

    fn read_prepared(
        &self,
        evidence: &JsonMap,
        bytes: &[u8],
    ) -> Result<(SchemaRef, Vec<RecordBatch>, u64)> {
        let reader = StreamReader::try_new(IoCursor::new(bytes), None)
            .map_err(|_| fail("recover", "invalid Arrow state segment"))?;
        let schema = reader.schema();
        if evidence.get("schema_hash") != Some(&json!(schema_hash(&schema))) {
            return Err(fail("recover", "prepared schema hash mismatch"));
        }
        let mut records = Vec::new();
        let mut rows = 0_u64;
        let mut memory = 0_u64;
        for record in reader {
            let record = record.map_err(|_| fail("recover", "invalid Arrow state record"))?;
            (rows, memory) = self.prepared_totals(&record, rows, memory)?;
            validate_record(&record)?;
            records.push(record);
        }
        Ok((schema, records, rows))
    }

    fn prepared_totals(&self, record: &RecordBatch, rows: u64, memory: u64) -> Result<(u64, u64)> {
        let rows = rows
            .checked_add(record.num_rows() as u64)
            .ok_or_else(|| fail("recover", "row count overflow"))?;
        let memory = memory
            .checked_add(record.get_array_memory_size() as u64)
            .ok_or_else(|| fail("recover", "byte count overflow"))?;
        if rows > self.config.rows || memory > self.config.bytes {
            return Err(fail("recover", "prepared rows exceed epoch bounds"));
        }
        Ok((rows, memory))
    }

    async fn commit_records(
        &mut self,
        records: &[RecordBatch],
        epoch: Option<Epoch>,
        bytes: &[u8],
    ) -> Result<()> {
        let mut conn = self
            .conn
            .take()
            .ok_or_else(|| fail("commit", "sink is not open"))?;
        let timeout = self.config.connection.timeout;
        let mut tx = database("commit", timeout, conn.start_transaction(TxOpts::default())).await?;
        if self.should_write(&mut tx, epoch, bytes, records).await? {
            self.write_records(&mut tx, records).await?;
        }
        database("commit", timeout, tx.commit()).await?;
        self.conn = Some(conn);
        Ok(())
    }

    async fn should_write(
        &self,
        tx: &mut Transaction<'_>,
        epoch: Option<Epoch>,
        bytes: &[u8],
        records: &[RecordBatch],
    ) -> Result<bool> {
        match epoch {
            Some(epoch) => self.claim_epoch(tx, epoch, bytes, records).await,
            None => Ok(true),
        }
    }

    async fn claim_epoch(
        &self,
        tx: &mut Transaction<'_>,
        epoch: Epoch,
        bytes: &[u8],
        records: &[RecordBatch],
    ) -> Result<bool> {
        let identity = crate::evidence::sha256_hex(
            serde_json::to_string(&(
                &self.config.pipeline,
                &self.config.output,
                &self.config.connection.table,
            ))
            .expect("string tuple serializes")
            .as_bytes(),
        );
        let hash = crate::evidence::sha256_hex(bytes);
        let rows = records
            .iter()
            .map(|record| record.num_rows() as u64)
            .sum::<u64>();
        let timeout = self.config.connection.timeout;
        let existing: Option<(Vec<u8>, u64)> = database("commit", timeout, tx.exec_first(
            format!("SELECT payload_hash, rows_written FROM {LEDGER} WHERE identity_hash = UNHEX(?) AND epoch = ? FOR UPDATE"), (&identity, epoch.as_u64())
        )).await?;
        if let Some((recorded_hash, recorded_rows)) = existing {
            if hex::encode(recorded_hash) != hash || recorded_rows != rows {
                return Err(fail(
                    "commit",
                    "epoch ledger conflicts with prepared content",
                ));
            }
            return Ok(false);
        }
        database("commit", timeout, tx.exec_drop(format!("INSERT INTO {LEDGER} (identity_hash, epoch, payload_hash, rows_written) VALUES (UNHEX(?), ?, UNHEX(?), ?)"), (&identity, epoch.as_u64(), &hash, rows))).await?;
        Ok(true)
    }

    async fn write_records(&self, tx: &mut Transaction<'_>, records: &[RecordBatch]) -> Result<()> {
        for record in records {
            let mut start = 0;
            while start < record.num_rows() {
                let chunk = prepare_insert(&self.config, record, start)?;
                database(
                    "write",
                    self.config.connection.timeout,
                    tx.exec_drop(&chunk.sql, chunk.values),
                )
                .await?;
                start = chunk.end;
            }
        }
        Ok(())
    }

    async fn close_connection(&mut self) -> Result<()> {
        self.reset();
        self.url = None;
        if let Some(conn) = self.conn.take() {
            database("close", self.config.connection.timeout, conn.disconnect()).await?;
        }
        Ok(())
    }
}

fn in_operation(operation: &str, error: calc_flow::CalcFlowError) -> calc_flow::CalcFlowError {
    match error {
        calc_flow::CalcFlowError::Connector(mut error) => {
            error.operation =
                calc_flow::ConnectorOperation::new(operation).expect("static operation");
            calc_flow::CalcFlowError::Connector(error)
        }
        other => other,
    }
}

fn validate_ledger_columns(columns: &[(String, String, String)]) -> Result<()> {
    let expected = [
        ("identity_hash", "binary(32)"),
        ("epoch", "bigint unsigned"),
        ("payload_hash", "binary(32)"),
        ("rows_written", "bigint unsigned"),
    ];
    if columns.len() != expected.len()
        || columns.iter().zip(expected).any(
            |((name, kind, nullable), (expected_name, expected_type))| {
                name != expected_name || kind != expected_type || nullable != "NO"
            },
        )
    {
        return Err(fail("open", "epoch ledger schema is incompatible"));
    }
    Ok(())
}

struct InsertChunk {
    sql: String,
    values: Vec<Value>,
    end: usize,
}

struct InsertBuffer {
    values: Vec<Value>,
    bytes: u64,
}

impl InsertBuffer {
    fn push_row(&mut self, values: Vec<Value>, limit: u64) -> bool {
        // Bound the combined SQL/execute size conservatively: each parameter
        // needs a placeholder, separators, a type tag, and a null-bitmap bit.
        let added = values
            .iter()
            .map(|value| value.bin_len().saturating_add(6))
            .fold(4, u64::saturating_add);
        let bytes = self.bytes.saturating_add(added);
        if bytes > limit {
            return false;
        }
        self.values.extend(values);
        self.bytes = bytes;
        true
    }
}

fn insert_row_limit(columns: usize) -> Result<usize> {
    let parameter_limit = usize::from(u16::MAX);
    if columns == 0 || columns > parameter_limit {
        return Err(fail(
            "write",
            "column count exceeds MySQL statement parameter bounds",
        ));
    }
    Ok((parameter_limit / columns).min(1000))
}

fn row_values(record: &RecordBatch, row: usize) -> Result<Vec<Value>> {
    record
        .columns()
        .iter()
        .map(|array| types::cell(array, row))
        .collect()
}

fn prepare_insert(config: &SinkConfig, record: &RecordBatch, start: usize) -> Result<InsertChunk> {
    let rows_limit = insert_row_limit(record.num_columns())?;
    let sql = insert_sql(config, record.schema().as_ref(), 1)?;
    let mut buffer = InsertBuffer {
        values: Vec::new(),
        bytes: sql.len() as u64 + 64,
    };
    for row in start..record.num_rows().min(start.saturating_add(rows_limit)) {
        if !buffer.push_row(row_values(record, row)?, config.bytes) {
            break;
        }
    }
    let rows = buffer.values.len() / record.num_columns();
    if rows == 0 {
        return Err(fail("write", "row exceeds MySQL statement byte bounds"));
    }
    Ok(InsertChunk {
        sql: insert_sql(config, record.schema().as_ref(), rows)?,
        values: buffer.values,
        end: start + rows,
    })
}

fn insert_sql(config: &SinkConfig, schema: &Schema, rows: usize) -> Result<String> {
    if schema.fields().is_empty() {
        return Err(fail("write", "empty column list"));
    }
    let mut unique = BTreeSet::new();
    let names = schema
        .fields()
        .iter()
        .map(|field| {
            types::validate_type(field.data_type())?;
            if !unique.insert(field.name().to_ascii_lowercase()) {
                return Err(fail("write", "duplicate column names"));
            }
            identifier(field.name())
        })
        .collect::<Result<Vec<_>>>()?;
    let row = format!("({})", vec!["?"; names.len()].join(", "));
    let mut sql = format!(
        "INSERT INTO {} ({}) VALUES {}",
        identifier(&config.connection.table)?,
        names.join(", "),
        vec![row; rows].join(", ")
    );
    if config.mode == SinkMode::Upsert {
        sql.push_str(" AS incoming ON DUPLICATE KEY UPDATE ");
        sql.push_str(
            &names
                .iter()
                .map(|name| format!("{name} = incoming.{name}"))
                .collect::<Vec<_>>()
                .join(", "),
        );
    }
    Ok(sql)
}

fn validate_record(record: &RecordBatch) -> Result<()> {
    for array in record.columns() {
        for row in 0..record.num_rows() {
            types::cell(array, row)?;
        }
    }
    Ok(())
}

fn schema_hash(schema: &Schema) -> String {
    // Arrow metadata uses randomly seeded hash maps. Hash sorted metadata so
    // reconstructing the exact persisted schema cannot change its identity.
    let fields = schema
        .fields()
        .iter()
        .map(|field| {
            (
                field.name(),
                format!("{:?}", field.data_type()),
                field.is_nullable(),
                field.metadata().iter().collect::<BTreeMap<_, _>>(),
            )
        })
        .collect::<Vec<_>>();
    let metadata = schema.metadata().iter().collect::<BTreeMap<_, _>>();
    let bytes = serde_json::to_vec(&(fields, metadata)).expect("data-only schema serializes");
    crate::evidence::sha256_hex(&bytes)
}

fn encode(records: &[RecordBatch], schema: &Schema, limit: u64) -> Result<Vec<u8>> {
    let mut writer = StreamWriter::try_new(Vec::new(), schema)
        .map_err(|_| fail("pre_commit", "cannot encode Arrow schema"))?;
    for record in records {
        writer
            .write(record)
            .map_err(|_| fail("pre_commit", "cannot encode Arrow record"))?;
        if writer.get_ref().len() as u64 > limit {
            return Err(fail("pre_commit", "encoded epoch exceeds max_epoch_bytes"));
        }
    }
    writer
        .finish()
        .map_err(|_| fail("pre_commit", "cannot finish Arrow segment"))?;
    let bytes = writer
        .into_inner()
        .map_err(|_| fail("pre_commit", "cannot finish Arrow segment"))?;
    if bytes.len() as u64 > limit {
        return Err(fail("pre_commit", "encoded epoch exceeds max_epoch_bytes"));
    }
    Ok(bytes)
}

#[async_trait]
impl StreamSink for MySqlSink {
    async fn open(&mut self) -> Result<()> {
        self.open_connection().await
    }
    async fn write(&mut self, batch: &Batch) -> Result<()> {
        self.stage(batch)?;
        let records = self.records.clone();
        self.commit_records(&records, None, &[]).await?;
        self.reset();
        Ok(())
    }
    async fn close(&mut self) -> Result<()> {
        self.close_connection().await
    }
}

#[async_trait]
impl TransactionalStreamSink for MySqlSink {
    async fn open(&mut self) -> Result<()> {
        self.open_connection().await
    }
    async fn begin_epoch(&mut self, epoch: Epoch) -> Result<()> {
        if self.conn.is_none() || self.active.is_some() {
            return Err(fail(
                "begin_epoch",
                "sink must be open with no active epoch",
            ));
        }
        self.active = Some(epoch);
        Ok(())
    }
    async fn write(&mut self, batch: &Batch) -> Result<()> {
        if self.active.is_none() {
            return Err(fail("write", "write requires an active epoch"));
        }
        self.stage(batch)
    }
    async fn pre_commit(&mut self, epoch: Epoch) -> Result<JsonMap> {
        if self.active != Some(epoch) {
            return Err(fail("pre_commit", "inactive epoch"));
        }
        if self.prepared.is_none() {
            self.prepared = Some(encode(
                &self.records,
                self.schema.as_deref().unwrap_or(&Schema::empty()),
                self.config.bytes,
            )?);
        }
        Ok(self.evidence(epoch, self.prepared.as_ref().expect("prepared")))
    }
    async fn pre_commit_segments(&mut self, epoch: Epoch) -> Result<BTreeMap<String, Vec<u8>>> {
        self.pre_commit(epoch).await?;
        Ok(BTreeMap::from([(
            SEGMENT.into(),
            self.prepared.clone().expect("prepared"),
        )]))
    }
    async fn commit(&mut self, epoch: Epoch, evidence: &JsonMap) -> Result<()> {
        if self.active != Some(epoch) {
            return Err(fail("commit", "inactive epoch"));
        }
        let bytes = self
            .prepared
            .clone()
            .ok_or_else(|| fail("commit", "epoch has not been prepared"))?;
        let records = self.validate_prepared("commit", epoch, evidence, &bytes)?;
        self.commit_records(&records, Some(epoch), &bytes)
            .await
            .map_err(|error| in_operation("commit", error))?;
        self.reset();
        Ok(())
    }
    async fn abort(&mut self, epoch: Epoch, _evidence: Option<&JsonMap>) -> Result<()> {
        if self.active.is_some_and(|active| active != epoch) {
            return Err(fail("abort", "inactive epoch"));
        }
        self.reset();
        Ok(())
    }
    async fn recover(&mut self, recovery: &SinkRecovery) -> Result<()> {
        if self.active.is_some() {
            return Err(fail("recover", "cannot recover during an active epoch"));
        }
        let bytes = recovery
            .segments()
            .get(SEGMENT)
            .ok_or_else(|| fail("recover", "missing prepared Arrow segment"))?;
        let records =
            self.validate_prepared("recover", recovery.epoch(), recovery.pre_commit(), bytes)?;
        self.commit_records(&records, Some(recovery.epoch()), bytes)
            .await
            .map_err(|error| in_operation("recover", error))
    }
    async fn close(&mut self) -> Result<()> {
        self.close_connection().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::{
        array::Int64Array,
        datatypes::{DataType, Field},
    };

    fn sink() -> MySqlSink {
        MySqlSink::new(
            SinkConfig::parse(&JsonMap::from([
                ("table".into(), json!("orders")),
                ("mode".into(), json!("transactional")),
                ("pipeline".into(), json!("p")),
                ("output".into(), json!("o")),
            ]))
            .unwrap(),
            String::new(),
        )
    }

    fn batch(name: &str, values: Vec<i64>) -> Batch {
        let schema = Arc::new(Schema::new(vec![Field::new(name, DataType::Int64, false)]));
        Batch::table(
            vec![RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(values))]).unwrap()],
            calc_flow::BatchMetadata::default(),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn prepared_validation_errors_report_the_calling_operation() {
        let mut sink = sink();
        sink.active = Some(Epoch::INITIAL);
        sink.stage(&batch("id", vec![1])).unwrap();
        let evidence = sink.pre_commit(Epoch::INITIAL).await.unwrap();
        let segments = sink.pre_commit_segments(Epoch::INITIAL).await.unwrap();
        let mut bad_schema = evidence.clone();
        bad_schema.insert("schema_hash".into(), json!("0".repeat(64)));
        let error = sink.commit(Epoch::INITIAL, &bad_schema).await.unwrap_err();
        let calc_flow::CalcFlowError::Connector(error) = error else {
            panic!("connector error expected")
        };
        assert_eq!(error.operation.to_string(), "commit");
        sink.abort(Epoch::INITIAL, None).await.unwrap();
        let mut bad_checksum = evidence;
        bad_checksum.insert("segment_sha256".into(), json!("0".repeat(64)));
        let recovery = SinkRecovery::from_parts(
            Epoch::INITIAL,
            false,
            calc_flow::SinkDelivery::Transactional,
            bad_checksum,
        )
        .with_segments(segments);
        let error = sink.recover(&recovery).await.unwrap_err();
        let calc_flow::CalcFlowError::Connector(error) = error else {
            panic!("connector error expected")
        };
        assert_eq!(error.operation.to_string(), "recover");
    }

    #[test]
    fn insert_chunks_preserve_values_and_bound_statement_rows() {
        let input = batch("id", (0..1005).collect());
        let record = &input.table_payload().unwrap().batches()[0];
        let config = sink().config;
        let first = prepare_insert(&config, record, 0).unwrap();
        assert_eq!(first.end, 1000);
        assert_eq!(first.sql.matches('?').count(), 1000);
        assert_eq!(first.values, (0..1000).map(Value::Int).collect::<Vec<_>>());
        let last = prepare_insert(&config, record, first.end).unwrap();
        assert_eq!(last.end, 1005);
        assert_eq!(
            last.values,
            (1000..1005).map(Value::Int).collect::<Vec<_>>()
        );
    }

    #[test]
    fn insert_chunks_respect_binary_packet_and_parameter_limits() {
        let mut config = sink().config;
        config.bytes = 1024;
        let record = RecordBatch::try_from_iter([(
            "payload",
            Arc::new(arrow::array::StringArray::from(vec!["x".repeat(200); 20]))
                as arrow::array::ArrayRef,
        )])
        .unwrap();
        let chunk = prepare_insert(&config, &record, 0).unwrap();
        assert!(chunk.end > 0 && chunk.end < record.num_rows());
        let encoded = chunk
            .values
            .iter()
            .map(|value| value.bin_len() + 3)
            .sum::<u64>();
        assert!(chunk.sql.len() as u64 + encoded + 64 <= config.bytes);
        let oversized = RecordBatch::try_from_iter([(
            "payload",
            Arc::new(arrow::array::StringArray::from(vec!["x".repeat(2048)]))
                as arrow::array::ArrayRef,
        )])
        .unwrap();
        assert!(prepare_insert(&config, &oversized, 0).is_err());
        config.bytes = 64 * 1024 * 1024;
        let columns = (0..100).map(|i| {
            (
                format!("c{i}"),
                Arc::new(Int64Array::from_iter_values(0..1000)) as arrow::array::ArrayRef,
            )
        });
        let wide = RecordBatch::try_from_iter(columns).unwrap();
        let chunk = prepare_insert(&config, &wide, 0).unwrap();
        assert_eq!(chunk.end, 655);
        assert_eq!(chunk.values.len(), 65_500);
        assert_eq!(chunk.sql.matches('?').count(), chunk.values.len());
    }

    #[test]
    fn rejected_batches_do_not_partially_change_the_epoch() {
        let mut sink = sink();
        sink.config.rows = 2;
        sink.stage(&batch("id", vec![1])).unwrap();
        assert!(sink.stage(&batch("id", vec![2, 3])).is_err());
        assert!(sink.stage(&batch("other", vec![2])).is_err());
        assert_eq!(sink.rows, 1);
        sink.stage(&batch("id", vec![2])).unwrap();
        assert_eq!(sink.rows, 2);
        sink.config.bytes = 1;
        assert!(sink.stage(&batch("id", vec![])).is_err());
        assert!(encode(&sink.records, sink.schema.as_ref().unwrap(), 1).is_err());
    }

    #[test]
    fn prepared_schema_hash_survives_metadata_reordering() {
        let metadata = (0..20)
            .map(|index| (format!("key-{index}"), format!("value-{index}")))
            .collect();
        let field = Field::new("id", DataType::Int64, false).with_metadata(metadata);
        let schema = Arc::new(
            Schema::new(vec![field]).with_metadata(
                (0..20)
                    .map(|index| (format!("schema-{index}"), index.to_string()))
                    .collect(),
            ),
        );
        let batch = Batch::table(
            vec![RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![1]))]).unwrap()],
            calc_flow::BatchMetadata::default(),
        )
        .unwrap();
        let mut sink = sink();
        sink.stage(&batch).unwrap();
        let bytes = encode(
            &sink.records,
            sink.schema.as_ref().unwrap(),
            sink.config.bytes,
        )
        .unwrap();
        let evidence = sink.evidence(Epoch::INITIAL, &bytes);
        assert_eq!(
            sink.validate_prepared("recover", Epoch::INITIAL, &evidence, &bytes)
                .unwrap(),
            sink.records
        );
    }

    #[test]
    fn unsupported_null_and_empty_arrays_are_rejected_before_staging() {
        for array in [
            Arc::new(arrow::array::NullArray::new(1)) as arrow::array::ArrayRef,
            Arc::new(arrow::array::LargeStringArray::from(Vec::<&str>::new())),
        ] {
            let schema = Arc::new(Schema::new(vec![Field::new(
                "id",
                array.data_type().clone(),
                true,
            )]));
            let batch = Batch::table(
                vec![RecordBatch::try_new(schema, vec![array]).unwrap()],
                calc_flow::BatchMetadata::default(),
            )
            .unwrap();
            assert!(sink().stage(&batch).is_err());
        }
    }

    #[tokio::test]
    async fn empty_epoch_and_prepared_segments_are_validated_before_io() {
        let mut sink = sink();
        sink.active = Some(Epoch::INITIAL);
        let evidence = sink.pre_commit(Epoch::INITIAL).await.unwrap();
        let bytes = sink.prepared.clone().unwrap();
        assert!(
            sink.validate_prepared("recover", Epoch::INITIAL, &evidence, &bytes)
                .unwrap()
                .is_empty()
        );
        assert!(sink.stage(&batch("id", vec![1])).is_err());
        assert!(sink.abort(Epoch::new(2).unwrap(), None).await.is_err());
        sink.abort(Epoch::INITIAL, None).await.unwrap();
        assert!(sink.pre_commit_segments(Epoch::INITIAL).await.is_err());
        let recovery = SinkRecovery::from_parts(
            Epoch::INITIAL,
            false,
            calc_flow::SinkDelivery::Transactional,
            evidence.clone(),
        );
        assert!(sink.recover(&recovery).await.is_err());
        let mut malformed = evidence;
        let bytes = b"not Arrow";
        malformed.insert("segment_bytes".into(), json!(bytes.len()));
        malformed.insert(
            "segment_sha256".into(),
            json!(crate::evidence::sha256_hex(bytes)),
        );
        assert!(
            sink.validate_prepared("recover", Epoch::INITIAL, &malformed, bytes)
                .is_err()
        );
    }
}
