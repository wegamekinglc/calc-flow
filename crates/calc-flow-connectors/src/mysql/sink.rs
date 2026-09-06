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
use mysql_async::{Conn, TxOpts, Value, prelude::Queryable};
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
        if self
            .schema
            .as_ref()
            .is_some_and(|current| current != schema)
        {
            return Err(fail("write", "Arrow schema changed within epoch"));
        }
        insert_sql(&self.config, schema.as_ref())?;
        let rows = self
            .rows
            .checked_add(batch.num_rows() as u64)
            .ok_or_else(|| fail("write", "epoch row overflow"))?;
        let memory = self
            .memory
            .checked_add(
                payload
                    .batches()
                    .iter()
                    .map(|record| record.get_array_memory_size() as u64)
                    .sum::<u64>(),
            )
            .ok_or_else(|| fail("write", "epoch byte overflow"))?;
        if rows > self.config.rows || memory > self.config.bytes {
            return Err(fail("write", "epoch exceeds configured bounds"));
        }
        for record in payload.batches() {
            validate_record(record)?;
        }
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
        epoch: Epoch,
        evidence: &JsonMap,
        bytes: &[u8],
    ) -> Result<Vec<RecordBatch>> {
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
            rows = rows
                .checked_add(record.num_rows() as u64)
                .ok_or_else(|| fail("recover", "row count overflow"))?;
            memory = memory
                .checked_add(record.get_array_memory_size() as u64)
                .ok_or_else(|| fail("recover", "byte count overflow"))?;
            if rows > self.config.rows || memory > self.config.bytes {
                return Err(fail("recover", "prepared rows exceed epoch bounds"));
            }
            validate_record(&record)?;
            records.push(record);
        }
        crate::evidence::check_rows(evidence, rows).map_err(protocol)?;
        if !schema.fields().is_empty() {
            insert_sql(&self.config, &schema)?;
        }
        Ok(records)
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
        let write = if let Some(epoch) = epoch {
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
                false
            } else {
                database("commit", timeout, tx.exec_drop(format!("INSERT INTO {LEDGER} (identity_hash, epoch, payload_hash, rows_written) VALUES (UNHEX(?), ?, UNHEX(?), ?)"), (&identity, epoch.as_u64(), &hash, rows))).await?;
                true
            }
        } else {
            true
        };
        if write {
            for record in records {
                let sql = insert_sql(&self.config, record.schema().as_ref())?;
                for row in 0..record.num_rows() {
                    let values = record
                        .columns()
                        .iter()
                        .map(|array| types::cell(array, row))
                        .collect::<Result<Vec<Value>>>()?;
                    database("write", timeout, tx.exec_drop(&sql, values)).await?;
                }
            }
        }
        database("commit", timeout, tx.commit()).await?;
        self.conn = Some(conn);
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

fn insert_sql(config: &SinkConfig, schema: &Schema) -> Result<String> {
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
    let mut sql = format!(
        "INSERT INTO {} ({}) VALUES ({})",
        identifier(&config.connection.table)?,
        names.join(", "),
        vec!["?"; names.len()].join(", ")
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
        let records = self.validate_prepared(epoch, evidence, &bytes)?;
        self.commit_records(&records, Some(epoch), &bytes).await?;
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
        let records = self.validate_prepared(recovery.epoch(), recovery.pre_commit(), bytes)?;
        self.commit_records(&records, Some(recovery.epoch()), bytes)
            .await
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
            sink.validate_prepared(Epoch::INITIAL, &evidence, &bytes)
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
            sink.validate_prepared(Epoch::INITIAL, &evidence, &bytes)
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
            sink.validate_prepared(Epoch::INITIAL, &malformed, bytes)
                .is_err()
        );
    }
}
