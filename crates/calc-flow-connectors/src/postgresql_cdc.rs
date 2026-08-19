//! Commit-ordered `PostgreSQL` `pgoutput` source.

use std::collections::{BTreeMap, VecDeque};
use std::str::FromStr as _;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

use arrow::array::{
    ArrayRef, BinaryArray, BooleanArray, Date32Array, FixedSizeBinaryBuilder, Float32Array,
    Float64Array, Int16Array, Int32Array, Int64Array, StringArray, StructArray,
    TimestampMicrosecondArray, UInt32Array, UInt64Array, new_null_array,
};
use arrow::buffer::NullBuffer;
use arrow::datatypes::{DataType, Field, Fields, Schema, TimeUnit};
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use calc_flow::{
    Batch, BatchMetadata, CalcFlowError, Cursor, DurableCursorAcknowledger, Result,
    SourceCapabilities, SourceCheckpointGate, SourceDeliveryCapability, SourceEvent, SourceSchema,
    StreamSource,
};
use pgwire_replication::auth::scram::ScramClient;
use pgwire_replication::error::PgWireError;
use pgwire_replication::protocol::framing::{
    read_backend_message, write_password_message, write_query, write_startup_message,
};
use pgwire_replication::protocol::messages::{parse_auth_request, parse_error_response};
use pgwire_replication::tls::rustls::{MaybeTlsStream, maybe_upgrade_to_tls};
use pgwire_replication::{
    Lsn, ReplicationClient, ReplicationConfig, ReplicationEvent, SslMode, TlsConfig,
};
use serde_json::Value;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::TcpStream;
#[cfg(unix)]
use tokio::net::UnixStream;
use tokio::sync::{Notify, watch};
use tokio_postgres::config::{Host, SslMode as PgSslMode};
use tokio_postgres::types::Type as PgType;
use tokio_postgres::{Client, Row};

use crate::database_types::{PgColumn, arrow_data_type, record_batch};
use crate::postgresql::{
    ConnectionDriver, PgSlotPolicy, PgSourceMode, PostgresSourceConfig, settle_connection,
};

const POSTGRES_EPOCH_UNIX_MICROS: i64 = 946_684_800_000_000;
const SNAPSHOT_OPERATION: &str = "snapshot";

fn cdc_error(operation: &str, detail: impl Into<String>) -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: format!("postgresql.logical_cdc.{operation}"),
        message: detail.into(),
    }
}

/// One standard `pgoutput` source over an existing durable slot.
pub(crate) struct PostgresCdcSource {
    capabilities: SourceCapabilities,
    config: PostgresSourceConfig,
    endpoint_url: String,
    client: Option<ReplicationClient>,
    relations: BTreeMap<u32, Arc<Relation>>,
    transaction: Option<Transaction>,
    pending_events: VecDeque<SourceEvent>,
    open_gate_on_next_poll: bool,
    sequence: u64,
    last_commit_lsn: u64,
    durable_lsn_tx: watch::Sender<u64>,
    durable_lsn_rx: watch::Receiver<u64>,
    acknowledgement: Arc<CdcAcknowledger>,
    checkpoint_gate: Arc<CdcCheckpointGate>,
    snapshot_client: Option<Client>,
    snapshot_driver: Option<ConnectionDriver>,
    snapshot_columns: Vec<SnapshotColumn>,
    snapshot_offset: u64,
    snapshot_lsn: Option<u64>,
    snapshot_time_unix_micros: i64,
}

impl PostgresCdcSource {
    pub(crate) fn new(config: PostgresSourceConfig, endpoint_url: String) -> Result<Self> {
        if config.mode != PgSourceMode::LogicalCdc {
            return Err(cdc_error(
                "open",
                "the dedicated source requires logical_cdc mode",
            ));
        }
        let (durable_lsn_tx, durable_lsn_rx) = watch::channel(0_u64);
        let capabilities = SourceCapabilities {
            replay_positioning: calc_flow::ReplayPositioning::ExactPauseReportAndSeek,
            delivery: SourceDeliveryCapability::Lossless,
            max_batch_rows: usize::try_from(config.max_batch_rows).unwrap_or(usize::MAX),
            max_batch_bytes: usize::try_from(config.max_batch_bytes).unwrap_or(usize::MAX),
            schema: SourceSchema::Exact(configured_cdc_schema(&config)?),
            native_watermarks: calc_flow::NativeWatermarkCapability::NeverEmits,
        };
        let acknowledgement = Arc::new(CdcAcknowledger {
            slot: config.slot.clone().expect("logical CDC validates a slot"),
            last_acknowledged: Mutex::new(0),
            durable_lsn_tx: durable_lsn_tx.clone(),
        });
        let checkpoint_gate = Arc::new(CdcCheckpointGate::new(matches!(
            config.slot_policy,
            Some(PgSlotPolicy::RequireExisting)
        )));
        Ok(Self {
            capabilities,
            config,
            endpoint_url,
            client: None,
            relations: BTreeMap::new(),
            transaction: None,
            pending_events: VecDeque::new(),
            open_gate_on_next_poll: false,
            sequence: 0,
            last_commit_lsn: 0,
            durable_lsn_tx,
            durable_lsn_rx,
            acknowledgement,
            checkpoint_gate,
            snapshot_client: None,
            snapshot_driver: None,
            snapshot_columns: Vec::new(),
            snapshot_offset: 0,
            snapshot_lsn: None,
            snapshot_time_unix_micros: 0,
        })
    }

    // Snapshot bootstrap owns a multi-resource transaction whose cleanup paths
    // must remain visible beside slot and exported-snapshot acquisition.
    // #lizard forgives
    async fn bootstrap_snapshot(&mut self, replace_existing: bool) -> Result<()> {
        self.preflight_relation().await?;
        if replace_existing {
            self.drop_inactive_slot().await?;
        }
        let replication = replication_config(&self.endpoint_url, &self.config, 0)?;
        let export = create_exported_slot(&replication).await?;
        let (client, driver) = crate::postgresql::connect_postgres(&self.endpoint_url).await?;
        let import = async {
            client
                .batch_execute("BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                .await
                .map_err(|error| {
                    cdc_error("snapshot", safe_replication_error(&error.to_string()))
                })?;
            client
                .batch_execute(&format!(
                    "SET TRANSACTION SNAPSHOT {}",
                    pg_literal(&export.snapshot_name)
                ))
                .await
                .map_err(|error| {
                    cdc_error("snapshot", safe_replication_error(&error.to_string()))
                })?;
            load_snapshot_columns(&client, &self.config).await
        }
        .await;
        // The imported transaction now owns the snapshot. Keeping the raw
        // replication connection any longer would unnecessarily hold server
        // resources and prevent the slot from becoming streamable.
        drop(export.stream);
        match import {
            Ok(columns) => {
                self.snapshot_columns = columns;
                self.snapshot_client = Some(client);
                self.snapshot_driver = Some(driver);
                self.snapshot_offset = 0;
                self.snapshot_lsn = Some(export.consistent_lsn);
                self.snapshot_time_unix_micros = unix_time_micros()?;
                self.last_commit_lsn = export.consistent_lsn;
                Ok(())
            }
            Err(error) => {
                drop(client);
                if !driver.is_finished() {
                    driver.abort();
                }
                let _ = driver.await;
                Err(error)
            }
        }
    }

    async fn preflight_relation(&self) -> Result<()> {
        let (client, driver) = crate::postgresql::connect_postgres(&self.endpoint_url).await?;
        let result = validate_cdc_relation(&client, &self.config).await;
        drop(client);
        if !driver.is_finished() {
            driver.abort();
        }
        let _ = driver.await;
        result
    }

    async fn drop_inactive_slot(&self) -> Result<()> {
        let (client, driver) = crate::postgresql::connect_postgres(&self.endpoint_url).await?;
        let result = async {
            let active = client
                .query_opt(
                    "SELECT active FROM pg_replication_slots WHERE slot_name = $1",
                    &[&self.slot()],
                )
                .await
                .map_err(|error| {
                    cdc_error("slot_policy", safe_replication_error(&error.to_string()))
                })?;
            if active.as_ref().is_some_and(|row| row.get::<_, bool>(0)) {
                return Err(cdc_error(
                    "slot_policy",
                    "recreate_with_snapshot refuses to drop an active slot",
                ));
            }
            if active.is_some() {
                client
                    .query_one("SELECT pg_drop_replication_slot($1)", &[&self.slot()])
                    .await
                    .map_err(|error| {
                        cdc_error("slot_policy", safe_replication_error(&error.to_string()))
                    })?;
            }
            Ok(())
        }
        .await;
        drop(client);
        if !driver.is_finished() {
            driver.abort();
        }
        let _ = driver.await;
        result
    }

    // Snapshot pagination validates and advances one atomic source cursor; its
    // terminal and empty-page branches are part of the same protocol step.
    // #lizard forgives
    async fn next_snapshot_batch(&mut self) -> Result<Option<SourceEvent>> {
        let client = self
            .snapshot_client
            .as_ref()
            .ok_or_else(|| cdc_error("snapshot", "snapshot client is not open"))?;
        let limit = i64::try_from(self.config.max_batch_rows)
            .map_err(|_| cdc_error("snapshot", "max_batch_rows exceeds PostgreSQL bigint"))?;
        let offset = i64::try_from(self.snapshot_offset)
            .map_err(|_| cdc_error("snapshot", "snapshot offset exceeds PostgreSQL bigint"))?;
        let selection = self
            .snapshot_columns
            .iter()
            .map(|column| column.pg.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        let rows = client
            .query(
                &format!(
                    "SELECT {selection} FROM {} ORDER BY ctid LIMIT $1 OFFSET $2",
                    self.config.table
                ),
                &[&limit, &offset],
            )
            .await
            .map_err(|error| cdc_error("snapshot", safe_replication_error(&error.to_string())))?;
        if rows.is_empty() {
            self.finish_snapshot().await?;
            return Ok(None);
        }
        self.snapshot_offset = self
            .snapshot_offset
            .checked_add(u64::try_from(rows.len()).unwrap_or(u64::MAX))
            .ok_or_else(|| cdc_error("snapshot", "snapshot offset exhausted"))?;
        let lsn = self
            .snapshot_lsn
            .ok_or_else(|| cdc_error("snapshot", "snapshot consistent LSN is missing"))?;
        let record = snapshot_record_batch(
            &self.config.table,
            &self.snapshot_columns,
            &rows,
            lsn,
            self.snapshot_time_unix_micros,
        )?;
        self.sequence = self
            .sequence
            .checked_add(1)
            .ok_or_else(|| cdc_error("sequence", "source sequence exhausted"))?;
        let batch = Batch::table(
            vec![record],
            BatchMetadata::new(
                "postgresql-cdc",
                self.sequence,
                BTreeMap::from([
                    ("slot".into(), Value::String(self.slot().to_string())),
                    ("snapshot".into(), Value::Bool(true)),
                    ("consistent_lsn".into(), Value::from(lsn)),
                ]),
            )?,
        )?;
        if u64::try_from(batch.estimated_bytes()?).unwrap_or(u64::MAX) > self.config.max_batch_bytes
        {
            return Err(cdc_error(
                "snapshot",
                "decoded snapshot batch exceeds max_batch_bytes",
            ));
        }
        let cursor = cdc_cursor(self.slot(), lsn, self.snapshot_offset, true)?;
        Ok(Some(SourceEvent::Data { batch, cursor }))
    }

    async fn finish_snapshot(&mut self) -> Result<()> {
        if let Some(client) = self.snapshot_client.as_ref() {
            client.batch_execute("COMMIT").await.map_err(|error| {
                cdc_error("snapshot", safe_replication_error(&error.to_string()))
            })?;
        }
        settle_connection(&mut self.snapshot_client, &mut self.snapshot_driver).await?;
        let lsn = self
            .snapshot_lsn
            .take()
            .ok_or_else(|| cdc_error("snapshot", "snapshot consistent LSN is missing"))?;
        self.connect(lsn).await?;
        self.checkpoint_gate.open();
        Ok(())
    }

    async fn connect(&mut self, start_lsn: u64) -> Result<()> {
        self.preflight_slot(start_lsn).await?;
        let config = replication_config(&self.endpoint_url, &self.config, start_lsn)?;
        let client = ReplicationClient::connect(config)
            .await
            .map_err(|error| cdc_error("open", safe_replication_error(&error.to_string())))?;
        self.client = Some(client);
        self.last_commit_lsn = start_lsn;
        let _ = self.durable_lsn_tx.send_if_modified(|current| {
            if *current < start_lsn {
                *current = start_lsn;
                true
            } else {
                false
            }
        });
        Ok(())
    }

    // Slot preflight keeps the complete server-side compatibility matrix at the
    // I/O boundary so no replication task starts from an ambiguous state.
    // #lizard forgives
    async fn preflight_slot(&self, start_lsn: u64) -> Result<()> {
        let (client, driver) = crate::postgresql::connect_postgres(&self.endpoint_url).await?;
        let result = async {
            let row = client
                .query_opt(
                    "SELECT plugin::text, slot_type, database::text, active, \
                     COALESCE(wal_status::text, ''), \
                     COALESCE(confirmed_flush_lsn::text, '0/0'), \
                     COALESCE(restart_lsn::text, '0/0') \
                     FROM pg_replication_slots \
                     WHERE slot_name = $1 AND database = current_database()",
                    &[&self.slot()],
                )
                .await
                .map_err(|error| {
                    cdc_error("preflight", safe_replication_error(&error.to_string()))
                })?
                .ok_or_else(|| cdc_error("preflight", "replication slot does not exist"))?;
            let plugin: String = row.get(0);
            let slot_type: String = row.get(1);
            let database: Option<String> = row.get(2);
            let active: bool = row.get(3);
            let wal_status: String = row.get(4);
            let confirmed = Lsn::parse(row.get::<_, String>(5).as_str())
                .map_err(|_| cdc_error("preflight", "slot confirmed_flush_lsn is invalid"))?
                .as_u64();
            let restart = Lsn::parse(row.get::<_, String>(6).as_str())
                .map_err(|_| cdc_error("preflight", "slot restart_lsn is invalid"))?
                .as_u64();
            if plugin != "pgoutput" || slot_type != "logical" || database.is_none() {
                return Err(cdc_error(
                    "preflight",
                    "slot is not a database-bound logical pgoutput slot",
                ));
            }
            if active {
                return Err(cdc_error("preflight", "replication slot is already active"));
            }
            if wal_status == "lost" {
                return Err(cdc_error(
                    "preflight",
                    "replication slot has lost required WAL",
                ));
            }
            if start_lsn != 0 && confirmed > start_lsn {
                return Err(cdc_error(
                    "preflight",
                    "slot confirmed_flush_lsn is ahead of the durable Calc-Flow cursor",
                ));
            }
            if start_lsn != 0 && restart > start_lsn {
                return Err(cdc_error(
                    "preflight",
                    "required WAL was recycled after the durable Calc-Flow cursor",
                ));
            }
            validate_cdc_relation(&client, &self.config).await
        }
        .await;
        drop(client);
        if !driver.is_finished() {
            driver.abort();
        }
        let _ = driver.await;
        result
    }

    // pgoutput messages form one transaction state machine; branching here is
    // the explicit mapping from wire variants to an atomic committed event.
    // #lizard forgives
    async fn next_committed_transaction(&mut self) -> Result<Option<SourceEvent>> {
        if self.open_gate_on_next_poll {
            self.checkpoint_gate.open();
            self.open_gate_on_next_poll = false;
        }
        if let Some(event) = self.pending_events.pop_front() {
            if self.pending_events.is_empty() {
                self.open_gate_on_next_poll = true;
            }
            return Ok(Some(event));
        }
        loop {
            if self.snapshot_client.is_some()
                && let Some(event) = self.next_snapshot_batch().await?
            {
                return Ok(Some(event));
            }
            let client = self
                .client
                .as_mut()
                .ok_or_else(|| cdc_error("read", "read before open"))?;
            let event = tokio::select! {
                changed = self.durable_lsn_rx.changed() => {
                    changed.map_err(|_| cdc_error("acknowledge", "durable cursor channel closed"))?;
                    client.update_applied_lsn(Lsn::from_u64(*self.durable_lsn_rx.borrow()));
                    continue;
                }
                event = client.recv() => event
                    .map_err(|error| cdc_error("read", safe_replication_error(&error.to_string())))?,
            };
            let Some(event) = event else {
                return Ok(None);
            };
            match event {
                ReplicationEvent::KeepAlive { .. } | ReplicationEvent::Message { .. } => {}
                ReplicationEvent::StoppedAt { .. } => return Ok(None),
                ReplicationEvent::Begin {
                    final_lsn,
                    xid,
                    commit_time_micros,
                } => self.begin_transaction(final_lsn, xid, commit_time_micros)?,
                ReplicationEvent::XLogData { data, .. } => self.decode_change(&data)?,
                ReplicationEvent::Commit {
                    lsn,
                    end_lsn,
                    commit_time_micros,
                } => {
                    if let Some(event) =
                        self.finish_transaction(lsn, end_lsn, commit_time_micros)?
                    {
                        return Ok(Some(event));
                    }
                }
            }
        }
    }

    fn begin_transaction(
        &mut self,
        final_lsn: Lsn,
        xid: u32,
        commit_time_micros: i64,
    ) -> Result<()> {
        if self.transaction.is_some() {
            return Err(cdc_error("decode", "nested pgoutput transaction boundary"));
        }
        self.checkpoint_gate.close();
        self.transaction = Some(Transaction {
            xid,
            final_lsn: final_lsn.as_u64(),
            commit_time_micros,
            changes: Vec::new(),
            buffered_bytes: 0,
        });
        Ok(())
    }

    fn decode_change(&mut self, data: &[u8]) -> Result<()> {
        let mut decoder = Decoder::new(data);
        let tag = decoder.byte()?;
        match tag {
            b'R' => {
                let relation = Arc::new(decoder.relation()?);
                decoder.finish()?;
                if relation.name != self.config.table {
                    return Err(cdc_error(
                        "schema",
                        format!(
                            "publication emitted unexpected relation {:?}.{:?}",
                            relation.namespace, relation.name
                        ),
                    ));
                }
                validate_relation_schema(&self.config, &relation)?;
                if self.config.require_before && relation.replica_identity != b'f' {
                    return Err(cdc_error(
                        "replica_identity",
                        "require_before needs REPLICA IDENTITY FULL",
                    ));
                }
                if let Some(previous) = self.relations.get(&relation.id)
                    && previous.as_ref() != relation.as_ref()
                {
                    return Err(cdc_error(
                        "schema",
                        "relation schema changed while the CDC source was running",
                    ));
                }
                self.relations.insert(relation.id, relation);
            }
            b'I' => {
                let relation = self.relation(decoder.u32()?)?;
                decoder.expect(b'N', "insert new tuple")?;
                let new = decoder.tuple(relation.columns.len())?;
                decoder.finish()?;
                let after = resolve_tuple(&new, None)?;
                let key = relation.key_from(&after);
                self.push_change(Change {
                    operation: Operation::Insert,
                    relation,
                    key,
                    before: None,
                    after: Some(after),
                })?;
            }
            b'U' => self.decode_update(decoder)?,
            b'D' => self.decode_delete(decoder)?,
            b'T' => {
                return Err(cdc_error(
                    "decode",
                    "TRUNCATE is outside the append-only CDC contract",
                ));
            }
            b'Y' | b'O' => {
                // Type and origin messages do not carry row changes. Relation
                // OIDs are still validated against the reviewed type matrix.
            }
            other => {
                return Err(cdc_error(
                    "decode",
                    format!("unsupported pgoutput message tag {other:#x}"),
                ));
            }
        }
        Ok(())
    }

    fn decode_update(&mut self, mut decoder: Decoder<'_>) -> Result<()> {
        let relation = self.relation(decoder.u32()?)?;
        let old_kind = decoder.byte()?;
        let (old, new_tag) = match old_kind {
            b'K' | b'O' => (
                Some((old_kind, decoder.tuple(relation.columns.len())?)),
                decoder.byte()?,
            ),
            b'N' => (None, b'N'),
            other => {
                return Err(cdc_error(
                    "decode",
                    format!("invalid update tuple tag {other:#x}"),
                ));
            }
        };
        if new_tag != b'N' {
            return Err(cdc_error("decode", "update is missing its new tuple"));
        }
        let new = decoder.tuple(relation.columns.len())?;
        decoder.finish()?;
        if self.config.require_before && !matches!(old.as_ref(), Some((b'O', _))) {
            return Err(cdc_error(
                "replica_identity",
                "UPDATE did not include the required complete old row",
            ));
        }
        let old_full = old
            .as_ref()
            .filter(|(kind, _)| *kind == b'O')
            .map(|(_, tuple)| resolve_tuple(tuple, None))
            .transpose()?;
        let after = resolve_tuple(&new, old_full.as_deref())?;
        let key = old
            .as_ref()
            .map(|(_, tuple)| resolve_tuple(tuple, None))
            .transpose()?
            .map_or_else(
                || relation.key_from(&after),
                |values| relation.key_from(&values),
            );
        self.push_change(Change {
            operation: Operation::Update,
            relation,
            key,
            before: old_full,
            after: Some(after),
        })
    }

    fn decode_delete(&mut self, mut decoder: Decoder<'_>) -> Result<()> {
        let relation = self.relation(decoder.u32()?)?;
        let kind = decoder.byte()?;
        if !matches!(kind, b'K' | b'O') {
            return Err(cdc_error("decode", "delete is missing its old/key tuple"));
        }
        let old = decoder.tuple(relation.columns.len())?;
        decoder.finish()?;
        if self.config.require_before && kind != b'O' {
            return Err(cdc_error(
                "replica_identity",
                "DELETE did not include the required complete old row",
            ));
        }
        let values = resolve_tuple(&old, None)?;
        let key = relation.key_from(&values);
        self.push_change(Change {
            operation: Operation::Delete,
            relation,
            key,
            before: (kind == b'O').then_some(values),
            after: None,
        })
    }

    fn relation(&self, id: u32) -> Result<Arc<Relation>> {
        self.relations
            .get(&id)
            .cloned()
            .ok_or_else(|| cdc_error("decode", format!("row change names unknown relation {id}")))
    }

    fn push_change(&mut self, change: Change) -> Result<()> {
        let transaction = self.transaction.as_mut().ok_or_else(|| {
            cdc_error(
                "decode",
                "row change arrived outside a transaction boundary",
            )
        })?;
        let change_bytes = change.encoded_bytes()?;
        transaction.buffered_bytes = transaction
            .buffered_bytes
            .checked_add(change_bytes)
            .ok_or_else(|| cdc_error("bounds", "transaction byte count exhausted usize"))?;
        transaction.changes.push(change);
        if transaction.changes.len()
            > usize::try_from(self.config.max_transaction_rows).unwrap_or(usize::MAX)
        {
            return Err(cdc_error(
                "bounds",
                "one database transaction exceeds max_transaction_rows",
            ));
        }
        if u64::try_from(transaction.buffered_bytes).unwrap_or(u64::MAX)
            > self.config.max_transaction_bytes
        {
            return Err(cdc_error(
                "bounds",
                "one database transaction exceeds max_transaction_bytes",
            ));
        }
        Ok(())
    }

    fn finish_transaction(
        &mut self,
        commit_lsn: Lsn,
        end_lsn: Lsn,
        commit_time_micros: i64,
    ) -> Result<Option<SourceEvent>> {
        let transaction = self
            .transaction
            .take()
            .ok_or_else(|| cdc_error("decode", "commit arrived without begin"))?;
        let end_lsn = end_lsn.as_u64();
        if end_lsn <= self.last_commit_lsn || commit_lsn.as_u64() > end_lsn {
            return Err(cdc_error(
                "cursor",
                "commit LSN did not advance monotonically",
            ));
        }
        if transaction.final_lsn != 0 && transaction.final_lsn != commit_lsn.as_u64() {
            return Err(cdc_error(
                "decode",
                "begin final LSN disagrees with the commit LSN",
            ));
        }
        self.last_commit_lsn = end_lsn;
        if transaction.changes.is_empty() {
            self.checkpoint_gate.open();
            return Ok(None);
        }
        let effective_time = if commit_time_micros == 0 {
            transaction.commit_time_micros
        } else {
            commit_time_micros
        };
        let records = bounded_transaction_records(
            &transaction,
            end_lsn,
            effective_time.saturating_add(POSTGRES_EPOCH_UNIX_MICROS),
            self.config.max_batch_rows,
            self.config.max_batch_bytes,
        )?;
        let batch_count = u64::try_from(records.len()).unwrap_or(u64::MAX);
        for (index, record) in records.into_iter().enumerate() {
            self.sequence = self
                .sequence
                .checked_add(1)
                .ok_or_else(|| cdc_error("sequence", "source sequence exhausted"))?;
            let batch_index = u64::try_from(index)
                .unwrap_or(u64::MAX)
                .checked_add(1)
                .ok_or_else(|| cdc_error("sequence", "transaction batch index exhausted"))?;
            let metadata = BatchMetadata::new(
                "postgresql-cdc",
                self.sequence,
                BTreeMap::from([
                    ("slot".into(), Value::String(self.slot().to_string())),
                    ("transaction_id".into(), Value::from(transaction.xid)),
                    ("commit_lsn".into(), Value::from(end_lsn)),
                    ("transaction_batch".into(), Value::from(batch_index)),
                    ("transaction_batches".into(), Value::from(batch_count)),
                    (
                        "transaction_complete".into(),
                        Value::Bool(batch_index == batch_count),
                    ),
                ]),
            )?;
            let batch = Batch::table(vec![record], metadata)?;
            let cursor = cdc_cursor(self.slot(), end_lsn, batch_index, false)?;
            self.pending_events
                .push_back(SourceEvent::Data { batch, cursor });
        }
        let event = self
            .pending_events
            .pop_front()
            .ok_or_else(|| cdc_error("batch", "committed transaction produced no batches"))?;
        if self.pending_events.is_empty() {
            self.open_gate_on_next_poll = true;
        }
        Ok(Some(event))
    }

    fn slot(&self) -> &str {
        self.config
            .slot
            .as_deref()
            .expect("logical CDC validates a slot")
    }
}

#[async_trait]
impl StreamSource for PostgresCdcSource {
    fn capabilities(&self) -> SourceCapabilities {
        self.capabilities.clone()
    }

    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        let start_lsn = cursor
            .as_ref()
            .map(|cursor| cursor_lsn(cursor, self.slot()))
            .transpose()?
            .unwrap_or(0);
        if start_lsn != 0 {
            self.checkpoint_gate.open();
            return self.connect(start_lsn).await;
        }
        match self
            .config
            .slot_policy
            .expect("logical CDC validates a slot policy")
        {
            PgSlotPolicy::RequireExisting => {
                self.checkpoint_gate.open();
                self.connect(0).await
            }
            PgSlotPolicy::CreateWithSnapshot => self.bootstrap_snapshot(false).await,
            PgSlotPolicy::RecreateWithSnapshot => self.bootstrap_snapshot(true).await,
        }
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        self.next_committed_transaction().await
    }

    fn durable_cursor_acknowledger(&self) -> Option<Arc<dyn DurableCursorAcknowledger>> {
        Some(self.acknowledgement.clone())
    }

    fn checkpoint_gate(&self) -> Option<Arc<dyn SourceCheckpointGate>> {
        Some(self.checkpoint_gate.clone())
    }

    async fn close(&mut self) -> Result<()> {
        if let Some(client) = self.snapshot_client.as_ref() {
            let _ = client.batch_execute("ROLLBACK").await;
        }
        let snapshot_result =
            settle_connection(&mut self.snapshot_client, &mut self.snapshot_driver).await;
        let replication_result = if let Some(client) = self.client.as_mut() {
            client
                .shutdown()
                .await
                .map_err(|error| cdc_error("close", safe_replication_error(&error.to_string())))
        } else {
            Ok(())
        };
        self.client = None;
        snapshot_result.and(replication_result)
    }
}

struct CdcAcknowledger {
    slot: String,
    last_acknowledged: Mutex<u64>,
    durable_lsn_tx: watch::Sender<u64>,
}

struct CdcCheckpointGate {
    ready: AtomicBool,
    notify: Notify,
}

impl CdcCheckpointGate {
    const fn new(ready: bool) -> Self {
        Self {
            ready: AtomicBool::new(ready),
            notify: Notify::const_new(),
        }
    }

    fn open(&self) {
        self.ready.store(true, Ordering::Release);
        self.notify.notify_waiters();
    }

    fn close(&self) {
        self.ready.store(false, Ordering::Release);
    }
}

#[async_trait]
impl SourceCheckpointGate for CdcCheckpointGate {
    async fn wait_ready(&self) -> Result<()> {
        while !self.ready.load(Ordering::Acquire) {
            let notified = self.notify.notified();
            if self.ready.load(Ordering::Acquire) {
                break;
            }
            notified.await;
        }
        Ok(())
    }
}

#[async_trait]
impl DurableCursorAcknowledger for CdcAcknowledger {
    async fn acknowledge(&self, cursor: &Cursor) -> Result<()> {
        let lsn = cursor_lsn(cursor, &self.slot)?;
        let mut current = self
            .last_acknowledged
            .lock()
            .map_err(|_| cdc_error("acknowledge", "durable cursor lock was poisoned"))?;
        if lsn < *current {
            return Err(cdc_error(
                "acknowledge",
                "durable commit LSN attempted to move backwards",
            ));
        }
        if lsn > *current {
            *current = lsn;
            self.durable_lsn_tx
                .send(lsn)
                .map_err(|_| cdc_error("acknowledge", "replication source is closed"))?;
        }
        Ok(())
    }
}

fn cursor_lsn(cursor: &Cursor, slot: &str) -> Result<u64> {
    if cursor.payload().get("slot").and_then(Value::as_str) != Some(slot) {
        return Err(cdc_error(
            "cursor",
            "cursor names a different replication slot",
        ));
    }
    let lsn = cursor
        .payload()
        .get("commit_lsn")
        .and_then(Value::as_u64)
        .ok_or_else(|| cdc_error("cursor", "cursor commit_lsn is missing"))?;
    let order = cursor.order();
    if (order.len() != 8 && order.len() != 16) || order[..8] != lsn.to_be_bytes() {
        return Err(cdc_error(
            "cursor",
            "cursor order does not match commit_lsn",
        ));
    }
    Ok(lsn)
}

fn cdc_cursor(slot: &str, lsn: u64, suffix: u64, snapshot_complete: bool) -> Result<Cursor> {
    let mut order = Vec::with_capacity(16);
    order.extend_from_slice(&lsn.to_be_bytes());
    order.extend_from_slice(&suffix.to_be_bytes());
    Cursor::unbound(
        order,
        BTreeMap::from([
            ("slot".into(), Value::String(slot.to_string())),
            ("commit_lsn".into(), Value::from(lsn)),
            (
                "initial_snapshot_complete".into(),
                Value::Bool(snapshot_complete),
            ),
        ]),
    )
}

fn replication_config(
    endpoint_url: &str,
    source: &PostgresSourceConfig,
    start_lsn: u64,
) -> Result<ReplicationConfig> {
    let parsed = tokio_postgres::Config::from_str(endpoint_url)
        .map_err(|_| cdc_error("open", "connection URL could not be parsed"))?;
    if parsed.get_hosts().len() != 1 || parsed.get_ports().len() > 1 {
        return Err(cdc_error(
            "open",
            "logical_cdc requires exactly one PostgreSQL host and port",
        ));
    }
    let host = match &parsed.get_hosts()[0] {
        Host::Tcp(host) => host.clone(),
        #[cfg(unix)]
        Host::Unix(path) => path
            .to_str()
            .ok_or_else(|| cdc_error("open", "Unix socket path is not UTF-8"))?
            .to_string(),
    };
    let user = parsed
        .get_user()
        .ok_or_else(|| cdc_error("open", "connection URL is missing a user"))?;
    let password = std::str::from_utf8(parsed.get_password().unwrap_or_default())
        .map_err(|_| cdc_error("open", "connection password is not UTF-8"))?;
    let database = parsed
        .get_dbname()
        .ok_or_else(|| cdc_error("open", "connection URL is missing a database"))?;
    let mut tls = TlsConfig::disabled();
    tls.mode = match parsed.get_ssl_mode() {
        PgSslMode::Disable => SslMode::Disable,
        PgSslMode::Prefer => SslMode::Prefer,
        PgSslMode::Require => SslMode::Require,
        _ => {
            return Err(cdc_error(
                "open",
                "unsupported PostgreSQL sslmode for logical_cdc",
            ));
        }
    };
    let mut config = ReplicationConfig::new(
        host,
        user,
        password,
        database,
        source.slot.clone().expect("logical CDC validates a slot"),
        source
            .publication
            .clone()
            .expect("logical CDC validates a publication"),
    )
    .with_start_lsn(Lsn::from_u64(start_lsn))
    .with_tls(tls)
    .with_buffer_size(1);
    config.port = parsed.get_ports().first().copied().unwrap_or(5432);
    Ok(config)
}

trait BootstrapIo: AsyncRead + AsyncWrite + Unpin + Send {}

impl<T: AsyncRead + AsyncWrite + Unpin + Send> BootstrapIo for T {}

type BootstrapStream = Box<dyn BootstrapIo>;

struct SlotExport {
    stream: BootstrapStream,
    consistent_lsn: u64,
    snapshot_name: String,
}

async fn create_exported_slot(config: &ReplicationConfig) -> Result<SlotExport> {
    let result = async {
        let mut stream = connect_bootstrap_stream(config).await?;
        let params = [
            ("user", config.user.as_str()),
            ("database", config.database.as_str()),
            ("replication", "database"),
            ("client_encoding", "UTF8"),
            ("application_name", "calc-flow-slot-bootstrap"),
        ];
        write_startup_message(&mut stream, 196_608, &params).await?;
        authenticate_bootstrap(&mut stream, config).await?;
        write_query(
            &mut stream,
            &format!(
                "CREATE_REPLICATION_SLOT {} LOGICAL pgoutput (SNAPSHOT 'export')",
                config.slot
            ),
        )
        .await?;
        let mut export = None;
        loop {
            let message = read_backend_message(&mut stream).await?;
            match message.tag {
                b'D' => export = Some(parse_slot_export_row(&message.payload)?),
                b'E' => {
                    return Err(PgWireError::Server(parse_error_response(&message.payload)));
                }
                b'Z' => {
                    let (consistent_lsn, snapshot_name) = export.ok_or_else(|| {
                        PgWireError::Protocol(
                            "CREATE_REPLICATION_SLOT returned no exported snapshot row".into(),
                        )
                    })?;
                    return Ok(SlotExport {
                        stream,
                        consistent_lsn,
                        snapshot_name,
                    });
                }
                _ => {}
            }
        }
    }
    .await;
    result.map_err(|error: PgWireError| {
        cdc_error("slot_bootstrap", safe_replication_error(&error.to_string()))
    })
}

async fn connect_bootstrap_stream(
    config: &ReplicationConfig,
) -> std::result::Result<BootstrapStream, PgWireError> {
    #[cfg(unix)]
    if config.is_unix_socket() {
        if config.tls.mode.requires_tls() {
            return Err(PgWireError::Tls(
                "TLS is unavailable for a Unix-domain replication socket".into(),
            ));
        }
        let stream = UnixStream::connect(config.unix_socket_path()).await?;
        return Ok(Box::new(stream));
    }
    let tcp = TcpStream::connect((config.host.as_str(), config.port)).await?;
    let stream: MaybeTlsStream = maybe_upgrade_to_tls(tcp, &config.tls, &config.host).await?;
    Ok(Box::new(stream))
}

async fn authenticate_bootstrap(
    stream: &mut BootstrapStream,
    config: &ReplicationConfig,
) -> std::result::Result<(), PgWireError> {
    loop {
        let message = read_backend_message(stream).await?;
        match message.tag {
            b'R' => {
                let (code, data) = parse_auth_request(&message.payload)?;
                match code {
                    0 => {}
                    3 => {
                        let mut password = config.password.as_bytes().to_vec();
                        password.push(0);
                        write_password_message(stream, &password).await?;
                    }
                    10 => authenticate_bootstrap_scram(stream, config, data).await?,
                    _ => {
                        return Err(PgWireError::Auth(format!(
                            "unsupported authentication method code {code}"
                        )));
                    }
                }
            }
            b'E' => {
                return Err(PgWireError::Server(parse_error_response(&message.payload)));
            }
            b'Z' => return Ok(()),
            _ => {}
        }
    }
}

async fn authenticate_bootstrap_scram(
    stream: &mut BootstrapStream,
    config: &ReplicationConfig,
    mechanisms: &[u8],
) -> std::result::Result<(), PgWireError> {
    if !mechanisms
        .split(|byte| *byte == 0)
        .any(|value| value == b"SCRAM-SHA-256")
    {
        return Err(PgWireError::Auth(
            "server does not offer SCRAM-SHA-256".into(),
        ));
    }
    let scram = ScramClient::new(&config.user);
    let mut initial = b"SCRAM-SHA-256\0".to_vec();
    initial.extend_from_slice(
        &i32::try_from(scram.client_first.len())
            .map_err(|_| PgWireError::Protocol("SCRAM message is too large".into()))?
            .to_be_bytes(),
    );
    initial.extend_from_slice(scram.client_first.as_bytes());
    write_password_message(stream, &initial).await?;
    let server_first = read_bootstrap_auth_data(stream, 11).await?;
    let (client_final, auth_message, salted_password) =
        scram.client_final(&config.password, &String::from_utf8_lossy(&server_first))?;
    write_password_message(stream, client_final.as_bytes()).await?;
    let server_final = read_bootstrap_auth_data(stream, 12).await?;
    ScramClient::verify_server_final(
        &String::from_utf8_lossy(&server_final),
        &salted_password,
        &auth_message,
    )
}

async fn read_bootstrap_auth_data(
    stream: &mut BootstrapStream,
    expected_code: i32,
) -> std::result::Result<Vec<u8>, PgWireError> {
    loop {
        let message = read_backend_message(stream).await?;
        match message.tag {
            b'R' => {
                let (code, data) = parse_auth_request(&message.payload)?;
                if code == expected_code {
                    return Ok(data.to_vec());
                }
                return Err(PgWireError::Auth(format!(
                    "unexpected authentication code {code}, expected {expected_code}"
                )));
            }
            b'E' => {
                return Err(PgWireError::Server(parse_error_response(&message.payload)));
            }
            _ => {}
        }
    }
}

fn parse_slot_export_row(payload: &[u8]) -> std::result::Result<(u64, String), PgWireError> {
    let mut offset = 0;
    let count = read_bootstrap_u16(payload, &mut offset)?;
    let mut values = Vec::with_capacity(usize::from(count));
    for _ in 0..count {
        let length = read_bootstrap_i32(payload, &mut offset)?;
        if length < 0 {
            values.push(None);
            continue;
        }
        let length = usize::try_from(length)
            .map_err(|_| PgWireError::Protocol("negative DataRow length".into()))?;
        let end = offset
            .checked_add(length)
            .ok_or_else(|| PgWireError::Protocol("DataRow length overflow".into()))?;
        let bytes = payload
            .get(offset..end)
            .ok_or_else(|| PgWireError::Protocol("truncated DataRow value".into()))?;
        let value = std::str::from_utf8(bytes)
            .map_err(|_| PgWireError::Protocol("DataRow value is not UTF-8".into()))?;
        values.push(Some(value.to_string()));
        offset = end;
    }
    if offset != payload.len() || values.len() != 4 {
        return Err(PgWireError::Protocol(
            "unexpected CREATE_REPLICATION_SLOT row shape".into(),
        ));
    }
    let consistent_lsn = values
        .get(1)
        .and_then(Option::as_deref)
        .ok_or_else(|| PgWireError::Protocol("slot consistent point is null".into()))?;
    let snapshot_name = values
        .get(2)
        .and_then(Option::as_deref)
        .ok_or_else(|| PgWireError::Protocol("slot exported snapshot is null".into()))?;
    let lsn = Lsn::parse(consistent_lsn)
        .map_err(|_| PgWireError::Protocol("slot consistent point is invalid".into()))?;
    Ok((lsn.as_u64(), snapshot_name.to_string()))
}

fn read_bootstrap_u16(payload: &[u8], offset: &mut usize) -> std::result::Result<u16, PgWireError> {
    let end = offset
        .checked_add(2)
        .ok_or_else(|| PgWireError::Protocol("DataRow offset overflow".into()))?;
    let bytes = payload
        .get(*offset..end)
        .ok_or_else(|| PgWireError::Protocol("truncated DataRow field count".into()))?;
    *offset = end;
    Ok(u16::from_be_bytes([bytes[0], bytes[1]]))
}

fn read_bootstrap_i32(payload: &[u8], offset: &mut usize) -> std::result::Result<i32, PgWireError> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| PgWireError::Protocol("DataRow offset overflow".into()))?;
    let bytes = payload
        .get(*offset..end)
        .ok_or_else(|| PgWireError::Protocol("truncated DataRow value length".into()))?;
    *offset = end;
    Ok(i32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn safe_replication_error(message: &str) -> String {
    message
        .split_whitespace()
        .take(8)
        .map(|part| {
            if part.contains("://") || part.to_ascii_lowercase().contains("password") {
                "<redacted>"
            } else {
                part
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[derive(Clone)]
struct SnapshotColumn {
    pg: PgColumn,
    key: bool,
}

async fn validate_cdc_relation(client: &Client, config: &PostgresSourceConfig) -> Result<()> {
    let row = client
        .query_one(
            "SELECT EXISTS (SELECT 1 FROM pg_publication_tables \
             WHERE pubname = $1 AND tablename = $2), \
             (SELECT c.relreplident::text FROM pg_class c \
              JOIN pg_namespace n ON n.oid = c.relnamespace \
              WHERE c.relname = $2 \
                AND n.nspname = ANY(current_schemas(false)) LIMIT 1)",
            &[&config.publication.as_deref(), &config.table],
        )
        .await
        .map_err(|error| cdc_error("preflight", safe_replication_error(&error.to_string())))?;
    if !row.get::<_, bool>(0) {
        return Err(cdc_error(
            "preflight",
            "publication does not include the configured table",
        ));
    }
    if config.require_before && row.get::<_, Option<String>>(1).as_deref() != Some("f") {
        return Err(cdc_error(
            "preflight",
            "require_before needs REPLICA IDENTITY FULL",
        ));
    }
    Ok(())
}

async fn load_snapshot_columns(
    client: &Client,
    config: &PostgresSourceConfig,
) -> Result<Vec<SnapshotColumn>> {
    let rows = client
        .query(
            "SELECT a.attname, format_type(a.atttypid, a.atttypmod), NOT a.attnotnull, \
             EXISTS (SELECT 1 FROM pg_index i WHERE i.indrelid = c.oid \
                     AND i.indisprimary AND a.attnum = ANY(i.indkey)) \
             FROM pg_attribute a \
             JOIN pg_class c ON c.oid = a.attrelid \
             JOIN pg_namespace n ON n.oid = c.relnamespace \
             WHERE c.relname = $1 AND n.nspname = ANY(current_schemas(false)) \
               AND c.relkind IN ('r', 'p') AND a.attnum > 0 AND NOT a.attisdropped \
             ORDER BY a.attnum",
            &[&config.table],
        )
        .await
        .map_err(|error| cdc_error("snapshot", safe_replication_error(&error.to_string())))?;
    if rows.is_empty() {
        return Err(cdc_error(
            "snapshot",
            "configured table is missing or has no visible columns",
        ));
    }
    let columns = rows
        .into_iter()
        .map(|row| {
            let data_type = crate::postgresql::parse_pg_type(row.get::<_, String>(1).as_str());
            arrow_data_type(&data_type)
                .map_err(|error| cdc_error("snapshot", error.to_string()))?;
            Ok(SnapshotColumn {
                pg: PgColumn {
                    name: row.get(0),
                    data_type,
                    nullable: row.get(2),
                },
                key: row.get(3),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    validate_frozen_snapshot_schema(config, &columns)?;
    Ok(columns)
}

fn validate_frozen_snapshot_schema(
    config: &PostgresSourceConfig,
    columns: &[SnapshotColumn],
) -> Result<()> {
    let expected = crate::arrow_schema::schema_from_spec(&config.columns)?;
    if expected.fields().len() != columns.len() {
        return Err(cdc_error(
            "schema",
            "database columns differ from the frozen logical_cdc schema",
        ));
    }
    for (field, column) in expected.fields().iter().zip(columns) {
        let actual = arrow_data_type(&column.pg.data_type)
            .map_err(|error| cdc_error("schema", error.to_string()))?;
        if field.name() != &column.pg.name
            || field.data_type() != &actual
            || field.is_nullable() != column.pg.nullable
        {
            return Err(cdc_error(
                "schema",
                format!(
                    "database column {:?} differs from the frozen schema",
                    column.pg.name
                ),
            ));
        }
    }
    Ok(())
}

fn snapshot_record_batch(
    relation: &str,
    columns: &[SnapshotColumn],
    rows: &[Row],
    consistent_lsn: u64,
    snapshot_time_unix_micros: i64,
) -> Result<RecordBatch> {
    let pg_columns = columns
        .iter()
        .map(|column| column.pg.clone())
        .collect::<Vec<_>>();
    let values = record_batch(&pg_columns, rows)
        .map_err(|error| cdc_error("snapshot", error.to_string()))?;
    let nested_fields = Fields::from(
        columns
            .iter()
            .map(|column| {
                arrow_data_type(&column.pg.data_type).map(|data_type| {
                    // One shared nested schema serves key/before/after. Its
                    // children must therefore be nullable even when the table
                    // column itself is NOT NULL, because non-key and missing
                    // before values are represented as null children.
                    Arc::new(Field::new(&column.pg.name, data_type, true))
                })
            })
            .collect::<Result<Vec<_>>>()?,
    );
    let row_count = rows.len();
    let after = StructArray::try_new(
        nested_fields.clone(),
        values.columns().to_vec(),
        Some(NullBuffer::from(vec![true; row_count])),
    )
    .map_err(|error| cdc_error("snapshot", error.to_string()))?;
    let key_present = columns.iter().any(|column| column.key);
    let key = StructArray::try_new(
        nested_fields.clone(),
        columns
            .iter()
            .enumerate()
            .map(|(index, column)| {
                if column.key {
                    values.column(index).clone()
                } else {
                    new_null_array(values.column(index).data_type(), row_count)
                }
            })
            .collect(),
        Some(NullBuffer::from(vec![key_present; row_count])),
    )
    .map_err(|error| cdc_error("snapshot", error.to_string()))?;
    let before = StructArray::try_new(
        nested_fields.clone(),
        values
            .columns()
            .iter()
            .map(|array| new_null_array(array.data_type(), row_count))
            .collect(),
        Some(NullBuffer::from(vec![false; row_count])),
    )
    .map_err(|error| cdc_error("snapshot", error.to_string()))?;
    let schema = cdc_event_schema(nested_fields);
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec![SNAPSHOT_OPERATION; row_count])),
            Arc::new(StringArray::from(vec![relation; row_count])),
            Arc::new(UInt32Array::from(vec![0; row_count])),
            Arc::new(UInt64Array::from(vec![consistent_lsn; row_count])),
            Arc::new(
                TimestampMicrosecondArray::from(vec![snapshot_time_unix_micros; row_count])
                    .with_timezone_utc(),
            ),
            Arc::new(key),
            Arc::new(before),
            Arc::new(after),
        ],
    )
    .map_err(|error| cdc_error("snapshot", error.to_string()))
}

fn cdc_event_schema(nested_fields: Fields) -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("operation", DataType::Utf8, false),
        Field::new("relation", DataType::Utf8, false),
        Field::new("transaction_id", DataType::UInt32, false),
        Field::new("commit_lsn", DataType::UInt64, false),
        Field::new(
            "commit_time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("+00:00".into())),
            false,
        ),
        Field::new("key", DataType::Struct(nested_fields.clone()), true),
        Field::new("before", DataType::Struct(nested_fields.clone()), true),
        Field::new("after", DataType::Struct(nested_fields), true),
    ]))
}

fn configured_cdc_schema(config: &PostgresSourceConfig) -> Result<Arc<Schema>> {
    let table_schema = crate::arrow_schema::schema_from_spec(&config.columns)?;
    let nested_fields = Fields::from(
        table_schema
            .fields()
            .iter()
            .map(|field| Arc::new(Field::new(field.name(), field.data_type().clone(), true)))
            .collect::<Vec<_>>(),
    );
    Ok(cdc_event_schema(nested_fields))
}

fn unix_time_micros() -> Result<i64> {
    let duration = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|_| cdc_error("snapshot", "system time precedes the Unix epoch"))?;
    i64::try_from(duration.as_micros())
        .map_err(|_| cdc_error("snapshot", "system time exceeds PostgreSQL timestamp range"))
}

fn pg_literal(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Relation {
    id: u32,
    namespace: String,
    name: String,
    replica_identity: u8,
    columns: Vec<RelationColumn>,
}

fn validate_relation_schema(config: &PostgresSourceConfig, relation: &Relation) -> Result<()> {
    let expected = crate::arrow_schema::schema_from_spec(&config.columns)?;
    if expected.fields().len() != relation.columns.len() {
        return Err(cdc_error(
            "schema",
            "pgoutput relation width differs from the frozen schema",
        ));
    }
    for (field, column) in expected.fields().iter().zip(&relation.columns) {
        let actual = arrow_data_type(&column.pg_type)
            .map_err(|error| cdc_error("schema", error.to_string()))?;
        if field.name() != &column.name || field.data_type() != &actual {
            return Err(cdc_error(
                "schema",
                format!(
                    "pgoutput column {:?} differs from the frozen schema",
                    column.name
                ),
            ));
        }
    }
    Ok(())
}

impl Relation {
    fn key_from(&self, values: &[Option<Vec<u8>>]) -> Option<Vec<Option<Vec<u8>>>> {
        self.columns.iter().any(|column| column.key).then(|| {
            self.columns
                .iter()
                .zip(values)
                .map(|(column, value)| column.key.then(|| value.clone()).flatten())
                .collect()
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RelationColumn {
    name: String,
    key: bool,
    pg_type: PgType,
}

struct Transaction {
    xid: u32,
    final_lsn: u64,
    commit_time_micros: i64,
    changes: Vec<Change>,
    buffered_bytes: usize,
}

struct Change {
    operation: Operation,
    relation: Arc<Relation>,
    key: Option<Vec<Option<Vec<u8>>>>,
    before: Option<Vec<Option<Vec<u8>>>>,
    after: Option<Vec<Option<Vec<u8>>>>,
}

impl Change {
    fn encoded_bytes(&self) -> Result<usize> {
        let tuple_bytes = |tuple: &Option<Vec<Option<Vec<u8>>>>| -> Result<usize> {
            tuple.as_ref().map_or(Ok(0), |values| {
                values.iter().try_fold(0_usize, |total, value| {
                    total
                        .checked_add(
                            value
                                .as_ref()
                                .map_or(1, |bytes| bytes.len().saturating_add(1)),
                        )
                        .ok_or_else(|| cdc_error("bounds", "tuple byte count exhausted usize"))
                })
            })
        };
        [
            tuple_bytes(&self.key)?,
            tuple_bytes(&self.before)?,
            tuple_bytes(&self.after)?,
            self.relation.name.len(),
            self.operation.as_str().len(),
        ]
        .into_iter()
        .try_fold(0_usize, |total, bytes| {
            total
                .checked_add(bytes)
                .ok_or_else(|| cdc_error("bounds", "change byte count exhausted usize"))
        })
    }
}

#[derive(Clone, Copy)]
enum Operation {
    Insert,
    Update,
    Delete,
}

impl Operation {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Insert => "insert",
            Self::Update => "update",
            Self::Delete => "delete",
        }
    }
}

#[derive(Clone)]
enum TupleValue {
    Null,
    Unchanged,
    Text(Vec<u8>),
}

fn resolve_tuple(
    tuple: &[TupleValue],
    previous: Option<&[Option<Vec<u8>>]>,
) -> Result<Vec<Option<Vec<u8>>>> {
    tuple
        .iter()
        .enumerate()
        .map(|(index, value)| match value {
            TupleValue::Null => Ok(None),
            TupleValue::Text(value) => Ok(Some(value.clone())),
            TupleValue::Unchanged => previous
                .and_then(|values| values.get(index))
                .cloned()
                .ok_or_else(|| {
                    cdc_error(
                        "decode",
                        "unchanged TOAST value cannot be reconstructed without a complete old row",
                    )
                }),
        })
        .collect()
}

struct Decoder<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Decoder<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn byte(&mut self) -> Result<u8> {
        let value = self
            .bytes
            .get(self.offset)
            .copied()
            .ok_or_else(|| cdc_error("decode", "truncated pgoutput message"))?;
        self.offset += 1;
        Ok(value)
    }

    fn expect(&mut self, expected: u8, field: &str) -> Result<()> {
        let actual = self.byte()?;
        if actual != expected {
            return Err(cdc_error(
                "decode",
                format!("{field} has tag {actual:#x}, expected {expected:#x}"),
            ));
        }
        Ok(())
    }

    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_be_bytes(self.take_array()?))
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_be_bytes(self.take_array()?))
    }

    fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_be_bytes(self.take_array()?))
    }

    fn take_array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let bytes = self.take(N)?;
        bytes
            .try_into()
            .map_err(|_| cdc_error("decode", "truncated pgoutput integer"))
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| cdc_error("decode", "pgoutput length overflow"))?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| cdc_error("decode", "truncated pgoutput payload"))?;
        self.offset = end;
        Ok(value)
    }

    fn string(&mut self) -> Result<String> {
        let tail = self
            .bytes
            .get(self.offset..)
            .ok_or_else(|| cdc_error("decode", "truncated pgoutput string"))?;
        let len = tail
            .iter()
            .position(|byte| *byte == 0)
            .ok_or_else(|| cdc_error("decode", "unterminated pgoutput string"))?;
        let value = std::str::from_utf8(&tail[..len])
            .map_err(|_| cdc_error("decode", "pgoutput identifier is not UTF-8"))?
            .to_string();
        self.offset += len + 1;
        Ok(value)
    }

    // Relation decoding is an exhaustive, bounds-checked wire parser kept
    // contiguous so a partially decoded schema can never escape.
    // #lizard forgives
    fn relation(&mut self) -> Result<Relation> {
        let id = self.u32()?;
        let namespace = self.string()?;
        let name = self.string()?;
        let replica_identity = self.byte()?;
        let count = usize::from(self.u16()?);
        let columns = (0..count)
            .map(|_| {
                let flags = self.byte()?;
                let name = self.string()?;
                let oid = self.u32()?;
                let _type_modifier = self.i32()?;
                let pg_type = PgType::from_oid(oid).ok_or_else(|| {
                    cdc_error("schema", format!("unknown PostgreSQL type OID {oid}"))
                })?;
                arrow_data_type(&pg_type)
                    .map_err(|error| cdc_error("schema", error.to_string()))?;
                Ok(RelationColumn {
                    name,
                    key: flags & 1 == 1,
                    pg_type,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Relation {
            id,
            namespace,
            name,
            replica_identity,
            columns,
        })
    }

    fn tuple(&mut self, width: usize) -> Result<Vec<TupleValue>> {
        let count = usize::from(self.u16()?);
        if count != width {
            return Err(cdc_error(
                "decode",
                "tuple width disagrees with its relation schema",
            ));
        }
        (0..count)
            .map(|_| match self.byte()? {
                b'n' => Ok(TupleValue::Null),
                b'u' => Ok(TupleValue::Unchanged),
                b't' => {
                    let len = usize::try_from(self.i32()?)
                        .map_err(|_| cdc_error("decode", "negative tuple value length"))?;
                    Ok(TupleValue::Text(self.take(len)?.to_vec()))
                }
                b'b' => Err(cdc_error(
                    "decode",
                    "binary pgoutput tuples are not enabled",
                )),
                other => Err(cdc_error(
                    "decode",
                    format!("unknown tuple value tag {other:#x}"),
                )),
            })
            .collect()
    }

    fn finish(&self) -> Result<()> {
        if self.offset != self.bytes.len() {
            return Err(cdc_error("decode", "pgoutput message has trailing bytes"));
        }
        Ok(())
    }
}

fn bounded_transaction_records(
    transaction: &Transaction,
    commit_lsn: u64,
    commit_time_unix_micros: i64,
    max_batch_rows: u64,
    max_batch_bytes: u64,
) -> Result<Vec<RecordBatch>> {
    let row_limit = usize::try_from(max_batch_rows)
        .map_err(|_| cdc_error("bounds", "max_batch_rows does not fit this platform"))?;
    let mut ranges = (0..transaction.changes.len())
        .step_by(row_limit)
        .map(|start| (start, (start + row_limit).min(transaction.changes.len())))
        .collect::<VecDeque<_>>();
    let mut records = Vec::new();
    while let Some((start, end)) = ranges.pop_front() {
        let record = transaction_record_batch(
            transaction,
            &transaction.changes[start..end],
            commit_lsn,
            commit_time_unix_micros,
        )?;
        let bytes = u64::try_from(record.get_array_memory_size()).unwrap_or(u64::MAX);
        if bytes > max_batch_bytes {
            if end - start == 1 {
                return Err(cdc_error("bounds", "one CDC row exceeds max_batch_bytes"));
            }
            let middle = start + (end - start) / 2;
            ranges.push_front((middle, end));
            ranges.push_front((start, middle));
        } else {
            records.push(record);
        }
    }
    Ok(records)
}

fn transaction_record_batch(
    transaction: &Transaction,
    changes: &[Change],
    commit_lsn: u64,
    commit_time_unix_micros: i64,
) -> Result<RecordBatch> {
    let relation = changes
        .first()
        .map(|change| Arc::clone(&change.relation))
        .ok_or_else(|| cdc_error("decode", "empty transaction cannot build a batch"))?;
    if changes
        .iter()
        .any(|change| change.relation.as_ref() != relation.as_ref())
    {
        return Err(cdc_error(
            "schema",
            "one transaction changed multiple relation schemas",
        ));
    }
    let nested_fields = relation
        .columns
        .iter()
        .map(|column| {
            arrow_data_type(&column.pg_type)
                .map(|data_type| Arc::new(Field::new(&column.name, data_type, true)))
                .map_err(|error| cdc_error("schema", error.to_string()))
        })
        .collect::<Result<Vec<_>>>()?;
    let nested_fields = Fields::from(nested_fields);
    let operations = StringArray::from(
        changes
            .iter()
            .map(|change| change.operation.as_str())
            .collect::<Vec<_>>(),
    );
    let relations = StringArray::from(vec![relation.name.as_str(); changes.len()]);
    let xids = UInt32Array::from(vec![transaction.xid; changes.len()]);
    let lsns = UInt64Array::from(vec![commit_lsn; changes.len()]);
    let times = TimestampMicrosecondArray::from(vec![commit_time_unix_micros; changes.len()])
        .with_timezone_utc();
    let key = change_struct_array(
        &relation,
        &nested_fields,
        changes.iter().map(|change| change.key.as_ref()),
    )?;
    let before = change_struct_array(
        &relation,
        &nested_fields,
        changes.iter().map(|change| change.before.as_ref()),
    )?;
    let after = change_struct_array(
        &relation,
        &nested_fields,
        changes.iter().map(|change| change.after.as_ref()),
    )?;
    let schema = cdc_event_schema(nested_fields);
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(operations),
            Arc::new(relations),
            Arc::new(xids),
            Arc::new(lsns),
            Arc::new(times),
            Arc::new(key),
            Arc::new(before),
            Arc::new(after),
        ],
    )
    .map_err(|error| cdc_error("batch", error.to_string()))
}

fn change_struct_array<'a>(
    relation: &Relation,
    fields: &Fields,
    rows: impl Iterator<Item = Option<&'a Vec<Option<Vec<u8>>>>>,
) -> Result<StructArray> {
    let rows = rows.collect::<Vec<_>>();
    let validity = NullBuffer::from(rows.iter().map(Option::is_some).collect::<Vec<_>>());
    let arrays = relation
        .columns
        .iter()
        .enumerate()
        .map(|(index, column)| {
            let values = rows
                .iter()
                .map(|row| {
                    row.and_then(|values| values.get(index))
                        .and_then(Option::as_deref)
                })
                .collect::<Vec<_>>();
            cdc_column_array(column, &values)
        })
        .collect::<Result<Vec<_>>>()?;
    StructArray::try_new(fields.clone(), arrays, Some(validity))
        .map_err(|error| cdc_error("batch", error.to_string()))
}

#[allow(
    clippy::too_many_lines,
    reason = "one exhaustive pgoutput-text-to-Arrow type matrix stays reviewable as a single dispatch"
)]
// The exhaustive type matrix is a single dispatch contract shared by every CDC
// column; splitting it would duplicate null and parse-error semantics.
// #lizard forgives
fn cdc_column_array(column: &RelationColumn, values: &[Option<&[u8]>]) -> Result<ArrayRef> {
    let text = values
        .iter()
        .map(|value| {
            value
                .map(|bytes| {
                    std::str::from_utf8(bytes)
                        .map(str::to_string)
                        .map_err(|_| cdc_error("decode", "text tuple value is not UTF-8"))
                })
                .transpose()
        })
        .collect::<Result<Vec<_>>>()?;
    macro_rules! parsed {
        ($array:ty, $type:ty) => {{
            let values = text
                .iter()
                .map(|value| {
                    value
                        .as_deref()
                        .map(str::parse::<$type>)
                        .transpose()
                        .map_err(|_| {
                            cdc_error(
                                "decode",
                                format!("column {:?} has an invalid value", column.name),
                            )
                        })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(Arc::new(<$array>::from(values)) as ArrayRef)
        }};
    }
    match column.pg_type.clone() {
        PgType::BOOL => {
            let values = text
                .iter()
                .map(|value| match value.as_deref() {
                    None => Ok(None),
                    Some("t" | "true") => Ok(Some(true)),
                    Some("f" | "false") => Ok(Some(false)),
                    Some(_) => Err(cdc_error("decode", "invalid boolean tuple value")),
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(Arc::new(BooleanArray::from(values)))
        }
        PgType::INT2 => parsed!(Int16Array, i16),
        PgType::INT4 => parsed!(Int32Array, i32),
        PgType::INT8 => parsed!(Int64Array, i64),
        PgType::FLOAT4 => parsed!(Float32Array, f32),
        PgType::FLOAT8 => parsed!(Float64Array, f64),
        PgType::TEXT
        | PgType::VARCHAR
        | PgType::BPCHAR
        | PgType::NAME
        | PgType::NUMERIC
        | PgType::JSON
        | PgType::JSONB => Ok(Arc::new(StringArray::from(text))),
        PgType::BYTEA => {
            let decoded = text
                .iter()
                .map(|value| {
                    value
                        .as_deref()
                        .map(|value| {
                            value
                                .strip_prefix("\\x")
                                .ok_or_else(|| cdc_error("decode", "bytea value is not hex"))
                                .and_then(|value| {
                                    hex::decode(value)
                                        .map_err(|_| cdc_error("decode", "invalid bytea hex"))
                                })
                        })
                        .transpose()
                })
                .collect::<Result<Vec<_>>>()?;
            let values = decoded.iter().map(Option::as_deref).collect::<Vec<_>>();
            Ok(Arc::new(BinaryArray::from_opt_vec(values)))
        }
        PgType::DATE => {
            let epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1).expect("epoch is valid");
            let values = text
                .iter()
                .map(|value| {
                    value
                        .as_deref()
                        .map(|value| {
                            chrono::NaiveDate::parse_from_str(value, "%Y-%m-%d")
                                .map(|date| {
                                    i32::try_from((date - epoch).num_days()).unwrap_or(i32::MAX)
                                })
                                .map_err(|_| cdc_error("decode", "invalid date tuple value"))
                        })
                        .transpose()
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(Arc::new(Date32Array::from(values)))
        }
        PgType::TIMESTAMP => {
            let values = text
                .iter()
                .map(|value| value.as_deref().map(parse_timestamp).transpose())
                .collect::<Result<Vec<_>>>()?;
            Ok(Arc::new(TimestampMicrosecondArray::from(values)))
        }
        PgType::TIMESTAMPTZ => {
            let values = text
                .iter()
                .map(|value| value.as_deref().map(parse_timestamptz).transpose())
                .collect::<Result<Vec<_>>>()?;
            Ok(Arc::new(
                TimestampMicrosecondArray::from(values).with_timezone_utc(),
            ))
        }
        PgType::UUID => {
            let mut builder = FixedSizeBinaryBuilder::with_capacity(values.len(), 16);
            for value in text {
                match value {
                    Some(value) => {
                        let uuid = uuid::Uuid::parse_str(&value)
                            .map_err(|_| cdc_error("decode", "invalid UUID tuple value"))?;
                        builder
                            .append_value(uuid.as_bytes())
                            .map_err(|error| cdc_error("batch", error.to_string()))?;
                    }
                    None => builder.append_null(),
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        other => Err(cdc_error(
            "schema",
            format!("unsupported PostgreSQL type {}", other.name()),
        )),
    }
}

fn parse_timestamp(value: &str) -> Result<i64> {
    chrono::NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S%.f")
        .map(|value| value.and_utc().timestamp_micros())
        .map_err(|_| cdc_error("decode", "invalid timestamp tuple value"))
}

fn parse_timestamptz(value: &str) -> Result<i64> {
    for format in ["%Y-%m-%d %H:%M:%S%.f%#z", "%Y-%m-%d %H:%M:%S%.f %:z"] {
        if let Ok(value) = chrono::DateTime::parse_from_str(value, format) {
            return Ok(value.timestamp_micros());
        }
    }
    Err(cdc_error("decode", "invalid timestamptz tuple value"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    // This is a single wire-format acceptance scenario. Its sequential fixture
    // construction is intentionally kept together for protocol readability.
    // #lizard forgives
    fn relation_and_insert_decode_to_typed_event() {
        let relation = relation_message();
        let mut decoder = Decoder::new(&relation[1..]);
        let relation = decoder.relation().unwrap();
        decoder.finish().unwrap();
        assert_eq!(relation.name, "orders");
        assert_eq!(relation.columns[0].pg_type, PgType::INT8);
        assert!(relation.columns[0].key);

        let insert = insert_message();
        let mut decoder = Decoder::new(&insert[1..]);
        assert_eq!(decoder.u32().unwrap(), 42);
        decoder.expect(b'N', "new").unwrap();
        let values = resolve_tuple(&decoder.tuple(2).unwrap(), None).unwrap();
        assert_eq!(values[0].as_deref(), Some(b"7".as_slice()));
        assert_eq!(values[1].as_deref(), Some(b"book".as_slice()));
    }

    #[tokio::test]
    async fn cursor_acknowledgement_is_monotonic_and_slot_bound() {
        let (tx, mut rx) = watch::channel(0);
        let ack = CdcAcknowledger {
            slot: "orders_slot".into(),
            last_acknowledged: Mutex::new(0),
            durable_lsn_tx: tx,
        };
        let cursor = Cursor::unbound(
            9_u64.to_be_bytes().to_vec(),
            BTreeMap::from([
                ("slot".into(), Value::String("orders_slot".into())),
                ("commit_lsn".into(), Value::from(9)),
            ]),
        )
        .unwrap();
        ack.acknowledge(&cursor).await.unwrap();
        assert!(rx.has_changed().unwrap());
        assert_eq!(*rx.borrow_and_update(), 9);

        let backwards = Cursor::unbound(
            8_u64.to_be_bytes().to_vec(),
            BTreeMap::from([
                ("slot".into(), Value::String("orders_slot".into())),
                ("commit_lsn".into(), Value::from(8)),
            ]),
        )
        .unwrap();
        assert!(ack.acknowledge(&backwards).await.is_err());
    }

    #[test]
    fn exported_slot_row_is_bounded_and_decoded_exactly() {
        let values = [
            "orders_slot",
            "0/16B6A50",
            "00000003-0000001B-1",
            "pgoutput",
        ];
        let mut row = 4_u16.to_be_bytes().to_vec();
        for value in values {
            row.extend_from_slice(&i32::try_from(value.len()).unwrap().to_be_bytes());
            row.extend_from_slice(value.as_bytes());
        }
        let (lsn, snapshot) = parse_slot_export_row(&row).unwrap();
        assert_eq!(lsn, Lsn::parse("0/16B6A50").unwrap().as_u64());
        assert_eq!(snapshot, "00000003-0000001B-1");
        assert!(parse_slot_export_row(&row[..row.len() - 1]).is_err());
    }

    #[tokio::test]
    async fn checkpoint_gate_stays_closed_until_snapshot_transition() {
        let gate = Arc::new(CdcCheckpointGate::new(false));
        let waiting = {
            let gate = Arc::clone(&gate);
            tokio::spawn(async move { gate.wait_ready().await })
        };
        tokio::task::yield_now().await;
        assert!(!waiting.is_finished());
        gate.open();
        waiting.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn committed_transaction_splits_batches_without_opening_the_checkpoint_gate_midway() {
        let config = PostgresSourceConfig::from_options(&BTreeMap::from([
            ("table".into(), serde_json::json!("orders")),
            ("mode".into(), serde_json::json!("logical_cdc")),
            ("slot".into(), serde_json::json!("orders_slot")),
            (
                "publication".into(),
                serde_json::json!("orders_publication"),
            ),
            ("slot_policy".into(), serde_json::json!("require_existing")),
            ("max_batch_rows".into(), serde_json::json!(2)),
            ("max_batch_bytes".into(), serde_json::json!(65536)),
            ("max_transaction_rows".into(), serde_json::json!(10)),
            ("max_transaction_bytes".into(), serde_json::json!(131_072)),
            (
                "columns".into(),
                serde_json::json!([
                    {"name": "id", "data_type": "int64", "nullable": false},
                    {"name": "label", "data_type": "string", "nullable": false}
                ]),
            ),
        ]))
        .unwrap();
        let mut source = PostgresCdcSource::new(config, "unused".into()).unwrap();
        let relation = Arc::new(Relation {
            id: 42,
            namespace: "public".into(),
            name: "orders".into(),
            replica_identity: b'f',
            columns: vec![
                RelationColumn {
                    name: "id".into(),
                    key: true,
                    pg_type: PgType::INT8,
                },
                RelationColumn {
                    name: "label".into(),
                    key: false,
                    pg_type: PgType::TEXT,
                },
            ],
        });
        source.begin_transaction(Lsn::from_u64(9), 7, 1).unwrap();
        assert!(!source.checkpoint_gate.ready.load(Ordering::Acquire));
        for id in 1..=5 {
            source
                .push_change(Change {
                    operation: Operation::Insert,
                    relation: Arc::clone(&relation),
                    key: Some(vec![Some(id.to_string().into_bytes()), None]),
                    before: None,
                    after: Some(vec![
                        Some(id.to_string().into_bytes()),
                        Some(format!("row-{id}").into_bytes()),
                    ]),
                })
                .unwrap();
        }
        let first = source
            .finish_transaction(Lsn::from_u64(9), Lsn::from_u64(10), 1)
            .unwrap()
            .expect("first batch");
        let SourceEvent::Data { batch, cursor } = first else {
            panic!("transaction emits data")
        };
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(cursor.payload()["commit_lsn"], Value::from(10));
        assert!(!source.checkpoint_gate.ready.load(Ordering::Acquire));

        for expected_rows in [2, 1] {
            let event = source
                .next_committed_transaction()
                .await
                .unwrap()
                .expect("remaining transaction batch");
            let SourceEvent::Data { batch, .. } = event else {
                panic!("transaction emits only data")
            };
            assert_eq!(batch.num_rows(), expected_rows);
            assert!(!source.checkpoint_gate.ready.load(Ordering::Acquire));
        }
        let error = source
            .next_committed_transaction()
            .await
            .expect_err("no network client is installed in the unit test");
        assert!(error.to_string().contains("read before open"), "{error}");
        assert!(source.checkpoint_gate.ready.load(Ordering::Acquire));
    }

    #[test]
    fn cdc_cursor_orders_snapshot_pages_before_later_wal() {
        let first = cdc_cursor("orders_slot", 9, 1, true).unwrap();
        let second = cdc_cursor("orders_slot", 9, 2, true).unwrap();
        let wal = cdc_cursor("orders_slot", 10, 0, false).unwrap();
        assert!(first.order() < second.order());
        assert!(second.order() < wal.order());
        assert_eq!(cursor_lsn(&second, "orders_slot").unwrap(), 9);
    }

    #[test]
    fn exported_snapshot_name_is_quoted_as_a_literal() {
        assert_eq!(pg_literal("0001-a'b"), "'0001-a''b'");
    }

    fn relation_message() -> Vec<u8> {
        let mut bytes = vec![b'R'];
        bytes.extend_from_slice(&42_u32.to_be_bytes());
        bytes.extend_from_slice(b"public\0orders\0");
        bytes.push(b'd');
        bytes.extend_from_slice(&2_u16.to_be_bytes());
        bytes.push(1);
        bytes.extend_from_slice(b"id\0");
        bytes.extend_from_slice(&20_u32.to_be_bytes());
        bytes.extend_from_slice(&(-1_i32).to_be_bytes());
        bytes.push(0);
        bytes.extend_from_slice(b"label\0");
        bytes.extend_from_slice(&25_u32.to_be_bytes());
        bytes.extend_from_slice(&(-1_i32).to_be_bytes());
        bytes
    }

    fn insert_message() -> Vec<u8> {
        let mut bytes = vec![b'I'];
        bytes.extend_from_slice(&42_u32.to_be_bytes());
        bytes.push(b'N');
        bytes.extend_from_slice(&2_u16.to_be_bytes());
        for value in [b"7".as_slice(), b"book".as_slice()] {
            bytes.push(b't');
            bytes.extend_from_slice(&i32::try_from(value.len()).unwrap().to_be_bytes());
            bytes.extend_from_slice(value);
        }
        bytes
    }
}
