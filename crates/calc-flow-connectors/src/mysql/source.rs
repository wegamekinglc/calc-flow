use super::{
    config::{SourceConfig, identifier},
    connect, database, fail, require_innodb, types,
};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use calc_flow::{
    Batch, BatchMetadata, Cursor, JsonMap, NativeWatermarkCapability, ReplayPositioning, Result,
    SourceCapabilities, SourceDeliveryCapability, SourceEvent, SourceSchema, StreamSource,
};
use mysql_async::{Conn, Row, Value, prelude::Queryable};
use serde_json::json;
use std::{collections::BTreeMap, fmt::Write as _, sync::Arc};

pub(super) struct MySqlSource {
    config: SourceConfig,
    url: Option<String>,
    conn: Option<Conn>,
    capabilities: SourceCapabilities,
    schema: SchemaRef,
    ordering: Vec<String>,
    values: Vec<Value>,
    sequence: u64,
    offset: u64,
    exhausted: bool,
    idle: bool,
}

impl MySqlSource {
    pub fn new(config: SourceConfig, url: String) -> Result<Self> {
        let schema = crate::arrow_schema::schema_from_spec(&config.columns)?;
        let capabilities = SourceCapabilities {
            replay_positioning: if config.incremental {
                ReplayPositioning::ExactPauseReportAndSeek
            } else {
                ReplayPositioning::Unsupported
            },
            delivery: if config.incremental {
                SourceDeliveryCapability::Lossless
            } else {
                SourceDeliveryCapability::Lossy
            },
            max_batch_rows: usize::try_from(config.rows).unwrap_or(usize::MAX),
            max_batch_bytes: usize::try_from(config.bytes).unwrap_or(usize::MAX),
            schema: if config.columns.is_empty() {
                SourceSchema::DynamicOrUnknown
            } else {
                SourceSchema::Exact(Arc::clone(&schema))
            },
            native_watermarks: NativeWatermarkCapability::NeverEmits,
        };
        Ok(Self {
            config,
            url: Some(url),
            conn: None,
            capabilities,
            schema,
            ordering: Vec::new(),
            values: Vec::new(),
            sequence: 0,
            offset: 0,
            exhausted: false,
            idle: false,
        })
    }

    async fn load_schema(&self, conn: &mut Conn) -> Result<SchemaRef> {
        let rows: Vec<(String, String, String, String)> = database("schema", self.config.connection.timeout, conn.exec(
            "SELECT COLUMN_NAME, DATA_TYPE, COLUMN_TYPE, IS_NULLABLE FROM information_schema.COLUMNS WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = ? ORDER BY ORDINAL_POSITION", (&self.config.connection.table,)
        )).await?;
        let fields = rows
            .iter()
            .filter(|(name, _, _, _)| {
                self.config.columns.is_empty()
                    || self.config.columns.iter().any(|field| &field.name == name)
            })
            .map(|(name, kind, full, nullable)| {
                identifier(name)?;
                Ok(Field::new(
                    name,
                    types::data_type(kind, full)?,
                    nullable == "YES",
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        if fields.is_empty() {
            return Err(fail("schema", "table projection is empty"));
        }
        let fields = if self.config.columns.is_empty() {
            fields
        } else {
            self.config
                .columns
                .iter()
                .map(|spec| {
                    fields
                        .iter()
                        .find(|field| field.name() == &spec.name)
                        .cloned()
                        .ok_or_else(|| fail("schema", "projection column is missing"))
                })
                .collect::<Result<Vec<_>>>()?
        };
        let schema = Arc::new(Schema::new(fields));
        if !self.config.columns.is_empty()
            && schema != crate::arrow_schema::schema_from_spec(&self.config.columns)?
        {
            return Err(fail(
                "schema",
                "table schema differs from the declared Arrow projection",
            ));
        }
        Ok(schema)
    }

    async fn load_ordering(&mut self, conn: &mut Conn) -> Result<()> {
        let indexes: Vec<(String, Option<String>, Option<u64>)> = database("schema", self.config.connection.timeout, conn.exec(
            "SELECT INDEX_NAME, COLUMN_NAME, SUB_PART FROM information_schema.STATISTICS WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = ? AND NON_UNIQUE = 0 ORDER BY INDEX_NAME, SEQ_IN_INDEX", (&self.config.connection.table,)
        )).await?;
        let mut unique: BTreeMap<String, Vec<Option<String>>> = BTreeMap::new();
        for (index, name, prefix) in indexes {
            unique
                .entry(index)
                .or_default()
                .push(if prefix.is_none() { name } else { None });
        }
        if self.config.incremental {
            self.ordering.clone_from(&self.config.cursors);
            for name in &self.ordering {
                let field = self
                    .schema
                    .field_with_name(name)
                    .map_err(|_| fail("open", "cursor column is missing"))?;
                if field.is_nullable() || !field.data_type().is_integer() {
                    return Err(fail("open", "cursor columns must be non-null integers"));
                }
            }
            if !unique.values().any(|names| {
                !names.is_empty()
                    && names.iter().all(|name| {
                        name.as_ref()
                            .is_some_and(|name| self.ordering.contains(name))
                    })
            }) {
                return Err(fail(
                    "open",
                    "cursor must contain a complete unique index without prefix or expression columns",
                ));
            }
        } else {
            self.ordering = unique
                .remove("PRIMARY")
                .ok_or_else(|| fail("open", "snapshot pagination requires a primary key"))?
                .into_iter()
                .map(|name| name.ok_or_else(|| fail("open", "unsupported primary key")))
                .collect::<Result<_>>()?;
        }
        for name in &self.ordering {
            identifier(name)?;
        }
        Ok(())
    }

    fn restore(&mut self, cursor: &Cursor) -> Result<()> {
        if !self.config.incremental {
            return Err(fail("recover", "snapshot transactions cannot be resumed"));
        }
        let payload = cursor.payload();
        let sequence = payload
            .get("sequence")
            .and_then(serde_json::Value::as_u64)
            .filter(|n| *n > 0)
            .ok_or_else(|| fail("recover", "invalid cursor sequence"))?;
        if cursor.order() != sequence.to_be_bytes()
            || payload.get("table") != Some(&json!(self.config.connection.table))
            || payload.get("columns") != Some(&json!(self.ordering))
            || payload.get("schema") != Some(&json!(self.schema_hash()))
        {
            return Err(fail(
                "recover",
                "cursor identity, schema, or order mismatch",
            ));
        }
        let values = payload
            .get("values")
            .and_then(serde_json::Value::as_array)
            .filter(|values| values.len() == self.ordering.len())
            .ok_or_else(|| fail("recover", "invalid cursor width"))?;
        let values = values
            .iter()
            .zip(&self.ordering)
            .map(|(value, name)| {
                let kind = self
                    .schema
                    .field_with_name(name)
                    .map_err(|_| fail("recover", "cursor field missing"))?
                    .data_type();
                let candidate = if kind.is_unsigned_integer() {
                    value.as_u64().map(Value::UInt)
                } else {
                    value.as_i64().map(Value::Int)
                };
                candidate
                    .filter(|value| cursor_in_range(value, kind))
                    .ok_or_else(|| fail("recover", "cursor integer out of range"))
            })
            .collect::<Result<Vec<_>>>()?;
        self.values = values;
        self.sequence = sequence;
        Ok(())
    }

    fn schema_hash(&self) -> String {
        crate::evidence::sha256_hex(format!("{:?}", self.schema).as_bytes())
    }

    fn cursor(&self) -> Result<Cursor> {
        let values = self
            .values
            .iter()
            .map(|value| match value {
                Value::Int(n) => json!(n),
                Value::UInt(n) => json!(n),
                _ => unreachable!("integer cursor"),
            })
            .collect::<Vec<_>>();
        Cursor::unbound(
            self.sequence.to_be_bytes().to_vec(),
            JsonMap::from([
                ("sequence".into(), json!(self.sequence)),
                ("table".into(), json!(self.config.connection.table)),
                ("columns".into(), json!(self.ordering)),
                ("schema".into(), json!(self.schema_hash())),
                ("values".into(), json!(values)),
            ]),
        )
    }

    fn query(&self) -> Result<(String, Vec<Value>)> {
        let names = self
            .schema
            .fields()
            .iter()
            .map(|field| identifier(field.name()))
            .collect::<Result<Vec<_>>>()?
            .join(", ");
        let ordering = self
            .ordering
            .iter()
            .map(|name| identifier(name))
            .collect::<Result<Vec<_>>>()?
            .join(", ");
        let mut sql = format!(
            "SELECT {names} FROM {}",
            identifier(&self.config.connection.table)?
        );
        let mut params = self.values.clone();
        if !params.is_empty() {
            write!(
                sql,
                " WHERE ({ordering}) > ({})",
                vec!["?"; params.len()].join(", ")
            )
            .expect("write to string");
        }
        write!(sql, " ORDER BY {ordering} LIMIT ?").expect("write to string");
        params.push(Value::UInt(self.config.rows));
        if !self.config.incremental {
            sql.push_str(" OFFSET ?");
            params.push(Value::UInt(self.offset));
        }
        Ok((sql, params))
    }

    async fn fetch(&self, conn: &mut Conn) -> Result<Vec<Row>> {
        let (sql, params) = self.query()?;
        let mut result = database(
            "read",
            self.config.connection.timeout,
            conn.exec_iter(sql, params),
        )
        .await?;
        let mut rows = Vec::new();
        let mut bytes = 0_u64;
        while let Some(row) =
            database("read", self.config.connection.timeout, result.next()).await?
        {
            let size = (0..row.len())
                .map(|index| match row.as_ref(index) {
                    Some(Value::Bytes(bytes)) => bytes.len() as u64 + 32,
                    _ => 32,
                })
                .sum::<u64>();
            bytes = bytes
                .checked_add(size)
                .ok_or_else(|| fail("read", "batch byte count overflow"))?;
            if bytes > self.config.bytes || rows.len() as u64 >= self.config.rows {
                return Err(fail("read", "batch exceeds configured bounds"));
            }
            rows.push(row);
        }
        database("read", self.config.connection.timeout, result.drop_result()).await?;
        Ok(rows)
    }
}

fn cursor_in_range(value: &Value, kind: &DataType) -> bool {
    match (value, kind) {
        (Value::Int(n), DataType::Int8) => i8::try_from(*n).is_ok(),
        (Value::Int(n), DataType::Int16) => i16::try_from(*n).is_ok(),
        (Value::Int(n), DataType::Int32) => i32::try_from(*n).is_ok(),
        (Value::UInt(n), DataType::UInt8) => u8::try_from(*n).is_ok(),
        (Value::UInt(n), DataType::UInt16) => u16::try_from(*n).is_ok(),
        (Value::UInt(n), DataType::UInt32) => u32::try_from(*n).is_ok(),
        (Value::Int(_), DataType::Int64) | (Value::UInt(_), DataType::UInt64) => true,
        _ => false,
    }
}

#[async_trait]
impl StreamSource for MySqlSource {
    fn capabilities(&self) -> SourceCapabilities {
        self.capabilities.clone()
    }
    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        if cursor.is_some() && !self.config.incremental {
            return Err(fail("open", "snapshot transactions cannot be resumed"));
        }
        let url = self
            .url
            .take()
            .ok_or_else(|| fail("open", "source has already been opened"))?;
        let mut conn = connect(&url, &self.config.connection, self.config.bytes).await?;
        require_innodb(
            &mut conn,
            &self.config.connection.table,
            self.config.connection.timeout,
        )
        .await?;
        if !self.config.incremental {
            database(
                "open",
                self.config.connection.timeout,
                conn.query_drop("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ"),
            )
            .await?;
            database(
                "open",
                self.config.connection.timeout,
                conn.query_drop("START TRANSACTION WITH CONSISTENT SNAPSHOT, READ ONLY"),
            )
            .await?;
        }
        self.schema = self.load_schema(&mut conn).await?;
        self.load_ordering(&mut conn).await?;
        if let Some(cursor) = cursor {
            self.restore(&cursor)?;
        }
        self.conn = Some(conn);
        Ok(())
    }
    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        if self.exhausted {
            return Ok(None);
        }
        if self.idle {
            tokio::time::sleep(self.config.poll).await;
        }
        // Own the connection across this operation: cancellation/error drops it
        // instead of leaving a partially consumed result available for reuse.
        let mut conn = self
            .conn
            .take()
            .ok_or_else(|| fail("read", "source is not open"))?;
        if self.load_schema(&mut conn).await? != self.schema {
            return Err(fail("read", "source schema changed"));
        }
        let rows = self.fetch(&mut conn).await?;
        if rows.is_empty() {
            self.idle = self.config.incremental;
            self.exhausted = !self.config.incremental;
            if self.exhausted {
                database(
                    "read",
                    self.config.connection.timeout,
                    conn.query_drop("ROLLBACK"),
                )
                .await?;
            }
            self.conn = Some(conn);
            return Ok(self.idle.then_some(SourceEvent::Idle));
        }
        let record = types::record(&rows, Arc::clone(&self.schema))?;
        if record.get_array_memory_size() as u64 > self.config.bytes {
            return Err(fail("read", "decoded batch exceeds max_batch_bytes"));
        }
        self.sequence = self
            .sequence
            .checked_add(1)
            .ok_or_else(|| fail("read", "cursor sequence exhausted"))?;
        self.offset = self
            .offset
            .checked_add(rows.len() as u64)
            .ok_or_else(|| fail("read", "snapshot offset exhausted"))?;
        if self.config.incremental {
            let last = rows.last().expect("nonempty rows");
            self.values = self
                .ordering
                .iter()
                .map(|name| {
                    let index = self
                        .schema
                        .index_of(name)
                        .map_err(|_| fail("read", "cursor column missing"))?;
                    last.as_ref(index)
                        .cloned()
                        .ok_or_else(|| fail("read", "cursor value missing"))
                })
                .collect::<Result<_>>()?;
        }
        let batch = Batch::table(
            vec![record],
            BatchMetadata::new("mysql", self.sequence, BTreeMap::new())?,
        )?;
        let cursor = self.cursor()?;
        self.idle = false;
        self.conn = Some(conn);
        Ok(Some(SourceEvent::Data { batch, cursor }))
    }
    async fn close(&mut self) -> Result<()> {
        self.url = None;
        if let Some(conn) = self.conn.take() {
            database("close", self.config.connection.timeout, conn.disconnect()).await?;
        }
        Ok(())
    }
}
