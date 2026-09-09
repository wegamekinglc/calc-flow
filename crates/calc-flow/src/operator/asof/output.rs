use super::{StreamAsofJoinSpec, codec};
use crate::{Batch, BatchMetadata, DataFusionConfig, Result, StateSegment};
use datafusion::{
    arrow::{
        array::{ArrayRef, UInt64Array},
        compute::concat_batches,
        datatypes::{DataType, Field, Schema, SchemaRef},
        record_batch::RecordBatch,
    },
    execution::{
        disk_manager::{DiskManagerBuilder, DiskManagerMode},
        memory_pool::{GreedyMemoryPool, MemoryPool},
        runtime_env::RuntimeEnvBuilder,
    },
    prelude::{SessionConfig, SessionContext},
};
use std::sync::Arc;

pub(super) struct OutputRuntime {
    context: Option<SessionContext>,
    pub pool: Arc<dyn MemoryPool>,
    config: DataFusionConfig,
}
impl OutputRuntime {
    pub fn new(limit: usize) -> Self {
        Self {
            context: None,
            pool: Arc::new(GreedyMemoryPool::new(limit)),
            config: DataFusionConfig::default(),
        }
    }
    pub const fn initialized(&self) -> bool {
        self.context.is_some()
    }
    pub fn reset(&mut self) {
        self.context = None;
    }
    pub fn configure(&mut self, config: DataFusionConfig) {
        self.config = config;
    }
    fn context(&mut self) -> Result<&SessionContext> {
        if self.context.is_none() {
            self.config.validate()?;
            let runtime = RuntimeEnvBuilder::new()
                .with_memory_pool(self.pool.clone())
                .with_disk_manager_builder(
                    DiskManagerBuilder::default().with_mode(DiskManagerMode::Disabled),
                )
                .build_arc()
                .map_err(|error| super::fusion_error(&error))?;
            let mut config = SessionConfig::new()
                .with_target_partitions(self.config.target_partitions)
                .with_batch_size(self.config.batch_size);
            config.options_mut().execution.sort_spill_reservation_bytes = 0;
            self.context = Some(SessionContext::new_with_config_rt(config, runtime));
        }
        Ok(self.context.as_ref().expect("initialized above"))
    }
    pub async fn materialize(
        &mut self,
        rows: &[(&StateSegment, Option<&StateSegment>)],
        spec: &StreamAsofJoinSpec,
        schemas: &[SchemaRef; 3],
        schema_digests: &[[u8; 32]; 2],
        node: &str,
    ) -> Result<Batch> {
        let left = candidate_batch(rows, false, &schemas[0], &schema_digests[0])?;
        let right = candidate_batch(rows, true, &schemas[1], &schema_digests[1])?;
        let query = query(spec, schemas);
        let context = self.context()?;
        let batches = query_candidates(context, left, right, &query, node)
            .await?
            .into_iter()
            .map(|batch| {
                RecordBatch::try_new(schemas[2].clone(), batch.columns().to_vec())
                    .map_err(|error| super::arrow_error(&error))
            })
            .collect::<Result<Vec<_>>>()?;
        Batch::table(batches, BatchMetadata::default())
    }
}

/// The session may outlive a cancelled materialization future.
struct CandidateTables<'a> {
    context: &'a SessionContext,
    active: bool,
}

impl<'a> CandidateTables<'a> {
    fn register(
        context: &'a SessionContext,
        left: RecordBatch,
        right: RecordBatch,
    ) -> Result<Self> {
        let tables = Self {
            context,
            active: true,
        };
        context
            .register_batch("asof_left", left)
            .map_err(|error| super::fusion_error(&error))?;
        context
            .register_batch("asof_right", right)
            .map_err(|error| super::fusion_error(&error))?;
        Ok(tables)
    }

    fn clear(&mut self) -> Result<()> {
        self.context
            .deregister_table("asof_left")
            .map_err(|error| super::fusion_error(&error))?;
        self.context
            .deregister_table("asof_right")
            .map_err(|error| super::fusion_error(&error))?;
        self.active = false;
        Ok(())
    }
}

impl Drop for CandidateTables<'_> {
    fn drop(&mut self) {
        if self.active {
            // This private session has only its default in-memory catalog.
            // Attempt both removals, including partial registration and unwinding.
            let _ = self.context.deregister_table("asof_left");
            let _ = self.context.deregister_table("asof_right");
        }
    }
}

async fn query_candidates(
    context: &SessionContext,
    left: RecordBatch,
    right: RecordBatch,
    query: &str,
    node: &str,
) -> Result<Vec<RecordBatch>> {
    let mut tables = CandidateTables::register(context, left, right)?;
    // Allow cancellation before bounded query planning and execution.
    tokio::task::yield_now().await;
    let result = async { context.sql(query).await?.collect().await }.await;
    tables.clear()?;
    result.map_err(|error| materialization_error(&error, node))
}

fn materialization_error(
    error: &datafusion::error::DataFusionError,
    node: &str,
) -> crate::CalcFlowError {
    if matches!(
        error.find_root(),
        datafusion::error::DataFusionError::ResourcesExhausted(_)
    ) {
        super::reason(
            node,
            crate::StreamingFailureReason::AsofWorkspaceLimitExceeded,
            "ASOF DataFusion workspace reservation exceeded",
        )
    } else {
        super::fusion_error(error)
    }
}

fn candidate_batch(
    rows: &[(&StateSegment, Option<&StateSegment>)],
    right: bool,
    schema: &SchemaRef,
    schema_digest: &[u8; 32],
) -> Result<RecordBatch> {
    let selected = rows
        .iter()
        .enumerate()
        .filter_map(|(ordinal, (left, candidate))| {
            if right {
                candidate.map(|row| (ordinal, row))
            } else {
                Some((ordinal, *left))
            }
        })
        .collect::<Vec<_>>();
    let batches = selected
        .iter()
        .map(|(_, row)| codec::decode_batch(row.bytes(), schema_digest))
        .collect::<Result<Vec<_>>>()?;
    let combined = concat_batches(schema, &batches).map_err(|error| super::arrow_error(&error))?;
    let mut columns = vec![Arc::new(UInt64Array::from(
        selected
            .iter()
            .map(|(ordinal, _)| u64::try_from(*ordinal).expect("bounded chunk"))
            .collect::<Vec<_>>(),
    )) as ArrayRef];
    columns.extend(combined.columns().iter().cloned());
    let fields = std::iter::once(Field::new("__asof_ordinal", DataType::UInt64, false))
        .chain(
            schema
                .fields()
                .iter()
                .enumerate()
                .map(|(index, field)| field.as_ref().clone().with_name(format!("c{index}"))),
        )
        .collect::<Vec<_>>();
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
        .map_err(|error| super::arrow_error(&error))
}

fn query(spec: &StreamAsofJoinSpec, schemas: &[SchemaRef; 3]) -> String {
    let projection = [("l", &schemas[0]), ("r", &schemas[1])]
        .into_iter()
        .flat_map(|(alias, schema)| {
            (0..schema.fields().len()).map(move |index| format!("{alias}.c{index}"))
        })
        .collect::<Vec<_>>()
        .join(", ");
    let mut condition = vec!["l.__asof_ordinal = r.__asof_ordinal".to_owned()];
    for (left, right) in spec.left().keys().iter().zip(spec.right().keys()) {
        condition.push(format!(
            "l.c{} = r.c{}",
            schemas[0].index_of(left).expect("validated"),
            schemas[1].index_of(right).expect("validated")
        ));
    }
    format!(
        "SELECT {projection} FROM asof_left l LEFT JOIN asof_right r ON {} ORDER BY l.__asof_ordinal",
        condition.join(" AND ")
    )
}

#[cfg(test)]
mod tests;
