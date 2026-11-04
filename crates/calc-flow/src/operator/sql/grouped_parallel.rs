use super::{
    Arc, ArrayRef, BooleanArray, CHUNK_ROWS, ChunkInput, Column, DataFusionRuntime, DataType,
    Field, HashMap, IncrementalSql, InputCandidates, KeyIndex, Literal, MemoryReservation,
    PartialGroups, PhysicalExpr, RandomState, RecordBatch, Result, RowConverter, Schema, SortField,
    StreamOperatorContext, checked_bytes, df_error, grouped_sum, native_key_width, predicate,
};

#[cfg(test)]
use super::grouped_parallel_tests;
use crate::runtime::streaming::gather_work::{
    AdmissionFailure, GatherOperatorId, GatherStop, ParallelCpuWork,
};
use datafusion::{
    arrow::array::Array,
    logical_expr::Operator,
    physical_expr::expressions::{BinaryExpr, IsNotNullExpr, IsNullExpr, NotExpr},
};
use std::{
    mem::{size_of, size_of_val},
    ops::Range,
    sync::Mutex,
};

const MIN_ROWS: usize = 4 * CHUNK_ROWS;
const MAX_UNITS: usize = 8;

pub(super) fn unbounded(runtime: &DataFusionRuntime) -> bool {
    runtime
        .incremental_memory_pool()
        .is::<datafusion::execution::memory_pool::UnboundedMemoryPool>()
}

struct Chunk {
    record: usize,
    start: usize,
    rows: usize,
}

struct LocalKey {
    encoded: Arc<[u8]>,
    chunk: usize,
    row: usize,
}

struct Lane {
    chunks: Range<usize>,
    partial: PartialGroups,
}

struct Dimensions {
    rows: usize,
    chunks: usize,
    units: usize,
    nodes: usize,
}

struct Work {
    records: Vec<RecordBatch>,
    chunks: Vec<Chunk>,
    keys: Vec<usize>,
    arguments: Vec<usize>,
    filters: Vec<Option<Arc<dyn PhysicalExpr>>>,
    predicate: Option<predicate::InputPredicate>,
    lanes: Vec<Mutex<Option<Lane>>>,
    name: String,
    scratch: Arc<MemoryReservation>,
    _input_owner: Option<Arc<MemoryReservation>>,
}

struct Fragment {
    keys: Vec<LocalKey>,
    states: Vec<Vec<ArrayRef>>,
    _partial: MemoryReservation,
    _scratch: Arc<MemoryReservation>,
}

struct LocalGroups<'a> {
    keys: Vec<LocalKey>,
    ranks: KeyIndex,
    converter: RowConverter,
    name: &'a str,
}

struct LocalInput {
    selection: Option<BooleanArray>,
    values: ChunkInput,
}

fn units(rows: usize, chunks: usize, cpus: usize) -> usize {
    if rows < MIN_ROWS || chunks < 4 {
        return 1;
    }
    cpus.min(MAX_UNITS).min(chunks / 2).max(1)
}

fn column(expression: &dyn PhysicalExpr) -> Option<usize> {
    expression.downcast_ref::<Column>().map(Column::index)
}

fn inputs_eligible(plan: &IncrementalSql) -> bool {
    !plan.keys.is_empty()
        && plan.input_checks.is_empty()
        && !native_shape(plan)
        && plan.keys.iter().all(|key| column(key.expression.as_ref()).is_some())
        && plan.aggregates.iter().all(|expression| {
            grouped_sum::integer_sum(expression)
                && matches!(expression.expressions().as_slice(), [argument] if column(argument.as_ref()).is_some())
        })
}

fn native_shape(plan: &IncrementalSql) -> bool {
    matches!(plan.keys.as_slice(), [key] if native_key_width(key.field.data_type()).is_some())
        && !plan.requires_explicit_grouping()
}

fn filter_nodes(plan: &IncrementalSql) -> Option<usize> {
    let initial = plan
        .predicate
        .as_ref()
        .map_or(Some(0), predicate::InputPredicate::parallel_nodes)?;
    plan.aggregate_filters
        .iter()
        .try_fold(initial, |nodes, filter| {
            nodes.checked_add(
                filter
                    .as_ref()
                    .map_or(Some(0), |filter| predicate_nodes(filter.as_ref(), 0))?,
            )
        })
}

pub(super) fn predicate_nodes(expression: &dyn PhysicalExpr, depth: usize) -> Option<usize> {
    if depth > 32 {
        return None;
    }
    if expression.downcast_ref::<Column>().is_some()
        || expression.downcast_ref::<Literal>().is_some()
    {
        return Some(1);
    }
    if let Some(binary) = expression.downcast_ref::<BinaryExpr>() {
        return binary_nodes(binary, depth);
    }
    unary(expression).and_then(|argument| predicate_nodes(argument, depth + 1)?.checked_add(1))
}

fn binary_nodes(binary: &BinaryExpr, depth: usize) -> Option<usize> {
    if !comparison(*binary.op()) {
        return None;
    }
    predicate_nodes(binary.left().as_ref(), depth + 1)?
        .checked_add(predicate_nodes(binary.right().as_ref(), depth + 1)?)?
        .checked_add(1)
}

fn unary(expression: &dyn PhysicalExpr) -> Option<&dyn PhysicalExpr> {
    if let Some(expression) = expression.downcast_ref::<NotExpr>() {
        return Some(expression.arg().as_ref());
    }
    if let Some(expression) = expression.downcast_ref::<IsNullExpr>() {
        return Some(expression.arg().as_ref());
    }
    expression
        .downcast_ref::<IsNotNullExpr>()
        .map(|expression| expression.arg().as_ref())
}

fn comparison(operator: Operator) -> bool {
    matches!(
        operator,
        Operator::Eq
            | Operator::NotEq
            | Operator::Lt
            | Operator::LtEq
            | Operator::Gt
            | Operator::GtEq
            | Operator::IsDistinctFrom
            | Operator::IsNotDistinctFrom
    )
}

fn chunk_count(records: &[RecordBatch]) -> Option<usize> {
    records.iter().try_fold(0usize, |count, record| {
        count.checked_add(record.num_rows().div_ceil(CHUNK_ROWS))
    })
}

fn row_count(records: &[RecordBatch]) -> Option<usize> {
    records
        .iter()
        .try_fold(0usize, |rows, record| rows.checked_add(record.num_rows()))
}

fn schema_charge(schema: &Schema, name: &str) -> Result<usize> {
    let mut bytes = checked_bytes(
        size_of::<Schema>(),
        [(schema.fields().len(), size_of::<Arc<Field>>())],
        name,
    )?;
    for field in schema.fields() {
        bytes = checked_bytes(bytes, [(field_charge(field, name)?, 1)], name)?;
    }
    checked_bytes(
        bytes,
        [(metadata_charge(schema.metadata(), name)?, 1)],
        name,
    )
}

fn field_charge(field: &Field, name: &str) -> Result<usize> {
    if field.data_type().primitive_width().is_none()
        && !matches!(
            field.data_type(),
            DataType::Null | DataType::Boolean | DataType::Utf8 | DataType::LargeUtf8
        )
    {
        return Err(df_error(name, "parallel SUM input schema is not flat"));
    }
    checked_bytes(
        size_of::<Field>(),
        [
            (field.name().capacity(), 1),
            (field.data_type().size(), 1),
            (metadata_charge(field.metadata(), name)?, 1),
        ],
        name,
    )
}

fn metadata_charge(
    metadata: &std::collections::HashMap<String, String>,
    name: &str,
) -> Result<usize> {
    let mut bytes = checked_bytes(
        0,
        [(metadata.capacity(), 2 * size_of::<(String, String)>())],
        name,
    )?;
    for (key, value) in metadata {
        bytes = checked_bytes(bytes, [(key.capacity(), 1), (value.capacity(), 1)], name)?;
    }
    Ok(bytes)
}

fn partition(chunks: usize, units: usize, ordinal: usize) -> Option<Range<usize>> {
    Some(boundary(chunks, units, ordinal)?..boundary(chunks, units, ordinal.checked_add(1)?)?)
}

fn boundary(chunks: usize, units: usize, ordinal: usize) -> Option<usize> {
    let base = chunks / units;
    let extra = chunks % units;
    base.checked_mul(ordinal)?
        .checked_add(extra.checked_mul(ordinal)? / units)
}

fn scratch_bytes(
    plan: &IncrementalSql,
    records: &[RecordBatch],
    dimensions: &Dimensions,
    has_owner: bool,
    name: &str,
) -> Result<usize> {
    let mut bytes = fixed_charge(plan, dimensions, name)?;
    bytes = checked_bytes(
        bytes,
        [(bound_plan_charge(plan, dimensions.units, name)?, 1)],
        name,
    )?;
    for record in records {
        bytes = checked_bytes(
            bytes,
            [(record_charge(plan, record, has_owner, name)?, 1)],
            name,
        )?;
    }
    Ok(bytes)
}

fn bound_plan_charge(plan: &IncrementalSql, units: usize, name: &str) -> Result<usize> {
    checked_bytes(
        0,
        [
            (schema_charge(&plan.schema, name)?, 2),
            (filter_charge(plan, units, name)?, 1),
        ],
        name,
    )
}

fn fixed_charge(plan: &IncrementalSql, dimensions: &Dimensions, name: &str) -> Result<usize> {
    let width = checked_bytes(256, [(plan.keys.len(), 64)], name)?;
    let mut bytes = checked_bytes(
        16384,
        [
            (dimensions.rows, width),
            (dimensions.chunks, 128),
            (dimensions.units, 4096),
            (dimensions.nodes, 512),
            (plan.plan_bytes, 1),
        ],
        name,
    )?;
    let width = checked_bytes(
        64,
        [(dimensions.nodes, 128), (plan.aggregates.len(), 64)],
        name,
    )?;
    let live_rows = checked_bytes(0, [(dimensions.units, CHUNK_ROWS)], name)?;
    bytes = checked_bytes(bytes, [(live_rows, width), (name.len(), 2)], name)?;
    Ok(bytes)
}

fn record_charge(
    plan: &IncrementalSql,
    record: &RecordBatch,
    has_owner: bool,
    name: &str,
) -> Result<usize> {
    let backing = if has_owner {
        0
    } else {
        record.get_array_memory_size()
    };
    let mut bytes = checked_bytes(
        256,
        [
            (record.num_columns(), 64),
            (backing, 1),
            (schema_charge(&record.schema(), name)?, 1),
        ],
        name,
    )?;
    for key in &plan.keys {
        let index = column(key.expression.as_ref()).expect("admitted column key");
        let size = record
            .column(index)
            .to_data()
            .get_slice_memory_size()
            .map_err(|error| df_error(name, error))?;
        bytes = checked_bytes(bytes, [(size, 8)], name)?;
    }
    Ok(bytes)
}

fn filter_charge(plan: &IncrementalSql, units: usize, name: &str) -> Result<usize> {
    let copies = checked_bytes(1, [(units, 4)], name)?;
    checked_bytes(0, [(retained_filter_charge(plan, name)?, copies)], name)
}

fn retained_filter_charge(plan: &IncrementalSql, name: &str) -> Result<usize> {
    let initial = plan.predicate.as_ref().map_or(Ok(0), |predicate| {
        predicate.parallel_charge(&plan.schema, name)
    })?;
    plan.aggregate_filters
        .iter()
        .try_fold(initial, |bytes, filter| {
            let payload = filter.as_ref().map_or(Ok(0), |filter| {
                predicate_charge(filter.as_ref(), &plan.schema, name)
            })?;
            checked_bytes(bytes, [(payload, 1)], name)
        })
}

pub(super) fn predicate_charge(
    expression: &dyn PhysicalExpr,
    schema: &Schema,
    name: &str,
) -> Result<usize> {
    let own = checked_bytes(
        128,
        [
            (size_of_val(expression), 1),
            (own_payload(expression, schema, name)?, 1),
        ],
        name,
    )?;
    if let Some(binary) = expression.downcast_ref::<BinaryExpr>() {
        return checked_bytes(
            own,
            [
                (predicate_charge(binary.left().as_ref(), schema, name)?, 1),
                (predicate_charge(binary.right().as_ref(), schema, name)?, 1),
            ],
            name,
        );
    }
    match unary(expression) {
        Some(argument) => {
            checked_bytes(own, [(predicate_charge(argument, schema, name)?, 1)], name)
        }
        None => Ok(own),
    }
}

fn own_payload(expression: &dyn PhysicalExpr, schema: &Schema, name: &str) -> Result<usize> {
    if let Some(column) = expression.downcast_ref::<Column>() {
        return checked_bytes(0, [(column.name().len(), 2)], name);
    }
    if let Some(literal) = expression.downcast_ref::<Literal>() {
        let field = literal
            .return_field(schema)
            .map_err(|error| df_error(name, error))?;
        return checked_bytes(
            0,
            [
                (field_charge(&field, name)?, 1),
                (literal.value().size(), 1),
            ],
            name,
        );
    }
    Ok(0)
}

fn chunks(records: &[RecordBatch], capacity: usize) -> Vec<Chunk> {
    let mut chunks = Vec::with_capacity(capacity);
    for (record, batch) in records.iter().enumerate() {
        for start in (0..batch.num_rows()).step_by(CHUNK_ROWS) {
            chunks.push(Chunk {
                record,
                start,
                rows: (batch.num_rows() - start).min(CHUNK_ROWS),
            });
        }
    }
    chunks
}

fn prepare(
    plan: &IncrementalSql,
    records: &[RecordBatch],
    input_owner: Option<Arc<MemoryReservation>>,
    cpus: usize,
    name: &str,
) -> Option<Arc<Work>> {
    let dimensions = dimensions(plan, records, cpus)?;
    let bytes = scratch_bytes(plan, records, &dimensions, input_owner.is_some(), name).ok()?;
    let scratch = Arc::new(plan.scratch.new_empty());
    scratch.try_grow(bytes).ok()?;
    let chunks = chunks(records, dimensions.chunks);
    let lanes = lanes(plan, &chunks, dimensions.units, name).ok()?;
    Some(Arc::new(work(
        plan,
        records,
        chunks,
        lanes,
        scratch,
        input_owner,
        name,
    )))
}

fn dimensions(plan: &IncrementalSql, records: &[RecordBatch], cpus: usize) -> Option<Dimensions> {
    if !plan.parallel_scratch || cpus < 2 {
        return None;
    }
    let chunks = chunk_count(records)?;
    let rows = row_count(records)?;
    let units = units(rows, chunks, cpus);
    if units < 2 || !inputs_eligible(plan) {
        return None;
    }
    Some(Dimensions {
        rows,
        chunks,
        units,
        nodes: filter_nodes(plan)?,
    })
}

fn lanes(
    plan: &IncrementalSql,
    chunks: &[Chunk],
    count: usize,
    name: &str,
) -> Result<Vec<Mutex<Option<Lane>>>> {
    let mut lanes = Vec::with_capacity(count);
    for ordinal in 0..count {
        let range = partition(chunks.len(), count, ordinal)
            .ok_or_else(|| df_error(name, "SUM lane range overflow"))?;
        let rows = lane_rows(chunks, range.clone(), name)?;
        let mut partial = PartialGroups::new(&plan.aggregates, plan.scratch.new_empty(), name)?;
        partial.reserve(rows, name)?;
        lanes.push(Mutex::new(Some(Lane {
            chunks: range,
            partial,
        })));
    }
    Ok(lanes)
}

fn work(
    plan: &IncrementalSql,
    records: &[RecordBatch],
    chunks: Vec<Chunk>,
    lanes: Vec<Mutex<Option<Lane>>>,
    scratch: Arc<MemoryReservation>,
    input_owner: Option<Arc<MemoryReservation>>,
    name: &str,
) -> Work {
    Work {
        records: records.to_vec(),
        chunks,
        keys: plan
            .keys
            .iter()
            .map(|key| column(key.expression.as_ref()).expect("admitted key"))
            .collect(),
        arguments: plan
            .aggregates
            .iter()
            .map(|expression| {
                column(expression.expressions()[0].as_ref()).expect("admitted SUM argument")
            })
            .collect(),
        filters: plan.aggregate_filters.clone(),
        predicate: plan.predicate.clone(),
        lanes,
        name: name.into(),
        scratch,
        _input_owner: input_owner,
    }
}

fn lane_rows(chunks: &[Chunk], range: Range<usize>, name: &str) -> Result<usize> {
    chunks[range]
        .iter()
        .try_fold(0usize, |rows, chunk| rows.checked_add(chunk.rows))
        .ok_or_else(|| df_error(name, "SUM lane row count overflow"))
}

impl Work {
    fn key_arrays(&self, chunk: usize) -> Vec<ArrayRef> {
        let chunk = &self.chunks[chunk];
        self.keys
            .iter()
            .map(|&index| {
                self.records[chunk.record]
                    .column(index)
                    .slice(chunk.start, chunk.rows)
            })
            .collect()
    }
}

impl ParallelCpuWork for Work {
    type Output = Fragment;

    fn unit_count(&self) -> usize {
        self.lanes.len()
    }

    fn run(&self, ordinal: usize, stop: &GatherStop) -> Result<Fragment> {
        #[cfg(test)]
        grouped_parallel_tests::before_unit(&self.name, ordinal, stop)?;
        let lane = self.lanes[ordinal]
            .lock()
            .map_err(|error| df_error(&self.name, error))?
            .take()
            .expect("owned SUM lane");
        #[cfg(test)]
        let _completion = grouped_parallel_tests::completion(&self.name, ordinal);
        self.aggregate_range(lane, stop)
    }
}

impl Work {
    fn aggregate_range(&self, mut lane: Lane, stop: &GatherStop) -> Result<Fragment> {
        let mut groups = self.local_groups(lane.chunks.clone())?;
        for ordinal in lane.chunks.clone() {
            stop.check()?;
            self.aggregate_chunk(ordinal, &mut lane.partial, &mut groups, stop)?;
        }
        stop.check()?;
        let states = lane.partial.emit_states(&self.name)?;
        let PartialGroups { reservation, .. } = lane.partial;
        Ok(Fragment {
            keys: groups.keys,
            states,
            _partial: reservation,
            _scratch: self.scratch.clone(),
        })
    }

    fn aggregate_chunk(
        &self,
        ordinal: usize,
        partial: &mut PartialGroups,
        groups: &mut LocalGroups<'_>,
        stop: &GatherStop,
    ) -> Result<()> {
        let input = self.chunk_input(ordinal)?;
        let filter_refs = input
            .values
            .filters
            .iter()
            .map(Option::as_ref)
            .collect::<Vec<_>>();
        let combined = predicate::combine(
            input.selection.as_ref(),
            &filter_refs,
            self.arguments.len(),
            &self.name,
        )?;
        let indices = groups.indices(ordinal, &input, partial, stop)?;
        let filters = if input.selection.is_some() {
            &combined
        } else {
            &input.values.filters
        };
        let filter_refs = filters.iter().map(Option::as_ref).collect::<Vec<_>>();
        partial.update(
            &input.values.arguments,
            &filter_refs,
            &indices,
            groups.keys.len(),
            &self.name,
        )
    }

    fn local_groups(&self, range: Range<usize>) -> Result<LocalGroups<'_>> {
        let rows = lane_rows(&self.chunks, range, &self.name)?;
        let fields = self
            .keys
            .iter()
            .map(|&index| SortField::new(self.records[0].column(index).data_type().clone()))
            .collect();
        let converter = RowConverter::new(fields).map_err(|error| df_error(&self.name, error))?;
        Ok(LocalGroups {
            keys: Vec::with_capacity(rows),
            ranks: HashMap::with_capacity_and_hasher(rows, RandomState::new()),
            converter,
            name: &self.name,
        })
    }

    fn chunk_input(&self, ordinal: usize) -> Result<LocalInput> {
        let descriptor = &self.chunks[ordinal];
        let chunk = self.records[descriptor.record].slice(descriptor.start, descriptor.rows);
        let selection = self
            .predicate
            .as_ref()
            .map(|predicate| predicate.evaluate(&chunk, &self.name))
            .transpose()?;
        let keys = self.key_arrays(ordinal);
        let arguments = self
            .arguments
            .iter()
            .map(|&index| vec![chunk.column(index).clone()])
            .collect::<Vec<_>>();
        let filters = self
            .filters
            .iter()
            .map(|filter| {
                filter
                    .as_ref()
                    .map(|filter| boolean(filter.as_ref(), &chunk, &self.name))
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(LocalInput {
            selection,
            values: ChunkInput {
                keys,
                arguments,
                filters,
            },
        })
    }
}

impl LocalGroups<'_> {
    fn indices(
        &mut self,
        ordinal: usize,
        input: &LocalInput,
        partial: &mut PartialGroups,
        stop: &GatherStop,
    ) -> Result<Vec<usize>> {
        let encoded = self
            .converter
            .convert_columns(&input.values.keys)
            .map_err(|error| df_error(self.name, error))?;
        let mut indices = Vec::with_capacity(encoded.num_rows());
        for row in 0..encoded.num_rows() {
            if row % 1024 == 0 {
                stop.check()?;
            }
            if input
                .selection
                .as_ref()
                .is_some_and(|selection| !predicate::selected(selection, row))
            {
                indices.push(0);
                continue;
            }
            indices.push(self.rank(encoded.row(row).as_ref(), ordinal, row, partial)?);
        }
        Ok(indices)
    }

    fn rank(
        &mut self,
        encoded: &[u8],
        chunk: usize,
        row: usize,
        partial: &mut PartialGroups,
    ) -> Result<usize> {
        if let Some(&rank) = self.ranks.get(encoded) {
            return Ok(rank);
        }
        let rank = self.keys.len();
        let encoded: Arc<[u8]> = Arc::from(encoded);
        self.ranks.insert(encoded.clone(), rank);
        self.keys.push(LocalKey {
            encoded,
            chunk,
            row,
        });
        partial.add_slot(rank, self.name)?;
        Ok(rank)
    }
}

fn boolean(expression: &dyn PhysicalExpr, chunk: &RecordBatch, name: &str) -> Result<BooleanArray> {
    let array = expression
        .evaluate(chunk)
        .and_then(|value| value.into_array(chunk.num_rows()))
        .map_err(|error| df_error(name, error))?;
    array
        .as_any()
        .downcast_ref::<BooleanArray>()
        .cloned()
        .ok_or_else(|| df_error(name, "SQL filter is not Boolean"))
}

pub(super) async fn update(
    plan: &IncrementalSql,
    records: &[RecordBatch],
    input_owner: Option<Arc<MemoryReservation>>,
    candidates: &mut InputCandidates,
    context: &StreamOperatorContext<'_>,
    name: &str,
) -> Result<bool> {
    let Some(work) = prepare(
        plan,
        records,
        input_owner,
        context.job().gather_owner().parallelism(),
        name,
    ) else {
        return Ok(false);
    };
    let output = submit(&work, plan.scratch.new_empty(), context, name).await?;
    drop(candidates.native.take());
    for fragment in &output.value {
        #[cfg(test)]
        grouped_parallel_tests::before_merge(name)?;
        merge_fragment(plan, &work, fragment, candidates, context, name).await?;
    }
    context.check_cancelled()?;
    Ok(true)
}

async fn submit(
    work: &Arc<Work>,
    credit: MemoryReservation,
    context: &StreamOperatorContext<'_>,
    name: &str,
) -> Result<crate::runtime::streaming::gather_work::WorkOutput<Vec<Fragment>>> {
    let scope = context
        .gather_client(GatherOperatorId::new(name.into()))
        .scope()?;
    let ticket = scope
        .submit_parallel_work(work.clone(), credit, GatherStop::from_job(context.job()))
        .await
        .map_err(|failure| match failure {
            AdmissionFailure::Budget { source, .. } => df_error(name, source),
            AdmissionFailure::Runtime(error) => error,
        })?;
    ticket.finish().await
}

async fn merge_fragment(
    plan: &IncrementalSql,
    work: &Work,
    fragment: &Fragment,
    candidates: &mut InputCandidates,
    context: &StreamOperatorContext<'_>,
    name: &str,
) -> Result<()> {
    let ranks = resolve_ranks(plan, work, &fragment.keys, candidates, context, name).await?;
    let partial = candidates.partial.as_mut().expect("grouped SUM partial");
    merge_states(partial, &fragment.states, &ranks, context, name).await
}

async fn resolve_ranks(
    plan: &IncrementalSql,
    work: &Work,
    keys: &[LocalKey],
    candidates: &mut InputCandidates,
    context: &StreamOperatorContext<'_>,
    name: &str,
) -> Result<Vec<usize>> {
    let mut ranks = Vec::with_capacity(keys.len());
    let mut chunk = usize::MAX;
    let mut arrays = Vec::new();
    for key in keys {
        if ranks.len() % 1024 == 0 {
            context.check_cancelled()?;
            tokio::task::yield_now().await;
        }
        if chunk != key.chunk {
            chunk = key.chunk;
            arrays = work.key_arrays(chunk);
        }
        ranks.push(plan.group_rank(&key.encoded, &arrays, key.row, candidates, name)?);
    }
    Ok(ranks)
}

async fn merge_states(
    partial: &mut PartialGroups,
    states: &[Vec<ArrayRef>],
    ranks: &[usize],
    context: &StreamOperatorContext<'_>,
    name: &str,
) -> Result<()> {
    for (index, states) in states.iter().enumerate() {
        for (chunk, indices) in ranks.chunks(CHUNK_ROWS).enumerate() {
            context.check_cancelled()?;
            let arrays = states
                .iter()
                .map(|array| array.slice(chunk * CHUNK_ROWS, indices.len()))
                .collect::<Vec<_>>();
            partial.accumulators[index]
                .merge_batch(&arrays, indices, partial.slots.len())
                .map_err(|error| df_error(name, error))?;
            tokio::task::yield_now().await;
        }
    }
    partial.check_size(name)
}

#[cfg(test)]
#[path = "grouped_parallel_admission_tests.rs"]
mod admission_tests;
