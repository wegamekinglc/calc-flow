use super::{
    AggregateFunctionExpr, Arc, ArrayRef, DataType, Expr, LogicalPlan, MemoryReservation,
    RecordBatch, Result, ScalarValue, SchemaRef, checked_bytes, df_error,
};
use crate::runtime::streaming::gather_work::{
    AdmissionFailure, GatherOperatorId, GatherStop, OwnedCpuWork,
};
use crate::{DataFusionConfig, DataFusionRuntime, StreamOperatorContext};
use serde::{Deserialize, Serialize};

#[path = "global_record/coalescer.rs"]
pub(in crate::operator::sql) mod coalescer;

type Arguments<'a> = (
    &'a [Arc<AggregateFunctionExpr>],
    &'a [Option<Arc<dyn super::PhysicalExpr>>],
    &'a [Vec<ScalarValue>],
    Option<&'a super::predicate::InputPredicate>,
);

#[derive(Clone)]
pub(super) struct Update {
    pub values: Vec<Vec<ScalarValue>>,
    pub coalesced: Option<Arc<coalescer::State>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::operator::sql) struct Policy {
    #[serde(deserialize_with = "super::grouped_float::required_config")]
    pub config: DataFusionConfig,
    pub factory: Factory,
    pub model: Model,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(in crate::operator::sql) enum Factory {
    ScalarNativeV1,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(in crate::operator::sql) enum Model {
    Df54SingleSourceSplitRecordsV1,
    Df54SingleSourceSplitFilterCoalescedV1,
}

#[derive(Clone)]
enum Kind {
    Scalar(DataType),
    Average,
    Count,
}

impl Kind {
    fn of(expression: &AggregateFunctionExpr) -> Option<Self> {
        let field = expression.field();
        let dtype = field.data_type();
        match expression.fun().name() {
            "count" if dtype == &DataType::Int64 => Some(Self::Count),
            "avg" if dtype == &DataType::Float64 => Some(Self::Average),
            "sum"
                if matches!(
                    dtype,
                    DataType::Float64 | DataType::Int64 | DataType::UInt64
                ) =>
            {
                Some(Self::Scalar(dtype.clone()))
            }
            "min" | "max"
                if dtype.is_integer() || matches!(dtype, DataType::Float32 | DataType::Float64) =>
            {
                Some(Self::Scalar(dtype.clone()))
            }
            _ => None,
        }
    }

    fn width(&self) -> usize {
        match self {
            Self::Scalar(_) | Self::Count => 1,
            Self::Average => 2,
        }
    }
}

pub(super) struct Proof {
    temporary_nodes: usize,
    pub policy: Policy,
    pub coalesced: Option<Arc<coalescer::State>>,
    kinds: Arc<[Kind]>,
    operator: GatherOperatorId,
    _reservation: MemoryReservation,
}

fn saved_count(kind: &Kind, values: &[ScalarValue], name: &str) -> Result<u64> {
    match (kind, values) {
        (_, []) => Ok(0),
        (Kind::Scalar(dtype), [value]) if value.data_type() == *dtype => Ok(0),
        (Kind::Count, [ScalarValue::Int64(Some(count))]) => {
            u64::try_from(*count).map_err(|error| df_error(name, error))
        }
        (Kind::Average, [ScalarValue::UInt64(Some(count)), ScalarValue::Float64(sum)])
            if (*count == 0) == sum.is_none() =>
        {
            Ok(*count)
        }
        _ => Err(df_error(name, "global record scalar state is invalid")),
    }
}

pub(super) fn raw_selected(raw: &LogicalPlan) -> bool {
    let Some((_, aggregate)) = super::shape(raw) else {
        return false;
    };
    if !aggregate.group_expr.is_empty() || aggregate.aggr_expr.is_empty() {
        return false;
    }
    if !matches!(aggregate.input.as_ref(), LogicalPlan::TableScan(_))
        && !matches!(aggregate.input.as_ref(), LogicalPlan::Filter(filter) if matches!(filter.input.as_ref(), LogicalPlan::TableScan(_)))
    {
        return false;
    }
    if !aggregate.aggr_expr.iter().any(|expression| {
        matches!(super::unalias(expression), Expr::AggregateFunction(function) if matches!(function.func.name(), "sum" | "avg"))
    }) {
        return false;
    }
    aggregate.aggr_expr.iter().all(|expression| {
        let Expr::AggregateFunction(function) = super::unalias(expression) else {
            return false;
        };
        if function.func.name() == "count" {
            return matches!(
                function.params.args.as_slice(),
                [Expr::Literal(ScalarValue::Int64(Some(1)), _)]
            ) || matches!(function.params.args.as_slice(), [argument] if super::argument_type(argument, aggregate.input.schema()).is_some());
        }
        if !matches!(function.func.name(), "sum" | "avg" | "min" | "max") {
            return false;
        }
        let [argument] = function.params.args.as_slice() else {
            return false;
        };
        super::argument_type(argument, aggregate.input.schema()).is_some_and(|dtype| {
            matches!(dtype, DataType::Float32 | DataType::Float64) || dtype.is_integer()
        })
    })
}

impl Proof {
    pub(super) fn new(
        runtime: &DataFusionRuntime,
        expressions: &[Arc<AggregateFunctionExpr>],
        filtered: bool,
        input_nodes: usize,
        name: &str,
    ) -> Result<Option<Self>> {
        if !runtime.grouped_float_model_supported(name)? {
            return Ok(None);
        }
        if expressions.is_empty()
            || expressions
                .iter()
                .any(|expression| Kind::of(expression).is_none())
        {
            return Ok(None);
        }
        let reservation = runtime.incremental_reservation(name);
        super::ensure_reservation(
            &reservation,
            checked_bytes(4096, [(name.len(), 2), (expressions.len(), 512)], name)?,
            name,
        )?;
        Ok(Some(Self {
            temporary_nodes: input_nodes.saturating_sub(expressions.len()),
            policy: Policy {
                config: runtime.compact_runtime_config(),
                factory: Factory::ScalarNativeV1,
                model: if filtered {
                    Model::Df54SingleSourceSplitFilterCoalescedV1
                } else {
                    Model::Df54SingleSourceSplitRecordsV1
                },
            },
            coalesced: None,
            kinds: expressions
                .iter()
                .map(|expression| Kind::of(expression).expect("selected kind"))
                .collect(),
            operator: GatherOperatorId::new(name.to_owned().into()),
            _reservation: reservation,
        }))
    }

    pub(super) fn result(
        &self,
        index: usize,
        values: &[ScalarValue],
        name: &str,
    ) -> Result<ScalarValue> {
        saved_count(&self.kinds[index], values, name)?;
        super::grouped_sum::result(values, name)
    }

    pub(super) fn validate_state(
        &self,
        records: &[RecordBatch],
        rows: u64,
        name: &str,
    ) -> Result<()> {
        if rows == 0 || records.iter().map(RecordBatch::num_rows).sum::<usize>() != 1 {
            return Err(df_error(
                name,
                "global checkpoint must contain one scalar row",
            ));
        }
        let record = records
            .iter()
            .find(|record| record.num_rows() != 0)
            .expect("scalar row");
        let mut columns = record.columns().iter();
        for kind in self.kinds.iter() {
            let mut values = [ScalarValue::Null, ScalarValue::Null];
            for value in &mut values[..kind.width()] {
                let array = columns
                    .next()
                    .ok_or_else(|| df_error(name, "global scalar state is missing"))?;
                *value =
                    ScalarValue::try_from_array(array, 0).map_err(|error| df_error(name, error))?;
            }
            let count = saved_count(kind, &values[..kind.width()], name)?;
            if count > rows {
                return Err(df_error(name, "global aggregate count exceeds input rows"));
            }
        }
        if columns.next().is_some() {
            return Err(df_error(name, "global scalar state has extra columns"));
        }
        Ok(())
    }

    pub(super) async fn update(
        &self,
        input: (&[RecordBatch], Option<Arc<MemoryReservation>>),
        arguments: Arguments<'_>,
        credit: MemoryReservation,
        context: &StreamOperatorContext<'_>,
        name: &str,
    ) -> Result<Update> {
        context.check_cancelled()?;
        let (records, input_owner) = input;
        let (expressions, filters, values, predicate) = arguments;
        if self.uses_coalescing() != predicate.is_some() {
            return Err(df_error(name, "global scalar predicate differs from model"));
        }
        if expressions.len() != self.kinds.len() || values.len() != self.kinds.len() {
            return Err(df_error(name, "global scalar argument count differs"));
        }
        if !filters.is_empty() && filters.len() != expressions.len() {
            return Err(df_error(name, "global scalar filter count differs"));
        }
        let empty = records.iter().all(|record| record.num_rows() == 0);
        let charge = self.update_charge(
            records,
            input_owner.is_some(),
            !filters.is_empty(),
            predicate,
            name,
        )?;
        let coalesced_credit =
            (!empty && predicate.is_some()).then(|| Arc::new(credit.new_empty()));
        super::ensure_reservation(&credit, charge, name)?;
        for (kind, state) in self.kinds.iter().zip(values) {
            saved_count(kind, state, name)?;
        }
        if empty {
            return Ok(Update {
                values: values.to_vec(),
                coalesced: None,
            });
        }
        let work = RecordWork {
            records: records.to_vec(),
            expressions: expressions.to_vec(),
            filters: filters.to_vec(),
            states: values.to_vec(),
            predicate: predicate.cloned(),
            previous: self.coalesced.clone(),
            coalesced_scratch: coalesced_credit.as_ref().map(|owner| owner.new_empty()),
            coalesced_credit,
            batch_size: self.policy.config.batch_size,
            name: name.to_owned(),
            _input_owner: input_owner,
        };
        let scope = context.gather_client(self.operator.clone()).scope()?;
        let ticket = scope
            .submit_work(work, credit, GatherStop::from_job(context.job()))
            .await
            .map_err(|failure| match failure {
                AdmissionFailure::Budget { source, .. } => df_error(name, source),
                AdmissionFailure::Runtime(error) => error,
            })?;
        let output = ticket.finish().await?;
        context.check_cancelled()?;
        let values = output.value.clone();
        drop(output);
        Ok(values)
    }

    fn update_charge(
        &self,
        records: &[RecordBatch],
        has_owner: bool,
        has_filters: bool,
        predicate: Option<&super::predicate::InputPredicate>,
        name: &str,
    ) -> Result<usize> {
        if records.iter().all(|record| record.num_rows() == 0) {
            return checked_bytes(4096, [(self.kinds.len(), 1024)], name);
        }
        let charge = request_charge(
            records,
            has_owner,
            has_filters,
            self.policy.config.batch_size,
            self.kinds.len(),
            name,
        )?;
        let workspace = expression_workspace(
            records,
            self.policy.config.batch_size,
            self.uses_coalescing(),
            self.temporary_nodes,
            name,
        )?;
        let predicate = predicate
            .map(|predicate| predicate.workspace(coalescer::ROWS, self.kinds.len(), name))
            .transpose()?
            .unwrap_or(0);
        checked_bytes(charge, [(workspace, 1), (predicate, 1)], name)
    }

    pub(super) fn uses_coalescing(&self) -> bool {
        self.policy.model == Model::Df54SingleSourceSplitFilterCoalescedV1
    }
}

fn expression_workspace(
    records: &[RecordBatch],
    batch_size: usize,
    coalesced: bool,
    nodes: usize,
    name: &str,
) -> Result<usize> {
    let rows = if coalesced {
        coalescer::ROWS
    } else {
        records
            .iter()
            .map(RecordBatch::num_rows)
            .max()
            .unwrap_or(0)
            .min(batch_size)
    };
    let width = checked_bytes(0, [(nodes, 64)], name)?;
    checked_bytes(0, [(rows, width)], name)
}

fn request_charge(
    records: &[RecordBatch],
    has_owner: bool,
    has_filters: bool,
    batch_size: usize,
    aggregates: usize,
    name: &str,
) -> Result<usize> {
    let rows = records
        .iter()
        .map(RecordBatch::num_rows)
        .max()
        .unwrap_or(0)
        .min(batch_size);
    let bytes = records.iter().try_fold(
        checked_bytes(
            131_072,
            [
                (rows, 16),
                (rows.div_ceil(8), 4),
                (records.len(), size_of::<RecordBatch>() * 2),
                (name.len(), 2),
                (aggregates, 1024),
            ],
            name,
        )?,
        |bytes, record| {
            let schema = super::super::ipc::schema_bytes(&record.schema())
                .map_err(|error| df_error(name, error))?;
            let backing = if has_owner {
                0
            } else {
                record.columns().iter().try_fold(0, |bytes, array| {
                    checked_bytes(bytes, [(array.get_array_memory_size(), 1)], name)
                })?
            };
            checked_bytes(
                bytes,
                [(schema, 4), (backing, 1), (record.num_columns(), 128)],
                name,
            )
        },
    )?;
    if has_filters {
        checked_bytes(
            bytes,
            [(filter_workspace(records, batch_size, name)?, 4)],
            name,
        )
    } else {
        Ok(bytes)
    }
}

fn filter_workspace(records: &[RecordBatch], batch_size: usize, name: &str) -> Result<usize> {
    let mut peak = 0;
    for record in records {
        for offset in (0..record.num_rows()).step_by(batch_size) {
            let rows = batch_size.min(record.num_rows() - offset);
            let bytes = record.columns().iter().try_fold(0, |bytes, array| {
                let size = array
                    .slice(offset, rows)
                    .to_data()
                    .get_slice_memory_size()
                    .map_err(|error| df_error(name, error))?;
                checked_bytes(bytes, [(size, 1)], name)
            })?;
            peak = peak.max(bytes);
        }
    }
    Ok(peak)
}

struct RecordWork {
    records: Vec<RecordBatch>,
    expressions: Vec<Arc<AggregateFunctionExpr>>,
    filters: Vec<Option<Arc<dyn super::PhysicalExpr>>>,
    states: Vec<Vec<ScalarValue>>,
    predicate: Option<super::predicate::InputPredicate>,
    previous: Option<Arc<coalescer::State>>,
    coalesced_scratch: Option<MemoryReservation>,
    coalesced_credit: Option<Arc<MemoryReservation>>,
    batch_size: usize,
    name: String,
    _input_owner: Option<Arc<MemoryReservation>>,
}

impl OwnedCpuWork for RecordWork {
    type Output = Update;

    fn run(mut self, stop: &GatherStop) -> Result<Self::Output> {
        if let Some(predicate) = self.predicate.take() {
            return coalescer::run(&self, &predicate, stop);
        }
        let check = || stop.check();
        let mut accumulators = self.accumulators(&self.states, &check)?;
        for record in &self.records {
            stop.check()?;
            for offset in (0..record.num_rows()).step_by(self.batch_size) {
                stop.check()?;
                let rows = self.batch_size.min(record.num_rows() - offset);
                self.update_batch(&record.slice(offset, rows), &mut accumulators, &check)?;
            }
        }
        Ok(Update {
            values: self.values(&mut accumulators, &check)?,
            coalesced: None,
        })
    }
}

impl RecordWork {
    fn accumulators(
        &self,
        states: &[Vec<ScalarValue>],
        check: &dyn Fn() -> Result<()>,
    ) -> Result<Vec<Box<dyn datafusion::logical_expr::Accumulator>>> {
        self.expressions
            .iter()
            .zip(states)
            .map(|(expression, state)| {
                check()?;
                let mut accumulator = expression
                    .create_accumulator()
                    .map_err(|error| df_error(&self.name, error))?;
                if !state.is_empty() {
                    let arrays = state
                        .iter()
                        .map(ScalarValue::to_array)
                        .collect::<datafusion::common::Result<Vec<_>>>()
                        .map_err(|error| df_error(&self.name, error))?;
                    accumulator
                        .merge_batch(&arrays)
                        .map_err(|error| df_error(&self.name, error))?;
                }
                Ok(accumulator)
            })
            .collect()
    }

    fn update_batch(
        &self,
        batch: &RecordBatch,
        accumulators: &mut [Box<dyn datafusion::logical_expr::Accumulator>],
        check: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        for (index, (expression, accumulator)) in self
            .expressions
            .iter()
            .zip(accumulators.iter_mut())
            .enumerate()
        {
            check()?;
            let filtered;
            let batch = if let Some(expression) = self.filters.get(index).and_then(Option::as_ref) {
                let filter = super::aggregate_filter(expression.as_ref(), batch, &self.name)?;
                filtered = datafusion::arrow::compute::filter_record_batch(batch, &filter)
                    .map_err(|error| df_error(&self.name, error))?;
                &filtered
            } else {
                batch
            };
            let arrays = expression
                .expressions()
                .iter()
                .map(|expression| {
                    expression
                        .evaluate(batch)
                        .and_then(|value| value.into_array(batch.num_rows()))
                        .map_err(|error| df_error(&self.name, error))
                })
                .collect::<Result<Vec<ArrayRef>>>()?;
            accumulator
                .update_batch(&arrays)
                .map_err(|error| df_error(&self.name, error))?;
        }
        Ok(())
    }

    fn values(
        &self,
        accumulators: &mut [Box<dyn datafusion::logical_expr::Accumulator>],
        check: &dyn Fn() -> Result<()>,
    ) -> Result<Vec<Vec<ScalarValue>>> {
        accumulators
            .iter_mut()
            .map(|accumulator| {
                check()?;
                accumulator
                    .state()
                    .map_err(|error| df_error(&self.name, error))
            })
            .collect()
    }
}
