use super::{
    AggregateFunctionExpr, Arc, ArrayRef, DataType, Expr, LogicalPlan, MemoryReservation,
    RecordBatch, Result, ScalarValue, SchemaRef, checked_bytes, df_error,
};
use crate::runtime::streaming::gather_work::{
    AdmissionFailure, GatherOperatorId, GatherStop, OwnedCpuWork,
};
use crate::{DataFusionConfig, DataFusionRuntime, StreamOperatorContext};
use serde::{Deserialize, Serialize};

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
}

#[derive(Clone, Copy)]
enum Kind {
    Sum,
    Average,
    Count,
    Extrema32,
    Extrema64,
}

impl Kind {
    fn of(expression: &AggregateFunctionExpr) -> Option<Self> {
        if expression.fun().name() == "count" && expression.field().data_type() == &DataType::Int64
        {
            return Some(Self::Count);
        }
        if matches!(expression.fun().name(), "min" | "max") {
            return match expression.field().data_type() {
                DataType::Float32 => Some(Self::Extrema32),
                DataType::Float64 => Some(Self::Extrema64),
                _ => None,
            };
        }
        if expression.field().data_type() != &DataType::Float64 {
            return None;
        }
        match expression.fun().name() {
            "sum" => Some(Self::Sum),
            "avg" => Some(Self::Average),
            _ => None,
        }
    }

    fn width(self) -> usize {
        match self {
            Self::Sum | Self::Count | Self::Extrema32 | Self::Extrema64 => 1,
            Self::Average => 2,
        }
    }
}

pub(super) struct Proof {
    pub policy: Policy,
    kinds: Arc<[Kind]>,
    operator: GatherOperatorId,
    _reservation: MemoryReservation,
}

fn saved_count(kind: Kind, values: &[ScalarValue], name: &str) -> Result<u64> {
    match (kind, values) {
        (_, [])
        | (Kind::Sum | Kind::Extrema64, [ScalarValue::Float64(_)])
        | (Kind::Extrema32, [ScalarValue::Float32(_)]) => Ok(0),
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

pub(super) fn raw_selected(raw: &LogicalPlan, schema: &SchemaRef) -> bool {
    let Some((_, aggregate)) = super::shape(raw) else {
        return false;
    };
    if !aggregate.group_expr.is_empty() || aggregate.aggr_expr.is_empty() {
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
                [Expr::Column(_) | Expr::Literal(ScalarValue::Int64(Some(1)), _)]
            );
        }
        if !matches!(function.func.name(), "sum" | "avg" | "min" | "max") {
            return false;
        }
        let [Expr::Column(column)] = function.params.args.as_slice() else {
            return false;
        };
        schema.field_with_name(&column.name).is_ok_and(|field| {
            matches!(field.data_type(), DataType::Float32 | DataType::Float64)
                || (function.func.name() == "avg" && field.data_type().is_integer())
        })
    })
}

impl Proof {
    pub(super) fn new(
        runtime: &DataFusionRuntime,
        expressions: &[Arc<AggregateFunctionExpr>],
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
            policy: Policy {
                config: runtime.compact_runtime_config(),
                factory: Factory::ScalarNativeV1,
                model: Model::Df54SingleSourceSplitRecordsV1,
            },
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
        saved_count(self.kinds[index], values, name)?;
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
        for &kind in self.kinds.iter() {
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
        arguments: (&[Arc<AggregateFunctionExpr>], &[Vec<ScalarValue>]),
        credit: MemoryReservation,
        context: &StreamOperatorContext<'_>,
        name: &str,
    ) -> Result<Vec<Vec<ScalarValue>>> {
        context.check_cancelled()?;
        let (records, input_owner) = input;
        let (expressions, values) = arguments;
        if expressions.len() != self.kinds.len() || values.len() != self.kinds.len() {
            return Err(df_error(name, "global scalar argument count differs"));
        }
        let empty = records.iter().all(|record| record.num_rows() == 0);
        let charge = if empty {
            checked_bytes(4096, [(self.kinds.len(), 1024)], name)?
        } else {
            request_charge(
                records,
                input_owner.is_some(),
                self.policy.config.batch_size,
                self.kinds.len(),
                name,
            )?
        };
        super::ensure_reservation(&credit, charge, name)?;
        for (&kind, state) in self.kinds.iter().zip(values) {
            saved_count(kind, state, name)?;
        }
        if empty {
            return Ok(values.to_vec());
        }
        let work = RecordWork {
            records: records.to_vec(),
            expressions: expressions.to_vec(),
            states: values.to_vec(),
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
}

fn request_charge(
    records: &[RecordBatch],
    has_owner: bool,
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
    records.iter().try_fold(
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
    )
}

struct RecordWork {
    records: Vec<RecordBatch>,
    expressions: Vec<Arc<AggregateFunctionExpr>>,
    states: Vec<Vec<ScalarValue>>,
    batch_size: usize,
    name: String,
    _input_owner: Option<Arc<MemoryReservation>>,
}

impl OwnedCpuWork for RecordWork {
    type Output = Vec<Vec<ScalarValue>>;

    fn run(self, stop: &GatherStop) -> Result<Self::Output> {
        let mut accumulators = self
            .expressions
            .iter()
            .zip(&self.states)
            .map(|(expression, state)| {
                stop.check()?;
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
            .collect::<Result<Vec<_>>>()?;
        for record in &self.records {
            stop.check()?;
            if record.num_rows() == 0 {
                continue;
            }
            for offset in (0..record.num_rows()).step_by(self.batch_size) {
                stop.check()?;
                let rows = self.batch_size.min(record.num_rows() - offset);
                let batch = record.slice(offset, rows);
                for (expression, accumulator) in self.expressions.iter().zip(&mut accumulators) {
                    stop.check()?;
                    let arrays = expression
                        .expressions()
                        .iter()
                        .map(|expression| {
                            expression
                                .evaluate(&batch)
                                .and_then(|value| value.into_array(rows))
                                .map_err(|error| df_error(&self.name, error))
                        })
                        .collect::<Result<Vec<ArrayRef>>>()?;
                    accumulator
                        .update_batch(&arrays)
                        .map_err(|error| df_error(&self.name, error))?;
                }
            }
        }
        accumulators
            .iter_mut()
            .map(|accumulator| {
                stop.check()?;
                accumulator
                    .state()
                    .map_err(|error| df_error(&self.name, error))
            })
            .collect()
    }
}
