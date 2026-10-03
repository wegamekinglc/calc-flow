use super::{
    AggregateFunctionExpr, Arc, ArrayRef, DataType, Expr, LogicalPlan, MemoryReservation,
    PhysicalExpr, RecordBatch, Result, ScalarValue, SchemaRef, checked_bytes, df_error,
};
use crate::runtime::streaming::gather_work::{
    AdmissionFailure, GatherOperatorId, GatherStop, OwnedCpuWork,
};
use crate::{DataFusionConfig, DataFusionRuntime, StreamOperatorContext};
use datafusion::arrow::{
    array::{Array, ArrowNativeTypeOp, Float64Array},
    compute::sum,
};
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
    ScalarFloat64V1,
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
}

pub(super) struct Proof {
    pub policy: Policy,
    kind: Kind,
    operator: GatherOperatorId,
    _reservation: MemoryReservation,
}

#[derive(Clone, Copy)]
struct Literal {
    sum: Option<f64>,
    count: u64,
}

impl Literal {
    fn saved(kind: Kind, values: &[ScalarValue], name: &str) -> Result<Self> {
        let (sum, count) = match (kind, values) {
            (_, []) => (None, 0),
            (Kind::Sum, [ScalarValue::Float64(sum)]) => (*sum, 0),
            (Kind::Average, [ScalarValue::UInt64(Some(count)), ScalarValue::Float64(sum)])
                if (*count == 0) == sum.is_none() =>
            {
                (*sum, *count)
            }
            _ => return Err(df_error(name, "global record scalar state is invalid")),
        };
        Ok(Self { sum, count })
    }

    fn update(&mut self, kind: Kind, array: &Float64Array, name: &str) -> Result<()> {
        if matches!(kind, Kind::Average) {
            self.count = self
                .count
                .checked_add(
                    u64::try_from(array.len() - array.null_count())
                        .map_err(|error| df_error(name, error))?,
                )
                .ok_or_else(|| df_error(name, "global AVG count overflowed"))?;
        }
        if let Some(value) = sum(array) {
            let saved = self.sum.get_or_insert(0.0);
            match kind {
                Kind::Sum => *saved = saved.add_wrapping(value),
                Kind::Average => *saved += value,
            }
        }
        Ok(())
    }

    fn scalars(self, kind: Kind) -> Vec<ScalarValue> {
        let mut values = Vec::with_capacity(if matches!(kind, Kind::Average) { 2 } else { 1 });
        if matches!(kind, Kind::Average) {
            values.push(ScalarValue::UInt64(Some(self.count)));
        }
        values.push(ScalarValue::Float64(self.sum));
        values
    }
}

pub(super) fn raw_selected(raw: &LogicalPlan, schema: &SchemaRef) -> bool {
    let Some((_, aggregate)) = super::shape(raw) else {
        return false;
    };
    if !aggregate.group_expr.is_empty() || aggregate.aggr_expr.len() != 1 {
        return false;
    }
    let Expr::AggregateFunction(function) = super::unalias(&aggregate.aggr_expr[0]) else {
        return false;
    };
    if !matches!(function.func.name(), "sum" | "avg") {
        return false;
    }
    let [Expr::Column(column)] = function.params.args.as_slice() else {
        return false;
    };
    schema
        .field_with_name(&column.name)
        .is_ok_and(|field| matches!(field.data_type(), DataType::Float32 | DataType::Float64))
}

impl Proof {
    pub(super) fn new(
        runtime: &DataFusionRuntime,
        expression: &AggregateFunctionExpr,
        name: &str,
    ) -> Result<Option<Self>> {
        if !runtime.grouped_float_model_supported(name)? {
            return Ok(None);
        }
        let kind = match expression.fun().name() {
            "sum" => Kind::Sum,
            "avg" => Kind::Average,
            _ => return Ok(None),
        };
        if expression.field().data_type() != &DataType::Float64 {
            return Ok(None);
        }
        let reservation = runtime.incremental_reservation(name);
        super::ensure_reservation(
            &reservation,
            checked_bytes(4096, [(name.len(), 2)], name)?,
            name,
        )?;
        Ok(Some(Self {
            policy: Policy {
                config: runtime.compact_runtime_config(),
                factory: Factory::ScalarFloat64V1,
                model: Model::Df54SingleSourceSplitRecordsV1,
            },
            kind,
            operator: GatherOperatorId::new(name.to_owned().into()),
            _reservation: reservation,
        }))
    }

    pub(super) fn result(&self, values: &[ScalarValue], name: &str) -> Result<ScalarValue> {
        Literal::saved(self.kind, values, name)?;
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
        let values = record
            .columns()
            .iter()
            .map(|array| {
                ScalarValue::try_from_array(array, 0).map_err(|error| df_error(name, error))
            })
            .collect::<Result<Vec<_>>>()?;
        let saved = Literal::saved(self.kind, &values, name)?;
        if saved.count > rows {
            return Err(df_error(name, "global AVG count exceeds input rows"));
        }
        Ok(())
    }

    pub(super) async fn update(
        &self,
        input: (&[RecordBatch], Option<Arc<MemoryReservation>>),
        arguments: (&Arc<dyn PhysicalExpr>, &[ScalarValue]),
        credit: MemoryReservation,
        context: &StreamOperatorContext<'_>,
        name: &str,
    ) -> Result<Vec<ScalarValue>> {
        context.check_cancelled()?;
        let (records, input_owner) = input;
        let (expression, values) = arguments;
        let state = Literal::saved(self.kind, values, name)?;
        if records.iter().all(|record| record.num_rows() == 0) {
            return Ok(state.scalars(self.kind));
        }
        let charge = request_charge(
            records,
            input_owner.is_some(),
            self.policy.config.batch_size,
            name,
        )?;
        super::ensure_reservation(&credit, charge, name)?;
        let work = RecordWork {
            records: records.to_vec(),
            expression: expression.clone(),
            state,
            kind: self.kind,
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
        let values = output.value.scalars(self.kind);
        drop(output);
        Ok(values)
    }
}

fn request_charge(
    records: &[RecordBatch],
    has_owner: bool,
    batch_size: usize,
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
    expression: Arc<dyn PhysicalExpr>,
    state: Literal,
    kind: Kind,
    batch_size: usize,
    name: String,
    _input_owner: Option<Arc<MemoryReservation>>,
}

impl OwnedCpuWork for RecordWork {
    type Output = Literal;

    fn run(mut self, stop: &GatherStop) -> Result<Self::Output> {
        for record in &self.records {
            stop.check()?;
            if record.num_rows() == 0 {
                continue;
            }
            for offset in (0..record.num_rows()).step_by(self.batch_size) {
                stop.check()?;
                let rows = self.batch_size.min(record.num_rows() - offset);
                let batch = record.slice(offset, rows);
                let array: ArrayRef = self
                    .expression
                    .evaluate(&batch)
                    .and_then(|value| value.into_array(rows))
                    .map_err(|error| df_error(&self.name, error))?;
                let values = array
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| {
                        df_error(&self.name, "global record argument must be Float64")
                    })?;
                self.state.update(self.kind, values, &self.name)?;
                stop.check()?;
            }
        }
        Ok(self.state)
    }
}
