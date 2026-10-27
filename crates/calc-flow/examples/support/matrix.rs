//! Trusted Rust matrix provider used by the two array/table examples.
use crate::support::invalid;
use async_trait::async_trait;
use calc_flow::{
    Batch, BatchMetadata, BatchOperator, BatchOperatorContext, BatchOperatorFactory, EventTime,
    ExternalOperatorSpec, JsonMap, OperatorMetadata, Port, ProviderRegistry, Result,
    StaticArraySnapshot, StaticArrayValues, StreamCollector, StreamOperator, StreamOperatorContext,
    StreamOperatorFactory, StreamOperatorLifecycle,
};
use datafusion::arrow::{array::Float64Array, record_batch::RecordBatch};
use std::{collections::BTreeMap, sync::Arc};

pub struct MatrixFactory;
pub fn register(registry: &ProviderRegistry) -> Result<()> {
    registry.register_batch("rust-array", "matmul", "1", Arc::new(MatrixFactory))?;
    registry.register_stream("rust-array", "matmul", "1", Arc::new(MatrixFactory))
}
struct Matrix {
    inputs: Vec<Port>,
    outputs: Vec<Port>,
    attach: bool,
    weights: Option<StaticArraySnapshot>,
}
fn create(spec: &ExternalOperatorSpec, inputs: Vec<Port>, outputs: Vec<Port>) -> Result<Matrix> {
    let attach = spec
        .options()
        .get("attach")
        .and_then(serde_json::Value::as_bool)
        .ok_or_else(|| invalid("attach", "expected boolean"))?;
    Ok(Matrix {
        inputs,
        outputs,
        attach,
        weights: None,
    })
}
impl BatchOperatorFactory for MatrixFactory {
    fn create(
        &self,
        spec: &ExternalOperatorSpec,
        inputs: Vec<Port>,
        outputs: Vec<Port>,
    ) -> Result<Box<dyn BatchOperator>> {
        Ok(Box::new(create(spec, inputs, outputs)?))
    }
}
impl StreamOperatorFactory for MatrixFactory {
    fn create(
        &self,
        spec: &ExternalOperatorSpec,
        inputs: Vec<Port>,
        outputs: Vec<Port>,
    ) -> Result<Box<dyn StreamOperator>> {
        Ok(Box::new(create(spec, inputs, outputs)?))
    }
}
impl OperatorMetadata for Matrix {
    fn name(&self) -> &'static str {
        "matmul"
    }
    fn input_ports(&self) -> &[Port] {
        &self.inputs
    }
    fn output_ports(&self) -> &[Port] {
        &self.outputs
    }
    fn configuration(&self) -> JsonMap {
        BTreeMap::from([("attach".into(), serde_json::json!(self.attach))])
    }
}
fn weights(batch: &Batch, columns: usize) -> Result<StaticArraySnapshot> {
    let snapshot = batch
        .static_array_snapshot()
        .ok_or_else(|| invalid("weights", "expected owned float array"))?;
    if snapshot.backend() != "rust-array"
        || snapshot.shape() != [2, u64::try_from(columns).unwrap()]
        || snapshot.nulls().is_some()
    {
        return Err(invalid(
            "weights",
            "expected non-null rust-array matrix with two input rows",
        ));
    }
    let StaticArrayValues::Float(_) = snapshot.values() else {
        return Err(invalid("weights", "expected float64"));
    };
    Ok(snapshot)
}
fn float_values(snapshot: &StaticArraySnapshot) -> &[f64] {
    let StaticArrayValues::Float(values) = snapshot.values() else {
        unreachable!("validated float weights")
    };
    values
}
impl Matrix {
    fn multiply(&self, table: &Batch, weights: &[f64], copy_bytes: usize) -> Result<Batch> {
        if self.attach {
            attach_scores(table, weights, copy_bytes)
        } else {
            multiply_columns(table, weights)
        }
    }
}

fn float_column<'a>(record: &'a RecordBatch, name: &str) -> Result<&'a Float64Array> {
    record
        .column_by_name(name)
        .and_then(|a| a.as_any().downcast_ref::<Float64Array>())
        .ok_or_else(|| invalid("table", "expected float64 columns"))
}

fn attach_record(record: &RecordBatch, weights: &[f64]) -> Result<RecordBatch> {
    let a = float_column(record, "return")?;
    let b = float_column(record, "volatility")?;
    let scores: Vec<_> = a
        .iter()
        .zip(b.iter())
        .map(|(a, b)| a.zip(b).map(|(a, b)| a * weights[0] + b * weights[1]))
        .collect();
    let mut fields = record.schema().fields().to_vec();
    fields.push(Arc::new(datafusion::arrow::datatypes::Field::new(
        "risk_score",
        datafusion::arrow::datatypes::DataType::Float64,
        true,
    )));
    let mut arrays = record.columns().to_vec();
    arrays.push(Arc::new(Float64Array::from(scores)));
    RecordBatch::try_new(
        Arc::new(datafusion::arrow::datatypes::Schema::new(fields)),
        arrays,
    )
    .map_err(|e| invalid("output", e.to_string()))
}

fn attach_scores(table: &Batch, weights: &[f64], copy_bytes: usize) -> Result<Batch> {
    let records = table
        .table_payload()?
        .batches()
        .iter()
        .map(|record| attach_record(record, weights))
        .collect::<Result<Vec<_>>>()?;
    let metadata = BatchMetadata::new(
        "rust-array",
        0,
        BTreeMap::from([(
            "copy_bytes".into(),
            serde_json::json!({"weights":copy_bytes}),
        )]),
    )?;
    Batch::table(records, metadata)
}

fn multiply_columns(table: &Batch, weights: &[f64]) -> Result<Batch> {
    let mut values = Vec::new();
    for record in table.table_payload()?.batches() {
        let a = float_column(record, "quantity")?;
        let b = float_column(record, "unit_price")?;
        for (a, b) in a.iter().zip(b.iter()) {
            let (a, b) = a
                .zip(b)
                .ok_or_else(|| invalid("table", "non-null demo columns required"))?;
            values.extend([
                a * weights[0] + b * weights[2],
                a * weights[1] + b * weights[3],
            ]);
        }
    }
    Batch::static_array_float(
        "rust-array",
        "float64",
        vec![u64::try_from(table.num_rows()).unwrap(), 2],
        None,
        values,
    )
}

#[async_trait]
impl BatchOperator for Matrix {
    async fn process(
        &mut self,
        inputs: &BTreeMap<String, Batch>,
        _: &BatchOperatorContext<'_>,
    ) -> Result<BTreeMap<String, Batch>> {
        let weights = weights(&inputs["weights"], if self.attach { 1 } else { 2 })?;
        Ok(BTreeMap::from([(
            "output".into(),
            self.multiply(
                &inputs["table"],
                float_values(&weights),
                size_of_val(float_values(&weights)),
            )?,
        )]))
    }
}
#[async_trait]
impl StreamOperator for Matrix {
    // Only a derived cache of immutable job inputs is retained; it changes no rows.
    fn lifecycle(&self) -> StreamOperatorLifecycle {
        StreamOperatorLifecycle::Stateless {
            microbatch_invariant: true,
            deterministic: true,
            replay_safe: true,
        }
    }
    async fn process_data(
        &mut self,
        ingress: &str,
        batch: Batch,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        if ingress != "table" {
            return Err(invalid("ingress", "weights must be a latched static input"));
        }
        let copied = if self.weights.is_none() {
            let values = weights(
                context
                    .static_input("weights")
                    .ok_or_else(|| invalid("weights", "missing static input"))?,
                1,
            )?;
            let bytes = size_of_val(float_values(&values));
            self.weights = Some(values);
            bytes
        } else {
            0
        };
        output
            .emit(
                "output",
                self.multiply(&batch, float_values(self.weights.as_ref().unwrap()), copied)?,
            )
            .await
    }
    async fn on_watermark(
        &mut self,
        _: EventTime,
        _: &StreamOperatorContext<'_>,
        _: &mut dyn StreamCollector,
    ) -> Result<()> {
        Ok(())
    }
    async fn on_end(
        &mut self,
        _: &StreamOperatorContext<'_>,
        _: &mut dyn StreamCollector,
    ) -> Result<()> {
        Ok(())
    }
}

pub fn project(attach: bool, streaming: bool) -> Result<calc_flow::ProjectSpec> {
    let columns = if attach {
        vec![
            serde_json::json!({"name":"return","data_type":"float64", "nullable":false}),
            serde_json::json!({"name":"volatility","data_type":"float64", "nullable":false}),
        ]
    } else {
        vec![
            serde_json::json!({"name":"quantity","data_type":"float64", "nullable":false}),
            serde_json::json!({"name":"unit_price","data_type":"float64", "nullable":false}),
        ]
    };
    let output_schema = if attach {
        let mut fields = columns.clone();
        fields
            .push(serde_json::json!({"name":"risk_score","data_type":"float64", "nullable":true}));
        fields
    } else {
        vec![]
    };
    let mut project = serde_json::json!({
        "format_version":3,"id":"rust-matrix","name":"rust-matrix","runtime":{"mode":if streaming {"stream"}else{"batch"},"options":{}},
        "graph":{"name":"rust-matrix","nodes":[{"id":"matmul","operator":{"kind":"external","provider":"rust-array","name":"matmul","version":"1","options":{"attach":attach}},
            "input_ports":[{"name":"table","kind":"table","schema":columns},{"name":"weights","kind":"array"}],
            "output_ports":[{"name":"output","kind":if attach {"table"}else{"array"},"schema":output_schema}]}]}
    });
    if !streaming {
        project["data_sources"] = serde_json::json!([{"id":"table","input":"table","format":"arrow_ipc","data":null},{"id":"weights","input":"weights","format":"arrow_ipc","data":null}]);
    }
    if streaming {
        project["static_inputs"] = serde_json::json!([{"name":"weights","kind":"array","mutability":"static","backend":"rust-array","dtype":"float64","shape":[2,1]}]);
    }
    serde_json::from_value(project).map_err(|e| invalid("project", e.to_string()))
}
