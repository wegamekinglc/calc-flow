use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    sync::Arc,
};

use async_trait::async_trait;
use datafusion::arrow::{
    datatypes::{Schema, SchemaRef},
    record_batch::{RecordBatch, RecordBatchOptions},
};

use crate::{
    Batch, BatchKind, BatchMetadata, CalcFlowError, EventTime, JsonMap, Port, Result, UdfReference,
};

use super::{
    BatchOperator, BatchOperatorContext, OperatorMetadata, StreamCollector, StreamOperator,
    StreamOperatorContext, validate_operator_name,
};

/// A multi-ingress forwarding operator (spec S4, API note A2.4).
///
/// Every ingress keeps its own FIFO. The low-level constructor forwards batches
/// unchanged and requires uniform exact input schemas (or all schema-less).
/// Typed frontends normalize schema metadata at the union boundary while sharing
/// Arrow buffers and preserving envelope metadata. Finite execution requires
/// required table ports with an exact schema and appends only the current inputs.
pub struct UnionOperator {
    name: String,
    input_ports: Vec<Port>,
    output_ports: [Port; 1],
}

impl fmt::Debug for UnionOperator {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("UnionOperator")
            .field("name", &self.name)
            .field("input_ports", &self.input_ports)
            .field("output_ports", &self.output_ports)
            .finish()
    }
}

impl UnionOperator {
    pub(crate) fn new_frontend(name: &str, inputs: Vec<Port>) -> Result<Self> {
        let normalized = inputs
            .iter()
            .map(|port| {
                Port::with_schema_ref(
                    port.name(),
                    port.kind(),
                    port.required(),
                    port.schema().map(normalized_union_schema),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let mut operator = Self::new(name, normalized)?;
        operator.input_ports = inputs;
        Ok(operator)
    }

    fn normalize_record(&self, record: &RecordBatch) -> Result<RecordBatch> {
        let Some(schema) = self.output_ports[0].schema() else {
            return Ok(record.clone());
        };
        if record.schema_ref() == schema {
            return Ok(record.clone());
        }
        RecordBatch::try_new_with_options(
            schema.clone(),
            record.columns().to_vec(),
            &RecordBatchOptions::new().with_row_count(Some(record.num_rows())),
        )
        .map_err(|error| CalcFlowError::Operator {
            node_id: self.name.clone(),
            message: format!("union output schema: {error}"),
        })
    }

    fn normalize_output(&self, batch: Batch) -> Result<Batch> {
        let Some(schema) = self.output_ports[0].schema() else {
            return Ok(batch);
        };
        if batch.table_payload()?.schema() == schema {
            return Ok(batch);
        }
        let records = batch
            .table_payload()?
            .batches()
            .iter()
            .map(|record| self.normalize_record(record))
            .collect::<Result<Vec<_>>>()?;
        Batch::table(records, batch.metadata().clone())
    }

    fn push_record(
        &self,
        records: &mut Vec<RecordBatch>,
        rows: &mut usize,
        bytes: &mut usize,
        record: &RecordBatch,
        context: &BatchOperatorContext<'_>,
    ) -> Result<()> {
        context.run.check_cancelled()?;
        let record = self.normalize_record(record)?;
        crate::datafusion::push_bounded_sql_batch(
            records,
            rows,
            bytes,
            record,
            crate::datafusion::MAX_SQL_RESULT_ROWS,
            crate::datafusion::MAX_SQL_RESULT_BYTES,
        )
        .map_err(|error| self.result_error(error))
    }
    fn validate_finite_inputs(&self, inputs: &BTreeMap<String, Batch>) -> Result<()> {
        self.validate_batch()?;
        for port in &self.input_ports {
            let batch = inputs
                .get(port.name())
                .ok_or_else(|| CalcFlowError::Operator {
                    node_id: self.name.clone(),
                    message: format!("missing required input {}", port.name()),
                })?;
            port.validate(batch, &format!("{}.{}", self.name, port.name()))?;
        }
        Ok(())
    }

    fn finite_records(
        &self,
        inputs: &BTreeMap<String, Batch>,
        context: &BatchOperatorContext<'_>,
    ) -> Result<Vec<RecordBatch>> {
        let mut records = Vec::new();
        let mut rows = 0;
        let mut bytes = 0;
        for port in &self.input_ports {
            for record in inputs[port.name()].table_payload()?.batches() {
                self.push_record(&mut records, &mut rows, &mut bytes, record, context)?;
            }
        }
        if records.is_empty() {
            records.push(RecordBatch::new_empty(
                self.output_ports[0]
                    .schema()
                    .expect("validated table schema")
                    .clone(),
            ));
        }
        Ok(records)
    }

    fn result_error(&self, error: CalcFlowError) -> CalcFlowError {
        match error {
            CalcFlowError::InvalidArgument { message, .. } => CalcFlowError::InvalidArgument {
                field: format!("{}.output", self.name),
                message,
            },
            error => error,
        }
    }

    pub(crate) fn validate_batch(&self) -> Result<()> {
        for port in &self.input_ports {
            if port.kind() != BatchKind::Table || port.schema().is_none() || !port.required() {
                return Err(CalcFlowError::Compile {
                    message: format!(
                        "node {:?} input {:?}: unsupported_mode: batch union requires table + exact schema + required=true",
                        self.name,
                        port.name()
                    ),
                });
            }
        }
        Ok(())
    }
    /// Creates a union operator over at least two uniform input ports.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when the name is empty,
    /// fewer than two input ports are given, two ports share a name, or the
    /// ports differ in kind, schema, schema presence, or the required flag.
    pub fn new(name: &str, input_ports: Vec<Port>) -> Result<Self> {
        validate_operator_name(name)?;
        if input_ports.len() < 2 {
            return Err(CalcFlowError::InvalidArgument {
                field: "operator.input_ports".into(),
                message: "union operators require at least two input ports".into(),
            });
        }
        let mut unique = BTreeSet::new();
        if input_ports.iter().any(|port| !unique.insert(port.name())) {
            return Err(CalcFlowError::InvalidArgument {
                field: "operator.input_ports".into(),
                message: "union input port names must be unique".into(),
            });
        }
        let first = &input_ports[0];
        if input_ports.iter().any(|port| {
            port.kind() != first.kind()
                || port.schema() != first.schema()
                || port.required() != first.required()
        }) {
            return Err(CalcFlowError::InvalidArgument {
                field: "operator.input_ports".into(),
                message: "union input ports must share one kind, one schema, and one required flag"
                    .into(),
            });
        }
        let output = Port::with_schema_ref(
            "output",
            first.kind(),
            first.required(),
            first.schema().cloned(),
        );
        debug_assert!(output.is_ok(), "input ports already passed Port validation");
        let output = output?;
        Ok(Self {
            name: name.into(),
            input_ports,
            output_ports: [output],
        })
    }
}

#[async_trait]
impl BatchOperator for UnionOperator {
    async fn process(
        &mut self,
        inputs: &BTreeMap<String, Batch>,
        context: &BatchOperatorContext<'_>,
    ) -> Result<BTreeMap<String, Batch>> {
        context.run.check_cancelled()?;
        self.validate_finite_inputs(inputs)?;
        let records = self.finite_records(inputs, context)?;
        let batch = Batch::table(records, BatchMetadata::new(&self.name, 0, BTreeMap::new())?)?;
        Ok(BTreeMap::from([("output".into(), batch)]))
    }
}

impl OperatorMetadata for UnionOperator {
    fn name(&self) -> &str {
        &self.name
    }

    fn input_ports(&self) -> &[Port] {
        &self.input_ports
    }

    fn output_ports(&self) -> &[Port] {
        &self.output_ports
    }

    fn configuration(&self) -> JsonMap {
        BTreeMap::new()
    }

    fn udf_references(&self) -> Vec<UdfReference> {
        Vec::new()
    }
}

#[async_trait]
impl StreamOperator for UnionOperator {
    async fn process_data(
        &mut self,
        ingress: &str,
        batch: Batch,
        _context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        let port = self
            .input_ports
            .iter()
            .find(|port| port.name() == ingress)
            .ok_or_else(|| CalcFlowError::Operator {
                node_id: self.name.clone(),
                message: format!("unknown ingress {ingress:?}"),
            })?;
        port.validate(&batch, &format!("{}.{ingress}", self.name))?;
        let batch = self.normalize_output(batch)?;
        output.emit("output", batch).await
    }

    async fn on_watermark(
        &mut self,
        _watermark: EventTime,
        _context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_end(
        &mut self,
        _context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        Ok(())
    }
}

pub(crate) fn normalized_union_schema(schema: &SchemaRef) -> SchemaRef {
    Arc::new(Schema::new(
        schema
            .fields()
            .iter()
            .map(|field| {
                field
                    .as_ref()
                    .clone()
                    .with_metadata(std::collections::HashMap::new())
            })
            .collect::<Vec<_>>(),
    ))
}
