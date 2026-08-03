use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
};

use async_trait::async_trait;

use crate::{Batch, CalcFlowError, EventTime, JsonMap, Port, Result, UdfReference};

use super::{
    OperatorMetadata, StreamCollector, StreamOperator, StreamOperatorContext,
    validate_operator_name,
};

/// A multi-ingress forwarding operator (spec S4, API note A2.4).
///
/// Every ingress keeps its own FIFO; the operator forwards each batch to the
/// single `output` port unchanged. All input ports share one kind and one
/// exact schema (or are all schema-less). Stream-only: batch graphs compose
/// multi-input logic through SQL aliases.
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
