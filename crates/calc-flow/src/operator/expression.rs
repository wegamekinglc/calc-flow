use std::{collections::BTreeMap, fmt};

use async_trait::async_trait;
use serde_json::Value;

use crate::{
    Batch, CalcFlowError, DataFusionConfig, DataFusionRuntime, EventTime, JsonMap, Port, Result,
    RunContext, UdfReference, UdfRegistrySnapshot,
    expression::{sql_projection, validate_select_query},
};

use super::{
    BatchOperator, BatchOperatorContext, OperatorMetadata, StreamCollector, StreamOperator,
    StreamOperatorContext, StreamRuntimeState, table_port, udf_configuration,
    validate_builtin_port, validate_operator_name, validate_required_input,
};

mod projection;

use projection::ColumnProjection;

/// A `DataFusion` expression or projection operator.
///
/// Implements both [`BatchOperator`] and [`StreamOperator`] (API note A2.4):
/// one input batch produces zero or one output batch. The batch executor
/// drives it through the run-scoped `DataFusion` session (v2 invariant); the
/// stream task drives it through one lazy, operator-scoped session (plan
/// section 2.2).
pub struct ExpressionOperator {
    name: String,
    expression: Option<String>,
    select: Vec<String>,
    filter_expression: Option<String>,
    query: String,
    column_projection: Option<ColumnProjection>,
    udfs: Vec<UdfReference>,
    input_ports: [Port; 1],
    output_ports: [Port; 1],
    stream_state: StreamRuntimeState,
}

impl Clone for ExpressionOperator {
    /// A clone carries the declaration but not the built runtime; the lazy
    /// operator-scoped session is rebuilt on first use.
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            expression: self.expression.clone(),
            select: self.select.clone(),
            filter_expression: self.filter_expression.clone(),
            query: self.query.clone(),
            column_projection: self.column_projection.clone(),
            udfs: self.udfs.clone(),
            input_ports: self.input_ports.clone(),
            output_ports: self.output_ports.clone(),
            stream_state: self.stream_state.clone(),
        }
    }
}

impl fmt::Debug for ExpressionOperator {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExpressionOperator")
            .field("name", &self.name)
            .field("expression", &self.expression)
            .field("select", &self.select)
            .field("filter_expression", &self.filter_expression)
            .field("query", &self.query)
            .field("udfs", &self.udfs)
            .field("input_ports", &self.input_ports)
            .field("output_ports", &self.output_ports)
            .finish_non_exhaustive()
    }
}

impl ExpressionOperator {
    /// Creates a `DataFusion` expression or projection operator.
    ///
    /// A non-empty `expression` and a non-empty `select` list are the two
    /// calculation modes. Exactly one must be supplied.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when the name is empty,
    /// exactly one calculation mode is not supplied, a projection is empty,
    /// or the generated read-only query is malformed.
    pub fn new(
        name: &str,
        expression: &str,
        select: Vec<String>,
        filter_expression: Option<String>,
        udfs: Vec<UdfReference>,
    ) -> Result<Self> {
        validate_operator_name(name)?;
        let has_expression = !expression.trim().is_empty();
        let has_select = !select.is_empty();
        if has_expression == has_select {
            return Err(CalcFlowError::InvalidArgument {
                field: "operator.calculation".into(),
                message: "provide exactly one expression or non-empty select list".into(),
            });
        }
        if select.iter().any(|projection| projection.trim().is_empty()) {
            return Err(CalcFlowError::InvalidArgument {
                field: "operator.select".into(),
                message: "projection expressions must not be empty".into(),
            });
        }
        let expression = has_expression.then(|| expression.into());
        let query = expression_query(expression.as_deref(), &select, filter_expression.as_deref())?;
        let column_projection = (expression.is_none() && filter_expression.is_none())
            .then(|| ColumnProjection::parse(&select))
            .flatten();
        Ok(Self {
            name: name.into(),
            expression,
            select,
            filter_expression,
            query,
            column_projection,
            udfs,
            input_ports: [table_port("input")?],
            output_ports: [table_port("output")?],
            stream_state: StreamRuntimeState::new(),
        })
    }

    /// Returns this operator with exact configuration-defined table ports.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] unless the ports are the
    /// built-in `input` and `output` table boundaries.
    pub fn with_ports(mut self, input: Port, output: Port) -> Result<Self> {
        validate_builtin_port(&input, "input", "operator.input_ports")?;
        validate_required_input(&input, "operator.input_ports")?;
        validate_builtin_port(&output, "output", "operator.output_ports")?;
        self.input_ports = [input];
        self.output_ports = [output];
        Ok(self)
    }

    /// The normalized read-only query executed by this operator.
    pub(crate) fn query_text(&self) -> &str {
        &self.query
    }

    /// Attaches the plan's `DataFusion` resources for the stream path.
    pub(crate) fn set_stream_resources(
        &mut self,
        config: DataFusionConfig,
        udfs: UdfRegistrySnapshot,
        selected_udfs: Vec<UdfReference>,
    ) {
        self.stream_state.set_resources(config, udfs, selected_udfs);
    }

    pub(crate) const fn stream_runtime_initialized(&self) -> bool {
        self.stream_state.is_initialized()
    }

    #[doc(hidden)]
    pub(crate) async fn process_table(
        &mut self,
        inputs: &BTreeMap<String, Batch>,
        run: &RunContext,
        datafusion: &DataFusionRuntime,
    ) -> Result<BTreeMap<String, Batch>> {
        run.check_cancelled()?;
        let input = required_input(inputs, "input", self.name(), run.node_id())?;
        self.input_ports[0].validate(input, &format!("{}.input", self.name))?;
        let tables = BTreeMap::from([("input".into(), input.clone())]);
        let output = datafusion.sql(&self.query, &tables, run.node_id()).await?;
        run.check_cancelled()?;
        Ok(BTreeMap::from([("output".into(), output)]))
    }

    async fn process_standalone(&mut self, batch: Batch) -> Result<Batch> {
        self.input_ports[0].validate(&batch, &format!("{}.input", self.name))?;
        if let Some(projection) = &self.column_projection
            && let Some(projected) = projection.apply(&batch, &self.name)?
        {
            return Ok(projected);
        }
        let tables = BTreeMap::from([("input".into(), batch)]);
        let runtime = self.stream_state.runtime()?;
        runtime.sql(&self.query, &tables, Some(&self.name)).await
    }
}

impl OperatorMetadata for ExpressionOperator {
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
        BTreeMap::from([
            (
                "expression".into(),
                self.expression
                    .as_ref()
                    .map_or(Value::Null, |value| Value::String(value.clone())),
            ),
            (
                "filter_expression".into(),
                self.filter_expression
                    .as_ref()
                    .map_or(Value::Null, |value| Value::String(value.clone())),
            ),
            (
                "select".into(),
                Value::Array(self.select.iter().cloned().map(Value::String).collect()),
            ),
            (
                "udfs".into(),
                Value::Array(self.udfs.iter().map(udf_configuration).collect()),
            ),
        ])
    }

    fn udf_references(&self) -> Vec<UdfReference> {
        self.udfs.clone()
    }
}

#[async_trait]
impl BatchOperator for ExpressionOperator {
    /// Standalone batch processing through the operator-scoped session. The
    /// batch executor instead drives this operator through the run-scoped
    /// session (v2 invariant), so this trait path is for direct use.
    async fn process(
        &mut self,
        inputs: &BTreeMap<String, Batch>,
        _context: &BatchOperatorContext<'_>,
    ) -> Result<BTreeMap<String, Batch>> {
        let input = required_input(inputs, "input", self.name(), None)?;
        let output = self.process_standalone(input.clone()).await?;
        Ok(BTreeMap::from([("output".into(), output)]))
    }
}

#[async_trait]
impl StreamOperator for ExpressionOperator {
    async fn process_data(
        &mut self,
        ingress: &str,
        batch: Batch,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        if ingress != "input" {
            return Err(CalcFlowError::Operator {
                node_id: self.name.clone(),
                message: format!("unknown ingress {ingress:?}; expected \"input\""),
            });
        }
        context.check_cancelled()?;
        let produced = self.process_standalone(batch).await?;
        context.check_cancelled()?;
        output.emit("output", produced).await
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

pub(crate) fn expression_query(
    expression: Option<&str>,
    select: &[String],
    filter_expression: Option<&str>,
) -> Result<String> {
    let mut query = if let Some(expression) = expression {
        sql_projection(expression, "input")?
    } else {
        format!("SELECT {} FROM input", select.join(", "))
    };
    if let Some(filter) = filter_expression {
        query.push_str(" WHERE (");
        query.push_str(filter);
        query.push(')');
    }
    validate_select_query(&query)
}

pub(crate) fn required_input<'a>(
    inputs: &'a BTreeMap<String, Batch>,
    input: &str,
    operator: &str,
    node_id: Option<&str>,
) -> Result<&'a Batch> {
    inputs.get(input).ok_or_else(|| CalcFlowError::Operator {
        node_id: node_id.unwrap_or(operator).into(),
        message: format!("missing required input {input}"),
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::arrow::{
        array::{ArrayRef, Float64Array, StringArray},
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };

    use super::*;
    use crate::BatchMetadata;

    fn projection_input() -> Batch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("Price", DataType::Float64, true),
            Field::new("symbol", DataType::Utf8, false),
        ]));
        let record = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![Some(10.0), None])) as ArrayRef,
                Arc::new(StringArray::from(vec!["A", "B"])) as ArrayRef,
            ],
        )
        .unwrap();
        Batch::table(
            vec![record],
            BatchMetadata::new("quotes", 7, BTreeMap::new()).unwrap(),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn stream_column_projection_reuses_arrow_buffers_without_sql_runtime() {
        let input = projection_input();
        let mut operator = ExpressionOperator::new(
            "projection",
            "",
            vec![r#""symbol""#.into(), r#""Price" AS "last""#.into()],
            None,
            vec![],
        )
        .unwrap();
        let output = operator.process_standalone(input.clone()).await.unwrap();
        let record = &output.table_payload().unwrap().batches()[0];
        let original = &input.table_payload().unwrap().batches()[0];
        assert_eq!(record.schema().field(0).name(), "symbol");
        assert_eq!(record.schema().field(1).name(), "last");
        assert!(record.schema().field(1).is_nullable());
        assert_eq!(record.column(1).null_count(), 1);
        assert_eq!(output.metadata(), input.metadata());
        assert!(Arc::ptr_eq(record.column(0), original.column(1)));
        assert!(Arc::ptr_eq(record.column(1), original.column(0)));
        assert!(!operator.stream_runtime_initialized());
    }

    #[tokio::test]
    async fn stream_identity_projection_keeps_empty_batches_and_exact_schema() {
        let input = projection_input();
        let schema = input.table_payload().unwrap().schema().clone();
        let empty = Batch::table(
            vec![RecordBatch::new_empty(schema.clone())],
            input.metadata().clone(),
        )
        .unwrap();
        let mut operator = ExpressionOperator::new(
            "projection",
            "",
            vec![r#""Price""#.into(), r#""symbol""#.into()],
            None,
            vec![],
        )
        .unwrap();
        let output = operator.process_standalone(empty).await.unwrap();
        assert_eq!(output.num_rows(), 0);
        assert_eq!(output.table_payload().unwrap().schema(), &schema);
        assert!(!operator.stream_runtime_initialized());
    }

    #[tokio::test]
    async fn stream_computed_and_filtered_projections_keep_sql_semantics() {
        for (select, filter) in [
            (vec![r#""Price" + 1 AS next"#.into()], None),
            (vec![r#""Price""#.into()], Some(r#""Price" > 5"#.into())),
        ] {
            let mut operator =
                ExpressionOperator::new("projection", "", select, filter, vec![]).unwrap();
            let result = operator
                .process_standalone(projection_input())
                .await
                .unwrap();
            assert!(result.num_rows() > 0);
            assert!(operator.stream_runtime_initialized());
        }
    }

    #[tokio::test]
    async fn stream_projection_schema_metadata_matches_the_sql_reference() {
        let input = projection_input();
        let record = &input.table_payload().unwrap().batches()[0];
        let fields = record
            .schema()
            .fields()
            .iter()
            .map(|field| {
                field
                    .as_ref()
                    .clone()
                    .with_metadata(std::collections::HashMap::from([(
                        "unit".into(),
                        "test".into(),
                    )]))
            })
            .collect::<Vec<_>>();
        let schema = Arc::new(Schema::new_with_metadata(
            fields,
            std::collections::HashMap::from([("origin".into(), "quotes".into())]),
        ));
        let input = Batch::table(
            vec![RecordBatch::try_new(schema, record.columns().to_vec()).unwrap()],
            input.metadata().clone(),
        )
        .unwrap();
        for select in [
            vec![
                r#"input."Price" AS "quoted alias""#.into(),
                "SYMBOL AS ticker".into(),
            ],
            vec![r#""Price""#.into(), "symbol".into()],
        ] {
            let mut fast = ExpressionOperator::new("projection", "", select, None, vec![]).unwrap();
            let mut reference = fast.clone();
            reference.column_projection = None;
            let actual = fast.process_standalone(input.clone()).await.unwrap();
            let expected = reference.process_standalone(input.clone()).await.unwrap();
            assert_eq!(
                actual.table_payload().unwrap().schema(),
                expected.table_payload().unwrap().schema()
            );
            assert_eq!(
                actual.table_payload().unwrap().batches(),
                expected.table_payload().unwrap().batches()
            );
            assert_eq!(actual.metadata(), expected.metadata());
            assert!(!fast.stream_runtime_initialized());
        }
    }

    #[tokio::test]
    async fn stream_projection_invalid_names_keep_sql_errors() {
        for select in [
            vec!["missing".into()],
            vec!["symbol AS dup".into(), r#""Price" AS dup"#.into()],
        ] {
            let mut fast = ExpressionOperator::new("projection", "", select, None, vec![]).unwrap();
            let mut reference = fast.clone();
            reference.column_projection = None;
            let actual = fast
                .process_standalone(projection_input())
                .await
                .unwrap_err();
            let expected = reference
                .process_standalone(projection_input())
                .await
                .unwrap_err();
            assert_eq!(actual.to_string(), expected.to_string());
            assert!(fast.stream_runtime_initialized());
        }
    }
}
