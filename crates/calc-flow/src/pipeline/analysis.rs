//! Borrowed graph analysis; it never initializes execution state.

use std::collections::BTreeMap;

use datafusion::arrow::datatypes::SchemaRef;

use crate::{
    AnalysisOptions, BatchExecutionPlan, CalcFlowError, NodeOperator, PipelineBuilder,
    PortabilityRequest, QueryAnalysis, QueryDiagnostic, QueryOrderingFacts, Result,
    StreamExecutionPlan, StreamRequirements, SupportStatus, UdfRegistrySnapshot,
    query_analysis::{
        Declaration, DeclarationEdge, DeclarationGraph, analyze_graph, schema_fields,
    },
};

impl PipelineBuilder {
    /// Analyzes declaration history in both lifecycles without executing rows.
    ///
    /// # Errors
    /// Returns errors for invalid graphs or an unavailable analysis foundation.
    pub async fn analyze(
        &self,
        udfs: &UdfRegistrySnapshot,
        options: &AnalysisOptions,
    ) -> Result<QueryAnalysis> {
        let order = super::compile::validate_and_order(
            self,
            self.nodes
                .values()
                .any(|node| node.operator.requires_datafusion()),
        )?;
        super::compile::selected_udf_catalog(&self.nodes, udfs)?;
        let mut schemas = BTreeMap::new();
        let mut nodes = vec![];
        let mut diagnostics = vec![];
        for id in order {
            let op = &self.nodes[&id].operator;
            let inputs = input_schemas(self, &id, &schemas);
            let (node, schema) = describe_operator(&id, op, &inputs, &mut diagnostics).await?;
            if let Some(schema) = schema {
                schemas.insert(id, schema);
            }
            nodes.push(node);
        }
        let edges = self
            .edges
            .iter()
            .map(|edge| DeclarationEdge {
                path: format!(
                    "graph.edges[{}][{}][{}][{}]",
                    quoted(&edge.source.node_id),
                    quoted(&edge.source.port),
                    quoted(&edge.target.node_id),
                    quoted(&edge.target.port)
                ),
                source_path: node_path(&edge.source.node_id),
                target_path: node_path(&edge.target.node_id),
            })
            .collect();
        analyze_graph(DeclarationGraph {
            nodes,
            edges,
            options: options.clone(),
            diagnostics,
        })
    }

    /// Checks the current declaration before invoking the original batch compiler.
    ///
    /// The builder must carry enough schema metadata to prove the comparison.
    /// Reports are recomputed and never accepted as execution permits.
    ///
    /// ```no_run
    /// use calc_flow::{BatchExecutionPlan, Comparison, ComparisonCutoff, HistoryScope,
    ///     PipelineBuilder, PortabilityRequest, Result, UdfRegistrySnapshot};
    /// async fn compile(builder: PipelineBuilder, udfs: &UdfRegistrySnapshot)
    ///     -> Result<BatchExecutionPlan> {
    ///     let request = PortabilityRequest::new(Comparison::PerBatch,
    ///         HistoryScope::CurrentBatch, ComparisonCutoff::BatchBoundary)?;
    ///     builder.compile_batch_checked(udfs, &request).await
    /// }
    /// ```
    ///
    /// # Errors
    /// Returns `QueryPortability` unless the comparison is proved; retains compiler errors.
    pub async fn compile_batch_checked(
        self,
        udfs: &UdfRegistrySnapshot,
        request: &PortabilityRequest,
    ) -> Result<BatchExecutionPlan> {
        self.analyze(
            udfs,
            &AnalysisOptions {
                portability: Some(request.clone()),
            },
        )
        .await?
        .check()?;
        self.compile_batch(udfs)
    }

    /// Checks the current declaration before invoking the original stream compiler.
    ///
    /// # Errors
    /// Returns `QueryPortability` unless the comparison is proved; retains compiler errors.
    pub async fn compile_stream_checked(
        self,
        udfs: &UdfRegistrySnapshot,
        requirements: &StreamRequirements,
        request: &PortabilityRequest,
    ) -> Result<StreamExecutionPlan> {
        self.analyze(
            udfs,
            &AnalysisOptions {
                portability: Some(request.clone()),
            },
        )
        .await?
        .check()?;
        self.compile_stream(udfs, requirements)
    }
}

fn input_schemas(
    builder: &PipelineBuilder,
    id: &str,
    schemas: &BTreeMap<String, SchemaRef>,
) -> BTreeMap<String, SchemaRef> {
    builder.nodes[id]
        .operator
        .input_ports()
        .iter()
        .filter_map(|port| {
            let upstream = builder
                .edges
                .iter()
                .find(|edge| edge.target.node_id == id && edge.target.port == port.name())
                .and_then(|edge| schemas.get(&edge.source.node_id));
            upstream
                .or_else(|| port.schema())
                .map(|schema| (port.name().to_owned(), schema.clone()))
        })
        .collect()
}

fn quoted(value: &str) -> String {
    serde_json::to_string(value).expect("strings serialize")
}
fn node_path(id: &str) -> String {
    format!("graph.nodes[{}]", quoted(id))
}

async fn describe_operator(
    id: &str,
    op: &NodeOperator,
    inputs: &BTreeMap<String, SchemaRef>,
    diagnostics: &mut Vec<QueryDiagnostic>,
) -> Result<(Declaration, Option<SchemaRef>)> {
    let (kind, query) = operator_declaration(op);
    let schema = match plan_schema(op, inputs).await {
        Ok(schema) => schema,
        Err(
            CalcFlowError::DataFusion { .. }
            | CalcFlowError::InvalidArgument { .. }
            | CalcFlowError::Compile { .. },
        ) => {
            for mode in ["batch", "stream"] {
                diagnostics.push(QueryDiagnostic {
                    path: format!("{}.{mode}.schema", node_path(id)), code: "schema_mismatch".into(), status: SupportStatus::Unsupported,
                    message: "SQL/expression schema planning could not resolve the declared fields or types; location is the node".into(),
                });
            }
            None
        }
        Err(error) => return Err(error),
    };
    if !op.udf_references().is_empty() {
        diagnostics.push(QueryDiagnostic {
            path: format!("{}.stream.support", node_path(id)),
            code: "capability_mismatch".into(),
            status: SupportStatus::Unknown,
            message: "selected UDF history and comparison semantics are not proved".into(),
        });
    }
    let declaration = Declaration {
        path: node_path(id),
        kind: kind.into(),
        query,
        schema: schema.as_ref().map(schema_fields),
        ordering: QueryOrderingFacts::default(),
        batch_support: matches!(op, NodeOperator::Batch(_)).then_some(SupportStatus::Supported),
        stream_support: (matches!(op, NodeOperator::Batch(_))
            || matches!(op, NodeOperator::Sql(_)) && op.input_ports().len() != 1)
            .then_some(SupportStatus::Unsupported),
    };
    Ok((declaration, schema))
}

fn operator_declaration(op: &NodeOperator) -> (&'static str, Option<String>) {
    match op {
        NodeOperator::Sql(sql) => ("sql", Some(sql.query_text().to_owned())),
        NodeOperator::Expression(expression) => {
            ("expression", Some(expression.query_text().to_owned()))
        }
        NodeOperator::SnapshotSql(sql) => ("snapshot_sql", Some(sql.query_text().to_owned())),
        NodeOperator::Batch(_) => ("batch_only", None),
        _ => (stream_operator_kind(op), None),
    }
}

fn stream_operator_kind(op: &NodeOperator) -> &'static str {
    match op {
        NodeOperator::Union(_) => "union",
        NodeOperator::Window(_) => "window",
        NodeOperator::StreamJoin(_) => "stream_join",
        NodeOperator::StreamAsofJoin(_) => "stream_asof_join",
        NodeOperator::Rolling(_) => "rolling",
        NodeOperator::CrossSection(_) => "cross_section",
        _ => "stream_only",
    }
}

async fn plan_schema(
    op: &NodeOperator,
    inputs: &BTreeMap<String, SchemaRef>,
) -> Result<Option<SchemaRef>> {
    match op {
        NodeOperator::Sql(sql)
            if inputs.len() == op.input_ports().len() && op.udf_references().is_empty() =>
        {
            sql.infer_schema(inputs).await.map(Some)
        }
        NodeOperator::Expression(expression) if op.udf_references().is_empty() => {
            match inputs.get("input") {
                Some(schema) => expression
                    .infer_stream_schema(schema.clone())
                    .await
                    .map(Some),
                None => Ok(None),
            }
        }
        NodeOperator::Sql(_) | NodeOperator::Expression(_) => Ok(None),
        _ => Ok(op
            .output_ports()
            .first()
            .and_then(|port| port.schema())
            .cloned()),
    }
}
