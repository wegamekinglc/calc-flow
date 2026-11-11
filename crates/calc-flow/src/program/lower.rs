use super::{CompileMode, CompileOptions, Program};
use crate::{
    AnalysisOptions, BatchKind, DataSourceSpec, Edge, EdgeSpec, ExpressionOperator, NodeOperator,
    NodeSpec, OperatorSpec, PipelineBuilder, PipelineSpec, Port, PortEndpoint, PortSpec,
    ProjectSpec, QueryAnalysis, QueryOrderingFacts, QueryTemporalIdentity, Result, RuntimeSpec,
    SqlOperator, TableExpr,
    dataframe::{RelationKind, digest, projection, quote},
    pipeline::FrontendBindings,
    query_analysis::{
        Declaration, DeclarationEdge, DeclarationGraph, analyze_graph, schema_fields,
    },
};
use datafusion::arrow::datatypes::SchemaRef;
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};

pub(super) struct Lowered {
    name: String,
    fingerprint: String,
    pub schemas: BTreeMap<String, SchemaRef>,
    orderings: BTreeMap<String, QueryOrderingFacts>,
    nodes: Vec<NodeSpec>,
    edges: Vec<EdgeSpec>,
    declarations: Vec<Declaration>,
    diagnostics: Vec<crate::QueryDiagnostic>,
    declaration_edges: Vec<DeclarationEdge>,
    bindings: FrontendBindings,
}

pub(super) use crate::dataframe::segment;
fn node_id(table: &TableExpr) -> String {
    format!("cf_rel_{}", &table.0.id[..56])
}
fn endpoint(table: &TableExpr, port: &str) -> PortEndpoint {
    PortEndpoint {
        node_id: node_id(table),
        port: port.into(),
    }
}
fn children(table: &TableExpr, path: &str) -> Vec<(String, TableExpr, String)> {
    match &table.0.kind {
        RelationKind::Input { .. } => vec![],
        RelationKind::Filter { input, .. } => vec![(
            "input".into(),
            input.clone(),
            format!("{path}.filter.value"),
        )],
        RelationKind::Project { input, .. } => vec![(
            "input".into(),
            input.clone(),
            format!("{path}.project.value"),
        )],
        RelationKind::Sql { tables, .. } => tables
            .iter()
            .map(|(alias, table)| {
                (
                    alias.clone(),
                    table.clone(),
                    segment(&format!("{path}.sql.tables"), alias),
                )
            })
            .collect(),
    }
}
fn visit(
    table: &TableExpr,
    path: String,
    visited: &mut BTreeSet<String>,
    order: &mut Vec<(TableExpr, String)>,
) {
    if !visited.insert(table.0.id.clone()) {
        return;
    }
    let path = match &table.0.kind {
        RelationKind::Input { name, .. } => segment("inputs", name),
        _ => path,
    };
    for (_, child, path) in children(table, &path) {
        visit(&child, path, visited, order);
    }
    order.push((table.clone(), path));
}

pub(super) async fn lower(program: &Program) -> Result<Lowered> {
    let mut order = vec![];
    let mut visited = BTreeSet::new();
    for (name, table) in &program.outputs {
        visit(table, segment("outputs", name), &mut visited, &mut order);
    }
    let paths: BTreeMap<_, _> = order
        .iter()
        .map(|(table, path)| (table.0.id.clone(), path.clone()))
        .collect();
    let mut lowered = Lowered {
        name: program.name.clone(),
        fingerprint: program.fingerprint.clone(),
        schemas: BTreeMap::new(),
        orderings: BTreeMap::new(),
        nodes: vec![],
        edges: vec![],
        declarations: vec![],
        diagnostics: vec![],
        declaration_edges: vec![],
        bindings: FrontendBindings {
            inputs: BTreeMap::new(),
            outputs: BTreeMap::new(),
        },
    };
    for (table, path) in order {
        let result = if children(&table, &path)
            .iter()
            .all(|(_, child, _)| lowered.schemas.contains_key(&child.0.id))
        {
            lowered.add_relation(&table, &path, &paths).await
        } else {
            Err(crate::dataframe::invalid(
                &path,
                "unresolved_type",
                "upstream relation schema could not be resolved",
            ))
        };
        if let Err(error) = result {
            lowered.add_error(&table, &path, &paths, error);
        }
    }
    for (name, table) in &program.outputs {
        let id = format!("cf_out_{}", &digest(&json!([name, table.0.id]))[..56]);
        let Some(schema) = lowered.schemas.get(&table.0.id).cloned() else {
            continue;
        };
        lowered.nodes.push(expression_node(
            &id,
            identity_select(&schema),
            None,
            &schema,
            &schema,
        ));
        lowered.edges.push(edge_spec(&node_id(table), &id, "input"));
        lowered.bindings.outputs.insert(
            name.clone(),
            PortEndpoint {
                node_id: id,
                port: "output".into(),
            },
        );
    }
    Ok(lowered)
}
impl Lowered {
    pub fn ensure_valid(&self) -> Result<()> {
        if let Some(issue) = self.diagnostics.first() {
            return Err(crate::CalcFlowError::Compile {
                message: format!("{}: {}: {}", issue.path, issue.code, issue.message),
            });
        }
        Ok(())
    }
    fn add_error(
        &mut self,
        table: &TableExpr,
        path: &str,
        paths: &BTreeMap<String, String>,
        error: crate::CalcFlowError,
    ) {
        let (error_path, code, message) = match error {
            crate::CalcFlowError::InvalidArgument { field, message }
                if field == "sql.output.schema" =>
            {
                (
                    format!("{path}.sql.schema"),
                    "duplicate_name".into(),
                    message,
                )
            }
            crate::CalcFlowError::InvalidArgument { field, message } => {
                let field = if field.starts_with("sql.") {
                    format!("{path}.sql.schema")
                } else {
                    field
                };
                let (code, message) = message
                    .split_once(": ")
                    .unwrap_or(("schema_mismatch", &message));
                (field, code.to_owned(), message.to_owned())
            }
            crate::CalcFlowError::DataFusion { message, .. } => {
                let code = if message.contains("Projections require unique expression names") {
                    "duplicate_name"
                } else {
                    "schema_mismatch"
                };
                let role = if matches!(table.0.kind, RelationKind::Sql { .. }) {
                    "sql.schema"
                } else {
                    "schema"
                };
                (
                    format!("{path}.{role}"),
                    code.into(),
                    format!(
                        "{message}; location is the relation node; no subexpression span is available"
                    ),
                )
            }
            error => (path.into(), "schema_mismatch".into(), error.to_string()),
        };
        self.diagnostics.push(crate::QueryDiagnostic {
            path: error_path,
            code,
            status: crate::SupportStatus::Unsupported,
            message,
        });
        let (kind, query) = match &table.0.kind {
            RelationKind::Input { .. } => ("table_input", None),
            RelationKind::Project { append, .. } => {
                (if *append { "with_columns" } else { "project" }, None)
            }
            RelationKind::Filter { .. } => ("filter", None),
            RelationKind::Sql { query, .. } => ("sql", Some(query.clone())),
        };
        self.declarations.push(Declaration {
            path: path.into(),
            kind: kind.into(),
            query,
            schema: None,
            ordering: QueryOrderingFacts::default(),
            batch_support: Some(crate::SupportStatus::Unsupported),
            stream_support: Some(crate::SupportStatus::Unsupported),
        });
        for (_, child, role) in children(table, path) {
            self.declaration_edges.push(DeclarationEdge {
                path: role,
                source_path: paths[&child.0.id].clone(),
                target_path: path.into(),
            });
        }
    }
    async fn add_relation(
        &mut self,
        table: &TableExpr,
        path: &str,
        paths: &BTreeMap<String, String>,
    ) -> Result<()> {
        let Prepared {
            node,
            schema,
            kind,
            query,
            ordering,
        } = self.prepare(table, path).await?;
        validate_schema(&schema, path)?;
        self.orderings.insert(table.0.id.clone(), ordering.clone());
        self.declarations.push(Declaration {
            path: path.into(),
            kind: kind.into(),
            query,
            schema: Some(schema_fields(&schema)),
            ordering,
            batch_support: None,
            stream_support:
                matches!(&table.0.kind, RelationKind::Sql { tables, .. } if tables.len() != 1)
                    .then_some(crate::SupportStatus::Unsupported),
        });
        self.schemas.insert(table.0.id.clone(), schema);
        if let RelationKind::Input { name, .. } = &table.0.kind {
            self.bindings
                .inputs
                .insert(name.clone(), endpoint(table, "input"));
        }
        for (port, child, role) in children(table, path) {
            self.edges
                .push(edge_spec(&node_id(&child), &node.id, &port));
            self.declaration_edges.push(DeclarationEdge {
                path: role,
                source_path: paths[&child.0.id].clone(),
                target_path: path.into(),
            });
        }
        self.nodes.push(node);
        Ok(())
    }
    async fn prepare(&self, table: &TableExpr, path: &str) -> Result<Prepared> {
        match &table.0.kind {
            RelationKind::Input {
                schema, options, ..
            } => Ok(prepare_input(&node_id(table), schema, options)),
            RelationKind::Project { .. } => self.prepare_projection(table, path).await,
            RelationKind::Filter { .. } => self.prepare_filter(table, path).await,
            RelationKind::Sql { .. } => self.prepare_sql(table).await,
        }
    }
    async fn prepare_projection(&self, table: &TableExpr, path: &str) -> Result<Prepared> {
        let RelationKind::Project {
            input,
            columns,
            append,
        } = &table.0.kind
        else {
            unreachable!("dispatched projection")
        };
        let id = node_id(table);
        let input_schema = &self.schemas[&input.0.id];
        let select = projection(input, input_schema, columns, *append, path)?;
        let operator = ExpressionOperator::new(&id, "", select.clone(), None, vec![])?;
        let schema = operator.infer_stream_schema(input_schema.clone()).await?;
        Ok(Prepared {
            node: expression_node(&id, select, None, input_schema, &schema),
            ordering: retained_ordering(&self.orderings[&input.0.id], &schema),
            schema,
            kind: if *append { "with_columns" } else { "project" },
            query: None,
        })
    }
    async fn prepare_filter(&self, table: &TableExpr, path: &str) -> Result<Prepared> {
        let RelationKind::Filter { input, predicate } = &table.0.kind else {
            unreachable!("dispatched filter")
        };
        let id = node_id(table);
        let input_schema = &self.schemas[&input.0.id];
        let role = format!("{path}.filter.predicate");
        let dtype = predicate.dtype(input.lineage(), input_schema, &role)?;
        if dtype != datafusion::arrow::datatypes::DataType::Boolean {
            return Err(crate::dataframe::invalid(
                &role,
                "unsupported_type",
                format!(
                    "filter predicate requires bool, found {}",
                    dtype.to_string().to_lowercase()
                ),
            ));
        }
        let select = identity_select(input_schema);
        let filter = Some(predicate.render());
        let operator = ExpressionOperator::new(&id, "", select.clone(), filter.clone(), vec![])?;
        let schema = operator.infer_stream_schema(input_schema.clone()).await?;
        Ok(Prepared {
            node: expression_node(&id, select, filter, input_schema, &schema),
            schema,
            kind: "filter",
            query: None,
            ordering: self.orderings[&input.0.id].clone(),
        })
    }
    async fn prepare_sql(&self, table: &TableExpr) -> Result<Prepared> {
        let RelationKind::Sql { query, tables } = &table.0.kind else {
            unreachable!("dispatched SQL")
        };
        let id = node_id(table);
        let inputs: BTreeMap<_, _> = tables
            .iter()
            .map(|(alias, table)| (alias.clone(), self.schemas[&table.0.id].clone()))
            .collect();
        let operator = SqlOperator::new(&id, query, inputs.keys().cloned().collect(), vec![])?;
        let schema = operator.infer_schema(&inputs).await?;
        let node = NodeSpec {
            id,
            operator: OperatorSpec::Sql {
                query: query.clone(),
                aliases: inputs.keys().cloned().collect(),
                udfs: vec![],
            },
            input_ports: inputs.iter().map(|(name, s)| port_spec(name, s)).collect(),
            output_ports: vec![port_spec("output", &schema)],
            position: None,
        };
        Ok(Prepared {
            node,
            schema,
            kind: "sql",
            query: Some(query.clone()),
            ordering: QueryOrderingFacts::default(),
        })
    }
    pub fn analyze(&self, options: &AnalysisOptions) -> Result<QueryAnalysis> {
        // Reconstruct data-only declarations without retaining execution owners.
        let nodes = self
            .declarations
            .iter()
            .map(|node| Declaration {
                path: node.path.clone(),
                kind: node.kind.clone(),
                query: node.query.clone(),
                schema: node.schema.clone(),
                ordering: node.ordering.clone(),
                batch_support: node.batch_support,
                stream_support: node.stream_support,
            })
            .collect();
        let edges = self
            .declaration_edges
            .iter()
            .map(|edge| DeclarationEdge {
                path: edge.path.clone(),
                source_path: edge.source_path.clone(),
                target_path: edge.target_path.clone(),
            })
            .collect();
        analyze_graph(DeclarationGraph {
            nodes,
            edges,
            options: options.clone(),
            diagnostics: self.diagnostics.clone(),
        })
    }
    pub fn gate(&self, options: &CompileOptions) -> Result<()> {
        if let Some(request) = &options.portability {
            self.analyze(&AnalysisOptions {
                portability: Some(request.clone()),
            })?
            .check()?;
        }
        self.ensure_valid()
    }
    pub fn builder(&self) -> Result<PipelineBuilder> {
        self.ensure_valid()?;
        let mut builder = PipelineBuilder::new(&self.name)?;
        for node in &self.nodes {
            builder = builder.add_node(&node.id, node_operator(node)?)?;
        }
        for edge in &self.edges {
            builder = builder.connect(Edge::new(
                PortEndpoint::new(&edge.source_node, &edge.source_port)?,
                PortEndpoint::new(&edge.target_node, &edge.target_port)?,
            ))?;
        }
        builder.frontend_bindings = Some(self.bindings.clone());
        Ok(builder)
    }
    pub fn project(&self, mode: CompileMode) -> Result<ProjectSpec> {
        self.ensure_valid()?;
        let data_sources = if mode == CompileMode::Batch {
            crate::pipeline::external_names(self.bindings.inputs.values().cloned().collect())
                .keys()
                .enumerate()
                .map(|(index, name)| DataSourceSpec {
                    id: format!("source_{index}"),
                    input: name.clone(),
                    format: "inline_json".into(),
                    data: json!([]),
                })
                .collect()
        } else {
            vec![]
        };
        Ok(ProjectSpec {
            format_version: crate::PROJECT_FORMAT_VERSION,
            id: format!("cf_program_{}", &self.fingerprint[..52]),
            name: self.name.clone(),
            description: String::new(),
            runtime: match mode {
                CompileMode::Batch => RuntimeSpec::Batch(crate::RunOptions::default()),
                CompileMode::Stream => RuntimeSpec::Stream(crate::StreamRunOptions::default()),
            },
            graph: PipelineSpec {
                name: self.name.clone(),
                nodes: self.nodes.clone(),
                edges: self.edges.clone(),
                datafusion: crate::DataFusionConfig::default(),
            },
            data_sources,
            sources: vec![],
            sinks: vec![],
            state: crate::StateConfig::default(),
            static_inputs: vec![],
        })
    }
}
fn port_spec(name: &str, schema: &SchemaRef) -> PortSpec {
    PortSpec {
        name: name.into(),
        kind: BatchKind::Table,
        required: true,
        schema: schema_fields(schema),
    }
}
fn expression_node(
    id: &str,
    select: Vec<String>,
    filter: Option<String>,
    input: &SchemaRef,
    output: &SchemaRef,
) -> NodeSpec {
    NodeSpec {
        id: id.into(),
        operator: OperatorSpec::Expression {
            expression: String::new(),
            select,
            filter,
            udfs: vec![],
        },
        input_ports: vec![port_spec("input", input)],
        output_ports: vec![port_spec("output", output)],
        position: None,
    }
}
fn edge_spec(source: &str, target: &str, port: &str) -> EdgeSpec {
    EdgeSpec {
        source_node: source.into(),
        source_port: "output".into(),
        target_node: target.into(),
        target_port: port.into(),
    }
}
fn native_port(spec: &PortSpec) -> Result<Port> {
    let fields = spec
        .schema
        .iter()
        .map(crate::config::field_from_spec)
        .collect::<Result<Vec<_>>>()?;
    Port::new(&spec.name, spec.kind, spec.required, Some(fields))
}
fn node_operator(node: &NodeSpec) -> Result<NodeOperator> {
    let inputs = node
        .input_ports
        .iter()
        .map(native_port)
        .collect::<Result<Vec<_>>>()?;
    let output = native_port(&node.output_ports[0])?;
    match &node.operator {
        OperatorSpec::Expression { select, filter, .. } => Ok(Box::new(
            ExpressionOperator::new(&node.id, "", select.clone(), filter.clone(), vec![])?
                .with_ports(inputs[0].clone(), output)?,
        )
        .into()),
        OperatorSpec::Sql { query, aliases, .. } => Ok(Box::new(
            SqlOperator::new(&node.id, query, aliases.clone(), vec![])?
                .with_ports(inputs, output)?,
        )
        .into()),
        _ => unreachable!("typed row-local frontend emits only expression and SQL nodes"),
    }
}

fn retained_ordering(ordering: &QueryOrderingFacts, schema: &SchemaRef) -> QueryOrderingFacts {
    let mut ordering = ordering.clone();
    if ordering.temporal_identity.as_ref().is_some_and(|identity| {
        std::iter::once(&identity.event_time)
            .chain(&identity.entity_by)
            .chain(&identity.sequence_by)
            .any(|name| schema.field_with_name(name).is_err())
    }) {
        ordering.temporal_identity = None;
    }
    ordering
}
fn validate_schema(schema: &SchemaRef, path: &str) -> Result<()> {
    for (index, field) in schema.fields().iter().enumerate() {
        let path = format!("{path}.schema[{index}].data_type");
        if field.data_type() == &datafusion::arrow::datatypes::DataType::Null {
            return Err(crate::dataframe::invalid(
                &path,
                "unresolved_type",
                "NULL output requires an explicit cast",
            ));
        }
        crate::dataframe::portable_type(field.data_type(), &path)?;
    }
    Ok(())
}

struct Prepared {
    node: NodeSpec,
    schema: SchemaRef,
    kind: &'static str,
    query: Option<String>,
    ordering: QueryOrderingFacts,
}
fn identity_select(schema: &SchemaRef) -> Vec<String> {
    if schema.fields().is_empty() {
        vec!["*".into()]
    } else {
        schema
            .fields()
            .iter()
            .map(|field| quote(field.name()))
            .collect()
    }
}
fn prepare_input(id: &str, schema: &SchemaRef, options: &crate::InputOptions) -> Prepared {
    let ordering = QueryOrderingFacts {
        temporal_identity: options
            .event_time
            .as_ref()
            .map(|event_time| QueryTemporalIdentity {
                entity_by: options.entity_by.clone(),
                event_time: event_time.clone(),
                sequence_by: options.sequence_by.clone(),
            }),
        ..QueryOrderingFacts::default()
    };
    Prepared {
        node: expression_node(id, identity_select(schema), None, schema, schema),
        schema: schema.clone(),
        kind: "table_input",
        query: None,
        ordering,
    }
}
