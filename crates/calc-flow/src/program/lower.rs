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

mod asof;
mod join;
mod projection;
mod union;
mod window;

pub(super) struct Lowered {
    name: String,
    fingerprint: String,
    mode: CompileMode,
    update_semantics: crate::UpdateSemantics,
    pub schemas: BTreeMap<String, SchemaRef>,
    orderings: BTreeMap<String, QueryOrderingFacts>,
    join_paths: BTreeSet<String>,
    nodes: Vec<NodeSpec>,
    edges: Vec<EdgeSpec>,
    declarations: Vec<Declaration>,
    diagnostics: Vec<crate::QueryDiagnostic>,
    declaration_edges: Vec<DeclarationEdge>,
    bindings: FrontendBindings,
    port_schemas: BTreeMap<String, PlannedPorts>,
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
pub(super) fn children(table: &TableExpr, path: &str) -> Vec<(String, TableExpr, String)> {
    match &table.0.kind {
        RelationKind::Order { input, sql, .. } => vec![(
            "input".into(),
            input.clone(),
            format!("{path}.{}.value", if *sql { "sql.order" } else { "sort" }),
        )],
        RelationKind::Distinct { input, sql, .. } => vec![(
            "input".into(),
            input.clone(),
            format!(
                "{path}.{}.value",
                if *sql { "sql.distinct" } else { "distinct" }
            ),
        )],
        RelationKind::AsofJoin { inputs, sql, .. } => inputs
            .iter()
            .enumerate()
            .map(|(i, input)| {
                let side = if i == 0 { "left" } else { "right" };
                (
                    side.into(),
                    input.clone(),
                    format!(
                        "{path}.{}.{side}",
                        if *sql { "sql.asof_join" } else { "asof_join" }
                    ),
                )
            })
            .collect(),
        RelationKind::InnerJoin { inputs, sql, .. } => inputs
            .iter()
            .enumerate()
            .map(|(i, input)| {
                let side = if i == 0 { "left" } else { "right" };
                (
                    side.into(),
                    input.clone(),
                    format!(
                        "{path}.{}.{side}",
                        if *sql { "sql.inner_join" } else { "inner_join" }
                    ),
                )
            })
            .collect(),
        RelationKind::Window { input, sql, .. } => vec![(
            "input".into(),
            input.clone(),
            format!(
                "{path}.{}.value",
                if *sql { "sql.window" } else { "window" }
            ),
        )],
        RelationKind::Union { inputs } => inputs
            .iter()
            .enumerate()
            .map(|(index, child)| {
                (
                    format!("input_{index}"),
                    child.clone(),
                    format!("{path}.union_all.inputs[{index}]"),
                )
            })
            .collect(),
        RelationKind::Input { .. } => vec![],
        RelationKind::Aggregate { input, .. } => vec![(
            "input".into(),
            input.clone(),
            format!("{path}.aggregate.value"),
        )],
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
        mode: program.mode,
        update_semantics: program.update_semantics,
        schemas: BTreeMap::new(),
        orderings: BTreeMap::new(),
        join_paths: join::sql_paths(&order),
        nodes: vec![],
        edges: vec![],
        declarations: vec![],
        diagnostics: vec![],
        declaration_edges: vec![],
        port_schemas: BTreeMap::new(),
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
        lowered
            .port_schemas
            .insert(id.clone(), PlannedPorts::identity(&schema));
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
        let other = if self.mode == CompileMode::Batch {
            "stream"
        } else {
            "batch"
        };
        if let Some(issue) = self.diagnostics.iter().find(|issue| {
            !self
                .declarations
                .iter()
                .any(|node| issue.path.starts_with(&format!("{}.{other}.", node.path)))
        }) {
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
        self.diagnostics.push(schema_diagnostic(table, path, error));
        let (kind, query) = relation_kind(table);
        self.declarations.push(Declaration {
            order: match &table.0.kind {
                RelationKind::Order {
                    by, fetch, spec, ..
                } => Some(crate::query_analysis::order::OrderDeclaration {
                    by: by.clone(),
                    fetch: *fetch,
                    spec: spec.clone(),
                }),
                _ => None,
            },
            distinct_spec: match &table.0.kind {
                RelationKind::Distinct { spec, .. } => Some(spec.clone()),
                _ => None,
            },
            path: path.into(),
            kind: kind.into(),
            query,
            schema: None,
            ordering: QueryOrderingFacts::default(),
            batch_support: Some(crate::SupportStatus::Unsupported),
            stream_support: Some(crate::SupportStatus::Unsupported),
        });
        for (port, child, role) in children(table, path) {
            self.declaration_edges.push(DeclarationEdge {
                path: role,
                source_path: paths[&child.0.id].clone(),
                target_path: path.into(),
                input_alias: Some(port),
            });
        }
    }
    async fn add_relation(
        &mut self,
        table: &TableExpr,
        path: &str,
        paths: &BTreeMap<String, String>,
    ) -> Result<()> {
        if !matches!(
            table.0.kind,
            RelationKind::Window { .. }
                | RelationKind::Input { .. }
                | RelationKind::Project { .. }
                | RelationKind::Filter { .. }
                | RelationKind::Sql { .. }
        ) && window::contains_window(table)
        {
            return Err(crate::dataframe::invalid(
                path,
                "capability_mismatch",
                "window outputs support only row-local transformations",
            ));
        }
        if self.add_sql_union(table, path, paths).await? {
            return Ok(());
        }
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
            order: match &table.0.kind {
                RelationKind::Order {
                    by, fetch, spec, ..
                } => Some(crate::query_analysis::order::OrderDeclaration {
                    by: by.clone(),
                    fetch: *fetch,
                    spec: spec.clone(),
                }),
                _ => None,
            },
            distinct_spec: match &table.0.kind {
                RelationKind::Distinct { spec, .. } => Some(spec.clone()),
                _ => None,
            },
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
        let ports = self.relation_ports(table, &schema);
        self.port_schemas.insert(node.id.clone(), ports);
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
                input_alias: Some(port),
            });
        }
        self.nodes.push(node);
        Ok(())
    }
    fn relation_ports(&self, table: &TableExpr, schema: &SchemaRef) -> PlannedPorts {
        let inputs = match &table.0.kind {
            RelationKind::Input { schema, .. } => vec![("input".into(), schema.clone())],
            _ => children(table, "")
                .into_iter()
                .map(|(name, child, _)| (name, self.schemas[&child.0.id].clone()))
                .collect(),
        };
        PlannedPorts {
            inputs,
            output: schema.clone(),
        }
    }
    async fn prepare(&self, table: &TableExpr, path: &str) -> Result<Prepared> {
        match &table.0.kind {
            RelationKind::Order {
                input,
                by,
                fetch,
                spec,
                ..
            } => self.prepare_order(table, input, by, *fetch, spec),
            RelationKind::Distinct {
                input,
                columns,
                spec,
                wildcard,
                ..
            } => self.prepare_distinct(table, input, columns, spec, *wildcard),
            RelationKind::AsofJoin { inputs, spec, sql } => self.prepare_asof(
                table,
                inputs,
                spec,
                &format!(
                    "{path}.{}",
                    if *sql { "sql.asof_join" } else { "asof_join" }
                ),
            ),
            RelationKind::InnerJoin {
                inputs,
                spec,
                ordering,
                sql,
            } => self.prepare_join(
                table,
                inputs,
                spec,
                ordering.as_ref(),
                &format!(
                    "{path}.{}",
                    if *sql { "sql.inner_join" } else { "inner_join" }
                ),
            ),
            RelationKind::Window { input, spec, sql } => self.prepare_window(
                table,
                input,
                spec,
                &format!("{path}.{}", if *sql { "sql.window" } else { "window" }),
            ),
            RelationKind::Union { inputs } => prepare_union(
                table,
                path,
                &self.schemas[&inputs[0].0.id],
                &self.schemas[&inputs[1].0.id],
            ),
            RelationKind::Input {
                schema, options, ..
            } => Ok(prepare_input(&node_id(table), schema, options)),
            RelationKind::Project { .. } => self.prepare_projection(table, path).await,
            RelationKind::Filter { .. } => self.prepare_filter(table, path).await,
            RelationKind::Aggregate {
                input,
                aggregates,
                group_by,
            } => {
                let query = crate::dataframe::aggregate::query(
                    input,
                    &self.schemas[&input.0.id],
                    aggregates,
                    group_by,
                    path,
                )?;
                let mut prepared = self
                    .prepare_sql_parts(
                        table,
                        &query,
                        &BTreeMap::from([("input".into(), input.clone())]),
                    )
                    .await?;
                prepared.kind = "aggregate";
                Ok(prepared)
            }
            RelationKind::Sql { .. } => self.prepare_sql(table).await,
        }
    }
    fn prepare_order(
        &self,
        table: &TableExpr,
        input: &TableExpr,
        by: &[crate::SortKey],
        fetch: Option<u64>,
        spec: &crate::SortSpec,
    ) -> Result<Prepared> {
        let schema = &self.schemas[&input.0.id];
        let op = crate::BoundedOrderOperator::new(
            &node_id(table),
            schema.clone(),
            by.to_vec(),
            fetch,
            spec.clone(),
        )?;
        let schema = crate::OperatorMetadata::output_ports(&op)[0]
            .schema()
            .unwrap()
            .clone();
        Ok(Prepared {
            node: NodeSpec {
                id: node_id(table),
                operator: OperatorSpec::BoundedOrder {
                    by: by.to_vec(),
                    fetch,
                    spec: spec.clone(),
                },
                input_ports: vec![port_spec("input", &schema)],
                output_ports: vec![port_spec("output", &schema)],
                position: None,
            },
            schema,
            kind: "bounded_order",
            query: None,
            ordering: QueryOrderingFacts::default(),
        })
    }
    fn prepare_distinct(
        &self,
        table: &TableExpr,
        input: &TableExpr,
        columns: &[String],
        spec: &crate::DistinctSpec,
        wildcard: bool,
    ) -> Result<Prepared> {
        let schema = &self.schemas[&input.0.id];
        let columns = if wildcard {
            schema.fields().iter().map(|f| f.name().clone()).collect()
        } else {
            columns.to_vec()
        };
        let operator = crate::DistinctOperator::new(
            &node_id(table),
            schema.clone(),
            columns.clone(),
            spec.clone(),
        )?;
        let output = crate::OperatorMetadata::output_ports(&operator)[0]
            .schema()
            .unwrap()
            .clone();
        Ok(Prepared {
            node: NodeSpec {
                id: node_id(table),
                operator: OperatorSpec::Distinct {
                    columns: columns.clone(),
                    spec: spec.clone(),
                },
                input_ports: vec![port_spec("input", schema)],
                output_ports: vec![port_spec("output", &output)],
                position: None,
            },
            schema: output,
            kind: "distinct",
            query: None,
            ordering: QueryOrderingFacts::default(),
        })
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
            ordering: self.projection_ordering(table, &schema),
            schema,
            kind: if *append { "with_columns" } else { "project" },
            query: Some(operator.query_text().to_owned()),
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
            query: Some(operator.query_text().to_owned()),
            ordering: self.orderings[&input.0.id].clone(),
        })
    }
    async fn prepare_sql(&self, table: &TableExpr) -> Result<Prepared> {
        let RelationKind::Sql { query, tables } = &table.0.kind else {
            unreachable!("dispatched SQL")
        };
        if window::contains_window(table)
            && (tables.len() != 1
                || crate::sql::sql_passthrough_columns(
                    query,
                    tables.first_key_value().expect("one input").0,
                )?
                .is_none())
        {
            return Err(crate::dataframe::invalid(
                "sql.window.value",
                "capability_mismatch",
                "window outputs require row-local SQL",
            ));
        }
        self.prepare_sql_parts(table, query, tables).await
    }
    async fn prepare_sql_parts(
        &self,
        table: &TableExpr,
        query: &str,
        tables: &BTreeMap<String, TableExpr>,
    ) -> Result<Prepared> {
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
                query: query.to_owned(),
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
            query: Some(query.to_owned()),
            ordering: self.join_sql_ordering(table, query, tables)?,
        })
    }
    pub fn analyze(&self, options: &AnalysisOptions) -> Result<QueryAnalysis> {
        // Reconstruct data-only declarations without retaining execution owners.
        let nodes = self
            .declarations
            .iter()
            .map(|node| Declaration {
                order: node.order.clone(),
                distinct_spec: node.distinct_spec.clone(),
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
                input_alias: edge.input_alias.clone(),
            })
            .collect();
        analyze_graph(DeclarationGraph {
            nodes,
            edges,
            options: options.clone(),
            diagnostics: self.diagnostics.clone(),
            update_semantics: self.update_semantics,
        })
    }
    pub fn gate(&self, options: &CompileOptions) -> Result<()> {
        self.ensure_valid()?;
        self.analyze(&AnalysisOptions::default())?
            .check_updates(self.mode)?;
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
        let mut builder = self.node_builder()?;
        for edge in &self.edges {
            builder = builder.connect(native_edge(edge)?)?;
        }
        builder.frontend_bindings = Some(self.bindings.clone());
        Ok(builder)
    }
    fn node_builder(&self) -> Result<PipelineBuilder> {
        let mut builder =
            PipelineBuilder::new(&self.name)?.with_update_semantics(self.update_semantics);
        for node in &self.nodes {
            let operator = node_operator(node, &self.port_schemas[&node.id])?;
            builder = builder.add_node(&node.id, operator)?;
        }
        Ok(builder)
    }
    pub fn project(&self, mode: CompileMode) -> Result<ProjectSpec> {
        self.ensure_valid()?;
        crate::pipeline::distinct::validate(&self.builder()?)?;
        crate::pipeline::order::validate(&self.builder()?)?;
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
                update_semantics: self.update_semantics,
            },
            data_sources,
            sources: vec![],
            sinks: vec![],
            state: crate::StateConfig::default(),
            static_inputs: vec![],
        })
    }
}
fn native_edge(edge: &EdgeSpec) -> Result<Edge> {
    Ok(Edge::new(
        PortEndpoint::new(&edge.source_node, &edge.source_port)?,
        PortEndpoint::new(&edge.target_node, &edge.target_port)?,
    ))
}
fn relation_kind(table: &TableExpr) -> (&'static str, Option<String>) {
    match &table.0.kind {
        RelationKind::Order { .. } => ("bounded_order", None),
        RelationKind::Distinct { .. } => ("distinct", None),
        RelationKind::AsofJoin { .. } => ("stream_asof_join", None),
        RelationKind::InnerJoin { .. } => ("stream_join", None),
        RelationKind::Window { .. } => ("window", None),
        RelationKind::Union { .. } => ("union", None),
        RelationKind::Input { .. } => ("table_input", None),
        RelationKind::Aggregate { .. } => ("aggregate", None),
        RelationKind::Project { append, .. } => {
            (if *append { "with_columns" } else { "project" }, None)
        }
        RelationKind::Filter { .. } => ("filter", None),
        RelationKind::Sql { query, .. } => ("sql", Some(query.clone())),
    }
}
fn schema_diagnostic(
    table: &TableExpr,
    path: &str,
    error: crate::CalcFlowError,
) -> crate::QueryDiagnostic {
    let (path, code, message) = match error {
        crate::CalcFlowError::InvalidArgument { field, message } => {
            argument_error(path, field, message)
        }
        crate::CalcFlowError::DataFusion { message, .. } => {
            native_schema_error(table, path, &message)
        }
        error => (path.into(), "schema_mismatch".into(), error.to_string()),
    };
    crate::QueryDiagnostic {
        path,
        code,
        message,
        status: crate::SupportStatus::Unsupported,
    }
}
fn argument_error(path: &str, field: String, message: String) -> (String, String, String) {
    if field == "sql.output.schema" {
        return (
            format!("{path}.sql.schema"),
            "duplicate_name".into(),
            message,
        );
    }
    let field = if field.starts_with("sql.") {
        format!("{path}.sql.schema")
    } else {
        field
    };
    let (code, message) = message
        .split_once(": ")
        .unwrap_or(("schema_mismatch", &message));
    (field, code.into(), message.into())
}
fn native_schema_error(table: &TableExpr, path: &str, message: &str) -> (String, String, String) {
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
        format!("{message}; location is the relation node; no subexpression span is available"),
    )
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
struct PlannedPorts {
    inputs: Vec<(String, SchemaRef)>,
    output: SchemaRef,
}
impl PlannedPorts {
    fn identity(schema: &SchemaRef) -> Self {
        Self {
            inputs: vec![("input".into(), schema.clone())],
            output: schema.clone(),
        }
    }
    fn inputs(&self) -> Result<Vec<Port>> {
        self.inputs
            .iter()
            .map(|(name, schema)| exact_port(name, schema))
            .collect()
    }
}
fn exact_port(name: &str, schema: &SchemaRef) -> Result<Port> {
    Port::with_schema_ref(name, BatchKind::Table, true, Some(schema.clone()))
}
fn node_operator(node: &NodeSpec, schemas: &PlannedPorts) -> Result<NodeOperator> {
    let inputs = schemas.inputs()?;
    let output = exact_port("output", &schemas.output)?;
    match &node.operator {
        OperatorSpec::BoundedOrder { by, fetch, spec } => Ok(crate::BoundedOrderOperator::new(
            &node.id,
            inputs[0].schema().unwrap().clone(),
            by.clone(),
            *fetch,
            spec.clone(),
        )?
        .into()),
        OperatorSpec::Distinct { columns, spec } => Ok(crate::DistinctOperator::new(
            &node.id,
            inputs[0].schema().unwrap().clone(),
            columns.clone(),
            spec.clone(),
        )?
        .into()),
        OperatorSpec::StreamAsofJoin { spec } => Ok(crate::StreamAsofJoinOperator::new(
            &node.id,
            inputs[0].schema().expect("exact left").clone(),
            inputs[1].schema().expect("exact right").clone(),
            spec.clone(),
        )?
        .into()),
        OperatorSpec::StreamJoin { spec } => Ok(crate::StreamJoinOperator::new(
            &node.id,
            inputs[0].schema().expect("exact left").clone(),
            inputs[1].schema().expect("exact right").clone(),
            spec.clone(),
        )?
        .into()),
        OperatorSpec::Window { spec } => Ok(crate::WindowAggregateOperator::new(
            &node.id,
            inputs[0].schema().expect("exact window input").clone(),
            spec.clone(),
        )?
        .into()),
        OperatorSpec::Union => Ok(NodeOperator::Union(crate::UnionOperator::new_frontend(
            &node.id, inputs,
        )?)),
        OperatorSpec::Expression { select, filter, .. } => {
            expression_operator(&node.id, select, filter.as_deref(), &inputs[0], output)
        }
        OperatorSpec::Sql { query, aliases, .. } => {
            sql_operator(&node.id, query, aliases, inputs, output)
        }
        _ => unreachable!("typed row-local frontend emits only expression and SQL nodes"),
    }
}
fn expression_operator(
    id: &str,
    select: &[String],
    filter: Option<&str>,
    input: &Port,
    output: Port,
) -> Result<NodeOperator> {
    Ok(Box::new(
        ExpressionOperator::new(id, "", select.to_vec(), filter.map(str::to_owned), vec![])?
            .with_ports(input.clone(), output)?,
    )
    .into())
}
fn sql_operator(
    id: &str,
    query: &str,
    aliases: &[String],
    inputs: Vec<Port>,
    output: Port,
) -> Result<NodeOperator> {
    Ok(
        Box::new(
            SqlOperator::new(id, query, aliases.to_vec(), vec![])?.with_ports(inputs, output)?,
        )
        .into(),
    )
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
fn prepare_union(
    table: &TableExpr,
    path: &str,
    first: &SchemaRef,
    second: &SchemaRef,
) -> Result<Prepared> {
    crate::dataframe::union_schema(first, second, &format!("{path}.union_all.inputs[1]"))?;
    let schema = crate::operator::normalized_union_schema(first);
    Ok(Prepared {
        node: NodeSpec {
            id: node_id(table),
            operator: OperatorSpec::Union,
            input_ports: vec![port_spec("input_0", first), port_spec("input_1", second)],
            output_ports: vec![port_spec("output", &schema)],
            position: None,
        },
        schema,
        kind: "union",
        query: None,
        ordering: QueryOrderingFacts::default(),
    })
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
