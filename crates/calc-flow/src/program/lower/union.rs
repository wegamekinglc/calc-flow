use super::{
    BTreeMap, Declaration, DeclarationEdge, Lowered, NodeSpec, OperatorSpec, PlannedPorts,
    QueryOrderingFacts, RelationKind, Result, TableExpr, digest, edge_spec, json, node_id,
    port_spec, schema_fields, segment, validate_schema,
};

impl Lowered {
    pub(super) async fn add_sql_union(
        &mut self,
        table: &TableExpr,
        path: &str,
        paths: &BTreeMap<String, String>,
    ) -> Result<bool> {
        let RelationKind::Sql { query, tables } = &table.0.kind else {
            return Ok(false);
        };
        let inputs = tables
            .iter()
            .map(|(alias, table)| (alias.clone(), self.schemas[&table.0.id].clone()))
            .collect();
        let Some(plan) = crate::sql::plan_union_all(query, &inputs).await? else {
            return Ok(false);
        };
        if let Some((index, message)) = &plan.planning_error {
            return Err(crate::dataframe::invalid(
                &format!("{path}.sql.arms[{index}]"),
                "schema_mismatch",
                message,
            ));
        }
        if let Some((index, suffix, message)) = &plan.mismatch {
            self.diagnostics.push(crate::QueryDiagnostic {
                path: format!(
                    "{}{suffix}",
                    segment(
                        &format!("{path}.stream.sql.tables"),
                        &plan.arms[*index].alias
                    )
                ),
                code: "schema_mismatch".into(),
                status: crate::SupportStatus::Unsupported,
                message: message.clone(),
            });
            return Ok(false);
        }
        validate_schema(&plan.arms[0].planned_schema, path)?;
        let schema = crate::operator::normalized_union_schema(&plan.arms[0].planned_schema);
        let mut union_ports = vec![];
        for (index, arm) in plan.arms.iter().enumerate() {
            let id = format!("cf_arm_{}", &digest(&json!([table.0.id, index]))[..56]);
            self.add_union_arm(&id, arm, &tables[&arm.alias], path, index, paths);
            let port = format!("input_{index}");
            self.edges.push(edge_spec(&id, &node_id(table), &port));
            self.declaration_edges.push(DeclarationEdge {
                path: segment(&format!("{path}.sql.tables"), &arm.alias),
                source_path: format!("{path}.sql.arms[{index}]"),
                target_path: path.into(),
                input_alias: Some(port.clone()),
            });
            union_ports.push((port, arm.planned_schema.clone()));
        }
        let ports = PlannedPorts {
            inputs: union_ports,
            output: schema.clone(),
        };
        self.nodes.push(NodeSpec {
            id: node_id(table),
            operator: OperatorSpec::Union,
            input_ports: ports
                .inputs
                .iter()
                .map(|(name, schema)| port_spec(name, schema))
                .collect(),
            output_ports: vec![port_spec("output", &schema)],
            position: None,
        });
        self.port_schemas.insert(node_id(table), ports);
        self.schemas.insert(table.0.id.clone(), schema.clone());
        self.orderings
            .insert(table.0.id.clone(), QueryOrderingFacts::default());
        self.declarations.push(Declaration {
            path: path.into(),
            kind: "union".into(),
            query: None,
            schema: Some(schema_fields(&schema)),
            ordering: QueryOrderingFacts::default(),
            batch_support: None,
            stream_support: None,
        });
        Ok(true)
    }

    fn add_union_arm(
        &mut self,
        id: &str,
        arm: &crate::sql::UnionArm,
        child: &TableExpr,
        path: &str,
        index: usize,
        paths: &BTreeMap<String, String>,
    ) {
        let input_schema = self.schemas[&child.0.id].clone();
        let arm_path = format!("{path}.sql.arms[{index}]");
        self.nodes.push(NodeSpec {
            id: id.into(),
            operator: OperatorSpec::Sql {
                query: arm.query.clone(),
                aliases: vec![arm.alias.clone()],
                udfs: vec![],
            },
            input_ports: vec![port_spec(&arm.alias, &input_schema)],
            output_ports: vec![port_spec("output", &arm.planned_schema)],
            position: None,
        });
        self.port_schemas.insert(
            id.into(),
            PlannedPorts {
                inputs: vec![(arm.alias.clone(), input_schema)],
                output: arm.planned_schema.clone(),
            },
        );
        self.edges.push(edge_spec(&node_id(child), id, &arm.alias));
        self.declarations.push(Declaration {
            path: arm_path.clone(),
            kind: "sql".into(),
            query: Some(arm.query.clone()),
            schema: Some(arm.schema.clone()),
            ordering: QueryOrderingFacts::default(),
            batch_support: None,
            stream_support: None,
        });
        self.declaration_edges.push(DeclarationEdge {
            path: format!("{arm_path}.input"),
            source_path: paths[&child.0.id].clone(),
            target_path: arm_path,
            input_alias: Some(arm.alias.clone()),
        });
    }
}
