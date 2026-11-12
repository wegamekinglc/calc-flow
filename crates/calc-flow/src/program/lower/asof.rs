use super::{Lowered, Prepared, node_id, port_spec};
use crate::{
    NodeSpec, OperatorMetadata, OperatorSpec, QueryOrderingFacts, QueryTemporalIdentity, Result,
    StreamAsofJoinOperator, StreamAsofJoinSpec, TableExpr, dataframe::invalid,
};

impl Lowered {
    pub(super) fn prepare_asof(
        &self,
        table: &TableExpr,
        inputs: &[TableExpr; 2],
        spec: &StreamAsofJoinSpec,
        path: &str,
    ) -> Result<Prepared> {
        validate_spec_schema(
            spec,
            &self.schemas[&inputs[0].0.id],
            &self.schemas[&inputs[1].0.id],
            path,
        )?;
        for (i, side) in [spec.left(), spec.right()].into_iter().enumerate() {
            let name = if i == 0 { "left" } else { "right" };
            self.validate_join_side(
                &inputs[i],
                &self.schemas[&inputs[i].0.id],
                name,
                side.event_time(),
                side.keys(),
                path,
            )?;
            let identity = self.orderings[&inputs[i].0.id]
                .temporal_identity
                .as_ref()
                .expect("validated join identity");
            if identity.event_time != side.event_time()
                || identity.sequence_by != side.sequence_by()
            {
                return Err(invalid(
                    &format!("{path}.spec.{name}"),
                    "ordering_required",
                    "ASOF time and sequence must match current temporal identity",
                ));
            }
        }
        let left = &self.schemas[&inputs[0].0.id];
        let right = &self.schemas[&inputs[1].0.id];
        let id = node_id(table);
        let operator = StreamAsofJoinOperator::new(&id, left.clone(), right.clone(), spec.clone())?;
        let schema = operator.output_ports()[0]
            .schema()
            .expect("exact ASOF output")
            .clone();
        let side = spec.left();
        let prefixed = |name: &str| format!("{}__{name}", side.prefix());
        Ok(Prepared {
            node: NodeSpec {
                id,
                operator: OperatorSpec::StreamAsofJoin { spec: spec.clone() },
                input_ports: vec![port_spec("left", left), port_spec("right", right)],
                output_ports: vec![port_spec("output", &schema)],
                position: None,
            },
            schema,
            kind: "stream_asof_join",
            query: None,
            ordering: QueryOrderingFacts {
                temporal_identity: Some(QueryTemporalIdentity {
                    entity_by: side.keys().iter().map(|n| prefixed(n)).collect(),
                    event_time: prefixed(side.event_time()),
                    sequence_by: side.sequence_by().iter().map(|n| prefixed(n)).collect(),
                }),
                ..Default::default()
            },
        })
    }
}

fn validate_spec_schema(
    spec: &StreamAsofJoinSpec,
    left: &datafusion::arrow::datatypes::SchemaRef,
    right: &datafusion::arrow::datatypes::SchemaRef,
    path: &str,
) -> Result<()> {
    if let Some(issue) = crate::operator::asof_schema_issues(spec, left, right).first() {
        return Err(invalid(
            &format!("{path}.spec.{}", issue.path),
            &issue.code,
            &issue.message,
        ));
    }
    Ok(())
}
