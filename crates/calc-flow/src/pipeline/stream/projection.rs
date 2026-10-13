//! Physical Join/ASOF output projections, after logical graph validation.

use std::{collections::BTreeSet, sync::Arc};

use crate::{NodeOperator, OperatorMetadata, PipelineBuilder, Result};

struct OutputProjection {
    node_id: String,
    columns: Vec<usize>,
    consumers: Vec<String>,
}

pub(super) fn push_output_projections(
    builder: &mut PipelineBuilder,
    table: Option<&super::TablePlanResources>,
) -> Result<()> {
    // Conversion registers the graph's selected UDFs into every expression
    // runtime. Node-local lists cannot prove the resolved function identity.
    let allow_join = table.is_none_or(|table| table.selected_udfs.is_empty());
    let projections = builder
        .nodes
        .iter()
        .filter_map(|(node_id, definition)| {
            output_projection(builder, node_id, &definition.operator, allow_join)
        })
        .collect::<Vec<_>>();
    for projection in projections {
        install_projection(builder, projection)?;
    }
    Ok(())
}

fn output_projection(
    builder: &PipelineBuilder,
    node_id: &str,
    operator: &NodeOperator,
    allow_join: bool,
) -> Option<OutputProjection> {
    let (schema, expressions) = projection_schema(operator, allow_join)?;
    let consumers = builder
        .edges
        .iter()
        .filter(|edge| edge.source.node_id == node_id && edge.source.port == "output")
        .map(|edge| projection_consumer(builder, edge, schema, expressions))
        .collect::<Option<Vec<_>>>()?;
    // No consumers means the full producer port is an external graph output.
    if consumers.is_empty() {
        return None;
    }
    let columns = consumers
        .iter()
        .flat_map(|(_, indices)| indices.iter().copied())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    if columns.len() == schema.fields().len() {
        return None;
    }
    Some(OutputProjection {
        node_id: node_id.into(),
        columns,
        consumers: consumers.into_iter().map(|(node_id, _)| node_id).collect(),
    })
}

fn projection_schema(
    operator: &NodeOperator,
    allow_join: bool,
) -> Option<(&datafusion::arrow::datatypes::SchemaRef, bool)> {
    match operator {
        NodeOperator::StreamAsofJoin(asof) => Some((asof.output_ports()[0].schema()?, false)),
        NodeOperator::StreamJoin(join) if allow_join => {
            Some((join.output_ports()[0].schema()?, true))
        }
        _ => None,
    }
}

fn projection_consumer(
    builder: &PipelineBuilder,
    edge: &crate::Edge,
    schema: &datafusion::arrow::datatypes::SchemaRef,
    expressions: bool,
) -> Option<(String, Vec<usize>)> {
    if edge.target.port != "input" {
        return None;
    }
    let NodeOperator::Expression(expression) = &builder.nodes[&edge.target.node_id].operator else {
        return None;
    };
    if expression.input_ports()[0].schema() != Some(schema) {
        return None;
    }
    let columns = if expressions {
        expression.stream_required_columns()?
    } else {
        expression.stream_projection_columns()?
    };
    Some((edge.target.node_id.clone(), columns))
}

fn install_projection(builder: &mut PipelineBuilder, projection: OutputProjection) -> Result<()> {
    let operator = &mut builder
        .nodes
        .get_mut(&projection.node_id)
        .expect("validated Join/ASOF node")
        .operator;
    let schema = match operator {
        NodeOperator::StreamAsofJoin(asof) => {
            asof.set_output_projection(projection.columns)?;
            Arc::clone(asof.output_ports()[0].schema().expect("exact ASOF schema"))
        }
        NodeOperator::StreamJoin(join) => {
            join.set_output_projection(projection.columns)?;
            Arc::clone(join.output_ports()[0].schema().expect("exact Join schema"))
        }
        _ => unreachable!(),
    };
    for consumer in projection.consumers {
        let NodeOperator::Expression(expression) = &mut builder
            .nodes
            .get_mut(&consumer)
            .expect("validated projection node")
            .operator
        else {
            unreachable!()
        };
        expression.set_projected_stream_input(Arc::clone(&schema))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    mod join;

    use std::{sync::Arc, time::Duration};

    use datafusion::arrow::datatypes::{DataType, Field, Schema, TimeUnit};

    use crate::{
        AsofJoinSide, AsofStateLimits, BatchKind, Edge, ExpressionOperator, OperatorMetadata,
        PipelineBuilder, Port, PortEndpoint, StreamAsofJoinOperator, StreamAsofJoinSpec,
        StreamRequirements, UdfRegistry,
    };

    fn graph(selects: &[&[&str]]) -> PipelineBuilder {
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new(
                "time",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("seq", DataType::Int64, false),
            Field::new("value", DataType::Int64, true),
        ]));
        let side = |prefix: &str| {
            AsofJoinSide::new(
                vec!["key".into()],
                "time".into(),
                vec!["seq".into()],
                prefix.into(),
            )
            .unwrap()
        };
        let spec = StreamAsofJoinSpec::new(
            side("left"),
            side("right"),
            Duration::ZERO,
            AsofStateLimits::new(100, 1_000_000).unwrap(),
        )
        .unwrap();
        let asof = StreamAsofJoinOperator::new("asof", schema.clone(), schema, spec).unwrap();
        let output = asof.output_ports()[0].schema().unwrap().clone();
        let mut builder = PipelineBuilder::new("projected asof")
            .unwrap()
            .add_node("asof", asof)
            .unwrap();
        for (index, select) in selects.iter().enumerate() {
            let name = format!("select{index}");
            let expression = ExpressionOperator::new(
                &name,
                "",
                select.iter().map(|value| (*value).into()).collect(),
                None,
                vec![],
            )
            .unwrap()
            .with_ports(
                Port::with_schema_ref("input", BatchKind::Table, true, Some(output.clone()))
                    .unwrap(),
                Port::new("output", BatchKind::Table, true, None).unwrap(),
            )
            .unwrap();
            builder = builder
                .add_node(&name, Box::new(expression))
                .unwrap()
                .connect(Edge::new(
                    PortEndpoint::new("asof", "output").unwrap(),
                    PortEndpoint::new(&name, "input").unwrap(),
                ))
                .unwrap();
        }
        builder
    }

    fn physical_fields(builder: PipelineBuilder) -> Vec<String> {
        let plan = builder
            .compile_stream(
                &UdfRegistry::new().snapshot(),
                &StreamRequirements::default(),
            )
            .unwrap();
        let asof = plan
            .nodes
            .iter()
            .find(|node| node.node_id == "asof")
            .unwrap();
        asof.output_ports["output"]
            .schema()
            .unwrap()
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect()
    }

    #[test]
    fn asof_materializes_only_columns_required_by_every_projection_consumer() {
        join::exact_consumers_prune_join_output();
        assert_eq!(
            physical_fields(graph(&[
                &["right__value AS price", "left__seq"],
                &["left__key", "left__seq AS sequence"],
            ])),
            vec!["left__key", "left__seq", "right__value"]
        );
    }

    #[test]
    fn arithmetic_consumer_requires_full_asof_output() {
        join::arithmetic_consumers_prune_join_output();
        assert_eq!(
            physical_fields(graph(&[&["left__seq"], &["right__value + 1 AS price"],])).len(),
            8
        );
    }

    #[test]
    fn unconsumed_asof_output_keeps_its_full_schema() {
        assert_eq!(physical_fields(graph(&[])).len(), 8);
    }

    fn input(schema: Arc<Schema>, right: bool) -> crate::Batch {
        use datafusion::arrow::{
            array::{ArrayRef, Int64Array, StringArray, TimestampMicrosecondArray},
            record_batch::RecordBatch,
        };
        let columns: Vec<ArrayRef> = if right {
            vec![
                Arc::new(StringArray::from(vec!["A"])),
                Arc::new(TimestampMicrosecondArray::from(vec![100]).with_timezone("UTC")),
                Arc::new(Int64Array::from(vec![10])),
                Arc::new(Int64Array::from(vec![500])),
            ]
        } else {
            vec![
                Arc::new(StringArray::from(vec!["A", "B"])),
                Arc::new(TimestampMicrosecondArray::from(vec![100, 101]).with_timezone("UTC")),
                Arc::new(Int64Array::from(vec![7, 8])),
                Arc::new(Int64Array::from(vec![1, 2])),
            ]
        };
        crate::Batch::table(
            vec![RecordBatch::try_new(schema, columns).unwrap()],
            crate::BatchMetadata::default(),
        )
        .unwrap()
    }

    async fn restored_join(
        plan: &mut super::super::StreamExecutionPlan,
        job: &crate::StreamJobContext,
        right_only: bool,
    ) -> crate::Batch {
        use super::super::CompiledStreamOperator;
        use crate::{EdgeCollector, Epoch, StreamOperator, StreamOperatorContext};
        let udfs = UdfRegistry::new().snapshot();
        let context = StreamOperatorContext::new(job, "asof", None);
        let node = plan
            .nodes
            .iter_mut()
            .find(|node| node.node_id == "asof")
            .unwrap();
        let CompiledStreamOperator::StreamAsofJoin(asof) = &mut node.operator else {
            unreachable!()
        };
        let schema = asof.input_ports()[0].schema().unwrap().clone();
        let mut collector = EdgeCollector::new(asof.output_ports().to_vec());
        asof.process_data(
            "right",
            input(schema.clone(), true),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
        asof.process_data("left", input(schema, false), &context, &mut collector)
            .await
            .unwrap();
        asof.prepare_checkpoint_async(&context).await.unwrap();
        let snapshot = asof.checkpoint(Epoch::new(1).unwrap()).unwrap();

        let mut original = graph(&[])
            .compile_stream(&udfs, &StreamRequirements::default())
            .unwrap();
        let CompiledStreamOperator::StreamAsofJoin(full) = &mut original.nodes[0].operator else {
            unreachable!()
        };
        assert!(full.restore(&snapshot).is_err());
        let mut full_collector = EdgeCollector::new(full.output_ports().to_vec());
        let schema = full.input_ports()[0].schema().unwrap().clone();
        full.process_data(
            "right",
            input(schema.clone(), true),
            &context,
            &mut full_collector,
        )
        .await
        .unwrap();
        full.process_data("left", input(schema, false), &context, &mut full_collector)
            .await
            .unwrap();
        full.on_end(&context, &mut full_collector).await.unwrap();
        let full_output = full_collector.drain("output");
        let full_record = &full_output[0]
            .as_data()
            .unwrap()
            .table_payload()
            .unwrap()
            .batches()[0];
        assert_eq!(full_record.num_columns(), 8);

        let selections: &[&[&str]] = if right_only {
            &[&["right__value AS price"]]
        } else {
            &[
                &["right__value AS price", "left__seq AS sequence"],
                &["right__seq"],
            ]
        };
        let mut restored = graph(selections)
            .compile_stream(&udfs, &StreamRequirements::default())
            .unwrap();
        let CompiledStreamOperator::StreamAsofJoin(cold) = &mut restored.nodes[0].operator else {
            unreachable!()
        };
        cold.restore(&snapshot).unwrap();
        let mut cold_collector = EdgeCollector::new(cold.output_ports().to_vec());
        cold.on_end(&context, &mut cold_collector).await.unwrap();
        let joined = cold_collector.drain("output");
        let joined = joined[0].as_data().unwrap().clone();
        let joined_record = &joined.table_payload().unwrap().batches()[0];
        let indices = if right_only { vec![7] } else { vec![2, 6, 7] };
        assert_eq!(joined_record, &full_record.project(&indices).unwrap());
        joined
    }

    #[tokio::test]
    async fn physical_asof_projection_preserves_aliases_nulls_and_projected_state_restore() {
        use crate::{
            CancellationToken, EdgeCollector, JsonMap, StreamJobContext, StreamOperatorContext,
        };
        use datafusion::arrow::array::{Array, Int64Array};

        join::execution_and_restore().await;
        for right_only in [false, true] {
            let selections: &[&[&str]] = if right_only {
                &[&["right__value AS price"]]
            } else {
                &[
                    &["right__value AS price", "left__seq AS sequence"],
                    &["right__seq"],
                ]
            };
            let builder = graph(selections);
            let udfs = UdfRegistry::new().snapshot();
            let fingerprint = super::super::compile_graph(&builder, "stream", &udfs)
                .unwrap()
                .fingerprint;
            let mut plan = builder
                .compile_stream(&udfs, &StreamRequirements::default())
                .unwrap();
            assert_eq!(plan.fingerprint, fingerprint);
            let job = StreamJobContext::new(
                1,
                &fingerprint,
                JsonMap::new(),
                None,
                CancellationToken::new(),
            );
            let joined = restored_join(&mut plan, &job, right_only).await;

            for (index, select) in selections.iter().enumerate() {
                let name = format!("select{index}");
                let node = plan
                    .nodes
                    .iter_mut()
                    .find(|node| node.node_id == name)
                    .unwrap();
                let mut output = EdgeCollector::new(node.output_ports.values().cloned().collect());
                node.operator
                    .process_data(
                        "input",
                        joined.clone(),
                        &StreamOperatorContext::new(&job, &name, None),
                        &mut output,
                    )
                    .await
                    .unwrap();
                let events = output.drain("output");
                let record = &events[0]
                    .as_data()
                    .unwrap()
                    .table_payload()
                    .unwrap()
                    .batches()[0];
                assert_eq!(record.num_columns(), select.len());
                let values = record
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap();
                assert_eq!(values.value(0), if index == 0 { 500 } else { 10 });
                assert!(values.is_null(1));
                if index == 0 {
                    assert_eq!(record.schema().field(0).name(), "price");
                    if !right_only {
                        let sequence = record
                            .column(1)
                            .as_any()
                            .downcast_ref::<Int64Array>()
                            .unwrap();
                        assert_eq!(sequence.values().as_ref(), &[7, 8]);
                        assert_eq!(record.schema().field(1).name(), "sequence");
                    }
                }
            }
        }
    }
}
