//! Native stages corresponding to Python's nested delta/mean expressions.
use crate::support::{connect_all, expression, expression_schemas, rolling, sql, sql_input_schema};
use calc_flow::{
    NodeOperator, OperatorMetadata, PipelineBuilder, Result, RollingFrameSpec, RollingOperator,
    RollingOutputSpec,
};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use std::sync::Arc;

pub fn delta_mean(
    name: &str,
    schema: SchemaRef,
    size: u64,
    gains: bool,
) -> Result<PipelineBuilder> {
    let delta = delta_operator(schema)?;
    let next_schema = delta.output_ports()[0].schema().unwrap().clone();
    let gain_schema = gain_schema(next_schema.clone(), gains);
    let mean = mean_operator(gain_schema.clone(), size, gains)?;
    let nodes = stages(delta, mean, next_schema, gain_schema, gains)?;
    let builder = nodes
        .into_iter()
        .try_fold(PipelineBuilder::new(name)?, |builder, (id, node)| {
            builder.add_node(id, node)
        })?;
    let connections = if gains {
        [
            ("delta", "output", "gain", "input"),
            ("gain", "output", "mean", "input"),
        ]
    } else {
        [
            ("delta", "output", "mean", "input"),
            ("mean", "output", "sql", "input"),
        ]
    };
    connect_all(builder, &connections)
}

fn delta_operator(schema: SchemaRef) -> Result<RollingOperator> {
    RollingOperator::new(
        "delta",
        schema,
        rolling(vec![RollingOutputSpec::Delta {
            primitive_version: 1,
            input: "price".into(),
            output: "change".into(),
            periods: 1,
        }]),
    )
}

fn gain_schema(schema: SchemaRef, gains: bool) -> SchemaRef {
    if gains {
        let mut fields = schema.fields().to_vec();
        fields.push(Arc::new(Field::new("gain", DataType::Float64, true)));
        Arc::new(Schema::new(fields))
    } else {
        schema
    }
}

fn mean_operator(schema: SchemaRef, size: u64, gains: bool) -> Result<RollingOperator> {
    let (input, output) = if gains {
        ("gain", "average_gain")
    } else {
        ("change", "mean_delta")
    };
    RollingOperator::new(
        "mean",
        schema,
        rolling(vec![RollingOutputSpec::Mean {
            primitive_version: 1,
            input: input.into(),
            output: output.into(),
            frame: RollingFrameSpec::Rows { size },
            min_periods: 1,
        }]),
    )
}

fn stages(
    delta: RollingOperator,
    mean: RollingOperator,
    input: SchemaRef,
    gain: SchemaRef,
    gains: bool,
) -> Result<Vec<(&'static str, NodeOperator)>> {
    if gains {
        let gain = expression_schemas(
            expression(
                "gain",
                "gain = CASE WHEN change IS NULL THEN NULL ELSE LEAST(GREATEST(change, 0.0), 1e100) END",
                &[],
                None,
            )?,
            Some(input),
            Some(gain),
        )?;
        Ok(vec![
            ("delta", delta.into()),
            ("gain", gain.into()),
            ("mean", mean.into()),
        ])
    } else {
        let mean_schema = mean.output_ports()[0].schema().unwrap().clone();
        let sql = sql_input_schema(
            sql(
                "sql",
                "SELECT change AS delta, mean_delta FROM input",
                &["input"],
            )?,
            mean_schema,
        )?;
        Ok(vec![
            ("delta", delta.into()),
            ("mean", mean.into()),
            ("sql", sql.into()),
        ])
    }
}
