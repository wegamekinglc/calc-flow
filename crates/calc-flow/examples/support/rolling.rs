//! Native stages corresponding to Python's nested delta/mean expressions.
use crate::support::{edge, expression, expression_schemas, rolling, sql_input_schema};
use calc_flow::{
    OperatorMetadata, PipelineBuilder, Result, RollingFrameSpec, RollingOperator, RollingOutputSpec,
};
use datafusion::arrow::datatypes::SchemaRef;

pub fn delta_mean(
    name: &str,
    schema: SchemaRef,
    size: u64,
    gains: bool,
) -> Result<PipelineBuilder> {
    let delta = RollingOperator::new(
        "delta",
        schema,
        rolling(vec![RollingOutputSpec::Delta {
            primitive_version: 1,
            input: "price".into(),
            output: "change".into(),
            periods: 1,
        }]),
    )?;
    let next_schema = delta.output_ports()[0]
        .schema()
        .expect("rolling schema")
        .clone();
    let (input, output) = if gains {
        ("gain", "average_gain")
    } else {
        ("change", "mean_delta")
    };
    let gain_schema = if gains {
        let mut fields = next_schema.fields().to_vec();
        fields.push(std::sync::Arc::new(
            datafusion::arrow::datatypes::Field::new(
                "gain",
                datafusion::arrow::datatypes::DataType::Float64,
                true,
            ),
        ));
        std::sync::Arc::new(datafusion::arrow::datatypes::Schema::new(fields))
    } else {
        next_schema.clone()
    };
    let mean = RollingOperator::new(
        "mean",
        gain_schema.clone(),
        rolling(vec![RollingOutputSpec::Mean {
            primitive_version: 1,
            input: input.into(),
            output: output.into(),
            frame: RollingFrameSpec::Rows { size },
            min_periods: 1,
        }]),
    )?;
    let mean_schema = mean.output_ports()[0].schema().unwrap().clone();
    let builder = PipelineBuilder::new(name)?.add_node("delta", delta)?;
    if gains {
        builder.add_node("gain", expression_schemas(expression("gain", "gain = CASE WHEN change IS NULL THEN NULL ELSE LEAST(GREATEST(change, 0.0), 1e100) END", &[], None)?, Some(next_schema), Some(gain_schema))?)?
            .add_node("mean", mean)?.connect(edge("delta", "output", "gain", "input")?)?.connect(edge("gain", "output", "mean", "input")?)
    } else {
        builder
            .add_node("mean", mean)?
            .connect(edge("delta", "output", "mean", "input")?)?
            .add_node(
                "sql",
                sql_input_schema(
                    crate::support::sql(
                        "sql",
                        "SELECT change AS delta, mean_delta FROM input",
                        &["input"],
                    )?,
                    mean_schema,
                )?,
            )?
            .connect(edge("mean", "output", "sql", "input")?)
    }
}
