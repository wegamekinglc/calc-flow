//! Python 03: register an exact, trusted Rust/`DataFusion` scalar UDF.
mod support;
use calc_flow::{
    ExecutionOptions, ExpressionOperator, PipelineBuilder, UdfKind, UdfReference, UdfRegistry,
};
use datafusion::{
    arrow::{array::Int64Array, datatypes::DataType},
    logical_expr::{ColumnarValue, Volatility, create_udf},
};
use std::{collections::BTreeMap, sync::Arc};
use support::{ExampleResult, i64_values, ints, table};

#[tokio::main]
async fn main() -> ExampleResult {
    let (snapshot, reference) = registered_udf()?;
    let plan = PipelineBuilder::new("registered-udf")?
        .add_node(
            "calculate",
            Box::new(ExpressionOperator::new(
                "calculate",
                "total = double_amount(amount)",
                vec![],
                None,
                vec![reference],
            )?),
        )?
        .compile_batch(&snapshot)?;
    let result = plan
        .execute(
            BTreeMap::from([(
                "input".into(),
                table(vec![("amount", ints(&[100, 250, 400]))]),
            )]),
            ExecutionOptions::default(),
        )
        .await?;
    assert_eq!(
        i64_values(&[result.outputs["output"].clone()], "total"),
        [Some(200), Some(500), Some(800)]
    );
    println!("registered catalog: {:?}", snapshot.catalog());
    support::print(&[result.outputs["output"].clone()])
}

fn registered_udf() -> calc_flow::Result<(calc_flow::UdfRegistrySnapshot, UdfReference)> {
    let reference = UdfReference::new("rust", "double_amount", "1", UdfKind::DataFusionScalar)?;
    let udf = create_udf(
        "double_amount",
        vec![DataType::Int64],
        DataType::Int64,
        Volatility::Immutable,
        Arc::new(|arguments| {
            let arrays = ColumnarValue::values_to_arrays(arguments)?;
            let values = arrays[0]
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("declared int64 input");
            Ok(ColumnarValue::Array(Arc::new(
                values
                    .iter()
                    .map(|v| v.map(|v| v * 2))
                    .collect::<Int64Array>(),
            )))
        }),
    );
    let mut registry = UdfRegistry::new();
    registry.register_datafusion(reference.clone(), Arc::new(udf), 1)?;
    Ok((registry.snapshot(), reference))
}
