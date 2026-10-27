//! Python 06: an equivalent Rust array provider, without a Python/`NumPy` host.
mod support;
use async_trait::async_trait;
use calc_flow::{
    Batch, BatchKind, BatchOperator, BatchOperatorContext, ExecutionOptions, JsonMap,
    OperatorMetadata, PipelineBuilder, Port, Result, StaticArrayValues, UdfRegistry,
};
use std::collections::BTreeMap;
use support::ExampleResult;

struct Center {
    inputs: Vec<Port>,
    outputs: Vec<Port>,
}
impl OperatorMetadata for Center {
    fn name(&self) -> &'static str {
        "center"
    }
    fn input_ports(&self) -> &[Port] {
        &self.inputs
    }
    fn output_ports(&self) -> &[Port] {
        &self.outputs
    }
    fn configuration(&self) -> JsonMap {
        BTreeMap::from([("expression".into(), serde_json::json!("x - mean(x)"))])
    }
}
#[async_trait]
impl BatchOperator for Center {
    async fn process(
        &mut self,
        inputs: &BTreeMap<String, Batch>,
        _: &BatchOperatorContext<'_>,
    ) -> Result<BTreeMap<String, Batch>> {
        let snapshot = inputs["input"]
            .static_array_snapshot()
            .expect("owned static array");
        let StaticArrayValues::Float(values) = snapshot.values() else {
            return Err(support::invalid("input", "expected float array"));
        };
        let count =
            u32::try_from(values.len()).map_err(|e| support::invalid("input", e.to_string()))?;
        if count == 0 {
            return Err(support::invalid("input", "expected nonempty array"));
        }
        let mean = values.iter().sum::<f64>() / f64::from(count);
        let result = Batch::static_array_float(
            "rust-array",
            "float64",
            snapshot.shape().to_vec(),
            None,
            values.iter().map(|v| v - mean).collect(),
        )?;
        Ok(BTreeMap::from([("output".into(), result)]))
    }
}
#[tokio::main]
async fn main() -> ExampleResult {
    let input = Batch::static_array_float(
        "rust-array",
        "float64",
        vec![4],
        None,
        vec![1.0, 2.0, 4.0, 6.0],
    )?;
    let before = input.static_array_snapshot().unwrap();
    let operator = Center {
        inputs: vec![Port::new("input", BatchKind::Array, true, None)?],
        outputs: vec![Port::new("output", BatchKind::Array, true, None)?],
    };
    let plan = PipelineBuilder::new("rust-array")?
        .add_node("center", Box::new(operator) as Box<dyn BatchOperator>)?
        .compile_batch(&UdfRegistry::new().snapshot())?;
    let run = plan
        .execute(
            BTreeMap::from([("input".into(), input.clone())]),
            ExecutionOptions::default(),
        )
        .await?;
    let snapshot = run.outputs["output"].static_array_snapshot().unwrap();
    let StaticArrayValues::Float(actual) = snapshot.values() else {
        unreachable!()
    };
    assert_eq!(actual, &[-2.25, -1.25, 0.75, 2.75]);
    let StaticArrayValues::Float(original) = before.values() else {
        unreachable!()
    };
    assert_eq!(original, &[1.0, 2.0, 4.0, 6.0]);
    assert!(run.datafusion_metrics.is_empty());
    println!("{actual:?}");
    Ok(())
}
