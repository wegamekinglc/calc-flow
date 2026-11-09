//! Compute all MACD components over immutable nullable Table rows.

use std::{collections::BTreeMap, sync::Arc};

use calc_flow::{
    Batch, BatchMetadata, ExecutionOptions, LateErrorScope, LatePolicySpec, MacdComponent,
    PipelineBuilder, ROLLING_CONFIGURATION_VERSION, ROLLING_MACD_STATE_LAYOUT_VERSION,
    RollingNumericalProfile, RollingOperator, RollingOutputSpec, RollingSpec, RollingValuePolicy,
    UdfRegistry,
};
use datafusion::arrow::{
    array::{ArrayRef, Float64Array, TimestampMicrosecondArray, UInt64Array},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};

fn quotes() -> Result<RecordBatch, Box<dyn std::error::Error>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("entity", DataType::UInt64, false),
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("seq", DataType::UInt64, false),
        Field::new("price", DataType::UInt64, true),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(UInt64Array::from(vec![1; 4])),
        Arc::new(TimestampMicrosecondArray::from(vec![1, 2, 3, 4]).with_timezone("UTC")),
        Arc::new(UInt64Array::from(vec![1, 2, 3, 4])),
        Arc::new(UInt64Array::from(vec![Some(10), Some(14), None, None])),
    ];
    Ok(RecordBatch::try_new(schema, columns)?)
}
fn specification() -> RollingSpec {
    let outputs = [
        ("line", MacdComponent::Line),
        ("signal", MacdComponent::Signal),
        ("histogram", MacdComponent::Histogram),
    ]
    .map(|(output, component)| RollingOutputSpec::MacdComponent {
        primitive_version: 1,
        input: "price".into(),
        output: output.into(),
        fast_span: 1,
        slow_span: 3,
        signal_span: 3,
        min_periods: 1,
        signal_min_periods: 1,
        component,
    })
    .to_vec();
    RollingSpec {
        configuration_version: ROLLING_CONFIGURATION_VERSION,
        state_layout_version: ROLLING_MACD_STATE_LAYOUT_VERSION,
        numerical_profile: RollingNumericalProfile::StableV1,
        partition_by: vec!["entity".into()],
        event_time: "ts".into(),
        sequence_by: vec!["seq".into()],
        outputs,
        allowed_lateness_micros: 0,
        late_policy: LatePolicySpec::Error {
            scope: LateErrorScope::Envelope,
        },
        value_policy: RollingValuePolicy::StatefulNumericV1,
    }
}
fn values(batch: &RecordBatch, name: &str) -> Vec<Option<f64>> {
    batch
        .column_by_name(name)
        .unwrap()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .iter()
        .collect()
}
async fn compute(input: RecordBatch) -> calc_flow::Result<Batch> {
    let plan = PipelineBuilder::new("native-table-macd")?
        .add_node(
            "features",
            RollingOperator::new("features", input.schema(), specification())?,
        )?
        .compile_batch(&UdfRegistry::new().snapshot())?;
    let mut result = plan
        .execute(
            BTreeMap::from([(
                "input".into(),
                Batch::table(vec![input], BatchMetadata::default())?,
            )]),
            ExecutionOptions::default(),
        )
        .await?;
    Ok(result.outputs.remove("output").unwrap())
}
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output = compute(quotes()?).await?;
    let batch = &output.table_payload()?.batches()[0];
    assert_eq!(
        values(batch, "line"),
        [Some(0.0), Some(2.0), Some(2.0), Some(2.0)]
    );
    assert_eq!(
        values(batch, "signal"),
        [Some(0.0), Some(1.0), Some(1.5), Some(1.75)]
    );
    assert_eq!(
        values(batch, "histogram"),
        [Some(0.0), Some(1.0), Some(0.5), Some(0.25)]
    );
    println!("MACD signal: {:?}", values(batch, "signal"));
    Ok(())
}
