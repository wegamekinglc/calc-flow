//! Combine native Table RSI, ATR and lag in one finite calculation.

use std::{collections::BTreeMap, sync::Arc};

use calc_flow::{
    Batch, BatchExecutionPlan, BatchMetadata, ExecutionOptions, LateErrorScope, LatePolicySpec,
    PipelineBuilder, ROLLING_CONFIGURATION_VERSION, ROLLING_INDICATOR_STATE_LAYOUT_VERSION,
    RollingNumericalProfile, RollingOperator, RollingOutputSpec, RollingSpec, RollingValuePolicy,
    StateBudget, UdfRegistry,
};
use datafusion::arrow::{
    array::{Array, ArrayRef, Float64Array, StringArray, TimestampMicrosecondArray, UInt64Array},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};

fn quotes() -> Result<RecordBatch, Box<dyn std::error::Error>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("entity", DataType::Utf8, false),
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("seq", DataType::UInt64, false),
        Field::new("price", DataType::Float64, false),
        Field::new("high", DataType::Float64, false),
        Field::new("low", DataType::Float64, false),
        Field::new("close", DataType::Float64, false),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(vec!["a"; 4])),
        Arc::new(TimestampMicrosecondArray::from(vec![1, 2, 3, 4]).with_timezone("UTC")),
        Arc::new(UInt64Array::from(vec![1, 2, 3, 4])),
        Arc::new(Float64Array::from(vec![1.0, 2.0, 1.0, 3.0])),
        Arc::new(Float64Array::from(vec![10.0, 12.0, 11.0, 14.0])),
        Arc::new(Float64Array::from(vec![8.0, 9.0, 8.0, 11.0])),
        Arc::new(Float64Array::from(vec![9.0, 10.0, 9.0, 13.0])),
    ];
    Ok(RecordBatch::try_new(schema, columns)?)
}

fn specification() -> RollingSpec {
    RollingSpec {
        configuration_version: ROLLING_CONFIGURATION_VERSION,
        state_layout_version: ROLLING_INDICATOR_STATE_LAYOUT_VERSION,
        numerical_profile: RollingNumericalProfile::StableV1,
        partition_by: vec!["entity".into()],
        event_time: "ts".into(),
        sequence_by: vec!["seq".into()],
        outputs: vec![
            RollingOutputSpec::Rsi {
                primitive_version: 1,
                input: "price".into(),
                output: "rsi".into(),
                period: 2,
            },
            RollingOutputSpec::Atr {
                primitive_version: 1,
                high: "high".into(),
                low: "low".into(),
                close: "close".into(),
                output: "atr".into(),
                period: 2,
            },
            RollingOutputSpec::Lag {
                primitive_version: 1,
                input: "price".into(),
                output: "previous_price".into(),
                periods: 1,
            },
        ],
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
        .expect("declared output exists")
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("indicator and price outputs are float64")
        .iter()
        .collect()
}

fn plan(schema: Arc<Schema>) -> calc_flow::Result<BatchExecutionPlan> {
    let mut operator = RollingOperator::new("features", schema, specification())?;
    operator.set_state_budget(StateBudget::new(100, 1 << 20)?)?;
    PipelineBuilder::new("native-table-indicators")?
        .add_node("features", operator)?
        .compile_batch(&UdfRegistry::new().snapshot())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input = quotes()?;
    let result = plan(input.schema())?
        .execute(
            BTreeMap::from([(
                "input".into(),
                Batch::table(vec![input], BatchMetadata::default())?,
            )]),
            ExecutionOptions::default(),
        )
        .await?;
    let batches = result.outputs["output"].table_payload()?.batches();
    assert_eq!(batches.len(), 1);
    let batch = &batches[0];
    assert_eq!(
        values(batch, "rsi"),
        [None, None, Some(50.0), Some(100.0 / 1.2)]
    );
    assert_eq!(values(batch, "atr"), [None, None, Some(3.0), Some(4.0)]);
    assert_eq!(
        values(batch, "previous_price"),
        [None, Some(1.0), Some(2.0), Some(1.0)]
    );
    println!("RSI: {:?}", values(batch, "rsi"));
    println!("ATR: {:?}", values(batch, "atr"));
    Ok(())
}
