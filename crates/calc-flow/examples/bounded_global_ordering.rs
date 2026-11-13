//! The core crate owns ordering; no Python or public facade dependency is needed.
use std::{collections::BTreeMap, sync::Arc};

use calc_flow::{
    Batch, BatchMetadata, CompileMode, CompileOptions, ExecutionOptions, InputOptions,
    NullPlacement, Program, Result, SortDirection, SortKey, SortSpec, StateBudget, UdfRegistry,
    arrow::{
        array::{Int64Array, UInt64Array},
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    },
    table_input,
};

#[tokio::main]
async fn main() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Int64, true),
        Field::new("seq", DataType::UInt64, false),
    ]));
    let events = table_input("events", schema.clone(), InputOptions::default())?;
    let spec = SortSpec::bounded_source(StateBudget::new(1000, 8 << 20)?)?
        .with_sequence_by(vec!["seq".into()])?;
    let out = events.top_k(
        1,
        vec![SortKey::new(
            "x",
            SortDirection::Ascending,
            NullPlacement::Last,
        )?],
        spec,
    )?;
    let program = Program::new(
        "smallest",
        CompileMode::Batch,
        BTreeMap::from([("out".into(), out)]),
    )?;
    let plan = program
        .compile_batch(&UdfRegistry::new().snapshot(), &CompileOptions::default())
        .await?;
    let batch = Batch::table(
        vec![
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(Int64Array::from(vec![3, 1, 2])),
                    Arc::new(UInt64Array::from(vec![10, 11, 12])),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )?;
    let result = plan
        .execute(
            BTreeMap::from([("events".into(), batch)]),
            ExecutionOptions::default(),
        )
        .await?;
    assert_eq!(result.outputs["out"].num_rows(), 1);
    let rows = result.outputs["out"].table_payload()?;
    assert_eq!(
        rows.batches()[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0),
        1
    );
    println!("complete H=[3,1,2], final top-1=[1]");
    Ok(())
}
