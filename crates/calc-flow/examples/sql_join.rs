use std::{collections::BTreeMap, sync::Arc};

use calc_flow::{
    Batch, BatchMetadata, ExecutionOptions, PipelineBuilder, SqlOperator, UdfRegistry,
};
use datafusion::arrow::{array::Int64Array, record_batch::RecordBatch};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let plan = PipelineBuilder::new("orders-and-fees")?
        .add_node(
            "join",
            Box::new(SqlOperator::new(
                "join",
                "SELECT orders.order_id, orders.amount - fees.fee AS net \
                 FROM orders JOIN fees ON orders.order_id = fees.order_id \
                 ORDER BY orders.order_id",
                vec!["orders".to_string(), "fees".to_string()],
                Vec::new(),
            )?),
        )?
        .compile_batch(&UdfRegistry::new().snapshot())?;
    let orders = RecordBatch::try_from_iter(vec![
        (
            "order_id",
            Arc::new(Int64Array::from(vec![1, 2, 3])) as Arc<dyn datafusion::arrow::array::Array>,
        ),
        ("amount", Arc::new(Int64Array::from(vec![75, 120, 40])) as _),
    ])?;
    let fees = RecordBatch::try_from_iter(vec![
        (
            "order_id",
            Arc::new(Int64Array::from(vec![1, 2, 3])) as Arc<dyn datafusion::arrow::array::Array>,
        ),
        ("fee", Arc::new(Int64Array::from(vec![5, 12, 4])) as _),
    ])?;
    let result = plan
        .execute(
            BTreeMap::from([
                (
                    "orders".into(),
                    Batch::table(vec![orders], BatchMetadata::default())?,
                ),
                (
                    "fees".into(),
                    Batch::table(vec![fees], BatchMetadata::default())?,
                ),
            ]),
            ExecutionOptions::default(),
        )
        .await?;
    let output = result.outputs["output"].table_payload()?;
    let net = output.batches()[0]
        .column_by_name("net")
        .expect("sql output contains net")
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("net is an Int64 column");

    assert_eq!(net.values(), &[70, 108, 36]);
    println!("net amounts: {net:?}");
    Ok(())
}
