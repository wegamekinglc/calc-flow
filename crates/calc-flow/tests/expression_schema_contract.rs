use calc_flow::ExpressionOperator;
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;

#[tokio::test]
async fn infer_stream_schema_preserves_projection_metadata() {
    let schema = Arc::new(Schema::new_with_metadata(
        vec![Field::new("Price", DataType::Float64, false).with_metadata(
            std::collections::HashMap::from([("unit".into(), "USD".into())]),
        )],
        std::collections::HashMap::from([("origin".into(), "quotes".into())]),
    ));
    let operator = ExpressionOperator::new(
        "inferred",
        "",
        vec![r#""Price" AS "renamed""#.into()],
        None,
        vec![],
    )
    .unwrap();
    let inferred = operator
        .infer_stream_schema(Arc::clone(&schema))
        .await
        .unwrap();
    assert_eq!(
        inferred.field(0),
        &schema.field(0).clone().with_name("renamed")
    );
    assert_eq!(inferred.metadata(), schema.metadata());
}
