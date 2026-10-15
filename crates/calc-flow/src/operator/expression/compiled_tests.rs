use super::*;
use crate::{BatchMetadata, CancellationToken, EdgeCollector, StreamJobContext};

pub(super) async fn repeated_stream_batches_skip_plan_rebinding() {
    for (select, filter) in [
        (vec![r#""Price" + 1 AS next"#.into()], None),
        (
            vec![r#""Price" + 1 AS next"#.into(), "symbol".into()],
            Some(r#""Price" > 5"#.into()),
        ),
    ] {
        let mut operator = ExpressionOperator::new("compiled", "", select, filter, vec![]).unwrap();
        let mut reference = operator.clone();
        let job = StreamJobContext::new(
            1,
            "compiled-expression-test",
            JsonMap::new(),
            None,
            CancellationToken::new(),
        );
        let context = StreamOperatorContext::new(&job, "compiled", None);
        let mut output = EdgeCollector::new(operator.output_ports().to_vec());
        for shift in [0.0, 20.0] {
            let input = shifted_input(shift);
            let expected = reference.process_standalone(input.clone()).await.unwrap();
            operator
                .process_data("input", input.clone(), &context, &mut output)
                .await
                .unwrap();
            let actual = output.drain("output");
            assert_eq!(actual.len(), 1);
            let actual = actual[0].as_data().unwrap();
            assert_eq!(actual.metadata(), expected.metadata());
            assert_eq!(
                actual.table_payload().unwrap().schema(),
                expected.table_payload().unwrap().schema()
            );
            assert_eq!(
                actual.table_payload().unwrap().batches(),
                expected.table_payload().unwrap().batches()
            );
            assert_eq!(
                input.table_payload().unwrap().batches(),
                shifted_input(shift).table_payload().unwrap().batches()
            );
        }
        let runtime = operator.stream_state.runtime.as_ref().unwrap();
        assert_eq!(
            runtime.metrics().len(),
            1,
            "only the first stream batch should need a SQL execution plan"
        );
    }
    constants_preserve_zero_column_input_rows().await;
    chunks_and_schema_changes().await;
    predicate_column_is_removed_after_filtering().await;
    retained_input_buffers_are_released().await;
    evaluation_errors_leave_the_cache_usable().await;
    unsupported_queries_keep_sql_execution().await;
    selected_udfs_keep_sql_execution().await;
    cancellation_and_abandonment_release_call_state().await;
    let job = job(CancellationToken::new());
    crate::datafusion_template::check_arrow_error_envelope(&StreamOperatorContext::new(
        &job, "compiled", None,
    ))
    .await;
}

fn shifted_input(shift: f64) -> Batch {
    let input = tests::projection_input();
    let record = &input.table_payload().unwrap().batches()[0];
    let record = datafusion::arrow::record_batch::RecordBatch::try_new(
        record.schema(),
        vec![
            std::sync::Arc::new(datafusion::arrow::array::Float64Array::from(vec![
                Some(10.0 + shift),
                None,
            ])),
            record.column(1).clone(),
        ],
    )
    .unwrap();
    Batch::table(vec![record], input.metadata().clone()).unwrap()
}

fn job(token: CancellationToken) -> StreamJobContext {
    StreamJobContext::new(1, "compiled-controls", JsonMap::new(), None, token)
}

fn operator(select: &[&str], filter: Option<&str>) -> ExpressionOperator {
    ExpressionOperator::new(
        "compiled",
        "",
        select.iter().map(|value| (*value).into()).collect(),
        filter.map(str::to_owned),
        vec![],
    )
    .unwrap()
}

async fn collect(
    operator: &mut ExpressionOperator,
    input: Batch,
    job: &StreamJobContext,
) -> Result<Batch> {
    let context = StreamOperatorContext::new(job, "compiled", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let result = operator
        .process_data("input", input, &context, &mut output)
        .await;
    if let Err(error) = result {
        assert!(output.drain("output").is_empty());
        return Err(error);
    }
    let messages = output.drain("output");
    assert_eq!(messages.len(), 1);
    Ok(messages[0].as_data().unwrap().clone())
}

async fn compare(
    operator: &mut ExpressionOperator,
    reference: &mut ExpressionOperator,
    input: Batch,
    job: &StreamJobContext,
) {
    let original = input.table_payload().unwrap().batches().to_vec();
    let expected = reference.process_standalone(input.clone()).await.unwrap();
    let actual = collect(operator, input.clone(), job).await.unwrap();
    assert_eq!(actual.metadata(), expected.metadata());
    assert_eq!(actual.source_cursor(), expected.source_cursor());
    assert_eq!(
        actual.table_payload().unwrap().schema(),
        expected.table_payload().unwrap().schema()
    );
    assert_eq!(
        actual.table_payload().unwrap().batches(),
        expected.table_payload().unwrap().batches()
    );
    assert_eq!(input.table_payload().unwrap().batches(), original);
}

fn chunks() -> Batch {
    let mut records = [0.0, 20.0, 40.0, 60.0]
        .into_iter()
        .map(|shift| shifted_input(shift).table_payload().unwrap().batches()[0].clone())
        .collect::<Vec<_>>();
    records.insert(
        1,
        datafusion::arrow::record_batch::RecordBatch::new_empty(records[0].schema()),
    );
    Batch::table(records, tests::projection_input().metadata().clone()).unwrap()
}

fn reschema(input: &Batch, float32: bool) -> Batch {
    use datafusion::arrow::{
        array::Float32Array,
        datatypes::{Field, Schema},
    };
    let record = &input.table_payload().unwrap().batches()[0];
    let mut columns = record.columns().to_vec();
    let mut fields = record
        .schema()
        .fields()
        .iter()
        .map(|field| field.as_ref().clone())
        .collect::<Vec<_>>();
    if float32 {
        columns[0] = std::sync::Arc::new(Float32Array::from(vec![Some(12.0), None]));
        fields[0] = Field::new(
            "Price",
            datafusion::arrow::datatypes::DataType::Float32,
            true,
        );
    }
    fields[0] = fields[0]
        .clone()
        .with_metadata(std::collections::HashMap::from([(
            "unit".into(),
            "test".into(),
        )]));
    let schema = std::sync::Arc::new(Schema::new_with_metadata(
        fields,
        std::collections::HashMap::from([("origin".into(), "changed".into())]),
    ));
    let record = datafusion::arrow::record_batch::RecordBatch::try_new(schema, columns).unwrap();
    Batch::table(vec![record], input.metadata().clone()).unwrap()
}

async fn chunks_and_schema_changes() {
    let job = job(CancellationToken::new());
    for filter in [None, Some(r#""Price" > 15"#)] {
        let mut operator = operator(
            &[r#""Price" + 1 AS next"#, "upper(symbol) AS ticker"],
            filter,
        );
        operator.set_stream_resources(
            DataFusionConfig {
                batch_size: 2,
                ..DataFusionConfig::default()
            },
            UdfRegistrySnapshot::default(),
            vec![],
        );
        let mut reference = operator.clone();
        compare(&mut operator, &mut reference, shifted_input(0.0), &job).await;
        compare(&mut operator, &mut reference, chunks(), &job).await;
        let empty = Batch::table(
            vec![datafusion::arrow::record_batch::RecordBatch::new_empty(
                shifted_input(0.0).table_payload().unwrap().schema().clone(),
            )],
            BatchMetadata::default(),
        )
        .unwrap();
        compare(&mut operator, &mut reference, empty, &job).await;
        for float32 in [false, false, true, true] {
            compare(
                &mut operator,
                &mut reference,
                reschema(&shifted_input(0.0), float32),
                &job,
            )
            .await;
        }
        assert_eq!(
            operator
                .stream_state
                .runtime
                .as_ref()
                .unwrap()
                .metrics()
                .len(),
            3
        );
        assert!(!operator.clone().stream_runtime_initialized());
    }
}

async fn constants_preserve_zero_column_input_rows() {
    use datafusion::arrow::{
        datatypes::Schema,
        record_batch::{RecordBatch, RecordBatchOptions},
    };
    let schema = std::sync::Arc::new(Schema::empty());
    let record = RecordBatch::try_new_with_options(
        schema,
        vec![],
        &RecordBatchOptions::new().with_row_count(Some(3)),
    )
    .unwrap();
    let input = Batch::table(
        vec![record],
        BatchMetadata::new("empty-schema", 11, JsonMap::new()).unwrap(),
    )
    .unwrap();
    let cursor =
        std::sync::Arc::new(crate::Cursor::new("empty-schema", vec![11], JsonMap::new()).unwrap());
    let input = input.with_source_cursor(Some(cursor.clone()));
    let job = job(CancellationToken::new());
    let mut operator = operator(&["42 AS answer"], None);
    let mut reference = operator.clone();
    for _ in 0..2 {
        compare(&mut operator, &mut reference, input.clone(), &job).await;
    }
    assert_eq!(input.num_rows(), 3);
    assert!(
        input
            .source_cursor()
            .is_some_and(|actual| std::sync::Arc::ptr_eq(&actual, &cursor))
    );
    assert_eq!(
        operator
            .stream_state
            .runtime
            .as_ref()
            .unwrap()
            .metrics()
            .len(),
        1
    );
}

async fn predicate_column_is_removed_after_filtering() {
    let job = job(CancellationToken::new());
    let mut operator = operator(&["upper(symbol) AS ticker"], Some(r#""Price" > 15"#));
    let mut reference = operator.clone();
    for shift in [20.0, 0.0, 40.0] {
        compare(&mut operator, &mut reference, shifted_input(shift), &job).await;
    }
    let metrics = operator.stream_state.runtime.as_ref().unwrap().metrics();
    assert_eq!(metrics.len(), 1);
    let physical = &metrics[0].physical_plan;
    assert!(
        physical
            .lines()
            .any(|line| line.contains("FilterExec") && line.contains("projection=[symbol@")),
        "{physical}"
    );
}

fn buffers(input: &Batch) -> Vec<datafusion::arrow::buffer::Buffer> {
    input
        .table_payload()
        .unwrap()
        .batches()
        .iter()
        .flat_map(|record| {
            record.columns().iter().flat_map(|column| {
                column
                    .to_data()
                    .buffers()
                    .iter()
                    .filter(|buffer| !buffer.is_empty())
                    .cloned()
                    .collect::<Vec<_>>()
            })
        })
        .collect()
}

async fn retained_input_buffers_are_released() {
    let job = job(CancellationToken::new());
    let mut operator = operator(&[r#""Price" + 1 AS next"#], None);
    for shift in [0.0, 20.0] {
        let input = shifted_input(shift);
        let owners = buffers(&input);
        drop(collect(&mut operator, input, &job).await.unwrap());
        assert!(owners.iter().all(|buffer| buffer.strong_count() == 1));
    }
    assert_eq!(
        operator
            .stream_state
            .runtime
            .as_ref()
            .unwrap()
            .metrics()
            .len(),
        1
    );
}

fn string_input(values: &[&str]) -> Batch {
    let input = shifted_input(0.0);
    let record = &input.table_payload().unwrap().batches()[0];
    let record = datafusion::arrow::record_batch::RecordBatch::try_new(
        record.schema(),
        vec![
            record.column(0).clone(),
            std::sync::Arc::new(datafusion::arrow::array::StringArray::from(values.to_vec())),
        ],
    )
    .unwrap();
    Batch::table(vec![record], input.metadata().clone()).unwrap()
}

async fn evaluation_errors_leave_the_cache_usable() {
    let job = job(CancellationToken::new());
    let mut operator = operator(&["CAST(symbol AS BIGINT) AS parsed"], None);
    let mut reference = operator.clone();
    compare(
        &mut operator,
        &mut reference,
        string_input(&["10", "20"]),
        &job,
    )
    .await;
    let input = string_input(&["bad", "20"]);
    let owners = buffers(&input);
    let expected = reference
        .process_standalone(input.clone())
        .await
        .unwrap_err();
    let actual = collect(&mut operator, input, &job).await.unwrap_err();
    assert_eq!(actual.to_string(), expected.to_string());
    assert!(owners.iter().all(|buffer| buffer.strong_count() == 1));
    compare(
        &mut operator,
        &mut reference,
        string_input(&["30", "40"]),
        &job,
    )
    .await;
    assert_eq!(
        operator
            .stream_state
            .runtime
            .as_ref()
            .unwrap()
            .metrics()
            .len(),
        1
    );
}

async fn unsupported_queries_keep_sql_execution() {
    let job = job(CancellationToken::new());
    for select in ["SUM(\"Price\") AS total", "row_number() OVER () AS ordinal"] {
        let mut operator = operator(&[select], None);
        let mut reference = operator.clone();
        for shift in [0.0, 20.0] {
            compare(&mut operator, &mut reference, shifted_input(shift), &job).await;
        }
        assert_eq!(
            operator
                .stream_state
                .runtime
                .as_ref()
                .unwrap()
                .metrics()
                .len(),
            2
        );
    }
    let mut operator = operator(&["now() AS stamp"], None);
    for shift in [0.0, 20.0] {
        assert_eq!(
            collect(&mut operator, shifted_input(shift), &job)
                .await
                .unwrap()
                .num_rows(),
            2
        );
    }
    assert_eq!(
        operator
            .stream_state
            .runtime
            .as_ref()
            .unwrap()
            .metrics()
            .len(),
        2
    );
}

async fn selected_udfs_keep_sql_execution() {
    use datafusion::logical_expr::{Volatility, create_udf};
    let reference =
        UdfReference::new("rust", "trusted", "1", crate::UdfKind::DataFusionScalar).unwrap();
    let udf = std::sync::Arc::new(create_udf(
        "trusted",
        vec![datafusion::arrow::datatypes::DataType::Float64],
        datafusion::arrow::datatypes::DataType::Float64,
        Volatility::Immutable,
        std::sync::Arc::new(|arguments| Ok(arguments[0].clone())),
    ));
    let mut registry = crate::UdfRegistry::new();
    registry
        .register_datafusion(reference.clone(), udf, 1)
        .unwrap();
    let job = job(CancellationToken::new());
    for declared in [false, true] {
        let mut operator = operator(&["trusted(\"Price\") AS value"], None);
        if declared {
            operator.udfs.push(reference.clone());
        }
        operator.set_stream_resources(
            DataFusionConfig::default(),
            registry.snapshot(),
            vec![reference.clone()],
        );
        let mut expected = operator.clone();
        for shift in [0.0, 20.0] {
            compare(&mut operator, &mut expected, shifted_input(shift), &job).await;
        }
        assert_eq!(
            operator
                .stream_state
                .runtime
                .as_ref()
                .unwrap()
                .metrics()
                .len(),
            2
        );
    }
}

async fn cancellation_and_abandonment_release_call_state() {
    for abandon in [false, true] {
        let token = CancellationToken::new();
        let job = job(token.clone());
        let mut operator = operator(&[r#""Price" + 1 AS next"#], Some(r#""Price" > 5"#));
        drop(
            collect(&mut operator, shifted_input(0.0), &job)
                .await
                .unwrap(),
        );
        let input = chunks();
        let owners = buffers(&input);
        stop_call(&mut operator, input, &job, &token, abandon).await;
        assert!(owners.iter().all(|buffer| buffer.strong_count() == 1));
        let retry_job = self::job(CancellationToken::new());
        let mut reference = operator.clone();
        compare(
            &mut operator,
            &mut reference,
            shifted_input(20.0),
            &retry_job,
        )
        .await;
        assert_eq!(
            operator
                .stream_state
                .runtime
                .as_ref()
                .unwrap()
                .metrics()
                .len(),
            1
        );
    }
}

async fn stop_call(
    operator: &mut ExpressionOperator,
    input: Batch,
    job: &StreamJobContext,
    token: &CancellationToken,
    abandon: bool,
) {
    if abandon {
        let context = StreamOperatorContext::new(job, "compiled", None);
        let mut output = EdgeCollector::new(operator.output_ports().to_vec());
        let mut future = Box::pin(operator.process_data("input", input, &context, &mut output));
        assert!(matches!(
            futures::poll!(future.as_mut()),
            std::task::Poll::Pending
        ));
        drop(future);
        assert!(output.drain("output").is_empty());
    } else {
        let cancel = async {
            tokio::task::yield_now().await;
            token.cancel();
        };
        let ((), result) = tokio::join!(cancel, collect(operator, input, job));
        assert!(matches!(result, Err(CalcFlowError::Cancelled { .. })));
    }
}
