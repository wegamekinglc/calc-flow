use std::{collections::BTreeMap, sync::Arc};

use calc_flow::{
    Batch, BatchMetadata, CompileMode, CompileOptions, ExecutionOptions, InputOptions,
    NullPlacement, Program, Result, SortDirection, SortKey, SortSpec, StateBudget, UdfRegistry,
    arrow::{
        array::Int64Array,
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    },
    table_input,
};

#[tokio::test]
async fn test_order_sequence_recovery_emits_one_final() -> Result<()> {
    use calc_flow::{
        BoundedOrderOperator, CancellationToken, EdgeCollector, Epoch, JsonMap, OperatorMetadata,
        StreamJobContext, StreamOperator, StreamOperatorContext,
    };
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Int64, true),
        Field::new("seq", DataType::Int64, false),
    ]));
    let make = || {
        BoundedOrderOperator::new(
            "order",
            schema.clone(),
            vec![SortKey::new(
                "x",
                SortDirection::Ascending,
                NullPlacement::Last,
            )?],
            Some(1),
            SortSpec::bounded_source(StateBudget::new(8, 8 << 20)?)?
                .with_sequence_by(vec!["seq".into()])?
                .with_offset(1)?,
        )
    };
    let data = |x, seq| {
        Batch::table(
            vec![
                RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(Int64Array::from(x)),
                        Arc::new(Int64Array::from(seq)),
                    ],
                )
                .unwrap(),
            ],
            BatchMetadata::default(),
        )
    };
    let job = StreamJobContext::new(1, "order", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "order", None);
    let mut op = make()?;
    let mut collector = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("input", data(vec![1], vec![20])?, &context, &mut collector)
        .await?;
    assert!(collector.drain("output").is_empty());
    let checkpoint = op.checkpoint(Epoch::INITIAL)?;
    assert_eq!(checkpoint.inline_metadata["state_kind"], "bounded_order_v1");
    let mut resumed = make()?;
    resumed.restore(&checkpoint)?;
    resumed
        .process_data(
            "input",
            data(vec![1, 2], vec![10, 40])?,
            &context,
            &mut collector,
        )
        .await?;
    assert!(collector.drain("output").is_empty());
    resumed.on_end(&context, &mut collector).await?;
    let out = collector.drain("output");
    assert_eq!(out.len(), 1);
    assert_eq!(
        out[0].as_data().unwrap().table_payload()?.batches()[0]
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0),
        20
    );
    let terminal = resumed.checkpoint(Epoch::INITIAL)?;
    let mut closed = make()?;
    closed.restore(&terminal)?;
    closed.on_end(&context, &mut collector).await?;
    assert!(collector.drain("output").is_empty());
    Ok(())
}

#[tokio::test]
async fn test_order_duplicate_sequence_rejects_whole_envelope_without_commit() -> Result<()> {
    use calc_flow::{
        BoundedOrderOperator, CancellationToken, EdgeCollector, Epoch, JsonMap, OperatorMetadata,
        StreamJobContext, StreamOperator, StreamOperatorContext,
    };
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Int64, true),
        Field::new("seq", DataType::Int64, false),
    ]));
    let record = |x, seq| {
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![x])),
                Arc::new(Int64Array::from(vec![seq])),
            ],
        )
        .unwrap()
    };
    let mut op = BoundedOrderOperator::new(
        "order",
        schema.clone(),
        vec![SortKey::new(
            "x",
            SortDirection::Ascending,
            NullPlacement::Last,
        )?],
        Some(1),
        SortSpec::bounded_source(StateBudget::new(8, 8 << 20)?)?
            .with_sequence_by(vec!["seq".into()])?,
    )?;
    let job = StreamJobContext::new(1, "order", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "order", None);
    let mut collector = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data(
        "input",
        Batch::table(vec![record(3, 10)], BatchMetadata::default())?,
        &context,
        &mut collector,
    )
    .await?;
    let before = op.checkpoint(Epoch::INITIAL)?;
    let invalid = Batch::table(vec![record(1, 11), record(2, 10)], BatchMetadata::default())?;
    let error = op
        .process_data("input", invalid, &context, &mut collector)
        .await;
    assert!(
        error.is_err(),
        "a sequence repeated across Data must reject the whole two-record envelope"
    );
    assert!(error.unwrap_err().to_string().contains("unique"));
    let after = op.checkpoint(Epoch::INITIAL)?;
    assert_eq!(after.inline_metadata, before.inline_metadata);
    assert_eq!(after.segments, before.segments);
    op.on_end(&context, &mut collector).await?;
    let out = collector.drain("output");
    assert_eq!(
        out[0].as_data().unwrap().table_payload()?.batches()[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0),
        3
    );
    Ok(())
}

#[tokio::test]
async fn test_order_sql_df_compile_both_modes_and_export() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Int64, true),
        Field::new("seq", DataType::Int64, false),
    ]));
    let input = table_input("events", schema, InputOptions::default())?;
    let sql = calc_flow::sql::sql(
        "SELECT * FROM CF_ORDER_BOUNDED('a',10,8388608,ARRAY['seq']) ORDER BY x ASC NULLS LAST LIMIT 1 OFFSET 1",
        BTreeMap::from([("a".into(), input.clone())]),
    )?;
    let df = input.top_k(
        1,
        vec![SortKey::new(
            "x",
            SortDirection::Ascending,
            NullPlacement::Last,
        )?],
        SortSpec::bounded_source(StateBudget::new(10, 8 << 20)?)?
            .with_offset(1)?
            .with_sequence_by(vec!["seq".into()])?,
    )?;
    let registry = UdfRegistry::new().snapshot();
    let mut configs = vec![];
    for out in [df, sql] {
        let program = Program::new(
            "both",
            CompileMode::Batch,
            BTreeMap::from([("out".into(), out)]),
        )?;
        program
            .compile_batch(&registry, &CompileOptions::default())
            .await?;
        program
            .for_mode(CompileMode::Stream)?
            .compile_stream(
                &registry,
                &calc_flow::StreamRequirements::default(),
                &CompileOptions::default(),
            )
            .await?;
        let project = serde_json::to_value(program.to_project(&registry).await?).unwrap();
        let node = project["graph"]["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|n| n["operator"]["kind"] == "bounded_order")
            .unwrap();
        configs.push(node["operator"].clone());
    }
    assert_eq!(configs[0], configs[1]);
    assert_eq!(configs[0]["fetch"], 1);
    Ok(())
}

#[tokio::test]
async fn test_order_native_gate_rejects_snapshot_and_rewritten_sequence() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Int64, true),
        Field::new("seq", DataType::Int64, false),
    ]));
    let input = table_input("events", schema, InputOptions::default())?;
    for upstream in [
        input.sql("SELECT SUM(x) AS x, MAX(seq) AS seq FROM input")?,
        input.sql("SELECT x, seq+1 AS seq FROM input")?,
    ] {
        let out = upstream.top_k(
            1,
            vec![SortKey::new(
                "x",
                SortDirection::Ascending,
                NullPlacement::Last,
            )?],
            SortSpec::bounded_source(StateBudget::new(10, 8 << 20)?)?
                .with_sequence_by(vec!["seq".into()])?,
        )?;
        let program = Program::new(
            "reject",
            CompileMode::Batch,
            BTreeMap::from([("out".into(), out)]),
        )?;
        assert!(
            program
                .compile_batch(&UdfRegistry::new().snapshot(), &CompileOptions::default())
                .await
                .is_err(),
            "snapshot or rewritten sequence must fail mandatory native admission"
        );
    }
    Ok(())
}

#[tokio::test]
async fn test_order_millisecond_window_copies_and_checkpoints_late_drop() -> Result<()> {
    use calc_flow::arrow::{array::TimestampMillisecondArray, datatypes::TimeUnit};
    use calc_flow::{
        BoundedOrderOperator, CancellationToken, EdgeCollector, Epoch, EventTime, JsonMap,
        LatePolicySpec, OperatorMetadata, StreamJobContext, StreamOperator, StreamOperatorContext,
    };
    let schema = Arc::new(Schema::new(vec![
        Field::new("ts", DataType::Timestamp(TimeUnit::Millisecond, None), true),
        Field::new("x", DataType::Int64, true),
    ]));
    let make = || {
        BoundedOrderOperator::new(
            "order",
            schema.clone(),
            vec![SortKey::new(
                "x",
                SortDirection::Ascending,
                NullPlacement::Last,
            )?],
            Some(1),
            SortSpec::closed_window(
                calc_flow::SortWindow::new(
                    "ts",
                    EventTime::from_micros(0),
                    EventTime::from_micros(10_000),
                )?,
                StateBudget::new(2, 8 << 20)?,
            )?
            .with_late_policy(LatePolicySpec::Drop { metrics_version: 1 })?,
        )
    };
    let data = |ts: Vec<Option<i64>>, x| {
        Batch::table(
            vec![
                RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(TimestampMillisecondArray::from(ts)),
                        Arc::new(Int64Array::from(x)),
                    ],
                )
                .unwrap(),
            ],
            BatchMetadata::default(),
        )
    };
    let mut op = make()?;
    let job = StreamJobContext::new(1, "order", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "order", None);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data(
        "input",
        data(vec![Some(0), Some(9)], vec![3, 1])?,
        &context,
        &mut output,
    )
    .await?;
    op.on_watermark(EventTime::from_micros(9_999), &context, &mut output)
        .await?;
    assert!(output.drain("output").is_empty());
    op.on_watermark(EventTime::from_micros(10_000), &context, &mut output)
        .await?;
    let final_rows = output.drain("output");
    assert_eq!(final_rows.len(), 1);
    assert_eq!(
        final_rows[0].as_data().unwrap().table_payload()?.batches()[0]
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0),
        1
    );
    op.process_data(
        "input",
        data(vec![Some(0), Some(9), Some(10), None], vec![0, -1, 5, 6])?,
        &context,
        &mut output,
    )
    .await?;
    let snapshot = op.checkpoint(Epoch::INITIAL)?;
    assert_eq!(snapshot.inline_metadata["state_layout"], 2);
    let control: serde_json::Value =
        serde_json::from_slice(snapshot.segments["control"].bytes()).unwrap();
    assert_eq!(control["accepted_rows"], 2);
    assert_eq!(control["late_metrics"]["late_rows"], 2);
    assert_eq!(control["late_metrics"]["affected_batches"], 1);
    assert_eq!(control["late_metrics"]["max_lateness_micros"], 10_000);
    let mut resumed = make()?;
    resumed.restore(&snapshot)?;
    resumed.on_end(&context, &mut output).await?;
    assert!(output.drain("output").is_empty());
    Ok(())
}

#[tokio::test]
async fn test_order_full_history_budget_is_typed_and_atomic_even_for_top_one() -> Result<()> {
    use calc_flow::{
        BoundedOrderOperator, CalcFlowError, CancellationToken, EdgeCollector, Epoch, JsonMap,
        OperatorMetadata, StreamJobContext, StreamOperator, StreamOperatorContext,
        StreamingFailureReason,
    };
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
    let mut op = BoundedOrderOperator::new(
        "order",
        schema.clone(),
        vec![SortKey::new(
            "x",
            SortDirection::Ascending,
            NullPlacement::Last,
        )?],
        Some(1),
        SortSpec::bounded_source(StateBudget::new(1, 8 << 20)?)?,
    )?;
    let job = StreamJobContext::new(1, "order", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "order", None);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    let raw = Batch::table(
        vec![RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![3, 1]))]).unwrap()],
        BatchMetadata::default(),
    )?;
    let error = op
        .process_data("input", raw, &context, &mut output)
        .await
        .unwrap_err();
    assert!(
        matches!(
            error,
            CalcFlowError::OperatorReason {
                reason_code: StreamingFailureReason::StateBudgetExceeded,
                ..
            }
        ),
        "{error}"
    );
    let snapshot = op.checkpoint(Epoch::INITIAL)?;
    let control: serde_json::Value =
        serde_json::from_slice(snapshot.segments["control"].bytes()).unwrap();
    assert_eq!(control["accepted_rows"], 0);
    assert_eq!(control["closed"], false);
    assert!(output.drain("output").is_empty());
    Ok(())
}

fn tie_fixture() -> Result<(Arc<Schema>, Batch)> {
    use calc_flow::arrow::array::StringArray;
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Int64, true),
        Field::new("seq", DataType::Int64, false),
        Field::new("label", DataType::Utf8, false),
    ]));
    let batch = Batch::table(
        vec![
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int64Array::from(vec![Some(1), None])),
                    Arc::new(Int64Array::from(vec![20, 30])),
                    Arc::new(StringArray::from(vec!["b", "n"])),
                ],
            )
            .unwrap(),
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int64Array::from(vec![1, 2])),
                    Arc::new(Int64Array::from(vec![10, 40])),
                    Arc::new(StringArray::from(vec!["a", "c"])),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )?;
    Ok((schema, batch))
}

#[tokio::test]
async fn test_order_nulls_stable_ties_and_slice_boundaries() -> Result<()> {
    use calc_flow::arrow::array::StringArray;
    let (schema, batch) = tie_fixture()?;
    let t = table_input("events", schema.clone(), InputOptions::default())?;
    for (direction, nulls, k, offset, expected) in [
        (
            SortDirection::Ascending,
            NullPlacement::Last,
            None,
            0,
            vec!["a", "b", "c", "n"],
        ),
        (
            SortDirection::Descending,
            NullPlacement::First,
            None,
            0,
            vec!["n", "c", "a", "b"],
        ),
        (
            SortDirection::Ascending,
            NullPlacement::Last,
            Some(1),
            1,
            vec!["b"],
        ),
        (
            SortDirection::Ascending,
            NullPlacement::Last,
            Some(0),
            0,
            vec![],
        ),
        (
            SortDirection::Ascending,
            NullPlacement::Last,
            Some(10),
            4,
            vec![],
        ),
        (
            SortDirection::Ascending,
            NullPlacement::Last,
            None,
            7,
            vec![],
        ),
    ] {
        let spec = SortSpec::bounded_source(StateBudget::new(4, 8 << 20)?)?
            .with_offset(offset)?
            .with_sequence_by(vec!["seq".into()])?;
        let by = vec![SortKey::new("x", direction, nulls)?];
        let out = if let Some(k) = k {
            t.top_k(k, by, spec)?
        } else {
            t.sort(by, spec)?
        };
        let p = Program::new(
            "ties",
            CompileMode::Batch,
            BTreeMap::from([("out".into(), out)]),
        )?;
        let plan = p
            .compile_batch(&UdfRegistry::new().snapshot(), &CompileOptions::default())
            .await?;
        let result = plan
            .execute(
                BTreeMap::from([("events".into(), batch.clone())]),
                ExecutionOptions::default(),
            )
            .await?;
        assert_eq!(result.outputs["out"].table_payload()?.schema(), &schema);
        let actual = result.outputs["out"]
            .table_payload()?
            .batches()
            .iter()
            .flat_map(|r| {
                r.column(2)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .iter()
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected.into_iter().map(Some).collect::<Vec<_>>());
    }
    Ok(())
}

#[tokio::test]
async fn test_order_export_reload_postprocessing_physically_preserves_order() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Int64, true),
        Field::new("seq", DataType::Int64, false),
    ]));
    let t = table_input("events", schema.clone(), InputOptions::default())?;
    let sorted = t.sort(
        vec![SortKey::new(
            "x",
            SortDirection::Ascending,
            NullPlacement::Last,
        )?],
        SortSpec::bounded_source(StateBudget::new(100, 8 << 20)?)?
            .with_sequence_by(vec!["seq".into()])?,
    )?;
    let post = sorted
        .sql("SELECT x, seq, x+1 AS y FROM input")?
        .sql("SELECT x, seq FROM input WHERE y>0")?;
    let program = Program::new(
        "post",
        CompileMode::Batch,
        BTreeMap::from([("out".into(), post)]),
    )?;
    let registry = UdfRegistry::new().snapshot();
    let mut project = program.to_project(&registry).await?;
    project.graph.datafusion = calc_flow::DataFusionConfig {
        parallelism_mode: calc_flow::DataFusionParallelismMode::Fixed,
        target_partitions: 8,
        max_partitions: 8,
        batch_size: 2,
        ..Default::default()
    };
    let project: calc_flow::ProjectSpec =
        serde_json::from_value(serde_json::to_value(project).unwrap()).unwrap();
    let plan =
        calc_flow::compile_project(&project, &calc_flow::ProviderRegistry::default(), &registry)?;
    let physical = plan.single_external_input()?.to_owned();
    let batch = Batch::table(
        vec![
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(Int64Array::from((0..50).rev().collect::<Vec<_>>())),
                    Arc::new(Int64Array::from((0..50).collect::<Vec<_>>())),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )?;
    let result = plan
        .execute(
            BTreeMap::from([(physical, batch)]),
            ExecutionOptions::default(),
        )
        .await?;
    let out = result.outputs.values().next().unwrap();
    let actual = out
        .table_payload()?
        .batches()
        .iter()
        .flat_map(|r| {
            r.column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values()
                .iter()
                .copied()
        })
        .collect::<Vec<_>>();
    assert_eq!(actual, (0..50).collect::<Vec<_>>());
    Ok(())
}

struct RefuseFinal;
#[async_trait::async_trait]
impl calc_flow::StreamCollector for RefuseFinal {
    async fn emit(&mut self, _: &str, _: Batch) -> Result<()> {
        Err(calc_flow::CalcFlowError::Internal {
            message: "refused final output".into(),
        })
    }
}

#[tokio::test]
async fn test_order_borrows_foreign_ingress_only_during_copy_and_failed_final_is_atomic()
-> Result<()> {
    use calc_flow::arrow::array::Array;
    use calc_flow::{
        BoundedOrderOperator, CancellationToken, EdgeCollector, Epoch, JsonMap, OperatorMetadata,
        StreamJobContext, StreamOperator, StreamOperatorContext,
    };
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
    let mut op = BoundedOrderOperator::new(
        "order",
        schema.clone(),
        vec![SortKey::new(
            "x",
            SortDirection::Ascending,
            NullPlacement::Last,
        )?],
        Some(1),
        SortSpec::bounded_source(StateBudget::new(2, 8 << 20)?)?,
    )?;
    let cancellation = CancellationToken::new();
    let job = StreamJobContext::new(1, "order", JsonMap::new(), None, cancellation.clone());
    let context = StreamOperatorContext::new(&job, "order", None);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    let source = Arc::new(Int64Array::from(vec![3; 1_048_576]));
    let observed = source.values().inner().clone();
    let slice = Array::slice(source.as_ref(), 1, 1);
    let raw = Batch::table(
        vec![RecordBatch::try_new(schema, vec![slice]).unwrap()],
        BatchMetadata::default(),
    )?;
    drop(source);
    op.process_data("input", raw, &context, &mut output).await?;
    assert_eq!(
        observed.strong_count(),
        1,
        "only the external observer may retain the foreign parent allocation"
    );
    let before = op.checkpoint(Epoch::INITIAL)?;
    let error = op.on_end(&context, &mut RefuseFinal).await.unwrap_err();
    assert!(error.to_string().contains("refused final output"));
    let after = op.checkpoint(Epoch::INITIAL)?;
    assert_eq!(after.inline_metadata, before.inline_metadata);
    assert_eq!(after.segments, before.segments);
    cancellation.cancel();
    assert!(op.on_end(&context, &mut output).await.is_err());
    assert!(output.drain("output").is_empty());
    let cancelled = op.checkpoint(Epoch::INITIAL)?;
    assert_eq!(cancelled.segments, before.segments);
    Ok(())
}

#[tokio::test]
async fn test_order_finite_top_k_and_same_plan_isolation() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
    let input = table_input("events", schema.clone(), InputOptions::default())?;
    let out = input.top_k(
        1,
        vec![SortKey::new(
            "x",
            SortDirection::Ascending,
            NullPlacement::Last,
        )?],
        SortSpec::bounded_source(StateBudget::new(10, 8 << 20)?)?,
    )?;
    let program = Program::new(
        "ordered",
        CompileMode::Batch,
        BTreeMap::from([("out".into(), out)]),
    )?;
    let plan = program
        .compile_batch(&UdfRegistry::new().snapshot(), &CompileOptions::default())
        .await?;
    for (values, expected) in [
        (vec![3, 1, 2], vec![1]),
        (vec![9], vec![9]),
        (vec![], vec![]),
    ] {
        let batch = Batch::table(
            vec![
                RecordBatch::try_new(schema.clone(), vec![Arc::new(Int64Array::from(values))])
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
        let actual = result.outputs["out"]
            .table_payload()?
            .batches()
            .iter()
            .flat_map(|r| {
                r.column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .iter()
                    .copied()
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }
    Ok(())
}
