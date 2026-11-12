use calc_flow::arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use calc_flow::{
    CompileMode, InputOptions, JoinStateLimits, JoinTimeBounds, Program, Result, StreamJoinSpec,
    table_input,
};
use std::{collections::BTreeMap, sync::Arc, time::Duration};

#[tokio::test]
async fn typed_and_sql_inner_join_export_and_compile_both_modes() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("seq", DataType::UInt64, false),
    ]));
    let options = InputOptions {
        entity_by: vec!["k".into()],
        event_time: Some("ts".into()),
        sequence_by: vec!["seq".into()],
    };
    let a = table_input("a", schema.clone(), options.clone())?;
    let batch_schema = schema.clone();
    let b = table_input("b", schema, options)?;
    let spec = StreamJoinSpec::inner(
        ["k"],
        ["k"],
        "ts",
        "ts",
        JoinTimeBounds::new(Duration::from_micros(2), Duration::from_micros(3))?,
        JoinStateLimits::new(100, 1_048_576, 1000)?,
    )?;
    let config = serde_json::json!({"spec": &spec});
    assert!(
        a.inner_join(
            &b,
            spec.clone(),
            Some(calc_flow::JoinOutputOrdering {
                entity_by: vec![],
                event_time: "left__ts".into(),
                sequence_by: vec![],
            })
        )
        .is_err()
    );
    let df = a.inner_join(&b, spec, None)?;
    let sq = calc_flow::sql::sql(
        &format!("SELECT * FROM CF_INNER_JOIN('a', 'b', '{config}')"),
        BTreeMap::from([("a".into(), a), ("b".into(), b)]),
    )?;
    let udfs = calc_flow::UdfRegistry::new().snapshot();
    let p = Program::new(
        "joins",
        CompileMode::Batch,
        BTreeMap::from([("df".into(), df), ("sql".into(), sq)]),
    )?;
    assert_eq!(p.collect_schema(&udfs).await?.len(), 2);
    let project = p.to_project(&udfs).await?;
    assert_eq!(
        project
            .graph
            .nodes
            .iter()
            .filter(|n| matches!(n.operator, calc_flow::OperatorSpec::StreamJoin { .. }))
            .count(),
        2
    );
    let plan = p
        .compile_batch(&udfs, &calc_flow::CompileOptions::default())
        .await?;
    let result = plan
        .execute(
            BTreeMap::from([
                ("a".into(), data(batch_schema.clone(), &[10, 10])?),
                ("b".into(), data(batch_schema.clone(), &[7, 8, 8, 13, 14])?),
            ]),
            calc_flow::ExecutionOptions::default(),
        )
        .await?;
    for output in result.outputs.values() {
        assert_eq!(
            pairs(std::slice::from_ref(output)),
            vec![(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3)]
        );
    }

    let stream_plan = p
        .for_mode(CompileMode::Stream)?
        .compile_stream(
            &udfs,
            &calc_flow::StreamRequirements::default(),
            &calc_flow::CompileOptions::default(),
        )
        .await?;
    stream::verify(
        stream_plan,
        BTreeMap::from([
            ("a".into(), data(batch_schema.clone(), &[10, 10])?),
            ("b".into(), data(batch_schema, &[7, 8, 8, 13, 14])?),
        ]),
    )
    .await?;
    Ok(())
}

fn data(schema: Arc<Schema>, times: &[i64]) -> Result<calc_flow::Batch> {
    use calc_flow::arrow::{
        array::{Int64Array, TimestampMicrosecondArray, UInt64Array},
        record_batch::RecordBatch,
    };
    calc_flow::Batch::table(
        vec![
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(Int64Array::from(vec![1; times.len()])),
                    Arc::new(TimestampMicrosecondArray::from(times.to_vec()).with_timezone("UTC")),
                    Arc::new(UInt64Array::from_iter_values(0..times.len() as u64)),
                ],
            )
            .unwrap(),
        ],
        calc_flow::BatchMetadata::default(),
    )
}

#[path = "finite_inner_frontend/stream.rs"]
mod stream;
fn pairs(batches: &[calc_flow::Batch]) -> Vec<(u64, u64)> {
    use calc_flow::arrow::array::UInt64Array;
    let mut pairs = Vec::new();
    for batch in batches {
        for record in batch.table_payload().unwrap().batches() {
            let left = record
                .column_by_name("left__seq")
                .unwrap()
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            let right = record
                .column_by_name("right__seq")
                .unwrap()
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            pairs.extend(
                left.values()
                    .iter()
                    .copied()
                    .zip(right.values().iter().copied()),
            );
        }
    }
    pairs.sort_unstable();
    pairs
}

#[tokio::test]
async fn public_join_rejects_incomplete_sequence_and_millisecond_keys() -> Result<()> {
    for millisecond_key in [false, true] {
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "k",
                if millisecond_key {
                    DataType::Timestamp(TimeUnit::Millisecond, None)
                } else {
                    DataType::Int64
                },
                false,
            ),
            Field::new(
                "ts",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("seq", DataType::UInt64, false),
        ]));
        let options = InputOptions {
            entity_by: vec!["seq".into()],
            event_time: Some("ts".into()),
            sequence_by: if millisecond_key {
                vec!["seq".into()]
            } else {
                vec![]
            },
        };
        let a = table_input("a", schema.clone(), options.clone())?;
        let b = table_input("b", schema, options)?;
        let spec = StreamJoinSpec::inner(
            ["k"],
            ["k"],
            "ts",
            "ts",
            JoinTimeBounds::new(Duration::ZERO, Duration::ZERO)?,
            JoinStateLimits::new(100, 1_000_000, 100)?,
        )?;
        let p = Program::new(
            "invalid",
            CompileMode::Batch,
            BTreeMap::from([("pairs".into(), a.inner_join(&b, spec, None)?)]),
        )?;
        assert!(
            p.collect_schema(&calc_flow::UdfRegistry::new().snapshot())
                .await
                .is_err()
        );
    }
    Ok(())
}

#[tokio::test]
async fn typed_nested_join_shares_first_owner_and_preserves_passthrough() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("seq", DataType::UInt64, false),
    ]));
    let options = InputOptions {
        entity_by: vec!["k".into()],
        event_time: Some("ts".into()),
        sequence_by: vec!["seq".into()],
    };
    let a = table_input("a", schema.clone(), options.clone())?;
    let b = table_input("b", schema.clone(), options.clone())?;
    let c = table_input("c", schema.clone(), options)?;
    let spec = StreamJoinSpec::inner(
        ["k"],
        ["k"],
        "ts",
        "ts",
        JoinTimeBounds::new(Duration::from_micros(2), Duration::from_micros(3))?,
        JoinStateLimits::new(100, 1_000_000, 100)?,
    )?;
    let first = a.sql("SELECT * FROM input")?.inner_join(
        &b,
        spec,
        Some(calc_flow::JoinOutputOrdering {
            entity_by: vec!["left__k".into()],
            event_time: "left__ts".into(),
            sequence_by: vec!["left__seq".into(), "right__seq".into()],
        }),
    )?;
    let second = first.sql("SELECT * FROM input")?.inner_join(
        &c,
        StreamJoinSpec::inner(
            ["left__k"],
            ["k"],
            "left__ts",
            "ts",
            JoinTimeBounds::new(Duration::ZERO, Duration::ZERO)?,
            JoinStateLimits::new(100, 1_000_000, 100)?,
        )?,
        None,
    )?;
    let p = Program::new(
        "nested",
        CompileMode::Batch,
        BTreeMap::from([("first".into(), first), ("second".into(), second)]),
    )?;
    let udfs = calc_flow::UdfRegistry::new().snapshot();
    let project = p.to_project(&udfs).await?;
    assert_eq!(
        project
            .graph
            .nodes
            .iter()
            .filter(|n| matches!(n.operator, calc_flow::OperatorSpec::StreamJoin { .. }))
            .count(),
        2
    );
    let result = p
        .compile_batch(&udfs, &calc_flow::CompileOptions::default())
        .await?
        .execute(
            BTreeMap::from([
                ("a".into(), data(schema.clone(), &[10, 10])?),
                ("b".into(), data(schema.clone(), &[8, 8, 13])?),
                ("c".into(), data(schema, &[10])?),
            ]),
            calc_flow::ExecutionOptions::default(),
        )
        .await?;
    assert_eq!(result.outputs["first"].num_rows(), 6);
    assert_eq!(result.outputs["second"].num_rows(), 6);
    p.for_mode(CompileMode::Stream)?
        .compile_stream(
            &udfs,
            &calc_flow::StreamRequirements::default(),
            &calc_flow::CompileOptions::default(),
        )
        .await?;
    Ok(())
}
