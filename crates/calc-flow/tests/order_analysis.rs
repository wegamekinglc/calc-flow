use calc_flow::{
    AnalysisOptions, Comparison, ComparisonCutoff, ComparisonOrdering, CompileMode, HistoryScope,
    InputOptions, NullPlacement, PortabilityRequest, Program, QueryFinality, QueryRowOrder,
    QueryTieOrder, Result, SortDirection, SortKey, SortSpec, StateBudget, SupportStatus,
    UdfRegistry,
    arrow::datatypes::{DataType, Field, Schema},
    table_input,
};
use std::{collections::BTreeMap, sync::Arc};

#[tokio::test]
async fn test_order_r10_proves_total_order_and_never_unordered_top_k_membership() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Int64, true),
        Field::new("seq", DataType::Int64, false),
    ]));
    let t = table_input("events", schema, InputOptions::default())?;
    for (stable, k, declared, expected) in [
        (false, None, false, SupportStatus::Supported),
        (false, Some(0), false, SupportStatus::Unknown),
        (false, Some(1), false, SupportStatus::Unknown),
        (true, Some(1), true, SupportStatus::Supported),
    ] {
        let spec = SortSpec::bounded_source(StateBudget::new(10, 8 << 20)?)?;
        let spec = if stable {
            spec.with_sequence_by(vec!["seq".into()])?
        } else {
            spec
        };
        let by = vec![SortKey::new(
            "x",
            SortDirection::Ascending,
            NullPlacement::Last,
        )?];
        let out = if let Some(k) = k {
            t.top_k(k, by, spec)?
        } else {
            t.sort(by, spec)?
        };
        let program = Program::new(
            "facts",
            CompileMode::Batch,
            BTreeMap::from([("out".into(), out)]),
        )?;
        let request = PortabilityRequest::new(
            Comparison::FinalRelation,
            HistoryScope::AcceptedHistory,
            ComparisonCutoff::AllSourcesEof,
        )?
        .with_ordering(if declared {
            ComparisonOrdering::Declared
        } else {
            ComparisonOrdering::Unordered
        });
        let report = program
            .analyze(
                &UdfRegistry::new().snapshot(),
                &AnalysisOptions {
                    portability: Some(request),
                },
            )
            .await?;
        assert_eq!(report.analysis_revision, 10);
        assert_eq!(report.portability, Some(expected), "{report:?}");
        let facts = report
            .nodes
            .iter()
            .find(|n| n.path == "outputs.out")
            .unwrap();
        assert_eq!(facts.stream.finality, QueryFinality::GroupFinal);
        assert_eq!(facts.batch.finality, QueryFinality::PerBatch);
        assert_eq!(facts.stream.ordering.kind, QueryRowOrder::Ordered);
        assert_eq!(
            facts.stream.ordering.ties,
            if stable {
                QueryTieOrder::Total
            } else {
                QueryTieOrder::Unspecified
            }
        );
        assert!(facts.stream.reason.contains("owned"));
    }
    Ok(())
}

#[tokio::test]
async fn test_order_postprocessing_proves_only_preserved_comparator_fields() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Int64, true),
        Field::new("seq", DataType::Int64, false),
    ]));
    let t = table_input("events", schema, InputOptions::default())?;
    let ordered = t.sort(
        vec![SortKey::new(
            "x",
            SortDirection::Ascending,
            NullPlacement::Last,
        )?],
        SortSpec::bounded_source(StateBudget::new(10, 8 << 20)?)?
            .with_sequence_by(vec!["seq".into()])?,
    )?;
    for (query, expected) in [
        (
            "SELECT x, seq, x+1 AS y FROM input",
            SupportStatus::Supported,
        ),
        ("SELECT x FROM input", SupportStatus::Unknown),
        ("SELECT x+1 AS x, seq FROM input", SupportStatus::Unknown),
    ] {
        let out = ordered.sql(query)?;
        let program = Program::new(
            "post",
            CompileMode::Batch,
            BTreeMap::from([("out".into(), out)]),
        )?;
        let request = PortabilityRequest::new(
            Comparison::FinalRelation,
            HistoryScope::AcceptedHistory,
            ComparisonCutoff::AllSourcesEof,
        )?
        .with_ordering(ComparisonOrdering::Declared);
        let report = program
            .analyze(
                &UdfRegistry::new().snapshot(),
                &AnalysisOptions {
                    portability: Some(request),
                },
            )
            .await?;
        assert_eq!(report.portability, Some(expected), "{query}: {report:?}");
    }
    Ok(())
}

#[tokio::test]
async fn test_order_closed_window_comparison_requires_one_interval() -> Result<()> {
    use calc_flow::EventTime;
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Int64, true),
        Field::new("seq", DataType::Int64, false),
        Field::new(
            "ts",
            DataType::Timestamp(calc_flow::arrow::datatypes::TimeUnit::Microsecond, None),
            false,
        ),
    ]));
    let t = table_input("events", schema, InputOptions::default())?;
    let mut outputs = BTreeMap::new();
    for end in [10, 20] {
        let spec = SortSpec::closed_window(
            calc_flow::SortWindow::new(
                "ts",
                EventTime::from_micros(0),
                EventTime::from_micros(end),
            )?,
            StateBudget::new(10, 8 << 20)?,
        )?
        .with_sequence_by(vec!["seq".into()])?;
        outputs.insert(
            format!("out{end}"),
            t.sort(
                vec![SortKey::new(
                    "x",
                    SortDirection::Ascending,
                    NullPlacement::Last,
                )?],
                spec,
            )?,
        );
    }
    let program = Program::new("domains", CompileMode::Stream, outputs)?;
    let request = PortabilityRequest::new(
        Comparison::FinalRelation,
        HistoryScope::EventWindow,
        ComparisonCutoff::ClosedWindow,
    )?
    .with_ordering(ComparisonOrdering::Declared);
    let report = program
        .analyze(
            &UdfRegistry::new().snapshot(),
            &AnalysisOptions {
                portability: Some(request),
            },
        )
        .await?;
    assert_eq!(
        report.portability,
        Some(SupportStatus::Unknown),
        "{report:?}"
    );
    assert!(
        report
            .diagnostics
            .iter()
            .any(|d| d.path == "options.portability.cutoff")
    );
    Ok(())
}

#[tokio::test]
async fn test_typed_projection_sequence_origin_agrees_with_strict_compile() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Int64, true),
        Field::new("seq", DataType::Int64, false),
    ]));
    let t = table_input("events", schema, InputOptions::default())?;
    for derived in [false, true] {
        let sequence = if derived {
            t.column("seq")? + calc_flow::lit(1_i64)?
        } else {
            t.column("seq")?
        };
        let projected = t.select(vec![t.column("x")?.alias("x")?, sequence.alias("qseq")?])?;
        let out = projected.sort(
            vec![SortKey::new(
                "x",
                SortDirection::Ascending,
                NullPlacement::Last,
            )?],
            SortSpec::bounded_source(StateBudget::new(10, 8 << 20)?)?
                .with_sequence_by(vec!["qseq".into()])?,
        )?;
        let p = Program::new(
            "typed_origin",
            CompileMode::Batch,
            BTreeMap::from([("out".into(), out)]),
        )?;
        let request = PortabilityRequest::new(
            Comparison::FinalRelation,
            HistoryScope::AcceptedHistory,
            ComparisonCutoff::AllSourcesEof,
        )?
        .with_ordering(ComparisonOrdering::Declared);
        let report = p
            .analyze(
                &UdfRegistry::new().snapshot(),
                &AnalysisOptions {
                    portability: Some(request.clone()),
                },
            )
            .await?;
        assert_eq!(
            report.portability,
            Some(if derived {
                SupportStatus::Unsupported
            } else {
                SupportStatus::Supported
            }),
            "derived={derived}: {report:?}"
        );
        let compiled = p
            .compile_batch(
                &UdfRegistry::new().snapshot(),
                &calc_flow::CompileOptions {
                    portability: Some(request),
                },
            )
            .await;
        assert_eq!(compiled.is_ok(), !derived);
    }
    Ok(())
}

#[tokio::test]
async fn test_typed_post_projection_preserves_only_original_comparator_fields() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Int64, true),
        Field::new("seq", DataType::Int64, false),
    ]));
    let t = table_input("events", schema, InputOptions::default())?;
    let sorted = t.sort(
        vec![SortKey::new(
            "x",
            SortDirection::Ascending,
            NullPlacement::Last,
        )?],
        SortSpec::bounded_source(StateBudget::new(10, 8 << 20)?)?
            .with_sequence_by(vec!["seq".into()])?,
    )?;
    for (index, columns) in [
        vec![
            sorted.column("x")?.alias("x")?,
            sorted.column("seq")?.alias("qseq")?,
        ],
        vec![sorted.column("x")?.alias("x")?],
        vec![
            (sorted.column("x")? + calc_flow::lit(1_i64)?).alias("computed_x")?,
            sorted.column("seq")?.alias("seq")?,
        ],
    ]
    .into_iter()
    .enumerate()
    {
        let expected = if index == 0 {
            SupportStatus::Supported
        } else {
            SupportStatus::Unknown
        };
        let out = sorted.select(columns)?;
        let p = Program::new(
            "typed_post",
            CompileMode::Batch,
            BTreeMap::from([("out".into(), out)]),
        )?;
        let request = PortabilityRequest::new(
            Comparison::FinalRelation,
            HistoryScope::AcceptedHistory,
            ComparisonCutoff::AllSourcesEof,
        )?
        .with_ordering(ComparisonOrdering::Declared);
        let report = p
            .analyze(
                &UdfRegistry::new().snapshot(),
                &AnalysisOptions {
                    portability: Some(request.clone()),
                },
            )
            .await?;
        assert_eq!(report.portability, Some(expected), "{report:?}");
        assert_eq!(
            p.compile_batch(
                &UdfRegistry::new().snapshot(),
                &calc_flow::CompileOptions {
                    portability: Some(request),
                }
            )
            .await
            .is_ok(),
            expected == SupportStatus::Supported
        );
    }
    Ok(())
}
