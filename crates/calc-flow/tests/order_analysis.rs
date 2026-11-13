use calc_flow::{
    AnalysisOptions, Comparison, ComparisonCutoff, ComparisonOrdering, CompileMode, HistoryScope,
    InputOptions, NullPlacement, PortabilityRequest, Program, QueryFinality, QueryRowOrder,
    QueryTieOrder, SortDirection, SortKey, SortSpec, StateBudget, SupportStatus, UdfRegistry,
    arrow::datatypes::{DataType, Field, Schema},
    table_input,
};
use std::{collections::BTreeMap, sync::Arc};

#[tokio::test]
async fn test_order_r10_proves_total_order_and_never_unordered_top_k_membership() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Int64, true),
        Field::new("seq", DataType::Int64, false),
    ]));
    let t = table_input("events", schema, InputOptions::default()).unwrap();
    for (stable, k, declared, expected) in [
        (false, None, false, SupportStatus::Supported),
        (false, Some(0), false, SupportStatus::Unknown),
        (false, Some(1), false, SupportStatus::Unknown),
        (true, Some(1), true, SupportStatus::Supported),
    ] {
        let spec = SortSpec::bounded_source(StateBudget::new(10, 8 << 20).unwrap()).unwrap();
        let spec = if stable {
            spec.with_sequence_by(vec!["seq".into()]).unwrap()
        } else {
            spec
        };
        let by = vec![SortKey::new("x", SortDirection::Ascending, NullPlacement::Last).unwrap()];
        let out = if let Some(k) = k {
            t.top_k(k, by, spec).unwrap()
        } else {
            t.sort(by, spec).unwrap()
        };
        let program = Program::new(
            "facts",
            CompileMode::Batch,
            BTreeMap::from([("out".into(), out)]),
        )
        .unwrap();
        let request = PortabilityRequest::new(
            Comparison::FinalRelation,
            HistoryScope::AcceptedHistory,
            ComparisonCutoff::AllSourcesEof,
        )
        .unwrap()
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
            .await
            .unwrap();
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
}

#[tokio::test]
async fn test_order_postprocessing_proves_only_preserved_comparator_fields() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Int64, true),
        Field::new("seq", DataType::Int64, false),
    ]));
    let t = table_input("events", schema, InputOptions::default()).unwrap();
    let ordered = t
        .sort(
            vec![SortKey::new("x", SortDirection::Ascending, NullPlacement::Last).unwrap()],
            SortSpec::bounded_source(StateBudget::new(10, 8 << 20).unwrap())
                .unwrap()
                .with_sequence_by(vec!["seq".into()])
                .unwrap(),
        )
        .unwrap();
    for (query, expected) in [
        (
            "SELECT x, seq, x+1 AS y FROM input",
            SupportStatus::Supported,
        ),
        ("SELECT x FROM input", SupportStatus::Unknown),
        ("SELECT x+1 AS x, seq FROM input", SupportStatus::Unknown),
    ] {
        let out = ordered.sql(query).unwrap();
        let program = Program::new(
            "post",
            CompileMode::Batch,
            BTreeMap::from([("out".into(), out)]),
        )
        .unwrap();
        let request = PortabilityRequest::new(
            Comparison::FinalRelation,
            HistoryScope::AcceptedHistory,
            ComparisonCutoff::AllSourcesEof,
        )
        .unwrap()
        .with_ordering(ComparisonOrdering::Declared);
        let report = program
            .analyze(
                &UdfRegistry::new().snapshot(),
                &AnalysisOptions {
                    portability: Some(request),
                },
            )
            .await
            .unwrap();
        assert_eq!(report.portability, Some(expected), "{query}: {report:?}");
    }
}

#[tokio::test]
async fn test_order_closed_window_comparison_requires_one_interval() {
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
    let t = table_input("events", schema, InputOptions::default()).unwrap();
    let mut outputs = BTreeMap::new();
    for end in [10, 20] {
        let spec = SortSpec::closed_window(
            calc_flow::SortWindow::new(
                "ts",
                EventTime::from_micros(0),
                EventTime::from_micros(end),
            )
            .unwrap(),
            StateBudget::new(10, 8 << 20).unwrap(),
        )
        .unwrap()
        .with_sequence_by(vec!["seq".into()])
        .unwrap();
        outputs.insert(
            format!("out{end}"),
            t.sort(
                vec![SortKey::new("x", SortDirection::Ascending, NullPlacement::Last).unwrap()],
                spec,
            )
            .unwrap(),
        );
    }
    let program = Program::new("domains", CompileMode::Stream, outputs).unwrap();
    let request = PortabilityRequest::new(
        Comparison::FinalRelation,
        HistoryScope::EventWindow,
        ComparisonCutoff::ClosedWindow,
    )
    .unwrap()
    .with_ordering(ComparisonOrdering::Declared);
    let report = program
        .analyze(
            &UdfRegistry::new().snapshot(),
            &AnalysisOptions {
                portability: Some(request),
            },
        )
        .await
        .unwrap();
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
}

#[tokio::test]
async fn test_typed_projection_sequence_origin_agrees_with_strict_compile() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Int64, true),
        Field::new("seq", DataType::Int64, false),
    ]));
    let t = table_input("events", schema, InputOptions::default()).unwrap();
    for derived in [false, true] {
        let sequence = if derived {
            t.column("seq").unwrap() + calc_flow::lit(1_i64).unwrap()
        } else {
            t.column("seq").unwrap()
        };
        let projected = t
            .select(vec![
                t.column("x").unwrap().alias("x").unwrap(),
                sequence.alias("qseq").unwrap(),
            ])
            .unwrap();
        let out = projected
            .sort(
                vec![SortKey::new("x", SortDirection::Ascending, NullPlacement::Last).unwrap()],
                SortSpec::bounded_source(StateBudget::new(10, 8 << 20).unwrap())
                    .unwrap()
                    .with_sequence_by(vec!["qseq".into()])
                    .unwrap(),
            )
            .unwrap();
        let p = Program::new(
            "typed_origin",
            CompileMode::Batch,
            BTreeMap::from([("out".into(), out)]),
        )
        .unwrap();
        let request = PortabilityRequest::new(
            Comparison::FinalRelation,
            HistoryScope::AcceptedHistory,
            ComparisonCutoff::AllSourcesEof,
        )
        .unwrap()
        .with_ordering(ComparisonOrdering::Declared);
        let report = p
            .analyze(
                &UdfRegistry::new().snapshot(),
                &AnalysisOptions {
                    portability: Some(request.clone()),
                },
            )
            .await
            .unwrap();
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
}

#[tokio::test]
async fn test_typed_post_projection_preserves_only_original_comparator_fields() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Int64, true),
        Field::new("seq", DataType::Int64, false),
    ]));
    let t = table_input("events", schema, InputOptions::default()).unwrap();
    let sorted = t
        .sort(
            vec![SortKey::new("x", SortDirection::Ascending, NullPlacement::Last).unwrap()],
            SortSpec::bounded_source(StateBudget::new(10, 8 << 20).unwrap())
                .unwrap()
                .with_sequence_by(vec!["seq".into()])
                .unwrap(),
        )
        .unwrap();
    for (index, columns) in [
        vec![
            sorted.column("x").unwrap().alias("x").unwrap(),
            sorted.column("seq").unwrap().alias("qseq").unwrap(),
        ],
        vec![sorted.column("x").unwrap().alias("x").unwrap()],
        vec![
            (sorted.column("x").unwrap() + calc_flow::lit(1_i64).unwrap())
                .alias("computed_x")
                .unwrap(),
            sorted.column("seq").unwrap().alias("seq").unwrap(),
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
        let out = sorted.select(columns).unwrap();
        let p = Program::new(
            "typed_post",
            CompileMode::Batch,
            BTreeMap::from([("out".into(), out)]),
        )
        .unwrap();
        let request = PortabilityRequest::new(
            Comparison::FinalRelation,
            HistoryScope::AcceptedHistory,
            ComparisonCutoff::AllSourcesEof,
        )
        .unwrap()
        .with_ordering(ComparisonOrdering::Declared);
        let report = p
            .analyze(
                &UdfRegistry::new().snapshot(),
                &AnalysisOptions {
                    portability: Some(request.clone()),
                },
            )
            .await
            .unwrap();
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
}
