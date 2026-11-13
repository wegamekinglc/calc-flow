use calc_flow::arrow::datatypes::{DataType, Field, Schema};
use calc_flow::{
    AnalysisOptions, Comparison, ComparisonCutoff, CompileMode, DistinctSpec, HistoryScope,
    InputOptions, PortabilityRequest, Program, QueryFinality, StateBudget, SupportStatus,
    UdfRegistry, table_input,
};
use std::{collections::BTreeMap, sync::Arc};

async fn analyze_window_time(projection: &str, sql: bool) -> calc_flow::QueryAnalysis {
    use calc_flow::arrow::datatypes::TimeUnit;
    use calc_flow::{CompileOptions, EventTime};
    let schema = Arc::new(Schema::new(vec![
        Field::new("ts", DataType::Timestamp(TimeUnit::Microsecond, None), true),
        Field::new("q", DataType::Int64, true),
    ]));
    let input = table_input("events", schema, InputOptions::default()).unwrap();
    let upstream = input
        .sql(&format!("SELECT {projection} AS t, q FROM input"))
        .unwrap();
    let out = if sql {
        calc_flow::sql::sql(
            "SELECT DISTINCT q FROM CF_DISTINCT_WINDOW('a','t',0,10,2,8388608)",
            BTreeMap::from([("a".into(), upstream)]),
        )
        .unwrap()
    } else {
        upstream
            .distinct(
                vec!["q".into()],
                DistinctSpec::closed_window(
                    "t",
                    EventTime::from_micros(0),
                    EventTime::from_micros(10),
                    StateBudget::new(2, 8 << 20).unwrap(),
                )
                .unwrap(),
            )
            .unwrap()
    };
    let program = Program::new(
        "lineage",
        CompileMode::Batch,
        BTreeMap::from([("out".into(), out)]),
    )
    .unwrap();
    let udfs = UdfRegistry::new().snapshot();
    let compiled = program
        .compile_batch(&udfs, &CompileOptions::default())
        .await;
    assert_eq!(compiled.is_ok(), projection == "ts");
    let options = AnalysisOptions {
        portability: Some(
            PortabilityRequest::new(
                Comparison::FinalRelation,
                HistoryScope::EventWindow,
                ComparisonCutoff::ClosedWindow,
            )
            .unwrap(),
        ),
    };
    program.analyze(&udfs, &options).await.unwrap()
}

#[tokio::test]
async fn test_window_analysis_proves_only_original_or_pure_renamed_time() {
    for sql in [false, true] {
        for projection in [
            "ts + INTERVAL '1 microsecond'",
            "CAST(ts AS TIMESTAMP)",
            "date_trunc('second', ts)",
        ] {
            let report = analyze_window_time(projection, sql).await;
            assert_eq!(
                report.portability,
                Some(SupportStatus::Unsupported),
                "{projection}: {report:?}"
            );
            assert!(
                report
                    .diagnostics
                    .iter()
                    .any(|d| d.code == "capability_mismatch"
                        && d.path.ends_with("distinct.spec.history"))
            );
        }
        assert_eq!(
            analyze_window_time("ts", sql).await.portability,
            Some(SupportStatus::Supported)
        );
    }
}

#[tokio::test]
async fn test_distinct_analysis_r10_exposes_conditional_finality_and_owned_peak() {
    let input = table_input(
        "events",
        Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)])),
        InputOptions::default(),
    )
    .unwrap();
    let out = input
        .distinct(
            vec!["x".into()],
            DistinctSpec::bounded_source(StateBudget::new(2, 8 << 20).unwrap()),
        )
        .unwrap();
    let program = Program::new(
        "analysis",
        CompileMode::Stream,
        BTreeMap::from([("out".into(), out)]),
    )
    .unwrap();
    let options = AnalysisOptions {
        portability: Some(
            PortabilityRequest::new(
                Comparison::FinalRelation,
                HistoryScope::AcceptedHistory,
                ComparisonCutoff::AllSourcesEof,
            )
            .unwrap(),
        ),
    };
    let report = program
        .analyze(&UdfRegistry::new().snapshot(), &options)
        .await
        .unwrap();
    assert_eq!(report.analysis_revision, 10);
    let node = report
        .nodes
        .iter()
        .find(|n| n.path == "outputs.out")
        .unwrap();
    assert_eq!(node.stream.support, SupportStatus::Supported);
    assert_eq!(node.stream.finality, QueryFinality::GroupFinal);
    assert_eq!(node.stream.history, HistoryScope::AcceptedHistory);
    assert_eq!(node.batch.history, HistoryScope::CurrentBatch);
    assert_eq!(node.batch.finality, QueryFinality::PerBatch);
    assert_eq!(report.portability, Some(SupportStatus::Supported));
    assert!(node.stream.reason.contains("copy_visible_v2"));
    assert!(node.stream.reason.contains("borrowed ingress"));
}
