use calc_flow::{
    AnalysisOptions, BatchKind, Comparison, ComparisonCutoff, HistoryScope, PipelineBuilder, Port,
    PortabilityRequest, SqlOperator, SupportStatus, UdfRegistrySnapshot,
};
use datafusion::arrow::datatypes::{DataType, Field};

fn row_local() -> PipelineBuilder {
    let schema = vec![Field::new("x", DataType::Int64, true)];
    let operator = SqlOperator::new(
        "select",
        "SELECT x FROM input",
        vec!["input".into()],
        vec![],
    )
    .unwrap()
    .with_ports(
        vec![Port::new("input", BatchKind::Table, true, Some(schema)).unwrap()],
        Port::new("output", BatchKind::Table, true, None).unwrap(),
    )
    .unwrap();
    PipelineBuilder::new("portable")
        .unwrap()
        .add_node("select", Box::new(operator))
        .unwrap()
}

#[tokio::test]
async fn row_local_checked_compile_keeps_the_original_identity() {
    let udfs = UdfRegistrySnapshot::default();
    let request = PortabilityRequest::new(
        Comparison::PerBatch,
        HistoryScope::CurrentBatch,
        ComparisonCutoff::BatchBoundary,
    )
    .unwrap();
    let report = row_local()
        .analyze(
            &udfs,
            &AnalysisOptions {
                portability: Some(request.clone()),
            },
        )
        .await
        .unwrap();
    assert_eq!(report.portability, Some(SupportStatus::Supported));
    assert_eq!(report.nodes[0].stream.history, HistoryScope::CurrentBatch);
    assert_eq!(
        report.nodes[0].schema.as_ref().unwrap().field(0).name(),
        "x"
    );
    let ordinary = row_local().compile_batch(&udfs).unwrap();
    let checked = row_local()
        .compile_batch_checked(&udfs, &request)
        .await
        .unwrap();
    assert_eq!(checked.fingerprint(), ordinary.fingerprint());
    let original_stream = row_local()
        .compile_stream(&udfs, &calc_flow::StreamRequirements::default())
        .unwrap();
    let checked_stream = row_local()
        .compile_stream_checked(&udfs, &calc_flow::StreamRequirements::default(), &request)
        .await
        .unwrap();
    assert_eq!(checked_stream.fingerprint(), original_stream.fingerprint());
}

#[tokio::test]
async fn aggregate_history_and_snapshot_append_edges_are_distinct() {
    use calc_flow::{Edge, PortEndpoint, QueryFinality, UpdateKind};
    let first = SqlOperator::new(
        "sum1",
        "SELECT SUM(x) AS total FROM input",
        vec!["input".into()],
        vec![],
    )
    .unwrap()
    .with_ports(
        vec![
            Port::new(
                "input",
                BatchKind::Table,
                true,
                Some(vec![Field::new("x", DataType::Int64, true)]),
            )
            .unwrap(),
        ],
        Port::new("output", BatchKind::Table, true, None).unwrap(),
    )
    .unwrap();
    let second = SqlOperator::new(
        "sum2",
        "SELECT SUM(total) AS total FROM input",
        vec!["input".into()],
        vec![],
    )
    .unwrap();
    let builder = PipelineBuilder::new("chain")
        .unwrap()
        .add_node("sum1", Box::new(first))
        .unwrap()
        .add_node("sum2", Box::new(second))
        .unwrap()
        .connect(Edge::new(
            PortEndpoint::new("sum1", "output").unwrap(),
            PortEndpoint::new("sum2", "input").unwrap(),
        ))
        .unwrap();
    let request = PortabilityRequest::new(
        Comparison::FinalRelation,
        HistoryScope::AcceptedHistory,
        ComparisonCutoff::AllSourcesEof,
    )
    .unwrap();
    let report = builder
        .analyze(
            &UdfRegistrySnapshot::default(),
            &AnalysisOptions {
                portability: Some(request),
            },
        )
        .await
        .unwrap();
    let first = &report.nodes[0];
    assert_eq!(first.batch.history, HistoryScope::CurrentBatch);
    assert_eq!(first.stream.history, HistoryScope::AcceptedHistory);
    assert_eq!(first.stream.finality, QueryFinality::RevisableSnapshot);
    assert_eq!(first.stream.update, UpdateKind::CompleteSnapshot);
    assert_eq!(report.portability, Some(SupportStatus::Unsupported));
    assert!(
        report
            .diagnostics
            .iter()
            .any(|d| d.code == "update_interpretation_mismatch"
                && d.path == "graph.edges[\"sum1\"][\"output\"][\"sum2\"][\"input\"]")
    );
    match builder
        .compile_batch_checked(
            &UdfRegistrySnapshot::default(),
            report.request.as_ref().unwrap(),
        )
        .await
    {
        Err(calc_flow::CalcFlowError::QueryPortability { report }) => {
            assert_eq!(report.portability, Some(SupportStatus::Unsupported));
        }
        _ => panic!("snapshot append comparison must reject before compilation"),
    }
}

#[tokio::test]
async fn sql_shapes_separate_per_batch_scope_from_unproved_semantics() {
    for (query, history, portability, code) in [
        (
            "SELECT x FROM input WHERE x > 0",
            HistoryScope::CurrentBatch,
            SupportStatus::Supported,
            None,
        ),
        (
            "SELECT DISTINCT x FROM input",
            HistoryScope::CurrentBatch,
            SupportStatus::Unsupported,
            Some("history_scope_mismatch"),
        ),
        (
            "SELECT x FROM input ORDER BY x LIMIT 1",
            HistoryScope::CurrentBatch,
            SupportStatus::Unsupported,
            Some("history_scope_mismatch"),
        ),
        (
            "SELECT row_number() OVER (ORDER BY x) AS n FROM input",
            HistoryScope::CurrentBatch,
            SupportStatus::Unsupported,
            Some("history_scope_mismatch"),
        ),
        (
            "WITH chosen AS (SELECT x FROM input) SELECT x FROM chosen",
            HistoryScope::Unknown,
            SupportStatus::Unknown,
            Some("sql_semantics_unproven"),
        ),
        (
            "SELECT SUM(x) AS total FROM input HAVING SUM(x) > 0",
            HistoryScope::AcceptedHistory,
            SupportStatus::Unknown,
            Some("sql_semantics_unproven"),
        ),
    ] {
        let operator = SqlOperator::new("q", query, vec!["input".into()], vec![])
            .unwrap()
            .with_ports(
                vec![
                    Port::new(
                        "input",
                        BatchKind::Table,
                        true,
                        Some(vec![Field::new("x", DataType::Int64, true)]),
                    )
                    .unwrap(),
                ],
                Port::new("output", BatchKind::Table, true, None).unwrap(),
            )
            .unwrap();
        let builder = PipelineBuilder::new("shapes")
            .unwrap()
            .add_node("q", Box::new(operator))
            .unwrap();
        let request = PortabilityRequest::new(
            Comparison::FinalRelation,
            HistoryScope::AcceptedHistory,
            ComparisonCutoff::AllSourcesEof,
        )
        .unwrap();
        let report = builder
            .analyze(
                &UdfRegistrySnapshot::default(),
                &AnalysisOptions {
                    portability: Some(request),
                },
            )
            .await
            .unwrap();
        assert_eq!(
            report.nodes[0].stream.support,
            SupportStatus::Supported,
            "{query}"
        );
        assert_eq!(report.nodes[0].stream.history, history, "{query}");
        assert_eq!(report.portability, Some(portability), "{query}");
        if let Some(code) = code {
            assert!(report.diagnostics.iter().any(|d| d.code == code), "{query}");
        }
    }
}

#[tokio::test]
async fn stream_only_modes_and_sql_schema_errors_are_reported_without_execution() {
    let union = calc_flow::UnionOperator::new(
        "union",
        vec![
            Port::new(
                "left",
                BatchKind::Table,
                true,
                Some(vec![Field::new("x", DataType::Int64, true)]),
            )
            .unwrap(),
            Port::new(
                "right",
                BatchKind::Table,
                true,
                Some(vec![Field::new("x", DataType::Int64, true)]),
            )
            .unwrap(),
        ],
    )
    .unwrap();
    let builder = PipelineBuilder::new("modes")
        .unwrap()
        .add_node("u\".x", Box::new(union))
        .unwrap();
    let report = builder
        .analyze(&UdfRegistrySnapshot::default(), &AnalysisOptions::default())
        .await
        .unwrap();
    assert_eq!(report.nodes[0].path, "graph.nodes[\"u\\\".x\"]");
    assert_eq!(report.nodes[0].batch.support, SupportStatus::Unsupported);
    assert_eq!(report.nodes[0].stream.support, SupportStatus::Supported);
    assert!(
        report
            .diagnostics
            .iter()
            .any(|issue| issue.code == "unsupported_mode")
    );

    let sql = SqlOperator::new(
        "bad",
        "SELECT missing FROM input",
        vec!["input".into()],
        vec![],
    )
    .unwrap()
    .with_ports(
        vec![
            Port::new(
                "input",
                BatchKind::Table,
                true,
                Some(vec![Field::new("x", DataType::Int64, true)]),
            )
            .unwrap(),
        ],
        Port::new("output", BatchKind::Table, true, None).unwrap(),
    )
    .unwrap();
    let builder = PipelineBuilder::new("invalid")
        .unwrap()
        .add_node("bad", Box::new(sql))
        .unwrap();
    let report = builder
        .analyze(&UdfRegistrySnapshot::default(), &AnalysisOptions::default())
        .await
        .unwrap();
    assert!(
        report
            .diagnostics
            .iter()
            .any(|issue| issue.code == "schema_mismatch")
    );
    assert_eq!(report.nodes[0].stream.support, SupportStatus::Unsupported);
}

#[tokio::test]
async fn known_missing_udf_keeps_the_existing_compilation_error() {
    let reference = calc_flow::UdfReference::new(
        "missing",
        "custom",
        "1",
        calc_flow::UdfKind::DataFusionScalar,
    )
    .unwrap();
    let sql = SqlOperator::new(
        "selected",
        "SELECT x FROM input",
        vec!["input".into()],
        vec![reference],
    )
    .unwrap();
    let builder = PipelineBuilder::new("registration")
        .unwrap()
        .add_node("selected", Box::new(sql))
        .unwrap();
    let result = builder
        .analyze(&UdfRegistrySnapshot::default(), &AnalysisOptions::default())
        .await;
    assert!(
        matches!(result, Err(calc_flow::CalcFlowError::Compile { message }) if message == "unknown UDF missing:custom@1")
    );
}

#[tokio::test]
async fn snapshot_edges_keep_the_original_stream_mode_rejection() {
    let fields = vec![Field::new("x", DataType::Int64, true)];
    let snapshot = calc_flow::SnapshotSqlOperator::new(
        "snapshot",
        "SELECT x FROM input",
        vec!["input".into()],
        vec![],
    )
    .unwrap()
    .with_ports(
        vec![Port::new("input", BatchKind::Table, true, Some(fields.clone())).unwrap()],
        Port::new("output", BatchKind::Table, true, Some(fields)).unwrap(),
    )
    .unwrap();
    let downstream = SqlOperator::new(
        "consumer",
        "SELECT x FROM input",
        vec!["input".into()],
        vec![],
    )
    .unwrap()
    .with_ports(
        vec![
            Port::new(
                "input",
                BatchKind::Table,
                true,
                Some(vec![Field::new("x", DataType::Int64, true)]),
            )
            .unwrap(),
        ],
        Port::new("output", BatchKind::Table, true, None).unwrap(),
    )
    .unwrap();
    let builder = PipelineBuilder::new("snapshot_edge")
        .unwrap()
        .add_node("snapshot", snapshot)
        .unwrap()
        .add_node("consumer", Box::new(downstream))
        .unwrap()
        .connect(calc_flow::Edge::new(
            calc_flow::PortEndpoint::new("snapshot", "output").unwrap(),
            calc_flow::PortEndpoint::new("consumer", "input").unwrap(),
        ))
        .unwrap();
    let report = builder
        .analyze(&UdfRegistrySnapshot::default(), &AnalysisOptions::default())
        .await
        .unwrap();
    assert_eq!(report.edges[0].stream.support, SupportStatus::Unsupported);
    assert!(
        report
            .diagnostics
            .iter()
            .any(|d| d.code == "unsupported_mode" && d.path.starts_with("graph.edges["))
    );
    assert!(matches!(
        builder.compile_stream(
            &UdfRegistrySnapshot::default(),
            &calc_flow::StreamRequirements::default()
        ),
        Err(calc_flow::CalcFlowError::Compile { .. })
    ));
}

#[test]
fn requests_validate_and_roundtrip_without_becoming_execution_permits() {
    let request = PortabilityRequest::new(
        Comparison::PerBatch,
        HistoryScope::CurrentBatch,
        ComparisonCutoff::BatchBoundary,
    )
    .unwrap()
    .with_ordering(calc_flow::ComparisonOrdering::Declared);
    let encoded = serde_json::to_string(&request).unwrap();
    let decoded: PortabilityRequest = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded.comparison(), Comparison::PerBatch);
    assert_eq!(decoded.history(), HistoryScope::CurrentBatch);
    assert_eq!(decoded.cutoff(), ComparisonCutoff::BatchBoundary);
    assert_eq!(decoded.ordering(), calc_flow::ComparisonOrdering::Declared);
    for invalid in [
        r#"{"comparison":"per_batch","history":"accepted_history","cutoff":"all_sources_eof","ordering":"unordered"}"#,
        r#"{"comparison":"per_batch","history":"unknown","cutoff":"batch_boundary","ordering":"unordered"}"#,
        r#"{"comparison":"per_batch","history":"current_batch","cutoff":"batch_boundary","ordering":"unordered","ignored":true}"#,
    ] {
        assert!(serde_json::from_str::<PortabilityRequest>(invalid).is_err());
    }
}
