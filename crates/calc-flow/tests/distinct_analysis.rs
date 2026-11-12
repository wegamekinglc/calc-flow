use calc_flow::arrow::datatypes::{DataType, Field, Schema};
use calc_flow::{
    AnalysisOptions, Comparison, ComparisonCutoff, CompileMode, DistinctSpec, HistoryScope,
    InputOptions, PortabilityRequest, Program, QueryFinality, StateBudget, SupportStatus,
    UdfRegistry, table_input,
};
use std::{collections::BTreeMap, sync::Arc};

#[tokio::test]
async fn test_distinct_analysis_r9_exposes_conditional_finality_and_owned_peak() {
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
    assert_eq!(report.analysis_revision, 9);
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
