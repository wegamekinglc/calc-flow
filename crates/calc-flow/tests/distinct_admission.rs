use calc_flow::arrow::datatypes::{DataType, Field, Schema};
use calc_flow::{AnalysisOptions, CompileOptions};
use calc_flow::{
    CompileMode, DistinctSpec, InputOptions, Program, StateBudget, UdfRegistry, table_input,
};
use std::{collections::BTreeMap, sync::Arc};

#[tokio::test]
async fn test_distinct_rejects_cumulative_snapshot_history_without_opt_in() {
    let input = table_input(
        "events",
        Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)])),
        InputOptions::default(),
    )
    .unwrap();
    let snapshot = input.sql("SELECT SUM(x) AS x FROM input").unwrap();
    let output = snapshot
        .distinct(
            vec!["x".into()],
            DistinctSpec::bounded_source(StateBudget::new(10, 8 << 20).unwrap()),
        )
        .unwrap();
    let program = Program::new(
        "snapshot_history",
        CompileMode::Batch,
        BTreeMap::from([("out".into(), output)]),
    )
    .unwrap();
    let report = program
        .analyze(&UdfRegistry::new().snapshot(), &AnalysisOptions::default())
        .await
        .unwrap();
    let facts = report
        .nodes
        .iter()
        .find(|node| node.path == "outputs.out")
        .unwrap();
    assert_eq!(facts.stream.support, calc_flow::SupportStatus::Unsupported);
    let Err(error) = program
        .compile_batch(&UdfRegistry::new().snapshot(), &CompileOptions::default())
        .await
    else {
        panic!("cumulative snapshots are not append history")
    };
    assert!(error.to_string().contains("capability_mismatch"), "{error}");
}
