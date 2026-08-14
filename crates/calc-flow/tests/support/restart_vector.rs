use std::path::Path;

use serde::Deserialize;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub(crate) struct RestartVector {
    pub(crate) schema_version: u32,
    pub(crate) plan: RestartPlanVector,
    pub(crate) checkpoint_after: usize,
    pub(crate) records: Vec<RestartRecordVector>,
    pub(crate) expected: RestartExpectedVector,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub(crate) struct RestartPlanVector {
    pub(crate) name: String,
    pub(crate) operator_id: String,
    pub(crate) expression: String,
    pub(crate) source_id: String,
    pub(crate) output_id: String,
    pub(crate) sink_id: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
pub(crate) struct RestartRecordVector {
    pub(crate) offset: usize,
    pub(crate) value: i64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub(crate) struct RestartExpectedVector {
    pub(crate) checkpoint_epoch: u64,
    pub(crate) terminal_epoch: u64,
    pub(crate) opened_offsets: Vec<usize>,
    pub(crate) values: Vec<i64>,
    pub(crate) duplicates: usize,
    pub(crate) missing: usize,
    pub(crate) terminal_tasks: usize,
    pub(crate) terminal_charged_edges: usize,
    pub(crate) temporary_artifacts: usize,
}

pub(crate) async fn restart_vector() -> RestartVector {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/a6/continuous_restart_vectors.json");
    serde_json::from_slice(&tokio::fs::read(path).await.unwrap()).unwrap()
}
