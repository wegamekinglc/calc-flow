use serde::Deserialize;

use super::super::ir::{DType, Execution, Node};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Document {
    pub ir_version: u8,
    pub inputs: Vec<Input>,
    pub nodes: Vec<Node>,
    pub outputs: Vec<Output>,
    #[serde(default)]
    pub execution: Execution,
}
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum Input {
    Array {
        name: String,
        binding: Binding,
        dtype: DType,
        shape: Vec<Option<usize>>,
    },
    Table {
        name: String,
        columns: Vec<String>,
    },
}
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum Output {
    Array {
        name: String,
        node: String,
        dtype: DType,
        shape: Vec<Option<usize>>,
    },
    Table {
        name: String,
        node: String,
        attach_to: String,
        names: Vec<String>,
    },
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Binding {
    Static,
}
impl Input {
    pub(super) fn name(&self) -> &str {
        match self {
            Self::Table { name, .. } | Self::Array { name, .. } => name,
        }
    }
}
