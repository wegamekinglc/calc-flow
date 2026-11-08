use super::{
    super::invalid,
    ir::{
        Assertion, Binary, DType, Execution, Input, Node, Output, Profile, Program, Reduce, Unary,
        one,
    },
};
use crate::{JsonMap, Result};
use serde::{Deserialize, de::DeserializeOwned};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Document<N, O> {
    ir_version: u8,
    inputs: Vec<Input>,
    nodes: Vec<N>,
    outputs: Vec<O>,
    #[serde(default)]
    execution: Execution,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AssertedOutput {
    name: String,
    node: String,
    dtype: DType,
    shape: Vec<Option<usize>>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReferencedOutput {
    name: String,
    node: String,
}
impl From<AssertedOutput> for Output {
    fn from(output: AssertedOutput) -> Self {
        Self {
            name: output.name,
            node: output.node,
            assertion: Some(Assertion {
                dtype: output.dtype,
                shape: output.shape,
            }),
        }
    }
}
impl From<ReferencedOutput> for Output {
    fn from(output: ReferencedOutput) -> Self {
        Self {
            name: output.name,
            node: output.node,
            assertion: None,
        }
    }
}

pub(super) fn decode(options: &JsonMap, version: u8, query: bool) -> Result<Program> {
    match (version, query) {
        (1, false) => typed::<NodeV1, AssertedOutput>(options),
        (1, true) => typed::<NodeV1, ReferencedOutput>(options),
        (2, false) => typed::<Node, AssertedOutput>(options),
        (2, true) => typed::<Node, ReferencedOutput>(options),
        _ => Err(invalid(
            "options.ir_version",
            "only versions 1 and 2 are supported",
        )),
    }
}
fn typed<N: DeserializeOwned + Into<Node>, O: DeserializeOwned + Into<Output>>(
    options: &JsonMap,
) -> Result<Program> {
    let doc: Document<N, O> = serde_json::from_value(serde_json::Value::Object(
        options.clone().into_iter().collect(),
    ))
    .map_err(|_| invalid("options", "invalid typed array program"))?;
    Ok(Program {
        ir_version: doc.ir_version,
        inputs: doc.inputs,
        nodes: doc.nodes.into_iter().map(Into::into).collect(),
        outputs: doc.outputs.into_iter().map(Into::into).collect(),
        execution: doc.execution,
    })
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum NodeV1 {
    Input {
        id: String,
        name: String,
    },
    Literal {
        id: String,
        dtype: DType,
        value: serde_json::Value,
    },
    Unary {
        id: String,
        function: Unary,
        input: String,
    },
    Binary {
        id: String,
        function: Binary,
        left: String,
        right: String,
    },
    Where {
        id: String,
        condition: String,
        if_true: String,
        if_false: String,
    },
    Reduce {
        id: String,
        function: Reduce,
        input: String,
        #[serde(default)]
        skip_nulls: bool,
    },
    RollingMean {
        id: String,
        input: String,
        window: u64,
        #[serde(default = "one")]
        min_periods: u64,
        #[serde(default)]
        numerical_profile: Profile,
    },
    Ewma {
        id: String,
        input: String,
        span: u64,
        #[serde(default = "one")]
        min_periods: u64,
    },
}

impl From<NodeV1> for Node {
    fn from(node: NodeV1) -> Self {
        match node {
            NodeV1::Input { id, name } => Self::Input { id, name },
            NodeV1::Literal { id, dtype, value } => Self::Literal { id, dtype, value },
            NodeV1::Unary {
                id,
                function,
                input,
            } => Self::Unary {
                id,
                function,
                input,
            },
            NodeV1::Binary {
                id,
                function,
                left,
                right,
            } => Self::Binary {
                id,
                function,
                left,
                right,
            },
            NodeV1::Where {
                id,
                condition,
                if_true,
                if_false,
            } => Self::Where {
                id,
                condition,
                if_true,
                if_false,
            },
            NodeV1::Reduce {
                id,
                function,
                input,
                skip_nulls,
            } => Self::Reduce {
                id,
                function,
                input,
                skip_nulls,
                axis: None,
                keepdims: false,
            },
            other => stateful(other),
        }
    }
}
fn stateful(node: NodeV1) -> Node {
    match node {
        NodeV1::RollingMean {
            id,
            input,
            window,
            min_periods,
            numerical_profile,
        } => Node::RollingMean {
            id,
            input,
            window,
            min_periods,
            numerical_profile,
        },
        NodeV1::Ewma {
            id,
            input,
            span,
            min_periods,
        } => Node::Ewma {
            id,
            input,
            span,
            min_periods,
        },
        _ => unreachable!("other version-1 nodes were converted separately"),
    }
}
