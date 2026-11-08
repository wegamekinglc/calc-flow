use serde::Deserialize;

use super::super::{ArrayDType, ArrayExecutionOptions, invalid};
use crate::{Result, RollingNumericalProfile};

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
pub(super) enum DType {
    Bool,
    Float32,
    Float64,
}

impl DType {
    pub(super) const fn native(self) -> ArrayDType {
        match self {
            Self::Bool => ArrayDType::Boolean,
            Self::Float32 => ArrayDType::Float32,
            Self::Float64 => ArrayDType::Float64,
        }
    }
    pub(super) const fn width(self) -> usize {
        match self {
            Self::Bool => 0,
            Self::Float32 => 4,
            Self::Float64 => 8,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<Option<usize>>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct Output {
    pub name: String,
    pub node: String,
    pub dtype: DType,
    pub shape: Vec<Option<usize>>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Unary {
    Neg,
    Abs,
    Sqrt,
    Log,
    Exp,
    LogicalNot,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Binary {
    Add,
    Subtract,
    Multiply,
    Divide,
    Pow,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    LogicalAnd,
    LogicalOr,
    LogicalXor,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Reduce {
    Sum,
    Mean,
    Min,
    Max,
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Profile {
    #[default]
    StableV1,
    StableV2,
}

impl Profile {
    pub(super) const fn native(self) -> RollingNumericalProfile {
        match self {
            Self::StableV1 => RollingNumericalProfile::StableV1,
            Self::StableV2 => RollingNumericalProfile::StableV2Preview,
        }
    }
}

const fn one() -> u64 {
    1
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum Node {
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

impl Node {
    pub(super) fn id(&self) -> &str {
        match self {
            Self::Input { id, .. }
            | Self::Literal { id, .. }
            | Self::Unary { id, .. }
            | Self::Binary { id, .. }
            | Self::Where { id, .. }
            | Self::Reduce { id, .. }
            | Self::RollingMean { id, .. }
            | Self::Ewma { id, .. } => id,
        }
    }
    pub(super) fn references(&self) -> Vec<&str> {
        match self {
            Self::Input { .. } | Self::Literal { .. } => vec![],
            Self::Unary { input, .. }
            | Self::Reduce { input, .. }
            | Self::RollingMean { input, .. }
            | Self::Ewma { input, .. } => vec![input],
            Self::Binary { left, right, .. } => vec![left, right],
            Self::Where {
                condition,
                if_true,
                if_false,
                ..
            } => vec![condition, if_true, if_false],
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct Execution {
    pub max_elements: usize,
    pub max_allocation_bytes: usize,
}
impl Default for Execution {
    fn default() -> Self {
        let defaults = ArrayExecutionOptions::default();
        Self {
            max_elements: defaults.max_elements(),
            max_allocation_bytes: defaults.max_allocation_bytes(),
        }
    }
}
impl Execution {
    pub(super) fn options(&self) -> ArrayExecutionOptions {
        ArrayExecutionOptions::default()
            .with_max_elements(self.max_elements)
            .with_max_allocation_bytes(self.max_allocation_bytes)
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Program {
    pub ir_version: u8,
    pub inputs: Vec<Input>,
    pub nodes: Vec<Node>,
    pub outputs: Vec<Output>,
    #[serde(default)]
    pub execution: Execution,
}

pub(super) fn validate_shape(shape: &[Option<usize>], field: &str) -> Result<()> {
    if shape.len() > 1 {
        Err(invalid(
            field,
            "only scalar and vector shapes are supported",
        ))
    } else {
        Ok(())
    }
}
