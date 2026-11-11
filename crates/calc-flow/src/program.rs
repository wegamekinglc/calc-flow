//! Named immutable relation programs and lifecycle selection.

use crate::{
    AnalysisOptions, BatchExecutionPlan, CalcFlowError, PortabilityRequest, ProjectSpec,
    QueryAnalysis, Result, StreamExecutionPlan, StreamRequirements, TableExpr, UdfRegistrySnapshot,
    dataframe::{RelationKind, check_name, digest, invalid},
};
use datafusion::arrow::datatypes::SchemaRef;
use serde_json::json;
use std::collections::BTreeMap;
mod lower;

/// The execution lifecycle, independent of `DataFrame` or SQL declaration syntax.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompileMode {
    Batch,
    Stream,
}
impl CompileMode {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Batch => "batch",
            Self::Stream => "stream",
        }
    }
}
/// Optional comparison proof required again at compilation.
#[derive(Clone, Debug, Default)]
pub struct CompileOptions {
    pub portability: Option<PortabilityRequest>,
}
/// A named declaration wrapper with no compiled plan or running state.
#[derive(Clone, Debug)]
pub struct Program {
    name: String,
    mode: CompileMode,
    inputs: Vec<TableExpr>,
    outputs: BTreeMap<String, TableExpr>,
    inferred: bool,
    fingerprint: String,
}

impl Program {
    /// Creates a wrapper, discovering inputs reachable from the named outputs.
    ///
    /// # Errors
    /// Rejects empty names, conflicting inputs and known incompatible mode shapes.
    pub fn new(
        name: &str,
        mode: CompileMode,
        outputs: BTreeMap<String, TableExpr>,
    ) -> Result<Self> {
        check_name(name, "Program.name")?;
        for name in outputs.keys() {
            check_name(name, "outputs")?;
        }
        let inputs = discover(&outputs)?;
        let mut program = Self {
            name: name.into(),
            mode,
            inputs,
            outputs,
            inferred: true,
            fingerprint: String::new(),
        };
        program.validate_mode()?;
        program.fingerprint = program.declaration_fingerprint();
        Ok(program)
    }
    /// Returns the declaration name.
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Returns the bound lifecycle.
    pub const fn mode(&self) -> CompileMode {
        self.mode
    }
    /// Returns a declaration identity that excludes lifecycle and runtime state.
    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }
    /// Returns the input declarations, ordered by logical name when inferred.
    pub fn inputs(&self) -> &[TableExpr] {
        &self.inputs
    }
    /// Returns named output declarations.
    pub const fn outputs(&self) -> &BTreeMap<String, TableExpr> {
        &self.outputs
    }
    /// Appends one explicit root input, returning a new wrapper.
    ///
    /// # Errors
    /// Rejects non-root declarations and duplicate logical input names.
    pub fn with_input(&self, input: TableExpr) -> Result<Self> {
        let RelationKind::Input { name, .. } = &input.0.kind else {
            return Err(invalid(
                "inputs",
                "invalid_literal",
                "with_input requires a table input",
            ));
        };
        if self.inputs.iter().any(|old| matches!(&old.0.kind, RelationKind::Input { name: old_name, .. } if old_name == name)) {
            return Err(invalid(&lower::segment("inputs", name), "duplicate_name", "input name must be unique"));
        }
        let mut candidate = self.clone();
        candidate.inputs.push(input);
        candidate.inferred = false;
        candidate.fingerprint = candidate.declaration_fingerprint();
        Ok(candidate)
    }
    /// Adds a uniquely named output, preserving automatic or explicit input selection.
    ///
    /// # Errors
    /// Rejects duplicate names, conflicting inputs and incompatible target shapes.
    pub fn output(&self, name: &str, value: TableExpr) -> Result<Self> {
        check_name(name, "outputs")?;
        if self.outputs.contains_key(name) {
            return Err(invalid(
                &lower::segment("outputs", name),
                "duplicate_name",
                "output name must be unique",
            ));
        }
        let mut candidate = self.clone();
        candidate.outputs.insert(name.into(), value);
        if candidate.inferred {
            candidate.inputs = discover(&candidate.outputs)?;
        }
        candidate.validate_mode()?;
        candidate.fingerprint = candidate.declaration_fingerprint();
        Ok(candidate)
    }
    /// Copies declarations into an independent lifecycle wrapper.
    ///
    /// # Errors
    /// Rejects graph shapes incompatible with the target mode before any source opens.
    pub fn for_mode(&self, mode: CompileMode) -> Result<Self> {
        let candidate = Self {
            mode,
            ..self.clone()
        };
        candidate.validate_mode()?;
        Ok(candidate)
    }
    fn declaration_fingerprint(&self) -> String {
        digest(&json!([
            "program-v1",
            self.name,
            self.inputs.iter().map(|t| &t.0.id).collect::<Vec<_>>(),
            self.outputs
                .iter()
                .map(|(n, t)| (n, &t.0.id))
                .collect::<Vec<_>>()
        ]))
    }
    fn validate_mode(&self) -> Result<()> {
        if self.mode == CompileMode::Stream {
            for (name, table) in &self.outputs {
                validate_stream(table, &lower::segment("outputs", name))?;
            }
        }
        Ok(())
    }
    /// Plans exact table output schemas without input data or job creation.
    ///
    /// # Errors
    /// Returns logical field/type/lineage planning errors or unresolved outputs.
    pub async fn collect_schema(
        &self,
        udfs: &UdfRegistrySnapshot,
    ) -> Result<BTreeMap<String, SchemaRef>> {
        let lowered = self.lower(udfs).await?;
        lowered.ensure_valid()?;
        Ok(self
            .outputs
            .iter()
            .map(|(name, table)| (name.clone(), lowered.schemas[&table.0.id].clone()))
            .collect())
    }
    /// Analyzes both lifecycles using the existing history/portability rules.
    ///
    /// # Errors
    /// Returns invalid-declaration or planning errors; no input data is accessed.
    pub async fn analyze(
        &self,
        udfs: &UdfRegistrySnapshot,
        options: &AnalysisOptions,
    ) -> Result<QueryAnalysis> {
        self.lower(udfs).await?.analyze(options)
    }
    /// Renders deterministic facts, bound mode and declaration identity.
    ///
    /// # Errors
    /// Returns the same planning errors as analysis.
    pub async fn explain(
        &self,
        udfs: &UdfRegistrySnapshot,
        options: &AnalysisOptions,
    ) -> Result<String> {
        let report = self.analyze(udfs, options).await?;
        Ok(format!(
            "Program {}\nmode {}\nfingerprint {}\n{}",
            self.name,
            self.mode.name(),
            self.fingerprint,
            serde_json::to_string_pretty(&report).map_err(|error| CalcFlowError::Internal {
                message: format!("query report serialization failed: {error}")
            })?
        ))
    }
    async fn lower(&self, udfs: &UdfRegistrySnapshot) -> Result<lower::Lowered> {
        if self.outputs.is_empty() {
            return Err(invalid(
                "outputs",
                "unresolved_type",
                "at least one output is required",
            ));
        }
        let _ = udfs;
        self.validate_inputs()?;
        lower::lower(self).await
    }
    fn validate_inputs(&self) -> Result<()> {
        let reachable = discover(&self.outputs)?;
        for input in &reachable {
            if !self
                .inputs
                .iter()
                .any(|declared| declared.0.id == input.0.id)
            {
                let RelationKind::Input { name, .. } = &input.0.kind else {
                    unreachable!()
                };
                return Err(CalcFlowError::Compile {
                    message: format!(
                        "{}: unresolved_type: referenced input is not declared",
                        lower::segment("inputs", name)
                    ),
                });
            }
        }
        for input in &self.inputs {
            if !reachable.iter().any(|used| used.0.id == input.0.id) {
                let RelationKind::Input { name, .. } = &input.0.kind else {
                    unreachable!()
                };
                return Err(CalcFlowError::Compile {
                    message: format!(
                        "{}: unresolved_type: declared input is not consumed",
                        lower::segment("inputs", name)
                    ),
                });
            }
        }
        Ok(())
    }
    /// Compiles a fresh finite plan, accepting and returning logical binding names.
    ///
    /// # Errors
    /// Rejects a stream wrapper, invalid declarations and unproved strict requests.
    pub async fn compile_batch(
        &self,
        udfs: &UdfRegistrySnapshot,
        options: &CompileOptions,
    ) -> Result<BatchExecutionPlan> {
        self.require_mode(CompileMode::Batch)?;
        let lowered = self.lower(udfs).await?;
        lowered.gate(options)?;
        lowered.builder()?.compile_batch(udfs)
    }
    /// Compiles a fresh stream plan whose source, sink and delivery keys are logical names.
    ///
    /// # Errors
    /// Rejects a batch wrapper, invalid declarations and unproved strict requests.
    pub async fn compile_stream(
        &self,
        udfs: &UdfRegistrySnapshot,
        requirements: &StreamRequirements,
        options: &CompileOptions,
    ) -> Result<StreamExecutionPlan> {
        self.require_mode(CompileMode::Stream)?;
        let lowered = self.lower(udfs).await?;
        lowered.gate(options)?;
        lowered.builder()?.compile_stream(udfs, requirements)
    }
    /// Exports a strict data-only project, with schema placeholders and physical binding names.
    ///
    /// # Errors
    /// Returns schema and declaration errors without serializing rows or running state.
    pub async fn to_project(&self, udfs: &UdfRegistrySnapshot) -> Result<ProjectSpec> {
        self.lower(udfs).await?.project(self.mode)
    }
    fn require_mode(&self, mode: CompileMode) -> Result<()> {
        if self.mode != mode {
            return Err(CalcFlowError::Compile {
                message: format!(
                    "Program.mode: unsupported_mode: bound {}, requested {}",
                    self.mode.name(),
                    mode.name()
                ),
            });
        }
        Ok(())
    }
}
fn discover(outputs: &BTreeMap<String, TableExpr>) -> Result<Vec<TableExpr>> {
    let mut inputs = BTreeMap::<String, TableExpr>::new();
    for table in outputs.values() {
        discover_table(table, &mut inputs)?;
    }
    Ok(inputs.into_values().collect())
}
fn discover_table(table: &TableExpr, inputs: &mut BTreeMap<String, TableExpr>) -> Result<()> {
    match &table.0.kind {
        RelationKind::Input { name, .. } => {
            if inputs.get(name).is_some_and(|old| old.0.id != table.0.id) {
                return Err(invalid(
                    &lower::segment("inputs", name),
                    "duplicate_name",
                    "conflicting declarations for one input name",
                ));
            }
            inputs.insert(name.clone(), table.clone());
        }
        RelationKind::Project { input, .. } | RelationKind::Filter { input, .. } => {
            discover_table(input, inputs)?;
        }
        RelationKind::Sql { tables, .. } => {
            for table in tables.values() {
                discover_table(table, inputs)?;
            }
        }
    }
    Ok(())
}
fn validate_stream(table: &TableExpr, path: &str) -> Result<()> {
    match &table.0.kind {
        RelationKind::Input { .. } => {}
        RelationKind::Project { input, .. } | RelationKind::Filter { input, .. } => {
            validate_stream(input, &format!("{path}.project.value"))?;
        }
        RelationKind::Sql { tables, .. } => {
            if tables.len() != 1 {
                return Err(invalid(
                    path,
                    "unsupported_mode",
                    "stream SQL accepts one table alias",
                ));
            }
            for (name, table) in tables {
                validate_stream(table, &lower::segment(&format!("{path}.sql.tables"), name))?;
            }
        }
    }
    Ok(())
}
