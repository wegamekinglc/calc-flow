use std::{collections::BTreeMap, sync::Arc};

use datafusion::logical_expr::ScalarUDF;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{CalcFlowError, Result};

#[derive(
    Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum UdfKind {
    DataFusionScalar,
    ExternalScalar,
    ExternalArray,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize, JsonSchema)]
pub struct UdfReference {
    pub provider: String,
    pub name: String,
    pub version: String,
    pub kind: UdfKind,
}

impl UdfReference {
    /// Creates a portable, data-only reference to a versioned UDF.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when a reference component
    /// is empty or contains characters outside the portable identifier set.
    pub fn new(provider: &str, name: &str, version: &str, kind: UdfKind) -> Result<Self> {
        for (field, value) in [("provider", provider), ("name", name), ("version", version)] {
            if value.is_empty()
                || !value.chars().all(|character| {
                    character == '-'
                        || character == '_'
                        || character == '.'
                        || character.is_ascii_alphanumeric()
                })
            {
                return Err(CalcFlowError::InvalidArgument {
                    field: field.into(),
                    message: "must be a non-empty portable identifier".into(),
                });
            }
        }
        Ok(Self {
            provider: provider.into(),
            name: name.into(),
            version: version.into(),
            kind,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct UdfCatalogEntry {
    pub provider: String,
    pub name: String,
    pub version: String,
    pub kind: UdfKind,
    pub argument_count: usize,
}

#[derive(Default)]
pub struct UdfRegistry {
    native: BTreeMap<UdfReference, Arc<ScalarUDF>>,
    catalog: BTreeMap<UdfReference, UdfCatalogEntry>,
}

#[derive(Clone, Default)]
pub struct UdfRegistrySnapshot {
    native: Arc<BTreeMap<UdfReference, Arc<ScalarUDF>>>,
    catalog: Arc<Vec<UdfCatalogEntry>>,
}

impl UdfRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a native `DataFusion` scalar UDF under an exact reference.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when the reference kind or
    /// name does not match the implementation, or the reference is duplicate.
    pub fn register_datafusion(
        &mut self,
        reference: UdfReference,
        udf: Arc<ScalarUDF>,
        argument_count: usize,
    ) -> Result<()> {
        if reference.kind != UdfKind::DataFusionScalar || udf.name() != reference.name {
            return Err(CalcFlowError::InvalidArgument {
                field: "udf".into(),
                message: "reference kind and DataFusion name must match".into(),
            });
        }
        self.insert_catalog(reference.clone(), argument_count)?;
        self.native.insert(reference, udf);
        Ok(())
    }

    /// Registers metadata for a UDF implemented outside the Rust runtime.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] for native or duplicate
    /// references.
    pub fn register_external(
        &mut self,
        reference: UdfReference,
        argument_count: usize,
    ) -> Result<()> {
        if reference.kind == UdfKind::DataFusionScalar {
            return Err(CalcFlowError::InvalidArgument {
                field: "udf.kind".into(),
                message: "external registration requires an external kind".into(),
            });
        }
        self.insert_catalog(reference, argument_count)
    }

    pub fn snapshot(&self) -> UdfRegistrySnapshot {
        UdfRegistrySnapshot {
            native: Arc::new(self.native.clone()),
            catalog: Arc::new(self.catalog.values().cloned().collect()),
        }
    }

    fn insert_catalog(&mut self, reference: UdfReference, argument_count: usize) -> Result<()> {
        if self.catalog.contains_key(&reference) {
            return Err(CalcFlowError::InvalidArgument {
                field: "udf".into(),
                message: "duplicate provider/name/version/kind".into(),
            });
        }
        let entry = UdfCatalogEntry {
            provider: reference.provider.clone(),
            name: reference.name.clone(),
            version: reference.version.clone(),
            kind: reference.kind,
            argument_count,
        };
        self.catalog.insert(reference, entry);
        Ok(())
    }
}

impl UdfRegistrySnapshot {
    /// Resolves an exact native reference from this immutable snapshot.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::Compile`] when the exact reference is unknown
    /// or represents an external UDF.
    pub fn resolve_native(&self, reference: &UdfReference) -> Result<Arc<ScalarUDF>> {
        self.native
            .get(reference)
            .cloned()
            .ok_or_else(|| CalcFlowError::Compile {
                message: format!(
                    "unknown UDF {}:{}@{}",
                    reference.provider, reference.name, reference.version
                ),
            })
    }

    pub fn catalog(&self) -> &[UdfCatalogEntry] {
        &self.catalog
    }
}

/// Rejects different selected versions that would share one `DataFusion` SQL
/// function name.
///
/// # Errors
///
/// Returns [`CalcFlowError::Compile`] when two selected native references use
/// the same `DataFusion` name with different versions.
pub fn validate_selected_udfs(references: &[UdfReference]) -> Result<()> {
    let mut versions = BTreeMap::new();
    for reference in references
        .iter()
        .filter(|reference| reference.kind == UdfKind::DataFusionScalar)
    {
        if versions
            .insert(reference.name.clone(), reference.version.clone())
            .is_some_and(|version| version != reference.version)
        {
            return Err(CalcFlowError::Compile {
                message: format!("conflicting versions selected for {}", reference.name),
            });
        }
    }
    Ok(())
}
