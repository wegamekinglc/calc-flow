use std::{collections::BTreeMap, sync::Arc};

use datafusion::logical_expr::ScalarUDF;
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize, de::Error as _};

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

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct UdfReference {
    provider: String,
    name: String,
    version: String,
    kind: UdfKind,
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

    pub fn provider(&self) -> &str {
        &self.provider
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn version(&self) -> &str {
        &self.version
    }

    pub const fn kind(&self) -> UdfKind {
        self.kind
    }
}

impl<'de> Deserialize<'de> for UdfReference {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Fields {
            provider: String,
            name: String,
            version: String,
            kind: UdfKind,
        }

        let fields = Fields::deserialize(deserializer)?;
        Self::new(&fields.provider, &fields.name, &fields.version, fields.kind)
            .map_err(D::Error::custom)
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
        if reference.kind() != UdfKind::DataFusionScalar || udf.name() != reference.name() {
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
        if reference.kind() == UdfKind::DataFusionScalar {
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
            provider: reference.provider().into(),
            name: reference.name().into(),
            version: reference.version().into(),
            kind: reference.kind(),
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
                    reference.provider(),
                    reference.name(),
                    reference.version()
                ),
            })
    }

    pub fn catalog(&self) -> &[UdfCatalogEntry] {
        &self.catalog
    }
}

/// Rejects distinct native references that would share one `DataFusion` SQL
/// function name while allowing exact duplicate selections.
///
/// # Errors
///
/// Returns [`CalcFlowError::Compile`] when two distinct selected native
/// references use the same primary `DataFusion` name.
pub fn validate_selected_udfs(references: &[UdfReference]) -> Result<()> {
    let mut owners: BTreeMap<&str, &UdfReference> = BTreeMap::new();
    for reference in references
        .iter()
        .filter(|reference| reference.kind() == UdfKind::DataFusionScalar)
    {
        if let Some(&owner) = owners.get(reference.name()) {
            if owner != reference {
                return Err(CalcFlowError::Compile {
                    message: format!(
                        "DataFusion SQL name '{}' collides between {} and {}",
                        reference.name(),
                        describe_reference(owner),
                        describe_reference(reference)
                    ),
                });
            }
        } else {
            owners.insert(reference.name(), reference);
        }
    }
    Ok(())
}

fn describe_reference(reference: &UdfReference) -> String {
    format!(
        "{}:{}@{} ({:?})",
        reference.provider(),
        reference.name(),
        reference.version(),
        reference.kind()
    )
}
