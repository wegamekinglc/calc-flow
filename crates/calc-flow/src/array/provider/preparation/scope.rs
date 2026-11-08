use crate::{Batch, CalcFlowError, Result, StaticInputDigest};
use std::{collections::BTreeMap, sync::Arc};

#[derive(Clone)]
pub(crate) struct Consumer {
    pub node: Arc<str>,
    pub ports: BTreeMap<Arc<str>, Arc<str>>,
    pub max_bytes: usize,
}
pub(super) struct Alias {
    pub frozen: Batch,
    pub placed: Batch,
    pub digest: StaticInputDigest,
}
pub(crate) struct NativeScope {
    pub(super) aliases: BTreeMap<Arc<str>, Alias>,
    pub(super) consumers: Arc<Vec<Consumer>>,
    pub(super) reservations: Vec<usize>,
}
impl std::fmt::Debug for NativeScope {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NativeStaticScope")
            .field("aliases", &self.aliases.len())
            .field("consumers", &self.consumers.len())
            .finish_non_exhaustive()
    }
}
impl NativeScope {
    pub(super) fn physical_name(&self, node: &str, port: &str) -> Option<&str> {
        self.consumers
            .iter()
            .find(|consumer| consumer.node.as_ref() == node)?
            .ports
            .get(port)
            .map(AsRef::as_ref)
    }
    pub(crate) fn verify_recovered(
        &self,
        recovered: &BTreeMap<String, StaticInputDigest>,
    ) -> Result<()> {
        for name in self.aliases.keys() {
            let (actual, digest) = recovered
                .get_key_value(name.as_ref())
                .ok_or_else(|| recovered_capacity_error(name))?;
            verify_recovered_descriptor(name, actual.capacity(), digest)?;
        }
        Ok(())
    }
    pub(crate) fn input(&self, node: &str, port: &str) -> Option<&Batch> {
        self.aliases
            .get(self.physical_name(node, port)?)
            .map(|alias| &alias.placed)
    }
    pub(crate) fn resolve(
        &self,
        node: &str,
        inputs: &BTreeMap<String, Batch>,
    ) -> Result<(BTreeMap<String, Batch>, usize)> {
        let mut inputs = inputs.clone();
        let Some((index, consumer)) = self
            .consumers
            .iter()
            .enumerate()
            .find(|(_, consumer)| consumer.node.as_ref() == node)
        else {
            return Ok((inputs, 0));
        };
        for (port, name) in &consumer.ports {
            let alias = self
                .aliases
                .get(name.as_ref())
                .ok_or_else(|| CalcFlowError::Internal {
                    message: "prepared native static alias is missing".into(),
                })?;
            inputs.insert(port.as_ref().to_owned(), alias.placed.clone());
        }
        Ok((inputs, self.reservations[index]))
    }
    pub(crate) fn latched(&self) -> BTreeMap<String, Batch> {
        self.aliases
            .iter()
            .map(|(name, alias)| (name.as_ref().to_owned(), alias.frozen.clone()))
            .collect()
    }
    pub(crate) fn digests(&self) -> BTreeMap<String, StaticInputDigest> {
        self.aliases
            .iter()
            .map(|(name, alias)| (name.as_ref().to_owned(), alias.digest.clone()))
            .collect()
    }
}

fn recovered_capacity_error(name: &str) -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: format!("static_inputs.{name}.bytes"),
        message: "recovered native descriptor capacity exceeds its reserved storage".into(),
    }
}
fn verify_recovered_descriptor(
    name: &str,
    capacity: usize,
    digest: &StaticInputDigest,
) -> Result<()> {
    let actual = capacity
        .checked_add(digest.digest_version.capacity())
        .and_then(|bytes| bytes.checked_add(digest.sha256.capacity()));
    let limit = super::retention::Mode::Finite.alias_bytes(name)?;
    if actual.is_none_or(|actual| actual > limit) {
        return Err(recovered_capacity_error(name));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BatchMetadata, StaticInputSpec, StaticMutability, array::NativeArray};

    #[test]
    fn test_recovered_native_descriptor_capacities_are_checked_before_scope_installation() {
        let name = "beta";
        let specs = BTreeMap::from([(
            name.into(),
            StaticInputSpec::Array {
                name: name.into(),
                mutability: StaticMutability::Static,
                backend: "native".into(),
                dtype: "float64".into(),
                shape: vec![2],
            },
        )]);
        let supplied = BTreeMap::from([(
            name.into(),
            Batch::external(
                Arc::new(NativeArray::from_f64(vec![1.0, 2.0], vec![2], None).unwrap()),
                BatchMetadata::default(),
            )
            .unwrap(),
        )]);
        let consumers = Arc::new(vec![Consumer {
            node: "node".into(),
            ports: BTreeMap::from([("coefficient".into(), name.into())]),
            max_bytes: 10_000,
        }]);
        let scope = super::super::build::prepare(
            &specs,
            &supplied,
            consumers,
            super::super::retention::Mode::Finite,
            &mut || Ok(()),
        )
        .unwrap();
        let mut recovered = scope.digests();
        scope.verify_recovered(&recovered).unwrap();
        let digest = recovered.remove(name).unwrap();
        let mut overallocated = String::with_capacity(1024);
        overallocated.push_str(name);
        recovered.insert(overallocated, digest);
        let result = scope.verify_recovered(&recovered);
        assert!(
            matches!(result,Err(CalcFlowError::InvalidArgument{field,..}) if field=="static_inputs.beta.bytes")
        );
    }
}
