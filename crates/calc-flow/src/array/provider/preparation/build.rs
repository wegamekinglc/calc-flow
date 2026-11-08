use super::{
    budget, encode,
    freeze::{Frozen, OwnedValues},
    group,
    owner::{self, Owner},
    retention::Mode,
    source,
};
use crate::{
    Batch, BatchMetadata, Result, StaticInputDigest, StaticInputSpec, array::NativeArray,
    batch::NativeStaticOwned,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use super::scope::{Alias, Consumer, NativeScope};

pub(crate) fn prepare(
    specs: &BTreeMap<String, StaticInputSpec>,
    supplied: &BTreeMap<String, Batch>,
    consumers: Arc<Vec<Consumer>>,
    mode: Mode,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<NativeScope> {
    check()?;
    let (owners, costs, reservations) = plan(specs, supplied, &consumers, mode, check)?;
    let mut aliases = BTreeMap::new();
    for (index, owner) in owners.iter().enumerate() {
        aliases.extend(
            prepare_owner(owner, costs[index], check).map_err(|error| owner.qualify(error))?,
        );
    }
    check()?;
    Ok(NativeScope {
        aliases,
        consumers,
        reservations,
    })
}
type Admitted<'a> = (Vec<Owner<'a>>, Vec<budget::Group>, Vec<usize>);
fn plan<'a>(
    specs: &'a BTreeMap<String, StaticInputSpec>,
    supplied: &BTreeMap<String, Batch>,
    consumers: &[Consumer],
    mode: Mode,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<Admitted<'a>> {
    coverage(specs, consumers)?;
    let owners = owner::collect(specs, supplied, check)?;
    let costs = costs(&owners, mode, check)?;
    let reservations = reservations(&owners, &costs, consumers)?;
    Ok((owners, costs, reservations))
}
fn coverage(specs: &BTreeMap<String, StaticInputSpec>, consumers: &[Consumer]) -> Result<()> {
    let names = consumers
        .iter()
        .flat_map(|consumer| consumer.ports.values().map(AsRef::as_ref))
        .collect::<BTreeSet<&str>>();
    if names != specs.keys().map(String::as_str).collect() {
        return Err(source::invalid(
            "names",
            "native static declarations and trusted consumer bindings differ",
        ));
    }
    Ok(())
}
fn costs(
    owners: &[Owner<'_>],
    mode: Mode,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<Vec<budget::Group>> {
    owners
        .iter()
        .map(|owner| cost(owner, mode, check).map_err(|error| owner.qualify(error)))
        .collect()
}
fn cost(
    owner: &Owner<'_>,
    mode: Mode,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<budget::Group> {
    let source = owner.source()?;
    group::validate(&source, &owner.aliases, check)?;
    let profile = source.profile()?;
    Ok(budget::Group {
        retained: retained(&profile, &owner.aliases, mode)?,
        workspace: encode::workspace(owner.aliases.len())?,
    })
}
fn retained(profile: &source::Profile, aliases: &[&StaticInputSpec], mode: Mode) -> Result<usize> {
    let mut bytes = add(profile.frozen, profile.placement)?;
    for spec in aliases {
        bytes = add(bytes, mode.alias_bytes(spec.name())?)?;
    }
    Ok(bytes)
}
fn reservations(
    owners: &[Owner<'_>],
    groups: &[budget::Group],
    consumers: &[Consumer],
) -> Result<Vec<usize>> {
    let indices = owners
        .iter()
        .enumerate()
        .flat_map(|(index, owner)| owner.aliases.iter().map(move |spec| (spec.name(), index)))
        .collect::<BTreeMap<_, _>>();
    let aliases = consumers
        .iter()
        .map(|consumer| {
            consumer
                .ports
                .values()
                .map(|name| budget::Alias {
                    group: indices[name.as_ref()],
                    name: name.as_ref(),
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let consumers = consumers
        .iter()
        .zip(&aliases)
        .map(|(consumer, aliases)| budget::Consumer {
            node: &consumer.node,
            limit: consumer.max_bytes,
            aliases,
        })
        .collect::<Vec<_>>();
    budget::admit(groups, &consumers)
}
fn prepare_owner(
    owner: &Owner<'_>,
    cost: budget::Group,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<BTreeMap<Arc<str>, Alias>> {
    let source = owner.source()?;
    let prepared = group::prepare(&source, &owner.aliases, check)?;
    let frozen = frozen(owner, prepared.frozen);
    let placed = placed(owner, prepared.placement)?;
    let aliases = owner
        .aliases
        .iter()
        .zip(prepared.digests)
        .map(|(spec, digest)| {
            (
                Arc::from(spec.name()),
                Alias {
                    frozen: frozen.clone(),
                    placed: placed.clone(),
                    digest,
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    verify(owner, &frozen, &placed, &aliases, cost.retained)?;
    check()?;
    Ok(aliases)
}
fn frozen(owner: &Owner<'_>, frozen: Option<Frozen>) -> Batch {
    frozen.map_or_else(
        || {
            Batch::external(Arc::clone(&owner.payload), BatchMetadata::default())
                .expect("trusted array payload")
        },
        |frozen| {
            let values = match frozen.values {
                OwnedValues::Boolean(values) => NativeStaticOwned::Bool(values),
                OwnedValues::Float(values) => NativeStaticOwned::Float(values),
            };
            Batch::from_prepared_native(
                source::name(frozen.dtype),
                frozen.shape,
                frozen.nulls,
                values,
            )
        },
    )
}
fn placed(owner: &Owner<'_>, placed: Option<NativeArray>) -> Result<Batch> {
    let payload = placed.map_or_else(|| Arc::clone(&owner.payload), |array| Arc::new(array));
    Batch::external(payload, BatchMetadata::default())
}
fn verify(
    owner: &Owner<'_>,
    frozen: &Batch,
    placed: &Batch,
    aliases: &BTreeMap<Arc<str>, Alias>,
    limit: usize,
) -> Result<()> {
    let data = verify_data(owner, frozen, placed)?;
    let aliases = verify_aliases(aliases)?;
    verify_capacity(add(data, aliases)?, limit)
}
fn verify_data(owner: &Owner<'_>, frozen: &Batch, placed: &Batch) -> Result<usize> {
    let profile = owner.source()?.profile()?;
    let actual = data_bytes(owner, frozen, placed)?;
    verify_capacity(actual, add(profile.frozen, profile.placement)?)?;
    Ok(actual)
}
fn verify_aliases(aliases: &BTreeMap<Arc<str>, Alias>) -> Result<usize> {
    let actual = aliases.iter().try_fold(0, |bytes, (name, alias)| {
        add(bytes, alias_bytes(name, &alias.digest)?)
    })?;
    let expected = aliases
        .keys()
        .try_fold(0, |bytes, name| add(bytes, Mode::Finite.alias_bytes(name)?))?;
    verify_capacity(actual, expected)?;
    Ok(actual)
}
fn verify_capacity(actual: usize, limit: usize) -> Result<()> {
    if actual > limit {
        return Err(source::invalid(
            "bytes",
            &format!("actual retained capacities require {actual} bytes, admitted {limit}"),
        ));
    }
    Ok(())
}
fn data_bytes(owner: &Owner<'_>, frozen: &Batch, placed: &Batch) -> Result<usize> {
    let source = owner.source()?;
    let original = match source {
        source::Source::Native(array) => array.retained_buffer_bytes(),
        source::Source::Latched(_) => 0,
    };
    let frozen = frozen
        .latched_array_payload()
        .expect("preparation yields a latched array")
        .checked_retained_bytes()?;
    add(add(original, frozen)?, placement_bytes(&source, placed)?)
}
fn placement_bytes(source: &source::Source<'_>, placed: &Batch) -> Result<usize> {
    match source {
        source::Source::Native(_) => Ok(0),
        source::Source::Latched(_) => Ok(placed
            .external_payload()?
            .as_any()
            .downcast_ref::<NativeArray>()
            .expect("native placement")
            .retained_buffer_bytes()),
    }
}
fn alias_bytes(name: &str, digest: &StaticInputDigest) -> Result<usize> {
    add(
        add(name.len(), digest.digest_version.capacity())?,
        digest.sha256.capacity(),
    )
}
fn add(left: usize, right: usize) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| source::invalid("bytes", "native preparation byte count overflowed"))
}

#[cfg(test)]
mod tests {
    use super::super::retention::Mode;
    use super::*;
    use crate::{Batch, BatchMetadata, StaticInputSpec, StaticMutability, array::NativeArray};
    use std::{collections::BTreeMap, sync::Arc};

    fn declaration(name: &str) -> StaticInputSpec {
        StaticInputSpec::Array {
            name: name.into(),
            mutability: StaticMutability::Static,
            backend: "native".into(),
            dtype: "float32".into(),
            shape: vec![2, 1],
        }
    }
    fn consumer(node: &str, physical: &str, limit: usize) -> Consumer {
        Consumer {
            node: node.into(),
            ports: BTreeMap::from([("coefficients".into(), physical.into())]),
            max_bytes: limit,
        }
    }
    #[test]
    fn test_shared_native_owner_freezes_once_per_scope_and_resolves_only_trusted_local_ports() {
        let array = NativeArray::from_f32(vec![1.0, 2.0], vec![2, 1], None).unwrap();
        let batch = Batch::external(Arc::new(array), BatchMetadata::default()).unwrap();
        let specs = BTreeMap::from([
            ("left.beta".into(), declaration("left.beta")),
            ("right.beta".into(), declaration("right.beta")),
        ]);
        let supplied = BTreeMap::from([
            ("left.beta".into(), batch.clone()),
            ("right.beta".into(), batch),
        ]);
        let consumers = Arc::new(vec![
            consumer("left", "left.beta", 16000),
            consumer("right", "right.beta", 16000),
        ]);
        let first = prepare(
            &specs,
            &supplied,
            Arc::clone(&consumers),
            Mode::Finite,
            &mut || Ok(()),
        )
        .unwrap();
        let second = prepare(&specs, &supplied, consumers, Mode::Finite, &mut || Ok(())).unwrap();
        let a = &first.aliases["left.beta"];
        let b = &first.aliases["right.beta"];
        assert!(Arc::ptr_eq(
            a.frozen.external_payload().unwrap(),
            b.frozen.external_payload().unwrap()
        ));
        assert!(Arc::ptr_eq(
            a.placed.external_payload().unwrap(),
            b.placed.external_payload().unwrap()
        ));
        assert!(!Arc::ptr_eq(
            a.frozen.external_payload().unwrap(),
            second.aliases["left.beta"]
                .frozen
                .external_payload()
                .unwrap()
        ));
        let (resolved, reserved) = first.resolve("right", &BTreeMap::new()).unwrap();
        assert_eq!(resolved.len(), 1);
        assert!(
            resolved["coefficients"]
                .external_payload()
                .unwrap()
                .as_any()
                .is::<NativeArray>()
        );
        assert!(reserved > 0);
        assert!(
            first
                .resolve("unknown", &BTreeMap::new())
                .unwrap()
                .0
                .is_empty()
        );
    }
    #[test]
    fn test_tight_consumer_is_admitted_before_any_freeze_and_failure_publishes_no_scope() {
        let array = NativeArray::from_f32(vec![1.0, 2.0], vec![2, 1], None).unwrap();
        let batch = Batch::external(Arc::new(array), BatchMetadata::default()).unwrap();
        let specs = BTreeMap::from([
            ("tight.beta".into(), declaration("tight.beta")),
            ("wide.beta".into(), declaration("wide.beta")),
        ]);
        let supplied = BTreeMap::from([
            ("tight.beta".into(), batch.clone()),
            ("wide.beta".into(), batch),
        ]);
        let consumers = Arc::new(vec![
            consumer("wide", "wide.beta", 16000),
            consumer("tight", "tight.beta", 100),
        ]);
        let mut checks = 0;
        let result = prepare(&specs, &supplied, consumers, Mode::Finite, &mut || {
            checks += 1;
            Ok(())
        });
        assert!(
            matches!(result,Err(crate::CalcFlowError::InvalidArgument{field,..}) if field=="static_inputs.tight.beta.bytes")
        );
        assert!(
            checks < 5,
            "budget failure should precede numerical preparation"
        );
    }
    #[test]
    fn test_managed_future_descriptor_reservation_cannot_hide_larger_observed_data_capacity() {
        let array = NativeArray::from_f32(vec![1.0, 2.0], vec![2, 1], None).unwrap();
        let payload: Arc<dyn crate::ExternalPayload> = Arc::new(array);
        let spec = declaration("node.beta");
        let owner = Owner {
            payload,
            aliases: vec![&spec],
        };
        let mut values = Vec::with_capacity(4096);
        values.extend([1.0, 2.0]);
        let frozen = Batch::from_prepared_native(
            "float32",
            vec![2, 1],
            None,
            NativeStaticOwned::Float(values),
        );
        let placed = placed(&owner, None).unwrap();
        let aliases = BTreeMap::from([(
            Arc::from("node.beta"),
            Alias {
                frozen: frozen.clone(),
                placed: placed.clone(),
                digest: StaticInputDigest {
                    digest_version: crate::STATIC_INPUT_DIGEST_VERSION.into(),
                    sha256: "0".repeat(64),
                },
            },
        )]);
        let limit = cost(
            &owner,
            Mode::Managed {
                checkpointing: true,
                context_copies: 1000,
            },
            &mut || Ok(()),
        )
        .unwrap()
        .retained;
        assert!(
            data_bytes(&owner, &frozen, &placed).unwrap() < limit,
            "future descriptors provide real reservation slack"
        );
        let result = verify(&owner, &frozen, &placed, &aliases, limit);
        assert!(
            result.is_err(),
            "future descriptor reservations cannot subsidize unexpectedly larger buffers"
        );
    }
    #[test]
    fn test_forged_native_backend_rejection_keeps_the_physical_static_alias_path() {
        #[derive(Debug)]
        struct Forged;
        impl crate::ExternalPayload for Forged {
            fn backend(&self) -> &str {
                "native"
            }
            fn len(&self) -> usize {
                2
            }
            fn estimated_bytes(&self) -> usize {
                8
            }
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }
        let batch = Batch::external(Arc::new(Forged), BatchMetadata::default()).unwrap();
        let specs = BTreeMap::from([("bridge.beta".into(), declaration("bridge.beta"))]);
        let supplied = BTreeMap::from([("bridge.beta".into(), batch)]);
        let consumers = Arc::new(vec![consumer("bridge", "bridge.beta", 16000)]);
        let result = prepare(&specs, &supplied, consumers, Mode::Finite, &mut || Ok(()));
        assert!(
            matches!(result,Err(crate::CalcFlowError::InvalidArgument{field,..}) if field=="static_inputs.bridge.beta.backend")
        );
    }
}
