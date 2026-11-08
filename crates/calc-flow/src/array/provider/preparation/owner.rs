use super::source::{self, Latched, Source, Values};
use crate::{
    Batch, ExternalPayload, Result, StaticInputSpec,
    array::{ArrayDType, NativeArray},
    batch::{LatchedArrayPayload, NativeStaticValues},
};
use std::{collections::BTreeMap, sync::Arc};

pub(super) struct Owner<'a> {
    pub payload: Arc<dyn ExternalPayload>,
    pub aliases: Vec<&'a StaticInputSpec>,
}
impl Owner<'_> {
    pub(super) fn qualify(&self, error: crate::CalcFlowError) -> crate::CalcFlowError {
        let crate::CalcFlowError::InvalidArgument { field, message } = error else {
            return error;
        };
        if self.aliases.iter().any(|spec| {
            let prefix = format!("static_inputs.{}", spec.name());
            field == prefix || field.starts_with(&format!("{prefix}."))
        }) {
            return crate::CalcFlowError::InvalidArgument { field, message };
        }
        let suffix = field.strip_prefix("static_inputs.").unwrap_or(&field);
        crate::CalcFlowError::InvalidArgument {
            field: format!("static_inputs.{}.{suffix}", self.aliases[0].name()),
            message,
        }
    }
    pub(super) fn source(&self) -> Result<Source<'_>> {
        let payload = self.payload.as_ref();
        if payload.type_id() == std::any::TypeId::of::<NativeArray>() {
            let array = payload
                .as_any()
                .downcast_ref::<NativeArray>()
                .expect("intrinsic native owner type was checked");
            return Ok(Source::Native(array));
        }
        if payload.type_id() != std::any::TypeId::of::<LatchedArrayPayload>() {
            return Err(source::invalid(
                "backend",
                "requires a trusted native or latched array",
            ));
        }
        let value = payload
            .as_any()
            .downcast_ref::<LatchedArrayPayload>()
            .expect("intrinsic latched owner type was checked");
        latched(value).map(Source::Latched)
    }
}
pub(super) fn collect<'a>(
    specs: &'a BTreeMap<String, StaticInputSpec>,
    supplied: &BTreeMap<String, Batch>,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<Vec<Owner<'a>>> {
    let mut owners: Vec<Owner<'_>> = Vec::new();
    let mut indices = BTreeMap::new();
    for (index, (name, spec)) in specs.iter().enumerate() {
        if index.is_multiple_of(64) {
            check()?;
        }
        let batch = supplied
            .get(name)
            .ok_or_else(|| source::invalid(name, "required static input is missing"))?;
        if batch.kind() != crate::BatchKind::Array {
            return Err(source::invalid(
                &format!("{name}.kind"),
                "native static input must be an array",
            ));
        }
        let payload = Arc::clone(batch.external_payload()?);
        insert(&mut owners, &mut indices, payload, spec);
    }
    Ok(owners)
}
fn insert<'a>(
    owners: &mut Vec<Owner<'a>>,
    indices: &mut BTreeMap<usize, usize>,
    payload: Arc<dyn ExternalPayload>,
    spec: &'a StaticInputSpec,
) {
    let identity = Arc::as_ptr(&payload).cast::<()>().addr();
    if let Some(index) = indices.get(&identity) {
        owners[*index].aliases.push(spec);
    } else {
        indices.insert(identity, owners.len());
        owners.push(Owner {
            payload,
            aliases: vec![spec],
        });
    }
}

fn latched(value: &LatchedArrayPayload) -> Result<Latched<'_>> {
    let dtype = dtype(value.dtype())?;
    let values = match value.native_values()? {
        NativeStaticValues::Bool(values) => Values::Boolean(values),
        NativeStaticValues::Float(values) => Values::Float(values),
    };
    Ok(Latched {
        backend: value.backend(),
        dtype,
        shape: value.shape(),
        nulls: value.null_flags(),
        values,
        retained: value.checked_retained_bytes()?,
    })
}
fn dtype(value: &str) -> Result<ArrayDType> {
    match value {
        "bool" => Ok(ArrayDType::Boolean),
        "float32" => Ok(ArrayDType::Float32),
        "float64" => Ok(ArrayDType::Float64),
        _ => Err(source::invalid(
            "dtype",
            "native arrays require bool, float32 or float64",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::super::source::{Source, Values};
    use super::*;
    use crate::{Batch, BatchMetadata, StaticInputSpec, StaticMutability, array::NativeArray};
    use std::collections::BTreeMap;
    use std::sync::Arc;

    #[test]
    fn test_payload_owner_identity_shares_aliases_but_never_equal_values_or_only_backing_buffers() {
        let array = NativeArray::from_f64(vec![1.0, 2.0], vec![2, 1], None).unwrap();
        let separate_owner =
            Batch::external(Arc::new(array.clone()), BatchMetadata::default()).unwrap();
        let first = Batch::external(Arc::new(array), BatchMetadata::default()).unwrap();
        let second =
            first.with_metadata(BatchMetadata::new("alias", 1, Default::default()).unwrap());
        let supplied = BTreeMap::from([
            ("a.beta".into(), first),
            ("b.beta".into(), second),
            ("c.beta".into(), separate_owner),
        ]);
        let specs = ["a.beta", "b.beta", "c.beta"]
            .map(|name| {
                (
                    name.into(),
                    StaticInputSpec::Array {
                        name: name.into(),
                        mutability: StaticMutability::Static,
                        backend: "native".into(),
                        dtype: "float64".into(),
                        shape: vec![2, 1],
                    },
                )
            })
            .into_iter()
            .collect();
        let groups = collect(&specs, &supplied, &mut || Ok(())).unwrap();
        assert_eq!(groups.len(), 2);
        assert_eq!(
            groups[0]
                .aliases
                .iter()
                .map(|spec| spec.name())
                .collect::<Vec<_>>(),
            ["a.beta", "b.beta"]
        );
        assert_eq!(groups[1].aliases[0].name(), "c.beta");
        assert_eq!(groups[0].source().unwrap().profile().unwrap().placement, 0);
    }
    #[test]
    fn test_latched_read_only_source_uses_retained_capacity_and_original_compact_values() {
        let mut values = Vec::with_capacity(16);
        values.extend([1.0, -0.0]);
        let batch =
            Batch::static_array_float("native", "float32", vec![2, 1], None, values).unwrap();
        let spec = StaticInputSpec::Array {
            name: "calc.beta".into(),
            mutability: StaticMutability::Static,
            backend: "native".into(),
            dtype: "float32".into(),
            shape: vec![2, 1],
        };
        let specs = BTreeMap::from([("calc.beta".into(), spec)]);
        let supplied = BTreeMap::from([("calc.beta".into(), batch)]);
        let groups = collect(&specs, &supplied, &mut || Ok(())).unwrap();
        let source = groups[0].source().unwrap();
        let profile = source.profile().unwrap();
        assert_eq!(profile.frozen, 157);
        assert_eq!(profile.placement, 24);
        let Source::Latched(value) = source else {
            panic!("latched carrier expected")
        };
        assert!(
            matches!(value.values, Values::Float(values) if values[1].to_bits() == (-0.0_f64).to_bits())
        );
    }
    #[test]
    fn test_external_wrapper_cannot_delegate_as_any_to_claim_a_trusted_native_owner() {
        #[derive(Debug)]
        struct Forwarder {
            inner: Arc<dyn ExternalPayload>,
        }
        impl ExternalPayload for Forwarder {
            fn backend(&self) -> &str {
                "native"
            }
            fn len(&self) -> usize {
                self.inner.len()
            }
            fn estimated_bytes(&self) -> usize {
                self.inner.estimated_bytes()
            }
            fn as_any(&self) -> &dyn std::any::Any {
                self.inner.as_any()
            }
        }
        let value = NativeArray::from_f64(vec![1.0, 2.0], vec![2, 1], None).unwrap();
        let payload = Arc::new(Forwarder {
            inner: Arc::new(value),
        });
        let batch = Batch::external(payload, BatchMetadata::default()).unwrap();
        let spec = StaticInputSpec::Array {
            name: "calc.beta".into(),
            mutability: StaticMutability::Static,
            backend: "native".into(),
            dtype: "float64".into(),
            shape: vec![2, 1],
        };
        let specs = BTreeMap::from([("calc.beta".into(), spec)]);
        let supplied = BTreeMap::from([("calc.beta".into(), batch)]);
        let groups = collect(&specs, &supplied, &mut || Ok(())).unwrap();
        assert!(
            groups[0].source().is_err(),
            "delegated as_any must not make the retained wrapper a native owner"
        );
    }
    #[test]
    fn test_owner_metadata_grouping_is_cancellable_before_many_aliases_are_retained() {
        let array = NativeArray::from_f64(vec![1.0, 2.0], vec![2, 1], None).unwrap();
        let batch = Batch::external(Arc::new(array), BatchMetadata::default()).unwrap();
        let specs = (0..129)
            .map(|index| {
                let name = format!("node_{index}.beta");
                (
                    name.clone(),
                    StaticInputSpec::Array {
                        name,
                        mutability: StaticMutability::Static,
                        backend: "native".into(),
                        dtype: "float64".into(),
                        shape: vec![2, 1],
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        let supplied = specs
            .keys()
            .map(|name| (name.clone(), batch.clone()))
            .collect::<BTreeMap<_, _>>();
        let mut checks = 0;
        let result = collect(&specs, &supplied, &mut || {
            checks += 1;
            if checks == 2 {
                Err(crate::CalcFlowError::Cancelled {
                    run_id: "metadata".into(),
                })
            } else {
                Ok(())
            }
        });
        assert!(matches!(
            result,
            Err(crate::CalcFlowError::Cancelled { .. })
        ));
        assert_eq!(checks, 2);
    }
    #[test]
    fn test_static_array_kind_rejection_keeps_the_physical_alias() {
        use datafusion::arrow::{
            array::Float64Array,
            datatypes::{DataType, Field, Schema},
            record_batch::RecordBatch,
        };
        let record = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, false)])),
            vec![Arc::new(Float64Array::from(vec![1.0]))],
        )
        .unwrap();
        let batch = Batch::table(vec![record], BatchMetadata::default()).unwrap();
        let spec = StaticInputSpec::Array {
            name: "calc.beta".into(),
            mutability: StaticMutability::Static,
            backend: "native".into(),
            dtype: "float64".into(),
            shape: vec![2, 1],
        };
        let specs = BTreeMap::from([("calc.beta".into(), spec)]);
        let values = BTreeMap::from([("calc.beta".into(), batch)]);
        let result = collect(&specs, &values, &mut || Ok(()));
        assert!(
            matches!(result,Err(crate::CalcFlowError::InvalidArgument{field,..}) if field=="static_inputs.calc.beta.kind")
        );
    }
}
