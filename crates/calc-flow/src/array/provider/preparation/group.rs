use super::{
    encode,
    freeze::{self, Frozen},
    place,
    source::{self, Source},
};
use crate::{CalcFlowError, Result, StaticInputDigest, StaticInputSpec};

pub(super) struct Prepared {
    pub frozen: Option<Frozen>,
    pub placement: Option<crate::array::NativeArray>,
    pub digests: Vec<StaticInputDigest>,
}
pub(super) fn prepare(
    source: &Source<'_>,
    aliases: &[&StaticInputSpec],
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<Prepared> {
    validate(source, aliases, check)?;
    let names = aliases.iter().map(|spec| spec.name()).collect::<Vec<_>>();
    match source {
        Source::Native(array) => {
            let frozen = freeze::freeze(array, check)?;
            let digests = encode::digests(&names, &frozen.view(), check)?;
            Ok(Prepared {
                frozen: Some(frozen),
                placement: None,
                digests,
            })
        }
        Source::Latched(value) => {
            let digests = encode::digests(&names, value, check)?;
            let placement = place::place(value, check)?;
            Ok(Prepared {
                frozen: None,
                placement: Some(placement),
                digests,
            })
        }
    }
}
pub(super) fn validate(
    source: &Source<'_>,
    aliases: &[&StaticInputSpec],
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<()> {
    check()?;
    for (index, spec) in aliases.iter().enumerate() {
        if index != 0 && index.is_multiple_of(64) {
            check()?;
        }
        validate_one(source, spec)?;
    }
    Ok(())
}
fn validate_one(source: &Source<'_>, spec: &StaticInputSpec) -> Result<()> {
    let StaticInputSpec::Array {
        name,
        backend,
        dtype,
        shape,
        ..
    } = spec
    else {
        return Err(failure(spec.name(), "kind"));
    };
    if backend != "native" {
        return Err(failure(name, "backend"));
    }
    let actual = match source {
        Source::Native(array) => array.dtype(),
        Source::Latched(value) => value.dtype,
    };
    if dtype != source::name(actual) {
        return Err(failure(name, "dtype"));
    }
    validate_shape(source, shape).map_err(|_| failure(name, "shape"))
}
fn validate_shape(source: &Source<'_>, shape: &[u64]) -> Result<()> {
    let matches = match source {
        Source::Native(array) => {
            array.shape().len() == shape.len()
                && array
                    .shape()
                    .iter()
                    .zip(shape)
                    .all(|(actual, expected)| u64::try_from(*actual).ok() == Some(*expected))
        }
        Source::Latched(value) => value.backend == "native" && value.shape == shape,
    };
    if !matches {
        return Err(source::invalid(
            "shape",
            "native static descriptor differs from declaration",
        ));
    }
    Ok(())
}
fn failure(name: &str, field: &str) -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: format!("static_inputs.{name}.{field}"),
        message: "native static descriptor differs from declaration".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{StaticInputSpec, StaticMutability, array::NativeArray};

    fn declaration(name: &str, dtype: &str, shape: Vec<u64>) -> StaticInputSpec {
        StaticInputSpec::Array {
            name: name.into(),
            mutability: StaticMutability::Static,
            backend: "native".into(),
            dtype: dtype.into(),
            shape,
        }
    }
    #[test]
    fn test_one_native_owner_prepares_one_frozen_carrier_for_all_physical_aliases() {
        let array = NativeArray::from_f32(vec![1.0, -0.0], vec![2, 1], None).unwrap();
        let aliases = [
            declaration("left.beta", "float32", vec![2, 1]),
            declaration("right.beta", "float32", vec![2, 1]),
        ];
        let prepared = prepare(
            &Source::Native(&array),
            &aliases.iter().collect::<Vec<_>>(),
            &mut || Ok(()),
        )
        .unwrap();
        assert!(prepared.placement.is_none());
        let frozen = prepared.frozen.unwrap();
        assert_eq!(frozen.shape, [2, 1]);
        assert_eq!(prepared.digests.len(), 2);
        assert_ne!(prepared.digests[0].sha256, prepared.digests[1].sha256);
    }
    #[test]
    fn test_same_owner_aliases_still_validate_each_declared_dtype_before_freeze() {
        let array = NativeArray::from_f32(vec![1.0; 8193], vec![8193], None).unwrap();
        let aliases = [
            declaration("a.beta", "float32", vec![8193]),
            declaration("b.beta", "float64", vec![8193]),
        ];
        let mut checks = 0;
        let result = prepare(
            &Source::Native(&array),
            &aliases.iter().collect::<Vec<_>>(),
            &mut || {
                checks += 1;
                Ok(())
            },
        );
        assert!(
            matches!(result, Err(CalcFlowError::InvalidArgument {field,..}) if field == "static_inputs.b.beta.dtype")
        );
        assert_eq!(checks, 1);
    }
    #[test]
    fn test_same_owner_aliases_still_validate_each_shape_and_only_native_specs() {
        let array = NativeArray::from_f64(vec![1.0, 2.0], vec![2, 1], None).unwrap();
        let good = declaration("a.beta", "float64", vec![2, 1]);
        let bad = declaration("b.beta", "float64", vec![1, 2]);
        let result = prepare(&Source::Native(&array), &[&good, &bad], &mut || Ok(()));
        assert!(
            matches!(result, Err(CalcFlowError::InvalidArgument {field,..}) if field == "static_inputs.b.beta.shape")
        );
    }
}
