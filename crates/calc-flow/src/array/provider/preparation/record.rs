use super::super::{infer::Plan, ir::Input};
use super::scope::Consumer;
use crate::{CalcFlowError, PortEndpoint, Result, StaticInputSpec};
use std::{collections::BTreeMap, sync::Arc};

pub(crate) struct Template {
    pub(super) node: Arc<str>,
    pub(super) plan: Arc<Plan>,
    table: Option<Arc<str>>,
    required: Option<Arc<str>>,
}
impl Template {
    pub(in crate::array::provider) fn array(node: &str, plan: Arc<Plan>) -> Self {
        Self {
            node: node.into(),
            plan,
            table: None,
            required: None,
        }
    }
    pub(in crate::array::provider) fn table(node: &str, plan: Arc<Plan>, table: &str) -> Self {
        let required = plan
            .program
            .inputs
            .iter()
            .find(|input| input.name != table)
            .map(|input| Arc::from(input.name.as_str()));
        Self {
            node: node.into(),
            plan,
            table: Some(table.into()),
            required,
        }
    }
    pub(super) fn validate_scope(&self, scope: &super::scope::NativeScope) -> Result<()> {
        if let Some(port) = &self.required {
            let batch = scope
                .input(&self.node, port)
                .expect("required static inputs were bound during compilation");
            let payload = batch.external_payload()?;
            let array = payload
                .as_any()
                .downcast_ref::<crate::array::NativeArray>()
                .expect("native placement is trusted");
            if array.as_arrow().null_count() != 0 {
                return Err(failure(
                    scope
                        .physical_name(&self.node, port)
                        .expect("required static alias was bound"),
                    "nulls",
                    "matmul static RHS cannot contain null values",
                ));
            }
        }
        Ok(())
    }
}
pub(super) fn bind(
    templates: &[Template],
    external: &BTreeMap<String, PortEndpoint>,
    specs: &BTreeMap<String, StaticInputSpec>,
) -> Result<Vec<Consumer>> {
    let consumers = templates
        .iter()
        .filter_map(|template| bind_one(template, external, specs).transpose())
        .collect::<Result<Vec<_>>>()?;
    native_coverage(specs, &consumers)?;
    Ok(consumers)
}
pub(super) fn validate_required(
    templates: &[Template],
    external: &BTreeMap<String, PortEndpoint>,
    specs: &BTreeMap<String, StaticInputSpec>,
) -> Result<()> {
    for template in templates {
        let Some(port) = &template.required else {
            continue;
        };
        let binding = external.iter().find(|(_, endpoint)| {
            endpoint.node_id == template.node.as_ref() && endpoint.port == port.as_ref()
        });
        if binding.is_some_and(|(name, _)| specs.contains_key(name)) {
            continue;
        }
        let name = binding.map_or_else(
            || format!("{}.{}", template.node, port),
            |(name, _)| name.clone(),
        );
        return Err(failure(
            &name,
            "",
            "declared static RHS requires a compiled static input binding",
        ));
    }
    Ok(())
}
fn native_coverage(
    specs: &BTreeMap<String, StaticInputSpec>,
    consumers: &[Consumer],
) -> Result<()> {
    let names = consumers
        .iter()
        .flat_map(|consumer| consumer.ports.values().map(AsRef::as_ref))
        .collect::<std::collections::BTreeSet<&str>>();
    for (name, spec) in specs {
        if matches!(spec,StaticInputSpec::Array{backend,..} if backend=="native")
            && !names.contains(name.as_str())
        {
            return Err(failure(
                name,
                "",
                "native static input has no trusted native consumer allocation budget",
            ));
        }
    }
    Ok(())
}
fn bind_one(
    template: &Template,
    external: &BTreeMap<String, PortEndpoint>,
    specs: &BTreeMap<String, StaticInputSpec>,
) -> Result<Option<Consumer>> {
    let mut ports = BTreeMap::new();
    for (name, endpoint) in external
        .iter()
        .filter(|(_, endpoint)| endpoint.node_id == template.node.as_ref())
    {
        if let Some(spec) = specs.get(name) {
            validate(template, &endpoint.port, spec)?;
            ports.insert(Arc::from(endpoint.port.as_str()), Arc::from(name.as_str()));
        }
    }
    required(template, external, &ports)?;
    Ok((!ports.is_empty()).then(|| Consumer {
        node: Arc::clone(&template.node),
        ports,
        max_bytes: template.plan.program.execution.max_allocation_bytes,
    }))
}
fn required(
    template: &Template,
    external: &BTreeMap<String, PortEndpoint>,
    ports: &BTreeMap<Arc<str>, Arc<str>>,
) -> Result<()> {
    if let Some(port) = &template.required
        && !ports.contains_key(port)
    {
        let name = external
            .iter()
            .find(|(_, endpoint)| {
                endpoint.node_id == template.node.as_ref() && endpoint.port == port.as_ref()
            })
            .map_or_else(
                || format!("{}.{}", template.node, port),
                |(name, _)| name.clone(),
            );
        return Err(failure(
            &name,
            "",
            "a table static RHS requires a compiled static input declaration",
        ));
    }
    Ok(())
}
fn validate(template: &Template, port: &str, spec: &StaticInputSpec) -> Result<()> {
    if template.plan.program.ir_version == 1 {
        return Err(failure(
            spec.name(),
            "",
            "native static inputs require array IR2",
        ));
    }
    if template.table.as_deref() == Some(port) {
        return Err(failure(
            spec.name(),
            "kind",
            "the table ingress must remain dynamic",
        ));
    }
    let input = template
        .plan
        .program
        .inputs
        .iter()
        .find(|input| input.name == port)
        .expect("compiled native input exists");
    descriptor(input, spec)
}
fn descriptor(input: &Input, spec: &StaticInputSpec) -> Result<()> {
    let StaticInputSpec::Array {
        name,
        backend,
        dtype,
        shape,
        ..
    } = spec
    else {
        return Err(failure(
            spec.name(),
            "kind",
            "native static inputs must be arrays",
        ));
    };
    if backend != "native" {
        return Err(failure(
            name,
            "backend",
            "native static inputs require the native backend",
        ));
    }
    if dtype != input.dtype.spelling() {
        return Err(failure(
            name,
            "dtype",
            "static dtype differs from native program declaration",
        ));
    }
    shape_matches(input, shape).then_some(()).ok_or_else(|| {
        failure(
            name,
            "shape",
            "static shape differs from native program declaration",
        )
    })
}
fn shape_matches(input: &Input, shape: &[u64]) -> bool {
    input.shape.len() == shape.len()
        && input.shape.iter().zip(shape).all(|(expected, actual)| {
            expected.is_none_or(|expected| usize::try_from(*actual).ok() == Some(expected))
        })
}
fn failure(name: &str, field: &str, message: &str) -> CalcFlowError {
    let suffix = if field.is_empty() {
        String::new()
    } else {
        format!(".{field}")
    };
    CalcFlowError::InvalidArgument {
        field: format!("static_inputs.{name}{suffix}"),
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        StaticInputSpec, StaticMutability,
        array::provider::{
            infer,
            ir::{DType, Execution, Input, Node, Output, Program},
        },
    };
    use std::{collections::BTreeMap, sync::Arc};
    fn table_template() -> Template {
        let inputs = vec![
            Input {
                name: "observations".into(),
                dtype: DType::Float32,
                shape: vec![None, Some(2)],
            },
            Input {
                name: "coefficients".into(),
                dtype: DType::Float64,
                shape: vec![Some(2), Some(1)],
            },
        ];
        let nodes = vec![
            Node::Input {
                id: "x".into(),
                name: "observations".into(),
            },
            Node::Input {
                id: "w".into(),
                name: "coefficients".into(),
            },
            Node::Matmul {
                id: "y".into(),
                left: "x".into(),
                right: "w".into(),
            },
        ];
        let plan = infer::plan(Program {
            ir_version: 2,
            inputs,
            nodes,
            outputs: vec![Output {
                name: "output".into(),
                node: "y".into(),
                assertion: None,
            }],
            execution: Execution {
                max_elements: 10,
                max_allocation_bytes: 9000,
            },
        })
        .unwrap();
        Template::table("bridge", Arc::new(plan), "observations")
    }
    fn spec(dtype: &str, shape: Vec<u64>) -> StaticInputSpec {
        StaticInputSpec::Array {
            name: "bridge.coefficients".into(),
            mutability: StaticMutability::Static,
            backend: "native".into(),
            dtype: dtype.into(),
            shape,
        }
    }
    #[test]
    fn test_table_static_projection_uses_physical_endpoint_and_rejects_missing_declaration() {
        let template = table_template();
        let external = BTreeMap::from([(
            "bridge.coefficients".into(),
            PortEndpoint::new("bridge", "coefficients").unwrap(),
        )]);
        let specs = BTreeMap::from([("bridge.coefficients".into(), spec("float64", vec![2, 1]))]);
        let records = bind(&[template], &external, &specs).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(
            records[0].ports["coefficients"].as_ref(),
            "bridge.coefficients"
        );
        assert_eq!(records[0].max_bytes, 9000);
        let error = bind(&[table_template()], &external, &BTreeMap::new())
            .err()
            .expect("a declared static RHS needs compiled evidence");
        assert!(
            matches!(error,CalcFlowError::InvalidArgument{field,..} if field=="static_inputs.bridge.coefficients")
        );
    }
    #[test]
    fn test_declared_static_descriptor_must_match_its_native_program_assertions() {
        let external = BTreeMap::from([(
            "bridge.coefficients".into(),
            PortEndpoint::new("bridge", "coefficients").unwrap(),
        )]);
        for (declaration, field) in [
            (spec("float32", vec![2, 1]), "dtype"),
            (spec("float64", vec![1, 2]), "shape"),
        ] {
            let specs = BTreeMap::from([("bridge.coefficients".into(), declaration)]);
            let result = bind(&[table_template()], &external, &specs);
            assert!(
                matches!(result,Err(CalcFlowError::InvalidArgument{field:actual,..}) if actual==format!("static_inputs.bridge.coefficients.{field}"))
            );
        }
    }
    #[test]
    fn test_null_static_rhs_error_uses_the_actual_physical_alias() {
        let template = table_template();
        let physical = "physical";
        let external = BTreeMap::from([(
            physical.into(),
            PortEndpoint::new("bridge", "coefficients").unwrap(),
        )]);
        let mut declaration = spec("float64", vec![2, 1]);
        let StaticInputSpec::Array { name, .. } = &mut declaration else {
            unreachable!()
        };
        *name = physical.into();
        let specs = BTreeMap::from([(physical.into(), declaration)]);
        let consumers = Arc::new(bind(&[table_template()], &external, &specs).unwrap());
        let array = crate::array::NativeArray::from_f64(
            vec![0.0, 2.0],
            vec![2, 1],
            Some(datafusion::arrow::buffer::NullBuffer::from(vec![
                false, true,
            ])),
        )
        .unwrap();
        let batch =
            crate::Batch::external(Arc::new(array), crate::BatchMetadata::default()).unwrap();
        let supplied = BTreeMap::from([(physical.into(), batch)]);
        let scope = super::super::build::prepare(
            &specs,
            &supplied,
            consumers,
            super::super::retention::Mode::Finite,
            &mut || Ok(()),
        )
        .unwrap();
        let result = template.validate_scope(&scope);
        assert!(
            matches!(result,Err(CalcFlowError::InvalidArgument{field,..}) if field=="static_inputs.physical.nulls")
        );
    }
    #[test]
    fn test_unknown_native_static_identity_cannot_bypass_a_trusted_consumer_budget() {
        let external = BTreeMap::from([(
            "bridge.coefficients".into(),
            PortEndpoint::new("bridge", "coefficients").unwrap(),
        )]);
        let specs = BTreeMap::from([("bridge.coefficients".into(), spec("float64", vec![2, 1]))]);
        let result = bind(&[], &external, &specs);
        assert!(
            matches!(result,Err(CalcFlowError::InvalidArgument{field,..}) if field=="static_inputs.bridge.coefficients")
        );
    }
}
