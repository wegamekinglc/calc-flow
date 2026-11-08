use std::{collections::BTreeMap, sync::Arc};

use async_trait::async_trait;
use calc_flow::{
    Batch, BatchMetadata, BatchOperator, BatchOperatorContext, BatchOperatorFactory,
    ExecutionOptions, ExternalOperatorSpec, JsonMap, OperatorMetadata, Port, ProjectSpec,
    ProviderRegistry, Result, UdfRegistrySnapshot, ValidationReport,
    array::{self, NativeArray},
    compile_project, validate_project,
};
use datafusion::arrow::{
    array::Float64Array,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use serde_json::{Value, json};

fn fields() -> Value {
    json!([{"name":"a","data_type":"float64","nullable":true},{"name":"b","data_type":"float64","nullable":true}])
}
fn source(name: &str) -> Value {
    json!({"id":format!("source_{}",name.replace('.', "_")),"input":name,"format":"inline_json","data":[]})
}
fn static_array(name: &str, backend: &str) -> Value {
    json!({"name":name,"kind":"array","mutability":"static","backend":backend,"dtype":"float64","shape":[2,1]})
}
fn matmul_nodes(input: &str, shifted: bool) -> Vec<Value> {
    let mut nodes = vec![
        json!({"id":"x","op":"input","name":input}),
        json!({"id":"w","op":"input","name":"beta"}),
    ];
    if shifted {
        nodes.extend([
            json!({"id":"one","op":"literal","dtype":"float64","value":1.0}),
            json!({"id":"adjusted","op":"binary","function":"add","left":"x","right":"one"}),
        ]);
    }
    nodes.push(
        json!({"id":"y","op":"matmul","left":if shifted {"adjusted"} else {"x"},"right":"w"}),
    );
    nodes
}
fn native_node(id: &str, table: bool, shifted: bool) -> Value {
    let input = if table { "rows" } else { "vectors" };
    let nodes = matmul_nodes(input, shifted);
    let inputs = if table {
        json!([{"name":input,"kind":"table","columns":["a","b"]},{"name":"beta","kind":"array","binding":"static","dtype":"float64","shape":[2,1]}])
    } else {
        json!([{"name":input,"dtype":"float64","shape":[null,2]},{"name":"beta","dtype":"float64","shape":[2,1]}])
    };
    let mut output = json!({"name":"output","node":"y","dtype":"float64","shape":[null,1]});
    if table {
        output["kind"] = json!("array");
    }
    json!({"id":id,"operator":{"kind":"external","provider":"native","name":if table {"table_array"} else {"array_expr"},"version":"1",
        "options":{"ir_version":2,"inputs":inputs,"nodes":nodes,"outputs":[output]}},
        "input_ports":[{"name":input,"kind":if table {"table"} else {"array"},"schema":if table {fields()} else {json!([])}},{"name":"beta","kind":"array"}],
        "output_ports":[{"name":"output","kind":"array"}]})
}
fn project(nodes: Vec<Value>, statics: Vec<Value>, sources: Vec<Value>) -> ProjectSpec {
    serde_json::from_value(json!({"format_version":3,"id":"static_project","name":"static_project",
        "runtime":{"mode":"batch","options":{}},"graph":{"name":"static_project","nodes":Value::Array(nodes)},"static_inputs":Value::Array(statics),"data_sources":Value::Array(sources)})).unwrap()
}
fn registry() -> ProviderRegistry {
    let registry = ProviderRegistry::default();
    array::register_native(&registry).unwrap();
    registry
        .register_batch("legacy", "echo", "1", Arc::new(LegacyFactory))
        .unwrap();
    registry
}
fn issue(report: &ValidationReport, path: &str, code: &str) {
    assert!(!report.valid, "unexpected valid report: {report:?}");
    assert!(
        report
            .issues
            .iter()
            .any(|issue| issue.path == path && issue.code == code),
        "{report:?}"
    );
}
fn matrix(values: Vec<f64>, shape: Vec<usize>) -> Batch {
    Batch::external(
        Arc::new(NativeArray::from_f64(values, shape, None).unwrap()),
        BatchMetadata::default(),
    )
    .unwrap()
}
fn rows() -> Batch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Float64, true),
        Field::new("b", DataType::Float64, true),
    ]));
    let record = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![1.0, 3.0])),
            Arc::new(Float64Array::from(vec![2.0, 4.0])),
        ],
    )
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}
fn values(batch: &Batch) -> Vec<f64> {
    batch
        .external_payload()
        .unwrap()
        .as_any()
        .downcast_ref::<NativeArray>()
        .unwrap()
        .as_arrow()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .values()
        .to_vec()
}

#[tokio::test]
async fn test_native_finite_static_needs_no_dynamic_datasource_and_executes() {
    let project = project(
        vec![native_node("native", false, false)],
        vec![static_array("beta", "native")],
        vec![source("vectors")],
    );
    let registry = registry();
    let udfs = UdfRegistrySnapshot::default();
    let report = validate_project(&project, &registry, &udfs);
    assert!(report.valid, "{report:?}");
    let plan = compile_project(&project, &registry, &udfs).unwrap();
    assert_eq!(report.fingerprint.as_deref(), Some(plan.fingerprint()));
    let result = plan
        .execute(
            BTreeMap::from([
                (
                    "vectors".into(),
                    matrix(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
                ),
                ("beta".into(), matrix(vec![2.0, 3.0], vec![2, 1])),
            ]),
            ExecutionOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(values(&result.outputs["output"]), [8.0, 18.0]);
}

fn fanout() -> ProjectSpec {
    project(
        vec![
            native_node("left", true, false),
            native_node("right", true, true),
            native_node("vector", false, false),
        ],
        ["left.beta", "right.beta", "vector.beta"]
            .map(|name| static_array(name, "native"))
            .to_vec(),
        ["left.rows", "right.rows", "vectors"].map(source).to_vec(),
    )
}
#[tokio::test]
async fn test_native_qualified_static_fanout_validates_compiles_and_executes_table_and_array_roots()
{
    let project = fanout();
    let registry = registry();
    let udfs = UdfRegistrySnapshot::default();
    let report = validate_project(&project, &registry, &udfs);
    assert!(report.valid, "{report:?}");
    let plan = compile_project(&project, &registry, &udfs).unwrap();
    assert_eq!(report.fingerprint.as_deref(), Some(plan.fingerprint()));
    let weights = matrix(vec![2.0, 3.0], vec![2, 1]);
    let mut inputs = ["left.beta", "right.beta", "vector.beta"]
        .map(|name| (name.into(), weights.clone()))
        .into_iter()
        .collect::<BTreeMap<_, _>>();
    inputs.extend([
        ("left.rows".into(), rows()),
        ("right.rows".into(), rows()),
        (
            "vectors".into(),
            matrix(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
        ),
    ]);
    let result = plan
        .execute(inputs, ExecutionOptions::default())
        .await
        .unwrap();
    assert_eq!(values(&result.outputs["left.output"]), [8.0, 18.0]);
    assert_eq!(values(&result.outputs["right.output"]), [13.0, 23.0]);
    assert_eq!(values(&result.outputs["vector.output"]), [8.0, 18.0]);
}

#[test]
fn test_native_static_names_must_match_actual_exposed_bindings_and_kinds() {
    let registry = registry();
    let udfs = UdfRegistrySnapshot::default();
    for (name, field, code) in [
        ("beta", "name", "unknown_binding"),
        ("missing.beta", "name", "unknown_binding"),
        ("left.missing", "name", "unknown_binding"),
        ("left.rows", "kind", "incompatible_batch_kind"),
    ] {
        let mut project = fanout();
        let calc_flow::StaticInputSpec::Array { name: actual, .. } = &mut project.static_inputs[0]
        else {
            unreachable!()
        };
        *actual = name.into();
        let report = validate_project(&project, &registry, &udfs);
        issue(&report, &format!("static_inputs[0].{field}"), code);
        assert!(
            compile_project(&project, &registry, &udfs)
                .unwrap_err()
                .to_string()
                .contains(&format!("static_inputs[0].{field}"))
        );
    }
}
#[test]
fn test_native_static_cannot_also_be_a_finite_datasource() {
    let project = project(
        vec![native_node("native", false, false)],
        vec![static_array("beta", "native")],
        vec![source("vectors"), source("beta")],
    );
    let registry = registry();
    let udfs = UdfRegistrySnapshot::default();
    issue(
        &validate_project(&project, &registry, &udfs),
        "data_sources",
        "source_input_mismatch",
    );
    assert!(compile_project(&project, &registry, &udfs).is_err());
}

struct LegacyFactory;
struct LegacyOperator {
    inputs: Vec<Port>,
    outputs: Vec<Port>,
}
impl OperatorMetadata for LegacyOperator {
    fn name(&self) -> &'static str {
        "legacy:echo@1"
    }
    fn input_ports(&self) -> &[Port] {
        &self.inputs
    }
    fn output_ports(&self) -> &[Port] {
        &self.outputs
    }
    fn configuration(&self) -> JsonMap {
        BTreeMap::new()
    }
}
#[async_trait]
impl BatchOperator for LegacyOperator {
    async fn process(
        &mut self,
        inputs: &BTreeMap<String, Batch>,
        _context: &BatchOperatorContext<'_>,
    ) -> Result<BTreeMap<String, Batch>> {
        Ok(BTreeMap::from([(
            "output".into(),
            inputs[self.inputs[0].name()].clone(),
        )]))
    }
}
impl BatchOperatorFactory for LegacyFactory {
    fn create(
        &self,
        _spec: &ExternalOperatorSpec,
        inputs: Vec<Port>,
        outputs: Vec<Port>,
    ) -> Result<Box<dyn BatchOperator>> {
        Ok(Box::new(LegacyOperator { inputs, outputs }))
    }
}
fn legacy_node(input: &str) -> Value {
    json!({"id":"legacy","operator":{"kind":"external","provider":"legacy","name":"echo","version":"1","options":{}},"input_ports":[{"name":input,"kind":"array"}],"output_ports":[{"name":"output","kind":"array"}]})
}
fn table_node() -> Value {
    json!({"id":"table","operator":{"kind":"expression","expression":"","select":["a","b"]},"input_ports":[{"name":"input","kind":"table","schema":fields()}],"output_ports":[{"name":"output","kind":"table","schema":fields()}]})
}
#[test]
fn test_legacy_array_names_and_batch_placeholders_keep_their_contract() {
    let registry = registry();
    let udfs = UdfRegistrySnapshot::default();
    for backend in ["numpy", "jax"] {
        let mut project = project(
            vec![legacy_node("legacy_values")],
            vec![static_array("legacy_values", backend)],
            vec![source("legacy_values")],
        );
        assert!(validate_project(&project, &registry, &udfs).valid);
        compile_project(&project, &registry, &udfs).unwrap();
        project.data_sources.clear();
        issue(
            &validate_project(&project, &registry, &udfs),
            "data_sources",
            "source_input_mismatch",
        );
        let calc_flow::StaticInputSpec::Array { name, .. } = &mut project.static_inputs[0] else {
            unreachable!()
        };
        *name = "legacy.legacy_values".into();
        issue(
            &validate_project(&project, &registry, &udfs),
            "static_inputs[0].name",
            "invalid_field",
        );
    }
}
#[test]
fn test_mixed_native_static_and_legacy_array_table_roots_keep_legacy_source_coverage() {
    let project = project(
        vec![
            native_node("native", false, false),
            legacy_node("legacy_values"),
            table_node(),
        ],
        vec![
            static_array("beta", "native"),
            static_array("legacy_values", "numpy"),
            json!({"kind":"table","name":"input","mutability":"static","schema":fields()}),
        ],
        ["vectors", "legacy_values", "input"].map(source).to_vec(),
    );
    let registry = registry();
    let udfs = UdfRegistrySnapshot::default();
    let report = validate_project(&project, &registry, &udfs);
    assert!(report.valid, "{report:?}");
    let plan = compile_project(&project, &registry, &udfs).unwrap();
    assert_eq!(report.fingerprint.as_deref(), Some(plan.fingerprint()));
    for omitted in ["legacy_values", "input"] {
        let mut missing = project.clone();
        missing
            .data_sources
            .retain(|source| source.input != omitted);
        issue(
            &validate_project(&missing, &registry, &udfs),
            "data_sources",
            "source_input_mismatch",
        );
        assert!(compile_project(&missing, &registry, &udfs).is_err());
    }
}

fn omitted_builtin_port_project(name: &str) -> ProjectSpec {
    let mut native = native_node("native", false, false);
    native["operator"]["options"]["inputs"][1]["name"] = json!("input");
    native["operator"]["options"]["nodes"][1]["name"] = json!("input");
    native["input_ports"][1]["name"] = json!("input");
    let builtin = json!({"id":"builtin","operator":{"kind":"expression","expression":"one = 1"}});
    project(
        vec![native, builtin],
        vec![static_array(name, "native")],
        ["vectors", "builtin.input"].map(source).to_vec(),
    )
}
#[test]
fn test_native_names_include_omitted_builtin_ports_and_reject_unnecessary_qualification() {
    let registry = registry();
    let udfs = UdfRegistrySnapshot::default();
    let exact = omitted_builtin_port_project("native.input");
    let report = validate_project(&exact, &registry, &udfs);
    assert!(report.valid, "{report:?}");
    compile_project(&exact, &registry, &udfs).unwrap();
    for (name, field, code) in [
        ("input", "name", "unknown_binding"),
        ("builtin.input", "kind", "incompatible_batch_kind"),
    ] {
        let project = omitted_builtin_port_project(name);
        issue(
            &validate_project(&project, &registry, &udfs),
            &format!("static_inputs[0].{field}"),
            code,
        );
        assert!(compile_project(&project, &registry, &udfs).is_err());
    }
    let overqualified = project(
        vec![native_node("native", false, false)],
        vec![static_array("native.beta", "native")],
        vec![source("vectors")],
    );
    issue(
        &validate_project(&overqualified, &registry, &udfs),
        "static_inputs[0].name",
        "unknown_binding",
    );
    assert!(compile_project(&overqualified, &registry, &udfs).is_err());
}

struct DerivedFactory(Arc<std::sync::atomic::AtomicUsize>);
impl BatchOperatorFactory for DerivedFactory {
    fn create(
        &self,
        _spec: &ExternalOperatorSpec,
        _inputs: Vec<Port>,
        _outputs: Vec<Port>,
    ) -> Result<Box<dyn BatchOperator>> {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(Box::new(LegacyOperator {
            inputs: vec![Port::new("beta", calc_flow::BatchKind::Array, true, None)?],
            outputs: vec![Port::new(
                "output",
                calc_flow::BatchKind::Array,
                true,
                None,
            )?],
        }))
    }
}
#[test]
fn test_native_names_use_derived_factory_metadata_without_creating_operators_twice() {
    let registry = registry();
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    registry
        .register_batch(
            "derived",
            "echo",
            "1",
            Arc::new(DerivedFactory(Arc::clone(&calls))),
        )
        .unwrap();
    let derived = json!({"id":"derived","operator":{"kind":"external","provider":"derived","name":"echo","version":"1","options":{}}});
    let project = project(
        vec![native_node("native", false, false), derived],
        vec![static_array("native.beta", "native")],
        ["vectors", "derived.beta"].map(source).to_vec(),
    );
    let udfs = UdfRegistrySnapshot::default();
    let report = validate_project(&project, &registry, &udfs);
    assert!(report.valid, "{report:?}");
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    compile_project(&project, &registry, &udfs).unwrap();
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 2);
}

#[test]
fn test_native_static_validator_rejects_an_untrusted_consumer_without_a_fingerprint() {
    let project = project(
        vec![legacy_node("legacy_values")],
        vec![static_array("legacy_values", "native")],
        Vec::new(),
    );
    let registry = registry();
    let udfs = UdfRegistrySnapshot::default();
    let error = compile_project(&project, &registry, &udfs).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("no trusted native consumer allocation budget"),
        "{error}"
    );
    let report = validate_project(&project, &registry, &udfs);
    assert!(
        !report.valid,
        "native static validation skipped actual trusted consumer admission: {report:?}"
    );
    assert!(report.fingerprint.is_none());
}
