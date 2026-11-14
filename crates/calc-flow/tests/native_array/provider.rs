use std::{collections::BTreeMap, sync::Arc};

use calc_flow::{
    Batch, BatchKind, BatchMetadata, BatchOperatorContext, BatchOperatorFactory, CancellationToken,
    ExternalOperatorSpec, JsonMap, Port, ProviderRegistry, RunContext,
    array::{NativeArray, NativeArrayFactory, register_native},
};
use datafusion::arrow::array::Float64Array;
use serde_json::{Value, json};

#[test]
fn test_native_registration_is_exact_batch_only_and_duplicate_atomic() {
    let registry = ProviderRegistry::default();
    register_native(&registry).unwrap();
    let before = registry.resolve_batch("native", "array_expr", "1").unwrap();
    assert!(register_native(&registry).is_err());
    assert!(Arc::ptr_eq(
        &before,
        &registry.resolve_batch("native", "array_expr", "1").unwrap()
    ));
    assert!(
        registry
            .resolve_stream("native", "array_expr", "1")
            .is_err()
    );
    let descriptor = NativeArrayFactory::capability_descriptor();
    assert_eq!(descriptor["provider"], "native");
    assert_eq!(descriptor["port_binding"], "options");
    assert_eq!(descriptor["modes"], json!(["batch"]));
    assert_eq!(descriptor["checkpoint_support"], "unproven");
    let schema: Value =
        serde_json::from_str(descriptor["options_json_schema"].as_str().unwrap()).unwrap();
    assert_eq!(schema["additionalProperties"], false);
    let wrong = ExternalOperatorSpec::new("other", "array_expr", "1", alias_options()).unwrap();
    assert!(
        NativeArrayFactory::default()
            .create(&wrong, ports(&["x"]), ports(&["output"]))
            .is_err()
    );
}

#[tokio::test]
async fn test_native_alias_outputs_preserve_owner_and_recheck_invocation_names() {
    let spec = ExternalOperatorSpec::new("native", "array_expr", "1", alias_options()).unwrap();
    let mut operator = NativeArrayFactory::default()
        .create(&spec, ports(&["x"]), ports(&["output"]))
        .unwrap();
    let array = NativeArray::from_f64(vec![1.0, 3.0, 5.0], vec![3], None).unwrap();
    let source = Arc::clone(array.as_arrow());
    let inputs = BTreeMap::from([("x".into(), batch(array))]);
    let run = RunContext::new(BTreeMap::new(), None, CancellationToken::new()).unwrap();
    let result = operator
        .process(&inputs, &BatchOperatorContext { run: &run })
        .await
        .unwrap();
    let output = result["output"]
        .external_payload()
        .unwrap()
        .as_any()
        .downcast_ref::<NativeArray>()
        .unwrap();
    assert!(Arc::ptr_eq(output.as_arrow(), &source));
    assert_eq!(output.shape(), &[3]);
    let wrong = BTreeMap::from([("y".into(), inputs["x"].clone())]);
    assert!(
        operator
            .process(&wrong, &BatchOperatorContext { run: &run })
            .await
            .is_err()
    );
    let values = output
        .as_arrow()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_eq!(values.values().as_ref(), &[1.0, 3.0, 5.0]);
}

fn alias_options() -> JsonMap {
    serde_json::from_value(json!({
        "ir_version": 1,
        "inputs": [{"name": "x", "dtype": "float64", "shape": [null]}],
        "nodes": [{"id": "x", "op": "input", "name": "x"}],
        "outputs": [{"name": "output", "node": "x", "dtype": "float64", "shape": [null]}]
    }))
    .unwrap()
}

#[tokio::test]
async fn test_permuted_typed_graph_combines_full_reduction_selection_and_fresh_rolling() {
    let options = json!({
        "ir_version":1,
        "inputs":[{"name":"x","dtype":"float64","shape":[null]}],
        "nodes":[
            {"id":"selected","op":"where","condition":"positive","if_true":"centered","if_false":"zero"},
            {"id":"smooth","op":"ewma","input":"x","span":3},
            {"id":"windowed","op":"rolling_mean","input":"x","window":2},
            {"id":"positive","op":"binary","function":"greater","left":"centered","right":"zero"},
            {"id":"centered","op":"binary","function":"subtract","left":"x","right":"average"},
            {"id":"average","op":"reduce","function":"mean","input":"x"},
            {"id":"zero","op":"literal","dtype":"float64","value":0.0},
            {"id":"x","op":"input","name":"x"}
        ],
        "outputs":[
            {"name":"selected","node":"selected","dtype":"float64","shape":[null]},
            {"name":"windowed","node":"windowed","dtype":"float64","shape":[null]},
            {"name":"smooth","node":"smooth","dtype":"float64","shape":[null]}
        ]
    });
    let spec = spec(options);
    let mut operator = NativeArrayFactory::default()
        .create(
            &spec,
            ports(&["x"]),
            ports(&["smooth", "selected", "windowed"]),
        )
        .unwrap();
    let input = BTreeMap::from([(
        "x".into(),
        batch(NativeArray::from_f64(vec![1.0, 3.0, 5.0], vec![3], None).unwrap()),
    )]);
    let run = RunContext::new(BTreeMap::new(), None, CancellationToken::new()).unwrap();
    for _ in 0..2 {
        let outputs = operator
            .process(&input, &BatchOperatorContext { run: &run })
            .await
            .unwrap();
        assert_values(&outputs["selected"], &[0.0, 0.0, 2.0]);
        assert_values(&outputs["windowed"], &[1.0, 2.0, 4.0]);
        assert_values(&outputs["smooth"], &[1.0, 2.0, 3.5]);
    }
}

#[tokio::test]
async fn test_unknown_vector_extents_are_rechecked_and_output_assertions_are_exact() {
    let mut options = json!({"ir_version":1,
        "inputs":[{"name":"x","dtype":"float64","shape":[null]},{"name":"y","dtype":"float64","shape":[null]}],
        "nodes":[{"id":"x","op":"input","name":"x"},{"id":"y","op":"input","name":"y"},{"id":"z","op":"binary","function":"add","left":"x","right":"y"}],
        "outputs":[{"name":"output","node":"z","dtype":"float64","shape":[null]}]});
    let mut operator = NativeArrayFactory::default()
        .create(
            &spec(options.clone()),
            ports(&["y", "x"]),
            ports(&["output"]),
        )
        .unwrap();
    let inputs = BTreeMap::from([
        (
            "x".into(),
            batch(NativeArray::from_f64(vec![1.0, 2.0], vec![2], None).unwrap()),
        ),
        (
            "y".into(),
            batch(NativeArray::from_f64(vec![3.0], vec![1], None).unwrap()),
        ),
    ]);
    let run = RunContext::new(BTreeMap::new(), None, CancellationToken::new()).unwrap();
    assert!(
        operator
            .process(&inputs, &BatchOperatorContext { run: &run })
            .await
            .is_err()
    );
    options["outputs"][0]["shape"] = json!([8]);
    assert!(
        NativeArrayFactory::default()
            .create(&spec(options), ports(&["x", "y"]), ports(&["output"]))
            .is_err()
    );
}

fn spec(options: Value) -> ExternalOperatorSpec {
    ExternalOperatorSpec::new(
        "native",
        "array_expr",
        "1",
        serde_json::from_value(options).unwrap(),
    )
    .unwrap()
}

fn assert_values(batch: &Batch, expected: &[f64]) {
    let payload = batch.external_payload().unwrap();
    let array = payload.as_any().downcast_ref::<NativeArray>().unwrap();
    let values = array
        .as_arrow()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_eq!(
        values
            .values()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
}

fn ports(names: &[&str]) -> Vec<Port> {
    names
        .iter()
        .map(|name| Port::new(name, BatchKind::Array, true, None).unwrap())
        .collect()
}

fn batch(array: NativeArray) -> Batch {
    Batch::external(Arc::new(array), BatchMetadata::default()).unwrap()
}

#[test]
fn test_literal_exponents_validate_original_json_numbers_before_float32_rounding() {
    let mut options = json!({"ir_version":1,
        "inputs":[{"name":"x","dtype":"float32","shape":[null]}],
        "nodes":[{"id":"x","op":"input","name":"x"},{"id":"p","op":"literal","dtype":"float32","value":2},{"id":"result","op":"binary","function":"pow","left":"x","right":"p"}],
        "outputs":[{"name":"output","node":"result","dtype":"float32","shape":[null]}]});
    let make = |options| {
        NativeArrayFactory::default().create(&spec(options), ports(&["x"]), ports(&["output"]))
    };
    assert!(make(options.clone()).is_ok());
    for value in [
        json!(64.000_000_1),
        json!(-64.000_000_1),
        json!(9_007_199_254_740_993_u64),
        json!(-9_007_199_254_740_993_i64),
        Value::Null,
        json!([2]),
    ] {
        options["nodes"][1]["value"] = value;
        assert!(make(options.clone()).is_err());
    }
    options["nodes"][1] = json!({"id":"p","op":"reduce","function":"mean","input":"x"});
    options["outputs"][0]["dtype"] = json!("float64");
    assert!(make(options).is_err());
}

#[test]
fn test_program_rejects_unknown_fields_duplicate_cycles_unreachable_and_wrong_ports() {
    let factory = NativeArrayFactory::default();
    let original = alias_options();
    let mut duplicate = original.clone();
    duplicate
        .get_mut("nodes")
        .unwrap()
        .as_array_mut()
        .unwrap()
        .push(json!({"id":"x","op":"input","name":"x"}));
    assert!(
        factory
            .create(
                &ExternalOperatorSpec::new("native", "array_expr", "1", duplicate).unwrap(),
                ports(&["x"]),
                ports(&["output"])
            )
            .is_err()
    );
    for field in ["unexpected", "ir_version"] {
        let mut options = original.clone();
        options.insert(field.into(), json!(99));
        assert!(
            factory
                .create(
                    &ExternalOperatorSpec::new("native", "array_expr", "1", options).unwrap(),
                    ports(&["x"]),
                    ports(&["output"])
                )
                .is_err()
        );
    }
    let mut unreachable = original.clone();
    unreachable
        .get_mut("nodes")
        .unwrap()
        .as_array_mut()
        .unwrap()
        .push(json!({"id":"unused","op":"literal","dtype":"float64","value":1}));
    assert!(
        factory
            .create(
                &ExternalOperatorSpec::new("native", "array_expr", "1", unreachable).unwrap(),
                ports(&["x"]),
                ports(&["output"])
            )
            .is_err()
    );
    let mut cycle = original.clone();
    cycle.insert(
        "nodes".into(),
        json!([{"id":"x","op":"unary","function":"neg","input":"x"}]),
    );
    assert!(
        factory
            .create(
                &ExternalOperatorSpec::new("native", "array_expr", "1", cycle).unwrap(),
                ports(&["x"]),
                ports(&["output"])
            )
            .is_err()
    );
    let correct = ExternalOperatorSpec::new("native", "array_expr", "1", original).unwrap();
    assert!(
        factory
            .create(
                &correct,
                vec![Port::new("x", BatchKind::Array, false, None).unwrap()],
                ports(&["output"])
            )
            .is_err()
    );
    assert!(
        factory
            .create(
                &correct,
                vec![Port::new("x", BatchKind::Table, true, None).unwrap()],
                ports(&["output"])
            )
            .is_err()
    );
}

#[tokio::test]
async fn test_program_budget_is_shared_before_work_and_alias_outputs_count_by_name() {
    let mut options = serde_json::to_value(alias_options()).unwrap();
    options["nodes"] = json!([{"id":"x","op":"input","name":"x"},{"id":"first","op":"unary","function":"neg","input":"x"},{"id":"second","op":"unary","function":"abs","input":"first"}]);
    options["outputs"][0]["node"] = json!("second");
    let each = 3 * size_of::<f64>() + size_of::<usize>();
    options["execution"] = json!({"max_allocation_bytes":each});
    let input = BTreeMap::from([(
        "x".into(),
        batch(NativeArray::from_f64(vec![1.0, 2.0, 3.0], vec![3], None).unwrap()),
    )]);
    let run = RunContext::new(BTreeMap::new(), None, CancellationToken::new()).unwrap();
    let make = |options| {
        NativeArrayFactory::default()
            .create(&spec(options), ports(&["x"]), ports(&["output"]))
            .unwrap()
    };
    assert!(
        make(options.clone())
            .process(&input, &BatchOperatorContext { run: &run })
            .await
            .is_err()
    );
    options["execution"]["max_allocation_bytes"] = json!(2 * each);
    assert_values(
        &make(options)
            .process(&input, &BatchOperatorContext { run: &run })
            .await
            .unwrap()["output"],
        &[1.0, 2.0, 3.0],
    );
    let mut aliases = serde_json::to_value(alias_options()).unwrap();
    aliases["outputs"] = json!([{"name":"a","node":"x","dtype":"float64","shape":[null]},{"name":"b","node":"x","dtype":"float64","shape":[null]}]);
    aliases["execution"] = json!({"max_elements":5,"max_allocation_bytes":0});
    let mut operator = NativeArrayFactory::default()
        .create(&spec(aliases), ports(&["x"]), ports(&["a", "b"]))
        .unwrap();
    assert!(
        operator
            .process(&input, &BatchOperatorContext { run: &run })
            .await
            .is_err()
    );
}

#[tokio::test(flavor = "current_thread")]
async fn test_long_native_node_yields_to_run_cancellation_before_returning() {
    let mut options = serde_json::to_value(alias_options()).unwrap();
    options["nodes"] = json!([{"id":"x","op":"input","name":"x"},{"id":"result","op":"ewma","input":"x","span":3}]);
    options["outputs"][0]["node"] = json!("result");
    let mut operator = NativeArrayFactory::default()
        .create(&spec(options), ports(&["x"]), ports(&["output"]))
        .unwrap();
    let input = BTreeMap::from([(
        "x".into(),
        batch(NativeArray::from_f64(vec![1.0; 2_000_000], vec![2_000_000], None).unwrap()),
    )]);
    let cancellation = CancellationToken::new();
    let run = RunContext::new(BTreeMap::new(), None, cancellation.clone()).unwrap();
    let cancel = tokio::spawn(async move {
        cancellation.cancel();
    });
    let result = operator
        .process(&input, &BatchOperatorContext { run: &run })
        .await;
    cancel.await.unwrap();
    assert!(matches!(
        result,
        Err(calc_flow::CalcFlowError::Cancelled { .. })
    ));
}

#[test]
fn test_capability_schema_exposes_strict_ir_variants_and_bounds_as_canonical_json() {
    let descriptor = NativeArrayFactory::capability_descriptor();
    let text = descriptor["options_json_schema"].as_str().unwrap();
    let schema: Value = serde_json::from_str(text).unwrap();
    assert_eq!(calc_flow::canonical_json(&schema).unwrap(), text);
    let schema = &schema["oneOf"][0];
    assert_eq!(schema["properties"]["inputs"]["maxItems"], 32);
    assert_eq!(schema["properties"]["outputs"]["maxItems"], 16);
    assert_eq!(schema["properties"]["nodes"]["maxItems"], 128);
    let variants = schema["properties"]["nodes"]["items"]["oneOf"]
        .as_array()
        .unwrap();
    assert_eq!(variants.len(), 10);
    for variant in variants {
        assert_eq!(variant["additionalProperties"], false);
        assert!(
            variant["required"]
                .as_array()
                .unwrap()
                .contains(&json!("id"))
        );
        assert!(
            variant["required"]
                .as_array()
                .unwrap()
                .contains(&json!("op"))
        );
    }
}

#[test]
fn test_options_encoding_and_json_depth_fail_before_typed_copy() {
    let factory = NativeArrayFactory::default();
    let mut options = alias_options();
    options.insert("padding".into(), json!("\n".repeat(33_000)));
    let error = factory
        .create(
            &ExternalOperatorSpec::new("native", "array_expr", "1", options).unwrap(),
            ports(&["x"]),
            ports(&["output"]),
        )
        .err()
        .unwrap();
    assert!(
        matches!(error,calc_flow::CalcFlowError::InvalidArgument{field,..} if field=="options.bytes")
    );
    let mut value = Value::Null;
    for _ in 0..40 {
        value = json!([value]);
    }
    let mut options = alias_options();
    options.insert("padding".into(), value);
    let error = factory
        .create(
            &ExternalOperatorSpec::new("native", "array_expr", "1", options).unwrap(),
            ports(&["x"]),
            ports(&["output"]),
        )
        .err()
        .unwrap();
    assert!(
        matches!(error,calc_flow::CalcFlowError::InvalidArgument{field,..} if field=="options.depth")
    );
}

#[test]
fn test_node_depth_count_and_named_port_limits_accept_exact_boundaries() {
    let mut options = serde_json::to_value(alias_options()).unwrap();
    for index in 1..24 {
        options["nodes"].as_array_mut().unwrap().push(json!({"id":format!("n{index}"),"op":"unary","function":"neg","input":if index==1 {"x".into()} else {format!("n{}",index-1)}}));
    }
    options["outputs"][0]["node"] = json!("n23");
    assert!(create_from_declarations(options.clone()).is_ok());
    options["nodes"]
        .as_array_mut()
        .unwrap()
        .push(json!({"id":"n24","op":"unary","function":"neg","input":"n23"}));
    options["outputs"][0]["node"] = json!("n24");
    assert!(create_from_declarations(options).is_err());
    let mut nodes: Vec<_> = (0..64)
        .map(|index| json!({"id":format!("n{index}"),"op":"literal","dtype":"float64","value":1}))
        .collect();
    let mut layer: Vec<_> = (0..64).map(|index| format!("n{index}")).collect();
    while layer.len() > 1 {
        layer = layer
            .chunks_exact(2)
            .map(|pair| {
                let id = format!("n{}", nodes.len());
                nodes.push(
                    json!({"id":id,"op":"binary","function":"add","left":pair[0],"right":pair[1]}),
                );
                id
            })
            .collect();
    }
    nodes.push(json!({"id":"root","op":"unary","function":"neg","input":layer[0]}));
    let mut options = serde_json::to_value(alias_options()).unwrap();
    options["nodes"] = json!(nodes);
    options["outputs"][0]["node"] = json!("root");
    options["outputs"][0]["shape"] = json!([]);
    assert_eq!(options["nodes"].as_array().unwrap().len(), 128);
    assert!(create_from_declarations(options.clone()).is_ok());
    options["nodes"]
        .as_array_mut()
        .unwrap()
        .push(json!({"id":"extra","op":"unary","function":"neg","input":"root"}));
    options["outputs"][0]["node"] = json!("extra");
    assert!(create_from_declarations(options).is_err());
    let mut options = serde_json::to_value(alias_options()).unwrap();
    for index in 1..32 {
        options["inputs"]
            .as_array_mut()
            .unwrap()
            .push(json!({"name":format!("x{index}"),"dtype":"float64","shape":[null]}));
    }
    for index in 1..16 {
        options["outputs"]
            .as_array_mut()
            .unwrap()
            .push(json!({"name":format!("o{index}"),"node":"x","dtype":"float64","shape":[null]}));
    }
    assert!(create_from_declarations(options.clone()).is_ok());
    let mut too_many_inputs = options.clone();
    too_many_inputs["inputs"]
        .as_array_mut()
        .unwrap()
        .push(json!({"name":"extra","dtype":"float64","shape":[null]}));
    assert!(create_from_declarations(too_many_inputs).is_err());
    options["outputs"]
        .as_array_mut()
        .unwrap()
        .push(json!({"name":"extra","node":"x","dtype":"float64","shape":[null]}));
    assert!(create_from_declarations(options).is_err());
}

fn create_from_declarations(
    options: Value,
) -> calc_flow::Result<Box<dyn calc_flow::BatchOperator>> {
    let names = |field: &str| {
        options[field]
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| entry["name"].as_str().unwrap())
            .collect::<Vec<_>>()
    };
    let inputs = ports(&names("inputs"));
    let outputs = ports(&names("outputs"));
    NativeArrayFactory::default().create(&spec(options), inputs, outputs)
}

#[derive(Debug)]
struct PretendNative;
impl calc_flow::ExternalPayload for PretendNative {
    fn backend(&self) -> &'static str {
        "native"
    }
    fn len(&self) -> usize {
        3
    }
    fn estimated_bytes(&self) -> usize {
        24
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[tokio::test]
async fn test_invocations_reject_backend_spoofing_dtype_and_concrete_shape_mismatches() {
    let mut options = alias_options();
    options.get_mut("inputs").unwrap()[0]["shape"] = json!([3]);
    options.get_mut("outputs").unwrap()[0]["shape"] = json!([3]);
    let specification = ExternalOperatorSpec::new("native", "array_expr", "1", options).unwrap();
    let mut operator = NativeArrayFactory::default()
        .create(&specification, ports(&["x"]), ports(&["output"]))
        .unwrap();
    let run = RunContext::new(BTreeMap::new(), None, CancellationToken::new()).unwrap();
    for (input, expected) in [
        (
            Batch::external(Arc::new(PretendNative), BatchMetadata::default()).unwrap(),
            "inputs.x.backend",
        ),
        (
            batch(NativeArray::from_f32(vec![1.0; 3], vec![3], None).unwrap()),
            "inputs.x.dtype",
        ),
        (
            batch(NativeArray::from_f64(vec![1.0], vec![], None).unwrap()),
            "inputs.x.shape",
        ),
    ] {
        let result = operator
            .process(
                &BTreeMap::from([("x".into(), input)]),
                &BatchOperatorContext { run: &run },
            )
            .await;
        assert!(
            matches!(result,Err(calc_flow::CalcFlowError::InvalidArgument{field,..}) if field==expected)
        );
    }
}

#[tokio::test]
async fn test_numeric_operation_registry_maps_every_binary_function_to_its_eager_kernel() {
    let cases = [
        ("add", Some(6.0), None),
        ("subtract", Some(-2.0), None),
        ("multiply", Some(8.0), None),
        ("divide", Some(0.5), None),
        ("pow", Some(16.0), None),
        ("equal", None, Some(false)),
        ("not_equal", None, Some(true)),
        ("less", None, Some(true)),
        ("less_equal", None, Some(true)),
        ("greater", None, Some(false)),
        ("greater_equal", None, Some(false)),
    ];
    let mut nodes = vec![
        json!({"id":"x","op":"input","name":"x"}),
        json!({"id":"y","op":"literal","dtype":"float64","value":4}),
    ];
    let mut outputs = Vec::new();
    for (name, number, _) in cases {
        nodes.push(json!({"id":name,"op":"binary","function":name,"left":"x","right":"y"}));
        outputs.push(json!({"name":name,"node":name,"dtype":if number.is_some(){"float64"}else{"bool"},"shape":[]}));
    }
    let mut operator=create_from_declarations(json!({"ir_version":1,"inputs":[{"name":"x","dtype":"float64","shape":[]}],"nodes":nodes,"outputs":outputs})).unwrap();
    let run = RunContext::new(BTreeMap::new(), None, CancellationToken::new()).unwrap();
    let result = operator
        .process(
            &BTreeMap::from([(
                "x".into(),
                batch(NativeArray::from_f64(vec![2.0], vec![], None).unwrap()),
            )]),
            &BatchOperatorContext { run: &run },
        )
        .await
        .unwrap();
    for (name, number, boolean) in cases {
        if let Some(number) = number {
            assert_values(&result[name], &[number]);
        } else {
            let value = result[name]
                .external_payload()
                .unwrap()
                .as_any()
                .downcast_ref::<NativeArray>()
                .unwrap();
            let value = value
                .as_arrow()
                .as_any()
                .downcast_ref::<datafusion::arrow::array::BooleanArray>()
                .unwrap();
            assert_eq!(Some(value.value(0)), boolean);
        }
    }
}

#[test]
fn test_standalone_literals_enforce_exact_integer_bounds_without_pow() {
    let mut options = serde_json::to_value(alias_options()).unwrap();
    options["nodes"] = json!([{"id":"constant","op":"literal","dtype":"float64","value":0}]);
    options["outputs"][0]["node"] = json!("constant");
    options["outputs"][0]["shape"] = json!([]);
    for value in [
        json!(9_007_199_254_740_992_u64),
        json!(-9_007_199_254_740_992_i64),
    ] {
        options["nodes"][0]["value"] = value;
        assert!(create_from_declarations(options.clone()).is_ok());
    }
    for value in [
        json!(9_007_199_254_740_993_u64),
        json!(-9_007_199_254_740_993_i64),
        json!(u64::MAX),
    ] {
        options["nodes"][0]["value"] = value;
        assert!(
            matches!(create_from_declarations(options.clone()),Err(calc_flow::CalcFlowError::InvalidArgument{field,..}) if field=="options.nodes.value")
        );
    }
    options["nodes"][0]["dtype"] = json!("float32");
    options["outputs"][0]["dtype"] = json!("float32");
    options["nodes"][0]["value"] = json!(f64::from(f32::MAX));
    assert!(create_from_declarations(options.clone()).is_ok());
    options["nodes"][0]["value"] = json!(1e100);
    assert!(create_from_declarations(options).is_err());
}

#[tokio::test]
async fn test_preparsed_large_integer_tokens_follow_the_retained_float_representation() {
    let positive = f64::from_bits(0x43f0_0000_0000_0000);
    let negative = f64::from_bits(0xc3f0_0000_0000_0000);
    for (token, expected) in [
        ("18446744073709551616", positive),
        ("18446744073709551617", positive),
        ("-18446744073709551616", negative),
        ("-18446744073709551617", negative),
    ] {
        let parsed: Value = serde_json::from_str(token).unwrap();
        assert!(parsed.is_f64());
        assert_eq!(parsed.as_f64().unwrap().to_bits(), expected.to_bits());
        let mut options = serde_json::to_value(alias_options()).unwrap();
        options["nodes"] =
            json!([{"id":"constant","op":"literal","dtype":"float64","value":parsed}]);
        options["outputs"][0]["node"] = json!("constant");
        options["outputs"][0]["shape"] = json!([]);
        let mut operator = create_from_declarations(options).unwrap();
        let inputs = BTreeMap::from([(
            "x".into(),
            batch(NativeArray::from_f64(vec![], vec![0], None).unwrap()),
        )]);
        let run = RunContext::new(BTreeMap::new(), None, CancellationToken::new()).unwrap();
        let result = operator
            .process(&inputs, &BatchOperatorContext { run: &run })
            .await
            .unwrap();
        assert_values(&result["output"], &[expected]);
    }
}
