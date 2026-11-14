use std::{collections::BTreeMap, sync::Arc};

use calc_flow::{
    Batch, BatchKind, BatchMetadata, BatchOperatorContext, BatchOperatorFactory, CancellationToken,
    ExternalOperatorSpec, JsonMap, Port, RunContext,
    array::{NativeArray, NativeArrayFactory},
};
use datafusion::arrow::array::{Float32Array, Float64Array};
use serde_json::{Value, json};

#[test]
fn test_ir2_query_infers_matmul_axis_output_without_rows_or_assertions() {
    let query = json!({"ir_version":2,
        "inputs":[{"name":"x","dtype":"float32","shape":[null,2]},{"name":"w","dtype":"float32","shape":[2,3]}],
        "nodes":[
            {"id":"rows","op":"reduce","function":"mean","input":"product","axis":1},
            {"id":"product","op":"matmul","left":"x","right":"w"},
            {"id":"x","op":"input","name":"x"},{"id":"w","op":"input","name":"w"}
        ],
        "outputs":[{"name":"output","node":"rows"}],
        "execution":{"max_elements":0,"max_allocation_bytes":0}});
    let result = NativeArrayFactory::infer_program(&mapping(query.clone())).unwrap();
    assert_eq!(
        serde_json::to_value(result).unwrap(),
        json!({"outputs":[{"name":"output","node":"rows","dtype":"float64","shape":[null]}]})
    );
    let mut asserted = query;
    asserted["outputs"][0]["dtype"] = json!("float64");
    assert!(NativeArrayFactory::infer_program(&mapping(asserted)).is_err());
}

#[tokio::test]
async fn test_ir2_executes_matmul_transpose_reshape_and_axis_through_shared_kernels() {
    let options = json!({"ir_version":2,
    "inputs":[{"name":"x","dtype":"float32","shape":[2,2]},{"name":"w","dtype":"float32","shape":[2,2]}],
    "nodes":[
        {"id":"x","op":"input","name":"x"},{"id":"w","op":"input","name":"w"},
        {"id":"y","op":"matmul","left":"x","right":"w"},
        {"id":"t","op":"transpose","input":"y"},
        {"id":"flat","op":"reshape","input":"t","shape":[4]},
        {"id":"rows","op":"reduce","function":"sum","input":"y","axis":1}
    ],
    "outputs":[
        {"name":"y","node":"y","dtype":"float32","shape":[2,2]},
        {"name":"t","node":"t","dtype":"float32","shape":[2,2]},
        {"name":"flat","node":"flat","dtype":"float32","shape":[4]},
        {"name":"rows","node":"rows","dtype":"float64","shape":[2]}
    ]});
    let spec = ExternalOperatorSpec::new("native", "array_expr", "1", mapping(options)).unwrap();
    let mut operator = NativeArrayFactory::default()
        .create(
            &spec,
            ports(&["w", "x"]),
            ports(&["rows", "flat", "t", "y"]),
        )
        .unwrap();
    let inputs = BTreeMap::from([
        (
            "x".into(),
            batch(NativeArray::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], None).unwrap()),
        ),
        (
            "w".into(),
            batch(NativeArray::from_f32(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2], None).unwrap()),
        ),
    ]);
    let run = RunContext::new(BTreeMap::new(), None, CancellationToken::new()).unwrap();
    let result = operator
        .process(&inputs, &BatchOperatorContext { run: &run })
        .await
        .unwrap();
    for (name, shape, expected) in [
        ("y", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]),
        ("t", vec![2, 2], vec![1.0, 3.0, 2.0, 4.0]),
        ("flat", vec![4], vec![1.0, 3.0, 2.0, 4.0]),
    ] {
        let array = native(&result[name]);
        assert_eq!(array.shape(), shape);
        let actual = array
            .as_arrow()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        assert_eq!(actual.values().as_ref(), expected);
    }
    let rows = native(&result["rows"]);
    assert_eq!(rows.shape(), &[2]);
    assert_eq!(
        rows.as_arrow()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .values()
            .as_ref(),
        &[3.0, 7.0]
    );
    assert!(Arc::ptr_eq(
        native(&result["flat"]).as_arrow(),
        native(&result["t"]).as_arrow()
    ));
}

fn mapping(value: Value) -> JsonMap {
    serde_json::from_value(value).unwrap()
}
fn ports(names: &[&str]) -> Vec<Port> {
    names
        .iter()
        .map(|name| Port::new(name, BatchKind::Array, true, None).unwrap())
        .collect()
}
fn batch(value: NativeArray) -> Batch {
    Batch::external(Arc::new(value), BatchMetadata::default()).unwrap()
}
fn native(batch: &Batch) -> &NativeArray {
    batch
        .external_payload()
        .unwrap()
        .as_any()
        .downcast_ref::<NativeArray>()
        .unwrap()
}

#[tokio::test]
async fn test_ir2_reshape_aliases_charge_one_data_buffer_and_each_descriptor() {
    let mut options = json!({"ir_version":2,"inputs":[{"name":"x","dtype":"float64","shape":[4]}],
        "nodes":[{"id":"x","op":"input","name":"x"},{"id":"a","op":"unary","function":"neg","input":"x"},{"id":"b","op":"reshape","input":"a","shape":[2,2]},{"id":"c","op":"reshape","input":"b","shape":[4]}],
        "outputs":[{"name":"output","node":"c","dtype":"float64","shape":[4]}],"execution":{"max_allocation_bytes":64}});
    let input = NativeArray::from_f64(vec![1.0, 2.0, 3.0, 4.0], vec![4], None).unwrap();
    let inputs = BTreeMap::from([("x".into(), batch(input))]);
    let result = run_program(options.clone(), inputs.clone()).await.unwrap();
    assert_eq!(
        native(&result["output"])
            .as_arrow()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .values()
            .as_ref(),
        &[-1.0, -2.0, -3.0, -4.0]
    );
    options["execution"]["max_allocation_bytes"] = json!(63);
    assert!(run_program(options, inputs).await.is_err());
    let mut aliases = json!({"ir_version":2,"inputs":[{"name":"x","dtype":"float64","shape":[4]}],
        "nodes":[{"id":"x","op":"input","name":"x"},{"id":"a","op":"reshape","input":"x","shape":[2,2]},{"id":"b","op":"reshape","input":"a","shape":[4]}],
        "outputs":[{"name":"output","node":"b","dtype":"float64","shape":[4]}],"execution":{"max_allocation_bytes":24}});
    let input = NativeArray::from_f64(vec![1.0, 2.0, 3.0, 4.0], vec![4], None).unwrap();
    let pointer = Arc::clone(input.as_arrow());
    let inputs = BTreeMap::from([("x".into(), batch(input))]);
    let result = run_program(aliases.clone(), inputs.clone()).await.unwrap();
    assert!(Arc::ptr_eq(native(&result["output"]).as_arrow(), &pointer));
    aliases["execution"]["max_allocation_bytes"] = json!(23);
    assert!(run_program(aliases, inputs).await.is_err());
}

#[tokio::test]
async fn test_ir2_transpose_preflight_accounts_for_an_explicit_all_valid_bitmap() {
    let mut options = json!({"ir_version":2,"inputs":[{"name":"x","dtype":"float64","shape":[2,2]}],
        "nodes":[{"id":"x","op":"input","name":"x"},{"id":"t","op":"transpose","input":"x"}],
        "outputs":[{"name":"output","node":"t","dtype":"float64","shape":[2,2]}],"execution":{"max_allocation_bytes":48}});
    let input = NativeArray::from_f64(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        Some(datafusion::arrow::buffer::NullBuffer::new_valid(4)),
    )
    .unwrap();
    let inputs = BTreeMap::from([("x".into(), batch(input))]);
    let error = run_program(options.clone(), inputs.clone())
        .await
        .unwrap_err();
    assert!(
        matches!(error,calc_flow::CalcFlowError::InvalidArgument{field,..} if field=="outputs.bytes")
    );
    options["execution"]["max_allocation_bytes"] = json!(49);
    let result = run_program(options, inputs).await.unwrap();
    assert!(native(&result["output"]).as_arrow().nulls().is_some());
}

async fn run_program(
    options: Value,
    inputs: BTreeMap<String, Batch>,
) -> calc_flow::Result<BTreeMap<String, Batch>> {
    let input_names = inputs.keys().map(String::as_str).collect::<Vec<_>>();
    let output_names = options["outputs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|output| output["name"].as_str().unwrap())
        .collect::<Vec<_>>();
    let in_ports = ports(&input_names);
    let out_ports = ports(&output_names);
    let spec = ExternalOperatorSpec::new("native", "array_expr", "1", mapping(options))?;
    let mut operator = NativeArrayFactory::default().create(&spec, in_ports, out_ports)?;
    let run = RunContext::new(BTreeMap::new(), None, CancellationToken::new())?;
    operator
        .process(&inputs, &BatchOperatorContext { run: &run })
        .await
}

#[test]
fn test_ir2_query_rejects_overflow_created_by_inferred_dimensions() {
    let query = json!({"ir_version":2,
        "inputs":[{"name":"x","dtype":"float64","shape":[usize::MAX,null]},
                  {"name":"w","dtype":"float64","shape":[null,2]}],
        "nodes":[{"id":"x","op":"input","name":"x"},
                 {"id":"w","op":"input","name":"w"},
                 {"id":"y","op":"binary","function":"add","left":"x","right":"w"}],
        "outputs":[{"name":"output","node":"y"}]});
    assert!(NativeArrayFactory::infer_program(&mapping(query)).is_err());
    let product = json!({"ir_version":2,
        "inputs":[{"name":"x","dtype":"float64","shape":[usize::MAX,0]},
                  {"name":"w","dtype":"float64","shape":[0,2]}],
        "nodes":[{"id":"x","op":"input","name":"x"},
                 {"id":"w","op":"input","name":"w"},
                 {"id":"y","op":"matmul","left":"x","right":"w"}],
        "outputs":[{"name":"output","node":"y"}]});
    assert!(NativeArrayFactory::infer_program(&mapping(product)).is_err());
}

#[test]
fn test_versioned_schema_keeps_ir1_fields_closed_and_advertises_ir2_layouts() {
    let capability = NativeArrayFactory::capability_descriptor();
    let schema: Value =
        serde_json::from_str(capability["options_json_schema"].as_str().unwrap()).unwrap();
    let versions = schema["oneOf"].as_array().unwrap();
    assert_eq!(versions.len(), 2);
    for (index, rank) in [1, 2].into_iter().enumerate() {
        let version = &versions[index];
        assert_eq!(version["additionalProperties"], false);
        assert_eq!(version["properties"]["ir_version"]["const"], rank);
        assert_eq!(
            version["properties"]["inputs"]["items"]["properties"]["shape"]["maxItems"],
            rank
        );
        let nodes = version["properties"]["nodes"]["items"]["oneOf"]
            .as_array()
            .unwrap();
        let reduce = nodes
            .iter()
            .find(|node| node["properties"]["op"]["const"] == "reduce")
            .unwrap();
        if index == 0 {
            assert!(reduce["properties"].get("axis").is_none());
            assert_eq!(nodes.len(), 10);
        } else {
            assert_eq!(reduce["properties"]["axis"]["default"], Value::Null);
            assert_eq!(reduce["properties"]["keepdims"]["default"], false);
            assert_eq!(nodes.len(), 13);
            for op in ["matmul", "reshape", "transpose"] {
                assert!(
                    nodes
                        .iter()
                        .any(|node| node["properties"]["op"]["const"] == op)
                );
            }
        }
    }
}

#[test]
fn test_ir2_query_preserves_axis_geometry_and_rejects_invalid_axes() {
    let base = json!({"ir_version":2,"inputs":[{"name":"x","dtype":"float32","shape":[null,3]}],
        "nodes":[{"id":"x","op":"input","name":"x"},{"id":"y","op":"reduce","function":"max","input":"x"}],
        "outputs":[{"name":"output","node":"y"}]});
    for (axis, keepdims, expected) in [
        (Value::Null, false, json!([])),
        (Value::Null, true, json!([1, 1])),
        (json!(0), false, json!([3])),
        (json!(0), true, json!([1, 3])),
        (json!(1), false, json!([null])),
        (json!(1), true, json!([null, 1])),
    ] {
        let mut query = base.clone();
        query["nodes"][1]["axis"] = axis;
        query["nodes"][1]["keepdims"] = json!(keepdims);
        let result = NativeArrayFactory::infer_program(&mapping(query)).unwrap();
        assert_eq!(result["outputs"][0]["shape"], expected);
        assert_eq!(result["outputs"][0]["dtype"], "float32");
    }
    for axis in [json!(-1), json!(2), json!([0, 1])] {
        let mut query = base.clone();
        query["nodes"][1]["axis"] = axis;
        assert!(NativeArrayFactory::infer_program(&mapping(query)).is_err());
    }
}

#[test]
fn test_ir1_queries_reject_new_fields_and_rank_without_reinterpreting_reductions() {
    let base = json!({"ir_version":1,"inputs":[{"name":"x","dtype":"float64","shape":[null]}],
        "nodes":[{"id":"x","op":"input","name":"x"},{"id":"y","op":"reduce","function":"sum","input":"x"}],
        "outputs":[{"name":"output","node":"y"}]});
    let result = NativeArrayFactory::infer_program(&mapping(base.clone())).unwrap();
    assert_eq!(result["outputs"][0]["shape"], json!([]));
    for (field, value) in [("axis", Value::Null), ("keepdims", json!(false))] {
        let mut query = base.clone();
        query["nodes"][1][field] = value;
        assert!(NativeArrayFactory::infer_program(&mapping(query)).is_err());
    }
    let mut matrix = base.clone();
    matrix["inputs"][0]["shape"] = json!([null, 2]);
    assert!(NativeArrayFactory::infer_program(&mapping(matrix)).is_err());
    for node in [
        json!({"id":"y","op":"transpose","input":"x"}),
        json!({"id":"y","op":"reshape","input":"x","shape":[1]}),
    ] {
        let mut query = base.clone();
        query["nodes"][1] = node;
        assert!(NativeArrayFactory::infer_program(&mapping(query)).is_err());
    }
}

#[tokio::test]
async fn test_ir2_runtime_checks_deferred_shape_equality_before_allocation() {
    let options = json!({"ir_version":2,"inputs":[{"name":"x","dtype":"float64","shape":[null]}],
        "nodes":[{"id":"x","op":"input","name":"x"},{"id":"y","op":"reshape","input":"x","shape":[2,2]}],
        "outputs":[{"name":"output","node":"y","dtype":"float64","shape":[2,2]}],
        "execution":{"max_allocation_bytes":0}});
    let inputs = BTreeMap::from([(
        "x".into(),
        batch(NativeArray::from_f64(vec![1.0; 3], vec![3], None).unwrap()),
    )]);
    let error = run_program(options, inputs).await.unwrap_err();
    assert!(
        matches!(error, calc_flow::CalcFlowError::InvalidArgument { field, .. } if field == "options.nodes.shape")
    );
}

#[tokio::test]
async fn test_ir2_named_output_aliases_count_flat_elements_without_recharging_inputs() {
    let mut options = json!({"ir_version":2,"inputs":[{"name":"x","dtype":"float64","shape":[2,2]}],
        "nodes":[{"id":"x","op":"input","name":"x"}],
        "outputs":[{"name":"a","node":"x","dtype":"float64","shape":[2,2]},
                   {"name":"b","node":"x","dtype":"float64","shape":[2,2]}],
        "execution":{"max_elements":8,"max_allocation_bytes":0}});
    let input = NativeArray::from_f64(vec![1.0; 4], vec![2, 2], None).unwrap();
    let owner = Arc::clone(input.as_arrow());
    let inputs = BTreeMap::from([("x".into(), batch(input))]);
    let result = run_program(options.clone(), inputs.clone()).await.unwrap();
    for output in result.values() {
        assert!(Arc::ptr_eq(native(output).as_arrow(), &owner));
    }
    options["execution"]["max_elements"] = json!(7);
    let error = run_program(options, inputs).await.unwrap_err();
    assert!(
        matches!(error, calc_flow::CalcFlowError::InvalidArgument { field, .. } if field == "outputs.elements")
    );
}

#[test]
fn test_ir2_matmul_limit_and_exact_rank_broadcast_are_validated_without_rows() {
    let base = json!({"ir_version":2,
        "inputs":[{"name":"x","dtype":"float32","shape":[null,2]}, {"name":"w","dtype":"float64","shape":[2,2]}],
        "nodes":[{"id":"x","op":"input","name":"x"},{"id":"w","op":"input","name":"w"},
                 {"id":"p","op":"matmul","left":"x","right":"w"}],
        "outputs":[{"name":"output","node":"p"}]});
    let inferred = NativeArrayFactory::infer_program(&mapping(base.clone())).unwrap();
    assert_eq!(inferred["outputs"][0]["dtype"], "float64");
    let mut two = base.clone();
    two["nodes"]
        .as_array_mut()
        .unwrap()
        .push(json!({"id":"q","op":"matmul","left":"p","right":"w"}));
    two["outputs"][0]["node"] = json!("q");
    assert!(NativeArrayFactory::infer_program(&mapping(two)).is_err());
    let mut mismatch = base;
    mismatch["inputs"][1]["shape"] = json!([2]);
    mismatch["nodes"][2]["op"] = json!("binary");
    mismatch["nodes"][2]["function"] = json!("add");
    assert!(NativeArrayFactory::infer_program(&mapping(mismatch)).is_err());
}

#[tokio::test]
async fn test_ir2_matmul_rejects_null_rhs_even_for_empty_output() {
    let options = json!({"ir_version":2,
        "inputs":[{"name":"x","dtype":"float64","shape":[0,2]}, {"name":"w","dtype":"float64","shape":[2,1]}],
        "nodes":[{"id":"x","op":"input","name":"x"},{"id":"w","op":"input","name":"w"},
                 {"id":"p","op":"matmul","left":"x","right":"w"}],
        "outputs":[{"name":"output","node":"p","dtype":"float64","shape":[0,1]}]});
    let x = NativeArray::from_f64(Vec::<f64>::new(), vec![0, 2], None).unwrap();
    let w = NativeArray::from_arrow(
        Arc::new(Float64Array::from(vec![Some(1.0), None])),
        vec![2, 1],
    )
    .unwrap();
    assert!(
        run_program(
            options,
            BTreeMap::from([("x".into(), batch(x)), ("w".into(), batch(w))])
        )
        .await
        .is_err()
    );
}
