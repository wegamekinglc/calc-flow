use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};

use calc_flow::{
    Batch, BatchKind, BatchMetadata, BatchOperatorContext, BatchOperatorFactory, CancellationToken,
    ExternalOperatorSpec, JsonMap, Port, RunContext, array::NativeTableArrayFactory,
};
use datafusion::arrow::{
    array::{Array, Float64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use serde_json::json;

#[tokio::test]
async fn test_native_table_bridge_reads_sliced_chunks_and_preserves_source_metadata() {
    let source_schema = Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("a", DataType::Float64, true)
                .with_metadata(HashMap::from([("source".into(), "test".into())])),
            Field::new("b", DataType::Float64, false),
        ],
        HashMap::new(),
    ));
    let record = RecordBatch::try_new(
        source_schema.clone(),
        vec![
            Arc::new(Float64Array::from(vec![
                Some(100.0),
                Some(2.0),
                None,
                Some(6.0),
                Some(100.0),
            ])),
            Arc::new(Float64Array::from(vec![100.0, 4.0, 8.0, 10.0, 100.0])),
        ],
    )
    .unwrap();
    let records = vec![record.slice(1, 2), record.slice(3, 1)];
    let metadata = BatchMetadata::new(
        "table-source",
        7,
        BTreeMap::from([("key".into(), json!("value"))]),
    )
    .unwrap();
    let input = Batch::table(records.clone(), metadata.clone()).unwrap();
    let mut fields = source_schema.fields().to_vec();
    fields.push(Arc::new(Field::new("score", DataType::Float64, true)));
    let output_schema = Arc::new(Schema::new_with_metadata(
        fields,
        source_schema.metadata().clone(),
    ));
    let options: JsonMap = serde_json::from_value(json!({"ir_version":2,
        "inputs":[{"name":"observations","kind":"table","columns":["b","a"]}],
        "nodes":[{"id":"x","op":"input","name":"observations"},
                 {"id":"negative","op":"unary","function":"neg","input":"x"},
                 {"id":"rows","op":"reduce","function":"mean","input":"negative","axis":1}],
        "outputs":[{"name":"result","kind":"table","node":"rows","attach_to":"observations","names":["score"]}]})).unwrap();
    let spec = ExternalOperatorSpec::new("native", "table_array", "1", options).unwrap();
    let mut operator = BatchOperatorFactory::create(
        &NativeTableArrayFactory::default(),
        &spec,
        vec![table_port("observations", &source_schema)],
        vec![table_port("result", &output_schema)],
    )
    .unwrap();
    let run = RunContext::new(BTreeMap::new(), None, CancellationToken::new()).unwrap();
    let result = operator
        .process(
            &BTreeMap::from([("observations".into(), input)]),
            &BatchOperatorContext { run: &run },
        )
        .await
        .unwrap();
    let output = &result["result"];
    assert_eq!(output.metadata(), &metadata);
    let table = output.table_payload().unwrap();
    assert_eq!(table.schema(), &output_schema);
    assert_eq!(table.batches().len(), 2);
    for (record, original) in table.batches().iter().zip(records) {
        assert!(Arc::ptr_eq(record.column(0), original.column(0)));
        assert!(Arc::ptr_eq(record.column(1), original.column(1)));
    }
    let first = table.batches()[0]
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_eq!(first.value(0).to_bits(), (-3.0_f64).to_bits());
    assert!(first.is_null(1));
    let second = table.batches()[1]
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_eq!(second.value(0).to_bits(), (-8.0_f64).to_bits());
}

#[tokio::test]
async fn test_native_table_budget_counts_packing_and_alias_descriptors_before_numeric_work() {
    use datafusion::arrow::buffer::NullBuffer;
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, true)]));
    let values = Float64Array::new(vec![1.0, -0.0].into(), Some(NullBuffer::new_valid(2)));
    let input = Batch::table(
        vec![RecordBatch::try_new(schema.clone(), vec![Arc::new(values)]).unwrap()],
        BatchMetadata::default(),
    )
    .unwrap();
    let mut fields = schema.fields().to_vec();
    fields.push(Arc::new(Field::new("copy", DataType::Float64, true)));
    let fields = fields
        .into_iter()
        .map(|field| field.as_ref().clone())
        .collect::<Vec<_>>();
    let options = json!({"ir_version":2,"inputs":[{"name":"table","kind":"table","columns":["x"]}],
        "nodes":[{"id":"x","op":"input","name":"table"},{"id":"r","op":"reshape","input":"x","shape":[2]}],
        "outputs":[{"name":"output","kind":"table","node":"r","attach_to":"table","names":["copy"]}],
        "execution":{"max_allocation_bytes":39}});
    let run = RunContext::new(BTreeMap::new(), None, CancellationToken::new()).unwrap();
    let process = |limit| {
        let mut options = options.clone();
        options["execution"]["max_allocation_bytes"] = json!(limit);
        let spec = ExternalOperatorSpec::new(
            "native",
            "table_array",
            "1",
            serde_json::from_value(options).unwrap(),
        )
        .unwrap();
        BatchOperatorFactory::create(
            &NativeTableArrayFactory::default(),
            &spec,
            vec![
                Port::new(
                    "table",
                    BatchKind::Table,
                    true,
                    Some(
                        schema
                            .fields()
                            .iter()
                            .map(|field| field.as_ref().clone())
                            .collect(),
                    ),
                )
                .unwrap(),
            ],
            vec![Port::new("output", BatchKind::Table, true, Some(fields.clone())).unwrap()],
        )
        .unwrap()
    };
    let inputs = BTreeMap::from([("table".into(), input)]);
    let error = process(39)
        .process(&inputs, &BatchOperatorContext { run: &run })
        .await
        .unwrap_err();
    assert!(
        matches!(error,calc_flow::CalcFlowError::InvalidArgument {field,..} if field=="outputs.bytes")
    );
    let result = process(40)
        .process(&inputs, &BatchOperatorContext { run: &run })
        .await
        .unwrap();
    let copy = result["output"].table_payload().unwrap().batches()[0]
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_eq!(copy.value(1).to_bits(), (-0.0_f64).to_bits());
    assert!(copy.nulls().is_none());
}

#[test]
fn test_native_table_bridge_rejects_duplicate_or_empty_column_selection() {
    for columns in [json!([]), json!(["a", "a"])] {
        let mut options = base_table_options();
        options["inputs"][0]["columns"] = columns;
        assert!(create_table(options).is_err());
    }
}

#[test]
fn test_native_table_bridge_rejects_row_count_coincidence_without_row_lineage() {
    for node in [
        json!({"id":"y","op":"reduce","function":"mean","input":"x","axis":0}),
        json!({"id":"y","op":"transpose","input":"x"}),
        json!({"id":"y","op":"reduce","function":"mean","input":"x"}),
    ] {
        let mut options = base_table_options();
        options["nodes"][1] = node;
        assert!(create_table(options).is_err());
    }
}

#[test]
fn test_native_table_bridge_rejects_conflicting_or_mismatched_attached_names() {
    for names in [json!(["a"]), json!(["score", "score"]), json!([])] {
        let mut options = base_table_options();
        options["outputs"][0]["names"] = names;
        assert!(create_table(options).is_err());
    }
}

fn base_table_options() -> serde_json::Value {
    json!({"ir_version":2,"inputs":[{"name":"table","kind":"table","columns":["a","b"]}],
        "nodes":[{"id":"x","op":"input","name":"table"},{"id":"y","op":"reduce","function":"mean","input":"x","axis":1}],
        "outputs":[{"name":"output","kind":"table","node":"y","attach_to":"table","names":["score"]}]})
}
fn create_table(
    options: serde_json::Value,
) -> calc_flow::Result<Box<dyn calc_flow::BatchOperator>> {
    let fields = vec![
        Field::new("a", DataType::Float64, true),
        Field::new("b", DataType::Float64, true),
    ];
    let mut output = fields.clone();
    output.extend(
        options["outputs"][0]["names"]
            .as_array()
            .unwrap()
            .iter()
            .map(|name| Field::new(name.as_str().unwrap(), DataType::Float64, true)),
    );
    let spec = ExternalOperatorSpec::new(
        "native",
        "table_array",
        "1",
        serde_json::from_value(options).unwrap(),
    )
    .unwrap();
    BatchOperatorFactory::create(
        &NativeTableArrayFactory::default(),
        &spec,
        vec![Port::new("table", BatchKind::Table, true, Some(fields)).unwrap()],
        vec![Port::new("output", BatchKind::Table, true, Some(output)).unwrap()],
    )
}

#[tokio::test]
async fn test_native_table_array_output_is_packed_in_selected_order_and_keeps_metadata() {
    use calc_flow::array::NativeArray;
    let fields = vec![
        Field::new("a", DataType::Float64, true),
        Field::new("b", DataType::Float64, false),
    ];
    let schema = Arc::new(Schema::new(fields.clone()));
    let record = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![Some(1.0), None])),
            Arc::new(Float64Array::from(vec![10.0, 20.0])),
        ],
    )
    .unwrap();
    let options = json!({"ir_version":2,"inputs":[{"name":"table","kind":"table","columns":["b","a"]}],
        "nodes":[{"id":"x","op":"input","name":"table"},{"id":"y","op":"unary","function":"neg","input":"x"}],
        "outputs":[{"name":"output","kind":"array","node":"y","dtype":"float64","shape":[null,2]}]});
    let spec = ExternalOperatorSpec::new(
        "native",
        "table_array",
        "1",
        serde_json::from_value(options).unwrap(),
    )
    .unwrap();
    let mut operator = BatchOperatorFactory::create(
        &NativeTableArrayFactory::default(),
        &spec,
        vec![Port::new("table", BatchKind::Table, true, Some(fields)).unwrap()],
        vec![Port::new("output", BatchKind::Array, true, None).unwrap()],
    )
    .unwrap();
    let run = RunContext::new(BTreeMap::new(), None, CancellationToken::new()).unwrap();
    let metadata = BatchMetadata::new("matrix-source", 11, BTreeMap::new()).unwrap();
    for record in [record.clone(), record.slice(0, 0)] {
        let rows = record.num_rows();
        let input = Batch::table(vec![record], metadata.clone()).unwrap();
        let result = operator
            .process(
                &BTreeMap::from([("table".into(), input)]),
                &BatchOperatorContext { run: &run },
            )
            .await
            .unwrap();
        let output = &result["output"];
        assert_eq!(output.metadata(), &metadata);
        let array = output
            .external_payload()
            .unwrap()
            .as_any()
            .downcast_ref::<NativeArray>()
            .unwrap();
        assert_eq!(array.shape(), [rows, 2]);
        assert_eq!(array.len(), rows);
        let values = array
            .as_arrow()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        let expected = if rows == 0 {
            vec![]
        } else {
            vec![Some(-10.0), Some(-1.0), Some(-20.0), None]
        };
        assert_eq!(values.iter().collect::<Vec<_>>(), expected);
    }
}

#[tokio::test]
async fn test_native_table_float32_packing_preserves_bits_and_uses_narrow_storage_budget() {
    use calc_flow::array::NativeArray;
    use datafusion::arrow::array::Float32Array;
    let fields = vec![
        Field::new("a", DataType::Float32, true),
        Field::new("b", DataType::Float32, false),
    ];
    let record = RecordBatch::try_new(
        Arc::new(Schema::new(fields.clone())),
        vec![
            Arc::new(Float32Array::from(vec![
                Some(f32::from_bits(0x7fc0_1234)),
                None,
            ])),
            Arc::new(Float32Array::from(vec![2.0, -0.0])),
        ],
    )
    .unwrap();
    let options = json!({"ir_version":2,"inputs":[{"name":"table","kind":"table","columns":["b","a"]}],
        "nodes":[{"id":"x","op":"input","name":"table"}],
        "outputs":[{"name":"output","kind":"array","node":"x","dtype":"float32","shape":[null,2]}],
        "execution":{"max_allocation_bytes":33}});
    let spec = ExternalOperatorSpec::new(
        "native",
        "table_array",
        "1",
        serde_json::from_value(options).unwrap(),
    )
    .unwrap();
    let mut operator = BatchOperatorFactory::create(
        &NativeTableArrayFactory::default(),
        &spec,
        vec![Port::new("table", BatchKind::Table, true, Some(fields)).unwrap()],
        vec![Port::new("output", BatchKind::Array, true, None).unwrap()],
    )
    .unwrap();
    let run = RunContext::new(BTreeMap::new(), None, CancellationToken::new()).unwrap();
    let result = operator
        .process(
            &BTreeMap::from([(
                "table".into(),
                Batch::table(vec![record], BatchMetadata::default()).unwrap(),
            )]),
            &BatchOperatorContext { run: &run },
        )
        .await
        .unwrap();
    let array = result["output"]
        .external_payload()
        .unwrap()
        .as_any()
        .downcast_ref::<NativeArray>()
        .unwrap();
    assert_eq!(array.shape(), [2, 2]);
    let values = array
        .as_arrow()
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap();
    assert_eq!(values.value(0).to_bits(), 2.0_f32.to_bits());
    assert_eq!(values.value(1).to_bits(), 0x7fc0_1234);
    assert_eq!(values.value(2).to_bits(), (-0.0_f32).to_bits());
    assert!(values.is_null(3));
    assert_eq!(array.retained_buffer_bytes(), 33);
}

#[tokio::test]
async fn test_native_table_boolean_packing_uses_independent_bit_offsets_and_shared_budget() {
    use calc_flow::array::NativeArray;
    use datafusion::arrow::{
        array::BooleanArray,
        buffer::{BooleanBuffer, NullBuffer},
    };
    let left = BooleanArray::new(
        BooleanBuffer::from(vec![
            false, true, true, false, true, false, true, true, false, true, false,
        ])
        .slice(1, 9),
        Some(NullBuffer::new(
            BooleanBuffer::from(vec![
                true, true, true, true, false, true, true, true, false, true, true, true,
            ])
            .slice(3, 9),
        )),
    );
    let right = BooleanArray::from(vec![
        false, true, true, false, false, true, false, true, false,
    ]);
    let fields = vec![
        Field::new("a", DataType::Boolean, true),
        Field::new("b", DataType::Boolean, false),
    ];
    let record = RecordBatch::try_new(
        Arc::new(Schema::new(fields.clone())),
        vec![Arc::new(left), Arc::new(right)],
    )
    .unwrap();
    let input = Batch::table(
        vec![record.slice(0, 4), record.slice(4, 5)],
        BatchMetadata::default(),
    )
    .unwrap();
    let options = json!({"ir_version":2,"inputs":[{"name":"table","kind":"table","columns":["b","a"]}],
        "nodes":[{"id":"x","op":"input","name":"table"},{"id":"y","op":"unary","function":"logical_not","input":"x"}],
        "outputs":[{"name":"output","kind":"array","node":"y","dtype":"bool","shape":[null,2]}],
        "execution":{"max_allocation_bytes":44}});
    let run = RunContext::new(BTreeMap::new(), None, CancellationToken::new()).unwrap();
    let inputs = BTreeMap::from([("table".into(), input)]);
    let result = array_operator(options.clone(), fields.clone())
        .unwrap()
        .process(&inputs, &BatchOperatorContext { run: &run })
        .await
        .unwrap();
    let array = result["output"]
        .external_payload()
        .unwrap()
        .as_any()
        .downcast_ref::<NativeArray>()
        .unwrap();
    assert_eq!(array.shape(), [9, 2]);
    let values = array
        .as_arrow()
        .as_any()
        .downcast_ref::<BooleanArray>()
        .unwrap();
    assert_eq!(
        values.iter().collect::<Vec<_>>(),
        vec![
            Some(true),
            Some(false),
            Some(false),
            None,
            Some(false),
            Some(true),
            Some(true),
            Some(false),
            Some(true),
            Some(true),
            Some(false),
            None,
            Some(true),
            Some(false),
            Some(false),
            Some(true),
            Some(true),
            Some(false)
        ]
    );
    assert_eq!(array.retained_buffer_bytes(), 22);
    let mut too_small = options;
    too_small["execution"]["max_allocation_bytes"] = json!(43);
    let error = array_operator(too_small, fields)
        .unwrap()
        .process(&inputs, &BatchOperatorContext { run: &run })
        .await
        .unwrap_err();
    assert!(
        matches!(error,calc_flow::CalcFlowError::InvalidArgument {field,..} if field=="outputs.bytes")
    );
}

fn array_operator(
    options: serde_json::Value,
    fields: Vec<Field>,
) -> calc_flow::Result<Box<dyn calc_flow::BatchOperator>> {
    let spec = ExternalOperatorSpec::new(
        "native",
        "table_array",
        "1",
        serde_json::from_value(options).unwrap(),
    )
    .unwrap();
    BatchOperatorFactory::create(
        &NativeTableArrayFactory::default(),
        &spec,
        vec![Port::new("table", BatchKind::Table, true, Some(fields)).unwrap()],
        vec![Port::new("output", BatchKind::Array, true, None).unwrap()],
    )
}

#[tokio::test]
async fn test_native_table_matrix_attachment_preserves_columns_and_charges_column_buffers() {
    let fields = vec![
        Field::new("a", DataType::Float64, true),
        Field::new("b", DataType::Float64, false),
    ];
    let record = RecordBatch::try_new(
        Arc::new(Schema::new(fields.clone())),
        vec![
            Arc::new(Float64Array::from(vec![Some(1.0), None])),
            Arc::new(Float64Array::from(vec![10.0, 20.0])),
        ],
    )
    .unwrap();
    let input = Batch::table(
        vec![record.slice(0, 1), record.slice(1, 1)],
        BatchMetadata::default(),
    )
    .unwrap();
    let mut output_fields = fields.clone();
    output_fields.extend([
        Field::new("copy_b", DataType::Float64, true),
        Field::new("copy_a", DataType::Float64, true),
    ]);
    let options = json!({"ir_version":2,"inputs":[{"name":"table","kind":"table","columns":["b","a"]}],
        "nodes":[{"id":"x","op":"input","name":"table"}],
        "outputs":[{"name":"output","kind":"table","node":"x","attach_to":"table","names":["copy_b","copy_a"]}],
        "execution":{"max_allocation_bytes":83}});
    let create = |options| {
        let spec = ExternalOperatorSpec::new(
            "native",
            "table_array",
            "1",
            serde_json::from_value(options).unwrap(),
        )
        .unwrap();
        BatchOperatorFactory::create(
            &NativeTableArrayFactory::default(),
            &spec,
            vec![Port::new("table", BatchKind::Table, true, Some(fields.clone())).unwrap()],
            vec![
                Port::new(
                    "output",
                    BatchKind::Table,
                    true,
                    Some(output_fields.clone()),
                )
                .unwrap(),
            ],
        )
        .unwrap()
    };
    let run = RunContext::new(BTreeMap::new(), None, CancellationToken::new()).unwrap();
    let inputs = BTreeMap::from([("table".into(), input)]);
    let result = create(options.clone())
        .process(&inputs, &BatchOperatorContext { run: &run })
        .await
        .unwrap();
    let records = result["output"].table_payload().unwrap().batches();
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].schema().fields().len(), 4);
    assert_eq!(
        column64(&records[0], 2).value(0).to_bits(),
        10.0_f64.to_bits()
    );
    assert_eq!(
        column64(&records[0], 3).value(0).to_bits(),
        1.0_f64.to_bits()
    );
    assert_eq!(
        column64(&records[1], 2).value(0).to_bits(),
        20.0_f64.to_bits()
    );
    assert!(records[1].column(3).is_null(0));
    assert!(records[0].column(2).nulls().is_none());
    for (record, source) in records
        .iter()
        .zip(inputs["table"].table_payload().unwrap().batches())
    {
        assert!(Arc::ptr_eq(record.column(0), source.column(0)));
        assert!(Arc::ptr_eq(record.column(1), source.column(1)));
    }
    let mut too_small = options;
    too_small["execution"]["max_allocation_bytes"] = json!(82);
    let error = create(too_small)
        .process(&inputs, &BatchOperatorContext { run: &run })
        .await
        .unwrap_err();
    assert!(
        matches!(error,calc_flow::CalcFlowError::InvalidArgument {field,..} if field=="outputs.bytes")
    );
}

#[tokio::test]
async fn test_native_table_float32_matrix_attachment_keeps_dtype_and_ieee_bits() {
    use datafusion::arrow::array::Float32Array;
    let fields = vec![
        Field::new("a", DataType::Float32, true),
        Field::new("b", DataType::Float32, false),
    ];
    let record = RecordBatch::try_new(
        Arc::new(Schema::new(fields.clone())),
        vec![
            Arc::new(Float32Array::from(vec![Some(1.0), None])),
            Arc::new(Float32Array::from(vec![-0.0, f32::from_bits(0x7fc0_1234)])),
        ],
    )
    .unwrap();
    let mut operator = matrix_copy_operator(fields, 51).unwrap();
    let run = RunContext::new(BTreeMap::new(), None, CancellationToken::new()).unwrap();
    let result = operator
        .process(
            &BTreeMap::from([(
                "table".into(),
                Batch::table(vec![record], BatchMetadata::default()).unwrap(),
            )]),
            &BatchOperatorContext { run: &run },
        )
        .await
        .unwrap();
    let table = result["output"].table_payload().unwrap();
    let copy_a = table.batches()[0]
        .column(2)
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap();
    let copy_b = table.batches()[0]
        .column(3)
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap();
    assert_eq!(copy_a.value(0).to_bits(), 1.0_f32.to_bits());
    assert!(copy_a.is_null(1));
    assert_eq!(copy_b.value(0).to_bits(), (-0.0_f32).to_bits());
    assert_eq!(copy_b.value(1).to_bits(), 0x7fc0_1234);
}

#[tokio::test]
async fn test_native_table_boolean_attachment_charges_each_column_bitmap() {
    use datafusion::arrow::array::BooleanArray;
    let fields = vec![
        Field::new("a", DataType::Boolean, true),
        Field::new("b", DataType::Boolean, false),
    ];
    let record = RecordBatch::try_new(
        Arc::new(Schema::new(fields.clone())),
        vec![
            Arc::new(BooleanArray::from(vec![
                Some(true),
                None,
                Some(false),
                Some(true),
                Some(false),
                Some(true),
                Some(false),
                Some(true),
                Some(true),
            ])),
            Arc::new(BooleanArray::from(vec![
                false, true, false, true, true, false, true, false, false,
            ])),
        ],
    )
    .unwrap();
    let inputs = BTreeMap::from([(
        "table".into(),
        Batch::table(vec![record], BatchMetadata::default()).unwrap(),
    )]);
    let run = RunContext::new(BTreeMap::new(), None, CancellationToken::new()).unwrap();
    let result = matrix_copy_operator(fields.clone(), 30)
        .unwrap()
        .process(&inputs, &BatchOperatorContext { run: &run })
        .await
        .unwrap();
    let table = result["output"].table_payload().unwrap();
    let copy_a = table.batches()[0]
        .column(2)
        .as_any()
        .downcast_ref::<BooleanArray>()
        .unwrap();
    let copy_b = table.batches()[0]
        .column(3)
        .as_any()
        .downcast_ref::<BooleanArray>()
        .unwrap();
    assert!(copy_a.value(0));
    assert!(copy_a.is_null(1));
    assert!(copy_a.value(8));
    assert!(!copy_b.value(0));
    assert!(!copy_b.value(8));
    assert!(copy_b.nulls().is_none());
    let error = matrix_copy_operator(fields, 29)
        .unwrap()
        .process(&inputs, &BatchOperatorContext { run: &run })
        .await
        .unwrap_err();
    assert!(
        matches!(error,calc_flow::CalcFlowError::InvalidArgument {field,..} if field=="outputs.bytes")
    );
}

fn matrix_copy_operator(
    fields: Vec<Field>,
    bytes: usize,
) -> calc_flow::Result<Box<dyn calc_flow::BatchOperator>> {
    let mut output_fields = fields.clone();
    output_fields.extend([
        Field::new("copy_a", fields[0].data_type().clone(), true),
        Field::new("copy_b", fields[0].data_type().clone(), true),
    ]);
    let options = json!({"ir_version":2,"inputs":[{"name":"table","kind":"table","columns":["a","b"]}],
        "nodes":[{"id":"x","op":"input","name":"table"}],
        "outputs":[{"name":"output","kind":"table","node":"x","attach_to":"table","names":["copy_a","copy_b"]}],
        "execution":{"max_allocation_bytes":bytes}});
    let spec = ExternalOperatorSpec::new(
        "native",
        "table_array",
        "1",
        serde_json::from_value(options).unwrap(),
    )
    .unwrap();
    BatchOperatorFactory::create(
        &NativeTableArrayFactory::default(),
        &spec,
        vec![Port::new("table", BatchKind::Table, true, Some(fields)).unwrap()],
        vec![Port::new("output", BatchKind::Table, true, Some(output_fields)).unwrap()],
    )
}

#[derive(Default)]
struct CollectBatches(Vec<(String, Batch)>);
#[async_trait::async_trait]
impl calc_flow::StreamCollector for CollectBatches {
    async fn emit(&mut self, port: &str, batch: Batch) -> calc_flow::Result<()> {
        self.0.push((port.to_owned(), batch));
        Ok(())
    }
}

#[tokio::test]
async fn test_native_table_stream_is_stateless_and_preserves_vector_transpose_rows() {
    use calc_flow::{
        Epoch, StreamJobContext, StreamOperatorContext, StreamOperatorFactory,
        StreamOperatorLifecycle,
    };
    let mut options = base_table_options();
    options["nodes"]
        .as_array_mut()
        .unwrap()
        .push(json!({"id":"t","op":"transpose","input":"y"}));
    options["outputs"][0]["node"] = json!("t");
    let fields = vec![
        Field::new("a", DataType::Float64, true),
        Field::new("b", DataType::Float64, true),
    ];
    let mut output = fields.clone();
    output.push(Field::new("score", DataType::Float64, true));
    let spec = ExternalOperatorSpec::new(
        "native",
        "table_array",
        "1",
        serde_json::from_value(options).unwrap(),
    )
    .unwrap();
    let mut operator = StreamOperatorFactory::create(
        &NativeTableArrayFactory::default(),
        &spec,
        vec![Port::new("table", BatchKind::Table, true, Some(fields.clone())).unwrap()],
        vec![Port::new("output", BatchKind::Table, true, Some(output)).unwrap()],
    )
    .unwrap();
    assert_eq!(
        operator.lifecycle(),
        StreamOperatorLifecycle::Stateless {
            microbatch_invariant: true,
            deterministic: true,
            replay_safe: true
        }
    );
    let record = RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        vec![
            Arc::new(Float64Array::from(vec![1.0, 5.0, 9.0])),
            Arc::new(Float64Array::from(vec![3.0, 7.0, 11.0])),
        ],
    )
    .unwrap();
    let job = StreamJobContext::new(
        17,
        "native-table",
        BTreeMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "calc", None);
    let mut collector = CollectBatches::default();
    for record in [record.slice(0, 2), record.slice(2, 1)] {
        operator
            .process_data(
                "table",
                Batch::table(vec![record], BatchMetadata::default()).unwrap(),
                &context,
                &mut collector,
            )
            .await
            .unwrap();
    }
    operator.on_end(&context, &mut collector).await.unwrap();
    assert_eq!(collector.0.len(), 2);
    let values = collector
        .0
        .iter()
        .flat_map(|(port, batch)| {
            assert_eq!(port, "output");
            batch.table_payload().unwrap().batches()[0]
                .column(2)
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .values()
                .iter()
                .copied()
        })
        .collect::<Vec<_>>();
    assert_eq!(values, [2.0, 6.0, 10.0]);
    let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
    assert!(snapshot.inline_metadata.is_empty());
    assert!(snapshot.segments.is_empty());
    operator.restore(&snapshot).unwrap();
}

#[test]
fn test_native_table_stream_rejects_concrete_reshape_that_locks_batch_length() {
    use calc_flow::StreamOperatorFactory;
    let fields = vec![
        Field::new("a", DataType::Float64, true),
        Field::new("b", DataType::Float64, true),
    ];
    let options = json!({"ir_version":2,"inputs":[{"name":"table","kind":"table","columns":["a","b"]}],
        "nodes":[{"id":"x","op":"input","name":"table"},{"id":"r","op":"reshape","input":"x","shape":[2,2]}],
        "outputs":[{"name":"output","kind":"array","node":"r","dtype":"float64","shape":[2,2]}]});
    assert!(array_operator(options.clone(), fields.clone()).is_ok());
    let spec = ExternalOperatorSpec::new(
        "native",
        "table_array",
        "1",
        serde_json::from_value(options).unwrap(),
    )
    .unwrap();
    let result = StreamOperatorFactory::create(
        &NativeTableArrayFactory::default(),
        &spec,
        vec![Port::new("table", BatchKind::Table, true, Some(fields)).unwrap()],
        vec![Port::new("output", BatchKind::Array, true, None).unwrap()],
    );
    assert!(
        matches!(result,Err(calc_flow::CalcFlowError::InvalidArgument {field,..}) if field=="options.nodes.shape")
    );
}

fn table_port(name: &str, schema: &Schema) -> Port {
    Port::new(
        name,
        BatchKind::Table,
        true,
        Some(
            schema
                .fields()
                .iter()
                .map(|field| field.as_ref().clone())
                .collect(),
        ),
    )
    .unwrap()
}
fn column64(record: &RecordBatch, index: usize) -> &Float64Array {
    record
        .column(index)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
}

#[test]
fn test_native_table_static_rhs_declares_exact_ports_and_promoted_output_schema() {
    use calc_flow::StreamOperatorFactory;
    let fields = vec![
        Field::new("a", DataType::Float32, false),
        Field::new("b", DataType::Float32, false),
    ];
    let mut output_fields = fields.clone();
    output_fields.push(Field::new("score", DataType::Float64, true));
    let spec = ExternalOperatorSpec::new(
        "native",
        "table_array",
        "1",
        serde_json::from_value(static_options()).unwrap(),
    )
    .unwrap();
    let inputs = vec![
        Port::new("observations", BatchKind::Table, true, Some(fields)).unwrap(),
        Port::new("coefficients", BatchKind::Array, true, None).unwrap(),
    ];
    let outputs = vec![Port::new("result", BatchKind::Table, true, Some(output_fields)).unwrap()];
    assert!(
        BatchOperatorFactory::create(
            &NativeTableArrayFactory::default(),
            &spec,
            inputs.clone(),
            outputs.clone()
        )
        .is_ok()
    );
    assert!(
        StreamOperatorFactory::create(&NativeTableArrayFactory::default(), &spec, inputs, outputs)
            .is_ok()
    );
}
fn static_options() -> serde_json::Value {
    json!({"ir_version":2,
        "inputs":[{"name":"observations","kind":"table","columns":["a","b"]},
                  {"name":"coefficients","kind":"array","binding":"static","dtype":"float64","shape":[2,1]}],
        "nodes":[{"id":"x","op":"input","name":"observations"},{"id":"beta","op":"input","name":"coefficients"},
                 {"id":"y","op":"matmul","left":"x","right":"beta"}],
        "outputs":[{"name":"result","kind":"table","node":"y","attach_to":"observations","names":["score"]}]})
}

#[test]
fn test_native_table_rejects_transforming_static_rhs_inside_the_bridge() {
    let mut options = static_options();
    options["nodes"]
        .as_array_mut()
        .unwrap()
        .push(json!({"id":"changed","op":"unary","function":"abs","input":"beta"}));
    options["nodes"][2]["right"] = json!("changed");
    assert!(static_factory(options).is_err());
}
#[test]
fn test_native_table_rejects_an_unused_static_declaration() {
    let mut options = static_options();
    options["nodes"] = json!([{"id":"x","op":"input","name":"observations"},{"id":"y","op":"reduce","function":"mean","input":"x","axis":1}]);
    assert!(static_factory(options).is_err());
}
fn static_factory(
    options: serde_json::Value,
) -> calc_flow::Result<Box<dyn calc_flow::BatchOperator>> {
    let fields = vec![
        Field::new("a", DataType::Float32, false),
        Field::new("b", DataType::Float32, false),
    ];
    let mut output = fields.clone();
    output.push(Field::new("score", DataType::Float64, true));
    let spec = ExternalOperatorSpec::new(
        "native",
        "table_array",
        "1",
        serde_json::from_value(options).unwrap(),
    )
    .unwrap();
    BatchOperatorFactory::create(
        &NativeTableArrayFactory::default(),
        &spec,
        vec![
            Port::new("observations", BatchKind::Table, true, Some(fields)).unwrap(),
            Port::new("coefficients", BatchKind::Array, true, None).unwrap(),
        ],
        vec![Port::new("result", BatchKind::Table, true, Some(output)).unwrap()],
    )
}

#[tokio::test]
async fn test_native_table_static_project_prepares_real_native_owner_before_execution() {
    use calc_flow::{
        ExecutionOptions, ProjectSpec, ProviderRegistry, UdfRegistrySnapshot, array::NativeArray,
        compile_project,
    };
    use datafusion::arrow::array::Float32Array;
    let fields = json!([{"name":"a","data_type":"float32","nullable":false},{"name":"b","data_type":"float32","nullable":false}]);
    let mut output = fields.as_array().unwrap().clone();
    output.push(json!({"name":"score","data_type":"float64","nullable":true}));
    let project: ProjectSpec = serde_json::from_value(json!({
        "format_version":3,"id":"native-static","name":"native-static","runtime":{"mode":"batch","options":{}},
        "static_inputs":[{"name":"coefficients","kind":"array","mutability":"static","backend":"native","dtype":"float64","shape":[2,1]}],
        "data_sources":[{"id":"rows","input":"observations","format":"arrow_ipc","data":null}],
        "graph":{"name":"native-static","nodes":[{"id":"calc",
            "operator":{"kind":"external","provider":"native","name":"table_array","version":"1","options":static_options()},
            "input_ports":[{"name":"observations","kind":"table","schema":fields},{"name":"coefficients","kind":"array"}],
            "output_ports":[{"name":"result","kind":"table","schema":output}]}]}})).unwrap();
    let registry = ProviderRegistry::default();
    registry
        .register_batch(
            "native",
            "table_array",
            "1",
            Arc::new(NativeTableArrayFactory::default()),
        )
        .unwrap();
    let plan = compile_project(&project, &registry, &UdfRegistrySnapshot::default()).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Float32, false),
        Field::new("b", DataType::Float32, false),
    ]));
    let rows = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float32Array::from(vec![1.0, 3.0])),
            Arc::new(Float32Array::from(vec![2.0, 4.0])),
        ],
    )
    .unwrap();
    let weights = NativeArray::from_f64(vec![2.0, 3.0], vec![2, 1], None).unwrap();
    let inputs = BTreeMap::from([
        (
            "observations".into(),
            Batch::table(vec![rows], BatchMetadata::default()).unwrap(),
        ),
        (
            "coefficients".into(),
            Batch::external(Arc::new(weights), BatchMetadata::default()).unwrap(),
        ),
    ]);
    let result = plan
        .execute(inputs, ExecutionOptions::default())
        .await
        .unwrap();
    let table = result.outputs["result"].table_payload().unwrap();
    let scores = column64(&table.batches()[0], 2);
    assert_eq!(
        scores
            .values()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        [8.0_f64.to_bits(), 18.0_f64.to_bits()]
    );
}
