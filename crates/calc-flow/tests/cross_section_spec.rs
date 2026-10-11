use calc_flow::{
    Batch, BatchKind, BatchMetadata, BatchOperator, BatchOperatorContext,
    CROSS_SECTION_CONFIGURATION_VERSION, CROSS_SECTION_STATE_LAYOUT_VERSION, CalcFlowError,
    CrossSectionGroupingSpec, CrossSectionOperator, CrossSectionOutputSpec, CrossSectionSpec,
    CrossSectionValuePolicy, JsonMap, LatePolicySpec, NullPlacement, OperatorMetadata, Port,
    RankTieMethod, SortDirection,
};
use datafusion::arrow::{
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use std::{collections::BTreeMap, sync::Arc};

fn input_schema() -> Schema {
    Schema::new(vec![
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC"))),
            false,
        ),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("industry", DataType::Utf8, true),
        Field::new("sequence", DataType::UInt64, false),
        Field::new("value", DataType::Float64, true),
    ])
}

fn rank_output(input: &str, output: &str) -> CrossSectionOutputSpec {
    CrossSectionOutputSpec::Rank {
        primitive_version: 1,
        input: input.into(),
        output: output.into(),
        direction: SortDirection::Ascending,
        tie_method: RankTieMethod::Average,
        null_placement: NullPlacement::Exclude,
        min_samples: 1,
    }
}

fn valid_spec() -> CrossSectionSpec {
    CrossSectionSpec {
        configuration_version: CROSS_SECTION_CONFIGURATION_VERSION,
        state_layout_version: CROSS_SECTION_STATE_LAYOUT_VERSION,
        event_time: "ts".into(),
        entity_by: vec!["symbol".into()],
        partition_by: vec!["industry".into()],
        sequence_by: vec!["sequence".into()],
        grouping: CrossSectionGroupingSpec::ExactTime,
        outputs: vec![rank_output("value", "rank")],
        allowed_lateness_micros: 0,
        late_policy: LatePolicySpec::Drop { metrics_version: 1 },
        value_policy: CrossSectionValuePolicy::NanExcludePreserveV1,
    }
}

#[test]
fn test_side_output_validates_versions() {
    for field in ["metrics_version", "schema_version"] {
        for version in [0, 2] {
            let mut policy = serde_json::json!({"kind": "side_output", "metrics_version": 1, "schema_version": 1});
            policy[field] = serde_json::json!(version);
            let mut spec = valid_spec();
            spec.late_policy = serde_json::from_value(policy).unwrap();
            let error = spec.validate(&input_schema()).unwrap_err();
            assert!(
                matches!(error, CalcFlowError::InvalidArgument { field: ref name, .. } if name.ends_with(field)),
                "{error}"
            );
            assert!(CrossSectionOperator::new("features", Arc::new(input_schema()), spec).is_err());
        }
        let mut policy =
            serde_json::json!({"kind": "side_output", "metrics_version": 1, "schema_version": 1});
        policy.as_object_mut().unwrap().remove(field);
        assert!(serde_json::from_value::<LatePolicySpec>(policy).is_err());
    }
}

#[test]
fn test_side_output_validates_reserved_prefix() {
    for name in ["_cf_late_reason", "_cf_late_future"] {
        let mut fields = input_schema().fields().to_vec();
        fields.push(Arc::new(Field::new(name, DataType::Utf8, true)));
        let input = Arc::new(Schema::new(fields));
        for policy in [
            LatePolicySpec::Error {
                scope: calc_flow::LateErrorScope::Envelope,
            },
            LatePolicySpec::Drop { metrics_version: 1 },
        ] {
            let mut spec = valid_spec();
            spec.late_policy = policy;
            assert!(spec.validate(&input).is_ok());
            assert_eq!(
                CrossSectionOperator::new("features", input.clone(), spec)
                    .unwrap()
                    .output_ports()
                    .len(),
                1
            );
        }
        let mut spec = valid_spec();
        spec.late_policy = LatePolicySpec::SideOutput {
            metrics_version: 1,
            schema_version: 1,
        };
        let error = spec.validate(&input).unwrap_err().to_string();
        assert!(
            error.contains(name) && error.contains("reserved"),
            "{error}"
        );
        assert!(CrossSectionOperator::new("features", input, spec).is_err());
    }
}

#[test]
fn test_side_output_derives_exact_ports_and_preserves_arrow_metadata() {
    let input = input_schema();
    let fields = input
        .fields()
        .iter()
        .map(|field| {
            field.as_ref().clone().with_metadata(
                [("field_key".into(), "field_value".into())]
                    .into_iter()
                    .collect(),
            )
        })
        .collect::<Vec<_>>();
    let input = Arc::new(Schema::new_with_metadata(
        fields,
        [("schema_key".into(), "schema_value".into())]
            .into_iter()
            .collect(),
    ));
    let mut spec = valid_spec();
    let normal = spec.validate(&input).unwrap();
    spec.late_policy = serde_json::from_value(serde_json::json!({
        "kind": "side_output", "metrics_version": 1, "schema_version": 1
    }))
    .unwrap();
    assert_eq!(spec.validate(&input).unwrap(), normal);
    let operator = CrossSectionOperator::new("features", input.clone(), spec).unwrap();
    let ports = operator.output_ports();
    assert_eq!(
        ports.iter().map(Port::name).collect::<Vec<_>>(),
        ["output", "late"]
    );
    assert!(
        ports
            .iter()
            .all(|port| port.required() && port.kind() == BatchKind::Table)
    );
    assert_eq!(ports[0].schema(), Some(&normal));
    let late = ports[1].schema().unwrap();
    assert_eq!(late.metadata(), input.metadata());
    assert_eq!(
        &late.fields()[..input.fields().len()],
        input.fields().as_ref()
    );
    let diagnostics = [
        ("_cf_late_node", DataType::Utf8),
        ("_cf_late_input_port", DataType::Utf8),
        ("_cf_late_event_time_micros", DataType::Int64),
        ("_cf_late_closing_time_micros", DataType::Int64),
        ("_cf_late_watermark_micros", DataType::Int64),
        ("_cf_late_reason", DataType::Utf8),
        ("_cf_late_source", DataType::Utf8),
        ("_cf_late_sequence", DataType::UInt64),
        ("_cf_late_row_index", DataType::UInt64),
    ];
    assert_eq!(
        late.fields().len(),
        input.fields().len() + diagnostics.len()
    );
    for (field, (name, data_type)) in late.fields()[input.fields().len()..]
        .iter()
        .zip(diagnostics)
    {
        assert_eq!(field.as_ref(), &Field::new(name, data_type, false));
    }
}

#[tokio::test]
async fn test_side_output_direct_batch_execution_is_rejected() {
    let mut spec = valid_spec();
    spec.late_policy = LatePolicySpec::SideOutput {
        metrics_version: 1,
        schema_version: 1,
    };
    let input = Arc::new(input_schema());
    let mut operator = CrossSectionOperator::new("features", input.clone(), spec).unwrap();
    let batch = Batch::table(
        vec![RecordBatch::new_empty(input)],
        BatchMetadata::default(),
    )
    .unwrap();
    let run = calc_flow::RunContext::new(JsonMap::new(), None, calc_flow::CancellationToken::new())
        .unwrap();
    let error = operator
        .process(
            &BTreeMap::from([("input".into(), batch)]),
            &BatchOperatorContext { run: &run },
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("unsupported_mode"), "{error}");
}
