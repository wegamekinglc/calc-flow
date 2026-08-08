use std::{sync::Arc, time::Duration};

use calc_flow::{
    AggregateFunction, CalcFlowError, OperatorMetadata, PipelineBuilder, StreamRequirements,
    UdfRegistry, WindowAggregateOperator, WindowGeometry, WindowSpec,
};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit};

fn schema(fields: Vec<Field>) -> SchemaRef {
    Arc::new(Schema::new(fields))
}

fn input_schema(value_type: DataType) -> SchemaRef {
    schema(vec![
        Field::new(
            "event_time",
            DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC"))),
            false,
        ),
        Field::new("group", DataType::Utf8, true),
        Field::new("value", value_type, true),
    ])
}

fn assert_invalid_field<T>(result: calc_flow::Result<T>, expected: &str) {
    assert!(
        matches!(result, Err(CalcFlowError::InvalidArgument { field, .. }) if field == expected),
        "expected InvalidArgument at {expected:?}"
    );
}

fn aggregate_operator(
    function: AggregateFunction,
    value_type: DataType,
) -> calc_flow::Result<WindowAggregateOperator> {
    let spec = WindowSpec::tumbling("event_time", Duration::from_secs(60))?
        .group_by(["group"])?
        .aggregate(function, "value", "result")?;
    WindowAggregateOperator::new("window", input_schema(value_type), spec)
}

#[test]
fn duration_and_hopping_geometry_validation_names_the_exact_field() {
    assert_invalid_field(
        WindowSpec::tumbling("event_time", Duration::ZERO),
        "window.geometry.size",
    );
    assert_invalid_field(
        WindowSpec::tumbling("event_time", Duration::from_nanos(1)),
        "window.geometry.size",
    );
    assert_invalid_field(
        WindowSpec::tumbling("event_time", Duration::new(u64::MAX, 0)),
        "window.geometry.size",
    );
    assert_invalid_field(
        WindowSpec::hopping("event_time", Duration::from_micros(10), Duration::ZERO),
        "window.geometry.slide",
    );
    assert_invalid_field(
        WindowSpec::hopping(
            "event_time",
            Duration::from_micros(10),
            Duration::from_nanos(1),
        ),
        "window.geometry.slide",
    );
    assert_invalid_field(
        WindowSpec::hopping(
            "event_time",
            Duration::from_micros(10),
            Duration::from_micros(3),
        ),
        "window.geometry",
    );
    assert_invalid_field(
        WindowSpec::hopping(
            "event_time",
            Duration::from_micros(1_025),
            Duration::from_micros(1),
        ),
        "window.geometry",
    );
}

#[test]
fn event_time_group_and_output_names_fail_before_execution() {
    let missing = WindowSpec::tumbling("missing", Duration::from_secs(1)).unwrap();
    assert!(matches!(
        WindowAggregateOperator::new("window", input_schema(DataType::Int64), missing),
        Err(CalcFlowError::Compile { .. })
    ));

    for data_type in [
        DataType::Int64,
        DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("Europe/London"))),
    ] {
        let input = schema(vec![Field::new("event_time", data_type, false)]);
        let spec = WindowSpec::tumbling("event_time", Duration::from_secs(1)).unwrap();
        assert!(matches!(
            WindowAggregateOperator::new("window", input, spec),
            Err(CalcFlowError::Compile { .. })
        ));
    }

    assert_invalid_field(
        WindowSpec::tumbling("event_time", Duration::from_secs(1))
            .unwrap()
            .group_by(["group", "group"]),
        "window.group_by[1]",
    );
    assert_invalid_field(
        WindowSpec::tumbling("event_time", Duration::from_secs(1))
            .unwrap()
            .group_by(["window_start"]),
        "window.group_by[0]",
    );
    assert_invalid_field(
        WindowSpec::tumbling("event_time", Duration::from_secs(1))
            .unwrap()
            .group_by(["group"])
            .unwrap()
            .aggregate(AggregateFunction::Count, "value", "group"),
        "window.aggregates[0].output",
    );
    let duplicate_output = WindowSpec::tumbling("event_time", Duration::from_secs(1))
        .unwrap()
        .aggregate(AggregateFunction::Count, "value", "result")
        .unwrap()
        .aggregate(AggregateFunction::Sum, "value", "result");
    assert_invalid_field(duplicate_output, "window.aggregates[1].output");
}

#[test]
fn accepted_aggregate_matrix_builds_the_exact_output_schema() {
    let input = schema(vec![
        Field::new(
            "event_time",
            DataType::Timestamp(TimeUnit::Millisecond, None),
            false,
        ),
        Field::new("symbol", DataType::Utf8, true),
        Field::new("signed", DataType::Int32, true),
        Field::new("unsigned", DataType::UInt32, true),
        Field::new("float", DataType::Float32, true),
        Field::new("text", DataType::LargeUtf8, true),
        Field::new("flag", DataType::Boolean, false),
    ]);
    let spec = WindowSpec::tumbling("event_time", Duration::from_secs(60))
        .unwrap()
        .group_by(["symbol"])
        .unwrap()
        .aggregate(AggregateFunction::Count, "text", "count_text")
        .unwrap()
        .aggregate(AggregateFunction::Sum, "signed", "sum_signed")
        .unwrap()
        .aggregate(AggregateFunction::Sum, "unsigned", "sum_unsigned")
        .unwrap()
        .aggregate(AggregateFunction::Sum, "float", "sum_float")
        .unwrap()
        .aggregate(AggregateFunction::Min, "text", "min_text")
        .unwrap()
        .aggregate(AggregateFunction::Max, "flag", "max_flag")
        .unwrap()
        .aggregate(AggregateFunction::Avg, "signed", "avg_signed")
        .unwrap();
    let operator = WindowAggregateOperator::new("window", input.clone(), spec).unwrap();

    assert_eq!(operator.input_ports()[0].schema(), Some(&input));
    let output = operator.output_ports()[0].schema().unwrap();
    let expected = schema(vec![
        Field::new(
            "window_start",
            DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC"))),
            false,
        ),
        Field::new(
            "window_end",
            DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC"))),
            false,
        ),
        Field::new("symbol", DataType::Utf8, true),
        Field::new("count_text", DataType::UInt64, false),
        Field::new("sum_signed", DataType::Int64, true),
        Field::new("sum_unsigned", DataType::UInt64, true),
        Field::new("sum_float", DataType::Float64, true),
        Field::new("min_text", DataType::LargeUtf8, true),
        Field::new("max_flag", DataType::Boolean, true),
        Field::new("avg_signed", DataType::Float64, true),
    ]);
    assert_eq!(output, &expected);
}

#[test]
fn accepted_and_rejected_type_matrix_is_closed() {
    let signed = [
        DataType::Int8,
        DataType::Int16,
        DataType::Int32,
        DataType::Int64,
    ];
    let unsigned = [
        DataType::UInt8,
        DataType::UInt16,
        DataType::UInt32,
        DataType::UInt64,
    ];
    let floats = [DataType::Float32, DataType::Float64];
    for data_type in signed.clone() {
        assert_eq!(
            aggregate_operator(AggregateFunction::Sum, data_type)
                .unwrap()
                .output_ports()[0]
                .schema()
                .unwrap()
                .field_with_name("result")
                .unwrap()
                .data_type(),
            &DataType::Int64
        );
    }
    for data_type in unsigned.clone() {
        assert_eq!(
            aggregate_operator(AggregateFunction::Sum, data_type)
                .unwrap()
                .output_ports()[0]
                .schema()
                .unwrap()
                .field_with_name("result")
                .unwrap()
                .data_type(),
            &DataType::UInt64
        );
    }
    for data_type in floats.clone() {
        assert!(aggregate_operator(AggregateFunction::Sum, data_type.clone()).is_ok());
        assert!(aggregate_operator(AggregateFunction::Avg, data_type).is_ok());
    }
    for data_type in signed.into_iter().chain(unsigned).chain(floats) {
        assert!(aggregate_operator(AggregateFunction::Avg, data_type).is_ok());
    }
    for data_type in [
        DataType::Int64,
        DataType::UInt64,
        DataType::Float32,
        DataType::Float64,
        DataType::Boolean,
        DataType::Utf8,
        DataType::LargeUtf8,
        DataType::Date32,
        DataType::Date64,
        DataType::Timestamp(TimeUnit::Microsecond, None),
        DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC"))),
    ] {
        assert!(aggregate_operator(AggregateFunction::Min, data_type.clone()).is_ok());
        assert!(aggregate_operator(AggregateFunction::Max, data_type).is_ok());
    }
    for data_type in [
        DataType::Binary,
        DataType::Decimal128(10, 2),
        DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
    ] {
        assert!(aggregate_operator(AggregateFunction::Count, data_type).is_ok());
    }

    for (function, data_type) in [
        (AggregateFunction::Sum, DataType::Boolean),
        (AggregateFunction::Avg, DataType::Utf8),
        (AggregateFunction::Min, DataType::Binary),
        (AggregateFunction::Max, DataType::Decimal128(10, 2)),
        (AggregateFunction::Avg, DataType::Decimal256(10, 2)),
        (
            AggregateFunction::Min,
            DataType::Timestamp(TimeUnit::Second, None),
        ),
    ] {
        assert!(matches!(
            aggregate_operator(function, data_type),
            Err(CalcFlowError::Compile { .. })
        ));
    }
}

#[test]
fn group_type_matrix_accepts_only_the_frozen_g1_types() {
    for data_type in [
        DataType::Boolean,
        DataType::Int8,
        DataType::Int64,
        DataType::UInt8,
        DataType::UInt64,
        DataType::Float32,
        DataType::Float64,
        DataType::Utf8,
        DataType::LargeUtf8,
        DataType::Date32,
        DataType::Date64,
        DataType::Timestamp(TimeUnit::Microsecond, None),
        DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC"))),
    ] {
        let input = input_schema(data_type);
        let spec = WindowSpec::tumbling("event_time", Duration::from_secs(1))
            .unwrap()
            .group_by(["value"])
            .unwrap();
        assert!(WindowAggregateOperator::new("window", input, spec).is_ok());
    }

    for data_type in [
        DataType::Binary,
        DataType::Decimal128(10, 2),
        DataType::Timestamp(TimeUnit::Second, None),
        DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("Asia/Shanghai"))),
    ] {
        let input = input_schema(data_type);
        let spec = WindowSpec::tumbling("event_time", Duration::from_secs(1))
            .unwrap()
            .group_by(["value"])
            .unwrap();
        assert!(matches!(
            WindowAggregateOperator::new("window", input, spec),
            Err(CalcFlowError::Compile { .. })
        ));
    }
}

#[test]
fn configuration_fingerprint_and_stream_only_mode_are_deterministic() {
    let input = input_schema(DataType::Int64);
    let spec = WindowSpec::hopping(
        "event_time",
        Duration::from_secs(60),
        Duration::from_secs(10),
    )
    .unwrap()
    .group_by(["group"])
    .unwrap()
    .aggregate(AggregateFunction::Sum, "value", "result")
    .unwrap();
    let operator = WindowAggregateOperator::new("window", input.clone(), spec.clone()).unwrap();
    assert_eq!(
        operator.configuration()["geometry"],
        serde_json::json!({"kind": "hopping", "size_micros": 60_000_000_u64, "slide_micros": 10_000_000_u64})
    );
    assert_eq!(operator.configuration()["state_layout_version"], 1);

    let udfs = UdfRegistry::new().snapshot();
    let fingerprint = PipelineBuilder::new("orders")
        .unwrap()
        .add_node("window", operator)
        .unwrap()
        .compile_stream(&udfs, &StreamRequirements::default())
        .unwrap()
        .fingerprint()
        .to_owned();
    let changed = WindowAggregateOperator::new(
        "window",
        input.clone(),
        spec.clone()
            .aggregate(AggregateFunction::Count, "value", "count")
            .unwrap(),
    )
    .unwrap();
    let changed_fingerprint = PipelineBuilder::new("orders")
        .unwrap()
        .add_node("window", changed)
        .unwrap()
        .compile_stream(&udfs, &StreamRequirements::default())
        .unwrap()
        .fingerprint()
        .to_owned();
    assert_ne!(fingerprint, changed_fingerprint);

    let graph = |order: [&str; 2]| {
        let mut builder = PipelineBuilder::new("orders").unwrap();
        for node in order {
            builder = builder
                .add_node(
                    node,
                    WindowAggregateOperator::new(node, input.clone(), spec.clone()).unwrap(),
                )
                .unwrap();
        }
        builder
            .compile_stream(&udfs, &StreamRequirements::default())
            .unwrap()
            .fingerprint()
            .to_owned()
    };
    assert_eq!(graph(["alpha", "beta"]), graph(["beta", "alpha"]));

    let batch_operator = WindowAggregateOperator::new(
        "window",
        input,
        WindowSpec {
            event_time_column: "event_time".into(),
            group_by: vec!["group".into()],
            geometry: WindowGeometry::Tumbling {
                size_micros: 1_000_000,
            },
            aggregates: Vec::new(),
        },
    )
    .unwrap();
    assert!(matches!(
        PipelineBuilder::new("orders")
            .unwrap()
            .add_node("window", batch_operator)
            .unwrap()
            .compile_batch(&udfs),
        Err(CalcFlowError::Compile { .. })
    ));
}
