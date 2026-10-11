use calc_flow::{
    JoinStateLimits, JoinTimeBounds, OperatorMetadata, Port, STREAM_JOIN_MAX_SAFE_JSON_INTEGER,
    StreamJoinOperator, StreamJoinSpec, StreamJoinType,
};
use datafusion::arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use std::{sync::Arc, time::Duration};

fn left_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("account_id", DataType::Int64, false),
        Field::new(
            "authorized_at",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("amount", DataType::Int64, true),
    ]))
}

fn right_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("account_id", DataType::Int64, false),
        Field::new(
            "paid_at",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("status", DataType::Utf8, true),
    ]))
}

fn spec() -> StreamJoinSpec {
    StreamJoinSpec::inner(
        ["account_id"],
        ["account_id"],
        "authorized_at",
        "paid_at",
        JoinTimeBounds::new(Duration::from_secs(300), Duration::from_secs(60)).unwrap(),
        JoinStateLimits::new(100, 1_000_000, 1_000).unwrap(),
    )
    .unwrap()
    .with_prefixes("authorization", "payment")
    .unwrap()
}

#[test]
fn derives_exact_prefixed_ports() {
    let operator = StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();

    assert_eq!(
        operator
            .input_ports()
            .iter()
            .map(Port::name)
            .collect::<Vec<_>>(),
        ["left", "right"]
    );
    let output = operator.output_ports()[0].schema().unwrap();
    assert_eq!(
        output
            .fields()
            .iter()
            .map(|field| field.name().as_str())
            .collect::<Vec<_>>(),
        [
            "authorization__account_id",
            "authorization__authorized_at",
            "authorization__amount",
            "payment__account_id",
            "payment__paid_at",
            "payment__status",
        ]
    );
    assert!(output.field(2).is_nullable());
    assert!(output.field(5).is_nullable());
}

#[test]
fn rejects_values_beyond_the_json_safe_integer_domain() {
    let too_large = STREAM_JOIN_MAX_SAFE_JSON_INTEGER + 1;
    assert!(JoinStateLimits::new(too_large, 1, 1).is_err());
    assert!(JoinTimeBounds::new(Duration::from_micros(too_large), Duration::ZERO).is_err());
    assert!(JoinStateLimits::new(1, 1, 1).is_ok());
}

#[test]
fn metadata_exposes_data_only_configuration_and_debug() {
    let operator = StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    assert_eq!(operator.name(), "match");
    assert_eq!(operator.input_ports().len(), 2);
    assert_eq!(operator.output_ports().len(), 1);
    let configuration = operator.configuration();
    assert_eq!(configuration["join_type"], "inner");
    assert_eq!(configuration["left_event_time"], "authorized_at");
    assert!(!configuration.contains_key("callable"));
    let debug = format!("{operator:?}");
    assert!(debug.contains("match"), "{debug}");
}

#[test]
fn spec_validation_rejects_invalid_declarations() {
    let bounds = JoinTimeBounds::new(Duration::from_secs(300), Duration::from_secs(60)).unwrap();
    let limits = JoinStateLimits::new(100, 1_000_000, 1_000).unwrap();

    let empty_keys = StreamJoinSpec::inner(
        Vec::<String>::new(),
        ["account_id"],
        "authorized_at",
        "paid_at",
        bounds,
        limits,
    );
    assert!(empty_keys.is_err());

    let unequal = StreamJoinSpec::inner(
        ["a", "b"],
        ["account_id"],
        "authorized_at",
        "paid_at",
        bounds,
        limits,
    );
    assert!(unequal.is_err());

    let duplicate = StreamJoinSpec::inner(
        ["account_id", "account_id"],
        ["account_id", "account_id"],
        "authorized_at",
        "paid_at",
        bounds,
        limits,
    );
    assert!(duplicate.is_err());

    let empty_event_time = StreamJoinSpec::inner(
        ["account_id"],
        ["account_id"],
        "",
        "paid_at",
        bounds,
        limits,
    );
    assert!(empty_event_time.is_err());

    let valid = StreamJoinSpec::inner(
        ["account_id"],
        ["account_id"],
        "authorized_at",
        "paid_at",
        bounds,
        limits,
    )
    .unwrap();
    assert!(valid.clone().with_prefixes("same", "same").is_err());
    assert!(valid.clone().with_prefixes("not valid", "right").is_err());

    let prefixed = valid.with_prefixes("authorization", "payment").unwrap();
    assert_eq!(prefixed.left_keys(), ["account_id"]);
    assert_eq!(prefixed.right_keys(), ["account_id"]);
    assert_eq!(prefixed.left_event_time(), "authorized_at");
    assert_eq!(prefixed.right_event_time(), "paid_at");
    assert_eq!(prefixed.left_prefix(), "authorization");
    assert_eq!(prefixed.right_prefix(), "payment");
    assert_eq!(prefixed.join_type(), StreamJoinType::Inner);
    assert_eq!(prefixed.bounds().before(), Duration::from_secs(300));
    assert_eq!(prefixed.bounds().after(), Duration::from_secs(60));
    assert_eq!(prefixed.limits().max_state_rows_per_side(), 100);
    assert_eq!(prefixed.limits().max_state_bytes_per_side(), 1_000_000);
    assert_eq!(prefixed.limits().max_matches_per_input_batch(), 1_000);
    assert!(format!("{prefixed:?}").contains("StreamJoinSpec"));
}

#[test]
fn serde_round_trips_and_rejects_unknown_or_wrong_kind_fields() {
    let source = r#"{
            "join_type": "inner",
            "left_keys": ["account_id"],
            "right_keys": ["account_id"],
            "left_event_time": "authorized_at",
            "right_event_time": "paid_at",
            "bounds": {"before_micros": 300000000, "after_micros": 60000000},
            "limits": {
                "max_state_rows_per_side": 100,
                "max_state_bytes_per_side": 1000000,
                "max_matches_per_input_batch": 1000
            }
        }"#;
    let parsed: StreamJoinSpec = serde_json::from_str(source).unwrap();
    assert_eq!(parsed.left_prefix(), "left");
    assert_eq!(parsed.right_prefix(), "right");
    let encoded = serde_json::to_value(&parsed).unwrap();
    assert_eq!(encoded["bounds"]["before_micros"], 300_000_000);

    let unknown = source.replace(
        "\"join_type\": \"inner\",",
        "\"join_type\": \"inner\", \"extra\": 1,",
    );
    assert!(serde_json::from_str::<StreamJoinSpec>(&unknown).is_err());

    let outer_join = source.replace("\"inner\"", "\"outer\"");
    assert!(serde_json::from_str::<StreamJoinSpec>(&outer_join).is_err());

    let unknown_bound = source.replace(
        "\"before_micros\": 300000000,",
        "\"before_micros\": 300000000, \"extra\": 1,",
    );
    assert!(serde_json::from_str::<StreamJoinSpec>(&unknown_bound).is_err());

    let unknown_limit = source.replace(
        "\"max_matches_per_input_batch\": 1000",
        "\"max_matches_per_input_batch\": 1000, \"extra\": 1",
    );
    assert!(serde_json::from_str::<StreamJoinSpec>(&unknown_limit).is_err());

    let zero_limit = source.replace(
        "\"max_state_rows_per_side\": 100",
        "\"max_state_rows_per_side\": 0",
    );
    assert!(serde_json::from_str::<StreamJoinSpec>(&zero_limit).is_err());
}

#[tokio::test]
async fn event_time_columns_accept_every_timestamp_unit_and_reject_others() {
    for (unit, _value) in [
        (TimeUnit::Second, 1_i64),
        (TimeUnit::Millisecond, 1_000),
        (TimeUnit::Microsecond, 1_000_000),
        (TimeUnit::Nanosecond, 1_000_000_000),
    ] {
        let schema = Arc::new(Schema::new(vec![
            Field::new("account_id", DataType::Int64, false),
            Field::new(
                "authorized_at",
                DataType::Timestamp(unit, Some("UTC".into())),
                false,
            ),
            Field::new("amount", DataType::Int64, true),
        ]));
        let operator = StreamJoinOperator::new("match", schema, right_schema(), spec());
        assert!(operator.is_ok(), "unit {unit:?} must be supported");
    }

    let not_a_timestamp = Arc::new(Schema::new(vec![
        Field::new("account_id", DataType::Int64, false),
        Field::new("authorized_at", DataType::Int64, false),
        Field::new("amount", DataType::Int64, true),
    ]));
    let rejected = StreamJoinOperator::new("match", not_a_timestamp, right_schema(), spec());
    assert!(
        rejected.is_err(),
        "non-timestamp event time must be rejected"
    );

    let missing_key = Arc::new(Schema::new(vec![
        Field::new("ledger_id", DataType::Int64, false),
        Field::new(
            "authorized_at",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
    ]));
    let rejected = StreamJoinOperator::new("match", missing_key, right_schema(), spec());
    assert!(rejected.is_err(), "missing key column must be rejected");

    let wrong_key_type = Arc::new(Schema::new(vec![
        Field::new("account_id", DataType::Float64, false),
        Field::new(
            "authorized_at",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("amount", DataType::Int64, true),
    ]));
    let rejected = StreamJoinOperator::new("match", wrong_key_type, right_schema(), spec());
    assert!(rejected.is_err(), "unsupported key type must be rejected");

    let zoned = Arc::new(Schema::new(vec![
        Field::new("account_id", DataType::Int64, false),
        Field::new(
            "authorized_at",
            DataType::Timestamp(TimeUnit::Microsecond, Some("America/New_York".into())),
            false,
        ),
        Field::new("amount", DataType::Int64, true),
    ]));
    let rejected = StreamJoinOperator::new("match", zoned, right_schema(), spec());
    assert!(rejected.is_err(), "non-UTC timezone must be rejected");
}
