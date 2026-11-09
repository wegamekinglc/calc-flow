use super::*;
use datafusion::arrow::{
    datatypes::{DataType, Field, Schema},
    ipc::{self, convert::IpcSchemaEncoder, writer::DictionaryTracker},
};

fn encoded(schema: &Schema) -> Vec<u8> {
    let mut dictionaries = DictionaryTracker::new(false);
    IpcSchemaEncoder::new()
        .with_dictionary_tracker(&mut dictionaries)
        .schema_to_fb(schema)
        .finished_data()
        .to_vec()
}
fn checked(actual: &Schema, expected: &Schema) -> Result<SchemaCounts> {
    let bytes = encoded(actual);
    validate(ipc::root_as_schema(&bytes).unwrap(), expected, &mut || {
        Ok(())
    })
}
#[test]
fn test_schema_validates_borrowed_fields_nullability_and_nested_order() {
    let fields = vec![
        Field::new("id", DataType::UInt64, false),
        Field::new(
            "items",
            DataType::List(std::sync::Arc::new(Field::new(
                "item",
                DataType::Utf8,
                true,
            ))),
            true,
        ),
    ];
    let expected = Schema::new(fields);
    let counts = checked(&expected, &expected).unwrap();
    assert_eq!(counts.fields, 3);
    assert_eq!(counts.dictionary_fields, 0);
    assert_eq!(counts.field_name_bytes, 11);
    let wrong = Schema::new(vec![
        Field::new("id", DataType::UInt64, true),
        expected.field(1).clone(),
    ]);
    assert!(matches!(
        checked(&wrong, &expected),
        Err(CalcFlowError::Format { .. })
    ));
    let wrong = Schema::new(vec![expected.field(1).clone(), expected.field(0).clone()]);
    assert!(checked(&wrong, &expected).is_err());
}

fn types() -> Vec<DataType> {
    use DataType::*;
    use datafusion::arrow::datatypes::{IntervalUnit as I, TimeUnit as T, UnionFields, UnionMode};
    let child = std::sync::Arc::new(Field::new("child", Int32, true));
    let entries = std::sync::Arc::new(Field::new(
        "entries",
        Struct(
            vec![
                Field::new("key", Utf8, false),
                Field::new("value", Int32, true),
            ]
            .into(),
        ),
        false,
    ));
    let mut types = vec![
        Null,
        Boolean,
        Int8,
        Int16,
        Int32,
        Int64,
        UInt8,
        UInt16,
        UInt32,
        UInt64,
        Float16,
        Float32,
        Float64,
        Date32,
        Date64,
        Time32(T::Second),
        Time32(T::Millisecond),
        Time64(T::Microsecond),
        Time64(T::Nanosecond),
        Interval(I::YearMonth),
        Interval(I::DayTime),
        Interval(I::MonthDayNano),
        Binary,
        LargeBinary,
        BinaryView,
        Utf8,
        LargeUtf8,
        Utf8View,
        FixedSizeBinary(7),
        List(child.clone()),
        LargeList(child.clone()),
        ListView(child.clone()),
        LargeListView(child.clone()),
        FixedSizeList(child.clone(), 3),
        Struct(vec![child.clone()].into()),
        Map(entries.clone(), false),
        Map(entries, true),
        RunEndEncoded(
            std::sync::Arc::new(Field::new("run_ends", Int32, false)),
            child,
        ),
        Decimal32(8, -2),
        Decimal64(17, 3),
        Decimal128(28, 4),
        Decimal256(60, -7),
    ];
    for unit in [T::Second, T::Millisecond, T::Microsecond, T::Nanosecond] {
        types.push(Timestamp(unit, None));
        types.push(Timestamp(unit, Some("UTC".into())));
        types.push(Duration(unit));
    }
    for mode in [UnionMode::Sparse, UnionMode::Dense] {
        types.push(Union(
            UnionFields::try_new(
                [3, 7],
                [Field::new("a", Float32, true), Field::new("b", Utf8, true)],
            )
            .unwrap(),
            mode,
        ));
    }
    for key in [Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64] {
        types.push(Dictionary(Box::new(key), Box::new(Utf8)));
    }
    types
}
#[test]
fn test_schema_covers_all_pinned_datatype_variants_and_rejects_other_types() {
    let expected = Schema::new(vec![Field::new("f", DataType::UInt64, true)]);
    for dtype in types() {
        let actual = Schema::new(vec![Field::new("f", dtype.clone(), true)]);
        checked(&actual, &actual).unwrap_or_else(|error| panic!("{dtype:?}: {error}"));
        if dtype != DataType::UInt64 {
            assert!(checked(&actual, &expected).is_err(), "{dtype:?}");
        }
    }
}
#[test]
fn test_schema_rejects_changed_nested_child_names_and_nullable_flags() {
    let expected = Schema::new(vec![Field::new(
        "f",
        DataType::Struct(vec![Field::new("inside", DataType::Float64, false)].into()),
        true,
    )]);
    for child in [
        Field::new("other", DataType::Float64, false),
        Field::new("inside", DataType::Float64, true),
    ] {
        let actual = Schema::new(vec![Field::new(
            "f",
            DataType::Struct(vec![child].into()),
            true,
        )]);
        assert!(checked(&actual, &expected).is_err());
    }
}

fn dictionary_schema() -> Schema {
    use DataType::*;
    let leaf = Field::new("leaf", Dictionary(Box::new(UInt16), Box::new(Utf8)), true)
        .with_dict_is_ordered(true);
    let outer = Field::new(
        "outer",
        Dictionary(Box::new(Int32), Box::new(Struct(vec![leaf].into()))),
        false,
    );
    Schema::new(vec![
        outer,
        Field::new("last", Dictionary(Box::new(Int8), Box::new(Binary)), true),
    ])
}
#[test]
fn test_schema_uses_postorder_dictionary_ids_for_nested_values_and_borrowed_lookup() {
    let expected = dictionary_schema();
    let counts = checked(&expected, &expected).unwrap();
    assert_eq!(counts.dictionary_fields, 3);
    let value = dictionary_value(&expected, 0, &mut || Ok(()))
        .unwrap()
        .unwrap();
    assert_eq!(value, &DataType::Utf8);
    let value = dictionary_value(&expected, 1, &mut || Ok(()))
        .unwrap()
        .unwrap();
    assert!(matches!(value, DataType::Struct(_)));
    assert_eq!(
        dictionary_value(&expected, 2, &mut || Ok(())).unwrap(),
        Some(&DataType::Binary)
    );
    assert_eq!(
        dictionary_value(&expected, 3, &mut || Ok(())).unwrap(),
        None
    );
}

fn metadata_root(pairs: &[(Option<&str>, Option<&str>)]) -> Vec<u8> {
    let mut builder = IpcSchemaEncoder::new().schema_to_fb(&Schema::empty());
    builder.reset();
    let pairs: Vec<_> = pairs
        .iter()
        .map(|(key, value)| {
            let key = key.map(|value| builder.create_string(value));
            let value = value.map(|value| builder.create_string(value));
            ipc::KeyValue::create(&mut builder, &ipc::KeyValueArgs { key, value })
        })
        .collect();
    let metadata = builder.create_vector(&pairs);
    let unused = ipc::Field::create(&mut builder, &ipc::FieldArgs::default());
    let fields = builder.create_vector(&[unused][..0]);
    let schema = ipc::Schema::create(
        &mut builder,
        &ipc::SchemaArgs {
            fields: Some(fields),
            custom_metadata: Some(metadata),
            ..Default::default()
        },
    );
    builder.finish(schema, None);
    builder.finished_data().to_vec()
}
#[test]
fn test_metadata_is_canonical_unique_complete_and_counted_without_copying() {
    let expected = Schema::empty()
        .with_metadata([("a".into(), "one".into()), ("z".into(), "two".into())].into());
    let good = metadata_root(&[(Some("a"), Some("one")), (Some("z"), Some("two"))]);
    let counts = validate(ipc::root_as_schema(&good).unwrap(), &expected, &mut || {
        Ok(())
    })
    .unwrap();
    assert_eq!(counts.metadata_pairs, 2);
    assert_eq!(counts.metadata_string_bytes, 8);
    for pairs in [
        [(Some("z"), Some("two")), (Some("a"), Some("one"))],
        [(Some("a"), Some("one")), (Some("a"), Some("one"))],
        [(Some("a"), Some("one")), (Some("unknown"), Some("two"))],
        [(None, Some("one")), (Some("z"), Some("two"))],
        [(Some("a"), None), (Some("z"), Some("two"))],
    ] {
        let bytes = metadata_root(&pairs);
        assert!(
            validate(ipc::root_as_schema(&bytes).unwrap(), &expected, &mut || Ok(
                ()
            ))
            .is_err()
        );
    }
}
#[test]
fn test_field_metadata_and_unicode_string_counts_are_exact() {
    let expected = Schema::new(vec![
        Field::new("字段", DataType::Utf8, false)
            .with_metadata([("键".into(), "值".into())].into()),
    ]);
    let counts = checked(&expected, &expected).unwrap();
    assert_eq!(counts.field_name_bytes, 6);
    assert_eq!(counts.metadata_pairs, 1);
    assert_eq!(counts.metadata_string_bytes, 6);
    let actual = Schema::new(vec![Field::new("字段", DataType::Utf8, false)]);
    assert!(checked(&actual, &expected).is_err());
}
#[test]
fn test_datatype_parameters_are_strict_for_same_physical_type() {
    use DataType::*;
    use datafusion::arrow::datatypes::{IntervalUnit as I, TimeUnit as T, UnionFields, UnionMode};
    let child = std::sync::Arc::new(Field::new("child", Float64, true));
    let pairs = [
        (
            Timestamp(T::Second, Some("UTC".into())),
            Timestamp(T::Second, None),
        ),
        (
            Timestamp(T::Second, Some("UTC".into())),
            Timestamp(T::Second, Some("+01:00".into())),
        ),
        (Timestamp(T::Second, None), Timestamp(T::Millisecond, None)),
        (Time32(T::Second), Time32(T::Millisecond)),
        (Time32(T::Second), Time64(T::Nanosecond)),
        (Duration(T::Second), Duration(T::Millisecond)),
        (Date32, Date64),
        (Interval(I::YearMonth), Interval(I::DayTime)),
        (Float32, Float64),
        (Int16, UInt16),
        (Decimal32(8, 2), Decimal64(8, 2)),
        (Decimal128(28, 2), Decimal128(29, 2)),
        (Decimal256(60, -2), Decimal256(60, 2)),
        (FixedSizeBinary(5), FixedSizeBinary(7)),
        (
            FixedSizeList(child.clone(), 2),
            FixedSizeList(child.clone(), 3),
        ),
        (Map(child.clone(), false), Map(child, true)),
        (
            Union(
                UnionFields::try_new([3], [Field::new("a", Float32, true)]).unwrap(),
                UnionMode::Sparse,
            ),
            Union(
                UnionFields::try_new([7], [Field::new("a", Float32, true)]).unwrap(),
                UnionMode::Sparse,
            ),
        ),
    ];
    for (actual, expected) in pairs {
        assert!(
            checked(
                &Schema::new(vec![Field::new("f", actual, true)]),
                &Schema::new(vec![Field::new("f", expected, true)])
            )
            .is_err()
        );
    }
}
#[test]
fn test_dictionary_index_order_and_canonical_id_are_strict() {
    let expected = dictionary_schema();
    let mut bytes = encoded(&expected);
    let actual = ipc::root_as_schema(&bytes).unwrap();
    let dictionary = actual.fields().unwrap().get(0).dictionary().unwrap();
    assert_eq!(dictionary.id(), 1);
    let offset = dictionary._tab.loc()
        + usize::from(dictionary._tab.vtable().get(ipc::DictionaryEncoding::VT_ID));
    bytes[offset..offset + 8].copy_from_slice(&9_i64.to_le_bytes());
    assert!(
        validate(ipc::root_as_schema(&bytes).unwrap(), &expected, &mut || Ok(
            ()
        ))
        .is_err()
    );
    let expected = Schema::new(vec![
        Field::new(
            "f",
            DataType::Dictionary(Box::new(DataType::Int16), Box::new(DataType::Utf8)),
            true,
        )
        .with_dict_is_ordered(true),
    ]);
    let actual = Schema::new(vec![
        Field::new(
            "f",
            DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8)),
            true,
        )
        .with_dict_is_ordered(true),
    ]);
    assert!(checked(&actual, &expected).is_err());
    let actual = Schema::new(vec![Field::new(
        "f",
        expected.field(0).data_type().clone(),
        true,
    )]);
    assert!(checked(&actual, &expected).is_err());
}

fn header_root(endian: ipc::Endianness, features: Option<&[ipc::Feature]>) -> Vec<u8> {
    let mut builder = IpcSchemaEncoder::new().schema_to_fb(&Schema::empty());
    builder.reset();
    let unused = ipc::Field::create(&mut builder, &ipc::FieldArgs::default());
    let fields = builder.create_vector(&[unused][..0]);
    let features = features.map(|value| builder.create_vector(value));
    let schema = ipc::Schema::create(
        &mut builder,
        &ipc::SchemaArgs {
            endianness: endian,
            fields: Some(fields),
            features,
            ..Default::default()
        },
    );
    builder.finish(schema, None);
    builder.finished_data().to_vec()
}
#[test]
fn test_schema_header_rejects_non_native_endianness_and_noncanonical_features() {
    let native = if cfg!(target_endian = "little") {
        ipc::Endianness::Little
    } else {
        ipc::Endianness::Big
    };
    let other = if native == ipc::Endianness::Little {
        ipc::Endianness::Big
    } else {
        ipc::Endianness::Little
    };
    let bytes = header_root(native, None);
    validate(
        ipc::root_as_schema(&bytes).unwrap(),
        &Schema::empty(),
        &mut || Ok(()),
    )
    .unwrap();
    for bytes in [
        header_root(other, None),
        header_root(native, Some(&[])),
        header_root(native, Some(&[ipc::Feature::COMPRESSED_BODY])),
    ] {
        assert!(
            validate(
                ipc::root_as_schema(&bytes).unwrap(),
                &Schema::empty(),
                &mut || Ok(())
            )
            .is_err()
        );
    }
}
#[test]
fn test_schema_counts_timezone_union_and_metadata_without_owned_allocations() {
    let expected = Schema::new(vec![
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Second, Some("UTC".into())),
            false,
        ),
        dictionary_schema().field(0).clone(),
    ])
    .with_metadata([("tag".into(), "value".into())].into());
    let bytes = encoded(&expected);
    let actual = ipc::root_as_schema(&bytes).unwrap();
    let mut counts = SchemaCounts::default();
    let info = allocation_counter::measure(|| {
        counts = validate(actual, &expected, &mut || Ok(())).unwrap();
        assert_eq!(
            dictionary_value(&expected, 0, &mut || Ok(())).unwrap(),
            Some(&DataType::Utf8)
        );
    });
    assert_eq!(info.count_total, 0);
    assert_eq!(counts.timezone_string_bytes, 3);
    assert_eq!(counts.metadata_string_bytes, 8);
    assert_eq!(counts.dictionary_fields, 2);
    assert_eq!(counts.fields, 3);
}
#[test]
fn test_schema_validation_and_dictionary_lookup_propagate_cancellation_without_payloads() {
    let expected = dictionary_schema();
    let bytes = encoded(&expected);
    let mut check = || {
        Err(CalcFlowError::Cancelled {
            run_id: "schema-control".into(),
        })
    };
    assert!(
        matches!(validate(ipc::root_as_schema(&bytes).unwrap(),&expected,&mut check),Err(CalcFlowError::Cancelled{run_id}) if run_id=="schema-control")
    );
    assert!(
        matches!(dictionary_value(&expected,1,&mut check),Err(CalcFlowError::Cancelled{run_id}) if run_id=="schema-control")
    );
    let actual = Schema::new(vec![Field::new("secret-value", DataType::Int32, true)]);
    let error = checked(&actual, &expected).unwrap_err();
    assert!(!format!("{error:?}").contains("secret-value"));
}
#[test]
fn test_long_metadata_comparison_observes_actual_control_between_bounded_chunks() {
    let expected = Schema::empty().with_metadata([("large".into(), "x".repeat(20_000))].into());
    let bytes = encoded(&expected);
    let mut calls = 0;
    let result = validate(ipc::root_as_schema(&bytes).unwrap(), &expected, &mut || {
        calls += 1;
        if calls == 5 {
            Err(CalcFlowError::Cancelled {
                run_id: "long-schema".into(),
            })
        } else {
            Ok(())
        }
    });
    assert!(matches!(result, Err(CalcFlowError::Cancelled { .. })));
    assert_eq!(calls, 5);
}

#[test]
fn test_schema_rejects_missing_or_additional_children_and_union_mode() {
    let expected = Schema::new(vec![Field::new(
        "f",
        DataType::Struct(vec![Field::new("a", DataType::Float64, true)].into()),
        true,
    )]);
    for children in [
        vec![],
        vec![
            Field::new("a", DataType::Float64, true),
            Field::new("b", DataType::Float64, true),
        ],
    ] {
        let actual = Schema::new(vec![Field::new(
            "f",
            DataType::Struct(children.into()),
            true,
        )]);
        assert!(checked(&actual, &expected).is_err());
    }
    let fields = UnionFields::try_new(
        [3, 7],
        [
            Field::new("a", DataType::Float32, true),
            Field::new("b", DataType::Utf8, true),
        ],
    )
    .unwrap();
    let expected = Schema::new(vec![Field::new(
        "f",
        DataType::Union(fields.clone(), UnionMode::Sparse),
        true,
    )]);
    let actual = Schema::new(vec![Field::new(
        "f",
        DataType::Union(fields, UnionMode::Dense),
        true,
    )]);
    assert!(checked(&actual, &expected).is_err());
    let counts = checked(&expected, &expected).unwrap();
    assert_eq!(counts.union_type_ids, 2);
}
#[test]
fn test_dictionary_lookup_is_controlled_during_one_wide_schema_scan() {
    let expected = Schema::new(
        (0..4097)
            .map(|_| Field::new("f", DataType::Int32, false))
            .collect::<Vec<_>>(),
    );
    let mut calls = 0;
    let result = dictionary_value(&expected, 0, &mut || {
        calls += 1;
        if calls == 4097 {
            Err(CalcFlowError::Cancelled {
                run_id: "dictionary-scan".into(),
            })
        } else {
            Ok(())
        }
    });
    assert!(matches!(result, Err(CalcFlowError::Cancelled { .. })));
    assert_eq!(calls, 4097);
}
#[test]
fn test_checked_counts_and_final_empty_control_fail_before_returning_results() {
    let mut count = usize::MAX;
    assert!(matches!(
        add(&mut count, 1),
        Err(CalcFlowError::Format { .. })
    ));
    assert_eq!(count, usize::MAX);
    let expected = Schema::empty();
    let bytes = encoded(&expected);
    let mut calls = 0;
    let result = validate(ipc::root_as_schema(&bytes).unwrap(), &expected, &mut || {
        calls += 1;
        if calls == 2 {
            Err(CalcFlowError::Cancelled {
                run_id: "final-schema".into(),
            })
        } else {
            Ok(())
        }
    });
    assert!(matches!(result, Err(CalcFlowError::Cancelled { .. })));
    assert_eq!(calls, 2);
}
