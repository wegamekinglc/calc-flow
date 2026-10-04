use super::super::{CandidateMap, RandomState};
use super::*;
use crate::{
    Batch, BatchMetadata, CalcFlowError, CancellationToken, DataFusionConfig, DataFusionRuntime,
    JsonMap, StreamJobContext, StreamOperatorContext,
};
use datafusion::{
    arrow::array::{BooleanArray, Int32Array, Int64Array, UInt64Array, new_empty_array},
    execution::memory_pool::{MemoryConsumer, MemoryPool},
};

#[tokio::test]
async fn test_compact_native_state_wire_ignores_cached_results_and_rebuilds_native_results() {
    let runtime = runtime();
    for grouped in [false, true] {
        let kind = DataType::Decimal128(20, 2);
        let mut original = plan(&runtime, &kind, grouped);
        apply(&mut original, &kind, &[Some(3), None, Some(7)]).await;
        let expected = result(&original);
        let saved = snapshot(&original);
        let first = original.export_native_state("codec", || Ok(())).unwrap();
        for group in &mut original.groups {
            group.results.clear();
        }
        let second = original.export_native_state("codec", || Ok(())).unwrap();
        assert_eq!(first.records(), second.records());
        let restored = plan(&runtime, &kind, grouped)
            .import_native_state(second.records(), 3, true, || Ok(()), "codec")
            .unwrap();
        assert_eq!(result(&restored), expected);
        assert_eq!(snapshot(&restored), saved);
        assert!(original.groups.iter().all(|group| group.results.is_empty()));
    }
}

#[tokio::test]
async fn test_compact_native_state_composite_string_null_keys_rebuild_lookup_and_credit() {
    for kind in [DataType::Utf8, DataType::LargeUtf8] {
        let runtime = runtime();
        let logical = Arc::new(Schema::new(vec![
            Field::new("name", kind.clone(), true),
            Field::new("flag", DataType::Boolean, true),
            Field::new("value", DataType::Int64, true),
        ]));
        let query = crate::expression::parse_select_query(
            "SELECT name, flag, SUM(value), COUNT(value), COUNT(*) FROM events GROUP BY name, flag",
        )
        .unwrap();
        let make = || {
            IncrementalSql::plan_sync(
                &runtime,
                &query,
                "events",
                logical.clone(),
                logical.clone(),
                "codec",
            )
            .unwrap()
            .unwrap()
        };
        let mut original = make();
        let wide = "é\0".repeat(32768);
        let names = ScalarValue::iter_to_array(
            [Some("z"), None, Some(wide.as_str()), Some("z")]
                .into_iter()
                .map(|name| {
                    ScalarValue::Utf8(name.map(str::to_owned))
                        .cast_to(&kind)
                        .unwrap()
                }),
        )
        .unwrap();
        let record = RecordBatch::try_new(
            logical.clone(),
            vec![
                names,
                Arc::new(BooleanArray::from(vec![
                    Some(true),
                    None,
                    Some(false),
                    Some(true),
                ])),
                Arc::new(Int64Array::from(vec![Some(3), None, Some(7), Some(4)])),
            ],
        )
        .unwrap();
        let input = Batch::table(vec![record], BatchMetadata::default()).unwrap();
        let job = StreamJobContext::new(1, "codec", JsonMap::new(), None, CancellationToken::new());
        let context = StreamOperatorContext::new(&job, "codec", None);
        let transaction = original.update(&input, &context, "codec").await.unwrap();
        original.commit(transaction);
        let before = pool(&runtime).reserved();
        let exported = original.export_native_state("codec", || Ok(())).unwrap();
        let mut restored = make()
            .import_native_state(exported.records(), 4, true, || Ok(()), "codec")
            .unwrap();
        assert_eq!(snapshot(&restored), snapshot(&original));
        for (slot, group) in restored.groups.iter().enumerate() {
            assert_eq!(restored.index.get(&group.key), Some(&slot));
            let Group {
                _reservation: credit,
                ..
            } = group;
            assert!(credit.size() >= group.key.len());
        }
        let update = original.update(&input, &context, "codec").await.unwrap();
        original.commit(update);
        let update = restored.update(&input, &context, "codec").await.unwrap();
        restored.commit(update);
        assert_eq!(snapshot(&restored), snapshot(&original));
        drop((restored, exported));
        assert_eq!(pool(&runtime).reserved(), before);
        drop(original);
        assert_eq!(pool(&runtime).reserved(), 0);
    }
}

#[tokio::test]
async fn test_compact_native_state_roundtrip_exact_widths_and_continuation() {
    for kind in [
        DataType::Int8,
        DataType::Int16,
        DataType::Int32,
        DataType::Int64,
        DataType::UInt8,
        DataType::UInt16,
        DataType::UInt32,
        DataType::UInt64,
        DataType::Decimal32(9, 2),
        DataType::Decimal64(18, 2),
        DataType::Decimal128(38, 2),
        DataType::Decimal256(76, 2),
    ] {
        for grouped in [false, true] {
            let runtime = runtime();
            let mut original = plan(&runtime, &kind, grouped);
            apply(&mut original, &kind, &[Some(3), None, Some(7)]).await;
            let saved = snapshot(&original);
            let descriptor = original.native_descriptor("codec").unwrap();
            let exported = original.export_native_state("codec", || Ok(())).unwrap();
            assert_eq!(exported.descriptor.wire_schema, descriptor.wire_schema);
            assert_eq!(exported.descriptor.group_count, original.groups.len());
            assert_eq!(descriptor.key_fields.len(), usize::from(grouped));
            assert_eq!(descriptor.aggregate_names.len(), original.aggregates.len());
            assert_eq!(descriptor.output_schema, original.output_schema);
            assert_eq!(descriptor.policy, "exact-numeric-v1");
            assert_eq!(descriptor.aggregate_inputs.len(), original.aggregates.len());
            let all_rows = descriptor
                .count_all_rows
                .iter()
                .position(|all| *all)
                .unwrap();
            assert!(
                matches!(descriptor.aggregate_inputs[all_rows].as_slice(), [NativeAggregateInput::Literal(value)] if !value.is_null())
            );
            let input = &descriptor.aggregate_inputs[0][0];
            if matches!(
                kind,
                DataType::Int8
                    | DataType::Int16
                    | DataType::Int32
                    | DataType::UInt8
                    | DataType::UInt16
                    | DataType::UInt32
            ) {
                let NativeAggregateInput::Cast { input, field, safe } = input else {
                    panic!("expected actual native SUM coercion")
                };
                assert!(!safe);
                assert!(
                    matches!(input.as_ref(), NativeAggregateInput::Column { index: 1, field } if field == &original.schema.fields()[1])
                );
                assert_eq!(
                    field.data_type(),
                    &if matches!(kind, DataType::UInt8 | DataType::UInt16 | DataType::UInt32) {
                        DataType::UInt64
                    } else {
                        DataType::Int64
                    }
                );
            } else {
                assert!(
                    matches!(input, NativeAggregateInput::Column { index: 1, field } if field == &original.schema.fields()[1])
                );
            }
            assert_eq!(
                descriptor.projection_slots[usize::from(grouped) + 4],
                usize::from(grouped) + all_rows
            );
            assert_eq!(
                descriptor.projection_slots[usize::from(grouped) + 5],
                usize::from(grouped) + all_rows
            );
            assert_eq!(
                descriptor.wire_schema.fields().len(),
                descriptor.key_fields.len()
                    + descriptor.state_fields.iter().map(Vec::len).sum::<usize>()
            );
            let mut restored = plan(&runtime, &kind, grouped)
                .import_native_state(exported.records(), 3, true, || Ok(()), "codec")
                .unwrap();
            assert_eq!(snapshot(&restored), saved);
            assert_eq!(snapshot(&original), saved);
            let reexported = restored.export_native_state("codec", || Ok(())).unwrap();
            assert_eq!(reexported.records(), exported.records());
            drop(reexported);
            for (slot, group) in restored.groups.iter().enumerate() {
                assert_eq!(restored.index.get(&group.key), Some(&slot));
            }
            apply(&mut original, &kind, &[None, Some(5), Some(9)]).await;
            apply(&mut restored, &kind, &[None, Some(5), Some(9)]).await;
            assert_eq!(snapshot(&restored), snapshot(&original));
            assert_eq!(result(&restored), result(&original));
            drop((exported, descriptor, restored, original));
            assert_eq!(pool(&runtime).reserved(), 0);
        }
    }
}

#[tokio::test]
async fn test_compact_native_state_avg_all_null_and_global_seed_preserve_states() {
    for kind in [
        DataType::Decimal32(9, 2),
        DataType::Decimal64(18, 2),
        DataType::Decimal128(38, 2),
        DataType::Decimal256(76, 2),
    ] {
        for grouped in [false, true] {
            for values in [vec![], vec![None, None, None]] {
                avg_null_state_case(&kind, grouped, &values).await;
            }
        }
    }
    let runtime = runtime();
    let original = plan(&runtime, &DataType::Int64, false);
    let exported = original.export_native_state("codec", || Ok(())).unwrap();
    let restored = plan(&runtime, &DataType::Int64, false)
        .import_native_state(exported.records(), 0, false, || Ok(()), "codec")
        .unwrap();
    assert!(restored.groups.is_empty());
    assert!(
        plan(&runtime, &DataType::Int64, false)
            .import_native_state(exported.records(), 1, false, || Ok(()), "codec")
            .is_err()
    );
    assert!(
        plan(&runtime, &DataType::Int64, false)
            .import_native_state(exported.records(), 0, true, || Ok(()), "codec")
            .is_err()
    );
}

async fn avg_null_state_case(kind: &DataType, grouped: bool, values: &[Option<i64>]) {
    let runtime = runtime();
    let mut original = plan(&runtime, kind, grouped);
    apply(&mut original, kind, values).await;
    let saved = snapshot(&original);
    let exported = original.export_native_state("codec", || Ok(())).unwrap();
    let mut restored = plan(&runtime, kind, grouped)
        .import_native_state(
            exported.records(),
            values.len() as u64,
            true,
            || Ok(()),
            "codec",
        )
        .unwrap();
    assert_eq!(snapshot(&restored), saved);
    assert_eq!(result(&restored), result(&original));
    if !values.is_empty() {
        let avg = exported
            .descriptor
            .aggregate_names
            .iter()
            .position(|function| function == "avg")
            .unwrap();
        let count_column = exported.descriptor.key_fields.len()
            + exported.descriptor.state_fields[..avg]
                .iter()
                .map(Vec::len)
                .sum::<usize>();
        let nullable = exported
            .records()
            .iter()
            .map(|record| {
                replace(
                    record,
                    count_column,
                    Arc::new(UInt64Array::from(vec![None; record.num_rows()])),
                )
            })
            .collect::<Vec<_>>();
        let mut alternate = plan(&runtime, kind, grouped)
            .import_native_state(&nullable, values.len() as u64, true, || Ok(()), "codec")
            .unwrap();
        assert!(
            alternate
                .groups
                .iter()
                .all(|group| group.states[avg][0] == ScalarValue::UInt64(None))
        );
        assert_eq!(result(&alternate), result(&original));
        apply(&mut alternate, kind, &[Some(4), Some(8), Some(12)]).await;
        apply(&mut restored, kind, &[Some(4), Some(8), Some(12)]).await;
        assert_eq!(snapshot(&alternate), snapshot(&restored));
        restored = plan(&runtime, kind, grouped)
            .import_native_state(
                exported.records(),
                values.len() as u64,
                true,
                || Ok(()),
                "codec",
            )
            .unwrap();
    }
    assert_eq!(
        original.groups.len(),
        if grouped { values.len().min(3) } else { 1 }
    );
    apply(&mut restored, kind, &[Some(4), Some(8)]).await;
    apply(&mut original, kind, &[Some(4), Some(8)]).await;
    assert_eq!(snapshot(&restored), snapshot(&original));
}

#[tokio::test]
async fn test_compact_native_state_rejects_count_schema_null_and_duplicate_corruption() {
    let runtime = runtime();
    let kind = DataType::Decimal128(20, 2);
    let mut original = plan(&runtime, &kind, true);
    apply(&mut original, &kind, &[Some(3), None, Some(7)]).await;
    let saved = snapshot(&original);
    let exported = original.export_native_state("codec", || Ok(())).unwrap();
    let record = &exported.records()[0];
    let count = exported.descriptor.key_fields.len() + exported.descriptor.state_fields[0].len();
    let avg = exported
        .descriptor
        .aggregate_names
        .iter()
        .position(|name| name == "avg")
        .unwrap();
    let avg_count = exported.descriptor.key_fields.len()
        + exported.descriptor.state_fields[..avg]
            .iter()
            .map(Vec::len)
            .sum::<usize>();
    let before = pool(&runtime).reserved();
    let corruptions = [
        replace(
            record,
            count,
            Arc::new(Int64Array::from(vec![-1; record.num_rows()])),
        ),
        replace(
            record,
            count,
            Arc::new(Int64Array::from(vec![4; record.num_rows()])),
        ),
        replace(
            record,
            count,
            Arc::new(Int64Array::from(vec![2; record.num_rows()])),
        ),
        replace(
            record,
            avg_count,
            Arc::new(UInt64Array::from(vec![4; record.num_rows()])),
        ),
        replace(
            record,
            avg_count,
            Arc::new(UInt64Array::from(vec![None; record.num_rows()])),
        ),
        replace(
            record,
            avg_count + 1,
            new_empty_or_null(record.column(avg_count + 1).data_type(), record.num_rows()),
        ),
    ];
    for corrupted in corruptions {
        assert!(
            plan(&runtime, &kind, true)
                .import_native_state(&[corrupted], 3, true, || Ok(()), "codec")
                .is_err()
        );
        assert_eq!(pool(&runtime).reserved(), before);
        assert_eq!(snapshot(&original), saved);
    }
    reject_native_record_shapes(&runtime, &kind, record, count);
    let wrong_width = plan(&runtime, &DataType::Decimal64(18, 2), true).import_native_state(
        exported.records(),
        3,
        true,
        || Ok(()),
        "codec",
    );
    assert!(wrong_width.is_err());
    assert_eq!(pool(&runtime).reserved(), before);
    assert_eq!(snapshot(&original), saved);
}

fn reject_native_record_shapes(
    runtime: &DataFusionRuntime,
    kind: &DataType,
    record: &RecordBatch,
    count: usize,
) {
    let mut null_columns = record.columns().to_vec();
    null_columns[count] = Arc::new(Int64Array::from(vec![None; record.num_rows()]));
    assert!(RecordBatch::try_new(record.schema(), null_columns.clone()).is_err());
    let mut nullable_fields = record
        .schema()
        .fields()
        .iter()
        .map(|field| field.as_ref().clone())
        .collect::<Vec<_>>();
    nullable_fields[count] = nullable_fields[count].clone().with_nullable(true);
    let nullable_count =
        RecordBatch::try_new(Arc::new(Schema::new(nullable_fields)), null_columns).unwrap();
    assert!(
        plan(runtime, kind, true)
            .import_native_state(&[nullable_count], 3, true, || Ok(()), "codec")
            .is_err()
    );
    let duplicate =
        datafusion::arrow::compute::concat_batches(&record.schema(), [record, record]).unwrap();
    assert!(
        plan(runtime, kind, true)
            .import_native_state(&[duplicate], 6, true, || Ok(()), "codec")
            .is_err()
    );
    let fields = record
        .schema()
        .fields()
        .iter()
        .cloned()
        .chain(std::iter::once(Arc::new(Field::new(
            "cached_result",
            DataType::Int64,
            true,
        ))))
        .collect::<Vec<_>>();
    let columns = record
        .columns()
        .iter()
        .cloned()
        .chain(std::iter::once(new_empty_or_null(
            &DataType::Int64,
            record.num_rows(),
        )))
        .collect();
    let extra = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap();
    assert!(
        plan(runtime, kind, true)
            .import_native_state(&[extra], 3, true, || Ok(()), "codec")
            .is_err()
    );
    let mut changed = record
        .schema()
        .fields()
        .iter()
        .map(|field| field.as_ref().clone())
        .collect::<Vec<_>>();
    changed[0] = changed[0]
        .clone()
        .with_metadata(std::collections::HashMap::new());
    let metadata =
        RecordBatch::try_new(Arc::new(Schema::new(changed)), record.columns().to_vec()).unwrap();
    assert!(
        plan(runtime, kind, true)
            .import_native_state(&[metadata], 3, true, || Ok(()), "codec")
            .is_err()
    );
}

#[tokio::test]
async fn test_compact_native_state_credit_refusal_and_cancellation_preserve_installed_state() {
    let runtime = runtime();
    let kind = DataType::Int64;
    let mut original = plan(&runtime, &kind, true);
    apply(
        &mut original,
        &kind,
        &(0..257).map(Some).collect::<Vec<_>>(),
    )
    .await;
    let saved = snapshot(&original);
    let before = pool(&runtime).reserved();
    let pressure = MemoryConsumer::new("codec-pressure").register(&pool(&runtime));
    pressure.try_grow((1 << 30) - before).unwrap();
    assert!(original.export_native_state("codec", || Ok(())).is_err());
    assert_eq!(pool(&runtime).reserved(), 1 << 30);
    drop(pressure);
    assert_eq!(pool(&runtime).reserved(), before);
    let exported = original.export_native_state("codec", || Ok(())).unwrap();
    let mut candidate = plan(&runtime, &kind, true);
    let owned = pool(&runtime).reserved();
    let pressure = MemoryConsumer::new("import-pressure").register(&pool(&runtime));
    pressure.try_grow((1 << 30) - owned).unwrap();
    assert!(
        candidate
            .import_native_state(exported.records(), 257, true, || Ok(()), "codec")
            .is_err()
    );
    assert_eq!(snapshot(&original), saved);
    drop(pressure);
    assert_eq!(
        pool(&runtime).reserved(),
        before + exported.reserved_bytes()
    );
    late_import_refusal(&runtime, &kind, &original, &exported, &saved, before);
    candidate = plan(&runtime, &kind, true);
    let cancellation = CancellationToken::new();
    let job = StreamJobContext::new(1, "codec", JsonMap::new(), None, cancellation.clone());
    let context = StreamOperatorContext::new(&job, "codec", None);
    let mut checks = 0;
    let result = candidate.import_native_state(
        exported.records(),
        257,
        true,
        || {
            checks += 1;
            if checks == 6 {
                cancellation.cancel();
            }
            context.check_cancelled()
        },
        "codec",
    );
    assert!(matches!(result, Err(CalcFlowError::Cancelled { .. })));
    assert_eq!(checks, 6);
    assert_eq!(
        pool(&runtime).reserved(),
        before + exported.reserved_bytes()
    );
    assert_eq!(snapshot(&original), saved);
    let cancellation = CancellationToken::new();
    let job = StreamJobContext::new(1, "codec", JsonMap::new(), None, cancellation.clone());
    let context = StreamOperatorContext::new(&job, "codec", None);
    let mut checks = 0;
    let result = original.export_native_state("codec", || {
        checks += 1;
        if checks == 3 {
            cancellation.cancel();
        }
        context.check_cancelled()
    });
    assert!(matches!(result, Err(CalcFlowError::Cancelled { .. })));
    assert_eq!(
        pool(&runtime).reserved(),
        before + exported.reserved_bytes()
    );
    assert_eq!(snapshot(&original), saved);
    drop((exported, original));
    assert_eq!(pool(&runtime).reserved(), 0);
}

fn late_import_refusal(
    runtime: &DataFusionRuntime,
    kind: &DataType,
    original: &IncrementalSql,
    exported: &PaidNativeStateRecords,
    saved: &Saved,
    before: usize,
) {
    let probe = plan(runtime, kind, true);
    let basis = pool(runtime).reserved();
    let mut peak = basis;
    let probe = probe
        .import_native_state(
            exported.records(),
            257,
            true,
            || {
                peak = peak.max(pool(runtime).reserved());
                Ok(())
            },
            "codec",
        )
        .unwrap();
    assert_eq!(&snapshot(&probe), saved);
    let needed = peak - basis;
    assert!(needed > 0);
    drop(probe);
    let candidate = plan(runtime, kind, true);
    let basis = pool(runtime).reserved();
    let pressure = MemoryConsumer::new("late-import-pressure").register(&pool(runtime));
    pressure.try_grow((1 << 30) - basis - (needed - 1)).unwrap();
    let mut checks = 0;
    let refused = candidate.import_native_state(
        exported.records(),
        257,
        true,
        || {
            checks += 1;
            Ok(())
        },
        "codec",
    );
    assert!(matches!(refused, Err(CalcFlowError::DataFusion { .. })));
    assert!(checks >= 5);
    drop(pressure);
    assert_eq!(pool(runtime).reserved(), before + exported.reserved_bytes());
    assert_eq!(&snapshot(original), saved);
}

fn runtime() -> DataFusionRuntime {
    DataFusionRuntime::new(DataFusionConfig::default()).unwrap()
}

#[test]
fn test_compact_native_descriptor_funds_output_schema_after_plan_drop() {
    let runtime = runtime();
    let columns = (0..8)
        .map(|slot| format!("SUM(value) AS \"{}_{}\"", "wide_alias".repeat(4096), slot))
        .collect::<Vec<_>>()
        .join(", ");
    let query =
        crate::expression::parse_select_query(&format!("SELECT {columns} FROM events")).unwrap();
    let make = || {
        IncrementalSql::plan_sync(
            &runtime,
            &query,
            "events",
            schema(&DataType::Int64),
            schema(&DataType::Int64),
            "descriptor",
        )
        .unwrap()
        .unwrap()
    };
    let original = make();
    let descriptor = original.native_descriptor("descriptor").unwrap();
    let output_bytes = descriptor
        .output_schema
        .fields()
        .iter()
        .map(|field| field.size())
        .sum::<usize>();
    assert!(output_bytes > 1 << 18);
    drop(original);
    assert!(pool(&runtime).reserved() >= output_bytes);
    drop(descriptor);
    assert_eq!(pool(&runtime).reserved(), 0);

    let original = make();
    let basis = pool(&runtime).reserved();
    let pressure = MemoryConsumer::new("descriptor-pressure").register(&pool(&runtime));
    pressure
        .try_grow((1 << 30) - basis - (output_bytes - 1))
        .unwrap();
    assert!(original.native_descriptor("descriptor").is_err());
    drop(pressure);
    assert_eq!(pool(&runtime).reserved(), basis);
    drop(original);
    assert_eq!(pool(&runtime).reserved(), 0);
}

fn schema(kind: &DataType) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, true).with_metadata(std::collections::HashMap::from([
            ("role".into(), "group".into()),
        ])),
        Field::new("value", kind.clone(), true),
    ]))
}

fn plan(runtime: &DataFusionRuntime, kind: &DataType, grouped: bool) -> IncrementalSql {
    let decimal = matches!(
        kind,
        DataType::Decimal32(..)
            | DataType::Decimal64(..)
            | DataType::Decimal128(..)
            | DataType::Decimal256(..)
    );
    let text = format!(
        "SELECT {}SUM(value), COUNT(value), MIN(value), MAX(value), COUNT(*), COUNT(1){} FROM events{}",
        if grouped { "key, " } else { "" },
        if decimal { ", AVG(value)" } else { "" },
        if grouped { " GROUP BY key" } else { "" }
    );
    let query = crate::expression::parse_select_query(&text).unwrap();
    IncrementalSql::plan_sync(
        runtime,
        &query,
        "events",
        schema(kind),
        schema(kind),
        "codec",
    )
    .unwrap()
    .unwrap()
}

async fn apply(plan: &mut IncrementalSql, kind: &DataType, values: &[Option<i64>]) {
    let value = if values.is_empty() {
        new_empty_array(kind)
    } else {
        ScalarValue::iter_to_array(
            values
                .iter()
                .map(|value| ScalarValue::Int64(*value).cast_to(kind).unwrap()),
        )
        .unwrap()
    };
    let record = RecordBatch::try_new(
        schema(kind),
        vec![
            Arc::new(Int32Array::from(
                values
                    .iter()
                    .enumerate()
                    .map(|(slot, _)| {
                        if slot % 3 == 2 {
                            None
                        } else {
                            Some(i32::try_from(slot).unwrap())
                        }
                    })
                    .collect::<Vec<_>>(),
            )),
            value,
        ],
    )
    .unwrap();
    let input = Batch::table(vec![record], BatchMetadata::default()).unwrap();
    let job = StreamJobContext::new(1, "codec", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "codec", None);
    let transaction = plan.update(&input, &context, "codec").await.unwrap();
    plan.commit(transaction);
}

type Saved = Vec<(Vec<ScalarValue>, Vec<Vec<ScalarValue>>, Vec<ScalarValue>)>;

fn snapshot(plan: &IncrementalSql) -> Saved {
    plan.groups
        .iter()
        .map(|group| {
            (
                group.values.to_vec(),
                group.states.clone(),
                group.results.clone(),
            )
        })
        .collect()
}

fn result(plan: &IncrementalSql) -> RecordBatch {
    if plan.groups.is_empty() {
        return RecordBatch::try_new(
            plan.output_schema.clone(),
            plan.output_schema
                .fields()
                .iter()
                .map(|field| new_empty_array(field.data_type()))
                .collect(),
        )
        .unwrap();
    }
    plan.output_chunk(
        0,
        plan.groups.len(),
        &CandidateMap::with_hasher(RandomState::new()),
        "codec",
    )
    .unwrap()
}

fn pool(runtime: &DataFusionRuntime) -> Arc<dyn MemoryPool> {
    runtime
        .incremental_planner_context(0)
        .runtime_env()
        .memory_pool
        .clone()
}

fn replace(record: &RecordBatch, column: usize, array: ArrayRef) -> RecordBatch {
    let mut columns = record.columns().to_vec();
    columns[column] = array;
    RecordBatch::try_new(record.schema(), columns).unwrap()
}

fn new_empty_or_null(kind: &DataType, rows: usize) -> ArrayRef {
    datafusion::arrow::array::new_null_array(kind, rows)
}
