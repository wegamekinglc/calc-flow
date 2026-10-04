use super::*;
use crate::operator::sql::incremental::IncrementalSql;

#[path = "grouped_checkpoint_reuse_tests.rs"]
mod checkpoint_reuse;

fn query(composite: bool, predicate: &str) -> String {
    let keys = if composite { "key, other" } else { "key" };
    format!("SELECT {keys}, {AGGREGATES} FROM events WHERE {predicate} GROUP BY {keys}")
}

#[tokio::test]
async fn test_grouped_where_exact_native_prefix_and_cold_continuation() {
    for dtype in [DataType::Float32, DataType::Float64] {
        for key_type in [DataType::Int64, DataType::Utf8, DataType::LargeUtf8] {
            for composite in [false, true] {
                case(
                    &dtype,
                    &key_type,
                    &query(composite, "(accepted OR other) AND value >= 0"),
                    if composite { 2 } else { 1 },
                )
                .await;
            }
        }
    }
}

#[tokio::test]
async fn test_grouped_where_all_rejected_prefix_remains_native_and_restores() {
    for key_type in [DataType::Int64, DataType::Utf8, DataType::LargeUtf8] {
        for composite in [false, true] {
            case(
                &DataType::Float64,
                &key_type,
                &query(composite, "accepted AND other"),
                if composite { 2 } else { 1 },
            )
            .await;
        }
    }
}

#[tokio::test]
async fn test_grouped_where_counts_exact_native_prefix_and_cold_continuation() {
    let text = COUNTS.replace(
        "FROM events GROUP BY",
        "FROM events WHERE NOT accepted GROUP BY",
    );
    case(&DataType::Float64, &DataType::Int64, &text, 1).await;
}

#[tokio::test]
async fn test_grouped_where_refusal_refunds_then_retries_exactly() {
    refusal(
        DataType::Float64,
        DataType::Utf8,
        query(true, "(accepted OR other) AND value >= 0"),
    )
    .await;
    refusal(
        DataType::Float64,
        DataType::Int64,
        COUNTS.replace(
            "FROM events GROUP BY",
            "FROM events WHERE NOT accepted GROUP BY",
        ),
    )
    .await;
}

#[tokio::test]
async fn test_grouped_where_unsupported_and_nondefault_remain_retained() {
    use crate::UdfRegistrySnapshot;

    for (predicate, config) in [
        ("value + 1 >= 0", DataFusionConfig::default()),
        (
            "accepted",
            DataFusionConfig {
                target_partitions: 4,
                ..DataFusionConfig::default()
            },
        ),
    ] {
        let text = query(true, predicate);
        let dtype = DataType::Float64;
        let key_type = DataType::Utf8;
        let job = job();
        let context = StreamOperatorContext::new(&job, "grouped_float", None);
        let mut state = filtered_operator(&dtype, &key_type, &text);
        state.set_stream_resources(config, UdfRegistrySnapshot::default(), vec![]);
        let parts = string_arrivals().remove(0);
        let actual = process(
            &mut state,
            filtered_input(&dtype, &key_type, &parts, 0),
            &context,
        )
        .await;
        oracle(&actual, &text, &dtype, &key_type, &[parts], 2).await;
        assert!(state.incremental.is_none() && state.compact.is_none() && state.retained.is_some());
        assert_eq!(
            state.checkpoint(Epoch::INITIAL).unwrap().inline_metadata["state_layout"],
            json!(4)
        );
        let pool = state
            .stream_state
            .runtime()
            .unwrap()
            .incremental_memory_pool();
        drop((state, actual));
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop(context);
        drop(job);
        assert_eq!(pool.reserved(), 0);
    }
}

#[tokio::test]
async fn test_grouped_where_restore_rejects_zero_unfiltered_group_count() {
    let text = COUNTS.replace(
        "FROM events GROUP BY",
        "FROM events WHERE NOT accepted GROUP BY",
    );
    let dtype = DataType::Float64;
    let key_type = DataType::Int64;
    let job = job();
    let context = StreamOperatorContext::new(&job, "grouped_float", None);
    let mut state = filtered_operator(&dtype, &key_type, &text);
    let parts = string_arrivals().remove(0);
    drop(
        process(
            &mut state,
            filtered_input(&dtype, &key_type, &parts, 0),
            &context,
        )
        .await,
    );
    let before = capture(&mut state);
    let exported = state
        .incremental
        .as_ref()
        .unwrap()
        .export_native_state("grouped_float", || Ok(()))
        .unwrap();
    let records = exported
        .records()
        .iter()
        .map(|record| {
            let mut columns = record.columns().to_vec();
            let last = columns.len() - 1;
            columns[last] = Arc::new(Int64Array::from(vec![0; record.num_rows()]));
            RecordBatch::try_new(record.schema(), columns).unwrap()
        })
        .collect::<Vec<_>>();
    let runtime = state.stream_state.runtime().unwrap();
    let pool = runtime.incremental_memory_pool();
    let basis = pool.reserved();
    let fresh = IncrementalSql::plan(
        runtime,
        &parse_select_query(&text).unwrap(),
        "events",
        filtered_schema(&dtype, &key_type),
        "grouped_float",
    )
    .await
    .unwrap()
    .unwrap();
    let restored = fresh.import_native_state(
        &records,
        parts.iter().map(Vec::len).sum::<usize>() as u64,
        true,
        || Ok(()),
        "grouped_float",
    );
    assert!(
        matches!(restored, Err(CalcFlowError::DataFusion {message, ..}) if message == "native grouped all-row COUNT is zero")
    );
    assert_eq!(pool.reserved(), basis);
    same_snapshot(&before, &state.checkpoint(Epoch::INITIAL).unwrap());
    drop((records, exported, before, state));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}
