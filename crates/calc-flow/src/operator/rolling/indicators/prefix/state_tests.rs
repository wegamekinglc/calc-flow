use super::*;
use crate::operator::rolling::{
    EntityRollingState, RollingOperator, RollingSpec, ScalarValue, WindowState,
};
use datafusion::arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use serde_json::json;
use std::sync::Arc;

fn operator() -> RollingOperator {
    let schema = Arc::new(Schema::new(vec![
        Field::new("entity", DataType::Utf8, false),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("sequence", DataType::UInt64, false),
        Field::new("price", DataType::Float64, false),
    ]));
    let spec: RollingSpec = serde_json::from_value(json!({
        "configuration_version":1,"state_layout_version":5,
        "partition_by":["entity"],"event_time":"time","sequence_by":["sequence"],
        "outputs":[
            {"kind":"rsi","primitive_version":1,"input":"price","output":"rsi","period":1},
            {"kind":"mean","primitive_version":1,"input":"price","output":"mean","frame":{"kind":"duration","micros":10},"min_periods":1}
        ],
        "allowed_lateness_micros":0,"late_policy":{"kind":"error","scope":"envelope"},
        "value_policy":"stateful_numeric_v1"
    })).unwrap();
    RollingOperator::new("prefix", schema, spec).unwrap()
}
fn row(entity: &str, time: i64) -> BufferedRow {
    BufferedRow::new(
        vec![Some(KeyValue::String(entity.into()))],
        vec![KeyValue::Unsigned(u64::try_from(time).unwrap())],
        time,
        vec![],
    )
}
fn source(compiled: &CompiledRollingSpec, rows: &[BufferedRow]) -> RollingStreamState {
    let mut state = RollingStreamState::default();
    let mut history = EntityRollingState::fresh(compiled);
    // The prefix simulator borrows only the already-validated event-time cell.
    history.rows.push_back(vec![
        ScalarValue::Utf8(Some("a".into())),
        ScalarValue::TimestampMicrosecond(Some(1), Some("UTC".into())),
        ScalarValue::UInt64(Some(1)),
        ScalarValue::Float64(Some(1.0)),
    ]);
    history.transition_count = 1;
    assert!(
        history
            .windows
            .iter()
            .any(|window| matches!(window, WindowState::Numeric(..)))
    );
    state
        .histories
        .by_entity
        .insert(vec![Some(KeyValue::String("a".into()))], history);
    state
        .buffer
        .extend(rows.iter().cloned().map(|row| (row.identity.clone(), row)));
    state
}

#[test]
fn test_stream_prefix_seeds_histories_and_counts_remaining_pending_rows() {
    let operator = operator();
    let rows = [row("b", 100), row("a", 101), row("c", 1_000)];
    let state = source(&operator.compiled, &rows);
    let budget = StateBudget::new(6, 1 << 20).unwrap();
    let quoted = stream_scratch(&state, &operator.compiled, 2, budget, "prefix", &mut || {
        Ok(())
    })
    .unwrap();
    let actual = allocation_counter::measure(|| {
        stream(
            &rows[..2],
            &state,
            &operator.compiled,
            budget,
            "prefix",
            &mut || Ok(()),
        )
        .unwrap();
    });
    assert!(
        quoted >= actual.bytes_max,
        "quote={quoted}, actual={actual:?}"
    );
    let error = stream(
        &rows[..2],
        &state,
        &operator.compiled,
        StateBudget::new(5, 1 << 20).unwrap(),
        "prefix",
        &mut || Ok(()),
    )
    .unwrap_err();
    assert!(error.to_string().contains("state_budget"));
    assert_eq!(state.buffer.len(), 3);
    assert_eq!(state.histories.by_entity.len(), 1);
    assert_eq!(
        state
            .histories
            .by_entity
            .values()
            .next()
            .unwrap()
            .rows
            .len(),
        1
    );
}

#[test]
fn test_stream_prefix_rejects_nonpending_selection_and_preserves_cancel() {
    let operator = operator();
    let rows = [row("b", 100), row("a", 101)];
    let state = source(&operator.compiled, &rows);
    let budget = StateBudget::new(6, 1 << 20).unwrap();
    assert!(
        stream(
            &[row("missing", 50)],
            &state,
            &operator.compiled,
            budget,
            "prefix",
            &mut || Ok(())
        )
        .is_err()
    );
    assert!(
        stream(
            &[rows[1].clone(), rows[0].clone()],
            &state,
            &operator.compiled,
            budget,
            "prefix",
            &mut || Ok(())
        )
        .is_err()
    );
    let error = stream_scratch(&state, &operator.compiled, 2, budget, "prefix", &mut || {
        Err(crate::CalcFlowError::Cancelled {
            run_id: "prefix".into(),
        })
    })
    .unwrap_err();
    assert!(matches!(error, crate::CalcFlowError::Cancelled { .. }));
}
