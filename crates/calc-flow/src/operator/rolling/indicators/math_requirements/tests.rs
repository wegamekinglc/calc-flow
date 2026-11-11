use std::sync::Arc;

use datafusion::arrow::datatypes::{Field, Schema, TimeUnit};
use serde_json::json;

use super::*;
use crate::operator::rolling::{BufferedRow, RollingHistories, RollingOperator, RollingSpec};

fn operator() -> RollingOperator {
    let schema = Schema::new(vec![
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC"))),
            false,
        ),
        Field::new("entity", DataType::Utf8, false),
        Field::new("seq", DataType::UInt64, false),
        Field::new("price", DataType::Float64, true),
        Field::new("label", DataType::Utf8, false),
    ]);
    let spec: RollingSpec = serde_json::from_value(json!({
        "configuration_version":1,"state_layout_version":5,
        "partition_by":["entity"],"event_time":"ts","sequence_by":["seq"],
        "outputs":[
            {"kind":"rsi","primitive_version":1,"input":"price","output":"rsi","period":2},
            {"kind":"lag","primitive_version":1,"input":"label","output":"lag","periods":1},
            {"kind":"min","primitive_version":1,"input":"label","output":"minimum","frame":{"kind":"rows","size":128},"min_periods":1},
            {"kind":"argmax","primitive_version":1,"input":"price","output":"argmax","frame":{"kind":"rows","size":128},"min_periods":1},
            {"kind":"unique_count","primitive_version":1,"input":"price","output":"unique_price","frame":{"kind":"rows","size":128},"min_periods":1},
            {"kind":"unique_count","primitive_version":1,"input":"label","output":"unique_label","frame":{"kind":"duration","micros":128},"min_periods":1}
        ],
        "allowed_lateness_micros":0,"late_policy":{"kind":"error","scope":"envelope"},
        "value_policy":"stateful_numeric_v1"
    })).unwrap();
    RollingOperator::new("features", Arc::new(schema), spec).unwrap()
}
fn rows(start: u32, count: u32, entities: u32) -> Vec<BufferedRow> {
    (start..start + count)
        .map(|index| {
            let entity = format!("entity-{}", index % entities);
            BufferedRow::new(
                vec![Some(KeyValue::String(entity.clone()))],
                vec![KeyValue::Unsigned(u64::from(index))],
                i64::from(index),
                vec![
                    ScalarValue::TimestampMicrosecond(
                        Some(i64::from(index)),
                        Some(Arc::from("UTC")),
                    ),
                    ScalarValue::Utf8(Some(entity)),
                    ScalarValue::UInt64(Some(u64::from(index))),
                    ScalarValue::Float64(Some(f64::from(index % 17))),
                    ScalarValue::Utf8(Some(format!("{:05}{}", index, "x".repeat(4097)))),
                ],
            )
        })
        .collect()
}
fn state(
    operator: &RollingOperator,
    closing: &[BufferedRow],
    prior: &[BufferedRow],
) -> RollingStreamState {
    let computed = super::super::compute_cells(
        prior,
        &RollingHistories::default(),
        &operator.compiled,
        "features",
        None,
        None,
        &mut || Ok(()),
    )
    .unwrap();
    let mut histories = RollingHistories::default();
    histories.apply(computed.touched);
    RollingStreamState {
        buffer: closing
            .iter()
            .map(|row| (row.identity.clone(), row.clone()))
            .collect(),
        histories,
        ..RollingStreamState::default()
    }
}

#[test]
fn math_quote_covers_real_cells_with_scans_strings_and_old_histories() {
    let operator = operator();
    for (entities, prior_count, count) in [(1, 0, 129), (1, 17, 65), (17, 17, 129)] {
        let closing = rows(prior_count, count, entities);
        let prior = rows(0, prior_count, entities);
        let source = state(&operator, &closing, &prior);
        let mut quote = 0;
        let planning = allocation_counter::measure(|| {
            quote = plan(&source, &operator.compiled, closing.len(), &mut || Ok(())).unwrap();
        });
        if prior_count == 0 {
            assert_eq!(
                quote,
                plan_finite(&closing, &operator.compiled, &mut || Ok(())).unwrap()
            );
        }
        assert_eq!(
            planning.bytes_max, 0,
            "planning must precede allocation admission"
        );
        let mut cells = None;
        let observed = allocation_counter::measure(|| {
            cells = Some(
                super::super::compute_cells(
                    &closing,
                    &source.histories,
                    &operator.compiled,
                    "features",
                    None,
                    None,
                    &mut || Ok(()),
                )
                .unwrap(),
            );
        });
        assert!(
            observed.bytes_max <= quote,
            "entities={entities}, old={prior_count}, rows={count}, {observed:?}, quote={quote}"
        );
        assert_eq!(
            cells.unwrap().columns.len(),
            operator.compiled.outputs.len()
        );
        assert_eq!(source.buffer.len(), closing.len());
    }
}

#[test]
fn math_planning_preserves_cancellation_identity() {
    let operator = operator();
    let closing = rows(0, 1, 1);
    let source = state(&operator, &closing, &[]);
    let result = plan(&source, &operator.compiled, 1, &mut || {
        Err(crate::CalcFlowError::Cancelled {
            run_id: "math-planning".into(),
        })
    });
    assert!(
        matches!(result, Err(crate::CalcFlowError::Cancelled { run_id }) if run_id == "math-planning")
    );
}

#[test]
fn empty_cells_quote_still_charges_derived_container() {
    let operator = operator();
    let source = RollingStreamState::default();
    let quote = plan(&source, &operator.compiled, 0, &mut || Ok(())).unwrap();
    assert_eq!(
        quote,
        plan_finite(&[], &operator.compiled, &mut || Ok(())).unwrap()
    );
    let mut cells = None;
    let observed = allocation_counter::measure(|| {
        cells = Some(
            super::super::compute_cells(
                &[],
                &source.histories,
                &operator.compiled,
                "features",
                None,
                None,
                &mut || Ok(()),
            )
            .unwrap(),
        );
    });
    assert_eq!(observed.bytes_max, quote);
    assert_eq!(
        cells.unwrap().columns.len(),
        operator.compiled.outputs.len()
    );
}
