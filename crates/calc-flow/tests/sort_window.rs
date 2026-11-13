use calc_flow::{
    CalcFlowError, EventTime, LatePolicySpec, Result, SortHistory, SortSpec, SortWindow,
    StateBudget, canonical_json,
};
use serde_json::json;

#[test]
fn test_sort_window_flattens_to_the_original_wire() -> Result<()> {
    let window = SortWindow::new(
        "ts",
        EventTime::from_micros(-10),
        EventTime::from_micros(10),
    )?;
    assert_eq!(window.event_time_column(), "ts");
    assert_eq!(window.start().as_micros(), -10);
    assert_eq!(window.end().as_micros(), 10);
    for late in [
        LatePolicySpec::Error {
            scope: calc_flow::LateErrorScope::Envelope,
        },
        LatePolicySpec::Drop { metrics_version: 1 },
    ] {
        let spec = SortSpec::closed_window(window.clone(), StateBudget::new(1000, 8 << 20)?)?
            .with_offset(1)?
            .with_sequence_by(vec!["seq".into()])?
            .with_late_policy(late)?;
        let expected = json!({"history":{"kind":"closed_window","event_time_column":"ts","start_micros":-10,"end_micros":10},"ingress_policy":"copy_visible_v2","state_budget":{"max_rows":1000,"max_bytes":8_388_608},"offset":1,"ties":{"kind":"sequence","columns":["seq"]},"late_policy":late});
        assert_eq!(
            canonical_json(&serde_json::to_value(&spec).unwrap())?,
            canonical_json(&expected)?
        );
        assert_eq!(serde_json::from_value::<SortSpec>(expected).unwrap(), spec);
        assert!(matches!(
            spec.history(),
            SortHistory::ClosedWindow {
                start_micros: -10,
                end_micros: 10,
                ..
            }
        ));
    }
    assert_eq!(window, window.clone());
    Ok(())
}

#[test]
fn test_sort_window_boundaries_and_native_wire_error_paths() {
    for (column, start, end, path) in [
        ("", 0, 10, "sort_window.event_time_column"),
        ("ts", 10, 10, "sort_window.end"),
        ("ts", 11, 10, "sort_window.end"),
    ] {
        let error = SortWindow::new(
            column,
            EventTime::from_micros(start),
            EventTime::from_micros(end),
        )
        .unwrap_err();
        assert!(matches!(error,CalcFlowError::InvalidArgument {field,..} if field==path));
    }
    let window = SortWindow::new(
        "ts",
        EventTime::from_micros(i64::MIN),
        EventTime::from_micros(i64::MAX),
    )
    .unwrap();
    assert_eq!(window.start().as_micros(), i64::MIN);
    assert_eq!(window.end().as_micros(), i64::MAX);
    let wire = json!({"history":{"kind":"closed_window","event_time_column":"","start_micros":0,"end_micros":10},"ingress_policy":"copy_visible_v2","state_budget":{"max_rows":1,"max_bytes":8_388_608},"offset":0,"ties":{"kind":"unspecified"},"late_policy":{"kind":"error","scope":"envelope"}});
    let error = serde_json::from_value::<SortSpec>(wire)
        .unwrap_err()
        .to_string();
    assert!(error.contains("sort.spec.history"), "{error}");
    assert!(!error.contains("sort_window"), "{error}");
}
