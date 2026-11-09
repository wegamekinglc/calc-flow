use super::*;
use serde_json::{Map, json};

fn fields() -> JsonMap {
    let mut metrics = Map::with_capacity(4096);
    for (key, value) in [
        ("late_rows", json!(0)),
        ("affected_batches", json!(0)),
        ("max_lateness_micros", Value::Null),
        ("null_event_time_rows", json!(0)),
        ("null_event_time_batches", json!(0)),
    ] {
        metrics.insert(key.into(), value);
    }
    [
        ("state_layout_version".into(), json!(5)),
        ("configuration_hash".into(), json!("f".repeat(64))),
        ("state_schema_fingerprint".into(), json!("f".repeat(64))),
        ("epoch".into(), json!(1)),
        ("pipeline_fingerprint".into(), json!("f".repeat(64))),
        ("operator_id".into(), json!("features")),
        ("last_input_watermark".into(), Value::Null),
        ("next_output_sequence".into(), json!(0)),
        ("ended".into(), json!(false)),
        ("metrics".into(), Value::Object(metrics)),
        ("segment_inventory".into(), json!([])),
    ]
    .into_iter()
    .collect()
}

#[test]
fn test_controlled_inline_copy_uses_visible_length_and_preserves_json_order() {
    let fields = fields();
    let stock = allocation_counter::measure(|| drop(fields.clone()));
    assert!(
        stock.bytes_max > 64 << 10,
        "pinned stock clone control: {stock:?}"
    );
    let mut quoted = None;
    let planning = allocation_counter::measure(|| {
        quoted = Some(requirement(&fields, &mut || Ok(())).unwrap());
    });
    assert_eq!(planning.count_total, 0);
    let quoted = quoted.unwrap();
    assert!(quoted < 64 << 10);
    let mut copied = None;
    let actual = allocation_counter::measure(|| {
        copied = Some(copy(&fields, &mut || Ok(())).unwrap());
    });
    assert!(
        actual.bytes_max <= quoted,
        "quote={quoted} actual={actual:?}"
    );
    let copied = copied.unwrap();
    assert_eq!(copied, fields);
    assert_eq!(
        copied["metrics"]
            .as_object()
            .unwrap()
            .keys()
            .collect::<Vec<_>>(),
        fields["metrics"]
            .as_object()
            .unwrap()
            .keys()
            .collect::<Vec<_>>()
    );
}

#[test]
fn test_inline_copy_planning_and_construction_preserve_cancellation() {
    let fields = fields();
    for plan_only in [false, true] {
        let mut check = || {
            Err(CalcFlowError::Cancelled {
                run_id: "inline".into(),
            })
        };
        let result = if plan_only {
            requirement(&fields, &mut check).map(|_| ())
        } else {
            copy(&fields, &mut check).map(|_| ())
        };
        assert!(matches!(result, Err(CalcFlowError::Cancelled { .. })));
    }
}

#[test]
fn test_inline_copy_allocation_guards_reject_unrepresentable_single_allocations() {
    assert!(allocation(usize::MAX, Layout::new::<Value>()).is_err());
    assert!(indices(usize::MAX).is_err());
    assert_eq!(object_containers(0).unwrap(), 0);
}

#[test]
fn test_inline_copy_quote_covers_nonempty_inventory_and_handle_objects() {
    let mut fields = fields();
    fields.insert("__calc_flow_operator_state_version".into(), json!(5));
    fields.insert(
        "segment_inventory".into(),
        json!([{
            "kind":"base", "state_layout_version":5,
            "schema_fingerprint":"f".repeat(64),
            "handle":{"operator_id":"features","epoch":1,
              "segment_id":"base-00000000000000000001-00000000",
              "relative_path":"committed/features/base.arrow", "byte_len":u64::MAX,
              "sha256":"f".repeat(64)}
        }]),
    );
    let quote = requirement(&fields, &mut || Ok(())).unwrap();
    let mut copied = None;
    let actual = allocation_counter::measure(|| {
        copied = Some(copy(&fields, &mut || Ok(())).unwrap());
    });
    assert!(actual.bytes_max <= quote, "quote={quote} actual={actual:?}");
    assert_eq!(copied.unwrap(), fields);
}

#[test]
fn test_segment_container_quote_excludes_already_funded_payload() {
    let segment = crate::StateSegment::new(vec![7; 1 << 20]);
    let pointer = segment.bytes().as_ptr();
    let quote = segment_requirement(Some("base")).unwrap();
    assert!(quote < 4096);
    let mut copied = None;
    let actual = allocation_counter::measure(|| {
        copied = Some(BTreeMap::from([("base".to_owned(), segment)]));
    });
    assert!(actual.bytes_max <= quote, "quote={quote} actual={actual:?}");
    assert_eq!(copied.unwrap()["base"].bytes().as_ptr(), pointer);
    assert_eq!(segment_requirement(None).unwrap(), 0);
}
