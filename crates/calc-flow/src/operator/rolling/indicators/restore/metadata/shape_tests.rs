use super::*;
use serde_json::json;
#[test]
fn test_restore_metadata_shape_counts_borrowed_strings_without_allocating() {
    let fields = json!({"state_layout_version":5,"configuration_hash":"configuration","state_schema_fingerprint":"schema","epoch":1,"pipeline_fingerprint":null,"operator_id":null,"last_input_watermark":null,"next_output_sequence":0,"ended":false,"metrics":{"late_rows":0,"affected_batches":0,"max_lateness_micros":null,"null_event_time_rows":0,"null_event_time_batches":0},"segment_inventory":[]});
    let fields = fields
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let mut quote = None;
    let actual = allocation_counter::measure(|| {
        quote = Some(plan(&fields, &mut || Ok(())).unwrap());
    });
    assert_eq!(actual.count_total, 0);
    assert_eq!(quote.unwrap().strings, 19);
}
#[test]
fn test_restore_metadata_shape_rejects_unknown_or_wrong_typed_fields_safely() {
    let fields = [("unknown".to_owned(), json!("sensitive".repeat(1024)))]
        .into_iter()
        .collect();
    let error = plan(&fields, &mut || Ok(())).err().unwrap();
    assert!(!error.to_string().contains("sensitive"));
    let fields = [("state_layout_version".to_owned(), json!("sensitive"))]
        .into_iter()
        .collect();
    assert!(plan(&fields, &mut || Ok(())).is_err());
    let error = plan(&fields, &mut || {
        Err(CalcFlowError::Cancelled {
            run_id: "restore".into(),
        })
    })
    .err()
    .unwrap();
    assert!(matches!(error, CalcFlowError::Cancelled { .. }));
}

#[test]
fn test_restore_metadata_shape_counts_expected_segment_id_copy_separately() {
    let fields = json!({"segment_inventory":[{"kind":"base","state_layout_version":5,"schema_fingerprint":"schema","handle":{"operator_id":"features","epoch":1,"segment_id":"seg","relative_path":"committed/features/seg.arrow","byte_len":4,"sha256":"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"}}]});
    let fields = fields
        .as_object()
        .unwrap()
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    let facts = plan(&fields, &mut || Ok(())).unwrap();
    assert_eq!(facts.segment_ids, 3);
    assert_eq!(facts.descriptors, 1);
}

#[test]
fn test_restore_metadata_shape_counts_path_validation_vector() {
    let fields = json!({"segment_inventory":[{"kind":"base","state_layout_version":5,"schema_fingerprint":"schema","handle":{"operator_id":"features","epoch":1,"segment_id":"seg","relative_path":"committed/features/seg.arrow","byte_len":4,"sha256":"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"}}]});
    let fields = fields
        .as_object()
        .unwrap()
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    let facts = plan(&fields, &mut || Ok(())).unwrap();
    assert_eq!(facts.path_components, 3);
}
