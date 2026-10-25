use super::{recovery::*, *};
mod respect_nulls;
use crate::StateSegment;
use datafusion::arrow::{
    array::BooleanArray,
    compute::concat_batches,
    datatypes::{DataType, Field, Schema, TimeUnit},
};

fn replace_control(snapshot: &mut OperatorStateSnapshot, value: &Value) {
    let segment = StateSegment::new(serde_json::to_vec(value).unwrap());
    snapshot
        .inline_metadata
        .insert("control_sha256".into(), json!(segment.sha256()));
    snapshot.segments.insert("control".into(), segment);
}

fn assert_refused(
    op: &mut SqlOperator,
    bad: &OperatorStateSnapshot,
    before: &OperatorStateSnapshot,
    label: &str,
) -> String {
    let pool = op.retention_runtime().unwrap().incremental_memory_pool();
    let basis = pool.reserved();
    let native = std::ptr::from_ref(op.incremental.as_ref().unwrap().as_ref());
    let error = StreamOperator::restore(op, bad).expect_err(label);
    assert!(!error.to_string().is_empty());
    assert_eq!(
        native,
        std::ptr::from_ref(op.incremental.as_ref().unwrap().as_ref()),
        "{label}"
    );
    assert_eq!(pool.reserved(), basis, "{label}: {error}");
    same_snapshot(before, &op.checkpoint(Epoch::INITIAL).unwrap());
    error.to_string()
}

#[tokio::test]
async fn test_ordered_last_layout_five_control_corruption_is_atomic_with_valid_control_digest() {
    let job = job();
    let ctx = StreamOperatorContext::new(&job, "ordered_last", None);
    let mut op = seeded(&ctx).await;
    let before = op.checkpoint(Epoch::INITIAL).unwrap();
    for (pointer, replacement) in [
        ("/state_layout", json!(3)),
        ("/state_accounting", json!(99)),
        ("/native_semantics", json!(99)),
        ("/datafusion_version", json!("future")),
        ("/state_policy", json!("unknown")),
        ("/state_policy", json!("exact_numeric_v1")),
        ("/identity/query_sha256", json!("0".repeat(64))),
        ("/identity/input_alias", json!("other")),
        ("/identity/physical_schema_sha256", json!("0".repeat(64))),
        ("/identity/state_schema_sha256", json!("0".repeat(64))),
        ("/identity/output_schema_sha256", json!("0".repeat(64))),
        ("/identity/retained_ordinals", json!([0])),
        ("/identity/native_descriptor/ordered_last/version", json!(2)),
        (
            "/identity/native_descriptor/ordered_last/model",
            json!("parallel"),
        ),
        (
            "/identity/native_descriptor/ordered_last/tie",
            json!("last"),
        ),
        (
            "/identity/native_descriptor/ordered_last/group_slots",
            json!("sorted"),
        ),
        (
            "/identity/native_descriptor/ordered_last/ignore_nulls",
            json!(false),
        ),
        (
            "/identity/native_descriptor/ordered_last/order/keys/0/descending",
            json!(true),
        ),
        (
            "/identity/native_descriptor/ordered_last/order/keys/0/nulls_first",
            json!(false),
        ),
        (
            "/identity/native_descriptor/ordered_last/order/keys/0/input/index",
            json!(3),
        ),
        (
            "/identity/native_descriptor/ordered_last/order/keys/0/input/field/0/schema_sha256",
            json!("0".repeat(64)),
        ),
        ("/group_log/version", json!(2)),
        ("/group_log/base_generation", json!(1)),
        ("/group_log/generation", json!(99)),
        ("/segments/group_state", json!("0".repeat(64))),
    ] {
        let mut bad = detached(&before);
        let mut value = control(&bad);
        *value.pointer_mut(pointer).expect(pointer) = replacement;
        replace_control(&mut bad, &value);
        assert_refused(&mut op, &bad, &before, pointer);
    }
    for field in [
        "identity",
        "native_semantics",
        "state_policy",
        "group_log",
        "ledger",
    ] {
        let mut bad = detached(&before);
        let mut value = control(&bad);
        value.as_object_mut().unwrap().remove(field);
        replace_control(&mut bad, &value);
        assert_refused(&mut op, &bad, &before, field);
    }
}

fn changed_record(record: &RecordBatch, case: &str) -> RecordBatch {
    let mut columns = record.columns().to_vec();
    let mut fields = record
        .schema()
        .fields()
        .iter()
        .map(|f| f.as_ref().clone())
        .collect::<Vec<_>>();
    let last = columns.len() - 1;
    match case {
        "unset" => columns[last] = Arc::new(BooleanArray::from(vec![false; record.num_rows()])),
        "null-winner" => columns[1] = Arc::new(Float64Array::from(vec![None; record.num_rows()])),
        "null-flag" => {
            columns[last] = Arc::new(BooleanArray::from(vec![None; record.num_rows()]));
            fields[last] = fields[last].clone().with_nullable(true);
        }
        "unit" => {
            fields[2] = Field::new(
                fields[2].name(),
                DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())),
                true,
            );
        }
        "timezone" => {
            fields[2] = Field::new(
                fields[2].name(),
                DataType::Timestamp(TimeUnit::Microsecond, None),
                true,
            );
        }
        _ => panic!("unknown corruption"),
    }
    if matches!(case, "unit" | "timezone") {
        columns[2] = datafusion::arrow::compute::cast(&columns[2], fields[2].data_type()).unwrap();
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
}

fn replace_wire(snapshot: &mut OperatorStateSnapshot, id: &str, records: Vec<RecordBatch>) {
    let batch = Batch::table(records, BatchMetadata::default()).unwrap();
    let segment = StateSegment::new(encode_sql_state(&batch).unwrap());
    let mut value = control(snapshot);
    if id == "group-state" {
        value["segments"]["group_state"] = json!(segment.sha256());
        value["group_log"]["base_groups"] = json!(batch.num_rows());
        value["groups"] = json!(batch.num_rows());
    } else {
        value["group_log"]["frames"][0]["sha256"] = json!(segment.sha256());
        value["group_log"]["frames"][0]["groups"] = json!(batch.num_rows());
    }
    snapshot.segments.insert(id.into(), segment);
    replace_control(snapshot, &value);
}

fn duplicate_wire(snapshot: &OperatorStateSnapshot, id: &str) -> Vec<RecordBatch> {
    let batch = decode_sql_state(snapshot.segments[id].bytes()).unwrap();
    let record = &batch.table_payload().unwrap().batches()[0];
    let first = record.slice(0, 1);
    vec![concat_batches(&record.schema(), [record, &first]).unwrap()]
}

#[tokio::test]
async fn test_ordered_last_full_and_delta_reject_bad_winners_types_and_duplicate_keys() {
    let job = job();
    let ctx = StreamOperatorContext::new(&job, "ordered_last", None);
    let mut op = seeded(&ctx).await;
    // Include repeated input so a duplicate-key wire passes the independent input census.
    drop(process(&mut op, input(&SEED), &ctx).await);
    let full = op.checkpoint(Epoch::INITIAL).unwrap();
    let rows = [(Some("A"), Some(99.0), Some(9), Some(0)); 2];
    drop(process(&mut op, input(&rows), &ctx).await);
    let before = op.checkpoint(Epoch::INITIAL).unwrap();
    let id = delta_ids(&before).remove(0);
    for (snapshot, id) in [(&full, "group-state"), (&before, id.as_str())] {
        for case in ["unset", "null-winner", "null-flag", "unit", "timezone"] {
            let mut bad = detached(snapshot);
            let batch = decode_sql_state(bad.segments[id].bytes()).unwrap();
            let record = changed_record(&batch.table_payload().unwrap().batches()[0], case);
            replace_wire(&mut bad, id, vec![record]);
            assert_refused(&mut op, &bad, &before, case);
        }
        let mut bad = detached(snapshot);
        replace_wire(&mut bad, id, duplicate_wire(snapshot, id));
        assert!(assert_refused(&mut op, &bad, &before, "duplicate key").contains("duplicate keys"));
    }
}

#[tokio::test]
async fn test_ordered_last_delta_inventory_and_generation_corruption_fail_closed() {
    let job = job();
    let ctx = StreamOperatorContext::new(&job, "ordered_last", None);
    let mut op = seeded(&ctx).await;
    drop(op.checkpoint(Epoch::INITIAL).unwrap());
    for n in 1..=2 {
        drop(
            process(
                &mut op,
                input(&[(Some("A"), Some(20.0), Some(n), Some(0))]),
                &ctx,
            )
            .await,
        );
        drop(op.checkpoint(Epoch::INITIAL).unwrap());
    }
    let before = op.checkpoint(Epoch::INITIAL).unwrap();
    for (pointer, replacement) in [
        ("/group_log/frames/0/id", json!("group-delta-unknown")),
        ("/group_log/frames/0/sha256", json!("0".repeat(64))),
        ("/group_log/frames/1/ledger/rows", json!(8)),
        ("/group_log/base_ledger/rows", json!(1)),
        ("/group_log/generation", json!(100)),
    ] {
        let mut bad = detached(&before);
        let mut value = control(&bad);
        *value.pointer_mut(pointer).unwrap() = replacement;
        replace_control(&mut bad, &value);
        assert_refused(&mut op, &bad, &before, pointer);
    }
    for id in before.segments.keys() {
        let mut bad = detached(&before);
        bad.segments.remove(id);
        assert_refused(&mut op, &bad, &before, id);
    }
    let mut bad = detached(&before);
    bad.segments
        .insert("group-delta-unlisted".into(), StateSegment::new(vec![]));
    assert_refused(&mut op, &bad, &before, "unlisted segment");
    // Version rejection precedes even decoding a broken control payload.
    bad.inline_metadata.insert("state_layout".into(), json!(99));
    bad.segments
        .insert("control".into(), StateSegment::new(vec![0xff]));
    assert!(
        StreamOperator::restore(&mut op, &bad)
            .unwrap_err()
            .to_string()
            .contains("layout is unsupported")
    );
    same_snapshot(&before, &op.checkpoint(Epoch::INITIAL).unwrap());
}
