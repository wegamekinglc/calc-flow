mod ownership;

use super::*;
use crate::{Batch, BatchMetadata, EdgeBudget, EdgeCollector};
use datafusion::arrow::{
    array::{Array, ArrayRef, Int64Array, StringArray, UInt64Array},
    record_batch::RecordBatch,
};

fn input(records: &[&[Option<&str>]]) -> Batch {
    let input_schema = Arc::new(Schema::new_with_metadata(
        vec![Field::new("value", DataType::Utf8, true)],
        [("owner".into(), "upstream".into())].into(),
    ));
    Batch::table(
        records
            .iter()
            .map(|values| {
                RecordBatch::try_new(
                    input_schema.clone(),
                    vec![Arc::new(StringArray::from(values.to_vec())) as ArrayRef],
                )
                .unwrap()
            })
            .collect(),
        BatchMetadata::new("source", 99, [("private".into(), true.into())].into()).unwrap(),
    )
    .unwrap()
}

#[tokio::test]
async fn test_late_plan_preserves_raw_rows_diagnostics_and_envelope_order() {
    let batch = input(&[&[Some("newer"), Some("normal")], &[None, Some("older")]]);
    let original = batch.table_payload().unwrap().batches().to_vec();
    let mut plan = LateOutputPlan::new(
        &batch,
        "roll",
        20,
        EdgeBudget::default(),
        7,
        schema(batch.table_payload().unwrap().schema()),
    )
    .unwrap();
    plan.push(0, 0, 0, 10, 15).unwrap();
    plan.push(1, 0, 2, 9, 14).unwrap();
    plan.push(1, 1, 3, 8, 13).unwrap();
    let prepared = plan.prepare().unwrap();
    let mut output = EdgeCollector::new(vec![
        Port::with_schema_ref(
            "late",
            BatchKind::Table,
            true,
            Some(schema(batch.table_payload().unwrap().schema())),
        )
        .unwrap(),
    ]);
    assert_eq!(prepared.emit(&mut output).await.unwrap(), 8);
    let messages = output.drain("late");
    let batches = messages
        .iter()
        .map(|message| message.as_data().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].metadata().source(), "roll.late");
    assert_eq!(batches[0].metadata().sequence(), 7);
    assert!(batches[0].metadata().attributes().is_empty());
    let rows = batches[0].table_payload().unwrap().batches();
    for (record, (value, event, closing, index)) in rows.iter().zip([
        (Some("newer"), 10, 15, 0),
        (None, 9, 14, 2),
        (Some("older"), 8, 13, 3),
    ]) {
        assert_eq!(record.num_columns(), 10);
        assert_eq!(record.schema().metadata().get("owner").unwrap(), "upstream");
        let values = record
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(values.iter().next().unwrap(), value);
        for (column, expected) in [(1, "roll"), (2, "input"), (6, "late_row"), (7, "source")] {
            assert_eq!(
                record
                    .column(column)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .value(0),
                expected
            );
        }
        for (column, expected) in [(3, event), (4, closing), (5, 20)] {
            assert_eq!(
                record
                    .column(column)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .value(0),
                expected
            );
        }
        for (column, expected) in [(8, 99), (9, index)] {
            assert_eq!(
                record
                    .column(column)
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .unwrap()
                    .value(0),
                expected
            );
        }
        assert!(
            record.columns()[1..]
                .iter()
                .all(|column| column.null_count() == 0)
        );
    }
    assert_eq!(batch.table_payload().unwrap().batches(), original);
}

fn collector(batch: &Batch) -> EdgeCollector {
    EdgeCollector::new(vec![
        Port::with_schema_ref(
            "late",
            BatchKind::Table,
            true,
            Some(schema(batch.table_payload().unwrap().schema())),
        )
        .unwrap(),
    ])
}

#[tokio::test]
async fn test_late_plan_budget_chunks_exact_arrow_charges_with_nulls() {
    let batch = input(&[&[Some("0123456789"), None], &[Some("0123456789")]]);
    // One non-null row costs 93 bytes; the null row costs 84 (including bitmap).
    let budget = EdgeBudget::new(3, 177).unwrap();
    let mut plan = LateOutputPlan::new(
        &batch,
        "roll",
        20,
        budget,
        7,
        schema(batch.table_payload().unwrap().schema()),
    )
    .unwrap();
    for (record, row, index) in [(0, 0, 0), (0, 1, 1), (1, 0, 2)] {
        plan.push(record, row, index, 1, 1).unwrap();
        let (rows, bytes) = plan.scratch_usage();
        assert!(rows <= budget.max_rows && bytes <= budget.max_bytes);
    }
    let prepared = plan.prepare().unwrap();
    let mut output = collector(&batch);
    assert_eq!(prepared.emit(&mut output).await.unwrap(), 9);
    let messages = output.drain("late");
    let batches = messages
        .iter()
        .map(|message| message.as_data().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(batches.len(), 2);
    assert_eq!(batches[0].estimated_bytes().unwrap(), 177);
    assert_eq!(batches[1].estimated_bytes().unwrap(), 93);
    assert_eq!(batches[1].metadata().sequence(), 8);
}

#[test]
fn test_late_plan_budget_preflights_later_oversize_row_and_exact_one_byte_limit() {
    let wide = "x".repeat(1024);
    let batch = input(&[&[Some(""), Some(&wide)]]);
    let mut plan = LateOutputPlan::new(
        &batch,
        "roll",
        20,
        EdgeBudget::new(2, 200).unwrap(),
        0,
        schema(batch.table_payload().unwrap().schema()),
    )
    .unwrap();
    plan.push(0, 0, 0, 1, 1).unwrap();
    plan.push(0, 1, 1, 1, 1).unwrap();
    let error = plan
        .prepare()
        .err()
        .expect("the later row must fail before emission");
    assert!(
        error.to_string().contains("output_row_too_large"),
        "{error}"
    );
    assert!(error.to_string().contains("row_index=1"), "{error}");

    let batch = input(&[&[Some("0123456789")]]);
    for (bytes, accepted) in [(93, true), (92, false)] {
        let mut plan = LateOutputPlan::new(
            &batch,
            "roll",
            20,
            EdgeBudget::new(1, bytes).unwrap(),
            0,
            schema(batch.table_payload().unwrap().schema()),
        )
        .unwrap();
        plan.push(0, 0, 0, 1, 1).unwrap();
        assert_eq!(plan.prepare().is_ok(), accepted, "max_bytes={bytes}");
    }
}

#[test]
fn test_late_plan_budget_bounds_actual_scratch_allocation() {
    let batch = input(&[&[Some(""), Some("")]]);
    for budget in [
        EdgeBudget::new(1, 1000).unwrap(),
        EdgeBudget::new(2, 2 * LateOutputPlan::scratch_row_bytes() - 1).unwrap(),
    ] {
        let mut plan = LateOutputPlan::new(
            &batch,
            "roll",
            20,
            budget,
            0,
            schema(batch.table_payload().unwrap().schema()),
        )
        .unwrap();
        plan.push(0, 0, 0, 1, 1).unwrap();
        let before = plan.scratch_usage();
        let error = plan.push(0, 1, 1, 1, 1).unwrap_err();
        assert!(error.to_string().contains("late scratch"), "{error}");
        assert_eq!(plan.scratch_usage(), before);
        assert!(before.0 <= budget.max_rows && before.1 <= budget.max_bytes);
    }
}

#[test]
fn test_late_preflight_rejects_wide_diagnostics_before_allocating_their_values() {
    let original = input(&[&[Some("")]]);
    let batch = Batch::table(
        original.table_payload().unwrap().batches().to_vec(),
        BatchMetadata::new("x".repeat(2 * 1024 * 1024), 0, crate::JsonMap::new()).unwrap(),
    )
    .unwrap();
    let mut plan = LateOutputPlan::new(
        &batch,
        "roll",
        20,
        EdgeBudget::new(1, 200).unwrap(),
        0,
        schema(batch.table_payload().unwrap().schema()),
    )
    .unwrap();
    plan.push(0, 0, 0, 1, 1).unwrap();
    let allocations = allocation_counter::measure(|| {
        let error = plan.prepare().err().expect("oversize diagnostic must fail");
        assert!(
            error.to_string().contains("output_row_too_large"),
            "{error}"
        );
    });
    assert!(
        allocations.bytes_max < 64 * 1024,
        "oversize metadata was copied: {} bytes",
        allocations.bytes_max
    );
}
