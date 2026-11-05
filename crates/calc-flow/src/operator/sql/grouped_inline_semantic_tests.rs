use super::*;
use crate::operator::{OperatorMetadata, StreamCollector, StreamOperator};
use crate::{EdgeCollector, Epoch, SqlOperator};
use datafusion::arrow::ipc::writer::FileWriter;

const SQL: &str =
    "SELECT key, SUM(value) FILTER (WHERE selected) AS total FROM events WHERE keep GROUP BY key";
const HISTORICAL: usize = CHUNK_ROWS + 1;

fn scalar(unsigned: bool, value: Option<u64>) -> ScalarValue {
    if unsigned {
        ScalarValue::UInt64(value)
    } else {
        ScalarValue::Int64(value.map(|bits| i64::from_ne_bytes(bits.to_ne_bytes())))
    }
}

fn boundary(unsigned: bool) -> u64 {
    if unsigned { u64::MAX } else { u64::MAX >> 1 }
}

fn input(unsigned: bool, phase: usize) -> Batch {
    let rows = phase_rows(unsigned, phase);
    let values = if rows.is_empty() {
        datafusion::arrow::array::new_empty_array(&scalar(unsigned, None).data_type())
    } else {
        ScalarValue::iter_to_array(rows.iter().map(|(_, value, _, _)| scalar(unsigned, *value)))
            .unwrap()
    };
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Utf8, false),
        Field::new("value", values.data_type().clone(), true),
        Field::new("selected", DataType::Boolean, false),
        Field::new("keep", DataType::Boolean, false),
    ]));
    let record = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|(key, ..)| format!("account-{key}")),
            )) as ArrayRef,
            values,
            Arc::new(BooleanArray::from(
                rows.iter()
                    .map(|(_, _, selected, _)| *selected)
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(BooleanArray::from(
                rows.iter().map(|(_, _, _, keep)| *keep).collect::<Vec<_>>(),
            )) as ArrayRef,
        ],
    )
    .unwrap();
    Batch::table(vec![record], crate::BatchMetadata::default()).unwrap()
}

fn phase_rows(unsigned: bool, phase: usize) -> Vec<(usize, Option<u64>, bool, bool)> {
    match phase {
        0 | 2 => (0..HISTORICAL).map(|key| (key, None, true, true)).collect(),
        1 => vec![
            (0, Some(boundary(unsigned)), true, true),
            (HISTORICAL, Some(99), false, true),
            (HISTORICAL + 1, Some(99), true, false),
        ],
        3 => vec![],
        _ => vec![
            (0, Some(1), true, true),
            (HISTORICAL - 1, Some(2), true, true),
            (HISTORICAL, None, true, true),
        ],
    }
}

async fn plan_for(runtime: &DataFusionRuntime, batch: &Batch, sql: &str) -> IncrementalSql {
    let query = crate::expression::parse_select_query(sql).unwrap();
    IncrementalSql::plan(
        runtime,
        &query,
        "events",
        batch.table_payload().unwrap().schema().clone(),
        "inline-semantics",
    )
    .await
    .unwrap()
    .unwrap()
}

fn rows(records: &[RecordBatch]) -> Vec<Vec<ScalarValue>> {
    records
        .iter()
        .flat_map(|record| {
            (0..record.num_rows()).map(|row| {
                record
                    .columns()
                    .iter()
                    .map(|array| ScalarValue::try_from_array(array, row).unwrap())
                    .collect()
            })
        })
        .collect()
}

fn expected(unsigned: bool, phase: usize) -> Vec<Vec<ScalarValue>> {
    let count = HISTORICAL + usize::from(phase > 0);
    (0..count)
        .map(|key| {
            let value = match (phase, key) {
                (1..=3, 0) => Some(boundary(unsigned)),
                (4, 0) => Some(boundary(unsigned).wrapping_add(1)),
                (4, key) if key == HISTORICAL - 1 => Some(2),
                _ => None,
            };
            vec![
                ScalarValue::Utf8(Some(format!("account-{key}"))),
                scalar(unsigned, value),
            ]
        })
        .collect()
}

fn assert_inline(plan: &IncrementalSql) {
    assert!(plan.inline_integer_sum());
    assert!(plan.groups.iter().all(|group| group.aggregates.is_inline()));
}

async fn apply_phase(
    plan: &mut IncrementalSql,
    unsigned: bool,
    phase: usize,
    context: &StreamOperatorContext<'_>,
) -> Vec<RecordBatch> {
    let batch = input(unsigned, phase);
    if phase == 2 {
        let discarded = plan
            .update(&input(unsigned, 4), context, "inline-semantics")
            .await
            .unwrap();
        drop(discarded);
    }
    let transaction = plan
        .update(&batch, context, "inline-semantics")
        .await
        .unwrap();
    assert_eq!(rows(&transaction.records), expected(unsigned, phase));
    let records = transaction.records.clone();
    plan.commit(transaction);
    assert_inline(plan);
    records
}

#[tokio::test]
async fn test_grouped_inline_nullable_signed_unsigned_prefix_restore_and_discard() {
    for unsigned in [false, true] {
        let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
        let first = input(unsigned, 0);
        let mut plan = plan_for(&runtime, &first, SQL).await;
        let job = StreamJobContext::new(
            972,
            "inline-semantics",
            JsonMap::new(),
            None,
            CancellationToken::new(),
        );
        let context = StreamOperatorContext::new(&job, "inline-semantics", None);
        let held = apply_phase(&mut plan, unsigned, 0, &context).await;
        for phase in 1..=3 {
            apply_phase(&mut plan, unsigned, phase, &context).await;
        }
        let exported = plan
            .export_native_state("inline-semantics", || Ok(()))
            .unwrap();
        let restored = plan_for(&runtime, &first, SQL)
            .await
            .import_native_state(
                exported.records(),
                u64::try_from(2 * HISTORICAL + 3).unwrap(),
                true,
                || Ok(()),
                "inline-semantics",
            )
            .unwrap();
        drop(plan);
        plan = restored;
        assert_inline(&plan);
        apply_phase(&mut plan, unsigned, 4, &context).await;
        assert_eq!(rows(&held), expected(unsigned, 0));
        drop((plan, exported));
        assert_eq!(runtime.incremental_memory_pool().reserved(), 0);
    }
}

fn ipc(records: &[RecordBatch]) -> Vec<u8> {
    let mut writer = FileWriter::try_new(Vec::new(), records[0].schema().as_ref()).unwrap();
    for record in records {
        writer.write(record).unwrap();
    }
    writer.finish().unwrap();
    writer.into_inner().unwrap()
}

#[tokio::test]
async fn test_grouped_inline_native_wire_is_identical_to_legacy_vec_storage() {
    let mut fixture = allocation_fixture(true).await;
    let inline = fixture
        .plan
        .export_native_state("inline-wire", || Ok(()))
        .unwrap();
    for group in &mut fixture.plan.groups {
        let value = group.aggregates.result(0).into_owned();
        group.aggregates = GroupAggregates::Generic {
            states: vec![vec![value.clone()]],
            results: vec![value],
        };
    }
    let legacy = fixture
        .plan
        .export_native_state("inline-wire", || Ok(()))
        .unwrap();
    assert_eq!(inline.records(), legacy.records());
    assert_eq!(ipc(inline.records()), ipc(legacy.records()));
    let fresh = plan_for(
        &fixture.runtime,
        &allocation_input(),
        "SELECT key, SUM(value) AS total FROM events GROUP BY key",
    )
    .await;
    let restored = fresh
        .import_native_state(
            legacy.records(),
            u64::try_from(GROUPS).unwrap(),
            true,
            || Ok(()),
            "inline-wire",
        )
        .unwrap();
    assert_inline(&restored);
    drop((restored, inline, legacy, fixture.partial, fixture.plan));
    assert_eq!(fixture.runtime.incremental_memory_pool().reserved(), 0);
}

struct Reject;

#[async_trait::async_trait]
impl StreamCollector for Reject {
    async fn emit(&mut self, _port: &str, _batch: Batch) -> Result<()> {
        Err(df_error("inline-reject", "injected output refusal"))
    }
}

#[tokio::test]
async fn test_grouped_inline_output_refusal_preserves_capture_and_retry_once() {
    for unsigned in [false, true] {
        let mut operator =
            SqlOperator::new("inline-semantics", SQL, vec!["events".into()], vec![]).unwrap();
        let job = StreamJobContext::new(
            973,
            "inline-semantics",
            JsonMap::new(),
            None,
            CancellationToken::new(),
        );
        let context = StreamOperatorContext::new(&job, "inline-semantics", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data("events", input(unsigned, 0), &context, &mut collector)
            .await
            .unwrap();
        collector.drain("output");
        let before = operator.checkpoint(Epoch::INITIAL).unwrap();
        let paid = operator
            .stream_state
            .runtime()
            .unwrap()
            .compact_state_bytes();
        let failure = operator
            .process_data("events", input(unsigned, 1), &context, &mut Reject)
            .await
            .unwrap_err();
        assert!(
            matches!(failure, CalcFlowError::DataFusion { node_id: Some(name), .. } if name == "inline-reject")
        );
        assert_eq!(
            operator
                .stream_state
                .runtime()
                .unwrap()
                .compact_state_bytes(),
            paid
        );
        let after = operator.checkpoint(Epoch::INITIAL).unwrap();
        assert_eq!(before.inline_metadata, after.inline_metadata);
        assert_eq!(
            before.segments["group-state"].bytes(),
            after.segments["group-state"].bytes()
        );
        operator
            .process_data("events", input(unsigned, 1), &context, &mut collector)
            .await
            .unwrap();
        assert_eq!(
            rows(
                collector.drain("output")[0]
                    .as_data()
                    .unwrap()
                    .table_payload()
                    .unwrap()
                    .batches()
            ),
            expected(unsigned, 1)
        );
        assert_inline(operator.incremental.as_ref().unwrap());
    }
}

fn global_input(floating: bool) -> Batch {
    let values: ArrayRef = if floating {
        Arc::new(datafusion::arrow::array::Float64Array::from(vec![
            Some(1.0),
            Some(2.0),
            None,
        ]))
    } else {
        Arc::new(Int64Array::from(vec![Some(1), Some(2), None]))
    };
    let record = RecordBatch::try_from_iter(vec![
        ("value", values),
        (
            "keep",
            Arc::new(BooleanArray::from(vec![true, false, true])) as ArrayRef,
        ),
    ])
    .unwrap();
    Batch::table(vec![record], crate::BatchMetadata::default()).unwrap()
}

async fn global_continuation(floating: bool) {
    let sql = "SELECT SUM(value) AS total FROM events WHERE keep";
    let make = || SqlOperator::new("inline-global", sql, vec!["events".into()], vec![]).unwrap();
    let mut operator = make();
    let job = StreamJobContext::new(
        974,
        "inline-global",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "inline-global", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    for total in [1, 2] {
        operator
            .process_data("events", global_input(floating), &context, &mut collector)
            .await
            .unwrap();
        let plan = operator.incremental.as_ref().unwrap();
        assert!(!plan.inline_integer_sum());
        assert!(!plan.groups[0].aggregates.is_inline());
        assert!(plan.requires_global_coalescer());
        let expected = if floating {
            ScalarValue::Float64(Some(f64::from(total)))
        } else {
            ScalarValue::Int64(Some(i64::from(total)))
        };
        assert_eq!(
            rows(
                collector.drain("output")[0]
                    .as_data()
                    .unwrap()
                    .table_payload()
                    .unwrap()
                    .batches()
            ),
            vec![vec![expected]]
        );
        let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
        drop(operator);
        operator = make();
        StreamOperator::restore(&mut operator, &snapshot).unwrap();
    }
    drop(operator);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
}

#[tokio::test]
async fn test_grouped_inline_global_single_sum_keeps_generic_coalescer_and_cold_continuation() {
    global_continuation(false).await;
    global_continuation(true).await;
}
