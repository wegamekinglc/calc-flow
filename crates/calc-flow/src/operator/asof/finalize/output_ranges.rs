//! Output planning must retain ranges without redundant left row positions.

use super::*;
use crate::runtime::streaming::gather_work::TestService;
use crate::{AsofStateLimits, CancellationToken, StreamAsofJoinSpec, StreamJobContext};
use datafusion::{
    arrow::{
        array::{
            ArrayRef, BinaryArray, Int64Array, LargeBinaryArray, LargeStringArray, StringArray,
            TimestampMicrosecondArray,
        },
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    },
    execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool},
};
use std::sync::OnceLock;

fn operator(count: usize) -> StreamAsofJoinOperator {
    let (template, _) = super::super::tests::fixture();
    let schema = template.schemas[0].clone();
    let spec = StreamAsofJoinSpec::new(
        template.spec.left().clone(),
        template.spec.right().clone(),
        std::time::Duration::ZERO,
        AsofStateLimits::new(100_000, 100_000_000).unwrap(),
    )
    .unwrap();
    let mut operator =
        StreamAsofJoinOperator::new("asof", schema.clone(), schema.clone(), spec).unwrap();
    let record = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec!["A"; count])),
            Arc::new(
                TimestampMicrosecondArray::from_iter_values(
                    (0..count).map(|row| i64::try_from(row).unwrap()),
                )
                .with_timezone("UTC"),
            ),
            Arc::new(Int64Array::from_iter_values(
                (0..count).map(|row| i64::try_from(row).unwrap()),
            )),
        ],
    )
    .unwrap();
    let keys = state::encode_columns(&record, operator.spec.left().keys()).unwrap();
    let sequences = state::encode_columns(&record, operator.spec.left().sequence_by()).unwrap();
    let rows = (0..count)
        .map(|row| {
            (
                (
                    i64::try_from(row).unwrap(),
                    keys.row(row),
                    sequences.row(row),
                ),
                state::AdmissionRef {
                    batch_index: 0,
                    row: u32::try_from(row).unwrap(),
                    key_index: 0,
                },
            )
        })
        .collect::<Vec<_>>();
    let owner = Arc::new(state::PayloadBatch {
        key: (0, 0),
        record: Arc::new(record),
        encoded: OnceLock::new(),
        encoded_charge_bytes: 0,
        body_bytes: 0,
    });
    let chunks =
        state::PreparedLeftChunk::prepare(&rows, &[owner], operator.spec.left(), "asof").unwrap();
    operator
        .state
        .left
        .install(chunks, &mut operator.state.batches);
    operator.state.rebuild_encoding_owners();
    operator
}

#[test]
fn planning_vector_allocations_scale_only_with_candidate_references() {
    let operator = operator(1);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 24));
    for count in [1_024, 0, 1, 17, 64_000] {
        let mut credit = MemoryConsumer::new("planning-allocation-test").register(&pool);
        let mut builder = None;
        let allocation = allocation_counter::measure(|| {
            builder = Some(OutputPlanBuilder::new(count, None, &mut credit, "asof").unwrap());
        });
        assert_eq!(credit.size(), count * 256 + 16 * 1_024);
        let expected = count * size_of::<(usize, usize)>() + 512;
        assert!(
            allocation.bytes_max <= u64::try_from(expected).unwrap(),
            "count={count}, expected={expected}, actual={allocation:?}"
        );
        let plan = builder
            .unwrap()
            .finish(operator.physical_schema(1), &mut credit, "asof")
            .unwrap();
        assert_eq!(plan.left.positions.capacity(), 0);
        assert_eq!(plan.left.spans.capacity(), 0);
        assert_eq!(plan.right.positions.capacity(), count);
    }
}

#[tokio::test]
async fn matching_left_planning_steps_scale_with_source_ranges() {
    let count = 1_024;
    let operator = operator(count);
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 24));
    for mode in 0..3 {
        let mut credit = MemoryConsumer::new("ranges-test").register(&pool);
        let mut builder = OutputPlanBuilder::new(count, None, &mut credit, "asof").unwrap();
        super::super::workspace::take_left_output_planning();
        let prefix = match mode {
            0 => {
                binary_search_candidate_rows(&operator, count, &context, &mut builder, &mut credit)
                    .await
                    .unwrap()
            }
            1 => monotonic_candidate_rows(&operator, count, &context, &mut builder, &mut credit)
                .await
                .unwrap(),
            _ => parallel_candidate_rows(
                &operator,
                count,
                &context,
                &vec![None; count],
                &mut builder,
                &mut credit,
            )
            .await
            .unwrap(),
        };
        let plan = builder
            .finish(&operator.schemas[1], &mut credit, "asof")
            .unwrap();
        assert_eq!(prefix.count, count);
        assert_eq!(plan.len, count);
        assert_eq!(plan.matched, 0);
        assert_eq!(
            plan.left
                .spans
                .iter()
                .map(|span| (span.source, span.start, span.end))
                .collect::<Vec<_>>(),
            vec![(0, 0, count)]
        );
        assert_eq!(
            super::super::workspace::take_left_output_planning(),
            (1, 1),
            "matching mode {mode}"
        );
        assert!(plan.left.positions.is_empty());
        assert!(plan.left.spans.capacity() <= 8);
        assert_eq!(plan.right.positions, vec![(0, 0); count]);
    }
}

#[test]
fn fragmented_span_allocations_fit_the_unchanged_planning_charge() {
    for count in [0, 1, 2, 3, 7, 17, 1_024, 64_000] {
        let operator = operator(count.max(1));
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(128 << 20));
        let mut credit = MemoryConsumer::new("fragment-allocation-test").register(&pool);
        let mut builder = OutputPlanBuilder::new(count, None, &mut credit, "asof").unwrap();
        let first = operator.state.left.iter().next().unwrap().1;
        let source = builder
            .left_source(operator.state.batches.view(first), &mut credit, "asof")
            .unwrap();
        let mut plan = None;
        let allocation = allocation_counter::measure(|| {
            for row in (0..count).rev() {
                builder
                    .push_reference(source, row, None, &mut credit, "asof")
                    .unwrap();
            }
            plan = Some(
                builder
                    .finish(operator.physical_schema(1), &mut credit, "asof")
                    .unwrap(),
            );
        });
        assert!(
            allocation.bytes_max <= u64::try_from(count * 256 + 16 * 1_024).unwrap(),
            "count={count}, actual={allocation:?}"
        );
        let plan = plan.unwrap();
        assert_eq!(plan.left.spans.len(), count);
        assert_eq!(plan.right.positions.len(), count);
        assert_eq!(plan.left.positions.capacity(), 0);
        drop((plan, credit));
        assert_eq!(pool.reserved(), 0);
    }
}

#[test]
fn dictionary_and_nested_payloads_keep_existing_schema_rejection() {
    let operator = operator(1);
    let item = Arc::new(Field::new("item", DataType::Int64, true));
    for data_type in [
        DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
        DataType::List(item.clone()),
        DataType::Struct(vec![item].into()),
    ] {
        for side in 0..2 {
            let mut schemas = [operator.schemas[0].clone(), operator.schemas[1].clone()];
            let fields = schemas[side]
                .fields()
                .iter()
                .cloned()
                .chain([Arc::new(Field::new("unsupported", data_type.clone(), true))])
                .collect::<Vec<_>>();
            schemas[side] = Arc::new(Schema::new(fields));
            let issues =
                super::super::schema::schema_issues(&operator.spec, &schemas[0], &schemas[1]);
            assert_eq!(issues.len(), 1);
            assert_eq!(
                issues[0].path,
                format!(
                    "{}_schema.unsupported",
                    if side == 0 { "left" } else { "right" }
                )
            );
            assert_eq!(issues[0].code, "invalid_type");
            assert!(issues[0].message.contains("flat Arrow payload type"));
            assert!(matches!(
                StreamAsofJoinOperator::new(
                    "asof",
                    schemas[0].clone(),
                    schemas[1].clone(),
                    operator.spec.clone()
                ),
                Err(CalcFlowError::InvalidArgument { .. })
            ));
        }
    }
}

fn mixed_record(extra_columns: usize) -> RecordBatch {
    let values = [
        Some("outside"),
        None,
        Some(""),
        Some("é"),
        Some("long payload"),
        Some("x"),
        None,
        Some("tail"),
    ];
    let bytes = values.map(|value| value.map(str::as_bytes));
    let mut columns = vec![
        Arc::new(StringArray::from(vec!["A"; values.len()])) as ArrayRef,
        Arc::new(TimestampMicrosecondArray::from_iter_values(0..8).with_timezone("UTC")),
        Arc::new(Int64Array::from_iter_values(0..8)),
        Arc::new(StringArray::from(values.to_vec())),
        Arc::new(LargeStringArray::from(values.to_vec())),
        Arc::new(BinaryArray::from(bytes.to_vec())),
        Arc::new(LargeBinaryArray::from(bytes.to_vec())),
    ];
    columns.extend((0..extra_columns).map(|_| {
        Arc::new(Int64Array::from(vec![
            Some(9),
            None,
            Some(7),
            Some(6),
            Some(5),
            Some(4),
            Some(3),
            Some(2),
        ])) as ArrayRef
    }));
    let schema = Arc::new(Schema::new(
        columns
            .iter()
            .enumerate()
            .map(|(index, column)| {
                Field::new(
                    format!("field_{index}"),
                    column.data_type().clone(),
                    index >= 3,
                )
            })
            .collect::<Vec<_>>(),
    ));
    RecordBatch::try_new(schema, columns).unwrap()
}

fn payload(record: RecordBatch, key: BatchKey) -> state::RowPayload {
    state::RowPayload {
        batch: Arc::new(state::PayloadBatch {
            key,
            record: Arc::new(record),
            encoded: OnceLock::new(),
            encoded_charge_bytes: 0,
            body_bytes: 0,
        }),
        row: 0,
    }
}

fn view(source: &state::RowPayload, row: usize) -> PayloadView<'_> {
    PayloadView {
        batch: &source.batch,
        row,
    }
}

#[test]
fn every_fragmented_cut_matches_rowwise_output_and_workspace() {
    let service = TestService::new(2, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let pools = runtime.block_on(fragmented_cuts(&service));
    drop(runtime);
    service.shutdown();
    for pool in pools {
        assert_eq!(pool.reserved(), 0);
    }
}

async fn fragmented_cuts(service: &TestService) -> Vec<Arc<dyn MemoryPool>> {
    let mut pools = Vec::new();
    for extra in [0, 12] {
        let record = mixed_record(extra);
        let fields = record.num_columns();
        let schema = Arc::new(Schema::new(
            (0..2)
                .flat_map(|side| {
                    record
                        .schema()
                        .fields()
                        .iter()
                        .enumerate()
                        .map(move |(index, field)| {
                            Field::new(
                                format!("side_{side}_{index}"),
                                field.data_type().clone(),
                                side == 1 || field.is_nullable(),
                            )
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>(),
        ));
        let left_a = payload(record.slice(1, 6), (0, 0));
        let left_b = payload(record.clone(), (0, 1));
        let right_a = payload(record.slice(1, 6), (1, 0));
        let right_b = payload(record, (1, 1));
        let rows = [
            (view(&left_a, 0), Some(view(&right_a, 3))),
            (view(&left_a, 1), None),
            (view(&left_a, 4), Some(view(&right_a, 3))),
            (view(&left_b, 2), Some(view(&right_b, 0))),
            (view(&left_b, 3), Some(view(&right_b, 1))),
            (view(&left_a, 2), None),
            (view(&left_a, 3), Some(view(&right_a, 0))),
            (view(&left_a, 4), Some(view(&right_a, 1))),
            (view(&left_b, 7), None),
            (view(&left_b, 0), Some(view(&right_b, 2))),
            (view(&left_a, 5), Some(view(&right_a, 2))),
            (view(&left_b, 1), None),
        ];
        let projections = [
            None,
            Some([vec![3, 4, 5, 6], vec![3, 4, 5, 6]]),
            Some([vec![], vec![]]),
            Some([vec![0, 2, fields - 1], vec![]]),
            Some([vec![], vec![3, fields - 1]]),
        ];
        for selected in &projections {
            for count in 0..=rows.len() {
                pools.push(
                    assert_cut(&rows[..count], selected.as_ref(), &schema, fields, service).await,
                );
            }
        }
    }
    pools
}

async fn assert_cut(
    rows: &[(PayloadView<'_>, Option<PayloadView<'_>>)],
    selected: Option<&[Vec<usize>; 2]>,
    schema: &Arc<Schema>,
    fields: usize,
    service: &TestService,
) -> Arc<dyn MemoryPool> {
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new())
        .with_gather_owner(service.owner("asof-output-ranges".into()));
    let context = StreamOperatorContext::new(&job, "asof", None);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(16 << 20));
    let mut credit = MemoryConsumer::new("range-oracle").register(&pool);
    let mut reference_credit = MemoryConsumer::new("rowwise-oracle").register(&pool);
    let mut builder = OutputPlanBuilder::new(rows.len(), selected, &mut credit, "asof").unwrap();
    let mut reference =
        OutputPlanBuilder::new(rows.len(), selected, &mut reference_credit, "asof").unwrap();
    let mut recent = None;
    let mut source = 0;
    for &(left, right) in rows {
        if recent != Some(left.batch.key) {
            source = builder.left_source(left, &mut credit, "asof").unwrap();
            recent = Some(left.batch.key);
        }
        builder
            .push_reference(source, left.row, right, &mut credit, "asof")
            .unwrap();
        reference
            .push(left, right, &mut reference_credit, "asof")
            .unwrap();
    }
    let right_schema = Schema::new(schema.fields()[fields..].to_vec());
    let plan = builder.finish(&right_schema, &mut credit, "asof").unwrap();
    let reference = reference
        .finish(&right_schema, &mut reference_credit, "asof")
        .unwrap();
    assert_eq!(plan.raw_bytes, rowwise_raw_bytes(rows, selected));
    assert_eq!(credit.size(), reference_credit.size());
    assert_eq!(plan.matched, reference.matched);
    assert_eq!(plan.right.positions, reference.right.positions);
    assert!(plan.left.positions.is_empty());
    assert_eq!(
        plan.left
            .spans
            .iter()
            .map(|span| (span.source, span.start, span.end))
            .collect::<Vec<_>>(),
        reference
            .left
            .spans
            .iter()
            .map(|span| (span.source, span.start, span.end))
            .collect::<Vec<_>>()
    );
    drop((reference, reference_credit));
    if rows.is_empty() {
        drop((plan, credit));
        assert_eq!(pool.reserved(), 0);
        return pool;
    }
    let full = super::super::output::materialize_rows(rows, schema).unwrap();
    let indices = selected.map_or_else(
        || (0..2 * fields).collect::<Vec<_>>(),
        |columns| {
            columns[0]
                .iter()
                .copied()
                .chain(columns[1].iter().map(|index| index + fields))
                .collect()
        },
    );
    let expected = full.table_payload().unwrap().batches()[0]
        .project(&indices)
        .unwrap();
    let mut runtime = super::super::output::OutputRuntime::new(16 << 20, "asof");
    if selected.is_some() {
        runtime.set_output_projection(indices.clone());
    }
    let schema = Arc::new(schema.project(&indices).unwrap());
    let (actual, credit) = runtime
        .materialize_plan(plan, &schema, credit, "asof", &context)
        .await
        .unwrap();
    assert_eq!(actual.table_payload().unwrap().batches(), &[expected]);
    drop((actual, credit));
    job.gather_owner().close_and_drain().await;
    pool
}

fn rowwise_raw_bytes(
    rows: &[(PayloadView<'_>, Option<PayloadView<'_>>)],
    selected: Option<&[Vec<usize>; 2]>,
) -> u64 {
    rows.iter()
        .flat_map(|(left, right)| [(0, Some(*left)), (1, *right)])
        .filter_map(|(side, row)| row.map(|row| (side, row)))
        .map(|(side, row)| {
            super::super::output_plan::selected_columns(
                &row.batch.record,
                selected.map(|columns| columns[side].as_slice()),
            )
            .map(|column| {
                column
                    .to_data()
                    .slice(row.row, 1)
                    .get_slice_memory_size()
                    .unwrap() as u64
            })
            .sum::<u64>()
        })
        .sum::<u64>()
}
