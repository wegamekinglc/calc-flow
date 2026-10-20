use super::*;
use crate::operator::asof::{
    output_plan::{OutputPlanBuilder, OutputSide, Span},
    state::{PayloadBatch, PayloadView},
};
use datafusion::{
    arrow::{
        array::Int64Array,
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    },
    execution::memory_pool::{MemoryConsumer, MemoryPool},
};

fn plan(
    left: ArrayRef,
    right: ArrayRef,
    range: std::ops::Range<usize>,
    positions: Vec<(usize, usize)>,
) -> OutputPlan {
    let rows = positions.len();
    let matched = positions.iter().filter(|(source, _)| *source != 0).count();
    OutputPlan {
        left: OutputSide {
            batches: vec![vec![left]],
            positions: Vec::new().into(),
            spans: vec![Span {
                source: 0,
                start: range.start,
                end: range.end,
            }],
            has_nulls: false,
        },
        right: OutputSide {
            batches: vec![vec![right]],
            positions: positions.into(),
            spans: Vec::new(),
            has_nulls: matched != rows,
        },
        len: rows,
        matched: matched as u64,
        raw_bytes: rows as u64 * 16,
    }
}

fn owned_payload(key: (u8, u64), column: ArrayRef) -> PayloadBatch {
    PayloadBatch {
        key,
        record: Arc::new(RecordBatch::try_from_iter(vec![("value", column)]).unwrap()),
        body_bytes: 0,
        encoded_charge_bytes: 0,
        encoded: std::sync::OnceLock::new(),
    }
}

fn compact_fragment(
    left: &ArrayRef,
    start: usize,
    multiple: bool,
    pool: &Arc<dyn MemoryPool>,
) -> (OutputPlan, MemoryReservation) {
    let left = owned_payload((0, 0), left.clone());
    let first = owned_payload((1, 0), Arc::new(Int64Array::from(vec![20])));
    let second = owned_payload((1, 1), Arc::new(Int64Array::from(vec![30])));
    let mut credit = MemoryConsumer::new("compact-fragment").register(pool);
    let mut builder = OutputPlanBuilder::new(3, None, &mut credit, "asof").unwrap();
    let final_source = if multiple { &second } else { &first };
    let right = [None, Some(&first), Some(final_source)];
    for (offset, right) in right.into_iter().enumerate() {
        builder
            .push(
                PayloadView {
                    batch: &left,
                    row: start + offset,
                },
                right.map(|batch| PayloadView { batch, row: 0 }),
                &mut credit,
                "asof",
            )
            .unwrap();
    }
    let plan = builder
        .finish(&first.record.schema(), &mut credit, "asof")
        .unwrap();
    (plan, credit)
}

#[tokio::test]
async fn promoted_fragments_preserve_runtime_and_cross_owner_ranges() {
    let left: ArrayRef = Arc::new(Int64Array::from(vec![10, 11, 12, 13, 14, 15]));
    let mut output = OutputRuntime::new(8 << 20, "asof");
    let pool = output.pool.clone();
    let (first, first_credit) = compact_fragment(&left, 0, false, &pool);
    let (second, second_credit) = compact_fragment(&left, 3, true, &pool);
    let input_credit = MemoryConsumer::new("compact-fragment-input").register(&pool);
    let credits = Arc::new(FragmentCredits {
        _fragments: vec![first_credit, second_credit],
        _input: Arc::new(input_credit),
    });
    let job = crate::StreamJobContext::new(
        922,
        "compact-fragments",
        crate::JsonMap::new(),
        None,
        crate::CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "asof", None);
    let input = SegmentedInput {
        plans: vec![first, second],
        requests: vec![ColumnRequest {
            index: 1,
            data_type: DataType::Int64,
        }],
        shared: vec![None],
        left_fields: 1,
        rows: 6,
        _credits: credits.clone(),
    };
    let column = input
        .gather_range(0, 1..4, &GatherStop::from_job(&job))
        .unwrap();
    assert_eq!(
        column
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .iter()
            .collect::<Vec<_>>(),
        [Some(20), Some(20), None]
    );
    let SegmentedInput {
        plans,
        _credits: input_credits,
        ..
    } = input;
    drop(input_credits);
    let schema = Arc::new(Schema::new(vec![
        Field::new("left", DataType::Int64, false),
        Field::new("right", DataType::Int64, true),
    ]));
    let workspace = MemoryConsumer::new("compact-fragment-output").register(&pool);
    let (batch, workspace) = output
        .materialize_fragments(plans, credits.clone(), &schema, workspace, "asof", &context)
        .await
        .unwrap();
    let record = &batch.table_payload().unwrap().batches()[0];
    assert!(Arc::ptr_eq(record.column(0), &left));
    assert_eq!(
        record
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .iter()
            .collect::<Vec<_>>(),
        [None, Some(20), Some(20), None, Some(20), Some(30)]
    );
    drop((batch, workspace, credits));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn cpu_pipeline_segment_sources_stay_local_and_left_spans_share_full_owner() {
    let left: ArrayRef = Arc::new(Int64Array::from(vec![0, 1, 2, 3, 4, 5]));
    let plans = vec![
        plan(
            left.clone(),
            Arc::new(Int64Array::from(vec![10, 11])),
            0..3,
            vec![(1, 0), (0, 0), (1, 1)],
        ),
        plan(
            left.clone(),
            Arc::new(Int64Array::from(vec![100, 101])),
            3..6,
            vec![(1, 0), (1, 1), (0, 0)],
        ),
    ];
    let schema = Arc::new(Schema::new(vec![
        Field::new("left", DataType::Int64, false),
        Field::new("right", DataType::Int64, true),
    ]));
    let job = crate::StreamJobContext::new(
        921,
        "segment-sources",
        crate::JsonMap::new(),
        None,
        crate::CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "asof", None);
    let mut output = OutputRuntime::new(8 << 20, "asof");
    let pool = output.pool.clone();
    let input = MemoryConsumer::new("input-and-fragments").register(&pool);
    input.try_grow(1 << 20).unwrap();
    let credits = Arc::new(FragmentCredits {
        _fragments: Vec::new(),
        _input: Arc::new(input),
    });
    let workspace = MemoryConsumer::new("segment-output").register(&pool);
    let (batch, workspace) = output
        .materialize_fragments(plans, credits.clone(), &schema, workspace, "asof", &context)
        .await
        .unwrap();
    let record = &batch.table_payload().unwrap().batches()[0];
    assert!(Arc::ptr_eq(record.column(0), &left));
    assert_eq!(record.schema(), schema);
    assert_eq!(
        record
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .iter()
            .collect::<Vec<_>>(),
        [Some(10), None, Some(11), Some(100), Some(101), None]
    );
    drop((batch, workspace, credits));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}
