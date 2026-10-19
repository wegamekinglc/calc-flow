use super::*;
use crate::operator::asof::output_plan::{OutputSide, Span};
use datafusion::{
    arrow::{
        array::Int64Array,
        datatypes::{DataType, Field, Schema},
    },
    execution::memory_pool::MemoryConsumer,
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
            positions: Vec::new(),
            spans: vec![Span {
                source: 0,
                start: range.start,
                end: range.end,
            }],
            has_nulls: false,
        },
        right: OutputSide {
            batches: vec![vec![right]],
            positions,
            spans: Vec::new(),
            has_nulls: matched != rows,
        },
        len: rows,
        matched: matched as u64,
        raw_bytes: rows as u64 * 16,
    }
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
