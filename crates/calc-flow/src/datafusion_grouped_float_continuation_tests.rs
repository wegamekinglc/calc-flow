use super::*;
use datafusion::{
    arrow::{
        array::{ArrayRef, Float32Array, Float64Array, Int64Array},
        datatypes::{DataType, Field, Schema},
    },
    logical_expr::{EmitTo, GroupsAccumulator},
    physical_expr::aggregate::AggregateFunctionExpr,
    physical_plan::{ExecutionPlan, aggregates::AggregateExec},
};

type Bits = (u32, u64);
type Row = Option<Bits>;
const ZERO: Bits = (0, 0);
const NEG_ZERO: Bits = (0x8000_0000, 0x8000_0000_0000_0000);
const ONE: Bits = (0x3f80_0000, 0x3ff0_0000_0000_0000);
const TWO: Bits = (0x4000_0000, 0x4000_0000_0000_0000);
const POS_INF: Bits = (0x7f80_0000, 0x7ff0_0000_0000_0000);
const NEG_INF: Bits = (0xff80_0000, 0xfff0_0000_0000_0000);
const MAX: Bits = (0x7f7f_ffff, 0x7fef_ffff_ffff_ffff);
const MIN: Bits = (0xff7f_ffff, 0xffef_ffff_ffff_ffff);
const NAN_A: Bits = (0x7fc0_0001, 0x7ff8_0000_0000_0001);
const NAN_B: Bits = (0x7fc0_0002, 0x7ff8_0000_0000_0002);
const SNAN_A: Bits = (0x7f80_0003, 0x7ff0_0000_0000_0003);
const SNAN_B: Bits = (0x7f80_0004, 0x7ff0_0000_0000_0004);
const NEG_NAN_A: Bits = (0xffc0_0005, 0xfff8_0000_0000_0005);
const NEG_NAN_B: Bits = (0xffc0_0006, 0xfff8_0000_0000_0006);
const NEG_SNAN_A: Bits = (0xff80_0007, 0xfff0_0000_0000_0007);
const NEG_SNAN_B: Bits = (0xff80_0008, 0xfff0_0000_0000_0008);

fn array(dtype: &DataType, rows: &[Row]) -> ArrayRef {
    match dtype {
        DataType::Float32 => Arc::new(Float32Array::from(
            rows.iter()
                .map(|row| row.map(|bits| f32::from_bits(bits.0)))
                .collect::<Vec<_>>(),
        )),
        DataType::Float64 => Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.map(|bits| f64::from_bits(bits.1)))
                .collect::<Vec<_>>(),
        )),
        _ => unreachable!(),
    }
}

fn bit_rows(array: &ArrayRef) -> Vec<Option<u64>> {
    (0..array.len())
        .map(|row| {
            (!array.is_null(row)).then(|| match array.data_type() {
                DataType::Float32 => u64::from(
                    array
                        .as_any()
                        .downcast_ref::<Float32Array>()
                        .unwrap()
                        .value(row)
                        .to_bits(),
                ),
                DataType::Float64 => array
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap()
                    .value(row)
                    .to_bits(),
                _ => unreachable!(),
            })
        })
        .collect()
}

fn aggregate(plan: &dyn ExecutionPlan) -> &AggregateExec {
    if let Some(aggregate) = plan.downcast_ref::<AggregateExec>() {
        return aggregate;
    }
    aggregate(plan.children().into_iter().next().unwrap().as_ref())
}

async fn expressions(dtype: &DataType) -> Vec<Arc<AggregateFunctionExpr>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new("value", dtype.clone(), true),
    ]));
    let record = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1])),
            array(dtype, &[Some(ONE)]),
        ],
    )
    .unwrap();
    let batch = Batch::table(vec![record], BatchMetadata::default()).unwrap();
    let tables = BTreeMap::from([("events".into(), batch)]);
    let query =
        parse_select_query("SELECT key, MIN(value), MAX(value) FROM events GROUP BY key").unwrap();
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let context = runtime.context_for_rows(1, None, "not_evaluated");
    let planned = runtime
        .prepare_query(context, &query, &tables, Some("seed-continuation"))
        .await
        .unwrap();
    let expressions = aggregate(planned.physical_plan.as_ref())
        .aggr_expr()
        .to_vec();
    assert_eq!(expressions.len(), 2);
    expressions
}

fn update(
    accumulator: &mut dyn GroupsAccumulator,
    dtype: &DataType,
    rows: &[Row],
    ranks: &[usize],
    count: usize,
) {
    assert!(!rows.is_empty());
    assert_eq!(rows.len(), ranks.len());
    accumulator
        .update_batch(&[array(dtype, rows)], ranks, None, count)
        .unwrap();
}

fn saved(
    expression: &AggregateFunctionExpr,
    dtype: &DataType,
    rows: &[Row],
    ranks: &[usize],
    count: usize,
) -> ArrayRef {
    let mut original = expression.create_groups_accumulator().unwrap();
    update(original.as_mut(), dtype, rows, ranks, count);
    let state = original.state(EmitTo::All).unwrap();
    assert_eq!(state.len(), 1);
    state[0].clone()
}

fn seed(accumulator: &mut dyn GroupsAccumulator, dtype: &DataType, state: &ArrayRef, count: usize) {
    let reset = (0..state.len())
        .map(|row| state.is_valid(row).then_some(NAN_A))
        .collect::<Vec<_>>();
    let ranks = (0..state.len()).collect::<Vec<_>>();
    accumulator
        .merge_batch(&[array(dtype, &reset)], &ranks, None, count)
        .unwrap();
    accumulator
        .merge_batch(&[state.clone()], &ranks, None, count)
        .unwrap();
}

fn prefixes() -> Vec<Vec<Row>> {
    let single = [
        None,
        Some(ZERO),
        Some(NEG_ZERO),
        Some(ONE),
        Some((0xbf80_0000, 0xbff0_0000_0000_0000)),
        Some(TWO),
        Some(MAX),
        Some(MIN),
        Some(POS_INF),
        Some(NEG_INF),
        Some(NAN_A),
        Some(NAN_B),
        Some(SNAN_A),
        Some(SNAN_B),
        Some(NEG_NAN_A),
        Some(NEG_NAN_B),
        Some(NEG_SNAN_A),
        Some(NEG_SNAN_B),
    ];
    single
        .into_iter()
        .map(|value| vec![value])
        .chain([
            vec![Some(NAN_A), Some(POS_INF)],
            vec![Some(NAN_A), Some(NEG_INF)],
            vec![Some(ZERO), Some(NEG_ZERO)],
            vec![Some(NEG_ZERO), Some(ZERO)],
            vec![None, None],
        ])
        .collect()
}

#[tokio::test]
async fn test_grouped_float_reset_seed_preserves_reachable_state_bits_and_validity() {
    for dtype in [DataType::Float32, DataType::Float64] {
        for expression in expressions(&dtype).await {
            for prefix in prefixes() {
                let state = saved(&expression, &dtype, &prefix, &vec![0; prefix.len()], 1);
                let mut continued = expression.create_groups_accumulator().unwrap();
                seed(continued.as_mut(), &dtype, &state, 1);
                let actual = continued.evaluate(EmitTo::All).unwrap();
                assert_eq!(
                    bit_rows(&actual),
                    bit_rows(&state),
                    "{dtype:?} {} {prefix:?}",
                    expression.name()
                );
            }
        }
    }
}

fn assert_continuation(
    expression: &AggregateFunctionExpr,
    dtype: &DataType,
    prefix: &[Row],
    next: &[Row],
) {
    let state = saved(expression, dtype, prefix, &vec![0; prefix.len()], 1);
    for end in 1..=next.len() {
        let mut actual = expression.create_groups_accumulator().unwrap();
        seed(actual.as_mut(), dtype, &state, 1);
        for row in &next[..end] {
            update(actual.as_mut(), dtype, &[*row], &[0], 1);
        }
        let output = actual.evaluate(EmitTo::All).unwrap();
        let all = prefix
            .iter()
            .chain(&next[..end])
            .copied()
            .collect::<Vec<_>>();
        let expected = saved(expression, dtype, &all, &vec![0; all.len()], 1);
        assert_eq!(
            bit_rows(&output),
            bit_rows(&expected),
            "{dtype:?} {} prefix={prefix:?} next={:?}",
            expression.name(),
            &next[..end]
        );
    }
}

#[tokio::test]
async fn test_grouped_float_reset_seed_continues_each_row_like_original_chronology() {
    let next = [
        None,
        Some(NAN_B),
        Some(TWO),
        Some(ZERO),
        Some(NEG_ZERO),
        Some(POS_INF),
        Some(NEG_INF),
        Some(NEG_SNAN_B),
        Some(ONE),
        Some(NEG_NAN_B),
    ];
    for dtype in [DataType::Float32, DataType::Float64] {
        for expression in expressions(&dtype).await {
            for prefix in prefixes() {
                assert_continuation(&expression, &dtype, &prefix, &next);
            }
        }
    }
}

fn sparse_continuation(expression: &AggregateFunctionExpr, dtype: &DataType) {
    let prefix = [Some(NAN_A), Some(POS_INF), None, Some(ZERO), Some(NEG_ZERO)];
    let prefix_ranks = [7, 7, 2, 5, 5];
    let chunks = [
        (
            [Some(NAN_B), Some(NEG_INF), Some(ONE), Some(NEG_ZERO)],
            [7, 7, 2, 0],
        ),
        (
            [Some(TWO), Some(NEG_NAN_A), Some(TWO), Some(ZERO)],
            [7, 5, 2, 0],
        ),
    ];
    let state = saved(expression, dtype, &prefix, &prefix_ranks, 8);
    let mut continued = expression.create_groups_accumulator().unwrap();
    seed(continued.as_mut(), dtype, &state, 8);
    for (rows, ranks) in &chunks {
        update(continued.as_mut(), dtype, rows, ranks, 8);
    }
    let actual = continued.evaluate(EmitTo::All).unwrap();
    let mut original = expression.create_groups_accumulator().unwrap();
    update(original.as_mut(), dtype, &prefix, &prefix_ranks, 8);
    for (rows, ranks) in &chunks {
        update(original.as_mut(), dtype, rows, ranks, 8);
    }
    let expected = original.evaluate(EmitTo::All).unwrap();
    assert_eq!(
        bit_rows(&actual),
        bit_rows(&expected),
        "{dtype:?} {}",
        expression.name()
    );
    assert!(
        bit_rows(&actual)[1].is_none()
            && bit_rows(&actual)[3].is_none()
            && bit_rows(&actual)[4].is_none()
            && bit_rows(&actual)[6].is_none()
    );
    let mut wrongly_reseeded = expression.create_groups_accumulator().unwrap();
    seed(wrongly_reseeded.as_mut(), dtype, &state, 8);
    update(
        wrongly_reseeded.as_mut(),
        dtype,
        &chunks[0].0,
        &chunks[0].1,
        8,
    );
    seed(wrongly_reseeded.as_mut(), dtype, &state, 8);
    update(
        wrongly_reseeded.as_mut(),
        dtype,
        &chunks[1].0,
        &chunks[1].1,
        8,
    );
    assert_ne!(
        bit_rows(&wrongly_reseeded.evaluate(EmitTo::All).unwrap()),
        bit_rows(&expected)
    );
}

#[tokio::test]
async fn test_grouped_float_reset_seed_sparse_ranks_two_chunks_do_not_reseed() {
    for dtype in [DataType::Float32, DataType::Float64] {
        for expression in expressions(&dtype).await {
            sparse_continuation(&expression, &dtype);
        }
    }
}
