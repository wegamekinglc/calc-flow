use super::*;
use datafusion::{
    arrow::{
        array::{ArrayRef, Float32Array, Float64Array, Int64Array},
        datatypes::{DataType, Field, Schema},
    },
    logical_expr::EmitTo,
    physical_plan::{
        ExecutionPlan, InputOrderMode,
        aggregates::{AggregateExec, AggregateMode},
    },
};

fn values(dtype: &DataType, maximum: bool) -> ArrayRef {
    match dtype {
        DataType::Float32 => Arc::new(Float32Array::from(vec![
            f32::from_bits(0x7fc0_0001),
            if maximum {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            },
        ])),
        DataType::Float64 => Arc::new(Float64Array::from(vec![
            f64::from_bits(0x7ff8_0000_0000_0001),
            if maximum {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            },
        ])),
        _ => unreachable!(),
    }
}

fn bits(array: &ArrayRef) -> u64 {
    assert_eq!(array.len(), 1);
    assert!(!array.is_null(0));
    match array.data_type() {
        DataType::Float32 => u64::from(
            array
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap()
                .value(0)
                .to_bits(),
        ),
        DataType::Float64 => array
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .value(0)
            .to_bits(),
        _ => unreachable!(),
    }
}

fn aggregate(plan: &dyn ExecutionPlan) -> &AggregateExec {
    if let Some(aggregate) = plan.downcast_ref::<AggregateExec>() {
        return aggregate;
    }
    let child = plan.children().into_iter().next().unwrap();
    aggregate(child.as_ref())
}

async fn assert_plain_seed_mismatch(dtype: DataType, maximum: bool) {
    let values = values(&dtype, maximum);
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new("value", dtype.clone(), false),
    ]));
    let record = RecordBatch::try_new(
        schema,
        vec![Arc::new(Int64Array::from(vec![1, 1])), values.clone()],
    )
    .unwrap();
    let batch = Batch::table(vec![record], BatchMetadata::default()).unwrap();
    let tables = BTreeMap::from([("events".into(), batch)]);
    let query = parse_select_query(if maximum {
        "SELECT key, MAX(value) FROM events GROUP BY key"
    } else {
        "SELECT key, MIN(value) FROM events GROUP BY key"
    })
    .unwrap();
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let context = runtime.context_for_rows(2, None, "not_evaluated");
    let planned = runtime
        .prepare_query(context, &query, &tables, Some("native-seed-control"))
        .await
        .unwrap();
    let aggregate = aggregate(planned.physical_plan.as_ref());
    assert_eq!(*aggregate.mode(), AggregateMode::Single);
    assert_eq!(*aggregate.input_order_mode(), InputOrderMode::Linear);
    let expression = &aggregate.aggr_expr()[0];
    let mut current = expression.create_groups_accumulator().unwrap();
    current.update_batch(&[values], &[0, 0], None, 1).unwrap();
    let state = current.state(EmitTo::All).unwrap();
    assert_eq!(state.len(), 1);
    let actual = bits(&state[0]);
    let mut reseeded = expression.create_groups_accumulator().unwrap();
    reseeded.merge_batch(&state, &[0], None, 1).unwrap();
    let output = reseeded.evaluate(EmitTo::All).unwrap();
    let plain = bits(&output);
    let (winner, sentinel) = match (&dtype, maximum) {
        (DataType::Float32, false) => (
            u64::from(f32::INFINITY.to_bits()),
            u64::from(f32::MAX.to_bits()),
        ),
        (DataType::Float32, true) => (
            u64::from(f32::NEG_INFINITY.to_bits()),
            u64::from(f32::MIN.to_bits()),
        ),
        (DataType::Float64, false) => (f64::INFINITY.to_bits(), f64::MAX.to_bits()),
        (DataType::Float64, true) => (f64::NEG_INFINITY.to_bits(), f64::MIN.to_bits()),
        _ => unreachable!(),
    };
    assert_eq!(actual, winner);
    assert_eq!(plain, sentinel);
    assert_ne!(actual, plain);
    println!(
        "GROUPED_FLOAT_PLAIN_SEED dtype={dtype:?} maximum={maximum} winner={actual:x} reseed={plain:x}"
    );
}

#[tokio::test]
async fn test_grouped_float_plain_native_seed_loses_reachable_infinity_winner() {
    for dtype in [DataType::Float32, DataType::Float64] {
        for maximum in [false, true] {
            assert_plain_seed_mismatch(dtype.clone(), maximum).await;
        }
    }
}
