use super::*;
use datafusion::{
    arrow::{
        array::{ArrayRef, BooleanArray, Float32Array, Float64Array},
        compute::filter_record_batch,
        datatypes::{DataType, Field, Schema},
    },
    datasource::memory::{DataSourceExec, MemorySourceConfig},
    physical_expr::expressions::Column,
    physical_plan::{
        ExecutionPlan, InputOrderMode,
        aggregates::{AggregateExec, AggregateMode},
        projection::ProjectionExec,
    },
};

fn input(dtype: &DataType, lengths: &[usize]) -> Batch {
    let schema = Arc::new(Schema::new(vec![Field::new("value", dtype.clone(), true)]));
    let records = lengths
        .iter()
        .map(|length| {
            let values = (0..*length)
                .map(|row| (row % 7 != 0).then_some(1.0))
                .collect::<Vec<_>>();
            let column: ArrayRef = match dtype {
                DataType::Float32 => Arc::new(Float32Array::from(values)),
                DataType::Float64 => Arc::new(Float64Array::from(
                    values
                        .into_iter()
                        .map(|value| value.map(f64::from))
                        .collect::<Vec<_>>(),
                )),
                _ => unreachable!(),
            };
            RecordBatch::try_new(schema.clone(), vec![column]).unwrap()
        })
        .collect();
    Batch::table(records, BatchMetadata::default()).unwrap()
}

fn inspect(plan: &dyn ExecutionPlan, original: &Batch, census: &mut [usize; 2]) {
    assert_eq!(plan.output_partitioning().partition_count(), 1);
    if let Some(aggregate) = plan.downcast_ref::<AggregateExec>() {
        census[0] += 1;
        assert_eq!(*aggregate.mode(), AggregateMode::Single);
        assert_eq!(*aggregate.input_order_mode(), InputOrderMode::Linear);
        assert!(aggregate.group_expr().expr().is_empty());
        assert!(aggregate.group_expr().groups().is_empty());
        assert!(!aggregate.aggr_expr().is_empty());
        assert!(aggregate.aggr_expr().iter().all(|expression| matches!(
            expression.fun().name(),
            "sum" | "avg" | "count" | "min" | "max"
        )));
        assert!(aggregate.limit_options().is_none());
        assert!(aggregate.filter_expr().iter().all(|filter| {
            filter.as_ref().is_none_or(|filter| {
                filter.downcast_ref::<Column>().is_some()
                    && filter.data_type(&aggregate.input().schema()).unwrap() == DataType::Boolean
            })
        }));
    } else if let Some(source) = plan.downcast_ref::<DataSourceExec>() {
        census[1] += 1;
        let scan = source
            .data_source()
            .downcast_ref::<MemorySourceConfig>()
            .unwrap();
        let expected = original.table_payload().unwrap();
        assert_eq!(scan.partitions().len(), 1);
        assert!(scan.sort_information().is_empty());
        assert!(source.data_source().fetch().is_none());
        assert_eq!(scan.original_schema(), *expected.schema());
        assert_eq!(scan.partitions()[0].len(), expected.batches().len());
        for (actual, expected) in scan.partitions()[0].iter().zip(expected.batches()) {
            assert_eq!(actual.num_rows(), expected.num_rows());
            assert_eq!(actual.columns().len(), expected.columns().len());
            for (actual, expected) in actual.columns().iter().zip(expected.columns()) {
                assert!(Arc::ptr_eq(actual, expected));
            }
        }
    } else {
        assert!(
            plan.is::<ProjectionExec>(),
            "unexpected node {}",
            plan.name()
        );
    }
    for child in plan.children() {
        inspect(child.as_ref(), original, census);
    }
}

fn source(plan: &dyn ExecutionPlan) -> &DataSourceExec {
    if let Some(source) = plan.downcast_ref::<DataSourceExec>() {
        return source;
    }
    let children = plan.children();
    assert_eq!(children.len(), 1);
    source(children.into_iter().next().unwrap().as_ref())
}

fn aggregate(plan: &dyn ExecutionPlan) -> &AggregateExec {
    if let Some(aggregate) = plan.downcast_ref::<AggregateExec>() {
        return aggregate;
    }
    let children = plan.children();
    assert_eq!(children.len(), 1);
    aggregate(children.into_iter().next().unwrap().as_ref())
}

impl DataFusionRuntime {
    pub(crate) async fn global_record_accumulator_state(
        &self,
        query: &str,
        input: &Batch,
    ) -> Vec<datafusion::common::ScalarValue> {
        let _guard = self.query_lock.lock().await;
        let context = self.context_for_rows(input.num_rows(), None, "not_evaluated");
        let query = parse_select_query(query).unwrap();
        let tables = BTreeMap::from([("events".into(), input.clone())]);
        let planned = self
            .prepare_query(context, &query, &tables, Some("global-state-oracle"))
            .await
            .unwrap();
        let mut census = [0, 0];
        inspect(planned.physical_plan.as_ref(), input, &mut census);
        assert_eq!(census, [1, 1]);
        let aggregate = aggregate(planned.physical_plan.as_ref());
        let mut accumulators = aggregate
            .aggr_expr()
            .iter()
            .map(|expression| expression.create_accumulator().unwrap())
            .collect::<Vec<_>>();
        let mut stream = aggregate
            .input()
            .execute(0, planned.task_ctx.clone())
            .unwrap();
        while let Some(record) = stream.next().await {
            let record = record.unwrap();
            for ((expression, filter), accumulator) in aggregate
                .aggr_expr()
                .iter()
                .zip(aggregate.filter_expr())
                .zip(&mut accumulators)
            {
                let filtered;
                let record = if let Some(filter) = filter {
                    let mask = filter
                        .evaluate(&record)
                        .unwrap()
                        .into_array(record.num_rows())
                        .unwrap();
                    filtered = filter_record_batch(
                        &record,
                        mask.as_any().downcast_ref::<BooleanArray>().unwrap(),
                    )
                    .unwrap();
                    &filtered
                } else {
                    &record
                };
                let values = expression
                    .expressions()
                    .iter()
                    .map(|argument| {
                        argument
                            .evaluate(record)
                            .and_then(|value| value.into_array(record.num_rows()))
                            .unwrap()
                    })
                    .collect::<Vec<_>>();
                accumulator.update_batch(&values).unwrap();
            }
        }
        accumulators
            .iter_mut()
            .flat_map(|accumulator| accumulator.state().unwrap())
            .collect()
    }
}

#[tokio::test]
async fn test_global_float_executed_source_splits_each_original_record() {
    for dtype in [DataType::Float32, DataType::Float64] {
        for query in [
            "SELECT SUM(value) FROM events",
            "SELECT AVG(value) FROM events",
            "SELECT SUM(value), AVG(value) FROM events",
            "SELECT SUM(value), AVG(value), COUNT(value), COUNT(*) FROM events",
            "SELECT SUM(value), AVG(value), MIN(value), MAX(value), COUNT(value) FROM events",
            "SELECT AVG(value) AS mean, SUM(value) AS total, AVG(value) AS again FROM events",
        ] {
            let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
            let batch = input(&dtype, &[3, 8194, 1, 2]);
            let _guard = runtime.query_lock.lock().await;
            let context = runtime.context_for_rows(batch.num_rows(), None, "not_evaluated");
            let query = parse_select_query(query).unwrap();
            let tables = BTreeMap::from([("events".into(), batch.clone())]);
            let planned = runtime
                .prepare_query(context, &query, &tables, Some("global-source-split"))
                .await
                .unwrap();
            let mut census = [0, 0];
            inspect(planned.physical_plan.as_ref(), &batch, &mut census);
            assert_eq!(census, [1, 1]);
            let mut stream = source(planned.physical_plan.as_ref())
                .execute(0, planned.task_ctx.clone())
                .unwrap();
            let mut actual = Vec::new();
            while let Some(record) = stream.next().await {
                actual.push(record.unwrap());
            }
            assert_eq!(
                actual.iter().map(RecordBatch::num_rows).collect::<Vec<_>>(),
                [3, 8192, 2, 1, 2]
            );
            let expected = batch
                .table_payload()
                .unwrap()
                .batches()
                .iter()
                .flat_map(|record| {
                    (0..record.num_rows()).step_by(8192).map(move |offset| {
                        record.slice(offset, 8192.min(record.num_rows() - offset))
                    })
                })
                .collect::<Vec<_>>();
            assert_eq!(actual, expected);
        }
    }
}

#[tokio::test]
async fn test_global_float_default_single_plan_preserves_original_records() {
    for dtype in [DataType::Float32, DataType::Float64] {
        for query in [
            "SELECT SUM(value) FROM events",
            "SELECT AVG(value) FROM events",
            "SELECT SUM(value), AVG(value) FROM events",
            "SELECT SUM(value), AVG(value), COUNT(value), COUNT(*) FROM events",
            "SELECT SUM(value), AVG(value), MIN(value), MAX(value), COUNT(value) FROM events",
            "SELECT AVG(value) AS mean, SUM(value) AS total, AVG(value) AS again FROM events",
        ] {
            let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
            assert!(
                runtime
                    .grouped_float_model_supported("global-plan")
                    .unwrap()
            );
            for lengths in [
                vec![3],
                vec![3, 8194],
                vec![3, 8194, 1, 2],
                vec![100_003],
                vec![257; 391],
            ] {
                let batch = input(&dtype, &lengths);
                let _guard = runtime.query_lock.lock().await;
                let context = runtime.context_for_rows(batch.num_rows(), None, "not_evaluated");
                let query = parse_select_query(query).unwrap();
                let tables = BTreeMap::from([("events".into(), batch.clone())]);
                let planned = runtime
                    .prepare_query(context, &query, &tables, Some("global-plan"))
                    .await
                    .unwrap();
                let mut census = [0, 0];
                inspect(planned.physical_plan.as_ref(), &batch, &mut census);
                assert_eq!(census, [1, 1]);
            }
        }
    }
    let unknown = DataFusionRuntime::new(DataFusionConfig {
        target_partitions: 4,
        ..DataFusionConfig::default()
    })
    .unwrap();
    assert!(
        !unknown
            .grouped_float_model_supported("global-plan")
            .unwrap()
    );
}
