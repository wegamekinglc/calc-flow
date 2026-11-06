use std::{
    cell::Cell,
    collections::BTreeMap,
    fmt::{self, Display, Formatter},
    hash::{Hash, Hasher},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use allocation_counter::AllocationInfo;
use datafusion::{
    arrow::{
        array::{
            Array, ArrayRef, Decimal128Array, Float64Array, Int32Array, Int64Array, StringArray,
            UInt64Array,
        },
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    },
    common::{DataFusionError, ScalarValue},
    datasource::memory::{DataSourceExec, MemorySourceConfig},
    logical_expr::{ColumnarValue, Operator},
    physical_expr::{
        PhysicalExpr,
        expressions::{BinaryExpr, CaseExpr, CastExpr, Column, Literal, NegativeExpr},
        projection::Projector,
    },
    physical_plan::{ExecutionPlan, projection::ProjectionExec},
};
use futures::FutureExt;

use super::CompiledExpression;
use crate::{
    Batch, BatchMetadata, CalcFlowError, CancellationToken, DataFusionConfig, DataFusionRuntime,
    EdgeCollector, ExpressionOperator, JsonMap, OperatorMetadata, StreamJobContext, StreamOperator,
    StreamOperatorContext, UdfRegistry,
};

const ROWS: usize = 1024;

#[derive(Clone, Copy)]
enum Mask {
    Mixed,
    Buy,
    Sell,
}

struct Fixture {
    input: Batch,
    record: RecordBatch,
    expected: RecordBatch,
    compiled: CompiledExpression,
    reference: Projector,
    mask: Option<Mask>,
}

#[derive(Clone, Copy)]
struct Probe {
    compiled: AllocationInfo,
    reference: AllocationInfo,
}

fn quantity(row: usize) -> Option<i64> {
    if row.is_multiple_of(31) {
        None
    } else if row.is_multiple_of(29) {
        Some(i64::MIN)
    } else {
        Some(i64::try_from(row % 97).unwrap() - 48)
    }
}

fn side(row: usize, mask: Mask) -> Option<&'static str> {
    match mask {
        Mask::Buy => Some("buy"),
        Mask::Sell => Some("sell"),
        Mask::Mixed => match row % 7 {
            0 => None,
            1 | 3 | 5 => Some("buy"),
            _ => Some("sell"),
        },
    }
}

fn input_record(mask: Mask) -> RecordBatch {
    let schema = Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("account", DataType::Utf8, false)
                .with_metadata([("logical".into(), "account-id".into())].into()),
            Field::new("stock", DataType::Utf8, false),
            Field::new("quantity", DataType::Int64, true)
                .with_metadata([("unit".into(), "shares".into())].into()),
            Field::new("side", DataType::Utf8, true),
        ],
        [("probe".into(), "compiled-sign-case".into())].into(),
    ));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from_iter_values(
                (0..ROWS).map(|row| format!("account-{}", row % 100)),
            )),
            Arc::new(StringArray::from_iter_values(
                (0..ROWS).map(|row| format!("stock-{}", row % 100)),
            )),
            Arc::new(((0..ROWS).map(quantity)).collect::<Int64Array>()),
            Arc::new(((0..ROWS).map(|row| side(row, mask))).collect::<StringArray>()),
        ],
    )
    .unwrap()
}

fn projection(mask: Option<Mask>) -> Arc<dyn PhysicalExpr> {
    let quantity: Arc<dyn PhysicalExpr> = Arc::new(Column::new("quantity", 2));
    if mask.is_none() {
        return quantity;
    }
    let predicate = Arc::new(BinaryExpr::new(
        Arc::new(Column::new("side", 3)),
        Operator::Eq,
        Arc::new(Literal::new(ScalarValue::Utf8(Some("buy".into())))),
    ));
    Arc::new(
        CaseExpr::try_new(
            None,
            vec![(predicate, quantity.clone())],
            Some(Arc::new(NegativeExpr::new(quantity))),
        )
        .unwrap(),
    )
}

fn expected(record: &RecordBatch, mask: Option<Mask>) -> RecordBatch {
    let delta = if mask.is_some() {
        Field::new("delta", DataType::Int64, true)
    } else {
        record.schema().field(2).clone().with_name("delta")
    };
    let schema = Arc::new(Schema::new_with_metadata(
        vec![
            record.schema().field(0).clone().with_name("account_out"),
            record.schema().field(1).clone(),
            delta,
        ],
        record.schema().metadata().clone(),
    ));
    let values = (0..ROWS).map(|row| {
        let value = quantity(row);
        match mask {
            Some(mask) if side(row, mask) != Some("buy") => value.map(i64::wrapping_neg),
            _ => value,
        }
    });
    RecordBatch::try_new(
        schema,
        vec![
            record.column(0).clone(),
            record.column(1).clone(),
            Arc::new((values).collect::<Int64Array>()),
        ],
    )
    .unwrap()
}

fn fixture(mask: Option<Mask>) -> Fixture {
    let record = input_record(mask.unwrap_or(Mask::Mixed));
    let (compiled, reference) = projection_pair(&record, projection(mask));
    let metadata = BatchMetadata::new(
        "compiled-sign-case",
        71,
        BTreeMap::from([("probe".into(), serde_json::json!(true))]),
    )
    .unwrap();
    let input = Batch::table(vec![record.clone()], metadata).unwrap();
    Fixture {
        expected: expected(&record, mask),
        input,
        record,
        compiled,
        reference,
        mask,
    }
}

fn projection_pair(
    record: &RecordBatch,
    expression: Arc<dyn PhysicalExpr>,
) -> (CompiledExpression, Projector) {
    let source = MemorySourceConfig::try_new(&[vec![]], record.schema(), None).unwrap();
    let plan = ProjectionExec::try_new(
        vec![
            (
                Arc::new(Column::new("account", 0)) as Arc<dyn PhysicalExpr>,
                "account_out".into(),
            ),
            (Arc::new(Column::new("stock", 1)), "stock".into()),
            (expression, "delta".into()),
        ],
        DataSourceExec::from_data_source(source),
    )
    .unwrap();
    let reference = plan
        .projection_expr()
        .make_projector(&record.schema())
        .unwrap();
    let plan: Arc<dyn ExecutionPlan> = Arc::new(plan);
    let compiled = CompiledExpression::try_new(&plan).unwrap();
    (compiled, reference)
}

fn evaluate(fixture: &Fixture, context: &StreamOperatorContext<'_>) -> Batch {
    fixture
        .compiled
        .evaluate(&fixture.input, "compiled-sign-case", context)
        .now_or_never()
        .expect("one-record projection has no suspension point")
        .unwrap()
}

fn check_input(fixture: &Fixture) {
    let record = &fixture.input.table_payload().unwrap().batches()[0];
    for (actual, original) in record.columns().iter().zip(fixture.record.columns()) {
        assert!(Arc::ptr_eq(actual, original));
    }
    let accounts = record
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let stocks = record
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let quantities = record
        .column(2)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let sides = record
        .column(3)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    for row in 0..ROWS {
        assert_eq!(accounts.value(row), format!("account-{}", row % 100));
        assert_eq!(stocks.value(row), format!("stock-{}", row % 100));
        assert_eq!(quantities.is_null(row), quantity(row).is_none());
        if let Some(value) = quantity(row) {
            assert_eq!(quantities.value(row), value);
        }
        let expected = side(row, fixture.mask.unwrap_or(Mask::Mixed));
        assert_eq!(sides.is_null(row), expected.is_none());
        if let Some(value) = expected {
            assert_eq!(sides.value(row), value);
        }
    }
}

fn check_outputs(fixture: &Fixture, compiled: &Batch, reference: &RecordBatch) {
    let table = compiled.table_payload().unwrap();
    assert_eq!(compiled.metadata(), fixture.input.metadata());
    assert_eq!(table.schema(), &fixture.expected.schema());
    assert_eq!(table.batches(), std::slice::from_ref(&fixture.expected));
    assert_eq!(reference, &fixture.expected);
    assert_eq!(reference.schema(), fixture.expected.schema());
    for column in 0..2 {
        assert!(Arc::ptr_eq(
            table.batches()[0].column(column),
            fixture.record.column(column)
        ));
        assert!(Arc::ptr_eq(
            reference.column(column),
            fixture.record.column(column)
        ));
    }
    check_input(fixture);
}

fn check_teardown(info: AllocationInfo) {
    assert_eq!(
        info.count_current, 0,
        "owned allocations must settle: {info:?}"
    );
    assert_eq!(info.bytes_current, 0, "owned bytes must settle: {info:?}");
}

fn measure_projection(name: &str, mask: Option<Mask>) -> Probe {
    let fixture = fixture(mask);
    let job = StreamJobContext::new(
        1,
        "compiled-sign-case",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "compiled-sign-case", None);
    drop(evaluate(&fixture, &context));
    drop(fixture.reference.project_batch(&fixture.record).unwrap());

    let mut compiled = None;
    let compiled_heap = allocation_counter::measure(|| {
        compiled = Some(evaluate(&fixture, &context));
    });
    let mut reference = None;
    let reference_heap = allocation_counter::measure(|| {
        reference = Some(fixture.reference.project_batch(&fixture.record).unwrap());
    });
    let compiled = compiled.unwrap();
    let reference = reference.unwrap();
    check_outputs(&fixture, &compiled, &reference);
    let compiled_shared = Arc::ptr_eq(
        compiled.table_payload().unwrap().batches()[0].column(2),
        fixture.record.column(2),
    );
    let reference_shared = Arc::ptr_eq(reference.column(2), fixture.record.column(2));
    drop((compiled, reference));

    let compiled_lifecycle = allocation_counter::measure(|| {
        drop(std::hint::black_box(evaluate(&fixture, &context)));
    });
    let reference_lifecycle = allocation_counter::measure(|| {
        drop(std::hint::black_box(
            fixture.reference.project_batch(&fixture.record).unwrap(),
        ));
    });
    check_teardown(compiled_lifecycle);
    check_teardown(reference_lifecycle);
    check_input(&fixture);
    eprintln!(
        "compiled_sign_case case={name} rows={ROWS} oracle=pass compiled_held={compiled_heap:?} reference_held={reference_heap:?} compiled_lifecycle={compiled_lifecycle:?} reference_lifecycle={reference_lifecycle:?} compiled_shared={compiled_shared} reference_shared={reference_shared}"
    );
    Probe {
        compiled: compiled_heap,
        reference: reference_heap,
    }
}

#[test]
fn test_compiled_sign_case_mixed_projection_reduces_owned_allocation() {
    let cases = [
        ("mixed", Some(Mask::Mixed)),
        ("uniform-buy", Some(Mask::Buy)),
        ("uniform-sell", Some(Mask::Sell)),
        ("columns", None),
    ]
    .map(|(name, mask)| measure_projection(name, mask));
    let control = cases[3];
    assert!(
        control.compiled.bytes_total >= control.reference.bytes_total,
        "the unchanged column projection retains the compiled envelope cost"
    );
    let mixed = cases[0];
    assert!(
        mixed.compiled.bytes_total < mixed.reference.bytes_total,
        "the compiled mixed sign CASE must reduce actual owned bytes against the same existing DF projector: compiled={:?}, reference={:?}",
        mixed.compiled,
        mixed.reference
    );
}

thread_local! {
    static REWRITES: Cell<usize> = const { Cell::new(0) };
    static EVALUATIONS: Cell<usize> = const { Cell::new(0) };
}

pub(super) fn note_rewrite(count: usize) {
    REWRITES.set(REWRITES.get() + count);
}

fn literal_integer(expression: &Arc<dyn PhysicalExpr>) -> Option<i64> {
    match expression.downcast_ref::<Literal>()?.value() {
        ScalarValue::Int64(value) => *value,
        _ => None,
    }
}

fn sign_product(expression: &Arc<dyn PhysicalExpr>) -> bool {
    let Some(binary) = expression.downcast_ref::<BinaryExpr>() else {
        return false;
    };
    let Some(case) = binary.left().downcast_ref::<CaseExpr>() else {
        return false;
    };
    let [(_, then)] = case.when_then_expr() else {
        return false;
    };
    binary.op() == &Operator::Multiply
        && binary.right().is::<Column>()
        && literal_integer(then) == Some(1)
        && case.else_expr().and_then(literal_integer) == Some(-1)
}

pub(super) fn note_evaluation(projector: &Projector) {
    let count = projector
        .projection()
        .iter()
        .filter(|item| sign_product(&item.expr))
        .count();
    EVALUATIONS.set(EVALUATIONS.get() + count);
}

fn reset_trace() {
    REWRITES.set(0);
    EVALUATIONS.set(0);
}
fn trace() -> (usize, usize) {
    (REWRITES.get(), EVALUATIONS.get())
}

#[derive(Clone, Copy)]
enum Condition {
    Side,
    Scalar(Option<bool>),
}

fn selected_buy(condition: Condition, sides: &StringArray, row: usize) -> Option<bool> {
    match condition {
        Condition::Side => (!sides.is_null(row)).then(|| sides.value(row) == "buy"),
        Condition::Scalar(value) => value,
    }
}

fn predicate(scalar: Condition) -> Arc<dyn PhysicalExpr> {
    match scalar {
        Condition::Scalar(value) => Arc::new(Literal::new(ScalarValue::Boolean(value))),
        Condition::Side => Arc::new(BinaryExpr::new(
            Arc::new(Column::new("side", 3)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Utf8(Some("buy".into())))),
        )),
    }
}

fn sign_expression(condition: Arc<dyn PhysicalExpr>) -> Arc<dyn PhysicalExpr> {
    let column: Arc<dyn PhysicalExpr> = Arc::new(Column::new("quantity", 2));
    Arc::new(
        CaseExpr::try_new(
            None,
            vec![(condition, column.clone())],
            Some(Arc::new(NegativeExpr::new(column))),
        )
        .unwrap(),
    )
}

fn flat_buffers(record: &RecordBatch) -> Vec<Vec<Vec<u8>>> {
    record
        .columns()
        .iter()
        .map(|array| {
            let mut buffers = array
                .to_data()
                .buffers()
                .iter()
                .map(|buffer| buffer.as_slice().to_vec())
                .collect::<Vec<_>>();
            if let Some(nulls) = array.nulls() {
                buffers.push(nulls.inner().values().to_vec());
            }
            buffers
        })
        .collect()
}

fn compiled_root(compiled: &CompiledExpression) -> &Arc<dyn PhysicalExpr> {
    compiled
        .steps
        .iter()
        .rev()
        .find_map(|step| match step {
            super::Step::Project(projector) => {
                projector.projection().iter().last().map(|item| &item.expr)
            }
            _ => None,
        })
        .unwrap()
}

fn pair_result(
    record: &RecordBatch,
    expression: &Arc<dyn PhysicalExpr>,
    unchanged: bool,
) -> Option<RecordBatch> {
    let (compiled, reference) = projection_pair(record, Arc::clone(expression));
    if unchanged {
        assert!(Arc::ptr_eq(compiled_root(&compiled), expression));
    }
    let before = flat_buffers(record);
    let input = Batch::table(vec![record.clone()], BatchMetadata::default()).unwrap();
    let job = StreamJobContext::new(
        1,
        "case-semantics",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "case-semantics", None);
    let expected = reference.project_batch(record);
    let actual = compiled
        .evaluate(&input, "case-semantics", &context)
        .now_or_never()
        .unwrap();
    assert_eq!(flat_buffers(record), before);
    match (actual, expected) {
        (Ok(actual), Ok(expected)) => {
            assert_eq!(actual.metadata(), input.metadata());
            assert_eq!(actual.table_payload().unwrap().schema(), &expected.schema());
            assert_eq!(
                actual.table_payload().unwrap().batches(),
                std::slice::from_ref(&expected)
            );
            Some(expected)
        }
        (Err(actual), Err(expected)) => {
            assert_eq!(
                actual.to_string(),
                crate::datafusion::datafusion_error(Some("case-semantics"), expected).to_string()
            );
            None
        }
        _ => panic!("compiled and original DF projector must have identical outcomes"),
    }
}

#[test]
fn test_compiled_sign_case_scalars_slices_empty_and_extremes() {
    let original = input_record(Mask::Mixed);
    let mut values = original
        .column(2)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .iter()
        .collect::<Vec<_>>();
    values[1] = Some(i64::MAX);
    let original = replace_quantity(&original, Arc::new(Int64Array::from(values)));
    for record in [
        original.clone(),
        original.slice(3, 41),
        original.slice(9, 0),
    ] {
        for scalar in [
            Condition::Side,
            Condition::Scalar(Some(true)),
            Condition::Scalar(Some(false)),
            Condition::Scalar(None),
        ] {
            let result = pair_result(&record, &sign_expression(predicate(scalar)), false).unwrap();
            let quantities = record
                .column(2)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let sides = record
                .column(3)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let oracle = quantities
                .iter()
                .enumerate()
                .map(|(row, value)| {
                    let buy = selected_buy(scalar, sides, row);
                    if buy == Some(true) {
                        value
                    } else {
                        value.map(i64::wrapping_neg)
                    }
                })
                .collect::<Int64Array>();
            assert_eq!(result.column(2).as_ref(), &oracle as &dyn Array);
        }
    }
}

fn replace_quantity(record: &RecordBatch, array: ArrayRef) -> RecordBatch {
    let mut fields = record
        .schema()
        .fields()
        .iter()
        .map(|field| field.as_ref().clone())
        .collect::<Vec<_>>();
    fields[2] = fields[2].clone().with_data_type(array.data_type().clone());
    let schema = Arc::new(Schema::new_with_metadata(
        fields,
        record.schema().metadata().clone(),
    ));
    let mut columns = record.columns().to_vec();
    columns[2] = array;
    RecordBatch::try_new(schema, columns).unwrap()
}

fn fallback_expressions() -> Vec<Arc<dyn PhysicalExpr>> {
    let column: Arc<dyn PhysicalExpr> = Arc::new(Column::new("quantity", 2));
    let negative: Arc<dyn PhysicalExpr> = Arc::new(NegativeExpr::new(column.clone()));
    let cast: Arc<dyn PhysicalExpr> =
        Arc::new(CastExpr::new(column.clone(), DataType::Int64, None));
    vec![
        Arc::new(
            CaseExpr::try_new(
                None,
                vec![
                    (predicate(Condition::Side), column.clone()),
                    (predicate(Condition::Scalar(Some(true))), column.clone()),
                ],
                Some(negative.clone()),
            )
            .unwrap(),
        ),
        Arc::new(
            CaseExpr::try_new(
                Some(Arc::new(Column::new("side", 3))),
                vec![(
                    Arc::new(Literal::new(ScalarValue::Utf8(Some("buy".into())))),
                    column.clone(),
                )],
                Some(negative.clone()),
            )
            .unwrap(),
        ),
        Arc::new(
            CaseExpr::try_new(
                None,
                vec![(predicate(Condition::Side), negative.clone())],
                Some(column.clone()),
            )
            .unwrap(),
        ),
        Arc::new(
            CaseExpr::try_new(
                None,
                vec![(predicate(Condition::Side), cast.clone())],
                Some(Arc::new(NegativeExpr::new(cast))),
            )
            .unwrap(),
        ),
        Arc::new(
            CaseExpr::try_new(None, vec![(predicate(Condition::Side), column)], None).unwrap(),
        ),
    ]
}

#[test]
fn test_compiled_sign_case_unsupported_shapes_keep_original_projector() {
    let record = input_record(Mask::Mixed).slice(17, 23);
    for expression in fallback_expressions() {
        let _ = pair_result(&record, &expression, true);
    }
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(((0..23).map(|row| Some(row - 11))).collect::<Int32Array>()),
        Arc::new(
            (0..23)
                .map(|row| Some(f64::from(row) - 11.0))
                .collect::<Float64Array>(),
        ),
        Arc::new(((0_u64..23).map(Some)).collect::<UInt64Array>()),
        Arc::new(
            ((0..23).map(|row| Some(i128::from(row) - 11)))
                .collect::<Decimal128Array>()
                .with_precision_and_scale(10, 0)
                .unwrap(),
        ),
    ];
    for array in arrays {
        let _ = pair_result(
            &replace_quantity(&record, array),
            &sign_expression(predicate(Condition::Side)),
            true,
        );
    }
    let ill_typed: Arc<dyn PhysicalExpr> = Arc::new(Column::new("quantity", 2));
    let _ = pair_result(&record, &sign_expression(ill_typed), true);
    different_quantity_column(&record);
    inconsistent_column_name_is_rejected(&record);
    metadata_guards(&record);
}

fn inconsistent_column_name_is_rejected(record: &RecordBatch) {
    let column: Arc<dyn PhysicalExpr> = Arc::new(Column::new("quantity", 2));
    let expression: Arc<dyn PhysicalExpr> = Arc::new(
        CaseExpr::try_new(
            None,
            vec![(predicate(Condition::Side), column)],
            Some(Arc::new(NegativeExpr::new(Arc::new(Column::new(
                "other_name",
                2,
            ))))),
        )
        .unwrap(),
    );
    let source = MemorySourceConfig::try_new(&[vec![]], record.schema(), None).unwrap();
    let error = ProjectionExec::try_new(
        [(expression, "delta".into())],
        DataSourceExec::from_data_source(source),
    )
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("Input field name quantity does not match")
    );
}

fn different_quantity_column(record: &RecordBatch) {
    let mut fields = record
        .schema()
        .fields()
        .iter()
        .map(|field| field.as_ref().clone())
        .collect::<Vec<_>>();
    fields.push(Field::new("other", DataType::Int64, true));
    let schema = Arc::new(Schema::new_with_metadata(
        fields,
        record.schema().metadata().clone(),
    ));
    let mut columns = record.columns().to_vec();
    columns.push(Arc::new(Int64Array::from_iter_values(
        (0..record.num_rows()).map(|row| i64::try_from(row).unwrap() + 100),
    )));
    let record = RecordBatch::try_new(schema, columns).unwrap();
    let column: Arc<dyn PhysicalExpr> = Arc::new(Column::new("quantity", 2));
    let expression: Arc<dyn PhysicalExpr> = Arc::new(
        CaseExpr::try_new(
            None,
            vec![(predicate(Condition::Side), column)],
            Some(Arc::new(NegativeExpr::new(Arc::new(Column::new(
                "other", 4,
            ))))),
        )
        .unwrap(),
    );
    let _ = pair_result(&record, &expression, true);
}

fn metadata_guards(record: &RecordBatch) {
    let (_, original) = projection_pair(record, sign_expression(predicate(Condition::Side)));
    for field_metadata in [false, true] {
        let mut fields = original
            .output_schema()
            .fields()
            .iter()
            .map(|field| field.as_ref().clone())
            .collect::<Vec<_>>();
        let mut metadata = original.output_schema().metadata().clone();
        if field_metadata {
            fields[2] = fields[2]
                .clone()
                .with_metadata([("custom-result".into(), "retained".into())].into());
        } else {
            metadata.insert("custom-schema".into(), "retained".into());
        }
        let schema = Schema::new_with_metadata(fields, metadata);
        let projector = original
            .projection()
            .make_projector_with_schema_metadata(&record.schema(), &schema)
            .unwrap();
        let before = projector.output_schema().clone();
        assert!(super::sign_case::rewrite(&projector, &record.schema()).is_none());
        assert_eq!(projector.output_schema(), &before);
    }
}

#[derive(Debug)]
struct ObservedExpr {
    inner: Arc<dyn PhysicalExpr>,
    calls: Arc<AtomicUsize>,
}

impl PartialEq for ObservedExpr {
    fn eq(&self, other: &Self) -> bool {
        self.inner.eq(&other.inner) && Arc::ptr_eq(&self.calls, &other.calls)
    }
}
impl Eq for ObservedExpr {}
impl Hash for ObservedExpr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
        Arc::as_ptr(&self.calls).hash(state);
    }
}
impl Display for ObservedExpr {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.inner, formatter)
    }
}
impl PhysicalExpr for ObservedExpr {
    fn data_type(&self, schema: &Schema) -> datafusion::common::Result<DataType> {
        self.inner.data_type(schema)
    }
    fn nullable(&self, schema: &Schema) -> datafusion::common::Result<bool> {
        self.inner.nullable(schema)
    }
    fn evaluate(&self, record: &RecordBatch) -> datafusion::common::Result<ColumnarValue> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner.evaluate(record)
    }
    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        vec![&self.inner]
    }
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> datafusion::common::Result<Arc<dyn PhysicalExpr>> {
        let [inner] = children.as_slice() else {
            return Err(DataFusionError::Internal(
                "observer requires one child".into(),
            ));
        };
        Ok(Arc::new(Self {
            inner: inner.clone(),
            calls: self.calls.clone(),
        }))
    }
    fn fmt_sql(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        self.inner.fmt_sql(formatter)
    }
}

fn observed(inner: Arc<dyn PhysicalExpr>) -> (Arc<dyn PhysicalExpr>, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    (
        Arc::new(ObservedExpr {
            inner,
            calls: calls.clone(),
        }),
        calls,
    )
}

fn text_predicate_record(bad: bool) -> RecordBatch {
    let record = input_record(Mask::Mixed).slice(0, 23);
    let mut columns = record.columns().to_vec();
    columns[3] = Arc::new(
        ((0_usize..23).map(|row| {
            if bad && row == 7 {
                Some("invalid")
            } else if row.is_multiple_of(2) {
                Some("true")
            } else {
                Some("false")
            }
        }))
        .collect::<StringArray>(),
    );
    RecordBatch::try_new(record.schema(), columns).unwrap()
}

#[test]
fn test_compiled_sign_case_predicate_once_errors_retry_and_lazy_fallbacks() {
    let (condition, calls) = observed(Arc::new(CastExpr::new(
        Arc::new(Column::new("side", 3)),
        DataType::Boolean,
        None,
    )));
    let bad = text_predicate_record(true);
    let (compiled, reference) = projection_pair(&bad, sign_expression(condition));
    assert!(sign_product(compiled_root(&compiled)));
    let job = StreamJobContext::new(
        1,
        "predicate-once",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "predicate-once", None);
    for record in [bad, text_predicate_record(false)] {
        let before = flat_buffers(&record);
        calls.store(0, Ordering::SeqCst);
        let expected = reference.project_batch(&record);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        calls.store(0, Ordering::SeqCst);
        let input = Batch::table(vec![record.clone()], BatchMetadata::default()).unwrap();
        let actual = compiled
            .evaluate(&input, "predicate-once", &context)
            .now_or_never()
            .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        match (actual, expected) {
            (Err(actual), Err(expected)) => assert_eq!(
                actual.to_string(),
                crate::datafusion::datafusion_error(Some("predicate-once"), expected).to_string()
            ),
            (Ok(actual), Ok(expected)) => {
                assert_eq!(actual.table_payload().unwrap().batches(), &[expected]);
            }
            _ => panic!("predicate errors and retry must match original DF"),
        }
        assert_eq!(flat_buffers(&record), before);
    }
    lazy_fallbacks();
}

fn lazy_fallbacks() {
    let record = input_record(Mask::Mixed).slice(0, 23);
    let column: Arc<dyn PhysicalExpr> = Arc::new(Column::new("quantity", 2));
    let (callback, calls) = observed(Arc::new(NegativeExpr::new(column.clone())));
    let expressions = [
        callback,
        Arc::new(CastExpr::new(
            Arc::new(Column::new("side", 3)),
            DataType::Int64,
            None,
        )),
    ];
    for otherwise in expressions {
        let expression: Arc<dyn PhysicalExpr> = Arc::new(
            CaseExpr::try_new(
                None,
                vec![(predicate(Condition::Scalar(Some(true))), column.clone())],
                Some(otherwise),
            )
            .unwrap(),
        );
        calls.store(0, Ordering::SeqCst);
        let result = pair_result(&record, &expression, true).unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert_eq!(result.column(2), record.column(2));
    }
}

fn stock_record(shift: i64, reordered: bool) -> RecordBatch {
    stock_record_rows(shift, reordered, 64)
}

fn stock_record_rows(shift: i64, reordered: bool, rows: usize) -> RecordBatch {
    let schema = Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("account", DataType::Utf8, true)
                .with_metadata([("logical".into(), "account".into())].into()),
            Field::new("stock", DataType::Utf8, true),
            Field::new("quantity", DataType::Int64, true),
            Field::new("side", DataType::Utf8, true),
            Field::new("price", DataType::Float64, true),
            Field::new("flow_type", DataType::Utf8, true),
        ],
        [("schema-probe".into(), reordered.to_string())].into(),
    ));
    let record = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(
                ((0..rows)
                    .map(|row| (!row.is_multiple_of(13)).then(|| format!("account-{}", row % 7))))
                .collect::<StringArray>(),
            ),
            Arc::new(
                ((0..rows)
                    .map(|row| (!row.is_multiple_of(17)).then(|| format!("stock-{}", row % 11))))
                .collect::<StringArray>(),
            ),
            Arc::new(
                (0..rows)
                    .map(|row| quantity(row).map(|value| value.wrapping_add(shift)))
                    .collect::<Int64Array>(),
            ),
            Arc::new(
                (0..rows)
                    .map(|row| side(row, Mask::Mixed))
                    .collect::<StringArray>(),
            ),
            Arc::new(Float64Array::from_iter_values(
                (0..rows).map(|row| f64::from(u32::try_from(row).unwrap())),
            )),
            Arc::new(StringArray::from_iter_values((0..rows).map(|row| {
                if row.is_multiple_of(5) {
                    "cash"
                } else {
                    "trade"
                }
            }))),
        ],
    )
    .unwrap();
    if reordered {
        record.project(&[0, 1, 4, 2, 3, 5]).unwrap()
    } else {
        record
    }
}

fn stock_batch(shift: i64, reordered: bool) -> Batch {
    Batch::table(
        vec![stock_record(shift, reordered)],
        BatchMetadata::new("stock-case", u64::try_from(shift).unwrap(), BTreeMap::new()).unwrap(),
    )
    .unwrap()
}

fn string_column<'a>(record: &'a RecordBatch, name: &str) -> &'a StringArray {
    record
        .column_by_name(name)
        .unwrap()
        .as_any()
        .downcast_ref()
        .unwrap()
}

fn stock_oracle(record: &RecordBatch) -> Vec<ArrayRef> {
    let flow = string_column(record, "flow_type");
    let account = string_column(record, "account");
    let stock = string_column(record, "stock");
    let sides = string_column(record, "side");
    let quantities = record
        .column_by_name("quantity")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let selected = (0..record.num_rows())
        .filter(|&row| flow.value(row) == "trade")
        .collect::<Vec<_>>();
    let optional = |array: &StringArray, row: usize| {
        (!array.is_null(row)).then(|| array.value(row).to_owned())
    };
    let accounts = (selected.iter().map(|&row| optional(account, row))).collect::<StringArray>();
    let stocks = (selected.iter().map(|&row| optional(stock, row))).collect::<StringArray>();
    let deltas = (selected.iter().map(|&row| {
        let value = (!quantities.is_null(row)).then(|| quantities.value(row));
        if !sides.is_null(row) && sides.value(row) == "buy" {
            value
        } else {
            value.map(i64::wrapping_neg)
        }
    }))
    .collect::<Int64Array>();
    vec![Arc::new(accounts), Arc::new(stocks), Arc::new(deltas)]
}

async fn collect_stock(
    operator: &mut ExpressionOperator,
    input: Batch,
    job: &StreamJobContext,
) -> crate::Result<Batch> {
    let context = StreamOperatorContext::new(job, "stock-case", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let result = operator
        .process_data("input", input, &context, &mut output)
        .await;
    if let Err(error) = result {
        assert!(output.drain("output").is_empty());
        return Err(error);
    }
    let messages = output.drain("output");
    assert_eq!(messages.len(), 1);
    Ok(messages[0].as_data().unwrap().clone())
}

fn stock_job(token: CancellationToken) -> StreamJobContext {
    StreamJobContext::new(1, "stock-case", JsonMap::new(), None, token)
}

#[tokio::test(flavor = "current_thread")]
async fn test_compiled_sign_case_actual_stock_cached_path_and_lifecycle() {
    let config = DataFusionConfig {
        target_partitions: 1,
        min_rows_per_partition: 1,
        ..DataFusionConfig::default()
    };
    let mut operator = ExpressionOperator::new(
        "stock-case",
        "",
        vec![
            "account".into(),
            "stock".into(),
            "CASE WHEN side = 'buy' THEN quantity ELSE -quantity END AS delta".into(),
        ],
        Some("flow_type = 'trade'".into()),
        vec![],
    )
    .unwrap();
    operator.set_stream_resources(config, UdfRegistry::new().snapshot(), vec![]);
    let reference = DataFusionRuntime::new(config).unwrap();
    let job = stock_job(CancellationToken::new());
    let mut held = Vec::new();
    for (index, reordered) in [false, false, true, true].into_iter().enumerate() {
        let input = stock_batch(i64::try_from(index * 100).unwrap(), reordered);
        let record = &input.table_payload().unwrap().batches()[0];
        let before = flat_buffers(record);
        let oracle = stock_oracle(record);
        let expected = reference.sql("SELECT account, stock, CASE WHEN side = 'buy' THEN quantity ELSE -quantity END AS delta FROM input WHERE flow_type = 'trade'", &BTreeMap::from([("input".into(), input.clone())]), Some("stock-case")).await.unwrap();
        reset_trace();
        let actual = collect_stock(&mut operator, input.clone(), &job)
            .await
            .unwrap();
        assert_eq!(
            trace(),
            if index.is_multiple_of(2) {
                (0, 0)
            } else {
                (1, 1)
            }
        );
        assert_eq!(actual.metadata(), expected.metadata());
        assert_eq!(
            actual.table_payload().unwrap().schema(),
            expected.table_payload().unwrap().schema()
        );
        assert_eq!(
            actual.table_payload().unwrap().batches(),
            expected.table_payload().unwrap().batches()
        );
        let oracle =
            RecordBatch::try_new(expected.table_payload().unwrap().schema().clone(), oracle)
                .unwrap();
        assert_eq!(
            actual.table_payload().unwrap().batches(),
            std::slice::from_ref(&oracle)
        );
        assert_eq!(flat_buffers(record), before);
        held.push((actual, oracle));
    }
    for abandon in [false, true] {
        cancel_stock_call(&mut operator, abandon).await;
    }
    for (actual, oracle) in held {
        assert_eq!(actual.table_payload().unwrap().batches(), &[oracle]);
    }
}

async fn cancel_stock_call(operator: &mut ExpressionOperator, abandon: bool) {
    let input = Batch::table(
        vec![
            stock_record_rows(400, true, 10_240),
            stock_record(500, true),
        ],
        BatchMetadata::default(),
    )
    .unwrap();
    let owners = input
        .table_payload()
        .unwrap()
        .batches()
        .iter()
        .flat_map(RecordBatch::columns)
        .flat_map(|array| array.to_data().buffers().to_vec())
        .filter(|buffer| !buffer.is_empty())
        .collect::<Vec<_>>();
    let token = CancellationToken::new();
    let job = stock_job(token.clone());
    let context = StreamOperatorContext::new(&job, "stock-case", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let mut future = Box::pin(operator.process_data("input", input, &context, &mut output));
    reset_trace();
    assert!(matches!(
        futures::poll!(future.as_mut()),
        std::task::Poll::Pending
    ));
    assert_eq!(trace(), (0, 1));
    if !abandon {
        token.cancel();
        assert!(matches!(
            future.as_mut().await,
            Err(CalcFlowError::Cancelled { .. })
        ));
    }
    drop(future);
    assert!(output.drain("output").is_empty());
    assert!(owners.iter().all(|buffer| buffer.strong_count() == 1));
    let retry = stock_batch(600, true);
    let oracle = stock_oracle(&retry.table_payload().unwrap().batches()[0]);
    let actual = collect_stock(operator, retry, &stock_job(CancellationToken::new()))
        .await
        .unwrap();
    let expected =
        RecordBatch::try_new(actual.table_payload().unwrap().schema().clone(), oracle).unwrap();
    assert_eq!(actual.table_payload().unwrap().batches(), &[expected]);
}
