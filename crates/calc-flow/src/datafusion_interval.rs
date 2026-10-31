//! Calendar-free fast path for `timestamp ± Interval(MonthDayNano)` SQL
//! arithmetic.
//!
//! Arrow 59 evaluates this expression with
//! [`arrow::compute::kernels::numeric::add_wrapping`] (or `sub_wrapping`),
//! which per row converts the raw value to a chrono `DateTime<Tz>` by
//! relabeling the UTC instant (it never resolves local time, so daylight-saving
//! gaps and folds cannot fail the conversion), applies the calendar month and
//! day adds, adds the nanoseconds as an exact instant duration, and
//! reconstructs the raw value. Any stage that cannot represent the value — an
//! operand outside chrono's range for the unit, a duration past the chrono
//! range, or a reconstructed value past the unit's i64 range — returns `None`,
//! which surfaces as `ComputeError("Timestamp out of range")`; for units
//! coarser than nanoseconds the reconstruction also floors sub-unit
//! nanoseconds. The fast path in this module therefore only accepts batches
//! where integer addition provably reproduces that kernel, and defers every
//! other batch to it unchanged.

use std::sync::{Arc, LazyLock};

use chrono::NaiveDateTime;
use datafusion::arrow::array::Array as _;
use datafusion::arrow::{
    array::{
        ArrayRef, PrimitiveArray, Scalar,
        cast::AsArray,
        types::{
            ArrowTimestampType, IntervalMonthDayNanoType, TimestampMicrosecondType,
            TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType,
        },
    },
    compute::kernels::numeric::{add_wrapping, sub_wrapping},
    datatypes::{DataType, Field, FieldRef, IntervalUnit, TimeUnit},
};
use datafusion::common::{
    DFSchema, DataFusionError, Result, ScalarValue, plan_err,
    tree_node::{Transformed, TreeNode},
};
use datafusion::logical_expr::{
    ColumnarValue, Expr, ExprSchemable, LogicalPlan, Operator, ReturnFieldArgs, ScalarFunctionArgs,
    ScalarUDF, ScalarUDFImpl, Signature, Volatility,
};

/// Nanoseconds in one unit of `unit`.
const fn nanos_per_unit(unit: TimeUnit) -> i64 {
    match unit {
        TimeUnit::Second => 1_000_000_000,
        TimeUnit::Millisecond => 1_000_000,
        TimeUnit::Microsecond => 1_000,
        TimeUnit::Nanosecond => 1,
    }
}

/// Units of `unit` in one second; the inverse of [`nanos_per_unit`].
const fn units_per_second(unit: TimeUnit) -> i64 {
    match unit {
        TimeUnit::Second => 1,
        TimeUnit::Millisecond => 1_000,
        TimeUnit::Microsecond => 1_000_000,
        TimeUnit::Nanosecond => 1_000_000_000,
    }
}

/// Raw-value bounds outside which arrow's calendar kernel can no longer
/// round-trip a timestamp through `chrono::NaiveDateTime`, so raw integer
/// arithmetic would diverge from it. Values inside the bounds convert and
/// reconstruct exactly; values outside make the fallback error, so the fast
/// path declines them and defers to arrow.
pub(super) fn representable_bounds(unit: TimeUnit) -> (i64, i64) {
    if matches!(unit, TimeUnit::Nanosecond) {
        // Every i64 nanosecond timestamp lies inside chrono's range, so the
        // i64 arithmetic itself is the only limit.
        return (i64::MIN, i64::MAX);
    }
    let scale = units_per_second(unit);
    let min = NaiveDateTime::MIN.and_utc().timestamp() * scale;
    let max = NaiveDateTime::MAX.and_utc().timestamp() * scale + (scale - 1);
    (min, max)
}

/// Evaluates `timestamp ± Interval(MonthDayNano)`, replacing arrow's
/// per-element timezone-aware calendar kernel with raw checked integer
/// addition whenever the result is provably identical: every interval must be
/// pure nanoseconds (months == 0, days == 0) divisible by the timestamp unit,
/// and both operands must sit inside [`representable_bounds`]. Any other
/// batch, and every overflow, falls back to the same arrow kernel `DataFusion`
/// uses, so values and errors match by construction.
pub(super) fn evaluate_timestamp_interval(
    timestamps: &ColumnarValue,
    intervals: &ColumnarValue,
    subtract: bool,
) -> Result<ColumnarValue> {
    match timestamps.data_type() {
        DataType::Timestamp(TimeUnit::Second, _) => {
            evaluate_typed::<TimestampSecondType>(timestamps, intervals, subtract)
        }
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            evaluate_typed::<TimestampMillisecondType>(timestamps, intervals, subtract)
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            evaluate_typed::<TimestampMicrosecondType>(timestamps, intervals, subtract)
        }
        DataType::Timestamp(TimeUnit::Nanosecond, _) => {
            evaluate_typed::<TimestampNanosecondType>(timestamps, intervals, subtract)
        }
        other => plan_err!(
            "calc_flow timestamp interval arithmetic expects a timestamp operand, got {other}"
        ),
    }
}

fn evaluate_typed<T: ArrowTimestampType>(
    timestamps: &ColumnarValue,
    intervals: &ColumnarValue,
    subtract: bool,
) -> Result<ColumnarValue> {
    match (timestamps, intervals) {
        (ColumnarValue::Array(ts), ColumnarValue::Array(iv)) => evaluate_array_pair(
            ts.as_primitive::<T>(),
            iv.as_primitive::<IntervalMonthDayNanoType>(),
            false,
            subtract,
        ),
        (ColumnarValue::Array(ts), ColumnarValue::Scalar(iv)) => evaluate_array_pair(
            ts.as_primitive::<T>(),
            &(iv.to_array()?
                .as_primitive::<IntervalMonthDayNanoType>()
                .clone()),
            true,
            subtract,
        ),
        (ColumnarValue::Scalar(ts), ColumnarValue::Array(iv)) => {
            let scalar = ts.to_array()?;
            evaluate_scalar_timestamp(
                scalar.as_primitive::<T>(),
                iv.as_primitive::<IntervalMonthDayNanoType>(),
                subtract,
            )
        }
        (ColumnarValue::Scalar(ts), ColumnarValue::Scalar(iv)) => {
            evaluate_scalar_pair::<T>(&ts.to_array()?, &iv.to_array()?, subtract)
        }
    }
}

/// Array timestamps against an array or one-element broadcast interval.
fn evaluate_array_pair<T: ArrowTimestampType>(
    ts: &PrimitiveArray<T>,
    iv: &PrimitiveArray<IntervalMonthDayNanoType>,
    iv_broadcast: bool,
    subtract: bool,
) -> Result<ColumnarValue> {
    let len = if iv_broadcast { ts.len() } else { iv.len() };
    Ok(ColumnarValue::Array(evaluate_arrays(
        ts,
        iv,
        false,
        iv_broadcast,
        len,
        subtract,
    )?))
}

/// A one-element timestamp broadcast against array intervals.
fn evaluate_scalar_timestamp<T: ArrowTimestampType>(
    ts: &PrimitiveArray<T>,
    iv: &PrimitiveArray<IntervalMonthDayNanoType>,
    subtract: bool,
) -> Result<ColumnarValue> {
    Ok(ColumnarValue::Array(evaluate_arrays(
        ts,
        iv,
        true,
        false,
        iv.len(),
        subtract,
    )?))
}

/// Two scalars evaluate to one row, staying scalar-valued.
fn evaluate_scalar_pair<T: ArrowTimestampType>(
    ts: &ArrayRef,
    iv: &ArrayRef,
    subtract: bool,
) -> Result<ColumnarValue> {
    let result = evaluate_arrays(
        ts.as_primitive::<T>(),
        iv.as_primitive::<IntervalMonthDayNanoType>(),
        true,
        true,
        1,
        subtract,
    )?;
    Ok(ColumnarValue::Scalar(ScalarValue::try_from_array(
        result.as_ref(),
        0,
    )?))
}

/// Shared array-level evaluation. `ts_broadcast`/`iv_broadcast` mark a
/// one-element operand that repeats for every row, mirroring arrow's scalar
/// `Datum` broadcasting.
fn evaluate_arrays<T: ArrowTimestampType>(
    ts: &PrimitiveArray<T>,
    iv: &PrimitiveArray<IntervalMonthDayNanoType>,
    ts_broadcast: bool,
    iv_broadcast: bool,
    len: usize,
    subtract: bool,
) -> Result<ArrayRef> {
    if let Some(fast) = fast_shift(ts, iv, ts_broadcast, iv_broadcast, len, subtract) {
        return Ok(Arc::new(fast) as ArrayRef);
    }
    let kernel = if subtract { sub_wrapping } else { add_wrapping };
    let fallback = if ts_broadcast && !iv_broadcast {
        kernel(&Scalar::new(ts.clone()), iv)
    } else if !ts_broadcast && iv_broadcast {
        kernel(ts, &Scalar::new(iv.clone()))
    } else {
        kernel(ts, iv)
    };
    fallback.map_err(|error| DataFusionError::ArrowError(Box::new(error), None))
}

fn fast_shift<T: ArrowTimestampType>(
    ts: &PrimitiveArray<T>,
    iv: &PrimitiveArray<IntervalMonthDayNanoType>,
    ts_broadcast: bool,
    iv_broadcast: bool,
    len: usize,
    subtract: bool,
) -> Option<PrimitiveArray<T>> {
    let scale = nanos_per_unit(T::UNIT);
    let (min, max) = representable_bounds(T::UNIT);
    fast_path_batch_admissible(ts)?;
    if !ts_broadcast && !iv_broadcast && ts.len() != iv.len() {
        return None;
    }
    let values = fill_shifted(
        ts,
        iv,
        (ts_broadcast, iv_broadcast),
        len,
        (scale, (min, max)),
        subtract,
    )?;
    Some(PrimitiveArray::<T>::from_iter(values).with_timezone_opt(ts.timezone()))
}

/// One output slot per row: null where either operand is null, otherwise
/// the checked fast shift, or `None` to decline the whole batch.
fn fill_shifted<T: ArrowTimestampType>(
    ts: &PrimitiveArray<T>,
    iv: &PrimitiveArray<IntervalMonthDayNanoType>,
    broadcast: (bool, bool),
    len: usize,
    shift: (i64, (i64, i64)),
    subtract: bool,
) -> Option<Vec<Option<i64>>> {
    let (ts_broadcast, iv_broadcast) = broadcast;
    let (scale, bounds) = shift;
    let mut values = Vec::with_capacity(len);
    for row in 0..len {
        let ts_index = if ts_broadcast { 0 } else { row };
        let iv_index = if iv_broadcast { 0 } else { row };
        if ts.is_null(ts_index) || iv.is_null(iv_index) {
            values.push(None);
            continue;
        }
        let shifted = shifted_value(
            ts.value(ts_index),
            iv.value(iv_index),
            scale,
            bounds,
            subtract,
        );
        values.push(Some(shifted?));
    }
    Some(values)
}

/// Shift one non-null timestamp by a pure-nanosecond-divisible interval,
/// declining (`None`) when the row must fall back to the calendar kernel.
fn shifted_value(
    timestamp: i64,
    interval: datafusion::arrow::datatypes::IntervalMonthDayNano,
    scale: i64,
    (min, max): (i64, i64),
    subtract: bool,
) -> Option<i64> {
    let units = whole_units(interval, scale, subtract)?;
    let shifted = timestamp.checked_add(units)?;
    both_within_bounds(timestamp, shifted, min, max).then_some(shifted)
}

/// Whole-unit shift for a pure-nanosecond interval, or `None` when calendar
/// components or an indivisible remainder force the fallback kernel.
fn whole_units(
    interval: datafusion::arrow::datatypes::IntervalMonthDayNano,
    scale: i64,
    subtract: bool,
) -> Option<i64> {
    if interval.months != 0 || interval.days != 0 {
        return None;
    }
    if interval.nanoseconds % scale != 0 {
        return None;
    }
    let units = interval.nanoseconds / scale;
    if subtract {
        units.checked_neg()
    } else {
        Some(units)
    }
}

/// Both the input and shifted timestamps must stay inside the range arrow's
/// kernel can represent for the unit.
fn both_within_bounds(timestamp: i64, shifted: i64, min: i64, max: i64) -> bool {
    timestamp >= min && timestamp <= max && shifted >= min && shifted <= max
}

/// Arrow's kernel parses the timezone up front and fails the whole batch on
/// a malformed offset string; parse once here so the fast path keeps that
/// contract instead of silently carrying a bogus zone through (declining
/// routes the batch to the fallback, which reproduces the parse error).
fn fast_path_batch_admissible<T: ArrowTimestampType>(ts: &PrimitiveArray<T>) -> Option<()> {
    if let Some(zone) = ts.timezone() {
        zone.parse::<datafusion::arrow::array::timezone::Tz>()
            .ok()?;
    }
    Some(())
}

/// Crate-internal names of the add and subtract interval UDFs.
#[cfg(test)]
pub(super) fn interval_udf_names() -> (&'static str, &'static str) {
    (TimestampIntervalAdd::NAME, TimestampIntervalSubtract::NAME)
}

fn interval_add_udf() -> &'static Arc<ScalarUDF> {
    static UDF: LazyLock<Arc<ScalarUDF>> =
        LazyLock::new(|| Arc::new(ScalarUDF::from(TimestampIntervalAdd)));
    &UDF
}

fn interval_subtract_udf() -> &'static Arc<ScalarUDF> {
    static UDF: LazyLock<Arc<ScalarUDF>> =
        LazyLock::new(|| Arc::new(ScalarUDF::from(TimestampIntervalSubtract)));
    &UDF
}

static INTERVAL_SIGNATURE: LazyLock<Signature> =
    LazyLock::new(|| Signature::any(2, Volatility::Immutable));

/// Rewrites `timestamp ± interval_month_daynano` binary expressions into the
/// crate-internal fast-path UDFs, which replace the timezone-aware calendar
/// kernel with raw integer arithmetic whenever that is provably identical.
#[derive(Debug)]
pub(crate) struct TimestampIntervalArithmetic;

#[derive(Debug, PartialEq, Eq, Hash)]
struct TimestampIntervalAdd;

#[derive(Debug, PartialEq, Eq, Hash)]
struct TimestampIntervalSubtract;

macro_rules! impl_interval_udf {
    ($udf:ty, $name:expr, $subtract:expr) => {
        impl $udf {
            const NAME: &'static str = $name;
            const SUBTRACT: bool = $subtract;
        }

        impl ScalarUDFImpl for $udf {
            fn name(&self) -> &str {
                Self::NAME
            }

            fn signature(&self) -> &Signature {
                &INTERVAL_SIGNATURE
            }

            fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
                match arg_types {
                    [
                        DataType::Timestamp(unit, timezone),
                        DataType::Interval(IntervalUnit::MonthDayNano),
                    ] => Ok(DataType::Timestamp(unit.clone(), timezone.clone())),
                    _ => plan_err!(
                        "{} expects timestamp and Interval(MonthDayNano) operands",
                        Self::NAME
                    ),
                }
            }

            fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
                let [timestamp, interval] = args.arg_fields else {
                    return plan_err!(
                        "{} expects (timestamp, interval) arguments",
                        Self::NAME
                    );
                };
                match (timestamp.data_type(), interval.data_type()) {
                    (
                        DataType::Timestamp(unit, timezone),
                        DataType::Interval(IntervalUnit::MonthDayNano),
                    ) => Ok(Arc::new(Field::new(
                        Self::NAME,
                        DataType::Timestamp(unit.clone(), timezone.clone()),
                        timestamp.is_nullable() || interval.is_nullable(),
                    ))),
                    (left, right) => plan_err!(
                        "{} expects timestamp and Interval(MonthDayNano) operands, got {left} and {right}",
                        Self::NAME
                    ),
                }
            }

            fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
                match args.args.as_slice() {
                    [timestamp, interval] => {
                        evaluate_timestamp_interval(timestamp, interval, Self::SUBTRACT)
                    }
                    _ => plan_err!(
                        "{} expects (timestamp, interval) arguments",
                        Self::NAME
                    ),
                }
            }
        }
    };
}

impl_interval_udf!(TimestampIntervalAdd, "calc_flow_ts_interval_add", false);
impl_interval_udf!(
    TimestampIntervalSubtract,
    "calc_flow_ts_interval_subtract",
    true
);

impl datafusion::optimizer::OptimizerRule for TimestampIntervalArithmetic {
    fn name(&self) -> &'static str {
        "calc_flow_timestamp_interval_arithmetic"
    }

    fn apply_order(&self) -> Option<datafusion::optimizer::optimizer::ApplyOrder> {
        Some(datafusion::optimizer::optimizer::ApplyOrder::BottomUp)
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn datafusion::optimizer::OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        match plan {
            LogicalPlan::Filter(filter) => rewrite_filter(filter),
            LogicalPlan::Projection(projection) => {
                let schema = Arc::clone(projection.input.schema());
                let mut transformed = false;
                let exprs = projection
                    .expr
                    .clone()
                    .into_iter()
                    .map(|expression| {
                        let original = expression.schema_name().to_string();
                        let rewritten = expression
                            .transform_up(|nested| {
                                Ok(match specialize_interval_arithmetic(&nested, &schema) {
                                    Some(replacement) => {
                                        transformed = true;
                                        Transformed::yes(replacement)
                                    }
                                    None => Transformed::no(nested),
                                })
                            })?
                            .data;
                        let name = rewritten.schema_name().to_string();
                        if name == original {
                            Ok(rewritten)
                        } else {
                            // Rewrites deeper in the expression would change
                            // the projection's derived column name; alias back
                            // to the original so output schemas are stable.
                            Ok(rewritten.alias(original))
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;
                if !transformed {
                    // Leave plans the classifier machinery inspects exactly as
                    // the analyzer produced them when nothing was rewritten.
                    return Ok(Transformed::no(LogicalPlan::Projection(projection)));
                }
                let input = projection.input;
                let projection =
                    datafusion::logical_expr::logical_plan::Projection::try_new(exprs, input)?;
                Ok(Transformed::yes(LogicalPlan::Projection(projection)))
            }
            other => Ok(Transformed::no(other)),
        }
    }
}

/// Returns the (timestamp, interval) operands in evaluation order, or `None`
/// when the expression is not a supported timestamp/interval pair.
fn classify_operands<'a>(
    binary: &'a datafusion::logical_expr::expr::BinaryExpr,
    schema: &DFSchema,
    subtract: bool,
) -> Option<(&'a Expr, &'a Expr)> {
    let left_type = binary.left.get_type(schema).ok()?;
    let right_type = binary.right.get_type(schema).ok()?;
    match (&left_type, &right_type) {
        (DataType::Timestamp(_, _), DataType::Interval(IntervalUnit::MonthDayNano)) => {
            Some((binary.left.as_ref(), binary.right.as_ref()))
        }
        // arrow evaluates mirrored `interval + timestamp` by swapping the
        // commutative operands; mirrored subtraction is not supported.
        (DataType::Interval(IntervalUnit::MonthDayNano), DataType::Timestamp(_, _))
            if !subtract =>
        {
            Some((binary.right.as_ref(), binary.left.as_ref()))
        }
        _ => None,
    }
}

/// Rewrites every interval arithmetic in a filter predicate.
fn rewrite_filter(
    mut filter: datafusion::logical_expr::logical_plan::Filter,
) -> Result<Transformed<LogicalPlan>> {
    let schema = Arc::clone(filter.input.schema());
    let predicate = filter.predicate.transform_up(|expression| {
        Ok(match specialize_interval_arithmetic(&expression, &schema) {
            Some(replacement) => Transformed::yes(replacement),
            None => Transformed::no(expression),
        })
    })?;
    filter.predicate = predicate.data;
    Ok(if predicate.transformed {
        Transformed::yes(LogicalPlan::Filter(filter))
    } else {
        Transformed::no(LogicalPlan::Filter(filter))
    })
}

/// Rebuilds one `timestamp ± interval_month_daynano` expression as the UDF
/// call, or returns `None` when the expression is not that shape.
fn specialize_interval_arithmetic(expression: &Expr, schema: &DFSchema) -> Option<Expr> {
    let Expr::BinaryExpr(binary) = expression else {
        return None;
    };
    let subtract = match binary.op {
        Operator::Plus => false,
        Operator::Minus => true,
        _ => return None,
    };
    let (timestamp, interval) = classify_operands(binary, schema, subtract)?;
    let DataType::Timestamp(unit, _) = timestamp.get_type(schema).ok()? else {
        return None;
    };
    if hopeless_literal(interval, unit) {
        return None;
    }
    let udf = if subtract {
        interval_subtract_udf()
    } else {
        interval_add_udf()
    };
    Some(udf.call(vec![timestamp.clone(), interval.clone()]))
}

/// A literal interval that can never take the fast path keeps the plain
/// binary expression instead of paying UDF dispatch just to fall back.
fn hopeless_literal(interval: &Expr, unit: TimeUnit) -> bool {
    let Expr::Literal(ScalarValue::IntervalMonthDayNano(Some(value)), _) = interval else {
        return false;
    };
    let scale = nanos_per_unit(unit);
    value.months != 0 || value.days != 0 || value.nanoseconds % scale != 0
}

#[cfg(test)]
mod tests {
    use super::{TimeUnit, representable_bounds};
    use std::sync::Arc;

    use chrono::{DateTime, Months, NaiveDateTime};
    use datafusion::arrow::{
        array::{
            Array, ArrayRef, PrimitiveArray, Scalar,
            cast::AsArray,
            types::{
                ArrowTimestampType, IntervalMonthDayNano as IntervalValue,
                IntervalMonthDayNanoType, TimestampMicrosecondType, TimestampMillisecondType,
                TimestampNanosecondType, TimestampSecondType,
            },
        },
        compute::kernels::numeric::{add_wrapping, sub_wrapping},
    };
    use datafusion::common::ScalarValue;
    use datafusion::logical_expr::ColumnarValue;

    use super::evaluate_timestamp_interval;

    fn ts_array<T: ArrowTimestampType>(
        timezone: Option<&str>,
        values: Vec<Option<i64>>,
    ) -> PrimitiveArray<T> {
        match timezone {
            Some(timezone) => PrimitiveArray::<T>::from_iter(values).with_timezone(timezone),
            None => PrimitiveArray::<T>::from_iter(values),
        }
    }

    fn iv_array(values: Vec<Option<IntervalValue>>) -> PrimitiveArray<IntervalMonthDayNanoType> {
        PrimitiveArray::from_iter(values)
    }

    #[allow(
        clippy::unnecessary_wraps,
        reason = "fixtures mix Some offsets with None slots"
    )]
    fn mdn(months: i32, days: i32, nanoseconds: i64) -> Option<IntervalValue> {
        Some(IntervalValue::new(months, days, nanoseconds))
    }

    fn evaluated(
        timestamps: ArrayRef,
        intervals: ArrayRef,
        subtract: bool,
    ) -> datafusion::common::Result<ColumnarValue> {
        evaluate_timestamp_interval(
            &ColumnarValue::Array(timestamps),
            &ColumnarValue::Array(intervals),
            subtract,
        )
    }

    fn array(columnar: datafusion::common::Result<ColumnarValue>) -> ArrayRef {
        match columnar.expect("interval arithmetic evaluates") {
            ColumnarValue::Array(array) => array,
            ColumnarValue::Scalar(_) => panic!("expected array-valued result"),
        }
    }

    #[test]
    fn timestamp_interval_non_divisible_nanoseconds_match_arrow_floor_semantics() {
        let timestamps: PrimitiveArray<TimestampSecondType> = ts_array(None, vec![Some(1000)]);
        // `timestamp(s) + 1.5s` floors to +1s and `- 1.5s` floors to -2s; a
        // naive truncating division would report -1s for the negative case.
        for (nanoseconds, expected) in [(1_500_000_000_i64, 1001_i64), (-1_500_000_000, 998)] {
            let intervals = iv_array(vec![mdn(0, 0, nanoseconds)]);
            let actual = array(evaluated(
                Arc::new(timestamps.clone()) as ArrayRef,
                Arc::new(intervals.clone()),
                false,
            ));
            assert_eq!(
                actual.as_primitive::<TimestampSecondType>().value(0),
                expected,
                "nanoseconds {nanoseconds}"
            );
            let fallback = add_wrapping(&timestamps, &intervals).unwrap();
            assert_eq!(actual.as_ref(), fallback.as_ref());
        }
    }

    #[test]
    fn timestamp_interval_month_offsets_delegate_to_arrow_calendar_kernel() {
        // 2024-01-31T12:00:00Z + 1 month clamps to 2024-02-29T12:00:00Z.
        let base = NaiveDateTime::parse_from_str("2024-01-31 12:00:00", "%Y-%m-%d %H:%M:%S")
            .unwrap()
            .and_utc();
        let microseconds = base.timestamp_micros();
        let timestamps: PrimitiveArray<TimestampMicrosecondType> =
            ts_array(Some("UTC"), vec![Some(microseconds)]);
        let intervals = iv_array(vec![mdn(1, 0, 0)]);
        let actual = array(evaluated(
            Arc::new(timestamps.clone()),
            Arc::new(intervals.clone()),
            false,
        ));
        let expected = base
            .checked_add_months(Months::new(1))
            .unwrap()
            .timestamp_micros();
        assert_eq!(
            actual.as_primitive::<TimestampMicrosecondType>().value(0),
            expected
        );
        let fallback = add_wrapping(&timestamps, &intervals).unwrap();
        assert_eq!(actual.as_ref(), fallback.as_ref());
    }

    #[test]
    fn timestamp_interval_fast_path_is_timezone_independent_across_dst() {
        // 2026-03-29 01:30 Europe/Paris local (CET) + 2 absolute hours crosses
        // the spring-forward gap: the Paris wall clock jumps 02:00 -> 03:00,
        // so wall-clock arithmetic would land on 03:30 local while instant
        // arithmetic (and raw integer addition) lands on 02:30Z = 04:30 local.
        let base = NaiveDateTime::parse_from_str("2026-03-29 00:30:00", "%Y-%m-%d %H:%M:%S")
            .unwrap()
            .and_utc();
        let microseconds = base.timestamp_micros();
        let timestamps: PrimitiveArray<TimestampMicrosecondType> =
            ts_array(Some("Europe/Paris"), vec![Some(microseconds)]);
        let intervals = iv_array(vec![mdn(0, 0, 7_200_000_000_000)]);
        let actual = array(evaluated(
            Arc::new(timestamps.clone()),
            Arc::new(intervals.clone()),
            false,
        ));
        let expected = NaiveDateTime::parse_from_str("2026-03-29 02:30:00", "%Y-%m-%d %H:%M:%S")
            .unwrap()
            .and_utc()
            .timestamp_micros();
        assert_eq!(
            actual.as_primitive::<TimestampMicrosecondType>().value(0),
            expected
        );
        assert_eq!(
            actual
                .as_primitive::<TimestampMicrosecondType>()
                .data_type(),
            timestamps.data_type()
        );
        let fallback = add_wrapping(&timestamps, &intervals).unwrap();
        assert_eq!(actual.as_ref(), fallback.as_ref());
    }

    #[test]
    fn timestamp_interval_null_operands_stay_null() {
        let timestamps: PrimitiveArray<TimestampMicrosecondType> =
            ts_array(None, vec![Some(1), None, Some(3), None]);
        let intervals = iv_array(vec![mdn(0, 0, 1_000), mdn(0, 0, 1_000), None, None]);
        let actual = array(evaluated(
            Arc::new(timestamps.clone()),
            Arc::new(intervals.clone()),
            false,
        ));
        let fallback = add_wrapping(&timestamps, &intervals).unwrap();
        assert_eq!(actual.as_ref(), fallback.as_ref());
        let values = actual.as_primitive::<TimestampMicrosecondType>();
        assert_eq!(values.value(0), 2);
        assert!(values.is_null(1));
        assert!(values.is_null(2));
        assert!(values.is_null(3));
    }

    #[test]
    fn timestamp_interval_mixed_batches_fall_back_wholesale() {
        let timestamps: PrimitiveArray<TimestampNanosecondType> =
            ts_array(None, vec![Some(1_000), Some(2_000), Some(3_000)]);
        let intervals = iv_array(vec![mdn(0, 0, 500), mdn(0, 1, 0), mdn(0, 0, 500)]);
        let actual = array(evaluated(
            Arc::new(timestamps.clone()),
            Arc::new(intervals.clone()),
            false,
        ));
        let fallback = add_wrapping(&timestamps, &intervals).unwrap();
        assert_eq!(actual.as_ref(), fallback.as_ref());
    }

    #[test]
    fn timestamp_interval_all_units_match_arrow_add_and_subtract() {
        use datafusion::arrow::datatypes::TimeUnit;
        // A modern epoch instant (2026-ish) expressed in each unit's scale.
        let modern_seconds = 1_771_000_000_i64;
        let scale = |unit: &TimeUnit| match unit {
            TimeUnit::Second => 1_i64,
            TimeUnit::Millisecond => 1_000,
            TimeUnit::Microsecond => 1_000_000,
            TimeUnit::Nanosecond => 1_000_000_000,
        };
        let nanos_per_unit = |unit: &TimeUnit| 1_000_000_000 / scale(unit);
        let multipliers = [0_i64, 1, -1, 123];
        // Divisible offsets exercise the fast path; sub-unit offsets and raw
        // nanosecond offsets exercise the fallback.
        let offsets = |unit: &TimeUnit| {
            [0_i64, 1, -1]
                .map(|sub| sub * nanos_per_unit(unit))
                .into_iter()
                .chain([0_i64, 1, -1, 5_000_000_000, -5_000_000_000])
        };
        for unit in [
            TimeUnit::Second,
            TimeUnit::Millisecond,
            TimeUnit::Microsecond,
            TimeUnit::Nanosecond,
        ] {
            let timestamps = with_unit(
                Some("UTC"),
                multipliers
                    .iter()
                    .map(|multiplier| Some(modern_seconds * scale(&unit) + multiplier))
                    .collect::<Vec<_>>(),
                unit,
            );
            for offset in offsets(&unit) {
                let intervals = iv_array(multipliers.iter().map(|_| mdn(0, 0, offset)).collect());
                for subtract in [false, true] {
                    let actual = array(evaluated(
                        Arc::clone(&timestamps),
                        Arc::new(intervals.clone()),
                        subtract,
                    ));
                    let fallback = if subtract {
                        sub_wrapping(&timestamps.as_ref(), &intervals).unwrap()
                    } else {
                        add_wrapping(&timestamps.as_ref(), &intervals).unwrap()
                    };
                    assert_eq!(
                        actual.data_type(),
                        fallback.data_type(),
                        "{unit:?} offset {offset} subtract {subtract}"
                    );
                    assert_eq!(
                        actual.as_ref(),
                        fallback.as_ref(),
                        "{unit:?} offset {offset} subtract {subtract}"
                    );
                }
            }
        }
    }

    fn with_unit(timezone: Option<&str>, values: Vec<Option<i64>>, unit: TimeUnit) -> ArrayRef {
        use datafusion::arrow::datatypes::TimeUnit;
        match unit {
            TimeUnit::Second => {
                Arc::new(ts_array::<TimestampSecondType>(timezone, values)) as ArrayRef
            }
            TimeUnit::Millisecond => {
                Arc::new(ts_array::<TimestampMillisecondType>(timezone, values))
            }
            TimeUnit::Microsecond => {
                Arc::new(ts_array::<TimestampMicrosecondType>(timezone, values))
            }
            TimeUnit::Nanosecond => Arc::new(ts_array::<TimestampNanosecondType>(timezone, values)),
        }
    }

    #[test]
    fn timestamp_interval_overflow_matches_arrow_error() {
        // The fast path declines overflowing batches, so the surfaced error is
        // arrow's own, wrapped as DataFusion wraps every kernel error.
        let wrapped = |raw: datafusion::arrow::error::ArrowError| format!("Arrow error: {raw}");
        let timestamps: PrimitiveArray<TimestampNanosecondType> =
            ts_array(None, vec![Some(i64::MAX)]);
        let intervals = iv_array(vec![mdn(0, 0, 1_000_000_000)]);
        let actual = evaluated(
            Arc::new(timestamps.clone()),
            Arc::new(intervals.clone()),
            false,
        )
        .unwrap_err()
        .to_string();
        let fallback = add_wrapping(&timestamps, &intervals).unwrap_err();
        assert_eq!(actual, wrapped(fallback));

        // A microsecond timestamp beyond chrono's representable range errors
        // the same way even though raw i64 arithmetic would succeed.
        let timestamps: PrimitiveArray<TimestampMicrosecondType> =
            ts_array(None, vec![Some(i64::MAX)]);
        let intervals = iv_array(vec![mdn(0, 0, 1_000_000)]);
        let actual = evaluated(
            Arc::new(timestamps.clone()),
            Arc::new(intervals.clone()),
            false,
        )
        .unwrap_err()
        .to_string();
        let fallback = add_wrapping(&timestamps, &intervals).unwrap_err();
        assert_eq!(actual, wrapped(fallback));

        // Both operands inside the representable range while only the shifted
        // result leaves it: the fast path must still defer to arrow's error.
        let (low, high) = representable_bounds(TimeUnit::Microsecond);
        let timestamps: PrimitiveArray<TimestampMicrosecondType> =
            ts_array(None, vec![Some(high - 5_000_000)]);
        let intervals = iv_array(vec![mdn(0, 0, 10_000_000_000)]);
        let actual = evaluated(
            Arc::new(timestamps.clone()),
            Arc::new(intervals.clone()),
            false,
        )
        .unwrap_err()
        .to_string();
        let fallback = add_wrapping(&timestamps, &intervals).unwrap_err();
        assert_eq!(actual, wrapped(fallback));
        let _ = low;
    }

    #[test]
    fn timestamp_interval_scalar_operands_match_arrow_broadcast() {
        let timestamps: PrimitiveArray<TimestampMicrosecondType> =
            ts_array(Some("UTC"), vec![Some(1), Some(2), None]);
        let single = iv_array(vec![mdn(0, 0, 5_000_000)]);

        let actual = array(evaluate_timestamp_interval(
            &ColumnarValue::Array(Arc::new(timestamps.clone()) as ArrayRef),
            &ColumnarValue::Scalar(ScalarValue::IntervalMonthDayNano(mdn(0, 0, 5_000_000))),
            false,
        ));
        let fallback = add_wrapping(&timestamps, &Scalar::new(single.clone())).unwrap();
        assert_eq!(actual.as_ref(), fallback.as_ref());

        let actual = array(evaluate_timestamp_interval(
            &ColumnarValue::Scalar(ScalarValue::TimestampMicrosecond(
                Some(10),
                Some(Arc::from("UTC")),
            )),
            &ColumnarValue::Array(Arc::new(iv_array(vec![
                mdn(0, 0, 1_000),
                mdn(1, 0, 0),
                None,
            ]))),
            false,
        ));
        let fallback = add_wrapping(
            &Scalar::new(ts_array::<TimestampMicrosecondType>(
                Some("UTC"),
                vec![Some(10)],
            )),
            &iv_array(vec![mdn(0, 0, 1_000), mdn(1, 0, 0), None]),
        )
        .unwrap();
        assert_eq!(actual.as_ref(), fallback.as_ref());

        let scalar = evaluate_timestamp_interval(
            &ColumnarValue::Scalar(ScalarValue::TimestampMicrosecond(
                Some(10),
                Some(Arc::from("UTC")),
            )),
            &ColumnarValue::Scalar(ScalarValue::IntervalMonthDayNano(mdn(0, 0, 1_000))),
            false,
        )
        .unwrap();
        let fallback = add_wrapping(
            &Scalar::new(ts_array::<TimestampMicrosecondType>(
                Some("UTC"),
                vec![Some(10)],
            )),
            &Scalar::new(iv_array(vec![mdn(0, 0, 1_000)])),
        )
        .unwrap();
        let ColumnarValue::Scalar(ScalarValue::TimestampMicrosecond(value, timezone)) = scalar
        else {
            panic!("expected scalar result");
        };
        assert_eq!(value, Some(11));
        assert_eq!(timezone, Some(Arc::from("UTC")));
        assert_eq!(
            fallback.as_primitive::<TimestampMicrosecondType>().value(0),
            11
        );

        let null_scalar = evaluate_timestamp_interval(
            &ColumnarValue::Scalar(ScalarValue::TimestampMicrosecond(Some(10), None)),
            &ColumnarValue::Scalar(ScalarValue::IntervalMonthDayNano(None)),
            false,
        )
        .unwrap();
        let ColumnarValue::Scalar(ScalarValue::TimestampMicrosecond(value, timezone)) = null_scalar
        else {
            panic!("expected scalar result");
        };
        assert_eq!(value, None);
        assert_eq!(timezone, None);
    }

    #[test]
    fn timestamp_interval_dst_zone_calendar_offsets_still_match_arrow() {
        // Europe/Paris with calendar offsets crossing the DST change: the
        // fast path must not apply because months or days are non-zero.
        let base = DateTime::parse_from_rfc3339("2026-03-29T00:30:00+00:00").unwrap();
        let microseconds = base.timestamp_micros();
        let timestamps: PrimitiveArray<TimestampMicrosecondType> =
            ts_array(Some("Europe/Paris"), vec![Some(microseconds)]);
        for offset in [mdn(1, 0, 0), mdn(0, 1, 0), mdn(-1, -1, 3_600_000_000_000)] {
            let intervals = iv_array(vec![offset]);
            let actual = array(evaluated(
                Arc::new(timestamps.clone()),
                Arc::new(intervals.clone()),
                false,
            ));
            let fallback = add_wrapping(&timestamps, &intervals).unwrap();
            assert_eq!(actual.as_ref(), fallback.as_ref(), "{offset:?}");
        }
    }

    mod rule {
        use std::sync::Arc;

        use datafusion::{
            arrow::datatypes::{DataType, Field, IntervalUnit, Schema, TimeUnit},
            common::{DFSchema, ScalarValue},
            logical_expr::{
                EmptyRelation, Expr, ExprSchemable, Filter, LogicalPlan, Operator, col,
                expr::BinaryExpr, logical_plan::Projection,
            },
            optimizer::{OptimizerContext, OptimizerRule, optimizer::ApplyOrder},
            prelude::lit,
        };

        use crate::datafusion_interval::{
            TimestampIntervalArithmetic, interval_udf_names, specialize_interval_arithmetic,
        };

        fn schema() -> DFSchema {
            DFSchema::try_from(Schema::new(vec![
                Field::new(
                    "event_time",
                    DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                    true,
                ),
                Field::new(
                    "offsets",
                    DataType::Interval(IntervalUnit::MonthDayNano),
                    true,
                ),
                Field::new("sequence", DataType::UInt64, true),
            ]))
            .unwrap()
        }

        fn interval(months: i32, days: i32, nanoseconds: i64) -> Expr {
            lit(ScalarValue::IntervalMonthDayNano(Some(
                datafusion::arrow::array::types::IntervalMonthDayNano::new(
                    months,
                    days,
                    nanoseconds,
                ),
            )))
        }

        fn udf_name(expression: &Expr) -> &str {
            let Expr::ScalarFunction(call) = expression else {
                panic!("expected scalar function, got {expression:?}");
            };
            call.func.name()
        }

        #[test]
        fn timestamp_interval_rule_rewrites_pure_nanosecond_offsets() {
            let schema = schema();
            let add = col("event_time") + interval(0, 0, 5_000_000_000);
            let rewritten = specialize_interval_arithmetic(&add, &schema)
                .unwrap_or_else(|| panic!("expected rewrite of {add}"));
            assert_eq!(
                udf_name(&rewritten),
                interval_udf_names().0,
                "rewritten as {rewritten}"
            );
            assert_eq!(
                rewritten.get_type(&schema).unwrap(),
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()))
            );
            assert!(rewritten.nullable(&schema).unwrap());

            let subtract = col("event_time") - interval(0, 0, 5_000_000_000);
            let rewritten = specialize_interval_arithmetic(&subtract, &schema).unwrap();
            assert_eq!(udf_name(&rewritten), interval_udf_names().1);

            // Mirrored `interval + timestamp` is commutative in arrow, so it
            // rewrites too; mirrored subtraction is not supported by arrow and
            // must stay untouched.
            let mirrored = interval(0, 0, 5_000_000_000) + col("event_time");
            assert!(specialize_interval_arithmetic(&mirrored, &schema).is_some());
            let mirrored_subtract = interval(0, 0, 5_000_000_000) - col("event_time");
            assert!(specialize_interval_arithmetic(&mirrored_subtract, &schema).is_none());

            // Column-valued intervals rewrite as well; the batch decides at
            // runtime whether the fast path applies.
            let column = col("event_time") + col("offsets");
            let rewritten = specialize_interval_arithmetic(&column, &schema).unwrap();
            assert_eq!(udf_name(&rewritten), interval_udf_names().0);
        }

        #[test]
        fn timestamp_interval_rule_skips_calendar_and_foreign_operands() {
            let schema = schema();
            for untouched in [
                // Calendar literals can never take the fast path, so the rule
                // leaves them on the plain binary expression.
                col("event_time") + interval(1, 0, 0),
                col("event_time") + interval(0, 1, 0),
                col("event_time") - interval(-1, 0, 0),
                // Nanoseconds not divisible by the microsecond unit.
                col("event_time") + interval(0, 0, 1),
                // Non-timestamp left operand.
                col("sequence") + interval(0, 0, 5_000_000_000),
                lit(1_i64) + interval(0, 0, 5_000_000_000),
                // Non-interval right operand.
                col("event_time") + lit(1_i64),
                col("event_time") - col("sequence"),
                // Non-arithmetic operators.
                Expr::BinaryExpr(BinaryExpr::new(
                    Box::new(col("event_time")),
                    Operator::Eq,
                    Box::new(interval(0, 0, 5_000_000_000)),
                )),
                col("event_time"),
            ] {
                assert!(
                    specialize_interval_arithmetic(&untouched, &schema).is_none(),
                    "unexpectedly rewrote {untouched}"
                );
            }
        }

        fn empty_input() -> Arc<LogicalPlan> {
            Arc::new(LogicalPlan::EmptyRelation(EmptyRelation {
                produce_one_row: false,
                schema: Arc::new(
                    DFSchema::try_from(Schema::new(vec![Field::new(
                        "event_time",
                        DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                        true,
                    )]))
                    .unwrap(),
                ),
            }))
        }

        #[test]
        fn timestamp_interval_rule_rewrites_filter_and_projection_plans() {
            let rule = TimestampIntervalArithmetic;
            let config = OptimizerContext::new();
            assert_eq!(rule.apply_order(), Some(ApplyOrder::BottomUp));
            let input = empty_input();
            let predicate = (col("event_time") + interval(0, 0, 5_000_000_000)).gt(lit(0_i64));
            let filter =
                LogicalPlan::Filter(Filter::try_new(predicate, Arc::clone(&input)).unwrap());
            let rewritten = rule.rewrite(filter, &config).unwrap();
            assert!(rewritten.transformed);
            let LogicalPlan::Filter(filter) = rewritten.data else {
                panic!("expected filter plan");
            };
            assert!(
                format!("{}", filter.predicate).contains(interval_udf_names().0),
                "predicate kept the binary expression: {}",
                filter.predicate
            );

            let projection = Projection::try_new(
                vec![col("event_time") + interval(0, 0, 5_000_000_000)],
                input,
            )
            .unwrap();
            let original_name = projection.schema.field(0).name().clone();
            let rewritten = rule
                .rewrite(LogicalPlan::Projection(projection), &config)
                .unwrap();
            assert!(rewritten.transformed);
            let LogicalPlan::Projection(projection) = rewritten.data else {
                panic!("expected projection plan");
            };
            assert_eq!(projection.schema.field(0).name(), &original_name);
            assert!(
                format!("{}", projection.expr[0]).contains(interval_udf_names().0),
                "projection kept the binary expression: {}",
                projection.expr[0]
            );
        }
    }

    mod end_to_end {
        use std::{collections::BTreeMap, fs, path::Path, sync::Arc};

        use datafusion::{
            arrow::{
                array::{
                    ArrayRef, Float64Array, PrimitiveArray, StringArray, UInt64Array,
                    types::TimestampMicrosecondType,
                },
                compute::concat_batches,
                datatypes::{DataType, Field, Schema, TimeUnit},
                record_batch::RecordBatch,
            },
            datasource::MemTable,
            execution::context::{SessionConfig, SessionContext},
        };

        use crate::{Batch, BatchMetadata, DataFusionConfig, DataFusionRuntime};

        /// Extracts the benchmark's `interval_join` query from
        /// `benchmarks/engine_comparison.py` so the engine test exercises the
        /// exact production SQL instead of a drifted copy. Panics loudly when
        /// the benchmark file restructures.
        fn benchmark_interval_join_sql() -> String {
            let source = fs::read_to_string(
                Path::new(env!("CARGO_MANIFEST_DIR")).join("../../benchmarks/engine_comparison.py"),
            )
            .expect("benchmarks/engine_comparison.py is readable");
            let entry = "\"interval_join\": (";
            let start = source
                .find(entry)
                .expect("interval_join query catalog entry exists")
                + entry.len();
            let end = start
                + source[start..]
                    .find("\n        ),")
                    .expect("interval_join entry terminates");
            let literal = &source[start..end];
            let mut query = String::new();
            let mut rest = literal;
            while let Some(open) = rest.find('"') {
                let after = &rest[open + 1..];
                let close = after.find('"').expect("terminated string segment");
                query.push_str(&after[..close]);
                rest = &after[close + 1..];
            }
            assert!(
                query.starts_with("WITH offsets"),
                "unexpected interval_join query shape: {query}"
            );
            query
        }

        fn input_batch() -> Batch {
            let schema = Arc::new(Schema::new(vec![
                Field::new(
                    "event_time",
                    DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                    false,
                ),
                Field::new("sequence", DataType::UInt64, false),
                Field::new("symbol", DataType::Utf8, false),
                Field::new("price", DataType::Float64, false),
            ]));
            let base = 1_771_000_000_i64; // microseconds
            let rows = 12_u32;
            let event_time = Arc::new(
                (0..rows)
                    .map(|row| Some(base + i64::from(row % 4) * 1_000_000))
                    .collect::<PrimitiveArray<TimestampMicrosecondType>>()
                    .with_timezone("UTC"),
            ) as ArrayRef;
            let sequence =
                Arc::new(UInt64Array::from_iter_values((0..rows).map(u64::from))) as ArrayRef;
            let symbol = Arc::new(StringArray::from_iter_values(
                (0..rows).map(|row| format!("S{:03}", row % 3)),
            )) as ArrayRef;
            let price = Arc::new(Float64Array::from_iter_values(
                (0..rows).map(|row| 100.0 + f64::from(row) / 8.0),
            )) as ArrayRef;
            let record =
                RecordBatch::try_new(schema, vec![event_time, sequence, symbol, price]).unwrap();
            Batch::table(vec![record], BatchMetadata::default()).unwrap()
        }

        fn sorted_rows(combined: &RecordBatch) -> Vec<(u64, u64, f64)> {
            let mut rows: Vec<(u64, u64, f64)> = (0..combined.num_rows())
                .map(|index| {
                    (
                        combined
                            .column(0)
                            .as_any()
                            .downcast_ref::<UInt64Array>()
                            .unwrap()
                            .value(index),
                        combined
                            .column(1)
                            .as_any()
                            .downcast_ref::<UInt64Array>()
                            .unwrap()
                            .value(index),
                        combined
                            .column(2)
                            .as_any()
                            .downcast_ref::<Float64Array>()
                            .unwrap()
                            .value(index),
                    )
                })
                .collect();
            rows.sort_by(|left, right| {
                (left.0, left.1, left.2.to_bits())
                    .partial_cmp(&(right.0, right.1, right.2.to_bits()))
                    .unwrap()
            });
            rows
        }

        fn engine_rows(result: &Batch) -> Vec<(u64, u64, f64)> {
            let table = result.table_payload().unwrap();
            sorted_rows(&concat_batches(table.schema(), table.batches()).unwrap())
        }

        async fn baseline_rows(query: &str, input: &Batch) -> Vec<(u64, u64, f64)> {
            let context =
                SessionContext::new_with_config(SessionConfig::new().with_target_partitions(1));
            let table = input.table_payload().unwrap();
            for name in ["input", "reference"] {
                context
                    .register_table(
                        name,
                        Arc::new(
                            MemTable::try_new(
                                Arc::clone(table.schema()),
                                vec![table.batches().to_vec()],
                            )
                            .unwrap(),
                        ),
                    )
                    .unwrap();
            }
            let dataframe = context.sql(query).await.unwrap();
            let schema = Arc::new(dataframe.schema().as_arrow().clone());
            sorted_rows(&concat_batches(&schema, &dataframe.collect().await.unwrap()).unwrap())
        }

        #[tokio::test]
        async fn interval_benchmark_sql_matches_rule_free_datafusion_and_uses_native_band() {
            let query = benchmark_interval_join_sql();
            let input = input_batch();
            let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
            let result = runtime
                .sql(
                    &query,
                    &BTreeMap::from([
                        ("input".into(), input.clone()),
                        ("reference".into(), input.clone()),
                    ]),
                    None,
                )
                .await
                .unwrap();
            assert_eq!(
                engine_rows(&result),
                baseline_rows(&query, &input).await,
                "engine result diverged from plain DataFusion for {query}"
            );
            let plan = &runtime.metrics()[0].physical_plan;
            assert!(
                plan.contains("CalcFlowBandJoinExec"),
                "interval_join did not route through the native band path: {plan}"
            );
        }
    }
}
