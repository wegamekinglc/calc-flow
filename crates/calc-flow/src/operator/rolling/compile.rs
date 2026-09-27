//! Rolling specification validation and compilation.

use super::{
    CalcFlowError, CompiledAggregate, CompiledDifference, CompiledEvaluation, CompiledEwma,
    CompiledFloatReadout, CompiledFrame, CompiledKeyColumn, CompiledPairAggregate,
    CompiledRollingOutput, CompiledRollingSpec, CompiledScan, CompiledWindowGroup, DataType,
    Digest, Field, JsonMap, ROLLING_COLUMNAR_STATE_LAYOUT_VERSION, ROLLING_CONFIGURATION_VERSION,
    ROLLING_EWMA_STATE_LAYOUT_VERSION, ROLLING_STATE_LAYOUT_VERSION, Result,
    RollingFloatPrimitiveSpec, RollingFrameSpec, RollingKernelPlan, RollingOutputSpec, RollingSpec,
    ScanKind, Schema, Sha256, Statistic, SumClass, TimeUnit, Value, canonical_json, compile_error,
    internal_error, json, state_schema_fingerprint,
};

pub(super) fn validate_arguments(spec: &RollingSpec) -> Result<()> {
    if spec.configuration_version != ROLLING_CONFIGURATION_VERSION {
        return Err(invalid_argument(
            "rolling.configuration_version",
            "unsupported rolling configuration version",
        ));
    }
    if !matches!(
        spec.state_layout_version,
        ROLLING_STATE_LAYOUT_VERSION | ROLLING_EWMA_STATE_LAYOUT_VERSION
    ) {
        return Err(invalid_argument(
            "rolling.state_layout_version",
            "unsupported rolling state layout version",
        ));
    }
    if spec
        .outputs
        .iter()
        .any(RollingOutputSpec::requires_ewma_layout)
        && spec.state_layout_version != ROLLING_EWMA_STATE_LAYOUT_VERSION
    {
        return Err(invalid_argument(
            "rolling.state_layout_version",
            "EWMA outputs require rolling state layout version 2",
        ));
    }
    validate_key_names("rolling.partition_by", &spec.partition_by)?;
    validate_key_names("rolling.sequence_by", &spec.sequence_by)?;
    validate_outputs(&spec.outputs)?;
    super::super::late_output::validate_policy(spec.late_policy, "rolling")
}

fn validate_key_names(field: &str, columns: &[String]) -> Result<()> {
    if columns.is_empty() {
        return Err(invalid_argument(field, "must not be empty"));
    }
    for (index, column) in columns.iter().enumerate() {
        let indexed = format!("{field}[{index}]");
        if column.is_empty() {
            return Err(invalid_argument(&indexed, "must not be empty"));
        }
        if columns[..index].contains(column) {
            return Err(invalid_argument(
                &indexed,
                "duplicates an earlier key column",
            ));
        }
    }
    Ok(())
}

fn validate_outputs(outputs: &[RollingOutputSpec]) -> Result<()> {
    if outputs.is_empty() {
        return Err(invalid_argument("rolling.outputs", "must not be empty"));
    }
    for (index, output) in outputs.iter().enumerate() {
        let base = format!("rolling.outputs[{index}]");
        if output.primitive_version() != 1 {
            return Err(invalid_argument(
                &format!("{base}.primitive_version"),
                "unsupported rolling primitive version",
            ));
        }
        if let RollingOutputSpec::Difference { left, right, .. } = output {
            validate_float_primitive(&format!("{base}.left"), left)?;
            validate_float_primitive(&format!("{base}.right"), right)?;
        }
        if output.span().is_some_and(|span| span == 0) {
            return Err(invalid_argument(
                &format!("{base}.span"),
                "must be greater than zero",
            ));
        }
        if let Some(frame) = output.frame() {
            let zero = match frame {
                RollingFrameSpec::Rows { size } => size == 0,
                RollingFrameSpec::Duration { micros } => micros == 0,
            };
            if zero {
                let field = match frame {
                    RollingFrameSpec::Rows { .. } => format!("{base}.frame.size"),
                    RollingFrameSpec::Duration { .. } => format!("{base}.frame.micros"),
                };
                return Err(invalid_argument(&field, "must be greater than zero"));
            }
        } else if output.span().is_none()
            && output.retained_rows() == 0
            && !matches!(
                output,
                RollingOutputSpec::Difference { .. } | RollingOutputSpec::CumulativeMean { .. }
            )
        {
            return Err(invalid_argument(
                &format!("{base}.periods"),
                "must be greater than zero",
            ));
        }
        if let Some(min_periods) = output.min_periods() {
            if min_periods == 0 {
                return Err(invalid_argument(
                    &format!("{base}.min_periods"),
                    "must be greater than zero",
                ));
            }
            // Only row-count frames cap min_periods at their size; a duration
            // frame has no row-count ceiling (SCE-00 D5).
            if matches!(output.frame(), Some(RollingFrameSpec::Rows { .. }))
                && min_periods > output.retained_rows()
            {
                return Err(invalid_argument(
                    &format!("{base}.min_periods"),
                    "must not exceed the row-frame size",
                ));
            }
        }
        if let Some(ddof) = output.ddof()
            && ddof > 1
        {
            return Err(invalid_argument(&format!("{base}.ddof"), "must be 0 or 1"));
        }
        if output.input().is_empty() {
            return Err(invalid_argument(
                &format!("{base}.input"),
                "must not be empty",
            ));
        }
        if let Some(right) = output.pair_right()
            && right.is_empty()
        {
            return Err(invalid_argument(
                &format!("{base}.right"),
                "must not be empty",
            ));
        }
        if output.output().is_empty() {
            return Err(invalid_argument(
                &format!("{base}.output"),
                "must not be empty",
            ));
        }
        if outputs[..index]
            .iter()
            .any(|earlier| earlier.output() == output.output())
        {
            return Err(invalid_argument(
                &format!("{base}.output"),
                "duplicates an earlier rolling output",
            ));
        }
    }
    Ok(())
}

fn validate_float_primitive(base: &str, primitive: &RollingFloatPrimitiveSpec) -> Result<()> {
    if primitive.primitive_version() != 1 {
        return Err(invalid_argument(
            &format!("{base}.primitive_version"),
            "unsupported rolling primitive version",
        ));
    }
    if primitive.input().is_empty() {
        return Err(invalid_argument(
            &format!("{base}.input"),
            "must not be empty",
        ));
    }
    match primitive {
        RollingFloatPrimitiveSpec::Ewma {
            span, min_periods, ..
        } => {
            if *span == 0 {
                return Err(invalid_argument(
                    &format!("{base}.span"),
                    "must be greater than zero",
                ));
            }
            validate_positive_min_periods(base, *min_periods, None)
        }
        RollingFloatPrimitiveSpec::Mean {
            frame, min_periods, ..
        } => validate_positive_min_periods(base, *min_periods, Some(*frame)),
        RollingFloatPrimitiveSpec::Variance {
            frame,
            min_periods,
            ddof,
            ..
        }
        | RollingFloatPrimitiveSpec::Stddev {
            frame,
            min_periods,
            ddof,
            ..
        } => {
            if *ddof > 1 {
                return Err(invalid_argument(&format!("{base}.ddof"), "must be 0 or 1"));
            }
            validate_positive_min_periods(base, *min_periods, Some(*frame))
        }
    }
}

fn validate_positive_min_periods(
    base: &str,
    min_periods: u64,
    frame: Option<RollingFrameSpec>,
) -> Result<()> {
    if min_periods == 0 {
        return Err(invalid_argument(
            &format!("{base}.min_periods"),
            "must be greater than zero",
        ));
    }
    if let Some(RollingFrameSpec::Rows { size }) = frame {
        if size == 0 {
            return Err(invalid_argument(
                &format!("{base}.frame.size"),
                "must be greater than zero",
            ));
        }
        if min_periods > size {
            return Err(invalid_argument(
                &format!("{base}.min_periods"),
                "must not exceed the row-frame size",
            ));
        }
    } else if matches!(frame, Some(RollingFrameSpec::Duration { micros: 0 })) {
        return Err(invalid_argument(
            &format!("{base}.frame.micros"),
            "must be greater than zero",
        ));
    }
    Ok(())
}

pub(super) fn compile_spec(
    spec: &RollingSpec,
    input_schema: &Schema,
) -> Result<CompiledRollingSpec> {
    compile_spec_against_schema(spec, input_schema, String::new())
}

pub(super) fn compile_spec_full(
    spec: &RollingSpec,
    input_schema: &Schema,
    configuration: &JsonMap,
) -> Result<CompiledRollingSpec> {
    let canonical = canonical_json(&Value::Object(configuration.clone().into_iter().collect()))?;
    let configuration_hash = hex::encode(Sha256::digest(canonical.as_bytes()));
    compile_spec_against_schema(spec, input_schema, configuration_hash)
}

fn compile_spec_against_schema(
    spec: &RollingSpec,
    input_schema: &Schema,
    configuration_hash: String,
) -> Result<CompiledRollingSpec> {
    super::super::late_output::validate_input(spec.late_policy, input_schema, "rolling")?;
    let event_time_index = exact_field_index(input_schema, &spec.event_time)?;
    validate_event_time(input_schema, event_time_index, &spec.event_time)?;
    let partition_columns = spec
        .partition_by
        .iter()
        .map(|column| compile_key_column(input_schema, column, KeyRole::Partition))
        .collect::<Result<Vec<_>>>()?;
    let sequence_columns = spec
        .sequence_by
        .iter()
        .map(|column| compile_key_column(input_schema, column, KeyRole::Sequence))
        .collect::<Result<Vec<_>>>()?;
    let mut window_groups = Vec::new();
    let outputs = spec
        .outputs
        .iter()
        .enumerate()
        .map(|(ordinal, output)| compile_output(input_schema, output, ordinal, &mut window_groups))
        .collect::<Result<Vec<_>>>()?;
    let max_row_retention = spec
        .outputs
        .iter()
        .map(RollingOutputSpec::retained_rows)
        .max()
        .unwrap_or(1);
    let max_duration_micros = spec
        .outputs
        .iter()
        .filter_map(RollingOutputSpec::retained_micros)
        .max();
    let kernel_plan = RollingKernelPlan::compile(
        input_schema,
        ROLLING_COLUMNAR_STATE_LAYOUT_VERSION,
        spec.numerical_profile,
        event_time_index,
        &partition_columns,
        &sequence_columns,
        &outputs,
        &window_groups,
    );
    Ok(CompiledRollingSpec {
        state_layout_version: ROLLING_COLUMNAR_STATE_LAYOUT_VERSION,
        legacy_state_layout_version: spec.state_layout_version,
        event_time_index,
        partition_columns,
        sequence_columns,
        outputs,
        window_groups,
        kernel_plan,
        max_row_retention,
        max_duration_micros,
        configuration_hash,
        state_schema_fingerprint: state_schema_fingerprint(
            input_schema,
            ROLLING_COLUMNAR_STATE_LAYOUT_VERSION,
        ),
        legacy_state_schema_fingerprint: state_schema_fingerprint(
            input_schema,
            spec.state_layout_version,
        ),
    })
}

#[derive(Clone, Copy)]
enum KeyRole {
    Partition,
    Sequence,
}

fn compile_key_column(
    input_schema: &Schema,
    column: &str,
    role: KeyRole,
) -> Result<CompiledKeyColumn> {
    let index = exact_field_index(input_schema, column)?;
    let field = input_schema.field(index);
    let data_type = field.data_type().clone();
    match role {
        KeyRole::Partition => {
            if !supports_total_order(&data_type) {
                return Err(compile_error(format!(
                    "rolling partition column {column:?} has unsupported type {data_type}"
                )));
            }
        }
        KeyRole::Sequence => {
            if field.is_nullable() {
                return Err(compile_error(format!(
                    "rolling sequence column {column:?} must be non-nullable"
                )));
            }
            if matches!(data_type, DataType::Float32 | DataType::Float64) {
                return Err(compile_error(format!(
                    "rolling sequence column {column:?} must not use a floating type"
                )));
            }
            if !supports_total_order(&data_type) {
                return Err(compile_error(format!(
                    "rolling sequence column {column:?} has unsupported type {data_type}"
                )));
            }
        }
    }
    Ok(CompiledKeyColumn { index })
}

fn compile_output(
    input_schema: &Schema,
    output: &RollingOutputSpec,
    ordinal: usize,
    window_groups: &mut Vec<CompiledWindowGroup>,
) -> Result<CompiledRollingOutput> {
    if input_schema
        .fields()
        .iter()
        .any(|field| field.name() == output.output())
    {
        return Err(invalid_argument(
            &format!("rolling.outputs[{ordinal}].output"),
            "collides with an input field name",
        ));
    }
    let input_index = exact_field_index(input_schema, output.input())?;
    let input_type = input_schema.field(input_index).data_type().clone();
    let evaluation = match output {
        RollingOutputSpec::Lag { periods, .. } => CompiledEvaluation::Lag { periods: *periods },
        RollingOutputSpec::Delta { periods, .. } => {
            require_numeric(output.input(), &input_type, "delta")?;
            CompiledEvaluation::Delta { periods: *periods }
        }
        RollingOutputSpec::Ewma {
            span, min_periods, ..
        } => {
            require_numeric(output.input(), &input_type, "ewma")?;
            let group = compile_ewma_group(input_index, *span, window_groups);
            CompiledEvaluation::Ewma(CompiledEwma {
                group,
                min_periods: *min_periods,
            })
        }
        RollingOutputSpec::CumulativeMean { min_periods, .. } => {
            require_numeric(output.input(), &input_type, "cumulative_mean")?;
            let group = compile_ewma_group(input_index, 0, window_groups);
            CompiledEvaluation::Ewma(CompiledEwma {
                group,
                min_periods: *min_periods,
            })
        }
        RollingOutputSpec::Covariance {
            left,
            right,
            frame,
            min_periods,
            ddof,
            ..
        }
        | RollingOutputSpec::Correlation {
            left,
            right,
            frame,
            min_periods,
            ddof,
            ..
        } => {
            let correlation = matches!(output, RollingOutputSpec::Correlation { .. });
            let left_index = exact_field_index(input_schema, left)?;
            let right_index = exact_field_index(input_schema, right)?;
            let left_type = input_schema.field(left_index).data_type().clone();
            let right_type = input_schema.field(right_index).data_type().clone();
            require_numeric(left, &left_type, "covariance")?;
            require_numeric(right, &right_type, "covariance")?;
            let group = compile_pair_group(left_index, right_index, *frame, window_groups);
            CompiledEvaluation::Pair(CompiledPairAggregate {
                group,
                correlation,
                min_periods: *min_periods,
                ddof: *ddof,
            })
        }
        RollingOutputSpec::Difference { left, right, .. } => {
            CompiledEvaluation::Difference(CompiledDifference {
                left: compile_float_readout(input_schema, left, window_groups)?,
                right: compile_float_readout(input_schema, right, window_groups)?,
            })
        }
        RollingOutputSpec::Argmax { .. }
        | RollingOutputSpec::Argmin { .. }
        | RollingOutputSpec::Rank { .. }
        | RollingOutputSpec::Quantile { .. }
        | RollingOutputSpec::UniqueCount { .. }
        | RollingOutputSpec::Decay { .. } => compile_scan_output(output, &input_type)?,
        aggregate => compile_aggregate_output(aggregate, input_index, &input_type, window_groups)?,
    };
    let output_type = compiled_output_type(&evaluation, &input_type);
    Ok(CompiledRollingOutput {
        input_index,
        name: output.output().to_owned(),
        output_type,
        input_type,
        evaluation,
    })
}

fn compiled_output_type(evaluation: &CompiledEvaluation, input_type: &DataType) -> DataType {
    match evaluation {
        CompiledEvaluation::Lag { .. } | CompiledEvaluation::Delta { .. } => input_type.clone(),
        CompiledEvaluation::Ewma(_)
        | CompiledEvaluation::Pair(_)
        | CompiledEvaluation::Difference(_) => DataType::Float64,
        CompiledEvaluation::Scan(scan) => match scan.kind {
            ScanKind::Argmax | ScanKind::Argmin | ScanKind::Rank | ScanKind::UniqueCount => {
                DataType::UInt64
            }
            ScanKind::Quantile | ScanKind::Decay => DataType::Float64,
        },
        CompiledEvaluation::Aggregate(aggregate) => match aggregate.statistic {
            Statistic::Count => DataType::UInt64,
            Statistic::Sum => match SumClass::from_input(input_type) {
                SumClass::Signed => DataType::Int64,
                SumClass::Unsigned => DataType::UInt64,
                _ => DataType::Float64,
            },
            Statistic::Mean | Statistic::Variance | Statistic::Stddev => DataType::Float64,
            // Min/max preserve the input type (SCE-00 D3, contract
            // section 5.2).
            Statistic::Min | Statistic::Max => input_type.clone(),
        },
    }
}

fn compile_scan_output(
    output: &RollingOutputSpec,
    input_type: &DataType,
) -> Result<CompiledEvaluation> {
    let (kind, frame, min_periods) = match output {
        RollingOutputSpec::Argmax {
            frame, min_periods, ..
        } => (ScanKind::Argmax, *frame, *min_periods),
        RollingOutputSpec::Argmin {
            frame, min_periods, ..
        } => (ScanKind::Argmin, *frame, *min_periods),
        RollingOutputSpec::Rank {
            frame, min_periods, ..
        } => (ScanKind::Rank, *frame, *min_periods),
        RollingOutputSpec::Quantile {
            frame, min_periods, ..
        } => (ScanKind::Quantile, *frame, *min_periods),
        RollingOutputSpec::UniqueCount {
            frame, min_periods, ..
        } => (ScanKind::UniqueCount, *frame, *min_periods),
        RollingOutputSpec::Decay {
            frame, min_periods, ..
        } => (ScanKind::Decay, *frame, *min_periods),
        _ => return Err(internal_error("non-scan output reached scan compiler")),
    };
    if matches!(kind, ScanKind::UniqueCount) {
        if !supports_total_order(input_type) {
            return Err(compile_error(format!(
                "rolling unique_count input {:?} has unsupported type {input_type}",
                output.input()
            )));
        }
    } else {
        require_numeric(output.input(), input_type, "rolling scan")?;
    }
    Ok(CompiledEvaluation::Scan(CompiledScan {
        kind,
        frame: compiled_frame(frame),
        min_periods,
    }))
}

fn compile_float_readout(
    input_schema: &Schema,
    primitive: &RollingFloatPrimitiveSpec,
    window_groups: &mut Vec<CompiledWindowGroup>,
) -> Result<CompiledFloatReadout> {
    let input_index = exact_field_index(input_schema, primitive.input())?;
    let input_type = input_schema.field(input_index).data_type();
    require_numeric(primitive.input(), input_type, "fused difference")?;
    match primitive {
        RollingFloatPrimitiveSpec::Ewma {
            span, min_periods, ..
        } => Ok(CompiledFloatReadout::Ewma(CompiledEwma {
            group: compile_ewma_group(input_index, *span, window_groups),
            min_periods: *min_periods,
        })),
        RollingFloatPrimitiveSpec::Mean {
            frame, min_periods, ..
        } => compile_float_aggregate_readout(
            input_index,
            input_type,
            *frame,
            *min_periods,
            0,
            Statistic::Mean,
            window_groups,
        ),
        RollingFloatPrimitiveSpec::Variance {
            frame,
            min_periods,
            ddof,
            ..
        } => compile_float_aggregate_readout(
            input_index,
            input_type,
            *frame,
            *min_periods,
            *ddof,
            Statistic::Variance,
            window_groups,
        ),
        RollingFloatPrimitiveSpec::Stddev {
            frame,
            min_periods,
            ddof,
            ..
        } => compile_float_aggregate_readout(
            input_index,
            input_type,
            *frame,
            *min_periods,
            *ddof,
            Statistic::Stddev,
            window_groups,
        ),
    }
}

fn compile_float_aggregate_readout(
    input_index: usize,
    input_type: &DataType,
    frame: RollingFrameSpec,
    min_periods: u64,
    ddof: u8,
    statistic: Statistic,
    window_groups: &mut Vec<CompiledWindowGroup>,
) -> Result<CompiledFloatReadout> {
    let CompiledEvaluation::Aggregate(aggregate) = compile_aggregate(
        input_index,
        input_type,
        frame,
        min_periods,
        ddof,
        statistic,
        window_groups,
    ) else {
        return Err(internal_error(
            "fused float aggregate did not compile as an aggregate",
        ));
    };
    Ok(CompiledFloatReadout::Aggregate(aggregate))
}

#[allow(
    clippy::cast_precision_loss,
    reason = "the frozen EWMA recurrence uses IEEE binary64"
)]
fn compile_ewma_group(
    input_index: usize,
    span: u64,
    window_groups: &mut Vec<CompiledWindowGroup>,
) -> usize {
    window_groups
        .iter()
        .position(|group| {
            matches!(
                group,
                CompiledWindowGroup::Ewma {
                    input_index: existing_input,
                    span: existing_span,
                    ..
                } if *existing_input == input_index && *existing_span == span
            )
        })
        .unwrap_or_else(|| {
            window_groups.push(CompiledWindowGroup::Ewma {
                input_index,
                span,
                alpha: if span == 0 {
                    0.0
                } else {
                    2.0 / (span as f64 + 1.0)
                },
            });
            window_groups.len() - 1
        })
}

fn compile_pair_group(
    left_index: usize,
    right_index: usize,
    frame: RollingFrameSpec,
    window_groups: &mut Vec<CompiledWindowGroup>,
) -> usize {
    let frame = compiled_frame(frame);
    window_groups
        .iter()
        .position(|group| match group {
            CompiledWindowGroup::Pair {
                left_index: existing_left,
                right_index: existing_right,
                frame: existing_frame,
            } => {
                *existing_left == left_index
                    && *existing_right == right_index
                    && *existing_frame == frame
            }
            _ => false,
        })
        .unwrap_or_else(|| {
            window_groups.push(CompiledWindowGroup::Pair {
                left_index,
                right_index,
                frame,
            });
            window_groups.len() - 1
        })
}

fn compiled_frame(frame: RollingFrameSpec) -> CompiledFrame {
    match frame {
        RollingFrameSpec::Rows { size } => CompiledFrame::Rows(size),
        RollingFrameSpec::Duration { micros } => CompiledFrame::Duration(micros),
    }
}

fn compile_aggregate_output(
    output: &RollingOutputSpec,
    input_index: usize,
    input_type: &DataType,
    window_groups: &mut Vec<CompiledWindowGroup>,
) -> Result<CompiledEvaluation> {
    let (frame, min_periods, ddof, statistic) = match output {
        RollingOutputSpec::Count {
            frame, min_periods, ..
        } => (*frame, *min_periods, 0, Statistic::Count),
        RollingOutputSpec::Sum {
            frame, min_periods, ..
        } => (*frame, *min_periods, 0, Statistic::Sum),
        RollingOutputSpec::Mean {
            frame, min_periods, ..
        } => (*frame, *min_periods, 0, Statistic::Mean),
        RollingOutputSpec::Variance {
            frame,
            min_periods,
            ddof,
            ..
        } => (*frame, *min_periods, *ddof, Statistic::Variance),
        RollingOutputSpec::Stddev {
            frame,
            min_periods,
            ddof,
            ..
        } => (*frame, *min_periods, *ddof, Statistic::Stddev),
        RollingOutputSpec::Min {
            frame, min_periods, ..
        } => (*frame, *min_periods, 0, Statistic::Min),
        RollingOutputSpec::Max {
            frame, min_periods, ..
        } => (*frame, *min_periods, 0, Statistic::Max),
        RollingOutputSpec::Lag { .. }
        | RollingOutputSpec::Delta { .. }
        | RollingOutputSpec::Ewma { .. }
        | RollingOutputSpec::CumulativeMean { .. }
        | RollingOutputSpec::Covariance { .. }
        | RollingOutputSpec::Correlation { .. }
        | RollingOutputSpec::Argmax { .. }
        | RollingOutputSpec::Argmin { .. }
        | RollingOutputSpec::Rank { .. }
        | RollingOutputSpec::Quantile { .. }
        | RollingOutputSpec::UniqueCount { .. }
        | RollingOutputSpec::Decay { .. }
        | RollingOutputSpec::Difference { .. } => {
            unreachable!("lag, delta, and pair outputs compile before aggregates")
        }
    };
    if !matches!(
        statistic,
        Statistic::Count | Statistic::Min | Statistic::Max
    ) {
        require_numeric(output.input(), input_type, statistic.name())?;
    } else if matches!(statistic, Statistic::Min | Statistic::Max)
        && !supports_total_order(input_type)
    {
        return Err(compile_error(format!(
            "rolling {} does not support column {:?} with type {input_type}",
            statistic.name(),
            output.input()
        )));
    }
    Ok(compile_aggregate(
        input_index,
        input_type,
        frame,
        min_periods,
        ddof,
        statistic,
        window_groups,
    ))
}

fn require_numeric(column: &str, input_type: &DataType, primitive: &str) -> Result<()> {
    if !is_numeric(input_type) {
        return Err(compile_error(format!(
            "rolling {primitive} does not support column {column:?} with type {input_type}"
        )));
    }
    Ok(())
}

pub(super) fn compile_aggregate(
    input_index: usize,
    input_type: &DataType,
    frame: RollingFrameSpec,
    min_periods: u64,
    ddof: u8,
    statistic: Statistic,
    window_groups: &mut Vec<CompiledWindowGroup>,
) -> CompiledEvaluation {
    let frame = compiled_frame(frame);
    let group = if matches!(statistic, Statistic::Min | Statistic::Max) {
        let descending = matches!(statistic, Statistic::Max);
        window_groups
            .iter()
            .position(|group| match group {
                CompiledWindowGroup::Extrema {
                    input_index: existing_input,
                    frame: existing_frame,
                    descending: existing_descending,
                } => {
                    *existing_input == input_index
                        && *existing_frame == frame
                        && *existing_descending == descending
                }
                _ => false,
            })
            .unwrap_or_else(|| {
                window_groups.push(CompiledWindowGroup::Extrema {
                    input_index,
                    frame,
                    descending,
                });
                window_groups.len() - 1
            })
    } else {
        window_groups
            .iter()
            .position(|group| match group {
                CompiledWindowGroup::Numeric {
                    input_index: existing_input,
                    frame: existing_frame,
                    ..
                } => *existing_input == input_index && *existing_frame == frame,
                _ => false,
            })
            .unwrap_or_else(|| {
                window_groups.push(CompiledWindowGroup::Numeric {
                    input_index,
                    frame,
                    sum_class: SumClass::from_input(input_type),
                });
                window_groups.len() - 1
            })
    };
    CompiledEvaluation::Aggregate(CompiledAggregate {
        group,
        statistic,
        min_periods,
        ddof,
    })
}

fn exact_field_index(schema: &Schema, column: &str) -> Result<usize> {
    let matches = schema
        .fields()
        .iter()
        .enumerate()
        .filter(|(_, field)| field.name() == column)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [index] => Ok(*index),
        [] => Err(compile_error(format!(
            "rolling column {column:?} does not exist in the input schema"
        ))),
        _ => Err(compile_error(format!(
            "rolling column {column:?} is ambiguous in the input schema"
        ))),
    }
}

fn validate_event_time(schema: &Schema, index: usize, column: &str) -> Result<()> {
    let field = schema.field(index);
    if field.is_nullable() {
        return Err(compile_error(format!(
            "rolling event-time column {column:?} must be non-nullable"
        )));
    }
    if !matches!(
        field.data_type(),
        DataType::Timestamp(TimeUnit::Microsecond, Some(timezone)) if timezone.as_ref() == "UTC"
    ) {
        return Err(compile_error(format!(
            "rolling event-time column {column:?} must be a non-null UTC timestamp[us], found {}",
            field.data_type()
        )));
    }
    Ok(())
}

fn supports_total_order(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
            | DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Date32
            | DataType::Date64
    ) || matches!(
        data_type,
        DataType::Timestamp(TimeUnit::Microsecond, timezone)
            if timezone.as_deref().is_none_or(|timezone| timezone == "UTC")
    )
}

fn is_numeric(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
    )
}

pub(super) fn output_schema(input_schema: &Schema, outputs: &[CompiledRollingOutput]) -> Schema {
    let mut fields = input_schema.fields().to_vec();
    fields.extend(
        outputs
            .iter()
            .map(|output| Field::new(&output.name, output.output_type.clone(), true).into()),
    );
    Schema::new(fields)
}

pub(super) fn configuration(spec: &RollingSpec) -> Result<JsonMap> {
    let spec_json = serde_json::to_value(spec).map_err(|error| format_error(&error))?;
    Ok(JsonMap::from([
        ("kind".into(), json!("rolling")),
        ("spec".into(), spec_json),
    ]))
}

fn invalid_argument(field: &str, message: &str) -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: field.into(),
        message: message.into(),
    }
}

pub(super) fn operator_error(node_id: &str, message: &str) -> CalcFlowError {
    CalcFlowError::Operator {
        node_id: node_id.into(),
        message: message.into(),
    }
}

pub(super) fn format_error(error: &serde_json::Error) -> CalcFlowError {
    CalcFlowError::Format {
        message: error.to_string(),
    }
}
