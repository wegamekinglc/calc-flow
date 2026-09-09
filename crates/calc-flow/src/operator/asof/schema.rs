use super::{AsofJoinSide, StreamAsofJoinSpec, spec::invalid};
use crate::{Result, ValidationIssue};
use datafusion::arrow::datatypes::{DataType, Schema, SchemaRef, TimeUnit};
use std::{collections::BTreeSet, sync::Arc};

pub(crate) fn schema_issues(
    spec: &StreamAsofJoinSpec,
    left: &Schema,
    right: &Schema,
) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();
    side_issues(spec.left(), left, "left", &mut issues);
    side_issues(spec.right(), right, "right", &mut issues);
    for (position, (l, r)) in spec
        .left()
        .keys()
        .iter()
        .zip(spec.right().keys())
        .enumerate()
    {
        if let (Ok(l), Ok(r)) = (left.field_with_name(l), right.field_with_name(r))
            && l.data_type() != r.data_type()
        {
            issue(
                &mut issues,
                &format!("right.keys[{position}]"),
                "incompatible_key_type",
                "key types must match exactly",
            );
        }
    }
    let mut output = BTreeSet::new();
    for (side, schema) in [(spec.left(), left), (spec.right(), right)] {
        for field in schema.fields() {
            if !output.insert(format!("{}__{}", side.prefix(), field.name())) {
                issue(
                    &mut issues,
                    "right.prefix",
                    "invalid_output_prefix",
                    "prefixed output names must be unique",
                );
            }
        }
    }
    issues
}

fn side_issues(
    side: &AsofJoinSide,
    schema: &Schema,
    name: &str,
    issues: &mut Vec<ValidationIssue>,
) {
    let mut names = BTreeSet::new();
    for field in schema.fields() {
        if !payload_type(field.data_type()) {
            issue(
                issues,
                &format!("{name}_schema.{}", field.name()),
                "invalid_type",
                "ASOF v1 requires a flat Arrow payload type with bounded materialization accounting",
            );
        }
        if field.name().is_empty() || !names.insert(field.name()) {
            issue(
                issues,
                &format!("{name}_schema"),
                "invalid_type",
                "schema field names must be distinct and non-empty",
            );
        }
    }
    identity_columns(side.keys(), schema, name, "keys", issues);
    identity_columns(side.sequence_by(), schema, name, "sequence_by", issues);
    let path = format!("{name}.event_time");
    match schema.field_with_name(side.event_time()) {
        Err(_) => issue(
            issues,
            &path,
            "missing_field",
            "event time column is absent",
        ),
        Ok(field) => {
            if field.data_type() != &DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()))
            {
                issue(
                    issues,
                    &path,
                    "invalid_event_time",
                    "event time must be timestamp[us, UTC]",
                );
            }
            if field.is_nullable() {
                issue(
                    issues,
                    &path,
                    "nullable_identity_field",
                    "event time must be non-nullable",
                );
            }
        }
    }
}

fn identity_columns(
    columns: &[String],
    schema: &Schema,
    side: &str,
    role: &str,
    issues: &mut Vec<ValidationIssue>,
) {
    for (index, column) in columns.iter().enumerate() {
        let path = format!("{side}.{role}[{index}]");
        match schema.field_with_name(column) {
            Err(_) => issue(issues, &path, "missing_field", "identity column is absent"),
            Ok(field) => {
                let supported = if role == "keys" {
                    super::super::supported_key_type(field.data_type())
                } else {
                    sequence_type(field.data_type())
                };
                if !supported {
                    issue(
                        issues,
                        &path,
                        if role == "keys" {
                            "incompatible_key_type"
                        } else {
                            "invalid_asof_sequence"
                        },
                        "identity type has no supported total ordering",
                    );
                }
                if field.is_nullable() {
                    issue(
                        issues,
                        &path,
                        "nullable_identity_field",
                        "identity fields must be non-nullable",
                    );
                }
            }
        }
    }
}

fn sequence_type(data_type: &DataType) -> bool {
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
            | DataType::Utf8
            | DataType::LargeUtf8
    )
}
fn issue(issues: &mut Vec<ValidationIssue>, path: &str, code: &str, message: &str) {
    issues.push(ValidationIssue {
        path: path.into(),
        code: code.into(),
        message: message.into(),
    });
}

pub(super) fn output_schema(
    spec: &StreamAsofJoinSpec,
    left: &Schema,
    right: &Schema,
) -> Result<SchemaRef> {
    if let Some(issue) = schema_issues(spec, left, right).first() {
        return Err(invalid(&issue.path, &issue.message));
    }
    let fields = [(spec.left(), left, false), (spec.right(), right, true)]
        .into_iter()
        .flat_map(|(side, schema, nullable)| {
            schema.fields().iter().map(move |field| {
                field
                    .as_ref()
                    .clone()
                    .with_name(format!("{}__{}", side.prefix(), field.name()))
                    .with_nullable(nullable || field.is_nullable())
            })
        })
        .collect::<Vec<_>>();
    Ok(Arc::new(Schema::new(fields)))
}

fn payload_type(data_type: &DataType) -> bool {
    if let DataType::FixedSizeBinary(width) = data_type {
        return *width >= 0;
    }
    matches!(
        data_type,
        DataType::Null
            | DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float16
            | DataType::Float32
            | DataType::Float64
            | DataType::Date32
            | DataType::Date64
            | DataType::Time32(TimeUnit::Second | TimeUnit::Millisecond)
            | DataType::Time64(TimeUnit::Microsecond | TimeUnit::Nanosecond)
            | DataType::Timestamp(_, _)
            | DataType::Duration(_)
            | DataType::Interval(_)
            | DataType::Decimal32(_, _)
            | DataType::Decimal64(_, _)
            | DataType::Decimal128(_, _)
            | DataType::Decimal256(_, _)
            | DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Binary
            | DataType::LargeBinary
    )
}
