//! Exact Arrow equality encoding, independent of IPC type normalization.

use std::collections::HashMap;

use datafusion::arrow::datatypes::{DataType, Field, SchemaRef};
use sha2::{Digest, Sha256};

pub(in crate::operator::join) fn digest(schema: &SchemaRef) -> String {
    let mut hash = Sha256::new();
    text(&mut hash, "calc-flow/join/retained-schema/v1");
    metadata(&mut hash, schema.metadata());
    count(&mut hash, schema.fields().len());
    for field in schema.fields() {
        field_hash(&mut hash, field);
    }
    hex::encode(hash.finalize())
}

fn text(hash: &mut Sha256, text: &str) {
    count(hash, text.len());
    hash.update(text.as_bytes());
}

fn count(hash: &mut Sha256, length: usize) {
    hash.update((length as u64).to_le_bytes());
}

fn metadata(hash: &mut Sha256, metadata: &HashMap<String, String>) {
    let mut entries = metadata.iter().collect::<Vec<_>>();
    entries.sort_unstable_by(|left, right| left.0.cmp(right.0));
    count(hash, entries.len());
    for (key, value) in entries {
        text(hash, key);
        text(hash, value);
    }
}

fn field_hash(hash: &mut Sha256, field: &Field) {
    text(hash, field.name());
    hash.update([u8::from(field.is_nullable())]);
    metadata(hash, field.metadata());
    type_hash(hash, field.data_type());
}

fn type_hash(hash: &mut Sha256, data_type: &DataType) {
    if let Some((tag, field, parameter)) = unary(data_type) {
        text(hash, tag);
        hash.update(parameter.to_le_bytes());
        field_hash(hash, field);
        return;
    }
    composite_or_scalar(hash, data_type);
}

fn unary(data_type: &DataType) -> Option<(&'static str, &Field, i64)> {
    match data_type {
        DataType::List(field) => Some(("List", field, 0)),
        DataType::LargeList(field) => Some(("LargeList", field, 0)),
        DataType::ListView(field) => Some(("ListView", field, 0)),
        DataType::LargeListView(field) => Some(("LargeListView", field, 0)),
        DataType::FixedSizeList(field, width) => Some(("FixedSizeList", field, i64::from(*width))),
        DataType::Map(field, ordered) => Some(("Map", field, i64::from(*ordered))),
        _ => None,
    }
}

fn composite_or_scalar(hash: &mut Sha256, data_type: &DataType) {
    match data_type {
        DataType::Dictionary(key, value) => {
            text(hash, "Dictionary");
            type_hash(hash, key);
            type_hash(hash, value);
        }
        DataType::Struct(fields) => {
            text(hash, "Struct");
            count(hash, fields.len());
            for field in fields {
                field_hash(hash, field);
            }
        }
        DataType::Union(fields, mode) => {
            text(hash, "Union");
            text(hash, &format!("{mode:?}"));
            count(hash, fields.len());
            for (id, field) in fields.iter() {
                hash.update(id.to_le_bytes());
                field_hash(hash, field);
            }
        }
        DataType::RunEndEncoded(ends, values) => {
            text(hash, "RunEndEncoded");
            field_hash(hash, ends);
            field_hash(hash, values);
        }
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
        | DataType::Timestamp(..)
        | DataType::Date32
        | DataType::Date64
        | DataType::Time32(_)
        | DataType::Time64(_)
        | DataType::Duration(_)
        | DataType::Interval(_)
        | DataType::Binary
        | DataType::FixedSizeBinary(_)
        | DataType::LargeBinary
        | DataType::BinaryView
        | DataType::Utf8
        | DataType::LargeUtf8
        | DataType::Utf8View
        | DataType::Decimal32(..)
        | DataType::Decimal64(..)
        | DataType::Decimal128(..)
        | DataType::Decimal256(..) => text(hash, &format!("{data_type:?}")),
        DataType::List(_)
        | DataType::LargeList(_)
        | DataType::ListView(_)
        | DataType::LargeListView(_)
        | DataType::FixedSizeList(..)
        | DataType::Map(..) => unreachable!("unary fields were encoded before scalar dispatch"),
    }
}
