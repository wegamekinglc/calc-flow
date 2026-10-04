use super::{RecordColumns, ScalarColumn, TypedValues};

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub(super) struct IntegerKey {
    values: [u64; 2],
    valid: u8,
}

impl IntegerKey {
    pub(super) fn read(columns: &RecordColumns<'_>, row: usize) -> Option<Self> {
        if !matches!(columns.groups.len(), 1 | 2) {
            return None;
        }
        let mut key = Self {
            values: [0; 2],
            valid: 0,
        };
        for (ordinal, (column, _)) in columns.groups.iter().enumerate() {
            match integer_value(column, row) {
                IntegerValue::Unsupported => return None,
                IntegerValue::Null => {}
                IntegerValue::Value(value) => {
                    key.values[ordinal] = value;
                    key.valid |= 1 << ordinal;
                }
            }
        }
        Some(key)
    }
}

enum IntegerValue {
    Unsupported,
    Null,
    Value(u64),
}

fn integer_value(column: &ScalarColumn<'_>, row: usize) -> IntegerValue {
    let value = match column.values {
        TypedValues::Int8(array) => u64::from_ne_bytes(i64::from(array.value(row)).to_ne_bytes()),
        TypedValues::Int16(array) => u64::from_ne_bytes(i64::from(array.value(row)).to_ne_bytes()),
        TypedValues::Int32(array) => u64::from_ne_bytes(i64::from(array.value(row)).to_ne_bytes()),
        TypedValues::Int64(array) => u64::from_ne_bytes(array.value(row).to_ne_bytes()),
        TypedValues::UInt8(array) => u64::from(array.value(row)),
        TypedValues::UInt16(array) => u64::from(array.value(row)),
        TypedValues::UInt32(array) => u64::from(array.value(row)),
        TypedValues::UInt64(array) => array.value(row),
        _ => return IntegerValue::Unsupported,
    };
    if column.is_null(row) {
        IntegerValue::Null
    } else {
        IntegerValue::Value(value)
    }
}
