use super::{
    AggregateFunctionExpr, Arc, DataType, MemoryReservation, Result, ScalarValue, checked_bytes,
    df_error, ensure_reservation, native_key_bytes,
};
use crate::DataFusionConfig;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(in crate::operator::sql) enum Policy {
    ExactNumericV1,
    SequentialGroupedFloatV1(SequentialPolicy),
    GlobalRecordFloatV1(super::global_record::Policy),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::operator::sql) struct SequentialPolicy {
    pub max_record_rows: u64,
    #[serde(deserialize_with = "required_config")]
    pub config: DataFusionConfig,
    pub factory: Factory,
    pub model: Model,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(in crate::operator::sql) enum Factory {
    PrimitiveV1,
    BooleanV1,
    Utf8V1,
    LargeUtf8V1,
    ColumnV1,
}

pub(super) fn key_layout(data_type: &DataType) -> Option<(Factory, usize)> {
    match data_type {
        DataType::Utf8 => Some((Factory::Utf8V1, size_of::<ScalarValue>())),
        DataType::LargeUtf8 => Some((Factory::LargeUtf8V1, size_of::<ScalarValue>())),
        _ => super::native_key_width(data_type).map(|width| {
            (
                if width == 0 {
                    Factory::BooleanV1
                } else {
                    Factory::PrimitiveV1
                },
                width,
            )
        }),
    }
}

pub(super) fn group_layout(
    keys: &[usize],
    schema: &datafusion::arrow::datatypes::Schema,
) -> Option<(Factory, usize)> {
    if let [key] = keys {
        return key_layout(schema.field(*key).data_type());
    }
    if keys.is_empty()
        || !keys
            .iter()
            .all(|key| key_layout(schema.field(*key).data_type()).is_some())
    {
        return None;
    }
    keys.len()
        .checked_mul(size_of::<ScalarValue>())
        .map(|width| (Factory::ColumnV1, width))
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(in crate::operator::sql) enum Model {
    #[serde(rename = "df54_single_linear_memtable_v1")]
    Df54SingleLinearMemtableV1,
}

pub(super) fn required_config<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<DataFusionConfig, D::Error> {
    let value = serde_json::Value::deserialize(deserializer)?;
    let object = value
        .as_object()
        .ok_or_else(|| serde::de::Error::custom("configuration must be an object"))?;
    let fields = [
        "batch_size",
        "target_partitions",
        "parallelism_mode",
        "max_partitions",
        "min_rows_per_partition",
        "small_rows_threshold",
        "enable_rolling_rewrite",
        "collect_diagnostics",
    ];
    if object.len() != fields.len() || fields.iter().any(|field| !object.contains_key(*field)) {
        return Err(serde::de::Error::custom(
            "configuration fields must be complete",
        ));
    }
    serde_json::from_value(value).map_err(serde::de::Error::custom)
}

pub(super) struct Proof {
    pub policy: SequentialPolicy,
    reservation: MemoryReservation,
}

impl Proof {
    pub fn new(
        reservation: MemoryReservation,
        config: DataFusionConfig,
        groups: usize,
        rows: usize,
        (factory, width): (Factory, usize),
        aggregates: &[Arc<AggregateFunctionExpr>],
        name: &str,
    ) -> Result<Self> {
        let bytes = headroom(groups, rows, width, aggregates, name)?;
        ensure_reservation(&reservation, bytes, name)?;
        Ok(Self {
            policy: SequentialPolicy {
                max_record_rows: u64::try_from(rows).map_err(|error| df_error(name, error))?,
                config,
                factory,
                model: Model::Df54SingleLinearMemtableV1,
            },
            reservation,
        })
    }

    pub fn grow(
        &self,
        groups: usize,
        width: usize,
        aggregates: &[Arc<AggregateFunctionExpr>],
        name: &str,
    ) -> Result<()> {
        let rows =
            usize::try_from(self.policy.max_record_rows).map_err(|error| df_error(name, error))?;
        ensure_reservation(
            &self.reservation,
            headroom(groups, rows, width, aggregates, name)?,
            name,
        )
    }
}

pub(super) fn selected(expression: &AggregateFunctionExpr) -> bool {
    (matches!(expression.fun().name(), "min" | "max")
        && matches!(
            expression.field().data_type(),
            DataType::Float32 | DataType::Float64
        ))
        || super::grouped_sum::selected(expression)
}

pub(super) fn reset(value: &ScalarValue, name: &str) -> Result<ScalarValue> {
    match value {
        ScalarValue::Float32(value) => Ok(ScalarValue::Float32(
            value.map(|_| f32::from_bits(0x7fc0_0001)),
        )),
        ScalarValue::Float64(value) => Ok(ScalarValue::Float64(
            value.map(|_| f64::from_bits(0x7ff8_0000_0000_0001)),
        )),
        _ => Err(df_error(
            name,
            "sequential extrema state must be floating point",
        )),
    }
}

pub(super) fn capacity(groups: usize, name: &str) -> Result<usize> {
    groups
        .max(4)
        .checked_next_power_of_two()
        .and_then(|capacity| capacity.checked_mul(2))
        .ok_or_else(|| df_error(name, "sequential group capacity overflowed"))
}

fn headroom(
    groups: usize,
    rows: usize,
    width: usize,
    aggregates: &[Arc<AggregateFunctionExpr>],
    name: &str,
) -> Result<usize> {
    let capacity = capacity(groups, name)?;
    let bitmap = checked_bytes(128, [(capacity.div_ceil(512), 64)], name)?;
    aggregates.iter().try_fold(
        native_key_bytes(groups, rows, width, name)?,
        |bytes, aggregate| {
            let fields = aggregate
                .state_fields()
                .map_err(|error| df_error(name, error))?;
            fields
                .iter()
                .try_fold(checked_bytes(bytes, [(1, 4096)], name)?, |bytes, field| {
                    let width = super::variable_extrema::state_width(field.data_type(), name)?;
                    checked_bytes(bytes, [(capacity, width * 2), (bitmap, 6)], name)
                })
        },
    )
}

impl Policy {
    pub fn label(&self) -> &'static str {
        match self {
            Self::ExactNumericV1 => "exact_numeric_v1",
            Self::SequentialGroupedFloatV1(_) => "sequential_grouped_float_v1",
            Self::GlobalRecordFloatV1(_) => "global_record_float_v1",
        }
    }

    pub fn validate(&self, expected: &Self, rows: u64, name: &str) -> Result<()> {
        match (self, expected) {
            (Self::ExactNumericV1, Self::ExactNumericV1) => Ok(()),
            (Self::GlobalRecordFloatV1(actual), Self::GlobalRecordFloatV1(trusted))
                if actual == trusted && rows != 0 =>
            {
                Ok(())
            }
            (Self::SequentialGroupedFloatV1(actual), Self::SequentialGroupedFloatV1(trusted)) => {
                if actual.config != trusted.config
                    || actual.factory != trusted.factory
                    || actual.model != trusted.model
                    || actual.max_record_rows > rows
                    || (rows != 0 && actual.max_record_rows == 0)
                {
                    return Err(df_error(
                        name,
                        "sequential checkpoint proof differs from trusted model",
                    ));
                }
                Ok(())
            }
            _ => Err(df_error(
                name,
                "native checkpoint strategy differs from trusted plan",
            )),
        }
    }
}

#[cfg(test)]
#[path = "grouped_float_proof_tests.rs"]
mod tests;
