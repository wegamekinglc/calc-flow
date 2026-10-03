use super::{StreamAsofJoinOperator, arrow_error, checked, codec, workspace};
use crate::{BatchKind, CalcFlowError, Port, Result};
use datafusion::{arrow::datatypes::SchemaRef, execution::memory_pool::MemoryReservation};
use serde::Serialize;
use std::sync::Arc;

pub(super) struct PayloadProjection {
    pub columns: [Vec<usize>; 2],
    pub schemas: [SchemaRef; 2],
    pub digests: [[u8; 32]; 2],
    pub headers: [u64; 2],
    _lease: MemoryReservation,
}

#[derive(Serialize)]
pub(super) struct RetainedDescriptor {
    pub columns: [Vec<usize>; 2],
    pub schema_digests: [String; 2],
}

impl StreamAsofJoinOperator {
    pub(super) fn configure_projection(&mut self, columns: Vec<usize>) -> Result<()> {
        let lease = self.projection_configuration_lease(columns.len())?;
        let output_schema = self.schemas[2]
            .project(&columns)
            .map_err(|error| arrow_error(&error))?;
        let logical_left = self.schemas[0].fields().len();
        let retained = self.retained_output_dependencies(&columns);
        let schemas = [
            Arc::new(
                self.schemas[0]
                    .project(&retained[0])
                    .map_err(|error| arrow_error(&error))?,
            ),
            Arc::new(
                self.schemas[1]
                    .project(&retained[1])
                    .map_err(|error| arrow_error(&error))?,
            ),
        ];
        let digests = [
            codec::schema_digest(&schemas[0])?,
            codec::schema_digest(&schemas[1])?,
        ];
        let headers = [
            workspace::payload_header_bytes(&schemas[0])?,
            workspace::payload_header_bytes(&schemas[1])?,
        ];
        let physical = columns
            .into_iter()
            .map(|index| {
                let side = usize::from(index >= logical_left);
                let logical = if side == 0 {
                    index
                } else {
                    index - logical_left
                };
                let column = retained[side]
                    .binary_search(&logical)
                    .expect("retained ASOF output");
                (side, column)
            })
            .collect::<Vec<_>>();
        let selected = std::array::from_fn(|side| {
            physical
                .iter()
                .filter_map(|&(source, column)| (source == side).then_some(column))
                .collect()
        });
        let combined = physical
            .into_iter()
            .map(|(side, column)| {
                column
                    + if side == 0 {
                        0
                    } else {
                        schemas[0].fields().len()
                    }
            })
            .collect();
        let output = Port::with_schema_ref(
            "output",
            BatchKind::Table,
            true,
            Some(Arc::new(output_schema)),
        )?;
        self.outputs[0] = output;
        self.runtime.set_output_projection(combined);
        self.output_columns = Some(selected);
        self.payload_projection = Some(Box::new(PayloadProjection {
            columns: retained,
            schemas,
            digests,
            headers,
            _lease: lease,
        }));
        Ok(())
    }

    fn projection_configuration_lease(&self, output_columns: usize) -> Result<MemoryReservation> {
        let mut charge = 16_384;
        for schema in &self.schemas {
            let fields = schema.fields().iter().try_fold(0_u64, |bytes, field| {
                checked(&self.name, bytes, field.size() as u64)
            })?;
            let metadata = schema
                .metadata()
                .iter()
                .try_fold(0_u64, |bytes, (key, value)| {
                    checked(
                        &self.name,
                        bytes,
                        (key.len() as u64)
                            .saturating_add(value.len() as u64)
                            .saturating_add(128),
                    )
                })?;
            charge = checked(
                &self.name,
                charge,
                fields.saturating_add(metadata).saturating_mul(8),
            )?;
            charge = checked(
                &self.name,
                charge,
                (schema.fields().len() as u64).saturating_mul(512),
            )?;
        }
        charge = checked(
            &self.name,
            charge,
            (output_columns as u64).saturating_mul(128),
        )?;
        self.reserve_workspace(charge)
    }

    fn retained_output_dependencies(&self, columns: &[usize]) -> [Vec<usize>; 2] {
        let logical_left = self.schemas[0].fields().len();
        std::array::from_fn(|side| {
            let declaration = if side == 0 {
                self.spec.left()
            } else {
                self.spec.right()
            };
            let schema = &self.schemas[side];
            let mut keep = vec![false; schema.fields().len()];
            for name in super::admission::identity_column_names(declaration) {
                keep[schema.index_of(name).expect("validated ASOF identity")] = true;
            }
            for &index in columns {
                if (index < logical_left) == (side == 0) {
                    keep[if side == 0 {
                        index
                    } else {
                        index - logical_left
                    }] = true;
                }
            }
            keep.into_iter()
                .enumerate()
                .filter_map(|(index, keep)| keep.then_some(index))
                .collect::<Vec<_>>()
        })
    }

    pub(super) fn physical_schema(&self, side: usize) -> &SchemaRef {
        self.payload_projection
            .as_ref()
            .map_or(&self.schemas[side], |plan| &plan.schemas[side])
    }

    pub(super) fn physical_digest(&self, side: usize) -> &[u8; 32] {
        self.payload_projection
            .as_ref()
            .map_or(&self.schema_digests[side], |plan| &plan.digests[side])
    }

    pub(super) fn physical_header(&self, side: usize) -> u64 {
        self.payload_projection
            .as_ref()
            .map_or(self.payload_header_bytes[side], |plan| plan.headers[side])
    }

    pub(super) fn retained_descriptor(&self) -> RetainedDescriptor {
        RetainedDescriptor {
            columns: std::array::from_fn(|side| {
                self.payload_projection.as_ref().map_or_else(
                    || (0..self.schemas[side].fields().len()).collect(),
                    |plan| plan.columns[side].clone(),
                )
            }),
            schema_digests: std::array::from_fn(|side| hex::encode(self.physical_digest(side))),
        }
    }

    pub(super) fn validate_retained_descriptor(
        &self,
        value: Option<&serde_json::Value>,
    ) -> Result<()> {
        let valid = value
            .and_then(serde_json::Value::as_object)
            .is_some_and(|object| {
                if object.len() != 2 {
                    return false;
                }
                let Some(columns) = object
                    .get("columns")
                    .and_then(serde_json::Value::as_array)
                    .filter(|values| values.len() == 2)
                else {
                    return false;
                };
                let Some(digests) = object
                    .get("schema_digests")
                    .and_then(serde_json::Value::as_array)
                    .filter(|values| values.len() == 2)
                else {
                    return false;
                };
                (0..2).all(|side| {
                    let Some(indices) = columns[side].as_array() else {
                        return false;
                    };
                    if indices.len() != self.physical_schema(side).fields().len() {
                        return false;
                    }
                    let matches = indices.iter().enumerate().all(|(index, value)| {
                        value.as_u64()
                            == Some(
                                self.payload_projection
                                    .as_ref()
                                    .map_or(index, |plan| plan.columns[side][index])
                                    as u64,
                            )
                    });
                    let Some(digest) = digests[side].as_str().filter(|text| {
                        text.len() == 64
                            && text
                                .bytes()
                                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
                    }) else {
                        return false;
                    };
                    let mut decoded = [0; 32];
                    matches
                        && hex::decode_to_slice(digest, &mut decoded).is_ok()
                        && decoded == *self.physical_digest(side)
                })
            });
        if valid {
            Ok(())
        } else {
            Err(CalcFlowError::CheckpointMismatch {
                message: "ASOF retained_payloads differ from the compiled physical projection"
                    .into(),
            })
        }
    }
}
