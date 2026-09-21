//! Python-hosted Kafka payload decoders (feature `connector-kafka`).
//!
//! A registered Python callable receives the raw payload `bytes` and
//! returns a `pyarrow.RecordBatch` or `pyarrow.Table`; the wrapper
//! enforces the explicit schema, the decode bounds, and the safe
//! connector error surface before the batch reaches the source edge.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use calc_flow::{
    ArrowFieldSpec, Batch, BatchMetadata, CalcFlowError, ConnectorError, ConnectorIdentity,
    ConnectorOperation, DecodeBounds, FormatDecoder, FormatIdentity, Result,
};
use calc_flow_connectors::arrow_schema::schema_from_spec;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::record_batch::RecordBatch;
use pyo3::prelude::*;
use pyo3_arrow::PyTable;

use crate::config::PythonRoot;
use crate::error::to_py_err;

/// The trusted provider namespace reported by Python-hosted decoders.
const PROVIDER: &str = "calc-flow-python";

/// A [`FormatDecoder`] backed by one Python callable.
///
/// The `pyarrow` module handle and the parsed explicit schema are fixed
/// per source configuration, so both are cached on first use behind
/// mutexes, mirroring `ProtobufCodec`'s plan cache; the per-message path
/// validates and normalizes output to the source's exact schema.
pub(crate) struct PythonKafkaDecoder {
    identity: FormatIdentity,
    root: Arc<PythonRoot>,
    pyarrow: Mutex<Option<Py<PyModule>>>,
    schema_plan: Mutex<Option<(Vec<ArrowFieldSpec>, SchemaRef)>>,
}

impl PythonKafkaDecoder {
    /// Wraps the callable under the given identity, returning the GC root
    /// the runtime must retain.
    pub(crate) fn new(
        py: Python<'_>,
        name: &str,
        version: &str,
        function: Py<PyAny>,
    ) -> PyResult<(Self, Arc<PythonRoot>)> {
        if !function.bind(py).is_callable() {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "function must be callable",
            ));
        }
        let identity = FormatIdentity::new(name, version).map_err(to_py_err)?;
        let root = Arc::new(PythonRoot::new(function));
        Ok((
            Self {
                identity,
                root: Arc::clone(&root),
                pyarrow: Mutex::new(None),
                schema_plan: Mutex::new(None),
            },
            root,
        ))
    }

    fn failure(&self, detail: &str) -> CalcFlowError {
        CalcFlowError::Connector(ConnectorError::new(
            ConnectorIdentity::new(PROVIDER, &self.identity.name, &self.identity.version)
                .expect("format identities carry non-empty components"),
            ConnectorOperation::new("decode").expect("the operation name is non-empty"),
            detail,
        ))
    }

    fn assemble(
        &self,
        batches: Vec<RecordBatch>,
        bounds: &DecodeBounds,
        schema: &[ArrowFieldSpec],
    ) -> Result<Batch> {
        let expected = self.expected_schema(schema)?;
        let batches = batches
            .into_iter()
            .map(|batch| self.normalize_schema(batch, expected.as_ref()))
            .collect::<Result<Vec<_>>>()?;
        let rows: u64 = batches
            .iter()
            .map(|batch| u64::try_from(batch.num_rows()).unwrap_or(u64::MAX))
            .sum();
        let bytes: u64 = batches
            .iter()
            .map(|batch| u64::try_from(batch.get_array_memory_size()).unwrap_or(u64::MAX))
            .sum();
        bounds
            .check(&self.identity, rows, bytes)
            .map_err(|error| self.failure(&error.to_string()))?;
        Batch::table(
            batches,
            BatchMetadata::new(self.identity.name.as_ref(), 0, BTreeMap::new())?,
        )
    }

    fn normalize_schema(
        &self,
        batch: RecordBatch,
        expected: Option<&SchemaRef>,
    ) -> Result<RecordBatch> {
        let Some(expected) = expected else {
            return Ok(batch);
        };
        let batch_schema = batch.schema();
        let fields = batch_schema.fields();
        let agrees = fields.len() == expected.fields().len()
            && fields
                .iter()
                .zip(expected.fields())
                .all(|(field, expected)| {
                    field.name() == expected.name() && field.data_type() == expected.data_type()
                });
        if !agrees {
            return Err(self.failure(
                "decoded batch fields do not match the explicit schema in name and type",
            ));
        }
        RecordBatch::try_new(Arc::clone(expected), batch.columns().to_vec())
            .map_err(|error| self.failure(&error.to_string()))
    }

    /// Caches the same exact schema the Kafka source advertises.
    fn expected_schema(&self, spec: &[ArrowFieldSpec]) -> Result<Option<SchemaRef>> {
        if spec.is_empty() {
            return Ok(None);
        }
        let mut cached = self
            .schema_plan
            .lock()
            .expect("the schema plan lock is never poisoned by planning");
        if let Some((cached_spec, plan)) = &*cached {
            if cached_spec.as_slice() == spec {
                return Ok(Some(Arc::clone(plan)));
            }
        }
        let plan = schema_from_spec(spec).map_err(|error| self.failure(&error.to_string()))?;
        *cached = Some((spec.to_vec(), Arc::clone(&plan)));
        Ok(Some(plan))
    }

    /// Imports `pyarrow` once per decoder instead of once per message.
    fn pyarrow_module(&self, py: Python<'_>) -> std::result::Result<Py<PyModule>, String> {
        let mut cached = self
            .pyarrow
            .lock()
            .expect("the pyarrow module lock is never poisoned by import");
        if let Some(module) = &*cached {
            return Ok(module.clone_ref(py));
        }
        let module = py
            .import("pyarrow")
            .map_err(|error| error.to_string())?
            .unbind();
        *cached = Some(module.clone_ref(py));
        Ok(module)
    }
}

impl FormatDecoder for PythonKafkaDecoder {
    fn identity(&self) -> FormatIdentity {
        self.identity.clone()
    }

    fn decode(
        &self,
        bytes: &[u8],
        bounds: &DecodeBounds,
        schema: &[ArrowFieldSpec],
    ) -> Result<Batch> {
        let batches = Python::attach(|py| {
            let pyarrow = self
                .pyarrow_module(py)
                .map_err(|error| self.failure(&error))?;
            let payload = pyo3::types::PyBytes::new(py, bytes);
            let output = self
                .root
                .object()
                .call1(py, (payload,))
                .map_err(|error| self.failure(&error.to_string()))?;
            python_output_to_batches(output.bind(py), pyarrow.bind(py))
                .map_err(|error| self.failure(&error))
        })?;
        self.assemble(batches, bounds, schema)
    }
}

fn python_output_to_batches(
    output: &Bound<'_, PyAny>,
    pyarrow: &Bound<'_, PyModule>,
) -> std::result::Result<Vec<RecordBatch>, String> {
    let batch_type = pyarrow
        .getattr("RecordBatch")
        .map_err(|error| error.to_string())?;
    let table_type = pyarrow
        .getattr("Table")
        .map_err(|error| error.to_string())?;
    if !is_instance_of(output, &batch_type)? && !is_instance_of(output, &table_type)? {
        return Err("decoder output must be a pyarrow.RecordBatch or pyarrow.Table".into());
    }
    let (mut batches, schema) = output
        .extract::<PyTable>()
        .map_err(|error| error.to_string())?
        .into_inner();
    if batches.is_empty() {
        batches.push(RecordBatch::new_empty(schema));
    }
    Ok(batches)
}

fn is_instance_of(
    output: &Bound<'_, PyAny>,
    ty: &Bound<'_, PyAny>,
) -> std::result::Result<bool, String> {
    output.is_instance(ty).map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::datatypes::DataType;

    fn order_decoder(py: Python<'_>) -> (PythonKafkaDecoder, Arc<PythonRoot>) {
        let module = PyModule::from_code(
            py,
            c"import pyarrow as pa\n\ndef decode(payload):\n    parts = payload.decode('utf-8').split('|')\n    return pa.record_batch([[int(parts[0])], [int(parts[1])], [float(parts[2])]], names=['id', 'quantity', 'price'])\n",
            c"orders_decode.py",
            c"orders_decode",
        )
        .expect("fixture module");
        let function = module.getattr("decode").expect("fixture callable").unbind();
        PythonKafkaDecoder::new(py, "pipe-orders", "1", function).expect("decoder wraps")
    }

    fn order_schema() -> Vec<ArrowFieldSpec> {
        vec![
            ArrowFieldSpec {
                name: "id".into(),
                data_type: "int64".into(),
                nullable: false,
            },
            ArrowFieldSpec {
                name: "quantity".into(),
                data_type: "int64".into(),
                nullable: false,
            },
            ArrowFieldSpec {
                name: "price".into(),
                data_type: "float64".into(),
                nullable: false,
            },
        ]
    }

    #[test]
    fn test_declared_nullability_normalizes_all_actual_nullability_combinations() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::from_code(
                py,
                c"import pyarrow as pa\n\ndef decode(payload):\n    schema = pa.schema([pa.field('value', pa.int64(), nullable=payload == b'nullable')])\n    return pa.record_batch([[1]], schema=schema)\n",
                c"nullable_decode.py",
                c"nullable_decode",
            )
            .unwrap();
            let (decoder, _root) = PythonKafkaDecoder::new(
                py,
                "nullable",
                "1",
                module.getattr("decode").unwrap().unbind(),
            )
            .unwrap();
            let bounds = DecodeBounds::new(16, 1 << 20).unwrap();
            for nullable in [false, true] {
                let spec = vec![ArrowFieldSpec {
                    name: "value".into(),
                    data_type: "int64".into(),
                    nullable,
                }];
                let expected = schema_from_spec(&spec).unwrap();
                for payload in [b"nullable".as_slice(), b"required".as_slice()] {
                    let batch = decoder.decode(payload, &bounds, &spec).unwrap();
                    assert_eq!(batch.table_payload().unwrap().schema(), &expected);
                    assert_eq!(batch.num_rows(), 1);
                }
            }
        });
    }

    #[test]
    fn test_declared_non_nullable_fields_reject_nulls() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::from_code(
                py,
                c"import pyarrow as pa\n\ndef decode(payload):\n    schema = pa.schema([pa.field('value', pa.int64(), nullable=payload != b'required')])\n    batch = pa.record_batch([[None]], schema=schema)\n    if payload == b'table':\n        return pa.Table.from_batches([pa.record_batch([[1]], schema=schema), batch])\n    return batch\n",
                c"null_decode.py",
                c"null_decode",
            ).unwrap();
            let (decoder, _root) = PythonKafkaDecoder::new(
                py,
                "nulls",
                "1",
                module.getattr("decode").unwrap().unbind(),
            )
            .unwrap();
            let bounds = DecodeBounds::new(16, 1 << 20).unwrap();
            for nullable in [false, true] {
                let spec = vec![ArrowFieldSpec {
                    name: "value".into(),
                    data_type: "int64".into(),
                    nullable,
                }];
                for payload in [
                    b"batch".as_slice(),
                    b"table".as_slice(),
                    b"required".as_slice(),
                ] {
                    let result = decoder.decode(payload, &bounds, &spec);
                    if nullable && payload != b"required" {
                        let batch = result.unwrap();
                        let records = batch.table_payload().unwrap().batches();
                        assert_eq!(records.last().unwrap().column(0).null_count(), 1);
                    } else {
                        let error =
                            result.expect_err("required fields must reject actual NULL values");
                        assert!(matches!(error, CalcFlowError::Connector(_)), "{error}");
                        assert!(error.to_string().contains("value"), "{error}");
                        assert!(error.to_string().contains("null"), "{error}");
                    }
                }
            }
        });
    }

    #[test]
    fn test_declared_schema_normalizes_callback_metadata_without_mutation() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::from_code(
                py,
                c"import pyarrow as pa\n\nschema = pa.schema([pa.field('value', pa.int64(), nullable=False, metadata={'unit': 'count'})], metadata={'producer': 'callback'})\noriginal = pa.record_batch([[1]], schema=schema)\ndef decode(payload):\n    return original\n",
                c"metadata_decode.py",
                c"metadata_decode",
            ).unwrap();
            let (decoder, _root) = PythonKafkaDecoder::new(
                py,
                "metadata",
                "1",
                module.getattr("decode").unwrap().unbind(),
            )
            .unwrap();
            let bounds = DecodeBounds::new(16, 1 << 20).unwrap();
            let spec = vec![ArrowFieldSpec {
                name: "value".into(),
                data_type: "int64".into(),
                nullable: false,
            }];
            let original = decoder.decode(b"ignored", &bounds, &[]).unwrap();
            let normalized = decoder.decode(b"ignored", &bounds, &spec).unwrap();
            let expected = schema_from_spec(&spec).unwrap();
            assert_eq!(normalized.table_payload().unwrap().schema(), &expected);
            let again = decoder.decode(b"ignored", &bounds, &[]).unwrap();
            let schema = again.table_payload().unwrap().schema();
            assert_eq!(schema, original.table_payload().unwrap().schema());
            assert_eq!(schema.metadata().get("producer").unwrap(), "callback");
            assert_eq!(schema.field(0).metadata().get("unit").unwrap(), "count");
        });
    }

    #[test]
    fn test_empty_table_and_record_batch_preserve_schema() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::from_code(
                py,
                c"import pyarrow as pa\n\nschema = pa.schema([pa.field('value', pa.int64(), nullable=False, metadata={'unit': 'count'})], metadata={'producer': 'callback'})\ndef decode(payload):\n    if payload == b'batch':\n        return pa.record_batch([[]], schema=schema)\n    return pa.Table.from_batches([], schema=schema)\n",
                c"empty_decode.py",
                c"empty_decode",
            ).unwrap();
            let (decoder, _root) = PythonKafkaDecoder::new(
                py,
                "empty",
                "1",
                module.getattr("decode").unwrap().unbind(),
            )
            .unwrap();
            let bounds = DecodeBounds::new(16, 1 << 20).unwrap();
            let spec = vec![ArrowFieldSpec {
                name: "value".into(),
                data_type: "int64".into(),
                nullable: false,
            }];
            for fields in [spec.as_slice(), &[]] {
                let empty_batch = decoder.decode(b"batch", &bounds, fields).unwrap();
                let empty_table = decoder.decode(b"table", &bounds, fields).unwrap();
                let table = empty_table.table_payload().unwrap();
                assert_eq!(empty_table.num_rows(), 0);
                assert_eq!(table.batches().len(), 1);
                assert_eq!(
                    table.schema(),
                    empty_batch.table_payload().unwrap().schema()
                );
                assert!(!table.schema().field(0).is_nullable());
                if fields.is_empty() {
                    assert_eq!(
                        table.schema().metadata().get("producer").unwrap(),
                        "callback"
                    );
                    assert_eq!(
                        table.schema().field(0).metadata().get("unit").unwrap(),
                        "count"
                    );
                }
            }
            for (name, data_type) in [("wrong", "int64"), ("value", "string")] {
                let wrong = vec![ArrowFieldSpec {
                    name: name.into(),
                    data_type: data_type.into(),
                    nullable: false,
                }];
                for payload in [b"batch".as_slice(), b"table".as_slice()] {
                    let error = decoder.decode(payload, &bounds, &wrong).unwrap_err();
                    assert!(matches!(error, CalcFlowError::Connector(_)), "{error}");
                    assert!(error.to_string().contains("name and type"), "{error}");
                }
            }
        });
    }

    #[test]
    fn test_fieldless_output_preserves_rows_and_decode_bounds() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::from_code(
                py,
                c"import pyarrow as pa\n\ndef decode(payload):\n    batch = pa.record_batch([[1, 2]], names=['value']).select([])\n    return batch if payload == b'batch' else pa.Table.from_batches([batch])\n",
                c"fieldless_decode.py",
                c"fieldless_decode",
            ).unwrap();
            let (decoder, _root) = PythonKafkaDecoder::new(
                py,
                "fieldless",
                "1",
                module.getattr("decode").unwrap().unbind(),
            )
            .unwrap();
            for payload in [b"batch".as_slice(), b"table".as_slice()] {
                let batch = decoder
                    .decode(payload, &DecodeBounds::new(2, 1024).unwrap(), &[])
                    .unwrap();
                assert_eq!(batch.num_rows(), 2);
                assert!(batch.table_payload().unwrap().schema().fields().is_empty());
                assert!(
                    decoder
                        .decode(payload, &DecodeBounds::new(1, 1024).unwrap(), &[])
                        .is_err()
                );
            }
        });
    }

    #[test]
    fn python_callable_decodes_within_bounds_and_schema() {
        Python::initialize();
        Python::attach(|py| {
            let (decoder, _root) = order_decoder(py);
            let bounds = DecodeBounds::new(16, 1 << 20).expect("bounds");
            let batch = decoder
                .decode(b"1|2|10.0", &bounds, &order_schema())
                .expect("python decoder decodes");
            assert_eq!(batch.num_rows(), 1);
            let payload = batch.table_payload().expect("table batch");
            let record = &payload.batches()[0];
            assert_eq!(record.num_rows(), 1);
            let record_schema = record.schema();
            let shapes: Vec<(&str, &DataType)> = record_schema
                .fields()
                .iter()
                .map(|field| (field.name().as_str(), field.data_type()))
                .collect();
            assert_eq!(
                shapes,
                vec![
                    ("id", &DataType::Int64),
                    ("quantity", &DataType::Int64),
                    ("price", &DataType::Float64),
                ]
            );

            let error = decoder
                .decode(b"garbage", &bounds, &order_schema())
                .expect_err("malformed payloads surface the python failure");
            assert!(matches!(error, CalcFlowError::Connector(_)), "{error}");

            let tight = DecodeBounds::new(16, 8).expect("tight bounds");
            assert!(
                decoder
                    .decode(b"1|2|10.0", &tight, &order_schema())
                    .is_err(),
                "the byte bound trips"
            );

            let mut mismatched = order_schema();
            mismatched[0] = ArrowFieldSpec {
                name: "id".into(),
                data_type: "string".into(),
                nullable: false,
            };
            assert!(
                decoder.decode(b"1|2|10.0", &bounds, &mismatched).is_err(),
                "schema disagreement fails closed"
            );
            assert!(
                decoder
                    .decode(b"1|2|10.0", &bounds, &order_schema()[1..])
                    .is_err(),
                "column count disagreement fails closed"
            );
        });
    }

    #[test]
    fn schema_replanning_replaces_the_cached_parse() {
        Python::initialize();
        Python::attach(|py| {
            let (decoder, _root) = order_decoder(py);
            let bounds = DecodeBounds::new(16, 1 << 20).expect("bounds");
            decoder
                .decode(b"1|2|10.0", &bounds, &order_schema())
                .expect("the first schema parses and caches");
            let mut renamed = order_schema();
            renamed[0] = ArrowFieldSpec {
                name: "order_id".into(),
                data_type: "int64".into(),
                nullable: false,
            };
            assert!(
                decoder.decode(b"1|2|10.0", &bounds, &renamed).is_err(),
                "a changed schema re-parses and fails closed"
            );
            decoder
                .decode(b"1|2|10.0", &bounds, &order_schema())
                .expect("returning to the original schema decodes again");
        });
    }

    #[test]
    fn non_callable_registration_fails() {
        Python::initialize();
        Python::attach(|py| {
            let not_callable = pyo3::types::PyInt::new(py, 42).unbind().into_any();
            assert!(PythonKafkaDecoder::new(py, "x", "1", not_callable).is_err());
        });
    }

    #[test]
    fn table_outputs_flatten_to_their_batches() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::from_code(
                py,
                c"import pyarrow as pa\n\ndef decode(payload):\n    return pa.table({'id': pa.array([1], type=pa.int64()), 'quantity': pa.array([2], type=pa.int64()), 'price': pa.array([10.0], type=pa.float64())})\n",
                c"orders_table.py",
                c"orders_table",
            )
            .expect("fixture module");
            let function = module.getattr("decode").expect("callable").unbind();
            let (decoder, _root) =
                PythonKafkaDecoder::new(py, "table-orders", "1", function).expect("decoder wraps");
            let bounds = DecodeBounds::new(16, 1 << 20).expect("bounds");
            let batch = decoder
                .decode(b"ignored", &bounds, &order_schema())
                .expect("table output decodes");
            assert_eq!(batch.num_rows(), 1);
        });
    }
}
