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
use datafusion::arrow::datatypes::DataType;
use datafusion::arrow::record_batch::RecordBatch;
use pyo3::prelude::*;
use pyo3_arrow::PyTable;

use crate::config::PythonRoot;
use crate::error::to_py_err;
use crate::udf::parse_arrow_type;

/// The trusted provider namespace reported by Python-hosted decoders.
const PROVIDER: &str = "calc-flow-python";

/// The parsed `(name, DataType)` plan resolved from one explicit schema.
type SchemaFields = Vec<(String, DataType)>;

/// A [`FormatDecoder`] backed by one Python callable.
///
/// The `pyarrow` module handle and the parsed explicit schema are fixed
/// per source configuration, so both are cached on first use behind
/// mutexes, mirroring `ProtobufCodec`'s plan cache; the per-message path
/// only calls the callable and compares column names and types.
pub(crate) struct PythonKafkaDecoder {
    identity: FormatIdentity,
    root: Arc<PythonRoot>,
    pyarrow: Mutex<Option<Py<PyModule>>>,
    schema_plan: Mutex<Option<(Vec<ArrowFieldSpec>, Arc<SchemaFields>)>>,
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
        for batch in &batches {
            self.check_schema(batch, schema)?;
        }
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

    fn check_schema(&self, batch: &RecordBatch, schema: &[ArrowFieldSpec]) -> Result<()> {
        let Some(expected) = self.expected_fields(schema)? else {
            return Ok(());
        };
        let batch_schema = batch.schema();
        let fields = batch_schema.fields();
        let agrees = fields.len() == expected.len()
            && fields
                .iter()
                .zip(expected.iter())
                .all(|(field, (name, data_type))| {
                    field.name() == name && field.data_type() == data_type
                });
        if !agrees {
            return Err(self.failure(
                "decoded batch fields do not match the explicit schema in name and type",
            ));
        }
        Ok(())
    }

    /// Resolves the parsed `(name, DataType)` list for one explicit
    /// schema, cached by spec so repeat decodes skip `parse_arrow_type`.
    fn expected_fields(&self, spec: &[ArrowFieldSpec]) -> Result<Option<Arc<SchemaFields>>> {
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
        let plan = Arc::new(
            spec.iter()
                .map(|field| {
                    let data_type = parse_arrow_type(&field.data_type).ok_or_else(|| {
                        self.failure(&format!(
                            "schema field {} has unsupported data type {:?}",
                            field.name, field.data_type
                        ))
                    })?;
                    Ok((field.name.clone(), data_type))
                })
                .collect::<Result<Vec<_>>>()?,
        );
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
    let (batches, _schema) = output
        .extract::<PyTable>()
        .map_err(|error| error.to_string())?
        .into_inner();
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
