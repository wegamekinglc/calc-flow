//! The protobuf format codec (feature `kafka`).
//!
//! Each payload is exactly one protobuf message, decoded through a
//! runtime-loaded `FileDescriptorSet`; unknown fields stay ignored so a
//! newer producer keeps decoding against an older descriptor. The
//! explicit schema projects the message's scalar fields by name.
//! Repeated, map, `bytes`, and nested message fields have no flat Arrow
//! representation and fail closed, as do unknown enum values. Proto3
//! implicit-presence fields decode their declared default when unset;
//! explicit-presence fields (proto2, `optional`, `oneof`) decode null
//! when the schema column is nullable and fail closed otherwise.

use std::path::Path;
use std::sync::Arc;

use arrow::array::{
    ArrayRef, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, StringArray,
    UInt32Array, UInt64Array,
};
use arrow::record_batch::RecordBatch;
use calc_flow::{ArrowFieldSpec, Batch, DecodeBounds, FormatDecoder, FormatIdentity, Result};
use prost::Message as _;
use prost_reflect::{DescriptorPool, DynamicMessage, FieldDescriptor, Kind, MessageDescriptor};

use crate::arrow_schema::{bounded_table_batch, codec_error, schema_from_spec};

/// The protobuf codec identity.
pub const IDENTITY: &str = "protobuf";

/// The codec implementation version.
pub const IDENTITY_VERSION: &str = "1";

/// Builds the protobuf codec identity.
///
/// # Errors
///
/// Returns [`calc_flow::CalcFlowError::InvalidArgument`] when the version is empty.
pub fn identity(version: &str) -> Result<FormatIdentity> {
    FormatIdentity::new(IDENTITY, version)
}

/// The single-message protobuf codec over a runtime-loaded descriptor set.
///
/// Field resolution, type validation, and the Arrow schema depend only on
/// the descriptor and the explicit schema, so they are computed once per
/// schema and cached; the per-message path decodes the wire bytes and
/// builds columns without re-validating the schema.
#[derive(Debug)]
pub struct ProtobufCodec {
    identity: FormatIdentity,
    message: MessageDescriptor,
    plan: std::sync::Mutex<Option<(Vec<ArrowFieldSpec>, Arc<SchemaPlan>)>>,
}

#[derive(Debug)]
struct SchemaPlan {
    schema: arrow::datatypes::SchemaRef,
    fields: Vec<FieldDescriptor>,
}

impl Clone for ProtobufCodec {
    fn clone(&self) -> Self {
        Self {
            identity: self.identity.clone(),
            message: self.message.clone(),
            plan: std::sync::Mutex::new(None),
        }
    }
}

impl ProtobufCodec {
    /// Creates the codec from a descriptor-set file and a message name.
    ///
    /// The descriptor set is a serialized `google.protobuf.FileDescriptorSet`
    /// as produced by `protoc --descriptor_set_out`; `message` is the
    /// fully-qualified message name, for example `events.Order`.
    ///
    /// # Errors
    ///
    /// Returns the codec's safe error when the file cannot be read, the
    /// bytes are not a valid descriptor set, or the message is absent.
    pub fn new(version: &str, descriptor_set: &Path, message: &str) -> Result<Self> {
        let identity = identity(version)?;
        let bytes = std::fs::read(descriptor_set).map_err(|error| {
            codec_error(
                &identity,
                "decode",
                &format!("the descriptor set cannot be read: {error}"),
            )
        })?;
        Self::from_descriptor_bytes(identity, &bytes, message)
    }

    fn from_descriptor_bytes(
        identity: FormatIdentity,
        bytes: &[u8],
        message: &str,
    ) -> Result<Self> {
        let set = prost_types::FileDescriptorSet::decode(bytes).map_err(|error| {
            codec_error(
                &identity,
                "decode",
                &format!("the descriptor set is not valid protobuf: {error}"),
            )
        })?;
        let pool = DescriptorPool::from_file_descriptor_set(set).map_err(|error| {
            codec_error(
                &identity,
                "decode",
                &format!("the descriptor set is invalid: {error}"),
            )
        })?;
        let message = pool.get_message_by_name(message).ok_or_else(|| {
            codec_error(
                &identity,
                "decode",
                &format!("message {message:?} is not in the descriptor set"),
            )
        })?;
        Ok(Self {
            identity,
            message,
            plan: std::sync::Mutex::new(None),
        })
    }

    fn ensure_plan(&self, spec: &[ArrowFieldSpec]) -> Result<Arc<SchemaPlan>> {
        let mut cached = self
            .plan
            .lock()
            .expect("the plan cache lock is never poisoned by planning");
        if let Some((cached_spec, plan)) = &*cached {
            if cached_spec.as_slice() == spec {
                return Ok(Arc::clone(plan));
            }
        }
        let plan = Arc::new(self.build_plan(spec)?);
        *cached = Some((spec.to_vec(), Arc::clone(&plan)));
        Ok(plan)
    }

    fn build_plan(&self, spec: &[ArrowFieldSpec]) -> Result<SchemaPlan> {
        if spec.is_empty() {
            return Err(codec_error(
                &self.identity,
                "decode",
                "protobuf payloads require an explicit schema",
            ));
        }
        let fields = spec
            .iter()
            .map(|field| self.plan_field(field))
            .collect::<Result<Vec<_>>>()?;
        Ok(SchemaPlan {
            schema: schema_from_spec(spec)?,
            fields,
        })
    }

    fn plan_field(&self, spec: &ArrowFieldSpec) -> Result<FieldDescriptor> {
        let field = self.message.get_field_by_name(&spec.name).ok_or_else(|| {
            codec_error(
                &self.identity,
                "decode",
                &format!(
                    "message {} has no field {:?}",
                    self.message.full_name(),
                    spec.name
                ),
            )
        })?;
        if field.is_list() || field.is_map() {
            return Err(codec_error(
                &self.identity,
                "decode",
                &format!(
                    "field {:?} is repeated; the protobuf codec maps scalar fields only",
                    spec.name
                ),
            ));
        }
        let Some(expected) = expected_data_type(&field.kind()) else {
            return Err(codec_error(
                &self.identity,
                "decode",
                &format!(
                    "field {:?} of kind {:?} has no flat Arrow representation",
                    spec.name,
                    field.kind()
                ),
            ));
        };
        if expected != spec.data_type {
            return Err(codec_error(
                &self.identity,
                "decode",
                &format!(
                    "field {:?} decodes to Arrow type {expected:?}, not {:?}",
                    spec.name, spec.data_type
                ),
            ));
        }
        Ok(field)
    }

    fn decode_column(
        &self,
        message: &DynamicMessage,
        field: &FieldDescriptor,
        spec: &ArrowFieldSpec,
    ) -> Result<ArrayRef> {
        if field.supports_presence() && !message.has_field(field) {
            if spec.nullable {
                return Ok(null_column(&spec.data_type));
            }
            return Err(codec_error(
                &self.identity,
                "decode",
                &format!("message omitted the non-nullable field {:?}", spec.name),
            ));
        }
        let value = message.get_field(field);
        scalar_column(&self.identity, field, &value)
    }
}

impl FormatDecoder for ProtobufCodec {
    fn identity(&self) -> FormatIdentity {
        self.identity.clone()
    }

    fn decode(
        &self,
        bytes: &[u8],
        bounds: &DecodeBounds,
        schema: &[ArrowFieldSpec],
    ) -> Result<Batch> {
        let plan = self.ensure_plan(schema)?;
        let message = DynamicMessage::decode(self.message.clone(), bytes)
            .map_err(|error| codec_error(&self.identity, "decode", &error.to_string()))?;
        let columns = plan
            .fields
            .iter()
            .zip(schema)
            .map(|(field, spec)| self.decode_column(&message, field, spec))
            .collect::<Result<Vec<_>>>()?;
        let batch = RecordBatch::try_new(plan.schema.clone(), columns)
            .map_err(|error| codec_error(&self.identity, "decode", &error.to_string()))?;
        bounded_table_batch(&self.identity, vec![batch], bounds, IDENTITY, 0)
    }
}

/// The Arrow type vocabulary entry a protobuf kind decodes into.
fn expected_data_type(kind: &Kind) -> Option<&'static str> {
    Some(match kind {
        Kind::Double => "float64",
        Kind::Float => "float32",
        Kind::Int32 | Kind::Sint32 | Kind::Sfixed32 => "int32",
        Kind::Int64 | Kind::Sint64 | Kind::Sfixed64 => "int64",
        Kind::Uint32 | Kind::Fixed32 => "uint32",
        Kind::Uint64 | Kind::Fixed64 => "uint64",
        Kind::Bool => "bool",
        Kind::String | Kind::Enum(_) => "string",
        Kind::Bytes | Kind::Message(_) => return None,
    })
}

fn null_column(data_type: &str) -> ArrayRef {
    match data_type {
        "bool" => Arc::new(BooleanArray::from(vec![None::<bool>])),
        "float32" => Arc::new(Float32Array::from(vec![None::<f32>])),
        "float64" => Arc::new(Float64Array::from(vec![None::<f64>])),
        "int32" => Arc::new(Int32Array::from(vec![None::<i32>])),
        "int64" => Arc::new(Int64Array::from(vec![None::<i64>])),
        "uint32" => Arc::new(UInt32Array::from(vec![None::<u32>])),
        "uint64" => Arc::new(UInt64Array::from(vec![None::<u64>])),
        _ => Arc::new(StringArray::from(vec![None::<&str>])),
    }
}

fn mismatched_value(identity: &FormatIdentity, name: &str) -> calc_flow::CalcFlowError {
    codec_error(
        identity,
        "decode",
        &format!("the wire value for field {name:?} disagrees with its descriptor"),
    )
}

// Every arm mirrors `expected_data_type`; the two functions must agree.
// #lizard forgives
fn scalar_column(
    identity: &FormatIdentity,
    field: &FieldDescriptor,
    value: &prost_reflect::Value,
) -> Result<ArrayRef> {
    let name = field.name();
    Ok(match field.kind() {
        Kind::Double => Arc::new(Float64Array::from(vec![
            value
                .as_f64()
                .ok_or_else(|| mismatched_value(identity, name))?,
        ])),
        Kind::Float => Arc::new(Float32Array::from(vec![
            value
                .as_f32()
                .ok_or_else(|| mismatched_value(identity, name))?,
        ])),
        Kind::Int32 | Kind::Sint32 | Kind::Sfixed32 => Arc::new(Int32Array::from(vec![
            value
                .as_i32()
                .ok_or_else(|| mismatched_value(identity, name))?,
        ])),
        Kind::Int64 | Kind::Sint64 | Kind::Sfixed64 => Arc::new(Int64Array::from(vec![
            value
                .as_i64()
                .ok_or_else(|| mismatched_value(identity, name))?,
        ])),
        Kind::Uint32 | Kind::Fixed32 => Arc::new(UInt32Array::from(vec![
            value
                .as_u32()
                .ok_or_else(|| mismatched_value(identity, name))?,
        ])),
        Kind::Uint64 | Kind::Fixed64 => Arc::new(UInt64Array::from(vec![
            value
                .as_u64()
                .ok_or_else(|| mismatched_value(identity, name))?,
        ])),
        Kind::Bool => Arc::new(BooleanArray::from(vec![
            value
                .as_bool()
                .ok_or_else(|| mismatched_value(identity, name))?,
        ])),
        Kind::String => Arc::new(StringArray::from(vec![
            value
                .as_str()
                .ok_or_else(|| mismatched_value(identity, name))?,
        ])),
        Kind::Enum(descriptor) => {
            let number = value
                .as_enum_number()
                .ok_or_else(|| mismatched_value(identity, name))?;
            let entry = descriptor.get_value(number).ok_or_else(|| {
                codec_error(
                    identity,
                    "decode",
                    &format!("field {name:?} carries the unknown enum value {number}"),
                )
            })?;
            Arc::new(StringArray::from(vec![entry.name()]))
        }
        Kind::Bytes | Kind::Message(_) => {
            return Err(codec_error(
                identity,
                "decode",
                &format!(
                    "field {name:?} of kind {:?} has no flat Arrow representation",
                    field.kind()
                ),
            ));
        }
    })
}

/// Shared fixtures for the protobuf codec and Kafka source tests.
#[cfg(test)]
pub(crate) mod fixtures {
    use std::path::{Path, PathBuf};

    use prost::Message as _;
    use prost_reflect::{DescriptorPool, DynamicMessage, Value};
    use prost_types::field_descriptor_proto::{Label, Type};
    use prost_types::{
        DescriptorProto, EnumDescriptorProto, EnumValueDescriptorProto, FieldDescriptorProto,
        FileDescriptorProto, FileDescriptorSet, OneofDescriptorProto,
    };

    /// The fully-qualified fixture message name.
    pub(crate) const ORDER_MESSAGE: &str = "events.Order";

    fn field(name: &str, number: i32, label: Label, r#type: Type) -> FieldDescriptorProto {
        FieldDescriptorProto {
            name: Some(name.to_string()),
            number: Some(number),
            label: Some(label as i32),
            r#type: Some(r#type as i32),
            json_name: Some(name.to_string()),
            ..Default::default()
        }
    }

    fn typed_field(name: &str, number: i32, r#type: Type, type_name: &str) -> FieldDescriptorProto {
        FieldDescriptorProto {
            type_name: Some(type_name.to_string()),
            ..field(name, number, Label::Optional, r#type)
        }
    }

    fn order_file() -> FileDescriptorProto {
        let mut note = field("note", 6, Label::Optional, Type::String);
        note.oneof_index = Some(0);
        let order = DescriptorProto {
            name: Some("Order".to_string()),
            field: vec![
                field("id", 1, Label::Optional, Type::Int64),
                field("label", 2, Label::Optional, Type::String),
                field("price", 3, Label::Optional, Type::Double),
                field("active", 4, Label::Optional, Type::Bool),
                typed_field("status", 5, Type::Enum, ".events.Status"),
                note,
                field("tags", 7, Label::Repeated, Type::Int32),
                field("blob", 8, Label::Optional, Type::Bytes),
                typed_field("nested", 9, Type::Message, ".events.Nested"),
                field("big", 10, Label::Optional, Type::Uint64),
                field("ratio", 11, Label::Optional, Type::Float),
                field("delta", 12, Label::Optional, Type::Sint32),
            ],
            oneof_decl: vec![OneofDescriptorProto {
                name: Some("note_choice".to_string()),
                options: None,
            }],
            ..Default::default()
        };
        let nested = DescriptorProto {
            name: Some("Nested".to_string()),
            field: vec![field("value", 1, Label::Optional, Type::Int32)],
            ..Default::default()
        };
        let status = EnumDescriptorProto {
            name: Some("Status".to_string()),
            value: vec![
                EnumValueDescriptorProto {
                    name: Some("STATUS_UNKNOWN".to_string()),
                    number: Some(0),
                    options: None,
                },
                EnumValueDescriptorProto {
                    name: Some("STATUS_PLACED".to_string()),
                    number: Some(1),
                    options: None,
                },
            ],
            ..Default::default()
        };
        FileDescriptorProto {
            name: Some("events.proto".to_string()),
            package: Some("events".to_string()),
            message_type: vec![order, nested],
            enum_type: vec![status],
            syntax: Some("proto3".to_string()),
            ..Default::default()
        }
    }

    /// Serializes the fixture `FileDescriptorSet`.
    pub(crate) fn order_descriptor_set() -> Vec<u8> {
        FileDescriptorSet {
            file: vec![order_file()],
        }
        .encode_to_vec()
    }

    /// Writes the fixture descriptor set into `directory` and returns its path.
    pub(crate) fn write_descriptor_set(directory: &Path) -> PathBuf {
        let path = directory.join("events.pb");
        std::fs::write(&path, order_descriptor_set()).expect("the fixture descriptor writes");
        path
    }

    fn order_message() -> prost_reflect::MessageDescriptor {
        let set = FileDescriptorSet::decode(order_descriptor_set().as_slice())
            .expect("the fixture descriptor set decodes");
        DescriptorPool::from_file_descriptor_set(set)
            .expect("the fixture pool is valid")
            .get_message_by_name(ORDER_MESSAGE)
            .expect("the fixture message resolves")
    }

    /// Encodes one fixture message with the given field values.
    pub(crate) fn order_payload(entries: &[(&str, Value)]) -> Vec<u8> {
        let mut message = DynamicMessage::new(order_message());
        for (name, value) in entries {
            message.set_field_by_name(name, value.clone());
        }
        message.encode_to_vec()
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::{Array, Int64Array, StringArray};

    use super::*;

    fn spec(name: &str, data_type: &str, nullable: bool) -> ArrowFieldSpec {
        ArrowFieldSpec {
            name: name.to_string(),
            data_type: data_type.to_string(),
            nullable,
        }
    }

    fn full_schema() -> Vec<ArrowFieldSpec> {
        vec![
            spec("id", "int64", false),
            spec("label", "string", false),
            spec("price", "float64", false),
            spec("active", "bool", false),
            spec("big", "uint64", false),
            spec("ratio", "float32", false),
            spec("delta", "int32", false),
            spec("status", "string", false),
            spec("note", "string", true),
        ]
    }

    fn codec() -> ProtobufCodec {
        ProtobufCodec::from_descriptor_bytes(
            identity(IDENTITY_VERSION).expect("identity"),
            &fixtures::order_descriptor_set(),
            fixtures::ORDER_MESSAGE,
        )
        .expect("the fixture codec builds")
    }

    fn bounds() -> DecodeBounds {
        DecodeBounds::new(16, 1 << 20).expect("bounds")
    }

    fn decode_record(payload: &[u8], schema: &[ArrowFieldSpec]) -> RecordBatch {
        let batch = codec().decode(payload, &bounds(), schema).expect("decodes");
        let payload = batch.table_payload().expect("a table batch");
        payload.batches()[0].clone()
    }

    fn string_column(record: &RecordBatch, index: usize) -> StringArray {
        record
            .column(index)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("a string column")
            .clone()
    }

    #[test]
    fn a_full_message_decodes_every_scalar_shape() {
        let payload = fixtures::order_payload(&[
            ("id", prost_reflect::Value::I64(7)),
            ("label", prost_reflect::Value::String("one".to_string())),
            ("price", prost_reflect::Value::F64(2.5)),
            ("active", prost_reflect::Value::Bool(true)),
            ("big", prost_reflect::Value::U64(1 << 40)),
            ("ratio", prost_reflect::Value::F32(0.5)),
            ("delta", prost_reflect::Value::I32(-3)),
            ("status", prost_reflect::Value::EnumNumber(1)),
            ("note", prost_reflect::Value::String("hi".to_string())),
        ]);
        let record = decode_record(&payload, &full_schema());
        assert_eq!(record.num_rows(), 1);
        let id = record
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("an int64 column");
        assert_eq!(id.value(0), 7);
        assert_eq!(string_column(&record, 1).value(0), "one");
        assert_eq!(string_column(&record, 7).value(0), "STATUS_PLACED");
        assert_eq!(string_column(&record, 8).value(0), "hi");
    }

    #[test]
    fn unset_fields_decode_defaults_and_nullable_absence() {
        let payload = fixtures::order_payload(&[]);
        let record = decode_record(&payload, &full_schema());
        let id = record
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("an int64 column");
        assert_eq!(id.value(0), 0, "proto3 implicit presence defaults");
        assert_eq!(string_column(&record, 1).value(0), "");
        assert_eq!(
            string_column(&record, 7).value(0),
            "STATUS_UNKNOWN",
            "enum fields default to their zero value name"
        );
        assert!(
            string_column(&record, 8).is_null(0),
            "an unset oneof member decodes null when nullable"
        );
    }

    #[test]
    fn a_non_nullable_absent_field_fails_closed() {
        let mut schema = full_schema();
        schema[8] = spec("note", "string", false);
        let payload = fixtures::order_payload(&[]);
        let error = codec()
            .decode(&payload, &bounds(), &schema)
            .expect_err("an omitted non-nullable field fails");
        assert!(error.to_string().contains("note"), "{error}");
    }

    #[test]
    fn schema_changes_replan_without_stale_fields() {
        let payload = fixtures::order_payload(&[("id", prost_reflect::Value::I64(7))]);
        let codec = codec();
        let first = vec![spec("id", "int64", false)];
        assert_eq!(
            codec
                .decode(&payload, &bounds(), &first)
                .expect("the first schema decodes")
                .num_rows(),
            1
        );
        let mismatched = vec![spec("id", "string", false)];
        let error = codec
            .decode(&payload, &bounds(), &mismatched)
            .expect_err("a changed schema replans and still fails closed");
        assert!(error.to_string().contains("int64"), "{error}");
        assert!(
            codec.decode(&payload, &bounds(), &first).is_ok(),
            "returning to the cached schema decodes again"
        );
    }

    #[test]
    fn unknown_mismatched_and_unsupported_schema_fields_fail() {
        let payload = fixtures::order_payload(&[]);
        let cases: [(Vec<ArrowFieldSpec>, &str); 5] = [
            (vec![spec("missing", "int64", false)], "missing"),
            (vec![spec("id", "string", false)], "int64"),
            (vec![spec("tags", "int32", false)], "repeated"),
            (vec![spec("blob", "string", false)], "flat Arrow"),
            (vec![spec("nested", "string", false)], "flat Arrow"),
        ];
        for (schema, needle) in cases {
            let error = codec()
                .decode(&payload, &bounds(), &schema)
                .expect_err("the schema field fails closed");
            assert!(
                error.to_string().contains(needle),
                "{error} lacks {needle:?}"
            );
        }
    }

    #[test]
    fn malformed_payloads_and_bounds_fail_closed() {
        let error = codec()
            .decode(b"\x00", &bounds(), &full_schema())
            .expect_err("an invalid tag fails");
        assert!(error.to_string().contains("protobuf"), "{error}");

        let error = codec()
            .decode(&fixtures::order_payload(&[]), &bounds(), &[])
            .expect_err("an empty schema fails");
        assert!(error.to_string().contains("schema"), "{error}");

        let tight = DecodeBounds::new(16, 8).expect("tight bounds");
        let error = codec()
            .decode(&fixtures::order_payload(&[]), &tight, &full_schema())
            .expect_err("the byte bound trips");
        assert!(error.to_string().contains("byte limit"), "{error}");
    }

    #[test]
    fn unknown_enum_values_fail_closed() {
        let payload = fixtures::order_payload(&[("status", prost_reflect::Value::EnumNumber(42))]);
        let error = codec()
            .decode(&payload, &bounds(), &full_schema())
            .expect_err("an unknown enum value fails");
        assert!(error.to_string().contains("42"), "{error}");
    }

    #[test]
    fn descriptor_loading_fails_closed() {
        let missing = ProtobufCodec::new(
            IDENTITY_VERSION,
            Path::new("/definitely/missing/events.pb"),
            fixtures::ORDER_MESSAGE,
        )
        .expect_err("a missing descriptor file fails");
        assert!(missing.to_string().contains("cannot be read"), "{missing}");

        let garbage = ProtobufCodec::from_descriptor_bytes(
            identity(IDENTITY_VERSION).expect("identity"),
            b"\xff\xff",
            fixtures::ORDER_MESSAGE,
        )
        .expect_err("garbage descriptor bytes fail");
        assert!(garbage.to_string().contains("descriptor set"), "{garbage}");

        let unknown = ProtobufCodec::from_descriptor_bytes(
            identity(IDENTITY_VERSION).expect("identity"),
            &fixtures::order_descriptor_set(),
            "events.Missing",
        )
        .expect_err("an unknown message name fails");
        assert!(unknown.to_string().contains("events.Missing"), "{unknown}");
    }
}

#[cfg(test)]
mod perf {
    use std::hint::black_box;
    use std::time::Instant;

    use super::*;

    const ITERATIONS: u32 = 100_000;

    fn bench_schema() -> Vec<ArrowFieldSpec> {
        vec![
            ArrowFieldSpec {
                name: "id".into(),
                data_type: "int64".into(),
                nullable: false,
            },
            ArrowFieldSpec {
                name: "label".into(),
                data_type: "string".into(),
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
    #[ignore = "informational decode throughput comparison; run explicitly with --ignored"]
    fn protobuf_and_json_decode_throughput() {
        let bounds = DecodeBounds::new(1024, 1 << 20).expect("bounds");
        let schema = bench_schema();

        let protobuf = ProtobufCodec::from_descriptor_bytes(
            identity(IDENTITY_VERSION).expect("identity"),
            &fixtures::order_descriptor_set(),
            fixtures::ORDER_MESSAGE,
        )
        .expect("codec");
        let protobuf_payload = fixtures::order_payload(&[
            ("id", prost_reflect::Value::I64(7)),
            (
                "label",
                prost_reflect::Value::String("benchmark-order".to_string()),
            ),
            ("price", prost_reflect::Value::F64(2.5)),
        ]);

        let json = crate::json_lines::JsonLinesCodec::new(crate::json_lines::IDENTITY_VERSION)
            .expect("json codec");
        let json_payload = b"{\"id\":7,\"label\":\"benchmark-order\",\"price\":2.5}\n";

        let start = Instant::now();
        for _ in 0..ITERATIONS {
            black_box(
                protobuf
                    .decode(black_box(protobuf_payload.as_slice()), &bounds, &schema)
                    .expect("decodes"),
            );
        }
        let protobuf_elapsed = start.elapsed();

        let start = Instant::now();
        for _ in 0..ITERATIONS {
            black_box(
                json.decode(black_box(json_payload.as_ref()), &bounds, &schema)
                    .expect("decodes"),
            );
        }
        let json_elapsed = start.elapsed();

        let protobuf_ns = protobuf_elapsed.as_secs_f64() * 1e9 / f64::from(ITERATIONS);
        let json_ns = json_elapsed.as_secs_f64() * 1e9 / f64::from(ITERATIONS);
        println!(
            "protobuf decode: {protobuf_ns:.0} ns/op ({:.0} ops/s)",
            1e9 / protobuf_ns
        );
        println!(
            "json decode:     {json_ns:.0} ns/op ({:.0} ops/s)",
            1e9 / json_ns
        );
        println!("protobuf/json ratio: {:.2}", protobuf_ns / json_ns);
    }
}
