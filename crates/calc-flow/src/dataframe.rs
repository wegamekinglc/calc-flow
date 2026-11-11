//! Immutable typed row-local table declarations.
//!
//! ```
//! use std::{collections::BTreeMap, sync::Arc};
//! use calc_flow::arrow::datatypes::{DataType, Field, Schema};
//! use calc_flow::{CompileMode, InputOptions, Program, Result, lit, table_input};
//!
//! fn declaration() -> Result<Program> {
//!     let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
//!     let source = table_input("events", schema, InputOptions::default())?;
//!     let scaled = source.select(vec![(source.column("x")? + lit(1_i64)?).alias("y")?])?;
//!     let result = scaled.sql("SELECT y FROM input")?;
//!     Program::new("mixed", CompileMode::Batch, BTreeMap::from([("result".into(), result)]))
//! }
//! # declaration().unwrap();
//! ```

use crate::{CalcFlowError, CompileMode, Result, UdfRegistrySnapshot};
use datafusion::arrow::datatypes::{DataType, SchemaRef};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

/// Temporal identity declared on an input; empty options make no ordering claim.
#[derive(Clone, Debug, Default)]
pub struct InputOptions {
    pub entity_by: Vec<String>,
    pub event_time: Option<String>,
    pub sequence_by: Vec<String>,
}

/// A scalar declaration, independent of input data.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum LiteralValue {
    Null,
    Boolean(bool),
    Int64(i64),
    UInt64(u64),
    Float64(f64),
    Utf8(String),
}
macro_rules! literal_from {
    ($ty:ty, $variant:ident) => {
        impl From<$ty> for LiteralValue {
            fn from(value: $ty) -> Self {
                Self::$variant(value)
            }
        }
    };
}
literal_from!(bool, Boolean);
literal_from!(i64, Int64);
literal_from!(u64, UInt64);
literal_from!(f64, Float64);
literal_from!(String, Utf8);
impl From<&str> for LiteralValue {
    fn from(value: &str) -> Self {
        Self::Utf8(value.into())
    }
}

/// A shared immutable relation. It owns no rows, plan or runtime state.
#[derive(Clone, Debug)]
pub struct TableExpr(pub(crate) Arc<Relation>);
#[derive(Debug)]
pub(crate) struct Relation {
    pub id: String,
    pub kind: RelationKind,
}
#[derive(Debug)]
pub(crate) enum RelationKind {
    Input {
        name: String,
        schema: SchemaRef,
        options: InputOptions,
    },
    Filter {
        input: TableExpr,
        predicate: ColumnExpr,
    },
    Sql {
        query: String,
        tables: BTreeMap<String, TableExpr>,
    },
    Project {
        input: TableExpr,
        columns: Vec<NamedExpr>,
        append: bool,
    },
}

/// A scalar expression with explicit relation lineage.
#[derive(Clone, Debug)]
#[must_use = "column expressions take effect when attached to a relation"]
pub struct ColumnExpr(pub(crate) Arc<Scalar>);
#[derive(Debug)]
pub(crate) enum Scalar {
    Column {
        lineage: String,
        name: String,
    },
    Literal(LiteralValue),
    Unary {
        op: &'static str,
        value: ColumnExpr,
    },
    Cast {
        value: ColumnExpr,
        data_type: DataType,
    },
    Binary {
        op: &'static str,
        left: ColumnExpr,
        right: ColumnExpr,
    },
}
/// A column and its result name.
#[derive(Clone, Debug)]
pub struct NamedExpr {
    pub(crate) name: String,
    pub(crate) value: ColumnExpr,
}

pub(crate) fn invalid(path: &str, code: &str, message: impl std::fmt::Display) -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: path.into(),
        message: format!("{code}: {message}"),
    }
}
pub(crate) fn check_name(name: &str, path: &str) -> Result<()> {
    if name.is_empty() {
        return Err(invalid(path, "invalid_literal", "name must not be empty"));
    }
    Ok(())
}
pub(crate) fn digest(value: &serde_json::Value) -> String {
    hex::encode(Sha256::digest(value.to_string().as_bytes()))
}
pub(crate) fn segment(root: &str, name: &str) -> String {
    let mut chars = name.chars();
    if chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        format!("{root}.{name}")
    } else {
        format!("{root}[{}]", json!(name))
    }
}
pub(crate) fn quote(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

/// Declares an exact Arrow input schema without capturing input rows.
///
/// # Errors
/// Rejects empty names, duplicate fields, unsupported types and invalid temporal keys.
pub fn table_input(name: &str, schema: SchemaRef, options: InputOptions) -> Result<TableExpr> {
    let path = segment("inputs", name);
    check_name(name, &path)?;
    validate_input_fields(&schema, &path)?;
    validate_input_options(&schema, &options, &path)?;
    let id = digest(&json!([
        "input",
        name,
        crate::query_analysis::schema_fields(&schema),
        options.entity_by,
        options.event_time,
        options.sequence_by
    ]));
    Ok(TableExpr(Arc::new(Relation {
        id,
        kind: RelationKind::Input {
            name: name.into(),
            schema,
            options,
        },
    })))
}

fn validate_input_fields(schema: &SchemaRef, path: &str) -> Result<()> {
    let mut names = BTreeSet::new();
    for (index, field) in schema.fields().iter().enumerate() {
        check_name(field.name(), &format!("{path}.schema[{index}]"))?;
        if !names.insert(field.name()) {
            return Err(invalid(
                &format!("{path}.schema[{index}]"),
                "duplicate_name",
                format!("field '{}' must be unique", field.name()),
            ));
        }
        portable_type(
            field.data_type(),
            &format!("{path}.schema[{index}].data_type"),
        )?;
    }
    Ok(())
}
fn validate_input_options(schema: &SchemaRef, options: &InputOptions, path: &str) -> Result<()> {
    validate_input_keys(schema, options, path)?;
    if let Some(key) = &options.event_time {
        let field = schema.field_with_name(key).map_err(|_| {
            invalid(
                &format!("{path}.event_time"),
                "schema_mismatch",
                "event time must be an existing field",
            )
        })?;
        if field.data_type()
            != &DataType::Timestamp(
                datafusion::arrow::datatypes::TimeUnit::Microsecond,
                Some("UTC".into()),
            )
            || field.is_nullable()
        {
            return Err(invalid(
                &format!("{path}.event_time"),
                "unsupported_type",
                "event time requires non-null timestamp[us, UTC]",
            ));
        }
    }
    Ok(())
}

fn validate_input_keys(schema: &SchemaRef, options: &InputOptions, path: &str) -> Result<()> {
    for (role, keys) in [
        ("entity_by", &options.entity_by),
        ("sequence_by", &options.sequence_by),
    ] {
        let mut seen = BTreeSet::new();
        for key in keys {
            if !seen.insert(key) || schema.field_with_name(key).is_err() {
                return Err(invalid(
                    &format!("{path}.{role}"),
                    "schema_mismatch",
                    "keys must be unique existing fields",
                ));
            }
        }
    }
    Ok(())
}

/// Declares a typed scalar literal.
///
/// # Errors
/// Rejects non-finite floating values.
pub fn lit(value: impl Into<LiteralValue>) -> Result<ColumnExpr> {
    let value = value.into();
    if matches!(&value, LiteralValue::Float64(v) if !v.is_finite()) {
        return Err(invalid(
            "literal",
            "invalid_literal",
            "floating literal must be finite",
        ));
    }
    Ok(ColumnExpr(Arc::new(Scalar::Literal(value))))
}

impl TableExpr {
    pub(crate) fn lineage(&self) -> &str {
        match &self.0.kind {
            RelationKind::Project { input, .. } | RelationKind::Filter { input, .. } => {
                input.lineage()
            }
            RelationKind::Input { .. } | RelationKind::Sql { .. } => &self.0.id,
        }
    }
    fn known_names(&self) -> Option<BTreeSet<String>> {
        match &self.0.kind {
            RelationKind::Input { schema, .. } => Some(
                schema
                    .fields()
                    .iter()
                    .map(|field| field.name().clone())
                    .collect(),
            ),
            RelationKind::Project {
                input,
                columns,
                append,
            } => {
                let mut names = if *append {
                    input.known_names()?
                } else {
                    BTreeSet::new()
                };
                names.extend(columns.iter().map(|column| column.name.clone()));
                Some(names)
            }
            RelationKind::Filter { input, .. } => input.known_names(),
            RelationKind::Sql { .. } => None,
        }
    }
    /// References a field from this row lineage.
    ///
    /// # Errors
    /// Rejects empty or known missing fields; SQL-derived fields are checked at planning.
    pub fn column(&self, name: &str) -> Result<ColumnExpr> {
        check_name(name, "column")?;
        if self
            .known_names()
            .is_some_and(|names| !names.contains(name))
        {
            let path = if let RelationKind::Input { name, .. } = &self.0.kind {
                format!("{}.column", segment("inputs", name))
            } else {
                "column".into()
            };
            return Err(invalid(
                &path,
                "schema_mismatch",
                format!("column '{name}' is not present in relation schema"),
            ));
        }
        Ok(ColumnExpr(Arc::new(Scalar::Column {
            lineage: self.lineage().into(),
            name: name.into(),
        })))
    }
    /// Appends uniquely named columns, retaining the input column order.
    ///
    /// # Errors
    /// Duplicate names and schema-dependent constraints are rejected at planning.
    pub fn with_columns(&self, columns: Vec<NamedExpr>) -> Result<Self> {
        Ok(self.project(columns, true))
    }
    /// Selects columns in declaration order, each with an explicit result name.
    ///
    /// # Errors
    /// Rejects an empty projection; schema and lineage constraints are checked at planning.
    pub fn select(&self, columns: Vec<NamedExpr>) -> Result<Self> {
        if columns.is_empty() {
            return Err(invalid(
                "select",
                "invalid_literal",
                "select at least one column",
            ));
        }
        Ok(self.project(columns, false))
    }
    /// Adds a read-only SQL relation using the fixed alias `input`.
    ///
    /// # Errors
    /// Rejects malformed or non-read-only SQL.
    pub fn sql(&self, query: &str) -> Result<Self> {
        crate::sql::sql(query, BTreeMap::from([("input".into(), self.clone())]))
    }
    /// Keeps rows for which the predicate is true; NULL predicates are discarded.
    ///
    /// # Errors
    /// Boolean type and row lineage are checked at planning.
    pub fn filter(&self, predicate: ColumnExpr) -> Result<Self> {
        let id = digest(&json!(["filter", self.0.id, predicate.identity()]));
        Ok(Self(Arc::new(Relation {
            id,
            kind: RelationKind::Filter {
                input: self.clone(),
                predicate,
            },
        })))
    }
    pub(crate) fn project(&self, columns: Vec<NamedExpr>, append: bool) -> Self {
        let id = digest(&json!([
            "project",
            self.0.id,
            append,
            columns
                .iter()
                .map(|col| (&col.name, col.value.identity()))
                .collect::<Vec<_>>()
        ]));
        Self(Arc::new(Relation {
            id,
            kind: RelationKind::Project {
                input: self.clone(),
                columns,
                append,
            },
        }))
    }
    /// Plans an exact schema without executing rows.
    ///
    /// # Errors
    /// Returns field, lineage and native planning errors with a logical path.
    pub async fn collect_schema(
        &self,
        udfs: &UdfRegistrySnapshot,
        mode: CompileMode,
    ) -> Result<SchemaRef> {
        let program = crate::Program::new(
            "table-schema",
            mode,
            BTreeMap::from([("result".into(), self.clone())]),
        )?;
        program
            .collect_schema(udfs)
            .await?
            .remove("result")
            .ok_or_else(|| {
                invalid(
                    "outputs.result.schema",
                    "unresolved_type",
                    "result schema was not resolved",
                )
            })
    }
}

pub(crate) fn projection(
    input: &TableExpr,
    schema: &SchemaRef,
    columns: &[NamedExpr],
    append: bool,
    path: &str,
) -> Result<Vec<String>> {
    let (mut names, mut select) = projection_prefix(schema, append);
    for column in columns {
        if !names.insert(column.name.clone()) {
            return Err(invalid(
                path,
                "duplicate_name",
                format!("field '{}' must be unique", column.name),
            ));
        }
        check_projection_name(input, schema, column, append, path)?;
        column
            .value
            .dtype(input.lineage(), schema, &segment(path, &column.name))?;
        select.push(format!(
            "{} AS {}",
            column.value.render(),
            quote(&column.name)
        ));
    }
    if select.is_empty() {
        select.push("*".into());
    }
    Ok(select)
}

fn projection_prefix(schema: &SchemaRef, append: bool) -> (BTreeSet<String>, Vec<String>) {
    let names: BTreeSet<String> = if append {
        schema.fields().iter().map(|f| f.name().clone()).collect()
    } else {
        BTreeSet::new()
    };
    let select = if append {
        schema.fields().iter().map(|f| quote(f.name())).collect()
    } else {
        vec![]
    };
    (names, select)
}
fn check_projection_name(
    input: &TableExpr,
    schema: &SchemaRef,
    column: &NamedExpr,
    append: bool,
    path: &str,
) -> Result<()> {
    if !append
        && schema.field_with_name(&column.name).is_ok()
        && !matches!(column.value.0.as_ref(), Scalar::Column { name, lineage } if name == &column.name && lineage == input.lineage())
    {
        return Err(invalid(
            path,
            "duplicate_name",
            "derived projection cannot replace an existing field",
        ));
    }
    Ok(())
}

impl ColumnExpr {
    /// Names this scalar expression in a projection.
    ///
    /// # Errors
    /// Rejects empty names.
    pub fn alias(&self, name: &str) -> Result<NamedExpr> {
        check_name(name, "alias")?;
        Ok(NamedExpr {
            name: name.into(),
            value: self.clone(),
        })
    }
    pub(crate) fn identity(&self) -> String {
        format!("{:?}", self.0)
    }
    pub(crate) fn render(&self) -> String {
        match self.0.as_ref() {
            Scalar::Column { name, .. } => quote(name),
            Scalar::Literal(value) => render_literal(value),
            Scalar::Cast { value, data_type } => format!(
                "arrow_cast({}, '{}')",
                value.render(),
                data_type.to_string().replace('\'', "''")
            ),
            Scalar::Unary { op, value } => render_unary(op, value),
            Scalar::Binary { op, left, right } => {
                format!("({} {op} {})", left.render(), right.render())
            }
        }
    }
    pub(crate) fn dtype(&self, lineage: &str, schema: &SchemaRef, path: &str) -> Result<DataType> {
        match self.0.as_ref() {
            Scalar::Column {
                lineage: origin,
                name,
            } => column_type(origin, name, lineage, schema, path),
            Scalar::Literal(value) => Ok(literal_type(value)),
            Scalar::Cast { value, data_type } => {
                value.dtype(lineage, schema, path)?;
                Ok(data_type.clone())
            }
            Scalar::Unary { op, value } => {
                unary_type(op, value.dtype(lineage, schema, path)?, path)
            }
            Scalar::Binary { op, left, right } => {
                binary_dtype(op, left, right, lineage, schema, path)
            }
        }
    }
}
fn render_literal(value: &LiteralValue) -> String {
    match value {
        LiteralValue::Null => "NULL".into(),
        LiteralValue::Boolean(v) => v.to_string(),
        LiteralValue::Int64(v) => format!("CAST('{v}' AS BIGINT)"),
        LiteralValue::UInt64(v) => format!("CAST('{v}' AS BIGINT UNSIGNED)"),
        LiteralValue::Float64(v) => format!("CAST('{v}' AS DOUBLE)"),
        LiteralValue::Utf8(v) => format!("'{}'", v.replace('\'', "''")),
    }
}
fn render_unary(op: &str, value: &ColumnExpr) -> String {
    match op {
        "IS NULL" | "IS NOT NULL" => format!("({} {op})", value.render()),
        _ => format!("({op} {})", value.render()),
    }
}
fn column_type(
    origin: &str,
    name: &str,
    lineage: &str,
    schema: &SchemaRef,
    path: &str,
) -> Result<DataType> {
    if origin != lineage {
        return Err(invalid(
            &format!("{path}.lineage"),
            "schema_mismatch",
            "columns must come from the same row lineage",
        ));
    }
    schema
        .field_with_name(name)
        .map(|f| f.data_type().clone())
        .map_err(|_| {
            invalid(
                path,
                "schema_mismatch",
                format!("column '{name}' is not present in relation schema"),
            )
        })
}
fn literal_type(value: &LiteralValue) -> DataType {
    match value {
        LiteralValue::Null => DataType::Null,
        LiteralValue::Boolean(_) => DataType::Boolean,
        LiteralValue::Int64(_) => DataType::Int64,
        LiteralValue::UInt64(_) => DataType::UInt64,
        LiteralValue::Float64(_) => DataType::Float64,
        LiteralValue::Utf8(_) => DataType::Utf8,
    }
}
fn unary_type(op: &str, dtype: DataType, path: &str) -> Result<DataType> {
    match op {
        "IS NULL" | "IS NOT NULL" => Ok(DataType::Boolean),
        "NOT" if dtype == DataType::Boolean => Ok(dtype),
        "-" if matches!(
            dtype,
            DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::Float32
                | DataType::Float64
        ) =>
        {
            Ok(dtype)
        }
        "-" => Err(invalid(
            path,
            "unsupported_type",
            "negation requires a signed integer or floating operand; cast operands explicitly",
        )),
        _ => Err(invalid(
            path,
            "unsupported_type",
            "unary operator requires boolean or numeric operands",
        )),
    }
}
fn binary_dtype(
    op: &str,
    left: &ColumnExpr,
    right: &ColumnExpr,
    lineage: &str,
    schema: &SchemaRef,
    path: &str,
) -> Result<DataType> {
    let left = left.dtype(lineage, schema, &format!("{path}.left"))?;
    let right = right.dtype(lineage, schema, &format!("{path}.right"))?;
    binary_type(op, left, &right, path)
}
fn binary_type(op: &str, left: DataType, right: &DataType, path: &str) -> Result<DataType> {
    if &left != right {
        return Err(invalid(
            path,
            "unsupported_type",
            "operands require matching types; cast operands explicitly",
        ));
    }
    match op {
        "AND" | "OR" if left == DataType::Boolean => Ok(DataType::Boolean),
        "=" | "!=" | "<" | "<=" | ">" | ">=" => Ok(DataType::Boolean),
        _ => arithmetic_type(op, left, path),
    }
}
fn arithmetic_type(op: &str, left: DataType, path: &str) -> Result<DataType> {
    match op {
        "/" if !matches!(left, DataType::Float32 | DataType::Float64) => Err(invalid(
            path,
            "unsupported_type",
            "division requires floating operands; cast operands explicitly",
        )),
        "+" | "-" | "*" | "/" if left.is_numeric() => Ok(left),
        _ => Err(invalid(
            path,
            "unsupported_type",
            "operator requires matching numeric or boolean types",
        )),
    }
}
macro_rules! binary_op {
    ($trait:ident, $method:ident, $op:literal) => {
        impl std::ops::$trait for ColumnExpr {
            type Output = Self;
            fn $method(self, rhs: Self) -> Self {
                self.binary($op, rhs)
            }
        }
    };
}
binary_op!(Add, add, "+");
binary_op!(Sub, sub, "-");
binary_op!(Mul, mul, "*");
binary_op!(Div, div, "/");
binary_op!(BitAnd, bitand, "AND");
binary_op!(BitOr, bitor, "OR");
impl std::ops::Neg for ColumnExpr {
    type Output = Self;
    fn neg(self) -> Self {
        self.unary("-")
    }
}
impl std::ops::Not for ColumnExpr {
    type Output = Self;
    fn not(self) -> Self {
        self.unary("NOT")
    }
}
impl ColumnExpr {
    fn binary(&self, op: &'static str, right: Self) -> Self {
        Self(Arc::new(Scalar::Binary {
            op,
            left: self.clone(),
            right,
        }))
    }
    fn unary(&self, op: &'static str) -> Self {
        Self(Arc::new(Scalar::Unary {
            op,
            value: self.clone(),
        }))
    }
    /// Casts through the native Arrow table engine.
    ///
    /// # Errors
    /// Rejects target types outside the supported public field domain.
    pub fn cast(&self, data_type: DataType) -> Result<Self> {
        portable_type(&data_type, "cast.data_type")?;
        Ok(Self(Arc::new(Scalar::Cast {
            value: self.clone(),
            data_type,
        })))
    }
    /// Declares equality, returning a nullable boolean column.
    pub fn eq(&self, rhs: Self) -> Self {
        self.binary("=", rhs)
    }
    /// Declares inequality.
    pub fn ne(&self, rhs: Self) -> Self {
        self.binary("!=", rhs)
    }
    /// Declares less-than.
    pub fn lt(&self, rhs: Self) -> Self {
        self.binary("<", rhs)
    }
    /// Declares less-than or equal.
    pub fn le(&self, rhs: Self) -> Self {
        self.binary("<=", rhs)
    }
    /// Declares greater-than.
    pub fn gt(&self, rhs: Self) -> Self {
        self.binary(">", rhs)
    }
    /// Declares greater-than or equal.
    pub fn ge(&self, rhs: Self) -> Self {
        self.binary(">=", rhs)
    }
    /// Returns a non-null boolean indicating NULL values.
    pub fn is_null(&self) -> Self {
        self.unary("IS NULL")
    }
    /// Returns a non-null boolean indicating non-NULL values.
    pub fn is_not_null(&self) -> Self {
        self.unary("IS NOT NULL")
    }
}

pub(crate) fn portable_type(dtype: &DataType, path: &str) -> Result<()> {
    use datafusion::arrow::datatypes::TimeUnit;
    if matches!(
        dtype,
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
            | DataType::Time32(TimeUnit::Second)
            | DataType::Time64(TimeUnit::Microsecond)
            | DataType::Timestamp(TimeUnit::Millisecond | TimeUnit::Microsecond, None)
    ) || matches!(dtype, DataType::Timestamp(TimeUnit::Microsecond, Some(zone)) if zone.as_ref() == "UTC")
    {
        return Ok(());
    }
    Err(invalid(
        path,
        "unsupported_type",
        "type is outside the row-local public domain",
    ))
}
