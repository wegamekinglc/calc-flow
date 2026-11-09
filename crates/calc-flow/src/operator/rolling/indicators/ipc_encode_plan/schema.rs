use super::{
    Check,
    math::{add, elements, growing, size, string, sum, table, vec_elements, vector},
};
use crate::Result;
use datafusion::arrow::datatypes::{DataType, Field, Fields, Schema, UnionFields};
use std::collections::HashMap;

#[derive(Default)]
pub(super) struct SchemaPlan {
    pub(super) bytes: u64,
    pub(super) tables: u64,
    pub(super) fields: u64,
    pub(super) dictionaries: u64,
    pub(super) workspace: u64,
    pub(super) clone_bytes: u64,
}
impl SchemaPlan {
    pub(super) fn collect(schema: &Schema, check: Check<'_>) -> Result<Self> {
        let mut plan = Self::default();
        plan.metadata(schema.metadata(), check)?;
        plan.fields_vector(schema.fields().len())?;
        for field in schema.fields() {
            plan.field(field, check)?;
        }
        plan.push_table(4)?;
        plan.clone_bytes = clone_schema(schema, check)?;
        Ok(plan)
    }
    fn push_table(&mut self, slots: u64) -> Result<()> {
        self.tables = add(self.tables, 1)?;
        self.bytes = add(self.bytes, table(slots)?)?;
        Ok(())
    }
    fn fields_vector(&mut self, count: usize) -> Result<()> {
        self.bytes = add(self.bytes, vector(size(count)?, 4)?)?;
        self.workspace = add(self.workspace, growing(size(count)?, 4)?)?;
        Ok(())
    }
    fn metadata(&mut self, metadata: &HashMap<String, String>, check: Check<'_>) -> Result<()> {
        if metadata.is_empty() {
            return Ok(());
        }
        for (key, value) in metadata {
            self.metadata_entry(key, value, check)?;
        }
        let n = size(metadata.len())?;
        self.bytes = sum([Ok(self.bytes), vector(n, 4)])?;
        self.workspace = sum([
            Ok(self.workspace),
            vec_elements::<&String>(n),
            elements::<&String>(n),
            vec_elements::<u32>(n),
        ])?;
        Ok(())
    }
    fn metadata_entry(&mut self, key: &str, value: &str, check: Check<'_>) -> Result<()> {
        check()?;
        self.bytes = sum([Ok(self.bytes), string(key.len()), string(value.len())])?;
        self.push_table(2)
    }
    fn field(&mut self, field: &Field, check: Check<'_>) -> Result<()> {
        check()?;
        self.fields = add(self.fields, 1)?;
        self.bytes = sum([Ok(self.bytes), string(field.name().len())])?;
        self.metadata(field.metadata(), check)?;
        self.data_type(field.data_type(), check)?;
        self.dictionary(field.data_type())?;
        self.push_table(8)
    }
    fn dictionary(&mut self, dt: &DataType) -> Result<()> {
        if matches!(dt, DataType::Dictionary(_, _)) {
            self.dictionaries = add(self.dictionaries, 1)?;
            self.push_table(2)?;
            self.push_table(3)?;
        }
        Ok(())
    }
    fn data_type(&mut self, data_type: &DataType, check: Check<'_>) -> Result<()> {
        check()?;
        validate_type(data_type)?;
        if let DataType::Timestamp(_, Some(zone)) = data_type {
            self.bytes = add(self.bytes, string(zone.len())?)?;
        }
        self.push_table(4)?;
        self.children(data_type, check)
    }
    fn children(&mut self, data_type: &DataType, check: Check<'_>) -> Result<()> {
        match data_type {
            DataType::Struct(fields) => self.struct_fields(fields, check),
            DataType::Union(fields, _) => self.union_fields(fields, check),
            DataType::RunEndEncoded(left, right) => self.run_fields(left, right, check),
            _ => self.other_children(data_type, check),
        }
    }
    fn struct_fields(&mut self, fields: &Fields, check: Check<'_>) -> Result<()> {
        self.fields_vector(fields.len())?;
        for field in fields {
            self.field(field, check)?;
        }
        Ok(())
    }
    fn union_fields(&mut self, fields: &UnionFields, check: Check<'_>) -> Result<()> {
        self.fields_vector(fields.len())?;
        let n = size(fields.len())?;
        self.bytes = sum([Ok(self.bytes), vector(n, 4)])?;
        self.workspace = add(self.workspace, growing(n, 4)?)?;
        self.union_children(fields, check)
    }
    fn union_children(&mut self, fields: &UnionFields, check: Check<'_>) -> Result<()> {
        for (_, field) in fields.iter() {
            self.field(field, check)?;
        }
        Ok(())
    }
    fn run_fields(&mut self, left: &Field, right: &Field, check: Check<'_>) -> Result<()> {
        self.fields_vector(2)?;
        self.field(left, check)?;
        self.field(right, check)
    }
    fn other_children(&mut self, data_type: &DataType, check: Check<'_>) -> Result<()> {
        match data_type {
            DataType::List(field)
            | DataType::LargeList(field)
            | DataType::ListView(field)
            | DataType::LargeListView(field)
            | DataType::FixedSizeList(field, _)
            | DataType::Map(field, _) => {
                self.fields_vector(1)?;
                self.field(field, check)
            }
            DataType::Dictionary(_, value) => self.data_type(value, check),
            _ => self.fields_vector(0),
        }
    }
}

fn validate_type(data_type: &DataType) -> Result<()> {
    if matches!(data_type,DataType::Dictionary(_,value) if matches!(value.as_ref(),DataType::Dictionary(_,_)))
    {
        return Err(super::math::invalid());
    }
    Ok(())
}
fn clone_schema(schema: &Schema, check: Check<'_>) -> Result<u64> {
    let mut bytes = sum([
        size(size_of::<Schema>()),
        elements::<usize>(2),
        super::hash_bytes(schema.metadata().capacity(), size_of::<(String, String)>()),
    ])?;
    for (key, value) in schema.metadata() {
        check()?;
        bytes = sum([Ok(bytes), size(key.len()), size(value.len())])?;
    }
    Ok(bytes)
}
