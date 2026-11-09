use datafusion::arrow::datatypes::SchemaRef as PreparedSchemaRef;

pub(super) struct FreshStateSchema {
    schema: PreparedSchemaRef,
}

impl FreshStateSchema {
    pub(super) const fn as_schema(&self) -> &PreparedSchemaRef {
        &self.schema
    }
}
