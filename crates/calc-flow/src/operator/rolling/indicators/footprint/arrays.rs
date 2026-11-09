use std::sync::Arc;

use datafusion::arrow::{
    array::{
        Array, ArrayRef, DictionaryArray, FixedSizeListArray, GenericListArray,
        GenericListViewArray, LargeListArray, LargeListViewArray, ListArray, ListViewArray,
        MapArray, OffsetSizeTrait, RunArray, StructArray, UnionArray,
    },
    datatypes::{
        ArrowDictionaryKeyType, DataType, Int8Type, Int16Type, Int32Type, Int64Type,
        RunEndIndexType, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
    },
};

use super::{Footprint, Owner};
use crate::{CalcFlowError, Result};

type Check<'a> = &'a mut dyn FnMut() -> Result<()>;

impl Footprint {
    pub(in super::super) fn array_checked(
        &mut self,
        value: &ArrayRef,
        node: &str,
        check: Check<'_>,
    ) -> Result<()> {
        Walk {
            footprint: self,
            owned: true,
            check,
        }
        .array(value, node)
    }

    pub(in super::super) fn backing(
        &mut self,
        value: &ArrayRef,
        node: &str,
        check: Check<'_>,
    ) -> Result<()> {
        Walk {
            footprint: self,
            owned: false,
            check,
        }
        .array(value, node)
    }
}

struct Walk<'a, 'c> {
    footprint: &'a mut Footprint,
    owned: bool,
    check: Check<'c>,
}

impl Walk<'_, '_> {
    fn array(&mut self, value: &ArrayRef, node: &str) -> Result<()> {
        (self.check)()?;
        if self.owned && !self.retain_descriptor(value, node)? {
            return Ok(());
        }
        if let Some(nulls) = value.nulls() {
            self.buffer(nulls.buffer(), node)?;
        }
        self.array_storage(value.as_ref(), node)
    }

    fn retain_descriptor(&mut self, value: &ArrayRef, node: &str) -> Result<bool> {
        let key = (1, Arc::as_ptr(value).cast::<()>() as usize);
        if self.footprint.allocations.contains_key(&key) {
            return Ok(false);
        }
        let bytes =
            super::super::scalar_materialization_plan::descriptor_bytes(value.as_ref(), self.check)
                .map_err(|error| match error {
                    CalcFlowError::Format { .. } => array_error(node),
                    other => other,
                })?;
        self.footprint.allocation(
            key,
            bytes,
            Owner::Array {
                _value: Arc::clone(value),
            },
        );
        Ok(true)
    }

    fn buffer(&mut self, value: &datafusion::arrow::buffer::Buffer, node: &str) -> Result<()> {
        self.footprint.buffer(value, node)
    }

    fn array_storage(&mut self, value: &dyn Array, node: &str) -> Result<()> {
        if self.borrowed_storage(value, node)? || self.encoded_storage(value, node)? {
            return Ok(());
        }
        self.leaf_storage(value, node)
    }

    fn borrowed_storage(&mut self, value: &dyn Array, node: &str) -> Result<bool> {
        if self.list_storage(value, node)? {
            return Ok(true);
        }
        self.composite_storage(value, node)
    }

    fn leaf_storage(&mut self, value: &dyn Array, node: &str) -> Result<()> {
        super::leaf::backing(self.footprint, value, node, self.check)
    }

    fn list_storage(&mut self, value: &dyn Array, node: &str) -> Result<bool> {
        let result = match value.data_type() {
            DataType::List(_) => self.list_array(typed::<ListArray>(value, node)?, node),
            DataType::LargeList(_) => self.list_array(typed::<LargeListArray>(value, node)?, node),
            _ => return self.fixed_list_storage(value, node),
        };
        result.map(|()| true)
    }

    fn fixed_list_storage(&mut self, value: &dyn Array, node: &str) -> Result<bool> {
        match value.data_type() {
            DataType::FixedSizeList(_, _) => {
                self.array(typed::<FixedSizeListArray>(value, node)?.values(), node)?;
            }
            _ => return self.view_storage(value, node),
        }
        Ok(true)
    }

    fn list_array<O: OffsetSizeTrait>(
        &mut self,
        value: &GenericListArray<O>,
        node: &str,
    ) -> Result<()> {
        self.buffer(value.offsets().inner().inner(), node)?;
        self.array(value.values(), node)
    }

    fn view_storage(&mut self, value: &dyn Array, node: &str) -> Result<bool> {
        match value.data_type() {
            DataType::ListView(_) => self.list_view(typed::<ListViewArray>(value, node)?, node)?,
            DataType::LargeListView(_) => {
                self.list_view(typed::<LargeListViewArray>(value, node)?, node)?;
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    fn list_view<O: OffsetSizeTrait>(
        &mut self,
        value: &GenericListViewArray<O>,
        node: &str,
    ) -> Result<()> {
        self.buffer(value.offsets().inner(), node)?;
        self.buffer(value.sizes().inner(), node)?;
        self.array(value.values(), node)
    }

    fn composite_storage(&mut self, value: &dyn Array, node: &str) -> Result<bool> {
        let result = match value.data_type() {
            DataType::Struct(_) => self.struct_array(typed::<StructArray>(value, node)?, node),
            DataType::Map(_, _) => self.map_array(typed::<MapArray>(value, node)?, node),
            _ => return self.union_storage(value, node),
        };
        result.map(|()| true)
    }

    fn union_storage(&mut self, value: &dyn Array, node: &str) -> Result<bool> {
        match value.data_type() {
            DataType::Union(_, _) => self.union_array(typed::<UnionArray>(value, node)?, node)?,
            _ => return Ok(false),
        }
        Ok(true)
    }

    fn struct_array(&mut self, value: &StructArray, node: &str) -> Result<()> {
        for child in value.columns() {
            self.array(child, node)?;
        }
        Ok(())
    }

    fn map_array(&mut self, value: &MapArray, node: &str) -> Result<()> {
        self.buffer(value.offsets().inner().inner(), node)?;
        self.array(value.keys(), node)?;
        self.array(value.values(), node)
    }

    fn union_array(&mut self, value: &UnionArray, node: &str) -> Result<()> {
        self.buffer(value.type_ids().inner(), node)?;
        if let Some(offsets) = value.offsets() {
            self.buffer(offsets.inner(), node)?;
        }
        let DataType::Union(fields, _) = value.data_type() else {
            return Err(array_error(node));
        };
        for (id, _) in fields.iter() {
            self.array(value.child(id), node)?;
        }
        Ok(())
    }

    fn encoded_storage(&mut self, value: &dyn Array, node: &str) -> Result<bool> {
        match value.data_type() {
            DataType::Dictionary(key, _) => self.dictionary(value, key, node)?,
            DataType::RunEndEncoded(field, _) => self.run_end(value, field.data_type(), node)?,
            _ => return Ok(false),
        }
        Ok(true)
    }

    fn dictionary(&mut self, value: &dyn Array, key: &DataType, node: &str) -> Result<()> {
        match key {
            DataType::Int8 => {
                self.dictionary_array(typed::<DictionaryArray<Int8Type>>(value, node)?, node)
            }
            DataType::Int16 => {
                self.dictionary_array(typed::<DictionaryArray<Int16Type>>(value, node)?, node)
            }
            _ => self.large_signed_dictionary(value, key, node),
        }
    }

    fn large_signed_dictionary(
        &mut self,
        value: &dyn Array,
        key: &DataType,
        node: &str,
    ) -> Result<()> {
        match key {
            DataType::Int32 => {
                self.dictionary_array(typed::<DictionaryArray<Int32Type>>(value, node)?, node)
            }
            DataType::Int64 => {
                self.dictionary_array(typed::<DictionaryArray<Int64Type>>(value, node)?, node)
            }
            _ => self.unsigned_dictionary(value, key, node),
        }
    }

    fn unsigned_dictionary(&mut self, value: &dyn Array, key: &DataType, node: &str) -> Result<()> {
        match key {
            DataType::UInt8 => {
                self.dictionary_array(typed::<DictionaryArray<UInt8Type>>(value, node)?, node)
            }
            DataType::UInt16 => {
                self.dictionary_array(typed::<DictionaryArray<UInt16Type>>(value, node)?, node)
            }
            _ => self.large_unsigned_dictionary(value, key, node),
        }
    }

    fn large_unsigned_dictionary(
        &mut self,
        value: &dyn Array,
        key: &DataType,
        node: &str,
    ) -> Result<()> {
        match key {
            DataType::UInt32 => {
                self.dictionary_array(typed::<DictionaryArray<UInt32Type>>(value, node)?, node)
            }
            DataType::UInt64 => {
                self.dictionary_array(typed::<DictionaryArray<UInt64Type>>(value, node)?, node)
            }
            _ => Err(array_error(node)),
        }
    }

    fn dictionary_array<K: ArrowDictionaryKeyType>(
        &mut self,
        value: &DictionaryArray<K>,
        node: &str,
    ) -> Result<()> {
        self.buffer(value.keys().values().inner(), node)?;
        self.array(value.values(), node)
    }

    fn run_end(&mut self, value: &dyn Array, key: &DataType, node: &str) -> Result<()> {
        match key {
            DataType::Int16 => self.run_array(typed::<RunArray<Int16Type>>(value, node)?, node),
            DataType::Int32 => self.run_array(typed::<RunArray<Int32Type>>(value, node)?, node),
            DataType::Int64 => self.run_array(typed::<RunArray<Int64Type>>(value, node)?, node),
            _ => Err(array_error(node)),
        }
    }

    fn run_array<R: RunEndIndexType>(&mut self, value: &RunArray<R>, node: &str) -> Result<()> {
        self.buffer(value.run_ends().inner().inner(), node)?;
        self.array(value.values(), node)
    }
}

fn typed<'a, T: Array + 'static>(value: &'a dyn Array, node: &str) -> Result<&'a T> {
    value
        .as_any()
        .downcast_ref()
        .ok_or_else(|| array_error(node))
}

fn array_error(node: &str) -> CalcFlowError {
    CalcFlowError::Operator {
        node_id: node.into(),
        message: "rolling participating array representation is inconsistent".into(),
    }
}
