use std::{mem::size_of, sync::Arc};

use datafusion::{arrow::array::Array, common::ScalarValue};

use super::{Footprint, Owner, size};
use crate::Result;

type Check<'a> = &'a mut dyn FnMut() -> Result<()>;

impl Footprint {
    pub(in super::super) fn scalar(&mut self, value: &ScalarValue, node: &str) -> Result<()> {
        self.scalar_checked(value, node, &mut || Ok(()))
    }

    pub(in super::super) fn scalar_checked(
        &mut self,
        value: &ScalarValue,
        node: &str,
        check: Check<'_>,
    ) -> Result<()> {
        check()?;
        self.add_owned(size(size_of::<ScalarValue>(), node)?, node)?;
        self.scalar_owned(value, node)?;
        self.scalar_nested(value, node, check)?;
        self.scalar_boxed(value, node, check)
    }

    fn scalar_owned(&mut self, value: &ScalarValue, node: &str) -> Result<()> {
        let bytes = match value {
            ScalarValue::Utf8(value)
            | ScalarValue::LargeUtf8(value)
            | ScalarValue::Utf8View(value) => value.as_ref().map_or(0, String::capacity),
            ScalarValue::Binary(value)
            | ScalarValue::LargeBinary(value)
            | ScalarValue::BinaryView(value)
            | ScalarValue::FixedSizeBinary(_, value) => value.as_ref().map_or(0, Vec::capacity),
            _ => 0,
        };
        self.add_owned(size(bytes, node)?, node)?;
        self.scalar_timezone(value, node)
    }

    fn scalar_timezone(&mut self, value: &ScalarValue, node: &str) -> Result<()> {
        let (ScalarValue::TimestampSecond(_, timezone)
        | ScalarValue::TimestampMillisecond(_, timezone)
        | ScalarValue::TimestampMicrosecond(_, timezone)
        | ScalarValue::TimestampNanosecond(_, timezone)) = value
        else {
            return Ok(());
        };
        if let Some(timezone) = timezone {
            self.allocation(
                (0, Arc::as_ptr(timezone).cast::<()>() as usize),
                size(timezone.len(), node)?,
                Owner::Timezone {
                    _value: Arc::clone(timezone),
                },
            );
        }
        Ok(())
    }

    fn scalar_nested(&mut self, value: &ScalarValue, node: &str, check: Check<'_>) -> Result<()> {
        match value {
            ScalarValue::List(value) => self.typed_array(value, node, check),
            ScalarValue::LargeList(value) => self.typed_array(value, node, check),
            ScalarValue::FixedSizeList(value) => self.typed_array(value, node, check),
            _ => self.scalar_composite(value, node, check),
        }
    }

    fn scalar_composite(
        &mut self,
        value: &ScalarValue,
        node: &str,
        check: Check<'_>,
    ) -> Result<()> {
        match value {
            ScalarValue::ListView(value) => self.typed_array(value, node, check),
            ScalarValue::LargeListView(value) => self.typed_array(value, node, check),
            ScalarValue::Struct(value) => self.typed_array(value, node, check),
            ScalarValue::Map(value) => self.typed_array(value, node, check),
            _ => Ok(()),
        }
    }

    fn scalar_boxed(&mut self, value: &ScalarValue, node: &str, check: Check<'_>) -> Result<()> {
        match value {
            ScalarValue::Dictionary(_, value) => {
                self.add_owned(
                    size(size_of::<datafusion::arrow::datatypes::DataType>(), node)?,
                    node,
                )?;
                self.scalar_checked(value, node, check)
            }
            ScalarValue::RunEndEncoded(_, _, value)
            | ScalarValue::Union(Some((_, value)), _, _) => self.scalar_checked(value, node, check),
            _ => Ok(()),
        }
    }

    fn typed_array<T: Array + 'static>(
        &mut self,
        value: &Arc<T>,
        node: &str,
        check: Check<'_>,
    ) -> Result<()> {
        let array: datafusion::arrow::array::ArrayRef = value.clone();
        self.array_checked(&array, node, check)
    }
}
