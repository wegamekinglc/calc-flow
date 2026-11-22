use super::{StreamAsofJoinOperator, arrow_error};
use crate::{BatchKind, Port, Result};
use std::sync::Arc;

impl StreamAsofJoinOperator {
    pub(super) fn configure_projection(&mut self, columns: Vec<usize>) -> Result<()> {
        let output_schema = self.schemas[2]
            .project(&columns)
            .map_err(|error| arrow_error(&error))?;
        let left_columns = self.schemas[0].fields().len();
        let projection = columns
            .into_iter()
            .map(|index| {
                if index < left_columns {
                    (0, index)
                } else {
                    (1, index - left_columns)
                }
            })
            .collect::<Vec<_>>();
        self.outputs[0] = Port::with_schema_ref(
            "output",
            BatchKind::Table,
            true,
            Some(Arc::new(output_schema)),
        )?;
        self.output_projection = Some(projection);
        Ok(())
    }
}
