//! Array rolling-mean state uses the crate-wide compensated numeric contract.

pub(crate) use crate::accumulate::ArithmeticError;
pub(crate) type FloatSummary = crate::accumulate::NumericAccumulator;
