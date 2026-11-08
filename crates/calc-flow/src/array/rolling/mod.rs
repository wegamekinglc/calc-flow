pub(super) mod ewma;

pub(crate) use ewma::next_ewma;
pub use ewma::{EwmaOptions, ewma};
