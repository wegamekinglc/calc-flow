use chrono::{DateTime, Utc};

use crate::{CalcFlowError, CancellationToken, JsonMap, Result};

/// The immutable, job-scoped context shared by every task of one streaming
/// job (plan task M2.1 derives per-task scopes from this value).
///
/// M1.1 introduces the type so `StreamOperatorContext` can expose the frozen
/// `job()` accessor (API note A2); the supervisor model of M2 owns
/// construction in production paths.
#[derive(Clone, Debug)]
pub struct StreamJobContext {
    job_id: u64,
    fingerprint: String,
    settings: JsonMap,
    deadline: Option<DateTime<Utc>>,
    cancellation: CancellationToken,
}

impl StreamJobContext {
    /// Creates a job context. `deadline` is `DateTime<Utc>`, so UTC is
    /// guaranteed by the type and needs no runtime validation.
    pub fn new(
        job_id: u64,
        fingerprint: impl Into<String>,
        settings: JsonMap,
        deadline: Option<DateTime<Utc>>,
        cancellation: CancellationToken,
    ) -> Self {
        Self {
            job_id,
            fingerprint: fingerprint.into(),
            settings,
            deadline,
            cancellation,
        }
    }

    pub const fn job_id(&self) -> u64 {
        self.job_id
    }

    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    pub const fn settings(&self) -> &JsonMap {
        &self.settings
    }

    pub const fn deadline(&self) -> Option<&DateTime<Utc>> {
        self.deadline.as_ref()
    }

    pub const fn cancellation(&self) -> &CancellationToken {
        &self.cancellation
    }

    /// Verifies that the job remains active.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::Cancelled`] when cancellation was requested or
    /// the deadline has passed.
    pub fn check_cancelled(&self) -> Result<()> {
        if self.cancellation.is_cancelled()
            || self.deadline.is_some_and(|deadline| Utc::now() >= deadline)
        {
            return Err(CalcFlowError::Cancelled {
                run_id: self.job_id.to_string(),
            });
        }
        Ok(())
    }
}
