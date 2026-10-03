use std::panic::{AssertUnwindSafe, catch_unwind};

use super::{
    Attempt, ClaimedWork, ErasedOutput, GatherHome, GatherOperatorId, GatherStop, PoolPhase,
    Request, Slot, detach, internal,
};
use crate::{CalcFlowError, Result};

struct Claim {
    id: u64,
    generation: u64,
    operator: GatherOperatorId,
    work: ClaimedWork,
    ordinal: usize,
    stop: GatherStop,
}

pub(super) fn run(
    pool: &super::process::NativePool,
    identity: &super::Mutex<Option<super::WorkerIdentity>>,
) {
    let result = catch_unwind(AssertUnwindSafe(|| worker_loop(pool, identity)));
    if let Err(payload) = result {
        pool.home.fail_worker(
            identity.lock().as_ref().and_then(|value| value.active),
            &super::super::failure::panic_message(payload.as_ref()),
        );
        pool.home.close();
    }
    pool.wake_joiner();
}

fn worker_loop(
    pool: &super::process::NativePool,
    identity: &super::Mutex<Option<super::WorkerIdentity>>,
) {
    while let Some(claim) = pool.home.claim(pool.generation) {
        let Claim {
            id,
            generation,
            operator,
            work,
            ordinal,
            stop,
        } = claim;
        let task_id = operator.task.unwrap_or_else(|| super::TaskId::new(0));
        {
            let mut identity = identity.lock();
            if identity.as_ref().is_none_or(|current| {
                current.task_id != task_id || !super::Arc::ptr_eq(&current.token, &operator.token)
            }) {
                *identity = Some(worker_identity(&operator));
            }
            if let Some(identity) = identity.as_mut() {
                identity.active = Some((generation, id, ordinal));
            }
        }
        let outcome = catch_unwind(AssertUnwindSafe(|| work.run(&stop)))
            .map_err(|payload| CalcFlowError::TaskPanicked {
                task_id: task_id.as_u64(),
                message: super::super::failure::panic_message(payload.as_ref()),
            })
            .and_then(std::convert::identity);
        drop(stop);
        pool.home.complete(generation, id, ordinal, outcome);
        if let Some(identity) = identity.lock().as_mut() {
            identity.active = None;
        }
    }
}

impl GatherHome {
    pub(super) fn retain_join_panic(
        &self,
        identity: Option<&super::WorkerIdentity>,
        payload: &(dyn std::any::Any + Send),
    ) {
        let task_id = identity.map_or_else(|| super::TaskId::new(0), |value| value.task_id);
        let name = identity.map_or("native CPU worker", |value| value.task_name.as_str());
        let message = panic_summary(payload);
        self.state
            .lock()
            .retain_diagnostic(task_id, || super::TaskFailure {
                task_id,
                task_name: bounded_message(&name, super::DIAGNOSTIC_NAME_BYTES),
                error: CalcFlowError::TaskPanicked {
                    task_id: task_id.as_u64(),
                    message,
                },
            });
    }

    fn claim(&self, generation: u64) -> Option<Claim> {
        let mut state = self.state.lock();
        loop {
            if !state.open || state.generation != generation || state.phase == PoolPhase::Retiring {
                return None;
            }
            if !claim_capacity(&state.slot) {
                self.native_changed.wait(&mut state);
                continue;
            }
            if let Some(request) = state.queue.pop_front() {
                if let Some(claim) = claim_request(&mut state.slot, request) {
                    refill_request(&mut state);
                    return Some(claim);
                }
                continue;
            }
            self.native_changed.wait(&mut state);
        }
    }

    fn complete(&self, generation: u64, id: u64, ordinal: usize, outcome: Result<ErasedOutput>) {
        let finished = {
            let mut state = self.state.lock();
            let Slot::Active(record) = &mut state.slot else {
                return;
            };
            if record.id != id || record.generation != generation {
                return;
            }
            let Some(work) = record.work.as_mut() else {
                return;
            };
            let complete = work.complete(ordinal, outcome);
            record.running = work.active() > 0;
            self.native_changed.notify_all();
            if !complete {
                return;
            }
            record.running = true;
            (
                record.work.take(),
                record.operator.clone(),
                record.stop.clone(),
            )
        };
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            finished
                .0
                .ok_or_else(|| internal("missing completed work"))
                .and_then(|work| work.finish(self, &finished.1, &finished.2))
        }))
        .map_err(|payload| self.retain_cleanup_panic(&finished.1, payload.as_ref()))
        .and_then(std::convert::identity);
        let abandoned = {
            let mut state = self.state.lock();
            let Slot::Active(record) = &mut state.slot else {
                return;
            };
            record.outcome = Some(outcome);
            record.running = false;
            record.settled = true;
            let abandoned = if record.abandoned {
                Some(detach(&mut state, id))
            } else {
                None
            };
            self.retire_if_pressured(&mut state);
            abandoned
        };
        if let Some(record) = abandoned {
            self.destroy_attempt(record);
        }
        self.changed.notify_waiters();
    }

    fn fail_worker(&self, active: Option<(u64, u64, usize)>, message: &str) {
        if let Some((generation, id, ordinal)) = active {
            self.complete(generation, id, ordinal, Err(internal(message)));
        }
    }

    pub(super) fn retain_cleanup_panic(
        &self,
        operator: &GatherOperatorId,
        payload: &(dyn std::any::Any + Send),
    ) -> CalcFlowError {
        let task_id = operator.task.unwrap_or_else(|| super::TaskId::new(0));
        let message = panic_summary(payload);
        let mut state = self.state.lock();
        state.retain_diagnostic(task_id, || super::TaskFailure {
            task_id,
            task_name: bounded_message(&operator.name, super::DIAGNOSTIC_NAME_BYTES),
            error: CalcFlowError::TaskPanicked {
                task_id: task_id.as_u64(),
                message: message.clone(),
            },
        });
        CalcFlowError::TaskPanicked {
            task_id: task_id.as_u64(),
            message,
        }
    }

    pub(super) fn retain_failure(&self, record: &Attempt) {
        let Some(Err(error)) = &record.outcome else {
            return;
        };
        self.retain_error(&record.operator, error);
    }

    pub(super) fn retain_error(&self, operator: &GatherOperatorId, error: &CalcFlowError) {
        if matches!(error, CalcFlowError::Cancelled { .. }) {
            return;
        }
        let mut state = self.state.lock();
        let task_id = operator.task.unwrap_or_else(|| super::TaskId::new(0));
        state.retain_diagnostic(task_id, || super::TaskFailure {
            task_id,
            task_name: bounded_message(&operator.name, super::DIAGNOSTIC_NAME_BYTES),
            error: internal(&bounded_message(error, super::DIAGNOSTIC_MESSAGE_BYTES)),
        });
    }
}

impl super::HomeState {
    fn retain_diagnostic(
        &mut self,
        task_id: super::TaskId,
        detail: impl FnOnce() -> super::TaskFailure,
    ) {
        if self.failures.len() < super::DIAGNOSTIC_LIMIT {
            self.failures.push(detail());
        } else {
            self.omitted_task.get_or_insert(task_id);
            self.omitted_failures = self.omitted_failures.and_then(|count| count.checked_add(1));
        }
    }

    pub(super) fn take_failures(&mut self) -> Vec<super::TaskFailure> {
        let omitted = self.omitted_failures.take();
        self.omitted_failures = Some(0);
        if omitted != Some(0) {
            let message = omitted.map_or_else(
                || "native CPU diagnostic omission counter overflow".into(),
                |count| format!("native CPU diagnostics omitted={count}"),
            );
            self.failures.push(super::TaskFailure {
                task_id: self
                    .omitted_task
                    .take()
                    .unwrap_or_else(|| super::TaskId::new(0)),
                task_name: "native CPU diagnostics overflow".into(),
                error: CalcFlowError::Internal { message },
            });
        }
        std::mem::take(&mut self.failures)
    }
}

fn claim_capacity(slot: &Slot) -> bool {
    let Slot::Active(record) = slot else {
        return false;
    };
    !record.abandoned
        && record
            .work
            .as_ref()
            .is_some_and(|work| work.active() < work.workers())
}

fn claim_request(slot: &mut Slot, request: Request) -> Option<Claim> {
    let Slot::Active(record) = slot else {
        return None;
    };
    if record.id != request.attempt || record.generation != request.generation || record.abandoned {
        return None;
    }
    let work = record.work.as_mut()?.claim(request.ordinal)?;
    record.running = true;
    Some(Claim {
        id: record.id,
        generation: record.generation,
        operator: record.operator.clone(),
        work,
        ordinal: request.ordinal,
        stop: record.stop.clone(),
    })
}

fn refill_request(state: &mut super::HomeState) {
    let Slot::Active(record) = &mut state.slot else {
        return;
    };
    let Some(work) = &record.work else {
        return;
    };
    if record.abandoned || record.next_unit >= work.units() {
        return;
    }
    state.queue.push_back(Request {
        generation: record.generation,
        attempt: record.id,
        ordinal: record.next_unit,
    });
    record.next_unit += 1;
}

pub(super) fn worker_identity(operator: &GatherOperatorId) -> super::WorkerIdentity {
    super::WorkerIdentity {
        active: None,
        token: operator.token.clone(),
        task_id: operator.task.unwrap_or_else(|| super::TaskId::new(0)),
        task_name: bounded_message(&operator.name, super::DIAGNOSTIC_NAME_BYTES),
    }
}

pub(super) fn panic_summary(payload: &(dyn std::any::Any + Send)) -> String {
    let message = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied())
        .unwrap_or("non-string native panic");
    bounded_message(&message, super::DIAGNOSTIC_MESSAGE_BYTES)
}

fn bounded_message(value: &impl std::fmt::Display, max_bytes: usize) -> String {
    use std::fmt::Write as _;
    struct Writer {
        text: String,
        max: usize,
    }
    impl std::fmt::Write for Writer {
        fn write_str(&mut self, value: &str) -> std::fmt::Result {
            let mut end = value.len().min(self.max - self.text.len());
            while !value.is_char_boundary(end) {
                end -= 1;
            }
            self.text.push_str(&value[..end]);
            Ok(())
        }
    }
    let mut writer = Writer {
        text: String::with_capacity(max_bytes),
        max: max_bytes,
    };
    let _ = write!(writer, "{value}");
    writer.text
}
