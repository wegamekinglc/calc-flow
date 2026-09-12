use std::ops::{Deref, DerefMut};

use super::{
    Arc, DriverReport, JobEntityWorkOwner, Mutex, SupervisionReport, TaskFailure, TaskSupervisor,
    task_runtime_failure,
};

#[derive(Default)]
struct SupervisionState {
    loaned: bool,
    returned: Option<TaskSupervisor>,
    prepared: Option<DriverReport>,
    secondary: Vec<TaskFailure>,
}

/// The whole supervisor returns here even when its driver future is aborted.
#[derive(Clone, Default)]
pub(super) struct SupervisionHome(Arc<Mutex<SupervisionState>>);

impl SupervisionHome {
    pub(super) fn install(
        &self,
        supervisor: TaskSupervisor,
        cpu: JobEntityWorkOwner,
    ) -> SupervisorLoan {
        let mut home = self.0.lock();
        assert!(!home.loaned && home.returned.is_none());
        home.loaned = true;
        SupervisorLoan {
            home: self.clone(),
            supervisor: Some(supervisor),
            cpu,
        }
    }

    pub(super) fn take(&self, cpu: JobEntityWorkOwner) -> Option<SupervisorLoan> {
        let mut home = self.0.lock();
        assert!(!home.loaned, "only one driver can own the supervisor");
        let supervisor = home.returned.take()?;
        home.loaned = true;
        Some(SupervisorLoan {
            home: self.clone(),
            supervisor: Some(supervisor),
            cpu,
        })
    }

    pub(super) fn prepare(&self, mut report: DriverReport) {
        let mut home = self.0.lock();
        assert!(home.prepared.is_none(), "prepare the unique report once");
        report
            .cleanup_failures
            .extend(home.secondary.drain(..).map(task_runtime_failure));
        home.prepared = Some(report);
    }

    pub(super) fn has_report(&self) -> bool {
        self.0.lock().prepared.is_some()
    }

    pub(super) fn take_report(&self) -> Option<DriverReport> {
        let mut home = self.0.lock();
        assert!(!home.loaned && home.returned.is_none());
        home.prepared.take()
    }

    pub(super) fn clear_joined(&self) {
        let supervisor = {
            let mut home = self.0.lock();
            assert!(!home.loaned);
            if let Some(supervisor) = &home.returned {
                assert_eq!(
                    supervisor.task_count(),
                    0,
                    "join all checkpoint participants before release"
                );
            }
            home.returned.take()
        };
        drop(supervisor);
    }

    #[cfg(test)]
    pub(super) fn is_loaned(&self) -> bool {
        self.0.lock().loaned
    }
}

pub(super) struct SupervisorLoan {
    home: SupervisionHome,
    supervisor: Option<TaskSupervisor>,
    cpu: JobEntityWorkOwner,
}

impl SupervisorLoan {
    pub(super) async fn join_all(&mut self) -> SupervisionReport {
        self.deref_mut().settle_tasks().await;
        self.cpu.close_admission();
        let secondary = self.cpu.drain().await;
        self.home.0.lock().secondary.extend(secondary);
        self.deref_mut().take_report()
    }
}

impl Deref for SupervisorLoan {
    type Target = TaskSupervisor;
    fn deref(&self) -> &Self::Target {
        self.supervisor.as_ref().expect("live supervisor loan")
    }
}

impl DerefMut for SupervisorLoan {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.supervisor.as_mut().expect("live supervisor loan")
    }
}

impl Drop for SupervisorLoan {
    fn drop(&mut self) {
        self.cpu.close_admission();
        if let Some(mut supervisor) = self.supervisor.take() {
            supervisor.cancel_and_abort();
            let mut home = self.home.0.lock();
            assert!(home.loaned && home.returned.is_none());
            home.returned = Some(supervisor);
            home.loaned = false;
        }
    }
}
