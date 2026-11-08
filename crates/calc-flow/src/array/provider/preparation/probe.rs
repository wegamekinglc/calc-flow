use crate::Result;
use std::sync::Mutex;
use tokio::sync::oneshot;

type Work = Box<dyn FnOnce(&mut dyn FnMut() -> Result<()>) -> Result<()> + Send>;

#[derive(Default)]
pub(super) struct Probe {
    admission: Mutex<Option<oneshot::Sender<()>>>,
    work: Mutex<Option<Work>>,
}
impl Probe {
    pub(super) fn observe_admission(&self) -> oneshot::Receiver<()> {
        let (sender, receiver) = oneshot::channel();
        *self.admission.lock().unwrap() = Some(sender);
        receiver
    }
    pub(super) fn install(&self, work: Work) {
        *self.work.lock().unwrap() = Some(work);
    }
    pub(super) fn entered(&self) {
        if let Some(sender) = self.admission.lock().unwrap().take() {
            let _ = sender.send(());
        }
    }
    pub(super) fn run(&self, check: &mut dyn FnMut() -> Result<()>) -> Result<()> {
        let work = self.work.lock().unwrap().take();
        if let Some(work) = work {
            work(check)?;
        }
        Ok(())
    }
}
