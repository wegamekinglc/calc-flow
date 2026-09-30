static DIAG_EVENTS: std::sync::OnceLock<Mutex<Vec<serde_json::Value>>> = std::sync::OnceLock::new();
static DIAG_ACTIVE: AtomicUsize = AtomicUsize::new(0);
static DIAG_MAX_ACTIVE: AtomicUsize = AtomicUsize::new(0);
static DIAG_DROPPED: AtomicUsize = AtomicUsize::new(0);
static DIAG_STARTED: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();

fn diag_emit(event: &str, value: serde_json::Value) {
    let mut events = DIAG_EVENTS
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .unwrap();
    if events.len() < 8192 {
        let sequence = events.len();
        let elapsed = DIAG_STARTED.get_or_init(std::time::Instant::now).elapsed();
        events.push(json!({"sequence": sequence, "elapsed_us": elapsed.as_micros(), "event": event, "thread": format!("{:?}", std::thread::current().id()), "data": value}));
    } else {
        DIAG_DROPPED.fetch_add(1, Ordering::SeqCst);
    }
}

fn diag_flush() {
    let events = DIAG_EVENTS
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .unwrap();
    if let Ok(root) = std::env::var("DAL313_DIAGNOSTIC_OUTPUT") {
        let path = std::path::PathBuf::from(root).join("rust-events.json");
        let evidence =
            json!({"events": &*events, "dropped_events": DIAG_DROPPED.load(Ordering::SeqCst)});
        if let Err(error) = std::fs::write(path, serde_json::to_vec(&evidence).unwrap()) {
            eprintln!("DAL313 observation write failed: {error}");
        }
    }
}

struct DiagActivity(&'static str);

impl DiagActivity {
    fn new(name: &'static str) -> Self {
        let active = DIAG_ACTIVE.fetch_add(1, Ordering::SeqCst) + 1;
        DIAG_MAX_ACTIVE.fetch_max(active, Ordering::SeqCst);
        diag_emit(
            "harness_case_enter",
            json!({"case": name, "active_cases": active, "available_parallelism": std::thread::available_parallelism().map(|value| value.get()).ok(), "rust_test_threads": std::env::var("RUST_TEST_THREADS").ok()}),
        );
        Self(name)
    }
}

impl Drop for DiagActivity {
    fn drop(&mut self) {
        let active = DIAG_ACTIVE.fetch_sub(1, Ordering::SeqCst) - 1;
        diag_emit(
            "harness_case_exit",
            json!({"case": self.0, "active_cases": active, "maximum_active_cases": DIAG_MAX_ACTIVE.load(Ordering::SeqCst)}),
        );
        diag_flush();
    }
}

struct DiagCase {
    name: &'static str,
    combo: Mutex<String>,
    started: std::time::Instant,
    sampled: std::sync::atomic::AtomicBool,
}

impl DiagCase {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            combo: Mutex::new(String::new()),
            started: std::time::Instant::now(),
            sampled: std::sync::atomic::AtomicBool::new(false),
        }
    }

    fn label(&self) -> String {
        format!("{}/{}", self.name, self.combo.lock().unwrap())
    }

    fn set_combo(&self, combo: String) {
        *self.combo.lock().unwrap() = combo;
        diag_emit(
            "combo_enter",
            json!({"case": self.label(), "elapsed_us": self.started.elapsed().as_micros()}),
        );
    }
}

struct DiagWaiting<'a> {
    case: &'a DiagCase,
    stage: &'static str,
    job: Option<&'a StreamingJob>,
    returned: bool,
}

impl Drop for DiagWaiting<'_> {
    fn drop(&mut self) {
        if !self.returned {
            diag_emit(
                "pending_future_dropped",
                json!({"case": self.case.label(), "stage": self.stage, "status": self.job.map(StreamingJob::status), "elapsed_us": self.case.started.elapsed().as_micros()}),
            );
            diag_flush();
        }
    }
}

async fn diag_poll<T>(
    case: &DiagCase,
    stage: &'static str,
    future: impl Future<Output = T>,
    job: Option<&StreamingJob>,
) -> T {
    let mut guard = DiagWaiting {
        case,
        stage,
        job,
        returned: false,
    };
    diag_emit(
        "wait_enter",
        json!({"case": case.label(), "stage": stage, "status": job.map(StreamingJob::status), "elapsed_us": case.started.elapsed().as_micros()}),
    );
    let mut future = Box::pin(future);
    let value = if case.sampled.load(Ordering::SeqCst) {
        future.await
    } else {
        let delay = Duration::from_secs(24).saturating_sub(case.started.elapsed());
        tokio::select! {
            biased;
            value = &mut future => value,
            () = tokio::time::sleep(delay) => {
                case.sampled.store(true, Ordering::SeqCst);
                diag_emit("diag_before_deadline", json!({"case": case.label(), "stage": stage, "status": job.map(StreamingJob::status), "elapsed_us": case.started.elapsed().as_micros(), "sampling_thread_backtrace": std::backtrace::Backtrace::force_capture().to_string(), "other_native_threads": "not available through existing public surface"}));
                diag_flush();
                future.await
            }
        }
    };
    guard.returned = true;
    diag_emit(
        "wait_return",
        json!({"case": case.label(), "stage": stage, "status": job.map(StreamingJob::status), "elapsed_us": case.started.elapsed().as_micros()}),
    );
    value
}

async fn diag_start<T>(case: &DiagCase, stage: &'static str, future: impl Future<Output = T>) -> T {
    diag_poll(case, stage, future, None).await
}

async fn diag_stage<T: std::fmt::Debug>(
    case: &DiagCase,
    stage: &'static str,
    future: impl Future<Output = T>,
    job: Option<&StreamingJob>,
) -> T {
    let value = diag_poll(case, stage, future, job).await;
    diag_emit(
        "outcome",
        json!({"case": case.label(), "stage": stage, "value": format!("{value:?}")}),
    );
    value
}
