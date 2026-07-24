# Windows Studio Launchers Design

**Status:** Implemented and merged in PR #15 (historical design)

## Goal

Provide native PowerShell commands that start and stop Calc Flow Studio on
Windows with the same managed lifecycle as the existing Bash wrappers. The
Windows flow must run the FastAPI backend and Vite frontend in the background,
wait until both are ready, retain logs and process state, and stop both process
trees reliably.

## Approach

Keep `web_ui_process.py` as the single lifecycle implementation and add
platform-specific process primitives behind its existing start, stop, and
status operations. Add thin `start_web_ui.ps1` and `stop_web_ui.ps1` wrappers
that locate the repository environment through `uv`, invoke the Python process
manager, forward command-line arguments, and return its exit code.

This avoids duplicating dependency installation, state-file handling,
readiness polling, partial-start cleanup, and user-facing messages in
PowerShell. The existing Bash wrappers and POSIX behavior remain unchanged.

## Process Lifecycle

The manager continues to write versioned state and logs beneath
`.calc-flow-web/`. It starts these commands from their current working
directories:

- API: `uv run --package calc-flow-studio calc-flow-web` from the repository
  root.
- Studio: `npm run dev -- --host 127.0.0.1 --port 5173 --strictPort` from
  `web-ui/`.

On Windows, each service starts in a new process group without opening an
extra console window. State records retain the root process ID and a process
creation token so a reused PID cannot be mistaken for the original service.
Windows liveness and creation-time checks use the standard-library Win32 API
bindings. Stopping uses the recorded identity and terminates the complete
descendant tree with Windows' built-in process-tree command. POSIX continues
to use process groups and signals.

Start remains idempotent. A complete live state returns the existing service
URLs. Stale or partial state is cleaned before a new start. If either service
exits or readiness times out, every service started by that attempt is stopped
and the state file is removed.

Stop remains idempotent. Missing state reports that Studio is already stopped.
Valid state stops the frontend and backend trees, removes the state file, and
leaves logs in place.

## PowerShell Interface

From the repository root, users run:

```powershell
.\web-ui\scripts\start_web_ui.ps1
.\web-ui\scripts\stop_web_ui.ps1
```

Both wrappers pass remaining arguments to the process manager, including
`--timeout` and `--runtime-dir`. They run Python through the repository's
existing `uv` environment and propagate nonzero exit codes. Missing `uv`, an
unsynchronized environment, missing `npm`, startup failure, corrupt state, and
timeout errors remain explicit rather than being hidden by the wrapper.

## Testing

Focused tests will cover:

- PowerShell syntax for both wrappers.
- Correct start/stop delegation, argument forwarding, and exit-code
  propagation.
- Windows process identity and liveness behavior without terminating unrelated
  processes.
- Windows process-tree termination command construction and errors.
- Existing stopped-state, idempotent-stop, workspace-backend, and POSIX wrapper
  behavior.

Tests will be written and observed failing before each production change. The
focused Studio backend tests, Ruff checks, and repository diff checks will be
run after implementation. README instructions will document both Bash and
native PowerShell commands.

## Acceptance Criteria

- Native PowerShell users can start both Studio services with one command and
  stop both with one command.
- Start does not return success until the API and frontend are reachable.
- Repeated start and stop calls are safe.
- A failed or partial start does not leave managed child processes or live
  state behind.
- Stop terminates descendant processes created by `uv`, `npm`, Vite, and the
  backend.
- Logs remain available beneath the configured runtime directory.
- Unix launcher behavior and commands remain compatible.
