# Calc Flow Studio

The studio is the local React/TypeScript client for Calc Flow's FastAPI service.
It uses React Flow 12 and types generated from `openapi.json`.

Start the API and Vite server together from the repository root on macOS,
Linux, or WSL:

```bash
./web-ui/scripts/start_web_ui.sh
```

On native Windows PowerShell:

```powershell
.\web-ui\scripts\start_web_ui.ps1
```

Open `http://127.0.0.1:5173`. Logs and PID state are written to
`.calc-flow-web/`. Stop both managed process groups with the matching command
for your platform:

```bash
./web-ui/scripts/stop_web_ui.sh
```

```powershell
.\web-ui\scripts\stop_web_ui.ps1
```

Vite binds to `127.0.0.1` and proxies `/api` to
`http://127.0.0.1:8765`. The start script runs `npm ci` automatically when
`node_modules` is missing and waits until both services are reachable.

The benchmark comparison panel accepts baseline and current JSON reports from
`pytest-benchmark`. It matches cases by scenario and array backend, displays
mean-time deltas and coefficient of variation, and keeps noisy cases
informational according to the repository benchmark policy.

The stream configuration editor switches a project between batch and stream
mode and edits checkpoint, batch, connector, format, secret-reference,
watermark, delivery, and managed-state settings without placing credentials in
the document. The Job observatory starts persistent stream jobs, resumes SSE
status after disconnects, displays bounded metrics and results, and exposes
checkpoint, graceful shutdown, and cancellation controls.

## Edit data sources

Each Data Source card shows a bounded preview of its current text. Select
**Edit data** to open the large, centered editor. The dialog starts with the
card's latest confirmed text and keeps typing in a temporary draft:

- **Confirm** validates inline JSON and applies valid text to the card. Invalid
  inline JSON stays in the dialog with an error; JSON/JSONL, CSV, and Arrow IPC
  text remains opaque at this step.
- **Cancel**, the close button, Escape, or a backdrop click discards the
  temporary draft and leaves the card unchanged.
- Confirming does not save or validate the whole project. Subsequent **Save**
  and **Validate** operations use the confirmed card text; inline data sources
  apply to batch-mode projects.

Keyboard focus starts in the editor, stays inside the open dialog while
tabbing, and returns to the exact **Edit data** button after the dialog closes.
While a file is loading for a source, its editor cannot open; while its dialog
is open, **Load file** is disabled.

The top action toolbar uses consistently sized controls and wraps as a group
on narrow screens. The dialog, editor, validation message, and action row stay
within the viewport.

To run the two development processes manually, use separate terminals:

```bash
uv run --package calc-flow-studio calc-flow-web
cd web-ui && npm ci && npm run dev
```

A production build can be served by the Python service:

```bash
cd web-ui
npm ci
npm run build:wheel
cd ..
uv run --package calc-flow-studio calc-flow-web
```

Regenerate the checked-in API contract after backend route or model changes:

```bash
cd web-ui
npm run sync:api
```

Verification:

```bash
cd web-ui
npm ci
npm run sync:api
npm run build
npm test
npx playwright install chromium
npm run test:e2e
npm audit --omit=dev
cd backend
uv run --project . --extra dev pytest --cov=calc_flow_studio
```
