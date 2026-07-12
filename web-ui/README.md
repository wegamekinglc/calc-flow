# Calc Flow Studio

The studio is the local React/TypeScript client for Calc Flow's FastAPI service.
It uses React Flow 12 and types generated from `openapi.json`.

Start the API and Vite server together from the repository root:

```bash
./web-ui/scripts/start_web_ui.sh
```

Open `http://127.0.0.1:5173`. Logs and PID state are written to
`.calc-flow-web/`. Stop both managed process groups with:

```bash
./web-ui/scripts/stop_web_ui.sh
```

Vite binds to `127.0.0.1` and proxies `/api` to
`http://127.0.0.1:8765`. The start script runs `npm ci` automatically when
`node_modules` is missing and waits until both services are reachable.

The benchmark comparison panel accepts baseline and current JSON reports from
`pytest-benchmark`. It matches cases by scenario and array backend, displays
mean-time deltas and coefficient of variation, and keeps noisy cases
informational according to the repository benchmark policy.

Runner recovery controls inspect and reset checkpoints stored for the current
pipeline. They report fingerprint compatibility, cursor, sequence, and stateful
node names. Bounded preview runs are intentionally stateless and do not create
runner checkpoints.

To run the two development processes manually, use separate terminals:

```bash
uv run --package calc-flow-studio calc-flow-web
cd web-ui && npm ci && npm run dev
```

A production build can be served by the Python service:

```bash
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
npm run build
npm test
npx playwright install chromium
npm run test:e2e
npm audit --omit=dev
cd backend
uv run --project . --extra dev pytest --cov=calc_flow_studio
```
