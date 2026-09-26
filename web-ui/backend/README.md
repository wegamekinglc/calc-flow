# Calc Flow Studio

`calc-flow-studio` packages the local FastAPI service and built React client for
Calc Flow. It depends on the matching v5 `calc-flow-python` native package and serves
the continuous-job API under `/api/v3`.

Studio is built as a separate wheel and is not published to PyPI. Follow the
[source installation guide](../../docs/getting-started.md#build-and-install-from-source)
to build and install the core and Studio wheels, including the React assets.
Then start the loopback-only service from the repository root:

```bash
uv run --no-sync --package calc-flow-studio calc-flow-web
```

Open `http://127.0.0.1:8765`. The server rejects non-loopback hosts by default.
HTTP clients can read a launch-specific token from `GET /api/v3/session` and
send it as `X-Calc-Flow-Token` on `POST`, `PUT`, `PATCH`, and `DELETE` requests.
The browser client handles this automatically. Cross-origin requests must come
from the local Studio development origin.
See the repository
[getting-started guide](https://github.com/wegamekinglc/calc-flow/blob/main/docs/getting-started.md)
for installation, lifecycle, and verification details.

Calc Flow Studio is licensed under Apache-2.0.
