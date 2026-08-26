# Calc Flow Studio

`calc-flow-studio` packages the local FastAPI service and built React client for
Calc Flow. It depends on the matching v4 `calc-flow` native package and serves
the continuous-job API under `/api/v3`.

Install and start the loopback-only service with:

```bash
uv tool install calc-flow-studio
calc-flow-web
```

Open `http://127.0.0.1:8765`. The server rejects non-loopback hosts by default.
See the repository
[getting-started guide](https://github.com/wegamekinglc/calc-flow/blob/main/docs/getting-started.md)
for installation, lifecycle, and verification details.

Calc Flow Studio is licensed under Apache-2.0.
