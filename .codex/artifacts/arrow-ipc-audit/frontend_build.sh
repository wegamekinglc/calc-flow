#!/usr/bin/env bash
set -euo pipefail
task_root="$PWD/target/arrow-ipc-extension"
export npm_config_cache="$task_root/npm-cache"
cd target/arroyo-arrow-source/webui
npm exec --yes --package=pnpm@9.7.1 -- \
  pnpm install --frozen-lockfile --store-dir "$task_root/pnpm-store"
npm exec --yes --package=pnpm@9.7.1 -- pnpm run build
