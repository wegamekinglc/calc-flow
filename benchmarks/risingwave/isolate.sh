#!/usr/bin/env bash
# No fallback: missing bubblewrap or namespace support prevents engine startup.
set -euo pipefail
if [[ $# -eq 0 ]]; then
  echo 'Usage: bash benchmarks/risingwave/isolate.sh COMMAND [ARG ...]' >&2
  exit 2
fi
comparison_parent_netns=$(/usr/bin/readlink /proc/self/ns/net)
comparison_parent_pidns=$(/usr/bin/readlink /proc/self/ns/pid)
exec /usr/bin/bwrap \
  --unshare-user --unshare-net --unshare-pid --die-with-parent \
  --bind / / --proc /proc --cap-drop ALL \
  --setenv CALC_FLOW_RW_PARENT_NETNS "$comparison_parent_netns" \
  --setenv CALC_FLOW_RW_PARENT_PIDNS "$comparison_parent_pidns" \
  -- "$@"
