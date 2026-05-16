#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_CONFIG="${1:-configs/runs/video_proxy_tt0032138.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi
EXTRA_ARGS=("$@")

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/expregaze-uv-cache}"

cd "$PROJECT_ROOT"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  "$PYTHON_BIN" -m expregaze.video_proxy.build_track_identities \
    --run-config "$RUN_CONFIG" \
    "${EXTRA_ARGS[@]}"
elif command -v uv >/dev/null 2>&1; then
  uv run python -m expregaze.video_proxy.build_track_identities \
    --run-config "$RUN_CONFIG" \
    "${EXTRA_ARGS[@]}"
else
  python3 -m expregaze.video_proxy.build_track_identities \
    --run-config "$RUN_CONFIG" \
    "${EXTRA_ARGS[@]}"
fi
