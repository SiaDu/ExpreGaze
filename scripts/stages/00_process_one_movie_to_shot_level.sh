#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_CONFIG="${1:-configs/runs/text_main_tt1591095.yaml}"

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/expregaze-uv-cache}"

cd "$PROJECT_ROOT"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  "$PYTHON_BIN" -m expregaze.data.process_one_movie_to_shot_level \
    --run-config "$RUN_CONFIG"
elif command -v uv >/dev/null 2>&1; then
  uv run python -m expregaze.data.process_one_movie_to_shot_level \
    --run-config "$RUN_CONFIG"
else
  python3 -m expregaze.data.process_one_movie_to_shot_level \
    --run-config "$RUN_CONFIG"
fi
