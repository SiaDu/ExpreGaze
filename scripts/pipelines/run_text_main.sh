#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_CONFIG="${1:-configs/runs/text_main_tt1591095.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi

ALIGN_MODE="${1:-raw}"
if [[ $# -gt 0 ]]; then
  shift
fi

EXTRA_ALIGN_ARGS=("$@")

cd "$PROJECT_ROOT"

echo "[text_main] Stage00: process one movie to shot-level"
bash scripts/stages/00_process_one_movie_to_shot_level.sh "$RUN_CONFIG"

echo "[text_main] Stage01: align full context ($ALIGN_MODE)"
bash scripts/stages/01_align_full_context_with_gpt.sh "$RUN_CONFIG" "$ALIGN_MODE" "${EXTRA_ALIGN_ARGS[@]}"

echo "[text_main] Done"
