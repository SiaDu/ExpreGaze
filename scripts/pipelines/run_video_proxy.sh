#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_CONFIG="${1:-configs/runs/video_proxy_tt0032138.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi

EXTRA_ARGS=("$@")

cd "$PROJECT_ROOT"

echo "[video_proxy] Stage04: extract selected sequence videos and shot clips"
bash scripts/stages/04_extract_video_sequences.sh "$RUN_CONFIG" "${EXTRA_ARGS[@]}"

echo "[video_proxy] Stage05: face detection and shot-local tracking"
bash scripts/stages/05_run_face_detection_tracks.sh "$RUN_CONFIG" "${EXTRA_ARGS[@]}"

echo "[video_proxy] Stage06: per-track OpenFace and gaze timebins"
bash scripts/stages/06_run_openface_per_face.sh "$RUN_CONFIG" "${EXTRA_ARGS[@]}"

echo "[video_proxy] Stage07: build track identities"
bash scripts/stages/07_build_track_identities.sh "$RUN_CONFIG" "${EXTRA_ARGS[@]}"

echo "[video_proxy] Stage08: identity-aware proxy gaze assignment"
bash scripts/stages/08_build_proxy_gaze_script.sh "$RUN_CONFIG" "${EXTRA_ARGS[@]}"

echo "[video_proxy] Stage09: build final proxy table"
bash scripts/stages/09_build_final_proxy_table.sh "$RUN_CONFIG" "${EXTRA_ARGS[@]}"

echo "[video_proxy] Done"
