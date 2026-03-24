#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${1:-/home/sdu/Desktop/ExpreGaze}
MOVIE_ID=${2:-tt0032138}
VIDEO_PATH=${3:-}

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

if [[ -n "$VIDEO_PATH" ]]; then
  python -m expregaze.pipeline.run_movie_pipeline \
    --project-root "$PROJECT_ROOT" \
    --movie-id "$MOVIE_ID" \
    --video "$VIDEO_PATH"
else
  python -m expregaze.pipeline.run_movie_pipeline \
    --project-root "$PROJECT_ROOT" \
    --movie-id "$MOVIE_ID"
fi
