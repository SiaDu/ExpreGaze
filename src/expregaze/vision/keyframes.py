from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd


DEFAULT_SAMPLE_RATIOS = (0.25, 0.50, 0.75)


def extract_keyframes_for_movie(
    video_path: Path,
    shot_df: pd.DataFrame,
    output_root: Path,
    sample_ratios: tuple[float, float, float] = DEFAULT_SAMPLE_RATIOS,
) -> Path:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_root.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        cap.release()
        raise RuntimeError(f"Invalid FPS from video: {video_path}")

    for _, row in shot_df.iterrows():
        try:
            shot_idx = int(row["shot_idx"])
            start_t = float(row["shot_start_time"])
            end_t = float(row["shot_end_time"])
        except (KeyError, TypeError, ValueError):
            continue

        if end_t <= start_t:
            continue

        duration = end_t - start_t
        sample_times = [start_t + duration * ratio for ratio in sample_ratios]
        shot_dir = output_root / f"shot_{shot_idx:04d}"
        shot_dir.mkdir(parents=True, exist_ok=True)

        for i, t in enumerate(sample_times, start=1):
            frame_id = int(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if not ret:
                continue
            save_path = shot_dir / f"frame_{i}.jpg"
            cv2.imwrite(str(save_path), frame)

    cap.release()
    return output_root
