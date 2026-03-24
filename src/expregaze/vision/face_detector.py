from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


YUNET_COLUMNS = [
    "x",
    "y",
    "w",
    "h",
    "left_eye_x",
    "left_eye_y",
    "right_eye_x",
    "right_eye_y",
    "nose_x",
    "nose_y",
    "mouth_left_x",
    "mouth_left_y",
    "mouth_right_x",
    "mouth_right_y",
    "score",
]


def create_yunet_detector(
    model_path: Path,
    score_threshold: float = 0.6,
    nms_threshold: float = 0.3,
    top_k: int = 5000,
) -> cv2.FaceDetectorYN:
    if not model_path.exists():
        raise FileNotFoundError(f"YuNet model not found: {model_path}")
    return cv2.FaceDetectorYN.create(
        model=str(model_path),
        config="",
        input_size=(320, 320),
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
        top_k=top_k,
    )


def detect_faces_yunet(image_path: Path, detector: cv2.FaceDetectorYN, resize_max: int = 960):
    img = cv2.imread(str(image_path))
    if img is None:
        return 0, pd.DataFrame(columns=YUNET_COLUMNS), None

    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > resize_max:
        scale = resize_max / max(h, w)
        img_in = cv2.resize(img, (int(w * scale), int(h * scale)))
    else:
        img_in = img.copy()

    ih, iw = img_in.shape[:2]
    detector.setInputSize((iw, ih))
    _, faces = detector.detect(img_in)

    if faces is None or len(faces) == 0:
        return 0, pd.DataFrame(columns=YUNET_COLUMNS), None

    rows = []
    inv = 1.0 / scale
    for f in faces:
        x, y, bw, bh = f[:4]
        le_x, le_y, re_x, re_y, nose_x, nose_y, lm_x, lm_y, rm_x, rm_y = f[4:14]
        score = f[14]
        rows.append(
            {
                "x": float(x * inv),
                "y": float(y * inv),
                "w": float(bw * inv),
                "h": float(bh * inv),
                "left_eye_x": float(le_x * inv),
                "left_eye_y": float(le_y * inv),
                "right_eye_x": float(re_x * inv),
                "right_eye_y": float(re_y * inv),
                "nose_x": float(nose_x * inv),
                "nose_y": float(nose_y * inv),
                "mouth_left_x": float(lm_x * inv),
                "mouth_left_y": float(lm_y * inv),
                "mouth_right_x": float(rm_x * inv),
                "mouth_right_y": float(rm_y * inv),
                "score": float(score),
            }
        )

    det_df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return len(det_df), det_df, faces


def summarize_one_shot_face_counts_yunet(shot_dir: Path, detector: cv2.FaceDetectorYN) -> dict:
    frame_paths = sorted(shot_dir.glob("*.jpg"))
    frame_counts: list[int] = []
    frame_scores: list[float] = []

    for fp in frame_paths:
        cnt, det_df, _ = detect_faces_yunet(fp, detector)
        frame_counts.append(cnt)
        frame_scores.append(float(det_df["score"].mean()) if len(det_df) > 0 else 0.0)

    if len(frame_counts) == 0:
        return {
            "shot_id": shot_dir.name,
            "face_count_f1": None,
            "face_count_f2": None,
            "face_count_f3": None,
            "consensus_count": None,
            "count_range": None,
            "mean_det_score": None,
            "is_stable": False,
            "is_candidate_1to3": False,
        }

    consensus_count = Counter(frame_counts).most_common(1)[0][0]
    count_range = max(frame_counts) - min(frame_counts)
    mean_det_score = float(np.mean(frame_scores)) if frame_scores else 0.0
    is_stable = count_range <= 1
    is_candidate_1to3 = is_stable and (1 <= consensus_count <= 3)

    return {
        "shot_id": shot_dir.name,
        "face_count_f1": frame_counts[0] if len(frame_counts) > 0 else None,
        "face_count_f2": frame_counts[1] if len(frame_counts) > 1 else None,
        "face_count_f3": frame_counts[2] if len(frame_counts) > 2 else None,
        "consensus_count": consensus_count,
        "count_range": count_range,
        "mean_det_score": mean_det_score,
        "is_stable": is_stable,
        "is_candidate_1to3": is_candidate_1to3,
    }


def build_face_count_df(frame_root: Path, detector: cv2.FaceDetectorYN) -> pd.DataFrame:
    shot_dirs = sorted([p for p in frame_root.iterdir() if p.is_dir()])
    results = [summarize_one_shot_face_counts_yunet(shot_dir, detector) for shot_dir in shot_dirs]
    return pd.DataFrame(results)


def add_valid_face_frames(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["num_frames_with_face"] = (
        out["face_count_f1"].fillna(0).gt(0).astype(int)
        + out["face_count_f2"].fillna(0).gt(0).astype(int)
        + out["face_count_f3"].fillna(0).gt(0).astype(int)
    )
    return out


def strict_filter_face_df(df: pd.DataFrame, min_score: float = 0.70) -> pd.DataFrame:
    out = add_valid_face_frames(df)
    mask = (
        out["consensus_count"].between(1, 3)
        & out["count_range"].le(1)
        & out["num_frames_with_face"].ge(2)
        & out["mean_det_score"].ge(min_score)
    )
    return out.loc[mask].copy().reset_index(drop=True)
