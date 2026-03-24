from __future__ import annotations

import math
from pathlib import Path

import cv2
import pandas as pd

from .face_detector import detect_faces_yunet


def select_main_subject_from_det_df(det_df: pd.DataFrame, image_shape) -> pd.Series | None:
    if det_df is None or len(det_df) == 0:
        return None

    h, w = image_shape[:2]
    cx_img, cy_img = w / 2.0, h / 2.0
    diag = math.sqrt(w * w + h * h)

    scored_rows = []
    for idx, r in det_df.iterrows():
        x, y, bw, bh = r["x"], r["y"], r["w"], r["h"]
        cx = x + bw / 2.0
        cy = y + bh / 2.0
        area_score = (bw * bh) / (w * h)
        dist = math.sqrt((cx - cx_img) ** 2 + (cy - cy_img) ** 2)
        center_score = 1.0 - (dist / diag)
        det_score = float(r["score"])
        total_score = 0.45 * area_score + 0.35 * center_score + 0.20 * det_score
        scored_rows.append((idx, total_score, area_score, center_score, det_score))

    scored_rows.sort(key=lambda x: x[1], reverse=True)
    best_idx, total_score, area_score, center_score, det_score = scored_rows[0]
    best_row = det_df.loc[best_idx].copy()
    best_row["main_subject_score"] = total_score
    best_row["area_score"] = area_score
    best_row["center_score"] = center_score
    best_row["det_score_used"] = det_score
    return best_row


def extract_main_subject_for_one_frame(frame_path: Path, detector) -> dict | None:
    img = cv2.imread(str(frame_path))
    if img is None:
        return None

    face_count, det_df, _ = detect_faces_yunet(frame_path, detector)
    best_row = select_main_subject_from_det_df(det_df, img.shape)

    if best_row is None:
        return {
            "frame_path": str(frame_path),
            "detected_faces": 0,
            "main_found": False,
        }

    keys = [
        "x", "y", "w", "h", "score",
        "left_eye_x", "left_eye_y", "right_eye_x", "right_eye_y",
        "nose_x", "nose_y", "mouth_left_x", "mouth_left_y", "mouth_right_x", "mouth_right_y",
        "main_subject_score",
    ]
    out = {
        "frame_path": str(frame_path),
        "detected_faces": int(face_count),
        "main_found": True,
        "main_x": float(best_row["x"]),
        "main_y": float(best_row["y"]),
        "main_w": float(best_row["w"]),
        "main_h": float(best_row["h"]),
        "main_score": float(best_row["main_subject_score"]),
        "det_score": float(best_row["score"]),
    }
    out.update(
        {
            "left_eye_x": float(best_row["left_eye_x"]),
            "left_eye_y": float(best_row["left_eye_y"]),
            "right_eye_x": float(best_row["right_eye_x"]),
            "right_eye_y": float(best_row["right_eye_y"]),
            "nose_x": float(best_row["nose_x"]),
            "nose_y": float(best_row["nose_y"]),
            "mouth_left_x": float(best_row["mouth_left_x"]),
            "mouth_left_y": float(best_row["mouth_left_y"]),
            "mouth_right_x": float(best_row["mouth_right_x"]),
            "mouth_right_y": float(best_row["mouth_right_y"]),
        }
    )
    return out


def build_main_subject_df(strict_candidate_df: pd.DataFrame, frame_root: Path, detector) -> pd.DataFrame:
    rows = []
    for _, row in strict_candidate_df.iterrows():
        shot_idx = int(row["shot_idx"])
        shot_dir = frame_root / f"shot_{shot_idx:04d}"
        row_out = {
            "shot_idx": shot_idx,
            "shot_id": f"shot_{shot_idx:04d}",
            "subtitle_text": row.get("subtitle_text", ""),
            "consensus_count": row.get("consensus_count"),
            "mean_det_score": row.get("mean_det_score"),
        }
        for fi in [1, 2, 3]:
            frame_path = shot_dir / f"frame_{fi}.jpg"
            info = extract_main_subject_for_one_frame(frame_path, detector)
            if info is None:
                row_out[f"f{fi}_main_found"] = False
                continue
            for k, v in info.items():
                row_out[f"f{fi}_{k}"] = v
        rows.append(row_out)
    return pd.DataFrame(rows).drop_duplicates(subset=["shot_id"]).reset_index(drop=True)
