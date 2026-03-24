from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd


def estimate_head_dir_from_landmarks(
    left_eye_x,
    left_eye_y,
    right_eye_x,
    right_eye_y,
    nose_x,
    nose_y,
    thresh: float = 0.10,
):
    vals = [left_eye_x, left_eye_y, right_eye_x, right_eye_y, nose_x, nose_y]
    if any(pd.isna(v) for v in vals):
        return "unclear", np.nan

    inter_eye = abs(right_eye_x - left_eye_x)
    if inter_eye < 1e-6:
        return "unclear", np.nan

    eye_mid_x = (left_eye_x + right_eye_x) / 2.0
    nose_offset_norm = (nose_x - eye_mid_x) / inter_eye

    if nose_offset_norm > thresh:
        return "right", nose_offset_norm
    if nose_offset_norm < -thresh:
        return "left", nose_offset_norm
    return "front", nose_offset_norm


def add_framewise_head_direction(df: pd.DataFrame, thresh: float = 0.10) -> pd.DataFrame:
    out = df.copy()
    for fi in [1, 2, 3]:
        dirs = []
        offsets = []
        for _, row in out.iterrows():
            head_dir, offset = estimate_head_dir_from_landmarks(
                row.get(f"f{fi}_left_eye_x", np.nan),
                row.get(f"f{fi}_left_eye_y", np.nan),
                row.get(f"f{fi}_right_eye_x", np.nan),
                row.get(f"f{fi}_right_eye_y", np.nan),
                row.get(f"f{fi}_nose_x", np.nan),
                row.get(f"f{fi}_nose_y", np.nan),
                thresh=thresh,
            )
            dirs.append(head_dir)
            offsets.append(offset)
        out[f"f{fi}_head_dir"] = dirs
        out[f"f{fi}_nose_offset_norm"] = offsets
    return out


def get_head_dir_consensus(row: pd.Series) -> pd.Series:
    dirs = [row.get("f1_head_dir", "unclear"), row.get("f2_head_dir", "unclear"), row.get("f3_head_dir", "unclear")]
    dirs = [d for d in dirs if d != "unclear"]
    if len(dirs) == 0:
        return pd.Series({"head_dir_consensus": "unclear", "head_dir_stable": False})
    most_common_dir, count = Counter(dirs).most_common(1)[0]
    return pd.Series({"head_dir_consensus": most_common_dir, "head_dir_stable": count >= 2})


def add_head_dir_consensus(df: pd.DataFrame) -> pd.DataFrame:
    consensus_df = df.apply(get_head_dir_consensus, axis=1)
    return pd.concat([df, consensus_df], axis=1)
