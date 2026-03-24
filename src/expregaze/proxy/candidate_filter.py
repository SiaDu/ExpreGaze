from __future__ import annotations

import pandas as pd


def add_valid_face_frames(df: pd.DataFrame) -> pd.DataFrame:
    """
    统计一个 shot 的 3 帧里，有几帧检测到了至少 1 张脸。
    """
    out = df.copy()
    out["num_frames_with_face"] = (
        (out["face_count_f1"] > 0).astype(int)
        + (out["face_count_f2"] > 0).astype(int)
        + (out["face_count_f3"] > 0).astype(int)
    )
    return out


def filter_strict_face_candidates(
    df: pd.DataFrame,
    min_consensus: int = 1,
    max_consensus: int = 3,
    max_count_range: int = 1,
    min_frames_with_face: int = 2,
    min_score: float = 0.70,
) -> pd.DataFrame:
    """
    从 face_df 中筛出更可靠的简单镜头：
    - 多数票人数在 1~3 之间
    - 三帧人数波动不超过 1
    - 至少 2 帧检测到脸
    - 平均检测分数 >= min_score
    """
    out = df.copy()

    if "num_frames_with_face" not in out.columns:
        out = add_valid_face_frames(out)

    mask = (
        out["consensus_count"].between(min_consensus, max_consensus)
        & (out["count_range"] <= max_count_range)
        & (out["num_frames_with_face"] >= min_frames_with_face)
        & (out["mean_det_score"] >= min_score)
    )

    return out.loc[mask].copy().reset_index(drop=True)
