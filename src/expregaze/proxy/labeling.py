from __future__ import annotations

import pandas as pd


def map_interaction_type(consensus_count) -> str:
    if pd.isna(consensus_count):
        return "unclear"
    if consensus_count == 1:
        return "single"
    if consensus_count == 2:
        return "two_person"
    if consensus_count >= 3:
        return "multi_person"
    return "unclear"


def infer_visual_target_coarse(row: pd.Series) -> str:
    count = row.get("consensus_count")
    head_dir = row.get("head_dir_consensus")

    if pd.isna(count) or pd.isna(head_dir):
        return "unclear"

    if count == 1:
        if head_dir == "front":
            return "front_center"
        if head_dir == "left":
            return "offscreen_left"
        if head_dir == "right":
            return "offscreen_right"
        return "unclear"

    if count >= 2:
        if head_dir in ["left", "right"]:
            return "onscreen_person"
        if head_dir == "front":
            return "front_center"
        return "unclear"

    return "unclear"


def duration_to_bin_v2(d) -> str:
    if pd.isna(d):
        return "DU_UNK"
    if d < 2:
        return "DU_01"
    if d < 5:
        return "DU_02"
    if d < 10:
        return "DU_03"
    if d < 20:
        return "DU_04"
    return "DU_05"


def add_proxy_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["interaction_type"] = out["consensus_count"].apply(map_interaction_type)
    out["visual_target_coarse"] = out.apply(infer_visual_target_coarse, axis=1)
    out["duration_bin"] = out["shot_duration"].apply(duration_to_bin_v2)
    return out
