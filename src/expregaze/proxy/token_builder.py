from __future__ import annotations

import pandas as pd


def make_proxy_token(row: pd.Series) -> str:
    subj = "SUBJ_MAIN"
    interaction = row["interaction_type"]
    target = row["visual_target_coarse"]
    dur = row["duration_bin"]
    return f"<{subj}> <{interaction.upper()}> <{target.upper()}> <{dur}>"


def add_proxy_token_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["proxy_token"] = out.apply(make_proxy_token, axis=1)
    return out
    
def map_interaction_type(consensus_count):
    if pd.isna(consensus_count):
        return "unclear"
    if consensus_count == 1:
        return "single"
    elif consensus_count == 2:
        return "two_person"
    elif consensus_count >= 3:
        return "multi_person"
    else:
        return "unclear"

def infer_visual_target_coarse(row):
    count = row["consensus_count"]
    head_dir = row["head_dir_consensus"]

    if pd.isna(count) or pd.isna(head_dir):
        return "unclear"

    if count == 1:
        if head_dir == "front":
            return "front_center"
        elif head_dir == "left":
            return "offscreen_left"
        elif head_dir == "right":
            return "offscreen_right"
        else:
            return "unclear"

    elif count >= 2:
        if head_dir in ["left", "right"]:
            return "onscreen_person"
        elif head_dir == "front":
            return "front_center"
        else:
            return "unclear"

    return "unclear"

def duration_to_bin_v2(d):
    if pd.isna(d):
        return "DU_UNK"
    elif d < 2:
        return "DU_01"
    elif d < 5:
        return "DU_02"
    elif d < 10:
        return "DU_03"
    elif d < 20:
        return "DU_04"
    else:
        return "DU_05"
