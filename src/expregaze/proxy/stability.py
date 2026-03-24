from __future__ import annotations

import numpy as np
import pandas as pd


def compute_main_subject_stability(row: pd.Series, center_std_thresh: float = 120.0) -> pd.Series:
    centers = []
    for fi in [1, 2, 3]:
        if row.get(f"f{fi}_main_found", False):
            x = row.get(f"f{fi}_main_x")
            y = row.get(f"f{fi}_main_y")
            w = row.get(f"f{fi}_main_w")
            h = row.get(f"f{fi}_main_h")
            if None not in [x, y, w, h] and not any(pd.isna(v) for v in [x, y, w, h]):
                centers.append((x + w / 2.0, y + h / 2.0))

    if len(centers) < 2:
        return pd.Series({"main_subject_center_std": np.nan, "main_subject_stable": False})

    xs = [c[0] for c in centers]
    ys = [c[1] for c in centers]
    std_val = float(np.std(xs) + np.std(ys))
    return pd.Series({
        "main_subject_center_std": std_val,
        "main_subject_stable": std_val < center_std_thresh,
    })


def bbox_iou(box_a, box_b):
    if any(pd.isna(v) for v in box_a) or any(pd.isna(v) for v in box_b):
        return np.nan

    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter_area
    if union <= 0:
        return np.nan
    return inter_area / union


def compute_bbox_stability(row: pd.Series, iou_thresh: float = 0.25) -> pd.Series:
    b1 = (row.get("f1_main_x", np.nan), row.get("f1_main_y", np.nan), row.get("f1_main_w", np.nan), row.get("f1_main_h", np.nan))
    b2 = (row.get("f2_main_x", np.nan), row.get("f2_main_y", np.nan), row.get("f2_main_w", np.nan), row.get("f2_main_h", np.nan))
    b3 = (row.get("f3_main_x", np.nan), row.get("f3_main_y", np.nan), row.get("f3_main_w", np.nan), row.get("f3_main_h", np.nan))

    ious = [bbox_iou(b1, b2), bbox_iou(b2, b3), bbox_iou(b1, b3)]
    valid_ious = [x for x in ious if not pd.isna(x)]

    if len(valid_ious) == 0:
        return pd.Series({"main_bbox_iou_mean": np.nan, "main_bbox_stable": False})

    iou_mean = float(np.mean(valid_ious))
    return pd.Series({
        "main_bbox_iou_mean": iou_mean,
        "main_bbox_stable": iou_mean >= iou_thresh,
    })


def add_stability_columns(main_subject_df: pd.DataFrame) -> pd.DataFrame:
    center_df = main_subject_df.apply(compute_main_subject_stability, axis=1)
    out = pd.concat([main_subject_df, center_df], axis=1)
    bbox_df = out.apply(compute_bbox_stability, axis=1)
    out = pd.concat([out, bbox_df], axis=1)
    return out


def filter_final_proxy_candidates(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["head_dir_stable"].eq(True) & (
        df["main_subject_stable"].eq(True) | df["main_bbox_stable"].eq(True)
    )
    return df.loc[mask].copy().reset_index(drop=True)
