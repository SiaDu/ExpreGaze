from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from expregaze.proxy.labeling import add_proxy_label_columns
from expregaze.vision.face_detector import detect_faces_yunet
from expregaze.vision.head_direction import estimate_head_dir_from_landmarks
from expregaze.vision.main_subject import select_main_subject_from_det_df


# ---------- low-level helpers ----------

def _read_rgb(image_path: str | Path) -> np.ndarray:
    image_path = Path(image_path)
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _to_int_xyxy(x: float, y: float, w: float, h: float) -> tuple[int, int, int, int]:
    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + w))
    y2 = int(round(y + h))
    return x1, y1, x2, y2


def _draw_box(
    img: np.ndarray,
    x: float,
    y: float,
    w: float,
    h: float,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    out = img.copy()
    x1, y1, x2, y2 = _to_int_xyxy(x, y, w, h)
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    return out


def _draw_landmarks(img: np.ndarray, row: pd.Series, radius: int = 4) -> np.ndarray:
    out = img.copy()
    points = [
        (row.get("left_eye_x"), row.get("left_eye_y")),
        (row.get("right_eye_x"), row.get("right_eye_y")),
        (row.get("nose_x"), row.get("nose_y")),
        (row.get("mouth_left_x"), row.get("mouth_left_y")),
        (row.get("mouth_right_x"), row.get("mouth_right_y")),
    ]
    for px, py in points:
        if pd.notna(px) and pd.notna(py):
            cv2.circle(out, (int(round(px)), int(round(py))), radius, (255, 255, 0), -1)
    return out


def _put_label(
    img: np.ndarray,
    text: str,
    xy: tuple[int, int],
    color: tuple[int, int, int] = (255, 255, 255),
    bg: tuple[int, int, int] = (0, 0, 0),
    scale: float = 0.55,
    thickness: int = 1,
) -> np.ndarray:
    out = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = xy
    y = max(y, th + 4)
    cv2.rectangle(out, (x, y - th - baseline - 4), (x + tw + 6, y + 2), bg, -1)
    cv2.putText(out, text, (x + 3, y - 2), font, scale, color, thickness, cv2.LINE_AA)
    return out


def _show(ax, img: np.ndarray, title: str | None = None) -> None:
    ax.imshow(img)
    ax.axis("off")
    if title:
        ax.set_title(title)


# ---------- 1) face detection ----------

def make_face_detection_overlay(
    frame_path: str | Path,
    detector,
    draw_landmarks: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    img = _read_rgb(frame_path)
    _, det_df, _ = detect_faces_yunet(Path(frame_path), detector)
    out = img.copy()

    for i, row in det_df.reset_index(drop=True).iterrows():
        out = _draw_box(out, row["x"], row["y"], row["w"], row["h"], color=(0, 255, 0), thickness=2)
        if draw_landmarks:
            out = _draw_landmarks(out, row)
        label = f"#{i} score={row['score']:.2f}"
        out = _put_label(out, label, (int(row["x"]), max(20, int(row["y"]) - 6)), bg=(0, 100, 0))

    out = _put_label(out, f"faces={len(det_df)}", (10, 24), bg=(20, 20, 20))
    return out, det_df


def show_face_detection(
    frame_path: str | Path,
    detector,
    figsize: tuple[int, int] = (8, 6),
) -> pd.DataFrame:
    overlay, det_df = make_face_detection_overlay(frame_path, detector)
    fig, ax = plt.subplots(figsize=figsize)
    _show(ax, overlay, Path(frame_path).name)
    plt.show()
    return det_df


def show_shot_face_detection_triplet(
    shot_dir: str | Path,
    detector,
    frame_names: Iterable[str] = ("frame_1.jpg", "frame_2.jpg", "frame_3.jpg"),
    figsize: tuple[int, int] = (18, 6),
) -> list[pd.DataFrame]:
    shot_dir = Path(shot_dir)
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    det_dfs: list[pd.DataFrame] = []

    for ax, frame_name in zip(axes, frame_names):
        frame_path = shot_dir / frame_name
        overlay, det_df = make_face_detection_overlay(frame_path, detector)
        det_dfs.append(det_df)
        mean_score = float(det_df["score"].mean()) if len(det_df) else float("nan")
        title = f"{frame_name} | n={len(det_df)} | mean={mean_score:.2f}" if len(det_df) else f"{frame_name} | n=0"
        _show(ax, overlay, title)

    plt.tight_layout()
    plt.show()
    return det_dfs


# ---------- 2) main subject ----------

def make_main_subject_overlay(
    frame_path: str | Path,
    detector,
    draw_landmarks: bool = True,
) -> tuple[np.ndarray, pd.DataFrame, pd.Series | None]:
    img = _read_rgb(frame_path)
    _, det_df, _ = detect_faces_yunet(Path(frame_path), detector)
    out = img.copy()

    if len(det_df) == 0:
        out = _put_label(out, "No face detected", (10, 24), bg=(120, 20, 20))
        return out, det_df, None

    best_row = select_main_subject_from_det_df(det_df, img.shape)
    best_idx = int(best_row.name) if best_row is not None else None

    for i, row in det_df.iterrows():
        is_best = i == best_idx
        color = (255, 0, 0) if is_best else (0, 255, 0)
        thick = 4 if is_best else 2
        out = _draw_box(out, row["x"], row["y"], row["w"], row["h"], color=color, thickness=thick)
        if draw_landmarks:
            out = _draw_landmarks(out, row)

        if is_best:
            label = (
                f"MAIN total={best_row['main_subject_score']:.3f} | "
                f"area={best_row['area_score']:.3f} | center={best_row['center_score']:.3f} | det={best_row['det_score_used']:.3f}"
            )
            out = _put_label(out, label, (int(row["x"]), max(20, int(row["y"]) - 8)), bg=(120, 20, 20))
        else:
            out = _put_label(out, f"cand#{i} score={row['score']:.2f}", (int(row["x"]), max(20, int(row["y"]) - 6)), bg=(0, 90, 0))

    return out, det_df, best_row


def show_main_subject(
    frame_path: str | Path,
    detector,
    figsize: tuple[int, int] = (8, 6),
) -> tuple[pd.DataFrame, pd.Series | None]:
    overlay, det_df, best_row = make_main_subject_overlay(frame_path, detector)
    fig, ax = plt.subplots(figsize=figsize)
    _show(ax, overlay, Path(frame_path).name)
    plt.show()
    return det_df, best_row


def show_shot_main_subject_triplet(
    shot_dir: str | Path,
    detector,
    frame_names: Iterable[str] = ("frame_1.jpg", "frame_2.jpg", "frame_3.jpg"),
    figsize: tuple[int, int] = (18, 6),
) -> list[pd.Series | None]:
    shot_dir = Path(shot_dir)
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    best_rows: list[pd.Series | None] = []

    for ax, frame_name in zip(axes, frame_names):
        frame_path = shot_dir / frame_name
        overlay, _, best_row = make_main_subject_overlay(frame_path, detector)
        best_rows.append(best_row)
        title = frame_name if best_row is None else f"{frame_name} | main={best_row['main_subject_score']:.3f}"
        _show(ax, overlay, title)

    plt.tight_layout()
    plt.show()
    return best_rows


# ---------- 3) head direction ----------

def _frame_landmark_series_from_row(row: pd.Series, frame_idx: int) -> pd.Series:
    prefix = f"f{frame_idx}_"
    return pd.Series(
        {
            "left_eye_x": row.get(prefix + "left_eye_x", np.nan),
            "left_eye_y": row.get(prefix + "left_eye_y", np.nan),
            "right_eye_x": row.get(prefix + "right_eye_x", np.nan),
            "right_eye_y": row.get(prefix + "right_eye_y", np.nan),
            "nose_x": row.get(prefix + "nose_x", np.nan),
            "nose_y": row.get(prefix + "nose_y", np.nan),
            "x": row.get(prefix + "main_x", np.nan),
            "y": row.get(prefix + "main_y", np.nan),
            "w": row.get(prefix + "main_w", np.nan),
            "h": row.get(prefix + "main_h", np.nan),
        }
    )


def make_head_direction_overlay(
    shot_row: pd.Series,
    frame_idx: int,
    thresh: float = 0.10,
) -> tuple[np.ndarray, dict]:
    frame_path = shot_row[f"f{frame_idx}_frame_path"]
    img = _read_rgb(frame_path)
    out = img.copy()
    lm = _frame_landmark_series_from_row(shot_row, frame_idx)

    if pd.isna(lm["x"]):
        out = _put_label(out, "No main subject for this frame", (10, 24), bg=(120, 20, 20))
        return out, {"head_dir": "unclear", "nose_offset_norm": np.nan}

    out = _draw_box(out, lm["x"], lm["y"], lm["w"], lm["h"], color=(255, 0, 0), thickness=3)
    out = _draw_landmarks(out, lm)

    head_dir, offset = estimate_head_dir_from_landmarks(
        lm["left_eye_x"],
        lm["left_eye_y"],
        lm["right_eye_x"],
        lm["right_eye_y"],
        lm["nose_x"],
        lm["nose_y"],
        thresh=thresh,
    )

    if pd.notna(lm["left_eye_x"]) and pd.notna(lm["right_eye_x"]) and pd.notna(lm["nose_x"]):
        eye_mid = (
            int(round((lm["left_eye_x"] + lm["right_eye_x"]) / 2.0)),
            int(round((lm["left_eye_y"] + lm["right_eye_y"]) / 2.0)),
        )
        nose = (int(round(lm["nose_x"])), int(round(lm["nose_y"])))
        cv2.arrowedLine(out, eye_mid, nose, (255, 128, 0), 3, tipLength=0.18)
        cv2.circle(out, eye_mid, 4, (255, 0, 255), -1)

    label = f"dir={head_dir} | offset={offset:.3f}" if pd.notna(offset) else f"dir={head_dir}"
    out = _put_label(out, label, (10, 24), bg=(0, 60, 120))

    return out, {"head_dir": head_dir, "nose_offset_norm": offset}


def show_head_direction_triplet(
    shot_row: pd.Series,
    thresh: float = 0.10,
    figsize: tuple[int, int] = (18, 6),
) -> list[dict]:
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    infos: list[dict] = []

    for ax, frame_idx in zip(axes, [1, 2, 3]):
        overlay, info = make_head_direction_overlay(shot_row, frame_idx=frame_idx, thresh=thresh)
        infos.append(info)
        _show(ax, overlay, f"frame_{frame_idx} | {info['head_dir']}")

    plt.tight_layout()
    plt.show()
    return infos


# ---------- 4) final proxy candidate card ----------

def _safe_text(v) -> str:
    if pd.isna(v):
        return "NA"
    return str(v)


def show_proxy_candidate_card(
    shot_row: pd.Series,
    thresh: float = 0.10,
    figsize: tuple[int, int] = (18, 10),
) -> pd.Series:
    enriched = shot_row.copy()
    # compute labels on the fly if they are not there yet
    need_cols = {"interaction_type", "visual_target_coarse", "duration_bin"}
    if not need_cols.issubset(set(enriched.index)):
        enriched = add_proxy_label_columns(pd.DataFrame([enriched])).iloc[0]

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, height_ratios=[2.2, 1.0])

    for col_idx, frame_idx in enumerate([1, 2, 3]):
        ax = fig.add_subplot(gs[0, col_idx])
        overlay, info = make_head_direction_overlay(enriched, frame_idx=frame_idx, thresh=thresh)
        _show(ax, overlay, f"frame_{frame_idx} | {info['head_dir']}")

    ax_text = fig.add_subplot(gs[1, :])
    ax_text.axis("off")

    lines = [
        f"shot_id: {_safe_text(enriched.get('shot_id'))}",
        f"subtitle: {_safe_text(enriched.get('subtitle_text'))}",
        f"consensus_count: {_safe_text(enriched.get('consensus_count'))}",
        f"mean_det_score: {_safe_text(enriched.get('mean_det_score'))}",
        f"head_dir_consensus: {_safe_text(enriched.get('head_dir_consensus'))}",
        f"head_dir_stable: {_safe_text(enriched.get('head_dir_stable'))}",
        f"shot_duration: {_safe_text(enriched.get('shot_duration'))}",
        f"interaction_type: {_safe_text(enriched.get('interaction_type'))}",
        f"visual_target_coarse: {_safe_text(enriched.get('visual_target_coarse'))}",
        f"duration_bin: {_safe_text(enriched.get('duration_bin'))}",
    ]
    if "proxy_token" in enriched.index:
        lines.append(f"proxy_token: {_safe_text(enriched.get('proxy_token'))}")

    ax_text.text(0.01, 0.98, "\n".join(lines), va="top", ha="left", fontsize=11, family="monospace")
    plt.tight_layout()
    plt.show()
    return enriched


# ---------- quick dataframe summaries ----------

def show_label_distributions(df: pd.DataFrame, figsize: tuple[int, int] = (18, 4)) -> None:
    cols = [c for c in ["interaction_type", "visual_target_coarse", "duration_bin"] if c in df.columns]
    if not cols:
        raise ValueError("DataFrame does not contain proxy label columns.")

    fig, axes = plt.subplots(1, len(cols), figsize=figsize)
    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        vc = df[col].fillna("NA").value_counts(dropna=False)
        ax.bar(vc.index.astype(str), vc.values)
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.show()
