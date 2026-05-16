#!/usr/bin/env python3
"""Compare OpenFace gaze/head-pose evidence with optional L2CS and 6DRepNet outputs."""

from __future__ import annotations

import argparse
import csv
import html
import importlib.util
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


MOVIES = ["tt0032138", "tt1591095", "tt1637725"]
OUTPUT_COLUMNS = [
    "movie_id",
    "sample_bucket",
    "sample_subtype",
    "sequence_id",
    "shot_id",
    "bin_idx",
    "subject_track_id",
    "crop_video_path",
    "overlay_path",
    "openface_vis_path",
    "l2cs_vis_path",
    "sixd_vis_path",
    "human_label",
    "human_notes",
    "openface_gaze_x",
    "openface_gaze_y",
    "openface_pose_Ry",
    "openface_pose_Rx",
    "openface_gaze_direction",
    "openface_pose_direction",
    "openface_conflict",
    "l2cs_yaw",
    "l2cs_pitch",
    "l2cs_direction",
    "sixd_yaw",
    "sixd_pitch",
    "sixd_roll",
    "sixd_pose_direction",
    "l2cs_vs_openface_gaze_agree",
    "sixd_vs_openface_pose_agree",
    "inference_status",
    "error_note",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed) or math.isinf(parsed):
        return default
    return parsed


def direction_bucket(value: Any, threshold: float) -> str:
    parsed = safe_float(value)
    if math.isnan(parsed):
        return "unknown"
    if parsed < -threshold:
        return "left"
    if parsed > threshold:
        return "right"
    return "center"


def key4(row: dict[str, str], track_col: str) -> tuple[str, str, str, str]:
    return (
        row.get("movie_id", ""),
        row.get("shot_id", ""),
        row.get("bin_idx", ""),
        row.get(track_col, ""),
    )


def sample_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row.get("movie_id", ""),
        row.get("shot_id", ""),
        row.get("bin_idx", ""),
        row.get("subject_local_track_id", ""),
        row.get("sample_index", ""),
    )


def gaze_pose_conflict(gaze_direction: str, pose_direction: str) -> str:
    if gaze_direction in {"left", "right"} and pose_direction in {"left", "right"} and gaze_direction != pose_direction:
        return "1"
    return "0"


def note_priority(row: dict[str, str]) -> int:
    note = (row.get("human_notes") or "").lower()
    score = 0
    if "gaze/head" in note or "direction" in note:
        score += 8
    if "x=left" in note and "pose=right" in note:
        score += 8
    if row.get("gaze_direction_bucket") in {"left", "right"} and row.get("pose_direction_bucket") in {"left", "right"} and row.get("gaze_direction_bucket") != row.get("pose_direction_bucket"):
        score += 5
    if row.get("gaze_quality") == "gaze_reliable":
        score += 2
    if row.get("gaze_quality") == "pose_fallback":
        score += 1
    return score


def load_reviewed(audit_dir: Path) -> dict[tuple[str, str, str, str], dict[str, str]]:
    path = audit_dir / "audit_labels_reviewed.csv"
    rows = [row for row in read_csv(path) if (row.get("human_label") or "").strip()]
    return {key4(row, "subject_local_track_id"): row for row in rows}


def enrich_samples_with_reviewed(samples: list[dict[str, str]], reviewed: dict[tuple[str, str, str, str], dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for sample in samples:
        row = dict(sample)
        review = reviewed.get(key4(sample, "subject_local_track_id"), {})
        row["human_label"] = review.get("human_label", "")
        row["human_notes"] = review.get("human_notes", "")
        out.append(row)
    return out


def take_top(rows: list[dict[str, str]], limit: int, used: set[tuple[str, str, str, str]]) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for row in sorted(rows, key=note_priority, reverse=True):
        key = sample_key(row)
        if key in used:
            continue
        used.add(key)
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def select_movie_samples(audit_dir: Path, movie_id: str) -> tuple[list[dict[str, str]], dict[str, int]]:
    samples = enrich_samples_with_reviewed(read_csv(audit_dir / "audit_samples.csv"), load_reviewed(audit_dir))
    used: set[tuple[str, str, str, str]] = set()
    shortfall: dict[str, int] = {}
    selected: list[dict[str, str]] = []

    assigned = [row for row in samples if row.get("sample_bucket") == "assigned"]
    wrong_target = [row for row in samples if row.get("human_label") == "wrong_target"]
    reserved_wrong_keys = {sample_key(row) for row in wrong_target}
    assigned_primary = [row for row in assigned if sample_key(row) not in reserved_wrong_keys]
    picked = take_top(assigned_primary, 10, used)
    selected.extend(dict(row, sample_subtype="assigned") for row in picked)
    shortfall["assigned"] = max(0, 10 - len(picked))

    picked = take_top(wrong_target, 5, used)
    if len(picked) < 5:
        fallback = [
            row
            for row in assigned
            if (
                row.get("gaze_direction_bucket") != row.get("pose_direction_bucket")
                or row.get("gaze_direction_bucket") in {"left", "right"}
                or row.get("pose_direction_bucket") in {"left", "right"}
            )
        ]
        extra = take_top(fallback, 5 - len(picked), used)
        selected.extend(dict(row, sample_subtype="wrong_target") for row in picked)
        selected.extend(dict(row, sample_subtype="assigned_high_risk") for row in extra)
        shortfall["wrong_target"] = max(0, 5 - len(picked) - len(extra))
    else:
        selected.extend(dict(row, sample_subtype="wrong_target") for row in picked)
        shortfall["wrong_target"] = 0

    low_score = [row for row in samples if row.get("sample_bucket") == "low_score"]
    picked = take_top(low_score, 5, used)
    selected.extend(dict(row, sample_subtype="low_score") for row in picked)
    shortfall["low_score"] = max(0, 5 - len(picked))

    low_margin = [row for row in samples if row.get("sample_bucket") in {"low_margin", "ambiguous"}]
    picked = take_top(low_margin, 5, used)
    selected.extend(dict(row, sample_subtype="low_margin_ambiguous") for row in picked)
    shortfall["low_margin_ambiguous"] = max(0, 5 - len(picked))

    for row in selected:
        row["movie_id"] = movie_id
    return selected, shortfall


def import_status(module_names: list[str]) -> tuple[dict[str, bool], dict[str, str]]:
    status: dict[str, bool] = {}
    errors: dict[str, str] = {}
    for name in module_names:
        if importlib.util.find_spec(name) is None:
            status[name] = False
            errors[name] = "module not found"
            continue
        try:
            __import__(name)
        except Exception as exc:  # pragma: no cover - environment dependent
            status[name] = False
            errors[name] = f"{type(exc).__name__}: {exc}"
        else:
            status[name] = True
    return status, errors


def dependency_status() -> tuple[dict[str, bool], list[str], dict[str, str]]:
    status, errors = import_status(["torch", "torchvision", "cv2", "numpy", "PIL"])
    missing = [name for name, ok in status.items() if not ok]
    return status, missing, errors


def resolve_device(device_name: str) -> Any:
    import torch

    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


class L2CSRunner:
    def __init__(self, repo: Path, weights: Path, device_name: str) -> None:
        if not repo.exists():
            raise FileNotFoundError(f"L2CS repo not found: {repo}")
        if not weights.exists():
            raise FileNotFoundError(f"L2CS weights not found: {weights}")
        sys.path.insert(0, str(repo.resolve()))
        import torch
        from l2cs.utils import getArch, prep_input_numpy

        self.torch = torch
        self.prep_input_numpy = prep_input_numpy
        self.device = resolve_device(device_name)
        self.model = getArch("ResNet50", 90)
        state = torch.load(str(weights), map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        self.softmax = torch.nn.Softmax(dim=1)
        self.idx_tensor = torch.arange(90, dtype=torch.float32, device=self.device)

    def predict(self, frame_bgr: Any) -> dict[str, str]:
        import cv2

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self.prep_input_numpy(frame_rgb, self.device)
        with self.torch.no_grad():
            pitch_logits, yaw_logits = self.model(image)
            pitch_pred = self.softmax(pitch_logits)
            yaw_pred = self.softmax(yaw_logits)
            pitch_deg = (self.torch.sum(pitch_pred * self.idx_tensor, dim=1) * 4 - 180).item()
            yaw_deg = (self.torch.sum(yaw_pred * self.idx_tensor, dim=1) * 4 - 180).item()
        pitch = math.radians(float(pitch_deg))
        yaw = math.radians(float(yaw_deg))
        return {
            "l2cs_yaw": f"{yaw:.6f}",
            "l2cs_pitch": f"{pitch:.6f}",
            "l2cs_direction": direction_bucket(yaw, 0.20),
        }


class SixDRepNetRunner:
    def __init__(self, repo: Path, weights: Path, device_name: str) -> None:
        if not repo.exists():
            raise FileNotFoundError(f"6DRepNet360 repo not found: {repo}")
        if not weights.exists():
            raise FileNotFoundError(f"6DRepNet360 weights not found: {weights}")
        sys.path.insert(0, str((repo / "sixdrepnet360").resolve()))
        import torch
        import torchvision
        from PIL import Image
        from test import SixDRepNet360
        from torchvision import transforms
        import utils as sixd_utils

        self.torch = torch
        self.Image = Image
        self.sixd_utils = sixd_utils
        self.device = resolve_device(device_name)
        self.model = SixDRepNet360(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 6)
        state = torch.load(str(weights), map_location=self.device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def predict(self, frame_bgr: Any) -> dict[str, str]:
        import cv2

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self.Image.fromarray(frame_rgb)
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            rotation = self.model(tensor)
            euler = self.sixd_utils.compute_euler_angles_from_rotation_matrices(rotation)
        pitch = float(euler[0, 0].item())
        yaw = float(euler[0, 1].item())
        roll = float(euler[0, 2].item())
        return {
            "sixd_yaw": f"{yaw:.6f}",
            "sixd_pitch": f"{pitch:.6f}",
            "sixd_roll": f"{roll:.6f}",
            "sixd_pose_direction": direction_bucket(yaw, 0.25),
        }


def read_crop_frame(crop_video_path: str, bin_start: Any, bin_end: Any) -> tuple[Any | None, str]:
    if not crop_video_path:
        return None, "missing crop_video_path"
    path = Path(crop_video_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        return None, f"crop video not found: {path}"
    try:
        import cv2
    except Exception as exc:  # pragma: no cover - environment dependent
        return None, f"cv2 unavailable: {type(exc).__name__}: {exc}"
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None, f"could not open crop video: {path}"
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        start = safe_float(bin_start, 0.0)
        end = safe_float(bin_end, start)
        midpoint = max(0.0, (start + end) / 2.0)
        if fps > 0 and frame_count > 0:
            frame_idx = min(max(int(round(midpoint * fps)), 0), frame_count - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if ok:
            return frame, ""
        if frame_count > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if ok:
            return frame, "used first frame fallback"
        return None, f"could not read frame from crop video: {path}"
    finally:
        cap.release()


def draw_text_lines(image: Any, lines: list[str]) -> None:
    import cv2

    x, y = 8, 20
    for line in lines:
        cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
        y += 18


def draw_gaze_arrow(image: Any, pitch: Any, yaw: Any, color: tuple[int, int, int]) -> bool:
    import cv2
    import numpy as np

    pitch_value = safe_float(pitch)
    yaw_value = safe_float(yaw)
    if math.isnan(pitch_value) or math.isnan(yaw_value):
        return False
    height, width = image.shape[:2]
    length = max(36, min(width, height) * 0.42)
    start = np.array([width / 2.0, height / 2.0])
    dx = -length * math.sin(pitch_value) * math.cos(yaw_value)
    dy = -length * math.sin(yaw_value)
    end = np.array([start[0] + dx, start[1] + dy])
    cv2.circle(image, tuple(np.round(start).astype(int)), 4, color, -1, cv2.LINE_AA)
    cv2.arrowedLine(
        image,
        tuple(np.round(start).astype(int)),
        tuple(np.round(end).astype(int)),
        color,
        3,
        cv2.LINE_AA,
        tipLength=0.22,
    )
    return True


def draw_pose_cube_degrees(
    image: Any,
    yaw_degrees: Any,
    pitch_degrees: Any,
    roll_degrees: Any,
    size_scale: float = 0.58,
) -> bool:
    import cv2

    yaw = safe_float(yaw_degrees)
    pitch = safe_float(pitch_degrees)
    roll = safe_float(roll_degrees)
    if math.isnan(yaw) or math.isnan(pitch) or math.isnan(roll):
        return False
    height, width = image.shape[:2]
    size = max(48.0, min(width, height) * size_scale)
    face_x = width / 2.0 - 0.5 * size
    face_y = height / 2.0 - 0.5 * size
    p = math.radians(pitch)
    y = -math.radians(yaw)
    r = math.radians(roll)

    x1 = size * (math.cos(y) * math.cos(r)) + face_x
    y1 = size * (math.cos(p) * math.sin(r) + math.cos(r) * math.sin(p) * math.sin(y)) + face_y
    x2 = size * (-math.cos(y) * math.sin(r)) + face_x
    y2 = size * (math.cos(p) * math.cos(r) - math.sin(p) * math.sin(y) * math.sin(r)) + face_y
    x3 = size * math.sin(y) + face_x
    y3 = size * (-math.cos(y) * math.sin(p)) + face_y

    def point(x: float, y_: float) -> tuple[int, int]:
        return int(round(x)), int(round(y_))

    cv2.line(image, point(face_x, face_y), point(x1, y1), (0, 0, 255), 3, cv2.LINE_AA)
    cv2.line(image, point(face_x, face_y), point(x2, y2), (0, 0, 255), 3, cv2.LINE_AA)
    cv2.line(image, point(x2, y2), point(x2 + x1 - face_x, y2 + y1 - face_y), (0, 0, 255), 3, cv2.LINE_AA)
    cv2.line(image, point(x1, y1), point(x1 + x2 - face_x, y1 + y2 - face_y), (0, 0, 255), 3, cv2.LINE_AA)
    cv2.line(image, point(face_x, face_y), point(x3, y3), (255, 0, 0), 2, cv2.LINE_AA)
    cv2.line(image, point(x1, y1), point(x1 + x3 - face_x, y1 + y3 - face_y), (255, 0, 0), 2, cv2.LINE_AA)
    cv2.line(image, point(x2, y2), point(x2 + x3 - face_x, y2 + y3 - face_y), (255, 0, 0), 2, cv2.LINE_AA)
    cv2.line(
        image,
        point(x2 + x1 - face_x, y2 + y1 - face_y),
        point(x3 + x1 + x2 - 2 * face_x, y3 + y2 + y1 - 2 * face_y),
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.line(
        image,
        point(x3 + x1 - face_x, y3 + y1 - face_y),
        point(x3 + x1 + x2 - 2 * face_x, y3 + y2 + y1 - 2 * face_y),
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.line(
        image,
        point(x2 + x3 - face_x, y2 + y3 - face_y),
        point(x3 + x1 + x2 - 2 * face_x, y3 + y2 + y1 - 2 * face_y),
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.line(image, point(x3, y3), point(x3 + x1 - face_x, y3 + y1 - face_y), (0, 255, 0), 2, cv2.LINE_AA)
    cv2.line(image, point(x3, y3), point(x3 + x2 - face_x, y3 + y2 - face_y), (0, 255, 0), 2, cv2.LINE_AA)
    return True


def rad_to_deg(value: Any) -> float:
    parsed = safe_float(value)
    if math.isnan(parsed):
        return float("nan")
    return math.degrees(parsed)


def visualization_basename(row_index: int, row: dict[str, Any]) -> str:
    parts = [
        f"{row_index:04d}",
        str(row.get("movie_id", "")),
        str(row.get("shot_id", "")),
        str(row.get("subject_track_id", "")),
        f"bin{row.get('bin_idx', '')}",
    ]
    return "_".join(part.replace("/", "_").replace(" ", "_") for part in parts if part)


def save_visualizations(row: dict[str, Any], frame: Any, output_dir: Path, row_index: int) -> list[str]:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover - environment dependent
        return [f"visualization cv2 unavailable: {type(exc).__name__}: {exc}"]

    errors: list[str] = []
    vis_dir = output_dir / "frames_visualized"
    vis_dir.mkdir(parents=True, exist_ok=True)
    basename = visualization_basename(row_index, row)

    def write_image(kind: str, image: Any) -> str:
        path = vis_dir / f"{basename}_{kind}.png"
        ok = cv2.imwrite(str(path), image)
        if not ok:
            raise OSError(f"cv2.imwrite failed for {path}")
        return str(path)

    try:
        openface_image = frame.copy()
        draw_gaze_arrow(openface_image, row.get("openface_gaze_x"), row.get("openface_gaze_y"), (255, 255, 0))
        draw_pose_cube_degrees(
            openface_image,
            rad_to_deg(row.get("openface_pose_Ry")),
            rad_to_deg(row.get("openface_pose_Rx")),
            rad_to_deg(row.get("openface_pose_Rz")),
            size_scale=0.48,
        )
        draw_text_lines(
            openface_image,
            [
                "OpenFace bin mean",
                f"gaze={row.get('openface_gaze_direction')} pose={row.get('openface_pose_direction')} conflict={row.get('openface_conflict')}",
            ],
        )
        row["openface_vis_path"] = write_image("openface", openface_image)
    except Exception as exc:  # pragma: no cover - image dependent
        errors.append(f"openface visualization: {type(exc).__name__}: {exc}")

    if row.get("l2cs_yaw") and row.get("l2cs_pitch"):
        try:
            l2cs_image = frame.copy()
            draw_gaze_arrow(l2cs_image, row.get("l2cs_pitch"), row.get("l2cs_yaw"), (255, 255, 0))
            draw_text_lines(l2cs_image, ["L2CS-Net", f"gaze={row.get('l2cs_direction')}"])
            row["l2cs_vis_path"] = write_image("l2cs", l2cs_image)
        except Exception as exc:  # pragma: no cover - image dependent
            errors.append(f"l2cs visualization: {type(exc).__name__}: {exc}")

    if row.get("sixd_yaw") and row.get("sixd_pitch") and row.get("sixd_roll"):
        try:
            sixd_image = frame.copy()
            draw_pose_cube_degrees(
                sixd_image,
                rad_to_deg(row.get("sixd_yaw")),
                rad_to_deg(row.get("sixd_pitch")),
                rad_to_deg(row.get("sixd_roll")),
                size_scale=0.58,
            )
            draw_text_lines(sixd_image, ["6DRepNet360", f"pose={row.get('sixd_pose_direction')}"])
            row["sixd_vis_path"] = write_image("sixdrepnet", sixd_image)
        except Exception as exc:  # pragma: no cover - image dependent
            errors.append(f"6drepnet visualization: {type(exc).__name__}: {exc}")

    return errors


def load_optional_runners(
    l2cs_repo: Path,
    l2cs_weights: Path,
    sixd_repo: Path,
    sixd_weights: Path,
    device: str,
) -> tuple[L2CSRunner | None, SixDRepNetRunner | None, dict[str, str]]:
    errors: dict[str, str] = {}
    l2cs_runner: L2CSRunner | None = None
    sixd_runner: SixDRepNetRunner | None = None
    try:
        l2cs_runner = L2CSRunner(l2cs_repo, l2cs_weights, device)
    except Exception as exc:  # pragma: no cover - environment/model dependent
        errors["l2cs"] = f"{type(exc).__name__}: {exc}"
    try:
        sixd_runner = SixDRepNetRunner(sixd_repo, sixd_weights, device)
    except Exception as exc:  # pragma: no cover - environment/model dependent
        errors["sixdrepnet"] = f"{type(exc).__name__}: {exc}"
    return l2cs_runner, sixd_runner, errors


def run_l2cs_placeholder() -> dict[str, str]:
    return {"l2cs_yaw": "", "l2cs_pitch": "", "l2cs_direction": ""}


def run_sixd_placeholder() -> dict[str, str]:
    return {"sixd_yaw": "", "sixd_pitch": "", "sixd_roll": "", "sixd_pose_direction": ""}


def compare_agreement(model_direction: str, openface_direction: str) -> str:
    if model_direction in {"", "unknown"} or openface_direction in {"", "unknown"}:
        return ""
    return "1" if model_direction == openface_direction else "0"


def build_compare_rows(
    base_dir: Path,
    output_dir: Path,
    movies: list[str],
    l2cs_repo: Path | None = None,
    l2cs_weights: Path | None = None,
    sixd_repo: Path | None = None,
    sixd_weights: Path | None = None,
    device: str = "cpu",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    deps, missing, dependency_errors = dependency_status()
    l2cs_repo = l2cs_repo or Path("models/gaze_estimation/L2CS-Net")
    l2cs_weights = l2cs_weights or Path("models/gaze_estimation/L2CS-Net/models/L2CSNet_gaze360.pkl")
    sixd_repo = sixd_repo or Path("models/head_pose/6DRepNet360")
    sixd_weights = sixd_weights or Path("models/head_pose/6DRepNet360/checkpoints/6DRepNet360_Full-Rotation_300W_LP_Panoptic.pth")
    model_errors: dict[str, str] = {}
    l2cs_runner: L2CSRunner | None = None
    sixd_runner: SixDRepNetRunner | None = None
    if not missing:
        l2cs_runner, sixd_runner, model_errors = load_optional_runners(
            l2cs_repo, l2cs_weights, sixd_repo, sixd_weights, device
        )
    all_rows: list[dict[str, Any]] = []
    shortfalls: dict[str, dict[str, int]] = {}
    for movie_id in movies:
        audit_dir = base_dir / "debug_audits" / "v0_2" / movie_id
        selected, shortfall = select_movie_samples(audit_dir, movie_id)
        shortfalls[movie_id] = shortfall
        openface_bins = {key4(row, "local_track_id"): row for row in read_csv(base_dir / movie_id / "openface" / "06_gaze_timebins.csv")}
        for sample in selected:
            key = key4(sample, "subject_local_track_id")
            openface = openface_bins.get(key, {})
            gaze_x = openface.get("gaze_angle_x_mean") or sample.get("gaze_angle_x_mean", "")
            gaze_y = openface.get("gaze_angle_y_mean") or sample.get("gaze_angle_y_mean", "")
            pose_ry = openface.get("pose_Ry_mean", "")
            pose_rx = openface.get("pose_Rx_mean", "")
            pose_rz = openface.get("pose_Rz_mean", "")
            openface_gaze = direction_bucket(gaze_x, 0.20)
            openface_pose = direction_bucket(pose_ry, 0.25)
            row: dict[str, Any] = {
                "movie_id": movie_id,
                "sample_bucket": sample.get("sample_bucket", ""),
                "sample_subtype": sample.get("sample_subtype", ""),
                "sequence_id": sample.get("sequence_id", ""),
                "shot_id": sample.get("shot_id", ""),
                "bin_idx": sample.get("bin_idx", ""),
                "subject_track_id": sample.get("subject_local_track_id", ""),
                "crop_video_path": openface.get("crop_video_path", ""),
                "overlay_path": sample.get("overlay_path", ""),
                "openface_vis_path": "",
                "l2cs_vis_path": "",
                "sixd_vis_path": "",
                "human_label": sample.get("human_label", ""),
                "human_notes": sample.get("human_notes", ""),
                "openface_gaze_x": gaze_x,
                "openface_gaze_y": gaze_y,
                "openface_pose_Ry": pose_ry,
                "openface_pose_Rx": pose_rx,
                "openface_pose_Rz": pose_rz,
                "openface_gaze_direction": openface_gaze,
                "openface_pose_direction": openface_pose,
                "openface_conflict": gaze_pose_conflict(openface_gaze, openface_pose),
                "l2cs_vs_openface_gaze_agree": "",
                "sixd_vs_openface_pose_agree": "",
                "inference_status": "",
                "error_note": "",
            }
            row.update(run_l2cs_placeholder())
            row.update(run_sixd_placeholder())
            errors: list[str] = []
            if missing:
                errors.extend(f"{name}: {dependency_errors.get(name, 'missing')}" for name in missing)
                row["inference_status"] = "baseline_only"
            else:
                frame, frame_error = read_crop_frame(
                    str(row.get("crop_video_path", "")),
                    openface.get("bin_start_sec") or sample.get("bin_start_sec"),
                    openface.get("bin_end_sec") or sample.get("bin_end_sec"),
                )
                if frame_error:
                    errors.append(frame_error)
                if frame is None:
                    row["inference_status"] = "frame_unavailable"
                else:
                    model_ok = 0
                    if l2cs_runner is not None:
                        try:
                            row.update(l2cs_runner.predict(frame))
                            model_ok += 1
                        except Exception as exc:  # pragma: no cover - model dependent
                            errors.append(f"l2cs inference: {type(exc).__name__}: {exc}")
                    if sixd_runner is not None:
                        try:
                            row.update(sixd_runner.predict(frame))
                            model_ok += 1
                        except Exception as exc:  # pragma: no cover - model dependent
                            errors.append(f"6drepnet inference: {type(exc).__name__}: {exc}")
                    if model_ok == 2:
                        row["inference_status"] = "model_inference_ok"
                    elif model_ok == 1:
                        row["inference_status"] = "partial_model_error"
                    else:
                        row["inference_status"] = "baseline_only"
                    errors.extend(save_visualizations(row, frame, output_dir, len(all_rows)))
                errors.extend(f"{name}: {message}" for name, message in model_errors.items())
            row["l2cs_vs_openface_gaze_agree"] = compare_agreement(str(row.get("l2cs_direction", "")), openface_gaze)
            row["sixd_vs_openface_pose_agree"] = compare_agreement(str(row.get("sixd_pose_direction", "")), openface_pose)
            row["error_note"] = "; ".join(dict.fromkeys(error for error in errors if error))
            all_rows.append(row)
    summary = {
        "row_count": len(all_rows),
        "movies": movies,
        "dependency_status": deps,
        "dependency_errors": dependency_errors,
        "model_errors": model_errors,
        "l2cs_available": l2cs_runner is not None,
        "sixdrepnet_available": sixd_runner is not None,
        "inference_status_counts": dict(Counter(row["inference_status"] for row in all_rows)),
        "visualization_counts": {
            "openface": sum(1 for row in all_rows if row.get("openface_vis_path")),
            "l2cs": sum(1 for row in all_rows if row.get("l2cs_vis_path")),
            "sixdrepnet": sum(1 for row in all_rows if row.get("sixd_vis_path")),
        },
        "sample_shortfall": shortfalls,
        "sample_subtype_counts": dict(Counter(row["sample_subtype"] for row in all_rows)),
        "output_csv": str(output_dir / "gaze_evidence_compare.csv"),
        "output_html": str(output_dir / "gaze_evidence_compare.html"),
    }
    return all_rows, summary


def fmt(value: Any) -> str:
    return html.escape(str(value or ""))


def relpath_for_html(path_value: str, report_path: Path) -> str:
    if not path_value:
        return ""
    path = Path(path_value)
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        return path.resolve().relative_to(report_path.parent.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_uri()


def render_html(rows: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> None:
    report_path = output_dir / "gaze_evidence_compare.html"
    cards: list[str] = []
    for idx, row in enumerate(rows):
        def figure(path_key: str, title: str) -> str:
            image_rel = relpath_for_html(str(row.get(path_key, "")), report_path)
            image_html = f'<img src="{fmt(image_rel)}" alt="{fmt(title)}">' if image_rel else '<div class="placeholder">Missing</div>'
            return f"<figure><figcaption>{fmt(title)}</figcaption>{image_html}</figure>"

        figures = "".join(
            [
                figure("overlay_path", "Debug overlay"),
                figure("openface_vis_path", "OpenFace bin mean"),
                figure("l2cs_vis_path", "L2CS gaze"),
                figure("sixd_vis_path", "6DRepNet pose"),
            ]
        )
        classes = ["card"]
        if row.get("openface_conflict") == "1":
            classes.append("conflict")
        if row.get("human_label") == "bad_frame_or_track":
            classes.append("badtrack")
        details = [
            ("movie", row.get("movie_id")),
            ("bucket", f"{row.get('sample_bucket')} / {row.get('sample_subtype')}"),
            ("shot/bin", f"{row.get('shot_id')} / {row.get('bin_idx')}"),
            ("subject", row.get("subject_track_id")),
            ("human", f"{row.get('human_label')} {row.get('human_notes')}"),
            ("OpenFace gaze", f"{row.get('openface_gaze_direction')} x={row.get('openface_gaze_x')} y={row.get('openface_gaze_y')}"),
            ("OpenFace pose", f"{row.get('openface_pose_direction')} Ry={row.get('openface_pose_Ry')} Rx={row.get('openface_pose_Rx')}"),
            ("L2CS", f"{row.get('l2cs_direction')} yaw={row.get('l2cs_yaw')} pitch={row.get('l2cs_pitch')}"),
            ("6DRepNet", f"{row.get('sixd_pose_direction')} yaw={row.get('sixd_yaw')} pitch={row.get('sixd_pitch')} roll={row.get('sixd_roll')}"),
            ("visualization", "OpenFace uses 0.5s bin mean values, not raw per-frame landmark output."),
            ("crop video", row.get("crop_video_path")),
            ("status", f"{row.get('inference_status')} {row.get('error_note')}"),
        ]
        dl = "".join(f"<dt>{fmt(k)}</dt><dd>{fmt(v)}</dd>" for k, v in details)
        cards.append(f'<article class="{" ".join(classes)}"><h2>#{idx} {fmt(row.get("movie_id"))} {fmt(row.get("shot_id"))} bin {fmt(row.get("bin_idx"))}</h2><div class="viz-grid">{figures}</div><dl>{dl}</dl></article>')
    css = """
    body{font-family:Arial,sans-serif;margin:24px;background:#f7f7f5;color:#202020}
    .summary,.card{background:white;border:1px solid #ddd;border-radius:8px;padding:14px;margin:16px 0}
    .conflict{border-color:#d18b00;background:#fffaf0}.badtrack{border-color:#a33;background:#fff6f6}
    .viz-grid{display:grid;grid-template-columns:repeat(4,minmax(180px,1fr));gap:12px;margin-bottom:14px}
    figure{margin:0}figcaption{font-size:13px;font-weight:700;color:#555;margin-bottom:4px}
    img{max-width:100%;border:1px solid #ccc;border-radius:6px;background:#222}
    .placeholder{aspect-ratio:4/3;display:grid;place-items:center;background:#eee;border:1px dashed #aaa;border-radius:6px}
    dl{display:grid;grid-template-columns:120px 1fr;gap:6px 12px;margin:0}dt{font-weight:700;color:#555}dd{margin:0;overflow-wrap:anywhere}
    h2{font-size:18px}@media(max-width:1100px){.viz-grid{grid-template-columns:repeat(2,minmax(180px,1fr))}}@media(max-width:640px){.viz-grid{grid-template-columns:1fr}}
    """
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Gaze evidence bakeoff</title><style>{css}</style></head>
<body><h1>Gaze evidence bakeoff</h1><section class="summary"><pre>{fmt(json.dumps(summary, ensure_ascii=False, indent=2))}</pre></section>{''.join(cards)}</body></html>
"""
    write_text(report_path, doc)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("outputs/video_proxy"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/video_proxy/model_bakeoff/v0_3"))
    parser.add_argument("--movies", default=",".join(MOVIES), help="Comma-separated movie IDs.")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu, cuda:0, or auto.")
    parser.add_argument("--l2cs-repo", type=Path, default=Path("models/gaze_estimation/L2CS-Net"))
    parser.add_argument("--l2cs-weights", type=Path, default=Path("models/gaze_estimation/L2CS-Net/models/L2CSNet_gaze360.pkl"))
    parser.add_argument("--sixd-repo", type=Path, default=Path("models/head_pose/6DRepNet360"))
    parser.add_argument(
        "--sixd-weights",
        type=Path,
        default=Path("models/head_pose/6DRepNet360/checkpoints/6DRepNet360_Full-Rotation_300W_LP_Panoptic.pth"),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib-expregaze"))
    movies = [item.strip() for item in args.movies.split(",") if item.strip()]
    rows, summary = build_compare_rows(
        args.base_dir,
        args.output_dir,
        movies,
        args.l2cs_repo,
        args.l2cs_weights,
        args.sixd_repo,
        args.sixd_weights,
        args.device,
    )
    write_csv(args.output_dir / "gaze_evidence_compare.csv", rows, OUTPUT_COLUMNS)
    write_json(args.output_dir / "gaze_evidence_bakeoff_summary.json", summary)
    render_html(rows, summary, args.output_dir)
    print(f"[Bakeoff] rows={len(rows)} csv={args.output_dir / 'gaze_evidence_compare.csv'}")
    if not summary["l2cs_available"] or not summary["sixdrepnet_available"]:
        print("[Bakeoff] model inference skipped:", summary["dependency_status"])


if __name__ == "__main__":
    main()
