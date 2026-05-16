"""Run face detection and shot-local tracking on Stage04 shot clips.

Stage05 consumes the Stage04 shot manifest and works only within each shot.
It does not perform cross-shot identity linking, cast assignment, face crop
video generation, or OpenFace extraction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from expregaze.video_proxy.stage_filter import (
    filter_manifest_rows_by_stage_type,
    resolve_stage_type_include,
    stage_type_include_label,
)


DETECTION_COLUMNS = [
    "movie_id",
    "sequence_id",
    "shot_id",
    "shot_idx",
    "shot_clip_path",
    "sample_order",
    "frame_idx",
    "timestamp_sec",
    "det_id",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "bbox_w",
    "bbox_h",
    "bbox_cx",
    "bbox_cy",
    "det_conf",
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
    "frame_width",
    "frame_height",
]

TRACK_COLUMNS = [
    "movie_id",
    "sequence_id",
    "shot_id",
    "shot_idx",
    "local_track_id",
    "sample_order",
    "frame_idx",
    "timestamp_sec",
    "det_id",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "bbox_w",
    "bbox_h",
    "bbox_cx",
    "bbox_cy",
    "det_conf",
    "track_conf",
    "track_len",
]

SHOT_SUMMARY_COLUMNS = [
    "movie_id",
    "sequence_id",
    "shot_id",
    "shot_idx",
    "shot_clip_path",
    "process_status",
    "note",
    "video_fps",
    "video_frame_count",
    "sampled_frame_count",
    "detected_frame_count",
    "detection_count",
    "track_count",
    "longest_track_len",
    "mean_det_conf",
]


@dataclass(frozen=True)
class Stage05Config:
    movie_id: str
    manifest_csv: Path
    face_track_dir: Path
    logs_dir: Path
    yunet_model_path: Path
    sample_fps: float
    score_threshold: float
    nms_threshold: float
    resize_max: int
    min_face_size_px: float
    save_debug_overlays: bool
    track_iou_threshold: float
    center_distance_max_frac: float
    max_gap_frames: int
    min_track_length_frames: int
    overwrite: bool
    max_shots: int | None
    sequence_id_list: list[str] | None
    shot_id_list: list[str] | None
    stage_type_include: set[str] | None

    @property
    def detections_csv(self) -> Path:
        return self.face_track_dir / "05_face_detections.csv"

    @property
    def tracks_csv(self) -> Path:
        return self.face_track_dir / "05_face_tracks.csv"

    @property
    def shot_summary_csv(self) -> Path:
        return self.face_track_dir / "05_shot_track_summary.csv"

    @property
    def summary_json(self) -> Path:
        return self.logs_dir / "05_face_detection_tracks_summary.json"

    @property
    def debug_overlay_dir(self) -> Path:
        return self.face_track_dir / "debug_overlays"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def resolve_path(value: str | Path | None, project_root: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def parse_id_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in {"none", "null", "auto"}:
            return None
        return [item.strip() for item in stripped.split(",") if item.strip()]
    if isinstance(value, list):
        ids = [str(item).strip() for item in value if str(item).strip()]
        return ids or None
    raise ValueError(f"Unsupported id list: {value!r}")


def parse_optional_int(value: Any, flag_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"none", "null", "auto", ""}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{flag_name} must be positive, got {value!r}")
    return parsed


def load_run_config(run_config_path: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    run_config = load_yaml(run_config_path)
    paths_config_path = Path(run_config.get("inputs", {}).get("paths_config", "configs/path_local.yaml"))
    if not paths_config_path.is_absolute():
        paths_config_path = run_config_path.parent.parent.parent / paths_config_path
    paths_config = load_yaml(paths_config_path)
    project_root = Path(paths_config.get("project", {}).get("root", run_config_path.parent.parent.parent))
    return run_config, paths_config, project_root


def make_config(args: argparse.Namespace) -> Stage05Config:
    run_config: dict[str, Any] = {}
    paths_config: dict[str, Any] = {}
    project_root = Path.cwd()

    if args.run_config:
        run_config_path = Path(args.run_config).resolve()
        run_config, paths_config, project_root = load_run_config(run_config_path)

    movie_id = args.movie_id or run_config.get("data", {}).get("movie_id")
    if not movie_id:
        raise ValueError("movie_id is required via --movie-id or run config data.movie_id")

    outputs = run_config.get("outputs", {})
    selection = run_config.get("selection", {})
    detect_stage = run_config.get("stages", {}).get("detect_and_crop_faces", {})
    track_stage = run_config.get("stages", {}).get("build_face_tracks", {})

    logs_dir = resolve_path(outputs.get("logs_dir") or f"outputs/video_proxy/{movie_id}/logs", project_root)
    assert logs_dir is not None
    manifest_csv = resolve_path(args.manifest_csv, project_root) or logs_dir / "04_shot_manifest.csv"

    face_track_dir = resolve_path(
        args.face_track_dir or outputs.get("face_track_dir") or f"outputs/video_proxy/{movie_id}/face_tracks",
        project_root,
    )
    assert face_track_dir is not None

    yunet_model_path = resolve_path(
        args.yunet_model or detect_stage.get("yunet_model_path") or "models/face_detection/face_detection_yunet_2023mar.onnx",
        project_root,
    )
    assert yunet_model_path is not None

    save_debug_overlays = bool(detect_stage.get("save_debug_overlays", True))
    if args.save_debug_overlays:
        save_debug_overlays = True
    if args.no_debug_overlays:
        save_debug_overlays = False

    return Stage05Config(
        movie_id=str(movie_id),
        manifest_csv=manifest_csv,
        face_track_dir=face_track_dir,
        logs_dir=logs_dir,
        yunet_model_path=yunet_model_path,
        sample_fps=float(args.sample_fps if args.sample_fps is not None else detect_stage.get("sample_fps", 5)),
        score_threshold=float(
            args.score_threshold if args.score_threshold is not None else detect_stage.get("score_threshold", 0.6)
        ),
        nms_threshold=float(args.nms_threshold if args.nms_threshold is not None else detect_stage.get("nms_threshold", 0.3)),
        resize_max=int(args.resize_max if args.resize_max is not None else detect_stage.get("resize_max", 960)),
        min_face_size_px=float(
            args.min_face_size_px if args.min_face_size_px is not None else detect_stage.get("min_face_size_px", 40)
        ),
        save_debug_overlays=save_debug_overlays,
        track_iou_threshold=float(
            args.track_iou_threshold
            if args.track_iou_threshold is not None
            else track_stage.get("track_iou_threshold", 0.3)
        ),
        center_distance_max_frac=float(
            args.center_distance_max_frac
            if args.center_distance_max_frac is not None
            else track_stage.get("center_distance_max_frac", 0.20)
        ),
        max_gap_frames=int(args.max_gap_frames if args.max_gap_frames is not None else track_stage.get("max_gap_frames", 3)),
        min_track_length_frames=int(
            args.min_track_length_frames
            if args.min_track_length_frames is not None
            else track_stage.get("min_track_length_frames", 3)
        ),
        overwrite=bool(args.overwrite or detect_stage.get("overwrite", False) or track_stage.get("overwrite", False)),
        max_shots=parse_optional_int(args.max_shots, "--max-shots"),
        sequence_id_list=parse_id_list(args.sequence_id_list),
        shot_id_list=parse_id_list(args.shot_id_list),
        stage_type_include=resolve_stage_type_include(args.stage_type_include, selection.get("stage_type_include")),
    )


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def filter_manifest_rows(rows: list[dict[str, str]], config: Stage05Config) -> list[dict[str, str]]:
    selected = [row for row in rows if row.get("movie_id") == config.movie_id]
    selected, _, _ = filter_manifest_rows_by_stage_type(selected, config.stage_type_include)
    if config.sequence_id_list:
        wanted_sequences = set(config.sequence_id_list)
        selected = [row for row in selected if row.get("sequence_id") in wanted_sequences]
    if config.shot_id_list:
        wanted_shots = set(config.shot_id_list)
        selected = [row for row in selected if row.get("shot_id") in wanted_shots]
    if config.max_shots is not None:
        selected = selected[: config.max_shots]
    if not selected:
        raise ValueError("No shot rows selected from Stage04 manifest")
    return selected


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed):
        return default
    return parsed


def bbox_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax1, ay1, ax2, ay2 = safe_float(a["bbox_x1"]), safe_float(a["bbox_y1"]), safe_float(a["bbox_x2"]), safe_float(a["bbox_y2"])
    bx1, by1, bx2, by2 = safe_float(b["bbox_x1"]), safe_float(b["bbox_y1"]), safe_float(b["bbox_x2"]), safe_float(b["bbox_y2"])
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def center_distance_frac(a: dict[str, Any], b: dict[str, Any]) -> float:
    frame_w = max(safe_float(a.get("frame_width"), safe_float(b.get("frame_width"), 1.0)), 1.0)
    frame_h = max(safe_float(a.get("frame_height"), safe_float(b.get("frame_height"), 1.0)), 1.0)
    diag = math.sqrt(frame_w * frame_w + frame_h * frame_h)
    dx = safe_float(a["bbox_cx"]) - safe_float(b["bbox_cx"])
    dy = safe_float(a["bbox_cy"]) - safe_float(b["bbox_cy"])
    return math.sqrt(dx * dx + dy * dy) / diag


def create_yunet_detector(config: Stage05Config):
    if not config.yunet_model_path.exists():
        raise FileNotFoundError(
            f"YuNet model not found: {config.yunet_model_path}. "
            "Pass --yunet-model or set stages.detect_and_crop_faces.yunet_model_path in the run yaml."
        )
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for Stage05. Run through uv or install opencv-python.") from exc
    if not hasattr(cv2, "FaceDetectorYN"):
        raise RuntimeError("This OpenCV build does not provide cv2.FaceDetectorYN, required for YuNet.")
    return cv2.FaceDetectorYN.create(
        model=str(config.yunet_model_path),
        config="",
        input_size=(320, 320),
        score_threshold=config.score_threshold,
        nms_threshold=config.nms_threshold,
        top_k=5000,
    )


def detect_faces_in_frame(
    frame: Any,
    detector: Any,
    config: Stage05Config,
    frame_idx: int,
    sample_order: int,
    timestamp_sec: float,
    manifest_row: dict[str, str],
) -> list[dict[str, Any]]:
    import cv2

    height, width = frame.shape[:2]
    scale = 1.0
    if max(height, width) > config.resize_max:
        scale = config.resize_max / max(height, width)
        frame_in = cv2.resize(frame, (int(width * scale), int(height * scale)))
    else:
        frame_in = frame

    input_h, input_w = frame_in.shape[:2]
    detector.setInputSize((input_w, input_h))
    _, faces = detector.detect(frame_in)
    if faces is None or len(faces) == 0:
        return []

    inv = 1.0 / scale
    rows: list[dict[str, Any]] = []
    for raw_det_id, face in enumerate(faces):
        x, y, bbox_w, bbox_h = [float(v) * inv for v in face[:4]]
        if bbox_w < config.min_face_size_px or bbox_h < config.min_face_size_px:
            continue
        x1 = max(0.0, x)
        y1 = max(0.0, y)
        x2 = min(float(width), x + bbox_w)
        y2 = min(float(height), y + bbox_h)
        if x2 <= x1 or y2 <= y1:
            continue
        landmarks = [float(v) * inv for v in face[4:14]]
        rows.append(
            {
                "movie_id": config.movie_id,
                "sequence_id": manifest_row.get("sequence_id", ""),
                "shot_id": manifest_row.get("shot_id", ""),
                "shot_idx": manifest_row.get("shot_idx", ""),
                "shot_clip_path": manifest_row.get("shot_clip_path", ""),
                "sample_order": sample_order,
                "frame_idx": frame_idx,
                "timestamp_sec": f"{timestamp_sec:.3f}",
                "det_id": raw_det_id,
                "bbox_x1": f"{x1:.3f}",
                "bbox_y1": f"{y1:.3f}",
                "bbox_x2": f"{x2:.3f}",
                "bbox_y2": f"{y2:.3f}",
                "bbox_w": f"{x2 - x1:.3f}",
                "bbox_h": f"{y2 - y1:.3f}",
                "bbox_cx": f"{(x1 + x2) / 2.0:.3f}",
                "bbox_cy": f"{(y1 + y2) / 2.0:.3f}",
                "det_conf": f"{float(face[14]):.6f}",
                "left_eye_x": f"{landmarks[0]:.3f}",
                "left_eye_y": f"{landmarks[1]:.3f}",
                "right_eye_x": f"{landmarks[2]:.3f}",
                "right_eye_y": f"{landmarks[3]:.3f}",
                "nose_x": f"{landmarks[4]:.3f}",
                "nose_y": f"{landmarks[5]:.3f}",
                "mouth_left_x": f"{landmarks[6]:.3f}",
                "mouth_left_y": f"{landmarks[7]:.3f}",
                "mouth_right_x": f"{landmarks[8]:.3f}",
                "mouth_right_y": f"{landmarks[9]:.3f}",
                "frame_width": width,
                "frame_height": height,
            }
        )

    rows.sort(key=lambda row: safe_float(row["det_conf"]), reverse=True)
    for det_id, row in enumerate(rows):
        row["det_id"] = det_id
    return rows


def sampled_frame_indices(frame_count: int, fps: float, sample_fps: float) -> list[int]:
    if frame_count <= 0:
        return []
    if fps <= 0 or sample_fps <= 0:
        return list(range(frame_count))
    step = max(1, int(round(fps / sample_fps)))
    return list(range(0, frame_count, step))


def run_detection_for_shot(
    manifest_row: dict[str, str],
    detector: Any,
    config: Stage05Config,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import cv2

    shot_clip_path = Path(manifest_row.get("shot_clip_path", ""))
    summary = {
        "movie_id": config.movie_id,
        "sequence_id": manifest_row.get("sequence_id", ""),
        "shot_id": manifest_row.get("shot_id", ""),
        "shot_idx": manifest_row.get("shot_idx", ""),
        "shot_clip_path": str(shot_clip_path),
        "process_status": "pending",
        "note": "",
        "video_fps": 0.0,
        "video_frame_count": 0,
        "sampled_frame_count": 0,
        "detected_frame_count": 0,
        "detection_count": 0,
        "track_count": 0,
        "longest_track_len": 0,
        "mean_det_conf": 0.0,
    }

    if not shot_clip_path.exists():
        summary["process_status"] = "skipped_missing_clip"
        summary["note"] = "shot clip not found"
        return [], summary

    cap = cv2.VideoCapture(str(shot_clip_path))
    if not cap.isOpened():
        summary["process_status"] = "error"
        summary["note"] = "could not open shot clip"
        return [], summary

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indices = sampled_frame_indices(frame_count, fps, config.sample_fps)
    detections: list[dict[str, Any]] = []

    for sample_order, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        timestamp_sec = frame_idx / fps if fps > 0 else float(sample_order)
        detections.extend(
            detect_faces_in_frame(
                frame,
                detector,
                config,
                frame_idx=frame_idx,
                sample_order=sample_order,
                timestamp_sec=timestamp_sec,
                manifest_row=manifest_row,
            )
        )

    cap.release()

    detected_orders = {row["sample_order"] for row in detections}
    confs = [safe_float(row["det_conf"]) for row in detections]
    summary.update(
        {
            "process_status": "done",
            "video_fps": f"{fps:.3f}",
            "video_frame_count": frame_count,
            "sampled_frame_count": len(indices),
            "detected_frame_count": len(detected_orders),
            "detection_count": len(detections),
            "mean_det_conf": f"{(sum(confs) / len(confs)):.6f}" if confs else "0.000000",
        }
    )
    return detections, summary


def build_tracks_for_shot(detections: list[dict[str, Any]], config: Stage05Config) -> list[dict[str, Any]]:
    if not detections:
        return []

    tracks: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    next_track_idx = 0

    detections_by_order: dict[int, list[dict[str, Any]]] = {}
    for det in detections:
        detections_by_order.setdefault(int(det["sample_order"]), []).append(det)

    for sample_order in sorted(detections_by_order):
        used_track_indices: set[int] = set()
        frame_detections = sorted(detections_by_order[sample_order], key=lambda row: safe_float(row["det_conf"]), reverse=True)

        for det in frame_detections:
            best_track_index: int | None = None
            best_score = -1e9
            for track_index, track in enumerate(tracks):
                if track_index in used_track_indices:
                    continue
                gap = sample_order - int(track["last_sample_order"])
                if gap <= 0 or gap > config.max_gap_frames + 1:
                    continue
                iou = bbox_iou(det, track["last_detection"])
                center_frac = center_distance_frac(det, track["last_detection"])
                if iou < config.track_iou_threshold and center_frac > config.center_distance_max_frac:
                    continue
                score = iou - center_frac
                if score > best_score:
                    best_score = score
                    best_track_index = track_index

            if best_track_index is None:
                local_track_id = f"trk_{next_track_idx:03d}"
                next_track_idx += 1
                tracks.append(
                    {
                        "local_track_id": local_track_id,
                        "last_sample_order": sample_order,
                        "last_detection": det,
                        "detections": [det],
                    }
                )
            else:
                track = tracks[best_track_index]
                local_track_id = str(track["local_track_id"])
                track["last_sample_order"] = sample_order
                track["last_detection"] = det
                track["detections"].append(det)
                used_track_indices.add(best_track_index)

            assignments.append({**det, "local_track_id": local_track_id})

    track_stats: dict[str, dict[str, Any]] = {}
    for track in tracks:
        local_track_id = str(track["local_track_id"])
        dets = track["detections"]
        confs = [safe_float(row["det_conf"]) for row in dets]
        track_stats[local_track_id] = {
            "track_len": len(dets),
            "track_conf": sum(confs) / len(confs) if confs else 0.0,
        }

    rows: list[dict[str, Any]] = []
    for assignment in assignments:
        local_track_id = str(assignment["local_track_id"])
        stats = track_stats[local_track_id]
        if int(stats["track_len"]) < config.min_track_length_frames:
            continue
        rows.append(
            {
                "movie_id": assignment["movie_id"],
                "sequence_id": assignment["sequence_id"],
                "shot_id": assignment["shot_id"],
                "shot_idx": assignment["shot_idx"],
                "local_track_id": local_track_id,
                "sample_order": assignment["sample_order"],
                "frame_idx": assignment["frame_idx"],
                "timestamp_sec": assignment["timestamp_sec"],
                "det_id": assignment["det_id"],
                "bbox_x1": assignment["bbox_x1"],
                "bbox_y1": assignment["bbox_y1"],
                "bbox_x2": assignment["bbox_x2"],
                "bbox_y2": assignment["bbox_y2"],
                "bbox_w": assignment["bbox_w"],
                "bbox_h": assignment["bbox_h"],
                "bbox_cx": assignment["bbox_cx"],
                "bbox_cy": assignment["bbox_cy"],
                "det_conf": assignment["det_conf"],
                "track_conf": f"{float(stats['track_conf']):.6f}",
                "track_len": int(stats["track_len"]),
            }
        )
    return rows


def draw_debug_overlays(
    manifest_row: dict[str, str],
    track_rows: list[dict[str, Any]],
    config: Stage05Config,
) -> int:
    if not config.save_debug_overlays or not track_rows:
        return 0

    import cv2

    shot_clip_path = Path(manifest_row.get("shot_clip_path", ""))
    cap = cv2.VideoCapture(str(shot_clip_path))
    if not cap.isOpened():
        return 0

    rows_by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in track_rows:
        rows_by_frame.setdefault(int(row["frame_idx"]), []).append(row)

    output_dir = config.debug_overlay_dir / str(manifest_row.get("sequence_id", "")) / str(manifest_row.get("shot_id", ""))
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = [
        (60, 220, 60),
        (60, 160, 255),
        (255, 120, 60),
        (220, 60, 220),
        (80, 255, 255),
        (255, 255, 80),
    ]
    written = 0
    for frame_idx in sorted(rows_by_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        for row in rows_by_frame[frame_idx]:
            track_number = int(str(row["local_track_id"]).split("_")[-1])
            color = colors[track_number % len(colors)]
            x1 = int(round(safe_float(row["bbox_x1"])))
            y1 = int(round(safe_float(row["bbox_y1"])))
            x2 = int(round(safe_float(row["bbox_x2"])))
            y2 = int(round(safe_float(row["bbox_y2"])))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{row['local_track_id']} conf={safe_float(row['track_conf']):.2f}"
            cv2.putText(frame, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        output_path = output_dir / f"frame_{int(rows_by_frame[frame_idx][0]['sample_order']):04d}_f{frame_idx:06d}.jpg"
        cv2.imwrite(str(output_path), frame)
        written += 1

    cap.release()
    return written


def write_summary_json(config: Stage05Config, payload: dict[str, Any]) -> None:
    config.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with config.summary_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def run(config: Stage05Config) -> None:
    if not config.manifest_csv.exists():
        raise FileNotFoundError(f"Stage04 manifest not found: {config.manifest_csv}")
    if not config.overwrite and config.detections_csv.exists() and config.tracks_csv.exists():
        print(f"[Stage05] outputs already exist; use --overwrite to regenerate: {config.face_track_dir}")
        return

    detector = create_yunet_detector(config)
    all_manifest_rows = [row for row in read_manifest(config.manifest_csv) if row.get("movie_id") == config.movie_id]
    _, skipped_stage_type_count, stage_counts = filter_manifest_rows_by_stage_type(all_manifest_rows, config.stage_type_include)
    manifest_rows = filter_manifest_rows(all_manifest_rows, config)

    all_detection_rows: list[dict[str, Any]] = []
    all_track_rows: list[dict[str, Any]] = []
    shot_summary_rows: list[dict[str, Any]] = []
    overlay_count = 0

    for manifest_row in manifest_rows:
        detections, shot_summary = run_detection_for_shot(manifest_row, detector, config)
        track_rows = build_tracks_for_shot(detections, config)
        unique_tracks = {row["local_track_id"] for row in track_rows}
        shot_summary["track_count"] = len(unique_tracks)
        shot_summary["longest_track_len"] = max([int(row["track_len"]) for row in track_rows], default=0)

        all_detection_rows.extend(detections)
        all_track_rows.extend(track_rows)
        shot_summary_rows.append(shot_summary)
        overlay_count += draw_debug_overlays(manifest_row, track_rows, config)

    write_csv(config.detections_csv, all_detection_rows, DETECTION_COLUMNS)
    write_csv(config.tracks_csv, all_track_rows, TRACK_COLUMNS)
    write_csv(config.shot_summary_csv, shot_summary_rows, SHOT_SUMMARY_COLUMNS)

    status_counts: dict[str, int] = {}
    for row in shot_summary_rows:
        status = str(row.get("process_status") or "")
        status_counts[status] = status_counts.get(status, 0) + 1

    payload = {
        "movie_id": config.movie_id,
        "manifest_csv": str(config.manifest_csv),
        "yunet_model_path": str(config.yunet_model_path),
        "sample_fps": config.sample_fps,
        "shot_count": len(manifest_rows),
        "stage_type_include": stage_type_include_label(config.stage_type_include),
        "stage_type_counts": stage_counts,
        "skipped_stage_type_count": skipped_stage_type_count,
        "status_counts": status_counts,
        "detection_count": len(all_detection_rows),
        "track_row_count": len(all_track_rows),
        "unique_shot_track_count": len({(row["shot_id"], row["local_track_id"]) for row in all_track_rows}),
        "debug_overlay_count": overlay_count,
        "outputs": {
            "detections_csv": str(config.detections_csv),
            "tracks_csv": str(config.tracks_csv),
            "shot_summary_csv": str(config.shot_summary_csv),
            "summary_json": str(config.summary_json),
            "debug_overlay_dir": str(config.debug_overlay_dir),
        },
    }
    write_summary_json(config, payload)

    print(f"[Stage05] movie_id={config.movie_id}")
    print(f"[Stage05] manifest_csv={config.manifest_csv}")
    print(f"[Stage05] yunet_model_path={config.yunet_model_path}")
    print(
        f"[Stage05] stage_type_include={stage_type_include_label(config.stage_type_include)} "
        f"skipped_stage_type={skipped_stage_type_count} stage_type_counts={json.dumps(stage_counts, sort_keys=True)}"
    )
    print(f"[Stage05] shots={len(manifest_rows)} status_counts={json.dumps(status_counts, sort_keys=True)}")
    print(f"[Stage05] detections={len(all_detection_rows)} track_rows={len(all_track_rows)} overlays={overlay_count}")
    print(f"[Stage05] detections_csv={config.detections_csv}")
    print(f"[Stage05] tracks_csv={config.tracks_csv}")
    print(f"[Stage05] summary_json={config.summary_json}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", help="Run YAML, e.g. configs/runs/video_proxy_tt0032138.yaml")
    parser.add_argument("--movie-id")
    parser.add_argument("--manifest-csv")
    parser.add_argument("--face-track-dir")
    parser.add_argument("--yunet-model")
    parser.add_argument("--sample-fps", type=float)
    parser.add_argument("--score-threshold", type=float)
    parser.add_argument("--nms-threshold", type=float)
    parser.add_argument("--resize-max", type=int)
    parser.add_argument("--min-face-size-px", type=float)
    parser.add_argument("--track-iou-threshold", type=float)
    parser.add_argument("--center-distance-max-frac", type=float)
    parser.add_argument("--max-gap-frames", type=int)
    parser.add_argument("--min-track-length-frames", type=int)
    parser.add_argument("--max-shots")
    parser.add_argument("--sequence-id-list")
    parser.add_argument("--shot-id-list")
    parser.add_argument("--stage-type-include", help="Comma list of stage_type values to process, or 'all'.")
    parser.add_argument("--no-sface-gallery", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--save-debug-overlays", action="store_true")
    parser.add_argument("--no-debug-overlays", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        run(make_config(args))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[Stage05] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
