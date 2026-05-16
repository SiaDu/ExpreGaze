"""Crop shot-local face tracks, run OpenFace, and build gaze timebins."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
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


TRACK_MANIFEST_COLUMNS = [
    "movie_id",
    "sequence_id",
    "shot_id",
    "shot_idx",
    "local_track_id",
    "shot_clip_path",
    "track_len",
    "track_conf",
    "first_timestamp_sec",
    "last_timestamp_sec",
    "crop_start_sec",
    "crop_end_sec",
    "crop_duration_sec",
    "crop_x",
    "crop_y",
    "crop_w",
    "crop_h",
    "crop_video_path",
    "openface_output_dir",
    "openface_csv_path",
    "process_status",
    "note",
]

RAW_INDEX_COLUMNS = [
    "movie_id",
    "sequence_id",
    "shot_id",
    "shot_idx",
    "local_track_id",
    "crop_video_path",
    "openface_output_dir",
    "openface_csv_path",
    "crop_status",
    "openface_status",
    "openface_returncode",
    "crop_file_size",
    "openface_csv_size",
    "note",
]

AU_R_CODES = [
    "AU01",
    "AU02",
    "AU04",
    "AU05",
    "AU06",
    "AU07",
    "AU09",
    "AU10",
    "AU12",
    "AU14",
    "AU15",
    "AU17",
    "AU20",
    "AU23",
    "AU25",
    "AU26",
    "AU45",
]

AU_C_CODES = ["AU01", "AU04", "AU06", "AU12", "AU15", "AU25", "AU26", "AU45"]
AU_R_COLUMNS = [f"{code}_r_mean" for code in AU_R_CODES]
AU_C_COLUMNS = [f"{code}_c_ratio" for code in AU_C_CODES]

TIMEBIN_COLUMNS = [
    "movie_id",
    "sequence_id",
    "shot_id",
    "shot_idx",
    "local_track_id",
    "bin_idx",
    "bin_start_sec",
    "bin_end_sec",
    "n_frames_total",
    "n_frames_valid",
    "valid_ratio",
    "confidence_mean",
    "confidence_median",
    "gaze_angle_x_mean",
    "gaze_angle_y_mean",
    "gaze_angle_x_std",
    "gaze_angle_y_std",
    "pose_Rx_mean",
    "pose_Ry_mean",
    "pose_Rz_mean",
    "pose_Rx_std",
    "pose_Ry_std",
    "pose_Rz_std",
    "gaze_quality",
    *AU_R_COLUMNS,
    *AU_C_COLUMNS,
    "expression_proxy",
    "crop_video_path",
    "openface_csv_path",
]


@dataclass(frozen=True)
class Stage06Config:
    movie_id: str
    manifest_csv: Path
    face_tracks_csv: Path
    face_crop_dir: Path
    openface_dir: Path
    logs_dir: Path
    ffmpeg_binary: str
    openface_binary: Path
    crop_margin: float
    track_time_padding_sec: float
    min_track_length_frames: int
    min_track_conf: float
    timebin_sec: float
    save_aligned_faces: bool
    openface_aus: bool
    expression_proxy: bool
    overwrite: bool
    max_tracks: int | None
    sequence_id_list: list[str] | None
    shot_id_list: list[str] | None
    track_id_list: list[str] | None
    stage_type_include: set[str] | None

    @property
    def track_manifest_csv(self) -> Path:
        return self.openface_dir / "06_track_manifest.csv"

    @property
    def raw_index_csv(self) -> Path:
        return self.openface_dir / "06_openface_raw_index.csv"

    @property
    def gaze_timebins_csv(self) -> Path:
        return self.openface_dir / "06_gaze_timebins.csv"

    @property
    def summary_json(self) -> Path:
        return self.logs_dir / "06_run_openface_per_track_summary.json"

    @property
    def raw_output_dir(self) -> Path:
        return self.openface_dir / "raw"


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


def make_config(args: argparse.Namespace) -> Stage06Config:
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
    stage = run_config.get("stages", {}).get("run_openface_per_face", {})

    logs_dir = resolve_path(outputs.get("logs_dir") or f"outputs/video_proxy/{movie_id}/logs", project_root)
    assert logs_dir is not None
    openface_dir = resolve_path(outputs.get("openface_dir") or f"outputs/video_proxy/{movie_id}/openface", project_root)
    assert openface_dir is not None
    face_crop_dir = resolve_path(outputs.get("face_crop_dir") or f"outputs/video_proxy/{movie_id}/face_crops", project_root)
    assert face_crop_dir is not None
    face_track_dir = resolve_path(outputs.get("face_track_dir") or f"outputs/video_proxy/{movie_id}/face_tracks", project_root)
    assert face_track_dir is not None

    manifest_csv = resolve_path(args.manifest_csv, project_root) or logs_dir / "04_shot_manifest.csv"
    face_tracks_csv = resolve_path(args.face_tracks_csv, project_root) or face_track_dir / "05_face_tracks.csv"
    openface_binary = resolve_path(
        args.openface_binary or stage.get("openface_binary") or paths_config.get("openface", {}).get("binary"),
        project_root,
    )
    if openface_binary is None:
        raise ValueError("OpenFace binary is required via config openface.binary or --openface-binary")

    save_aligned_faces = bool(stage.get("save_aligned_faces", False))
    if args.save_aligned_faces:
        save_aligned_faces = True

    return Stage06Config(
        movie_id=str(movie_id),
        manifest_csv=manifest_csv,
        face_tracks_csv=face_tracks_csv,
        face_crop_dir=face_crop_dir,
        openface_dir=openface_dir,
        logs_dir=logs_dir,
        ffmpeg_binary=str(args.ffmpeg_binary or paths_config.get("openface", {}).get("ffmpeg_binary") or "ffmpeg"),
        openface_binary=openface_binary,
        crop_margin=float(args.crop_margin if args.crop_margin is not None else stage.get("crop_margin", 0.35)),
        track_time_padding_sec=float(
            args.track_time_padding_sec
            if args.track_time_padding_sec is not None
            else stage.get("track_time_padding_sec", 0.25)
        ),
        min_track_length_frames=int(
            args.min_track_length_frames
            if args.min_track_length_frames is not None
            else stage.get("min_track_length_frames", 3)
        ),
        min_track_conf=float(args.min_track_conf if args.min_track_conf is not None else stage.get("min_track_conf", 0.6)),
        timebin_sec=float(args.timebin_sec if args.timebin_sec is not None else stage.get("timebin_sec", 0.5)),
        save_aligned_faces=save_aligned_faces,
        openface_aus=bool(stage.get("openface_aus", True)) and not args.no_openface_aus,
        expression_proxy=bool(stage.get("expression_proxy", True)) and not args.no_expression_proxy,
        overwrite=bool(args.overwrite or stage.get("overwrite", False)),
        max_tracks=parse_optional_int(args.max_tracks, "--max-tracks"),
        sequence_id_list=parse_id_list(args.sequence_id_list),
        shot_id_list=parse_id_list(args.shot_id_list),
        track_id_list=parse_id_list(args.track_id_list),
        stage_type_include=resolve_stage_type_include(args.stage_type_include, selection.get("stage_type_include")),
    )


def read_csv(path: Path) -> list[dict[str, str]]:
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


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed):
        return default
    return parsed


def finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def group_track_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    order: list[tuple[str, str, str, str]] = []
    for row in rows:
        key = (
            row.get("movie_id", ""),
            row.get("sequence_id", ""),
            row.get("shot_id", ""),
            row.get("local_track_id", ""),
        )
        if key not in groups:
            order.append(key)
            groups[key] = []
        groups[key].append(row)

    out: list[dict[str, Any]] = []
    for key in order:
        track_rows = groups[key]
        first = track_rows[0]
        out.append(
            {
                "movie_id": first.get("movie_id", ""),
                "sequence_id": first.get("sequence_id", ""),
                "shot_id": first.get("shot_id", ""),
                "shot_idx": first.get("shot_idx", ""),
                "local_track_id": first.get("local_track_id", ""),
                "track_len": max(int(safe_float(row.get("track_len"), 0)) for row in track_rows),
                "track_conf": sum(safe_float(row.get("track_conf"), 0.0) for row in track_rows) / len(track_rows),
                "rows": sorted(track_rows, key=lambda row: safe_float(row.get("timestamp_sec"), 0.0)),
            }
        )
    return out


def filter_tracks(
    groups: list[dict[str, Any]],
    config: Stage06Config,
    allowed_shot_keys: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    selected = [group for group in groups if group["movie_id"] == config.movie_id]
    selected = [group for group in selected if (group["sequence_id"], group["shot_id"]) in allowed_shot_keys]
    if config.sequence_id_list:
        selected = [group for group in selected if group["sequence_id"] in set(config.sequence_id_list)]
    if config.shot_id_list:
        selected = [group for group in selected if group["shot_id"] in set(config.shot_id_list)]
    if config.track_id_list:
        selected = [group for group in selected if group["local_track_id"] in set(config.track_id_list)]
    selected = [
        group
        for group in selected
        if int(group["track_len"]) >= config.min_track_length_frames and float(group["track_conf"]) >= config.min_track_conf
    ]
    if config.max_tracks is not None:
        selected = selected[: config.max_tracks]
    if not selected:
        raise ValueError("No Stage05 tracks selected for OpenFace")
    return selected


def get_video_info(path: Path) -> dict[str, float]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for Stage06 video metadata. Run through uv or install opencv-python.") from exc
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)
    height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)
    cap.release()
    duration = frame_count / fps if fps > 0 else 0.0
    return {"fps": fps, "frame_count": frame_count, "width": width, "height": height, "duration": duration}


def even_int(value: float, minimum: int = 2) -> int:
    parsed = max(minimum, int(round(value)))
    if parsed % 2 == 1:
        parsed -= 1
    return max(minimum, parsed)


def build_track_manifest(
    tracks: list[dict[str, Any]],
    shot_manifest_rows: list[dict[str, str]],
    config: Stage06Config,
) -> list[dict[str, Any]]:
    shot_lookup = {
        (row.get("sequence_id", ""), row.get("shot_id", "")): row
        for row in shot_manifest_rows
        if row.get("movie_id") == config.movie_id
    }
    video_info_cache: dict[Path, dict[str, float]] = {}
    manifest: list[dict[str, Any]] = []

    for group in tracks:
        key = (group["sequence_id"], group["shot_id"])
        shot_row = shot_lookup.get(key)
        if shot_row is None:
            raise ValueError(f"Shot not found in Stage04 manifest: {key}")
        shot_clip_path = Path(shot_row.get("shot_clip_path", ""))
        if shot_clip_path not in video_info_cache:
            video_info_cache[shot_clip_path] = get_video_info(shot_clip_path)
        info = video_info_cache[shot_clip_path]

        rows = group["rows"]
        x1 = min(safe_float(row["bbox_x1"]) for row in rows)
        y1 = min(safe_float(row["bbox_y1"]) for row in rows)
        x2 = max(safe_float(row["bbox_x2"]) for row in rows)
        y2 = max(safe_float(row["bbox_y2"]) for row in rows)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        side = max(x2 - x1, y2 - y1) * (1.0 + 2.0 * config.crop_margin)
        side = min(side, info["width"], info["height"])
        crop_w = even_int(side)
        crop_h = crop_w
        crop_x = even_int(max(0.0, min(cx - crop_w / 2.0, info["width"] - crop_w)), minimum=0)
        crop_y = even_int(max(0.0, min(cy - crop_h / 2.0, info["height"] - crop_h)), minimum=0)
        crop_w = even_int(min(crop_w, info["width"] - crop_x))
        crop_h = even_int(min(crop_h, info["height"] - crop_y))

        first_t = min(safe_float(row["timestamp_sec"]) for row in rows)
        last_t = max(safe_float(row["timestamp_sec"]) for row in rows)
        crop_start = max(0.0, first_t - config.track_time_padding_sec)
        crop_end = min(info["duration"], last_t + config.track_time_padding_sec)
        if crop_end <= crop_start:
            crop_end = min(info["duration"], crop_start + config.timebin_sec)

        stem = f"{group['shot_id']}__{group['local_track_id']}"
        crop_video_path = config.face_crop_dir / group["sequence_id"] / f"{stem}.mp4"
        openface_output_dir = config.raw_output_dir / group["sequence_id"] / stem
        openface_csv_path = openface_output_dir / f"{stem}.csv"
        manifest.append(
            {
                "movie_id": config.movie_id,
                "sequence_id": group["sequence_id"],
                "shot_id": group["shot_id"],
                "shot_idx": group["shot_idx"],
                "local_track_id": group["local_track_id"],
                "shot_clip_path": str(shot_clip_path),
                "track_len": int(group["track_len"]),
                "track_conf": f"{float(group['track_conf']):.6f}",
                "first_timestamp_sec": f"{first_t:.3f}",
                "last_timestamp_sec": f"{last_t:.3f}",
                "crop_start_sec": f"{crop_start:.3f}",
                "crop_end_sec": f"{crop_end:.3f}",
                "crop_duration_sec": f"{max(0.0, crop_end - crop_start):.3f}",
                "crop_x": crop_x,
                "crop_y": crop_y,
                "crop_w": crop_w,
                "crop_h": crop_h,
                "crop_video_path": str(crop_video_path),
                "openface_output_dir": str(openface_output_dir),
                "openface_csv_path": str(openface_csv_path),
                "process_status": "pending",
                "note": "",
            }
        )
    return manifest


def run_ffmpeg_crop(row: dict[str, Any], config: Stage06Config) -> tuple[str, str]:
    source = Path(row["shot_clip_path"])
    target = Path(row["crop_video_path"])
    if not source.exists():
        return "error", f"shot clip not found: {source}"
    if target.exists() and not config.overwrite:
        return "skipped_existing", ""
    target.parent.mkdir(parents=True, exist_ok=True)
    crop_filter = f"crop={int(row['crop_w'])}:{int(row['crop_h'])}:{int(row['crop_x'])}:{int(row['crop_y'])}"
    cmd = [
        config.ffmpeg_binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if config.overwrite else "-n",
        "-ss",
        str(row["crop_start_sec"]),
        "-i",
        str(source),
        "-t",
        str(row["crop_duration_sec"]),
        "-vf",
        crop_filter,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(target),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode == 0:
        return "done", ""
    note = (result.stderr or result.stdout or "").strip().replace("\n", " ")
    return "error", note[-500:]


def find_openface_csv(output_dir: Path, expected_path: Path) -> Path | None:
    if expected_path.exists():
        return expected_path
    csvs = sorted(output_dir.glob("*.csv"))
    if not csvs:
        return None
    exact = [path for path in csvs if path.stem == expected_path.stem]
    return exact[0] if exact else csvs[0]


def run_openface(row: dict[str, Any], config: Stage06Config) -> tuple[str, int, Path | None, str]:
    crop_video = Path(row["crop_video_path"])
    output_dir = Path(row["openface_output_dir"])
    expected_csv = Path(row["openface_csv_path"])
    if not crop_video.exists():
        return "error", -1, None, f"crop video not found: {crop_video}"
    if expected_csv.exists() and not config.overwrite:
        return "skipped_existing", 0, expected_csv, ""

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(config.openface_binary),
        "-f",
        str(crop_video),
        "-out_dir",
        str(output_dir),
        "-of",
        expected_csv.stem,
        "-gaze",
        "-pose",
    ]
    if config.openface_aus:
        cmd.append("-aus")
    if not config.save_aligned_faces:
        cmd.append("-q")

    cwd = config.openface_binary.parent.parent if config.openface_binary.parent.name == "bin" else config.openface_binary.parent
    result = subprocess.run(cmd, cwd=cwd, check=False, capture_output=True, text=True)
    csv_path = find_openface_csv(output_dir, expected_csv)
    if result.returncode == 0 and csv_path is not None:
        return "done", result.returncode, csv_path, ""
    note = (result.stderr or result.stdout or "").strip().replace("\n", " ")
    if csv_path is None and not note:
        note = "OpenFace did not produce a CSV"
    return "error", result.returncode, csv_path, note[-800:]


def stripped_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, str]] = []
        for row in reader:
            rows.append({str(k).strip(): v for k, v in row.items() if k is not None})
        return rows


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def stdev(values: list[float]) -> float | None:
    return statistics.pstdev(values) if len(values) > 1 else (0.0 if len(values) == 1 else None)


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def au_values(rows: list[dict[str, Any]], column: str) -> list[float]:
    values = [finite_float(raw.get(column)) for raw in rows]
    return [value for value in values if value is not None]


def expression_from_aus(au_r_means: dict[str, float | None], au_c_ratios: dict[str, float | None]) -> str:
    au12_r = au_r_means.get("AU12")
    au12_c = au_c_ratios.get("AU12")
    au04_r = au_r_means.get("AU04")
    au04_c = au_c_ratios.get("AU04")
    au25_c = au_c_ratios.get("AU25")
    au26_c = au_c_ratios.get("AU26")
    au45_c = au_c_ratios.get("AU45")

    if (au12_r is not None and au12_r >= 1.0) or (au12_c is not None and au12_c >= 0.5):
        return "smile"
    if (au04_r is not None and au04_r >= 1.0) or (au04_c is not None and au04_c >= 0.5):
        return "concern_or_frown"
    if (au25_c is not None and au25_c >= 0.5) or (au26_c is not None and au26_c >= 0.5):
        return "surprise_or_open_mouth"
    if au45_c is not None and au45_c >= 0.5:
        return "blink_or_eyes_closed"
    return "neutral_or_unknown"


def build_timebins_for_track(row: dict[str, Any], csv_path: Path, config: Stage06Config) -> list[dict[str, Any]]:
    rows = stripped_rows(csv_path)
    if not rows:
        return []
    crop_start = safe_float(row["crop_start_sec"])
    bins: dict[int, list[dict[str, Any]]] = {}
    for raw in rows:
        timestamp = finite_float(raw.get("timestamp"))
        if timestamp is None:
            frame = finite_float(raw.get("frame")) or 0.0
            timestamp = frame / 30.0
        shot_time = crop_start + timestamp
        bin_idx = int(math.floor(shot_time / config.timebin_sec))
        bins.setdefault(bin_idx, []).append(raw)

    out: list[dict[str, Any]] = []
    for bin_idx in sorted(bins):
        bin_rows = bins[bin_idx]
        valid_rows = [
            raw
            for raw in bin_rows
            if int(safe_float(raw.get("success"), 1)) == 1 and finite_float(raw.get("confidence")) is not None
        ]
        confs = [finite_float(raw.get("confidence")) for raw in valid_rows]
        conf_values = [value for value in confs if value is not None]
        gaze_x = [finite_float(raw.get("gaze_angle_x")) for raw in valid_rows]
        gaze_y = [finite_float(raw.get("gaze_angle_y")) for raw in valid_rows]
        gaze_x_values = [value for value in gaze_x if value is not None]
        gaze_y_values = [value for value in gaze_y if value is not None]

        pose_rows = [
            raw
            for raw in bin_rows
            if finite_float(raw.get("pose_Rx")) is not None
            and finite_float(raw.get("pose_Ry")) is not None
            and finite_float(raw.get("pose_Rz")) is not None
        ]
        pose_rx = [finite_float(raw.get("pose_Rx")) for raw in pose_rows]
        pose_ry = [finite_float(raw.get("pose_Ry")) for raw in pose_rows]
        pose_rz = [finite_float(raw.get("pose_Rz")) for raw in pose_rows]
        pose_rx_values = [value for value in pose_rx if value is not None]
        pose_ry_values = [value for value in pose_ry if value is not None]
        pose_rz_values = [value for value in pose_rz if value is not None]

        n_total = len(bin_rows)
        n_valid = len(valid_rows)
        valid_ratio = n_valid / n_total if n_total else 0.0
        pose_valid_ratio = len(pose_rows) / n_total if n_total else 0.0
        conf_mean = mean(conf_values)

        if n_valid >= 5 and valid_ratio >= 0.5 and conf_mean is not None and conf_mean >= 0.80:
            gaze_quality = "gaze_reliable"
        elif len(pose_rows) >= 5 and pose_valid_ratio >= 0.5:
            gaze_quality = "pose_fallback"
        else:
            gaze_quality = "unknown"

        au_r_means = {code: mean(au_values(bin_rows, f"{code}_r")) for code in AU_R_CODES}
        au_c_ratios = {code: mean(au_values(bin_rows, f"{code}_c")) for code in AU_C_CODES}
        expression_proxy = expression_from_aus(au_r_means, au_c_ratios) if config.expression_proxy else ""

        timebin_row = {
                "movie_id": row["movie_id"],
                "sequence_id": row["sequence_id"],
                "shot_id": row["shot_id"],
                "shot_idx": row["shot_idx"],
                "local_track_id": row["local_track_id"],
                "bin_idx": bin_idx,
                "bin_start_sec": f"{bin_idx * config.timebin_sec:.3f}",
                "bin_end_sec": f"{(bin_idx + 1) * config.timebin_sec:.3f}",
                "n_frames_total": n_total,
                "n_frames_valid": n_valid,
                "valid_ratio": f"{valid_ratio:.6f}",
                "confidence_mean": format_float(conf_mean),
                "confidence_median": format_float(median(conf_values)),
                "gaze_angle_x_mean": format_float(mean(gaze_x_values)),
                "gaze_angle_y_mean": format_float(mean(gaze_y_values)),
                "gaze_angle_x_std": format_float(stdev(gaze_x_values)),
                "gaze_angle_y_std": format_float(stdev(gaze_y_values)),
                "pose_Rx_mean": format_float(mean(pose_rx_values)),
                "pose_Ry_mean": format_float(mean(pose_ry_values)),
                "pose_Rz_mean": format_float(mean(pose_rz_values)),
                "pose_Rx_std": format_float(stdev(pose_rx_values)),
                "pose_Ry_std": format_float(stdev(pose_ry_values)),
                "pose_Rz_std": format_float(stdev(pose_rz_values)),
                "gaze_quality": gaze_quality,
                "expression_proxy": expression_proxy,
                "crop_video_path": row["crop_video_path"],
                "openface_csv_path": str(csv_path),
        }
        for code in AU_R_CODES:
            timebin_row[f"{code}_r_mean"] = format_float(au_r_means[code])
        for code in AU_C_CODES:
            timebin_row[f"{code}_c_ratio"] = format_float(au_c_ratios[code])
        out.append(timebin_row)
    return out


def file_size(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    return path.stat().st_size


def ensure_inputs(config: Stage06Config) -> None:
    if not config.manifest_csv.exists():
        raise FileNotFoundError(f"Stage04 manifest not found: {config.manifest_csv}")
    if not config.face_tracks_csv.exists():
        raise FileNotFoundError(f"Stage05 face tracks not found: {config.face_tracks_csv}")
    if not config.openface_binary.exists():
        raise FileNotFoundError(f"OpenFace binary not found: {config.openface_binary}")


def run(config: Stage06Config) -> None:
    ensure_inputs(config)
    if (
        not config.overwrite
        and config.track_manifest_csv.exists()
        and config.raw_index_csv.exists()
        and config.gaze_timebins_csv.exists()
    ):
        print(f"[Stage06] outputs already exist; use --overwrite to regenerate: {config.openface_dir}")
        return

    all_manifest_rows = [row for row in read_csv(config.manifest_csv) if row.get("movie_id") == config.movie_id]
    allowed_manifest_rows, skipped_stage_type_count, stage_counts = filter_manifest_rows_by_stage_type(
        all_manifest_rows, config.stage_type_include
    )
    allowed_shot_keys = {(row.get("sequence_id", ""), row.get("shot_id", "")) for row in allowed_manifest_rows}
    track_groups = filter_tracks(group_track_rows(read_csv(config.face_tracks_csv)), config, allowed_shot_keys)
    track_manifest_rows = build_track_manifest(track_groups, allowed_manifest_rows, config)

    raw_index_rows: list[dict[str, Any]] = []
    all_timebin_rows: list[dict[str, Any]] = []
    crop_success_count = 0
    openface_success_count = 0

    for row in track_manifest_rows:
        crop_status, crop_note = run_ffmpeg_crop(row, config)
        if crop_status in {"done", "skipped_existing"}:
            crop_success_count += 1
        openface_status = "skipped_crop_failed"
        returncode = -1
        csv_path: Path | None = None
        openface_note = ""

        if crop_status in {"done", "skipped_existing"}:
            openface_status, returncode, csv_path, openface_note = run_openface(row, config)
            if openface_status in {"done", "skipped_existing"} and csv_path is not None:
                openface_success_count += 1
                row["openface_csv_path"] = str(csv_path)
                all_timebin_rows.extend(build_timebins_for_track(row, csv_path, config))

        notes = [note for note in [crop_note, openface_note] if note]
        row["process_status"] = "done" if openface_status in {"done", "skipped_existing"} else "error"
        row["note"] = " | ".join(notes)
        raw_index_rows.append(
            {
                "movie_id": row["movie_id"],
                "sequence_id": row["sequence_id"],
                "shot_id": row["shot_id"],
                "shot_idx": row["shot_idx"],
                "local_track_id": row["local_track_id"],
                "crop_video_path": row["crop_video_path"],
                "openface_output_dir": row["openface_output_dir"],
                "openface_csv_path": str(csv_path or row["openface_csv_path"]),
                "crop_status": crop_status,
                "openface_status": openface_status,
                "openface_returncode": returncode,
                "crop_file_size": file_size(Path(row["crop_video_path"])),
                "openface_csv_size": file_size(csv_path),
                "note": row["note"],
            }
        )

    write_csv(config.track_manifest_csv, track_manifest_rows, TRACK_MANIFEST_COLUMNS)
    write_csv(config.raw_index_csv, raw_index_rows, RAW_INDEX_COLUMNS)
    write_csv(config.gaze_timebins_csv, all_timebin_rows, TIMEBIN_COLUMNS)

    status_counts: dict[str, int] = {}
    for row in track_manifest_rows:
        status = str(row.get("process_status") or "")
        status_counts[status] = status_counts.get(status, 0) + 1
    quality_counts: dict[str, int] = {}
    for row in all_timebin_rows:
        quality = str(row.get("gaze_quality") or "")
        quality_counts[quality] = quality_counts.get(quality, 0) + 1
    expression_counts: dict[str, int] = {}
    for row in all_timebin_rows:
        expression = str(row.get("expression_proxy") or "")
        expression_counts[expression] = expression_counts.get(expression, 0) + 1
    au_columns_present = sorted(
        {
            column
            for row in all_timebin_rows
            for column in [*AU_R_COLUMNS, *AU_C_COLUMNS]
            if row.get(column)
        }
    )

    payload = {
        "movie_id": config.movie_id,
        "requested_count": len(track_manifest_rows),
        "crop_success_count": crop_success_count,
        "openface_success_count": openface_success_count,
        "timebin_count": len(all_timebin_rows),
        "stage_type_include": stage_type_include_label(config.stage_type_include),
        "stage_type_counts": stage_counts,
        "skipped_stage_type_count": skipped_stage_type_count,
        "status_counts": status_counts,
        "gaze_quality_counts": quality_counts,
        "expression_proxy_counts": expression_counts,
        "au_columns_present": au_columns_present,
        "inputs": {
            "manifest_csv": str(config.manifest_csv),
            "face_tracks_csv": str(config.face_tracks_csv),
            "openface_binary": str(config.openface_binary),
        },
        "outputs": {
            "track_manifest_csv": str(config.track_manifest_csv),
            "raw_index_csv": str(config.raw_index_csv),
            "gaze_timebins_csv": str(config.gaze_timebins_csv),
            "summary_json": str(config.summary_json),
        },
    }
    write_json(config.summary_json, payload)

    print(f"[Stage06] movie_id={config.movie_id}")
    print(
        f"[Stage06] stage_type_include={stage_type_include_label(config.stage_type_include)} "
        f"skipped_stage_type={skipped_stage_type_count} stage_type_counts={json.dumps(stage_counts, sort_keys=True)}"
    )
    print(f"[Stage06] requested={len(track_manifest_rows)} crop_success={crop_success_count} openface_success={openface_success_count}")
    print(f"[Stage06] timebins={len(all_timebin_rows)} status_counts={json.dumps(status_counts, sort_keys=True)}")
    print(f"[Stage06] gaze_quality_counts={json.dumps(quality_counts, sort_keys=True)}")
    print(f"[Stage06] expression_proxy_counts={json.dumps(expression_counts, sort_keys=True)}")
    print(f"[Stage06] track_manifest_csv={config.track_manifest_csv}")
    print(f"[Stage06] raw_index_csv={config.raw_index_csv}")
    print(f"[Stage06] gaze_timebins_csv={config.gaze_timebins_csv}")
    print(f"[Stage06] summary_json={config.summary_json}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config")
    parser.add_argument("--movie-id")
    parser.add_argument("--manifest-csv")
    parser.add_argument("--face-tracks-csv")
    parser.add_argument("--ffmpeg-binary")
    parser.add_argument("--openface-binary")
    parser.add_argument("--crop-margin", type=float)
    parser.add_argument("--track-time-padding-sec", type=float)
    parser.add_argument("--min-track-length-frames", type=int)
    parser.add_argument("--min-track-conf", type=float)
    parser.add_argument("--timebin-sec", type=float)
    parser.add_argument("--max-tracks")
    parser.add_argument("--sequence-id-list")
    parser.add_argument("--shot-id-list")
    parser.add_argument("--track-id-list")
    parser.add_argument("--stage-type-include", help="Comma list of stage_type values to process, or 'all'.")
    parser.add_argument("--no-sface-gallery", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--save-aligned-faces", action="store_true")
    parser.add_argument("--no-openface-aus", action="store_true")
    parser.add_argument("--no-expression-proxy", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        run(make_config(args))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[Stage06] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
