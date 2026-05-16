"""Build the final subject-bin proxy table for downstream debugging and reranking."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from expregaze.video_proxy.stage_filter import (
    filter_manifest_rows_by_stage_type,
    resolve_stage_type_include,
    stage_type_include_label,
)


FINAL_COLUMNS = [
    "movie_id",
    "sequence_id",
    "shot_id",
    "shot_idx",
    "stage_type",
    "bin_idx",
    "bin_start_sec",
    "bin_end_sec",
    "subject_local_track_id",
    "subject_global_person_id",
    "subject_cast_pid",
    "identity_confidence",
    "identity_status",
    "gaze_quality",
    "gaze_direction_bucket",
    "pose_direction_bucket",
    "gaze_angle_x_mean",
    "gaze_angle_y_mean",
    "valid_ratio",
    "openface_confidence",
    "track_confidence",
    "candidate_count",
    "has_onscreen_person_candidate",
    "has_offscreen_person_candidate",
    "current_speaker_available",
    "candidate_list",
    "target_type",
    "target_id",
    "target_global_person_id",
    "proxy_confidence",
    "proxy_status",
    "proxy_source",
    "failure_reason",
    "top_score",
    "second_score",
    "score_margin",
]


@dataclass(frozen=True)
class Stage09Config:
    movie_id: str
    shot_manifest_csv: Path
    timebins_csv: Path
    face_tracks_csv: Path
    track_identity_csv: Path
    candidate_targets_csv: Path
    assignments_csv: Path
    final_proxy_dir: Path
    logs_dir: Path
    stage_type_include: set[str] | None
    overwrite: bool

    @property
    def final_proxy_csv(self) -> Path:
        return self.final_proxy_dir / "09_final_proxy_table.csv"

    @property
    def final_proxy_jsonl(self) -> Path:
        return self.final_proxy_dir / "09_final_proxy_table.jsonl"

    @property
    def summary_json(self) -> Path:
        return self.logs_dir / "09_build_final_proxy_table_summary.json"


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


def load_run_config(run_config_path: Path) -> tuple[dict[str, Any], Path]:
    run_config = load_yaml(run_config_path)
    paths_config_path = Path(run_config.get("inputs", {}).get("paths_config", "configs/path_local.yaml"))
    if not paths_config_path.is_absolute():
        paths_config_path = run_config_path.parent.parent.parent / paths_config_path
    paths_config = load_yaml(paths_config_path)
    project_root = Path(paths_config.get("project", {}).get("root", run_config_path.parent.parent.parent))
    return run_config, project_root


def make_config(args: argparse.Namespace) -> Stage09Config:
    run_config: dict[str, Any] = {}
    project_root = Path.cwd()
    if args.run_config:
        run_config, project_root = load_run_config(Path(args.run_config).resolve())

    movie_id = args.movie_id or run_config.get("data", {}).get("movie_id")
    if not movie_id:
        raise ValueError("movie_id is required via --movie-id or run config data.movie_id")
    movie_id = str(movie_id)

    outputs = run_config.get("outputs", {})
    selection = run_config.get("selection", {})
    stage = run_config.get("stages", {}).get("build_final_proxy_table", {})
    logs_dir = resolve_path(outputs.get("logs_dir") or f"outputs/video_proxy/{movie_id}/logs", project_root)
    openface_dir = resolve_path(outputs.get("openface_dir") or f"outputs/video_proxy/{movie_id}/openface", project_root)
    face_track_dir = resolve_path(outputs.get("face_track_dir") or f"outputs/video_proxy/{movie_id}/face_tracks", project_root)
    identity_dir = resolve_path(
        outputs.get("track_identity_dir") or f"outputs/video_proxy/{movie_id}/track_identities", project_root
    )
    proxy_gaze_dir = resolve_path(outputs.get("proxy_gaze_dir") or f"outputs/video_proxy/{movie_id}/proxy_gaze_scripts", project_root)
    final_proxy_dir = resolve_path(
        args.final_proxy_dir or outputs.get("final_proxy_dir") or f"outputs/video_proxy/{movie_id}/final_proxy",
        project_root,
    )
    assert (
        logs_dir is not None
        and openface_dir is not None
        and face_track_dir is not None
        and identity_dir is not None
        and proxy_gaze_dir is not None
        and final_proxy_dir is not None
    )

    return Stage09Config(
        movie_id=movie_id,
        shot_manifest_csv=resolve_path(args.shot_manifest_csv, project_root) or logs_dir / "04_shot_manifest.csv",
        timebins_csv=resolve_path(args.timebins_csv, project_root) or openface_dir / "06_gaze_timebins.csv",
        face_tracks_csv=resolve_path(args.face_tracks_csv, project_root) or face_track_dir / "05_face_tracks.csv",
        track_identity_csv=resolve_path(args.track_identity_csv, project_root) or identity_dir / "07_track_identity.csv",
        candidate_targets_csv=resolve_path(args.candidate_targets_csv, project_root)
        or proxy_gaze_dir / "08_candidate_targets.csv",
        assignments_csv=resolve_path(args.assignments_csv, project_root) or proxy_gaze_dir / "08_proxy_assignments.csv",
        final_proxy_dir=final_proxy_dir,
        logs_dir=logs_dir,
        stage_type_include=resolve_stage_type_include(args.stage_type_include, selection.get("stage_type_include")),
        overwrite=bool(args.overwrite or stage.get("overwrite", False)),
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed) or math.isinf(parsed):
        return default
    return parsed


def direction_bucket(value: Any, threshold: float = 0.20) -> str:
    parsed = safe_float(value, default=float("nan"))
    if math.isnan(parsed):
        return "unknown"
    if parsed < -threshold:
        return "left"
    if parsed > threshold:
        return "right"
    return "center"


def key4(row: dict[str, str], track_col: str = "local_track_id") -> tuple[str, str, str, str]:
    return (
        row.get("sequence_id", ""),
        row.get("shot_id", ""),
        row.get(track_col, ""),
        row.get("bin_idx", ""),
    )


def build_track_conf_lookup(face_tracks: list[dict[str, str]]) -> dict[tuple[str, str], str]:
    lookup: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in face_tracks:
        lookup[(row.get("shot_id", ""), row.get("local_track_id", ""))].append(safe_float(row.get("track_conf")))
    return {key: f"{sum(values) / len(values):.6f}" for key, values in lookup.items() if values}


def build_candidate_lists(candidates: list[dict[str, str]]) -> dict[tuple[str, str, str, str], str]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[
            (
                row.get("sequence_id", ""),
                row.get("shot_id", ""),
                row.get("subject_local_track_id", ""),
                row.get("bin_idx", ""),
            )
        ].append(
            {
                "candidate_type": row.get("candidate_type", ""),
                "candidate_id": row.get("candidate_id", ""),
                "candidate_global_person_id": row.get("candidate_global_person_id", ""),
                "candidate_cast_pid": row.get("candidate_cast_pid", ""),
                "candidate_side": row.get("candidate_side", ""),
                "total_score": safe_float(row.get("total_score")),
            }
        )
    return {key: json.dumps(value, ensure_ascii=False) for key, value in grouped.items()}


def build_candidate_summaries(candidates: list[dict[str, str]]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        grouped[
            (
                row.get("sequence_id", ""),
                row.get("shot_id", ""),
                row.get("subject_local_track_id", ""),
                row.get("bin_idx", ""),
            )
        ].append(row)
    summaries: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for key, rows in grouped.items():
        candidate_types = {row.get("candidate_type", "") for row in rows}
        summaries[key] = {
            "candidate_count": len(rows),
            "has_onscreen_person_candidate": "1" if "onscreen_local_track" in candidate_types else "0",
            "has_offscreen_person_candidate": "1"
            if {"current_speaker", "offscreen_participant"} & candidate_types
            else "0",
            "current_speaker_available": "1" if "current_speaker" in candidate_types else "0",
        }
    return summaries


def ensure_inputs(config: Stage09Config) -> None:
    for path in [
        config.shot_manifest_csv,
        config.timebins_csv,
        config.face_tracks_csv,
        config.track_identity_csv,
        config.candidate_targets_csv,
        config.assignments_csv,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required Stage09 input not found: {path}")


def run(config: Stage09Config) -> None:
    ensure_inputs(config)
    if not config.overwrite and config.final_proxy_csv.exists() and config.final_proxy_jsonl.exists():
        print(f"[Stage09] outputs already exist; use --overwrite to regenerate: {config.final_proxy_dir}")
        return

    raw_manifest = [row for row in read_csv(config.shot_manifest_csv) if row.get("movie_id") == config.movie_id]
    manifest_rows, skipped_stage_type_count, stage_counts = filter_manifest_rows_by_stage_type(
        raw_manifest, config.stage_type_include
    )
    allowed_shots = {row.get("shot_id", "") for row in manifest_rows}
    timebins = [row for row in read_csv(config.timebins_csv) if row.get("movie_id") == config.movie_id and row.get("shot_id", "") in allowed_shots]
    face_tracks = [
        row for row in read_csv(config.face_tracks_csv) if row.get("movie_id") == config.movie_id and row.get("shot_id", "") in allowed_shots
    ]
    identities = [
        row
        for row in read_csv(config.track_identity_csv)
        if row.get("movie_id") == config.movie_id and row.get("shot_id", "") in allowed_shots
    ]
    candidates = [
        row
        for row in read_csv(config.candidate_targets_csv)
        if row.get("movie_id") == config.movie_id and row.get("shot_id", "") in allowed_shots
    ]
    assignments = [
        row
        for row in read_csv(config.assignments_csv)
        if row.get("movie_id") == config.movie_id and row.get("shot_id", "") in allowed_shots
    ]

    identity_lookup = {(row.get("shot_id", ""), row.get("local_track_id", "")): row for row in identities}
    assignment_lookup = {key4(row): row for row in assignments}
    manifest_lookup = {row.get("shot_id", ""): row for row in manifest_rows}
    track_conf_lookup = build_track_conf_lookup(face_tracks)
    candidate_lists = build_candidate_lists(candidates)
    candidate_summaries = build_candidate_summaries(candidates)

    final_rows: list[dict[str, Any]] = []
    for tb in sorted(timebins, key=lambda row: (row.get("sequence_id", ""), row.get("shot_idx", ""), row.get("local_track_id", ""), safe_float(row.get("bin_start_sec")))):
        key = key4(tb)
        identity = identity_lookup.get((tb.get("shot_id", ""), tb.get("local_track_id", "")), {})
        assignment = assignment_lookup.get(key, {})
        manifest = manifest_lookup.get(tb.get("shot_id", ""), {})
        candidate_summary = candidate_summaries.get(
            key,
            {
                "candidate_count": 0,
                "has_onscreen_person_candidate": "0",
                "has_offscreen_person_candidate": "0",
                "current_speaker_available": "0",
            },
        )
        final_rows.append(
            {
                "movie_id": tb.get("movie_id", ""),
                "sequence_id": tb.get("sequence_id", ""),
                "shot_id": tb.get("shot_id", ""),
                "shot_idx": tb.get("shot_idx", ""),
                "stage_type": manifest.get("stage_type", ""),
                "bin_idx": tb.get("bin_idx", ""),
                "bin_start_sec": tb.get("bin_start_sec", ""),
                "bin_end_sec": tb.get("bin_end_sec", ""),
                "subject_local_track_id": tb.get("local_track_id", ""),
                "subject_global_person_id": assignment.get("subject_global_person_id") or identity.get("global_person_id", ""),
                "subject_cast_pid": assignment.get("subject_cast_pid") or identity.get("cast_pid", ""),
                "identity_confidence": assignment.get("subject_identity_confidence") or identity.get("identity_confidence", "0.000000"),
                "identity_status": identity.get("identity_status", "unknown") or "unknown",
                "gaze_quality": tb.get("gaze_quality", ""),
                "gaze_direction_bucket": assignment.get("gaze_direction_bucket") or direction_bucket(tb.get("gaze_angle_x_mean")),
                "pose_direction_bucket": assignment.get("pose_direction_bucket") or direction_bucket(tb.get("pose_Ry_mean"), 0.25),
                "gaze_angle_x_mean": tb.get("gaze_angle_x_mean", ""),
                "gaze_angle_y_mean": tb.get("gaze_angle_y_mean", ""),
                "valid_ratio": tb.get("valid_ratio", ""),
                "openface_confidence": tb.get("confidence_mean", ""),
                "track_confidence": identity.get("track_conf") or track_conf_lookup.get((tb.get("shot_id", ""), tb.get("local_track_id", "")), ""),
                "candidate_count": assignment.get("candidate_count") or candidate_summary["candidate_count"],
                "has_onscreen_person_candidate": candidate_summary["has_onscreen_person_candidate"],
                "has_offscreen_person_candidate": candidate_summary["has_offscreen_person_candidate"],
                "current_speaker_available": candidate_summary["current_speaker_available"],
                "candidate_list": candidate_lists.get(key, "[]"),
                "target_type": assignment.get("target_type", "unknown"),
                "target_id": assignment.get("target_id", "unknown"),
                "target_global_person_id": assignment.get("target_global_person_id", ""),
                "proxy_confidence": assignment.get("proxy_confidence", "0.000000"),
                "proxy_status": assignment.get("proxy_status", "unknown"),
                "proxy_source": assignment.get("proxy_source", ""),
                "failure_reason": assignment.get("failure_reason", "missing_assignment"),
                "top_score": assignment.get("top_score", "0.000000"),
                "second_score": assignment.get("second_score", "0.000000"),
                "score_margin": assignment.get("score_margin", "0.000000"),
            }
        )

    write_csv(config.final_proxy_csv, final_rows, FINAL_COLUMNS)
    write_jsonl(config.final_proxy_jsonl, final_rows)

    payload = {
        "movie_id": config.movie_id,
        "final_row_count": len(final_rows),
        "stage_type_include": stage_type_include_label(config.stage_type_include),
        "stage_type_counts": stage_counts,
        "skipped_stage_type_count": skipped_stage_type_count,
        "target_type_counts": dict(Counter(row.get("target_type", "") for row in final_rows)),
        "proxy_status_counts": dict(Counter(row.get("proxy_status", "") for row in final_rows)),
        "failure_reason_counts": dict(Counter(row.get("failure_reason", "") for row in final_rows)),
        "outputs": {
            "final_proxy_csv": str(config.final_proxy_csv),
            "final_proxy_jsonl": str(config.final_proxy_jsonl),
            "summary_json": str(config.summary_json),
        },
    }
    write_json(config.summary_json, payload)

    print(f"[Stage09] movie_id={config.movie_id}")
    print(
        f"[Stage09] stage_type_include={stage_type_include_label(config.stage_type_include)} "
        f"skipped_stage_type={skipped_stage_type_count} stage_type_counts={json.dumps(stage_counts, sort_keys=True)}"
    )
    print(f"[Stage09] final_rows={len(final_rows)} target_type_counts={json.dumps(payload['target_type_counts'], sort_keys=True)}")
    print(f"[Stage09] final_proxy_csv={config.final_proxy_csv}")
    print(f"[Stage09] summary_json={config.summary_json}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config")
    parser.add_argument("--movie-id")
    parser.add_argument("--shot-manifest-csv")
    parser.add_argument("--timebins-csv")
    parser.add_argument("--face-tracks-csv")
    parser.add_argument("--track-identity-csv")
    parser.add_argument("--candidate-targets-csv")
    parser.add_argument("--assignments-csv")
    parser.add_argument("--final-proxy-dir")
    parser.add_argument("--stage-type-include", help="Comma list of stage_type values to process, or 'all'.")
    parser.add_argument("--no-sface-gallery", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        run(make_config(args))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[Stage09] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
