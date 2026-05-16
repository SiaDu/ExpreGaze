"""Cut selected Stage02 candidate sequences into sequence and shot clips.

Stage04 is deliberately manifest-first: it reads the selected candidate
sequence JSONL, writes a shot-level manifest, and only then cuts videos unless
``--dry-run`` is requested. The old ``data/interim/shot_level`` folders are not
used as input because they can drift from the current Stage02 selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from expregaze.video_proxy.stage_filter import (
    filter_manifest_rows_by_stage_type,
    resolve_stage_type_include,
    stage_type_allowed,
    stage_type_include_label,
)


MANIFEST_COLUMNS = [
    "movie_id",
    "sequence_id",
    "shot_id",
    "shot_idx",
    "shot_start",
    "shot_end",
    "duration",
    "shot_start_hms",
    "shot_end_hms",
    "subtitle_text",
    "aligned_speakers",
    "cast_pids",
    "num_cast",
    "video_path",
    "sequence_video_path",
    "shot_clip_path",
    "stage_type",
    "process_status",
    "note",
]

VIDEO_SUFFIXES = {
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".m4v",
    ".webm",
    ".rmvb",
}


@dataclass(frozen=True)
class Stage04Config:
    movie_id: str
    candidate_sequences_jsonl: Path
    movie_video_path: Path | None
    movie_video_dir: Path
    sequence_video_dir: Path
    shot_clip_dir: Path
    manifest_csv: Path
    ffmpeg_binary: str
    padding_sec: float
    overwrite: bool
    dry_run: bool
    max_sequences: int | None
    sequence_id_list: list[str] | None
    stage_type_include: set[str] | None


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


def parse_sequence_id_list(value: Any) -> list[str] | None:
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
    raise ValueError(f"Unsupported sequence id list: {value!r}")


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


def make_config(args: argparse.Namespace) -> Stage04Config:
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
    stage = run_config.get("stages", {}).get("extract_video_sequences", {})
    selection = run_config.get("selection", {})
    text_inputs = run_config.get("inputs_from_text_pipeline", {})

    candidate_sequences_jsonl = resolve_path(
        args.candidate_sequences_jsonl
        or text_inputs.get("candidate_sequences_jsonl")
        or f"data/processed/candidate_sequences/{movie_id}__candidate_sequences.jsonl",
        project_root,
    )
    assert candidate_sequences_jsonl is not None

    movie_video_dir = resolve_path(
        args.movie_video_dir
        or paths_config.get("paths", {}).get("movie_video_dir")
        or "data/raw/Movie",
        project_root,
    )
    assert movie_video_dir is not None

    sequence_video_dir = resolve_path(
        args.sequence_video_dir
        or outputs.get("sequence_video_dir")
        or f"outputs/video_proxy/{movie_id}/sequence_videos",
        project_root,
    )
    assert sequence_video_dir is not None

    shot_clip_dir = resolve_path(
        args.shot_clip_dir
        or outputs.get("shot_clip_dir")
        or f"outputs/video_proxy/{movie_id}/shot_clips",
        project_root,
    )
    assert shot_clip_dir is not None

    logs_dir = resolve_path(outputs.get("logs_dir") or f"outputs/video_proxy/{movie_id}/logs", project_root)
    assert logs_dir is not None
    manifest_csv = resolve_path(args.manifest_csv, project_root) or logs_dir / "04_shot_manifest.csv"

    ffmpeg_binary = args.ffmpeg_binary or paths_config.get("openface", {}).get("ffmpeg_binary") or "ffmpeg"
    padding_sec = float(args.padding_sec if args.padding_sec is not None else stage.get("padding_sec", 0.2))
    overwrite = bool(args.overwrite or stage.get("overwrite", False))
    dry_run = bool(args.dry_run)
    max_sequences = parse_optional_int(
        args.max_sequences if args.max_sequences is not None else stage.get("max_sequences"),
        "--max-sequences",
    )
    sequence_id_list = parse_sequence_id_list(args.sequence_id_list)
    if sequence_id_list is None:
        sequence_id_list = parse_sequence_id_list(stage.get("sequence_id_list"))
    if sequence_id_list is None and selection.get("mode") == "manual":
        sequence_id_list = parse_sequence_id_list(selection.get("sequence_ids"))

    movie_video_path = resolve_path(args.movie_video, project_root)

    return Stage04Config(
        movie_id=str(movie_id),
        candidate_sequences_jsonl=candidate_sequences_jsonl,
        movie_video_path=movie_video_path,
        movie_video_dir=movie_video_dir,
        sequence_video_dir=sequence_video_dir,
        shot_clip_dir=shot_clip_dir,
        manifest_csv=manifest_csv,
        ffmpeg_binary=str(ffmpeg_binary),
        padding_sec=max(0.0, padding_sec),
        overwrite=overwrite,
        dry_run=dry_run,
        max_sequences=max_sequences,
        sequence_id_list=sequence_id_list,
        stage_type_include=resolve_stage_type_include(args.stage_type_include, selection.get("stage_type_include")),
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def filter_sequences(sequences: list[dict[str, Any]], config: Stage04Config) -> list[dict[str, Any]]:
    selected = sequences
    if config.sequence_id_list:
        wanted = set(config.sequence_id_list)
        selected = [seq for seq in sequences if str(seq.get("sequence_id")) in wanted]
        missing = sorted(wanted - {str(seq.get("sequence_id")) for seq in selected})
        if missing:
            raise ValueError(f"Sequence ids not found in candidate JSONL: {missing}")
    if config.max_sequences is not None:
        selected = selected[: config.max_sequences]
    if not selected:
        raise ValueError("No candidate sequences selected")
    return selected


def find_movie_video(config: Stage04Config) -> Path:
    if config.movie_video_path is not None:
        if not config.movie_video_path.exists():
            raise FileNotFoundError(f"Movie video does not exist: {config.movie_video_path}")
        return config.movie_video_path

    if not config.movie_video_dir.exists():
        raise FileNotFoundError(f"Movie video dir does not exist: {config.movie_video_dir}")

    candidates = [
        path
        for path in sorted(config.movie_video_dir.glob(f"{config.movie_id}*"))
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    ]
    if not candidates:
        raise FileNotFoundError(f"No video matching {config.movie_id}* in {config.movie_video_dir}")

    exact_stem = [path for path in candidates if path.stem == config.movie_id]
    if exact_stem:
        return sorted(exact_stem, key=lambda p: (len(p.name), p.name))[0]
    return sorted(candidates, key=lambda p: (len(p.name), p.name))[0]


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed):
        return default
    return parsed


def as_json_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def subtitle_nonempty(shot: dict[str, Any]) -> bool:
    return bool(str(shot.get("subtitle_text") or "").strip())


def stage_type_for_shot(shot: dict[str, Any]) -> str:
    num_cast = int(safe_float(shot.get("num_cast"), 0.0))
    has_subtitle = subtitle_nonempty(shot)
    if num_cast <= 1 and has_subtitle:
        return "single_speaking"
    if num_cast == 2 and has_subtitle:
        return "two_person_dialogue_simple"
    if num_cast >= 3:
        return "multi_person"
    return "unknown"


def build_manifest(
    sequences: list[dict[str, Any]],
    video_path: Path,
    config: Stage04Config,
    initial_status: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sequence in sequences:
        sequence_id = str(sequence.get("sequence_id"))
        sequence_video_path = config.sequence_video_dir / f"{sequence_id}.mp4"
        for shot in sequence.get("shots", []):
            shot_id = str(shot.get("shot_id") or f"shot_{int(safe_float(shot.get('shot_idx'))):04d}")
            shot_start = safe_float(shot.get("shot_start_time", shot.get("shot_start")))
            shot_end = safe_float(shot.get("shot_end_time", shot.get("shot_end")))
            duration = max(0.0, shot_end - shot_start)
            rows.append(
                {
                    "movie_id": config.movie_id,
                    "sequence_id": sequence_id,
                    "shot_id": shot_id,
                    "shot_idx": int(safe_float(shot.get("shot_idx"), -1)),
                    "shot_start": f"{shot_start:.3f}",
                    "shot_end": f"{shot_end:.3f}",
                    "duration": f"{duration:.3f}",
                    "shot_start_hms": shot.get("shot_start_time_hms") or "",
                    "shot_end_hms": shot.get("shot_end_time_hms") or "",
                    "subtitle_text": shot.get("subtitle_text") or "",
                    "aligned_speakers": as_json_cell(shot.get("aligned_speakers")),
                    "cast_pids": as_json_cell(shot.get("cast_pids")),
                    "num_cast": int(safe_float(shot.get("num_cast"), 0.0)),
                    "video_path": str(video_path),
                    "sequence_video_path": str(sequence_video_path),
                    "shot_clip_path": str(config.shot_clip_dir / sequence_id / f"{shot_id}.mp4"),
                    "stage_type": stage_type_for_shot(shot),
                    "process_status": initial_status,
                    "note": "",
                }
            )
    return rows


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def ffmpeg_cut(
    ffmpeg_binary: str,
    source: Path,
    target: Path,
    start_sec: float,
    duration_sec: float,
    overwrite: bool,
) -> tuple[str, str]:
    if duration_sec <= 0:
        return "error", "non-positive duration"
    if target.exists() and not overwrite:
        return "skipped_existing", ""

    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-ss",
        f"{max(0.0, start_sec):.3f}",
        "-i",
        str(source),
        "-t",
        f"{duration_sec:.3f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(target),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode == 0:
        return "done", ""
    note = (result.stderr or result.stdout or "").strip().replace("\n", " ")
    return "error", note[-500:]


def cut_sequence_videos(
    sequences: list[dict[str, Any]],
    video_path: Path,
    config: Stage04Config,
) -> dict[str, tuple[str, str]]:
    results: dict[str, tuple[str, str]] = {}
    for sequence in sequences:
        sequence_id = str(sequence.get("sequence_id"))
        start = max(0.0, safe_float(sequence.get("start_time_sec")) - config.padding_sec)
        end = safe_float(sequence.get("end_time_sec")) + config.padding_sec
        target = config.sequence_video_dir / f"{sequence_id}.mp4"
        results[sequence_id] = ffmpeg_cut(
            config.ffmpeg_binary,
            video_path,
            target,
            start,
            max(0.0, end - start),
            config.overwrite,
        )
    return results


def cut_shot_clips(rows: list[dict[str, Any]], video_path: Path, config: Stage04Config) -> None:
    for row in rows:
        if not stage_type_allowed(row, config.stage_type_include):
            row["process_status"] = "skipped_stage_type"
            row["note"] = f"stage_type not included: {row.get('stage_type', '')}"
            continue
        status, note = ffmpeg_cut(
            config.ffmpeg_binary,
            video_path,
            Path(row["shot_clip_path"]),
            safe_float(row["shot_start"]),
            safe_float(row["duration"]),
            config.overwrite,
        )
        row["process_status"] = status
        row["note"] = note


def summarize(rows: list[dict[str, Any]]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for row in rows:
        status = str(row.get("process_status") or "")
        summary[status] = summary.get(status, 0) + 1
    return summary


def run(config: Stage04Config) -> None:
    if not config.candidate_sequences_jsonl.exists():
        raise FileNotFoundError(f"Candidate JSONL does not exist: {config.candidate_sequences_jsonl}")

    sequences = filter_sequences(read_jsonl(config.candidate_sequences_jsonl), config)
    video_path = find_movie_video(config)
    initial_status = "dry_run" if config.dry_run else "pending"
    rows = build_manifest(sequences, video_path, config, initial_status=initial_status)
    selected_rows, skipped_stage_type_count, stage_counts = filter_manifest_rows_by_stage_type(rows, config.stage_type_include)

    if config.dry_run:
        for row in rows:
            if not stage_type_allowed(row, config.stage_type_include):
                row["process_status"] = "skipped_stage_type"
                row["note"] = f"stage_type not included: {row.get('stage_type', '')}"
        write_manifest(config.manifest_csv, rows)
    else:
        sequence_results = cut_sequence_videos(sequences, video_path, config)
        cut_shot_clips(rows, video_path, config)
        for row in rows:
            seq_status, seq_note = sequence_results.get(row["sequence_id"], ("unknown", ""))
            if seq_status == "error" and not row["note"]:
                row["note"] = f"sequence_video_error: {seq_note}"
        write_manifest(config.manifest_csv, rows)

    selected_sequence_ids = [str(seq.get("sequence_id")) for seq in sequences]
    status_counts = summarize(rows)
    print(f"[Stage04] movie_id={config.movie_id}")
    print(f"[Stage04] candidate_jsonl={config.candidate_sequences_jsonl}")
    print(f"[Stage04] video_path={video_path}")
    print(f"[Stage04] selected_sequences={len(sequences)} {selected_sequence_ids}")
    print(f"[Stage04] manifest_rows={len(rows)}")
    print(
        f"[Stage04] stage_type_include={stage_type_include_label(config.stage_type_include)} "
        f"included_rows={len(selected_rows)} skipped_stage_type={skipped_stage_type_count} "
        f"stage_type_counts={json.dumps(stage_counts, sort_keys=True)}"
    )
    print(f"[Stage04] manifest_csv={config.manifest_csv}")
    print(f"[Stage04] sequence_video_dir={config.sequence_video_dir}")
    print(f"[Stage04] shot_clip_dir={config.shot_clip_dir}")
    print(f"[Stage04] dry_run={config.dry_run} overwrite={config.overwrite}")
    print(f"[Stage04] status_counts={json.dumps(status_counts, sort_keys=True)}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", help="Run YAML, e.g. configs/runs/video_proxy_tt0032138.yaml")
    parser.add_argument("--movie-id")
    parser.add_argument("--candidate-sequences-jsonl")
    parser.add_argument("--movie-video")
    parser.add_argument("--movie-video-dir")
    parser.add_argument("--sequence-video-dir")
    parser.add_argument("--shot-clip-dir")
    parser.add_argument("--manifest-csv")
    parser.add_argument("--ffmpeg-binary")
    parser.add_argument("--padding-sec", type=float)
    parser.add_argument("--max-sequences")
    parser.add_argument("--sequence-id-list")
    parser.add_argument("--stage-type-include", help="Comma list of stage_type values to process, or 'all'.")
    parser.add_argument("--no-sface-gallery", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run(make_config(args))


if __name__ == "__main__":
    main()
