"""Build identity-aware proxy gaze targets from OpenFace timebins."""

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


CANDIDATE_COLUMNS = [
    "movie_id",
    "sequence_id",
    "shot_id",
    "shot_idx",
    "subject_local_track_id",
    "bin_idx",
    "bin_start_sec",
    "bin_end_sec",
    "gaze_quality",
    "gaze_direction_bucket",
    "pose_direction_bucket",
    "candidate_type",
    "candidate_id",
    "candidate_local_track_id",
    "candidate_global_person_id",
    "candidate_cast_pid",
    "candidate_identity_confidence",
    "candidate_side",
    "direction_score",
    "pose_score",
    "identity_score",
    "dialogue_score",
    "quality_score",
    "uncertainty_penalty",
    "total_score",
    "score_note",
]

ASSIGNMENT_COLUMNS = [
    "movie_id",
    "sequence_id",
    "shot_id",
    "shot_idx",
    "local_track_id",
    "bin_idx",
    "bin_start_sec",
    "bin_end_sec",
    "gaze_quality",
    "gaze_direction_bucket",
    "pose_direction_bucket",
    "subject_global_person_id",
    "subject_cast_pid",
    "subject_identity_confidence",
    "target_type",
    "target_id",
    "target_global_person_id",
    "target_cast_pid",
    "target_identity_confidence",
    "proxy_confidence",
    "proxy_status",
    "proxy_source",
    "failure_reason",
    "raw_proxy_status",
    "raw_failure_reason",
    "smoothing_applied",
    "top_score",
    "second_score",
    "score_margin",
    "candidate_count",
    "prev_target_id",
    "next_target_id",
    "subtitle_text",
    "aligned_speakers",
    "active_speakers",
]

CANDIDATE_TYPE_PRIORITY = {
    "onscreen_local_track": 5,
    "current_speaker": 4,
    "offscreen_participant": 3,
    "offscreen_place_or_away": 2,
    "down_self_or_think": 1,
}


@dataclass(frozen=True)
class Stage08Config:
    movie_id: str
    timebins_csv: Path
    face_tracks_csv: Path
    shot_manifest_csv: Path
    candidate_sequences_jsonl: Path
    track_identity_csv: Path
    proxy_gaze_dir: Path
    logs_dir: Path
    direction_threshold: float
    pose_direction_threshold: float
    min_proxy_score: float
    ambiguous_margin: float
    require_gaze_quality: bool
    include_offscreen_participants: bool
    include_current_speaker: bool
    high_precision: bool
    stage_type_include: set[str] | None
    overwrite: bool

    @property
    def candidate_targets_csv(self) -> Path:
        return self.proxy_gaze_dir / "08_candidate_targets.csv"

    @property
    def assignments_csv(self) -> Path:
        return self.proxy_gaze_dir / "08_proxy_assignments.csv"

    @property
    def assignments_jsonl(self) -> Path:
        return self.proxy_gaze_dir / "08_proxy_assignments.jsonl"

    @property
    def sequence_packages_jsonl(self) -> Path:
        return self.proxy_gaze_dir / "08_proxy_sequence_packages.jsonl"

    @property
    def summary_json(self) -> Path:
        return self.logs_dir / "08_build_proxy_gaze_script_summary.json"


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


def load_run_config(run_config_path: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    run_config = load_yaml(run_config_path)
    paths_config_path = Path(run_config.get("inputs", {}).get("paths_config", "configs/path_local.yaml"))
    if not paths_config_path.is_absolute():
        paths_config_path = run_config_path.parent.parent.parent / paths_config_path
    paths_config = load_yaml(paths_config_path)
    project_root = Path(paths_config.get("project", {}).get("root", run_config_path.parent.parent.parent))
    return run_config, paths_config, project_root


def make_config(args: argparse.Namespace) -> Stage08Config:
    run_config: dict[str, Any] = {}
    project_root = Path.cwd()
    if args.run_config:
        run_config_path = Path(args.run_config).resolve()
        run_config, _, project_root = load_run_config(run_config_path)

    movie_id = args.movie_id or run_config.get("data", {}).get("movie_id")
    if not movie_id:
        raise ValueError("movie_id is required via --movie-id or run config data.movie_id")

    outputs = run_config.get("outputs", {})
    text_inputs = run_config.get("inputs_from_text_pipeline", {})
    selection = run_config.get("selection", {})
    stage = run_config.get("stages", {}).get("build_proxy_gaze_script", {})

    openface_dir = resolve_path(outputs.get("openface_dir") or f"outputs/video_proxy/{movie_id}/openface", project_root)
    face_track_dir = resolve_path(outputs.get("face_track_dir") or f"outputs/video_proxy/{movie_id}/face_tracks", project_root)
    identity_dir = resolve_path(
        outputs.get("track_identity_dir") or f"outputs/video_proxy/{movie_id}/track_identities", project_root
    )
    proxy_gaze_dir = resolve_path(outputs.get("proxy_gaze_dir") or f"outputs/video_proxy/{movie_id}/proxy_gaze_scripts", project_root)
    logs_dir = resolve_path(outputs.get("logs_dir") or f"outputs/video_proxy/{movie_id}/logs", project_root)
    assert (
        openface_dir is not None
        and face_track_dir is not None
        and identity_dir is not None
        and proxy_gaze_dir is not None
        and logs_dir is not None
    )

    return Stage08Config(
        movie_id=str(movie_id),
        timebins_csv=resolve_path(args.timebins_csv, project_root) or openface_dir / "06_gaze_timebins.csv",
        face_tracks_csv=resolve_path(args.face_tracks_csv, project_root) or face_track_dir / "05_face_tracks.csv",
        shot_manifest_csv=resolve_path(args.shot_manifest_csv, project_root) or logs_dir / "04_shot_manifest.csv",
        candidate_sequences_jsonl=resolve_path(
            args.candidate_sequences_jsonl
            or text_inputs.get("candidate_sequences_jsonl")
            or f"data/processed/candidate_sequences/{movie_id}__candidate_sequences.jsonl",
            project_root,
        )
        or Path(),
        track_identity_csv=resolve_path(args.track_identity_csv, project_root) or identity_dir / "07_track_identity.csv",
        proxy_gaze_dir=proxy_gaze_dir,
        logs_dir=logs_dir,
        direction_threshold=float(
            args.direction_threshold if args.direction_threshold is not None else stage.get("direction_threshold", 0.20)
        ),
        pose_direction_threshold=float(
            args.pose_direction_threshold
            if args.pose_direction_threshold is not None
            else stage.get("pose_direction_threshold", 0.25)
        ),
        min_proxy_score=float(args.min_proxy_score if args.min_proxy_score is not None else stage.get("min_proxy_score", 0.55)),
        ambiguous_margin=float(
            args.ambiguous_margin if args.ambiguous_margin is not None else stage.get("ambiguous_margin", 0.15)
        ),
        require_gaze_quality=bool(stage.get("require_gaze_quality", True)) if not args.no_require_gaze_quality else False,
        include_offscreen_participants=bool(stage.get("include_offscreen_participants", True)),
        include_current_speaker=bool(stage.get("include_current_speaker", True)),
        high_precision=bool(stage.get("high_precision", True)),
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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
    if math.isnan(parsed) or math.isinf(parsed):
        return default
    return parsed


def parse_json_cell(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = value
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item)]
    if isinstance(parsed, str):
        return [item.strip() for item in parsed.split("|") if item.strip()]
    return []


def load_sequences(path: Path) -> dict[str, dict[str, Any]]:
    sequences: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seq = json.loads(line)
            sequences[str(seq.get("sequence_id"))] = seq
    return sequences


def direction_bucket(value: Any, threshold: float) -> str:
    parsed = safe_float(value, default=float("nan"))
    if math.isnan(parsed):
        return "unknown"
    if parsed < -threshold:
        return "left"
    if parsed > threshold:
        return "right"
    return "center"


def mean_bbox_for_bin(track_rows: list[dict[str, str]], bin_start: float, bin_end: float) -> dict[str, float] | None:
    rows = [row for row in track_rows if bin_start <= safe_float(row.get("timestamp_sec")) < bin_end]
    if not rows:
        return None
    keys = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "bbox_cx", "bbox_cy"]
    return {key: sum(safe_float(row.get(key)) for row in rows) / len(rows) for key in keys}


def build_track_index(face_tracks: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    index: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in face_tracks:
        index[(row.get("shot_id", ""), row.get("local_track_id", ""))].append(row)
    for key in index:
        index[key].sort(key=lambda row: safe_float(row.get("timestamp_sec")))
    return index


def build_identity_lookup(track_identity_rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row.get("shot_id", ""), row.get("local_track_id", "")): row for row in track_identity_rows}


def shot_contexts(
    shot_manifest: list[dict[str, str]],
    sequences: dict[str, dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    contexts: dict[tuple[str, str], dict[str, Any]] = {}
    for row in shot_manifest:
        sequence_id = row.get("sequence_id", "")
        shot_id = row.get("shot_id", "")
        seq = sequences.get(sequence_id, {})
        active_speakers = [str(item) for item in seq.get("active_speakers", []) if str(item)]
        contexts[(sequence_id, shot_id)] = {
            "subtitle_text": row.get("subtitle_text", ""),
            "aligned_speakers": parse_json_cell(row.get("aligned_speakers")),
            "active_speakers": active_speakers,
            "num_cast": row.get("num_cast", ""),
            "stage_type": row.get("stage_type", ""),
            "shot_idx": row.get("shot_idx", ""),
            "cast_pids": parse_json_cell(row.get("cast_pids")),
        }
    return contexts


def candidate_side(subject_bbox: dict[str, float] | None, target_bbox: dict[str, float] | None) -> str:
    if subject_bbox is None or target_bbox is None:
        return "unknown"
    dx = target_bbox["bbox_cx"] - subject_bbox["bbox_cx"]
    if dx < -1.0:
        return "left"
    if dx > 1.0:
        return "right"
    return "center"


def side_match_score(direction: str, side: str, match_value: float, conflict_value: float, center_value: float = 0.0) -> float:
    if direction == "unknown" or side == "unknown":
        return 0.0
    if direction == "center" or side == "center":
        return center_value
    return match_value if direction == side else conflict_value


def normalize_label(value: Any) -> str:
    return "".join(ch for ch in str(value or "").upper() if ch.isalnum())


def is_single_visible_context(context: dict[str, Any], visible_track_count: int) -> bool:
    stage_type = str(context.get("stage_type", ""))
    return stage_type == "single_speaking" or visible_track_count <= 1


def build_reverse_shot_contexts(
    contexts: dict[tuple[str, str], dict[str, Any]],
    tracks_by_shot: dict[str, set[str]],
    identity_lookup: dict[tuple[str, str], dict[str, str]],
) -> dict[tuple[str, str], set[str]]:
    shots_by_sequence: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for (sequence_id, shot_id), context in contexts.items():
        shots_by_sequence[sequence_id].append((safe_float(context.get("shot_idx")), shot_id))

    visible_labels_by_shot: dict[str, set[str]] = defaultdict(set)
    for shot_id, track_ids in tracks_by_shot.items():
        for local_track_id in track_ids:
            identity = identity_lookup.get((shot_id, local_track_id), {})
            if safe_float(identity.get("identity_confidence")) < 0.60:
                continue
            for value in [identity.get("cast_pid"), identity.get("character_name"), identity.get("cast_name")]:
                label = normalize_label(value)
                if label:
                    visible_labels_by_shot[shot_id].add(label)

    reverse_contexts: dict[tuple[str, str], set[str]] = defaultdict(set)
    for sequence_id, items in shots_by_sequence.items():
        ordered = [shot_id for _, shot_id in sorted(items)]
        for idx, shot_id in enumerate(ordered):
            labels: set[str] = set()
            for neighbor_idx in [idx - 1, idx + 1]:
                if 0 <= neighbor_idx < len(ordered):
                    labels.update(visible_labels_by_shot.get(ordered[neighbor_idx], set()))
            reverse_contexts[(sequence_id, shot_id)] = labels
    return reverse_contexts


def score_candidate(
    timebin: dict[str, str],
    candidate_type: str,
    candidate_side_value: str,
    candidate_identity_confidence: float,
    gaze_direction: str,
    pose_direction: str,
    config: Stage08Config,
    context: dict[str, Any],
    subject_identity: dict[str, str],
    visible_track_count: int,
    reverse_shot_labels: set[str],
) -> tuple[float, float, float, float, float, float, str]:
    direction_score = 0.0
    pose_score = 0.0
    identity_score = 0.0
    dialogue_score = 0.0
    quality_score = 0.0
    uncertainty_penalty = 0.0
    notes: list[str] = []

    gaze_quality = timebin.get("gaze_quality", "")
    if gaze_quality == "gaze_reliable":
        quality_score += 0.15
    elif gaze_quality == "pose_fallback":
        quality_score += 0.05
        uncertainty_penalty += 0.05
    elif gaze_quality == "unknown":
        uncertainty_penalty += 0.5

    stage_type = str(context.get("stage_type", ""))
    active_labels = {normalize_label(item) for item in context.get("active_speakers", [])}
    aligned_labels = {normalize_label(item) for item in context.get("aligned_speakers", [])}
    candidate_label = normalize_label(timebin.get("_candidate_id", ""))
    subject_labels = {
        normalize_label(subject_identity.get("cast_pid")),
        normalize_label(subject_identity.get("character_name")),
        normalize_label(subject_identity.get("cast_name")),
    }

    if candidate_type == "onscreen_local_track":
        side_matches_gaze = gaze_direction not in {"unknown", "center"} and gaze_direction == candidate_side_value
        direction_score += side_match_score(gaze_direction, candidate_side_value, 0.45, -0.30, 0.05)
        pose_score += side_match_score(pose_direction, candidate_side_value, 0.25, -0.15, 0.03)
        if stage_type == "two_person_dialogue_simple" and side_matches_gaze:
            direction_score += 0.15
            notes.append("two_person_visible_direction_prior")
        if candidate_identity_confidence >= 0.60:
            identity_score += 0.08
        elif candidate_identity_confidence > 0.0:
            uncertainty_penalty += 0.04
        notes.append(f"onscreen_side={candidate_side_value}")
    elif candidate_type == "current_speaker":
        dialogue_score += 0.35
        if gaze_direction != "center":
            direction_score += 0.10
        if is_single_visible_context(context, visible_track_count) and candidate_label in subject_labels:
            dialogue_score -= 0.22
            notes.append("single_visible_speaker_is_subject")
        notes.append("speaker_prior")
    elif candidate_type == "offscreen_participant":
        dialogue_score += 0.20
        if gaze_direction != "center":
            direction_score += 0.10
        if is_single_visible_context(context, visible_track_count) and candidate_label not in aligned_labels:
            dialogue_score += 0.25
            notes.append("single_visible_interlocutor_prior")
        if candidate_label in reverse_shot_labels:
            dialogue_score += 0.18
            notes.append("reverse_shot_prior")
        if candidate_label in active_labels:
            dialogue_score += 0.05
            notes.append("active_participant_prior")
        notes.append("participant_prior")
    elif candidate_type == "down_self_or_think":
        gaze_y = safe_float(timebin.get("gaze_angle_y_mean"), default=float("nan"))
        if not math.isnan(gaze_y) and gaze_y > config.direction_threshold:
            direction_score += 0.25
            notes.append("positive_gaze_y")
        dialogue_score += 0.05
    elif candidate_type == "offscreen_place_or_away":
        if gaze_direction in {"left", "right"} or pose_direction in {"left", "right"}:
            direction_score += 0.20
            notes.append("lateral_away")
        dialogue_score += 0.05
    elif candidate_type == "unknown":
        uncertainty_penalty = 0.0

    total = max(0.0, min(1.0, direction_score + pose_score + identity_score + dialogue_score + quality_score - uncertainty_penalty))
    return direction_score, pose_score, identity_score, dialogue_score, quality_score, uncertainty_penalty, ";".join(notes), total


def build_candidates(
    timebins: list[dict[str, str]],
    track_index: dict[tuple[str, str], list[dict[str, str]]],
    identity_lookup: dict[tuple[str, str], dict[str, str]],
    contexts: dict[tuple[str, str], dict[str, Any]],
    config: Stage08Config,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    tracks_by_shot: dict[str, set[str]] = defaultdict(set)
    for shot_id, local_track_id in track_index:
        tracks_by_shot[shot_id].add(local_track_id)
    reverse_contexts = build_reverse_shot_contexts(contexts, tracks_by_shot, identity_lookup)

    for tb in timebins:
        if tb.get("movie_id") != config.movie_id:
            continue
        shot_id = tb.get("shot_id", "")
        subject_id = tb.get("local_track_id", "")
        bin_start = safe_float(tb.get("bin_start_sec"))
        bin_end = safe_float(tb.get("bin_end_sec"))
        subject_bbox = mean_bbox_for_bin(track_index.get((shot_id, subject_id), []), bin_start, bin_end)
        context = contexts.get((tb.get("sequence_id", ""), shot_id), {})
        subject_identity = identity_lookup.get((shot_id, subject_id), {})
        visible_track_count = len(tracks_by_shot.get(shot_id, set()))
        reverse_shot_labels = reverse_contexts.get((tb.get("sequence_id", ""), shot_id), set())
        gaze_direction = direction_bucket(tb.get("gaze_angle_x_mean"), config.direction_threshold)
        pose_direction = direction_bucket(tb.get("pose_Ry_mean"), config.pose_direction_threshold)

        candidate_specs: list[tuple[str, str, str, dict[str, str]]] = []
        for other_track_id in sorted(tracks_by_shot.get(shot_id, set()) - {subject_id}):
            other_bbox = mean_bbox_for_bin(track_index.get((shot_id, other_track_id), []), bin_start, bin_end)
            candidate_specs.append(
                (
                    "onscreen_local_track",
                    other_track_id,
                    candidate_side(subject_bbox, other_bbox),
                    identity_lookup.get((shot_id, other_track_id), {}),
                )
            )

        if config.include_current_speaker:
            for speaker in context.get("aligned_speakers", []):
                candidate_specs.append(("current_speaker", speaker, "offscreen", {}))

        if config.include_offscreen_participants:
            onscreen_like = set(tracks_by_shot.get(shot_id, set()))
            for speaker in context.get("active_speakers", []):
                if speaker and speaker not in onscreen_like:
                    candidate_specs.append(("offscreen_participant", speaker, "offscreen", {}))

        candidate_specs.extend(
            [
                ("down_self_or_think", "down_self_or_think", "self", {}),
                ("offscreen_place_or_away", "offscreen_place_or_away", "offscreen", {}),
                ("unknown", "unknown", "unknown", {}),
            ]
        )

        seen: set[tuple[str, str]] = set()
        for candidate_type, candidate_id, side, identity in candidate_specs:
            key = (candidate_type, candidate_id)
            if key in seen:
                continue
            seen.add(key)
            candidate_identity_confidence = safe_float(identity.get("identity_confidence"))
            score_timebin = dict(tb)
            score_timebin["_candidate_id"] = candidate_id
            direction_score, pose_score, identity_score, dialogue_score, quality_score, uncertainty_penalty, note, total = score_candidate(
                score_timebin,
                candidate_type,
                side,
                candidate_identity_confidence,
                gaze_direction,
                pose_direction,
                config,
                context,
                subject_identity,
                visible_track_count,
                reverse_shot_labels,
            )
            candidates.append(
                {
                    "movie_id": tb.get("movie_id", ""),
                    "sequence_id": tb.get("sequence_id", ""),
                    "shot_id": shot_id,
                    "shot_idx": tb.get("shot_idx", ""),
                    "subject_local_track_id": subject_id,
                    "bin_idx": tb.get("bin_idx", ""),
                    "bin_start_sec": tb.get("bin_start_sec", ""),
                    "bin_end_sec": tb.get("bin_end_sec", ""),
                    "gaze_quality": tb.get("gaze_quality", ""),
                    "gaze_direction_bucket": gaze_direction,
                    "pose_direction_bucket": pose_direction,
                    "candidate_type": candidate_type,
                    "candidate_id": candidate_id,
                    "candidate_local_track_id": candidate_id if candidate_type == "onscreen_local_track" else "",
                    "candidate_global_person_id": identity.get("global_person_id", ""),
                    "candidate_cast_pid": identity.get("cast_pid", ""),
                    "candidate_identity_confidence": f"{candidate_identity_confidence:.6f}" if identity else "0.000000",
                    "candidate_side": side,
                    "direction_score": f"{direction_score:.6f}",
                    "pose_score": f"{pose_score:.6f}",
                    "identity_score": f"{identity_score:.6f}",
                    "dialogue_score": f"{dialogue_score:.6f}",
                    "quality_score": f"{quality_score:.6f}",
                    "uncertainty_penalty": f"{uncertainty_penalty:.6f}",
                    "total_score": f"{total:.6f}",
                    "score_note": note,
                }
            )
    return candidates


def assignment_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("sequence_id", "")),
        str(row.get("shot_id", "")),
        str(row.get("subject_local_track_id", "")),
        str(row.get("bin_idx", "")),
    )


def candidate_sort_key(row: dict[str, Any]) -> tuple[float, int, str]:
    return (
        safe_float(row.get("total_score")),
        CANDIDATE_TYPE_PRIORITY.get(str(row.get("candidate_type", "")), 0),
        str(row.get("candidate_id", "")),
    )


def status_for_failure(failure_reason: str) -> str:
    if failure_reason == "assigned":
        return "assigned"
    if failure_reason in {"gaze_quality_unknown", "no_candidate"}:
        return "unknown"
    if failure_reason == "low_margin":
        return "ambiguous"
    if failure_reason == "low_score":
        return "rejected"
    return "unknown"


def is_generic_away(candidate: dict[str, Any]) -> bool:
    return str(candidate.get("candidate_type", "")) in {"offscreen_place_or_away", "down_self_or_think", "unknown"}


def active_participant_match(candidate: dict[str, Any], context: dict[str, Any]) -> int:
    candidate_label = normalize_label(candidate.get("candidate_id"))
    active_labels = {normalize_label(item) for item in context.get("active_speakers", [])}
    aligned_labels = {normalize_label(item) for item in context.get("aligned_speakers", [])}
    if not candidate_label:
        return 0
    if candidate_label in active_labels:
        return 2
    if candidate_label in aligned_labels:
        return 1
    return 0


def choose_low_margin_tiebreak(
    tied_candidates: list[dict[str, Any]],
    context: dict[str, Any],
    previous_target_id: str,
) -> tuple[dict[str, Any] | None, str]:
    if len(tied_candidates) < 2:
        return (tied_candidates[0], "single_candidate") if tied_candidates else (None, "")

    def tie_key(candidate: dict[str, Any]) -> tuple[int, float, int, int, int]:
        temporal_match = 1 if previous_target_id and str(candidate.get("candidate_id", "")) == previous_target_id else 0
        non_generic = 0 if is_generic_away(candidate) else 1
        return (
            active_participant_match(candidate, context),
            safe_float(candidate.get("candidate_identity_confidence")),
            temporal_match,
            non_generic,
            CANDIDATE_TYPE_PRIORITY.get(str(candidate.get("candidate_type", "")), 0),
        )

    ranked = sorted(tied_candidates, key=tie_key, reverse=True)
    if tie_key(ranked[0]) <= tie_key(ranked[1]):
        return None, ""
    reasons = ["active_participant", "identity_confidence", "temporal_continuity", "non_generic_away"]
    return ranked[0], "tie_break:" + ",".join(reasons)


def apply_assignment_from_candidate(
    row: dict[str, Any],
    candidate: dict[str, Any],
    confidence: float,
    proxy_source: str,
) -> None:
    row["target_type"] = str(candidate.get("candidate_type", "unknown"))
    row["target_id"] = str(candidate.get("candidate_id", "unknown"))
    row["target_global_person_id"] = str(candidate.get("candidate_global_person_id", ""))
    row["target_cast_pid"] = str(candidate.get("candidate_cast_pid", ""))
    row["target_identity_confidence"] = f"{safe_float(candidate.get('candidate_identity_confidence')):.6f}"
    row["proxy_confidence"] = f"{confidence:.6f}"
    row["proxy_status"] = "assigned"
    row["proxy_source"] = proxy_source
    row["failure_reason"] = "assigned"


def smooth_isolated_unknown_or_ambiguous(rows: list[dict[str, Any]]) -> int:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["sequence_id"], row["shot_id"], row["local_track_id"])].append(row)

    smoothed = 0
    for stream_rows in grouped.values():
        stream_rows.sort(key=lambda row: safe_float(row.get("bin_start_sec")))
        for idx in range(1, len(stream_rows) - 1):
            row = stream_rows[idx]
            if row.get("proxy_status") not in {"unknown", "ambiguous"}:
                continue
            prev_row = stream_rows[idx - 1]
            next_row = stream_rows[idx + 1]
            if prev_row.get("proxy_status") != "assigned" or next_row.get("proxy_status") != "assigned":
                continue
            if prev_row.get("target_id") != next_row.get("target_id") or prev_row.get("target_type") != next_row.get("target_type"):
                continue
            for column in [
                "target_type",
                "target_id",
                "target_global_person_id",
                "target_cast_pid",
                "target_identity_confidence",
            ]:
                row[column] = prev_row.get(column, "")
            prev_conf = safe_float(prev_row.get("proxy_confidence"))
            next_conf = safe_float(next_row.get("proxy_confidence"))
            row["proxy_confidence"] = f"{min(prev_conf, next_conf) * 0.90:.6f}"
            row["proxy_status"] = "assigned"
            row["failure_reason"] = "assigned"
            row["proxy_source"] = "openface_rule+temporal_smoothing"
            row["smoothing_applied"] = "1"
            smoothed += 1
    return smoothed


def build_assignments(
    timebins: list[dict[str, str]],
    candidates: list[dict[str, Any]],
    identity_lookup: dict[tuple[str, str], dict[str, str]],
    contexts: dict[tuple[str, str], dict[str, Any]],
    config: Stage08Config,
) -> list[dict[str, Any]]:
    candidates_by_key: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        candidates_by_key[assignment_key(candidate)].append(candidate)

    rows: list[dict[str, Any]] = []
    previous_target_by_stream: dict[tuple[str, str, str], str] = {}
    sorted_timebins = sorted(
        [tb for tb in timebins if tb.get("movie_id") == config.movie_id],
        key=lambda row: (row.get("sequence_id", ""), row.get("shot_idx", ""), row.get("local_track_id", ""), safe_float(row.get("bin_start_sec"))),
    )
    for tb in sorted_timebins:
        if tb.get("movie_id") != config.movie_id:
            continue
        key = (tb.get("sequence_id", ""), tb.get("shot_id", ""), tb.get("local_track_id", ""), tb.get("bin_idx", ""))
        stream_key = (tb.get("sequence_id", ""), tb.get("shot_id", ""), tb.get("local_track_id", ""))
        group = sorted(
            candidates_by_key.get(key, []),
            key=candidate_sort_key,
            reverse=True,
        )
        non_unknown = [row for row in group if row.get("candidate_type") != "unknown"]
        top = non_unknown[0] if non_unknown else None
        second = non_unknown[1] if len(non_unknown) > 1 else None
        top_score = safe_float(top.get("total_score")) if top else 0.0
        second_score = safe_float(second.get("total_score")) if second else 0.0
        margin = top_score - second_score

        target_type = "unknown"
        target_id = "unknown"
        target_global_person_id = ""
        target_cast_pid = ""
        target_identity_confidence = 0.0
        confidence = 0.0
        proxy_status = "unknown"
        failure_reason = "no_candidate"
        proxy_source = "openface_rule"
        if config.require_gaze_quality and tb.get("gaze_quality") == "unknown":
            failure_reason = "gaze_quality_unknown"
        elif top is None:
            failure_reason = "no_candidate"
        elif top_score < config.min_proxy_score:
            failure_reason = "low_score"
        elif len(non_unknown) > 1 and margin < config.ambiguous_margin:
            context = contexts.get((tb.get("sequence_id", ""), tb.get("shot_id", "")), {})
            tied = [row for row in non_unknown if top_score - safe_float(row.get("total_score")) < config.ambiguous_margin]
            chosen, reason = choose_low_margin_tiebreak(tied, context, previous_target_by_stream.get(stream_key, ""))
            if chosen is not None:
                target_type = str(chosen.get("candidate_type", "unknown"))
                target_id = str(chosen.get("candidate_id", "unknown"))
                target_global_person_id = str(chosen.get("candidate_global_person_id", ""))
                target_cast_pid = str(chosen.get("candidate_cast_pid", ""))
                target_identity_confidence = safe_float(chosen.get("candidate_identity_confidence"))
                confidence = safe_float(chosen.get("total_score"))
                failure_reason = "assigned"
                proxy_status = "assigned"
                proxy_source = "openface_rule+" + reason
            else:
                target_type = "ambiguous"
                target_id = str(top.get("candidate_id", "ambiguous"))
                confidence = top_score
                failure_reason = "low_margin"
        else:
            target_type = str(top.get("candidate_type", "unknown"))
            target_id = str(top.get("candidate_id", "unknown"))
            target_global_person_id = str(top.get("candidate_global_person_id", ""))
            target_cast_pid = str(top.get("candidate_cast_pid", ""))
            target_identity_confidence = safe_float(top.get("candidate_identity_confidence"))
            confidence = top_score
            failure_reason = "assigned"
            proxy_status = "assigned"
        if proxy_status != "assigned":
            proxy_status = status_for_failure(failure_reason)

        context = contexts.get((tb.get("sequence_id", ""), tb.get("shot_id", "")), {})
        subject_identity = identity_lookup.get((tb.get("shot_id", ""), tb.get("local_track_id", "")), {})
        subject_identity_confidence = safe_float(subject_identity.get("identity_confidence"))
        if subject_identity_confidence < 0.60:
            target_global_person_id = ""
            target_cast_pid = ""
            target_identity_confidence = 0.0
        rows.append(
            {
                "movie_id": tb.get("movie_id", ""),
                "sequence_id": tb.get("sequence_id", ""),
                "shot_id": tb.get("shot_id", ""),
                "shot_idx": tb.get("shot_idx", ""),
                "local_track_id": tb.get("local_track_id", ""),
                "bin_idx": tb.get("bin_idx", ""),
                "bin_start_sec": tb.get("bin_start_sec", ""),
                "bin_end_sec": tb.get("bin_end_sec", ""),
                "gaze_quality": tb.get("gaze_quality", ""),
                "gaze_direction_bucket": direction_bucket(tb.get("gaze_angle_x_mean"), config.direction_threshold),
                "pose_direction_bucket": direction_bucket(tb.get("pose_Ry_mean"), config.pose_direction_threshold),
                "subject_global_person_id": subject_identity.get("global_person_id", ""),
                "subject_cast_pid": subject_identity.get("cast_pid", ""),
                "subject_identity_confidence": f"{subject_identity_confidence:.6f}" if subject_identity else "0.000000",
                "target_type": target_type,
                "target_id": target_id,
                "target_global_person_id": target_global_person_id,
                "target_cast_pid": target_cast_pid,
                "target_identity_confidence": f"{target_identity_confidence:.6f}",
                "proxy_confidence": f"{confidence:.6f}",
                "proxy_status": proxy_status,
                "proxy_source": proxy_source,
                "failure_reason": failure_reason,
                "raw_proxy_status": proxy_status,
                "raw_failure_reason": failure_reason,
                "smoothing_applied": "0",
                "top_score": f"{top_score:.6f}",
                "second_score": f"{second_score:.6f}",
                "score_margin": f"{margin:.6f}",
                "candidate_count": len(group),
                "prev_target_id": "",
                "next_target_id": "",
                "subtitle_text": context.get("subtitle_text", ""),
                "aligned_speakers": json.dumps(context.get("aligned_speakers", []), ensure_ascii=False),
                "active_speakers": json.dumps(context.get("active_speakers", []), ensure_ascii=False),
            }
        )
        if proxy_status == "assigned":
            previous_target_by_stream[stream_key] = target_id

    rows.sort(key=lambda row: (row["sequence_id"], row["shot_idx"], row["local_track_id"], safe_float(row["bin_start_sec"])))
    smooth_isolated_unknown_or_ambiguous(rows)
    for i, row in enumerate(rows):
        if i > 0 and rows[i - 1]["sequence_id"] == row["sequence_id"] and rows[i - 1]["local_track_id"] == row["local_track_id"]:
            row["prev_target_id"] = rows[i - 1]["target_id"]
        if i + 1 < len(rows) and rows[i + 1]["sequence_id"] == row["sequence_id"] and rows[i + 1]["local_track_id"] == row["local_track_id"]:
            row["next_target_id"] = rows[i + 1]["target_id"]
    return rows


def build_sequence_packages(assignments: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidate_counts_by_key = Counter(assignment_key(row) for row in candidates)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in assignments:
        grouped[str(row["sequence_id"])].append(row)

    packages: list[dict[str, Any]] = []
    for sequence_id, rows in sorted(grouped.items()):
        packages.append(
            {
                "sequence_id": sequence_id,
                "movie_id": rows[0]["movie_id"] if rows else "",
                "proxy_events": rows,
                "summary": {
                    "assignment_count": len(rows),
                    "candidate_count": sum(
                        candidate_counts_by_key[
                            (row["sequence_id"], row["shot_id"], row["local_track_id"], row["bin_idx"])
                        ]
                        for row in rows
                    ),
                    "target_type_counts": dict(Counter(row["target_type"] for row in rows)),
                    "proxy_status_counts": dict(Counter(row["proxy_status"] for row in rows)),
                    "failure_reason_counts": dict(Counter(row["failure_reason"] for row in rows)),
                },
            }
        )
    return packages


def group_rows(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, ""))].append(row)
    return dict(grouped)


def ensure_inputs(config: Stage08Config) -> None:
    for path in [
        config.timebins_csv,
        config.face_tracks_csv,
        config.shot_manifest_csv,
        config.candidate_sequences_jsonl,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required Stage07 input not found: {path}")


def run(config: Stage08Config) -> None:
    ensure_inputs(config)
    if (
        not config.overwrite
        and config.assignments_csv.exists()
        and config.candidate_targets_csv.exists()
        and config.sequence_packages_jsonl.exists()
    ):
        print(f"[Stage08] outputs already exist; use --overwrite to regenerate: {config.proxy_gaze_dir}")
        return

    timebins = read_csv(config.timebins_csv)
    face_tracks = read_csv(config.face_tracks_csv)
    raw_shot_manifest = [row for row in read_csv(config.shot_manifest_csv) if row.get("movie_id") == config.movie_id]
    shot_manifest, skipped_stage_type_count, stage_counts = filter_manifest_rows_by_stage_type(
        raw_shot_manifest, config.stage_type_include
    )
    allowed_shots = {row.get("shot_id", "") for row in shot_manifest}
    timebins = [row for row in timebins if row.get("movie_id") == config.movie_id and row.get("shot_id", "") in allowed_shots]
    face_tracks = [row for row in face_tracks if row.get("movie_id") == config.movie_id and row.get("shot_id", "") in allowed_shots]
    track_identity_rows = read_csv(config.track_identity_csv) if config.track_identity_csv.exists() else []
    track_identity_rows = [
        row for row in track_identity_rows if row.get("movie_id") == config.movie_id and row.get("shot_id", "") in allowed_shots
    ]
    sequences = load_sequences(config.candidate_sequences_jsonl)
    contexts = shot_contexts(shot_manifest, sequences)
    track_index = build_track_index(face_tracks)
    identity_lookup = build_identity_lookup(track_identity_rows)

    candidates = build_candidates(timebins, track_index, identity_lookup, contexts, config)
    assignments = build_assignments(timebins, candidates, identity_lookup, contexts, config)
    packages = build_sequence_packages(assignments, candidates)

    write_csv(config.candidate_targets_csv, candidates, CANDIDATE_COLUMNS)
    write_csv(config.assignments_csv, assignments, ASSIGNMENT_COLUMNS)
    write_jsonl(config.assignments_jsonl, assignments)
    write_jsonl(config.sequence_packages_jsonl, packages)

    target_type_counts = dict(Counter(row["target_type"] for row in assignments))
    proxy_status_counts = dict(Counter(row["proxy_status"] for row in assignments))
    failure_reason_counts = dict(Counter(row["failure_reason"] for row in assignments))
    raw_proxy_status_counts = dict(Counter(row["raw_proxy_status"] for row in assignments))
    raw_failure_reason_counts = dict(Counter(row["raw_failure_reason"] for row in assignments))
    target_type_by_gaze_quality = {
        quality: dict(Counter(row["target_type"] for row in rows))
        for quality, rows in group_rows(assignments, "gaze_quality").items()
    }
    proxy_status_by_gaze_quality = {
        quality: dict(Counter(row["proxy_status"] for row in rows))
        for quality, rows in group_rows(assignments, "gaze_quality").items()
    }
    failure_reason_by_gaze_quality = {
        quality: dict(Counter(row["failure_reason"] for row in rows))
        for quality, rows in group_rows(assignments, "gaze_quality").items()
    }
    assigned_count = sum(1 for row in assignments if row["proxy_status"] == "assigned")
    proxy_count = len(assignments)
    payload = {
        "movie_id": config.movie_id,
        "timebin_count": len([row for row in timebins if row.get("movie_id") == config.movie_id]),
        "candidate_count": len(candidates),
        "proxy_count": proxy_count,
        "sequence_package_count": len(packages),
        "assigned_count": assigned_count,
        "assigned_ratio": assigned_count / proxy_count if proxy_count else 0.0,
        "stage_type_include": stage_type_include_label(config.stage_type_include),
        "stage_type_counts": stage_counts,
        "skipped_stage_type_count": skipped_stage_type_count,
        "track_identity_csv": str(config.track_identity_csv),
        "target_type_counts": target_type_counts,
        "proxy_status_counts": proxy_status_counts,
        "failure_reason_counts": failure_reason_counts,
        "raw_proxy_status_counts": raw_proxy_status_counts,
        "raw_failure_reason_counts": raw_failure_reason_counts,
        "smoothing_applied_count": sum(1 for row in assignments if row.get("smoothing_applied") == "1"),
        "target_type_by_gaze_quality": target_type_by_gaze_quality,
        "proxy_status_by_gaze_quality": proxy_status_by_gaze_quality,
        "failure_reason_by_gaze_quality": failure_reason_by_gaze_quality,
        "config": {
            "direction_threshold": config.direction_threshold,
            "pose_direction_threshold": config.pose_direction_threshold,
            "min_proxy_score": config.min_proxy_score,
            "ambiguous_margin": config.ambiguous_margin,
            "require_gaze_quality": config.require_gaze_quality,
            "high_precision": config.high_precision,
        },
        "outputs": {
            "candidate_targets_csv": str(config.candidate_targets_csv),
            "assignments_csv": str(config.assignments_csv),
            "assignments_jsonl": str(config.assignments_jsonl),
            "sequence_packages_jsonl": str(config.sequence_packages_jsonl),
            "summary_json": str(config.summary_json),
        },
    }
    write_json(config.summary_json, payload)

    print(f"[Stage08] movie_id={config.movie_id}")
    print(
        f"[Stage08] stage_type_include={stage_type_include_label(config.stage_type_include)} "
        f"skipped_stage_type={skipped_stage_type_count} stage_type_counts={json.dumps(stage_counts, sort_keys=True)}"
    )
    print(f"[Stage08] timebins={payload['timebin_count']} candidates={len(candidates)} proxies={len(assignments)}")
    print(f"[Stage08] target_type_counts={json.dumps(target_type_counts, sort_keys=True)}")
    print(f"[Stage08] proxy_status_counts={json.dumps(proxy_status_counts, sort_keys=True)}")
    print(f"[Stage08] failure_reason_counts={json.dumps(failure_reason_counts, sort_keys=True)}")
    print(f"[Stage08] candidate_targets_csv={config.candidate_targets_csv}")
    print(f"[Stage08] assignments_csv={config.assignments_csv}")
    print(f"[Stage08] summary_json={config.summary_json}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config")
    parser.add_argument("--movie-id")
    parser.add_argument("--timebins-csv")
    parser.add_argument("--face-tracks-csv")
    parser.add_argument("--shot-manifest-csv")
    parser.add_argument("--candidate-sequences-jsonl")
    parser.add_argument("--track-identity-csv")
    parser.add_argument("--stage-type-include", help="Comma list of stage_type values to process, or 'all'.")
    parser.add_argument("--no-sface-gallery", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--direction-threshold", type=float)
    parser.add_argument("--pose-direction-threshold", type=float)
    parser.add_argument("--min-proxy-score", type=float)
    parser.add_argument("--ambiguous-margin", type=float)
    parser.add_argument("--no-require-gaze-quality", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        run(make_config(args))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[Stage08] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
