"""Build event-level gaze proxy assignments from bin-level evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
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


SHOT_CONTEXT_COLUMNS = [
    "movie_id",
    "sequence_id",
    "shot_id",
    "shot_idx",
    "prev_shot_id",
    "next_shot_id",
    "visible_global_persons",
    "current_speaker",
    "prev_speaker",
    "next_speaker",
    "active_participants",
    "is_single_closeup",
    "is_two_person_shot",
    "is_reaction_shot",
    "prev_visible_focus",
    "next_visible_focus",
    "likely_interlocutor",
]

GAZE_EVENT_COLUMNS = [
    "movie_id",
    "sequence_id",
    "shot_id",
    "subject_global_person_id",
    "subject_local_track_id",
    "event_id",
    "event_start",
    "event_end",
    "start_bin",
    "end_bin",
    "dominant_direction",
    "dominant_pose_direction",
    "gaze_quality_summary",
    "candidate_target_type",
    "candidate_target_id",
    "event_confidence",
    "event_status",
    "evidence_notes",
    "subject_cast_pid",
    "target_global_person_id",
    "target_cast_pid",
    "proxy_status",
    "failure_reason",
    "top_score",
    "second_score",
    "score_margin",
    "bin_count",
]

GAZE_EVENT_BIN_COLUMNS = [
    "movie_id",
    "sequence_id",
    "shot_id",
    "subject_local_track_id",
    "bin_idx",
    "event_id",
    "event_status",
    "event_target_type",
    "event_target_id",
    "event_target_global_person_id",
    "event_target_cast_pid",
    "event_confidence",
    "event_failure_reason",
]

CANDIDATE_TYPE_PRIORITY = {
    "onscreen_local_track": 5,
    "current_speaker": 4,
    "offscreen_participant": 3,
    "offscreen_place_or_away": 2,
    "down_self_or_think": 1,
    "unknown": 0,
}


@dataclass(frozen=True)
class Stage08bConfig:
    movie_id: str
    timebins_csv: Path
    shot_manifest_csv: Path
    candidate_sequences_jsonl: Path
    track_identity_csv: Path
    assignments_csv: Path
    candidate_targets_csv: Path
    gaze_event_dir: Path
    logs_dir: Path
    min_event_duration_sec: float
    min_event_score: float
    ambiguous_margin: float
    identity_confidence_threshold: float
    stage_type_include: set[str] | None
    overwrite: bool

    @property
    def shot_context_csv(self) -> Path:
        return self.gaze_event_dir / "08b_shot_context.csv"

    @property
    def gaze_events_csv(self) -> Path:
        return self.gaze_event_dir / "08b_gaze_events.csv"

    @property
    def gaze_event_bins_csv(self) -> Path:
        return self.gaze_event_dir / "08b_gaze_event_bins.csv"

    @property
    def summary_json(self) -> Path:
        return self.logs_dir / "08b_build_gaze_events_summary.json"


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


def make_config(args: argparse.Namespace) -> Stage08bConfig:
    run_config: dict[str, Any] = {}
    project_root = Path.cwd()
    if args.run_config:
        run_config, project_root = load_run_config(Path(args.run_config).resolve())

    movie_id = args.movie_id or run_config.get("data", {}).get("movie_id")
    if not movie_id:
        raise ValueError("movie_id is required via --movie-id or run config data.movie_id")
    movie_id = str(movie_id)

    outputs = run_config.get("outputs", {})
    text_inputs = run_config.get("inputs_from_text_pipeline", {})
    selection = run_config.get("selection", {})
    stage = run_config.get("stages", {}).get("build_gaze_events", {})
    logs_dir = resolve_path(outputs.get("logs_dir") or f"outputs/video_proxy/{movie_id}/logs", project_root)
    openface_dir = resolve_path(outputs.get("openface_dir") or f"outputs/video_proxy/{movie_id}/openface", project_root)
    identity_dir = resolve_path(
        outputs.get("track_identity_dir") or f"outputs/video_proxy/{movie_id}/track_identities", project_root
    )
    proxy_gaze_dir = resolve_path(outputs.get("proxy_gaze_dir") or f"outputs/video_proxy/{movie_id}/proxy_gaze_scripts", project_root)
    gaze_event_dir = resolve_path(
        args.gaze_event_dir or outputs.get("gaze_event_dir") or f"outputs/video_proxy/{movie_id}/gaze_events",
        project_root,
    )
    assert logs_dir is not None and openface_dir is not None and identity_dir is not None
    assert proxy_gaze_dir is not None and gaze_event_dir is not None

    return Stage08bConfig(
        movie_id=movie_id,
        timebins_csv=resolve_path(args.timebins_csv, project_root) or openface_dir / "06_gaze_timebins.csv",
        shot_manifest_csv=resolve_path(args.shot_manifest_csv, project_root) or logs_dir / "04_shot_manifest.csv",
        candidate_sequences_jsonl=resolve_path(
            args.candidate_sequences_jsonl
            or text_inputs.get("candidate_sequences_jsonl")
            or f"data/processed/candidate_sequences/{movie_id}__candidate_sequences.jsonl",
            project_root,
        )
        or Path(),
        track_identity_csv=resolve_path(args.track_identity_csv, project_root) or identity_dir / "07_track_identity.csv",
        assignments_csv=resolve_path(args.assignments_csv, project_root) or proxy_gaze_dir / "08_proxy_assignments.csv",
        candidate_targets_csv=resolve_path(args.candidate_targets_csv, project_root)
        or proxy_gaze_dir / "08_candidate_targets.csv",
        gaze_event_dir=gaze_event_dir,
        logs_dir=logs_dir,
        min_event_duration_sec=float(
            args.min_event_duration_sec
            if args.min_event_duration_sec is not None
            else stage.get("min_event_duration_sec", 1.0)
        ),
        min_event_score=float(args.min_event_score if args.min_event_score is not None else stage.get("min_event_score", 0.55)),
        ambiguous_margin=float(
            args.ambiguous_margin if args.ambiguous_margin is not None else stage.get("ambiguous_margin", 0.10)
        ),
        identity_confidence_threshold=float(stage.get("identity_confidence_threshold", 0.60)),
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


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed) or math.isinf(parsed):
        return default
    return parsed


def parse_json_cell(value: Any) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        parsed = value
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item)]
    if isinstance(parsed, str):
        return [item.strip() for item in parsed.split("|") if item.strip()]
    return []


def load_sequences(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    sequences: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seq = json.loads(line)
            sequences[str(seq.get("sequence_id"))] = seq
    return sequences


def normalize_label(value: Any) -> str:
    return "".join(ch for ch in str(value or "").upper() if ch.isalnum())


def direction_side(value: Any) -> str:
    text = str(value or "")
    if "left" in text:
        return "left"
    if "right" in text:
        return "right"
    if "center" in text:
        return "center"
    return "unknown"


def directions_similar(a: Any, b: Any) -> bool:
    side_a = direction_side(a)
    side_b = direction_side(b)
    if "unknown" in {side_a, side_b}:
        return False
    if side_a == side_b:
        return True
    return "center" in {side_a, side_b}


def dominant(values: list[str]) -> str:
    clean = [value for value in values if value]
    if not clean:
        return ""
    return Counter(clean).most_common(1)[0][0]


def median_float(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def group_by(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(column, "")) for column in columns)].append(row)
    return grouped


def identity_label(identity: dict[str, str]) -> str:
    for key in ["cast_pid", "character_name", "cast_name", "global_person_id"]:
        value = str(identity.get(key, ""))
        if value:
            return value
    return ""


def visible_identities_by_shot(
    identities: list[dict[str, str]],
    threshold: float,
) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for identity in identities:
        if safe_float(identity.get("identity_confidence")) >= threshold:
            grouped[identity.get("shot_id", "")].append(identity)
    return grouped


def visible_focus(visible: list[dict[str, str]]) -> str:
    labels = [identity_label(row) for row in visible if identity_label(row)]
    if not labels:
        return ""
    return Counter(labels).most_common(1)[0][0]


def build_shot_contexts(
    shot_manifest: list[dict[str, str]],
    identities: list[dict[str, str]],
    sequences: dict[str, dict[str, Any]],
    identity_confidence_threshold: float = 0.60,
) -> list[dict[str, Any]]:
    visible_by_shot = visible_identities_by_shot(identities, identity_confidence_threshold)
    shots_by_sequence: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in shot_manifest:
        shots_by_sequence[row.get("sequence_id", "")].append(row)
    for sequence_rows in shots_by_sequence.values():
        sequence_rows.sort(key=lambda row: safe_float(row.get("shot_idx")))

    contexts: list[dict[str, Any]] = []
    for sequence_id, sequence_rows in shots_by_sequence.items():
        seq = sequences.get(sequence_id, {})
        active_participants = [str(item) for item in seq.get("active_speakers", []) if str(item)]
        for idx, row in enumerate(sequence_rows):
            shot_id = row.get("shot_id", "")
            prev_row = sequence_rows[idx - 1] if idx > 0 else {}
            next_row = sequence_rows[idx + 1] if idx + 1 < len(sequence_rows) else {}
            visible = visible_by_shot.get(shot_id, [])
            visible_labels = [identity_label(item) for item in visible if identity_label(item)]
            current_speakers = parse_json_cell(row.get("aligned_speakers"))
            prev_speakers = parse_json_cell(prev_row.get("aligned_speakers")) if prev_row else []
            next_speakers = parse_json_cell(next_row.get("aligned_speakers")) if next_row else []
            current_speaker = current_speakers[0] if current_speakers else ""
            prev_visible = visible_focus(visible_by_shot.get(prev_row.get("shot_id", ""), [])) if prev_row else ""
            next_visible = visible_focus(visible_by_shot.get(next_row.get("shot_id", ""), [])) if next_row else ""
            visible_norm = {normalize_label(item) for item in visible_labels}
            current_norm = normalize_label(current_speaker)
            is_single = len(set(visible_labels)) == 1
            is_two = len(set(visible_labels)) == 2
            is_reaction = is_single and bool(current_speaker) and current_norm not in visible_norm
            likely = ""
            for candidate in [prev_visible, next_visible, *active_participants]:
                if candidate and normalize_label(candidate) not in visible_norm:
                    likely = candidate
                    break
            contexts.append(
                {
                    "movie_id": row.get("movie_id", ""),
                    "sequence_id": sequence_id,
                    "shot_id": shot_id,
                    "shot_idx": row.get("shot_idx", ""),
                    "prev_shot_id": prev_row.get("shot_id", ""),
                    "next_shot_id": next_row.get("shot_id", ""),
                    "visible_global_persons": json.dumps(sorted(set(visible_labels)), ensure_ascii=False),
                    "current_speaker": current_speaker,
                    "prev_speaker": prev_speakers[0] if prev_speakers else "",
                    "next_speaker": next_speakers[0] if next_speakers else "",
                    "active_participants": json.dumps(active_participants, ensure_ascii=False),
                    "is_single_closeup": "1" if is_single else "0",
                    "is_two_person_shot": "1" if is_two else "0",
                    "is_reaction_shot": "1" if is_reaction else "0",
                    "prev_visible_focus": prev_visible,
                    "next_visible_focus": next_visible,
                    "likely_interlocutor": likely,
                }
            )
    return contexts


def bin_key(row: dict[str, Any], track_key: str = "local_track_id") -> tuple[str, str, str, str]:
    return (
        str(row.get("sequence_id", "")),
        str(row.get("shot_id", "")),
        str(row.get(track_key, "")),
        str(row.get("bin_idx", "")),
    )


def stream_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("sequence_id", "")), str(row.get("shot_id", "")), str(row.get("local_track_id", "")))


def should_merge_bins(prev_bin: dict[str, Any], next_bin: dict[str, Any]) -> bool:
    if directions_similar(prev_bin.get("gaze_direction_bucket"), next_bin.get("gaze_direction_bucket")):
        return True
    if directions_similar(prev_bin.get("pose_direction_bucket"), next_bin.get("pose_direction_bucket")):
        return True
    return False


def initial_event_groups(stream_rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for row in sorted(stream_rows, key=lambda item: safe_float(item.get("bin_start_sec"))):
        if not groups or not should_merge_bins(groups[-1][-1], row):
            groups.append([row])
        else:
            groups[-1].append(row)
    return groups


def bridge_single_unknown_groups(groups: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    if len(groups) < 3:
        return groups
    output: list[list[dict[str, Any]]] = []
    idx = 0
    while idx < len(groups):
        if 0 < idx < len(groups) - 1 and len(groups[idx]) == 1:
            row = groups[idx][0]
            prev = groups[idx - 1][-1]
            nxt = groups[idx + 1][0]
            unknownish = row.get("proxy_status") in {"unknown", "ambiguous"} or row.get("gaze_quality") == "unknown"
            same_target = (
                prev.get("target_type") == nxt.get("target_type")
                and prev.get("target_id") == nxt.get("target_id")
                and prev.get("proxy_status") == "assigned"
                and nxt.get("proxy_status") == "assigned"
            )
            if unknownish and same_target:
                if output and output[-1] is groups[idx - 1]:
                    output[-1].extend(groups[idx])
                    output[-1].extend(groups[idx + 1])
                else:
                    merged = [*groups[idx - 1], *groups[idx], *groups[idx + 1]]
                    if output and output[-1] == groups[idx - 1]:
                        output[-1] = merged
                    else:
                        output.append(merged)
                idx += 2
                continue
        if not output or output[-1] is not groups[idx]:
            output.append(groups[idx])
        idx += 1
    compact: list[list[dict[str, Any]]] = []
    for group in output:
        if compact and group and compact[-1] and group[0] is compact[-1][-1]:
            compact[-1].extend(group[1:])
        else:
            compact.append(group)
    return compact


def event_duration(group: list[dict[str, Any]]) -> float:
    if not group:
        return 0.0
    return max(0.0, safe_float(group[-1].get("bin_end_sec")) - safe_float(group[0].get("bin_start_sec")))


def remove_flicker_groups(groups: list[list[dict[str, Any]]], min_duration_sec: float) -> list[list[dict[str, Any]]]:
    if len(groups) < 3:
        return groups
    output: list[list[dict[str, Any]]] = []
    idx = 0
    while idx < len(groups):
        if 0 < idx < len(groups) - 1 and event_duration(groups[idx]) < min_duration_sec:
            prev = groups[idx - 1][-1]
            nxt = groups[idx + 1][0]
            if prev.get("target_type") == nxt.get("target_type") and prev.get("target_id") == nxt.get("target_id"):
                if output and output[-1] == groups[idx - 1]:
                    output[-1].extend(groups[idx])
                    output[-1].extend(groups[idx + 1])
                else:
                    output.append([*groups[idx - 1], *groups[idx], *groups[idx + 1]])
                idx += 2
                continue
        if not output or output[-1] != groups[idx]:
            output.append(groups[idx])
        idx += 1
    return output


def segment_events(assignments: list[dict[str, Any]], min_duration_sec: float) -> dict[tuple[str, str, str], list[list[dict[str, Any]]]]:
    segmented: dict[tuple[str, str, str], list[list[dict[str, Any]]]] = {}
    for key, rows in group_by(assignments, ("sequence_id", "shot_id", "local_track_id")).items():
        groups = initial_event_groups(rows)
        groups = bridge_single_unknown_groups(groups)
        groups = remove_flicker_groups(groups, min_duration_sec)
        segmented[key] = groups
    return segmented


def context_lookup(contexts: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["sequence_id"], row["shot_id"]): row for row in contexts}


def score_candidate_for_event(
    event_bins: list[dict[str, Any]],
    candidate_rows: list[dict[str, str]],
    context: dict[str, Any],
    previous_target_id: str,
) -> tuple[dict[str, str], float, str]:
    scores = [safe_float(row.get("total_score")) for row in candidate_rows]
    score = (sum(scores) / len(scores)) if scores else 0.0
    candidate = candidate_rows[0]
    candidate_label = normalize_label(candidate.get("candidate_id"))
    likely = normalize_label(context.get("likely_interlocutor"))
    active = {normalize_label(item) for item in parse_json_cell(context.get("active_participants"))}
    notes: list[str] = []
    if likely and candidate_label == likely:
        score += 0.16
        notes.append("likely_interlocutor_prior")
    if candidate_label in active:
        score += 0.06
        notes.append("active_participant_prior")
    if context.get("is_reaction_shot") == "1" and candidate_label == likely:
        score += 0.12
        notes.append("reaction_shot_prior")
    if context.get("is_two_person_shot") == "1" and candidate.get("candidate_type") == "onscreen_local_track":
        score += 0.08
        notes.append("two_person_visible_prior")
    if previous_target_id and candidate.get("candidate_id") == previous_target_id:
        score += 0.08
        notes.append("event_temporal_continuity")
    assigned_bins = sum(1 for row in event_bins if row.get("proxy_status") == "assigned")
    score += min(0.10, 0.02 * assigned_bins)
    return candidate, max(0.0, min(1.0, score)), ";".join(notes)


def choose_event_target(
    event_bins: list[dict[str, Any]],
    candidates_by_bin: dict[tuple[str, str, str, str], list[dict[str, str]]],
    context: dict[str, Any],
    config: Stage08bConfig,
    previous_target_id: str,
) -> tuple[dict[str, str] | None, float, float, float, str, str, str]:
    by_candidate: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for event_bin in event_bins:
        key = bin_key(event_bin)
        for candidate in candidates_by_bin.get(key, []):
            if candidate.get("candidate_type") == "unknown":
                continue
            by_candidate[(candidate.get("candidate_type", ""), candidate.get("candidate_id", ""))].append(candidate)
    if not by_candidate:
        return None, 0.0, 0.0, 0.0, "unknown", "no_candidate", ""

    scored: list[tuple[dict[str, str], float, str]] = []
    for rows in by_candidate.values():
        scored.append(score_candidate_for_event(event_bins, rows, context, previous_target_id))
    scored.sort(
        key=lambda item: (
            item[1],
            CANDIDATE_TYPE_PRIORITY.get(item[0].get("candidate_type", ""), 0),
            item[0].get("candidate_id", ""),
        ),
        reverse=True,
    )
    top_candidate, top_score, top_note = scored[0]
    second_score = scored[1][1] if len(scored) > 1 else 0.0
    margin = top_score - second_score
    if top_score < config.min_event_score:
        return top_candidate, top_score, second_score, margin, "rejected", "low_score", top_note
    if len(scored) > 1 and margin < config.ambiguous_margin:
        if previous_target_id and top_candidate.get("candidate_id") == previous_target_id:
            return top_candidate, top_score, second_score, margin, "assigned", "assigned", top_note + ";event_tie_break_temporal"
        return top_candidate, top_score, second_score, margin, "ambiguous", "low_margin", top_note
    return top_candidate, top_score, second_score, margin, "assigned", "assigned", top_note


def summarize_quality(event_bins: list[dict[str, Any]]) -> str:
    counts = Counter(str(row.get("gaze_quality", "")) for row in event_bins)
    return json.dumps(dict(counts), ensure_ascii=False, sort_keys=True)


def build_candidate_lookup(candidates: list[dict[str, str]]) -> dict[tuple[str, str, str, str], list[dict[str, str]]]:
    lookup: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        lookup[
            (
                row.get("sequence_id", ""),
                row.get("shot_id", ""),
                row.get("subject_local_track_id", ""),
                row.get("bin_idx", ""),
            )
        ].append(row)
    return lookup


def build_identity_lookup(identities: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row.get("shot_id", ""), row.get("local_track_id", "")): row for row in identities}


def build_events(
    assignments: list[dict[str, Any]],
    candidates: list[dict[str, str]],
    identities: list[dict[str, str]],
    contexts: list[dict[str, Any]],
    config: Stage08bConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_lookup = build_candidate_lookup(candidates)
    identity_lookup = build_identity_lookup(identities)
    contexts_by_shot = context_lookup(contexts)
    segmented = segment_events(assignments, config.min_event_duration_sec)
    events: list[dict[str, Any]] = []
    event_bins: list[dict[str, Any]] = []
    previous_target_by_stream: dict[tuple[str, str, str], str] = {}

    for stream, groups in sorted(segmented.items()):
        sequence_id, shot_id, local_track_id = stream
        identity = identity_lookup.get((shot_id, local_track_id), {})
        context = contexts_by_shot.get((sequence_id, shot_id), {})
        for event_index, group in enumerate(groups):
            event_id = f"{shot_id}__{local_track_id}__ev_{event_index:03d}"
            previous_target = previous_target_by_stream.get(stream, "")
            target, top_score, second_score, margin, event_status, failure_reason, score_note = choose_event_target(
                group, candidate_lookup, context, config, previous_target
            )
            target_type = target.get("candidate_type", "unknown") if target else "unknown"
            target_id = target.get("candidate_id", "unknown") if target else "unknown"
            target_global_person_id = target.get("candidate_global_person_id", "") if target else ""
            target_cast_pid = target.get("candidate_cast_pid", "") if target else ""
            if event_status == "assigned":
                previous_target_by_stream[stream] = target_id
            dominant_direction = dominant([str(row.get("gaze_direction_bucket", "")) for row in group])
            dominant_pose_direction = dominant([str(row.get("pose_direction_bucket", "")) for row in group])
            event_start = safe_float(group[0].get("bin_start_sec"))
            event_end = safe_float(group[-1].get("bin_end_sec"))
            notes = [
                score_note,
                f"duration={event_end - event_start:.3f}",
                f"bin_count={len(group)}",
                f"likely_interlocutor={context.get('likely_interlocutor', '')}",
            ]
            event_row = {
                "movie_id": group[0].get("movie_id", config.movie_id),
                "sequence_id": sequence_id,
                "shot_id": shot_id,
                "subject_global_person_id": identity.get("global_person_id", ""),
                "subject_local_track_id": local_track_id,
                "event_id": event_id,
                "event_start": f"{event_start:.3f}",
                "event_end": f"{event_end:.3f}",
                "start_bin": group[0].get("bin_idx", ""),
                "end_bin": group[-1].get("bin_idx", ""),
                "dominant_direction": dominant_direction,
                "dominant_pose_direction": dominant_pose_direction,
                "gaze_quality_summary": summarize_quality(group),
                "candidate_target_type": target_type if event_status == "assigned" else event_status,
                "candidate_target_id": target_id if event_status == "assigned" else target_id,
                "event_confidence": f"{top_score:.6f}",
                "event_status": event_status,
                "evidence_notes": ";".join(note for note in notes if note),
                "subject_cast_pid": identity.get("cast_pid", ""),
                "target_global_person_id": target_global_person_id if event_status == "assigned" else "",
                "target_cast_pid": target_cast_pid if event_status == "assigned" else "",
                "proxy_status": event_status,
                "failure_reason": failure_reason,
                "top_score": f"{top_score:.6f}",
                "second_score": f"{second_score:.6f}",
                "score_margin": f"{margin:.6f}",
                "bin_count": len(group),
            }
            events.append(event_row)
            for row in group:
                event_bins.append(
                    {
                        "movie_id": row.get("movie_id", config.movie_id),
                        "sequence_id": sequence_id,
                        "shot_id": shot_id,
                        "subject_local_track_id": local_track_id,
                        "bin_idx": row.get("bin_idx", ""),
                        "event_id": event_id,
                        "event_status": event_status,
                        "event_target_type": event_row["candidate_target_type"],
                        "event_target_id": event_row["candidate_target_id"],
                        "event_target_global_person_id": event_row["target_global_person_id"],
                        "event_target_cast_pid": event_row["target_cast_pid"],
                        "event_confidence": event_row["event_confidence"],
                        "event_failure_reason": failure_reason,
                    }
                )
    return events, event_bins


def ensure_inputs(config: Stage08bConfig) -> None:
    for path in [
        config.timebins_csv,
        config.shot_manifest_csv,
        config.track_identity_csv,
        config.assignments_csv,
        config.candidate_targets_csv,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required Stage08b input not found: {path}")


def enrich_assignments_with_timebins(assignments: list[dict[str, str]], timebins: list[dict[str, str]]) -> list[dict[str, Any]]:
    timebin_lookup = {bin_key(row): row for row in timebins}
    enriched: list[dict[str, Any]] = []
    for assignment in assignments:
        key = bin_key(assignment)
        tb = timebin_lookup.get(key, {})
        row = {**tb, **assignment}
        row.setdefault("gaze_direction_bucket", assignment.get("gaze_direction_bucket", "unknown"))
        row.setdefault("pose_direction_bucket", assignment.get("pose_direction_bucket", "unknown"))
        enriched.append(row)
    return enriched


def run(config: Stage08bConfig) -> None:
    ensure_inputs(config)
    if (
        not config.overwrite
        and config.shot_context_csv.exists()
        and config.gaze_events_csv.exists()
        and config.gaze_event_bins_csv.exists()
    ):
        print(f"[Stage08b] outputs already exist; use --overwrite to regenerate: {config.gaze_event_dir}")
        return

    raw_manifest = [row for row in read_csv(config.shot_manifest_csv) if row.get("movie_id") == config.movie_id]
    manifest, skipped_stage_type_count, stage_counts = filter_manifest_rows_by_stage_type(raw_manifest, config.stage_type_include)
    allowed_shots = {row.get("shot_id", "") for row in manifest}
    timebins = [row for row in read_csv(config.timebins_csv) if row.get("movie_id") == config.movie_id and row.get("shot_id") in allowed_shots]
    identities = [
        row for row in read_csv(config.track_identity_csv) if row.get("movie_id") == config.movie_id and row.get("shot_id") in allowed_shots
    ]
    assignments = [
        row for row in read_csv(config.assignments_csv) if row.get("movie_id") == config.movie_id and row.get("shot_id") in allowed_shots
    ]
    candidates = [
        row for row in read_csv(config.candidate_targets_csv) if row.get("movie_id") == config.movie_id and row.get("shot_id") in allowed_shots
    ]
    sequences = load_sequences(config.candidate_sequences_jsonl)
    shot_context_rows = build_shot_contexts(manifest, identities, sequences, config.identity_confidence_threshold)
    enriched_assignments = enrich_assignments_with_timebins(assignments, timebins)
    events, event_bins = build_events(enriched_assignments, candidates, identities, shot_context_rows, config)

    write_csv(config.shot_context_csv, shot_context_rows, SHOT_CONTEXT_COLUMNS)
    write_csv(config.gaze_events_csv, events, GAZE_EVENT_COLUMNS)
    write_csv(config.gaze_event_bins_csv, event_bins, GAZE_EVENT_BIN_COLUMNS)

    payload = {
        "movie_id": config.movie_id,
        "timebin_count": len(timebins),
        "assignment_count": len(assignments),
        "shot_context_count": len(shot_context_rows),
        "event_count": len(events),
        "event_bin_count": len(event_bins),
        "stage_type_include": stage_type_include_label(config.stage_type_include),
        "stage_type_counts": stage_counts,
        "skipped_stage_type_count": skipped_stage_type_count,
        "event_status_counts": dict(Counter(row["event_status"] for row in events)),
        "target_type_counts": dict(Counter(row["candidate_target_type"] for row in events)),
        "config": {
            "min_event_duration_sec": config.min_event_duration_sec,
            "min_event_score": config.min_event_score,
            "ambiguous_margin": config.ambiguous_margin,
            "identity_confidence_threshold": config.identity_confidence_threshold,
        },
        "outputs": {
            "shot_context_csv": str(config.shot_context_csv),
            "gaze_events_csv": str(config.gaze_events_csv),
            "gaze_event_bins_csv": str(config.gaze_event_bins_csv),
            "summary_json": str(config.summary_json),
        },
    }
    write_json(config.summary_json, payload)

    print(f"[Stage08b] movie_id={config.movie_id}")
    print(
        f"[Stage08b] stage_type_include={stage_type_include_label(config.stage_type_include)} "
        f"skipped_stage_type={skipped_stage_type_count} stage_type_counts={json.dumps(stage_counts, sort_keys=True)}"
    )
    print(f"[Stage08b] events={len(events)} event_bins={len(event_bins)} status_counts={json.dumps(payload['event_status_counts'], sort_keys=True)}")
    print(f"[Stage08b] shot_context_csv={config.shot_context_csv}")
    print(f"[Stage08b] gaze_events_csv={config.gaze_events_csv}")
    print(f"[Stage08b] gaze_event_bins_csv={config.gaze_event_bins_csv}")
    print(f"[Stage08b] summary_json={config.summary_json}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config")
    parser.add_argument("--movie-id")
    parser.add_argument("--timebins-csv")
    parser.add_argument("--shot-manifest-csv")
    parser.add_argument("--candidate-sequences-jsonl")
    parser.add_argument("--track-identity-csv")
    parser.add_argument("--assignments-csv")
    parser.add_argument("--candidate-targets-csv")
    parser.add_argument("--gaze-event-dir")
    parser.add_argument("--min-event-duration-sec", type=float)
    parser.add_argument("--min-event-score", type=float)
    parser.add_argument("--ambiguous-margin", type=float)
    parser.add_argument("--stage-type-include", help="Comma list of stage_type values to process, or 'all'.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        run(make_config(args))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[Stage08b] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
