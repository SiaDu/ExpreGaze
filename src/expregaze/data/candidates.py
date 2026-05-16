#!/usr/bin/env python3
"""
Build candidate shot sequences from an existing full_context table.

Stage02 only reads the full_context CSV produced by Stage01. It does not read
raw subtitle, annotation, meta, or screenplay files. Screenplay visual/action
cues are extracted from the context text already present in full_context.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


SCENE_HEADING_RE = re.compile(
    r"^\s*(?:INT\.|EXT\.|INT/EXT\.|EXT/INT\.|I/E\.|FADE IN|FADE OUT|CUT TO|DISSOLVE TO)\b.*$",
    re.IGNORECASE | re.MULTILINE,
)
CAMERA_CUE_RE = re.compile(
    r"\b(?:CAMERA|ANGLE|POV|PAN|PANS|PANNING|TRACK|TRACKS|TRUCK|TRUCKS|DOLLY|"
    r"CLOSE[- ]?UP|CLOSE SHOT|WIDE SHOT|LONG SHOT|MEDIUM SHOT|MS|MCS|MCU|CU|ECU|LS)\b",
    re.IGNORECASE,
)
GAZE_VERB_RE = re.compile(
    r"\b(?:LOOKS?|GLANCES?|STARES?|WATCHES?|SEES?|NOTICES?|EYES?|PEERS?|GAZES?|"
    r"TURNS? TO|LOOKS? AWAY|LOOKS? DOWN|LOOKS? UP)\b",
    re.IGNORECASE,
)
MOVEMENT_CUE_RE = re.compile(
    r"\b(?:ENTERS?|EXITS?|CROSSES?|MOVES?|WALKS?|RUNS?|STEPS?|TURNS?|FOLLOWS?|"
    r"APPROACHES?|BACKS? AWAY|LEANS?|SITS?|STANDS?)\b",
    re.IGNORECASE,
)
NARRATION_SPEAKER_RE = re.compile(r"\b(?:NARRATOR|VOICE|VOICE OVER|VO|ANNOUNCER)\b", re.IGNORECASE)
SINGING_TEXT_RE = re.compile(r"(?:\[SINGING\]|\bSINGING\b|\bSONG\b|\bSINGS\b|\bSUNG\b|\bMUSIC\b)", re.IGNORECASE)

TEXT_COLUMNS = [
    "subtitle_text",
    "aligned_raw_text",
    "aligned_script_dialogue",
    "aligned_script_text",
    "prev_other_text",
    "bridge_other_text",
    "next_other_text",
    "match_source",
]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def resolve_path(path_value: str | Path | None, base_dir: Path) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else base_dir / path


def parse_jsonish_list(value: Any) -> list[Any]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            for loader in (json.loads, ast.literal_eval):
                try:
                    out = loader(text)
                    if isinstance(out, list):
                        return out
                except Exception:
                    pass
        if " | " in text:
            return [part.strip() for part in text.split(" | ") if part.strip()]
        return [text]
    return [value]


def clean_speaker_name(name: Any) -> str:
    text = str(name).strip()
    text = re.sub(r"\s*\(.*?\)\s*$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.upper().strip()


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).replace("\xa0", " ").strip()


def unique_join(parts: list[str], sep: str = " ") -> str:
    seen: set[str] = set()
    out: list[str] = []
    for part in parts:
        text = re.sub(r"\s+", " ", clean_text(part))
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return sep.join(out)


def truncate(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", clean_text(text))
    return text if len(text) <= max_chars else text[: max_chars - 3].rstrip() + "..."


def load_full_context(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"movie_id", "shot_idx", "shot_start_time", "shot_end_time"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"full_context is missing required columns: {missing}")

    for col in TEXT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    if "subtitle_sentences" not in df.columns:
        df["subtitle_sentences"] = [[] for _ in range(len(df))]
    else:
        df["subtitle_sentences"] = df["subtitle_sentences"].apply(parse_jsonish_list)

    if "aligned_speakers" not in df.columns:
        df["aligned_speakers"] = [[] for _ in range(len(df))]
    else:
        df["aligned_speakers"] = df["aligned_speakers"].apply(parse_jsonish_list)

    df["aligned_speakers"] = df["aligned_speakers"].apply(
        lambda xs: [clean_speaker_name(x) for x in xs if clean_speaker_name(x)]
    )

    if "cast_pids" not in df.columns:
        df["cast_pids"] = [[] for _ in range(len(df))]
    else:
        df["cast_pids"] = df["cast_pids"].apply(parse_jsonish_list)

    if "shot_id" not in df.columns:
        df["shot_id"] = df["shot_idx"].apply(lambda x: f"shot_{int(x):04d}")

    df["shot_idx"] = df["shot_idx"].astype(int)
    df["shot_start_time"] = df["shot_start_time"].astype(float)
    df["shot_end_time"] = df["shot_end_time"].astype(float)
    df["shot_duration"] = df["shot_end_time"] - df["shot_start_time"]
    df["has_dialogue_text"] = (
        df["subtitle_text"].str.strip().ne("")
        | df["aligned_script_dialogue"].str.strip().ne("")
        | df["aligned_raw_text"].str.strip().ne("")
    )
    df["utterance_count_shot"] = df.apply(count_shot_utterances, axis=1)
    df["main_speaker"] = df["aligned_speakers"].apply(lambda xs: xs[0] if xs else "")
    return df.sort_values("shot_idx").reset_index(drop=True)


def count_shot_utterances(row: pd.Series) -> int:
    sentences = [x for x in row.get("subtitle_sentences", []) if clean_text(x)]
    if sentences:
        return len(sentences)
    text = clean_text(row.get("subtitle_text", ""))
    if " | " in text:
        return len([part for part in text.split(" | ") if part.strip()])
    return 1 if text else 0


def compute_speaker_changes(speakers: list[str]) -> int:
    cleaned = [clean_speaker_name(x) for x in speakers if clean_speaker_name(x)]
    return sum(1 for a, b in zip(cleaned[:-1], cleaned[1:]) if a != b)


def collect_script_action_text(seq_df: pd.DataFrame) -> str:
    parts: list[str] = []
    for col in ["prev_other_text", "bridge_other_text", "next_other_text"]:
        if col in seq_df.columns:
            parts.extend(seq_df[col].fillna("").astype(str).tolist())
    return unique_join(parts, sep="\n")


def extract_visual_cues(script_action_text: str) -> dict[str, Any]:
    headings = [re.sub(r"\s+", " ", x.strip()).upper() for x in SCENE_HEADING_RE.findall(script_action_text)]
    camera_count = len(CAMERA_CUE_RE.findall(script_action_text))
    gaze_count = len(GAZE_VERB_RE.findall(script_action_text))
    movement_count = len(MOVEMENT_CUE_RE.findall(script_action_text))
    unique_headings = sorted(set(headings))
    scene_boundary_count = max(len(unique_headings) - 1, 0)
    visual_action_score = min(
        10.0,
        camera_count * 1.0 + gaze_count * 1.5 + movement_count * 0.75 + min(len(script_action_text) / 240.0, 2.0),
    )
    return {
        "visual_action_score": round(float(visual_action_score), 3),
        "camera_cue_count": int(camera_count),
        "gaze_verb_count": int(gaze_count),
        "movement_cue_count": int(movement_count),
        "scene_heading_count": int(len(headings)),
        "scene_boundary_count": int(scene_boundary_count),
        "script_action_preview": truncate(script_action_text, 350),
    }


def is_probable_narration_sequence(active_speakers: list[str]) -> bool:
    return bool(NARRATION_SPEAKER_RE.search(" ".join(active_speakers)))


def is_probable_singing_sequence(seq_df: pd.DataFrame) -> bool:
    text_cols = ["subtitle_text", "aligned_script_dialogue", "prev_other_text", "bridge_other_text", "next_other_text"]
    blob = " ".join(seq_df[col].fillna("").astype(str).str.cat(sep=" ") for col in text_cols if col in seq_df.columns)
    return bool(SINGING_TEXT_RE.search(blob))


def score_sequence(row: dict[str, Any], ideal_min_sec: float = 8.0, ideal_max_sec: float = 18.0) -> float:
    duration = float(row["total_sec"])
    if duration < ideal_min_sec:
        duration_score = duration / ideal_min_sec
    elif duration > ideal_max_sec:
        duration_score = max(0.0, 1.0 - (duration - ideal_max_sec) / max(ideal_max_sec, 1.0))
    else:
        duration_score = 1.0

    score = 0.0
    score += min(float(row["speaker_changes"]), 5.0) * 2.0
    score += min(float(row["utterance_count"]), 12.0) * 0.8
    score += min(float(row["active_speaker_count"]), 4.0) * 1.5
    score += duration_score * 3.0
    score += float(row["match_coverage"]) * 4.0
    score += min(float(row["cast_count_max"]), 4.0) * 0.75
    score += float(row["visual_action_score"]) * 0.7
    score += min(float(row["gaze_verb_count"]), 4.0) * 1.0
    score += min(float(row["camera_cue_count"]), 4.0) * 0.6
    score -= float(row["scene_boundary_count"]) * 2.0
    return round(score, 4)


def build_candidate_sequences(
    df: pd.DataFrame,
    *,
    min_duration_sec: float = 5.0,
    max_duration_sec: float = 20.0,
    min_shots: int = 3,
    max_shots: int = 10,
    min_utterances: int = 2,
    min_speaker_changes: int = 1,
    min_avg_shot_duration: float = 1.0,
    min_active_speakers: int = 1,
    max_active_speakers: int = 4,
    require_all_shots_dialogue: bool = True,
    allow_unknown_speaker: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n = len(df)
    for start in range(n):
        for end in range(start + min_shots - 1, min(start + max_shots, n)):
            seq_df = df.iloc[start : end + 1].copy()
            shot_idxs = seq_df["shot_idx"].astype(int).tolist()
            if shot_idxs != list(range(shot_idxs[0], shot_idxs[0] + len(shot_idxs))):
                continue

            total_sec = float(seq_df["shot_end_time"].iloc[-1] - seq_df["shot_start_time"].iloc[0])
            shot_count = len(seq_df)
            avg_shot_sec = total_sec / max(shot_count, 1)
            if total_sec < min_duration_sec or total_sec > max_duration_sec:
                continue
            if avg_shot_sec < min_avg_shot_duration:
                continue
            if require_all_shots_dialogue and not bool(seq_df["has_dialogue_text"].all()):
                continue

            utterance_count = int(seq_df["utterance_count_shot"].sum())
            if utterance_count < min_utterances:
                continue

            speaker_list: list[str] = []
            for speakers in seq_df["aligned_speakers"].tolist():
                speaker_list.extend(clean_speaker_name(s) for s in speakers if clean_speaker_name(s))
            active_speakers = sorted(set(speaker_list))
            active_speaker_count = len(active_speakers)
            if active_speaker_count == 0 and not allow_unknown_speaker:
                continue
            if active_speaker_count > 0 and active_speaker_count < min_active_speakers:
                continue
            if active_speaker_count > max_active_speakers:
                continue

            speaker_changes = compute_speaker_changes(seq_df["main_speaker"].tolist())
            if speaker_changes < min_speaker_changes:
                continue
            if is_probable_narration_sequence(active_speakers) or is_probable_singing_sequence(seq_df):
                continue

            script_action_text = collect_script_action_text(seq_df)
            visual = extract_visual_cues(script_action_text)
            cast_pid_union = sorted(
                {
                    str(pid)
                    for pids in seq_df["cast_pids"].tolist()
                    for pid in (pids if isinstance(pids, list) else [])
                    if str(pid).strip()
                }
            )
            match_source_counts = seq_df["match_source"].fillna("").astype(str).value_counts().to_dict()
            matched_count = int((seq_df["match_source"].fillna("").astype(str).str.strip() != "").sum())
            row = {
                "sequence_id": f"{seq_df['movie_id'].iloc[0]}_seq_{shot_idxs[0]:04d}_{shot_idxs[-1]:04d}",
                "movie_id": seq_df["movie_id"].iloc[0],
                "start_shot_idx": int(shot_idxs[0]),
                "end_shot_idx": int(shot_idxs[-1]),
                "start_time_sec": round(float(seq_df["shot_start_time"].iloc[0]), 3),
                "end_time_sec": round(float(seq_df["shot_end_time"].iloc[-1]), 3),
                "start_time_hms": clean_text(seq_df["shot_start_time_hms"].iloc[0]) if "shot_start_time_hms" in seq_df else "",
                "end_time_hms": clean_text(seq_df["shot_end_time_hms"].iloc[-1]) if "shot_end_time_hms" in seq_df else "",
                "shot_count": int(shot_count),
                "total_sec": round(total_sec, 3),
                "avg_shot_sec": round(avg_shot_sec, 3),
                "utterance_count": int(utterance_count),
                "active_speakers": active_speakers,
                "active_speaker_count": int(active_speaker_count),
                "speaker_changes": int(speaker_changes),
                "cast_pid_union": cast_pid_union,
                "cast_count_max": int(max((len(x) for x in seq_df["cast_pids"].tolist() if isinstance(x, list)), default=0)),
                "match_source_counts": match_source_counts,
                "match_coverage": round(matched_count / max(shot_count, 1), 4),
                "subtitle_preview": truncate(" ".join(seq_df["subtitle_text"].fillna("").astype(str).tolist()), 350),
                **visual,
            }
            row["score"] = score_sequence(row)
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values(
        by=["score", "speaker_changes", "utterance_count", "visual_action_score", "shot_count"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)


def select_non_overlapping_sequences(candidates_df: pd.DataFrame, max_sequences: int | None = None) -> pd.DataFrame:
    if candidates_df.empty:
        return candidates_df.copy()
    selected: list[dict[str, Any]] = []
    occupied: set[int] = set()
    for _, row in candidates_df.iterrows():
        shot_range = set(range(int(row["start_shot_idx"]), int(row["end_shot_idx"]) + 1))
        if occupied & shot_range:
            continue
        selected.append(row.to_dict())
        occupied |= shot_range
        if max_sequences is not None and len(selected) >= max_sequences:
            break
    return pd.DataFrame(selected)


def row_to_shot_dict(row: pd.Series) -> dict[str, Any]:
    return {
        "shot_idx": int(row["shot_idx"]),
        "shot_id": clean_text(row.get("shot_id", "")),
        "shot_start_time": float(row["shot_start_time"]),
        "shot_end_time": float(row["shot_end_time"]),
        "shot_start_time_hms": clean_text(row.get("shot_start_time_hms", "")),
        "shot_end_time_hms": clean_text(row.get("shot_end_time_hms", "")),
        "subtitle_text": clean_text(row.get("subtitle_text", "")),
        "aligned_script_dialogue": clean_text(row.get("aligned_script_dialogue", "")),
        "aligned_speakers": row.get("aligned_speakers", []) if isinstance(row.get("aligned_speakers", []), list) else [],
        "prev_other_text": clean_text(row.get("prev_other_text", "")),
        "bridge_other_text": clean_text(row.get("bridge_other_text", "")),
        "next_other_text": clean_text(row.get("next_other_text", "")),
        "cast_pids": row.get("cast_pids", []) if isinstance(row.get("cast_pids", []), list) else [],
        "num_cast": int(float(row.get("num_cast", 0) or 0)),
        "match_source": clean_text(row.get("match_source", "")),
        "match_score": float(row.get("match_score", 0.0) or 0.0),
    }


def build_sequence_packages(selected_df: pd.DataFrame, full_df: pd.DataFrame) -> list[dict[str, Any]]:
    packages: list[dict[str, Any]] = []
    for _, seq in selected_df.iterrows():
        seq_df = full_df[
            full_df["shot_idx"].between(int(seq["start_shot_idx"]), int(seq["end_shot_idx"]))
        ].copy()
        packages.append(
            {
                "sequence_id": seq["sequence_id"],
                "movie_id": seq["movie_id"],
                "start_shot_idx": int(seq["start_shot_idx"]),
                "end_shot_idx": int(seq["end_shot_idx"]),
                "start_time_sec": float(seq["start_time_sec"]),
                "end_time_sec": float(seq["end_time_sec"]),
                "total_sec": float(seq["total_sec"]),
                "shot_count": int(seq["shot_count"]),
                "active_speakers": seq["active_speakers"] if isinstance(seq["active_speakers"], list) else [],
                "score": float(seq["score"]),
                "script_action_preview": clean_text(seq.get("script_action_preview", "")),
                "shots": [row_to_shot_dict(row) for _, row in seq_df.iterrows()],
            }
        )
    return packages


def serialize_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["active_speakers", "cast_pid_union", "match_source_counts"]:
        if col in out.columns:
            out[col] = out[col].apply(lambda x: json.dumps(x, ensure_ascii=False))
    return out


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def values_from_run_config(run_config_path: Path) -> dict[str, Any]:
    run_config = load_yaml(run_config_path)
    config_base_dir = run_config_path.parent.parent.parent
    paths_config_path = resolve_path(run_config.get("inputs", {}).get("paths_config"), config_base_dir)
    paths_config = load_yaml(paths_config_path) if paths_config_path is not None and paths_config_path.exists() else {}
    project_root = Path(paths_config.get("project", {}).get("root", config_base_dir))

    def from_project(path_value: str | Path | None) -> Path | None:
        return resolve_path(path_value, project_root)

    outputs = run_config.get("outputs", {})
    stage = run_config.get("stages", {}).get("build_candidate_sequences", {})
    logs_dir = from_project(outputs.get("logs_dir")) or project_root / "outputs" / "logs"
    return {
        "enabled": bool(stage.get("enabled", True)),
        "overwrite": bool(stage.get("overwrite", False)),
        "movie_id": run_config.get("data", {}).get("movie_id"),
        "full_context_csv": from_project(outputs.get("full_context_csv")),
        "output_csv": from_project(outputs.get("candidate_sequences_csv")),
        "output_jsonl": from_project(outputs.get("candidate_sequences_jsonl")),
        "all_candidates_csv": logs_dir / "02_candidate_sequences_all.csv",
        "summary_json": logs_dir / "02_candidate_sequences_summary.json",
        "min_duration_sec": float(stage.get("min_duration_sec", 5)),
        "max_duration_sec": float(stage.get("max_duration_sec", 20)),
        "min_shots": int(stage.get("min_shots", 3)),
        "max_shots": int(stage.get("max_shots", 10)),
        "min_utterances": int(stage.get("min_utterances", 2)),
        "min_speaker_changes": int(stage.get("min_speaker_changes", 1)),
        "min_avg_shot_duration": float(stage.get("min_avg_shot_duration", 1.0)),
        "min_active_speakers": int(stage.get("min_active_speakers", 1)),
        "max_active_speakers": int(stage.get("max_active_speakers", 4)),
        "require_all_shots_dialogue": bool(stage.get("require_all_shots_dialogue", True)),
        "allow_unknown_speaker": bool(stage.get("allow_unknown_speaker", True)),
        "select_non_overlapping": bool(stage.get("select_non_overlapping", True)),
        "max_sequences": stage.get("max_sequences"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build candidate shot sequences from full_context.")
    parser.add_argument("--run-config", type=Path, default=None)
    parser.add_argument("--full-context", dest="full_context_csv", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--all-candidates-csv", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--min-duration-sec", type=float, default=None)
    parser.add_argument("--max-duration-sec", type=float, default=None)
    parser.add_argument("--min-shots", type=int, default=None)
    parser.add_argument("--max-shots", type=int, default=None)
    parser.add_argument("--min-utterances", type=int, default=None)
    parser.add_argument("--min-speaker-changes", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values: dict[str, Any] = {}
    if args.run_config is not None:
        values.update(values_from_run_config(args.run_config))

    if not bool(values.get("enabled", True)):
        print("Skipping build_candidate_sequences because it is disabled in the run config.")
        return

    full_context_csv = args.full_context_csv or values.get("full_context_csv")
    output_csv = args.output_csv or values.get("output_csv")
    output_jsonl = args.output_jsonl or values.get("output_jsonl")
    all_candidates_csv = args.all_candidates_csv or values.get("all_candidates_csv")
    summary_json = args.summary_json or values.get("summary_json")
    if full_context_csv is None or output_csv is None or output_jsonl is None:
        raise ValueError("full context, output CSV, and output JSONL paths are required.")

    full_context_csv = Path(full_context_csv)
    output_csv = Path(output_csv)
    output_jsonl = Path(output_jsonl)
    all_candidates_csv = Path(all_candidates_csv) if all_candidates_csv is not None else None
    summary_json = Path(summary_json) if summary_json is not None else None

    overwrite = bool(args.overwrite or values.get("overwrite", False))
    existing = [path for path in [output_csv, output_jsonl] if path.exists()]
    if existing and not overwrite:
        print("Skipping build_candidate_sequences because outputs already exist and overwrite=false.")
        for path in existing:
            print(f"Existing output: {path}")
        return

    max_sequences = args.max_sequences if args.max_sequences is not None else values.get("max_sequences")
    max_sequences = int(max_sequences) if max_sequences is not None else None

    full_df = load_full_context(full_context_csv)
    all_df = build_candidate_sequences(
        full_df,
        min_duration_sec=args.min_duration_sec if args.min_duration_sec is not None else values.get("min_duration_sec", 5.0),
        max_duration_sec=args.max_duration_sec if args.max_duration_sec is not None else values.get("max_duration_sec", 20.0),
        min_shots=args.min_shots if args.min_shots is not None else values.get("min_shots", 3),
        max_shots=args.max_shots if args.max_shots is not None else values.get("max_shots", 10),
        min_utterances=args.min_utterances if args.min_utterances is not None else values.get("min_utterances", 2),
        min_speaker_changes=args.min_speaker_changes
        if args.min_speaker_changes is not None
        else values.get("min_speaker_changes", 1),
        min_avg_shot_duration=values.get("min_avg_shot_duration", 1.0),
        min_active_speakers=values.get("min_active_speakers", 1),
        max_active_speakers=values.get("max_active_speakers", 4),
        require_all_shots_dialogue=values.get("require_all_shots_dialogue", True),
        allow_unknown_speaker=values.get("allow_unknown_speaker", True),
    )
    if bool(values.get("select_non_overlapping", True)):
        selected_df = select_non_overlapping_sequences(all_df, max_sequences=max_sequences)
    else:
        selected_df = all_df.head(max_sequences).copy() if max_sequences is not None else all_df.copy()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    serialize_for_csv(selected_df).to_csv(output_csv, index=False, encoding="utf-8-sig")
    packages = build_sequence_packages(selected_df, full_df)
    write_jsonl(packages, output_jsonl)

    if all_candidates_csv is not None:
        all_candidates_csv.parent.mkdir(parents=True, exist_ok=True)
        serialize_for_csv(all_df).to_csv(all_candidates_csv, index=False, encoding="utf-8-sig")

    summary = {
        "movie_id": values.get("movie_id") or (full_df["movie_id"].iloc[0] if len(full_df) else None),
        "full_context_csv": str(full_context_csv),
        "output_csv": str(output_csv),
        "output_jsonl": str(output_jsonl),
        "all_candidates_csv": str(all_candidates_csv) if all_candidates_csv is not None else None,
        "row_count": int(len(full_df)),
        "all_candidate_count": int(len(all_df)),
        "selected_candidate_count": int(len(selected_df)),
        "max_sequences": max_sequences,
        "visual_candidates_with_score": int((all_df.get("visual_action_score", pd.Series(dtype=float)).fillna(0) > 0).sum())
        if len(all_df)
        else 0,
        "camera_cue_total": int(all_df.get("camera_cue_count", pd.Series(dtype=int)).fillna(0).sum()) if len(all_df) else 0,
        "gaze_verb_total": int(all_df.get("gaze_verb_count", pd.Series(dtype=int)).fillna(0).sum()) if len(all_df) else 0,
        "movement_cue_total": int(all_df.get("movement_cue_count", pd.Series(dtype=int)).fillna(0).sum()) if len(all_df) else 0,
        "top_sequences": selected_df.head(10)[["sequence_id", "score", "start_shot_idx", "end_shot_idx", "total_sec"]].to_dict(
            orient="records"
        )
        if len(selected_df)
        else [],
    }
    if summary_json is not None:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Movie: {summary['movie_id']}")
    print(f"Full context rows: {summary['row_count']}")
    print(f"All candidates: {summary['all_candidate_count']}")
    print(f"Selected candidates: {summary['selected_candidate_count']}")
    print(f"Candidate sequences CSV: {output_csv}")
    print(f"Candidate sequences JSONL: {output_jsonl}")
    if all_candidates_csv is not None:
        print(f"All candidates CSV: {all_candidates_csv}")
    if summary_json is not None:
        print(f"Summary: {summary_json}")


if __name__ == "__main__":
    main()
