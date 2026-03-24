from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


def sec_to_hms(seconds: float | None) -> str:
    if seconds is None:
        return ""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def safe_join_sentences(sentences: list[str]) -> str:
    if not sentences:
        return ""
    return " ".join(s.strip() for s in sentences if isinstance(s, str) and s.strip())


def build_shot_cast_map(cast_list: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    shot_to_pids: dict[int, set[str]] = defaultdict(set)

    for item in cast_list:
        shot_idx = item.get("shot_idx")
        pid = item.get("pid")
        if shot_idx is None or pid is None:
            continue
        shot_to_pids[int(shot_idx)].add(str(pid))

    shot_cast_map: dict[int, dict[str, Any]] = {}
    for shot_idx, pid_set in shot_to_pids.items():
        pid_list = sorted(pid_set)
        shot_cast_map[shot_idx] = {
            "cast_pids": pid_list,
            "num_cast": len(pid_list),
        }
    return shot_cast_map


def extract_shot_level_rows(movie_id: str, annotation_dir: Path) -> list[dict[str, Any]]:
    ann_path = annotation_dir / f"{movie_id}.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    with ann_path.open("r", encoding="utf-8") as f:
        ann = json.load(f)

    cast_list = ann.get("cast", [])
    story_list = ann.get("story", [])
    shot_cast_map = build_shot_cast_map(cast_list)
    rows: list[dict[str, Any]] = []

    if not isinstance(story_list, list):
        return rows

    for story_item in story_list:
        story_id = story_item.get("id", "")
        story_description = story_item.get("description", "")
        story_subtitle_list = story_item.get("subtitle", [])

        if not isinstance(story_subtitle_list, list):
            continue

        for sub_item in story_subtitle_list:
            shot_idx = sub_item.get("shot")
            if shot_idx is None:
                continue

            duration = sub_item.get("duration", [None, None])
            sentences = sub_item.get("sentences", [])
            shot_start_time = duration[0] if isinstance(duration, list) and len(duration) > 0 else None
            shot_end_time = duration[1] if isinstance(duration, list) and len(duration) > 1 else None

            cast_info = shot_cast_map.get(int(shot_idx), {"cast_pids": [], "num_cast": 0})

            rows.append(
                {
                    "movie_id": movie_id,
                    "story_id": story_id,
                    "story_description": story_description,
                    "shot_idx": int(shot_idx),
                    "shot_id": f"shot_{int(shot_idx):04d}",
                    "shot_start_time": shot_start_time,
                    "shot_end_time": shot_end_time,
                    "shot_start_time_hms": sec_to_hms(shot_start_time),
                    "shot_end_time_hms": sec_to_hms(shot_end_time),
                    "subtitle_sentences": sentences,
                    "subtitle_text": safe_join_sentences(sentences),
                    "cast_pids": cast_info["cast_pids"],
                    "num_cast": cast_info["num_cast"],
                }
            )

    return rows


def rows_to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["shot_idx", "shot_start_time"], na_position="last").reset_index(drop=True)


def filter_usable_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    mask = df["subtitle_text"].fillna("").str.strip().ne("") & df["num_cast"].fillna(0).gt(0)
    return df.loc[mask].copy().reset_index(drop=True)


def save_shot_level_outputs(df: pd.DataFrame, output_csv_path: Path, output_jsonl_path: Path) -> None:
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    df_to_save = df.copy()
    if "subtitle_sentences" in df_to_save.columns:
        df_to_save["subtitle_sentences"] = df_to_save["subtitle_sentences"].apply(
            lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else x
        )
    if "cast_pids" in df_to_save.columns:
        df_to_save["cast_pids"] = df_to_save["cast_pids"].apply(
            lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else x
        )

    df_to_save.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

    with output_jsonl_path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")


def process_one_movie_to_shot_level(movie_id: str, annotation_dir: Path, output_dir: Path) -> tuple[pd.DataFrame, Path, Path]:
    rows = extract_shot_level_rows(movie_id, annotation_dir)
    df = rows_to_dataframe(rows)
    csv_path = output_dir / f"{movie_id}_shot_level.csv"
    jsonl_path = output_dir / f"{movie_id}_shot_level.jsonl"
    save_shot_level_outputs(df, csv_path, jsonl_path)
    return df, csv_path, jsonl_path
