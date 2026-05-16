from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def sec_to_hms(seconds: float | None) -> str:
    if seconds is None:
        return ""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def normalize_subtitle_sentences(sentences: Any) -> str:
    if not isinstance(sentences, list):
        return ""
    return " | ".join(s.strip() for s in sentences if isinstance(s, str) and s.strip())


def build_cast_name_map(meta_cast_list: list[dict[str, Any]]) -> dict[str, str]:
    cast_name_map: dict[str, str] = {}
    for item in meta_cast_list:
        pid = item.get("id")
        name = item.get("name")
        if pid is None or name is None:
            continue
        cast_name_map[str(pid)] = str(name)
    return cast_name_map


def build_shot_cast_map(
    cast_list: list[dict[str, Any]],
    cast_name_map: dict[str, str],
) -> dict[int, dict[str, Any]]:
    shot_to_names: dict[int, set[str]] = defaultdict(set)

    for item in cast_list:
        shot_idx = item.get("shot_idx")
        pid = item.get("pid")
        if shot_idx is None or pid is None:
            continue
        pid_str = str(pid)
        shot_to_names[int(shot_idx)].add(cast_name_map.get(pid_str, pid_str))

    shot_cast_map: dict[int, dict[str, Any]] = {}
    for shot_idx, name_set in shot_to_names.items():
        name_list = sorted(name_set)
        shot_cast_map[shot_idx] = {
            "cast_pids": name_list,
            "num_cast": len(name_list),
        }
    return shot_cast_map


def extract_shot_level_rows(movie_id: str, annotation_dir: Path, meta_dir: Path) -> list[dict[str, Any]]:
    ann_path = annotation_dir / f"{movie_id}.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")
    meta_path = meta_dir / f"{movie_id}.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")

    with ann_path.open("r", encoding="utf-8") as f:
        ann = json.load(f)
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    cast_list = ann.get("cast", [])
    story_list = ann.get("story", [])
    cast_name_map = build_cast_name_map(meta.get("cast", []))
    shot_cast_map = build_shot_cast_map(cast_list, cast_name_map)
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
                    "subtitle_sentences": normalize_subtitle_sentences(sentences),
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


def save_shot_level_outputs(df: pd.DataFrame, output_csv_path: Path | None, output_jsonl_path: Path | None) -> None:
    if output_csv_path is None and output_jsonl_path is None:
        raise ValueError("At least one output path must be provided.")

    if output_csv_path is not None:
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_to_save = df.copy()
        if "cast_pids" in df_to_save.columns:
            df_to_save["cast_pids"] = df_to_save["cast_pids"].apply(
                lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else x
            )

        df_to_save.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

    if output_jsonl_path is not None:
        output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with output_jsonl_path.open("w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")


def process_one_movie_to_shot_level(
    movie_id: str,
    annotation_dir: Path,
    output_dir: Path | None = None,
    meta_dir: Path | None = None,
    output_csv_path: Path | None = None,
    output_jsonl_path: Path | None = None,
) -> tuple[pd.DataFrame, Path | None, Path | None]:
    resolved_meta_dir = meta_dir if meta_dir is not None else annotation_dir.parent / "meta"
    rows = extract_shot_level_rows(movie_id, annotation_dir, resolved_meta_dir)
    df = rows_to_dataframe(rows)
    if output_csv_path is None and output_jsonl_path is None:
        if output_dir is None:
            raise ValueError("output_dir is required when explicit output paths are not provided.")
        csv_path: Path | None = output_dir / f"{movie_id}_shot_level.csv"
        jsonl_path: Path | None = output_dir / f"{movie_id}_shot_level.jsonl"
    else:
        csv_path = output_csv_path
        jsonl_path = output_jsonl_path
    save_shot_level_outputs(df, csv_path, jsonl_path)
    return df, csv_path, jsonl_path


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def resolve_path(path_value: str | Path | None, base_dir: Path) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else base_dir / path


def setup_logger(log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger("expregaze.process_shot_level")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def load_json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def count_items(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if not isinstance(value, (str, bytes)):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
    return value


def build_debug_summary(
    movie_id: str,
    annotation_path: Path,
    meta_path: Path,
    output_csv_path: Path | None,
    output_jsonl_path: Path | None,
    df: pd.DataFrame,
) -> dict[str, Any]:
    ann = load_json_file(annotation_path)
    meta = load_json_file(meta_path)
    story_list = ann.get("story", []) if isinstance(ann, dict) else []
    cast_list = ann.get("cast", []) if isinstance(ann, dict) else []
    meta_cast_list = meta.get("cast", []) if isinstance(meta, dict) else []

    output_file_sizes: dict[str, int] = {}
    for label, path in (("csv", output_csv_path), ("jsonl", output_jsonl_path)):
        if path is not None and path.exists():
            output_file_sizes[label] = path.stat().st_size

    empty_subtitle_count = 0
    empty_cast_count = 0
    null_time_count = 0
    unique_shot_count = 0
    duplicate_shot_row_count = 0
    min_shot_start_time = None
    max_shot_end_time = None

    if not df.empty:
        empty_subtitle_count = int((df["subtitle_sentences"].fillna("").astype(str).str.strip() == "").sum())
        empty_cast_count = int((df["num_cast"].fillna(0).astype(int) == 0).sum())
        null_time_count = int((df["shot_start_time"].isna() | df["shot_end_time"].isna()).sum())
        unique_shot_count = int(df["shot_idx"].nunique())
        duplicate_shot_row_count = int(len(df) - unique_shot_count)
        min_shot_start_time = df["shot_start_time"].min()
        max_shot_end_time = df["shot_end_time"].max()

    return json_safe(
        {
            "movie_id": movie_id,
            "annotation_path": str(annotation_path),
            "meta_path": str(meta_path),
            "output_csv_path": str(output_csv_path) if output_csv_path is not None else None,
            "output_jsonl_path": str(output_jsonl_path) if output_jsonl_path is not None else None,
            "story_count": count_items(story_list),
            "annotation_cast_count": count_items(cast_list),
            "meta_cast_count": count_items(meta_cast_list),
            "row_count": int(len(df)),
            "unique_shot_count": unique_shot_count,
            "duplicate_shot_row_count": duplicate_shot_row_count,
            "empty_subtitle_count": empty_subtitle_count,
            "empty_cast_count": empty_cast_count,
            "null_time_count": null_time_count,
            "min_shot_start_time": min_shot_start_time,
            "max_shot_end_time": max_shot_end_time,
            "columns": list(df.columns),
            "first_rows": df.head(5).to_dict(orient="records"),
            "output_file_sizes": output_file_sizes,
        }
    )


def write_debug_outputs(summary: dict[str, Any], summary_path: Path | None, preview_path: Path | None, df: pd.DataFrame) -> None:
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if preview_path is not None:
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        preview_df = df.head(25).copy()
        if "cast_pids" in preview_df.columns:
            preview_df["cast_pids"] = preview_df["cast_pids"].apply(
                lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else x
            )
        preview_df.to_csv(preview_path, index=False, encoding="utf-8-sig")


def cli_values_from_run_config(run_config_path: Path) -> dict[str, Any]:
    run_config = load_yaml(run_config_path)
    config_base_dir = run_config_path.parent.parent.parent

    paths_config_path = resolve_path(run_config.get("inputs", {}).get("paths_config"), config_base_dir)
    paths_config = load_yaml(paths_config_path) if paths_config_path is not None else {}

    project_root = paths_config.get("project", {}).get("root")
    project_root_path = Path(project_root) if project_root is not None else config_base_dir

    def from_project(path_value: str | Path | None) -> Path | None:
        return resolve_path(path_value, project_root_path)

    outputs = run_config.get("outputs", {})
    logs_dir = from_project(outputs.get("logs_dir")) or project_root_path / "outputs" / "logs"
    paths = paths_config.get("paths", {})

    return {
        "enabled": bool(run_config.get("stages", {}).get("process_shot_level", {}).get("enabled", True)),
        "overwrite": bool(run_config.get("stages", {}).get("process_shot_level", {}).get("overwrite", False)),
        "movie_id": run_config.get("data", {}).get("movie_id"),
        "annotation_dir": from_project(paths.get("annotation_dir")),
        "meta_dir": from_project(paths.get("meta_dir")),
        "output_csv_path": from_project(outputs.get("shot_level_csv")),
        "output_jsonl_path": from_project(outputs.get("shot_level_jsonl")),
        "log_file": logs_dir / "00_process_one_movie_to_shot_level.log",
        "debug_summary_json": logs_dir / "00_process_one_movie_to_shot_level_summary.json",
        "preview_csv": logs_dir / "00_process_one_movie_to_shot_level_preview.csv",
        "base_config_path": from_project(run_config.get("inputs", {}).get("base_config")),
        "paths_config_path": paths_config_path,
        "movie_config_path": from_project(run_config.get("run", {}).get("movie_config")),
    }


def validate_config_paths(values: dict[str, Any]) -> None:
    for key in ("base_config_path", "paths_config_path", "movie_config_path"):
        path = values.get(key)
        if path is not None and not path.exists():
            raise FileNotFoundError(f"Config file not found from run config field {key}: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MovieNet shot-level CSV/JSONL for one movie.")
    parser.add_argument("--run-config", type=Path, default=None, help="Run YAML, e.g. configs/runs/text_main_tt1591095.yaml")
    parser.add_argument("--movie-id", type=str, default=None)
    parser.add_argument("--annotation-dir", type=Path, default=None)
    parser.add_argument("--meta-dir", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--debug-summary-json", type=Path, default=None)
    parser.add_argument("--preview-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    values: dict[str, Any] = {}
    if args.run_config is not None:
        values.update(cli_values_from_run_config(args.run_config))
        validate_config_paths(values)

    movie_id = args.movie_id or values.get("movie_id")
    annotation_dir = args.annotation_dir or values.get("annotation_dir")
    meta_dir = args.meta_dir or values.get("meta_dir")
    output_csv_path = args.output_csv or values.get("output_csv_path")
    output_jsonl_path = args.output_jsonl or values.get("output_jsonl_path")
    overwrite = bool(args.overwrite or values.get("overwrite", False))
    enabled = bool(values.get("enabled", True))
    log_file = args.log_file or values.get("log_file")
    debug_summary_json = args.debug_summary_json or values.get("debug_summary_json")
    preview_csv = args.preview_csv or values.get("preview_csv")

    logger = setup_logger(log_file)

    if not enabled:
        logger.info("Skipping process_shot_level because it is disabled in the run config.")
        return

    if movie_id is None:
        raise ValueError("movie_id is required. Provide --movie-id or data.movie_id in --run-config.")
    if annotation_dir is None:
        raise ValueError("annotation_dir is required. Provide --annotation-dir or paths.annotation_dir in paths_config.")
    if output_csv_path is None and output_jsonl_path is None:
        raise ValueError("At least one output path is required. Provide --output-csv/--output-jsonl or run config outputs.")

    annotation_dir = Path(annotation_dir)
    resolved_meta_dir = Path(meta_dir) if meta_dir is not None else annotation_dir.parent / "meta"
    annotation_path = annotation_dir / f"{movie_id}.json"
    meta_path = resolved_meta_dir / f"{movie_id}.json"

    existing_outputs = [path for path in (output_csv_path, output_jsonl_path) if path is not None and path.exists()]
    expected_outputs = [path for path in (output_csv_path, output_jsonl_path) if path is not None]
    if existing_outputs and not overwrite:
        if len(existing_outputs) == len(expected_outputs):
            logger.info("Skipping process_shot_level because outputs already exist and overwrite=false.")
            for path in existing_outputs:
                logger.info("Existing output: %s", path)
            return
        existing_text = ", ".join(str(path) for path in existing_outputs)
        raise FileExistsError(f"Refusing to overwrite partial existing outputs with overwrite=false: {existing_text}")

    logger.info("Processing movie_id=%s", movie_id)
    logger.info("Annotation: %s", annotation_path)
    logger.info("Meta: %s", meta_path)

    df, csv_path, jsonl_path = process_one_movie_to_shot_level(
        movie_id=movie_id,
        annotation_dir=annotation_dir,
        output_dir=None,
        meta_dir=resolved_meta_dir,
        output_csv_path=Path(output_csv_path) if output_csv_path is not None else None,
        output_jsonl_path=Path(output_jsonl_path) if output_jsonl_path is not None else None,
    )

    summary = build_debug_summary(
        movie_id=movie_id,
        annotation_path=annotation_path,
        meta_path=meta_path,
        output_csv_path=csv_path,
        output_jsonl_path=jsonl_path,
        df=df,
    )
    write_debug_outputs(
        summary=summary,
        summary_path=Path(debug_summary_json) if debug_summary_json is not None else None,
        preview_path=Path(preview_csv) if preview_csv is not None else None,
        df=df,
    )

    logger.info("Rows: %s", len(df))
    logger.info("Unique shots: %s", summary["unique_shot_count"])
    if csv_path is not None:
        logger.info("CSV: %s", csv_path)
    if jsonl_path is not None:
        logger.info("JSONL: %s", jsonl_path)
    if debug_summary_json is not None:
        logger.info("Debug summary: %s", debug_summary_json)
    if preview_csv is not None:
        logger.info("Preview CSV: %s", preview_csv)


if __name__ == "__main__":
    main()
