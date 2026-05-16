#!/usr/bin/env python3
"""Compare OpenFace gaze/head-pose evidence with optional L2CS and 6DRepNet outputs."""

from __future__ import annotations

import argparse
import csv
import html
import importlib.util
import json
import math
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


def dependency_status() -> tuple[dict[str, bool], list[str]]:
    status = {name: importlib.util.find_spec(name) is not None for name in ["torch", "cv2", "numpy"]}
    missing = [name for name, ok in status.items() if not ok]
    return status, missing


def run_l2cs_placeholder() -> dict[str, str]:
    return {"l2cs_yaw": "", "l2cs_pitch": "", "l2cs_direction": ""}


def run_sixd_placeholder() -> dict[str, str]:
    return {"sixd_yaw": "", "sixd_pitch": "", "sixd_roll": "", "sixd_pose_direction": ""}


def build_compare_rows(base_dir: Path, output_dir: Path, movies: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    deps, missing = dependency_status()
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
                "human_label": sample.get("human_label", ""),
                "human_notes": sample.get("human_notes", ""),
                "openface_gaze_x": gaze_x,
                "openface_gaze_y": gaze_y,
                "openface_pose_Ry": pose_ry,
                "openface_pose_Rx": pose_rx,
                "openface_gaze_direction": openface_gaze,
                "openface_pose_direction": openface_pose,
                "openface_conflict": gaze_pose_conflict(openface_gaze, openface_pose),
                "l2cs_vs_openface_gaze_agree": "",
                "sixd_vs_openface_pose_agree": "",
                "inference_status": "baseline_only" if missing else "model_inference_not_implemented",
                "error_note": f"missing dependencies: {','.join(missing)}" if missing else "model hooks are placeholders",
            }
            row.update(run_l2cs_placeholder())
            row.update(run_sixd_placeholder())
            all_rows.append(row)
    summary = {
        "row_count": len(all_rows),
        "movies": movies,
        "dependency_status": deps,
        "l2cs_available": bool(deps.get("torch") and deps.get("cv2") and deps.get("numpy")),
        "sixdrepnet_available": bool(deps.get("torch") and deps.get("cv2") and deps.get("numpy")),
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
        image_rel = relpath_for_html(str(row.get("overlay_path", "")), report_path)
        image_html = f'<img src="{fmt(image_rel)}" alt="debug overlay">' if image_rel else '<div class="placeholder">No overlay</div>'
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
            ("crop video", row.get("crop_video_path")),
            ("status", f"{row.get('inference_status')} {row.get('error_note')}"),
        ]
        dl = "".join(f"<dt>{fmt(k)}</dt><dd>{fmt(v)}</dd>" for k, v in details)
        cards.append(f'<article class="{" ".join(classes)}"><h2>#{idx} {fmt(row.get("movie_id"))} {fmt(row.get("shot_id"))} bin {fmt(row.get("bin_idx"))}</h2><div class="content"><div>{image_html}</div><dl>{dl}</dl></div></article>')
    css = """
    body{font-family:Arial,sans-serif;margin:24px;background:#f7f7f5;color:#202020}
    .summary,.card{background:white;border:1px solid #ddd;border-radius:8px;padding:14px;margin:16px 0}
    .conflict{border-color:#d18b00;background:#fffaf0}.badtrack{border-color:#a33;background:#fff6f6}
    .content{display:grid;grid-template-columns:minmax(320px,42%) 1fr;gap:16px}
    img{max-width:100%;border:1px solid #ccc;border-radius:6px;background:#222}
    .placeholder{min-height:220px;display:grid;place-items:center;background:#eee;border:1px dashed #aaa}
    dl{display:grid;grid-template-columns:120px 1fr;gap:6px 12px;margin:0}dt{font-weight:700;color:#555}dd{margin:0;overflow-wrap:anywhere}
    h2{font-size:18px}@media(max-width:900px){.content{grid-template-columns:1fr}}
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
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    movies = [item.strip() for item in args.movies.split(",") if item.strip()]
    rows, summary = build_compare_rows(args.base_dir, args.output_dir, movies)
    write_csv(args.output_dir / "gaze_evidence_compare.csv", rows, OUTPUT_COLUMNS)
    write_json(args.output_dir / "gaze_evidence_bakeoff_summary.json", summary)
    render_html(rows, summary, args.output_dir)
    print(f"[Bakeoff] rows={len(rows)} csv={args.output_dir / 'gaze_evidence_compare.csv'}")
    if not summary["l2cs_available"] or not summary["sixdrepnet_available"]:
        print("[Bakeoff] model inference skipped:", summary["dependency_status"])


if __name__ == "__main__":
    main()
