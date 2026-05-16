#!/usr/bin/env python3
"""Build a small HTML workflow for calibrating OpenFace gaze/pose directions."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


LABEL_COLUMNS = [
    "movie_id",
    "sequence_id",
    "shot_id",
    "bin_idx",
    "subject_track_id",
    "sample_index",
    "overlay_path",
    "human_screen_direction",
    "human_notes",
    "gaze_angle_x_mean",
    "gaze_angle_y_mean",
    "pose_Ry_mean",
    "pose_Rx_mean",
    "openface_gaze_direction",
    "openface_pose_direction",
    "is_gaze_sign_correct",
    "is_pose_sign_correct",
]

REPORT_COLUMNS = LABEL_COLUMNS
DIRECTION_LABELS = ["left", "right", "up", "down", "center", "unclear", "bad_track"]


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


def fmt(value: Any) -> str:
    return html.escape(str(value or ""))


def key4(row: dict[str, str], track_col: str) -> tuple[str, str, str, str]:
    return (
        row.get("movie_id", ""),
        row.get("shot_id", ""),
        row.get("bin_idx", ""),
        row.get(track_col, ""),
    )


def direction_bucket(value: Any, threshold: float) -> str:
    parsed = safe_float(value)
    if math.isnan(parsed):
        return "unknown"
    if parsed < -threshold:
        return "left"
    if parsed > threshold:
        return "right"
    return "center"


def csv_cell(value: Any) -> str:
    value = "" if value is None else str(value)
    return '"' + value.replace('"', '""').replace("\n", " ") + '"'


def label_row_text(row: dict[str, Any], human_direction: str) -> str:
    payload = dict(row)
    payload["human_screen_direction"] = human_direction
    payload["human_notes"] = ""
    payload["is_gaze_sign_correct"] = ""
    payload["is_pose_sign_correct"] = ""
    return ",".join(csv_cell(payload.get(column, "")) for column in LABEL_COLUMNS)


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


def load_reviewed_keys(path: Path | None) -> tuple[set[tuple[str, str, str, str]], dict[tuple[str, str, str, str], dict[str, str]]]:
    if path is None or not path.exists():
        return set(), {}
    rows = [row for row in read_csv(path) if (row.get("human_label") or "").strip()]
    lookup = {key4(row, "subject_local_track_id"): row for row in rows}
    return set(lookup), lookup


def build_calibration_rows(
    audit_labels_reviewed: Path | None,
    audit_samples: Path,
    final_proxy_table: Path,
    openface_bins: Path,
    include_unreviewed: bool,
    max_samples: int,
) -> list[dict[str, Any]]:
    reviewed_keys, reviewed_lookup = load_reviewed_keys(audit_labels_reviewed)
    samples = read_csv(audit_samples)
    final_lookup = {key4(row, "subject_local_track_id"): row for row in read_csv(final_proxy_table)}
    openface_lookup = {key4(row, "local_track_id"): row for row in read_csv(openface_bins)}

    selected: list[dict[str, str]] = []
    for sample in samples:
        key = key4(sample, "subject_local_track_id")
        if key in reviewed_keys:
            selected.append(sample)
    if include_unreviewed and len(selected) < max_samples:
        seen = {key4(row, "subject_local_track_id") for row in selected}
        candidates = [row for row in samples if key4(row, "subject_local_track_id") not in seen]

        def priority(row: dict[str, str]) -> tuple[int, int, int]:
            conflict = 1 if row.get("gaze_direction_bucket") in {"left", "right"} and row.get("pose_direction_bucket") in {"left", "right"} and row.get("gaze_direction_bucket") != row.get("pose_direction_bucket") else 0
            reliable = 1 if row.get("gaze_quality") == "gaze_reliable" else 0
            pose_fallback = 1 if row.get("gaze_quality") == "pose_fallback" else 0
            return (conflict, reliable, pose_fallback)

        candidates.sort(key=priority, reverse=True)
        selected.extend(candidates[: max(0, max_samples - len(selected))])
    if max_samples > 0:
        selected = selected[:max_samples]

    rows: list[dict[str, Any]] = []
    for sample in selected:
        key = key4(sample, "subject_local_track_id")
        final = final_lookup.get(key, {})
        openface = openface_lookup.get(key, {})
        reviewed = reviewed_lookup.get(key, {})
        gaze_x = openface.get("gaze_angle_x_mean") or final.get("gaze_angle_x_mean", "")
        gaze_y = openface.get("gaze_angle_y_mean") or final.get("gaze_angle_y_mean", "")
        pose_ry = openface.get("pose_Ry_mean", "")
        pose_rx = openface.get("pose_Rx_mean", "")
        rows.append(
            {
                "movie_id": sample.get("movie_id", ""),
                "sequence_id": sample.get("sequence_id", ""),
                "shot_id": sample.get("shot_id", ""),
                "bin_idx": sample.get("bin_idx", ""),
                "subject_track_id": sample.get("subject_local_track_id", ""),
                "sample_index": sample.get("sample_index", ""),
                "overlay_path": sample.get("overlay_path", ""),
                "human_screen_direction": "",
                "human_notes": reviewed.get("human_notes", ""),
                "gaze_angle_x_mean": gaze_x,
                "gaze_angle_y_mean": gaze_y,
                "pose_Ry_mean": pose_ry,
                "pose_Rx_mean": pose_rx,
                "openface_gaze_direction": direction_bucket(gaze_x, 0.20),
                "openface_pose_direction": direction_bucket(pose_ry, 0.25),
                "is_gaze_sign_correct": "",
                "is_pose_sign_correct": "",
                "audit_human_label": reviewed.get("human_label", ""),
                "gaze_quality": final.get("gaze_quality", sample.get("gaze_quality", "")),
                "target_type": sample.get("target_type", ""),
                "target_id": sample.get("target_id", ""),
            }
        )
    return rows


def render_report(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    template_path = output_dir / "direction_labels_template.csv"
    report_path = output_dir / "direction_calibration.html"
    write_csv(template_path, rows, LABEL_COLUMNS)

    cards = []
    for row in rows:
        image_rel = relpath_for_html(str(row.get("overlay_path", "")), report_path)
        image_html = f'<img src="{fmt(image_rel)}" alt="debug frame">' if image_rel else '<div class="placeholder">No overlay frame</div>'
        label_buttons = "".join(
            f'<button type="button" onclick="copyLabel({fmt(json.dumps(row.get("sample_index", "")))}, {fmt(json.dumps(label))})">{fmt(label)}</button>'
            for label in DIRECTION_LABELS
        )
        details = [
            ("sample", row.get("sample_index")),
            ("shot/bin", f"{row.get('shot_id')} / {row.get('bin_idx')}"),
            ("subject", row.get("subject_track_id")),
            ("gaze x/y", f"{row.get('gaze_angle_x_mean')} / {row.get('gaze_angle_y_mean')}"),
            ("pose Ry/Rx", f"{row.get('pose_Ry_mean')} / {row.get('pose_Rx_mean')}"),
            ("OpenFace buckets", f"gaze={row.get('openface_gaze_direction')} pose={row.get('openface_pose_direction')}"),
            ("audit label", row.get("audit_human_label")),
            ("audit note", row.get("human_notes")),
            ("proxy target", f"{row.get('target_type')} / {row.get('target_id')}"),
        ]
        dl = "".join(f"<dt>{fmt(k)}</dt><dd>{fmt(v)}</dd>" for k, v in details)
        default_row = label_row_text(row, "")
        cards.append(
            f"""
            <article class="sample">
              <header><h2>#{fmt(row.get('sample_index'))} · {fmt(row.get('shot_id'))} bin {fmt(row.get('bin_idx'))}</h2><div>{label_buttons}</div></header>
              <div class="content"><div>{image_html}</div><div><dl>{dl}</dl><textarea id="label-{fmt(row.get('sample_index'))}" readonly>{fmt(default_row)}</textarea></div></div>
            </article>
            """
        )
    css = """
    body{font-family:Arial,sans-serif;margin:24px;background:#f7f7f5;color:#1f1f1f}
    .topbar{position:sticky;top:0;background:#f7f7f5;border-bottom:1px solid #ddd;padding:12px 0;z-index:2}
    .sample{background:#fff;border:1px solid #ddd;border-radius:8px;margin:18px 0;padding:14px}
    header{display:flex;justify-content:space-between;gap:12px;align-items:center}
    h2{font-size:18px;margin:0}.content{display:grid;grid-template-columns:minmax(320px,42%) 1fr;gap:16px;margin-top:12px}
    img{max-width:100%;border:1px solid #ccc;border-radius:6px;background:#222}.placeholder{min-height:220px;display:grid;place-items:center;background:#eee;border:1px dashed #aaa}
    dl{display:grid;grid-template-columns:130px 1fr;gap:6px 12px;margin:0}dt{font-weight:700;color:#555}dd{margin:0;overflow-wrap:anywhere}
    button{margin:3px;padding:6px 8px;border:1px solid #aaa;border-radius:6px;background:white;cursor:pointer}button:hover{background:#eef3ff}
    textarea{width:100%;min-height:64px;margin-top:12px;font-family:monospace;font-size:12px}
    @media(max-width:900px){.content{grid-template-columns:1fr}header{display:block}}
    """
    script = """
    function csvCell(value){const s=String(value??"");return '"' + s.replaceAll('"','""').replaceAll('\\n',' ') + '"';}
    function copyLabel(sampleIndex, direction){
      const textarea=document.getElementById(`label-${sampleIndex}`);
      const parts=textarea.value.split(',');
      parts[7]=csvCell(direction);
      parts[8]=csvCell("");
      textarea.value=parts.join(',');
      navigator.clipboard.writeText(textarea.value);
    }
    """
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Direction calibration</title><style>{css}</style></head>
<body><div class="topbar"><h1>Direction calibration</h1><p>Label the visible screen direction of the subject gaze/head: left, right, up, down, center, unclear, or bad_track. Paste copied rows into <code>direction_labels_reviewed.csv</code>.</p><p>Samples: {len(rows)}</p></div>{''.join(cards)}<script>{script}</script></body></html>
"""
    write_text(report_path, doc)


def correctness(openface_direction: str, human_direction: str) -> str:
    if human_direction not in {"left", "right", "center"}:
        return ""
    if openface_direction not in {"left", "right", "center"}:
        return ""
    return "true" if openface_direction == human_direction else "false"


def summarize_labels(labels_path: Path, output_dir: Path) -> None:
    rows = read_csv(labels_path)
    report_rows: list[dict[str, Any]] = []
    for row in rows:
        human = (row.get("human_screen_direction") or "").strip()
        if not human:
            continue
        out = dict(row)
        out["is_gaze_sign_correct"] = correctness(row.get("openface_gaze_direction", ""), human)
        out["is_pose_sign_correct"] = correctness(row.get("openface_pose_direction", ""), human)
        report_rows.append(out)
    write_csv(output_dir / "direction_calibration_report.csv", report_rows, REPORT_COLUMNS)

    reviewed = [row for row in report_rows if row.get("human_screen_direction") not in {"", "unclear", "bad_track", "up", "down"}]
    gaze_eval = [row for row in reviewed if row.get("is_gaze_sign_correct")]
    pose_eval = [row for row in reviewed if row.get("is_pose_sign_correct")]
    gaze_correct = sum(1 for row in gaze_eval if row["is_gaze_sign_correct"] == "true")
    pose_correct = sum(1 for row in pose_eval if row["is_pose_sign_correct"] == "true")
    conflicts = sum(
        1
        for row in report_rows
        if row.get("openface_gaze_direction") in {"left", "right"}
        and row.get("openface_pose_direction") in {"left", "right"}
        and row.get("openface_gaze_direction") != row.get("openface_pose_direction")
    )
    gaze_false = sum(1 for row in gaze_eval if row["is_gaze_sign_correct"] == "false")
    pose_false = sum(1 for row in pose_eval if row["is_pose_sign_correct"] == "false")
    summary = {
        "reviewed_count": len(report_rows),
        "bad_track_count": sum(1 for row in report_rows if row.get("human_screen_direction") == "bad_track"),
        "unclear_count": sum(1 for row in report_rows if row.get("human_screen_direction") == "unclear"),
        "gaze_lr_accuracy": gaze_correct / len(gaze_eval) if gaze_eval else None,
        "pose_lr_accuracy": pose_correct / len(pose_eval) if pose_eval else None,
        "gaze_center_agreement": sum(1 for row in report_rows if row.get("human_screen_direction") == "center" and row.get("openface_gaze_direction") == "center"),
        "pose_center_agreement": sum(1 for row in report_rows if row.get("human_screen_direction") == "center" and row.get("openface_pose_direction") == "center"),
        "gaze_pose_conflict_count": conflicts,
        "likely_gaze_sign_flipped": bool(gaze_eval and gaze_false > gaze_correct),
        "likely_pose_sign_flipped": bool(pose_eval and pose_false > pose_correct),
        "human_direction_counts": dict(Counter(row.get("human_screen_direction", "") for row in report_rows)),
    }
    write_json(output_dir / "direction_calibration_summary.json", summary)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-labels-reviewed", type=Path)
    parser.add_argument("--audit-samples", type=Path)
    parser.add_argument("--final-proxy-table", type=Path)
    parser.add_argument("--openface-bins", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--export-html", action="store_true")
    parser.add_argument("--summarize-labels", type=Path)
    parser.add_argument("--include-unreviewed", action="store_true")
    parser.add_argument("--max-samples", type=int, default=50)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.export_html:
        for path in [args.audit_samples, args.final_proxy_table, args.openface_bins]:
            if path is None:
                raise SystemExit("--audit-samples, --final-proxy-table, and --openface-bins are required with --export-html")
            if not path.exists():
                raise SystemExit(f"Required input not found: {path}")
        rows = build_calibration_rows(
            args.audit_labels_reviewed,
            args.audit_samples,
            args.final_proxy_table,
            args.openface_bins,
            args.include_unreviewed,
            args.max_samples,
        )
        render_report(rows, args.output_dir)
    if args.summarize_labels:
        summarize_labels(args.summarize_labels, args.output_dir)


if __name__ == "__main__":
    main()
