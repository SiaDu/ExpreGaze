#!/usr/bin/env python3
"""Analyze v0.2 proxy assignment failures and export debug audit samples."""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


GROUP_FIELDS = [
    "movie_id",
    "stage_type",
    "gaze_quality",
    "identity_status",
    "candidate_count",
    "has_onscreen_person_candidate",
    "has_offscreen_person_candidate",
    "current_speaker_available",
    "proxy_status",
    "failure_reason",
]

SUMMARY_FIELDS = GROUP_FIELDS + [
    "row_count",
    "mean_proxy_confidence",
    "mean_identity_confidence",
    "mean_top_score",
    "mean_score_margin",
]

AUDIT_BUCKETS = [
    ("assigned", 30),
    ("low_score", 30),
    ("low_margin", 30),
    ("unknown", 20),
    ("ambiguous", 20),
]

LABEL_COLUMNS = [
    "movie_id",
    "sample_index",
    "sample_bucket",
    "sequence_id",
    "shot_id",
    "bin_idx",
    "subject_local_track_id",
    "target_type",
    "target_id",
    "proxy_status",
    "failure_reason",
    "human_label",
    "human_target_type",
    "human_target_id",
    "human_notes",
]

HUMAN_LABELS = [
    "correct",
    "wrong_target",
    "should_be_unknown",
    "ambiguous_ok",
    "bad_identity",
    "bad_frame_or_track",
    "unsure",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path | None, rows: list[dict[str, Any]], columns: list[str]) -> None:
    fh = sys.stdout if path is None else path.open("w", encoding="utf-8", newline="")
    try:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if path is not None:
            fh.close()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def mean(values: list[float]) -> str:
    if not values:
        return "0.000000"
    return f"{sum(values) / len(values):.6f}"


def group_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field, "") for field in GROUP_FIELDS)].append(row)

    summary_rows: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        out = {field: value for field, value in zip(GROUP_FIELDS, key)}
        out.update(
            {
                "row_count": len(group),
                "mean_proxy_confidence": mean([safe_float(row.get("proxy_confidence")) for row in group]),
                "mean_identity_confidence": mean([safe_float(row.get("identity_confidence")) for row in group]),
                "mean_top_score": mean([safe_float(row.get("top_score")) for row in group]),
                "mean_score_margin": mean([safe_float(row.get("score_margin")) for row in group]),
            }
        )
        summary_rows.append(out)
    return summary_rows


def status_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = {
        "assigned": 0,
        "unknown": 0,
        "ambiguous": 0,
        "rejected": 0,
        "low_score": 0,
        "low_margin": 0,
    }
    for row in rows:
        status = row.get("proxy_status", "")
        failure = row.get("failure_reason", "")
        if not status:
            if failure in {"", "assigned"}:
                status = "assigned"
            elif failure == "low_margin":
                status = "ambiguous"
            elif failure == "low_score":
                status = "rejected"
            elif failure in {"gaze_quality_unknown", "no_candidate"}:
                status = "unknown"
        if status in counts:
            counts[status] += 1
        if failure in {"low_score", "low_margin"}:
            counts[failure] += 1
    return counts


def compare_tables(before_path: Path, after_path: Path) -> list[dict[str, Any]]:
    before = status_counts(read_csv(before_path))
    after = status_counts(read_csv(after_path))
    rows: list[dict[str, Any]] = []
    for key in ["assigned", "unknown", "ambiguous", "rejected", "low_score", "low_margin"]:
        rows.append({"metric": key, "before": before[key], "after": after[key], "delta": after[key] - before[key]})
    return rows


def infer_movie_root(final_proxy_csv: Path, movie_id: str) -> Path:
    if final_proxy_csv.parent.name == "final_proxy":
        return final_proxy_csv.parent.parent
    return Path("outputs/video_proxy") / movie_id


def key4(row: dict[str, str], track_col: str = "subject_local_track_id") -> tuple[str, str, str, str]:
    return (
        row.get("sequence_id", ""),
        row.get("shot_id", ""),
        row.get(track_col, ""),
        row.get("bin_idx", ""),
    )


def csv_cell(value: Any) -> str:
    value = "" if value is None else str(value)
    return '"' + value.replace('"', '""').replace("\n", " ") + '"'


def label_row_for_sample(sample: dict[str, str], human_label: str = "") -> dict[str, str]:
    row = {column: sample.get(column, "") for column in LABEL_COLUMNS}
    row["human_label"] = human_label
    row["human_target_type"] = ""
    row["human_target_id"] = ""
    row["human_notes"] = ""
    return row


def label_row_text(sample: dict[str, str], human_label: str) -> str:
    row = label_row_for_sample(sample, human_label)
    return ",".join(csv_cell(row[column]) for column in LABEL_COLUMNS)


def relpath_for_html(path_value: str, report_path: Path) -> str:
    if not path_value:
        return ""
    path = Path(path_value)
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        return Path(path).resolve().relative_to(report_path.parent.resolve()).as_posix()
    except ValueError:
        return Path(path).resolve().as_uri()


def fmt(value: Any) -> str:
    return html.escape(str(value or ""))


def parse_candidate_list(value: str) -> list[dict[str, Any]]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def sample_rows(rows: list[dict[str, str]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    for bucket, limit in AUDIT_BUCKETS:
        if bucket == "assigned":
            pool = [row for row in rows if row.get("proxy_status") == "assigned"]
        elif bucket == "unknown":
            pool = [row for row in rows if row.get("proxy_status") == "unknown"]
        elif bucket == "ambiguous":
            pool = [row for row in rows if row.get("proxy_status") == "ambiguous"]
        else:
            pool = [row for row in rows if row.get("failure_reason") == bucket]
        chosen = list(pool)
        rng.shuffle(chosen)
        for row in chosen[:limit]:
            out = dict(row)
            out["sample_bucket"] = bucket
            sampled.append(out)
    return sampled


def load_related_tables(movie_root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    candidates_path = movie_root / "proxy_gaze_scripts" / "08_candidate_targets.csv"
    tracks_path = movie_root / "face_tracks" / "05_face_tracks.csv"
    candidates = read_csv(candidates_path) if candidates_path.exists() else []
    tracks = read_csv(tracks_path) if tracks_path.exists() else []
    return candidates, tracks


def find_overlay(movie_root: Path, sample: dict[str, str], track_rows: list[dict[str, str]]) -> Path | None:
    overlay_dir = movie_root / "face_tracks" / "debug_overlays" / sample.get("sequence_id", "") / sample.get("shot_id", "")
    if not overlay_dir.exists():
        return None
    midpoint = (safe_float(sample.get("bin_start_sec")) + safe_float(sample.get("bin_end_sec"))) / 2.0
    nearest = min(track_rows, key=lambda row: abs(safe_float(row.get("timestamp_sec")) - midpoint), default=None)
    if nearest is not None:
        frame_idx = nearest.get("frame_idx", "")
        try:
            pattern = f"*f{int(float(frame_idx)):06d}.jpg"
            matches = sorted(overlay_dir.glob(pattern))
            if matches:
                return matches[0]
        except ValueError:
            pass
    matches = sorted(overlay_dir.glob("*.jpg"))
    return matches[0] if matches else None


def export_audit(final_proxy_csv: Path, out_root: Path, seed: int) -> dict[str, Any]:
    rows = read_csv(final_proxy_csv)
    if not rows:
        return {"final_proxy_csv": str(final_proxy_csv), "sample_count": 0}
    movie_id = rows[0].get("movie_id", "unknown")
    movie_root = infer_movie_root(final_proxy_csv, movie_id)
    out_dir = out_root if movie_id in out_root.parts else out_root / movie_id
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    candidates, tracks = load_related_tables(movie_root)
    candidates_by_key: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        candidates_by_key[key4(row)].append(row)
    tracks_by_key: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in tracks:
        tracks_by_key[(row.get("shot_id", ""), row.get("local_track_id", ""))].append(row)

    samples = sample_rows(rows, seed)
    candidate_rows: list[dict[str, Any]] = []
    subject_track_rows: list[dict[str, Any]] = []
    enriched_samples: list[dict[str, Any]] = []
    for sample_idx, sample in enumerate(samples):
        sample_key = key4(sample)
        sample_tracks = tracks_by_key.get((sample.get("shot_id", ""), sample.get("subject_local_track_id", "")), [])
        overlay = find_overlay(movie_root, sample, sample_tracks)
        overlay_status = "missing"
        overlay_path = ""
        if overlay is not None:
            filename = (
                f"{sample_idx:04d}_{sample.get('sample_bucket')}_{sample.get('movie_id')}_"
                f"{sample.get('shot_id')}_{sample.get('subject_local_track_id')}_bin{sample.get('bin_idx')}.jpg"
            )
            target = frames_dir / filename
            shutil.copy2(overlay, target)
            overlay_status = "copied"
            overlay_path = str(target)

        enriched = dict(sample)
        enriched["sample_index"] = sample_idx
        enriched["overlay_status"] = overlay_status
        enriched["overlay_path"] = overlay_path
        enriched_samples.append(enriched)

        for candidate in candidates_by_key.get(sample_key, []):
            out = dict(candidate)
            out["sample_index"] = sample_idx
            out["sample_bucket"] = sample.get("sample_bucket", "")
            candidate_rows.append(out)
        for track in sample_tracks:
            out = dict(track)
            out["sample_index"] = sample_idx
            out["sample_bucket"] = sample.get("sample_bucket", "")
            subject_track_rows.append(out)

    sample_columns = list(dict.fromkeys([key for row in enriched_samples for key in row.keys()]))
    candidate_columns = list(dict.fromkeys([key for row in candidate_rows for key in row.keys()]))
    track_columns = list(dict.fromkeys([key for row in subject_track_rows for key in row.keys()]))
    write_csv(out_dir / "audit_samples.csv", enriched_samples, sample_columns)
    write_jsonl(out_dir / "audit_samples.jsonl", enriched_samples)
    write_csv(out_dir / "audit_candidate_scores.csv", candidate_rows, candidate_columns or ["sample_index"])
    write_csv(out_dir / "audit_subject_tracks.csv", subject_track_rows, track_columns or ["sample_index"])
    summary = {
        "movie_id": movie_id,
        "sample_count": len(enriched_samples),
        "candidate_score_rows": len(candidate_rows),
        "subject_track_rows": len(subject_track_rows),
        "overlay_status_counts": {
            status: sum(1 for row in enriched_samples if row.get("overlay_status") == status)
            for status in ["copied", "missing"]
        },
        "output_dir": str(out_dir),
    }
    write_json(out_dir / "audit_summary.json", summary)
    return summary


def load_audit_tables(audit_dir: Path) -> tuple[list[dict[str, str]], dict[int, list[dict[str, str]]]]:
    samples = read_csv(audit_dir / "audit_samples.csv")
    candidate_scores_path = audit_dir / "audit_candidate_scores.csv"
    candidates = read_csv(candidate_scores_path) if candidate_scores_path.exists() else []
    candidates_by_sample: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        try:
            sample_index = int(float(row.get("sample_index", "")))
        except ValueError:
            continue
        candidates_by_sample[sample_index].append(row)
    for rows in candidates_by_sample.values():
        rows.sort(key=lambda row: safe_float(row.get("total_score")), reverse=True)
    return samples, candidates_by_sample


def build_label_template(samples: list[dict[str, str]], out_path: Path) -> None:
    rows = [label_row_for_sample(sample) for sample in samples]
    write_csv(out_path, rows, LABEL_COLUMNS)


def render_candidate_table(sample: dict[str, str], candidates: list[dict[str, str]]) -> str:
    if not candidates:
        candidates = [
            {
                "candidate_type": str(item.get("candidate_type", "")),
                "candidate_id": str(item.get("candidate_id", "")),
                "candidate_side": str(item.get("candidate_side", "")),
                "total_score": str(item.get("total_score", "")),
            }
            for item in parse_candidate_list(sample.get("candidate_list", ""))
        ]
    rows = []
    for candidate in candidates:
        rows.append(
            "<tr>"
            f"<td>{fmt(candidate.get('candidate_type'))}</td>"
            f"<td>{fmt(candidate.get('candidate_id'))}</td>"
            f"<td>{fmt(candidate.get('candidate_side'))}</td>"
            f"<td>{fmt(candidate.get('direction_score'))}</td>"
            f"<td>{fmt(candidate.get('pose_score'))}</td>"
            f"<td>{fmt(candidate.get('identity_score'))}</td>"
            f"<td>{fmt(candidate.get('dialogue_score'))}</td>"
            f"<td>{fmt(candidate.get('quality_score'))}</td>"
            f"<td>{fmt(candidate.get('uncertainty_penalty'))}</td>"
            f"<td><strong>{fmt(candidate.get('total_score'))}</strong></td>"
            f"<td>{fmt(candidate.get('score_note'))}</td>"
            "</tr>"
        )
    if not rows:
        rows.append("<tr><td colspan=\"11\">No candidate scores available.</td></tr>")
    return (
        "<table><thead><tr>"
        "<th>type</th><th>id</th><th>side</th><th>dir</th><th>pose</th>"
        "<th>id score</th><th>dialogue</th><th>quality</th><th>penalty</th><th>total</th><th>note</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def render_sample_card(
    sample: dict[str, str],
    candidates: list[dict[str, str]],
    report_path: Path,
) -> str:
    sample_index = sample.get("sample_index", "")
    image_rel = relpath_for_html(sample.get("overlay_path", ""), report_path)
    if image_rel:
        image_html = f'<img src="{fmt(image_rel)}" alt="debug frame for sample {fmt(sample_index)}">'
    else:
        image_html = '<div class="placeholder">No overlay frame</div>'
    label_controls = "".join(
        f'<button type="button" onclick="copyLabel({fmt(json.dumps(sample_index))}, {fmt(json.dumps(label))})">{fmt(label)}</button>'
        for label in HUMAN_LABELS
    )
    default_label = label_row_text(sample, "")
    metadata = [
        ("movie", sample.get("movie_id")),
        ("bucket", sample.get("sample_bucket")),
        ("sequence", sample.get("sequence_id")),
        ("shot/bin", f"{sample.get('shot_id')} / {sample.get('bin_idx')}"),
        ("subject", sample.get("subject_local_track_id")),
        ("identity", f"{sample.get('subject_global_person_id')} {sample.get('subject_cast_pid')} ({sample.get('identity_status')}, {sample.get('identity_confidence')})"),
        ("gaze", f"{sample.get('gaze_quality')} x={sample.get('gaze_direction_bucket')} pose={sample.get('pose_direction_bucket')}"),
        ("target", f"{sample.get('target_type')} / {sample.get('target_id')}"),
        ("status", f"{sample.get('proxy_status')} / {sample.get('failure_reason')} conf={sample.get('proxy_confidence')}"),
        ("scores", f"top={sample.get('top_score')} second={sample.get('second_score')} margin={sample.get('score_margin')}"),
        ("subtitle", sample.get("subtitle_text", "")),
        ("speakers", f"aligned={sample.get('aligned_speakers', '')} active={sample.get('active_speakers', '')}"),
    ]
    metadata_html = "".join(f"<dt>{fmt(k)}</dt><dd>{fmt(v)}</dd>" for k, v in metadata)
    return f"""
    <article class="sample" id="sample-{fmt(sample_index)}">
      <header>
        <h2>#{fmt(sample_index)} · {fmt(sample.get('sample_bucket'))} · {fmt(sample.get('shot_id'))} bin {fmt(sample.get('bin_idx'))}</h2>
        <div class="controls">{label_controls}</div>
      </header>
      <div class="content">
        <div class="frame">{image_html}</div>
        <div class="details">
          <dl>{metadata_html}</dl>
          <h3>Candidate scores</h3>
          {render_candidate_table(sample, candidates)}
          <h3>Label row</h3>
          <textarea id="label-{fmt(sample_index)}" readonly>{fmt(default_label)}</textarea>
        </div>
      </div>
    </article>
    """


def render_audit_report(audit_dir: Path) -> dict[str, Any]:
    samples, candidates_by_sample = load_audit_tables(audit_dir)
    report_path = audit_dir / "audit_report.html"
    build_label_template(samples, audit_dir / "audit_labels_template.csv")
    cards = []
    for sample in samples:
        try:
            sample_index = int(float(sample.get("sample_index", "")))
        except ValueError:
            sample_index = -1
        cards.append(render_sample_card(sample, candidates_by_sample.get(sample_index, []), report_path))

    css = """
    body { font-family: Arial, sans-serif; margin: 24px; background: #f6f6f4; color: #1d1d1f; }
    .topbar { position: sticky; top: 0; background: #f6f6f4; padding: 12px 0; border-bottom: 1px solid #ddd; z-index: 2; }
    .sample { background: white; border: 1px solid #ddd; border-radius: 8px; margin: 20px 0; padding: 16px; }
    .sample header { display: flex; justify-content: space-between; gap: 16px; align-items: center; }
    .sample h2 { font-size: 18px; margin: 0; }
    .content { display: grid; grid-template-columns: minmax(320px, 42%) 1fr; gap: 16px; margin-top: 12px; }
    img { max-width: 100%; border: 1px solid #ccc; border-radius: 6px; background: #222; }
    .placeholder { min-height: 220px; display: grid; place-items: center; background: #eee; border: 1px dashed #aaa; border-radius: 6px; }
    dl { display: grid; grid-template-columns: 110px 1fr; gap: 6px 12px; margin: 0; }
    dt { font-weight: 700; color: #555; }
    dd { margin: 0; overflow-wrap: anywhere; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { border-bottom: 1px solid #eee; padding: 5px; text-align: left; vertical-align: top; }
    th { background: #fafafa; position: sticky; top: 58px; }
    button { margin: 3px; padding: 6px 8px; border: 1px solid #aaa; border-radius: 6px; background: #fff; cursor: pointer; }
    button:hover { background: #eef3ff; }
    textarea { width: 100%; min-height: 64px; font-family: monospace; font-size: 12px; }
    @media (max-width: 900px) { .content { grid-template-columns: 1fr; } .sample header { display: block; } }
    """
    script = """
    const labelHeader = %s;
    function csvCell(value) {
      const s = String(value ?? "");
      return '"' + s.replaceAll('"', '""').replaceAll('\\n', ' ') + '"';
    }
    function copyLabel(sampleIndex, humanLabel) {
      const textarea = document.getElementById(`label-${sampleIndex}`);
      const parts = textarea.value.split(',');
      parts[11] = csvCell(humanLabel);
      textarea.value = parts.join(',');
      navigator.clipboard.writeText(textarea.value);
    }
    function copyHeader() {
      navigator.clipboard.writeText(labelHeader);
    }
    """ % json.dumps(",".join(LABEL_COLUMNS))
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Audit report · {fmt(audit_dir.name)}</title>
  <style>{css}</style>
</head>
<body>
  <div class="topbar">
    <h1>Audit report · {fmt(audit_dir.name)}</h1>
    <p>Review each sample, copy label rows, then paste/fill them into <code>audit_labels_template.csv</code> or save as <code>audit_labels_reviewed.csv</code>.</p>
    <p><button type="button" onclick="copyHeader()">Copy CSV header</button> Samples: {len(samples)}</p>
  </div>
  {''.join(cards)}
  <script>{script}</script>
</body>
</html>
"""
    write_text(report_path, html_doc)
    return {"movie_id": audit_dir.name, "sample_count": len(samples), "report_path": str(report_path)}


def render_audit_index(audit_root: Path, reports: list[dict[str, Any]]) -> None:
    links = "".join(
        f'<li><a href="{fmt(Path(report["report_path"]).resolve().relative_to(audit_root.resolve()).as_posix())}">'
        f'{fmt(report["movie_id"])}</a> · {fmt(report["sample_count"])} samples</li>'
        for report in reports
    )
    doc = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Proxy audit index</title>
<style>body{{font-family:Arial,sans-serif;margin:32px;line-height:1.5}}</style></head>
<body>
<h1>Proxy audit index</h1>
<ul>{links}</ul>
</body></html>
"""
    write_text(audit_root / "index.html", doc)


def summarize_labels(path: Path) -> list[dict[str, Any]]:
    rows = read_csv(path)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("human_label", "").strip():
            grouped[(row.get("movie_id", ""), row.get("sample_bucket", ""))].append(row)
    out: list[dict[str, Any]] = []
    assigned_precision_by_movie: dict[str, float] = {}
    for (movie_id, bucket), group in sorted(grouped.items()):
        reviewed = len(group)
        correct = sum(1 for row in group if row.get("human_label") == "correct")
        precision = correct / reviewed if reviewed else 0.0
        if bucket == "assigned":
            assigned_precision_by_movie[movie_id] = precision
        out.append(
            {
                "movie_id": movie_id,
                "sample_bucket": bucket,
                "reviewed_count": reviewed,
                "correct_count": correct,
                "precision": f"{precision:.6f}",
                "wrong_target_count": sum(1 for row in group if row.get("human_label") == "wrong_target"),
                "bad_identity_count": sum(1 for row in group if row.get("human_label") == "bad_identity"),
                "bad_frame_or_track_count": sum(1 for row in group if row.get("human_label") == "bad_frame_or_track"),
                "sft_ready": "true" if bucket != "assigned" or precision >= 0.80 else "false",
            }
        )
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("final_proxy_tables", nargs="*", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--compare-before", type=Path)
    parser.add_argument("--compare-after", type=Path)
    parser.add_argument("--export-audit-dir", type=Path)
    parser.add_argument("--export-html-report", action="store_true")
    parser.add_argument("--summarize-labels", type=Path)
    parser.add_argument("--seed", type=int, default=13)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.summarize_labels:
        rows = summarize_labels(args.summarize_labels)
        columns = [
            "movie_id",
            "sample_bucket",
            "reviewed_count",
            "correct_count",
            "precision",
            "wrong_target_count",
            "bad_identity_count",
            "bad_frame_or_track_count",
            "sft_ready",
        ]
        write_csv(args.output_csv, rows, columns)
        if args.output_json:
            write_json(args.output_json, rows)
        return

    if args.compare_before or args.compare_after:
        if not args.compare_before or not args.compare_after:
            raise SystemExit("--compare-before and --compare-after must be provided together")
        comparison = compare_tables(args.compare_before, args.compare_after)
        write_csv(args.output_csv, comparison, ["metric", "before", "after", "delta"])
        if args.output_json:
            write_json(args.output_json, comparison)
        return

    if args.final_proxy_tables:
        all_rows: list[dict[str, str]] = []
        for path in args.final_proxy_tables:
            all_rows.extend(read_csv(path))
        summary_rows = group_summary(all_rows)
        write_csv(args.output_csv, summary_rows, SUMMARY_FIELDS)
        if args.output_json:
            write_json(args.output_json, summary_rows)

    if args.export_audit_dir and args.final_proxy_tables:
        audit_summaries = [export_audit(path, args.export_audit_dir, args.seed) for path in args.final_proxy_tables]
        write_json(args.export_audit_dir / "audit_summary.json", audit_summaries)
        if args.export_html_report:
            reports = [render_audit_report(Path(summary["output_dir"])) for summary in audit_summaries]
            render_audit_index(args.export_audit_dir, reports)
    elif args.export_html_report:
        audit_root = args.export_audit_dir or Path("outputs/video_proxy/debug_audits/v0_2")
        if not audit_root.exists():
            raise SystemExit(f"Audit root not found: {audit_root}")
        audit_dirs = [path for path in sorted(audit_root.iterdir()) if (path / "audit_samples.csv").exists()]
        reports = [render_audit_report(path) for path in audit_dirs]
        render_audit_index(audit_root, reports)


if __name__ == "__main__":
    main()
