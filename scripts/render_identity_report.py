#!/usr/bin/env python3
"""Render an HTML spot-check report for Stage07 identity links."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


def load_identity_display_aliases(run_config_path: Path) -> dict[str, dict[str, str]]:
    run_config, project_root = load_run_config(run_config_path)
    movie_config_value = run_config.get("run", {}).get("movie_config")
    movie_config_path = resolve_path(movie_config_value, project_root) if movie_config_value else None
    if movie_config_path is None or not movie_config_path.exists():
        return {}
    movie_config = load_yaml(movie_config_path)
    aliases = movie_config.get("identity_display_aliases", {})
    if not isinstance(aliases, dict):
        return {}
    normalized: dict[str, dict[str, str]] = {}
    for cast_pid, value in aliases.items():
        if not isinstance(value, dict):
            continue
        normalized[str(cast_pid)] = {str(key): str(item) for key, item in value.items() if item is not None}
    return normalized


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


def group_tracks(face_tracks: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in face_tracks:
        grouped[(row.get("shot_id", ""), row.get("local_track_id", ""))].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: safe_float(row.get("det_conf"), 0.0), reverse=True)
    return grouped


def index_manifest(manifest: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row.get("sequence_id", ""), row.get("shot_id", "")): row for row in manifest}


def track_sort_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("identity_source", "") or "none", row.get("shot_id", ""), row.get("local_track_id", ""))


def select_identity_samples(identity_rows: list[dict[str, str]], max_per_source: int) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in identity_rows:
        grouped[row.get("identity_source") or "none"].append(row)
    selected: list[dict[str, str]] = []
    priority = ["insightface_gallery", "movienet_body_bbox", "sface_gallery", "single_speaker_single_track", "none"]
    for source in priority + sorted(set(grouped) - set(priority)):
        rows = sorted(grouped.get(source, []), key=track_sort_key)
        selected.extend(rows[:max_per_source] if max_per_source > 0 else rows)
    return selected


def display_fallback(row: dict[str, str]) -> str:
    return row.get("character_name") or row.get("cast_name") or row.get("cast_pid") or "unknown"


def enrich_identity_display(
    rows: list[dict[str, str]],
    aliases: dict[str, dict[str, str]] | None,
) -> list[dict[str, str]]:
    alias_map = aliases or {}
    enriched: list[dict[str, str]] = []
    for row in rows:
        cast_pid = str(row.get("cast_pid") or "")
        alias = alias_map.get(cast_pid, {})
        payload = dict(row)
        payload["display_role"] = alias.get("display_role") or display_fallback(row)
        payload["actor_name"] = row.get("cast_name", "")
        payload["movienet_character"] = alias.get("movienet_character") or row.get("character_name", "")
        enriched.append(payload)
    return enriched


def enrich_gallery_display(
    rows: list[dict[str, str]],
    identity_rows: list[dict[str, str]],
    aliases: dict[str, dict[str, str]] | None,
) -> list[dict[str, str]]:
    identity_by_pid: dict[str, dict[str, str]] = {}
    for row in identity_rows:
        cast_pid = str(row.get("cast_pid") or "")
        if cast_pid and cast_pid not in identity_by_pid:
            identity_by_pid[cast_pid] = row

    alias_map = aliases or {}
    enriched: list[dict[str, str]] = []
    for row in rows:
        cast_pid = str(row.get("cast_pid") or "")
        identity = identity_by_pid.get(cast_pid, {})
        alias = alias_map.get(cast_pid, {})
        base = {
            "cast_pid": cast_pid,
            "cast_name": identity.get("cast_name", row.get("cast_name", "")),
            "character_name": identity.get("character_name", row.get("character_name", "")),
        }
        payload = dict(row)
        payload["display_role"] = alias.get("display_role") or display_fallback(base)
        payload["actor_name"] = base.get("cast_name", "")
        payload["movienet_character"] = alias.get("movienet_character") or base.get("character_name", "")
        enriched.append(payload)
    return enriched


def best_track_row(track_rows: list[dict[str, str]]) -> dict[str, str] | None:
    if not track_rows:
        return None
    return sorted(track_rows, key=lambda row: safe_float(row.get("det_conf"), 0.0), reverse=True)[0]


def crop_bounds(row: dict[str, str], width: int, height: int, margin: float = 0.20) -> tuple[int, int, int, int] | None:
    x1 = safe_float(row.get("bbox_x1"))
    y1 = safe_float(row.get("bbox_y1"))
    x2 = safe_float(row.get("bbox_x2"))
    y2 = safe_float(row.get("bbox_y2"))
    if any(math.isnan(value) for value in [x1, y1, x2, y2]) or x2 <= x1 or y2 <= y1:
        return None
    bw = x2 - x1
    bh = y2 - y1
    pad = max(bw, bh) * margin
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    side = max(bw, bh) + 2.0 * pad
    ix1 = max(0, int(round(cx - side / 2.0)))
    iy1 = max(0, int(round(cy - side / 2.0)))
    ix2 = min(width, int(round(cx + side / 2.0)))
    iy2 = min(height, int(round(cy + side / 2.0)))
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return ix1, iy1, ix2, iy2


def render_track_images(
    row: dict[str, str],
    manifest_by_key: dict[tuple[str, str], dict[str, str]],
    tracks_by_key: dict[tuple[str, str], list[dict[str, str]]],
    images_dir: Path,
) -> dict[str, str]:
    sequence_id = row.get("sequence_id", "")
    shot_id = row.get("shot_id", "") or row.get("source_shot_id", "")
    local_track_id = row.get("local_track_id", "") or row.get("source_local_track_id", "")
    manifest = manifest_by_key.get((sequence_id, shot_id), {})
    track_row = best_track_row(tracks_by_key.get((shot_id, local_track_id), []))
    shot_clip = Path(manifest.get("shot_clip_path", ""))
    if not sequence_id or not shot_id or not local_track_id:
        return {"status": "missing_track_key", "frame_path": "", "crop_path": ""}
    if not shot_clip.exists():
        return {"status": "missing_shot_clip", "frame_path": "", "crop_path": ""}
    if track_row is None:
        return {"status": "missing_track_rows", "frame_path": "", "crop_path": ""}

    try:
        import cv2
    except ImportError:
        return {"status": "missing_opencv", "frame_path": "", "crop_path": ""}

    cap = cv2.VideoCapture(str(shot_clip))
    if not cap.isOpened():
        return {"status": "cannot_open_shot_clip", "frame_path": "", "crop_path": ""}
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(safe_float(track_row.get("frame_idx"), 0.0)))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return {"status": "cannot_read_frame", "frame_path": "", "crop_path": ""}

    height, width = frame.shape[:2]
    bounds = crop_bounds(track_row, width, height)
    if bounds is None:
        return {"status": "missing_bbox", "frame_path": "", "crop_path": ""}

    x1, y1, x2, y2 = bounds
    images_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{shot_id}__{local_track_id}".replace("/", "_")
    frame_path = images_dir / f"{stem}__frame.jpg"
    crop_path = images_dir / f"{stem}__crop.jpg"
    preview = frame.copy()
    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(preview, local_track_id, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    crop = frame[y1:y2, x1:x2]
    cv2.imwrite(str(frame_path), preview)
    cv2.imwrite(str(crop_path), crop)
    return {"status": "rendered", "frame_path": str(frame_path), "crop_path": str(crop_path)}


def enrich_samples(
    rows: list[dict[str, str]],
    manifest: list[dict[str, str]],
    face_tracks: list[dict[str, str]],
    images_dir: Path,
) -> list[dict[str, Any]]:
    manifest_by_key = index_manifest(manifest)
    tracks_by_key = group_tracks(face_tracks)
    enriched = []
    for row in rows:
        payload: dict[str, Any] = dict(row)
        image_info = render_track_images(row, manifest_by_key, tracks_by_key, images_dir)
        payload.update(image_info)
        enriched.append(payload)
    return enriched


def count_by(rows: list[dict[str, Any]], column: str) -> dict[str, int]:
    return dict(Counter(str(row.get(column) or "none") for row in rows))


def image_pair_html(row: dict[str, Any], report_path: Path) -> str:
    frame_rel = relpath_for_html(str(row.get("frame_path", "")), report_path)
    crop_rel = relpath_for_html(str(row.get("crop_path", "")), report_path)
    if not frame_rel and not crop_rel:
        return f'<div class="placeholder">{fmt(row.get("status") or "No preview")}</div>'
    frame = f'<figure><figcaption>frame</figcaption><img src="{fmt(frame_rel)}" alt="frame preview"></figure>' if frame_rel else ""
    crop = f'<figure><figcaption>crop</figcaption><img src="{fmt(crop_rel)}" alt="face crop"></figure>' if crop_rel else ""
    return f'<div class="images">{frame}{crop}</div>'


def details_html(row: dict[str, Any], fields: list[tuple[str, str]]) -> str:
    return "<dl>" + "".join(f"<dt>{fmt(label)}</dt><dd>{fmt(row.get(key, ''))}</dd>" for label, key in fields) + "</dl>"


def render_gallery_section(gallery_rows: list[dict[str, str]]) -> str:
    cards = []
    for row in gallery_rows:
        fields = [
            ("prototype", "prototype_id"),
            ("role", "display_role"),
            ("actor", "actor_name"),
            ("MovieNet character", "movienet_character"),
            ("cast pid", "cast_pid"),
            ("source shot", "source_shot_id"),
            ("source track", "source_local_track_id"),
            ("backend", "visual_backend"),
            ("quality", "quality_score"),
            ("crops", "crop_count"),
            ("note", "note"),
        ]
        title = f'{row.get("display_role") or row.get("prototype_id") or "unknown"} / {row.get("prototype_id", "")}'
        cards.append(f'<article class="card compact"><h3>{fmt(title)}</h3>{details_html(row, fields)}</article>')
    return '<section><h2>Gallery</h2><div class="grid compact-grid">' + "".join(cards) + "</div></section>"


def render_match_section(sample_rows: list[dict[str, Any]], report_path: Path) -> str:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        grouped[row.get("identity_source") or "none"].append(row)
    sections = []
    for source in sorted(grouped):
        cards = []
        for row in grouped[source]:
            fields = [
                ("role", "display_role"),
                ("actor", "actor_name"),
                ("MovieNet character", "movienet_character"),
                ("cast pid", "cast_pid"),
                ("shot", "shot_id"),
                ("track", "local_track_id"),
                ("confidence", "identity_confidence"),
                ("visual score", "visual_score"),
                ("visual margin", "visual_margin"),
                ("prototype", "prototype_id"),
                ("weak_fallback_source", "weak_fallback_source"),
                ("note", "evidence_note"),
            ]
            title = f'{row.get("display_role") or "unknown"} / {row.get("shot_id", "")} / {row.get("local_track_id", "")}'
            cards.append(f'<article class="card"><h3>{fmt(title)}</h3>{image_pair_html(row, report_path)}{details_html(row, fields)}</article>')
        sections.append(f'<section><h2>{fmt(source)}</h2><div class="grid">{"".join(cards)}</div></section>')
    return "".join(sections)


def render_report(
    identity_rows: list[dict[str, str]],
    gallery_rows: list[dict[str, str]],
    manifest_rows: list[dict[str, str]],
    face_tracks: list[dict[str, str]],
    output_dir: Path,
    max_samples_per_source: int,
    identity_display_aliases: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "identity_report.html"
    images_dir = output_dir / "images"
    identity_rows = enrich_identity_display(identity_rows, identity_display_aliases)
    gallery_rows = enrich_gallery_display(gallery_rows, identity_rows, identity_display_aliases)
    selected = select_identity_samples(identity_rows, max_samples_per_source)
    samples = enrich_samples(selected, manifest_rows, face_tracks, images_dir)
    summary = {
        "identity_count": len(identity_rows),
        "gallery_count": len(gallery_rows),
        "sample_count": len(samples),
        "identity_source_counts": count_by(identity_rows, "identity_source"),
        "identity_status_counts": count_by(identity_rows, "identity_status"),
        "visual_backend_counts": count_by(identity_rows, "visual_backend"),
        "preview_status_counts": count_by(samples, "status"),
        "report_path": str(report_path),
    }

    css = """
    body{font-family:Inter,Arial,sans-serif;margin:0;background:#f7f7f4;color:#202124}
    .topbar{padding:24px 32px;background:#202124;color:white}
    .summary{padding:18px 32px;background:#fff;border-bottom:1px solid #ddd}
    section{padding:22px 32px}
    pre{white-space:pre-wrap;margin:0;font-size:13px}
    .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:14px}
    .compact-grid{grid-template-columns:repeat(auto-fill,minmax(260px,1fr))}
    .card{background:white;border:1px solid #ddd;border-radius:8px;padding:12px;overflow:hidden}
    .compact{font-size:13px}
    h1,h2,h3{margin:0 0 10px}
    .images{display:grid;grid-template-columns:1fr 120px;gap:8px;margin-bottom:10px;align-items:start}
    figure{margin:0}
    figcaption{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#666;margin-bottom:4px}
    img{max-width:100%;border:1px solid #ddd;background:#eee}
    dl{display:grid;grid-template-columns:110px 1fr;gap:4px 10px;margin:0}
    dt{font-weight:700;color:#555}
    dd{margin:0;word-break:break-word}
    .placeholder{display:flex;align-items:center;justify-content:center;min-height:160px;background:#eee;border:1px dashed #aaa;color:#666;margin-bottom:10px}
    """
    doc = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Stage07 Identity Report</title><style>{css}</style></head>
<body>
<div class="topbar"><h1>Stage07 Identity Report</h1><p>Spot-check visual identity gallery matches and weak fallbacks. Role is a display alias when configured; MovieNet character is the credited metadata role.</p></div>
<section class="summary"><h2>Summary</h2><pre>{fmt(json.dumps(summary, ensure_ascii=False, indent=2))}</pre></section>
{render_gallery_section(gallery_rows)}
{render_match_section(samples, report_path)}
</body></html>
"""
    write_text(report_path, doc)
    write_json(output_dir / "identity_report_summary.json", summary)
    return summary


def make_paths(args: argparse.Namespace) -> dict[str, Path]:
    run_config, project_root = load_run_config(Path(args.run_config).resolve())
    movie_id = args.movie_id or run_config.get("data", {}).get("movie_id")
    if not movie_id:
        raise ValueError("movie_id is required via --movie-id or run config data.movie_id")
    outputs = run_config.get("outputs", {})
    logs_dir = resolve_path(outputs.get("logs_dir") or f"outputs/video_proxy/{movie_id}/logs", project_root)
    face_track_dir = resolve_path(outputs.get("face_track_dir") or f"outputs/video_proxy/{movie_id}/face_tracks", project_root)
    identity_dir = resolve_path(outputs.get("track_identity_dir") or f"outputs/video_proxy/{movie_id}/track_identities", project_root)
    assert logs_dir is not None and face_track_dir is not None and identity_dir is not None
    output_dir = resolve_path(args.output_dir, project_root) if args.output_dir else identity_dir / "debug_identity_report"
    assert output_dir is not None
    return {
        "manifest": resolve_path(args.shot_manifest_csv, project_root) or logs_dir / "04_shot_manifest.csv",
        "face_tracks": resolve_path(args.face_tracks_csv, project_root) or face_track_dir / "05_face_tracks.csv",
        "identity": resolve_path(args.track_identity_csv, project_root) or identity_dir / "07_track_identity.csv",
        "gallery": resolve_path(args.identity_gallery_csv, project_root) or identity_dir / "07_identity_gallery.csv",
        "output_dir": output_dir,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--movie-id")
    parser.add_argument("--shot-manifest-csv")
    parser.add_argument("--face-tracks-csv")
    parser.add_argument("--track-identity-csv")
    parser.add_argument("--identity-gallery-csv")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-samples-per-source", type=int, default=24)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    paths = make_paths(args)
    identity_display_aliases = load_identity_display_aliases(Path(args.run_config).resolve())
    summary = render_report(
        read_csv(paths["identity"]),
        read_csv(paths["gallery"]),
        read_csv(paths["manifest"]),
        read_csv(paths["face_tracks"]),
        paths["output_dir"],
        args.max_samples_per_source,
        identity_display_aliases=identity_display_aliases,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
