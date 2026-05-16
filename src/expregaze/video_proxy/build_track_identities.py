"""Build shot-local-to-global track identities for the video proxy pipeline.

Stage07 is deliberately conservative. It uses MovieNet body annotations as the
primary weak supervision source, augments with optional SFace gallery matches,
and falls back to single-speaker/single-track evidence. It does not read or
modify proxy assignments.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
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


TRACK_IDENTITY_COLUMNS = [
    "movie_id",
    "sequence_id",
    "shot_id",
    "shot_idx",
    "local_track_id",
    "global_person_id",
    "cast_pid",
    "cast_name",
    "character_name",
    "identity_confidence",
    "identity_source",
    "identity_status",
    "evidence_note",
    "track_len",
    "track_conf",
]

@dataclass(frozen=True)
class Stage07Config:
    movie_id: str
    annotation_json: Path
    meta_json: Path
    shot_manifest_csv: Path
    face_tracks_csv: Path
    face_detections_csv: Path
    identity_dir: Path
    logs_dir: Path
    min_body_match_score: float
    single_speaker_track_confidence: float
    enable_sface_gallery: bool
    sface_model_path: Path
    sface_match_threshold: float
    sface_match_margin: float
    sface_min_track_confidence: float
    sface_max_crops_per_track: int
    stage_type_include: set[str] | None
    overwrite: bool

    @property
    def track_identity_csv(self) -> Path:
        return self.identity_dir / "07_track_identity.csv"

    @property
    def identity_gallery_csv(self) -> Path:
        return self.identity_dir / "07_identity_gallery.csv"

    @property
    def identity_gallery_pkl(self) -> Path:
        return self.identity_dir / "07_identity_gallery.pkl"

    @property
    def summary_json(self) -> Path:
        return self.logs_dir / "07_track_identity_summary.json"


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


def load_run_config(run_config_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    run_config = load_yaml(run_config_path)
    paths_config_path = Path(run_config.get("inputs", {}).get("paths_config", "configs/path_local.yaml"))
    if not paths_config_path.is_absolute():
        paths_config_path = run_config_path.parent.parent.parent / paths_config_path
    paths_config = load_yaml(paths_config_path)
    project_root = Path(paths_config.get("project", {}).get("root", run_config_path.parent.parent.parent))

    movie_config_path = Path(run_config.get("run", {}).get("movie_config", f"configs/{run_config.get('data', {}).get('movie_id')}.yaml"))
    if not movie_config_path.is_absolute():
        movie_config_path = project_root / movie_config_path
    movie_config = load_yaml(movie_config_path)
    return run_config, paths_config, movie_config, project_root


def make_config(args: argparse.Namespace) -> Stage07Config:
    run_config: dict[str, Any] = {}
    paths_config: dict[str, Any] = {}
    movie_config: dict[str, Any] = {}
    project_root = Path.cwd()

    if args.run_config:
        run_config, paths_config, movie_config, project_root = load_run_config(Path(args.run_config).resolve())

    movie_id = args.movie_id or run_config.get("data", {}).get("movie_id") or movie_config.get("movie", {}).get("movie_id")
    if not movie_id:
        raise ValueError("movie_id is required via --movie-id or run config data.movie_id")
    movie_id = str(movie_id)

    outputs = run_config.get("outputs", {})
    selection = run_config.get("selection", {})
    stage = run_config.get("stages", {}).get("build_track_identities", {})
    paths = paths_config.get("paths", {})
    files = movie_config.get("files", {})

    logs_dir = resolve_path(outputs.get("logs_dir") or f"outputs/video_proxy/{movie_id}/logs", project_root)
    face_track_dir = resolve_path(outputs.get("face_track_dir") or f"outputs/video_proxy/{movie_id}/face_tracks", project_root)
    identity_dir = resolve_path(
        args.identity_dir or outputs.get("track_identity_dir") or f"outputs/video_proxy/{movie_id}/track_identities",
        project_root,
    )
    annotation_json = resolve_path(
        args.annotation_json
        or (Path(paths.get("annotation_dir", "data/raw/MovieNet/files/annotation")) / str(files.get("annotation_file", f"{movie_id}.json"))),
        project_root,
    )
    meta_json = resolve_path(
        args.meta_json or (Path(paths.get("meta_dir", "data/raw/MovieNet/files/meta")) / str(files.get("meta_file", f"{movie_id}.json"))),
        project_root,
    )

    assert logs_dir is not None and face_track_dir is not None
    assert identity_dir is not None and annotation_json is not None and meta_json is not None

    return Stage07Config(
        movie_id=movie_id,
        annotation_json=annotation_json,
        meta_json=meta_json,
        shot_manifest_csv=resolve_path(args.shot_manifest_csv, project_root) or logs_dir / "04_shot_manifest.csv",
        face_tracks_csv=resolve_path(args.face_tracks_csv, project_root) or face_track_dir / "05_face_tracks.csv",
        face_detections_csv=resolve_path(args.face_detections_csv, project_root) or face_track_dir / "05_face_detections.csv",
        identity_dir=identity_dir,
        logs_dir=logs_dir,
        min_body_match_score=float(
            args.min_body_match_score if args.min_body_match_score is not None else stage.get("min_body_match_score", 0.55)
        ),
        single_speaker_track_confidence=float(
            args.single_speaker_track_confidence
            if args.single_speaker_track_confidence is not None
            else stage.get("single_speaker_track_confidence", 0.58)
        ),
        enable_sface_gallery=bool(stage.get("enable_sface_gallery", True)) and not args.no_sface_gallery,
        sface_model_path=resolve_path(
            args.sface_model_path or stage.get("sface_model_path") or "models/face_recognition/face_recognition_sface.onnx",
            project_root,
        )
        or project_root / "models/face_recognition/face_recognition_sface.onnx",
        sface_match_threshold=float(
            args.sface_match_threshold if args.sface_match_threshold is not None else stage.get("sface_match_threshold", 0.62)
        ),
        sface_match_margin=float(
            args.sface_match_margin if args.sface_match_margin is not None else stage.get("sface_match_margin", 0.08)
        ),
        sface_min_track_confidence=float(
            args.sface_min_track_confidence
            if args.sface_min_track_confidence is not None
            else stage.get("sface_min_track_confidence", 0.60)
        ),
        sface_max_crops_per_track=int(
            args.sface_max_crops_per_track
            if args.sface_max_crops_per_track is not None
            else stage.get("sface_max_crops_per_track", 5)
        ),
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def normalize_name(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", " ", value.upper()).strip()


def canonical_pid(value: str, name_to_pid: dict[str, str]) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.startswith("nm") or text == "others":
        return text
    return name_to_pid.get(normalize_name(text), text)


def global_person_id(cast_pid: str, shot_id: str, local_track_id: str) -> str:
    if cast_pid and cast_pid != "others":
        return f"pid:{cast_pid}"
    if cast_pid == "others":
        return f"other:{shot_id}:{local_track_id}"
    return f"local:{shot_id}:{local_track_id}"


def bbox_center(bbox: dict[str, float]) -> tuple[float, float]:
    return ((bbox["x1"] + bbox["x2"]) / 2.0, (bbox["y1"] + bbox["y2"]) / 2.0)


def scale_bbox(raw_bbox: list[Any], resolution: list[Any], frame_size: tuple[float, float]) -> dict[str, float] | None:
    if len(raw_bbox) != 4 or len(resolution) != 2:
        return None
    src_w = safe_float(resolution[0])
    src_h = safe_float(resolution[1])
    dst_w, dst_h = frame_size
    if src_w <= 0 or src_h <= 0 or dst_w <= 0 or dst_h <= 0:
        return None
    x1, y1, x2, y2 = [safe_float(v) for v in raw_bbox]
    return {
        "x1": x1 * dst_w / src_w,
        "y1": y1 * dst_h / src_h,
        "x2": x2 * dst_w / src_w,
        "y2": y2 * dst_h / src_h,
    }


def mean_track_bbox(rows: list[dict[str, str]]) -> dict[str, float]:
    keys = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "bbox_cx", "bbox_cy"]
    return {key: sum(safe_float(row.get(key)) for row in rows) / len(rows) for key in keys}


def bbox_overlap_1d(a1: float, a2: float, b1: float, b2: float) -> float:
    return max(0.0, min(a2, b2) - max(a1, b1))


def body_match_score(face_bbox: dict[str, float], body_bbox: dict[str, float]) -> tuple[float, str]:
    cx = safe_float(face_bbox.get("bbox_cx"))
    cy = safe_float(face_bbox.get("bbox_cy"))
    body_w = max(1.0, body_bbox["x2"] - body_bbox["x1"])
    body_h = max(1.0, body_bbox["y2"] - body_bbox["y1"])
    face_w = max(1.0, safe_float(face_bbox.get("bbox_x2")) - safe_float(face_bbox.get("bbox_x1")))
    inside_body = body_bbox["x1"] <= cx <= body_bbox["x2"] and body_bbox["y1"] <= cy <= body_bbox["y2"]
    upper_limit = body_bbox["y1"] + body_h * 0.62
    inside_upper = body_bbox["x1"] <= cx <= body_bbox["x2"] and body_bbox["y1"] <= cy <= upper_limit
    body_head_cx = (body_bbox["x1"] + body_bbox["x2"]) / 2.0
    body_head_cy = body_bbox["y1"] + body_h * 0.25
    dx = abs(cx - body_head_cx) / (body_w / 2.0)
    dy = abs(cy - body_head_cy) / (body_h * 0.35)
    center_score = max(0.0, 1.0 - 0.5 * dx - 0.5 * dy)
    x_overlap = bbox_overlap_1d(
        safe_float(face_bbox.get("bbox_x1")), safe_float(face_bbox.get("bbox_x2")), body_bbox["x1"], body_bbox["x2"]
    ) / face_w
    score = 0.45 * float(inside_body) + 0.20 * float(inside_upper) + 0.20 * x_overlap + 0.15 * center_score
    note = f"body_score={score:.3f};inside_body={int(inside_body)};inside_upper={int(inside_upper)};x_overlap={x_overlap:.3f}"
    return max(0.0, min(1.0, score)), note


def load_meta_cast(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, str]]:
    meta = json.loads(path.read_text(encoding="utf-8"))
    cast_rows = meta.get("cast", []) if isinstance(meta, dict) else []
    cast_by_pid: dict[str, dict[str, str]] = {}
    name_to_pid: dict[str, str] = {}
    speaker_meta_map: dict[str, str] = {}
    for row in cast_rows:
        pid = str(row.get("id", ""))
        name = str(row.get("name", ""))
        character = str(row.get("character", ""))
        if not pid:
            continue
        cast_by_pid[pid] = {"cast_name": name, "character_name": character}
        if name:
            name_to_pid[normalize_name(name)] = pid
        for alias in {character, name, character.split("/")[0], character.split("(")[0]}:
            norm = normalize_name(alias)
            if norm:
                speaker_meta_map[norm] = pid
                for token in norm.split():
                    if len(token) > 2:
                        speaker_meta_map.setdefault(token, pid)
    return cast_by_pid, name_to_pid, speaker_meta_map


def build_speaker_pid_map(
    shot_manifest: list[dict[str, str]],
    name_to_pid: dict[str, str],
    speaker_meta_map: dict[str, str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Counter[str]]]:
    evidence: dict[str, Counter[str]] = defaultdict(Counter)
    for row in shot_manifest:
        speakers = parse_json_cell(row.get("aligned_speakers"))
        pids = [canonical_pid(pid, name_to_pid) for pid in parse_json_cell(row.get("cast_pids"))]
        pids = [pid for pid in pids if pid]
        if len(speakers) == 1 and len(pids) == 1:
            evidence[normalize_name(speakers[0])][pids[0]] += 3
        else:
            for speaker in speakers:
                for pid in pids:
                    evidence[normalize_name(speaker)][pid] += 1

    mapping: dict[str, dict[str, Any]] = {}
    for speaker_norm, pid in speaker_meta_map.items():
        mapping[speaker_norm] = {
            "cast_pid": pid,
            "confidence": 0.88,
            "source": "meta_cast_alias",
            "evidence_note": "matched speaker alias to MovieNet meta cast",
        }

    for speaker_norm, counts in evidence.items():
        if not speaker_norm or not counts:
            continue
        pid, count = counts.most_common(1)[0]
        total = sum(counts.values())
        confidence = min(0.85, 0.45 + 0.08 * count + 0.20 * (count / total if total else 0.0))
        if speaker_norm not in mapping or confidence > safe_float(mapping[speaker_norm].get("confidence")):
            mapping[speaker_norm] = {
                "cast_pid": pid,
                "confidence": confidence,
                "source": "speaker_cast_cooccurrence",
                "evidence_note": f"speaker_pid_counts={dict(counts)}",
            }
    return mapping, evidence


def create_sface_recognizer(config: Stage07Config):
    if not config.enable_sface_gallery:
        return None
    if not config.sface_model_path.exists():
        raise FileNotFoundError(
            f"SFace model not found: {config.sface_model_path}. "
            "Place face_recognition_sface.onnx there, set stages.build_track_identities.sface_model_path, "
            "or pass --no-sface-gallery to run without embedding/gallery matching."
        )
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for SFace gallery matching.") from exc
    if not hasattr(cv2, "FaceRecognizerSF_create"):
        raise RuntimeError("This OpenCV build does not provide cv2.FaceRecognizerSF_create, required for SFace.")
    return cv2.FaceRecognizerSF_create(str(config.sface_model_path), "")


def vector_norm(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def normalize_embedding(values: list[float]) -> list[float]:
    norm = vector_norm(values)
    if norm <= 0:
        return []
    return [value / norm for value in values]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    return sum(x * y for x, y in zip(a, b))


def crop_face_from_shot(shot_clip_path: Path, track_row: dict[str, str], margin: float = 0.20):
    import cv2

    cap = cv2.VideoCapture(str(shot_clip_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(safe_float(track_row.get("frame_idx"), 0)))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None

    height, width = frame.shape[:2]
    x1 = safe_float(track_row.get("bbox_x1"))
    y1 = safe_float(track_row.get("bbox_y1"))
    x2 = safe_float(track_row.get("bbox_x2"))
    y2 = safe_float(track_row.get("bbox_y2"))
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
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
    crop = frame[iy1:iy2, ix1:ix2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (112, 112))


def embedding_for_track(
    recognizer: Any,
    shot_clip_path: Path,
    track_rows: list[dict[str, str]],
    config: Stage07Config,
) -> tuple[list[float], int, str]:
    if recognizer is None:
        return [], 0, "sface_disabled"
    embeddings: list[list[float]] = []
    selected_rows = sorted(track_rows, key=lambda row: safe_float(row.get("det_conf")), reverse=True)[
        : max(1, config.sface_max_crops_per_track)
    ]
    for row in selected_rows:
        crop = crop_face_from_shot(shot_clip_path, row)
        if crop is None:
            continue
        try:
            raw = recognizer.feature(crop)
        except Exception:
            continue
        values = normalize_embedding([float(value) for value in raw.flatten().tolist()])
        if values:
            embeddings.append(values)
    if not embeddings:
        return [], 0, "no_sface_embeddings"
    dim = len(embeddings[0])
    mean_values = [sum(vec[i] for vec in embeddings) / len(embeddings) for i in range(dim)]
    return normalize_embedding(mean_values), len(embeddings), "ok"


def write_gallery_outputs(config: Stage07Config, gallery_rows: list[dict[str, Any]], gallery_vectors: list[dict[str, Any]]) -> None:
    config.identity_gallery_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "movie_id",
        "global_person_id",
        "cast_pid",
        "prototype_id",
        "source_shot_id",
        "source_local_track_id",
        "quality_score",
        "embedding_dim",
        "crop_count",
        "note",
    ]
    write_csv(config.identity_gallery_csv, gallery_rows, columns)
    with config.identity_gallery_pkl.open("wb") as f:
        pickle.dump(gallery_vectors, f)


def choose_sface_match(
    vector: list[float],
    gallery_vectors: list[dict[str, Any]],
    cast_constraint: set[str],
    threshold: float,
    margin_threshold: float,
) -> tuple[dict[str, Any] | None, float, float, float]:
    scored = []
    for proto in gallery_vectors:
        if cast_constraint and proto["cast_pid"] not in cast_constraint:
            continue
        score = cosine_similarity(vector, proto["embedding"])
        scored.append((score, proto))
    if not scored:
        return None, -1.0, -1.0, 0.0
    scored.sort(key=lambda item: item[0], reverse=True)
    top_score, top_proto = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else -1.0
    margin = top_score - second_score
    if top_score >= threshold and margin >= margin_threshold:
        return top_proto, top_score, second_score, margin
    return None, top_score, second_score, margin


def build_sface_gallery_and_matches(
    config: Stage07Config,
    shot_manifest: list[dict[str, str]],
    face_tracks: list[dict[str, str]],
    name_to_pid: dict[str, str],
) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not config.enable_sface_gallery:
        write_gallery_outputs(config, [], [])
        return {}, [], []

    recognizer = create_sface_recognizer(config)
    manifest_by_key = {(row.get("sequence_id", ""), row.get("shot_id", "")): row for row in shot_manifest}
    tracks_by_key: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    tracks_by_shot: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in face_tracks:
        key = (row.get("sequence_id", ""), row.get("shot_id", ""), row.get("local_track_id", ""))
        tracks_by_key[(key[1], key[2])].append(row)
        tracks_by_shot[(key[0], key[1])].add(key[2])

    embeddings: dict[tuple[str, str], tuple[list[float], int, str]] = {}
    for (shot_id, local_track_id), rows in tracks_by_key.items():
        first = rows[0]
        manifest = manifest_by_key.get((first.get("sequence_id", ""), shot_id), {})
        shot_clip_path = Path(manifest.get("shot_clip_path", ""))
        if not shot_clip_path.exists():
            embeddings[(shot_id, local_track_id)] = ([], 0, "missing_shot_clip")
            continue
        embeddings[(shot_id, local_track_id)] = embedding_for_track(recognizer, shot_clip_path, rows, config)

    gallery_rows: list[dict[str, Any]] = []
    gallery_vectors: list[dict[str, Any]] = []
    prototype_idx = 0
    for (shot_id, local_track_id), rows in sorted(tracks_by_key.items()):
        first = rows[0]
        manifest = manifest_by_key.get((first.get("sequence_id", ""), shot_id), {})
        if manifest.get("stage_type") != "single_speaking":
            continue
        if len(tracks_by_shot.get((first.get("sequence_id", ""), shot_id), set())) != 1:
            continue
        speakers = parse_json_cell(manifest.get("aligned_speakers"))
        cast_pids = [canonical_pid(pid, name_to_pid) for pid in parse_json_cell(manifest.get("cast_pids")) if str(pid)]
        cast_pids = [pid for pid in cast_pids if pid]
        if len(speakers) != 1 or len(cast_pids) != 1:
            continue
        track_conf = safe_float(first.get("track_conf"))
        if track_conf < config.sface_min_track_confidence:
            continue
        vector, crop_count, note = embeddings.get((shot_id, local_track_id), ([], 0, "missing_embedding"))
        if not vector:
            continue
        cast_pid = cast_pids[0]
        proto_id = f"proto_{prototype_idx:05d}"
        prototype_idx += 1
        gallery_rows.append(
            {
                "movie_id": config.movie_id,
                "global_person_id": global_person_id(cast_pid, shot_id, local_track_id),
                "cast_pid": cast_pid,
                "prototype_id": proto_id,
                "source_shot_id": shot_id,
                "source_local_track_id": local_track_id,
                "quality_score": f"{track_conf:.6f}",
                "embedding_dim": len(vector),
                "crop_count": crop_count,
                "note": note,
            }
        )
        gallery_vectors.append(
            {
                "movie_id": config.movie_id,
                "global_person_id": global_person_id(cast_pid, shot_id, local_track_id),
                "cast_pid": cast_pid,
                "prototype_id": proto_id,
                "embedding": vector,
                "source_shot_id": shot_id,
                "source_local_track_id": local_track_id,
                "quality_score": track_conf,
            }
        )

    write_gallery_outputs(config, gallery_rows, gallery_vectors)

    matches: dict[tuple[str, str], dict[str, Any]] = {}
    for (shot_id, local_track_id), rows in sorted(tracks_by_key.items()):
        first = rows[0]
        manifest = manifest_by_key.get((first.get("sequence_id", ""), shot_id), {})
        cast_constraint = {
            canonical_pid(pid, name_to_pid)
            for pid in parse_json_cell(manifest.get("cast_pids"))
            if canonical_pid(pid, name_to_pid)
        }
        vector, crop_count, note = embeddings.get((shot_id, local_track_id), ([], 0, "missing_embedding"))
        if not vector or not gallery_vectors:
            continue
        top_proto, top_score, second_score, margin = choose_sface_match(
            vector, gallery_vectors, cast_constraint, config.sface_match_threshold, config.sface_match_margin
        )
        if top_proto is not None:
            matches[(shot_id, local_track_id)] = {
                "cast_pid": top_proto["cast_pid"],
                "confidence": min(0.95, max(0.0, top_score)),
                "source": "sface_gallery",
                "status": "linked_pid",
                "note": (
                    f"sface_top={top_score:.3f};second={second_score:.3f};margin={margin:.3f};"
                    f"prototype={top_proto['prototype_id']};crops={crop_count};{note}"
                ),
            }
    return matches, gallery_rows, gallery_vectors


def load_annotations_by_shot(path: Path, name_to_pid: dict[str, str]) -> dict[str, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("cast", []) if isinstance(payload, dict) else []
    by_shot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        body = row.get("body") or {}
        bbox = body.get("bbox") if isinstance(body, dict) else None
        if not bbox:
            continue
        shot_idx = str(row.get("shot_idx", ""))
        by_shot[f"shot_{int(shot_idx):04d}" if shot_idx.isdigit() else shot_idx].append(
            {
                "annotation_id": str(row.get("id", "")),
                "shot_idx": shot_idx,
                "pid": canonical_pid(str(row.get("pid", "")), name_to_pid),
                "possible_pids": [canonical_pid(str(pid), name_to_pid) for pid in row.get("possible_pids", [])],
                "resolution": row.get("resolution", []),
                "body_bbox": bbox,
            }
        )
    return dict(by_shot)


def detection_frame_sizes(detections: list[dict[str, str]]) -> dict[str, tuple[float, float]]:
    sizes: dict[str, tuple[float, float]] = {}
    for row in detections:
        shot_id = row.get("shot_id", "")
        if shot_id and shot_id not in sizes:
            sizes[shot_id] = (safe_float(row.get("frame_width")), safe_float(row.get("frame_height")))
    return sizes


def build_track_identities(
    config: Stage07Config,
    shot_manifest: list[dict[str, str]],
    face_tracks: list[dict[str, str]],
    detections: list[dict[str, str]],
    annotations_by_shot: dict[str, list[dict[str, Any]]],
    cast_by_pid: dict[str, dict[str, str]],
    speaker_map: dict[str, dict[str, Any]],
    sface_matches: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    sface_matches = sface_matches or {}
    frame_sizes = detection_frame_sizes(detections)
    manifest_by_shot = {row.get("shot_id", ""): row for row in shot_manifest}
    tracks_by_key: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    tracks_by_shot: dict[str, set[str]] = defaultdict(set)
    for row in face_tracks:
        key = (row.get("shot_id", ""), row.get("local_track_id", ""))
        tracks_by_key[key].append(row)
        tracks_by_shot[key[0]].add(key[1])

    rows: list[dict[str, Any]] = []
    match_by_track: dict[tuple[str, str], tuple[dict[str, Any], float, str]] = {}
    candidates: list[tuple[float, str, str, dict[str, Any], str]] = []
    for (shot_id, local_track_id), track_rows in tracks_by_key.items():
        frame_size = frame_sizes.get(shot_id)
        if not frame_size:
            continue
        face_bbox = mean_track_bbox(track_rows)
        for ann in annotations_by_shot.get(shot_id, []):
            body_bbox = scale_bbox(ann["body_bbox"], ann["resolution"], frame_size)
            if body_bbox is None:
                continue
            score, note = body_match_score(face_bbox, body_bbox)
            candidates.append((score, shot_id, local_track_id, ann, note))

    used_tracks: set[tuple[str, str]] = set()
    used_annotations: set[str] = set()
    for score, shot_id, local_track_id, ann, note in sorted(candidates, key=lambda item: item[0], reverse=True):
        key = (shot_id, local_track_id)
        ann_id = str(ann.get("annotation_id", ""))
        if score < config.min_body_match_score or key in used_tracks or ann_id in used_annotations:
            continue
        used_tracks.add(key)
        used_annotations.add(ann_id)
        match_by_track[key] = (ann, score, note)

    for (shot_id, local_track_id), track_rows in sorted(tracks_by_key.items()):
        first = track_rows[0]
        manifest = manifest_by_shot.get(shot_id, {})
        speakers = parse_json_cell(manifest.get("aligned_speakers"))
        track_count = len(tracks_by_shot.get(shot_id, set()))
        cast_pid = ""
        confidence = 0.0
        source = ""
        status = "unknown"
        note = "no_identity_evidence"

        matched = match_by_track.get((shot_id, local_track_id))
        if matched is not None:
            ann, confidence, note = matched
            cast_pid = str(ann.get("pid", ""))
            source = "movienet_body_bbox"
            status = "linked_pid" if cast_pid and cast_pid != "others" else "linked_other"
        elif (shot_id, local_track_id) in sface_matches:
            match = sface_matches[(shot_id, local_track_id)]
            cast_pid = str(match.get("cast_pid", ""))
            confidence = safe_float(match.get("confidence"))
            source = str(match.get("source", "sface_gallery"))
            status = str(match.get("status", "linked_pid" if cast_pid else "unknown"))
            note = str(match.get("note", "sface_gallery_match"))
        elif len(speakers) == 1 and track_count == 1:
            speaker_info = speaker_map.get(normalize_name(speakers[0]), {})
            if speaker_info:
                cast_pid = str(speaker_info.get("cast_pid", ""))
                confidence = min(config.single_speaker_track_confidence, safe_float(speaker_info.get("confidence"), 0.0))
                source = "single_speaker_single_track"
                status = "linked_pid" if cast_pid and cast_pid != "others" else "linked_other"
                note = f"single_speaker={speakers[0]};speaker_source={speaker_info.get('source', '')}"

        cast_info = cast_by_pid.get(cast_pid, {})
        rows.append(
            {
                "movie_id": config.movie_id,
                "sequence_id": first.get("sequence_id", ""),
                "shot_id": shot_id,
                "shot_idx": first.get("shot_idx", ""),
                "local_track_id": local_track_id,
                "global_person_id": global_person_id(cast_pid, shot_id, local_track_id),
                "cast_pid": cast_pid,
                "cast_name": cast_info.get("cast_name", ""),
                "character_name": cast_info.get("character_name", ""),
                "identity_confidence": f"{confidence:.6f}",
                "identity_source": source,
                "identity_status": status,
                "evidence_note": note,
                "track_len": first.get("track_len", ""),
                "track_conf": first.get("track_conf", ""),
            }
        )
    return rows


def identity_lookup(track_identity_rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row["shot_id"]), str(row["local_track_id"])): row for row in track_identity_rows}


def resolve_speaker_identity(
    speaker: str,
    speaker_map: dict[str, dict[str, Any]],
    cast_by_pid: dict[str, dict[str, str]],
) -> dict[str, str]:
    info = speaker_map.get(normalize_name(speaker), {})
    cast_pid = str(info.get("cast_pid", ""))
    cast_info = cast_by_pid.get(cast_pid, {})
    return {
        "global_person_id": global_person_id(cast_pid, "speaker", normalize_name(speaker) or "unknown"),
        "cast_pid": cast_pid,
        "cast_name": cast_info.get("cast_name", ""),
        "character": cast_info.get("character_name", speaker),
        "source": str(info.get("source", "")),
        "confidence": f"{safe_float(info.get('confidence')):.6f}" if info else "0.000000",
        "status": "linked_pid" if cast_pid and cast_pid != "others" else ("linked_other" if cast_pid == "others" else "unknown"),
    }


def ensure_inputs(config: Stage07Config) -> None:
    required = [
        config.annotation_json,
        config.meta_json,
        config.shot_manifest_csv,
        config.face_tracks_csv,
        config.face_detections_csv,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required Stage07 input not found: {path}")


def run(config: Stage07Config) -> None:
    ensure_inputs(config)
    if not config.overwrite and config.track_identity_csv.exists() and config.identity_gallery_csv.exists():
        print(f"[Stage07] track identity already exists; use --overwrite to regenerate: {config.track_identity_csv}")
        return

    cast_by_pid, name_to_pid, speaker_meta_map = load_meta_cast(config.meta_json)
    raw_shot_manifest = [row for row in read_csv(config.shot_manifest_csv) if row.get("movie_id") == config.movie_id]
    shot_manifest, skipped_stage_type_count, stage_counts = filter_manifest_rows_by_stage_type(
        raw_shot_manifest, config.stage_type_include
    )
    allowed_shots = {row.get("shot_id", "") for row in shot_manifest}
    face_tracks = read_csv(config.face_tracks_csv)
    face_tracks = [row for row in face_tracks if row.get("movie_id") == config.movie_id and row.get("shot_id", "") in allowed_shots]
    detections = read_csv(config.face_detections_csv)
    detections = [row for row in detections if row.get("movie_id") == config.movie_id and row.get("shot_id", "") in allowed_shots]
    speaker_map, speaker_evidence = build_speaker_pid_map(shot_manifest, name_to_pid, speaker_meta_map)
    annotations_by_shot = load_annotations_by_shot(config.annotation_json, name_to_pid)
    sface_matches, gallery_rows, _gallery_vectors = build_sface_gallery_and_matches(config, shot_manifest, face_tracks, name_to_pid)
    track_identity_rows = build_track_identities(
        config, shot_manifest, face_tracks, detections, annotations_by_shot, cast_by_pid, speaker_map, sface_matches
    )
    write_csv(config.track_identity_csv, track_identity_rows, TRACK_IDENTITY_COLUMNS)

    identity_status_counts = dict(Counter(row["identity_status"] for row in track_identity_rows))
    identity_source_counts = dict(Counter(row["identity_source"] or "none" for row in track_identity_rows))
    payload = {
        "movie_id": config.movie_id,
        "track_identity_count": len(track_identity_rows),
        "identity_status_counts": identity_status_counts,
        "identity_source_counts": identity_source_counts,
        "speaker_identity_count": len(speaker_map),
        "gallery_count": len(gallery_rows),
        "stage_type_include": stage_type_include_label(config.stage_type_include),
        "stage_type_counts": stage_counts,
        "skipped_stage_type_count": skipped_stage_type_count,
        "config": {
            "min_body_match_score": config.min_body_match_score,
            "single_speaker_track_confidence": config.single_speaker_track_confidence,
        },
        "outputs": {
            "track_identity_csv": str(config.track_identity_csv),
            "identity_gallery_csv": str(config.identity_gallery_csv),
            "identity_gallery_pkl": str(config.identity_gallery_pkl),
            "summary_json": str(config.summary_json),
        },
        "speaker_evidence": {speaker: dict(counts) for speaker, counts in sorted(speaker_evidence.items())},
    }
    write_json(config.summary_json, payload)

    print(f"[Stage07] movie_id={config.movie_id}")
    print(
        f"[Stage07] stage_type_include={stage_type_include_label(config.stage_type_include)} "
        f"skipped_stage_type={skipped_stage_type_count} stage_type_counts={json.dumps(stage_counts, sort_keys=True)}"
    )
    print(f"[Stage07] track_identities={len(track_identity_rows)} status_counts={json.dumps(identity_status_counts, sort_keys=True)}")
    print(f"[Stage07] gallery={len(gallery_rows)}")
    print(f"[Stage07] track_identity_csv={config.track_identity_csv}")
    print(f"[Stage07] summary_json={config.summary_json}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config")
    parser.add_argument("--movie-id")
    parser.add_argument("--annotation-json")
    parser.add_argument("--meta-json")
    parser.add_argument("--shot-manifest-csv")
    parser.add_argument("--face-tracks-csv")
    parser.add_argument("--face-detections-csv")
    parser.add_argument("--identity-dir")
    parser.add_argument("--min-body-match-score", type=float)
    parser.add_argument("--single-speaker-track-confidence", type=float)
    parser.add_argument("--stage-type-include", help="Comma list of stage_type values to process, or 'all'.")
    parser.add_argument("--sface-model-path")
    parser.add_argument("--sface-match-threshold", type=float)
    parser.add_argument("--sface-match-margin", type=float)
    parser.add_argument("--sface-min-track-confidence", type=float)
    parser.add_argument("--sface-max-crops-per-track", type=int)
    parser.add_argument("--no-sface-gallery", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        run(make_config(args))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[Stage07] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
