from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class MovieNetPaths:
    project_root: Path
    raw_files_dir: Path
    interim_dir: Path
    processed_dir: Path
    outputs_dir: Path
    yunet_model_path: Path

    @property
    def annotation_dir(self) -> Path:
        return resolve_first_existing(
            self.raw_files_dir / "annotation_v1" / "annotation",
            self.raw_files_dir / "annotation.v1" / "annotation",
        )

    @property
    def meta_dir(self) -> Path:
        return resolve_first_existing(
            self.raw_files_dir / "meta_v1" / "meta",
            self.raw_files_dir / "meta.v1" / "meta",
        )

    @property
    def script_dir(self) -> Path:
        return resolve_first_existing(
            self.raw_files_dir / "script1K_v1" / "script",
            self.raw_files_dir / "script1K.v1" / "script",
        )

    @property
    def subtitle_dir(self) -> Path:
        return resolve_first_existing(
            self.raw_files_dir / "subtitle1K_v1" / "subtitle",
            self.raw_files_dir / "subtitle1K.v1" / "subtitle",
        )

    def shot_level_dir(self) -> Path:
        path = self.interim_dir / "shot_level"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def keyframes_dir(self, movie_id: str) -> Path:
        path = self.interim_dir / "keyframes" / movie_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def face_detection_dir(self, movie_id: str) -> Path:
        path = self.interim_dir / "face_detection" / movie_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def main_subject_dir(self, movie_id: str) -> Path:
        path = self.interim_dir / "main_subject" / movie_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def proxy_candidates_dir(self, movie_id: str) -> Path:
        path = self.interim_dir / "proxy_candidates" / movie_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def final_proxy_dir(self) -> Path:
        path = self.processed_dir / "final_proxy"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def reports_dir(self) -> Path:
        path = self.processed_dir / "reports"
        path.mkdir(parents=True, exist_ok=True)
        return path


def resolve_first_existing(*candidates: Path) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "None of the candidate paths exist:\n" + "\n".join(str(p) for p in candidates)
    )


def build_default_paths(project_root: str | Path) -> MovieNetPaths:
    root = Path(project_root).expanduser().resolve()
    raw_files_dir = root / "data" / "raw" / "MovieNet" / "files"
    interim_dir = root / "data" / "interim"
    processed_dir = root / "data" / "processed"
    outputs_dir = root / "outputs"
    yunet_model_path = Path.home() / "models_yunet" / "face_detection_yunet_2023mar.onnx"

    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    return MovieNetPaths(
        project_root=root,
        raw_files_dir=raw_files_dir,
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        outputs_dir=outputs_dir,
        yunet_model_path=yunet_model_path,
    )
