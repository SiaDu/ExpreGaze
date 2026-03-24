from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from expregaze.config import build_default_paths
from expregaze.movienet.shot_builder import process_one_movie_to_shot_level
from expregaze.proxy.labeling import add_proxy_label_columns
from expregaze.proxy.stability import add_stability_columns, filter_final_proxy_candidates
from expregaze.proxy.token_builder import add_proxy_token_column
from expregaze.vision.face_detector import build_face_count_df, create_yunet_detector, strict_filter_face_df
from expregaze.vision.head_direction import add_framewise_head_direction, add_head_dir_consensus
from expregaze.vision.keyframes import extract_keyframes_for_movie
from expregaze.vision.main_subject import build_main_subject_df


def find_video_for_movie(raw_files_dir: Path, movie_id: str, explicit_video: str | None = None) -> Path:
    if explicit_video:
        path = Path(explicit_video).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Explicit video path not found: {path}")
        return path

    candidates = list(raw_files_dir.glob(f"{movie_id}.*"))
    if candidates:
        return candidates[0]

    video_candidates = []
    for ext in ("*.mkv", "*.mp4", "*.avi", "*.mov"):
        video_candidates.extend(raw_files_dir.glob(ext))

    if len(video_candidates) == 1:
        return video_candidates[0]

    raise FileNotFoundError(
        "Cannot uniquely find the movie video. Pass --video explicitly."
    )


def run_movie_pipeline(project_root: str, movie_id: str, video_path: str | None = None) -> dict[str, Path]:
    paths = build_default_paths(project_root)

    shot_df, shot_csv, shot_jsonl = process_one_movie_to_shot_level(
        movie_id=movie_id,
        annotation_dir=paths.annotation_dir,
        output_dir=paths.shot_level_dir(),
    )

    video = find_video_for_movie(paths.raw_files_dir, movie_id, explicit_video=video_path)
    keyframe_root = paths.keyframes_dir(movie_id)
    extract_keyframes_for_movie(video, shot_df, keyframe_root)

    detector = create_yunet_detector(paths.yunet_model_path)
    face_df = build_face_count_df(keyframe_root, detector)
    face_df = strict_filter_face_df(face_df)
    face_csv = paths.face_detection_dir(movie_id) / f"{movie_id}_strict_face_candidates.csv"
    face_df.to_csv(face_csv, index=False, encoding="utf-8-sig")

    shot_df = shot_df.copy()
    if "shot_id" not in shot_df.columns:
        shot_df["shot_id"] = shot_df["shot_idx"].apply(lambda x: f"shot_{int(x):04d}")

    strict_candidate_df = shot_df.merge(face_df, on="shot_id", how="inner").copy()
    strict_candidate_df = strict_candidate_df.drop_duplicates(subset=["shot_id"]).reset_index(drop=True)

    main_subject_df = build_main_subject_df(strict_candidate_df, keyframe_root, detector)
    main_subject_df = add_stability_columns(main_subject_df)
    main_subject_df = add_framewise_head_direction(main_subject_df)
    main_subject_df = add_head_dir_consensus(main_subject_df)

    main_subject_csv = paths.main_subject_dir(movie_id) / f"{movie_id}_main_subject_candidates.csv"
    main_subject_df.to_csv(main_subject_csv, index=False, encoding="utf-8-sig")

    final_proxy_df = filter_final_proxy_candidates(main_subject_df)

    shot_meta_df = shot_df[["shot_id", "shot_idx", "shot_start_time", "shot_end_time", "subtitle_text"]].drop_duplicates(subset=["shot_id"]).copy()
    final_proxy_df = final_proxy_df.merge(shot_meta_df, on=["shot_id", "shot_idx"], how="left")
    final_proxy_df["shot_duration"] = final_proxy_df["shot_end_time"] - final_proxy_df["shot_start_time"]
    final_proxy_df = add_proxy_label_columns(final_proxy_df)
    final_proxy_df = add_proxy_token_column(final_proxy_df)

    final_proxy_csv = paths.final_proxy_dir() / f"{movie_id}_final_proxy_candidates.csv"
    final_proxy_df.to_csv(final_proxy_csv, index=False, encoding="utf-8-sig")

    report = {
        "movie_id": movie_id,
        "shot_level_rows": int(len(shot_df)),
        "strict_face_candidates": int(len(face_df)),
        "main_subject_rows": int(len(main_subject_df)),
        "final_proxy_rows": int(len(final_proxy_df)),
    }
    report_path = paths.reports_dir() / f"{movie_id}_summary.json"
    pd.Series(report).to_json(report_path, force_ascii=False, indent=2)

    return {
        "shot_csv": shot_csv,
        "shot_jsonl": shot_jsonl,
        "face_csv": face_csv,
        "main_subject_csv": main_subject_csv,
        "final_proxy_csv": final_proxy_csv,
        "report_json": report_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--movie-id", required=True)
    parser.add_argument("--video", default=None)
    args = parser.parse_args()

    outputs = run_movie_pipeline(
        project_root=args.project_root,
        movie_id=args.movie_id,
        video_path=args.video,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
