from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from expregaze.video_proxy.build_final_proxy_table import Stage09Config, run as run_stage09
from expregaze.video_proxy.build_gaze_events import (
    Stage08bConfig,
    build_events,
    build_shot_contexts,
    segment_events,
)
from expregaze.video_proxy.build_proxy_gaze_script import (
    Stage08Config,
    build_assignments,
    build_candidates,
    build_track_index,
)
from expregaze.video_proxy.build_track_identities import (
    TRACK_IDENTITY_COLUMNS,
    Stage07Config,
    choose_sface_match,
    choose_visual_match,
    run as run_stage07,
)
from expregaze.video_proxy.run_openface_per_track import annotate_crop_quality, build_timebins_for_track
from expregaze.video_proxy.stage_filter import filter_manifest_rows_by_stage_type, parse_stage_type_include


ANALYZER_SPEC = importlib.util.spec_from_file_location("analyze_proxy_failures", Path("scripts/analyze_proxy_failures.py"))
assert ANALYZER_SPEC is not None and ANALYZER_SPEC.loader is not None
analyze_proxy_failures = importlib.util.module_from_spec(ANALYZER_SPEC)
ANALYZER_SPEC.loader.exec_module(analyze_proxy_failures)

CALIBRATION_SPEC = importlib.util.spec_from_file_location("calibrate_gaze_direction", Path("scripts/calibrate_gaze_direction.py"))
assert CALIBRATION_SPEC is not None and CALIBRATION_SPEC.loader is not None
calibrate_gaze_direction = importlib.util.module_from_spec(CALIBRATION_SPEC)
CALIBRATION_SPEC.loader.exec_module(calibrate_gaze_direction)

BAKEOFF_SPEC = importlib.util.spec_from_file_location("run_gaze_evidence_bakeoff", Path("scripts/run_gaze_evidence_bakeoff.py"))
assert BAKEOFF_SPEC is not None and BAKEOFF_SPEC.loader is not None
run_gaze_evidence_bakeoff = importlib.util.module_from_spec(BAKEOFF_SPEC)
BAKEOFF_SPEC.loader.exec_module(run_gaze_evidence_bakeoff)

IDENTITY_REPORT_SPEC = importlib.util.spec_from_file_location("render_identity_report", Path("scripts/render_identity_report.py"))
assert IDENTITY_REPORT_SPEC is not None and IDENTITY_REPORT_SPEC.loader is not None
render_identity_report = importlib.util.module_from_spec(IDENTITY_REPORT_SPEC)
IDENTITY_REPORT_SPEC.loader.exec_module(render_identity_report)


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


class VideoProxyPipelineTests(unittest.TestCase):
    def test_stage06_crop_quality_marks_single_face_clean(self) -> None:
        manifest_rows = [
            {
                "sequence_id": "seq",
                "shot_id": "shot_0001",
                "local_track_id": "trk_000",
                "crop_x": 0,
                "crop_y": 0,
                "crop_w": 100,
                "crop_h": 100,
            }
        ]
        track_groups = [
            {
                "sequence_id": "seq",
                "shot_id": "shot_0001",
                "local_track_id": "trk_000",
                "rows": [{"frame_idx": "10", "det_id": "0"}],
            }
        ]
        detections = [
            {
                "movie_id": "tt",
                "sequence_id": "seq",
                "shot_id": "shot_0001",
                "frame_idx": "10",
                "det_id": "0",
                "bbox_cx": "50",
                "bbox_cy": "50",
            },
            {
                "movie_id": "tt",
                "sequence_id": "seq",
                "shot_id": "shot_0001",
                "frame_idx": "10",
                "det_id": "1",
                "bbox_cx": "150",
                "bbox_cy": "50",
            },
        ]

        annotate_crop_quality(manifest_rows, track_groups, detections)

        self.assertEqual(manifest_rows[0]["crop_quality"], "single_face_clean")
        self.assertEqual(manifest_rows[0]["other_face_count_max"], 0)
        self.assertEqual(manifest_rows[0]["contaminated_frame_count"], 0)
        self.assertEqual(manifest_rows[0]["checked_frame_count"], 1)

    def test_stage06_crop_quality_marks_multi_face_contaminated(self) -> None:
        manifest_rows = [
            {
                "sequence_id": "seq",
                "shot_id": "shot_0001",
                "local_track_id": "trk_000",
                "crop_x": 0,
                "crop_y": 0,
                "crop_w": 100,
                "crop_h": 100,
            }
        ]
        track_groups = [
            {
                "sequence_id": "seq",
                "shot_id": "shot_0001",
                "local_track_id": "trk_000",
                "rows": [{"frame_idx": "10", "det_id": "0"}],
            }
        ]
        detections = [
            {
                "movie_id": "tt",
                "sequence_id": "seq",
                "shot_id": "shot_0001",
                "frame_idx": "10",
                "det_id": "0",
                "bbox_cx": "50",
                "bbox_cy": "50",
            },
            {
                "movie_id": "tt",
                "sequence_id": "seq",
                "shot_id": "shot_0001",
                "frame_idx": "10",
                "det_id": "1",
                "bbox_cx": "80",
                "bbox_cy": "50",
            },
        ]

        annotate_crop_quality(manifest_rows, track_groups, detections)

        self.assertEqual(manifest_rows[0]["crop_quality"], "multi_face_contaminated")
        self.assertEqual(manifest_rows[0]["other_face_count_max"], 1)
        self.assertEqual(manifest_rows[0]["contaminated_frame_count"], 1)
        self.assertEqual(manifest_rows[0]["checked_frame_count"], 1)

    def test_stage06_timebins_downgrade_contaminated_crop_quality(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "openface.csv"
            write_csv(
                csv_path,
                [
                    {
                        "timestamp": f"{idx / 30.0:.6f}",
                        "success": "1",
                        "confidence": "0.95",
                        "gaze_angle_x": "0.1",
                        "gaze_angle_y": "0.0",
                        "pose_Rx": "0.0",
                        "pose_Ry": "0.1",
                        "pose_Rz": "0.0",
                    }
                    for idx in range(5)
                ],
                [
                    "timestamp",
                    "success",
                    "confidence",
                    "gaze_angle_x",
                    "gaze_angle_y",
                    "pose_Rx",
                    "pose_Ry",
                    "pose_Rz",
                ],
            )
            rows = build_timebins_for_track(
                {
                    "movie_id": "tt",
                    "sequence_id": "seq",
                    "shot_id": "shot_0001",
                    "shot_idx": "1",
                    "local_track_id": "trk_000",
                    "crop_start_sec": "0.000",
                    "crop_quality": "multi_face_contaminated",
                    "crop_video_path": "crop.mp4",
                },
                csv_path,
                SimpleNamespace(timebin_sec=0.5, expression_proxy=False),
            )

            self.assertEqual(rows[0]["crop_quality"], "multi_face_contaminated")
            self.assertEqual(rows[0]["gaze_quality"], "crop_multi_face_contaminated")

    def test_stage_type_filter_defaults_skip_multi_person(self) -> None:
        rows = [
            {"shot_id": "shot_0001", "stage_type": "single_speaking"},
            {"shot_id": "shot_0002", "stage_type": "two_person_dialogue_simple"},
            {"shot_id": "shot_0003", "stage_type": "multi_person"},
            {"shot_id": "shot_0004", "stage_type": "unknown"},
        ]
        selected, skipped, counts = filter_manifest_rows_by_stage_type(rows, parse_stage_type_include(None))
        self.assertEqual([row["shot_id"] for row in selected], ["shot_0001", "shot_0002"])
        self.assertEqual(skipped, 2)
        self.assertEqual(counts["multi_person"], 1)

    def test_pipeline_uses_renumbered_stage_order(self) -> None:
        script = Path("scripts/pipelines/run_video_proxy.sh").read_text(encoding="utf-8")
        self.assertIn("07_build_track_identities.sh", script)
        self.assertIn("08_build_proxy_gaze_script.sh", script)
        self.assertIn("08b_build_gaze_events.sh", script)
        self.assertNotIn("--mode pre", script)
        self.assertNotIn("--mode post", script)

    def test_pre_identity_writes_track_identity_without_assignments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "04_shot_manifest.csv"
            tracks = root / "05_face_tracks.csv"
            detections = root / "05_face_detections.csv"
            annotation = root / "annotation.json"
            meta = root / "meta.json"
            identity_dir = root / "identity"
            logs_dir = root / "logs"

            write_csv(
                manifest,
                [
                    {
                        "movie_id": "tt",
                        "sequence_id": "seq",
                        "shot_id": "shot_0001",
                        "shot_idx": "1",
                        "stage_type": "single_speaking",
                        "aligned_speakers": json.dumps(["DOROTHY"]),
                        "cast_pids": json.dumps(["nm1"]),
                    }
                ],
                ["movie_id", "sequence_id", "shot_id", "shot_idx", "stage_type", "aligned_speakers", "cast_pids"],
            )
            write_csv(
                tracks,
                [
                    {
                        "movie_id": "tt",
                        "sequence_id": "seq",
                        "shot_id": "shot_0001",
                        "shot_idx": "1",
                        "local_track_id": "trk_000",
                        "track_len": "4",
                        "track_conf": "0.9",
                    }
                ],
                ["movie_id", "sequence_id", "shot_id", "shot_idx", "local_track_id", "track_len", "track_conf"],
            )
            write_csv(
                detections,
                [{"movie_id": "tt", "shot_id": "shot_0001", "frame_width": "640", "frame_height": "360"}],
                ["movie_id", "shot_id", "frame_width", "frame_height"],
            )
            annotation.write_text(json.dumps({"cast": []}), encoding="utf-8")
            meta.write_text(
                json.dumps({"cast": [{"id": "nm1", "name": "Judy Garland", "character": "Dorothy"}]}),
                encoding="utf-8",
            )

            config = Stage07Config(
                movie_id="tt",
                annotation_json=annotation,
                meta_json=meta,
                shot_manifest_csv=manifest,
                face_tracks_csv=tracks,
                face_detections_csv=detections,
                identity_dir=identity_dir,
                logs_dir=logs_dir,
                min_body_match_score=0.55,
                single_speaker_track_confidence=0.58,
                enable_sface_gallery=False,
                sface_model_path=root / "missing_sface.onnx",
                sface_match_threshold=0.62,
                sface_match_margin=0.08,
                sface_min_track_confidence=0.60,
                sface_max_crops_per_track=5,
                stage_type_include={"single_speaking", "two_person_dialogue_simple"},
                overwrite=True,
                identity_backend="none",
            )
            run_stage07(config)
            rows = read_csv(identity_dir / "07_track_identity.csv")
            self.assertEqual(rows[0]["cast_pid"], "nm1")
            self.assertEqual(rows[0]["identity_source"], "single_speaker_single_track")
            self.assertIn("visual_backend", rows[0])
            self.assertIn("weak_fallback_source", rows[0])

    def test_sface_matcher_rejects_low_score_or_low_margin(self) -> None:
        gallery = [
            {"cast_pid": "nm1", "embedding": [1.0, 0.0], "prototype_id": "p1"},
            {"cast_pid": "nm2", "embedding": [0.99, 0.01], "prototype_id": "p2"},
        ]
        match, top, second, margin = choose_sface_match([1.0, 0.0], gallery, set(), 0.62, 0.08)
        self.assertIsNone(match)
        self.assertGreater(top, 0.9)
        self.assertLess(margin, 0.08)

        match, *_ = choose_sface_match([0.0, 1.0], gallery, {"nm1"}, 0.62, 0.08)
        self.assertIsNone(match)

    def test_visual_matcher_falls_back_from_cast_constraint_to_global(self) -> None:
        gallery = [
            {"cast_pid": "nm1", "embedding": [1.0, 0.0], "prototype_id": "p1"},
            {"cast_pid": "nm2", "embedding": [0.0, 1.0], "prototype_id": "p2"},
        ]
        match, top, second, margin, scope = choose_visual_match([1.0, 0.0], gallery, {"nm2"}, 0.90, 0.10)
        self.assertEqual(match["cast_pid"], "nm1")
        self.assertGreaterEqual(top, 0.90)
        self.assertGreaterEqual(margin, 0.10)
        self.assertEqual(scope, "global_after_constraint")

    def test_track_identity_columns_keep_core_and_visual_evidence(self) -> None:
        for column in ["global_person_id", "cast_pid", "identity_confidence", "identity_source", "identity_status"]:
            self.assertIn(column, TRACK_IDENTITY_COLUMNS)
        for column in ["visual_backend", "visual_score", "visual_margin", "prototype_id", "weak_fallback_source"]:
            self.assertIn(column, TRACK_IDENTITY_COLUMNS)

    def test_stage07_uses_identity_columns_when_present(self) -> None:
        config = Stage08Config(
            movie_id="tt",
            timebins_csv=Path("timebins.csv"),
            face_tracks_csv=Path("tracks.csv"),
            shot_manifest_csv=Path("manifest.csv"),
            candidate_sequences_jsonl=Path("seq.jsonl"),
            track_identity_csv=Path("identity.csv"),
            proxy_gaze_dir=Path("proxy"),
            logs_dir=Path("logs"),
            direction_threshold=0.2,
            pose_direction_threshold=0.25,
            min_proxy_score=0.1,
            ambiguous_margin=0.01,
            require_gaze_quality=False,
            include_offscreen_participants=False,
            include_current_speaker=False,
            high_precision=True,
            stage_type_include={"single_speaking", "two_person_dialogue_simple"},
            overwrite=True,
        )
        timebins = [
            {
                "movie_id": "tt",
                "sequence_id": "seq",
                "shot_id": "shot_0001",
                "shot_idx": "1",
                "local_track_id": "trk_000",
                "bin_idx": "0",
                "bin_start_sec": "0.0",
                "bin_end_sec": "0.5",
                "gaze_quality": "gaze_reliable",
                "gaze_angle_x_mean": "0.5",
                "pose_Ry_mean": "0.4",
            }
        ]
        tracks = [
            {"shot_id": "shot_0001", "local_track_id": "trk_000", "timestamp_sec": "0.1", "bbox_cx": "10", "bbox_cy": "10", "bbox_x1": "0", "bbox_y1": "0", "bbox_x2": "20", "bbox_y2": "20"},
            {"shot_id": "shot_0001", "local_track_id": "trk_001", "timestamp_sec": "0.1", "bbox_cx": "100", "bbox_cy": "10", "bbox_x1": "90", "bbox_y1": "0", "bbox_x2": "110", "bbox_y2": "20"},
        ]
        identity_lookup = {
            ("shot_0001", "trk_000"): {"global_person_id": "pid:nm1", "cast_pid": "nm1", "identity_confidence": "0.9"},
            ("shot_0001", "trk_001"): {"global_person_id": "pid:nm2", "cast_pid": "nm2", "identity_confidence": "0.8"},
        }
        candidates = build_candidates(timebins, build_track_index(tracks), identity_lookup, {("seq", "shot_0001"): {}}, config)
        assignments = build_assignments(timebins, candidates, identity_lookup, {("seq", "shot_0001"): {}}, config)
        self.assertEqual(assignments[0]["target_id"], "trk_001")
        self.assertEqual(assignments[0]["target_global_person_id"], "pid:nm2")
        self.assertEqual(assignments[0]["proxy_status"], "assigned")
        self.assertEqual(assignments[0]["failure_reason"], "assigned")

    def test_stage08_failure_status_mapping_and_smoothing(self) -> None:
        config = Stage08Config(
            movie_id="tt",
            timebins_csv=Path("timebins.csv"),
            face_tracks_csv=Path("tracks.csv"),
            shot_manifest_csv=Path("manifest.csv"),
            candidate_sequences_jsonl=Path("seq.jsonl"),
            track_identity_csv=Path("identity.csv"),
            proxy_gaze_dir=Path("proxy"),
            logs_dir=Path("logs"),
            direction_threshold=0.2,
            pose_direction_threshold=0.25,
            min_proxy_score=0.55,
            ambiguous_margin=0.1,
            require_gaze_quality=True,
            include_offscreen_participants=False,
            include_current_speaker=False,
            high_precision=True,
            stage_type_include={"single_speaking", "two_person_dialogue_simple"},
            overwrite=True,
        )
        timebins = [
            {
                "movie_id": "tt",
                "sequence_id": "seq",
                "shot_id": "shot_0001",
                "shot_idx": "1",
                "local_track_id": "trk_000",
                "bin_idx": str(idx),
                "bin_start_sec": str(idx * 0.5),
                "bin_end_sec": str((idx + 1) * 0.5),
                "gaze_quality": quality,
                "gaze_angle_x_mean": "0.5",
                "pose_Ry_mean": "0.4",
            }
            for idx, quality in enumerate(["gaze_reliable", "unknown", "gaze_reliable"])
        ]
        tracks = [
            {"shot_id": "shot_0001", "local_track_id": "trk_000", "timestamp_sec": "0.1", "bbox_cx": "10", "bbox_cy": "10", "bbox_x1": "0", "bbox_y1": "0", "bbox_x2": "20", "bbox_y2": "20"},
            {"shot_id": "shot_0001", "local_track_id": "trk_001", "timestamp_sec": "0.1", "bbox_cx": "100", "bbox_cy": "10", "bbox_x1": "90", "bbox_y1": "0", "bbox_x2": "110", "bbox_y2": "20"},
            {"shot_id": "shot_0001", "local_track_id": "trk_000", "timestamp_sec": "1.1", "bbox_cx": "10", "bbox_cy": "10", "bbox_x1": "0", "bbox_y1": "0", "bbox_x2": "20", "bbox_y2": "20"},
            {"shot_id": "shot_0001", "local_track_id": "trk_001", "timestamp_sec": "1.1", "bbox_cx": "100", "bbox_cy": "10", "bbox_x1": "90", "bbox_y1": "0", "bbox_x2": "110", "bbox_y2": "20"},
        ]
        identity_lookup = {
            ("shot_0001", "trk_000"): {"global_person_id": "pid:nm1", "cast_pid": "nm1", "identity_confidence": "0.9"},
            ("shot_0001", "trk_001"): {"global_person_id": "pid:nm2", "cast_pid": "nm2", "identity_confidence": "0.8"},
        }
        contexts = {("seq", "shot_0001"): {"stage_type": "two_person_dialogue_simple", "active_speakers": [], "aligned_speakers": []}}
        candidates = build_candidates(timebins, build_track_index(tracks), identity_lookup, contexts, config)
        assignments = build_assignments(timebins, candidates, identity_lookup, contexts, config)
        self.assertEqual(assignments[1]["raw_proxy_status"], "unknown")
        self.assertEqual(assignments[1]["raw_failure_reason"], "gaze_quality_unknown")
        self.assertEqual(assignments[1]["proxy_status"], "assigned")
        self.assertEqual(assignments[1]["failure_reason"], "assigned")
        self.assertEqual(assignments[1]["smoothing_applied"], "1")

        low_score = build_assignments(
            [dict(timebins[0], bin_idx="10", gaze_quality="gaze_reliable")],
            [
                {
                    "sequence_id": "seq",
                    "shot_id": "shot_0001",
                    "subject_local_track_id": "trk_000",
                    "bin_idx": "10",
                    "candidate_type": "offscreen_place_or_away",
                    "candidate_id": "offscreen_place_or_away",
                    "total_score": "0.1",
                }
            ],
            identity_lookup,
            contexts,
            config,
        )
        self.assertEqual(low_score[0]["proxy_status"], "rejected")
        self.assertEqual(low_score[0]["failure_reason"], "low_score")

        low_margin = build_assignments(
            [dict(timebins[0], bin_idx="11", gaze_quality="gaze_reliable")],
            [
                {
                    "sequence_id": "seq",
                    "shot_id": "shot_0001",
                    "subject_local_track_id": "trk_000",
                    "bin_idx": "11",
                    "candidate_type": "offscreen_participant",
                    "candidate_id": "A",
                    "candidate_identity_confidence": "0.0",
                    "total_score": "0.6",
                },
                {
                    "sequence_id": "seq",
                    "shot_id": "shot_0001",
                    "subject_local_track_id": "trk_000",
                    "bin_idx": "11",
                    "candidate_type": "offscreen_participant",
                    "candidate_id": "B",
                    "candidate_identity_confidence": "0.0",
                    "total_score": "0.58",
                },
            ],
            identity_lookup,
            contexts,
            config,
        )
        self.assertEqual(low_margin[0]["proxy_status"], "ambiguous")
        self.assertEqual(low_margin[0]["failure_reason"], "low_margin")

        contaminated = build_assignments(
            [dict(timebins[0], bin_idx="12", gaze_quality="crop_multi_face_contaminated")],
            [
                {
                    "sequence_id": "seq",
                    "shot_id": "shot_0001",
                    "subject_local_track_id": "trk_000",
                    "bin_idx": "12",
                    "candidate_type": "offscreen_place_or_away",
                    "candidate_id": "offscreen_place_or_away",
                    "total_score": "0.9",
                }
            ],
            identity_lookup,
            contexts,
            config,
        )
        self.assertEqual(contaminated[0]["proxy_status"], "unknown")
        self.assertEqual(contaminated[0]["failure_reason"], "crop_multi_face_contaminated")
        self.assertEqual(contaminated[0]["smoothing_applied"], "0")

    def test_stage08b_shot_context_and_event_segmentation(self) -> None:
        manifest = [
            {"movie_id": "tt", "sequence_id": "seq", "shot_id": "shot_0001", "shot_idx": "1", "aligned_speakers": json.dumps(["A"])},
            {"movie_id": "tt", "sequence_id": "seq", "shot_id": "shot_0002", "shot_idx": "2", "aligned_speakers": json.dumps(["B"])},
            {"movie_id": "tt", "sequence_id": "seq", "shot_id": "shot_0003", "shot_idx": "3", "aligned_speakers": json.dumps(["A"])},
        ]
        identities = [
            {"shot_id": "shot_0001", "local_track_id": "trk_000", "cast_pid": "A", "global_person_id": "pid:A", "identity_confidence": "0.9"},
            {"shot_id": "shot_0002", "local_track_id": "trk_000", "cast_pid": "A", "global_person_id": "pid:A", "identity_confidence": "0.9"},
            {"shot_id": "shot_0003", "local_track_id": "trk_000", "cast_pid": "A", "global_person_id": "pid:A", "identity_confidence": "0.9"},
            {"shot_id": "shot_0003", "local_track_id": "trk_001", "cast_pid": "B", "global_person_id": "pid:B", "identity_confidence": "0.9"},
        ]
        contexts = build_shot_contexts(manifest, identities, {"seq": {"active_speakers": ["A", "B"]}})
        by_shot = {row["shot_id"]: row for row in contexts}
        self.assertEqual(by_shot["shot_0002"]["is_single_closeup"], "1")
        self.assertEqual(by_shot["shot_0002"]["is_reaction_shot"], "1")
        self.assertEqual(by_shot["shot_0003"]["is_two_person_shot"], "1")
        self.assertEqual(by_shot["shot_0002"]["likely_interlocutor"], "B")

        assignments = [
            {
                "movie_id": "tt",
                "sequence_id": "seq",
                "shot_id": "shot_0001",
                "local_track_id": "trk_000",
                "bin_idx": str(idx),
                "bin_start_sec": str(idx * 0.5),
                "bin_end_sec": str((idx + 1) * 0.5),
                "gaze_direction_bucket": direction,
                "pose_direction_bucket": direction,
                "gaze_quality": quality,
                "proxy_status": status,
                "target_type": "offscreen_participant",
                "target_id": "B",
            }
            for idx, (direction, quality, status) in enumerate(
                [
                    ("left", "gaze_reliable", "assigned"),
                    ("left", "unknown", "unknown"),
                    ("left", "gaze_reliable", "assigned"),
                    ("right", "gaze_reliable", "assigned"),
                ]
            )
        ]
        segmented = segment_events(assignments, min_duration_sec=1.0)
        groups = segmented[("seq", "shot_0001", "trk_000")]
        self.assertEqual(len(groups), 2)
        self.assertEqual([row["bin_idx"] for row in groups[0]], ["0", "1", "2"])
        self.assertEqual([row["bin_idx"] for row in groups[1]], ["3"])

    def test_stage08b_event_assignment_and_stage09_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = Stage08bConfig(
                movie_id="tt",
                timebins_csv=root / "timebins.csv",
                shot_manifest_csv=root / "manifest.csv",
                candidate_sequences_jsonl=root / "seq.jsonl",
                track_identity_csv=root / "identity.csv",
                assignments_csv=root / "assignments.csv",
                candidate_targets_csv=root / "candidates.csv",
                gaze_event_dir=root / "events",
                logs_dir=root / "logs",
                min_event_duration_sec=1.0,
                min_event_score=0.55,
                ambiguous_margin=0.10,
                identity_confidence_threshold=0.60,
                stage_type_include={"single_speaking", "two_person_dialogue_simple"},
                overwrite=True,
            )
            assignments = [
                {
                    "movie_id": "tt",
                    "sequence_id": "seq",
                    "shot_id": "shot_0001",
                    "local_track_id": "trk_000",
                    "bin_idx": str(idx),
                    "bin_start_sec": str(idx * 0.5),
                    "bin_end_sec": str((idx + 1) * 0.5),
                    "gaze_direction_bucket": "left",
                    "pose_direction_bucket": "left",
                    "gaze_quality": "gaze_reliable",
                    "proxy_status": "assigned",
                    "target_type": "offscreen_place_or_away",
                    "target_id": "offscreen_place_or_away",
                }
                for idx in range(3)
            ]
            candidates = [
                {
                    "movie_id": "tt",
                    "sequence_id": "seq",
                    "shot_id": "shot_0001",
                    "subject_local_track_id": "trk_000",
                    "bin_idx": str(idx),
                    "candidate_type": "offscreen_participant",
                    "candidate_id": "B",
                    "candidate_global_person_id": "pid:B",
                    "candidate_cast_pid": "B",
                    "total_score": "0.62",
                }
                for idx in range(3)
            ]
            identities = [
                {"shot_id": "shot_0001", "local_track_id": "trk_000", "cast_pid": "A", "global_person_id": "pid:A", "identity_confidence": "0.9"}
            ]
            contexts = [
                {
                    "sequence_id": "seq",
                    "shot_id": "shot_0001",
                    "is_reaction_shot": "1",
                    "is_two_person_shot": "0",
                    "likely_interlocutor": "B",
                    "active_participants": json.dumps(["A", "B"]),
                }
            ]
            events, event_bins = build_events(assignments, candidates, identities, contexts, config)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["event_status"], "assigned")
            self.assertEqual(events[0]["candidate_target_id"], "B")
            self.assertEqual(len(event_bins), 3)

    def test_stage09_final_table_preserves_subject_bin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.csv"
            timebins = root / "timebins.csv"
            tracks = root / "tracks.csv"
            identities = root / "identity.csv"
            candidates = root / "candidates.csv"
            assignments = root / "assignments.csv"
            events = root / "events.csv"
            event_bins = root / "event_bins.csv"
            out_dir = root / "final"
            logs_dir = root / "logs"

            write_csv(
                manifest,
                [
                    {"movie_id": "tt", "sequence_id": "seq", "shot_id": "shot_0001", "stage_type": "two_person_dialogue_simple"},
                    {"movie_id": "tt", "sequence_id": "seq", "shot_id": "shot_0002", "stage_type": "multi_person"},
                ],
                ["movie_id", "sequence_id", "shot_id", "stage_type"],
            )
            write_csv(
                timebins,
                [
                    {
                        "movie_id": "tt",
                        "sequence_id": "seq",
                        "shot_id": "shot_0001",
                        "shot_idx": "1",
                        "local_track_id": "trk_000",
                        "bin_idx": "0",
                        "bin_start_sec": "0",
                        "bin_end_sec": "0.5",
                        "gaze_quality": "gaze_reliable",
                        "gaze_angle_x_mean": "0.3",
                        "gaze_angle_y_mean": "0.0",
                        "pose_Ry_mean": "0.2",
                        "valid_ratio": "1.0",
                        "confidence_mean": "0.9",
                    }
                ],
                [
                    "movie_id",
                    "sequence_id",
                    "shot_id",
                    "shot_idx",
                    "local_track_id",
                    "bin_idx",
                    "bin_start_sec",
                    "bin_end_sec",
                    "gaze_quality",
                    "gaze_angle_x_mean",
                    "gaze_angle_y_mean",
                    "pose_Ry_mean",
                    "valid_ratio",
                    "confidence_mean",
                ],
            )
            write_csv(
                tracks,
                [{"movie_id": "tt", "shot_id": "shot_0001", "local_track_id": "trk_000", "track_conf": "0.88"}],
                ["movie_id", "shot_id", "local_track_id", "track_conf"],
            )
            write_csv(
                identities,
                [
                    {
                        "movie_id": "tt",
                        "shot_id": "shot_0001",
                        "local_track_id": "trk_000",
                        "global_person_id": "pid:nm1",
                        "cast_pid": "nm1",
                        "identity_confidence": "0.9",
                        "track_conf": "0.88",
                    }
                ],
                ["movie_id", "shot_id", "local_track_id", "global_person_id", "cast_pid", "identity_confidence", "track_conf"],
            )
            write_csv(
                candidates,
                [
                    {
                        "movie_id": "tt",
                        "sequence_id": "seq",
                        "shot_id": "shot_0001",
                        "subject_local_track_id": "trk_000",
                        "bin_idx": "0",
                        "candidate_type": "offscreen_place_or_away",
                        "candidate_id": "offscreen_place_or_away",
                        "total_score": "0.5",
                    }
                ],
                ["movie_id", "sequence_id", "shot_id", "subject_local_track_id", "bin_idx", "candidate_type", "candidate_id", "total_score"],
            )
            write_csv(
                assignments,
                [
                    {
                        "movie_id": "tt",
                        "sequence_id": "seq",
                        "shot_id": "shot_0001",
                        "local_track_id": "trk_000",
                        "bin_idx": "0",
                        "subject_global_person_id": "pid:nm1",
                        "subject_cast_pid": "nm1",
                        "subject_identity_confidence": "0.9",
                        "target_type": "offscreen_place_or_away",
                        "target_id": "offscreen_place_or_away",
                        "target_global_person_id": "",
                        "proxy_confidence": "0.5",
                        "proxy_status": "assigned",
                        "proxy_source": "openface_rule",
                        "failure_reason": "assigned",
                        "top_score": "0.5",
                        "second_score": "0.0",
                        "score_margin": "0.5",
                    }
                ],
                [
                    "movie_id",
                    "sequence_id",
                    "shot_id",
                    "local_track_id",
                    "bin_idx",
                    "subject_global_person_id",
                    "subject_cast_pid",
                    "subject_identity_confidence",
                    "target_type",
                    "target_id",
                    "target_global_person_id",
                    "proxy_confidence",
                    "proxy_status",
                    "proxy_source",
                    "failure_reason",
                    "top_score",
                    "second_score",
                    "score_margin",
                ],
            )
            write_csv(
                events,
                [
                    {
                        "movie_id": "tt",
                        "sequence_id": "seq",
                        "shot_id": "shot_0001",
                        "event_id": "shot_0001__trk_000__ev_000",
                        "event_start": "0.000",
                        "event_end": "0.500",
                        "event_status": "assigned",
                        "event_confidence": "0.750000",
                    }
                ],
                ["movie_id", "sequence_id", "shot_id", "event_id", "event_start", "event_end", "event_status", "event_confidence"],
            )
            write_csv(
                event_bins,
                [
                    {
                        "movie_id": "tt",
                        "sequence_id": "seq",
                        "shot_id": "shot_0001",
                        "subject_local_track_id": "trk_000",
                        "bin_idx": "0",
                        "event_id": "shot_0001__trk_000__ev_000",
                        "event_status": "assigned",
                        "event_target_type": "offscreen_participant",
                        "event_target_id": "B",
                        "event_target_global_person_id": "pid:B",
                        "event_confidence": "0.750000",
                        "event_failure_reason": "assigned",
                    }
                ],
                [
                    "movie_id",
                    "sequence_id",
                    "shot_id",
                    "subject_local_track_id",
                    "bin_idx",
                    "event_id",
                    "event_status",
                    "event_target_type",
                    "event_target_id",
                    "event_target_global_person_id",
                    "event_confidence",
                    "event_failure_reason",
                ],
            )
            config = Stage09Config(
                movie_id="tt",
                shot_manifest_csv=manifest,
                timebins_csv=timebins,
                face_tracks_csv=tracks,
                track_identity_csv=identities,
                candidate_targets_csv=candidates,
                assignments_csv=assignments,
                final_proxy_dir=out_dir,
                logs_dir=logs_dir,
                stage_type_include={"single_speaking", "two_person_dialogue_simple"},
                overwrite=True,
                gaze_events_csv=events,
                gaze_event_bins_csv=event_bins,
            )
            run_stage09(config)
            rows = read_csv(out_dir / "09_final_proxy_table.csv")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["subject_global_person_id"], "pid:nm1")
            self.assertEqual(rows[0]["proxy_status"], "assigned")
            self.assertEqual(rows[0]["failure_reason"], "assigned")
            self.assertEqual(rows[0]["target_type"], "offscreen_participant")
            self.assertEqual(rows[0]["target_id"], "B")
            self.assertEqual(rows[0]["target_global_person_id"], "pid:B")
            self.assertEqual(rows[0]["proxy_confidence"], "0.750000")
            self.assertEqual(rows[0]["event_id"], "shot_0001__trk_000__ev_000")
            self.assertEqual(rows[0]["identity_status"], "unknown")
            self.assertEqual(rows[0]["has_offscreen_person_candidate"], "0")
            self.assertIn("offscreen_place_or_away", rows[0]["candidate_list"])

    def test_analyzer_groups_and_exports_audit_without_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            movie_root = root / "outputs" / "video_proxy" / "tt"
            final_dir = movie_root / "final_proxy"
            final = final_dir / "09_final_proxy_table.csv"
            candidates = movie_root / "proxy_gaze_scripts" / "08_candidate_targets.csv"
            tracks = movie_root / "face_tracks" / "05_face_tracks.csv"
            write_csv(
                final,
                [
                    {
                        "movie_id": "tt",
                        "sequence_id": "seq",
                        "shot_id": "shot_0001",
                        "stage_type": "single_speaking",
                        "subject_local_track_id": "trk_000",
                        "bin_idx": "0",
                        "bin_start_sec": "0",
                        "bin_end_sec": "0.5",
                        "gaze_quality": "gaze_reliable",
                        "identity_status": "linked_pid",
                        "candidate_count": "2",
                        "has_onscreen_person_candidate": "0",
                        "has_offscreen_person_candidate": "1",
                        "current_speaker_available": "1",
                        "proxy_status": "assigned",
                        "failure_reason": "assigned",
                        "proxy_confidence": "0.7",
                        "identity_confidence": "0.9",
                        "top_score": "0.7",
                        "score_margin": "0.2",
                    }
                ],
                [
                    "movie_id",
                    "sequence_id",
                    "shot_id",
                    "stage_type",
                    "subject_local_track_id",
                    "bin_idx",
                    "bin_start_sec",
                    "bin_end_sec",
                    "gaze_quality",
                    "identity_status",
                    "candidate_count",
                    "has_onscreen_person_candidate",
                    "has_offscreen_person_candidate",
                    "current_speaker_available",
                    "proxy_status",
                    "failure_reason",
                    "proxy_confidence",
                    "identity_confidence",
                    "top_score",
                    "score_margin",
                ],
            )
            write_csv(
                candidates,
                [
                    {
                        "movie_id": "tt",
                        "sequence_id": "seq",
                        "shot_id": "shot_0001",
                        "subject_local_track_id": "trk_000",
                        "bin_idx": "0",
                        "candidate_type": "current_speaker",
                        "candidate_id": "A",
                        "total_score": "0.7",
                    }
                ],
                ["movie_id", "sequence_id", "shot_id", "subject_local_track_id", "bin_idx", "candidate_type", "candidate_id", "total_score"],
            )
            write_csv(
                tracks,
                [{"movie_id": "tt", "shot_id": "shot_0001", "local_track_id": "trk_000", "timestamp_sec": "0.1"}],
                ["movie_id", "shot_id", "local_track_id", "timestamp_sec"],
            )
            summary = analyze_proxy_failures.group_summary(read_csv(final))
            self.assertEqual(summary[0]["row_count"], 1)
            self.assertEqual(summary[0]["mean_proxy_confidence"], "0.700000")
            audit_summary = analyze_proxy_failures.export_audit(final, movie_root / "debug_audits" / "v0_2", seed=1)
            self.assertEqual(audit_summary["sample_count"], 1)
            self.assertEqual(audit_summary["overlay_status_counts"]["missing"], 1)
            self.assertTrue((movie_root / "debug_audits" / "v0_2" / "audit_candidate_scores.csv").exists())

            report_summary = analyze_proxy_failures.render_audit_report(movie_root / "debug_audits" / "v0_2")
            report = Path(report_summary["report_path"])
            html = report.read_text(encoding="utf-8")
            self.assertIn("current_speaker", html)
            self.assertIn("correct", html)
            self.assertIn("No overlay frame", html)
            labels = read_csv(movie_root / "debug_audits" / "v0_2" / "audit_labels_template.csv")
            self.assertEqual(len(labels), 1)
            self.assertEqual(labels[0]["human_label"], "")

            reviewed = movie_root / "debug_audits" / "v0_2" / "audit_labels_reviewed.csv"
            write_csv(
                reviewed,
                [
                    {
                        **labels[0],
                        "human_label": "wrong_target",
                    }
                ],
                analyze_proxy_failures.LABEL_COLUMNS,
            )
            label_summary = analyze_proxy_failures.summarize_labels(reviewed)
            self.assertEqual(label_summary[0]["reviewed_count"], 1)
            self.assertEqual(label_summary[0]["precision"], "0.000000")
            self.assertEqual(label_summary[0]["sft_ready"], "false")

    def test_direction_calibration_html_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit_samples = root / "audit_samples.csv"
            reviewed = root / "audit_labels_reviewed.csv"
            final = root / "final.csv"
            openface = root / "openface.csv"
            out_dir = root / "calibration"
            image = root / "frame.jpg"
            image.write_bytes(b"fake")
            write_csv(
                audit_samples,
                [
                    {
                        "movie_id": "tt",
                        "sequence_id": "seq",
                        "shot_id": "shot_0001",
                        "bin_idx": "0",
                        "subject_local_track_id": "trk_000",
                        "sample_index": "0",
                        "overlay_path": str(image),
                        "gaze_direction_bucket": "left",
                        "pose_direction_bucket": "right",
                    }
                ],
                [
                    "movie_id",
                    "sequence_id",
                    "shot_id",
                    "bin_idx",
                    "subject_local_track_id",
                    "sample_index",
                    "overlay_path",
                    "gaze_direction_bucket",
                    "pose_direction_bucket",
                ],
            )
            write_csv(
                reviewed,
                [
                    {
                        "movie_id": "tt",
                        "sequence_id": "seq",
                        "shot_id": "shot_0001",
                        "bin_idx": "0",
                        "subject_local_track_id": "trk_000",
                        "sample_index": "0",
                        "human_label": "correct",
                        "human_notes": "looks left",
                    }
                ],
                [
                    "movie_id",
                    "sequence_id",
                    "shot_id",
                    "bin_idx",
                    "subject_local_track_id",
                    "sample_index",
                    "human_label",
                    "human_notes",
                ],
            )
            write_csv(
                final,
                [
                    {
                        "movie_id": "tt",
                        "shot_id": "shot_0001",
                        "bin_idx": "0",
                        "subject_local_track_id": "trk_000",
                        "gaze_quality": "gaze_reliable",
                        "target_type": "offscreen_participant",
                        "target_id": "A",
                    }
                ],
                ["movie_id", "shot_id", "bin_idx", "subject_local_track_id", "gaze_quality", "target_type", "target_id"],
            )
            write_csv(
                openface,
                [
                    {
                        "movie_id": "tt",
                        "shot_id": "shot_0001",
                        "bin_idx": "0",
                        "local_track_id": "trk_000",
                        "gaze_angle_x_mean": "-0.4",
                        "gaze_angle_y_mean": "0.1",
                        "pose_Ry_mean": "0.5",
                        "pose_Rx_mean": "-0.1",
                    }
                ],
                [
                    "movie_id",
                    "shot_id",
                    "bin_idx",
                    "local_track_id",
                    "gaze_angle_x_mean",
                    "gaze_angle_y_mean",
                    "pose_Ry_mean",
                    "pose_Rx_mean",
                ],
            )
            rows = calibrate_gaze_direction.build_calibration_rows(
                reviewed,
                audit_samples,
                final,
                openface,
                include_unreviewed=False,
                max_samples=50,
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["openface_gaze_direction"], "left")
            self.assertEqual(rows[0]["openface_pose_direction"], "right")
            calibrate_gaze_direction.render_report(rows, out_dir)
            html = (out_dir / "direction_calibration.html").read_text(encoding="utf-8")
            self.assertIn("frame.jpg", html)
            self.assertIn("-0.4", html)
            self.assertIn("bad_track", html)
            labels = read_csv(out_dir / "direction_labels_template.csv")
            labels[0]["human_screen_direction"] = "left"
            reviewed_labels = out_dir / "direction_labels_reviewed.csv"
            write_csv(reviewed_labels, labels, calibrate_gaze_direction.LABEL_COLUMNS)
            calibrate_gaze_direction.summarize_labels(reviewed_labels, out_dir)
            report = read_csv(out_dir / "direction_calibration_report.csv")
            self.assertEqual(report[0]["is_gaze_sign_correct"], "true")
            self.assertEqual(report[0]["is_pose_sign_correct"], "false")
            summary = json.loads((out_dir / "direction_calibration_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["reviewed_count"], 1)
            self.assertEqual(summary["bad_track_count"], 0)

    def test_identity_report_html_handles_missing_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "identity_report"
            manifest = [
                {
                    "movie_id": "tt",
                    "sequence_id": "seq",
                    "shot_id": "shot_0001",
                    "shot_clip_path": str(root / "missing.mp4"),
                }
            ]
            identities = [
                {
                    "movie_id": "tt",
                    "sequence_id": "seq",
                    "shot_id": "shot_0001",
                    "local_track_id": "trk_000",
                    "global_person_id": "pid:nm1",
                    "cast_pid": "nm1",
                    "cast_name": "Actor",
                    "character_name": "Character",
                    "identity_confidence": "0.92",
                    "identity_source": "insightface_gallery",
                    "identity_status": "linked_pid",
                    "visual_backend": "insightface",
                    "visual_score": "0.91",
                    "visual_margin": "0.12",
                    "prototype_id": "proto_00001",
                    "weak_fallback_source": "",
                    "evidence_note": "visual_top=0.910",
                },
                {
                    "movie_id": "tt",
                    "sequence_id": "seq",
                    "shot_id": "shot_0002",
                    "local_track_id": "trk_000",
                    "cast_pid": "nm2",
                    "cast_name": "Fallback Actor",
                    "character_name": "Fallback Character",
                    "identity_confidence": "0.00",
                    "identity_source": "",
                    "identity_status": "unknown",
                    "weak_fallback_source": "single_speaker_single_track",
                    "evidence_note": "no_identity_evidence",
                },
            ]
            gallery = [
                {
                    "movie_id": "tt",
                    "global_person_id": "pid:nm1",
                    "cast_pid": "nm1",
                    "prototype_id": "proto_00001",
                    "visual_backend": "insightface",
                    "source_shot_id": "shot_0001",
                    "source_local_track_id": "trk_000",
                    "quality_score": "0.9",
                    "crop_count": "2",
                    "note": "ok",
                }
            ]
            tracks = [
                {
                    "movie_id": "tt",
                    "shot_id": "shot_0001",
                    "local_track_id": "trk_000",
                    "frame_idx": "0",
                    "det_conf": "0.9",
                    "bbox_x1": "10",
                    "bbox_y1": "10",
                    "bbox_x2": "40",
                    "bbox_y2": "40",
                }
            ]

            summary = render_identity_report.render_report(
                identities,
                gallery,
                manifest,
                tracks,
                out_dir,
                max_samples_per_source=10,
                identity_display_aliases={
                    "nm1": {
                        "display_role": "Dorothy",
                        "movienet_character": "MovieNet Character",
                    }
                },
            )
            html = (out_dir / "identity_report.html").read_text(encoding="utf-8")
            self.assertIn("insightface_gallery", html)
            self.assertIn("Dorothy", html)
            self.assertIn("MovieNet Character", html)
            self.assertIn("Fallback Character", html)
            self.assertIn("cast pid", html)
            self.assertIn("visual score", html)
            self.assertIn("prototype", html)
            self.assertIn("weak_fallback_source", html)
            self.assertIn("missing_shot_clip", html)
            self.assertEqual(summary["sample_count"], 2)
            self.assertEqual(summary["preview_status_counts"]["missing_shot_clip"], 1)

    def test_gaze_evidence_bakeoff_sampling_and_baseline_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "outputs" / "video_proxy"
            movie = "tt0032138"
            audit_dir = base / "debug_audits" / "v0_2" / movie
            openface_dir = base / movie / "openface"
            rows = []
            for idx in range(20):
                rows.append(
                    {
                        "movie_id": movie,
                        "sequence_id": "seq",
                        "shot_id": f"shot_a{idx}",
                        "bin_idx": "0",
                        "subject_local_track_id": "trk_000",
                        "sample_index": str(idx),
                        "sample_bucket": "assigned",
                        "gaze_direction_bucket": "left",
                        "pose_direction_bucket": "right",
                        "overlay_path": "frame.jpg",
                    }
                )
            for idx in range(20, 25):
                rows.append({**rows[0], "shot_id": f"shot_l{idx}", "sample_index": str(idx), "sample_bucket": "low_score"})
            for idx in range(25, 30):
                rows.append({**rows[0], "shot_id": f"shot_m{idx}", "sample_index": str(idx), "sample_bucket": "low_margin"})
            write_csv(
                audit_dir / "audit_samples.csv",
                rows,
                [
                    "movie_id",
                    "sequence_id",
                    "shot_id",
                    "bin_idx",
                    "subject_local_track_id",
                    "sample_index",
                    "sample_bucket",
                    "gaze_direction_bucket",
                    "pose_direction_bucket",
                    "overlay_path",
                ],
            )
            write_csv(
                audit_dir / "audit_labels_reviewed.csv",
                [
                    {
                        "movie_id": movie,
                        "sequence_id": "seq",
                        "shot_id": "shot_a0",
                        "bin_idx": "0",
                        "subject_local_track_id": "trk_000",
                        "human_label": "wrong_target",
                        "human_notes": "gaze/head direction wrong",
                    }
                ],
                ["movie_id", "sequence_id", "shot_id", "bin_idx", "subject_local_track_id", "human_label", "human_notes"],
            )
            openface_rows = []
            for row in rows:
                openface_rows.append(
                    {
                        "movie_id": movie,
                        "shot_id": row["shot_id"],
                        "bin_idx": row["bin_idx"],
                        "local_track_id": row["subject_local_track_id"],
                        "gaze_angle_x_mean": "-0.4",
                        "gaze_angle_y_mean": "0.1",
                        "pose_Ry_mean": "0.5",
                        "pose_Rx_mean": "-0.2",
                        "crop_video_path": f"/tmp/{row['shot_id']}__trk_000.mp4",
                    }
                )
            write_csv(
                openface_dir / "06_gaze_timebins.csv",
                openface_rows,
                [
                    "movie_id",
                    "shot_id",
                    "bin_idx",
                    "local_track_id",
                    "gaze_angle_x_mean",
                    "gaze_angle_y_mean",
                    "pose_Ry_mean",
                    "pose_Rx_mean",
                    "crop_video_path",
                ],
            )
            selected, shortfall = run_gaze_evidence_bakeoff.select_movie_samples(audit_dir, movie)
            self.assertEqual(len(selected), 25)
            self.assertEqual(shortfall["wrong_target"], 0)
            self.assertIn("assigned_high_risk", {row["sample_subtype"] for row in selected})
            compare_rows, summary = run_gaze_evidence_bakeoff.build_compare_rows(base, root / "bakeoff", [movie])
            self.assertEqual(len(compare_rows), 25)
            self.assertTrue(compare_rows[0]["crop_video_path"])
            self.assertEqual(compare_rows[0]["openface_conflict"], "1")
            self.assertIn(compare_rows[0]["inference_status"], {"baseline_only", "model_inference_not_implemented"})
            run_gaze_evidence_bakeoff.render_html(compare_rows, summary, root / "bakeoff")
            html = (root / "bakeoff" / "gaze_evidence_compare.html").read_text(encoding="utf-8")
            self.assertIn("OpenFace gaze", html)
            self.assertIn("L2CS", html)
            self.assertIn("6DRepNet", html)


if __name__ == "__main__":
    unittest.main()
