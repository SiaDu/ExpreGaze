from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from expregaze.video_proxy.build_final_proxy_table import Stage09Config, run as run_stage09
from expregaze.video_proxy.build_proxy_gaze_script import (
    Stage08Config,
    build_assignments,
    build_candidates,
    build_track_index,
)
from expregaze.video_proxy.build_track_identities import Stage07Config, choose_sface_match, run as run_stage07
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
            )
            run_stage07(config)
            rows = read_csv(identity_dir / "07_track_identity.csv")
            self.assertEqual(rows[0]["cast_pid"], "nm1")
            self.assertEqual(rows[0]["identity_source"], "single_speaker_single_track")

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

    def test_stage09_final_table_preserves_subject_bin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.csv"
            timebins = root / "timebins.csv"
            tracks = root / "tracks.csv"
            identities = root / "identity.csv"
            candidates = root / "candidates.csv"
            assignments = root / "assignments.csv"
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
            )
            run_stage09(config)
            rows = read_csv(out_dir / "09_final_proxy_table.csv")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["subject_global_person_id"], "pid:nm1")
            self.assertEqual(rows[0]["proxy_status"], "assigned")
            self.assertEqual(rows[0]["failure_reason"], "assigned")
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
