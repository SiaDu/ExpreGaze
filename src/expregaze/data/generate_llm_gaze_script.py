#!/usr/bin/env python3
"""
Generate per-character LLM gaze scripts from Stage02 candidate sequences.

Stage03 reads the selected candidate sequence JSONL produced by Stage02. It
does not read full_context directly and does not run video processing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import yaml

try:
    from openai import OpenAI, RateLimitError
except Exception:  # pragma: no cover
    OpenAI = None
    RateLimitError = None


MODEL_NAME = "gpt-5-mini"
MAX_OUTPUT_TOKENS = 1000
MAX_RETRIES = 5
DEFAULT_RETRY_WAIT = 3.0
AUTO = "auto"

SYSTEM_PROMPT = """
You are a cinematic actor and gaze-script annotator.

Task:
Given one film sequence package, generate gaze events ONLY for the character
specified as main_char. Infer how main_char is likely performing gaze across
the sequence using only the provided package.

Rules:
1. Generate events only for main_char.
2. A single shot may contain multiple gaze events.
3. Each gaze event must have exactly one target.
4. Split into multiple events only when there is evidence of a meaningful
   target or intent change.
5. Strong split signals include direct address changes, insults/references
   pointing to different characters, and script cues like looks, turns, pans,
   or camera direction.
6. Within the same shot, duration_ratio values should sum to about 1.0.
7. Stay conservative. Do not invent rapid gaze shifts unsupported by text.
8. Return valid JSON only.
""".strip()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def resolve_path(path_value: str | Path | None, base_dir: Path) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else base_dir / path


def safe_slug(value: Any) -> str:
    text = str(value).strip().upper()
    text = re.sub(r"[^A-Z0-9]+", "_", text).strip("_")
    return text or "UNKNOWN"


def clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def normalize_target_name(target: Any) -> str:
    return f"TGT_{safe_slug(target)}"


def parse_sequence_id_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        return [part.strip() for part in text.split(",") if part.strip()]
    return [str(value)]


def parse_auto_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip().lower()
        if text == AUTO:
            return None
        if not text:
            return default
        value = text
    parsed = int(value)
    if parsed < 0:
        raise ValueError("count limits must be non-negative or 'auto'.")
    return parsed


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def get_retry_wait_seconds(exc: Exception, default_wait: float = DEFAULT_RETRY_WAIT) -> float:
    match = re.search(r"Please try again in\s+([0-9.]+)s", str(exc))
    if match:
        try:
            return max(float(match.group(1)), 0.5)
        except Exception:
            pass
    return default_wait


def make_openai_client(api_key_env: str) -> Any:
    if OpenAI is None:
        raise RuntimeError("openai package is not installed.")
    api_key = os.environ.get(api_key_env)
    if api_key is None:
        raise RuntimeError(f"{api_key_env} is not set.")
    return OpenAI(api_key=api_key)


def call_llm_with_retry(client: Any, prompt: str, model_name: str, max_output_tokens: int) -> str:
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.responses.create(
                model=model_name,
                instructions=SYSTEM_PROMPT,
                input=prompt,
                max_output_tokens=max_output_tokens,
            )
            output_text = getattr(response, "output_text", "") or ""
            if output_text.strip():
                return output_text
            response_dict = response.model_dump() if hasattr(response, "model_dump") else {}
            texts: list[str] = []
            for item in response_dict.get("output", []) or []:
                for content in item.get("content", []) or []:
                    text = content.get("text")
                    if text:
                        texts.append(str(text))
            if texts:
                return "\n".join(texts)
            return json.dumps(response_dict, ensure_ascii=False)
        except Exception as exc:
            last_exc = exc
            is_rate_limit = RateLimitError is not None and isinstance(exc, RateLimitError)
            if not is_rate_limit or attempt == MAX_RETRIES - 1:
                raise
            time.sleep(get_retry_wait_seconds(exc))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("LLM request failed without exception.")


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find JSON object in model output: {text[:500]}")
    return json.loads(match.group(0))


def unique_texts(values: list[Any], max_items: int = 5) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = clean_text(value)
        if text and text not in seen:
            out.append(text)
            seen.add(text)
        if len(out) >= max_items:
            break
    return out


def build_pre_script(sequence: dict[str, Any], max_items: int = 5) -> list[str]:
    if isinstance(sequence.get("pre_script"), list):
        return unique_texts(sequence["pre_script"], max_items=max_items)
    shots = sequence.get("shots", []) or []
    values: list[Any] = [sequence.get("script_action_preview", "")]
    values.extend(shot.get("prev_other_text", "") for shot in shots[:3])
    return unique_texts(values, max_items=max_items)


def slim_shot(shot: dict[str, Any]) -> dict[str, Any]:
    return {
        "shot_id": shot.get("shot_id", ""),
        "shot_idx": shot.get("shot_idx"),
        "start_time_hms": shot.get("shot_start_time_hms", ""),
        "end_time_hms": shot.get("shot_end_time_hms", ""),
        "subtitle_text": shot.get("subtitle_text", ""),
        "aligned_script_dialogue": shot.get("aligned_script_dialogue", ""),
        "aligned_speakers": shot.get("aligned_speakers", []),
        "script_before": shot.get("prev_other_text", ""),
        "script_after": shot.get("next_other_text", ""),
    }


def build_sequence_package(sequence: dict[str, Any], main_char: str) -> dict[str, Any]:
    return {
        "sequence_id": sequence.get("sequence_id"),
        "movie_id": sequence.get("movie_id"),
        "main_char": main_char,
        "pre_script": build_pre_script(sequence),
        "script_action_preview": sequence.get("script_action_preview", ""),
        "active_speakers": sequence.get("active_speakers", []),
        "shots": [slim_shot(shot) for shot in sequence.get("shots", [])],
    }


def build_prompt(
    sequence_package: dict[str, Any],
    allowed_intents: list[str],
    allowed_targets: list[str],
) -> str:
    return "\n".join(
        [
            "Return exactly one JSON object with this shape:",
            json.dumps(
                {
                    "sequence_id": sequence_package.get("sequence_id"),
                    "movie_id": sequence_package.get("movie_id"),
                    "main_char": sequence_package.get("main_char"),
                    "events": [
                        {
                            "shot_id": "shot_0001",
                            "order_in_shot": 1,
                            "phase": "early|mid|late",
                            "duration_ratio": 1.0,
                            "target": "character/object/offscreen target",
                            "gaze_intent": allowed_intents[0] if allowed_intents else "ADDRESS",
                            "evidence": "short quote or cue from the package",
                        }
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            "",
            "Allowed gaze_intent labels:",
            ", ".join(allowed_intents),
            "",
            "Allowed broad target classes:",
            ", ".join(allowed_targets),
            "",
            "Use specific character names as targets when supported by evidence.",
            "Use OFFSCREEN, DOWN, UP, LEFT, RIGHT, GROUP, or OBJECT only when a specific character is not supported.",
            "",
            "Sequence package:",
            json.dumps(sequence_package, ensure_ascii=False, indent=2),
        ]
    )


def build_shot_time_index(sequence: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for shot in sequence.get("shots", []) or []:
        shot_id = str(shot.get("shot_id", ""))
        if not shot_id:
            continue
        start = float(shot.get("shot_start_time", 0.0))
        end = float(shot.get("shot_end_time", start))
        out[shot_id] = {
            "start_time": start,
            "end_time": end,
            "duration": max(end - start, 0.0),
            "start_time_hms": shot.get("shot_start_time_hms", ""),
            "end_time_hms": shot.get("shot_end_time_hms", ""),
        }
    return out


def normalize_output(parsed: dict[str, Any], sequence: dict[str, Any], main_char: str) -> dict[str, Any]:
    events = parsed.get("events", [])
    if not isinstance(events, list):
        events = []
    return {
        "sequence_id": parsed.get("sequence_id") or sequence.get("sequence_id"),
        "movie_id": parsed.get("movie_id") or sequence.get("movie_id"),
        "main_char": parsed.get("main_char") or main_char,
        "events": events,
    }


def validate_output(
    output: dict[str, Any],
    sequence: dict[str, Any],
    allowed_intents: list[str],
) -> list[str]:
    warnings: list[str] = []
    valid_shot_ids = {str(shot.get("shot_id")) for shot in sequence.get("shots", []) if shot.get("shot_id")}
    allowed_intent_set = set(allowed_intents)
    for idx, event in enumerate(output.get("events", []) or []):
        shot_id = str(event.get("shot_id", ""))
        if shot_id not in valid_shot_ids:
            warnings.append(f"event[{idx}] invalid shot_id: {shot_id}")
        try:
            ratio = float(event.get("duration_ratio", 0.0))
            if ratio <= 0:
                warnings.append(f"event[{idx}] non-positive duration_ratio: {ratio}")
        except Exception:
            warnings.append(f"event[{idx}] invalid duration_ratio: {event.get('duration_ratio')}")
        intent = str(event.get("gaze_intent", "")).strip()
        if allowed_intent_set and intent not in allowed_intent_set:
            warnings.append(f"event[{idx}] unknown gaze_intent: {intent}")
    if not output.get("events"):
        warnings.append("zero events")
    return warnings


def convert_output_to_timeline(output: dict[str, Any], sequence: dict[str, Any], round_digits: int = 2) -> dict[str, Any]:
    shot_time = build_shot_time_index(sequence)
    if not shot_time:
        return {
            "sequence_id": output["sequence_id"],
            "movie_id": output["movie_id"],
            "main_char": output["main_char"],
            "events": [],
        }
    seq_start = min(info["start_time"] for info in shot_time.values())
    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in output.get("events", []) or []:
        shot_id = str(event.get("shot_id", ""))
        if shot_id in shot_time:
            grouped.setdefault(shot_id, []).append(event)

    timeline_events: list[dict[str, Any]] = []
    for shot_id, events in grouped.items():
        info = shot_time[shot_id]
        events = sorted(events, key=lambda x: int(float(x.get("order_in_shot", 1) or 1)))
        ratios: list[float] = []
        for event in events:
            try:
                ratio = max(float(event.get("duration_ratio", 0.0)), 0.0)
            except Exception:
                ratio = 0.0
            ratios.append(ratio)
        total_ratio = sum(ratios)
        if total_ratio <= 0:
            continue

        current_abs = info["start_time"]
        for idx, (event, ratio) in enumerate(zip(events, ratios)):
            t0_abs = current_abs
            if idx == len(events) - 1:
                t1_abs = info["end_time"]
            else:
                t1_abs = current_abs + info["duration"] * (ratio / total_ratio)
            timeline_events.append(
                {
                    "t0": round(t0_abs - seq_start, round_digits),
                    "t1": round(t1_abs - seq_start, round_digits),
                    "target": normalize_target_name(event.get("target", "UNKNOWN")),
                    "source_shot_id": shot_id,
                    "gaze_intent": event.get("gaze_intent", ""),
                    "evidence": event.get("evidence", ""),
                }
            )
            current_abs = t1_abs

    return {
        "sequence_id": output["sequence_id"],
        "movie_id": output["movie_id"],
        "main_char": output["main_char"],
        "events": sorted(timeline_events, key=lambda x: (x["t0"], x["t1"], x["source_shot_id"])),
    }


def response_cache_key(sequence_id: str, main_char: str, model_name: str, prompt: str) -> str:
    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    payload = json.dumps(
        {
            "sequence_id": sequence_id,
            "main_char": main_char,
            "model": model_name,
            "prompt_sha256": prompt_sha,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_cache(cache_path: Path | None) -> dict[str, dict[str, Any]]:
    if cache_path is None or not cache_path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            key = row.get("cache_key")
            if key:
                out[str(key)] = row
    return out


def append_cache(cache_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def select_sequences(records: list[dict[str, Any]], sequence_id_list: list[str] | None, max_sequences: int | None) -> list[dict[str, Any]]:
    if sequence_id_list:
        wanted = set(sequence_id_list)
        records = [record for record in records if record.get("sequence_id") in wanted]
    if max_sequences is not None:
        records = records[:max_sequences]
    return records


def select_characters(sequence: dict[str, Any], max_characters: int | None) -> list[str]:
    speakers = [str(x).strip() for x in sequence.get("active_speakers", []) if str(x).strip()]
    if not speakers:
        return []
    return speakers if max_characters is None else speakers[:max_characters]


def values_from_run_config(run_config_path: Path) -> dict[str, Any]:
    run_config = load_yaml(run_config_path)
    config_base_dir = run_config_path.parent.parent.parent
    base_config_path = resolve_path(run_config.get("inputs", {}).get("base_config"), config_base_dir)
    base_config = load_yaml(base_config_path) if base_config_path is not None and base_config_path.exists() else {}
    paths_config_path = resolve_path(run_config.get("inputs", {}).get("paths_config"), config_base_dir)
    paths_config = load_yaml(paths_config_path) if paths_config_path is not None and paths_config_path.exists() else {}
    project_root = Path(paths_config.get("project", {}).get("root", config_base_dir))

    def from_project(path_value: str | Path | None) -> Path | None:
        return resolve_path(path_value, project_root)

    outputs = run_config.get("outputs", {})
    stage = run_config.get("stages", {}).get("generate_llm_gaze_script", {})
    base_llm = base_config.get("llm", {})
    gaze_script = base_config.get("gaze_script", {})
    llm_gaze_dir = from_project(outputs.get("llm_gaze_dir")) or project_root / "outputs" / "llm_gaze_scripts"
    logs_dir = from_project(outputs.get("logs_dir")) or project_root / "outputs" / "logs"
    cache_dir = from_project(outputs.get("cache_dir")) or logs_dir

    return {
        "enabled": bool(stage.get("enabled", True)),
        "overwrite": bool(stage.get("overwrite", False)),
        "movie_id": run_config.get("data", {}).get("movie_id"),
        "candidate_sequences_jsonl": from_project(outputs.get("candidate_sequences_jsonl")),
        "llm_gaze_dir": llm_gaze_dir,
        "summary_json": logs_dir / "03_generate_llm_gaze_script_summary.json",
        "cache_jsonl": cache_dir / "03_llm_gaze_script_cache.jsonl",
        "max_sequences": stage.get("max_sequences", 20),
        "sequence_id_list": stage.get("sequence_id_list"),
        "max_characters_per_sequence": stage.get("max_characters_per_sequence", 1),
        "cache_responses": bool(stage.get("cache_responses", True)),
        "save_prompt": bool(stage.get("save_prompt", True)),
        "save_raw_response": bool(stage.get("save_raw_response", True)),
        "validate_schema": bool(stage.get("validate_schema", True)),
        "model": str(stage.get("model", base_llm.get("model", MODEL_NAME))),
        "api_key_env": str(stage.get("api_key_env", base_llm.get("api_key_env", "OPENAI_API_KEY"))),
        "max_output_tokens": int(stage.get("max_output_tokens", base_llm.get("max_output_tokens", MAX_OUTPUT_TOKENS))),
        "allowed_intents": list(gaze_script.get("allowed_intents", [])),
        "allowed_targets": list(gaze_script.get("allowed_targets", [])),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LLM gaze scripts for candidate sequences.")
    parser.add_argument("--run-config", type=Path, default=None)
    parser.add_argument("--candidate-sequences-jsonl", type=Path, default=None)
    parser.add_argument("--llm-gaze-dir", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--cache-jsonl", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Build tasks/prompts and summary without calling OpenAI.")
    parser.add_argument("--max-sequences", type=str, default=None)
    parser.add_argument("--sequence-id-list", type=str, default=None)
    parser.add_argument("--max-characters-per-sequence", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--api-key-env", type=str, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values: dict[str, Any] = {}
    if args.run_config is not None:
        values.update(values_from_run_config(args.run_config))

    if not bool(values.get("enabled", True)):
        print("Skipping generate_llm_gaze_script because it is disabled in the run config.")
        return

    candidate_path = args.candidate_sequences_jsonl or values.get("candidate_sequences_jsonl")
    llm_gaze_dir = args.llm_gaze_dir or values.get("llm_gaze_dir")
    summary_json = args.summary_json or values.get("summary_json")
    cache_jsonl = args.cache_jsonl or values.get("cache_jsonl")
    if candidate_path is None or llm_gaze_dir is None:
        raise ValueError("candidate sequences JSONL and llm gaze output directory are required.")

    candidate_path = Path(candidate_path)
    llm_gaze_dir = Path(llm_gaze_dir)
    summary_json = Path(summary_json) if summary_json is not None else None
    cache_jsonl = Path(cache_jsonl) if cache_jsonl is not None else None
    prompt_dir = llm_gaze_dir / "prompts"
    raw_dir = llm_gaze_dir / "raw_responses"

    overwrite = bool(args.overwrite or values.get("overwrite", False))
    max_sequences = parse_auto_int(args.max_sequences if args.max_sequences is not None else values.get("max_sequences", 20), 20)
    max_chars = parse_auto_int(
        args.max_characters_per_sequence
        if args.max_characters_per_sequence is not None
        else values.get("max_characters_per_sequence", 1),
        1,
    )
    sequence_id_list = parse_sequence_id_list(
        args.sequence_id_list if args.sequence_id_list is not None else values.get("sequence_id_list")
    )
    model_name = args.model or values.get("model", MODEL_NAME)
    api_key_env = args.api_key_env or values.get("api_key_env", "OPENAI_API_KEY")
    max_output_tokens = args.max_output_tokens or values.get("max_output_tokens", MAX_OUTPUT_TOKENS)
    allowed_intents = values.get("allowed_intents") or ["ADDRESS", "LISTEN", "THINK", "AVOID", "REACT", "OBSERVE"]
    allowed_targets = values.get("allowed_targets") or ["CHARACTER", "OBJECT", "OFFSCREEN", "DOWN", "UP", "LEFT", "RIGHT", "GROUP"]
    cache_responses = bool(values.get("cache_responses", True))
    save_prompt = bool(values.get("save_prompt", True))
    save_raw_response = bool(values.get("save_raw_response", True))
    validate_schema = bool(values.get("validate_schema", True))

    sequences = select_sequences(load_jsonl(candidate_path), sequence_id_list, max_sequences)
    existing_cache = load_cache(cache_jsonl) if cache_responses else {}
    new_cache_rows: list[dict[str, Any]] = []
    client: Any | None = None

    requested = 0
    skipped_existing = 0
    cache_hits = 0
    api_calls = 0
    zero_event_outputs = 0
    validation_warnings: dict[str, list[str]] = {}
    written_outputs: list[str] = []
    dry_run_tasks: list[dict[str, Any]] = []

    for sequence in sequences:
        for main_char in select_characters(sequence, max_chars):
            sequence_id = str(sequence.get("sequence_id", "sequence"))
            char_slug = safe_slug(main_char)
            output_path = llm_gaze_dir / f"{sequence_id}__{char_slug}.json"
            timeline_path = llm_gaze_dir / f"{sequence_id}__{char_slug}__timeline.json"
            package = build_sequence_package(sequence, main_char)
            prompt = build_prompt(package, allowed_intents, allowed_targets)
            key = response_cache_key(sequence_id, main_char, str(model_name), prompt)
            requested += 1

            if output_path.exists() and timeline_path.exists() and not overwrite:
                skipped_existing += 1
                continue

            if save_prompt:
                prompt_path = prompt_dir / f"{sequence_id}__{char_slug}.txt"
                prompt_path.parent.mkdir(parents=True, exist_ok=True)
                prompt_path.write_text(prompt, encoding="utf-8")

            if args.dry_run:
                dry_run_tasks.append({"sequence_id": sequence_id, "main_char": main_char, "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()})
                continue

            cached = existing_cache.get(key)
            if cached is not None:
                raw_response = cached.get("raw_response", "")
                parsed = cached.get("parsed_response", {})
                cache_hits += 1
            else:
                if client is None:
                    client = make_openai_client(str(api_key_env))
                raw_response = call_llm_with_retry(client, prompt, str(model_name), int(max_output_tokens))
                parsed = extract_json_object(raw_response)
                api_calls += 1
                new_cache_rows.append(
                    {
                        "cache_key": key,
                        "sequence_id": sequence_id,
                        "movie_id": sequence.get("movie_id"),
                        "main_char": main_char,
                        "model": model_name,
                        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                        "prompt": prompt,
                        "raw_response": raw_response,
                        "parsed_response": parsed,
                    }
                )

            if save_raw_response:
                raw_path = raw_dir / f"{sequence_id}__{char_slug}.txt"
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_text(str(raw_response), encoding="utf-8")

            output = normalize_output(parsed, sequence, main_char)
            warnings = validate_output(output, sequence, allowed_intents) if validate_schema else []
            if warnings:
                validation_warnings[f"{sequence_id}__{char_slug}"] = warnings
            if not output.get("events"):
                zero_event_outputs += 1
            timeline = convert_output_to_timeline(output, sequence)
            write_json(output_path, output)
            write_json(timeline_path, timeline)
            written_outputs.extend([str(output_path), str(timeline_path)])

    if cache_jsonl is not None and cache_responses and new_cache_rows:
        append_cache(cache_jsonl, new_cache_rows)

    summary = {
        "movie_id": values.get("movie_id"),
        "candidate_sequences_jsonl": str(candidate_path),
        "llm_gaze_dir": str(llm_gaze_dir),
        "sequence_count": int(len(sequences)),
        "requested_count": int(requested),
        "skipped_existing_count": int(skipped_existing),
        "cache_hit_count": int(cache_hits),
        "api_call_count": int(api_calls),
        "written_file_count": int(len(written_outputs)),
        "zero_event_output_count": int(zero_event_outputs),
        "validation_warning_count": int(sum(len(x) for x in validation_warnings.values())),
        "validation_warnings": validation_warnings,
        "dry_run": bool(args.dry_run),
        "dry_run_tasks": dry_run_tasks[:50],
        "model": str(model_name),
        "max_sequences": AUTO if max_sequences is None else int(max_sequences),
        "max_characters_per_sequence": AUTO if max_chars is None else int(max_chars),
    }
    if summary_json is not None:
        write_json(summary_json, summary)

    print(f"Movie: {summary['movie_id']}")
    print(f"Sequences: {summary['sequence_count']}")
    print(f"Requested sequence-character tasks: {summary['requested_count']}")
    print(f"Skipped existing: {summary['skipped_existing_count']}")
    print(f"Cache hits: {summary['cache_hit_count']}")
    print(f"API calls: {summary['api_call_count']}")
    print(f"Written files: {summary['written_file_count']}")
    if summary_json is not None:
        print(f"Summary: {summary_json}")


if __name__ == "__main__":
    main()
