#!/usr/bin/env python3
"""
Align an existing shot-level table with a MovieNet screenplay.

Stage 01 intentionally treats the shot-level table as the single source of
subtitle/shot truth. It does not read the original .srt, annotation, or meta
files. The default mode is "raw", which exports inspectable alignment debug
files before any final full_context table or LLM repair is produced.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd
import yaml

try:
    from openai import OpenAI, RateLimitError
except Exception:  # pragma: no cover
    OpenAI = None
    RateLimitError = None


MODEL_NAME = "gpt-5-mini"
LLM_MAX_OUTPUT_TOKENS = 1000
LLM_MAX_RETRIES = 5
LLM_DEFAULT_RETRY_WAIT = 3.0
LLM_CANDIDATE_RADIUS = 8
LLM_REPAIR_SCOPE = "unmatched_or_low_confidence"
LLM_AUTO_MAX_ROWS = "auto"

RAW_MODE = "raw"
FINAL_MODE = "final"
LLM_MODE = "llm"
VALID_MODES = {RAW_MODE, FINAL_MODE, LLM_MODE}

SHOT_PREFIXES = [
    "INT.",
    "EXT.",
    "INT/EXT.",
    "EXT/INT.",
    "I/E.",
    "MS",
    "LS",
    "MLS",
    "MCS",
    "CS",
    "CU",
    "MCU",
    "ECU",
    "FS",
    "WS",
    "POV",
    "PAN",
    "ANGLE",
    "CLOSE SHOT",
    "LONG SHOT",
    "MED. SHOT",
    "MED SHOT",
    "DISSOLVE TO",
    "FADE IN",
    "FADE OUT",
    "CUT TO",
]
SHOT_PREFIX_PATTERN = r"(?:" + "|".join(re.escape(x) for x in sorted(SHOT_PREFIXES, key=len, reverse=True)) + r")"
HEADING_LINE_RE = re.compile(rf"^\s*(?:{SHOT_PREFIX_PATTERN})(?:\b|\s|:|-|\.|/).*$", re.IGNORECASE)
PAGE_NUMBER_RE = re.compile(r"^\s*\d+\.?\s*$")
CONTINUED_RE = re.compile(r"^\s*\(?CONTINUED\)?[: ]?.*$", re.IGNORECASE)
SPEAKER_ALLOWED_RE = re.compile(r"^[A-Z0-9\s\-()'.#/&]+$")

LOCAL_FILL_SYSTEM_PROMPT = """
You align screenplay dialogue candidates to one shot-level subtitle segment.
Choose the single best candidate dialogue segment, or return no match if none
of the candidates are plausible. Preserve chronology and prefer dialogue
overlap. Output valid JSON only.
""".strip()

LOCAL_FILL_USER_TEMPLATE = """
Return exactly one JSON object:
{{
  "match": true or false,
  "dialogue_segment_idx": integer or null,
  "confidence": number from 0 to 100,
  "reason": "short reason"
}}

Movie ID: {movie_id}
Shot ID: {shot_id}
Shot index: {shot_idx}
Raw subtitle segment:
{raw_text}

Candidate screenplay dialogue segments:
{candidate_segments}
""".strip()


@dataclass(frozen=True)
class TextRep:
    text: str
    norm: str
    tokens: frozenset[str]
    token_count: int


@dataclass(frozen=True)
class ScoreDetail:
    score: float
    overlap_recall: float
    overlap_precision: float
    token_f1: float
    length_ratio: float


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def resolve_path(path_value: str | Path | None, base_dir: Path) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else base_dir / path


def clean_line(s: Any) -> str:
    text = str(s).replace("\ufeff", "").replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def get_indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def is_noise_line(line: str) -> bool:
    text = line.strip()
    if not text:
        return False
    if PAGE_NUMBER_RE.match(text):
        return True
    if CONTINUED_RE.match(text):
        return True
    return False


def is_heading_line(line: str) -> bool:
    text = clean_line(line)
    return bool(text and HEADING_LINE_RE.match(text))


def is_speaker_line(line: str) -> bool:
    text = line.strip()
    indent = get_indent(line.rstrip("\n"))
    if not text or is_noise_line(line):
        return False
    if is_heading_line(line):
        return False
    if indent < 8 or len(text) > 48:
        return False
    if not SPEAKER_ALLOWED_RE.fullmatch(text):
        return False
    if not any(ch.isalpha() for ch in text):
        return False
    if text.upper().startswith(("FADE", "CUT TO", "DISSOLVE", "INT.", "EXT.", "INT/", "EXT/")):
        return False
    return True


def is_dialogue_candidate(line: str) -> bool:
    text = line.strip()
    indent = get_indent(line.rstrip("\n"))
    if not text or is_noise_line(line):
        return False
    if is_heading_line(line) or is_speaker_line(line):
        return False
    return 1 <= indent <= 36


def canonical_speaker(s: str) -> str:
    speaker = clean_line(s).upper()
    speaker = speaker.replace(" (CONT'D)", "").replace("(CONT'D)", "")
    return re.sub(r"\s+", " ", speaker).strip()


def parse_jsonish_list(value: Any) -> list[Any]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            for loader in (json.loads, ast.literal_eval):
                try:
                    out = loader(text)
                    if isinstance(out, list):
                        return out
                except Exception:
                    pass
        if " | " in text:
            return [part.strip() for part in text.split(" | ") if part.strip()]
        return [text]
    return [value]


def join_subtitle_sentences(value: Any) -> str:
    return " ".join(clean_line(x) for x in parse_jsonish_list(value) if clean_line(x)).strip()


REQUIRED_SHOT_COLUMNS = [
    "movie_id",
    "story_id",
    "story_description",
    "shot_idx",
    "shot_start_time",
    "shot_end_time",
    "shot_start_time_hms",
    "shot_end_time_hms",
    "subtitle_sentences",
    "cast_pids",
    "num_cast",
]


def load_shot_level(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        df = pd.DataFrame(rows)
    elif suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and isinstance(raw.get("data"), list):
            df = pd.DataFrame(raw["data"])
        else:
            df = pd.DataFrame(raw)
    else:
        raise ValueError(f"Unsupported shot_level format: {path}")

    missing = [col for col in REQUIRED_SHOT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"shot_level is missing required columns: {missing}")

    df = df.copy()
    if "shot_id" not in df.columns:
        df["shot_id"] = df["shot_idx"].apply(lambda x: f"shot_{int(x):04d}")
    if "subtitle_text" not in df.columns:
        df["subtitle_text"] = df["subtitle_sentences"].apply(join_subtitle_sentences)
    else:
        df["subtitle_text"] = df["subtitle_text"].fillna("").astype(str)

    df["cast_pids"] = df["cast_pids"].apply(lambda x: json.dumps(parse_jsonish_list(x), ensure_ascii=False))
    df = df.sort_values(["shot_idx", "shot_start_time"], kind="stable").reset_index(drop=True)
    return df


def build_raw_segments(shot_df: pd.DataFrame) -> pd.DataFrame:
    raw = shot_df[
        [
            "movie_id",
            "shot_idx",
            "shot_id",
            "shot_start_time",
            "shot_end_time",
            "shot_start_time_hms",
            "shot_end_time_hms",
            "subtitle_text",
        ]
    ].copy()
    raw.insert(0, "raw_segment_idx", range(len(raw)))
    raw = raw.rename(
        columns={
            "shot_start_time": "start_sec",
            "shot_end_time": "end_sec",
            "subtitle_text": "raw_text",
        }
    )
    return raw


def parse_dialogue_segments(script_text: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    current_speaker: str | None = None
    current_lines: list[str] = []
    current_start: int | None = None

    def flush(end_line: int) -> None:
        nonlocal current_speaker, current_lines, current_start
        if current_speaker is not None and current_lines:
            rows.append(
                {
                    "dialogue_segment_idx": len(rows),
                    "speaker": canonical_speaker(current_speaker),
                    "dialogue_text": clean_line(" ".join(current_lines)),
                    "start_line": current_start,
                    "end_line": end_line,
                }
            )
        current_speaker = None
        current_lines = []
        current_start = None

    for line_no, raw in enumerate(script_text.splitlines(), start=1):
        if is_noise_line(raw):
            continue
        if is_speaker_line(raw):
            flush(line_no - 1)
            current_speaker = raw.strip()
            continue
        if current_speaker is not None and is_dialogue_candidate(raw):
            if current_start is None:
                current_start = line_no
            current_lines.append(raw.strip())
            continue
        if current_speaker is not None:
            flush(line_no - 1)

    flush(len(script_text.splitlines()))
    return pd.DataFrame(rows)


def parse_other_chunks(script_text: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    buffer: list[tuple[int, str]] = []
    in_dialogue = False

    def flush() -> None:
        nonlocal buffer
        text = "\n".join(line for _, line in buffer if line.strip()).strip()
        if text:
            rows.append(
                {
                    "other_idx": len(rows),
                    "start_line": buffer[0][0],
                    "end_line": buffer[-1][0],
                    "other_text": text,
                }
            )
        buffer = []

    for line_no, raw in enumerate(script_text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or is_noise_line(raw):
            if buffer:
                flush()
            in_dialogue = False
            continue
        if is_speaker_line(raw):
            if buffer:
                flush()
            in_dialogue = True
            continue
        if in_dialogue and is_dialogue_candidate(raw):
            continue
        in_dialogue = False
        buffer.append((line_no, stripped))

    if buffer:
        flush()
    return pd.DataFrame(rows)


def normalize_text(s: Any) -> str:
    text = "" if s is None else str(s)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\b[A-Z][A-Z'\-\s]{1,25}:\s*", " ", text)
    text = text.lower().replace("...", " ").replace("--", " ")
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def text_rep(text: str) -> TextRep:
    norm = normalize_text(text)
    tokens = frozenset(tok for tok in norm.split() if tok)
    return TextRep(text=text, norm=norm, tokens=tokens, token_count=len(norm.split()))


def combine_reps(reps: Sequence[TextRep]) -> TextRep:
    text = " ".join(rep.text for rep in reps if rep.text).strip()
    norm = " ".join(rep.norm for rep in reps if rep.norm).strip()
    tokens: set[str] = set()
    token_count = 0
    for rep in reps:
        tokens.update(rep.tokens)
        token_count += rep.token_count
    return TextRep(text=text, norm=norm, tokens=frozenset(tokens), token_count=token_count)


def build_span_reps(texts: Sequence[str], max_span: int) -> list[list[Optional[TextRep]]]:
    base = [text_rep(text) for text in texts]
    spans: list[list[Optional[TextRep]]] = []
    for idx in range(len(base)):
        row: list[Optional[TextRep]] = [None] * (max_span + 1)
        for span_len in range(1, max_span + 1):
            if idx + span_len <= len(base):
                row[span_len] = combine_reps(base[idx : idx + span_len])
        spans.append(row)
    return spans


def score_reps(raw_rep: TextRep, dialogue_rep: TextRep) -> ScoreDetail:
    if not raw_rep.norm or not dialogue_rep.norm or not raw_rep.tokens or not dialogue_rep.tokens:
        return ScoreDetail(0.0, 0.0, 0.0, 0.0, 0.0)
    overlap = len(raw_rep.tokens & dialogue_rep.tokens)
    if overlap == 0:
        return ScoreDetail(0.0, 0.0, 0.0, 0.0, 0.0)
    recall = overlap / max(len(raw_rep.tokens), 1)
    precision = overlap / max(len(dialogue_rep.tokens), 1)
    token_f1 = 0.0 if recall + precision == 0 else 2 * recall * precision / (recall + precision)
    length_ratio = min(raw_rep.token_count, dialogue_rep.token_count) / max(raw_rep.token_count, dialogue_rep.token_count, 1)
    score = 0.50 * recall + 0.35 * token_f1 + 0.15 * length_ratio
    return ScoreDetail(
        score=max(0.0, min(1.0, score)),
        overlap_recall=recall,
        overlap_precision=precision,
        token_f1=token_f1,
        length_ratio=length_ratio,
    )


def align_raw_to_dialogue_dp(
    raw_df: pd.DataFrame,
    dialogue_df: pd.DataFrame,
    max_raw_span: int = 4,
    max_dialogue_span: int = 4,
    skip_raw_penalty: float = 0.15,
    skip_dialogue_penalty: float = 0.15,
    min_match_score: float = 0.45,
) -> list[dict[str, Any]]:
    n = len(raw_df)
    m = len(dialogue_df)
    if n == 0 or m == 0:
        return []

    raw_spans = build_span_reps(raw_df["raw_text"].fillna("").astype(str).tolist(), max_raw_span)
    dialogue_spans = build_span_reps(dialogue_df["dialogue_text"].fillna("").astype(str).tolist(), max_dialogue_span)

    neg = -1e18
    dp = [[neg] * (m + 1) for _ in range(n + 1)]
    back: list[list[Optional[tuple[Any, ...]]]] = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    for i in range(n + 1):
        for j in range(m + 1):
            cur = dp[i][j]
            if cur <= neg / 2:
                continue

            if i < n:
                cand = cur - skip_raw_penalty
                if cand > dp[i + 1][j]:
                    dp[i + 1][j] = cand
                    back[i + 1][j] = ("skip_raw", 1)

            if j < m:
                cand = cur - skip_dialogue_penalty
                if cand > dp[i][j + 1]:
                    dp[i][j + 1] = cand
                    back[i][j + 1] = ("skip_dialogue", 1)

            if i >= n or j >= m:
                continue
            for raw_span_len in range(1, max_raw_span + 1):
                raw_rep = raw_spans[i][raw_span_len] if i < n else None
                if raw_rep is None:
                    continue
                for dialogue_span_len in range(1, max_dialogue_span + 1):
                    dialogue_rep = dialogue_spans[j][dialogue_span_len] if j < m else None
                    if dialogue_rep is None:
                        continue
                    detail = score_reps(raw_rep, dialogue_rep)
                    if detail.score < min_match_score:
                        continue
                    cand = cur + detail.score
                    ni = i + raw_span_len
                    nj = j + dialogue_span_len
                    if cand > dp[ni][nj]:
                        dp[ni][nj] = cand
                        back[ni][nj] = ("match", raw_span_len, dialogue_span_len)

    i = n
    j = m
    steps: list[dict[str, Any]] = []
    while i > 0 or j > 0:
        item = back[i][j]
        if item is None:
            break
        if item[0] == "skip_raw":
            steps.append({"type": "skip_raw", "raw_range": (i - 1, i - 1), "dialogue_range": None})
            i -= 1
        elif item[0] == "skip_dialogue":
            steps.append({"type": "skip_dialogue", "raw_range": None, "dialogue_range": (j - 1, j - 1)})
            j -= 1
        else:
            _, raw_span_len, dialogue_span_len = item
            raw_start = i - raw_span_len
            dialogue_start = j - dialogue_span_len
            raw_rep = raw_spans[raw_start][raw_span_len]
            dialogue_rep = dialogue_spans[dialogue_start][dialogue_span_len]
            assert raw_rep is not None and dialogue_rep is not None
            detail = score_reps(raw_rep, dialogue_rep)
            steps.append(
                {
                    "type": "match",
                    "raw_range": (raw_start, i - 1),
                    "dialogue_range": (dialogue_start, j - 1),
                    "detail": detail,
                }
            )
            i = raw_start
            j = dialogue_start

    steps.reverse()
    return steps


def unique_join(values: Sequence[Any], sep: str = "\n\n") -> str:
    seen: list[str] = []
    for value in values:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        text = str(value).strip()
        if text and text not in seen:
            seen.append(text)
    return sep.join(seen)


def unique_json_list(values: Sequence[Any]) -> str:
    seen: list[str] = []
    for value in values:
        if isinstance(value, str):
            items = [value]
        elif isinstance(value, Sequence):
            items = [str(x) for x in value]
        else:
            items = [str(value)]
        for item in items:
            item = item.strip()
            if item and item not in seen:
                seen.append(item)
    return json.dumps(seen, ensure_ascii=False)


def find_prev_other(line_start: Any, other_df: pd.DataFrame, max_gap: int = 25) -> tuple[Any, Any, str]:
    if line_start is None or pd.isna(line_start) or other_df.empty:
        return None, None, ""
    cand = other_df[other_df["end_line"] < int(line_start)].copy()
    if cand.empty:
        return None, None, ""
    cand["gap"] = int(line_start) - cand["end_line"]
    cand = cand[cand["gap"] <= max_gap]
    if cand.empty:
        return None, None, ""
    row = cand.sort_values("gap").iloc[0]
    return int(row["other_idx"]), int(row["gap"]), str(row["other_text"])


def find_next_other(line_end: Any, other_df: pd.DataFrame, max_gap: int = 25) -> tuple[Any, Any, str]:
    if line_end is None or pd.isna(line_end) or other_df.empty:
        return None, None, ""
    cand = other_df[other_df["start_line"] > int(line_end)].copy()
    if cand.empty:
        return None, None, ""
    cand["gap"] = cand["start_line"] - int(line_end)
    cand = cand[cand["gap"] <= max_gap]
    if cand.empty:
        return None, None, ""
    row = cand.sort_values("gap").iloc[0]
    return int(row["other_idx"]), int(row["gap"]), str(row["other_text"])


def build_alignment_steps_df(
    steps: Sequence[dict[str, Any]],
    raw_df: pd.DataFrame,
    dialogue_df: pd.DataFrame,
    other_df: pd.DataFrame,
    confident_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for step_idx, step in enumerate(steps):
        row: dict[str, Any] = {"step_idx": step_idx, "type": step["type"]}
        raw_range = step.get("raw_range")
        dialogue_range = step.get("dialogue_range")

        if raw_range is not None:
            r0, r1 = raw_range
            raw_sub = raw_df.iloc[r0 : r1 + 1]
            row.update(
                {
                    "raw_segment_start": int(r0),
                    "raw_segment_end": int(r1),
                    "shot_idx_list": json.dumps([int(x) for x in raw_sub["shot_idx"].tolist()]),
                    "shot_id_list": json.dumps([str(x) for x in raw_sub["shot_id"].tolist()], ensure_ascii=False),
                    "best_shot_idx": int(raw_sub.iloc[0]["shot_idx"]),
                    "best_shot_id": str(raw_sub.iloc[0]["shot_id"]),
                    "start_sec": float(raw_sub.iloc[0]["start_sec"]) if pd.notna(raw_sub.iloc[0]["start_sec"]) else None,
                    "end_sec": float(raw_sub.iloc[-1]["end_sec"]) if pd.notna(raw_sub.iloc[-1]["end_sec"]) else None,
                    "raw_text": " | ".join(raw_sub["raw_text"].fillna("").astype(str).tolist()).strip(),
                }
            )
        else:
            row.update(
                {
                    "raw_segment_start": None,
                    "raw_segment_end": None,
                    "shot_idx_list": json.dumps([]),
                    "shot_id_list": json.dumps([], ensure_ascii=False),
                    "best_shot_idx": None,
                    "best_shot_id": "",
                    "start_sec": None,
                    "end_sec": None,
                    "raw_text": "",
                }
            )

        if dialogue_range is not None:
            d0, d1 = dialogue_range
            dialogue_sub = dialogue_df.iloc[d0 : d1 + 1]
            speaker_list = dialogue_sub["speaker"].tolist()
            dialogue_text = " | ".join(dialogue_sub["dialogue_text"].fillna("").astype(str).tolist()).strip()
            start_line = int(dialogue_sub.iloc[0]["start_line"])
            end_line = int(dialogue_sub.iloc[-1]["end_line"])
            prev_id, prev_gap, prev_text = find_prev_other(start_line, other_df)
            next_id, next_gap, next_text = find_next_other(end_line, other_df)
            row.update(
                {
                    "dialogue_segment_start": int(d0),
                    "dialogue_segment_end": int(d1),
                    "speaker_list": json.dumps(speaker_list, ensure_ascii=False),
                    "dialogue_text": dialogue_text,
                    "script_start_line": start_line,
                    "script_end_line": end_line,
                    "prev_other_idx": prev_id,
                    "prev_other_gap": prev_gap,
                    "prev_other_text": prev_text,
                    "next_other_idx": next_id,
                    "next_other_gap": next_gap,
                    "next_other_text": next_text,
                }
            )
        else:
            row.update(
                {
                    "dialogue_segment_start": None,
                    "dialogue_segment_end": None,
                    "speaker_list": json.dumps([], ensure_ascii=False),
                    "dialogue_text": "",
                    "script_start_line": None,
                    "script_end_line": None,
                    "prev_other_idx": None,
                    "prev_other_gap": None,
                    "prev_other_text": "",
                    "next_other_idx": None,
                    "next_other_gap": None,
                    "next_other_text": "",
                }
            )

        detail = step.get("detail")
        if detail is not None:
            row.update(
                {
                    "match_score": float(detail.score),
                    "overlap_recall": float(detail.overlap_recall),
                    "overlap_precision": float(detail.overlap_precision),
                    "token_f1": float(detail.token_f1),
                    "length_ratio": float(detail.length_ratio),
                    "is_confident_match": bool(detail.score >= confident_threshold),
                }
            )
        else:
            row.update(
                {
                    "match_score": None,
                    "overlap_recall": None,
                    "overlap_precision": None,
                    "token_f1": None,
                    "length_ratio": None,
                    "is_confident_match": False,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def make_full_context_df(
    shot_df: pd.DataFrame,
    steps_df: pd.DataFrame,
    confident_threshold: float,
    match_source: str = "raw_dp",
) -> pd.DataFrame:
    out = shot_df.copy()
    out["best_shot_idx"] = out["shot_idx"]
    out["best_shot_id"] = out["shot_id"]
    out["aligned_raw_text"] = ""
    out["aligned_script_dialogue"] = ""
    out["aligned_speakers"] = json.dumps([], ensure_ascii=False)
    out["aligned_script_text"] = ""
    out["prev_other_text"] = ""
    out["bridge_other_text"] = ""
    out["next_other_text"] = ""
    out["match_score"] = 0.0
    out["match_source"] = ""
    out["script_block_start_idx"] = None
    out["script_block_window_len"] = None
    out["script_start_line"] = None
    out["script_end_line"] = None

    matched = steps_df[
        (steps_df["type"] == "match")
        & (steps_df["match_score"].fillna(0.0) >= confident_threshold)
        & steps_df["raw_segment_start"].notna()
    ].copy()

    for _, step in matched.iterrows():
        raw_start = int(step["raw_segment_start"])
        raw_end = int(step["raw_segment_end"])
        speakers = step["speaker_list"] if isinstance(step["speaker_list"], str) else json.dumps([], ensure_ascii=False)
        for raw_idx in range(raw_start, raw_end + 1):
            if raw_idx >= len(out):
                continue
            out.at[raw_idx, "aligned_raw_text"] = step["raw_text"]
            out.at[raw_idx, "aligned_script_dialogue"] = step["dialogue_text"]
            out.at[raw_idx, "aligned_speakers"] = speakers
            out.at[raw_idx, "aligned_script_text"] = unique_join(
                [step["prev_other_text"], step["dialogue_text"], step["next_other_text"]]
            )
            out.at[raw_idx, "prev_other_text"] = step["prev_other_text"]
            out.at[raw_idx, "next_other_text"] = step["next_other_text"]
            out.at[raw_idx, "match_score"] = round(float(step["match_score"]), 4)
            out.at[raw_idx, "match_source"] = match_source
            out.at[raw_idx, "script_block_start_idx"] = int(step["dialogue_segment_start"])
            out.at[raw_idx, "script_block_window_len"] = int(step["dialogue_segment_end"] - step["dialogue_segment_start"] + 1)
            out.at[raw_idx, "script_start_line"] = int(step["script_start_line"])
            out.at[raw_idx, "script_end_line"] = int(step["script_end_line"])

    return out


def build_summary(
    movie_id: str,
    mode: str,
    shot_level_path: Path,
    script_path: Path,
    raw_df: pd.DataFrame,
    dialogue_df: pd.DataFrame,
    other_df: pd.DataFrame,
    steps_df: pd.DataFrame,
    confident_threshold: float,
) -> dict[str, Any]:
    match_mask = steps_df["type"].eq("match") if "type" in steps_df.columns else pd.Series(dtype=bool)
    confident_mask = (
        match_mask & steps_df["match_score"].fillna(0.0).ge(confident_threshold)
        if "match_score" in steps_df.columns
        else pd.Series(dtype=bool)
    )
    return {
        "movie_id": movie_id,
        "mode": mode,
        "shot_level_path": str(shot_level_path),
        "script_path": str(script_path),
        "raw_segment_count": int(len(raw_df)),
        "dialogue_segment_count": int(len(dialogue_df)),
        "other_chunk_count": int(len(other_df)),
        "alignment_step_count": int(len(steps_df)),
        "match_count": int(match_mask.sum()) if len(steps_df) else 0,
        "confident_match_count": int(confident_mask.sum()) if len(steps_df) else 0,
        "raw_segments_with_text": int((raw_df["raw_text"].fillna("").astype(str).str.strip() != "").sum()),
        "mean_match_score": (
            float(steps_df.loc[match_mask, "match_score"].mean())
            if len(steps_df) and match_mask.any()
            else None
        ),
    }


def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in df.to_dict(orient="records"):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_outputs(
    mode: str,
    raw_df: pd.DataFrame,
    dialogue_df: pd.DataFrame,
    steps_df: pd.DataFrame,
    full_df: pd.DataFrame | None,
    llm_candidates_df: pd.DataFrame | None,
    paths: dict[str, Path | None],
    summary: dict[str, Any],
) -> None:
    for key, df in (
        ("raw_segments_csv", raw_df),
        ("script_segments_csv", dialogue_df),
        ("raw_alignment_csv", steps_df),
    ):
        path = paths.get(key)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False, encoding="utf-8-sig")

    summary_path = paths.get("summary_json")
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if full_df is None:
        return

    candidates_path = paths.get("llm_candidates_csv")
    if candidates_path is not None and llm_candidates_df is not None:
        candidates_path.parent.mkdir(parents=True, exist_ok=True)
        llm_candidates_df.to_csv(candidates_path, index=False, encoding="utf-8-sig")

    preview_path = paths.get("preview_csv")
    if preview_path is not None:
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        full_df.head(50).to_csv(preview_path, index=False, encoding="utf-8-sig")

    if mode in {FINAL_MODE, LLM_MODE}:
        if mode == LLM_MODE:
            output_csv = paths.get("llm_output_csv")
            output_jsonl = paths.get("llm_output_jsonl")
        else:
            output_csv = paths.get("output_csv")
            output_jsonl = paths.get("output_jsonl")
        if output_csv is not None:
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            full_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        if output_jsonl is not None:
            write_jsonl(full_df, output_jsonl)


def get_retry_wait_seconds(exc: Exception, default_wait: float = LLM_DEFAULT_RETRY_WAIT) -> float:
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
    for attempt in range(LLM_MAX_RETRIES):
        try:
            response = client.responses.create(
                model=model_name,
                instructions=LOCAL_FILL_SYSTEM_PROMPT,
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
            if not is_rate_limit or attempt == LLM_MAX_RETRIES - 1:
                raise
            time.sleep(get_retry_wait_seconds(exc))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("LLM request failed without exception.")


def extract_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find JSON object in model output: {text[:500]}")
    return json.loads(match.group(0))


def choose_llm_candidates(row_idx: int, full_df: pd.DataFrame, dialogue_df: pd.DataFrame) -> pd.DataFrame:
    matched = full_df[full_df["script_block_start_idx"].notna()].copy()
    if not matched.empty:
        before = matched[matched.index < row_idx]
        after = matched[matched.index > row_idx]
        prev_idx = int(before.iloc[-1]["script_block_start_idx"]) if not before.empty else None
        next_idx = int(after.iloc[0]["script_block_start_idx"]) if not after.empty else None
        if prev_idx is not None and next_idx is not None and prev_idx <= next_idx:
            lo = max(0, prev_idx - LLM_CANDIDATE_RADIUS)
            hi = min(len(dialogue_df) - 1, next_idx + LLM_CANDIDATE_RADIUS)
        elif prev_idx is not None:
            lo = max(0, prev_idx - LLM_CANDIDATE_RADIUS)
            hi = min(len(dialogue_df) - 1, prev_idx + LLM_CANDIDATE_RADIUS)
        elif next_idx is not None:
            lo = max(0, next_idx - LLM_CANDIDATE_RADIUS)
            hi = min(len(dialogue_df) - 1, next_idx + LLM_CANDIDATE_RADIUS)
        else:
            lo = 0
            hi = min(len(dialogue_df) - 1, 2 * LLM_CANDIDATE_RADIUS)
    else:
        approx = round(row_idx / max(len(full_df) - 1, 1) * max(len(dialogue_df) - 1, 0))
        lo = max(0, approx - LLM_CANDIDATE_RADIUS)
        hi = min(len(dialogue_df) - 1, approx + LLM_CANDIDATE_RADIUS)
    return dialogue_df.iloc[lo : hi + 1].copy()


def parse_max_llm_rows(value: Any) -> int | None:
    if value is None:
        return 25
    if isinstance(value, str):
        text = value.strip().lower()
        if text == LLM_AUTO_MAX_ROWS:
            return None
        if not text:
            return 25
        value = text
    parsed = int(value)
    if parsed < 0:
        raise ValueError("max_llm_rows must be non-negative or 'auto'.")
    return parsed


def format_max_llm_rows(value: int | None) -> str | int:
    return LLM_AUTO_MAX_ROWS if value is None else int(value)


def build_llm_repair_candidates(full_df: pd.DataFrame, repair_scope: str = LLM_REPAIR_SCOPE) -> pd.DataFrame:
    if repair_scope != LLM_REPAIR_SCOPE:
        raise ValueError(f"Unsupported llm_repair_scope: {repair_scope}")
    mask = (
        (full_df["subtitle_text"].fillna("").astype(str).str.strip() != "")
        & (full_df["match_source"].fillna("").astype(str).str.strip() == "")
    )
    rows: list[dict[str, Any]] = []
    for row_idx, row in full_df.loc[mask].iterrows():
        rows.append(
            {
                "row_idx": int(row_idx),
                "shot_id": row.get("shot_id", ""),
                "shot_idx": row.get("shot_idx", ""),
                "subtitle_text": row.get("subtitle_text", ""),
                "match_source": row.get("match_source", ""),
                "repair_reason": repair_scope,
            }
        )
    return pd.DataFrame(
        rows,
        columns=["row_idx", "shot_id", "shot_idx", "subtitle_text", "match_source", "repair_reason"],
    )


def limit_llm_repair_candidates(candidates_df: pd.DataFrame, max_llm_rows: int | None) -> pd.DataFrame:
    if max_llm_rows is None:
        return candidates_df.copy()
    return candidates_df.head(max_llm_rows).copy()


def cache_key(movie_id: str, row_idx: int, shot_id: Any, model_name: str, prompt: str) -> str:
    payload = json.dumps(
        {
            "movie_id": movie_id,
            "row_idx": int(row_idx),
            "shot_id": str(shot_id),
            "model": model_name,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_llm_cache(cache_path: Path | None) -> dict[str, dict[str, Any]]:
    if cache_path is None or not cache_path.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            key = row.get("cache_key")
            if not key:
                key = "|".join(
                    [
                        str(row.get("movie_id", "")),
                        str(row.get("row_idx", "")),
                        str(row.get("shot_id", "")),
                        str(row.get("model", "")),
                    ]
                )
            cache[str(key)] = row
    return cache


def apply_llm_parsed_response(
    repaired: pd.DataFrame,
    dialogue_df: pd.DataFrame,
    row_idx: int,
    parsed: dict[str, Any],
    confidence_threshold: float,
) -> bool:
    if not parsed.get("match"):
        return False
    confidence = float(parsed.get("confidence", 0.0) or 0.0)
    if confidence < confidence_threshold * 100.0:
        return False
    dialogue_idx = parsed.get("dialogue_segment_idx")
    if dialogue_idx is None:
        return False
    matched = dialogue_df[dialogue_df["dialogue_segment_idx"].eq(int(dialogue_idx))]
    if matched.empty:
        return False
    row = repaired.loc[row_idx]
    dlg = matched.iloc[0]
    repaired.at[row_idx, "aligned_raw_text"] = row["subtitle_text"]
    repaired.at[row_idx, "aligned_script_dialogue"] = dlg["dialogue_text"]
    repaired.at[row_idx, "aligned_speakers"] = json.dumps([dlg["speaker"]], ensure_ascii=False)
    repaired.at[row_idx, "aligned_script_text"] = dlg["dialogue_text"]
    repaired.at[row_idx, "match_score"] = round(confidence / 100.0, 4)
    repaired.at[row_idx, "match_source"] = "llm_local_fill"
    repaired.at[row_idx, "script_block_start_idx"] = int(dlg["dialogue_segment_idx"])
    repaired.at[row_idx, "script_block_window_len"] = 1
    repaired.at[row_idx, "script_start_line"] = int(dlg["start_line"])
    repaired.at[row_idx, "script_end_line"] = int(dlg["end_line"])
    return True


def repair_with_llm(
    full_df: pd.DataFrame,
    dialogue_df: pd.DataFrame,
    movie_id: str,
    model_name: str,
    api_key_env: str,
    max_output_tokens: int,
    max_llm_rows: int | None,
    confidence_threshold: float,
    cache_path: Path | None,
    cache_responses: bool,
    repair_scope: str = LLM_REPAIR_SCOPE,
) -> tuple[pd.DataFrame, dict[str, int]]:
    all_candidates = build_llm_repair_candidates(full_df, repair_scope)
    selected_candidates = limit_llm_repair_candidates(all_candidates, max_llm_rows)
    stats = {
        "llm_candidate_count": int(len(all_candidates)),
        "llm_requested_count": int(len(selected_candidates)),
        "llm_cache_hit_count": 0,
        "llm_api_call_count": 0,
        "llm_repaired_count": 0,
        "llm_remaining_count": int(len(all_candidates)),
    }
    if selected_candidates.empty:
        return full_df.copy(), stats

    client: Any | None = None
    repaired = full_df.copy()
    existing_cache = load_llm_cache(cache_path) if cache_responses else {}
    new_cache_rows: list[dict[str, Any]] = []

    for _, candidate in selected_candidates.iterrows():
        row_idx = int(candidate["row_idx"])
        row = repaired.loc[row_idx]
        cand_df = choose_llm_candidates(int(row_idx), repaired, dialogue_df)
        rendered = []
        for _, cand in cand_df.iterrows():
            rendered.append(
                json.dumps(
                    {
                        "dialogue_segment_idx": int(cand["dialogue_segment_idx"]),
                        "speaker": cand["speaker"],
                        "dialogue_text": cand["dialogue_text"],
                        "start_line": int(cand["start_line"]),
                        "end_line": int(cand["end_line"]),
                    },
                    ensure_ascii=False,
                )
            )
        prompt = LOCAL_FILL_USER_TEMPLATE.format(
            movie_id=movie_id,
            shot_id=row["shot_id"],
            shot_idx=row["shot_idx"],
            raw_text=row["subtitle_text"],
            candidate_segments="\n".join(rendered),
        )
        key = cache_key(movie_id, row_idx, row["shot_id"], model_name, prompt)
        legacy_key = "|".join([movie_id, str(row_idx), str(row["shot_id"]), ""])
        cached = existing_cache.get(key) or existing_cache.get(legacy_key)
        if cached is not None:
            parsed = cached.get("parsed_response", {})
            stats["llm_cache_hit_count"] += 1
        else:
            if client is None:
                client = make_openai_client(api_key_env)
            raw_response = call_llm_with_retry(client, prompt, model_name, max_output_tokens)
            parsed = extract_json_object(raw_response)
            stats["llm_api_call_count"] += 1
            new_cache_rows.append(
                {
                    "cache_key": key,
                    "movie_id": movie_id,
                    "row_idx": int(row_idx),
                    "shot_id": row["shot_id"],
                    "model": model_name,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "prompt": prompt,
                    "raw_response": raw_response,
                    "parsed_response": parsed,
                }
            )
        if apply_llm_parsed_response(repaired, dialogue_df, row_idx, parsed, confidence_threshold):
            stats["llm_repaired_count"] += 1

    remaining_candidates = build_llm_repair_candidates(repaired, repair_scope)
    stats["llm_remaining_count"] = int(len(remaining_candidates))

    if cache_path is not None and cache_responses and new_cache_rows:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("a", encoding="utf-8") as f:
            for cache_row in new_cache_rows:
                f.write(json.dumps(cache_row, ensure_ascii=False) + "\n")
    return repaired, stats


def values_from_run_config(run_config_path: Path) -> dict[str, Any]:
    run_config = load_yaml(run_config_path)
    config_base_dir = run_config_path.parent.parent.parent
    base_config_path = resolve_path(run_config.get("inputs", {}).get("base_config"), config_base_dir)
    base_config = load_yaml(base_config_path) if base_config_path is not None and base_config_path.exists() else {}
    paths_config_path = resolve_path(run_config.get("inputs", {}).get("paths_config"), config_base_dir)
    paths_config = load_yaml(paths_config_path) if paths_config_path is not None else {}
    project_root = Path(paths_config.get("project", {}).get("root", config_base_dir))

    def from_project(path_value: str | Path | None) -> Path | None:
        return resolve_path(path_value, project_root)

    movie_config_path = from_project(run_config.get("run", {}).get("movie_config"))
    movie_config = load_yaml(movie_config_path) if movie_config_path is not None and movie_config_path.exists() else {}
    paths = paths_config.get("paths", {})
    files = movie_config.get("files", {})
    outputs = run_config.get("outputs", {})
    stage = run_config.get("stages", {}).get("align_full_context_with_gpt", {})
    base_llm = base_config.get("llm", {})
    logs_dir = from_project(outputs.get("logs_dir")) or project_root / "outputs" / "logs"
    cache_dir = from_project(outputs.get("cache_dir")) or logs_dir
    script_file = files.get("script_file")
    script_dir = from_project(paths.get("script_dir"))

    return {
        "enabled": bool(stage.get("enabled", True)),
        "overwrite": bool(stage.get("overwrite", False)),
        "mode": str(stage.get("mode", RAW_MODE)),
        "movie_id": run_config.get("data", {}).get("movie_id") or movie_config.get("movie", {}).get("movie_id"),
        "shot_level": from_project(outputs.get("shot_level_csv")),
        "script": script_dir / script_file if script_dir is not None and script_file else None,
        "output_csv": from_project(outputs.get("full_context_csv")),
        "output_jsonl": from_project(outputs.get("full_context_jsonl")),
        "llm_output_csv": from_project(stage.get("llm_output_csv")),
        "llm_output_jsonl": from_project(stage.get("llm_output_jsonl")),
        "raw_segments_csv": logs_dir / "01_raw_segments.csv",
        "script_segments_csv": logs_dir / "01_script_dialogue_segments.csv",
        "raw_alignment_csv": logs_dir / "01_raw_alignment_steps.csv",
        "summary_json": logs_dir / "01_align_summary.json",
        "preview_csv": logs_dir / "01_full_context_preview.csv",
        "llm_candidates_csv": logs_dir / "01_llm_repair_candidates.csv",
        "llm_cache_jsonl": cache_dir / "01_llm_local_fill_cache.jsonl",
        "max_raw_span": int(stage.get("max_raw_span", 4)),
        "max_dialogue_span": int(stage.get("max_dialogue_span", 4)),
        "skip_raw_penalty": float(stage.get("skip_raw_penalty", 0.15)),
        "skip_dialogue_penalty": float(stage.get("skip_dialogue_penalty", 0.15)),
        "min_match_score": float(stage.get("min_match_score", 0.45)),
        "confident_threshold": float(stage.get("confident_threshold", 0.60)),
        "model": str(stage.get("model", base_llm.get("model", MODEL_NAME))),
        "api_key_env": str(stage.get("api_key_env", base_llm.get("api_key_env", "OPENAI_API_KEY"))),
        "llm_max_output_tokens": int(stage.get("llm_max_output_tokens", base_llm.get("max_output_tokens", LLM_MAX_OUTPUT_TOKENS))),
        "llm_repair_scope": str(stage.get("llm_repair_scope", LLM_REPAIR_SCOPE)),
        "max_llm_rows": stage.get("max_llm_rows", 25),
        "cache_responses": bool(stage.get("cache_responses", True)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align shot-level rows with screenplay dialogue.")
    parser.add_argument("--run-config", type=Path, default=None)
    parser.add_argument("--mode", choices=sorted(VALID_MODES), default=None)
    parser.add_argument("--shot-level", "--shot_level", dest="shot_level", type=Path, default=None)
    parser.add_argument("--script", type=Path, default=None)
    parser.add_argument("--output-csv", "--output", dest="output_csv", type=Path, default=None)
    parser.add_argument("--output-jsonl", dest="output_jsonl", type=Path, default=None)
    parser.add_argument("--llm-output-csv", dest="llm_output_csv", type=Path, default=None)
    parser.add_argument("--llm-output-jsonl", dest="llm_output_jsonl", type=Path, default=None)
    parser.add_argument("--raw-segments-csv", type=Path, default=None)
    parser.add_argument("--script-segments-csv", type=Path, default=None)
    parser.add_argument("--raw-alignment-csv", type=Path, default=None)
    parser.add_argument("--llm-candidates-csv", type=Path, default=None)
    parser.add_argument("--preview-csv", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--disable-llm", action="store_true", help="Compatibility alias: forces mode=final when mode=llm.")
    parser.add_argument("--max-raw-span", type=int, default=None)
    parser.add_argument("--max-dialogue-span", type=int, default=None)
    parser.add_argument("--skip-raw-penalty", type=float, default=None)
    parser.add_argument("--skip-dialogue-penalty", type=float, default=None)
    parser.add_argument("--min-match-score", type=float, default=None)
    parser.add_argument("--confident-threshold", type=float, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--api-key-env", type=str, default=None)
    parser.add_argument("--llm-max-output-tokens", type=int, default=None)
    parser.add_argument("--llm-repair-scope", type=str, default=None)
    parser.add_argument("--max-llm-rows", type=str, default=None)
    return parser.parse_args()


def existing_mode_outputs(mode: str, paths: dict[str, Path | None]) -> list[Path]:
    if mode == RAW_MODE:
        keys = ["raw_segments_csv", "script_segments_csv", "raw_alignment_csv", "summary_json"]
    elif mode == LLM_MODE:
        keys = ["llm_output_csv", "llm_output_jsonl"]
    else:
        keys = ["output_csv", "output_jsonl"]
    return [paths[key] for key in keys if paths.get(key) is not None and paths[key].exists()]


def main() -> None:
    args = parse_args()
    values: dict[str, Any] = {}
    if args.run_config is not None:
        values.update(values_from_run_config(args.run_config))

    mode = args.mode or values.get("mode") or RAW_MODE
    if args.disable_llm and mode == LLM_MODE:
        mode = FINAL_MODE
    if mode not in VALID_MODES:
        raise ValueError(f"Unsupported mode: {mode}")

    enabled = bool(values.get("enabled", True))
    if not enabled:
        print("Skipping align_full_context_with_gpt because it is disabled in the run config.")
        return

    movie_id = values.get("movie_id")
    shot_level_path = args.shot_level or values.get("shot_level")
    script_path = args.script or values.get("script")
    if shot_level_path is None:
        raise ValueError("shot-level path is required. Provide --shot-level or outputs.shot_level_csv in run config.")
    if script_path is None:
        raise ValueError("script path is required. Provide --script or movie_config files.script_file + paths.script_dir.")
    shot_level_path = Path(shot_level_path)
    script_path = Path(script_path)
    if movie_id is None:
        movie_id = shot_level_path.stem.split("_shot_level")[0].replace("__", "")

    paths = {
        "output_csv": args.output_csv or values.get("output_csv"),
        "output_jsonl": args.output_jsonl or values.get("output_jsonl"),
        "llm_output_csv": args.llm_output_csv or values.get("llm_output_csv"),
        "llm_output_jsonl": args.llm_output_jsonl or values.get("llm_output_jsonl"),
        "raw_segments_csv": args.raw_segments_csv or values.get("raw_segments_csv"),
        "script_segments_csv": args.script_segments_csv or values.get("script_segments_csv"),
        "raw_alignment_csv": args.raw_alignment_csv or values.get("raw_alignment_csv"),
        "summary_json": args.summary_json or values.get("summary_json"),
        "preview_csv": args.preview_csv or values.get("preview_csv"),
        "llm_candidates_csv": args.llm_candidates_csv or values.get("llm_candidates_csv"),
        "llm_cache_jsonl": values.get("llm_cache_jsonl"),
    }
    paths = {key: Path(value) if value is not None else None for key, value in paths.items()}
    if paths.get("llm_output_csv") is None and paths.get("output_csv") is not None:
        output_csv = paths["output_csv"]
        paths["llm_output_csv"] = output_csv.with_name(f"{output_csv.stem}_llm{output_csv.suffix}")
    if paths.get("llm_output_jsonl") is None and paths.get("output_jsonl") is not None:
        output_jsonl = paths["output_jsonl"]
        paths["llm_output_jsonl"] = output_jsonl.with_name(f"{output_jsonl.stem}_llm{output_jsonl.suffix}")
    overwrite = bool(args.overwrite or values.get("overwrite", False))
    existing_outputs = existing_mode_outputs(mode, paths)
    if existing_outputs and not overwrite:
        print(f"Skipping align_full_context_with_gpt because {mode} outputs already exist and overwrite=false.")
        for path in existing_outputs:
            print(f"Existing output: {path}")
        return

    max_raw_span = args.max_raw_span or values.get("max_raw_span", 4)
    max_dialogue_span = args.max_dialogue_span or values.get("max_dialogue_span", 4)
    skip_raw_penalty = args.skip_raw_penalty or values.get("skip_raw_penalty", 0.15)
    skip_dialogue_penalty = args.skip_dialogue_penalty or values.get("skip_dialogue_penalty", 0.15)
    min_match_score = args.min_match_score or values.get("min_match_score", 0.45)
    confident_threshold = args.confident_threshold or values.get("confident_threshold", 0.60)
    model_name = args.model or values.get("model", MODEL_NAME)
    api_key_env = args.api_key_env or values.get("api_key_env", "OPENAI_API_KEY")
    llm_max_output_tokens = args.llm_max_output_tokens or values.get("llm_max_output_tokens", LLM_MAX_OUTPUT_TOKENS)
    llm_repair_scope = args.llm_repair_scope or values.get("llm_repair_scope", LLM_REPAIR_SCOPE)
    max_llm_rows = parse_max_llm_rows(args.max_llm_rows if args.max_llm_rows is not None else values.get("max_llm_rows", 25))
    cache_responses = bool(values.get("cache_responses", True))

    shot_df = load_shot_level(shot_level_path)
    raw_df = build_raw_segments(shot_df)
    script_text = script_path.read_text(encoding="utf-8", errors="ignore")
    dialogue_df = parse_dialogue_segments(script_text)
    other_df = parse_other_chunks(script_text)
    if dialogue_df.empty:
        raise RuntimeError(f"No dialogue segments parsed from script: {script_path}")

    steps = align_raw_to_dialogue_dp(
        raw_df=raw_df,
        dialogue_df=dialogue_df,
        max_raw_span=int(max_raw_span),
        max_dialogue_span=int(max_dialogue_span),
        skip_raw_penalty=float(skip_raw_penalty),
        skip_dialogue_penalty=float(skip_dialogue_penalty),
        min_match_score=float(min_match_score),
    )
    steps_df = build_alignment_steps_df(
        steps=steps,
        raw_df=raw_df,
        dialogue_df=dialogue_df,
        other_df=other_df,
        confident_threshold=float(confident_threshold),
    )
    full_df = make_full_context_df(shot_df, steps_df, float(confident_threshold))
    llm_candidates_df = build_llm_repair_candidates(full_df, str(llm_repair_scope))
    selected_llm_candidates_df = limit_llm_repair_candidates(llm_candidates_df, max_llm_rows)
    llm_stats = {
        "llm_candidate_count": int(len(llm_candidates_df)),
        "llm_requested_count": int(len(selected_llm_candidates_df)) if mode == LLM_MODE else 0,
        "llm_cache_hit_count": 0,
        "llm_api_call_count": 0,
        "llm_repaired_count": 0,
        "llm_remaining_count": int(len(llm_candidates_df)),
        "max_llm_rows": format_max_llm_rows(max_llm_rows),
    }
    if mode == LLM_MODE:
        full_df, repair_stats = repair_with_llm(
            full_df=full_df,
            dialogue_df=dialogue_df,
            movie_id=str(movie_id),
            model_name=str(model_name),
            api_key_env=str(api_key_env),
            max_output_tokens=int(llm_max_output_tokens),
            max_llm_rows=max_llm_rows,
            confidence_threshold=float(confident_threshold),
            cache_path=paths.get("llm_cache_jsonl"),
            cache_responses=cache_responses,
            repair_scope=str(llm_repair_scope),
        )
        llm_stats.update(repair_stats)
        llm_stats["max_llm_rows"] = format_max_llm_rows(max_llm_rows)

    summary = build_summary(
        movie_id=str(movie_id),
        mode=mode,
        shot_level_path=shot_level_path,
        script_path=script_path,
        raw_df=raw_df,
        dialogue_df=dialogue_df,
        other_df=other_df,
        steps_df=steps_df,
        confident_threshold=float(confident_threshold),
    )
    summary.update(llm_stats)
    write_outputs(
        mode=mode,
        raw_df=raw_df,
        dialogue_df=dialogue_df,
        steps_df=steps_df,
        full_df=full_df,
        llm_candidates_df=llm_candidates_df,
        paths=paths,
        summary=summary,
    )

    print(f"Movie: {movie_id}")
    print(f"Mode: {mode}")
    print(f"Raw segments: {summary['raw_segment_count']}")
    print(f"Dialogue segments: {summary['dialogue_segment_count']}")
    print(f"Matches: {summary['match_count']}")
    print(f"Confident matches: {summary['confident_match_count']}")
    print(f"LLM repair candidates: {summary['llm_candidate_count']}")
    if mode == LLM_MODE:
        print(f"LLM requested rows: {summary['llm_requested_count']}")
        print(f"LLM cache hits: {summary['llm_cache_hit_count']}")
        print(f"LLM API calls: {summary['llm_api_call_count']}")
        print(f"LLM repaired rows: {summary['llm_repaired_count']}")
        print(f"LLM remaining rows: {summary['llm_remaining_count']}")
    if paths.get("summary_json") is not None:
        print(f"Summary: {paths['summary_json']}")
    if paths.get("llm_candidates_csv") is not None:
        print(f"LLM repair candidates CSV: {paths['llm_candidates_csv']}")
    if mode == FINAL_MODE and paths.get("output_csv") is not None:
        print(f"Full context CSV: {paths['output_csv']}")
    if mode == LLM_MODE and paths.get("llm_output_csv") is not None:
        print(f"LLM full context CSV: {paths['llm_output_csv']}")


if __name__ == "__main__":
    main()
