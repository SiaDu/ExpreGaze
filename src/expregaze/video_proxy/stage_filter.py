"""Shared stage-type filtering helpers for the video proxy pipeline."""

from __future__ import annotations

from collections import Counter
from typing import Any


DEFAULT_STAGE_TYPE_INCLUDE = ("single_speaking", "two_person_dialogue_simple")


def parse_stage_type_include(value: Any, *, default: tuple[str, ...] | None = DEFAULT_STAGE_TYPE_INCLUDE) -> set[str] | None:
    """Parse a stage_type include value.

    Returns None for "all", otherwise a set of stage_type strings. If value is
    absent, the supplied default is used.
    """
    if value is None:
        return None if default is None else set(default)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in {"none", "null"}:
            return None if default is None else set(default)
        if stripped.lower() == "all":
            return None
        return {item.strip() for item in stripped.split(",") if item.strip()}
    if isinstance(value, (list, tuple, set)):
        items = {str(item).strip() for item in value if str(item).strip()}
        return items or (None if default is None else set(default))
    raise ValueError(f"Unsupported stage_type_include value: {value!r}")


def resolve_stage_type_include(cli_value: Any, config_value: Any) -> set[str] | None:
    if cli_value is not None:
        return parse_stage_type_include(cli_value)
    return parse_stage_type_include(config_value)


def stage_type_allowed(row: dict[str, Any], include: set[str] | None) -> bool:
    if include is None:
        return True
    return str(row.get("stage_type", "")).strip() in include


def stage_type_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("stage_type", "") or "missing") for row in rows))


def filter_manifest_rows_by_stage_type(
    rows: list[dict[str, Any]],
    include: set[str] | None,
) -> tuple[list[dict[str, Any]], int, dict[str, int]]:
    selected = [row for row in rows if stage_type_allowed(row, include)]
    return selected, len(rows) - len(selected), stage_type_counts(rows)


def stage_type_include_label(include: set[str] | None) -> str:
    return "all" if include is None else ",".join(sorted(include))
