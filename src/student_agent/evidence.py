"""Defensive readers for MCP evidence payloads.

The gateway guarantees the envelope (``mcp-evidence-response-v1``) but not the shape of
``data``. These helpers look fields up by a list of accepted aliases so agents never
crash on a missing or renamed field; a missing value is reported as ``None`` and is
never replaced by a guess.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from typing import Any

OLIST_ID = re.compile(r"\b[0-9a-f]{32}\b")
MONEY_TOLERANCE = 0.01


def walk_dicts(data: Any, max_depth: int = 6) -> Iterator[dict[str, Any]]:
    """Breadth-first iteration over every dict inside ``data``."""
    queue: list[tuple[Any, int]] = [(data, 0)]
    while queue:
        value, depth = queue.pop(0)
        if isinstance(value, dict):
            yield value
            if depth < max_depth:
                queue.extend((child, depth + 1) for child in value.values())
        elif isinstance(value, list) and depth < max_depth:
            queue.extend((child, depth + 1) for child in value)


def pick(data: Any, *names: str) -> Any:
    """First non-null value for any alias in ``names`` (case-insensitive, nested)."""
    wanted = [name.lower() for name in names]
    for mapping in walk_dicts(data):
        lowered = {str(key).lower(): value for key, value in mapping.items()}
        for name in wanted:
            value = lowered.get(name)
            if value is not None and value != "":
                return value
    return None


def pick_shallow(mapping: Any, *names: str) -> Any:
    if not isinstance(mapping, dict):
        return None
    lowered = {str(key).lower(): value for key, value in mapping.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value is not None and value != "":
            return value
    return None


def records(data: Any, *container_keys: str, marker: str | None = None) -> list[dict[str, Any]]:
    """Return a list of record dicts from a payload.

    ``data`` may already be a list, a dict holding a list under one of
    ``container_keys``, or a single record. When ``marker`` is given, any nested dict
    containing that key is accepted as a record.
    """
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in container_keys:
            value = pick_shallow(data, key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        if marker:
            found = [item for item in walk_dicts(data) if marker in item]
            if found:
                return found
        return [data]
    return []


def to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace("R$", "").replace(" ", "")
        if cleaned.count(",") == 1 and cleaned.count(".") == 0:
            cleaned = cleaned.replace(",", ".")
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def to_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    for candidate in (text, text.replace(" ", "T", 1)):
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            continue
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%Y %H:%M"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


def money(value: float | None) -> float:
    return round(value or 0.0, 2)


def close(a: float | None, b: float | None, tolerance: float = MONEY_TOLERANCE) -> bool:
    return a is not None and b is not None and abs(a - b) <= tolerance + 1e-9


def unique(values: Iterable[Any]) -> list[Any]:
    seen: set[Any] = set()
    result: list[Any] = []
    for value in values:
        if value is None or value == "" or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def strings(values: Iterable[Any]) -> list[str]:
    return unique(str(value) for value in values if value is not None and str(value).strip())


def text_blob(data: Any) -> str:
    """Concatenate every string value (used for keyword/ID extraction from the input)."""
    parts: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(data)
    return "\n".join(parts)
