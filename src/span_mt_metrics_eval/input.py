"""Input normalization and optional text-bound validation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from span_mt_metrics_eval.spans import ErrorSpan


def normalize_input(value: list, name: str) -> list[list[ErrorSpan]]:
    """Normalize user-provided spans into a list of segment span lists.

    ``value`` may be a flat list of spans for one segment or a nested list of
    per-segment spans. ``name`` is only used in error messages. The output
    always has shape ``list[list[ErrorSpan]]``.
    """

    if value is None:
        raise ValueError(f"{name} cannot be None")

    if not isinstance(value, list):
        raise TypeError(f"{name} must be a  list of spans or a list of lists of spans")

    if not value:
        return [[]]

    if is_span_like(value[0]):
        return [[ErrorSpan.from_any(item) for item in value]]
    if all(isinstance(segment, list) for segment in value):
        normalized_segments: list[list[ErrorSpan]] = []
        for segment_idx, segment in enumerate(value):
            if not all(is_span_like(item) for item in segment):
                raise TypeError(f"{name}[{segment_idx}] must be a list of spans")
            normalized_segments.append([ErrorSpan.from_any(item) for item in segment])
        return normalized_segments
    raise TypeError(f"{name} must be a list of spans or a list of lists of spans")


def is_span_like(value: Any) -> bool:
    """Return whether a value looks like an accepted span object.

    The input can be an ``ErrorSpan``, a dictionary with ``start``/``end``, or an
    object exposing ``start`` and ``end`` attributes. The output is used by
    ``normalize_input`` to distinguish flat and nested span lists.
    """

    if isinstance(value, ErrorSpan):
        return True
    if isinstance(value, dict):
        return "start" in value and "end" in value
    return hasattr(value, "start") and hasattr(value, "end")


def coerce_texts(
    texts: str | Sequence[str],
    num_segments: int,
    name: str,
) -> list[str]:
    """Normalize optional source/target text arguments.

    ``texts`` may be a single string for one segment or a sequence of strings.
    The returned list is aligned with the normalized segment list length.
    """

    if isinstance(texts, str):
        if num_segments != 1:
            raise ValueError(f"{name} must contain {num_segments} entries")
        return [texts]

    text_list = list(texts)
    if len(text_list) != num_segments:
        raise ValueError(f"{name} must contain {num_segments} entries")
    for idx, text in enumerate(text_list):
        if text is None:
            raise ValueError(f"{name} must contain {num_segments} non-None entries")
        if not isinstance(text, str):
            raise TypeError(f"{name}[{idx}] must be a string")
    return text_list


def validate_text_bounds(
    segments: list[list[ErrorSpan]],
    source_texts: list[str] | None,
    target_texts: list[str] | None,
    name: str,
) -> None:
    """Validate that span offsets fit inside optional source/target texts.

    Inputs are normalized segments and text lists aligned by segment. The
    function returns nothing and raises when a span end offset exceeds the text
    length for its side.
    """

    for segment_idx, spans in enumerate(segments):
        for span_idx, span in enumerate(spans):
            texts = source_texts if span.side == "source" else target_texts
            if texts is None:
                continue
            text = texts[segment_idx]
            if span.end > len(text):
                raise ValueError(
                    f"{name}[{segment_idx}][{span_idx}] ends at {span.end}, "
                    f"which exceeds the {span.side} text length {len(text)}"
                )
