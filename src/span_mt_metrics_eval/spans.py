"""Span model and normalization helpers for error annotations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal, Mapping


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ErrorSpan:
    """A half-open error span on either the source or target side.

    ``start`` and ``end`` are character offsets using normal Python slicing
    semantics: ``start`` is included and ``end`` is excluded. ``side`` selects
    the source or target text. ``severity`` is optional unless metric options
    require severity-aware scoring.
    """

    start: int
    end: int
    side: Literal["source", "target"] = "target"
    severity: str | None = None

    def __post_init__(self) -> None:
        """Validate offsets and normalize side/severity after initialization.

        The method reads the dataclass fields supplied by the caller and returns
        nothing. It raises a descriptive exception if the span cannot be used by
        the metric code.
        """

        _validate_offset(self.start, "start")
        _validate_offset(self.end, "end")
        normalized_side = normalize_side(self.side)
        normalized_severity = normalize_severity(self.severity)

        if self.start >= self.end:
            raise ValueError(f"Invalid span ({self.start}, {self.end}): start must be < end")

        object.__setattr__(self, "side", normalized_side)
        object.__setattr__(self, "severity", normalized_severity)

    @property
    def length(self) -> int:
        """Return the number of characters covered by this span."""

        return self.end - self.start

    @classmethod
    def from_any(cls, value: Any) -> "ErrorSpan":
        """Create an ``ErrorSpan`` from an ``ErrorSpan`` or span dictionary.

        Accepted dictionaries contain ``start`` and ``end`` with an optional
        ``side`` field. Extra dictionary fields are ignored, except for the
        removed ``is_source_error`` alias, which raises a clear error so callers
        migrate to ``side`` instead. The returned span is fully validated and
        normalized.
        """

        if isinstance(value, ErrorSpan):
            return value

        if isinstance(value, Mapping):
            try:
                start = value["start"]
                end = value["end"]
            except KeyError as exc:
                raise ValueError("Span dictionaries must contain 'start' and 'end'") from exc

            if "is_source_error" in value:
                raise ValueError("Span dictionaries must use 'side', not 'is_source_error'")
            if "side" in value:
                side = value["side"]
            else:
                side = "target"
                logger.warning("Span dictionary is missing 'side'; defaulting to 'target'")

            return cls(
                start=start,
                end=end,
                side=side,
                severity=value.get("severity"),
            )

        raise TypeError("Spans must be ErrorSpan instances or dictionaries")


def span_key(span: ErrorSpan) -> tuple[str, int, int]:
    """Return the side/start/end identity used for exact span matching."""

    return span.side, span.start, span.end


def _validate_offset(value: Any, field_name: str) -> None:
    """Validate one user-provided span offset.

    ``field_name`` is used only in error messages. The function returns nothing
    and raises when the offset is not a non-negative integer.
    """

    if not isinstance(value, int):
        raise TypeError(f"Span {field_name} must be an integer")
    if value < 0:
        raise ValueError(f"Span {field_name} must be non-negative")


def normalize_side(value: Any) -> Literal["source", "target"]:
    """Normalize side aliases to ``source`` or ``target``.

    The input may be a full side name or a short string such as ``src``/``tgt``.
    The normalized side string is returned.
    """

    normalized = str(value).strip().lower()
    if normalized in {"source", "src"}:
        return "source"
    if normalized in {"target", "tgt", "translation"}:
        return "target"
    raise ValueError(f"Invalid span side {value!r}; expected 'source' or 'target'")


def normalize_severity(value: Any) -> str | None:
    """Normalize an optional severity label.

    ``None`` and blank strings are treated as missing severity. Non-empty string
    labels are stripped and lowercased for deterministic comparisons.
    """

    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("Span severity must be a string when provided")

    normalized = value.strip().lower()
    return normalized or None
