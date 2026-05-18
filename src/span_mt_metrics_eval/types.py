"""Public types for span-level metric computation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal, Mapping


Measure = Literal["EM", "MP", "WMT23", "MPP"]
MeasureOrAll = Literal["EM", "MP", "WMT23", "MPP", "all"]
MatchingStrategy = Literal["one_to_one", "many_to_many"]
MatchingAlgorithm = Literal["optimal", "greedy"]
AveragingStrategy = Literal["micro", "macro"]

MEASURES: tuple[Measure, ...] = ("EM", "MP", "WMT23", "MPP")
MATCHING_STRATEGIES: tuple[MatchingStrategy, ...] = ("one_to_one", "many_to_many")
MATCHING_ALGORITHMS: tuple[MatchingAlgorithm, ...] = ("optimal", "greedy")
AVERAGING_STRATEGIES: tuple[AveragingStrategy, ...] = ("micro", "macro")


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ErrorSpan:
    """A half-open error span on either the source or target side.

    Offsets follow the Python slicing convention: ``start`` is included and
    ``end`` is excluded. All metrics use these offsets as the authoritative span
    length. ``severity`` is optional unless the caller enables a non-zero
    severity penalty during metric computation.
    """

    start: int
    end: int
    side: Literal["source", "target"] = "target"
    severity: str | None = None

    def __post_init__(self) -> None:
        """Validate offsets and normalize the side after dataclass creation.

        Inputs are the dataclass fields supplied by the user. The method returns
        nothing; it raises a descriptive exception when the span cannot be used
        safely by the metric code.
        """

        _validate_offset(self.start, "start")
        _validate_offset(self.end, "end")
        normalized_side = _normalize_side(self.side)
        normalized_severity = _normalize_severity(self.severity)

        if self.start >= self.end:
            raise ValueError(f"Invalid span ({self.start}, {self.end}): start must be < end")

        object.__setattr__(self, "side", normalized_side)
        object.__setattr__(self, "severity", normalized_severity)

    @property
    def length(self) -> int:
        """Return the number of characters covered by the half-open span."""

        return self.end - self.start

    @property
    def is_source_error(self) -> bool:
        """Return whether this span belongs to the source text."""

        return self.side == "source"

    @classmethod
    def from_any(cls, value: Any) -> "ErrorSpan":
        """Create an ErrorSpan from an ErrorSpan or user-provided dictionary.

        The accepted dictionary shape is intentionally permissive: users may
        provide either ``side`` or the WMT-style ``is_source_error`` flag.
        Optional ``severity`` values are normalized for severity-aware scoring.
        The method returns a validated ``ErrorSpan``.
        """

        if isinstance(value, ErrorSpan):
            return value

        if isinstance(value, Mapping):
            try:
                start = value["start"]
                end = value["end"]
            except KeyError as exc:
                raise ValueError("Span dictionaries must contain 'start' and 'end'") from exc

            if "side" in value:
                side = value["side"]
            elif "is_source_error" in value:
                side = "source" if value["is_source_error"] else "target"
            else:
                side = "target"
                logger.warning("Span dictionary is missing 'side'; defaulting to 'target'")
            return cls(
                start=start,
                end=end,
                side=side,
                severity=value.get("severity"),
            )

        raise TypeError(
            "Spans must be ErrorSpan instances or dictionaries"
        )


@dataclass(frozen=True)
class TPCounts:
    """Raw counts used to compute precision, recall, and F-score.

    The original span metrics can assign different partial credit on the
    prediction side and the reference side. For example, one predicted span may
    cover half of a long reference span: precision and recall should then use
    different true-positive numerators.
    """

    tp_for_precision: float = 0.0
    tp_for_recall: float = 0.0
    fp: float = 0.0
    fn: float = 0.0

    def __add__(self, other: "TPCounts") -> "TPCounts":
        """Add two count containers field by field and return a new container."""

        return TPCounts(
            tp_for_precision=self.tp_for_precision + other.tp_for_precision,
            tp_for_recall=self.tp_for_recall + other.tp_for_recall,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn,
        )

    def as_dict(self) -> dict[str, float]:
        """Return a JSON-serializable representation of the raw counts."""

        return {
            "tp_for_precision": self.tp_for_precision,
            "tp_for_recall": self.tp_for_recall,
            "fp": self.fp,
            "fn": self.fn,
        }


@dataclass(frozen=True)
class MetricConfig:
    """Configuration used for a metric result.

    The object records the normalized options that produced a score. It is
    returned with every ``MetricResult`` so callers can safely log or serialize
    what was computed.
    """

    measure: Measure
    matching: MatchingStrategy
    matching_algorithm: MatchingAlgorithm | None
    averaging: AveragingStrategy
    severity_penalty: float = 0.0

    def as_dict(self) -> dict[str, str | float | None]:
        """Return a JSON-serializable representation of the metric options."""

        return {
            "measure": self.measure,
            "matching": self.matching,
            "matching_algorithm": self.matching_algorithm,
            "averaging": self.averaging,
            "severity_penalty": self.severity_penalty,
        }


@dataclass(frozen=True)
class MetricResult:
    """Precision, recall, F-score, counts, and computation config.

    ``tp_counts`` stores the aggregate counts used for the final score, while
    ``segments_tp_counts`` preserves the per-segment counts that feed macro
    averaging and can be useful for debugging.
    """

    precision: float
    recall: float
    f_score: float
    tp_counts: TPCounts
    segments_tp_counts: list[TPCounts]
    config: MetricConfig

    @property
    def counts(self) -> TPCounts:
        """Backward-compatible alias for the aggregate true-positive counts."""

        return self.tp_counts

    def as_dict(self) -> dict[str, Any]:
        """Return the complete metric result in a JSON-serializable shape."""

        return {
            "precision": self.precision,
            "recall": self.recall,
            "f_score": self.f_score,
            "counts": self.tp_counts.as_dict(),
            "segments_counts": [counts.as_dict() for counts in self.segments_tp_counts],
            "config": self.config.as_dict(),
        }


def _validate_offset(value: Any, field_name: str) -> None:
    """Validate one span offset.

    ``value`` is the user-provided offset and ``field_name`` is used only for
    error messages. The function returns nothing and raises when the offset is
    not a non-negative integer.
    """

    if not isinstance(value, int):
        raise TypeError(f"Span {field_name} must be an integer")
    if value < 0:
        raise ValueError(f"Span {field_name} must be non-negative")


def _normalize_side(value: Any) -> Literal["source", "target"]:
    """Normalize side aliases to ``source`` or ``target``.

    The input may be a boolean ``is_source_error`` value or a short string such
    as ``src``/``tgt``. The normalized side string is returned.
    """

    if isinstance(value, bool):
        return "source" if value else "target"

    normalized = str(value).strip().lower()
    if normalized in {"source", "src"}:
        return "source"
    if normalized in {"target", "tgt", "translation"}:
        return "target"
    raise ValueError(f"Invalid span side {value!r}; expected 'source' or 'target'")


def _normalize_severity(value: Any) -> str | None:
    """Normalize an optional severity label.

    ``None`` and blank strings are treated as missing severity. Non-empty string
    labels are stripped and lowercased so callers can use flexible labels while
    still getting deterministic equality checks during scoring.
    """

    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("Span severity must be a string when provided")

    normalized = value.strip().lower()
    return normalized or None
