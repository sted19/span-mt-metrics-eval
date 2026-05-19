"""Result containers returned by span metric computation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

from span_mt_metrics_eval.options import (
    AveragingStrategy,
    MatchingAlgorithm,
    MatchingStrategy,
    Measure,
)


@dataclass(frozen=True)
class TPCounts:
    """Raw counts used by metrics with one shared true-positive value.

    Count-style metrics compute precision as ``tp / (tp + fp)`` and recall as
    ``tp / (tp + fn)``. Metrics with different prediction-side and
    reference-side numerators should use ``SideScoreComponents`` instead.
    """

    tp: float = 0.0
    fp: float = 0.0
    fn: float = 0.0

    def __add__(self, other: "TPCounts") -> "TPCounts":
        """Add two count containers field by field and return a new container."""

        return TPCounts(
            tp=self.tp + other.tp,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn,
        )

    def as_dict(self) -> dict[str, float]:
        """Return a JSON-serializable representation of the raw counts."""

        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }


@dataclass(frozen=True)
class SideScoreComponents:
    """Raw components for metrics with side-specific score numerators.

    The object directly stores the precision and recall fractions used to
    compute F-score. This fits metrics such as MPP, where prediction spans and
    reference spans receive separate average partial-credit scores.
    """

    precision_numerator: float = 0.0
    precision_denominator: float = 0.0
    recall_numerator: float = 0.0
    recall_denominator: float = 0.0

    def __add__(self, other: "SideScoreComponents") -> "SideScoreComponents":
        """Add two side-specific score containers and return a new container."""

        return SideScoreComponents(
            precision_numerator=self.precision_numerator + other.precision_numerator,
            precision_denominator=(
                self.precision_denominator + other.precision_denominator
            ),
            recall_numerator=self.recall_numerator + other.recall_numerator,
            recall_denominator=self.recall_denominator + other.recall_denominator,
        )

    def as_dict(self) -> dict[str, float]:
        """Return a JSON-serializable representation of the score components."""

        return {
            "precision_numerator": self.precision_numerator,
            "precision_denominator": self.precision_denominator,
            "recall_numerator": self.recall_numerator,
            "recall_denominator": self.recall_denominator,
        }


ScoreComponents: TypeAlias = TPCounts | SideScoreComponents


@dataclass(frozen=True)
class CountDetails:
    """Aggregate and per-segment raw counts for a count-style metric result."""

    counts: TPCounts
    segments_counts: list[TPCounts]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of count details."""

        return {
            "type": "counts",
            "counts": self.counts.as_dict(),
            "segments_counts": [counts.as_dict() for counts in self.segments_counts],
        }


@dataclass(frozen=True)
class SideScoreDetails:
    """Aggregate and per-segment components for side-specific score results."""

    components: SideScoreComponents
    segments_components: list[SideScoreComponents]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of side-score details."""

        return {
            "type": "side_score",
            "components": self.components.as_dict(),
            "segments_components": [
                components.as_dict() for components in self.segments_components
            ],
        }


MetricDetails: TypeAlias = CountDetails | SideScoreDetails


@dataclass(frozen=True)
class MetricConfig:
    """Configuration used to produce a metric result.

    The object records normalized options so callers can log or serialize the
    exact computation setup together with scores.
    """

    measure: Measure
    matching: MatchingStrategy
    matching_algorithm: MatchingAlgorithm | None
    averaging: AveragingStrategy
    severity_penalty: float = 0.0
    severity_weights: dict[str, float] | None = None

    def as_dict(self) -> dict[str, str | float | dict[str, float] | None]:
        """Return a JSON-serializable representation of the metric options."""

        return {
            "measure": self.measure,
            "matching": self.matching,
            "matching_algorithm": self.matching_algorithm,
            "averaging": self.averaging,
            "severity_penalty": self.severity_penalty,
            "severity_weights": self.severity_weights,
        }


@dataclass(frozen=True)
class MetricResult:
    """Precision, recall, F-score, raw details, and computation config.

    ``details`` stores the raw aggregate and per-segment values that fed the
    final score. Count-style metrics expose ``CountDetails``; metrics with
    side-specific precision/recall components expose ``SideScoreDetails``.
    """

    precision: float
    recall: float
    f_score: float
    details: MetricDetails
    config: MetricConfig

    def as_dict(self) -> dict[str, Any]:
        """Return the complete metric result in a JSON-serializable shape."""

        return {
            "precision": self.precision,
            "recall": self.recall,
            "f_score": self.f_score,
            "details": self.details.as_dict(),
            "config": self.config.as_dict(),
        }
