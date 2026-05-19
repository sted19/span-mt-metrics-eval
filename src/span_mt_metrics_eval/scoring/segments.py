"""Segment-level metric dispatch."""

from __future__ import annotations

from span_mt_metrics_eval.options import MatchingAlgorithm, MatchingStrategy, Measure
from span_mt_metrics_eval.results import ScoreComponents
from span_mt_metrics_eval.scoring.many_to_many import compute_m2m_score_components
from span_mt_metrics_eval.scoring.one_to_one import compute_o2o_score_components
from span_mt_metrics_eval.spans import ErrorSpan


def compute_score_components(
    predictions: list[ErrorSpan],
    references: list[ErrorSpan],
    *,
    measure: Measure,
    matching: MatchingStrategy,
    matching_algorithm: MatchingAlgorithm,
    severity_penalty: float,
    severity_weights: dict[str, float] | None,
) -> ScoreComponents:
    """Compute one segment's raw score components for the selected metric.

    Count-style measures return ``TPCounts``. Measures that have separate
    prediction-side and reference-side score numerators return
    ``SideScoreComponents``.
    """

    if matching == "one_to_one":
        return compute_o2o_score_components(
            predictions,
            references,
            measure,
            matching_algorithm,
            severity_penalty,
            severity_weights,
        )
    return compute_m2m_score_components(
        predictions, references, measure, severity_penalty, severity_weights
    )
