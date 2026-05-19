"""Public metric computation entrypoint."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import cast

from span_mt_metrics_eval.input import coerce_texts, normalize_input, validate_text_bounds
from span_mt_metrics_eval.options import (
    AveragingStrategy,
    MatchingAlgorithm,
    MatchingStrategy,
    Measure,
)
from span_mt_metrics_eval.results import (
    CountDetails,
    MetricConfig,
    MetricResult,
    SideScoreComponents,
    SideScoreDetails,
    TPCounts,
)
from span_mt_metrics_eval.scoring.aggregation import (
    precision_recall_fscore,
    sum_counts,
    sum_side_score_components,
)
from span_mt_metrics_eval.scoring.segments import compute_score_components
from span_mt_metrics_eval.spans import ErrorSpan
from span_mt_metrics_eval.validation import (
    validate_averaging,
    validate_matching,
    validate_matching_algorithm,
    validate_measure,
    validate_required_severities,
    validate_required_severity_weights,
    validate_severity_penalty,
    validate_severity_weight_compatibility,
    validate_severity_weights,
)


def compute(
    predictions: list[ErrorSpan] | list[dict] | list[list[ErrorSpan]] | list[list[dict]],
    references: list[ErrorSpan] | list[dict] | list[list[ErrorSpan]] | list[list[dict]],
    *,
    measure: Measure = "MPP",
    matching: MatchingStrategy = "one_to_one",
    matching_algorithm: MatchingAlgorithm = "optimal",
    averaging: AveragingStrategy = "micro",
    severity_penalty: float = 0.0,
    severity_weights: Mapping[str, Real] | None = None,
    source_texts: str | list[str] | None = None,
    target_texts: str | list[str] | None = None,
) -> MetricResult:
    """Compute span-level precision, recall, and F-score.

    ``predictions`` and ``references`` can be a single segment's spans or a
    sequence of per-segment span sequences. Each span can be an ``ErrorSpan`` or
    a dict with ``start``, ``end``, and ``side`` keys.

    The metric is selected with ``measure``. This function returns a single
    ``MetricResult``. ``severity_penalty`` discounts matches whose normalized
    severity labels differ. ``severity_weights`` assigns severity-specific
    importance mass for MPP.
    """

    validate_measure(measure)
    validate_matching(matching)
    validate_matching_algorithm(matching_algorithm)
    validate_averaging(averaging)
    normalized_severity_penalty = validate_severity_penalty(severity_penalty)
    normalized_severity_weights = validate_severity_weights(severity_weights)
    validate_severity_weight_compatibility(
        measure, normalized_severity_penalty, normalized_severity_weights
    )

    normalized_predictions = normalize_input(predictions, "predictions")
    normalized_references = normalize_input(references, "references")

    if len(normalized_predictions) != len(normalized_references):
        raise ValueError(
            "predictions and references must contain the same number of elements"
            f"({len(normalized_predictions)} != {len(normalized_references)})"
        )

    if normalized_severity_penalty > 0.0:
        validate_required_severities(normalized_predictions, "predictions")
        validate_required_severities(normalized_references, "references")
    if normalized_severity_weights is not None:
        validate_required_severity_weights(
            normalized_predictions, "predictions", normalized_severity_weights
        )
        validate_required_severity_weights(
            normalized_references, "references", normalized_severity_weights
        )

    source_text_list = (
        coerce_texts(source_texts, len(normalized_predictions), "source_texts")
        if source_texts is not None
        else None
    )
    target_text_list = (
        coerce_texts(target_texts, len(normalized_predictions), "target_texts")
        if target_texts is not None
        else None
    )
    if source_text_list is not None or target_text_list is not None:
        validate_text_bounds(
            normalized_predictions, source_text_list, target_text_list, "predictions"
        )
        validate_text_bounds(
            normalized_references, source_text_list, target_text_list, "references"
        )

    segment_components = [
        compute_score_components(
            segment_predictions,
            segment_references,
            measure=measure,
            matching=matching,
            matching_algorithm=matching_algorithm,
            severity_penalty=normalized_severity_penalty,
            severity_weights=normalized_severity_weights,
        )
        for segment_predictions, segment_references in zip(
            normalized_predictions, normalized_references
        )
    ]

    config = MetricConfig(
        measure=measure,
        matching=matching,
        matching_algorithm=matching_algorithm if matching == "one_to_one" else None,
        averaging=averaging,
        severity_penalty=normalized_severity_penalty,
        severity_weights=normalized_severity_weights,
    )

    if all(isinstance(components, TPCounts) for components in segment_components):
        segment_counts = cast(list[TPCounts], segment_components)
        total_components = sum_counts(segment_counts)
        details = CountDetails(total_components, segment_counts)
    else:
        side_segment_components = cast(list[SideScoreComponents], segment_components)
        total_components = sum_side_score_components(side_segment_components)
        details = SideScoreDetails(total_components, side_segment_components)

    if averaging == "micro":
        precision, recall, f_score = precision_recall_fscore(total_components)
    else:
        per_segment_scores = [
            precision_recall_fscore(components) for components in segment_components
        ]
        precision = sum(score[0] for score in per_segment_scores) / len(
            per_segment_scores
        )
        recall = sum(score[1] for score in per_segment_scores) / len(per_segment_scores)
        f_score = sum(score[2] for score in per_segment_scores) / len(
            per_segment_scores
        )

    return MetricResult(
        precision=precision,
        recall=recall,
        f_score=f_score,
        details=details,
        config=config,
    )
