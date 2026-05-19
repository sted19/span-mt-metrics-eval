"""Score aggregation helpers for precision, recall, and F-score."""

from __future__ import annotations

from collections.abc import Iterable

from span_mt_metrics_eval.results import ScoreComponents, SideScoreComponents, TPCounts


def precision_recall_fscore(components: ScoreComponents) -> tuple[float, float, float]:
    """Convert raw score components into precision, recall, and F-score.

    The input can be a count-style ``TPCounts`` object or a side-specific
    ``SideScoreComponents`` object. The output uses the package zero-division
    convention of returning 1.0 for precision or recall when that denominator is
    empty.
    """

    if isinstance(components, TPCounts):
        return precision_recall_fscore_from_counts(components)
    return precision_recall_fscore_from_side_score(components)


def precision_recall_fscore_from_counts(
    counts: TPCounts,
) -> tuple[float, float, float]:
    """Convert count-style components into precision, recall, and F-score."""

    precision_denominator = counts.tp + counts.fp
    recall_denominator = counts.tp + counts.fn

    precision = counts.tp / precision_denominator if precision_denominator > 0 else 1.0
    recall = counts.tp / recall_denominator if recall_denominator > 0 else 1.0
    return precision, recall, harmonic_mean(precision, recall)


def precision_recall_fscore_from_side_score(
    components: SideScoreComponents,
) -> tuple[float, float, float]:
    """Convert side-specific components into precision, recall, and F-score."""

    precision = (
        components.precision_numerator / components.precision_denominator
        if components.precision_denominator > 0
        else 1.0
    )
    recall = (
        components.recall_numerator / components.recall_denominator
        if components.recall_denominator > 0
        else 1.0
    )
    return precision, recall, harmonic_mean(precision, recall)


def harmonic_mean(precision: float, recall: float) -> float:
    """Return the harmonic mean of precision and recall."""

    return (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )


def sum_counts(list_of_counts: Iterable[TPCounts]) -> TPCounts:
    """Aggregate a sequence of count-style component objects."""

    total_counts = TPCounts()
    for counts in list_of_counts:
        total_counts += counts
    return total_counts


def sum_side_score_components(
    list_of_components: Iterable[SideScoreComponents],
) -> SideScoreComponents:
    """Aggregate a sequence of side-specific component objects."""

    total_components = SideScoreComponents()
    for components in list_of_components:
        total_components += components
    return total_components
