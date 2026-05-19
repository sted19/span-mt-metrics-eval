"""Pair-level span predicates and scoring functions."""

from __future__ import annotations

from collections.abc import Callable

from span_mt_metrics_eval.options import Measure
from span_mt_metrics_eval.spans import ErrorSpan


MatchPairs = list[tuple[int, int]]


def overlap_length(span1: ErrorSpan, span2: ErrorSpan) -> int:
    """Return the half-open character overlap length between two spans.

    The inputs are two validated ``ErrorSpan`` objects. The result is zero when
    the spans are on different sides or when their intervals do not overlap.
    """

    if span1.side != span2.side:
        return 0
    return max(0, min(span1.end, span2.end) - max(span1.start, span2.start))


def spans_exactly_match(span1: ErrorSpan, span2: ErrorSpan) -> bool:
    """Return whether two spans have the same side and offsets.

    This predicate is the pair-level objective for the ``EM`` metric and is also
    used when turning selected matches into exact-match counts.
    """

    return (
        span1.side == span2.side
        and span1.start == span2.start
        and span1.end == span2.end
    )


def harmonic_overlap_score(span1: ErrorSpan, span2: ErrorSpan) -> float:
    """Return the harmonic overlap ratio used by greedy and ``MPP`` matching.

    The score is ``2 * overlap / (len(prediction) + len(reference))`` and ranges
    from 0 to 1. It returns zero for spans on different sides.
    """

    overlap = overlap_length(span1, span2)
    if overlap == 0:
        return 0.0
    return 2 * overlap / (span1.length + span2.length)


def severity_reward(span1: ErrorSpan, span2: ErrorSpan, severity_penalty: float) -> float:
    """Return the reward multiplier for a pair's severity labels.

    With no penalty, severity never changes matching or counting. Otherwise,
    equal normalized labels receive full credit and mismatches receive
    ``1 - severity_penalty`` credit.
    """

    if severity_penalty == 0.0:
        return 1.0
    return 1.0 if span1.severity == span2.severity else 1.0 - severity_penalty


def metric_objective(
    measure: Measure,
    severity_penalty: float = 0.0,
) -> Callable[[ErrorSpan, ErrorSpan], float]:
    """Return the pair-scoring function for optimal one-to-one matching.

    ``measure`` selects the metric-specific objective. The returned callable
    accepts a predicted span and reference span and returns a non-negative score
    that the assignment solver maximizes.
    """

    if measure == "EM":
        return lambda pred, ref: (
            (1.0 if spans_exactly_match(pred, ref) else 0.0)
            * severity_reward(pred, ref, severity_penalty)
        )
    if measure == "MP":
        return lambda pred, ref: (
            (1.0 if overlap_length(pred, ref) > 0 else 0.0)
            * severity_reward(pred, ref, severity_penalty)
        )
    if measure == "WMT23":
        return lambda pred, ref: (
            float(overlap_length(pred, ref))
            * severity_reward(pred, ref, severity_penalty)
        )
    raise ValueError(f"Unknown measure: {measure}")
