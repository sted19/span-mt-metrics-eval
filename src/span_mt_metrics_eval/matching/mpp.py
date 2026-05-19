"""MPP-specific matching objective helpers."""

from __future__ import annotations

from collections.abc import Mapping

from span_mt_metrics_eval.matching.pairs import MatchPairs, overlap_length, severity_reward
from span_mt_metrics_eval.spans import ErrorSpan


FLOAT_TOLERANCE = 1e-12


def mpp_matching_key(
    matches: MatchPairs,
    predictions: list[ErrorSpan],
    references: list[ErrorSpan],
    severity_penalty: float = 0.0,
    severity_weights: Mapping[str, float] | None = None,
) -> tuple[float, float, float, float, float]:
    """Return the objective key for an MPP matching.

    The input is a set of selected one-to-one pairs and the spans for one
    segment. The returned tuple starts with the final F-score and then includes
    deterministic tie-break values.
    """

    precision_credit_sum = 0.0
    recall_credit_sum = 0.0
    for pred_idx, ref_idx in matches:
        pred = predictions[pred_idx]
        ref = references[ref_idx]
        overlap = overlap_length(pred, ref)
        reward = mpp_pair_credit_mass(pred, ref, severity_penalty, severity_weights)
        precision_credit_sum += (overlap / pred.length) * reward
        recall_credit_sum += (overlap / ref.length) * reward

    return mpp_matching_key_from_credits(
        precision_credit_sum,
        recall_credit_sum,
        mpp_denominator(predictions, severity_weights),
        mpp_denominator(references, severity_weights),
        len(matches),
    )


def mpp_matching_key_from_credits(
    precision_credit_sum: float,
    recall_credit_sum: float,
    prediction_count: float,
    reference_count: float,
    match_count: int,
) -> tuple[float, float, float, float, float]:
    """Return comparison values for an MPP matching.

    The first element is the paper objective. Later elements only make ties
    deterministic when multiple matchings reach the same F-score.
    """

    precision = precision_credit_sum / prediction_count if prediction_count else 1.0
    recall = recall_credit_sum / reference_count if reference_count else 1.0
    f_score = harmonic_mean(precision, recall)
    return (f_score, precision + recall, precision, recall, float(match_count))


def mpp_f_score_from_credits(
    precision_credit_sum: float,
    recall_credit_sum: float,
    prediction_count: float,
    reference_count: float,
) -> float:
    """Compute MPP F-score from aggregate precision/recall credits."""

    precision = precision_credit_sum / prediction_count if prediction_count else 1.0
    recall = recall_credit_sum / reference_count if reference_count else 1.0
    return harmonic_mean(precision, recall)


def harmonic_mean(precision: float, recall: float) -> float:
    """Return the harmonic mean using the paper's zero-denominator convention."""

    return (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )


def mpp_key_is_better(
    candidate_key: tuple[float, float, float, float, float],
    best_key: tuple[float, float, float, float, float],
    candidate_matches: MatchPairs,
    best_matches: MatchPairs,
) -> bool:
    """Return whether one MPP objective key should replace another.

    The inputs are comparison tuples plus their corresponding match lists. The
    output is ``True`` when the candidate is better or ties with a lexicographic
    smaller match list.
    """

    for candidate_value, best_value in zip(candidate_key, best_key):
        if candidate_value > best_value + FLOAT_TOLERANCE:
            return True
        if candidate_value < best_value - FLOAT_TOLERANCE:
            return False
    return sorted(candidate_matches) < sorted(best_matches)


def mpp_pair_credit_mass(
    pred: ErrorSpan,
    ref: ErrorSpan,
    severity_penalty: float,
    severity_weights: Mapping[str, float] | None,
) -> float:
    """Return the pair-level MPP credit multiplier or severity mass."""

    if severity_weights is None:
        return severity_reward(pred, ref, severity_penalty)
    return min(
        severity_weights[pred.severity or ""],
        severity_weights[ref.severity or ""],
    )


def mpp_denominator(
    spans: list[ErrorSpan],
    severity_weights: Mapping[str, float] | None,
) -> float:
    """Return the MPP denominator for one side of a segment."""

    if severity_weights is None:
        return float(len(spans))
    return sum(severity_weights[span.severity or ""] for span in spans)
