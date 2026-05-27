"""One-to-one metric score computation."""

from __future__ import annotations

from span_mt_metrics_eval.matching import (
    MatchPairs,
    find_greedy_matches,
    find_optimal_matches,
    overlap_length,
    severity_reward,
    spans_exactly_match,
)
from span_mt_metrics_eval.options import MatchingAlgorithm, Measure
from span_mt_metrics_eval.results import ScoreComponents, SideScoreComponents, TPCounts
from span_mt_metrics_eval.scoring.coverage import (
    span_severity_mass_total,
    span_severity_weight,
)
from span_mt_metrics_eval.spans import ErrorSpan


def compute_o2o_score_components(
    predictions: list[ErrorSpan],
    references: list[ErrorSpan],
    measure: Measure,
    matching_algorithm: MatchingAlgorithm,
    severity_penalty: float,
    severity_weights: dict[str, float] | None,
) -> ScoreComponents:
    """Compute one-to-one score components after choosing matched span pairs.

    ``matching_algorithm`` controls how prediction/reference pairs are selected.
    Once matching is done, count-style measures return ``TPCounts`` and MPP
    returns side-specific score components.
    """

    if matching_algorithm == "greedy":
        matches = find_greedy_matches(
            predictions, references, severity_penalty, severity_weights
        )
    else:
        matches = find_optimal_matches(
            predictions, references, measure, severity_penalty, severity_weights
        )

    if measure == "EM":
        return compute_o2o_em_counts(predictions, references, matches, severity_penalty)
    if measure == "MP":
        return compute_o2o_mp_counts(predictions, references, matches, severity_penalty)
    if measure == "WMT25":
        return compute_o2o_wmt25_counts(
            predictions, references, matches, severity_penalty
        )
    if measure == "MPP":
        return compute_o2o_mpp_components(
            predictions, references, matches, severity_penalty, severity_weights
        )
    raise ValueError(f"Unknown measure: {measure}")


def compute_o2o_em_counts(
    predictions: list[ErrorSpan],
    references: list[ErrorSpan],
    matches: MatchPairs,
    severity_penalty: float,
) -> TPCounts:
    """Compute exact-match counts for a one-to-one matching.

    Inputs are the spans and already-selected match pairs. Exact offset matches
    receive full or severity-discounted credit; all remaining mass is counted as
    false positives or false negatives.
    """

    exact_match_credit = 0.0
    for pred_idx, ref_idx in matches:
        pred = predictions[pred_idx]
        ref = references[ref_idx]
        if spans_exactly_match(pred, ref):
            exact_match_credit += severity_reward(pred, ref, severity_penalty)

    return TPCounts(
        tp=exact_match_credit,
        fp=float(len(predictions)) - exact_match_credit,
        fn=float(len(references)) - exact_match_credit,
    )


def compute_o2o_mp_counts(
    predictions: list[ErrorSpan],
    references: list[ErrorSpan],
    matches: MatchPairs,
    severity_penalty: float,
) -> TPCounts:
    """Compute binary partial-overlap counts for a one-to-one matching.

    Every selected pair has positive overlap and receives full or
    severity-discounted binary credit on both precision and recall sides.
    """

    true_positive_credit = sum(
        severity_reward(predictions[pred_idx], references[ref_idx], severity_penalty)
        for pred_idx, ref_idx in matches
    )
    return TPCounts(
        tp=true_positive_credit,
        fp=float(len(predictions)) - true_positive_credit,
        fn=float(len(references)) - true_positive_credit,
    )


def compute_o2o_wmt25_counts(
    predictions: list[ErrorSpan],
    references: list[ErrorSpan],
    matches: MatchPairs,
    severity_penalty: float,
) -> TPCounts:
    """Compute character-count ``WMT25`` counts for a one-to-one matching.

    Overlapping characters in matched pairs are true positives after applying
    the severity reward. Unrewarded prediction/reference characters become
    false positives and false negatives.
    """

    matched_by_prediction = {pred_idx: ref_idx for pred_idx, ref_idx in matches}
    matched_by_reference = {ref_idx: pred_idx for pred_idx, ref_idx in matches}

    true_positive_chars = 0.0
    false_positive_chars = 0.0
    false_negative_chars = 0.0

    for pred_idx, pred in enumerate(predictions):
        ref_idx = matched_by_prediction.get(pred_idx)
        if ref_idx is None:
            false_positive_chars += pred.length
            continue
        ref = references[ref_idx]
        overlap_reward = overlap_length(pred, ref) * severity_reward(
            pred, ref, severity_penalty
        )
        true_positive_chars += overlap_reward
        false_positive_chars += pred.length - overlap_reward

    for ref_idx, ref in enumerate(references):
        pred_idx = matched_by_reference.get(ref_idx)
        if pred_idx is None:
            false_negative_chars += ref.length
            continue
        pred = predictions[pred_idx]
        overlap_reward = overlap_length(pred, ref) * severity_reward(
            pred, ref, severity_penalty
        )
        false_negative_chars += ref.length - overlap_reward

    return TPCounts(
        tp=true_positive_chars,
        fp=false_positive_chars,
        fn=false_negative_chars,
    )


def compute_o2o_mpp_components(
    predictions: list[ErrorSpan],
    references: list[ErrorSpan],
    matches: MatchPairs,
    severity_penalty: float,
    severity_weights: dict[str, float] | None,
) -> SideScoreComponents:
    """Compute side-specific ``MPP`` components for a one-to-one matching.

    MPP averages prediction-span precision credits and reference-span recall
    credits separately. The returned object stores those two score fractions
    directly instead of pretending they are TP/FP/FN counts.
    """

    if severity_weights is not None:
        return compute_weighted_o2o_mpp_components(
            predictions, references, matches, severity_weights
        )

    matched_by_prediction = {pred_idx: ref_idx for pred_idx, ref_idx in matches}
    matched_by_reference = {ref_idx: pred_idx for pred_idx, ref_idx in matches}

    precision_numerator = 0.0
    for pred_idx, pred in enumerate(predictions):
        ref_idx = matched_by_prediction.get(pred_idx)
        if ref_idx is None:
            continue
        ref = references[ref_idx]
        precision_numerator += (overlap_length(pred, ref) / pred.length) * (
            severity_reward(pred, ref, severity_penalty)
        )

    recall_numerator = 0.0
    for ref_idx, ref in enumerate(references):
        pred_idx = matched_by_reference.get(ref_idx)
        if pred_idx is None:
            continue
        pred = predictions[pred_idx]
        recall_numerator += (overlap_length(pred, ref) / ref.length) * (
            severity_reward(pred, ref, severity_penalty)
        )

    return SideScoreComponents(
        precision_numerator=precision_numerator,
        precision_denominator=float(len(predictions)),
        recall_numerator=recall_numerator,
        recall_denominator=float(len(references)),
    )


def compute_weighted_o2o_mpp_components(
    predictions: list[ErrorSpan],
    references: list[ErrorSpan],
    matches: MatchPairs,
    severity_weights: dict[str, float],
) -> SideScoreComponents:
    """Compute severity-weighted one-to-one MPP score components.

    Precision and recall denominators are the total severity mass on their own
    side. A matched pair can receive at most the lower of the two severity
    weights, so severity mismatches are discounted by construction.
    """

    matched_by_prediction = {pred_idx: ref_idx for pred_idx, ref_idx in matches}
    matched_by_reference = {ref_idx: pred_idx for pred_idx, ref_idx in matches}

    precision_numerator = 0.0
    for pred_idx, pred in enumerate(predictions):
        ref_idx = matched_by_prediction.get(pred_idx)
        if ref_idx is None:
            continue
        ref = references[ref_idx]
        matched_weight = min(
            span_severity_weight(pred, severity_weights),
            span_severity_weight(ref, severity_weights),
        )
        precision_numerator += (overlap_length(pred, ref) / pred.length) * (
            matched_weight
        )

    recall_numerator = 0.0
    for ref_idx, ref in enumerate(references):
        pred_idx = matched_by_reference.get(ref_idx)
        if pred_idx is None:
            continue
        pred = predictions[pred_idx]
        matched_weight = min(
            span_severity_weight(pred, severity_weights),
            span_severity_weight(ref, severity_weights),
        )
        recall_numerator += (overlap_length(pred, ref) / ref.length) * matched_weight

    return SideScoreComponents(
        precision_numerator=precision_numerator,
        precision_denominator=span_severity_mass_total(predictions, severity_weights),
        recall_numerator=recall_numerator,
        recall_denominator=span_severity_mass_total(references, severity_weights),
    )
