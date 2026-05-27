"""Many-to-many metric score computation."""

from __future__ import annotations

from collections import Counter

from span_mt_metrics_eval.options import Measure
from span_mt_metrics_eval.results import ScoreComponents, SideScoreComponents, TPCounts
from span_mt_metrics_eval.scoring.coverage import (
    exact_span_severity_counters,
    m2m_binary_overlap_reward,
    m2m_decompositions_by_side,
    span_severity_mass_total,
    span_severity_weighted_m2m_credits,
    span_weighted_m2m_credits,
)
from span_mt_metrics_eval.spans import ErrorSpan


def compute_m2m_score_components(
    predictions: list[ErrorSpan],
    references: list[ErrorSpan],
    measure: Measure,
    severity_penalty: float,
    severity_weights: dict[str, float] | None,
) -> ScoreComponents:
    """Compute one segment's many-to-many components for the selected measure.

    Many-to-many scoring does not choose explicit pairs. Instead, each helper
    compares all predicted and reference span coverage for the segment and
    returns either count-style or side-specific score components.
    """

    if measure == "EM":
        return compute_m2m_em_counts(predictions, references, severity_penalty)
    if measure == "MP":
        return compute_m2m_mp_components(predictions, references, severity_penalty)
    if measure == "WMT25":
        return compute_m2m_wmt25_counts(predictions, references, severity_penalty)
    if measure == "MPP":
        return compute_m2m_mpp_components(
            predictions, references, severity_penalty, severity_weights
        )
    raise ValueError(f"Unknown measure: {measure}")


def compute_m2m_em_counts(
    predictions: list[ErrorSpan],
    references: list[ErrorSpan],
    severity_penalty: float,
) -> TPCounts:
    """Compute many-to-many exact-match counts as multiset overlap.

    Exact offset matches are resolved by severity first. Remaining exact-offset
    pairs with different severities receive discounted credit.
    """

    pred_groups = exact_span_severity_counters(predictions)
    ref_groups = exact_span_severity_counters(references)

    true_positive_credit = 0.0
    for key in set(pred_groups).union(ref_groups):
        pred_counter = pred_groups.get(key, Counter())
        ref_counter = ref_groups.get(key, Counter())
        severities = set(pred_counter).union(ref_counter)
        exact_credit = sum(
            min(pred_counter[severity], ref_counter[severity])
            for severity in severities
        )
        pred_remaining = sum(pred_counter.values()) - exact_credit
        ref_remaining = sum(ref_counter.values()) - exact_credit
        mismatch_credit = min(pred_remaining, ref_remaining)
        true_positive_credit += exact_credit + (
            1.0 - severity_penalty
        ) * mismatch_credit

    return TPCounts(
        tp=true_positive_credit,
        fp=float(len(predictions)) - true_positive_credit,
        fn=float(len(references)) - true_positive_credit,
    )


def compute_m2m_mp_components(
    predictions: list[ErrorSpan],
    references: list[ErrorSpan],
    severity_penalty: float,
) -> SideScoreComponents:
    """Compute side-specific binary partial-overlap score components.

    In many-to-many MP, prediction spans and reference spans are credited from
    their own side. Duplicate coverage can therefore create different precision
    and recall numerators, so this measure is not represented as TP/FP/FN.
    """

    decompositions, severity_to_idx = m2m_decompositions_by_side(
        predictions, references
    )

    precision_numerator = 0.0
    for pred in predictions:
        _, ref_counts, *_ = decompositions[pred.side]
        reward = m2m_binary_overlap_reward(
            pred, ref_counts, severity_to_idx, severity_penalty
        )
        if reward is not None:
            precision_numerator += reward

    recall_numerator = 0.0
    for ref in references:
        pred_counts, _, *_ = decompositions[ref.side]
        reward = m2m_binary_overlap_reward(
            ref, pred_counts, severity_to_idx, severity_penalty
        )
        if reward is not None:
            recall_numerator += reward

    return SideScoreComponents(
        precision_numerator=precision_numerator,
        precision_denominator=float(len(predictions)),
        recall_numerator=recall_numerator,
        recall_denominator=float(len(references)),
    )


def compute_m2m_wmt25_counts(
    predictions: list[ErrorSpan],
    references: list[ErrorSpan],
    severity_penalty: float,
) -> TPCounts:
    """Compute many-to-many character-count ``WMT25`` counts.

    Per-character coverage is decomposed into same-severity matches,
    different-severity matches, and unmatched mass, mirroring the paper repo.
    """

    true_positive_chars = 0.0
    false_positive_chars = 0.0
    false_negative_chars = 0.0

    decompositions, _ = m2m_decompositions_by_side(predictions, references)
    for (
        _pred_counts,
        _ref_counts,
        pred_exact,
        pred_mismatch,
        pred_unmatched,
        _ref_exact,
        ref_mismatch,
        ref_unmatched,
    ) in decompositions.values():
        true_positive_chars += float(
            (pred_exact + (1.0 - severity_penalty) * pred_mismatch).sum()
        )
        false_positive_chars += float(
            (severity_penalty * pred_mismatch + pred_unmatched).sum()
        )
        false_negative_chars += float(
            (severity_penalty * ref_mismatch + ref_unmatched).sum()
        )

    return TPCounts(
        tp=true_positive_chars,
        fp=false_positive_chars,
        fn=false_negative_chars,
    )


def compute_m2m_mpp_components(
    predictions: list[ErrorSpan],
    references: list[ErrorSpan],
    severity_penalty: float,
    severity_weights: dict[str, float] | None,
) -> SideScoreComponents:
    """Compute many-to-many partial-credit ``MPP`` score components.

    Each span receives average per-character true-positive credit for its own
    severity layer. MPP then averages prediction-side and reference-side span
    credits separately.
    """

    if severity_weights is not None:
        return compute_weighted_m2m_mpp_components(
            predictions, references, severity_weights
        )

    decompositions, severity_to_idx = m2m_decompositions_by_side(
        predictions, references
    )

    precision_numerator = 0.0
    for pred in predictions:
        (
            pred_counts,
            _ref_counts,
            pred_exact,
            pred_mismatch,
            pred_unmatched,
            _ref_exact,
            _ref_mismatch,
            _ref_unmatched,
        ) = decompositions[pred.side]
        tp_credit, _error_credit = span_weighted_m2m_credits(
            pred,
            severity_to_idx,
            pred_counts,
            pred_exact,
            pred_mismatch,
            pred_unmatched,
            severity_penalty,
        )
        precision_numerator += tp_credit

    recall_numerator = 0.0
    for ref in references:
        (
            _pred_counts,
            ref_counts,
            _pred_exact,
            _pred_mismatch,
            _pred_unmatched,
            ref_exact,
            ref_mismatch,
            ref_unmatched,
        ) = decompositions[ref.side]
        tp_credit, _error_credit = span_weighted_m2m_credits(
            ref,
            severity_to_idx,
            ref_counts,
            ref_exact,
            ref_mismatch,
            ref_unmatched,
            severity_penalty,
        )
        recall_numerator += tp_credit

    return SideScoreComponents(
        precision_numerator=precision_numerator,
        precision_denominator=float(len(predictions)),
        recall_numerator=recall_numerator,
        recall_denominator=float(len(references)),
    )


def compute_weighted_m2m_mpp_components(
    predictions: list[ErrorSpan],
    references: list[ErrorSpan],
    severity_weights: dict[str, float],
) -> SideScoreComponents:
    """Compute severity-weighted many-to-many MPP score components.

    Weighted coverage is decomposed per character and severity. Same-severity
    mass matches first, and remaining cross-severity mass matches up to the
    smaller side, which mirrors the one-to-one ``min(weight1, weight2)`` rule.
    """

    decompositions, severity_to_idx = m2m_decompositions_by_side(
        predictions, references, severity_weights
    )

    precision_numerator = 0.0
    for pred in predictions:
        (
            pred_counts,
            _ref_counts,
            pred_exact,
            pred_mismatch,
            pred_unmatched,
            _ref_exact,
            _ref_mismatch,
            _ref_unmatched,
        ) = decompositions[pred.side]
        tp_credit, _error_credit = span_severity_weighted_m2m_credits(
            pred,
            severity_to_idx,
            pred_counts,
            pred_exact,
            pred_mismatch,
            pred_unmatched,
            severity_weights,
        )
        precision_numerator += tp_credit

    recall_numerator = 0.0
    for ref in references:
        (
            _pred_counts,
            ref_counts,
            _pred_exact,
            _pred_mismatch,
            _pred_unmatched,
            ref_exact,
            ref_mismatch,
            ref_unmatched,
        ) = decompositions[ref.side]
        tp_credit, _error_credit = span_severity_weighted_m2m_credits(
            ref,
            severity_to_idx,
            ref_counts,
            ref_exact,
            ref_mismatch,
            ref_unmatched,
            severity_weights,
        )
        recall_numerator += tp_credit

    return SideScoreComponents(
        precision_numerator=precision_numerator,
        precision_denominator=span_severity_mass_total(predictions, severity_weights),
        recall_numerator=recall_numerator,
        recall_denominator=span_severity_mass_total(references, severity_weights),
    )
