"""Span-level precision, recall, and F-score metrics.

This module contains the public ``compute`` entrypoint plus the metric-specific
counting helpers. The implementation is organized in three layers:

1. normalize user input into per-segment ``ErrorSpan`` lists;
2. compute raw true-positive/false-positive/false-negative counts per segment;
3. aggregate those counts with either micro or macro averaging.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from numbers import Real
from typing import Any
import logging

import numpy as np

from span_mt_metrics_eval.matching import (
    MatchPairs,
    find_greedy_matches,
    find_optimal_matches,
    overlap_length,
    severity_reward,
    spans_exactly_match,
)
from span_mt_metrics_eval.types import (
    AVERAGING_STRATEGIES,
    MATCHING_ALGORITHMS,
    MATCHING_STRATEGIES,
    MEASURES,
    AveragingStrategy,
    TPCounts,
    ErrorSpan,
    MatchingAlgorithm,
    MatchingStrategy,
    Measure,
    MetricConfig,
    MetricResult,
)

logger = logging.getLogger(__name__)

def compute(
    predictions: list[ErrorSpan] | list[dict] | list[list[ErrorSpan]] | list[list[dict]],
    references: list[ErrorSpan] | list[dict] | list[list[ErrorSpan]] | list[list[dict]],
    *,
    measure: Measure | str = "MPP",
    matching: MatchingStrategy = "one_to_one",
    matching_algorithm: MatchingAlgorithm = "optimal",
    averaging: AveragingStrategy = "micro",
    severity_penalty: float = 0.0,
    source_texts: str | list[str] | None = None,
    target_texts: str | list[str] | None = None,
) -> MetricResult | dict[Measure, MetricResult]:
    """Compute span-level precision, recall, and F-score.

    ``predictions`` and ``references`` can be a single segment's spans or a
    sequence of per-segment span sequences. Each span can be an ``ErrorSpan`` or a
    dict with ``start``, ``end``, and ``side`` keys.

    The metric is selected with ``measure``. Passing ``measure="all"`` returns a
    dictionary with one ``MetricResult`` per metric; otherwise this function
    returns a single ``MetricResult``. ``severity_penalty`` discounts matches
    whose normalized severity labels differ. Optional text inputs are currently
    kept in the signature for users who want to pass source/target context
    alongside offsets.
    """

    _validate_measure(measure)
    _validate_matching(matching)
    _validate_matching_algorithm(matching_algorithm)
    _validate_averaging(averaging)
    severity_penalty = _validate_severity_penalty(severity_penalty)

    if measure == "all":
        result = {
            item: compute(
                predictions,
                references,
                measure=item,
                matching=matching,
                matching_algorithm=matching_algorithm,
                averaging=averaging,
                severity_penalty=severity_penalty,
                source_texts=source_texts,
                target_texts=target_texts,
            )
            for item in MEASURES
        }
        return result
    else:
        measure : Measure

    predictions = _normalize_input(predictions, "predictions")
    references = _normalize_input(references, "references")

    if len(predictions) != len(references):
        raise ValueError(
            "predictions and references must contain the same number of elements"
            f"({len(predictions)} != {len(references)})"
        )

    if severity_penalty > 0.0:
        _validate_required_severities(predictions, "predictions")
        _validate_required_severities(references, "references")

    source_text_list = (
        _coerce_texts(source_texts, len(predictions), "source_texts")
        if source_texts is not None
        else None
    )
    target_text_list = (
        _coerce_texts(target_texts, len(predictions), "target_texts")
        if target_texts is not None
        else None
    )
    if source_text_list is not None or target_text_list is not None:
        _validate_text_bounds(
            predictions, source_text_list, target_text_list, "predictions"
        )
        _validate_text_bounds(
            references, source_text_list, target_text_list, "references"
        )

    segments_tp_counts = [
        _compute_tp_counts(
            segment_predictions,
            segment_references,
            measure=measure,
            matching=matching,
            matching_algorithm=matching_algorithm,
            severity_penalty=severity_penalty,
        )
        for segment_predictions, segment_references in zip(predictions, references)
    ]

    total_tp_counts = _sum_counts(segments_tp_counts)
    config = MetricConfig(
        measure=measure,
        matching=matching,
        matching_algorithm=(
            matching_algorithm if matching == "one_to_one" else None
        ),
        averaging=averaging,
        severity_penalty=severity_penalty,
    )

    if averaging == "micro":
        precision, recall, f_score = _precision_recall_fscore(total_tp_counts)
    else:
        per_segment_scores = [
            _precision_recall_fscore(tp_counts) for tp_counts in segments_tp_counts
        ]
        precision = sum(score[0] for score in per_segment_scores) / len(per_segment_scores)
        recall = sum(score[1] for score in per_segment_scores) / len(per_segment_scores)
        f_score = sum(score[2] for score in per_segment_scores) / len(per_segment_scores)

    return MetricResult(
        precision=precision,
        recall=recall,
        f_score=f_score,
        tp_counts=total_tp_counts,
        segments_tp_counts=segments_tp_counts,
        config=config,
    )


def _compute_tp_counts(
    predictions: list[ErrorSpan],
    references: list[ErrorSpan],
    *,
    measure: Measure,
    matching: MatchingStrategy,
    matching_algorithm: MatchingAlgorithm,
    severity_penalty: float,
) -> TPCounts:
    """Compute one segment's raw counts for the requested matching strategy.

    Inputs are normalized prediction and reference spans for a single segment,
    plus the selected metric options. The output is a ``TPCounts`` object ready
    for aggregation.
    """

    if matching == "one_to_one":
        return _compute_o2o_tp_counts(
            predictions, references, measure, matching_algorithm, severity_penalty
        )
    return _compute_m2m_tp_counts(predictions, references, measure, severity_penalty)


def _compute_o2o_tp_counts(
    predictions: list[ErrorSpan],
    references: list[ErrorSpan],
    measure: Measure,
    matching_algorithm: MatchingAlgorithm,
    severity_penalty: float,
) -> TPCounts:
    """Compute one-to-one counts after choosing matched span pairs.

    ``matching_algorithm`` controls how prediction/reference pairs are selected.
    Once matching is done, the selected measure determines how each pair
    contributes to true-positive and error counts.
    """

    if matching_algorithm == "greedy":
        matches = find_greedy_matches(predictions, references, severity_penalty)
    else:
        matches = find_optimal_matches(
            predictions, references, measure, severity_penalty
        )

    if measure == "EM":
        return _compute_o2o_em_tp_counts(
            predictions, references, matches, severity_penalty
        )
    if measure == "MP":
        return _compute_o2o_mp_tp_counts(
            predictions, references, matches, severity_penalty
        )
    if measure == "WMT23":
        return _compute_o2o_wmt23_tp_counts(
            predictions, references, matches, severity_penalty
        )
    if measure == "MPP":
        return _compute_o2o_mpp_tp_counts(
            predictions, references, matches, severity_penalty
        )
    raise ValueError(f"Unknown measure: {measure}")


def _compute_o2o_em_tp_counts(
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
        tp_for_precision=exact_match_credit,
        tp_for_recall=exact_match_credit,
        fp=float(len(predictions)) - exact_match_credit,
        fn=float(len(references)) - exact_match_credit,
    )


def _compute_o2o_mp_tp_counts(
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
        tp_for_precision=true_positive_credit,
        tp_for_recall=true_positive_credit,
        fp=float(len(predictions)) - true_positive_credit,
        fn=float(len(references)) - true_positive_credit,
    )


def _compute_o2o_wmt23_tp_counts(
    predictions: list[ErrorSpan],
    references: list[ErrorSpan],
    matches: MatchPairs,
    severity_penalty: float,
) -> TPCounts:
    """Compute character-count ``WMT23`` counts for a one-to-one matching.

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
        tp_for_precision=true_positive_chars,
        tp_for_recall=true_positive_chars,
        fp=false_positive_chars,
        fn=false_negative_chars,
    )


def _compute_o2o_mpp_tp_counts(
    predictions: list[ErrorSpan],
    references: list[ErrorSpan],
    matches: MatchPairs,
    severity_penalty: float,
) -> TPCounts:
    """Compute partial-credit ``MPP`` counts for a one-to-one matching.

    Matched predictions and references receive overlap-fraction credit scaled by
    the severity reward for the selected pair.
    """

    matched_by_prediction = {pred_idx: ref_idx for pred_idx, ref_idx in matches}
    matched_by_reference = {ref_idx: pred_idx for pred_idx, ref_idx in matches}

    tp_for_precision = 0.0
    fp = 0.0
    for pred_idx, pred in enumerate(predictions):
        ref_idx = matched_by_prediction.get(pred_idx)
        if ref_idx is None:
            fp += 1.0
            continue
        ref = references[ref_idx]
        credit = (overlap_length(pred, ref) / pred.length) * severity_reward(
            pred, ref, severity_penalty
        )
        tp_for_precision += credit
        fp += 1.0 - credit

    tp_for_recall = 0.0
    fn = 0.0
    for ref_idx, ref in enumerate(references):
        pred_idx = matched_by_reference.get(ref_idx)
        if pred_idx is None:
            fn += 1.0
            continue
        pred = predictions[pred_idx]
        credit = (overlap_length(pred, ref) / ref.length) * severity_reward(
            pred, ref, severity_penalty
        )
        tp_for_recall += credit
        fn += 1.0 - credit

    return TPCounts(
        tp_for_precision=tp_for_precision,
        tp_for_recall=tp_for_recall,
        fp=fp,
        fn=fn,
    )


def _compute_m2m_tp_counts(
    predictions: list[ErrorSpan],
    references: list[ErrorSpan],
    measure: Measure,
    severity_penalty: float,
) -> TPCounts:
    """Compute one segment's many-to-many counts for the selected measure.

    Many-to-many scoring does not choose explicit pairs. Instead, each helper
    compares all predicted and reference span coverage for the segment and
    returns raw counts for one metric.
    """

    if measure == "EM":
        return _compute_m2m_em_tp_counts(predictions, references, severity_penalty)
    if measure == "MP":
        return _compute_m2m_mp_tp_counts(predictions, references, severity_penalty)
    if measure == "WMT23":
        return _compute_m2m_wmt23_tp_counts(
            predictions, references, severity_penalty
        )
    if measure == "MPP":
        return _compute_m2m_mpp_tp_counts(predictions, references, severity_penalty)
    raise ValueError(f"Unknown measure: {measure}")


def _compute_m2m_em_tp_counts(
    predictions: list[ErrorSpan],
    references: list[ErrorSpan],
    severity_penalty: float,
) -> TPCounts:
    """Compute many-to-many exact-match counts as multiset overlap.

    Exact offset matches are resolved by severity first. Remaining exact-offset
    pairs with different severities receive discounted credit.
    """

    pred_groups = _exact_span_severity_counters(predictions)
    ref_groups = _exact_span_severity_counters(references)

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
        tp_for_precision=true_positive_credit,
        tp_for_recall=true_positive_credit,
        fp=float(len(predictions)) - true_positive_credit,
        fn=float(len(references)) - true_positive_credit,
    )


def _compute_m2m_mp_tp_counts(
    predictions: list[ErrorSpan],
    references: list[ErrorSpan],
    severity_penalty: float,
) -> TPCounts:
    """Compute many-to-many binary partial-overlap counts.

    A span receives full binary credit when it overlaps any span with the same
    severity, discounted credit when it only overlaps different severities, and
    no credit when it has no overlap.
    """

    counts_by_side, severity_to_idx = _char_count_arrays_by_severity(
        predictions, references
    )

    tp_for_precision = 0.0
    fp = 0.0
    for pred in predictions:
        _, ref_counts = counts_by_side[pred.side]
        reward = _m2m_binary_overlap_reward(
            pred, ref_counts, severity_to_idx, severity_penalty
        )
        if reward is None:
            fp += 1.0
        else:
            tp_for_precision += reward
            fp += 1.0 - reward

    tp_for_recall = 0.0
    fn = 0.0
    for ref in references:
        pred_counts, _ = counts_by_side[ref.side]
        reward = _m2m_binary_overlap_reward(
            ref, pred_counts, severity_to_idx, severity_penalty
        )
        if reward is None:
            fn += 1.0
        else:
            tp_for_recall += reward
            fn += 1.0 - reward

    return TPCounts(
        tp_for_precision=tp_for_precision,
        tp_for_recall=tp_for_recall,
        fp=fp,
        fn=fn,
    )


def _compute_m2m_wmt23_tp_counts(
    predictions: list[ErrorSpan],
    references: list[ErrorSpan],
    severity_penalty: float,
) -> TPCounts:
    """Compute many-to-many character-count ``WMT23`` counts.

    Per-character coverage is decomposed into same-severity matches,
    different-severity matches, and unmatched mass, mirroring the paper repo.
    """

    true_positive_chars = 0.0
    false_positive_chars = 0.0
    false_negative_chars = 0.0

    decompositions, _ = _m2m_decompositions_by_side(predictions, references)
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
        tp_for_precision=true_positive_chars,
        tp_for_recall=true_positive_chars,
        fp=false_positive_chars,
        fn=false_negative_chars,
    )


def _compute_m2m_mpp_tp_counts(
    predictions: list[ErrorSpan],
    references: list[ErrorSpan],
    severity_penalty: float,
) -> TPCounts:
    """Compute many-to-many partial-credit ``MPP`` counts.

    Each span receives the average per-character true-positive and error credit
    for its own severity layer.
    """

    decompositions, severity_to_idx = _m2m_decompositions_by_side(
        predictions, references
    )

    tp_for_precision = 0.0
    fp = 0.0
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
        tp_credit, fp_credit = _span_weighted_m2m_credits(
            pred,
            severity_to_idx,
            pred_counts,
            pred_exact,
            pred_mismatch,
            pred_unmatched,
            severity_penalty,
        )
        tp_for_precision += tp_credit
        fp += fp_credit

    tp_for_recall = 0.0
    fn = 0.0
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
        tp_credit, fn_credit = _span_weighted_m2m_credits(
            ref,
            severity_to_idx,
            ref_counts,
            ref_exact,
            ref_mismatch,
            ref_unmatched,
            severity_penalty,
        )
        tp_for_recall += tp_credit
        fn += fn_credit

    return TPCounts(
        tp_for_precision=tp_for_precision,
        tp_for_recall=tp_for_recall,
        fp=fp,
        fn=fn,
    )


def _exact_span_severity_counters(
    spans: list[ErrorSpan],
) -> dict[tuple[str, int, int], Counter[str | None]]:
    """Group exact span identities by severity label."""

    groups: dict[tuple[str, int, int], Counter[str | None]] = {}
    for span in spans:
        groups.setdefault(_span_key(span), Counter())[span.severity] += 1
    return groups


def _m2m_binary_overlap_reward(
    span: ErrorSpan,
    other_counts: np.ndarray,
    severity_to_idx: dict[str | None, int],
    severity_penalty: float,
) -> float | None:
    """Return binary many-to-many reward for one span, or ``None``."""

    other_slice = other_counts[:, span.start : span.end]
    if not bool(np.any(other_slice.sum(axis=0) > 0)):
        return None

    severity_idx = severity_to_idx[span.severity]
    same_severity_overlap = bool(np.any(other_slice[severity_idx] > 0))
    return 1.0 if same_severity_overlap else 1.0 - severity_penalty


def _m2m_decompositions_by_side(
    predictions: list[ErrorSpan],
    references: list[ErrorSpan],
) -> tuple[
    dict[str, tuple[np.ndarray, ...]],
    dict[str | None, int],
]:
    """Return per-side severity decompositions for many-to-many metrics."""

    counts_by_side, severity_to_idx = _char_count_arrays_by_severity(
        predictions, references
    )
    decompositions: dict[str, tuple[np.ndarray, ...]] = {}
    for side, (pred_counts, ref_counts) in counts_by_side.items():
        decompositions[side] = (
            pred_counts,
            ref_counts,
            *_decompose_counts(pred_counts, ref_counts),
        )
    return decompositions, severity_to_idx


def _span_weighted_m2m_credits(
    span: ErrorSpan,
    severity_to_idx: dict[str | None, int],
    counts: np.ndarray,
    exact: np.ndarray,
    mismatch: np.ndarray,
    unmatched: np.ndarray,
    severity_penalty: float,
) -> tuple[float, float]:
    """Return average true-positive and error credit for one span."""

    severity_idx = severity_to_idx[span.severity]
    denominator = counts[severity_idx, span.start : span.end]
    true_positive_numerator = (
        exact[severity_idx, span.start : span.end]
        + (1.0 - severity_penalty) * mismatch[severity_idx, span.start : span.end]
    )
    error_numerator = (
        severity_penalty * mismatch[severity_idx, span.start : span.end]
        + unmatched[severity_idx, span.start : span.end]
    )

    true_positive_by_char = np.divide(
        true_positive_numerator,
        denominator,
        out=np.zeros_like(true_positive_numerator, dtype=np.float64),
        where=denominator > 0,
    )
    error_by_char = np.divide(
        error_numerator,
        denominator,
        out=np.zeros_like(error_numerator, dtype=np.float64),
        where=denominator > 0,
    )
    return (
        float(true_positive_by_char.sum() / span.length),
        float(error_by_char.sum() / span.length),
    )


def _char_count_arrays_by_severity(
    predictions: list[ErrorSpan],
    references: list[ErrorSpan],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str | None, int]]:
    """Build per-side, per-severity character coverage arrays."""

    labels = _collect_severity_labels(predictions, references)
    severity_to_idx = {label: idx for idx, label in enumerate(labels)}
    counts_by_side: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for side in ("source", "target"):
        side_predictions = [span for span in predictions if span.side == side]
        side_references = [span for span in references if span.side == side]
        max_end = max(
            [0]
            + [span.end for span in side_predictions]
            + [span.end for span in side_references]
        )
        pred_counts = np.zeros((len(labels), max_end), dtype=np.float64)
        ref_counts = np.zeros((len(labels), max_end), dtype=np.float64)

        for span in side_predictions:
            pred_counts[severity_to_idx[span.severity], span.start : span.end] += 1.0
        for span in side_references:
            ref_counts[severity_to_idx[span.severity], span.start : span.end] += 1.0

        counts_by_side[side] = (pred_counts, ref_counts)

    return counts_by_side, severity_to_idx


def _collect_severity_labels(
    predictions: list[ErrorSpan],
    references: list[ErrorSpan],
) -> list[str | None]:
    """Return stable severity labels present in one segment."""

    labels: list[str | None] = []
    seen: set[str | None] = set()
    for span in [*predictions, *references]:
        if span.severity in seen:
            continue
        seen.add(span.severity)
        labels.append(span.severity)
    return labels or [None]


def _decompose_counts(
    pred_counts: np.ndarray,
    ref_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split coverage into exact, severity-mismatch, and unmatched mass."""

    num_severities, length = pred_counts.shape
    pred_exact = np.zeros((num_severities, length), dtype=np.float64)
    pred_mismatch = np.zeros((num_severities, length), dtype=np.float64)
    pred_unmatched = np.zeros((num_severities, length), dtype=np.float64)
    ref_exact = np.zeros((num_severities, length), dtype=np.float64)
    ref_mismatch = np.zeros((num_severities, length), dtype=np.float64)
    ref_unmatched = np.zeros((num_severities, length), dtype=np.float64)

    for idx in range(length):
        pred = pred_counts[:, idx].astype(np.float64)
        ref = ref_counts[:, idx].astype(np.float64)
        if not (pred.any() or ref.any()):
            continue

        exact = np.minimum(pred, ref)
        pred_exact[:, idx] = exact
        ref_exact[:, idx] = exact

        pred_remaining = pred - exact
        ref_remaining = ref - exact
        pred_remaining_total = float(pred_remaining.sum())
        ref_remaining_total = float(ref_remaining.sum())
        mismatch_pairs = min(pred_remaining_total, ref_remaining_total)

        if mismatch_pairs > 0.0:
            pred_mismatch[:, idx] = pred_remaining * (
                mismatch_pairs / pred_remaining_total
            )
            ref_mismatch[:, idx] = ref_remaining * (
                mismatch_pairs / ref_remaining_total
            )

        pred_unmatched[:, idx] = np.maximum(
            0.0, pred_remaining - pred_mismatch[:, idx]
        )
        ref_unmatched[:, idx] = np.maximum(
            0.0, ref_remaining - ref_mismatch[:, idx]
        )

    return (
        pred_exact,
        pred_mismatch,
        pred_unmatched,
        ref_exact,
        ref_mismatch,
        ref_unmatched,
    )


def _precision_recall_fscore(counts: TPCounts) -> tuple[float, float, float]:
    """Convert raw counts into precision, recall, and F-score.

    The input is one aggregate or per-segment ``TPCounts`` object. The output is
    ``(precision, recall, f_score)`` using the package's zero-division convention
    of returning 1.0 for precision or recall when that denominator is empty.
    """

    precision_denominator = counts.tp_for_precision + counts.fp
    recall_denominator = counts.tp_for_recall + counts.fn

    precision = (
        counts.tp_for_precision / precision_denominator
        if precision_denominator > 0
        else 1.0
    )
    recall = (
        counts.tp_for_recall / recall_denominator if recall_denominator > 0 else 1.0
    )
    f_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return precision, recall, f_score


def _sum_counts(list_of_tp_counts: Iterable[TPCounts]) -> TPCounts:
    """Aggregate a sequence of ``TPCounts`` objects.

    The input is usually the per-segment count list. The output is a single
    count object used by micro averaging.
    """

    total_tp_counts = TPCounts()
    for counts in list_of_tp_counts:
        total_tp_counts += counts
    return total_tp_counts


def _span_key(span: ErrorSpan) -> tuple[str, int, int]:
    """Return the hashable identity used for exact multiset matching."""

    return span.side, span.start, span.end


def _normalize_input(value: list, name: str) -> list[list[ErrorSpan]]:
    """Normalize user-provided spans into a list of segment span lists.

    ``value`` may be a flat list of spans for one segment or a nested list of
    spans for multiple segments. ``name`` is only used in error messages. The
    output always has shape ``list[list[ErrorSpan]]``.
    """

    if value is None:
        raise ValueError(f"{name} cannot be None")

    if not isinstance(value, list):
        raise TypeError(f"{name} must be a  list of spans or a list of lists of spans")

    if not value:
        return [[]]

    if _is_span_like(value[0]):
        return [[ErrorSpan.from_any(item) for item in value]]
    if all(isinstance(segment, list) for segment in value):
        normalized_segments: list[list[ErrorSpan]] = []
        for segment_idx, segment in enumerate(value):
            if not all(_is_span_like(item) for item in segment):
                raise TypeError(
                    f"{name}[{segment_idx}] must be a list of spans"
                )
            normalized_segments.append(
                [ErrorSpan.from_any(item) for item in segment]
            )
        return normalized_segments
    raise TypeError(
        f"{name} must be a list of spans or a list of lists of spans"
    )

def _is_span_like(value: Any) -> bool:
    """Return whether a value looks like an accepted span object.

    The input can be an ``ErrorSpan``, a dictionary with ``start``/``end``, or an
    object exposing ``start`` and ``end`` attributes. The output is a boolean
    used by input normalization.
    """

    if isinstance(value, ErrorSpan):
        return True
    if isinstance(value, dict):
        return "start" in value and "end" in value
    return hasattr(value, "start") and hasattr(value, "end")


def _coerce_texts(
    texts: str | Sequence[str],
    num_segments: int,
    name: str,
) -> list[str]:
    """Normalize optional source/target text arguments.

    ``texts`` may be a single string for one segment, or a sequence of
    strings. The output is aligned with the normalized segment
    list length.
    """

    if isinstance(texts, str):
        if num_segments != 1:
            raise ValueError(f"{name} must contain {num_segments} entries")
        return [texts]

    text_list = list(texts)
    if len(text_list) != num_segments:
        raise ValueError(f"{name} must contain {num_segments} entries")
    for idx, text in enumerate(text_list):
        if text is None:
            raise ValueError(f"{name} must contain {num_segments} non-None entries")
        if not isinstance(text, str):
            raise TypeError(f"{name}[{idx}] must be a string")
    return text_list


def _validate_text_bounds(
    segments: list[list[ErrorSpan]],
    source_texts: list[str] | None,
    target_texts: list[str] | None,
    name: str,
) -> None:
    """Validate that span offsets fit inside optional source/target texts.

    Inputs are normalized segments and text lists aligned by segment. The
    function returns nothing and raises when a span end offset exceeds the text
    length for its side.
    """

    for segment_idx, spans in enumerate(segments):
        for span_idx, span in enumerate(spans):
            texts = source_texts if span.side == "source" else target_texts
            if texts is None:
                continue
            text = texts[segment_idx]
            if span.end > len(text):
                raise ValueError(
                    f"{name}[{segment_idx}][{span_idx}] ends at {span.end}, "
                    f"which exceeds the {span.side} text length {len(text)}"
                )


def _validate_measure(value: Any):
    """Validate the requested metric name.

    The input is the user-provided ``measure`` option. The function returns
    nothing and raises if the value is not one of the package-supported measures
    or ``"all"``.
    """

    if value not in set(MEASURES).union({"all"}):
        raise ValueError(f"measure must be one of {MEASURES} or 'all'")

def _validate_matching(value: Any):
    """Validate the requested matching strategy.

    The input is the user-provided matching option. The function returns nothing
    and raises if the value is not supported.
    """

    if value not in MATCHING_STRATEGIES:
        raise ValueError(f"matching must be one of {MATCHING_STRATEGIES}")
    
def _validate_matching_algorithm(value: Any):
    """Validate the requested one-to-one matching algorithm.

    The input is the user-provided matching algorithm option. The function
    returns nothing and raises if the value is not supported.
    """

    if value not in MATCHING_ALGORITHMS:
        raise ValueError(f"matching_algorithm must be one of {MATCHING_ALGORITHMS}")
    
def _validate_averaging(value: Any):
    """Validate the requested averaging strategy.

    The input is the user-provided averaging option. The function returns
    nothing and raises if the value is not supported.
    """

    if value not in AVERAGING_STRATEGIES:
        raise ValueError(f"averaging must be one of {AVERAGING_STRATEGIES}")


def _validate_severity_penalty(value: Any) -> float:
    """Validate and normalize the caller-provided severity penalty."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("severity_penalty must be a number between 0.0 and 1.0")

    penalty = float(value)
    if not np.isfinite(penalty) or penalty < 0.0 or penalty > 1.0:
        raise ValueError("severity_penalty must be between 0.0 and 1.0")
    return penalty


def _validate_required_severities(
    segments: list[list[ErrorSpan]],
    name: str,
) -> None:
    """Require every span to carry severity for non-zero penalties."""

    for segment_idx, spans in enumerate(segments):
        for span_idx, span in enumerate(spans):
            if span.severity is None:
                raise ValueError(
                    "severity_penalty > 0 requires every span to include a "
                    f"non-empty severity; missing {name}[{segment_idx}][{span_idx}]"
                )
