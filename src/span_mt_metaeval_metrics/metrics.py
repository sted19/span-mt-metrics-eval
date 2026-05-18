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
from typing import Any
import logging

import numpy as np

from span_mt_metaeval_metrics.matching import (
    MatchPairs,
    find_greedy_matches,
    find_optimal_matches,
    overlap_length,
    spans_exactly_match,
)
from span_mt_metaeval_metrics.types import (
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
    source_texts: str | list[str] | None = None,
    target_texts: str | list[str] | None = None,
) -> MetricResult | dict[Measure, MetricResult]:
    """Compute span-level precision, recall, and F-score.

    ``predictions`` and ``references`` can be a single segment's spans or a
    sequence of per-segment span sequences. Each span can be an ``ErrorSpan`` or a
    dict with ``start``, ``end``, and ``side`` keys.

    The metric is selected with ``measure``. Passing ``measure="all"`` returns a
    dictionary with one ``MetricResult`` per metric; otherwise this function
    returns a single ``MetricResult``. Optional text inputs are currently kept in
    the signature for users who want to pass source/target context alongside
    offsets.
    """

    _validate_measure(measure)
    _validate_matching(matching)
    _validate_matching_algorithm(matching_algorithm)
    _validate_averaging(averaging)

    if measure == "all":
        result = {
            item: compute(
                predictions,
                references,
                measure=item,
                matching=matching,
                matching_algorithm=matching_algorithm,
                averaging=averaging,
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
) -> TPCounts:
    """Compute one segment's raw counts for the requested matching strategy.

    Inputs are normalized prediction and reference spans for a single segment,
    plus the selected metric options. The output is a ``TPCounts`` object ready
    for aggregation.
    """

    if matching == "one_to_one":
        return _compute_o2o_tp_counts(
            predictions, references, measure, matching_algorithm
        )
    return _compute_m2m_tp_counts(predictions, references, measure)


def _compute_o2o_tp_counts(
    predictions: list[ErrorSpan],
    references: list[ErrorSpan],
    measure: Measure,
    matching_algorithm: MatchingAlgorithm,
) -> TPCounts:
    """Compute one-to-one counts after choosing matched span pairs.

    ``matching_algorithm`` controls how prediction/reference pairs are selected.
    Once matching is done, the selected measure determines how each pair
    contributes to true-positive and error counts.
    """

    if matching_algorithm == "greedy":
        matches = find_greedy_matches(predictions, references)
    else:
        matches = find_optimal_matches(predictions, references, measure)

    if measure == "EM":
        return _compute_o2o_em_tp_counts(predictions, references, matches)
    if measure == "MP":
        return _compute_o2o_mp_tp_counts(predictions, references, matches)
    if measure == "WMT23":
        return _compute_o2o_wmt23_tp_counts(predictions, references, matches)
    if measure == "MPP":
        return _compute_o2o_mpp_tp_counts(predictions, references, matches)
    raise ValueError(f"Unknown measure: {measure}")


def _compute_o2o_em_tp_counts(
    predictions: list[ErrorSpan],
    references: list[ErrorSpan],
    matches: MatchPairs,
) -> TPCounts:
    """Compute exact-match counts for a one-to-one matching.

    Inputs are the spans and already-selected match pairs. The output gives one
    true positive only to pairs whose side and offsets are identical; all other
    predictions/references are counted as false positives or false negatives.
    """

    exact_matches = sum(
        1.0 for pred_idx, ref_idx in matches if spans_exactly_match(predictions[pred_idx], references[ref_idx])
    )
    return TPCounts(
        tp_for_precision=exact_matches,
        tp_for_recall=exact_matches,
        fp=float(len(predictions)) - exact_matches,
        fn=float(len(references)) - exact_matches,
    )


def _compute_o2o_mp_tp_counts(
    predictions: list[ErrorSpan],
    references: list[ErrorSpan],
    matches: MatchPairs,
) -> TPCounts:
    """Compute binary partial-overlap counts for a one-to-one matching.

    Every selected pair has positive overlap and therefore receives one full
    true positive on both the precision and recall side.
    """

    true_positives = float(len(matches))
    return TPCounts(
        tp_for_precision=true_positives,
        tp_for_recall=true_positives,
        fp=float(len(predictions)) - true_positives,
        fn=float(len(references)) - true_positives,
    )


def _compute_o2o_wmt23_tp_counts(
    predictions: list[ErrorSpan],
    references: list[ErrorSpan],
    matches: MatchPairs,
) -> TPCounts:
    """Compute character-count ``WMT23`` counts for a one-to-one matching.

    Overlapping characters in matched pairs are true positives. Uncovered
    prediction characters become false positives, and uncovered reference
    characters become false negatives.
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
        overlap = overlap_length(pred, references[ref_idx])
        true_positive_chars += overlap
        false_positive_chars += pred.length - overlap

    for ref_idx, ref in enumerate(references):
        pred_idx = matched_by_reference.get(ref_idx)
        if pred_idx is None:
            false_negative_chars += ref.length
            continue
        overlap = overlap_length(predictions[pred_idx], ref)
        false_negative_chars += ref.length - overlap

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
) -> TPCounts:
    """Compute partial-credit ``MPP`` counts for a one-to-one matching.

    Matched predictions receive precision-side credit according to the fraction
    of their span that overlaps the reference. Matched references receive
    recall-side credit according to the fraction of their span that overlaps the
    prediction.
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
        credit = overlap_length(pred, references[ref_idx]) / pred.length
        tp_for_precision += credit
        fp += 1.0 - credit

    tp_for_recall = 0.0
    fn = 0.0
    for ref_idx, ref in enumerate(references):
        pred_idx = matched_by_reference.get(ref_idx)
        if pred_idx is None:
            fn += 1.0
            continue
        credit = overlap_length(predictions[pred_idx], ref) / ref.length
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
) -> TPCounts:
    """Compute one segment's many-to-many counts for the selected measure.

    Many-to-many scoring does not choose explicit pairs. Instead, each helper
    compares all predicted and reference span coverage for the segment and
    returns raw counts for one metric.
    """

    if measure == "EM":
        return _compute_m2m_em_tp_counts(predictions, references)
    if measure == "MP":
        return _compute_m2m_mp_tp_counts(predictions, references)
    if measure == "WMT23":
        return _compute_m2m_wmt23_tp_counts(predictions, references)
    if measure == "MPP":
        return _compute_m2m_mpp_tp_counts(predictions, references)
    raise ValueError(f"Unknown measure: {measure}")


def _compute_m2m_em_tp_counts(
    predictions: list[ErrorSpan],
    references: list[ErrorSpan],
) -> TPCounts:
    """Compute many-to-many exact-match counts as multiset overlap.

    Inputs are normalized spans for one segment. The output compares the
    multisets of ``(side, start, end)`` triples, preserving duplicate identical
    spans through their counts.
    """

    pred_counter = Counter(_span_key(span) for span in predictions)
    ref_counter = Counter(_span_key(span) for span in references)

    true_positives = float(
        sum(min(count, ref_counter[key]) for key, count in pred_counter.items())
    )
    return TPCounts(
        tp_for_precision=true_positives,
        tp_for_recall=true_positives,
        fp=float(len(predictions)) - true_positives,
        fn=float(len(references)) - true_positives,
    )


def _compute_m2m_mp_tp_counts(
    predictions: list[ErrorSpan],
    references: list[ErrorSpan],
) -> TPCounts:
    """Compute many-to-many binary partial-overlap counts.

    A predicted span receives one precision-side true positive if it overlaps at
    least one reference span. A reference span receives one recall-side true
    positive if it overlaps at least one prediction span.
    """

    tp_for_precision = sum(
        1.0
        for pred in predictions
        if any(overlap_length(pred, ref) > 0 for ref in references)
    )
    tp_for_recall = sum(
        1.0
        for ref in references
        if any(overlap_length(pred, ref) > 0 for pred in predictions)
    )

    return TPCounts(
        tp_for_precision=tp_for_precision,
        tp_for_recall=tp_for_recall,
        fp=float(len(predictions)) - tp_for_precision,
        fn=float(len(references)) - tp_for_recall,
    )


def _compute_m2m_wmt23_tp_counts(
    predictions: list[ErrorSpan],
    references: list[ErrorSpan],
) -> TPCounts:
    """Compute many-to-many character-count ``WMT23`` counts.

    The input spans are converted to per-character coverage arrays separately
    for source and target sides. Per-character overlap contributes true
    positives; excess prediction/reference coverage contributes false positives
    and false negatives.
    """

    true_positive_chars = 0.0
    false_positive_chars = 0.0
    false_negative_chars = 0.0

    for side in ("source", "target"):
        pred_counts, ref_counts = _char_count_arrays(predictions, references, side)
        matched = np.minimum(pred_counts, ref_counts)
        true_positive_chars += float(matched.sum())
        false_positive_chars += float(np.maximum(pred_counts - ref_counts, 0).sum())
        false_negative_chars += float(np.maximum(ref_counts - pred_counts, 0).sum())

    return TPCounts(
        tp_for_precision=true_positive_chars,
        tp_for_recall=true_positive_chars,
        fp=false_positive_chars,
        fn=false_negative_chars,
    )


def _compute_m2m_mpp_tp_counts(
    predictions: list[ErrorSpan],
    references: list[ErrorSpan],
) -> TPCounts:
    """Compute many-to-many partial-credit ``MPP`` counts.

    The function first computes matched character mass per side, then assigns
    each span the average matched fraction across its own characters. The output
    keeps separate true-positive numerators for precision and recall.
    """

    matched_by_side: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for side in ("source", "target"):
        pred_counts, ref_counts = _char_count_arrays(predictions, references, side)
        matched = np.minimum(pred_counts, ref_counts)
        matched_by_side[side] = (pred_counts, ref_counts, matched)

    tp_for_precision = 0.0
    for pred in predictions:
        pred_counts, _, matched = matched_by_side[pred.side]
        tp_for_precision += _span_average_credit(pred, matched, pred_counts)

    tp_for_recall = 0.0
    for ref in references:
        _, ref_counts, matched = matched_by_side[ref.side]
        tp_for_recall += _span_average_credit(ref, matched, ref_counts)

    return TPCounts(
        tp_for_precision=tp_for_precision,
        tp_for_recall=tp_for_recall,
        fp=float(len(predictions)) - tp_for_precision,
        fn=float(len(references)) - tp_for_recall,
    )


def _span_average_credit(
    span: ErrorSpan,
    matched_counts: np.ndarray,
    denominator_counts: np.ndarray,
) -> float:
    """Return a span's average per-character matched credit.

    ``matched_counts`` contains matched mass at each character position, and
    ``denominator_counts`` contains the prediction or reference mass used as the
    denominator. The returned value is a floating-point credit in ``[0, 1]`` for
    the given span.
    """

    if len(denominator_counts) < span.end:
        raise ValueError(f"The array denominator_counts is too short! It must be at least as long as span.end={span.end}")

    matched_slice = matched_counts[span.start : span.end]
    denominator_slice = denominator_counts[span.start : span.end]
    credit_by_char = np.divide(
        matched_slice,
        denominator_slice,
        out=np.zeros_like(matched_slice, dtype=np.float64),
        where=denominator_slice > 0,
    )
    return float(credit_by_char.sum() / span.length)


def _char_count_arrays(
    predictions: list[ErrorSpan],
    references: list[ErrorSpan],
    side: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-character prediction and reference coverage arrays.

    Inputs are all segment spans plus the side to inspect. The output arrays have
    one entry per character position and count how many prediction/reference
    spans cover that position.
    """

    side_predictions = [span for span in predictions if span.side == side]
    side_references = [span for span in references if span.side == side]
    max_end = max(
        [0]
        + [span.end for span in side_predictions]
        + [span.end for span in side_references]
    )

    pred_counts = np.zeros(max_end, dtype=np.float64)
    ref_counts = np.zeros(max_end, dtype=np.float64)

    for span in side_predictions:
        pred_counts[span.start : span.end] += 1.0
    for span in side_references:
        ref_counts[span.start : span.end] += 1.0

    return pred_counts, ref_counts


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
