"""Optimal one-to-one span matching."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment

from span_mt_metrics_eval.matching.mpp import (
    mpp_denominator,
    mpp_key_is_better,
    mpp_matching_key_from_credits,
    mpp_pair_credit_mass,
)
from span_mt_metrics_eval.matching.pairs import (
    MatchPairs,
    metric_objective,
    overlap_length,
)
from span_mt_metrics_eval.options import Measure
from span_mt_metrics_eval.spans import ErrorSpan


def find_optimal_matches(
    predictions: list[ErrorSpan],
    references: list[ErrorSpan],
    measure: Measure,
    severity_penalty: float = 0.0,
    severity_weights: Mapping[str, float] | None = None,
) -> MatchPairs:
    """Find a score-maximizing one-to-one matching for the selected measure.

    Inputs are one segment's predicted/reference spans and the metric whose
    objective should be optimized. The output has the same index-pair format as
    ``find_greedy_matches``.
    """

    if not predictions or not references:
        return []

    if measure == "MPP":
        return find_optimal_mpp_matches(
            predictions, references, severity_penalty, severity_weights
        )

    return find_linear_sum_matches(
        predictions, references, metric_objective(measure, severity_penalty)
    )


def find_linear_sum_matches(
    predictions: list[ErrorSpan],
    references: list[ErrorSpan],
    objective: Callable[[ErrorSpan, ErrorSpan], float],
) -> MatchPairs:
    """Find the assignment maximizing a separable pairwise objective.

    This is exact for ``EM``, ``MP``, and ``WMT23`` because their segment-level
    F-scores are monotonic in a single summed pairwise score. ``MPP`` is handled
    separately because its paper definition maximizes the final harmonic mean of
    aggregate span-averaged precision and recall.
    """

    scores = np.zeros((len(predictions), len(references)), dtype=np.float64)

    for pred_idx, pred in enumerate(predictions):
        for ref_idx, ref in enumerate(references):
            scores[pred_idx, ref_idx] = objective(pred, ref)

    row_ind, col_ind = linear_sum_assignment(scores, maximize=True)

    return [
        (int(pred_idx), int(ref_idx))
        for pred_idx, ref_idx in zip(row_ind, col_ind)
        if scores[pred_idx, ref_idx] > 0.0
    ]


def find_optimal_mpp_matches(
    predictions: list[ErrorSpan],
    references: list[ErrorSpan],
    severity_penalty: float = 0.0,
    severity_weights: Mapping[str, float] | None = None,
) -> MatchPairs:
    """Find the one-to-one MPP matching maximizing final segment F-score.

    Inputs are one segment's predicted and reference spans plus optional severity
    scoring configuration. The returned list contains ``(prediction_index,
    reference_index)`` pairs, with each prediction and reference used at most
    once.

    MPP is different from ``EM``, ``MP``, and ``WMT23`` because selected pairs
    contribute two different credits:

    - ``overlap / len(prediction)`` to the precision numerator
    - ``overlap / len(reference)`` to the recall numerator

    The final objective is the harmonic mean of aggregate precision and recall
    after all pairs are selected, so the linear assignment shortcut used by the
    other measures is not valid. This function keeps the exact objective but
    implements it directly: it filters out pairs with no usable MPP credit, then
    enumerates all feasible one-to-one matchings over the remaining candidates.
    """

    # Build the candidate graph. Each prediction stores only references with
    # positive usable MPP credit. For each candidate we keep the reference index
    # plus the precision and recall credits that the pair would add.
    candidates_by_prediction: list[list[tuple[int, float, float]]] = []
    for pred in predictions:
        candidates: list[tuple[int, float, float]] = []
        for ref_idx, ref in enumerate(references):
            overlap = overlap_length(pred, ref)
            if overlap <= 0:
                continue
            reward = mpp_pair_credit_mass(pred, ref, severity_penalty, severity_weights)
            if reward <= 0.0:
                continue
            precision_credit = (overlap / pred.length) * reward
            recall_credit = (overlap / ref.length) * reward
            candidates.append((ref_idx, precision_credit, recall_credit))
        candidates_by_prediction.append(candidates)

    if not any(candidates_by_prediction):
        return []

    # Denominators are fixed for the whole segment. They are span counts for
    # ordinary MPP and severity mass totals for severity-weighted MPP. Keeping
    # them here lets the search compare partial matchings with the same final
    # precision/recall normalization the result will use.
    prediction_denominator = mpp_denominator(predictions, severity_weights)
    reference_denominator = mpp_denominator(references, severity_weights)

    best_matches: MatchPairs = []
    best_key = mpp_matching_key_from_credits(
        0.0,
        0.0,
        prediction_denominator,
        reference_denominator,
        0,
    )
    current_matches: MatchPairs = []

    def search(
        pred_idx: int,
        used_references: int,
        precision_credit_sum: float,
        recall_credit_sum: float,
    ) -> None:
        """Enumerate one-to-one continuations from the current prediction."""

        nonlocal best_key, best_matches

        # Once every prediction has been considered, compare the completed
        # matching against the incumbent using the final MPP objective plus
        # deterministic tie-breakers.
        if pred_idx == len(predictions):
            candidate_key = mpp_matching_key_from_credits(
                precision_credit_sum,
                recall_credit_sum,
                prediction_denominator,
                reference_denominator,
                len(current_matches),
            )
            if mpp_key_is_better(
                candidate_key, best_key, current_matches, best_matches
            ):
                best_key = candidate_key
                best_matches = list(current_matches)
            return

        # Try every still-available candidate reference for this prediction.
        # References are tracked as bits so conflict checks stay cheap.
        for ref_idx, precision_credit, recall_credit in candidates_by_prediction[
            pred_idx
        ]:
            ref_mask = 1 << ref_idx
            if used_references & ref_mask:
                continue

            current_matches.append((pred_idx, ref_idx))
            search(
                pred_idx + 1,
                used_references | ref_mask,
                precision_credit_sum + precision_credit,
                recall_credit_sum + recall_credit,
            )
            current_matches.pop()

        # Also explore the possibility that this prediction remains unmatched.
        # This is necessary when matching it would consume a reference that is
        # more valuable to a later prediction or when all available candidates
        # reduce the best aggregate precision/recall balance.
        search(pred_idx + 1, used_references, precision_credit_sum, recall_credit_sum)

    search(0, 0, 0.0, 0.0)
    best_matches.sort()
    return best_matches
