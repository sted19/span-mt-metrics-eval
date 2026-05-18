"""One-to-one span matching helpers.

This module keeps matching separate from metric counting. Matchers return only
index pairs; the metric module decides how each selected pair contributes to
precision, recall, false positives, and false negatives.
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Callable, Mapping

import numpy as np

from span_mt_metrics_eval.types import ErrorSpan, Measure

MatchPairs = list[tuple[int, int]]
_LINEAR_SUM_ASSIGNMENT = None
_SCIPY_IMPORT_ATTEMPTED = False
_FLOAT_TOLERANCE = 1e-12


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
    if measure == "MPP":
        return lambda pred, ref: (
            harmonic_overlap_score(pred, ref)
            * severity_reward(pred, ref, severity_penalty)
        )
    raise ValueError(f"Unknown measure: {measure}")


def find_greedy_matches(
    predictions: list[ErrorSpan],
    references: list[ErrorSpan],
    severity_penalty: float = 0.0,
    severity_weights: Mapping[str, float] | None = None,
) -> MatchPairs:
    """Greedily select non-conflicting pairs by harmonic overlap score.

    Inputs are the predicted and reference spans for one segment. The output is a
    list of ``(prediction_index, reference_index)`` pairs, with each index used
    at most once.
    """

    candidates: list[tuple[float, int, int]] = []
    for pred_idx, pred in enumerate(predictions):
        for ref_idx, ref in enumerate(references):
            score = harmonic_overlap_score(pred, ref) * _mpp_pair_credit_mass(
                pred, ref, severity_penalty, severity_weights
            )
            if score > 0:
                candidates.append((score, pred_idx, ref_idx))

    candidates.sort(key=lambda item: item[0], reverse=True)

    used_predictions: set[int] = set()
    used_references: set[int] = set()
    matches: MatchPairs = []

    for _, pred_idx, ref_idx in candidates:
        if pred_idx in used_predictions or ref_idx in used_references:
            continue
        used_predictions.add(pred_idx)
        used_references.add(ref_idx)
        matches.append((pred_idx, ref_idx))

    return matches


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
        return _find_optimal_mpp_matches(
            predictions, references, severity_penalty, severity_weights
        )

    return _find_linear_sum_matches(
        predictions, references, metric_objective(measure, severity_penalty)
    )


def _find_linear_sum_matches(
    predictions: list[ErrorSpan],
    references: list[ErrorSpan],
    objective: Callable[[ErrorSpan, ErrorSpan], float],
) -> MatchPairs:
    """Find the assignment maximizing a separable pairwise objective.

    This is exact for ``EM``, ``MP``, and ``WMT23`` because their segment-level
    F-scores are monotonic in a single summed pairwise score. ``MPP`` is handled
    separately because its paper definition maximizes the final harmonic mean of
    aggregate span-averaged precision and recall, which is not generally
    equivalent to maximizing summed pairwise F-scores.
    """

    scores = np.zeros((len(predictions), len(references)), dtype=np.float64)

    for pred_idx, pred in enumerate(predictions):
        for ref_idx, ref in enumerate(references):
            scores[pred_idx, ref_idx] = objective(pred, ref)

    scipy_linear_sum_assignment = _get_scipy_linear_sum_assignment()
    if scipy_linear_sum_assignment is None:
        row_ind, col_ind = _linear_sum_assignment_fallback(scores)
    else:
        row_ind, col_ind = scipy_linear_sum_assignment(scores, maximize=True)

    return [
        (int(pred_idx), int(ref_idx))
        for pred_idx, ref_idx in zip(row_ind, col_ind)
        if scores[pred_idx, ref_idx] > 0.0
    ]


def _find_optimal_mpp_matches(
    predictions: list[ErrorSpan],
    references: list[ErrorSpan],
    severity_penalty: float = 0.0,
    severity_weights: Mapping[str, float] | None = None,
) -> MatchPairs:
    """Find the one-to-one MPP matching maximizing final segment F-score.

    For MPP, pair ``(pred, ref)`` contributes ``overlap / len(pred)`` to the
    precision numerator and ``overlap / len(ref)`` to the recall numerator. The
    selected matching must maximize the harmonic mean of the resulting
    segment-level precision and recall, not the sum of pairwise harmonic scores.
    """

    candidates_by_prediction: list[list[tuple[int, float, float, float]]] = []
    for pred in predictions:
        candidates: list[tuple[int, float, float, float]] = []
        for ref_idx, ref in enumerate(references):
            overlap = overlap_length(pred, ref)
            if overlap <= 0:
                continue
            reward = _mpp_pair_credit_mass(
                pred, ref, severity_penalty, severity_weights
            )
            pair_score = harmonic_overlap_score(pred, ref) * reward
            if pair_score <= 0:
                continue
            precision_credit = (overlap / pred.length) * reward
            recall_credit = (overlap / ref.length) * reward
            candidates.append(
                (ref_idx, precision_credit, recall_credit, pair_score)
            )
        candidates.sort(key=lambda item: item[3], reverse=True)
        candidates_by_prediction.append(candidates)

    if not any(candidates_by_prediction):
        return []

    prediction_denominator = _mpp_denominator(predictions, severity_weights)
    reference_denominator = _mpp_denominator(references, severity_weights)

    prediction_order = sorted(
        range(len(predictions)),
        key=lambda pred_idx: (
            len(candidates_by_prediction[pred_idx]),
            -max(
                (
                    pair_score
                    for _, _, _, pair_score in candidates_by_prediction[pred_idx]
                ),
                default=0.0,
            ),
            pred_idx,
        ),
    )
    ordered_candidates = [
        candidates_by_prediction[pred_idx] for pred_idx in prediction_order
    ]

    suffix_precision_credit = [0.0] * (len(prediction_order) + 1)
    suffix_recall_credit = [0.0] * (len(prediction_order) + 1)
    for pos in range(len(prediction_order) - 1, -1, -1):
        candidates = ordered_candidates[pos]
        suffix_precision_credit[pos] = suffix_precision_credit[pos + 1] + max(
            (precision_credit for _, precision_credit, _, _ in candidates),
            default=0.0,
        )
        suffix_recall_credit[pos] = suffix_recall_credit[pos + 1] + max(
            (recall_credit for _, _, recall_credit, _ in candidates),
            default=0.0,
        )

    # A strong starting point makes the branch-and-bound search much smaller,
    # while the exhaustive search below still decides the final answer.
    best_matches = _find_linear_sum_matches(
        predictions,
        references,
        lambda pred, ref: harmonic_overlap_score(pred, ref)
        * _mpp_pair_credit_mass(pred, ref, severity_penalty, severity_weights),
    )
    best_key = _mpp_matching_key(
        best_matches, predictions, references, severity_penalty, severity_weights
    )
    current_matches: MatchPairs = []

    def search(
        pos: int,
        used_references: int,
        precision_credit_sum: float,
        recall_credit_sum: float,
    ) -> None:
        nonlocal best_key, best_matches

        upper_f_score = _mpp_f_score_from_credits(
            precision_credit_sum + suffix_precision_credit[pos],
            recall_credit_sum + suffix_recall_credit[pos],
            prediction_denominator,
            reference_denominator,
        )
        if upper_f_score < best_key[0] - _FLOAT_TOLERANCE:
            return

        if pos == len(prediction_order):
            candidate_key = _mpp_matching_key_from_credits(
                precision_credit_sum,
                recall_credit_sum,
                prediction_denominator,
                reference_denominator,
                len(current_matches),
            )
            if _mpp_key_is_better(
                candidate_key, best_key, current_matches, best_matches
            ):
                best_key = candidate_key
                best_matches = list(current_matches)
            return

        pred_idx = prediction_order[pos]
        for ref_idx, precision_credit, recall_credit, _ in ordered_candidates[pos]:
            ref_mask = 1 << ref_idx
            if used_references & ref_mask:
                continue

            current_matches.append((pred_idx, ref_idx))
            search(
                pos + 1,
                used_references | ref_mask,
                precision_credit_sum + precision_credit,
                recall_credit_sum + recall_credit,
            )
            current_matches.pop()

        search(pos + 1, used_references, precision_credit_sum, recall_credit_sum)

    search(0, 0, 0.0, 0.0)
    best_matches.sort()
    return best_matches


def _mpp_matching_key(
    matches: MatchPairs,
    predictions: list[ErrorSpan],
    references: list[ErrorSpan],
    severity_penalty: float = 0.0,
    severity_weights: Mapping[str, float] | None = None,
) -> tuple[float, float, float, float, float]:
    """Return the objective key for an MPP matching."""

    precision_credit_sum = 0.0
    recall_credit_sum = 0.0
    for pred_idx, ref_idx in matches:
        pred = predictions[pred_idx]
        ref = references[ref_idx]
        overlap = overlap_length(pred, ref)
        reward = _mpp_pair_credit_mass(pred, ref, severity_penalty, severity_weights)
        precision_credit_sum += (overlap / pred.length) * reward
        recall_credit_sum += (overlap / ref.length) * reward

    return _mpp_matching_key_from_credits(
        precision_credit_sum,
        recall_credit_sum,
        _mpp_denominator(predictions, severity_weights),
        _mpp_denominator(references, severity_weights),
        len(matches),
    )


def _mpp_matching_key_from_credits(
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
    f_score = _harmonic_mean(precision, recall)
    return (f_score, precision + recall, precision, recall, float(match_count))


def _mpp_f_score_from_credits(
    precision_credit_sum: float,
    recall_credit_sum: float,
    prediction_count: float,
    reference_count: float,
) -> float:
    """Compute MPP F-score from aggregate precision/recall credits."""

    precision = precision_credit_sum / prediction_count if prediction_count else 1.0
    recall = recall_credit_sum / reference_count if reference_count else 1.0
    return _harmonic_mean(precision, recall)


def _harmonic_mean(precision: float, recall: float) -> float:
    """Return the harmonic mean using the paper's zero-denominator convention."""

    return (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )


def _mpp_key_is_better(
    candidate_key: tuple[float, float, float, float, float],
    best_key: tuple[float, float, float, float, float],
    candidate_matches: MatchPairs,
    best_matches: MatchPairs,
) -> bool:
    """Return whether one MPP objective key should replace another."""

    for candidate_value, best_value in zip(candidate_key, best_key):
        if candidate_value > best_value + _FLOAT_TOLERANCE:
            return True
        if candidate_value < best_value - _FLOAT_TOLERANCE:
            return False
    return sorted(candidate_matches) < sorted(best_matches)



def _mpp_pair_credit_mass(
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


def _mpp_denominator(
    spans: list[ErrorSpan],
    severity_weights: Mapping[str, float] | None,
) -> float:
    """Return the MPP denominator for one side of a segment."""

    if severity_weights is None:
        return float(len(spans))
    return sum(severity_weights[span.severity or ""] for span in spans)

def _get_scipy_linear_sum_assignment():
    """Return SciPy's assignment solver when it imports cleanly.

    The local development environment may have an incompatible SciPy/Numpy wheel
    pair. Importing lazily keeps non-optimal paths importable and lets the pure
    Python fallback handle optimal matching when SciPy is unavailable.
    """

    global _LINEAR_SUM_ASSIGNMENT, _SCIPY_IMPORT_ATTEMPTED

    if _SCIPY_IMPORT_ATTEMPTED:
        return _LINEAR_SUM_ASSIGNMENT

    _SCIPY_IMPORT_ATTEMPTED = True
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            from scipy.optimize import linear_sum_assignment
    except Exception:
        _LINEAR_SUM_ASSIGNMENT = None
    else:
        _LINEAR_SUM_ASSIGNMENT = linear_sum_assignment

    return _LINEAR_SUM_ASSIGNMENT


def _linear_sum_assignment_fallback(scores: np.ndarray) -> tuple[list[int], list[int]]:
    """Pure-Python rectangular Hungarian fallback for maximum assignment.

    ``scores`` is a dense matrix where rows are predictions and columns are
    references. The returned row and column index lists mirror SciPy's
    ``linear_sum_assignment`` output for the pairs selected by the solver.
    """

    if scores.size == 0:
        return [], []

    transposed = scores.shape[0] > scores.shape[1]
    working = scores.T if transposed else scores
    row_count, col_count = working.shape
    max_score = float(working.max()) if working.size else 0.0
    costs = (max_score - working).tolist()

    assignment = _hungarian_minimize(costs, row_count, col_count)

    row_indices: list[int] = []
    col_indices: list[int] = []
    for row_idx, col_idx in enumerate(assignment):
        if col_idx is None:
            continue
        if transposed:
            row_indices.append(col_idx)
            col_indices.append(row_idx)
        else:
            row_indices.append(row_idx)
            col_indices.append(col_idx)

    return row_indices, col_indices


def _hungarian_minimize(
    costs: list[list[float]],
    row_count: int,
    col_count: int,
) -> list[int | None]:
    """Return a minimum-cost assignment for ``row_count <= col_count``.

    This is the classic 1-indexed Hungarian algorithm for rectangular matrices.
    ``costs`` is minimized, and the returned list maps each row to either a
    column index or ``None`` when no column was assigned.
    """

    potentials_rows = [0.0] * (row_count + 1)
    potentials_cols = [0.0] * (col_count + 1)
    matched_row_for_col = [0] * (col_count + 1)
    previous_col = [0] * (col_count + 1)

    for row in range(1, row_count + 1):
        matched_row_for_col[0] = row
        current_col = 0
        min_values = [float("inf")] * (col_count + 1)
        used = [False] * (col_count + 1)

        # Grow one augmenting path from the current row. The potentials maintain
        # reduced costs, while previous_col lets us reconstruct the path once an
        # unmatched column is reached.
        while True:
            used[current_col] = True
            current_row = matched_row_for_col[current_col]
            delta = float("inf")
            next_col = 0

            for col in range(1, col_count + 1):
                if used[col]:
                    continue
                current_cost = (
                    costs[current_row - 1][col - 1]
                    - potentials_rows[current_row]
                    - potentials_cols[col]
                )
                if current_cost < min_values[col]:
                    min_values[col] = current_cost
                    previous_col[col] = current_col
                if min_values[col] < delta:
                    delta = min_values[col]
                    next_col = col

            for col in range(0, col_count + 1):
                if used[col]:
                    potentials_rows[matched_row_for_col[col]] += delta
                    potentials_cols[col] -= delta
                else:
                    min_values[col] -= delta

            current_col = next_col
            if matched_row_for_col[current_col] == 0:
                break

        while True:
            prior_col = previous_col[current_col]
            matched_row_for_col[current_col] = matched_row_for_col[prior_col]
            current_col = prior_col
            if current_col == 0:
                break

    assignment: list[int | None] = [None] * row_count
    for col in range(1, col_count + 1):
        matched_row = matched_row_for_col[col]
        if matched_row:
            assignment[matched_row - 1] = col - 1

    return assignment
