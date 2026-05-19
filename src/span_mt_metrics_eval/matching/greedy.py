"""Greedy one-to-one span matching."""

from __future__ import annotations

from collections.abc import Mapping

from span_mt_metrics_eval.matching.mpp import mpp_pair_credit_mass
from span_mt_metrics_eval.matching.pairs import MatchPairs, harmonic_overlap_score
from span_mt_metrics_eval.spans import ErrorSpan


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
            score = harmonic_overlap_score(pred, ref) * mpp_pair_credit_mass(
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
