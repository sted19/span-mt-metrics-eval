"""Matching algorithms and pair-level scoring helpers."""

from span_mt_metrics_eval.matching.greedy import find_greedy_matches
from span_mt_metrics_eval.matching.optimal import find_optimal_matches
from span_mt_metrics_eval.matching.pairs import (
    MatchPairs,
    harmonic_overlap_score,
    overlap_length,
    severity_reward,
    spans_exactly_match,
)

__all__ = [
    "MatchPairs",
    "find_greedy_matches",
    "find_optimal_matches",
    "harmonic_overlap_score",
    "overlap_length",
    "severity_reward",
    "spans_exactly_match",
]

