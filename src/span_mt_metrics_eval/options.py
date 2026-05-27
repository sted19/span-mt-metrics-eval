"""Supported metric options and their type aliases."""

from __future__ import annotations

from typing import Literal


Measure = Literal["EM", "MP", "WMT25", "MPP"]
MatchingStrategy = Literal["one_to_one", "many_to_many"]
MatchingAlgorithm = Literal["optimal", "greedy"]
AveragingStrategy = Literal["micro", "macro"]

MEASURES: tuple[Measure, ...] = ("EM", "MP", "WMT25", "MPP")
MATCHING_STRATEGIES: tuple[MatchingStrategy, ...] = ("one_to_one", "many_to_many")
MATCHING_ALGORITHMS: tuple[MatchingAlgorithm, ...] = ("optimal", "greedy")
AVERAGING_STRATEGIES: tuple[AveragingStrategy, ...] = ("micro", "macro")
