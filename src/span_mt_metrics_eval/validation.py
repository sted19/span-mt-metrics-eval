"""Validation helpers for compute options and severity requirements."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import Any

import numpy as np

from span_mt_metrics_eval.options import (
    AVERAGING_STRATEGIES,
    MATCHING_ALGORITHMS,
    MATCHING_STRATEGIES,
    MEASURES,
)
from span_mt_metrics_eval.spans import ErrorSpan, normalize_severity


def validate_measure(value: Any) -> None:
    """Validate the requested metric name.

    The input is the user-provided ``measure`` option. The function returns
    nothing and raises if the value is not one of the supported measures.
    """

    if value not in set(MEASURES):
        raise ValueError(f"measure must be one of {MEASURES}")


def validate_matching(value: Any) -> None:
    """Validate the requested matching strategy.

    The input is the user-provided matching option. The function returns nothing
    and raises if the value is not supported.
    """

    if value not in MATCHING_STRATEGIES:
        raise ValueError(f"matching must be one of {MATCHING_STRATEGIES}")


def validate_matching_algorithm(value: Any) -> None:
    """Validate the requested one-to-one matching algorithm.

    The input is the user-provided matching algorithm option. The function
    returns nothing and raises if the value is not supported.
    """

    if value not in MATCHING_ALGORITHMS:
        raise ValueError(f"matching_algorithm must be one of {MATCHING_ALGORITHMS}")


def validate_averaging(value: Any) -> None:
    """Validate the requested averaging strategy.

    The input is the user-provided averaging option. The function returns
    nothing and raises if the value is not supported.
    """

    if value not in AVERAGING_STRATEGIES:
        raise ValueError(f"averaging must be one of {AVERAGING_STRATEGIES}")


def validate_severity_penalty(value: Any) -> float:
    """Validate and normalize the caller-provided severity penalty.

    The input is the raw option value. The returned float is finite and in the
    inclusive range ``[0.0, 1.0]``.
    """

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("severity_penalty must be a number between 0.0 and 1.0")

    penalty = float(value)
    if not np.isfinite(penalty) or penalty < 0.0 or penalty > 1.0:
        raise ValueError("severity_penalty must be between 0.0 and 1.0")
    return penalty


def validate_severity_weights(
    value: Mapping[str, Real] | None,
) -> dict[str, float] | None:
    """Validate and normalize caller-provided severity weights.

    The input may be ``None`` or a mapping from severity labels to non-negative
    finite weights. The returned mapping uses normalized severity labels.
    """

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("severity_weights must be a mapping of severity labels to weights")

    normalized_weights: dict[str, float] = {}
    for raw_label, raw_weight in value.items():
        if not isinstance(raw_label, str):
            raise TypeError("severity_weights keys must be non-empty severity labels")
        label = normalize_severity(raw_label)
        if label is None:
            raise ValueError("severity_weights keys must be non-empty severity labels")
        if label in normalized_weights:
            raise ValueError("severity_weights contains duplicate labels after normalization")
        if isinstance(raw_weight, bool) or not isinstance(raw_weight, Real):
            raise TypeError("severity_weights values must be finite non-negative numbers")

        weight = float(raw_weight)
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError("severity_weights values must be finite non-negative numbers")
        normalized_weights[label] = weight

    return normalized_weights


def validate_severity_weight_compatibility(
    measure: Any,
    severity_penalty: float,
    severity_weights: dict[str, float] | None,
) -> None:
    """Validate that severity weighting is used only for supported MPP calls."""

    if severity_weights is None:
        return
    if severity_penalty > 0.0:
        raise ValueError("severity_weights cannot be combined with a non-zero severity_penalty")
    if measure not in {"MPP"}:
        raise ValueError("severity_weights is only supported for measure='MPP'")


def validate_required_severity_weights(
    segments: list[list[ErrorSpan]],
    name: str,
    severity_weights: dict[str, float],
) -> None:
    """Require every span to carry a severity with a configured weight."""

    for segment_idx, spans in enumerate(segments):
        for span_idx, span in enumerate(spans):
            if span.severity is None:
                raise ValueError(
                    "severity_weights requires every span to include a "
                    f"non-empty severity; missing {name}[{segment_idx}][{span_idx}]"
                )
            if span.severity not in severity_weights:
                raise ValueError(
                    "severity_weights is missing a weight for severity "
                    f"{span.severity!r} at {name}[{segment_idx}][{span_idx}]"
                )


def validate_required_severities(
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
