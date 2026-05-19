"""Character coverage decomposition for many-to-many metrics."""

from __future__ import annotations

from collections import Counter

import numpy as np

from span_mt_metrics_eval.spans import ErrorSpan, span_key


def exact_span_severity_counters(
    spans: list[ErrorSpan],
) -> dict[tuple[str, int, int], Counter[str | None]]:
    """Group exact span identities by severity label.

    The input is a segment's spans. The returned mapping is used by many-to-many
    exact match scoring to handle duplicate exact spans as a multiset.
    """

    groups: dict[tuple[str, int, int], Counter[str | None]] = {}
    for span in spans:
        groups.setdefault(span_key(span), Counter())[span.severity] += 1
    return groups


def m2m_binary_overlap_reward(
    span: ErrorSpan,
    other_counts: np.ndarray,
    severity_to_idx: dict[str | None, int],
    severity_penalty: float,
) -> float | None:
    """Return binary many-to-many reward for one span, or ``None``.

    The function checks whether ``span`` overlaps any span represented by
    ``other_counts``. Same-severity overlap receives full credit; overlap only
    with different severities receives discounted credit.
    """

    other_slice = other_counts[:, span.start : span.end]
    if not bool(np.any(other_slice.sum(axis=0) > 0)):
        return None

    severity_idx = severity_to_idx[span.severity]
    same_severity_overlap = bool(np.any(other_slice[severity_idx] > 0))
    return 1.0 if same_severity_overlap else 1.0 - severity_penalty


def m2m_decompositions_by_side(
    predictions: list[ErrorSpan],
    references: list[ErrorSpan],
    severity_weights: dict[str, float] | None = None,
) -> tuple[
    dict[str, tuple[np.ndarray, ...]],
    dict[str | None, int],
]:
    """Return per-side severity decompositions for many-to-many metrics.

    The output maps ``source`` and ``target`` to prediction/reference coverage
    arrays plus exact, severity-mismatch, and unmatched coverage components.
    """

    counts_by_side, severity_to_idx = char_count_arrays_by_severity(
        predictions, references, severity_weights
    )
    decompositions: dict[str, tuple[np.ndarray, ...]] = {}
    for side, (pred_counts, ref_counts) in counts_by_side.items():
        decompositions[side] = (
            pred_counts,
            ref_counts,
            *decompose_counts(pred_counts, ref_counts),
        )
    return decompositions, severity_to_idx


def span_weighted_m2m_credits(
    span: ErrorSpan,
    severity_to_idx: dict[str | None, int],
    counts: np.ndarray,
    exact: np.ndarray,
    mismatch: np.ndarray,
    unmatched: np.ndarray,
    severity_penalty: float,
) -> tuple[float, float]:
    """Return average true-positive and error credit for one span.

    Inputs are per-character decomposition arrays for the span's side. The
    returned pair is the span-level true-positive credit and error credit.
    """

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


def span_severity_weighted_m2m_credits(
    span: ErrorSpan,
    severity_to_idx: dict[str | None, int],
    counts: np.ndarray,
    exact: np.ndarray,
    mismatch: np.ndarray,
    unmatched: np.ndarray,
    severity_weights: dict[str, float],
) -> tuple[float, float]:
    """Return severity-weighted average MPP credit for one span.

    ``counts`` and decomposition arrays contain weighted mass. Dividing by the
    weighted coverage for the span's severity distributes each character's
    matched mass proportionally across duplicate covering spans.
    """

    severity_idx = severity_to_idx[span.severity]
    span_weight = span_severity_weight(span, severity_weights)
    denominator = counts[severity_idx, span.start : span.end]
    true_positive_numerator = (
        exact[severity_idx, span.start : span.end]
        + mismatch[severity_idx, span.start : span.end]
    )
    error_numerator = unmatched[severity_idx, span.start : span.end]

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
        float(span_weight * true_positive_by_char.sum() / span.length),
        float(span_weight * error_by_char.sum() / span.length),
    )


def span_severity_weight(
    span: ErrorSpan,
    severity_weights: dict[str, float] | None,
) -> float:
    """Return one span's configured severity mass, defaulting to one."""

    if severity_weights is None:
        return 1.0
    return severity_weights[span.severity or ""]


def span_severity_mass_total(
    spans: list[ErrorSpan],
    severity_weights: dict[str, float],
) -> float:
    """Return the total configured severity mass for a list of spans."""

    return sum(span_severity_weight(span, severity_weights) for span in spans)


def char_count_arrays_by_severity(
    predictions: list[ErrorSpan],
    references: list[ErrorSpan],
    severity_weights: dict[str, float] | None = None,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str | None, int]]:
    """Build per-side, per-severity character coverage arrays.

    The inputs are one segment's predictions and references. The returned
    coverage arrays count unweighted span coverage by default, or severity
    weight mass when ``severity_weights`` is provided.
    """

    labels = collect_severity_labels(predictions, references)
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
            pred_counts[severity_to_idx[span.severity], span.start : span.end] += (
                span_severity_weight(span, severity_weights)
            )
        for span in side_references:
            ref_counts[severity_to_idx[span.severity], span.start : span.end] += (
                span_severity_weight(span, severity_weights)
            )

        counts_by_side[side] = (pred_counts, ref_counts)

    return counts_by_side, severity_to_idx


def collect_severity_labels(
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


def decompose_counts(
    pred_counts: np.ndarray,
    ref_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split coverage into exact, severity-mismatch, and unmatched mass.

    Inputs are per-severity character coverage arrays for one side. The returned
    arrays represent exact matched mass, cross-severity matched mass, and
    unmatched mass from both the prediction and reference perspectives.
    """

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
