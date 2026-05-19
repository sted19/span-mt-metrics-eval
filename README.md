# span-mt-metrics-eval

Standalone Python package for span-level precision, recall, and F-score metrics
for machine translation error annotations.

The package exposes the metric variants used in span-level MT meta-evaluation:

- `EM`: exact match
- `MP`: binary partial overlap
- `WMT23`: character-count precision/recall/F-score
- `MPP`: partial-overlap metric with partial credit

It supports one-to-one matching with greedy or optimal assignment, many-to-many
matching, and micro or macro averaging over multiple segments.

## Installation

From a local checkout:

```bash
python -m pip install .
```

For development:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Quick Start

```python
from span_mt_metrics_eval import compute

predictions = [
    {"start": 8, "end": 13, "side": "target", "severity": "major"},
]
references = [
    {"start": 8, "end": 13, "side": "target", "severity": "major"},
]

result = compute(
    predictions,
    references,
    measure="MPP",
    matching="one_to_one",
    matching_algorithm="optimal",
    averaging="micro",
)

print(result.precision, result.recall, result.f_score)
print(result.details.as_dict())
```

Span offsets are half-open character offsets: `start` is included and `end` is
excluded, matching normal Python slicing.

## Input Format

`compute` accepts annotations relative to either one segment:

```python
predictions = [{"start": 0, "end": 5, "side": "target"}]
references = [{"start": 1, "end": 6, "side": "target"}]
```

or multiple segments:

```python
predictions = [
    [{"start": 0, "end": 5, "side": "target"}],
    [{"start": 2, "end": 7, "side": "source"}],
]
references = [
    [{"start": 1, "end": 6, "side": "target"}],
    [],
]
```

Each span can be a dictionary or an `ErrorSpan` instance:

```python
from span_mt_metrics_eval import ErrorSpan

span = ErrorSpan(start=0, end=5, side="source")
```

Dictionary spans may use either `side` or the WMT-style `is_source_error` flag:

```python
{"start": 0, "end": 5, "side": "source"}
{"start": 0, "end": 5, "is_source_error": True}
```

Extra dictionary fields, such as `category`, are accepted and ignored by the
metric calculation. `severity` is optional by default, but it is used when
`severity_penalty` is greater than zero or `severity_weights` are provided.
Severity labels are stripped and lowercased before comparison.

## Options

`measure`:

- `EM`
- `MP`
- `WMT23`
- `MPP`

`matching`:

- `one_to_one`
- `many_to_many`

`matching_algorithm` for one-to-one matching:

- `optimal`
- `greedy`

`averaging`:

- `micro`
- `macro`

`severity_penalty`:

- float in `[0.0, 1.0]`
- `0.0` keeps severity labels from affecting scores
- values greater than `0.0` require every span to include a non-empty `severity`
- mismatched severities receive `1 - severity_penalty` credit

`severity_weights`:

- dictionary of severity labels to finite non-negative weights, for example
  `{"minor": 0.2, "major": 1.0}`
- supported for `measure="MPP"` with both `one_to_one` and `many_to_many` matching
- if specified, every evaluated span must have a
  non-empty severity with a configured weight
- cannot be combined with a non-zero `severity_penalty`

Optional `source_texts` and `target_texts` can be supplied to validate that span
offsets fit within the corresponding segment text:

```python
compute(
    predictions,
    references,
    source_texts=["source sentence"],
    target_texts=["translated sentence"],
)
```

## Results

`compute` returns a `MetricResult` with:

- `precision`
- `recall`
- `f_score`
- `details`
- `config`

`details` is either count-style or side-score-style. Count-style results expose
real `tp`, `fp`, and `fn` counts. Side-score results expose the precision and
recall numerators and denominators directly; this is used for `MPP` and for
many-to-many `MP`, where the prediction and reference sides are scored
separately.

Use `result.as_dict()` when you need a JSON-serializable representation.

## Project Structure

The public entrypoint lives in `span_mt_metrics_eval.api`, and the top-level
package re-exports the main user-facing objects for concise imports. Internals
are split by responsibility:

- `options.py`: supported metric, matching, and averaging options
- `spans.py`: `ErrorSpan` plus span side/severity normalization
- `results.py`: result details, score component containers, and config objects
- `input.py`: flat/nested input normalization and optional text-bound checks
- `validation.py`: option and severity validation
- `matching/`: pair scoring, greedy matching, optimal matching, and assignment fallback
- `scoring/`: one-to-one components, many-to-many coverage components, and aggregation
