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

After this directory is pushed to GitHub, it can be installed directly from the
repository:

```bash
python -m pip install "span-mt-metrics-eval @ git+https://github.com/sted19/span-mt-metrics-eval.git"
```

The package depends on NumPy. SciPy is optional; when installed, it is used for
faster optimal one-to-one assignment. Without SciPy, the package uses its bundled
pure-Python fallback:

```bash
python -m pip install ".[speed]"
```

## Quick Start

```python
from span_mt_metrics_eval import compute

predictions = [
    {"start": 8, "end": 13, "side": "target"},
]
references = [
    {"start": 8, "end": 13, "side": "target"},
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
print(result.counts.as_dict())
```

Span offsets are half-open character offsets: `start` is included and `end` is
excluded, matching normal Python slicing.

## Input Format

`compute` accepts either one segment:

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

Extra dictionary fields, such as `category` or `severity`, are accepted and
ignored by the metric calculation.

## Computing All Metrics

Pass `measure="all"` to compute every metric with the same matching and averaging
configuration:

```python
results = compute(predictions, references, measure="all")

for name, metric_result in results.items():
    print(name, metric_result.as_dict())
```

## Options

`measure`:

- `EM`
- `MP`
- `WMT23`
- `MPP`
- `all`

`matching`:

- `one_to_one`
- `many_to_many`

`matching_algorithm` for one-to-one matching:

- `optimal`
- `greedy`

`averaging`:

- `micro`
- `macro`

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
- `counts`
- `segments_tp_counts`
- `config`

Use `result.as_dict()` when you need a JSON-serializable representation.

## Tests

Run the package tests with:

```bash
python -m pytest
```

Some parity tests compare this standalone package against the original paper
repository. They run automatically when a sibling `span-mt-metaeval` checkout is
available and are skipped in a standalone clone.
