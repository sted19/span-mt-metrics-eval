# Span-Level Machine Translation Meta-Evaluation

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2603.19921-b31b1b.svg)](https://arxiv.org/abs/2603.19921)
[![MT Evaluation](https://img.shields.io/badge/MT-span--level%20meta--evaluation-0f766e.svg)](https://arxiv.org/abs/2603.19921)
[![Package](https://img.shields.io/badge/package-span--mt--metrics--eval-4b5563.svg)](pyproject.toml)

This repository implements the meta-evaluation measures discussed in the arXiv paper
[Span-Level Machine Translation Meta-Evaluation](https://arxiv.org/abs/2603.19921). Span-level precision, recall, and F-score are used to measure the evaluation capabilities of machine translation auto-evaluators.

The recommended (and default) measure is MPP (Match with Partial overlap and Partial credit) with micro-averaging. 

This repository also extends MPP with severity weighting, which reweights precision and recall based on error severity.

## Installation

From a local checkout:

```bash
python -m pip install .
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

result = compute(predictions, references)

print(result.precision, result.recall, result.f_score)
print(result.details.as_dict())
```

Span offsets are half-open character offsets: `start` is included and `end` is
excluded, matching normal Python slicing.

## Input Format

`compute` accepts annotations for one segment:

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

Dictionary spans use `start`, `end`, and optionally `side`. If `side` is omitted,
it defaults to `target`. Use `side="source"` for source-side errors and
`side="target"` for target-side errors.

Extra dictionary fields, such as `category`, are accepted and ignored by the
metric calculation. `severity` is optional by default, but it is used when
`severity_penalty` is greater than zero or `severity_weights` are provided.
Severity labels are stripped and lowercased before comparison.

## Defaults

The quick-start call uses these defaults:

```python
compute(
    predictions,
    references,
    measure="MPP",
    matching="one_to_one",
    matching_algorithm="optimal",
    averaging="micro",
    severity_penalty=0.0,
    severity_weights=None,
    source_texts=None,
    target_texts=None,
)
```

Default span behavior:

- `side` defaults to `target` when omitted.
- `severity` defaults to missing and has no effect unless severity options are enabled.

## Customization

Choose a measure with `measure`:

- `EM` requires exact side, start, and end matches.
- `MP` gives binary credit for any same-side overlap.
- `MPP` gives partial credit by overlap proportion and is the default.
- `WMT25` scores based on overlapping character counts.

Choose a matching strategy with `matching`:

- `one_to_one` pairs each prediction and reference at most once.
- `many_to_many` scores all same-side coverage without explicit pair assignment.

For `one_to_one`, choose the assignment method with `matching_algorithm`:

- `optimal` maximizes the metric objective and is the default.
- `greedy` uses deterministic greedy matching.

Choose corpus aggregation with `averaging`:

- `micro` aggregates score components across segments before computing the final score.
- `macro` computes each segment score first, then averages segment scores.

Customize severity handling with `severity_penalty` or `severity_weights`:

```python
compute(
    predictions,
    references,
    measure="MPP",
    severity_weights={"minor": 0.2, "major": 1.0},
)
```

`severity_penalty` must be a float in `[0.0, 1.0]`. Values greater than `0.0`
require every span to include a non-empty `severity`; mismatched severities
receive `1 - severity_penalty` credit.

`severity_weights` maps severity labels to finite non-negative weights. It is
supported for `measure="MPP"`, requires every evaluated span to have a
configured severity, and cannot be combined with a non-zero `severity_penalty`.

Optional `source_texts` and `target_texts` can validate that offsets fit inside
the corresponding segment text:

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

## Citation

If you use this repository, please cite the span-level MT meta-evaluation paper:

```bibtex
@misc{perrella2026spanlevelmachinetranslationmetaevaluation,
  title = {Span-Level Machine Translation Meta-Evaluation},
  author = {Stefano Perrella and Eric Morales Agostinho and Hugo Zaragoza},
  year = {2026},
  eprint = {2603.19921},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  doi = {10.48550/arXiv.2603.19921},
  url = {https://arxiv.org/abs/2603.19921},
}
```
