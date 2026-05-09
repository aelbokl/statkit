# analytikit

analytikit is a small collection of helper utilities focused on lightweight statistical helpers and descriptive printing.

Current minimal package contents (prepared for upload):

- `stat_kit` — the main statistics utilities (version 1.1.2 as reported by `stat_kit.version()`).
- `helpers` — small shared helpers used by `stat_kit` (bootstrap CI, printing helpers, percent utility).

Note: `cleaning_kit` is present in the repository but was intentionally omitted from the minimal upload; include it if downstream projects depend on it.

## Features

- Descriptive helpers: `print_mean_std`, `print_median_iqr`, `print_missing` (from `helpers`).
- High-level comparison routines: `compare_ind`, `compare_dep` (auto-select tests, print results, optional post-hoc).
- Correlation helper: `correlate` (auto-selects Pearson/Spearman).
- Confidence-interval utilities: `compute_confidence_interval_difference` and related helpers.

## Version

The `stat_kit` module reports version `1.1.2` via `stat_kit.version()`; package metadata in `setup.py` currently uses `version='0.1.0'` for the distribution. Update `setup.py` if you want the distribution version to match the module version.

## Installation

From the package root (editable install, recommended while developing):

```bash
pip install -e .
# or install runtime deps from requirements.txt
pip install -r requirements.txt
```

## Quick examples

Recommended minimal imports for the prepared package:

```python
import pandas as pd
import analytikit

# version
print('stat_kit version:', analytikit.version())

# use exported functions directly
# compare two groups
df = pd.DataFrame({
    'group': ['A']*5 + ['B']*5,
    'score': [1,2,3,4,5,2,3,2,4,5]
})

gA = df[df['group']=='A']['score']
gB = df[df['group']=='B']['score']

analytikit.compare_ind([gA, gB], group_labels=['A','B'])

# or import modules explicitly
from analytikit import stat_kit
stat_kit.compare_ind([gA, gB], group_labels=['A','B'])
```

## Local compatibility notes

- Keeping the full `analytikit` folder in your PYTHONPATH ensures existing projects that import `analytikit.cleaning_kit` continue to work.
- If you publish only the minimal package (stat_kit + helpers), update downstream projects to avoid importing `cleaning_kit`.

## Smoke test

Create a small `smoke.py` with the example above and run:

```bash
python smoke.py
```

## Contributing

Contributions welcome. When adding features please include tests and update this README.

## Authors

Ahmed Elbokl (ahmed.elbokl@med.asu.edu.eg) and contributors.

## License

See `setup.py` for package metadata including license information.
