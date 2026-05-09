# analytikit

analytikit is a small Python package for practical data cleaning, descriptive summaries, and common statistical comparisons.

It currently includes the full package surface in this repository:

- `stat_kit` for statistical testing, confidence intervals, and correlation helpers.
- `cleaning_kit` for cleaning utilities and missing-data reporting.
- `helpers` for shared descriptive-statistics and formatting helpers.

## Package layout

### `stat_kit`

Main statistical utilities, including:

- `compare_ind` for comparing independent groups.
- `compare_dep` for comparing dependent or repeated-measures groups.
- `correlate` for Pearson or Spearman correlation.
- `compute_confidence_interval_difference` for mean or median difference confidence intervals.
- plotting helpers used during comparisons.

`stat_kit.version()` currently returns `1.1.2`.

### `cleaning_kit`

Data cleaning and reporting helpers, including:

- `drop_columns` to safely drop columns while reporting missing column names.
- `lookup_columns` to find columns by partial match.
- `compare_columns` to compare two column lists.
- `print_missing` to report missing counts and percentages.
- `suggest_deidentification` to flag columns that may contain identifiers.

`cleaning_kit.version()` currently returns `1.0.0`.

### `helpers`

Shared helper functions used across the package, including:

- `percent`
- `plus_minus`
- `print_title`
- `mean_ci`
- `median_ci`
- `print_mean_std`
- `print_median_iqr`

## Installation

From the package root:

```bash
pip install -e .
```

Or install dependencies first:

```bash
pip install -r requirements.txt
```

## Dependencies

Runtime dependencies currently declared in `setup.py`:

- pandas
- numpy
- scipy
- pingouin
- statsmodels
- scikit-posthocs
- seaborn
- matplotlib

## Quick start

You can import from the package root:

```python
import analytikit

print(analytikit.version())
print(analytikit.percent(25, 40))
```

Or import individual modules explicitly:

```python
from analytikit import stat_kit, cleaning_kit
```

## Usage examples

### 1. Cleaning utilities

```python
import pandas as pd
from analytikit import cleaning_kit

df = pd.DataFrame({
    'patient_id': [1, 2, 3],
    'age': [30, 42, None],
    'group': ['A', 'A', 'B']
})

cleaning_kit.print_missing(df['age'])
print(cleaning_kit.lookup_columns('age', df))
cleaning_kit.suggest_deidentification(df)
```

### 2. Independent-group comparison

```python
import pandas as pd
from analytikit import stat_kit

df = pd.DataFrame({
    'group': ['A'] * 5 + ['B'] * 5,
    'score': [1, 2, 3, 4, 5, 2, 3, 2, 4, 5]
})

group_a = df[df['group'] == 'A']['score']
group_b = df[df['group'] == 'B']['score']

stat_kit.compare_ind([group_a, group_b], group_labels=['A', 'B'])
```

### 3. Dependent-group comparison

```python
import pandas as pd
from analytikit import stat_kit

baseline = pd.Series([10, 12, 9, 11, 10], name='baseline')
followup = pd.Series([12, 13, 11, 14, 12], name='followup')

stat_kit.compare_dep([baseline, followup], group_labels=['Baseline', 'Follow-up'])
```

### 4. Correlation

```python
import pandas as pd
from analytikit import stat_kit

df = pd.DataFrame({
    'age': [21, 22, 23, 24, 25],
    'score': [60, 65, 68, 74, 80]
})

result = stat_kit.correlate(df, 'age', 'score')
print(result)
```

### 5. Confidence interval for group differences

```python
from analytikit import stat_kit

group1 = [1, 2, 3, 4, 5]
group2 = [2, 3, 4, 5, 6]

ci = stat_kit.compute_confidence_interval_difference(
    group1,
    group2,
    method='mean',
    paired=False,
)
print(ci)
```

## Import behavior

The package root currently exports:

- selected `stat_kit` functions such as `version`, `compare_ind`, `compare_dep`, `correlate`, and `compute_confidence_interval_difference`
- selected `helpers` functions such as `percent`, `mean_ci`, and descriptive print helpers
- the `cleaning_kit` module as `analytikit.cleaning_kit`

This means all of the following are valid:

```python
import analytikit
from analytikit import stat_kit
from analytikit import cleaning_kit
```

## Version note

There is currently a version difference between module and distribution metadata:

- `stat_kit.version()` returns `1.1.2`
- `cleaning_kit.version()` returns `1.0.0`
- `setup.py` declares the package version as `0.1.0`

If you want PyPI or GitHub release metadata to match the module versions, update `setup.py` before publishing a tagged release.

## Smoke test

After installation, run a quick import check:

```bash
python -c "import analytikit; from analytikit import stat_kit, cleaning_kit; print(analytikit.version(), cleaning_kit.version())"
```

## Contributing

If you add or change functionality:

- update this README
- keep examples aligned with the public API
- add tests where practical

## Authors

Ahmed Elbokl (ahmed.elbokl@med.asu.edu.eg) and contributors.

## License

See `setup.py` for package metadata including license information.
