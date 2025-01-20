# StatKit

A Python package for streamlined data cleaning, statistical analysis, and visualization. StatKit provides wrapper functions that make common data analysis tasks more convenient and robust.

## Features

- Drop columns with graceful handling of missing columns
- Search columns with partial name matching
- Calculate and format percentages
- Print formatted statistical summaries
- Automated statistical testing with appropriate test selection
- Visualization of normality tests
- Comprehensive group comparisons with effect size calculations

## Installation

Copy the repo to your local environment.

## Dependencies

- pandas
- numpy
- scipy
- seaborn
- matplotlib

## Usage Guide

### Data Cleaning Functions

#### `drop_columns(df, cols_to_drop)`
Safely drops columns from a DataFrame, handling missing columns gracefully.

```python
from statkit import drop_columns

# Example
cols_to_remove = ['column1', 'column2', 'non_existent_column']
drop_columns(df, cols_to_remove)
# Will report missing columns and drop only existing ones
```

#### `lookup_columns(search_string, df)`
Search for column names containing a specific string.

```python
matches = lookup_columns('age', df)
# Returns all column names containing 'age'
```

### Formatting Functions

#### `percent(nom, denom)`
Calculate percentage with consistent rounding.

```python
success_rate = percent(successful_cases, total_cases)
# Returns rounded percentage
```

#### `plus_minus()`
Returns a standardized plus-minus symbol (±) for consistency in reporting.

#### `print_title(string)`
Creates formatted section headers for output clarity.

```python
print_title("Analysis Results")
# Outputs:
# ----------------
# | Analysis Results |
# ----------------
```

### Statistical Summary Functions

#### `print_mean_std(label, series)`
Prints formatted mean and standard deviation.

```python
print_mean_std("Age", df['age'])
# Output: Age mean (± std): 45.2 (± 15.3)
```

#### `print_median_iqr(label, series)`
Prints formatted median and interquartile range.

```python
print_median_iqr("BMI", df['bmi'])
# Output: BMI median: 24.5 (IQR = 21.3-27.8)
```

#### `print_missing(series)`
Reports missing value counts and percentages.

```python
print_missing(df['blood_pressure'])
# Output: Missing values in 'blood_pressure': 15 (3.5%)
```

### Statistical Comparison Function

#### `compare(groups, group_labels=None, alpha=0.05, categorical_limit=20, forced_test=None)`
Comprehensive statistical comparison between groups with automatic test selection.

Key features:
- Automatically determines appropriate statistical test
- Performs normality testing with visualization
- Handles both continuous and categorical data
- Calculates effect sizes when applicable
- Provides formatted output with clear interpretation

```python
# Example for continuous data
group1 = df[df['treatment'] == 'A']['outcome']
group2 = df[df['treatment'] == 'B']['outcome']
compare([group1, group2], 
        group_labels=['Treatment A', 'Treatment B'])

# Example for categorical data
compare([df['category1'], df['category2']])
```

The function automatically:
1. Tests for normality using Shapiro-Wilk test
2. Generates Q-Q plots for visual normality assessment
3. Selects appropriate statistical test:
   - For continuous data:
     - Two groups: t-test (normal) or Mann-Whitney U (non-normal)
     - More than two groups: ANOVA (normal) or Kruskal-Wallis (non-normal)
   - For categorical data:
     - Chi-square test (when assumptions met)
     - Fisher's exact test (for 2x2 tables when Chi-square assumptions not met)
4. Provides effect size calculations when applicable
5. Outputs formatted results with clear interpretation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

Ahmed Elbokl (ahmed.elbokl@med.asu.edu.eg)
