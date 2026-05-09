"""
Shared helper utilities used across analytikit modules.
"""

from typing import Tuple
import numpy as np
import scipy.stats as stats

def percent(nom, denom):
    """
    Returns the percentage of nom/denom rounded to 2 decimals.
    """
    return round((nom / denom) * 100, 2)


def plus_minus():
    """
    Returns a plus/minus sign.
    """
    return "±"


def print_title(string: str):
    """
    Print a simple titled header to stdout.
    """
    number_of_dashes = len(string) + 4
    print(
        "\n"
        + "-" * number_of_dashes
        + "\n"
        + "| "
        + string
        + " |"
        + "\n"
        + "-" * number_of_dashes
        + "\n"
    )


# --- Statistical descriptive helpers (shared) ---
def mean_ci(series) -> Tuple[float, float]:
    """
    95% CI for the mean using Student's t interval.
    """
    mean = series.mean()
    std = series.std()
    ci = stats.t.interval(0.95, len(series) - 1, loc=mean, scale=std / np.sqrt(len(series)))
    return ci[0], ci[1]


def median_ci(series, num_samples: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Bootstrap CI for the median.
    """
    medians = []
    for _ in range(num_samples):
        sample = np.random.choice(series, size=len(series), replace=True)
        medians.append(np.median(sample))
    lower = np.percentile(medians, 100 * (alpha / 2))
    upper = np.percentile(medians, 100 * (1 - alpha / 2))
    return lower, upper


def print_mean_std(label: str, series) -> None:
    """
    Print mean ± SD and 95% CI for a numeric series.
    """
    series_mean = round(series.mean(), 2)
    series_std = round(series.std(), 2)
    lower, upper = mean_ci(series)
    print(label + " mean (± std):", series_mean, "(" + "± " + str(series_std) + ")")
    print(f"95% CI: [{lower:.2f}, {upper:.2f}]")


def print_median_iqr(label: str, series) -> None:
    """
    Print median and IQR with a bootstrap 95% CI for the median.
    """
    series_median = round(series.median(), 2)
    q1 = round(series.quantile(0.25), 2)
    q3 = round(series.quantile(0.75), 2)
    ci_lower, ci_upper = median_ci(series)
    print(label + " median:", series_median, "(" + "IQR = " + str(q1) + "-" + str(q3) + ")")
    print(f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
