# Stat Kit
# Helper and wrapper functions for data cleaning, display and statistical analysis.
# Ahmed Elbokl (ahmed.elbokl@med.asu.edu.eg), 2023

# Imports
import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from .helpers import percent, plus_minus, print_title, print_mean_std, print_median_iqr

sns.set()

# test successful import
def version():
    """
    Returns the version of the package
    """
    return "1.1.2"


# Drop columns more conveniently
def drop_columns(
    df, cols_to_drop
):  # Ignores missing columns and reports them. cols_to_drop is a python list of column names that need to be dropped.
    """
    Drops columns from a dataframe smoothly as it ignores missing columns (but reports them).
    """
    print("Number of columns to be dropped: {}".format(len(cols_to_drop)))
    missing_cols = [col for col in cols_to_drop if col not in df.columns]
    if len(missing_cols) > 0:
        print(
            "There are {} missing columns that cannot be dropped: {}".format(
                len(missing_cols), missing_cols
            )
        )
        cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        print("Dropping {} columns instead ..".format(len(cols_to_drop)))

    df.drop(cols_to_drop, axis=1, inplace=True)
    print("Done.")
    return cols_to_drop


# Lookup column names in a dataframe with partial match
def lookup_columns(search_string, df):
    """
    Looks up column names in a dataframe with partial match.
    """
    return [col for col in df.columns if search_string in col]


# compare two column lists to find common and different columns
def compare_columns(list1, list2):
    """
    Compares two column lists to find common and different columns.
    Returns three lists: common columns, only in list1, only in list2.
    """
    set1 = set(list1)
    set2 = set(list2)
    common = list(set1.intersection(set2))
    only_in_list1 = list(set1 - set2)
    only_in_list2 = list(set2 - set1)
    return common, only_in_list1, only_in_list2


# percent, plus_minus, and print_title are imported from helpers


## print_mean_std and print_median_iqr are imported from helpers


# Print Missing Value count and percentage in a pandas series/column
def print_missing(series):  ## Dependent on percent()
    """
    Returns missing value count and percentage of a series printed
    """
    # Count the missing values
    missing_count = series.isna().sum()
    # Calculate the percentage of missing values
    missing_percent = percent(missing_count, len(series))
    # Print the results
    print(
        "Missing values in '" + series.name + "':",
        missing_count,
        "(" + str(missing_percent) + "%)",
    )

# Suggest de-identification of columns in a dataframe
def suggest_deidentification(df: pd.DataFrame):
    """
    Suggests columns that may need de-identification based on common identifiers.
    """
    identifiers = [
        "name",
        "address",
        "phone",
        "email",
        "ssn",
        "dob",
        "date of birth",
        "patient id",
        "mrn",
        "medical record number",
        "zip code",
        "zip",
        "city",
        "state",
        "country",
    ]

    cols_to_deidentify = []
    for col in df.columns:
        for identifier in identifiers:
            if identifier in col.lower():
                cols_to_deidentify.append(col)
                break
    if len(cols_to_deidentify) > 0:
        print("Columns that may need de-identification:")
        for col in cols_to_deidentify:
            print("- " + col)
    else:
        print("No columns found that may need de-identification.")
