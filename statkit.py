# Stat Kit
# Helper and wrapper functions for data cleaning, display and statistical analysis.
# Ahmed Elbokl (ahmed.elbokl@med.asu.edu.eg), 2023

# Imports
import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

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


# Lookup column names in a dataframe with partial match
def lookup_columns(search_string, df):
    """
    Looks up column names in a dataframe with partial match.
    """
    return [col for col in df.columns if search_string in col]


# Percent
def percent(nom, denom):
    """
    Returns the percentage of nom/denom
    """
    return round((nom / denom) * 100, 2)


# Plus Minus sign
def plus_minus():
    """
    Returns a plus minus sign
    """
    return "±"


# Print Title
def print_title(string):
    """
    Returns a title
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


# Print Mean and SD
def print_mean_std(label, series):
    """
    Returns mean and standard deviation of a series printed
    """
    # Round the mean and std to 2 decimal places
    series_mean = round(series.mean(), 2)
    series_std = round(series.std(), 2)
    print(label + " mean (± std):", series_mean, "(" + "± " + str(series_std) + ")")


# Print Median and IQR
def print_median_iqr(label, series):
    """
    Returns median and IQR of a series printed
    """
    # Round the median and IQR to 2 decimal places
    series_median = round(series.median(), 2)
    # series_iqr = round(series.quantile(0.75) - series.quantile(0.25), 2)
    q1 = round(series.quantile(0.25), 2)
    q3 = round(series.quantile(0.75), 2)
    print(
        label + " median:",
        series_median,
        "(" + "IQR = " + str(q1) + "-" + str(q3) + ")",
    )


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


# Compare two series/columns
def compare(
    groups,
    group_labels=None,
    alpha=0.05,
    categorical_limit=20,
    forced_test=None,
    force_normality=False,
    force_non_normality=False,
):  # Depends on print_mean_std()
    # Note: this function does not handle the independent variable (grouping variable). It assumes that the series are already grouped.
    # groups: a list of pandas series/columns to be compared
    # group_labels: a list of labels for the groups. If not provided, the group names + group number will be used as labels. The list of labels has to correspond to the list of groups.
    # forced_test: if provided, the function will use the test provided instead of determining it automatically. The test has to be a string and has to be one of the following: "ttest", "anova", "mannwhitney", "kruskalwallis", "chisquare", 'fisherexact'
    # alpha: the alpha value to be used for the test. Default is 0.05
    # categorical_limit: the number of unique values in a series/column that determines whether the data is continuous or categorical. If it is below this limit, data will be treated as categorical. Default is 20.

    # Declare vars
    test_name = ""
    test_statistic_sign = ""
    line1 = ""
    line2 = ""
    effect_size = {}

    # If group_labels is not provided, use the group names + group number as labels
    if group_labels is None:
        group_labels = []
        index = 0
        for group in groups:
            index += 1
            group_labels.append(group.name + " (Group " + str(index) + ")")

    # Drop isna values from each group
    index = 0
    for group in groups:
        missing_in_group = group.isna().sum()
        if missing_in_group > 0:
            print(
                "Dropping",
                group.isna().sum(),
                'missing values from "' + group_labels[index] + '".',
            )
            group.dropna(inplace=True)
        else:
            print('No missing values in "' + group_labels[index] + '" to drop.')
        index += 1
    print()
    print("alpha is set to", alpha)

    # Check if the data in groups is continuous or categorical
    # Get the largest number of unique values in any of the groups ==> if less than 20, all groups are assumed to be categorical
    nunique = max([group.nunique() for group in groups])
    continuous = True if nunique > categorical_limit else False

    #### IF DATA IS CONTINUOUS ####
    if continuous:
        print("Data is assumed to be continuous based on number of unique values.")

        ### Normality Test ###
        # Use Shapiro-Wilk test to check for normality of each group
        if force_non_normality:
            all_normal = False
            print("The compare function is forced to use non-parametric tests.")
        elif force_normality:
            all_normal = True
            print("The compare function is forced to use parametric tests.")
        else:
            all_normal = True

        index = 0

        for group in groups:
            if not force_normality and not force_non_normality:
                print_title("Test for Normality")
                statistic, p_value = stats.shapiro(group)
                statistic = round(statistic, 3)
                p_value = round(p_value, 3)
                # print group label for this group
                print("Test for normality (Shapiro-Wilk) for", group_labels[index])
                print("-------------------------------------------------")
                print("Statistic:", statistic)
                print("p-value:", p_value)
                if p_value > alpha:
                    print("Data is normally distributed.")
                else:
                    print("Data is not normally distributed.")
                    all_normal = False

            # Plot QQ
            print()
            plt.figure(figsize=(4, 3))
            stats.probplot(group, dist="norm", plot=plt)
            plt.title("Normal Q-Q Plot for " + group_labels[index])
            plt.show()
            print("-------------------------------------------------")
            index += 1
            if index < len(groups):
                print()

        # Print mean and standard deviation for each group if all groups are normally distributed
        print_title("Central Tendency")
        if all_normal:
            index = 0
            for group in groups:  # if you want to get index
                print_mean_std(
                    group_labels[index],
                    group,
                )

                # Print n = number of observations in each group
                print("n:", len(group))
                if index < len(groups):
                    print()
                index += 1

        # Print median and IQR for each group if not all groups are normally distributed
        else:
            index = 0
            for group in groups:
                print_median_iqr(group_labels[index], group)

                # Print n = number of observations in each group
                print("n:", len(group))
                if index < len(groups):
                    print()
                index += 1

        # If all groups are normally distributed, and we have 2 groups/group,use independent t-test
        if all_normal and len(groups) == 2:
            test_name = "Independent t-test"
            test_statistic_sign = "t"
            line1 = "All data is normally distributed."

            # Use t-test to compare all group
            statistic, p_value = stats.ttest_ind(groups[0], groups[1])
            statistic = round(statistic, 3)
            p_value = round(p_value, 3)

            # Compute the pooled standard deviation
            n1 = len(groups[0])
            n2 = len(groups[1])
            df = n1 + n2 - 2
            pooled_std = np.sqrt(
                (
                    (n1 - 1) * np.std(groups[0], ddof=1) ** 2
                    + (n2 - 1) * np.std(groups[1], ddof=1) ** 2
                )
                / df
            )

            # Compute Cohen's d
            cohen_d = (np.mean(groups[0]) - np.mean(groups[1])) / pooled_std

            effect_size["label"] = "Cohen's d"
            effect_size["value"] = cohen_d

        # If all groups are normally distributed, and we have more than 2 groups/group, use ANOVA
        elif all_normal and len(groups) > 2:
            test_name = "ANOVA"
            test_statistic_sign = "F"
            line1 = "Using ANOVA because there are more than 2 groups."

            # Use ANOVA to compare all group
            statistic, p_value = stats.f_oneway(*groups)
            statistic = round(statistic, 3)
            p_value = round(p_value, 3)

        # If all group are not normally distributed, and we have 2 groups/group, use Mann-Whitney U test
        elif not all_normal and len(groups) == 2:
            test_name = "Mann-Whitney U"
            test_statistic_sign = "Statistic"
            line1 = "Not all data is normally distributed."

            # Use Mann-Whitney U test to compare all group
            statistic, p_value = stats.mannwhitneyu(groups[0], groups[1])
            p_value = round(p_value, 3)

            # Compute the effect size (Cohen's U3)
            n1 = len(groups[0])
            n2 = len(groups[1])

            R1 = (n1 * (n1 + n2 + 1)) / 2
            R2 = n1 * n2 - R1

            u3 = (R1 - R2) / n1
            # u3 = (statistic - (n1 * (n1 + 1)) / 2) / (n1 * n2)

            effect_size["label"] = "Cohen's U3"
            effect_size["value"] = u3

        # If all group are not normally distributed, and we have more than 2 groups/group, use Kruskal-Wallis H test
        elif not all_normal and len(groups) > 2:
            test_name = "Kruskal-Wallis H test"
            test_statistic_sign = "Statistic"
            line1 = "Not all data is normally distributed."

            # Use Kruskal-Wallis H test to compare all group
            statistic, p_value = stats.kruskal(*groups)
            statistic = round(statistic, 3)
            p_value = round(p_value, 3)

    #### IF DATA IS CATEGORICAL ####
    if not continuous:
        # Check Chi-squared assumptions to determine which test to use: Chi-squared or Fisher's exact test.
        # Use Chi-squared test to check for normality of each
        print("Data is assumed to be categorical based on number of unique values.")
        contingency_table = pd.crosstab(*groups)

        # Let congtingency table show percentages between brackets after counts (for printing only)
        contingency_table_with_percent = (
            contingency_table.astype(str)
            + " ("
            + (contingency_table / contingency_table.sum() * 100).round(2).astype(str)
            + "%)"
        )

        # Add lines to the table
        line = "---------------------------------------------"

        # Print Table
        print()
        print(line)
        print(contingency_table_with_percent)
        print(line)
        print("* percent values are per column")

        print()

        # Calculate expected counts
        statistic, p_value, dof, expected_frequencies = stats.chi2_contingency(
            contingency_table
        )

        # Check if Chi-suqare assumptions are met
        # Assumption 1: At least 80% of the cells have expected frequencies >= 5
        cells_five_or_larger = np.sum(expected_frequencies >= 5)
        percent_of_five_or_greater = cells_five_or_larger / contingency_table.size

        # Assumption 2: No cell has expected frequencies less than 1
        cells_less_than_one = np.sum(expected_frequencies < 1)

        # Check assumptions
        if percent_of_five_or_greater >= 0.8 and cells_less_than_one == 0:
            print("Chi-squared assumptions are met")

            # Perform test
            test_name = "Chi-squared"
            test_statistic_sign = "Chi2"

            statistic = round(statistic, 3)
            p_value = round(p_value, 3)

        else:
            print("Chi-squared assumptions are not met")

            # Check if it's a 2x2 contingency table
            is_2x2_table = contingency_table.shape == (2, 2)
            if is_2x2_table:
                # Perform Fisher's exact test
                statistic, p_value = stats.fisher_exact(contingency_table)
                p_value = round(p_value, 3)

                test_name = "Fisher's exact test"
                test_statistic_sign = "Odd's ratio"

            else:
                print(
                    "The contingency table is not 2x2. Cannot perform Fisher's exact test"
                )
                print("Using Chi-squared test instead")
                # Perform test
                test_name = "Chi-squared"
                test_statistic_sign = "Chi2"

                statistic = round(statistic, 3)
                p_value = round(p_value, 3)

    #### Print Output ####
    print_title(test_name)
    if line1 != "":
        print(line1)
    if line2 != "":
        print(line2)

    print(test_statistic_sign + ":", statistic)
    print("p-value:", p_value)
    print()

    if p_value <= alpha:
        print("There is a significant difference between groups.")
    else:
        print("There is no significant difference between groups.")

    # Print effect size
    # if len(effect_size) > 0:
    #     print("Effect size (" + effect_size["label"] + "):", effect_size["value"])
