import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["Province", "TotalClaims", "TotalPremium"])
    df["HasClaim"] = df["TotalClaims"] > 0
    return df

# --- Prepare Group A and Group B ---
def prepare_ab_groups(df: pd.DataFrame, group_a: str, group_b: str, feature_col: str = "Province") -> pd.DataFrame:
    df_ab = df[df[feature_col].isin([group_a, group_b])].copy()
    df_ab["Group"] = df_ab[feature_col].apply(lambda x: "A" if x == group_a else "B")
    return df_ab

# --- Balance Checks ---
def check_categorical_balance(df_ab, col: str) -> float:
    table = pd.crosstab(df_ab['Group'], df_ab[col])
    _, p, _, _ = chi2_contingency(table)
    return p

def check_numeric_balance(df_ab, col: str) -> float:
    a = df_ab[df_ab['Group'] == 'A'][col]
    b = df_ab[df_ab['Group'] == 'B'][col]
    return ttest_ind(a, b, equal_var=False).pvalue

# --- Claim Metrics ---
def calculate_claim_frequency(df: pd.DataFrame) -> float:
    return df['HasClaim'].sum() / len(df)

def calculate_claim_severity(df: pd.DataFrame, claim_col='TotalClaims') -> float:
    """
    Calculate the average claim severity among policies that had a claim.
    """
    if claim_col not in df.columns:
        raise KeyError(f"'{claim_col}' column not found in DataFrame.")
    return df.loc[df['HasClaim'] == 1, claim_col].mean()

# --- Statistical Tests ---
def test_claim_frequency(df_ab: pd.DataFrame) -> float:
    contingency = pd.crosstab(df_ab['Group'], df_ab['HasClaim'])
    _, p, _, _ = chi2_contingency(contingency)
    return p

def test_claim_severity(df_ab):
    """
    Perform a t-test (or Mann-Whitney) on claim severity between Group A and Group B.
    Only use rows where HasClaim == 1.
    """
    # Filter rows with claims
    df_claims = df_ab[df_ab["HasClaim"] == 1]

    # Extract claim amounts (e.g. TotalClaims)
    severity_a = df_claims[df_claims["Group"] == "A"]["TotalClaims"]
    severity_b = df_claims[df_claims["Group"] == "B"]["TotalClaims"]

    # If data is heavily skewed, use non-parametric test
    # Otherwise t-test (assuming equal variance)
    _, p_value = ttest_ind(severity_a, severity_b, equal_var=False)

    return p_value

# --- Visualization ---
def plot_metric_by_group(df_ab: pd.DataFrame, metric_func, title: str):
    values = df_ab.groupby("Group").apply(metric_func).reset_index(name="value")
    sns.barplot(data=values, x="Group", y="value")
    plt.title(title)
    plt.ylabel("")
    plt.show()

def calculate_margin(df: pd.DataFrame) -> float:
    """
    Calculate the average profit margin: (Premium - Claims) / Premium
    """
    df = df[df["TotalPremium"] > 0]  # Avoid divide-by-zero
    margins = (df["TotalPremium"] - df["TotalClaims"]) / df["TotalPremium"]
    return margins.mean()

def test_margin_difference(df_ab: pd.DataFrame) -> float:
    """
    Perform t-test on margins between Group A and Group B
    """
    df_ab = df_ab[df_ab["TotalPremium"] > 0].copy()
    df_ab["Margin"] = (df_ab["TotalPremium"] - df_ab["TotalClaims"]) / df_ab["TotalPremium"]

    a = df_ab[df_ab["Group"] == "A"]["Margin"]
    b = df_ab[df_ab["Group"] == "B"]["Margin"]

    return ttest_ind(a, b, equal_var=False).pvalue
