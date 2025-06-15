import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Plot histograms for numerical features
def plot_numerical_distributions(df: pd.DataFrame, numeric_cols: list, bins: int = 30):
    for col in numeric_cols:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col].dropna(), bins=bins, kde=True)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Plot bar charts for categorical features
def plot_categorical_distributions(df: pd.DataFrame, categorical_cols: list, top_n: int = 10):
    for col in categorical_cols:
        plt.figure(figsize=(8, 4))
        value_counts = df[col].value_counts(dropna=False).head(top_n)
        sns.barplot(x=value_counts.index.astype(str), y=value_counts.values)
        plt.title(f"Top {top_n} Categories in '{col}'")
        plt.xticks(rotation=45)
        plt.ylabel("Count")
        plt.xlabel(col)
        plt.tight_layout()
        plt.grid(True)
        plt.show()

def analyze_monthly_changes_by_zip(df: pd.DataFrame):
    """
    Analyze the monthly changes in TotalPremium and TotalClaims as a function of ZipCode.
    """
    # 1. Group by month and zip code
    grouped = (
        df.groupby(['PostalCode', 'TransactionMonth'])[['TotalPremium', 'TotalClaims']]
        .sum()
        .reset_index()
        .sort_values(by=['PostalCode', 'TransactionMonth'])
    )

    # 2. Calculate monthly change within each zip code
    grouped['PremiumChange'] = grouped.groupby('PostalCode')['TotalPremium'].diff()
    grouped['ClaimsChange'] = grouped.groupby('PostalCode')['TotalClaims'].diff()

    # Drop NaNs from first diffs
    grouped_clean = grouped.dropna(subset=['PremiumChange', 'ClaimsChange'])

    # 3. Scatter plot: Premium change vs Claims change
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=grouped_clean,
        x='PremiumChange',
        y='ClaimsChange',
        alpha=0.5
    )
    plt.title("Monthly Changes in Premium vs Claims by ZipCode")
    plt.xlabel("Monthly Change in Total Premium")
    plt.ylabel("Monthly Change in Total Claims")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 4. Correlation
    corr = grouped_clean[['PremiumChange', 'ClaimsChange']].corr()
    plt.figure(figsize=(5, 4))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation of Monthly Changes: Premium vs Claims")
    plt.tight_layout()
    plt.show()

def plot_geographic_trends(df: pd.DataFrame, geo_col: str = "Province"):
    """
    Plot trends in CoverType, CoverGroup, CoverCategory, Premium, and Make across geographic regions.
    
    Parameters:
    - df: The main DataFrame
    - geo_col: The geographic column to group by (default is 'Province')
    """
    # Set style
    sns.set(style="whitegrid")

    # Plot 1: Average Premium per Province
    plt.figure(figsize=(10, 6))
    premium_by_geo = df.groupby(geo_col)["TotalPremium"].mean().sort_values(ascending=False)
    sns.barplot(x=premium_by_geo.values, y=premium_by_geo.index)
    plt.title(f"Average Total Premium by {geo_col}")
    plt.xlabel("Average Premium")
    plt.ylabel(geo_col)
    plt.tight_layout()
    plt.show()

    # Plot 2: Count of CoverType per Province
    plt.figure(figsize=(12, 6))
    cover_counts = df.groupby([geo_col, "CoverType"]).size().reset_index(name="Count")
    sns.barplot(data=cover_counts, x=geo_col, y="Count", hue="CoverType")
    plt.title(f"CoverType Distribution by {geo_col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot 3: Most Common Vehicle Makes per Province (Top 5)
    top_makes = df["make"].value_counts().head(5).index
    filtered_df = df[df["make"].isin(top_makes)]
    plt.figure(figsize=(12, 6))
    make_counts = filtered_df.groupby([geo_col, "make"]).size().reset_index(name="Count")
    sns.barplot(data=make_counts, x=geo_col, y="Count", hue="make")
    plt.title(f"Top 5 Vehicle Makes by {geo_col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_boxplots_for_numerical(df: pd.DataFrame, columns: list[str]) -> None:
    """
    Plots boxplots for given numerical columns to detect outliers.

    Parameters:
    - df: pandas DataFrame containing the data
    - columns: list of column names (numerical) to plot boxplots for
    """
    sns.set(style="whitegrid")
    
    for col in columns:
        if col not in df.columns:
            print(f"Column '{col}' not found in DataFrame.")
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"Column '{col}' is not numeric and will be skipped.")
            continue

        plt.figure(figsize=(10, 4))
        sns.boxplot(x=df[col], color='skyblue')
        plt.title(f"Box Plot of {col}")
        plt.xlabel(col)
        plt.tight_layout()
        plt.show()

def plot_loss_ratio_by_province(df: pd.DataFrame):
    df['LossRatio'] = df['TotalClaims'] / df['TotalPremium']
    province_loss = df.groupby('Province')['LossRatio'].mean().sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=province_loss.index, y=province_loss.values)
    plt.xticks(rotation=45)
    plt.title("Average Loss Ratio by Province")
    plt.ylabel("Average Loss Ratio")
    plt.xlabel("Province")
    plt.tight_layout()
    plt.show()


def plot_claims_by_vehicle_make(df: pd.DataFrame, top_n: int = 10):
    make_claims = (
        df.groupby('make')['TotalClaims']
        .mean()
        .dropna()
        .sort_values(ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(x=make_claims.index, y=make_claims.values)
    plt.xticks(rotation=45)
    plt.title(f"Top {top_n} Vehicle Makes by Avg. Total Claims")
    plt.ylabel("Average Claim Amount")
    plt.xlabel("Vehicle Make")
    plt.tight_layout()
    plt.show()


def plot_monthly_claim_trends(df: pd.DataFrame):
    df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])
    monthly = (
        df.groupby(df['TransactionMonth'].dt.to_period('M'))
        .agg(
            TotalClaims=('TotalClaims', 'sum'),
            ClaimCount=('TotalClaims', lambda x: (x > 0).sum())
        )
    ).reset_index()
    monthly['TransactionMonth'] = monthly['TransactionMonth'].dt.to_timestamp()

    plt.figure(figsize=(14, 6))
    ax1 = sns.lineplot(data=monthly, x='TransactionMonth', y='TotalClaims', label='Total Claim Amount', color='blue')
    ax2 = sns.lineplot(data=monthly, x='TransactionMonth', y='ClaimCount', label='Claim Frequency', color='orange')

    plt.title("Monthly Trends: Claim Severity and Frequency")
    plt.xlabel("Month")
    plt.ylabel("Amount / Count")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()