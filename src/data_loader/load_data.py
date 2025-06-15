import pandas as pd

def load_portfolio_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="|", parse_dates=['TransactionMonth'])
    return df

def preview_top_categorical_values(df, columns, top_n=10):
    """
    Display top N most frequent values for a list of categorical columns.
    
    Parameters:
    - df: pandas DataFrame
    - columns: list of column names to inspect
    - top_n: number of top values to display per column (default: 10)
    """
    for col in columns:
        print(f"\nTop {top_n} values in '{col}':")
        print(df[col].value_counts(dropna=False).head(top_n))

def summarize_missing_values(df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """
    Returns a DataFrame summarizing missing values per column.
    threshold: only return columns with missing % greater than this.
    """
    total = df.isnull().sum()
    percent = (total / len(df)) * 100
    missing_df = pd.DataFrame({
        'missing_count': total,
        'missing_percent': percent
    }).sort_values(by='missing_percent', ascending=False)

    return missing_df[missing_df['missing_percent'] > threshold]
