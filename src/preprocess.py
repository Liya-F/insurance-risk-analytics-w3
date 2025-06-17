import pandas as pd
import numpy as np
from datetime import datetime

def clean_and_prepare_data(input_path: str):
    df = pd.read_csv(input_path, sep="|", low_memory=False)

    # Drop columns with >60% missing values
    df = df.loc[:, df.isnull().mean() <= 0.6]

    # Impute missing values
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode().iloc[0])
            else:
                df[col] = df[col].fillna(df[col].mean())

    # Feature Engineering
    if 'RegistrationYear' in df.columns:
        df['VehicleAge'] = datetime.now().year - df['RegistrationYear']
    if 'SumInsured' in df.columns and 'TotalPremium' in df.columns:
        df['PremiumRatio'] = df['TotalPremium'] / (df['SumInsured'] + 1e-6)
    if 'CalculatedPremiumPerTerm' in df.columns and 'TotalClaims' in df.columns:
        df['ClaimToPremium'] = df['TotalClaims'] / (df['CalculatedPremiumPerTerm'] + 1e-6)

    # Encode categoricals (smart encoding)
    for col in df.select_dtypes(include='object').columns:
        if df[col].nunique() <= 10:
            df = pd.get_dummies(df, columns=[col], drop_first=True)
        else:
            df[col] = df[col].astype('category').cat.codes

    # Drop obviously useless identifiers
    drop_cols = ['PolicyID', 'UnderwrittenCoverID']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    return df

def save_model_datasets(df: pd.DataFrame):
    # Save cleaned full dataset
    df.to_csv("../data/cleaned_full.csv", index=False)

    # Subset 1: Claim Severity (TotalClaims > 0)
    if "TotalClaims" in df.columns:
        claims_df = df[df["TotalClaims"] > 0].copy()
        claims_df.to_csv("../data/claims_only.csv", index=False)

    # Subset 2: Classification target (HasClaim: 0/1)
    if "TotalClaims" in df.columns:
        class_df = df.copy()
        class_df["HasClaim"] = (class_df["TotalClaims"] > 0).astype(int)
        class_df.to_csv("../data/classification.csv", index=False)

    # Subset 3: Premium prediction (drop missing premium)
    if "CalculatedPremiumPerTerm" in df.columns:
        premium_df = df[df["CalculatedPremiumPerTerm"].notna()].copy()
        premium_df.to_csv("../data/premium.csv", index=False)

if __name__ == "__main__":
    input_path = "../data/MachineLearningRating_v3.txt"
    df_cleaned = clean_and_prepare_data(input_path)
    save_model_datasets(df_cleaned)
    print("✅ All datasets saved:")
    print("- cleaned_full.csv")
    print("- claims_only.csv")
    print("- classification.csv")
    print("- premium.csv")
