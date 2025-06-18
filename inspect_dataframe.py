import os
import pandas as pd
from common_imports import load_data

def inspect_data():
    # Load the data using the same function as in the training script
    df, df_original = load_data()
    
    # Display basic information
    print("\n" + "="*80)
    print("DATAFRAME INSPECTION")
    print("="*80)
    
    # Print shape and columns
    print(f"\nDataFrame Shape: {df.shape}")
    print("\nColumns:")
    for col in df.columns:
        print(f"- {col}")
    
    # Print data types
    print("\nData Types:")
    print(df.dtypes)
    
    # Print first 10 rows
    print("\nFirst 10 Rows:")
    print(df.head(10).to_string())
    
    # Print basic statistics
    print("\nBasic Statistics:")
    print(df.describe().to_string())
    
    # Print datetime range and frequency
    print("\nDatetime Range:")
    print(f"Start: {df.index.min()}")
    print(f"End: {df.index.max()}")
    print(f"Duration: {df.index.max() - df.index.min()}")
    
    # Print any missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    print("\n" + "="*80)
    print("END OF INSPECTION")
    print("="*80)

if __name__ == "__main__":
    inspect_data()
