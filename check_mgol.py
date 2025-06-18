import pandas as pd

def check_mgol_data():
    try:
        # Read the MGOL.csv file
        df = pd.read_csv('MGOL.csv')
        
        print("\nMGOL.csv Structure:")
        print("=" * 50)
        print(f"Total Rows: {len(df)}")
        print(f"Columns: {df.columns.tolist()}")
        
        print("\nFirst 10 Rows:")
        print("-" * 50)
        print(df.head(10).to_string())
        
        print("\nData Types:")
        print("-" * 50)
        print(df.dtypes)
        
        # Check date range
        print("\nDate Range:")
        print("-" * 50)
        print(f"Start: {df['datetime'].min()}")
        print(f"End: {df['datetime'].max()}")
        
        # Check for missing values
        print("\nMissing Values:")
        print("-" * 50)
        print(df.isnull().sum())
        
    except Exception as e:
        print(f"Error reading MGOL.csv: {e}")

if __name__ == "__main__":
    check_mgol_data()
