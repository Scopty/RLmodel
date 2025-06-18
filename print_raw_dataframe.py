import pandas as pd

def print_sample_dataframe():
    # Create a sample DataFrame with timestamps
    data = {
        'datetime': pd.date_range(start='2025-01-01 09:30:00', periods=10, freq='1h'),
        'open': [100 + i for i in range(10)],
        'high': [101 + i for i in range(10)],
        'low': [99 + i for i in range(10)],
        'close': [100.5 + i for i in range(10)],
        'volume': [1000 + (i * 100) for i in range(10)]
    }
    df = pd.DataFrame(data)
    
    print("Raw input DataFrame:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    print_sample_dataframe()
