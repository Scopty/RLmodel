import pandas as pd
import numpy as np
from trading_env import TradingEnv

def check_time_normalization():
    # Create a sample DataFrame with timestamps
    data = {
        'datetime': pd.date_range(start='2025-01-01 09:30:00', periods=10, freq='1H'),
        'open': [100 + i for i in range(10)],
        'high': [101 + i for i in range(10)],
        'low': [99 + i for i in range(10)],
        'close': [100.5 + i for i in range(10)],
        'volume': [1000 + (i * 100) for i in range(10)]
    }
    df = pd.DataFrame(data)
    
    # Initialize environment with debug mode
    env = TradingEnv(df=df, debug=True, stoploss=True)
    
    # Print the first 10 rows with time-related columns
    print("\nFirst 10 rows of the DataFrame with time calculations:")
    print(df[['datetime', 'hour', 'minute', 'time_since_open', 'normalized_time', 
              'time_until_close', 'normalized_time_until_close']].head(10).to_string())
    
    # Print market open/close info
    print("\nMarket Open:", env.market_open)
    print("Market Close:", env.market_close)
    
    # Print expected vs actual values for first row
    first_row = df.iloc[0]
    print("\nFirst row analysis:")
    print(f"Datetime: {first_row['datetime']}")
    print(f"Time since market open (minutes): {first_row['time_since_open']}")
    print(f"Normalized time: {first_row['normalized_time']} (should be close to 0)")
    print(f"Time until close (minutes): {first_row['time_until_close']}")
    print(f"Normalized time until close: {first_row['normalized_time_until_close']} (should be close to 1)")

if __name__ == "__main__":
    check_time_normalization()
