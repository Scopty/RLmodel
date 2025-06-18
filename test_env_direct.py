"""
Minimal script to test TradingEnv in isolation.
Run with: python test_env_direct.py
"""
import pandas as pd
import numpy as np
from trading_env import TradingEnv

# Create a small dummy DataFrame for testing
def create_test_data():
    data = {
        'datetime': pd.date_range(start='2025-06-13 09:30', periods=10, freq='1min').strftime('%m/%d/%y %H:%M'),
        'open': [100 + i for i in range(10)],
        'high': [101 + i for i in range(10)],
        'low': [99 + i for i in range(10)],
        'close': [100.5 + i for i in range(10)],
        'volume': [1000] * 10
    }
    return pd.DataFrame(data)

def main():
    print("=== Starting TradingEnv Test ===")
    
    # Create test data
    df = create_test_data()
    print(f"Created test data with shape: {df.shape}")
    
    try:
        # Initialize environment
        print("\n=== Initializing TradingEnv ===")
        env = TradingEnv(df, debug=True, test_mode=True)
        
        # Test reset
        print("\n=== Testing reset() ===")
        obs, _ = env.reset()
        print(f"Reset successful. Observation keys: {list(obs.keys())}")
        
        # Test step
        print("\n=== Testing step() ===")
        action = (0, np.array([0.05], dtype=np.float32))  # HOLD action with stoploss
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step successful. Reward: {reward}, Done: {done}")
        
    except Exception as e:
        print(f"\n!!! ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Test Complete ===")
    print("Check 'training_debug.log' for debug output")

if __name__ == "__main__":
    main()
