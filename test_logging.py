import os
import pandas as pd
from trading_env import TradingEnv

def test_logging():
    # Clean up any existing log file
    log_file = 'training_debug.log'
    if os.path.exists(log_file):
        os.remove(log_file)
    
    print(f"Starting logging test. Log file will be created at: {os.path.abspath(log_file)}")
    
    # Create a simple DataFrame for testing
    data = {
        'open': [100, 101, 102, 101, 100],
        'high': [101, 102, 103, 102, 101],
        'low': [99, 100, 101, 100, 99],
        'close': [100.5, 101.5, 102.5, 101.5, 100.5],
        'volume': [1000, 2000, 3000, 2000, 1000],
        'datetime': pd.date_range(start='2023-01-01', periods=5, freq='D')
    }
    df = pd.DataFrame(data)
    
    # Initialize environment with debug mode
    env = TradingEnv(df=df, debug=True, stoploss=True)
    
    # Take a few steps to generate logs
    obs, _ = env.reset()
    print("Environment reset complete")
    
    # Test different actions
    actions = [
        (0, 0.02),  # HOLD with stoploss (should be ignored)
        (1, 0.05),  # BUY with 5% stoploss
        (0, 0.02),  # HOLD with position open
        (2, 0.0),   # SELL (stoploss ignored)
    ]
    
    for action in actions:
        obs, reward, done, _, info = env.step(action)
        print(f"Action: {action}, Reward: {reward:.4f}, Done: {done}")
    
    print(f"\nTest complete. Check {os.path.abspath(log_file)} for detailed logs.")
    
    # Verify log file was created and has content
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            lines = f.readlines()
            print(f"\nLog file contains {len(lines)} lines.")
            print("First few lines of log:")
            for line in lines[:5]:
                print(f"  {line.strip()}")
    else:
        print("ERROR: Log file was not created!")

if __name__ == "__main__":
    test_logging()
