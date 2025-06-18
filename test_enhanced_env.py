import numpy as np
import pandas as pd
from trading_env_enhanced import TradingEnv

def test_enhanced_environment():
    # Create a simple test dataframe
    data = {
        'datetime': pd.date_range(start='2023-01-01', periods=100, freq='1min'),
        'open': np.linspace(100, 200, 100),
        'high': np.linspace(101, 201, 100),
        'low': np.linspace(99, 199, 100),
        'close': np.linspace(100, 200, 100),
        'volume': np.ones(100) * 1000
    }
    df = pd.DataFrame(data)
    
    # Initialize environment with test data
    env = TradingEnv(df=df, debug=True, stoploss=True)
    
    # Test initial state
    print("\n=== Testing Initial State ===")
    obs, _ = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Initial valid actions: {env.get_valid_actions()}")
    
    # Test with sufficient balance
    env.balance = 200000  # Enough to buy at least 1000 shares
    print("\n=== Testing with Sufficient Balance ===")
    print(f"Balance: {env.balance}")
    print(f"Current price: {df.iloc[0]['close']}")
    print(f"Valid actions: {env.get_valid_actions()}")
    
    # Test with insufficient balance
    env.balance = 50000  # Not enough to buy 1000 shares
    print("\n=== Testing with Insufficient Balance ===")
    print(f"Balance: {env.balance}")
    print(f"Current price: {df.iloc[0]['close']}")
    print(f"Valid actions: {env.get_valid_actions()}")
    
    # Test after buying
    env.balance = 200000
    action = (1, 0.02)  # BUY with 2% stoploss
    obs, reward, done, _, _ = env.step(action)
    print("\n=== After BUY Action ===")
    print(f"Position open: {env.position_open}")
    print(f"Shares: {env.shares}")
    print(f"Valid actions: {env.get_valid_actions()}")
    
    # Test after selling
    action = (2, 0)  # SELL
    obs, reward, done, _, _ = env.step(action)
    print("\n=== After SELL Action ===")
    print(f"Position open: {env.position_open}")
    print(f"Shares: {env.shares}")
    print(f"Valid actions: {env.get_valid_actions()}")

if __name__ == "__main__":
    test_enhanced_environment()
