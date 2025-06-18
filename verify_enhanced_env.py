import numpy as np
import pandas as pd
from trading_env_enhanced import TradingEnv

def test_enhanced_environment():
    # Create a simple test dataframe
    np.random.seed(42)
    n_steps = 100
    
    # Initialize lists for each column
    datetime = pd.date_range(start='2023-01-01', periods=n_steps, freq='1min')
    open_prices = 100 + np.cumsum(np.random.randn(n_steps) * 0.5)
    high = []
    low = []
    close = []
    volume = np.ones(n_steps) * 1000
    
    # Calculate high, low, close with random variations
    for i in range(n_steps):
        current_high = open_prices[i] + abs(np.random.randn() * 0.3)
        current_low = open_prices[i] - abs(np.random.randn() * 0.3)
        current_close = np.random.uniform(current_low, current_high)
        
        high.append(current_high)
        low.append(current_low)
        close.append(current_close)
    
    # Create the DataFrame
    data = {
        'datetime': datetime,
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }
    
    df = pd.DataFrame(data)
    
    # Initialize environment with test data
    print("Initializing environment...")
    env = TradingEnv(df=df, debug=True, stoploss=True)
    
    # Test initial state
    print("\n=== Testing Initial State ===")
    obs, _ = env.reset()
    print(f"Initial observation shape: {obs['obs'].shape}")
    print(f"Initial action mask: {obs['action_mask']}")
    print(f"Initial valid actions: {[i for i, valid in enumerate(obs['action_mask']) if valid]}")
    
    # Test with sufficient balance
    print("\n=== Testing with Sufficient Balance ===")
    print(f"Balance: ${env.balance:,.2f}")
    print(f"Current price: ${df.iloc[0]['close']:.2f}")
    print(f"Valid actions: {[i for i, valid in enumerate(env.get_valid_actions()) if valid]}")
    
    # Test buy action with stoploss
    print("\n=== Testing BUY Action with Stoploss ===")
    action = (1, 0.05)  # BUY with 5% stoploss
    obs, reward, done, _, info = env.step(action)
    print(f"Action: BUY with 5% stoploss")
    print(f"New balance: ${env.balance:,.2f}")
    print(f"Shares: {env.shares}")
    print(f"Stoploss price: ${env.stoploss_price:.2f}" if env.stoploss_price else "No stoploss set")
    print(f"Valid actions: {[i for i, valid in enumerate(obs['action_mask']) if valid]}")
    
    # Test invalid action (trying to buy again while holding)
    print("\n=== Testing Invalid BUY Action ===")
    invalid_action = (1, 0.1)  # Try to BUY again (invalid)
    obs, reward, done, _, info = env.step(invalid_action)
    print(f"Action: BUY (invalid - already holding)")
    print(f"Reward: {reward:.4f} (should be negative for invalid action)")
    
    # Test SELL action
    print("\n=== Testing SELL Action ===")
    sell_action = (2, 0)  # SELL
    obs, reward, done, _, info = env.step(sell_action)
    print(f"Action: SELL")
    print(f"Reward: {reward:.4f}")
    print(f"New balance: ${env.balance:,.2f}")
    print(f"Shares: {env.shares}")
    print(f"Valid actions: {[i for i, valid in enumerate(obs['action_mask']) if valid]}")
    
    # Test stoploss trigger
    print("\n=== Testing Stoploss Trigger ===")
    # Buy again
    env.step((1, 0.05))  # BUY with 5% stoploss
    buy_price = df.iloc[env.current_step-1]['close']
    print(f"Bought at: ${buy_price:.2f}")
    print(f"Stoploss set at: ${env.stoploss_price:.2f}")
    
    # Simulate price drop below stoploss
    print("Simulating price drop below stoploss...")
    df.iloc[env.current_step]['close'] = env.stoploss_price * 0.99  # Just below stoploss
    obs, reward, done, _, info = env.step((0, 0))  # HOLD
    print(f"Stoploss triggered at: ${df.iloc[env.current_step-1]['close']:.2f}")
    print(f"Reward: {reward:.4f} (should be negative)")
    print(f"Position open: {env.position_open} (should be False)")
    
    print("\n=== Test Complete ===")
    env.close()

if __name__ == "__main__":
    test_enhanced_environment()
