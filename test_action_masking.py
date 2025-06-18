import gymnasium as gym
from trading_env import TradingEnv
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to import common_imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common_imports import load_data

def load_mgol_data(max_steps=1000):
    """Load and prepare MGOL data for the trading environment."""
    # Load the data
    df, _ = load_data(max_steps=max_steps)
    
    # Reset index to get datetime as a column
    df = df.reset_index()
    
    # Rename Date to datetime for compatibility with TradingEnv
    if 'Date' in df.columns and 'datetime' not in df.columns:
        df = df.rename(columns={'Date': 'datetime'})
    
    print(f"Loaded {len(df)} rows of MGOL data")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"First few rows:\n{df[['datetime', 'open', 'high', 'low', 'close', 'volume']].head()}")
    
    return df

# Load MGOL dataset
print("Loading MGOL dataset...")
df = load_mgol_data(max_steps=1000)  # Limit to 1000 rows for testing

# Create a test environment with test data
env = TradingEnv(
    df=df,
    debug=True,
    max_steps=1000,
    stoploss=True,
    stoploss_min=0.01,
    stoploss_max=0.1
)

# Reset the environment
obs, info = env.reset()

# Helper function to get valid actions from observation
def get_valid_actions(obs):
    action_mask = obs['action_mask']
    valid_actions = []
    for i, is_valid in enumerate(action_mask):
        if is_valid:
            valid_actions.append(["HOLD", "BUY", "SELL"][i])
    return valid_actions

# Helper function to get position status
def get_position_status(env):
    return {
        'position_open': env.position_open,
        'shares': env.shares,
        'balance': env.balance,
        'net_worth': env.net_worth
    }

# Test 1: Initial state - should only allow HOLD or BUY
print("\n=== Test 1: Initial state ===")
valid_actions = get_valid_actions(obs)
position = get_position_status(env)
print(f"Valid actions: {valid_actions}")
print(f"Position status: {position}")

# Take a BUY action with stoploss
print("\nTaking BUY action with stoploss...")
buy_action = (1, 0.02)  # BUY with 2% stoploss
obs, reward, done, truncated, info = env.step(buy_action)
position = get_position_status(env)
print(f"Position status: {position}")
print(f"Valid actions: {get_valid_actions(obs)}")

# Test 2: After BUY - should only allow HOLD or SELL
print("\n=== Test 2: After BUY ===")
position = get_position_status(env)
print(f"Position status: {position}")
print(f"Valid actions: {get_valid_actions(obs)}")

# Try to take an invalid action (BUY when position is open)
print("\nTesting invalid BUY action...")
prev_obs = obs
invalid_buy = (1, 0.03)  # Invalid BUY with 3% stoploss
obs, reward, done, truncated, info = env.step(invalid_buy)
position = get_position_status(env)
if np.array_equal(prev_obs['obs'], obs['obs']):
    print("State didn't change - action was likely ignored")
print(f"Position status: {position}")
print(f"Valid actions after invalid BUY: {get_valid_actions(obs)}")

# Take a SELL action
print("\nTaking SELL action...")
sell_action = (2, 0.0)  # SELL (stoploss doesn't matter here)
obs, reward, done, truncated, info = env.step(sell_action)
position = get_position_status(env)
print(f"Position status after SELL: {position}")
print(f"Valid actions after SELL: {get_valid_actions(obs)}")

# Test 3: After SELL - should reset to initial state
print("\n=== Test 3: After SELL ===")
position = get_position_status(env)
print(f"Position status: {position}")
print(f"Valid actions: {get_valid_actions(obs)}")
