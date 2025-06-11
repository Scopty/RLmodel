import pandas as pd
import numpy as np
from trading_env import TradingEnv
from common_imports import df  # Import the processed DataFrame

# Calculate market duration in minutes (4 AM to 8 PM = 16 hours)
market_minutes = 16 * 60
print(f"Market duration: {market_minutes} minutes")

# Initialize environment
env = TradingEnv(df, debug=True)

# Test time features and rewards for specific times
first_time = True
for i in range(len(df)):  # Step through entire dataset
    # Get current time information
    current_row = df.iloc[i]
    timestamp = current_row['datetime']
    hour = timestamp.hour
    minute = timestamp.minute
    
    # Print the first time and 7:59 PM
    if first_time or (hour == 19 and minute == 59):
        first_time = False
        print(f"\nTime: {timestamp.strftime('%H:%M') if minute != 59 else '19:59'}")
        print(f"Hour: {hour}, Minute: {minute}")
        
        # Reset environment to current step
        env.current_step = i
        obs, info = env.reset()
        normalized_time = obs[-4]  # normalized_time is the fourth to last feature
        is_pre_market = obs[-3]  # is_pre_market is the third to last feature
        is_after_hours = obs[-2]  # is_after_hours is the second to last feature
        time_until_close = obs[-1]  # time_until_close is the last feature
        
        print(f"Normalized time since open: {normalized_time:.4f}")
        print(f"Is pre-market: {is_pre_market}")
        print(f"Is after-hours: {is_after_hours}")
        print(f"Normalized time until close: {time_until_close:.4f}")
        
        # Test rewards for different actions
        print("\nTesting rewards for different actions:")
        print(f"Current Step: {i}")
        
        # Test buy action
        print(" Action 1")
        action = 1
        shares = 1000  # Fixed number of shares
        env.shares = shares
        env.balance = env.balance - (shares * current_row['close'])
        env.net_worth = env.balance + (shares * current_row['close'])
        print(f"Buy {shares} shares at price: {current_row['close']}")
        print(f" Shares: {shares}")
        print(f" Balance: {env.balance}")
        print(f" Market position: {shares * current_row['close']}")
        print(f" Net Worth: {env.net_worth}")
        print(f" Max Steps: {env.max_steps}")
        
        # Step environment
        obs, reward, done, _, info = env.step(action)
        print(f"Buy reward: {reward}")
        
        # Test hold action
        print(" Action 0")
        action = 0
        # Calculate holding profit
        holding_profit = (current_row['close'] - env.buy_price) * env.shares
        duration_bonus = 0.005 if env.current_step > 0 else 0
        print(f"Holding profit: {holding_profit}, Duration bonus: {duration_bonus}")
        print(f" Shares: {env.shares}")
        print(f" Balance: {env.balance}")
        print(f" Market position: {env.shares * current_row['close']}")
        print(f" Net Worth: {env.balance + (env.shares * current_row['close'])}")
        print(f" Max Steps: {env.max_steps}")
        
        # Step environment
        obs, reward, done, _, info = env.step(action)
        print(f"Hold reward: {reward}")
        
        # Test sell action
        print(" Action 2")
        action = 2
        sell_profit = (current_row['close'] - env.buy_price) * env.shares
        print(f"Sell profit: {sell_profit}")
        print(f" Shares: {env.shares}")
        print(f" Balance: {env.balance + sell_profit}")
        print(f" Market position: 0.0")
        print(f" Net Worth: {env.balance + sell_profit}")
        print(f" Max Steps: {env.max_steps}")
        
        # Step environment
        obs, reward, done, _, info = env.step(action)
        print(f"Sell reward: {reward}")
        
        print("\n")
    
    # Reset environment for next test
    env.reset()
