import sys
import os
import numpy as np
import pandas as pd
from trading_env_enhanced import TradingEnv
from common_imports import load_data

def test_observation_normalization():
    print("=== Testing Observation Normalization ===")
    
    # Load sample data
    df = load_and_preprocess_data('MGOL.csv')
    
    # Create environment with debug mode
    env = TradingEnv(df=df, debug=True, stoploss=True)
    
    # Reset environment and get initial observation
    obs, _ = env.reset()
    
    print("\nInitial observation:")
    print(f"Shape: {obs['obs'].shape}")
    print(f"Min: {obs['obs'].min():.4f}, Max: {obs['obs'].max():.4f}, Mean: {obs['obs'].mean():.4f}")
    print("\nObservation breakdown:")
    print(f"Price features (should be ~N(0,1)): {obs['obs'][:4]}")
    print(f"Volume (log-normalized): {obs['obs'][4]:.4f}")
    print(f"Position flag (should be -1 or 1): {obs['obs'][5]:.4f}")
    print(f"Balance (scaled): {obs['obs'][6]:.4f}")
    print(f"Position size (scaled): {obs['obs'][7]:.4f}")
    print(f"Progress (scaled to [-1,1]): {obs['obs'][8]:.4f}")
    print(f"PnL (tanh scaled): {obs['obs'][9]:.4f}")
    
    # Take a few steps and check observations
    print("\nTaking 5 steps and checking observations...")
    for i in range(5):
        action = (1, [0.05])  # BUY with 5% stoploss
        obs, reward, done, _, _ = env.step(action)
        print(f"\nStep {i+1}:")
        print(f"  Reward: {reward:.6f}")
        print(f"  Obs range: [{obs['obs'].min():.4f}, {obs['obs'].max():.4f}]")
        
        # Verify observations are within expected ranges
        assert -5.0 <= obs['obs'].min() <= obs['obs'].max() <= 5.0, "Observations should be in [-5, 5] range"
        
    print("\nObservation normalization tests passed!")

def test_reward_scaling():
    print("\n=== Testing Reward Scaling ===")
    
    # Load sample data
    df = load_and_preprocess_data('MGOL.csv')
    
    # Create environment with debug mode
    env = TradingEnv(df=df, debug=True, stoploss=True)
    
    # Take actions and collect rewards
    env.reset()
    rewards = []
    
    # Test BUY, HOLD, SELL cycle
    actions = [
        (1, [0.05]),  # BUY with 5% stoploss
        (0, [0.0]),    # HOLD
        (2, [0.0])     # SELL
    ]
    
    print("Testing reward scaling with BUY-HOLD-SELL cycle...")
    for i, action in enumerate(actions):
        _, reward, done, _, _ = env.step(action)
        rewards.append(reward)
        print(f"Step {i+1} (Action: {action}): Reward = {reward:.6f}")
    
    # Test stoploss trigger
    print("\nTesting stoploss trigger...")
    env.reset()
    env.step((1, [0.05]))  # BUY
    
    # Force stoploss by setting price below stoploss
    env.current_step = 1
    current_price = env.df.iloc[env.current_step]['close']
    env.df.at[env.current_step, 'close'] = current_price * 0.9  # 10% drop
    
    _, reward, _, _, _ = env.step((0, [0.0]))  # HOLD (should trigger stoploss)
    print(f"Stoploss reward: {reward:.6f}")
    
    # Verify rewards are in reasonable range
    all_rewards = rewards + [reward]
    print(f"\nAll rewards: {[f'{r:.6f}' for r in all_rewards]}")
    print(f"Min reward: {min(all_rewards):.6f}")
    print(f"Max reward: {max(all_rewards):.6f}")
    
    assert all(-10.0 <= r <= 10.0 for r in all_rewards), "Rewards should be in [-10, 10] range"
    print("\nReward scaling tests passed!")

if __name__ == "__main__":
    print("=== Starting Normalization Tests ===\n")
    test_observation_normalization()
    test_reward_scaling()
    print("\n=== All normalization tests completed successfully! ===")
