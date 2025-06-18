"""
Test script for vectorized environments with TradingEnv.
Run with: python test_vec_env.py
"""
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from trading_env import TradingEnv
import pandas as pd

def create_test_data():
    """Create a small test DataFrame for the trading environment."""
    data = {
        'datetime': pd.date_range(start='2025-06-13 09:30', periods=100, freq='1min').strftime('%m/%d/%y %H:%M'),
        'open': [100 + i for i in range(100)],
        'high': [101 + i for i in range(100)],
        'low': [99 + i for i in range(100)],
        'close': [100.5 + i for i in range(100)],
        'volume': [1000] * 100
    }
    return pd.DataFrame(data)

def make_env(rank=0, stoploss=True):
    """Create an environment function for vectorized environments."""
    def _init():
        df = create_test_data()
        env = TradingEnv(df, debug=True, test_mode=True, stoploss=stoploss)
        return env
    return _init

def test_dummy_vec_env():
    """Test the environment with DummyVecEnv."""
    print("\n=== Testing DummyVecEnv ===")
    try:
        # Create a vectorized environment with 2 environments
        env = DummyVecEnv([make_env(i) for i in range(2)])
        print("DummyVecEnv created successfully")
        
        # Test reset
        obs = env.reset()
        print(f"Reset successful. Observation shape: {obs.shape}")
        
        # Test step with actions for both environments
        actions = [np.array([0, 0.05]), np.array([1, 0.1])]  # Two actions for two environments
        obs, rewards, dones, infos = env.step(actions)
        print(f"Step successful. Rewards: {rewards}, Dones: {dones}")
        
        # Test another step
        obs, rewards, dones, infos = env.step([np.array([2, 0.0]), np.array([0, 0.0])])
        print(f"Second step successful. Rewards: {rewards}, Dones: {dones}")
        
        env.close()
        print("DummyVecEnv test completed successfully")
        return True
    except Exception as e:
        print(f"Error with DummyVecEnv: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_subproc_vec_env():
    """Test the environment with SubprocVecEnv."""
    print("\n=== Testing SubprocVecEnv ===")
    try:
        # Create a vectorized environment with 2 environments
        env = SubprocVecEnv([make_env(i) for i in range(2)])
        print("SubprocVecEnv created successfully")
        
        # Test reset
        obs = env.reset()
        print(f"Reset successful. Observation shape: {obs.shape}")
        
        # Test step with actions for both environments
        actions = [np.array([0, 0.05]), np.array([1, 0.1])]
        obs, rewards, dones, infos = env.step(actions)
        print(f"Step successful. Rewards: {rewards}, Dones: {dones}")
        
        # Test another step
        obs, rewards, dones, infos = env.step([np.array([2, 0.0]), np.array([0, 0.0])])
        print(f"Second step successful. Rewards: {rewards}, Dones: {dones}")
        
        env.close()
        print("SubprocVecEnv test completed successfully")
        return True
    except Exception as e:
        print(f"Error with SubprocVecEnv: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=== Starting Vectorized Environment Tests ===")
    
    # Run DummyVecEnv test
    dummy_success = test_dummy_vec_env()
    
    # Only run SubprocVecEnv test if DummyVecEnv was successful
    subproc_success = False
    if dummy_success:
        subproc_success = test_subproc_vec_env()
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"DummyVecEnv: {'PASSED' if dummy_success else 'FAILED'}")
    print(f"SubprocVecEnv: {'PASSED' if subproc_success else 'SKIPPED or FAILED'}")
    
    if dummy_success and subproc_success:
        print("\nAll vectorized environment tests passed!")
    else:
        print("\nSome tests failed. Check the output above for error messages.")

if __name__ == "__main__":
    main()
