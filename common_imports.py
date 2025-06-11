import os
import psutil
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# Gymnasium imports
import gymnasium as gym
from gymnasium import Env, Wrapper
from gymnasium.spaces import Discrete, Box

# Stable-Baselines3 imports
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

# SB3 Contrib imports
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

# Environment imports
import importlib
import trading_env
importlib.reload(trading_env)
from trading_env import TradingEnv
from stable_baselines3.common.env_checker import check_env

# Callback imports
import debug_eval_callback
importlib.reload(debug_eval_callback)
from debug_eval_callback import DebugEvalCallback

import learning_progress_callback
importlib.reload(learning_progress_callback)
from learning_progress_callback import LearningProgressCallback

def load_data(max_steps=None):
    """
    Load and preprocess the MGOL data.
    
    Args:
        max_steps (int, optional): Maximum number of rows to return. If None, returns all data.
        
    Returns:
        tuple: (df, df_original) where df is the processed DataFrame and df_original is a copy
               of the original data before any modifications.
    """
    # Load Data
    df = pd.read_csv("MGOL.csv")
    
    # If max_steps is provided, limit the data
    if max_steps is not None and max_steps > 0:
        df = df.head(max_steps)
    
    # Parse datetime with explicit year setting
    # First split datetime into date and time parts
    df['date'] = df['datetime'].str.split(' ').str[0]
    df['time'] = df['datetime'].str.split(' ').str[1]
    
    # Convert date to datetime with explicit year setting
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%y')
    # Set the year to 2025 explicitly
    df['date'] = df['date'].apply(lambda x: x.replace(year=2025))
    
    # Combine date and time back together
    df['datetime'] = df['date'].astype(str) + ' ' + df['time']
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Create a copy of the original data before modifying the index
    df_original = df.copy()
    
    # Drop temporary date and time columns
    df = df.drop(['date', 'time'], axis=1)
    
    # Keep all columns except symbol and frame
    df = df.drop(columns=['symbol', 'frame'])
    
    # Make sure to keep the datetime column for TradingEnv
    if 'datetime' not in df.columns:
        df.reset_index(inplace=True)
        df.rename(columns={'Date': 'datetime'}, inplace=True)
        df.set_index('datetime', inplace=True)
    
    # Set index to datetime and add 3 hours offset
    df.set_index('datetime', inplace=True)
    df.index = df.index + pd.Timedelta(hours=3)
    df.index.name = 'Date'
    
    # Reorder columns to match mplfinance expectations while keeping all other columns
    df = df[['open', 'high', 'low', 'close', 'volume']].copy()
    
    return df, df_original

# Load the full dataset by default
df, df_original = load_data()