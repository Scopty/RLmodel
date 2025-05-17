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
import trading_env_sb3_ver2c
importlib.reload(trading_env_sb3_ver2c)
from trading_env_sb3_ver2c import TradingEnv
from stable_baselines3.common.env_checker import check_env

# Callback imports
import debug_eval_callback
importlib.reload(debug_eval_callback)
from debug_eval_callback import DebugEvalCallback

import learning_progress_callback
importlib.reload(learning_progress_callback)
from learning_progress_callback import LearningProgressCallback

# Load Data
df = pd.read_csv("MGOL.csv")  # Replace with actual file
df['datetime'] = pd.to_datetime(df['datetime'], format='%m/%d/%y %H:%M')
df.set_index('datetime', inplace=True)

df.index = df.index + pd.Timedelta(hours=3)
df.index.name = 'Date'

# Keep all columns except symbol and frame
df = df.drop(columns=['symbol', 'frame'])

# Reorder columns to match mplfinance expectations while keeping all other columns
df = df[['open', 'high', 'low', 'close', 'volume']].copy()

df = df.iloc[:30]  # Subset the rows to maintain 30-bar window
df_original = df.copy()
# Keep all OHLC data for plotting, but only use close for training