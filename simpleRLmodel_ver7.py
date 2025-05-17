import os, psutil

# Determine number of available (idle) cores
IDLE_THRESHOLD = 20.0  # percent
cpu_usages = psutil.cpu_percent(percpu=True, interval=1)
available_cores = sum(usage < IDLE_THRESHOLD for usage in cpu_usages)
available_cores = max(1, available_cores) - 4 # At least 1

available_cores = os.cpu_count()

import mplfinance
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# Stable-Baselines3 imports
from stable_baselines3 import PPO,A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv  # Use this instead of DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
#from stable_baselines3.common.policies import ActorCriticPolicy
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

# Gymnasium (updated from Gym)
import gymnasium as gym
from gymnasium import Env,Wrapper
from gymnasium.spaces import Discrete, Box

import importlib
import trading_env_sb3_ver2c
importlib.reload(trading_env_sb3_ver2c)
from trading_env_sb3_ver2c import TradingEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.ppo_mask import MaskablePPO

# Load Data
df = pd.read_csv("MGOL.csv")  # Replace with actual file
df['datetime'] = pd.to_datetime(df['datetime'], format='%m/%d/%y %H:%M')
df.set_index('datetime', inplace=True)

df.index = df.index + pd.Timedelta(hours=3)
df.index.name = 'Date'
df = df.drop(columns=['symbol', 'frame'])
df = df.iloc[:10]  # Select the first 30 rows
df_original = df
df = df[["close"]]

# Create directories for saving
log_dir = "./tensorboard_logs/"
os.makedirs(log_dir, exist_ok=True)

num_cpu = 1

def make_env():
    env = TradingEnv(df)  # Initialize env
    check_env(env, warn=True)
    return env

if __name__ == "__main__":
    # Create parallel environments using SubprocVecEnv
    env = SubprocVecEnv([make_env for _ in range(num_cpu)])

    # Wrap with VecMonitor first
    env = VecMonitor(env)

    # Wrap with VecNormalize
    env = VecNormalize(env)

    # Create and train the model with custom PPO settings
    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        verbose=0,
        tensorboard_log="./tensorboard_logs/",
        learning_rate=5e-4,
        ent_coef=0.001,
        gamma=0.99,
        gae_lambda=0.95,
        n_steps=64,
        clip_range=0.1,
        clip_range_vf=0.1,
        n_epochs=20,
        batch_size=32,
        max_grad_norm=0.5,
        vf_coef=0.5,
        normalize_advantage=False,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        )
    )
    model.learn(total_timesteps=10000, tb_log_name="ppo_custom")
    model.save("ppo_custom")
