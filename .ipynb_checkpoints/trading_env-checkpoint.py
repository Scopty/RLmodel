# TradingEnv

import gymnasium as gym
from gymnasium.spaces import Discrete, Box

import numpy as np
import random
import jax.numpy as jnp

debug = False

class TradingEnv(gym.Env):
    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)
        return [seed]
        
    def __init__(self, df):
        super(TradingEnv, self).__init__()
        self.df = df
        self.current_step = 0
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.shares = 0
        self.buy_price = 0
        self.total_reward = 0
        self.position_open = False
        self.round_trip_trades = 0

        # Action Space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = Discrete(3)

        # Observation space must use NumPy
        self.obs_shape = len(self.df.columns) + 7
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float32)

    def get_action_mask(self) -> jnp.ndarray:
        #"""Returns a JAX binary mask indicating valid actions."""
        action_mask = jnp.zeros(self.action_space.n, dtype=jnp.float32)
        action_mask = action_mask.at[0].set(1)  # Hold is always valid
        if not self.position_open:
            action_mask = action_mask.at[1].set(1)  # Buy only if no position open
        if self.position_open:
            action_mask = action_mask.at[2].set(1)  # Sell only if position open
        return action_mask

    def get_obs(self) -> jnp.ndarray:
        """ Generates the current observation in JAX."""
        if self.current_step >= len(self.df):
            return jnp.zeros(self.obs_shape, dtype=jnp.float32)  # or some terminal obs
    
        close_price = self.df.iloc[self.current_step]["close"]
        obs = jnp.concatenate([
            #jnp.array(self.df.iloc[self.current_step][["open", "high", "low", "close", "volume"]].values, dtype=jnp.float32),  # (5,)
            jnp.array(self.df.iloc[self.current_step][["close"]].values, dtype=jnp.float32),  # (5,)
            jnp.array([self.shares, self.balance, self.net_worth, self.current_step], dtype=jnp.float32)  # (4,)
        ])
    
        # Get action mask and combine
        action_mask = self.get_action_mask()  # Ensure get_action_mask() is correctly implemented
        obs = jnp.concatenate([obs, jnp.array(action_mask, dtype=jnp.float32)])
    
        return obs


    def reset(self, seed=None, options=None):
        #"""Resets the environment and returns an initial NumPy observation."""
        self.current_step = 0
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.shares = 0
        self.position_open = False
        self.round_trip_trades = 0

        obs = self.get_obs()  # JAX array
        obs = np.array(obs, dtype=np.float32)  # Convert to NumPy for Gymnasium compatibility

        if debug:
            print(f"Environment Reset: Initial Observation (shape {obs.shape}): {obs}")
        return obs, {}

    def step(self, action):
        #"""Take a step in the environment and return a NumPy-compatible output."""
    
        # Debugging: Check action format
        if debug: print(f"Action received: {action}, shape: {np.shape(action)}")
    
        if isinstance(action, (np.ndarray, jnp.ndarray)):
            if action.shape == (3,):  # If it's a vector of logits
                action = np.argmax(action)  # Choose the action with highest probability
            else:
                action = int(action.item())  # Convert to scalar


    
        action = np.clip(action, 0, 2)  # Ensure action is within [0, 1, 2]
    
        if debug:
            print(f"Action taken: {action}")
    
        close_price = self.df.iloc[self.current_step]["close"]
        reward = 0
        done = False
    
        # Action Handling
        if action == 1 and not self.position_open:
            self.shares = 1000
            self.balance -= close_price * self.shares
            self.position_open = True
            self.buy_price = close_price
            if debug:
                print("Bought shares")
    
        elif action == 2 and self.position_open:
            self.balance += close_price * self.shares
            reward = (close_price - self.buy_price) * self.shares
            self.shares = 0
            self.buy_price = 0
            self.position_open = False
            self.round_trip_trades += 1
            if debug:
                print(f"Sold shares, Reward: {reward}")
    
        self.net_worth = self.balance + (self.shares * close_price)
        done = self.round_trip_trades >= 10 or self.net_worth <= 0 or self.current_step >= len(self.df) - 1
    
        if debug:
            print(f"Net Worth: {self.net_worth}, Total Reward: {self.total_reward}")
    
        self.current_step += 1
    
        obs = self.get_obs()  # JAX array
        obs = np.array(obs, dtype=np.float32)  # Convert to NumPy
    
        info = {
            "valid_actions": [0, 1, 2],
            "action_mask": np.array(self.get_action_mask(), dtype=np.float32)  # Convert JAX to NumPy
        }
    
        # ✅ Ensure action shape is (1,)
        action = np.array(action, dtype=np.int32).reshape(-1, 1)  # Fix shape
    
        return obs, float(reward), bool(done), False, info

