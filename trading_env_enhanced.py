import numpy as np
import pandas as pd
from gymnasium import Env
from gymnasium.spaces import Discrete, Box, Dict, Tuple
from collections import OrderedDict
import gymnasium
import os
import datetime
from pathlib import Path
import sys

class TradingEnv(Env):
    def __init__(self, df, debug=False, max_steps=None, model_name=None, input_dir_name=None, test_mode=False, stoploss=False, stoploss_min=0.01, stoploss_max=1.0, log_raw_obs=True, log_norm_obs=True, log_rewards=True):
        super().__init__()
        
        self.total_reward = 0
        self.df = df
        self.debug = debug
        self.max_steps = max_steps if max_steps is not None else len(df) - 1 if df is not None else 100
        self.model_name = model_name
        self.test_mode = test_mode
        
        # Debug logging controls
        self.log_raw_obs = log_raw_obs  # Log raw observation values
        self.log_norm_obs = log_norm_obs  # Log normalized observation values
        self.log_rewards = log_rewards  # Log reward calculations
        
        # Initialize state variables
        self.current_step = 0
        self.initial_balance = 1000000  # $1,000,000 initial balance
        self.balance = self.initial_balance
        self.shares = 0
        self.buy_price = 0
        self.position_open = False
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        
        self.stoploss = stoploss
        self.stoploss_min = stoploss_min
        self.stoploss_max = stoploss_max
        self.stoploss_price = None
        self.stoploss_distance = None
        
        # Define action and observation space
        if self.stoploss:
            self.action_space = gymnasium.spaces.Tuple([
                Discrete(3),  # Action type (0=HOLD, 1=BUY, 2=SELL)
                Box(low=np.array([self.stoploss_min], dtype=np.float32), 
                    high=np.array([self.stoploss_max], dtype=np.float32), 
                    shape=(1,), 
                    dtype=np.float32)  # Stoploss percentage
            ])
        else:
            self.action_space = Discrete(3)  # 0=HOLD, 1=BUY, 2=SELL
        
        # Observation space
        self.observation_space = Dict({
            'obs': Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(10,),  # Adjust based on your features
                dtype=np.float32
            ),
            'action_mask': Box(0, 1, shape=(3,), dtype=np.int8)  # Mask for 3 actions
        })
        
        # Initialize debug logging first
        # Use the same log file as the main environment
        self.debug_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_debug.log')
        if self.debug:
            with open(self.debug_file, 'a') as f:
                f.write("\n=== New TradingEnv Instance ===\n")
        
        # Initialize normalization statistics (after debug logging is set up)
        self._init_normalization()
    
    def _init_normalization(self):
        """Initialize normalization statistics based on the dataset."""
        if self.df is None:
            # Default values if no data is provided
            self.price_mean = 100.0
            self.price_std = 10.0
            self.volume_mean = 1000.0
            self.volume_std = 100.0
            return
            
        # Calculate price statistics (using close price as reference)
        self.price_mean = float(self.df[['open', 'high', 'low', 'close']].mean().mean())
        self.price_std = float(self.df[['open', 'high', 'low', 'close']].std().mean())
        
        # Calculate volume statistics (log scale for better normalization)
        volume = np.log1p(self.df['volume'])
        self.volume_mean = float(volume.mean())
        self.volume_std = float(volume.std())
        
        # Avoid division by zero
        self.price_std = max(self.price_std, 1e-8)
        self.volume_std = max(self.volume_std, 1e-8)
        
        if self.debug:
            self._debug_print(f"Normalization stats - Price: mean={self.price_mean:.2f}, std={self.price_std:.2f}")
            self._debug_print(f"Normalization stats - Volume: mean={self.volume_mean:.2f}, std={self.volume_std:.2f}")
    
    def _debug_print(self, message):
        """Helper method for debug logging"""
        if self.debug:
            with open(self.debug_file, 'a') as f:
                f.write(f"{message}\n")
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares = 0
        self.buy_price = 0
        self.position_open = False
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.stoploss_price = None
        self.stoploss_distance = None
        self.total_reward = 0
        
        # Get initial observation
        obs = self._get_obs()
        
        if self.debug:
            self._debug_print("\n=== Environment Reset ===")
            self._debug_print(f"Initial balance: ${self.balance:,.2f}")
            self._debug_print(f"Initial valid actions: {self.get_valid_actions()}")
        
        return obs, {}
    
    def _get_obs(self):
        """Get current observation with proper normalization.
        
        Returns:
            dict: Dictionary containing:
                - 'obs': Normalized observation vector
                - 'action_mask': Binary mask of valid actions
        """
        if self.df is None or self.current_step >= len(self.df):
            # Return zero observation if no data
            return {
                'obs': np.zeros(10, dtype=np.float32),
                'action_mask': np.array([1, 0, 0], dtype=np.int8)  # Only HOLD is valid
            }
            
        current_row = self.df.iloc[self.current_step]
        current_price = current_row['close']
        
        # Log raw values before normalization
        if self.debug and hasattr(self, 'log_raw_obs') and self.log_raw_obs:
            self._debug_print("\n[DEBUG_OBS] Raw values:")
            self._debug_print(f"  Price - O: {current_row['open']:.4f}, H: {current_row['high']:.4f}, L: {current_row['low']:.4f}, C: {current_row['close']:.4f}")
            self._debug_print(f"  Volume: {current_row['volume']:.2f}")
            self._debug_print(f"  Position: {'OPEN' if self.position_open else 'CLOSED'}, Shares: {self.shares}, Buy Price: {self.buy_price:.4f}")
            self._debug_print(f"  Balance: ${self.balance:,.2f}, Net Worth: ${self.net_worth:,.2f}")
            self._debug_print(f"  Step: {self.current_step}/{len(self.df)-1} ({(self.current_step/len(self.df)*100):.1f}%)")
            
            # Log normalization statistics
            self._debug_print("\n[DEBUG_OBS] Normalization stats:")
            self._debug_print(f"  Price - Mean: {self.price_mean:.4f}, Std: {self.price_std:.4f}")
            self._debug_print(f"  Volume - Mean: {self.volume_mean:.4f}, Std: {self.volume_std:.4f}")
            
        # Calculate normalized values
        normalized_open = (current_row['open'] - self.price_mean) / self.price_std
        normalized_high = (current_row['high'] - self.price_mean) / self.price_std
        normalized_low = (current_row['low'] - self.price_mean) / self.price_std
        normalized_close = (current_row['close'] - self.price_mean) / self.price_std
        
        # Calculate log volume for normalization
        log_volume = np.log1p(current_row['volume'])
        normalized_volume = (log_volume - self.volume_mean) / self.volume_std
        
        # Portfolio metrics
        normalized_balance = (self.balance / self.initial_balance - 1.0) * 0.5  # Scale to [-0.5, 0.5]
        position_size = self.shares * current_price / self.initial_balance
        normalized_position = np.clip(position_size, 0, 2.0) - 1.0  # Scale to [-1, 1]
        
        # Episode progress
        episode_progress = self.current_step / max(len(self.df) - 1, 1)
        
        # PnL calculation
        pnl = (self.net_worth - self.initial_balance) / self.initial_balance
        normalized_pnl = np.tanh(pnl)  # Scale to [-1, 1]
        
        # Log normalized values if debug is enabled
        if self.debug and hasattr(self, 'log_norm_obs') and self.log_norm_obs:
            self._debug_print("\n[DEBUG_OBS] Normalized values:")
            self._debug_print(f"  Price - O: {normalized_open:7.4f}, H: {normalized_high:7.4f}, L: {normalized_low:7.4f}, C: {normalized_close:7.4f}")
            self._debug_print(f"  Volume: {normalized_volume:7.4f} (log_vol: {log_volume:.2f})")
            self._debug_print(f"  Position: {normalized_position:7.4f} (size: {position_size:.4f})")
            self._debug_print(f"  Balance: {normalized_balance:7.4f}, PnL: {normalized_pnl:7.4f} (raw: {pnl*100:6.2f}%)")
            self._debug_print(f"  Progress: {episode_progress*100:5.1f}%")
            
            # Log normalization ranges
            self._debug_print("\n[DEBUG_OBS] Normalization ranges:")
            self._debug_print(f"  Price: ~N(0,1) - 99.7% in [{self.price_mean-3*self.price_std:.2f}, {self.price_mean+3*self.price_std:.2f}]")
            self._debug_print(f"  Volume: ~N(0,1) - 99.7% in [{np.expm1(self.volume_mean-3*self.volume_std):.0f}, {np.expm1(self.volume_mean+3*self.volume_std):.0f}]")
        
        # Create normalized observation vector
        obs = np.array([
            normalized_open,
            normalized_high,
            normalized_low,
            normalized_close,
            normalized_volume,
            float(self.position_open) * 2 - 1,  # Convert to [-1, 1] range
            normalized_balance,
            normalized_position,
            episode_progress * 2 - 1,  # Scale to [-1, 1]
            normalized_pnl
        ], dtype=np.float32)
        
        # Clip observations to reasonable range
        obs = np.clip(obs, -5.0, 5.0)
        
        # Get valid actions
        valid_actions = self.get_valid_actions()
        action_mask = np.zeros(3, dtype=np.int8)
        for a in valid_actions:
            action_mask[a] = 1
            
        return {
            'obs': obs,
            'action_mask': action_mask
        }
    
    def step(self, action):
        """Execute one step in the environment"""
        if self.df is None or self.current_step >= len(self.df) - 1:
            return self._get_obs(), 0, True, False, {}
            
        # Handle action (with stoploss if enabled)
        if self.stoploss and isinstance(action, (tuple, list)) and len(action) == 2:
            action_type = int(action[0])
            stoploss_distance = float(action[1][0]) if isinstance(action[1], (list, np.ndarray)) else float(action[1])
        else:
            action_type = int(action)
            stoploss_distance = None
        
        current_row = self.df.iloc[self.current_step]
        current_price = current_row['close']
        
        # Execute action
        reward = 0
        done = False
        info = {}
        
        # Action logic here (simplified)
        if action_type == 1 and not self.position_open:  # BUY
            self.position_open = True
            self.buy_price = current_price
            self.shares = self.balance // current_price
            self.balance -= self.shares * current_price
            
            if self.stoploss and stoploss_distance is not None:
                self.stoploss_distance = stoploss_distance
                self.stoploss_price = current_price * (1 - stoploss_distance)
                if self.debug:
                    self._debug_print(f"Set stoploss at {self.stoploss_price:.2f} ({stoploss_distance*100:.1f}% below buy price)")
            
            if self.debug:
                self._debug_print(f"BUY {self.shares} shares at {current_price:.2f}")
                
        elif action_type == 2 and self.position_open:  # SELL
            self.balance += self.shares * current_price
            profit = (current_price - self.buy_price) * self.shares
            raw_sell_reward = profit / self.initial_balance  # Normalized reward
            reward = raw_sell_reward
            
            if self.debug:
                self._debug_print(f"SELL {self.shares} shares at {current_price:.2f}")
                self._debug_print(f"Profit: ${profit:,.2f} ({profit/self.initial_balance*100:.2f}%)")
                
            if self.debug and hasattr(self, 'log_rewards') and self.log_rewards:
                self._debug_print("\n[DEBUG_REWARD] Sell Action:")
                self._debug_print(f"  Buy Price: ${self.buy_price:.4f}")
                self._debug_print(f"  Sell Price: ${current_price:.4f}")
                self._debug_print(f"  Shares: {self.shares}")
                self._debug_print(f"  Raw Profit: ${profit:,.2f}")
                self._debug_print(f"  Raw Sell Reward: {raw_sell_reward:.6f}")
                self._debug_print(f"  Final Sell Reward: {reward:.6f}")
            
            self.position_open = False
            self.shares = 0
            self.buy_price = 0
            self.stoploss_price = None
            self.stoploss_distance = None
        
        # Update net worth and track max drawdown
        position_value = self.shares * current_price if self.position_open else 0
        prev_net_worth = self.net_worth
        self.net_worth = self.balance + position_value
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        
        # Calculate return since last step (as a fraction of portfolio value)
        if prev_net_worth > 0:
            step_return = (self.net_worth - prev_net_worth) / prev_net_worth
        else:
            step_return = 0.0
        
        # Log raw return and net worth
        if self.debug and hasattr(self, 'log_rewards') and self.log_rewards:
            self._debug_print("\n[DEBUG_REWARD] Raw metrics:")
            self._debug_print(f"  Prev Net Worth: ${prev_net_worth:,.2f}")
            self._debug_print(f"  Current Net Worth: ${self.net_worth:,.2f}")
            self._debug_print(f"  Step Return: {step_return*100:.4f}%")
        
        # Calculate Sharpe ratio inspired reward (scale by volatility)
        # Using a small constant to avoid division by zero
        volatility = max(self.price_std / self.price_mean, 0.01)
        
        # Log volatility metrics
        if self.debug and hasattr(self, 'log_rewards') and self.log_rewards:
            self._debug_print(f"  Price Volatility: {volatility*100:.4f}%")
        
        # Scale reward by volatility to make it more consistent across different stocks
        # and time periods. The factor of 100 is to make the rewards more numerically stable.
        raw_reward = step_return / (volatility + 1e-8) * 100
        
        # Log raw reward before clipping
        if self.debug and hasattr(self, 'log_rewards') and self.log_rewards:
            self._debug_print(f"  Raw Reward (scaled): {raw_reward:.6f}")
        
        # Clip rewards to reasonable range to prevent extreme updates
        reward = np.clip(raw_reward, -10.0, 10.0)
        
        # Log if reward was clipped
        if self.debug and hasattr(self, 'log_rewards') and self.log_rewards and (reward <= -10.0 or reward >= 10.0):
            self._debug_print(f"  WARNING: Reward clipped from {raw_reward:.6f} to {reward:.6f}")
        
        # Add small penalty for holding positions to encourage decisive actions
        hold_penalty = 0.0
        if action_type == 0:  # HOLD
            hold_penalty = 0.01
            reward -= hold_penalty
            
            if self.debug and hasattr(self, 'log_rewards') and self.log_rewards:
                self._debug_print(f"  Hold Penalty: -{hold_penalty:.4f} (action=HOLD)")
        
        # Log final reward
        if self.debug and hasattr(self, 'log_rewards') and self.log_rewards:
            self._debug_print(f"  Final Reward: {reward:.6f}")
        
        # Check stoploss
        if self.position_open and self.stoploss_price is not None and current_price <= self.stoploss_price:
            if self.debug:
                self._debug_print(f"STOPLOSS TRIGGERED at {current_price:.2f} (stoploss: {self.stoploss_price:.2f})")
            
            self.balance += self.shares * current_price
            loss = (current_price - self.buy_price) * self.shares
            
            # Calculate stoploss reward (negative for loss, positive if stoploss saved us from bigger loss)
            potential_loss = (current_price - self.buy_price) / self.buy_price
            raw_stoploss_reward = np.tanh(potential_loss * 10)  # Scale the loss to [-1, 1] range
            reward = raw_stoploss_reward
            
            if self.debug and hasattr(self, 'log_rewards') and self.log_rewards:
                self._debug_print("\n[DEBUG_REWARD] Stoploss Triggered:")
                self._debug_print(f"  Buy Price: ${self.buy_price:.4f}")
                self._debug_print(f"  Trigger Price: ${current_price:.4f}")
                self._debug_print(f"  Potential Loss: {potential_loss*100:.2f}%")
                self._debug_print(f"  Raw Stoploss Reward: {raw_stoploss_reward:.6f}")
                self._debug_print(f"  Final Stoploss Reward: {reward:.6f}")
            
            self.position_open = False
            self.shares = 0
            self.buy_price = 0
            self.stoploss_price = None
            self.stoploss_distance = None
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        if self.current_step >= len(self.df) - 1 or self.current_step >= self.max_steps:
            done = True
        
        # Get next observation
        obs = self._get_obs()
        
        # Add info
        info['current_step'] = self.current_step
        info['balance'] = self.balance
        info['shares'] = self.shares
        info['position_open'] = self.position_open
        info['net_worth'] = self.net_worth
        
        return obs, reward, done, False, info
    
    def get_valid_actions(self):
        """Get valid actions based on current state.
        
        Returns:
            list: List of valid action indices (0=HOLD, 1=BUY, 2=SELL)
        """
        valid_actions = [0]  # HOLD is always valid
        
        # Check if we can BUY (only when no position is open and sufficient balance)
        if not self.position_open and self.df is not None and self.current_step < len(self.df):
            try:
                current_row = self.df.iloc[self.current_step]
                current_price = current_row['close']
                if current_price > 0 and self.balance >= current_price * 1000:  # Minimum 1000 shares
                    valid_actions.append(1)  # BUY
            except (IndexError, KeyError, AttributeError) as e:
                if self.debug:
                    self._debug_print(f"[WARNING] Error checking BUY conditions: {str(e)}")
        
        # Check if we can SELL (only when position is open)
        if self.position_open and self.shares > 0:
            valid_actions.append(2)  # SELL
            
        if self.debug and self.current_step % 10 == 0:  # Log every 10 steps to avoid spam
            self._debug_print(f"[Step {self.current_step}] Valid actions: {valid_actions}")
            self._debug_print(f"  Position: {'OPEN' if self.position_open else 'CLOSED'}, "
                           f"Shares: {self.shares}, "
                           f"Balance: ${self.balance:,.2f}, "
                           f"Net Worth: ${self.net_worth:,.2f}")
            
        return valid_actions
    
    def render(self, mode='human'):
        """Render the environment (optional)"""
        if mode == 'human':
            print(f"Step: {self.current_step}, "
                  f"Balance: ${self.balance:,.2f}, "
                  f"Shares: {self.shares}, "
                  f"Position: {'OPEN' if self.position_open else 'CLOSED'}, "
                  f"Net Worth: ${self.net_worth:,.2f}")
    
    def close(self):
        """Clean up resources"""
        if self.debug:
            self._debug_print("\n=== Environment Closed ===")
            self._debug_print(f"Final Balance: ${self.balance:,.2f}")
            self._debug_print(f"Final Net Worth: ${self.net_worth:,.2f} ({(self.net_worth/self.initial_balance-1)*100:.2f}%)")
