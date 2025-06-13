import numpy as np
import pandas as pd
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
import gymnasium
import os
import datetime
from pathlib import Path

class TradingEnv(Env):
    def __init__(self, df, debug=False, max_steps=None, model_name=None, input_dir_name=None, test_mode=False):
        super().__init__()
        
        # Define action and observation space first (required by gym.Env)
        # Action space: 0=HOLD, 1=BUY, 2=SELL
        self.action_space = Discrete(3)
        
        # Observation space: 13 features
        # 1. Normalized OHLCV (5 features)
        # 2. Position info (1 feature: position_open)
        # 3. Time features (3 features: hour, minute, seconds_since_midnight)
        # 4. Time until market close (1 feature)
        # 5. Time of day features (3 features: is_pre_market, is_after_hours, normalized_time_until_close)
        self.observation_space = Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(13,),  # 13 features total
            dtype=np.float32
        )
        
        # If datetime is the index, reset it to be a column
        if 'datetime' not in df.columns:
            df = df.reset_index()
            df.rename(columns={'Date': 'datetime'}, inplace=True)
        
        self.df = df
        self.current_step = 0
        self.initial_balance = 10000  # Starting cash
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.shares = 0
        self.buy_price = 0
        self.total_reward = 0
        self.position_open = False  # Track open positions
        self.round_trip_trades = 0  # Counter for buy-sell cycles
        self.max_steps = max_steps or len(df) - 1  # Use provided max_steps or default to df length
        self.debug = debug
        self.market_open = pd.Timestamp('04:00:00').time()  # 4:00 AM
        self.market_close = pd.Timestamp('20:00:00').time()  # 8:00 PM
        self.model_name = model_name
        self.input_dir_name = input_dir_name or 'default'
        self.test_mode = test_mode
        
        # Initialize debug logging if enabled and in test mode
        if self.debug and self.test_mode:
            self._init_debug_logging()

        # Preprocess time information
        self._preprocess_time()

    def _init_debug_logging(self):
        """Initialize debug logging."""
        # Create debug logs directory with input directory name
        debug_dir = os.path.join('debug_logs', self.input_dir_name)
        os.makedirs(debug_dir, exist_ok=True)
        
        if self.model_name:
            # Extract just the model name without the full path
            model_base = os.path.basename(str(self.model_name))
            self.debug_file = os.path.join(debug_dir, f'trading_env_debug_{model_base}.txt')
        else:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            self.debug_file = os.path.join(debug_dir, f'trading_env_debug_{timestamp}.txt')
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.debug_file), exist_ok=True)
        
        # Set debug_file_path for backward compatibility
        self.debug_file_path = self.debug_file
            
        # Write header to debug file
        with open(self.debug_file, 'w') as f:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"TradingEnv Debug Log - {timestamp}\n")
            if self.model_name:
                f.write(f"Model: {self.model_name}\n")
            f.write("=" * 40 + "\n\n")
            f.flush()  # Ensure header is written immediately

    def _debug_print(self, message):
        """Write debug message to debug log file only."""
        # Skip market hours debug messages
        if 'Market day duration:' in message or 'Is pre-market:' in message or 'Is after-hours:' in message:
            return
            
        # Don't include timestamp in the debug message
        debug_message = message
        
        # Ensure debug file is initialized
        if not hasattr(self, 'debug_file'):
            self._init_debug_logging()
            
        try:
            with open(self.debug_file, 'a') as f:
                f.write(f"{debug_message}\n")
                f.flush()  # Ensure message is written immediately
        except Exception as e:
            print(f"Error writing to debug file: {e}", file=sys.stderr)

        # Features: open, high, low, close, volume, shares, balance, net worth, current step
        obs_high = np.array([
            np.finfo(np.float32).max,  # open
            np.finfo(np.float32).max,  # high
            np.finfo(np.float32).max,  # low
            np.finfo(np.float32).max,  # close
            np.finfo(np.float32).max,  # volume
            np.finfo(np.float32).max,  # shares
            np.finfo(np.float32).max,  # balance
            np.finfo(np.float32).max,  # net_worth
            1.0,  # position_open
            1.0,  # normalized_time
            1.0,  # is_pre_market
            1.0,  # is_after_hours
            1.0   # normalized_time_until_close
        ], dtype=np.float32)

        self.observation_space = gymnasium.spaces.Box(
            low=-np.finfo(np.float32).max,
            high=obs_high,
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.shares = 0
        self.position_open = False
        self.round_trip_trades = 0
        self.buy_price = 0
        self.max_steps = 0
        self.total_reward = 0
        self.buy_step = None  # Reset buy step to None (not 0)

        # Get initial observation
        obs = self.get_obs()

        return obs, {}  # Return observation and empty info dictionary

    def _preprocess_time(self):
        """Calculate time-based features."""
        # Convert datetime column to proper datetime format
        # Handle two-digit years by assuming they are in the 2000s
        self.df['datetime'] = pd.to_datetime(self.df['datetime'], format='%m/%d/%y %H:%M', exact=False)
        # Set the year to 2025 explicitly
        self.df['datetime'] = self.df['datetime'].apply(lambda x: x.replace(year=2025))
        
        # Extract hour and minute from datetime
        self.df['hour'] = self.df['datetime'].dt.hour
        self.df['minute'] = self.df['datetime'].dt.minute
        
        # Calculate market session duration in minutes
        # Convert time objects to minutes since midnight
        market_open_minutes = self.market_open.hour * 60 + self.market_open.minute
        market_close_minutes = self.market_close.hour * 60 + self.market_close.minute
        market_minutes = market_close_minutes - market_open_minutes
        if self.debug:
            self._debug_print(f"\nMarket day duration: {market_minutes} minutes")

        # Calculate time since market open (in minutes)
        # Convert current time to minutes since midnight
        self.df['time_since_open'] = (self.df['hour'] * 60 + self.df['minute']) - market_open_minutes
        
        # Normalize time since open to [0, 1] using market hours
        self.df['normalized_time'] = self.df['time_since_open'] / market_minutes
        if self.debug:
            self._debug_print(f"Normalized time (first row): {self.df['normalized_time'].iloc[0]}")
        
        # Calculate time until market close (in minutes)
        # 7:59 PM is market close, so calculate time remaining
        # Ensure time_until_close is always non-negative
        self.df['time_until_close'] = np.maximum(0, ((self.market_close.hour * 60 + self.market_close.minute) - (self.df['hour'] * 60 + self.df['minute'])))
        self.df['normalized_time_until_close'] = self.df['time_until_close'] / market_minutes
        if self.debug:
            self._debug_print(f"Normalized time until close (first row): {self.df['normalized_time_until_close'].iloc[0]}")
        
        # Define session periods
        pre_market_end = 9.5  # 9:30 AM
        after_hours_start = 16  # 4:00 PM
        
        # Calculate is_pre_market
        self.df['is_pre_market'] = ((self.df['hour'] >= self.market_open.hour) & 
                                 (self.df['hour'] < 9.5) & 
                                 (self.df['minute'] < 30)).astype(float)
        
        # Calculate is_after_hours
        self.df['is_after_hours'] = ((self.df['hour'] >= 16) & 
                                   (self.df['hour'] < self.market_close.hour) & 
                                   (self.df['minute'] < 59)).astype(float)
        
        # Debug print for specific hours
        if self.debug:
            self._debug_print("\nDebug information for session periods:")
            self._debug_print("4:00 AM - Start of day:")
            self._debug_print(f"Is pre-market: {((4 >= 4) and (4 < 9.5) and (0 < 30))}")
            self._debug_print(f"Is after-hours: {((4 >= 16) and (4 < 19) and (0 < 59))}")
            
            self._debug_print("\n9:30 AM - End of pre-market:")
            self._debug_print(f"Is pre-market: {((9.5 >= 4) and (9.5 < 9.5) and (0 < 30))}")
            self._debug_print(f"Is after-hours: {((9.5 >= 16) and (9.5 < 19) and (0 < 59))}")
            
            self._debug_print("\n4:00 PM - Start of after-hours:")
            self._debug_print(f"Is pre-market: {((16 >= 4) and (16 < 9.5) and (0 < 30))}")
            self._debug_print(f"Is after-hours: {((16 >= 16) and (16 < 19) and (0 < 59))}")
            
            self._debug_print("\n7:59 PM - End of day:")
            self._debug_print(f"Is pre-market: {((19 >= 4) and (19 < 9.5) and (59 < 30))}")
            self._debug_print(f"Is after-hours: {((19 >= 16) and (19 < 19) and (59 < 59))}")

    def get_obs(self):
        # Get current row of data
        current_row = self.df.iloc[self.current_step]
        
        # Extract time features
        normalized_time = current_row['normalized_time']
        is_pre_market = current_row['is_pre_market']
        is_after_hours = current_row['is_after_hours']
        normalized_time_until_close = current_row['normalized_time_until_close']
        
        # Ensure all observations are float32
        obs = np.concatenate([
            current_row[['open']].values.astype(np.float32),  # (1,)
            current_row[['high']].values.astype(np.float32),  # (1,)
            current_row[['low']].values.astype(np.float32),   # (1,)
            current_row[['close']].values.astype(np.float32), # (1,)
            current_row[['volume']].values.astype(np.float32),# (1,)
            np.array([self.shares], dtype=np.float32),        # (1,)
            np.array([self.balance], dtype=np.float32),       # (1,)
            np.array([self.net_worth], dtype=np.float32),     # (1,)
            np.array([float(self.position_open)], dtype=np.float32),  # (1,)
            np.array([normalized_time], dtype=np.float32),  # (1,)
            np.array([is_pre_market], dtype=np.float32),  # (1,)
            np.array([is_after_hours], dtype=np.float32),   # (1,)
            np.array([normalized_time_until_close], dtype=np.float32)  # (1,)
        ])
        
        return obs

    def _calculate_reward(self, action, current_row):
        """Calculate reward for the given action.
        
        Rewards are structured to:
        - Heavily penalize buying to strongly discourage overtrading
        - Reward holding longer positions more
        - Only reward selling after minimum hold time
        """
        reward = 0
        current_price = current_row['close']
        
        # Calculate holding period if in a position
        hold_steps = (self.current_step - self.buy_step) if self.position_open else 0
        
        # Reward for holding to encourage patience
        if action == 0:  # Hold action
            if self.position_open:
                # Scale reward with holding time (more reward the longer held)
                hold_bonus = min(0.1 * (1 + hold_steps * 0.1), 1.0)  # Up to 1.0 max bonus (10x)
                reward = 0.5 + hold_bonus  # Base reward + scaled bonus (10x)
                if self.debug:
                    self._debug_print(f"[DEBUG_REWARD] HOLD action - Reward: +{reward:.4f} "
                                     f"(held for {hold_steps} steps, bonus: {hold_bonus:.4f})")
            else:
                # Minimal reward when not in a position
                reward = 0.001  # 10x
                if self.debug:
                    self._debug_print(f"[DEBUG_REWARD] HOLD action - Reward: +{reward:.4f} (no position)")
        
        # Strongly penalize buying to discourage overtrading
        elif action == 1:  # Buy action
            if not self.position_open and self.balance >= current_price * 1000:
                # Base penalty + additional penalty based on recent trades
                reward = -1.0  # 10x increased base penalty
                if self.debug:
                    self._debug_print(f"[DEBUG_REWARD] BUY action - Penalty: {reward:.4f}")
            else:
                # Invalid BUY action - should be masked
                reward = 0
                if self.debug:
                    self._debug_print("[DEBUG_REWARD] BUY action - Invalid (masked), reward: 0")
        
        # Reward selling based on position P&L and hold time
        elif action == 2:  # Sell action
            if self.position_open and self.shares > 0:
                # Base profit/loss
                reward = (current_price - self.buy_price) * self.shares
                
                if self.debug:
                    self._debug_print(f"[DEBUG_REWARD] SELL action - Raw profit: ${reward:.2f} "
                                     f"(Bought @ {self.buy_price:.4f}, Sold @ {current_price:.4f}, "
                                     f"Shares: {self.shares}, Held for: {hold_steps} steps)")
                
                # Strong penalty for very quick trades (10x)
                if hold_steps < 10:  # First 10 steps
                    quick_sell_penalty = 1.0 * (10 - hold_steps)  # Up to 10.0 penalty (10x)
                    reward -= quick_sell_penalty
                    if self.debug:
                        self._debug_print(f"[DEBUG_REWARD]   Quick sell penalty: -{quick_sell_penalty:.4f}")
                
                # Bonus for holding longer (10x)
                if hold_steps >= 10:  # Only reward if held for minimum period
                    hold_bonus = min(0.01 * hold_steps, 0.5)  # Up to 0.5 bonus (10x)
                    reward += hold_bonus
                    if self.debug:
                        self._debug_print(f"[DEBUG_REWARD]   Hold time bonus: +{hold_bonus:.4f}")
                
                # Additional profit bonus (only for profitable trades held long enough) (10x)
                if reward > 0 and hold_steps >= 5:
                    profit_bonus = min(reward * 0.1, 1.0)  # Up to 1.0 bonus (10x)
                    reward += profit_bonus
                    if self.debug:
                        self._debug_print(f"[DEBUG_REWARD]   Profit bonus: +{profit_bonus:.4f}")
                
                if self.debug:
                    self._debug_print(f"[DEBUG_REWARD]   Final reward: ${reward:.4f}")
            else:
                # Invalid SELL action - should be masked
                reward = 0
                if self.debug:
                    self._debug_print("[DEBUG_REWARD] SELL action - Invalid (masked), reward: 0")
        
        self.total_reward += reward
        return reward

    def _update_state(self, action, current_row):
        """Update environment state based on the action taken."""
        # Store previous state for validation
        prev_position_open = self.position_open
        prev_shares = self.shares
        
        if action == 1:  # Buy
            if not self.position_open and self.balance >= current_row['close'] * 1000:
                self.shares = 1000
                self.balance -= current_row['close'] * 1000
                self.position_open = True
                self.buy_price = current_row['close']
                self.buy_step = self.current_step
                self.net_worth = self.balance + (self.shares * current_row['close'])
                self.max_net_worth = max(self.max_net_worth, self.net_worth)
                # Record successful buy signal
                if self.debug:
                    self._debug_print(f"[DEBUG] BUY executed at {current_row['datetime']} - "
                                     f"Price: {current_row['close']}, "
                                     f"Shares: {self.shares}, "
                                     f"Balance: {self.balance}, "
                                     f"Net Worth: {self.net_worth}")
            elif self.debug:
                self._debug_print(f"[DEBUG] BUY failed - Position open: {self.position_open}, "
                                  f"Balance: {self.balance}, "
                                  f"Required: {current_row['close'] * 1000}")
        elif action == 2:  # Sell
            if self.position_open and self.shares > 0:
                sell_value = current_row['close'] * self.shares
                profit = sell_value - (self.buy_price * self.shares)
                self.balance += sell_value
                self.shares = 0
                self.position_open = False
                self.buy_price = 0
                self.buy_step = None
                self.net_worth = self.balance
                self.max_net_worth = max(self.max_net_worth, self.net_worth)
                # Record successful sell signal
                if self.debug:
                    self._debug_print(f"[DEBUG] SELL executed at {current_row['datetime']} - "
                                     f"Price: {current_row['close']}, "
                                     f"Profit: {profit}, "
                                     f"Balance: {self.balance}, "
                                     f"Net Worth: {self.net_worth}")
            else:
                if self.debug:
                    self._debug_print(f"[DEBUG] SELL failed - Position open: {self.position_open}, "
                                      f"Shares: {self.shares}")
                # Just ensure buy_step is None if no shares
                if self.shares == 0:
                    self.buy_step = None
        else:  # Hold
            if self.position_open:
                self.net_worth = self.balance + (self.shares * current_row['close'])
                self.max_net_worth = max(self.max_net_worth, self.net_worth)
            else:
                self.net_worth = self.balance
                self.max_net_worth = max(self.max_net_worth, self.net_worth)
        
        # Validate state consistency
        if self.shares > 0:
            assert self.position_open, f"Shares > 0 but position_open=False (shares: {self.shares}, position_open: {self.position_open})"
            assert self.buy_price > 0, f"Shares > 0 but buy_price=0 (shares: {self.shares}, buy_price: {self.buy_price})"
            assert self.buy_step is not None, f"Shares > 0 but buy_step is None (shares: {self.shares}, buy_step: {self.buy_step})"
        else:
            assert not self.position_open, f"Shares == 0 but position_open=True (shares: {self.shares}, position_open: {self.position_open})"
            assert self.buy_price == 0, f"Shares == 0 but buy_price > 0 (shares: {self.shares}, buy_price: {self.buy_price})"
            assert self.buy_step is None, f"Shares == 0 but buy_step is not None (shares: {self.shares}, buy_step: {self.buy_step})"
            
        # Update action mask based on current state
        action_mask = np.zeros(self.action_space.n, dtype=bool)
        if not self.position_open:  # Allow Buy only if no position is open
            action_mask[0] = True  # Hold
            action_mask[1] = True  # Buy
        else:  # Allow Sell only if position is open
            action_mask[0] = True  # Hold
            action_mask[2] = True  # Sell
        
        return action_mask

    def get_valid_actions(self):
        """Get valid actions based on current state."""
        valid_actions = [0]  # Hold is always valid
        if not self.position_open:  # Allow Buy only if no position is open
            valid_actions.append(1)
        if self.position_open:  # Allow Sell only if position is open
            valid_actions.append(2)
        return valid_actions

    def get_action_mask(self) -> np.ndarray:
        """
        Returns a binary mask indicating valid actions.
        True indicates a valid action, False indicates an invalid action.
        """
        action_mask = np.zeros(self.action_space.n, dtype=bool)
        for act in self.get_valid_actions():
            action_mask[act] = True
        return action_mask
    
    def action_masks(self) -> np.ndarray:
        """
        Returns a binary mask indicating valid actions.
        True indicates a valid action, False indicates an invalid action.
        """
        action_mask = np.zeros(self.action_space.n, dtype=bool)
        for act in self.get_valid_actions():
            action_mask[act] = True
        return action_mask

    def step(self, action):
        """Take a step in the environment."""
        # Get valid actions and action mask BEFORE updating the state
        valid_actions_before = self.get_valid_actions()
        action_mask_before = self.get_action_mask()
        
        # Track if the action is invalid based on the CURRENT state
        is_invalid = action not in valid_actions_before
        
        if self.debug:
            action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
            # Safely convert action to int if it's a numpy array
            action_int = int(action) if hasattr(action, 'item') else int(action) if isinstance(action, (int, float)) else action
            
            self._debug_print(f"\n[DEBUG] Step {self.current_step} - Action: {action_names.get(action_int, 'UNKNOWN')} ({action_int})")
            self._debug_print(f"[DEBUG] Valid actions: {[action_names.get(a, '?') for a in valid_actions_before]}")
            self._debug_print(f"[DEBUG] Position open: {self.position_open}, Shares: {self.shares}, Balance: {self.balance}")
            
            if is_invalid:
                self._debug_print(f"[WARNING] Invalid action selected!")
            if action_int == 1 and self.position_open:
                self._debug_print("[WARNING] Attempting to BUY when position is already open!")
            if action_int == 2 and not self.position_open:
                self._debug_print("[WARNING] Attempting to SELL when no position is open!")
        
        # Check if we've reached the end of the episode
        if self.current_step >= len(self.df) - 1:
            # Return done=True if we've reached the end of the data
            # Use the last valid row for the observation
            return self.get_obs(), 0, True, False, {}
            
        # Get current row data
        current_row = self.df.iloc[self.current_step]
        
        # Calculate reward
        reward = self._calculate_reward(action, current_row)
        
        # Log action execution details
        action = int(action) if hasattr(action, 'item') else int(action)  # Convert numpy types to Python int
        executed_action = action
        action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        
        if is_invalid:
            # If invalid, choose the first valid action as fallback
            if valid_actions_before:
                executed_action = int(valid_actions_before[0])  # Default to first valid action
                if self.debug:
                    self._debug_print(f"[DEBUG_ACTION] Invalid action {action_names.get(action, 'UNKNOWN')} selected. "
                                    f"Falling back to {action_names.get(executed_action, 'UNKNOWN')}")
            else:
                executed_action = 0  # Default to HOLD if no valid actions
                if self.debug:
                    self._debug_print("[DEBUG_ACTION] No valid actions available, defaulting to HOLD")
    
        # Log the action that will be executed
        if self.debug:
            self._debug_print(f"[DEBUG_ACTION] Model action: {action_names.get(action, 'UNKNOWN')} ({action})")
            self._debug_print(f"[DEBUG_ACTION] Executing action: {action_names.get(executed_action, 'UNKNOWN')} ({executed_action})")
            self._debug_print(f"[DEBUG_ACTION] Valid actions: {[action_names.get(a, '?') for a in valid_actions_before]}")
    
        # Update the state with the executed action
        self._update_state(executed_action, current_row)
        
        # Get new observation after state update
        obs = self.get_obs()
        
        # Move to next step at the end of the current step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.df)
        
        # Get valid actions for next step (after state update)
        next_valid_actions = self.get_valid_actions()
        next_action_mask = self.get_action_mask()
        
        info = {
            "valid_actions": next_valid_actions, 
            "action_mask": next_action_mask,
            "invalid_action": is_invalid,
            "previous_valid_actions": valid_actions_before,
            "model_action": action,
            "executed_action": executed_action,
            "action_valid": not is_invalid,
            "position_open": self.position_open,
            "shares": self.shares,
            "balance": self.balance,
            "net_worth": self.net_worth
        }
        
        if self.debug:
            action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
            self._debug_print(f"[DEBUG] Step {self.current_step} completed")
            self._debug_print(f"[DEBUG] Action taken: {action_names.get(action, 'UNKNOWN')} ({action})")
            self._debug_print(f"[DEBUG] Next valid actions: {[action_names[a] for a in next_valid_actions]}")
            self._debug_print(f"[DEBUG] Position open: {self.position_open}, Shares: {self.shares}, Balance: {self.balance}, Net Worth: {self.net_worth}")
            self._debug_print(f"[DEBUG] Current price: {current_row['close']}")
            if self.position_open and self.buy_price > 0:
                profit_pct = ((current_row['close'] - self.buy_price) / self.buy_price) * 100
                self._debug_print(f"[DEBUG] Position profit: {profit_pct}%")
            self._debug_print("-" * 40)
        
        return obs, reward, done, False, info
