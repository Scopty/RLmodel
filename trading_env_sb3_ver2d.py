import gymnasium as gym
from gymnasium import Env,Wrapper
from gymnasium.spaces import Discrete, Box
import numpy as np


class TradingEnv(Env):
    def __init__(self, df, debug=False):
        super(TradingEnv, self).__init__()
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
        self.max_steps = 0
        self.debug = debug
        reward = 0

        # Action Space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = Discrete(3)

        # Features: close price, shares, balance, net worth, current step
        self.obs_shape = 8  # Original shape to match trained model
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float32
        )

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

    def get_valid_actions(self):
        valid_actions = [0]  # Hold is always valid
        if not self.position_open:  # Allow Buy only if no position is open
            valid_actions.append(1)
        if self.position_open:  # Allow Sell only if a position is open
            valid_actions.append(2)
        return valid_actions

    def get_current_step(self):
        """Return the current step in the environment."""
        return self.current_step

    def get_obs(self):
        # Ensure all observations are float32
        obs = np.concatenate([
            self.df.iloc[self.current_step][["close"]].values.astype(np.float32),  # (1,)
            np.array([self.shares], dtype=np.float32),   # (1,)
            np.array([self.balance], dtype=np.float32),  # (1,)
            np.array([self.net_worth], dtype=np.float32),# (1,)
            np.array([self.current_step], dtype=np.float32),  # (1,)
            np.zeros(3, dtype=np.float32)  # Add zeros for mask placeholders
        ])
        
        if self.debug:
            print(f"Step {self.current_step}:")
            print(f"  - Observation (shape {obs.shape}): {obs}")
        
        return obs
    
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
        reward = 0
        obs = self.get_obs()  # Use the updated observation method

        if self.debug:
            print(f"Environment reset: Initial Observation (shape {obs.shape}): {obs}")
        
        return obs, {}

    def step(self, action): 
        if self.debug: 
            print("Current Step:", self.current_step)
        
        if self.current_step >= len(self.df) - 1:
            return np.zeros(self.observation_space.shape), 0, True, False, {}
    
        close_price = self.df.iloc[self.current_step]["close"]
        reward = 0
        done = False
        truncated = False
        valid_actions = self.get_valid_actions()
        
        # Create an action mask (1 for valid, 0 for invalid actions)
        action_mask = np.zeros(3, dtype=np.float32)
        for act in valid_actions:
            action_mask[act] = 1.0

        if self.debug: print(' Action', action)

        if action == 1:  # Buy
            # Buy 100% of available balance
            shares = int(self.balance / close_price)
            if shares > 0:
                self.shares = shares
                self.balance -= close_price * shares
                self.position_open = True
                self.buy_price = close_price
                reward = -0.0001 * shares  # Very small transaction cost
                if self.debug: print(f'Buy {shares} shares at price: {close_price}')
            else:
                reward = -0.05  # Penalty for trying to buy with insufficient funds
                if self.debug: print('Insufficient funds to buy')
            
        elif action == 2:  # Sell
            if self.shares > 0:
                self.balance += close_price * self.shares
                reward = (close_price - self.buy_price) * self.shares  # Actual profit
                if self.debug: print(f'Sell profit: {reward}')
                self.shares = 0
                self.buy_price = 0
                self.position_open = False
                self.round_trip_trades += 1
            else:
                reward = -0.05  # Penalty for trying to sell without shares
                if self.debug: print('No shares to sell')

        else:  # Hold
            if self.position_open:
                profit = (close_price - self.buy_price) * self.shares
                reward = profit * 0.01  # Higher reward for holding profitable positions
                if self.debug: print(f'Holding profit: {reward}')
            else:
                reward = -0.001  # Small penalty for holding without position
        
        # Update the total reward for the episode
        self.total_reward += reward
    
        # Calculate the net worth
        self.net_worth = self.balance + (self.shares * close_price)

        # Update the max net worth
        self.max_net_worth = max(self.net_worth, self.max_net_worth)

        #if self.debug: print(" Market Data:", self.df.iloc[self.current_step][["open", "high", "low", "close", "volume"]].values)
        if self.debug: print(" Shares:", self.shares)
        if self.debug: print(" Balance:", self.balance)
        if self.debug: print(" Market position:", self.shares*close_price)
        if self.debug: print(" Net Worth:", self.net_worth)
            
        # Terminate after 10 round-trip trades or end of data
        if self.round_trip_trades >= 10:
            done = True
        if self.current_step >= len(self.df) - 1:
            done = True
        # Check if the episode is done
        if self.net_worth <= 0:
            done = True

        if self.debug: print(" Max Steps:", self.max_steps)
            
        self.current_step += 1
        
        if self.debug: print(f"Step: {self.current_step}, Action: {action}, Valid Actions: {valid_actions}, Action Mask: {action_mask}")
        obs = self.get_obs()  # Use the updated observation method
        # Track if the action was invalid
        is_invalid = action not in valid_actions
        info = {
            "valid_actions": valid_actions, 
            "action_mask": action_mask,
            "invalid_action": is_invalid
        }  # Include action mask and invalid action flag in info

        if self.debug: 
            print(f"  - New Observation (shape {obs.shape}): {obs}")
            print(f"  - Action Mask (shape {action_mask.shape}): {action_mask}")
        
        return obs, reward, done, truncated, info

