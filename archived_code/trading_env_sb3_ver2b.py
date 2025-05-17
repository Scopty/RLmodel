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

        # ohlcv, balance, net worth, shares held, current step
        self.obs_shape = 8  # 9 original features + 3 mask elements
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float32
        )

    def get_action_mask(self) -> np.ndarray:
        """
        Returns a binary mask indicating valid actions.
        1 indicates a valid action, 0 indicates an invalid action.
        """
        action_mask = np.zeros(self.action_space.n, dtype=np.float32)
        for act in self.get_valid_actions():
            action_mask[act] = 1.0
        return action_mask
    
    def get_valid_actions(self):
        valid_actions = [0]  # Hold is always valid
        if not self.position_open:  # Allow Buy only if no position is open
            valid_actions.append(1)
        if self.position_open:  # Allow Sell only if a position is open
            valid_actions.append(2)
        return valid_actions

    def get_obs(self):

        # Include the current position
        obs = np.concatenate([
            self.df.iloc[self.current_step][["close"]].values.astype(np.float64),  # (5,)
            np.array([self.shares], dtype=np.float64),   # (1,)
            np.array([self.balance], dtype=np.float64),  # (1,)
            np.array([self.net_worth], dtype=np.float64),# (1,)
            np.array([self.current_step], dtype=np.float64)  # (1,)
        ])
    
          # Get the action mask (for instance, shape: (3,))
        mask = self.get_action_mask()  # This should return an array like [1, 0, 1]
        
        # Concatenate the observation with the mask (resulting shape: (9+3=12,))
        combined_obs = np.concatenate([obs, mask])
        
        if self.debug: 
            print("Observation with mask:", combined_obs)

        # Debugging output
        if self.debug:
            print(f"Step {self.current_step}:")
            print(f"  - Original Observation (shape {obs.shape}): {obs}")
            print(f"  - Action Mask (shape {mask.shape}): {mask}")
            print(f"  - Combined Observation (shape {combined_obs.shape}): {combined_obs}")

        
        return combined_obs
    
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
            self.shares = 1000
            self.balance -= close_price * self.shares
            self.position_open = True
            self.buy_price = close_price
            reward = -0.01  # Small cost to enter trade
            if self.debug: print('Buy at price:', close_price)
            
        elif action == 2:  # Sell
            self.balance += close_price*self.shares
            reward = (close_price - self.buy_price)*self.shares / 1000  # Reward profit
            if self.debug: print('Trade profit (PnL):', reward)
            self.shares = 0
            self.buy_price = 0
            self.position_open = False
            self.round_trip_trades += 1

        else:
            # Reward for holding a profitable position
            if self.position_open:
                profit = (close_price - self.buy_price) / self.buy_price
                reward = profit * 0.1  # Small reward for holding profitable positions
            else:
                reward = 0  # No penalty for holding without position
        
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

