import gymnasium as gym
from gymnasium import Env,Wrapper
from gymnasium.spaces import Discrete, Box
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

# TradingEnv
debug = False

class TradingEnv(Env):
    def __init__(self, df):
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
        
        if debug: 
            print("Observation with mask:", combined_obs)

        # Debugging output
        if debug:
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

        if debug:
            print(f"Environment reset: Initial Observation (shape {obs.shape}): {obs}")
        
        return obs, {}

    def step(self, action): 
        if debug: print("Current Step:", self.current_step)
        
        if self.current_step >= len(self.df) - 1:
            return np.zeros(self.observation_space.shape), 0, True, False, {}
    
        close_price = self.df.iloc[self.current_step]["close"]
        if debug: print(' Close_price:', close_price)
        
        reward = 0
        done = False
        truncated = False
        valid_actions = self.get_valid_actions()
        
        # Create an action mask (1 for valid, 0 for invalid actions)
        action_mask = np.zeros(3, dtype=np.float32)
        for act in valid_actions:
            action_mask[act] = 1.0

        if debug: print(' Action', action)

        if action == 1:  # Buy
            self.shares = 1000
            self.balance -= close_price*self.shares
            self.position_open = True
            self.buy_price = close_price  # Store the buy price
            
        elif action == 2:  # Sell
            self.balance += close_price*self.shares
            reward = (close_price - self.buy_price)*self.shares  # Reward profit
            if debug: print('Trade profit (PnL):', reward)
            self.shares = 0
            self.buy_price = 0
            self.position_open = False
            self.round_trip_trades += 1

        # Update the total reward for the episode
        self.total_reward += reward ** 3
    
        # Calculate the net worth
        self.net_worth = self.balance + (self.shares * close_price)

        # Update the max net worth
        self.max_net_worth = max(self.net_worth, self.max_net_worth)

        if debug: print(" Market Data:", self.df.iloc[self.current_step][["open", "high", "low", "close", "volume"]].values)
        if debug: print(" Shares:", self.shares)
        if debug: print(" Balance:", self.balance)
        if debug: print(" Market position:", self.shares*close_price)
        if debug: print(" Net Worth:", self.net_worth)
            
        # Terminate after 10 round-trip trades or end of data
        if self.round_trip_trades >= 10:
            done = True
        if self.current_step >= len(self.df) - 1:
            done = True
        # Check if the episode is done
        if self.net_worth <= 0:
            done = True

        if debug: print(" Max Steps:", self.max_steps)
            
        self.current_step += 1
        
        if debug: print(f"Step: {self.current_step}, Valid Actions: {valid_actions}, Action Mask: {action_mask}, Selected Action: {original_action}, Invalid Action: {str(original_action not in valid_actions)}")
        obs = self.get_obs()  # Use the updated observation method
        info = {"valid_actions": valid_actions, "action_mask": action_mask}  # Include action mask in info

        if debug: 
            print(f"  - New Observation (shape {obs.shape}): {obs}")
            print(f"  - Action Mask (shape {action_mask.shape}): {action_mask}")
        
        return obs, reward, done, truncated, info

class RewardCallback(BaseCallback):
    def __init__(self, verbose=0, debug=False):
        super(RewardCallback, self).__init__(verbose)
        self.debug = debug
        
        # Metrics for tracking actions and rewards
        self.episode_rewards = []
        self.episode_steps = []
        self.iteration_rewards = []
        self.iteration_invalid_actions = []
        self.invalid_actions = []
        self.valid_actions = []
        self.current_episode_steps = 0
        
        # Metrics for TensorBoard logging
        self.total_reward = 0
        self.reward = 0
        self.num_trades = 0

    def _on_step(self) -> bool:
        # Collect rewards and actions
        rewards = self.locals.get("rewards", [])
        actions = self.locals.get("actions", [])
        
        if len(rewards) > 0:  # Check if rewards is not empty
            self.episode_rewards.extend(rewards)
        if len(actions) > 0:  # Check if actions is not empty
            infos = self.locals.get("infos", [])
            for idx, info in enumerate(infos):
                valid_actions = info.get("valid_actions", [0, 1, 2])
                action = actions[idx]
                if action not in valid_actions:
                    self.invalid_actions.append(action)
                else:
                    self.valid_actions.append(action)

        self.current_episode_steps += 1

        # Access the environment metrics using get_attr for SubprocVecEnv
        if isinstance(self.training_env, SubprocVecEnv):
            try:
                inner_envs = self.training_env.get_attr('env')  # ActionMasker
                for env in inner_envs:
                    if hasattr(env, 'env'):  # Unwrap ActionMasker
                        env = env.env
                    self.total_reward += getattr(env, "total_reward", 0)
                    self.num_trades += getattr(env, "round_trip_trades", 0)
            except Exception as e:
                if self.debug:
                    print(f"Failed to access env attributes: {e}")
        else:
            # For DummyVecEnv or single environments
            for env in self.training_env.envs:
                if hasattr(env, 'env'):
                    env = env.env
                self.total_reward += getattr(env, "total_reward", 0)
                self.num_trades += getattr(env, "round_trip_trades", 0)

        # TensorBoard logging
        self.logger.record("custom/num_trades", self.num_trades)
        self.logger.record("custom/total_reward", self.total_reward)

        # Entropy logging
        if hasattr(self.model.policy, "action_dist"):
            action_dist = self.model.policy.action_dist
            entropy = action_dist.entropy().mean().item()
            self.logger.record("policy/entropy", entropy)
        elif hasattr(self.model.policy, "get_distribution"):
            obs = self.locals.get("obs", [])
            if len(obs) > 0:  # Check if observations exist
                action_dist = self.model.policy.get_distribution(obs)
                entropy = action_dist.entropy().mean().item()
                self.logger.record("policy/entropy", entropy)

        # Value loss logging
        if "value_loss" in self.locals:
            value_loss = self.locals["value_loss"]
            self.logger.record("loss/value_loss", value_loss)

        # Episode done handling
        dones = self.locals.get("dones", [])
        if any(dones):
            self.episode_steps.append(self.current_episode_steps)
            self.current_episode_steps = 0
            total_reward = np.sum(self.episode_rewards)
            self.iteration_rewards.append(total_reward)
            self.episode_rewards = []

            invalid_count = len(self.invalid_actions)
            valid_count = len(self.valid_actions)
            self.iteration_invalid_actions.append(invalid_count)

            if self.debug:
                print(f"Invalid actions in this episode: {invalid_count}")
                print(f"Valid actions in this episode: {valid_count}")
                print(f"Invalid actions: {self.invalid_actions}")

            self.invalid_actions = []

        return True



