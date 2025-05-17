from gymnasium import Env,Wrapper
from gymnasium.spaces import Discrete, Box
import numpy as np

class TradingEnv(Env):
    def __init__(self, df, debug=False):
        super(TradingEnv, self).__init__()
        self.df = df
        self.debug = debug
        self.current_step = 0
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.shares = 0
        self.buy_price = 0
        self.total_reward = 0
        self.position_open = False
        self.round_trip_trades = 0
        self.max_steps = len(df)
        self.portfolio_value_history = []  # Track portfolio value over time

        # Action Space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = Discrete(3)

        # Observation space: [close, shares, balance, net_worth, current_step]
        # Define reasonable bounds for each component
        max_balance = 1e6  # Maximum expected balance
        max_shares = 1e6   # Maximum expected shares
        max_steps = len(df) * 2  # Allow some buffer
        
        self.observation_space = Box(
            low=np.array([0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([np.finfo(np.float32).max, max_shares, max_balance, max_balance, max_steps], 
                         dtype=np.float32),
            shape=(5,),
            dtype=np.float32
        )

    def get_obs(self):
        """Return only the observation without the action mask."""
        try:
            # Initialize with safe defaults
            close_price = 1.0
            shares = float(getattr(self, 'shares', 0))
            balance = float(getattr(self, 'balance', self.initial_balance))
            net_worth = float(getattr(self, 'net_worth', self.initial_balance))
            current_step = int(getattr(self, 'current_step', 0))
            
            # Safely get close price from DataFrame if available
            if hasattr(self, 'df') and self.df is not None and not self.df.empty:
                try:
                    # Ensure we don't go out of bounds
                    idx = min(current_step, len(self.df) - 1) if len(self.df) > 0 else 0
                    if 'close' in self.df.columns:
                        close_price = float(self.df.iloc[idx]['close'])
                        # Ensure we have a valid price
                        if np.isnan(close_price) or close_price <= 0:
                            close_price = 1.0
                except Exception as e:
                    print(f"Warning in get_obs: {e}")
                    close_price = 1.0  # Fallback value
            
            obs = np.array([
                close_price,
                shares,
                balance,
                net_worth,
                float(current_step)
            ], dtype=np.float32)
            
            if self.debug:
                print(f"\nTradingEnv get_obs:")
                print(f"  - Current step: {current_step}")
                print(f"  - obs shape: {obs.shape}")
                print(f"  - obs: {obs}")
                
            return obs
            
        except Exception as e:
            print(f"Error in get_obs: {str(e)}")
            # Return a safe default observation
            return np.array([1.0, 0.0, float(self.initial_balance), float(self.initial_balance), 0.0], dtype=np.float32)
            print(f"  - obs shape: {obs.shape}")
            print(f"  - obs: {obs}")
        
        return obs

    def get_action_mask(self):
        """Return action mask based on current state."""
        action_mask = np.zeros(self.action_space.n, dtype=np.float32)
        valid_actions = self.get_valid_actions()
        for act in valid_actions:
            action_mask[act] = 1.0
        if self.debug:
            print(f"\nAction mask: {action_mask}")
            print(f"Valid actions: {valid_actions}")
            print(f"Position open: {self.position_open}")
            print(f"Shares: {self.shares}, Balance: {self.balance}")
        return action_mask

    def get_valid_actions(self):
        valid_actions = [0]  # Hold is always valid
        if not self.position_open:
            valid_actions.append(1)  # Allow Buy
        if self.position_open:
            valid_actions.append(2)  # Allow Sell
        return valid_actions

    def reset(self, seed=None, options=None):
        try:
            super().reset(seed=seed)
            self.current_step = 0
            self.balance = float(self.initial_balance)
            self.net_worth = float(self.initial_balance)
            self.max_net_worth = float(self.initial_balance)
            self.shares = 0
            self.position_open = False
            self.round_trip_trades = 0
            self.buy_price = 0.0
            self.total_reward = 0.0
            
            obs = self.get_obs()
            if self.debug:
                print(f"\nTradingEnv reset:")
                print(f"  - Initial balance: {self.balance}")
                print(f"  - obs shape: {obs.shape}")
                print(f"  - obs: {obs}")
                print(f"  - Observation space: {self.observation_space}")
                print(f"  - Action space: {self.action_space}")
            
            info = {
                'valid_actions': self.get_valid_actions(),
                'action_mask': self.get_action_mask().tolist()
            }
            
            return obs, info
            
        except Exception as e:
            print(f"Error in reset: {str(e)}")
            # Return a safe default observation with proper info
            default_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            info = {
                'valid_actions': [0],  # At least hold is always valid
                'action_mask': [1.0, 0.0, 0.0]  # Only hold is valid initially
            }
            return default_obs, info

    def step(self, action):
        # Initialize variables
        reward = 0.0
        profit = 0.0
        truncated = False  # Initialize truncated flag
        
        try:
            if self.debug:
                print(f"\n=== Step {self.current_step} ===")
                print(f"Action: {action} (type: {type(action)})")
            
            # Ensure action is an integer
            try:
                action = int(action) if hasattr(action, 'item') else int(action)
            except (ValueError, TypeError) as e:
                print(f"Warning: Invalid action {action} received, defaulting to Hold (0)")
                action = 0
            if self.debug:
                print(f"\n=== Step {self.current_step} ===")
                print(f"Action: {action} (0=Hold, 1=Buy, 2=Sell)")
            
            # Get current price
            try:
                # Ensure we don't go out of bounds
                if self.current_step >= len(self.df):
                    close_price = float(self.df.iloc[-1]['close'])
                    self.current_step = len(self.df) - 1
                else:
                    close_price = float(self.df.iloc[self.current_step]['close'])
                
                if np.isnan(close_price) or close_price <= 0:
                    raise ValueError(f"Invalid close price: {close_price}")
                    
            except Exception as e:
                print(f"Error getting close price at step {self.current_step}: {e}")
                print(f"DataFrame length: {len(self.df)}")
                print(f"Columns: {self.df.columns.tolist() if hasattr(self.df, 'columns') else 'No columns'}")
                print(f"Sample data: {self.df.head() if hasattr(self, 'df') else 'No DataFrame'}")
                
                # Try to get a valid close price
                if hasattr(self, 'df') and not self.df.empty:
                    close_price = float(self.df['close'].iloc[-1] if 'close' in self.df.columns else 1.0)
                else:
                    close_price = 1.0  # Fallback value
            
            # Get valid actions and action mask
            valid_actions = self.get_valid_actions()
            action_mask = self.get_action_mask()
            
            if self.debug:
                print(f"Valid actions: {valid_actions}")
                print(f"Action mask: {action_mask}")
                print(f"Current position: {'Open' if self.position_open else 'Closed'}")
                print(f"Balance: {self.balance:.2f}, Shares: {self.shares}")
                print(f"Close price: {close_price:.2f}")
            
            if self.debug:
                print(f"Valid actions: {valid_actions}")
                print(f"Action mask: {action_mask}")
                print(f"Current position: {'Open' if self.position_open else 'Closed'}")
                print(f"Balance: {self.balance:.2f}, Shares: {self.shares}")

            if action not in [0, 1, 2]:
                if self.debug:
                    print(f"Invalid action: {action}, defaulting to Hold (0)")
                action = 0
                
            if action == 1 and 1 in valid_actions:  # Buy
                # Calculate position size as 10% of current balance
                position_size = 0.1  # 10% of current balance
                max_affordable = (self.balance * position_size) / close_price
                shares_to_buy = int(max_affordable)
                
                if shares_to_buy > 0 and self.balance >= shares_to_buy * close_price:
                    cost = shares_to_buy * close_price
                    self.shares += shares_to_buy
                    self.balance -= cost
                    self.position_open = True
                    self.buy_price = close_price
                    if self.debug: 
                        print(f"Bought {shares_to_buy} shares at {close_price:.2f}")
                        print(f"New balance: {self.balance:.2f}, New shares: {self.shares}")
                    # Small reward for entering a position
                    reward += 0.01
                elif self.debug:
                    print(f"Not enough balance to buy 1000 shares. Needed: {cost:.2f}, Available: {self.balance:.2f}")

            elif action == 2 and 2 in valid_actions:  # Sell
                if self.shares > 0:
                    # Sell all shares at once
                    shares_to_sell = self.shares
                    sale_value = shares_to_sell * close_price
                    cost_basis = shares_to_sell * self.buy_price
                    profit = sale_value - cost_basis
                    
                    # Calculate return on investment (ROI)
                    roi = profit / cost_basis if cost_basis > 0 else 0
                    
                    # Scale reward by ROI
                    reward = roi * 10  # Scale ROI to make it more significant
                    
                    # Update state
                    self.balance += sale_value
                    self.shares = 0
                    self.position_open = False
                    self.round_trip_trades += 1
                    
                    # Additional reward for profitable trades
                    if profit > 0:
                        reward += 0.1
                        
                    if self.debug: 
                        print(f"Sold {shares_to_sell} shares at {close_price:.2f}")
                        print(f"Profit: {profit:.2f} (ROI: {roi*100:.2f}%), New balance: {self.balance:.2f}")
                        print(f"Round trip completed. Total: {self.round_trip_trades}")
            else:  # Hold
                # Small negative reward for holding to encourage active trading
                reward = -0.001
                
                # If in a position, give small reward/penalty based on price movement
                if self.position_open:
                    price_change = (close_price - self.buy_price) / self.buy_price
                    reward += price_change * 0.1  # Scale down the price change effect

            # Update net worth with safety checks
            try:
                self.net_worth = float(self.balance) + (float(self.shares) * float(close_price))
                # Track portfolio value for normalization
                self.portfolio_value_history.append(self.net_worth)
                
                # Calculate Sharpe ratio (simplified)
                if len(self.portfolio_value_history) > 1:
                    returns = np.diff(np.log(self.portfolio_value_history))
                    sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-9)
                    # Add small reward for good risk-adjusted returns
                    reward += sharpe_ratio * 0.01
                    
            except Exception as e:
                if self.debug:
                    print(f"Error calculating net worth: {e}")
                    print(f"Balance: {self.balance}, Shares: {self.shares}, Close Price: {close_price}")
                self.net_worth = float(self.balance)  # Fallback to just balance
            
            # Small penalty for each step to encourage efficiency
            step_penalty = 0.001
            reward -= step_penalty
            
            # Immediate reward for taking valid actions
            if action in [1, 2]:  # Buy or Sell
                reward += 0.01  # Small positive reward for taking trading actions
            
            # Calculate value change percentage
            value_change_pct = (self.net_worth - self.initial_balance) / self.initial_balance
            
            # Reward based on portfolio value change (scaled to be in a reasonable range)
            reward += value_change_pct * 0.1  # Scale down the value change reward
            
            # Reward for profit taking (only on sell actions)
            if action == 2:  # Sell action
                if profit > 0:
                    # Reward for profitable trade (scaled by initial balance)
                    profit_reward = (profit / self.initial_balance) * 0.5
                    reward += profit_reward
                    if self.debug:
                        print(f"Profit reward: {profit_reward:.4f}")
                else:
                    # Small penalty for unprofitable trade
                    reward -= 0.01
            
            # Update max net worth and total reward
            self.max_net_worth = max(self.net_worth, self.max_net_worth)
            self.total_reward += reward
            
            if self.debug:
                print(f"Close price: {close_price:.2f}")
                print(f"Net worth: {self.net_worth:.2f} (Change: {value_change_pct:+.2%})")
                print(f"Step reward: {reward:.4f}")
                print(f"Total reward: {self.total_reward:.4f}")

            # Move to next step
            self.current_step += 1
            
            # Check termination conditions
            done = bool(
                self.round_trip_trades >= 10 or
                self.current_step >= len(self.df) - 2 or
                self.net_worth <= self.initial_balance * 0.7
            )
            
            if done and self.debug:
                print("\n--- Episode Done ---")
                if self.round_trip_trades >= 10:
                    print("Reason: Reached maximum round trip trades (10)")
                elif self.current_step >= len(self.df) - 2:
                    print("Reason: Reached end of data")
                elif self.net_worth <= self.initial_balance * 0.7:
                    print(f"Reason: Net worth dropped below 70% of initial balance ({self.net_worth:.2f} <= {self.initial_balance * 0.7:.2f})")
            
            # Prepare next observation
            obs = self.get_obs()
        
            info = {
                'valid_actions': valid_actions,
                'action_mask': action_mask.tolist(),
                'net_worth': float(self.net_worth),
                'shares': int(self.shares),
                'position_open': bool(self.position_open),
                'step': int(self.current_step),
                'close_price': float(close_price),
                'profit': float(profit),
                'done': done,
                'truncated': bool(truncated)
            }

            if self.debug:
                print(f"New observation: {obs}")
                print(f"Info: {info}")
                print("=== Step End ===\n")
        
            return obs, float(reward), done, truncated, info
            
        except Exception as e:
            # If an error occurs, return a default observation and info
            import traceback
            error_msg = f"Error in step: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            
            # Try to get a valid observation if possible
            try:
                obs = self.get_obs()
            except:
                obs = np.zeros(self.observation_space.shape, dtype=np.float32)
                
            info = {
                'valid_actions': [0],  # Only hold is valid in error case
                'action_mask': [1.0, 0.0, 0.0],
                'net_worth': float(self.balance if hasattr(self, 'balance') else self.initial_balance),
                'shares': int(self.shares) if hasattr(self, 'shares') else 0,
                'position_open': bool(self.position_open) if hasattr(self, 'position_open') else False,
                'step': int(self.current_step) if hasattr(self, 'current_step') else 0,
                'close_price': float(close_price) if 'close_price' in locals() else 0.0,
                'profit': float(profit) if 'profit' in locals() else 0.0,
                'done': True,
                'truncated': True,
                'error': error_msg
            }
            return obs, 0.0, True, True, info
            
    def render(self, mode='human'):
        """Render the environment to the screen."""
        try:
            if mode == 'human':
                print(f"Step: {self.current_step}")
                print(f"Balance: ${self.balance:.2f}")
                print(f"Shares: {self.shares}")
                print(f"Net Worth: ${self.net_worth:.2f}")
                print(f"Position Open: {self.position_open}")
                if self.position_open:
                    print(f"Buy Price: ${self.buy_price:.2f}")
                print("-" * 40)
                return None
            elif mode == 'rgb_array':
                # Return a numpy array representing the current state (placeholder)
                return np.zeros((100, 100, 3), dtype=np.uint8)
            return None
        except Exception as e:
            print(f"Error in render: {str(e)}")
            return None