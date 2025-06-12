from common_imports import *
from trading_env import TradingEnv
import pickle
import numpy as np
import sys
import datetime
import io
import os
import re
import traceback
from collections import defaultdict

def get_debug_log_filename(model_name):
    """Generate a debug log filename with model name."""
    return f'test_script_debug_{model_name}.txt'

def debug_log(message, model_name, input_dir_name):
    """Log debug messages to a file.
    
    Args:
        message: The message to log
        model_name: Name of the model being tested
        input_dir_name: Name of the input directory (will be used as subdirectory in debug_logs)
    """
    if not hasattr(debug_log, 'log_files'):
        debug_log.log_files = {}
    
    # Ensure we have valid directory and model names
    input_dir_name = input_dir_name or 'default'
    model_name = model_name or 'unknown_model'
    
    # Create a cache key for this combination
    cache_key = f"{input_dir_name}/{model_name}"
    
    if cache_key not in debug_log.log_files:
        # Create debug directory: debug_logs/input_dir_name/
        debug_dir = os.path.join('debug_logs', input_dir_name)
        os.makedirs(debug_dir, exist_ok=True)
        
        # Create log file with model_name as the filename
        filename = os.path.join(debug_dir, f"test_script_debug_{model_name}.txt")
        debug_log.log_files[cache_key] = filename
        
        # Write header with timestamp
        with open(filename, 'w') as f:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"=== Debug Log - {model_name} - {timestamp} ===\n")
            f.write(f"Input Directory: {input_dir_name}\n")
            f.write(f"Model: {model_name}\n")
            f.write("=" * 40 + "\n\n")
    else:
        filename = debug_log.log_files[cache_key]
    
    # Only add timestamp for error messages or important events
    with open(filename, 'a') as f:
        if any(keyword in message.lower() for keyword in ['error', 'warning', 'exception', 'traceback']):
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{timestamp}] {message}\n")
        else:
            f.write(f"{message}\n")

def debug_print(message, input_dir_name, model_name):
    """Print debug messages to both console and file.
    
    Args:
        message: The message to print and log
        input_dir_name: Name of the input directory (for log file organization)
        model_name: Name of the model being tested (for log file naming)
    """
    print(message)
    
    # Don't log empty messages or separators to file
    if message.strip() and not message.startswith('-' * 10) and not message.startswith('=' * 10):
        debug_log(message, model_name, input_dir_name)

def load_model_and_stats(model_path, norm_path=None, model_name=None, input_dir_name=None):
    """Load the trained model and normalization stats."""
    try:
        if model_name is None:
            model_name = os.path.basename(model_path).replace('.zip', '')
            
        debug_print(f"\n=== Loading Model ===", input_dir_name, model_name)
        debug_print(f"Model path: {model_path}", input_dir_name, model_name)
        debug_print(f"Normalization path: {norm_path}", input_dir_name, model_name)
        
        if not os.path.exists(model_path):
            debug_print(f"Error: Model file not found at {model_path}", input_dir_name, model_name)
            sys.exit(1)
        
        # Extract number of bars from model path
        model_dir = os.path.dirname(model_path)
        match = re.search(r'(\d+)_bars', model_dir)
        if match:
            max_steps = int(match.group(1))
            debug_print(f"Using {max_steps} bars from model directory name", input_dir_name, model_name)
        else:
            debug_print("Error: Could not determine number of bars from model path", input_dir_name, model_name)
            sys.exit(1)
            
        # Load data using the common_imports function
        debug_print("Loading data using common_imports.load_data()...", input_dir_name, model_name)
        global df
        df, _ = load_data(max_steps=max_steps)
        debug_print(f"Loaded {len(df)} rows of data", input_dir_name, model_name)
        
        debug_print("Loading model...", input_dir_name, model_name)
        model = MaskablePPO.load(model_path)
        debug_print("Model loaded successfully", input_dir_name, model_name)
        obs_space_shape = model.observation_space.shape
        debug_print(f"Model observation space shape: {obs_space_shape}", input_dir_name, model_name)
        debug_print(f"Model unnormalized observation space: {model.observation_space}", input_dir_name, model_name)
        debug_print(f"Model action space: {model.action_space}", input_dir_name, model_name)
        
        debug_print("Creating environment...", input_dir_name, model_name)
        num_cpu = 1  # Using 1 process for testing
        
        # Create the environment function with model name and debug settings
        def make_env():
            return TradingEnv(
                df=df, 
                max_steps=max_steps, 
                debug=True, 
                model_name=model_name, 
                input_dir_name=input_dir_name,
                test_mode=True  # Enable debug logging in test mode
            )
            
        # Check for normalization stats first
        if norm_path is not None and os.path.exists(norm_path):
            try:
                debug_print(f"Loading normalization stats from {norm_path}...", input_dir_name, model_name)
                # Create environment with normalization
                vec_env = SubprocVecEnv([make_env for _ in range(num_cpu)])
                vec_env = VecMonitor(vec_env)
                env = VecNormalize.load(norm_path, vec_env)
                debug_print("Using normalization stats from file", input_dir_name, model_name)
                # Set training to False for testing
                env.training = False
                env.norm_reward = False
            except Exception as e:
                debug_print(f"Warning: Failed to load normalization stats: {str(e)}", input_dir_name, model_name)
                debug_print("Creating environment without normalization.", input_dir_name, model_name)
                env = VecNormalize(SubprocVecEnv([make_env for _ in range(num_cpu)]), 
                                 training=False, norm_obs=False, norm_reward=False)
                env = VecMonitor(env)
        else:
            debug_print("No normalization stats provided, using environment without normalization.", input_dir_name, model_name)
            env = VecNormalize(SubprocVecEnv([make_env for _ in range(num_cpu)]), 
                             training=False, norm_obs=False, norm_reward=False)
            env = VecMonitor(env)
            
        debug_print(f"Environment created with max_steps={max_steps}", input_dir_name, model_name)
        debug_print(f"Environment observation space: {env.observation_space}", input_dir_name, model_name)
        
        env.training = False
        env.norm_reward = False
        debug_print("Environment configuration complete", input_dir_name, model_name)
        return model, env
    except Exception as e:
        debug_print(f"Error in load_model_and_stats: {str(e)}", input_dir_name, model_name)
        debug_print("Full traceback:", input_dir_name, model_name)
        traceback.print_exc()
        sys.exit(1)

def format_reward_summary(actions, rewards, max_steps, close_prices=None, time_features=None):
    """Format a summary of rewards per step with action information and close prices.
    
    Args:
        actions: List of actions taken
        rewards: List of rewards received
        max_steps: Maximum number of steps to display
        close_prices: Optional list of close prices for each step
        time_features: Optional list of dicts containing time features for each step
    """
    if not actions or not rewards or len(actions) != len(rewards):
        return "Incomplete data for reward summary"
    
    # Initialize summary
    summary = ["\n=== REWARD SUMMARY ===\n"]
    
    # Adjust column widths based on whether we have close prices
    if close_prices is not None and len(close_prices) == len(actions):
        summary.append(f"{'Step':<6} | {'Action':<7} | {'Close':<10} | {'Reward':<10} | Description")
        summary.append("-" * 80)
    else:
        summary.append(f"{'Step':<6} | {'Action':<7} | {'Reward':<10} | Description")
        summary.append("-" * 60)
    
    # Track position and trade information
    position_open = False
    position_shares = 0
    position_avg_price = 0.0
    trade_count = 0
    total_profit = 0.0
    
    for step in range(min(max_steps, len(actions))):
        action = actions[step]
        reward = rewards[step] if step < len(rewards) else 0
        # Convert action to Python int if it's a numpy array
        action = int(action) if hasattr(action, 'item') else int(action) if isinstance(action, (int, float)) else action
        action_str = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}.get(action, 'UNKNOWN')
        
        # Format description - only show realized P&L (no unrealized P&L for HOLD actions)
        desc = ""
        if action == 1:  # BUY
            if close_prices is not None and len(close_prices) > step:
                position_avg_price = close_prices[step]
                position_shares = 1000  # Assuming fixed position size for now
                desc = f"BUY {position_shares} @ {position_avg_price:.4f}"
            else:
                desc = "+0.01 reward for BUY action"
            position_open = True
            
        elif action == 2:  # SELL
            if position_open and close_prices is not None and len(close_prices) > step:
                sell_price = close_prices[step]
                raw_profit = (sell_price - position_avg_price) * position_shares
                
                # Apply time-based modifiers if time_features is provided
                time_mod = ""
                if time_features is not None and step < len(time_features):
                    time_info = time_features[step]
                    if time_info.get('is_pre_market', False):
                        raw_profit *= 1.1  # 10% bonus during pre-market
                        time_mod = " (Pre-Market +10%)"
                    elif time_info.get('is_after_hours', False):
                        raw_profit *= 0.9  # 10% penalty during after-hours
                        time_mod = " (After-Hours -10%)"
                
                profit_pct = ((sell_price / position_avg_price) - 1) * 100 if position_avg_price > 0 else 0
                desc = f"SELL {position_shares} | Bought @ {position_avg_price:.4f} | Sold @ {sell_price:.4f}{time_mod} | Profit: ${raw_profit:.2f} ({profit_pct:+.2f}%)"
                trade_count += 1
                total_profit += raw_profit  # Add to total profit
            else:
                profit = reward - 0.01  # Fallback to reward-based calculation
                desc = f"Profit from trade: {profit:.4f}"
            position_open = False
            
        elif position_open:
            # For HOLD actions, just show that we're holding without P&L
            desc = f"Holding {position_shares} shares"
        
        # Add to summary with or without close price
        if close_prices is not None and len(close_prices) > step:
            close_price = close_prices[step]
            summary.append(f"{step:<6} | {action_str:<7} | {close_price:<10.4f} | {reward:<10.4f} | {desc}")
        else:
            summary.append(f"{step:<6} | {action_str:<7} | {reward:<10.4f} | {desc}")
    
    # Add trade summary at the end
    if trade_count > 0:
        summary.append("\n" + "=" * 80)
        summary.append(f"Total Trades: {trade_count}")
        # Total profit is already calculated in the loop with time-based modifiers
        summary.append(f"Total Profit: ${total_profit:.2f}")
    
    return "\n".join(summary)

def analyze_time_patterns(buy_signals, sell_signals, time_features):
    """Analyze trading patterns based on time features."""
    time_distribution = {
        'Morning (4:00-9:29)': 0,
        'Regular Hours (9:30-15:59)': 0,
        'Afternoon (16:00-19:59)': 0
    }
    pre_market_trades = 0
    after_hours_trades = 0
    buy_times = []
    sell_times = []

    if not time_features:
        return {
            'time_distribution': time_distribution,
            'pre_market_trades': pre_market_trades,
            'after_hours_trades': after_hours_trades,
            'avg_buy_time': 'N/A',
            'avg_sell_time': 'N/A',
            'buy_times': buy_times,
            'sell_times': sell_times
        }

    for signal in buy_signals + sell_signals:
        time = signal['time'].time()
        if time < datetime.time(9, 30):
            pre_market_trades += 1
            time_distribution['Morning (4:00-9:29)'] += 1
        elif time <= datetime.time(15, 59):
            time_distribution['Regular Hours (9:30-15:59)'] += 1
        else:
            after_hours_trades += 1
            time_distribution['Afternoon (16:00-19:59)'] += 1

    # Convert times to minutes for easier calculations
    buy_times_minutes = []
    sell_times_minutes = []
    
    for signal in buy_signals:
        time = signal['time'].time()
        minutes = time.hour * 60 + time.minute
        buy_times_minutes.append(minutes)
    
    for signal in sell_signals:
        time = signal['time'].time()
        minutes = time.hour * 60 + time.minute
        sell_times_minutes.append(minutes)

    # Calculate average times in minutes
    avg_buy_time = 'N/A'
    avg_sell_time = 'N/A'
    if buy_times_minutes:
        avg_buy_minutes = sum(buy_times_minutes) / len(buy_times_minutes)
        avg_buy_time = datetime.time(int(avg_buy_minutes // 60), int(avg_buy_minutes % 60))
    if sell_times_minutes:
        avg_sell_minutes = sum(sell_times_minutes) / len(sell_times_minutes)
        avg_sell_time = datetime.time(int(avg_sell_minutes // 60), int(avg_sell_minutes % 60))

    # Return the analysis results
    return {
        'time_distribution': time_distribution,
        'pre_market_trades': pre_market_trades,
        'after_hours_trades': after_hours_trades,
        'avg_buy_time': avg_buy_time,
        'avg_sell_time': avg_sell_time,
        'buy_times': buy_times,
        'sell_times': sell_times
    }

def test_trading_model(model_path, norm_path, render=True, debug=False, output_dir=None, input_dir_name=None):
    """Test the trained trading model with normalization and return signals.
    
    Args:
        model_path: Path to the model file
        norm_path: Path to the normalization stats file
        render: Whether to render the environment
        debug: Whether to print debug information
        output_dir: Directory to save output files (default: same as model directory)
        input_dir_name: Name of the input directory (for logging purposes)
    """
    # Extract model name for logging
    model_name = os.path.basename(model_path).replace('.zip', '')
    
    # Create a null device to suppress output
    class NullDevice():
        def write(self, s):
            pass
        def flush(self):
            pass
    
    # Set up output buffer for logging
    output_buffer = io.StringIO()
    
    # Save original stdout/stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Redirect stdout/stderr to buffer only (suppress terminal output)
    sys.stdout = output_buffer
    sys.stderr = output_buffer
    
    try:
        # Extract max_steps from model path
        import re
        match = re.search(r'(\d+)_bars', model_path)
        if match:
            max_steps = int(match.group(1))
            debug_print(f"Using {max_steps} steps from model path", input_dir_name, model_name)
        else:
            debug_print("Error: Could not determine number of steps from model path", input_dir_name, model_name)
            raise ValueError("Could not determine number of steps from model path")
            
        # Initialize variables
        buy_signals = []
        sell_signals = []
        buy_signal_indices = []
        sell_signal_indices = []
        episode_reward = 0
        actions = []
        rewards = []
        rewards_per_step = []  # Track rewards per step
        timesteps = []
        time_features = []
        
        debug_print(f"\n=== Starting Trading Simulation ===", input_dir_name, model_name)
        debug_print(f"Model path: {model_path}", input_dir_name, model_name)
        debug_print(f"Normalization path: {norm_path}", input_dir_name, model_name)
        debug_print(f"Max steps: {max_steps}", input_dir_name, model_name)
        debug_print(f"Render mode: {render}", input_dir_name, model_name)
        
        # Initialize environment and model
        model, env = load_model_and_stats(model_path, norm_path, model_name=model_name, input_dir_name=input_dir_name)
        obs_space_shape = model.observation_space.shape
        
        # Reset the environment and get the initial observation
        debug_print("\n=== Resetting Environment ===", input_dir_name, model_name)
        obs = env.reset()
        debug_print(f"Initial observation shape: {obs.shape if hasattr(obs, 'shape') else 'N/A'}", input_dir_name, model_name)
        debug_print(f"Initial observation: {obs}", input_dir_name, model_name)
        
        # Get environment attributes with support for both vectorized and non-vectorized envs
        def get_env_attr(env, attr, default=None):
            try:
                # First try to get the attribute directly
                if hasattr(env, attr):
                    return getattr(env, attr, default)
                    
                # Try to get attribute using get_attr (for vectorized environments)
                if hasattr(env, 'get_attr'):
                    try:
                        return env.get_attr(attr)[0]  # Get from first environment
                    except Exception:
                        pass
                        
                # Try to get attribute using env_method
                if hasattr(env, 'env_method'):
                    try:
                        return env.env_method('get_attr', attr)[0]
                    except Exception:
                        pass
                        
                # Try to get attribute from the underlying environment
                if hasattr(env, 'env'):
                    return get_env_attr(env.env, attr, default)
                    
                # Try to get attribute from the vectorized environment
                if hasattr(env, 'venv'):
                    return get_env_attr(env.venv, attr, default)
                    
                # If we get here, we couldn't find the attribute
                return default
                
            except Exception as e:
                debug_print(f"Error getting attribute {attr}: {str(e)}", input_dir_name, model_name)
                return default
                
        # Get the length of the dataframe from the environment
        df = get_env_attr(env, 'df')
        if df is None or len(df) == 0:
            debug_print("Error: Empty dataframe", input_dir_name, model_name)
            return None
            
        # Initialize step counter
        step = 0
        done = False
        
        # Main test loop
        while not done and step < max_steps and get_env_attr(env, 'current_step', 0) < len(df) - 1:
            current_step = get_env_attr(env, 'current_step')
            debug_print(f"\n=== Loop Step: {step}/{max_steps}, Env Step: {current_step} ===", input_dir_name, model_name)
            
            # Get current state
            current_step = get_env_attr(env, 'current_step')
            position_open = get_env_attr(env, 'position_open')
            shares = get_env_attr(env, 'shares')
            balance = get_env_attr(env, 'balance')
            
            # Get current time from dataframe
            current_time = df.iloc[current_step] if current_step < len(df) else df.iloc[-1]
            
            debug_print(f"Current time: {current_time['datetime']}, Price: {current_time['close']}", input_dir_name, model_name)
            debug_print(f"Position: {'Open' if position_open else 'Closed'}, Shares: {shares}, Balance: {balance}", input_dir_name, model_name)
            
            # Get action masks and valid actions
            if hasattr(env, 'get_attr'):
                action_masks = env.get_attr('action_masks')[0]()
            else:
                action_masks = env.action_masks()
                
            valid_actions = np.where(action_masks)[0]
            debug_print(f"\nAction masks: {action_masks}", input_dir_name, model_name)
            debug_print(f"Valid actions: {valid_actions}", input_dir_name, model_name)
            
            # Handle both tuple and array observations
            if isinstance(obs, tuple):
                # If observation is a tuple, take the first element (assuming it's the actual observation)
                obs = obs[0] if len(obs) > 0 else np.array([])
            
            # Convert to numpy array if it isn't already
            obs = np.array(obs)
            
            # Ensure proper shape
            if len(obs.shape) == 0:  # scalar
                obs = np.array([[obs]])
            elif len(obs.shape) == 1:  # 1D array
                obs = obs.reshape(1, -1)  # Make it 2D with shape (1, n_features)
            elif len(obs.shape) > 2:  # More than 2D, try to squeeze
                obs = np.squeeze(obs)
            
            # Get denormalized observation
            denormalized_obs = obs.copy()
            # Skip denormalization to avoid errors
            denormalized_obs = obs
            debug_print(f"\nObservation shape: {obs.shape}", input_dir_name, model_name)
            debug_print(f"Observation: {obs}", input_dir_name, model_name)
            
            # Get action from the model
            # Get environment attributes
            current_step = get_env_attr(env, 'current_step', 0)
            position_open = get_env_attr(env, 'position_open', False)
            shares = get_env_attr(env, 'shares', 0)
            balance = get_env_attr(env, 'balance', 0.0)
            net_worth = get_env_attr(env, 'net_worth', 0.0)
            df = get_env_attr(env, 'df')
            
            # Get current price from dataframe
            current_price = df.iloc[current_step]['close'] if df is not None and current_step < len(df) else 0.0
            
            # Get valid actions from the vectorized environment
            try:
                if hasattr(env, 'env_method'):
                    # For vectorized environments, call the method on the first environment
                    valid_actions = env.env_method('get_valid_actions', indices=0)[0]
                elif hasattr(env, 'get_attr'):
                    # For environments with get_attr
                    valid_actions = env.get_attr('get_valid_actions')[0]()
                else:
                    # For non-vectorized environments
                    valid_actions = env.get_valid_actions()
                debug_print(f"Valid actions: {valid_actions} (0=HOLD, 1=BUY, 2=SELL)", input_dir_name, model_name)
            except Exception as e:
                debug_print(f"Error getting valid actions: {str(e)}", input_dir_name, model_name)
                valid_actions = [0, 1, 2]  # Default to all actions if we can't get valid actions
            
            # Convert observation to PyTorch tensor if it's a numpy array
            import torch as th
            if isinstance(obs, np.ndarray):
                obs_tensor = th.as_tensor(obs, device=model.device)
            else:
                obs_tensor = obs
                
            # Get action probabilities using the model's policy
            with th.no_grad():
                action_probs = model.policy.get_distribution(obs_tensor).distribution.probs.detach().cpu().numpy()[0]
                
            # Get action from model
            action, _states = model.predict(obs, action_masks=action_masks)
            
            # Store the model's chosen action before executing
            model_action = action[0] if isinstance(action, np.ndarray) else action
            
            # Check if action is valid
            if action not in valid_actions:
                debug_print(f"WARNING: Model chose invalid action {action} when valid actions are {valid_actions}", input_dir_name, model_name)
                # Choose first valid action as fallback
                if valid_actions:
                    action = valid_actions[0]
                    debug_print(f"Falling back to action: {action} ({'HOLD' if action == 0 else 'BUY' if action == 1 else 'SELL'})", input_dir_name, model_name)
                else:
                    action = 0  # Default to HOLD if no valid actions
                    debug_print("No valid actions available, defaulting to HOLD", input_dir_name, model_name)
            
            debug_print("===================\n", input_dir_name, model_name)
            
            # Store the model's action before execution
            model_action = action[0] if isinstance(action, np.ndarray) else action
            
            # Execute the action in the environment
            df = get_env_attr(env, 'df')
            current_step = get_env_attr(env, 'current_step', 0)
            position_open = get_env_attr(env, 'position_open', False)
            shares = get_env_attr(env, 'shares', 0)
            balance = get_env_attr(env, 'balance', 0.0)
            net_worth = get_env_attr(env, 'net_worth', 0.0)
            
            # Log action details
            action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
            debug_print(f"\n[ACTION] Model chose: {action_names.get(model_action, 'UNKNOWN')} ({model_action})", input_dir_name, model_name)
            debug_print(f"[ACTION] Valid actions: {[action_names[a] for a in valid_actions]}", input_dir_name, model_name)
            
            # Get current price from dataframe
            current_price = df.iloc[current_step]['close'] if df is not None and current_step < len(df) else 0.0
            
            # Skip duplicate step info
            if debug:
                debug_print(f"\n--- Step {current_step} ---", input_dir_name, model_name)
                debug_print(f"Current price: {current_price:.2f}", input_dir_name, model_name)
                debug_print(f"Current position: {'OPEN' if position_open else 'CLOSED'}", input_dir_name, model_name)
                debug_print(f"Current shares: {shares}", input_dir_name, model_name)
                debug_print(f"Current balance: {balance:.2f}", input_dir_name, model_name)
                debug_print(f"Current net worth: {net_worth:.2f}", input_dir_name, model_name)
                
                if position_open:
                    buy_price = get_env_attr(env, 'buy_price', 0.0)
                    if buy_price > 0:
                        profit = (current_price - buy_price) / buy_price * 100
                        debug_print(f"Position profit: {profit:.2f}%", input_dir_name, model_name)
                
                debug_print(f"Executing action: {action} ({'HOLD' if action == 0 else 'BUY' if action == 1 else 'SELL'})", input_dir_name, model_name)
            
            # Handle both Gym v0.26.0+ (5 return values) and older versions (4 return values)
            # For vectorized environments, we need to pass the action as a list
            step_result = env.step([action])  # Wrap action in a list for vectorized env
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result
                
            # For vectorized environments, extract the first element from the results
            if isinstance(reward, np.ndarray) and len(reward) > 0:
                reward = float(reward[0])
                done = bool(done[0]) if isinstance(done, (list, np.ndarray)) else done
                info = info[0] if isinstance(info, (list, tuple)) and len(info) > 0 else info
            
            # Get updated environment state after step
            new_position_open = get_env_attr(env, 'position_open', False)
            new_shares = get_env_attr(env, 'shares', 0)
            new_balance = get_env_attr(env, 'balance', 0.0)
            new_net_worth = get_env_attr(env, 'net_worth', 0.0)
            
            debug_print(f"Action result - Reward: {reward:.4f}, Done: {done}", input_dir_name, model_name)
            debug_print(f"New position: {'OPEN' if new_position_open else 'CLOSED'}", input_dir_name, model_name)
            debug_print(f"New shares: {new_shares}", input_dir_name, model_name)
            debug_print(f"New balance: {new_balance:.2f}", input_dir_name, model_name)
            debug_print(f"New net worth: {new_net_worth:.2f}", input_dir_name, model_name)
            
            if 'invalid_action' in info and info['invalid_action']:
                debug_print("WARNING: Invalid action was taken in the environment", input_dir_name, model_name)
                debug_print("Previous valid actions: " + 
                          ", ".join([f'{a} ({"HOLD" if a == 0 else "BUY" if a == 1 else "SELL"})' 
                                  for a in info.get('previous_valid_actions', [])]), input_dir_name, model_name)
                debug_print("Next valid actions: " + 
                          ", ".join([f'{a} ({"HOLD" if a == 0 else "BUY" if a == 1 else "SELL"})' 
                                  for a in info.get('valid_actions', [])]), input_dir_name, model_name)
            
            debug_print("-" * 60, input_dir_name, model_name)
            debug_print(f"New observation shape: {obs.shape}", input_dir_name, model_name)
            debug_print(f"New observation: {obs}", input_dir_name, model_name)
            
            # Increment step counter
            step += 1
            debug_print(f"Incremented step counter to: {step}", input_dir_name, model_name)
            
            # Record signals with next time step's data
            if action == 1:
                buy_signals.append({'time': current_time['datetime'], 'price': current_time['close']})
                buy_signal_indices.append(current_step)
            elif action == 2:
                sell_signals.append({'time': current_time['datetime'], 'price': current_time['close']})
                sell_signal_indices.append(current_step)

            # Track time features
            time_features.append({
                'time': current_time['datetime'],
                'normalized_time': current_time['normalized_time'],
                'is_pre_market': current_time['is_pre_market'],
                'is_after_hours': current_time['is_after_hours'],
                'normalized_time_until_close': current_time['normalized_time_until_close']
            })

            # Track progress
            step_reward = reward[0] if isinstance(reward, np.ndarray) else reward
            episode_reward += step_reward
            actions.append(action)
            rewards_per_step.append(step_reward)
            timesteps.append(current_step)

            # Get the final state
            position_open = get_env_attr(env, 'position_open')
            shares = get_env_attr(env, 'shares')
            balance = get_env_attr(env, 'balance')
            debug_print(f"Post-action: Position: {'Open' if position_open else 'Closed'}, Shares: {shares}, Balance: {balance}", input_dir_name, model_name)

        
        # Analyze results
        time_analysis = analyze_time_patterns(buy_signals, sell_signals, time_features)
        
        # Save signals
        model_base_name = os.path.basename(model_path).split('.')[0]
        
        # Create output directory if it doesn't exist
        if output_dir is None:
            output_dir = os.path.dirname(model_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create results dictionary
        results = {
            'model': [model_base_name],
            'max_steps': [max_steps],  # Use max_steps from model path
            'reward': [episode_reward],
            'buy_signals': [buy_signals],
            'sell_signals': [sell_signals],
            'buy_signal_indices': [buy_signal_indices],
            'sell_signal_indices': [sell_signal_indices],
            'actions': [actions],
            'timesteps': [timesteps]
        }
        
        # Get close prices for each step
        close_prices = []
        for i in range(min(len(actions), len(df))):
            close_prices.append(df.iloc[i]['close'])
        
        # Add reward summary to debug output
        debug_print("\n" + "="*20 + " REWARD SUMMARY " + "="*20, input_dir_name, model_name)
        debug_print(f"Total reward: {episode_reward:.2f}", input_dir_name, model_name)
        debug_print(f"Number of BUY actions: {actions.count(1)}", input_dir_name, model_name)
        debug_print(f"Number of SELL actions: {actions.count(2)}", input_dir_name, model_name)
        debug_print("\nDetailed reward per step:", input_dir_name, model_name)
        
        # Prepare time features in the format expected by format_reward_summary
        time_features_formatted = []
        for tf in time_features:
            time_features_formatted.append({
                'is_pre_market': tf['is_pre_market'],
                'is_after_hours': tf['is_after_hours']
            })
            
        debug_print(format_reward_summary(actions, rewards_per_step, max_steps, close_prices, time_features_formatted), input_dir_name, model_name)
        debug_print("\n" + "="*80 + "\n", input_dir_name, model_name)
        
        # Save trade signals to CSV
        csv_filename = os.path.join(output_dir, f'trade_signals_{model_base_name}.csv')
        pd.DataFrame(results).to_csv(csv_filename, index=False)
        debug_print(f"Trade signals saved to '{csv_filename}'", input_dir_name, model_name)
        
        env.close()
        
        return {
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'buy_signal_indices': buy_signal_indices,
            'sell_signal_indices': sell_signal_indices,
            'reward': episode_reward,
            'time_features': time_features,
            'time_analysis': time_analysis,
            'actions': actions,
            'timesteps': timesteps
        }
        
    except Exception as e:
        error_msg = f"Error in test_trading_model: {str(e)}"
        debug_print(error_msg, model_name)
        debug_print(f"Full traceback:\n{traceback.format_exc()}", model_name)
        print(f"TEST FAILED: {error_msg}", file=sys.stderr)
        return None
        
def find_model_files(directory):
    """Find model and normalization files in the given directory."""
    print(f"\n=== find_model_files({directory}) ===")
    print(f"Directory exists: {os.path.exists(directory)}")
    print(f"Is directory: {os.path.isdir(directory)}")
    
    model_files = []
    norm_files = []
    
    try:
        # List all files in the directory
        files = os.listdir(directory)
        print(f"Found {len(files)} files in directory")
        
        # Find model and normalization files
        for file in files:
            print(f"  - {file} (is_file: {os.path.isfile(os.path.join(directory, file))})")
            if file.startswith('best_model') and file.endswith('.zip'):
                model_files.append(file)
                print(f"    Added as model file")
            elif file.startswith('final_model') and file.endswith('.zip'):
                model_files.append(file)
                print(f"    Added as model file")
            elif file.startswith('vec_normalize') and file.endswith('.pkl'):
                norm_files.append(file)
                print(f"    Added as normalization file")
        
        # Sort model files to ensure consistent order (best_model first, then final_model)
        model_files.sort()
        
        print(f"Found {len(model_files)} model files and {len(norm_files)} normalization files")
        return model_files, norm_files
        
    except Exception as e:
        print(f"Error in find_model_files: {str(e)}")
        import traceback
        traceback.print_exc()
        return [], []

def main():
    try:
        print("=== Starting test script ===")
        print(f"Current working directory: {os.getcwd()}")
        
        # Set up argument parser
        parser = argparse.ArgumentParser(description='Test a trained trading model')
        parser.add_argument('--input', type=str, required=True,
                          help='Path to the input model directory or file')
        parser.add_argument('--output_dir', type=str, default='test_output',
                          help='Directory to save test outputs')
        parser.add_argument('--debug', action='store_true',
                          help='Enable debug mode')
        args = parser.parse_args()
        
        print(f"Input path: {args.input}")
        print(f"Output directory: {args.output_dir}")
        print(f"Debug mode: {args.debug}")
        
        # Set up output directory with input directory name as subdirectory
        input_dir_name = os.path.basename(os.path.normpath(args.input))
        output_dir = os.path.join(args.output_dir, input_dir_name)
        os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
        
        print(f"Input directory name: {input_dir_name}")
        print(f"Full output directory: {output_dir}")
        print(f"Output directory exists: {os.path.exists(output_dir)}")
        
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"Input path does not exist: {args.input}")
            
        if not os.path.isdir(args.input):
            raise NotADirectoryError(f"Input path is not a directory: {args.input}")

        # Extract number of bars from the input directory name
        import re
        match = re.search(r'(\d+)_bars', input_dir_name)
        if not match:
            raise ValueError(f"Could not determine number of bars from directory name: {input_dir_name}")
            
        max_steps = int(match.group(1))
        print(f"Using max_steps: {max_steps}")
        
        # Find model and normalization files
        print("\n=== Finding model files ===")
        model_files, norm_files = find_model_files(args.input)
        print(f"Found {len(model_files)} model files and {len(norm_files)} normalization files")

        # Check if there are any model files
        if not model_files:
            raise FileNotFoundError("No model files found in the specified directory")
        
        # Test each model file found
        for model_file in model_files:
            print(f"\n=== Testing model: {model_file} ===")
            model_path = os.path.join(args.input, model_file)
            norm_path = os.path.join(args.input, norm_files[0]) if norm_files else None
            
            if not os.path.exists(model_path):
                print(f"Warning: Model file not found: {model_path}")
                continue
                
            if norm_path and not os.path.exists(norm_path):
                print(f"Warning: Normalization file not found: {norm_path}")
                norm_path = None
            
            try:
                results = test_trading_model(
                    model_path=model_path,
                    norm_path=norm_path,
                    render=False,
                    debug=args.debug,
                    output_dir=output_dir,
                    input_dir_name=input_dir_name
                )
                
                if results:
                    print(f"\n=== Results ===")
                    print(f"Total reward: {results.get('reward', 0):.2f}")
                    print(f"Buy signals: {len(results.get('buy_signals', []))}")
                    print(f"Sell signals: {len(results.get('sell_signals', []))}")
                    
                    if 'time_analysis' in results:
                        print("\nTime Analysis:")
                        time_dist = results['time_analysis'].get('time_distribution', {})
                        for time_range, count in time_dist.items():
                            print(f"{time_range}: {count}")
                    
                    # Save signals for plotting
                    model_base_name = os.path.splitext(model_file)[0]
                    signals_df = pd.DataFrame({
                        'model': [model_base_name],
                        'max_steps': [max_steps],
                        'reward': [results.get('reward', 0)],
                        'buy_signals': [str(results.get('buy_signal_indices', []))],
                        'sell_signals': [str(results.get('sell_signal_indices', []))],
                        'actions': [str(results.get('actions', []))],
                        'timesteps': [str(results.get('timesteps', []))]
                    })
                    
                    csv_filename = os.path.join(output_dir, f'trade_signals_{model_base_name}.csv')
                    signals_df.to_csv(csv_filename, index=False)
                    print(f"\nTrade signals saved to '{csv_filename}'")
                
            except Exception as e:
                print(f"\nError testing model {model_file}:")
                print(f"Type: {type(e).__name__}")
                print(f"Message: {str(e)}")
                print("\nTraceback:")
                import traceback
                traceback.print_exc()
                print("\nContinuing with next model...")
    
    except Exception as e:
        print(f"\nFatal error in main execution:")
        print(f"Type: {type(e).__name__}")
        print(f"Message: {str(e)}")
        print("\nTraceback:")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    import argparse
    sys.exit(main())