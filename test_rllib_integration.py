import os
import ray
import numpy as np
import pandas as pd
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from trading_env_enhanced import TradingEnv
from common_imports import load_data

def env_creator(env_config):
    """Create and return a TradingEnv instance with the given configuration."""
    df = env_config.get("df")
    debug = env_config.get("debug", False)
    stoploss = env_config.get("stoploss", False)
    
    env = TradingEnv(
        df=df,
        debug=debug,
        stoploss=stoploss,
        stoploss_min=0.01,  # 1% minimum stoploss
        stoploss_max=0.1,   # 10% maximum stoploss
        max_steps=1000,     # Limit steps per episode for training
        test_mode=False
    )
    return env

def log_episode_stats(episode, result, action_counts):
    """Log detailed statistics for each training episode."""
    print(f"\n=== Episode {episode + 1} ===")
    print(f"Reward: {result['episode_reward_mean']:.4f}")
    print(f"Length: {result['episode_len_mean']:.1f} steps")
    
    # Log action distribution
    total_actions = sum(action_counts.values())
    if total_actions > 0:
        print("Action distribution:")
        for action, count in sorted(action_counts.items()):
            print(f"  {action}: {count} ({count/total_actions*100:.1f}%)")
    
    # Log other relevant metrics
    if 'info' in result and 'learner' in result['info']:
        learner_info = result['info']['learner']
        if 'default_policy' in learner_info:
            policy_info = learner_info['default_policy']
            if 'learner_stats' in policy_info:
                stats = policy_info['learner_stats']
                print("Learning stats:")
                for k, v in stats.items():
                    if isinstance(v, (int, float)):
                        print(f"  {k}: {v:.4f}")
                    elif isinstance(v, dict) and 'mean' in v:
                        print(f"  {k}: {v['mean']:.4f} ± {v['std']:.4f}")

def test_rllib_integration():
    """Test the integration of TradingEnv with RLlib."""
    # Set up logging
    debug_log = open("training_debug.log", "w")
    
    def log(msg):
        print(msg)
        debug_log.write(f"{msg}\n")
        debug_log.flush()
    
    log("Starting RLlib integration test...")
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    # Register the environment
    register_env("TradingEnv-v0", env_creator)
    
    # Load and preprocess data
    log("\n=== Data Loading ===")
    try:
        # Load data using the load_data function from common_imports
        df, _ = load_data()
        log(f"Successfully loaded data with {len(df)} rows")
        log(f"Data columns: {df.columns.tolist()}")
        log(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        log(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    except Exception as e:
        log(f"Error loading data: {e}")
        log("Creating sample data for testing...")
        # Create a simple dataframe for testing if loading fails
        np.random.seed(42)
        n_steps = 1000
        df = pd.DataFrame({
            'datetime': pd.date_range(start='2023-01-01', periods=n_steps, freq='1min'),
            'open': 100 + np.cumsum(np.random.randn(n_steps) * 0.5),
            'high': 0.0,
            'low': 0.0,
            'close': 0.0,
            'volume': np.ones(n_steps) * 1000
        })
        # Calculate high/low/close based on open with some random variation
        df['high'] = df['open'] + np.abs(np.random.randn(n_steps) * 0.3)
        df['low'] = df['open'] - np.abs(np.random.randn(n_steps) * 0.3)
        df['close'] = np.random.uniform(df['low'], df['high'])
        log(f"Created sample data with {len(df)} rows")
    
    # Configuration for the environment
    env_config = {
        "df": df,
        "debug": True,
        "stoploss": True
    }
    
    # Create a sample environment to get observation and action space
    sample_env = env_creator(env_config)
    obs_space = sample_env.observation_space
    act_space = sample_env.action_space
    
    print("\n=== Environment Details ===")
    print(f"Observation space: {obs_space}")
    print(f"Action space: {act_space}")
    print(f"Stoploss enabled: {sample_env.stoploss}")
    
    # Configure PPO
    print("\nConfiguring PPO algorithm...")
    config = (
        PPOConfig()
        .environment(
            env="TradingEnv-v0",
            env_config=env_config,
            observation_space=obs_space,
            action_space=act_space,
            disable_env_checking=True,  # Disable environment checking to avoid issues with custom spaces
        )
        .framework("torch")
        .training(
            gamma=0.99,
            lr=0.0003,
            train_batch_size=4000,
            model={
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "tanh",
                "custom_model_config": {
                    "action_space": act_space,  # Pass action space to model
                },
            },
        )
        .rollouts(
            num_rollout_workers=1,  # Reduce workers for testing
            num_envs_per_worker=1,
            batch_mode="truncate_episodes",
            create_env_on_local_worker=True,  # Create env on local worker to avoid serialization issues
        )
        .evaluation(
            evaluation_interval=1,
            evaluation_duration=1,
            evaluation_config={
                "explore": False,
                "disable_env_checking": True,
            },
        )
    )
    
    # Create the PPO algorithm
    algo = config.build()
    
    # Run training with detailed logging
    log("\n=== Starting Training ===")
    num_episodes = 10  # Increased for better observation
    
    for i in range(num_episodes):
        # Reset action counts for this episode
        action_counts = {0: 0, 1: 0, 2: 0}  # HOLD, BUY, SELL
        
        # Train for one episode
        result = algo.train()
        
        # Extract action distribution from the result
        if 'hist_stats' in result and 'action_0' in result['hist_stats']:
            actions = result['hist_stats']['action_0']
            for action in actions:
                if isinstance(action, (list, np.ndarray)):
                    action = action[0]  # Take first element if action is an array
                action = int(action)
                if action in action_counts:
                    action_counts[action] += 1
        
        # Log detailed statistics
        log_episode_stats(i, result, action_counts)
        
        # Save checkpoint every few episodes
        if (i + 1) % 5 == 0:
            checkpoint_dir = algo.save()
            log(f"\nSaved checkpoint to {checkpoint_dir}")
        
        # Early stopping if we're not making progress
        if i > 2 and result.get('episode_reward_mean', 0) < -100:
            log("\nWarning: Poor performance detected. Stopping early.")
            break
    
    # Test the trained policy with detailed logging
    log("\n=== Testing Trained Policy ===")
    env = env_creator(env_config)
    obs, _ = env.reset()
    done = False
    total_reward = 0
    step_count = 0
    max_steps = 200  # Increased steps for better observation
    
    # Track metrics during testing
    test_action_counts = {0: 0, 1: 0, 2: 0}
    portfolio_values = []
    
    log("Starting policy test...")
    
    while not done and step_count < max_steps:
        # Get action from policy
        action = algo.compute_single_action(obs, explore=False)
        
        # Log action details
        action_type = action[0] if isinstance(action, (tuple, list)) else action
        action_type = int(action_type)  # Ensure it's an integer
        test_action_counts[action_type] += 1
        
        # Take step in environment
        next_obs, reward, done, _, info = env.step(action)
        
        # Log step details
        if step_count % 10 == 0 or abs(reward) > 0.1:  # Log more frequently for significant rewards
            log(f"\nStep {step_count}:")
            log(f"  Action: {action}")
            log(f"  Reward: {reward:.4f}")
            log(f"  Portfolio: ${env.net_worth:.2f}")
            if hasattr(env, 'position_open') and env.position_open:
                log(f"  Position: {env.shares} shares @ ${env.buy_price:.2f}")
                if hasattr(env, 'stoploss_price'):
                    log(f"  Stoploss: ${env.stoploss_price:.2f}")
        
        total_reward += reward
        portfolio_values.append(env.net_worth)
        obs = next_obs
        step_count += 1
    
    # Log test summary
    log("\n=== Test Summary ===")
    log(f"Total reward: {total_reward:.4f}")
    log(f"Final portfolio value: ${env.net_worth:.2f}")
    log("Action distribution during test:")
    for action, count in sorted(test_action_counts.items()):
        log(f"  {action}: {count} ({count/step_count*100:.1f}%)")
    
    log(f"\nTest complete. Total reward: {total_reward:.4f} in {step_count} steps.")
    
    # Plot portfolio value over time
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_values)
        plt.title("Portfolio Value During Testing")
        plt.xlabel("Step")
        plt.ylabel("Portfolio Value ($)")
        plt.grid(True)
        plot_path = "portfolio_value.png"
        plt.savefig(plot_path)
        log(f"\nSaved portfolio value plot to {plot_path}")
    except Exception as e:
        log(f"\nCould not generate plot: {e}")
    
    # Final log message before closing
    final_msg = "\nRLlib integration test completed. Check training_debug.log for details."
    print(final_msg)
    debug_log.write(final_msg + "\n")
    
    # Clean up
    debug_log.close()
    ray.shutdown()

if __name__ == "__main__":
    test_rllib_integration()
