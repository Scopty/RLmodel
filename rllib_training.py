import os
import json
import numpy as np
import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.air.config import RunConfig, CheckpointConfig
from ray.tune import CLIReporter
from trading_env_enhanced import TradingEnv
from common_imports import load_data

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_env(env_config):
    """Create and return a TradingEnv instance with the given config."""
    env = TradingEnv(
        df=env_config['df'],
        max_steps=env_config.get('max_steps', 100),
        stoploss=True,
        stoploss_min=env_config.get('stoploss_min', 0.01),
        stoploss_max=env_config.get('stoploss_max', 0.10),
        debug=env_config.get('debug', True),  # Enable debug for training
        model_name=env_config.get('model_name', 'rllib_training')
    )
    return env

def evaluate(algorithm: Algorithm, env_config: dict, num_episodes: int = 5) -> dict:
    """Evaluate the current policy and return a dictionary of results."""
    env = create_env(env_config)
    episodes = []
    
    for _ in range(num_episodes):
        episode_reward = 0.0
        episode_length = 0
        obs, _ = env.reset()
        done = False
        
        while not done:
            action = algorithm.compute_single_action(obs, explore=False)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
        
        episodes.append({
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "final_balance": env.balance + (env.shares * env.df.iloc[env.current_step]['close'] if env.shares > 0 else 0)
        })
    
    env.close()
    return {
        "mean_reward": np.mean([e["episode_reward"] for e in episodes]),
        "mean_episode_length": np.mean([e["episode_length"] for e in episodes]),
        "mean_final_balance": np.mean([e["final_balance"] for e in episodes]),
        "num_episodes": len(episodes)
    }

def train():
    # Initialize Ray with better resource management
    ray.init(
        ignore_reinit_error=True,
        log_to_driver=True,
        _temp_dir=os.path.expanduser("~/ray_tmp"),
        num_cpus=4,  # Adjust based on your system
        num_gpus=0   # Set to 1 if you have a GPU
    )
    
    # Load and prepare data
    df, _ = load_data()  # Use processed DataFrame from MGOL.csv
    
    # Environment configuration with enhanced logging
    env_config = {
        'df': df,
        'max_steps': 100,  # Reduced from full episode length
        'stoploss_min': 0.01,
        'stoploss_max': 0.10,
        'debug': True,  # Enable debug logging
        'model_name': 'rllib_training_enhanced',
        'log_raw_obs': True,  # Log raw observation values
        'log_norm_obs': True,  # Log normalized observation values
        'log_rewards': True    # Log reward calculations
    }
    
    # Register the environment
    register_env('TradingEnv-v0', lambda config: create_env(config))
    env_name = 'TradingEnv-v0'
    
    # Configure the algorithm with optimized parameters for normalized environment
    config = (
        PPOConfig()
        .environment(
            env=env_name,
            env_config=env_config,
            disable_env_checking=True
        )
        .framework("torch")
        .rollouts(
            num_rollout_workers=4,  # Increased from 1 for better parallelization
            batch_mode="truncate_episodes",
            num_envs_per_worker=1,
        )
        .training(
            lr=0.0003,  # Slightly higher learning rate for faster learning
            train_batch_size=1000,  # Smaller batch size for faster iterations
            sgd_minibatch_size=250,  # Smaller minibatch size
            num_sgd_iter=3,  # Fewer gradient steps per batch for faster iterations
            gamma=0.99,  # Standard discount factor
            lambda_=0.95,  # GAE parameter
            clip_param=0.2,  # Default PPO clip parameter
            kl_coeff=0.5,  # KL divergence coefficient
            kl_target=0.01,  # Target KL divergence
            vf_loss_coeff=0.5,  # Value function loss coefficient
            entropy_coeff=0.01,  # Encourage exploration
            model={
                "fcnet_hiddens": [128, 128],  # Smaller network for faster training
                "fcnet_activation": "relu",
                "vf_share_layers": False,
                "use_lstm": False,  # Disable LSTM for simplicity
            },
        )
        .evaluation(
            evaluation_interval=5,
            evaluation_duration=10,
        )
    )
    
    # Configure the training run
    checkpoint_config = CheckpointConfig(
        num_to_keep=5,  # Keep the 5 most recent checkpoints
        checkpoint_frequency=5,  # Save a checkpoint every 5 iterations
        checkpoint_at_end=True,
    )
    
    # Configure the reporter for training progress
    reporter = CLIReporter(
        metric_columns={
            "training_iteration": "iter",
            "time_total_s": "time",
            "episodes_this_iter": "episodes",
            "episode_reward_mean": "reward_mean",
            "episode_reward_max": "reward_max",
            "episode_reward_min": "reward_min",
            "episode_len_mean": "length_mean",
        },
        max_report_frequency=60,  # Report at most every 60 seconds
    )
    
    # Configure the training run with reduced timesteps for faster iteration
    tuner = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=RunConfig(
            name="trading_ppo_10k",
            local_dir="./ray_results",
            stop={
                "timesteps_total": 10000,  # Further reduced to 10K timesteps
                "episode_reward_mean": 1000,  # Early stopping if reward is high
                "training_iteration": 20,  # Keep same max iterations
                "time_total_s": 900,  # Maximum time in seconds (15 minutes)
            },
            checkpoint_config=checkpoint_config,
            verbose=2,
            log_to_file=True,
            progress_reporter=reporter,
        ),
    )
    
    # Execute the training
    results = tuner.fit()
    
    # Get the best result
    best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
    print(f"Best trial final report: {best_result.metrics}")
    
    # Save the best model
    best_checkpoint = best_result.checkpoint
    best_checkpoint_dir = os.path.join("best_model")
    os.makedirs(best_checkpoint_dir, exist_ok=True)
    best_checkpoint.to_directory(best_checkpoint_dir)
    print(f"Best model saved to: {os.path.abspath(best_checkpoint_dir)}")
    
    return results

if __name__ == "__main__":
    # Run training
    results = train()
    
    # Example of loading the best model for inference
    from ray.rllib.algorithms.algorithm import Algorithm
    
    best_model = Algorithm.from_checkpoint("best_model")
    print("Successfully loaded the best model for inference.")
    
    # Example of running inference with the best model
    env_config = {
        'df': load_data()[0],  # Load fresh data
        'max_steps': 100,
        'stoploss_min': 0.01,
        'stoploss_max': 0.10,
        'debug': True,
        'model_name': 'inference'
    }
    env = create_env(env_config)
    
    print("\nRunning inference with the trained model...")
    obs, _ = env.reset()
    episode_reward = 0
    done = False
    
    while not done:
        action = best_model.compute_single_action(obs, explore=False)
        obs, reward, done, _, _ = env.step(action)
        episode_reward += reward
        
        if done:
            print(f"Episode finished with reward: {episode_reward:.2f}")
            break
    
    env.close()
    
    ray.shutdown()
