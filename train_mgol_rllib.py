import os
import sys
import logging
import ray
import gymnasium as gym
import numpy as np
import pandas as pd
from datetime import datetime
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.air.config import RunConfig, CheckpointConfig
from ray.tune import CLIReporter
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.sample_batch import SampleBatch
from common_imports import load_data
from trading_env import TradingEnv


class ActionProbCallback(DefaultCallbacks):
    """Custom callback to log action and stoploss probabilities during training."""
    
    def on_episode_start(self, worker, base_env, policies, episode, **kwargs):
        """Called at the start of each episode."""
        episode.user_data["action_probs"] = []
        episode.user_data["stoploss_values"] = []
        
    def on_episode_step(self, worker, base_env, episode, **kwargs):
        """Called on each environment step during an episode."""
        # Get the last observation and policy
        policy = worker.policy_for(episode.policy_id)
        last_obs = episode.last_observation_for()
        
        if last_obs is not None and hasattr(policy, 'model'):
            try:
                # Get model outputs for the last observation
                model_out, _ = policy.model({"obs": np.array([last_obs])}, [], None)
                
                # Get action distribution
                action_dist = policy.distr_class(*policy.model.get_action_model_outputs(model_out, []))
                
                # Get probabilities for each discrete action
                probs = action_dist.distribution.probs.detach().numpy()[0]
                episode.user_data["action_probs"].append(probs)
                
                # Get stoploss values (continuous action)
                if hasattr(policy.model, 'get_stoploss_output'):
                    stoploss_out = policy.model.get_stoploss_output(model_out, [])
                    # Assuming stoploss is a single value between [0, 1]
                    stoploss_value = stoploss_out.detach().numpy()[0][0]
                    episode.user_data["stoploss_values"].append(stoploss_value)
                
            except Exception as e:
                logger.warning(f"Could not get action probabilities: {str(e)}")
    
    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        """Called at the end of each episode."""
        # Log the mean action probabilities for this episode
        if episode.user_data["action_probs"]:
            try:
                mean_probs = np.mean(episode.user_data["action_probs"], axis=0)
                for i, prob in enumerate(mean_probs):
                    episode.custom_metrics[f"action_{i}_prob"] = float(prob)
            except Exception as e:
                logger.warning(f"Error calculating mean action probabilities: {str(e)}")
        
        # Log stoploss statistics if available
        if episode.user_data["stoploss_values"]:
            try:
                stoploss_values = np.array(episode.user_data["stoploss_values"])
                episode.custom_metrics["stoploss_mean"] = float(np.mean(stoploss_values))
                episode.custom_metrics["stoploss_std"] = float(np.std(stoploss_values))
                episode.custom_metrics["stoploss_min"] = float(np.min(stoploss_values))
                episode.custom_metrics["stoploss_max"] = float(np.max(stoploss_values))
                
                # Log distribution of stoploss values in buckets
                buckets = np.linspace(0, 1, 11)  # 0.0, 0.1, 0.2, ..., 1.0
                hist, _ = np.histogram(stoploss_values, bins=buckets)
                for i in range(len(hist)):
                    episode.custom_metrics[f"stoploss_bucket_{i*10:02d}pct"] = float(hist[i] / len(stoploss_values))
                    
            except Exception as e:
                logger.warning(f"Error calculating stoploss statistics: {str(e)}")
                
        # Log the action mask if available
        if hasattr(episode, "last_info_for") and episode.last_info_for() is not None:
            info = episode.last_info_for()
            if "action_mask" in info:
                episode.custom_metrics["action_mask"] = str(info["action_mask"])


# Set up logging to file in project directory
log_dir = os.path.dirname(os.path.abspath(__file__))
ray_log_dir = os.path.join(log_dir, 'ray_logs')

# Create directories if they don't exist
os.makedirs(ray_log_dir, exist_ok=True)

def setup_logging():
    # Generate timestamp for log filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_debug.txt')
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    # Set up file handler (no rotation since we're creating new files per run)
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    # Set up console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Add handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return logging.getLogger(__name__), log_file

# Initialize logging
logger, log_file = setup_logging()

# Configure Ray to use project directory for logs
os.environ['RAY_LOG_TO_STDERR'] = '1'
os.environ['RAY_LOGS'] = ray_log_dir

# Function to log to both file and console
def log_message(message, level=logging.INFO):
    logger.log(level, message)
    if level >= logging.INFO:  # Only print INFO and above to console
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")

log_message(f"Logging initialized. All output will be saved to: {log_file}")

class DebugWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.step_count = 0
        self.debug_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_debug.txt')
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.step_count = 0
        with open(self.debug_file, 'a') as f:
            f.write("\n=== Environment Reset ===\n")
            f.write(f"Initial observation: {obs}\n")
        return obs, info
        
    def step(self, action):
        # Unpack the step results
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        
        # Log first 10 steps in detail
        if self.step_count <= 10:
            with open(self.debug_file, 'a') as f:
                f.write(f"\n=== Step {self.step_count} ===\n")
                f.write(f"Action: {action}\n")
                f.write(f"Observation: {obs}\n")
                f.write(f"Reward: {reward}\n")
                f.write(f"Terminated: {terminated}, Truncated: {truncated}\n")
                f.write(f"Info: {info}\n")
        
        # Combine terminated and truncated for compatibility with older Gym versions
        done = terminated or truncated
        return obs, reward, terminated, truncated, info

def create_env(env_config):
    """Create and return a TradingEnv instance with the given config."""
    # Load MGOL data
    df, _ = load_data()
    
    # Clear previous debug file
    debug_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_debug.txt')
    if os.path.exists(debug_file):
        os.remove(debug_file)
    
    # Print first 10 rows of the loaded data
    with open(debug_file, 'a') as f:
        f.write("\nFirst 10 rows of the loaded MGOL data:\n")
        f.write("-" * 80 + "\n")
        f.write(df.head(10).to_string() + "\n")
        f.write("-" * 80 + "\n\n")
        
        # Print data summary
        f.write("\nData Summary:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Rows: {len(df)}\n")
        f.write(f"Date Range: {df.index.min()} to {df.index.max()}\n")
        f.write(f"Columns: {', '.join(df.columns)}\n")
        f.write("-" * 40 + "\n\n")
    
    # Reset index to make datetime a column again
    df = df.reset_index()
    
    # Create environment with debug enabled
    env = TradingEnv(
        df=df,
        max_steps=env_config.get('max_steps', len(df) - 1),
        stoploss=True,
        stoploss_min=env_config.get('stoploss_min', 0.01),
        stoploss_max=env_config.get('stoploss_max', 0.10),
        debug=True,  # Force debug on
        model_name='mgol_rllib',
        test_mode=True  # Ensure debug output is enabled
    )
    
    # Wrap with debug wrapper
    env = DebugWrapper(env)
    return env

def train():
    # Initialize Ray with minimal configuration
    ray.init(
        ignore_reinit_error=True,
        local_mode=True,  # Run in local mode for simplicity
        include_dashboard=False,
        _temp_dir=os.path.join(log_dir, 'ray_temp'),
        logging_level=logging.DEBUG  # Enable debug logging for Ray
    )
    
    logger.info("Ray initialized. Available resources: %s", ray.available_resources())
    
    # Set log level for ray to DEBUG
    logging.getLogger('ray').setLevel(logging.DEBUG)
    
    # Register the environment
    logger.info("Registering TradingEnv with Ray...")
    try:
        register_env(
            "TradingEnv-v1",
            lambda config: TradingEnv(df=load_data()[0], **config)
        )
        logger.info("Successfully registered TradingEnv-v1")
    except Exception as e:
        logger.error("Failed to register environment: %s", str(e))
        raise
    
    # Load MGOL data to get the number of steps
    df, _ = load_data()
    
    # Log data info before training
    log_message("\n" + "="*80)
    log_message("MGOL Data Info:")
    log_message("-"*80)
    log_message(f"Total Rows: {len(df)}")
    log_message(f"Date Range: {df.index.min()} to {df.index.max()}")
    log_message(f"Columns: {', '.join(df.columns)}")
    log_message("\nFirst 10 rows:")
    log_message("\n" + df.head(10).to_string())
    log_message("="*80 + "\n")
    
    # Configuration
    config = (
        PPOConfig()
        .environment(
            "TradingEnv-v1",
            env_config={
                'max_steps': len(df) - 1,
                'stoploss': True,
                'stoploss_min': 0.01,
                'stoploss_max': 0.10,
                'debug': True
            },
            # Enable action masking
            disable_env_checking=True
        )
        .callbacks(ActionProbCallback)  # Add our custom callback
        .framework("torch")
        .training(
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            kl_coeff=0.5,
            num_sgd_iter=10,
            sgd_minibatch_size=64,
            train_batch_size=4000,
            model={
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
                "vf_share_layers": False,
                # Configure model to handle action masking
                "custom_model_config": {
                    "action_mask_key": "action_mask"
                },
                # Use the built-in action masking model
                "custom_model": "ActionMaskModel",
            },
        )
        .experimental(_enable_new_api_stack=False)
        .resources(num_gpus=1 if ray.available_resources().get("GPU", 0) > 0 else 0)
        .rollouts(
            num_rollout_workers=2,  # Reduced for stability
            num_envs_per_worker=1,
            rollout_fragment_length=1000,
            # Ensure action masking is properly handled in rollouts
            batch_mode="complete_episodes"
        )
        .evaluation(
            evaluation_interval=10,
            evaluation_duration=10,
            evaluation_config={
                "explore": False,
                "render_env": False,
            },
        )
    )
    
    # Define stopping conditions
    stop_conditions = {
        "training_iteration": 100,  # Stop after 100 iterations
        "timesteps_total": 100000,  # Or stop after 100k timesteps
        "episode_reward_mean": 1000,  # Stop if we reach a good reward
    }
    
    # Set up the reporter with default metrics, action probabilities, and stoploss metrics
    reporter = CLIReporter(
        max_progress_rows=10,
        metric_columns={
            "training_iteration": "iter",
            "time_total_s": "time",
            "timesteps_total": "ts",
            "episodes_this_iter": "epis",
            "episode_reward_mean": "reward_mean",
            "episode_reward_max": "reward_max",
            "episode_reward_min": "reward_min",
            # Add action probabilities to the metrics
            "custom_metrics/action_0_prob_mean": "P(SELL)",
            "custom_metrics/action_1_prob_mean": "P(HOLD)",
            "custom_metrics/action_2_prob_mean": "P(BUY)",
            # Add stoploss metrics
            "custom_metrics/stoploss_mean_mean": "SL_Mean",
            "custom_metrics/stoploss_std_mean": "SL_Std",
            "custom_metrics/stoploss_min_mean": "SL_Min",
            "custom_metrics/stoploss_max_mean": "SL_Max",
        },
        sort_by_metric=True,
        max_report_frequency=30  # Report at least every 30 seconds
    )
    
    # Set up the run configuration
    run_config = air.RunConfig(
        name="PPO_TradingEnv",
        local_dir="./ray_results",
        verbose=1,
        progress_reporter=reporter,
        stop=stop_conditions  # Add stop conditions to run config
    )
    
    # Enhanced logging function
    def log_progress(trial_id, result):
        log_message("\n" + "="*50)
        log_message(f"Training Progress - Iteration {result['training_iteration']}")
        log_message("="*50)
        
        # Basic metrics
        log_message(f"Trial ID: {trial_id}")
        log_message(f"Timesteps: {result.get('timesteps_total', 0):,}")
        log_message(f"Episodes: {result.get('episodes_total', 0):,}")
        
        # Episode metrics
        log_message("\nEpisode Statistics:")
        log_message(f"  Reward (mean): {result.get('episode_reward_mean', 0):.2f}")
        log_message(f"  Reward (min): {result.get('episode_reward_min', 0):.2f}")
        log_message(f"  Reward (max): {result.get('episode_reward_max', 0):.2f}")
        log_message(f"  Length (mean): {result.get('episode_len_mean', 0):.2f}")
        
        # Training metrics
        if 'info' in result and 'learner' in result['info']:
            learner_info = result['info']['learner']
            log_message("\nTraining Metrics:")
            for pid, stats in learner_info.items():
                if 'learner_stats' in stats:
                    for k, v in stats['learner_stats'].items():
                        log_message(f"  {k}: {v:.4f}")
        
        # Custom metrics
        if 'custom_metrics' in result:
            log_message("\nCustom Metrics:")
            for k, v in result['custom_metrics'].items():
                log_message(f"  {k}: {v:.4f}")
                
        log_message("="*50 + "\n")
        
    config.checkpoint_config = CheckpointConfig(
        num_to_keep=3,
        checkpoint_score_attribute="episode_reward_mean",
        checkpoint_score_order="max",
        checkpoint_frequency=10,
    )
    
    config.callbacks = {
        "on_episode_end": log_progress
    }
    
    # Create a test environment for debugging
    test_env = create_env({})
    obs, _ = test_env.reset()
    
    # Log initial observation
    with open('training_debug.txt', 'a') as f:
        f.write("\n=== Starting Test Run ===\n")
        f.write(f"Initial observation: {obs}\n")
    
    # Run 10 steps manually
    for step in range(10):
        # Sample a random action
        action = test_env.action_space.sample()
        
        # Take a step
        obs, reward, done, truncated, info = test_env.step(action)
        
        # Log the step
        with open('training_debug.txt', 'a') as f:
            f.write(f"\n--- Step {step + 1} ---\n")
            f.write(f"Action: {action}\n")
            f.write(f"Observation: {obs}\n")
            f.write(f"Reward: {reward}\n")
            f.write(f"Done: {done}, Truncated: {truncated}\n")
            f.write(f"Info: {info}\n")
        
        if done or truncated:
            with open('training_debug.txt', 'a') as f:
                f.write("\n--- Episode Finished ---\n")
            break
    
    with open('training_debug.txt', 'a') as f:
        f.write("\n=== Test Run Completed ===\n")
    
    # Now run the actual training
    with open('training_debug.txt', 'a') as f:
        f.write("\n=== Starting RLlib Training ===\n")
    
    # Configure the run
    run_config = RunConfig(
        name="PPO_MGOL_Trading",
        stop={"timesteps_total": 100000},
        checkpoint_config=CheckpointConfig(
            num_to_keep=3,
            checkpoint_score_attribute="episode_reward_mean",
            checkpoint_score_order="max",
            checkpoint_frequency=10,
        ),
        progress_reporter=reporter,
        verbose=1,
    )
    
    try:
        # Run the training with stop conditions
        logger.info("Creating Tuner...")
        try:
            tuner = tune.Tuner(
                "PPO",
                param_space=config,
                run_config=run_config,
            )
            logger.info("Tuner created successfully")
            
            logger.info("Starting training with stop conditions: %s", stop_conditions)
            logger.info("Ray cluster resources: %s", ray.cluster_resources())
            logger.info("Ray available resources: %s", ray.available_resources())
            
            # Start training
            results = tuner.fit()
            
            logger.info("Training completed. Results: %s", results)
            return results
        except Exception as e:
            logger.error("Error during training: %s", str(e), exc_info=True)
            raise
            
    finally:
        # Clean up Ray
        ray.shutdown()

if __name__ == "__main__":
    try:
        # Run training
        log_message("="*80)
        log_message(f"Starting RLlib training with MGOL data at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_message("="*80)
        
        try:
            results = train()
            
            log_message("\n" + "="*80)
            log_message(f"Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            if results:
                best_trial = results.get_best_result(metric="episode_reward_mean", mode="max")
                log_message(f"Best trial: {best_trial}")
                log_message(f"Best reward: {best_trial.metrics.get('episode_reward_mean')}")
            log_message("="*80)
        except Exception as e:
            log_message(f"Error during training: {str(e)}", level=logging.ERROR)
            raise
            
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise
