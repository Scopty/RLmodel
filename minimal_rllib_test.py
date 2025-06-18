import os
import warnings
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import ray
from ray.rllib.algorithms.ppo import PPOConfig
import gymnasium as gym
from gymnasium import Wrapper

# Configure logging to only show ERROR level and above
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class RenderDuringTrainingWrapper(Wrapper):
    """Wrapper that renders the environment during training."""
    def __init__(self, env, render_freq: int = 1):
        super().__init__(env)
        self.render_freq = render_freq
        self.episode_count = 0
        self.render_mode = "human"
        
    def step(self, action):
        # Render before taking the step
        if self.render_freq > 0 and self.episode_count % self.render_freq == 0:
            self.render()
            
        obs, reward, done, truncated, info = self.env.step(action)
        return obs, reward, done, truncated, info
    
    def reset(self, **kwargs):
        self.episode_count += 1
        return self.env.reset(**kwargs)

def create_env(env_config: Optional[Dict[str, Any]] = None) -> gym.Env:
    """Create and return a wrapped environment for training with rendering."""
    env = gym.make("CartPole-v1")
    # Wrap the environment to enable rendering during training
    env = RenderDuringTrainingWrapper(env, render_freq=1)  # Render every episode
    return env

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def render_episode(env, policy, max_steps=200, training_iter=None):
    """Render a single episode using the trained policy."""
    obs, _ = env.reset()
    total_reward = 0
    frames = []
    
    for step in range(max_steps):
        # Render the environment
        if env.render_mode == 'human':
            env.render()
        elif env.render_mode == 'rgb_array':
            frames.append(env.render())
        
        # Get action from policy - handle different action formats
        action_info = policy.compute_single_action(obs, explore=False)
        
        # Extract action from the action info
        if isinstance(action_info, tuple):
            action = action_info[0]
        elif isinstance(action_info, dict):
            action = action_info['action']
        else:
            action = action_info
            
        if hasattr(env.action_space, 'n') and isinstance(action, np.ndarray):
            action = int(action.item())
        
        # Take the action
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        
        if done or truncated:
            if training_iter is not None:
                print(f"Training Iteration {training_iter}: Episode finished after {step+1} steps. Reward: {total_reward:.1f}")
            else:
                print(f"Episode finished after {step+1} steps. Total reward: {total_reward:.1f}")
            break
    
    return total_reward, frames

def test_minimal_rllib():
    """Test basic RLlib functionality with CartPole-v1."""
    try:
        # Initialize Ray with minimal logging
        ray.init(ignore_reinit_error=True, logging_level=logging.ERROR)
        
        # Configure PPO with minimal output
        config = (
            PPOConfig()
            .environment("CartPole-v1")
            .framework("torch")
            .training(lr=0.001)
            .rollouts(num_rollout_workers=1)
            .debugging(log_level="ERROR")
            # Faster training for demo purposes
            .training(train_batch_size=400, sgd_minibatch_size=100, num_sgd_iter=5)
        )
        
        # Create a test environment for rendering
        test_env = gym.make("CartPole-v1", render_mode='human')
        
        # Build the algorithm
        algo = config.build()
        
        # Get the policy
        policy = algo.get_policy()
        
        # Training loop with rendering
        num_iterations = 10
        print(f"Training for {num_iterations} iterations with rendering...\n")
        
        for i in range(num_iterations):
            # Train for one iteration
            result = algo.train()
            mean_reward = result['episode_reward_mean']
            
            # Render a test episode
            print(f"\n--- Iteration {i+1} (Mean Reward: {mean_reward:.1f}) ---")
            episode_reward, _ = render_episode(test_env, policy, training_iter=i+1)
            
            # Early stopping if we've solved the environment
            if episode_reward >= 195:  # CartPole is considered solved at 195
                print("\nEnvironment solved! Stopping training.")
                break
        
        # Final evaluation
        print("\n--- Final Evaluation ---")
        num_episodes = 3
        total_rewards = []
        
        for i in range(num_episodes):
            print(f"\n--- Test Episode {i+1} ---")
            episode_reward, _ = render_episode(test_env, policy)
            total_rewards.append(episode_reward)
        
        print(f"\nAverage reward over {num_episodes} test episodes: {np.mean(total_rewards):.1f}")
        return True
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'test_env' in locals():
            test_env.close()
        ray.shutdown()

if __name__ == "__main__":
    test_minimal_rllib()
