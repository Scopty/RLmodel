import os, psutil

# Determine number of available (idle) cores
IDLE_THRESHOLD = 20.0  # percent
cpu_usages = psutil.cpu_percent(percpu=True, interval=1)
available_cores = sum(usage < IDLE_THRESHOLD for usage in cpu_usages)
available_cores = max(1, available_cores) - 4 # At least 1

available_cores = os.cpu_count()

import mplfinance
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# Stable-Baselines3 imports
from stable_baselines3 import PPO,A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv  # Use this instead of DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# Gymnasium (updated from Gym)
import gymnasium as gym
from gymnasium import Env,Wrapper
from gymnasium.spaces import Discrete, Box

# Load Data
df = pd.read_csv("MGOL.csv")  # Replace with actual file
df['datetime'] = pd.to_datetime(df['datetime'], format='%m/%d/%y %H:%M')
df.set_index('datetime', inplace=True)

df.index = df.index + pd.Timedelta(hours=3)
df.index.name = 'Date'
df = df.drop(columns=['symbol', 'frame'])
df = df.iloc[:10]  # Select the first 30 rows
df_original = df
df = df[["close"]]

class ActionMasker(gym.Wrapper):
    """
    Wrapper for action masking in environments.
    Adds action mask as a part of the environment step.
    """
    def __init__(self, env: gym.Env, mask_fn: callable):
        super().__init__(env)
        self.mask_fn = mask_fn

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Get the action mask
        action_mask = self.mask_fn(self.env)
        
        # Add the action mask to the info dictionary
        info['action_mask'] = action_mask
        
        return obs, reward, done, truncated, info
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs


from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn

# Define custom features extractor
class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        
        # Define the neural network architecture
        self.net = nn.Sequential(
            nn.Linear(8, 128),  # Input size matches the environment's observation space
            nn.ReLU(),
            nn.Linear(128, 64),  # Match the features_dim
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # For vectorized environments, observations will have shape (num_envs, obs_dim)
        # We need to handle this batch dimension correctly
        batch_size = observations.shape[0]
        
        # Extract the first 8 features and action mask
        features = observations[:, :8].to(dtype=torch.float32)  # First 8 elements are features
        action_mask = observations[:, 8:11].to(dtype=torch.float32)  # Next 3 elements are action mask
        
        # Process features through the network
        processed_features = self.net(features)
        return processed_features.view(batch_size, -1)

#import torch
#import torch.nn.functional as F

class MaskedPPOPolicy(ActorCriticPolicy):
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, lr_schedule,
                 net_arch=None, activation_fn=nn.Tanh, features_extractor_class=CustomFeaturesExtractor,
                 features_extractor_kwargs=None, normalize_images=True, optimizer_class=torch.optim.Adam,
                 optimizer_kwargs=None, use_sde=False):
        super().__init__(observation_space, action_space, lr_schedule, net_arch=net_arch,
                         activation_fn=activation_fn, features_extractor_class=features_extractor_class,
                         features_extractor_kwargs=features_extractor_kwargs, normalize_images=normalize_images,
                         optimizer_class=optimizer_class, optimizer_kwargs=optimizer_kwargs,
                         use_sde=use_sde)

    def forward(self, combined_obs, action_mask=None, deterministic=False):
        # For vectorized environments, combined_obs will have shape (num_envs, obs_dim)
        batch_size = combined_obs.shape[0]
        
        # Use the entire observation as features (12 dimensions)
        features = combined_obs.to(dtype=torch.float32)
        
        # Extract action mask from the last 3 elements
        action_mask = features[:, -3:]  # Last 3 elements are action mask
        
        # Pass through the MLP extractor
        latent_pi, latent_vf = self.mlp_extractor(features[:, :8])  # Only pass the first 8 features

        # Compute action distribution and value
        distribution = self._get_action_dist_from_latent(latent_pi)
        values = self.value_net(latent_vf).to(dtype=torch.float32)

        # Handle action mask
        if action_mask is None:
            # Extract action mask from the last 3 elements of the observation
            action_mask = combined_obs[:, -3:]  # Last 3 elements are action mask

        if action_mask is not None:
            action_mask_tensor = torch.as_tensor(action_mask, dtype=torch.float32, device=combined_obs.device)
            action_mask_tensor = action_mask_tensor.view(-1, distribution.distribution.logits.shape[-1])
            distribution.distribution.logits = distribution.distribution.logits.masked_fill(action_mask_tensor == 0, -1e9)

        # Action selection
        if deterministic:
            actions = torch.argmax(distribution.distribution.probs, dim=-1)
        else:
            actions = distribution.sample()

        return actions, values, distribution.log_prob(actions)
        latent_pi, latent_vf = self.mlp_extractor(features.to(dtype=torch.float32))

        # Compute action distribution and value
        distribution = self._get_action_dist_from_latent(latent_pi)
        values = self.value_net(latent_vf).to(dtype=torch.float32)

        # Handle action mask
        if action_mask is None:
            action_mask = combined_obs[:, 5:]  # Last 3 elements are action mask

        if action_mask is not None:
            action_mask_tensor = torch.as_tensor(action_mask, dtype=torch.float32, device=combined_obs.device)
            action_mask_tensor = action_mask_tensor.view(-1, distribution.distribution.logits.shape[-1])
            distribution.distribution.logits = distribution.distribution.logits.masked_fill(action_mask_tensor == 0, -1e9)

        # Action selection
        if deterministic:
            actions = torch.argmax(distribution.distribution.probs, dim=-1)
        else:
            actions = distribution.sample()

        return actions, values, distribution.log_prob(actions)

        log_probs = distribution.log_prob(actions)

        if debug: print(f"  - Selected Actions: {actions}")

        return actions, values, log_probs

    def predict(self, observation, state=None, mask=None, action_mask=None, deterministic=False):
        with torch.no_grad():
            observation = torch.as_tensor(observation, device=self.device, dtype=torch.float32)
            actions, _, _ = self.forward(observation, action_mask=action_mask, deterministic=deterministic)
            actions = actions.cpu().numpy()
        return actions, state

debug = False

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



from stable_baselines3.common.callbacks import BaseCallback

class DebugCallback(BaseCallback):
    def __init__(self, debug_episodes=None, verbose=0, debug=False):
        super().__init__(verbose)
        self.debug_episodes = set(debug_episodes) if debug_episodes is not None else set()
        self.episode_counts = []
        self.episode_step_counts = {}  # Track steps per environment
        self.debug = debug
        self.debug_triggered = False  # Optional: Only needed if you want to trigger once
        self.debug_on_step = None     # Optional: If you plan to use this
        self.last_printed_episodes = []  # To store the last printed episode number
        self.max_episode_steps = {}  # Track the max steps per environment

    def _init_callback(self) -> None:
        n_envs = self.training_env.num_envs
        # Initialize the counts and step tracking for all environments
        self.episode_counts = [0] * n_envs
        self.last_printed_episodes = [None] * n_envs  # Make sure this list is of the correct size
        self.episode_step_counts = {i: 0 for i in range(n_envs)}  # Initialize step counts for all environments
        self.max_episode_steps = {i: 0 for i in range(n_envs)}  # Initialize the max episode steps per environment

    def _on_training_start(self) -> None:
        num_envs = getattr(self.training_env, "num_envs", 1)
        self.episode_counts = [0] * num_envs
        self.current_episode_steps = [0] * num_envs

    def on_training_end(self) -> None:
        # Print the maximum steps per environment after training ends
        for env_id, max_steps in self.max_episode_steps.items():
            if self.debug: print(f"Max steps in episode for env {env_id}: {max_steps}")

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for i in range(len(dones)):
            # Ensure that episode_step_counts has the environment index initialized
            if i not in self.episode_step_counts:
                self.episode_step_counts[i] = 0

            # Track steps
            self.episode_step_counts[i] += 1

            # Update the maximum episode steps for each environment
            if self.episode_step_counts[i] > self.max_episode_steps[i]:
                self.max_episode_steps[i] = self.episode_step_counts[i]
            
            episode_num = self.episode_counts[i]

            # Print start of episode (optional)
            if self.last_printed_episodes[i] != episode_num:
                if self.debug: print(f"Current episode (env {i}): {episode_num}")
                self.last_printed_episodes[i] = episode_num

            # Debug output if in debug_episodes list
            if episode_num in self.debug_episodes:
                if self.debug: print(f"Current step (env {i}): {self.episode_step_counts[i]}")
                if self.debug: print(f"[Env {i}] dones: {dones[i]} infos: {infos[i]}")

            # If episode is done, print step count and reset counter
            if dones[i]:
                if self.debug: print(f"Episode {episode_num} (env {i}) finished in {self.episode_step_counts[i]} steps.")
                self.episode_counts[i] += 1
                self.episode_step_counts[i] = 0  # Reset for next episode

        return True


class RewardCallback(BaseCallback):
    def __init__(self, debug=False, verbose=0):
        super().__init__(verbose)
        self.debug = debug
        self.n_envs = None
        self.episode_count = 0
        self.episode_count_per_env = None
        self.total_steps_per_env = None
        
        # New tracking
        self.current_rewards = None
        self.invalid_actions = None
        self.current_steps = None
        self.total_steps = 0
        
        # Old tracking (for compatibility)
        self.iteration_rewards = []
        self.iteration_invalid_actions = []

    def _on_training_start(self) -> None:
        self.n_envs = self.training_env.num_envs
        self.episode_count_per_env = [0 for _ in range(self.n_envs)]
        self.total_steps_per_env = [0 for _ in range(self.n_envs)]
        self.current_rewards = [0.0 for _ in range(self.n_envs)]
        self.invalid_actions = [0 for _ in range(self.n_envs)]
        self.current_steps = [0 for _ in range(self.n_envs)]

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        infos = self.locals["infos"]

        for i in range(self.n_envs):
            if not dones[i]:
                self.current_steps[i] += 1
                self.total_steps += 1
                self.total_steps_per_env[i] += 1
            
            self.current_rewards[i] += rewards[i]
            
            if infos[i].get("invalid_action", False):
                self.invalid_actions[i] += 1

            if dones[i]:
                self.episode_count += 1
                self.episode_count_per_env[i] += 1
                
                if self.debug:
                    print(f"[Env {i}] Episode {self.episode_count_per_env[i]} Done. "
                          f"Steps: {self.current_steps[i]}, "
                          f"Reward: {self.current_rewards[i]:.2f}, "
                          f"Invalid: {self.invalid_actions[i]}")
                
                # Add to old tracking system
                self.iteration_rewards.append(self.current_rewards[i])
                self.iteration_invalid_actions.append(self.invalid_actions[i])
                
                # Reset episode-specific metrics
                self.current_steps[i] = 0
                self.current_rewards[i] = 0.0
                self.invalid_actions[i] = 0

        return True

    def _on_training_end(self) -> None:
        print("\n=== Final Training Summary ===")
        print(f"Total episodes completed: {self.episode_count}")
        print(f"Total steps taken: {self.total_steps}")
        
        for i in range(self.n_envs):
            print(f"\n[Env {i}] Summary:")
            print(f"Total episodes: {self.episode_count_per_env[i]}")
            print(f"Total steps: {self.total_steps_per_env[i]}")
            print(f"Average steps per episode: {self.total_steps_per_env[i] / self.episode_count_per_env[i]:.1f}")
        
        # Print old format for compatibility
        formatted_rewards = ', '.join(f"{reward:.1f}" for reward in self.iteration_rewards)
        formatted_invalid_actions = ', '.join(str(invalid) for invalid in self.iteration_invalid_actions)
        print("\nTotal rewards per iteration:", formatted_rewards)
        print("Invalid actions per iteration:", formatted_invalid_actions)

# Initialize parallel environments and train model
import torch
from trading_env_sb3_ver1 import TradingEnv
from stable_baselines3.common.callbacks import CallbackList

def mask_fn(env: gym.Env):
    return env.get_action_mask()

# Define the number of CPU cores to use
num_cpu = psutil.cpu_count(logical=False)  # Use physical cores

def make_env():
    def _init():
        env = TradingEnv(df)  # Your DataFrame must be accessible here
        return ActionMasker(env, mask_fn)
    return _init

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    
    # Create parallel environments using SubprocVecEnv
    env = SubprocVecEnv([make_env() for _ in range(num_cpu)])

    # Define PPO model with the custom policy using the vectorized environment
    ppo_masked_model = PPO(
        policy=MaskedPPOPolicy,
        env=env,
        verbose=0,
        tensorboard_log="./tensorboard_logs/",
        policy_kwargs={
            "features_extractor_class": CustomFeaturesExtractor,
            "net_arch": dict(pi=[64, 64], vf=[64, 64]),  # Use dictionary format
            "activation_fn": nn.ReLU  # Match activation function
        },
        n_steps=1024,
        batch_size=64,
        learning_rate=3e-4,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=42,
        use_sde=False
    )

    # Initialize the callback
    reward_callback = RewardCallback(debug=True)
    debug_callback = DebugCallback(debug_episodes={0, 100, 200})
    
    # Train the model with the callback
    try:
        ppo_masked_model.learn(
            total_timesteps=10000,
            progress_bar=False,
            tb_log_name="sb3_ppo",
            callback=CallbackList([reward_callback, debug_callback])
        )
        
        # Print training statistics after training
        print("\nTraining statistics:")
        print("Total rewards per iteration:", 
              ", ".join(f"{reward:.1f}" for reward in reward_callback.iteration_rewards))
        print("Invalid actions per iteration:", 
              ", ".join(str(invalid) for invalid in reward_callback.iteration_invalid_actions))
        
        # Print episode statistics per environment
        n_envs = env.num_envs
        for i in range(n_envs):
            print(f"\nEnvironment {i} statistics:")
            print(f"Total episodes: {reward_callback.episode_count_per_env[i]}")
            print(f"Total steps: {reward_callback.total_steps_per_env[i]}")
            print(f"Average steps per episode: {reward_callback.total_steps_per_env[i] / reward_callback.episode_count_per_env[i]:.1f}")
            
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        # Save the model
        ppo_masked_model.save("ppo_masked_model")
        print("Model saved successfully")
        
        # Close the environment
        try:
            env.close()
        except EOFError:
            pass

