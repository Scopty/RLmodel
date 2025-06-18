import os
import ray
import logging
from datetime import datetime
import numpy as np
import gymnasium as gym
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env, register_input
from ray.air import RunConfig
from ray.tune.logger import pretty_print
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch

# Import torch
torch, nn = try_import_torch()

# Define a simple action masking model
class ActionMaskModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs)
        nn.Module.__init__(self)
        
        self.original_space = obs_space.original_space if hasattr(obs_space, 'original_space') else obs_space
        
        # Define network architecture
        self.fc1 = nn.Linear(obs_space.shape[0], 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, num_outputs)
        self.value_out = nn.Linear(64, 1)
        
        # Store the last computed value
        self._value = None
    
    def forward(self, input_dict, state, seq_lens):
        x = torch.relu(self.fc1(input_dict["obs_flat"]))
        x = torch.relu(self.fc2(x))
        self._value = self.value_out(x).squeeze(1)
        return self.out(x), state
    
    def value_function(self):
        return self._value

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Clear the log file at the start
with open("training_debug.log", "w") as f:
    f.write("")

# Import your environment after setting up logging
from trading_env import TradingEnv
from common_imports import load_data

def create_env(config):
    df, _ = load_data()
    return TradingEnv(df=df, **config)

def main():
    # Initialize Ray
    if ray.is_initialized():
        ray.shutdown()
    
    ray.init(ignore_reinit_error=True, num_cpus=4, num_gpus=0)
    logger.info(f"Ray initialized. Resources: {ray.available_resources()}")
    
    # Register the action masking model
    try:
        ModelCatalog.register_custom_model("ActionMaskModel", ActionMaskModel)
        logger.info("Successfully registered ActionMaskModel")
    except Exception as e:
        logger.error(f"Failed to register ActionMaskModel: {e}")
        return
    
    # Register the environment
    try:
        register_env("TradingEnv-v1", create_env)
        logger.info("Successfully registered TradingEnv-v1")
    except Exception as e:
        logger.error(f"Failed to register environment: {e}")
        return
    
    # Configuration
    config = (
        PPOConfig()
        .environment(
            "TradingEnv-v1",
            env_config={
                'max_steps': 1000,  # Reduced for testing
                'stoploss': True,
                'stoploss_min': 0.01,
                'stoploss_max': 0.10,
                'debug': True
            },
            disable_env_checking=True
        )
        .framework("torch")
        .training(
            lr=3e-4,
            gamma=0.99,
            train_batch_size=1000,
            model={
                "custom_model": "ActionMaskModel",
                "custom_model_config": {
                    "action_mask_key": "action_mask"
                },
                # Disable default model layers since we're using a custom model
                "_disable_preprocessor_api": True,
                "_disable_action_flattening": True,
            },
        )
        .rollouts(
            num_rollout_workers=1,  # Reduced for testing
            num_envs_per_worker=1,
            batch_mode="complete_episodes"
        )
    )
    
    # Run configuration
    run_config = RunConfig(
        name="PPO_TestRun",
        local_dir="./ray_results",
        verbose=1,
        stop={"training_iteration": 5}  # Just run 5 iterations for testing
    )
    
    # Run training
    try:
        logger.info("Starting training...")
        tuner = tune.Tuner(
            "PPO",
            param_space=config,
            run_config=run_config,
        )
        
        results = tuner.fit()
        logger.info("Training completed successfully!")
        logger.info(f"Best trial results: {results.get_best_result().metrics}")
        
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
    finally:
        ray.shutdown()
        logger.info("Ray shutdown complete")

if __name__ == "__main__":
    main()
