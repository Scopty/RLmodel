from common_imports import *
from trading_env_sb3_ver2d import TradingEnv
import time
import datetime

# === Configuration ===
DEBUG_MODE = False  # Set to True to enable verbose output and debugging

# === Minimal Output Mode ===
MINIMAL_OUTPUT = True  # Set to True for minimal console output

# Create directories for saving
log_dir = "./tensorboard_logs/"
os.makedirs(log_dir, exist_ok=True)

num_cpu = 12  # Use more CPUs for parallel training

def make_env():
    env = TradingEnv(df)  # Initialize env
    check_env(env, warn=True)
    return env

if __name__ == "__main__":
    # Add timing measurements
    start_time = time.time()
    print(f"Starting training at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create parallel environments using SubprocVecEnv
    train_env = SubprocVecEnv([make_env for _ in range(num_cpu)])
    train_env = VecMonitor(train_env)
    train_env = VecNormalize(train_env, training=True, norm_reward=True, norm_obs=True)

    # Create parallel environments for evaluation
    eval_env = SubprocVecEnv([make_env for _ in range(1)])
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, training=False, norm_reward=False)

    # Create evaluation callback
    eval_callback = DebugEvalCallback(
        eval_env,
        best_model_save_path="best_model",
        log_path=log_dir,
        eval_freq=1000,  # More frequent evaluation
        deterministic=True,
        render=False,
        n_eval_episodes=10,
        verbose=1
    )

    # Create and train the model with enhanced PPO settings
    model = MaskablePPO(
        MaskableActorCriticPolicy,
        train_env,
        verbose=0,  # Always silent mode
        tensorboard_log=log_dir,
        learning_rate=1e-4,
        ent_coef=0.02,
        gamma=0.995,
        gae_lambda=0.98,
        n_steps=128,
        clip_range=0.2,
        clip_range_vf=0.2,
        n_epochs=20,
        batch_size=64,
        max_grad_norm=0.5,
        vf_coef=0.5,
        normalize_advantage=True,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[512, 512, 256],
                vf=[512, 512, 256]
            )
        )
    )
    
    # Train the model with minimal output
    print("Training started...")
    model.learn(
        total_timesteps=500000,
        callback=[eval_callback],
        tb_log_name="ppo_custom_overfit"
    )
    print("Training completed")
    
    # Save final model and normalization stats
    model.save("final_model")
    train_env.save("vec_normalize.pkl")
    
    # Calculate and print training time
    end_time = time.time()
    training_time = end_time - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
