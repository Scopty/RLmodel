import importlib
import common_imports
importlib.reload(common_imports)
from common_imports import *
import time
import os
import shutil
import json
import argparse
from datetime import datetime

# === Configuration ===
DEBUG_MODE = False  # Set to True to enable verbose output and debugging

# === Minimal Output Mode ===
MINIMAL_OUTPUT = True  # Set to True for minimal console output

# Default training parameters
DEFAULT_MAX_STEPS = 100
DEFAULT_TOTAL_TIMESTEPS = 500000
DEFAULT_NUM_CPU = 12  # Use more CPUs for parallel training

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a trading model with specified parameters')
    parser.add_argument('--max_steps', type=int, default=DEFAULT_MAX_STEPS,
                      help=f'Maximum number of steps per episode (default: {DEFAULT_MAX_STEPS})')
    parser.add_argument('--total_timesteps', type=int, default=DEFAULT_TOTAL_TIMESTEPS,
                      help=f'Total number of timesteps for training (default: {DEFAULT_TOTAL_TIMESTEPS:,})')
    parser.add_argument('--num_cpu', type=int, default=DEFAULT_NUM_CPU,
                      help=f'Number of parallel environments (default: {DEFAULT_NUM_CPU})')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug output')
    return parser.parse_args()

# Parse command line arguments
args = parse_arguments()
max_steps = args.max_steps
total_timesteps = args.total_timesteps
num_cpu = args.num_cpu

if args.debug:
    DEBUG_MODE = True
    MINIMAL_OUTPUT = False
    print("Debug mode enabled. Verbose output will be shown.")

def get_next_run_id():
    """Get the next run ID in sequence (A0001, A0002, etc.)"""
    # Create training_output directory if it doesn't exist
    os.makedirs("training_output", exist_ok=True)
    
    # Try to read the last used serial from file
    serial_file = os.path.join("training_output", ".last_serial")
    
    try:
        if os.path.exists(serial_file):
            with open(serial_file, 'r') as f:
                last_serial = int(f.read().strip())
        else:
            # If file doesn't exist, find the highest existing run ID
            last_serial = 0
            for name in os.listdir("training_output"):
                if name.startswith('A') and name[5] == '_' and name[1:5].isdigit():
                    run_num = int(name[1:5])
                    if run_num > last_serial:
                        last_serial = run_num
        
        # Increment the serial
        next_serial = last_serial + 1
        
        # Save the new serial for next time
        with open(serial_file, 'w') as f:
            f.write(str(next_serial))
            
        return f"A{next_serial:04d}"
        
    except Exception as e:
        print(f"Warning: Could not read/write serial file: {e}")
        # Fallback to timestamp-based ID if there's an error
        return f"T{int(time.time())}"

def setup_output_dir():
    """Set up the output directory with a sequential run ID."""
    # Get the next run ID
    run_id = get_next_run_id()
    
    # Generate the timestamp for the output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create the file suffix and output directory name
    global file_suffix, output_dir
    file_suffix = f"{max_steps}_bars_{total_timesteps}_timesteps_{timestamp}"
    output_dir = os.path.join("training_output", f"{run_id}_output_{file_suffix}")
    
    # Create the directory
    os.makedirs(output_dir, exist_ok=True)
    return file_suffix, output_dir

# Set up output directory
file_suffix, output_dir = setup_output_dir()

# Load the data
df, _ = load_data(max_steps=max_steps)


def make_env():
    env = TradingEnv(df, max_steps=max_steps)  # Pass max_steps to TradingEnv
    check_env(env, warn=True)
    return env

if __name__ == "__main__":
    # Set up output directory with timestamp
    file_suffix, output_dir = setup_output_dir()
    
    # Add timing measurements
    start_time = time.time()
    print(f"Starting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output will be saved to: {os.path.abspath(output_dir)}")
    
    # Create parallel environments using SubprocVecEnv
    train_env = SubprocVecEnv([make_env for _ in range(num_cpu)])
    train_env = VecMonitor(train_env)
    train_env = VecNormalize(train_env, training=True, norm_reward=True, norm_obs=True)

    # Create parallel environments for evaluation
    eval_env = SubprocVecEnv([make_env for _ in range(1)])
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, training=False, norm_reward=False)
    
    # Create directory for logs
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Set best model path with timestamp suffix and .zip extension
    best_model_path = os.path.join(output_dir, f'best_model_{file_suffix}.zip')
    # Evaluate every 500 timesteps for better feedback during short training runs
    eval_callback = DebugEvalCallback(
        eval_env,
        best_model_save_path=best_model_path,  # This will create best_model_<timestamp>.zip in output_dir
        log_path=log_dir,
        eval_freq=500,  # More frequent evaluation for short runs
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
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback],
            tb_log_name=os.path.basename(output_dir)
        )
        
        # Save final model and normalization stats
        final_model_path = os.path.join(output_dir, f"final_model_{file_suffix}")
        norm_stats_path = os.path.join(output_dir, f"vec_normalize_{file_suffix}.pkl")
        
        model.save(final_model_path)
        train_env.save(norm_stats_path)
        
        # Save a copy of the training script for reference
        training_script_path = os.path.join(output_dir, "training_script.py.bak")
        shutil.copy2(__file__, training_script_path)
        
        # Save training configuration
        config = {
            'max_steps': max_steps,
            'total_timesteps': total_timesteps,
            'start_time': datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_architecture': str(model.policy),
            'env_parameters': str(train_env.get_attr('envs')[0].env.get_parameters() if hasattr(train_env, 'envs') else {})
        }
        
        with open(os.path.join(output_dir, 'training_config.json'), 'w') as f:
            json.dump(config, f, indent=4)
            
        # Save normalization stats
        vec_normalize_path = os.path.join(output_dir, f'vec_normalize_{file_suffix}.pkl')
        model.get_vec_normalize_env().save(vec_normalize_path)
        
        print("\n=== Training completed successfully ===")
        print(f"Final model saved to: {final_model_path}")
        print(f"Best model saved to: {best_model_path}")
        print(f"Normalization stats saved to: {vec_normalize_path}")
        
    except Exception as e:
        print(f"\n!!! Training failed with error: {str(e)}")
        # Save partial results if possible
        try:
            final_model_path = os.path.join(output_dir, f'final_model_{file_suffix}')
            model.save(final_model_path)
            print(f"\nFinal model saved to: {final_model_path}")
            print(f"Best model saved to: {best_model_path}")
        except Exception as save_error:
            print(f"Failed to save partial model: {str(save_error)}")
        raise
    
    finally:
        # Always calculate and print training time
        end_time = time.time()
        training_time = end_time - start_time
        hours, rem = divmod(training_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"\nTotal training time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d} (h:m:s)")
        print(f"Output directory: {os.path.abspath(output_dir)}")
