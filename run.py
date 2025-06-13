#!/usr/bin/env python3
"""
Master script that trains a model, tests it, and generates plots in one go.

Usage:
    python train_test_plot.py [--total_timesteps TOTAL_TIMESTEPS] [--max_steps MAX_STEPS] [--debug]

Example:
    python train_test_plot.py --total_timesteps 100000 --max_steps 100 --debug
"""

import os
import sys
import subprocess
import argparse
import re
from datetime import datetime

def run_command(command, cwd=None):
    """Run a shell command and return its output."""
    print(f"\n[RUNNING] {' '.join(command)}")
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            text=True,
            capture_output=True
        )
        print(result.stdout)
        if result.stderr:
            print(f"[STDERR] {result.stderr}", file=sys.stderr)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed with code {e.returncode}", file=sys.stderr)
        print(f"[STDERR] {e.stderr}", file=sys.stderr)
        sys.exit(1)

def find_latest_training_dir():
    """Find the most recent training output directory."""
    training_dirs = [d for d in os.listdir('training_output') 
                    if os.path.isdir(os.path.join('training_output', d)) and d.startswith('A')]
    
    if not training_dirs:
        print("No training directories found in training_output/")
        return None
        
    # Sort by creation time (newest first)
    training_dirs.sort(key=lambda x: os.path.getmtime(os.path.join('training_output', x)), reverse=True)
    return training_dirs[0]

def main():
    # Parse command line arguments (same as training_script.py)
    parser = argparse.ArgumentParser(description='Train, test, and plot trading model')
    parser.add_argument('--total_timesteps', type=int, default=10000, help='Total number of timesteps for training')
    parser.add_argument('--max_steps', type=int, default=100, help='Maximum steps per episode')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()
    
    # Step 1: Train the model
    train_cmd = [
        'python', 'training_script.py',
        f'--total_timesteps={args.total_timesteps}',
        f'--max_steps={args.max_steps}'
    ]
    if args.debug:
        train_cmd.append('--debug')
    
    train_output = run_command(train_cmd)
    
    # Extract the training directory from the output
    match = re.search(r'Output will be saved to: training_output/(A\d+_output_\d+_bars_\d+_timesteps_\d{8}_\d{6})', train_output)
    if not match:
        print("Could not determine training directory from output. Trying to find latest...")
        training_dir = find_latest_training_dir()
        if not training_dir:
            print("Failed to find training directory. Exiting.")
            sys.exit(1)
    else:
        training_dir = match.group(1)
    
    print(f"\n[INFO] Using training directory: {training_dir}")
    
    # Step 2: Test the model
    test_cmd = [
        'python', 'test_script.py',
        f'--input=training_output/{training_dir}'
    ]
    if args.debug:
        test_cmd.append('--debug')
    
    test_output = run_command(test_cmd)
    
    # Step 3: Generate plots
    plot_cmd = [
        'python', 'plot_signals.py',
        f'--input=test_output/{training_dir}'
    ]
    if args.debug:
        plot_cmd.append('--debug')
    
    plot_output = run_command(plot_cmd)
    
    print("\n[COMPLETE] Training, testing, and plotting completed successfully!")
    print(f"- Training output: training_output/{training_dir}")
    print(f"- Test results: test_output/{training_dir}")
    print(f"- Plots: plot_results/{training_dir}")

if __name__ == "__main__":
    main()
