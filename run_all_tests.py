#!/usr/bin/env python3
"""
Run tests on all trained models in the training_output directory.

This script will:
1. Find all model directories in training_output/
2. Run test_script.py on each model
3. Save results in test_output/<model_name>/
4. Generate a summary report
"""

import os
import subprocess
import argparse
import json
from pathlib import Path
import pandas as pd
from datetime import datetime

def find_model_directories(training_dir="training_output"):
    """Find all model directories in the training_output directory."""
    if not os.path.exists(training_dir):
        print(f"Error: Directory not found: {training_dir}")
        return []
    
    # Get all directories that match the output pattern
    model_dirs = []
    for item in os.listdir(training_dir):
        item_path = os.path.join(training_dir, item)
        if os.path.isdir(item_path) and item.startswith('output_'):
            model_dirs.append(item_path)
    
    return sorted(model_dirs)

def run_test(model_dir, output_dir, debug=False):
    """Run test_script.py on a single model directory."""
    model_name = os.path.basename(model_dir)
    print(f"\n{'='*80}")
    print(f"Testing model: {model_name}")
    print(f"{'='*80}")
    
    # Create output directory for this test
    test_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Build the test command
    cmd = [
        'python', 'test_script.py',
        '--input', model_dir,
        '--output_dir', test_output_dir
    ]
    
    if debug:
        cmd.append('--debug')
    
    # Run the test
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        # Save the output
        with open(os.path.join(test_output_dir, 'test_output.txt'), 'w') as f:
            f.write(f"Command: {' '.join(cmd)}\n\n")
            f.write("="*80 + "\nSTDOUT:\n" + "="*80 + "\n")
            f.write(result.stdout)
            f.write("\n" + "="*80 + "\nSTDERR:\n" + "="*80 + "\n")
            f.write(result.stderr)
        
        print(f"Test completed. Output saved to: {test_output_dir}")
        output_dir = os.path.join('test_output', model_name)
        # Check if any CSV file exists in the output directory
        csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
        success = result.returncode == 0 and len(csv_files) > 0
        return {
            'model': os.path.basename(model_dir),
            'test_dir': os.path.basename(test_output_dir),
            'returncode': result.returncode,
            'success': success
        }
    except Exception as e:
        print(f"Error running test: {e}")
        return {
            'model': os.path.basename(model_dir),
            'test_dir': os.path.basename(test_output_dir),
            'returncode': -1,
            'success': False,
            'error': str(e)
        }

def generate_summary(results, output_dir):
    """Generate a summary of test results (console only)."""
    if not results:
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Calculate summary stats
    total = len(df)
    passed = df['success'].sum()
    failed = total - passed
    
    # Print summary to console only
    print("\n" + "="*80)
    print(f"Test Summary: {passed}/{total} tests passed ({(passed/total*100):.1f}%)")
    if failed > 0:
        failed_tests = df[~df['success']]
        print("\nFailed tests:")
        for _, row in failed_tests.iterrows():
            print(f"- {row['model']}" + (f" - {row['error']}" if 'error' in row and pd.notna(row['error']) else ""))
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description='Test trained models')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--training_dir',
                      help='Directory containing trained models')
    group.add_argument('--model_dir',
                      help='Test a single model directory')
    
    parser.add_argument('--output_dir', default='test_output',
                      help='Directory to save test results (default: test_output)')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug output')
    args = parser.parse_args()
    
    # Create base output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Saving test results to: {os.path.abspath(args.output_dir)}")
    
    # Handle single model directory
    if args.model_dir:
        if not os.path.isdir(args.model_dir):
            print(f"Error: Model directory not found: {args.model_dir}")
            return
            
        model_dirs = [args.model_dir]
        print(f"Testing single model: {os.path.basename(args.model_dir)}")
    # Handle training directory with multiple models
    else:
        model_dirs = find_model_directories(args.training_dir)
        if not model_dirs:
            print(f"No model directories found in: {args.training_dir}")
            return
        print(f"Found {len(model_dirs)} models to test in: {args.training_dir}")
    
    # Test each model
    results = []
    for model_dir in model_dirs:
        result = run_test(model_dir, args.output_dir, args.debug)
        results.append(result)
    
    # Generate summary report
    generate_summary(results, args.output_dir)
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()
