# RL Trading Model

A Reinforcement Learning (RL) based trading system using Proximal Policy Optimization (PPO) from Stable Baselines 3, designed for intraday trading on stocks that have gapped up 50%.

## Features

- **Trading Environment**: Custom Gym environment with realistic market simulation
- **Action Space**: HOLD (0), BUY (1), SELL (2)
- **Observation Space**: 13 features including OHLCV, position info, and time features
- **Reward Function**: Custom reward system with position-based rewards and penalties
- **Visualization**: Built-in tools for visualizing trading signals and performance
- **Parallel Training**: Support for multiple parallel environments

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/RLmodel.git
   cd RLmodel
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training a New Model
```bash
python training_script.py [options]
```

### Testing a Trained Model
```bash
python test_script.py [options]
```

### Visualizing Results
```bash
python plot_signals.py [options]
```

### Examples

#### Training Example
```bash
# Train a model with default settings
python training_script.py

# Train with custom parameters
python training_script.py --max_steps 200 --total_timesteps 500000 --num_cpu 8 --debug
```

#### Testing Example
```bash
# Test a trained model
python test_script.py --input training_output/A0001_output_100_bars_500000_timesteps

# Test with custom output directory
python test_script.py --input training_output/A0002_output_200_bars_1000000_timesteps --output_dir my_test_results
```

#### Visualization Example
```bash
# Plot signals from test results
python plot_signals.py --input training_output/A0001_output_100_bars_500000_timesteps/test_results/trade_signals_*.csv
```

## Project Structure

```
RLmodel/
├── common_imports.py           # Shared imports and utilities
├── trading_env.py             # Trading environment (gym.Env implementation)
├── training_script.py         # Script to train new models
├── test_script.py             # Script to test trained models
├── plot_signals.py            # Visualization of trading signals
├── debug_logs/                # Debug logs from test runs
└── training_output/           # Output from training runs
    └── A[ID]_output_*/        # Training run directories
        ├── best_model_*.zip   # Best model checkpoint
        ├── final_model_*.zip  # Final trained model
        └── test_results/      # Test results
```

## Configuration

Customize the trading environment and training parameters in the respective script files. Key configuration options include:

- `max_steps`: Number of steps per episode
- `total_timesteps`: Total training timesteps
- `num_cpu`: Number of parallel environments
- Reward function parameters in `trading_env.py`

## License

This project is proprietary and private. All rights reserved.
