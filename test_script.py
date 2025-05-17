from common_imports import *
from common_imports import df
from trading_env import TradingEnv
import pickle

# Load the model and normalization stats
def load_model_and_stats(model_path="final_model", norm_path=None):
    """Load the trained model and normalization stats."""
    model = MaskablePPO.load(model_path)
    env = DummyVecEnv([lambda: TradingEnv(df)])
    if norm_path is not None:
        try:
            env = VecNormalize.load(norm_path, env)
        except AssertionError:
            print("Warning: Observation space shape mismatch. Using new environment without normalization.")
            env = VecNormalize(env)
    env.training = False
    env.norm_reward = False
    return model, env

def test_trading_model(model_path="final_model", norm_path="vec_normalize.pkl", render=False):
    """Test the trained trading model with normalization and return signals.
    
    Args:
        model_path: Path to the trained MaskablePPO model
        norm_path: Path to saved VecNormalize stats
        render: Whether to print debug info
        
    Returns:
        tuple: (results, signals)
            results: dict containing results from the episode
            signals: dict with 'buy_signals' and 'sell_signals'
    """
    model, env = load_model_and_stats(model_path, norm_path)
    
    # Run one episode
    obs = env.reset()
    buy_signals = []
    sell_signals = []
    episode_reward = 0
    actions = []
    timesteps = []
    
    done = False
    step = 0
    while not done:
        action_masks = env.get_attr('action_masks')[0]()
        if render:
            print(f"\nStep {step} action masks: {action_masks}")
        
        action, _ = model.predict(obs, action_masks=action_masks)
        obs, reward, done, info = env.step(action)
        
        # Store signals
        if action == 1:  # Buy action
            buy_signals.append(step)
        elif action == 2:  # Sell action
            sell_signals.append(step)
        
        # Handle reward as float
        if isinstance(reward, np.ndarray):
            episode_reward += reward[0]
        else:
            episode_reward += reward
        actions.append(action)
        timesteps.append(step)
        step += 1
        
        if done:
            break
    
    # Print results
    print(f"\n=== Test Results for {model_path} ===")
    print(f"Reward: {float(episode_reward):.4f}")
    print(f"Buy signals: {buy_signals}")
    print(f"Sell signals: {sell_signals}")
    print(f"\nActions taken: {actions}")
    print(f"Timesteps: {timesteps}")
    
    # Save signals to CSV
    model_base_name = os.path.basename(model_path).split('.')[0]
    signals_df = pd.DataFrame({
        'model': model_base_name,
        'buy_signals': [buy_signals],
        'sell_signals': [sell_signals],
        'reward': float(episode_reward),
        'actions': [actions],
        'timesteps': [timesteps]
    })
    signals_df.to_csv(f'trade_signals_{model_base_name}.csv', index=False)
    
    print(f"\nTrade signals saved to 'trade_signals_{model_base_name}.csv'")
    
    env.close()
    
    return {
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'reward': episode_reward,
        'actions': actions,
        'timesteps': timesteps
    }

if __name__ == "__main__":
    # Test both models
    model_names = ['best_model/best_model.zip', 'final_model']
    for model_name in model_names:
        print(f"\nTesting {model_name}...")
        
        # Test the model and get results
        results = test_trading_model(
            model_path=model_name,
            render=True
        )
        
        # Print action masks for debugging
        print(f"\nAction masks during testing for {model_name}:")
        model, env = load_model_and_stats(model_name)
        obs = env.reset()[0]
        action_masks = env.env_method('action_masks')[0]
        print(f"Initial action masks: {action_masks}")
        
        # Print signals
        print(f"\nBuy signals for {model_name}: {results['buy_signals']}")
        print(f"Sell signals for {model_name}: {results['sell_signals']}")
        
        # Print action masks for first few steps
        print("\nAction masks during testing for", model_name)
        print("Initial action masks:", action_masks)
        
        # Reset environment for action mask testing
        obs = env.reset()
        action_masks = env.get_attr('action_masks')[0]()
        print("\nTesting action masks:")
        for i in range(3):
            action, _ = model.predict(obs, action_masks=action_masks)
            obs, reward, done, info = env.step(action)
            action_masks = env.get_attr('action_masks')[0]()
            print(f"Step {i+1} action masks:", action_masks)
        
        # Test a few steps to see action masks
        for i in range(3):
            action, _ = model.predict(obs, action_masks=action_masks)
            obs, reward, done, info = env.step([action])
            obs = obs[0]
            action_masks = env.env_method('action_masks')[0]
            print(f"Step {i+1} action masks: {action_masks}")
        env.close()
        
        # Print summary
        print(f"\n=== Test Results for {model_name} ===")
        print(f"Reward: {results['reward']:.2f}")
        print(f"Buy signals: {results['buy_signals']}")
        print(f"Sell signals: {results['sell_signals']}")
        print(f"\nActions taken: {results['actions']}")
        print(f"Timesteps: {results['timesteps']}")