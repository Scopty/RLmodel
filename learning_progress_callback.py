from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os

class LearningProgressCallback(BaseCallback):
    def __init__(self, verbose=0, save_path='.'):
        super().__init__(verbose)
        self.best_reward = -np.inf
        self.episode_rewards = []
        self.episode_count = 0
        self.last_stats_print = 0
        self.save_path = save_path
    
    def _on_step(self) -> bool:
        if self.n_calls % 100 == 0:
            print(f"\nTraining progress at step {self.num_timesteps}:")
            print(f"Learning rate: {self.model.learning_rate:.6f}")
            
            # Print entropy if available
            if hasattr(self.model, 'policy'):
                try:
                    entropy = self.model.policy.action_dist.entropy().mean().item()
                    print(f"Entropy: {entropy:.4f}")
                except:
                    print("Entropy not available")
            
            # Print recent episode stats if we have any episodes
            if len(self.episode_rewards) > 0:
                recent_rewards = self.episode_rewards[-10:]
                print(f"\nRecent (last 10) episodes:")
                print(f"Mean reward: {np.mean(recent_rewards):.4f}")
                print(f"Std reward: {np.std(recent_rewards):.4f}")
                print(f"Min reward: {np.min(recent_rewards):.4f}")
                print(f"Max reward: {np.max(recent_rewards):.4f}")
            
        return True
    
    def _on_rollout_end(self) -> None:
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            episode_reward = ep_info.get('r', None)
            if episode_reward is not None:
                self.episode_rewards.append(episode_reward)
                self.episode_count += 1
                
                if episode_reward > self.best_reward:
                    self.best_reward = episode_reward
                    print(f"\nNew best reward: {self.best_reward:.4f} at episode {self.episode_count}")
                    # Save the model
                    model_path = os.path.join(self.save_path, f"best_model_ep{self.episode_count}_rew{self.best_reward:.4f}")
                    self.model.save(model_path)
                    print(f"Saved model to {model_path}")
                
                # Print episode stats every 50 episodes
                if self.episode_count % 50 == 0 and self.episode_count != self.last_stats_print:
                    print(f"\nEpisode stats after {self.episode_count} episodes:")
                    print(f"Mean reward: {np.mean(self.episode_rewards):.4f}")
                    print(f"Std reward: {np.std(self.episode_rewards):.4f}")
                    print(f"Min reward: {np.min(self.episode_rewards):.4f}")
                    print(f"Max reward: {np.max(self.episode_rewards):.4f}")
                    self.last_stats_print = self.episode_count
        
        return True
