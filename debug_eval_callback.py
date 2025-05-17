from stable_baselines3.common.callbacks import EvalCallback

class DebugEvalCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Eval callback initialized!")
        
    def _on_step(self) -> bool:
        return super()._on_step()
    
    def _on_eval_end(self) -> None:
        if self.last_mean_reward is not None:
            print(f"\nEvaluation results:")
            print(f"Mean reward: {self.last_mean_reward:.4f}")
            print(f"Std reward: {self.last_std_reward:.4f}")
            print(f"Episode length: {self.last_mean_length:.2f}")
            print(f"Is best model: {self.is_best}")
        return super()._on_eval_end()
