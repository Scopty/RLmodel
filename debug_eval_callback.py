import os
import shutil
from stable_baselines3.common.callbacks import EvalCallback

class DebugEvalCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        # Store the original best_model_save_path
        original_path = kwargs.get('best_model_save_path', None)
        
        # Create directory for the best model if it doesn't exist
        if original_path:
            # Ensure the path has a .zip extension
            if not original_path.endswith('.zip'):
                original_path += '.zip'
                kwargs['best_model_save_path'] = original_path
                
            os.makedirs(os.path.dirname(original_path), exist_ok=True)
            # Store the full path we want to save to
            self.original_best_model_path = original_path
            self.best_model_file_name = os.path.basename(original_path)
            # Pass just the directory to the parent
            kwargs['best_model_save_path'] = os.path.dirname(original_path)
        else:
            self.original_best_model_path = None
            
        super().__init__(*args, **kwargs)
        print(f"Eval callback initialized! Best model will be saved to: {self.original_best_model_path}")
        
    def _on_step(self) -> bool:
        # Store the current best_mean_reward before calling parent's _on_step
        prev_best = self.best_mean_reward
        
        # Call parent's _on_step which handles evaluation
        result = super()._on_step()
        
        # Check if we just found a new best model
        if hasattr(self, 'best_mean_reward') and self.best_mean_reward > prev_best:
            print(f"\n=== New Best Model Found! ===")
            print(f"New best mean reward: {self.best_mean_reward}")
            
            if self.original_best_model_path:
                # The parent saved to best_model.zip in the directory
                temp_zip_path = os.path.join(os.path.dirname(self.original_best_model_path), 'best_model.zip')
                new_path = self.original_best_model_path
                
                if os.path.exists(temp_zip_path):
                    print(f"Temporary best model saved to: {temp_zip_path}")
                    print(f"Will move to: {new_path}")
                    
                    try:
                        # Ensure the target directory exists
                        os.makedirs(os.path.dirname(new_path), exist_ok=True)
                        
                        # Remove existing file if it exists
                        if os.path.exists(new_path):
                            try:
                                if os.path.isdir(new_path):
                                    shutil.rmtree(new_path)
                                else:
                                    os.remove(new_path)
                            except Exception as e:
                                print(f"Warning: Could not remove existing file: {e}")
                        
                        # Move the file to the desired name
                        shutil.move(temp_zip_path, new_path)
                        print(f"Successfully moved best model to: {new_path}")
                    except Exception as e:
                        print(f"Error handling best model: {e}")
        
        return result
