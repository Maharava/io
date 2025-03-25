"""
Quick fix for the model saving issue in the wake word training system.

This script patches the trainer class to properly handle model saving
after training with BCEWithLogitsLoss.
"""
import os
import sys
import importlib
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelSaveFix")

def find_ui_directory():
    """Find the UI directory"""
    possible_locations = [
        "ui",
        "io_wake_word/ui",
        os.path.join(Path.home(), ".io", "ui"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui")
    ]
    
    for location in possible_locations:
        if os.path.isdir(location):
            return location
    
    return None

def find_training_module():
    """Find the training.py module"""
    possible_locations = [
        "model/training.py",
        "io_wake_word/model/training.py",
        os.path.join(Path.home(), ".io", "model", "training.py"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "training.py")
    ]
    
    return [loc for loc in possible_locations if os.path.exists(loc)]

def fix_training_panel():
    """Fix the training_panel.py to handle None models"""
    ui_dir = find_ui_directory()
    if not ui_dir:
        logger.error("Could not find UI directory")
        return False
    
    training_panel_path = os.path.join(ui_dir, "training_panel.py")
    if not os.path.exists(training_panel_path):
        logger.error(f"Could not find training_panel.py in {ui_dir}")
        return False
    
    logger.info(f"Found training_panel.py at: {training_panel_path}")
    
    # Read the file
    with open(training_panel_path, 'r') as f:
        content = f.read()
    
    # Find the Run method in TrainingThread
    if "class TrainingThread" in content and "def run(self):" in content:
        # Add fix for the NoneType error
        modified_content = content.replace(
            "model = trainer.train(train_loader, val_loader)",
            """model = trainer.train(train_loader, val_loader)
                    
                    # Check if model is None and handle accordingly
                    if model is None:
                        self.result = {
                            "success": False,
                            "error": "Model training failed - returned None"
                        }
                        return"""
        )
        
        # Also make sure we check model before saving
        modified_content = modified_content.replace(
            "trainer.save_trained_model(model, model_path)",
            """if hasattr(model, 'state_dict'):
                    trainer.save_trained_model(model, model_path)
                else:
                    # Direct save fallback
                    import torch
                    from pathlib import Path
                    
                    # Ensure directory exists
                    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
                    
                    # Save the model
                    torch.save(model.state_dict(), model_path)"""
        )
        
        # Write back the modified content
        if content != modified_content:
            with open(training_panel_path, 'w') as f:
                f.write(modified_content)
            logger.info(f"Successfully updated {training_panel_path}")
            return True
        else:
            logger.info("No changes needed to training_panel.py")
            return True
    
    logger.error("Could not find required code sections in training_panel.py")
    return False

def fix_training_module():
    """Fix the model/training.py module"""
    module_paths = find_training_module()
    if not module_paths:
        logger.error("Could not find training.py module")
        return False
    
    for training_path in module_paths:
        if not os.path.exists(training_path):
            continue
        
        logger.info(f"Found training.py at: {training_path}")
        
        # Read the file
        with open(training_path, 'r') as f:
            content = f.read()
        
        # Fix the final model preparation section
        modified_content = content
        
        # Find and fix the section where we restore the sigmoid
        if "# Final model preparation" in content:
            # Already has the section, update it to ensure it works properly
            section_start = content.find("# Final model preparation")
            section_end = content.find("# Final inference test:", section_start)
            
            if section_start > 0 and section_end > 0:
                old_section = content[section_start:section_end]
                new_section = """        # Final model preparation
        if best_model is not None:
            model.load_state_dict(best_model)
            training_logger.info("Loaded best model for final evaluation")
        else:
            training_logger.warning("No best model found, using final model state")
        
        # If we were using BCEWithLogitsLoss, we need to add the sigmoid back for inference
        if isinstance(criterion, nn.BCEWithLogitsLoss):
            training_logger.info("Creating inference model with sigmoid for deployment")
            try:
                # Create a new model with sigmoid for inference
                if self.use_simple_model:
                    inference_model = SimpleWakeWordModel(n_mfcc=self.n_mfcc, num_frames=self.num_frames)
                else:
                    inference_model = WakeWordModel(n_mfcc=self.n_mfcc, num_frames=self.num_frames)
                
                # Copy trained weights
                inference_model.load_state_dict(model.state_dict())
                model = inference_model
            except Exception as e:
                training_logger.error(f"Error creating inference model: {e}")
                # Continue with the existing model as fallback
                training_logger.warning("Using training model for inference (without sigmoid)")
                pass
            
        # Final inference test
        training_logger.info("Final inference test:")
        self.test_inference(model, val_loader)

"""
                modified_content = content.replace(old_section, new_section)
        else:
            # Doesn't have the section yet, we need to find where to add it
            logger.warning("Could not find model preparation section to fix")
        
        # Update save_trained_model to handle None models
        if "def save_trained_model(self, model, path):" in content:
            save_method_start = content.find("def save_trained_model(self, model, path):")
            save_method_end = content.find("def ", save_method_start + 1)
            if save_method_start > 0:
                if save_method_end < 0:  # If it's the last method
                    save_method_end = len(content)
                
                old_method = content[save_method_start:save_method_end]
                new_method = """def save_trained_model(self, model, path):
        \"\"\"Save trained model to disk with proper error handling\"\"\"
        if model is None:
            training_logger.error("Cannot save model: model is None")
            return False
            
        # Ensure the directory exists
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save the model
            torch.save(model.state_dict(), path)
            training_logger.info(f"Model saved to {path}")
            
            # Create a README with training info
            readme_path = os.path.join(os.path.dirname(path), "training_info.txt")
            with open(readme_path, 'w') as f:
                f.write(f"Model trained with enhanced training pipeline\\n")
                f.write(f"n_mfcc: {self.n_mfcc}\\n")
                f.write(f"n_fft: {self.n_fft}\\n")
                f.write(f"hop_length: {self.hop_length}\\n")
                f.write(f"num_frames: {self.num_frames}\\n")
                f.write(f"model_type: {'simple' if self.use_simple_model else 'standard'}\\n")
                f.write(f"Training diagnostic files in: {self.diagnostic_dir}\\n")
            
            return True
        except Exception as e:
            training_logger.error(f"Error saving model: {e}")
            return False
    
"""
                modified_content = modified_content.replace(old_method, new_method)
        
        # Update the train method's return value check
        train_return_fix = "return model"
        if train_return_fix in modified_content:
            # Make sure we check the model before returning
            modified_content = modified_content.replace(
                train_return_fix,
                """        # Ensure we return a valid model
        if model is None:
            training_logger.error("Training failed to produce a valid model")
            return None
            
        return model"""
            )
        
        # Write back the modified content
        if content != modified_content:
            # Make a backup
            backup_path = training_path + ".bak"
            with open(backup_path, 'w') as f:
                f.write(content)
            logger.info(f"Created backup at {backup_path}")
            
            # Write the fixed content
            with open(training_path, 'w') as f:
                f.write(modified_content)
            logger.info(f"Successfully updated {training_path}")
            return True
        else:
            logger.info("No changes needed to training.py")
            return True
    
    return False

def create_direct_save_helper():
    """Create a direct save helper function in the current directory"""
    content = """
import torch
import os
from pathlib import Path

def save_wake_word_model(model, path):
    \"\"\"
    Save a wake word model directly, bypassing the trainer.
    
    Args:
        model: The PyTorch model to save
        path: Where to save the model
    
    Returns:
        bool: Success status
    \"\"\"
    try:
        # Make sure we have a valid model
        if model is None:
            print("Error: Cannot save None model")
            return False
            
        # Ensure the directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save the model
        torch.save(model.state_dict(), path)
        print(f"Model saved to {path}")
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False
"""
    
    # Save to current directory
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "save_helper.py")
    with open(save_path, 'w') as f:
        f.write(content)
    
    logger.info(f"Created direct save helper at {save_path}")
    return True

if __name__ == "__main__":
    logger.info("Starting model save fix script...")
    
    # Apply fixes
    training_module_fixed = fix_training_module()
    training_panel_fixed = fix_training_panel()
    
    # Always create the save helper as a fallback
    create_direct_save_helper()
    
    if training_module_fixed and training_panel_fixed:
        logger.info("Successfully applied all fixes!")
        logger.info("\nTry training your model again now.")
    else:
        logger.warning("\nCould not apply all fixes automatically.")
        logger.info("If you still encounter issues, you can use the save_helper.py directly:")
        logger.info("import save_helper")
        logger.info("save_helper.save_wake_word_model(model, path)")