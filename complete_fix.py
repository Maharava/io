#!/usr/bin/env python3
"""
Complete Fix for Wake Word Training System

This script performs several fixes to resolve model saving issues:
1. Patches the training_panel.py to handle None models and add proper imports
2. Patches training.py to fix model creation and saving
3. Creates a standalone save_helper.py as a fallback option
4. Adds a model rescue utility to extract trained models from checkpoints

Usage:
    python complete_fix.py

Author: Claude
Date: March 25, 2025
"""
import os
import sys
import re
import logging
import shutil
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("WakeWordFix")

# ----- PATH FINDING FUNCTIONS -----

def find_ui_directory():
    """Find the UI directory containing training_panel.py"""
    possible_locations = [
        "ui",
        "io_wake_word/ui",
        os.path.join(str(Path.home()), ".io", "ui"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui")
    ]
    
    for location in possible_locations:
        if os.path.isdir(location):
            if os.path.exists(os.path.join(location, "training_panel.py")):
                return location
    
    return None

def find_training_module():
    """Find the training.py module location"""
    possible_locations = [
        "model/training.py",
        "io_wake_word/model/training.py",
        os.path.join(str(Path.home()), ".io", "model", "training.py"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "training.py")
    ]
    
    return [loc for loc in possible_locations if os.path.exists(loc)]

# ----- CODE MODIFICATION FUNCTIONS -----

def fix_training_panel():
    """Fix the training_panel.py to properly handle model saving"""
    ui_dir = find_ui_directory()
    if not ui_dir:
        logger.error("Could not find UI directory with training_panel.py")
        return False
    
    training_panel_path = os.path.join(ui_dir, "training_panel.py")
    if not os.path.exists(training_panel_path):
        logger.error(f"Could not find training_panel.py in {ui_dir}")
        return False
    
    logger.info(f"Found training_panel.py at: {training_panel_path}")
    
    # Make a backup
    backup_path = training_panel_path + ".bak"
    try:
        shutil.copy2(training_panel_path, backup_path)
        logger.info(f"Created backup at {backup_path}")
    except Exception as e:
        logger.warning(f"Could not create backup: {e}")
    
    # Read the file
    with open(training_panel_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add Path import if not present
    import_section_match = re.search(r"import\s+.*?(?=\n\n|\nclass|\ndef)", content, re.DOTALL)
    if import_section_match:
        import_section = import_section_match.group(0)
        if "from pathlib import Path" not in import_section:
            updated_imports = import_section + "\nfrom pathlib import Path\n"
            content = content.replace(import_section, updated_imports)
            logger.info("Added Path import to training_panel.py")
    
    # Fix the run method
    run_method_pattern = r"def run\(self\):(.*?)(?=\n    def|\n\nclass|\Z)"
    run_method_match = re.search(run_method_pattern, content, re.DOTALL)
    
    if run_method_match:
        run_method = run_method_match.group(0)
        
        # Replace model training line to check for None result
        if "model = trainer.train(train_loader, val_loader)" in run_method:
            fixed_run_method = run_method.replace(
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
        else:
            fixed_run_method = run_method
            logger.warning("Could not find model training line in run method")
        
        # Replace model saving line with safer version
        if "trainer.save_trained_model(model, model_path)" in fixed_run_method:
            fixed_run_method = fixed_run_method.replace(
                "trainer.save_trained_model(model, model_path)",
                """try:
                    # First check if model has state_dict
                    if hasattr(model, 'state_dict'):
                        # Try the standard method first
                        if hasattr(trainer, 'save_trained_model'):
                            trainer.save_trained_model(model, model_path)
                        else:
                            # Direct save fallback
                            import torch
                            
                            # Ensure directory exists
                            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
                            
                            # Save the model
                            torch.save(model.state_dict(), model_path)
                            print(f"Model saved directly to {model_path}")
                    else:
                        self.result = {
                            "success": False, 
                            "error": "Model has no state_dict, cannot save"
                        }
                        return
                except Exception as e:
                    self.result = {
                        "success": False,
                        "error": f"Error saving model: {str(e)}"
                    }
                    return"""
            )
        else:
            logger.warning("Could not find model saving line in run method")
        
        # Replace the run method in the content
        content = content.replace(run_method, fixed_run_method)
        logger.info("Updated run method in training_panel.py")
    else:
        logger.error("Could not find run method in training_panel.py")
        return False
    
    # Write back the modified content
    try:
        with open(training_panel_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info("Successfully updated training_panel.py")
        return True
    except Exception as e:
        logger.error(f"Error writing to file: {e}")
        return False

def fix_training_module():
    """Fix the model/training.py module to properly handle model creation and saving"""
    module_paths = find_training_module()
    if not module_paths:
        logger.error("Could not find training.py module")
        return False
    
    for training_path in module_paths:
        if not os.path.exists(training_path):
            continue
        
        logger.info(f"Found training.py at: {training_path}")
        
        # Make a backup
        backup_path = training_path + ".bak"
        try:
            shutil.copy2(training_path, backup_path)
            logger.info(f"Created backup at {backup_path}")
        except Exception as e:
            logger.warning(f"Could not create backup: {e}")
        
        # Read the file
        with open(training_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Make sure Path is imported
        if "from pathlib import Path" not in content:
            if "import os" in content:
                content = content.replace("import os", "import os\nfrom pathlib import Path")
                logger.info("Added Path import to training.py")
        
        # Fix the save_trained_model method
        save_method_pattern = r"def save_trained_model\(self, model, path\):(.*?)(?=\n    def|\Z)"
        save_method_match = re.search(save_method_pattern, content, re.DOTALL)
        
        if save_method_match:
            save_method = save_method_match.group(0)
            new_save_method = """def save_trained_model(self, model, path):
        \"\"\"Save trained model to disk with error handling\"\"\"
        if model is None:
            training_logger.error("Cannot save model: model is None")
            return False
            
        # Check if model has state_dict
        if not hasattr(model, 'state_dict'):
            training_logger.error("Model has no state_dict method")
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
            content = content.replace(save_method, new_save_method)
            logger.info("Updated save_trained_model method in training.py")
        else:
            logger.warning("Could not find save_trained_model method in training.py")
        
        # Find the train method's return statement and ensure it checks for None
        train_method_pattern = r"def train\(self.*?\):(.*?return model)"
        train_method_match = re.search(train_method_pattern, content, re.DOTALL)
        
        if train_method_match:
            train_method = train_method_match.group(0)
            if "return model" in train_method:
                modified_train_method = train_method.replace(
                    "return model",
                    """        # Ensure we have a valid model to return
        if model is None:
            training_logger.error("Training failed to produce a valid model")
            return None
        
        return model"""
                )
                content = content.replace(train_method, modified_train_method)
                logger.info("Updated train method return value check in training.py")
        else:
            logger.warning("Could not find train method return statement in training.py")
        
        # Fix the final model preparation section
        model_prep_pattern = r"# Final model preparation(.*?)# Final inference test"
        model_prep_match = re.search(model_prep_pattern, content, re.DOTALL)
        
        if model_prep_match:
            old_section = model_prep_match.group(0)
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
                training_logger.warning("Using training model for inference (without sigmoid)")
        
        # Final inference test"""
            
            # Be careful to leave intact the "Final inference test" text
            if "# Final inference test" in old_section:
                new_section_with_test = new_section
                content = content.replace(old_section, new_section_with_test)
                logger.info("Updated final model preparation section in training.py")
            else:
                logger.warning("Could not safely replace model preparation section")
        else:
            logger.warning("Could not find model preparation section in training.py")
        
        # Write back the modified content
        try:
            with open(training_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Successfully updated {training_path}")
            return True
        except Exception as e:
            logger.error(f"Error writing to file: {e}")
            return False
    
    return False

# ----- CREATE HELPER UTILITIES -----

def create_save_helper():
    """Create a standalone save_helper.py file"""
    helper_content = """#!/usr/bin/env python3
\"\"\"
Save Helper for Wake Word Models

This module provides standalone functions to save and load wake word models
without depending on the trainer class.
\"\"\"
import os
import torch
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WakeWordSaveHelper")

def save_model(model, path, metadata=None):
    \"\"\"
    Save a wake word model directly to disk
    
    Args:
        model: The PyTorch model to save
        path: Path where to save the model
        metadata: Optional dictionary with model metadata
        
    Returns:
        bool: True if successful, False otherwise
    \"\"\"
    try:
        # Validate model
        if model is None:
            logger.error("Cannot save None model")
            return False
        
        # Check for state_dict
        if not hasattr(model, 'state_dict'):
            logger.error("Model has no state_dict method")
            return False
        
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save the model
        torch.save(model.state_dict(), path)
        logger.info(f"Model saved to {path}")
        
        # Save metadata if provided
        if metadata:
            metadata_path = os.path.join(os.path.dirname(path), "model_info.txt")
            with open(metadata_path, 'w') as f:
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\\n")
            logger.info(f"Metadata saved to {metadata_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return False

def load_model(path, model_class=None, **kwargs):
    \"\"\"
    Load a wake word model from disk
    
    Args:
        path: Path to the saved model
        model_class: Model class to use (will try to import if None)
        **kwargs: Parameters to pass to model constructor
        
    Returns:
        The loaded model or None if failed
    \"\"\"
    try:
        # Check if file exists
        if not os.path.exists(path):
            logger.error(f"Model file not found: {path}")
            return None
        
        # Get model class if not provided
        if model_class is None:
            try:
                # Try to import from local modules
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                
                # Try different import paths
                try:
                    from model.architecture import WakeWordModel
                    model_class = WakeWordModel
                except ImportError:
                    try:
                        from io_wake_word.model.architecture import WakeWordModel
                        model_class = WakeWordModel
                    except ImportError:
                        logger.error("Could not import WakeWordModel")
                        return None
            except ImportError as e:
                logger.error(f"Error importing model classes: {e}")
                return None
        
        # Create model instance
        model = model_class(**kwargs)
        
        # Load the state dict
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        
        # Set to evaluation mode
        model.eval()
        
        logger.info(f"Model loaded successfully from {path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

def rescue_model():
    \"\"\"
    Try to rescue the latest trained model from diagnostic checkpoint
    
    Returns:
        str: Path to the rescued model or None if failed
    \"\"\"
    try:
        # Find the diagnostic directory
        diagnostics_dir = Path.home() / ".io" / "training_diagnostics"
        if not diagnostics_dir.exists():
            logger.error("Diagnostics directory not found")
            return None
        
        # Look for model checkpoints in priority order
        checkpoint_files = [
            "model_best.pth",
            "model_epoch_80.pth",
            "model_epoch_60.pth", 
            "model_epoch_40.pth",
            "model_epoch_20.pth"
        ]
        
        checkpoint_path = None
        for filename in checkpoint_files:
            path = diagnostics_dir / filename
            if path.exists():
                checkpoint_path = path
                logger.info(f"Found checkpoint: {path}")
                break
        
        if checkpoint_path is None:
            logger.error("No checkpoint files found")
            return None
        
        # Create models directory
        models_dir = Path.home() / ".io" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Rescued model path
        rescued_path = models_dir / "rescued_model.pth"
        
        # Load checkpoint and save to rescued path
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        torch.save(state_dict, rescued_path)
        
        logger.info(f"Model rescued to {rescued_path}")
        return str(rescued_path)
    except Exception as e:
        logger.error(f"Error rescuing model: {e}")
        return None

if __name__ == "__main__":
    print("Wake Word Model Save Helper")
    print("This module provides functions to save and load wake word models")
    print("Example usage:")
    print("  from save_helper import save_model, load_model, rescue_model")
    print("  save_model(model, 'path/to/model.pth')")
    print("  model = load_model('path/to/model.pth')")
    print("  rescued_path = rescue_model()")
"""
    
    # Save the helper file
    helper_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "save_helper.py")
    try:
        with open(helper_path, 'w', encoding='utf-8') as f:
            f.write(helper_content)
        logger.info(f"Created save helper at {helper_path}")
        return True
    except Exception as e:
        logger.error(f"Error creating save helper: {e}")
        return False

def create_model_rescue_script():
    """Create a standalone model rescue script"""
    rescue_content = """#!/usr/bin/env python3
\"\"\"
Wake Word Model Rescue Utility

This script attempts to rescue trained wake word models from diagnostic checkpoints.
Use this if your model training completed but the model failed to save.
\"\"\"
import os
import sys
import torch
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelRescue")

def rescue_model():
    \"\"\"Attempt to rescue the latest trained model\"\"\"
    try:
        # Find the diagnostic directory
        diagnostics_dir = Path.home() / ".io" / "training_diagnostics"
        if not diagnostics_dir.exists():
            logger.error(f"Diagnostics directory not found: {diagnostics_dir}")
            return False
        
        # Look for model checkpoints in priority order
        checkpoint_files = [
            "model_best.pth",
            "model_epoch_80.pth",
            "model_epoch_60.pth", 
            "model_epoch_40.pth",
            "model_epoch_20.pth"
        ]
        
        checkpoint_path = None
        for filename in checkpoint_files:
            path = diagnostics_dir / filename
            if path.exists():
                checkpoint_path = path
                logger.info(f"Found checkpoint: {path}")
                break
        
        if checkpoint_path is None:
            logger.error("No checkpoint files found")
            return False
        
        # Create models directory
        models_dir = Path.home() / ".io" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Rescued model path
        rescued_path = models_dir / "rescued_model.pth"
        
        # Load checkpoint and save to rescued path
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        torch.save(state_dict, rescued_path)
        
        logger.info(f"Model rescued to {rescued_path}")
        print(f"\\nSUCCESS: Model rescued to {rescued_path}")
        print("You can now use this model in your wake word detector")
        
        return True
    except Exception as e:
        logger.error(f"Error rescuing model: {e}")
        print(f"\\nERROR: Could not rescue model: {e}")
        return False

if __name__ == "__main__":
    print("Wake Word Model Rescue Utility")
    print("Attempting to rescue your trained model from checkpoints...")
    rescue_model()
"""
    
    # Save the rescue script
    rescue_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rescue_model.py")
    try:
        with open(rescue_path, 'w', encoding='utf-8') as f:
            f.write(rescue_content)
        logger.info(f"Created model rescue script at {rescue_path}")
        return True
    except Exception as e:
        logger.error(f"Error creating rescue script: {e}")
        return False

# ----- MAIN FUNCTION -----

def main():
    """Main entry point"""
    logger.info("Wake Word Training System Fix")
    logger.info("This script will apply fixes to resolve model saving issues")
    
    # Fix training_panel.py
    panel_fixed = fix_training_panel()
    if panel_fixed:
        logger.info("✓ Successfully fixed training_panel.py")
    else:
        logger.warning("✗ Could not fix training_panel.py completely")
    
    # Fix training.py
    module_fixed = fix_training_module()
    if module_fixed:
        logger.info("✓ Successfully fixed training.py")
    else:
        logger.warning("✗ Could not fix training.py completely")
    
    # Create helper utilities
    helper_created = create_save_helper()
    rescue_created = create_model_rescue_script()
    
    # Summary
    print("\n" + "=" * 60)
    print(" WAKE WORD TRAINING SYSTEM FIX - SUMMARY ")
    print("=" * 60)
    
    if panel_fixed and module_fixed:
        print("\n✓ All fixes were successfully applied!")
        print("\nYou can now try training your model again.")
    else:
        print("\n⚠ Some fixes could not be applied automatically.")
        print("\nAlternative solutions have been created:")
    
    if helper_created:
        print("\n1. Use save_helper.py for direct model saving:")
        print("   from save_helper import save_model")
        print("   save_model(model, 'path/to/model.pth')")
    
    if rescue_created:
        print("\n2. Use rescue_model.py to recover a trained model:")
        print("   python rescue_model.py")
    
    print("\nIf you've already trained a model that failed to save, run:")
    print("python rescue_model.py")
    
    print("\n" + "=" * 60 + "\n")

if __name__ == "__main__":
    main()