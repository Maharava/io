#!/usr/bin/env python3
"""
Wake Word Training System Fix

This script fixes issues with model saving in the wake word detection training system:
1. Adds missing Path imports
2. Fixes model saving logic for BCEWithLogitsLoss
3. Improves error handling during model saving
4. Ensures consistent directory creation
5. Adds model recovery utilities

Usage:
    python wake_word_fix.py

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

def find_project_files():
    """Find key project files that need to be fixed"""
    # Start with the current directory as the base
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Possible locations for files
    candidates = {
        "training_panel": [
            os.path.join(base_dir, "ui", "training_panel.py"),
            os.path.join(base_dir, "io_wake_word", "ui", "training_panel.py"),
            os.path.join(str(Path.home()), ".io", "ui", "training_panel.py"),
        ],
        "training_module": [
            os.path.join(base_dir, "model", "training.py"),
            os.path.join(base_dir, "io_wake_word", "model", "training.py"),
            os.path.join(str(Path.home()), ".io", "model", "training.py"),
        ],
        "model_architecture": [
            os.path.join(base_dir, "model", "architecture.py"),
            os.path.join(base_dir, "io_wake_word", "model", "architecture.py"),
            os.path.join(str(Path.home()), ".io", "model", "architecture.py"),
        ]
    }
    
    # Find which files actually exist
    files_found = {}
    for file_type, paths in candidates.items():
        for path in paths:
            if os.path.exists(path):
                files_found[file_type] = path
                logger.info(f"Found {file_type} at: {path}")
                break
        
        if file_type not in files_found:
            logger.warning(f"Could not find {file_type}")
    
    return files_found

# ----- CODE FIXING FUNCTIONS -----

def backup_file(file_path):
    """Create a backup of a file before modifying it"""
    backup_path = file_path + ".bak"
    try:
        shutil.copy2(file_path, backup_path)
        logger.info(f"Created backup at: {backup_path}")
        return True
    except Exception as e:
        logger.warning(f"Could not create backup: {e}")
        return False

def ensure_path_import(content):
    """Ensure Path is imported in the file"""
    # Check if Path is already imported
    if "from pathlib import Path" in content:
        return content
    
    # Add Path import to the file
    import_section_match = re.search(r"import\s+.*?(?=\n\n|\nclass|\ndef)", content, re.DOTALL)
    if import_section_match:
        import_section = import_section_match.group(0)
        updated_imports = import_section + "\nfrom pathlib import Path\n"
        content = content.replace(import_section, updated_imports)
        return content
    
    # If no import section found, add it at the top after any docstring
    docstring_match = re.search(r'""".*?"""', content, re.DOTALL)
    if docstring_match:
        docstring = docstring_match.group(0)
        updated_content = content.replace(docstring, docstring + "\nfrom pathlib import Path\n")
        return updated_content
    
    # Last resort, add at the very top
    return "from pathlib import Path\n" + content

def fix_training_panel(training_panel_path):
    """Fix issues in the training_panel.py file"""
    if not training_panel_path or not os.path.exists(training_panel_path):
        logger.error("Invalid training_panel.py path")
        return False
    
    # Backup the file
    backup_file(training_panel_path)
    
    # Read the file
    with open(training_panel_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Ensure Path is imported
    content = ensure_path_import(content)
    
    # Fix the run method in TrainingThread class
    run_method_pattern = r"def run\(self\):(.*?)(?=\n    def|\n\nclass|\Z)"
    run_method_match = re.search(run_method_pattern, content, re.DOTALL)
    
    if run_method_match:
        run_method = run_method_match.group(0)
        fixed_run_method = run_method
        
        # Fix model training check
        if "model = trainer.train" in run_method and "if model is None" not in run_method:
            fixed_run_method = fixed_run_method.replace(
                "model = trainer.train(train_loader, val_loader)",
                """model = trainer.train(train_loader, val_loader)
                    
                    # Check if model is None and handle accordingly
                    if model is None:
                        self.result = {
                            "success": False,
                            "error": "Model training failed - returned None"
                        }
                        if self.progress_callback:
                            self.progress_callback("Error: Training failed to produce a valid model", -1)
                        return"""
            )
        
        # Add robust model saving with multiple fallbacks
        if "model_path = str(models_dir / self.model_name)" in fixed_run_method:
            # Make sure the path creation is correct
            fixed_run_method = fixed_run_method.replace(
                "model_path = str(models_dir / self.model_name)",
                """# Ensure models directory exists
                    models_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Create full model path using Path objects
                    model_path = str(models_dir / self.model_name)"""
            )
        
        # Improve model saving logic
        if "trainer.save_trained_model" in fixed_run_method:
            fixed_run_method = fixed_run_method.replace(
                "try:",
                """try:
                    # First verify the model is valid
                    if model is None:
                        self.result = {
                            "success": False,
                            "error": "Cannot save None model"
                        }
                        if self.progress_callback:
                            self.progress_callback("Error: Model is None", -1)
                        return"""
            )
            
            # Replace model saving with improved version
            save_pattern = r"try:(.*?)except AttributeError:"
            save_match = re.search(save_pattern, fixed_run_method, re.DOTALL)
            
            if save_match:
                old_save_block = save_match.group(0)
                new_save_block = """try:
                    # First verify the model is valid
                    if model is None:
                        self.result = {
                            "success": False,
                            "error": "Cannot save None model"
                        }
                        if self.progress_callback:
                            self.progress_callback("Error: Model is None", -1)
                        return
                        
                    # Check for state_dict
                    if not hasattr(model, 'state_dict'):
                        self.result = {
                            "success": False,
                            "error": "Model has no state_dict method"
                        }
                        if self.progress_callback:
                            self.progress_callback("Error: Model has no state_dict", -1)
                        return
                        
                    # Create directory if needed
                    Path(os.path.dirname(model_path)).mkdir(parents=True, exist_ok=True)
                    
                    # Try trainer's save method first
                    if hasattr(trainer, 'save_trained_model'):
                        success = trainer.save_trained_model(model, model_path)
                        if not success:
                            raise Exception("Trainer's save_trained_model returned False")
                    else:
                        # Fallback to direct save
                        import torch
                        torch.save(model.state_dict(), model_path)
                except AttributeError:"""
                
                fixed_run_method = fixed_run_method.replace(old_save_block, new_save_block)
        
        # Update the run method in the content
        content = content.replace(run_method, fixed_run_method)
    else:
        logger.warning("Could not find run method in training_panel.py")
    
    # Write back the modified content
    try:
        with open(training_panel_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info("Successfully fixed training_panel.py")
        return True
    except Exception as e:
        logger.error(f"Error writing to training_panel.py: {e}")
        return False

def fix_training_module(training_module_path):
    """Fix issues in the training.py module"""
    if not training_module_path or not os.path.exists(training_module_path):
        logger.error("Invalid training.py path")
        return False
    
    # Backup the file
    backup_file(training_module_path)
    
    # Read the file
    with open(training_module_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Ensure Path is imported
    content = ensure_path_import(content)
    
    # Fix save_trained_model method
    save_method_pattern = r"def save_trained_model\(self, model, path\):(.*?)(?=\n    def|\Z)"
    save_method_match = re.search(save_method_pattern, content, re.DOTALL)
    
    if save_method_match:
        save_method = save_method_match.group(0)
        new_save_method = """def save_trained_model(self, model, path):
        \"\"\"Save trained model to disk with robust error handling\"\"\"
        # Basic validation
        if model is None:
            training_logger.error("Cannot save model: model is None")
            return False
            
        # Check for state_dict
        if not hasattr(model, 'state_dict'):
            training_logger.error("Model has no state_dict method")
            return False
            
        try:
            # Ensure directory exists
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save the model
            torch.save(model.state_dict(), path)
            training_logger.info(f"Model saved to {path}")
            
            # Save model metadata for reference
            try:
                metadata_path = os.path.join(os.path.dirname(path), "model_info.txt")
                with open(metadata_path, 'w') as f:
                    f.write(f"Model trained with wake word engine\\n")
                    f.write(f"n_mfcc: {self.n_mfcc}\\n")
                    f.write(f"n_fft: {self.n_fft}\\n")
                    f.write(f"hop_length: {self.hop_length}\\n")
                    f.write(f"num_frames: {self.num_frames}\\n")
                    f.write(f"Date: {__import__('datetime').datetime.now()}\\n")
            except Exception as e:
                training_logger.warning(f"Error saving metadata (but model was saved): {e}")
            
            return True
        except Exception as e:
            training_logger.error(f"Error saving model: {e}")
            return False
    
"""
        content = content.replace(save_method, new_save_method)
    else:
        logger.warning("Could not find save_trained_model method in training.py")
    
    # Fix training end to handle BCEWithLogitsLoss
    train_end_pattern = r"(?:# Final inference test|# Final model preparation)(.*?)(?=\n    def|\Z)"
    train_end_match = re.search(train_end_pattern, content, re.DOTALL)
    
    if train_end_match:
        train_end = train_end_match.group(0)
        
        if "BCEWithLogitsLoss" not in train_end:
            # Need to add sigmoid handling for BCEWithLogitsLoss
            new_train_end = """        # Final model preparation
        if best_model is not None:
            model.load_state_dict(best_model)
            training_logger.info("Loaded best model from validation")
        
        # If we were using BCEWithLogitsLoss, create inference model with sigmoid
        if isinstance(criterion, nn.BCEWithLogitsLoss):
            training_logger.info("Creating inference model with sigmoid")
            try:
                # Create a new model with sigmoid for inference
                if self.use_simple_model:
                    inference_model = SimpleWakeWordModel(n_mfcc=self.n_mfcc, num_frames=self.num_frames)
                else:
                    inference_model = WakeWordModel(n_mfcc=self.n_mfcc, num_frames=self.num_frames)
                
                # Transfer the trained weights
                inference_model.load_state_dict(model.state_dict())
                model = inference_model
                training_logger.info("Successfully created inference model with sigmoid")
            except Exception as e:
                training_logger.error(f"Error creating inference model: {e}")
                training_logger.warning("Using training model for inference (without sigmoid)")
        
        # Final inference test
        training_logger.info("Running final inference test")
        self.test_inference(model, val_loader)
        
        return model"""
            content = content.replace(train_end, new_train_end)
    else:
        logger.warning("Could not find final model preparation in training.py")
    
    # Make sure train method returns model with proper checks
    train_return_pattern = r"return model"
    if train_return_pattern in content:
        content = content.replace(
            "return model",
            """        # Final check before returning
        if model is None:
            training_logger.error("Training failed to produce a valid model")
            return None
            
        return model"""
        )
    
    # Write back the modified content
    try:
        with open(training_module_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info("Successfully fixed training.py")
        return True
    except Exception as e:
        logger.error(f"Error writing to training.py: {e}")
        return False

def fix_model_architecture(architecture_path):
    """Fix issues in the model/architecture.py file"""
    if not architecture_path or not os.path.exists(architecture_path):
        logger.error("Invalid architecture.py path")
        return False
    
    # Backup the file
    backup_file(architecture_path)
    
    # Read the file
    with open(architecture_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Ensure Path is imported
    content = ensure_path_import(content)
    
    # Fix the load_model function to handle path properly
    load_pattern = r"def load_model\(.*?\):(.*?)(?=\n    def|\Z)"
    load_match = re.search(load_pattern, content, re.DOTALL)
    
    if load_match:
        load_function = load_match.group(0)
        
        if "Path(path).parent.mkdir" not in load_function and "if not path or not os.path.exists(path):" in load_function:
            # Fix path checking
            fixed_load = load_function.replace(
                "if not path or not os.path.exists(path):",
                """if not path:
        logger.error("Model path is None")
        return None
        
    # Convert to Path object for consistent handling
    path = Path(path) if not isinstance(path, Path) else path
    
    if not path.exists():"""
            )
            content = content.replace(load_function, fixed_load)
    else:
        logger.warning("Could not find load_model function in architecture.py")
    
    # Fix save_model function to handle Path properly
    save_pattern = r"def save_model\(.*?\):(.*?)(?=\n    def|\Z)"
    save_match = re.search(save_pattern, content, re.DOTALL)
    
    if save_match:
        save_function = save_match.group(0)
        
        if "Path(os.path.dirname(path)).mkdir" not in save_function and "os.makedirs(os.path.dirname(path)" in save_function:
            # Replace os.makedirs with Path.mkdir
            fixed_save = save_function.replace(
                "os.makedirs(os.path.dirname(path), exist_ok=True)",
                "Path(path).parent.mkdir(parents=True, exist_ok=True)"
            )
            content = content.replace(save_function, fixed_save)
    else:
        logger.warning("Could not find save_model function in architecture.py")
    
    # Write back the modified content
    try:
        with open(architecture_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info("Successfully fixed architecture.py")
        return True
    except Exception as e:
        logger.error(f"Error writing to architecture.py: {e}")
        return False

# ----- HELPER UTILITIES -----

def create_model_recovery_script():
    """Create a script to recover models from checkpoints"""
    script_content = """#!/usr/bin/env python3
\"\"\"
Wake Word Model Recovery Tool

This script helps you recover trained models that failed to save properly.
It searches for checkpoint files and creates a usable model.

Usage:
    python recover_model.py
\"\"\"
import os
import sys
import torch
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelRecovery")

def recover_model():
    \"\"\"Attempt to recover the latest trained model\"\"\"
    # Find the diagnostic directory
    diagnostics_dir = Path.home() / ".io" / "training_diagnostics"
    if not diagnostics_dir.exists():
        print(f"❌ Diagnostic directory not found: {diagnostics_dir}")
        return False
    
    # Look for model checkpoints in priority order
    checkpoint_files = [
        "model_best.pth",
        "model_epoch_80.pth",
        "model_epoch_60.pth", 
        "model_epoch_40.pth",
        "model_epoch_20.pth",
        "model_epoch_10.pth",
    ]
    
    # Find the first existing checkpoint
    checkpoint_path = None
    for filename in checkpoint_files:
        path = diagnostics_dir / filename
        if path.exists():
            checkpoint_path = path
            print(f"Found checkpoint: {path}")
            break
    
    if checkpoint_path is None:
        print("❌ No checkpoint files found in diagnostic directory")
        return False
    
    # Create models directory
    models_dir = Path.home() / ".io" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the recovery destination
    recovered_path = models_dir / "recovered_model.pth"
    
    # Try to load and save the checkpoint
    try:
        print(f"Loading checkpoint from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        
        print(f"Saving recovered model to {recovered_path}...")
        torch.save(state_dict, recovered_path)
        
        print(f"✅ SUCCESS: Model recovered to {recovered_path}")
        print("You can now use this model in your wake word detector")
        
        # Create a simple note about the recovery
        note_path = models_dir / "recovered_model_info.txt"
        with open(note_path, 'w') as f:
            f.write(f"Model recovered from checkpoint: {checkpoint_path.name}\\n")
            f.write(f"Recovery date: {__import__('datetime').datetime.now()}\\n")
        
        return True
    except Exception as e:
        print(f"❌ Error recovering model: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print(" WAKE WORD MODEL RECOVERY TOOL ")
    print("=" * 60)
    print("This tool will try to recover trained models from checkpoints")
    
    recover_model()
"""
    
    # Save the recovery script
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recover_model.py")
    
    try:
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        logger.info(f"Created model recovery script at: {script_path}")
        return True
    except Exception as e:
        logger.error(f"Error creating recovery script: {e}")
        return False

def clean_unnecessary_files():
    """Identify and suggest unnecessary files for removal"""
    unnecessary_files = [
        "model_save_fix.py",        # Superseded by this script
        "training_diagnostic.py",   # No longer needed once fixes are applied
        "complete_fix.py",          # Superseded by this script
    ]
    
    # Find the actual files that exist
    found_files = []
    for filename in unnecessary_files:
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        if os.path.exists(file_path):
            found_files.append(file_path)
    
    # Find backup files that can be removed after testing
    backup_files = []
    for root, dirs, files in os.walk(os.path.dirname(os.path.abspath(__file__))):
        for filename in files:
            if filename.endswith(".bak") and filename != "wake_word_fix.py.bak":
                backup_files.append(os.path.join(root, filename))
    
    return found_files, backup_files

# ----- MAIN FUNCTION -----

def main():
    """Main function to fix all issues"""
    logger.info("Wake Word Training System Fix")
    logger.info("Fixing model saving issues...")
    
    # Find project files
    project_files = find_project_files()
    
    # Apply fixes
    results = {
        "training_panel": False,
        "training_module": False,
        "model_architecture": False,
        "recovery_script": False
    }
    
    # Fix training_panel.py
    if "training_panel" in project_files:
        results["training_panel"] = fix_training_panel(project_files["training_panel"])
    
    # Fix training.py
    if "training_module" in project_files:
        results["training_module"] = fix_training_module(project_files["training_module"])
    
    # Fix architecture.py
    if "model_architecture" in project_files:
        results["model_architecture"] = fix_model_architecture(project_files["model_architecture"])
    
    # Create recovery script
    results["recovery_script"] = create_model_recovery_script()
    
    # Identify unnecessary files
    unnecessary_files, backup_files = clean_unnecessary_files()
    
    # Print summary
    print("\n" + "=" * 60)
    print(" WAKE WORD TRAINING SYSTEM FIX - SUMMARY ")
    print("=" * 60)
    
    if all(results.values()):
        print("\n✅ All fixes were successfully applied!")
    else:
        print("\n⚠️ Some fixes could not be applied:")
        
        for fix_type, success in results.items():
            status = "✅" if success else "❌"
            print(f"  {status} {fix_type}")
    
    print("\nChanges made:")
    print("  1. Fixed Path imports and handling")
    print("  2. Fixed model saving logic with BCEWithLogitsLoss")
    print("  3. Improved error handling for model saving")
    print("  4. Added model recovery script")
    
    if unnecessary_files:
        print("\nUnnecessary files that can be deleted:")
        for file_path in unnecessary_files:
            print(f"  - {os.path.basename(file_path)}")
    
    if backup_files:
        print("\nBackup files (can be deleted after testing):")
        for file_path in backup_files:
            print(f"  - {os.path.basename(file_path)}")
    
    print("\nNext steps:")
    print("  1. Try training your model again")
    print("  2. If needed, use the recovery script: python recover_model.py")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()