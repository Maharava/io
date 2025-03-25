#!/usr/bin/env python3
"""
Wake Word Model Architecture Fix

This script fixes the architecture mismatch issue by modifying the load_model function 
to automatically detect which model architecture to use based on the saved model's keys.

Usage:
    python fix_architecture.py
"""
import os
import sys
import logging
import shutil
from pathlib import Path
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ArchitectureFix")

def find_architecture_module():
    """Find the model/architecture.py file"""
    # Start with the current directory as the base
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Possible locations
    candidates = [
        os.path.join(base_dir, "model", "architecture.py"),
        os.path.join(base_dir, "io_wake_word", "model", "architecture.py"),
        os.path.join(str(Path.home()), ".io", "model", "architecture.py"),
    ]
    
    # Find which file actually exists
    for path in candidates:
        if os.path.exists(path):
            logger.info(f"Found architecture.py at: {path}")
            return path
    
    logger.error("Could not find architecture.py file")
    return None

def fix_architecture_module(architecture_path):
    """Fix the model/architecture.py file to handle both model types"""
    if not architecture_path or not os.path.exists(architecture_path):
        logger.error("Invalid architecture.py path")
        return False
    
    # Backup the file
    backup_path = architecture_path + ".bak"
    try:
        shutil.copy2(architecture_path, backup_path)
        logger.info(f"Created backup at: {backup_path}")
    except Exception as e:
        logger.warning(f"Could not create backup: {e}")
    
    # Read the file
    with open(architecture_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix the load_model function to detect architecture automatically
    load_model_pattern = r"def load_model\(.*?\):(.*?)(?=\n\ndef|\Z)"
    import re
    load_model_match = re.search(load_model_pattern, content, re.DOTALL)
    
    if load_model_match:
        old_load_model = load_model_match.group(0)
        
        # Create new load_model function with auto-detection
        new_load_model = """def load_model(path, n_mfcc=13, num_frames=101):
    """Load model from disk with robust error handling and automatic architecture detection"""
    if not path or not os.path.exists(path):
        logger.error(f"Model file not found: {path}")
        return None
    
    try:
        # Load state dictionary to check architecture
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        
        # Check for model architecture by looking at state_dict keys
        is_simple_model = any('conv_layer' in key for key in state_dict.keys())
        logger.info(f"Detected {'SimpleWakeWordModel' if is_simple_model else 'WakeWordModel'} architecture")
        
        # Create the appropriate model based on detected architecture
        if is_simple_model:
            logger.info(f"Creating SimpleWakeWordModel with n_mfcc={n_mfcc}, num_frames={num_frames}")
            model = SimpleWakeWordModel(n_mfcc=n_mfcc, num_frames=num_frames)
        else:
            logger.info(f"Creating WakeWordModel with n_mfcc={n_mfcc}, num_frames={num_frames}")
            model = WakeWordModel(n_mfcc=n_mfcc, num_frames=num_frames)
        
        # Check for compatibility
        expected_keys = set(model.state_dict().keys())
        loaded_keys = set(state_dict.keys())
        
        if expected_keys != loaded_keys:
            # Architecture mismatch even after detection, something is wrong
            logger.error(f"Model architecture mismatch. Expected keys: {expected_keys}, got: {loaded_keys}")
            logger.error("This probably means the loaded model was trained with a different architecture.")
            return None
        
        # Load the weights
        model.load_state_dict(state_dict)
        model.eval()  # Set to evaluation mode
        
        logger.info(f"Model loaded successfully from {path}")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None"""
        
        # Replace the load_model function in the content
        content = content.replace(old_load_model, new_load_model)
        logger.info("Updated load_model function with architecture detection")
    else:
        logger.warning("Could not find load_model function in architecture.py")
    
    # Write back the modified content
    try:
        with open(architecture_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info("Successfully fixed architecture.py")
        return True
    except Exception as e:
        logger.error(f"Error writing to architecture.py: {e}")
        return False

def main():
    """Main function to fix architecture issue"""
    logger.info("Wake Word Model Architecture Fix")
    logger.info("This script fixes the model architecture mismatch issue")
    
    # Find architecture.py
    architecture_path = find_architecture_module()
    if not architecture_path:
        logger.error("Cannot continue without finding architecture.py")
        return
    
    # Fix architecture.py
    success = fix_architecture_module(architecture_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print(" WAKE WORD MODEL ARCHITECTURE FIX - SUMMARY ")
    print("=" * 60)
    
    if success:
        print("\n✅ Successfully fixed model architecture loading!")
        print("\nThe system will now automatically detect which model architecture to use.")
        print("This allows loading models created with either SimpleWakeWordModel or WakeWordModel.")
    else:
        print("\n❌ Could not fix architecture.py")
        print("\nManual intervention required:")
        print("1. Open architecture.py and modify the load_model function to detect model type")
    
    print("\nNext steps:")
    print("  1. Restart your Io application")
    print("  2. The application should now correctly load your model")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
