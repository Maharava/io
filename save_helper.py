#!/usr/bin/env python3
"""
Save Helper for Wake Word Models

This module provides standalone functions to save and load wake word models
without depending on the trainer class.
"""
import os
import torch
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WakeWordSaveHelper")

def save_model(model, path, metadata=None):
    """
    Save a wake word model directly to disk
    
    Args:
        model: The PyTorch model to save
        path: Path where to save the model
        metadata: Optional dictionary with model metadata
        
    Returns:
        bool: True if successful, False otherwise
    """
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
                    f.write(f"{key}: {value}\n")
            logger.info(f"Metadata saved to {metadata_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return False

def load_model(path, model_class=None, **kwargs):
    """
    Load a wake word model from disk
    
    Args:
        path: Path to the saved model
        model_class: Model class to use (will try to import if None)
        **kwargs: Parameters to pass to model constructor
        
    Returns:
        The loaded model or None if failed
    """
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
    """
    Try to rescue the latest trained model from diagnostic checkpoint
    
    Returns:
        str: Path to the rescued model or None if failed
    """
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
