"""
Emergency Model Handler for Wake Word Detection

This script handles the model loading and saving issues that might occur
in the wake word training system. It can be used as a standalone utility
to save a model that failed to save during training.
"""
import os
import sys
import torch
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("ModelHandler")

class ModelHandler:
    """Utility class for handling wake word models"""
    
    @staticmethod
    def save_model(model, path, metadata=None):
        """
        Save a wake word model to disk
        
        Args:
            model: The PyTorch model to save
            path: Path where to save the model
            metadata: Optional dictionary of metadata to save alongside
            
        Returns:
            bool: Success status
        """
        try:
            # Validate model
            if model is None:
                logger.error("Cannot save None model")
                return False
                
            # Make sure model has state_dict
            if not hasattr(model, 'state_dict'):
                logger.error("Model does not have state_dict method")
                return False
            
            # Ensure the directory exists
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save the model
            torch.save(model.state_dict(), path)
            logger.info(f"Model saved to {path}")
            
            # Save metadata if provided
            if metadata:
                meta_path = str(Path(path).with_suffix('.txt'))
                with open(meta_path, 'w') as f:
                    for key, value in metadata.items():
                        f.write(f"{key}: {value}\n")
                logger.info(f"Metadata saved to {meta_path}")
            
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    @staticmethod
    def load_model(path, model_class=None, **kwargs):
        """
        Load a wake word model from disk
        
        Args:
            path: Path to the saved model
            model_class: Optional model class to use (if None, will try to import)
            **kwargs: Parameters to pass to the model constructor
            
        Returns:
            The loaded model or None if failed
        """
        try:
            # Validate path
            if not os.path.exists(path):
                logger.error(f"Model file not found: {path}")
                return None
            
            # Get model class if not provided
            if model_class is None:
                try:
                    # Try to import from local modules
                    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                    from model.architecture import WakeWordModel, SimpleWakeWordModel
                    
                    # Default to WakeWordModel
                    model_class = WakeWordModel
                    logger.info(f"Using model class: {model_class.__name__}")
                except ImportError:
                    logger.error("Could not import model classes")
                    return None
            
            # Create a new instance
            model = model_class(**kwargs)
            
            # Load the state dictionary
            state_dict = torch.load(path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            
            # Set to evaluation mode
            model.eval()
            
            logger.info(f"Model loaded successfully from {path}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    @staticmethod
    def extract_last_state_dict(output_dir, target_path):
        """
        Extract the last saved state dict from the output directory and save it properly
        
        Args:
            output_dir: Directory with training outputs
            target_path: Where to save the extracted model
            
        Returns:
            bool: Success status
        """
        try:
            # Look for model files
            model_files = [
                os.path.join(output_dir, "model_best.pth"),
                os.path.join(output_dir, "model_epoch_80.pth"),
                os.path.join(output_dir, "model_epoch_60.pth"),
                os.path.join(output_dir, "model_epoch_40.pth"),
                os.path.join(output_dir, "model_epoch_20.pth"),
            ]
            
            # Find the first existing model
            model_path = None
            for file_path in model_files:
                if os.path.exists(file_path):
                    model_path = file_path
                    break
            
            if model_path is None:
                logger.error(f"No model files found in {output_dir}")
                return False
            
            logger.info(f"Found model file: {model_path}")
            
            # Load the state dict
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            
            # Save it to the target path
            Path(target_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(state_dict, target_path)
            
            logger.info(f"Extracted model saved to {target_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error extracting model: {e}")
            return False

def rescue_last_trained_model():
    """
    Try to rescue the last trained model from the diagnostic directory
    """
    diagnostics_dir = Path.home() / ".io" / "training_diagnostics"
    if not diagnostics_dir.exists():
        logger.error(f"Diagnostics directory not found: {diagnostics_dir}")
        return False
    
    # Target model location
    models_dir = Path.home() / ".io" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    target_path = models_dir / "rescued_model.pth"
    
    return ModelHandler.extract_last_state_dict(str(diagnostics_dir), str(target_path))

def main():
    """Main entry point"""
    logger.info("Wake Word Model Handler")
    logger.info("1. Attempting to rescue latest trained model...")
    
    if rescue_last_trained_model():
        logger.info("\nModel rescued successfully!")
        logger.info(f"You can now use the model at: {Path.home() / '.io' / 'models' / 'rescued_model.pth'}")
    else:
        logger.error("\nCould not rescue model automatically.")
        logger.info("You can still use this script to manage models manually.")
        logger.info("Example usage:")
        logger.info("from model_handler import ModelHandler")
        logger.info("# To save: ModelHandler.save_model(model, 'path/to/save.pth')")
        logger.info("# To load: model = ModelHandler.load_model('path/to/model.pth')")

if __name__ == "__main__":
    main()
