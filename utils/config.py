"""
Configuration utility for wake word engine
"""
import json
import logging
from pathlib import Path

logger = logging.getLogger("WakeWord.Config")

def get_config_path():
    """Get path to configuration file"""
    config_dir = Path.home() / ".wakeword" / "config"
    config_file = config_dir / "config.json"
    return config_file

def load_config():
    """
    Load configuration from file
    
    Returns:
        dict: Configuration dictionary
    """
    config_file = get_config_path()
    
    try:
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_file}")
            return config
        else:
            logger.warning(f"Configuration file not found: {config_file}")
            return create_default_config()
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return create_default_config()

def save_config(config):
    """
    Save configuration to file
    
    Args:
        config: Configuration dictionary
        
    Returns:
        bool: Success or failure
    """
    config_file = get_config_path()
    
    try:
        # Create parent directory if it doesn't exist
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
        
        logger.info(f"Configuration saved to {config_file}")
        return True
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        return False

def create_default_config():
    """
    Create default configuration
    
    Returns:
        dict: Default configuration
    """
    default_config = {
        "audio_device": None,  # Will be selected on first run
        "sample_rate": 16000,
        "frame_size": 512,
        "model_path": None,    # Will be selected on first run
        "threshold": 0.85,
        "debounce_time": 3.0,  # seconds
        "action": {
            "type": "notification",
            "params": {"message": "Wake word detected!"}
        }
    }
    
    # Save the default configuration
    save_config(default_config)
    
    return default_config

def update_config(key, value):
    """
    Update a specific configuration value
    
    Args:
        key: Configuration key
        value: New value
        
    Returns:
        dict: Updated configuration
    """
    config = load_config()
    
    # Handle nested keys (e.g., "action.type")
    if "." in key:
        parts = key.split(".")
        current = config
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    else:
        config[key] = value
    
    save_config(config)
    return config
