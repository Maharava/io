"""
Configuration utility for wake word engine
"""
import json
import logging
from pathlib import Path
import os

logger = logging.getLogger("Neptune.Config")

def get_config_path():
    """Get path to configuration file"""
    config_dir = Path.home() / ".neptune" / "config"
    config_file = config_dir / "config.json"
    return config_file

def load_config():
    """Load configuration from file"""
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
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file: {e}")
        return create_default_config()
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return create_default_config()

def save_config(config):
    """Save configuration to file with proper resource management"""
    config_file = get_config_path()
    
    try:
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        
        # Write config with atomic operation for better reliability
        # First write to temporary file, then rename
        temp_file = config_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(config, f, indent=4)
        
        # On Windows, we need to remove the target file first
        if os.name == 'nt' and config_file.exists():
            os.remove(config_file)
            
        # Rename temp file to actual config file
        os.rename(temp_file, config_file)
        
        logger.info(f"Configuration saved to {config_file}")
        return True
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        return False

def create_default_config():
    """Create default configuration"""
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
    """Update a specific configuration value safely"""
    config = load_config()
    
    try:
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
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        return config  # Return unchanged config on error

def validate_config(config):
    """Validate configuration and fix any issues"""
    required_keys = {
        "audio_device": None,
        "sample_rate": 16000,
        "frame_size": 512,
        "model_path": None,
        "threshold": 0.85,
        "debounce_time": 3.0,
        "action": {
            "type": "notification",
            "params": {"message": "Wake word detected!"}
        }
    }
    
    # Check for missing keys and set defaults
    fixed = False
    for key, default_value in required_keys.items():
        if key not in config:
            config[key] = default_value
            fixed = True
    
    # Check action substructure
    if "action" in config:
        if not isinstance(config["action"], dict):
            config["action"] = required_keys["action"]
            fixed = True
        elif "type" not in config["action"]:
            config["action"]["type"] = "notification"
            fixed = True
        elif "params" not in config["action"]:
            config["action"]["params"] = {"message": "Wake word detected!"}
            fixed = True
    
    # Ensure threshold is within valid range
    if "threshold" in config:
        config["threshold"] = max(0.5, min(0.99, float(config["threshold"])))
    
    # Save fixed config if needed
    if fixed:
        logger.warning("Fixed invalid configuration")
        save_config(config)
    
    return config