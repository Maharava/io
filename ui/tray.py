"""
System tray application for wake word engine
"""
import sys
import logging
import threading
import pystray
from PIL import Image, ImageDraw
from ..utils.config import load_config, save_config
from .config import ConfigWindow
from .training_ui import TrainingWindow

logger = logging.getLogger("WakeWord.UI")

class SystemTrayApp:
    def __init__(self, audio_processor, config):
        """
        System tray application
        
        Args:
            audio_processor: AudioProcessor instance
            config: Configuration dictionary
        """
        self.audio_processor = audio_processor
        self.config = config
        self.tray_icon = None
        self.is_running = True
        
        # Create an image for the system tray icon
        self.icon_image = self._create_icon_image()
    
    def _create_icon_image(self, size=(64, 64)):
        """Create a simple icon image"""
        image = Image.new('RGB', size, color=(0, 0, 0))
        dc = ImageDraw.Draw(image)
        
        # Draw a microphone-like shape
        dc.rectangle((16, 16, 48, 40), fill=(0, 128, 255))
        dc.ellipse((20, 8, 44, 24), fill=(0, 128, 255))
        dc.rectangle((28, 40, 36, 56), fill=(0, 128, 255))
        dc.ellipse((20, 48, 44, 64), fill=(0, 128, 255))
        
        return image
    
    def _handle_tray_action(self, icon, item):
        """Handle tray menu actions"""
        item_text = item.text.lower()
        
        if item_text == 'enable':
            self.audio_processor.start()
            logger.info("Wake word detection enabled")
        
        elif item_text == 'disable':
            self.audio_processor.stop()
            logger.info("Wake word detection disabled")
        
        elif item_text == 'settings':
            # Open settings window
            self._open_settings_window()
        
        elif item_text == 'training':
            # Open training window
            self._open_training_window()
        
        elif item_text == 'exit':
            # Stop the audio processor
            self.audio_processor.stop()
            
            # Stop the tray icon
            icon.stop()
            
            # Mark as not running
            self.is_running = False
    
    def _open_settings_window(self):
        """Open the settings window"""
        # Run in a separate thread to not block the tray icon
        settings_thread = threading.Thread(target=self._run_settings_window)
        settings_thread.daemon = True
        settings_thread.start()
    
    def _run_settings_window(self):
        """Run the settings window"""
        # Create and run the settings window
        config_window = ConfigWindow(
            self.config, 
            self.audio_processor.audio_capture, 
            self.audio_processor.detector
        )
        new_config = config_window.run()
        
        # Update config if changes were made
        if new_config:
            self.config = new_config
            save_config(new_config)
            
            # Update the audio processor with new config
            self.audio_processor.update_config(new_config)
    
    def _open_training_window(self):
        """Open the training window"""
        # Run in a separate thread to not block the tray icon
        training_thread = threading.Thread(target=self._run_training_window)
        training_thread.daemon = True
        training_thread.start()
    
    def _run_training_window(self):
        """Run the training window"""
        # Create and run the training window
        training_window = TrainingWindow(self.config)
        result = training_window.run()
        
        # If training completed successfully, update the model
        if result and result.get('success') and 'model_path' in result:
            self.config['model_path'] = result['model_path']
            save_config(self.config)
            
            # Update the audio processor with new config
            self.audio_processor.update_config(self.config)
    
    def run(self):
        """Run the system tray application"""
        menu_items = [
            pystray.MenuItem('Enable', self._handle_tray_action),
            pystray.MenuItem('Disable', self._handle_tray_action),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem('Settings', self._handle_tray_action),
            pystray.MenuItem('Training', self._handle_tray_action),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem('Exit', self._handle_tray_action)
        ]
        
        # Create the tray icon
        self.tray_icon = pystray.Icon(
            'WakeWordDetector',
            self.icon_image,
            'Wake Word Detector',
            menu=pystray.Menu(*menu_items)
        )
        
        # Run the tray icon (this will block until exit)
        self.tray_icon.run()