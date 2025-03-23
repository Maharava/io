"""
Trigger action handler - Manages actions when wake word is detected
"""
import subprocess
import threading
import time
import logging
import platform
import os

logger = logging.getLogger("WakeWord.Actions")

class TriggerHandler:
    def __init__(self, action_config, debounce_time=3.0):
        """
        Handle actions when wake word is detected
        
        Args:
            action_config: Dictionary with action configuration
            debounce_time: Minimum time between triggers (seconds)
        """
        self.action_config = action_config
        self.debounce_time = debounce_time
        self.last_trigger_time = 0
        self.lock = threading.Lock()
    
    def trigger(self):
        """
        Execute configured action if debounce period has passed
        """
        current_time = time.time()
        
        with self.lock:
            # Check if debounce period has passed
            if current_time - self.last_trigger_time < self.debounce_time:
                logger.debug("Ignoring trigger due to debounce period")
                return False
            
            # Update last trigger time
            self.last_trigger_time = current_time
        
        # Execute action in a separate thread to not block audio processing
        action_thread = threading.Thread(target=self._execute_action)
        action_thread.daemon = True
        action_thread.start()
        
        return True
    
    def _execute_action(self):
        """Execute the configured action"""
        try:
            action_type = self.action_config.get("type", "notification")
            params = self.action_config.get("params", {})
            
            logger.info(f"Executing action: {action_type}")
            
            if action_type == "notification":
                self._show_notification(params.get("message", "Wake word detected!"))
            
            elif action_type == "command":
                command = params.get("command", "")
                if command:
                    self._run_command(command)
            
            elif action_type == "keyboard":
                shortcut = params.get("shortcut", "")
                if shortcut:
                    self._send_keyboard_shortcut(shortcut)
            
            elif action_type == "custom_script":
                script_path = params.get("script_path", "")
                if script_path and os.path.exists(script_path):
                    self._run_command(script_path)
            
            else:
                logger.warning(f"Unknown action type: {action_type}")
        
        except Exception as e:
            logger.error(f"Error executing action: {e}")
    
    def _show_notification(self, message):
        """Show a desktop notification"""
        try:
            system = platform.system()
            
            if system == "Windows":
                # Use Windows toast notifications
                from win10toast import ToastNotifier
                toaster = ToastNotifier()
                toaster.show_toast("Wake Word Detected", message, duration=3)
            
            elif system == "Darwin":  # macOS
                # Use AppleScript for notification
                cmd = f'osascript -e \'display notification "{message}" with title "Wake Word Detected"\''
                subprocess.run(cmd, shell=True)
            
            elif system == "Linux":
                # Use notify-send on Linux
                cmd = f'notify-send "Wake Word Detected" "{message}"'
                subprocess.run(cmd, shell=True)
            
            else:
                logger.warning(f"Notifications not implemented for {system}")
                
        except Exception as e:
            logger.error(f"Error showing notification: {e}")
    
    def _run_command(self, command):
        """Run a shell command"""
        try:
            subprocess.Popen(command, shell=True)
        except Exception as e:
            logger.error(f"Error running command: {e}")
    
    def _send_keyboard_shortcut(self, shortcut):
        """Send a keyboard shortcut"""
        try:
            system = platform.system()
            
            if system == "Windows":
                import pyautogui
                # Split shortcut into keys (e.g., "ctrl+shift+a")
                keys = shortcut.lower().split('+')
                pyautogui.hotkey(*keys)
            
            elif system == "Darwin":  # macOS
                # Convert shortcut to applescript format and execute
                keys = shortcut.lower().split('+')
                
                # Map common key names
                key_map = {
                    "ctrl": "control", 
                    "cmd": "command",
                    "alt": "option",
                    "win": "command"
                }
                
                # Convert keys to AppleScript format
                as_keys = []
                for key in keys:
                    as_keys.append(key_map.get(key, key))
                
                # Last item in the list is the main key, others are modifiers
                main_key = as_keys.pop()
                modifiers = ', '.join(f'"{k} down"' for k in as_keys)
                
                cmd = f'osascript -e \'tell application "System Events" to keystroke "{main_key}" using {{{modifiers}}}\''
                subprocess.run(cmd, shell=True)
            
            elif system == "Linux":
                import pyautogui
                keys = shortcut.lower().split('+')
                pyautogui.hotkey(*keys)
            
            else:
                logger.warning(f"Keyboard shortcuts not implemented for {system}")
                
        except Exception as e:
            logger.error(f"Error sending keyboard shortcut: {e}")
    
    def update_config(self, new_config):
        """Update action configuration"""
        with self.lock:
            self.action_config = new_config
            logger.info(f"Updated action configuration: {new_config}")
