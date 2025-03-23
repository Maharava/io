"""
Trigger action handler - Manages actions when wake word is detected
"""
import subprocess
import threading
import time
import logging
import platform
import os
import queue

logger = logging.getLogger("WakeWord.Actions")

class TriggerHandler:
    def __init__(self, action_config, debounce_time=3.0):
        """Handle actions when wake word is detected"""
        self.action_config = action_config
        self.debounce_time = debounce_time
        self.last_trigger_time = 0
        self.lock = threading.Lock()
        
        # Queue for actions to be executed
        self.action_queue = queue.Queue()
        
        # Start the action worker thread
        self.worker_running = True
        self.worker_thread = threading.Thread(target=self._action_worker)
        self.worker_thread.daemon = True
        self.worker_thread.start()
    
    def trigger(self):
        """Execute configured action if debounce period has passed"""
        current_time = time.time()
        
        with self.lock:
            # Check if debounce period has passed
            if current_time - self.last_trigger_time < self.debounce_time:
                logger.debug("Ignoring trigger due to debounce period")
                return False
            
            # Update last trigger time
            self.last_trigger_time = current_time
        
        # Queue the action for execution
        action_config = self.action_config.copy()  # Copy to avoid race conditions
        self.action_queue.put(action_config)
        
        return True
    
    def _action_worker(self):
        """Worker thread to execute actions from the queue"""
        while self.worker_running:
            try:
                # Get action with timeout to allow thread to exit
                action_config = self.action_queue.get(timeout=1.0)
                
                # Execute the action
                self._execute_action(action_config)
                
                # Mark task as done
                self.action_queue.task_done()
            except queue.Empty:
                # Timeout waiting for action, just continue
                pass
            except Exception as e:
                logger.error(f"Error in action worker: {e}")
                # Don't crash the thread on error
    
    def _execute_action(self, action_config):
        """Execute the configured action"""
        try:
            action_type = action_config.get("type", "notification")
            params = action_config.get("params", {})
            
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
        """Show a desktop notification with better platform detection"""
        try:
            system = platform.system()
            
            if system == "Windows":
                # Use Windows toast notifications if available
                try:
                    from win10toast import ToastNotifier
                    toaster = ToastNotifier()
                    toaster.show_toast("Wake Word Detected", message, duration=3, threaded=True)
                except ImportError:
                    # Fall back to system messaging
                    self._run_command(f'msg "%username%" "Wake Word Detected: {message}"')
            
            elif system == "Darwin":  # macOS
                # Use AppleScript for notification
                cmd = f'osascript -e \'display notification "{message}" with title "Wake Word Detected"\''
                subprocess.run(cmd, shell=True, timeout=3)
            
            elif system == "Linux":
                # Try different notification methods
                try:
                    # Try notify-send first
                    subprocess.run(['notify-send', 'Wake Word Detected', message], timeout=3)
                except FileNotFoundError:
                    # Try zenity as fallback
                    try:
                        subprocess.run(['zenity', '--info', '--text', f'Wake Word Detected: {message}'], timeout=3)
                    except FileNotFoundError:
                        logger.warning("No notification tool found on Linux")
            
            else:
                logger.warning(f"Notifications not implemented for {system}")
                
        except Exception as e:
            logger.error(f"Error showing notification: {e}")
    
    def _run_command(self, command):
        """Run a shell command with better error handling"""
        try:
            # Use subprocess.Popen with proper arguments
            # Don't use shell=True on Windows for security reasons
            if platform.system() == "Windows":
                # Split command into args for Windows
                subprocess.Popen(command, creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                # Use shell on Unix systems
                subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
        except Exception as e:
            logger.error(f"Error running command: {e}")
    
    def _send_keyboard_shortcut(self, shortcut):
        """Send a keyboard shortcut with better error handling"""
        try:
            system = platform.system()
            
            if system in ["Windows", "Linux"]:
                try:
                    import pyautogui
                    # Split shortcut into keys (e.g., "ctrl+shift+a")
                    keys = shortcut.lower().split('+')
                    pyautogui.hotkey(*keys)
                except ImportError:
                    logger.error("PyAutoGUI not installed for keyboard shortcuts")
            
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
                subprocess.run(cmd, shell=True, timeout=3)
            
            else:
                logger.warning(f"Keyboard shortcuts not implemented for {system}")
                
        except Exception as e:
            logger.error(f"Error sending keyboard shortcut: {e}")
    
    def update_config(self, new_config):
        """Update action configuration thread-safely"""
        with self.lock:
            self.action_config = new_config
            logger.info(f"Updated action configuration: {new_config}")
    
    def shutdown(self):
        """Shutdown the action worker cleanly"""
        self.worker_running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)