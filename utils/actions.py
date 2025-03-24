"""
Actions handler for Io wake word engine
"""
import subprocess
import threading
import time
import logging
import platform
import os
import queue

logger = logging.getLogger("Io.Actions")

class ActionStrategy:
    """Base class for action strategies"""
    
    def execute(self, params):
        """Execute the action with given parameters"""
        raise NotImplementedError


class NotificationAction(ActionStrategy):
    """Show a desktop notification"""
    
    def execute(self, params):
        message = params.get("message", "Wake word detected!")
        system = platform.system()
        
        try:
            if system == "Windows":
                try:
                    # Try to use Windows 10 toast notifications
                    from win10toast import ToastNotifier
                    toaster = ToastNotifier()
                    toaster.show_toast("Io Wake Word", message, duration=3, threaded=True)
                    return True
                except ImportError:
                    try:
                        # Try using Windows Balloon Tip (works on older Windows versions)
                        import win32api
                        import win32con
                        import win32gui
                        
                        # Create a dummy window for notifications
                        wc = win32gui.WNDCLASS()
                        wc.lpszClassName = "IoWakeWord"
                        wc.lpfnWndProc = lambda *args: None
                        wc.hInstance = win32api.GetModuleHandle(None)
                        wc.hIcon = win32gui.LoadIcon(0, win32con.IDI_APPLICATION)
                        wc.hCursor = win32gui.LoadCursor(0, win32con.IDC_ARROW)
                        wc.hbrBackground = win32con.COLOR_WINDOW
                        
                        try:
                            class_atom = win32gui.RegisterClass(wc)
                            hwnd = win32gui.CreateWindow(class_atom, "IoWakeWord", 0, 0, 0, 0, 0, 0, 0, wc.hInstance, None)
                            
                            # Display balloon notification
                            nid = (hwnd, 0, win32gui.NIF_INFO, win32con.WM_USER + 20, 
                                   win32gui.LoadIcon(0, win32con.IDI_APPLICATION), 
                                   "Io Wake Word", message, 200, 10)
                            win32gui.Shell_NotifyIcon(win32gui.NIM_ADD, nid)
                            
                            # Clean up after a delay
                            def cleanup():
                                time.sleep(3)
                                win32gui.DestroyWindow(hwnd)
                                win32gui.UnregisterClass(class_atom, wc.hInstance)
                                
                            threading.Thread(target=cleanup, daemon=True).start()
                            return True
                        except Exception as e:
                            logger.debug(f"Error with balloon tip: {e}")
                    except ImportError:
                        # If all else fails, use PowerShell to show notification
                        try:
                            ps_script = f'''
                            [System.Reflection.Assembly]::LoadWithPartialName("System.Windows.Forms")
                            $notify = New-Object System.Windows.Forms.NotifyIcon
                            $notify.Icon = [System.Drawing.SystemIcons]::Information
                            $notify.Visible = $true
                            $notify.ShowBalloonTip(3000, "Io Wake Word", "{message}", [System.Windows.Forms.ToolTipIcon]::Info)
                            Start-Sleep -s 3
                            $notify.Dispose()
                            '''
                            subprocess.run(['powershell', '-Command', ps_script], 
                                           capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
                            return True
                        except Exception as e:
                            logger.debug(f"Error with PowerShell notification: {e}")
                    
                    # Visual indicator as backup
                    logger.info("Wake word detected! (GUI notification not available)")
                    return True
            
            elif system == "Darwin":  # macOS
                cmd = f'osascript -e \'display notification "{message}" with title "Io Wake Word"\''
                subprocess.run(cmd, shell=True, timeout=3)
                return True
            
            elif system == "Linux":
                try:
                    # Try notify-send first
                    subprocess.run(['notify-send', 'Io Wake Word', message], timeout=3)
                    return True
                except FileNotFoundError:
                    # Try zenity as fallback
                    try:
                        subprocess.run(['zenity', '--info', '--text', f'Io Wake Word: {message}'], timeout=3)
                        return True
                    except FileNotFoundError:
                        logger.warning("No notification tool found on Linux")
                        return False
            
            else:
                logger.warning(f"Notifications not implemented for {system}")
                return False
                
        except Exception as e:
            logger.error(f"Error showing notification: {e}")
            return False


class CommandAction(ActionStrategy):
    """Run a shell command"""
    
    def execute(self, params):
        command = params.get("command", "")
        if not command:
            logger.warning("Empty command provided")
            return False
            
        try:
            # Use subprocess.Popen with proper arguments
            if platform.system() == "Windows":
                # Split command into args for Windows
                subprocess.Popen(command, creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                # Use shell on Unix systems
                subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            return True
                
        except Exception as e:
            logger.error(f"Error running command: {e}")
            return False


class KeyboardAction(ActionStrategy):
    """Send a keyboard shortcut"""
    
    def execute(self, params):
        shortcut = params.get("shortcut", "")
        if not shortcut:
            logger.warning("Empty shortcut provided")
            return False
            
        try:
            system = platform.system()
            
            if system in ["Windows", "Linux"]:
                try:
                    import pyautogui
                    # Split shortcut into keys (e.g., "ctrl+shift+a")
                    keys = shortcut.lower().split('+')
                    pyautogui.hotkey(*keys)
                    return True
                except ImportError:
                    logger.error("PyAutoGUI not installed for keyboard shortcuts")
                    return False
            
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
                return True
            
            else:
                logger.warning(f"Keyboard shortcuts not implemented for {system}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending keyboard shortcut: {e}")
            return False


class ScriptAction(ActionStrategy):
    """Run a custom script"""
    
    def execute(self, params):
        script_path = params.get("script_path", "")
        if not script_path or not os.path.exists(script_path):
            logger.warning(f"Script not found: {script_path}")
            return False
            
        try:
            if platform.system() == "Windows":
                subprocess.Popen(script_path, creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                subprocess.Popen(script_path, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            return True
                
        except Exception as e:
            logger.error(f"Error running script: {e}")
            return False


class ActionHandler:
    """Handle actions when wake word is detected"""
    
    def __init__(self, action_config, debounce_time=3.0):
        """Initialize action handler"""
        self.action_config = action_config
        self.debounce_time = debounce_time
        self.last_trigger_time = 0
        self.lock = threading.Lock()
        
        # Set up action strategies
        self.strategies = {
            "notification": NotificationAction(),
            "command": CommandAction(),
            "keyboard": KeyboardAction(),
            "custom_script": ScriptAction()
        }
        
        # Queue for actions to be executed
        self.action_queue = queue.Queue()
        
        # Start the action worker thread
        self.running = True
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
        while self.running:
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
        """Execute the configured action using strategy pattern"""
        try:
            action_type = action_config.get("type", "notification")
            params = action_config.get("params", {})
            
            logger.info(f"Executing action: {action_type}")
            
            # Get the appropriate strategy
            strategy = self.strategies.get(action_type)
            
            if strategy:
                strategy.execute(params)
            else:
                logger.warning(f"Unknown action type: {action_type}")
                
        except Exception as e:
            logger.error(f"Error executing action: {e}")
    
    def update_config(self, action_config, debounce_time=None):
        """Update action configuration thread-safely"""
        with self.lock:
            self.action_config = action_config
            
            if debounce_time is not None:
                self.debounce_time = debounce_time
                
            logger.info(f"Updated action configuration: {action_config}")
    
    def shutdown(self):
        """Shutdown the action worker cleanly"""
        self.running = False
        
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)