"""
Configuration window for Neptune wake word engine
"""
import os
import PySimpleGUI as sg
import logging
from pathlib import Path
from utils.config import load_config, save_config

logger = logging.getLogger("Neptune.UI.Config")

class ConfigWindow:
    def __init__(self, config, audio_capture, detector):
        """
        Configuration window
        
        Args:
            config: Configuration dictionary
            audio_capture: AudioCapture instance
            detector: WakeWordDetector instance
        """
        self.config = config.copy()
        self.audio_capture = audio_capture
        self.detector = detector
        self.window = None
    
    def run(self):
        """
        Run the configuration window
        
        Returns:
            dict: Updated configuration or None if cancelled
        """
        # Get list of audio devices
        audio_devices = self.audio_capture.list_devices()
        device_names = [f"{device['index']}: {device['name']}" for device in audio_devices]
        
        # Get list of models
        models_dir = Path.home() / ".neptune" / "models"
        model_files = list(models_dir.glob("*.pth"))
        model_names = [model.name for model in model_files]
        
        # Calculate current device index for dropdown
        current_device_idx = 0
        if self.config['audio_device'] is not None:
            for i, device in enumerate(audio_devices):
                if device['index'] == self.config['audio_device']:
                    current_device_idx = i
                    break
        
        # Calculate current model index for dropdown
        current_model_idx = 0
        if self.config['model_path'] is not None:
            model_name = Path(self.config['model_path']).name
            if model_name in model_names:
                current_model_idx = model_names.index(model_name)
        
        # Create layout
        layout = [
            [sg.Text('Audio Device', size=(15, 1)), 
             sg.Combo(device_names, default_value=device_names[current_device_idx] if device_names else '', 
                      key='-DEVICE-', size=(40, 1))],
            
            [sg.Text('Wake Word Model', size=(15, 1)), 
             sg.Combo(model_names, default_value=model_names[current_model_idx] if model_names else '',
                      key='-MODEL-', size=(30, 1)),
             sg.Button('Browse', key='-BROWSE_MODEL-')],
            
            [sg.Text('Detection Threshold', size=(15, 1)), 
             sg.Slider(range=(0.5, 0.99), default_value=self.config['threshold'], resolution=0.01, 
                      orientation='h', key='-THRESHOLD-', size=(40, 15))],
            
            [sg.Text('Debounce Time (s)', size=(15, 1)), 
             sg.Slider(range=(0.5, 10.0), default_value=self.config['debounce_time'], resolution=0.5, 
                      orientation='h', key='-DEBOUNCE-', size=(40, 15))],
            
            [sg.Frame('Action on Detection', [
                [sg.Radio('Notification', 'ACTION_TYPE', default=self.config['action']['type'] == 'notification',
                         key='-ACTION_NOTIFICATION-')],
                [sg.Radio('Run Command', 'ACTION_TYPE', default=self.config['action']['type'] == 'command',
                         key='-ACTION_COMMAND-')],
                [sg.Radio('Keyboard Shortcut', 'ACTION_TYPE', default=self.config['action']['type'] == 'keyboard',
                         key='-ACTION_KEYBOARD-')],
                [sg.Radio('Custom Script', 'ACTION_TYPE', default=self.config['action']['type'] == 'custom_script',
                         key='-ACTION_SCRIPT-')],
                
                [sg.Text('Parameters:', size=(10, 1))],
                [sg.Text('Message', size=(10, 1), key='-PARAM_LABEL-'), 
                 sg.Input(default_text=self.get_action_param(), key='-PARAM_VALUE-', size=(40, 1))]
            ])],
            
            [sg.Button('Test'), sg.Button('Save'), sg.Button('Cancel')]
        ]
        
        # Create window
        self.window = sg.Window('Wake Word Configuration', layout, finalize=True)
        
        # Set up event handler for action type changes
        for action_key in ['-ACTION_NOTIFICATION-', '-ACTION_COMMAND-', '-ACTION_KEYBOARD-', '-ACTION_SCRIPT-']:
            self.window[action_key].bind('<ButtonRelease-1>', '-CLICKED')
        
        # Event loop
        while True:
            event, values = self.window.read()
            
            if event == sg.WINDOW_CLOSED or event == 'Cancel':
                self.window.close()
                return None
            
            elif event in ['-ACTION_NOTIFICATION-CLICKED', '-ACTION_COMMAND-CLICKED', 
                          '-ACTION_KEYBOARD-CLICKED', '-ACTION_SCRIPT-CLICKED']:
                # Update parameter label based on selected action
                self.update_action_ui(values)
            
            elif event == '-BROWSE_MODEL-':
                # Open file browser to select model
                model_path = sg.popup_get_file('Select Model File', 
                                             initial_folder=str(models_dir),
                                             file_types=(('PyTorch Models', '*.pth'),))
                if model_path:
                    model_name = Path(model_path).name
                    if model_name not in model_names:
                        model_names.append(model_name)
                        self.window['-MODEL-'].update(values=model_names, value=model_name)
                    else:
                        self.window['-MODEL-'].update(value=model_name)
            
            elif event == 'Test':
                # Test the current configuration
                self.test_config(values)
            
            elif event == 'Save':
                # Save configuration and close
                updated_config = self.save_config(values)
                self.window.close()
                return updated_config
    
    def update_action_ui(self, values):
        """Update action parameter UI based on selected action type"""
        if values['-ACTION_NOTIFICATION-']:
            self.window['-PARAM_LABEL-'].update('Message')
            self.window['-PARAM_VALUE-'].update(self.config['action'].get('params', {}).get('message', ''))
        
        elif values['-ACTION_COMMAND-']:
            self.window['-PARAM_LABEL-'].update('Command')
            self.window['-PARAM_VALUE-'].update(self.config['action'].get('params', {}).get('command', ''))
        
        elif values['-ACTION_KEYBOARD-']:
            self.window['-PARAM_LABEL-'].update('Shortcut')
            self.window['-PARAM_VALUE-'].update(self.config['action'].get('params', {}).get('shortcut', ''))
        
        elif values['-ACTION_SCRIPT-']:
            self.window['-PARAM_LABEL-'].update('Script Path')
            self.window['-PARAM_VALUE-'].update(self.config['action'].get('params', {}).get('script_path', ''))
    
    def get_action_param(self):
        """Get current action parameter value based on action type"""
        action_type = self.config['action']['type']
        params = self.config['action'].get('params', {})
        
        if action_type == 'notification':
            return params.get('message', '')
        elif action_type == 'command':
            return params.get('command', '')
        elif action_type == 'keyboard':
            return params.get('shortcut', '')
        elif action_type == 'custom_script':
            return params.get('script_path', '')
        else:
            return ''
    
    def test_config(self, values):
        """Test the current configuration"""
        # Extract threshold
        threshold = float(values['-THRESHOLD-'])
        
        # Update detector threshold temporarily
        original_threshold = self.detector.threshold
        self.detector.set_threshold(threshold)
        
        # Show test message
        sg.popup('Configuration Test',
                'Speak your wake word now to test the detection.',
                'The detector is listening with the current threshold setting.',
                'Close this window when done testing.',
                non_blocking=True)
        
        # Restore original threshold
        self.detector.set_threshold(original_threshold)
    
    def save_config(self, values):
        """Save configuration from UI values"""
        try:
            # Parse audio device
            device_str = values['-DEVICE-']
            if device_str:
                device_index = int(device_str.split(':')[0])
            else:
                device_index = None
            
            # Parse model path
            model_name = values['-MODEL-']
            if model_name:
                model_path = str(Path.home() / ".neptune" / "models" / model_name)
            else:
                model_path = None
            
            # Parse threshold and debounce
            threshold = float(values['-THRESHOLD-'])
            debounce_time = float(values['-DEBOUNCE-'])
            
            # Parse action type
            action_type = 'notification'
            if values['-ACTION_COMMAND-']:
                action_type = 'command'
            elif values['-ACTION_KEYBOARD-']:
                action_type = 'keyboard'
            elif values['-ACTION_SCRIPT-']:
                action_type = 'custom_script'
            
            # Parse action parameters
            param_value = values['-PARAM_VALUE-']
            action_params = {}
            
            if action_type == 'notification':
                action_params = {'message': param_value}
            elif action_type == 'command':
                action_params = {'command': param_value}
            elif action_type == 'keyboard':
                action_params = {'shortcut': param_value}
            elif action_type == 'custom_script':
                action_params = {'script_path': param_value}
            
            # Update configuration
            self.config['audio_device'] = device_index
            self.config['model_path'] = model_path
            self.config['threshold'] = threshold
            self.config['debounce_time'] = debounce_time
            self.config['action'] = {
                'type': action_type,
                'params': action_params
            }
            
            return self.config
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            sg.popup_error(f"Error saving configuration: {e}")
            return None