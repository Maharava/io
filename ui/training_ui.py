"""
Training interface for wake word engine
"""
import os
import time
import threading
import PySimpleGUI as sg
import logging
import pyaudio
import numpy as np
import wave
from pathlib import Path
from ..model.training import WakeWordTrainer
from ..utils.config import load_config, save_config

logger = logging.getLogger("WakeWord.UI.Training")

class RecordingThread(threading.Thread):
    """Thread for recording audio samples"""
    def __init__(self, filename, duration, sample_rate=16000):
        super().__init__()
        self.filename = filename
        self.duration = duration
        self.sample_rate = sample_rate
        self.is_running = False
        self.daemon = True
    
    def run(self):
        """Record audio for specified duration"""
        self.is_running = True
        
        try:
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=1024
            )
            
            frames = []
            for _ in range(0, int(self.sample_rate / 1024 * self.duration)):
                if not self.is_running:
                    break
                data = stream.read(1024)
                frames.append(data)
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # Save to WAV file
            wf = wave.open(self.filename, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_format_size(pyaudio.paInt16))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(frames))
            wf.close()
            
        except Exception as e:
            logger.error(f"Error recording audio: {e}")
    
    def stop(self):
        """Stop recording"""
        self.is_running = False

class TrainingThread(threading.Thread):
    """Thread for training wake word model"""
    def __init__(self, wake_word_files, negative_files, model_name, progress_callback=None):
        super().__init__()
        self.wake_word_files = wake_word_files
        self.negative_files = negative_files
        self.model_name = model_name
        self.progress_callback = progress_callback
        self.result = None
        self.daemon = True
    
    def run(self):
        """Run training process"""
        try:
            # Create trainer
            trainer = WakeWordTrainer()
            
            # Prepare data
            if self.progress_callback:
                self.progress_callback("Preparing training data...", 10)
            
            train_loader, val_loader = trainer.prepare_data(
                self.wake_word_files, self.negative_files
            )
            
            # Train model
            if self.progress_callback:
                self.progress_callback("Training model...", 30)
            
            model = trainer.train(train_loader, val_loader)
            
            # Save model
            if self.progress_callback:
                self.progress_callback("Saving model...", 90)
            
            models_dir = Path.home() / ".wakeword" / "models"
            models_dir.mkdir(exist_ok=True)
            
            model_path = str(models_dir / self.model_name)
            trainer.save_trained_model(model, model_path)
            
            self.result = {
                "success": True,
                "model_path": model_path
            }
            
            if self.progress_callback:
                self.progress_callback("Training complete!", 100)
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            
            if self.progress_callback:
                self.progress_callback(f"Error: {e}", -1)
            
            self.result = {
                "success": False,
                "error": str(e)
            }

class TrainingWindow:
    def __init__(self, config):
        """
        Training window for wake word models
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.window = None
        self.recording_thread = None
        self.training_thread = None
        
        # Training data
        self.wake_word_files = []
        self.negative_files = []
    
    def run(self):
        """
        Run the training window
        
        Returns:
            dict: Training result with model_path or None if cancelled
        """
        # Set theme
        sg.theme('DarkBlue')
        
        # Create folder structure
        data_dir = Path.home() / ".wakeword" / "training_data"
        wake_word_dir = data_dir / "wake_word"
        negative_dir = data_dir / "negative"
        
        for directory in [data_dir, wake_word_dir, negative_dir]:
            directory.mkdir(exist_ok=True)
        
        # Load existing samples if any
        self.wake_word_files = list(wake_word_dir.glob("*.wav"))
        self.negative_files = list(negative_dir.glob("*.wav"))
        
        # Create layout
        layout = [
            [sg.Text('Wake Word Training', font=('Helvetica', 16))],
            
            [sg.Frame('Step 1: Record Wake Word Samples', [
                [sg.Text(f'Current samples: {len(self.wake_word_files)}', key='-WAKE_COUNT-')],
                [sg.Text('Speak your wake word clearly when recording')],
                [sg.Button('Record Wake Word (2s)', key='-RECORD_WAKE-'), 
                 sg.Button('Stop', key='-STOP_WAKE-', disabled=True),
                 sg.Button('Play', key='-PLAY_WAKE-', disabled=len(self.wake_word_files) == 0),
                 sg.Button('Delete Last', key='-DELETE_WAKE-', disabled=len(self.wake_word_files) == 0)]
            ])],
            
            [sg.Frame('Step 2: Record Background Noise', [
                [sg.Text(f'Current samples: {len(self.negative_files)}', key='-NEG_COUNT-')],
                [sg.Text('Record typical background sounds, music, speech, etc.')],
                [sg.Button('Record Background (5s)', key='-RECORD_NEG-'), 
                 sg.Button('Stop', key='-STOP_NEG-', disabled=True),
                 sg.Button('Play', key='-PLAY_NEG-', disabled=len(self.negative_files) == 0),
                 sg.Button('Delete Last', key='-DELETE_NEG-', disabled=len(self.negative_files) == 0)]
            ])],
            
            [sg.Frame('Step 3: Train Model', [
                [sg.Text('Model Name:'), 
                 sg.Input('my_wake_word.pth', key='-MODEL_NAME-', size=(30, 1))],
                [sg.Text('Required: At least 50 wake word samples and 10 background recordings')],
                [sg.Button('Start Training', key='-TRAIN-', 
                          disabled=(len(self.wake_word_files) < 50 or len(self.negative_files) < 10))]
            ])],
            
            [sg.ProgressBar(100, orientation='h', size=(40, 20), key='-PROGRESS-', visible=False)],
            [sg.Text('', key='-STATUS-')],
            
            [sg.Button('Close')]
        ]
        
        # Create window
        self.window = sg.Window('Wake Word Training', layout, finalize=True)
        
        # Event loop
        result = None
        while True:
            event, values = self.window.read(timeout=100)
            
            if event == sg.WINDOW_CLOSED or event == 'Close':
                # Stop any running threads
                if self.recording_thread and self.recording_thread.is_running:
                    self.recording_thread.stop()
                break
            
            # Wake word recording
            elif event == '-RECORD_WAKE-':
                self.start_recording(wake_word_dir / f"wake_{len(self.wake_word_files) + 1}.wav", 2)
                self.window['-RECORD_WAKE-'].update(disabled=True)
                self.window['-STOP_WAKE-'].update(disabled=False)
                self.window['-STATUS-'].update('Recording wake word...')
            
            elif event == '-STOP_WAKE-':
                if self.recording_thread:
                    self.recording_thread.stop()
                    self.recording_thread = None
                    self.window['-RECORD_WAKE-'].update(disabled=False)
                    self.window['-STOP_WAKE-'].update(disabled=True)
                    
                    # Update file list
                    self.wake_word_files = list(wake_word_dir.glob("*.wav"))
                    self.window['-WAKE_COUNT-'].update(f'Current samples: {len(self.wake_word_files)}')
                    self.window['-PLAY_WAKE-'].update(disabled=len(self.wake_word_files) == 0)
                    self.window['-DELETE_WAKE-'].update(disabled=len(self.wake_word_files) == 0)
                    self.window['-TRAIN-'].update(
                        disabled=(len(self.wake_word_files) < 50 or len(self.negative_files) < 10)
                    )
                    self.window['-STATUS-'].update('Wake word sample recorded')
            
            # Background recording
            elif event == '-RECORD_NEG-':
                self.start_recording(negative_dir / f"background_{len(self.negative_files) + 1}.wav", 5)
                self.window['-RECORD_NEG-'].update(disabled=True)
                self.window['-STOP_NEG-'].update(disabled=False)
                self.window['-STATUS-'].update('Recording background sounds...')
            
            elif event == '-STOP_NEG-':
                if self.recording_thread:
                    self.recording_thread.stop()
                    self.recording_thread = None
                    self.window['-RECORD_NEG-'].update(disabled=False)
                    self.window['-STOP_NEG-'].update(disabled=True)
                    
                    # Update file list
                    self.negative_files = list(negative_dir.glob("*.wav"))
                    self.window['-NEG_COUNT-'].update(f'Current samples: {len(self.negative_files)}')
                    self.window['-PLAY_NEG-'].update(disabled=len(self.negative_files) == 0)
                    self.window['-DELETE_NEG-'].update(disabled=len(self.negative_files) == 0)
                    self.window['-TRAIN-'].update(
                        disabled=(len(self.wake_word_files) < 50 or len(self.negative_files) < 10)
                    )
                    self.window['-STATUS-'].update('Background sample recorded')
            
            # Playback
            elif event == '-PLAY_WAKE-' and len(self.wake_word_files) > 0:
                self.play_audio(str(self.wake_word_files[-1]))
            
            elif event == '-PLAY_NEG-' and len(self.negative_files) > 0:
                self.play_audio(str(self.negative_files[-1]))
            
            # Delete samples
            elif event == '-DELETE_WAKE-' and len(self.wake_word_files) > 0:
                try:
                    os.remove(str(self.wake_word_files[-1]))
                    self.wake_word_files = self.wake_word_files[:-1]
                    self.window['-WAKE_COUNT-'].update(f'Current samples: {len(self.wake_word_files)}')
                    self.window['-PLAY_WAKE-'].update(disabled=len(self.wake_word_files) == 0)
                    self.window['-DELETE_WAKE-'].update(disabled=len(self.wake_word_files) == 0)
                    self.window['-TRAIN-'].update(
                        disabled=(len(self.wake_word_files) < 50 or len(self.negative_files) < 10)
                    )
                    self.window['-STATUS-'].update('Wake word sample deleted')
                except Exception as e:
                    logger.error(f"Error deleting file: {e}")
            
            elif event == '-DELETE_NEG-' and len(self.negative_files) > 0:
                try:
                    os.remove(str(self.negative_files[-1]))
                    self.negative_files = self.negative_files[:-1]
                    self.window['-NEG_COUNT-'].update(f'Current samples: {len(self.negative_files)}')
                    self.window['-PLAY_NEG-'].update(disabled=len(self.negative_files) == 0)
                    self.window['-DELETE_NEG-'].update(disabled=len(self.negative_files) == 0)
                    self.window['-TRAIN-'].update(
                        disabled=(len(self.wake_word_files) < 50 or len(self.negative_files) < 10)
                    )
                    self.window['-STATUS-'].update('Background sample deleted')
                except Exception as e:
                    logger.error(f"Error deleting file: {e}")
            
            # Training
            elif event == '-TRAIN-':
                model_name = values['-MODEL_NAME-']
                if not model_name.endswith('.pth'):
                    model_name += '.pth'
                
                # Disable all buttons during training
                for key in ['-RECORD_WAKE-', '-STOP_WAKE-', '-PLAY_WAKE-', '-DELETE_WAKE-',
                           '-RECORD_NEG-', '-STOP_NEG-', '-PLAY_NEG-', '-DELETE_NEG-', '-TRAIN-']:
                    self.window[key].update(disabled=True)
                
                # Show progress bar
                self.window['-PROGRESS-'].update(visible=True, current_count=0)
                self.window['-STATUS-'].update('Preparing for training...')
                
                # Start training in a separate thread
                self.training_thread = TrainingThread(
                    self.wake_word_files, 
                    self.negative_files,
                    model_name,
                    progress_callback=self.update_training_progress
                )
                self.training_thread.start()
            
            # Check if training is complete
            if self.training_thread and not self.training_thread.is_alive() and self.training_thread.result:
                if self.training_thread.result.get('success', False):
                    result = self.training_thread.result
                    sg.popup('Training Complete', 'The wake word model has been trained successfully!')
                    break
                else:
                    error = self.training_thread.result.get('error', 'Unknown error')
                    sg.popup_error(f'Training Error: {error}')
                    
                    # Re-enable buttons
                    self.window['-TRAIN-'].update(disabled=False)
                    self.window['-RECORD_WAKE-'].update(disabled=False)
                    self.window['-RECORD_NEG-'].update(disabled=False)
                    self.window['-PLAY_WAKE-'].update(disabled=len(self.wake_word_files) == 0)
                    self.window['-DELETE_WAKE-'].update(disabled=len(self.wake_word_files) == 0)
                    self.window['-PLAY_NEG-'].update(disabled=len(self.negative_files) == 0)
                    self.window['-DELETE_NEG-'].update(disabled=len(self.negative_files) == 0)
                
                self.training_thread = None
        
        # Close window
        self.window.close()
        return result
    
    def start_recording(self, filename, duration):
        """Start recording audio"""
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.stop()
        
        self.recording_thread = RecordingThread(filename, duration)
        self.recording_thread.start()
    
    def play_audio(self, filename):
        """Play an audio file"""
        try:
            import platform
            system = platform.system()
            
            if system == "Windows":
                os.system(f'start /min "" "{filename}"')
            elif system == "Darwin":  # macOS
                os.system(f'afplay "{filename}"')
            elif system == "Linux":
                os.system(f'aplay "{filename}"')
            else:
                logger.warning(f"Audio playback not implemented for {system}")
                
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
    
    def update_training_progress(self, status_text, progress_value):
        """Update training progress in UI"""
        if self.window:
            self.window['-STATUS-'].update(status_text)
            if progress_value >= 0:
                self.window['-PROGRESS-'].update(current_count=progress_value)
