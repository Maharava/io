"""
Audio Capture Module - Handles microphone input and buffering
"""
import numpy as np
import pyaudio
import threading
import collections
import logging
from .vad import VoiceActivityDetector

logger = logging.getLogger("WakeWord.Audio")

class AudioCapture:
    def __init__(self, device_index=None, sample_rate=16000, frame_size=512, callback=None):
        """
        Initialize audio capture with PyAudio
        
        Args:
            device_index: Index of audio input device (None for default)
            sample_rate: Audio sampling rate in Hz
            frame_size: Number of samples per frame
            callback: Function to call with each audio frame
        """
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.callback = callback
        self.stream = None
        self.pyaudio = None
        self.is_running = False
        
        # Create circular buffer (2 seconds of audio)
        buffer_frames = int(2 * sample_rate / frame_size)
        self.buffer = collections.deque(maxlen=buffer_frames)
        
        # Create VAD for filtering silent frames
        self.vad = VoiceActivityDetector(sample_rate=sample_rate)
        
        # Lock for thread safety
        self.lock = threading.Lock()
    
    def list_devices(self):
        """List available audio input devices"""
        p = pyaudio.PyAudio()
        devices = []
        
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            if device_info["maxInputChannels"] > 0:
                devices.append({
                    "index": i,
                    "name": device_info["name"],
                    "channels": device_info["maxInputChannels"],
                    "sample_rate": device_info["defaultSampleRate"]
                })
        
        p.terminate()
        return devices
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback function for streaming audio data"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Convert bytes to numpy array (16-bit audio)
        audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
        
        # Normalize to [-1.0, 1.0]
        audio_data = audio_data / 32768.0
        
        # Apply automatic gain control (simple normalization)
        if np.abs(audio_data).max() > 0:
            audio_data = audio_data / np.abs(audio_data).max() * 0.9
        
        # Check if audio contains voice
        if self.vad.is_speech(audio_data):
            # Add to buffer
            with self.lock:
                self.buffer.append(audio_data)
            
            # Process the audio if callback is set
            if self.callback:
                self.callback(audio_data)
        
        # Continue the stream
        return (None, pyaudio.paContinue)
    
    def start(self):
        """Start audio capture"""
        if self.is_running:
            return
        
        self.pyaudio = pyaudio.PyAudio()
        
        # If no device index specified, use default
        if self.device_index is None:
            self.device_index = self.pyaudio.get_default_input_device_info()["index"]
            logger.info(f"Using default audio device with index {self.device_index}")
        
        # Open audio stream in callback mode
        self.stream = self.pyaudio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.frame_size,
            stream_callback=self._audio_callback
        )
        
        self.is_running = True
        logger.info("Audio capture started")
    
    def stop(self):
        """Stop audio capture"""
        if not self.is_running:
            return
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        if self.pyaudio:
            self.pyaudio.terminate()
            self.pyaudio = None
        
        self.is_running = False
        logger.info("Audio capture stopped")
    
    def get_buffer(self):
        """Get the current audio buffer (thread-safe)"""
        with self.lock:
            # Convert deque to numpy array
            if len(self.buffer) > 0:
                return np.concatenate(list(self.buffer))
            else:
                return np.array([], dtype=np.float32)
