"""
Audio Capture Module - Handles microphone input and buffering
"""
import numpy as np
import pyaudio
import threading
import collections
import logging
import wave
import time

from .vad import VoiceActivityDetector

logger = logging.getLogger("Io.Audio")

class AudioCapture:
    """Thread-safe audio capture with PyAudio"""
    
    def __init__(self, device_index=None, sample_rate=16000, frame_size=512, callback=None):
        """Initialize audio capture with PyAudio"""
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
        
        # Keep track of audio levels for visualization
        self.current_audio_level = 0.0
    
    def list_devices(self):
        """List available audio input devices"""
        try:
            p = pyaudio.PyAudio()
            devices = []
            
            for i in range(p.get_device_count()):
                device_info = p.get_device_info_by_index(i)
                if device_info["maxInputChannels"] > 0:
                    devices.append({
                        "index": i,
                        "name": device_info["name"],
                        "channels": device_info["maxInputChannels"],
                        "sample_rate": int(device_info["defaultSampleRate"])
                    })
            
            p.terminate()
            return devices
        except Exception as e:
            logger.error(f"Error listing audio devices: {e}")
            return []
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback function for streaming audio data"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        try:
            # Convert bytes to numpy array (16-bit audio)
            audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
            
            # Normalize to [-1.0, 1.0]
            audio_data = audio_data / 32768.0
            
            # Calculate audio level for visualization
            self.current_audio_level = float(np.abs(audio_data).mean())
            
            # Apply automatic gain control (simple normalization)
            if np.abs(audio_data).max() > 0:
                audio_data = audio_data / np.abs(audio_data).max() * 0.9
            
            # Check if audio contains voice
            if self.vad.is_speech(audio_data):
                # Add to buffer with thread safety
                with self.lock:
                    self.buffer.append(audio_data)
                
                # Process the audio if callback is set
                if self.callback:
                    self.callback(audio_data)
        except Exception as e:
            logger.error(f"Error in audio callback: {e}")
        
        # Continue the stream
        return (None, pyaudio.paContinue)
    
    def start(self):
        """Start audio capture with proper resource management"""
        if self.is_running:
            return
        
        try:
            self.pyaudio = pyaudio.PyAudio()
            
            # If no device index specified, use default
            if self.device_index is None:
                try:
                    self.device_index = self.pyaudio.get_default_input_device_info()["index"]
                    logger.info(f"Using default audio device with index {self.device_index}")
                except Exception as e:
                    logger.error(f"Could not get default device: {e}. Using device 0.")
                    self.device_index = 0
            
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
            logger.info(f"Audio capture started on device {self.device_index}")
        except Exception as e:
            logger.error(f"Error starting audio capture: {e}")
            self._cleanup_resources()
    
    def stop(self):
        """Stop audio capture with proper resource cleanup"""
        if not self.is_running:
            return
        
        self.is_running = False
        self._cleanup_resources()
        logger.info("Audio capture stopped")
    
    def _cleanup_resources(self):
        """Clean up PyAudio resources properly"""
        try:
            if self.stream:
                if self.stream.is_active():
                    self.stream.stop_stream()
                self.stream.close()
                self.stream = None
        except Exception as e:
            logger.error(f"Error closing stream: {e}")
        
        try:
            if self.pyaudio:
                self.pyaudio.terminate()
                self.pyaudio = None
        except Exception as e:
            logger.error(f"Error terminating PyAudio: {e}")
    
    def get_buffer(self):
        """Get the current audio buffer (thread-safe)"""
        with self.lock:
            # Convert deque to numpy array
            if len(self.buffer) > 0:
                return np.concatenate(list(self.buffer))
            else:
                return np.array([], dtype=np.float32)
    
    def get_audio_level(self, device_index=None):
        """Get current audio level (for visualization)"""
        if device_index is not None and device_index != self.device_index:
            # If we're checking a different device, return random data for now
            # In a real implementation, we'd use PyAudio to check the device
            import random
            return random.uniform(0.0, 0.3)
        
        return self.current_audio_level
    
    def save_sample(self, filename, duration=3):
        """Record a sample to a WAV file for testing"""
        if self.is_running:
            logger.warning("Cannot save sample while capture is running")
            return False
        
        p = None
        stream = None
        
        try:
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.frame_size
            )
            
            logger.info(f"Recording {duration} second sample to {filename}")
            
            frames = []
            for i in range(0, int(self.sample_rate / self.frame_size * duration)):
                data = stream.read(self.frame_size)
                frames.append(data)
            
            logger.info("Finished recording")
            
            # Save to WAV file
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(p.get_sample_format_size(pyaudio.paInt16))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(frames))
            
            return True
        
        except Exception as e:
            logger.error(f"Error saving audio sample: {e}")
            return False
        
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            
            if p:
                p.terminate()
    
    def __del__(self):
        """Destructor to ensure resources are released"""
        self.stop()