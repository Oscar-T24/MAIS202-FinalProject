import numpy as np
import sounddevice as sd
import librosa
import torch
import torch.nn.functional as F
from scipy.signal import find_peaks
from queue import Queue
import time

class LiveKeystrokeDetector:
    def __init__(self, model, sample_rate=44100, window_size=0.2, threshold=0.9999999):
        self.model = model
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.threshold = threshold
        self.window_samples = int(window_size * sample_rate)
        self.audio_buffer = np.zeros(self.window_samples)
        self.prediction_queue = Queue()
        self.is_recording = False
        self.debounce_time = time.time()
        
    def audio_callback(self, indata, frames, time_info, status):
        """Callback function for audio streaming"""
        if status:
            print(f"Audio callback status: {status}")
            
        # Update buffer with new audio data
        self.audio_buffer = np.roll(self.audio_buffer, -frames)
        self.audio_buffer[-frames:] = indata.flatten()
        
        # Generate mel spectrogram
        mel_spect = librosa.feature.melspectrogram(
            y=self.audio_buffer,
            sr=self.sample_rate,
            n_mels=80,
            n_fft=2048,
            hop_length=512,
            window='hann',
            power=2.0
        )

        # Convert to log scale (dB)
        mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)

        # Normalize to 0-1 range
        mel_spect_norm = (mel_spect_db - mel_spect_db.min()) / (mel_spect_db.max() - mel_spect_db.min())

        # Convert to tensor
        spectrogram_tensor = torch.FloatTensor(mel_spect_norm)
        
        # Pad or truncate to 300 in the last dimension
        current_length = spectrogram_tensor.shape[-1]
        if current_length < 300:
            # Pad to 300
            padding = (0, 300 - current_length)
            spectrogram_tensor = F.pad(spectrogram_tensor, padding, mode='constant', value=0)
        elif current_length > 300:
            # Center crop to 300
            start = (current_length - 300) // 2
            spectrogram_tensor = spectrogram_tensor[..., start:start+300]
        
        # Get the energy over time
        energy = np.mean(mel_spect, axis=0)
        
        # Find peaks with minimum height and distance
        peaks, _ = find_peaks(energy, 
                            height=0.5,          # Minimum height
                            distance=20,         # Minimum samples between peaks
                            prominence=0.3)      # Minimum prominence of peaks
        
        if len(peaks) > 0 and time.time() - self.debounce_time > 1:
            print("Keystroke detected")
            # Make prediction
            with torch.no_grad():
                spectrogram_tensor = spectrogram_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
                output = self.model(spectrogram_tensor)  # Forward pass
                probabilities = F.softmax(output, dim=1)  # Convert logits to probabilities
                predicted_idx = torch.argmax(probabilities, dim=1).item()  # Get class index
                
                # Convert index to letter
                letters = "abcdefghijklmnopqrstuvwxyz"
                predicted_letter = letters[predicted_idx]
                
                self.prediction_queue.put(predicted_letter)
                print(f"Predicted key: {predicted_letter}")

            # Add a delay to prevent multiple detections for the same keystroke
            self.debounce_time = time.time()
    
    def start_recording(self):
        """Start recording and processing audio"""
        self.is_recording = True
        print("Starting live keystroke detection...")
        
        try:
            with sd.InputStream(samplerate=self.sample_rate, 
                              channels=1, 
                              callback=self.audio_callback):
                while self.is_recording:
                    time.sleep(0.01)  # Prevent high CPU usage
                    
        except Exception as e:
            print(f"Error in audio recording: {e}")
            self.is_recording = False
    
    def stop_recording(self):
        """Stop recording"""
        self.is_recording = False
        print("Stopping keystroke detection...") 