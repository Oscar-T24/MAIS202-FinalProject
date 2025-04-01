from flask import Flask, render_template, jsonify, request
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write
import threading
import time
from pynput import keyboard
import csv
import os
import datetime
import soundfile as sf
import librosa
from data_recording import data_recording

app = Flask(__name__)

# Global variables
recording = False
recording_thread = None
stop_flag = threading.Event()
sample_rate = 44100
channels = 4

# Create necessary directories
AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audio')
SPECTROGRAMS_DIR = os.path.join(AUDIO_DIR, 'spectrograms')
NUMPY_DIR = os.path.join(AUDIO_DIR, 'numpy_arrays')

# Create all necessary directories
for directory in [AUDIO_DIR, SPECTROGRAMS_DIR, NUMPY_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def create_spectrogram_and_numpy(audio_segment, key, idx, output_dir, numpy_dir):
    """Generate and save spectrogram and numpy array for a keystroke"""
    if len(audio_segment.shape) > 1:  # If it's multi-channel
        print(f"Processing keystroke {idx + 1} for key '{key}'")
        print(f"Audio segment shape: {audio_segment.shape}")
        mel_specs = []
        for channel in range(audio_segment.shape[1]):
            mel_spec = librosa.feature.melspectrogram(
                y=audio_segment[:, channel].astype(np.float32),
                sr=sample_rate,
                n_mels=80,
                n_fft=2048,
                hop_length=512,
                window='hann',
                power=2.0
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
            mel_specs.append(mel_spec_norm)
        
        mel_spect_stacked = np.stack(mel_specs, axis=-1)
        numpy_array_path = os.path.join(numpy_dir, f"keystroke_{idx + 1}_{key}.npy")
        
        # Ensure the numpy directory exists
        os.makedirs(numpy_dir, exist_ok=True)
        
        # Save the numpy array
        np.save(numpy_array_path, mel_spect_stacked)
        print(f"Saved 4D NumPy array for '{key}' at {numpy_array_path}")
        print(f"Array shape: {mel_spect_stacked.shape}")
    else:
        print(f"Warning: Audio segment for keystroke {idx + 1} is not multi-channel")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_recording')
def start_recording():
    global recording, recording_thread, stop_flag

    if recording:
        return jsonify({"status": "error", "message": "Recording already in progress"})

    # Reset stop flag
    stop_flag.clear()

    # Generate timestamp for filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_file = os.path.join(AUDIO_DIR, f'recording_{timestamp}.wav')
    log_file = os.path.join(AUDIO_DIR, f'keylog_{timestamp}.csv')

    print(f"Starting recording session:")
    print(f"Audio file: {audio_file}")
    print(f"Log file: {log_file}")
    print(f"Numpy directory: {NUMPY_DIR}")
    print(f"Spectrograms directory: {SPECTROGRAMS_DIR}")

    # Start recording in a separate thread
    recording = True
    recording_thread = threading.Thread(target=data_recording, args=(audio_file, log_file, stop_flag))
    recording_thread.start()
    print("Recording started in separate thread")

    return jsonify({"status": "success", "message": "Recording started"})

@app.route('/stop_recording')
def stop_recording():
    global recording, recording_thread, stop_flag
    print(f"Stopping recording. Previous state: {recording}")  # Debug print
    
    # Set stop flag
    stop_flag.set()
    
    recording = False
    if recording_thread:
        recording_thread.join()
        print("Recording thread joined")
    
    print(f"Recording stopped. New state: {recording}")  # Debug print
    return jsonify({"status": "success", "message": "Recording stopped"})

if __name__ == '__main__':
    app.run(port="5001",debug=True)
