from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import read as wav_read, write as wav_write
import threading
import time
from pynput import keyboard
import csv
import os
import datetime
import shutil
from data_recording import data_recording
from create_spectrogram import create_spectrogram_and_numpy

app = Flask(__name__)
socketio = SocketIO(app)

# Global variables
recording = False
recording_thread = None
stop_flag = threading.Event()

# Create necessary directories
AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audio')
NUMPY_DIR = os.path.join(AUDIO_DIR, 'numpy_arrays')

# Create necessary directories
for directory in [AUDIO_DIR, NUMPY_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

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

    # Start recording in a separate thread
    recording = True
    recording_thread = threading.Thread(target=data_recording, args=(audio_file, log_file, stop_flag))
    recording_thread.start()

    return jsonify({"status": "success", "message": "Recording started"})

@app.route('/stop_recording')
def stop_recording():
    global recording, recording_thread, stop_flag
    
    # Set stop flag
    stop_flag.set()
    
    recording = False
    if recording_thread:
        recording_thread.join()
    
    return jsonify({"status": "success", "message": "Recording stopped"})

@app.route('/train_model')
def train_model():
    # Get the most recent recording files
    audio_files = sorted([f for f in os.listdir(AUDIO_DIR) if f.startswith('recording_') and f.endswith('.wav')])
    log_files = sorted([f for f in os.listdir(AUDIO_DIR) if f.startswith('keylog_') and f.endswith('.csv')])
    
    if not audio_files or not log_files:
        return jsonify({"status": "error", "message": "No recording files found"})
    
    latest_audio = os.path.join(AUDIO_DIR, audio_files[-1])
    latest_log = os.path.join(AUDIO_DIR, log_files[-1])
    
    try:
        # Read the audio file
        sample_rate, audio_data = wav_read(latest_audio)
        
        # Read the keystroke log
        keystrokes = []
        with open(latest_log, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                keystrokes.append(row)
        
        # Process each keystroke
        for idx, (key, action, timestamp) in enumerate(keystrokes):
            if action == 'Pressed':  # Only process key press events
                # Extract audio segment around the keystroke
                start_sample = int(float(timestamp) * sample_rate)
                duration_samples = int(0.1 * sample_rate)  # 100ms window
                audio_segment = audio_data[start_sample:start_sample + duration_samples]
                
                # Create spectrogram and numpy array
                create_spectrogram_and_numpy(audio_segment, key, idx)
        
        return jsonify({
            "status": "success",
            "message": "Numpy arrays created successfully",
            "num_keystrokes": len(keystrokes)
        })
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    socketio.run(app, port=5001, debug=True)
