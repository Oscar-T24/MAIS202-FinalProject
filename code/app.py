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
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from data_processing import generate_spectrograms
from model import key_to_idx_tensors, KeystrokeCNN, train
import torch

app = Flask(__name__)
socketio = SocketIO(app)

# Global variables
recording = False
recording_thread = None
stop_flag = threading.Event()

# Create necessary directories
AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),'dataset','demo')  # so relative to the file app, demo directory is at dataset/audio
NUMPY_DIR = os.path.join(AUDIO_DIR,'numpy_arrays')

# Create necessary directories
for directory in [AUDIO_DIR, NUMPY_DIR]:
    print("created",directory)
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

@app.route('/train_model',methods=["POST","GET"])
def train_model():
    # Get the most recent recording files
    audio_files = sorted([f for f in os.listdir(AUDIO_DIR) if f.startswith('recording_') and f.endswith('.wav')])
    log_files = sorted([f for f in os.listdir(AUDIO_DIR) if f.startswith('keylog_') and f.endswith('.csv')])
    
    if not audio_files or not log_files:
        return jsonify({"status": "error", "message": "No recording files found"})
    
    latest_audio = os.path.join(AUDIO_DIR, audio_files[-1])
    latest_log = os.path.join(AUDIO_DIR, log_files[-1])
    

    try:

        data = generate_spectrograms(0,"demo","FFT") # this is a list of tuples for every (spectrogram, key )

        # then from this train the model, save it to the same directory and load it for prediction 

        spectrograms, labels = zip(*data) # unzip the list 

        spectrogram_tensors, label_tensors = torch.tensor(spectrograms).float(),key_to_idx_tensors(labels) # convert the label and features to tensors

        model = KeystrokeCNN()

        losses, *t = train(model,spectrogram_tensors,label_tensors,0.01,3,"demo",0.001)

        
        return jsonify({
            "status": "success",
            "message": "Numpy arrays created successfully",
            "num_keystrokes": len(data),
        })
    
    except Exception as e:
        #print(e)
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    socketio.run(app, port=5001, debug=True)
