from flask import Flask, render_template, jsonify, request
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sounddevice as sd
import scipy.signal as signal
from scipy.io.wavfile import write as wav_write, read as wav_read
import threading
import time
from pynput import keyboard
import csv
import os
import nbformat
import importlib.util
import tempfile
import datetime
import soundfile as sf
import librosa

# Import the CNN model from the notebook
def notebook_to_module(notebook_path):
    with open(notebook_path, 'r') as f:
        nb = nbformat.read(f, as_version=4)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        for cell in nb.cells:
            if cell.cell_type == 'code':
                if any(x in cell.source for x in [
                    'load_spectrograms_from_directory',
                    'train',
                    'input(',
                    'DATA_DIR =',
                    'NUMPY_DIR =',
                    'spectrogram_tensors, keys, max_width =',
                    'train_dataset =',
                    'train_loader =',
                    'data_to_save =',
                    'torch.save',
                    'loaded_data = torch.load'
                ]):
                    continue
                f.write(cell.source + '\n')
        return f.name

# Import the model
notebook_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'code', 'nn.ipynb')
module_path = notebook_to_module(notebook_path)
spec = importlib.util.spec_from_file_location("nn_module", module_path)
nn_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nn_module)
KeystrokeCNN = nn_module.KeystrokeCNN

app = Flask(__name__)

# Global variables
model = KeystrokeCNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
recording = False
recording_thread = None
keyboard_listener = None
start_time_audio = None
stop_recording = False
audio_buffer = []
key_log = []
sample_rate = 44100
channels = 4

# Create necessary directories
AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audio')
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
SPECTROGRAMS_DIR = os.path.join(AUDIO_DIR, 'spectrograms')
NUMPY_DIR = os.path.join(AUDIO_DIR, 'numpy_arrays')

# Create all necessary directories
for directory in [AUDIO_DIR, MODELS_DIR, SPECTROGRAMS_DIR, NUMPY_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def audio_callback(indata, frames, time, status):
    """Callback function for audio streaming"""
    if status:
        print(f"Audio callback status: {status}")
    if recording:
        audio_buffer.append(indata.copy())

def on_press(key):
    """Callback for key press events"""
    if start_time_audio is None or not recording:
        return

    timestamp = time.time() - start_time_audio
    try:
        key_str = key.char
    except AttributeError:
        key_str = str(key)

    with open(log_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([key_str, 'Pressed', round(timestamp, 6)])

def on_release(key):
    """Callback for key release events"""
    global recording
    if start_time_audio is None or not recording:
        return

    timestamp = time.time() - start_time_audio
    try:
        key_str = key.char
    except AttributeError:
        key_str = str(key)

    with open(log_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([key_str, 'Released', round(timestamp, 6)])

    if key == keyboard.Key.esc:
        recording = False
        return False

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

def record_audio():
    """Record audio and keystrokes"""
    global start_time_audio, stop_recording, audio_buffer, key_log, log_file
    print("Recording audio...")
    
    # Reset buffers
    audio_buffer = []
    key_log = []
    start_time_audio = time.time()
    stop_recording = False
    
    try:
        # Generate timestamp for filenames
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_file = os.path.join(AUDIO_DIR, f'recording_{timestamp}.wav')
        log_file = os.path.join(AUDIO_DIR, f'keylog_{timestamp}.csv')

        print(f"Starting recording session:")
        print(f"Audio file: {audio_file}")
        print(f"Log file: {log_file}")
        print(f"Numpy directory: {NUMPY_DIR}")
        print(f"Spectrograms directory: {SPECTROGRAMS_DIR}")

        # Initialize keystroke log file
        with open(log_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Key', 'Action', 'Timestamp'])

        with sd.InputStream(samplerate=sample_rate, channels=channels, callback=audio_callback):
            while not stop_recording:
                time.sleep(0.1)

        # Save the recorded audio
        if audio_buffer:
            audio_data = np.concatenate(audio_buffer, axis=0)
            wav_write(audio_file, sample_rate, audio_data)
            print(f"Saved audio data with shape: {audio_data.shape}")
        
        # Process keystrokes and generate spectrograms
        keystroke_times = []
        with open(log_file, "r") as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            stack = {}

            for row in reader:
                key, action, timestamp = row[0], row[1], float(row[2])
                if action == "Pressed":
                    stack[key] = timestamp
                elif action == "Released" and key in stack:
                    keystroke_times.append((key, stack.pop(key), timestamp))

        print(f"Found {len(keystroke_times)} keystrokes to process")

        # Generate spectrograms for each keystroke
        BUFFER = 0.1
        for idx, (key, press_time, release_time) in enumerate(keystroke_times):
            start_time = max(0, press_time - BUFFER)
            end_time = min(len(audio_data) / sample_rate, release_time + BUFFER)
            
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            keystroke_audio = audio_data[start_sample:end_sample]
            
            if len(keystroke_audio) == 0:
                print(f"Warning: Empty audio segment for keystroke {idx + 1}")
                continue
            
            create_spectrogram_and_numpy(
                keystroke_audio, 
                key, 
                idx,
                SPECTROGRAMS_DIR,
                NUMPY_DIR
            )

    except Exception as e:
        print(f"Recording error: {e}")
    
    print("Recording finished")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_recording')
def start_recording():
    global recording, audio_buffer, start_time_audio, log_file

    if recording:
        return jsonify({"status": "error", "message": "Recording already in progress"})

    # Generate timestamp for filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_file = os.path.join(AUDIO_DIR, f'recording_{timestamp}.wav')
    log_file = os.path.join(AUDIO_DIR, f'keylog_{timestamp}.csv')

    print(f"Starting recording session:")
    print(f"Audio file: {audio_file}")
    print(f"Log file: {log_file}")
    print(f"Numpy directory: {NUMPY_DIR}")
    print(f"Spectrograms directory: {SPECTROGRAMS_DIR}")

    # Initialize keystroke log file
    with open(log_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Key', 'Action', 'Timestamp'])

    # Clear buffers and set recording flag
    audio_buffer = []
    recording = True
    start_time_audio = time.time()

    # Start audio recording thread
    def record_audio():
        try:
            with sd.InputStream(samplerate=sample_rate, channels=channels, callback=audio_callback):
                while recording:
                    time.sleep(0.1)

            # Save recorded audio
            if audio_buffer:
                audio_data = np.concatenate(audio_buffer, axis=0)
                wav_write(audio_file, sample_rate, audio_data)
                print(f"Saved audio data with shape: {audio_data.shape}")

                # Process keystrokes and generate spectrograms
                keystroke_times = []
                with open(log_file, "r") as file:
                    reader = csv.reader(file)
                    next(reader)  # Skip header
                    stack = {}

                    for row in reader:
                        key, action, timestamp = row[0], row[1], float(row[2])
                        if action == "Pressed":
                            stack[key] = timestamp
                        elif action == "Released" and key in stack:
                            keystroke_times.append((key, stack.pop(key), timestamp))

                print(f"Found {len(keystroke_times)} keystrokes to process")

                # Generate spectrograms for each keystroke
                BUFFER = 0.1
                for idx, (key, press_time, release_time) in enumerate(keystroke_times):
                    start_time = max(0, press_time - BUFFER)
                    end_time = min(len(audio_data) / sample_rate, release_time + BUFFER)
                    
                    start_sample = int(start_time * sample_rate)
                    end_sample = int(end_time * sample_rate)
                    keystroke_audio = audio_data[start_sample:end_sample]
                    
                    if len(keystroke_audio) == 0:
                        print(f"Warning: Empty audio segment for keystroke {idx + 1}")
                        continue
                    
                    create_spectrogram_and_numpy(
                        keystroke_audio, 
                        key, 
                        idx,
                        SPECTROGRAMS_DIR,
                        NUMPY_DIR
                    )

        except Exception as e:
            print(f"Error in audio recording: {e}")
            import traceback
            traceback.print_exc()

    # Start recording threads
    audio_thread = threading.Thread(target=record_audio)
    audio_thread.start()

    # Start keyboard listener
    keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    keyboard_listener.start()

    return jsonify({"status": "success", "message": "Recording started"})

@app.route('/stop_recording')
def stop_recording():
    global recording
    recording = False
    return jsonify({"status": "success", "message": "Recording stopped"})

@app.route('/train_model', methods=['POST'])
def train_model():
    """Train the model on the recorded data"""
    # Get the latest recorded data
    audio_files = sorted([f for f in os.listdir(AUDIO_DIR) if f.startswith('audio_')])
    if not audio_files:
        return jsonify({"status": "error", "message": "No recorded data available"})
    
    latest_audio = audio_files[-1]
    latest_log = latest_audio.replace('audio_', 'key_log_').replace('.wav', '.csv')
    
    # Load the audio and key log
    audio_path = os.path.join(AUDIO_DIR, latest_audio)
    log_path = os.path.join(AUDIO_DIR, latest_log)
    
    # Load WAV file properly
    sample_rate, audio_data = wav_read(audio_path)
    audio_data = audio_data.astype(float) / 32768.0  # Normalize to [-1, 1]
    
    # Process the audio into spectrograms
    f, t, Sxx = signal.spectrogram(audio_data, sample_rate)
    Sxx_log = 10 * np.log10(Sxx + 1e-10)
    
    # Train the model
    model.train()
    total_loss = 0
    num_epochs = 10
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        with open(log_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                key = row['Key']
                timestamp = float(row['Timestamp'])
                # Get the spectrogram slice at this timestamp
                idx = int(timestamp * len(t))
                if idx < len(t):
                    spectrogram = Sxx_log[:, idx:idx+300]
                    spectrogram_tensor = torch.FloatTensor(spectrogram).unsqueeze(0)
                    label_idx = ord(key.lower()) - ord('a')
                    
                    optimizer.zero_grad()
                    output = model(spectrogram_tensor)
                    label_tensor = torch.tensor([label_idx], dtype=torch.long)
                    loss = criterion(output, label_tensor)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(key_log)
        total_loss += avg_epoch_loss
        
        # Save the model after each epoch
        model_path = os.path.join(MODELS_DIR, 'latest_model.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_epoch_loss,
        }, model_path)
    
    return jsonify({
        "status": "success",
        "loss": total_loss / num_epochs,
        "message": f"Model trained for {num_epochs} epochs"
    })

@app.route('/predict')
def predict():
    """Get prediction for the current audio input"""
    if not recording:
        return jsonify({"status": "error", "message": "Not recording"})
    
    # Get the latest audio buffer
    if not audio_buffer:
        return jsonify({"status": "error", "message": "No audio data available"})
    
    # Process the latest audio into a spectrogram
    latest_audio = np.concatenate(audio_buffer[-100:], axis=0)  # Use last 100 frames
    f, t, Sxx = signal.spectrogram(latest_audio, 44100)
    Sxx_log = 10 * np.log10(Sxx + 1e-10)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        spectrogram_tensor = torch.FloatTensor(Sxx_log).unsqueeze(0)
        output = model(spectrogram_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        
        # Convert predictions to list of {key, probability} objects
        predictions = []
        for i, prob in enumerate(probabilities):
            if prob > 0.1:  # Only include predictions with >10% probability
                key = chr(i + ord('a'))  # Convert index back to character
                predictions.append({
                    "key": key,
                    "probability": float(prob)
                })
        
        # Sort by probability
        predictions.sort(key=lambda x: x["probability"], reverse=True)
        
        return jsonify({
            "status": "success",
            "predictions": predictions[:5]  # Return top 5 predictions
        })

if __name__ == '__main__':
    app.run(debug=True)
