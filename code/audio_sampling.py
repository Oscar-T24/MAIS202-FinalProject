# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +

import csv
import numpy as np
from scipy.io.wavfile import write as wav_write
import os
import librosa


import scipy.io.wavfile as wav
import scipy.signal as signal
import csv
from scipy.ndimage import zoom
import os

# Parameters


# Read keystroke data from CSv

def data_processing(dataset):
    """
    Function to preprocess the keystroke data

    In particular, it will : 
    * remove the multiple presses of the same key
    * remove the releases of keys that weren't pressed
    * calculate the average keystroke duration (to optimize the BUFFER parameter)

    Using a stack to keep track of the keys that were pressed and the times at which they were pressed
    """


    DATA_DIR = dataset

    log_file = f'{DATA_DIR}/key_log.csv'

    averages = []

    total_keystrokes = 0
    # Read keystroke timestamps from CSV
    keystroke_times = []
    with open(log_file, "r") as audio_data_file:
        reader = csv.reader(audio_data_file)
        stack = {}  # Dictionary to store key press times
        key_order = []  # List to track order of key presses
        
        # Skip the header row
        next(reader)
        
        # Count total rows (excluding header)
        rows = list(reader)
        total_keystrokes = len(rows)
        
        # Reset file pointer to after header
        audio_data_file.seek(0)
        next(reader)  # Skip header again
        
        for row in reader:
            key = row[0]
            action = row[1]
            timestamp = float(row[2])

            if action == "Pressed":
                if key in stack:
                    # Ignore multiple presses of the same key
                    continue
                stack[key] = [key, timestamp]
                key_order.append(key)

            elif action == "Released":
                if key not in stack:
                    # Ignore releases of keys that weren't pressed
                    continue
                
                # Only process the release if it's the most recently pressed key
                if key == key_order[-1]:
                    stack[key].append(timestamp)
                    keystroke_times.append(stack[key])
                    del stack[key]
                    key_order.pop()
                else:
                    # If releasing a key that wasn't the last pressed, skip it
                    continue

    print("Summary of the keystroke data : ")
    print(f"Total valid keystrokes: {len(keystroke_times)}")
    print(f"Total invalid keystrokes: {total_keystrokes//2 - len(keystroke_times)}")  #each key represents a press / release pair so divide by two
    print("Keystroke times:")
    for key, press, release in keystroke_times:
        #print(f"Key: {key}, Press: {press:.3f}, Release: {release:.3f}, Duration: {release-press:.3f}")
        averages.append(release-press)

    print("Average keystroke duration: ", sum(averages)/len(keystroke_times))

    return keystroke_times, sum(averages)/len(keystroke_times)

# Function to create and save the spectrogram and numpy arrays

def create_spectrogram_and_numpy(audio_segment, dataset,extraction_method, key, idx):
    # Generate the spectrogram using scipy

    sample_rate = 44100
    NUMPY_OUTPUT_DIR = dataset + "/numpy_arrays"
    OUTPUT_DIR = dataset + "/keystroke_spectrograms"

    if len(audio_segment.shape) > 1:  # If it's multi-channel
        #print("channel dimension",audio_segment.shape)
        if extraction_method == "mel":
            mel_specs = []
            for channel in range(audio_segment.shape[1]):  # Process each channel
                mel_spec = librosa.feature.melspectrogram(
                    y=audio_segment[:, channel],
                    sr=sample_rate,
                    n_mels=80,
                    n_fft=2048,
                    hop_length=512,
                    window='hann',
                    power=2.0
                )
                # Convert to log scale (dB) and normalize
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
                mel_specs.append(mel_spec_norm)
            
            # Stack the spectrograms along a new axis
            # Shape will be (n_mels, time_steps, channels)
            mel_spect_stacked = np.stack(mel_specs, axis=-1)
            
            # Save the stacked spectrograms as a NumPy array
            numpy_array_path = os.path.join(NUMPY_OUTPUT_DIR, f"keystroke_{idx + 1}_{key}.npy")
            np.save(numpy_array_path, mel_spect_stacked)
            #print(f"Saved 4D NumPy array for '{key}' at {numpy_array_path}")
        elif extraction_method == "FFT":
            spectrograms = []
            target_time_bins = 300
            for channel in range(audio_segment.shape[1]):
                f, t, Sxx = signal.spectrogram(audio_segment[:, channel], sample_rate)
                Sxx_log = 10 * np.log10(Sxx + 1e-10)
                
                time_zoom_factor = target_time_bins / Sxx_log.shape[1]
                Sxx_resampled = zoom(Sxx_log, (1, time_zoom_factor), order=5)
                spectrograms.append(Sxx_resampled)
            
            # Stack spectrograms along a new axis and transpose for PyTorch format
            Sxx_stacked = np.stack(spectrograms, axis=-1)
            Sxx_stacked = np.transpose(Sxx_stacked, (2, 0, 1))
            
            numpy_array_path = os.path.join(NUMPY_OUTPUT_DIR, f"keystroke_{idx + 1}_{key}.npy")
            np.save(numpy_array_path, Sxx_stacked)
            # Create time points for plotting
        else: 
            if extraction_method == "FFT":
                f, t, Sxx = signal.spectrogram(audio_segment, sample_rate)
                Sxx_log = 10 * np.log10(Sxx + 1e-10)
                
                time_zoom_factor = target_time_bins / Sxx_log.shape[1]
                Sxx_resampled = zoom(Sxx_log, (1, time_zoom_factor), order=5)
                
                # Add channel dimension for mono audio
                Sxx_stacked = np.expand_dims(Sxx_resampled, axis=0)
                
                # Create time points for plotting
                numpy_array_path = os.path.join(NUMPY_OUTPUT_DIR, f"keystroke_{idx + 1}_{key}.npy")
                np.save(numpy_array_path, Sxx_stacked)

            elif extraction_method == "mel":

                
                mel_spec = librosa.feature.melspectrogram(
                    y=audio_segment[:, 1],
                    sr=sample_rate,
                    n_mels=80,
                    n_fft=2048,
                    hop_length=512,
                    window='hann',
                    power=2.0
                )
                
                mel_spect_stacked = np.stack(mel_spec, axis=-1)

                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
                mel_specs.append(mel_spec_norm)

                # Stack the spectrograms along a new axis
                # Shape will be (n_mels, time_steps, channels)
                mel_spect_stacked = np.stack(mel_specs, axis=-1)
                
                # Save the stacked spectrograms as a NumPy array
                numpy_array_path = os.path.join(NUMPY_OUTPUT_DIR, f"keystroke_{idx + 1}_{key}.npy")
                np.save(numpy_array_path, mel_spect_stacked)


    #print(f"Saved spectrogram for '{key}' at {spectrogram_path}")
    

def generate_spectrograms(BUFFER,dataset,extraction_method):

    keystroke_times, average_keystroke_duration = data_processing(dataset)

    audio_file = f'{dataset}/aligned_iphone.wav'
    log_file = f'{dataset}/key_log.csv'
    AUDIO_FILE = audio_file
    KEYSTROKE_CSV = log_file
    OUTPUT_DIR = dataset + "/keystroke_spectrograms"
    NUMPY_OUTPUT_DIR = dataset + "/numpy_arrays"  # New directory for NumPy arrays
    sample_rate, audio_data = wav.read(AUDIO_FILE)

    # process each keystroke by sampling each key with press / release times
    for idx, (key, press_time, release_time) in enumerate(keystroke_times):
        # Use exact press and release times without buffer
        start_time = max(0, press_time - BUFFER)  # Ensure we don't go before 0
        end_time = min(len(audio_data) / sample_rate, release_time + BUFFER)  # Ensure we don't go beyond audio length
        
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        # Extract audio segment
        keystroke_audio = audio_data[start_sample:end_sample]
        
        if len(keystroke_audio) == 0:
            print(f"Warning: Empty audio segment for keystroke {idx + 1}")
            continue
            
        create_spectrogram_and_numpy(keystroke_audio,dataset,extraction_method, key, idx)

    print("Processing complete. Spectrograms and NumPy arrays saved.")

    return average_keystroke_duration

#generate_spectrograms(audio_data=None,keystroke_times=None,sample_rate=None,BUFFER=0.01)
#generate_spectrograms(keystroke_times,tim)