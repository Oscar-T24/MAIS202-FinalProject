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


def create_spectrogram_and_numpy(audio_segment, key, idx):
    # Generate the spectrogram using scipy
    print(f"Processing keystroke {idx} for key {key}")
    print(f"Audio segment shape: {audio_segment.shape}")

    sample_rate = 44100
    NUMPY_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audio', 'numpy_arrays')
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audio', 'keystroke_spectrograms')

    # Create directories if they don't exist
    os.makedirs(NUMPY_OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Saving to directory: {NUMPY_OUTPUT_DIR}")

    # Handle both single and multi-channel audio
    if len(audio_segment.shape) == 1:  # Single channel
        f, t, Sxx = signal.spectrogram(audio_segment, sample_rate)
        Sxx_log = 10 * np.log10(Sxx + 1e-10)
        
        # Resample to target time bins
        target_time_bins = 300
        time_zoom_factor = target_time_bins / Sxx_log.shape[1]
        Sxx_resampled = zoom(Sxx_log, (1, time_zoom_factor), order=5)
        
        # Add channel dimension for consistency
        Sxx_final = np.expand_dims(Sxx_resampled, axis=-1)
        Sxx_final = np.transpose(Sxx_final, (2, 0, 1))
        
    else:  # Multi-channel
        spectrograms = []
        target_time_bins = 300
        for channel in range(audio_segment.shape[1]):
            f, t, Sxx = signal.spectrogram(audio_segment[:, channel], sample_rate)
            Sxx_log = 10 * np.log10(Sxx + 1e-10)
            
            time_zoom_factor = target_time_bins / Sxx_log.shape[1]
            Sxx_resampled = zoom(Sxx_log, (1, time_zoom_factor), order=5)
            spectrograms.append(Sxx_resampled)
        
        # Stack spectrograms along a new axis and transpose for PyTorch format
        Sxx_final = np.stack(spectrograms, axis=-1)
        Sxx_final = np.transpose(Sxx_final, (2, 0, 1))
    
    # Save the numpy array
    numpy_array_path = os.path.join(NUMPY_OUTPUT_DIR, f"keystroke_{idx + 1}_{key}.npy")
    np.save(numpy_array_path, Sxx_final)
    print(f"Saved numpy array to: {numpy_array_path}")

    #print(f"Saved spectrogram for '{key}' at {spectrogram_path}")