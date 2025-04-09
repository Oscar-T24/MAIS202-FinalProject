
import soundfile as sf
import csv
import time
import threading
import sounddevice as sd
import numpy as np
from pynput import keyboard
import os
import librosa
import numpy as np
import csv
import time
import sounddevice as sd
import threading
import scipy.io.wavfile as wav
from pynput import keyboard
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import scipy.signal as signal
import csv
from scipy.ndimage import zoom
import os
import numpy as np
import csv
import time
import sounddevice as sd
import threading
import scipy.io.wavfile as wav
from pynput import keyboard


def set_keyboard_layout():
    DATA_DIR = input("Enter the name of the keyboard")
    try:
        os.mkdir(DATA_DIR)
    except FileExistsError:
        print(f"Warning: The directory {DATA_DIR} already exists.")

    return DATA_DIR


def data_recording():
    """
    Records both audio and keystrokes into a wav file and csv file respectively

    The recording starts when the function is called and ends when the ESC key is pressed
    """
    start_time_audio = None
    stop_recording = False
    # Parameters for sound recording
    sample_rate = 44100  # Hz
    channels = 2  # Try stereo first
    audio_buffer = []  # Buffer to store audio data

    # File paths

    DATA_DIR = set_keyboard_layout()
    
    audio_file = f'{DATA_DIR}/audio.wav'
    log_file = f'{DATA_DIR}/key_log.csv'

    # Initialize the keystroke log file
    with open(log_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Key', 'Action', 'Timestamp'])

    # Global variables

    def audio_callback(indata, frames, time, status):
        """Callback function for audio streaming"""
        if status:
            print(f"Audio callback status: {status}")
        audio_buffer.append(indata.copy())

    def record_audio():
        global start_time_audio, stop_recording
        print("Recording audio...")
        
        # Set the start time of the recording
        start_time_audio = time.time()
        
        try:
            # Try stereo recording first
            with sd.InputStream(samplerate=sample_rate, channels=channels, callback=audio_callback):
                while not stop_recording:
                    time.sleep(0.01)
        except Exception as e:
            print("Your computer does not support stereo recording. Defaulting to mono.")
            # Try mono recording
            with sd.InputStream(samplerate=sample_rate, channels=1, callback=audio_callback):
                while not stop_recording:
                    time.sleep(0.01)
        
        # When stopped, save the recorded audio
        if audio_buffer:
            audio_data = np.concatenate(audio_buffer, axis=0)
            wav.write(audio_file, sample_rate, audio_data)
            print(f"Audio saved to {audio_file}")
        
        print("Audio recording finished")

        # Keystroke listener function
    def on_press(key,debug=False):
            """"
            helper function to listen for keystrokes and record them on a csv file
            :key : keyboard key object
            """
            if start_time_audio is None:
                return  # Don't log if the audio hasn't started yet

            timestamp = time.time() - start_time_audio  # Calculate relative timestamp
            try:
                key_str = key.char  # Normal keys
            except AttributeError:
                key_str = str(key)  # Special keys like shift, ctrl, etc.

            # Log the key press with relative timestamp
            with open(log_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([key_str, 'Pressed', round(timestamp, 6)])  # Round for cleaner timestamp

            if debug:
                print(f"Key {key_str} Pressed at {timestamp:.6f} seconds")

    def on_release(key,debug=False):
            """"
            helper function to listen for keystrokes and record them on a csv file
            :key : keyboard key object
            """
            if start_time_audio is None:
                return  # Don't log if the audio hasn't started yet

            timestamp = time.time() - start_time_audio  # Calculate relative timestamp
            try:
                key_str = key.char
            except AttributeError:
                key_str = str(key)

            # Log the key release with relative timestamp
            with open(log_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([key_str, 'Released', round(timestamp, 6)])  # Round for cleaner timestamp

            if debug:
                print(f"Key {key_str} Released at {timestamp:.6f} seconds")

            # Stop listener if 'Esc' key is pressed
            if key == keyboard.Key.esc:
                global stop_recording
                stop_recording = True  # Set flag to stop both recordings
                return False

        # Start recording audio in a separate thread
    audio_thread = threading.Thread(target=record_audio)
    audio_thread.start()

        # Start the keyboard listener in the main thread to avoid blocking
    def start_keyboard_listener():
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()

        # Run the keyboard listener in a separate thread
    keyboard_thread = threading.Thread(target=start_keyboard_listener)
    keyboard_thread.start()

        # Wait for both threads to finish, while checking for stop condition
    while not stop_recording:
        time.sleep(0.01)  # Prevent high CPU usage by sleeping briefly

        # Once 'Esc' is pressed, both threads will finish
    audio_thread.join()
    keyboard_thread.join()
    print("Recording process finished.")

def data_processing(log_file:str) -> dict[str:list,str:float,str:float]:
    """
    Function to preprocess the keystroke data

    In particular, it will : 
    * remove the multiple presses of the same key
    * remove the releases of keys that weren't pressed
    * calculate the average keystroke duration (to optimize the BUFFER parameter)

    Using a stack to keep track of the keys that were pressed and the times at which they were pressed
    """
    #DATA_DIR = dataset#set_keyboard_layout()

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
                    # then add the time between the last release and this press into the list of in-between intervals
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
    
    averages_ = []
    for i in range(len(keystroke_times)-1):
        averages_.append(keystroke_times[i+1][1]-keystroke_times[i][2])
    
    print("Average time between two keys ", sum(averages_)/len(averages_))

    print("Average keystroke duration: ", sum(averages)/len(averages))

    return {"keystroke_times":keystroke_times,"average_duration":sum(averages)/len(averages),"average_interval":sum(averages_)/len(averages_)}


def create_spectrogram_and_numpy(audio_segment, dataset:str,extraction_method:str, key:str, idx:int,debug=False):
    """
    Generates the spectrogram using FFT or Mel, and returns a list of tuples containing the spectrograms
    """

    sample_rate = 44100

    NUMPY_OUTPUT_DIR = os.path.join(os.path.abspath(__file__),"dataset",dataset,"/numpy_arrays")
    OUTPUT_DIR = os.path.join(os.path.abspath(__file__),"dataset",dataset,"/keystroke_spectrogram")


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

            spectrogram_array = (mel_spect_stacked,key)
            
            if debug:
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
            
            spectrogram_array = (Sxx_stacked,key)

            if debug:
                numpy_array_path = os.path.join(NUMPY_OUTPUT_DIR, f"keystroke_{idx + 1}_{key}.npy")
                np.save(numpy_array_path, Sxx_stacked)

                plt.plot(np.array(range(len(audio_segment))),Sxx_stacked)
                plt.save(f"keystroke_spectrograms/keystroke_{idx + 1}_{key}")
            # Create time points for plotting
    else: 
            if extraction_method == "FFT":

                target_time_bins = 300

                f, t, Sxx = signal.spectrogram(audio_segment, sample_rate)
                Sxx_log = 10 * np.log10(Sxx + 1e-10)
                
                #time_zoom_factor = target_time_bins / Sxx_log.shape[1]
                #Sxx_resampled = zoom(Sxx_log, (1, time_zoom_factor), order=5)
                
                # Add channel dimension for mono audio
                Sxx_stacked = np.expand_dims(Sxx_log, axis=0)
                
                spectrogram_array = (Sxx_stacked,key)

                if debug: 
                    # Create time points for plotting
                    numpy_array_path = os.path.join(NUMPY_OUTPUT_DIR, f"keystroke_{idx + 1}_{key}.npy")

                    np.save(numpy_array_path, Sxx_stacked)

                    # Plot and save the spectrogram
                    plt.figure(figsize=(10, 4))
                    plt.pcolormesh(t, f, Sxx_log, shading='gouraud')
                    plt.ylabel('Frequency [Hz]')
                    plt.xlabel('Time [s]')
                    plt.colorbar(label='Log Power Spectral Density')
                    plt.title(f'Keystroke {idx + 1} - {key}')

                    spectrogram_path = os.path.join(OUTPUT_DIR, f"keystroke_{idx + 1}_{key}.png")
                    plt.savefig(spectrogram_path, dpi=300)
                    plt.close()

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
#
                plt.plot(mel_spect_stacked)
                plt.save(f"keystroke_spectrograms/keystroke_{idx + 1}_{key}")

    return spectrogram_array
    #print(f"Saved spectrogram for '{key}' at {spectrogram_path}")
    

def generate_spectrograms(BUFFER,dataset="demo",extraction_method="FFT"):
    """
    generates the spectrogram
    """

        # find the latest audio file
    audio_files = sorted([f for f in os.listdir(os.path.join("dataset",dataset)) if f.startswith('recording_') and f.endswith('.wav')])
    log_files = sorted([f for f in os.listdir(os.path.join("dataset",dataset)) if f.startswith('keylog_') and f.endswith('.csv')])

    AUDIO_FILE = os.path.join("dataset",dataset, audio_files[-1])
    LOG_FILE = os.path.join("dataset",dataset, log_files[-1])
        # set the audio directory as dataset

    stats = data_processing(LOG_FILE) # get the keystroke statistics

    keystroke_times = stats["keystroke_times"]

    sample_rate, audio_data = wav.read(AUDIO_FILE)

    print(f"{len(keystroke_times)}  keys to process")

    data = []

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
            
        feature,label = create_spectrogram_and_numpy(keystroke_audio,dataset,extraction_method, key, idx)

        data.append((feature,label))

    print("Processing complete. Spectrograms and NumPy arrays saved.")

    return data