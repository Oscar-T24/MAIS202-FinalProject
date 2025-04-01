import numpy as np
import csv
import time
import sounddevice as sd
import threading
import scipy.io.wavfile as wav
from pynput import keyboard


def data_recording(audio_file, log_file, stop_flag):
    # Parameters for sound recording
    sample_rate = 44100  # Hz
    channels = 1  # Try stereo first
    audio_buffer = []  # Buffer to store audio data
    start_time_audio = None

    # Initialize the keystroke log file
    with open(log_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Key', 'Action', 'Timestamp'])

    def audio_callback(indata, frames, time, status):
        """Callback function for audio streaming"""
        if status:
            print(f"Audio callback status: {status}")
        audio_buffer.append(indata.copy())

    def record_audio():
        nonlocal start_time_audio
        print("Recording audio...")
        
        # Set the start time of the recording
        start_time_audio = time.time()
        
        try:
            # Try stereo recording first
            with sd.InputStream(samplerate=sample_rate, channels=channels, callback=audio_callback):
                while not stop_flag.is_set():
                    time.sleep(0.01)
        except Exception as e:
            print("Your computer does not support stereo recording. Defaulting to mono.")
            # Try mono recording
            with sd.InputStream(samplerate=sample_rate, channels=1, callback=audio_callback):
                while not stop_flag.is_set():
                    time.sleep(0.01)
        
        # When stopped, save the recorded audio
        if audio_buffer:
            audio_data = np.concatenate(audio_buffer, axis=0)
            wav.write(audio_file, sample_rate, audio_data)
            print(f"Audio saved to {audio_file}")
            print(f"Audio data shape: {audio_data.shape}")
        else:
            print("Warning: No audio data was recorded")
        
        print("Audio recording finished")

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
            stop_flag.set()  # Set flag to stop both recordings
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
    while not stop_flag.is_set():
        time.sleep(0.01)  # Prevent high CPU usage by sleeping briefly

    # Once stopped, both threads will finish
    audio_thread.join()
    keyboard_thread.join()
    print("Recording process finished.")

