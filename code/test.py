from flask import Flask
import threading
import time
import numpy as np
import sounddevice as sd

app = Flask(__name__)

word = ""
is_playing = True

def play_sound(frequency):
    global is_playing
    sample_rate = 44100
    
    # Play sound while waiting for word to become "start"
    while is_playing and word != "start":
        t = np.linspace(0, 5, int(sample_rate * 5))  # 0.1 second duration
        wave = np.sin(2 * np.pi * 700 * t)
        sd.play(wave, sample_rate)
        sd.wait()
    
    # Then switch to 500Hz beeps
    while is_playing and word == "start":
        try:
            time.sleep(0.75)    

            t = np.linspace(0, 10, int(sample_rate * 10))  # 0.1 second duration
            wave = np.sin(2 * np.pi * 1500 * t)
            sd.play(wave, sample_rate)
            sd.wait()       

        except Exception as e:
            print(f"Error playing sound: {e}")
            break

def set_word():
    global word, is_playing
    is_playing = True
    time.sleep(5)
    word = "start"
    print("Word set to: ", word)

@app.route('/', methods=['GET'])
def trigger():
    global word
    return word, 200  # This will return "start" as the response

def run_flask():
    app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)

if __name__ == '__main__':
    # Start the sound playing thread
    sound_thread = threading.Thread(target=play_sound, args=(1000,))
    sound_thread.daemon = True  # Make thread daemon so it exits when main program exits
    sound_thread.start()
    
    # Start the word setting thread
    word_thread = threading.Thread(target=set_word)
    word_thread.daemon = True  # Make thread daemon so it exits when main program exits
    word_thread.start()
    
    # Run Flask in the main thread
    run_flask()