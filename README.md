## Our MAIS project proposal : an acoustic based keystroke transcriber using deep learning 

We plan to train our model on isolated key spectrograms

* To run our audio dataset constructor, have a look at  `audio_sampling.ipynb` which records keyboard press/release and computer audio input simultaneously for a period of 10s.

* Then, a short time fourrier transform (STFT) is performed on the data chunked into small buffers
  
* The spectrograms are saved

Example spectrogram for key "r" 

![keystroke_6_r](https://github.com/user-attachments/assets/114ad7b2-a44b-4006-8a10-9597e8fc58a9)
