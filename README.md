## Our MAIS project proposal : an acoustic based keystroke transcriber using deep learning 

We plan to train our model on isolated key spectrograms

* To run our audio dataset constructor, have a look at  `audio_sampling.ipynb` which records keyboard press/release and computer audio input simultaneously for a period of 10s.

* Then, a short time fourrier transform (STFT) is performed on the data chunked into small buffers
  
* The spectrograms are saved

Example spectrogram for key "r" 

*Note here that the buffering size is 0.5s, but in the case of a fast typer we might reduce it so that our spectrogram does not contain parasite keys*

![keystroke_6_r](https://github.com/user-attachments/assets/114ad7b2-a44b-4006-8a10-9597e8fc58a9)

* Bets result so far from training a Dell mechanical keyboard
* 
![confusion_matrix_20250318_093204](https://github.com/user-attachments/assets/13ffa579-c202-4ca7-afe6-fa5e7e7efbc3)


![keystroke_102_g](https://github.com/user-attachments/assets/d10ed852-f746-474f-bb1b-47cf19ad4666)
![keystroke_109_s](https://github.com/user-attachments/assets/c48f4430-3a2f-47ba-a7c0-fc8fd2478f12)
