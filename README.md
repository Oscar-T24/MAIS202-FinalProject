## Our MAIS project proposal : an acoustic based keystroke transcriber using deep learning 

The training relies on isolated keys

* To run our audio dataset constructor, have a look at `data_recording.py` which provides the necessary methods to record both audio channel microphone input and keyboard press/releases simultaneously. 

* Then, a short time fourrier transform (STFT) is performed on the data chunked into small buffers. The STFT is applied to a short time frame between a key's press to its release, and a correction that ensures each window fits in a consistent time interval of 0.3s. 

* Data is preprocessed in the following manner
- Overlapping keys are removed with a pair comparison using a stack that iterates over the entire set of keys pressed
- Keys that were pressed and never released (and vice versa) are omitted

*The spectrograms are generated in following manner : 
- The STFT is applied on a time "frame", then all the frames are stacked to form a 2D matrix.
  
* The spectrograms are exported to numpy arrays of shape (129,300,`channel`) with `channel` the number of channels (mono = 1, stereo = 2, etc). 

* The model 

Example spectrogram for key "r" 

*Note here that the buffering size is 0.5s, but in the case of a fast typer we might reduce it so that our spectrogram does not contain parasite keys*

![keystroke_6_r](https://github.com/user-attachments/assets/114ad7b2-a44b-4006-8a10-9597e8fc58a9)

* Bets result so far from training a Dell mechanical keyboard
* 
![confusion_matrix_20250318_093204](https://github.com/user-attachments/assets/13ffa579-c202-4ca7-afe6-fa5e7e7efbc3)


![keystroke_102_g](https://github.com/user-attachments/assets/d10ed852-f746-474f-bb1b-47cf19ad4666)

![keystroke_109_s](https://github.com/user-attachments/assets/c48f4430-3a2f-47ba-a7c0-fc8fd2478f12)

# TODO ! 

Account for data imbalance : when there is an imbalance in the keys of the training dataset (one key is disproportionately present) apply some weighting to correct imbalance
