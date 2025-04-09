import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import string
import re
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
import os
import torch.nn.functional as F  # Import F for padding
from datetime import datetime
import numpy as np
import sounddevice as sd
import librosa
import torch
import torch.nn.functional as F
from scipy.signal import find_peaks
from queue import Queue
import time
from data_processing import create_spectrogram_and_numpy

matplotlib.use('Agg')  # Use a non-interactive backend

BASE_SIZE = torch.Size([1,129,300])
#DATA_DIR = "dell_test"
#NUMPY_DIR = DATA_DIR + "/numpy_arrays"

alphabet = "abcdefghijklmnopqrstuvwxyz"
char2idx = {char: idx for idx, char in enumerate(alphabet)}


def directory_setting():
    """
    sets up the workflow for the project by removing numpy arrays (*.py) and adding that to .gitignore 
    If run with the GPU server, also changes to the working directory (/home/test)
    """
    try:
        os.chdir('/home/test')
        os.listdir()
    except:
        print("Could not change directory to /home/test")
        print("If you intended to run the code locally, you can skip this message")

    gitignore_path = "../.gitignore"
    if os.path.exists(gitignore_path):
        # Read current content
        with open(gitignore_path, "r") as f:
            content = f.read()
        
        # Only append if *.npy is not already in the file
        if "*.npy" not in content:
            with open(gitignore_path, "a") as f:
                f.write("\n*.npy")  # Add newline before appending
    else:
        # Create new .gitignore if it doesn't exist
        with open(gitignore_path, "w") as f:
            f.write("*.npy")


    DATA_DIR = input("Enter the name of the directory")
    if not os.path.exists(DATA_DIR):
        print(f"Error: The directory {DATA_DIR} does not exist.")

# Padding function
def pad_tensor(tensor, target_width):
    current_width = tensor.shape[2]
    if current_width < target_width:
        padding = target_width - current_width
        return F.pad(tensor, (0, padding))  # Pad the time dimension
    elif current_width > target_width:
        return tensor[:, :, :target_width]  # Truncate to target width
    return tensor

# Function to load all spectrograms from the directory
def load_spectrograms_from_directory(directory:str) -> tuple[list,list,int]:
    """
    Loads the spectrogram tensors from the numpy directory into a list (features)
    Extracts the key (labels)
    Returns the maximum width (frequency)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spectrograms = []
    keys = []
    widths = []
    filenames = os.listdir(directory)

    # Loop through each file in the directory
    for filename in filenames:
        if filename.endswith('.npy'):
            file_path = os.path.join(directory, filename)

            # Load the NumPy array from file
            spectrogram = np.load(file_path)

            widths.append(spectrogram.shape[1])

            try: 
                key = re.search(r"keystroke_\d+_([A-Za-z])\.npy", filename).group(1)
            except AttributeError:
                print(f"Error: Could not extract key from filename: {filename}. Make sure it is a valid key (aA-zZ)")
                continue

            # Convert the spectrogram to a PyTorch tensor
            spectrogram_tensor = torch.tensor(spectrogram).float().to(device)

            if spectrogram_tensor.shape != BASE_SIZE:
                print(f"Warning : Spectrogram tensor shape {spectrogram_tensor.shape} does not match the expected size {BASE_SIZE}")
                continue
            
            #spectrogram_tensor = spectrogram_tensor.unsqueeze(0)

            
            # Rearrange dimensions from [80, 13, 4] to [4, 80, 13]
            #spectrogram_tensor = spectrogram_tensor.permute(2, 0, 1)
            
            # Pad or truncate the time dimension to match BASE_SIZE[2] (300)
            """"
            if spectrogram_tensor.shape[2] < BASE_SIZE[2]:
                # Pad with zeros
                padding = BASE_SIZE[2] - spectrogram_tensor.shape[2]
                spectrogram_tensor = F.pad(spectrogram_tensor, (0, padding))
            else:
                # Truncate to desired length
                spectrogram_tensor = spectrogram_tensor[:, :, :BASE_SIZE[2]]

            if spectrogram_tensor.shape != BASE_SIZE:
                print(f"Warning: Spectrogram tensor shape {spectrogram_tensor.shape} does not match the expected size {BASE_SIZE}")
            """

            # Add the tensor to the list
            spectrograms.append(spectrogram_tensor)
            keys.append(key)

    print(f"loaded {len(spectrograms)} spectrograms")
    assert len(spectrograms) == len(keys), "The number of spectrograms and keys do not match!"
    return spectrograms, keys, max(widths) 


## Need to pad the spectrograms to the same width


# Apply padding to all spectrograms
"""
for i in range(len(spectrogram_tensors)):
        if (padded := pad_tensor(spectrogram_tensors[i], 300)) is not None:
            spectrogram_tensors[i] = padded
"""

def key_to_idx_tensors(labels:list[str]) -> list[torch.Tensor]:
    """
    converts a list of string characters (like 'a', 'b' etc) into a list of integer tensors
    """
    label_indices = [char2idx[char] for char in labels]
    label_tensor = torch.tensor(label_indices, dtype=torch.long) # Can we use int imstead ? 
    return label_tensor


class KeystrokeDataset(Dataset):
    def __init__(self, spectrograms_tensors, labels):
        self.spectrograms = spectrograms_tensors
        self.labels = labels

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        return self.spectrograms[idx], self.labels[idx]


class KeystrokeCNN(nn.Module):
    def __init__(self,input_height=129, input_width=300, num_classes=26,input_channels=1):
        super(KeystrokeCNN, self).__init__()

        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.num_classes = num_classes
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)  # (1, H, W) -> (32, H, W)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # (32, H, W) -> (64, H, W)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # (64, H, W) -> (128, H, W)

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)  # Reduces dimensions by half (H/2, W/2)

        # Calculate the final feature map size dynamically
        self._to_linear = self._get_conv_output_size(input_height, input_width)

        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 512)  
        self.fc2 = nn.Linear(512, num_classes)  # 26 output classes (A-Z)
        #list(string.ascii_lowercase) for generating the output classes

    def _get_conv_output_size(self, height, width):
        """Pass a dummy tensor to determine final feature map size after convolutions"""
        x = torch.zeros(1, self.input_channels, height, width)  # Batch size = 1, 1 channel, (H, W)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x.numel()  # Flattened size

    def forward(self, x):

        batch_size = x.shape[0] # extract the batch size (first dimension)
        dummy_tensor = torch.Tensor(batch_size, 1, self.input_height, self.input_width)

        try: 
            assert dummy_tensor.shape == x.shape, f"The input tensors do not match the model's parameters (width/height) \n Expected {dummy_tensor.shape} but got {x.shape}"
        except AssertionError as e: 
            print(e)
            print("skipping incorrect vector")
    
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
       # print("After conv1+pool:", x.shape)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
       # print("After conv2+pool:", x.shape)
        
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
       # print("After conv3+pool:", x.shape)
        
        x = x.view(x.size(0), -1)
        #print("After flattening:", x.shape)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model
#model = KeystrokeCNN()

def train(model:KeystrokeCNN, spectrogram_tensors:list[torch.Tensor], label_tensor:list[torch.Tensor],lr:float, epochs:int,dataset:str,BUFFER:float,batch_size=64,debug=False)->tuple[list[float],str]: #:list[torch.Tensor] #:list[torch.Tensor]
    """
    Trains the model based on a set of features and labels

    Inputs : 
    model : instance of KeystrokeCNN CNN model
    spectrogram_tensor : a list of spectrogram (tensors)
    label_tensors : a list of labels (tensors)
    lr : the learning rate
    epochs : the number of epochs
    BUFFER (experimental) : adjust the buffer time of audio
    batch_size : number of samples used in one iteration of gradient descent

    Outputs: 

    the list of losses
    """
    train_dataset = KeystrokeDataset(spectrogram_tensors,label_tensor)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)

    model.train()
    losses = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        epoch_losses = []
        
        for i, (inputs, labels) in enumerate(train_loader, 0):
            optimizer.zero_grad()
            
            try:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                epoch_losses.append(loss.item())
                
                if i % 10 == 9:  # Print every 10 batches
                    print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 10:.3f}")
                    running_loss = 0.0
                    
            except Exception as e:
                print("Input tensors may not be of the right shape", e)
                raise e
        
        # Store average loss for this epoch
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_epoch_loss)
        print(f"Epoch {epoch + 1} average loss: {avg_epoch_loss:.3f}")
        """Save model after each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_epoch_loss,
        }, f"{save_path}_epoch_{epoch+1}.pt")
        """
    filename = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'dataset',
    dataset,
    "model_"+datetime.now().strftime("%Y%m%d_%H%M%S")+'.pt'
    )
    # Save final model

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': losses[-1],
        'epochs': epochs, 
    }, filename)

    if debug:
        plt.plot(range(epochs), losses)
        plt.title(f'Training Loss for {epochs} epochs, {model} model, {criterion} loss function, and {lr} learning rate')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.savefig(f'training_loss_{BUFFER}.png', dpi=300, bbox_inches='tight')
        #plt.show()
        
    return losses, filename

# Training call
#losses, filename = train(model, train_loader, criterion, optimizer, 30,"macbook")

#conv_layers = [layer for layer in model.modules() if isinstance(layer, nn.Conv2d)]

""""
for layer in conv_layers:
    # Get weights for this specific layer
    weights = layer.weight.data  # Shape: (out_channels, in_channels, kernel_size, kernel_size)
    
    # Calculate grid size based on number of output channels
    n_filters = layer.out_channels
    n_cols = 8  # Fixed number of columns
    n_rows = (n_filters + n_cols - 1) // n_cols  # Calculate rows needed
    
    # Create figure with appropriate size
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 2*n_rows))
    axes = axes.ravel()
    
    # Plot each filter
    for i in range(n_filters):
        # Get the first channel of the filter (since we want to visualize 2D)
        filter_weights = weights[i, 0].cpu().numpy()
        
        # Plot the filter
        axes[i].imshow(filter_weights, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Filter {i+1}')
    
    # Hide empty subplots if any
    for i in range(n_filters, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Filters of {layer.__class__.__name__} layer (in_channels={layer.in_channels}, out_channels={layer.out_channels})')
    plt.tight_layout()
    #plt.show()
"""

class LiveKeystrokeDetector:
    """
    The class in charge of handling live detection 

    :params : threshold : parameter used to finetune the ambiant background noise to avoid advert triggering of keystroke detection
    """
    def __init__(self, model, sample_rate=44100, window_size=0.2, threshold=0.9999999):
        self.model = model
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.threshold = threshold
        self.window_samples = int(window_size * sample_rate)
        self.audio_buffer = np.zeros(self.window_samples)
        self.prediction_queue = Queue()
        self.is_recording = False
        self.debounce_time = time.time()
        self.letters = "abcdefghijklmnopqrstuvwxyz"
        self.threshold = 75
        self.prominence = 200

    def set_threshold(self,threshold,prominence):
        if int(threshold) > 100 or int(threshold) < 0:
            return None
        self.threshold = int(threshold)
        self.prominence = int(prominence)
        return True
        
    def audio_callback(self, indata, frames, time_info, status):
        """Callback function for audio streaming"""
        if status:
            print(f"Audio callback status: {status}")
            
        # Update buffer with new audio data
        self.audio_buffer = np.roll(self.audio_buffer, -frames)
        self.audio_buffer[-frames:] = indata.flatten()

        # use the create_spectrogram function to retrive numpy spectrogram

        spectrogram_array = create_spectrogram_and_numpy(self.audio_buffer,"demo","FFT",None,None)[0]
        
        """
        # Generate mel spectrogram
        mel_spect = librosa.feature.melspectrogram(
            y=self.audio_buffer,
            sr=self.sample_rate,
            n_mels=80,
            n_fft=2048,
            hop_length=512,
            window='hann',
            power=2.0
        )

        # Convert to log scale (dB)
        mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)

        # Normalize to 0-1 range
        mel_spect_norm = (mel_spect_db - mel_spect_db.min()) / (mel_spect_db.max() - mel_spect_db.min())

        # Convert to tensor
        spectrogram_tensor = torch.FloatTensor(mel_spect_norm)
        
        # Pad or truncate to 300 in the last dimension
        current_length = spectrogram_tensor.shape[-1]
        if current_length < 300:
            # Pad to 300
            padding = (0, 300 - current_length)
            spectrogram_tensor = F.pad(spectrogram_tensor, padding, mode='constant', value=0)
        elif current_length > 300:
            # Center crop to 300
            start = (current_length - 300) // 2
            spectrogram_tensor = spectrogram_tensor[..., start:start+300]

        """
        
        # Get the energy over time
        energy = np.sum(spectrogram_array[0], axis=0)  # shape: (time_bins,)

        peaks, properties = find_peaks(
        energy,
        height=np.percentile(energy, self.threshold),
        distance=10,
        prominence=self.prominence
        )   

        # need to finetune 
        
        if len(peaks) > 0 and time.time() - self.debounce_time > 1:
            print("Keystroke detected")
            # Make prediction
            with torch.no_grad():
                spectrogram_tensor = torch.FloatTensor(spectrogram_array).unsqueeze(0) # add batch size
                if spectrogram_tensor.shape[-1] < 300:
                    padding = (0, 300 - spectrogram_tensor.shape[-1])
                    spectrogram_tensor = F.pad(spectrogram_tensor,padding,"constant",0) # pad with zeros
                output = self.model(spectrogram_tensor)  # Forward pass
                probabilities = F.softmax(output, dim=1)  # Convert logits to probabilities
                """
                predicted_idx = torch.argmax(probabilities, dim=1).item()  # Get class index
                
                # Convert index to letter
                letters = "abcdefghijklmnopqrstuvwxyz"
                predicted_letter = letters[predicted_idx]
                
                self.prediction_queue.put(predicted_letter)
                print(f"Predicted key: {predicted_letter}")
                """
                # Version 2 : put the top k keys into a list 
                topk_probs, topk_indices = torch.topk(probabilities, k=10, dim=1)

                topk_indices = topk_indices.squeeze().tolist()
                topk_probs = topk_probs.squeeze().tolist()

                topk_letters = [(self.letters[i],round(prob,2)) for i,prob in zip(topk_indices,topk_probs)]

                print(f"Top prediction is {topk_letters[0]}")

                self.prediction_queue.put(topk_letters[:10]) # appends a tuple to the queue (key,normalised probability)

            # Add a delay to prevent multiple detections for the same keystroke
            self.debounce_time = time.time()
    
    def start_recording(self):
        """Start recording and processing audio"""
        self.is_recording = True
        print("Starting live keystroke detection...")
        
        try:
            with sd.InputStream(samplerate=self.sample_rate, 
                              channels=1, 
                              callback=self.audio_callback):
                while self.is_recording:
                    time.sleep(0.01)  # Prevent high CPU usage
                    
        except Exception as e:
            print(f"Error in audio recording: {e}")
            self.is_recording = False
    
    def stop_recording(self):
        """Stop recording"""
        self.is_recording = False
        print("Stopping keystroke detection...") 