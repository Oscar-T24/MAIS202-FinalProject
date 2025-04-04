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
import numpy as np
import torch
import os
import torch.nn.functional as F  # Import F for padding
from datetime import datetime

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



