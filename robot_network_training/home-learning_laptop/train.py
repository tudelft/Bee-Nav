import os
import random
import argparse
import yaml
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Grayscale
from tqdm import tqdm

from models.model_rgb import CompactCNN_rgb, InceptAttentionCNN_rgb
from dataset import GazeDataset
from generate_label import generate_label
from utils.reclinear_processor import RectilinearProcessor
import shutil

# Constants
SEED = 38
BATCH_SIZE = 4
# LEARNING_RATE = 1e-3 #9e-4
LEARNING_RATE = 9e-4
LOG_INTERVAL = 100

# Set random seed for reproducibility
torch.manual_seed(SEED)

class HorizontalShiftConcat:
    def __init__(self, shift_pixels):
        self.shift_pixels = shift_pixels

    def __call__(self, img):
        # Convert the image to a tensor
        img_tensor = transforms.ToTensor()(img)

        # Perform the horizontal shift
        shifted_img = torch.roll(img_tensor, shifts=self.shift_pixels, dims=2)

        # Concatenate the original image and the shifted image along the channel dimension
        six_channel_img = torch.cat((img_tensor, shifted_img), dim=0)

        return six_channel_img

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as config_file:
        return yaml.safe_load(config_file)

def get_network_and_transform(config):
    """Get the appropriate network and transform based on the configuration."""
    input_size = config['input_size']
    rgb = config['rgb']
    net_size = config['net_size']

    if rgb and input_size == '192x1800':
        if net_size == 'small':
            net = CompactCNN_rgb()
            suffix = '_rgb_small'
        elif net_size == 'inceptattention':
            net = InceptAttentionCNN_rgb()
            suffix = '_rgb_inceptattention'
        transform = transforms.Compose([
            transforms.Resize((192, 1800)),
            # HorizontalShiftConcat(900)
            transforms.ToTensor()
        ])
    else:
        raise ValueError("Invalid configuration for network and transform.")
    
    return net, transform, suffix

def preprocess_data(config, file_name, suffix):
    """Preprocess the data based on the configuration."""
    crop_type = config['crop_type']
    wind_correction = config['wind_correction']
    
    if crop_type == 'upper':
        suffix += '_upper'
    elif crop_type == 'full':
        suffix += '_full'
    elif crop_type == 'middle':
        suffix += '_middle'
    if wind_correction:
        suffix += '_windcorrectednew3'
    else:
        suffix += '_nowindcorrected'

    if config['additional_suffix'] != None:
        suffix += config['additional_suffix']

    preprocessed_data_path = f'./data/preprocessed_train/{file_name}{suffix}'
    preprocessor = RectilinearProcessor(
        config['training_data_path'], 
        config['training_data_path'] + '/data.csv', 
        preprocessed_data_path, 
        wind_correction=wind_correction, 
        crop_type=crop_type,
        input_size=config['input_size']
    )
    preprocessor.run()
    return preprocessed_data_path, suffix

def label_data(preprocessed_data_path):
    """Generate labels for the preprocessed data."""
    generate_label(preprocessed_data_path)

def train(net, dataloader, optimizer, epochs, log_file_path):
    """Train the network."""
    net.train()
    with open(log_file_path, 'w') as f:
        f.write('Iteration,Loss\n')
    for epoch in range(epochs):
        loss_100 = 0.0
        direction_loss_100 = 0.0
        distance_loss_100 = 0.0
        for i, data in enumerate(dataloader):
            inputs, labels = data

            optimizer.zero_grad()
            outputs = net(inputs)

            loss = F.mse_loss(outputs, labels[..., :2])

            # Compute predicted direction and distance
            pred_direction = torch.atan2(outputs[:, 1], outputs[:, 0])
            pred_distance = torch.sqrt(outputs[:, 0]**2 + outputs[:, 1]**2)

            # Compute target direction and distance
            target_direction = torch.atan2(labels[:, 1], labels[:, 0])
            target_distance = torch.sqrt(labels[:, 0]**2 + labels[:, 1]**2)

            # Compute direction and distance errors
            direction_error = F.mse_loss(pred_direction, target_direction)
            distance_error = F.mse_loss(pred_distance, target_distance)

            loss.backward()
            optimizer.step()

            loss_100 += loss.item()
            
            # if i % LOG_INTERVAL == LOG_INTERVAL - 1:
            #     print(f'[{i+1}, {i+1}] loss: {loss_100/LOG_INTERVAL:.3f}')
            #     with open(log_file_path, 'a') as f:
            #         f.write(f'{i+1},{loss_100/LOG_INTERVAL}\n')
            #     loss_100 = 0.0

            direction_loss_100 += direction_error.item()
            distance_loss_100 += distance_error.item()
            
            if i % LOG_INTERVAL == LOG_INTERVAL - 1:
                print(f'[{i+1}, {i+1}] Direction Loss: {direction_loss_100/LOG_INTERVAL:.3f}, Distance Loss: {distance_loss_100/LOG_INTERVAL:.3f}, Total Loss: {loss_100/LOG_INTERVAL:.3f}')
                with open(log_file_path, 'a') as f:
                    f.write(f'{i+1},{direction_loss_100/LOG_INTERVAL},{distance_loss_100/LOG_INTERVAL},{loss_100/LOG_INTERVAL}\n')
                loss_100 = 0.0
                direction_loss_100 = 0.0
                distance_loss_100 = 0.0

def main():
    config = load_config("./config.yaml")
    data_path = config['training_data_path']
    num_epochs = config['num_epochs']
    file_name = data_path.split('/')[-1]

    net, transform, suffix = get_network_and_transform(config)
    preprocessed_data_path, suffix = preprocess_data(config, file_name, suffix)
    label_data(preprocessed_data_path)

    dataset = GazeDataset(
        csv_file=preprocessed_data_path + f'/label_training_{file_name}{suffix}.csv',
        root_dir=preprocessed_data_path + f'/training_{file_name}{suffix}/',
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True) # drop_last=True to avoid batch size mismatch
    # dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    log_file_path = f'./training_logs/{file_name}{suffix}.csv'
    train(net, dataloader, optimizer, num_epochs, log_file_path)

    torch.save(net.state_dict(), f'./networks/gazenet_{file_name}{suffix}.pth')
    os.remove(preprocessed_data_path + f'/label_training_{file_name}{suffix}.csv')
    shutil.rmtree(preprocessed_data_path + f'/training_{file_name}{suffix}')
    print(f'Finished Training, network saved as {file_name}{suffix}.pth')

if __name__ == "__main__":
    main()