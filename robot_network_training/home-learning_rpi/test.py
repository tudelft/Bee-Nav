import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import Grayscale
import os
import pandas as pd
from models.model_rgb import CompactCNN_rgb, InceptAttentionCNN_rgb
import tqdm
import yaml
from train import load_config, get_network_and_transform
from utils.reclinear_processor import RectilinearProcessor

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

    preprocessed_data_path = f'./data/preprocessed_test/{file_name}{suffix}'
    preprocessor = RectilinearProcessor(
        config['testing_data_path'], 
        config['testing_data_path'] + '/data.csv', 
        preprocessed_data_path, 
        wind_correction=wind_correction, 
        crop_type=crop_type,
        input_size=config['input_size']
    )
    preprocessor.run()
    return preprocessed_data_path, suffix

def load_model(model_path, net):
    """Load the model from the specified path."""
    net.load_state_dict(torch.load(model_path))
    net.eval()

def predict(img_path, transform, net):
    """Predict the output for a given image."""
    image = Image.open(img_path)

    image = transform(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        output = net(image)
        prediction = torch.atan2(output[0][1], output[0][0]).item()
        distance = torch.norm(output).item()
    
    return prediction, distance

def update_predictions(preprocessed_data_path, net, model, transform):
    """Update the predictions in the CSV file."""
    file_path = preprocessed_data_path
    csv_path = os.path.join(preprocessed_data_path, 'data.csv')

    image_info = pd.read_csv(csv_path)
    image_info[f'{model}_output'] = 0
    image_info[f'{model}_distance'] = 0

    print('Predicting...')

    for file in os.listdir(file_path):
        if file.endswith('.jpg'):
            img_path = os.path.join(file_path, file)
            pred, conf = predict(img_path, transform, net)
            i = file.split('_')[0]
            date = file.split('_')[1]
            time = file.split('_')[2].split('.')[0]
            i = f"{i}_{date}_{time}"

            idx = image_info[image_info['path_idx'] == i].index[0]
            image_info.at[idx, f'{model}_output'] = pred
            image_info.at[idx, f'{model}_distance'] = conf

    image_info.to_csv(csv_path, index=False)
    print('Done')

def main():
    config = load_config("./config.yaml")    
    file_name = config['testing_data_path'].split('/')[-1]
    net, transform, suffix = get_network_and_transform(config)
    load_model(config['network_path'], net)
    net_name = config['network_path'].split('/')[-1].split('.')[0]
    preprocessed_data_path, suffix = preprocess_data(config, file_name, suffix)
    update_predictions(preprocessed_data_path, net, net_name, transform)

if __name__ == "__main__":
    main()
