# import wandb
from data_loader import get_data_loaders, inspect_dataset
from simple_network import SimpleCNN, CompactCNN_rgb, IncepAttentionCNN_rgb
from insect_utils.train_utils import train, test, evaluate_on_dataset, evaluate
from insect_utils.plot_utils import make_nice_plot
import json
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from matplotlib import pyplot as plt

def load_config(config_file="config/config/config_training.json"):
    with open(config_file, "r") as file:
        config = json.load(file)
    return config

def train_model(config = None, only_training = False, append_name_evaluation_figure=''):
    if config is None:
        config = load_config()

    # finish any existing run and start a new one:
    # wandb.finish()

    # try:
    #     # wandb.init(project="VisualHoming", config=config, reinit=True)
    # except Exception as e:
    #     print(e)

    # Create model and data loaders
    if config['model_type'] == 'simple':
        model = SimpleCNN()
    elif config['model_type'] == 'compact':
        model = CompactCNN_rgb()
    elif config['model_type'] == 'attention':
        model = IncepAttentionCNN_rgb()
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])

    train_loader, val_loader, test_loader, position_map, rotation_map, noisy_position_map, target_map, train_indices, val_indices, test_indices = \
        get_data_loaders(config)
    
    # Train and test
    model_filename = train(model, train_loader, val_loader, criterion, optimizer, config)
    
    if not only_training:
        evaluate(model, train_loader, position_map, rotation_map, noisy_position_map, target_map, train_indices, append_name=append_name_evaluation_figure)
        print("Performance on training set:")
        test(model, train_loader, criterion)
        print("Performance on validation set:")
        test(model, val_loader, criterion)
        print("Performance on test set:")
        test(model, test_loader, criterion)
        evaluate_on_dataset(model, config['evaluation_dataset_folder'], config['evaluation_image_folder'], config['evaluation_csv_filename'], config=config)
    
    return model_filename

def test_model(model_filename='models/model.pth', config=None, append_name_evaluation_figure=''):
    
    if config == None:
        # Load the config parameters:
        config = load_config()

    # load the model from the pth file:
    if config['model_type'] == 'simple':
        model = SimpleCNN()
    elif config['model_type'] == 'compact':
        model = CompactCNN_rgb()
    elif config['model_type'] == 'attention':
        model = IncepAttentionCNN_rgb()
    model.load_state_dict(torch.load(model_filename))
    
    train_loader, val_loader, test_loader, position_map, rotation_map, noisy_position_map, target_map, train_indices, val_indices, test_indices = \
        get_data_loaders(config)

    # Evaluate and plot predictions on the test set:
    # check if the dataset_folder exists:
    if not os.path.exists(config['dataset_folder']):
        print("Dataset folder does not exist. Please check the path.")
    else:
        evaluate_on_dataset(model, config['dataset_folder'], config['image_folder'], config['csv_filename'], config = config, \
                        load_indices = True, evaluation_set='test', augment = True)

    # Test the model:
    criterion = nn.MSELoss()
    print("Performance on training set:")
    test(model, train_loader, criterion)
    print("Performance on validation set:")
    test(model, val_loader, criterion)
    print("Performance on test set:")
    test(model, test_loader, criterion, graphics = False)

    if not os.path.exists(config['evaluation_dataset_folder']):
        print("Dataset folder does not exist. Please check the path.")
        predictions = None
        ground_truths = None
        position_map = None
        rotation_map = None
        noisy_position_map = None
    else:
        # Evaluate and plot predictions on a separate evaluation set:

        predictions, ground_truths, position_map, rotation_map, noisy_position_map = \
            evaluate_on_dataset(model, config['evaluation_dataset_folder'], config['evaluation_image_folder'], \
                                config['evaluation_csv_filename'], config = config, \
                                    append_name_evaluation_figure=append_name_evaluation_figure, graphics=True)
    
        # Determine the distance and angle errors for positions within 10 meters from the nest:
        d_errs = []
        a_errs = []
        for i in range(len(predictions)):
            if np.linalg.norm(ground_truths[i]) < 10:
                distance_error = np.linalg.norm(predictions[i]) - np.linalg.norm(ground_truths[i])
                angle_error = np.arccos(np.clip(np.dot(predictions[i], ground_truths[i]) / (np.linalg.norm(predictions[i]) * np.linalg.norm(ground_truths[i])), -1.0, 1.0))
                d_errs.append(distance_error)
                a_errs.append(angle_error)
        d_errs = np.array(d_errs)
        a_errs = np.array(a_errs)
        a_errs = np.rad2deg(a_errs)  # Convert angle errors to degrees
        
        plt.figure(figsize=(5, 5))
        plt.hist(d_errs, bins=50, alpha=0.5, label='Distance Errors')
        plt.xlabel('Distance Error (m)')
        plt.ylabel('Frequency')
        plt.savefig('distance_errors.svg')
        plt.close()

        plt.figure(figsize=(5, 5))
        plt.hist(a_errs, bins=50, alpha=0.5, label='Angle Errors')
        plt.xlabel('Angle Error (radians)')
        plt.ylabel('Frequency')
        plt.savefig('angle_errors.svg')
        plt.close()

    return predictions, ground_truths, position_map, rotation_map, noisy_position_map

if __name__ == "__main__":

    mode = 'train' # 'train', 'test', 'inspect_dataset'

    if mode == 'train':
        train_model(only_training=True)
    elif mode == 'test':
        test_model(model_filename='models/model_4trees.pth')
    elif mode == 'inspect_dataset':
        model = SimpleCNN()
        model_filename='models/model.pth'
        model.load_state_dict(torch.load(model_filename))
        config = load_config()
        inspect_dataset(config, model = model)
