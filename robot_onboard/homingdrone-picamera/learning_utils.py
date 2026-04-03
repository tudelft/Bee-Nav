import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
import cv2 
import numpy as np
import yaml
import random
import time 

# --- CONSTANTS ---
GAZE_STEP_DEGREES = 1
LEARNING_RATE = 9e-4

def center_gaze_direction(image, degree, gaze_org):
    """
    Centers the gaze direction in the image.
    """
    cols_per_degree = image.shape[1] // 360
    north_col = -cols_per_degree * gaze_org + image.shape[1] // 2
    center_col = degree * cols_per_degree + north_col
    if center_col >= image.shape[1]:
        center_col -= image.shape[1]

    left_col = center_col - image.shape[1] // 2
    output_image = np.concatenate((image[:, left_col:], image[:, :left_col]), axis=1)

    return output_image

def get_home_direction(pos_x, pos_y):
    """
    Calculates the home direction in degrees.
    """
    if pos_x < 0:
        return np.rad2deg(np.arctan(pos_y / pos_x))
    else:
        return np.rad2deg(min(np.pi + np.arctan(pos_y / pos_x), -np.pi + np.arctan(pos_y / pos_x), key=abs))

def deg_to_unit_vector(deg):
    """
    Converts degrees to a unit vector.
    """
    return np.array([np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))])

def get_relative_home_direction(home_direction, gaze):
    """
    Calculates the relative home direction based on the gaze.
    """
    gaze_matrix = np.array([
        [np.cos(np.deg2rad(gaze)), np.sin(np.deg2rad(gaze))],
        [-np.sin(np.deg2rad(gaze)), np.cos(np.deg2rad(gaze))]
    ])
    relative_home_direction = np.matmul(gaze_matrix, home_direction)
    return relative_home_direction

class GazeDataset(Dataset):
    GAZE_STEP_DEGREES = GAZE_STEP_DEGREES
    num_gaze_steps = 360 // GAZE_STEP_DEGREES
    num_augmentations = num_gaze_steps 
    
    def __init__(self, csv_file, root_dir, transform=None, ram_cache=None):
        self.csv_path = csv_file
        self.root_dir = root_dir
        self.transform = transform
        self.ram_cache = ram_cache if ram_cache is not None else {}
        self.to_tensor = transforms.ToTensor()
        self.data = []
        
        if os.path.exists(self.csv_path):
            try:
                df = pd.read_csv(self.csv_path)
                self.data = df.to_dict('records')
            except Exception as e:
                print(f"Warning: could not read CSV {self.csv_path}: {e}")

    def append_data(self, row_dict):
        self.data.append(row_dict)
        return len(self.data) - 1

    def __len__(self):
        return len(self.data) * self.num_augmentations
    
    def __getitem__(self, index):
        t0 = time.time() 

        # 1. Calculate base index and gaze degree
        base_index = index // self.num_augmentations
        gaze_step_index = index % self.num_augmentations
        gaze_degree = (gaze_step_index + 1) * self.GAZE_STEP_DEGREES

        try:
            row = self.data[base_index]
        except IndexError:
            row = self.data[0] 

        # 2. Get odometry info
        pos_x, pos_y = float(row['pos_x']), float(row['pos_y'])
        gaze_org = int((float(row['heading']) * 180 / np.pi) % 360)
        
        # 3. load image from RAM cache or disk
        t_ram_start = time.time()
        cache_key = row['recti_path']
        
        if cache_key in self.ram_cache:
            image_np = self.ram_cache[cache_key].copy()
        else:
            full_path = os.path.join(self.root_dir, row['recti_path'])
            image_np = cv2.imread(full_path)
            if image_np is None:
                image_np = np.zeros((192, 1800, 3), dtype=np.uint8)
        t_ram_end = time.time()

        # 4. Augmentation 
        t_aug_start = time.time()
        image_gaze = center_gaze_direction(image_np, gaze_degree, gaze_org)
        t_aug_end = time.time()
        
        # 5. Conversion
        t_conv_start = time.time()
        image_gaze = cv2.cvtColor(image_gaze, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_gaze)
            
        if self.transform:
            image_tensor = self.transform(image_pil)
        else:
            image_tensor = self.to_tensor(image_pil)
        t_conv_end = time.time()
            
        # 6. Labels
        home_angle_deg = get_home_direction(pos_x, pos_y)
        home_vector = deg_to_unit_vector(home_angle_deg)
        distance = np.sqrt(pos_x**2 + pos_y**2)
        
        label_vec = get_relative_home_direction(home_vector, gaze_degree)
        label_vec *= distance
        y_label = torch.tensor(label_vec, dtype=torch.float)

        # print timing info occasionally
        if random.random() < 0.05: 
            ram_ms = (t_ram_end - t_ram_start) * 1000
            aug_ms = (t_aug_end - t_aug_start) * 1000
            conv_ms = (t_conv_end - t_conv_start) * 1000
            total_ms = (t_conv_end - t0) * 1000
            print(f"    [TIMING-ITEM] Total: {total_ms:.1f}ms | RAM: {ram_ms:.1f}ms | Aug: {aug_ms:.1f}ms | Conv: {conv_ms:.1f}ms")

        return (image_tensor, y_label)


def load_config(config_path="./online_learning_config.yaml"):
    with open(config_path, 'r') as config_file:
        return yaml.safe_load(config_file)

def get_learning_components(config):
    from single_image_test import IncepAttentionCNN_rgb, CompactCNN_rgb
    if config['net_size'] == 'small':
        net = CompactCNN_rgb()
    elif config['net_size'] == 'inceptattention':
        net = IncepAttentionCNN_rgb()
    transform = transforms.Compose([
        transforms.Resize((192, 1800)),
        transforms.ToTensor()
    ])
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    return net, transform, optimizer, GazeDataset.num_augmentations