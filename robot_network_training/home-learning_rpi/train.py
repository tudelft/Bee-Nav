#!/usr/bin/python3
import os
import argparse
import yaml
import random
import pickle
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# --- CONSTANTS ---
BATCH_SIZE = 4
LEARNING_RATE = 9e-4
GAZE_STEP_DEGREES = 5
SEED = 42
LOG_INTERVAL = 50  # New: Log metrics every 100 steps

# --- Set Random Seeds ---
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ----------------------------------------------------------------------------
# 1. HELPER FUNCTIONS
# ----------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------
# 2. DATASET CLASS
# ----------------------------------------------------------------------------

class GazeDataset(Dataset):
    GAZE_STEP_DEGREES = GAZE_STEP_DEGREES
    num_gaze_steps = 360 // GAZE_STEP_DEGREES
    num_augmentations = num_gaze_steps
    
    def __init__(self, csv_file, root_dir, transform=None):
        self.csv_path = csv_file
        self.root_dir = root_dir
        self.transform = transform
        self.to_tensor = transforms.ToTensor()
        
        try:
            self.annotations = pd.read_csv(self.csv_path)
            print(f"✅ GazeDataset loaded {len(self.annotations)} base images.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find CSV file: {self.csv_path}")

    def __len__(self):
        return len(self.annotations) * self.num_augmentations
    
    def __getitem__(self, index):
        base_index = index // self.num_augmentations
        gaze_step_index = index % self.num_gaze_steps
        gaze_degree = (gaze_step_index + 1) * self.GAZE_STEP_DEGREES

        row = self.annotations.iloc[base_index]
        pos_x, pos_y = float(row['pos_x']), float(row['pos_y'])
        gaze_org = int((float(row['heading']) * 180 / np.pi) % 360)

        home_angle_deg = get_home_direction(pos_x, pos_y)
        home_vector = deg_to_unit_vector(home_angle_deg)
        distance = np.sqrt(pos_x**2 + pos_y**2)
        
        img_path = os.path.join(self.root_dir, row['recti_path'])
        try:
            image = Image.open(img_path) 
        except FileNotFoundError:
            # Return dummy if file not found
            return torch.zeros(3, 192, 1800), torch.zeros(2), -1, -1

        image_np = np.array(image)
        image_gaze = center_gaze_direction(image_np, gaze_degree, gaze_org)
        image_pil = Image.fromarray(image_gaze)
            
        if self.transform:
            image_tensor = self.transform(image_pil)
        else:
            image_tensor = self.to_tensor(image_pil)
            
        label_vec = get_relative_home_direction(home_vector, gaze_degree)
        label_vec *= distance
        y_label = torch.tensor(label_vec, dtype=torch.float)

        # UPDATED: Return base_index and gaze_degree for logging
        return image_tensor, y_label, base_index, gaze_degree

# ----------------------------------------------------------------------------
# 3. MODEL DEFINITIONS
# ----------------------------------------------------------------------------

class CompactCNN_rgb(nn.Module):
    def __init__(self):
        super(CompactCNN_rgb, self).__init__()
        self.conv1 = nn.Conv2d(3, 2, kernel_size=5, stride=4, padding=2)
        self.conv2 = nn.Conv2d(2, 2, kernel_size=5, stride=4, padding=2)
        self.conv3 = nn.Conv2d(2, 2, kernel_size=12, stride=4)
        self.conv4 = nn.Conv2d(2, 2, kernel_size=1, stride=4)
        self.fc = nn.Linear(2*1*7, 2)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = torch.tanh(self.conv3(x))
        x = torch.tanh(self.conv4(x))
        x = x.view(-1, 1*7*2)
        x = self.fc(x)
        return x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class InceptionModule2(nn.Module):
    def __init__(self, in_channels):
        super(InceptionModule2, self).__init__()
        self.branch5x5_1 = nn.Conv2d(in_channels, 4, kernel_size=5, stride=4, padding=2)
        self.branch3x3dbl_2 = nn.Conv2d(in_channels, 4, kernel_size=3, stride=2, padding=1)
        self.branch3x3dbl_3 = nn.Conv2d(4, 4, kernel_size=3, stride=2, padding=1)
        self.branch_pool = nn.Conv2d(in_channels, 2, kernel_size=1)
        self.branch_dilated = nn.Conv2d(in_channels, 4, kernel_size=3, stride=4, padding=2, dilation=2)
        self.spatial_attention = SpatialAttention()
    def forward(self, x):
        branch5x5 = self.branch5x5_1(x)
        branch3x3dbl = self.branch3x3dbl_2(x)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=4, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        branch_dilated = self.branch_dilated(x)
        outputs = torch.cat([branch5x5, branch3x3dbl, branch_pool, branch_dilated], dim=1)
        attention_map = self.spatial_attention(outputs)
        return outputs * attention_map

class InceptAttentionCNN_rgb(nn.Module):
    def __init__(self):
        super(InceptAttentionCNN_rgb, self).__init__()
        self.inception1 = InceptionModule2(3)
        self.inception2 = InceptionModule2(14)
        self.conv3 = nn.Conv2d(14, 8, kernel_size=5, stride=2, padding=2)
        self.conv4 = nn.Conv2d(8, 4, kernel_size=6, stride=1)
        self.fc = nn.Linear(4*1*52, 16)
        self.fc2 = nn.Linear(16, 2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None: init.constant_(m.bias, 0)
    def forward(self, x):
        x = torch.tanh(self.inception1(x))
        x = torch.tanh(self.inception2(x))
        x = torch.tanh(self.conv3(x))
        x = torch.tanh(self.conv4(x))
        x = x.view(-1, 4 * 1 * 52) 
        x = torch.tanh(self.fc(x))
        x = self.fc2(x)
        return x

# ----------------------------------------------------------------------------
# 4. PREPROCESSING & MAIN
# ----------------------------------------------------------------------------

def load_config(config_path):
    with open(config_path, 'r') as config_file: return yaml.safe_load(config_file)

def get_network_and_transform(config):
    input_size = config['input_size']
    rgb = config['rgb']
    net_size = config['net_size']
    print(f"Model: {net_size}, RGB: {rgb}, Input: {input_size}")

    if rgb and input_size == '192x1800':
        if net_size == 'small': net = CompactCNN_rgb()
        elif net_size == 'inceptattention': net = InceptAttentionCNN_rgb()
        else: raise ValueError(f"Unknown net_size {net_size}")
        transform = transforms.Compose([transforms.Resize((192, 1800)), transforms.ToTensor()])
    else:
        raise ValueError("Invalid config")
    
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    return net, transform, optimizer

class RectilinearProcessor:
    def __init__(self, file_location, csv_location, output_location, wind_correction, crop_type, input_size):
        self.file_location = file_location
        self.csv_location = csv_location
        self.output_location = output_location
        self.wind_correction = wind_correction
        self.crop_type = crop_type
        self.input_size = input_size
        self.df = pd.read_csv(csv_location)
        os.makedirs(output_location, exist_ok=True)

    def _load_mask(self):
        try:
            with open("./utils/mask.pkl", "rb") as f:
                mask_file = pickle.load(f)
            return mask_file['mask'], mask_file['x'], mask_file['y'], mask_file['r']
        except:
            return np.ones((1232, 1640), dtype=np.uint8)*255, 1640//2, 1232//2, 1232//2

    def _wind_correct(self, image, pitch, roll):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.2, 1000, param1=50, param2=30, minRadius=240, maxRadius=243)
        if circles is not None:
            x, y, _ = np.round(circles[0, :]).astype("int")[0]
            return int(x), int(y)
        return int(12.9546*pitch - 130.7885*roll + 835.85), int(120.2679*pitch + 14.7156*roll + 643.05)

    def _process_image(self, image, file_id, x, y, r):
        image = cv2.linearPolar(image, (x, y), r, cv2.WARP_FILL_OUTLIERS)
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        image = np.flipud(image)
        image = cv2.resize(image[840:], (1800, 192))
        image = np.fliplr(image)
        image = cv2.resize(image[15:165, :], (1800, 192))
        image = np.concatenate((image[:, 450:], image[:, :450]), axis=1) # Rotate
        
        if self.crop_type == 'upper':
            h, w = image.shape[:2]
            image = cv2.resize(image[h//2:], (w, h))
        
        cv2.imwrite(os.path.join(self.output_location, f"{file_id}_preprocessed.jpg"), image)

    def run(self):
        try: shutil.copy(self.csv_location, os.path.join(self.output_location, "data.csv"))
        except: pass
        mask, dx, dy, dr = self._load_mask()
        for entry in tqdm(list(os.scandir(self.file_location)), desc="Preprocessing"):
            if not entry.name.endswith('.jpg'): continue
            fid = entry.name.split('.')[0]
            try: row = self.df.loc[self.df['path_idx'] == fid].iloc[0]
            except: continue
            img = cv2.imread(entry.path)
            if img is None: continue
            
            x, y = dx, dy
            if self.wind_correction: x, y = self._wind_correct(img, row['pitch'], row['roll'])
            self._process_image(img, fid, x, y, dr)

def preprocess_data(config, file_name, suffix):
    crop_type = config['crop_type']
    wind = config['wind_correction']
    suffix += '_upper' if crop_type == 'upper' else ''
    suffix += '_windcorrected' if wind else '_nowind'
    if config['additional_suffix']: suffix += config['additional_suffix']
    
    out_path = f'./data/preprocessed_train/{file_name}{suffix}'
    if os.path.exists(out_path) and len(os.listdir(out_path)) > 1:
        print(f"Found existing data: {out_path}")
        return out_path, suffix
        
    print("Preprocessing...")
    proc = RectilinearProcessor(config['training_data_path'], config['training_data_path']+'/data.csv', out_path, wind, crop_type, config['input_size'])
    proc.run()
    return out_path, suffix

def label_data(prep_path, suffix):
    path = Path(prep_path)
    out_csv = path / f'label_training_{path.name}.csv'
    if out_csv.exists(): return str(out_csv)
    
    base_df = pd.read_csv(path / 'data.csv')
    base_df['id'] = base_df['path_idx'].apply(lambda x: x.split('_')[0])
    base_df = base_df.set_index('id')
    
    with out_csv.open('w') as f:
        f.write('recti_path,pos_x,pos_y,heading\n')
        imgs = sorted(list(path.glob('*_preprocessed.jpg')), key=lambda x: int(x.name.split('_')[0]))
        for img in tqdm(imgs, desc="Labeling"):
            fid = img.name.split('_')[0]
            try:
                row = base_df.loc[fid]
                f.write(f"{img.name},{row['pos_x']},{row['pos_y']},{row['heading']}\n")
            except: continue
    return str(out_csv)

def main():
    # START TIMER (Script Start)
    t_start_total = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="./config.yaml")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # 1. Prepare Data
    print("--- Preparing Data ---")
    t_prep_start = time.time()
    
    data_path, suffix = preprocess_data(config, config['training_data_path'].split('/')[-1], "")
    csv_path = label_data(data_path, suffix)
    
    t_prep_end = time.time()
    
    # 2. Setup
    print("--- Setup ---")
    
    # Define save directory EARLY so we can log into it
    save_dir = f'./offline_training_results/{config["training_data_path"].split("/")[-1]}{suffix}'
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup CSV Loggers
    log_steps_path = os.path.join(save_dir, "training_steps.csv")
    log_metrics_path = os.path.join(save_dir, "training_metrics.csv")
    
    print(f"Logging indices to: {log_steps_path}")
    print(f"Logging metrics to: {log_metrics_path}")
    
    with open(log_steps_path, 'w') as f:
        f.write("step,image_index,rotation_angle\n")
        
    with open(log_metrics_path, 'w') as f:
        f.write("step,direction_loss,distance_loss,total_loss\n")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, transform, optimizer = get_network_and_transform(config)
    net.to(device)
    
    dataset = GazeDataset(csv_path, data_path, transform)
    
    # Standard Shuffling
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    print(f"Total Training Items: {len(dataset)}")
    
    # 3. Train Loop
    print("--- Training (1 Epoch) ---")
    t_train_start = time.time()
    
    net.train()
    total_loss = 0.0
    
    # New Accumulators
    acc_loss_100 = 0.0
    acc_dir_loss_100 = 0.0
    acc_dist_loss_100 = 0.0
    
    steps = 0
    
    pbar = tqdm(dataloader)
    
    # UPDATED Loop to unpack indices and angles
    for inputs, labels, img_indices, rot_angles in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float16 if torch.cuda.is_available() else torch.bfloat16):
            outputs = net(inputs)
            
            # 1. Cast to float32 for geometric calc
            outputs_f32 = outputs.to(torch.float32)
            labels_f32 = labels.to(torch.float32)

            # 2. Main Loss
            loss = F.mse_loss(outputs_f32, labels_f32)
            
            # 3. Geometric Calculations
            # Compute predicted direction and distance
            pred_direction = torch.atan2(outputs_f32[:, 1], outputs_f32[:, 0])
            pred_distance = torch.sqrt(outputs_f32[:, 0]**2 + outputs_f32[:, 1]**2)

            # Compute target direction and distance
            target_direction = torch.atan2(labels_f32[:, 1], labels_f32[:, 0])
            target_distance = torch.sqrt(labels_f32[:, 0]**2 + labels_f32[:, 1]**2)

            # Compute errors
            direction_error = F.mse_loss(pred_direction, target_direction)
            distance_error = F.mse_loss(pred_distance, target_distance)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate metrics
        acc_loss_100 += loss.item()
        acc_dir_loss_100 += direction_error.item()
        acc_dist_loss_100 += distance_error.item()
        
        # LOGGING STEP (Indices)
        with open(log_steps_path, 'a') as f:
            for i in range(len(img_indices)):
                f.write(f"{steps},{img_indices[i].item()},{rot_angles[i].item()}\n")

        # LOGGING METRICS (Every LOG_INTERVAL)
        if (steps + 1) % LOG_INTERVAL == 0:
            avg_loss = acc_loss_100 / LOG_INTERVAL
            avg_dir = acc_dir_loss_100 / LOG_INTERVAL
            avg_dist = acc_dist_loss_100 / LOG_INTERVAL
            
            # Console Log
            tqdm.write(f'[{steps+1}, {steps+1}] Direction Loss: {avg_dir:.3f}, Distance Loss: {avg_dist:.3f}, Total Loss: {avg_loss:.3f}')
            
            # File Log
            with open(log_metrics_path, 'a') as f:
                f.write(f'{steps+1},{avg_dir},{avg_dist},{avg_loss}\n')
            
            # Reset
            acc_loss_100 = 0.0
            acc_dir_loss_100 = 0.0
            acc_dist_loss_100 = 0.0

        total_loss += loss.item()
        steps += 1
        pbar.set_postfix({'loss': loss.item()})
    
    t_train_end = time.time()
        
    # 4. Save Model
    save_path = os.path.join(save_dir, 'gazenet_offline.pth')
    
    torch.save(net.state_dict(), save_path)
    print(f"✅ Model saved to: {save_path}")
    
    t_end_total = time.time()
    
    # --- WRITE TIMING STATS ---
    log_file = os.path.join(save_dir, "timing_stats.txt")
    
    prep_time = t_prep_end - t_prep_start
    train_time = t_train_end - t_train_start
    total_time = t_end_total - t_start_total
    
    stats = (
        f"--- Performance Report ---\n"
        f"Data: {len(dataset)} items (Batch: {BATCH_SIZE})\n"
        f"Preprocessing Time: {prep_time:.2f} sec\n"
        f"Training Time:      {train_time:.2f} sec\n"
        f"Total Execution:    {total_time:.2f} sec\n"
        f"Avg Step Time:      {(train_time / steps * 1000):.1f} ms\n"
        f"Final Avg Loss:     {(total_loss / steps):.4f}\n"
    )
    
    print("\n" + stats)
    with open(log_file, "w") as f:
        f.write(stats) 

if __name__ == "__main__":
    main()