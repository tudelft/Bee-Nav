import numpy as np
import os
import cv2
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import torch

GAZE_STEP_DEGREES = 5   # Use 5 degrees (72 steps)

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

def generate_augmentations(preprocessed_pil_image, gaze_org, home_vector, distance, transform):
    """
    Takes one preprocessed PIL image and its metadata,
    and returns a list of all augmented (image_tensor, label_tensor) pairs.
    """
    
    # 72 steps
    num_gaze_steps = 360 // GAZE_STEP_DEGREES
    # 144 augmentations
    num_augmentations = num_gaze_steps
    
    augmentations = []
    

    
    # Convert to numpy once for fast rotation
    image_np = np.array(preprocessed_pil_image)

    for i in range(num_augmentations):
        gaze_step_index = i % num_gaze_steps
        
        # Calculate the degree: (0+1)*5=5, (1+1)*5=10, ..., (71+1)*5=360
        gaze_degree = (gaze_step_index + 1) * GAZE_STEP_DEGREES
        
        # A. Gaze Direction Shift (np.roll)
        image_gaze_np = center_gaze_direction(image_np, gaze_degree, gaze_org)
        image_pil = Image.fromarray(image_gaze_np)


        
        # C. Final transform (Resize, ToTensor)
        image_tensor = transform(image_pil)
            
        # D. Calculate the label on-the-fly
        label_vec = get_relative_home_direction(home_vector, gaze_degree)
        label_vec *= distance
        y_label = torch.tensor(label_vec, dtype=torch.float)
        
        augmentations.append((image_tensor, y_label))
        
    return augmentations

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

def apply_color_augmentation(image):
    """
    Applies slight color augmentation to the image.
    """
    alpha = np.random.uniform(0.9, 1.1)  # Random brightness adjustment
    beta = np.random.uniform(-10, 10)    # Random contrast adjustment
    augmented_image = image.copy()
    augmented_image = cv2.convertScaleAbs(augmented_image, alpha=alpha, beta=beta)
    return augmented_image

def generate_label_csv(rectilinear_path, image_info):
    """
    Generates a new CSV file with all information needed for 
    on-the-fly augmentation. DOES NOT WRITE IMAGES.
    """
    data_name = rectilinear_path.name
    output_csv = rectilinear_path / f'label_training_{data_name}.csv'

    with output_csv.open('w') as f:
        f.write('filename,pos_x,pos_y,gaze_org,home_vector_x,home_vector_y,distance\n')
        
        for file in tqdm(rectilinear_path.iterdir()):
            if not file.suffix == '.jpg':
                continue

            path_idx = '_'.join(file.stem.split('_')[:-1])

            try:
                row = image_info.loc[path_idx]
            except KeyError:
                continue

            pos_x = row['pos_x']
            pos_y = row['pos_y']
            heading_org = row['heading']
            gaze_org = int((heading_org * 180 / np.pi) % 360)

            home_angle_deg = get_home_direction(pos_x, pos_y)
            home_vector = deg_to_unit_vector(home_angle_deg)
            distance = np.sqrt(pos_x**2 + pos_y**2)
            
            # We just write the *base* image info. The 720 augmentations
            # will be created *virtually* by the Dataset.
            f.write(f"{file.name},{pos_x},{pos_y},{gaze_org},{home_vector[0]},{home_vector[1]},{distance}\n")

def generate_label(rectilinear_path):
    """
    Generates labels for the images in the given path, including augmented images.
    """
    rectilinear_path = Path(rectilinear_path)
    data_name = rectilinear_path.name
    data_csv = rectilinear_path / 'data.csv'
    image_info = pd.read_csv(data_csv)

    output_csv = rectilinear_path / f'label_training_{data_name}.csv'
    output_location = rectilinear_path / f'training_{data_name}'

    output_location.mkdir(parents=True, exist_ok=True)

    with output_csv.open('w') as f:
        f.write('filename, gaze, pos_x, pos_y, label_1, label_2\n')
        for file in tqdm(rectilinear_path.iterdir()):
            if not file.suffix == '.jpg':
                continue
            img = cv2.imread(str(file))

            path_idx = '_'.join(file.stem.split('_')[:-1])

            pos_x = image_info.loc[image_info['path_idx'] == path_idx, 'pos_x'].values[0]
            pos_y = image_info.loc[image_info['path_idx'] == path_idx, 'pos_y'].values[0]
            heading_org = image_info.loc[image_info['path_idx'] == path_idx, 'heading'].values[0]
            gaze_org = int((heading_org * 180 / np.pi) % 360)

            home_angle_deg = get_home_direction(pos_x, pos_y)
            home_vector = deg_to_unit_vector(home_angle_deg)

            for gaze in range(1, 361):
                image_gaze = center_gaze_direction(img, gaze, gaze_org)

                label = get_relative_home_direction(home_vector, gaze)
                distance = np.sqrt(pos_x**2 + pos_y**2)
                label *= distance

                filename_gaze = f"{path_idx}_{gaze:03d}.jpg"
                cv2.imwrite(str(output_location / filename_gaze), image_gaze)
                f.write(f"{filename_gaze},{gaze},{pos_x},{pos_y},{label[0]},{label[1]}\n")

                # Generate augmented image
                augmented_image = apply_color_augmentation(image_gaze)
                filename_gaze_aug = f"{path_idx}_{gaze:03d}_caug.jpg"
                cv2.imwrite(str(output_location / filename_gaze_aug), augmented_image)
                f.write(f"{filename_gaze_aug},{gaze},{pos_x},{pos_y},{label[0]},{label[1]}\n")
