import argparse
import os
import sys
import numpy as np
import cv2
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.pop() 
sys.path.append(os.getcwd())

from mushroom_body.memory_own import MushroomBody
from mushroom_body.process_images_gaby import create_pn, rotate_image

from mushroom_body.run_mushroom_learning import train_network

# --- Helper Functions ---

def parse_tuple_string(s):
    s = s.replace('"', '').replace('(', '').replace(')', '')
    return [float(x) for x in s.split(',')]

def load_data(csv_path, image_folder):
    data = []
    with open(csv_path, 'r') as f:
        lines = f.readlines()[1:] # Skip header
        for line in lines:
            reader = csv.reader([line])
            parts = next(reader)
            
            filename = parts[0]
            pos_str = parts[1]
            rot_str = parts[2]
            
            pos = np.array(parse_tuple_string(pos_str))
            rot = np.array(parse_tuple_string(rot_str))
            
            image_path = os.path.join(image_folder, filename)
            
            data.append({
                'image_path': image_path,
                'pos': pos,
                'yaw': rot[2], 
                'filename': filename
            })
    return data

def get_home_direction(pos, home_pos, yaw):
    to_home = home_pos[:2] - pos[:2]
    rad = np.deg2rad(yaw)
    heading = np.array([np.cos(rad), np.sin(rad)])
    cross = heading[0] * to_home[1] - heading[1] * to_home[0]
    cross = round(cross, 2)
    
    if cross > 0: return 'Left'
    elif cross < 0: return 'Right'
    else: return 'Center'

# --- Main Debug Logic ---

def run_multi_image_test(args, training_ratio=0.80):
    # 1. Load Data
    print(f"Loading data from {args.csv_path}...")
    full_data = load_data(args.csv_path, args.image_folder)
    home_pos = np.array(args.home_pos)

    # 2. Select Subset of Images
    # We take the first N images based on arguments
    num_images = len(full_data)
    n_training = int(num_images * training_ratio)
    # randomly select n_training images, storing their indices
    indices = np.random.choice(len(full_data), n_training, replace=False)
    training_data = [full_data[i] for i in indices]
    # retrieve the remaining indices for testing
    test_indices = [i for i in range(len(full_data)) if i not in indices]
    testing_data = [full_data[i] for i in test_indices]
    
    print("\n" + "="*60)
    print(f"DEBUG: Processing {len(training_data)} Images")
    for i, item in enumerate(training_data):
        print(f"  Image {i+1}: {item['filename']} at {item['pos']}")
    print("="*60 + "\n")

    # 3. Initialize Mushroom Body (ONCE - it will learn all images)
    print("Initializing Mushroom Body...")
    
    # Use the first image to determine input size
    first_img = cv2.imread(training_data[0]['image_path'])
    if first_img is None: raise ValueError("Could not read first image")
    pn_sample, _ = create_pn(first_img, show_images = True)
    pn_size = len(pn_sample)
    
    seeds = [5, 6, 7, 8]
    mbon_nb = 2 
    mbs = [MushroomBody(PN_nb=pn_size, KC_nb=5000, MBON_nb=2, KCtoPN_synapses=4, KC_norm_param=0.01, seed=seeds[i], verbose=False) for i in range(mbon_nb)]

    # # ---------------------------------------------------------
    # # 4. TRAINING PHASE (Accumulate knowledge)
    # # ---------------------------------------------------------
    print(">>> PHASE 1: TRAINING ALL IMAGES")
    step_size = 5
    
    # Outer loop: Iterate through the images
    for idx, item in enumerate(training_data):
        print(f"  Training Image {idx+1}/{len(training_data)}: {item['filename']}")
        
        base_img = cv2.imread(item['image_path'])
        if base_img is None: continue

        # Inner loop: Rotate and Learn
        for angle in tqdm(range(0, 360, step_size), desc=f"    Rotations Img {idx+1}", leave=False):
            rotated_img = rotate_image(base_img, angle)
            
            # Virtual heading & Label
            virtual_yaw = item['yaw'] + angle
            direction = get_home_direction(item['pos'], home_pos, virtual_yaw)

            print(f"    Angle {angle}: {direction}")
            
            # PN
            pn, _ = create_pn(rotated_img)
            
            # Refresh & Learn
            for mb in mbs:
                mb.refresh(pn)
                if direction == 'Right':
                    pre_nonzero_weights = mb.get_nonzero_KCtoMBON_weights(mbon_index=0)
                    mb.learn(mbon_index=0)
                    post_nonzero_weights = mb.get_nonzero_KCtoMBON_weights(mbon_index=0)
                    print(f"MBON 0 Weights before: {pre_nonzero_weights}, after: {post_nonzero_weights}")
                elif direction == 'Left':
                    pre_nonzero_weights = mb.get_nonzero_KCtoMBON_weights(mbon_index=1)
                    mb.learn(mbon_index=1)
                    post_nonzero_weights = mb.get_nonzero_KCtoMBON_weights(mbon_index=1)
                    print(f"MBON 1 Weights before: {pre_nonzero_weights}, after: {post_nonzero_weights}")
                elif direction == 'Center':
                    pre_nonzero_weights_0 = mb.get_nonzero_KCtoMBON_weights(mbon_index=0)
                    pre_nonzero_weights_1 = mb.get_nonzero_KCtoMBON_weights(mbon_index=1)
                    mb.learn(mbon_index=0)
                    mb.learn(mbon_index=1)
                    post_nonzero_weights_0 = mb.get_nonzero_KCtoMBON_weights(mbon_index=0)
                    post_nonzero_weights_1 = mb.get_nonzero_KCtoMBON_weights(mbon_index=1)
                    print(f"MBON 0 Weights before: {pre_nonzero_weights_0}, after: {post_nonzero_weights_0}")
                    print(f"MBON 1 Weights before: {pre_nonzero_weights_1}, after: {post_nonzero_weights_1}")

    # data = load_data(args.csv_path, args.image_folder)
    # home_pos = np.array([0, 0, 1.5])
    # mbs = train_network(data, home_pos)
    print("\nTraining complete.\n")
    
    # ---------------------------------------------------------
    # 5. TESTING PHASE (Test each one individually)
    # ---------------------------------------------------------
    print(">>> PHASE 2: TESTING INDIVIDUAL IMAGES")
    
    # Outer loop: Iterate through the images again to test recall
    for idx, item in enumerate(training_data):
        print("\n" + "#"*80)
        print(f"TEST RESULTS FOR IMAGE {idx+1}: {item['filename']}")
        print(f"Pos: {item['pos']}")
        print("#"*80)
        
        base_img = cv2.imread(item['image_path'])
        
        # Header
        header = f"{'Ang':<4} | {'True Dir':<8} | {'Signal':<8} | {'MB1_R':<8} {'MB1_L':<8} | {'MB2_R':<8} {'MB2_L':<8} | {'Pred'}"
        print(header)
        print("-" * 110)

        # Inner loop: Rotate and Test
        for angle in range(0, 360, step_size):
            rotated_img = rotate_image(base_img, angle)
            
            # Ground Truth
            virtual_yaw = item['yaw'] + angle
            true_direction = get_home_direction(item['pos'], home_pos, virtual_yaw)
            
            # Neural Net Process (No Learning here!)
            pn, _ = create_pn(rotated_img)
            
            familiarities = []
            for mb in mbs:
                familiarities.append(mb.get_familiarity(pn))
            
            mb1_r = familiarities[0][0]
            mb1_l = familiarities[0][1]
            mb2_r = familiarities[1][0]
            mb2_l = familiarities[1][1]
            
            # Signal
            total_left = mb1_l + mb2_l
            total_right = mb1_r + mb2_r
            signal = total_left - total_right
            
            # Prediction
            pred = "Center"
            if signal > 0.001: pred = "Left"
            if signal < -0.001: pred = "Right"
            
            match_marker = ""
            if pred == true_direction: match_marker = "OK"
            elif true_direction == "Center": match_marker = "~"
            else: match_marker = "FAIL"
            
            print(f"{angle:<4} | {true_direction:<8} | {signal:>8.4f} | "
                  f"{mb1_r:>8.4f} {mb1_l:>8.4f} | "
                  f"{mb2_r:>8.4f} {mb2_l:>8.4f} | {match_marker}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="./forest/forest_40_trees_20x20_area_0_bee_noisy/dataset_navigation.csv")
    parser.add_argument("--image_folder", type=str, default="./forest/forest_40_trees_20x20_area_0_bee_noisy/Replicator/rgb")
    parser.add_argument("--home_pos", type=float, nargs=3, default=[0.0, 0.0, 1.5])
    args = parser.parse_args()
    run_multi_image_test(args)