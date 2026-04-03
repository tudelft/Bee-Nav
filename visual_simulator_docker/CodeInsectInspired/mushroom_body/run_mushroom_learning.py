import argparse
import os
import sys
import numpy as np
import cv2
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import random

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.pop() # Go up one level to include project root
sys.path.append(os.getcwd())

from mushroom_body.memory_own import MushroomBody
from mushroom_body.process_images_gaby import create_pn, rotate_image
from mushroom_body.rgb_to_fisheye import rgb_to_fisheye
from mushroom_body.cropper import crop_and_zoom_process


def parse_tuple_string(s):
    # Remove quotes and parentheses
    s = s.replace('"', '').replace('(', '').replace(')', '')
    return [float(x) for x in s.split(',')]

def load_data(csv_path, image_folder):
    """
    Loads data from the CSV file.
    Returns a list of dictionaries containing image path and metadata.
    """
    data = []
    with open(csv_path, 'r') as f:
        lines = f.readlines()[1:] # Skip header
        for line in lines:
            # Simple split by comma might fail if there are commas inside quotes
            # But here the format seems consistent: filename, "pos", "rot", [target]
            # We can use csv module or manual parsing. Manual is fine if careful.
            # Let's use csv module for robustness against quoted strings
            reader = csv.reader([line])
            parts = next(reader)
            
            filename = parts[0]
            pos_str = parts[1]
            rot_str = parts[2]
            
            pos = np.array(parse_tuple_string(pos_str))
            rot = np.array(parse_tuple_string(rot_str))
            
            # Construct full image path
            image_path = os.path.join(image_folder, filename)
            
            data.append({
                'image_path': image_path,
                'pos': pos,
                'yaw': rot[2], # Assuming (roll, pitch, yaw)
                'filename': filename,
                'pos_str': pos_str
            })
    return data

def get_home_direction(pos, home_pos, yaw):
    """
    Determines if home is to the Left or Right of the current heading.
    Returns: 'Left', 'Right', or 'Center'
    """
    # Vector to home
    to_home = home_pos[:2] - pos[:2]
    
    # Current heading vector
    # Yaw is in degrees, usually 0 is East (X+), 90 is North (Y+) in standard math,
    # but in Isaac Sim/Unreal it might vary. 
    # Assuming standard: x = cos(yaw), y = sin(yaw)
    rad = np.deg2rad(yaw)
    heading = np.array([np.cos(rad), np.sin(rad)])
    
    # Cross product (z-component)
    # cross = Fx * Hy - Fy * Hx
    cross = heading[0] * to_home[1] - heading[1] * to_home[0]

    # round to 2 decimal places
    cross = round(cross, 2)
    
    if cross > 0:
        return 'Left' # Home is to the Left
    elif cross < 0:
        return 'Right' # Home is to the Right
    else:
        return 'Center'

def calculate_heading(pos1, pos2):
    """
    Calculates the heading (angle) from pos1 to pos2.
    Returns: Heading in degrees (0 to 360)
    """
    # Vector from pos1 to pos2
    to_pos2 = pos2[:2] - pos1[:2]
    
    # Current heading vector
    # Yaw is in degrees, usually 0 is East (X+), 90 is North (Y+) in standard math,
    # but in Isaac Sim/Unreal it might vary. 
    # Assuming standard: x = cos(yaw), y = sin(yaw)
    rad = np.arctan2(to_pos2[1], to_pos2[0])
    heading = np.degrees(rad)
    
    # Ensure heading is in range [0, 360)
    if heading < 0:
        heading += 360
    
    return heading

def train_network(data, home_pos, pn_nb=None, mbon_nb=2, facing='rotate',rotate=45, keep_ratio=1/2):
    """
    Trains the Mushroom Body network using virtual rotation.
    """
    print("Initializing Mushroom Body...")
    # Initialize MB
    # We need to know PN_nb first. Let's process one image to find out.
    first_img = cv2.imread(data[0]['image_path'])
    if first_img is None:
        raise ValueError(f"Could not read image: {data[0]['image_path']}")
    
    # Gaby's create_pn returns (pn, img_sobel). pn is the vector.
    pn_sample, _ = create_pn(first_img)

    pn_size = len(pn_sample)

    seeds = [5, 6, 7, 8]
    
    # define mbon_nb mbs
    mbs = [MushroomBody(PN_nb=pn_size, KC_nb=50000, MBON_nb=2, KCtoPN_synapses=16, KC_norm_param=0.01, seed=seeds[i], verbose=True) for i in range(mbon_nb)]
    
    print("Starting Training with Virtual Rotation...")

    # Training Loop
    # Iterate through data
    prev_item = None
    for item in tqdm(data):
        img = cv2.imread(item['image_path'])
        if img is None:
            continue

        # convert image to fisheye
        img = rgb_to_fisheye(img)

        # crop the ground
        img = crop_and_zoom_process(img, keep_ratio)
            
        original_yaw = item['yaw']
        pos = item['pos']


        if facing == 'rotate':
        
            # Virtual Rotation Loop (0 to 360, step rotate)
            for angle in range(0, 360, rotate):
                # Rotate image
                # angle is counter-clockwise
                rotated_img = rotate_image(img, angle)
                
                # Calculate virtual heading
                # If we rotate image by 'angle' (CCW), it simulates the agent turning 'angle' (CCW).
                # So new heading = original_yaw + angle
                virtual_yaw = original_yaw + angle
                
                # Determine Home Direction relative to virtual heading
                direction = get_home_direction(pos, home_pos, virtual_yaw)

                # print(f"pos: {pos}, yaw: {virtual_yaw}, direction: {direction}")

                # Create PN
                pn, image_sobel = create_pn(rotated_img)

                # show image
                # cv2.imshow('image', rotated_img)
                # cv2.waitKey(0)
                
                # Refresh MB
                for mb in mbs:
                    mb.refresh(pn)
                
                # Learning Rule
                
                if direction == 'Right':
                    for mb in mbs:
                        pre_nonzero_weights = mb.get_nonzero_KCtoMBON_weights(mbon_index=0)
                        mb.learn(mbon_index=0)
                        post_nonzero_weights = mb.get_nonzero_KCtoMBON_weights(mbon_index=0)
                        print(f"MBON 0 Weights before: {pre_nonzero_weights}, after: {post_nonzero_weights}")
                elif direction == 'Left':
                    for mb in mbs:
                        pre_nonzero_weights = mb.get_nonzero_KCtoMBON_weights(mbon_index=1)
                        mb.learn(mbon_index=1)
                        post_nonzero_weights = mb.get_nonzero_KCtoMBON_weights(mbon_index=1)
                        print(f"MBON 1 Weights before: {pre_nonzero_weights}, after: {post_nonzero_weights}")
                elif direction == 'Center':
                    for mb in mbs:
                        pre_nonzero_weights_0 = mb.get_nonzero_KCtoMBON_weights(mbon_index=0)
                        pre_nonzero_weights_1 = mb.get_nonzero_KCtoMBON_weights(mbon_index=1)
                        mb.learn(mbon_index=0)
                        mb.learn(mbon_index=1)
                        post_nonzero_weights_0 = mb.get_nonzero_KCtoMBON_weights(mbon_index=0)
                        post_nonzero_weights_1 = mb.get_nonzero_KCtoMBON_weights(mbon_index=1)
                        print(f"MBON 0 Weights before: {pre_nonzero_weights_0}, after: {post_nonzero_weights_0}")
                        print(f"MBON 1 Weights before: {pre_nonzero_weights_1}, after: {post_nonzero_weights_1}")
                # If Center, do nothing (or inhibit both? usually nothing)

                print(f"pos: {pos}, yaw: {virtual_yaw}, direction: {direction}")   

        elif facing == 'fixed_next': # fixed_next means we use the heading from the previous location to rotate the current image
            
            if prev_item is not None:
                # calculate a heading of prev location to current location, use that as the heading the rotating the current image
                heading = calculate_heading(prev_item['pos'], item['pos'])
                rotated_img = rotate_image(img, heading)
                
                # Create PN
                pn, image_sobel = create_pn(rotated_img)
                
                # Refresh MB
                for mb in mbs:
                    mb.refresh(pn)
                
                # Determine Home Direction relative to virtual heading
                direction = get_home_direction(pos, home_pos, heading)

                # Learning Rule
                if direction == 'Right':
                    for mb in mbs:
                        mb.learn(mbon_index=0)
                elif direction == 'Left':
                    for mb in mbs:
                        mb.learn(mbon_index=1)
                elif direction == 'Center':
                    for mb in mbs:
                        mb.learn(mbon_index=0)
                        mb.learn(mbon_index=1)
                # If Center, do nothing (or inhibit both? usually nothing)

                prev_item = item
                # show the rotated image, with the title showing the direction
                # print(f"heading: {heading}, direction: {direction}")
                # cv2.imshow('rotated_img', rotated_img)
                # cv2.waitKey(0)

            else:
                prev_item = item

    return mbs

def run_simulation(args):
    # Load Data
    print(f"Loading data from {args.csv_path}...")
    
    data = load_data(args.csv_path, args.image_folder)
    home_pos = np.array(args.home_pos)
    
    # Train
    mbs = train_network(data, home_pos)
    
    print("Training complete.")
    
    # Evaluation (Optional, but good for verification)
    # We can run a quick evaluation on the original images (no rotation) to see the vector field
    print("Evaluating on original images...")
    
            
    plot_data_x, plot_data_y = [], []
    plot_data_u, plot_data_v = [], []

    # Iterate through every physical location in your CSV
    for item in tqdm(data):
        # 1. Load the base image for this location ONCE
        base_img = cv2.imread(item['image_path'])
        if base_img is None: 
            continue
        
        pos = item['pos']
        original_yaw = item['yaw']
        
        # We will accumulate the vectors here
        total_turn_vec = np.array([0.0, 0.0])
        NUM_ROTATIONS = 10
        
        # 2. The "Group" Loop: Generate 10 random views at this location
        for _ in range(NUM_ROTATIONS):
            # Pick a random angle (0 to 360 degrees)
            angle = random.uniform(0, 360)
            
            # Create the synthetic view
            img = rotate_image(base_img, angle)
            
            # Calculate the "Virtual Yaw" (The direction the agent is virtually facing)
            # Note: We must track this to know which way "Left" is
            virtual_yaw = original_yaw + angle
            
            # --- Neural Network Process ---
            pn, _ = create_pn(img)
            familiarities = []
            for mb in mbs:
                familiarities.append(mb.get_familiarity(pn))

            print("Activities:", familiarities)
            
            # Calculate Signal
            left_activity = familiarities[0][1] + familiarities[1][1]
            right_activity = familiarities[0][0] + familiarities[1][0]
            turn_signal = left_activity - right_activity

            print("Signal:", turn_signal)
            
            # --- Vector Calculation ---
            # Calculate vector perpendicular to the VIRTUAL heading
            rad = np.deg2rad(virtual_yaw)
            perp_vec = np.array([-np.sin(rad), np.cos(rad)])
            
            # Add to the running total for this location
            total_turn_vec += perp_vec * turn_signal

        # 3. Average the result
        avg_vec = total_turn_vec / NUM_ROTATIONS
        
        # 4. Store for plotting
        plot_data_x.append(pos[0])
        plot_data_y.append(pos[1])
        plot_data_u.append(avg_vec[0])
        plot_data_v.append(avg_vec[1])

        print(avg_vec)

        # Optional: Print if you still need to debug
        # print(f"Turn Signal: {turn_signal}")
            
    # Plot
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # normalize vectors
    norm = np.linalg.norm(plot_data_u) + np.linalg.norm(plot_data_v)
    plot_data_u = plot_data_u / norm
    plot_data_v = plot_data_v / norm    

        
    plt.figure(figsize=(10, 10))
    plt.quiver(plot_data_x, plot_data_y, plot_data_u, plot_data_v)
    plt.plot(home_pos[0], home_pos[1], 'rx', markersize=10, label='Home')
    plt.title("Mushroom Body Navigation Vector Field")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis('equal')
    plt.savefig(os.path.join(args.output_dir, 'quiver_plot_gaby.png'))
    print(f"Evaluation plot saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True, help="Path to dataset CSV")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to image folder (e.g. Replicator/rgb)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--home_pos", type=float, nargs=3, default=[0.0, 0.0, 1.5])
    parser.add_argument("--keep_ratio", type=float, default=1/2)
    
    args = parser.parse_args()
    run_simulation(args, keep_ratio=args.keep_ratio)
