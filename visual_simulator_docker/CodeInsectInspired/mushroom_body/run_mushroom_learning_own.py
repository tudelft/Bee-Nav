import os
import numpy as np
import json
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import sys
import csv
from collections import defaultdict

# Add current directory to path to import memory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from memory_own import MushroomBody

def parse_tuple(s):
    return np.fromstring(s.strip('()'), sep=',')

def load_data(csv_path, pn_folder):
    """
    Loads metadata and PN vectors using standard csv library.
    """
    data = []
    pn_vectors = {}
    
    print("Loading data...")
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader):
            filename = row['Filename']
            pos_str = row['Position']
            rot_str = row['Rotation']
            
            # Parse vectors
            pos_vec = parse_tuple(pos_str)
            rot_vec = parse_tuple(rot_str)
            
            # Store parsed data
            entry = {
                'Filename': filename,
                'Position_Vec': pos_vec,
                'Rotation_Vec': rot_vec,
                'Position_Str': pos_str # Keep string for grouping
            }
            
            base_name = os.path.splitext(filename)[0]
            pn_path = os.path.join(pn_folder, f'{base_name}.npy')
            
            if os.path.exists(pn_path):
                pn_vectors[len(data)] = np.load(pn_path)
                data.append(entry)
            else:
                # Skip if PN not found
                pass
                
    return data, pn_vectors

def get_home_direction(pos, home_pos, heading_angle_deg):
    """
    Determines if Home is Left or Right of the current heading.
    Returns: 'Left', 'Right', or 'Center'
    """
    # Vector to Home
    to_home = home_pos - pos
    to_home_2d = to_home[:2] # X, Y
    
    # Heading Vector
    # Yaw is rotation around Z. 0 = +X.
    rad = np.deg2rad(heading_angle_deg)
    heading_vec = np.array([np.cos(rad), np.sin(rad)])
    
    # Cross Product (2D determinant)
    # CP = Fx * Hy - Fy * Hx
    cp = heading_vec[0] * to_home_2d[1] - heading_vec[1] * to_home_2d[0]
    
    if cp > 1e-6:
        return 'Left'
    elif cp < -1e-6:
        return 'Right'
    else:
        return 'Center'

def run_simulation(csv_path, pn_folder, output_dir, home_pos=np.array([0.0, 0.0, 1.5])):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 1. Load Data
    data, pn_vectors = load_data(csv_path, pn_folder)
    
    if not data:
        print("No data found. Check paths.")
        return

def train_network(data, pn_vectors, home_pos, pn_nb=None, mbon_nb=4):
    """
    Trains the Mushroom Body network on the provided data.
    """
    if pn_nb is None:
        # Assume PN size from first vector
        first_pn = next(iter(pn_vectors.values()))
        pn_nb = len(first_pn)
        
    mb = MushroomBody(PN_nb=pn_nb, MBON_nb=mbon_nb, verbose=True)
    
    print("Training Mushroom Body...")
    
    # Create a list from data if it's not already (it is a list of dicts from load_data)
    # If it's a DataFrame (old pandas version), iterate differently. 
    # But we are using the list of dicts version now.
    
    iterator = data if isinstance(data, list) else data.iterrows()
    
    for item in tqdm(iterator, total=len(data)):
        if isinstance(data, list):
            row = item
            idx = data.index(row) # This might be slow for large lists, better to enumerate outside
            # But wait, load_data returns a list of dicts.
            # And pn_vectors is a dict keyed by index in that list.
            # Let's just iterate with enumerate.
            pass
        else:
             # Pandas case (not used anymore but for safety)
             idx, row = item
             
    # Re-doing the loop cleanly
    for idx, row in enumerate(tqdm(data)):
        pos = row['Position_Vec']
        rot = row['Rotation_Vec']
        yaw = rot[2]
        
        direction = get_home_direction(pos, home_pos, yaw)
        
        if idx not in pn_vectors:
            continue
            
        pn = pn_vectors[idx]
        
        # Refresh MB with current view
        mb.refresh(pn)
        
        # Learning Rule
        if direction == 'Left':
            # Home is Left -> Turn Left -> Inhibit Right MBONs
            mb.learn(mbon_index=2)
            mb.learn(mbon_index=3)
        elif direction == 'Right':
            # Home is Right -> Turn Right -> Inhibit Left MBONs
            mb.learn(mbon_index=0)
            mb.learn(mbon_index=1)
            
    return mb

def run_simulation(csv_path, pn_folder, output_dir, home_pos=np.array([0.0, 0.0, 1.5])):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 1. Load Data
    data, pn_vectors = load_data(csv_path, pn_folder)
    
    if not data:
        print("No data found. Check paths.")
        return

    # 2. Train Network
    mb = train_network(data, pn_vectors, home_pos)
            
    # 4. Evaluation
    print("Evaluating...")
    results = []
    
    # Group by location
    grouped = defaultdict(list)
    for idx, row in enumerate(data):
        grouped[row['Position_Str']].append((idx, row))
    
    plot_data_x = []
    plot_data_y = []
    plot_data_u = []
    plot_data_v = []
    
    for pos_str, group in tqdm(grouped.items()):
        # Get unique position vector from first item
        pos = group[0][1]['Position_Vec']
        
        # Accumulate turn vectors
        total_turn_vec = np.array([0.0, 0.0])
        
        for idx, row in group:
            pn = pn_vectors[idx]
            rot = row['Rotation_Vec']
            yaw = rot[2]
            
            mb.refresh(pn)
            activity = mb.mbon_activity
            
            left_activity = np.sum(activity[0:2])
            right_activity = np.sum(activity[2:4])

            # print(left_activity, right_activity)
            
            turn_signal = left_activity - right_activity
            
            # Convert turn signal to a vector relative to current heading
            rad = np.deg2rad(yaw)
            # Perpendicular vector (Left)
            perp_vec = np.array([-np.sin(rad), np.cos(rad)])
            
            # Add to total
            total_turn_vec += perp_vec * turn_signal
            
            results.append({
                'filename': row['Filename'],
                'position': pos.tolist(),
                'rotation': rot.tolist(),
                'mbon_activity': activity.tolist(),
                'turn_signal': float(turn_signal)
            })
            
        # Average vector for this location
        if len(group) > 0:
            avg_vec = total_turn_vec / len(group)
            plot_data_x.append(pos[0])
            plot_data_y.append(pos[1])
            plot_data_u.append(avg_vec[0])
            plot_data_v.append(avg_vec[1])

            print(avg_vec)

    # 5. Save Results
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
    # 6. Plot (Raw Magnitude)
    plt.figure(figsize=(10, 10))
    plt.quiver(plot_data_x, plot_data_y, plot_data_u, plot_data_v)
    plt.plot(home_pos[0], home_pos[1], 'rx', markersize=10, label='Home')
    plt.title("Mushroom Body Navigation Vector Field (Magnitude)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(os.path.join(output_dir, 'quiver_plot.png'))
    
    # 7. Plot (Normalized Direction)
    plt.figure(figsize=(10, 10))
    # Normalize vectors
    u = np.array(plot_data_u)
    v = np.array(plot_data_v)
    norm = np.sqrt(u**2 + v**2)
    # Avoid division by zero
    norm[norm == 0] = 1
    u_norm = u / norm
    v_norm = v / norm
    
    plt.quiver(plot_data_x, plot_data_y, u_norm, v_norm)
    plt.plot(home_pos[0], home_pos[1], 'rx', markersize=10, label='Home')
    plt.title("Mushroom Body Navigation Vector Field (Direction Only)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(os.path.join(output_dir, 'quiver_plot_normalized.png'))
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--pn_folder", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    
    args = parser.parse_args()
    
    run_simulation(args.csv_path, args.pn_folder, args.output_dir)