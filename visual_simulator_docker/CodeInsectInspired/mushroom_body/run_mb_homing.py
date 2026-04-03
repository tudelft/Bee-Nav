import os
import omni
import sys
import json
import numpy as np
import argparse
import time
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2


from omni.isaac.kit import SimulationApp

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.pop() # Go up one level to include project root
sys.path.append(os.getcwd())

# Project imports
from mushroom_body.memory_own import MushroomBody
from mushroom_body.process_images_gaby import create_pn, rotate_image
from mushroom_body.run_mushroom_learning import load_data, train_network, calculate_heading, get_home_direction
from mushroom_body.rgb_to_fisheye import rgb_to_fisheye
from mushroom_body.cropper import crop_and_zoom_process



# debug: find the closest image in the learning dataset

# load the image and show it

def find_closest_image(position, data):
    # data is a list of dictionaries ({
            #     'image_path': image_path,
            #     'pos': pos,
            #     'yaw': rot[2], # Assuming (roll, pitch, yaw)
            #     'filename': filename,
            #     'pos_str': pos_str
            # })
    closest_image = None
    closest_image_idx = -1
    closest_image_dist = float('inf')
    for idx, row in enumerate(data):
    #     find the closest image based on position
        dist = np.linalg.norm(position - row['pos'])
        if dist < closest_image_dist:
            closest_image = row
            closest_image_idx = idx
            closest_image_dist = dist
    return closest_image, closest_image_idx

def get_image(orientation, position, camera, my_world, keep_ratio, sleep_time=0.05):
    """
    Captures an image using a robust synchronization loop to ensure 
    the camera is actually at the requested position.
    """
    image = []
    tries = 0
    
    # --- SYNCHRONIZATION LOOP ---
    # We loop until we get data AND we have stepped at least 3 times
    # to flush previous frames from the render buffer.
    while (len(image) == 0 or tries < 3) and tries < 10:
        tries += 1
        
        # Force the pose every step
        camera.set_world_pose(position, orientation)
        my_world.step(render=True)
        image = camera.get_rgba()
        
        # Sleep logic for stability
        short_sleep = 6
        if tries > 1:
            if tries < short_sleep:
                time.sleep(sleep_time)
            else:
                # Exponential backoff if it's really stuck
                time.sleep((tries - (short_sleep - 1)) * sleep_time)

    if tries >= 10:
        raise RuntimeError("Failed to get image after 10 tries. Check camera setup.")

    # --- IMAGE PROCESSING ---
    # (Retained from your original code)
    
    # Swap channels (channel 0 and 2)
    image = image[:, :, [2, 1, 0, 3]]
    
    # Remove Alpha channel
    image = image[:, :, :3]

    # Convert to fisheye
    image = rgb_to_fisheye(image)

    # Crop and zoom
    image = crop_and_zoom_process(image, keep_ratio)
    
    return image

def run_homing_simulation(config, simulation_app = None):
    
    if simulation_app is None:
        # Configuration for Isaac Sim
        # We need to start SimulationApp before importing other Isaac modules
        # Load server config to check for headless mode
        with open("config/server.json", 'r') as file:
            options = json.load(file)
        server = options['server']

        simulation_app = SimulationApp({"headless": server})

    # Import other modules after SimulationApp
    from omni.isaac.core import World
    from omni.isaac.sensor import Camera
    from omni.isaac.core.utils.rotations import euler_angles_to_quat
    import omni.replicator.core as rep


    # 1. Setup Environment
    print("Setting up environment...")
    usd_path = config['homing']['usd_path']
    omni.usd.get_context().open_stage(usd_path)
    my_world = World(stage_units_in_meters=1.0)
    
    # Setup Camera
    # We use a resolution of 1024 (or whatever process_images expects/resizes to)
    # process_images resizes to 32x32 eventually, but fisheye conversion needs reasonable input.
    camera = Camera(prim_path="/World/Camera", name="camera1", resolution=(1024, 1024))
    camera.set_projection_type("fisheyeSpherical")
    camera.initialize()
    
    my_world.reset()
    
    # 2. Train Network
    print("Training Mushroom Body...")


    data = load_data(config['paths']['csv_path'], config['paths']['rgb_folder'])

    print(f"DEBUG: Loaded {len(data)} images from dataset.")
    if len(data) == 0:
        print("DEBUG: Data is empty! Check CSV and image folder.")
    home_pos = np.array(config['homing']['home_pos'])    # Train
    mbs = train_network(data, home_pos, keep_ratio=config['process']['keep_ratio'], rotate=config['process']['rotate_step'], facing=config['process']['facing'])

    # # save the network
    # np.save(config['homing']['output_dir'] + '/mbs.npy', mbs)
    
    # if config validation is True
    if config['homing']['validation']:
        # --- VALIDATION ---
        print("Running Validation...")
        validation_results = []
        
        # Iterate through all training images
        prev_item = None
        for idx, item in enumerate(data):
            base_img = cv2.imread(item['image_path'])
            if base_img is None: 
                print(f"Warning: Could not read image {item['image_path']}")
                continue

            # fisheye conversion
            base_img = rgb_to_fisheye(base_img)

            # crop and zoom
            base_img = crop_and_zoom_process(base_img, keep_ratio=config['process']['keep_ratio'])

            if config['process']['facing'] == 'rotate':
            
                # Test rotations
                # We test 0, 90, 180, 270 degrees (or more fine-grained if needed)
                # Let's do 45 degree steps
                for angle in range(0, 360, config['process']['rotate_step']):
                    rotated_img = rotate_image(base_img, angle)
                    
                    # Ground Truth Direction
                    # Calculate virtual yaw
                    virtual_yaw = item['yaw'] + angle
                    
                    # Vector to home
                    to_home = home_pos[:2] - item['pos'][:2]
                    
                    # Heading vector
                    rad = np.deg2rad(virtual_yaw)
                    heading = np.array([np.cos(rad), np.sin(rad)])
                    
                    # Cross product to determine Left/Right
                    cross = heading[0] * to_home[1] - heading[1] * to_home[0]
                    
                    # Threshold for Center
                    if cross > 0.1: true_direction = 'Left'
                    elif cross < -0.1: true_direction = 'Right'
                    else: true_direction = 'Center'
                    
                    # Neural Net Prediction
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
                    
                    match = (pred == true_direction)
                    
                    validation_results.append({
                        "image": item['filename'],
                        "angle": angle,
                        "true_direction": true_direction,
                        "predicted_direction": pred,
                        "signal": signal,
                        "match": match
                    })

            elif config['process']['facing'] == 'fixed_next':
                if prev_item is not None:
                    # calculate a heading of prev location to current location, use that as the heading the rotating the current image
                    heading = calculate_heading(prev_item['pos'], item['pos'])
                    rotated_img = rotate_image(base_img, heading)
                    pos = item['pos']
                    
                    # Create PN
                    pn, image_sobel = create_pn(rotated_img)
                    
                    # Determine Home Direction relative to virtual heading
                    true_direction = get_home_direction(pos, home_pos, heading)

                    # Neural Net Prediction
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
                    
                    match = (pred == true_direction)
                    
                    validation_results.append({
                        "image": item['filename'],
                        "angle": heading,
                        "true_direction": true_direction,
                        "predicted_direction": pred,
                        "signal": signal,
                        "match": match
                    })

                prev_item = item
            
        
        # Calculate OK Rate
        matches = [r['match'] for r in validation_results]
        ok_rate = sum(matches) / len(matches) if matches else 0
        print(f"Validation OK Rate: {ok_rate:.2f}")
        
        # Save Validation Results
        print(f"DEBUG: Saving validation results to {config['homing']['output_dir']}")
        if not os.path.exists(config['homing']['output_dir']):
            os.makedirs(config['homing']['output_dir'])
            
        with open(os.path.join(config['homing']['output_dir'], 'validation_results.json'), 'w') as f:
            json.dump({
                "ok_rate": ok_rate
            }, f, indent=4)

    
    # 3. Simulation Loop
    print("Starting Simulation...")

    # create the output directory
    if not os.path.exists(config['homing']['output_dir']):
        os.makedirs(config['homing']['output_dir'])
    
    # Define starting positions (Circle)
    radius = config['homing']['radius']
    num_starts = config['homing']['num_starts']
    angles = np.linspace(0, 2*np.pi, num_starts, endpoint=False)
    # one angle, which is pi
    # angles = [np.pi]
    
    start_positions = []
    for theta in angles:
        x = home_pos[0] + radius * np.cos(theta)
        y = home_pos[1] + radius * np.sin(theta)
        z = home_pos[2]
        start_positions.append(np.array([x, y, z]))
        
    results = []
    
    for run_idx, start_pos in enumerate(start_positions):
        print(f"Run {run_idx+1}/{num_starts}")
        
        # Random initial yaw
        yaw = np.random.uniform(0, 360)
        current_pos = start_pos.copy()
        
        trajectory = [current_pos.tolist()]
        success = False
        
        for step in range(config['homing']['max_steps']):
            # 1. Define Orientation
            # Isaac Sim uses Quaternions (w, x, y, z)
            quat = euler_angles_to_quat([0, 0, 180], degrees=True)
            
            # 2. REMOVED: camera.set_world_pose(current_pos, quat)
            # The new function handles this internally.

            # 3. UPDATED: Call get_image with position and orientation
            img = get_image(
                orientation=quat, 
                position=current_pos, 
                camera=camera, 
                my_world=my_world, 
                keep_ratio=config['process']['keep_ratio']
            )
            # Call it twice to be sure that the camera is in position:
            img = get_image(
                orientation=quat, 
                position=current_pos, 
                camera=camera, 
                my_world=my_world, 
                keep_ratio=config['process']['keep_ratio']
            )

            # rotate the image based on the yaw
            img = rotate_image(img, yaw)
            # img = img.copy()

            # save the image, with yaw
            cv2.imwrite(config['homing']['output_dir'] + f'/img_{run_idx}_{step}_{yaw}.png', img)

            # DEBUG
            # find the closest image in the learning dataset
            # closest_image, closest_image_idx = find_closest_image(current_pos, data)
            # closest_image_filename = closest_image['filename']
            # closest_image_path = os.path.join(config['paths']['rgb_folder'], closest_image_filename)
            # # # show the image and the closest image next to each other
            # # # print the position of both images
            # print(f"Position of current image: {current_pos}")
            # print(f"Position of closest image: {closest_image['pos']}")
            # print(f"Yaw: {yaw}")
            # # plt.subplot(1, 2, 1)
            # # plt.imshow(img)
            # # plt.title(f"Step {step}")
            # # plt.subplot(1, 2, 2)
            # closest_image = plt.imread(closest_image_path)
            # # fisheye and crop
            # closest_image = rgb_to_fisheye(closest_image)
            # closest_image = crop_and_zoom_process(closest_image)

            # closest_image = rotate_image(closest_image, yaw)

            # # save the closest image
            # cv2.imwrite(config['homing']['output_dir'] + f'/closest_img_{run_idx}_{step}_{yaw}.png', closest_image) 
            
            # plt.imshow(closest_image)
            # plt.title(f"Closest Image")
            # plt.show()
            
            # save image
            # plt.imsave(f"img_{run_idx}_{step}.png", img)
            
            # PN Vector
            pn, _ = create_pn(img)
            # pn, _ = create_pn(closest_image)    

            # save the pn as image
            # cv2.imwrite(config['homing']['output_dir'] + f'/pn_{run_idx}_{step}_{yaw}.png', _)
            
            # MB Refresh
            for mb in mbs:
                mb.refresh(pn)

            # For the first two steps, just take the image and refresh MBs, don't move or turn
            if step < 2:
                trajectory.append(current_pos.tolist()) # Still record initial position
                continue

            # Control Logic
            activities = []
            for mb in mbs:
                activities.append(mb.get_familiarity(pn))
            print("Activities:", activities)
            left_activity = activities[0][1] + activities[1][1]
            right_activity = activities[0][0] + activities[1][0]
            
            turn_signal = left_activity - right_activity
            
            # Turn
            # Gain: How many degrees per unit of signal?
            yaw_change = config['homing']['gain'] * turn_signal
            print("Yaw change:", yaw_change, "turn signal:", turn_signal)
            yaw += yaw_change
            
            # Move
            step_size = config['homing']['step_size']
            rad = np.deg2rad(yaw)
            dx = step_size * np.cos(rad)
            dy = step_size * np.sin(rad)

            print("Step size:", step_size, "rad:", rad, "dx:", dx, "dy:", dy)   
            
            current_pos[0] += dx
            current_pos[1] += dy
            
            trajectory.append(current_pos.tolist())
            
            # Check Home
            dist = np.linalg.norm(current_pos[:2] - home_pos[:2])
            if dist < config['homing']['threshold']:
                success = True
                break
                
        results.append({
            "run": run_idx,
            "success": success,
            "trajectory": trajectory
        })
        print(f"Result: {'Success' if success else 'Fail'}")

    # Save Results
    if not os.path.exists(config['homing']['output_dir']):
        os.makedirs(config['homing']['output_dir'])
        
    with open(os.path.join(config['homing']['output_dir'], 'homing_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    # save the config as well
    with open(os.path.join(config['homing']['output_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
        
    # Plot
    plt.figure(figsize=(10, 10))
    for res in results:
        traj = np.array(res['trajectory'])
        plt.plot(traj[:, 0], traj[:, 1], 'g-' if res['success'] else 'r-', alpha=0.5)
        plt.plot(traj[0, 0], traj[0, 1], 'bo') # Start
        
    plt.plot(home_pos[0], home_pos[1], 'kx', markersize=12, label='Home')
    
    # Draw Start Circle
    circle = plt.Circle((home_pos[0], home_pos[1]), radius, color='b', fill=False, linestyle='--')
    plt.gca().add_patch(circle)
    
    plt.title(f"Homing Trajectories (Radius={radius}m)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis('equal')
    plt.legend()
    plt.savefig(os.path.join(config['homing']['output_dir'], 'homing_trajectories.png'))
    
    simulation_app.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to config JSON")
    
    # Load Config
    with open('mushroom_body/config/config_mb_homing.json', 'r') as f:
        config = json.load(f)
    
    # Run
    run_homing_simulation(config)
