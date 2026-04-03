import csv
from omni.isaac.kit import SimulationApp
import omni
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from omni.isaac.kit import SimulationApp
from PIL import Image
from insect_utils.flight_path_functions import generate_grid_path, calculate_target_vectors, normalize_vectors, \
                                         calculate_absolute_angular_error, calculate_distance, rotate_vector_by_yaw
from matplotlib.patches import Circle
from simple_network import SimpleCNN, CompactCNN_rgb, IncepAttentionCNN_rgb
import torch
import torch.nn as nn
from torchvision.transforms import functional as TF
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle
import os
import time

# Load configuration
def load_config(config_file):
    with open(config_file, 'r') as file:
        return json.load(file)

def get_image(orientation, position, camera, my_world, sleep_time = 0.05, save_images = False):

    pos = position

    image = []
    tries = 0
    while (len(image) == 0 or tries < 3) and tries < 10:
        tries += 1
        camera.set_world_pose(pos, orientation)
        my_world.step(render=True)
        image = camera.get_rgba()
        
        short_sleep = 6
        if tries > 1:
            if tries < short_sleep:
                time.sleep(sleep_time)
            else: 
                # sleep longer:
                time.sleep((tries - (short_sleep-1)) * sleep_time)

    if tries >= 10:
        raise RuntimeError("Failed to get image after 10 tries. Check if the camera is set up correctly.")
    else:
        image = image[:,:,:3]
        image_tensor = preprocess(image, save_images = save_images)
        return image, image_tensor

def preprocess(image, save_images = False, counter = None, run_index = None, debug = False, cropbox_params = None):
    """Custom preprocessing logic for image data.
    
    Args:
        image: Input image in NumPy array format.

    Returns:
        Preprocessed image as a torch tensor.
    """
    # Example preprocessing (users can adjust as needed)
    rgb_img = Image.fromarray(image, "RGB")
    # Example crop, resize, or any custom transformation
    if cropbox_params is None:
        cropbox_params = (0, 312, 1024, 712)
    rgb_img = rgb_img.crop(cropbox_params)
    
    if save_images:
        debug_dir = "debug_images"
        os.makedirs(debug_dir, exist_ok=True)

        # check how many files there already are in the directory:
        files = os.listdir(debug_dir)
        if counter == None:
            it = len(files)
        else:
            it = counter
        image_name = f"{debug_dir}/image_run_{run_index}_step_{it}.png"
        rgb_img.save(image_name)

    if debug:
        plt.figure()
        plt.imshow(rgb_img)
        plt.show()
        plt.close()

    image_tensor = TF.to_tensor(rgb_img).unsqueeze(0)
    
    return image_tensor

def correct_vector(x, y):
    # sometimes due to different cam orientation in render requires correction
    return [y,x]

# TODO: this should go to utils:
def get_locations_path(map_path):
    # find '_area' in the map_path:
    area_index = map_path.find('_area')
    length_area = len('_area')
    # insert '_locations' after '_area':
    locations_path = map_path[:area_index + length_area] + '_locations' + map_path[area_index + length_area:]
    # change the extension from usd to csv:
    locations_path = locations_path[:-4] + '.csv'
    return locations_path

# TODO: this should go to utils:
def load_landmark_locations(locations_path):
    landmark_filename = locations_path
    if os.path.exists(landmark_filename):
        with open(landmark_filename, mode='r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header row
            X = []
            Y = []
            Z = []
            for row in reader:
                x, y, z = row
                X.append(float(x))
                Y.append(float(y))
                Z.append(float(z))
            x = np.asarray(X)
            y = np.asarray(Y)
            z = np.asarray(Z)
            n_landmarks = len(X)
            landmark_positions = np.zeros([n_landmarks, 3])
            landmark_positions[:, 0] = x
            landmark_positions[:, 1] = y
            landmark_positions[:, 2] = z
    else:
        print(f"File {landmark_filename} does not exist.")
        landmark_positions = None
    return landmark_positions

# Main function
def main(config, verbose = True, graphics = True, save_images = False, simulation_app = None, my_world = None, network_type = "simple"):

    if simulation_app is None:
        with open("config/server.json", 'r') as file:
            options = json.load(file)
        server = options['server']
        simulation_app = SimulationApp({"headless": server})

    from omni.isaac.core.utils.rotations import euler_angles_to_quat
    from omni.isaac.sensor import Camera
    from omni.isaac.core import World
    from omni.isaac.core.utils.rotations import euler_angles_to_quat

    # Extract parameters from config
    usd_path = config['map']
    model_path = config['model']
    params = config['params']
    step_size = params['step_size']
    count_limit = params['count_limit'] # max number of steps to take in the sim
    homing_threshold = params['threshold'] # distance to home position to consider successful
    start_location_pattern = config['start_location_pattern'] # square or circle
    virtual_home_position = np.array(config['virtual_home_point']) # home position to reach
    pitch_roll = config['pitch_roll']  # pitch and roll noise
    north = config['north'] # Whether the robot always faces north
    fixed_yaw = config['fixed_yaw'] # Whether the robot always faces a fixed yaw

    # load landmark positions:
    landmark_path = get_locations_path(usd_path)
    landmark_positions = load_landmark_locations(landmark_path)
    # obstacle avoidance settings:
    too_close = 1.0 # minimal distance from landmarks, not to be inside a tree
    # artificial potential field parameters:
    d_lim = 2.0  # distance limit for repulsive potential
    max_step = 2.5  # maximum step size

    # get the cropbox params:
    # load the config render file:
    config_render_file = "config/config_render.json"
    config_render = load_config(config_render_file)
    cropbox_params = config_render['crop_box_params']
    image_size = cropbox_params[2] if cropbox_params is not None else 1024

    if my_world is None:
        omni.usd.get_context().open_stage(usd_path)
        my_world = World(stage_units_in_meters=1.0)

    # Initialize camera and load model
    camera = Camera(prim_path="/World/Camera", name="camera1", resolution=(image_size, image_size))
    camera.set_projection_type("fisheyeSpherical") # fisheyeSpherical
    
    my_world.reset()
    camera.initialize()
    my_world.step(render=True)
    # pause the program to prevent an out of memory crash:
    print("Sleeping to prevent out of memory crash")
    time.sleep(15)
    
    sleep_time = 0.1

    # New code:
    if network_type == "simple":
        model = SimpleCNN()
    elif network_type == "compact":
        model = CompactCNN_rgb()
    elif network_type == "attention":
        model = IncepAttentionCNN_rgb()
    
    model.load_state_dict(torch.load(model_path))

    # Generate initial positions based on the selected pattern
    if start_location_pattern == "square":
        side_length = config['square_params'].get('side_length', 20)
        num_points = config['square_params'].get('num_points', 16)
        offset = side_length # / 2

        # Determine how many points to place per side
        points_per_side = max(1, num_points // 4)
        linear_offsets = np.linspace(-offset, offset, points_per_side)

        offset_list = []
        offset_list.extend([(virtual_home_position[0] + x, virtual_home_position[1] + offset, 1.5) for x in linear_offsets])
        offset_list.extend([(virtual_home_position[0] + x, virtual_home_position[1] - offset, 1.5) for x in linear_offsets])
        offset_list.extend([(virtual_home_position[0] - offset, virtual_home_position[1] + y, 1.5) for y in linear_offsets[1:-1]])
        offset_list.extend([(virtual_home_position[0] + offset, virtual_home_position[1] + y, 1.5) for y in linear_offsets[1:-1]])
    elif start_location_pattern == "circle":
        offset_list = []
        radius_scales = config['circle_params'].get('radius_scales', [1.0])
        n_radius_scales = len(radius_scales)
        for rs in radius_scales:
            radius = rs * config['circle_params'].get('radius', 12)
            num_points = config['circle_params'].get('num_points', 16)
            angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
            offset_list.append([(virtual_home_position[0] + radius*np.cos(theta), virtual_home_position[1] + radius*np.sin(theta), virtual_home_position[2]) for theta in angles])
        offset_list = [item for sublist in offset_list for item in sublist]  # Flatten the list
    else:
        raise ValueError(f"Unsupported start location pattern: {start_location_pattern}")

    # TODO: make this a function in insect utils:
    # make sure the offet_list is not inside a tree:
    if landmark_positions is not None:
        for i in range(len(offset_list)):
            pos = np.array(offset_list[i])
            for landmark in landmark_positions:
                landmark_pos = np.array(landmark)
                dist = calculate_distance(pos, landmark_pos)
                if dist < too_close:
                    # move the position away from the landmark along the vector:
                    direction = pos[:2] - landmark_pos[:2]
                    dist_xy = np.linalg.norm(direction[:2])
                    if dist_xy < 1e-5:
                        direction = np.array([too_close, 0.0])
                    else:
                        direction = direction[:2] / dist_xy
                    new_pos = list(landmark_pos)
                    new_pos[:2] = new_pos[:2] + direction * too_close
                    offset_list[i] = (new_pos[0], new_pos[1], pos[2])
                    if verbose:
                        print(f"Adjusted starting position {i} to avoid being too close to a landmark {landmark_pos}, from {pos} to {offset_list[i]}.")

    if north:
        # set all orientations to 180 (North)
        yaw_list = [180.0] * len(offset_list)
    else:
        # take random orientations
        yaw_list = np.random.uniform(0, 360, len(offset_list))

    # Simulation loop variables
    successful_runs = 0
    run_index = 0
    total_distances = []
    all_homing_positions = []
    all_angular_errors = []
    all_distance_errors = []
    trajectories = []
    colors = plt.cm.get_cmap('viridis', len(offset_list))

    # Set up plotting for visualization
    if graphics:
        fig, ax = plt.subplots()
        fig2, ax2 = plt.subplots()

    n_runs = len(offset_list)
    successes = np.zeros(n_runs)

    for run_index, initial_position in enumerate(offset_list):
        # Reset the world and camera to the initial position
        yaw = yaw_list[run_index]
        initial_orientation = euler_angles_to_quat([0.0, 0.0, yaw], degrees=True)
        camera.set_world_pose(initial_position, initial_orientation)        
        
        current_position = initial_position
        current_orientation = initial_orientation

        homing_positions = [current_position]
        counter = 0
        predictions = []
        norm_predictions = []
        angular_errors = []
        distance_errors = []

        success = False

        while calculate_distance(homing_positions[-1], virtual_home_position) > homing_threshold and counter < count_limit:
            
            # Yes, the double get_image is on purpose.
            image, image_tensor = get_image(current_orientation, current_position, camera, my_world)
            image, image_tensor = get_image(current_orientation, current_position, camera, my_world)

            with torch.no_grad():
                prediction = model(image_tensor).squeeze()
                prediction = [prediction[0].item(), prediction[1].item()]
                # Determine the distance error:
                magn_prediction = np.linalg.norm(prediction)
                GT_world_frame = virtual_home_position[:2] - current_position[:2]
                magn_GT_world_frame = np.linalg.norm(GT_world_frame)
                distance_errors.append(magn_prediction - magn_GT_world_frame)
                norm_prediction = normalize_vectors(np.array([prediction]))

            # Calculate new position
            step_size = 0.1 + 0.4 * (np.linalg.norm(np.asarray(prediction)) / 3.0)
            move = step_size * norm_prediction.flatten()

            if fixed_yaw:
                # derotate the vector from the robot's body coordinates to the world frame
                move = -np.asarray(rotate_vector_by_yaw(move, yaw))
            else:
                # (1) turn into the direction of the network
                
                # determine the angle of the vector:
                desired_yaw_vector = -np.asarray(rotate_vector_by_yaw(move, yaw))
                angle = np.arctan2(desired_yaw_vector[1], desired_yaw_vector[0]) 
                angle = np.degrees(angle)
                
                # rotate the yaw to the angle:
                yaw = angle 

                # (2) move to the front of the robot
                # move into the yaw direction:
                move = step_size * np.array([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])      
            
            # apply obstacle avoidance using artificial potential fields:
            if landmark_positions is not None:
                for landmark in landmark_positions:
                    landmark_pos = np.array(landmark)
                    dist = calculate_distance(current_position, landmark_pos)
                    if dist < d_lim:
                        if verbose:
                            print(f"Applying obstacle avoidance for landmark at {landmark_pos} with distance {dist:.2f}, old move = {move}", end  ='')
                        # calculate repulsive force:
                        repulsive_magnitude = 0.5 * (1.0 / dist - 1.0 / d_lim) / (dist ** 2)
                        direction_away = current_position[:2] - landmark_pos[:2]
                        direction_away = direction_away / np.linalg.norm(direction_away)
                        repulsive_force = repulsive_magnitude * direction_away
                        move += repulsive_force
                        if verbose:
                            print(f", repulsive_force = {repulsive_force}, new move = {move}")

            # clip the move to maximum step size:
            move_magnitude = np.linalg.norm(move)
            if move_magnitude > max_step:
                if verbose:
                    print(f"Clipping move from {move_magnitude:.2f} to {max_step:.2f} - move was {move}")
                move = (move / move_magnitude) * max_step

            new_position = [current_position[0] + move[0], current_position[1] + move[1], current_position[2]]
            homing_positions.append(new_position)
            predictions.append(prediction)
            norm_predictions.append(norm_prediction)
            
            # Change the orientation of the camera:
            pitch_noise = np.random.uniform(-pitch_roll, pitch_roll)
            roll_noise = np.random.uniform(-pitch_roll, pitch_roll)
            current_orientation = euler_angles_to_quat([pitch_noise, roll_noise, yaw], degrees=True)
            current_position = new_position
            
            # Calculate and store angular error
            # We now use the world frame vectors, making things simpler:
            target_vector = [virtual_home_position[0] - current_position[0], virtual_home_position[1] - current_position[1]]
            move_vector = [move[0], move[1]]
            angular_error = calculate_absolute_angular_error(target_vector, move_vector)

            angular_errors.append(angular_error)
            counter += 1
            
            if calculate_distance(new_position, virtual_home_position) <= homing_threshold:
                successes[run_index] = 1
                successful_runs += 1
                break

        # Visualization of homing path
        current_color = colors(run_index)
        # plot text with the run index:
        if graphics:
            ax.text(homing_positions[0][0], homing_positions[0][1], f"{run_index + 1}", fontsize=8, color='black')
            ax.text(homing_positions[-1][0], homing_positions[-1][1], f"{run_index + 1}", fontsize=8, color='black')
            
        for pos_index in range(len(homing_positions) - 1):
            start_pos = homing_positions[pos_index]
            end_pos = homing_positions[pos_index + 1]
            vector = [end_pos[0] - start_pos[0], end_pos[1] - start_pos[1]]
            if graphics:
                ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], color=current_color, marker='*')

        # plot a green marker circle if the homing was successful, else a red circle
        if graphics:
            if successes[run_index] == 1:
                ax.plot(homing_positions[-1][0], homing_positions[-1][1], 'go', markersize = 5)
            else:
                ax.plot(homing_positions[-1][0], homing_positions[-1][1], 'ro', markersize = 5)

        # Calculate total distance traveled and straight-line distance
        total_distance = sum(np.linalg.norm(np.array(homing_positions[i + 1]) - np.array(homing_positions[i])) for i in range(len(homing_positions) - 1))

        # Shortest distance should take into account that the run ends when it crosses the homing threshold:
        straight_line_distance = calculate_distance(homing_positions[0], virtual_home_position) - homing_threshold
        total_distances.append((straight_line_distance, total_distance))
        
        # Add a placeholder angular error for the initial position
        angular_errors.insert(0, 0)

        # Store angular errors and positions
        trajectories.append(homing_positions)
        all_homing_positions.extend(homing_positions)
        all_angular_errors.extend(angular_errors)
        all_distance_errors.extend(distance_errors)
        
        run_index += 1
        if verbose:
            print(f"Run {run_index} out of {n_runs} completed. Total successes: {successful_runs}")

    if verbose:
        print(f"Total successful runs: {successful_runs} out of {n_runs}. Success percentage = {successful_runs / n_runs * 100:.2f}%")

    # Finalize plotting paths
    home_x, home_z = virtual_home_position[0], virtual_home_position[1]
    if graphics:
        ax.scatter(home_x, home_z, c='r', marker='x', label='Virtual Home')
        ax.set_xlabel('X position')
        ax.set_ylabel('Z position')
        ax.set_title('Homing Paths')
        ax.grid(True)
   
    # Create scatter plot of angular errors
    homing_positions_np = np.array(all_homing_positions)
    if graphics:
        scatter = ax2.scatter(homing_positions_np[:, 0], homing_positions_np[:, 1], c=all_angular_errors, cmap='coolwarm', edgecolor='k')
        plt.colorbar(scatter, ax=ax2, label='Absolute Angular Error (degrees)')
        ax2.scatter(home_x, home_z, c='r', marker='x', label='Virtual Home')
        ax2.set_xlabel('X position')
        ax2.set_ylabel('Z position')
        ax2.set_title('Angular Errors')
        plt.show()

    if graphics:
        # Bar plot of distances
        fig3, ax3 = plt.subplots()
        x_labels = [f"Run {i+1}" for i in range(len(total_distances))]

    straight_line_distances = [dist[0] for dist in total_distances]
    total_distances_traveled = [dist[1] for dist in total_distances]

    # determine the indices of the runs that were successful
    indices = np.where(successes == 1)[0]
    if len(indices) > 0:
        total_distances_traveled = np.array(total_distances_traveled)
        straight_line_distances = np.array(straight_line_distances)
        if verbose:
            print(f'Ratio of total distance travelled to straight-line distance: {np.mean(total_distances_traveled[indices]) / np.mean(straight_line_distances[indices]):.2f}')

    if graphics:
        bar_width = 0.35
        index = np.arange(len(x_labels))
        ax3.bar(index, straight_line_distances, bar_width, label='Straight-line Distance')
        ax3.bar(index + bar_width, total_distances_traveled, bar_width, label='Total Distance')
        ax3.set_xlabel('Runs')
        ax3.set_ylabel('Distance')
        ax3.set_title('Straight-line vs Total Distance for Each Run')
        ax3.set_xticks(index + bar_width / 2)
        ax3.set_xticklabels(x_labels)
        ax3.legend()
        plt.tight_layout()
        plt.show()
    

    if start_location_pattern == "circle" and n_radius_scales > 1:
        # Determine the success rate for each radius scale:
        success_rates = []
        for i, rs in enumerate(radius_scales):
            indices = list(np.asarray(range(num_points)) + i * num_points)
            success_rate = np.sum(successes[indices]) / len(indices)
            success_rates.append(success_rate)
        if graphics:
            plt.figure()
            plt.plot(radius_scales, success_rates)
            plt.xlabel('Radius Scale')
            plt.ylabel('Success Rate')
            plt.title('Success Rate by Radius Scale')
            plt.show()
    else:
        radius_scales = None
        success_rates = None

    # Return all relevant statistics:
    return successful_runs, total_distances_traveled, straight_line_distances, all_angular_errors, all_distance_errors, radius_scales, success_rates, trajectories, successes

# Entry point
if __name__ == "__main__":
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'config/config_virt_homing.json'
    config = load_config(config_file)
    save_images = config['save_images'] 
    successful_runs, total_distances_traveled, straight_line_distances, all_angular_errors, all_distance_errors, radius_scales, success_rates, trajectories, successes = \
        main(config, save_images = save_images)
