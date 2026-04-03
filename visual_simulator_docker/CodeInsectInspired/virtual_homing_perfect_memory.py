# Determine the catchment area of a central snapshot
from omni.isaac.kit import SimulationApp
import omni
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from omni.isaac.kit import SimulationApp
from PIL import Image
from insect_utils.flight_path_functions import generate_grid_path, calculate_target_vectors, normalize_vectors, \
                                         calculate_absolute_angular_error, calculate_distance, rotate_vector_by_yaw
from matplotlib.patches import Circle
import torch
import torch.nn as nn
from torchvision.transforms import functional as TF
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle
import os
import time
import csv

# TODO: also this function happens in many files... move to insect_utils?
# Load configuration
def load_config(config_file):
    with open(config_file, 'r') as file:
        return json.load(file)

# TODO: This function is now repeated in each virtual homing file - move to insect_utils?
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

    return rgb_img

# TODO: this function is now redundant with the one in virtual homing snapshot - move to insect_utils?
def rotation_match_images(image, home_image, resize_factor = 0.1, show_error_plot = False):

    # Resize the images to speed up the process
    image = image.resize((int(image.width * resize_factor), int(image.height * resize_factor)))
    home_image = home_image.resize((int(home_image.width * resize_factor), int(home_image.height * resize_factor)))
    # rotate the image over all pixels and find the best match
    width = image.width
    # get an np.array of the image:
    image = np.array(image)
    home_image = np.array(home_image)
    # convert to grayscale:
    image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
    home_image = np.dot(home_image[...,:3], [0.2989, 0.5870, 0.1140])

    mse = np.zeros([width,1])
    for rot in range(width):
        # rotate the image
        rotated_image = np.roll(image, rot, axis=1)
        # calculate the difference between the images
        diff = np.abs(rotated_image - home_image)
        # calculate the mean squared error
        mse[rot] = np.mean(diff ** 2)
        if rot == 0:
            best_mse = mse[rot]
            best_rot = rot
        else:
            if mse[rot] < best_mse:
                best_mse = mse[rot]
                best_rot = rot
    
    if show_error_plot:
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(image)
        plt.title("Image")
        plt.subplot(2, 2, 2)
        plt.imshow(home_image)
        plt.title("Home Image")
        plt.subplot(2, 2, 3)
        plt.plot(np.arange(width), mse)
        plt.title("Mean Squared Error")
        plt.xlabel("Rotation")
        plt.ylabel("MSE")
        plt.subplot(2, 2, 4)
        # show the difference image after the best rotation:
        rotated_image = np.roll(image, best_rot, axis=1)
        diff = np.abs(rotated_image - home_image)
        plt.imshow(diff, cmap='RdBu')
        plt.title("Difference Image")
        plt.colorbar()
        plt.show()

    best_rot /= resize_factor

    return best_rot, best_mse

def match_images(image, home_image, resize_factor = 0.1):

    # Resize the images to speed up the process
    image = image.resize((int(image.width * resize_factor), int(image.height * resize_factor)))
    home_image = home_image.resize((int(home_image.width * resize_factor), int(home_image.height * resize_factor)))
    width = image.width
    # get an np.array of the image:
    image = np.array(image)
    home_image = np.array(home_image)
    # convert to grayscale:
    image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
    home_image = np.dot(home_image[...,:3], [0.2989, 0.5870, 0.1140])

    diff = np.abs(image - home_image)
    # calculate the mean squared error
    mse = np.mean(diff ** 2)

    return mse

# TODO: also repeated many times - move to insect_utils?
def get_image(orientation, position, camera, my_world, sleep_time = 0.05):

    image = []
    tries = 0
    while (len(image) == 0 or tries < 3) and tries < 10:
        tries += 1
        camera.set_world_pose(position, orientation)
        my_world.step(render=True)
        image = camera.get_rgba()
        
        short_sleep = 5
        if tries > 2:
            if tries < short_sleep:
                time.sleep(sleep_time)
            else: 
                # sleep longer:
                time.sleep((tries - (short_sleep-1)) * sleep_time)

    if tries >= 10:
        raise RuntimeError("Failed to get image after 10 tries. Check if the camera is set up correctly.")
    else:
        image = image[:,:,:3]
        image = preprocess(image, save_images = False)
        return image

def get_vector_from_perfect_memory(image, matching_method, memory_images, memory_vectors, memory_positions, memory_rotations, k = 1, debug = False, position = None, verbose = False):
    ''' Given the current image, find the best matching k images, and return the average target vector.
    '''

    if matching_method == "rotation":
        # find the best matching image based on rotation
        best_mses = [float('inf')] * k
        best_indices = [-1] * k
        for i in range(len(memory_images)):
            best_rot, mse = rotation_match_images(image, memory_images[i], resize_factor = 0.1)
            # mse = match_images(image, memory_images[i], resize_factor = 0.1)
            # check if this is among the k best matches
            max_mse = max(best_mses)
            if mse < max_mse:
                max_index = best_mses.index(max_mse)
                best_mses[max_index] = mse
                best_indices[max_index] = i
        
        if debug:
            # plot all memory positions and then the selected ones:
            plt.figure()
            memory_pos_array = np.array(memory_positions)
            plt.plot(memory_pos_array[:,0], memory_pos_array[:,1], 'bo', label='Memory Positions')
            for index in best_indices:
                plt.plot(memory_positions[index][0], memory_positions[index][1], 'ro', label='Selected Position')
            if position is not None:
                plt.plot(position[0], position[1], 'gx', label='Current Position')
            plt.title("Memory Positions and Selected Matches")
            plt.xlabel("X Position")
            plt.ylabel("Y Position")
            plt.axis('equal')
            plt.legend()
            plt.show()

        # get the average target vector of the k best matches
        dx_total = 0.0
        dy_total = 0.0
        for index in best_indices:
            # target_vector = memory_vectors[index]
            target_vector = memory_positions[index]
            dx_total += target_vector[0]
            dy_total += target_vector[1]
            if verbose: 
                print(f"Matched position: {memory_positions[index]}, rotation: {memory_rotations[index]}, target vector: {target_vector}")
                print(f"MSE = {best_mses[best_indices.index(index)]}")
        
        dx_avg = dx_total / k
        dy_avg = dy_total / k
        if verbose:
            print(f"Average target vector: ({dx_avg}, {dy_avg})")

        return dx_avg, dy_avg
    else:
        raise ValueError("Unknown matching method: {}".format(matching_method))


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

def perfect_memory_navigation(config_homing, verbose = True, graphics = True, save_images = False, simulation_app = None, my_world = None, \
                         usd_path = None, debug = False, dataset_folder = None, image_folder = None):
    
    if simulation_app is None:
        close_app_at_end = True
        with open("config/server.json", 'r') as file:
            options = json.load(file)
        server = options['server']
        simulation_app = SimulationApp({"headless": server})
    else:
        close_app_at_end = False

    from omni.isaac.core.utils.rotations import euler_angles_to_quat
    from omni.isaac.sensor import Camera
    from omni.isaac.core import World
    from omni.isaac.core.utils.rotations import euler_angles_to_quat

    # Extract parameters from virtual homing config
    
    fixed_yaw = True

    # rotation: determine the best rotation between the current and home image
    matching_method = "rotation"

    # load the config render file:
    config_render_file = "config/config_render.json"
    with open(config_render_file, 'r') as f:
        config_render = json.load(f)
    if usd_path is None:
        usd_path = config_render['map']
    virtual_home_position = np.array(config_render['home_position']) # home position to reach
    cropbox_params = config_render['crop_box_params']
    image_size = cropbox_params[2] if cropbox_params is not None else 1024

    if my_world is None:
        omni.usd.get_context().open_stage(usd_path)
        my_world = World(stage_units_in_meters=1.0)

    # load landmark positions:
    landmark_path = get_locations_path(usd_path)
    landmark_positions = load_landmark_locations(landmark_path)
    # obstacle avoidance settings:
    too_close = 1.0 # minimal distance from landmarks, not to be inside a tree
    # artificial potential field parameters:
    d_lim = 2.0  # distance limit for repulsive potential
    max_step = 2.5  # maximum step size

    # Initialize camera and load model
    camera = Camera(prim_path="/World/Camera", name="camera1", resolution=(image_size, image_size))
    camera.set_projection_type("fisheyeSpherical") # fisheyeSpherical
    
    my_world.reset()
    camera.initialize()
    my_world.step(render=True)
    
    # pause the program to prevent an out of memory crash:
    print("Sleeping to prevent out of memory crash")
    time.sleep(15)
    
    sleep_time = 0.05
    
    # load the virt_homing config:
    config_virtual_homing_file = "config/config_virt_homing.json"
    with open(config_virtual_homing_file, 'r') as f:
        config_virtual_homing = json.load(f)

    start_location_pattern = config_virtual_homing['start_location_pattern'] # square or circle
    north = True # config_virtual_homing['north'] # Whether the robot always faces north # TODO: this should be a setting
    params = config_virtual_homing['params']
    step_size = params['step_size']
    gradient_step_size = 1.0
    count_limit = params['count_limit'] # max number of steps to take in the sim
    homing_threshold = params['threshold'] # distance to home position to consider successful
    

    if start_location_pattern == "circle":
        offset_list = []
        radius_scales = config_virtual_homing['circle_params'].get('radius_scales', [1.0])
        n_radius_scales = len(radius_scales)
        for rs in radius_scales:
            radius = rs * config_virtual_homing['circle_params'].get('radius', 10)
            num_points = config_virtual_homing['circle_params'].get('num_points', 16)
            angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
            offset_list.append([(virtual_home_position[0] + radius*np.cos(theta), virtual_home_position[1] + radius*np.sin(theta), virtual_home_position[2]) for theta in angles])
        offset_list = [item for sublist in offset_list for item in sublist]  # Flatten the list

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
    trajectories = []

    all_image_positions = []
    all_image_MSEs = []

    n_runs = len(offset_list)
    successes = np.zeros(n_runs)

    initial_orientation = euler_angles_to_quat([0.0, 0.0, yaw_list[0]], degrees=True)
    
    if dataset_folder is None:
        # TODO: load the dataset images and vectors
        ...
        # home_image = get_image(initial_orientation, virtual_home_position, camera, my_world)
    else:
        dataset_CSV = os.path.join(dataset_folder, 'dataset_navigation.csv')
        data = pd.read_csv(dataset_CSV)
        # get the filenames ('Filename'), targets ('Target'), positions ('Position') and rotations ('Rotation') from the CSV:
        filenames = data['Filename'].values
        targets = data['Target'].values
        positions = data['Position'].values
        rotations = data['Rotation'].values

        memory_images = []
        memory_vectors = []
        memory_positions = []
        memory_rotations = []
        for i in range(len(filenames)):
            memory_image_path = os.path.join(image_folder, filenames[i])
            memory_image = Image.open(memory_image_path)
            memory_images.append(memory_image)
            target_vector = np.fromstring(targets[i].strip('[]'), sep=' ')
            memory_vectors.append(target_vector)
            position = np.fromstring(positions[i].strip('()'), sep=', ')
            memory_positions.append(position)
            rotation = np.fromstring(rotations[i].strip('()'), sep=', ')
            memory_rotations.append(rotation)
        
        if debug:
            plt.figure()
            plt.scatter([pos[0] for pos in memory_positions], [pos[1] for pos in memory_positions])
            plt.scatter(virtual_home_position[0], virtual_home_position[1], color='r', marker='x', s=100)
            plt.title("Memory Image Positions")
            plt.xlabel("X Position")
            plt.ylabel("Y Position")
            plt.show()

    for run_index, initial_position in enumerate(offset_list):
        # Reset the world and camera to the initial position
        yaw = yaw_list[run_index]
        orientation = euler_angles_to_quat([0.0, 0.0, yaw], degrees=True)

        homing_positions = [initial_position]
        position = initial_position
        counter = 0
        predictions = []
        norm_predictions = []
        angular_errors = []

        success = False

        while calculate_distance(homing_positions[-1], virtual_home_position) > homing_threshold and counter < count_limit:
            # Get the current image
            image = get_image(orientation, position, camera, my_world, sleep_time = sleep_time)
            image = get_image(orientation, position, camera, my_world, sleep_time = sleep_time)
            
            dx, dy = get_vector_from_perfect_memory(image, matching_method, memory_images, memory_vectors, memory_positions, memory_rotations, position = position)

            norm = np.sqrt(dx**2 + dy**2)
            if norm > 0:
                dx /= norm
                dy /= norm
            else:
                dx = 0
                dy = 0

            # apply obstacle avoidance using artificial potential fields:
            move = np.array([-dx, -dy])
            # check distance to all landmarks:
            if landmark_positions is not None:
                for landmark in landmark_positions:
                    landmark_pos = np.array(landmark)
                    dist = calculate_distance(position, landmark_pos)
                    if dist < d_lim:
                        if verbose:
                            print(f"Applying obstacle avoidance for landmark at {landmark_pos} with distance {dist:.2f}, old move = {move}", end  ='')
                        # calculate repulsive force:
                        repulsive_magnitude = 0.5 * (1.0 / dist - 1.0 / d_lim) / (dist ** 2)
                        direction_away = position[:2] - landmark_pos[:2]
                        direction_away = direction_away / np.linalg.norm(direction_away)
                        repulsive_force = repulsive_magnitude * direction_away
                        move = np.array([-dx, -dy]) + repulsive_force
                        magn_move = np.linalg.norm(move)
                        if magn_move > 1e-5:
                            move = move / magn_move
                        if verbose:
                            print(f", repulsive_force = {repulsive_force}, old move = {-dx}, {-dy}, new move = {move}")
            
            target_dx  = virtual_home_position[0] - position[0]
            target_dy  = virtual_home_position[1] - position[1]
            target_vector = np.asarray([target_dx, target_dy])
            target_magn = np.linalg.norm(target_vector)
            prediction_vector = np.asarray([-dx, -dy])
            # calculate the angular error:
            angular_error = calculate_absolute_angular_error(target_vector, prediction_vector)
            angular_errors.append(angular_error)

            # new_position = (position[0] - dx * step_size, position[1] - dy * step_size, position[2])
            new_position = (position[0] + move[0] * step_size, position[1] + move[1] * step_size, position[2])
            
            position = new_position
            homing_positions.append(position)

            counter += 1

            if calculate_distance(position, virtual_home_position) <= homing_threshold:
                successes[run_index] = 1
                successful_runs += 1
                break

        # Calculate total distance traveled and straight-line distance
        total_distance = sum(np.linalg.norm(np.array(homing_positions[i + 1]) - np.array(homing_positions[i])) for i in range(len(homing_positions) - 1))
        # Shortest distance should take into account that the run ends when it crosses the homing threshold:
        straight_line_distance = calculate_distance(homing_positions[0], virtual_home_position) - homing_threshold
        total_distances.append((straight_line_distance, total_distance))

        # Store angular errors and positions
        trajectories.append(homing_positions)
        all_homing_positions.extend(homing_positions)
        all_angular_errors.extend(angular_errors)
        
        run_index += 1
        if verbose:
            print(f"Run {run_index} out of {n_runs} completed. Total successes: {successful_runs}")

    if verbose:
        print(f"Total successful runs: {successful_runs} out of {n_runs}. Success percentage = {successful_runs / n_runs * 100:.2f}%")

    straight_line_distances = [dist[0] for dist in total_distances]
    total_distances_traveled = [dist[1] for dist in total_distances]

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


    if graphics:
        # plot the trajectories:
        plt.figure()
        for trajectory in trajectories:
            x = [pos[0] for pos in trajectory]
            y = [pos[1] for pos in trajectory]
            plt.plot(x, y, marker='o', markersize=2)
            # draw a circle around the home position:
            circle = Circle((virtual_home_position[0], virtual_home_position[1]), homing_threshold, color='r', fill=False)
            plt.gca().add_patch(circle)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Trajectories of Homing Positions')
        plt.show()
    
    return successful_runs, total_distances_traveled, straight_line_distances, all_angular_errors, radius_scales, success_rates, trajectories, successes

# Entry point
if __name__ == "__main__":
    perfect_memory_navigation()