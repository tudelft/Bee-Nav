from omni.isaac.kit import SimulationApp
import omni
import json
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
import csv
import shutil
import matplotlib
os.environ["LC_NUMERIC"] = "C"
from insect_utils.flight_path_functions import calculate_distance, generate_spiral_path, generate_bee_path, generate_grid_path, \
                generate_noisy_spiral_allan, generate_circle_centered_on_landmarks, generate_uniform_area, \
                calculate_target_vectors, write_csv_dataset, generate_circle_centered_on_home
from insect_utils.augment_and_save import crop_images
def load_config(config_file):
    with open(config_file, 'r') as file:
        return json.load(file)

# TODO: this should go to utils:
def get_locations_path(map_path):
    # find '_area' in the map_path:
    area_index = map_path.find('_area')
    if area_index == -1:
        return None
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


def generate_dataset(config, simulation_app=None, my_world = None, show_positions = False):

    with open("config/server.json", 'r') as file:
        options = json.load(file)
    server = options['server']

    if simulation_app is None:
        simulation_app = SimulationApp({"headless": server})
    
    import omni.replicator.core as rep
    from omni.isaac.sensor import Camera
    from omni.isaac.core import World
    import omni.isaac.core.utils.numpy.rotations as rot_utils
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.replicator.core import settings
    
    # Load parameters from the config
    usd_path = config['map']
    shape_flight = config['shape_flight']
    shape_params = config['shape_params']   # E.g., number of spirals, number of points, 
    output_folder = config['output_folder'] 
    home_position = np.array(config['home_position'])
    number_cameras = config['number_cameras']
    use_noisy_spiral = config.get('use_noisy_spiral', False)  # Default to False if not specified
    cropbox_params = config.get('crop_box_params', None)
    # Prioritize explicit 'image_size' in config, otherwise derive from crop params, else default to 1024
    image_size = config.get('image_size', cropbox_params[2] if cropbox_params is not None else 1024)

    # load landmark positions:
    landmark_path = get_locations_path(usd_path)
    if landmark_path is not None:
        landmark_positions = load_landmark_locations(landmark_path)
    else:
        landmark_positions = None
    # obstacle avoidance settings:
    too_close = 0.5 # minimal distance from landmarks, not to be inside a tree

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Set the root directory for Replicator output
    settings.carb_settings("/omni/replicator/backends/disk/root_dir", output_folder)

    if my_world is None:
        # Load USD map
        omni.usd.get_context().open_stage(usd_path)
        my_world = World(stage_units_in_meters=1.0)
    
    # Set up replicator cameras
    # the rotation for the camera will be overwritten later:
    cameras = [rep.create.camera(projection_type='fisheyeSpherical', rotation=(0, 0, 0)) for _ in range(number_cameras)] 
    render_products = [rep.create.render_product(camera, (image_size, image_size)) for camera in cameras]
    noisy_path = None
    noisy_heading_values_2d = None

    positions = None
    rotations = None

    # Generate the positions at which to take the images:
    # home circle: n_positions and shape_params = [radius]
    # spiral: shape_params = [n_spirals, n_points_on_spiral]
    # bee: shape_params = [n_traverses, n_points_per_traverse, resolution]
    # for a small area: n=4, m=36, b=0.1. For a bigger area: n=5, m=56, b=0.2
    # grid: shape_params = [x_min, x_max, y_min, y_max, n_steps]
    # area: shape_params = [x_min, x_max, y_min, y_max], + n_positions
    if shape_flight == "spiral":
        positions = generate_spiral_path(*shape_params, home_position)
        if use_noisy_spiral:
            # Check for noise parameters and generate noisy path
            noise_params = config.get('noise_params')
            if noise_params is None:
                raise ValueError("Noise parameters are required when using a noisy spiral.")
            noisy_path = generate_noisy_spiral_allan(*shape_params, home_position, *noise_params)
    elif shape_flight == "bee":
        positions = generate_bee_path(*shape_params, home_position)
        if use_noisy_spiral:
            noise_params = config.get('noise_params')
            if noise_params is None:
                raise ValueError("Noise parameters are required when using a noisy bee path.")
            noisy_path = generate_bee_path(*shape_params, home_position, add_noise=True, noise_params=noise_params)
    elif shape_flight == "grid":
        # grid_params = [x_min, x_max, y_min, y_max, n_steps]
        positions = generate_grid_path(config)
    elif shape_flight == "debug":
        positions = [(shape_params[0], shape_params[1], shape_params[2])] * 100
        n_positions = len(positions)
        rotations = []
        rotation_step = 360.0 / n_positions
        for i in range(len(positions)):
            #rotations.append(rot_utils.euler_to_quaternion([0, 0, 0]))
            rotations.append((0, 0, i * rotation_step))
    elif shape_flight == "area":
        n_positions = config['n_positions']
        positions = generate_uniform_area(n_positions, home_position, area = shape_params)
    elif shape_flight == "home_circle":
        n_positions = config['n_positions']
        radius = shape_params[0]
        positions = generate_circle_centered_on_home(n_positions, radius, home_position)
    elif shape_flight == "landmark_circle":
        radius = shape_params[0]
        n_positions = config['n_positions']
        positions = generate_circle_centered_on_landmarks(n_positions, radius, home_position)
    elif shape_flight == "grid_mushroom":
        # 1. Generate the base grid waypoints
        base_positions = generate_grid_path(config)
        
        # 2. Initialize new, empty lists for the expanded path
        positions = []
        rotations = []
        
        # 3. Loop through each base position and create 36 entries
        for pos in base_positions:
            for i in range(36):
                # Calculate the angle (0, 10, 20, ... 350)
                angle = i * 10.0 
                
                # Append the *same* position 36 times
                positions.append(pos)
                
                # Append the new rotation
                rotations.append((0.0, 0.0, angle))
        
        # Now, 'positions' and 'rotations' are 36x longer and in sync.
        # The 'if rotations is None:' block later will be skipped, which is correct.
    else:
        raise ValueError(f"Unknown flight shape: {shape_flight}")
    
    #  housekeeping:
    if positions is None:
        raise ValueError("No positions generated. Check the shape flight type and parameters.")
    n_positions = len(positions)

    # make sure that the positions are not too close to landmarks:
    if landmark_positions is not None and shape_flight != "grid":
        for i in range(n_positions):
            pos = positions[i]
            for landmark_pos in landmark_positions:
                dist = calculate_distance(pos, landmark_pos)
                if dist < too_close:
                    # move the position away from the landmark along the vector:
                    direction = pos[:2] - landmark_pos[:2]
                    dist_xy = np.linalg.norm(direction[:2])
                    if dist_xy < 1e-5:
                        direction = np.array([too_close, 0.0])
                    else:
                        direction = direction[:2] / dist_xy
                    # make a deep copy of the position:
                    new_pos = list(landmark_pos)
                    new_pos[:2] = new_pos[:2] + direction * too_close
                    positions[i] = (new_pos[0], new_pos[1], new_pos[2])
                    print(f"Adjusted position {i} to avoid being too close to a landmark with position {landmark_pos}, from {pos} to {new_pos}.")

    if show_positions:

        if server:
            # Will have to be installed with ISAACSIM_PYTHON -m pip install PyQt5:
            matplotlib.use('Qt5Agg')

        plt.figure()
        pos = np.array(positions)
        plt.plot(pos[:, 0], pos[:, 1], 'o')
        plt.show()

    if noisy_path is None:
        noisy_path = positions

    # Generate the rotations for the cameras:
    if rotations is None:
        if(config['only_point_north']):
            # Point all cameras north
            rotations = [(0.0, 0.0, 0.0) for _ in range(n_positions)]
        else:
            # Randomly rotate the cameras around the z-axis:
            rotations = []
            for i in range(n_positions):
                rot = np.random.rand() * 2 * 360.0
                rotations.append((0.0, 0.0, rot))

    # Save images and labels using replicator
    if use_noisy_spiral:
        pos_variable = noisy_path   
    else:
        pos_variable = positions
    my_world.step(render=True)
    num_positions = len(pos_variable)
    # Each camera takes care of one subsequent chunk of positions, which are stored in the directory corresponding to the camera.
    # Images are numbered per camera from 0 to chunk_size-1.
    chunk_size = num_positions // number_cameras

    # pause the program to prevent an out of memory crash:
    print("Sleeping to prevent out of memory crash")
    time.sleep(15)
    
    print("Generating images")
    with rep.trigger.on_frame(num_frames=chunk_size, rt_subframes=2):
        for i in range(number_cameras):
            with cameras[i]:
                start_index = i * chunk_size
                end_index = (i + 1) * chunk_size if i < number_cameras - 1 else num_positions
                rep.modify.pose(position=rep.distribution.sequence(pos_variable[start_index:end_index]), 
                                rotation=rep.distribution.sequence(rotations[start_index:end_index]))
        writer = rep.writers.get("BasicWriter")
        writer.initialize(output_dir=output_folder, rgb=True)
        writer.attach(render_products)
        rep.orchestrator.run_until_complete()
    print("Images generated")

    if (shape_flight == "spiral" or shape_flight == "bee") and use_noisy_spiral:
        # Targets are calculated from the planned positions, while images are made at the noisy positions:
        targets = calculate_target_vectors(positions, rotations, home_position)
        noisy_targets = calculate_target_vectors(noisy_path, rotations, home_position)
        write_csv_dataset(config, positions, rotations, targets, noisy_path, rotations, noisy_targets)
    else:
        targets = calculate_target_vectors(positions, rotations, home_position)
        write_csv_dataset(config, positions, rotations, targets)
    
    # Crop the images:
    if config['crop']:
        print("Cropping images")
        crop_images(os.path.join(output_folder, 'Replicator/rgb'), crop_box_params = config['crop_box_params'])
        print("Cropping done")

    return my_world            
    

if __name__ == "__main__":
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'config/config_render.json'
    config = load_config(config_file)
    generate_dataset(config)
