
import numpy as np
import os
import shutil
import csv
import json


def generate_circle_centered_on_home(n_points, radius, home_position):
    ''' Generate a circle path centered on the home position
    Args:
    n_points: number of points in the circle
    radius: radius of the circle
    home_position: nest position
    Returns:
    path: list of points in the circle
    '''

    #generate n_points in a circle centered on the home position:
    x_ori = home_position[0]
    y_ori = home_position[1]
    z = home_position[2]
    # we make sure all orientations are used:
    theta = np.linspace(0, 2 * np.pi, n_points)
    # we take random distances within the circle:
    x = radius * np.random.random(theta.shape) * np.cos(theta) + x_ori
    y = radius * np.random.random(theta.shape) * np.sin(theta) + y_ori
    path = [(float(x[i]), float(y[i]), z) for i in range(n_points)]
    return path

def generate_circle_centered_on_landmarks(n_points, radius, home_position):
    ''' Generate a circle path centered on the centroid of the landmarks 
    Args:
    n_points: number of points in the circle
    radius: radius of the circle
    home_position: nest position
    
    Returns:
    path: list of points in the circle

    Please note that for the landmark positions, the function loads the config/config_simple_environment.json file. 
    The landmarks in that file hence have to correspond to the landmark positions in the environment for this path generation to make sense.
    '''

    # load the config/config_simple_environment.json file:
    with open('config/config_simple_environment.json') as f:
        config_simple_environment = json.load(f)
    
    # get the landmarks from the config/config_simple_environment.json file:
    landmarks = config_simple_environment['landmarks']
    landmarks = np.array(landmarks)
    centroid_landmarks = np.mean(landmarks, axis=0)

    #generate n_points in a circle centered on the centroid of the landmarks:
    x_ori = centroid_landmarks[0]
    y_ori = centroid_landmarks[1]
    z = home_position[2]
    # we make sure all orientations are used:
    theta = np.linspace(0, 2 * np.pi, n_points)
    # we take random distances within the circle:
    x = radius * np.random.random(theta.shape) * np.cos(theta) + x_ori
    y = radius * np.random.random(theta.shape) * np.sin(theta) + y_ori

    path = [(float(x[i]), float(y[i]), z) for i in range(n_points)]

    return path

def generate_uniform_area(n_points, home_position, area = [-10, 10, -10, 10]):
    ''' Generate a uniform area path 
    Args:
    n_points: number of points in the area
    home_position: nest position
    area: area of the square,  in the form [x_min, x_max, y_min, y_max] Default = [-10, 10, -10, 10]
    
    Returns:
    path: list of points in the area
    '''

    x = np.random.uniform(area[0], area[1], n_points)
    y = np.random.uniform(area[2], area[3], n_points)
    z = home_position[2]

    path = [(float(x[i]), float(y[i]), z) for i in range(n_points)]

    return path


def generate_spiral_path(n, m, home_position):
    ''' Generate a spiral path 
    - n: proportion of a full rotation (0,1] /  number of rotations (if bigger than 1).
    - m: number of points in the spiral
    '''

    a = 0
    b = 0.5
    x_ori = home_position[0]
    y_ori = home_position[1]
    z = home_position[2]

    theta = np.linspace(0, 2 * n * np.pi, m)
    r = a - b * theta
    x = r * np.cos(theta) + x_ori
    y = r * np.sin(theta) + y_ori

    path = [(float(x[i]), float(y[i]), z) for i in range(m)]
    return path

def add_noise_to_path(ideal_path, noise_params):

    path = []
    cumulative_yaw_noise = 0
    cumulative_time = 0
    z = ideal_path[0][2]
    prev_ideal_x = ideal_path[0][0]
    prev_ideal_y = ideal_path[0][1]
    prev_noisy_x = ideal_path[0][0]
    prev_noisy_y = ideal_path[0][1] 
    noise_x = 0
    noise_y = 0

    arw = noise_params[0]
    bi = noise_params[1]
    rrw = noise_params[2]
    rr = noise_params[3]
    velocity = noise_params[4]
    noise_per_meter = noise_params[5]

    # Conversion factors
    arw_per_sec = arw / np.sqrt(3600)  # ARW in deg/sqrt(hour) to deg/sqrt(second)
    bi_per_sec = bi / 3600  # BI in deg/hour to deg/second
    rrw_per_sec = rrw / (3600**0.5)  # RRW in deg/hour^0.5 to deg/second^1.5
    rr_per_sec = rr / (3600**2)  # RR in deg/hour^2 to deg/second^2

    for i in range(len(ideal_path)):  # Changed num_points to len(ideal_path)

        # Calculate ideal position without noise
        x_ideal = ideal_path[i][0]
        y_ideal = ideal_path[i][1]

        # Distance traveled from previous ideal point
        distance_traveled = np.sqrt((x_ideal - prev_ideal_x)**2 + (y_ideal - prev_ideal_y)**2)

        # Calculate time increment
        delta_t = distance_traveled / velocity
        cumulative_time += delta_t

        # Calculate yaw change to the next ideal point
        yaw_change = np.arctan2(y_ideal - prev_ideal_y, x_ideal - prev_ideal_x)
        
        # Add Allan Variance noise components to yaw change
        arw_noise = arw_per_sec * np.sqrt(cumulative_time) * np.random.randn()
        bi_noise = bi_per_sec * cumulative_time
        rrw_noise = rrw_per_sec * cumulative_time**(3/2) * np.random.randn()
        rr_noise = rr_per_sec * cumulative_time**2
        
        total_yaw_noise_deg = arw_noise + bi_noise + rrw_noise + rr_noise
        total_yaw_noise = np.radians(total_yaw_noise_deg) 
        
        cumulative_yaw_noise += total_yaw_noise
        noisy_yaw = yaw_change + cumulative_yaw_noise
        
        # Calculate new position with noisy yaw
        x_noisy = prev_noisy_x + distance_traveled * np.cos(noisy_yaw)
        y_noisy = prev_noisy_y + distance_traveled * np.sin(noisy_yaw)
        
        # Add noise to x and y based on distance traveled
        distance_x = distance_traveled * np.cos(noisy_yaw)
        distance_y = distance_traveled * np.sin(noisy_yaw)

        noise_x = np.random.normal(0, noise_per_meter) * distance_x
        noise_y = np.random.normal(0, noise_per_meter) * distance_y

        # Add bias to x and y
        x_noisy += noise_x
        y_noisy += noise_y

        path.append((float(x_noisy), float(y_noisy), z))

        # Update previous ideal and noisy positions
        prev_ideal_x = x_ideal
        prev_ideal_y = y_ideal
        prev_noisy_x = x_noisy
        prev_noisy_y = y_noisy
        cumulative_time += 3  # Add 2 seconds of delay at each point

    return path
    
def generate_bee_path(n, m, b, home_position, add_noise = False, noise_params = None):
    ''' Generate a bee-inspired path '''

    # for small learning area: n=4, m=36, b=0.1
    # for bigger learning area: n=5, m=56, b=0.2

    a = 0
    x_ori = home_position[0]
    y_ori = home_position[1]
    z = home_position[2]

    theta = np.linspace(0, 2 * n * np.pi, m)
    r = a + b * theta
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    path = []
    sign = 1
    for i in range(3, m):
        x_i, y_i = float(x[i]), float(y[i])
        if np.sign(x_i) != np.sign(x[i - 1]) and i != 3 and y_i < 0:
            sign *= -1
        path.append((float(sign * x_i + x_ori), float(y_i + y_ori), float(z)))

    if add_noise:
        ideal_path = path
        path = add_noise_to_path(ideal_path, noise_params)

    return path

def generate_grid_path(config):
    ''' Generate a grid path '''

    # Extract grid parameters
    grid_parameters = config['shape_params']
    x_min, x_max, y_min, y_max, grid_size = grid_parameters
    # make a mesh:
    x = np.linspace(x_min, x_max, grid_size)
    y = np.linspace(y_min, y_max, grid_size)
    z = config['home_position'][2] 

    # make a mesh with all positions:
    path = [(float(x[i]), float(y[j]), z) for i in range(grid_size) for j in range(grid_size)]
    
    return path


def generate_noisy_spiral_allan(num_rots, num_points, virtual_home_position, arw, bi, rrw, rr, velocity, noise_per_meter):
    ''' Generate a noisy spiral path with Allan variance '''
    a = 0
    b = 0.5
    x_ori = virtual_home_position[0]
    y_ori = virtual_home_position[1]
    z = virtual_home_position[2]

    theta = np.linspace(0, 2 * np.pi * num_rots, num_points)

    path = []
    cumulative_yaw_noise = 0
    cumulative_time = 0

    prev_ideal_x = x_ori
    prev_ideal_y = y_ori
    prev_noisy_x = x_ori
    prev_noisy_y = y_ori
    noise_x = 0
    noise_y = 0

    # Conversion factors
    arw_per_sec = arw / np.sqrt(3600)  # ARW in deg/sqrt(hour) to deg/sqrt(second)
    bi_per_sec = bi / 3600  # BI in deg/hour to deg/second
    rrw_per_sec = rrw / (3600**0.5)  # RRW in deg/hour^0.5 to deg/second^1.5
    rr_per_sec = rr / (3600**2)  # RR in deg/hour^2 to deg/second^2

    for i in range(num_points):
        r = a - b * theta[i]

        # Calculate ideal position without noise
        x_ideal = r * np.cos(theta[i]) + x_ori
        y_ideal = r * np.sin(theta[i]) + y_ori

        # Distance traveled from previous ideal point
        distance_traveled = np.sqrt((x_ideal - prev_ideal_x)**2 + (y_ideal - prev_ideal_y)**2)

        # Calculate time increment
        delta_t = distance_traveled / velocity
        cumulative_time += delta_t
        #print(cumulative_time)

        # Calculate yaw change to the next ideal point
        yaw_change = np.arctan2(y_ideal - prev_ideal_y, x_ideal - prev_ideal_x)
        
        # Add Allan Variance noise components to yaw change
        arw_noise = arw_per_sec * np.sqrt(cumulative_time) * np.random.randn()
        bi_noise = bi_per_sec * cumulative_time
        rrw_noise = rrw_per_sec * cumulative_time**(3/2) * np.random.randn()
        rr_noise = rr_per_sec * cumulative_time**2
        
        total_yaw_noise_deg = arw_noise + bi_noise + rrw_noise + rr_noise
        total_yaw_noise = np.radians(total_yaw_noise_deg) 
        
        
        
        cumulative_yaw_noise += total_yaw_noise
        noisy_yaw = yaw_change + cumulative_yaw_noise
        # Calculate new position with noisy yaw
        x_noisy = prev_noisy_x + distance_traveled * np.cos(noisy_yaw)
        y_noisy = prev_noisy_y + distance_traveled * np.sin(noisy_yaw)
         # Add noise to x and y based on distance traveled
        
        distance_x = distance_traveled * np.cos(noisy_yaw)
        distance_y = distance_traveled * np.sin(noisy_yaw)

        noise_x = np.random.normal(0, noise_per_meter) * distance_x
        noise_y = np.random.normal(0, noise_per_meter) * distance_y

        # Add bias to x and y
        x_noisy += noise_x
        y_noisy += noise_y

        path.append((float(x_noisy), float(y_noisy), z))

        # Update previous ideal and noisy positions
        prev_ideal_x = x_ideal
        prev_ideal_y = y_ideal
        prev_noisy_x = x_noisy
        prev_noisy_y = y_noisy
        cumulative_time += 3  # Add 2 seconds of delay at each point
    
    return path

def calculate_target_vectors(positions, rotations, home_position):
    ''' Calculate the target vector pointing towards the home position in the body frame '''
    target_vectors = np.array(home_position) - np.array(positions)
    target_vectors = target_vectors[:, :2]  # Ignore the z component

    # Rotate the direction vector by the yaw angle
    for i in range(len(target_vectors)):
        target_vectors[i] = rotate_vector_by_yaw(target_vectors[i], -rotations[i][2])
    
    return target_vectors

def rotate_vector_by_yaw(vector, yaw):
    yaw = np.radians(yaw)
    R = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw),  np.cos(yaw)]
    ])
    
    # Perform the matrix multiplication (rotate the vector)
    vector = [vector[0],vector[1]]
    rotated_vector = R.dot(vector)
    rotated_vector = [rotated_vector[0],rotated_vector[1]]
    return rotated_vector

def list_folders(location):
    # Get all items in the specified location
    items = os.listdir(location)
    
    # Filter out only the folders and extend with /rgb/
    folders = [os.path.join(location, item, 'rgb') for item in items if os.path.isdir(os.path.join(location, item))]
    
    return folders

def write_csv_dataset(config, positions, rotations, targets, noisy_positions = None, noisy_rotations = None, noisy_targets = None):
    
    # get the relevant variables from the configuration file:
    output_folder = config['output_folder']
    use_noisy_spiral = config['use_noisy_spiral']
    home_position = config['home_position']
    csv_filename = config['csv_filename']

    # Setup CSV to store targets and positions
    csv_file_path = os.path.join(output_folder, csv_filename)
    with open(csv_file_path, mode='w', newline='') as file:
        if not noisy_positions is None:
            writer = csv.writer(file)
            writer.writerow(['Filename', 'Position', 'Rotation', 'Target', 'Noisy Position', 'Noisy Rotation', 'Noisy Target'])
        else:
            writer = csv.writer(file)
            writer.writerow(['Filename', 'Position', 'Rotation', 'Target'])

    # Typically, multiple cameras are used at the same time, resulting in images in different folders.
    # Here we join these images in one folder and write the CSV file.
    
    # count will be the number of the image in the joined dataset
    count = 0
    # Open the CSV file for writing
    with open(csv_file_path, mode='a', newline='') as file:

        csv_writer = csv.writer(file)
        
        folders = list_folders(output_folder)
        folders = sorted(folders)

        # Process each folder and each file within it
        for folder in folders:
            # Get all the files in the folder
            files = sorted([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])
            for filename in files:
                # Check if we still have positions left
                if count < len(positions):
                    old_file_path = os.path.join(folder, filename)

                    # Generate a new filename using a global index
                    new_file_name = f"rgb_{count:04d}.png"  # Format index as four digits
                    print(f"Renaming {old_file_path} to {new_file_name}")
                    new_file_path = os.path.join(folder, new_file_name)

                    # Rename the file
                    os.rename(old_file_path, new_file_path)

                    # Write the old and new file names to the CSV file
                    if not noisy_positions is None:
                        csv_writer.writerow([new_file_name, positions[count], rotations[count], targets[count], noisy_positions[count], noisy_rotations[count], noisy_targets[count]])
                    else:
                        csv_writer.writerow([new_file_name, positions[count], rotations[count], targets[count]])

                    # Increment the counter
                    count += 1
                else:
                    print("Ran out of positions.")
                    break

    # Output the result
    print(f"Total files renamed and written to CSV: {count}")
    if count == 0:
        # raise an error:
        raise ValueError("No files were created - something went wrong in the data rendering.")

    # The destination folder is the first one in the list
    destination_folder = folders[0]

    # Move files from other folders to the destination folder
    for folder in folders[1:]:
        for file_name in os.listdir(folder):
            # Construct full file path
            file_path = os.path.join(folder, file_name)
            # Move file to the destination folder
            shutil.move(file_path, destination_folder)
        # remove the folder:
        os.rmdir(folder)
        index = folder.find('rgb')
        os.rmdir(folder[:index])
        

    print("Files have been moved successfully.")
    
def calculate_distance(position, virtual_home_position):
    """Calculate distance between a position and the virtual home position."""
    position_xy = np.array(position)[:2]
    virtual_home_position_xy = np.array(virtual_home_position)[:2]
    return np.linalg.norm(position_xy - virtual_home_position_xy)

def calculate_absolute_angular_error(gt_vector, pred_vector):
    """Calculate the absolute angular error between ground truth and predicted vectors."""
    unit_gt = gt_vector / np.linalg.norm(gt_vector)
    unit_pred = pred_vector / np.linalg.norm(pred_vector)
    angle_gt = np.arctan2(unit_gt[1], unit_gt[0])
    angle_pred = np.arctan2(unit_pred[1], unit_pred[0])
    angular_error = np.abs(np.degrees(angle_pred - angle_gt))
    if angular_error > 180:
        angular_error = 360 - angular_error
    return angular_error

def normalize_vectors(vectors):
    """Normalize 2D vectors to unit vectors."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

def generate_perimeter(virtual_home_position, num_points, side_length):
    """Generate a perimeter around the virtual home position."""
    half_side = side_length / 2

    # Determine how many points to place per side
    points_per_side = max(1, num_points // 4)
    linear_offsets = np.linspace(-half_side, half_side, points_per_side)

    offset_list = []
    offset_list.extend([(virtual_home_position[0] + x, virtual_home_position[1] + half_side, 1.5) for x in linear_offsets])
    offset_list.extend([(virtual_home_position[0] + x, virtual_home_position[1] - half_side, 1.5) for x in linear_offsets])
    offset_list.extend([(virtual_home_position[0] - half_side, virtual_home_position[1] + y, 1.5) for y in linear_offsets[1:-1]])
    offset_list.extend([(virtual_home_position[0] + half_side, virtual_home_position[1] + y, 1.5) for y in linear_offsets[1:-1]])
    return offset_list