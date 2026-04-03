import numpy as np
import matplotlib.pyplot as plt
from plotting import *
from output_utils import _save_or_show

def occlude_landmarks(X, X_lm, landmarks, radius = 3):
    # X world positions robot, landmarks world positions, X_lm observations in body frame
    n_samples = X.shape[0]
    n_landmarks = landmarks.shape[0]
    # copy X_lm:
    X_lm_new = np.copy(X_lm)
    if(len(X_lm.shape) == 3):
        X_lm_new = np.reshape(X_lm_new, (n_samples, -1))
    inside_landmarks = np.zeros(n_samples)
    for i in range(n_samples):
        occluded, inside_landmark = get_occluded_landmarks(landmarks, X[i], radius = radius)
        for j in range(n_landmarks):
            if(occluded[j]):
                # set the occluded observation to zero
                X_lm_new[i, 2*j:2*j+2] = 0
        if(inside_landmark):
            inside_landmarks[i] = 1
            # print('Inside landmark ', i , ': observation = ', X_lm_new[i,:])
    return X_lm_new, inside_landmarks

def get_occluded_landmarks(landmarks, ground_truth_location, radius = 3, graphics = False):
    
    # Determine observation first:
    n_landmarks = landmarks.shape[0]
    lm_body_north = np.zeros((n_landmarks, 2))
    for j in range(n_landmarks):
        lm_body_north[j,:] = landmarks[j] - ground_truth_location

    # heading is irrelevant for occlusion with an omnidirectional camera.
        
    # Use geometry to determine the intersections of the camera rays with the edges of the landmark circles:
    d_landmarks = np.linalg.norm(lm_body_north, axis=1)
    # if there is a landmark closer than the radius:

    if(np.min(d_landmarks) < radius):
        # if inside of a landmark, only that landmark is visible:
        ind = np.argmin(d_landmarks)
        occluded = np.ones(n_landmarks)
        occluded[ind] = 0
        return occluded, True
    
    # We use the fact that the intersection of the camera rays with the landmark circles are on the circle itself.
    # Hence, the angle of the intersection points is 90 degrees with the camera ray. 
    # With the distance to the landmark and the radius of the object, we can know the distance to the intersection points,
    # and their angle with respect to the world frame.
    intersections = np.zeros([n_landmarks, 4])
    intersections_world = np.zeros([n_landmarks, 4])
    angles_intersections = np.zeros([n_landmarks, 2])
    for j in range(n_landmarks):
        d_robot_circle = np.sqrt(d_landmarks[j]**2 - radius**2)
        alpha = np.arctan2(radius, d_robot_circle)
        # rotate normalized lm_body_north[j] with alpha and -alpha:
        norm_lm = lm_body_north[j] / d_landmarks[j]
        R = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
        intersections[j, 0:2] = np.dot(R, norm_lm) * d_robot_circle
        R = np.array([[np.cos(-alpha), -np.sin(-alpha)], [np.sin(-alpha), np.cos(-alpha)]])
        intersections[j, 2:4] = np.dot(R, norm_lm) * d_robot_circle
        # determine the angles of the intersections and sort them from low to high:
        angles_intersections[j,:] =  np.sort([np.arctan2(intersections[j,1], intersections[j,0]),
                                     np.arctan2(intersections[j,3], intersections[j,2])])
        intersections_world[j, 0:2] = intersections[j, 0:2] + ground_truth_location
        intersections_world[j, 2:4] = intersections[j, 2:4] + ground_truth_location
    
    # sort the distances to landmarks and get the indices:
    indices = np.argsort(d_landmarks) # sort from close to far
    occluded = np.zeros(n_landmarks)
    for j in range(n_landmarks):
        for i in range(j+1, n_landmarks):

            alpha_1 = angles_intersections[indices[j],0]
            alpha_2 = angles_intersections[indices[j],1]
            beta_1 = angles_intersections[indices[i],0]
            beta_2 = angles_intersections[indices[i],1]

            # make sure that we don't have a landmark with a negative and positive angle, 
            # occluding almost the whole field of view:
            if(alpha_1 < 0 and alpha_2 > 0):
                alpha_1 += 2 * np.pi
                if(beta_1 < 0):
                    beta_1 += 2 * np.pi
                if(beta_2 < 0):
                    beta_2 += 2 * np.pi
            if(beta_1 < 0 and beta_2 > 0):
                beta_1 += 2 * np.pi
                if(alpha_1 < 0):
                    alpha_1 += 2 * np.pi
                if(alpha_2 < 0):
                    alpha_2 += 2 * np.pi
            if alpha_1 > alpha_2:
                alpha_1, alpha_2 = alpha_2, alpha_1
            if beta_1 > beta_2:
                beta_1, beta_2 = beta_2, beta_1


            # Angles have to sorted (first the lowest, then the highest):
            # If the angles of the intersections are in between the angles of another landmark, it is occluded:
            # if((alpha_1 < beta_1 and alpha_2 > beta_2) or (alpha_1 > beta_1 and alpha_2 < beta_2)):
            if(alpha_1 < beta_1 and alpha_2 > beta_2):
                occluded[indices[i]] = 1
            
    if(graphics):
        plt.figure()
        # plot the landmarks as circles:
        # generate a list of colors, for each landmark one, with a color map:
        color_values = np.linspace(0.1, 0.9, n_landmarks)
        cmap = plt.get_cmap('viridis')

        plt.plot(ground_truth_location[0], ground_truth_location[1], 'go')

        for j in range(n_landmarks):
            lm_color = cmap(color_values[j])
            # make a dotted circle when occluded:
            if(occluded[j]):
                circle = plt.Circle(landmarks[j], radius, color=lm_color, fill=False, linestyle='--')
            else:
                circle = plt.Circle(landmarks[j], radius, color=lm_color, fill=False)
            plt.gca().add_artist(circle)
            plt.plot(landmarks[j,0], landmarks[j,1], 'o', color=lm_color)
            # add text with the landmark number:
            plt.text(landmarks[j,0], landmarks[j,1], str(j), color=lm_color)
            plt.plot(intersections_world[j,0], intersections_world[j,1], 'o', color=lm_color)
            plt.plot(intersections_world[j,2], intersections_world[j,3], 'o', color=lm_color)
            # plot a line from the robot to the intersection points:
            plt.plot([ground_truth_location[0], intersections_world[j,0]], [ground_truth_location[1], intersections_world[j,1]], '--', color=lm_color)
            plt.plot([ground_truth_location[0], intersections_world[j,2]], [ground_truth_location[1], intersections_world[j,3]], '--', color=lm_color)
            plt.axis('equal')

        _save_or_show(None, 'occlusion_visualization')

    return occluded, False

def shuffle_landmarks(X, order = None):
    # shuffle the landmarks
    n_samples, n_landmarks2 = X.shape
    n_landmarks = n_landmarks2 // 2 # each landmark has 2 coordinates

    X_new = np.zeros((n_samples, n_landmarks2))
    for i in range(n_samples):
        # determine a random order of the landmarks:
        if order is None:
            perm = np.random.permutation(n_landmarks)
        else:
            perm = order
        # shuffle the landmarks accordingly:
        for j in range(n_landmarks):
            X_new[i, 2*j:2*j+2] = X[i, 2*perm[j]:2*perm[j]+2]

    return X_new

def add_angular_noise(X, std_angle_degrees=5):
    # add noise to the angles
    n_samples, n_landmarks2 = X.shape
    n_landmarks = n_landmarks2 // 2 # each landmark has 2 coordinates

    X_new = np.zeros((n_samples, n_landmarks2))
    for i in range(n_samples):
        for j in range(n_landmarks):
            norm = np.linalg.norm(X[i, 2*j:2*j+2])
            angle = np.arctan2(X[i, 2*j+1], X[i, 2*j])
            angle += np.random.randn() * np.deg2rad(std_angle_degrees)
            X_new[i, 2*j] = np.cos(angle) * norm
            X_new[i, 2*j+1] = np.sin(angle) * norm

    return X_new

def analyze_ambiguities(inputs, xy, heading, nest_location, landmarks, region_limits, OCC, parameters = None, output_dir = None):
    # Analyze which inputs are identical or very similar:
    n_samples = inputs.shape[0]
    observation_distances = np.zeros([n_samples,n_samples])
    for i in range(n_samples):
        observation_distances[i] = np.linalg.norm(inputs[i,:] - inputs, axis=1)
    
    # show the histogram of the distances:
    plt.figure()
    plt.hist(observation_distances.flatten(), bins=100)
    plt.xlabel('Distance between observations')
    plt.ylabel('Frequency')
    plt.title('Histogram of distances between observations')
    _save_or_show(output_dir, 'observation_distance_histogram')

    observation_threshold = 1
    # find the indices of observations that have at least two distances below the threshold:
    for i in range(n_samples):
        distances = observation_distances[i]
        indices = np.where(distances < observation_threshold)[0]
        if(len(indices) > 1):
            # show the locations for which the observation is the same:
            plt.figure()
            plt.plot(xy[indices,0], xy[indices,1], 'rx')
            # plot the landmarks:
            plt.plot(landmarks[:,0], landmarks[:,1], 'bo')
            # plot the nest location:
            plt.plot(nest_location[0], nest_location[1], 'k+')
            if(OCC):
                # get the ax:
                ax = plt.gca()
                # plot circles of radius radius around the landmarks:
                for j in range(parameters['n_landmarks']):
                    circle = plt.Circle(landmarks[j], parameters['landmark_radius'], linestyle = ':', color='k', fill=False)
                    ax.add_artist(circle)
            _save_or_show(output_dir, f'ambiguous_locations_{i}')

            print(f'Observations {indices} are the same')

def add_noise_to_path(ideal_path, noise_params):

    path = []
    cumulative_yaw_noise = 0
    cumulative_time = 0
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

        path.append((float(x_noisy), float(y_noisy)))

        # Update previous ideal and noisy positions
        prev_ideal_x = x_ideal
        prev_ideal_y = y_ideal
        prev_noisy_x = x_noisy
        prev_noisy_y = y_noisy
        cumulative_time += 3  # Add 2 seconds of delay at each point

    return path

def generate_bee_path(n=5, m=150, b=0.3, add_noise = True, noise_params =  None, graphics = False, output_dir = None):
    ''' Generate a bee-inspired path '''

    if noise_params is None:
        noise_params = [0.2, 3.5, 0.03, 0.005, 2, 0.015]

    a = 0
    x_ori = 0.0
    y_ori = 0.0

    theta = np.linspace(0, 2 * n * np.pi, m)
    r = a + b * theta
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    ideal_path = []
    sign = 1
    for i in range(3, m):
        x_i, y_i = float(x[i]), float(y[i])
        if np.sign(x_i) != np.sign(x[i - 1]) and i != 3 and y_i < 0:
            sign *= -1
        ideal_path.append((float(sign * x_i + x_ori), float(y_i + y_ori)))

    noisy_path = []
    if add_noise:
        noisy_path = add_noise_to_path(ideal_path, noise_params)
        noisy_path = np.array(noisy_path)

    ideal_path = np.array(ideal_path)

    if graphics:

        # use Times New Roman font for the plot
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 11

        plt.figure()
        plt.plot(ideal_path[:,0], ideal_path[:,1], color='black', linestyle='--', label='Ideal Path')
        plt.plot(noisy_path[:,0], noisy_path[:,1], color='grey', label='Noisy Path')
        plt.legend()
        plt.xlabel('$x$ [m]')
        plt.ylabel('$y$ [m]')
        plt.axis('equal')
        _save_or_show(output_dir, 'bee_path')

    return ideal_path, noisy_path

def get_samples(n_samples, parameters = None, output_dir = None):
    # random samples in the region of interest
    if parameters['bee_path']:
        # With noise, the targets should be generated with the original path, but the observations with the noisy path:
        X_or, X = generate_bee_path(graphics = True, output_dir = output_dir)
        n_samples = X.shape[0]
    else:
        if(not parameters['circular_learning_region']):
            X = parameters['size_learning_region'] * 2 * (np.random.rand(n_samples, 2)-0.5) + parameters['offset_learning_region']
        else:
            # generate random samples in a circular region:
            offset_landmarks = np.mean(parameters['landmarks'], axis=0)
            r = parameters['size_learning_region'] * np.sqrt(np.random.rand(n_samples))
            theta = 2 * np.pi * np.random.rand(n_samples)
            X = np.column_stack([r * np.cos(theta), r * np.sin(theta)]) + parameters['offset_learning_region'] + offset_landmarks

    # random heading for the agent, at each location:
    heading = np.random.rand(n_samples) * 2 * np.pi
    # first put the landmarks in the body frame, without rotation:
    lm_body_north = np.zeros((n_samples, parameters['n_landmarks'], 2))
    for i in range(n_samples):
        for j in range(parameters['n_landmarks']):
            lm_body_north[i,j,:] = parameters['landmarks'][j] - X[i]

    if(not parameters['N']):
        # rotate in 2D with the heading:
        R = np.array([[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]])
        # Rotate the landmarks:
        lm_body = np.zeros((n_samples, parameters['n_landmarks'], 2))
        for i in range(n_samples):
            for j in range(parameters['n_landmarks']):
                lm_body[i,j] = np.dot(R[:,:,i], lm_body_north[i,j])
    else:
        lm_body = lm_body_north

    if(not parameters['AD']):
        # normalize each observed landmark to have unit norm:
        for i in range(n_samples):
            for j in range(parameters['n_landmarks']):
                lm_body[i,j] = lm_body[i,j] / np.linalg.norm(lm_body[i,j])

    # create the input to the neural network, in body frame coordinates:
    X_lm = lm_body.reshape(n_samples, -1)
    # create the targets, first in world frame coordinates:
    if not parameters['bee_path']:
        t = parameters['nest_location'] - X
    else:
        t = parameters['nest_location'] - X_or

    if(not parameters['N']):
        # Now in body by rotating t:
        for i in range(n_samples):
            t[i] = np.dot(R[:,:,i], t[i])

    if(parameters['OCC']):
        # occlude the landmarks:
        X_lm, inside_landmarks = occlude_landmarks(X, X_lm, parameters['landmarks'], radius = parameters['landmark_radius'])
    
    if(parameters['POLAR']):
        # convert to polar coordinates:
        P = cartesian_to_polar(X_lm)

    return X, X_lm, t, heading

def cartesian_to_polar(X):
    n_samples, n_landmarks2 = X.shape
    n_landmarks = n_landmarks2 // 2 # each landmark has 2 coordinates

    P = np.zeros(X.shape)
    for i in range(n_samples):
        for j in range(n_landmarks):
            P[i,2*j] = np.arctan2(X[i,2*j+1], X[i,2*j])
            P[i,2*j+1] = np.linalg.norm(X[i,2*j:2*j+2])
    return P

def polar_to_cartesian(P):
    n_samples, n_landmarks2 = P.shape
    n_landmarks = n_landmarks2 // 2 # each landmark has 2 coordinates

    X = np.zeros(P.shape)
    for i in range(n_samples):
        for j in range(n_landmarks):
            X[i,2*j] = P[i,2*j+1] * np.cos(P[i,2*j])
            X[i,2*j+1] = P[i,2*j+1] * np.sin(P[i,2*j])
    return X

def get_samples_locations(xy, heading, shuffle_order = None, parameters = None):
    n_samples = xy.shape[0]

    # transform to body frame:
    lm_body_north = np.zeros((n_samples, parameters['n_landmarks'], 2))
    for i in range(n_samples):
        for j in range(parameters['n_landmarks']):
            lm_body_north[i,j,:] = parameters['landmarks'][j] - xy[i]

    if(not parameters['N']):
        # rotate in 2D with the heading:
        R = np.array([[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]])
        # Rotate the landmarks:
        lm_body = np.zeros((n_samples, parameters['n_landmarks'], 2))
        for i in range(n_samples):
            for j in range(parameters['n_landmarks']):
                lm_body[i,j] = np.dot(R, lm_body_north[i,j])
    else:
        lm_body = lm_body_north


    if(not parameters['AD']):
        # normalize each observed landmark to have unit norm:
        for i in range(n_samples):
            for j in range(parameters['n_landmarks']):
                lm_body[i,j] = lm_body[i,j] / np.linalg.norm(lm_body[i,j])

    if(parameters['OCC']):
        # occlude the landmarks (has to happen before shuffling):
        lm_body, inside_landmarks = occlude_landmarks(xy, lm_body, parameters['landmarks'], radius = parameters['landmark_radius'])

    inputs = lm_body.reshape(n_samples, -1)

    if(not parameters['ID']):
        # shuffle the landmarks
        inputs = shuffle_landmarks(inputs, shuffle_order)

    if(parameters['angular_noise']):
        inputs = add_angular_noise(inputs)

    return inputs

def get_grid_samples(n_grid_points = 25, order = None, show_inputs = False, single_heading = False, parameters = None):
    x = np.linspace(-parameters['size_region'], parameters['size_region'], n_grid_points)
    y = np.linspace(-parameters['size_region'], parameters['size_region'], n_grid_points)
    x, y = np.meshgrid(x, y)
    xy = np.column_stack([x.flatten(), y.flatten()])
    n_samples_grid = xy.shape[0]

    # determine the predicted directions for different headings:
    if(not parameters['N'] and not single_heading):
        # headings = [np.pi/4]
        headings = [0, np.pi/4, np.pi/2]
        # headings = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
        # headings = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4]
    else:
        headings = [0]

    #  ground truth locations and headings:
    n_headings = len(headings)
    xy = np.tile(xy, (n_headings, 1))
    heading = np.repeat(headings, n_samples_grid, axis=0)

    n_samples = n_samples_grid * n_headings

    # transform to body frame:
    lm_body_north = np.zeros((n_samples, parameters['n_landmarks'], 2))
    for i in range(n_samples):
        for j in range(parameters['n_landmarks']):
            lm_body_north[i,j,:] = parameters['landmarks'][j] - xy[i]

    if(not parameters['N']):
        # rotate in 2D with the heading:
        R = np.array([[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]])
        # Rotate the landmarks:
        lm_body = np.zeros((n_samples, parameters['n_landmarks'], 2))
        for i in range(n_samples):
            for j in range(parameters['n_landmarks']):
                lm_body[i,j] = np.dot(R[:,:,i], lm_body_north[i,j])
    else:
        lm_body = lm_body_north


    if(not parameters['AD']):
        # normalize each observed landmark to have unit norm:
        for i in range(n_samples):
            for j in range(parameters['n_landmarks']):
                lm_body[i,j] = lm_body[i,j] / np.linalg.norm(lm_body[i,j])

    if(parameters['OCC']):
        # occlude the landmarks (has to happen before shuffling):
        lm_body, inside_landmarks = occlude_landmarks(xy, lm_body, parameters['landmarks'], radius = parameters['landmark_radius'])

    inputs = lm_body.reshape(n_samples, -1)

    if(not parameters['ID']):
        # shuffle the landmarks
        inputs = shuffle_landmarks(inputs, order = order)

    if(parameters['angular_noise']):
        inputs = add_angular_noise(inputs)

    if(show_inputs):
        plot_inputs(inputs, xy, headings, n_samples_grid, n_headings)

    # here we can analyze the relation between locations and inputs:
    if(parameters['analyze_ambiguity']):
        # only pass inputs from one heading:
        analyze_ambiguities(inputs[:n_samples_grid], xy[:n_samples_grid], heading[:n_samples_grid], parameters['nest_location'], parameters['landmarks'], [-parameters['size_region'], parameters['size_region']], parameters['OCC'])

    return x, y, xy, heading, inputs, n_samples_grid, n_headings, headings, n_grid_points