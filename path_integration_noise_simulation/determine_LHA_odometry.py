import json
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.lines import Line2D
from tqdm import tqdm
from skopt import gp_minimize
from skopt.space import Real

# In this simple simulation script, we simulate a drone flying outwards from a starting position and then trying to return home.
# The drone has a noisy gyroscope that accumulates error over time.
# Moreover, it has a noisy distance sensor that accumulates error over time.
# In the script, both the true and estimated positions and yaw angles are stored.

def _save_or_show(output_dir, filename):
    ''' Save the current figure to output_dir as PDF and PNG, or show interactively if output_dir is None. '''
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{filename}.pdf'), bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f'{filename}.png'), bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()

# Simulate the outbound flight:
def simulate_outbound(config, debug = False, output_dir = None):
    ''' Simulate an outbound flight with a defined behavior (e.g., random_walk) and odometry model.
    '''

    # number of time steps and step size:
    num_time_steps = config["num_time_steps"]
    step_size = config["velocity"] * config["delta_t"]

    # Estimated position is the position assumed by the drone - unaware of the noise.
    estimated_position = np.zeros((num_time_steps, 2))
    # Noisy position is the actual position of the drone, influenced by the noise.
    noisy_position = np.zeros((num_time_steps, 2))

    # leave in random directions:
    estimated_angle = np.random.uniform(-np.pi, np.pi) * np.ones((num_time_steps, 1))
    noisy_angle = estimated_angle * np.ones((num_time_steps, 1))

    # We can have different behaviors:
    if config["behavior"] == "random_walk":
        max_turn_angle = np.deg2rad(config["max_turn_angle"])
    elif config["behavior"] == "square":
        square_length = config["square_length"]
        states = ["right", "up", "left", "down"]
        current_state = 0
        actions = np.asarray([[1, 0], [0, 1], [-1, 0], [0, -1]])
        distance_in_state = 0
    elif config["behavior"] == "outbound_robot":
        states = ["left", "right", "right", "left"]
        current_state = 0
        actions = np.asarray([-90, 90, 90, -90])
        actions = np.deg2rad(actions)
        leg_length = config["outbound_robot_leg_length"] 
        distance_in_state = 0   

    # We can have noise per meter or per second (typically, distance noise per meter and yaw noise per second):
    if config.get("distance_noise_meter", True):
        distance_noise_var_per_meter = config["distance_noise_std_per_meter"]**2
    else:
        distance_noise_var_per_second = config["distance_noise_std_per_second"]**2
    if config.get("yaw_noise_meter", True):
        yaw_rate_noise_var_per_meter = np.deg2rad(config["yaw_rate_noise_std_per_meter"])**2
    else:
        yaw_rate_noise_var_per_second = np.deg2rad(config["yaw_rate_noise_std_per_second"])**2

    # Simulation loop:
    for i in range(1, num_time_steps):

        if config["behavior"] == "random_walk":
            turn = np.random.uniform(-max_turn_angle, max_turn_angle)
        elif config["behavior"] == "square":
            if distance_in_state >= square_length:
                # change state
                current_state = (current_state + 1) % 4
                distance_in_state = 0
            turn = np.arctan2(actions[current_state, 1], actions[current_state, 0]) - estimated_angle[i-1, 0]
            distance_in_state += step_size
        elif config["behavior"] == "outbound_robot":
            if distance_in_state >= leg_length:
                # change state
                turn = actions[current_state]
                current_state = (current_state + 1) % 4
                distance_in_state = 0
            else:
                turn = 0
            distance_in_state += step_size
        elif config["behavior"] == "straight":
            turn = 0

        estimated_angle[i] = estimated_angle[i-1] + turn
        noisy_angle[i] = noisy_angle[i-1] + turn

        # The robot thinks it moves forward with a different step size than it actually does:
        if config.get("distance_noise_meter", True):
            step_size_noise = np.random.normal(0, np.sqrt(distance_noise_var_per_meter * step_size))
        else:
            step_size_noise = np.random.normal(0, np.sqrt(distance_noise_var_per_second * config["delta_t"]))
        noisy_step_size = step_size + step_size_noise 

        # The robot thinks it turns with a different yaw rate than it actually does:
        if config.get("yaw_noise_meter", True):
            yaw_rate_noise = np.random.normal(0, np.sqrt(yaw_rate_noise_var_per_meter * step_size))
        else:
            yaw_rate_noise = np.random.normal(0, np.sqrt(yaw_rate_noise_var_per_second * config["delta_t"]))
        
        if config.get("heading_measurement", True):
            # If the drone has a heading measurement, we assume it can correct its yaw angle error each step.
            # The measurement is then only subject to instantaneous noise, not accumulated noise.
            noisy_angle[i] = estimated_angle[i] + yaw_rate_noise
        else:
            # Without absolute heading measurement, the yaw noise accumulates over time.
            noisy_angle[i] += yaw_rate_noise

        estimated_position[i] = estimated_position[i-1] + step_size * np.array([np.cos(estimated_angle[i,0]), np.sin(estimated_angle[i,0])])
        noisy_position[i] = noisy_position[i-1] + noisy_step_size * np.array([np.cos(noisy_angle[i,0]), np.sin(noisy_angle[i,0])])
    
    if debug:
        # plot the estimated and actual, noisy position over time:
        plt.figure(figsize=(10, 10))
        plt.plot(noisy_position[:, 0], noisy_position[:, 1], 'k--', label='Actual Position')
        plt.plot(estimated_position[:, 0], estimated_position[:, 1], label='Estimated Position')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.axis('equal')
        # set the axis limits to [-4,4, -4,4]:
        xmin = min(np.min(estimated_position[:, 0]), np.min(noisy_position[:, 0]))
        xmax = max(np.max(estimated_position[:, 0]), np.max(noisy_position[:, 0]))
        ymin = min(np.min(estimated_position[:, 1]), np.min(noisy_position[:, 1]))
        ymax = max(np.max(estimated_position[:, 1]), np.max(noisy_position[:, 1]))
        lim = 4
        xmin = min(xmin, -lim)
        xmax = max(xmax, lim)
        ymin = min(ymin, -lim)
        ymax = max(ymax, lim)
        plt.xlim([xmin-0.5, xmax+0.5])
        plt.ylim([ymin-0.5, ymax+0.5])
        plt.legend()
        _save_or_show(output_dir, 'position_trajectories')

        # plot the noise over distance:
        distance = np.arange(0, num_time_steps * step_size, step_size)
        # plot the position error over distance:
        position_error = np.linalg.norm(estimated_position - noisy_position, axis=1)
        plt.figure(figsize=(10, 5))
        plt.plot(distance, position_error)
        plt.title('Position error over distance')
        plt.xlabel('Distance [m]')
        plt.ylabel('Position error [m]')
        _save_or_show(output_dir, 'position_error_vs_distance')

        # plot the yaw angle error over distance:
        angle_error = np.abs(estimated_angle - noisy_angle)
        plt.figure(figsize=(10, 5))
        plt.plot(distance, np.rad2deg(angle_error))
        plt.title('Yaw angle error over distance')
        plt.xlabel('Distance [m]')
        plt.ylabel('Yaw angle error [deg]')
        _save_or_show(output_dir, 'yaw_error_vs_distance')

        # plot the yaw angle error over time:
        time = np.arange(0, num_time_steps * config["delta_t"], config["delta_t"])
        plt.figure(figsize=(10, 5))
        plt.plot(time, np.rad2deg(angle_error))
        plt.title('Yaw angle error over time')
        plt.xlabel('Time [s]')
        plt.ylabel('Yaw angle error [deg]')
        _save_or_show(output_dir, 'yaw_error_vs_time')

    return estimated_position, noisy_position, estimated_angle, noisy_angle

def simulate_return(config, estimated_position, noisy_position, estimated_angle, noisy_angle, initial_position = [0.0,0.0], debug = False, output_dir = None):
    ''' Simulate the return of the drone to the home position.
    
        The drone tries to return straight home, but due to drift it typically misses the nest.
    '''

    step_size = config["return_velocity"] * config["delta_t"]

    end_position_noisy = noisy_position[-1]
    return_estimated_position = []
    return_noisy_position = []
    return_estimated_position.append(estimated_position[-1])
    return_noisy_position.append(end_position_noisy)
    return_estimated_angle = estimated_angle[-1,0]
    return_noisy_angle = noisy_angle[-1,0]
    delta_angle = return_noisy_angle - return_estimated_angle

    if config.get("distance_noise_meter", True):
        distance_noise_var_per_meter = config["distance_noise_std_per_meter"]**2
    else:
        distance_noise_var_per_second = config["distance_noise_std_per_second"]**2
    if config.get("yaw_noise_meter", True):
        yaw_rate_noise_var_per_meter = np.deg2rad(config["yaw_rate_noise_std_per_meter"])**2
    else:
        yaw_rate_noise_var_per_second = np.deg2rad(config["yaw_rate_noise_std_per_second"])**2

    # Keep taking steps if the drone is not home yet
    while np.linalg.norm(return_estimated_position[-1] - initial_position) > step_size:

        # The drone tries to know where home is and goes there
        global_direction_vector = initial_position - return_estimated_position[-1]
        global_direction_unit_vector = global_direction_vector / np.linalg.norm(global_direction_vector)
        desired_angle = np.arctan2(global_direction_unit_vector[1], global_direction_unit_vector[0])

        # The robot thinks it turns with a different yaw rate than it actually does:
        if config.get("yaw_noise_meter", True):
            yaw_rate_noise = np.random.normal(0, np.sqrt(yaw_rate_noise_var_per_meter * step_size))
        else:
            yaw_rate_noise = np.random.normal(0, np.sqrt(yaw_rate_noise_var_per_second * config["delta_t"]))
        if config.get("heading_measurement", True):
            # If the drone has a heading measurement, we assume it can correct its yaw angle error each step.
            # The measurement is then only subject to instantaneous noise, not accumulated noise.
            delta_angle = yaw_rate_noise
        else:
            delta_angle += yaw_rate_noise

        # The robot thinks it moves forward with a different step size than it actually does:
        if config.get("distance_noise_meter", True):
            step_size_noise = np.random.normal(0, np.sqrt(distance_noise_var_per_meter * step_size))
        else:
            step_size_noise = np.random.normal(0, np.sqrt(distance_noise_var_per_second * config["delta_t"]))
        noisy_step_size = step_size + step_size_noise 

        # The difference between outbound and inbound journey is the following:
        # For the outbound journey, the drone movement is independent of its estimate, so the estimate does not influence the actual position.
        # For the inbound journey, the drone tries to move in a straight line towards home, so the estimate does influence the actual position.
        return_estimated_position.append(return_estimated_position[-1] + step_size * np.array([np.cos(desired_angle), np.sin(desired_angle)]))
        return_noisy_position.append(return_noisy_position[-1] + noisy_step_size * np.array([np.cos(desired_angle + delta_angle), np.sin(desired_angle + delta_angle)]))

    #  Prepare the return values:
    true_distance_to_home = np.linalg.norm(return_noisy_position[-1] - initial_position)
    return_estimated_position = np.array(return_estimated_position)
    return_noisy_position = np.array(return_noisy_position)

    if debug:
        # plot the estimated and noisy position over time:
        plt.figure(figsize=(10, 10))
        plt.plot(return_estimated_position[:, 0], return_estimated_position[:, 1], label='Estimated Position')
        plt.plot(return_noisy_position[:, 0], return_noisy_position[:, 1], label='Noisy Position')
        plt.plot(initial_position[0], initial_position[1], 'go', label='Home Position')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        _save_or_show(output_dir, 'return_trajectory')

    return return_estimated_position, return_noisy_position, true_distance_to_home

def cost_drift(x, target_drift_m, target_drift_yaw_deg, target_distance, num_simulations, num_time_steps, step_size, max_turn_angle):
    ''' Cost function for optimization of noise parameters.

        The cost is the squared difference between the target drift and the actual drift after flying a certain distance.
    '''

    # get the noise parameters from the x vector, to be optimized:
    distance_noise_var_per_meter = x[0]**2
    yaw_rate_noise_var_per_meter = x[1]**2
    print(f'Testing distance noise std per meter: {x[0]}, yaw rate noise std per meter: {x[1]}')

    # get the drift at the target distance:
    drift_after_target_distance = get_drift_after_distance(target_distance, num_simulations, num_time_steps, step_size, max_turn_angle, distance_noise_var_per_meter, yaw_rate_noise_var_per_meter)
    mean_drift_m = np.mean(drift_after_target_distance[:,0])
    mean_drift_yaw_deg = np.mean(drift_after_target_distance[:,1])

    # compare the drift with the desired drift:
    cost = (mean_drift_m - target_drift_m)**2 + (mean_drift_yaw_deg - target_drift_yaw_deg)**2
    
    return cost

def get_drift_after_distance(target_distance, num_simulations, num_time_steps, step_size, max_turn_angle, distance_noise_var_per_meter, yaw_rate_noise_var_per_meter):
    ''' Get the drift after a certain distance.'''
    drift_after_target_distance = np.zeros(num_simulations)

    distances, end_positions, furthest_positions, max_distances, noisy_position, return_noisy_position, outbound_lengths, inbound_lengths, drift_after_outbound, drifts_Xm, drifts_Xs = \
        get_statistics_multiple_runs(config, drift_after_Xm=[target_distance], drift_after_Xs=[])

    return np.mean(drifts_Xm)

def get_statistics_multiple_runs(config, drift_after_Xm =[35, 100], drift_after_Xs = [100]):
    ''' Perform multiple runs of outbound and inbound flights and return statistics. 
    Returns:
    - distances: the distances to home at the end of each run
    - end_positions: the end positions of the drone after the inbound flight
    - furthest_positions: the furthest positions of the drone from home during the outbound flight
    - max_distances: the furthest distances from home during the outbound flight
    '''

    if len(config['target_distance']) > 0:
        drift_after_Xm = config['target_distance']

    num_simulations = config["num_simulations"]
    distances = np.zeros(num_simulations)
    end_positions = np.zeros((num_simulations, 2))
    furthest_positions = np.zeros((num_simulations, 2))
    max_distances = np.zeros(num_simulations)
    outbound_lengths = np.zeros(num_simulations)
    inbound_lengths = np.zeros(num_simulations)
    drift_after_outbound = np.zeros(num_simulations)
    drifts_Xm = np.zeros((num_simulations, len(drift_after_Xm), 2))
    drifts_Xs = np.zeros((num_simulations, len(drift_after_Xs), 2))

    for i in tqdm(range(num_simulations), desc="Simulations"):
        # simulate the outbound:
        estimated_position, noisy_position, estimated_angle, noisy_angle = \
            simulate_outbound(config)
        
        # determine position drift after a certain distance:
        # get the flown distances:
        flown_distances = np.zeros(len(noisy_position))
        for j in range(len(noisy_position)):
            if j == 0:
                flown_distances[j] = 0
            else:
                flown_distances[j] = flown_distances[j-1] + np.linalg.norm(noisy_position[j] - noisy_position[j-1])
        
        for j, Xm in enumerate(drift_after_Xm):
            # find the index where the distance from home is just above Xm:
            index_Xm = np.where(flown_distances >= Xm)[0]
            if len(index_Xm) > 0:
                index_Xm = index_Xm[0]
                drifts_Xm[i, j, 0] = np.linalg.norm(noisy_position[index_Xm] - estimated_position[index_Xm])
                drifts_Xm[i, j, 1] = np.rad2deg(np.abs(noisy_angle[index_Xm] - estimated_angle[index_Xm]))
            else:
                # take the furthest distance reached:
                # give a warning:
                print(f'Warning: Drone did not reach {Xm}m in simulation {i}. Taking drift at maximum distance {flown_distances[-1]:.2f}m.')
                drifts_Xm[i, j, 0] = np.linalg.norm(noisy_position[-1] - estimated_position[-1])
                drifts_Xm[i, j, 1] = np.rad2deg(np.abs(noisy_angle[-1] - estimated_angle[-1]))
        
        # determine yaw drift after a certain time:
        for j, Xs in enumerate(drift_after_Xs):
            # find the index of time where the time is just above Xs:
            index_Xs = int(Xs / (config["delta_t"]))
            if index_Xs < len(noisy_angle):
                drifts_Xs[i, j, 0] = np.linalg.norm(noisy_position[index_Xs] - estimated_position[index_Xs])
                drifts_Xs[i, j, 1] = np.rad2deg(np.abs(noisy_angle[index_Xs] - estimated_angle[index_Xs]))
                        
        drift_after_outbound[i] = np.linalg.norm(noisy_position[-1] - estimated_position[-1])
        furthest_positions[i], max_distances[i] = get_furthest_position(noisy_position)
        outbound_lengths[i] = calculate_flown_distance(noisy_position)

        # simulate the return:
        return_estimated_position, return_noisy_position, true_distance_to_home = \
            simulate_return(config, estimated_position, noisy_position, estimated_angle, noisy_angle)

        inbound_lengths[i] = calculate_flown_distance(return_noisy_position)
        distances[i] = true_distance_to_home
        end_positions[i] = return_noisy_position[-1]
    
    if len(drift_after_Xm) > 0:
        for i, Xm in enumerate(drift_after_Xm):
            print(f'Mean position drift after {Xm}m: {np.mean(drifts_Xm[:, i, 0]):.2f}m')
            print(f'Mean yaw drift after {Xm}m: {np.mean(drifts_Xm[:, i, 1]):.2f}°')
    if len(drift_after_Xs) > 0:
        for i, Xs in enumerate(drift_after_Xs):
            print(f'Mean position drift after {Xs}s: {np.mean(drifts_Xs[:, i, 0]):.2f}m')
            print(f'Mean yaw drift after {Xs}s: {np.mean(drifts_Xs[:, i, 1]):.2f}°')

    return distances, end_positions, furthest_positions, max_distances, noisy_position, return_noisy_position, outbound_lengths, inbound_lengths, drift_after_outbound, drifts_Xm, drifts_Xs

# general functions:
def calculate_statistics(distances, max_distances, perc = 99):
    ''' Calculate statistics of the end of max distances to home of the drone.'''

    radius_LHA = np.percentile(distances, perc)
    radius_outbound = np.percentile(max_distances, perc)

    area_LHA = np.pi * radius_LHA**2
    area_outbound = np.pi * radius_outbound**2

    ratio_area = area_LHA / area_outbound
    ratio_radius = radius_LHA / radius_outbound

    return ratio_area, ratio_radius, radius_LHA, radius_outbound, area_LHA, area_outbound

def get_furthest_position(path):
    ''' Get the furthest position from the home position in the path. '''
    distances = np.linalg.norm(path, axis=1)
    max_distance = np.max(distances)
    max_distance_index = np.argmax(distances)
    return path[max_distance_index], max_distance

def calculate_flown_distance(path):
    ''' Calculate the total distance flown by the drone. '''
    return np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))

def display_statistics(config, distances, end_positions, furthest_positions, max_distances, true_position, return_true_position, outbound_lengths, inbound_lengths, drift_after_outbound, output_dir=None):
    # Calculate the percentiles of the distances to home, to determine what radius the hive catchment area should have.
    percentiles = config["percentiles"]
    percentile_values = np.percentile(distances, percentiles)
    for perc, value in zip(percentiles, percentile_values):
        print(f"{perc}th percentile distance to home: {value:.2f}")

    # TODO: 99 is still hard-coded here.
    radius_LHA = np.percentile(distances, 99)
    radius_outbound = np.percentile(max_distances, 99)

    # Plot the end positions of the drone for all simulations
    plt.figure(figsize=(10, 10))
    # plot the end positions after odometry-based homing in grey
    plt.scatter(end_positions[:, 0], end_positions[:, 1], marker='.', color=(0.4,0.4,0.4))
    # plot the furthest positions in lighter grey:
    plt.scatter(furthest_positions[:, 0], furthest_positions[:, 1], marker='.', color=(0.7,0.7,0.7))
    # plot the home position in red:
    plt.scatter(0, 0, marker='x', color='red', label='Home position')
    # draw a dashed circle around the home position to indicate the catchment area
    circle = plt.Circle((0, 0), radius_LHA, color='red', fill=False, linestyle='dashed')
    plt.gca().add_artist(circle)
    # draw a dashed circle around the home position to indicate the outbound radius
    circle_outbound = plt.Circle((0,0), radius_outbound, color=(0.7,0.7,0.7), fill=False, linestyle='dotted')
    plt.gca().add_artist(circle_outbound)
    # plot the path consisting of true position and return_true_position in black
    #  plot to outbound in blue:
    plt.plot(true_position[:, 0], true_position[:, 1], color='blue', label='Outbound path')
    #  plot the return path in orange:
    plt.plot(return_true_position[:, 0], return_true_position[:, 1], color='orange', label='Return path')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('End positions of the drone')
    # scale the axes equally:
    plt.axis('equal')
    _save_or_show(output_dir, 'end_positions')

    # plot a histogram of the distances to home
    bin_edges = np.linspace(0, max(max(distances), max(max_distances)), 50)

    plt.figure(figsize=(10, 5))
    plt.hist(distances, bins=bin_edges, facecolor='blue', alpha=0.75, label='Inbound distance to home')
    # plot a histogram of the furthest distances from home
    plt.hist(max_distances, bins=bin_edges, facecolor='green', alpha=0.75, label='Outbound distance from home')
    plt.xlabel('Distance from home [m]')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Histograms of distances from home')
    _save_or_show(output_dir, 'distance_histogram')

    print(f"99th percentile furthest distance to home: {radius_outbound:.2f}")
    print(f"99th percentile distance to home: {radius_LHA:.2f}")
    area_LHA = np.pi * radius_LHA**2
    area_outbound = np.pi * radius_outbound**2
    ratio = area_LHA / area_outbound
    if len(config['target_distance']) == 0 and config['behavior'] == 'outbound_robot':
        print(f'Ratio of LHA area divided by outbound area: {ratio:4.4f}')
        print(f'Percentage of the outbound area that is covered by the LHA: {ratio*100:4.4f}%')

    ratio_radius = radius_LHA / radius_outbound
    if len(config['target_distance']) == 0 and config['behavior'] == 'outbound_robot':
        print(f'Ratio of LHA radius divided by outbound radius: {ratio_radius:4.4f}')
        print(f'Percentage of the outbound radius that is covered by the LHA: {ratio_radius*100:4.4f}%')

    ratio_inbound_outbound = np.mean(inbound_lengths / outbound_lengths)
    print(f'Ratio of inbound length divided by outbound length: {ratio_inbound_outbound:4.4f}')

    # plot a histogram of the outbound drift:
    plt.figure(figsize=(10, 5))
    plt.hist(drift_after_outbound, bins=50, facecolor='magenta', alpha=0.75)
    plt.xlabel('Drift after outbound [m]')
    plt.ylabel('Frequency')
    plt.title('Histogram of drift after outbound')
    _save_or_show(output_dir, 'drift_histogram')

    mean_drift = np.mean(drift_after_outbound)
    mean_drift_per_meter = mean_drift / calculate_flown_distance(true_position)
    num_time_steps = config["num_time_steps"]
    delta_t = config["delta_t"]
    mean_drift_per_second = mean_drift / (num_time_steps * delta_t)

    print(f'Mean drift after outbound: {np.mean(drift_after_outbound):4.4f}')
    print(f'Mean outbound distance flown: {calculate_flown_distance(true_position):4.4f}')
    print(f'Mean drift after outbound per meter flown: {mean_drift_per_meter:4.4f}')
    print(f'Mean drift after outbound per second flown: {mean_drift_per_second:4.4f}')

    # Save statistics to CSV
    if output_dir is not None:
        stats = {
            'num_simulations': config['num_simulations'],
            'behavior': config['behavior'],
            'distance_noise_std_per_meter': config['distance_noise_std_per_meter'],
            'yaw_rate_noise_std_per_second': config['yaw_rate_noise_std_per_second'],
            'heading_measurement': config.get('heading_measurement', False),
            'radius_LHA_99pct': f'{radius_LHA:.4f}',
            'radius_outbound_99pct': f'{radius_outbound:.4f}',
            'area_ratio': f'{ratio:.4f}',
            'radius_ratio': f'{ratio_radius:.4f}',
            'ratio_inbound_outbound': f'{ratio_inbound_outbound:.4f}',
            'mean_drift_after_outbound': f'{mean_drift:.4f}',
            'mean_outbound_distance': f'{calculate_flown_distance(true_position):.4f}',
            'mean_drift_per_meter': f'{mean_drift_per_meter:.4f}',
            'mean_drift_per_second': f'{mean_drift_per_second:.4f}',
        }
        for perc, value in zip(percentiles, percentile_values):
            stats[f'percentile_{perc}_distance_to_home'] = f'{value:.4f}'

        csv_path = os.path.join(output_dir, 'statistics.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            for key, val in stats.items():
                writer.writerow([key, val])

        # Also save the config used for this experiment
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)

def get_ratios(config, percentages):
    ''' Get the ratios of LHA area to outbound area and LHA radius to outbound radius for different noise settings.'''
    noise_ratios = config["noise_ratios"]
    # Loop over all noise settings:
    n_noise_settings = len(noise_ratios)
    n_percentages = len(percentages)
    ratio_areas = np.zeros([n_noise_settings,  n_percentages])
    ratio_radii = np.zeros([n_noise_settings, n_percentages])
    ratio_inbound_outbound = np.zeros(n_noise_settings)

    for i in range(len(noise_ratios)):

        noise_ratio = noise_ratios[i]
        print(f'Noise ratio: {noise_ratio}')

        # copy the config to avoid modifying the original:
        config_nr = config.copy()
        config_nr["distance_noise_std_per_meter"] *= noise_ratio
        config_nr["yaw_rate_noise_std_per_meter"] *= noise_ratio
        config_nr["distance_noise_std_per_second"] *= noise_ratio
        config_nr["yaw_rate_noise_std_per_second"] *= noise_ratio

        # Perform the simulation runs for a given noise setting:
        distances, end_positions, furthest_positions, max_distances, true_position, return_true_position, \
            outbound_lengths, inbound_lengths, drift_after_outbound, drift_after_Xm, drift_after_Xs = \
            get_statistics_multiple_runs(config_nr)

        ratio_inbound_outbound[i] = np.mean(inbound_lengths / outbound_lengths)
        print(f'Ratio of inbound length divided by outbound length: {ratio_inbound_outbound[i]:4.4f}')

        for j in range(n_percentages):
            perc = percentages[j]
            ratio_area, ratio_radius, radius_LHA, radius_outbound, area_LHA, area_outbound = \
                calculate_statistics(distances, max_distances, perc = perc)
        
            ratio_areas[i, j] = ratio_area
            ratio_radii[i, j] = ratio_radius
            print(f'Percentage = {perc}: Ratio of LHA area divided by outbound area: {ratio_area:4.4f}')
            print(f'Percentage = {perc}: Ratio of LHA radius divided by outbound radius: {ratio_radius:4.4f}')
    
    return ratio_areas, ratio_radii, ratio_inbound_outbound
    
def main(config = None, output_dir = None, experiment_name = None):
    seednr = 2025
    np.random.seed(seednr)

    # Load the configuration from the JSON file
    if config is None:
        with open('config_odometry.json', 'r') as f:
            config = json.load(f)

    # Set up the experiment output directory
    exp_output_dir = None
    if output_dir is not None and experiment_name is not None:
        exp_output_dir = os.path.join(output_dir, experiment_name)
        os.makedirs(exp_output_dir, exist_ok=True)

    # Extract parameters from config
    num_simulations = config["num_simulations"]

    if num_simulations > 1:
        SINGLE_SIM = False
    else:
        SINGLE_SIM = True

    # Whether to investigate noise ratios (many experiments).
    investigate_noise_ratios = config["investigate_noise_ratios"]
    initial_position = np.array(config["initial_position"])

    # Percentiles of final distances between the noisy position and the initial, home position.
    # This is used to calculate the home catchment area radius required for returning with a given certainty.
    percentiles = config["percentiles"]

    if config["optimization"]:

        # use scipy's bayesian optimization to find the best noise parameters:
        # "distance_noise_std_per_meter"
        # "yaw_rate_noise_std_per_meter"
        noise_parameter_space = [
            Real(0.001, 0.5, name="distance_noise_std"),
            Real(0.001, 1.0, name="yaw_rate_noise_std")
        ]
        res = gp_minimize(
            func=lambda x: cost_drift(x, config),
            dimensions=noise_parameter_space,
            acq_func="EI",  # Expected Improvement
            n_calls=250,
            n_initial_points=10,
            random_state=42,
        )

        print('\n**************************\n')
        print("Best parameters found:")
        print(f"Distance noise std: {res.x[0]:.4f}")
        print(f"Yaw rate noise std: {res.x[1]:.4f}")
        print(f"Cost: {res.fun:.4f}")
        print('\n**************************\n')
        exit()

    # Running the simulation
    if SINGLE_SIM:
        estimated_position, noisy_position, estimated_angle, noisy_angle = \
            simulate_outbound(config, debug = True, output_dir = exp_output_dir)

        return_estimated_position, return_noisy_position, true_distance_to_home = \
            simulate_return(config, estimated_position, noisy_position, estimated_angle, noisy_angle, debug = True, output_dir = exp_output_dir)

    elif not investigate_noise_ratios:
        #  Perform num_simulations simulations and determine the end distances to home and end positions:
        distances, end_positions, furthest_positions, max_distances, true_position, return_true_position, \
            outbound_lengths, inbound_lengths, drift_after_outbound, drift_after_Xm, drift_after_Xs = \
            get_statistics_multiple_runs(config)
        # display the results:
        display_statistics(config, distances, end_positions, furthest_positions, \
            max_distances, true_position, return_true_position, outbound_lengths, inbound_lengths, drift_after_outbound, output_dir=exp_output_dir)
    else:
        noise_ratios = config["noise_ratios"]
        percentages = config['percentiles']
        n_percentages = len(percentages)
        # find the index of 99 in percentages:
        for i in range(n_percentages):
            if np.abs(percentages[i] - 99) < 1e-3:
                index_99 = i
        # determine the ratios with yaw drift:
        config['heading_measurement'] = False
        ratio_areas_yd, ratio_radii_yd, ratio_inbound_outbound_yd = get_ratios(config, percentages)
        # determine the ratios without yaw drift:
        config['heading_measurement'] = True
        ratio_areas_nyd, ratio_radii_nyd, ratio_inbound_outbound_nyd = get_ratios(config, percentages)

        # Show the results:
        # Purple: RGB (93, 58, 155) or RGB (116, 2, 177)
        # Dark Yellow: RGB (174, 159, 15) or RGB (186, 142, 35
        colors = [(93/255, 58/255, 155/255), (174/255, 159/255, 15/255)]
        #  Plot the percentage of the area covered by the LHA as a function of the noise ratio:
        plt.figure(figsize=(10, 5))
        # plot the percentage of the area covered by the LHA as a function of the noise ratio.
        # use a black line with a dashed style:
        styles = ['-', '--', '-.']
        widths = [1, 1.5, 2]
        for j in range(n_percentages):
            if j == index_99:
                # add a label for the legend:
                plt.plot(noise_ratios, 100 * ratio_areas_yd[:, j], linestyle=styles[0], color = colors[0], linewidth = widths[j], label='Without compass')
            else:
                plt.plot(noise_ratios, 100 * ratio_areas_yd[:, j], linestyle=styles[0], color = colors[0], linewidth = widths[j])

            plt.text(noise_ratios[-1], 100 * ratio_areas_yd[-1, j], f'{percentages[j]}%', fontsize=8, ha='right', va='center', color=colors[0])
            if j == index_99: # only plot the 99% one for without yaw drift
                if j == index_99:
                    # add a label for the legend:
                    plt.plot(noise_ratios, 100 * ratio_areas_nyd[:, j], linestyle=styles[1], color = colors[1], linewidth = widths[j], label='With compass')
                else:
                    plt.plot(noise_ratios, 100 * ratio_areas_nyd[:, j], linestyle=styles[1], color = colors[1], linewidth = widths[j])

                plt.plot(noise_ratios, 100 * ratio_areas_nyd[:, j], linestyle=styles[1], color = colors[1], linewidth = widths[j])
                plt.text(noise_ratios[-1], 100 * ratio_areas_nyd[-1, j], f'{percentages[j]}%', fontsize=8, ha='right', va='center', color=colors[1])

        # plot a marker at the noise_ratio of 1, for yd:
        plt.plot(1.0, 100 * ratio_areas_yd[noise_ratios.index(1.0), index_99], marker='o', color=colors[0])

        # plot a marker for SVO GTSAM parameters:
        # "distance_noise_std_per_meter": 0.015,
        # "yaw_rate_noise_std_per_meter": 0.25,
        distance_noise_std_per_meter = config["distance_noise_std_per_meter"]
        ratio_SVO  = (0.015 / distance_noise_std_per_meter)
        plt.plot(ratio_SVO, 0.7394, marker = 'X', color=colors[0], markersize=10)

        # plot a maker for Stankiewicz & Webb parameters:
        # "distance_noise_std_per_meter": 0.15
        # "yaw_rate_noise_std_per_meter": 5.5
        ratio_Webb  = (0.15 / distance_noise_std_per_meter)
        # make it a star marker:
        plt.plot(ratio_Webb, 0.24, marker = '*', color=colors[1], markersize=10)

        # replace the ticks with the noise variances used:
        plt.xticks(noise_ratios, [f'{distance_noise_std_per_meter*nr:.2f} m/m' for nr in noise_ratios])

        plt.xlabel('Distance noise, $\sigma_{d}$')
        plt.ylabel('Percentage covered')
        plt.legend()
        _save_or_show(exp_output_dir, 'area_coverage_vs_noise_ratio')

        #  Plot the percentage of the radius covered by the LHA as a function of the noise ratio:
        plt.figure(figsize=(10, 5))
        # plot the radii for both with and without yaw drift:
        for j in range(n_percentages):
            if j == 1:
                # add a label for the legend:
                plt.plot(noise_ratios, 100 * ratio_radii_yd[:, j], linestyle=styles[0], color = colors[0], linewidth = widths[j], label='With yaw drift')
            else:
                plt.plot(noise_ratios, 100 * ratio_radii_yd[:, j], linestyle=styles[0], color = colors[0], linewidth = widths[j])
            plt.text(noise_ratios[-1], 100 * ratio_radii_yd[-1, j], f'{percentages[j]}%', fontsize=8, ha='right', va='center', color=colors[0])
            if j == 1:
                plt.plot(noise_ratios, 100 * ratio_radii_nyd[:, j], linestyle=styles[1], color = colors[1], linewidth = widths[j], label='Without yaw drift')
            else:
                plt.plot(noise_ratios, 100 * ratio_radii_nyd[:, j], linestyle=styles[1], color = colors[1], linewidth = widths[j])
            plt.text(noise_ratios[-1], 100 * ratio_radii_nyd[-1, j], f'{percentages[j]}%', fontsize=8, ha='right', va='center', color=colors[1])
        # plot a marker at the noise_ratio of 1, for yd:
        plt.plot(1.0, 100 * ratio_radii_yd[noise_ratios.index(1.0), 1], marker='o', color=colors[0])
        # plot a marker for SVO GTSAM parameters:
        plt.plot(ratio_SVO, 8.825, marker = 'X', color=colors[0], markersize=10)
        # plot a marker for Stankiewicz & Webb parameters:
        plt.plot(ratio_Webb, 5.199, marker = '*', color=colors[1], markersize=10)

        # replace the ticks with the noise variances used:
        plt.xticks(noise_ratios, [f'{distance_noise_std_per_meter*nr:.3f} m/m' for nr in noise_ratios])
        plt.xlabel('Noise ratio')
        plt.ylabel('Percentage of outbound radius covered by LHA')
        plt.legend()
        _save_or_show(exp_output_dir, 'radius_coverage_vs_noise_ratio')

        # Show them both in the same figure:
        plt.figure(figsize=(10, 5))
        color_area = (0, 0, 0)  # black # (230/255.0, 97/255.0, 0.0)
        color_radius = (0, 0, 0)  # black # (93/255.0, 58/255.0, 1.0)
        plt.plot(noise_ratios, 100 * ratio_areas_yd, label='Area', linestyle='--', color=color_area)
        plt.plot(noise_ratios, 100 * ratio_radii_yd, label='Radius', color=color_radius)
        plt.xlabel('Noise ratio')
        plt.ylabel('Percentage covered')
        plt.legend()
        _save_or_show(exp_output_dir, 'area_radius_combined')

        # Show the percentage of inbound length divided by outbound length as a function of the noise ratio:
        plt.figure(figsize=(10, 5))
        plt.plot(noise_ratios, 100 * ratio_inbound_outbound_yd, color='green')
        plt.grid()
        plt.xlabel('Noise ratio')
        plt.ylabel('Percentage of inbound length divided by outbound length')
        # set the y axis limits to start at 0 and end at 100%:
        plt.ylim(0, 100)
        plt.title('Percentage of inbound length divided by outbound length as a function of the noise ratio')
        _save_or_show(exp_output_dir, 'inbound_outbound_ratio')

        # Save noise ratio statistics to CSV
        if exp_output_dir is not None:
            csv_path = os.path.join(exp_output_dir, 'noise_ratio_statistics.csv')
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ['noise_ratio']
                for perc in percentages:
                    header.append(f'area_ratio_{perc}pct_no_compass')
                    header.append(f'radius_ratio_{perc}pct_no_compass')
                for perc in percentages:
                    header.append(f'area_ratio_{perc}pct_with_compass')
                    header.append(f'radius_ratio_{perc}pct_with_compass')
                header.append('inbound_outbound_ratio_no_compass')
                header.append('inbound_outbound_ratio_with_compass')
                writer.writerow(header)
                for i, nr in enumerate(noise_ratios):
                    row = [nr]
                    for j in range(n_percentages):
                        row.append(f'{ratio_areas_yd[i, j]:.6f}')
                        row.append(f'{ratio_radii_yd[i, j]:.6f}')
                    for j in range(n_percentages):
                        row.append(f'{ratio_areas_nyd[i, j]:.6f}')
                        row.append(f'{ratio_radii_nyd[i, j]:.6f}')
                    row.append(f'{ratio_inbound_outbound_yd[i]:.6f}')
                    row.append(f'{ratio_inbound_outbound_nyd[i]:.6f}')
                    writer.writerow(row)

            with open(os.path.join(exp_output_dir, 'config.json'), 'w') as f:
                json.dump(config, f, indent=4)

        print('Done')

if __name__ == "__main__":
    main()
    