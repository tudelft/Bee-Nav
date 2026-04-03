# show results of virtual homing experiments
import json
import os
import virtual_homing_ANN
from matplotlib import pyplot as plt
import numpy as np
import insect_utils.plot_utils

results_file = 'results40_trees_snapshot.json'

show_aggregate_results = False
show_runs_specific_experiment = True

# The number of tree locations will be used to retrieve the right map name, etc.
num_locations = 40
exp = 0

# Whether to show success rates as a percentage:
percentage = True
# Whether to show histograms with frequency or as a probability distribution:
probability = False

# import the results.json file:
def load_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# load the results file:
results = load_json_file(results_file)
successful = results['successful']
total_distances = results['total_distances']
straight_line = results['straight_line']
radii = results['radii']
rates = results['rates']
if "success_per_run" in results:
    success_per_run = results["success_per_run"]
    process_success = True
else:
    process_success = False

if show_aggregate_results:
    # Result 1: Success rates and radii
    config_virt_homing = load_json_file('config/config_virt_homing.json')
    base_radius = config_virt_homing['circle_params']['radius']
    # Make a figure to plot the success rates and radii
    success_rates = np.asarray(rates)
    initial_radii = np.asarray(radii)
    # determine the mean:
    mean_success_rate = np.mean(success_rates, axis=0)
    median_success_rate = np.median(success_rates, axis=0)
    mean_radius = np.mean(initial_radii, axis=0)
    lower_bound = np.percentile(success_rates, 25, axis=0)
    upper_bound = np.percentile(success_rates, 75, axis=0)

    if percentage:
        factor = 100
    else:
        factor = 1
    

    plt.figure()
    # plot the 25th and 75th percentiles as a shaded area:
    # use light red for the fill color:
    color = 'lightcoral'
    plt.fill_between(mean_radius, factor * lower_bound, factor * upper_bound, color=color, alpha=0.5)
    # plot all individual lines for rates and radii in light gray:
    for i in range(len(success_rates)):
        plt.plot(initial_radii[i], factor * success_rates[i], color='lightgray')
    # plot the mean line:
    # use dark red:
    col = 'darkred'
    plt.plot(mean_radius, factor * median_success_rate, color=col)
    # Indicate where the learning area ends:
    plt.axvline(x=1.0, color='black', linestyle='--')
    # set the y limits:
    plt.ylim(0, factor)
    plt.xlabel('Radius factor')
    if percentage:
        plt.ylabel('Success rate [%]')
    else:
        plt.ylabel('Success rate')
    plt.show()

    # Histogram of total homing distances vs. straight line distances
    total_distances = np.asarray(total_distances)
    straight_line = np.asarray(straight_line)
    # Select those runs with <= 1.0 radius factor:
    n_radii_inside = len(np.where(mean_radius <= 1.0)[0])
    n_radii = len(mean_radius)
    n_inds = int(n_radii_inside * (total_distances.shape[1] / n_radii))
    total_distances = total_distances[:n_inds]
    straight_line = straight_line[:n_inds]
    
    if process_success:
        tot_dist = []
        str_line = []
        for j in range(len(success_per_run)):
            for k in range(n_inds):
                if success_per_run[j][k] == 1:
                    tot_dist.append(total_distances[j][k])
                    str_line.append(straight_line[j][k])
        total_distances = np.asarray(tot_dist)
        straight_line = np.asarray(str_line)

    # flatten the arrays:
    total_distances = total_distances.flatten()
    straight_line = straight_line.flatten()
    total_runs = len(total_distances)
    # divide the arrays by each other:
    travel_ratios = total_distances / straight_line

    # plot the histogram:
    plt.figure()
    # plot the histogram of the travel ratios, with as face color (230, 159, 0)
    face_color = (230, 159, 0)
    face_color = [x / 255 for x in face_color]
    plt.hist(travel_ratios, bins=50, facecolor=face_color, edgecolor='black', density=probability)
    plt.xlabel('Total distance / straight line distance')
    if probability:
        plt.ylabel('Probability density')
    else:
        plt.ylabel('Frequency')
    plt.show()

if show_runs_specific_experiment:
    with open("config/server.json", 'r') as file:
        options = json.load(file)
    server = options['server']

    from omni.isaac.kit import SimulationApp

    # Create the simulation app 
    simulation_app = SimulationApp({"headless": server})
    my_world = None

    # Determine the map path that should be loaded for the experiment:
    config_generate_file = "config/config_forest_environment.json"
    with open(config_generate_file, 'r') as file:
        config_generate = json.load(file)
    area_bounds = config_generate['area_bounds'] # area bounds as in [-area_bounds, area_bounds]
    map_name = "forest_" + str(num_locations) + "_trees_" + str(area_bounds) + "x" + str(area_bounds) +"_area_" + str(exp)
    map_path = os.getcwd() + "/maps/forest_" + str(num_locations) + "_trees_" + str(area_bounds) + "x" + str(area_bounds) +"_area_" + str(exp) + '.usd'

    import virtual_homing_snapshot
    filename_grid = map_name + "_grid.json"
    # Check if the filename_grid already exists
    if os.path.exists(filename_grid):
        # load the predictions and positions from the file:
        with open(filename_grid, 'r') as f:
            data = json.load(f)
        predictions = np.asarray(data['predictions'])
        positions = np.asarray(data['positions'])
    else:
        # 1. Get a grid of arrows and positions:
        predictions, positions, my_world = virtual_homing_snapshot.catchment_area_grid(verbose = True, graphics = False, save_images = False, \
                                                                         simulation_app = simulation_app, usd_path = map_path)
        # save the predictions and positions to a file:
        with open(filename_grid, 'w') as f:
            json.dump({'predictions': predictions.tolist(), 'positions': positions.tolist()}, f)

    #  2. Perform virtual snapshot homing:
    successful_runs, total_distances_traveled, straight_line_distances, radius_scales, success_rates, trajectories, successes = \
        virtual_homing_snapshot.snapshot_navigation(verbose = True, graphics = False, save_images = False, simulation_app = simulation_app, \
                                                    my_world = my_world, usd_path = map_path)
    
    # 3. Plot the results:
    landmark_filename = insect_utils.plot_utils.get_landmark_filename(os.getcwd() + "/maps/forest_" + str(num_locations) + \
                                        "_trees_" + str(area_bounds) + "x" + str(area_bounds) +"_area_" + str(exp))
    insect_utils.plot_utils.make_nice_plot(predictions, positions, \
                                            homing_positions=trajectories, successes= successes,
                                            landmark_filename = landmark_filename, append_name = 'snapshot', show_plot=False)

    simulation_app.close()

print('Done')
